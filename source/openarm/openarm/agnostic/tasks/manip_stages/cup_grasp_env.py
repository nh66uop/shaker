"""CupGraspEnv — `CupPickEnv` + 손가락 **첫마디** 접촉·표면 입력 + critic 접촉 관측 (09.27 사용자 결정 "1,3을 같이").

왜: 두 팔 모두 무작위 초기에서 컵 집기 전 과정을 배웠지만(cup_grasp iter_00·01), 영상에서 4번 관절이 과하게 말린 손끝 파지였다.
m18 재생에서 네 손가락이 쥔 스텝의 가운데 마디 접촉률은 0.00, 손끝 패드 0.90~1.00. 첫마디(`_2`)는 센서도 ctx 점도 없어
뿌리부터 감싸는 파지를 보상이 볼 수 없었고, actor·critic 모두 접촉을 몰랐다.

더하는 것 셋(과제·actor 관측·액션·종료·리셋·성공은 그대로 → s2r 배포 계약(actor obs 133·action 26) 불변):
  1. 보상 입력: `GraspRewardContext` 의 `prox_*` 필드(첫마디 위치·컵 접촉력·표면 거리/방향/법선).
  2. critic 전용 관측: 링크별 컵 접촉력 21 = 첫마디 5 + 가운데·끝마디·손끝 15 + 손바닥 1, tanh(F / CRITIC_FORCE_REF_N).
     critic 은 배포되지 않는다(비대칭 actor-critic) — 실기에서 볼 수 없는 값을 넣어도 s2r 에 영향이 없다.
  3. 로그: `contact/prox_touching`, `contact/prox_<finger>`, `task/prox_surface_dist_min_mean`.
★cup_pick gym id·로그·체크포인트는 그대로다 — critic 차원이 달라 이 env 는 새 gym id(`cup_grasp`) 로만 쓴다.
"""

from __future__ import annotations

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from ..grasp_fj.robot_profiles import PROFILES
from .cup_pick_env import CupPickEnv
from .hand_action import zero_open_raw
from .real_delays import LagQueue
from .grasp_context import CRITIC_CONTACT_DIM, CRITIC_FORCE_REF_N, PROXIMAL_SUFFIX, GraspRewardContext, from_object
from .surface_query import nearest_surface

#: 로그에서 "닿았다"로 셀 힘 [N] — 부모 로그와 같은 값.
_TOUCH_LOG_N = 0.1


class CupGraspEnv(CupPickEnv):

    def __init__(self, cfg, render_mode: str | None = None, **kw):
        super().__init__(cfg, render_mode, **kw)
        self._install_real_delays()

    def _install_real_delays(self) -> None:
        """09.28 실기 지연: cfg 범위가 (0, 0) 이 아니면 부모 지연 큐 객체를 `LagQueue` 로 바꿔 끼운다(인터페이스 동일)."""
        c, n, dev = self.cfg, self.num_envs, self.device
        arm = tuple(int(v) for v in getattr(c, "real_arm_lag_steps", (0, 0)))
        hand = tuple(int(v) for v in getattr(c, "real_hand_lag_steps", (0, 0)))
        obj = tuple(int(v) for v in getattr(c, "real_obj_lag_steps", (0, 0)))
        drop_p = float(getattr(c, "real_obj_dropout_p", 0.0))
        if arm != (0, 0) or hand != (0, 0):
            off, adim = int(self._hand_action_offset), int(self._act_delay.dim)   # 액션 폭은 부모 큐에서 읽는다(차원 불변)
            self._act_delay = LagQueue(n, adim, [(0, off, *arm), (off, adim, *hand)], dev)
        if obj != (0, 0) or drop_p > 0.0:
            self._obj_delay = LagQueue(n, self._obj_delay.dim, [(0, self._obj_delay.dim, *obj)], dev, dropout_p=drop_p,
                                       dropout_steps=tuple(int(v) for v in getattr(c, "real_obj_dropout_steps", (0, 0))))
        print(f"[cup_grasp] 실기 지연 팔 {arm} · 손 {hand} · 물체 {obj} 스텝(16.7 ms) · 끊김 p {drop_p} · "
              f"목표 고정 유지 {int(getattr(c, 'goal_hold_after', 0))} 성공 뒤", flush=True)

    def _advance_goals(self, is_success: torch.Tensor) -> None:
        """`cfg.goal_hold_after` > 0 이면 그만큼 성공한 뒤로는 목표를 옮기지 않는다(09.28 실기 인계: 긴 유지 구간). 아니면 부모 그대로."""
        k = int(getattr(self.cfg, "goal_hold_after", 0))
        if k <= 0:
            return super()._advance_goals(is_success)
        before = self._trk.successes.clone()
        pos, quat = self.goal_pos.clone(), self.goal_quat.clone()
        super()._advance_goals(is_success)
        freeze = (is_success & (before + 1 >= k)).unsqueeze(1)
        self.goal_pos = torch.where(freeze, pos, self.goal_pos)
        self.goal_quat = torch.where(freeze, quat, self.goal_quat)

    def _hand_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """`cfg.hand_action_zero_open` 이면 액션 0 = 리셋(편 손) 자세(09.28). 아니면 부모 그대로.
        EMA·clamp·`_syn_close` 는 부모(`grasp_fj_env._hand_targets`)와 같은 식이다."""
        if not (bool(self.cfg.hand_direct) and bool(getattr(self.cfg, "hand_action_zero_open", False))):
            return super()._hand_targets(a_hand)
        lo, hi = self._act_lo.unsqueeze(0), self._act_hi.unsqueeze(0)
        raw = zero_open_raw(a_hand, lo, hi, self._hand_reset_q.unsqueeze(0))
        alpha = float(self.cfg.hand_ema)
        tgt = (alpha * raw + (1.0 - alpha) * self._syn_target).clamp(lo, hi)
        self._syn_close = (tgt - lo) / self._act_span.unsqueeze(0)
        return tgt

    def _setup_scene(self) -> None:
        """부모 씬(보상 전용 마디·손바닥 센서 포함) + 첫마디 센서. body 하나당 센서 하나(부모 규약)."""
        super()._setup_scene()
        prof = PROFILES[self.cfg.profile_name]
        self._prox_bodies: dict[str, str] = {}
        self._prox_sensors: dict[str, ContactSensor] = {}
        for finger, bodies in prof.finger_sensor_bodies.items():
            body = bodies[0].rsplit("_", 1)[0] + PROXIMAL_SUFFIX          # `<side>_hl_<finger>_3` → `_2`
            s = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{body}",
                filter_prim_paths_expr=self._t2r_filter,
                history_length=1, track_air_time=False))
            self.scene.sensors[f"grasp_contact_{body}"] = s
            self._prox_bodies[finger] = body
            self._prox_sensors[finger] = s

    def _prox_body_ids(self) -> torch.Tensor:
        ids = getattr(self, "_prox_ids_t", None)
        if ids is None:
            idx = [self.robot.find_bodies(self._prox_bodies[f])[0][0] for f in self._finger_names]
            ids = torch.tensor(idx, device=self.device, dtype=torch.long)
            self._prox_ids_t = ids
            print(f"[cup_grasp] 첫마디 = {[self._prox_bodies[f] for f in self._finger_names]} · 컵 접촉 센서 5 · "
                  f"critic 접촉 관측 {CRITIC_CONTACT_DIM} (tanh(F/{CRITIC_FORCE_REF_N} N)) · actor 관측 불변", flush=True)
        return ids

    def _prox_cup_forces(self) -> torch.Tensor:
        """(N,5) 첫마디 컵 필터 접촉력 크기 [N] — 부모 `_link_cup_forces` 와 같은 계산. 손가락 순서 = `_finger_names`."""
        return torch.stack([self._prox_sensors[f].data.force_matrix_w.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)
                            for f in self._finger_names], dim=1)

    # ------------------------------------------------------------------
    def _build_context(self, kw: dict, out: dict) -> GraspRewardContext:
        base = super()._build_context(kw, out)
        prox_pos = self.robot.data.body_pos_w[:, self._prox_body_ids()] - self.scene.env_origins[:, None, :]   # (N,5,3)
        pts, nrm = self._surface_tensors()
        dist, direction, normal = nearest_surface(prox_pos, base.cup_pos, base.cup_quat, pts, nrm, self._species_ids)
        return from_object(base, prox_pos=prox_pos, prox_cup_force=self._prox_cup_forces(),
                           prox_surface_dist=dist, prox_surface_dir=direction, prox_surface_normal=normal)

    def _privileged_blocks(self, ob: dict) -> list[torch.Tensor]:
        """부모 critic 블록 + 링크별 컵 접촉력(센서 직접 읽기, 이번 스텝 물리 상태)."""
        links_f, palm_f = self._link_cup_forces()
        contact = torch.cat([self._prox_cup_forces(), links_f.reshape(self.num_envs, -1), palm_f[:, None]], dim=1)
        return super()._privileged_blocks(ob) + [torch.tanh(contact / CRITIC_FORCE_REF_N)]

    def _log_fabric_metrics(self) -> None:
        super()._log_fabric_metrics()
        ctx = self._t2r_ctx
        if not isinstance(ctx, GraspRewardContext):
            return
        ex = self.extras
        touch = (ctx.prox_cup_force > _TOUCH_LOG_N).float()                                           # (N,5)
        ex["contact/prox_touching"] = touch.sum(dim=1).mean()
        for i, f in enumerate(self._finger_names):
            ex[f"contact/prox_{f}"] = touch[:, i].mean()
        ex["task/prox_surface_dist_min_mean"] = ctx.prox_surface_dist.amin(dim=1).mean()
        # 마디 기준 접촉(보상과 같은 정의의 진단): 손끝 패드 말고 첫마디·가운데·끝마디 중 하나라도 닿은 손가락 수
        seg = torch.cat([ctx.prox_cup_force[..., None], ctx.link_cup_force[..., :2]], dim=-1) > _TOUCH_LOG_N
        ex["contact/phalanx_digits"] = seg.any(dim=-1).float().sum(dim=1).mean()
        self._log_joint_flexion(ctx, ex)

    def _log_joint_flexion(self, ctx: GraspRewardContext, ex: dict) -> None:
        """09.28 사용자: 관절별 평균 굽힘을 국면별로 로그 — 접근 중 손이 뭉치는지, 컵(종)마다 비율만 다른 같은 파지인지 보려고.

        굽힘 = (q − 리셋 자세) / (상한 − 리셋 자세), 0 = 편 손 · 1 = 관절 한계(보상의 rel_flex 와 같은 정의). 가동 관절만.
        국면: approach = 손바닥 표면 5 cm 밖·손 어디도 컵에 안 닿음·안 들림 / contact = 손바닥 5 cm 안이거나 닿음·안 들림 /
        lifted = 들림 래치. 해당 env 가 없는 스텝은 키를 넣지 않는다. 컵 종별은 contact·lifted 국면에서 관절별로 따로 남긴다.
        """
        names = getattr(self, "_flex_log_names", None)
        if names is None:
            jn = self.robot.joint_names
            raw = [jn[i] for i in self._syn_ids]
            mov = (self._act_span > 0.05).tolist()
            self._flex_log_cols = [k for k, m in enumerate(mov) if m]
            self._flex_log_names = names = [raw[k].split("_", 1)[1] if raw[k][:2] in ("r_", "l_") else raw[k]
                                            for k in self._flex_log_cols]
        cols = self._flex_log_cols
        head = (1.0 - ctx.hand_default_q_norm[:, cols]).clamp(min=0.1)
        flex = ((ctx.hand_q_norm[:, cols] - ctx.hand_default_q_norm[:, cols]) / head).clamp(0.0, 1.0)   # (N,M)
        touching = ((ctx.link_cup_force > _TOUCH_LOG_N).any(dim=-1).any(dim=-1)
                    | (ctx.prox_cup_force > _TOUCH_LOG_N).any(dim=-1) | (ctx.palm_cup_force > _TOUCH_LOG_N))
        lifted = ctx.lifted.bool()
        far = ctx.palm_surface_dist > 0.05
        phases = {"approach": far & ~touching & ~lifted, "contact": (~far | touching) & ~lifted, "lifted": lifted}
        sp = self._species_ids
        # 빈 국면은 키를 아예 넣지 않는다 — 센티널(−1)을 넣으면 epoch 평균에 섞여 값이 틀어진다(09.28 probe 에서 확인).
        #   env 수 동기화는 스텝당 국면·종 조합 한 번씩(텐서 하나로 모아 .tolist() 1회).
        masks = {ph: m for ph, m in phases.items()}
        for s, sname in enumerate(self._species_names):
            for ph in ("contact", "lifted"):
                masks[f"{sname}|{ph}"] = phases[ph] & (sp == s)
        keys = list(masks)
        W = torch.stack([masks[k] for k in keys]).float()                      # (K,N)
        counts = W.sum(dim=1)
        means = (W @ flex) / counts.clamp(min=1.0).unsqueeze(1)                # (K,M)
        for ph, m in phases.items():
            ex[f"hand_phase/{ph}_frac"] = m.float().mean()
        for k, n in zip(keys, counts.tolist()):
            if n <= 0:
                continue
            row = means[keys.index(k)]
            if "|" in k:
                sname, ph = k.split("|")
                for j, nm in enumerate(names):
                    ex[f"hand_flex_{sname}/{ph}/{nm}"] = row[j]
            else:
                ex[f"hand_flex/{k}/mean"] = row.mean()
                for j, nm in enumerate(names):
                    ex[f"hand_flex/{k}/{nm}"] = row[j]
