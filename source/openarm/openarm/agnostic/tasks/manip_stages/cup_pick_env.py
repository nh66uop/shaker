"""CupPickEnv — `GraspFJT2REnv`(s2r 동결) + 물체 모양과 무관한 보상 입력(손 → 실제 표면 거리). 09.22 사용자 결정.

★이 클래스가 하는 일은 **보상 입력 추가 + 외란 DR 게이트**뿐이다(계약 테스트가 잠근다): 관측·액션·종료·리셋·성공 판정은 부모 그대로.
  덮는 것은 넷(+ 도우미 `_surface_tensors`·`_update_wrench_adr`) — `__init__`(종별 표면 점군 · 외란 ADR 상태) ·
  `_build_context`(부모 ctx + 표면 필드) · `_apply_wrench`(목표 게이트 외란, 09.23) · `_log_fabric_metrics`(task/·dr/ 로그).
★점군 = 뱅크 종마다 충돌 형상에서 K 점(`object_surface`) — env 배정은 부모의 `_species_ids`(env_id % N, MultiAssetSpawner 와
  같은 규칙). 프레임 = 강체 actor 프레임 = `ctx.cup_pos`/`cup_quat` 그대로(오프셋 없음, `object_surface` 머리말).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..grasp_fj_t2r.grasp_fj_t2r_env import GraspFJT2REnv
from .goal_wrench import (ADR_EPISODES_PER_UPDATE, ADR_LOWER_DWELL, ADR_RAISE_DWELL, NEAR_GOAL_RADIUS_M, NO_EVENT,
                          episode_ema, near_goal_mask, next_adr_level, wrench_gate, wrench_scales)
from .object_context import ObjectRewardContext, from_base
from .object_surface import DEFAULT_K, DEFAULT_SEED, species_cloud
from .surface_query import nearest_surface

#: 진단 로그에서 "닿았다"로 셀 컵 접촉력 [N] — 부모 `_TOUCH_LOG_N` 과 같은 값.
_TOUCH_LOG_N = 0.1
#: "닿았는데 표면에서 멀다"로 셀 마디 원점↔표면 거리 [m] — 표면 거리 진단용.
_TOUCH_FAR_M = 0.02


class CupPickEnv(GraspFJT2REnv):

    def __init__(self, cfg, render_mode: str | None = None, **kw):
        # 점군은 부팅 **전에** 만든다(numpy) — 부모 부팅 중 스텝이 돌아도 `_build_context` 가 쓸 수 있게. 장치 이동은 첫 사용 때.
        from ...modules import object_bank as _ob

        t0 = time.time()
        bank = _ob.get(cfg.object_bank)
        remap = getattr(cfg, "_remap_object_usd", lambda p: p)
        clouds = [species_cloud(remap(s.usd_path), s.scale, rigid_body_name=s.rigid_body_name,
                                k=DEFAULT_K, seed=DEFAULT_SEED) for s in bank.specs]
        self._surf_ids = [s.id for s in bank.specs]
        self._surf_np = (np.stack([c.points for c in clouds]), np.stack([c.normals for c in clouds]))    # (S,K,3) ×2
        self._surf_t = None
        super().__init__(cfg, render_mode, **kw)
        if self._surf_ids != list(self._species_names):
            raise RuntimeError(f"[cup_pick] 뱅크 순서 {self._surf_ids} ≠ env 종 {self._species_names}")
        print(f"[cup_pick] 표면 점군 {len(clouds)}종 × {DEFAULT_K}점 (충돌 형상, 강체 프레임) · 간격 "
              f"{min(c.spacing for c in clouds) * 1e3:.1f}–{max(c.spacing for c in clouds) * 1e3:.1f} mm · "
              f"{time.time() - t0:.1f} s · 관측 불변", flush=True)
        # ---- 목표 게이트 외란 ADR 상태(09.23) — 전부 **텐서**로 굴린다(레벨을 host 로 내리면 매 스텝 동기화가 걸린다).
        self._wrench_adr_level = torch.zeros((), device=self.device)          # 0 = W1, 1 = W2
        self._wrench_adr_ep_count = torch.zeros((), device=self.device)       # 갱신까지 모은 완료 에피소드 수
        self._wrench_dwell_ema = torch.full((), NO_EVENT, device=self.device)  # 완료 에피소드의 목표 체류율 EMA
        self._wrench_dwell_steps = torch.zeros(self.num_envs, device=self.device)   # 에피소드 내 목표 체류 스텝
        _f0, _t0s = wrench_scales(0.0)
        _f1, _t1s = wrench_scales(1.0)
        # ★조용한 no-op 방지 — 부모는 cfg 값 0.0 을 읽고 "외란 DR OFF" 를 찍는다. 여기서 실효값을 덮어 쓴다는 것을 남긴다.
        print(f"[cup_pick] 목표 게이트 외란 ON (부모 cfg 0.0 을 런타임 덮어씀) — 게이트 = 높이 래치 ∧ "
              f"goal_dist ≤ {NEAR_GOAL_RADIUS_M} m · ADR W1 {_f0}/{_t0s} → W2 {_f1}/{_t1s} N/kg·N·m/kg · "
              f"승급 체류율 > {ADR_RAISE_DWELL} · 강등 < {ADR_LOWER_DWELL} · {ADR_EPISODES_PER_UPDATE} 에피소드마다 "
              f"±{float(_f1 - _f0) * 0.05:.2f} N/kg · TB `dr/wrench_*`", flush=True)

    def _surface_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._surf_t is None:
            self._surf_t = tuple(torch.from_numpy(a).to(self.device, torch.float32) for a in self._surf_np)
        return self._surf_t

    # ------------------------------------------------------------------
    def _build_context(self, kw: dict, out: dict) -> ObjectRewardContext:
        base = super()._build_context(kw, out)
        n = base.num_envs
        query = torch.cat([base.palm_pos[:, None, :], base.link_pos.reshape(n, -1, 3)], dim=1)       # (N,16,3)
        pts, nrm = self._surface_tensors()
        dist, direction, normal = nearest_surface(query, base.cup_pos, base.cup_quat, pts, nrm, self._species_ids)
        shp = base.link_pos.shape[:-1]                                                                  # (N,5,3)
        return from_base(base,
                         palm_surface_dist=dist[:, 0], palm_surface_dir=direction[:, 0],
                         palm_surface_normal=normal[:, 0],
                         link_surface_dist=dist[:, 1:].reshape(shp),
                         link_surface_dir=direction[:, 1:].reshape(*shp, 3),
                         link_surface_normal=normal[:, 1:].reshape(*shp, 3))

    # ------------------------------------------------------------------
    def _apply_wrench(self) -> None:
        """외란을 **높이 래치 ∧ 목표 근방**에서만 터뜨리고, 크기를 ADR 레벨로 보간한다(09.23 사용자 결정).

        왜: `cp_l_m08` 정책은 들고 나르지만 목표에 안착하지 않고 맴돈다. 목표에 외란이 있어야 "머무르기"가
        견고한 파지를 요구하는 기술이 된다. 반경 선택 이유·회피 수입 비교·ADR 상수 근거는 전부 `goal_wrench` 머리말.

        ★1스텝 지연 규약: `_t2r_ctx` 는 직전 스텝 `_get_rewards` 가 채운 값이다.
        ★부팅 중 스텝(ADR 상태·ctx 가 아직 없다)은 부모로 넘긴다 — 부모 cfg 크기가 0.0 이라 실질 no-op.
        ★★09.25 버그 수정: 게이트는 `ctx.lifted` 로 잡는다(`_wrench_gate_now`). 전에는 부모의 `self._lifted` 를 썼는데,
          그 값은 `fj_core_env._get_rewards` 에서만 갱신되고 이 계통은 `fj_kp_env._get_rewards` 가 그 함수를 덮어
          높이 래치를 `self._latched` 에 둔다 → `self._lifted` 는 초기값 0 에 고정. 그래서 iter_09~14 내내 ADR 레벨은
          1.0 까지 올랐는데(`ctx.lifted` 로 잰 체류) 실제 힘은 한 번도 안 들어갔다(`dr/wrench_f_norm_max` 0.000).
          발화 게이트와 로그·ADR 게이트가 같은 함수를 쓰게 해 다시 갈라지지 않게 한다.
        """
        ctx = getattr(self, "_t2r_ctx", None)
        level = getattr(self, "_wrench_adr_level", None)
        if ctx is None or level is None:
            super()._apply_wrench()
            return
        f_scale, t_scale = wrench_scales(level)
        self._wrench.force_scale, self._wrench.torque_scale = f_scale, t_scale
        forces, torques = self._wrench.step(self._obj_mass, self._wrench_gate_now(ctx))
        self.object.set_external_force_and_torque(forces, torques, is_global=True)
        _fn = forces.view(self.num_envs, -1).norm(dim=-1)
        ex = self.extras
        ex["dr/wrench_fire_frac"] = (_fn > 0.0).float().mean()      # 부모와 같은 이름 — 실제 발화 확인
        ex["dr/wrench_f_norm_max"] = _fn.max()
        ex["dr/wrench_adr_level"] = level
        ex["dr/wrench_force_scale"] = f_scale

    @staticmethod
    def _wrench_gate_now(ctx: ObjectRewardContext) -> torch.Tensor:
        """외란 게이트의 단일 출처 — 발화(`_apply_wrench`)와 ADR·로그(`_update_wrench_adr`)가 같은 값을 쓴다."""
        return wrench_gate(ctx.lifted, ctx.goal_dist)

    def _log_fabric_metrics(self) -> None:
        super()._log_fabric_metrics()
        ctx = self._t2r_ctx
        if not isinstance(ctx, ObjectRewardContext):
            return
        ex = self.extras
        ex["task/palm_surface_dist"] = ctx.palm_surface_dist.mean()
        ex["task/link_surface_dist_min_mean"] = ctx.link_surface_dist.amin(dim=(1, 2)).mean()
        # 표면 거리 ↔ 접촉 센서 정합(host 동기화 0): 컵에 닿은 마디의 평균 표면 거리 · 그중 _TOUCH_FAR_M 넘는 비율
        touch = (ctx.link_cup_force > _TOUCH_LOG_N).to(ctx.link_surface_dist.dtype)
        cnt = touch.sum().clamp(min=1.0)
        ex["task/link_surface_dist_touching"] = (ctx.link_surface_dist * touch).sum() / cnt
        ex["task/touching_link_far_frac"] = ((ctx.link_surface_dist > _TOUCH_FAR_M).to(touch.dtype) * touch).sum() / cnt
        ex["task/touching_link_count"] = touch.sum()
        self._update_wrench_adr(ctx, ex)

    def _update_wrench_adr(self, ctx: ObjectRewardContext, ex: dict) -> None:
        """목표 체류율을 완료 에피소드 EMA 로 모아 외란 ADR 레벨을 느리게 올리고 내린다(host 동기화 0).

        여기서 도는 이유: `_get_dones` → `_get_rewards`(→ 이 로그) → `_reset_idx` 순이라, 이 시점의
        `reset_buf` 가 **이번 스텝에 끝나는 에피소드**이고 `episode_length_buf` 는 아직 그 길이다.
        부모 `_reset_idx` 는 계약상 덮지 않는다(동결 leaf 의 리셋 규약).
        """
        level = getattr(self, "_wrench_adr_level", None)
        if level is None:
            return
        gate = self._wrench_gate_now(ctx)
        self._wrench_dwell_steps = self._wrench_dwell_steps + gate.float()
        ended = self.reset_buf & (self.episode_length_buf > 0)       # 길이 0 = 첫 reset(), 에피소드 끝이 아니다
        dwell_ep = self._wrench_dwell_steps / self.episode_length_buf.clamp(min=1).float()
        self._wrench_dwell_ema = episode_ema(self._wrench_dwell_ema, dwell_ep, ended)
        self._wrench_adr_ep_count = self._wrench_adr_ep_count + ended.sum()
        _due = self._wrench_adr_ep_count >= float(ADR_EPISODES_PER_UPDATE)
        self._wrench_adr_level = torch.where(_due, next_adr_level(level, self._wrench_dwell_ema), level)
        self._wrench_adr_ep_count = torch.where(_due, torch.zeros_like(self._wrench_adr_ep_count),
                                                self._wrench_adr_ep_count)
        self._wrench_dwell_steps = torch.where(ended, torch.zeros_like(self._wrench_dwell_steps),
                                               self._wrench_dwell_steps)
        ex["task/wrench_near_goal_frac"] = near_goal_mask(ctx.goal_dist).float().mean()
        ex["dr/wrench_gate_frac"] = gate.float().mean()              # 발화 **가능** 비율(실제 발화는 dr/wrench_fire_frac)
        ex["task/wrench_goal_dwell_ep"] = self._wrench_dwell_ema
