"""SideRig — 한쪽 팔의 배선 묶음: Fabrics 팔 + 관절공간 시너지 손 + 접촉 + palm 지령 (mimic 판).

★`pour_fabric/side_rig.py` 사본. 저차원(언더액추) 손·양팔 26 DOF fabric 을 위해 바뀐 곳:
  1. Fabrics: 클래스 생성자가 tesollo 와 다르다(robot_dir·params·tip_per_finger 인자 없음, side 고정).
     fabric cspace 는 **양팔 26** 이고 이 팔은 `bimanual.fabric_slots` 슬라이스만 읽고 쓴다.
     반대팔 슬라이스는 매 스텝 상대 rig 의 상태로 덮는다(`sync_other`) — 안 맞추면 body 반발이
     존재하지 않는 반대팔 자세를 피하려 팔을 민다.
  2. 손 레이아웃 `per_finger`: 프로필 `hand_finger_channels` 가 **구동관절 ↔ 액션 슬롯 1:1** 을
     준다(RH56F1: 채널 접미사가 {1,2} 뿐이라 채널×손가락 격자는 슬롯 4개가 아무 관절에도 안
     걸린다 — grasp_fj_rh 09.07 실측). 4지 결합(`couple_four_fingers`)은 같은 접미사끼리 평균.
  3. 접촉 동결: 접미사 "3" 리터럴 대신 프로필 `hand_freeze_suffixes` 로 굴곡 집합을 정한다
     (RH 4지 구동은 "_1", 종속 "_2" 마디의 접촉이 그 손가락의 구동관절을 멈춘다 = finger 스코프).

grasp_s2r 의 `GraspS2RControlMixin`(한 팔 전용, self 에 묶임)을 **인스턴스 단위**로 옮겼다 —
양팔이 각자 하나씩 갖는다. 로봇 종속 정보는 전부 프로필에서 온다(이름 리터럴 금지 계약).

포팅 출처(수식·규약 동일, 09.13):
  · 시너지 `_setup_synergy` / `_synergy_targets`(contact 모드) / `_close_progress`
    / `_apply_pose_knobs`(oppose delta) — grasp_s2r_control.py
  · fabric 설정·적분·FK 부팅 게이트 — grasp_s2r_control.py `_setup_fabrics`/`_init_home_palm`
  · palm 액션 = 앵커 + 델타 — grasp_s2r_env.py `_pre_physics_step`
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat

_DEG = math.pi / 180.0


# ---------------------------------------------------------------- ★09.28 증분 palm 지령 (DG-5F pour_rules 이식)
def palm_vmax_vec(vel_pos_m_s: float, vel_rot_deg_s: float) -> torch.Tensor:
    """palm 6D 최대 속도 벡터 (m/s ×3, rad/s ×3)."""
    return torch.tensor([float(vel_pos_m_s)] * 3 + [math.radians(float(vel_rot_deg_s))] * 3)


def step_palm_increment(target: torch.Tensor, v_norm: torch.Tensor, vmax: torch.Tensor, dt: float,
                        anchor: torch.Tensor, delta_lo: torch.Tensor, delta_hi: torch.Tensor) -> torch.Tensor:
    """증분 지령 한 스텝: 목표 += v_norm·vmax·dt, 앵커 기준 델타 박스 [lo, hi] 로 자른다. 새 텐서.

    v_norm 은 [−1, 1] (거른 액션), vmax 는 (6,) 물리 최대 속도, 회전 델타는 wrap 뒤 자른다.
    """
    raw = target + v_norm.clamp(-1.0, 1.0) * vmax.to(target) * float(dt)
    d = raw - anchor
    d_rot = torch.remainder(d[:, 3:] + math.pi, 2.0 * math.pi) - math.pi
    d = torch.cat([d[:, :3], d_rot], dim=1)
    d = torch.maximum(torch.minimum(d, delta_hi.to(d)), delta_lo.to(d))
    return anchor + d


def leash_palm_target(target: torch.Tensor, pose: torch.Tensor, max_pos_m: float,
                      max_rot_deg: float) -> torch.Tensor:
    """palm 6D 목표를 현재 실제 palm 자세 근처로 묶는다(목표−실제 간극 상한). 새 텐서.

    위치는 간극 벡터 길이를 max_pos_m 로, 회전은 euler 슬롯별(wrap 후) 차이를 max_rot_deg 로 자른다.
    둘 중 하나가 0 이하면 그 부분은 끈다. target·pose 는 같은 프레임(N,6), 회전은 rad.
    """
    out = target.clone()
    if max_pos_m > 0.0:
        d = target[:, :3] - pose[:, :3]
        n = d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        out[:, :3] = pose[:, :3] + d * torch.clamp(float(max_pos_m) / n, max=1.0)
    if max_rot_deg > 0.0:
        r = torch.remainder(target[:, 3:] - pose[:, 3:] + math.pi, 2.0 * math.pi) - math.pi
        lim = math.radians(float(max_rot_deg))
        out[:, 3:] = pose[:, 3:] + r.clamp(-lim, lim)
    return out


class SideRig:
    def __init__(self, env, profile, *, role: str, sensors: dict, palm_sensor,
                 delta_lo: tuple, delta_hi: tuple, oppose_sign: float) -> None:
        self.profile = profile
        self.role = role                      # "src" | "rcv"
        self.env = env
        robot, device, N = env.robot, env.device, env.num_envs
        self.robot, self.device, self.N = robot, device, N
        cfg = env.cfg

        # ---- 관절/바디 인덱스 -----------------------------------------------------------
        self.arm_ids, _ = robot.find_joints(profile.arm_joint_regex)
        self.hand_ids, _ = robot.find_joints(profile.hand_joint_regex)
        if (len(self.arm_ids) != profile.num_arm_joints
                or len(self.hand_ids) != profile.num_hand_joints):
            raise RuntimeError(
                f"[{profile.name}] 프로필 조인트 수 불일치: arm {len(self.arm_ids)}"
                f"!={profile.num_arm_joints}, hand {len(self.hand_ids)}!={profile.num_hand_joints}")
        self.arm_t = torch.tensor(self.arm_ids, device=device, dtype=torch.long)
        self.hand_t = torch.tensor(self.hand_ids, device=device, dtype=torch.long)
        self.palm_idx = self._one_body(profile.palm_body)
        self.fingers = list(profile.finger_sensor_bodies.keys())
        self.tip_t = torch.tensor([self._one_body(b) for b in profile.fingertip_bodies],
                                  device=device, dtype=torch.long)
        self.grp_a = torch.tensor([self.fingers.index(f) for f in profile.contact_group_a],
                                  device=device, dtype=torch.long)
        self.grp_b = torch.tensor([self.fingers.index(f) for f in profile.contact_group_b],
                                  device=device, dtype=torch.long)
        self.sensors = sensors                # finger -> [ContactSensor] (프로필 body 순)
        self.palm_sensor = palm_sensor

        # ---- palm 지령 -------------------------------------------------------------------
        self.box_lo = torch.tensor(profile.palm_box_min, device=device)
        self.box_hi = torch.tensor(profile.palm_box_max, device=device)
        lo = [float(v) for v in delta_lo]
        hi = [float(v) for v in delta_hi]
        self.delta_lo = torch.tensor(lo[:3] + [v * _DEG for v in lo[3:]], device=device)
        self.delta_hi = torch.tensor(hi[:3] + [v * _DEG for v in hi[3:]], device=device)
        self.anchor = torch.zeros(N, 6, device=device)       # fabric 프레임
        self.anchor_env = torch.zeros(N, 6, device=device)   # env-local 실측(관측·로깅용)
        self.palm_targets = torch.zeros(N, 6, device=device)
        self.fab_to_env = torch.zeros(3, device=device)

        # ---- 시너지 손 --------------------------------------------------------------------
        self._setup_synergy(oppose_sign)

        # ---- 시작 자세(에피소드 리셋 관절값) --------------------------------------------------
        q_default = robot.data.default_joint_pos[0]
        self.reset_q = q_default.clone()
        if profile.arm_reset_joint_pos:
            if len(profile.arm_reset_joint_pos) != profile.num_arm_joints:
                raise RuntimeError(f"[{profile.name}] arm_reset_joint_pos 길이 오류")
            self.reset_q[self.arm_t] = torch.tensor(profile.arm_reset_joint_pos, device=device)
        self.reset_q[torch.tensor(self.syn_ids, device=device)] = self.syn_open

        self.fabric = None
        self.integrator = None
        self.fab_t = None
        self.joint_arm = str(cfg.arm_control) == "joint"
        self.arm_q_target = self.reset_q[self.arm_t].unsqueeze(0).repeat(N, 1)
        self.arm_lo = self.arm_hi = None

    # ==================================================================
    def _one_body(self, name: str) -> int:
        ids, _ = self.robot.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"[{self.profile.name}] body '{name}' 해석 실패: {ids}")
        return ids[0]

    # ==================================================================
    # 시너지 손 (grasp_s2r `_setup_synergy` 이식)
    # ==================================================================
    def _setup_synergy(self, oppose_sign: float) -> None:
        p, cfg, dev, N = self.profile, self.env.cfg, self.device, self.N
        for _f in ("hand_joint_names", "hand_open_pose", "hand_grip_pose"):
            if not getattr(p, _f):
                raise RuntimeError(f"[{p.name}] 시너지 손 제어에 필요한 프로필 필드 {_f} 가 없다")
        n = len(p.hand_joint_names)
        if len(p.hand_open_pose) != n or len(p.hand_grip_pose) != n:
            raise RuntimeError(f"[{p.name}] 자세 배열 길이 불일치")
        jn = self.robot.data.joint_names
        self.syn_ids = [jn.index(nm) for nm in p.hand_joint_names]
        self.syn_t = torch.tensor(self.syn_ids, device=dev, dtype=torch.long)
        self.syn_open = torch.tensor(p.hand_open_pose, device=dev)
        self.syn_grip = torch.tensor(p.hand_grip_pose, device=dev)

        # ★per_finger 레이아웃: 프로필 `hand_finger_channels[finger][sfx]` = 액션 슬롯. 구동관절과
        #   슬롯이 1:1 이어야 한다(빠지거나 겹치면 부팅 실패 — 슬롯이 죽는 것은 지표에 안 보인다).
        if not p.hand_finger_channels:
            raise RuntimeError(f"[{p.name}] per_finger 레이아웃에 필요한 hand_finger_channels 가 없다")
        ch, fi, slot = [], [], []
        for nm in p.hand_joint_names:
            sfx = nm.rsplit("_", 1)[1]
            if sfx not in p.hand_channel_of_joint:
                raise RuntimeError(f"[{p.name}] hand_channel_of_joint 에 접미사 {sfx} 없음")
            ch.append(int(p.hand_channel_of_joint[sfx]))
            hit = [i for i, f in enumerate(self.fingers) if f"_{f}_" in nm]
            if len(hit) != 1:
                raise RuntimeError(f"[{p.name}] 관절 {nm} 의 손가락을 특정 못함: {hit}")
            fi.append(hit[0])
            fmap = p.hand_finger_channels.get(self.fingers[hit[0]], {})
            if sfx not in fmap:
                raise RuntimeError(f"[{p.name}] hand_finger_channels[{self.fingers[hit[0]]}] 에 접미사 {sfx} 없음")
            slot.append(int(fmap[sfx]))
        if sorted(slot) != list(range(n)):
            raise RuntimeError(f"[{p.name}] per_finger 슬롯이 0..{n-1} 을 정확히 덮지 않는다: {slot}")
        self.syn_ch = torch.tensor(ch, device=dev, dtype=torch.long)
        self.syn_fi = torch.tensor(fi, device=dev, dtype=torch.long)
        self.syn_slot = torch.tensor(slot, device=dev, dtype=torch.long)
        self.syn_nch = len(set(ch))
        self.hand_action_width = n
        # 4지 결합 행렬(n×n): contact_group_a(엄지) 밖 손가락의 같은-접미사 관절은 지령 평균을 쓴다.
        grp_a = set(int(i) for i in self.grp_a.tolist())
        M = torch.eye(n, device=dev)
        sfx_all = [nm.rsplit("_", 1)[1] for nm in p.hand_joint_names]
        for i in range(n):
            if fi[i] in grp_a:
                continue
            group = [j for j in range(n) if fi[j] not in grp_a and sfx_all[j] == sfx_all[i]]
            M[i, :] = 0.0
            M[i, group] = 1.0 / len(group)
        self.syn_couple = M

        # 대향 관절 grip = open + delta (grasp_s2r `_apply_pose_knobs`). 부호는 측별.
        d = float(cfg.oppose_grip_delta_rad) * float(oppose_sign)
        if d != 0.0:
            idx = [i for i, nm in enumerate(p.hand_joint_names)
                   if ch[i] == 1 and any(f"_{f}_" in nm for f in p.contact_group_a)]
            if not idx:
                raise RuntimeError(f"[{p.name}] 대향 손가락 ch1 관절을 못 찾았다")
            for i in idx:
                self.syn_grip[i] = self.syn_open[i] + d

        # 동결 집합은 프로필이 정한다(접미사 리터럴 금지). RH56F1: ("1","2") = 구동 6 전부.
        sfx = [nm.rsplit("_", 1)[1] for nm in p.hand_joint_names]
        if not p.hand_freeze_suffixes:
            raise RuntimeError(f"[{p.name}] hand_freeze_suffixes 가 비어 접촉 동결이 아무 관절도 안 잡는다")
        self.syn_flex = torch.tensor([s in p.hand_freeze_suffixes for s in sfx], device=dev)
        self.syn_movable = (self.syn_grip - self.syn_open).abs() > 1e-4
        if not bool(self.syn_movable.any()):
            raise RuntimeError(f"[{p.name}] 가동 손관절이 없다 — open/grip 자세 확인")
        self.syn_close = torch.zeros(N, n, device=dev)
        self.syn_target = self.syn_open.unsqueeze(0).expand(N, n).clone()
        self.syn_vel = torch.zeros(N, n, device=dev)
        lim = self.robot.data.soft_joint_pos_limits[0, self.syn_t, :]
        self.syn_lo, self.syn_hi = lim[:, 0].contiguous(), lim[:, 1].contiguous()
        _bad_open = ((self.syn_open < self.syn_lo) | (self.syn_open > self.syn_hi)).nonzero()
        if len(_bad_open):
            raise RuntimeError(
                f"[{p.name}] hand_open_pose 가 관절한계 밖: "
                f"{[p.hand_joint_names[int(i)] for i in _bad_open.flatten()]}")
        _grip_c = self.syn_grip.clamp(self.syn_lo, self.syn_hi)
        print(f"[pour_fabric_mimic:{self.role}] synergy {p.name}: 관절 {n} · 채널 {self.syn_nch} · "
              f"가동 {int(self.syn_movable.sum())} · grip 한계clamp "
              f"{int((self.syn_grip != _grip_c).sum())}개 · oppose {d:+.2f}", flush=True)

    def direct_hand_targets(self, a_hand: torch.Tensor) -> torch.Tensor:
        """★09.28 손 full-joint(cfg `hand_control=direct`) — grasp_fj `_hand_targets`(hand_direct) 와 같은 꼴 + 벤더 속도 상한.

            raw  = lo + ½(a+1)(hi−lo)                [lo, hi] = cfg `hand_direct_range`:
                                                     "limits" = 관절 한계 전 범위, "grip" = 편 자세 ~ 쥔 자세(관절별 min·max)
            q*_t = α·raw + (1−α)·q*_{t-1}            관절 목표 EMA
            q*_t = q*_{t-1} + clamp(q*_t − q*_{t-1}, ±관절 한계 전 범위/(hand_full_range_s·60))   ← 실기 속도는 관절 전 범위 기준
            (hand_direct_contact_freeze) 컵 접촉력 > contact_freeze_threshold 인 손가락은 닫는 방향 변화 0 — 실기 RH56F1 펌웨어 힘 멈춤
            q*_t = clamp(q*_t, lo, hi)
        ★09.28 사용자 "1번, 2.a" — f01 ep1650 재생: 조기 종료 12번이 전부 mimic 폭주, 9번은 손가락 끝이 컵 벽 안(축 0.7~2.7 cm < r 2.8 cm),
          3번은 컵 접촉 0 에서 약지 종속 관절(손가락끼리 짓누름, 관절 한계 1.53 rad > 쥔 자세 0.85~1.08).
        `syn_close` 는 로그용 정규화 목표 (q*−open)/(grip−open).
        """
        cfg = self.env.cfg
        lim_lo, lim_hi = self.syn_lo.unsqueeze(0), self.syn_hi.unsqueeze(0)
        if str(cfg.hand_direct_range) == "grip":
            lo = torch.minimum(self.syn_open, self.syn_grip).clamp(self.syn_lo, self.syn_hi).unsqueeze(0)
            hi = torch.maximum(self.syn_open, self.syn_grip).clamp(self.syn_lo, self.syn_hi).unsqueeze(0)
        else:
            lo, hi = lim_lo, lim_hi
        floor = float(getattr(cfg, "hand_finger_open_floor_rad", 0.0))
        if floor > 0.0:                                       # ★09.30 실기: 네 손가락은 레지스터 1740(2.2~5.2°)까지만 편다
            lo = torch.where(self._finger_mask.unsqueeze(0), lo.clamp(min=floor), lo)
            hi = torch.maximum(hi, lo)
        a_j = a_hand.clamp(-1.0, 1.0)[:, self.syn_slot]
        raw = lo + 0.5 * (a_j + 1.0) * (hi - lo)
        alpha = float(cfg.hand_ema)
        prev = self.syn_target
        ema = alpha * raw + (1.0 - alpha) * prev
        cap = (lim_hi - lim_lo) / (float(cfg.hand_full_range_s) * 60.0)
        if float(getattr(cfg, "hand_vel_cap_rad_s", 0.0)) > 0.0:               # ★09.30 실기 속도(배포 pd 2.1 rad/s, 엄지 굽힘 따로)
            cap = self._real_hand_cap.unsqueeze(0)
        delta = (ema - prev).clamp(-cap, cap)
        if bool(cfg.hand_direct_contact_freeze):
            mid, dist, _ = self.finger_link_forces()
            thr = float(cfg.contact_freeze_threshold)
            touch = ((mid > thr) | (dist > thr))[:, self.syn_fi] & self.syn_flex          # 손가락당 구동관절(시너지 동결과 같은 스코프)
            close_dir = torch.sign(self.syn_grip - self.syn_open).unsqueeze(0)          # 관절별 닫는 방향 부호
            closing = (delta * close_dir) > 0.0
            delta = torch.where(touch & closing, torch.zeros_like(delta), delta)
        tgt = (prev + delta).clamp(lo, hi)
        span = (self.syn_grip - self.syn_open).unsqueeze(0)
        safe = torch.where(span.abs() > 1e-6, span, torch.ones_like(span))
        self.syn_close = ((tgt - self.syn_open.unsqueeze(0)) / safe).clamp(0.0, 1.0)
        return tgt

    def synergy_targets(self, a_hand: torch.Tensor, close_gate: torch.Tensor) -> torch.Tensor:
        """액션(손가락×채널) → 관절 목표 (N, n). grasp_s2r contact 모드 그대로."""
        cfg, N = self.env.cfg, self.N
        if str(cfg.hand_control) == "direct":
            return self.direct_hand_targets(a_hand)
        a_j = a_hand.clamp(-1.0, 1.0)[:, self.syn_slot]           # (N, n) 관절 순서
        if bool(cfg.couple_four_fingers):
            common = a_j @ self.syn_couple.T
            rs = float(cfg.finger_residual_scale)
            a_j = common if rs == 0.0 else common + rs * (a_j - common)
        rate = float(cfg.synergy_close_speed)
        if str(cfg.hand_action_mode) == "increment":
            delta = a_j * rate                                  # ★09.28 증분: a = 닫힘 속도(최대 rate 의 비율)
        else:
            cmd_j = 0.5 * (a_j + 1.0)
            delta = (cmd_j - self.syn_close).clamp(-rate, rate)
        g = close_gate.unsqueeze(1)
        delta = torch.where(delta > 0.0, delta * g, delta)      # 닫는 방향만 게이트
        if bool(cfg.synergy_contact_freeze):
            mid, dist, _ = self.finger_link_forces()
            thr = float(cfg.contact_freeze_threshold)       # 09.17: 파지 판정(1 N)과 분리
            h_mid = (mid > thr)[:, self.syn_fi]
            h_dist = (dist > thr)[:, self.syn_fi]
            # 손가락당 구동관절이 하나뿐이라 "닿은 마디만" 스코프는 정의되지 않는다 — finger 스코프만.
            hold = (h_mid | h_dist) & self.syn_flex
            delta = torch.where(hold & (delta > 0.0), torch.zeros_like(delta), delta)
        self.syn_close = (self.syn_close + delta).clamp(0.0, 1.0)
        tgt = torch.lerp(self.syn_open.unsqueeze(0), self.syn_grip.unsqueeze(0), self.syn_close)
        return tgt.clamp(self.syn_lo.unsqueeze(0), self.syn_hi.unsqueeze(0))

    def closure(self) -> torch.Tensor:
        """실측 평균 폐쇄도 (N,) [0,1] — 가동 관절만."""
        q = self.robot.data.joint_pos[:, self.syn_t]
        span = (self.syn_grip - self.syn_open).unsqueeze(0)
        prog = ((q - self.syn_open.unsqueeze(0)) / span).clamp(0.0, 1.0)
        return prog[:, self.syn_movable].mean(dim=1)

    def joint_err(self) -> torch.Tensor:
        err = self.syn_target - self.robot.data.joint_pos[:, self.syn_t]
        return (err / float(self.env.cfg.joint_pos_err_max)).clamp(-1.0, 1.0)

    # ==================================================================
    # 접촉
    # ==================================================================
    def _mag(self, sensor) -> torch.Tensor:
        return sensor.data.force_matrix_w.view(self.N, -1, 3).sum(dim=1).norm(dim=-1)

    def finger_forces(self) -> torch.Tensor:
        """손가락별 자기 컵 접촉력 합 (N,F)."""
        out = []
        for f in self.fingers:
            t = torch.zeros(self.N, device=self.device)
            for s in self.sensors[f]:
                t = t + self._mag(s)
            out.append(t)
        return torch.stack(out, dim=1)

    def finger_link_forces(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(중간, 원위, 팁) (N,F). `finger_sensor_bodies` 규약: 마지막=팁, 앞이 (중간, 원위).

        RH56F1 은 (중간, 센서) 2개다(09.15) — 손끝 마디 collider 가 `_sensor` 하나라 원위=팁=센서.
        """
        mids, dists, tips = [], [], []
        for f in self.fingers:
            ss = self.sensors[f]
            mi, di = (0, 1) if len(ss) >= 2 else (0, 0)
            mids.append(self._mag(ss[mi])); dists.append(self._mag(ss[di])); tips.append(self._mag(ss[-1]))
        return torch.stack(mids, 1), torch.stack(dists, 1), torch.stack(tips, 1)

    def palm_force(self) -> torch.Tensor:
        return self._mag(self.palm_sensor)

    def foreign_force(self) -> torch.Tensor:
        """손(손가락 마디+손바닥)이 **자기 컵 외**의 것에 닿는 힘 (N,) — 09.14 s2r 충돌 신호.

        `net_forces_w` 는 링크가 받는 모든 접촉, `force_matrix_w` 는 자기 컵 필터 접촉이다.
        둘의 크기 차(≥0)가 상대 손·상대 컵·테이블과의 접촉이다(grasp_s2r `_mag_net` 진단 규약).
        """
        tot = torch.zeros(self.N, device=self.device)
        sensors = [s for f in self.fingers for s in self.sensors[f]] + [self.palm_sensor]
        for s in sensors:
            net = s.data.net_forces_w.view(self.N, -1, 3).sum(dim=1).norm(dim=-1)
            tot = tot + (net - self._mag(s)).clamp(min=0.0)
        return tot

    def tip_tactile(self) -> torch.Tensor:
        """손끝 촉각 (N,F) [N] — 실기 RH56F1 TouchData1.finger_forces 대응.

        실기 센서는 닿은 대상과 무관하게 재므로 컵 필터 행렬이 아니라 손끝 링크 전체 접촉력(net)을 쓴다.
        """
        out = [self.sensors[f][-1].data.net_forces_w.view(self.N, -1, 3).sum(dim=1).norm(dim=-1)
               for f in self.fingers]
        return torch.stack(out, dim=1)

    def grasped(self, forces: torch.Tensor) -> torch.Tensor:
        thr = float(self.env.cfg.contact_force_threshold)
        return (forces[:, self.grp_a] > thr).any(dim=1) & (forces[:, self.grp_b] > thr).any(dim=1)

    # ==================================================================
    # 좌표
    # ==================================================================
    def palm_pos(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.palm_idx] - self.env.scene.env_origins

    def palm_lin_vel(self) -> torch.Tensor:
        """손바닥 선속도 [m/s] (world 축 = env-local 축)."""
        return self.robot.data.body_lin_vel_w[:, self.palm_idx]

    def palm_R(self) -> torch.Tensor:
        return matrix_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])

    def palm_pose_6d(self) -> torch.Tensor:
        """env-local xyz + [yaw, pitch, roll] — fabric "euler_zyx" 지령과 같은 규약."""
        r, pi, y = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])
        return torch.cat([self.palm_pos(), torch.stack([y, pi, r], dim=1)], dim=1)

    def tips_pos(self) -> torch.Tensor:
        """(N,F,3) env-local."""
        return self.robot.data.body_pos_w[:, self.tip_t] - self.env.scene.env_origins[:, None, :]

    # ==================================================================
    # Fabrics
    # ==================================================================
    def setup_fabric(self, fabric_class, integrator_cls, arm_slice: slice, hand_slice: slice) -> None:
        """양팔 26 DOF fabric 인스턴스(이 팔의 palm 만 활성). 슬라이스는 `bimanual.fabric_slots`."""
        p, cfg = self.profile, self.env.cfg
        if not p.fabric_class or not p.fabric_robot_dir:
            raise RuntimeError(f"[{p.name}] fabric_class/fabric_robot_dir 가 없다")
        # ★RH56F1 fabric 생성자: (batch_size, device, timestep, graph_capturable, use_hand_fabric,
        #   hand_mode). robot_dir·params 는 클래스가 고정 로드한다 — tesollo 식 kwargs 를 주면 TypeError.
        self.fabric = fabric_class(
            batch_size=self.N, device=self.device, timestep=float(cfg.fabrics_dt),
            graph_capturable=bool(cfg.fabric_use_cuda_graph),
            use_hand_fabric=False, hand_mode="direct")
        self.integrator = integrator_cls(self.fabric)
        self._apply_fabric_preset(str(cfg.fabric_param_preset))
        self.arm_sl, self.hand_sl = arm_slice, hand_slice
        expect = len(p.fabric_joint_order)
        if self.fabric.num_joints != expect or expect != 2 * (p.num_arm_joints + p.num_hand_joints):
            raise RuntimeError(f"[{p.name}] fabric num_joints={self.fabric.num_joints} != {expect} "
                               f"(양팔 26 DOF 규약)")
        if (arm_slice.stop - arm_slice.start) != p.num_arm_joints or (hand_slice.stop - hand_slice.start) != p.num_hand_joints:
            raise RuntimeError(f"[{p.name}] fabric 슬라이스 폭이 프로필과 다르다: {arm_slice} {hand_slice}")
        idx = []
        for name in p.fabric_joint_order:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise RuntimeError(f"[{p.name}] fabric 관절 '{name}' 해석 실패: {ids}")
            idx.append(ids[0])
        self.fab_t = torch.tensor(idx, device=self.device, dtype=torch.long)
        # 이 팔의 fabric 팔 슬라이스가 프로필 팔 관절과 같은 관절인지(순서까지) 대조.
        if self.fab_t[self.arm_sl].tolist() != self.arm_ids:
            raise RuntimeError(f"[{p.name}] fabric 팔 슬라이스 {self.fab_t[self.arm_sl].tolist()} "
                               f"≠ 프로필 팔 관절 {self.arm_ids}")
        syn_pos = {int(j): k for k, j in enumerate(self.syn_ids)}
        fab_hand = self.fab_t[self.hand_sl].tolist()
        missing = [int(j) for j in fab_hand if int(j) not in syn_pos]
        if missing:
            raise RuntimeError(f"[{p.name}] synergy 자세에 없는 fabric 손 관절 {missing}")
        self.syn_to_fab_idx = torch.tensor([syn_pos[int(j)] for j in fab_hand],
                                           device=self.device, dtype=torch.long)
        q0 = self.reset_q.unsqueeze(0).expand(self.N, -1)
        self.fabric_q = q0[:, self.fab_t].contiguous()
        self.fabric_qd = torch.zeros(self.N, self.fabric.num_joints, device=self.device)
        self.fabric_qdd = torch.zeros_like(self.fabric_qd)
        self.fabric_hand_cmd = torch.zeros(self.N, p.num_hand_joints, device=self.device)   # direct 계약(무시됨)
        # cspace rest = 시작 자세(널스페이스가 시작 자세로 당긴다) — 양팔 전부.
        self.fabric.default_config.copy_(self.fabric_q)
        self.fabric_damping = float(cfg.fabrics_damping_gain) * torch.ones(self.N, 1, device=self.device)
        self.velocity_ff_scale = float(cfg.fabric_velocity_ff_scale)   # fabric_adr 가 켜지면 env 가 ADR 로 옮긴다
        if not p.palm_box_verified:
            print(f"[pour_fabric_mimic:{self.role}] ⚠ palm_box 미검증({p.name}) — probe 후 승격할 것",
                  flush=True)

    def _apply_fabric_preset(self, preset: str) -> None:
        """cfg `fabric_param_preset` — "dextrah" 면 yaml 의 DEXTRAH 이탈 값 4개를 이 인스턴스에서만 되돌린다.
        attractor·repulsion 항은 fabric_params 의 같은 dict 를 참조하고, engage_depth 만 생성 때 warp 입력 dict 로
        복사되므로 거기도 맞춘다."""
        if preset == "track":
            return
        if preset != "dextrah":
            raise ValueError(f"fabric_param_preset 는 track|dextrah: {preset}")
        fp = self.fabric.fabric_params
        fp["palm_attractor"]["conical_gain"] = 50.0
        br = fp["body_repulsion"]
        br["engage_depth"], br["constant_accel"], br["constant_accel_geom"] = 0.5, 1.0, 5.0
        self.fabric.base_fabric_repulsion.allocated_data["engage_depth"] = 0.5
        print(f"[pour_fabric_mimic:{self.role}] fabric 프리셋 dextrah: palm conical_gain 50 · repulsion "
              f"engage 0.5 · accel 1 · accel_geom 5", flush=True)

    def init_anchor(self) -> None:
        """시작 자세의 palm 실측 → 앵커 + fabric FK 정합 게이트. 물리 2스텝 뒤에 부를 것."""
        p, cfg = self.profile, self.env.cfg
        home = self.palm_pose_6d()[0]
        q0f = self.reset_q.unsqueeze(0).expand(self.N, -1)[:, self.fab_t].contiguous()
        nt = len(self.fingers)
        tips_fab = self.fabric._fingertip_taskmap(q0f, None)[0].reshape(self.N, nt, 3)[0]
        tips_sim = self.tips_pos()[0]
        delta = tips_sim - tips_fab
        spread = float(delta.std(dim=0).max())
        # ★09.13: 물리 2스텝 중력 처짐이 손끝마다 달라 산포 1~2 mm 가 노이즈 바닥이다(2 mm 로 두면
        #   경계에서 무작위로 죽는다 — 로컬 실측 2.1 mm). 기구학 불일치는 수십 mm 로 나오므로 5 mm.
        if spread > 5e-3:
            raise RuntimeError(
                f"[{p.name}] fabric↔env 프레임이 순수 평행이동이 아니다(산포 {spread*1000:.1f}mm)")
        self.fab_to_env = delta.mean(dim=0)
        fab = self.fabric.get_palm_pose(self.fabric_q.detach(), "euler_zyx")[0]
        dp = float(torch.norm((fab[:3] + self.fab_to_env) - home[:3]))
        # ★각도 차는 wrap 한다 — 좌팔은 yaw 가 ±π 경계에 걸려(−3.127 vs +3.13) 원본 식이면 359° 가 난다.
        _d = torch.remainder(fab[3:] - home[3:] + math.pi, 2 * math.pi) - math.pi
        dr = float(torch.max(torch.abs(_d)))
        print(f"[pour_fabric_mimic:{self.role}] 시작 palm={[round(v, 4) for v in home.tolist()]} | "
              f"fabric FK 정합 pos {dp*1000:.2f}mm rot {math.degrees(dr):.2f}° | "
              f"fab→env {[round(float(v)*1000) for v in self.fab_to_env]}mm", flush=True)
        if dp > float(cfg.fabric_fk_pos_tol) or dr > math.radians(2.0):
            raise RuntimeError(
                f"[{p.name}] fabric FK 가 USD palm 과 어긋난다: {dp*1000:.1f}mm / "
                f"{math.degrees(dr):.1f}° — fabric URDF·joint_order·palm_body 를 본다")
        out = (home[:3] < self.box_lo) | (home[:3] > self.box_hi)
        if bool(out.any()):
            raise RuntimeError(f"[{p.name}] 시작 palm 이 워크스페이스 박스 밖: {home[:3].tolist()}")
        self.anchor_env[:] = home.unsqueeze(0)
        anchor_fab = home.clone()
        anchor_fab[:3] -= self.fab_to_env
        self.anchor[:] = anchor_fab.unsqueeze(0)
        self.palm_targets[:] = self.anchor
        # 델타 박스가 팔 박스에 잘리는 축을 부팅에서 알린다.
        lo_env, hi_env = home[:3] + self.delta_lo[:3], home[:3] + self.delta_hi[:3]
        cut = [ax for i, ax in enumerate("xyz")
               if float(lo_env[i]) < float(self.box_lo[i]) or float(hi_env[i]) > float(self.box_hi[i])]
        if cut:
            print(f"[pour_fabric_mimic:{self.role}] ⚠ 델타 박스가 프로필 palm 박스에 잘리는 축 {cut}",
                  flush=True)

    def compose_palm_target(self, a6: torch.Tensor, active: torch.Tensor) -> None:
        """palm 6D = 앵커 + 델타. hold 중(active=False)은 앵커.

        ★델타 박스가 lo/hi **비대칭**이라 grasp_s2r 의 선형 매핑(0.5(a+1)(hi−lo)+lo)을 쓰면
          a=0 이 앵커가 아니라 박스 중점이 된다(09.13 부팅 실측). a≥0 → a·hi, a<0 → −a·lo
          로 부호별로 매핑해 **a=0 = 앵커**를 지킨다(구 pour `symmetric_action_scale=False`).
        """
        delta = torch.where(a6 >= 0.0, a6 * self.delta_hi, -a6 * self.delta_lo)
        raw = self.anchor + delta
        pos_env = raw[:, :3] + self.fab_to_env
        pos_env = pos_env.clamp(self.box_lo, self.box_hi)
        raw = torch.cat([pos_env - self.fab_to_env, raw[:, 3:]], dim=1)
        self.palm_targets = torch.where(active.unsqueeze(1), raw, self.anchor)

    def workspace_palm_target(self, a6: torch.Tensor, active: torch.Tensor) -> None:
        """★09.28 DEXTRAH 지령: 위치 = palm 박스 전체 선형 사상(a=0 = 박스 중심), 회전 = 앵커 ± 델타 회전 구간(부호별,
        a=0 = 앵커). 누적·필터 없음 — 목표까지 가는 궤적은 fabric 이 만든다. hold env 는 앵커.
        """
        pos_env = 0.5 * (a6[:, :3] + 1.0) * (self.box_hi - self.box_lo) + self.box_lo
        rot = self.anchor[:, 3:] + torch.where(a6[:, 3:] >= 0.0, a6[:, 3:] * self.delta_hi[3:],
                                               -a6[:, 3:] * self.delta_lo[3:])
        raw = torch.cat([pos_env - self.fab_to_env, rot], dim=1)
        self.palm_targets = torch.where(active.unsqueeze(1), raw, self.anchor)

    def init_joint_arm(self) -> None:
        """joint 모드 관절 한계 = soft 한계(= 하드, soft_joint_pos_limit_factor 1.0). 시뮬 시작 뒤에 부른다."""
        lim = self.robot.data.soft_joint_pos_limits[0, self.arm_t, :]
        self.arm_lo, self.arm_hi = lim[:, 0].clone(), lim[:, 1].clone()
        cfg = self.env.cfg
        if str(cfg.arm_joint_mode) == "absolute":
            print(f"[pour_fabric_mimic:{self.role}] 팔 관절 절대 목표(fabric 없음) · a=0 시작 자세, ±1 관절 한계 · "
                  f"EMA α {cfg.arm_ema} · 속도 상한 {cfg.arm_abs_vmax} rad/s · 위치 목표만", flush=True)
            return
        print(f"[pour_fabric_mimic:{self.role}] 팔 관절 증분 지령(fabric 없음) · k_arm {cfg.k_arm} rad/스텝 · "
              f"EMA α {cfg.arm_ema} · 포화 slew "
              f"{float(cfg.arm_ema) * float(cfg.k_arm) * 60.0:.2f} rad/s · 위치 목표만", flush=True)

    def step_joint_arm(self, a_arm: torch.Tensor, active: torch.Tensor) -> None:
        """grasp_fj `_arm_command`: q_raw = clamp(q* + k·a) → q* = clamp(α·q_raw + (1−α)·q*). hold env 는 시작 자세."""
        cfg = self.env.cfg
        alpha = float(cfg.arm_ema)
        q_home = self.reset_q[self.arm_t].unsqueeze(0).expand_as(self.arm_q_target)
        if str(cfg.arm_joint_mode) == "absolute":
            # ★09.29 시작 자세 기준 절대 목표: a=0 → home, ±1 → 관절 한계(구간별 선형), EMA + 속도 상한
            a = a_arm.clamp(-1.0, 1.0)
            q_raw = torch.where(a >= 0.0, q_home + a * (self.arm_hi - q_home), q_home + a * (q_home - self.arm_lo))
            cap = float(cfg.arm_abs_vmax) * float(self.env.step_dt)
            step = (alpha * (q_raw - self.arm_q_target)).clamp(-cap, cap)
            q_new = (self.arm_q_target + step).clamp(self.arm_lo, self.arm_hi)
        else:
            q_raw = (self.arm_q_target + float(cfg.k_arm) * a_arm).clamp(self.arm_lo, self.arm_hi)
            q_new = (alpha * q_raw + (1.0 - alpha) * self.arm_q_target).clamp(self.arm_lo, self.arm_hi)
        self.arm_q_target = torch.where(active.unsqueeze(1), q_new, q_home)

    def fabric_arm_q(self) -> torch.Tensor:
        """DEXTRAH 관측: fabric 팔 슬라이스 q (N, A). qd·qdd 는 DEXTRAH 도 0 을 넣는다(cfg 주석)."""
        return self.fabric_q[:, self.arm_sl]

    def step_palm_target(self, v_norm: torch.Tensor, vmax: torch.Tensor, dt: float, leash: tuple,
                         active: torch.Tensor) -> None:
        """★09.28 증분 지령(DG-5F pour_fabric 9d300ef4 이식): 목표 += v·vmax·dt → 델타 박스 → palm 박스
        → 현재 palm 근처로 묶기(leash). 목표는 누적 상태라 묶이면 거기서 멈춘다(와인드업 없음). hold env 는 앵커.
        """
        raw = step_palm_increment(self.palm_targets, v_norm, vmax, dt, self.anchor, self.delta_lo, self.delta_hi)
        pos_env = (raw[:, :3] + self.fab_to_env).clamp(self.box_lo, self.box_hi)
        raw = torch.cat([pos_env - self.fab_to_env, raw[:, 3:]], dim=1)
        pose = self.palm_pose_6d()
        pose_fab = torch.cat([pose[:, :3] - self.fab_to_env, pose[:, 3:]], dim=1)
        tied = leash_palm_target(raw, pose_fab, float(leash[0]), float(leash[1]))
        # leash 는 회전을 pose+wrap 차이로 돌려준다 — 앵커 기준 연속 각도로 되돌린다
        d_rot = torch.remainder(tied[:, 3:] - self.anchor[:, 3:] + math.pi, 2.0 * math.pi) - math.pi
        tied = torch.cat([tied[:, :3], self.anchor[:, 3:] + d_rot], dim=1)
        self.palm_targets = torch.where(active.unsqueeze(1), tied, self.anchor)

    def step_fabric(self, world_ids, world_indicator) -> None:
        cfg = self.env.cfg
        self.fabric.set_features(
            self.fabric_hand_cmd, self.palm_targets, "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            world_ids, world_indicator, self.fabric_damping)
        for _ in range(int(cfg.fabric_decimation)):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.fabric_qdd.detach(), float(cfg.fabrics_dt))

    def pin_fabric(self, hold: torch.Tensor, q_pin: torch.Tensor) -> None:
        """hold env 는 적분 결과를 시작 자세로 되돌린다(마스크 곱 — GPU sync 없음)."""
        h = hold.unsqueeze(1).float()
        self.fabric_q = h * q_pin + (1.0 - h) * self.fabric_q
        self.fabric_qd = (1.0 - h) * self.fabric_qd
        self.fabric_qdd = (1.0 - h) * self.fabric_qdd

    # ==================================================================
    # ★09.30 실기 제어 응답: 명령 지연(손 고정 스텝·팔 에피소드마다 균등)·손 속도 상한·네 손가락 편 한계
    # ==================================================================
    def init_real_control(self) -> None:
        cfg, dev, N = self.env.cfg, self.device, self.N
        names = [self.robot.data.joint_names[i] for i in self.syn_ids]
        self._finger_mask = torch.tensor(["thumb" not in nm for nm in names], device=dev)
        cap_all = float(cfg.hand_vel_cap_rad_s) / 60.0
        cap_th = float(cfg.hand_thumb_flex_vel_cap_rad_s) / 60.0
        self._real_hand_cap = torch.tensor(
            [cap_th if (cap_th > 0.0 and nm.endswith("thumb_2")) else cap_all for nm in names], device=dev)
        self._hand_delay = int(cfg.hand_cmd_delay_steps)
        lo, hi = (int(v) for v in cfg.arm_cmd_delay_steps)
        if self._hand_delay < 0 or lo < 0 or hi < lo:
            raise ValueError(f"명령 지연 설정이 잘못됐다: hand {self._hand_delay} arm {cfg.arm_cmd_delay_steps}")
        self._arm_delay_lo, self._arm_delay_hi = lo, hi
        self._arm_hist = self.arm_q_target.unsqueeze(1).repeat(1, hi + 1, 1) if hi > 0 else None
        self._syn_hist = self.syn_target.unsqueeze(1).repeat(1, self._hand_delay + 1, 1) if self._hand_delay > 0 else None
        self._arm_delay = torch.randint(lo, hi + 1, (N,), device=dev) if hi > 0 else None
        self.arm_applied, self.syn_applied = self.arm_q_target.clone(), self.syn_target.clone()

    def push_command(self) -> None:
        """정책 스텝마다 한 번: 새 목표를 기록하고 지연된 목표를 물리에 보낼 값으로 둔다(관측·EMA 는 지연 없는 q* — 배포 디코더와 같다)."""
        E = torch.arange(self.N, device=self.device)
        if self._arm_hist is not None:
            self._arm_hist = torch.roll(self._arm_hist, 1, dims=1)
            self._arm_hist[:, 0] = self.arm_q_target
            self.arm_applied = self._arm_hist[E, self._arm_delay]
        else:
            self.arm_applied = self.arm_q_target
        if self._syn_hist is not None:
            self._syn_hist = torch.roll(self._syn_hist, 1, dims=1)
            self._syn_hist[:, 0] = self.syn_target
            self.syn_applied = self._syn_hist[:, self._hand_delay]
        else:
            self.syn_applied = self.syn_target

    def reset_real_control(self, env_ids: torch.Tensor) -> None:
        if self._arm_hist is not None:
            self._arm_hist[env_ids] = self.arm_q_target[env_ids].unsqueeze(1)
            self._arm_delay[env_ids] = torch.randint(self._arm_delay_lo, self._arm_delay_hi + 1,
                                                     (len(env_ids),), device=self.device)
        if self._syn_hist is not None:
            self._syn_hist[env_ids] = self.syn_target[env_ids].unsqueeze(1)
        self.arm_applied = self.arm_applied.clone(); self.syn_applied = self.syn_applied.clone()
        self.arm_applied[env_ids] = self.arm_q_target[env_ids]
        self.syn_applied[env_ids] = self.syn_target[env_ids]

    def apply_action(self) -> None:
        cfg = self.env.cfg
        if self.joint_arm:
            # grasp_fj 규약: 위치 목표만 — 속도 목표는 0(실기 JTC 와 같은 PD). ★09.30 지연된 목표(arm_applied)를 보낸다
            self.robot.set_joint_position_target(self.arm_applied, joint_ids=self.arm_ids)
            self.robot.set_joint_velocity_target(torch.zeros_like(self.arm_q_target), joint_ids=self.arm_ids)
        else:
            arm_target = self.fabric_q[:, self.arm_sl]
            self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
            self.robot.set_joint_velocity_target(
                self.velocity_ff_scale * self.fabric_qd[:, self.arm_sl], joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.syn_applied, joint_ids=self.syn_ids)
        self.robot.set_joint_velocity_target(
            float(cfg.hand_velocity_ff_scale) * self.syn_vel, joint_ids=self.syn_ids)

    def sync_fabric_hand(self) -> None:
        """fabric 의 손 상태 = 시너지 목표 — 안 맞추면 fabric 이 없는 자기충돌을 피하려 팔을 민다."""
        self.fabric_q[:, self.hand_sl] = self.syn_target[:, self.syn_to_fab_idx]

    def sync_other(self, other: "SideRig") -> None:
        """반대팔 슬라이스 = 상대 rig 의 fabric 상태(양팔 26 DOF 를 두 인스턴스가 나눠 가진다)."""
        for sl in (other.arm_sl, other.hand_sl):
            self.fabric_q[:, sl] = other.fabric_q[:, sl]
            self.fabric_qd[:, sl] = other.fabric_qd[:, sl]
            self.fabric_qdd[:, sl] = other.fabric_qdd[:, sl]

    # ==================================================================
    def reset(self, env_ids: torch.Tensor) -> None:
        q0 = self.reset_q.unsqueeze(0)
        self.syn_close[env_ids] = 0.0
        self.syn_target[env_ids] = q0[:, self.syn_t]
        self.syn_vel[env_ids] = 0.0
        self.fabric_q[env_ids] = q0[:, self.fab_t]
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0
        self.palm_targets[env_ids] = self.anchor[env_ids]
        self.arm_q_target[env_ids] = q0[:, self.arm_t]
