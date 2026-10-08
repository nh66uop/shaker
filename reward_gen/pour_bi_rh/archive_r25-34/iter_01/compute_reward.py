import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.sqrt((d * d).sum(dim=-1) + 1e-12)


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기. DG-5F pour_bi2 iter_02 구조를 이식하고 파지 항만 RH(iter_00) 형태를 쓴다.
    모든 길이는 ctx.cup_radius 배수(들기 높이 허용만 절대값 예외), 각도는 LIM 배수."""
    tilt_src = ctx.src_cup_tilt
    zeros = torch.zeros_like(tilt_src)
    dt = tilt_src.dtype

    # ---- 공통 척도 ---------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # lip 수평 허가 반경
    wide = 10.0 * r_cup                      # WIDE 커널 (DG-5F 그대로)
    # premature_tilt_limit 대체 스칼라: RH 실측 방출 101° - 20° (DG-5F 는 유출 93.6° 대 한계 p50 73.6°)
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))   # bead_active_count 대체 (RH 20 개 고정)

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    ones = torch.ones_like(tilt_src)

    in_frac = _sat(ctx.bead_in_target_frac)
    state_decay = 1.0 - 0.5 * in_frac        # iter_00 파지 항의 감쇠 (담김이 오르면 상태 수입 축소)

    # ---- RH 에 없는 DG-5F 필드를 RH ctx 로 계산 (env pour_rules.py 와 같은 식) -----
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)                                   # pour_dir_xy
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])    # src_tilt_toward_rcv (부호 있음)
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)    # src_pour_lip_pos
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]

    # ---- A. 접근 (RH iter_00 형태, 팔당 0.4: fresh 학습이라 DG-5F 자릿수) ---------
    d_src = 0.5 * (_dist(ctx.src_palm_pos, ctx.src_cup_pos) + _dist(ctx.src_tips_pos[:, 0, :], ctx.src_cup_pos))
    d_rcv = 0.5 * (_dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos) + _dist(ctx.rcv_tips_pos[:, 0, :], ctx.rcv_cup_pos))
    near_src = 1.0 - torch.tanh(d_src / (4.0 * r_cup))
    near_rcv = 1.0 - torch.tanh(d_rcv / (4.0 * r_cup))
    approach_src = 0.4 * torch.where(ctx.src_grasped, ones, near_src) * state_decay
    approach_rcv = 0.4 * torch.where(ctx.rcv_grasped, ones, near_rcv) * state_decay

    # ---- A. 파지 유지 (팔당 0.5, DG-5F grasp 자릿수) ------------------------------
    grasp_hold = 0.5 * (src_g + rcv_g) * state_decay

    # ---- B. 들기 (RH iter_00 형태; 합 상한 0.5 = DG-5F lift_both 규모) -------------
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # 소스는 높게(K3), 과들기 세금은 충분히 높은 곳에서만 (높이 허용은 절대값 예외)
    lift_src = (0.3 * _sat(h_src / 0.28) - 0.15 * _sat((h_src - 0.45) / 0.20)) * src_g * state_decay
    # 리시버는 떼기만, 낮게
    lift_rcv = 0.2 * _sat(h_rcv / two_r) * torch.exp(-torch.clamp(h_rcv - 0.15, min=0.0) / 0.10) \
        * rcv_g * state_decay
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- C. 두 컵 수렴 (1.2, WIDE 커널, DG-5F 그대로) -----------------------------
    converge_cups = 1.2 * both_g * lifted * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- D. lip 조준 (0.8, DG-5F 그대로) ------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부: pour_target_frac = 1.0 (RH 는 전량 이송) -----------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F. 회전 구동 (8.0): 0° 부터 선형, 1.4*LIM(약 113°)에서 포화 ---------------
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3                          # 바닥 위 가산, 정밀 게이트 아님
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    # grasp_ref_valid 는 RH 에 없어 곱 인자에서 뺀다
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog

    # ---- F. 실제 도착한 비드 (알당 30) --------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 흘림 (알당 40 > 배달 30: 조준 불량 붓기는 순손실) ------------------------
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 마무리: 다 부은 뒤 소스를 세우고 쥐고 있기 (1.5 < 회전 수입 8) ------------
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # ---- 성공 보너스 (env 판정, 재정의 못 함; 기울인 채 버티기 최대 8 보다 크게) ----
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- 조기 기울임: 벽(0.95~1.0 LIM)에서만, lip 이 2r 밖일 때만 (래치 부분 제외) -
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- 실기 안전: 컵끼리 접촉, 포개기 (cup_hit 은 RH 에 없어 뺀다) ----------------
    cup_clash = -(2.0 * torch.tanh(ctx.cup_cup_force / 5.0)
                  + 3.0 * ctx.cups_nested.to(dt))

    # ---- 손은 자기 컵만 (self-collision ON 에서 팔끼리 접촉도 여기로 잡힌다) --------
    hand_foreign = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                           + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 붓는 방향이 몸 바깥(+x)을 향하지 않게 ------------------------------------
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 리시버 컵: 직립, 중심선 기준 자기 쪽(+y) ---------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)   # rcv_side_margin = env-local y
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- G. 관절 여유: 붓는 중에만 작은 가산 (최대 0.3, 회전 구동에 곱하지 않음) ----
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # ---- 제어 정규화 (DG-5F 그대로) ------------------------------------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))
    # adr_progress 없음: 상수 0.1 포화 벌점 (DG-5F 원형대로 전 차원 평균)
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -0.1 * sat

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_hold": grasp_hold,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "converge_cups": converge_cups,
        "aim_lip": aim_lip,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "spill": spill,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "premature": premature,
        "cup_clash": cup_clash,
        "hand_foreign": hand_foreign,
        "pour_dir": pour_dir,
        "rcv_keep": rcv_keep,
        "posture_margin": posture_margin,
        "action_rate": action_rate,
        "action_flip": action_flip,
        "arm_tremor": arm_tremor,
        "action_reg_adr": action_reg_adr,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
