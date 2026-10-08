import torch
import math


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d, r):
    """손끝-벽 거리 성형 (iter_03 c1r2 그대로, 척도는 컵 반지름 배수: 1.2 r, 0.22 r)."""
    return 0.5 * torch.exp(-d / (1.2 * r)) + 0.5 * torch.exp(-d / (0.22 * r))


def _limit_cost(margin: torch.Tensor) -> torch.Tensor:
    """iter_08 변경 3: 관절 한계 근접 연속 비용 (관절 번호·각도 상수 없음).
    관절별 (1 - sat(m / 0.25))^2 : 여유 0.25 이상 0, 0.03 -> 0.77. 관절 합을 tanh 로 묶어 유계 [0,1)
    (i15: 관절 한계 강한 직접 벌점 금지 -> 팔당 최대 W_MARGIN)."""
    MARGIN_K = 0.25
    c = (1.0 - _sat(margin / MARGIN_K)) ** 2
    return torch.tanh(c.sum(dim=-1))


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r, other_cup_pos):
    """컵 좌표(축 = cup_up)에서의 손끝 기하. iter_07 그대로.
    손끝-벽 거리는 컵 바깥 면 기준. 입구 안(축 거리 < 벽 반경, 입구 높이 근처) 손끝은 벽에서 먼 것(2 r)으로 친다.
    입구 안 가중 w_in 을 손가락별로 돌려주고, 바깥 가중 out_w = 1 - w_in 은 접촉력 가중에 쓴다."""
    EPS = 1e-6
    H_W = 2.0
    grasp_h = 0.25 * cup_mouth_z
    thumb_hi = 0.5 * cup_mouth_z
    #   축 거리: 0.9 r 부터 0.6 r 까지 0 -> 1 (가설: 손끝 좌표가 바깥 벽 접촉에서도 r 을 약간 밑돌 수 있어 여유 0.1 r)
    #   높이: 입구 - 1.5 r ~ 입구 + 1.0 r 에서 1, 양끝 0.5 r 에 걸쳐 0 (가설: 아래쪽 테이퍼 몸통 오판 방지)
    IN_RAD_LO, IN_RAD_W = 0.9, 0.3
    IN_H_LO, IN_H_HI, IN_H_W = 1.5, 1.0, 0.5
    IN_FAR = 2.0          # 입구 안 손끝의 유효 벽 거리 = 2 r

    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)
    rad_vec = rel - h_tip.unsqueeze(-1) * up
    rad = rad_vec.norm(dim=-1)
    gap_raw = torch.clamp(rad - tip_r, min=0.0)

    w_rad = _sat((IN_RAD_LO * tip_r - rad) / (IN_RAD_W * tip_r))
    w_h = _sat((h_tip - (cup_mouth_z - (IN_H_LO + IN_H_W) * tip_r)) / (IN_H_W * tip_r)) \
        * _sat((cup_mouth_z + (IN_H_HI + IN_H_W) * tip_r - h_tip) / (IN_H_W * tip_r))
    w_in = w_rad * w_h                                             # (N,F) 1 = 입구 안
    gap = (1.0 - w_in) * gap_raw + w_in * (IN_FAR * tip_r)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_each = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2)          # (N,4) 손가락별 손끝-바깥 벽 거리
    d_oth = d_each.min(dim=1).values

    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))

    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2 + 1e-12)

    to_other = other_cup_pos[:, :2] - cup_pos[:, :2]
    e_other = to_other / (to_other.norm(dim=-1, keepdim=True) + EPS)
    lat = (rel_m[:, :2] * e_other).sum(dim=-1)
    cross_d = torch.clamp(lat - tip_r, min=0.0)

    d_app = rad_m + H_W * torch.abs(h_m - grasp_h) + H_W * cross_d
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.7 * tip_r) * (1.0 - w_in[:, 0])
    return {"d_thumb": d_thumb, "d_oth": d_oth, "d_each": d_each, "opp": opp, "d_grip": d_grip,
            "d_app": d_app, "thumb_band": thumb_band, "out_w": 1.0 - w_in}


def _envelope_score(palm_pos, palm_axes, cup_pos, cup_up):
    """K7 인벨롭 방향 점수. iter_07 그대로."""
    EPS = 1e-6
    ENV_N, ENV_F = 0.67, 0.73
    NORMAL_TOL = 0.3
    NORMAL_PEN = 0.3
    FINGER_TOL = 0.4
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    ax_dot = (f_dir * cup_up).sum(dim=-1)
    hand_level = torch.exp(-(ax_dot / FINGER_TOL) ** 2)
    n_dot = (n_dir * cup_up).sum(dim=-1)
    normal_level = torch.exp(-(n_dot / NORMAL_TOL) ** 2)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    tgt = ENV_N * n_dir + ENV_F * f_dir
    tgt_h = tgt - (tgt * cup_up).sum(dim=-1, keepdim=True) * cup_up
    tgt_hat = tgt_h / (tgt_h.norm(dim=-1, keepdim=True) + EPS)
    cos_env = (e_in * tgt_hat).sum(dim=-1)
    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    score = hand_level * (0.25 * normal_level + 0.75 * dir_env) - NORMAL_PEN * (1.0 - normal_level)
    return _sat(score), hand_level, normal_level


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos, other_cup_pos,
                    cup_r, cup_mouth_z, pose_ok, lift_split, lift_target, over_h, over_span, lift_relief):
    """팔별 무가중 항 (두 팔 같은 식, 한 팔에만 붙는 항·부호·상수 없음).
    iter_08 변경 2: lift_target·over_h 는 텐서도 받는다(소스: 리시버 입구를 넘을 여유 높이, 기하로 계산).
      lift_relief (N,) [0,1]: 들기 진척 lin 을 1 쪽으로 채우는 비율. 소스는 '조준 창 안·기울이는 중·주둥이가
      입구 위'일 때 1 -> 붓는 높이로 내려와도 들기 수입이 줄지 않는다(변경 1 과 짝). 리시버는 0.
    그 밖은 iter_07 c1r 그대로."""
    ORIENT_FLOOR = 0.3
    APP_FLOOR = 0.5
    APP_LIN_K = 20.0
    APP_NEAR_K = 3.0
    PALM_H_K = 0.15       # K7: 손바닥 높이 = 컵 원점 + 0.15 x cup_mouth_z
    SIDE_K = 1.5          # K7: 손바닥-컵 축 1.5 r
    LOW_H_K = 1.0
    LOW_H_FAR_K = 6.0
    LOW_SIDE_K = 3.0
    LOW_SIDE_FLOOR = 0.3
    LEVEL_FLOOR = 0.4
    LOW_W = 0.5           # iter_07 재시도 [23] 값 그대로
    HIGH_K = 1.5
    HIGH_FLOOR = 0.2
    FINGER_F = 0.7        # 손가락별 접촉 척도 [N]
    FIVE_FLOOR = 0.3
    WRAP_CAP = 0.8
    WRAP_K = 0.5
    HOLD_ANG_TOL = 0.4
    GRASP_TOUCH = math.tanh(1.0 / FINGER_F)   # 1 N 바깥 접촉 점수 -> 판정 인정 1

    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r, other_cup_pos)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    out_w = geo["out_w"]
    f_raw = torch.clamp(finger_force, min=0.0)
    f = f_raw * out_w
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)

    touch_each = torch.tanh(f[:, 1:] / FINGER_F)
    part = touch_each.mean(dim=1)
    finger_fac = FIVE_FLOOR + (1.0 - FIVE_FLOOR) * part * part
    oth_ok = _sat(touch_each.max(dim=1).values / GRASP_TOUCH)
    g_eff = g_f * out_w[:, 0] * oth_ok

    # ---- 높이
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)
    t_up = torch.tanh(h_pos / 0.02)
    lift_hold = g_eff * t_up

    # ---- K7 방향 인자
    env_score, hand_level, normal_level = _envelope_score(palm_pos, palm_axes, cup_pos, cup_up)
    orient_fac = ORIENT_FLOOR + (1.0 - ORIENT_FLOOR) * env_score
    orient_app = APP_FLOOR + (1.0 - APP_FLOOR) * env_score
    o_grip = orient_fac + (1.0 - orient_fac) * lift_hold
    o_app = orient_app + (1.0 - orient_app) * lift_hold

    # ---- 손바닥 높이 (컵 좌표)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    rad_p = (to_cup - h_c * cup_up).norm(dim=-1)
    h_palm = -h_c.squeeze(-1)
    palm_h_err = torch.abs(h_palm - PALM_H_K * cup_mouth_z)
    palm_h_near = torch.exp(-(palm_h_err / (LOW_H_K * cup_r)) ** 2)
    palm_h_far = _sat(1.0 - palm_h_err / (LOW_H_FAR_K * cup_r))
    palm_h_ok = 0.5 * palm_h_near + 0.5 * palm_h_far
    palm_hi = torch.clamp(h_palm - PALM_H_K * cup_mouth_z, min=0.0)
    high_fac = HIGH_FLOOR + (1.0 - HIGH_FLOOR) * torch.exp(-(palm_hi / (HIGH_K * cup_r)) ** 2)

    # ---- A. 접근
    d_app = geo["d_app"]
    reach_lin = _sat(1.0 - d_app / (APP_LIN_K * cup_r))
    reach_near = 1.0 - torch.tanh(d_app / (APP_NEAR_K * cup_r))
    reach_far = 0.7 * reach_lin + 0.3 * reach_near
    side_t = torch.maximum(_close_in(geo["d_thumb"], cup_r), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"], cup_r), torch.tanh(f_o / 0.5))
    reach_tips = high_fac * 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach_raw = 0.6 * reach_far + 0.4 * reach_tips

    # ---- A'. 잡는 높이로 먼저 내려오기
    side_gap = torch.clamp(rad_p - SIDE_K * cup_r, min=0.0)
    side_gate = LOW_SIDE_FLOOR + (1.0 - LOW_SIDE_FLOOR) * torch.exp(-(side_gap / (LOW_SIDE_K * cup_r)) ** 2)
    level_fac = LEVEL_FLOOR + (1.0 - LEVEL_FLOOR) * hand_level * normal_level
    level_fac = level_fac + (1.0 - level_fac) * lift_hold
    low_score = level_fac * palm_h_ok * side_gate

    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w * high_fac
    near_gate = _sat((2.0 * cup_r - geo["d_grip"]) / cup_r)
    release = torch.maximum(torch.maximum(_sat(in_hand / 0.5), near_gate), g_f)
    openness = _sat(1.0 - hand_closure)
    open_fac = release + (1.0 - release) * (0.45 + 0.55 * openness)
    approach = open_fac * (o_app * reach_raw + LOW_W * low_score)

    # ---- B. 닫기
    near_grip = torch.exp(-(geo["d_grip"] / (1.1 * cup_r)) ** 2)
    close = o_grip * pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # ---- B'. 손 안 감싸기
    tip_near = WRAP_CAP * torch.exp(-geo["d_each"] / (WRAP_K * cup_r))
    per_finger = torch.maximum(tip_near, touch_each)
    wrap_part = per_finger.mean(dim=1)
    in_gate = torch.maximum(_sat(in_hand / 0.5), g_f)
    wrap = in_gate * o_grip * pose_ok * wrap_part * wrap_part

    # ---- C. 파지
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    pre_q = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    post_q = torch.maximum((0.4 + 0.6 * pinch) * finger_fac, pre_q)
    contact_q = g_eff * post_q + (1.0 - g_eff) * pre_q
    grasp = o_grip * pose_ok * contact_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C'. 컵 좌표 파지 자세 유지
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    n_perp = torch.exp(-((n_dir * cup_up).sum(dim=-1) / HOLD_ANG_TOL) ** 2)
    f_perp = torch.exp(-((f_dir * cup_up).sum(dim=-1) / HOLD_ANG_TOL) ** 2)
    side_ok = torch.exp(-((rad_p - SIDE_K * cup_r) / (LOW_H_K * cup_r)) ** 2)
    hold_pose = 0.35 * n_perp + 0.35 * f_perp + 0.15 * palm_h_near + 0.15 * side_ok
    grip_pose = g_eff * pose_ok * finger_fac * hold_pose

    # ---- D. 들기 (iter_08 변경 2: 목표는 인자로 받은 여유 높이, 붓는 자리에서는 lift_relief 로 진척 유지)
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_eff * finger_fac * torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2) * calm
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin = lin + (1.0 - lin) * lift_relief
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- G. 교란
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    f_t_raw = f_raw[:, 0]
    f_o_raw = f_raw[:, 1:].sum(dim=1)
    f_o_max = f_raw[:, 1:].max(dim=1).values
    thumb_excess = torch.clamp(f_t_raw - f_o_raw - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o_max - f_t_raw, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / (3.5 * cup_r)) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "reach_raw": reach_raw, "close": close, "wrap": wrap, "grasp": grasp,
            "grip_pose": grip_pose, "g_eff": g_eff,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 4번째 사이클 iter_08 (full-joint 26, 팔 절대 목표).
    iter_07 c1r 위의 변경:
      1. 붓는 높이: 조준 창 안에서 기울이는 동안 주둥이 높이 차에 넓은 선형 받침(10 r)을 준다 -> aim 에 합산.
      2. 들기 높이: 소스 들기 목표 = 리시버 높이(0~리시버 목표) + 컵 높이 + r (기하). 절대 28 cm 목표 삭제.
         여유 높이 + 2 r 위로는 세금. 붓는 자리(창 안·기울임·주둥이가 입구 위)에서는 들기 진척을 유지한다.
      3. 관절 여유: 컵을 쥔 동안(g_eff) 두 팔 같은 식의 연속 비용. 이전 posture_margin(붓는 중 수입)을 대신한다.
      그 밖은 iter_07 c1r 그대로. 두 팔 같은 식·같은 무게. 항 25 개."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (iter_07 그대로) -------------------------------------------------------------------------
    W_APPROACH = 3.0
    W_CLOSE = 0.8
    W_WRAP = 1.2
    W_GRASP = 4.0
    W_GRIP_POSE = 2.5
    W_LIFT = 5.5
    W_RUSH = 0.06
    W_ALIVE = 0.5
    UPRIGHT_TOL = math.radians(20.0)
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25
    # iter_08 변경 2: 소스 여유 높이 위 세금 시작 +2 r, 폭 8 r (컵 반지름 배수)
    #   검사 [52]: 22 -> 28 cm 는 진척 포화(+0.02 이하) - 세금 0.07 -> +0.05 이하, 35.5 cm 는 세금으로 손해
    OVER_H_K_SRC, OVER_S_K_SRC = 2.0, 8.0
    # iter_08 변경 1: 붓는 높이 넓은 받침 (10 r = 28 cm 선형, 21 cm 이상에서도 기울기). 무게 4.0 x rot_prog
    #   54 도(rot_prog 0.48): 주둥이 21.3 -> 15 -> 10 -> 5.6 cm 칸별 약 +0.43·+0.34·+0.30 (추정, 들기 손실 없음 가정)
    W_POUR_H = 4.0
    POUR_H_WIDE_K = 10.0
    # iter_08 변경 3: 관절 여유 비용 무게 (팔당 최대 0.6). 한 관절 0.5 -> 0.03: 0.6 x tanh(0.77) = 0.39,
    #   두 관절(j6·j7): 0.6 x tanh(1.55) = 0.55. 붓는 중·기울이기 전 같은 값 (검사 [51] +0.3 / +0.2)
    W_MARGIN = 0.6

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    cup_h = mouth_z - float(ctx.cup_bottom_z)     # 컵 높이 (바닥 -> 림), 자산 속성에서 계산
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))

    src_g_raw = ctx.src_grasped.to(dt)
    rcv_g_raw = ctx.rcv_grasped.to(dt)
    both_g_raw = src_g_raw * rcv_g_raw
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) ----------------------------------------------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)

    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]

    # ---- iter_08 변경 2: 소스 들기 목표 = 리시버 입구를 넘을 여유 (두 컵 상대 기하) ------------------------
    #   리시버 높이는 0 ~ 리시버 들기 목표로 자른다 -> 리시버를 더 들어 소스 목표를 올리는 높이 경쟁 차단 (가설)
    clear_src = torch.clamp(h_rcv, 0.0, LIFT_T_RCV) + cup_h + r_cup       # 리시버 8 cm 에서 약 22 cm
    over_h_src = clear_src + OVER_H_K_SRC * r_cup
    over_s_src = OVER_S_K_SRC * r_cup
    # 붓는 자리 판정: 조준 창 안 x 기울임(0.5 LIM 에서 1) x 주둥이가 리시버 입구 위(-1 r 에서 0)
    #   -> 이 자리에서는 소스 들기 진척을 채워, 입구 바로 위로 내려와도 들기 수입이 줄지 않게 한다 (변경 1 과 짝)
    theta_pos = torch.clamp(theta, min=0.0)
    tilt_w = _sat(theta_pos / (0.5 * LIM))
    above_rim = _sat(1.0 + dz_lip / r_cup)
    lift_relief_src = lip_gate * tilt_w * above_rim

    # ---- 팔별 항 (두 팔 같은 식·같은 무게) -----------------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, pose_ok_src,
                          LIFT_T_RCV, clear_src, over_h_src, over_s_src, lift_relief_src)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, pose_ok_rcv,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV, zeros)

    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones

    # ---- A 접근 / B 닫기 / B' 감싸기 / C 파지 / C' 파지 자세 -------------------------------------------
    approach_src = W_APPROACH * src["approach"]
    approach_rcv = W_APPROACH * rcv["approach"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    finger_wrap = W_WRAP * (src["wrap"] + rcv["wrap"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
    grip_pose_src = W_GRIP_POSE * src["grip_pose"]
    grip_pose_rcv = W_GRIP_POSE * rcv["grip_pose"]

    # ---- D 들기 (소스: 여유 높이까지, 리시버 낮게) --------------------------------------------------
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- E 배치 ------------------------------------------------------------------------------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad
    place = converge_cups + pour_dir + rcv_keep

    # ---- E 조준 (운반 중, iter_07 그대로) --------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_carry = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 --------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F 회전 (8.0): iter_07 그대로 -----------------------------------------------------------------
    rot_prog = _sat(theta_pos / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    h_err = torch.abs(dz_lip - 0.25 * r_cup)
    aim_h = 1.0 - torch.tanh(h_err / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog * rcv_gate

    # ---- iter_08 변경 1: 붓는 높이 (창 안·기울이는 동안, 두 컵 상대 높이만) ---------------------------------
    #   회전 수입과 같은 게이트(want·lip_gate·rot_prog) -> 기울이지 않고 낮은 자리에 서 있기만 해서는 받지 못한다
    h_wide = _sat(1.0 - h_err / (POUR_H_WIDE_K * r_cup))
    pour_h = W_POUR_H * src_g * want * lip_gate * rcv_gate * rot_prog * h_wide
    aim = aim_carry + pour_h

    # ---- F 배달 (알당 30, 증분만): 주역 ------------------------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- iter_08 변경 3: 관절 여유 비용 (쥔 동안 운반·기울이기·붓기 전부, 두 팔 같은 식·같은 무게) -----------
    joint_margin = -W_MARGIN * (src_g * _limit_cost(ctx.src_arm_limit_margin)
                                + rcv_g * _limit_cost(ctx.rcv_arm_limit_margin))

    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- G 흘림 (벌점 게이트는 원 판정) ----------------------------------------------------------------
    post = torch.maximum(both_g_raw, src["lifted_w"])
    spill = -40.0 * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g_raw * pre_ramp * out_gate

    # ---- G 충돌 (실기 안전) -----------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g_raw) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g_raw) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                                + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- G 컵 교란: 두 팔 같은 무게 ---------------------------------------------------------------
    tilt_pen_src = (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = 1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    dist_src = 0.5 * src["slide"] + 0.3 * tilt_pen_src + 0.3 * src["one_sided"] \
        + 0.2 * src["palm_push"] + 0.3 * src["fling"] + W_RUSH * src["rush"]
    dist_rcv = 0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv + 0.3 * rcv["one_sided"] \
        + 0.2 * rcv["palm_push"] + 0.3 * rcv["fling"] + W_RUSH * rcv["rush"]
    cup_disturb = -(dist_src + dist_rcv)

    # ---- G 제어 정규화 (cup_grasp 수준). 팔 액션 크기 비용 없음 (a=0 = 시작 자세) ----------------------
    a = ctx.actions
    pa = ctx.prev_actions
    a_arm = torch.cat([a[:, 0:7], a[:, 13:20]], dim=-1)
    pa_arm = torch.cat([pa[:, 0:7], pa[:, 13:20]], dim=-1)
    a_hand = torch.cat([a[:, 7:13], a[:, 20:26]], dim=-1)
    pa_hand = torch.cat([pa[:, 7:13], pa[:, 20:26]], dim=-1)
    da_arm = a_arm - pa_arm
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (
        torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
        + torch.tanh(torch.clamp(qd_rcv.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0))
    ctrl_arm = -(rate_arm + speed_arm + runaway)
    da_hand = a_hand - pa_hand
    ctrl_hand = -0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "finger_wrap": finger_wrap,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grip_pose_src": grip_pose_src,
        "grip_pose_rcv": grip_pose_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "place": place,
        "aim": aim,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "joint_margin": joint_margin,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "spill": spill,
        "premature": premature,
        "collision": collision,
        "cup_disturb": cup_disturb,
        "ctrl_arm": ctrl_arm,
        "ctrl_hand": ctrl_hand,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
