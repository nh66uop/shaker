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
    """손끝-벽 거리 성형 (iter_04 형태, 척도를 컵 반지름 배수로: 1/30 m ~ 1.2 r, 1/160 m ~ 0.22 r).
    먼 척도 + 짧은 척도: 마지막 1 cm 도 기울기가 남는다."""
    return 0.5 * torch.exp(-d / (1.2 * r)) + 0.5 * torch.exp(-d / (0.22 * r))


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """컵 좌표(축 = cup_up)에서의 손끝 기하 (iter_04 구조).
    iter_03: 깔때기 '파지점 위 도착(ARRIVE_RISE)' 없음 -> 옆·잡는 높이로 도착.
    d_app = 집는 중심 -> 파지점 거리, 높이 차는 수평의 2 배 (브리프 5-1)."""
    EPS = 1e-6
    H_W = 2.0                           # 높이 차 가중 (위에서 내려오는 경로는 덜 받는다)
    grasp_h = 0.25 * cup_mouth_z        # 잡는 높이 (컵 원점 위)
    thumb_hi = 0.5 * cup_mouth_z        # 엄지 높이 대역 상한

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F)
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values

    # 엄지 대향: 엄지와 다른 손가락이 컵 축 반대편에 있으면 1
    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))

    # 집는 중심 -> 파지점 (옆·잡는 높이 도착, rise 없음)
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2 + 1e-12)
    d_app = torch.sqrt(rad_m ** 2 + (H_W * (h_m - grasp_h)) ** 2 + 1e-12)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.7 * tip_r)    # 0.02 m ~ 0.7 r
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_grip": d_grip, "d_app": d_app,
            "thumb_band": thumb_band}


def _envelope_score(palm_pos, palm_axes, cup_pos, cup_up):
    """K7 인벨롭 방향 점수 (iter_02 env_score 그대로): 법선 수평 + 손가락 축 수평 +
    (손바닥 -> 컵 축 수평 방향)이 0.67 x 법선 + 0.73 x 손가락 축에 가까울수록 1. 컵 좌표 기준이라 붓는 중에도 유지."""
    EPS = 1e-6
    ENV_N, ENV_F = 0.67, 0.73
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    ax_dot = (f_dir * cup_up).sum(dim=-1)
    hand_level = _sat(1.0 - ax_dot * ax_dot)                   # 수직 손(z -0.95) -> 약 0.1
    n_dot = (n_dir * cup_up).sum(dim=-1)
    normal_level = _sat(1.0 - n_dot * n_dot)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    tgt = ENV_N * n_dir + ENV_F * f_dir
    tgt_h = tgt - (tgt * cup_up).sum(dim=-1, keepdim=True) * cup_up
    tgt_hat = tgt_h / (tgt_h.norm(dim=-1, keepdim=True) + EPS)
    cos_env = (e_in * tgt_hat).sum(dim=-1)
    # 손끝 겨눔(cos 0.74) -> 약 0.35, 인벨롭 -> 1
    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    return hand_level * (0.25 * normal_level + 0.75 * dir_env)


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                    cup_r, cup_mouth_z, pose_ok, lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (두 팔 같은 구조).
    iter_03: 들기 전 = iter_04 손끝 기준 사다리 (접근 -> 닫기 -> 판정 뒤 파지) + K7 방향 인자(바닥 0.3)를
    접근·닫기·파지에 곱한다. 들기에는 걸지 않는다. 밀림 벌점은 속도형만."""
    ORIENT_FLOOR = 0.3
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f = torch.clamp(finger_force, min=0.0)
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)

    # ---- K7 방향 인자 (바닥 있음, 0 이 되지 않음 -> 곱 사다리 아님)
    env_score = _envelope_score(palm_pos, palm_axes, cup_pos, cup_up)
    orient_fac = ORIENT_FLOOR + (1.0 - ORIENT_FLOOR) * env_score

    # ---- A. 접근: 원거리 깔때기(원점 기준, 높이 2 배) + 손끝-벽 한쪽씩 합, 힘이 닿으면 그쪽 완성, 대향 가중
    d_app = geo["d_app"]
    reach_far = 0.5 * (1.0 - torch.tanh(d_app / (9.0 * cup_r))) \
        + 0.5 * (1.0 - torch.tanh(d_app / (2.5 * cup_r)))      # 9 r ~ iter_04 의 0.25 m, 시작 자세에서도 기울기
    side_t = torch.maximum(_close_in(geo["d_thumb"], cup_r), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"], cup_r), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach_raw = 0.4 * reach_far + 0.6 * reach_tips

    # 편 손 인자: 컵에서 떨어져 있는 동안만 (0.45 + 0.55 x 펴짐), 컵이 손가락 사이·파지점 근처·판정이면 해제
    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w
    near_gate = _sat((2.0 * cup_r - geo["d_grip"]) / cup_r)
    release = torch.maximum(torch.maximum(_sat(in_hand / 0.5), near_gate), g_f)
    openness = _sat(1.0 - hand_closure)
    open_fac = release + (1.0 - release) * (0.45 + 0.55 * openness)
    approach = orient_fac * open_fac * reach_raw

    # ---- B. 닫기: 파지점 근처 x 폐쇄 (0.7 에서 포화, 작은 무게)
    near_grip = torch.exp(-(geo["d_grip"] / (1.1 * cup_r)) ** 2)   # 0.03 m ~ 1.1 r
    close = orient_fac * pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # ---- C. 파지: 판정(엄지 AND 다른 손가락 > 1 N) 뒤 크게, 판정 전은 작은 사전 항
    #      경계(1 N / 1 N): 사전 0.355 -> 판정 0.75, 오르는 계단 (골 없음)
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    pre_q = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_q)
    grasp = orient_fac * pose_ok * contact_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- D. 들기: held = 판정 x 파지점 근처 x 컵 속도 (방향 인자 없음 -> 들기 경계 연속)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)
    t_up = torch.tanh(h_pos / 0.02)
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2) * calm   # 0.08 m ~ 3 r
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- G. 교란 (유계, 래치 없음). slide 는 미는 속도만 (밀린 상태 자체는 비용 없음).
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / (3.5 * cup_r)) ** 2)    # 0.10 m ~ 3.5 r
    rush = near * (1.0 - g_f) * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "reach_raw": reach_raw, "close": close, "grasp": grasp,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 4번째 사이클 iter_03 (full-joint 26, 팔 절대 목표).
    iter_02 위의 변경: 들기 전을 iter_04 손끝 기준 사다리로, K7 방향 인자를 접근·닫기·파지에,
    깔때기 옆 도착, 밀림 속도형만, 두 팔 같은 구조·무게, 팔 액션 크기 비용 제거. 붓기 이후는 iter_02 그대로. 항 22 개."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 -------------------------------------------------------------------------------------
    # 팔별 들기 전: 접근 1.0 + 닫기 0.5 + 파지 3.0. 벽에 닿고 닫힘(판정 전) 약 1.5 -> 파지(1.5 N, 판정) 약 4.1,
    # 한 팔만 바뀌어도 5/4 비 >= 1.6 (가설, 원장 미확인). 방향 인자가 셋 모두에 걸려 수직 손은 팔당 약 2.4 낮다.
    # 들기 전 항은 쥔 채로 유지되므로 들 때 사라지는 수입이 없다 (들기 빠른 성분 0.7 x 5.5 = 3.85 가 그대로 더해짐).
    W_APPROACH = 1.0
    W_CLOSE = 0.5          # 작은 무게 (iter_03 1항)
    W_GRASP = 3.0          # 판정 뒤 크게
    W_LIFT = 5.5
    W_RUSH = 0.1
    W_ALIVE = 0.5          # 상수: 경사를 만들지 않는다
    UPRIGHT_TOL = math.radians(20.0)
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25

    # ---- 공통 척도 ----------------------------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준, iter_02 그대로) ------------------------------------------------
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

    # ---- 팔별 항 (두 팔 같은 구조·무게) --------------------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, r_cup, mouth_z, pose_ok_src,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, r_cup, mouth_z, pose_ok_rcv,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 게이트 (바닥 있음): 소스의 들기 이후 수입에만
    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones

    # ---- A 접근 (상태형, 손끝 기준, 편 손, K7 방향 인자) -------------------------------------------
    approach_src = W_APPROACH * src["approach"]
    approach_rcv = W_APPROACH * rcv["approach"]

    # ---- B 닫기 / C 파지 --------------------------------------------------------------------------
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- D 들기 (소스 높게, 리시버 낮게) ------------------------------------------------------------
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- E 배치 (iter_02 그대로) --------------------------------------------------------------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad
    place = converge_cups + pour_dir + rcv_keep

    # ---- E 조준 (iter_02 그대로) --------------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 --------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F 회전 (8.0, iter_02 그대로): 창 안·조준 품질 x 회전 진척. 각도 처방 없음 ---------------------
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog * rcv_gate

    # ---- F 배달 (알당 30, 증분만): 주역 ------------------------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # 관절 여유: 붓는 중에만 작은 가산 (최대 0.3)
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # 마무리 (1.5 < 회전 수입 8)
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # 성공 보너스 (env 판정)
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- G 흘림: 두 컵 쥔 뒤 또는 소스 든 뒤 알당 40 ------------------------------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill = -40.0 * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # 창 밖 조기 기울임
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- G 충돌 (실기 안전, iter_02 그대로) ---------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- G 컵 교란: 두 팔 같은 무게 (역할별 가중 없음). 급한 팔(rush)도 여기에 ----------------------------
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
    ctrl_hand = -0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)   # 손 목표 변경은 싸게, 크기 비용 없음

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "place": place,
        "aim": aim,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "posture_margin": posture_margin,
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
