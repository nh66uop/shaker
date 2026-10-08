brief 에 가설 id 줄이 없어서 prompt 전체(iter_02 운영 노트 1~6 항)를 후보 하나로 구현했다.

규칙 충돌 처리(생성기 메모):
- 상태 수입 감쇠: 브리프 5-1 대 운영 노트 1 항(rh2_i03 가중 유지) — 최신 운영 노트를 따라 c5 의 state_decay 를 파지 블록에 곱하지 않았다. 5-1 의 "배달·회전 절반 이상"은 성공 보너스 구간 말고는 충족되지 않는다.
- 손·컵 치수 상수: TIP_R·깔때기 상수는 rh2_i03 값 유지, 붓기 기하는 ctx.cup_radius 배수(c5).

```python
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


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    rh2_i03 원문 그대로."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (0/336 knock-overs at +45 mm) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm, inside the 0-25 mm lift group)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin (no lift with thumb >= 50 mm)

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    # gap is zero only at the measured contact radius (tip frame 26-28 mm, cup outer 28.5 mm)
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)        # (N,)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values   # nearest non-thumb tip, (N,)

    # opposition: 1 when the thumb and the other fingers sit on opposite sides of the cup axis
    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # (N,)

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup_terms(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok, lift_target,
                    over_h, over_span):
    """팔별 무가중 항, 각 [0, 1]. rh2_i03 원문에서 두 가지만 바꿨다:
    (1) lift 에 과들기 세금(over_h 위에서만, 최대 절반) 추가, (2) lifted_w 를 반환(흘림 벌점 전환용)."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: 먼 깔때기 + 면별 손끝-벽. 접촉력이 있어야 그 면이 완성된다 (rh2_i01 엄지 0 N 주차 대책)
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: 성립 전 접촉도 단계적 (pre_grasp 0.355 -> 성립 0.75, 계단 상승이지 골짜기가 아님)
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

    # closure: 파지 자세에서만, 0.7 에서 포화 (측정 파지 폐쇄도 0.68-0.98, 주차 손 0.19)
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift: 실측 높이, 쥐고 있을 때만
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)                # 5 mm dead band: tipping on the rim is not lifting
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * torch.exp(-(geo["d_grip"] / 0.08) ** 2) * calm
    lift_prog = 0.5 * torch.tanh(h_pos / 0.02) + 0.5 * torch.clamp(h_pos / lift_target, max=1.0)
    # 과들기 세금: over_h 위에서만 작동, 최대 lift 의 절반 (c5 비율 0.15/0.3 과 같음)
    lift_prog = lift_prog - 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)
    lift = held * pose_ok * lift_prog
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)            # "held and at least 5 cm up"

    # penalties (unweighted, bounded)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)       # 1 N dead band: a slightly stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)             # finger-only contact: adding thumb force lowers this
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "lift": lift, "up_w": up_w,
            "lifted_w": lifted_w, "slide": slide, "one_sided": one_sided, "palm_push": palm_push,
            "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기 iter_02. 접근·파지·들기 = rh2_i03 블록(fresh 에서 파지를 배운 형태),
    붓기 = iter_01 c5 구조(두 컵을 쥐고 든 뒤). 소스의 들기 이후 수입은 리시버 진행도 계수(바닥 0.25)로 게이트."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (사다리: 접근 1 < 파지 2 < 들기 3, rh2_i03 자릿수 유지) -----------------
    W_REACH, W_CLOSE, W_GRASP, W_LIFT = 1.0, 0.3, 2.0, 3.0
    W_ALIVE = 0.5          # <= 1.0 (운영 노트 5): alive + reach(<=1) = 1.5 < 파지 자세 수입(reach+close+grasp >= 2.1)
    TIP_R = 0.027          # rh2_i03 측정값: 양면 접촉 시 손끝 프레임 반경 (엄지 27.9 mm, 가장 가까운 손가락 26.2 mm)
    UPRIGHT_TOL = math.radians(20.0)   # 받는 컵 성공 한계 20 deg 와 같은 척도
    # 들기 목표·세금 높이 (높이 허용만 절대값 예외, 5-3): 소스는 높게(K3, c5 0.28 / 세금 0.45),
    # 리시버는 낮게(rh2_i03 0.08 / c5 세금 0.15)
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25  # 곱 사다리 금지: 리시버가 멀어도 소스 들기 이후 수입의 1/4 은 남는다
    RCV_PEN_SCALE = 0.5    # 리시버 근처 벌점 합 최대 0.9 < 리시버 파지 자세 수입 2.1 이상 (rh2_i04 교훈)

    # ---- 공통 척도 (c5) ----------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # lip 수평 허가 반경
    wide = 10.0 * r_cup                      # WIDE 커널
    LIM = math.radians(81.0)                 # c5 premature 기준 스칼라
    active = float(max(int(ctx.num_beads), 1))

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (c5 원문: 부호 있는 기울기 theta, 붓는 lip) ----------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)                                   # pour_dir_xy
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])    # 받는 컵 쪽 기울기 (부호 있음)
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)    # K6: 기울이면 주둥이가 내려간다
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)

    # 컵 자세 허용: 소스는 받는 컵 위(lip 창 안)에서만 기울여도 된다, 리시버는 늘 직립
    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    # ---- A/B. 접근·파지·들기 (rh2_i03 블록) ------------------------------------------------
    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                          ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up,
                          ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_src, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                          ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                          ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_rcv, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 (reach -> 파지 -> 들기), 바닥 있는 계수 (운영 노트 2, rh2_i04)
    rcv_stage = _sat(0.4 * rcv["reach"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones
    reach_src = W_REACH * src["reach"]
    reach_rcv = W_REACH * rcv["reach"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
    lift_src = W_LIFT * src["lift"] * rcv_gate          # 리시버가 없으면 소스 들기 가치 1/4
    lift_rcv = W_LIFT * rcv["lift"]

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- C. 두 컵 수렴 (c5 1.2, WIDE 커널) ------------------------------------------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- D. lip 조준 (c5 0.8) ------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 (c5: 전량 이송) --------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F. 회전 구동 (c5 8.0): 0 deg 부터 선형, 1.4*LIM 에서 포화, 조준 바닥 0.3 ----------
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog * rcv_gate

    # ---- F. 실제 도착한 비드 (c5 알당 30, 증분만) -----------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 흘림: 파지 전 알당 3 (rh2_i03 W_SPILL 60 x 1/20), 두 컵을 쥔 뒤 또는 소스를 든 뒤 알당 40 (c5)
    post = torch.maximum(both_g, src["lifted_w"])
    spill_w = 3.0 + 37.0 * post
    spill = -spill_w * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 마무리 (c5 1.5 < 회전 수입 8) ----------------------------------------------------
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # ---- 성공 보너스 (env 판정, c5 20) ----------------------------------------------------
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- 조기 기울임 (c5: 벽 0.95~1.0 LIM 에서만, lip 이 2r 밖일 때만) --------------------
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- 실기 안전: 파지 전 rh2_i03 크기(1.0 / 손당 0.5), 두 컵 쥔 뒤 c5 크기(2.0 / 손당 1.0)
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 붓는 방향이 몸 바깥(+x)을 향하지 않게 (c5) ----------------------------------------
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 리시버 컵: 직립, 중심선 기준 자기 쪽(+y) (c5) -------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- G. 관절 여유: 붓는 중에만 작은 가산 (c5 최대 0.3) --------------------------------
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # ---- 컵 교란 벌점 (rh2_i03 가중; 리시버 쪽은 RCV_PEN_SCALE 로 축소) -------------------
    tilt_pen_src = (1.0 - lip_gate) * torch.tanh(tilt_src / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    cup_disturb_src = -(0.5 * src["slide"] + 0.3 * tilt_pen_src)
    cup_disturb_rcv = -RCV_PEN_SCALE * (0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv)
    contact_push = -(0.3 * (src["one_sided"] + RCV_PEN_SCALE * rcv["one_sided"])
                     + 0.2 * (src["palm_push"] + RCV_PEN_SCALE * rcv["palm_push"]))
    rush_near_cup = -0.2 * (src["rush"] + RCV_PEN_SCALE * rcv["rush"])
    cup_fling = -0.3 * (src["fling"] + RCV_PEN_SCALE * rcv["fling"])

    # ---- 제어 정규화 (c5, 게이트 없음: i21 교훈) -------------------------------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -0.1 * sat

    components = {
        "alive": alive,
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
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
        "cup_disturb_src": cup_disturb_src,
        "cup_disturb_rcv": cup_disturb_rcv,
        "contact_push": contact_push,
        "rush_near_cup": rush_near_cup,
        "cup_fling": cup_fling,
        "action_rate": action_rate,
        "action_flip": action_flip,
        "arm_tremor": arm_tremor,
        "action_reg_adr": action_reg_adr,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
```
