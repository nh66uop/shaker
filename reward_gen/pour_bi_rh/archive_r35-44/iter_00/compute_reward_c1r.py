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
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """0 N -> 0, 0.1 N -> ~0.6, 수십 분의 1 N 위에서 포화 (힘 스파이크가 지배하지 못함)."""
    return torch.tanh(torch.clamp(force, min=0.0) / scale)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """컵 좌표(축 = cup_up)에서의 손끝 기하. 이전 코드와 같다.
    straddle = 컵 축이 엄지 손끝과 다른 손가락 손끝(평균) 사이에 끼는 배치 (거리와 무관하게 정의)."""
    EPS = 1e-6
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

    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # 엄지 대향 [0,1]

    a_t = rad_vec[:, 0]
    ab = o_vec - a_t
    ab2 = (ab * ab).sum(dim=-1)
    t_par = -(a_t * ab).sum(dim=-1) / (ab2 + EPS)
    width_ok = _sat(torch.sqrt(ab2 + 1e-12) / (2.0 * tip_r))
    straddle = _band(t_par, 0.25, 0.75, 0.15) * width_ok

    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_funnel": d_grip, "d_grip": d_grip,
            "thumb_band": thumb_band, "h_err_t": h_err_t, "h_err_o": h_err_o, "straddle": straddle}


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                    cup_r, cup_mouth_z, pose_ok, cup_upright, progress,
                    lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항.
    iter_00 (재편 두 번째 판) + 재시도 변경:
      - 접근 유효거리: L2 -> L1 (수평 틈 + 2 x 높이 오차). L2 에서는 높이 오차가 클 때 수평 기울기가 사라졌다([24] 수평 6cm +0.006).
      - 접근 먼 척도: 1 - tanh -> 선형 (18 r 까지 일정한 기울기). 가까운 척도 3 r tanh 는 유지.
      - 편 손 인자 해제: in_hand 외에 손바닥이 옆 목표(2.5 r) 안이면 해제 ([25] 닫기 수입 상쇄 방지).
      - push_in 은 손바닥이 2 r 안으로 들어올 때만 ([15] 6 cm 옆에서 비용이 접근 수입을 깎음).
      - 닫기 touch_q 바닥 0.7 -> 0.85 (최대 1.0 유지)."""
    EPS = 1e-6
    SIDE_K = 2.5          # 손바닥 목표 = 컵 축에서 2.5 r 옆
    H_W = 2.0             # 높이 차는 수평의 2 배로 센다 (브리프 5-1)
    APP_LIN_K = 18.0      # 접근 선형 척도 18 r: 시작 자세(L1 유효거리 약 13 r)에서도 일정한 기울기
    APP_NEAR_K = 3.0      # 접근 가까운 척도 3 r: 15 cm -> 6 cm 구간에서 수입이 가파르게 오름
    FAR_K = 4.0
    NEAR_K = 1.5
    FUNNEL_FAR_K = 2.0
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r)
    g_f = grasped.to(cup_pos.dtype)
    f = torch.clamp(finger_force, min=0.0)
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)

    # ---- 연속 held: 엄지 x 다른 손가락 (한쪽 접촉은 held 아님)
    c = _soft_contact(f)
    c_t = c[:, 0]
    c_o = c[:, 1:].max(dim=1).values
    n_touch = c.sum(dim=1)
    palm_c = _soft_contact(palm_force)
    pinch_c = c_t * c_o
    count_sat = torch.clamp((n_touch - 2.0) / 2.0, 0.0, 1.0)
    held = pinch_c * (0.6 + 0.4 * count_sat) * (0.85 + 0.15 * palm_c)
    firm = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    pre_grasp = 1.0 - torch.maximum(held, g_f)

    # ---- 높이
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)
    t_up = torch.tanh(h_pos / 0.02)
    fade = 1.0 - held * t_up                                   # 들기 전 수입 끄기 (lift 와 같은 held x t_up)
    on_table = 1.0 - torch.tanh(h_pos / 0.04)
    late = torch.clamp((progress - 0.2) / 0.8, 0.0, 1.0)
    table_decay = 1.0 - 0.55 * late * on_table

    # ---- 손바닥의 컵 좌표 (원점 기준)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up
    rad_p = to_axis.norm(dim=-1)
    h_palm = -h_c.squeeze(-1)
    side_d = SIDE_K * cup_r
    grasp_h = 0.25 * cup_mouth_z

    # ---- 원점 기준 유효 거리 (손가락 방향·리시버 게이트용 L2, 이전과 같음)
    d3 = to_cup.norm(dim=-1)
    gap_xy = torch.clamp(rad_p - side_d, min=0.0)
    h_err_p = torch.abs(h_palm - grasp_h)
    d_eff = torch.sqrt(gap_xy ** 2 + (H_W * h_err_p) ** 2 + 1e-12)

    # ---- reach_raw: 지급하지 않음 (리시버 진행도 게이트 전용, 이전 코드와 같음)
    side_d3 = (side_d ** 2 + grasp_h ** 2) ** 0.5
    gap3 = torch.clamp(d3 - side_d3, min=0.0)
    reach_far = 1.0 - torch.tanh(gap3 / (7.0 * cup_r))
    reach_near = 1.0 - torch.tanh(d_eff / (2.0 * cup_r))
    reach_raw = 0.6 * reach_far + 0.4 * reach_near

    # ---- 컵이 손가락 사이에 있는 정도 (편 손 인자 해제 + 닫기 게이트)
    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w

    # ---- A. 접근 + 편 손: 상태형 (브리프 5-0, 5-1). 파지가 서면 pre_grasp 로 사라진다.
    # L1 유효거리: 수평 틈과 높이 오차(x2)가 각자 기울기를 가진다 (높이가 커도 수평 6 cm 가 값을 가짐).
    d_app = gap_xy + H_W * h_err_p
    app_lin = _sat(1.0 - d_app / (APP_LIN_K * cup_r))          # 먼 척도: 일정 기울기
    app_near = 1.0 - torch.tanh(d_app / (APP_NEAR_K * cup_r))  # 가까운 척도
    app_state = 0.6 * app_lin + 0.4 * app_near
    openness = _sat(1.0 - hand_closure)
    # 해제: 컵이 손가락 사이(in_hand) 이거나 손바닥이 옆 목표(2.5 r) 안이면 편 손 인자를 푼다.
    #   2.5 r + 1 r 밖(12 cm 옆 1b, 10 cm 비교)에서는 인자가 그대로 -> 편 손 > 쥔 손.
    near_gate = _sat((side_d + cup_r - rad_p) / cup_r)
    release = torch.maximum(_sat(in_hand / 0.5), near_gate)
    open_fac = release + (1.0 - release) * (0.45 + 0.55 * openness)
    approach = pre_grasp * cup_upright * app_state * open_fac

    # ---- B. 손가락 방향 (근거리 0.7 + 원거리 0.3, 곱 사슬 아닌 합)
    f_dir = palm_axes[:, 3:6]
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    dir_dot = _sat((f_dir * e_in).sum(dim=-1))
    level = _sat(1.0 - torch.abs((f_dir * cup_up).sum(dim=-1)))
    near_w = torch.exp(-(d_eff / (NEAR_K * cup_r)) ** 2)
    near_part = near_w * dir_dot * level
    far_w = 1.0 - torch.tanh(d_eff / (FAR_K * cup_r))
    far_part = far_w * (0.5 * dir_dot + 0.5 * geo["straddle"])
    finger_dir = fade * (0.7 * near_part + 0.3 * far_part)

    # ---- B 비용: 입구 위 정지 / 손바닥 밀려 들어감
    inside_w = _sat((side_d - rad_p) / (0.5 * cup_r))
    above_w = _sat((h_palm - cup_mouth_z) / (0.1 * cup_mouth_z))
    hover = (1.0 - held) * inside_w * above_w
    # push_in: 손바닥이 컵 축에서 2 r 안(벽에 거의 붙음)일 때만. 옆 목표(2.5 r) 부근 접근은 비용 없음.
    push_w = _sat((2.0 * cup_r - rad_p) / (0.5 * cup_r))
    push_in = pre_grasp * push_w * (1.0 - above_w)

    # ---- B. 잡는 높이로 와서 양쪽에서 감싸기
    d_fn = geo["d_funnel"]
    funnel = 0.5 * (1.0 - torch.tanh(d_fn / (FUNNEL_FAR_K * cup_r))) + 0.5 * torch.exp(-(d_fn / 0.03) ** 2)
    band_t = torch.exp(-(geo["h_err_t"] / 0.02) ** 2)
    band_o = torch.exp(-(geo["h_err_o"] / 0.02) ** 2)
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t * band_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh((f[:, 1:] * band_o).sum(dim=1) / 0.5))
    tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    descend = fade * (0.3 * funnel + 0.7 * tips)

    # ---- C. 닫기: 컵이 손가락 사이에 있을 때만, 폐쇄도·손끝 접촉에 지불
    # touch_q 바닥 0.85 (최대 1.0 은 이전과 같음): 접촉 전에도 닫는 기울기가 충분하게.
    clos_eff = torch.maximum(torch.clamp(hand_closure / 0.7, 0.0, 1.0), pinch_c)
    touch_q = 0.85 + 0.05 * (c_t + c_o) + 0.05 * pinch_c
    close = fade * table_decay * pose_ok * in_hand * clos_eff * touch_q

    # ---- C. 파지: 한 손가락 < 양쪽 < 여러 손가락·단단히
    grasp_q = 0.2 * c_o + 0.3 * pinch_c + 0.5 * held * (0.4 + 0.6 * firm)
    grasp = table_decay * pose_ok * grasp_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- D. 들기 (held x 자세만, 과들기 세금은 over_h 위에서만)
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- G. 교란 (무가중, 유계, 래치 없음). slide 는 미는 속도(차분)만 (브리프 5-3 유지).
    v_lin = cup_lin_vel.norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    # 급한 팔: 관절 속도 rms / 1 rad/s
    rush = near * (1.0 - g_f) * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "reach_raw": reach_raw, "finger_dir": finger_dir,
            "descend": descend, "close": close, "grasp": grasp, "hover": hover, "push_in": push_in,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 4번째 사이클 iter_00 재편 두 번째 판 재시도 (full-joint 26 = 팔마다 [팔 증분 7, 손 절대 목표 6]).
    이전 응답 위의 변경 (Review items [15]·[24]·[25]):
    1. 접근: L1 유효거리 + 선형 먼 척도(18 r) + 가까운 척도(3 r), W 0.5 -> 0.7.
    2. 편 손 인자: 손바닥이 옆 목표(2.5 r) 안이거나 컵이 손가락 사이면 해제.
    3. push_in 은 손바닥 2 r 안에서만. 닫기 touch_q 바닥 0.85.
    4. 들기 전 상태 합 = 0.7 + 0.4 + 1.6 + 1.0 = 3.7 < 들기 빠른 성분 0.7 x 5.5 = 3.85 (골 없음). 항 25 개."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 ------------------------------------------------------------------------------------
    # 들기 전 팔별 상태형 최대 = W_APPROACH 0.7 + W_FINGER_DIR 0.4 + W_DESCEND 1.6 + W_CLOSE 1.0 = 3.7
    #   < 들기 빠른 성분 0.7 x W_LIFT 5.5 = 3.85 -> 들기로 넘어갈 때 골짜기 없음.
    #   approach 는 파지 때 사라지고 grasp(W 2.0) 가 이어받는다.
    W_DESCEND, W_CLOSE, W_GRASP, W_LIFT = 1.6, 1.0, 2.0, 5.5
    W_APPROACH = 0.7       # 상태형 접근: 시작 자세 수평 6 cm 에서 약 +0.05 (선형 척도 0.6 x 0.06/18r x 0.7)
    W_FINGER_DIR = 0.4
    W_HOVER = 0.2          # 컵 근처 비용 < 접근 수입 (브리프 5-1)
    W_PUSH_IN = 0.15
    W_RUSH = 0.1
    W_ALIVE = 0.5          # 상수: 경사를 만들지 않는다
    W_RCV_CROSS = 0.3
    UPRIGHT_TOL = math.radians(20.0)   # 받는 컵 성공 한계 20 deg 와 같은 척도
    # 들기 목표·세금 높이 (높이 허용만 절대값 예외, 이전 코드 유지): 소스는 높게(K3), 리시버는 낮게
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25
    RCV_PEN_SCALE = 0.5

    # ---- 공통 척도 ----------------------------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
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
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)   # 컵 반지름 창

    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    # ---- 팔별 항 ------------------------------------------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, r_cup, ctx.cup_mouth_z,
                          pose_ok_src, upright_src, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, r_cup, ctx.cup_mouth_z,
                          pose_ok_rcv, upright_rcv, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 게이트 (바닥 있음): 소스의 들기 이후 수입에만
    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones

    # ---- A 접근: 상태형 (원점 기준 L1 거리 x 편 손 인자, 파지 후 소멸) ---------------------------------
    approach_src = W_APPROACH * src["approach"]
    approach_rcv = W_APPROACH * rcv["approach"]

    # ---- B 입구: 손가락 방향(원거리부터 연속) + 잡는 높이로 내려와 감싸기 ------------------------------
    mouth_src = W_FINGER_DIR * src["finger_dir"] + W_DESCEND * src["descend"]
    mouth_rcv = W_FINGER_DIR * rcv["finger_dir"] + W_DESCEND * rcv["descend"]

    # B 비용 하나로: 입구 위 정지 + 손바닥 밀려 들어감 + 컵 근처 급한 팔 + 리시버 손바닥 중심선 넘어감
    cross_x = _sat((ctx.rcv_cup_pos[:, 1] - ctx.rcv_palm_pos[:, 1] - r_cup) / two_r)
    mouth_cost = -(W_HOVER * (src["hover"] + rcv["hover"])
                   + W_PUSH_IN * (src["push_in"] + rcv["push_in"])
                   + W_RUSH * (src["rush"] + RCV_PEN_SCALE * rcv["rush"])
                   + W_RCV_CROSS * (1.0 - rcv["held"]) * cross_x)

    # ---- C 파지: 닫기(W 1.0) -> 양쪽 접촉 파지(W 2.0 이 이어받음) ------------------------------------
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- D 들기: 소스는 리시버 목표 높이 위(K3 붓기용)·세금만 리시버 진행도 게이트 ---------------------
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- E 배치: 두 컵 수렴 (1.2) + 붓는 방향 (-1.5) + 리시버 직립·자기 쪽 (0.5 / -1.5) -------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad
    place = converge_cups + pour_dir + rcv_keep

    # ---- E 조준: 소스 주둥이 lip 을 입구 위 창 안에 (0.8) ----------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 --------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F 회전 (8.0): 창 안·조준 품질 x 회전 진척. 각도 처방 없음 ------------------------------------
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

    # 관절 여유: 붓는 중에만 작은 가산 (최대 0.3, 강한 직접 벌점 아님)
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # 마무리 (1.5 < 회전 수입 8)
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # 성공 보너스 (env 판정)
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- G 흘림: 파지 전 0, 두 컵 쥔 뒤 또는 소스 든 뒤 알당 40 ----------------------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill = -40.0 * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # 창 밖 조기 기울임
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- G 충돌 (실기 안전): 컵-컵 + 포개짐 + 남의 것 건드리기 ----------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- G 컵 교란 (넘어뜨림 방지, 리시버 쪽 RCV_PEN_SCALE) --------------------------------------------
    # 기울기 벌점은 1 - exp(-(tilt/20deg)^2): 작은 기울기(5 deg -> 0.06)는 싸고 넘어짐(90 deg)은 1.
    tilt_pen_src = (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = 1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    dist_src = 0.5 * src["slide"] + 0.3 * tilt_pen_src + 0.3 * src["one_sided"] \
        + 0.2 * src["palm_push"] + 0.3 * src["fling"]
    dist_rcv = 0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv + 0.3 * rcv["one_sided"] \
        + 0.2 * rcv["palm_push"] + 0.3 * rcv["fling"]
    cup_disturb = -(dist_src + RCV_PEN_SCALE * dist_rcv)

    # ---- G 제어 정규화 (cup_grasp 수준, 브리프 5-2, 이전 응답과 같음) ------------------------------------
    # 팔 채널 = [0:7] (소스), [13:20] (리시버): 관절 증분. 손 채널 = [7:13], [20:26]: 절대 목표 (|a|~1 은 정상, 크기 벌점 없음).
    # 반전(flip)·|a|>0.9 포화 벌점은 관절 증분의 탐색을 억누르므로 두지 않는다.
    a = ctx.actions
    pa = ctx.prev_actions
    a_arm = torch.cat([a[:, 0:7], a[:, 13:20]], dim=-1)
    pa_arm = torch.cat([pa[:, 0:7], pa[:, 13:20]], dim=-1)
    a_hand = torch.cat([a[:, 7:13], a[:, 20:26]], dim=-1)
    pa_hand = torch.cat([pa[:, 7:13], pa[:, 20:26]], dim=-1)
    da_arm = a_arm - pa_arm
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    mag_arm = 0.01 * (a_arm * a_arm).mean(dim=-1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (
        torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
        + torch.tanh(torch.clamp(qd_rcv.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0))
    ctrl_arm = -(rate_arm + mag_arm + speed_arm + runaway)
    da_hand = a_hand - pa_hand
    ctrl_hand = -0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)   # 손 목표 변경은 싸게

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "mouth_src": mouth_src,
        "mouth_rcv": mouth_rcv,
        "mouth_cost": mouth_cost,
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
