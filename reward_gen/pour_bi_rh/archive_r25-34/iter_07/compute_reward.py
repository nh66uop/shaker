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


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """cup_grasp 형태: 0 N -> 0, 0.1 N -> ~0.6, 수십 분의 1 N 위에서 포화 (힘 스파이크가 지배하지 못함)."""
    return torch.tanh(torch.clamp(force, min=0.0) / scale)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    rh2_i03 블록 재사용. 높이 오차(h_err_t, h_err_o)를 추가로 돌려준다 (접촉력 대체 경로의 높이 게이트용).
    iter_05 변경 2 유지: 깔때기 목표는 처음부터 잡는 높이다 (d_funnel == d_grip, ARRIVE_RISE 없음)."""
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
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

    # pinch centre: 목표는 잡는 높이의 컵 축 (위에서 도착하는 rise 없음)
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    d_funnel = d_grip
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_funnel": d_funnel, "d_grip": d_grip,
            "thumb_band": thumb_band, "h_err_t": h_err_t, "h_err_o": h_err_o}


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd,
                    cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                    cup_r, cup_mouth_z, cup_bottom_z, pose_ok, progress,
                    lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (iter_03 cup_grasp 골격 + iter_04 손가락 방향 + iter_05 수평 접근 + iter_06 접근 두 성분
    + iter_07 닫기 조건부 기울기·손바닥 밀려 들어감 비용).
    두 팔 사이 곱 없음. 들기 전 수입(reach·descend·close·finger_dir)은 held x 컵 높이로 꺼진다.
    cup_bottom_z 는 쓰지 않는다 (시그니처만 유지)."""
    EPS = 1e-6
    SIDE_K = 2.5       # 옆 파지 자세의 손바닥-컵 축 수평거리 = 컵 반지름의 약 2.5 배 (i03 ep5900 왼손 7.0 cm)
    H_W = 2.0          # 근거리 성분에서만 높이 오차를 수평 간격의 2 배로 센다 (iter_05 1, iter_06 에서 근거리로 한정)
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r)   # 접촉 반경 = ctx.cup_radius
    g_f = grasped.to(cup_pos.dtype)
    f = torch.clamp(finger_force, min=0.0)                     # (N,F)
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)

    # ---- 연속 held (cup_grasp 3 번 형태, RH 판): 엄지 x 다른 손가락, 손가락 수·손바닥 접촉이 오를수록 더
    c = _soft_contact(f)                                       # (N,F)
    c_t = c[:, 0]
    c_o = c[:, 1:].max(dim=1).values
    n_touch = c.sum(dim=1)
    palm_c = _soft_contact(palm_force)
    pinch_c = c_t * c_o                                        # 양쪽 접촉 (없으면 0: 한쪽 접촉은 held 아님)
    count_sat = torch.clamp((n_touch - 2.0) / 2.0, 0.0, 1.0)   # 2 지 -> 0, 4 지 -> 1 (K8: 소스 3 지)
    held = pinch_c * (0.6 + 0.4 * count_sat) * (0.85 + 0.15 * palm_c)
    firm = torch.tanh(torch.minimum(f_t, f_o) / 1.5)           # lifts had thumb ~2.3 N vs fingers ~2.7 N

    # ---- 높이 --------------------------------------------------------------------------------------
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)               # 5 mm dead band: tipping on the rim is not lifting
    t_up = torch.tanh(h_pos / 0.02)
    fade = 1.0 - held * t_up                                   # 들기 전 수입 끄기: 컵 높이 기준 (테이블 위 = 1)
    on_table = 1.0 - torch.tanh(h_pos / 0.04)
    late = torch.clamp((progress - 0.2) / 0.8, 0.0, 1.0)
    table_decay = 1.0 - 0.55 * late * on_table                 # cup_grasp 5 번: 후반 테이블 위 파지 수입 감쇠

    # ---- 손바닥의 컵 좌표 (컵 축 기준): 수평거리 rad_p, 높이 h_palm -----------------------------------
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up                            # 컵 축에 수직인 성분 (직립이면 수평)
    rad_p = to_axis.norm(dim=-1)                               # (N,) 손바닥-컵 축 수평거리
    h_palm = -h_c.squeeze(-1)                                  # (N,) 컵 origin 위 손바닥 높이 (컵 축 방향)
    side_d = SIDE_K * cup_r
    grasp_h = 0.25 * cup_mouth_z                               # 잡는 높이 (옆 파지 손바닥 높이 +1.3 cm 와 같은 척도)

    # ---- A. 접근 (iter_06 1): 원거리 3D 성분 + 근거리 '수평 간격 + 2 배 높이 오차' 성분의 합 (곱 아님) --------
    d3 = to_cup.norm(dim=-1)                                   # (N,) 손바닥-컵 origin 3D 거리
    side_d3 = (side_d ** 2 + grasp_h ** 2) ** 0.5              # 옆 파지 자세의 3D 거리 (ctx 치수에서 계산)
    gap3 = torch.clamp(d3 - side_d3, min=0.0)
    reach_far = 1.0 - torch.tanh(gap3 / (7.0 * cup_r))
    gap_xy = torch.clamp(rad_p - side_d, min=0.0)
    h_err_p = torch.abs(h_palm - grasp_h)
    d_eff = torch.sqrt(gap_xy ** 2 + (H_W * h_err_p) ** 2 + 1e-12)
    reach_near = 1.0 - torch.tanh(d_eff / (2.0 * cup_r))
    reach_raw = 0.6 * reach_far + 0.4 * reach_near             # 최대 1.0 (iter_05 와 같은 상한)
    reach = fade * reach_raw

    # ---- A1. 손가락 방향 (iter_04 2 + iter_05 4) ------------------------------------------------------
    f_dir = palm_axes[:, 3:6]
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    dir_dot = _sat((f_dir * e_in).sum(dim=-1))
    level = _sat(1.0 - torch.abs((f_dir * cup_up).sum(dim=-1)))
    near_w = torch.exp(-(d_eff / (2.0 * cup_r)) ** 2)
    finger_dir = fade * near_w * dir_dot * level

    # ---- A3. 입구 위 정지 비용 (iter_05 3): 손바닥이 컵 축에서 옆 파지 거리 안 AND 입구 높이 위. 파지 전만, 래치 없음, 유계 [0,1].
    inside_w = _sat((side_d - rad_p) / (0.5 * cup_r))
    above_w = _sat((h_palm - cup_mouth_z) / (0.1 * cup_mouth_z))
    hover = (1.0 - held) * inside_w * above_w

    # ---- A4. 손바닥 밀려 들어감 비용 (iter_07 3): 옆 파지 거리(2.5r) 안쪽 AND 입구 높이 아래(입구 위는 hover 가 맡는다).
    # 파지 전만: 연속 held 또는 grasped 가 서면 0 (쥔 뒤 0). 래치 없음, 유계 [0,1].
    # ep950 왼손: 수평 7.7 cm -> 3.0~3.4 cm 로 밀려 들어가며 손가락 방향 0.91 -> 0.19, 손끝-벽 3~8 cm.
    pre_grasp = 1.0 - torch.maximum(held, g_f)
    push_in = pre_grasp * inside_w * (1.0 - above_w)

    # ---- A2. 잡는 높이로 와서 양쪽에서 감싸기 (rh2_i03 기하, 깔때기 rise 없음). 접촉력 대체 경로는 높이 대역 안에서만
    funnel = 0.5 * (1.0 - torch.tanh(4.0 * geo["d_funnel"])) + 0.5 * torch.exp(-(geo["d_funnel"] / 0.03) ** 2)
    band_t = torch.exp(-(geo["h_err_t"] / 0.02) ** 2)          # (N,)
    band_o = torch.exp(-(geo["h_err_o"] / 0.02) ** 2)          # (N,F-1)
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t * band_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh((f[:, 1:] * band_o).sum(dim=1) / 0.5))
    tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    descend = fade * (0.4 * funnel + 0.6 * tips)

    # ---- 닫기 (iter_07 2, 재시도 [25]): 컵이 손가락 사이에 있을 때만, 닫힘이 오를수록 확실히 오른다 ------------
    # 조건(연속 가중, 곱): 대향(opp) x 두 손끝이 벽 가까이(d_thumb, d_oth; 높이 오차 포함 = 잡는 높이) x 집는 점이 컵 축 근처.
    # 재시도 [25]: ep950 왼손 실제 자세(수평 7.7 cm, 손끝-벽 0~1.5 cm, 손가락 컵 방향 0.91)에서 in_hand x touch_q 가
    #   약 0.22 로 작아 닫힘 0.7 에서 +0.22 에 그쳤다. 조건은 그대로 두고 척도만 푼다:
    #   - opp_w: 0.4 -> 0, 0.7 이상 1 (직각 배치 0.5 -> 0.33). 엄지가 반대편에 없으면 여전히 작다.
    #   - tip_w: 두 손끝 중 먼 쪽 기준 가우시안, 척도 1.5r. 1.5 cm -> 0.88, 3 cm -> 0.60, 5 cm -> 0.24, 8 cm -> 0.03
    #     (합 대신 max: 한 손끝만 붙고 다른 쪽이 멀면 지급하지 않는다).
    #   - grip_w: 척도 2r -> 3r (3 cm -> 0.88, 10 cm -> 0.24). 잡는 높이는 d_thumb·d_oth 의 높이 오차가 이미 본다.
    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w
    # 닫힘 0.7 에서 포화. 양쪽 접촉(pinch)이 성립하면 contact freeze 로 닫힘이 멈춰도 가득 지급
    clos_eff = torch.maximum(torch.clamp(hand_closure / 0.7, 0.0, 1.0), pinch_c)
    # 접촉 품질: 무접촉 0.7 < 한쪽 접촉 0.75 < 양쪽 동시(pinch) 1.0 -> 한쪽만 누르는 것보다 양쪽이 크게
    touch_q = 0.7 + 0.05 * (c_t + c_o) + 0.2 * pinch_c
    close = fade * table_decay * pose_ok * in_hand * clos_eff * touch_q   # 무가중 최대 1.0

    # ---- B. 파지: 성립 전 접촉부터 단계적 (한 손가락 < 양쪽 < 여러 손가락·단단히) ---------------------
    grasp_q = 0.2 * c_o + 0.3 * pinch_c + 0.5 * held * (0.4 + 0.6 * firm)
    grasp = table_decay * pose_ok * grasp_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C. 들기: 품질로 곱하지 않는다 (held x 직립만). 빠른 성분 0.7 x W_LIFT >= 들기 전 수입 합 -> 골짜기 없음
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)   # 과들기 세금: over_h 위에서만
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)              # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)           # "held and at least 5 cm up"

    # ---- 벌점 (무가중, 유계, 래치 없음) --------------------------------------------------------------
    v_lin = cup_lin_vel.norm(dim=-1)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "reach_raw": reach_raw, "finger_dir": finger_dir, "descend": descend,
            "close": close, "grasp": grasp, "hover": hover, "push_in": push_in,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기 iter_07. iter_06 위에 세 가지를 바꾼다 (이름 유지, 새 ctx 필드 없음):
    (1) 파지 전 흘림 벌점 제거 (비드당 3 -> 0). 두 컵 파지 또는 소스를 든 뒤 비드당 40 은 유지.
    (2) close_*: 컵이 손가락 사이(대향·손끝 벽 근처·잡는 높이)에 있을 때만 닫힘에 비례해 지급, pinch 가 한쪽 접촉보다 크게.
        W_CLOSE 0.3 -> 1.0. 재시도: 조건 척도 완화([25]), 들기 이득 확보를 위해 W_LIFT 5.5([16]).
    (3) palm_inside_*: 파지 전 손바닥이 옆 파지 거리(2.5r) 안쪽으로 밀려 들어가는 작은 비용.
    접근(원거리 3D + 근거리), 입구 위 비용, 깔때기, finger_dir, 붓기 부분(c5)은 iter_06 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 ------------------------------------------------------------------------------------
    # 들기 전 팔별 최대 = W_REACH 0.8 + W_FINGER_DIR 0.4 + W_DESCEND 0.7 + W_CLOSE 1.0 = 2.9
    #   < 들기 빠른 성분 0.7 x W_LIFT 5.5 = 3.85 -> 컵을 드는 순간 끊기는 수입보다 들기 수입이 늘 크다 (골짜기 없음).
    #   재시도 [16]: W_LIFT 4.5 (여유 0.25) 에서 테이블 파지 대비 들기 이득이 +0.30(3 cm)·+0.54(8 cm)로 작았다.
    #   5.5 로 여유 0.95 (iter_06 의 2.8 - 2.2 = 0.6 보다 큼). 3 cm 에서 held x (5.5 x 0.62 - 0.848 x 2.9) ~ +0.95 x held.
    # W_CLOSE 1.0: 조건을 다 채운 자세(in_hand ~1, 무접촉)에서 닫힘 0 -> 0.7 이 +0.7/스텝, pinch 면 +1.0/스텝.
    # W_HOVER 0.3: 입구 위 정지 비용. W_PUSH_IN 0.3: 입구 아래에서 옆 파지 거리 안쪽으로 밀려 들어가는 비용 (같은 크기, 작게).
    W_REACH, W_DESCEND, W_CLOSE, W_GRASP, W_LIFT = 0.8, 0.7, 1.0, 2.0, 5.5
    W_FINGER_DIR = 0.4
    W_HOVER = 0.3
    W_PUSH_IN = 0.3
    W_ALIVE = 0.5          # 상수: 경사를 만들지 않는다
    W_RCV_CROSS = 0.3      # 작은 비용: 리시버 파지 전 손바닥이 자기 컵보다 중심선 쪽으로 넘어감
    UPRIGHT_TOL = math.radians(20.0)   # 받는 컵 성공 한계 20 deg 와 같은 척도
    # 들기 목표·세금 높이 (높이 허용만 절대값 예외): 소스는 높게(K3), 리시버는 낮게
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25  # 바닥 있는 리시버 진행도 게이트 (소스의 들기 이후 수입에만)
    RCV_PEN_SCALE = 0.5    # 리시버 근처 벌점 축소 (rh2_i04 교훈)

    # ---- 공통 척도 (c5) ----------------------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (c5 원문) -------------------------------------------------------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)    # K6
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)

    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    # ---- A/B/C. 팔별 접근·손가락 방향·파지·들기 ----------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                          ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped, ctx.src_arm_qd,
                          ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_src, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                          ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                          ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_rcv, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 (reach -> 파지 -> 들기), 바닥 있는 계수. reach 는 끄기 전 값(들고 나서 진행도가 줄지 않게)
    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones
    reach_src = W_REACH * src["reach"]           # 소스·리시버 같은 크기로 독립 지급
    reach_rcv = W_REACH * rcv["reach"]
    finger_dir_src = W_FINGER_DIR * src["finger_dir"]   # 손가락이 수평으로 컵 축을 향할 때만
    finger_dir_rcv = W_FINGER_DIR * rcv["finger_dir"]
    descend_src = W_DESCEND * src["descend"]
    descend_rcv = W_DESCEND * rcv["descend"]
    close_src = W_CLOSE * src["close"]           # 컵이 손가락 사이일 때만 닫힘에 비례 (iter_07 2)
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
    # 입구 위 정지 비용: 두 손 같은 크기 (ep850 재생에서 두 손 모두 입구 위 +6.2 cm 에 정지)
    mouth_hover_src = -W_HOVER * src["hover"]
    mouth_hover_rcv = -W_HOVER * rcv["hover"]
    # 옆 파지 거리 안쪽으로 밀려 들어감 비용: 두 손 같은 크기, 파지 전만 (iter_07 3)
    palm_inside_src = -W_PUSH_IN * src["push_in"]
    palm_inside_rcv = -W_PUSH_IN * rcv["push_in"]
    # 소스: 리시버 목표 높이까지는 팔별(게이트 없음), 그 위 높이(K3 붓기용)·세금만 리시버 진행도 게이트
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    # 리시버 손바닥이 자기 컵보다 중심선 쪽(y 가 컵 y 보다 작은 쪽)으로 넘어가면 작은 비용 (1r 여유, 3r 에서 포화)
    cross_x = _sat((ctx.rcv_cup_pos[:, 1] - ctx.rcv_palm_pos[:, 1] - r_cup) / two_r)
    rcv_cross = -W_RCV_CROSS * (1.0 - rcv["held"]) * cross_x

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- C. 두 컵 수렴 (c5 1.2) --------------------------------------------------------------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- D. lip 조준 (c5 0.8) ----------------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 (c5: 전량 이송) -----------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F. 회전 구동 (c5 8.0) ---------------------------------------------------------------------
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog * rcv_gate

    # ---- F. 실제 도착한 비드 (c5 알당 30, 증분만) ---------------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 흘림 (iter_07 1): 파지 전 0, 두 컵을 쥔 뒤 또는 소스를 든 뒤 알당 40 ------------------------
    # 쥐기 전 소스 컵 건드림은 cup_disturb_*·contact_push 가 막는다 (파지 전 비드당 3 이 소스 컵 회피를 가르쳤다는 것은 [가설]).
    post = torch.maximum(both_g, src["lifted_w"])
    spill_w = 40.0 * post
    spill = -spill_w * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 마무리 (c5 1.5 < 회전 수입 8) -------------------------------------------------------------
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # ---- 성공 보너스 (env 판정, c5 20) -------------------------------------------------------------
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- 조기 기울임 (c5) ---------------------------------------------------------------------------
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- 실기 안전 (c5 / rh2_i03 크기) -------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 붓는 방향이 몸 바깥(+x)을 향하지 않게 (c5) -------------------------------------------------
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 리시버 컵: 직립, 중심선 기준 자기 쪽(+y) (c5) ----------------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- G. 관절 여유: 붓는 중에만 작은 가산 (c5 최대 0.3) -----------------------------------------
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # ---- 컵 교란 벌점 (넘어뜨림 방지용, 작게; 리시버 쪽 RCV_PEN_SCALE) -----------------------------
    tilt_pen_src = (1.0 - lip_gate) * torch.tanh(tilt_src / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    cup_disturb_src = -(0.5 * src["slide"] + 0.3 * tilt_pen_src)
    cup_disturb_rcv = -RCV_PEN_SCALE * (0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv)
    contact_push = -(0.3 * (src["one_sided"] + RCV_PEN_SCALE * rcv["one_sided"])
                     + 0.2 * (src["palm_push"] + RCV_PEN_SCALE * rcv["palm_push"]))
    rush_near_cup = -0.2 * (src["rush"] + RCV_PEN_SCALE * rcv["rush"])
    cup_fling = -0.3 * (src["fling"] + RCV_PEN_SCALE * rcv["fling"])

    # ---- 제어 정규화 (c5, 게이트 없음) --------------------------------------------------------------
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
        "finger_dir_src": finger_dir_src,
        "finger_dir_rcv": finger_dir_rcv,
        "mouth_hover_src": mouth_hover_src,
        "mouth_hover_rcv": mouth_hover_rcv,
        "palm_inside_src": palm_inside_src,
        "palm_inside_rcv": palm_inside_rcv,
        "descend_src": descend_src,
        "descend_rcv": descend_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "rcv_cross": rcv_cross,
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
