가설 id 줄이 brief 에 없다. 그래서 prompt 전체를 한 후보로 구현했다. 구현한 범위는 iter_05 "할 것" 1~4(prompt.md:725-732)다.

## 과제 해석과 단계

1. 두 손을 편 채 테이블과 수평을 맞추고, 각자 자기 컵 옆에서 다가간다(K7 인벨롭 자리: 손바닥이 컵 축에서 1.5 r, 높이는 원점 + 0.15 x 입구).
2. 손가락 사이에 컵이 들어오면 다섯 손가락을 모두 감아 쥔다. env 판정은 엄지와 다른 손가락 하나지만, 보상은 다섯 손가락 참여를 더 쳐 준다.
3. 두 컵을 든다. 소스는 높게, 리시버는 낮게 든다.
4. 소스 주둥이를 리시버 입구 위로 모으고 조준한다.
5. 소스 컵을 기울여 붓는다. 손은 컵과 함께 돌아 파지 자세(컵 좌표)를 유지한다.
6. 리시버는 20도 안으로 세워 둔다. 흘림과 충돌은 줄인다. 성공을 20 스텝 유지한다.

## 코드

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


def _close_in(d, r):
    """손끝-벽 거리 성형 (iter_03 c1r2 그대로, 척도는 컵 반지름 배수: 1.2 r, 0.22 r)."""
    return 0.5 * torch.exp(-d / (1.2 * r)) + 0.5 * torch.exp(-d / (0.22 * r))


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r, other_cup_pos):
    """컵 좌표(축 = cup_up)에서의 손끝 기하 (iter_03 c1r2 그대로)."""
    EPS = 1e-6
    H_W = 2.0
    grasp_h = 0.25 * cup_mouth_z
    thumb_hi = 0.5 * cup_mouth_z

    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)
    rad_vec = rel - h_tip.unsqueeze(-1) * up
    rad = rad_vec.norm(dim=-1)
    gap = torch.clamp(rad - tip_r, min=0.0)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values

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
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.7 * tip_r)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_grip": d_grip, "d_app": d_app,
            "thumb_band": thumb_band}


def _envelope_score(palm_pos, palm_axes, cup_pos, cup_up):
    """K7 인벨롭 방향 점수 (iter_04 그대로). 손가락 축 수평 인자(hand_level)도 함께 돌려준다."""
    EPS = 1e-6
    ENV_N, ENV_F = 0.67, 0.73
    NORMAL_TOL = 0.5
    NORMAL_PEN = 0.3
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    ax_dot = (f_dir * cup_up).sum(dim=-1)
    hand_level = _sat(1.0 - ax_dot * ax_dot)
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
    return _sat(score), hand_level


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos, other_cup_pos,
                    cup_r, cup_mouth_z, pose_ok, lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (두 팔 같은 식, 한 팔에만 붙는 항·부호·상수 없음).
    iter_05 변경 1: 다섯 손가락 참여 인자 finger_fac (엄지 외 손가락별 연속 접촉의 평균, 바닥 0.4)를
      판정 뒤 파지(contact_q)·파지 자세 유지(grip_pose)·들기 인정(held)에 곱한다.
      다섯 손가락 = 1.0, 엄지 + 한 손가락 = 약 0.55 (0 이 되지 않아 들기 경계 연속).
      one_sided 의 finger_excess 를 나머지 손가락 합이 아니라 가장 센 한 손가락으로 본다
      (합으로 보면 다섯 손가락 각 1.5 N 이 팔당 약 0.24 벌점을 받는다, iter_04 원장 차 0.48 과 같은 크기, 추정).
    iter_05 변경 2: grip_pose 를 컵 좌표 파지 자세로 (법선 ⟂ 컵 축, 손가락 축 ⟂ 컵 축,
      손바닥 높이·축 거리는 컵 축 기준). 들거나 기울여도 풀지 않는다 (lift_hold 해제 없음)."""
    ORIENT_FLOOR = 0.3
    APP_FLOOR = 0.5
    APP_LIN_K = 20.0
    APP_NEAR_K = 3.0
    PALM_H_K = 0.15       # K7: 손바닥 높이 = 컵 원점 + 0.15 x cup_mouth_z
    SIDE_K = 1.5          # K7: 손바닥-컵 축 1.5 r
    LOW_H_K = 1.0         # 높이·축 거리 허용 척도 1 r
    LOW_SIDE_K = 3.0
    LOW_SIDE_FLOOR = 0.3
    LEVEL_FLOOR = 0.4
    LOW_W = 0.4
    # iter_05 변경 1
    FINGER_F = 0.7        # 손가락별 접촉 척도 [N]: 1.5 N -> 0.97, 1 N(env 판정 문턱) -> 0.89
    FIVE_FLOOR = 0.4      # 참여 인자 바닥: 엄지 + 한 손가락 -> 0.4 + 0.6 x 0.25 = 0.55
    # iter_05 변경 2
    HOLD_ANG_TOL = 0.4    # 축 내적 척도 (무차원). K7 손가락 축 z -0.11 -> 0.93, f04 -0.57 -> 0.13, 45 도 0.71 -> 0.04

    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r, other_cup_pos)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f = torch.clamp(finger_force, min=0.0)
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)
    f_o_max = f[:, 1:].max(dim=1).values

    # ---- iter_05 변경 1: 다섯 손가락 참여 (손가락별 연속 접촉, 엄지 외 평균)
    part = torch.tanh(f[:, 1:] / FINGER_F).mean(dim=1)
    finger_fac = FIVE_FLOOR + (1.0 - FIVE_FLOOR) * part

    # ---- 높이
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)
    t_up = torch.tanh(h_pos / 0.02)
    lift_hold = g_f * t_up

    # ---- K7 방향 인자 (바닥 있음). 들린 뒤에는 1 로 풀림 -> 들기 경계 연속 (iter_04 그대로)
    env_score, hand_level = _envelope_score(palm_pos, palm_axes, cup_pos, cup_up)
    orient_fac = ORIENT_FLOOR + (1.0 - ORIENT_FLOOR) * env_score
    orient_app = APP_FLOOR + (1.0 - APP_FLOOR) * env_score
    o_grip = orient_fac + (1.0 - orient_fac) * lift_hold
    o_app = orient_app + (1.0 - orient_app) * lift_hold

    # ---- A. 접근 (iter_03 c1r2 그대로)
    d_app = geo["d_app"]
    reach_lin = _sat(1.0 - d_app / (APP_LIN_K * cup_r))
    reach_near = 1.0 - torch.tanh(d_app / (APP_NEAR_K * cup_r))
    reach_far = 0.7 * reach_lin + 0.3 * reach_near
    side_t = torch.maximum(_close_in(geo["d_thumb"], cup_r), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"], cup_r), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach_raw = 0.6 * reach_far + 0.4 * reach_tips

    # ---- A'. 잡는 높이로 내려오기 (손바닥 기준, 컵 좌표)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    rad_p = (to_cup - h_c * cup_up).norm(dim=-1)
    h_palm = -h_c.squeeze(-1)
    palm_h_err = torch.abs(h_palm - PALM_H_K * cup_mouth_z)
    palm_h_ok = torch.exp(-(palm_h_err / (LOW_H_K * cup_r)) ** 2)
    side_gap = torch.clamp(rad_p - SIDE_K * cup_r, min=0.0)
    side_gate = LOW_SIDE_FLOOR + (1.0 - LOW_SIDE_FLOOR) * torch.exp(-(side_gap / (LOW_SIDE_K * cup_r)) ** 2)
    level_fac = LEVEL_FLOOR + (1.0 - LEVEL_FLOOR) * hand_level
    level_fac = level_fac + (1.0 - level_fac) * lift_hold
    low_score = level_fac * palm_h_ok * side_gate

    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w
    near_gate = _sat((2.0 * cup_r - geo["d_grip"]) / cup_r)
    release = torch.maximum(torch.maximum(_sat(in_hand / 0.5), near_gate), g_f)
    openness = _sat(1.0 - hand_closure)
    open_fac = release + (1.0 - release) * (0.45 + 0.55 * openness)
    approach = open_fac * (o_app * reach_raw + LOW_W * low_score)

    # ---- B. 닫기 (그대로)
    near_grip = torch.exp(-(geo["d_grip"] / (1.1 * cup_r)) ** 2)
    close = o_grip * pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # ---- C. 파지: 판정 뒤 수입에 다섯 손가락 참여 인자 (iter_05 변경 1).
    #      판정 순간 사전 항보다 떨어지지 않게 max(판정 뒤, 사전) -> 오르는 계단 유지
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    pre_q = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    post_q = torch.maximum((0.4 + 0.6 * pinch) * finger_fac, pre_q)
    contact_q = torch.where(g, post_q, pre_q)
    grasp = o_grip * pose_ok * contact_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C'. iter_05 변경 2: 컵 좌표 파지 자세 유지 (들거나 기울여도 풀지 않는다) x 다섯 손가락 참여
    #      법선·손가락 축이 컵 축에 수직(0.35 + 0.35), 손바닥 높이 K7(0.15), 손바닥-컵 축 1.5 r(0.15). 합 항.
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    n_perp = torch.exp(-((n_dir * cup_up).sum(dim=-1) / HOLD_ANG_TOL) ** 2)
    f_perp = torch.exp(-((f_dir * cup_up).sum(dim=-1) / HOLD_ANG_TOL) ** 2)
    side_ok = torch.exp(-((rad_p - SIDE_K * cup_r) / (LOW_H_K * cup_r)) ** 2)
    hold_pose = 0.35 * n_perp + 0.35 * f_perp + 0.15 * palm_h_ok + 0.15 * side_ok
    grip_pose = g_f * pose_ok * finger_fac * hold_pose

    # ---- D. 들기: 들기 인정(held)에 다섯 손가락 참여 인자 (iter_05 변경 1). 방향 인자 없음
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * finger_fac * torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2) * calm
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- G. 교란 (유계, 래치 없음). iter_05: finger_excess 는 가장 센 한 손가락 기준
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o_max - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / (3.5 * cup_r)) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "reach_raw": reach_raw, "close": close, "grasp": grasp,
            "grip_pose": grip_pose,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 4번째 사이클 iter_05 (full-joint 26, 팔 절대 목표).
    iter_04 c1 위의 변경:
      1. 다섯 손가락 참여 인자를 판정 뒤 파지·파지 자세 유지·들기 인정에 곱한다 (바닥 0.4). one_sided 는 한 손가락 기준.
      2. 파지 자세 유지 항을 컵 좌표(법선·손가락 축 ⟂ 컵 축, 손바닥 높이·축 거리)로 바꾸고 들거나 기울여도 풀지 않는다.
      3. 소스 들기 높이(28 cm, 과하게 든 벌점 45 cm)는 그대로. 두 팔 같은 식·같은 무게. 그 밖은 iter_04 c1 그대로. 항 24 개."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 -------------------------------------------------------------------------------------
    W_APPROACH = 3.0
    W_CLOSE = 0.8
    W_GRASP = 4.0          # 판정 뒤 크게
    # iter_05: 파지 자세 유지 1.5 -> 2.5. 45 도에서 컵과 함께 돈 손(점수 약 1) 대 세운 채 남은 손
    # (컵 축 내적 약 0.5 씩 -> 점수 약 0.35)이면 이 항만으로 약 1.6 차 (추정, 원장 [42] 미확인).
    # 판정 순간 오르는 계단이고 들기·기울임에서 풀리지 않아 들기·붓기 쪽으로 가는 데 손해가 없다.
    W_GRIP_POSE = 2.5
    W_LIFT = 5.5
    W_RUSH = 0.06
    W_ALIVE = 0.5
    UPRIGHT_TOL = math.radians(20.0)
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20   # 사용자: 소스 들기 높이 그대로
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25

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

    # ---- 팔별 항 (두 팔 같은 식·같은 무게) -----------------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, pose_ok_src,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, pose_ok_rcv,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones

    # ---- A 접근 / B 닫기 / C 파지 / C' 파지 자세 ------------------------------------------------------
    approach_src = W_APPROACH * src["approach"]
    approach_rcv = W_APPROACH * rcv["approach"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
    grip_pose_src = W_GRIP_POSE * src["grip_pose"]
    grip_pose_rcv = W_GRIP_POSE * rcv["grip_pose"]

    # ---- D 들기 (소스 높게, 리시버 낮게) ------------------------------------------------------------
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
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

    # ---- E 조준 ------------------------------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 --------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F 회전 (8.0): 창 안·조준 품질 x 회전 진척. 각도 처방 없음 -----------------------------------
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

    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- G 흘림 ------------------------------------------------------------------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill = -40.0 * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- G 충돌 (실기 안전) -----------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
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
```

## 근거 메모

가설 id 없음 (brief 에 가설 줄이 없어 prompt 전체, 즉 iter_05 할 것 1~4 를 한 후보로 구현했다).

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| finger_fac (엄지 외 손가락별 tanh(f/0.7) 평균, 바닥 0.4) | 다섯 손가락 참여, 엄지 + 한 손가락은 덜 받되 0 이 아님 | prompt.md:726-727 |
| grasp 판정 뒤 수입 x finger_fac, max(판정 뒤, 사전 항) | 원장에서 두 손가락이 20.84, 다섯 손가락이 20.36 을 받았다. pinch 는 손가락 수와 무관하다. 판정 순간 계단은 내려가지 않는다 | prompt.md:720-722, 630 |
| held x finger_fac (lift_lo, lift_hi, up_w 로 전파) | 들기 인정도 다섯 손가락일 때 크게. 검사 [41] 의 3 cm 들기 | prompt.md:726-728 |
| one_sided 의 finger_excess 를 가장 센 한 손가락 기준으로 바꿈 | 다섯 손가락 각 1.5 N 이면 합 6 N 이라 팔당 0.3 x tanh(1.125) = 0.24, 두 팔 0.48 이 된다. 원장 차 0.48 과 같은 크기다 (추정. 원장 분해는 확인하지 않았다) | prompt.md:720, 378-380 |
| grip_pose = 판정 x pose_ok x finger_fac x (0.35 법선⟂축 + 0.35 손가락축⟂축 + 0.15 높이 K7 + 0.15 축 거리 1.5 r) | 기울이는 동안 파지 자세를 컵 좌표로 유지하고 풀지 않는다. 오른손 법선 z -0.67, 손바닥 축 2.79 r | prompt.md:717, 723, 729-730, 643 |
| W_GRIP_POSE 1.5 -> 2.5, HOLD_ANG_TOL 0.4 | 검사 [42]: 45도에서 컵과 함께 돈 손이 1.5 이상 더 받아야 한다 (이전 차 1.07). 이 항만으로 약 1.0~1.6 차 (추정) | prompt.md:721, 730 |
| 소스 들기 28 cm / 45 cm 그대로 | 사용자 결정 | prompt.md:714, 731 |
| 두 팔 같은 식·같은 무게, 그 밖은 iter_04 c1 그대로, 항 24 개 | [38] 미러 차 0, 항 25 개 이하 | prompt.md:732, 692 |

범위 밖 변경은 하나다. one_sided 수정은 "할 것" 목록에 적혀 있지 않다. 이전 코드의 합 기준 finger_excess 가 다섯 손가락 파지에만 벌점을 매겨 변경 1 과 정면으로 부딪히므로 변경 1 의 일부로 고쳤다. 원장 [41], [42] 수치는 확인하지 않았다(추정). W_GRIP_POSE 를 올렸기 때문에 붓는 중 상태 수입 비중이 조금 는다. 붓기 몫이 절반 이상이어야 한다는 조건(prompt.md:661)은 원장으로 확인해야 한다.

prompt 경로: /home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/iter_05/prompt.md