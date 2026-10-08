가설 지정 없음: brief 에 가설 id 줄이 없어서, 재시도 prompt 전체를 한 후보로 구현했습니다. prompt 안에도 Hypothesis A/B 표기가 없습니다. Review items(prompt_c1_repair.md:1169-1215)만 고쳤고 새 보상 component 는 더하지 않았습니다. 코드는 실행하지 않았고 검사기도 돌리지 않았으므로, 아래 수치는 모두 손으로 계산한 추정입니다.

```python
import torch
import math


def _sat(x, lo: float = 0.0, hi: float = 1.0):
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def _sabs(x: torch.Tensor, eps: float) -> torch.Tensor:
    """매끄러운 |x| (0 에서 꺾이지 않음)."""
    return torch.sqrt(x * x + eps * eps) - eps


def _close_in(d, r):
    """손끝-바깥 벽 거리 성형 (이전 코드 그대로, 척도는 컵 반지름 배수)."""
    return 0.5 * torch.exp(-d / (1.2 * r)) + 0.5 * torch.exp(-d / (0.22 * r))


def _limit_cost(margin: torch.Tensor) -> torch.Tensor:
    """관절 한계 근접 연속 비용 (관절 번호 없음, 유계 [0,1)). 이전 코드 그대로."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    return torch.tanh(c.sum(dim=-1))


def _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r):
    """컵 좌표(축 = cup_up) 손끝-바깥 벽 거리. 입구 안 손끝은 '먼 것'(2 r)으로 친다 (f07 교훈)."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)                                   # (N,F)
    rad = (rel - h_tip.unsqueeze(-1) * up).norm(dim=-1)              # (N,F)
    gap_raw = torch.clamp(rad - r, min=0.0)
    w_rad = _sat((0.9 * r - rad) / (0.3 * r))
    w_h = _sat((h_tip - (mouth_z - 2.0 * r)) / (0.5 * r)) * _sat((mouth_z + 1.5 * r - h_tip) / (0.5 * r))
    w_in = w_rad * w_h                                               # 1 = 입구 안
    gap = (1.0 - w_in) * gap_raw + w_in * (2.0 * r)
    h_t = h_tip[:, 0]
    h_err_t = torch.clamp(-h_t, min=0.0) + torch.clamp(h_t - 0.5 * mouth_z, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2 + 1e-12)      # (N,)
    h_o = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * mouth_z - h_o, min=0.0) + torch.clamp(h_o - 0.8 * mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2 + 1e-12)       # (N,F-1)
    return d_thumb, d_oth, 1.0 - w_in


def _hand_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos, other_cup_pos,
                r, mouth_z, lift_target, lift_relief):
    """팔별 무가중 항. 두 팔 같은 식 (한 팔에만 붙는 부호·상수 없음). 모든 거리 척도는 컵 반지름 배수."""
    EPS = 1e-6
    dt = cup_pos.dtype
    g_raw = grasped.to(dt)

    # ---- 손바닥 기하 (컵 좌표) : K7 자리 = 축에서 1.5 r, 높이 원점 + 0.15 x 입구
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    rad_vec = to_cup - h_c * cup_up
    rad_p = rad_vec.norm(dim=-1)
    dh = -h_c.squeeze(-1) - 0.15 * mouth_z
    dr = rad_p - 1.5 * r
    n_dot = (n_dir * cup_up).sum(dim=-1)
    f_dot = (f_dir * cup_up).sum(dim=-1)
    e_in = rad_vec / (rad_p.unsqueeze(-1) + EPS)
    tgt = 0.67 * n_dir + 0.73 * f_dir                                 # K7 손바닥 -> 컵 축 방향
    tgt_h = tgt - (tgt * cup_up).sum(dim=-1, keepdim=True) * cup_up
    tgt_hat = tgt_h / (tgt_h.norm(dim=-1, keepdim=True) + EPS)
    cos_env = (e_in * tgt_hat).sum(dim=-1)

    # ---- 손끝 기하 / 바깥 접촉 (입구 안 제외)
    d_thumb, d_oth, out_w = _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r)
    f_raw = torch.clamp(finger_force, min=0.0)
    f_out = f_raw * out_w
    touch = torch.tanh(f_out / 0.7)                                   # (N,F) 손가락별 접촉 점수
    touch_o = touch[:, 1:]
    oth_ok = _sat(touch_o.max(dim=1).values / math.tanh(1.0 / 0.7))
    g_eff = g_raw * out_w[:, 0] * oth_ok                              # 쥠 판정 x 바깥 접촉 (같은 단계 인자)

    # ---- 1 접근 거리 (L1: 수평 성분이 높이에 묻히지 않게 [24], 높이 x2 로 먼저 내려오기)
    #   컵보다 앞(+x)·상대 컵 쪽(중심선 쪽, [14])으로 나간 만큼 가산 - 쥐기 전에만 (쥘 때 수입은 오르기만 한다)
    far_side = torch.clamp(palm_pos[:, 0] - cup_pos[:, 0], min=0.0) * (1.0 - g_raw)
    to_other = other_cup_pos[:, :2] - cup_pos[:, :2]
    e_other = to_other / (to_other.norm(dim=-1, keepdim=True) + EPS)
    lat = ((palm_pos[:, :2] - cup_pos[:, :2]) * e_other).sum(dim=-1)
    cross = torch.clamp(lat - 0.5 * r, min=0.0) * (1.0 - g_raw)
    #   축 쪽으로 밀려 들어간 손바닥은 3 배로 친다 ([26])
    dr_eff = torch.clamp(dr, min=0.0) + 3.0 * torch.clamp(-dr, min=0.0)
    dh_abs = _sabs(dh, 0.1 * r)
    side_d = dr_eff + far_side + cross
    d_app = side_d + 2.0 * dh_abs
    #   먼 척도 30 r (시작 자세·25 cm 밖에서도 기울기) + 가까운 척도 2 r
    reach = 0.7 * _sat(1.0 - d_app / (30.0 * r)) + 0.3 * (1.0 - torch.tanh(d_app / (2.0 * r)))

    # ---- 손 방향 점수 (같은 단계 안의 곱): 수평(손가락 축 x 법선) x (기본 + 가까이서 K7 방위)
    #   넓은 받침(1 - |dot|) 을 섞어 수직 손에서도 기울기 ([34][35][49])
    s_fl = 0.5 * torch.exp(-(f_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(f_dot))
    s_nl = 0.5 * torch.exp(-(n_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(n_dot))
    level = s_fl * (0.4 + 0.6 * s_nl)
    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    near_w = torch.exp(-(d_app / (4.0 * r)) ** 2)
    orient = level * (0.4 + 0.6 * near_w * dir_env)

    # ---- 손 안 판정 (pocket): 손바닥이 축 3.2 r 안·잡는 높이 근처·K7 방위 쪽. 10 cm(약 3.6 r) 밖은 0 ([28])
    pocket = _sat((3.2 * r - rad_p) / r) * torch.exp(-(dh / (2.0 * r)) ** 2) * _sat((cos_env + 0.3) / 0.6)
    release = torch.maximum(pocket, g_raw)
    openness = _sat(1.0 - hand_closure)
    open_fac = release + (1.0 - release) * (0.6 + 0.4 * openness)    # 편 손 요구는 손 안이 아닐 때만
    approach = open_fac * (0.65 * reach + 0.35 * orient)

    # ---- 1' 먼저 내려오기: 잡는 높이 점수 (수평이 멀어도 바닥 0.6) ([23][24])
    low = (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 : 손끝별 바깥 벽 거리 볼록 평균(제곱), 자리·수평 손일수록 크게 + 손 안이면 닫기
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    wrap = part * part * (0.3 + 0.7 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)
    close = pocket * _sat(hand_closure / 0.7)

    # ---- 3 파지 : 판정 뒤 계단(>= 0.5) + 판정 전 접촉 사전 수입(<= 0.4)
    f_t = f_out[:, 0]
    f_o = f_out[:, 1:].sum(dim=1)
    pre = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    part_o = touch_o.mean(dim=1)
    post = 0.5 + part_o * part_o * part_o                             # 세제곱: 5지째 몫이 가장 크다 ([43])
    grasp = g_eff * post + (1.0 - g_eff) * pre

    # ---- 4 인벨롭 모양 점수 (가중 평균의 제곱, 컵 좌표라 기울여도 불변) + 손바닥 굴림(법선 기울기) 비용
    s_h = torch.exp(-(dh / (0.6 * r)) ** 2)
    s_r = torch.exp(-(dr / (0.5 * r)) ** 2)
    s_f = torch.exp(-(f_dot / 0.3) ** 2)
    s_n = torch.exp(-(n_dot / 0.3) ** 2)
    q = 0.30 * s_h + 0.25 * s_r + 0.30 * s_f + 0.15 * s_n
    q2 = q * q
    roll = _sat((torch.abs(n_dot) - 0.25) / 0.45)                     # 받침 자세(법선 z +0.72) -> 1 ([39])

    # ---- 5 들기 : 모양과 무관, 처음 몇 cm 가 가장 가파르고 목표 위는 이득 없음
    #   붓기 창 안에서 기울였으면 목표 높이 부족분을 면제 (내려와 붓는 것이 손해가 아니게, [50])
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.2 * r, min=0.0)
    lin = _sat(h_pos / lift_target)
    lin = lin + (1.0 - lin) * lift_relief
    prog = 0.7 * torch.tanh(h_pos / (0.7 * r)) + 0.3 * lin
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    lift = g_eff * calm * prog
    lifted_w = _sat(h_pos / (0.7 * r))

    # ---- 교란 (쥐기 전 밀림·쓰러뜨림은 속도로만, i09) + 손바닥을 컵 축 1.3 r 안으로 밀어 넣기 ([26])
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    topple = (1.0 - g_eff) * torch.tanh(cup_ang_vel[:, :2].norm(dim=-1) / 2.0)
    f_t_raw = f_raw[:, 0]
    f_o_raw = f_raw[:, 1:].sum(dim=1)
    f_o_max = f_raw[:, 1:].max(dim=1).values
    thumb_excess = torch.clamp(f_t_raw - f_o_raw - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o_max - f_t_raw, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    rush = near_w * (1.0 - g_raw) * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    jam = _sat((1.3 * r - rad_p) / (0.3 * r))
    disturb = 0.5 * slide + 0.3 * topple + 0.3 * one_sided + 0.2 * palm_push + 0.3 * fling \
        + 0.06 * rush + 0.6 * jam

    return {"approach": approach, "low": low, "wrap": wrap, "close": close, "grasp": grasp,
            "g_eff": g_eff, "q2": q2, "roll": roll, "prog": prog, "lift": lift,
            "lifted_w": lifted_w, "disturb": disturb}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 5번째 사이클 iter_00 재시도: 단계별 합 사다리 (곱 사다리 없음). 두 팔 같은 식·같은 무게."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 -----------------------------------------------------------------------------------
    W_ALIVE = 0.5         # 종료가 이득이 되지 않도록 작은 양의 기반
    W_APP = 2.0           # 팔당 접근 최대 2
    W_LOW = 1.2           # 먼저 내려오기: 내린 뒤 수평 진입이 대각선보다 스텝당 약 0.3 이득 ([23], 추정)
    W_WRAP = 1.2
    W_CLOSE = 0.8         # 손 안(pocket)에서만 닫기 수입: 0.12 -> 0.7 에 약 +0.5 ([25], 추정)
    W_GRASP = 3.0         # 판정 뒤 >= 1.5, 판정 전 <= 1.2 -> 파지 판정이 계단으로 이득; 5지 1.5 x 3 = 4.5
    W_SHAPE = 1.5         # 쥔 동안 인벨롭 모양 보너스 (테이블 위)
    W_SHAPE_LIFT = 5.0    # 모양 보너스가 들린 만큼 커짐 (더하는 항; 들기 기본 수입에는 곱하지 않음)
    W_ROLL = 1.5          # 쥔 채 손바닥 굴림(받침 자세) 비용, 팔당 최대 1.5 ([39])
    W_LIFT = 2.5          # 3 cm 이득 약 1.6 (모양 무관, [56] >= 1.4)
    W_AIM = 2.5
    W_POUR_H = 4.0        # 기울인 채 주둥이 높이: 21.3 -> 5.6 cm 합 약 +1.1 ([50], 추정)
    W_ROT = 23.0          # weight 23 with delivery 30/bead: 정적 수입 증가(파지 5지·내려오기·붓는 높이)와 짝, 회전+배달 몫 약 52 % (추정)
    W_DELIVER = 30.0      # 알당 30, 부호 있는 증분 (i20: 진행형 비대칭 clamp 금지)
    W_SPILL = 40.0
    W_SUCCESS = 20.0
    W_FINISH = 1.5
    W_MARGIN = 0.8        # 쥔 동안 관절 여유 비용, 팔당 최대 0.8 (유계, i15)
    UPRIGHT_TOL = math.radians(20.0)
    LIM = math.radians(81.0)

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    cup_h = mouth_z - float(ctx.cup_bottom_z)          # 컵 높이 (자산 속성에서 계산)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    active = float(max(int(ctx.num_beads), 1))
    LIFT_RCV = 3.0 * r_cup                             # 리시버는 낮게 (반지름 배수)

    src_g_raw = ctx.src_grasped.to(dt)
    rcv_g_raw = ctx.rcv_grasped.to(dt)
    both_g_raw = src_g_raw * rcv_g_raw
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) -----------------------------------------------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])     # 리시버 쪽으로 기운 부호 있는 각
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    above_rim = _sat(1.0 + dz_lip / r_cup)
    window = lip_gate * above_rim                       # 붓기 창 (같은 단계 인자)
    theta_pos = torch.clamp(theta, min=0.0)

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    clear_src = torch.clamp(h_rcv, 0.0, LIFT_RCV) + cup_h + r_cup      # 리시버 입구를 넘을 여유까지만
    #   창 안에서 기울였으면 들기 목표 부족분 면제 (주둥이를 입구 쪽으로 내려도 들기 수입이 줄지 않음)
    lift_relief_src = window * _sat(theta_pos / (0.5 * LIM))

    src = _hand_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                      ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped, ctx.src_arm_qd,
                      ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                      ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, clear_src, lift_relief_src)
    rcv = _hand_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                      ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                      ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                      ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, LIFT_RCV, zeros)

    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    alive = W_ALIVE * ones

    # ---- 1~3 접근(+먼저 내려오기) / 감싸기 전 / 파지 ------------------------------------------------------
    approach_src = W_APP * src["approach"] + W_LOW * src["low"]
    approach_rcv = W_APP * rcv["approach"] + W_LOW * rcv["low"]
    prewrap = W_WRAP * (src["wrap"] + rcv["wrap"]) + W_CLOSE * (src["close"] + rcv["close"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- 4 인벨롭 모양 보너스 (쥔 동안 더함) - 손바닥 굴림 비용 ----------------------------------------------
    envelope = src_g * (src["q2"] * (W_SHAPE + W_SHAPE_LIFT * src["prog"]) - W_ROLL * src["roll"]) \
        + rcv_g * (rcv["q2"] * (W_SHAPE + W_SHAPE_LIFT * rcv["prog"]) - W_ROLL * rcv["roll"])

    # ---- 5 들기 (모양 무관) ------------------------------------------------------------------------
    lift_src = W_LIFT * src["lift"]
    lift_rcv = W_LIFT * rcv["lift"]

    # ---- 6 조준 (두 컵이 들린 뒤, 주둥이 기준, 넓은 받침: 21 cm 위·12 cm 옆에서도 기울기) ----------------
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))
    h_err = torch.abs(dz_lip - 0.25 * r_cup)
    #   높이: 선형 몫 0.7 (12 r = 약 34 cm 받침) -> 21 cm 위에서도 cm 당 기울기가 일정 ([50])
    hz = 0.3 * (1.0 - torch.tanh(h_err / (1.5 * r_cup))) + 0.7 * _sat(1.0 - h_err / (12.0 * r_cup))
    xy = 0.5 * (1.0 - torch.tanh(lip_d / wide)) + 0.5 * (1.0 - torch.tanh(lip_d / two_r))
    aim_carry = W_AIM * both_g * lifted * (0.6 * xy + 0.4 * hz)

    # ---- 비드 장부 ----------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- 7 붓기: 창 안 회전 진척 (방향 있는 기울기) + 기울이는 동안 주둥이 높이 -------------------------
    rot_prog = _sat(theta_pos / (1.4 * LIM))            # 95 도까지 단조 증가
    pour_rot = W_ROT * both_g * want * window * rot_prog
    pour_h = W_POUR_H * both_g * want * window * rot_prog * hz
    aim = aim_carry + pour_h

    # ---- 7 배달 (주역, 부호 있는 증분: 들락날락 반복 지불 없음) ---------------------------------------------
    pour_delta = W_DELIVER * ctx.d_in_target * active

    # ---- 다 부으면 세우기 / 성공 --------------------------------------------------------------------------
    finish_upright = W_FINISH * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))
    success_bonus = W_SUCCESS * ctx.success.to(dt)

    # ---- 흘림 (영구, 소스를 쥐었거나 든 뒤) / 창 밖 조기 기울임 ------------------------------------------------
    post = torch.maximum(src_g_raw, src["lifted_w"])
    spill = -W_SPILL * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g_raw * pre_ramp * out_gate

    # ---- 충돌 (실기 안전, 유계) ----------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g_raw) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g_raw) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                                + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- 컵 교란: 속도형 + 쥔 컵의 기울기(소스는 붓기 창 밖에서만) --------------------------------------------
    tilt_pen_src = src_g * (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = rcv_g * (1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2))
    cup_disturb = -(src["disturb"] + rcv["disturb"] + 0.3 * tilt_pen_src + 0.3 * tilt_pen_rcv)

    # ---- 리시버 세움·중심선 쪽 ----------------------------------------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- 관절 여유 비용 (쥔 동안) -------------------------------------------------------------------
    joint_margin = -W_MARGIN * (src_g * _limit_cost(ctx.src_arm_limit_margin)
                                + rcv_g * _limit_cost(ctx.rcv_arm_limit_margin))

    # ---- 제어 비용 (cup_grasp 수준; 팔 액션 크기 비용 없음, 손 채널 포화 벌점 없음) ------------------------
    a = ctx.actions
    pa = ctx.prev_actions
    da_arm = torch.cat([a[:, 0:7] - pa[:, 0:7], a[:, 13:20] - pa[:, 13:20]], dim=-1)
    da_hand = torch.cat([a[:, 7:13] - pa[:, 7:13], a[:, 20:26] - pa[:, 20:26]], dim=-1)
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (
        torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
        + torch.tanh(torch.clamp(qd_rcv.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0))
    rate_hand = 0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)
    ctrl = -(rate_arm + speed_arm + runaway + rate_hand)

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "prewrap": prewrap,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "envelope": envelope,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "aim": aim,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "spill": spill,
        "premature": premature,
        "collision": collision,
        "cup_disturb": cup_disturb,
        "rcv_keep": rcv_keep,
        "joint_margin": joint_margin,
        "ctrl": ctrl,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
```

근거 메모 (prompt 파일: /home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/iter_00/prompt_c1_repair.md)

component 는 21개로 그대로입니다. 새로 생긴 몫은 모두 기존 component 안에 넣었습니다: 내려오기(low)는 approach_* 에, 굴림 비용(roll)은 envelope 에, 밀어넣기(jam)는 cup_disturb 에 들어 있습니다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| approach_*: d_app 를 L1 로 바꾸고 먼 척도 30 r, 편 손 바닥 0.6 | [24] 시작 자세에서 수평 6 cm 이득이 +0.022 로 부족. 수평 성분이 높이 x2 에 묻혀 있었음. 이제 약 +0.05 (추정) | :1174-1177, :774 |
| approach_*: cross (쥐기 전 상대 컵 쪽 가산) | [14] 리시버 손이 중심선 쪽 12 cm 에서 받는 보상이 바깥쪽 12 cm 와 같음 | :1169-1170 |
| approach_*: dr_eff (축 안쪽 3배) + cup_disturb 의 jam (1.3 r 안) | [26] 밀려 들어간 자세가 손가락 사이 자세보다 0.4 이상 손해가 아님 | :1181-1182 |
| approach_*: W_LOW 1.2 x low (잡는 높이 점수, 수평 바닥 0.6) | [23] 먼저 내리고 수평 진입하는 이득이 +5.1 로 부족(15 필요). 설계의 "먼저 내려오기". 스텝당 약 +0.3 (추정) | :1171-1173, :774, :746 |
| orient = level x (0.4 + 0.6 x 가까움 x K7 방위), level = 손가락 수평 x 법선 수평 | [34] 수직 손 비 0.37, [35] 인벨롭 대 손끝 겨눔·수직 손, [49] 손가락 축과 법선 수평 이득 부족. 곱은 같은 단계(방향) 안에서만 | :1189-1195, :1202-1213, :768 |
| open_fac 와 close 를 pocket 에 묶음 (손바닥 축 3.2 r 안 + 높이 + 방위) | [28]/[36] 컵 10 cm 앞에서 반쯤 쥔 손이 이득. 이전 release 가 5 r 안에서 완전히 풀렸음 | :1183-1186, :1196-1197 |
| W_CLOSE 0.8 x pocket x 닫힘 | [25] 손가락 사이 자세에서 0.7 까지 닫을 때 +0.3 미만. [57] 제자리 닫기 >= 0 은 유지 (추정) | :1178-1180, :823, :829 |
| wrap 에 (0.3 + 0.7 x 자리 x level) | [34] 수직 손이 입구 쪽 사전 수입을 받는 문제 | :1189-1190 |
| grasp post = 0.5 + part_o^3 | [43] 5지째 몫 1.43 이 2.5 미만. 세제곱이면 두 손 약 3.0 (추정) | :1200-1201, :776 |
| envelope 의 - W_ROLL x roll (쥔 동안) | [39] 받침 자세가 인벨롭보다 두 손 합 2.0 이상 싸지 않음 (f04 j5 굴림) | :1198-1199, :756 |
| lift prog 의 lift_relief (창 안에서 기울였을 때) | [50] 내려와 부을 때 들기 선형 몫이 줄어 손해 | :1214-1215, :778 |
| hz 선형 몫 0.7 (12 r), W_POUR_H 4.0 | [50] 21.3 -> 5.6 cm 합 +0.57 이 1.0 미만. 이제 약 +1.1 (추정) | :1214-1215, :779 |
| W_ROT 12 -> 23 | 정적 수입(5지 파지·low·pour_h)이 늘어도 붓는 상태 회전+배달 몫 >= 절반 유지 (추정 약 52%) | :786, :762 |
| 나머지 (lift, aim_carry xy, pour_delta, spill, premature, collision, rcv_keep, joint_margin, ctrl) | 이전 응답 그대로 | :1053-1134 |

확인되지 않은 점:
- **리뷰 항목 통과 여부는 추정입니다.** 검사 상태의 정확한 값을 모르기 때문입니다. 특히 [25]/[31]/[57] 은 "손가락 사이 자세"의 손바닥 축 거리를 약 1.3~2.5 r 로 가정했습니다.
- **[39] 는 한 손만 굴린 경우라면 차이가 경계(약 2.0~2.2)입니다.**
- **W_ROT 23 은 붓는 상태 정적 수입 약 42 에 대한 추정입니다.** 게이트 [1] 에서 다시 봐야 합니다.
- **[53]/[54] 는 q 가중치를 그대로 둬서 유지된다고 보았습니다.** 비스듬한 파지의 모양 점수는 늘지 않고 방향 차이만 커집니다.