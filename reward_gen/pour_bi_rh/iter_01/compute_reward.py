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
    """컵 좌표(축 = cup_up) 손끝-바깥 벽 거리. 입구 안 손끝은 '먼 것'(2 r)으로 친다 (f07 교훈).
    재시도 2 [T1]: 입구 안 판정 높이 하한을 (입구 - 2 r) -> (입구 - 1 r) 로 (_surround 와 같은 값).
    (가설) 입구 - 2 r 은 컵 원점 부근이라, 잡는 높이(원점 + 0.15 x 입구)에서 벽 안쪽으로 찍힌 재생 손끝을
    '입구 안 = 먼 것'으로 바꿔 wrap 과 바깥 접촉을 한 스텝에 떨어뜨렸다 (T1 t=386, prewrap -0.545, grasp -0.009).
    f07 엄지 걸기(엄지 4.7 cm, 입구 - 1 r = 2.6 cm 위)는 그대로 입구 안으로 잡힌다."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)                                   # (N,F)
    rad = (rel - h_tip.unsqueeze(-1) * up).norm(dim=-1)              # (N,F)
    gap_raw = torch.clamp(rad - r, min=0.0)
    w_rad = _sat((0.9 * r - rad) / (0.3 * r))
    w_h = _sat((h_tip - (mouth_z - r)) / (0.5 * r)) * _sat((mouth_z + 1.5 * r - h_tip) / (0.5 * r))
    w_in = w_rad * w_h                                               # 1 = 입구 안
    gap = (1.0 - w_in) * gap_raw + w_in * (2.0 * r)
    h_t = h_tip[:, 0]
    h_err_t = torch.clamp(-h_t, min=0.0) + torch.clamp(h_t - 0.5 * mouth_z, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2 + 1e-12)      # (N,)
    h_o = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * mouth_z - h_o, min=0.0) + torch.clamp(h_o - 0.8 * mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2 + 1e-12)       # (N,F-1)
    return d_thumb, d_oth, 1.0 - w_in


def _surround(tips_pos, palm_pos, cup_pos, cup_up, mouth_z, r):
    """iter_01 [5-5 1] 손가락이 컵을 둘러쌌다는 기하 [0,1].
    = (서로 반대쪽) x (네 손가락 끝이 벽 가까이) x (엄지 끝이 벽 가까이).
    재시도 2 [T4]: 엄지를 다시 곱한다. 반쯤 쥔 손으로 다가갈 때(손바닥 축 2.41 r)는 네 손가락 끝이 이미
    벽(1.09 r)에 닿지만 엄지 끝은 2.28 r 로 멀다 -> 엄지 곱이 없으면 해제·닫기 수입을 받는다 (-0.989).
    엄지 높이 허용은 손가락과 같은 컵 몸통 높이대(-0.4 ~ 0.8 x 입구)로 넓혀 곱의 과민함을 줄인다.
    척도 0.4 r: 벽 0.15 r -> 0.87, 0.45 r -> 0.28, 1.28 r -> 0 (추정)."""
    EPS = 1e-6
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h = (rel * up).sum(dim=-1)                                       # (N,F)
    rad_v = rel - h.unsqueeze(-1) * up                               # (N,F,3)
    rad = rad_v.norm(dim=-1)                                         # (N,F)
    w_in = _sat((0.9 * r - rad) / (0.3 * r)) * _sat((h - (mouth_z - r)) / (0.5 * r))
    gap = (1.0 - w_in) * torch.clamp(rad - r, min=0.0) + w_in * (2.0 * r)
    herr = torch.clamp(-0.4 * mouth_z - h, min=0.0) + torch.clamp(h - 0.8 * mouth_z, min=0.0)
    d = torch.sqrt(gap * gap + herr * herr + 1e-12)
    cl = torch.exp(-(d / (0.4 * r)) ** 2)                            # (N,F)
    cl_t = cl[:, 0]
    cl_f = cl[:, 1:].mean(dim=1)

    u = rad_v / (rad.unsqueeze(-1) + EPS)
    p_rel = palm_pos - cup_pos
    p_h = p_rel - (p_rel * cup_up).sum(dim=-1, keepdim=True) * cup_up
    u_p = p_h / (p_h.norm(dim=-1, keepdim=True) + EPS)
    near = u_p + u[:, 0, :]
    near_hat = near / (near.norm(dim=-1, keepdim=True) + EPS)
    cos_f = (u[:, 1:, :] * near_hat.unsqueeze(1)).sum(dim=-1).mean(dim=1)
    #   재시도 2: 문턱 0.3 -> 0.5. 감싸 쥔 손끝(cos 약 -0.3, 추정)이 1 을 받게 한다 (D 단계 끝 하락 방지)
    opp = _sat((0.5 - cos_f) / 0.8)
    return opp * cl_f * cl_t


def _hand_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos, other_cup_pos,
                r, mouth_z, lift_target, lift_relief):
    """팔별 무가중 항. 두 팔 같은 식 (한 팔에만 붙는 부호·상수 없음). 모든 거리 척도는 컵 반지름 배수.
    iter_01 변경 (DESIGN_BRIEF 5-5 1):
      - 편 손 인자 open_fac 를 접근 거리·방향(approach)과 높이(low) 전체에 곱한다 (벌점 아님, 곱하는 상점).
      - 편 손 요구 해제(release)와 닫기(close)는 손바닥 자리(pocket) x 손가락 둘러싸기 기하(surround) 에서만.
    재시도 2 (Review items):
      [T4] surround 에 엄지 곱 복원. 닫힘 결손을 선형 shut = sat((닫힘 - 0.1) / 0.4) 로
           (반쯤 쥔 손 0.5 -> 1, 편 손 0.05 -> 0). k 는 잡는 자리 근처 0.15 (D 단계 닫기 기울기 약 1.2/팔/닫힘, 추정).
      [T1] _tip_geometry 입구 안 판정 높이 하한 (위 docstring)."""
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
    free = 1.0 - g_eff                                                # 쥐기 전 게이트

    # ---- 손 방향 점수 (수평 손)
    s_fl = 0.5 * torch.exp(-(f_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(f_dot))
    s_nl = 0.5 * torch.exp(-(n_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(n_dot))
    level = s_fl * (0.4 + 0.6 * s_nl)
    level_fac = 0.3 + 0.7 * level                                     # 바닥 0.3 인 수평 손 인자

    # ---- 1 접근 거리 (높이 x2 로 먼저 내려오기, 앞·중심선 쪽 가산은 쥐기 전에만)
    far_side = torch.clamp(palm_pos[:, 0] - cup_pos[:, 0], min=0.0) * free
    to_other = other_cup_pos[:, :2] - cup_pos[:, :2]
    e_other = to_other / (to_other.norm(dim=-1, keepdim=True) + EPS)
    lat = ((palm_pos[:, :2] - cup_pos[:, :2]) * e_other).sum(dim=-1)
    cross = torch.clamp(lat - 0.5 * r, min=0.0) * free
    dr_eff = torch.clamp(dr, min=0.0) + 3.0 * torch.clamp(-dr, min=0.0)
    dh_abs = _sabs(dh, 0.1 * r)
    side_d = dr_eff + far_side + cross
    d_app = side_d + 2.0 * dh_abs
    reach_far = _sat(1.0 - d_app / (30.0 * r))
    reach_near = (1.0 - torch.tanh(d_app / (2.0 * r))) * level_fac
    reach = 0.5 * reach_far + 0.5 * reach_near

    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    near_w = torch.exp(-(d_app / (4.0 * r)) ** 2)
    orient = level * (0.4 + 0.6 * near_w * dir_env)

    # ---- 손 안 판정: 손바닥 자리(pocket) x 손가락 둘러싸기(surround) (iter_01 [5-5 1])
    pocket = _sat((3.2 * r - rad_p) / r) * torch.exp(-(dh / (2.0 * r)) ** 2) * _sat((cos_env + 0.3) / 0.6)
    surround = _surround(tips_pos, palm_pos, cup_pos, cup_up, mouth_z, r)
    in_hand = _sat(pocket / 0.4) * surround                           # 같은 단계(컵이 손 안) 인자끼리의 곱
    release = torch.maximum(_sat(in_hand / 0.5), g_eff)

    # ---- 편 손 인자 (iter_01 [5-5 1], 재시도 2 [T4])
    #   shut: 닫힘 0.1 이하(편 손, 최소 약 0.05) -> 0, 0.5(반쯤 쥔 손) 이상 -> 1 (선형)
    #   k: 컵에서 멀면 0.9, 잡는 자리 근처 0.15.
    #      잡는 자리 근처 반쯤 쥔 손 결손 약 0.15 x 3.2 = 0.48/팔 > 손끝이 벽에 가까워져 얻는 wrap 약 0.3 (T4, 추정)
    #      D 단계 닫기 기울기 0.15 x 3.2 / 0.4 = 1.2/팔/닫힘 -> 닫힘 0.017/스텝이면 두 팔 약 0.04/스텝 (T1, 추정)
    shut = _sat((hand_closure - 0.1) / 0.4)
    k_open = 0.15 + 0.75 * (1.0 - near_w)
    open_fac = release + (1.0 - release) * (1.0 - k_open * shut)
    approach = open_fac * (0.65 * reach + 0.35 * orient)

    # ---- 1' 먼저 내려오기: 잡는 높이 점수 - iter_01: 편 손 인자를 여기에도 곱한다 (f00 원인)
    low = open_fac * (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 : 손끝별 바깥 벽 거리 볼록 평균(제곱) + 손가락이 둘러쌌을 때만 닫기
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    wrap = part * part * (0.15 + 0.85 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)
    #   iter_01: pocket 대신 in_hand (손바닥 자리만으로는 닫기 수입 없음)
    close = in_hand * level_fac * _sat(hand_closure / 0.7)

    # ---- 3 파지 : 판정 뒤 계단(>= 0.5) + 판정 전 접촉 사전 수입(<= 0.4)
    f_t = f_out[:, 0]
    f_o = f_out[:, 1:].sum(dim=1)
    pre = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    part_o = touch_o.mean(dim=1)
    post = 0.5 + part_o * part_o * part_o                             # 세제곱: 5지째 몫이 가장 크다
    grasp = g_eff * post + (1.0 - g_eff) * pre

    # ---- 4 인벨롭 모양 점수 + 손바닥 굴림 비용
    s_h = torch.exp(-(dh / (0.6 * r)) ** 2)
    s_r = torch.exp(-(dr / (0.5 * r)) ** 2)
    s_f = torch.exp(-(f_dot / 0.3) ** 2)
    s_n = torch.exp(-(n_dot / 0.3) ** 2)
    q = 0.30 * s_h + 0.25 * s_r + 0.30 * s_f + 0.15 * s_n
    q2 = q * q
    roll = _sat((torch.abs(n_dot) - 0.25) / 0.45)

    # ---- 5 들기 : 모양과 무관, 처음 몇 cm 가 가장 가파르고 목표 위는 이득 없음
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.2 * r, min=0.0)
    lin = _sat(h_pos / lift_target)
    lin = lin + (1.0 - lin) * lift_relief
    prog = 0.7 * torch.tanh(h_pos / (0.7 * r)) + 0.3 * lin
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    lift = g_eff * calm * prog
    lifted_w = _sat(h_pos / (0.7 * r))

    # ---- 교란
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
    rush = near_w * free * torch.tanh(_rms(arm_qd) / 1.0)
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
    """RH 양팔 붓기, 5번째 사이클 iter_01 (재시도 2): 단계별 합 사다리 (곱 사다리 없음). 두 팔 같은 식·같은 무게.
    iter_01 변경 (DESIGN_BRIEF 5-5): [1] 편 손 접근·둘러싸기 기하 해제 (_hand_terms),
    [2] 다 부으면 끝: 기울임 수입이 남은 비드 비율에 따라 줄고 세우기가 이어받는다.
    재시도 2 [1]: 기울임 수입은 남은 비드가 마지막 몫(TAIL = 25 %) 안으로 들어온 뒤에만 줄인다 ->
    붓는 도중(남은 비드 > 25 %) 이어받기 수준 항이 0 이라 회전·배달 몫이 이전 iter_00 수준으로 돌아간다.
    나머지 항·가중 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (iter_00 c1r2 그대로) ----------------------------------------------------------------
    W_ALIVE = 0.5
    W_APP = 2.0
    W_LOW = 1.2
    W_WRAP = 1.2
    W_CLOSE = 1.3
    W_GRASP = 3.0
    W_SHAPE = 1.5
    W_SHAPE_LIFT = 5.0
    W_ROLL = 1.5
    W_LIFT = 2.5
    W_AIM = 2.5
    W_POUR_H = 4.0
    W_ROT = 23.0          # weight 23 with delivery 30/bead: 회전+배달 몫 약 52 % (추정)
    W_DELIVER = 30.0      # 알당 30, 부호 있는 증분 (i20: 진행형 비대칭 clamp 금지)
    W_SPILL = 40.0
    W_SUCCESS = 20.0
    W_FINISH = 1.5
    W_MARGIN = 0.8
    #   이어받기 무게 = 회전 + 주둥이 높이 무게의 합 (새 크기 아님). 마지막 몫에서 잃는 기울임 수입을 같은 기울기로 이어받는다 (T1).
    W_CARRY = W_ROT + W_POUR_H
    #   재시도 2 [1]: 기울임 수입을 줄이기 시작하는 남은 비율. 성공 문턱(남은 20 %)보다 조금 넓게 (추정)
    TAIL = 0.25
    UPRIGHT_TOL = math.radians(20.0)
    LIM = math.radians(81.0)

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    cup_h = mouth_z - float(ctx.cup_bottom_z)          # 컵 높이 (자산 속성에서 계산)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    active = float(max(int(ctx.num_beads), 1))
    LIFT_RCV = 3.0 * r_cup

    src_g_raw = ctx.src_grasped.to(dt)
    rcv_g_raw = ctx.rcv_grasped.to(dt)
    both_g_raw = src_g_raw * rcv_g_raw
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) --------------------------------------------------------------
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
    above_rim = _sat(1.0 + dz_lip / r_cup)
    window = lip_gate * above_rim
    theta_pos = torch.clamp(theta, min=0.0)

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    clear_src = torch.clamp(h_rcv, 0.0, LIFT_RCV) + cup_h + r_cup
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

    # ---- 1~3 접근(+먼저 내려오기) / 감싸기 전 / 파지 ---------------------------------------------------
    approach_src = W_APP * src["approach"] + W_LOW * src["low"]
    approach_rcv = W_APP * rcv["approach"] + W_LOW * rcv["low"]
    prewrap = W_WRAP * (src["wrap"] + rcv["wrap"]) + W_CLOSE * (src["close"] + rcv["close"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- 4 인벨롭 모양 보너스 - 손바닥 굴림 비용 ---------------------------------------------------------
    envelope = src_g * (src["q2"] * (W_SHAPE + W_SHAPE_LIFT * src["prog"]) - W_ROLL * src["roll"]) \
        + rcv_g * (rcv["q2"] * (W_SHAPE + W_SHAPE_LIFT * rcv["prog"]) - W_ROLL * rcv["roll"])

    # ---- 5 들기 (모양 무관) --------------------------------------------------------------------------
    lift_src = W_LIFT * src["lift"]
    lift_rcv = W_LIFT * rcv["lift"]

    # ---- 6 조준 -----------------------------------------------------------------------------------
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))
    h_err = torch.abs(dz_lip - 0.25 * r_cup)
    hz = 0.3 * (1.0 - torch.tanh(h_err / (1.5 * r_cup))) + 0.7 * _sat(1.0 - h_err / (12.0 * r_cup))
    xy = 0.5 * (1.0 - torch.tanh(lip_d / wide)) + 0.5 * (1.0 - torch.tanh(lip_d / two_r))
    aim_carry = W_AIM * both_g * lifted * (0.6 * xy + 0.4 * hz)

    # ---- 비드 장부 (iter_01 [5-5 2], 재시도 2 [1]) --------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    #   남은 비율이 TAIL 이상이면 1, 그 아래에서 남은 비율에 비례해 0 까지.
    #   한 알(5 %) 남으면 0.2 (이전 iter_00 의 sat(need x 20) 은 한 알에도 1 -> 80 % 붓고 머묾 +1269).
    want = _sat(need_frac / TAIL)
    spill_frac = _sat(ctx.bead_spill_frac)

    # ---- 7 붓기: 창 안 회전 진척 x want ---------------------------------------------------------------
    rot_prog = _sat(theta_pos / (1.4 * LIM))            # 95 도까지 단조 증가
    tilt_share = want * window * rot_prog
    pour_rot = W_ROT * both_g * tilt_share               # 다 부으면 0 (같은 기울기에서도)
    pour_h = W_POUR_H * both_g * tilt_share * hz
    aim = aim_carry + pour_h

    # ---- 7 배달 (주역, 부호 있는 증분) ---------------------------------------------------------------
    pour_delta = W_DELIVER * ctx.d_in_target * active

    # ---- 다 부으면 끝: 마지막 몫 이어받기 + 세우기 / 성공 (5-5 2) ---------------------------------------------
    #   이어받기 W_CARRY x (1 - want) 는 기울기와 무관. 마지막 몫에서 want 가 줄 때 잃는 기울임 수입
    #   ((23 + 4 hz) x window x rot_prog <= 27) 을 같은 기울기(27)로 받으므로 한 스텝 하락이 없고 (T1),
    #   다 부은 뒤 세우는 동안 떨어지지 않으며 세울수록 W_FINISH 몫이 오른다.
    #   80 % 붓고 80 도 유지: 기울임 23 x 0.8 x 0.705 = 13.0 + 높이 <= 2.3 + 이어받기 27 x 0.2 = 5.4 -> <= 20.7
    #   < 끝까지 붓고 세우기 27 + 1.5 = 28.5 (스텝당 약 -7.8, 남은 4 알 배달 수입 별도) (T3, 추정)
    #   붓는 도중(남은 > 25 %) 이어받기 0 -> 붓는 상태 총수입에 수준 항이 끼지 않는다 ([1], 추정)
    upright_src = 1.0 - torch.tanh(tilt_src / (0.3 * LIM))
    finish_upright = both_g * (1.0 - want) * (W_CARRY * (1.0 - spill_frac) + W_FINISH * upright_src)
    success_bonus = W_SUCCESS * ctx.success.to(dt)

    # ---- 흘림 / 창 밖 조기 기울임 -----------------------------------------------------------------------
    post = torch.maximum(src_g_raw, src["lifted_w"])
    spill = -W_SPILL * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g_raw * pre_ramp * out_gate

    # ---- 충돌 (실기 안전, 유계) ------------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g_raw) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g_raw) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                                + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- 컵 교란 -------------------------------------------------------------------------------------
    tilt_pen_src = src_g * (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = rcv_g * (1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2))
    cup_disturb = -(src["disturb"] + rcv["disturb"] + 0.3 * tilt_pen_src + 0.3 * tilt_pen_rcv)

    # ---- 리시버 세움·중심선 쪽 --------------------------------------------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- 관절 여유 비용 (쥔 동안) -----------------------------------------------------------------------
    joint_margin = -W_MARGIN * (src_g * _limit_cost(ctx.src_arm_limit_margin)
                                + rcv_g * _limit_cost(ctx.rcv_arm_limit_margin))

    # ---- 제어 비용 (cup_grasp 수준; 팔 액션 크기 비용 없음, 손 채널 포화 벌점 없음) -----------------------------
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
