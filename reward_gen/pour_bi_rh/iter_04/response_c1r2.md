brief 에 가설 id 줄이 없습니다. 규칙대로 prompt 전체를 한 후보로 구현했습니다. prompt 에 있는 가설은 DESIGN_BRIEF 5-8 의 하나뿐이고 변경은 둘입니다. 이번 재시도에서는 Review items 두 건만 고쳤습니다.

- [36] 원장 단조 실패: 2 행 8.65 가 3 행 8.58 보다 큽니다.
- T1 절벽: D 구간 t=366 에서 한 스텝 -0.051 이고, 팔마다 approach 가 -0.030 입니다.

바꾼 범위는 5-8 1 의 편 손 배수 `shut`, `k_open` 과 docstring 및 주석뿐입니다. 나머지 코드는 이전 응답 그대로입니다. 아래 효과 수치는 모두 손계산 추정이고 원장을 돌려 보지 않았습니다.

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
    """관절 한계 근접 연속 비용 (관절 번호 없음, 유계 [0,1)). 이전 코드 그대로 (쥔 뒤 joint_margin 용)."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    return torch.tanh(c.sum(dim=-1))


def _margin_fac(margin: torch.Tensor) -> torch.Tensor:
    """iter_02 [5-6 1] 쥐기 전 접근 단계 수입에 곱하는 관절 여유 인자 [0.5, 1] (관절 번호 없음, 유계, 바닥 0.5).
    관절별 c = (1 - sat(여유 / 0.25))^2, q = prod_j (1 - 0.5 c_j), 인자 = 0.5 + 0.5 q.
    참 궤적(여유 0.5)에서는 정확히 1. 이전 코드 그대로."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    q = torch.prod(1.0 - 0.5 * c, dim=-1)
    return 0.5 + 0.5 * q


def _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r):
    """컵 좌표(축 = cup_up) 손끝-바깥 벽 거리. 입구 안 손끝은 '먼 것'(2 r)으로 친다 (f07 교훈).
    입구 안 판정 높이 하한 = 입구 - 1 r (_surround 와 같은 값). 이전 코드 그대로."""
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
    """iter_01 [5-5 1] 손가락이 컵을 둘러쌌다는 기하 [0,1]. 이전 코드 그대로.
    = (서로 반대쪽) x (네 손가락 끝이 벽 가까이) x (엄지 끝이 벽 가까이)."""
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
    opp = _sat((0.5 - cos_f) / 0.8)
    return opp * cl_f * cl_t


def _thumb_oppose(tips_pos, cup_pos, cup_up, r):
    """iter_04 [5-8 2] 엄지 끝과 네 손끝의 컵 축 둘레 방위 반대 정도 [0,1] (0 = 같은 방위, 1 = 정반대).
    부드러운 정규화 u = v / sqrt(|v|^2 + r^2) (척도 = 컵 반지름): 축 근처 손끝의 방위 뒤집힘 없이 매끄럽게.
    네 손끝은 다시 정규화하지 않고 평균만 낸다 (평균 길이 <= 1 -> cos 은 [-1, 1]). 이전 응답 그대로."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h = (rel * up).sum(dim=-1, keepdim=True)
    rad_v = rel - h * up                                             # (N,F,3)
    u = rad_v / torch.sqrt((rad_v * rad_v).sum(dim=-1, keepdim=True) + r * r)
    u_f = u[:, 1:, :].mean(dim=1)                                    # 네 손끝 방위 평균 (길이 <= 1)
    cos_tf = (u[:, 0, :] * u_f).sum(dim=-1)
    return _sat(0.5 * (1.0 - cos_tf))


def _hand_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                arm_qd, arm_limit_margin, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                other_cup_pos, r, mouth_z, lift_target, lift_relief):
    """팔별 무가중 항. 두 팔 같은 식 (한 팔에만 붙는 부호·상수 없음). 모든 거리 척도는 컵 반지름 배수.
    iter_04 변경 (DESIGN_BRIEF 5-8, 한 가설·두 변경):
      [5-8 1] 편 손 배수 shut 이 닫힘 0.05~1.0 전 범위에서 기울기를 갖게 하고, 감싸기 전 수입 wrap 도
              편 손 인자 안 (둘러싸면 release 로 풀림).
      [5-8 2] 접근 수입 안에 엄지 대향 몫(손바닥이 잡는 자리 가까이 x 엄지-네 손끝 반대 방위).
    repair2 ([36] 2>3 골, T1 D 구간 -0.051): shut 모양을 컵에서의 거리(near_w)로 섞는다 -
      먼 곳은 이전 응답의 오목 모양(T4·T4b 먼 쪽 그대로), 잡는 자리 근처는 볼록 모양(낮은 닫힘에서 완만,
      높은 닫힘에서 가파름). 근처 k_open 0.35 -> 0.30 (먼 곳 0.9 그대로). 나머지는 이전 응답 그대로."""
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

    # ---- iter_02 [5-6 1] 쥐기 전 관절 여유 인자 (접근 단계 안에서만; 쥔 뒤 1)
    j_pre = 1.0 - free * (1.0 - _margin_fac(arm_limit_margin))

    # ---- 손 방향 점수 (수평 손)
    s_fl = 0.5 * torch.exp(-(f_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(f_dot))
    s_nl = 0.5 * torch.exp(-(n_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(n_dot))
    #   iter_03 [5-7 1] 뒤집힌 손(법선 z > 0 쪽으로 크게)은 아래로 기운 손보다 싸게: 위쪽으로만 깎는 인자 (그대로).
    up_ok = 1.0 - _sat((n_dot - 0.25) / 0.45)
    level = s_fl * (0.4 + 0.6 * s_nl) * (0.2 + 0.8 * up_ok)
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

    # ---- 편 손 인자 (iter_04 [5-8 1]: 전 범위 기울기, repair2)
    #   x = sat((c - 0.05)/0.95) (편 손 최소 약 0.05, 브리프 1절).
    #   먼 곳 shut_far = 0.7 x + 0.3 (1 - (1 - x)^2): 이전 응답 그대로 (오목, /닫힘 기울기 1.37 -> 0.74).
    #     k_open 먼 곳 0.9 그대로 -> 시작 자세 T4·T4b(0.05>0.3) 차는 이전 응답과 같다.
    #   근처 shut_near = 0.6 x + 0.4 x^2: 볼록, /닫힘 기울기 0.63 (0.05) -> 1.05 (0.5) -> 1.26 (0.75) -> 1.47 (1.0),
    #     평평한 구간 없음. 섞은 shut = near_w x 근처 + (1 - near_w) x 먼 곳 도 두 모양 모두 단조 증가라 전 범위 기울기.
    #   [36] 3 행(닫힘 0.2, 근처): 근처 편 손 비용 k x shut = 0.35 x 0.198 = 0.069 -> 0.30 x 0.105 = 0.032,
    #     wrap 비용 0.6 x 0.198 -> 0.6 x 0.105. 2 행(닫힘 0, shut 0)은 불변 -> 3 - 2 약 +0.1~0.2 상승 (계산, 가설: release 약 0.4).
    #   T1 D 구간(근처, 닫는 중): 근처 실효 기울기 k x slope = 0.19 (0.05) / 0.24 (0.2) / 0.32 (0.5) / 0.44 (1.0),
    #     이전 응답 0.48 / 0.44 / 0.37 / 0.26. 최대 하락이 낮은 닫힘 쪽(가설)이라 그 구간 기울기 약 45 % 감소 (계산).
    #   [62] 머묾 자세(근처) 0.1 칸 최소 Δshut 0.072 (0.1->0.2) x k 0.31~0.36: iter_03 기준 역산 M 으로 약 +0.05/칸 (계산, 목표 +0.03).
    #   접근 A~C 의 편 손(닫힘 0.05)은 x = 0 이라 shut = 0, 섞음 비율이 바뀌어도 참 궤적 접근 수입 불변.
    C_OPEN = 0.05
    x_c = _sat((hand_closure - C_OPEN) / (1.0 - C_OPEN))
    shut_far = 0.7 * x_c + 0.3 * (1.0 - (1.0 - x_c) ** 2)
    shut_near = 0.6 * x_c + 0.4 * x_c * x_c
    shut = near_w * shut_near + (1.0 - near_w) * shut_far
    k_open = 0.3 + 0.6 * (1.0 - near_w)
    open_fac = release + (1.0 - release) * (1.0 - k_open * shut)
    #   wrap 전용 편 손 계수 (같은 편 손 인자의 더 강한 형태, 둘러싸면 release 로 풀림). 이전 응답 그대로.
    k_wrap = 0.6 + 0.3 * (1.0 - near_w)
    open_wrap = release + (1.0 - release) * (1.0 - k_wrap * shut)

    # ---- iter_04 [5-8 2] 엄지 대향 몫 (접근 단계 안, 상점): 손바닥이 잡는 자리 가까이 x 엄지-네 손끝 반대 방위.
    #   편 손 인자·관절 여유 인자 안. free 게이트 없음 -> 쥠 판정 순간 절벽 없음. 이전 응답 그대로.
    oppose = _thumb_oppose(tips_pos, cup_pos, cup_up, r)
    opp_site = _sat(pocket / 0.4) * oppose

    #   iter_03 [5-7 1] 거리와 무관한 손 방향 몫 0.2 x level (그대로) + iter_04 엄지 대향 몫 0.25 x opp_site.
    approach = j_pre * open_fac * (0.65 * reach + 0.35 * orient + 0.2 * level + 0.25 * opp_site)

    # ---- 1' 먼저 내려오기: 잡는 높이 점수 (편 손 인자 + iter_02 관절 여유 인자) (그대로)
    low = j_pre * open_fac * (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 : 손끝별 바깥 벽 거리 볼록 평균(제곱) + 손가락이 둘러쌌을 때만 닫기
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    #   iter_04 [5-8 1] wrap 은 편 손 인자(wrap 전용 계수) 안: 둘러싸기 전 손끝을 말아 벽에 대는 수입은 손을 쥔 만큼 깎이고,
    #   둘러싸면(release -> 1) 원래대로.
    wrap = j_pre * open_wrap * part * part * (0.15 + 0.85 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)
    close = j_pre * in_hand * level_fac * _sat(hand_closure / 0.7)

    # ---- 3 파지 : 판정 뒤 계단(>= 0.5) + 판정 전 접촉 사전 수입(<= 0.4) (그대로)
    f_t = f_out[:, 0]
    f_o = f_out[:, 1:].sum(dim=1)
    pre_raw = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    pre_site = _sat(pocket / 0.4) * _sat(level / 0.6)
    pre = pre_raw * pre_site
    part_o = touch_o.mean(dim=1)
    post = 0.5 + part_o * part_o * part_o                             # 세제곱: 5지째 몫이 가장 크다
    grasp = g_eff * post + (1.0 - g_eff) * pre

    # ---- 4 인벨롭 모양 점수 + 손바닥 굴림 비용 (그대로)
    s_h = torch.exp(-(dh / (0.6 * r)) ** 2)
    s_r = torch.exp(-(dr / (0.5 * r)) ** 2)
    s_f = torch.exp(-(f_dot / 0.3) ** 2)
    s_n = torch.exp(-(n_dot / 0.3) ** 2)
    q = 0.30 * s_h + 0.25 * s_r + 0.30 * s_f + 0.15 * s_n
    q2 = q * q
    roll = _sat((torch.abs(n_dot) - 0.25) / 0.45)

    # ---- 5 들기 : 모양과 무관, 처음 몇 cm 가 가장 가파르고 목표 위는 이득 없음 (그대로)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.2 * r, min=0.0)
    lin = _sat(h_pos / lift_target)
    lin = lin + (1.0 - lin) * lift_relief
    prog = 0.7 * torch.tanh(h_pos / (0.7 * r)) + 0.3 * lin
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    lift = g_eff * calm * prog
    lifted_w = _sat(h_pos / (0.7 * r))

    # ---- 교란 (그대로)
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
    """RH 양팔 붓기, 5번째 사이클 iter_04: 단계별 합 사다리 (곱 사다리 없음). 두 팔 같은 식·같은 무게.
    iter_04 변경 (DESIGN_BRIEF 5-8, 한 가설·두 변경): _hand_terms 의 편 손 배수 전 범위 기울기(+ wrap 을 편 손 인자 안으로),
    쥐기 전 엄지 대향 몫(접근 수입 안). repair2: 편 손 배수 모양을 근처/먼 곳으로 섞음. 나머지 항·가중은 iter_03 c1 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (iter_03 c1 그대로) ----------------------------------------------------------------
    W_ALIVE = 0.5
    W_APP = 2.0           # 접근 몫 최대: reach 0.65 + orient 0.35 + level 0.2 + 엄지 대향 0.25 (iter_04) -> 팔당 최대 2.9
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
    W_MARGIN = 0.8        # 쥔 뒤 관절 여유 비용 (5-6: 그대로)
    W_CARRY = W_ROT + W_POUR_H
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
                      ctx.src_arm_limit_margin,
                      ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                      ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, clear_src, lift_relief_src)
    rcv = _hand_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                      ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                      ctx.rcv_arm_limit_margin,
                      ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                      ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, LIFT_RCV, zeros)

    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    alive = W_ALIVE * ones

    # ---- 1~3 접근(+먼저 내려오기) / 감싸기 전 / 파지 ----
    #   iter_04: approach 안에 엄지 대향 몫 (W_APP x 0.25 x opp_site, 팔당 최대 +0.5) 포함, wrap 은 편 손 인자 안
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

    # ---- 비드 장부 --------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac / TAIL)
    spill_frac = _sat(ctx.bead_spill_frac)

    # ---- 7 붓기: 창 안 회전 진척 x want ---------------------------------------------------------------
    rot_prog = _sat(theta_pos / (1.4 * LIM))            # 95 도까지 단조 증가
    tilt_share = want * window * rot_prog
    pour_rot = W_ROT * both_g * tilt_share
    pour_h = W_POUR_H * both_g * tilt_share * hz
    aim = aim_carry + pour_h

    # ---- 7 배달 (주역, 부호 있는 증분) ---------------------------------------------------------------
    pour_delta = W_DELIVER * ctx.d_in_target * active

    # ---- 다 부으면 끝: 마지막 몫 이어받기 + 세우기 / 성공 ---------------------------------
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

    # ---- 관절 여유 비용 (쥔 동안, 5-6: 그대로) -------------------------------------------------------------
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
```

가설: 지정 없음. prompt 의 단일 가설 5-8(두 변경)을 그대로 구현했고, 이번 재시도는 5-8 1 의 편 손 배수만 고쳤습니다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| `shut` 을 `near_w` 로 섞음: 먼 곳은 이전 오목 모양, 근처는 볼록 모양 `0.6x+0.4x^2` | [36] 2 행 8.65 > 3 행 8.58. 3 행은 닫힘 0.2 이고 근처라서, 근처 편 손 비용 0.069 를 0.032 로 낮춥니다. 2 행은 닫힘 0 이라 바뀌지 않습니다 (계산, 가설: release 약 0.4) | prompt_c1_repair2.md:1303-1304, 1314-1315 |
| 근처 `k_open` 0.35 를 0.30 으로 (먼 곳 0.9 는 그대로) | T1 D 구간 t=366 에서 -0.051, approach 가 팔마다 -0.030 입니다. 근처의 낮은 닫힘 구간 실효 기울기가 0.44~0.48 에서 0.19~0.24 로 내려갑니다 (최대 하락이 낮은 닫힘 쪽이라는 것은 가설) | :1305-1308 |
| 두 모양 모두 기울기가 0 보다 큼 (`/닫힘` 0.63~1.47, 0.74~1.37) | 5-8 1 "닫힘 0.05~1.0 평평한 구간 없음". [62] 머묾 자세의 0.1 칸 최소 차는 약 +0.05 입니다 (계산, 목표 +0.03) | :788, 813 |
| 먼 곳 모양과 `k_far` 0.9 는 이전과 같음 | T4·T4b 의 시작 자세 쪽 차이는 이전 응답과 같습니다 | :694, 788 |
| 편 손(닫힘 0.05)이면 x=0 이라 `shut`=0 | 참 궤적 A~C 의 접근 수입은 섞음 비율과 무관하게 그대로입니다 (T1~T5) | :690-695 |
| `open_wrap`, `k_wrap`, 엄지 대향, 나머지 항 | 이전 응답 그대로입니다 (5-8 1 wrap, 5-8 2 [63]) | :788-789, 818 |

위험 하나가 남아 있습니다. T1 최대 하락이 높은 닫힘(0.7 이상) 쪽에서 났다면, 이번 볼록 모양은 그 구간 기울기가 이전보다 큽니다(0.3 x 1.26~1.47 대 0.35 x 0.90~0.74). 그 경우 T1 이 다시 실패할 수 있습니다. 원장으로 확인해 주세요. prompt 파일 경로: /home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/iter_04/prompt_c1_repair2.md