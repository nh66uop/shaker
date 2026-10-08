## 과제 해석과 단계

양팔 붓기 과제다. 오른손은 비드가 든 소스 컵을, 왼손은 빈 리시버 컵을 옆에서 인벨롭 파지로 잡는다. 손은 테이블과 수평이고, 손바닥은 세워서 컵을 보며, 손가락은 수평으로 컵을 감싼다. 두 컵을 들어 올린 뒤 소스 주둥이를 리시버 입구 위로 옮기고, 소스를 기울여 비드를 붓는다. 리시버 기울기는 20도 이하로 유지한다. 단계는 다음 순서다.

1. 손을 편 채 접근한다(먼저 내려와 수평으로).
2. 손바닥이 K7 자리에 들어오고, 손가락 사이에 컵이 있으며, 엄지가 반대쪽에 있다.
3. 닫아서 파지한다.
4. 들어 올린다.
5. 조준한다.
6. 회전해서 붓는다(배달 증분이 주된 보상).
7. 세우고 유지한다.

이번 응답은 iter_05 의 재시도다. 고친 것은 Review items 의 [64] 하나이고, 변경은 5-9 의 한 가설 범위 안에 있다.

## [64] 원인 진단 (계산, 일부 가설)

- [64] 픽스처에서 닫힘에 따라 바뀌는 것은 손끝 위치뿐이다. 손바닥 위치와 방향은 고정이다.
- 그래서 `pre_geo` 의 네 인자 중 손끝을 보는 것은 `opp_g`(엄지 대향) 하나다.
- 손가락이 말리면 두 가지가 일어난다.
  - 네 손끝이 컵 축으로 다가온다(2.25 r 에서 1.02 r). 그러면 부드러운 정규화의 길이가 줄어 cos 가 0 쪽으로 간다.
  - 네 손끝의 방위가 컵을 감싸며 엄지 쪽으로 돈다(가설).
- 이 둘 때문에 oppose 가 문턱(0.45~0.70) 아래로 내려가고, 그에 따라 release 와 close 가 줄어든다.
- 이 해석은 성분 기록과 맞는다. 0.45 에서 0.6 으로 갈 때 approach_src 가 -0.175 였고, 닫기 수입이 들어 있는 prewrap 은 +0.044 에 그쳤다.

## 수정

해제와 닫기에 쓰는 엄지 대향 문턱을 네 손끝이 벽에 닿은 정도(`wall_f`)만큼 낮춘다. 문턱 하한은 0.45 에서 최저 0.15 까지 내려간다.

- `wall_f` 는 손가락마다 max(벽 근접 exp(-(d/0.3r)^2), 바깥 접촉 touch)를 구해 네 손가락 평균을 낸 값이다.
- 닫기 전(손끝이 벽에서 먼 상태)에는 `wall_f` 가 0 에 가까우므로 문턱이 이전 응답과 같다.
- 입구 안 손끝은 gap 2r 이고 입구 안 접촉은 out_w 로 빠지므로 문턱을 낮추지 않는다.
- 엄지가 같은 방위(cos 1)이면 oppose 가 0 에 가까워 낮춘 하한 0.15 에도 못 미친다. 그래서 opp_g 는 0 이고 계속 손해다.
- 상태마다 release 는 이전 값 이상이고, approach·low·wrap·close 는 release 에 대해 단조 증가한다. 그래서 같은 상태의 보상이 이전 응답보다 낮아지지 않는다.
- [64] 의 0.45 에서 0.6 칸은 이렇게 추정한다(계산). close 가 최소 1.3 x 0.94 x 0.214 = +0.26 이고, grasp +0.121, disturb -0.018 이다. 합하면 +0.3 안팎이다.
- 나머지 항과 가중은 이전 응답(iter_04 c1r2 + 5-9) 그대로다.

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
    """iter_02 [5-6 1] 쥐기 전 접근 단계 수입에 곱하는 관절 여유 인자 [0.5, 1]. 이전 코드 그대로."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    q = torch.prod(1.0 - 0.5 * c, dim=-1)
    return 0.5 + 0.5 * q


def _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r):
    """컵 좌표(축 = cup_up) 손끝-바깥 벽 거리. 입구 안 손끝은 '먼 것'(2 r)으로 친다 (f07 교훈). 이전 코드 그대로."""
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


def _thumb_oppose(tips_pos, cup_pos, cup_up, r):
    """iter_04 [5-8 2] 엄지 끝과 네 손끝의 컵 축 둘레 방위 반대 정도 [0,1] (0 = 같은 방위, 1 = 정반대). 이전 응답 그대로."""
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
    iter_05 변경 (DESIGN_BRIEF 5-9, 한 가설·한 변경):
      [5-9 1] 편 손 요구 해제(release)·닫기 수입(close)의 '손이 컵을 둘러쌌다' 조건에서 손끝-벽 거리를 빼고,
              닫기 전에 이미 확인되는 기하 = 손바닥 자리 x 수평 손 x 인벨롭 방향 x 엄지 대향 으로 정한다 (같은 단계 인자, 상점).
              손끝-벽 거리·접촉은 감싸기 전(wrap)·파지(grasp) 수입에서 계속 본다.
      repair ([64] 0.45 -> 0.6 칸 -0.027): 닫으며 네 손끝이 컵 벽으로 말려 들면 엄지 대향 값이 내려가
              (부드러운 정규화 길이 감소 + 감싸며 방위가 엄지 쪽으로 돎 (가설)) 해제가 풀렸다.
              해제·닫기용 엄지 대향 문턱만 네 손끝이 벽에 닿은 정도만큼 낮춘다 (닫기 전에는 이전 문턱 그대로).
    나머지는 iter_04 c1r2 그대로."""
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

    # ---- 손바닥 자리(pocket): 판정 전 접촉 사전 수입(pre)·엄지 대향 몫이 쓴다 (그대로)
    pocket = _sat((3.2 * r - rad_p) / r) * torch.exp(-(dh / (2.0 * r)) ** 2) * _sat((cos_env + 0.3) / 0.6)

    # ---- iter_04 [5-8 2] 엄지 대향 (iter_05 에서는 해제·닫기 조건에도 쓰므로 앞으로 옮김, 식은 그대로)
    oppose = _thumb_oppose(tips_pos, cup_pos, cup_up, r)

    # ---- iter_05 [5-9 1] 닫기 전 기하 pre_geo [0,1] (손끝-벽 거리 없음, 네 인자 모두 같은 단계 = '컵이 손 안에 들어옴')
    #   site_c : 손바닥이 잡는 자리 가까이 - 축 거리 <= 2.2 r 에서 1, 2.6 r 에서 0, 높이는 잡는 높이에서 1 r 척도 (그대로).
    #   level_g: 손이 수평 - level 0.4 -> 0, 0.8 -> 1 (그대로).
    #   dir_g  : 손바닥 -> 컵 축 방향이 인벨롭 쪽(0.67 n + 0.73 f) - cos 0.6 -> 0, 0.9 -> 1 (그대로).
    #   opp_g  : 엄지가 네 손끝과 컵 축 둘레 반대쪽. 닫기 전(손끝이 벽에서 멂) 문턱은 이전 응답 그대로 0.45 -> 0, 0.70 -> 1.
    #     repair: 네 손끝이 바깥 벽에 닿은 정도 wall_f 만큼 문턱을 최대 0.30 낮춘다 (벽에 다 닿으면 0.15 -> 0, 0.40 -> 1).
    #       wall_f = 네 손가락 평균 max(벽 근접 exp(-(d/0.3r)^2), 바깥 접촉 touch). 입구 안 손끝은 gap 2 r·접촉 제외라 낮추지 않는다.
    #       엄지가 같은 방위(cos 1, oppose 약 0.05~0.13)면 낮춘 문턱 0.15 에도 못 미쳐 0 그대로 ([63] 머묾은 계속 손해).
    #       문턱은 해제 쪽으로만 움직이므로 같은 상태 수입은 이전 응답 이상 (release 에 대해 open_fac·close 단조).
    #       [64] 0.45 -> 0.6: 해제 유지 시 close 만 1.3 x 0.94 x 0.214 = +0.26 이상 (계산).
    site_c = _sat((2.6 * r - rad_p) / (0.4 * r)) * torch.exp(-(dh / r) ** 2)
    level_g = _sat((level - 0.4) / 0.4)
    dir_g = _sat((cos_env - 0.6) / 0.3)
    near_o = torch.exp(-(d_oth / (0.3 * r)) ** 2)                     # (N,F-1) 네 손끝 바깥 벽 근접
    wall_f = torch.maximum(near_o, touch_o).mean(dim=1)               # [0,1]
    opp_lo = 0.45 - 0.30 * wall_f
    opp_g = _sat((oppose - opp_lo) / 0.25)
    pre_geo = site_c * level_g * dir_g * opp_g
    #   해제: 닫기 전 기하가 서면(또는 쥠 판정) 편 손 요구를 푼다. 닫힘 0.05 의 편 손은 shut = 0 이라 해제가 켜지는 순간 절벽 없음.
    release = torch.maximum(pre_geo, g_eff)

    # ---- 편 손 인자 (iter_04 [5-8 1] repair2 그대로)
    C_OPEN = 0.05
    x_c = _sat((hand_closure - C_OPEN) / (1.0 - C_OPEN))
    shut_far = 0.7 * x_c + 0.3 * (1.0 - (1.0 - x_c) ** 2)
    shut_near = 0.6 * x_c + 0.4 * x_c * x_c
    shut = near_w * shut_near + (1.0 - near_w) * shut_far
    k_open = 0.3 + 0.6 * (1.0 - near_w)
    open_fac = release + (1.0 - release) * (1.0 - k_open * shut)
    k_wrap = 0.6 + 0.3 * (1.0 - near_w)
    open_wrap = release + (1.0 - release) * (1.0 - k_wrap * shut)

    # ---- iter_04 [5-8 2] 엄지 대향 몫 (접근 단계 안, 상점). 이전 응답 그대로.
    opp_site = _sat(pocket / 0.4) * oppose

    approach = j_pre * open_fac * (0.65 * reach + 0.35 * orient + 0.2 * level + 0.25 * opp_site)

    # ---- 1' 먼저 내려오기: 잡는 높이 점수 (그대로)
    low = j_pre * open_fac * (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 : 손끝별 바깥 벽 거리 볼록 평균(제곱) - 손끝-벽 거리는 여기서 계속 본다 (그대로)
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    wrap = j_pre * open_wrap * part * part * (0.15 + 0.85 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)
    #   iter_05 [5-9 1] 닫기 수입: 손이 둘러쌌다(손끝-벽) 대신 닫기 전 기하 pre_geo 로 켠다 (그대로).
    close = j_pre * pre_geo * level_fac * _sat(hand_closure / 0.7)

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
    """RH 양팔 붓기, 5번째 사이클 iter_05: 단계별 합 사다리 (곱 사다리 없음). 두 팔 같은 식·같은 무게.
    iter_05 변경 (DESIGN_BRIEF 5-9, 한 가설·한 변경): _hand_terms 의 release·close 조건을 닫기 전 기하
    (손바닥 자리·수평·인벨롭 방향·엄지 대향)로. repair: 그 엄지 대향 문턱을 손끝이 벽에 닿은 만큼 낮춤.
    나머지 항·가중은 iter_04 c1r2 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (iter_03 c1 그대로) ----------------------------------------------------------------
    W_ALIVE = 0.5
    W_APP = 2.0           # 접근 몫 최대: reach 0.65 + orient 0.35 + level 0.2 + 엄지 대향 0.25 -> 팔당 최대 2.9
    W_LOW = 1.2
    W_WRAP = 1.2
    W_CLOSE = 1.3         # iter_05: 닫기 전 기하로 켜지는 닫기 수입, 감싸기 직전 닫기를 이득으로 만든다
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

## 근거 메모

이번에 구현한 것은 단일 가설이다. brief 에 가설 id 줄이 없어서 prompt 전체를 한 후보로 구현했다. prompt 가 이번 라운드에 정한 변경은 5-9 의 한 가설·한 변경뿐이다. 이번 수정은 Review items [64] 에 대한 것이고, 그 가설 범위 안에서만 고쳤다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| `wall_f` (네 손끝 벽 근접 또는 바깥 접촉 평균) | [64] 0.45 에서 0.6 으로 갈 때 네 손끝 벽 접촉 비율이 0.52 에서 0.99 로 오르고 approach_src 가 -0.175 떨어졌다. 손끝을 보는 `pre_geo` 인자는 opp_g 하나뿐이라 원인을 거기로 본다(계산, 방위 이동은 가설). | prompt_c1_repair.md:1309-1312, 1013-1019 |
| `opp_lo = 0.45 - 0.30 * wall_f` (해제·닫기용 엄지 대향 문턱) | 5-9: 해제·닫기 조건은 닫기 전 기하로 정하고, 손끝-벽 거리는 "빼거나 약하게" 한다. 닫기 전에는 문턱이 그대로라 T4·T4b·[62] 유지를 노린다(가설). | prompt_c1_repair.md:826-829, 858 |
| 같은 방위 엄지는 계속 0 | [63]: 엄지가 같은 쪽에 머무는 것은 계속 손해여야 한다(oppose 약 0.05~0.13 은 0.15 미만). | prompt_c1_repair.md:828 |
| `close` 와 `release` 식 | 이전 응답 그대로다. 문턱만 바뀌어 close 와 release 가 단조 증가하므로 같은 상태의 수입은 이전 응답 이상이다. | prompt_c1_repair.md:1021, 1050 |
| 나머지 전 항과 가중 | "나머지 iter_04 c1r2 그대로" | prompt_c1_repair.md:826, 859 |

검증하지 않은 위험이 하나 있다(가설). 반쯤 쥔 손의 손끝이 C 구간 K7 자리에서 이미 벽에 아주 가까우면(0.3 r 척도 안) 그 손도 해제를 일부 받는다. 그러면 T4·T4b 의 편 손 우위가 줄 수 있다. 이 부분은 원장 T4·T4b 로 확인해야 한다.

재시도 prompt 경로: /home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/iter_05/prompt_c1_repair.md