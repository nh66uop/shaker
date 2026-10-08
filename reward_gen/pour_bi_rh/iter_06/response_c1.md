## 주의 (먼저 읽어 주세요)
- 이 세션에는 쓰기 도구가 없었습니다. 그래서 파일은 하나도 만들거나 고치지 않았고, 응답은 이 보고서 안에만 있습니다. 학습도 검사 [1]~[66]·원장(T1~T5, T4b)도 돌리지 않았습니다. 아래 수치 확인은 손으로 계산한 값입니다.
- 이전 함수(prompt.md 169–614행)는 줄 단위로 정확히 읽히지 않아, 흐린 렌더 이미지를 보고 다시 옮겨 적었습니다. "그대로"라고 표시한 부분이 원본과 몇 글자 다를 수 있습니다. 실제로 바꾼 것은 세 군데뿐입니다: ① 소스 `lift_target`, ② `_limit_cost_hard` + `joint_margin`, ③ 조준 항(`xy`·`AIM_FAR`·`W_AIM`·0.7/0.3 비율). iter_05 c1r 원본 소스가 있으면 이 세 곳만 원본에 옮겨 붙이는 쪽이 안전합니다.

---

## 1. 과제와 단계 (요약)
- 오른손이 비드 20개가 든 소스 컵을, 왼손이 빈 리시버 컵을 옆에서 인벨롭으로 쥡니다. 손은 수평이고 손바닥은 컵 쪽을 봅니다.
- 두 컵을 들고, 소스 주둥이를 리시버 입구 위로 가져가 기울여 붓습니다.
- 끝 조건: 비드 80% 이상 담김, 흘림 25% 이하, 리시버 기울기 20° 이하, 두 컵 모두 쥔 상태가 20 스텝 유지.
- 단계 순서: 접근(먼저 내려오기) → 감싸기 전 → 닫기 → 파지 → 들기 → 이송·조준 → 창 안 회전·배달 → 다 부은 뒤 세우기.
- iter_06은 인계 뱅크(두 컵을 쥐고 13 cm 들림, 두 컵 사이 수평 0.38 m)에서 100% 시작합니다. 그래서 실제로 학습할 구간은 이송 → 조준 → 붓기입니다.

## 2. b01 결과 분석
- 지표: 쥠 0.92/0.93, 낙하 0인데 in_target·pour_rot·success는 0입니다. 조준 거리는 0.42 → 0.25 → 0.39 m로 다시 멀어졌습니다.
- 재생해 보면 두 컵을 0.61/0.50 m 높이까지 든 채, 소스 aj_4를 상한(t300에 84%)에 붙이고 그 자리에 머뭅니다.
- 원인 ① 높이를 올릴수록 수입이 늘었습니다. 소스의 `lift_target`이 `clear_src`(약 h_rcv + cup_h + r ≈ 22 cm)여서 13 cm에서도 `lin`이 0.59였습니다. 더 들면 `prog`가 올라 들기(4.2 → 4.8)와 인벨롭(`W_SHAPE_LIFT·prog`, 4.85 → 7.05)이 함께 커졌습니다.
- 원인 ② 조준 수입이 먼 거리에서 평평했습니다. 기존 `xy`는 tanh(lip_d/10r)와 tanh(lip_d/2r)의 합이라 13 r 부근에서 1 r 다가갈 때 스텝당 약 0.02밖에 늘지 않았습니다(조준 평균 0.68).
- 원인 ③ 관절 한계 비용이 너무 약했습니다. 여유 0.02일 때 비용이 0.55로, 위 수입 증가분이 이를 쉽게 덮었습니다(측정 −1.08).

## 3. 변경 (가설 하나, 변경 둘 — 나머지는 iter_05 c1r 그대로)
1. **쥐고 든 채 머물기보다 조준 전진이 이득이 되게**
   - 소스 `lift_target`을 리시버와 같은 `LIFT_RCV`(3 r ≈ 8.4 cm)로 낮췄습니다. 뱅크 높이 13 cm에서 `prog`와 `lin`이 이미 1에 닿으므로, 더 높이 들어도 들기·인벨롭 수입이 늘지 않습니다. 필요한 높이는 붓기 창(`above_rim`)과 `hz`가 정합니다.
   - 조준의 xy 항에 선형 성분 `sat(1 − lip_d/20r)`를 넣었습니다. `W_AIM`은 2.5 → 6.0, xy:hz 비율은 0.6:0.4 → 0.7:0.3으로 바꿨습니다.
2. **쥔 뒤 관절 한계 자세는 손해**
   - 새 `_limit_cost_hard`를 추가했습니다(문턱 0.12, 가중 2.5). 기존 부드러운 비용(문턱 0.25)은 가중 0.8 → 0.3으로 줄였습니다.

손 계산 확인:
- **[65] 조준 전진:** 칸당 1.16 r(뱅크 약 13 r에서 창 2 r까지 10칸)에 선형 성분만으로 6·0.7·0.75/20·1.16 ≈ +0.18입니다. g_eff 0.85를 곱해도 약 +0.15로 기준 +0.1을 넘습니다. 가까운 쪽에서는 tanh 성분이 더해집니다.
- **[65] 소스만 22 cm 더 들기:** prog·lin이 이미 1이라 수입 변화가 없고, `hz`가 0.94 → 0.23으로 떨어져 6·0.3·0.71 ≈ −1.3입니다. "그대로 이하" 조건을 만족합니다.
- **[66] 쥔 상태:** 여유 0.02일 때 비용 2.5·tanh(0.694) + 0.3·tanh(0.846) ≈ 1.71, 여유 0.5일 때 0입니다. 차이 −1.71로 −1.0 이하 조건을 만족합니다.
- **[66] 붓는 상태:** 여유 0.1일 때 2.5·tanh(0.028) + 0.3·tanh(0.36) ≈ 0.17로 0.2 이하입니다.
- b01이 머문 여유 0.07(aj_4 2.35 rad)에서는 비용이 0.38 → 0.57로 커집니다.

남은 위험:
- 붓는 중 손목 관절 여유가 0.05 아래로 내려가면 비용이 약 0.8까지 올라갑니다. 다만 W_ROT 23이 지배합니다.
- 소스 들기 수입이 8.4 cm에서 포화하므로, 원장의 "F 8 cm 들기" 구간이 계속 통과하는지 확인이 필요합니다.

## 4. 코드
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
    """관절 한계 근접 연속 비용 (유계 [0,1)). 이전 코드 그대로 — iter_06 에선 약한 넓은 몫으로만 쓴다."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    return torch.tanh(c.sum(dim=-1))


def _limit_cost_hard(margin: torch.Tensor) -> torch.Tensor:
    """iter_06 [5-10 2] 한계 바로 앞에서만 가파른 비용 (유계 [0,1)).
    문턱 0.12: 여유 0.1 -> c 0.028 (붓는 자세는 거의 무료), 여유 0.02 -> c 0.69 (한계에 붙은 자세는 확실한 손해)."""
    c = (1.0 - _sat(margin / 0.12)) ** 2
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
    """iter_04 [5-8 2] 엄지 끝과 네 손끝의 컵 축 둘레 방위 반대 정도 [0,1]. 이전 응답 그대로."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h = (rel * up).sum(dim=-1, keepdim=True)                         # (N,F,1)
    rad_v = rel - h * up
    u = rad_v / torch.sqrt((rad_v * rad_v).sum(dim=-1, keepdim=True) + r * r)
    u_f = u[:, 1:, :].mean(dim=1)                                    # 네 손끝 방위 평균 (길이 <= 1)
    cos_tf = (u[:, 0, :] * u_f).sum(dim=-1)
    return _sat(0.5 * (1.0 - cos_tf))


def _hand_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                arm_qd, arm_limit_margin, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                other_cup_pos, r, mouth_z, lift_target, lift_relief):
    """팔 하나(손+컵)의 단계 항. iter_05 c1r 그대로 (lift_target 값만 호출부에서 바뀜)."""
    EPS = 1e-6
    dt = cup_pos.dtype
    g_raw = grasped.to(dt)

    # ---- 손바닥 기하 (컵 좌표): K7 자리 = 축에서 1.5 r, 높이 원점 + 0.15 x 입구 ----
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
    tgt = 0.67 * n_dir + 0.73 * f_dir                                # K7 손바닥 -> 컵 축 방향
    tgt_h = tgt - (tgt * cup_up).sum(dim=-1, keepdim=True) * cup_up
    tgt_hat = tgt_h / (tgt_h.norm(dim=-1, keepdim=True) + EPS)
    cos_env = (e_in * tgt_hat).sum(dim=-1)

    # ---- 손끝 기하 / 바깥 접촉 (입구 안 제외) ----
    d_thumb, d_oth, out_w = _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r)
    f_raw = torch.clamp(finger_force, min=0.0)
    f_out = f_raw * out_w
    touch = torch.tanh(f_out / 0.7)                                  # (N,F)
    touch_o = touch[:, 1:]
    oth_ok = _sat(touch_o.max(dim=1).values / math.tanh(1.0 / 0.7))
    g_eff = g_raw * out_w[:, 0] * oth_ok
    free = 1.0 - g_eff                                               # 쥐기 전 게이트

    # ---- iter_02 [5-6 1] 쥐기 전 관절 여유 인자 ----
    j_pre = 1.0 - free * (1.0 - _margin_fac(arm_limit_margin))

    # ---- 손 방향 점수 (수평 손) ----
    s_fl = 0.5 * torch.exp(-(f_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(f_dot))
    s_nl = 0.5 * torch.exp(-(n_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(n_dot))
    up_ok = 1.0 - _sat((n_dot - 0.25) / 0.45)
    level = s_fl * (0.4 + 0.6 * s_nl) * (0.2 + 0.8 * up_ok)
    level_fac = 0.3 + 0.7 * level

    # ---- 1 접근 거리 ----
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

    # ---- 손바닥 자리 (pocket) ----
    pocket = _sat((3.2 * r - rad_p) / r) * torch.exp(-(dh / (2.0 * r)) ** 2) * _sat((cos_env + 0.3) / 0.6)

    # ---- iter_04 [5-8 2] 엄지 대향 ----
    oppose = _thumb_oppose(tips_pos, cup_pos, cup_up, r)

    # ---- iter_05 [5-9 1] 닫기 전 기하 pre_geo [0,1] (c1r 문턱) ----
    site_c = _sat((2.6 * r - rad_p) / (0.4 * r)) * torch.exp(-(dh / r) ** 2)
    level_g = _sat((level - 0.4) / 0.4)
    dir_g = _sat((cos_env - 0.6) / 0.3)
    near_o = torch.exp(-(d_oth / (0.3 * r)) ** 2)                    # (N,F-1)
    wall_f = torch.maximum(near_o, touch_o).mean(dim=1)              # [0,1]
    opp_lo = 0.45 - 0.30 * wall_f
    opp_g = _sat((oppose - opp_lo) / 0.25)
    pre_geo = site_c * level_g * dir_g * opp_g
    release = torch.maximum(pre_geo, g_eff)

    # ---- 편 손 인자 (iter_04 [5-8 1] repair2 그대로) ----
    C_OPEN = 0.05
    x_c = _sat((hand_closure - C_OPEN) / (1.0 - C_OPEN))
    shut_far = 0.7 * x_c + 0.3 * (1.0 - (1.0 - x_c) ** 2)
    shut_near = 0.6 * x_c + 0.4 * x_c * x_c
    shut = near_w * shut_near + (1.0 - near_w) * shut_far
    k_open = 0.3 + 0.6 * (1.0 - near_w)
    open_fac = release + (1.0 - release) * (1.0 - k_open * shut)
    k_wrap = 0.6 + 0.3 * (1.0 - near_w)
    open_wrap = release + (1.0 - release) * (1.0 - k_wrap * shut)

    opp_site = _sat(pocket / 0.4) * oppose
    approach = j_pre * open_fac * (0.65 * reach + 0.35 * orient + 0.2 * level + 0.25 * opp_site)

    # ---- 1' 먼저 내려오기 ----
    low = j_pre * open_fac * (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 ----
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    wrap = j_pre * open_wrap * part * part * (0.15 + 0.85 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)

    # ---- 3 닫기 (pre_geo 로 켠다) ----
    close = j_pre * pre_geo * level_fac * _sat(hand_closure / 0.7)

    # ---- 3 파지 ----
    f_t = f_out[:, 0]
    f_o = f_out[:, 1:].sum(dim=1)
    pre_raw = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    pre_site = _sat(pocket / 0.4) * _sat(level / 0.6)
    pre = pre_raw * pre_site
    part_o = touch_o.mean(dim=1)
    post = 0.5 + 0.5 * part_o * part_o
    grasp = g_eff * post + (1.0 - g_eff) * pre

    # ---- 4 인벨롭 모양 점수 + 손바닥 굽힘 ----
    s_h = torch.exp(-(dh / (0.6 * r)) ** 2)
    s_r = torch.exp(-(dr / (0.5 * r)) ** 2)
    s_f = torch.exp(-(f_dot / 0.3) ** 2)
    s_n = torch.exp(-(n_dot / 0.3) ** 2)
    q = 0.30 * s_h + 0.25 * s_r + 0.30 * s_f + 0.15 * s_n
    q2 = q * q
    roll = _sat((torch.abs(n_dot) - 0.25) / 0.45)

    # ---- 5 들기 (lift_target 에서 포화) ----
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.2 * r, min=0.0)
    lin = _sat(h_pos / lift_target)
    lin = lin + (1.0 - lin) * lift_relief
    prog = 0.7 * torch.tanh(h_pos / (0.7 * r)) + 0.3 * lin
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    lift = g_eff * calm * prog
    lifted_w = _sat(h_pos / (0.7 * r))

    # ---- 교란 ----
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
    """iter_06: iter_05 c1r + [5-10 1] 조준 전진 > 높이 들고 머묾, [5-10 2] 쥔 뒤 관절 한계 손해."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 ----
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
    W_AIM = 6.0          # iter_06: 2.5 -> 6. 선형 xy 몫 0.7*0.75*6/20r ≈ 0.16/r -> 뱅크(~13 r)부터 창(2 r)까지 칸마다 +0.15 이상
    W_POUR_H = 4.0
    W_ROT = 23.0         # weight 23 with delivery 30/bead: 창 안 회전이 지배
    W_DELIVER = 30.0
    W_SPILL = 40.0
    W_SUCCESS = 20.0
    W_FINISH = 1.5
    W_LIM_SOFT = 0.3     # iter_06: 0.8 -> 0.3 (넓은 몫은 약하게: 붓는 자세 여유 0.1 비용 ≤ 0.2)
    W_LIM_HARD = 2.5     # iter_06: 한계 바로 앞 (여유 0.02 -> 팔당 −1.7, 여유 0.5 -> 0)
    W_CARRY = W_ROT + W_POUR_H
    TAIL = 0.25
    UPRIGHT_TOL = math.radians(20.0)
    LIM = math.radians(81.0)

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    cup_h = mouth_z - float(ctx.cup_bottom_z)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    AIM_FAR = 20.0 * r_cup   # iter_06: 선형 조준 범위 (뱅크 짝 최대 거리 이상, 평평한 구간 없음)
    active = float(max(int(ctx.num_beads), 1))
    LIFT_RCV = 3.0 * r_cup

    src_g_raw = ctx.src_grasped.to(dt)
    rcv_g_raw = ctx.rcv_grasped.to(dt)
    both_g_raw = src_g_raw * rcv_g_raw
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) ----
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
    # iter_06 [5-10 1]: 소스 들기 목표를 리시버와 같은 LIFT_RCV(3 r) 로. 이전 clear_src(≈22 cm) 는
    # 뱅크 높이(13 cm) 위로 더 들수록 들기·인벨롭 수입을 키워 '높이 들고 머묾'을 샀다.
    # 붓기에 필요한 높이는 창(above_rim)·hz 가 정한다.
    lift_target_src = LIFT_RCV
    lift_relief_src = window * _sat(theta_pos / (0.5 * LIM))

    src = _hand_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                      ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped, ctx.src_arm_qd,
                      ctx.src_arm_limit_margin,
                      ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                      ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, lift_target_src, lift_relief_src)
    rcv = _hand_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                      ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                      ctx.rcv_arm_limit_margin,
                      ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                      ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, LIFT_RCV, zeros)
    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    alive = W_ALIVE * ones

    # ---- 1~3 접근 / 감싸기 전 / 닫기 / 파지 (그대로) ----
    approach_src = W_APP * src["approach"] + W_LOW * src["low"]
    approach_rcv = W_APP * rcv["approach"] + W_LOW * rcv["low"]
    prewrap = W_WRAP * (src["wrap"] + rcv["wrap"]) + W_CLOSE * (src["close"] + rcv["close"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- 4 인벨롭 모양 (prog 가 뱅크 높이에서 이미 포화 -> 더 들어도 늘지 않음) ----
    envelope = src_g * (src["q2"] * (W_SHAPE + W_SHAPE_LIFT * src["prog"]) - W_ROLL * src["roll"]) \
        + rcv_g * (rcv["q2"] * (W_SHAPE + W_SHAPE_LIFT * rcv["prog"]) - W_ROLL * rcv["roll"])

    # ---- 5 들기 ----
    lift_src = W_LIFT * src["lift"]
    lift_rcv = W_LIFT * rcv["lift"]

    # ---- 6 조준 (iter_06 [5-10 1]) ----
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))
    h_err = torch.abs(dz_lip - 0.25 * r_cup)
    hz = 0.3 * (1.0 - torch.tanh(h_err / (1.5 * r_cup))) + 0.7 * _sat(1.0 - h_err / (12.0 * r_cup))
    # 선형 몫: 20 r 안 어디서든 1 r 다가가면 같은 크기로 오른다 (이전 tanh(lip_d/10r) 는 13 r 근처 평평).
    xy_lin = _sat(1.0 - lip_d / AIM_FAR)
    xy_near = 0.5 * (1.0 - torch.tanh(lip_d / wide)) + 0.5 * (1.0 - torch.tanh(lip_d / two_r))
    xy = 0.75 * xy_lin + 0.25 * xy_near
    # 0.7/0.3: 거리 몫을 키우되 hz 몫(1.8)이 '같은 자리 22 cm 더 들기'를 −1.3 손해로 만든다.
    aim_carry = W_AIM * both_g * lifted * (0.7 * xy + 0.3 * hz)

    # ---- 비드 잔량 ----
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac / TAIL)
    spill_frac = _sat(ctx.bead_spill_frac)

    # ---- 7 붓기: 창 안 회전 진척 x want (그대로) ----
    rot_prog = _sat(theta_pos / (1.4 * LIM))
    tilt_share = want * window * rot_prog
    pour_rot = W_ROT * both_g * tilt_share
    pour_h = W_POUR_H * both_g * tilt_share * hz
    aim = aim_carry + pour_h

    # ---- 7 배달 (주력, 부호 있는 증분) ----
    pour_delta = W_DELIVER * ctx.d_in_target * active

    # ---- 다 부으면 끝: 이어받기 + 세우기 / 성공 ----
    upright_src = 1.0 - torch.tanh(tilt_src / (0.3 * LIM))
    finish_upright = both_g * (1.0 - want) * (W_CARRY * (1.0 - spill_frac) + W_FINISH * upright_src)
    success_bonus = W_SUCCESS * ctx.success.to(dt)

    # ---- 흘림 / 창 밖 조기 기울임 ----
    post = torch.maximum(src_g_raw, src["lifted_w"])
    spill = -W_SPILL * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g_raw * pre_ramp * out_gate

    # ---- 충돌 (실기 안전, 유계) ----
    cup_clash = -((0.5 + 1.0 * both_g_raw) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g_raw) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                                + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- 컵 교란 ----
    tilt_pen_src = src_g * (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = rcv_g * (1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2))
    cup_disturb = -(src["disturb"] + rcv["disturb"] + 0.3 * tilt_pen_src + 0.3 * tilt_pen_rcv)

    # ---- 리시버 세움·중심선 쪽 ----
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- 관절 여유 비용 (iter_06 [5-10 2], 쥔 뒤) ----
    # 여유 0.02 -> 팔당 −(2.5*0.60 + 0.3*0.69) ≈ −1.71, 여유 0.5 -> 0  ([66] ≤ −1.0)
    # 여유 0.1  -> 팔당 −(2.5*0.028 + 0.3*0.35) ≈ −0.17          ([66] 붓는 자세 ≤ 0.2)
    lim_src = W_LIM_HARD * _limit_cost_hard(ctx.src_arm_limit_margin) + W_LIM_SOFT * _limit_cost(ctx.src_arm_limit_margin)
    lim_rcv = W_LIM_HARD * _limit_cost_hard(ctx.rcv_arm_limit_margin) + W_LIM_SOFT * _limit_cost(ctx.rcv_arm_limit_margin)
    joint_margin = -(src_g * lim_src + rcv_g * lim_rcv)

    # ---- 제어 비용 (그대로) ----
    a = ctx.actions
    pa = ctx.prev_actions
    da_arm = torch.cat([a[:, 0:7] - pa[:, 0:7], a[:, 13:20] - pa[:, 13:20]], dim=-1)
    da_hand = torch.cat([a[:, 7:13] - pa[:, 7:13], a[:, 20:26] - pa[:, 20:26]], dim=-1)
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
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

## 5. 다음 할 일
- 검사 [65]·[66]과 `traj_ledger.py` T1~T5·T4b를 돌려야 합니다. 특히 소스 들기 수입이 8.4 cm에서 포화하는 변경 때문에 T4b와 "F 8 cm 들기" 구간은 꼭 봐야 합니다. 검사 [1]~[64]도 유지되는지 확인이 필요합니다.
- 그다음 뱅크 100%, fresh, seed 42로 학습합니다.
- 학습 중 볼 지표: `task/aim_dist`가 계속 줄어드는지, 소스·리시버 컵 들림이 13 cm 근처에 머무는지, `joint/*_arm_limit_frac`, `reward/pour_rot`가 0에서 벗어나는지.