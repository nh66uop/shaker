## 1. 과제의 의미와 로봇이 거쳐야 할 단계

오른팔(source)은 비드가 든 컵을, 왼팔(receiver)은 빈 컵을 잡는다. 두 컵을 들고 소스 컵의 붓는 림 점(lip)을 리시버 입구 위로 가져간 뒤, 리시버 쪽으로 기울여 비드를 **공중으로 떨어뜨려** 옮긴다. 요청한 개수(±1개)만 붓고, 소스 컵을 다시 세워 기울이기 전 자세로 돌아가며, 그동안 두 컵을 계속 쥐고 있어야 한다.

| 단계 | 로봇이 할 일 | 보상에서 담당하는 항 |
|---|---|---|
| 0. 접근 | 양손 손바닥을 각자 컵으로 가져간다 | approach_src / approach_rcv |
| 1. 감싸 쥐기 | 다섯 손가락 모두 중간·원위 마디로 컵을 감싸고 손바닥도 닿게 한다 | grasp_*, posture_* (들기 전) |
| 2. 들기 | 두 컵을 스폰 높이보다 들어 올린다. 리시버는 직립 유지 | lift_src / lift_rcv |
| 3. 쥔 상태 유지 | 들어 올린 순간의 손-컵 관계(자세, 힘 분포, 힘 크기, 닿아 있던 손가락)를 유지한다 | posture_* (들고 난 뒤) |
| 4. 만나기 | 리시버는 자기 쪽에 머물되 중심선 쪽으로 조금 오고(측면 여유 약 8 cm), 소스가 중심선 부근까지 온다. 붓는 방향은 옆 또는 몸 쪽 | home_rcv, center_src, aim(dir_ok, side_ok) |
| 5. 조준 | lip 을 리시버 입구 중앙, 림 바로 위 낮은 높이에 둔다 | aim |
| 6. 기울이기 | 리시버 **쪽으로** 기울인다. 첫 비드가 나오는 각도는 채움에 따라 다르고(약 90° 이상) 보상에 각도를 넣지 않는다 | tilt_rate (첫 비드 전까지만) |
| 7. 붓기 | 비드가 리시버에 도착하는 만큼 보상하고, 요청량을 넘기거나 흘리면 벌점을 준다 | pour_flow, overshoot, spill |
| 8. 마무리 | 요청량에 도달하면 소스 컵을 세우고 기울이기 전 자세로 돌아간다. 두 컵을 계속 쥔다 | finish, success |
| 전 과정 | 컵끼리 충돌 없음, 손은 자기 컵에만 닿음, 조준 전 과도한 기울기 없음, 떨림 없음 | cup_collision, hand_foreign, premature, action_rate, tremor |

## 2. 피드백 분석 (t2r_i26r, 로그와 관찰 노트 기준)

| 관찰 | 해석 |
|---|---|
| grasp_* 1.1, lift_src 2.2, lift_rcv 1.2 로 포화. 양손 grasp 0.83 에서 0.87, 끝에 두 컵 모두 쥐고 있는 비율 0.98 | 잡기, 들기, 나르기는 해결됐다 |
| tilt_rate 평균 -0.21/step, 모든 구간에서 음수(가장 큰 벌점). 지급 0, 에피소드당 -76 에서 -125 | iter_26 회전 항은 지급 게이트에 spout 를 곱했고, trace 전체에서 spout 는 1e-17 이하였다. 그래서 기울기를 늘리면 0, 줄이면 전액 벌점이었고, 결국 기울기를 유지하거나 컵을 움직이는 데 매기는 세금이 됐다 |
| src_tilt_deg 22 에서 36°. 그런데 리시버 쪽 기울기 p50 은 6° | 기울기가 붓는 동작이 아니라 몸통 쪽으로 젖힌 것이다. 첫 비드는 약 90° 이상에서 나오므로 붓기가 시작된 적이 없다 |
| aim 0 에서 0.23 으로 증가, aim_dist 끝 p50 0.24 m (바디 기준 5.6 cup radii) | 접근은 진행 중이지만 느리다. 1 cm 가까워질 때 aim 이 +0.025 늘지만, 움직일 때 tilt 벌점은 -0.06 에서 -0.22 이다 |
| home_rcv 0.18 에서 0.53, 리시버는 스폰에서 3 cm 이내(y 약 +0.16) | 앵커가 스폰이라 리시버가 마중 나오지 않는다. 8 cm 옮기면 aim + home 이 -0.042 로 손해다 |
| pour_flow, overshoot, finish, success 모두 0 | 비드가 한 번도 나오지 않았다(want 가 항상 1). iter_26 의 finish, spill, want 변경은 아직 검증되지 않았다 |
| 소스 5지 접촉 0.93 에서 0.38(새끼손가락), 리시버 0.73 에서 0.17 | 들기 전 자세는 3지에서 포화된다. 소스 keep 에는 손가락 유지 인자가 없고, 리시버 인자는 절대값 0.2 N 기준이라 약하게 닿은 새끼손가락의 이탈을 보지 못한다 |
| tremor 0, adr/progress 0 | ADR 이 0 이라 제어 벌점 ramp 가 꺼져 있다(의도대로) |

## 3. iter_27 변경 (이 세 가지만)

### 변경 1. 회전 보조를 퍼텐셜의 정확한 변화율로 (`tilt_rate`, 이름·TILT_W·TILT_PAY_CAP·첫 비드 뒤 꺼짐 유지)

- Φ = latches · spout · lean
  - lean = max(û_cup · d, 0), d = 소스 컵 원점에서 리시버 입구로 향하는 수평 단위벡터. 리시버 쪽으로 기운 정도만 센다(각도 상수 없음).
  - spout = iter_26 의 fine · height (폭은 cup_radius 배수).
  - latches = settled · (1 - released) · want. 첫 비드 전에는 한 방향으로만 움직인다. settled 는 컵이 스폰에 떨어져 있어 spout 가 약 0 일 때 오르고, released 는 의도된 꺼짐이며, want 는 비드가 나가기 전까지 고정이다. 그래서 latches 는 변화율에 곱하는 배율로 둔다.
- 변화율: 두 컵을 각자의 lin_vel / ang_vel 로 τ 만큼 되돌린 상태에서 입구, d, θ, lip 를 ctx 주석의 식(lip = 입구 + r(cos θ d - sin θ z), θ = atan2(û·d, û_z))으로 다시 만들고, (Φ(지금) - Φ(되돌린 상태)) / τ 를 계산한다. 리시버 이동으로 생기는 d, spout 변화까지 포함한 **전체 변화율**이다. r 은 |src_pour_lip_pos - src_cup_mouth_pos| 로 ctx 에서 얻는다(자산 상수를 하드코딩하지 않음).
- 양의 변화율에는 변화율이 없는 [0,1] 인자 aim_gate(carry, not_nested, dir_ok, side_ok, up_rcv, grip_gate)를 곱하고 스텝당 +1 로 자른다. 음의 변화율은 자르지 않고 전액 벌점이다. 이 인자들은 Φ 안에 넣지 않았다.
- τ = 1/120 s (가설: 물리 스텝). 되돌림 시간이 물리 스텝과 같으면 스텝별 변화율의 합이 정확히 망원합(telescoping)이 된다. τ 가 물리 스텝보다 크면 입구 중심(오목한 곳)에서 진동 펌프가 생기므로 크게 잡지 않았다. 물리 스텝이 1/120 s 가 아니면 이 값 하나만 맞추면 된다.
- 결정론적 점검 (물리 1/120 s, 제어 1/60 s, 전체 compute_reward 호출, 결과는 에피소드 합):

| 점검 | 결과 |
|---|---|
| (a) 입구 위에서 0→92°→0 기울임, 4 s | -189 (지급 92.7, 벌점 -282) |
| (a) 57° 로 기운 채 lip 을 입구 안팎으로(-8 에서 +2 cm) | -312 |
| (a) 기운 채 lip 을 위아래로 | -145 |
| (a) 소스는 고정, 리시버가 움직이고 기울어짐 | -312 |
| (a) 두 컵이 함께 움직이는 무작위 닫힌 궤적 100개 | 최악 -61 |
| (a) 지급 게이트 토글(3 Hz, 올릴 때만 0, 내릴 때만 0) | -190 에서 -282 (자르지 않은 변화율 합 -0.0025) |
| (a) 적대적 저속 진동(자르기가 걸리지 않는 속도) | 물리 1/120 s 에서 최악 +0.0034/step, 물리 1/240 에서 1/60 s 범위에서 최악 +0.034/step (스텝 샘플링 잔차) |
| (b) 같은 기울임을 lip 이 입구에서 40 cm 떨어진 곳에서 | 매 스텝 0 |
| (c) 리시버 반대로 기울임 / 옆으로 기울임 | 매 스텝 0 |
| 입구 위에서 69° 로 정지 | 매 스텝 0 |
| 참고: 입구에서 0→92° 한 번 기울임(실제 붓기) | +166 (스텝당 상한 1) |

- 웜스타트 효과: ep800 상태에서는 spout 가 약 0 이라 새 항도 약 0 이다. 그 자리에서 받던 -0.16/step 벌점이 사라지고, 몸통 쪽으로 젖히는 동작은 지급도 벌점도 없다.

### 변경 2. 중심선 부근에서 만나기 (`home_rcv`)

- 형태, HOME_W, carry_rcv, up_rcv, side_ok, still_rcv 는 그대로 두고 앵커만 옮긴다. 앵커는 리시버 스폰 x 이면서 측면 여유 = SIDE_OK(0.08 m, side_ok 가 1 인 가장 작은 여유)인 점이다. rcv_side_margin 은 리시버의 측면 위치와 1:1 로 움직이므로 d_home = sqrt((x - x_spawn)² + (margin - SIDE_OK)²) 이다.
- 여유 0.16 / 0.12 / 0.08 / 0.05 m 에서 home_rcv = 0.73 / 0.85 / 1.00 / 0.31. 이제 8 cm 마중 나오면 +0.27/step 이다(이전에는 -0.042). SIDE_VOID 앞에서는 home 과 side_ok 가 함께 떨어진다. 소스는 CENTER_BAND 를 넘지 않고 y 약 -0.02 에서 만나게 된다.

### 변경 3. 양손 다섯 손가락 감싸기, 들어 올린 순간 기준 (`posture_src`, `posture_rcv`)

- 들기 전: wrap_q = wrap_count / F (F = finger_force.shape[1]). 3지에서 포화하지 않는다. 소스 들기 전 자세 점수는 3 / 4 / 5 wrap 에서 1.86 / 2.16 / 2.46 이다.
- 들고 난 뒤: 소스 keep 에도 리시버와 같은 fingers-kept 인자를 곱한다. 두 손 모두 들 때 닿아 있던 손가락(힘 비율 > 0.02)이 **자기 기준 힘의 절반 이상**(ref share · ref total · 0.5)으로 누르고 있으면 유지로 친다. 절대 힘 바닥이나 힘 목표는 없다. 기준 1 N 인 새끼손가락이 1.0 / 0 / 0.25 / 0.5 N 일 때 posture_src = 1.5 / 1.2 / 1.35 / 1.5 (keep = 1 인 경우). 새끼손가락을 놓치면 이전 -0.04/step 대신 약 -0.19/step 이다.

### 예산

- tilt_rate 는 변화율이라 자세를 유지하면 0 이다. 상태 수입을 더하지 않는다.
- home_rcv 최대값은 HOME_W 그대로다.
- posture 는 kept ≤ 1 이고 wrap/F ≤ min(wrap/3, 1) 이라 iter_26 보다 커지지 않는다. 따라서 성공 시 결과 수입이 상태 수입보다 크다는 조건은 유지된다.

### 그대로 둔 것

approach, grasp, lift, aim(정밀도·높이 포함), pour_flow, overshoot, spill(요청량 단위, 하한 OVER_DELTA_W), finish(허용오차에서 열림, 바디 기준 pre-tilt pose), success, premature, cup_collision, hand_foreign, center_src, iter_24 grip_keep, ADR 에 ramp 하는 action_rate / flips / tremor(리시버 가중 유지)를 그대로 두었다. 각도 목표, 각도 대역, 각도 상한은 없다.

### 다음 학습에서 볼 것 (ep800 웜스타트, 300 epoch 뒤 게이트)

- tilt_rate 의 에피소드당 지급과 벌점: lip 이 입구에 닿기 전에는 0 근처여야 하고, 기울임이 시작되면 지급이 나타나야 한다.
- 게이트: 리시버 쪽 최대 기울기 p50 > 20°, 끝 spout-입구 p50 < 0.12 m, 리시버 측면 여유 p50 < 0.12 m, 소스 5지 운반 ≥ 0.8, 리시버 ≥ 0.5. 실패하면 iter_27 로 fresh 학습.
- fingers-kept 는 들 때 닿아 있던 손가락만 지킨다. 들어 올린 기준에 새끼손가락(힘 비율 > 0.02)이 들어간 비율도 함께 본다.

## 4. 코드

```python
import torch
import math


def _step(x: torch.Tensor) -> torch.Tensor:
    """smoothstep on an already normalised argument -> [0,1]"""
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


def _grasp_quality(finger_force: torch.Tensor, palm_force: torch.Tensor,
                   wrap_count: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """[0,1] quality of one hand on its own cup BEFORE the lift reference exists. Only the SHAPE of
    the contact enters (wrapped fingers, palm contact, thumb contact), no cup dimension.
    iter_27 change 3: wrapped fingers are counted out of ALL fingers of the hand (F from the tensor
    shape) instead of saturating at 3, so the lift reference records a five-finger envelope."""
    n_fingers = float(max(int(finger_force.shape[1]), 1))
    wrap_q = torch.clamp(wrap_count.to(dt) / n_fingers, 0.0, 1.0)
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)      # [N] force sensor scale
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    return 0.5 * wrap_q + 0.25 * palm_q + 0.25 * thumb_q


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """iter_24, KEPT UNCHANGED: how closely the hand still holds the cup the way it did at the
    instant it lifted it (hand pose in CUP coordinates, force distribution and force level RELATIVE
    to that instant). Never an absolute force target. Before the reference exists the factor is 1."""
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = torch.clamp(0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim, 0.0, 1.0)
    return torch.where(valid, keep, torch.ones_like(keep))


def _fingers_kept(ref_share, ref_total, finger_force, valid, on_share, keep_frac):
    """iter_27 change 3 (BOTH hands): share of the fingers that touched the cup AT LIFT that still
    press with at least keep_frac of their OWN lift force (ref share * ref total). Purely relative to
    the lift hold: no absolute force floor, no force target. 1 before the reference exists or if no
    finger touched at lift. share = [palm, thumb..pinky], so the fingers are share[:, 1:]."""
    ref_f = torch.clamp(ref_share[:, 1:], min=0.0) * ref_total[:, None]     # [N] each finger at lift
    ref_on = (ref_share[:, 1:] > on_share).to(finger_force.dtype)
    need = torch.clamp(keep_frac * ref_f, min=1e-6)                         # numeric guard only
    touch = _step(torch.clamp(finger_force, min=0.0) / need)
    n_ref = torch.sum(ref_on, dim=-1)
    kept = torch.sum(ref_on * touch, dim=-1) / torch.clamp(n_ref, min=1.0)
    kept = torch.where(n_ref > 0.5, kept, torch.ones_like(kept))
    return torch.where(valid, kept, torch.ones_like(kept))


def _spout(lip, rcv_mouth, cup_r, geo):
    """iter_26 spout geometry, unchanged: (horizontal lip offset, fine, height).
    geo = (precision width, below-rim width, drop width, drop floor); widths in cup radii."""
    prec_r, below_r, drop_r, drop_floor = geo
    delta = lip - rcv_mouth
    lip_xy = torch.norm(delta[:, :2], dim=-1)
    lip_dz = delta[:, 2]
    fine = torch.exp(-(lip_xy / (prec_r * cup_r)) ** 2)                 # spout centred in the mouth
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (below_r * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (drop_r * cup_r)) ** 2)
    height = over_rim * (drop_floor + (1.0 - drop_floor) * low_drop)
    return lip_xy, fine, height


def _turned(vec, ang_vel, tau):
    """vec carried by a body that turns at ang_vel, as it was tau seconds EARLIER (Rodrigues)."""
    wn = torch.norm(ang_vel, dim=-1, keepdim=True)
    k = ang_vel / torch.clamp(wn, min=1e-9)
    phi = -wn * tau
    return vec * torch.cos(phi) + torch.cross(k, vec, dim=-1) * torch.sin(phi) \
        + k * torch.sum(k * vec, dim=-1, keepdim=True) * (1.0 - torch.cos(phi))


def _lean_spout(src_pos, src_up, src_mouth, rcv_mouth, lip_r, cup_r, geo):
    """G = spout * lean of ONE state, rebuilt with the ctx formulas: d = horizontal unit vector
    source cup origin -> receiver mouth, theta = signed tilt toward d in the (d, z) plane,
    lip = source mouth + r (cos theta d - sin theta z), lean = max(up . d, 0)."""
    h = rcv_mouth - src_pos
    hn = torch.clamp(torch.sqrt(h[:, 0] ** 2 + h[:, 1] ** 2), min=1e-6)
    dx, dy = h[:, 0] / hn, h[:, 1] / hn
    toward = src_up[:, 0] * dx + src_up[:, 1] * dy                      # axis component toward d
    theta = torch.atan2(toward, src_up[:, 2])
    c, s = torch.cos(theta), torch.sin(theta)
    lip = src_mouth + lip_r[:, None] * torch.stack([c * dx, c * dy, -s], dim=-1)
    _, fine, height = _spout(lip, rcv_mouth, cup_r, geo)
    return fine * height * torch.clamp(toward, min=0.0)


def _lean_spout_rate(ctx, lip_r, cup_r, geo, tau, dt):
    """(G(now) - G(both cups moved back by tau along their lin/ang velocities)) / tau [1/s].
    The receiver mouth, d, theta and the lip are all rebuilt at the earlier state, so this is the
    FULL rate of G. (hypothesis, as in iter_26: velocities are in the env frame of the positions)"""
    sp, su = ctx.src_cup_pos.to(dt), ctx.src_cup_up.to(dt)
    sm, rp, rm = ctx.src_cup_mouth_pos.to(dt), ctx.rcv_cup_pos.to(dt), ctx.rcv_cup_mouth_pos.to(dt)
    vs, ws = ctx.src_cup_lin_vel.to(dt), ctx.src_cup_ang_vel.to(dt)
    vr, wr = ctx.rcv_cup_lin_vel.to(dt), ctx.rcv_cup_ang_vel.to(dt)
    g_now = _lean_spout(sp, su, sm, rm, lip_r, cup_r, geo)
    sp_b = sp - tau * vs
    g_back = _lean_spout(sp_b, _turned(su, ws, tau), sp_b + _turned(sm - sp, ws, tau),
                         rp - tau * vr + _turned(rm - rp, wr, tau), lip_r, cup_r, geo)
    return (g_now - g_back) / tau


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dt = ctx.src_palm_pos.dtype

    # =============================== weights (iter_26 unless marked iter_27) =================
    APPROACH_W, APPROACH_K = 0.6, 4.0      # reach income, small: it must not compete with pouring
    GRASP_W, CLOSE_W, CLOSE_NORM = 0.8, 0.8, 0.35
    LIFT_SRC_W, LIFT_RCV_W = 3.0, 1.5      # iter_24 levels: these lifted reliably, do not weaken
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03   # carry heights: workspace, not cup sizes
    POSTURE_W_SRC, POSTURE_W_RCV = 3.0, 1.2    # before the lift: build the envelope (now all fingers)
    KEEP_W_SRC, KEEP_W_RCV = 1.5, 1.5          # after the lift: keep THAT grasp (iter_24 design)
    KEEP_POS, KEEP_MAG = 0.02, 0.7             # [m] drift and |log force ratio| at which sim = 0.37
    FINGER_ON_SHARE = 0.02     # "touched at lift" = force share of the lift hold above this
    # iter_27 change 3: a lift finger still counts if it presses with >= half of its OWN lift force
    # (replaces the absolute 0.2 N floor). Losing 1 of 5 costs KEEP_W * keep * 0.2 (~0.19 per step).
    KEEP_FORCE_FRAC = 0.5
    AIM_W, AIM_K, AIM_FAR = 5.0, 10.0, 0.40    # workspace-scale guidance toward the receiver
    AIM_PREC_R = 0.5       # gaussian width of the spout offset inside the receiver mouth [cup radii]
    DROP_R = 1.5           # spout height above the receiver rim at which low_drop = 0.37
    BELOW_R = 1.0          # spout below the receiver rim
    DROP_FLOOR = 0.4       # a high spout keeps 40 % of the aim income (collision margin)
    # iter_27 change 1: rotation help = TILT_W * rate of Phi = latches * spout * lean [1/s], summed
    # per step. Pay gated by rate-free factors and capped at +TILT_PAY_CAP per step, charge in full.
    TILT_W, TILT_PAY_CAP = 5.0, 1.0
    # back-shift of the rate [s]. Hypothesis: one physics step. Equal to the physics step, the
    # per-step sum of rates telescopes (sampling bias 0); measured worst sampling pump for physics
    # steps 1/240..1/60 s under adversarial oscillation: <= 0.034 per step (cap and charge aside).
    TILT_TAU = 1.0 / 120.0
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 60.0, 0.03, -0.05   # paid per unit of bead fraction that ARRIVES
    OVER_DELTA_W = 120.0       # 2x FLOW_W: a bead past the request costs more than it ever paid
    OVER_LEVEL_W = 4.0         # sustained charge for sitting on an overshoot
    SPILL_W = 50.0             # per REQUEST-worth of spilled beads, floored at OVER_DELTA_W
    FINISH_W = 10.0            # sustained: request closed + cup upright at the pre-tilt pose + held
    FIN_UP_FLOOR, FIN_POSE_FLOOR = 0.5, 0.4
    POSE_TOL_R, POSE_SIG_R = 2.0, 2.0   # plateau beyond ctx.cup_mouth_z / gaussian falloff [cup radii]
    SUCCESS_W = 12.0           # env verdict on top of it
    HOME_W = 1.0
    CENTER_W, CENTER_BAND, CENTER_SCALE = 1.5, 0.05, 0.10   # source side centre-line term
    SIDE_VOID, SIDE_OK = 0.03, 0.08     # env voids success below 3 cm receiver side margin
    DIR_ZERO = 0.25                     # env voids success at pour_dir_xy[:,0] > 0.3
    PRE_W, PRE_LO, LATCH_W = 2.0, 0.7, 1.0     # premature tilt: FRACTION of ctx.premature_tilt_limit
    CONTACT_W, HIT_W, HIT_LATCH_W = 2.0, 5.0, 0.3
    FOREIGN_DEAD, FOREIGN_W_FREE, FOREIGN_W_HELD = 1.0, 0.3, 1.5
    RATE_W_BASE, RATE_W_FULL, FLIP_W = 0.005, 0.03, 0.4
    RCV_CONTROL_MULT = 2.0     # receiver arm trembled more -> charge it harder
    TREMOR_SRC_W, TREMOR_RCV_W = 0.3, 0.9      # on the measured arm joint speeds
    TREMOR_FREE, TREMOR_SCALE = 0.20, 0.50     # [rad/s]
    ADR_RAMP_FULL = 0.2        # control charges reach full weight at adr_progress 0.2
    SETTLE_FRAC = 0.05

    # =============================== basic state (unchanged) =================================
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    off_src = _step((h_src - 0.01) / 0.04)          # lift-off ramp (reward shape, not an asset size)
    off_rcv = _step((h_rcv - 0.01) / 0.04)
    held_src = torch.maximum(g_src, off_src)        # a flickering contact flag must not kill income
    held_rcv = torch.maximum(g_rcv, off_rcv)
    carry_src = held_src * off_src
    carry_rcv = held_rcv * off_rcv
    carry_both = carry_src * carry_rcv
    not_nested = (~ctx.cups_nested).to(dt)
    settled = _step(ctx.episode_progress.to(dt) / SETTLE_FRAC)
    up_src = torch.clamp(ctx.src_cup_up[:, 2], 0.0, 1.0)             # 1 = upright, 0 = horizontal
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], 0.0, 1.0)
    cup_r = max(float(ctx.cup_radius), 1e-3)                         # geometry unit from ctx
    geo = (AIM_PREC_R, BELOW_R, DROP_R, DROP_FLOOR)

    # grasp quality / grasp maintenance
    q_src = _grasp_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_wrap_count, dt)
    q_rcv = _grasp_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_wrap_count, dt)
    keep_src = _grip_keep(ctx.src_palm_in_cup, ctx.src_tips_in_cup, ctx.src_ref_palm_in_cup,
                          ctx.src_ref_tips_in_cup, ctx.src_force_share, ctx.src_ref_force_share,
                          ctx.src_force_total, ctx.src_ref_force_total, ctx.src_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    keep_rcv = _grip_keep(ctx.rcv_palm_in_cup, ctx.rcv_tips_in_cup, ctx.rcv_ref_palm_in_cup,
                          ctx.rcv_ref_tips_in_cup, ctx.rcv_force_share, ctx.rcv_ref_force_share,
                          ctx.rcv_force_total, ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    # iter_27 change 3: fingers-kept on BOTH hands, relative to each finger's own lift force
    kept_src = _fingers_kept(ctx.src_ref_force_share.to(dt), ctx.src_ref_force_total.to(dt),
                             ctx.src_finger_force.to(dt), ctx.src_grasp_ref_valid,
                             FINGER_ON_SHARE, KEEP_FORCE_FRAC)
    kept_rcv = _fingers_kept(ctx.rcv_ref_force_share.to(dt), ctx.rcv_ref_force_total.to(dt),
                             ctx.rcv_finger_force.to(dt), ctx.rcv_grasp_ref_valid,
                             FINGER_ON_SHARE, KEEP_FORCE_FRAC)
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    grip_gate = 0.3 + 0.7 * keep_src        # a ruined grasp earns a fraction of the pour income

    # =============================== requested amount (unchanged) ============================
    poured = torch.clamp(ctx.bead_in_target_frac.to(dt), 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac.to(dt), 0.0, 1.0)
    in_src = torch.clamp(ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    tol = 1.0 / torch.clamp(ctx.bead_active_count.to(dt), min=1.0)   # env tolerance = ONE bead
    d_in = ctx.d_in_target.to(dt)
    poured_prev = torch.clamp(poured - d_in, min=0.0)
    deficit_prev = torch.clamp(target - poured_prev, min=0.0)        # what was still requested
    err = torch.abs(poured - target)
    on_target = 1.0 - _step((err - tol) / (4.0 * tol))                # 1 within one bead, 0 at five
    need = torch.clamp(torch.minimum(target - poured, in_src), min=0.0)   # spilled beads are gone
    want = _step((need - tol) / tol)          # more than one bead still requested AND available
    closed = 1.0 - want
    spilled = torch.clamp(ctx.bead_spill_frac.to(dt), 0.0, 1.0)
    released = _step((poured + spilled) / tol)   # the first bead has LEFT the source (cumulative)

    # =============================== stage 1-3: reach, grasp, lift ===========================
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = APPROACH_W * torch.exp(-APPROACH_K * d_src)
    approach_rcv = APPROACH_W * torch.exp(-APPROACH_K * d_rcv)
    grasp_src = GRASP_W * held_src + CLOSE_W * torch.exp(-6.0 * d_src) \
        * torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_rcv = GRASP_W * held_rcv + CLOSE_W * torch.exp(-6.0 * d_rcv) \
        * torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    lift_shape_src = 0.5 * (1.0 - torch.exp(-torch.clamp(h_src, min=0.0) / LIFTOFF_H)) \
        + 0.5 * torch.clamp(h_src / LIFT_SRC_H, 0.0, 1.0)
    lift_src = LIFT_SRC_W * held_src * lift_shape_src
    lift_rcv = LIFT_RCV_W * held_rcv * torch.clamp(h_rcv / LIFT_RCV_H, 0.0, 1.0) * up_rcv
    # posture: build the envelope before the lift (change 3: out of all F fingers), KEEP the lift
    # grasp afterwards (iter_24) times the share of lift fingers still pressing (change 3: both hands)
    posture_src = held_src * ((1.0 - v_src_ref) * POSTURE_W_SRC * q_src
                              + v_src_ref * KEEP_W_SRC * keep_src * kept_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * POSTURE_W_RCV * q_rcv
                              + v_rcv_ref * KEEP_W_RCV * keep_rcv * kept_rcv)

    # =============================== stage 4: aim the LIP over the receiver (unchanged) ======
    lip_xy, fine, height = _spout(ctx.src_pour_lip_pos.to(dt), ctx.rcv_cup_mouth_pos.to(dt),
                                  cup_r, geo)
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)     # x > 0.3 voids success
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    aim_xy = 0.2 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) \
        + 0.3 * torch.exp(-AIM_K * lip_xy) + 0.5 * fine
    aim_geo = aim_xy * height
    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_geo * want                            # fades once the request closes

    # =============================== stage 4b: rotation potential (iter_27 change 1) =========
    # Phi = latches * spout * lean. lean = max(up . d, 0) is how far the cup axis leans TOWARD the
    # receiver (no angle constant); spout = fine * height as in the aim. latches = settled *
    # (1 - released) * want only move one way before the first bead (settled rises while the cups
    # stand apart at spawn where spout ~ 0, released is the intended switch-off, want is fixed until
    # a bead leaves), so they scale the rate of spout * lean. The rate is the FULL rate: both cups
    # are moved back by TILT_TAU and the mouth, d, theta and lip are rebuilt. Holding any pose earns
    # 0; a tilt made while spout = 0 pays 0 and costs 0; a lean away or sideways has lean = 0.
    # The pay gate (aim_gate: carry, not_nested, dir_ok, side_ok, up_rcv, grip_gate) has no rate and
    # is NOT inside Phi: it can only shrink the pay, never the charge.
    lip_r = torch.norm(ctx.src_pour_lip_pos - ctx.src_cup_mouth_pos, dim=-1).to(dt)   # r from ctx
    latches = settled * (1.0 - released) * want
    rate = latches * _lean_spout_rate(ctx, lip_r, cup_r, geo, TILT_TAU, dt)
    tilt_rate = torch.where(rate > 0.0,
                            torch.clamp(TILT_W * aim_gate * rate, max=TILT_PAY_CAP),
                            TILT_W * rate)

    # =============================== stage 5: the flow IS the pour (unchanged) ===============
    useful = torch.clamp(torch.minimum(d_in, deficit_prev), FLOW_FLOOR, FLOW_CAP)
    pour_flow = FLOW_W * useful
    overshoot = -OVER_DELTA_W * torch.clamp(d_in - deficit_prev, min=0.0) \
        - OVER_LEVEL_W * torch.clamp(poured - target, min=0.0)
    spill_w = torch.clamp(SPILL_W / torch.maximum(target, tol), min=OVER_DELTA_W)
    spill = -spill_w * settled * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # =============================== stage 6: finish upright at the pre-tilt pose (unchanged) =
    body_xy = torch.norm(ctx.src_cup_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    pose_tol = max(float(ctx.cup_mouth_z), 0.0) + POSE_TOL_R * cup_r
    pre_tilt_pose = not_nested * torch.exp(
        -(torch.clamp(body_xy - pose_tol, min=0.0) / (POSE_SIG_R * cup_r)) ** 2)
    finish = FINISH_W * closed * (0.3 + 0.7 * on_target) \
        * (FIN_UP_FLOOR + (1.0 - FIN_UP_FLOOR) * up_src) \
        * (FIN_POSE_FLOOR + (1.0 - FIN_POSE_FLOOR) * pre_tilt_pose) \
        * carry_both * up_rcv * (0.4 + 0.6 * keep_src)
    success = SUCCESS_W * ctx.success.to(dt)

    # =============================== workspace constraints ===================================
    center_src = -CENTER_W * held_src * torch.tanh(
        torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)
    # iter_27 change 2: home anchor = receiver spawn x at side margin SIDE_OK (was: its spawn).
    # rcv_side_margin moves 1:1 with the receiver's lateral position, so the lateral offset from the
    # anchor is (margin - SIDE_OK). Form, HOME_W, carry_rcv, up_rcv, side_ok, still_rcv unchanged.
    dx_home = ctx.rcv_cup_pos[:, 0] - ctx.rcv_cup_spawn_pos[:, 0]
    dm_home = ctx.rcv_side_margin.to(dt) - SIDE_OK
    d_home = torch.sqrt(dx_home ** 2 + dm_home ** 2 + 1e-12)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05,
                                             min=0.0) / 0.15)
    home_rcv = HOME_W * carry_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv
    limit = torch.clamp(ctx.premature_tilt_limit.to(dt), min=1e-3)
    risk = _step((ctx.src_cup_tilt.to(dt) / limit - PRE_LO) / (1.0 - PRE_LO))
    premature = -PRE_W * held_src * (1.0 - torch.exp(-AIM_K * lip_xy)) * risk \
        - LATCH_W * ctx.premature_tilt.to(dt)

    # =============================== safety (real robot, unchanged) ==========================
    f_cup = ctx.cup_cup_force
    cup_collision = -CONTACT_W * torch.tanh(f_cup / 2.0) \
        - HIT_W * settled * (f_cup > 0.5).to(dt) - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_s = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    f_r = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    hand_foreign = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) \
        * torch.tanh(f_s / 5.0) \
        - (FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) * torch.tanh(f_r / 5.0)

    # =============================== control (ramped on adr_progress, unchanged) =============
    ramp = _step(torch.full_like(up_src, float(ctx.adr_progress) / ADR_RAMP_FULL))
    rate_w = RATE_W_BASE + (RATE_W_FULL - RATE_W_BASE) * ramp
    da = ctx.actions - ctx.prev_actions
    flips = torch.relu(-(ctx.actions * ctx.prev_actions))
    action_rate = -rate_w * (torch.sum(da[:, 0:9] ** 2, dim=-1)
                             + RCV_CONTROL_MULT * torch.sum(da[:, 9:18] ** 2, dim=-1)) \
        - FLIP_W * ramp * (torch.sum(flips[:, 0:9], dim=-1)
                           + RCV_CONTROL_MULT * torch.sum(flips[:, 9:18], dim=-1)) \
        * torch.maximum(held_src, held_rcv)
    qd_s = torch.sqrt(torch.mean(ctx.src_arm_qd ** 2, dim=-1) + 1e-8)
    qd_r = torch.sqrt(torch.mean(ctx.rcv_arm_qd ** 2, dim=-1) + 1e-8)
    tremor = -ramp * (TREMOR_SRC_W * torch.tanh(torch.clamp(qd_s - TREMOR_FREE, min=0.0) / TREMOR_SCALE)
                      + TREMOR_RCV_W * torch.tanh(torch.clamp(qd_r - TREMOR_FREE, min=0.0) / TREMOR_SCALE))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "posture_src": posture_src,
        "posture_rcv": posture_rcv,
        "aim": aim,
        "tilt_rate": tilt_rate,
        "pour_flow": pour_flow,
        "overshoot": overshoot,
        "spill": spill,
        "finish": finish,
        "success": success,
        "center_src": center_src,
        "home_rcv": home_rcv,
        "premature": premature,
        "cup_collision": cup_collision,
        "hand_foreign": hand_foreign,
        "action_rate": action_rate,
        "tremor": tremor,
    }

    reward = torch.zeros_like(approach_src)
    for v in components.values():
        reward = reward + v

    return reward, components
```
