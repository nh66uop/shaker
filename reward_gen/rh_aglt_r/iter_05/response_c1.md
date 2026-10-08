brief 에 가설 id 줄이 없어서 prompt 전체를 한 후보로 구현했다. 요구 변경 두 가지(prompt.md:422, :424)만 반영했다. 파일은 쓰지 않았고 명령도 실행하지 않았다. 아래 행 비교(원장 90 % 기준 등)는 가정한 상태를 손으로 계산한 추정이고, 드라이런 게이트로 확인한 값이 아니다.

## 1. 과제와 단계

1. 먼 출발(열린 손, 손바닥 수평)에서 컵으로 이동한다.
2. 손바닥을 컵 옆벽 쪽 파지 링(축에서 1.5 r, 높이 원점 + 0.15 x 입구)에 둔다. 엄지는 먼 쪽 바깥 벽, 손가락 넷은 가까운 쪽에 둔다.
3. 손가락을 닫아 감싸 쥔다. 네 손가락 모두 닿고, 첫마디도 닿는다.
4. 컵을 곧게 든다.
5. `ctx.goal_pos` 로 옮긴다.
6. 목표에서 곧은 자세로 쥔 채 멈춰 있는다.
7. 목표가 바뀌면 되풀이한다.

iter_05 에서 바꾼 것:
- (1) 입구 안에 넣은 엄지는 파지로 치지 않는다.
- (2) 파지 모양(새끼손가락까지 네 손가락 접촉, 손가락 수평, 엄지가 바깥 벽에서 맞섬)이 파지 뒤 수입 전체(grasp_hold, lift, transport, goal_hold)를 바닥값 있는 인자로 조정한다.

## 2. 코드

```python
import torch
import math


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _axial_radial(p: torch.Tensor, cup_pos: torch.Tensor, up: torch.Tensor):
    # cylinder coordinates of point(s) p around the cup axis (broadcasts over a finger dim)
    v = p - cup_pos
    h = (v * up).sum(-1)
    rv = v - h.unsqueeze(-1) * up
    return h, rv


def _horizontal(v: torch.Tensor) -> torch.Tensor:
    # world-horizontal part of a (N,3) vector (z component removed)
    return torch.stack([v[:, 0], v[:, 1], torch.zeros_like(v[:, 0])], dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    up = _unit(ctx.cup_up)
    held = ctx.grasped.float()
    free = 1.0 - held                                   # approach income fades once held
    rad = max(float(ctx.cup_radius), 1e-3)
    mouth = max(float(ctx.cup_mouth_z), 1e-3)

    # ---------------- palm vs. the measured grasp ring ----------------
    h_p, rv_p = _axial_radial(ctx.palm_pos, ctx.cup_pos, up)
    r_p = rv_p.norm(dim=-1)
    h_target = 0.15 * ctx.cup_mouth_z                   # grasp height: origin + 0.15 x mouth
    d_ring = torch.sqrt((r_p - 1.5 * rad) ** 2 + (h_p - h_target) ** 2 + 1e-12)

    far_k = 1.0 - torch.tanh(d_ring / 0.25)             # long-range pull from the far start
    near_k = 1.0 - torch.tanh(d_ring / 0.05)            # last few cm
    at_pose = torch.exp(-d_ring / 0.015)                # essentially at the grasp pose

    # ---------------- hand orientation quality (average ramp, not a min/product) ----------------
    n = _unit(ctx.palm_axes[:, 0:3])
    f = _unit(ctx.palm_axes[:, 3:6])
    n_ax = (n * up).sum(-1)
    f_ax = (f * up).sum(-1)
    horiz = ((1.0 - n_ax ** 2) * (1.0 - f_ax ** 2)).clamp(0.0, 1.0)   # both axes horizontal
    w = 0.67 * n + 0.73 * f                              # measured palm->axis direction
    w_h = _unit(w - (w * up).sum(-1, keepdim=True) * up)
    to_axis = _unit(-rv_p)
    align = 0.5 * (1.0 + (to_axis * w_h).sum(-1))
    pos_q = torch.exp(-d_ring / 0.02)
    pose_q = (horiz + align + pos_q) / 3.0               # vertical pinch from above scores low

    # palm-plane level factor for the held grasp quality.
    # s = combined out-of-plane component of palm normal and finger direction (vs the cup axis).
    # Full credit up to s = 0.2 (e.g. fingers ~10 deg down); 25 deg tilt of both axes
    # (s ~ 0.59) -> ~0.21; fingers pointing straight down (s >= 0.7) -> 0.
    s_tilt = torch.sqrt(n_ax ** 2 + f_ax ** 2 + 1e-12)
    plane_q = (1.0 - (s_tilt - 0.2) / 0.5).clamp(0.0, 1.0)

    # steeper ramp on the same measure s_tilt: 1 up to s = 0.2 (fingers ~10 deg down),
    # 0 at s = 0.5; fingers 23 deg down (s ~ 0.39) -> ~0.37.
    # CHANGED (iter_05 item 2): also used as part (b) of the grasp-shape factor.
    plane_level = (1.0 - (s_tilt - 0.2) / 0.3).clamp(0.0, 1.0)
    # iter_04 near-cup level factor, numerically unchanged: floor 0.5, full credit up to s = 0.2
    level_pre = 0.5 + 0.5 * plane_level

    # thumb tip above the little-finger tip (world z).
    # 1 once the thumb tip is >= 2 cm above, 0 once it is 1 cm below ("a few centimetres").
    thumb_dz = ctx.tips_pos[:, 0, 2] - ctx.tips_pos[:, 4, 2]
    thumb_up = ((thumb_dz + 0.01) / 0.03).clamp(0.0, 1.0)

    # ---------------- fingertips in cup cylinder coordinates ----------------
    h_t, rv_t = _axial_radial(ctx.tips_pos, ctx.cup_pos.unsqueeze(1), up.unsqueeze(1))
    r_t = rv_t.norm(dim=-1)                              # (N,5) distance of each tip from the axis
    r_thumb = r_t[:, 0]
    h_thumb = h_t[:, 0]

    # CHANGED (iter_05 item 1): continuous "thumb tip inside the mouth" weight.
    # radially inside the wall: 0 at r >= cup_radius, 1 at r <= 0.8 x cup_radius;
    # near or above the mouth: 0 at h <= 0.4 x mouth height, 1 at h >= 0.7 x mouth height.
    # The height ramp keeps a thumb on the lower, narrower part of the outer wall at weight 0
    # (the cup is tapered, cup_radius is its widest radius).
    in_r = ((rad - r_thumb) / (0.2 * rad)).clamp(0.0, 1.0)
    in_h = ((h_thumb - 0.4 * mouth) / (0.3 * mouth)).clamp(0.0, 1.0)
    thumb_inside = in_r * in_h
    thumb_out = 1.0 - thumb_inside                       # outside-only factor

    # ---------------- contacts (all five fingers treated by the same rule) ----------------
    c = (ctx.finger_force / 1.0).clamp(0.0, 1.0)        # 1 N = env grasp threshold
    # CHANGED (iter_05 item 1): the thumb's contact counts only in proportion to (1 - inside)
    thumb_c = c[:, 0] * thumb_out
    fing_c = c[:, 1:].mean(-1)                           # four fingers alike, more is better
    wrap = thumb_c * fing_c                              # needs both sides -> no one-sided push income

    # first-link contact of the four fingers, treated alike.
    # 0.5 N counts a first link as touching (a finger's links share the ~1 N force stop).
    pc = (ctx.finger_prox_force / 0.5).clamp(0.0, 1.0)
    envelope = pc[:, 1:].mean(-1)                        # 0 = fingertip pads only, 1 = all first links on the cup

    u = _unit(rv_t)
    u_thumb = u[:, 0]
    u_fing = _unit(u[:, 1:].mean(1))
    oppose = 0.5 * (1.0 - (u_thumb * u_fing).sum(-1))   # 1 = thumb on the far side of the fingers
    # the thumb tip must be on the side wall, not over the rim.
    # ramps to 0 over the top 20 % of the mouth height (ctx value, no cup constant)
    below_rim = ((mouth - h_thumb) / (0.2 * mouth)).clamp(0.0, 1.0)
    # thumb TIP on the wall: full credit within 0.5 r of the wall, 0 at 1.0 r off the wall.
    # CHANGED (iter_05 item 1): OUTER side only - a tip inside the mouth earns nothing.
    on_wall_band = (1.0 - ((r_thumb - rad).abs() - 0.5 * rad) / (0.5 * rad)).clamp(0.0, 1.0)
    on_wall = on_wall_band * thumb_out
    oppose_q = oppose * below_rim * on_wall              # thumb outside, on the wall, opposite

    # ---------------- CHANGED (iter_05 item 2): one grasp-shape factor ----------------
    # (a) all four fingers touching, weighted towards the weakest finger:
    #     mean/min blend -> one finger missing gives 0.375 (a drop of 0.625, not 0.25);
    #     the mean part keeps a gradient where the min is flat.
    fc = c[:, 1:]
    fingers_all = 0.5 * fc.mean(-1) + 0.5 * fc.min(-1).values
    # (b) finger direction level: plane_level (10 deg down -> 1, 23 deg down -> ~0.37)
    # (c) thumb outside and opposite: oppose_q (contains the outside-only factor once)
    parts = torch.stack([fingers_all, plane_level, oppose_q], dim=-1)
    shape = 0.5 * parts.mean(-1) + 0.5 * parts.min(-1).values
    # floored: 0.6 floor so the gradient to lift and carry never vanishes;
    # little finger off / fingers 23 deg down -> ~0.83, thumb inside the mouth -> ~0.73
    shape_f = 0.6 + 0.4 * shape

    closure = ctx.hand_closure.clamp(0.0, 1.0)
    open_ = 1.0 - closure

    # uprightness (same factor as lift / transport / hold)
    upright = torch.cos(ctx.cup_tilt).clamp(min=0.0) ** 2

    # ---------------- travel (far pull never multiplied, G2) ----------------
    # only the near-cup part (0.4 near_k) is scaled by level_pre.
    approach = free * 0.35 * (0.6 * far_k + 0.4 * level_pre * near_k)
    open_hand = free * level_pre * 0.08 * open_ * near_k        # only near the cup -> no far parking spot
    pre_pose = free * level_pre * 0.07 * near_k * pose_q
    close_at_pose = free * level_pre * 0.10 * closure * at_pose * pose_q   # closing is not a loss at the pose

    # approach orientation, added (never multiplied onto the far approach, G2), not held only.
    # (a) palm normal perpendicular to gravity, (b) palm normal pointing at the cup horizontally,
    # (c) thumb tip above the little-finger tip. Ceiling 0.14 < the ~0.3 the approach pays for
    # closing the distance, so getting closer always pays more than turning the hand.
    level = 1.0 - n[:, 2].abs()                          # 1 = horizontal palm normal (0.2 at normal z 0.8)
    to_cup_h = _unit(_horizontal(ctx.cup_pos - ctx.palm_pos))
    face_cos = (_horizontal(n) * to_cup_h).sum(-1)       # >0 facing the cup, <0 facing away
    facing = 0.5 * (1.0 + torch.tanh(face_cos / 0.2))    # ~1 facing the cup, ~0 facing away
    # fades to half in the last few cm (near_k), where the grasp should take over.
    orient_fade = 1.0 - 0.5 * near_k
    approach_orient = free * level_pre * orient_fade * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    # no flipped grasp; 1 with the thumb >= 2 cm above the little finger, floor 0.25 below
    hand_up = 0.25 + 0.75 * thumb_up

    # ---------------- grasp (>= 2x the pre-grasp step) ----------------
    grasp_contact = 0.10 * upright * hand_up * wrap      # bridges 0 -> 1 N before the grasp flag
    # quality on top of the base: wrap 0.08, pose 0.10, opposition (thumb tip on the OUTER wall,
    # below the rim) 0.20, first-link envelope 0.20, scaled by the palm-plane level factor.
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * oppose_q + 0.20 * envelope
    # CHANGED (iter_05 item 2): whole grasp_hold scaled by the floored shape factor.
    # Base 0.65 -> 1.05 so that a sloppy first grasp (shape_f ~0.675) still earns at least what
    # it earned in iter_04 (keeps "sloppy >= 2x parked"); envelope grasp ~1.6-1.7 per step.
    grasp_hold = held * upright * shape_f * (1.05 + hand_up * plane_q * grasp_quality)

    # ---------------- lift / transport / hold ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)               # cup off the table
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)

    # lowering a held cup lowers this state income -> worth less than keeping still
    # CHANGED (iter_05 item 2): x shape_f
    lift = 2.5 * held * upright * shape_f * (height.clamp(min=0.0) / goal_h).clamp(max=1.0)

    carry_gate = held * off * upright                    # no goal income while on the table
    kp = ctx.goal_kp_dist
    # CHANGED (iter_05 item 2): x shape_f
    transport = 2.5 * carry_gate * shape_f * (0.6 * torch.exp(-kp / 0.10) + 0.4 * torch.exp(-kp / 0.03))

    tol = max(float(ctx.goal_tol), 1e-3)
    prox = torch.exp(-kp / tol)                          # sharpens as the curriculum shrinks tol
    cup_v = ctx.cup_lin_vel.norm(dim=-1)
    still = torch.exp(-cup_v / 0.05)
    # CHANGED (iter_05 item 2): x shape_f; still the dominant income
    goal_hold = 10.0 * carry_gate * shape_f * prox * (0.5 + 0.5 * still)

    first = (ctx.goal_success & (ctx.goal_successes <= 1)).float()
    later = (ctx.goal_success & (ctx.goal_successes > 1)).float()
    success_bonus = 60.0 * first + 3.0 * later

    # ---------------- short-lived, bounded penalties ----------------
    cup_w = ctx.cup_ang_vel.norm(dim=-1)
    # tipping: charged only when the cup is tilted beyond a few degrees AND its axis is actually
    # moving (faster than resting solver jitter), weighted up while the tilt is increasing.
    # A standing cup (tilt ~0) earns exactly 0 here regardless of its angular-velocity jitter.
    tilt = ctx.cup_tilt
    tilt_gate = torch.tanh((tilt - 0.05).clamp(min=0.0) / 0.1)        # ~3 deg deadband, ~1 by 20 deg
    axis_dot = torch.cross(ctx.cup_ang_vel, up, dim=-1)               # d(cup axis)/dt
    axis_rate = axis_dot.norm(dim=-1)
    axis_move = torch.tanh((axis_rate - 0.2).clamp(min=0.0) / 0.3)    # 0.2 rad/s: above resting jitter
    tilt_rate = -axis_dot[:, 2] / torch.sin(tilt).clamp(min=1e-3)     # d(tilt)/dt from d(up_z)/dt
    rising = torch.tanh(tilt_rate.clamp(min=0.0) / 0.3)
    tip_penalty = -0.6 * free * (1.0 - off) * tilt_gate * axis_move * (0.5 + 0.5 * rising)
    # charged only while a lifted, released cup is falling (not proportional to `lifted` itself)
    fall_speed = (-ctx.cup_lin_vel[:, 2]).clamp(min=0.0)
    drop_penalty = -1.0 * ctx.lifted.float() * free * torch.tanh(fall_speed / 0.5)
    # small: lifting / carrying must always beat standing still
    motion_penalty = -held * (0.10 * torch.tanh(cup_v / 0.5)
                              + 0.05 * torch.tanh(cup_w / 2.0)
                              + 0.10 * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0))
    action_rate = -0.01 * ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    foreign_contact = -0.10 * torch.tanh(ctx.hand_foreign_force / 5.0)

    comps = {
        "approach": approach,
        "open_hand": open_hand,
        "pre_pose": pre_pose,
        "close_at_pose": close_at_pose,
        "approach_orient": approach_orient,
        "grasp_contact": grasp_contact,
        "grasp_hold": grasp_hold,
        "lift": lift,
        "transport": transport,
        "goal_hold": goal_hold,
        "success_bonus": success_bonus,
        "tip_penalty": tip_penalty,
        "drop_penalty": drop_penalty,
        "motion_penalty": motion_penalty,
        "action_rate": action_rate,
        "foreign_contact": foreign_contact,
    }
    reward = torch.zeros_like(approach)
    for v in comps.values():
        reward = reward + v
    return reward, comps
```

## 3. 근거 메모

구현 범위: brief 에 가설 id 줄이 없어서 prompt 전체(iter_05 요구 변경 1, 2)를 한 후보로 구현했다. 나머지 항은 iter_04 그대로다.

| 보상 항 / 변경 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| thumb_inside, thumb_out: 반경 안쪽(r 1.0에서 0.8 r) x 입구 근처 높이(0.4에서 0.7 x 입구) 램프 | 우팔 ep4600 에서 엄지를 입구 안에 넣었고, 원장에서 엄지 안쪽이 99.2 % / 90.0 % 로 벌이가 거의 같았다. 모든 길이는 r 과 입구 높이의 배수다. | prompt.md:414-416, 422 |
| thumb_c = c_thumb x thumb_out (wrap, grasp_contact, grasp_quality 에 반영) | 엄지 접촉력은 (1 - inside) 비율로만 파지 수입에 들어간다. | prompt.md:422 |
| on_wall = 벽 띠 x thumb_out (바깥쪽만 인정) | 기존 on_wall 은 \|r - r_cup\| 기반이라 벽 안쪽 0.5 r 에서도 만점이었다. | prompt.md:415, 422 |
| 높이 램프로 아래쪽 벽 오탐을 피함 | 컵이 테이퍼라서 아래쪽 바깥 벽은 r < cup_radius 다(가설: 바깥 벽을 쥔 엄지 끝이 그 높이에 있을 수 있음). | prompt.md:148 (R1 테이퍼) |
| fingers_all = 0.5 mean + 0.5 min (네 손가락 같은 규칙) | 새끼손가락 접촉 0.011 / 0.014 였다. 한 손가락이 빠지면 1/4 보다 훨씬 크게(0.625) 떨어진다. min 이 평평한 곳은 mean 이 기울기를 준다. | prompt.md:412-413, 424; 169 |
| plane_level (같은 s_tilt, 0.2 에서 0.5 로 가파른 램프) | 손가락 z -0.38 / -0.39(약 23도 아래)였고, 원장에서 98.6 % 로 거의 같았다. 이제 23도는 약 0.37, 10도는 1 이다. level_pre 값은 그대로다. | prompt.md:412-415, 424 |
| shape = 부분 셋의 mean/min 혼합, shape_f = 0.6 + 0.4 shape | 하나의 모양 인자에 바닥 0.6 을 두어 들기·이송 기울기가 사라지지 않는다. 새끼 빠짐·23도는 약 0.83, 엄지 안쪽은 약 0.73 으로 목표 행이 90 % 미만이다(추정 계산). | prompt.md:424; 140 (G8) |
| lift, transport, goal_hold, grasp_hold 에 x shape_f | 파지 뒤 수입 약 15/16.3 이 파지 모양과 무관했다. | prompt.md:415, 424 |
| grasp_hold 기저 0.65 에서 1.05 로 상향 | shape_f 를 곱하면 엉성한 첫 파지(shape_f 약 0.675)가 iter_04 벌이 아래로 떨어져 "엉성한 파지 >= 2x 주차" 행이 깨진다. 1.05 면 iter_04 이상을 유지한다(추정 계산). 부작용: 인벨롭 파지 한 스텝 약 1.7 로 천장 "about 1.2" 를 넘는다. 손끝 패드만 행은 약 0.88 로 90 % 기준에 가깝다(추정). | prompt.md:81, 100, 424 |
| 나머지 항(approach, open_hand, pre_pose, close_at_pose, approach_orient, success_bonus, 벌점 5종) | 바꾸지 말라는 요구에 따라 iter_04 그대로 두었다. | prompt.md:426 |

규칙 충돌: 파지 천장 "about 1.2"(prompt.md:81)와 "sloppy >= 2x parked 유지"(prompt.md:424)가 함께 걸린다. 원장 행 유지가 더 보수적이라고 보고 기저를 올렸다.
