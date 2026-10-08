## 1. What the task means and its stages

This is one continuous reward that has to teach the whole task from random weights, with no stage switch. The stages are:
1. Travel from the far start (0.20-0.42 m away) to the cup with the hand open.
2. Bring the palm level beside the side wall, with the thumb on the far side and the fingers on the near side.
3. Close all fingers into an envelope grasp. The first links should touch, and the thumb tip should sit on the wall below the rim.
4. Lift the cup straight up to the goal height.
5. Carry it to `ctx.goal_pos`.
6. Hold it there upright and at rest. This is the dominant income.
7. Repeat for later goals.

In iter_03 the left arm hovered around the cup for 800 epochs with the palm tilted and the thumb over the mouth. It earned 0.44 per step without ever grasping, because the palm-plane level factor multiplied only the held-grasp quality. Tilting the palm before the grasp cost only 4 % (0.356 vs 0.371).

The only change this round is item 1. A palm-plane level factor `level_pre` now scales all income paid near the cup before the grasp:
- the near part of `approach`
- `open_hand`
- `pre_pose`
- `close_at_pose`
- `approach_orient`

`level_pre` uses the same measure as the held grasp, `s_tilt = sqrt(n_ax^2 + f_ax^2)`. It is a single factor applied to the whole near-cup sum, not chained into a product with `pose_q` or `wrap`. Its floor is 0.5. The far part of the approach is not scaled, so there is always a gradient towards the cup.

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

    # CHANGED (iter_04 item 1): palm-plane level factor for the income paid near the cup BEFORE
    # the grasp, built from the same measure s_tilt. Full credit up to s = 0.2 (level palm,
    # fingers ~10 deg down), linear down to the floor 0.5 at s = 0.5 (palm normal 0.40 out of
    # the horizontal -> s >= 0.40 -> factor <= 0.67). The floor keeps approach income and its
    # gradient towards the cup alive; it is one factor on the near-cup sum, not chained into
    # the pose / wrap quality products.
    level_pre = 0.5 + 0.5 * (1.0 - (s_tilt - 0.2) / 0.3).clamp(0.0, 1.0)

    # thumb tip above the little-finger tip (world z).
    # 1 once the thumb tip is >= 2 cm above, 0 once it is 1 cm below ("a few centimetres").
    thumb_dz = ctx.tips_pos[:, 0, 2] - ctx.tips_pos[:, 4, 2]
    thumb_up = ((thumb_dz + 0.01) / 0.03).clamp(0.0, 1.0)

    # ---------------- contacts (all five fingers treated by the same rule) ----------------
    c = (ctx.finger_force / 1.0).clamp(0.0, 1.0)        # 1 N = env grasp threshold
    thumb_c = c[:, 0]
    fing_c = c[:, 1:].mean(-1)                           # four fingers alike, more is better
    wrap = thumb_c * fing_c                              # needs both sides -> no one-sided push income

    # first-link contact of the four fingers, treated alike.
    # 0.5 N counts a first link as touching (a finger's links share the ~1 N force stop).
    pc = (ctx.finger_prox_force / 0.5).clamp(0.0, 1.0)
    envelope = pc[:, 1:].mean(-1)                        # 0 = fingertip pads only, 1 = all first links on the cup

    h_t, rv_t = _axial_radial(ctx.tips_pos, ctx.cup_pos.unsqueeze(1), up.unsqueeze(1))
    u = _unit(rv_t)
    u_thumb = u[:, 0]
    u_fing = _unit(u[:, 1:].mean(1))
    oppose = 0.5 * (1.0 - (u_thumb * u_fing).sum(-1))   # 1 = thumb on the far side of the fingers
    # the thumb tip must be on the side wall, not over the rim.
    # ramps to 0 over the top 20 % of the mouth height (ctx value, no cup constant)
    mouth = max(float(ctx.cup_mouth_z), 1e-3)
    below_rim = ((mouth - h_t[:, 0]) / (0.2 * mouth)).clamp(0.0, 1.0)
    # the thumb TIP must be on the wall: its distance from the cup axis close to ctx.cup_radius.
    # Full credit within 0.5 r of the wall, 0 at 1.0 r off the wall.
    r_thumb = rv_t[:, 0].norm(dim=-1)
    on_wall = (1.0 - ((r_thumb - rad).abs() - 0.5 * rad) / (0.5 * rad)).clamp(0.0, 1.0)
    oppose_q = oppose * below_rim * on_wall

    closure = ctx.hand_closure.clamp(0.0, 1.0)
    open_ = 1.0 - closure

    # uprightness (same factor as lift / transport / hold)
    upright = torch.cos(ctx.cup_tilt).clamp(min=0.0) ** 2

    # ---------------- travel (far pull never multiplied, G2) ----------------
    # CHANGED (iter_04 item 1): only the near-cup part (0.4 near_k) is scaled by level_pre.
    approach = free * 0.35 * (0.6 * far_k + 0.4 * level_pre * near_k)
    open_hand = free * level_pre * 0.08 * open_ * near_k        # only near the cup -> no far parking spot
    pre_pose = free * level_pre * 0.07 * near_k * pose_q
    close_at_pose = free * level_pre * 0.10 * closure * at_pose * pose_q   # closing is not a loss at the pose

    # approach orientation, ADDED (never multiplied onto the far approach, G2), not held only.
    # (a) palm normal perpendicular to gravity, (b) palm normal pointing at the cup horizontally,
    # (c) thumb tip above the little-finger tip. Ceiling 0.14 < the ~0.3 the approach pays for
    # closing the distance, so getting closer always pays more than turning the hand.
    level = 1.0 - n[:, 2].abs()                          # 1 = horizontal palm normal (0.2 at normal z 0.8)
    to_cup_h = _unit(_horizontal(ctx.cup_pos - ctx.palm_pos))
    face_cos = (_horizontal(n) * to_cup_h).sum(-1)       # >0 facing the cup, <0 facing away
    facing = 0.5 * (1.0 + torch.tanh(face_cos / 0.2))    # ~1 facing the cup, ~0 facing away
    # fades to half in the last few cm (near_k), where the grasp should take over.
    orient_fade = 1.0 - 0.5 * near_k
    # CHANGED (iter_04 item 1): scaled by the same level_pre factor.
    approach_orient = free * level_pre * orient_fade * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    # no flipped grasp; 1 with the thumb >= 2 cm above the little finger, floor 0.25 below
    hand_up = 0.25 + 0.75 * thumb_up

    # ---------------- grasp (ceiling ~1.33, >= 2x the pre-grasp step) ----------------
    grasp_contact = 0.10 * upright * hand_up * wrap      # bridges 0 -> 1 N before the grasp flag
    # base 0.65 is paid for ANY grasp (flag on, cup upright) and is not shrunk by the quality
    # terms; quality is paid ON TOP: wrap 0.08, pose 0.10, opposition (thumb tip on the wall,
    # below the rim) 0.20, first-link envelope 0.20, scaled by the palm-plane level factor.
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * oppose_q + 0.20 * envelope
    grasp_hold = held * upright * (0.65 + hand_up * plane_q * grasp_quality)

    # ---------------- lift / transport / hold ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)               # cup off the table
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)

    # lowering a held cup lowers this state income -> worth less than keeping still
    lift = 2.5 * held * upright * (height.clamp(min=0.0) / goal_h).clamp(max=1.0)

    carry_gate = held * off * upright                    # no goal income while on the table
    kp = ctx.goal_kp_dist
    transport = 2.5 * carry_gate * (0.6 * torch.exp(-kp / 0.10) + 0.4 * torch.exp(-kp / 0.03))

    tol = max(float(ctx.goal_tol), 1e-3)
    prox = torch.exp(-kp / tol)                          # sharpens as the curriculum shrinks tol
    cup_v = ctx.cup_lin_vel.norm(dim=-1)
    still = torch.exp(-cup_v / 0.05)
    goal_hold = 10.0 * carry_gate * prox * (0.5 + 0.5 * still)   # dominant income

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

## 근거 메모

가설 id 미지정: brief 에 가설 id 줄이 없어 prompt 전체(Required change 1개)를 한 후보로 구현했다. 바뀐 곳은 `level_pre` 정의 하나와, 그 인자를 곱한 5개 항뿐이다. 나머지는 iter_03 c1r 과 같다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| `level_pre` (신규 인자, 바닥 0.5, s_tilt 0.2~0.5 선형) | 파지 전 컵 근처에서 손바닥이 기울면 수입이 4 % 만 줄었다(0.356 vs 0.371). 왼팔이 800 epoch 동안 기울어진 채 컵 주변에 떠 있었다. 요구는 "쥔 파지와 같은 척도, 바닥 약 절반, 다른 품질 인자와 곱 사슬 금지"다 | prompt.md:405-408, 416 |
| `approach` 가까운 부분 0.4·near_k 에 `level_pre` | "near-cup part of the approach" 를 스케일한다. 먼 부분 0.6·far_k 는 곱셈 인자 없이 둔다(G2, 먼 접근 기울기 유지) | prompt.md:134, 416 |
| `open_hand`, `pre_pose`, `close_at_pose`, `approach_orient` 에 `level_pre` | Required change 가 열거한 항들 | prompt.md:416 |
| ramp 폭 0.3 (쥔 파지 `plane_q` 는 0.5) | 부등식 여유를 위해 폭을 좁혔다(추정). 법선 z 0.40 이면 s ≥ 0.40 이라 인자가 0.67 이하가 된다. 직전 파지 자리 수입 중 가까운 부분(약 0.19 로 추정)이 1/3 줄어 기울어진 떠 있기 행이 수평 행의 90 % 밑으로 간다. 이 비율은 추정이고 드라이런에서 확인해야 한다 | prompt.md:100, 408, 416 |
| 수평 손(s ≤ 0.2) 인자 1 | 시작 자세(손가락 10° 아래, s 약 0.17)와 올바른 방향의 모든 행은 값이 그대로다. 그래서 단조 사다리, 주차 행, 엉성한 첫 파지 ≥ 2x 주차 행이 유지된다 | prompt.md:5, 100, 416 |
| `grasp_hold`, `lift`, `transport`, `goal_hold`, `success_bonus`, 벌점들 | 변경 없음("Keep every other part") | prompt.md:418 |
