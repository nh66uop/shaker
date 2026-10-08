brief 에 가설 id 줄이 없어서 prompt 전체를 한 후보로 구현했습니다. prompt 에는 Hypothesis A/B 표기가 없고 "Required changes" 1·2 만 있습니다. 재시도 범위는 Review item 1 하나입니다. 손가락 파지 전 엄지 유도 항 `thumb_steer` 만 고쳤습니다. Review item 2 에 따라 `thumb_f`, `post_f`, `grasp_contact`, `grasp_hold`, `lift`, `transport`, `goal_hold` 는 이전 응답과 같습니다. 코드는 실행하지 않았고, ledger 행(특히 sloppy >= 2x hover)의 통과 여부는 아래 끝의 추정입니다.

1. 단계별 과제: 먼 출발에서 편 손으로 이동한다. 손바닥을 컵 옆벽의 측정 파지 링(축에서 1.5 r, 원점 + 0.15 x 입구 높이)에 수평으로 댄다. 엄지는 네 손가락 반대편의 바깥 벽, 입구 아래에 둔다. 손가락을 닫아 감싸 쥐고, 곧게 들고, `ctx.goal_pos` 로 옮긴 뒤 쥔 채 정지 유지한다. 첫 성공 보너스가 크다.

2. 수정: 이전 `thumb_steer` 는 `thumb_place = outer_wall * below_rim * box_attract` 를 써서 테두리 위와 벽 안쪽에서 값과 기울기가 정확히 0 이었습니다(Review item 1). 이제 유도는 같은 목표 상자(1.15-1.45 r, 입구 높이의 0.45-0.8)로의 연속 끌림 `box_attract` 만 씁니다. 안쪽/테두리 위 램프는 곱하지 않습니다. 천장 0.04, near_k·level_pre·free 곱, 반대편 가중 `(0.4 + 0.6 * oppose)` 는 그대로입니다.

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
    # also used as part (b) of the grasp-shape factor.
    plane_level = (1.0 - (s_tilt - 0.2) / 0.3).clamp(0.0, 1.0)
    # near-cup level factor: floor 0.5, full credit up to s = 0.2
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

    # "thumb tip inside the mouth" weight (unchanged from iter_06):
    # radially: 1 at r <= 1.0 x cup_radius, 0 at r >= 1.1 x cup_radius (continuous ramp);
    # height: 0 at h <= 0.4 x mouth height, 1 at h >= 0.7 x mouth height.
    in_r = ((1.1 * rad - r_thumb) / (0.1 * rad)).clamp(0.0, 1.0)
    in_h = ((h_thumb - 0.4 * mouth) / (0.3 * mouth)).clamp(0.0, 1.0)
    thumb_inside = in_r * in_h
    thumb_out = 1.0 - thumb_inside                       # outside-only factor

    # ---------------- contacts (all five fingers treated by the same rule) ----------------
    c = (ctx.finger_force / 1.0).clamp(0.0, 1.0)        # 1 N = env grasp threshold
    # the thumb's contact counts only in proportion to (1 - inside)
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
    # thumb-tip band centred on the outer-wall tip position (unchanged from iter_06)
    wall_tip_r = 1.18 * rad
    on_wall_band = (1.0 - ((r_thumb - wall_tip_r).abs() - 0.10 * rad) / (0.12 * rad)).clamp(0.0, 1.0)
    on_wall = on_wall_band * thumb_out
    oppose_q = oppose * below_rim * on_wall              # thumb outside, on the wall, opposite

    # ---------------- thumb-placement target box (iter_07) ----------------
    # Target box from the measured good grasps: outer wall 1.15-1.45 cup radii from the axis,
    # 0.45-0.8 of the mouth height. d_box = 0 inside the box, grows with the distance outside it.
    dr_box = (r_thumb - 1.45 * rad).clamp(min=0.0) + (1.15 * rad - r_thumb).clamp(min=0.0)
    dh_box = (h_thumb - 0.8 * mouth).clamp(min=0.0) + (0.45 * mouth - h_thumb).clamp(min=0.0)
    d_box = torch.sqrt(dr_box ** 2 + dh_box ** 2 + 1e-12)
    # smooth attraction to the box (length scale 1 cup radius): 1 in the box, > 0 everywhere,
    # increasing toward the box from any position (over the rim, inside the wall included)
    box_attract = torch.exp(-d_box / rad)

    # grasp-side placement (unchanged from the previous response): additionally exactly 0 for a
    # thumb tip radially inside the wall (0 at r <= 1.0 r, 1 at r >= 1.15 r) or above the rim.
    outer_wall = ((r_thumb - 1.0 * rad) / (0.15 * rad)).clamp(0.0, 1.0)
    thumb_place = outer_wall * below_rim * box_attract   # [0,1], no opposition term here

    # grasp-side thumb factor with a LOW floor (0.25): a grasp flag made by the thumb on/over the
    # rim or inside the mouth keeps only a quarter of the grasp and post-grasp income. (unchanged)
    thumb_f = 0.25 + 0.75 * thumb_place

    # ---------------- one grasp-shape factor ----------------
    # (a) all four fingers touching, weighted towards the weakest finger (mean/min blend)
    fc = c[:, 1:]
    fingers_all = 0.5 * fc.mean(-1) + 0.5 * fc.min(-1).values
    # (b) finger direction level: plane_level
    # (c) thumb outside and opposite: oppose_q
    parts = torch.stack([fingers_all, plane_level, oppose_q], dim=-1)
    shape = 0.5 * parts.mean(-1) + 0.5 * parts.min(-1).values
    # floored at 0.6 so the gradient to lift and carry never vanishes
    shape_f = 0.6 + 0.4 * shape

    # thumb-inside weight (unchanged): 1 outside, 0.5 fully inside
    inside_f = 1.0 - 0.5 * thumb_inside
    # post-grasp income also scaled by the thumb-placement factor (iter_07 item 2, unchanged)
    post_f = shape_f * inside_f * thumb_f

    closure = ctx.hand_closure.clamp(0.0, 1.0)
    open_ = 1.0 - closure

    # uprightness (same factor as lift / transport / hold)
    upright = torch.cos(ctx.cup_tilt).clamp(min=0.0) ** 2

    # ---------------- travel (far pull never multiplied, G2) ----------------
    approach = free * 0.35 * (0.6 * far_k + 0.4 * level_pre * near_k)
    open_hand = free * level_pre * 0.08 * open_ * near_k        # only near the cup -> no far parking spot
    pre_pose = free * level_pre * 0.07 * near_k * pose_q
    close_at_pose = free * level_pre * 0.10 * closure * at_pose * pose_q   # closing is not a loss at the pose

    # approach orientation, added (never multiplied onto the far approach, G2)
    level = 1.0 - n[:, 2].abs()                          # 1 = horizontal palm normal
    to_cup_h = _unit(_horizontal(ctx.cup_pos - ctx.palm_pos))
    face_cos = (_horizontal(n) * to_cup_h).sum(-1)       # >0 facing the cup, <0 facing away
    facing = 0.5 * (1.0 + torch.tanh(face_cos / 0.2))
    orient_fade = 1.0 - 0.5 * near_k
    approach_orient = free * level_pre * orient_fade * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    # CHANGED (repair, review item 1): pre-grasp thumb steering uses the plain box attraction,
    # NOT the inside / over-rim ramps, so it is non-zero and rises toward the box from the hat
    # (thumb over the rim) and thumb-inside positions too. Still inside the near-cup income
    # (x near_k, x level_pre, x free), same ceiling 0.04 (< the 0.14 near part of approach),
    # so it never outweighs getting closer. Opposite side of the four fingers weighted up
    # (0.4 floor keeps a gradient).
    thumb_steer = free * level_pre * 0.04 * near_k * box_attract * (0.4 + 0.6 * oppose)

    # no flipped grasp; 1 with the thumb >= 2 cm above the little finger, floor 0.25 below
    hand_up = 0.25 + 0.75 * thumb_up

    # ---------------- grasp (>= 2x the pre-grasp step) ----------------
    # x thumb_f (thumb pressing the rim from above earns 0.25 of it) - unchanged
    grasp_contact = 0.10 * upright * hand_up * wrap * thumb_f
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * oppose_q + 0.20 * envelope
    # base 1.05 x post_f (includes thumb_f)
    grasp_hold = held * upright * post_f * (1.05 + hand_up * plane_q * grasp_quality)

    # ---------------- lift / transport / hold (post_f includes thumb_f) ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)               # cup off the table
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)

    # lowering a held cup lowers this state income -> worth less than keeping still
    lift = 2.5 * held * upright * post_f * (height.clamp(min=0.0) / goal_h).clamp(max=1.0)

    carry_gate = held * off * upright                    # no goal income while on the table
    kp = ctx.goal_kp_dist
    transport = 2.5 * carry_gate * post_f * (0.6 * torch.exp(-kp / 0.10) + 0.4 * torch.exp(-kp / 0.03))

    tol = max(float(ctx.goal_tol), 1e-3)
    prox = torch.exp(-kp / tol)                          # sharpens as the curriculum shrinks tol
    cup_v = ctx.cup_lin_vel.norm(dim=-1)
    still = torch.exp(-cup_v / 0.05)
    goal_hold = 10.0 * carry_gate * post_f * prox * (0.5 + 0.5 * still)   # dominant income

    first = (ctx.goal_success & (ctx.goal_successes <= 1)).float()
    later = (ctx.goal_success & (ctx.goal_successes > 1)).float()
    success_bonus = 60.0 * first + 3.0 * later

    # ---------------- short-lived, bounded penalties (unchanged) ----------------
    cup_w = ctx.cup_ang_vel.norm(dim=-1)
    tilt = ctx.cup_tilt
    tilt_gate = torch.tanh((tilt - 0.05).clamp(min=0.0) / 0.1)        # ~3 deg deadband
    axis_dot = torch.cross(ctx.cup_ang_vel, up, dim=-1)               # d(cup axis)/dt
    axis_rate = axis_dot.norm(dim=-1)
    axis_move = torch.tanh((axis_rate - 0.2).clamp(min=0.0) / 0.3)    # above resting jitter
    tilt_rate = -axis_dot[:, 2] / torch.sin(tilt).clamp(min=1e-3)
    rising = torch.tanh(tilt_rate.clamp(min=0.0) / 0.3)
    tip_penalty = -0.6 * free * (1.0 - off) * tilt_gate * axis_move * (0.5 + 0.5 * rising)
    fall_speed = (-ctx.cup_lin_vel[:, 2]).clamp(min=0.0)
    drop_penalty = -1.0 * ctx.lifted.float() * free * torch.tanh(fall_speed / 0.5)
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
        "thumb_steer": thumb_steer,
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

근거 메모

가설 id 없음(brief 에 가설 id 줄 없음). prompt 전체(Required changes 1·2)를 한 후보로 구현했고, 재시도에서는 Review item 1 만 고쳤습니다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| thumb_steer (수정) | 이전 유도가 hat 62 %, thumb-in 100 %, good 19 % 스텝에서 0이었고 i07 에서 reward/thumb_steer 0.000. 안쪽/테두리 위 램프를 떼고 목표 상자 연속 끌림만 곱함. 천장 0.04, near-cup 안, 반대편 가중 유지 | prompt_c1_repair.md:722, 467 |
| box_attract | 목표 상자 1.15-1.45 r, 입구 높이의 0.45-0.8. 길이는 모두 ctx.cup_radius, ctx.cup_mouth_z 배수 | prompt_c1_repair.md:467, 460 |
| thumb_place, thumb_f, post_f (그대로) | 파지 쪽 엄지 인자는 낮은 바닥 0.25 그대로 둠. 실제 trace 에서 bad grasp 가 good 의 5-11 % | prompt_c1_repair.md:723, 469 |
| grasp_contact, grasp_hold, lift, transport, goal_hold (그대로) | 파지 뒤 수입이 엄지 위치에 의존(Required change 2) | prompt_c1_repair.md:469 |
| approach, open_hand, pre_pose, close_at_pose, approach_orient | 곱셈 없는 먼 접근과 near-cup 가산(G2, G3, R7). 변경 없음 | prompt_c1_repair.md:134-135, 185-187 |
| tip_penalty, drop_penalty, motion_penalty, action_rate, foreign_contact | 짧게 물리는 유한 벌점(R1, R6, G6). 변경 없음 | prompt_c1_repair.md:87, 190-191 |
| success_bonus | 첫 성공 60, 이후 3. 변경 없음 | prompt_c1_repair.md:85, 133 |

ledger 영향(추정):
- thumb_steer 는 free 스텝에서만 나오므로 파지 상태인 "3h hat" 행과 sloppy grasp 행 값은 바뀌지 않습니다.
- level hover(2h)는 엄지가 상자 밖일 때 이전 대비 많아야 약 +0.04 * box_attract 오릅니다. 그래서 "sloppy grasp >= 2x hover" 여유가 줄어듭니다. 손계산으로 sloppy 약 0.80, hover 약 0.39 라 2x 는 약 0.78 로, 통과하더라도 여유가 아주 작습니다. 드라이런 게이트에서 이 행을 확인해야 합니다.

prompt 파일: /home/user/rl_ws/hdgp/reward_gen/rh_aglt_r/iter_07/prompt_c1_repair.md
