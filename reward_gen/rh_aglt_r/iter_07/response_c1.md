Brief 에 가설 id 줄이 없어서 prompt 전체를 한 후보로 구현했다. 대상은 iter_07 필수 변경 1·2 (prompt.md:465-471). 파일은 쓰지 않았다. 정적 검사와 드라이런은 돌리지 않았고, 아래 수치는 손으로 계산한 추정이다.

**과제와 단계 (prompt 요구 1)**
- 과제: 먼 출발점에서 편 손으로 컵에 다가가 옆벽을 감싸 쥔다. 손바닥 법선과 손가락은 수평이고, 엄지는 손가락 반대편 바깥벽의 입구 아래에 둔다. 그대로 곧게 들어 `ctx.goal_pos` 로 옮기고, 직립·정지 상태로 쥔 채 유지한다. 목표는 여러 번 반복된다.
- 단계는 여섯이다.
  1. 먼 접근
  2. 컵 옆 편 손, 이번 라운드부터 이 단계에서 엄지를 바깥벽으로 유도
  3. 감싸 쥐기 (grasp 플래그)
  4. 들기
  5. 이송 (컵이 테이블에서 떨어졌을 때만)
  6. 목표 유지와 첫 성공 보너스
- iter_06 의 함정은 "모자" 파지였다. 엄지가 입구 위를 누르는 것만으로 grasp 플래그가 켜지고, 그래도 기본 파지 수입이 나갔다. 이번 변경은 엄지 위치를 (1) 파지 전 쉐이핑으로 유도하고 (2) 파지 후 모든 수입에 곱하는 인자로 만든다.

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

    # ---------------- NEW (iter_07 items 1+2): thumb-placement quality ----------------
    # Target box from the measured good grasps: outer wall 1.15-1.45 cup radii from the axis,
    # 0.45-0.8 of the mouth height. Smooth attraction to the box (length scale 1 cup radius),
    # times two continuous ramps that make it exactly 0 for a thumb tip radially inside the wall
    # (0 at r <= 1.0 r, 1 at r >= 1.15 r) or above the rim (below_rim: 0 at h >= mouth).
    # Hat (1.48 r, 1.05 mouth) -> 0; rim (0.91 r) -> 0; thumb-in (0.95 r) -> 0;
    # good_mirror (1.39 r, 0.74 mouth) and left-arm grasps (1.18-1.28 r, 0.50-0.67) -> 1.
    dr_box = (r_thumb - 1.45 * rad).clamp(min=0.0) + (1.15 * rad - r_thumb).clamp(min=0.0)
    dh_box = (h_thumb - 0.8 * mouth).clamp(min=0.0) + (0.45 * mouth - h_thumb).clamp(min=0.0)
    d_box = torch.sqrt(dr_box ** 2 + dh_box ** 2 + 1e-12)
    box_attract = torch.exp(-d_box / rad)
    outer_wall = ((r_thumb - 1.0 * rad) / (0.15 * rad)).clamp(0.0, 1.0)
    thumb_place = outer_wall * below_rim * box_attract   # [0,1], no opposition term here

    # grasp-side thumb factor with a LOW floor (0.25): a grasp flag made by the thumb on/over the
    # rim or inside the mouth keeps only a quarter of the grasp and post-grasp income.
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
    # CHANGED (iter_07 item 2): post-grasp income also scaled by the thumb-placement factor
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

    # NEW (iter_07 item 1): pre-grasp thumb steering, inside the near-cup income (x near_k,
    # x level_pre, x free). Ceiling 0.04 (< the 0.14 near part of approach), so it never outweighs
    # getting closer. Opposite side of the four fingers weighted up (0.4 floor keeps a gradient).
    thumb_steer = free * level_pre * 0.04 * near_k * thumb_place * (0.4 + 0.6 * oppose)

    # no flipped grasp; 1 with the thumb >= 2 cm above the little finger, floor 0.25 below
    hand_up = 0.25 + 0.75 * thumb_up

    # ---------------- grasp (>= 2x the pre-grasp step) ----------------
    # CHANGED (iter_07 item 2): x thumb_f (thumb pressing the rim from above earns 0.25 of it)
    grasp_contact = 0.10 * upright * hand_up * wrap * thumb_f
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * oppose_q + 0.20 * envelope
    # base 1.05 x post_f (now includes thumb_f)
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

**근거 메모**
가설 id 미지정. prompt 전체(iter_07 필수 변경 1·2)를 한 후보로 구현했다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| `thumb_place` (새 인자: 상자 1.15-1.45 r · 0.45-0.8 입구높이로 당김, 벽 안쪽·입구 위는 연속 램프로 0) | 좋은 파지와 모자·rim·엄지안 파지를 가르는 것은 엄지 끝 반경과 높이다. 모자는 1.48 r / 1.05, rim 0.91 r, 엄지안 0.95 r 이고 셋 다 0 이 된다. 좋은 파지 1.39 r / 0.74 와 왼팔 1.18-1.28 r 은 1 이다 | prompt.md:460, 467 |
| `thumb_steer` (새 항, 0.04 × near_k × level_pre × free × thumb_place × (0.4+0.6 oppose)) | 변경 1: 파지 전에 엄지를 바깥벽·입구 아래·손가락 반대편으로 유도한다. near-cup 수입 안에 두고, 상한 0.04 로 접근 near 부분 0.14 보다 작게 했다 | prompt.md:461, 467 |
| `thumb_f = 0.25 + 0.75 thumb_place` 를 `post_f` 에 곱함 (grasp_hold·lift·transport·goal_hold) | 변경 2: 파지 수입이 엄지 위치에 의존하고 바닥이 낮다. 모자 held 추정은 0.72×0.25+0.05 ≈ 0.23 으로 level hover 0.371 보다 작다 (추정) | prompt.md:459, 469 |
| `grasp_contact` × `thumb_f` | 모자 함정의 입구: 입구 위를 누르는 엄지 접촉이 파지 플래그 전 가교 수입을 받았다 | prompt.md:459, 469 |
| 기존 항 (approach, open_hand, pre_pose, close_at_pose, approach_orient, 벌점 5종, success_bonus) | 모두 변경 없음 | prompt.md:471 |
| 오름차순 필수 (sloppy 첫 파지 ≥ 2x hover) | sloppy 파지는 엄지가 바깥벽에 있어 thumb_f ≈ 1 이다. 그래서 `thumb_place` 에 대립(oppose)을 넣지 않았다. 손가락 쪽 엄지라도 깎이지 않는다 | prompt.md:100, 469 |

**호출자가 판단할 위험 3건**
1. **sloppy ≥ 2x hover 여유가 거의 없다.** 손 계산으로 sloppy ≈ 0.80 이고 hover 0.371 의 2배는 0.742 다. `thumb_steer` 는 hover 에 최대 0.04 × near_k 를 더한다. 드라이런에서 실패하면 가설 범위 안의 해법은 `thumb_steer` 가중치를 낮추는 것이다.
2. **iter_06 의 `on_wall_band` 를 고치지 않았다.** 이 값은 1.18 r ± 0.22 r 에서 0 이 되므로 good_mirror 엄지 1.39 r 에서 약 0.08 이다. 이것이 `oppose_q` 를 거쳐 좋은 파지의 grasp_quality 를 깎는다. "나머지는 유지" 지시를 따라 그대로 두었지만 다음 라운드에서 볼 만하다.
3. **항 수가 17 개다.** prompt 의 "About 15 terms" 보다 둘 많다.

Prompt 파일: /home/user/rl_ws/hdgp/reward_gen/rh_aglt_r/iter_07/prompt.md
