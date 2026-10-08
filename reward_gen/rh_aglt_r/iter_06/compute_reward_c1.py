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

    # CHANGED (iter_06 item 1): "thumb tip inside the mouth" weight, radial boundary moved to the
    # measured tip-origin positions (outer-wall grip 1.18 r, hooked inside the mouth 0.95 r):
    # radially: 1 at r <= 1.0 x cup_radius, 0 at r >= 1.1 x cup_radius (continuous ramp);
    # height (unchanged): 0 at h <= 0.4 x mouth height, 1 at h >= 0.7 x mouth height, so a thumb
    # on the lower, narrower part of the tapered outer wall stays at weight 0.
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
    # CHANGED (iter_06 item 1): thumb-tip band centred on the measured OUTER-wall tip position
    # (1.18 x cup_radius, the tip origin sits ~0.18 r outside the wall when gripping the wall):
    # full credit within 0.10 r of it (1.08-1.28 r), 0 at 0.22 r off (<= 0.96 r or >= 1.40 r),
    # so a tip at 1.18 r keeps full credit and a tip at 0.95 r gets none.
    wall_tip_r = 1.18 * rad
    on_wall_band = (1.0 - ((r_thumb - wall_tip_r).abs() - 0.10 * rad) / (0.12 * rad)).clamp(0.0, 1.0)
    on_wall = on_wall_band * thumb_out
    oppose_q = oppose * below_rim * on_wall              # thumb outside, on the wall, opposite

    # ---------------- one grasp-shape factor ----------------
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

    # CHANGED (iter_06 item 2): the thumb-inside weight lowers the post-grasp income on its own,
    # continuously: 1 with the thumb outside (other shape defects keep the 0.6 floor above),
    # 0.5 with the thumb fully inside -> combined with shape_f ~0.73 an inside-thumb hold keeps
    # ~0.37 of the post-grasp income instead of 0.72-0.77.
    inside_f = 1.0 - 0.5 * thumb_inside
    post_f = shape_f * inside_f

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
    # base 1.05 x floored shape factor; CHANGED (iter_06 item 2): x inside_f (via post_f)
    grasp_hold = held * upright * post_f * (1.05 + hand_up * plane_q * grasp_quality)

    # ---------------- lift / transport / hold ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)               # cup off the table
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)

    # lowering a held cup lowers this state income -> worth less than keeping still
    # CHANGED (iter_06 item 2): x inside_f (via post_f)
    lift = 2.5 * held * upright * post_f * (height.clamp(min=0.0) / goal_h).clamp(max=1.0)

    carry_gate = held * off * upright                    # no goal income while on the table
    kp = ctx.goal_kp_dist
    # CHANGED (iter_06 item 2): x inside_f (via post_f)
    transport = 2.5 * carry_gate * post_f * (0.6 * torch.exp(-kp / 0.10) + 0.4 * torch.exp(-kp / 0.03))

    tol = max(float(ctx.goal_tol), 1e-3)
    prox = torch.exp(-kp / tol)                          # sharpens as the curriculum shrinks tol
    cup_v = ctx.cup_lin_vel.norm(dim=-1)
    still = torch.exp(-cup_v / 0.05)
    # CHANGED (iter_06 item 2): x inside_f (via post_f); still the dominant income
    goal_hold = 10.0 * carry_gate * post_f * prox * (0.5 + 0.5 * still)

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
