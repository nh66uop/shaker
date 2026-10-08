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

    # ---------------- joint room (unchanged) ----------------
    m = ctx.arm_limit_margin.clamp(0.0, 1.0)
    room_j = (m / 0.05).clamp(0.0, 1.0)                 # (N,7)
    room = 0.5 * room_j.min(-1).values + 0.5 * room_j.mean(-1)
    room_pre = 0.5 + 0.5 * room
    room_post = 0.8 + 0.2 * room

    # ---------------- palm vs. the measured grasp ring (unchanged) ----------------
    h_p, rv_p = _axial_radial(ctx.palm_pos, ctx.cup_pos, up)
    r_p = rv_p.norm(dim=-1)
    h_target = 0.15 * ctx.cup_mouth_z                   # grasp height: origin + 0.15 x mouth
    d_ring = torch.sqrt((r_p - 1.5 * rad) ** 2 + (h_p - h_target) ** 2 + 1e-12)

    far_k = 1.0 - torch.tanh(d_ring / 0.25)             # long-range pull from the far start
    near_k = 1.0 - torch.tanh(d_ring / 0.05)            # last few cm
    at_pose = torch.exp(-d_ring / 0.015)                # essentially at the grasp pose

    # ---------------- NEW (iter_11): grasp-height factor on the palm's axial height ----------------
    # 1 within +-0.1 x mouth of the target (origin + 0.15 x mouth); smooth drop above it
    # (palm 0.41 x mouth -> ~0.81, floor 0.5); only a mild (max 15 %) cut below it
    dz = (h_p - h_target) / mouth
    up_ex = (dz - 0.1).clamp(min=0.0)
    lo_ex = (-dz - 0.1).clamp(min=0.0)
    height_f = (1.0 - 0.5 * torch.tanh((up_ex / 0.25) ** 2)) * (1.0 - 0.15 * torch.tanh((lo_ex / 0.3) ** 2))

    # ---------------- hand orientation quality (unchanged) ----------------
    n = _unit(ctx.palm_axes[:, 0:3])
    f = _unit(ctx.palm_axes[:, 3:6])
    n_ax = (n * up).sum(-1)
    f_ax = (f * up).sum(-1)
    horiz = ((1.0 - n_ax ** 2) * (1.0 - f_ax ** 2)).clamp(0.0, 1.0)
    w = 0.67 * n + 0.73 * f                              # measured palm->axis direction
    w_h = _unit(w - (w * up).sum(-1, keepdim=True) * up)
    to_axis = _unit(-rv_p)
    align = 0.5 * (1.0 + (to_axis * w_h).sum(-1))
    pos_q = torch.exp(-d_ring / 0.02)
    pose_q = (horiz + align + pos_q) / 3.0

    s_tilt = torch.sqrt(n_ax ** 2 + f_ax ** 2 + 1e-12)
    plane_q = (1.0 - (s_tilt - 0.2) / 0.5).clamp(0.0, 1.0)
    plane_level = (1.0 - (s_tilt - 0.2) / 0.3).clamp(0.0, 1.0)
    level_pre = 0.5 + 0.5 * plane_level

    thumb_dz = ctx.tips_pos[:, 0, 2] - ctx.tips_pos[:, 4, 2]
    thumb_up = ((thumb_dz + 0.01) / 0.03).clamp(0.0, 1.0)

    # ---------------- fingertips in cup cylinder coordinates ----------------
    h_t, rv_t = _axial_radial(ctx.tips_pos, ctx.cup_pos.unsqueeze(1), up.unsqueeze(1))
    r_t = rv_t.norm(dim=-1)                              # (N,5)
    r_thumb = r_t[:, 0]
    h_thumb = h_t[:, 0]

    in_r = ((1.1 * rad - r_thumb) / (0.1 * rad)).clamp(0.0, 1.0)
    in_h = ((h_thumb - 0.4 * mouth) / (0.3 * mouth)).clamp(0.0, 1.0)
    thumb_inside = in_r * in_h
    thumb_out = 1.0 - thumb_inside

    # ---------------- four fingers onto the OUTER wall ----------------
    # same "inside the mouth" rule as the thumb: radially inside the wall AND above 0.4 x mouth -> counts as far
    f_r = r_t[:, 1:]                                     # (N,4) index, middle, ring, little alike
    f_h = h_t[:, 1:]
    f_in_r = ((1.1 * rad - f_r) / (0.1 * rad)).clamp(0.0, 1.0)
    f_in_h = ((f_h - 0.4 * mouth) / (0.3 * mouth)).clamp(0.0, 1.0)
    f_outside = 1.0 - f_in_r * f_in_h
    # radial gap to the outer wall + distance outside the height window
    # NEW (iter_11): window widened to [-1.0, 0.6] x mouth so the low grasp keeps the little finger inside it
    gap_r = (f_r - rad).clamp(min=0.0)
    gap_h = (-1.0 * mouth - f_h).clamp(min=0.0) + (f_h - 0.6 * mouth).clamp(min=0.0)
    d_wall = torch.sqrt(gap_r ** 2 + gap_h ** 2 + 1e-12)
    wall_close = torch.exp(-d_wall / rad) * f_outside   # 1 = tip on the outer wall inside the window
    wall = (wall_close ** 2).mean(-1)                   # convex (squared) mean: each finger has its own gradient

    # NEW (iter_11): per-finger score CONTINUOUS from approach to contact:
    # the larger of the outer-wall contact score and 0.55 x the tip's outer-wall closeness
    f_contact = torch.tanh(ctx.finger_force[:, 1:].clamp(min=0.0) / 0.7) * f_outside
    f_each = torch.maximum(f_contact, 0.55 * wall_close)          # (N,4), four fingers alike
    # mean + min mix: ONE missing finger pulls the min down and costs a lot
    f_g = 0.4 * f_each.mean(-1) + 0.6 * f_each.min(-1).values
    # floor 0.75: a firm grasp with one finger missing still far above letting go (held income 0)
    finger_f = 0.75 + 0.25 * f_g ** 2

    # ---------------- contacts ----------------
    c = (ctx.finger_force / 1.0).clamp(0.0, 1.0)
    thumb_c = c[:, 0] * thumb_out
    fing_c = c[:, 1:].mean(-1)
    wrap = thumb_c * fing_c

    pc = (ctx.finger_prox_force / 0.5).clamp(0.0, 1.0)
    envelope = pc[:, 1:].mean(-1)

    u = _unit(rv_t)
    u_thumb = u[:, 0]
    u_fing = _unit(u[:, 1:].mean(1))
    oppose = 0.5 * (1.0 - (u_thumb * u_fing).sum(-1))

    # NEW (iter_11): thumb target box moved down to 0.25-0.6 x mouth (centre 0.40), radial box unchanged
    dr_box = (r_thumb - 1.45 * rad).clamp(min=0.0) + (1.15 * rad - r_thumb).clamp(min=0.0)
    dh_box = (h_thumb - 0.6 * mouth).clamp(min=0.0) + (0.25 * mouth - h_thumb).clamp(min=0.0)
    d_box = torch.sqrt(dr_box ** 2 + dh_box ** 2 + 1e-12)
    box_attract = torch.exp(-d_box / rad)

    outer_wall = ((r_thumb - 1.0 * rad) / (0.1 * rad)).clamp(0.0, 1.0)
    # iter_10 form kept: full anywhere on the outer wall below the mouth (a first grasp with the thumb
    # at 0.9 x mouth must not be cut); the lower grasp is driven by height_f and the thumb box
    not_above = ((mouth - h_thumb) / (0.08 * mouth)).clamp(0.0, 1.0)
    not_far = ((1.8 * rad - r_thumb) / (0.35 * rad)).clamp(0.0, 1.0)
    thumb_place = outer_wall * not_above * not_far
    thumb_f = 0.25 + 0.75 * thumb_place

    deep = ((2.3 * rad - r_p) / (0.5 * rad)).clamp(0.0, 1.0)
    proj = (rv_t[:, 1:] * to_axis.unsqueeze(1)).sum(-1)
    past = (0.5 + proj / (0.5 * rad)).clamp(0.0, 1.0).mean(-1)
    fc = c[:, 1:]
    fingers_all = 0.5 * fc.mean(-1) + 0.5 * fc.min(-1).values
    parts = torch.stack([fingers_all, plane_level, past], dim=-1)
    blend = 0.5 * parts.mean(-1) + 0.5 * parts.min(-1).values
    shape_f = (0.3 + 0.7 * deep) * (0.6 + 0.4 * blend)

    inside_f = 1.0 - 0.5 * thumb_inside
    # scales grasp_hold, lift, transport, goal_hold; NEW (iter_11): x height_f
    post_f = shape_f * inside_f * thumb_f * finger_f * height_f

    closure = ctx.hand_closure.clamp(0.0, 1.0)
    open_ = 1.0 - closure

    upright = torch.cos(ctx.cup_tilt).clamp(min=0.0) ** 2

    # ---------------- travel (unchanged) ----------------
    approach = room_pre * free * 0.35 * (0.6 * far_k + 0.4 * level_pre * near_k)
    open_hand = room_pre * free * level_pre * 0.08 * open_ * near_k
    pre_pose = room_pre * free * level_pre * 0.07 * near_k * pose_q
    close_at_pose = room_pre * free * level_pre * 0.10 * closure * at_pose * pose_q

    level = 1.0 - n[:, 2].abs()
    to_cup_h = _unit(_horizontal(ctx.cup_pos - ctx.palm_pos))
    face_cos = (_horizontal(n) * to_cup_h).sum(-1)
    facing = 0.5 * (1.0 + torch.tanh(face_cos / 0.2))
    orient_fade = 1.0 - 0.5 * near_k
    approach_orient = room_pre * free * level_pre * orient_fade * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    thumb_steer = room_pre * free * level_pre * 0.04 * near_k * box_attract * (0.4 + 0.6 * oppose)

    hand_up = 0.25 + 0.75 * thumb_up

    # fingertips onto the outer wall, before AND while grasping (small, gated by the palm at the grasp ring)
    finger_wall = 0.10 * room_pre * level_pre * near_k * pose_q * hand_up * upright * wall

    # ---------------- grasp (unchanged form; post_f now includes height_f) ----------------
    grasp_contact = 0.10 * upright * hand_up * wrap * thumb_f
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * deep * past + 0.20 * envelope
    grasp_hold = room_post * held * upright * post_f * (1.05 + hand_up * plane_q * grasp_quality)

    # ---------------- lift / transport / hold (unchanged form) ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)

    lift = room_post * 2.5 * held * upright * post_f * (height.clamp(min=0.0) / goal_h).clamp(max=1.0)

    carry_gate = held * off * upright
    kp = ctx.goal_kp_dist
    transport = room_post * 2.5 * carry_gate * post_f * (0.6 * torch.exp(-kp / 0.10) + 0.4 * torch.exp(-kp / 0.03))

    tol = max(float(ctx.goal_tol), 1e-3)
    prox = torch.exp(-kp / tol)
    cup_v = ctx.cup_lin_vel.norm(dim=-1)
    still = torch.exp(-cup_v / 0.05)
    goal_hold = room_post * 10.0 * carry_gate * post_f * prox * (0.5 + 0.5 * still)   # dominant income

    first = (ctx.goal_success & (ctx.goal_successes <= 1)).float()
    later = (ctx.goal_success & (ctx.goal_successes > 1)).float()
    success_bonus = 60.0 * first + 3.0 * later

    # ---------------- short-lived, bounded penalties (unchanged) ----------------
    cup_w = ctx.cup_ang_vel.norm(dim=-1)
    tilt = ctx.cup_tilt
    tilt_gate = torch.tanh((tilt - 0.05).clamp(min=0.0) / 0.1)
    axis_dot = torch.cross(ctx.cup_ang_vel, up, dim=-1)
    axis_rate = axis_dot.norm(dim=-1)
    axis_move = torch.tanh((axis_rate - 0.2).clamp(min=0.0) / 0.3)
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
        "finger_wall": finger_wall,
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
        "joint_room": room,          # diagnostic only (not summed)
        "finger_factor": finger_f,    # diagnostic only (not summed)
        "height_factor": height_f,    # diagnostic only (not summed)
    }
    diag = ("joint_room", "finger_factor", "height_factor")
    reward = torch.zeros_like(approach)
    for k, v in comps.items():
        if k in diag:
            continue
        reward = reward + v
    return reward, comps
