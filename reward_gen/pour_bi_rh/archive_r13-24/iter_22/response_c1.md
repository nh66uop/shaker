## 1. What the task means and the stages the robot goes through

Two arms, two cups. The right hand (source) takes the cup that holds 20 beads. The left hand (receiver) takes the empty cup. The task runs in these stages:

- **A. Grasp.** Each hand makes a fingertip grasp on its own cup.
- **B. Lift.** Both cups come off the table. The source goes high, because the wrist only produces the commanded rotation when the palm is high (K3).
- **C. Place.** The receiver mouth goes under the point where the source spout will be. That point is on the source's front-left side (K2), near the y = 0 centreline.
- **D. Aim.** The source spout ends up horizontally within a few cup radii of the receiver mouth, and above it but not far above it.
- **E. Direction.** The source leans toward the receiver.
- **F. Outcome.** Beads fall through the air into the receiver. Spill is a cost. Success is judged by the environment.
- **G. Control and H. Height.** Tilting stays at a stable speed with smooth commands, and the source does not sink while it pours.

## Diagnosis of i21 (from the prompt only)

The policy grasps and lifts, then stops.

- **Grasp and lift work:** src_grasped is about 0.74, rcv_grasped about 0.80, and state income is about 3.0 per step.
- **Nothing is delivered:** outcome income is about 1.5e-4, in_target is about 1e-4, and there are no successes.
- **The cups never meet:** the spout is about 17 cm from the receiver mouth horizontally, and place_rcv and aim_lip only pay about 0.08 and 0.17 per step.
- **The source barely tilts:** the curve shows about 20 deg; the deterministic replay shows a median maximum of 48 deg against about 70 deg for the first bead.
- **Why:** no term pays for tilting, only costs attach to it (operator cause 1). The placement target was tied to the source, so it rose whenever the source rose (cause 2).

iter_22 applies only the three operator changes and keeps the rest of iter_21 unchanged:

1. **Placement (place_rcv).** The receiver mouth target is now a fixed world height: the rim of a cup standing on the table plus 0.15 m. Three more rules come with it:
   - The receiver mouth is pulled horizontally until the spout is within 1 r of it.
   - The spout gets a height ceiling of 4 r above the rim. It sits in the aim potential and in the permission window.
   - The receiver mouth may be at most 2 r from y = 0.
   - I kept iter_21's pourable-side knowledge (K2) as a guard. It costs nothing inside 68 ± 45 deg around the source.
2. **New term tilt_open.** It pays for tilting toward the receiver only while the spout is inside the window, and only until the first bead has landed.
   - It is the exact rate of the potential `win * theta+`, so moving in and out of the window earns nothing.
   - There is no angle target and no band.
   - It is capped at 0.5 per step, which saturates at the pour-phase spin limit of 1 rad/s.
3. **Direction cost and spill cost.**
   - tilt_dir keeps weight 1.5 outside the window and drops to 0.3 inside it.
   - Inside the aimed window, spill costs x0.25 (-3 per bead) until the first bead has landed in the receiver.

All speed, control and safety penalties, lift_src, the outcome weights, and the budget structure are unchanged. There are now 26 summed terms: iter_21's 25 plus tilt_open.

## 2. Reward function

```python
import torch
import math


# =====================================================================================================
# iter_22 = iter_21 design + ONLY the three round-22 operator changes (09.25 evening, user approved):
#   (1) place_rcv : receiver MOUTH target at an ABSOLUTE height (rim of a cup standing on the table, raised
#                   by RCV_POUR_LIFT) - no longer dragged up by the source. Horizontal part judged against the
#                   source SPOUT with a cup_radius window; pour point near the y = 0 centreline; the spout
#                   height gets a ceiling (aim_lip potential and the permission window).
#   (2) tilt_open : NEW small difference-form signal: rate of tilt TOWARD the receiver, paid only while the
#                   spout is inside the window and until the first bead has landed in the receiver. It is the
#                   exact rate of the potential win * theta+, so moving in / out of the window cannot farm it.
#                   No angle target, no band, capped at W_TILT_OPEN per step.
#   (3) tilt_dir  : strong only OUTSIDE the window (1.5), weak inside (0.3). spill_delta: x0.25 inside the
#                   aimed window until the first bead has landed (i17 lesson: the first bead must get out).
# Unchanged on purpose: grasp, lift, pour_sink, all outcome weights, every speed / control / safety penalty.
# Summed terms: iter_21's 25 + tilt_open = 26.
# =====================================================================================================


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _smoothstep(x):
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_slope(x):
    """d smoothstep / dx (exactly 0 outside [0, 1])."""
    t = torch.clamp(x, 0.0, 1.0)
    return 6.0 * t * (1.0 - t)


def _late_ramp(x, lo, hi):
    """0.5*t + 0.5*t^2: never flat above lo, last part worth 3x the first."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _progress(rate_closing, v_scale):
    """Difference-form payment from an exact potential rate, positive = getting closer.
    Symmetric clamp (iter_21): +-1 per step, reached at the stable-speed limit v_scale, so going faster than
    allowed buys nothing and exploration jitter nets zero (i20: an asymmetric clamp made jitter a net loss)."""
    return torch.clamp(rate_closing / v_scale, -1.0, 1.0)


def _limit_pen(margin, free):
    d = torch.clamp((free - torch.clamp(margin, 0.0, 1.0)) / free, 0.0, 1.0)
    worst = d.max(dim=-1).values
    return 0.7 * worst * worst + 0.3 * (d * d).mean(dim=-1)


def _shake_pen(qdd, free_mean, scale_mean, free_peak, scale_peak):
    q = qdd.abs()
    mean_part = torch.tanh(torch.clamp(q.mean(dim=-1) - free_mean, min=0.0) / scale_mean)
    peak_part = torch.tanh(torch.clamp(q.max(dim=-1).values - free_peak, min=0.0) / scale_peak)
    return 0.65 * mean_part + 0.35 * peak_part


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the CUP frame. VERBATIM since iter_04 (grasp 0.75/0.82 measured)."""
    ARRIVE_RISE = 0.045
    FUNNEL_R0 = 0.01
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z
    thumb_hi = 0.5 * cup_mouth_z

    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)
    rad_vec = rel - h_tip.unsqueeze(-1) * up
    rad = rad_vec.norm(dim=-1)
    gap = torch.clamp(rad - tip_r, min=0.0)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values

    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))

    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
              cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok):
    """Per-arm stage-A value in [0, 1] plus the table-disturbance penalties (iter_21 VERBATIM).
    ONE monotone ladder, grasped (>= 0.4) always above not-grasped (<= 0.39)."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach = 0.4 * reach_far + 0.6 * 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])

    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = near_grip * torch.clamp(hand_closure / 0.7, max=1.0)
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    contact_q = (0.4 + 0.6 * pinch) * (0.1 + 0.9 * geo["thumb_band"])
    balance = torch.minimum(f_t, f_o) / (torch.maximum(f_t, f_o) + 1e-6)
    keep = near_grip * (0.3 + 0.7 * balance)
    held_val = 0.4 + 0.3 * contact_q + 0.3 * keep                       # in [0.4, 1.0]
    free_val = 0.25 * reach + 0.10 * close + 0.10 * pre_grasp           # at most 0.39
    value = torch.where(g, held_val, free_val) * pose_ok

    h_pos = torch.clamp(cup_pos[:, 2] - cup_spawn_pos[:, 2] - 0.005, min=0.0)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    v_lin = cup_lin_vel.norm(dim=-1)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"value": value, "h": h_pos, "v_lin": v_lin, "slide": slide, "one_sided": one_sided,
            "palm_push": palm_push, "rush": rush, "fling": fling, "keep": keep, "balance": balance}


def _pour_geometry(ctx, r):
    """Directed-spout geometry, iter_08/09 VERBATIM.
      d_hat : horizontal unit vector source cup origin -> receiver mouth
      theta : SIGNED tilt toward the receiver (locates the lip; never a target)
      lip   : rim point the beads leave from; lip_xy / lip_dz relative to the receiver mouth."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    sep = v.norm(dim=-1)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / 0.03, max=1.0)          # azimuth undefined when coaxial (3 cm floor)
    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    cos_t = torch.cos(torch.clamp(theta_eff, 0.0, math.pi))
    lip_drop = r * sin_t + ctx.cup_mouth_z * (1.0 - cos_t)
    lip_xy_vec = ctx.src_cup_mouth_pos[:, :2] \
        + (r * torch.cos(theta)).unsqueeze(-1) * d_hat - ctx.rcv_cup_mouth_pos[:, :2]
    lip_xy = lip_xy_vec.norm(dim=-1)
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r * torch.sin(theta)
    lip_dz = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_pt = torch.cat([ctx.rcv_cup_mouth_pos[:, :2] + lip_xy_vec, lip_z.unsqueeze(-1)], dim=-1)
    return {"d_hat": d_hat, "sep_ok": sep_ok, "theta": theta, "lip_drop": lip_drop,
            "lip_xy_vec": lip_xy_vec, "lip_xy": lip_xy, "lip_dz": lip_dz, "lip_pt": lip_pt}


def _point_vel(lin_vel, ang_vel, origin, point):
    return lin_vel + torch.cross(ang_vel, point - origin, dim=-1)


def _tilt_direction_error(ctx, d_hat):
    """iter_19 VERBATIM: psi = angle between the cup's lean (horizontal part of its up axis) and d_hat.
    dir_err = (1 - cos psi)/2 is 0 when leaning at the receiver at ANY tilt - no magnitude is valued."""
    u_xy = ctx.src_cup_up[:, :2]
    u_h = u_xy.norm(dim=-1)
    cos_psi = torch.clamp((u_xy * d_hat).sum(dim=-1) / torch.clamp(u_h, min=1e-6), -1.0, 1.0)
    return 0.5 * (1.0 - cos_psi), torch.acos(cos_psi)


def _tilt_rate_toward(up, ang_vel, d_hat):
    """iter_22: d(theta)/dt [rad/s] of the signed tilt toward d_hat, theta = atan2(u . d_hat, u_z), from
    du/dt = w x u (d_hat held fixed over one step). A RATE only - no angle value is used as a target."""
    du = torch.cross(ang_vel, up, dim=-1)
    u_d = (up[:, :2] * d_hat).sum(dim=-1)
    du_d = (du[:, :2] * d_hat).sum(dim=-1)
    den = torch.clamp(u_d * u_d + up[:, 2] * up[:, 2], min=1e-3)
    return (up[:, 2] * du_d - u_d * du[:, 2]) / den


def _spout_window(lip_xy, lip_dz, dlip_xy, dlip_dz, r_in, r_out, z_lo, z_hi, z_fade):
    """iter_22: spout window in [0, 1] and its exact time derivative (same state function both ways).
    horizontal: 1 inside r_in, smoothstep to 0 at r_out (iter_13 window, unchanged);
    height    : 0 -> 1 linearly over [0, z_lo] (iter_17 floor, unchanged), 1 up to the NEW ceiling z_hi,
                smoothstep to 0 at z_hi + z_fade."""
    t_xy = (lip_xy - r_in) / (r_out - r_in)
    w_xy = 1.0 - _smoothstep(t_xy)
    dw_xy = -_smoothstep_slope(t_xy) * dlip_xy / (r_out - r_in)
    w_lo = torch.clamp(lip_dz / z_lo, 0.0, 1.0)
    dw_lo = ((lip_dz > 0.0) & (lip_dz < z_lo)).to(lip_dz.dtype) * dlip_dz / z_lo
    t_hi = (lip_dz - z_hi) / z_fade
    w_hi = 1.0 - _smoothstep(t_hi)
    dw_hi = -_smoothstep_slope(t_hi) * dlip_dz / z_fade
    w_z = w_lo * w_hi
    dw_z = dw_lo * w_hi + w_lo * dw_hi
    return {"w_xy": w_xy, "w_z": w_z, "win": w_xy * w_z, "dwin": dw_xy * w_z + w_xy * dw_z}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ------------------------------------------------------------------ constants and their sources
    W_ALIVE = 1.0            # constant survival floor, identical in every state (not A-D income)
    # A grasp (iter_21 verbatim)
    W_GRASP_SRC, W_GRASP_RCV = 1.5, 1.5
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35       # rad, the ENV's own receiver-upright success limit (20 deg)
    # B lift (iter_21 verbatim; brief 5-1 B keeps the raised source allowance)
    W_LIFT_SRC = 1.5
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.30
    LIFT_FREE_SRC0 = 0.40
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    W_LIFT_RCV = 0.5
    LIFT_TARGET_RCV = 0.08
    CARRY_H = 0.03
    # C placement - iter_22 change 1: WORLD-FIXED height for the receiver mouth, horizontal vs the SPOUT
    RCV_POUR_LIFT = 0.15     # m, height allowance (the only absolute constant brief 5-3 permits). Mouth target
                             # = rim of a cup standing on the table + 0.15 m. i21 rcv lift reached 0.182 m max.
                             # With the spout window 0.7 r..4 r above that mouth, the source origin sits at
                             # lift ~0.23-0.34 m while pouring (70-90 deg), inside the lift_src paid range
                             # (LIFT_TARGET_SRC 0.30), so placement and lift_src do not fight.
    Z_TOL_R = 1.0            # +-1 r free band around the mouth height target
    PLACE_WIN_R = 1.0        # horizontal: receiver mouth pulled until the spout is within 1 r (= PERM_IN_R)
    PLACE_AZ = math.radians(68.0)       # K2 pourable side, kept from iter_21 as a GUARD only
    PLACE_AZ_TOL = math.radians(45.0)   # K1 palm-yaw box +-45 deg: 23..113 deg free, outside costs distance
    CENTER_FREE_R = 2.0      # |y| of the receiver mouth (= pour point) free up to 2 r (y ~= 0 centreline)
    W_PLACE = 1.0
    # D aim
    W_AIM = 1.5
    V_PROG = 0.15            # m/s: progress terms saturate at the stable cup speed limit (SRC_V_FREE)
    LIP_Z_HI_R = 0.7         # spout floor above the rim, 0.7 r ~= 2 cm (iter_17 permission height)
    LIP_Z_MAX_R = 4.0        # iter_22 change 1: spout CEILING 4 r ~= 11 cm above the rim (i21 hovered at
                             # +13.8 cm). Window height 3.3 r > lip drop at a quarter turn (r + mouth_z ~= 2.9 r
                             # with K5/K6), so a spout at the ceiling upright is still above the floor when level.
    LIP_Z_FADE_R = 2.0       # window fades out between 4 r and 6 r
    SINK_KEEP_R = 1.0        # pour_sink geometry (iter_19)
    # E direction - iter_22 change 3
    W_TILT_DIR_OUT = 1.5     # outside the window: iter_19/21 strength (1.03/step at the i18 failure state)
    W_TILT_DIR_IN = 0.3      # inside the window: a fifth, the spout is already over the mouth
    # tilt_open - iter_22 change 2
    W_TILT_OPEN = 0.5        # <= 0.5/step: far below one bead (+7.5) and transferred (~7/step at 0.9 in)
    OPEN_BEADS = 1.0         # active until the first bead has landed in the receiver, then pour_delta takes over
    # H height keep while pouring (iter_19 verbatim)
    W_POUR_SINK = 1.0
    V_Z_SCALE = 0.04
    # F outcome - unchanged weights (per-bead units with 20 beads)
    W_POUR = 150.0           # +7.5 per bead delivered
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0            # -2 per extra bead dumped in the same step
    W_SPILL = 240.0          # -12 per bead spilled
    SPILL_RELIEF = 0.25      # iter_22 change 3: x0.25 (-3/bead) inside the AIMED window before the first landing
    SETTLE_FRAC = 0.03
    W_UNAIMED = 60.0         # -3 per bead released outside the permission window
    W_TRANSFER = 8.0         # level term: 0.9 in, 0.05 spill -> 5.76/step (> A+B, brief 5-2 (3))
    W_TAIL = 4.0
    TAIL_LO, TAIL_HI = 0.10, 0.90
    EMPTY_LO = 0.20
    W_SUCCESS = 20.0
    SPILL_ZERO = 0.25        # env spill limit
    PERM_IN_R, PERM_OUT_R = 1.0, 3.0      # spout window in cup radii (iter_13, kept)
    LEAD_FULL_R, LEAD_ZERO_R = 1.75, 5.0  # pour-phase window for the stricter spin limit (kept)
    # G control - KEEP LIST values (unchanged, user principle: no speed / control relaxation)
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, W_ROT_POUR = 1.0, 1.0
    RCV_W_FREE, RCV_W_SCALE, W_RCV_SPIN = 0.5, 1.0, 0.5
    STILL_FREE, STILL_SCALE, W_STILL_POUR = 0.05, 0.15, 0.5
    W_TILT_PEN_RCV = 0.3
    W_FLIP_PALM, W_FLIP_HAND = 0.3, 0.2
    W_RATE_PALM_SRC, W_RATE_PALM_RCV, W_RATE_HAND = 0.03, 0.05, 0.01
    W_ACT_RATE = 0.05
    BANG_DEAD, W_BANG_SRC, W_BANG_RCV = 0.1, 1.5, 2.0
    SAT_LO, W_PALM_SAT_SRC, W_PALM_SAT_RCV = 0.7, 3.0, 1.5
    MARGIN_FREE, W_ARM_LIMIT = 0.30, 0.4
    QDD_FREE, QDD_SCALE = 2.5, 10.3
    QDD_PEAK_FREE, QDD_PEAK_SCALE = 12.8, 12.8
    W_ARM_SHAKE = 0.4
    TILT_RATE_SCALE = ROT_POUR_FREE   # rad/s: tilt_open saturates at the pour-phase spin limit
    # safety / disturbance (unchanged)
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0

    dtype = ctx.src_cup_tilt.dtype
    r = ctx.cup_radius
    geo = _pour_geometry(ctx, r)
    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    nested = ctx.cups_nested.to(dtype)
    not_nested = 1.0 - nested

    # ------------------------------------------------------------------ A. grasp (iter_21 verbatim)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    h_src_raw = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    carry_src = g_src * torch.clamp(h_src_raw / CARRY_H, max=1.0)
    src_table_upright = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = carry_src + (1.0 - carry_src) * src_table_upright
    src = _hand_cup(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                    ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel,
                    ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z, TIP_R, pose_ok_src)
    rcv = _hand_cup(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                    ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel,
                    ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z, TIP_R, upright_rcv)
    h_src, h_rcv = src["h"], rcv["h"]
    carry_rcv = g_rcv * torch.clamp(h_rcv / CARRY_H, max=1.0)
    carry_both = carry_src * carry_rcv * not_nested
    up_w_src = g_src * torch.clamp(h_src / 0.05, max=1.0)
    up_w_rcv = g_rcv * torch.clamp(h_rcv / 0.05, max=1.0)
    ctrl_gate = up_w_src * up_w_rcv

    # ------------------------------------------------------------------ B. lift (iter_21 verbatim)
    lift_free_src = LIFT_FREE_SRC0 + geo["lip_drop"]
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    over_lift = torch.clamp((h_src - lift_free_src) / OVER_LIFT_RANGE, 0.0, 1.0)
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_band - W_OVER_LIFT * over_lift
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * (0.5 + 0.5 * upright_rcv)

    # ------------------------------------------------------------------ bead bookkeeping (gates)
    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    has_beads = torch.clamp(fill / 0.10, max=1.0)        # >= 2 beads left
    stage_cd = carry_both * has_beads                     # placement / aim live while there is a pour ahead
    n_beads = float(max(int(ctx.num_beads), 1))
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    first_open = torch.clamp(OPEN_BEADS - n_beads * in_t, 0.0, 1.0)   # 1 until the first bead has landed

    # velocities of the two points that matter, and the spout rates relative to the mouth
    v_mouth = _point_vel(ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_pos, ctx.rcv_cup_mouth_pos)
    v_lip = _point_vel(ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_pos, geo["lip_pt"])
    dv = v_lip - v_mouth
    dlip_xy = (geo["lip_xy_vec"] * dv[:, :2]).sum(dim=-1) / torch.clamp(geo["lip_xy"], min=1e-4)
    dlip_dz = dv[:, 2]

    # ------------------------------------------------------------------ spout window (a GATE, never a state income)
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    z_lo, z_hi = LIP_Z_HI_R * r, LIP_Z_MAX_R * r
    wnd = _spout_window(geo["lip_xy"], geo["lip_dz"], dlip_xy, dlip_dz, r_in, r_out, z_lo, z_hi,
                        LIP_Z_FADE_R * r)
    win = wnd["win"]                                      # position window (horizontal x height)
    perm_xy = wnd["w_xy"]
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = win * toward                                   # aimed window: position + leaning toward
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r) / ((LEAD_ZERO_R - LEAD_FULL_R) * r))

    # ------------------------------------------------------------------ C. placement (iter_22 change 1)
    # Receiver mouth: world-fixed height (not tied to the source), horizontally until the SPOUT is within
    # PLACE_WIN_R*r, |y| near the centreline, and not outside the pourable side of the source (guard).
    # Potential = norm of the excesses; paid as its exact rate -> 0 when parked anywhere.
    z_mouth_tgt = ctx.table_z - ctx.cup_bottom_z + ctx.cup_mouth_z + RCV_POUR_LIFT
    dz_m = ctx.rcv_cup_mouth_pos[:, 2] - z_mouth_tgt
    ez = torch.sign(dz_m) * torch.clamp(dz_m.abs() - Z_TOL_R * r, min=0.0)
    eh = torch.clamp(geo["lip_xy"] - PLACE_WIN_R * r, min=0.0)
    y_m = ctx.rcv_cup_mouth_pos[:, 1]
    ey = torch.clamp(y_m.abs() - CENTER_FREE_R * r, min=0.0)
    rel = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    drel = v_mouth[:, :2] - ctx.src_cup_lin_vel[:, :2]
    a_lo, a_hi = PLACE_AZ - PLACE_AZ_TOL, PLACE_AZ + PLACE_AZ_TOL
    n_lo_x, n_lo_y = math.sin(a_lo), -math.cos(a_lo)      # outward normal of the 23 deg edge
    n_hi_x, n_hi_y = -math.sin(a_hi), math.cos(a_hi)      # outward normal of the 113 deg edge
    es_lo = torch.clamp(rel[:, 0] * n_lo_x + rel[:, 1] * n_lo_y, min=0.0)
    es_hi = torch.clamp(rel[:, 0] * n_hi_x + rel[:, 1] * n_hi_y, min=0.0)
    des_lo = drel[:, 0] * n_lo_x + drel[:, 1] * n_lo_y
    des_hi = drel[:, 0] * n_hi_x + drel[:, 1] * n_hi_y
    d_place = torch.sqrt(eh ** 2 + ez ** 2 + ey ** 2 + es_lo ** 2 + es_hi ** 2)
    dd_place = (eh * dlip_xy + ez * v_mouth[:, 2] + ey * torch.sign(y_m) * v_mouth[:, 1]
                + es_lo * des_lo + es_hi * des_hi) / torch.clamp(d_place, min=1e-4)
    place = stage_cd * _progress(-dd_place, V_PROG)

    # ------------------------------------------------------------------ D. aim (3D, difference form)
    # Potential: horizontal spout->mouth distance + how far the spout is BELOW its floor + (iter_22) how far
    # it is ABOVE its ceiling. Tilting toward the receiver lowers the spout, so a spout hovering too high is
    # brought into the window either by descending or by tilting - no angle is named.
    below = torch.clamp(z_lo - geo["lip_dz"], min=0.0)
    above = torch.clamp(geo["lip_dz"] - z_hi, min=0.0)
    d_aim = torch.sqrt(geo["lip_xy"] ** 2 + below ** 2 + above ** 2)
    dd_aim = ((geo["lip_xy_vec"] * dv[:, :2]).sum(dim=-1) + (above - below) * dv[:, 2]) \
        / torch.clamp(d_aim, min=1e-4)
    aim = stage_cd * _progress(-dd_aim, V_PROG)

    # ------------------------------------------------------------------ tilt_open (iter_22 change 2)
    # Exact rate of the potential win * theta+ : pays for tilting TOWARD the receiver only inside the window,
    # charges untilting inside it and carrying a tilted spout OUT of it, so no in/out cycle nets income.
    # Off once the first bead has landed (pour_delta takes over) or the cup is empty. No target, no band.
    theta_rate = _tilt_rate_toward(ctx.src_cup_up, ctx.src_cup_ang_vel, geo["d_hat"])
    theta_pos = torch.clamp(geo["theta"], min=0.0)
    open_rate = wnd["dwin"] * theta_pos + win * toward * theta_rate
    tilt_open = carry_both * geo["sep_ok"] * has_beads * first_open * _progress(open_rate, TILT_RATE_SCALE)

    # ------------------------------------------------------------------ E. direction (cost only, iter_22 change 3)
    dir_err, psi = _tilt_direction_error(ctx, geo["d_hat"])
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)          # magnitude only as a multiplier, no target
    dir_w = W_TILT_DIR_IN * win + W_TILT_DIR_OUT * (1.0 - win)   # strong only outside the window
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err * dir_w

    # ------------------------------------------------------------------ H. keep height while pouring (verbatim)
    org_vz = ctx.src_cup_lin_vel[:, 2] - v_mouth[:, 2]
    h_org = ctx.src_cup_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    org_low = (h_org < (LIP_Z_HI_R + SINK_KEEP_R) * r).to(dtype)
    pour_sink = ctrl_gate * not_nested * org_low * torch.clamp(org_vz / V_Z_SCALE, -1.0, 1.0)

    # ------------------------------------------------------------------ F. outcome (the beads)
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    # iter_22 change 3: spill inside the AIMED window before the first landed bead costs x0.25 (-3/bead,
    # still a loss vs +7.5 delivered); spill while carrying / outside the window keeps the full -12/bead.
    spill_w = 1.0 - (1.0 - SPILL_RELIEF) * perm * first_open
    spill_step = settled * torch.clamp(ctx.d_spill, min=0.0) * spill_w
    spill_lvl = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    clean_f = torch.clamp(1.0 - spill_lvl / SPILL_ZERO, 0.0, 1.0)
    emptied = 1.0 - fill
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)
    fill_ok = torch.clamp(in_t / TAIL_LO, max=1.0)       # emptying onto the table pays nothing
    finish_q = 0.5 * (fill_tail + empty_q * fill_ok)
    release_step = torch.clamp(d_in, min=0.0) + torch.clamp(ctx.d_spill, min=0.0)
    unaimed = settled * release_step * (1.0 - perm)
    out_gate = not_nested * upright_rcv * clean_f

    # ------------------------------------------------------------------ G. control (iter_21 verbatim)
    v_src = src["v_lin"]
    w_src = ctx.src_cup_ang_vel.norm(dim=-1)
    w_rcv = ctx.rcv_cup_ang_vel.norm(dim=-1)
    pour_phase = ctrl_gate * lead_gate * perm_xy
    src_speed = ctrl_gate * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = W_SRC_SPIN * ctrl_gate * torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE) \
        + W_ROT_POUR * pour_phase * torch.tanh(torch.clamp(w_src - ROT_POUR_FREE, min=0.0) / SRC_W_SCALE)
    rcv_still_over = torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    rcv_steady = W_TILT_PEN_RCV * torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL) \
        + W_RCV_SPIN * ctrl_gate * torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE) \
        + W_STILL_POUR * ctrl_gate * perm_xy * rcv_still_over

    a_abs = ctx.actions.abs()
    da = ctx.actions - ctx.prev_actions
    flips = torch.clamp(-(ctx.actions * ctx.prev_actions), min=0.0)
    flip_palm_src = flips[:, 0:6].sum(dim=-1)
    flip_palm_rcv = flips[:, 12:18].sum(dim=-1)
    flip_hand = flips[:, 6:12].sum(dim=-1) + flips[:, 18:24].sum(dim=-1)
    act_flip = W_FLIP_PALM * (flip_palm_src + flip_palm_rcv) \
        + W_FLIP_HAND * flip_hand
    both_big = (a_abs > BANG_DEAD) & (ctx.prev_actions.abs() > BANG_DEAD)
    bang = (both_big & (ctx.actions * ctx.prev_actions < 0.0)).to(dtype)
    bang_src = bang[:, 0:6].mean(dim=-1)
    bang_rcv = bang[:, 12:18].mean(dim=-1)
    palm_bang = W_BANG_SRC * bang_src + W_BANG_RCV * bang_rcv
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 12:18] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    palm_sat = W_PALM_SAT_SRC * sat_src + W_PALM_SAT_RCV * sat_rcv
    cmd_rate = W_RATE_PALM_SRC * (da[:, 0:6] ** 2).sum(dim=-1) \
        + W_RATE_PALM_RCV * (da[:, 12:18] ** 2).sum(dim=-1) \
        + W_RATE_HAND * ((da[:, 6:12] ** 2).sum(dim=-1) + (da[:, 18:24] ** 2).sum(dim=-1)) \
        + W_ACT_RATE * (da ** 2).mean(dim=-1)
    arm_posture = W_ARM_LIMIT * (_limit_pen(ctx.src_arm_limit_margin, MARGIN_FREE)
                                 + _limit_pen(ctx.rcv_arm_limit_margin, MARGIN_FREE)) \
        + W_ARM_SHAKE * (_shake_pen(ctx.src_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)
                         + _shake_pen(ctx.rcv_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE))

    # ------------------------------------------------------------------ safety / disturbance (verbatim)
    cup_disturb = W_SLIDE_SRC * src["slide"] + W_SLIDE_RCV * rcv["slide"] \
        + W_ONE_SIDED_SRC * src["one_sided"] + W_ONE_SIDED_RCV * rcv["one_sided"] \
        + W_PALM_SRC * src["palm_push"] + W_PALM_RCV * rcv["palm_push"] \
        + W_RUSH_SRC * src["rush"] + W_RUSH_RCV * rcv["rush"] \
        + W_FLING * (src["fling"] + rcv["fling"])
    safety = W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0) \
        + W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)) \
        + W_NESTED * nested

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        # A
        "grasp_src": W_GRASP_SRC * src["value"],
        "grasp_rcv": W_GRASP_RCV * rcv["value"],
        # B
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        # C / D / E / H / open: difference forms and a cost - none of them pays for standing still
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": -tilt_dir,
        "tilt_open": W_TILT_OPEN * tilt_open,
        "pour_sink": W_POUR_SINK * pour_sink,
        # F
        "pour_delta": W_POUR * pour_paid - W_FLOW * flow_excess,
        "transferred": W_TRANSFER * in_t * out_gate,
        "transfer_tail": W_TAIL * finish_q * out_gate,
        "spill_delta": -W_SPILL * spill_step,
        "unaimed_release": -W_UNAIMED * unaimed,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype) * clean_f,
        # G
        "src_speed": -W_SRC_SPEED * src_speed,
        "src_spin": -src_spin,
        "rcv_steady": -rcv_steady,
        "act_flip": -act_flip,
        "palm_bang": -palm_bang,
        "palm_sat": -palm_sat,
        "cmd_rate": -cmd_rate,
        "arm_posture": -arm_posture,
        # safety
        "cup_disturb": -cup_disturb,
        "safety": -safety,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value

    # diagnostics only (added AFTER the sum)
    comps["diag_place_dist_r"] = d_place / r
    comps["diag_place_az_deg"] = torch.atan2(rel[:, 1], rel[:, 0]) * (180.0 / math.pi)
    comps["diag_side_excess_r"] = (es_lo + es_hi) / r
    comps["diag_rcv_mouth_dz_r"] = dz_m / r
    comps["diag_rcv_mouth_y"] = y_m
    comps["diag_aim_dist_r"] = d_aim / r
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_lip_dz_r"] = geo["lip_dz"] / r
    comps["diag_h_org"] = h_org
    comps["diag_win"] = win
    comps["diag_perm"] = perm
    comps["diag_toward"] = toward
    comps["diag_theta_rate"] = theta_rate
    comps["diag_open_rate"] = open_rate
    comps["diag_first_open"] = first_open
    comps["diag_spill_w"] = spill_w
    comps["diag_dir_err_deg"] = psi * (180.0 / math.pi)
    comps["diag_src_tilt_deg"] = ctx.src_cup_tilt * (180.0 / math.pi)
    comps["diag_lift_budget"] = lift_free_src
    comps["diag_ctrl_gate"] = ctrl_gate
    comps["diag_stage_cd"] = stage_cd
    comps["diag_release_beads"] = release_step * n_beads
    comps["diag_unaimed_beads"] = unaimed * n_beads
    comps["diag_clean"] = clean_f
    comps["diag_finish_q"] = finish_q
    comps["diag_grip_keep_src"] = src["keep"]
    comps["diag_balance_src"] = src["balance"]
    comps["diag_palm_sat_src"] = sat_src
    comps["diag_src_w"] = w_src
    comps["diag_src_v"] = v_src
    comps["diag_src_margin_min"] = torch.clamp(ctx.src_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_state_income"] = comps["grasp_src"] + comps["grasp_rcv"] + comps["lift_src"] + comps["lift_rcv"]
    comps["diag_outcome_income"] = comps["transferred"] + comps["transfer_tail"] + comps["success_bonus"]
    return reward, comps
```

## 근거 메모

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 줄) |
|---|---|---|
| place_rcv: 받는 컵 입구 목표를 절대 높이(테이블 위 컵의 림 + 0.15 m)에 둠 | i21 목표가 소스를 따라 올라감. 받는 컵 z 0.31~0.37, 수평 23.4 cm | 860-861, 866, 849-850 |
| place_rcv: 수평은 주둥이가 입구에서 1 r 안에 들 때까지 끌어당김 | lip_xy 0.15~0.42 m, place_dist 5~15 r, place_rcv 0.08/스텝 | 866-867, 670, 675, 692 |
| place_rcv: 받는 컵 입구 \|y\| 가 2 r 를 넘으면 비용 | 두 컵 y: 소스 -0.17, 리시버 +0.10. y_mid 는 -0.03 이라 i21 무료 구간 안에 있었음 | 750, 851, 685, 867 |
| place_rcv: 부을 수 있는 쪽 가드(68±45°) | place_az 가 61°에서 104°로 커짐(K2 범위 가장자리) | 760, 759, 674, 801 |
| aim_lip: 주둥이 천장 4 r(높이 창 3.3 r > 1/4 회전 때 주둥이 하강 약 2.9 r) | 주둥이가 입구보다 +13.8 cm 위. lip_dz 평균 0.12 m. lift_src 가 높이 머물기에 값을 줌 | 849, 669, 862, 867, 763-764 |
| RCV_POUR_LIFT 0.15 m(높이 허용만 예외) | i21 리시버 들기 최대 0.182 m. 이 높이면 소스가 붓는 동안 들기 0.23~0.34 m 로 LIFT_TARGET_SRC 0.30 과 겹침 | 712, 820, 761, 800 |
| tilt_open(신규): 창 안에서만 받는 컵 쪽 기울기 속도에 차분 보상, 스텝당 0.5 이하, 첫 알이 들어가면 꺼짐 | 기울기에 양의 수입이 없었음. 최대 기울기 중앙 48°, 곡선 약 20° | 858-859, 868-869, 847, 678 |
| tilt_open 을 win*theta+ 의 정확한 변화율로 계산 | 창 들락날락 반복으로 돈을 버는 것 방지. 각도 목표·대역 없음 | 748, 817, 869 |
| tilt_dir: 창 밖 1.5, 창 안 0.3 | 방향 오차 44°, toward 0.9, perm 0.45 | 870, 664, 683, 673, 770 |
| spill_delta: 겨냥한 창 안에서 첫 알이 들어가기 전 x0.25(한 알 -3, 배달 +7.5 보다 여전히 손해) | 소스를 떠난 146 알 중 2 알만 들어감. 첫 한 알이 나가야 경사가 생김 | 852, 870-871, 778, 813 |
| perm 에 천장 반영(너무 높은 곳에서 떨어진 비드는 unaimed) | 창은 cup_radius 배수로 유지 | 750, 773, 867 |
| G 제어·안전·lift_src·결과 가중치는 그대로 | 속도·제어 벌점 완화 금지. 나머지 설계는 iter_21 그대로 | 749, 864, 873-874, 125 |
| 예산: 새 상태 수입 없음(C·D·tilt_open 모두 차분 형태) | 상태 수입 약 3.0/스텝, 결과 수입 약 0 | 808-812, 682, 671 |

Assumptions (가설):
- The step dt is not given in the prompt, so the episode total of each difference-form term (at most the per-step cap) cannot be converted to absolute numbers.
- src_cup_ang_vel is assumed to be in the world frame, the same assumption iter_21's _point_vel already makes.
- Whether the receiver arm can actually lift to 0.15 m and move toward the centreline has not been checked.

There are 26 summed terms, one over the brief's target of 25. The extra one is tilt_open, which the operator asked for (change 2).
