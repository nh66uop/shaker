import torch
import math


# =====================================================================================================
# iter_24 = iter_23 design + ONLY the three round-24 operator changes (09.26):
#   (1) aim_lip / place_rcv / tilt_open rates = POSITION DIFFERENCE of their potential against the previous
#       control step: rate = (now - prev) / CONTROL_DT, CONTROL_DT = 1/60 s (sim 1/120 x decimation 2).
#       Potentials unchanged: d_aim, d_place, PHI = win * theta+ * max(cos psi, 0). The previous values live in a
#       module-level cache (operator decision, the one exception to "no state"): refreshed on EVERY call, delta = 0
#       on the first call, when N / device / dtype changed (validator dry run) and in envs whose episode_progress
#       did not increase (new episode). i23 trace: the ctx velocity readings of a held cup are 2-4x the real motion
#       and biased (aim paid 3 vs a position difference of 106 per episode at ep400, parked aim charged -27..-61,
#       tilt_open taxed -1..-7 per episode with a zero PHI difference). Every velocity-built derivative (v_mouth,
#       du, dd_hat, dlip_*, dwin, phi_rate, dclear, dr_free, des_*) is gone. Weights, gates and clamps unchanged.
#   (2) grasp value counts the fingers on the own cup (user decision Q9, K = 5):
#       held_val = 0.4 + 0.2 contact_q + 0.2 keep + 0.2 n_on / 4, n_on = non-thumb fingers pressing > 1.0 N
#       (same threshold as the env grasp flag and diag_*_n_contact). Max still 1.0, ladder grasped >= 0.4 >
#       free <= 0.39 unchanged, W_GRASP 1.5 unchanged -> state-income ceiling unchanged. A 3-finger grip (n_on 2)
#       is 0.1 below a 5-finger grip. i23: all 15 envs that tilted past 60 deg lost the source grasp.
#   (3) W_TILT_OPEN 0.1 -> 0.5 (meaningful only with (1)): episode ceiling 0.5 * pi * 60 = 94 (47 at 90 deg,
#       ~31 at 60 deg ~= the measured S3 - S2 transition cost of ~33), far below one clean pour (+8262).
# Unchanged on purpose: every window, the placement target, lift, tilt_dir (0.3 / 1.5), first-bead spill relief,
# every outcome weight, every speed / control / safety penalty and its gate (no angle target, no speed relaxation).
# =====================================================================================================

_STEP_CACHE = {}   # iter_24 change 1: previous control step's potentials per env (detached clones)


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _smoothstep(x):
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _late_ramp(x, lo, hi):
    """0.5*t + 0.5*t^2: never flat above lo, last part worth 3x the first."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _progress(rate_closing, v_scale):
    """Difference-form payment, positive = getting closer. Symmetric clamp (iter_21): +-1 per step, reached at the
    stable-speed limit v_scale, so going faster buys nothing and exploration jitter nets zero.
    iter_24: rate_closing is a position difference per control step, not a velocity reading."""
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


def _step_deltas(cur):
    """iter_24 change 1: per-env change of each potential since the previous call (= previous control step).
    `cur` maps name -> (N,) tensor and must contain "prog" (episode_progress). Delta is 0 on the first call, when
    N / device / dtype changed (validator dry run), and in envs whose episode_progress did not INCREASE (a new
    episode, or a repeated call with no time elapsed). The cache is refreshed on every call, gated or not."""
    prog = cur["prog"]
    prev = _STEP_CACHE.get("last")
    usable = prev is not None and all(
        k in prev and prev[k].shape == v.shape and prev[k].device == v.device and prev[k].dtype == v.dtype
        for k, v in cur.items())
    if usable:
        same_ep = prog > prev["prog"]
        deltas = {k: torch.where(same_ep, v - prev[k], torch.zeros_like(v))
                  for k, v in cur.items() if k != "prog"}
    else:
        deltas = {k: torch.zeros_like(v) for k, v in cur.items() if k != "prog"}
    _STEP_CACHE["last"] = {k: v.detach().clone() for k, v in cur.items()}
    return deltas


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
              cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok, contact_f):
    """Per-arm stage-A value in [0, 1] plus the table-disturbance penalties (iter_21, held_val iter_24 change 2).
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
    # iter_24 change 2: non-thumb fingers pressing on the own cup (> contact_f, the env grasp-flag threshold).
    n_on = (finger_force[:, 1:] > contact_f).to(cup_pos.dtype).sum(dim=1)
    n_other = float(max(finger_force.shape[1] - 1, 1))                   # 4 on the RH56F1
    held_val = 0.4 + 0.2 * contact_q + 0.2 * keep + 0.2 * (n_on / n_other)   # in [0.4, 1.0], 5-finger grip = max
    free_val = 0.25 * reach + 0.10 * close + 0.10 * pre_grasp               # at most 0.39 (unchanged)
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
    """Directed-spout geometry, iter_08/09 VERBATIM (lip_pt dropped: nothing uses it any more).
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
    return {"d_hat": d_hat, "sep_ok": sep_ok, "theta": theta, "lip_drop": lip_drop,
            "lip_xy_vec": lip_xy_vec, "lip_xy": lip_xy, "lip_dz": lip_dz}


def _lean_cos(up, d_hat):
    """cos psi: angle between the cup's lean (horizontal part of its up axis) and d_hat (iter_19 formula)."""
    u_xy = up[:, :2]
    return torch.clamp((u_xy * d_hat).sum(dim=-1) / torch.clamp(u_xy.norm(dim=-1), min=1e-6), -1.0, 1.0)


def _tilt_direction_error(ctx, d_hat):
    """iter_19 VERBATIM: dir_err = (1 - cos psi)/2 is 0 when leaning at the receiver at ANY tilt."""
    cos_psi = _lean_cos(ctx.src_cup_up, d_hat)
    return 0.5 * (1.0 - cos_psi), torch.acos(cos_psi)


def _spout_window(lip_xy, lip_dz, r_in, r_out, z_lo, z_hi, z_fade):
    """Spout window in [0, 1] (iter_22 values, the velocity-built derivative is gone in iter_24).
    horizontal: 1 inside r_in, smoothstep to 0 at r_out (iter_13 window);
    height    : 0 -> 1 linearly over [0, z_lo] (iter_17 floor), 1 up to the ceiling z_hi, smoothstep to 0 at
                z_hi + z_fade."""
    w_xy = 1.0 - _smoothstep((lip_xy - r_in) / (r_out - r_in))
    w_lo = torch.clamp(lip_dz / z_lo, 0.0, 1.0)
    w_hi = 1.0 - _smoothstep((lip_dz - z_hi) / z_fade)
    w_z = w_lo * w_hi
    return {"w_xy": w_xy, "w_z": w_z, "win": w_xy * w_z}


def _tilt_open_phi(ctx, geo, win):
    """iter_22 potential PHI = win * theta+ * max(cos psi, 0). No angle target, no band: monotone in the part of
    the lean that points at the receiver, only inside the spout window."""
    cos_psi = _lean_cos(ctx.src_cup_up, geo["d_hat"])
    return win * torch.clamp(geo["theta"], min=0.0) * torch.clamp(cos_psi, min=0.0)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ------------------------------------------------------------------ constants and their sources
    W_ALIVE = 1.0            # constant survival floor, identical in every state (not A-D income)
    # A grasp (iter_21 weights; held_val iter_24 change 2)
    W_GRASP_SRC, W_GRASP_RCV = 1.5, 1.5
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35       # rad, the ENV's own receiver-upright success limit (20 deg)
    CONTACT_F = 1.0          # N, a finger counts as touching its cup above this force (env grasp flag threshold)
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
    # C placement (iter_22 target verbatim): WORLD-FIXED height for the receiver mouth, horizontal vs the SPOUT
    RCV_POUR_LIFT = 0.15     # m, height allowance (the only absolute constant brief 5-3 permits)
    Z_TOL_R = 1.0            # +-1 r free band around the mouth height target
    PLACE_WIN_R = 1.0        # horizontal: receiver mouth pulled until the spout is within 1 r (= PERM_IN_R)
    BODY_CLEAR_R = 1.0       # free radius 2 r until the source BOTTOM is 1 r above the receiver RIM (no body drag)
    PLACE_AZ = math.radians(68.0)       # K2 pourable side, kept as a GUARD only
    PLACE_AZ_TOL = math.radians(45.0)   # K1 palm-yaw box +-45 deg: 23..113 deg free, outside costs distance
    CENTER_FREE_R = 2.0      # |y| of the receiver mouth (= pour point) free up to 2 r (y ~= 0 centreline)
    W_PLACE = 1.0
    # D aim
    W_AIM = 1.5
    V_PROG = 0.15            # m/s: progress terms saturate at the stable cup speed limit (SRC_V_FREE)
    LIP_Z_HI_R = 0.7         # spout floor above the rim, 0.7 r ~= 2 cm (iter_17 permission height)
    LIP_Z_MAX_R = 4.0        # spout CEILING 4 r ~= 11 cm above the rim (iter_22)
    LIP_Z_FADE_R = 2.0       # window fades out between 4 r and 6 r
    # E direction (iter_22 verbatim)
    W_TILT_DIR_OUT = 1.5     # outside the window: iter_19/21 strength
    W_TILT_DIR_IN = 0.3      # inside the window: a fifth, the spout is already over the mouth
    # tilt_open
    W_TILT_OPEN = 0.5        # iter_24 change 3 (was 0.1). Per step at most +0.5; whole-episode income bounded by
                             # W_TILT_OPEN * PHI_max / (TILT_RATE_SCALE * CONTROL_DT) = 0.5 * pi * 60 = 94
                             # (47 at 90 deg, ~31 at 60 deg ~= the S3 - S2 transition cost), << a clean pour (+8262)
    OPEN_BEADS = 1.0         # active until the first bead has LEFT the source
    OPEN_CHARGE_CAP = 100.0  # charges ungated, capped only at 100x the payment cap (physics glitches)
    # F outcome - unchanged weights (per-bead units with 20 beads)
    W_POUR = 150.0           # +7.5 per bead delivered
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0            # -2 per extra bead dumped in the same step
    W_SPILL = 240.0          # -12 per bead spilled
    SPILL_RELIEF = 0.25      # x0.25 (-3/bead) inside the AIMED window, first bead(s) only, at most one bead
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
    TILT_RATE_SCALE = ROT_POUR_FREE   # rad/s: tilt_open payment saturates at the pour-phase spin limit
    # safety / disturbance (unchanged)
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    # iter_24 change 1: one control step (trace meta dt_env 0.016667 = sim dt 1/120 x decimation 2)
    CONTROL_DT = 1.0 / 60.0

    dtype = ctx.src_cup_tilt.dtype
    r = ctx.cup_radius
    geo = _pour_geometry(ctx, r)
    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    nested = ctx.cups_nested.to(dtype)
    not_nested = 1.0 - nested

    # ------------------------------------------------------------------ A. grasp (iter_21, held_val iter_24)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    h_src_raw = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    carry_src = g_src * torch.clamp(h_src_raw / CARRY_H, max=1.0)
    src_table_upright = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = carry_src + (1.0 - carry_src) * src_table_upright
    src = _hand_cup(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                    ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel,
                    ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z, TIP_R, pose_ok_src, CONTACT_F)
    rcv = _hand_cup(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                    ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel,
                    ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z, TIP_R, upright_rcv, CONTACT_F)
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
    released = 1.0 - fill                                 # beads no longer in the source (receiver, air, spill)
    first_out = torch.clamp(OPEN_BEADS - n_beads * released, 0.0, 1.0)   # 1 until the first bead LEFT

    # ------------------------------------------------------------------ spout window (a GATE, never a state income)
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    z_lo, z_hi = LIP_Z_HI_R * r, LIP_Z_MAX_R * r
    wnd = _spout_window(geo["lip_xy"], geo["lip_dz"], r_in, r_out, z_lo, z_hi, LIP_Z_FADE_R * r)
    win = wnd["win"]                                      # position window (horizontal x height)
    perm_xy = wnd["w_xy"]
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = win * toward                                   # aimed window: position + leaning toward
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r) / ((LEAD_ZERO_R - LEAD_FULL_R) * r))

    # ------------------------------------------------------------------ C. placement potential (iter_22 verbatim)
    # Receiver mouth: world-fixed height, horizontally until the SPOUT is within PLACE_WIN_R*r (free radius 2 r
    # until the source bottom clears the receiver rim by BODY_CLEAR_R*r), |y| near the centreline, and not
    # outside the pourable side of the source (guard). d_place = norm of the excesses.
    z_mouth_tgt = ctx.table_z - ctx.cup_bottom_z + ctx.cup_mouth_z + RCV_POUR_LIFT
    dz_m = ctx.rcv_cup_mouth_pos[:, 2] - z_mouth_tgt
    ez = torch.sign(dz_m) * torch.clamp(dz_m.abs() - Z_TOL_R * r, min=0.0)
    z_bot = ctx.src_cup_pos[:, 2] + ctx.cup_bottom_z * ctx.src_cup_up[:, 2]      # source bottom centre
    clear = z_bot - ctx.rcv_cup_mouth_pos[:, 2]                                   # height above the receiver rim
    ramp = BODY_CLEAR_R * r
    r_free = PLACE_WIN_R * r + ramp - torch.clamp(clear, 0.0, ramp)
    eh = torch.clamp(geo["lip_xy"] - r_free, min=0.0)
    y_m = ctx.rcv_cup_mouth_pos[:, 1]
    ey = torch.clamp(y_m.abs() - CENTER_FREE_R * r, min=0.0)
    rel = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    a_lo, a_hi = PLACE_AZ - PLACE_AZ_TOL, PLACE_AZ + PLACE_AZ_TOL
    n_lo_x, n_lo_y = math.sin(a_lo), -math.cos(a_lo)      # outward normal of the 23 deg edge
    n_hi_x, n_hi_y = -math.sin(a_hi), math.cos(a_hi)      # outward normal of the 113 deg edge
    es_lo = torch.clamp(rel[:, 0] * n_lo_x + rel[:, 1] * n_lo_y, min=0.0)
    es_hi = torch.clamp(rel[:, 0] * n_hi_x + rel[:, 1] * n_hi_y, min=0.0)
    d_place = torch.sqrt(eh ** 2 + ez ** 2 + ey ** 2 + es_lo ** 2 + es_hi ** 2)

    # ------------------------------------------------------------------ D. aim potential (3D, iter_22 verbatim)
    # horizontal spout->mouth distance + how far the spout is BELOW its floor + how far it is ABOVE its ceiling.
    below = torch.clamp(z_lo - geo["lip_dz"], min=0.0)
    above = torch.clamp(geo["lip_dz"] - z_hi, min=0.0)
    d_aim = torch.sqrt(geo["lip_xy"] ** 2 + below ** 2 + above ** 2)

    # ------------------------------------------------------------------ tilt_open potential (iter_22 verbatim)
    phi = _tilt_open_phi(ctx, geo, win)

    # ------------------------------------------------------------------ iter_24 change 1: step differences
    # rate = (potential now - potential at the previous control step) / CONTROL_DT. Standing still pays exactly 0,
    # whatever the velocity readings say; axial spin leaves every potential where it is and pays 0 too.
    delta = _step_deltas({"d_place": d_place, "d_aim": d_aim, "phi": phi, "theta": geo["theta"],
                          "prog": ctx.episode_progress})
    place = stage_cd * _progress(-delta["d_place"] / CONTROL_DT, V_PROG)   # clamp(-dd/(V_PROG dt), -1, +1)
    aim = stage_cd * _progress(-delta["d_aim"] / CONTROL_DT, V_PROG)
    # tilt_open per step: first_out * ( carry_both * sep_ok * clamp(dPHI/(scale dt), 0, +1)
    #                                   + clamp(dPHI/(scale dt), -OPEN_CHARGE_CAP, 0) )
    # Payments <= the potential gained and charges = the potential lost, so every closed cycle sums to <= 0
    # while first_out is constant; tilting and then holding still earns nothing more (only the beads do).
    rate_n = delta["phi"] / (TILT_RATE_SCALE * CONTROL_DT)
    open_pay = carry_both * geo["sep_ok"] * torch.clamp(rate_n, 0.0, 1.0)
    open_charge = torch.clamp(rate_n, -OPEN_CHARGE_CAP, 0.0)
    tilt_open = first_out * (open_pay + open_charge)
    theta_step = torch.remainder(delta["theta"] + math.pi, 2.0 * math.pi) - math.pi   # diagnostic, wrap-safe

    # ------------------------------------------------------------------ E. direction (cost only, iter_22 verbatim)
    dir_err, psi = _tilt_direction_error(ctx, geo["d_hat"])
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)          # magnitude only as a multiplier, no target
    dir_w = W_TILT_DIR_IN * win + W_TILT_DIR_OUT * (1.0 - win)   # strong only outside the window
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err * dir_w

    # ------------------------------------------------------------------ H. removed (iter_23 change 2, kept out)
    h_org = ctx.src_cup_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]   # kept for the existing diagnostic only

    # ------------------------------------------------------------------ F. outcome (the beads, iter_22 verbatim)
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    # A spilled bead has necessarily LEFT the source; the relief gate uses the count WITHOUT this step's spill and
    # relieves at most one bead's worth: x0.25 (-3 instead of -12, still a loss vs +7.5 delivered).
    spill_now = torch.clamp(ctx.d_spill, min=0.0)
    first_spill = torch.clamp(OPEN_BEADS - n_beads * (released - spill_now), 0.0, 1.0)
    relieved = perm * first_spill * torch.clamp(spill_now, max=OPEN_BEADS / n_beads)
    spill_step = settled * (spill_now - (1.0 - SPILL_RELIEF) * relieved)
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
        # C / D / E / open: step-difference forms and a cost - none of them pays for standing still
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": -tilt_dir,
        "tilt_open": W_TILT_OPEN * tilt_open,
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
    comps["diag_body_clear_r"] = clear / r
    comps["diag_place_free_r"] = r_free / r
    comps["diag_aim_dist_r"] = d_aim / r
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_lip_dz_r"] = geo["lip_dz"] / r
    comps["diag_h_org"] = h_org
    comps["diag_win"] = win
    comps["diag_perm"] = perm
    comps["diag_toward"] = toward
    comps["diag_theta_rate"] = theta_step / CONTROL_DT         # iter_24: step difference, not omega x up
    comps["diag_open_phi"] = phi
    comps["diag_open_rate"] = delta["phi"] / CONTROL_DT       # iter_24: step difference of PHI
    comps["diag_first_out"] = first_out
    comps["diag_first_spill"] = first_spill
    comps["diag_relieved_beads"] = settled * relieved * n_beads
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
    comps["diag_src_axial_w"] = (ctx.src_cup_ang_vel * ctx.src_cup_up).sum(dim=-1)
    comps["diag_src_n_contact"] = (ctx.src_finger_force > CONTACT_F).to(dtype).sum(dim=-1)
    comps["diag_rcv_n_contact"] = (ctx.rcv_finger_force > CONTACT_F).to(dtype).sum(dim=-1)
    return reward, comps
