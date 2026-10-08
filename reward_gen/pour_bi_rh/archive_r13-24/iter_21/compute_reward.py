import torch
import math


# =====================================================================================================
# iter_20 - REDESIGN (operator brief 09.25, sections 0-7). 25 summed terms, one role each, staged A-H.
#   A grasp      : grasp_src, grasp_rcv                                   (small state)
#   B lift       : lift_src, lift_rcv                                     (small state)
#   C placement  : place_rcv        (source-relative target, difference form)
#   D aim        : aim_lip          (3D spout -> above receiver mouth, difference form)
#   E direction  : tilt_dir         (cost = tilt magnitude x direction error; no angle target)
#   H height     : pour_sink        (difference form on the source origin height, iter_19 verbatim)
#   F outcome    : pour_delta, transferred, transfer_tail, spill_delta, unaimed_release, success_bonus
#   G control    : src_speed, src_spin, rcv_steady, act_flip, palm_bang, palm_sat, cmd_rate, arm_posture
#   safety       : cup_disturb, safety
#   constant     : alive
# Removed on purpose: meet_cross / src_cross (spawn-relative rendezvous), converge, carry_src,
# align_mouths, hold_rcv, rcv_home, pour_dir_pen, settle_finish - all STATE income for a pose (brief 4-4,
# 5-2) or spawn-relative (brief 4-1, 5-3). No tilt target, band, release ladder or angle ceiling exists.
# =====================================================================================================


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
    """Difference-form payment from an exact potential rate [m/s], positive = getting closer.
    ASYMMETRIC clamp: closing pays at most +1/step (reached at the stable-speed limit v_scale), but
    opening is charged up to -4/step, so "back off fast, come back slowly" can never net income."""
    # iter_21: symmetric (was -4..+1). Under exploration noise an asymmetric clamp makes jitter itself a net
    # loss (i20: grasp 0.69 -> 0.01 in 100 epochs while episode reward rose -384 -> +712).
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
    """Per-arm stage-A value in [0, 1] plus the table-disturbance penalties.
    Reach / pinch / close / grip_keep formulas are the validated iter_04..19 ones; only the way they
    are combined is new: ONE monotone ladder, grasped (>= 0.4) always above not-grasped (<= 0.39)."""
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
    # keep the grasp that lifted the cup: RATIO of the two contact sides (no absolute force - i24 rule)
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
    """Directed-spout geometry, iter_08/09 VERBATIM (minus the old d_align).
      d_hat : horizontal unit vector source cup origin -> receiver mouth
      theta : SIGNED tilt toward the receiver (used to LOCATE the lip only, never as a target)
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


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ------------------------------------------------------------------ constants and their sources
    W_ALIVE = 1.0            # constant survival floor, identical in every state (not A-D income)
    # A grasp. Parked value ~0.8/arm (contact_q ~0.85 at thumb 3-5 N vs fingers 2.5-5 N, keep ~0.45).
    W_GRASP_SRC, W_GRASP_RCV = 1.5, 1.5   # iter_21: 1.0 -> 1.5, holding must clearly out-earn letting go
    TIP_R = 0.027            # verbatim grip geometry constant
    UPRIGHT_TOL = 0.35       # rad, the ENV's own receiver-upright success limit (20 deg)
    # B lift. KEEP (operator item 5): LIFT_FREE_SRC0 0.40 / LIFT_TARGET_SRC 0.30 (K3 reachability probe:
    # palm z 0.69-0.76 -> roll error 10-23 deg). Weight 3.0 -> 1.5 so A+B stays under the outcome.
    W_LIFT_SRC = 1.5
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.30
    LIFT_FREE_SRC0 = 0.40
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    # receiver: a small "it is off the table" state only. The old 0.10 m ceiling is gone because the
    # placement below (receiver mouth 1.7 r under a source origin at z ~0.57) needs h_rcv ~0.2 m.
    W_LIFT_RCV = 0.5
    LIFT_TARGET_RCV = 0.08   # measured rcv lift 0.073 m
    CARRY_H = 0.03           # "carried" height, both arms (a height, not an angle)
    # C placement - SOURCE-relative target for the receiver MOUTH (brief 5-1 C, K2, K5, K6).
    PLACE_AZ = math.radians(68.0)   # K2: median lean azimuth at tilt > 60 deg (IQR 59-81). The palm
                                    # yaw box (+-45 deg, K1) is what makes +-45 around it reachable.
    PLACE_R_R = 2.0          # horizontal source-origin -> receiver-mouth distance = 2 r = 0.056 m:
                             # where the lip sits at a quarter turn (K6 mouth_z 0.054 ~= 1.9 r)
    LIP_Z_HI_R = 0.7         # spout clearance above the rim, 0.7 r ~= 2 cm (iter_17 permission height)
    SINK_KEEP_R = 1.0        # lip falls r*sin + mouth_z*(1-cos); iter_19 derivation: origin must sit
                             # (0.7 + 1.0) r = 1.7 r above the receiver rim to clear it when horizontal
    PLACE_LOW_W = 0.5        # receiver lower than the target height costs half (the aim term decides
                             # the final spout height; being too HIGH is the dangerous side)
    CENTER_FREE_R = 2.0      # |y of the cup-pair midpoint| free up to 2 r (y ~= 0 centreline, FPP view)
    W_PLACE = 1.0
    # D aim - spout onto a point LIP_Z_HI_R*r above the receiver mouth (horizontal + height, 3D)
    W_AIM = 1.5
    V_PROG = 0.15            # m/s: progress terms saturate at the stable cup speed limit (SRC_V_FREE),
                             # so going faster than allowed buys nothing
    # E direction (iter_19): (1.344/1.571)*0.801 at the i18 failure state -> 1.03/step
    W_TILT_DIR = 1.5
    # H height keep (iter_19 verbatim): i18 sank 0.617 -> 0.359 m in 6.7 s = 0.039 m/s
    W_POUR_SINK = 1.0
    V_Z_SCALE = 0.04
    # F outcome - unchanged weights (per-bead units with 20 beads)
    W_POUR = 150.0           # +7.5 per bead delivered
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0            # -2 per extra bead dumped in the same step
    W_SPILL = 240.0          # -12 per bead spilled
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
    # G control - KEEP LIST values
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, W_ROT_POUR = 1.0, 1.0  # stricter limit while the spout is at the mouth
    RCV_W_FREE, RCV_W_SCALE, W_RCV_SPIN = 0.5, 1.0, 0.5
    STILL_FREE, STILL_SCALE, W_STILL_POUR = 0.05, 0.15, 0.5
    W_TILT_PEN_RCV = 0.3
    W_FLIP_PALM, W_FLIP_HAND = 0.3, 0.2
    W_RATE_PALM_SRC, W_RATE_PALM_RCV, W_RATE_HAND = 0.03, 0.05, 0.01
    W_ACT_RATE = 0.05
    BANG_DEAD, W_BANG_SRC, W_BANG_RCV = 0.1, 1.5, 2.0
    SAT_LO, W_PALM_SAT_SRC, W_PALM_SAT_RCV = 0.7, 3.0, 1.5
    MARGIN_FREE, W_ARM_LIMIT = 0.30, 0.4  # weak rudder (i15: a strong limit penalty failed)
    QDD_FREE, QDD_SCALE = 2.5, 10.3
    QDD_PEAK_FREE, QDD_PEAK_SCALE = 12.8, 12.8
    W_ARM_SHAKE = 0.4
    # safety / disturbance
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

    # ------------------------------------------------------------------ A. grasp
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    h_src_raw = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    carry_src = g_src * torch.clamp(h_src_raw / CARRY_H, max=1.0)
    # knock-over guard only while the source cup is still on the table; no attitude opinion in the air
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

    # ------------------------------------------------------------------ B. lift
    lift_free_src = LIFT_FREE_SRC0 + geo["lip_drop"]     # geometric allowance, never an angle target
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    over_lift = torch.clamp((h_src - lift_free_src) / OVER_LIFT_RANGE, 0.0, 1.0)   # ungated on purpose
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_band - W_OVER_LIFT * over_lift
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * (0.5 + 0.5 * upright_rcv)

    # ------------------------------------------------------------------ bead bookkeeping (gates)
    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    has_beads = torch.clamp(fill / 0.10, max=1.0)        # >= 2 beads left
    stage_cd = carry_both * has_beads                     # placement / aim live while there is a pour ahead

    # velocities of the two points that matter
    v_mouth = _point_vel(ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_pos, ctx.rcv_cup_mouth_pos)
    v_lip = _point_vel(ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_pos, geo["lip_pt"])

    # ------------------------------------------------------------------ C. placement (source-relative)
    # Target for the receiver MOUTH: 2 r out from the source origin at azimuth +68 deg (front-left, the
    # only side the source can pour to - K1/K2), 1.7 r BELOW the source origin (spout clears the rim by
    # 0.7 r even at a quarter turn). The target moves with the source cup, so either arm may do the work.
    # Potential = distance to it (+ centreline excess); paid as its exact rate -> 0 when parked anywhere.
    off = torch.tensor([PLACE_R_R * math.cos(PLACE_AZ), PLACE_R_R * math.sin(PLACE_AZ),
                        -(LIP_Z_HI_R + SINK_KEEP_R)], dtype=dtype, device=ctx.src_cup_pos.device) * r
    e = ctx.rcv_cup_mouth_pos - (ctx.src_cup_pos + off.unsqueeze(0))
    de = v_mouth - ctx.src_cup_lin_vel
    k_z = torch.where(e[:, 2] > 0.0, torch.ones_like(e[:, 2]), PLACE_LOW_W * torch.ones_like(e[:, 2]))
    ez = k_z * e[:, 2]
    d_place = torch.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2 + ez ** 2)
    dd_place = (e[:, 0] * de[:, 0] + e[:, 1] * de[:, 1] + ez * k_z * de[:, 2]) \
        / torch.clamp(d_place, min=1e-4)
    y_mid = 0.5 * (ctx.src_cup_pos[:, 1] + ctx.rcv_cup_pos[:, 1])
    vy_mid = 0.5 * (ctx.src_cup_lin_vel[:, 1] + ctx.rcv_cup_lin_vel[:, 1])
    off_center = (y_mid.abs() > CENTER_FREE_R * r).to(dtype)
    dd_center = off_center * torch.sign(y_mid) * vy_mid
    place = stage_cd * _progress(-(dd_place + dd_center), V_PROG)

    # ------------------------------------------------------------------ D. aim (3D, difference form)
    # Potential: horizontal lip->mouth distance plus how far the lip is BELOW its clearance height.
    # Above that height nothing is charged, so pouring from higher is never taxed here.
    z_tgt = LIP_Z_HI_R * r
    below = torch.clamp(z_tgt - geo["lip_dz"], min=0.0)
    d_aim = torch.sqrt(geo["lip_xy"] ** 2 + below ** 2)
    dv = v_lip - v_mouth
    dd_aim = ((geo["lip_xy_vec"] * dv[:, :2]).sum(dim=-1) - below * dv[:, 2]) / torch.clamp(d_aim, min=1e-4)
    aim = stage_cd * _progress(-dd_aim, V_PROG)

    # permission (a GATE, never paid as a state): spout inside the r-window, above the rim, leaning toward
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    perm_xy = 1.0 - _smoothstep((geo["lip_xy"] - r_in) / (r_out - r_in))
    lip_above = torch.clamp(geo["lip_dz"] / z_tgt, 0.0, 1.0)
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = perm_xy * lip_above * toward
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r) / ((LEAD_ZERO_R - LEAD_FULL_R) * r))

    # ------------------------------------------------------------------ E. direction (cost only)
    dir_err, psi = _tilt_direction_error(ctx, geo["d_hat"])
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)          # magnitude only as a multiplier, no target
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err

    # ------------------------------------------------------------------ H. keep height while pouring
    org_vz = ctx.src_cup_lin_vel[:, 2] - v_mouth[:, 2]
    h_org = ctx.src_cup_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    org_low = (h_org < (LIP_Z_HI_R + SINK_KEEP_R) * r).to(dtype)
    pour_sink = ctrl_gate * not_nested * org_low * torch.clamp(org_vz / V_Z_SCALE, -1.0, 1.0)

    # ------------------------------------------------------------------ F. outcome (the beads)
    n_beads = float(max(int(ctx.num_beads), 1))
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_step = settled * torch.clamp(ctx.d_spill, min=0.0)
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
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

    # ------------------------------------------------------------------ G. control
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
        + W_STILL_POUR * ctrl_gate * perm_xy * rcv_still_over   # still only once the spout is over it

    a_abs = ctx.actions.abs()
    da = ctx.actions - ctx.prev_actions
    flips = torch.clamp(-(ctx.actions * ctx.prev_actions), min=0.0)
    flip_palm_src = flips[:, 0:6].sum(dim=-1)
    flip_palm_rcv = flips[:, 12:18].sum(dim=-1)
    flip_hand = flips[:, 6:12].sum(dim=-1) + flips[:, 18:24].sum(dim=-1)
    # iter_21: control penalties no longer multiplied by the grasp flags - in i20 releasing the cups removed
    # every one of them, so letting go out-earned holding. They now cost the same whether or not a cup is held.
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

    # ------------------------------------------------------------------ safety / disturbance
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
        # C / D / E / H: difference forms and a cost - none of them pays for standing still
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": -W_TILT_DIR * tilt_dir,
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
    comps["diag_place_az_deg"] = torch.atan2(ctx.rcv_cup_mouth_pos[:, 1] - ctx.src_cup_pos[:, 1],
                                             ctx.rcv_cup_mouth_pos[:, 0] - ctx.src_cup_pos[:, 0]) * (180.0 / math.pi)
    comps["diag_y_mid"] = y_mid
    comps["diag_aim_dist_r"] = d_aim / r
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_h_org"] = h_org
    comps["diag_perm"] = perm
    comps["diag_toward"] = toward
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
