import torch
import math


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    iter_07: unchanged since iter_04 — the grasp works (src_grasped 0.59, rcv_grasped 0.60, both best-ever)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (0/336 knock-overs at +45 mm) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm, inside the 0-25 mm lift group)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin (no lift with thumb >= 50 mm)

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)        # (N,)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values   # nearest non-thumb tip, (N,)

    # opposition: 1 when the thumb and the other fingers sit on opposite sides of the cup axis
    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # (N,)

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup_terms(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok):
    """Unweighted per-arm terms in [0, 1]. Reach / grasp / close / lift-state and every penalty are
    iter_06 VERBATIM — they are on the keep list (grasp_src 1.11, grasp_rcv 0.70-0.81 measured, both
    lift bands held for 800 epochs)."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: long-range funnel term + per-side fingertip-on-wall term (unchanged).
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: balanced two-sided pinch (unchanged). Flag boundary 1 N / 1 N: 0.355 -> 0.75, never a valley.
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # trace: thumb 3-5 N vs fingers 2.5-5 N -> ~0.98
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok
    # NOTE: pose_ok = src_up rides on the attitude CORRIDOR. iter_07 widens that corridor to 65 deg once
    # the cup is carried, which is what stops tilting from silently taxing this term (see compute_reward).

    # closure shaping (unchanged): measured closure 0.55-0.65 on both hands while holding
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift STATE only. 5 mm dead band: tipping on the rim is not lifting.
    h_pos = torch.clamp(cup_pos[:, 2] - cup_spawn_pos[:, 2] - 0.005, min=0.0)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    v_lin = cup_lin_vel.norm(dim=-1)

    # penalties (unweighted, bounded) — all unchanged from iter_04
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)        # 1 N dead band: a stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "h_pos": h_pos, "lifted_w": lifted_w,
            "held": g_f, "pinch": pinch, "disp": disp, "v_lin": v_lin, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling,
            "far": torch.tanh(4.0 * geo["d_funnel"])}


def _pour_geometry(ctx, outer_r, aim_k, aim_sigma):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre,
    plus the SINGLE smooth aim factor that iter_07 uses both to open the attitude corridor and to pay the
    release tilt. Geometry (lip, clearance, half-weighted height error) is iter_06 verbatim.

    iter_06 used two Gaussians here and both were numerically dead at the distance the run holds
    (measured d_align 0.30 m from cups_center_dist 0.358 m + src lift 0.106 m + rcv lift 0.054 m):
    pour_gate exp(-(0.30/0.15)^2) = 0.018 and align_gate exp(-(0.30/0.12)^2) = 0.002. That is the whole
    reason pour_tilt never left 0.002 and the corridor stayed pinned at 45 deg."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    # 0.33 at the measured 0.30 m, 0.55 at 0.20 m, 0.80 at 0.10 m — a live gradient over the exact
    # interval the run occupies; still only 0.12 at 0.45 m and 0.04 at 0.60 m, so dumping beads from
    # far away is never bought.
    aim_shape = 0.5 * (1.0 - torch.tanh(aim_k * d_align)) + 0.5 * torch.exp(-(d_align / aim_sigma) ** 2)
    return d_align, aim_shape


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt < pour.
    # iter_07 is a focused change to the TILT stage only. Measured: every earlier stage held for 800
    # epochs (cups 0.875 -> 0.358 m, src lift 0.35 -> 0.106 m, rcv lift 0.221 -> 0.054 m, both grasps
    # best-ever) while task/src_tilt_deg sat in 21-27 deg for 380 epochs. At the d_align 0.30 m that this
    # state implies, iter_06's pour_gate was 0.018 and its corridor opener align_gate 0.002, so past 45 deg
    # the only paying term was switched off AND the corridor charged grasp/lift income for the excess.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.80-0.87 measured:
    W_AWAY_RCV = 0.8                       # it keeps the left hand at its own cup)
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (best-ever grasp rates)
    # --- source lift: band + over-lift ramp. KEEP LIST — held 0.106 m for 800 epochs ----------------
    W_LIFT_SRC = 3.0
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.15
    LIFT_FREE_SRC = 0.24
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5       # measured over_lift_src ~ -0.01: the policy sits inside the band, not dragged
    OVER_LIFT_RANGE = 0.20
    LIFT_FLOOR = 0.60
    SRC_UP_FLOOR = 0.5
    LIFT_PINCH_BONUS = 0.2
    # --- receiver lift band — KEEP LIST (rcv lift 0.221 -> 0.054 m, lift_rcv 0.024 -> 0.878) --------
    W_LIFT_RCV = 2.0
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10
    RCV_LIFT_SIGMA = 0.07
    W_HOLD_RCV = 1.2        # measured hold_rcv 0 -> 0.112: the receiver income is being collected
    RCV_DISC_FLOOR = 0.35
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7
    # --- convergence / carry / align — KEEP LIST (cups 0.875 -> 0.358 m, align 0.201 -> 0.848) ------
    W_CONVERGE = 3.0
    CONV_FAR = 1.00
    CONV_NEAR = 0.16
    CONV_SIGMA = 0.25
    W_CARRY, W_ALIGN = 3.0, 5.0
    CARRY_K, CARRY_RANGE = 1.2, 1.00
    ALIGN_K, ALIGN_SIGMA = 2.0, 0.15
    # --- TILT LADDER: three segments, one continuous slope 0 -> 100 deg ----------------------------
    # Slope per degree, evaluated at the measured operating point (cups 0.36 m, d_align 0.30 m):
    #   lead 0.033  ->  rise 0.093  ->  pour 0.054 (and 0.137 once d_align reaches 0.10 m).
    # Nowhere in 0..100 deg is the tilt income flat, which is the whole point of this round.
    W_TILT_LEAD = 2.0       # was 1.5. At the 45 deg cap the ENTIRE tilt stage was worth 1.11/step at the
                            # measured distance — the same as grasp_src (1.11) and less than lift_src (1.41)
    TILT_PRE = math.radians(45.0)    # end of the lead-in segment (unchanged angle)
    W_TILT_RISE = 2.5       # NEW 45 -> 65 deg bridge. Same gates as tilt_lead, so nothing switches at 45 deg
    TILT_SAFE = math.radians(65.0)   # first bead leaves at ~70 deg with a full cup -> 5 deg of margin:
                                     # this whole segment is spill-free at ANY distance
    W_TILT = 8.0            # unchanged weight, now spanning 65 -> 100 deg and paid by aim_shape
    TILT_FULL = math.radians(100.0)
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # Gaussian width on the corridor EXCESS (unchanged)
    TILT_LIFT_H = 0.03      # corridor and tilt_lead fully open at 3 cm — the policy lives at 0.106 m
    LEAD_FLOOR = 0.25       # distance factor of lead+rise is convergence, not a gate (unchanged form)
    AIM_K = 3.0             # far tail: 0.12 at 0.45 m, 0.04 at 0.60 m over the visited aim_dist range
    AIM_SIGMA = 0.30        # = the measured operating d_align. Was an 0.15 m Gaussian (0.018 there) for the
                            # pay and an 0.12 m one (0.002) for the corridor — both numerically dead.
    # --- receiver discipline — KEEP LIST -----------------------------------------------------------
    W_HOME_RCV = 0.8
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5
    STILL_FREE, STILL_SCALE = 0.05, 0.15
    # --- task / penalties (all unchanged) ----------------------------------------------------------
    W_POUR = 150.0          # x d_in_target: 7.5 per bead with 20 beads
    W_SPILL = 60.0          # -3 per bead spilled; beads_left also deletes the whole tilt income
    W_SUCCESS = 20.0
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_TILT_PEN = 0.3
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05
    CUP_OUTER_R = 0.0285    # 0.044 x cup_scale 0.65
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35

    dtype = ctx.src_cup_tilt.dtype
    d_align, aim_shape = _pour_geometry(ctx, CUP_OUTER_R, AIM_K, AIM_SIGMA)

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)

    # ---- source attitude CORRIDOR: free to 65 deg once carried, to 100 deg once aimed --------------
    # 0.15 rad on the table -> 65 deg when grasped and 3 cm up -> 100 deg when the lip is over the
    # receiver. The 45 -> 65 widening is bought by the measured spill threshold (first bead at ~70 deg):
    # below 65 deg nothing can leave the cup, so charging for it only taxed grasp_src and lift_src.
    # A toppled cup on the table is still 1.57 rad against 0.15 rad, so src_up ~ 0 exactly as before.
    carry_soft_src = g_src * torch.clamp(h_src / TILT_LIFT_H, max=1.0)
    tilt_free = TILT_FREE_TABLE + (TILT_SAFE - TILT_FREE_TABLE) * carry_soft_src
    tilt_free = tilt_free + (TILT_FULL - TILT_SAFE) * aim_shape   # 77 deg at 0.30 m, 89 at 0.15, 93 at 0.10
    tilt_excess_src = torch.clamp(ctx.src_cup_tilt - tilt_free, min=0.0)
    src_up = torch.exp(-(tilt_excess_src / TILT_TOL) ** 2)
    pose_ok_src = src_up
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                          ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up,
                          ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_src)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                          ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                          ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_rcv)

    # ---- receiver progress gate on the source lift (unchanged) -------------------------------------
    up_w_src = g_src * torch.clamp(h_src / 0.05, max=1.0)      # "held and at least 5.5 cm up"
    up_w_rcv = g_rcv * torch.clamp(h_rcv / 0.05, max=1.0)
    prog_reach = torch.clamp((rcv["reach"] - REACH_AWAY) / (REACH_NEAR - REACH_AWAY), 0.0, 1.0)
    prog_grasp = torch.clamp(rcv["grasp"] / 0.4, max=1.0)
    rcv_prog = PROG_REACH * prog_reach + PROG_GRASP * torch.maximum(prog_grasp, up_w_rcv)
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # 0.60 .. 1.00 (measured rcv_prog 0.62-0.68)

    # ---- source lift: band + linear cost above it (unchanged — measured 0.106 m for 800 epochs) ----
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_lift_band = torch.exp(-(torch.clamp(h_src - LIFT_FREE_SRC, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_lift_band * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])
    # ungated by the grasp flag on purpose: throwing the cup up must not dodge the cost
    over_lift = torch.clamp((h_src - LIFT_FREE_SRC) / OVER_LIFT_RANGE, 0.0, 1.0)

    # ---- receiver lift band (unchanged — measured 0.054 m, inside the 0.10 m band) -----------------
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * rcv_band * (0.5 + 0.5 * upright_rcv)

    # ---- receiver discipline: home, still, and one factor on all convergence income (unchanged) ----
    d_home = rcv["disp"]
    home_near = torch.exp(-(d_home / HOME_SIGMA) ** 2)
    still_f = 1.0 - torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    rcv_disc = RCV_DISC_FLOOR + (1.0 - RCV_DISC_FLOOR) * rcv_band * home_near
    rcv_home_pen = rcv["held"] * torch.tanh(torch.clamp(d_home - HOME_FREE, min=0.0) / HOME_SCALE)
    still_w = W_STILL_RCV + W_STILL_POUR * up_w_src               # stricter while the source is carried
    rcv_still_pen = still_w * rcv["held"] \
        * torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    hold_rcv = W_HOLD_RCV * g_rcv * rcv_band * home_near * still_f * (0.4 + 0.6 * carry_soft_src)

    # ---- convergence (unchanged): the SOURCE closes the gap, on the quantity the success test uses --
    d_cups_xy = (ctx.src_cup_pos[:, :2] - ctx.rcv_cup_pos[:, :2]).norm(dim=-1)
    conv_lin = torch.clamp((CONV_FAR - d_cups_xy) / (CONV_FAR - CONV_NEAR), 0.0, 1.0)
    conv_close = torch.exp(-(torch.clamp(d_cups_xy - CONV_NEAR, min=0.0) / CONV_SIGMA) ** 2)
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close     # 0.65 at the measured 0.358 m, 0.96 at 0.20 m
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.60 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * (1.0 - nested) * conv_shape

    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * (1.0 - nested)
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- TILT: three segments of ONE slope. Only the density changes at 45 deg, never the gates -----
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    # beads_left is the anti-spill interlock: beads on the table delete the whole ladder (3.3-5.2 / step
    # at the measured distance) for the rest of the episode, on top of spill_delta's -3 per bead.
    tilt_hold = carry_soft_src * beads_left * (1.0 - nested)          # "grasped, carried, still full"
    lead_dist = LEAD_FLOOR + (1.0 - LEAD_FLOOR) * conv_shape          # 0.74 at the measured 0.358 m
    lead_frac = torch.clamp(ctx.src_cup_tilt / TILT_PRE, max=1.0)                       # 0 -> 45 deg
    rise_frac = torch.clamp((ctx.src_cup_tilt - TILT_PRE) / (TILT_SAFE - TILT_PRE), 0.0, 1.0)  # 45 -> 65
    pour_frac = torch.clamp((ctx.src_cup_tilt - TILT_SAFE) / (TILT_FULL - TILT_SAFE), 0.0, 1.0)  # 65 -> 100
    tilt_lead = tilt_hold * lead_frac * lead_dist
    # the 45-65 deg bridge: identical gating to tilt_lead (spill-free by geometry, so no distance gate),
    # 2.7x the per-degree density so the ladder steepens exactly where the policy has been stuck
    tilt_rise = tilt_hold * rise_frac * lead_dist
    # the release segment: now a consequence of convergence (aim_shape) instead of a dead 0.15 m Gaussian
    pour_tilt = both_up * aim_shape * beads_left * upright_rcv * (1.0 - nested) * pour_frac

    tilt_pen_src = torch.tanh(tilt_excess_src / TILT_TOL)       # corridor excess only
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "reach_src": W_REACH_SRC * src["reach"],
        "reach_rcv": W_REACH_RCV * rcv["reach"],
        "away_rcv": -W_AWAY_RCV * rcv["far"],
        "close_src": W_CLOSE_SRC * src["close"],
        "close_rcv": W_CLOSE_RCV * rcv["close"],
        "grasp_src": W_GRASP * src["grasp"],
        "grasp_rcv": W_GRASP * rcv["grasp"],
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "over_lift_src": -W_OVER_LIFT * over_lift,
        "hold_rcv": hold_rcv,
        "converge": converge,
        "carry_src": W_CARRY * carry,
        "align_mouths": W_ALIGN * align,
        "tilt_lead": W_TILT_LEAD * tilt_lead,
        "tilt_rise": W_TILT_RISE * tilt_rise,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE_SRC * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE_RCV * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
        "rcv_home": -W_HOME_RCV * rcv_home_pen,
        "rcv_still": -rcv_still_pen,
        "contact_push": -(W_ONE_SIDED_SRC * src["one_sided"] + W_ONE_SIDED_RCV * rcv["one_sided"]
                          + W_PALM_SRC * src["palm_push"] + W_PALM_RCV * rcv["palm_push"]),
        "rush_near_cup": -(W_RUSH_SRC * src["rush"] + W_RUSH_RCV * rcv["rush"]),
        "cup_fling": -W_FLING * (src["fling"] + rcv["fling"]),
        "cup_collision": -W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0),
        "hand_foreign": -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                      + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)),
        "nested": -W_NESTED * nested,
        "action_rate": -W_ACT_RATE * act_rate,
        "rcv_progress": torch.zeros_like(rcv_prog),
        "aim_gate": torch.zeros_like(aim_shape),
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    # diagnostics only (added AFTER the sum, not part of the reward): rcv_progress as before, and the
    # aim factor itself so the next round can tell "not tilting" from "gate closed" without guessing.
    comps["rcv_progress"] = rcv_prog
    comps["aim_gate"] = aim_shape
    return reward, comps
