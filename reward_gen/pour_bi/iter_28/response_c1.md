1. Task and stages

The goal is a bimanual pour. The right (source) hand grips the bead cup with a five-finger power grasp. The left (receiver) hand grips the empty cup the same way. Both cups are lifted upright. The source cup is brought over the receiver, which stays upright near its home on its own side, and tilted until the requested number of beads falls through the air into the receiver. The source cup is then set upright again at its pre-tilt pose while both hands still hold their cups.

Stages:
- (0) Settle.
- (1) Reach each cup with its own hand.
- (2) Wrap all five fingers around the cup.
- (3) Lift both cups upright. The env records the lift reference at 3 cm if the tilt is below 20 deg.
- (4) Carry. The receiver stays near home, upright and still. The source moves toward the centre line, above the receiver rim, with the spout within about 2 r of the receiver mouth.
- (5) Turn the source lean into the pour plane (d, z) and tilt toward the receiver. Tilt past the premature limit only once the lip is near the mouth.
- (6) Pour exactly the requested amount. Reward the flow, stop at the target, and do not spill.
- (7) Set the source cup upright at the pre-tilt pose while both cups are still held. Success.

What the t2r_i27 data shows:
- The tilt stops at 0.72-0.79 of premature_tilt_limit. That is where the premature ramp starts, and the spout is still 2-3 r from the mouth there. At that distance Phi is below 1e-7, so tilt_rate pays nothing.
- The source cup is tilted more than 20 deg before it has risen 3 cm. The lift reference is therefore never recorded, and the pre-lift posture income (up to 3.0 per step) is paid for the whole carry.
- The receiver lifts with 4 fingers. kept_rcv divides by the number of fingers that were on at lift, so losing the fifth finger costs nothing.

The three required changes are applied and nothing else is touched.

```python
import torch
import math


def _step(x: torch.Tensor) -> torch.Tensor:
    """smoothstep on an already normalised argument -> [0,1]"""
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


def _grasp_quality(finger_force: torch.Tensor, palm_force: torch.Tensor,
                   wrap_count: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """[0,1] quality of one hand on its own cup BEFORE the lift reference exists. Only the SHAPE of
    the contact enters (wrapped fingers, palm contact, thumb contact), no cup dimension.
    iter_27: wrapped fingers are counted out of ALL fingers of the hand (F from the tensor shape)."""
    n_fingers = float(max(int(finger_force.shape[1]), 1))
    wrap_q = torch.clamp(wrap_count.to(dt) / n_fingers, 0.0, 1.0)
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)      # [N] force sensor scale
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    return 0.5 * wrap_q + 0.25 * palm_q + 0.25 * thumb_q


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """iter_24, KEPT UNCHANGED: how closely the hand still holds the cup the way it did at the
    instant it lifted it (hand pose in CUP coordinates, force distribution and force level RELATIVE
    to that instant). Never an absolute force target. Before the reference exists the factor is 1."""
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = torch.clamp(0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim, 0.0, 1.0)
    return torch.where(valid, keep, torch.ones_like(keep))


def _fingers_kept(ref_share, ref_total, finger_force, valid, on_share, keep_frac):
    """iter_28 change 3 (BOTH hands): fraction of ALL F fingers (F = finger_force.shape[1]) that were
    on the cup AT LIFT and still press with at least keep_frac of their OWN lift force (ref share *
    ref total). A finger that was not on at lift (share <= on_share) counts as not kept, so a 4-finger
    lift keeps at most 4/F. Purely relative to the lift hold: no absolute force floor, no force
    target. 1 before the reference exists. share = [palm, thumb..pinky] -> fingers are share[:, 1:]."""
    n_fingers = float(max(int(finger_force.shape[1]), 1))
    ref_f = torch.clamp(ref_share[:, 1:], min=0.0) * ref_total[:, None]     # [N] each finger at lift
    ref_on = (ref_share[:, 1:] > on_share).to(finger_force.dtype)
    need = torch.clamp(keep_frac * ref_f, min=1e-6)                         # numeric guard only
    touch = _step(torch.clamp(finger_force, min=0.0) / need)
    kept = torch.sum(ref_on * touch, dim=-1) / n_fingers
    return torch.where(valid, kept, torch.ones_like(kept))


def _spout(lip, rcv_mouth, cup_r, geo):
    """iter_26 spout geometry, unchanged (used by aim): (horizontal lip offset, fine, height).
    geo = (precision width, below-rim width, drop width, drop floor); widths in cup radii."""
    prec_r, below_r, drop_r, drop_floor = geo
    delta = lip - rcv_mouth
    lip_xy = torch.norm(delta[:, :2], dim=-1)
    lip_dz = delta[:, 2]
    fine = torch.exp(-(lip_xy / (prec_r * cup_r)) ** 2)                 # spout centred in the mouth
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (below_r * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (drop_r * cup_r)) ** 2)
    height = over_rim * (drop_floor + (1.0 - drop_floor) * low_drop)
    return lip_xy, fine, height


def _reach_over_rim(lip, rcv_mouth, cup_r, reach_r, below_r):
    """iter_28 change 1: spout geometry INSIDE Phi only. reach = gaussian of the horizontal spout
    offset with width reach_r cup radii (the env's premature latch radius); over_rim = the iter_26
    factor for a spout below the receiver rim. No drop factor: raising the spout above the receiver
    rim leaves Phi unchanged, so the rate does not charge it (the drop height stays in aim)."""
    delta = lip - rcv_mouth
    lip_xy = torch.norm(delta[:, :2], dim=-1)
    reach = torch.exp(-(lip_xy / (reach_r * cup_r)) ** 2)
    over_rim = torch.exp(-(torch.clamp(-delta[:, 2], min=0.0) / (below_r * cup_r)) ** 2)
    return reach * over_rim


def _turned(vec, ang_vel, tau):
    """vec carried by a body that turns at ang_vel, as it was tau seconds EARLIER (Rodrigues)."""
    wn = torch.norm(ang_vel, dim=-1, keepdim=True)
    k = ang_vel / torch.clamp(wn, min=1e-9)
    phi = -wn * tau
    return vec * torch.cos(phi) + torch.cross(k, vec, dim=-1) * torch.sin(phi) \
        + k * torch.sum(k * vec, dim=-1, keepdim=True) * (1.0 - torch.cos(phi))


def _lean_reach(src_pos, src_up, src_mouth, rcv_mouth, lip_r, cup_r, reach_r, below_r):
    """G = reach * over_rim * lean of ONE state, rebuilt with the ctx formulas: d = horizontal unit
    vector source cup origin -> receiver mouth, theta = signed tilt toward d in the (d, z) plane,
    lip = source mouth + r (cos theta d - sin theta z), lean = max(up . d, 0) (as in iter_27)."""
    h = rcv_mouth - src_pos
    hn = torch.clamp(torch.sqrt(h[:, 0] ** 2 + h[:, 1] ** 2), min=1e-6)
    dx, dy = h[:, 0] / hn, h[:, 1] / hn
    toward = src_up[:, 0] * dx + src_up[:, 1] * dy                      # axis component toward d
    theta = torch.atan2(toward, src_up[:, 2])
    c, s = torch.cos(theta), torch.sin(theta)
    lip = src_mouth + lip_r[:, None] * torch.stack([c * dx, c * dy, -s], dim=-1)
    geo_phi = _reach_over_rim(lip, rcv_mouth, cup_r, reach_r, below_r)
    return geo_phi * torch.clamp(toward, min=0.0)


def _lean_reach_rate(ctx, lip_r, cup_r, reach_r, below_r, tau, dt):
    """(G(both cups moved ahead by tau) - G(both cups moved back by tau)) / (2 tau) [1/s], moved
    along their lin/ang velocities. The receiver mouth, d, theta and the lip are all rebuilt at both
    states, so this is the FULL rate of G. Central difference as in iter_27 c1r; only G changed.
    (hypothesis, as in iter_26/27: velocities are in the env frame of the positions; ctx lin_vel is
    the COM velocity, taken as the cup origin's)"""
    sp, su = ctx.src_cup_pos.to(dt), ctx.src_cup_up.to(dt)
    sm, rp, rm = ctx.src_cup_mouth_pos.to(dt), ctx.rcv_cup_pos.to(dt), ctx.rcv_cup_mouth_pos.to(dt)
    vs, ws = ctx.src_cup_lin_vel.to(dt), ctx.src_cup_ang_vel.to(dt)
    vr, wr = ctx.rcv_cup_lin_vel.to(dt), ctx.rcv_cup_ang_vel.to(dt)
    sp_f, sp_b = sp + tau * vs, sp - tau * vs
    g_fwd = _lean_reach(sp_f, _turned(su, ws, -tau), sp_f + _turned(sm - sp, ws, -tau),
                        rp + tau * vr + _turned(rm - rp, wr, -tau), lip_r, cup_r, reach_r, below_r)
    g_back = _lean_reach(sp_b, _turned(su, ws, tau), sp_b + _turned(sm - sp, ws, tau),
                         rp - tau * vr + _turned(rm - rp, wr, tau), lip_r, cup_r, reach_r, below_r)
    return (g_fwd - g_back) / (2.0 * tau)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dt = ctx.src_palm_pos.dtype

    # =============================== weights (iter_27 unless marked iter_28) =================
    APPROACH_W, APPROACH_K = 0.6, 4.0      # reach income, small: it must not compete with pouring
    GRASP_W, CLOSE_W, CLOSE_NORM = 0.8, 0.8, 0.35
    LIFT_SRC_W, LIFT_RCV_W = 3.0, 1.5      # iter_24 levels: these lifted reliably, do not weaken
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03   # carry heights: workspace, not cup sizes
    POSTURE_W_SRC, POSTURE_W_RCV = 3.0, 1.2    # before the lift: build the envelope (all fingers)
    KEEP_W_SRC, KEEP_W_RCV = 1.5, 1.5          # after the lift: keep THAT grasp (iter_24 design)
    KEEP_POS, KEEP_MAG = 0.02, 0.7             # [m] drift and |log force ratio| at which sim = 0.37
    FINGER_ON_SHARE = 0.02     # "touched at lift" = force share of the lift hold above this
    KEEP_FORCE_FRAC = 0.5      # a lift finger still counts if it presses >= half its OWN lift force
    AIM_W, AIM_K, AIM_FAR = 5.0, 10.0, 0.40    # workspace-scale guidance toward the receiver
    AIM_PREC_R = 0.5       # gaussian width of the spout offset inside the receiver mouth [cup radii]
    DROP_R = 1.5           # spout height above the receiver rim at which low_drop = 0.37
    BELOW_R = 1.0          # spout below the receiver rim
    DROP_FLOOR = 0.4       # a high spout keeps 40 % of the aim income (collision margin)
    # iter_28 change 1: width of the reach factor inside Phi [cup radii] = the env's premature latch
    # radius (8.1 cm = 2.0 r), in place of the 0.5 r precision width. Aim keeps AIM_PREC_R.
    REACH_R = 2.0
    # rotation help = TILT_W * rate of Phi [1/s] per step; pay gated by rate-free factors and capped
    # at +TILT_PAY_CAP per step, charge in full (unchanged from iter_27)
    TILT_W, TILT_PAY_CAP = 5.0, 1.0
    TILT_TAU = 1.0 / 120.0     # half-width of the central difference [s], numerical step only
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 60.0, 0.03, -0.05   # paid per unit of bead fraction that ARRIVES
    OVER_DELTA_W = 120.0       # 2x FLOW_W: a bead past the request costs more than it ever paid
    OVER_LEVEL_W = 4.0         # sustained charge for sitting on an overshoot
    SPILL_W = 50.0             # per REQUEST-worth of spilled beads, floored at OVER_DELTA_W
    FINISH_W = 10.0            # sustained: request closed + cup upright at the pre-tilt pose + held
    FIN_UP_FLOOR, FIN_POSE_FLOOR = 0.5, 0.4
    POSE_TOL_R, POSE_SIG_R = 2.0, 2.0   # plateau beyond ctx.cup_mouth_z / gaussian falloff [cup radii]
    SUCCESS_W = 12.0           # env verdict on top of it
    HOME_W = 1.0
    CENTER_W, CENTER_BAND, CENTER_SCALE = 1.5, 0.05, 0.10   # source side centre-line term
    SIDE_VOID, SIDE_OK = 0.03, 0.08     # env voids success below 3 cm receiver side margin
    DIR_ZERO = 0.25                     # env voids success at pour_dir_xy[:,0] > 0.3
    PRE_W, PRE_LO, LATCH_W = 2.0, 0.7, 1.0     # premature tilt: FRACTION of ctx.premature_tilt_limit
    CONTACT_W, HIT_W, HIT_LATCH_W = 2.0, 5.0, 0.3
    FOREIGN_DEAD, FOREIGN_W_FREE, FOREIGN_W_HELD = 1.0, 0.3, 1.5
    RATE_W_BASE, RATE_W_FULL, FLIP_W = 0.005, 0.03, 0.4
    RCV_CONTROL_MULT = 2.0     # receiver arm trembled more -> charge it harder
    TREMOR_SRC_W, TREMOR_RCV_W = 0.3, 0.9      # on the measured arm joint speeds
    TREMOR_FREE, TREMOR_SCALE = 0.20, 0.50     # [rad/s]
    ADR_RAMP_FULL = 0.2        # control charges reach full weight at adr_progress 0.2
    SETTLE_FRAC = 0.05

    # =============================== basic state (unchanged) =================================
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    off_src = _step((h_src - 0.01) / 0.04)          # lift-off ramp (reward shape, not an asset size)
    off_rcv = _step((h_rcv - 0.01) / 0.04)
    held_src = torch.maximum(g_src, off_src)        # a flickering contact flag must not kill income
    held_rcv = torch.maximum(g_rcv, off_rcv)
    carry_src = held_src * off_src
    carry_rcv = held_rcv * off_rcv
    carry_both = carry_src * carry_rcv
    not_nested = (~ctx.cups_nested).to(dt)
    settled = _step(ctx.episode_progress.to(dt) / SETTLE_FRAC)
    up_src = torch.clamp(ctx.src_cup_up[:, 2], 0.0, 1.0)             # 1 = upright, 0 = horizontal
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], 0.0, 1.0)
    cup_r = max(float(ctx.cup_radius), 1e-3)                         # geometry unit from ctx
    geo = (AIM_PREC_R, BELOW_R, DROP_R, DROP_FLOOR)

    # grasp quality / grasp maintenance
    q_src = _grasp_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_wrap_count, dt)
    q_rcv = _grasp_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_wrap_count, dt)
    keep_src = _grip_keep(ctx.src_palm_in_cup, ctx.src_tips_in_cup, ctx.src_ref_palm_in_cup,
                          ctx.src_ref_tips_in_cup, ctx.src_force_share, ctx.src_ref_force_share,
                          ctx.src_force_total, ctx.src_ref_force_total, ctx.src_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    keep_rcv = _grip_keep(ctx.rcv_palm_in_cup, ctx.rcv_tips_in_cup, ctx.rcv_ref_palm_in_cup,
                          ctx.rcv_ref_tips_in_cup, ctx.rcv_force_share, ctx.rcv_ref_force_share,
                          ctx.rcv_force_total, ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    # iter_28 change 3: fingers-kept out of ALL F fingers on both hands (inside _fingers_kept)
    kept_src = _fingers_kept(ctx.src_ref_force_share.to(dt), ctx.src_ref_force_total.to(dt),
                             ctx.src_finger_force.to(dt), ctx.src_grasp_ref_valid,
                             FINGER_ON_SHARE, KEEP_FORCE_FRAC)
    kept_rcv = _fingers_kept(ctx.rcv_ref_force_share.to(dt), ctx.rcv_ref_force_total.to(dt),
                             ctx.rcv_finger_force.to(dt), ctx.rcv_grasp_ref_valid,
                             FINGER_ON_SHARE, KEEP_FORCE_FRAC)
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    # iter_28 change 2: a cup carried WITHOUT a lift reference counts as keep = 0 (1 at the table),
    # so dodging the env's lift record earns no grip income. _grip_keep itself is untouched.
    keep_eff_src = v_src_ref * keep_src + (1.0 - v_src_ref) * (1.0 - off_src)
    keep_eff_rcv = v_rcv_ref * keep_rcv + (1.0 - v_rcv_ref) * (1.0 - off_rcv)
    grip_gate = 0.3 + 0.7 * keep_eff_src    # a ruined / dodged grasp earns 0.3 of the pour income

    # =============================== requested amount (unchanged) ============================
    poured = torch.clamp(ctx.bead_in_target_frac.to(dt), 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac.to(dt), 0.0, 1.0)
    in_src = torch.clamp(ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    tol = 1.0 / torch.clamp(ctx.bead_active_count.to(dt), min=1.0)   # env tolerance = ONE bead
    d_in = ctx.d_in_target.to(dt)
    poured_prev = torch.clamp(poured - d_in, min=0.0)
    deficit_prev = torch.clamp(target - poured_prev, min=0.0)        # what was still requested
    err = torch.abs(poured - target)
    on_target = 1.0 - _step((err - tol) / (4.0 * tol))                # 1 within one bead, 0 at five
    need = torch.clamp(torch.minimum(target - poured, in_src), min=0.0)   # spilled beads are gone
    want = _step((need - tol) / tol)          # more than one bead still requested AND available
    closed = 1.0 - want
    spilled = torch.clamp(ctx.bead_spill_frac.to(dt), 0.0, 1.0)
    released = _step((poured + spilled) / tol)   # the first bead has LEFT the source (cumulative)

    # =============================== stage 1-3: reach, grasp, lift ===========================
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = APPROACH_W * torch.exp(-APPROACH_K * d_src)
    approach_rcv = APPROACH_W * torch.exp(-APPROACH_K * d_rcv)
    grasp_src = GRASP_W * held_src + CLOSE_W * torch.exp(-6.0 * d_src) \
        * torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_rcv = GRASP_W * held_rcv + CLOSE_W * torch.exp(-6.0 * d_rcv) \
        * torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    lift_shape_src = 0.5 * (1.0 - torch.exp(-torch.clamp(h_src, min=0.0) / LIFTOFF_H)) \
        + 0.5 * torch.clamp(h_src / LIFT_SRC_H, 0.0, 1.0)
    lift_src = LIFT_SRC_W * held_src * lift_shape_src
    lift_rcv = LIFT_RCV_W * held_rcv * torch.clamp(h_rcv / LIFT_RCV_H, 0.0, 1.0) * up_rcv
    # posture: pre-lift envelope branch only while the cup is still at the table (change 2: x(1-off));
    # after the reference: KEEP the lift grasp (iter_24) x share of ALL fingers still pressing (change 3)
    posture_src = held_src * ((1.0 - v_src_ref) * (1.0 - off_src) * POSTURE_W_SRC * q_src
                              + v_src_ref * KEEP_W_SRC * keep_eff_src * kept_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * (1.0 - off_rcv) * POSTURE_W_RCV * q_rcv
                              + v_rcv_ref * KEEP_W_RCV * keep_eff_rcv * kept_rcv)

    # =============================== stage 4: aim the LIP over the receiver (unchanged) ======
    lip_xy, fine, height = _spout(ctx.src_pour_lip_pos.to(dt), ctx.rcv_cup_mouth_pos.to(dt),
                                  cup_r, geo)
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)     # x > 0.3 voids success
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    aim_xy = 0.2 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) \
        + 0.3 * torch.exp(-AIM_K * lip_xy) + 0.5 * fine
    aim_geo = aim_xy * height
    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_geo * want                            # fades once the request closes

    # =============================== stage 4b: rotation potential (iter_28 change 1) =========
    # Phi = latches * reach * over_rim * lean. Only the geometry inside Phi changed vs iter_27:
    # reach = exp(-(spout offset / (REACH_R r))^2) (latch radius instead of the 0.5 r precision),
    # over_rim = the below-rim factor, NO drop factor (raising above the rim is not charged here).
    # lean = max(up . d, 0). latches = settled * (1 - released) * want, as in iter_27. The rate is
    # the FULL rate (both cups moved ahead and back by TILT_TAU, mouth, d, theta, lip rebuilt).
    # Holding any pose earns 0; a tilt made with the spout far away pays and costs ~0; a lean away
    # or sideways has lean = 0. Pay gate (aim_gate) is rate-free and outside Phi: it only shrinks
    # the pay, never the charge; pay capped at TILT_PAY_CAP, negative rate charged in full.
    lip_r = torch.norm(ctx.src_pour_lip_pos - ctx.src_cup_mouth_pos, dim=-1).to(dt)   # r from ctx
    latches = settled * (1.0 - released) * want
    rate = latches * _lean_reach_rate(ctx, lip_r, cup_r, REACH_R, BELOW_R, TILT_TAU, dt)
    tilt_rate = torch.where(rate > 0.0,
                            torch.clamp(TILT_W * aim_gate * rate, max=TILT_PAY_CAP),
                            TILT_W * rate)

    # =============================== stage 5: the flow IS the pour (unchanged) ===============
    useful = torch.clamp(torch.minimum(d_in, deficit_prev), FLOW_FLOOR, FLOW_CAP)
    pour_flow = FLOW_W * useful
    overshoot = -OVER_DELTA_W * torch.clamp(d_in - deficit_prev, min=0.0) \
        - OVER_LEVEL_W * torch.clamp(poured - target, min=0.0)
    spill_w = torch.clamp(SPILL_W / torch.maximum(target, tol), min=OVER_DELTA_W)
    spill = -spill_w * settled * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # =============================== stage 6: finish upright at the pre-tilt pose =============
    body_xy = torch.norm(ctx.src_cup_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    pose_tol = max(float(ctx.cup_mouth_z), 0.0) + POSE_TOL_R * cup_r
    pre_tilt_pose = not_nested * torch.exp(
        -(torch.clamp(body_xy - pose_tol, min=0.0) / (POSE_SIG_R * cup_r)) ** 2)
    # iter_28 change 2: the grip factor uses keep_eff_src (a dodged lift reference counts as keep 0)
    finish = FINISH_W * closed * (0.3 + 0.7 * on_target) \
        * (FIN_UP_FLOOR + (1.0 - FIN_UP_FLOOR) * up_src) \
        * (FIN_POSE_FLOOR + (1.0 - FIN_POSE_FLOOR) * pre_tilt_pose) \
        * carry_both * up_rcv * (0.4 + 0.6 * keep_eff_src)
    success = SUCCESS_W * ctx.success.to(dt)

    # =============================== workspace constraints (unchanged) =======================
    center_src = -CENTER_W * held_src * torch.tanh(
        torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)
    # iter_27 home anchor = receiver spawn x at side margin SIDE_OK
    dx_home = ctx.rcv_cup_pos[:, 0] - ctx.rcv_cup_spawn_pos[:, 0]
    dm_home = ctx.rcv_side_margin.to(dt) - SIDE_OK
    d_home = torch.sqrt(dx_home ** 2 + dm_home ** 2 + 1e-12)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05,
                                             min=0.0) / 0.15)
    home_rcv = HOME_W * carry_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv
    limit = torch.clamp(ctx.premature_tilt_limit.to(dt), min=1e-3)
    risk = _step((ctx.src_cup_tilt.to(dt) / limit - PRE_LO) / (1.0 - PRE_LO))
    premature = -PRE_W * held_src * (1.0 - torch.exp(-AIM_K * lip_xy)) * risk \
        - LATCH_W * ctx.premature_tilt.to(dt)

    # =============================== safety (real robot, unchanged) ==========================
    f_cup = ctx.cup_cup_force
    cup_collision = -CONTACT_W * torch.tanh(f_cup / 2.0) \
        - HIT_W * settled * (f_cup > 0.5).to(dt) - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_s = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    f_r = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    hand_foreign = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) \
        * torch.tanh(f_s / 5.0) \
        - (FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) * torch.tanh(f_r / 5.0)

    # =============================== control (ramped on adr_progress, unchanged) =============
    ramp = _step(torch.full_like(up_src, float(ctx.adr_progress) / ADR_RAMP_FULL))
    rate_w = RATE_W_BASE + (RATE_W_FULL - RATE_W_BASE) * ramp
    da = ctx.actions - ctx.prev_actions
    flips = torch.relu(-(ctx.actions * ctx.prev_actions))
    action_rate = -rate_w * (torch.sum(da[:, 0:9] ** 2, dim=-1)
                             + RCV_CONTROL_MULT * torch.sum(da[:, 9:18] ** 2, dim=-1)) \
        - FLIP_W * ramp * (torch.sum(flips[:, 0:9], dim=-1)
                           + RCV_CONTROL_MULT * torch.sum(flips[:, 9:18], dim=-1)) \
        * torch.maximum(held_src, held_rcv)
    qd_s = torch.sqrt(torch.mean(ctx.src_arm_qd ** 2, dim=-1) + 1e-8)
    qd_r = torch.sqrt(torch.mean(ctx.rcv_arm_qd ** 2, dim=-1) + 1e-8)
    tremor = -ramp * (TREMOR_SRC_W * torch.tanh(torch.clamp(qd_s - TREMOR_FREE, min=0.0) / TREMOR_SCALE)
                      + TREMOR_RCV_W * torch.tanh(torch.clamp(qd_r - TREMOR_FREE, min=0.0) / TREMOR_SCALE))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "posture_src": posture_src,
        "posture_rcv": posture_rcv,
        "aim": aim,
        "tilt_rate": tilt_rate,
        "pour_flow": pour_flow,
        "overshoot": overshoot,
        "spill": spill,
        "finish": finish,
        "success": success,
        "center_src": center_src,
        "home_rcv": home_rcv,
        "premature": premature,
        "cup_collision": cup_collision,
        "hand_foreign": hand_foreign,
        "action_rate": action_rate,
        "tremor": tremor,
    }

    reward = torch.zeros_like(approach_src)
    for v in components.values():
        reward = reward + v

    return reward, components
```

근거 메모 (규칙 충돌 없음, 변경은 prompt 가 지정한 세 가지뿐)

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| tilt_rate (Phi 안 기하만 교체: reach 폭 REACH_R=2.0 r, over_rim 유지, drop 인자 제외. latches, 중심차분, pay gate, cap, 전액 charge 는 그대로) | 정체 지점에서 Phi 가 1e-7 이하라 회전 보조가 0. 스파우트가 입구에서 2.1-3.3 r 떨어져 있음 | prompt.md:633-635, 673-688 |
| posture_src/rcv 의 들기 전 가지 x(1-off) | 소스가 3 cm 들기 전에 20도 넘게 기울여 기준 기록을 피하고, 운반 내내 들기 전 posture(최대 3.0)를 받음 | prompt.md:640-649, 694-698 |
| keep_eff (grip_gate, finish, posture keep 가지) | 기준 없이 운반하면 keep=0, grip_gate 0.3. _grip_keep 는 바이트 단위로 그대로 | prompt.md:696-702 |
| _fingers_kept 분모를 F 로, 들 때 안 닿은 손가락은 미유지 | 리시버가 4손가락으로 들어도 kept_rcv 0.96, 5손가락 운반은 0.13 | prompt.md:662-670, 703-709 |
| aim, premature, flow/overshoot/spill, finish(keep 외), success, center, home, 안전·제어 항 | KEEP 목록에 따라 변경 없음 | prompt.md:711-725 |
| 예산 | 변경 1 은 상태 수입을 늘리지 않음(유지 시 0), 변경 2·3 은 수입을 줄이기만 함 | prompt.md:726-728 |

검증 필요: 닫힌 사이클의 합이 0 이하인지, 먼 스파우트에서의 기울임이 ~0 인지, 옆·반대 기울임과 유지가 0 인지를 결정론 검사(check_tilt_rate.py 방식)로 확인해야 한다(prompt.md:689-693). 이 세 성질은 아직 검증되지 않았다(가설). 명령을 실행하지 않았기 때문이다.

파일 쓰기: 에이전트 계약상 파일을 쓰지 않고, Write 도구도 없다. 그래서 /home/user/rl_ws/hdgp/reward_gen/pour_bi/iter_28/response_c1.md 는 쓰지 않았다. 호출자가 경로를 검사한 뒤 이 본문을 그 파일에 쓴다.