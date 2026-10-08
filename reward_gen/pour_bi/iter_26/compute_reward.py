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
    """[0,1] quality of one hand on its own cup. Only the SHAPE of the contact enters here
    (wrapped fingers, palm contact, thumb contact) - no cup dimension, so it survives a cup swap."""
    wrap_q = torch.clamp(wrap_count.to(dt) / 3.0, 0.0, 1.0)          # >= 3 wrapped fingers = envelope
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)      # [N] force sensor scale
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    return 0.5 * wrap_q + 0.25 * palm_q + 0.25 * thumb_q


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """iter_24, KEPT UNCHANGED (operator requirement 4): how closely the hand still holds the cup
    the way it did at the instant it lifted it - hand pose in CUP coordinates, force distribution
    and force level RELATIVE to that instant. Never an absolute force target. Before the reference
    exists the factor is 1 (nothing to keep yet)."""
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = torch.clamp(0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim, 0.0, 1.0)
    return torch.where(valid, keep, torch.ones_like(keep))


def _fingers_kept(ref_share, finger_force, valid, on_share, touch_n):
    """iter_26 change 5: share of the fingers that touched the cup AT LIFT that still touch it now.
    Purely relative to the lift hold; force enters only through a contact floor (a robot constant),
    never as a target. 1 before the reference exists or if no finger touched at lift."""
    ref_on = (ref_share[:, 1:] > on_share).to(finger_force.dtype)   # share = [palm, thumb..pinky]
    touch = _step(torch.clamp(finger_force, min=0.0) / touch_n)
    n_ref = torch.sum(ref_on, dim=-1)
    kept = torch.sum(ref_on * touch, dim=-1) / torch.clamp(n_ref, min=1.0)
    kept = torch.where(n_ref > 0.5, kept, torch.ones_like(kept))
    return torch.where(valid, kept, torch.ones_like(kept))


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dt = ctx.src_palm_pos.dtype

    # =============================== weights ==================================================
    APPROACH_W, APPROACH_K = 0.6, 4.0      # reach income, small: it must not compete with pouring
    GRASP_W, CLOSE_W, CLOSE_NORM = 0.8, 0.8, 0.35
    LIFT_SRC_W, LIFT_RCV_W = 3.0, 1.5      # iter_24 levels: these lifted reliably, do not weaken
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03   # carry heights: workspace, not cup sizes
    POSTURE_W_SRC, POSTURE_W_RCV = 3.0, 1.2    # before the lift: build an enveloping grasp
    KEEP_W_SRC = 1.5                           # after the lift: keep THAT grasp (iter_24 design)
    # change 5: receiver keep 0.8 -> 1.5 (same as the source) and multiplied by the share of the
    # lift-time fingers still touching: losing one of five fingers costs 1.5 * 0.2 = 0.3 per step
    KEEP_W_RCV = 1.5
    KEEP_POS, KEEP_MAG = 0.02, 0.7             # [m] drift and |log force ratio| at which sim = 0.37
    FINGER_ON_SHARE, FINGER_TOUCH_N = 0.02, 0.2    # "touched at lift" force share / contact floor [N]
    AIM_W, AIM_K, AIM_FAR = 5.0, 10.0, 0.40    # workspace-scale guidance toward the receiver
    # change 4: spout precision and drop height, all in multiples of ctx.cup_radius
    AIM_PREC_R = 0.5       # gaussian width of the spout offset inside the receiver mouth
    DROP_R = 1.5           # spout height above the receiver rim at which low_drop = 0.37
    BELOW_R = 1.0          # spout below the receiver rim
    DROP_FLOOR = 0.4       # a high spout keeps 40 % of the aim income (collision margin)
    # change 1: rotation help as a potential RATE, only until the first bead has left the cup
    # c1r note: summed per step, not multiplied by dt: one rotation totals about TILT_W * dPhi / dt
    TILT_W, TILT_PAY_CAP = 5.0, 1.0            # pay capped at +1 per step, charge uncapped
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 60.0, 0.03, -0.05   # paid per unit of bead fraction that ARRIVES
    OVER_DELTA_W = 120.0       # 2x FLOW_W: a bead past the request costs more than it ever paid
    OVER_LEVEL_W = 4.0         # sustained charge for sitting on an overshoot
    # change 4: spill charged per REQUEST-worth of beads (spilling as many beads as were requested
    # costs SPILL_W). target <= 1, so a spilled bead is never cheaper than in iter_25.
    # c1r M2: the per-fraction weight is floored at OVER_DELTA_W (2x FLOW_W), see the spill term
    SPILL_W = 50.0
    FINISH_W = 10.0            # sustained: request closed + cup upright at the pre-tilt pose + held
    # c1r H1: the pose factor is measured on the source cup BODY (tilt-independent), so the finish
    # rises monotonically as the cup comes upright at any pose and peaks upright at the pre-tilt pose.
    # tilted handover: FINISH_W * FIN_UP_FLOOR * (FIN_POSE_FLOOR + 0.6 * pose) >= AIM_W * geo, since
    # pose = 1 at the pour pose (inside the plateau) and 10 * 0.5 * 0.4 = 2.0 >= AIM_W * geo if geo <= 0.4
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

    # =============================== basic state ==============================================
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
    settled = _step(ctx.episode_progress / SETTLE_FRAC)

    # cup attitude as a dimensionless measure - NO angle constant anywhere
    up_src = torch.clamp(ctx.src_cup_up[:, 2], 0.0, 1.0)             # 1 = upright, 0 = horizontal
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], 0.0, 1.0)
    cup_r = max(float(ctx.cup_radius), 1e-3)                         # change 4: geometry unit from ctx

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
    kept_rcv = _fingers_kept(ctx.rcv_ref_force_share, ctx.rcv_finger_force,
                             ctx.rcv_grasp_ref_valid, FINGER_ON_SHARE, FINGER_TOUCH_N)
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    grip_gate = 0.3 + 0.7 * keep_src        # a ruined grasp earns a fraction of the pour income

    # =============================== requested amount =========================================
    # everything below is expressed in FRACTIONS taken from ctx, so it follows any bead count
    poured = torch.clamp(ctx.bead_in_target_frac.to(dt), 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac.to(dt), 0.0, 1.0)
    in_src = torch.clamp(ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    tol = 1.0 / torch.clamp(ctx.bead_active_count.to(dt), min=1.0)   # env tolerance = ONE bead
    d_in = ctx.d_in_target.to(dt)
    poured_prev = torch.clamp(poured - d_in, min=0.0)
    deficit_prev = torch.clamp(target - poured_prev, min=0.0)        # what was still requested
    err = torch.abs(poured - target)
    on_target = 1.0 - _step((err - tol) / (4.0 * tol))                # 1 within one bead, 0 at five
    # change 2: need = min(requested - poured, beads still in the source) - spilled beads are gone
    need = torch.clamp(torch.minimum(target - poured, in_src), min=0.0)
    # changes 2 + 3: want is on while MORE than one bead is still requested AND still available.
    # It turns off inside the env's one-bead tolerance or when the source can no longer supply the
    # request; exactly then the finishing income opens (closed = 1 - want).
    want = _step((need - tol) / tol)
    closed = 1.0 - want
    # change 1 (c1r M1): has the first bead LEFT the source cup? Built from cumulative bead events
    # (arrived + spilled), so a bead that falls back into the source cannot re-arm the tilt help
    spilled = torch.clamp(ctx.bead_spill_frac.to(dt), 0.0, 1.0)
    released = _step((poured + spilled) / tol)

    # =============================== stage 1-3: reach, grasp, lift ============================
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
    # posture: build the envelope before the lift, KEEP the lift grasp afterwards (iter_24).
    # change 5: the receiver keep also requires every finger that touched at lift to still touch,
    # during the whole carry and pour (held_rcv), relative to the lift hold only.
    posture_src = held_src * ((1.0 - v_src_ref) * POSTURE_W_SRC * q_src
                              + v_src_ref * KEEP_W_SRC * keep_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * POSTURE_W_RCV * q_rcv
                              + v_rcv_ref * KEEP_W_RCV * keep_rcv * kept_rcv)

    # =============================== stage 4: aim the LIP over the receiver ===================
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)     # x > 0.3 voids success
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    # change 4: spill grew with the spout offset INSIDE the mouth and with the drop height, so the
    # aim now has a precision part and a height part, both in cup radii (no angle)
    fine = torch.exp(-(lip_xy / (AIM_PREC_R * cup_r)) ** 2)            # spout centred in the mouth
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (BELOW_R * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (DROP_R * cup_r)) ** 2)
    height = over_rim * (DROP_FLOOR + (1.0 - DROP_FLOOR) * low_drop)
    aim_xy = 0.2 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) \
        + 0.3 * torch.exp(-AIM_K * lip_xy) + 0.5 * fine
    aim_geo = aim_xy * height                 # spout geometry the aim pays for (c1r: not the finish)
    spout = fine * height                     # precise, low spout: gate for the rotation help
    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_geo * want                            # fades once the request closes

    # =============================== stage 4b: rotation help as a RATE (change 1) =============
    # potential phi = (1 - up_z) / 2 (0 upright, 1 inverted; the cup's own axis, no angle constant).
    # exact rate of the same expression: d(up)/dt = omega x up
    #   -> d(phi)/dt = 0.5 * (w_y * up_x - w_x * up_y)   [1/s]
    # (hypothesis: src_cup_ang_vel is in the same world/env frame as src_cup_up)
    # Holding any tilt earns 0. Pay is capped at +1 per step and its gate is <= the charge gate;
    # the charge is uncapped, so a closed tilt cycle before the first bead sums to <= 0 (step
    # sampling aside). The term switches off for good once the first bead has LEFT the source cup:
    # from then on only arriving beads pay (pour_flow) and turning the cup back costs nothing here.
    u = ctx.src_cup_up.to(dt)
    w = ctx.src_cup_ang_vel.to(dt)
    dphi = 0.5 * (w[:, 1] * u[:, 0] - w[:, 0] * u[:, 1])
    g_charge = settled * (1.0 - released) * want        # constant while no bead has left
    g_pay = g_charge * aim_gate * spout                  # <= g_charge everywhere
    tilt_rate = torch.where(dphi > 0.0,
                            torch.clamp(TILT_W * g_pay * dphi, max=TILT_PAY_CAP),
                            TILT_W * g_charge * dphi)

    # =============================== stage 5: the flow IS the pour ============================
    # income for every bead fraction that ARRIVES while it is still requested, capped per step so
    # that dumping the cup at once pays less than a controlled trickle
    useful = torch.clamp(torch.minimum(d_in, deficit_prev), FLOW_FLOOR, FLOW_CAP)
    pour_flow = FLOW_W * useful
    # charge for the part of this step's arrival that went PAST the request, plus a sustained charge
    # for staying overshot: pouring everything is strictly worse than stopping on time
    overshoot = -OVER_DELTA_W * torch.clamp(d_in - deficit_prev, min=0.0) \
        - OVER_LEVEL_W * torch.clamp(poured - target, min=0.0)
    # change 4: spilled beads measured in units of the REQUEST (the actor cannot observe spill);
    # c1r M2: floored at OVER_DELTA_W so one spilled bead always costs more than one arriving pays
    spill_w = torch.clamp(SPILL_W / torch.maximum(target, tol), min=OVER_DELTA_W)
    spill = -spill_w * settled * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # =============================== stage 6: finish upright at the pre-tilt pose =============
    # change 3: opens at the env tolerance (|error| <= 1 bead) or when the request became
    # unreachable (closed = 1 - want). Pays for standing the source cup upright AND returning to
    # the pre-tilt pose, both cups still held.
    # c1r H1: the reward has no memory and the lip moves with the tilt, so the pre-tilt pose is
    # measured on the source cup BODY: horizontal distance from ctx.src_cup_pos to the receiver
    # mouth. Full credit up to cup_mouth_z + POSE_TOL_R cup radii (covers the upright side-by-side
    # pose and the body position while pouring), gaussian falloff beyond, none while nested (the
    # aim never paid a nested pose). This factor does not depend on the tilt, so the up factor makes
    # the finish rise monotonically as the cup comes upright; the maximum is upright at that pose.
    body_xy = torch.norm(ctx.src_cup_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    pose_tol = max(float(ctx.cup_mouth_z), 0.0) + POSE_TOL_R * cup_r
    pre_tilt_pose = not_nested * torch.exp(
        -(torch.clamp(body_xy - pose_tol, min=0.0) / (POSE_SIG_R * cup_r)) ** 2)
    finish = FINISH_W * closed * (0.3 + 0.7 * on_target) \
        * (FIN_UP_FLOOR + (1.0 - FIN_UP_FLOOR) * up_src) \
        * (FIN_POSE_FLOOR + (1.0 - FIN_POSE_FLOOR) * pre_tilt_pose) \
        * carry_both * up_rcv * (0.4 + 0.6 * keep_src)
    success = SUCCESS_W * ctx.success.to(dt)

    # =============================== workspace constraints ====================================
    # the source cup must not do its work deep on the receiver's side; with side_ok on the
    # receiver this makes the two cups meet near the centre line
    center_src = -CENTER_W * held_src * torch.tanh(
        torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05,
                                             min=0.0) / 0.15)
    home_rcv = HOME_W * carry_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv
    # premature tilt: a FRACTION of the env's own limit, charged only while the lip is not yet over
    # the receiver, plus the episode latch itself (unchanged)
    limit = torch.clamp(ctx.premature_tilt_limit.to(dt), min=1e-3)
    risk = _step((ctx.src_cup_tilt.to(dt) / limit - PRE_LO) / (1.0 - PRE_LO))
    premature = -PRE_W * held_src * (1.0 - torch.exp(-AIM_K * lip_xy)) * risk \
        - LATCH_W * ctx.premature_tilt.to(dt)

    # =============================== safety (real robot) ======================================
    f_cup = ctx.cup_cup_force
    cup_collision = -CONTACT_W * torch.tanh(f_cup / 2.0) \
        - HIT_W * settled * (f_cup > 0.5).to(dt) - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_s = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    f_r = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    hand_foreign = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) \
        * torch.tanh(f_s / 5.0) \
        - (FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) * torch.tanh(f_r / 5.0)

    # =============================== control (ramped on adr_progress, unchanged) ==============
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
