

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
    """[0,1] quality of one hand on its own cup. Only the SHAPE of the contact enters here
    (wrapped fingers, palm contact, thumb contact) - no cup dimension, so it survives a cup swap."""
    wrap_q = torch.clamp(wrap_count.to(dt) / 3.0, 0.0, 1.0)          # >= 3 wrapped fingers = envelope
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)      # [N] force sensor scale
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    return 0.5 * wrap_q + 0.25 * palm_q + 0.25 * thumb_q


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """iter_24, KEPT UNCHANGED: how closely the hand still holds the cup
    the way it did at the instant it lifted it - hand pose in CUP coordinates, force distribution
    and force level RELATIVE to that instant. Before the reference exists the factor is 1 (nothing to keep yet)."""
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = torch.clamp(0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim, 0.0, 1.0)
    return torch.where(valid, keep, torch.ones_like(keep))


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dt = ctx.src_palm_pos.dtype

    # =============================== weights ==================================================
    APPROACH_W, APPROACH_K = 0.6, 4.0      # reach income, small: it must not compete with pouring
    GRASP_W, CLOSE_W, CLOSE_NORM = 0.8, 0.8, 0.35
    LIFT_SRC_W, LIFT_RCV_W = 3.0, 1.5      # iter_24 levels: these lifted reliably, do not weaken
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03   # carry heights: workspace, not cup sizes
    POSTURE_W_SRC, POSTURE_W_RCV = 3.0, 1.2    # before the lift: build an enveloping grasp
    KEEP_W_SRC, KEEP_W_RCV = 1.5, 0.8          # after the lift: keep THAT grasp (iter_24 design)
    KEEP_POS, KEEP_MAG = 0.02, 0.7             # [m] drift and |log force ratio| at which sim = 0.37
    AIM_W, AIM_K, AIM_FAR, LIP_SIGMA = 5.0, 10.0, 0.40, 0.05
    TILT_W = 12.0              # rotation help only; gated by bead events per iter_26 rule 1
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 80.0, 0.04, -0.05   # paid per unit of bead fraction that ARRIVES
    OVER_DELTA_W = 150.0       # 2x FLOW_W: a bead past the request costs more than it ever paid
    OVER_LEVEL_W = 5.0         # sustained charge for sitting on an overshoot (success is void anyway)
    SPILL_W = 100.0            # iter_26 rule 4: spill must be strictly more expensive than pouring
    FINISH_W = 12.0            # sustained: request met + cup back upright + pose retention + both held
    SUCCESS_W = 15.0           # env verdict on top of it
    HOME_W, CENTER_W, CENTER_BAND, CENTER_SCALE = 1.0, 1.5, 0.05, 0.10   # source side centre-line term (req. 5)
    SIDE_VOID, SIDE_OK = 0.03, 0.08     # env voids success below 3 cm receiver side margin
    DIR_ZERO = 0.25                     # env voids success at pour_dir_xy[:,0] > 0.3
    PRE_W, PRE_LO, LATCH_W = 2.0, 0.7, 1.0     # premature tilt: FRACTION of ctx.premature_tilt_limit
    CONTACT_W, HIT_W, HIT_LATCH_W = 2.0, 5.0, 0.3
    FOREIGN_DEAD, FOREIGN_W_FREE, FOREIGN_W_HELD = 1.0, 0.3, 1.5
    RATE_W_BASE, RATE_W_FULL, FLIP_W = 0.005, 0.03, 0.4
    RCV_CONTROL_MULT = 2.0     # req. 6: the receiver arm trembled 1.5x more -> charge it harder
    TREMOR_SRC_W, TREMOR_RCV_W = 0.3, 0.9      # req. 6, on the measured arm joint speeds
    TREMOR_FREE, TREMOR_SCALE = 0.20, 0.50     # [rad/s]
    ADR_RAMP_FULL = 0.2        # req. 7: control charges reach full weight at adr_progress 0.2
    SETTLE_FRAC = 0.05
    RCV_GRIP_W = 0.6           # iter_26 rule 5: reward keeping receiver 5-finger grip while carrying

    # =============================== basic state ==============================================
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    off_src = _step((h_src - 0.01) / 0.04)          # lift-off ramp (reward shape, not an asset size)
    off_rcv = _step((h_rcv - 0.01) / 0.04)
    held_src = torch.maximum(g_src, off_src)        # a flickering contact flag must not kill income
    held_rcv = torch.maximum(g_rcv, off_rcv)
    carry_both = held_src * held_rcv * off_src * off_rcv
    not_nested = (~ctx.cups_nested).to(dt)
    settled = _step(ctx.episode_progress / SETTLE_FRAC)

    # cup attitude as a dimensionless measure - NO angle constant anywhere
    up_src = torch.clamp(ctx.src_cup_up[:, 2], 0.0, 1.0)             # 1 = upright, 0 = horizontal
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], 0.0, 1.0)
    tipped = torch.clamp((1.0 - ctx.src_cup_up[:, 2]) * 0.5, 0.0, 1.0)   # 0 upright ... 1 inverted

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
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    grip_gate = 0.3 + 0.7 * keep_src        # a ruined grasp earns a fraction of the pour income

    # =============================== requested amount & bead bookkeeping (iter_26 rules 1-2) ========
    poured = torch.clamp(ctx.bead_in_target_frac.to(dt), 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac.to(dt), 0.0, 1.0)
    tol = 1.0 / torch.clamp(ctx.bead_active_count.to(dt), min=1.0)   # env tolerance = ONE bead
    d_in = ctx.d_in_target.to(dt)
    beads_in_src = torch.clamp(ctx.bead_in_source_frac.to(dt), 0.0, 1.0)

    # need = min(requested - poured, beads still in source). If spill makes request unreachable,
    # beads_in_src -> 0, so need -> 0, and want turns off automatically (rule 2).
    need_raw = torch.clamp(target - poured, min=0.0)
    need = torch.minimum(need_raw, beads_in_src)
    err = torch.abs(poured - target)
    on_target = 1.0 - _step((err - tol) / (4.0 * tol))                # 1 within one bead, 0 at five
    want = _step(need / torch.clamp(tol, min=1e-3))                   # fades when request met or unreachable

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
    # posture: build the envelope before the lift, KEEP the lift grasp afterwards (iter_24)
    posture_src = held_src * ((1.0 - v_src_ref) * POSTURE_W_SRC * q_src
                              + v_src_ref * KEEP_W_SRC * keep_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * POSTURE_W_RCV * q_rcv
                              + v_rcv_ref * KEEP_W_RCV * keep_rcv)
    # iter_26 rule 5: reward keeping receiver 5-finger grip while carrying
    rcv_grip = RCV_GRIP_W * held_rcv * keep_rcv

    # =============================== stage 4: aim the LIP over the receiver ===================
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)     # x > 0.3 voids success
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    above = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / LIP_SIGMA) ** 2)   # lip must not be below the rim

    # iter_26 rule 4: geometry scaled by cup_radius (no hardcoded asset sizes)
    r_scale = ctx.cup_radius
    aim_xy = 0.4 * torch.clamp(1.0 - lip_xy / (2.0 * r_scale), 0.0, 1.0) \
           + 0.6 * torch.exp(-AIM_K * lip_xy / r_scale)
    above = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (2.0 * r_scale)) ** 2)

    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_xy * above * want                     # fades once the request is met

    # iter_26 rule 1: tilt income stops once the first bead has LEFT the source cup
    # beads_in_src drops from 1.0 as soon as beads leave, naturally gating tilt to 0
    tilt_gate = aim_gate * beads_in_src * torch.exp(-AIM_K * lip_xy / r_scale) * above
    tilt_aimed = TILT_W * tilt_gate * tipped

    # =============================== stage 5: the flow IS the pour ============================
    # income for every bead fraction that ARRIVES while it is still needed, capped per step
    useful = torch.clamp(torch.minimum(d_in, need), FLOW_FLOOR, FLOW_CAP)
    pour_flow = FLOW_W * useful
    # charge for the part of this step's arrival that went PAST the request, plus a sustained charge
    # for staying overshot: pouring everything is now strictly worse than stopping on time
    overshoot = -OVER_DELTA_W * torch.clamp(d_in - need, min=0.0) \
        - OVER_LEVEL_W * torch.clamp(poured - target, min=0.0)
    # iter_26 rule 4: spill made more expensive relative to request
    spill = -SPILL_W * settled * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # =============================== stage 6: finish upright & pose retention ===================
    # iter_26 rule 3: opens at env tolerance, pays for upright source + pre-tilt pose (keep_src)
    # + both cups held + receiver upright. No angle target, purely outcome-driven.
    finish = FINISH_W * on_target * up_src * carry_both * up_rcv * keep_src
    success = SUCCESS_W * ctx.success.to(dt)

    # =============================== workspace constraints ====================================
    # req. 5: the source cup must not do its work deep on the receiver's side (head-camera view).
    center_src = -CENTER_W * held_src * torch.tanh(
        torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05,
                                             min=0.0) / 0.15)
    home_rcv = HOME_W * held_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv
    # premature tilt: expressed as a FRACTION of the env's own limit, charged only while the lip is
    # not yet over the receiver, plus the episode latch itself
    limit = torch.clamp(ctx.premature_tilt_limit.to(dt), min=1e-3)
    risk = _step((ctx.src_cup_tilt.to(dt) / limit - PRE_LO) / (1.0 - PRE_LO))
    premature = -PRE_W * held_src * (1.0 - torch.exp(-AIM_K * lip_xy / r_scale)) * risk \
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

    # =============================== control (req. 6 and 7) ===================================
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
        "rcv_grip": rcv_grip,
        "aim": aim,
        "tilt_aimed": tilt_aimed,
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