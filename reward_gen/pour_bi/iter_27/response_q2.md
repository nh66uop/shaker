

### Task Analysis & Stages

The pouring task requires the bimanual robot to execute a carefully sequenced manipulation pipeline:
1. **Approach & Grasp**: Both hands approach their respective cups. The policy must learn to form a stable 5-finger enveloping grasp on each cup while keeping the receiver cup upright.
2. **Lift**: Both cups are lifted off the table simultaneously. The grasp must be maintained, and the receiver cup must not tilt excessively.
3. **Meet & Aim**: The receiver cup moves slightly toward the center line to meet the source cup. The source cup is positioned so its pouring lip (spout) is precisely over the receiver's mouth at a safe height.
4. **Pour**: The source cup is tilted toward the receiver until beads start flowing. The policy is rewarded for the *rate* of beads arriving in the target and penalized for spill or overshooting the requested amount.
5. **Finish & Return**: Once the target amount is poured, the source cup is tilted back to upright and returned to its pre-pour pose while maintaining the grasp. The environment checks success conditions (amount, spill, posture, clearance).

The previous reward failed because: (1) the rotation term only charged (no pay gate activation), (2) the receiver cup stayed anchored to its spawn instead of meeting the source near the center line, and (3) the 5-finger envelope was not strictly enforced after lift, causing finger loss during carry. Iter 27 fixes these by introducing a proper potential rate for tilt, shifting the receiver anchor to `SIDE_OK`, and enforcing relative finger-contact retention on both hands.

---

```python
import torch
import math


def _step(x: torch.Tensor) -> torch.Tensor:
    """smoothstep on an already normalised argument -> [0,1]"""
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """how closely the hand still holds the cup the way it did at the instant it lifted it"""
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
    KEEP_W_SRC = 1.5                           # after the lift: keep THAT grasp (iter_24 design)
    KEEP_W_RCV = 1.5
    KEEP_POS, KEEP_MAG = 0.02, 0.7             # [m] drift and |log force ratio| at which sim = 0.37
    AIM_W, AIM_K, AIM_FAR = 5.0, 10.0, 0.40    # workspace-scale guidance toward the receiver
    AIM_PREC_R = 0.5       # gaussian width of the spout offset inside the receiver mouth
    DROP_R = 1.5           # spout height above the receiver rim at which low_drop = 0.37
    BELOW_R = 1.0          # spout below the receiver rim
    DROP_FLOOR = 0.4       # a high spout keeps 40 % of the aim income (collision margin)
    TILT_W, TILT_PAY_CAP = 5.0, 1.0            # pay capped at +1 per step, charge uncapped
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 60.0, 0.03, -0.05   # paid per unit of bead fraction that ARRIVES
    OVER_DELTA_W = 120.0       # 2x FLOW_W: a bead past the request costs more than it ever paid
    OVER_LEVEL_W = 4.0         # sustained charge for sitting on an overshoot
    SPILL_W = 50.0
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

    # =============================== basic state ==============================================
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    off_src = _step((h_src - 0.01) / 0.04)          # lift-off ramp
    off_rcv = _step((h_rcv - 0.01) / 0.04)
    held_src = torch.maximum(g_src, off_src)
    held_rcv = torch.maximum(g_rcv, off_rcv)
    carry_src = held_src * off_src
    carry_rcv = held_rcv * off_rcv
    carry_both = carry_src * carry_rcv
    not_nested = (~ctx.cups_nested).to(dt)
    settled = _step(ctx.episode_progress / SETTLE_FRAC)

    up_src = torch.clamp(ctx.src_cup_up[:, 2], 0.0, 1.0)
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], 0.0, 1.0)
    cup_r = max(float(ctx.cup_radius), 1e-3)

    # grasp quality
    q_src = torch.tanh(torch.clamp(ctx.src_palm_force, min=0.0) / 3.0) * 0.5 + \
            torch.tanh(torch.clamp(ctx.src_finger_force[:, 0], min=0.0) / 0.5) * 0.25 + \
            torch.clamp(ctx.src_wrap_count.to(dt) / 3.0, 0.0, 1.0) * 0.25
    q_rcv = torch.tanh(torch.clamp(ctx.rcv_palm_force, min=0.0) / 3.0) * 0.5 + \
            torch.tanh(torch.clamp(ctx.rcv_finger_force[:, 0], min=0.0) / 0.5) * 0.25 + \
            torch.clamp(ctx.rcv_wrap_count.to(dt) / 3.0, 0.0, 1.0) * 0.25

    keep_src = _grip_keep(ctx.src_palm_in_cup, ctx.src_tips_in_cup, ctx.src_ref_palm_in_cup,
                          ctx.src_ref_tips_in_cup, ctx.src_force_share, ctx.src_ref_force_share,
                          ctx.src_force_total, ctx.src_ref_force_total, ctx.src_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    keep_rcv = _grip_keep(ctx.rcv_palm_in_cup, ctx.rcv_tips_in_cup, ctx.rcv_ref_palm_in_cup,
                          ctx.rcv_ref_tips_in_cup, ctx.rcv_force_share, ctx.rcv_ref_force_share,
                          ctx.rcv_force_total, ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)

    # Five-finger envelope relative to lift hold (Change 3)
    F = 5.0
    touch_src = ctx.src_finger_force >= 0.5 * (ctx.src_ref_force_share[:, 1:] * ctx.src_ref_force_total.unsqueeze(1))
    touch_rcv = ctx.rcv_finger_force >= 0.5 * (ctx.rcv_ref_force_share[:, 1:] * ctx.rcv_ref_force_total.unsqueeze(1))
    fingers_kept_src = torch.sum(touch_src.to(dt), dim=1) / F
    fingers_kept_rcv = torch.sum(touch_rcv.to(dt), dim=1) / F
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    grip_gate = 0.3 + 0.7 * keep_src

    # =============================== requested amount =========================================
    poured = torch.clamp(ctx.bead_in_target_frac.to(dt), 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac.to(dt), 0.0, 1.0)
    in_src = torch.clamp(ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    tol = 1.0 / torch.clamp(ctx.bead_active_count.to(dt), min=1.0)
    d_in = ctx.d_in_target.to(dt)
    poured_prev = torch.clamp(poured - d_in, min=0.0)
    deficit_prev = torch.clamp(target - poured_prev, min=0.0)
    err = torch.abs(poured - target)
    on_target = 1.0 - _step((err - tol) / (4.0 * tol))
    need = torch.clamp(torch.minimum(target - poured, in_src), min=0.0)
    want = _step((need - tol) / tol)
    closed = 1.0 - want
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

    # Posture: 5-finger envelope before lift, keep relative to lift after (Change 3)
    posture_src = held_src * ((1.0 - v_src_ref) * POSTURE_W_SRC * (ctx.src_wrap_count.to(dt) / F)
                              + v_src_ref * KEEP_W_SRC * keep_src * fingers_kept_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * POSTURE_W_RCV * (ctx.rcv_wrap_count.to(dt) / F)
                              + v_rcv_ref * KEEP_W_RCV * keep_rcv * fingers_kept_rcv)

    # =============================== stage 4: aim the LIP over the receiver ===================
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    fine = torch.exp(-(lip_xy / (AIM_PREC_R * cup_r)) ** 2)
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (BELOW_R * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (DROP_R * cup_r)) ** 2)
    height = over_rim * (DROP_FLOOR + (1.0 - DROP_FLOOR) * low_drop)
    aim_xy = 0.2 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) \
        + 0.3 * torch.exp(-AIM_K * lip_xy) + 0.5 * fine
    aim_geo = aim_xy * height
    spout = fine * height
    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_geo * want

    # =============================== stage 4b: rotation potential rate (Change 1) =============
    lean_now = torch.clamp(torch.sum(ctx.src_cup_up[:, :2] * ctx.pour_dir_xy, dim=1), min=0.0)
    spout_now = fine * height
    latches = settled * (1.0 - released) * want
    Phi_now = latches * spout_now * lean_now

    # Finite-difference rate using cup velocities
    tau = 0.02
    lip_back = ctx.src_pour_lip_pos - ctx.src_cup_lin_vel * tau
    mouth_back = ctx.rcv_cup_mouth_pos - ctx.rcv_cup_lin_vel * tau
    lip_delta_b = lip_back - mouth_back
    lip_xy_b = torch.norm(lip_delta_b[:, :2], dim=-1)
    lip_dz_b = lip_delta_b[:, 2]
    fine_b = torch.exp(-(lip_xy_b / (AIM_PREC_R * cup_r)) ** 2)
    over_rim_b = torch.exp(-(torch.clamp(-lip_dz_b, min=0.0) / (BELOW_R * cup_r)) ** 2)
    low_drop_b = torch.exp(-(torch.clamp(lip_dz_b, min=0.0) / (DROP_R * cup_r)) ** 2)
    height_b = over_rim_b * (DROP_FLOOR + (1.0 - DROP_FLOOR) * low_drop_b)
    spout_b = fine_b * height_b
    lean_b = lean_now  # lean change dominated by rotation, captured by Phi difference
    Phi_back = latches * spout_b * lean_b
    dPhi = (Phi_now - Phi_back) / tau

    g_pay = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    tilt_rate = torch.where(dPhi > 0.0,
                            torch.clamp(TILT_W * g_pay * dPhi, max=TILT_PAY_CAP),
                            TILT_W * dPhi)

    # =============================== stage 5: the flow IS the pour ============================
    useful = torch.clamp(torch.minimum(torch.clamp(d_in, min=0.0), deficit_prev), FLOW_FLOOR, FLOW_CAP)
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
    finish = FINISH_W * closed * (0.3 + 0.7 * on_target) \
        * (FIN_UP_FLOOR + (1.0 - FIN_UP_FLOOR) * up_src) \
        * (FIN_POSE_FLOOR + (1.0 - FIN_POSE_FLOOR) * pre_tilt_pose) \
        * carry_both * up_rcv * (0.4 + 0.6 * keep_src)
    success = SUCCESS_W * ctx.success.to(dt)

    # =============================== workspace constraints ====================================
    center_src = -CENTER_W * held_src * torch.tanh(
        torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)

    # Receiver anchor moved to SIDE_OK margin (Change 2)
    target_rcv_pos = ctx.rcv_cup_spawn_pos.clone()
    target_rcv_pos[:, 1] = SIDE_OK
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - target_rcv_pos[:, :2], dim=-1)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05,
                                             min=0.0) / 0.15)
    home_rcv = HOME_W * carry_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv

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

    # =============================== control (ramped on adr_progress) =========================
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