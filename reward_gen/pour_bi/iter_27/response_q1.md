

### Task Analysis & Stages

The task requires the bimanual robot to execute a coordinated pour:
1. **Approach & Grasp:** Both arms move to their respective cups. The right (source) hand grasps the bead-filled cup, and the left (receiver) hand grasps the empty cup. The policy must learn an enveloping grasp (all five fingers) to prevent collapse during tilt.
2. **Lift & Carry:** Both cups are lifted off the table to a safe height. The receiver cup must stay upright, and the source cup must maintain its grasp while being transported.
3. **Aim & Meet:** The source arm moves its cup over the receiver. Crucially, the receiver arm must also move inward to meet the source near the workspace center line, ensuring the pour lip is accurately positioned over the receiver mouth without crossing safety boundaries.
4. **Tilt & Pour:** The source cup is tilted forward. The reward must only pay for the *rate* of lean when the spout is correctly positioned over the receiver. Beads flow out, and the policy is rewarded for each bead fraction arriving in the receiver while the target is still unmet.
5. **Finish & Reset:** Once the requested bead count is transferred, the source cup is returned upright to its pre-tilt pose. The policy is rewarded for holding the final pose, keeping both cups grasped, and minimizing spill/overshoot.

### Improved Reward Function (Iter-27)

```python
import torch
import math


def _step(x: torch.Tensor) -> torch.Tensor:
    """Smoothstep on a normalised argument -> [0,1]"""
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


def _grasp_quality(finger_force: torch.Tensor, palm_force: torch.Tensor,
                   wrap_count: torch.Tensor, dt: torch.dtype, F: int) -> torch.Tensor:
    """[0,1] grasp quality. Uses F fingers so it scales correctly if asset changes."""
    wrap_q = torch.clamp(wrap_count.to(dt) / F, 0.0, 1.0)
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    return 0.5 * wrap_q + 0.25 * palm_q + 0.25 * thumb_q


def _fingers_kept(ref_share: torch.Tensor, finger_force: torch.Tensor,
                  valid: torch.Tensor, ref_total: torch.Tensor, touch_frac: float) -> torch.Tensor:
    """Fingers that touched at lift relative to lift force. No absolute force target."""
    # Threshold = fraction * (lift share * lift total force)
    touch_thresh = touch_frac * ref_total.unsqueeze(1) * ref_share[:, 1:]
    touch = _step(torch.clamp(finger_force, min=0.0) / torch.clamp(touch_thresh, min=1e-6))
    ref_on = (ref_share[:, 1:] > 0.0).to(finger_force.dtype)
    n_ref = torch.sum(ref_on, dim=-1)
    kept = torch.sum(ref_on * touch, dim=-1) / torch.clamp(n_ref, min=1.0)
    kept = torch.where(n_ref > 0.5, kept, torch.ones_like(kept))
    return torch.where(valid, kept, torch.ones_like(kept))


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """Pose and force-magnitude maintenance relative to lift reference."""
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = torch.clamp(0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim, 0.0, 1.0)
    return torch.where(valid, keep, torch.ones_like(keep))


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dt = ctx.src_palm_pos.dtype
    device = ctx.src_palm_pos.device
    tau = 0.02  # simulation timestep for potential rate differentiation
    F = 5       # number of fingers

    # =============================== weights ==================================================
    APPROACH_W, APPROACH_K = 0.6, 4.0
    GRASP_W, CLOSE_W, CLOSE_NORM = 0.8, 0.8, 0.35
    LIFT_SRC_W, LIFT_RCV_W = 3.0, 1.5
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03
    POSTURE_W_SRC, POSTURE_W_RCV = 3.0, 1.2
    KEEP_W_SRC, KEEP_W_RCV = 1.5, 1.5
    KEEP_POS, KEEP_MAG = 0.02, 0.7
    FINGER_ON_SHARE, FINGER_TOUCH_N = 0.02, 0.2
    AIM_W, AIM_K, AIM_FAR = 5.0, 10.0, 0.40
    AIM_PREC_R, DROP_R, BELOW_R = 0.5, 1.5, 1.0
    DROP_FLOOR = 0.4
    TILT_W, TILT_PAY_CAP = 5.0, 1.0
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 60.0, 0.03, -0.05
    OVER_DELTA_W, OVER_LEVEL_W = 120.0, 4.0
    SPILL_W = 50.0
    FINISH_W = 10.0
    FIN_UP_FLOOR, FIN_POSE_FLOOR = 0.5, 0.4
    POSE_TOL_R, POSE_SIG_R = 2.0, 2.0
    SUCCESS_W = 12.0
    HOME_W = 1.0
    CENTER_W, CENTER_BAND, CENTER_SCALE = 1.5, 0.05, 0.10
    SIDE_VOID, SIDE_OK = 0.03, 0.08
    DIR_ZERO = 0.25
    PRE_W, PRE_LO, LATCH_W = 2.0, 0.7, 1.0
    CONTACT_W, HIT_W, HIT_LATCH_W = 2.0, 5.0, 0.3
    FOREIGN_DEAD, FOREIGN_W_FREE, FOREIGN_W_HELD = 1.0, 0.3, 1.5
    RATE_W_BASE, RATE_W_FULL, FLIP_W = 0.005, 0.03, 0.4
    RCV_CONTROL_MULT = 2.0
    TREMOR_SRC_W, TREMOR_RCV_W = 0.3, 0.9
    TREMOR_FREE, TREMOR_SCALE = 0.20, 0.50
    ADR_RAMP_FULL = 0.2
    SETTLE_FRAC = 0.05

    # =============================== basic state ==============================================
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    off_src = _step((h_src - 0.01) / 0.04)
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

    # =============================== grasp & keep =============================================
    q_src = _grasp_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_wrap_count, dt, F)
    q_rcv = _grasp_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_wrap_count, dt, F)
    keep_src = _grip_keep(ctx.src_palm_in_cup, ctx.src_tips_in_cup, ctx.src_ref_palm_in_cup,
                          ctx.src_ref_tips_in_cup, ctx.src_force_share, ctx.src_ref_force_share,
                          ctx.src_force_total, ctx.src_ref_force_total, ctx.src_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    keep_rcv = _grip_keep(ctx.rcv_palm_in_cup, ctx.rcv_tips_in_cup, ctx.rcv_ref_palm_in_cup,
                          ctx.rcv_ref_tips_in_cup, ctx.rcv_force_share, ctx.rcv_ref_force_share,
                          ctx.rcv_force_total, ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    kept_src = _fingers_kept(ctx.src_ref_force_share, ctx.src_finger_force,
                             ctx.src_grasp_ref_valid, ctx.src_ref_force_total, 0.5)
    kept_rcv = _fingers_kept(ctx.rcv_ref_force_share, ctx.rcv_finger_force,
                             ctx.rcv_grasp_ref_valid, ctx.rcv_ref_force_total, 0.5)
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    grip_gate = 0.3 + 0.7 * keep_src

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
    posture_src = held_src * ((1.0 - v_src_ref) * POSTURE_W_SRC * q_src
                              + v_src_ref * KEEP_W_SRC * keep_src * kept_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * POSTURE_W_RCV * q_rcv
                              + v_rcv_ref * KEEP_W_RCV * keep_rcv * kept_rcv)

    # =============================== stage 4: aim & tilt potential ============================
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))

    fine = torch.exp(-(lip_xy / (AIM_PREC_R * cup_r)) ** 2)
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (BELOW_R * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (DROP_R * cup_r)) ** 2)
    height = over_rim * (DROP_FLOOR + (1.0 - DROP_FLOOR) * low_drop)
    fine_back = torch.exp(-(lip_xy / (AIM_PREC_R * cup_r)) ** 2)  # recompute for rollback below
    over_rim_back = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (BELOW_R * cup_r)) ** 2)
    low_drop_back = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (DROP_R * cup_r)) ** 2)
    height_back = over_rim_back * (DROP_FLOOR + (1.0 - DROP_FLOOR) * low_drop_back)

    aim_xy = 0.2 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) \
        + 0.3 * torch.exp(-AIM_K * lip_xy) + 0.5 * fine
    aim_geo = aim_xy * height
    spout = fine * height
    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_geo * want

    # Tilt potential & rate (Iter-27 change 1)
    latches = settled * (1.0 - released) * want
    # lean = horizontal projection of cup up-axis onto pour direction
    d_hat = ctx.pour_dir_xy.to(dt)
    lean = torch.clamp(ctx.src_cup_up.to(dt)[:, 0] * d_hat[:, 0] + ctx.src_cup_up.to(dt)[:, 1] * d_hat[:, 1], 0.0, 1.0)
    Phi_now = latches * spout * lean

    # Rollback state by tau to compute rate deterministically
    pos_back = ctx.src_cup_pos - ctx.src_cup_lin_vel * tau
    up_back = ctx.src_cup_up - torch.cross(ctx.src_cup_ang_vel, ctx.src_cup_up, dim=-1) * tau
    lip_back = ctx.src_pour_lip_pos - ctx.src_cup_lin_vel * tau

    d_xy_back = pos_back[:, :2] - ctx.rcv_cup_mouth_pos[:, :2]
    d_norm_back = torch.norm(d_xy_back, dim=-1, keepdim=True) + 1e-6
    d_hat_back = d_xy_back / d_norm_back
    lean_back = torch.clamp(up_back[:, 0] * d_hat_back[:, 0] + up_back[:, 1] * d_hat_back[:, 1], 0.0, 1.0)

    lip_delta_back = lip_back - ctx.rcv_cup_mouth_pos
    lip_xy_back = torch.norm(lip_delta_back[:, :2], dim=-1)
    lip_dz_back = lip_delta_back[:, 2]
    fine_b = torch.exp(-(lip_xy_back / (AIM_PREC_R * cup_r)) ** 2)
    over_rim_b = torch.exp(-(torch.clamp(-lip_dz_back, min=0.0) / (BELOW_R * cup_r)) ** 2)
    low_drop_b = torch.exp(-(torch.clamp(lip_dz_back, min=0.0) / (DROP_R * cup_r)) ** 2)
    height_b = over_rim_b * (DROP_FLOOR + (1.0 - DROP_FLOOR) * low_drop_b)
    spout_back = fine_b * height_b

    Phi_back = latches * spout_back * lean_back
    dPhi = (Phi_now - Phi_back) / tau

    pay_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    tilt_pay = torch.clamp(TILT_W * torch.clamp(dPhi, min=0.0) * pay_gate, max=TILT_PAY_CAP)
    tilt_charge = -TILT_W * torch.clamp(-dPhi, min=0.0) * latches  # uncapped, stops after first bead
    tilt_rate = tilt_pay + tilt_charge

    # =============================== stage 5: flow & spill ====================================
    useful = torch.clamp(torch.minimum(d_in, deficit_prev), FLOW_FLOOR, FLOW_CAP)
    pour_flow = FLOW_W * useful
    overshoot = -OVER_DELTA_W * torch.clamp(d_in - deficit_prev, min=0.0) \
        - OVER_LEVEL_W * torch.clamp(poured - target, min=0.0)
    spill_w = torch.clamp(SPILL_W / torch.maximum(target, tol), min=OVER_DELTA_W)
    spill = -spill_w * settled * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # =============================== stage 6: finish upright ==================================
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
    # Iter-27 change 2: anchor moves to side_margin == SIDE_OK
    anchor_rcv = torch.cat([ctx.rcv_cup_spawn_pos[:, :1], torch.full_like(ctx.rcv_cup_spawn_pos[:, 1:1], SIDE_OK), ctx.rcv_cup_spawn_pos[:, 2:3]], dim=1)
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - anchor_rcv[:, :2], dim=-1)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05, min=0.0) / 0.15)
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