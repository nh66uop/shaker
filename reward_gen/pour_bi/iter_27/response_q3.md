

```python
import torch
import math


def _step(x: torch.Tensor) -> torch.Tensor:
    """Smoothstep on a normalised argument -> [0,1]"""
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """Drift in pose and log-force-ratio relative to the lift reference. Returns [0,1]."""
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = torch.clamp(0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim, 0.0, 1.0)
    return torch.where(valid, keep, torch.ones_like(keep))


def _finger_keep_ratio(finger_force, ref_share, ref_total, valid, eps=1e-6):
    """Fraction of fingers that still touch relative to their own force at the lift reference.
    ref_share is (N, F+1) [palm, thumb..pinky]. Finger forces are (N, F)."""
    ref_finger_share = ref_share[:, 1:]
    ref_finger_force = ref_finger_share * ref_total.unsqueeze(1) + eps
    touch = _step(finger_force / ref_finger_force)
    kept = torch.mean(touch, dim=1)
    return torch.where(valid, kept, torch.ones_like(kept))


def _compute_phi(ctx, src_pos, rcv_mouth, src_up, tau=0.01):
    """Potential Phi = latches * spout * lean. Used for tilt_rate."""
    cup_r = max(float(ctx.cup_radius), 1e-3)
    
    # lean: horizontal projection of cup up-axis onto pour direction
    d_hat = torch.cat([ctx.pour_dir_xy, torch.zeros_like(ctx.pour_dir_xy[:, :1])], dim=1)
    lean = torch.clamp(torch.sum(src_up * d_hat, dim=1), min=0.0)

    # spout geometry (cup-radius scaled)
    lip_delta = src_pos - rcv_mouth
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    fine = torch.exp(-(lip_xy / (0.5 * cup_r)) ** 2)
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (1.0 * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (1.5 * cup_r)) ** 2)
    height = over_rim * (0.4 + 0.6 * low_drop)
    spout = fine * height

    # latches: active only before first bead leaves and request is still open
    poured = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac, 0.0, 1.0)
    in_src = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    tol = 1.0 / torch.clamp(ctx.bead_active_count, min=1.0)
    need = torch.clamp(torch.minimum(target - poured, in_src), min=0.0)
    want = _step((need - tol) / tol)
    spilled = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    released = _step((poured + spilled) / tol)
    settled = _step(ctx.episode_progress / 0.05)
    latches = settled * (1.0 - released) * want

    return latches * spout * lean


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dt = ctx.src_palm_pos.dtype
    cup_r = max(float(ctx.cup_radius), 1e-3)
    
    # ---- Constants & thresholds ----
    SIDE_OK, SIDE_VOID = 0.08, 0.03
    DIR_ZERO = 0.25
    SETTLE_FRAC = 0.05
    POSE_TOL_R, POSE_SIG_R = 2.0, 2.0
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03
    AIM_PREC_R, DROP_R, BELOW_R, DROP_FLOOR = 0.5, 1.5, 1.0, 0.4
    AIM_K, AIM_FAR = 10.0, 0.40
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 60.0, 0.03, -0.05
    OVER_DELTA_W, OVER_LEVEL_W = 120.0, 4.0
    SPILL_W = 50.0
    FINISH_W = 10.0
    FIN_UP_FLOOR, FIN_POSE_FLOOR = 0.5, 0.4
    SUCCESS_W = 12.0
    HOME_W = 1.0
    CENTER_W, CENTER_BAND, CENTER_SCALE = 1.5, 0.05, 0.10
    PRE_W, PRE_LO, LATCH_W = 2.0, 0.7, 1.0
    CONTACT_W, HIT_W, HIT_LATCH_W = 2.0, 5.0, 0.3
    FOREIGN_DEAD, FOREIGN_W_FREE, FOREIGN_W_HELD = 1.0, 0.3, 1.5
    RATE_W_BASE, RATE_W_FULL, FLIP_W = 0.005, 0.03, 0.4
    RCV_CONTROL_MULT = 2.0
    TREMOR_SRC_W, TREMOR_RCV_W = 0.3, 0.9
    TREMOR_FREE, TREMOR_SCALE = 0.20, 0.50
    ADR_RAMP_FULL = 0.2
    TILT_W, TILT_PAY_CAP = 5.0, 1.0

    # ---- Basic state flags ----
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
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))

    # ---- Grasp quality & maintenance (Change 3: 5-finger envelope, relative force keep) ----
    F = ctx.src_finger_force.shape[1]
    q_src_pre = torch.clamp(ctx.src_wrap_count.to(dt) / F, 0.0, 1.0)
    q_rcv_pre = torch.clamp(ctx.rcv_wrap_count.to(dt) / F, 0.0, 1.0)
    keep_src = _grip_keep(ctx.src_palm_in_cup, ctx.src_tips_in_cup, ctx.src_ref_palm_in_cup,
                          ctx.src_ref_tips_in_cup, ctx.src_force_share, ctx.src_ref_force_share,
                          ctx.src_force_total, ctx.src_ref_force_total, ctx.src_grasp_ref_valid,
                          0.02, 0.7)
    keep_rcv = _grip_keep(ctx.rcv_palm_in_cup, ctx.rcv_tips_in_cup, ctx.rcv_ref_palm_in_cup,
                          ctx.rcv_ref_tips_in_cup, ctx.rcv_force_share, ctx.rcv_ref_force_share,
                          ctx.rcv_force_total, ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid,
                          0.02, 0.7)
    kept_src_fingers = _finger_keep_ratio(ctx.src_finger_force, ctx.src_ref_force_share,
                                          ctx.src_ref_force_total, ctx.src_grasp_ref_valid)
    kept_rcv_fingers = _finger_keep_ratio(ctx.rcv_finger_force, ctx.rcv_ref_force_share,
                                          ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid)
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    grip_gate = 0.3 + 0.7 * keep_src

    # ---- Requested amount bookkeeping ----
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

    # ---- Stage 1-3: reach, grasp, lift, posture ----
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 0.6 * torch.exp(-4.0 * d_src)
    approach_rcv = 0.6 * torch.exp(-4.0 * d_rcv)
    grasp_src = 0.8 * held_src + 0.8 * torch.exp(-6.0 * d_src) * torch.clamp(ctx.src_hand_closure / 0.35, 0.0, 1.0)
    grasp_rcv = 0.8 * held_rcv + 0.8 * torch.exp(-6.0 * d_rcv) * torch.clamp(ctx.rcv_hand_closure / 0.35, 0.0, 1.0)
    lift_shape_src = 0.5 * (1.0 - torch.exp(-torch.clamp(h_src, min=0.0) / LIFTOFF_H)) + 0.5 * torch.clamp(h_src / LIFT_SRC_H, 0.0, 1.0)
    lift_src = 3.0 * held_src * lift_shape_src
    lift_rcv = 1.5 * held_rcv * torch.clamp(h_rcv / LIFT_RCV_H, 0.0, 1.0) * up_rcv
    # Posture: pre-lift uses wrap fraction, post-lift uses pose keep * finger-keep ratio
    posture_src = held_src * ((1.0 - v_src_ref) * 3.0 * q_src_pre + v_src_ref * 1.5 * keep_src * kept_src_fingers)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * 1.5 * q_rcv_pre + v_rcv_ref * 1.5 * keep_rcv * kept_rcv_fingers)

    # ---- Stage 4: aim spout over receiver mouth ----
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    fine = torch.exp(-(lip_xy / (AIM_PREC_R * cup_r)) ** 2)
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (BELOW_R * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (DROP_R * cup_r)) ** 2)
    height = over_rim * (DROP_FLOOR + (1.0 - DROP_FLOOR) * low_drop)
    aim_xy = 0.2 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) + 0.3 * torch.exp(-AIM_K * lip_xy) + 0.5 * fine
    aim_geo = aim_xy * height
    aim = 5.0 * carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate * aim_geo * want

    # ---- Stage 4b: Rotation help as potential rate (Change 1) ----
    # Compute Phi at current and virtual past state (tau=0.01s)
    src_pos_now = ctx.src_cup_pos
    rcv_mouth_now = ctx.rcv_cup_mouth_pos
    src_up_now = ctx.src_cup_up
    Phi_now = _compute_phi(ctx, src_pos_now, rcv_mouth_now, src_up_now, tau=0.01)

    tau = 0.01
    src_pos_past = src_pos_now - ctx.src_cup_lin_vel * tau
    rcv_mouth_past = rcv_mouth_now - ctx.rcv_cup_lin_vel * tau
    src_up_past = src_up_now - torch.cross(ctx.src_cup_ang_vel, src_up_now, dim=-1) * tau
    src_up_past = torch.nn.functional.normalize(src_up_past, dim=1)
    Phi_past = _compute_phi(ctx, src_pos_past, rcv_mouth_past, src_up_past, tau=0.01)
    dPhi_dt = (Phi_now - Phi_past) / tau

    # Pay gate (factors that have no rate)
    g_pay = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate * settled * (1.0 - released) * want
    tilt_pay = torch.where(dPhi_dt > 0.0,
                           torch.clamp(TILT_W * dPhi_dt, max=TILT_PAY_CAP) * g_pay,
                           0.0)
    # Charge: uncapped as requested, but gated by carry to avoid penalising table state
    tilt_charge = torch.where(dPhi_dt < 0.0,
                              -TILT_W * dPhi_dt * carry_both,
                              0.0)
    tilt_rate = tilt_pay + tilt_charge

    # ---- Stage 5: pour flow & spill ----
    useful = torch.clamp(torch.minimum(d_in, deficit_prev), FLOW_FLOOR, FLOW_CAP)
    pour_flow = 60.0 * useful
    overshoot = -120.0 * torch.clamp(d_in - deficit_prev, min=0.0) - 4.0 * torch.clamp(poured - target, min=0.0)
    spill_w = torch.clamp(50.0 / torch.maximum(target, tol), min=120.0)
    spill = -spill_w * settled * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # ---- Stage 6: finish upright at pre-tilt pose ----
    body_xy = torch.norm(ctx.src_cup_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    pose_tol = max(float(ctx.cup_mouth_z), 0.0) + POSE_TOL_R * cup_r
    pre_tilt_pose = not_nested * torch.exp(-(torch.clamp(body_xy - pose_tol, min=0.0) / (POSE_SIG_R * cup_r)) ** 2)
    finish = 10.0 * closed * (0.3 + 0.7 * on_target) * (FIN_UP_FLOOR + (1.0 - FIN_UP_FLOOR) * up_src) * (FIN_POSE_FLOOR + (1.0 - FIN_POSE_FLOOR) * pre_tilt_pose) * carry_both * up_rcv * (0.4 + 0.6 * keep_src)
    success = 12.0 * ctx.success.to(dt)

    # ---- Workspace constraints (Change 2: meet near centre line) ----
    # Anchor moves from spawn to spawn-x at SIDE_OK margin
    target_rcv_xy = torch.stack([ctx.rcv_cup_spawn_pos[:, 0], torch.full_like(ctx.rcv_cup_spawn_pos[:, 0], SIDE_OK)], dim=1)
    d_home = _dist(ctx.rcv_cup_pos[:, :2], target_rcv_xy)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05, min=0.0) / 0.15)
    home_rcv = 1.0 * carry_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv

    center_src = -1.5 * held_src * torch.tanh(torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)

    # ---- Premature tilt ----
    limit = torch.clamp(ctx.premature_tilt_limit.to(dt), min=1e-3)
    risk = _step((ctx.src_cup_tilt.to(dt) / limit - PRE_LO) / (1.0 - PRE_LO))
    premature = -2.0 * held_src * (1.0 - torch.exp(-10.0 * lip_xy)) * risk - 1.0 * ctx.premature_tilt.to(dt)

    # ---- Safety (real robot) ----
    f_cup = ctx.cup_cup_force
    cup_collision = -2.0 * torch.tanh(f_cup / 2.0) - 5.0 * settled * (f_cup > 0.5).to(dt) - 0.3 * ctx.cup_hit.to(dt)
    f_s = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    f_r = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    hand_foreign = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) * torch.tanh(f_s / 5.0) \
                 - (FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) * torch.tanh(f_r / 5.0)

    # ---- Control & tremor (ramped on adr_progress) ----
    ramp = _step(torch.full_like(up_src, float(ctx.adr_progress) / ADR_RAMP_FULL))
    rate_w = RATE_W_BASE + (RATE_W_FULL - RATE_W_BASE) * ramp
    da = ctx.actions - ctx.prev_actions
    flips = torch.relu(-(ctx.actions * ctx.prev_actions))
    action_rate = -rate_w * (torch.sum(da[:, 0:9] ** 2, dim=-1) + RCV_CONTROL_MULT * torch.sum(da[:, 9:18] ** 2, dim=-1)) \
                - FLIP_W * ramp * (torch.sum(flips[:, 0:9], dim=-1) + RCV_CONTROL_MULT * torch.sum(flips[:, 9:18], dim=-1)) * torch.maximum(held_src, held_rcv)
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