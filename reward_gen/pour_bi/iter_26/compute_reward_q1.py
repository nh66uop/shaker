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
    """[0,1] quality of one hand on its own cup. Envelope + palm + thumb."""
    wrap_q = torch.clamp(wrap_count.to(dt) / 3.0, 0.0, 1.0)
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    return 0.5 * wrap_q + 0.25 * palm_q + 0.25 * thumb_q


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """How closely the hand still holds the cup the way it did at lift. 
    Returns 1 (neutral) before reference exists."""
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
    # Reach & Grasp
    APPROACH_W, APPROACH_K = 0.6, 4.0
    GRASP_W, CLOSE_W, CLOSE_NORM = 0.8, 0.8, 0.35
    # Lift & Posture
    LIFT_SRC_W, LIFT_RCV_W = 3.0, 1.5
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03
    POSTURE_W_SRC, POSTURE_W_RCV = 3.0, 1.2
    KEEP_W_SRC, KEEP_W_RCV = 1.5, 0.8
    KEEP_POS, KEEP_MAG = 0.02, 0.7
    # Aim & Tilt (Iter 26: precision & bead-gating)
    AIM_W, AIM_K, AIM_FAR, LIP_SIGMA = 5.0, 10.0, 0.40, 0.05
    TILT_W = 10.0
    # Flow, Spill, Overshoot (Iter 26: spill heavier, flow uses reachable need)
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 40.0, 0.03, -0.05
    OVER_DELTA_W = 100.0
    OVER_LEVEL_W = 3.0
    SPILL_W = 150.0
    # Finish & Success (Iter 26: opens at tolerance, rewards upright return)
    FINISH_W = 8.0
    SUCCESS_W = 12.0
    # Receiver Grip (Iter 26: reward 5-finger envelope while carrying)
    RCV_GRIP_W = 2.0
    # Workspace & Safety & Control (kept from iter_25)
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
    carry_both = held_src * off_src * held_rcv * off_rcv
    not_nested = (~ctx.cups_nested).to(dt)
    settled = _step(ctx.episode_progress / SETTLE_FRAC)

    up_src = torch.clamp(ctx.src_cup_up[:, 2], 0.0, 1.0)
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], 0.0, 1.0)
    tipped = torch.clamp((1.0 - ctx.src_cup_up[:, 2]) * 0.5, 0.0, 1.0)

    # Grasp quality & maintenance
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
    grip_gate = 0.3 + 0.7 * keep_src

    # =============================== bead tracking & "want" ===================================
    # Iter 26: "want" reflects only reachable beads. Spill reduces still_in_src, turning want off.
    poured = torch.clamp(ctx.bead_in_target_frac.to(dt), 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac.to(dt), 0.0, 1.0)
    still_in_src = torch.clamp(ctx.bead_fill_level.to(dt) - ctx.bead_in_source_frac.to(dt), min=0.0)
    need_frac = torch.clamp(target - poured, min=0.0)
    need = torch.minimum(need_frac, still_in_src)
    tol = 1.0 / torch.clamp(ctx.bead_active_count.to(dt), min=1.0)
    want = _step(need / torch.clamp(tol, min=1e-3))
    err = torch.abs(poured - target)
    on_target = 1.0 - _step((err - tol) / (4.0 * tol))

    # =============================== stage 1-3: reach, grasp, lift ============================
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = APPROACH_W * torch.exp(-APPROACH_K * d_src)
    approach_rcv = APPROACH_W * torch.exp(-APPROACH_K * d_rcv)
    grasp_src = GRASP_W * held_src + CLOSE_W * torch.exp(-6.0 * d_src) * torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_rcv = GRASP_W * held_rcv + CLOSE_W * torch.exp(-6.0 * d_rcv) * torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    lift_shape_src = 0.5 * (1.0 - torch.exp(-torch.clamp(h_src, min=0.0) / LIFTOFF_H)) + 0.5 * torch.clamp(h_src / LIFT_SRC_H, 0.0, 1.0)
    lift_src = LIFT_SRC_W * held_src * lift_shape_src
    lift_rcv = LIFT_RCV_W * held_rcv * torch.clamp(h_rcv / LIFT_RCV_H, 0.0, 1.0) * up_rcv
    posture_src = held_src * ((1.0 - v_src_ref) * POSTURE_W_SRC * q_src + v_src_ref * KEEP_W_SRC * keep_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * POSTURE_W_RCV * q_rcv + v_rcv_ref * KEEP_W_RCV * keep_rcv)

    # =============================== stage 4: aim & tilt (Iter 26) ============================
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]  # positive = lip above receiver mouth
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    above = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / LIP_SIGMA) ** 2)
    # Iter 26: penalty for drop height > 0 (lip too high above receiver rim)
    drop_penalty = torch.exp(-torch.clamp(lip_dz, min=0.0) / 0.04)
    aim_xy = 0.4 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) + 0.6 * torch.exp(-AIM_K * lip_xy)
    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_xy * above * drop_penalty * want

    # Iter 26: tilt income stops immediately after first bead leaves source cup
    first_bead_left = (ctx.bead_in_source_frac.to(dt) < ctx.bead_fill_level.to(dt) * 0.99).to(dt)
    tilt_gate = carry_both * not_nested * up_rcv * grip_gate * (1.0 - first_bead_left)
    tilt_income = TILT_W * tilt_gate * want * tipped * torch.exp(-AIM_K * lip_xy) * above

    # =============================== stage 5: flow, spill, overshoot ==========================
    d_in = ctx.d_in_target.to(dt)
    # Flow pays for beads arriving while still needed
    useful = torch.clamp(torch.minimum(d_in, need), FLOW_FLOOR, FLOW_CAP)
    pour_flow = FLOW_W * useful
    # Overshoot charges past-request arrivals and sustained overshoot
    overshoot = -OVER_DELTA_W * torch.clamp(d_in - need, min=0.0) - OVER_LEVEL_W * torch.clamp(poured - target, min=0.0)
    # Iter 26: spill penalty increased relative to request
    spill = -SPILL_W * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # =============================== stage 6: finish upright (Iter 26) ========================
    # Finishing opens when want is off (request met or unreachable due to spill)
    finish_gate = (1.0 - want) * carry_both * up_rcv * grip_gate
    finish = FINISH_W * finish_gate * up_src * (0.5 + 0.5 * keep_src)
    success = SUCCESS_W * ctx.success.to(dt)

    # =============================== receiver grip quality (Iter 26) ============================
    rcv_wrap_q = torch.clamp(ctx.rcv_wrap_count.to(dt) / 3.0, 0.0, 1.0)
    rcv_grip_carry = RCV_GRIP_W * carry_rcv * rcv_wrap_q * keep_rcv

    # =============================== workspace & safety & control ===============================
    center_src = -CENTER_W * held_src * torch.tanh(torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05, min=0.0) / 0.15)
    home_rcv = HOME_W * carry_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv
    limit = torch.clamp(ctx.premature_tilt_limit.to(dt), min=1e-3)
    risk = _step((ctx.src_cup_tilt.to(dt) / limit - PRE_LO) / (1.0 - PRE_LO))
    premature = -PRE_W * held_src * (1.0 - torch.exp(-AIM_K * lip_xy)) * risk - LATCH_W * ctx.premature_tilt.to(dt)
    f_cup = ctx.cup_cup_force
    cup_collision = -CONTACT_W * torch.tanh(f_cup / 2.0) - HIT_W * settled * (f_cup > 0.5).to(dt) - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_s = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    f_r = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    hand_foreign = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) * torch.tanh(f_s / 5.0) - (FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) * torch.tanh(f_r / 5.0)
    ramp = _step(torch.full_like(up_src, float(ctx.adr_progress) / ADR_RAMP_FULL))
    rate_w = RATE_W_BASE + (RATE_W_FULL - RATE_W_BASE) * ramp
    da = ctx.actions - ctx.prev_actions
    flips = torch.relu(-(ctx.actions * ctx.prev_actions))
    action_rate = -rate_w * (torch.sum(da[:, 0:9] ** 2, dim=-1) + RCV_CONTROL_MULT * torch.sum(da[:, 9:18] ** 2, dim=-1)) - FLIP_W * ramp * (torch.sum(flips[:, 0:9], dim=-1) + RCV_CONTROL_MULT * torch.sum(flips[:, 9:18], dim=-1)) * torch.maximum(held_src, held_rcv)
    qd_s = torch.sqrt(torch.mean(ctx.src_arm_qd ** 2, dim=-1) + 1e-8)
    qd_r = torch.sqrt(torch.mean(ctx.rcv_arm_qd ** 2, dim=-1) + 1e-8)
    tremor = -ramp * (TREMOR_SRC_W * torch.tanh(torch.clamp(qd_s - TREMOR_FREE, min=0.0) / TREMOR_SCALE) + TREMOR_RCV_W * torch.tanh(torch.clamp(qd_r - TREMOR_FREE, min=0.0) / TREMOR_SCALE))

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
        "tilt_income": tilt_income,
        "pour_flow": pour_flow,
        "overshoot": overshoot,
        "spill": spill,
        "finish": finish,
        "success": success,
        "rcv_grip_carry": rcv_grip_carry,
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
