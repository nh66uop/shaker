

### 1. Task Analysis & Stages

The task is a **bimanual pouring operation** requiring coordinated control of two 7-DOF arms with underactuated hands. The policy must learn a sequential skill chain:

1. **Approach & Grasp**: Both arms must approach their respective cups. The right hand (src) grasps the bead-filled cup; the left hand (rcv) grasps the empty cup. A fingertip pinch is sufficient. The reward must reward proximity, finger closure, and confirmed contact (`grasped` flag).
2. **Lift**: Both cups must leave the table. The source cup needs to be lifted higher (~0.4–0.6 m) to enable future tilting, while the receiver cup needs moderate lift (~0.1–0.2 m). Height should be rewarded only after grasp is stable.
3. **Placement**: The receiver cup's mouth must be positioned under the source cup's spout. Crucially, the placement target must be **absolute** (fixed in space), not relative to the source cup, so the left arm isn't dragged upward when the right arm lifts. The receiver cup must also stay upright.
4. **Aim**: The source cup's lip must be slightly above the receiver mouth and horizontally close (~few cm). This should use a **difference form** (rewarded only when closing the gap), preventing the policy from getting credit for parking far away.
5. **Tilt & Direction**: The source cup must tilt toward the receiver cup. No fixed angle target is needed (the policy finds the optimal ~70–90°), but the **direction** must be correct. A small positive reward for increasing tilt toward the receiver (when in the pouring window) combined with a penalty for tilting away will guide exploration without over-constraining.
6. **Outcome**: Beads transferring to the receiver cup must be the dominant reward signal. Incremental transfer (`d_in_target`) is rewarded; spill (`d_spill`) is penalized. A level term (`bead_in_target_frac`) and success bonus close the loop.
7. **Control & Safety**: Velocity, acceleration, flip penalties, joint limits, and collision forces must be bounded and scaled so they don't swamp the task signal but keep deployment safe.

**Why the previous reward failed:**
- `task/episode_success` stayed at 0. State income (grasp/lift/place) averaged ~3.0/step, while outcome income was ~0.0001. The policy learned to grasp, lift, and park.
- Placement/aim used relative targets that moved up with the source cup, making it kinematically hard for the receiver arm to catch up.
- Tilt had only penalties (`tilt_dir`, `src_spin`), so tilting was always a net loss. No positive signal existed to encourage the first bead to fall.
- `lift_src` paid heavily for height, encouraging the policy to hoist the cup and stop.

**Fixes applied:**
- Absolute height target for receiver placement.
- Difference forms for placement/aim (credit only for closing gaps).
- Small positive `tilt_progress` reward inside the permission window.
- Scaled down state income, scaled up outcome weights so pouring dominates once it starts.
- Gating ensures penalties/rewards activate only when physically meaningful.

---

### 2. Improved Reward Function

```python
import torch
import math


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ===================================================================================================
    # iter_23: Rebuilt from operator briefs. Fixes iter_21's "parking trap" by:
    #   1. Using absolute height targets for placement/aim (rcv doesn't get dragged up).
    #   2. Difference forms for placement/aim (credit only for closing gaps).
    #   3. Adding small positive tilt_progress reward inside the permission window.
    #   4. Scaling outcome >> state income once beads start moving.
    # ===================================================================================================

    # ---- Constants & Weights --------------------------------------------------------------------------
    W_ALIVE = 1.0
    # A. Grasp
    W_GRASP_SRC, W_GRASP_RCV = 1.5, 1.5
    # B. Lift
    W_LIFT_SRC, W_LIFT_RCV = 1.0, 0.5
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.40, 0.15
    # C. Placement (absolute targets)
    W_PLACE = 2.0
    PLACE_Z_TARGET = 0.15        # m, absolute height for rcv mouth
    PLACE_H_DIST_TARGET = 0.06   # m, horizontal src->rcv mouth distance
    # D. Aim (lip above rcv mouth, close horizontally)
    W_AIM = 2.0
    AIM_Z_MIN = 0.02             # m, lip must be above rim
    AIM_H_DIST_MAX = 0.08        # m, horizontal lip->rcv mouth distance
    # E. Tilt & Direction
    W_TILT_DIR = 1.0             # penalty for leaning away from rcv
    W_TILT_PROGRESS = 0.3        # small reward for leaning towards rcv inside window
    # F. Outcome
    W_POUR = 10.0                # per-bead transfer (scaled by bead count internally)
    W_SPILL = 15.0               # per-bead spill penalty
    W_TRANSFER = 2.0             # level term
    W_SUCCESS = 50.0             # success bonus
    # G. Control & Safety
    W_SPEED, W_SPIN = 0.5, 0.5
    W_FLIP, W_BANG, W_SAT = 0.2, 0.5, 0.3
    W_RATE = 0.05
    W_ARM = 0.3
    W_COLLISION, W_FOREIGN = 0.5, 0.5
    # Helpers
    EPS = 1e-6
    dtype = ctx.src_cup_tilt.dtype

    # ---- A. Grasp -------------------------------------------------------------------------------------
    def _grasp_val(tips, force, palm_f, closure, grasped, cup_pos, cup_up, cup_mouth_z, spawn_z):
        h = torch.clamp(cup_pos[:, 2] - spawn_z, 0.0, 1.0)
        tip_r = 0.027
        up = cup_up.unsqueeze(1)
        rel = tips - cup_pos.unsqueeze(1)
        h_tip = (rel * up).sum(dim=-1)
        rad_vec = rel - h_tip.unsqueeze(-1) * up
        rad = rad_vec.norm(dim=-1)
        gap = torch.clamp(rad - tip_r, 0.0)
        # thumb vs fingers geometry
        thumb_h = h_tip[:, 0]
        h_err_t = torch.clamp(-thumb_h, 0.0) + torch.clamp(thumb_h - 0.5 * cup_mouth_z, 0.0)
        d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
        h_oth = h_tip[:, 1:]
        h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, 0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, 0.0)
        d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values
        reach = torch.exp(-10.0 * (d_thumb + d_oth))
        close = torch.tanh(closure / 0.7)
        force_t = torch.tanh(force[:, 0] / 0.5)
        force_o = torch.tanh(force[:, 1:].sum(dim=1) / 1.0)
        contact = 0.5 * (force_t + torch.minimum(force_t, force_o))
        held = 0.4 + 0.6 * contact
        free = 0.5 * reach + 0.5 * close
        return torch.where(grasped.bool(), held, free) * torch.clamp(h / 0.05, 0.0, 1.0)

    grasp_src = _grasp_val(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force,
                           ctx.src_hand_closure, ctx.src_grasped, ctx.src_cup_pos, ctx.src_cup_up,
                           ctx.cup_mouth_z, ctx.src_cup_spawn_pos[:, 2])
    grasp_rcv = _grasp_val(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force,
                           ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                           ctx.cup_mouth_z, ctx.rcv_cup_spawn_pos[:, 2])

    # ---- B. Lift --------------------------------------------------------------------------------------
    h_src_raw = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2], 0.0, 1.0)
    h_rcv_raw = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2], 0.0, 1.0)
    lift_src = torch.tanh(h_src_raw / LIFT_TARGET_SRC) * ctx.src_grasped.to(dtype)
    lift_rcv = torch.tanh(h_rcv_raw / LIFT_TARGET_RCV) * ctx.rcv_grasped.to(dtype)

    # ---- C. Placement (difference form, absolute rcv height) -------------------------------------------
    e_z = ctx.rcv_cup_mouth_pos[:, 2] - PLACE_Z_TARGET
    e_h = (ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_mouth_pos[:, :2]).norm(dim=1)
    dd_z = ctx.rcv_cup_lin_vel[:, 2]  # target is fixed, so dd = rcv velocity
    dd_h = (ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_mouth_pos[:, :2]).norm(dim=1)
    # approximate closing rate for horizontal
    rcv_mouth_v_xy = ctx.rcv_cup_lin_vel[:, :2]
    src_mouth_v_xy = ctx.src_cup_lin_vel[:, :2]
    v_rel_xy = rcv_mouth_v_xy - src_mouth_v_xy
    dd_h_rate = -torch.clamp((v_rel_xy * (ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_mouth_pos[:, :2])).sum(dim=1) / (dd_h + EPS), 0.0, 1.0)
    place_z_rate = torch.clamp(-dd_z / 0.10, -1.0, 1.0)
    place_h_rate = torch.clamp(dd_h_rate / 0.10, -1.0, 1.0)
    place = torch.where(e_z < 0.0, place_z_rate, torch.zeros_like(place_z_rate)) \
          + torch.where(e_h < PLACE_H_DIST_TARGET, place_h_rate, torch.zeros_like(place_h_rate))
    place *= (ctx.src_grasped * ctx.rcv_grasped).to(dtype)

    # ---- D. Aim (lip above rcv mouth, close horizontally) ----------------------------------------------
    lip_z = ctx.src_cup_mouth_pos[:, 2] - ctx.cup_radius * torch.sin(torch.clamp(ctx.src_cup_tilt, 0.0, math.pi))
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + ctx.cup_radius * torch.cos(torch.clamp(ctx.src_cup_tilt, 0.0, math.pi)) * torch.tensor([1.0, 0.0], dtype=dtype, device=ctx.src_cup_mouth_pos.device)
    # simplified lip position: assume tilt is along x for geometry, adjust to actual cup_up projection if needed.
    # Using robust projection:
    src_up_xy = ctx.src_cup_up[:, :2]
    src_up_h = src_up_xy.norm(dim=1, keepdim=True) + EPS
    d_hat = (ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]) / ((ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]).norm(dim=1, keepdim=True) + EPS)
    cos_psi = torch.clamp((src_up_xy * d_hat).sum(dim=1) / src_up_h, -1.0, 1.0)
    theta_eff = torch.acos(cos_psi) * torch.clamp(ctx.src_cup_tilt, 0.0, 1.0)
    lip_xy_vec = ctx.src_cup_mouth_pos[:, :2] + ctx.cup_radius * torch.cos(theta_eff).unsqueeze(-1) * d_hat
    lip_xy_dist = (lip_xy_vec - ctx.rcv_cup_mouth_pos[:, :2]).norm(dim=1)
    lip_above_z = torch.clamp(lip_z - ctx.rcv_cup_mouth_pos[:, 2], 0.0, 1.0)
    
    # Difference form: reward when closing gap to target
    e_lip_z = lip_above_z - AIM_Z_MIN
    e_lip_h = lip_xy_dist - AIM_H_DIST_MAX
    dd_lip_z = ctx.src_cup_lin_vel[:, 2] - ctx.rcv_cup_lin_vel[:, 2]
    dd_lip_h_rate = torch.clamp(-(ctx.rcv_cup_lin_vel[:, :2] - ctx.src_cup_lin_vel[:, :2]).norm(dim=1) / 0.10, -1.0, 1.0)
    
    aim_z_rate = torch.clamp(-dd_lip_z / 0.10, -1.0, 1.0) * torch.where(lip_above_z < AIM_Z_MIN, torch.ones_like(lip_above_z), torch.zeros_like(lip_above_z))
    aim_h_rate = dd_lip_h_rate * torch.where(lip_xy_dist > AIM_H_DIST_MAX, torch.ones_like(lip_xy_dist), torch.zeros_like(lip_xy_dist))
    aim = (aim_z_rate + aim_h_rate) * (ctx.src_grasped * ctx.rcv_grasped).to(dtype)

    # ---- E. Tilt & Direction ----------------------------------------------------------------------------
    # Direction error: angle between cup lean and vector to rcv mouth
    dir_err = 0.5 * (1.0 - cos_psi)  # 0 when leaning towards rcv
    tilt_dir_pen = -W_TILT_DIR * ctx.src_grasped.to(dtype) * ctx.src_cup_tilt * dir_err
    # Permission window: lip close enough horizontally, above rim, leaning towards rcv
    perm = torch.where(lip_xy_dist < 0.15, torch.ones_like(lip_xy_dist), torch.zeros_like(lip_xy_dist)) \
         * torch.where(lip_above_z > 0.0, torch.ones_like(lip_above_z), torch.zeros_like(lip_above_z)) \
         * torch.where(cos_psi > 0.5, torch.ones_like(cos_psi), torch.zeros_like(cos_psi))
    # Small positive progress reward for increasing tilt towards rcv inside window
    tilt_progress = W_TILT_PROGRESS * perm * torch.clamp(ctx.src_cup_tilt - ctx.prev_actions[:, 0] * 0.5, 0.0, 1.0) * (ctx.src_grasped * ctx.rcv_grasped).to(dtype)
    # (Note: prev_actions[:,0] is src palm delta; using a simple tilt proxy or just current tilt rate if available. 
    #  Since ctx doesn't have prev tilt, we approximate progress via tilt magnitude inside window)
    tilt_progress = W_TILT_PROGRESS * perm * torch.tanh(ctx.src_cup_tilt / 0.5) * (ctx.src_grasped * ctx.rcv_grasped).to(dtype)

    # ---- F. Outcome -----------------------------------------------------------------------------------
    n_beads = max(int(ctx.num_beads), 1)
    d_in = ctx.d_in_target * n_beads
    pour_paid = torch.clamp(d_in, 0.0, 1.0)
    d_spill = ctx.d_spill * n_beads
    spill_pen = -W_SPILL * torch.clamp(d_spill, 0.0, 1.0)
    transferred = W_TRANSFER * torch.tanh(ctx.bead_in_target_frac / 0.5) * (1.0 - ctx.cups_nested.to(dtype))
    success_bonus = W_SUCCESS * ctx.success.to(dtype) * torch.clamp(1.0 - ctx.bead_spill_frac / 0.25, 0.0, 1.0)

    # ---- G. Control & Safety --------------------------------------------------------------------------
    v_src = ctx.src_cup_lin_vel.norm(dim=1)
    w_src = ctx.src_cup_ang_vel.norm(dim=1)
    v_rcv = ctx.rcv_cup_lin_vel.norm(dim=1)
    w_rcv = ctx.rcv_cup_ang_vel.norm(dim=1)
    
    src_speed = -W_SPEED * torch.tanh(torch.clamp(v_src - 0.15, 0.0, 1.0) / 0.15) * ctx.src_grasped.to(dtype)
    src_spin = -W_SPIN * torch.tanh(torch.clamp(w_src - 1.5, 0.0, 1.0) / 1.5) * ctx.src_grasped.to(dtype)
    rcv_steady = -W_SPIN * torch.tanh(torch.clamp(v_rcv - 0.05, 0.0, 1.0) / 0.10) * ctx.rcv_grasped.to(dtype)
    
    a_abs = ctx.actions.abs()
    da = ctx.actions - ctx.prev_actions
    flips = torch.clamp(-(ctx.actions * ctx.prev_actions), 0.0)
    act_flip = -W_FLIP * (flips[:, 0:6].sum(dim=1) + flips[:, 12:18].sum(dim=1) + flips[:, 6:12].sum(dim=1) + flips[:, 18:24].sum(dim=1))
    bang = -W_BANG * (torch.mean(flips[:, 0:6] > 0.1, dim=1) + torch.mean(flips[:, 12:18] > 0.1, dim=1))
    sat = -W_SAT * (torch.mean(torch.clamp(a_abs[:, 0:6] - 0.7, 0.0, 1.0) / 0.3, dim=1) + torch.mean(torch.clamp(a_abs[:, 12:18] - 0.7, 0.0, 1.0) / 0.3, dim=1))
    cmd_rate = -W_RATE * ((da[:, 0:6] ** 2).sum(dim=1) + (da[:, 12:18] ** 2).sum(dim=1) + (da[:, 6:12] ** 2).sum(dim=1) + (da[:, 18:24] ** 2).sum(dim=1))
    
    arm_limit = W_ARM * (torch.clamp(0.3 - ctx.src_arm_limit_margin.min(dim=1).values, 0.0, 1.0) + torch.clamp(0.3 - ctx.rcv_arm_limit_margin.min(dim=1).values, 0.0, 1.0))
    arm_shake = W_ARM * (torch.tanh(ctx.src_arm_qdd.abs().mean(dim=1) / 10.0) + torch.tanh(ctx.rcv_arm_qdd.abs().mean(dim=1) / 10.0))
    
    cup_coll = -W_COLLISION * torch.tanh(ctx.cup_cup_force / 5.0)
    foreign = -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    nested = -W_COLLISION * ctx.cups_nested.to(dtype)

    # ---- Combine --------------------------------------------------------------------------------------
    reward = W_ALIVE * torch.ones_like(ctx.src_cup_tilt)
    reward = reward + W_GRASP_SRC * grasp_src + W_GRASP_RCV * grasp_rcv
    reward = reward + W_LIFT_SRC * lift_src + W_LIFT_RCV * lift_rcv
    reward = reward + W_PLACE * place
    reward = reward + W_AIM * aim
    reward = reward + tilt_dir_pen + tilt_progress
    reward = reward + W_POUR * pour_paid + spill_pen + transferred + success_bonus
    reward = reward + src_speed + src_spin + rcv_steady + act_flip + bang + sat + cmd_rate + arm_limit + arm_shake
    reward = reward + cup_coll + foreign + nested

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "grasp_src": W_GRASP_SRC * grasp_src,
        "grasp_rcv": W_GRASP_RCV * grasp_rcv,
        "lift_src": W_LIFT_SRC * lift_src,
        "lift_rcv": W_LIFT_RCV * lift_rcv,
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": tilt_dir_pen,
        "tilt_progress": tilt_progress,
        "pour_delta": W_POUR * pour_paid,
        "spill_delta": spill_pen,
        "transferred": transferred,
        "success_bonus": success_bonus,
        "src_speed": src_speed,
        "src_spin": src_spin,
        "rcv_steady": rcv_steady,
        "act_flip": act_flip,
        "palm_bang": bang,
        "palm_sat": sat,
        "cmd_rate": cmd_rate,
        "arm_posture": -arm_limit - arm_shake,
        "safety": cup_coll + foreign + nested,
    }
    return reward, comps
```