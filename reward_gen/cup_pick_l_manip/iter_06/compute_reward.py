import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
    """0 when a link is off the cup, ->1 once it presses with a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate the sum and extra force buys nothing."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.cup_pos.device
    dtype = ctx.cup_pos.dtype

    # =============================================================================================
    # WEIGHTS
    # Ladder of per-step income (measured): travelling 1.21 < arrived 1.68 < latched 1.98
    #   < curled and near 2.69 < one digit touching 4.40 < five digits touching 14.3
    #   < closed envelope 16.8 < lifted 20.4 < still at the goal 30.1 (+20 per success).
    # THE RULE OF THIS ROUND: no term may sit near its maximum while contact is zero. Every
    # position-only term is multiplied by `pose_mul` (0.40 at zero contact), the proximity ramp is
    # capped at 0.60 of itself without contact, and the two largest terms are pure contact.
    # =============================================================================================
    W_TRAVEL = 1.00          # coarse pre-grasp geometry (0.25 m scale) -- x pose_mul
    W_SEAT = 0.50            # fine seating at the gate (0.02 m) -- was 0.80, cut AND x pose_mul
    W_ORIENT = 0.15          # start orientation, 40 deg dead zone -- x pose_mul
    W_LATCH = 3.00           # approach_done; x pose_mul -> 1.20 at zero contact, i.e. still more
                             #   than the 1.00 open-hand term it switches off. Latching stays a
                             #   step UP (round-1 lesson) and becomes a big one once gripping.
    W_OPEN = 1.00            # default open hand BEFORE the latch, shaped on the WORST joint
                             #   (the gate tests a maximum). KEEP: this is the pre-latch discipline.
    P_PRE_FLEX = 4.00        # flexing before the latch, charged on the joint angles. Raised 2->4
                             #   because closing now pays ~4x what it did.
    P_PRE_TOUCH = 1.20       # touching the cup before the latch. KEEP.
    P_PRE_DISTURB = 0.50     # moving the cup before the latch. KEEP.
    W_REACH = 0.80           # NEW standalone proximity on-ramp, flexion- AND far-side-gated,
                             #   x (0.60 + 0.40*touch) so its zero-contact ceiling is 0.48 --
                             #   strictly less than the 0.70 one single digit's first contact pays.
    W_UNFOLD = 0.20          # leaving the default open posture, MEASURED joints, fades on approach
    W_THUMB_OPP = 0.80       # thumb across the body AND close to the surface, x (0.4 + 0.6*touch)
    W_WRAP = 1.00            # links on the surface, far-side-gated, x (0.25 + 0.75*touch)
    W_DIGIT_MEAN = 5.00      # PURE CONTACT, mean over the 5 digits (1.00 per digit)
    W_DIGIT_LAST2 = 3.00     # PURE CONTACT, the two LOWEST digits (1.50 each): the last two to
                             #   arrive are worth 1.30 each against 1.00 for the first three
    W_PALM_IN = 0.90         # palm-to-cup distance, steepest in the last millimetres,
                             #   x (0.15 + 0.85*digit_engage) so it is not payable on its own
    W_PALM = 0.80            # palm CONTACT, only together with digits that are closing
    W_ENV_LATCH = 1.20       # the environment's own envelope latch
    P_CMD_GAP = 0.40         # NEW: commanded flexion that the joints do not follow, on digits that
                             #   are NOT touching (a finger blocked by the cup is what we want)
    P_SQUEEZE = 0.80         # contact force beyond what holding the cup needs. KEEP.
    P_SHOVE = 1.20           # pushing the cup around while closing on the table. KEEP.
    # ---- lift / carry / hold: structure untouched (never reached in 5 rounds), scaled x~2 so the
    #      ladder stays strictly increasing now that the closing block is twice as large ----------
    W_LIFT = 4.50            # height of the cup above its own spawn height
    W_RISE = 1.20            # upward cup velocity, only until `lifted` latches
    W_GOAL = 4.00            # closing on goal_pos
    W_HOLD = 5.50            # at rest at the goal -> worth more than repeating lifts
    W_SUCCESS = 20.0         # bonus on the step the env counts a success
    P_DESCEND = 1.00         # lowering the held cup < holding it still < raising it
    P_STALL = 0.60           # closed grasp parked on the table, late in the episode
    P_TILT = 0.60            # keep the cup upright
    P_DROP = 1.00            # lifted and then let go
    P_TABLE = 0.50           # palm / fingers driven into the table top
    P_ARM_VEL = 0.12         # smoothness: tiny next to W_LIFT, a slow lift always wins
    P_ARM_FAST = 0.80        # only speeds above 3 rad/s: runaway only. KEEP.
    P_CUP_VEL = 0.15         # linear cup speed while held
    P_CUP_SPIN = 0.08        # angular cup speed while held
    P_ACT_RATE = 0.02        # action jitter
    P_ARM_ACT = 0.01         # arm action magnitude only (fingers must be free to reach +-1)

    zero = torch.zeros_like(ctx.cup_radius)

    # ---- stage flags as floats -------------------------------------------------------------
    approach = ctx.approach_done.to(dtype)
    envelope = ctx.envelope_done.to(dtype)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    pre_latch = 1.0 - approach

    # movable finger joints (ranges <= 0.05 rad are locked; their normalisation is noise):
    # locked = thumb_2(0), index_1(3), middle_1(7), ring_1(11), pinky_1(15), pinky_2(16)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    # per-digit masks over the 19 joints, same order as hand_q_norm
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0      # thumb_3, thumb_4
    digit_joint[1, 4:7] = 1.0      # index_2, _3, _4
    digit_joint[2, 8:11] = 1.0     # middle_2, _3, _4
    digit_joint[3, 12:15] = 1.0    # ring_2, _3, _4
    digit_joint[4, 17:19] = 1.0    # pinky_3, pinky_4
    n_digit_joint = digit_joint.sum(-1)                                   # (5,)

    # =============================================================================================
    # CONTACT (computed first: the whole pose family is scaled by it)
    # =============================================================================================
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5) does this digit touch
    digit_links = link_c.mean(dim=-1)                              # (N,5) how many of its links
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    any_touch = torch.maximum(digit_touch.amax(dim=-1), palm_c)    # (N,)
    n_touch = digit_touch.sum(dim=-1)                              # (N,) 0..5

    # THE ENFORCEMENT OF THIS ROUND'S RULE: every position-only term is multiplied by this, so it
    # can reach at most 40 % of its own maximum while nothing touches the cup.
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)
    pose_mul = 0.40 + 0.60 * grip

    # =============================================================================================
    # Stage A: arrive at the pre-grasp pose with an OPEN hand and latch it
    #          (kept from rounds 4-5: approach 0.95, no-touch 0.99, cup tilt < 0.15 deg)
    # =============================================================================================
    cup_from_palm = ctx.cup_pos - ctx.palm_pos
    d_normal = (cup_from_palm * ctx.palm_normal).sum(-1) - ctx.cup_radius        # gate: [-0.01, 0.02]
    d_finger = (cup_from_palm * ctx.palm_finger_dir).sum(-1) - ctx.cup_radius    # gate: [-0.005, 0.02]
    h_palm = ((ctx.palm_pos - ctx.cup_pos) * ctx.cup_axis).sum(-1)
    d_band = (h_palm.abs() - ctx.cup_half_height).clamp(min=0.0)                 # gate: 0
    pregrasp_err = torch.sqrt(d_normal * d_normal + d_finger * d_finger + d_band * d_band + 1e-12)
    # coarse pull: 0.25 m scale keeps a usable gradient over the whole 0.15-0.49 m spawn range
    pregrasp = 1.0 - torch.tanh(pregrasp_err / 0.25)
    r_travel = W_TRAVEL * pregrasp * pose_mul

    # fine seating, tangential part only, peaking inside the gate's own dead zone
    dz_finger = (d_finger - 0.02).clamp(min=0.0) + (-0.005 - d_finger).clamp(min=0.0)
    seat_err = torch.sqrt(dz_finger * dz_finger + d_band * d_band + 1e-12)
    r_seat = W_SEAT * torch.exp(-seat_err / 0.02) * pose_mul

    # `ready`: the closing terms are gated on the SAME geometry the env gate measures -- flat inside
    # its 2 cm tolerance, 4 cm decay outside -- not on a latch flag the policy cannot observe.
    ready = torch.exp(-(pregrasp_err - 0.02).clamp(min=0.0) / 0.04)
    # ... but closing is worth 3.3x more once the latch is actually set, which protects the approach
    gate = ready * (0.30 + 0.70 * approach)

    # the gate's pose condition is a MAXIMUM over movable joints, so shape the worst joint
    dev_max = ((ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable).amax(dim=-1)
    r_open = W_OPEN * pre_latch * torch.exp(-dev_max / 0.10)
    # and flexing before the latch is charged directly on the joint angles:
    # tanh(0.3/0.15) = 0.96, i.e. ~full price exactly at the threshold the gate tests
    p_pre_flex = P_PRE_FLEX * pre_latch * torch.tanh(dev_max / 0.15)

    # roughly the start orientation (palm_normal ~ -y, palm_finger_dir ~ +x), 40 deg dead zone
    minus_y = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)
    plus_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    cos40 = math.cos(math.radians(40.0))
    cos_n = (ctx.palm_normal * minus_y).sum(-1)
    cos_f = (ctx.palm_finger_dir * plus_x).sum(-1)
    orient_err = (cos40 - cos_n).clamp(min=0.0) + (cos40 - cos_f).clamp(min=0.0)
    r_orient = W_ORIENT * torch.exp(-orient_err / 0.25) * pose_mul

    # latch income: 3.00 * 0.40 = 1.20 with no contact > the 1.00 open-hand term it replaces
    r_latch = W_LATCH * approach * pose_mul

    # before the latch: contact costs more than pressing in could buy, and the cup stays put
    p_pre_touch = P_PRE_TOUCH * pre_latch * any_touch
    cup_disp = (ctx.cup_pos - ctx.cup_spawn_pos).norm(dim=-1)
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    disturb = (torch.tanh(cup_disp / 0.01)
               + torch.tanh(cup_speed / 0.05)
               + torch.tanh(ctx.cup_tilt / math.radians(5.0))) / 3.0
    p_pre_disturb = P_PRE_DISTURB * pre_latch * disturb

    # =============================================================================================
    # Stage B: close a five-digit envelope
    # =============================================================================================
    # ---- how far each digit has CLOSED, from the MEASURED joints (never the command) ------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    # geometric credit requires the digit itself to be closing. The thumb needs less flexion to drop
    # onto the near side than the four fingers need to come round the far side.
    flex_ref = torch.tensor([0.20, 0.35, 0.35, 0.35, 0.35], device=device, dtype=dtype)
    flex_gate = (closure / flex_ref).clamp(min=0.0, max=1.0)                                # (N,5)

    # ---- cylinder geometry of every measured finger link ----------------------------------------
    link_vec = ctx.link_pos - ctx.cup_pos.view(-1, 1, 1, 3)        # (N,5,3,3)
    axis = ctx.cup_axis.view(-1, 1, 1, 3)
    axial = (link_vec * axis).sum(-1)                              # (N,5,3)
    radial_vec = link_vec - axial.unsqueeze(-1) * axis             # (N,5,3,3)
    radial = radial_vec.norm(dim=-1)                               # (N,5,3)
    radial_dir = radial_vec / radial.unsqueeze(-1).clamp(min=1e-6)
    surf_err = (radial - ctx.cup_radius.view(-1, 1, 1)).abs()      # peak ON the surface, never inward
    band_err = (axial.abs() - ctx.cup_half_height.view(-1, 1, 1)).clamp(min=0.0)
    link_surf_dist = torch.sqrt(surf_err * surf_err + band_err * band_err + 1e-12)   # (N,5,3)
    digit_surf_dist = link_surf_dist.amin(dim=-1)                                    # (N,5)

    # ---- FAR-SIDE test (the round-5 fix): where is the link AROUND the body? ---------------------
    # u = cos angle between the link's radial direction about the cup axis and palm_normal.
    # A finger extended PAST the cup has its near-surface link at u <= 0 (palm side / equator);
    # a finger CURLED around the far side has u > 0. So a finger lying alongside the cup, which
    # collected full proximity and wrap credit in rounds 4-5, now collects almost nothing.
    u = (radial_dir * ctx.palm_normal.view(-1, 1, 1, 3)).sum(-1)                     # (N,5,3)
    side = ((u + 0.30) / 0.90).clamp(min=0.0, max=1.0)                               # (N,5,3)

    # the thumb gets no absolute side (nothing tells us which way the opposed thumb must reach);
    # it uses the frame-free opposition measure that worked in round 5.
    tip_dir = radial_dir[:, :, 2, :]                                                 # (N,5,3)
    thumb_dir = tip_dir[:, 0, :]
    fing_mean = tip_dir[:, 1:, :].mean(dim=1)
    fing_dir = fing_mean / fing_mean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    oppose = ((1.0 - (thumb_dir * fing_dir).sum(-1)) * 0.5).clamp(min=0.0, max=1.0)   # (N,)
    thumb_side = (0.35 + 0.65 * oppose).clamp(min=0.0, max=1.0)                       # (N,)
    side = torch.cat([thumb_side.view(-1, 1, 1).expand(-1, 1, 3), side[:, 1:, :]], dim=1)

    # ---- the proximity ON-RAMP, restored in strength but bounded --------------------------------
    # two scales: 0.03 m for long-range guidance, 0.006 m for the last centimetre where round 4
    # stalled. Flexion-gated and far-side-gated, and capped at 0.60 of itself without contact, so
    # ALL FIVE digits hovering perfectly (0.48) are worth less than ONE digit's first solid contact
    # on one link (0.70 from grasp_digit_mean alone).
    prox_link = 0.5 * torch.exp(-link_surf_dist / 0.03) + 0.5 * torch.exp(-link_surf_dist / 0.006)
    near_digit = (prox_link * side).amax(dim=-1) * flex_gate                          # (N,5)
    r_reach = (W_REACH * gate * near_digit.mean(dim=-1)
               * (0.60 + 0.40 * digit_touch.mean(dim=-1)))

    # ---- leaving the deeply reinforced open posture, on the MEASURED joints, fading on approach --
    prox_broad = torch.exp(-digit_surf_dist / 0.03)                                   # (N,5)
    unfold_fade = (1.0 - prox_broad.mean(dim=-1)).clamp(min=0.15)
    r_unfold = W_UNFOLD * gate * unfold_fade * (closure / 0.5).clamp(max=1.0).mean(dim=-1)

    # ---- PURE CONTACT: the two biggest terms in the reward, exactly 0 without it ------------------
    touch_digit = 0.55 * digit_touch + 0.45 * digit_links                             # (N,5) [0,1]
    # equal weight per digit: index and ring pay exactly as much as thumb / middle / pinky
    r_digit_mean = W_DIGIT_MEAN * gate * touch_digit.mean(dim=-1)
    # the TWO digits that arrive LAST, scored on their own: 3->4 and 4->5 digits are +2.5 each, so
    # recruiting the index finger is by far the most profitable single move available.
    sorted_touch, _ = torch.sort(touch_digit, dim=-1)
    r_digit_last2 = W_DIGIT_LAST2 * gate * 0.5 * (sorted_touch[:, 0] + sorted_touch[:, 1])

    # wrapping ALONG the surface (never towards the axis), far-side-gated, flexion-gated, and
    # mostly contact-gated: at most 25 % of this term is reachable without touching.
    wrap_link = torch.exp(-surf_err / 0.02) * torch.exp(-band_err / 0.02) * side
    wrap_digit = wrap_link.mean(dim=-1) * flex_gate * (0.25 + 0.75 * digit_touch)      # (N,5)
    r_wrap = W_WRAP * gate * wrap_digit.mean(dim=-1)

    # thumb across the body AND close to the surface (round-5 tightening kept: it taught the thumb
    # to oppose), now also scaled by whether the thumb actually touches.
    prox_tight = torch.exp(-digit_surf_dist / 0.02)                                   # (N,5)
    r_thumb_opp = (W_THUMB_OPP * gate * oppose * prox_tight[:, 0] * prox_tight[:, 1:].mean(dim=-1)
                   * (0.40 + 0.60 * digit_touch[:, 0]))

    # ---- palm: keeps paying all the way IN, steepest in the last millimetres, never on its own ---
    digit_engage = 0.3 * flex_gate.mean(dim=-1) + 0.7 * digit_touch.mean(dim=-1)       # (N,)
    palm_err = (d_normal - 0.015).clamp(min=0.0) + (-0.005 - d_normal).clamp(min=0.0)
    palm_close = 0.45 * torch.exp(-palm_err / 0.030) + 0.55 * torch.exp(-palm_err / 0.006)
    r_palm_in = W_PALM_IN * gate * palm_close * (0.15 + 0.85 * digit_engage)
    r_palm = W_PALM * palm_c * digit_engage
    r_env_latch = W_ENV_LATCH * envelope

    # ---- commanded flexion that the joints do not follow, on digits that are NOT touching --------
    # (a finger blocked BY THE CUP is exactly what we want and is excluded; a finger commanded shut
    #  that does not move is a command going nowhere -- round 5: command 0.325, measured 0.138)
    cmd_over = (ctx.hand_target_norm - ctx.hand_q_norm).clamp(min=0.0) * movable
    gap_digit = (cmd_over.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint
    p_cmd_gap = P_CMD_GAP * gate * (torch.tanh(gap_digit / 0.20) * (1.0 - digit_touch)).mean(dim=-1)

    # "secure" = palm plus at least four digits (full at five): the envelope the task demands
    secure = ((n_touch - 3.0) / 2.0).clamp(min=0.0, max=1.0) * (0.4 + 0.6 * palm_c)

    # closing stays restrained: the contact reward saturates at a few tenths of a newton, so force
    # beyond what holding the cup needs buys nothing and costs.
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - 15.0).clamp(min=0.0) / 15.0)
                                   + torch.tanh((max_link_force - 6.0).clamp(min=0.0) / 6.0))

    # and the cup must not be shoved around while the hand closes. Fades out as soon as a real lift
    # starts, so it can never tax the lift itself.
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    on_table = 1.0 - torch.tanh(height / 0.02)
    horiz_disp = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    shove = (torch.tanh(horiz_disp / 0.015)
             + torch.tanh(horiz_speed / 0.05)
             + torch.tanh(ctx.cup_tilt / math.radians(10.0))) / 3.0
    p_shove = P_SHOVE * on_table * shove

    # =============================================================================================
    # Stage C: lift   (structure unchanged, still gated on a real envelope, scaled with the block)
    # =============================================================================================
    h_scale = max(2.0 * ctx.lift_latch_height, 0.02)               # ~0 on the table, ~1 near goal
    r_lift = W_LIFT * secure * torch.tanh(height / h_scale)
    vz = ctx.cup_lin_vel[:, 2]
    # upward speed pays only until the lift latches: teaches the first lift fast, and cannot be
    # farmed by setting the cup down after the goal and raising it again
    r_rise = W_RISE * secure * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.15)
    # descending while holding costs, so descending < staying still < rising
    p_descend = P_DESCEND * secure * torch.tanh(vz.clamp(max=0.0).abs() / 0.15)
    # a closed grasp parked on the table is not worth staying for (free early, costly late)
    p_stall = P_STALL * secure * (1.0 - lifted) * ctx.episode_progress

    # =============================================================================================
    # Stage D: carry to the goal and hold still   (structure unchanged)
    # =============================================================================================
    r_goal = W_GOAL * secure * (1.0 - torch.tanh(ctx.goal_dist / 0.12))
    near_goal = 1.0 - torch.tanh(ctx.goal_dist / 0.05)             # fixed scale, not success_tol
    still = torch.exp(-cup_speed / 0.08)
    r_hold = W_HOLD * secure * near_goal * still                   # resting at the goal > relifting
    r_success = W_SUCCESS * success

    # =============================================================================================
    # Safety and smoothness (all small next to the stage income)
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    p_drop = P_DROP * lifted * (1.0 - secure)
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = P_TABLE * (palm_pen + floor_pen)
    arm_speed = ctx.arm_qd.norm(dim=-1)
    # arm joints are commanded at ~0.3 rad/s, so this stays tiny: a slow lift always wins
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * secure) * torch.tanh(arm_speed / 1.0)
    # and this one only bites above 3 rad/s, i.e. in the runaway states of round 3
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = secure * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                          + P_CUP_SPIN * torch.tanh(ctx.cup_ang_vel.norm(dim=-1) / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_travel + r_seat + r_orient + r_open + r_latch
        + r_reach + r_unfold + r_thumb_opp + r_wrap
        + r_digit_mean + r_digit_last2
        + r_palm_in + r_palm + r_env_latch
        + r_lift + r_rise
        + r_goal + r_hold + r_success
        - p_pre_flex - p_pre_touch - p_pre_disturb
        - p_cmd_gap - p_squeeze - p_shove
        - p_descend - p_stall
        - p_tilt - p_drop - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero  # keeps shape (N,) even if a term ever broadcasts

    components = {
        "pregrasp_travel": r_travel,
        "pregrasp_seat": r_seat,
        "pregrasp_orient": r_orient,
        "pregrasp_open_hand": r_open,
        "approach_latch": r_latch,
        "pen_pre_flex": -p_pre_flex,
        "pen_pre_touch": -p_pre_touch,
        "pen_pre_disturb": -p_pre_disturb,
        "grasp_reach": r_reach,
        "grasp_unfold": r_unfold,
        "grasp_thumb_oppose": r_thumb_opp,
        "grasp_wrap": r_wrap,
        "grasp_digit_mean": r_digit_mean,
        "grasp_digit_last2": r_digit_last2,
        "grasp_palm_in": r_palm_in,
        "grasp_palm_contact": r_palm,
        "envelope_latch": r_env_latch,
        "pen_cmd_gap": -p_cmd_gap,
        "pen_squeeze": -p_squeeze,
        "pen_shove": -p_shove,
        "lift_height": r_lift,
        "lift_rise": r_rise,
        "goal_reach": r_goal,
        "goal_hold": r_hold,
        "success_bonus": r_success,
        "pen_descend": -p_descend,
        "pen_stall": -p_stall,
        "pen_tilt": -p_tilt,
        "pen_drop": -p_drop,
        "pen_table": -p_table,
        "pen_arm_vel": -p_arm_vel,
        "pen_arm_fast": -p_arm_fast,
        "pen_cup_vel": -p_cup_vel,
        "pen_action": -p_action,
    }
    return reward, components
