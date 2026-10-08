import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
    """0 when a link is off the cup, ->1 once it presses with a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate the sum."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.cup_pos.device
    dtype = ctx.cup_pos.dtype

    # ---- weights -------------------------------------------------------------------------
    # Ladder of per-step ceilings: pre-latch ~1.15 < latched ~1.6 < envelope ~5.4
    # < lifted ~7.6 < still at the goal ~12.2 (+ success spike). Every stage transition is a
    # step UP that STAYS up, so no stage is a place to park and latching is never a loss.
    W_PREGRASP = 0.70      # pre-grasp geometry of the env gate -- PAID BEFORE *AND AFTER* THE LATCH
    W_OPEN = 0.30          # default open hand, only before the latch (the gate demands it)
    W_ORIENT = 0.15        # start orientation, only before the latch, 40 deg dead zone
    W_LATCH = 0.90         # approach_done income: > W_OPEN + W_ORIENT that it replaces
    P_PRE_TOUCH = 1.20     # NEW: touching the cup before the latch. This is what broke round 1:
                           #      contact maximised the reach term and blocked the no_touch gate.
    P_PRE_DISTURB = 0.80   # NEW: moving/tilting the cup before the latch (it tilted 5 deg)
    W_DIGIT_MEAN = 1.00    # mean over the 5 digits, EVERY digit weighted the same
    W_DIGIT_LAST = 0.80    # min over the 5 digits: only the LAST digit to arrive earns it
    W_DIGIT_NEAR = 0.60    # NEW: worst digit's pre-contact distance to the cup surface
    W_WRAP = 0.80          # links ON the cup surface inside the graspable band
    W_OPPOSE = 0.40        # contacts spread around the body, not bunched on one side
    W_CURL = 0.25          # small closing drive, only for digits not yet touching
    W_PALM = 0.40          # palm against the cup side (envelope_done needs it)
    W_ENV_LATCH = 0.40     # the environment's own envelope latch
    W_LIFT = 2.20          # height of the cup above its own spawn height
    W_RISE = 0.60          # upward cup velocity, only until `lifted` latches
    W_GOAL = 2.00          # closing on goal_pos
    W_HOLD = 2.60          # at rest at the goal -> worth more than repeating lifts
    W_SUCCESS = 10.0       # bonus on the step the env counts a success
    P_DESCEND = 0.40       # lowering the held cup < holding it still < raising it
    P_STALL = 0.15         # closed grasp parked on the table, late in the episode
    P_TILT = 0.60          # keep the cup upright
    P_DROP = 0.50          # lifted and then let go
    P_TABLE = 0.50         # palm / fingers driven into the table top
    P_ARM_VEL = 0.12       # smoothness: tiny next to W_LIFT, a slow lift always wins
    P_CUP_VEL = 0.15       # linear cup speed while held
    P_CUP_SPIN = 0.08      # angular cup speed while held
    P_ACT_RATE = 0.02      # action jitter
    P_ARM_ACT = 0.01       # arm action magnitude only (fingers must be free to reach +-1)

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
    n_movable = movable.sum()
    # per-digit masks over the 19 joints, same order as hand_q_norm
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0      # thumb_3, thumb_4
    digit_joint[1, 4:7] = 1.0      # index_2..4
    digit_joint[2, 8:11] = 1.0     # middle_2..4
    digit_joint[3, 12:15] = 1.0    # ring_2..4
    digit_joint[4, 17:19] = 1.0    # pinky_3, pinky_4

    # =======================================================================================
    # Stage A: arrive at the pre-grasp pose and LATCH it
    # =======================================================================================
    # The env gate checks three geometric quantities; shape exactly those, so the shaping peak
    # and the gate coincide and there is no way to sit just outside the gate for more money.
    cup_from_palm = ctx.cup_pos - ctx.palm_pos
    d_normal = (cup_from_palm * ctx.palm_normal).sum(-1) - ctx.cup_radius        # -> 0
    d_finger = (cup_from_palm * ctx.palm_finger_dir).sum(-1) - ctx.cup_radius    # -> 0
    h_palm = ((ctx.palm_pos - ctx.cup_pos) * ctx.cup_axis).sum(-1)
    d_band = (h_palm.abs() - ctx.cup_half_height).clamp(min=0.0)                 # -> 0
    pregrasp_err = torch.sqrt(d_normal * d_normal + d_finger * d_finger + d_band * d_band + 1e-12)
    # scale 0.25 m keeps a usable gradient over the whole 0.15-0.49 m spawn range
    pregrasp = 1.0 - torch.tanh(pregrasp_err / 0.25)
    # FIX 1: paid with the SAME weight after the latch as before it, so latching never removes
    # income and there is a continuous slope from arriving through closing (a closed envelope
    # satisfies this geometry too, so it keeps paying all the way to the goal).
    r_pregrasp = W_PREGRASP * pregrasp

    # the gate also demands the default OPEN hand; only meaningful before the latch
    open_err = ((ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable).sum(-1) / n_movable
    r_open = W_OPEN * pre_latch * torch.exp(-open_err / 0.15)

    # the gate demands roughly the start orientation (palm_normal ~ -y, palm_finger_dir ~ +x).
    # Dead zone out to 40 deg because far corners legitimately need up to ~40 deg of yaw.
    minus_y = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)
    plus_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    cos40 = math.cos(math.radians(40.0))
    cos_n = (ctx.palm_normal * minus_y).sum(-1)
    cos_f = (ctx.palm_finger_dir * plus_x).sum(-1)
    orient_err = (cos40 - cos_n).clamp(min=0.0) + (cos40 - cos_f).clamp(min=0.0)
    r_orient = W_ORIENT * pre_latch * torch.exp(-orient_err / 0.25)

    # FIX 1b: the latch bonus is larger than the two terms it switches off, so the step on which
    # approach_done fires is strictly a step up, and it stays up for the rest of the episode.
    r_latch = W_LATCH * approach

    # ---- contacts (used by several stages) --------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5) does this digit touch
    digit_links = link_c.mean(dim=-1)                              # (N,5) how many of its links
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    any_touch = torch.maximum(digit_touch.amax(dim=-1), palm_c)

    # FIX 2 (the round-1 exploit): before the latch, ANY contact with the cup costs more than the
    # shaping that pressing into it could ever buy, and the cup must be left exactly where it is.
    p_pre_touch = P_PRE_TOUCH * pre_latch * any_touch
    cup_disp = (ctx.cup_pos - ctx.cup_spawn_pos).norm(dim=-1)
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    disturb = (torch.tanh(cup_disp / 0.01)
               + torch.tanh(cup_speed / 0.05)
               + torch.tanh(ctx.cup_tilt / math.radians(5.0))) / 3.0
    p_pre_disturb = P_PRE_DISTURB * pre_latch * disturb

    # =======================================================================================
    # Stage B: close a five-digit envelope (behind the latch, which is now cheap to pass)
    # =======================================================================================
    # cylinder geometry of every measured finger link
    link_vec = ctx.link_pos - ctx.cup_pos.view(-1, 1, 1, 3)        # (N,5,3,3)
    axis = ctx.cup_axis.view(-1, 1, 1, 3)
    axial = (link_vec * axis).sum(-1)                              # (N,5,3)
    radial_vec = link_vec - axial.unsqueeze(-1) * axis             # (N,5,3,3)
    radial = radial_vec.norm(dim=-1)                               # (N,5,3)
    surf_err = (radial - ctx.cup_radius.view(-1, 1, 1)).abs()      # peak ON the surface
    band_err = (axial.abs() - ctx.cup_half_height.view(-1, 1, 1)).clamp(min=0.0)

    # per-digit score rises as the 2nd and 3rd link of that digit arrive (never saturates on one)
    digit_score = 0.5 * digit_touch + 0.5 * digit_links            # (N,5) in [0,1]
    # equal weight per digit: index and ring pay exactly as much as thumb / middle / pinky
    r_digit_mean = W_DIGIT_MEAN * approach * digit_score.mean(dim=-1)
    # the weakest digit: 0 until the LAST digit lands, so 5 digits >> 4 digits >> 3 digits
    r_digit_last = W_DIGIT_LAST * approach * digit_score.amin(dim=-1)

    # FIX 3: a pre-contact slope for the digit that is furthest from the cup. Contact terms give
    # a finger that never touches no gradient at all -- that is how the ring finger stayed at
    # 0.001 in the earlier round. Distance is to the cup SURFACE, inside the band.
    link_surf_dist = torch.sqrt(surf_err * surf_err + band_err * band_err + 1e-12)   # (N,5,3)
    digit_near = torch.exp(-link_surf_dist.amin(dim=-1) / 0.03)                      # (N,5)
    r_digit_near = W_DIGIT_NEAR * approach * digit_near.amin(dim=-1)

    # wrapping ALONG the surface: moving further inward (towards the axis) loses reward, so this
    # never pays for squeezing towards the centre. Equal weight per digit and per link.
    wrap_link = torch.exp(-surf_err / 0.02) * torch.exp(-band_err / 0.02)
    r_wrap = W_WRAP * approach * wrap_link.mean(dim=-1).mean(dim=-1)

    # encirclement: contact-weighted mean of the fingertip radial directions. Norm ~1 when every
    # contact sits on one side, ~0 when the digits surround the body. Frame-free, so it assumes
    # nothing about where the thumb ends up.
    tip_radial = radial_vec[:, :, 2, :]                            # (N,5,3)
    tip_dir = tip_radial / tip_radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    w_sum = digit_touch.sum(dim=-1)                                # 0..5
    mean_dir = (digit_touch.unsqueeze(-1) * tip_dir).sum(dim=1) / w_sum.clamp(min=1e-6).unsqueeze(-1)
    r_oppose = W_OPPOSE * approach * (1.0 - mean_dir.norm(dim=-1)).clamp(min=0.0) * (w_sum / 5.0)

    # small closing drive that switches itself OFF per digit on contact, so it can never fight
    # the surface terms by driving a finger through the cup.
    curl = (ctx.hand_q_norm.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / digit_joint.sum(-1)
    r_curl = W_CURL * approach * ((1.0 - digit_touch) * curl.clamp(max=0.7) / 0.7).mean(dim=-1)

    r_palm = W_PALM * approach * palm_c
    r_env_latch = W_ENV_LATCH * envelope

    # FIX 4: "secure" = palm plus at least four digits (full at five), the envelope the task
    # demands. Round 1 gated the lift on bare digit contact, which paid for fingertip dragging.
    secure = ((w_sum - 3.0) / 2.0).clamp(min=0.0, max=1.0) * (0.4 + 0.6 * palm_c)

    # =======================================================================================
    # Stage C: lift
    # =======================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
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

    # =======================================================================================
    # Stage D: carry to the goal and hold still
    # =======================================================================================
    r_goal = W_GOAL * secure * (1.0 - torch.tanh(ctx.goal_dist / 0.12))
    near_goal = 1.0 - torch.tanh(ctx.goal_dist / 0.05)             # fixed scale, not success_tol
    still = torch.exp(-cup_speed / 0.08)
    r_hold = W_HOLD * secure * near_goal * still                   # resting at the goal > relifting
    r_success = W_SUCCESS * success

    # =======================================================================================
    # Safety and smoothness (all small next to the stage income)
    # =======================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    p_drop = P_DROP * lifted * (1.0 - secure)
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = P_TABLE * (palm_pen + floor_pen)
    # arm joints are capped near 0.3 rad/s, so this stays <= ~0.15: a slow lift always wins
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * secure) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_cup_vel = secure * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                          + P_CUP_SPIN * torch.tanh(ctx.cup_ang_vel.norm(dim=-1) / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_pregrasp + r_open + r_orient + r_latch
        + r_digit_mean + r_digit_last + r_digit_near + r_wrap + r_oppose + r_curl
        + r_palm + r_env_latch
        + r_lift + r_rise
        + r_goal + r_hold + r_success
        - p_pre_touch - p_pre_disturb
        - p_descend - p_stall
        - p_tilt - p_drop - p_table
        - p_arm_vel - p_cup_vel - p_action
    ) + zero  # keeps shape (N,) even if a term ever broadcasts

    components = {
        "pregrasp_geom": r_pregrasp,
        "pregrasp_open_hand": r_open,
        "pregrasp_orient": r_orient,
        "approach_latch": r_latch,
        "pen_pre_touch": -p_pre_touch,
        "pen_pre_disturb": -p_pre_disturb,
        "grasp_digit_mean": r_digit_mean,
        "grasp_digit_last": r_digit_last,
        "grasp_digit_near": r_digit_near,
        "grasp_wrap": r_wrap,
        "grasp_opposition": r_oppose,
        "grasp_curl": r_curl,
        "grasp_palm": r_palm,
        "envelope_latch": r_env_latch,
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
        "pen_cup_vel": -p_cup_vel,
        "pen_action": -p_action,
    }
    return reward, components
