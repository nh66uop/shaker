# Stage 2 (manip), round 4 — what the task is and where round 3 broke

## 1. What the task means, stage by stage

The policy inherits a stage-1 habit: fly the open hand from the raised start to the side of the cup and hold it
there. From that pose it must (a) keep passing the environment's `approach_done` gate, (b) close a five-digit
envelope with the palm on the cup body, (c) raise the cup off the table, (d) carry it to `goal_pos` and stand
still there so the environment counts successes.

Round 3 showed the two halves are in conflict under the previous reward: the hand learned to flex, the gate's
"every movable joint within 0.3 of default" condition collapsed at epoch 400 (0.997 -> 0.213), and by epoch 2000
the approach latch itself fell from 0.95 to 0.35. Everything that paid for closing was multiplied by
`approach_done`, so closing killed the gate that funded closing. On top of that the grasp converged to a
fingertip pinch driven by pushing: palm gap 0.031 -> 0.082 m, cup tilt to 12.3 deg, arm joint speed to 11,591 rad/s.

## 2. Changes relative to the round-3 code, term by term

**Removed the hard `approach_done` gate from every closing term (feedback 3).** The policy cannot observe the
latch. All closing terms are now multiplied by `ready`, a purely geometric readiness built from exactly the three
quantities the gate measures (`d_normal`, `d_finger`, `d_band`): flat at 1 inside the gate's own 2 cm tolerance and
decaying with a 4 cm scale outside it. Flexing 10 cm away is now worth almost nothing on its own, flexing at the
cup is worth full price, and the gradient points at the gate instead of depending on which side of it the policy is.

**`pregrasp_open_hand` is now shaped on the WORST movable joint and its weight is raised 0.30 -> 1.20
(feedback 1, 2).** The gate tests a maximum, so the reward now tests a maximum: `exp(-dev_max/0.10)` where
`dev_max = max_j |hand_q_norm - hand_default_q_norm|` over movable joints only. One finger flexing past 0.3 now
costs almost the whole term, exactly as it costs the whole gate.

**New `pen_pre_flex` (weight 2.00), a direct penalty on joint angles before the latch (feedback 1).** It charges
`tanh(dev_max/0.15)`, i.e. ~0.96 of full price at the gate's 0.3 threshold. Together with the 1.20 open-posture
term, holding the hand open until the latch is now worth about 3.2 per step against the ~4.95 that closing pays
afterwards — the same order of magnitude instead of one twenty-fifth — and closing early is additionally charged
the 1.20 pre-touch penalty, so arriving open dominates by a wide margin (numbers in section 3).

**`approach_latch` raised 0.90 -> 1.60** so it still strictly exceeds the (now much larger) open-posture term it
switches off. **`pregrasp_orient` is now paid before and after the latch** (one hard switch fewer; the 40 degree
dead zone means a grasped, lifted cup keeps collecting it).

**New `grasp_palm_stay` (weight 0.60), distance-based (feedback 5).** Palm contact alone never rose above 0.16
last round while the gap grew to 0.082 m, so the binary contact term carried no gradient back to the palm. This
term pays `exp(-|d_normal|/0.02) * digit_engage` continuously through closing — the palm is pulled back onto the
cup by distance, not by a contact event, and it still cannot fund a flat open hand because it is multiplied by
digit engagement. `grasp_palm` (binary contact, 0.40) is kept, also multiplied by `digit_engage`.

**`grasp_digit_last` -> `grasp_digit_last2` (feedback 6).** Instead of only the minimum over digits, the reward now
sorts the five per-digit scores and pays the mean of the two lowest with weight 1.60. With `grasp_digit_mean` at
1.00 (0.20 per digit), the two digits that arrive last are worth 1.00 each against 0.20 for the three that arrive
first — the index finger (0.002-0.011 for 2,868 epochs) is now by far the most profitable digit to recruit.

**New `pen_squeeze` (0.80) and `pen_shove` (1.20) (feedback 4).** `pen_squeeze` charges total and worst-link
contact force above 15 N / 6 N; the contact reward itself saturates at a few tenths of a newton, so extra force
buys nothing and costs. `pen_shove` charges horizontal cup displacement, horizontal cup speed and tilt while the
cup is still on the table (`on_table = 1 - tanh(height/0.02)`, so it vanishes as soon as a real lift starts). Its
full value 1.20 exceeds the ~1.00 that recruiting one more digit buys, so shoving the cup into the fingers is a
net loss. **New `pen_arm_fast` (0.80)** charges only joint speeds above 3 rad/s — unreachable by the 0.3 rad/s
command rate, so it is free for any legitimate motion and only bites in the runaway states seen last round.

**Kept unchanged:** flexion-gated proximity and wrap (`flex_gate`), the single continuous per-digit ladder
(`NEAR_SHARE 0.35` / `TOUCH_SHARE 0.65`), `grasp_unfold`, `grasp_thumb_oppose`, `pen_pre_touch`, and the whole
lift / goal / hold / success block with its `secure` gate (palm plus at least four digits). Those were never
reached in a stable state, so per the caution they are not rewritten; `pen_pre_disturb` is lowered 0.80 -> 0.50
only because `pen_shove` now covers the same quantities more broadly.

## 3. Per-step income by stage (and it never decreases)

| stage | terms paid | total |
|---|---|---|
| travelling, open hand, far from cup | pregrasp 0.11-0.55 + orient 0.15 + open 1.20 | **1.46 -> 1.90** |
| arrived, open hand, not yet latched | pregrasp 0.70 + orient 0.15 + open 1.20 | **2.05** |
| latched, hand still open | pregrasp 0.70 + orient 0.15 + latch 1.60 + closing ~0 | **2.45** |
| latched, five digits curled and near, no contact | + unfold 0.35, thumb 0.50, digit_mean 0.35, last2 0.56, wrap 0.25, palm_stay 0.18, palm 0.12 | **4.76** |
| closed five-digit envelope, palm on cup | + digit_mean 1.00, last2 1.60, wrap 0.40, palm_stay 0.60, palm 0.40, env_latch 0.60 | **7.90** |
| lifted to goal height, still secure | + lift 2.20 (rise 0.60 transient), shove/stall penalties gone | **~9.80** |
| at rest at the goal | + goal 2.00 + hold 2.60 | **~14.70** (+10.0 on each success) |

1.90 < 2.05 < 2.45 < 4.76 < 7.90 < 9.80 < 14.70: every transition is a step up that stays up. Closing before the
latch, at the cup, is worth 0.70 + 0.15 + 4.95 - 2.00 (pre_flex) - 1.20 (pre_touch) - 0.00 (open lost) = 2.60
against 7.90 for latching first and then closing, so the early-closing strategy that destroyed round 3 is now
strictly dominated. Lowering the held cup costs `pen_descend` plus the lift term itself; putting it down after the
goal cannot re-earn `lift_rise`, which only pays while `lifted` is False; and a closed grasp parked on the table
pays 7.90 against 14.70 at the goal, with `pen_stall` growing over the episode.

```python
import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
    """0 when a link is off the cup, ->1 once it presses with a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate the sum and extra force buys nothing."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.cup_pos.device
    dtype = ctx.cup_pos.dtype

    # ---- weights ---------------------------------------------------------------------------
    # Ladder of per-step ceilings:
    #   open hand travelling ~1.9  <  open hand at the cup ~2.05  <  latched ~2.45
    #   <  five digits curled and near ~4.8  <  closed five-digit envelope ~7.9
    #   <  lifted ~9.8  <  still at the goal ~14.7 (+ success spike)
    W_PREGRASP = 0.70        # env pre-grasp geometry -- PAID BEFORE *AND AFTER* THE LATCH
    W_OPEN = 1.20            # default open hand BEFORE the latch, shaped on the WORST joint
                             #   (the gate tests a maximum, so the reward tests a maximum).
                             #   Raised 0.30 -> 1.20: holding open until the latch must be worth
                             #   roughly what closing pays afterwards.
    W_ORIENT = 0.15          # start orientation, 40 deg dead zone, now paid on BOTH sides
    W_LATCH = 1.60           # approach_done income: > W_OPEN, the term it switches off
    P_PRE_FLEX = 2.00        # NEW: flexing the fingers before the latch, charged directly on the
                             #   joint angles (worst joint), not only when it produces contact
    P_PRE_TOUCH = 1.20       # touching the cup before the latch (round-1 exploit; keep)
    P_PRE_DISTURB = 0.50     # moving the cup before the latch (pen_shove now covers most of this)
    W_UNFOLD = 0.35          # leaving the default OPEN posture, joints only: the gradient out of
                             #   the stage-1 habit. Small: closing onto nothing must never beat
                             #   closing onto the cup.
    W_THUMB_OPP = 0.50       # thumb across the body from the fingers, paid WITHOUT contact
    W_DIGIT_MEAN = 1.00      # mean over the 5 digits; every digit weighted the same (0.20 each)
    W_DIGIT_LAST2 = 1.60     # the TWO LOWEST-scoring digits (0.80 each): the two that arrive last
                             #   are worth 1.00 each against 0.20 for the first three
    W_WRAP = 0.50            # links ON the cup surface inside the band, gated on real flexion
    W_PALM_STAY = 0.60       # NEW: palm-to-cup DISTANCE through the whole close (not just contact)
    W_PALM = 0.40            # palm contact, only together with digits that are closing
    W_ENV_LATCH = 0.60       # the environment's own envelope latch
    W_LIFT = 2.20            # height of the cup above its own spawn height
    W_RISE = 0.60            # upward cup velocity, only until `lifted` latches
    W_GOAL = 2.00            # closing on goal_pos
    W_HOLD = 2.60            # at rest at the goal -> worth more than repeating lifts
    W_SUCCESS = 10.0         # bonus on the step the env counts a success
    P_SQUEEZE = 0.80         # NEW: contact force beyond what holding the cup needs
    P_SHOVE = 1.20           # NEW: pushing the cup around while closing on the table. Full value
                             #   > the ~1.00 that recruiting one more digit buys.
    P_DESCEND = 0.40         # lowering the held cup < holding it still < raising it
    P_STALL = 0.15           # closed grasp parked on the table, late in the episode
    P_TILT = 0.60            # keep the cup upright
    P_DROP = 0.50            # lifted and then let go
    P_TABLE = 0.50           # palm / fingers driven into the table top
    P_ARM_VEL = 0.12         # smoothness: tiny next to W_LIFT, a slow lift always wins
    P_ARM_FAST = 0.80        # NEW: only speeds above 3 rad/s (command rate is ~0.3): runaway only
    P_CUP_VEL = 0.15         # linear cup speed while held
    P_CUP_SPIN = 0.08        # angular cup speed while held
    P_ACT_RATE = 0.02        # action jitter
    P_ARM_ACT = 0.01         # arm action magnitude only (fingers must be free to reach +-1)

    # per-digit ladder split: "curled and near the surface" is worth at most 0.35 of a digit,
    # real contact carries the other 0.65, so stopping at "near" is a loss, not a plateau.
    NEAR_SHARE = 0.35
    TOUCH_SHARE = 0.65

    zero = torch.zeros_like(ctx.cup_radius)

    # ---- stage flags as floats ---------------------------------------------------------------
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

    # =========================================================================================
    # Stage A: arrive at the pre-grasp pose with an OPEN hand and latch it
    # =========================================================================================
    # the three geometric quantities the env gate itself tests
    cup_from_palm = ctx.cup_pos - ctx.palm_pos
    d_normal = (cup_from_palm * ctx.palm_normal).sum(-1) - ctx.cup_radius        # -> 0
    d_finger = (cup_from_palm * ctx.palm_finger_dir).sum(-1) - ctx.cup_radius    # -> 0
    h_palm = ((ctx.palm_pos - ctx.cup_pos) * ctx.cup_axis).sum(-1)
    d_band = (h_palm.abs() - ctx.cup_half_height).clamp(min=0.0)                 # -> 0
    pregrasp_err = torch.sqrt(d_normal * d_normal + d_finger * d_finger + d_band * d_band + 1e-12)
    # scale 0.25 m keeps a usable gradient over the whole 0.15-0.49 m spawn range
    pregrasp = 1.0 - torch.tanh(pregrasp_err / 0.25)
    # same weight after the latch as before it: latching never removes income, and a closed
    # envelope satisfies this geometry too, so it keeps paying all the way to the goal.
    r_pregrasp = W_PREGRASP * pregrasp

    # FIX (feedback 3): `ready` replaces `approach_done` as the gate of every closing term. It is
    # built from the SAME geometry the env gate measures, flat inside the gate's own 2 cm
    # tolerance and decaying with a 4 cm scale outside, so the closing gradient points at the gate
    # instead of depending on a latch the policy cannot observe.
    ready = torch.exp(-(pregrasp_err - 0.02).clamp(min=0.0) / 0.04)

    # FIX (feedback 2): the gate's pose condition is a MAXIMUM over movable joints, so shape the
    # worst joint, not the average. One finger past 0.3 breaks the gate and now loses the term.
    dev_max = ((ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable).amax(dim=-1)
    r_open = W_OPEN * pre_latch * torch.exp(-dev_max / 0.10)
    # FIX (feedback 1): and flexing before the latch is charged directly on the joint angles.
    # tanh(0.3/0.15) = 0.96, i.e. ~full price exactly at the threshold the gate tests.
    p_pre_flex = P_PRE_FLEX * pre_latch * torch.tanh(dev_max / 0.15)

    # the gate demands roughly the start orientation (palm_normal ~ -y, palm_finger_dir ~ +x).
    # Dead zone out to 40 deg: the far corners legitimately need up to ~40 deg of yaw. Paid on
    # both sides of the latch now (one hard switch fewer; a held cup keeps collecting it).
    minus_y = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)
    plus_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    cos40 = math.cos(math.radians(40.0))
    cos_n = (ctx.palm_normal * minus_y).sum(-1)
    cos_f = (ctx.palm_finger_dir * plus_x).sum(-1)
    orient_err = (cos40 - cos_n).clamp(min=0.0) + (cos40 - cos_f).clamp(min=0.0)
    r_orient = W_ORIENT * torch.exp(-orient_err / 0.25)

    # latch bonus > the open-posture term it switches off: the latch step is strictly a step up
    r_latch = W_LATCH * approach

    # ---- contacts (shared by several stages) --------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5) does this digit touch
    digit_links = link_c.mean(dim=-1)                              # (N,5) how many of its links
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    any_touch = torch.maximum(digit_touch.amax(dim=-1), palm_c)

    # before the latch: any contact costs more than pressing in could buy, and the cup must be
    # left where it spawned.
    p_pre_touch = P_PRE_TOUCH * pre_latch * any_touch
    cup_disp = (ctx.cup_pos - ctx.cup_spawn_pos).norm(dim=-1)
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    disturb = (torch.tanh(cup_disp / 0.01)
               + torch.tanh(cup_speed / 0.05)
               + torch.tanh(ctx.cup_tilt / math.radians(5.0))) / 3.0
    p_pre_disturb = P_PRE_DISTURB * pre_latch * disturb

    # =========================================================================================
    # Stage B: close a five-digit envelope (gated on `ready`, i.e. on geometry, not on the latch)
    # =========================================================================================
    # ---- how far each digit has CLOSED, relative to the default open pose -------------------
    # hand_q_norm is 0 = straight, 1 = most flexed for every joint, so the advance beyond the
    # default pose is a direct, contact-free measure of flexion.
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)

    # geometric credit requires the digit itself to be closing: an extended finger skimming the
    # surface earns nothing. Thumb needs less flexion to drop onto the near side than the four
    # fingers need to come round the far side.
    flex_ref = torch.tensor([0.20, 0.35, 0.35, 0.35, 0.35], device=device, dtype=dtype)
    flex_gate = (closure / flex_ref).clamp(min=0.0, max=1.0)                                # (N,5)

    # pay for leaving the deeply reinforced open posture as such, on the joints alone. Capped at
    # half flexion so it cannot drive the fingers through the cup, and small next to contact.
    r_unfold = W_UNFOLD * ready * (closure / 0.5).clamp(max=1.0).mean(dim=-1)

    # ---- cylinder geometry of every measured finger link ------------------------------------
    link_vec = ctx.link_pos - ctx.cup_pos.view(-1, 1, 1, 3)        # (N,5,3,3)
    axis = ctx.cup_axis.view(-1, 1, 1, 3)
    axial = (link_vec * axis).sum(-1)                              # (N,5,3)
    radial_vec = link_vec - axial.unsqueeze(-1) * axis             # (N,5,3,3)
    radial = radial_vec.norm(dim=-1)                               # (N,5,3)
    surf_err = (radial - ctx.cup_radius.view(-1, 1, 1)).abs()      # peak ON the surface
    band_err = (axial.abs() - ctx.cup_half_height.view(-1, 1, 1)).clamp(min=0.0)
    link_surf_dist = torch.sqrt(surf_err * surf_err + band_err * band_err + 1e-12)   # (N,5,3)
    digit_surf_dist = link_surf_dist.amin(dim=-1)                                    # (N,5)

    # ONE continuous, strictly increasing path per digit:
    #   far -> curled and near the surface (at most NEAR_SHARE) -> one link touching
    #   -> all three links touching (TOUCH_SHARE). Contact keeps `near` at its maximum, so the
    #   sum only ever rises, and "near" alone is worth barely a third of "touching".
    near = torch.exp(-digit_surf_dist / 0.03) * flex_gate                            # (N,5)
    contact = 0.5 * digit_touch + 0.5 * digit_links                                  # (N,5)
    digit_stage = NEAR_SHARE * near + TOUCH_SHARE * contact                          # (N,5) in [0,1]

    # equal weight per digit: index and ring pay exactly as much as thumb / middle / pinky
    r_digit_mean = W_DIGIT_MEAN * ready * digit_stage.mean(dim=-1)
    # FIX (feedback 6): the TWO digits that arrive last, scored on their own. With W_DIGIT_MEAN
    # spreading 0.20 per digit, the last two are worth 1.00 each against 0.20 for the first
    # three, so recruiting the index finger is the most profitable single move available.
    stage_sorted, _ = torch.sort(digit_stage, dim=-1)
    r_digit_last2 = W_DIGIT_LAST2 * ready * 0.5 * (stage_sorted[:, 0] + stage_sorted[:, 1])

    # wrapping ALONG the surface (never towards the axis), and only for digits that are curling
    wrap_link = torch.exp(-surf_err / 0.02) * torch.exp(-band_err / 0.02)             # (N,5,3)
    r_wrap = W_WRAP * ready * (wrap_link.mean(dim=-1) * flex_gate).mean(dim=-1)

    # the thumb must sit ACROSS the body from the fingers, paid before any contact. Frame-free:
    # compare radial directions about the cup axis.
    tip_radial = radial_vec[:, :, 2, :]                                              # (N,5,3)
    tip_dir = tip_radial / tip_radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    thumb_dir = tip_dir[:, 0, :]                                                     # (N,3)
    fing_mean = tip_dir[:, 1:, :].mean(dim=1)
    fing_dir = fing_mean / fing_mean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    oppose = ((1.0 - (thumb_dir * fing_dir).sum(-1)) * 0.5).clamp(min=0.0, max=1.0)
    prox_soft = torch.exp(-digit_surf_dist / 0.08)                                   # (N,5)
    r_thumb_opp = W_THUMB_OPP * ready * oppose * prox_soft[:, 0] * prox_soft[:, 1:].mean(dim=-1)

    # palm income never pays on its own: it is multiplied by digits that are closing / touching,
    # so it can no longer fund a flat open hand pressed against the cup.
    digit_engage = 0.3 * flex_gate.mean(dim=-1) + 0.7 * digit_touch.mean(dim=-1)
    # FIX (feedback 5): keep the palm on the cup by DISTANCE (|d_normal| is the gate's own
    # palm-plane-to-cup-surface gap, which grew 0.031 -> 0.082 m last round while the fingers
    # engaged). Continuous through the whole close, unlike the binary contact term.
    palm_close = torch.exp(-d_normal.abs() / 0.02)
    r_palm_stay = W_PALM_STAY * palm_close * digit_engage
    r_palm = W_PALM * palm_c * digit_engage
    r_env_latch = W_ENV_LATCH * envelope

    # "secure" = palm plus at least four digits (full at five): the envelope the task demands.
    # Unchanged -- the lift was never reached in a stable state, so it is not rewritten.
    w_sum = digit_touch.sum(dim=-1)                                                  # 0..5
    secure = ((w_sum - 3.0) / 2.0).clamp(min=0.0, max=1.0) * (0.4 + 0.6 * palm_c)

    # FIX (feedback 4): closing must be restrained. The contact reward saturates at a few tenths
    # of a newton, so force beyond what holding the cup needs buys nothing and now costs.
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - 15.0).clamp(min=0.0) / 15.0)
                                   + torch.tanh((max_link_force - 6.0).clamp(min=0.0) / 6.0))

    # and the cup must not be shoved around while the hand closes on it. Fades out as soon as a
    # real lift starts, so it can never tax the lift itself.
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    on_table = 1.0 - torch.tanh(height / 0.02)
    horiz_disp = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    shove = (torch.tanh(horiz_disp / 0.015)
             + torch.tanh(horiz_speed / 0.05)
             + torch.tanh(ctx.cup_tilt / math.radians(10.0))) / 3.0
    p_shove = P_SHOVE * on_table * shove

    # =========================================================================================
    # Stage C: lift   (unchanged: still gated on a real envelope, not on bare contact)
    # =========================================================================================
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

    # =========================================================================================
    # Stage D: carry to the goal and hold still   (unchanged)
    # =========================================================================================
    r_goal = W_GOAL * secure * (1.0 - torch.tanh(ctx.goal_dist / 0.12))
    near_goal = 1.0 - torch.tanh(ctx.goal_dist / 0.05)             # fixed scale, not success_tol
    still = torch.exp(-cup_speed / 0.08)
    r_hold = W_HOLD * secure * near_goal * still                   # resting at the goal > relifting
    r_success = W_SUCCESS * success

    # =========================================================================================
    # Safety and smoothness (all small next to the stage income)
    # =========================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    p_drop = P_DROP * lifted * (1.0 - secure)
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = P_TABLE * (palm_pen + floor_pen)
    arm_speed = ctx.arm_qd.norm(dim=-1)
    # arm joints are commanded at ~0.3 rad/s, so this stays <= ~0.04: a slow lift always wins
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * secure) * torch.tanh(arm_speed / 1.0)
    # and this one only bites above 3 rad/s, i.e. in the runaway states of the previous round
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = secure * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                          + P_CUP_SPIN * torch.tanh(ctx.cup_ang_vel.norm(dim=-1) / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_pregrasp + r_open + r_orient + r_latch
        + r_unfold + r_thumb_opp
        + r_digit_mean + r_digit_last2 + r_wrap
        + r_palm_stay + r_palm + r_env_latch
        + r_lift + r_rise
        + r_goal + r_hold + r_success
        - p_pre_flex - p_pre_touch - p_pre_disturb
        - p_squeeze - p_shove
        - p_descend - p_stall
        - p_tilt - p_drop - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero  # keeps shape (N,) even if a term ever broadcasts

    components = {
        "pregrasp_geom": r_pregrasp,
        "pregrasp_open_hand": r_open,
        "pregrasp_orient": r_orient,
        "approach_latch": r_latch,
        "pen_pre_flex": -p_pre_flex,
        "pen_pre_touch": -p_pre_touch,
        "pen_pre_disturb": -p_pre_disturb,
        "grasp_unfold": r_unfold,
        "grasp_thumb_oppose": r_thumb_opp,
        "grasp_digit_mean": r_digit_mean,
        "grasp_digit_last2": r_digit_last2,
        "grasp_wrap": r_wrap,
        "grasp_palm_stay": r_palm_stay,
        "grasp_palm": r_palm,
        "envelope_latch": r_env_latch,
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
```
