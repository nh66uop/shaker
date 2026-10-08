# Round 5 — cup_pick_l_manip, stage 2 (close → lift → carry → hold)

## 1. What the task means and which stages the robot passes through

The episode always starts with the hand raised beside the robot, so the policy first *travels* to
wherever the cup spawned, arrives in the pre-grasp pose with the default open hand and the start
orientation (this is what the environment latches as `approach_done`), then *closes* a five-digit
envelope with the palm on the cup side and the thumb opposed to the four fingers (`envelope_done`),
then *lifts* the cup straight off the table, *carries* it to `goal_pos` and *holds* it there,
upright and still, so the environment can count repeated successes.

Rounds 1-4 have already established which parts of this are solved:

* the approach and the pre-latch discipline are solved (latch 0.96, no-touch 0.96, cup tilt < 0.6
  degrees, no runaway terminations) — round 4 must not be weakened anywhere;
* the hand does now leave its stage-1 open habit and curls into a claw beside the cup;
* but it **stops one gap short**: the fingertips halt a visible distance from the cup wall, the palm
  stops short too (gap 0.038 → 0.052 m), the thumb stays folded into the palm, contact is 0.037
  digits / 0.011 N, and flexion actually *fell back* (0.164 → 0.121) after epoch 1200.

So the entire problem of this round is the **last centimetre**: the loose claw is currently a
comfortable plateau, and every closing term saturates there.

## 2. What changed relative to round 4, term by term

**Removed / re-gated**

* `pregrasp_open_hand`, `pen_pre_flex`, `pregrasp_orient`, `pen_pre_touch`, `pen_pre_disturb`,
  `pen_squeeze`, `pen_shove`, `pen_arm_fast`: **unchanged**, identical weights and identical
  shaping (worst-joint maximum, direct joint-angle pre-latch penalty). These produced round 4's
  clean approach and I do not touch them.
* `grasp_unfold` (0.35): now **fades out** as the digits get near the cup
  (`fade = 1 - mean(exp(-d_surf/0.03))`). Round 4 showed it saturating at 0.144 and then holding
  the hand at a flexion level that moves fingertips *away* from the cup. Once a digit is near the
  surface, flexion as such stops paying and only distance/contact does (feedback 4).
* `grasp_wrap` (0.50 → 0.40): now multiplied by `0.3 + 0.7 * digit_touch`, so surface-band
  geometry alone is worth at most 30 % of it. It shapes *where* contact happens, it is no longer
  an income for a claw that never lands (feedback 1).
* `grasp_thumb_oppose` (0.50): the proximity factor is tightened from a 0.08 m to a **0.02 m**
  scale, on both the thumb and the four fingers, and the thumb's own factor now uses its distance
  to the *cup surface inside the band*. A thumb tucked into the palm is 4-5 cm from the cup and
  now earns ~0 instead of 0.27 (feedback 3).
* `grasp_palm` (0.40): kept as the *contact* term, still multiplied by digit engagement, so palm
  contact alone can never re-fund the flat open hand of round 2.

**Re-shaped (the core fix)**

* `grasp_digit_mean` (1.00 → **1.50**) and `grasp_digit_last2` (1.60 → **2.00**) use a new
  per-digit ladder whose shares are re-cut:
  `digit_stage = 0.15 * near + 0.50 * first_touch + 0.35 * all_links`.
  Proximity is capped at **0.15** of a digit (it was 0.35) while the first real contact on that
  digit is worth **0.50** on its own. Being near is now worth less than a third of touching, so
  stopping at "near" is a loss rather than a plateau (feedback 1), and the two-weakest-digit term
  now carries most of its value in the contact part (feedback 5): recruiting the index finger is
  worth about +0.66 per step, the largest single move available to the policy.
* `near` itself is a **two-scale** pull, `0.5*exp(-d/0.03) + 0.5*exp(-d/0.008)`, still gated on the
  digit's own flexion. The 0.008 m term gives a steep gradient in exactly the last centimetre where
  round 4 stalled, and because a link cannot pass through the cup, `d → 0` is continuous into
  contact: the path approach → near → touching → all links rises at every step.

**Added**

* `pregrasp_seat` (**0.80**, new): fine-scale (0.02 m) credit for the *tangential* part of the
  gate geometry — the cup axis one radius ahead along `palm_finger_dir`, inside the graspable band,
  with the gate's own dead zone. `pregrasp_geom` uses a 0.25 m tanh whose gradient in the last
  3 cm is worth only ~0.08, which is part of why the whole hand stops short; this term puts real
  money on the last centimetres of the *hand*, and it peaks exactly at the gate, so it helps the
  latch instead of competing with it.
* `grasp_palm_in` (**0.60**, replaces round 4's `grasp_palm_stay`): pays on the palm-to-cup
  distance `d_normal` **alone**, with a dead zone at the gate's own `[-0.005, +0.015] m` so it can
  never drive the palm through the cup, and with **no dependence on the digits** (feedback 2).
  This breaks round 4's circular gating — palm credit needed digits, digit credit needed proximity,
  and the hand sat where neither paid — by always leaving a gradient that moves the whole hand onto
  the cup.
* `envelope_latch` 0.60 → **0.80**.

**Untouched, as instructed (never reached, nothing to conclude)**

`lift_height`, `lift_rise`, `goal_reach`, `goal_hold`, `success_bonus`, `pen_descend`, `pen_stall`,
`pen_tilt`, `pen_drop`, `pen_table`, `pen_arm_vel`, `pen_cup_vel`, `pen_action`, and the `secure`
gate (palm + at least four digits) that stands in front of the lift.

## 3. Per-step income by stage

Typical per-step totals (penalties of a well-behaved episode included; success spike excluded):

| stage | main terms present | income/step |
|---|---|---|
| travelling (far, open hand, unlatched) | pregrasp 0.12, orient 0.15, open 1.20 | **≈ 1.42** |
| arrived, open hand, not yet latched | + seat 0.80, pregrasp 0.67, palm_in 0.60 | **≈ 3.37** |
| latched (`approach_done`) | open −1.20, latch +1.60 | **≈ 3.77** |
| curled and near the cup, NOT touching | + digit_mean 0.14, last2 0.18, wrap 0.06, unfold 0.05 | **≈ 4.20** |
| fingers actually touching (5 digits, one link each) | digit_mean 1.00, last2 1.33, wrap 0.25, thumb_opp 0.40, palm 0.18 | **≈ 6.85** |
| closed envelope (5 digits, all links, palm, latch) | digit_mean 1.50, last2 2.00, wrap 0.32, thumb_opp 0.45, palm 0.40, env_latch 0.80 | **≈ 9.10** |
| lifted (~0.10 m, envelope held) | + lift 1.87, rise 0.30, goal 0.35, shove penalty gone | **≈ 11.5** |
| at the goal, at rest | + lift 2.19, goal 1.83, hold 2.09 | **≈ 15.3** (+10 on each success step) |

**Check 1 — income never decreases:**
1.42 < 3.37 < 3.77 < 4.20 < 6.85 < 9.10 < 11.5 < 15.3. Every transition is a step up and stays up.
The two places where a term is switched off are both covered: the latch removes `open` (1.20) but
adds `latch` (1.60), net +0.40; `unfold` (≤0.35) fades as the digits approach but is replaced many
times over by the proximity and contact ladder.

**Check 2 — "near but not touching" → "actually touching" is a large step, not a small one:**
+2.65 per step (4.20 → 6.85), the largest single jump below the goal. Per digit it is
0.15 → 0.66 of `digit_stage`, i.e. the first contact of one digit is worth 3.4x everything that
digit can earn by hovering, and proximity for all five digits together (0.15 each) is worth less
than one digit landing. The loose-claw plateau of round 4 is therefore strictly unprofitable.

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
    # Ladder of per-step income:
    #   travelling ~1.4 < arrived open hand ~3.4 < latched ~3.8 < curled and near ~4.2
    #   < five digits TOUCHING ~6.9 < closed envelope ~9.1 < lifted ~11.5 < still at goal ~15.3
    # The big step is "near" -> "touching" (+2.6), so hovering beside the cup is never a plateau.
    W_PREGRASP = 0.70        # env pre-grasp geometry, coarse 0.25 m -- PAID BEFORE AND AFTER LATCH
    W_SEAT = 0.80            # NEW: the SAME geometry at a fine 0.02 m scale (tangential part).
                             #      Round 4's hand stopped ~2 cm short because the coarse tanh pays
                             #      only ~0.08 over the last 3 cm. This puts money on that gap.
    W_OPEN = 1.20            # default open hand BEFORE the latch, shaped on the WORST joint
                             #      (the gate tests a maximum, so the reward tests a maximum). KEEP.
    W_ORIENT = 0.15          # start orientation, 40 deg dead zone, paid on both sides. KEEP.
    W_LATCH = 1.60           # approach_done income: > W_OPEN, the term it switches off. KEEP.
    P_PRE_FLEX = 2.00        # flexing before the latch, charged on the joint angles. KEEP.
    P_PRE_TOUCH = 1.20       # touching the cup before the latch. KEEP.
    P_PRE_DISTURB = 0.50     # moving the cup before the latch. KEEP.
    W_PALM_IN = 0.60         # NEW (replaces grasp_palm_stay): palm-to-cup DISTANCE alone, with NO
                             #      dependence on the digits, so there is always a gradient that
                             #      moves the whole hand onto the cup (breaks round 4's circularity)
    W_UNFOLD = 0.35          # leaving the default OPEN posture, joints only -- now FADES OUT once
                             #      the digits are near the cup, handing over to distance/contact
    W_THUMB_OPP = 0.50       # thumb ACROSS the body from the fingers AND close to the surface
    W_DIGIT_MEAN = 1.50      # mean over the 5 digits; every digit weighted the same (0.30 each)
    W_DIGIT_LAST2 = 2.00     # the TWO LOWEST-scoring digits (1.00 each): the last two to arrive
                             #      are worth 1.30 each against 0.30 for the first three
    W_WRAP = 0.40            # links ON the cup surface, flexion-gated AND mostly contact-gated
    W_PALM = 0.40            # palm contact, only together with digits that are closing
    W_ENV_LATCH = 0.80       # the environment's own envelope latch
    W_LIFT = 2.20            # height of the cup above its own spawn height          (unchanged)
    W_RISE = 0.60            # upward cup velocity, only until `lifted` latches       (unchanged)
    W_GOAL = 2.00            # closing on goal_pos                                    (unchanged)
    W_HOLD = 2.60            # at rest at the goal -> worth more than repeating lifts (unchanged)
    W_SUCCESS = 10.0         # bonus on the step the env counts a success             (unchanged)
    P_SQUEEZE = 0.80         # contact force beyond what holding the cup needs. KEEP.
    P_SHOVE = 1.20           # pushing the cup around while closing on the table. KEEP.
    P_DESCEND = 0.40         # lowering the held cup < holding it still < raising it
    P_STALL = 0.15           # closed grasp parked on the table, late in the episode
    P_TILT = 0.60            # keep the cup upright
    P_DROP = 0.50            # lifted and then let go
    P_TABLE = 0.50           # palm / fingers driven into the table top
    P_ARM_VEL = 0.12         # smoothness: tiny next to W_LIFT, a slow lift always wins
    P_ARM_FAST = 0.80        # only speeds above 3 rad/s: runaway only. KEEP.
    P_CUP_VEL = 0.15         # linear cup speed while held
    P_CUP_SPIN = 0.08        # angular cup speed while held
    P_ACT_RATE = 0.02        # action jitter
    P_ARM_ACT = 0.01         # arm action magnitude only (fingers must be free to reach +-1)

    # per-digit ladder shares (sum to 1.0). Proximity is capped at 0.15 of a digit while the FIRST
    # real contact of that digit is worth 0.50 on its own: "near" is worth less than a third of
    # "touching", so the loose claw of round 4 is strictly unprofitable.
    NEAR_SHARE = 0.15
    TOUCH_FIRST = 0.50
    TOUCH_LINKS = 0.35

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
    # Stage A: arrive at the pre-grasp pose with an OPEN hand and latch it   (round 4, kept)
    # =========================================================================================
    # the three geometric quantities the env gate itself tests
    cup_from_palm = ctx.cup_pos - ctx.palm_pos
    d_normal = (cup_from_palm * ctx.palm_normal).sum(-1) - ctx.cup_radius        # gate: [-0.01, 0.02]
    d_finger = (cup_from_palm * ctx.palm_finger_dir).sum(-1) - ctx.cup_radius    # gate: [-0.005, 0.02]
    h_palm = ((ctx.palm_pos - ctx.cup_pos) * ctx.cup_axis).sum(-1)
    d_band = (h_palm.abs() - ctx.cup_half_height).clamp(min=0.0)                 # gate: 0
    pregrasp_err = torch.sqrt(d_normal * d_normal + d_finger * d_finger + d_band * d_band + 1e-12)
    # coarse pull: scale 0.25 m keeps a usable gradient over the whole 0.15-0.49 m spawn range
    pregrasp = 1.0 - torch.tanh(pregrasp_err / 0.25)
    r_pregrasp = W_PREGRASP * pregrasp

    # FIX A (round 4: "the hand stops one gap short"). Fine-scale credit for the TANGENTIAL part of
    # the gate geometry, inside the gate's own dead zone, so the last 2 cm of hand travel is worth
    # real money. It peaks exactly at the gate, so it helps the latch instead of competing with it.
    dz_finger = (d_finger - 0.02).clamp(min=0.0) + (-0.005 - d_finger).clamp(min=0.0)
    seat_err = torch.sqrt(dz_finger * dz_finger + d_band * d_band + 1e-12)
    r_seat = W_SEAT * torch.exp(-seat_err / 0.02)

    # `ready` (round 3 fix, kept): the closing terms are gated on the SAME geometry the gate
    # measures -- flat inside the gate's 2 cm tolerance, 4 cm decay outside -- not on the latch
    # flag, which the policy cannot observe.
    ready = torch.exp(-(pregrasp_err - 0.02).clamp(min=0.0) / 0.04)

    # the gate's pose condition is a MAXIMUM over movable joints, so shape the worst joint
    dev_max = ((ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable).amax(dim=-1)
    r_open = W_OPEN * pre_latch * torch.exp(-dev_max / 0.10)
    # and flexing before the latch is charged directly on the joint angles:
    # tanh(0.3/0.15) = 0.96, i.e. ~full price exactly at the threshold the gate tests
    p_pre_flex = P_PRE_FLEX * pre_latch * torch.tanh(dev_max / 0.15)

    # roughly the start orientation (palm_normal ~ -y, palm_finger_dir ~ +x), 40 deg dead zone
    # because the far corners legitimately need up to ~40 deg of yaw. Paid on both sides.
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
    # surface earns nothing. The thumb needs less flexion to drop onto the near side than the four
    # fingers need to come round the far side.
    flex_ref = torch.tensor([0.20, 0.35, 0.35, 0.35, 0.35], device=device, dtype=dtype)
    flex_gate = (closure / flex_ref).clamp(min=0.0, max=1.0)                                # (N,5)

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

    # FIX B (feedback 4): leaving the deeply reinforced open posture still pays from the first
    # step, but FADES OUT as the digits come near the cup, so late in the close the only way to
    # earn more is to actually reach the surface. Capped at half flexion.
    prox_broad = torch.exp(-digit_surf_dist / 0.03)                                  # (N,5)
    unfold_fade = (1.0 - prox_broad.mean(dim=-1)).clamp(min=0.0)
    r_unfold = W_UNFOLD * ready * unfold_fade * (closure / 0.5).clamp(max=1.0).mean(dim=-1)

    # FIX C (feedback 1 + 5): ONE continuous, strictly increasing per-digit path
    #   far -> curled and near (<= 0.15) -> FIRST contact (+0.50) -> all three links (+0.35).
    # `near` is a two-scale pull: 0.03 m for long-range guidance and 0.008 m for the last
    # centimetre where round 4 stalled. A link cannot pass through the cup, so d -> 0 runs
    # continuously into contact and nothing is ever lost by closing further.
    near = (0.5 * prox_broad + 0.5 * torch.exp(-digit_surf_dist / 0.008)) * flex_gate  # (N,5)
    digit_stage = (NEAR_SHARE * near
                   + TOUCH_FIRST * digit_touch
                   + TOUCH_LINKS * digit_links)                                       # (N,5) [0,1]

    # equal weight per digit: index and ring pay exactly as much as thumb / middle / pinky
    r_digit_mean = W_DIGIT_MEAN * ready * digit_stage.mean(dim=-1)
    # the TWO digits that arrive last, scored on their own, now dominated by the CONTACT part:
    # recruiting the index finger is worth ~+0.66/step, the largest single move available.
    stage_sorted, _ = torch.sort(digit_stage, dim=-1)
    r_digit_last2 = W_DIGIT_LAST2 * ready * 0.5 * (stage_sorted[:, 0] + stage_sorted[:, 1])

    # wrapping ALONG the surface (never towards the axis), for digits that are curling AND mostly
    # only once they touch: band geometry alone is worth at most 30 % of this term now.
    wrap_link = torch.exp(-surf_err / 0.02) * torch.exp(-band_err / 0.02)             # (N,5,3)
    wrap_digit = wrap_link.mean(dim=-1) * flex_gate * (0.3 + 0.7 * digit_touch)       # (N,5)
    r_wrap = W_WRAP * ready * wrap_digit.mean(dim=-1)

    # FIX D (feedback 3): opposition requires the thumb to be ACROSS the cup from the fingers AND
    # close to its surface. The 0.02 m proximity scale (was 0.08 m) means a thumb folded back into
    # the palm, 4-5 cm from the cup, now earns ~0 instead of 0.27.
    tip_radial = radial_vec[:, :, 2, :]                                              # (N,5,3)
    tip_dir = tip_radial / tip_radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    thumb_dir = tip_dir[:, 0, :]                                                     # (N,3)
    fing_mean = tip_dir[:, 1:, :].mean(dim=1)
    fing_dir = fing_mean / fing_mean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    oppose = ((1.0 - (thumb_dir * fing_dir).sum(-1)) * 0.5).clamp(min=0.0, max=1.0)
    prox_tight = torch.exp(-digit_surf_dist / 0.02)                                  # (N,5)
    r_thumb_opp = W_THUMB_OPP * ready * oppose * prox_tight[:, 0] * prox_tight[:, 1:].mean(dim=-1)

    # FIX E (feedback 2): bringing the palm IN pays on the palm-to-cup distance ALONE, with no
    # dependence on the digits, so the circular gating of round 4 (palm needs digits, digits need
    # proximity, hand sits where neither pays) is broken. Dead zone = the gate's own window, so it
    # can never drive the palm through the cup.
    palm_err = (d_normal - 0.015).clamp(min=0.0) + (-0.005 - d_normal).clamp(min=0.0)
    r_palm_in = W_PALM_IN * ready * torch.exp(-palm_err / 0.015)
    # palm CONTACT still only counts together with digits that are closing, so it cannot re-fund
    # the flat open hand pressed against the cup (round 2).
    digit_engage = 0.3 * flex_gate.mean(dim=-1) + 0.7 * digit_touch.mean(dim=-1)
    r_palm = W_PALM * palm_c * digit_engage
    r_env_latch = W_ENV_LATCH * envelope

    # "secure" = palm plus at least four digits (full at five): the envelope the task demands.
    # Unchanged -- the lift has never been reached, so it is not rewritten.
    w_sum = digit_touch.sum(dim=-1)                                                  # 0..5
    secure = ((w_sum - 3.0) / 2.0).clamp(min=0.0, max=1.0) * (0.4 + 0.6 * palm_c)

    # closing must stay restrained (round 3): the contact reward saturates at a few tenths of a
    # newton, so force beyond what holding the cup needs buys nothing and costs.
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
    # Safety and smoothness (all small next to the stage income)   (unchanged)
    # =========================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    p_drop = P_DROP * lifted * (1.0 - secure)
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = P_TABLE * (palm_pen + floor_pen)
    arm_speed = ctx.arm_qd.norm(dim=-1)
    # arm joints are commanded at ~0.3 rad/s, so this stays <= ~0.04: a slow lift always wins
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * secure) * torch.tanh(arm_speed / 1.0)
    # and this one only bites above 3 rad/s, i.e. in the runaway states of round 3
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = secure * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                          + P_CUP_SPIN * torch.tanh(ctx.cup_ang_vel.norm(dim=-1) / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_pregrasp + r_seat + r_orient + r_latch + r_open
        + r_palm_in + r_unfold + r_thumb_opp
        + r_digit_mean + r_digit_last2 + r_wrap
        + r_palm + r_env_latch
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
        "pregrasp_seat": r_seat,
        "pregrasp_open_hand": r_open,
        "pregrasp_orient": r_orient,
        "approach_latch": r_latch,
        "pen_pre_flex": -p_pre_flex,
        "pen_pre_touch": -p_pre_touch,
        "pen_pre_disturb": -p_pre_disturb,
        "grasp_palm_in": r_palm_in,
        "grasp_unfold": r_unfold,
        "grasp_thumb_oppose": r_thumb_opp,
        "grasp_digit_mean": r_digit_mean,
        "grasp_digit_last2": r_digit_last2,
        "grasp_wrap": r_wrap,
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
