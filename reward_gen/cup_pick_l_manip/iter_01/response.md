# 1. What the task means, and the stages the robot must go through

## What is being asked

This reward is for **stage 2 only** of a two-stage pick-and-hold task, applied on top of weights that
already know how to fly the hand from the raised start pose to the pre-grasp pose beside the cup.
So every episode still contains the travel phase, but the travel phase is *solved*: it must keep
being paid (otherwise the warm-started behaviour is destroyed by the new reward), yet it must not be
where the return sits, or the policy will park in the pre-grasp pose and collect that income forever.

The rest of the episode is what has to be learned: close a **five-digit envelope** around the cup,
**lift** it off the table, **carry** it to `goal_pos` and **hold it there, upright and still**, so the
environment keeps latching successes until `max_successes` ends the episode.

Two hard constraints from the environment shape the design:

* `approach_done` requires the hand to still be in the **default open pose** (every movable joint
  within 0.3 of `hand_default_q_norm`) and **not touching the cup**. So any reward for closing or for
  contact that is paid *before* the latch actively fights the gate: the policy would squeeze early,
  never latch `approach_done`, and therefore never be able to latch `envelope_done`. All contact and
  wrap terms are gated on `approach_done`.
* A success requires `goal_dist <= success_tol` for `success_hold_steps` consecutive steps **and**
  the fingers wrapped around the cup on that step, with the wrap threshold rising during training.
  So the grasp quality terms must stay paid all the way through the carry and hold — they are not a
  phase that can be dropped after the lift.

## Stages

**Stage A — travel to the pre-grasp pose (already solved, keep it alive).**
From the raised start, 0.15–0.49 m from a cup whose position is random, bring the palm to the cup's
side with the hand open and in roughly the start orientation, until the environment latches
`approach_done`. Reward: a bounded distance shaping to the pre-grasp point
(`cup_pos - r*(palm_normal + palm_finger_dir)`, i.e. the cup one radius ahead of the palm plane and
one radius ahead along the finger direction, which is exactly the gate's geometry), plus a term that
keeps the hand at `hand_default_q_norm`, both switched off the moment the latch is set, plus a small
constant latch income. Total ceiling ≈ 0.4–1.1 per step: enough to hold the behaviour, far below
what the later stages pay. No orientation term is added — the gate already allows ±45°, and the far
corners of the cup region genuinely need up to ~40° of rotation, so paying for the start orientation
would fight reachability.

**Stage B — close the envelope (the part that failed last time).**
Palm against the side of the cup, thumb on the near side, four fingers around the far side, every
digit's links on the cup body. The previous attempt ended with thumb + middle + pinky in contact and
the **ring (and index) never arriving**, which is exactly what an "average contact force" reward
produces: three digits pressing hard score the same as five digits touching. The fix here is three
separate terms:

* `grasp_contact`: the **mean over the five digits** of a per-digit contact score, every digit with
  the *same* weight (index and ring pay exactly as much as thumb, middle, pinky), and the per-digit
  score itself is `0.5*max_link + 0.5*mean_link`, so a digit that gets a second and third link onto
  the cup still keeps improving instead of saturating at first touch. This part is linear in the
  number of digits: 3 digits ≈ 0.6 of the term, 5 digits = 1.0.
* `grasp_last_digit`: the **minimum over the five digits**. This is zero until the *last* digit
  arrives, which makes the fifth digit worth several times the fourth and removes the three-digit
  local optimum entirely.
* `grasp_wrap` + `grasp_opposition`: geometry, not force. `grasp_wrap` pays each finger link for
  sitting **on the cup surface** — `exp(-|radial_distance - cup_radius|/0.02)` — and inside the
  graspable band, so the reward peaks *at* the surface and falls off again if the link moves toward
  the axis; there is no incentive to press inward. `grasp_opposition` measures how far the contacts
  surround the cup: the contact-weighted mean of the fingertips' unit radial directions has small
  norm when the digits are distributed around the body and norm 1 when they are all bunched on one
  side. That is a frame-free encirclement measure, so it does not assume where the thumb ends up.

Plus a palm-contact term and a small income on the environment's own `envelope_done` latch.
Ceiling with a full five-digit envelope ≈ 2.9 per step.

**Stage C — lift.** With the envelope closed, raise the cup straight up. Paid as
`tanh(height / (2*lift_latch_height))` on `cup_pos[:,2] - cup_spawn_pos[:,2]`, so the reward rises
with height and is worth ~0 while the cup is on the table: *holding the closed grasp on the table is
not worth staying for*. It is multiplied by a `held` factor built from the digit contacts, so the cup
cannot be scooped or knocked upward for credit. A separate `lift_rise` term pays upward cup velocity
**only while `lifted` is still False**, which makes the first lift fast to discover and, because the
flag latches for the episode, makes it impossible to farm by repeated up-down cycling. A
`pen_descend` term charges for downward cup velocity whenever the cup is held. Descending therefore
loses height reward *and* pays a penalty, holding still keeps the height reward, and raising gains —
strictly ordered as required. A small `pen_stall`, scaled by `episode_progress` and active only while
held-but-never-lifted, removes the last bit of value from sitting on the table with a closed hand,
without punishing the first moments of the close.

**Stage D — carry and hold at the goal.** `goal_reach` pays `1 - tanh(goal_dist/0.12)` and `goal_hold`
pays a sharp proximity (`1 - tanh(goal_dist/0.05)`) multiplied by a stillness factor
`exp(-|cup_lin_vel|/0.08)`, both gated by `held`. Coming to rest at the goal is therefore worth ~4.5
per step on top of the lift and grasp income (~9.5 total), which is much more than cycling lifts, and
because the grasp terms stay gated on contact and the lift term on current height, putting the cup
back down loses everything rather than re-opening an income stream. `ctx.success` adds a bonus of 10.

**Throughout.** Tilt penalty (`tanh(cup_tilt/30°)`), a drop penalty that fires when the cup has been
lifted and the digits are no longer on it, a table penalty built from `palm_clearance` and
`hand_z_min - table_z` so neither the palm nor the fingers are driven into the table, and small
smoothness penalties on arm joint velocity, cup linear/angular velocity while held, and action rate.
The smoothness weights are deliberately ~0.05–0.15 against a lift term of 2.0: with arm joints capped
near 0.3 rad/s these terms can never make "no lift" better than "slow lift".

**Per-step ladder** (approximate ceilings, strictly increasing):
approach only ≈ 0.4 → three digits on the cup ≈ 1.6 → full five-digit envelope on the table ≈ 3.3 →
lifted 0.25 m ≈ 5.3 → at rest at the goal ≈ 9.8 → plus 10 on each counted success.

# 2. Reward function

```python
import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
    """0 when a link is off the cup, ->1 once it presses with a few tenths of a newton.
    Saturating so that force spikes from the solver cannot dominate the sum."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.cup_pos.device
    dtype = ctx.cup_pos.dtype

    # ---- weights -------------------------------------------------------------------------
    # Stage A is already solved by the warm start: it is kept alive but capped near 1.1/step,
    # while a closed envelope pays ~2.9 and a still grasp at the goal ~9.8. The ladder is
    # strictly increasing so no stage is a place to park.
    W_APPROACH_SHAPE = 0.8    # pre-grasp distance + open-hand shaping, only before the latch
    W_APPROACH_LATCH = 0.4    # small constant income for keeping approach_done set
    W_CONTACT = 1.0           # mean over the 5 digits, every digit weighted the SAME
    W_LAST_DIGIT = 0.8        # min over the 5 digits: only the last digit to arrive earns it
    W_WRAP = 0.8              # links on the cup SURFACE inside the graspable band
    W_OPPOSE = 0.5            # contacts distributed around the body, not bunched on one side
    W_PALM = 0.3              # palm against the cup side (envelope_done needs it)
    W_ENV_LATCH = 0.3         # the environment's own envelope latch
    W_LIFT = 2.0              # height of the cup above its spawn height
    W_RISE = 0.6              # upward cup velocity, only until `lifted` latches
    W_GOAL = 2.0              # closing on goal_pos
    W_HOLD = 2.5              # at rest at the goal -> more than repeating lifts
    W_SUCCESS = 10.0          # bonus on the step the env counts a success
    W_DESCEND = 0.4           # moving the held cup down is worse than keeping it still
    W_STALL = 0.15            # closed grasp parked on the table late in the episode
    W_TILT = 0.6              # keep the cup upright
    W_DROP = 0.5              # lifted and then let go
    W_TABLE = 0.5             # palm / fingers driven into the table top
    W_ARM_VEL = 0.12          # smoothness: tiny next to W_LIFT, a slow lift always wins
    W_CUP_VEL = 0.15          # linear cup speed while held
    W_CUP_SPIN = 0.08         # angular cup speed while held
    W_ACT_RATE = 0.02         # action jitter
    W_ARM_ACT = 0.01          # arm action magnitude only (fingers must be free to reach +-1)

    # ---- stage flags as floats -------------------------------------------------------------
    approach = ctx.approach_done.to(dtype)
    envelope = ctx.envelope_done.to(dtype)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    before_approach = 1.0 - approach

    # movable finger joints (ranges <= 0.05 rad are locked and their normalisation is noise):
    # locked = thumb_2(0), index_1(3), middle_1(7), ring_1(11), pinky_1(15), pinky_2(16)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )

    # =======================================================================================
    # Stage A: keep the solved approach alive (bounded, and switched off once latched)
    # =======================================================================================
    # Pre-grasp geometry of the env's own gate: the cup axis one radius ahead of the palm plane
    # (along palm_normal) and one radius ahead along palm_finger_dir.
    cup_from_palm = ctx.cup_pos - ctx.palm_pos
    pregrasp_offset = ctx.cup_radius.unsqueeze(-1) * (ctx.palm_normal + ctx.palm_finger_dir)
    pregrasp_err = (cup_from_palm - pregrasp_offset).norm(dim=-1)
    # scale 0.25 m keeps a usable gradient over the whole 0.15-0.49 m spawn range
    reach = 1.0 - torch.tanh(pregrasp_err / 0.25)
    # the gate also demands the default OPEN hand, so pay for it until the latch is set
    open_err = ((ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable).sum(-1) / movable.sum()
    open_hand = torch.exp(-open_err / 0.15)
    r_approach_shape = W_APPROACH_SHAPE * before_approach * (0.7 * reach + 0.3 * open_hand)
    r_approach_latch = W_APPROACH_LATCH * approach

    # =======================================================================================
    # Stage B: close a five digit envelope (gated on approach_done so it cannot pre-empt the gate)
    # =======================================================================================
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.max(dim=-1).values                        # (N,5) does this digit touch at all
    digit_links = link_c.mean(dim=-1)                              # (N,5) how many of its links touch
    # per digit score keeps rising as the 2nd and 3rd link of that digit arrive
    digit_score = 0.5 * digit_touch + 0.5 * digit_links            # (N,5) in [0,1]
    # equal weight per digit: index and ring pay exactly as much as thumb/middle/pinky
    contact_mean = digit_score.mean(dim=-1)                        # linear in the digit count
    # the weakest digit: 0 until the LAST digit lands, so 5 digits >> 4 digits >> 3 digits
    contact_last = digit_score.min(dim=-1).values
    palm_c = _soft_contact(ctx.palm_cup_force)

    # cylinder geometry of every measured finger link
    link_vec = ctx.link_pos - ctx.cup_pos.view(-1, 1, 1, 3)        # (N,5,3,3)
    axis = ctx.cup_axis.view(-1, 1, 1, 3)
    axial = (link_vec * axis).sum(-1)                              # (N,5,3)
    radial_vec = link_vec - axial.unsqueeze(-1) * axis             # (N,5,3,3)
    radial = radial_vec.norm(dim=-1)                               # (N,5,3)
    # peak ON the surface: moving further inward (towards the axis) LOSES reward, so this pays
    # for wrapping along the cup body, never for squeezing towards the centre
    surf_err = (radial - ctx.cup_radius.view(-1, 1, 1)).abs()
    band_err = (axial.abs() - ctx.cup_half_height.view(-1, 1, 1)).clamp(min=0.0)
    wrap_link = torch.exp(-surf_err / 0.02) * torch.exp(-band_err / 0.02)
    wrap = wrap_link.mean(dim=-1).mean(dim=-1)                     # equal weight per digit and link

    # encirclement: contact weighted mean of the fingertip radial directions. Norm ~1 when every
    # contact sits on one side, ~0 when the digits surround the body. Frame free, so it makes no
    # assumption about where the thumb ends up.
    tip_radial = radial_vec[:, :, 2, :]                            # (N,5,3)
    tip_dir = tip_radial / tip_radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    w_sum = digit_touch.sum(dim=-1)                                # 0..5
    mean_dir = (digit_touch.unsqueeze(-1) * tip_dir).sum(dim=1) / w_sum.clamp(min=1e-6).unsqueeze(-1)
    opposition = (1.0 - mean_dir.norm(dim=-1)).clamp(min=0.0) * (w_sum / 5.0)

    r_contact = W_CONTACT * approach * contact_mean
    r_last_digit = W_LAST_DIGIT * approach * contact_last
    r_wrap = W_WRAP * approach * wrap
    r_oppose = W_OPPOSE * approach * opposition
    r_palm = W_PALM * approach * palm_c
    r_env_latch = W_ENV_LATCH * envelope

    # "held": a real multi digit grasp. Gates everything that follows so the cup cannot be
    # scooped, batted or balanced to earn the lift and goal income.
    held = (digit_touch.sum(dim=-1) / 3.0).clamp(max=1.0)

    # =======================================================================================
    # Stage C: lift
    # =======================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    h_scale = max(2.0 * ctx.lift_latch_height, 0.02)               # ~0 on the table, ~1 near the goal
    r_lift = W_LIFT * held * torch.tanh(height / h_scale)
    vz = ctx.cup_lin_vel[:, 2]
    # upward speed pays only until the lift latches: it teaches the first lift fast and can never
    # be farmed by putting the cup down and raising it again
    r_rise = W_RISE * held * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.15)
    # going down while holding costs, so descending < staying still < rising
    p_descend = W_DESCEND * held * torch.tanh(vz.clamp(max=0.0).abs() / 0.15)
    # a closed grasp parked on the table is not worth staying for (free early, costly late)
    p_stall = W_STALL * held * (1.0 - lifted) * ctx.episode_progress

    # =======================================================================================
    # Stage D: carry to the goal and hold still
    # =======================================================================================
    goal_dist = ctx.goal_dist
    r_goal = W_GOAL * held * (1.0 - torch.tanh(goal_dist / 0.12))
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    near_goal = 1.0 - torch.tanh(goal_dist / 0.05)                 # fixed scale: stationary reward
    still = torch.exp(-cup_speed / 0.08)
    r_hold = W_HOLD * held * near_goal * still                     # resting at the goal > relifting
    r_success = W_SUCCESS * success

    # =======================================================================================
    # Safety and smoothness
    # =======================================================================================
    p_tilt = W_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))   # keep the cup upright
    p_drop = W_DROP * lifted * (1.0 - held)                        # lifted, then let go
    # neither the palm nor any finger may be pushed into the table top
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = W_TABLE * (palm_pen + floor_pen)
    # smoothness: with arm joints capped near 0.3 rad/s these stay <= ~0.15, far under W_LIFT
    p_arm_vel = W_ARM_VEL * (0.4 + 0.6 * held) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_cup_vel = held * (W_CUP_VEL * torch.tanh(cup_speed / 0.5)
                        + W_CUP_SPIN * torch.tanh(ctx.cup_ang_vel.norm(dim=-1) / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = W_ACT_RATE * torch.tanh(act_rate / 0.1) + W_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1)

    reward = (
        r_approach_shape + r_approach_latch
        + r_contact + r_last_digit + r_wrap + r_oppose + r_palm + r_env_latch
        + r_lift + r_rise
        + r_goal + r_hold + r_success
        - p_descend - p_stall
        - p_tilt - p_drop - p_table
        - p_arm_vel - p_cup_vel - p_action
    )

    components = {
        "approach_shape": r_approach_shape,
        "approach_latch": r_approach_latch,
        "grasp_contact": r_contact,
        "grasp_last_digit": r_last_digit,
        "grasp_wrap": r_wrap,
        "grasp_opposition": r_oppose,
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
```
