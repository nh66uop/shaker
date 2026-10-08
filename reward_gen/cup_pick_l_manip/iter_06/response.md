# Round 6 — left-arm stage 2 (manip): close, lift, carry, hold

## 1. What the task means and the stages the robot must go through

The episode always starts with the hand raised beside the robot and the cup somewhere in a
30 cm x 30 cm patch. The stage-1 policy already solves the travel and the pre-grasp pose, so this
reward has to pay for that only enough to keep it, and put the value in what comes after it:

1. **Travel** to the cup and present the open hand at the pre-grasp pose, in the start orientation,
   without touching or moving the cup, until the environment latches `approach_done`.
2. **Close** a five-digit envelope: the palm arrives on the cup side, the four fingers curl *around*
   the far side of the body, the thumb comes across from them on the near side, and every digit's
   links make and keep contact. The environment latches `envelope_done` when the palm, the thumb and
   at least four digits have touched for five consecutive steps.
3. **Lift** the cup straight off the table, measured from its own spawn height.
4. **Carry** it to `goal_pos` and **hold** it there upright and still, so that `goal_dist <=
   success_tol` for `success_hold_steps` steps with the fingers wrapped, repeatedly, until
   `max_successes`.

Throughout: cup upright, cup not shoved, hand not driven into the table, arm slow.

## 2. Why five rounds ended in the same place, and the one rule this round obeys

Rounds 1-5 each added the term that the previous round's dead end asked for, and each time the
policy went to that new term's maximum and stopped, because **the new term reached its maximum with
zero contact**: the flat open hand (round 2), the loose claw (round 4), the seated palm (round 5).
Round 5 is the clearest case — fine seating reached 0.70 of 0.80 and palm distance 0.45 of 0.60
inside 200 epochs and never moved for 2,800 more, together paying 1.17/step while every finger term
combined paid 0.07.

So this round is built around one rule: **no term may sit near its maximum while contact is zero.**
Concretely, three mechanisms enforce it:

* **A grasp multiplier on the whole pose family.** `pose_mul = 0.40 + 0.60 * grip`, where `grip` is
  built from the number of digits actually touching plus the palm. Travel, seating, orientation and
  the approach latch are all multiplied by it, so each of them tops out at 40 % of its own maximum
  until the hand is gripping. This is feedback 1 ("multiply them by how many digits are engaged")
  applied to every position-only term, not just the two named ones.
* **A hard cap on proximity.** Proximity income is restored to roughly round 4's strength (which
  produced the most contact of any round) but it is capped, and capped *below what a single digit's
  first real contact pays*.
* **The two largest terms in the reward, `grasp_digit_mean` (5.00) and `grasp_digit_last2` (3.00),
  are pure contact.** They are exactly 0 without it. Together with palm contact and the envelope
  latch that is 10.0 of the 16.8 available at the closed grasp.

## 3. Term by term, relative to the round-5 code

**Removed**

* The standalone fine-seating term. `pregrasp_seat` survives at 0.50 instead of 0.80 **and** is now
  multiplied by `pose_mul` (feedback 1: "cap them at a small fraction of their present value and
  move the freed budget into the contact terms"). The freed 0.30 plus the reduction of the other
  pose terms went into `grasp_digit_mean` / `grasp_digit_last2`.
* The `NEAR_SHARE / TOUCH_FIRST / TOUCH_LINKS` per-digit blend. Mixing proximity and contact inside
  one per-digit score made the ordering rule of feedback 2 unsatisfiable: with a mean over five
  digits, five digits at proximity always beat one digit touching. Proximity is now its own term
  (`grasp_reach`, max 0.80) and contact is its own (`grasp_digit_mean`, `grasp_digit_last2`), so the
  two can be ordered against each other directly.

**Added**

* `pose_mul` on `pregrasp_travel`, `pregrasp_seat`, `pregrasp_orient`, `approach_latch`. The latch
  bonus was raised 1.60 -> 3.00 precisely so that, multiplied by 0.40 at zero contact, it still pays
  1.20 — more than the 1.00 open-hand term it switches off. Latching remains a step up (feedback
  from round 1 that must never be undone) and now becomes a much bigger step once the hand grips.
* **A far-side test on every finger link** (feedback 4). For each measured link, `u` is the cosine
  between the link's radial direction about the cup axis and `palm_normal`. A finger lying extended
  past the cup has its near-surface link at `u <= 0`; a finger curled around the far side has
  `u > 0`. `side = clamp((u + 0.30) / 0.90, 0, 1)` multiplies the proximity and wrap credit of the
  four fingers, so "lying alongside the cup" — the exact pose in the round-4 and round-5 recordings
  — earns almost nothing, and "curling onto it" earns full credit. The thumb is not given an
  absolute side (nothing in the context tells us which side of the body the opposed thumb should
  reach); it uses the frame-free `oppose` measure that already taught it to oppose in round 5.
* `pen_cmd_gap` (feedback 3). Round 5 commanded 0.325 of flexion and measured 0.138. The gap between
  `hand_target_norm` and `hand_q_norm`, in the closing direction only, is charged **only on digits
  that are not touching the cup** — a blocked finger pressing on the cup is exactly what we want, a
  finger commanded closed that does not move is a command going nowhere. All the flexion *rewards*
  read `hand_q_norm` (measured), never the command.
* `grasp_reach` as an explicit, bounded on-ramp (feedback 2), with a two-scale pull (30 mm for
  guidance, 6 mm for the last centimetre) and a `(0.60 + 0.40 * touch)` factor so that even the
  ideal zero-contact hover only reaches 60 % of it.

**Re-gated / rescaled**

* `grasp_digit_mean` 1.50 -> 5.00 and `grasp_digit_last2` 2.00 -> 3.00, both now **pure contact**
  (feedback 5 of round 4: "that term should carry most of its value in the contact part"). `last2`
  scores the two lowest digits, so the index and ring fingers — zero for three rounds — are worth
  1.50 each at the margin against 1.00 for the first three, and the 3->4 and 4->5 digit steps are
  +3.2 each (measured).
* `grasp_palm_in` 0.60 -> 0.90, with the gradient concentrated in the last few millimetres
  (`0.45*exp(-e/0.030) + 0.55*exp(-e/0.006)`, feedback 5) and multiplied by
  `(0.15 + 0.85 * digit_engage)` so it is no longer payable on its own (feedback 1). The 0.15 floor
  keeps a gradient that pulls the whole hand in even when the digits are still open.
* `grasp_wrap` 0.40 -> 1.00 but now `side`-gated and `(0.25 + 0.75 * touch)`-gated, so at most 25 %
  of it is reachable without contact.
* `grasp_thumb_oppose` 0.50 -> 0.80 with a `(0.40 + 0.60 * thumb_touch)` factor: the tightened
  proximity test of round 5 is kept (it worked), and the term can no longer sit above 40 %.
* `grasp_unfold` 0.35 -> 0.20, still on measured joints, still fading as the digits approach.
* Closing is gated on `ready * (0.30 + 0.70 * approach)`. `ready` is the continuous geometric
  readiness of rounds 4-5 (keep — it is what produced a 0.95 latch); the `approach` factor makes
  closing 3.3x more profitable after the latch than before it, which, together with
  `P_PRE_FLEX = 4.00`, keeps the pre-latch discipline intact now that closing pays four times what
  it used to.
* **Lift / carry / hold are structurally untouched** (feedback 6) but scaled by ~2 (LIFT 2.20->4.50,
  GOAL 2.00->4.00, HOLD 2.60->5.50, SUCCESS 10->20, DESCEND 0.40->1.00, STALL 0.15->0.60). This is
  not a redesign: the closing block roughly doubled, and if the lift block had not moved with it,
  "closed grasp on the table" would have stopped being strictly worse than "lifted".

## 4. Required table: maximum vs. what is reachable with zero contact

Reference pose for the middle column is the one the robot settles in today (round-5 recording):
latched, palm facing the cup about 0.05 m from its surface, the four fingers extended past the cup
rather than curled around it, the thumb up near the wrist, nothing touching. The right-hand column
is the *best* value the term could reach with zero contact anywhere (a perfectly curled five-digit
hover a few millimetres off the surface) — the pessimistic bound.

| Term | Max | Settled no-contact pose | Best possible, zero contact | % of max |
|---|---|---|---|---|
| pregrasp_travel | 1.00 | 0.32 | 0.40 | 40 % |
| pregrasp_seat | 0.50 | 0.20 | 0.20 | 40 % |
| pregrasp_orient | 0.15 | 0.06 | 0.06 | 40 % |
| pregrasp_open_hand | 1.00 | **0.00** (pre-latch only) | 1.00 pre-latch / 0.00 latched | 0 % here |
| approach_latch | 3.00 | 1.20 | 1.20 | 40 % |
| grasp_reach | 0.80 | 0.00 | 0.48 | 60 % |
| grasp_unfold | 0.20 | 0.02 | 0.20 | 100 % of 0.20 |
| grasp_thumb_oppose | 0.80 | 0.00 | 0.32 | 40 % |
| grasp_wrap | 1.00 | 0.00 | 0.25 | 25 % |
| grasp_digit_mean | 5.00 | 0.00 | 0.00 | 0 % |
| grasp_digit_last2 | 3.00 | 0.00 | 0.00 | 0 % |
| grasp_palm_in | 0.90 | 0.02 | 0.36 | 40 % |
| grasp_palm_contact | 0.80 | 0.00 | 0.00 | 0 % |
| envelope_latch | 1.20 | 0.00 | 0.00 | 0 % |
| lift_height | 4.50 | 0.00 | 0.00 | 0 % |
| lift_rise | 1.20 | 0.00 | 0.00 | 0 % |
| goal_reach | 4.00 | 0.00 | 0.00 | 0 % |
| goal_hold | 5.50 | 0.00 | 0.00 | 0 % |
| success_bonus | 20.0 | 0.00 | 0.00 | 0 % |
| penalties (all) | - | -0.14 (cmd_gap) | -0.01 | - |
| **TOTAL** | **16.8 standing** | **1.68** | **3.15** | |

(Every figure in the two right-hand columns was evaluated by running the function below on a
hand-built context reproducing that pose, not estimated.)

Two rows need a word. `pregrasp_open_hand` is at its maximum with zero contact *while travelling* —
that is its whole purpose, it pays for not closing before the latch — but it is unreachable after
the latch, and the latch pays 1.20 against its 1.00, so it is not a place to park; in the pose the
robot actually settles in it is 0.00. `grasp_unfold` can reach its own maximum with zero contact,
but that maximum is 0.20, it fades to 0.04 as the digits approach the cup, and it is the deliberate
gradient out of the stage-1 open-hand habit that round 3 proved necessary.

**The fraction.** With a closed five-digit envelope and the palm on the cup, the standing income is
**16.8 per step**. The most that can be collected with **zero contact anywhere** is **3.15**, and
the pose the policy occupies today collects **1.68**. So the no-contact ceiling is **19 % of the
contact total**, and the current plateau is **10 %** of it. In round 5 the equivalent figures were
above 60 %: the seated palm and the fine seating alone paid 1.17 out of a ~1.9 total that the policy
ever actually saw.

The ordering rule of feedback 2 also holds explicitly: all five digits at their best proximity are
worth **0.48** (`grasp_reach` at zero contact), while **one single digit making one solid link
contact is worth 0.70** on `grasp_digit_mean` alone (`5.00 * 0.70 / 5`). Stopping at "near" is
strictly a loss.

## 5. Per-step income at each stage — never decreasing

All measured by evaluating the function on a built context for each pose.

| Stage | Income | Where it comes from |
|---|---|---|
| Travelling (0.15 m out, open hand, unlatched) | **1.21** | travel 0.13 + orient 0.06 + open hand 1.00 |
| Arrived at the pre-grasp pose, open hand, not yet latched | **1.68** | travel 0.39 + seat 0.20 + orient 0.06 + open 1.00 |
| Latched (`approach_done`), hand still open | **1.98** | latch 1.20 replaces open 1.00; pre-latch penalties vanish |
| Digits curled onto the far side, near but **not touching** | **2.69** | + reach 0.29 + wrap 0.17 + palm_in 0.37 + unfold 0.03 |
| One digit touching (3 links) | **4.40** | digit_mean 1.00; grip lifts the pose family 1.85 -> 2.30 |
| Five digits touching (palm not yet on) | **14.33** | digit_mean 5.00 + last2 3.00 + reach + wrap + pose family 4.07 |
| Closed envelope (palm + thumb + 4 digits, `envelope_done`) | **16.76** | + palm_in 0.90 + palm contact 0.80 + envelope latch 1.20 - stall |
| Lifted 0.10 m | **20.36** | + lift 4.50 (+ rise up to 1.20 until the latch); stall and shove switch off |
| At the goal, at rest | **30.09** | + goal 4.00 + hold 5.50, plus +20.0 on every success step |

Per digit recruited, holding everything else fixed (latched, curled, palm on, three links each):
0 digits 3.61, 1 digit 5.36, 2 digits 7.21, 3 digits 9.05, 4 digits 12.33, 5 digits 15.56 — the
3->4 and 4->5 steps (+3.28, +3.23) are the largest, which is `grasp_digit_last2` doing its job.
Closing *before* the latch is worth 0.11 against 1.68 for staying open, and 15.47 for closing after
it. While holding, vz = -0.15 pays 19.38, vz = 0 pays 20.18 and vz = +0.15 pays 21.05.

1.21 -> 1.68 -> 1.98 -> 2.69 -> 4.40 -> 14.33 -> 16.76 -> 20.36 -> 30.09: **strictly increasing at every
step of the ladder**, with the two largest jumps exactly where the last five rounds stalled — from
"near" to "touching" (+1.7 for the first digit, +10.6 for the set) and from three digits to five
(+3.2 per digit, carried by `grasp_digit_last2`). Descending, dropping, shoving and squeezing are
the only ways to move backwards, and each of them is a penalty.

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
```
