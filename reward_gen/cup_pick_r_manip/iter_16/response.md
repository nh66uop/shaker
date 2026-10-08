# Round 16: the downstream income scales with ONE envelope quality

## 1. What the task means and which stages the robot goes through

One left OpenArm with a five-finger DG-5F hand has to pick up a cup (several sizes, and a closed shaker) that stands
anywhere in a 0.30 x 0.30 m patch of the table, carry it to a goal 0.21-0.28 m above its start, hold it there upright
while the environment pushes and twists it (up to 20 N/kg as the ADR level rises), and keep producing successes until the
episode ends. The stages, all inside one continuous function with no switch:

1. Reach: palm travels to the object with the hand open (paid on plain palm-to-surface distance, open-pose factor only on
   the way in).
2. Seat and close: palm against a side wall, digits curl onto the wall (palm seating tied to finger engagement).
3. Grasp: several digits with several links each, palm contact, thumb/palm opposing the fingers.
4. Lift: raise the object off the table; this is the only door to everything that follows.
5. Carry: reduce goal_dist over the whole 0-0.30 m range.
6. Arrest and defend: stay at the goal, still and upright, against the wrench; successes are counted there.

## 2. What round 15 settled, and what it left wrong

Round 15 works and is kept whole: both arms lift (0.82-0.96), carry, settle and survive the real wrench (far-start
success 0.68-0.83, the left held ADR 1.0 for 456 epochs), the falling-only drop and set-down costs removed the
latch-driven collapse, and none of the terms collapsed. The metrics table agrees: total reward rises 0.56 -> 2.69,
hold 0.29 -> 1.10, lift 0.40 -> 1.03, drop -0.07 -> -0.013, no termination class above 0.25 %.

What is wrong is the grasp the reward makes optimal. Every downstream income (lift 2.10 + rise 0.40, transport 1.60 +
progress 0.90, hold 10.0, success 60) is multiplied by `held` or `hold_grip`, and `held` saturates at four touching digits
(`digit_hold = (n_touch - 2) / 2`). The fifth digit is paid only inside the 1.2 grasp budget: at most 0.08 (top rung of
the count ladder) + 0.06 (one fifth of the link mean) = 0.14 per step, under 1 % of what four digits unlock (about 13.7
per step at rest on the goal). A deeper wrap likewise only moves the 1.2 budget. The table shows the result:
`contact/finger_index` rises to 0.29 while the policy is still searching (epochs 2-4 of the 10 checkpoints) and then
falls to 0.000 as the policy converges; fingers at success settle at 3.9-4.0 and links at success at 5.2-5.5 of 15; palm
at success 0.52-0.95 and not rising. Nothing pays for the envelope, so nothing forms it.

So the missed step is structural: the reward's notion of "a real hold" stops improving at four digits, and the dominant
income is keyed to that notion.

## 3. The change: one smooth envelope quality multiplying the downstream income

`env_q` in [0, 1] is computed once from four quantities, each symmetric across the digits (sorted, summed or averaged;
no digit is ever named, weighted or thresholded on its own) and each object-agnostic, measured on the same soft contact
forces (`tanh(F / 0.15)`) that `held` uses:

- count (weight 0.30): a sorted ladder over per-digit engagement, rungs [0.02, 0.02, 0.06, 0.15, 0.75] from the strongest
  to the weakest digit. Four digits reach 0.25 of it, the fifth digit is the real step (the other 0.75). Per-digit
  engagement is the digit's soft contact, or at most a quarter of that for a digit whose nearest link is within about a
  centimetre of the surface (`0.25 * exp(-(link_surface_dist - 0.025) / 0.01)`), so a digit that has never touched still
  sees a slope towards the surface, a hovering digit is worth a quarter of a touching one, and contact flicker under the
  wrench does not drop a digit to zero instantly. This on-ramp is capped and sits inside a max with the contact term, so
  the quality is never measured on geometry alone.
- depth (weight 0.35): mean over the five digits of (soft links in contact / 2), capped at 1: full at two links per digit,
  a fingertip-only digit counts half, a missing digit counts zero.
- palm (weight 0.25): palm contact.
- opposition (weight 0.10): the same `opp_sat` that `held` uses (thumb/palm against the fingers from the surface normals).

`env_q = 0.30 * count + 0.35 * depth + 0.25 * palm + 0.10 * oppose`, and the multiplier is floored:
`env_mult = 0.10 + 0.90 * env_q`.

`env_mult` multiplies exactly the six incomes the feedback lists: lift, rise, goal (transport), progress, hold-at-goal
and the success bonus. `held` is untouched and keeps every GATE: the reach/palm-seating/closing fade `(1 - held)`, the
drop cost (`loose = 1 - held`), stall, drift, descend, and the held-scaled smoothness costs. The disturbance therefore
cannot turn a real four-digit hold into penalties, and those gates do not weaken when the envelope quality plateaus.

Why a weighted sum and why these weights: a four-digit grasp must keep earning (the floor plus the palm, opposition and
depth shares give any seated four-digit hold 0.59-0.73), the fifth digit must be the largest single step in the
multiplier (count's 0.75 rung plus the extra depth share), and a five-fingertip pinch without palm or wrap must lose to
a deep four-digit wrap (palm and depth together weigh 0.60, count only 0.30).

## 4. Numbers checked (multiplier values and per-step ceilings)

Multiplier `env_mult`:

| grasp | env_mult |
|---|---|
| five digits, two or more links each, palm, opposition | 1.000 |
| five digits, one link each, palm, opposition | 0.842 |
| four digits, two or more links each, palm, opposition (deep four-digit wrap) | 0.734 |
| same, fifth digit hovering at a 31 mm link reading (the right index today) | 0.762 |
| four digits, 1.35 links each, palm seated (four-digit wrap, palm seated) | 0.653 |
| round-15 right at success (4 digits, 1.35 links, palm 0.7) | 0.585 |
| round-15 left at success (4 digits, 1.2 links, palm 0.92) | 0.616 |
| five fingertips, no palm, opposition | 0.617 |
| five fingertips, no palm, no opposition | 0.527 |
| three digits deep, palm, opposition (held is also only 0.5 here) | 0.631 |

So a four-digit wrap with the palm seated earns 0.65 (shallow) to 0.73 (deep) of the downstream income, five digits with
two or more links plus palm and opposition earn 1.0, the missing fifth digit costs 27 % (deep) to 35 % (shallow) of it on
every step, and the five-fingertip pinch (0.62) is below both the deep four-digit wrap (0.73) and the seated shallow one
(0.65).

Per-step downstream income at rest on the goal (lift 2.1 x held + goal 1.6 x held + hold 10 x hold_grip, dwell 1):

| grasp | before x env_mult | after |
|---|---|---|
| five digits deep, palm, opposition | 13.70 | 13.70 |
| four digits deep, palm, opposition | 13.70 | 10.06 |
| four digits shallow, palm seated | 13.70 | 8.94 |
| round-15 right grasp (palm 0.7) | 9.40 | 5.50 |
| same + the fifth digit touching with one link | 9.40 | 7.70 |
| five fingertips, no palm | 2.31 | 1.43 |

The fifth digit is now worth 2.2-3.6 per step while carrying and holding, instead of 0.14.

Rule (b), checked for four digits: everything available with the object on the table is unchanged (grasp budget at most
1.2 per step and decaying to 45 % late in the episode, reach and closing faded by `(1 - held)`, stall time pressure).
Lifting stays purely additive on top of it: the round-15 right grasp earns +0.43 of lift income at 3 cm (0.73 before),
plus the rise bonus, and about 5.5 per step at rest on the goal, more than four times the whole on-table ceiling. The
smoothness and safety costs that apply while carrying (squeeze, tilt, drift, descend, arm and object speed) summed to
about 0.39 per step at the end of round 15, well under the scaled carrying income, so lifting and carrying still clearly
beat standing still with four digits.

The other orderings are unchanged because `env_mult` scales all downstream incomes of a given grasp by the same factor:
descending < still < rising, drifting off costs more than the hold it gives up, a set-down cannot re-earn the rise
bonus, and later successes stay cheap next to the first.

## 5. What did not change

Everything else is round 15 byte for byte: all weights and budgets, rules (a)-(e), the off_table gate on the second half,
uprightness as a multiplier, the falling-only drop and descending-only set-down costs (no penalty on a persistent latch),
the open-pose factor limited to the approach, palm seating tied to engagement, the first-success-only bonus, the
tolerance-scaled hold region and the success-count dwell. No new additive term was added; the component dict has the
same keys as round 15, so the curves stay comparable (the six downstream components now already include `env_mult`).

The file is frame-agnostic (only dot products of context vectors, normal z components and heights), so it carries no
side-specific line at all: the right arm uses this exact file byte for byte, as every shared round since 10.

## 6. What should move

`contact/finger_index` (0.000 right, about 0.1 left) and `contact/links_touching_at_success` (5.4 / 4.8) should rise,
together with `contact/palm_touching_at_success`, while grasp, lift and success hold. The downstream reward components
drop by roughly 40 % at the start (the multiplier is 0.59-0.62 for the current grasps) with no change in behaviour
ordering, then recover as the fifth digit and the palm arrive. If the index still reads 0.000 after several hundred
epochs with the pre-contact on-ramp in place, that is the point to check geometrically whether that digit can reach the
wall in the current hand pose, before concluding anything from the flat curve.

```python
import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """0 off the object, ~0.6 at 0.1 N, ->1 above a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate and extra force buys nothing."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe normalisation: a zero vector stays zero (its dot products are then 0)."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _sidewall(normal_z: torch.Tensor) -> torch.Tensor:
    """How HORIZONTAL a surface normal is: 1 on a vertical side wall, 0 on a rim face, a lid or the
    table, ~0.3 on a 45 deg rim edge. Only the z component of a context normal is used, and gravity
    is shared by the left arm and the mirrored right arm, so this stays mirror-safe.

    It is NEVER applied to the long-range approach (round 11: from 0.15-0.49 m away the nearest
    point of an open cup is its rim, the approach paid ~0 and both arms retreated). It is used only
    from a few centimetres in, always with a floor."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Object-agnostic and arm-agnostic: surface distances/normals, contact forces, object
    # pose/velocity, goal. No approach/envelope latch, no radius/axis/half-height, no world-frame
    # direction constant (only normal z components and heights, identical under the mirror).
    #
    # Per-step ceilings (the budget IS the design):
    #   reach            0.60   pre-contact only, x (1 - held); it vanishes once the object is held
    #                           = 0.22 long range (PLAIN distance) + 0.24 palm seating + 0.14 closing
    #   keep the grasp   1.20   links-per-digit + count ladder (only the TOP rungs need the palm)
    #                           + palm contact (0.30) + opposition
    #   lift             2.50   x uprightness x real hold, rising to the height the goal needs
    #   transport        2.50   1.60 on goal_dist over 0-0.30 m + 0.90 on the signed RATE it falls
    #   hold at rest    10.00   real hold, upright, at the goal, still, growing with successes
    #   first success   60.00   one-off; later successes 2.0, so staying pays, not re-arriving
    #   ROUND 16: lift, rise, transport, progress, hold and success are ALL x env_mult in [0.10, 1]:
    #   a deep four-digit wrap with the palm keeps 0.73 of them, five digits with two or more links,
    #   palm and opposition keep all of them.
    #
    # STRUCTURAL RULES, every one of them a scar from an earlier round:
    #  (a) Everything from "transport" down is x `off_table`, a smooth function of the object's rise
    #      above its OWN spawn height: nothing there pays while the object rests (round 10).
    #  (b) Stage one is ALWAYS profitable: no factor zeroes the long-range approach and no penalty
    #      exceeds the income of the stage the policy is in (round 11).
    #  (c) The hand is paid to stay OPEN only on the way in: the open-pose factor fades with palm
    #      distance (round 12 arrived curled; round 13 parked a flat open hand on the wall).
    #  (d) Palm seating is paired with finger engagement, never paid on proximity alone (round 13).
    #  (e) ROUND 15: NO PENALTY IS PROPORTIONAL TO A PERSISTENT LATCH. Round 14 charged
    #      8 x lifted x (1 - held): after one lift every later step without a full hold cost 8 until
    #      the episode ended, it became the largest term (-0.5 to -0.96/step) while the right arm was
    #      at its best, and the arm unlearned lifting, then grasping, and moved 0.29 m away. Losing
    #      the object now costs a bounded amount ONLY WHILE IT IS FALLING (current vertical velocity),
    #      setting it down costs only WHILE IT COMES DOWN onto the table, and the main cost of a drop
    #      is the lift / transport / hold income that stops. `lifted` survives only as the switch
    #      that stops the rise bonus from being paid twice (income, never negative).
    #  (f) ROUND 16: THE DOWNSTREAM INCOME SCALES WITH ONE ENVELOPE QUALITY. Round 15 multiplied all
    #      of it (about 13.7/step at rest on the goal) by `held`, which saturates at FOUR digits; the
    #      fifth digit was worth at most 0.14/step inside the 1.2 grasp budget, and both arms settled
    #      on a four-digit grasp (right index 0.000 on every species, links at success 5.4 of 15).
    #      `env_q` is built only from quantities that treat the digits alike (a sorted engagement
    #      ladder whose top rung is the fifth digit, mean links per digit, palm contact, opposition),
    #      on the same soft contact forces as `held`; no term names a digit. It multiplies the six
    #      incomes, floored so four digits still earn and still lift. `held` keeps every GATE (the
    #      reach fade, drop, stall, drift, descend), so the wrench cannot turn a real four-digit hold
    #      into penalties and those gates do not weaken when the envelope quality plateaus.
    #
    # Steady states: grasp parked on the table ~1.2 early, ~0.2 late (unchanged); at rest on the
    # goal ~5.5 (the round-15 right grasp) / ~10 (deep four-digit wrap) / ~13.7 (five digits deep);
    # a first-success spike of up to 60.
    # =============================================================================================
    W_REACH = 0.22        # long range, PLAIN palm-to-surface distance, x open-pose  } 0.60 total,
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted } x (1 - held)
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat   }
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm }
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike            } 1.20:
    W_PALM = 0.30         # palm contact: a pinch must not score like a whole-hand wrap } build and
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)            } keep a grasp
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height }
    W_RISE = 0.40         # upward speed, only until `lifted` latches (income gate, not penalty) } 2.50
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m        } 2.50 total,
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls                     } x off_table
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 2.00         # ROUND 15: per step, ONLY while the object is falling without a hold.
    #                       Below one step of lift income (2.5); a fall from goal height lasts ~14
    #                       steps, so a whole drop costs ~25 once, not 8/step for the rest of the
    #                       episode as in round 14.
    P_SETDOWN = 1.00      # ROUND 15: per step, ONLY while the object is coming down onto the table
    #                       (band 0.5-6 cm above it). Resting there costs nothing extra.
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it: costs more than the hold it gives up
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    P_SQUEEZE = 0.50      # force beyond what a hold needs (thresholds measured in round 7)
    P_TABLE = 0.60        # palm or finger links driven into the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: 0.02 at 0.1 m/s against +0.7 of transport income,
    P_CUP_SPIN = 0.06     #   so lifting and carrying always beat standing still
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    PALM_STEEP = 0.02     # seating scale: 0.05 at 0.08 m, 0.22 at 0.05 m, 0.61 at 0.03 m, 1 at 0.02
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link (contact geometry is noisier)
    OPEN_FLOOR = 0.15     # a fully curled hand still earns 15 % of the approach: rule (b)
    OPEN_MEAN_SCALE = 0.12   # mean deviation from the default open pose
    OPEN_MAX_SCALE = 0.30    # and the WORST joint, which is what actually blocks a wrap
    OPEN_NEAR = 0.08      # open-pose pays in full at this palm-surface distance or beyond
    OPEN_CONTACT = 0.02   # and its credit is gone by here, where contact starts
    PALMIN_ENGAGE_FLOOR = 0.35   # seating credit an entirely disengaged hand still keeps
    TABLE_RISE = 0.04     # smooth "off the table" scale: 0 at 0, 0.46 at 2 cm, 0.96 at 8 cm
    TILT_DEAD = math.radians(5.0)    # a few degrees of wobble is not a tilted carry
    TILT_SCALE = math.radians(10.0)  # 15 deg -> x0.37, 30 deg -> x0.08 of lift/transport/hold
    SQ_TOTAL = 40.0       # measured: a grip that holds fine paid -0.45/step at a 15 N threshold
    SQ_LINK = 15.0
    # ---- ROUND 15 constants ----------------------------------------------------------------------
    FALL_DEAD = 0.03      # m/s: solver jitter and a slow deliberate lowering are not a fall
    FALL_SCALE = 0.20     # m/s: free fall passes this within ~2 steps, so a real drop is charged fully
    SET_DEAD = 0.02       # m/s: a resting object never triggers the set-down cost
    SET_SCALE = 0.05      # m/s
    SET_LIFTOFF = 0.005   # m: the set-down band starts 5 mm above the spawn height (0 when resting)
    SET_BAND = 0.04       # m: and fades out over the last ~6 cm above the table
    GOAL_TOL_MULT = 3.0   # broad hold region = 3 tolerances: 0.89 at one tolerance, 0.64 at two
    GOAL_TOL_MIN = 0.03   # never narrower than 3 cm, however small the tolerance gets
    GOAL_FINE = 0.04      # fixed fine ramp for the last centimetres
    SETTLE_FLOOR = 0.40   # a push that moves the object costs part of the hold income, not 2/3 of it
    DWELL_BASE = 0.70     # hold income before the first success of the episode
    DWELL_SCALE = 1.5     # -> 0.87 after one success, 0.96 after two
    # ---- ROUND 16 constants: the envelope quality (see rule (f)) --------------------------------
    ENV_FLOOR = 0.10      # env_mult = 0.10 + 0.90 * env_q: never zero, a four-digit hold keeps earning
    ENV_W_COUNT = 0.30    # sorted engagement ladder (the fifth digit is its biggest rung)  } sum 1.0;
    ENV_W_DEPTH = 0.35    # mean links in contact per digit, full at ENV_DEPTH_LINKS        } palm + depth
    ENV_W_PALM = 0.25     # palm contact                                                    } (0.60) >
    ENV_W_OPP = 0.10      # thumb/palm opposition, same saturation as `held`                } count (0.30)
    ENV_DEPTH_LINKS = 2.0  # "two or more links each" is a full wrap; a fingertip alone is half
    ENV_NEAR_SHARE = 0.25  # a digit hovering at the surface is worth at most 1/4 of a touching one
    ENV_NEAR_SCALE = 0.01  # m: pre-contact on-ramp, about the last 1-2 cm before a link touches

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    n_succ = ctx.num_successes.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    # locked = thumb_2(0), index_1(3), middle_1(7), ring_1(11), pinky_1(15), pinky_2(16)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    n_movable = movable.sum().clamp(min=1.0)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0      # thumb_3, thumb_4
    digit_joint[1, 4:7] = 1.0      # index_2, _3, _4
    digit_joint[2, 8:11] = 1.0     # middle_2, _3, _4
    digit_joint[3, 12:15] = 1.0    # ring_2, _3, _4
    digit_joint[4, 17:19] = 1.0    # pinky_3, pinky_4
    n_digit_joint = digit_joint.sum(-1)                                        # (5,)

    # =============================================================================================
    # HEIGHT AND VERTICAL VELOCITY: the spine of this reward.
    # =============================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)      # rise above ITS start
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)    # read from the goal
    off_table = torch.tanh(height / TABLE_RISE)     # smooth, exactly 0 on the table
    on_table = 1.0 - off_table
    vz = ctx.cup_lin_vel[:, 2]

    # uprightness: ONE multiplier on every lift / transport / hold income (tilt fell 28 -> 5 deg when
    # it became a multiplier instead of a small additive penalty).
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)     # (N,) in (0,1]

    # =============================================================================================
    # CONTACT AND GRASP QUALITY -- paid by LINKS, not by "this digit touches"
    # =============================================================================================
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3) plain contact
    digit_touch = link_c.amax(dim=-1)                              # (N,5) does this digit touch
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)                              # (N,) 0..5
    finger_any = digit_touch[:, 1:].amax(dim=-1)                   # (N,) any of the four fingers
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    # WHERE the links touch: a link on a vertical WALL counts fully, one hooked over a rim or lying on
    # a lid counts 0.35 (a touching link is close in by definition). Scoring only: `digit_touch` and
    # `held` stay on the plain force, so geometry noise cannot dissolve a real hold under the wrench.
    link_side = _sidewall(ctx.link_surface_normal[..., 2])                          # (N,5,3)
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side             # (N,5,3)
    link_cw = link_c * link_side_w                                                  # (N,5,3)
    # NUMBER OF LINKS of each digit, convex: 1 wall link -> 0.19, 2 -> 0.54, 3 -> 1.00; one fingertip
    # on the rim -> 0.04.
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- the hand's OPEN-POSE measure (mean AND worst movable joint) -----------------------------
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable               # (N,19)
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))                      # (N,) in (0,1]

    # ---- measured per-digit flexion (contact-free) -----------------------------------------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)        # saturates at half flexion: no crushing drive

    # ---- palm against the object's real surface --------------------------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    # facing: the palm normal points at the nearest surface point AND against the outward normal
    # there (dot products of context vectors only, nothing tied to a world axis)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])           # 1 at a wall, ~0 at a rim/lid
    # side-wall preference only close in: full at 3 cm, 0.37 at 6 cm, 0.05 at 12 cm, floored
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    # steep seating gradient across 0.08 -> 0.02 m (both arms once stalled at a 0.05 m palm gap)
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    # contact quality: a first partial wrap is worth forming before the palm has arrived
    cq = 0.35 + 0.65 * seat

    # open-pose factor, rule (c): full weight at 8 cm or more, gone by 2 cm, floored, never zero
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)
    open_mul = OPEN_FLOOR + (1.0 - OPEN_FLOOR) * open_pose * open_gate

    # ---- opposition from the surface normals at the touching links ------------------------------
    normals = ctx.link_surface_normal                                        # (N,5,3,3)
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))           # (N,3)
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))   # (N,3)
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)             # (N,)

    # "held" = palm plus several OPPOSING digits (smooth, full at four digits)
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)                      # (N,)
    # the 10/step asks for more than the lift does: palm and opposition enter a SECOND time, so a
    # fingertip touch can never collect the dominant income, even under the wrench
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # =============================================================================================
    # ROUND 16: ENVELOPE QUALITY -- ONE smooth multiplier on every downstream income (rule (f)).
    # Symmetric across the digits (sorted / summed / averaged, nothing names a digit), object-agnostic,
    # and on the SAME soft contact forces as `held`. `held` itself is unchanged and stays the gate.
    # =============================================================================================
    # per-digit engagement: its soft contact, or at most ENV_NEAR_SHARE of it while its nearest link
    # is within ~1-2 cm of the surface. The on-ramp gives a digit that has never touched a slope
    # towards the surface, and keeps a flickering contact under the wrench from dropping to zero.
    digit_gap = (ctx.link_surface_dist.amin(dim=-1) - LINK_DEAD).clamp(min=0.0)           # (N,5)
    digit_eng = torch.maximum(digit_touch, ENV_NEAR_SHARE * torch.exp(-digit_gap / ENV_NEAR_SCALE))
    # sorted ladder, strongest -> weakest digit: four digits reach 0.25 of it, the FIFTH is the real
    # step (0.75). Sorting makes it symmetric: whichever digit arrives last carries the big rung.
    env_rank = torch.tensor([0.02, 0.02, 0.06, 0.15, 0.75], device=device, dtype=dtype)
    eng_desc, _ = torch.sort(digit_eng, dim=-1, descending=True)
    env_count = (eng_desc * env_rank).sum(dim=-1)                                          # (N,)
    # wrap depth: mean over the five digits of links in contact, full at two links per digit
    env_depth = (link_c.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0).mean(dim=-1)          # (N,)
    env_q = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
             + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)          # (N,)
    # floored: 5 digits x 2+ links + palm + opposition -> 1.00; deep four-digit wrap + palm -> 0.73;
    # four digits at 1.35 links + palm -> 0.65; five fingertips without palm -> 0.62
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- 0.60 total, fading out once the object is really held (unchanged from round 14)
    # =============================================================================================
    fade = loose
    reach_far = (1.0 - torch.tanh(palm_d / 0.20)) * (0.4 + 0.6 * face)   # 0.20 m covers 0.15-0.49 m
    r_reach = W_REACH * fade * open_mul * reach_far * (0.45 + 0.55 * grip)

    # palm seating paired with finger engagement (rule (d)): a flat open hand on the wall earns a
    # small fraction; a hand that has started closing keeps the full steep gradient
    engage = (0.5 * flex.mean(dim=-1) + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    r_palm_in = (W_PALM_IN * fade * open_mul * palm_near * (0.3 + 0.7 * face) * palm_side_w
                 * seat_engage)

    # closing only pays once the links are near a WALL with the palm seated
    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)      # (N,5,3)
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)                       # (N,5)
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total. Every digit pays alike; two digits are worth far less than four.
    # =============================================================================================
    # sorted rank weights (sum 1.0): 2 digits 0.18, 3 = 0.42, 4 = 0.80, 5 = 1.00
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w                                            # (N,5)
    # lower rungs pay WITHOUT the palm (a partial wrap is worth forming); the top rungs need it
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)
    # palm contact is worth having, but only together with digits
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # a closed grasp parked on the table becomes unattractive as the episode goes on: the on-table
    # GRASP income decays to 45 % (never the reach terms, rule (b)). episode_progress restarts after
    # a success, so this is time pressure, not a latch.
    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_oppose = r_oppose * table_decay

    # =============================================================================================
    # LIFT -- 2.50 total, the largest single step up in income and the ONLY door to the goal terms.
    # Height-shaped, so after a drop re-grasping and lifting again restores it (recovery is worth
    # learning) while a set-down-and-relift cycle still earns strictly less than staying up.
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)     # steep first 3 cm
    r_lift = W_LIFT * held * upright * lift_shape * env_mult
    # upward speed pays only until `lifted` latches: the ACT of lifting is never paid twice. This is
    # the only remaining use of the latch, and it gates income, so it can never become a penalty.
    r_rise = (W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)
              * env_mult)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, ALL of it x off_table
    # =============================================================================================
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)           # constant slope over 0-0.30 m
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = (W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)
              * env_mult)

    # the REDUCTION of goal_dist between steps, read statelessly as the closing speed towards the
    # goal. Signed, so receding costs what approaching pays; faded inside ~3 cm so jitter at rest
    # and the environment's pushes neither pay nor punish.
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10) * env_mult)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, the dominant income, also x off_table
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    # ROUND 15: a flat-topped broad region on three CURRENT tolerances (0.89 at one tolerance, 0.64
    # at two: a large fraction is paid before the success condition can be met) that falls off
    # beyond it faster than round 14's tanh, plus a fixed 4 cm fine ramp for the last centimetres.
    # The wrench only acts within 0.15 m of the goal; with round 14's shape, hovering just outside
    # that radius paid about as much as holding station inside it once pushes cut the stillness.
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    # floor 0.40 (was 0.30): the pushes are real now; holding station against one must stay clearly
    # profitable, while drifting off still loses near_goal (and pays pen_drift below)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    # ROUND 15 dwell: grows with the successes already counted in this episode, i.e. with completed
    # hold windows at the goal. Round 14 used episode_progress, which the environment restarts after
    # every success, so each success knocked dwell back to 0.55 while a policy that never succeeded
    # kept it near 1.0: the factor paid MORE for not staying.
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    # the FIRST success of an episode is the big one; later ones are cheap, because staying must pay
    # through r_hold. The factor averages the two possible num_successes conventions (whether it
    # already includes this step's success), so it decays under either one.
    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded (rule (e))
    # =============================================================================================
    # DROP: charged only while the object is actually FALLING without a hold, from its current
    # vertical velocity. No latch: it is exactly 0 once the object rests, wherever that is. The main
    # cost of a drop is the lift/transport/hold income (2.5-13/step) that it switches off.
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    # SET-DOWN: charged only while the object is coming DOWN onto the table (a band from 5 mm to ~6 cm
    # above its spawn height), held or not. A resting object is outside the band (height ~0) and not
    # moving, so nothing is charged for lying there.
    set_band = torch.tanh(height / SET_LIFTOFF) * torch.exp(-height / SET_BAND)
    p_setdown = P_SETDOWN * set_band * torch.tanh(((-vz) - SET_DEAD).clamp(min=0.0) / SET_SCALE)
    # time pressure, gated on the hand being ON the object; a policy that has not reached it pays 0
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    # drifting away from the goal once near it (current velocity only, not a latch)
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    # lowering a held object costs, but only BELOW goal height: descending < still < rising
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS (all small next to the stage income)
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    p_shove = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                          + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    # neither the palm nor any finger link may be driven into the table top
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = P_TABLE * 0.5 * (palm_pen + floor_pen)
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * held) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = held * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                        + P_CUP_SPIN * torch.tanh(spin / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_reach + r_palm_in + r_close
        + r_count + r_digit + r_palm + r_oppose
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_drift - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero  # keeps shape (N,) even if a term ever broadcasts

    components = {
        "reach": r_reach,
        "palm_in": r_palm_in,
        "close": r_close,
        "grasp_count": r_count,
        "grasp_digit": r_digit,
        "grasp_palm": r_palm,
        "grasp_oppose": r_oppose,
        "lift_height": r_lift,
        "lift_rise": r_rise,
        "goal_reach": r_goal,
        "goal_progress": r_progress,
        "goal_hold": r_hold,
        "success_bonus": r_success,
        "pen_drop": -p_drop,          # round 15: falling-only, no latch
        "pen_setdown": -p_setdown,    # round 15: descending-onto-the-table only, no latch
        "pen_stall": -p_stall,
        "pen_drift": -p_drift,
        "pen_descend": -p_descend,
        "pen_tilt": -p_tilt,
        "pen_shove": -p_shove,
        "pen_squeeze": -p_squeeze,
        "pen_table": -p_table,
        "pen_arm_vel": -p_arm_vel,
        "pen_arm_fast": -p_arm_fast,
        "pen_cup_vel": -p_cup_vel,
        "pen_action": -p_action,
    }
    return reward, components
```
