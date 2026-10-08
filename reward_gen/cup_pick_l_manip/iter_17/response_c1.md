## 1. What the task means and the stages the robot goes through

The robot is a left OpenArm with a Tesollo DG-5F hand. It has to pick up an upright cup whose position is randomized each episode. There are 8 object species, and radius and band height differ per environment. It then carries the cup, upright, to a goal 0.21-0.28 m above the spawn point and holds it there. While it holds, a goal-gated wrench pushes and twists the cup (5 N/kg, rising to 20 N/kg). The whole cycle repeats. A success needs `goal_dist <= success_tol` for the hold window, with the fingers wrapped around the body. The stages all come from one continuous reward, with no stage switch:

1. **Travel with an open hand.** The palm moves toward the object's real surface on the plain palm-to-surface distance. The hand is paid to stay open only on the way in: the open-pose factor fades out between 8 cm and 2 cm.
2. **Seat the palm on the side wall and close onto it.** Palm seating has a steep slope from 0.08 m to 0.02 m and is paired with finger engagement. The wall-over-rim preference applies only close in, and closing pays only once the links are near the wall.
3. **Form the envelope.** This means five digits, the palm, and the thumb opposing the fingers. New this round: the digits must carry their grip on their inner links, i.e. curl around the body instead of pressing with the fingertips.
4. **Lift.** Lifting is the only way into the second half of the task. Every second-half income is multiplied by `off_table`, a smooth function of how far the object has risen above its own spawn height, and by uprightness.
5. **Carry.** Income is shaped on `goal_dist` over 0 to 0.30 m, plus the signed rate at which it falls.
6. **Settle and defend at the goal.** The hold term (10/step, growing with dwell) is the dominant income. The first success of an episode pays a large bonus and later ones pay little. The drop and set-down costs are charged only while the object is actually falling or coming down, never on a persistent latch.

**What round 16 shows.** Round 16 works. Lifted fraction is 0.72, 4.81 fingers touch at success, and 6.03 links touch at success. The index finger went from 0.000 to 0.69, and the wrench ADR held 20 N/kg. What went wrong is the wrap-depth part of `env_q`, `(link_c.sum / 2).clamp(1)`. `link_c` saturates at a few tenths of a newton, so a fingertip pressing with several newtons while an inner link brushes the wall at 0.1 N already scores about 0.8 of a full wrap. The reward cannot tell a tip press from a curl, and the video shows exactly a tip press. The metrics agree: mean link force rose from 3.4 to 4.95 N and `pen_squeeze` from -0.17 to -0.26, so pressing harder is what the hand learned.

**The one change.** For each digit, compute the share of its own cup force carried by the inner links (k=0 and k=1) against the fingertip (k=2):

- It is a relative measure and the same for every digit; no digit is named.
- A digit with no inner-link force keeps 0.30 of its round-16 depth. A share of 0.5 or more counts in full.
- The share is trusted as tanh(digit force / 0.3 N). As a digit's force goes to zero, the weight returns smoothly to 1, which is the round-16 link count.

Resulting `env_mult` values:

| Grasp | Round 16 | Round 17 |
|---|---|---|
| Five digits, deep, with palm and opposition | 1.00 | 1.00 |
| Deep four-digit wrap with palm | 0.73 | 0.73 |
| Five digits pressing with the tips, with palm | 0.84-0.94 | about 0.73-0.77 |
| Five fingertips, no palm | 0.62 | about 0.51 |

Curling around the body is now worth about a quarter of the downstream income on every step. Nothing else changes: the squeeze constants, the budgets, the gates and the structural rules all stay.

**What to watch.** `pen_squeeze` and `contact/link_force_mean` should stop rising. `contact/links_touching_at_success` should rise from about 6. Grasp, lift and success should hold. The success tolerance restarting on resume is an environment matter, not a reward one.

### Conflict note (before the code)

- **Conflict:** The prompt's `RewardContext` list (prompt.md:41-96) has no `palm_surface_dist/dir/normal` or `link_surface_dist/normal`. prompt.md:199 says "never access a field that is not listed". On the other hand, prompt.md:2444-2448 says the environment provides these fields. The round 7-16 code read them and trained with no Traceback (prompt.md:2712-2714, 6767-6768), and the final instruction is "Nothing else changes" (prompt.md:6812).
- **Conservative choice:** Keep those proven accesses exactly as round 16 had them. The alternative, rewriting every surface term on cylinder fields, is a much larger behaviour change, and the user banned the cylinder template in round 6 (prompt.md:2435-2442).
- **Scope of the new term:** The only new term reads `link_cup_force`, which is a listed field.
- `approach_done` and `envelope_done` are listed but still unused, following prompt.md:2435-2442.
- The component dict keeps its names and remains a sum decomposition of the reward. I added no diagnostic keys.

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
    # Per-step ceilings (the budget IS the design, unchanged since round 15):
    #   reach            0.60   pre-contact only, x (1 - held); it vanishes once the object is held
    #                           = 0.22 long range (PLAIN distance) + 0.24 palm seating + 0.14 closing
    #   keep the grasp   1.20   links-per-digit + count ladder (only the TOP rungs need the palm)
    #                           + palm contact (0.30) + opposition
    #   lift             2.50   x uprightness x real hold, rising to the height the goal needs
    #   transport        2.50   1.60 on goal_dist over 0-0.30 m + 0.90 on the signed RATE it falls
    #   hold at rest    10.00   real hold, upright, at the goal, still, growing with successes
    #   first success   60.00   one-off; later successes 2.0, so staying pays, not re-arriving
    #   lift, rise, transport, progress, hold and success are ALL x env_mult in [0.10, 1] (round 16).
    #
    # STRUCTURAL RULES, every one of them a scar from an earlier round:
    #  (a) Everything from "transport" down is x `off_table`, a smooth function of the object's rise
    #      above its OWN spawn height: nothing there pays while the object rests (round 10).
    #  (b) Stage one is ALWAYS profitable: no factor zeroes the long-range approach and no penalty
    #      exceeds the income of the stage the policy is in (round 11).
    #  (c) The hand is paid to stay OPEN only on the way in: the open-pose factor fades with palm
    #      distance (round 12 arrived curled; round 13 parked a flat open hand on the wall).
    #  (d) Palm seating is paired with finger engagement, never paid on proximity alone (round 13).
    #  (e) NO PENALTY IS PROPORTIONAL TO A PERSISTENT LATCH (round 14). Losing the object costs a
    #      bounded amount ONLY WHILE IT IS FALLING, setting it down only WHILE IT COMES DOWN onto the
    #      table; the main cost of a drop is the downstream income that stops. `lifted` survives
    #      only as the switch that stops the rise bonus from being paid twice (income, never negative).
    #  (f) THE DOWNSTREAM INCOME SCALES WITH ONE ENVELOPE QUALITY `env_q` (round 16), built only from
    #      quantities that treat the digits alike, on the same soft contact forces as `held`; no
    #      term names a digit. `held` keeps every GATE (reach fade, drop, stall, drift, descend).
    #  (g) ROUND 17 (the ONE change): WRAP DEPTH IS WHERE A DIGIT CARRIES ITS GRIP, not how many of
    #      its links touch. Round 16's videos showed index/middle/ring/pinky pressing with their
    #      FINGERTIPS, and a digit whose tip pressed hard while an inner link brushed the wall already
    #      scored as a two-link wrap. The depth part of env_q is now weighted per digit by the share
    #      of that digit's OWN cup force carried by its two inner links (k=0, k=1) against its
    #      fingertip (k=2): relative (never an absolute force target), symmetric, no digit named, and
    #      falling back smoothly to the round-16 link count as the digit's force goes to zero.
    #      P_SQUEEZE / SQ_TOTAL / SQ_LINK stay: pressing harder with the tips no longer raises env_q,
    #      so the squeeze charge stops competing with it.
    #
    # Steady states: grasp parked on the table ~1.2 early, ~0.2 late (unchanged); at rest on the
    # goal ~10 (deep four-digit wrap) / ~10.5 (five digits pressing with the tips + palm) / ~13.7
    # (five digits curled deep); a first-success spike of up to 60.
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

    P_DROP = 2.00         # per step, ONLY while the object is falling without a hold (below one
    #                       step of lift income; a whole drop costs a bounded amount once)
    P_SETDOWN = 1.00      # per step, ONLY while the object is coming down onto the table
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it: costs more than the hold it gives up
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    P_SQUEEZE = 0.50      # force beyond what a hold needs (kept as is in round 17, by instruction)
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
    # ---- round 15 constants -------------------------------------------------------------------
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
    # ---- round 16 constants: the envelope quality (rule (f)) -----------------------------------
    ENV_FLOOR = 0.10      # env_mult = 0.10 + 0.90 * env_q: never zero, a four-digit hold keeps earning
    ENV_W_COUNT = 0.30    # sorted engagement ladder (the fifth digit is its biggest rung)  } sum 1.0;
    ENV_W_DEPTH = 0.35    # wrap depth per digit (ROUND 17: force-distribution weighted)     } palm + depth
    ENV_W_PALM = 0.25     # palm contact                                                    } (0.60) >
    ENV_W_OPP = 0.10      # thumb/palm opposition, same saturation as `held`                } count (0.30)
    ENV_DEPTH_LINKS = 2.0  # link-count part: "two or more links each" is full, a fingertip alone half
    ENV_NEAR_SHARE = 0.25  # a digit hovering at the surface is worth at most 1/4 of a touching one
    ENV_NEAR_SCALE = 0.01  # m: pre-contact on-ramp, about the last 1-2 cm before a link touches
    # ---- ROUND 17 constants: where each digit carries its grip (rule (g), the ONE change) ------
    DEPTH_SHARE_FULL = 0.50   # inner links (k=0, k=1) carrying half or more of the digit's OWN cup
    #                           force count as a full wrap (design choice; that a curled digit on
    #                           these objects reaches this share is a hypothesis)
    DEPTH_TIP_FLOOR = 0.30    # a digit carrying ALL of its force on the fingertip keeps 30 % of its
    #                           link-count depth: a tip press is a SHALLOW wrap, not a lost contact
    DEPTH_FORCE_REF = 0.30    # N: the share is trusted as tanh(digit force / 0.30); as the digit's
    #                           force goes to zero the depth returns smoothly to the link count

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

    # "held" = palm plus several OPPOSING digits (smooth, full at four digits). Unchanged: it is the
    # GATE for the reach fade and every cost, so a real four-digit hold is never turned into penalties.
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)                      # (N,)
    # the 10/step asks for more than the lift does: palm and opposition enter a SECOND time, so a
    # fingertip touch can never collect the dominant income, even under the wrench
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # =============================================================================================
    # ENVELOPE QUALITY -- ONE smooth multiplier on every downstream income (rule (f)).
    # Symmetric across the digits (sorted / summed / averaged, nothing names a digit), object-agnostic.
    # =============================================================================================
    # per-digit engagement: its soft contact, or at most ENV_NEAR_SHARE of it while its nearest link
    # is within ~1-2 cm of the surface (unchanged from round 16)
    digit_gap = (ctx.link_surface_dist.amin(dim=-1) - LINK_DEAD).clamp(min=0.0)           # (N,5)
    digit_eng = torch.maximum(digit_touch, ENV_NEAR_SHARE * torch.exp(-digit_gap / ENV_NEAR_SCALE))
    # sorted ladder, strongest -> weakest digit: four digits reach 0.25 of it, the FIFTH is the real
    # step (0.75). Sorting makes it symmetric: whichever digit arrives last carries the big rung.
    env_rank = torch.tensor([0.02, 0.02, 0.06, 0.15, 0.75], device=device, dtype=dtype)
    eng_desc, _ = torch.sort(digit_eng, dim=-1, descending=True)
    env_count = (eng_desc * env_rank).sum(dim=-1)                                          # (N,)

    # ---- ROUND 17 (rule (g), the ONE change): wrap depth = where each digit CARRIES its grip -------
    # round-16 depth, kept as the base: links in contact per digit, full at two links. On its own it
    # scored a hard-pressing fingertip plus a brushing inner link as a two-link wrap.
    count_depth = (link_c.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0)                   # (N,5)
    # share of each digit's OWN cup force carried by its two inner links (k=0 moved by joint _3,
    # k=1 moved by joint _4) against its fingertip (k=2). A RELATIVE measure: pressing harder with
    # the tip lowers it, and no absolute force is ever asked for. Same formula for all five digits.
    link_f = ctx.link_cup_force.clamp(min=0.0)                                             # (N,5,3)
    digit_f = link_f.sum(dim=-1)                                                           # (N,5)
    inner_share = ((link_f[..., 0] + link_f[..., 1])
                   / digit_f.clamp(min=1e-6)).clamp(min=0.0, max=1.0)                      # (N,5)
    # tip only (share 0) -> DEPTH_TIP_FLOOR of the depth; half of the force or more inside -> all of it
    share_w = (DEPTH_TIP_FLOOR
               + (1.0 - DEPTH_TIP_FLOOR) * (inner_share / DEPTH_SHARE_FULL).clamp(max=1.0))  # (N,5)
    # the share is only trusted once the digit actually presses; as its force goes to zero the weight
    # returns smoothly to 1, i.e. to the round-16 link count (which is itself ~0 there)
    share_trust = torch.tanh(digit_f / DEPTH_FORCE_REF)                                    # (N,5)
    depth_w = (1.0 - share_trust) + share_trust * share_w                                  # (N,5)
    env_depth = (count_depth * depth_w).mean(dim=-1)                                       # (N,)

    env_q = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
             + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)          # (N,)
    # floored: five digits curled (2+ links, half their force or more on the inner links) + palm +
    # opposition -> 1.00; deep four-digit wrap + palm -> 0.73; five digits pressing with the TIPS +
    # palm -> ~0.73-0.77 (round 16 scored it 0.84-0.94); five fingertips without palm -> ~0.51
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- 0.60 total, fading out once the object is really held
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
    # a flat-topped broad region on three CURRENT tolerances (0.89 at one tolerance, 0.64 at two)
    # plus a fixed 4 cm fine ramp for the last centimetres
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    # floor 0.40: holding station against a real push must stay clearly profitable, while drifting
    # off still loses near_goal (and pays pen_drift below)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    # dwell grows with the successes already counted in this episode (completed hold windows)
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    # the FIRST success of an episode is the big one; later ones are cheap. The factor averages the
    # two possible num_successes conventions, so it decays under either one.
    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded (rule (e))
    # =============================================================================================
    # DROP: charged only while the object is actually FALLING without a hold (current velocity)
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    # SET-DOWN: charged only while the object is coming DOWN onto the table (band 5 mm to ~6 cm)
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
    # squeeze unchanged (round-17 instruction): with rule (g) a harder tip press no longer raises
    # env_q, so this charge no longer competes with the envelope income
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
        "pen_drop": -p_drop,          # falling-only, no latch
        "pen_setdown": -p_setdown,    # descending-onto-the-table only, no latch
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

### 근거 메모

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| env_q 의 wrap depth (`env_depth`: 링크 수 x 내측 링크 힘 비율 가중, 이번 유일 변경) | 영상에서 검지~새끼가 손끝으로 누르며 감싸지 않음. 링크별 힘 k=0,1,2 사용 가능. 손끝 강압 + 내측 링크 스침이 2링크 wrap 로 채점됨. 상대 비율, 모든 손가락 동일 식, 손가락 이름 없음, 힘이 0에 가까우면 링크 수로 매끄럽게 복귀 | prompt.md:6784-6791, 6800-6807 |
| DEPTH_SHARE_FULL 0.5, DEPTH_TIP_FLOOR 0.3, DEPTH_FORCE_REF 0.3 N | 값 자체는 설계 선택. 손가락을 말았을 때 내측 비율 0.5 도달 가능성은 (가설) | (추정) |
| env_mult 가 곱해지는 lift/rise/goal/progress/hold/success | round 16 구조 유지 지시, 5지 1.0 / 4지 깊은 wrap 0.65-0.75 / 손끝 5지 pinch < 4지 wrap 목표 유지 | prompt.md:185-190, 6793-6796 |
| pen_squeeze (P_SQUEEZE, SQ_TOTAL, SQ_LINK 불변) | pen_squeeze -0.17 -> -0.255, link_force_mean 3.36 -> 4.95 N 상승. 완화 금지 | prompt.md:6780-6782, 6809-6810, 6832, 6865 |
| reach (평거리, open_mul, x(1-held)) | 접근은 평거리로 항상 이익, open 은 접근 중에만, 상한 약 0.6 | prompt.md:115, 126, 150-151, 164 |
| palm_in | 0.08-0.02 m 급경사, 근접만으로는 지급 안 함(손가락 engagement 결합) | prompt.md:116, 165 |
| close, side-wall 가중 | 표면 근처에서만 닫기 보상, 가까이서만 벽 선호(floor 포함) | prompt.md:115, 139, 152-154 |
| grasp_count / grasp_digit / grasp_palm / grasp_oppose | 합 1.2, 개수 사다리 가파르게, 하위 단은 palm 없이, 손가락별 링크 수 | prompt.md:117, 127, 139 |
| off_table 게이트 | 후반부 전체를 상승 높이의 매끄러운 함수로 곱함 | prompt.md:119-121 |
| lift_height / lift_rise | 약 2.5, 목표 높이까지 상승, 내려놓고 다시 들어 올려도 lift 수입 재획득 불가 | prompt.md:122, 128, 137 |
| goal_reach / goal_progress | 약 2.5, 0-0.30 m 전 구간 + 단계 간 goal_dist 감소 | prompt.md:129 |
| goal_hold | 약 10, 허용오차 2배 안에서 큰 비율, dwell 로 증가, 실제 hold 게이트 | prompt.md:130, 135 |
| success_bonus | 첫 성공만 크게 | prompt.md:131 |
| upright 곱 | lift/transport/hold 에 곱 | prompt.md:133 |
| pen_drop / pen_setdown | 지속 래치 비례 금지, 낙하 중/하강 중만 한정 비용 | prompt.md:172-175 |
| pen_stall | 물체에 닿아 있을 때만, 테이블 위 파지는 시간이 갈수록 불리 | prompt.md:123, 155-156 |
| pen_drift / pen_descend | 목표에서 이탈은 hold 수입보다 비싸게, 내리는 것은 정지보다 불리 | prompt.md:135, 137 |
| pen_table | 손/손바닥을 테이블에 누르지 않기 | prompt.md:137 |
| pen_arm_vel / pen_arm_fast / pen_cup_vel / pen_action | 빠른 팔/물체 움직임에 작은 벌점, lift 와 carry 가 항상 이김 | prompt.md:137 |
| pen_tilt / pen_shove | round 16 그대로 유지(변경 금지 지시) | prompt.md:6812 |
| 표면 필드(palm_surface_*, link_surface_*) 계속 사용 | 클래스 목록(41-96)에는 없음. 환경 제공 명시, round 7-16 Traceback 없이 실행, "Nothing else changes" | prompt.md:199, 2444-2448, 2712-2714, 6767-6768, 6812 |
| success tolerance 재시작 | 보상이 아니라 환경 문제라서 반영하지 않음 | prompt.md:6776-6778, 6812 |

Prompt file: /home/user/rl_ws/hdgp/reward_gen/cup_pick_l_manip/iter_17/prompt.md (read to its end, line 6880).