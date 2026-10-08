# 1. What the task means, and which step was missed

## What the policy already is

The weights this reward starts from (round 8, epoch 3200) already solve four of the five things a pick-and-hold
needs. The hand reaches the object, closes a five-digit grasp with palm contact (thumb 0.79, index 0.77, middle 0.76,
ring 0.40, pinky 0.77, palm 0.64), lifts in 0.89 of episodes, and carries the object upright (tilt 4.3-6.7 degrees).
The one thing it has never done, in either round of the object-agnostic reward, is **stop**. The goal keypoint
distance has been flat at 0.163-0.174 m for 1,500 epochs while the tolerance shrank to 0.066 m, the near-goal step
fraction is 0.019-0.050, and the recording shows the cup being raised, lowered and raised again for nine seconds
without ever coming to rest.

## Which step is missed or wrong

The task is not "grasp, lift, carry". It is **grasp -> lift -> transport -> arrest -> defend**. Two of those five
steps were never priced.

1. **Arrest.** The reward paid for *being* near the goal and never for *getting* nearer. A level-only term gives a
   policy no reason to convert velocity into position: at 0.17 m the fine ramp (0.05 m wide) contributed
   ~0.001/step, so the landscape between the stall point and the tolerance was flat, and the value of arriving was
   indistinguishable from the value of passing through. This is why successes were counted "in passing": the policy
   sweeps the goal region while carrying and collects the success as a side effect.

2. **Defend.** Nothing made holding position under an external push more valuable than drifting. The hold term
   required stillness multiplicatively, which under the new disturbance would actually *punish* the correction the
   task wants.

And one thing was outright wrong:

3. **The budget.** 92 % of the per-step income was earned before the object ever arrived. Grasping paid 3.86/step,
   lifting 2.25/step, everything about the goal 1.04/step. Holding the object up *anywhere* was worth 6 times
   finishing the task, so hovering was not a failure of exploration - it was the optimum of this reward. No reshaping
   of the goal term can fix that while the standing income is larger than the terminal income.

## The stages, and the ceiling each one gets

Every term below is a smooth function of the current state - no latch, no stage flag, no pose template, no cylinder
parameters. What changes between stages is only *how much a state is worth*, and the ordering is strict:

| stage | what the state is | per-step ceiling |
|---|---|---|
| reach / close | hand approaching, fingers curling onto the surface | 0.30, and x(1-held) so it vanishes once the object is held |
| keep the grasp | digits in contact, count, palm, opposition | 0.80 |
| lift | object above spawn height, x uprightness, **x horizontal closeness to the goal** | 1.00 |
| transport | goal_dist over 0-0.30 m, **plus the rate at which goal_dist falls** | 2.50 (1.70 + 0.80) |
| arrest and defend | held, upright, at the goal, still | **10.00** |
| success | the environment counts one | 100.0 one-off |

Hovering upright with a good grasp 0.15-0.20 m from the goal is worth about 1.3/step (0.80 grasp + ~0.2 lift +
~0.35 transport). Sitting at the goal is worth about 14/step. The terminal state is now eleven times the hovering
state instead of 0.4 times it, and one second of hovering (78) is worth less than a single success bonus (100).

Three structural changes carry this:

- **The lift is multiplied by horizontal closeness to the goal** (`exp(-d_xy/0.12)`). The goal is at most 5 cm from
  the spawn point horizontally, so carrying the object away towards the robot - exactly what the recording shows -
  cuts the lift income to a fifth. Height alone is no longer a source of income.
- **Transport pays for the derivative.** The function must be pure, so the per-step reduction of `goal_dist` is
  computed from the state as the closing speed `v . e_goal`, which is that reduction divided by dt. It is *signed*:
  receding costs what approaching pays, so an out-and-back cycle nets negative (tanh is concave, so the fast return
  recovers less than the slow departure lost) and the term cannot be farmed by orbiting the goal.
- **The hold is gated on a real hold and tolerant of disturbance.** It needs four opposing digits *and* the palm
  (`held` squared through its palm and opposition factors), so a two-digit pinch collects nothing; but stillness
  enters as `0.3 + 0.7*still`, so fighting an external push at the goal still earns most of the income, while
  drifting off loses all 10 of it plus the 6.0 drop penalty. Losing the object costs ~16/step against a hold income
  of 10/step.

Uprightness stays a multiplier on lift, rise, transport and hold - that single change took tilt from 28 degrees to 5
and is not touched. Lift, carry and hold penalties (descend, stall, setdown, drop, squeeze, arm speed, object speed)
are kept small: at 0.1 m/s the transport term pays 0.61 and the speed penalties cost 0.02, so carrying is always
better than standing still.

# 2. Code

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


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.cup_pos.device
    dtype = ctx.cup_pos.dtype

    # =============================================================================================
    # WEIGHTS -- the budget is the point of this round.
    # Object-agnostic throughout: only surface distances/normals, contact forces, object pose and
    # velocity, and the goal. No approach/envelope latch, no radius/axis/half-height, no template.
    #
    # Per-step ceilings (the previous round earned 92 % of its income before arriving):
    #   reach + close   0.30   and x(1-held): it disappears once the object is really held
    #   keep the grasp  0.80   contact per digit + count + palm + opposition
    #   lift            1.00   x uprightness x HORIZONTAL closeness to the goal, capped at goal height
    #   transport       2.50   1.70 on goal_dist over 0-0.30 m + 0.80 on the RATE goal_dist falls
    #   hold at rest   10.00   held, upright, at the goal, still -- the dominant income
    #   success       100.0    one-off; one second of hovering (60 steps x ~1.3) is worth ~78
    #
    # Steady states: hovering upright 0.17 m away ~1.3/step; at rest on the goal ~14/step.
    # =============================================================================================
    W_REACH = 0.18        # palm towards the surface and facing it            } 0.30 total,
    W_CLOSE = 0.12        # measured flexion x link nearness x palm seat      } x (1 - held)
    W_COUNT = 0.25        # digits in contact, sorted rank weights            }
    W_DIGIT = 0.30        # per-digit contact mean, all five weighted alike   } 0.80 total:
    W_PALM = 0.12         # palm contact, mostly together with digits         } enough not to
    W_OPPOSE = 0.13       # thumb/palm vs fingers from opposite sides         } forget the grasp
    W_LIFT = 0.80         # height above spawn, capped at what the goal needs }
    W_RISE = 0.20         # upward speed, only until `lifted` latches         } 1.00 total
    W_GOAL = 1.70         # goal_dist, shaped over the whole 0-0.30 m range   }
    W_PROGRESS = 0.80     # the rate at which goal_dist falls (signed)        } 2.50 total
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held
    W_SUCCESS = 100.0     # a success is worth more than a second of hovering

    P_DROP = 6.00         # lifted and no longer held: losing it >> the hold income it replaces
    P_SETDOWN = 3.00      # back on the table after a lift (no second lift income)
    P_STALL = 1.00        # closed grasp parked on the table, growing with episode progress
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still
    P_TILT = 0.30         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    P_SQUEEZE = 0.50      # force beyond what holding needs (round-7 thresholds, round-8 weight)
    P_TABLE = 0.50        # palm / finger links driven into the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: 0.02 at 0.1 m/s vs +0.61 of transport income,
    P_CUP_SPIN = 0.06     #   so carrying to the goal always beats standing still
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    TILT_DEAD = math.radians(5.0)    # a few degrees of wobble is not a tilted carry
    TILT_SCALE = math.radians(10.0)  # 15 deg -> x0.37, 30 deg -> x0.08 of lift/transport/hold
    SQ_TOTAL = 40.0       # measured: a grip that works paid -0.45/step at a 15 N threshold
    SQ_LINK = 15.0

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    # locked = thumb_2(0), index_1(3), middle_1(7), ring_1(11), pinky_1(15), pinky_2(16)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0      # thumb_3, thumb_4
    digit_joint[1, 4:7] = 1.0      # index_2, _3, _4
    digit_joint[2, 8:11] = 1.0     # middle_2, _3, _4
    digit_joint[3, 12:15] = 1.0    # ring_2, _3, _4
    digit_joint[4, 17:19] = 1.0    # pinky_3, pinky_4
    n_digit_joint = digit_joint.sum(-1)                                        # (5,)

    # =============================================================================================
    # CONTACT AND GRASP QUALITY
    # =============================================================================================
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5) digit touches at all
    digit_links = link_c.mean(dim=-1)                              # (N,5) how many of its links
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)                              # (N,) 0..5
    finger_any = digit_touch[:, 1:].amax(dim=-1)                   # (N,) any of the four fingers
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    # uprightness: ONE multiplier on every lift / transport / hold income. Round 8 proved this is
    # what keeps the object upright; a small additive tilt penalty does not.
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)

    # ---- palm against the object's real surface (object-agnostic) -------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / 0.02)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face), palm_c)
    # contact quality: a fingertip poke with the palm off the object is worth 15 % of a seated grasp
    cq = 0.15 + 0.85 * seat

    # ---- opposition from the surface normals at the touching links ------------------------------
    normals = ctx.link_surface_normal                                        # (N,5,3,3)
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))           # (N,3)
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))   # (N,3)
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = 0.65 * thumb_opp + 0.35 * palm_opp                          # (N,) in [0,1]

    # "held" = palm plus at least four opposing digits (smooth, full at four). Floors keep force and
    # normal flicker from zeroing a real grasp when the environment pushes the object.
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    # the hold income asks for more than the lift does: palm AND opposition enter a second time, so
    # a grip that is merely touching cannot collect the 10/step.
    hold_grip = held * (0.5 + 0.5 * palm_c) * (0.5 + 0.5 * opp_sat)

    # =============================================================================================
    # REACH AND CLOSE -- 0.30 total, and it FADES OUT once the object is held
    # (the policy already knows how to grasp; this only has to survive as a memory)
    # =============================================================================================
    fade = (1.0 - held).clamp(min=0.0, max=1.0)
    reach_far = (1.0 - torch.tanh(palm_d / 0.20)) * (0.4 + 0.6 * face)   # 0.2 m covers the spawn range
    reach_near = palm_near * face
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near)

    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)        # saturates at half flexion: no crushing drive
    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)      # (N,5,3)
    near_digit = link_near.amax(dim=-1)                                                     # (N,5)
    # a straight finger lying along the object gets nothing (flex=0), a curled finger in the air
    # gets nothing (near=0): only curling ONTO the surface with the palm seated pays.
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 0.80 total. Enough not to forget it, not enough to live on.
    # =============================================================================================
    # sorted rank weights (sum 1.0): 2 digits 0.20, 3 = 0.46, 4 = 0.85, 5 = 1.00, so the digits that
    # arrive LAST carry the most and a thumb-and-pinky pinch is worth a fifth of a real envelope.
    rank_w = torch.tensor([0.07, 0.13, 0.26, 0.39, 0.15], device=device, dtype=dtype)
    touch_desc, _ = torch.sort(digit_touch, dim=-1, descending=True)
    r_count = W_COUNT * cq * (touch_desc * rank_w).sum(dim=-1)

    # per-digit mean, every digit weighted alike (the ring finger now touches, so no special term)
    digit_score = 0.6 * digit_touch + 0.4 * digit_links                                     # (N,5)
    r_digit = W_DIGIT * cq * digit_score.mean(dim=-1)

    # palm contact counts mostly together with digits, never on its own
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # =============================================================================================
    # LIFT -- 1.00 total, x uprightness, capped at the height the goal needs, and MULTIPLIED BY
    # HORIZONTAL CLOSENESS TO THE GOAL: hovering at the right height in the wrong place earns ~0.2.
    # =============================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    on_table = 1.0 - torch.tanh(height / 0.02)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)   # from the goal itself
    h_eff = torch.minimum(height, h_need)          # nothing to gain from lifting past the goal
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)
    # the goal sits within 5 cm of the spawn point horizontally, so this factor is ~0.66 straight up
    # from the object and ~0.2 once the object has been carried 0.2 m away towards the robot.
    d_xy = (ctx.cup_pos[:, :2] - ctx.goal_pos[:, :2]).norm(dim=-1)
    goal_xy = torch.exp(-d_xy / 0.12)
    r_lift = W_LIFT * held * upright * goal_xy * lift_shape
    vz = ctx.cup_lin_vel[:, 2]
    # upward speed pays only until `lifted` latches: putting the object down and raising it again
    # can never re-earn it
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total: 1.70 on the distance, 0.80 on the RATE at which it falls
    # =============================================================================================
    # coarse ramp with a CONSTANT slope over the whole 0-0.30 m range (the previous round's fine
    # 0.05 m ramp was flat at the 0.17 m the policy sat at), plus a fine ramp for the last few cm.
    goal_far = (1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0))
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = W_GOAL * held * upright * (0.55 * goal_far + 0.45 * goal_fine)

    # paying for the REDUCTION of goal_dist between steps. The function must be pure, so the rate is
    # read from the state: the closing speed towards the goal IS d(goal_dist)/dt / dt, up to the
    # rotation part, which the uprightness factor already handles. Signed, so receding costs what
    # approaching pays and an out-and-back orbit nets negative (tanh is concave). Faded inside
    # ~3 cm so that jitter at rest, and the environment's pushes, neither pay nor punish.
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = W_PROGRESS * held * upright * approach_scale * torch.tanh(closing / 0.10)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, the dominant income of the whole reward
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    # two scales, fixed (never success_tol): arriving already pays, settling pays 3-4x more. At the
    # current tolerance (0.066) this is ~2.9/step, already twice the whole hovering income, and 10
    # once the object is on the goal.
    near_goal = 0.25 * torch.exp(-ctx.goal_dist / 0.12) + 0.75 * torch.exp(-ctx.goal_dist / 0.04)
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    # 0.3 + 0.7*still, not `still` alone: the environment pushes and twists the object at the goal,
    # and holding station against a push must stay profitable while drifting off loses all of it.
    r_hold = W_HOLD * hold_grip * upright * near_goal * (0.3 + 0.7 * still)
    r_success = W_SUCCESS * success

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING -- all worth more than the income they could replace
    # =============================================================================================
    p_drop = P_DROP * lifted * (1.0 - held)             # 6.0 on top of losing up to 10 of hold
    p_setdown = P_SETDOWN * lifted * on_table           # no second lift income after a set-down
    p_stall = P_STALL * held * on_table * ctx.episode_progress   # a closed grasp on the table
    # lowering the held object costs, but only BELOW goal height, so descending onto the goal from
    # above is not taxed: descending < still < rising.
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS (small next to the stage income)
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    p_shove = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                          + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = P_TABLE * (palm_pen + floor_pen)
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * held) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    # object motion while held: 0.02 at 0.1 m/s against +0.61 of transport income at the same speed
    p_cup_vel = held * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                        + P_CUP_SPIN * torch.tanh(spin / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_reach + r_close
        + r_count + r_digit + r_palm + r_oppose
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero  # keeps shape (N,) even if a term ever broadcasts

    components = {
        "reach": r_reach,
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
        "pen_drop": -p_drop,
        "pen_setdown": -p_setdown,
        "pen_stall": -p_stall,
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
