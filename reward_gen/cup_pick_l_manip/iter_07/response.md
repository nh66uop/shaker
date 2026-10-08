## 1. What the task means, and what was missing or wrong

**The task.** One left OpenArm with a DG-5F hand has to pick up an object and hold it at a goal 0.21-0.28 m above where it started. The goal must be held upright and still until the environment counts `max_successes`. A success needs `goal_dist <= success_tol` held for `success_hold_steps` steps, with the fingers wrapped around the object on that step. The starting policy (`cup_pick_l_approach_e3400`) already reaches the object and closes an envelope (palm plus thumb, index, middle and pinky) in about 87 % of episodes, and lifts in 19 %. So this reward has three jobs: keep reaching and closing worth doing, make the grasp stronger (add the ring finger, keep the thumb opposed), and make lift, carry and hold clearly the most profitable things the robot can do.

The robot goes through these stages. It never switches between them in the reward; they follow one after another as smooth state quantities change.
1. **Reach**: move the palm from 0.15-0.49 m away until it rests on the object's surface, facing it.
2. **Close**: flex the fingers, with more payoff the nearer the palm and the finger links are to the surface.
3. **Grasp**: palm contact, then the thumb and fingers touching from opposite sides, then all five digits.
4. **Lift**: raise the object above its spawn height while it is held.
5. **Carry and hold**: bring it to `goal_pos`, keep it upright and still, and collect successes.

**What went wrong in rounds 1-6.**
- **The reward was keyed on the environment's latches.** `approach_done` and `envelope_done` are hand-written templates of one grasp on one cylinder. They are not in the observation and they stay set once set. When the policy satisfied the approach template (rounds 2-5, approach 0.91-0.96), it learned to stand still with an open hand. When the reward pushed contact (round 6), contact happened outside the template and the approach latch collapsed. The envelope latch was about 0 in every round. **Fix:** no latch appears in this reward, and neither do cup_radius, cup_axis or cup_half_height. Everything is computed from the measured surface distances, directions and normals, the contact forces and the object pose.
- **Position terms saturated without contact** (flat hand, loose claw, seated palm). **Kept rule:** no term may sit near its maximum while contact is zero. Reach is multiplied by `(0.4 + 0.6*grip)`. Closing onto nothing is capped well below one real digit contact.
- **Two digits were enough to plateau** (thumb plus pinky, round 6). **Fix:** digit contact is scored on the digits sorted by strength, with rank weights 0.05 / 0.10 / 0.20 / 0.30 / 0.35. Two digits earn 0.15 of that term. The fourth digit adds 0.30 and the fifth 0.35. The ranking is symmetric, so it does not care which digit arrives last. A per-digit mean on top pays index, middle and ring 1.1 each, against 0.9 for the thumb and 0.8 for the pinky.
- **Fingertip pokes from beside the object paid.** **Fix:** every contact income is multiplied by `cq = 0.15 + 0.85*seat`. `seat` is palm contact, or the palm within 2 cm of the surface and facing it. A poke with the palm away from the surface earns 15 % of the contact value.
- **Opposition was measured on cylinder geometry.** **Fix:** it now uses the surface normals at the touching links. The contact-weighted thumb normal is compared with the contact-weighted finger normal, and the palm normal with the finger normal. Opposing normals mean opposition.
- **Lift was gated on digit contact alone** (round 1) and later on the latch. **Fix:** `held` = at least 4 digits (smooth, full at 4) x palm contact (0.4 floor, so the flicker in palm contact force does not zero it) x opposition (0.5 floor). Lift, rise, goal and hold are all gated on it.
- **Lift, goal and hold were never reached in six rounds.** Their structure is kept: height above spawn, a rise bonus only before `lifted`, a descend penalty, hold at the goal worth more than re-lifting, and a success bonus. The weights are raised so that a held object at the goal pays about 3x the closed grasp on the table. I added a set-down penalty, so putting the object back on the table after `lifted` cannot earn the lift income again.

**Approximate per-step ladder** (seat, contacts and opposition near their best values):

| State | Reward per step |
|---|---|
| Travelling | 0.1-0.3 |
| Palm seated, fingers open | about 0.35 |
| Palm seated, fingers curled near the surface, no contact | about 0.9 |
| Two digits plus palm | about 1.9 |
| Four digits plus palm, opposed (the starting policy's grasp) | about 5.3 |
| Five digits | about 6.3 |
| Lifted 2 cm | about 8.1 |
| Lifted to goal height | about 11 |
| Still at the goal | about 17, plus 10 per counted success |

The grasp on the table also pays `-P_STALL * episode_progress`, so parking there loses value as the episode goes on.

## 2. Reward function

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
    # WEIGHTS  (no stage switch: every term is a smooth function of the current state)
    # Ladder (per step): travel 0.1-0.3 < seated open 0.35 < curled near 0.9 < 2 digits 1.9
    #   < 4 digits opposed 5.3 < 5 digits 6.3 < lifted 2 cm 8.1 < at goal height 11
    #   < still at goal 17 (+10 per success). The closed grasp on the table is ~1/3 of the goal.
    # =============================================================================================
    W_REACH = 0.50        # palm toward the surface, facing it; x(0.4+0.6*grip): <=0.2 with no contact
    W_CLOSE = 0.60        # measured finger flexion x link nearness x palm seat (closing onto nothing
                          #   is capped at 0.6, well under one real contact)
    W_COUNT = 2.50        # digits in contact, SORTED rank weights: 2 digits 0.38, 4th +0.75, 5th +0.88
    W_DIGIT = 0.80        # per-digit contact mean; index/middle/ring weighted 1.1, thumb 0.9, pinky 0.8
    W_LINKS = 0.40        # more links of each digit on the object (wrapping, not a fingertip)
    W_PALM = 0.60         # palm contact, mostly together with digits (palm alone 0.15)
    W_OPPOSE = 1.00       # thumb vs fingers and palm vs fingers pushing from opposite sides (normals)
    W_LIFT = 5.00         # object height above its own spawn height, while held
    W_RISE = 0.80         # upward object speed while held, only until `lifted` latches (no re-farming)
    W_GOAL = 3.00         # held object closing on goal_pos
    W_HOLD = 5.00         # held, at the goal, still and upright -> worth more than re-lifting
    W_SUCCESS = 10.0      # bonus on the step the environment counts a success
    P_DESCEND = 1.00      # lowering the held object < keeping it still < raising it
    P_STALL = 1.00        # closed grasp parked on the table, growing with episode progress
    P_SETDOWN = 1.50      # object back on the table after it was lifted (no second lift income)
    P_DROP = 1.00         # lifted and then no longer held
    P_TILT = 0.60         # keep the object upright
    P_SHOVE = 0.50        # pushing / tipping the object around while it is still on the table
    P_SQUEEZE = 0.80      # contact force beyond what holding needs
    P_TABLE = 0.50        # palm / finger links driven into the table
    P_ARM_VEL = 0.12      # smoothness: <=~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.15      # linear object speed while held (a 5 cm/s lift costs ~0.015)
    P_CUP_SPIN = 0.08     # angular object speed while held
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512-point sampling)
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)

    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)

    # movable finger joints (ranges <= 0.05 rad are locked; their normalisation is noise)
    # locked = thumb_2(0), index_1(3), middle_1(7), ring_1(11), pinky_1(15), pinky_2(16)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0      # thumb_3, thumb_4
    digit_joint[1, 4:7] = 1.0      # index_2, _3, _4
    digit_joint[2, 8:11] = 1.0     # middle_2, _3, _4
    digit_joint[3, 12:15] = 1.0    # ring_2, _3, _4
    digit_joint[4, 17:19] = 1.0    # pinky_3, pinky_4
    n_digit_joint = digit_joint.sum(-1)                                        # (5,)

    # =============================================================================================
    # CONTACT
    # =============================================================================================
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5) digit touches at all
    digit_links = link_c.mean(dim=-1)                              # (N,5) how many of its links
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)                              # (N,) 0..5
    finger_any = digit_touch[:, 1:].amax(dim=-1)                   # (N,) any of the four fingers
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    # =============================================================================================
    # PALM relative to the REAL surface (object-agnostic)
    # =============================================================================================
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    # facing: palm normal points at the nearest surface point AND against the outward normal there
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / 0.02)   # steepest in the last 2 cm
    # seat: the palm is on the object (contact) or within ~2 cm of it and facing it
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face), palm_c)
    # contact quality: a fingertip poke with the palm away from the surface earns 15 %
    cq = 0.15 + 0.85 * seat

    # ---- Reach: long-range pull (0.2 m scale covers the 0.15-0.49 m spawn range) + final 2 cm ----
    reach_far = (1.0 - torch.tanh(palm_d / 0.20)) * (0.4 + 0.6 * face)
    reach_near = palm_near * face
    # no term near its maximum without contact: 40 % of it at zero grip
    r_reach = W_REACH * (0.6 * reach_far + 0.4 * reach_near) * (0.4 + 0.6 * grip)

    # =============================================================================================
    # CLOSE: measured flexion, worth more the closer the palm and the links are to the surface
    # =============================================================================================
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)          # saturates at half flexion: no crushing drive
    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)       # (N,5,3)
    near_digit = link_near.amax(dim=-1)                                                      # (N,5)
    # a straight finger lying along the object gets nothing (flex=0); a curled finger far away
    # gets little (near~0); only curling ONTO the surface with the palm seated pays.
    r_close = W_CLOSE * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # GRASP: digits (counted separately, steep in the count), links, palm, opposition
    # =============================================================================================
    # sorted rank weights: the k-th strongest digit earns rank_w[k]. Symmetric in the digits, and the
    # LAST digits to arrive carry the most: 2 digits = 0.15, 3 = 0.35, 4 = 0.65, 5 = 1.00.
    rank_w = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.35], device=device, dtype=dtype)
    touch_desc, _ = torch.sort(digit_touch, dim=-1, descending=True)
    r_count = W_COUNT * cq * (touch_desc * rank_w).sum(dim=-1)

    # per-digit mean: index, middle and ring weighted at least as much as thumb and pinky
    digit_w = torch.tensor([0.9, 1.1, 1.1, 1.1, 0.8], device=device, dtype=dtype)
    digit_score = 0.6 * digit_touch + 0.4 * digit_links                                      # (N,5)
    r_digit = W_DIGIT * cq * (digit_score * digit_w).sum(dim=-1) / digit_w.sum()

    # more links per digit on the object = wrapping along the body, not a fingertip
    r_links = W_LINKS * cq * digit_links.mean(dim=-1)

    # palm contact counts mostly together with digits (palm alone = 0.25 of it, round-2 lesson)
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))

    # opposition from the surface normals at the touching links (opposite normals = opposite sides)
    normals = ctx.link_surface_normal                                        # (N,5,3,3)
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))           # (N,3)
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))   # (N,3)
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = 0.65 * thumb_opp + 0.35 * palm_opp                          # (N,) in [0,1]
    r_oppose = W_OPPOSE * opposition

    # "held" = palm contact plus opposing digits, at least four digits (smooth, full at 4).
    # Floors on palm (0.4) and opposition (0.5) keep force/normal flicker from zeroing a real grasp.
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)

    # =============================================================================================
    # LIFT
    # =============================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    on_table = 1.0 - torch.tanh(height / 0.02)
    # steep over the first 3 cm (breaks the table habit), still rising all the way to goal height
    lift_shape = 0.5 * torch.tanh(height / 0.03) + 0.5 * torch.tanh(height / 0.15)
    r_lift = W_LIFT * held * lift_shape
    vz = ctx.cup_lin_vel[:, 2]
    # upward speed pays only until `lifted` latches: putting the object down and raising it again
    # cannot re-earn it
    r_rise = W_RISE * held * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)
    # lowering the held object costs: descending < still < raising
    p_descend = P_DESCEND * held * torch.tanh((-vz).clamp(min=0.0) / 0.10)
    # a closed grasp parked on the table is not worth staying for (free early, costly late)
    p_stall = P_STALL * held * on_table * ctx.episode_progress
    # object back on the table after a lift, and a lifted object no longer held
    p_setdown = P_SETDOWN * lifted * on_table
    p_drop = P_DROP * lifted * (1.0 - held)

    # =============================================================================================
    # GOAL: carry, then rest there upright and still
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    upright = torch.exp(-ctx.cup_tilt / 0.20)
    r_goal = W_GOAL * held * (1.0 - torch.tanh(ctx.goal_dist / 0.15))
    near_goal = 1.0 - torch.tanh(ctx.goal_dist / 0.04)          # fixed scale, not success_tol
    still = torch.exp(-cup_speed / 0.05)
    r_hold = W_HOLD * held * near_goal * still * upright        # resting at the goal > re-lifting
    r_success = W_SUCCESS * success

    # =============================================================================================
    # SAFETY AND SMOOTHNESS (small next to the stage income)
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    p_shove = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                          + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - 15.0).clamp(min=0.0) / 15.0)
                                   + torch.tanh((max_link_force - 6.0).clamp(min=0.0) / 6.0))
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = P_TABLE * (palm_pen + floor_pen)
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * held) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = held * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                        + P_CUP_SPIN * torch.tanh(ctx.cup_ang_vel.norm(dim=-1) / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_reach + r_close
        + r_count + r_digit + r_links + r_palm + r_oppose
        + r_lift + r_rise
        + r_goal + r_hold + r_success
        - p_descend - p_stall - p_setdown - p_drop
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    )

    components = {
        "reach": r_reach,
        "close": r_close,
        "grasp_count": r_count,
        "grasp_digit": r_digit,
        "grasp_links": r_links,
        "grasp_palm": r_palm,
        "grasp_oppose": r_oppose,
        "lift_height": r_lift,
        "lift_rise": r_rise,
        "goal_reach": r_goal,
        "goal_hold": r_hold,
        "success_bonus": r_success,
        "pen_descend": -p_descend,
        "pen_stall": -p_stall,
        "pen_setdown": -p_setdown,
        "pen_drop": -p_drop,
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
