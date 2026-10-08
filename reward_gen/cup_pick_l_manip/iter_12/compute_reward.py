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

    ROUND 12: this factor is NEVER applied to the long-range approach. From 0.15-0.49 m away the
    nearest surface point of an open cup is its rim, so multiplying the approach by it (round 11)
    paid ~0 for closing in and both arms walked away. It is used only from a few centimetres in,
    always with a floor, which is where the rim-versus-wall choice is actually made."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Object-agnostic and arm-agnostic: surface distances/normals, contact forces, object
    # pose/velocity, goal. No approach/envelope latch, no radius/axis/half-height, no world-frame
    # direction constant (only normal z components and heights, identical under the mirror).
    #
    # Per-step ceilings -- the budget IS the design:
    #   reach            0.60   pre-contact only, x (1 - held); it vanishes once the object is held
    #   keep the grasp   1.20   links-per-digit + count ladder (top rungs need the palm) + palm + opposition
    #   lift             2.50   x uprightness x real hold, rising to the height the goal needs
    #   transport        2.50   1.60 on goal_dist over 0-0.30 m + 0.90 on the signed RATE it falls
    #   hold at rest    10.00   real hold, upright, at the goal, still, growing with dwell
    #   first success   60.00   one-off; later successes 2.0, so staying pays, not re-arriving
    #
    # TWO STRUCTURAL RULES, both of them scars:
    #  (a) Everything from "transport" down is multiplied by `off_table`, a smooth function of the
    #      object's rise above its OWN spawn height: the second half of the task pays exactly ZERO
    #      while the object sits on the table (round 10 collected 0.54-0.69/step of hold income
    #      without ever lifting).
    #  (b) Stage one must ALWAYS be profitable. No factor may zero the long-range approach, and no
    #      penalty may exceed the income of the stage the policy is in -- round 11 broke both and
    #      both arms retreated from the object instead of approaching it.
    #
    # Steady states: grasp parked on the table ~1.2 early, ~0.5 late (minus the time pressure);
    # lifted and hovering ~4.5; at rest on the goal ~13; a success spike of 60.
    # =============================================================================================
    W_REACH = 0.38        # palm towards the surface and facing it (PLAIN distance)  } 0.60 total,
    W_CLOSE = 0.22        # measured flexion x link nearness to a wall x palm seat   } x (1 - held)
    W_COUNT = 0.40        # digit-count ladder, sorted; the top rungs need the palm  }
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike         } 1.20 total:
    W_PALM = 0.30         # palm contact (2x round 10: a pinch must not score alike) } build and keep
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)         } a grasp, not live on it
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height }
    W_RISE = 0.40         # upward speed, only until `lifted` latches                         } 2.50
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m        } 2.50 total,
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls                     } x off_table
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode (~5 steps of a perfect hold)
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 8.00         # lifted and no longer held: on top of losing up to 10 of hold income
    P_SETDOWN = 4.00      # back on the table after a lift: lift income is never re-earnable
    P_STALL = 0.40        # time pressure -- ROUND 12: gated on being ON the object, and halved.
    #                       Ungated at 0.80 it was the largest term in the whole reward for policies
    #                       that had never touched anything, and they learned to walk away.
    P_DRIFT = 1.50        # moving AWAY from the goal near it: drifting off must cost more than it saves
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
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link (contact geometry is noisier)
    TABLE_RISE = 0.04     # smooth "off the table" scale: 0 at 0, 0.46 at 2 cm, 0.96 at 8 cm
    TILT_DEAD = math.radians(5.0)    # a few degrees of wobble is not a tilted carry
    TILT_SCALE = math.radians(10.0)  # 15 deg -> x0.37, 30 deg -> x0.08 of lift/transport/hold
    SQ_TOTAL = 40.0       # measured: a grip that holds fine paid -0.45/step at a 15 N threshold
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
    # HEIGHT: the spine of this reward. Everything in the second half is multiplied by `off_table`.
    # =============================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)      # rise above ITS start
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)    # read from the goal
    off_table = torch.tanh(height / TABLE_RISE)     # smooth, exactly 0 on the table
    on_table = 1.0 - off_table

    # uprightness: ONE multiplier on every lift / transport / hold income. An additive tilt penalty
    # does not keep the object upright; multiplying does (tilt fell 28 deg -> 5 deg when it did).
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

    # ---- WHERE the links touch: a link on a vertical WALL counts fully, one hooked over a rim or
    # lying on a lid counts 0.35. A link in contact is close in by definition, so this is exactly
    # the "only from close in" rule. Used for SCORING only; `digit_touch` / `held` stay on the plain
    # force, so geometry noise can never dissolve a real hold when the wrench arrives.
    link_side = _sidewall(ctx.link_surface_normal[..., 2])                          # (N,5,3)
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side             # (N,5,3)
    link_cw = link_c * link_side_w                                                  # (N,5,3)
    # NUMBER OF LINKS of each digit, made convex so one link is worth far less than two or three:
    # 1 wall link -> 0.19, 2 -> 0.54, 3 -> 1.00; a single fingertip on the rim -> 0.04.
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- palm against the object's real surface ------------------------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    # facing: the palm normal points at the nearest surface point AND against the outward normal
    # there. Both are dot products of context vectors, so nothing is tied to a world axis.
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])           # 1 at a wall, ~0 at a rim/lid
    # the side-wall preference EXISTS ONLY CLOSE IN: 1.0 at 3 cm, 0.37 at 6 cm, 0.05 at 12 cm, and
    # even at full strength a rim keeps SIDE_FLOOR_PALM of the value.
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / 0.02)
    # seating means: close to the surface, facing it, and the surface there is a WALL. Real palm
    # contact always counts in full, whatever the geometry read says.
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    # contact quality: a fingertip poke with the palm off the object is worth 15 % of a seated grasp
    cq = 0.15 + 0.85 * seat

    # ---- opposition from the surface normals at the touching links ------------------------------
    # opposite outward normals = contacts on opposite sides of the body, whatever its shape.
    normals = ctx.link_surface_normal                                        # (N,5,3,3)
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))           # (N,3)
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))   # (N,3)
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)             # (N,)

    # "held" = palm plus several OPPOSING digits (smooth, full at four). The palm share is raised
    # over round 10 (0.4+0.6 -> 0.3+0.7): a three-digit pinch with palm 0.001 collects only ~0.3 of
    # the lift income, so seating the palm is worth acquiring on BOTH arms.
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)                      # (N,)
    # the 10/step asks for more than the lift does: palm and opposition enter a SECOND time, so a
    # fingertip touch can never collect the dominant income even under the wrench.
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)

    # =============================================================================================
    # REACH -- 0.60 total with `close`, fading out once the object is really held.
    # ROUND 12 FIX: the far part is shaped on the PLAIN palm-to-surface distance. No side-wall
    # factor, no late-episode decay, no gate of any kind: from random weights this is the only
    # income that exists, and it must never be switchable off. (Round 11 multiplied it by
    # (1 - |normal_z|), which is ~0 at the rim of an open cup seen from above; approaching stopped
    # paying and both arms retreated to 0.36-0.48 m.)
    # =============================================================================================
    fade = (1.0 - held).clamp(min=0.0, max=1.0)
    reach_far = (1.0 - torch.tanh(palm_d / 0.20)) * (0.4 + 0.6 * face)   # 0.20 m covers 0.15-0.49 m
    # the last couple of centimetres MAY prefer the wall -- with a floor of 0.3, so even a pure rim
    # read keeps a third of the value and the pull towards the object never disappears.
    reach_near = palm_near * face * (0.3 + 0.7 * palm_side)
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near) * (0.45 + 0.55 * grip)

    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)        # saturates at half flexion: no crushing drive
    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)      # (N,5,3)
    # nearness is scored where the link would land: a link a centimetre off a WALL counts fully, one
    # a centimetre off the rim face counts a quarter. This is a close-in term by construction.
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)                       # (N,5)
    # a straight finger lying along the object gets nothing (flex = 0), a curled finger in the air
    # gets nothing (near = 0), a fingertip hooked over the rim gets a quarter: only curling ONTO a
    # wall with the palm seated pays.
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total. Every digit pays alike; two digits are worth far less than four.
    # =============================================================================================
    # sorted rank weights (sum 1.0): 2 digits 0.18, 3 = 0.42, 4 = 0.80, 5 = 1.00. The digits that
    # arrive LAST carry the most, so a two-digit pinch is worth under a quarter of a real envelope.
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w                                            # (N,5)
    # the TOP rungs (the 4th and 5th digit, 0.58 of the ladder) need the palm: a pinch cannot climb
    # to the top of the ladder however many fingertips it lands.
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.3 + 0.7 * palm_c))

    # per-digit mean of the LINK COUNT, every digit weighted alike (no digit-specific term)
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)

    # palm contact is worth having (0.30 of 1.20, double round 10), but it still counts only
    # together with digits: on its own it funded a flat open hand pressed against the object.
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # A closed grasp parked on the table must become actively unattractive as the episode goes on:
    # the on-table GRASP income decays to 45 %. ROUND 12: the decay no longer touches `r_reach`.
    # Decaying the approach income was the second way round 11 made stage one unprofitable late in
    # an episode. Releasing a decayed grasp (>= 0.5) to collect reach income (~0.13 at contact
    # distance) is still a clear loss, so "letting go must never pay" survives the change.
    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_oppose = r_oppose * table_decay

    # =============================================================================================
    # LIFT -- 2.50 total, the largest single step up in income and the ONLY door to the goal terms.
    # x uprightness, x a real hold, capped at the height the goal needs so there is nothing to gain
    # from lifting higher instead of moving to the goal.
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)     # steep first 3 cm
    r_lift = W_LIFT * held * upright * lift_shape
    vz = ctx.cup_lin_vel[:, 2]
    # upward speed pays only until `lifted` latches, so setting the object down and raising it again
    # can never re-earn the lift income
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, ALL of it x off_table: nothing here exists while the object rests.
    # =============================================================================================
    # a CONSTANT slope over the whole 0-0.30 m range (a narrow ramp left the landscape flat exactly
    # where the policy sat), plus a fine ramp for the last few centimetres
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)

    # paying for the REDUCTION of goal_dist between steps. The function must be pure, so the rate is
    # read from the state: the closing speed towards the goal is d(goal_dist)/dt up to the rotation
    # part, which uprightness already handles. SIGNED, so drifting away costs what approaching pays
    # and an out-and-back orbit nets negative (tanh is concave). Faded inside ~3 cm so that jitter at
    # rest, and the environment's pushes, neither pay nor punish.
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10))

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, the dominant income, also x off_table
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    # the paying region is scaled by the CURRENT tolerance so it always covers where the curriculum
    # is, and it is wide: 0.56 at one tolerance and 0.41 at two, so a large fraction is already
    # collectable before the success condition can be met. The fixed fine ramp keeps the last
    # centimetres worth crossing as the tolerance shrinks towards 0.015 m.
    tol_wide = (5.0 * ctx.success_tol).clamp(min=0.08)
    near_goal = (0.65 * (1.0 - torch.tanh(ctx.goal_dist / tol_wide))
                 + 0.35 * torch.exp(-ctx.goal_dist / 0.05))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    # 0.30 + 0.70*still, not `still` alone: the environment pushes and twists the object at the goal,
    # so holding station against a push must stay profitable while drifting off loses all of it.
    settle = 0.30 + 0.70 * still
    # dwell: episode_progress restarts after every success, so at the goal it measures how long the
    # object has been standing there. Settling therefore pays strictly more than arriving.
    dwell = 0.55 + 0.45 * torch.tanh(ctx.episode_progress / 0.25)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell

    # the FIRST success of an episode is the big one; later ones decay hard, because the episode
    # counts up to five at a goal that does not move and the income for staying must come from
    # r_hold, not from re-arriving. The factor is the mean of the two possible conventions for
    # num_successes (whether it already includes this step's success), so it decays either way.
    n_succ = ctx.num_successes.to(dtype)
    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- each priced above the income it replaces
    # =============================================================================================
    p_drop = P_DROP * lifted * (1.0 - held)             # 8.0 on top of losing up to 10 of hold
    p_setdown = P_SETDOWN * lifted * on_table           # no second lift income after a set-down
    # TIME PRESSURE, ROUND 12. Gated on the hand actually being ON the object: `grip` needs contact,
    # `held` needs palm + opposing digits. A policy that has not reached the object pays NOTHING --
    # round 11 charged it 0.28-0.32/step for existing, which was the largest term in the reward and
    # taught both arms to retreat. At 0.40 it also stays below the ~1.2 of grasp income that letting
    # go would forfeit, so it makes parking costly without ever making release attractive.
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    # drifting away from the goal once near it costs more than the hold income it gives up
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    # lowering a held object costs, but only BELOW goal height, so descending onto the goal from
    # above is not taxed: descending < still < rising.
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
    # object motion while held: 0.02 at 0.1 m/s against +0.7 of transport income at the same speed,
    # so lifting and carrying always beat standing still
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
        - p_drop - p_setdown - p_stall - p_drift - p_descend
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
