You are an expert in robotics, reinforcement learning and code generation.

We control one 7-DOF OpenArm robot arm (the left arm) carrying a five-finger Tesollo DG-5F hand, standing at a table. +x points from the robot toward the table. Every episode starts with the arm raised beside the robot: the palm is just outside the table edge nearest the robot, about 0.25 m above the table top, turned sideways, with the fingers pointing forward over the table edge. A cup stands upright on the table at a position drawn uniformly at random for every episode, with x between 0.10 m and 0.40 m and y between 0.00 m and 0.30 m, so at the start the palm is anywhere from about 0.15 m to 0.49 m from the cup (0.31 m on average) and the hand has to go to wherever the cup is; ctx.cup_pos gives its position. Near the far corners of that region (y close to 0.30 m, or x close to 0.40 m) the arm may need the hand turned by up to about 40 degrees from its start orientation to reach the cup's +y side. Parallel environments use different cups (open cups of several sizes and a closed shaker), so the graspable radius (44 mm to 81 mm) and half height (42 mm to 65 mm) differ between environments and are given per environment. There is only this one start pose: approach_done is never set when an episode starts. Positions are in metres in each environment's local frame, with +z pointing up; the table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (26,), float32)` with direct joint control. There is no grasp primitive, no hand synergy and no automatic finger stopping:
    actions[0:7]  = arm joint increments: each arm joint target moves by 0.05 * a rad per step and then passes a first-order filter (factor 0.1), so each arm joint moves at most about 0.3 rad/s.
    actions[7:26] = finger joint targets: each value is mapped linearly onto that joint's commandable range (a = -1 gives the lower limit, a = +1 the upper limit) and then low-pass filtered (factor 0.1). The order, which is also the order of ctx.hand_q / hand_q_norm / hand_target_norm / hand_qd, is:
    actions[ 7] = hand index  0: l_hj_thumb_2   range [+1.561, +1.581] rad  locked
    actions[ 8] = hand index  1: l_hj_thumb_3   range [-1.571, +0.000] rad  movable
    actions[ 9] = hand index  2: l_hj_thumb_4   range [-1.571, +0.000] rad  movable
    actions[10] = hand index  3: l_hj_index_1   range [-0.010, +0.010] rad  locked
    actions[11] = hand index  4: l_hj_index_2   range [+0.000, +2.007] rad  movable
    actions[12] = hand index  5: l_hj_index_3   range [+0.000, +1.571] rad  movable
    actions[13] = hand index  6: l_hj_index_4   range [+0.000, +1.571] rad  movable
    actions[14] = hand index  7: l_hj_middle_1  range [-0.010, +0.010] rad  locked
    actions[15] = hand index  8: l_hj_middle_2  range [+0.000, +2.007] rad  movable
    actions[16] = hand index  9: l_hj_middle_3  range [+0.000, +1.571] rad  movable
    actions[17] = hand index 10: l_hj_middle_4  range [+0.000, +1.571] rad  movable
    actions[18] = hand index 11: l_hj_ring_1    range [-0.010, +0.010] rad  locked
    actions[19] = hand index 12: l_hj_ring_2    range [+0.000, +1.920] rad  movable
    actions[20] = hand index 13: l_hj_ring_3    range [+0.000, +1.571] rad  movable
    actions[21] = hand index 14: l_hj_ring_4    range [+0.000, +1.571] rad  movable
    actions[22] = hand index 15: l_hj_pinky_1   range [-0.010, +0.000] rad  locked
    actions[23] = hand index 16: l_hj_pinky_2   range [-0.010, +0.010] rad  locked
    actions[24] = hand index 17: l_hj_pinky_3   range [+0.000, +1.571] rad  movable
    actions[25] = hand index 18: l_hj_pinky_4   range [+0.000, +1.571] rad  movable
Joints marked "locked" have a range of 0.05 rad or less and effectively do not move. The thumb's first joint is welded in this hand; `thumb_2`, which rotates the thumb into opposition, is locked at the opposed angle, and the thumb flexes with `thumb_3` and `thumb_4`. On the index, middle and ring fingers `_1` spreads the finger sideways (locked), `_2` flexes the knuckle, and `_3` and `_4` flex the two outer joints. On the pinky, `pinky_1` and `pinky_2` are locked and the finger flexes with `pinky_3` and `pinky_4`. On this left hand, `thumb_3` and `thumb_4` flex towards negative angles (a more negative angle is a more flexed thumb); for every other movable joint a larger angle means a more flexed (more closed) finger. ctx.hand_q_norm, hand_target_norm and hand_default_q_norm are nevertheless oriented the same way for every joint: 0 = straight, 1 = most flexed. The finger joints are position-controlled (PD), so a finger that meets the cup stops there and presses with a force that grows with the gap between its target angle and its actual angle. One control step is 1/60 s.

Now I want you to help me write a reward function for reinforcement learning.

Typically, the reward function of a manipulation task consists of these parts (some are optional — include them only if really necessary):
1. the distance between the robot's hand and the target object
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target object implied by the task
5. [optional] extra constraints on the robot implied by the task

The reward function receives a single argument `ctx`, an instance of this class (all positions env-local, metres; angles rad; forces N):

```python
@dataclass(frozen=True)
class RewardContext:
    # ---- constants (python numbers, fixed for the whole run) ---------------------------
    table_z: float                 # height of the table top [m]
    lift_latch_height: float       # the env sets `lifted` once the cup has risen this far above its starting height [m]
    success_hold_steps: int        # consecutive steps with goal_dist <= success_tol needed to count one success; a success also requires, on that step, that the fingers are wrapped around the cup (the env measures how far the finger links enclose the cup body and requires it above a threshold that starts low and rises as the policy succeeds more often)
    max_successes: int             # the episode ends after this many successes

    # ---- hand: left Tesollo DG-5F, finger index 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky ----
    palm_pos: torch.Tensor         # (N,3) centre of the palm (a virtual point on the palm, not a collision surface)
    palm_normal: torch.Tensor      # (N,3) unit vector pointing out of the palm surface, towards an object held in the hand
    palm_side: torch.Tensor        # (N,3) unit vector lying in the palm plane (palm frame y axis)
    palm_finger_dir: torch.Tensor  # (N,3) unit vector lying in the palm plane, pointing from the palm towards the fingers (palm frame z axis); palm_normal, palm_side, palm_finger_dir form a right-handed frame. In the start pose palm_normal points along -y and palm_finger_dir along +x
    link_pos: torch.Tensor         # (N,5,3,3) finger link positions: [:, f, 0] link moved by joint _3, [:, f, 1] link moved by joint _4, [:, f, 2] fingertip
    link_cup_force: torch.Tensor   # (N,5,3) contact force magnitude between each of those links and the cup only [N] (0 = not touching the cup)
    palm_cup_force: torch.Tensor   # (N,) contact force magnitude between the palm and the cup only [N]
    hand_q: torch.Tensor           # (N,19) finger joint angles [rad], order = the hand joint table in the robot description
    hand_q_norm: torch.Tensor      # (N,19) joint angles normalised to each joint's commandable range: 0 = straight, 1 = most flexed, for every joint (on this left hand thumb_3 and thumb_4 flex towards negative angles, so for them 0 is the upper limit and 1 the lower limit)
    hand_target_norm: torch.Tensor  # (N,19) commanded finger joint targets (after filtering), same normalisation as hand_q_norm
    hand_default_q_norm: torch.Tensor  # (N,19) the hand's default pose: the finger joint angles every episode starts with, same normalisation as hand_q_norm (the same in every environment); in this pose the four fingers are straight, and the thumb, rotated into opposition, is straight and points along palm_normal, reaching about 0.12 m out from the palm surface at the wrist end of the palm (behind palm_pos along palm_finger_dir)
    hand_qd: torch.Tensor          # (N,19) finger joint velocities [rad/s]
    hand_z_min: torch.Tensor       # (N,) height of the lowest finger/thumb link [m]; the palm is not included in this one
    palm_clearance: torch.Tensor   # (N,) height of the lowest point of the palm itself above the table top [m]: 0 means the palm is resting on the table, negative means it is pressed into it. The palm is a large flat body and palm_pos is a virtual point up to 0.065 m away from its surface, so palm_pos alone cannot tell you whether the palm is on the table; this is the measured distance. Note that the episode-ending floor check looks only at the finger and thumb links, so the palm resting on the table does not end the episode by itself

    # ---- arm: 7-DOF ------------------------------------------------------------------------
    arm_q: torch.Tensor            # (N,7) arm joint angles [rad]
    arm_qd: torch.Tensor           # (N,7) arm joint velocities [rad/s]

    # ---- cup: a roughly cylindrical cup standing upright on the table at the start ----------
    cup_pos: torch.Tensor          # (N,3) cup reference point on its axis; the graspable band (cup_half_height above and below) is centred on it
    cup_quat: torch.Tensor         # (N,4) cup orientation quaternion (w,x,y,z)
    cup_axis: torch.Tensor         # (N,3) unit vector along the cylinder axis towards its top; (0,0,1) when upright
    cup_tilt: torch.Tensor         # (N,) angle between cup_axis and world +z [rad]
    cup_lin_vel: torch.Tensor      # (N,3) cup linear velocity [m/s]
    cup_ang_vel: torch.Tensor      # (N,3) cup angular velocity [rad/s]
    cup_spawn_pos: torch.Tensor    # (N,3) cup_pos at the start of the episode, resting on the table
    cup_radius: torch.Tensor       # (N,) radius of the cup's outer surface in the graspable band [m]; differs between environments
    cup_half_height: torch.Tensor  # (N,) half height of the graspable band, centred on cup_pos along cup_axis [m]

    # ---- goal and task status: computed by the environment --------------------------------
    goal_pos: torch.Tensor         # (N,3) position where cup_pos must be held (with the cup upright)
    goal_dist: torch.Tensor        # (N,) largest distance between keypoints fixed on the cup and the same keypoints at the goal pose [m]
    success_tol: torch.Tensor      # (N,) current success tolerance on goal_dist [m]; shrinks as training succeeds
    lifted: torch.Tensor           # (N,) bool, True once the cup has risen above lift_latch_height in this episode (stays True)
    success: torch.Tensor          # (N,) bool, True on the step a success is counted
    num_successes: torch.Tensor    # (N,) number of successes counted so far in this episode
    episode_progress: torch.Tensor  # (N,) elapsed fraction [0,1] of the step budget; the budget restarts after every success

    # ---- stage completion flags: kept by the environment, stay True until the episode ends ----
    approach_done: torch.Tensor    # (N,) bool, set on the first step on which all of these held at once, i.e. the cup sat between the extended thumb and the four fingers in front of the palm: the palm plane was within 2 cm of the cup's side with the cup in front of the palm (the distance from palm_pos to the cup axis along palm_normal, minus cup_radius, was between -0.01 m and 0.02 m); the cup axis was ahead of palm_pos along palm_finger_dir by between cup_radius - 0.005 m and cup_radius + 0.02 m; palm_pos was within the height of the graspable band (at most cup_half_height from cup_pos along cup_axis); the hand kept its start-pose orientation (palm_normal within about 45 degrees of -y and palm_finger_dir within about 45 degrees of +x); every movable finger joint was within 0.3 of hand_default_q_norm (joints with a range of 0.05 rad or less are ignored); and no finger or thumb link touched the cup (contact force above 0.1 N). Episodes that start beside the cup (see the scene description) have approach_done already set on the first step
    envelope_done: torch.Tensor    # (N,) bool, set once, after approach_done, the palm, the thumb and at least 4 digits in total (the thumb included) touched the cup for 5 consecutive steps; the palm or a finger counts as touching when its contact force with the cup is above 0.1 N

    # ---- actions ---------------------------------------------------------------------------
    actions: torch.Tensor          # (N,26) policy action of this step, clipped to [-1,1]; each step the joints receive one of the last 3 policy actions picked at random (a 0-2 step delay)
    prev_actions: torch.Tensor     # (N,26) policy action of the previous step (zeros right after a reset)

    # ---- object surface: measured on the object's actual collision surface, the same way for any object shape ----
    palm_surface_dist: torch.Tensor    # (N,) distance from palm_pos to the nearest point of the object's surface [m]; the surface is represented by 512 points spread evenly over the object's collision geometry (about 1.1-1.7 cm apart), so this reads up to about 1 cm more than the true distance when the point is at the surface and is within a few mm of it beyond 3 cm
    palm_surface_dir: torch.Tensor     # (N,3) unit vector from palm_pos towards that nearest surface point
    palm_surface_normal: torch.Tensor  # (N,3) outward unit normal of the object's surface at that nearest point; palm_surface_dir . palm_surface_normal is close to -1 when palm_pos is outside the object, facing it. Surface points seen from behind are skipped, so next to a thin-walled cup the nearest point is on the outer side of the wall
    link_surface_dist: torch.Tensor    # (N,5,3) the same distance for each finger link position in link_pos (same finger and link order) [m]; link_pos are points inside the finger links, not on their skin, so a link that touches the object (link_cup_force > 0) reads about 1-3 cm (1.8 cm on average) rather than 0
    link_surface_dir: torch.Tensor     # (N,5,3,3) unit vector from each finger link position towards its nearest surface point
    link_surface_normal: torch.Tensor  # (N,5,3,3) outward unit normal of the object's surface at each link's nearest surface point
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors, e.g. `staged = torch.where(ctx.lifted, r_hold, torch.zeros_like(r_hold))`.
3. Prefer bounded, smooth shaping such as `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative distances, so that no single term dominates.
4. Contact: `ctx.link_cup_force[:, f, k]` is the contact force between link k (0 the link moved by joint `_3`, 1 the link moved by joint `_4`, 2 the fingertip) of finger f (0 thumb, 1 index, 2 middle, 3 ring, 4 pinky) and the cup, and `ctx.palm_cup_force` the force between the palm and the cup. Contact is measured only on these links and the palm — the finger links nearest the palm are not measured. Contacts with anything else (the table, the hand itself) are not included. A value of 0 means that link is not touching the cup. Contact forces come from the physics engine each step and can spike. These forces are only available to the reward; the policy itself does not observe contact.
5. Cylinder geometry (the cups are roughly cylindrical): for a point p (e.g. `ctx.link_pos[:, f, k]`), with `v = p - ctx.cup_pos`, the axial coordinate is `h = (v * ctx.cup_axis).sum(-1)` and the radial vector is `v - h.unsqueeze(-1) * ctx.cup_axis`; its norm is the distance from the cup axis. The cup surface is at `ctx.cup_radius` and the graspable band is `|h| <= ctx.cup_half_height`. The palm normal `ctx.palm_normal` points out of the palm.
6. Height: `ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]` is how far the cup has been raised above its starting height (0 while it rests on the table). `ctx.lifted` becomes True once that height exceeds `ctx.lift_latch_height` and stays True for the rest of the episode, even if the cup is dropped again.
7. Goal and success are computed by the environment and cannot be redefined by the reward. The goal position is 0.21 m to 0.28 m above the cup's starting position and at most 5 cm away from it horizontally, with the cup upright. A success is counted when `ctx.goal_dist <= ctx.success_tol` for `ctx.success_hold_steps` consecutive steps and, on that same step, the fingers are wrapped around the cup: the environment measures how far the finger links enclose the cup body and requires that measure to be above a threshold, which starts low and rises as the policy succeeds more often. A cup that reaches the goal without being held that way does not count as a success. `ctx.success` is True on the step a success is counted. After a success the next goal is at the same place, so holding the cup still there keeps producing successes until `ctx.max_successes`, which ends the episode. The keypoints are fixed on the cup, so tilting the cup also increases `goal_dist`. `ctx.success_tol` starts at 0.1125 m and shrinks towards 0.015 m as the policy succeeds more often during training. You may add a bonus on `ctx.success`.
8. An episode lasts at most 900 steps (15 s), and the step budget restarts after every success. The episode ends early when the cup falls below z = 0.15 (off the table), leaves the allowed area around the table, or tilts more than 60 degrees; when any hand link goes below z = table_z - 0.005 (wherever the hand is; this check does not include the palm, whose own clearance is ctx.palm_clearance); or when an arm joint goes past its limit or moves faster than 20 rad/s. On that last kind of physics violation the environment replaces the reward of that step with a fixed -1 (your function's value is not used on that step).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and may be shown back to you after training, so name components meaningfully (e.g. "approach", "lift", "success_bonus").

I want it to fulfil the following task: Pick an object up with a five-fingered hand, carry it to the goal, and hold it there against disturbances, repeatedly.

THE POLICY THIS REWARD IS APPLIED TO ALREADY GRASPS. It was trained by a separate stage-1 reward and, on both arms, it walks in with an open hand, seats the palm against the object's side and wraps four digits around it (envelope 0.55 left, 0.82 right; palm contact 0.50; about 3 of fifteen links). What it does NOT do is lift: lift episodes 0.06-0.13 and the airborne fraction 0.02-0.06. Your reward has to keep that grasp intact and teach the rest - lift, carry to the goal, settle there, survive the disturbance - as one continuous function with no stage switch.

KEEP THE GRASP THE POLICY ALREADY HAS:
- The hand must still be paid to stay OPEN while it travels to the object: multiply the approach income by how close the movable finger joints are to their default open pose (use both the mean deviation and the worst joint). Without this term the previous round, trained from random weights, arrived with the fingers already curled and never wrapped (palm contact 0.000 for thousands of epochs on both arms). Closing must only begin to pay once the finger links are near the surface.
- Palm proximity needs a steep gradient between about 0.08 and 0.02 m: both arms stalled at a 0.05 m palm gap.
- The lower rungs of the digit-count ladder must pay without palm contact, so a partial wrap is worth forming; keep the palm requirement for the top rungs.

ALSO KEEP FROM THE LAST TWO ROUNDS. In the previous round both arms learned to reach and grasp (grasp episodes 0.83 left, 0.98 right, four or five digits in contact) and then stopped: lift stayed at 0.000-0.008 and the airborne fraction at 0.001 for thousands of epochs on both machines. The cause is that income was available while the object still sat on the table. The goal terms were shaped on goal_dist alone, and the goal sits about 0.25 m above the object's start, so an object resting on the table with the hand closed around it was already collecting about 0.55-0.69 per step of hold income, with no risk. Holding still was better than trying to lift. Fix it structurally:

- Every term that belongs to the second half of the task - transport towards the goal, being near the goal, being at rest at the goal, the success bonus - must be multiplied by how far the object has actually risen above its own start height (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]), so that none of them pays anything while the object is on the table. Use a smooth factor of that height, not a threshold.
- Lifting must be the only way to increase income once the hand holds the object: pay generously for the object's height while it is held, and make that term, together with what it unlocks, clearly larger than anything obtainable with the object on the table.
- Holding a closed grasp on the table must become actively unattractive as the episode goes on.

Per-step ceilings:
- reaching the object (palm towards the surface, facing it): about 0.6, fading to near zero once the object is held.
- closing the hand and keeping the grasp (per-digit contact with a steep ladder in the number of digits, palm contact, opposition between thumb and fingers): about 1.2. Contact on every digit must pay alike; two digits must be worth far less than four. Pay palm contact properly: in the previous round the right arm settled on a three-digit pinch with palm contact 0.001 while the left arm used four digits and a palm at 0.38, from the identical reward, so the palm term needs to be worth having.
- lifting the held object: about 2.5, rising with height up to the height the goal needs (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).
- moving towards the goal: about 2.5, shaped over the whole 0 to 0.30 m range and also paying for the reduction of ctx.goal_dist between steps, all of it gated on the object being off the table.
- at rest at the goal, upright, really held: about 10, the dominant income, also gated on the object being off the table. It must already pay a large fraction within about two success tolerances of the goal and grow the longer the object stays there.
- success: a large bonus for the first success of an episode; later successes in the same episode must be worth little, because the environment counts up to five and re-samples the goal after each one.

Keep the object upright by multiplying the lift, transport and hold income by uprightness computed from ctx.cup_tilt.

The environment applies an external force and torque to the object once it is lifted and near the goal, growing as the policy succeeds. Make losing the object, or drifting from the goal, cost much more than the hold income it replaces, and gate the hold income on a real hold (palm plus several opposing digits), so a fingertip touch cannot collect it.

Also: putting the object down after lifting must not let the lift income be earned again; lowering a held object towards the table must be worth less than keeping it still; do not push the hand or the palm into the table; penalise fast arm motion and fast object motion while holding, but keep those small enough that lifting and carrying always beat standing still.

WHERE ON THE OBJECT TO GRIP. In the previous round both arms closed their fingers at the very top of the object, just under its rim, and never seated the palm: the lowest finger link averaged 0.33-0.34 while the graspable wall spans 0.23-0.33, the palm-to-object gap stayed at 0.07 m, and only 2.2-2.6 of fifteen finger links touched. The reason is that proximity was measured to the NEAREST surface point, and for a hand coming in from above that point is the rim edge, whose surface normal points partly upward. Ask for the side wall instead, without naming any object dimension: weight the proximity and facing terms by how horizontal the surface normal at the relevant point is (a side wall has a horizontal normal; a rim, a lid or a table has a vertical one - use ctx vectors and the world up direction only through the z component of those normals, which is identical for the left and the mirrored right arm), and pay for the NUMBER OF LINKS of each digit in contact rather than for a digit touching at all, so that a fingertip on the rim is worth much less than two or three links wrapped around the wall. The palm must be worth seating: a palm against the wall is what separates a real envelope from a pinch at the lip.

WHAT ROUND 11 GOT WRONG (measured on both arms, 500-880 epochs, identical file):
The side-wall weighting was applied to the long-range reach term as well. From the start pose the nearest surface point
of an open cup is its rim, whose normal is nearly vertical, so the factor (1 - |normal_z|) drove the reach term to
almost zero and approaching the object stopped paying at all. With no income and an unconditional stall penalty of
-0.28 to -0.32 per step, both arms walked away: palm-to-surface distance rose 0.22 to 0.37 m (left) and stayed
0.36-0.48 m (right), contacts fell to 0.000-0.003, grasp episodes to 0.001, total reward stayed negative. Round 10,
with the plain distance term, had grasp 0.22 at the same epoch.

FIX IT LIKE THIS:
- The long-range approach must always pay: shape it on the plain palm-to-surface distance, with no side-wall factor,
  or with a factor that never falls below about 0.3.
- Prefer the side wall only from close in: once the palm is within a few centimetres of the surface, weight the
  seating, the per-link nearness and the contact terms by how horizontal the surface normal is. That is where the
  choice between the rim and the wall is actually made.
- The stall penalty must not punish a policy that has not reached the object yet. Gate it on the hand being at the
  object (in contact, or within a few centimetres), and keep it smaller than the reach income it competes with.
- Everything else from round 11 stays: the second half gated on how far the object has risen, lift at 2.5, per-digit
  link-count scoring, the doubled palm share, first-success-only bonus.

WHAT ROUND 13 GOT WRONG (measured, both arms, identical file):
The open-pose factor was applied to the reach income at all times, and the palm-proximity term paid on proximity alone. Together they created a corner the right arm fell into within 200 epochs and never left for 1,400 more: hand fully open (hand_curl 0.000), palm flat against the cup (contact 0.19-0.28, gap 0.023 m), no digit contact at all (grasp episodes 0.000, links 0.05), +0.195 per step. The left arm, which ended stage 1 slightly more curled (hand_curl 0.32 against 0.20), kept its wrap and lifted in 46 % of episodes. Same reward, different basin.

FIX IT LIKE THIS:
- The open-pose factor pays only on the way in. Give it full weight while the palm is far from the surface (about 8 cm or more) and fade it to zero as contact begins, or multiply it by (1 - held). Once the hand is at the object, keeping the fingers straight must earn nothing.
- Palm proximity must not pay on its own. Pair it with digit contact (or require contact outright) so that a flat hand pressed against the object is worth much less than a wrap.
- Everything else from round 13 stays: the second half gated on how far the object has risen, lift 2.5 as the only door downstream, per-digit link-count scoring, palm-free lower rungs with the palm required for the top ones, uprightness as a multiplier, first-success-only bonus, the goal-gated wrench.
- The policy again starts from the stage-1 weights (left envelope 0.55, right 0.82, both with palm contact 0.50 and hand_curl 0.2-0.3). The measure of success in the first 300 epochs is that hand_curl stays near its stage-1 value and grasp episodes do not fall below about 0.4.

WHAT ROUND 14 GOT WRONG (measured on both arms):
The drop penalty was P_DROP x lifted x (1 - held), where `lifted` is the environment's episode-long height latch. After one lift, every later step without a full hold was charged 8 until the episode ended. On the right arm this became the largest term of the whole reward (-0.5 to -0.75 per step) exactly while the arm was doing best (grasp 0.88, lift 0.43, success 0.16, goal distance falling to 0.22 m); the arm then unlearned lifting, unlearned grasping and moved its hand 0.29 m away from the cup, after which the penalty fell to -0.10. The left arm was heading the same way (drop penalty -0.96 per step against lift +0.21 and hold +0.17).

FIX IT LIKE THIS:
- No penalty may be proportional to a persistent latch (`lifted`, or anything that stays set for the rest of the episode). Losing the object must cost a short-lived, bounded amount: charge it while the object is actually falling or lying on the table after having been lifted, using the object's current height and velocity, and keep its size comparable to one step of the lift income. The main cost of dropping is the income the policy stops earning.
- The set-down penalty had the same form (P_SETDOWN x lifted x on_table) and needs the same change.
- Everything else from round 14 stays, including the open-pose factor limited to the approach (the flat-hand corner of round 13 did not return) and palm seating tied to finger engagement.

THE DISTURBANCE IS NOW REAL. An environment bug kept the goal-gated wrench from ever firing since round 9 (fixed). From this round on the object is pushed and twisted while it is lifted and within 0.15 m of the goal, starting at 5 N/kg and growing with success. A slip under that push must not make lifting unattractive - that is exactly what the latch-driven drop penalty would do.
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {"component_name": component_tensor, ...}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never access a field or method that is not listed in the class definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` and contains only the function (plus small helper functions if needed). Add short comments explaining the weights.

Generated code shown as below:
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 1 of the left-arm manipulation stage (`cp_l_m01`) applied this reward to the stage-1 policy, which arrives beside
the cup in the pre-grasp pose and holds it without touching the cup. It ran 1364 epochs, 2.0 hours, and was stopped early
because the run had entered a state from which the envelope can never fire.

**The approach latch collapsed instead of being kept alive.** Far-start episodes fired `approach_done` 0.49 of the time
at the start of the round and 0.06 by epoch 1160: 0.49 -> 0.34 -> 0.31 -> 0.31 -> 0.15 -> 0.11 over the 200-epoch
buckets. The stage-1 policy this round started from had 0.94.

**The cause is in this reward, not in the policy.** The approach shaping term is paid only while the latch is NOT set
(`before_approach * (0.7 * reach + 0.3 * open_hand)`) and nothing in the reward penalises touching the cup before the
latch. Pressing the palm onto the cup maximises the reach term, and that same contact breaks the `no_touch` condition the
environment requires for the latch, so the latch never fires and the shaping income keeps being paid. The measurements
follow exactly that loop: the no-contact condition falls 0.93 -> 0.50, palm contact rises 0.000 -> 0.44, cup tilt rises
0.11 -> 5.26 degrees, while the approach shaping term keeps paying 0.37 per step and the latch term only 0.16.

**Because every closing term is gated on `approach_done`, the grasp was never paid at all.** Digit contact stalled at
0.51 digits (thumb 0.19, index 0.000, ring 0.001), the envelope latch stayed at exactly 0.000 for the whole round, and
lift, airborne fraction and success stayed at 0.000. The five-digit shaping that this round was written for never got a
chance to act.

**The lift terms are gated only on `held` (digit contact), not on the envelope.** They paid 0.003-0.004 per step from
contact alone. Nothing was lifted, but the gating is looser than the task demands.

Placement rejection behaved as designed (0.42 resampled, none failed), the hand never rested on the table (lowest point
0.238 m) and there were no physics errors.

To make the code more accurate and train better robot, the feedback for improvement is:
What to keep from this attempt:
- Counting the digits separately, with the weakest digit scored on its own, so that five digits are worth much more than
  three. This was written to fix the three-digit grasp of the earlier round and it has not been tested yet, because the
  closing terms never paid. Keep it.
- Paying for the finger links on the cup's surface rather than towards its axis, and the palm term.
- Lift income measured from the cup's own starting height, the rise bonus that only pays while the cup is not yet
  airborne, and holding at the goal being worth more than lifting repeatedly.

What has to change, in order of importance:

**Touching the cup before the approach latch must cost something.** There is no contact penalty before the latch, so the
policy maximises the approach shaping by pressing the palm into the cup, which is exactly what stops the environment from
latching. Charge for any finger, thumb or palm contact while `approach_done` is not set, and charge for moving the cup at
all (it tilted 5 degrees this round), the same way the approach stage does.

**Latching must never be a loss of income.** The approach shaping is paid only while the latch is unset, so the policy is
paid to avoid latching. Either keep paying the same shaping after the latch, or make the latch bonus clearly larger than
the shaping it replaces, so that the step on which the latch fires is a step up in reward and stays up.

**The closing terms must not be the only thing behind the gate.** Everything that pays for the grasp is multiplied by
`approach_done`, so when the latch fails the policy sees no path at all towards the task. Give a small, bounded income for
correct pre-grasp geometry with an open hand (the configuration the gate checks) that survives after the latch too, so
there is a continuous slope from arriving to closing.

**Lifting should require the envelope, not just contact.** The lift and rise terms are gated on digit contact alone. Gate
them on `envelope_done` (or on the palm plus four digits), so that a cup dragged upward by a few fingertips does not pay.

One caution. Success, tolerance and lift stayed at zero for this whole round, so nothing can be concluded about the lift
or goal parts of this reward; do not rewrite them in response to those zeros.

Generated code shown as below:
```python
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
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 2 of the left-arm stage-2 (manip) track (`cp_l_m02`) ran 1,530 epochs in 2.2 hours, PPO-LSTM with 4,096
environments, starting from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`, which approaches 0.94 of far
starts and never touches the cup). Placement rejection behaved as designed: 0.417 of resets resampled a placement and
none failed.

**The fixes to the approach latch worked and should be kept.** The approach gate now latches on 0.576 of episodes in the
first 200 epochs, 0.893 by epoch 400-600, 0.942 by epoch 1000-1200 and 0.949 at the end. The collapse of round 1 (0.49
falling to 0.06) did not recur: the no-touch condition holds at 0.957-0.982 for the whole round, cup tilt stays at
2.4-3.4 degrees, and the two new pre-latch penalties are barely charged (contact -0.003 to -0.021, disturbance -0.004 to
-0.011), which means the policy stopped pressing the palm into the cup before the latch. The palm-to-cup gap closes from
0.043 to 0.028 m and the hand never reaches the table (lowest hand point 0.236-0.245 m).

**Nothing behind the latch moved.** The envelope latch is exactly 0.000 in every one of the 1,530 epochs. Lift is 0.001,
the airborne fraction 0.000, success 0.000, and the tolerance never left its starting value of 0.1125.

**The fingers never touch the cup.** Per-finger contact over the round: index 0.000, middle 0.000 and ring 0.000 in every
200-epoch bucket; thumb 0.002 rising only to 0.018-0.041; pinky 0.019 falling to 0.000. Mean digits in contact is
0.022-0.043 across the whole round with no trend. Palm contact, by contrast, rises 0.289 to 0.620 to 0.667 and stays
there. The hand is also not closing: the closing command is flat at 0.263 to 0.331 over 1,530 epochs and the grasp
quality measure moves only 0.223 to 0.306.

**The reward decomposition shows why.** The terms that pay without any contact all rose - palm 0.103 to 0.232,
worst-digit surface proximity 0.231 to 0.306 (of a 0.60 maximum), surface wrap 0.182 to 0.252 - while every term that
requires real finger contact stayed dead for the entire round: per-digit mean 0.002-0.005, weakest digit exactly 0.000,
opposition exactly 0.000, closing drive 0.001, envelope latch 0.000, lift 0.000, goal 0.000. Total reward rose 1.83 to
2.11 entirely on the no-contact terms.

**The recording (final weights, 12 far-start environments, tolerance 0.1125) shows one pose held for the whole episode.**
The hand travels from the raised start to the cup and parks beside it with the flat of the palm against the cup's side.
All four fingers stay fully extended, pointing past the cup and down towards the table, and the thumb is folded back
against the palm instead of opposing the fingers. That pose is reached at about 4 seconds and is still held at 15
seconds. The cup is never enclosed, never moves and never leaves the table.

To make the code more accurate and train better robot, the feedback for improvement is:
What is working and must be preserved: the approach latch and the pre-latch discipline. Approach 0.95, no-touch 0.97,
cup tilt 2.6 degrees, cup left where it spawned. Paying the pre-grasp geometry after the latch as well as before it, and
making the latch bonus larger than the terms it switches off, removed the incentive to sit unlatched. Keep all of that.

The feedback for improvement is that the flat open hand is currently the best-paying pose in the whole reward, so the
policy has correctly found it and will not leave it.

1. Proximity and wrap are collected by a straight finger lying alongside the cup. Both terms measure only where a link
   is, not whether the digit is flexed around the body, and an extended finger skimming the surface sits exactly where
   they pay most. Curling would move links off that surface band and lose money. Credit for being near or on the surface
   should require the digit itself to be closing around the cup, so that an extended hand resting against it earns
   nothing.

2. Nothing rewards the thumb opposing the fingers. In the video the thumb is folded back against the palm, yet the
   envelope the task asks for needs the thumb plus at least four digits on opposite sides of the body. The thumb being
   across from the fingers should be paid for in its own right, before any contact happens.

3. The contact terms are an unreachable ramp. Per-digit contact, weakest-digit contact and opposition were exactly zero
   for 1,530 epochs, so they carried no gradient at any point in the round - they cannot teach anything. There must be a
   single continuous path from "approaching" through "nearly touching" to "closed on the cup" that strictly increases at
   every step of closing, and the near part of that path must be clearly worth less than the touching part, so that
   stopping at "near" is a loss rather than a comfortable plateau.

4. Palm-only contact is currently the largest contact income and it rose to 0.67 while the fingers stayed at zero. Palm
   contact should not pay on its own; it should count only together with digits that are also closing, otherwise it
   funds exactly the flat-hand pose seen in the video.

5. The hand's closing command never moved (0.26 to 0.33 over the round), so the policy is not even exploring closure.
   Whatever replaces the current closing drive has to make each additional increment of finger flexion profitable from
   the pose the hand is already in, not only once contact has been made.

6. Lifting, carrying and holding were never reached and cannot be judged from this round. They should stay gated behind
   a real enclosing grasp rather than being loosened, since the failure is upstream of them.

7. The policy this round starts from was trained for 4,280 epochs by a stage-1 reward whose largest standing terms were
   "hold the pre-grasp pose" and "keep the hand in its default open posture", with any cup contact penalised. Holding an
   open hand beside the cup is therefore a deeply reinforced habit, not a random starting point, and the closing command
   did not move at all this round (0.26 to 0.33 over 1,530 epochs). Stage 2 needs a term that pays for leaving the
   default open posture as such - measured on the finger joints, not on contact - so that there is a gradient out of
   that habit from the very first step, before anything touches the cup. It should be smaller than the terms that pay
   for real contact and a real envelope, so closing onto nothing is never as good as closing onto the cup.

Generated code shown as below:
```python
import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
    """0 when a link is off the cup, ->1 once it presses with a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate the sum."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.cup_pos.device
    dtype = ctx.cup_pos.dtype

    # ---- weights ---------------------------------------------------------------------------
    # Ladder of per-step ceilings:
    #   flat open hand latched at the cup ~1.9  <  five digits curled and near ~3.5
    #   <  closed five-digit envelope ~6.2  <  lifted ~8.4  <  still at the goal ~12.9 (+ success)
    # Every stage transition is a step UP that STAYS up, so no pose is a place to park.
    W_PREGRASP = 0.70        # env pre-grasp geometry -- PAID BEFORE *AND AFTER* THE LATCH
    W_OPEN = 0.30            # default open hand, only before the latch (the gate demands it)
    W_ORIENT = 0.15          # start orientation, only before the latch, 40 deg dead zone
    W_LATCH = 0.90           # approach_done income: > W_OPEN + W_ORIENT that it replaces
    P_PRE_TOUCH = 1.20       # touching the cup before the latch (round-1 exploit; keep)
    P_PRE_DISTURB = 0.80     # moving / tilting the cup before the latch (keep)
    W_UNFOLD = 0.35          # NEW: leaving the default OPEN posture, measured on the joints only.
                             #      Small on purpose: closing onto nothing must never beat closing
                             #      onto the cup. This is the gradient out of the stage-1 habit.
    W_THUMB_OPP = 0.50       # NEW: thumb across the body from the fingers, paid WITHOUT contact
    W_DIGIT_MEAN = 1.20      # mean over the 5 digits; EVERY digit weighted the same
    W_DIGIT_LAST = 1.00      # min over the 5 digits: the LAST digit to arrive earns it
    W_WRAP = 0.50            # links ON the cup surface inside the band, gated on real flexion
    W_PALM = 0.40            # palm on the cup side, but ONLY together with closing digits
    W_ENV_LATCH = 0.60       # the environment's own envelope latch
    W_LIFT = 2.20            # height of the cup above its own spawn height
    W_RISE = 0.60            # upward cup velocity, only until `lifted` latches
    W_GOAL = 2.00            # closing on goal_pos
    W_HOLD = 2.60            # at rest at the goal -> worth more than repeating lifts
    W_SUCCESS = 10.0         # bonus on the step the env counts a success
    P_DESCEND = 0.40         # lowering the held cup < holding it still < raising it
    P_STALL = 0.15           # closed grasp parked on the table, late in the episode
    P_TILT = 0.60            # keep the cup upright
    P_DROP = 0.50            # lifted and then let go
    P_TABLE = 0.50           # palm / fingers driven into the table top
    P_ARM_VEL = 0.12         # smoothness: tiny next to W_LIFT, a slow lift always wins
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
    n_movable = movable.sum()
    # per-digit masks over the 19 joints, same order as hand_q_norm
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0      # thumb_3, thumb_4
    digit_joint[1, 4:7] = 1.0      # index_2, _3, _4
    digit_joint[2, 8:11] = 1.0     # middle_2, _3, _4
    digit_joint[3, 12:15] = 1.0    # ring_2, _3, _4
    digit_joint[4, 17:19] = 1.0    # pinky_3, pinky_4
    n_digit_joint = digit_joint.sum(-1)                                   # (5,)

    # =========================================================================================
    # Stage A: arrive at the pre-grasp pose and LATCH it   (kept from round 2 -- it works)
    # =========================================================================================
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

    # the gate also demands the default OPEN hand; only meaningful before the latch
    open_err = ((ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable).sum(-1) / n_movable
    r_open = W_OPEN * pre_latch * torch.exp(-open_err / 0.15)

    # the gate demands roughly the start orientation (palm_normal ~ -y, palm_finger_dir ~ +x).
    # Dead zone out to 40 deg: the far corners legitimately need up to ~40 deg of yaw.
    minus_y = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)
    plus_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    cos40 = math.cos(math.radians(40.0))
    cos_n = (ctx.palm_normal * minus_y).sum(-1)
    cos_f = (ctx.palm_finger_dir * plus_x).sum(-1)
    orient_err = (cos40 - cos_n).clamp(min=0.0) + (cos40 - cos_f).clamp(min=0.0)
    r_orient = W_ORIENT * pre_latch * torch.exp(-orient_err / 0.25)

    # latch bonus > the two terms it switches off: the latch step is strictly a step up
    r_latch = W_LATCH * approach

    # ---- contacts (shared by several stages) --------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5) does this digit touch
    digit_links = link_c.mean(dim=-1)                              # (N,5) how many of its links
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    any_touch = torch.maximum(digit_touch.amax(dim=-1), palm_c)

    # before the latch: any contact costs more than the shaping pressing in could buy, and the
    # cup must be left exactly where it spawned.
    p_pre_touch = P_PRE_TOUCH * pre_latch * any_touch
    cup_disp = (ctx.cup_pos - ctx.cup_spawn_pos).norm(dim=-1)
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    disturb = (torch.tanh(cup_disp / 0.01)
               + torch.tanh(cup_speed / 0.05)
               + torch.tanh(ctx.cup_tilt / math.radians(5.0))) / 3.0
    p_pre_disturb = P_PRE_DISTURB * pre_latch * disturb

    # =========================================================================================
    # Stage B: close a five-digit envelope
    # =========================================================================================
    # ---- how far each digit has CLOSED, relative to the default open pose -------------------
    # hand_q_norm is 0 = straight, 1 = most flexed for every joint, so the advance beyond the
    # default pose is a direct, contact-free measure of flexion.
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)

    # FIX 1 (round 2's flat hand): geometric credit requires the digit itself to be closing.
    # The thumb only needs a modest flexion to drop onto the near side; the four fingers have to
    # curl properly to come round the far side, so they need more.
    flex_ref = torch.tensor([0.20, 0.35, 0.35, 0.35, 0.35], device=device, dtype=dtype)
    flex_gate = (closure / flex_ref).clamp(min=0.0, max=1.0)                                # (N,5)

    # FIX 2 (feedback 7): pay for leaving the deeply reinforced open posture as such, on the
    # joints alone, from the first step behind the latch. Capped at half flexion so it cannot
    # drive the fingers through the cup, and small next to the contact terms.
    r_unfold = W_UNFOLD * approach * (closure / 0.5).clamp(max=1.0).mean(dim=-1)

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

    # FIX 3: ONE continuous, strictly increasing path per digit:
    #   far -> curled and near the surface (at most NEAR_SHARE) -> one link touching
    #   -> all three links touching (TOUCH_SHARE). Contact keeps `near` at its maximum, so the
    #   sum only ever rises; and "near" alone is worth barely a third of "touching".
    near = torch.exp(-digit_surf_dist / 0.03) * flex_gate                            # (N,5)
    contact = 0.5 * digit_touch + 0.5 * digit_links                                  # (N,5)
    digit_stage = NEAR_SHARE * near + TOUCH_SHARE * contact                          # (N,5) in [0,1]

    # equal weight per digit: index and ring pay exactly as much as thumb / middle / pinky
    r_digit_mean = W_DIGIT_MEAN * approach * digit_stage.mean(dim=-1)
    # the weakest digit, scored on its own: 5 digits >> 4 >> 3. It is no longer a dead ramp,
    # because the near part gives it a gradient long before the last digit makes contact.
    r_digit_last = W_DIGIT_LAST * approach * digit_stage.amin(dim=-1)

    # wrapping ALONG the surface (never towards the axis), and only for digits that are curling:
    # an extended finger skimming the cup collects nothing here any more.
    wrap_link = torch.exp(-surf_err / 0.02) * torch.exp(-band_err / 0.02)             # (N,5,3)
    r_wrap = W_WRAP * approach * (wrap_link.mean(dim=-1) * flex_gate).mean(dim=-1)

    # FIX 4 (feedback 2): the thumb must sit ACROSS the body from the fingers, and this is paid
    # before any contact. Frame-free: compare radial directions about the cup axis.
    tip_radial = radial_vec[:, :, 2, :]                                              # (N,5,3)
    tip_dir = tip_radial / tip_radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    thumb_dir = tip_dir[:, 0, :]                                                     # (N,3)
    fing_mean = tip_dir[:, 1:, :].mean(dim=1)
    fing_dir = fing_mean / fing_mean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    oppose = ((1.0 - (thumb_dir * fing_dir).sum(-1)) * 0.5).clamp(min=0.0, max=1.0)
    # only meaningful when both the thumb and the fingers are actually at the cup (soft, 8 cm)
    prox_soft = torch.exp(-digit_surf_dist / 0.08)                                   # (N,5)
    r_thumb_opp = W_THUMB_OPP * approach * oppose * prox_soft[:, 0] * prox_soft[:, 1:].mean(dim=-1)

    # FIX 5 (feedback 4): palm contact does not pay on its own any more. It is worth something
    # only together with digits that are closing (and much more once they touch), so it can no
    # longer fund the flat open hand pressed against the cup.
    digit_engage = 0.3 * flex_gate.mean(dim=-1) + 0.7 * digit_touch.mean(dim=-1)
    r_palm = W_PALM * approach * palm_c * digit_engage
    r_env_latch = W_ENV_LATCH * envelope

    # "secure" = palm plus at least four digits (full at five): the envelope the task demands.
    # Unchanged from round 2 -- the lift was never reached, so it is not rewritten.
    w_sum = digit_touch.sum(dim=-1)                                                  # 0..5
    secure = ((w_sum - 3.0) / 2.0).clamp(min=0.0, max=1.0) * (0.4 + 0.6 * palm_c)

    # =========================================================================================
    # Stage C: lift   (unchanged)
    # =========================================================================================
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
    # arm joints are capped near 0.3 rad/s, so this stays <= ~0.15: a slow lift always wins
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * secure) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_cup_vel = secure * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                          + P_CUP_SPIN * torch.tanh(ctx.cup_ang_vel.norm(dim=-1) / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_pregrasp + r_open + r_orient + r_latch
        + r_unfold + r_thumb_opp
        + r_digit_mean + r_digit_last + r_wrap
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
        "grasp_unfold": r_unfold,
        "grasp_thumb_oppose": r_thumb_opp,
        "grasp_digit_mean": r_digit_mean,
        "grasp_digit_last": r_digit_last,
        "grasp_wrap": r_wrap,
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 3 of the left-arm stage-2 (manip) track (`cp_l_m03`) ran 2,868 epochs in 4.5 hours, PPO-LSTM with 4,096
environments, again starting from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`). Placement rejection
behaved as designed: 0.416 of resets resampled a placement and none failed. There is no recording for this round: the
playback was started while the training process still held the GPU, the physics engine failed to launch its collision
kernels, and no frames were written. Everything below is read from the training metrics.

**The fingers closed for the first time in three rounds.** Mean digits in contact rose 0.02 (epochs 0-200) to 0.57
(400-600) to 1.15 (800-1000), dipped to 0.61 (1600-1800), then reached 1.59 (2200-2400). Per finger at the peak: thumb
0.605, middle 0.467, ring 0.468, pinky 0.128, index 0.011. The middle finger, which had been exactly 0.000 for two whole
rounds, first touched at epoch 1800 and reached 0.467. The weakest-digit term left zero for the first time (0.001 to
0.116). Contact force per link rose 0.003 to 0.43 N and the closing command rose 0.215 to 0.366.

**The envelope latched for the first time, briefly.** It is 0.000 up to epoch 2000, then 0.115 (2200-2400) and 0.152
(2400-2600), then falls back to 0.046. The lift fraction reached only 0.022 and success stayed 0.000 throughout, with
the tolerance never leaving 0.1125.

**The approach latch then collapsed.** Far-start approach holds 0.93-0.95 until epoch 1800 and then falls: 0.877
(2000-2200), 0.723, 0.539, 0.347, 0.373. Two of the six gate conditions explain it. The default-pose condition, which
requires EVERY movable finger joint to stay within 0.3 of its default angle, passes 0.997 of steps in the first 200
epochs and then 0.213 from epoch 400 onward - it broke as soon as the hand learned to flex, long before the collapse
became visible in the latch. The no-touch condition falls 0.979 to 0.376. The instantaneous pass rates of the other
conditions are also low late in the round (gap 0.22, along 0.29), so the latch depends on a narrow window early in the
episode, and once the hand starts closing before it arrives, that window never opens.

**Closing was done by pushing, not by wrapping.** Cup tilt rises 0.06 degrees to 3.4, then spikes to 12.3 degrees at
epochs 2200-2400 and settles at 4.0. The largest arm joint velocity per episode explodes over the same interval: 11.8,
then 22.9, then 1,901, 2,020, 11,591 rad/s, and abnormal-runaway terminations appear (0.002 of episodes). The envelope
numbers of that interval therefore come from a physically unstable state and cannot be taken at face value.

**The palm left the cup and never came back.** Palm contact stays between 0.07 and 0.16 for the whole round (it was
0.67 in the previous round), the palm reward term is 0.014-0.036, and the palm-to-cup gap grows 0.031 m to 0.082 m as
the fingers engage. Since the envelope latch needs the palm, the thumb and at least four digits together, the grasp was
converging to a fingertip pinch that cannot satisfy it.

**The index finger never participated**, staying between 0.002 and 0.011 for all 2,868 epochs.

To make the code more accurate and train better robot, the feedback for improvement is:
What worked and must be kept: the flexion-gated proximity and wrap credit, the single continuous per-digit ladder from
"curled and near" to "touching", the term that pays for leaving the default open posture, and the thumb-opposition term.
Together they did what two previous rounds could not - the hand closes, the middle and ring fingers engage, and the
envelope latched for the first time. Do not go back to contact-only closing terms.

The feedback for improvement is that closing now destroys the gate that pays for closing.

1. Closing before the latch must cost more than it earns. Every closing term is gated on the approach latch, but there
   is nothing that discourages flexing the hand before that latch, and the only counterweight - holding the default open
   posture before the latch - is worth about one twenty-fifth of what closing pays afterwards. The optimal policy under
   that arrangement is to close as early as possible, which is exactly what happened. Holding the open posture until the
   latch has to be worth roughly as much as closing is afterwards, and flexing the fingers before the latch should be
   penalised directly on the joint angles, not only when it produces contact.

2. The gate's pose condition is a maximum, not an average. It requires every movable finger joint to stay within 0.3 of
   its default angle, so a single finger that flexes past that threshold breaks the whole approach gate. Measured pass
   rate fell from 0.997 to 0.213 at epoch 400 and never recovered. The pre-latch posture term should therefore be shaped
   on the worst joint, matching what the gate actually tests, instead of on an average over the hand.

3. The reward regime switches on a state the policy cannot observe. The observation vector contains joint angles and
   velocities, palm pose, fingertip positions relative to the palm, the commanded hand state, cup and goal keypoints and
   the previous action - it contains neither the approach latch, nor the envelope latch, nor any contact force. The
   policy has to infer from its own memory that its reward regime has changed. Terms that turn on and off hard at the
   latch are therefore weak signals; prefer terms whose value changes smoothly with the same geometric quantities the
   gate measures, so that the gradient points towards the gate rather than depending on knowing which side of it the
   policy is on.

4. The grasp is being made by pushing the cup. Cup tilt reached 12 degrees and the largest arm joint velocity per
   episode reached 11,591 rad/s, with runaway terminations appearing, exactly in the interval where the envelope first
   latched. Closing has to be restrained: excessive contact force, cup displacement and cup rotation while closing
   should cost clearly more than the extra contact they buy, so that the fingers settle onto the cup rather than drive
   through it.

5. The palm must stay on the cup while the fingers close. Palm contact never rose above 0.16 and the palm-to-cup gap
   grew from 0.031 m to 0.082 m as the fingers engaged, so the grasp converged to a fingertip pinch. The envelope needs
   the palm, the thumb and four digits at once. Keeping the palm against the cup should be paid for continuously
   throughout closing, using the distance itself rather than only the binary contact.

6. The index finger still never participates (0.002 to 0.011 over the whole round) while the middle and ring fingers now
   do. Whatever asymmetry keeps it out should be addressed by making the per-digit score of the two digits that arrive
   last worth clearly more than the ones that arrive first.

Generated code shown as below:
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 4 of the left-arm stage-2 (manip) track (`cp_l_m04`) ran 2,611 epochs in 3.6 hours, PPO-LSTM with 4,096
environments, from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`). Placement rejection behaved as
designed: 0.417 of resets resampled a placement and none failed. The round was ended because the approach gate held at
0.93-0.96 from epoch 2000 while the envelope latch stayed at exactly 0.000 for the whole run.

**Everything the previous round broke is now fixed.** The gate's default-pose condition passes 1.000 of steps for the
first 400 epochs and settles at 0.23-0.26 once the hand starts to flex, but unlike the previous round the approach latch
does not follow it down: it rises 0.582 to 0.891 and ends at 0.957. The no-touch condition holds at 0.957-0.999. Cup
tilt stays at 0.03-0.62 degrees against 12.3 degrees before, the largest arm joint velocity per episode peaks at 12.6
rad/s against 11,591 before, no runaway terminations occur, and the hand never reaches the table (lowest point 0.239 m).
The pre-latch flexion penalty is only -0.02 to -0.06 per step, which means the policy has learned to keep the hand open
until the latch is set and to flex only afterwards - the conditional behaviour the reward was asking for, learned even
though the latch is not in the observation.

**The closing terms rose and then saturated without any contact.** Over the round: leaving the open posture 0.001 to
0.144, per-digit mean 0.001 to 0.142, the two last digits 0.000 to 0.134, wrap 0.000 to 0.114, palm-distance 0.000 to
0.061. All five are flat over the last 800 epochs. Meanwhile mean digits in contact is 0.037, mean contact force per
link is 0.011 N, and per-finger contact is thumb 0.004, index 0.001, middle 0.003, ring 0.013, pinky 0.016. Palm contact
is 0.009 and the palm-to-cup gap is 0.052 m, slightly wider than the 0.038 m it reached at epoch 500. The envelope latch
is 0.000, the lift fraction 0.002 and success 0.000, with the tolerance never leaving 0.1125.

**Hand flexion went backwards.** It rises 0.126 to 0.164 by epoch 1200 and then falls steadily to 0.121, while the
closing command flattens at 0.288. The policy found a flexion level and stopped there.

**The recording (final weights, 12 far-start environments, tolerance 0.1125) shows why.** The hand travels to the cup
and stops beside it in a loose claw: the four fingers are bent at the knuckles, roughly matching the cup's curvature,
but the fingertips end a visible gap short of the cup wall and never advance. The thumb is folded back against the palm
rather than across the cup. The palm also stops short of the cup surface. That pose is reached at about 4 seconds and
held unchanged to 15 seconds, with the cup untouched and upright on the table for the whole episode.

To make the code more accurate and train better robot, the feedback for improvement is:
What worked and must be kept exactly as it is: holding the open hand until the latch (the strong open-posture term shaped
on the worst joint, plus the direct pre-latch flexion penalty), gating the closing terms on the continuous geometric
readiness instead of the unobservable latch flag, and the restraint penalties on contact force, cup shoving and arm
speed. Together they gave an approach latch of 0.96, a no-touch condition of 0.96, cup tilt under 0.6 degrees, arm speed
under 13 rad/s and no runaway terminations, while the hand still learned to flex only after the latch. None of that
should be weakened to buy contact.

The feedback for improvement is that the hand now stops one gap short of the cup and nothing pays for crossing it.

1. The proximity part of the per-digit score peaks where the finger is near the surface, and the policy has settled
   exactly there. Closing terms saturated (0.13-0.14 each) while contact stayed at 0.037 digits and 0.011 N, and hand
   flexion actually fell back from 0.164 to 0.121 after epoch 1200, so at that pose flexing further is a loss. Credit
   for being near the surface has to be capped well below what a single real contact pays, and the gap between "near"
   and "touching" has to be a large step up rather than a small one, so that the last centimetre is where most of the
   money is.

2. The gating is circular and settles in a low-value equilibrium. Palm credit is multiplied by digit engagement, digit
   credit needs the digits near the surface, and the hand sits at a distance where neither pays much: palm-distance
   credit reached only 0.061 of its 0.60 maximum and the palm-to-cup gap grew from 0.038 m to 0.052 m. Bringing the palm
   in to the cup should pay on the palm-to-cup distance alone once the approach has latched, with no dependence on the
   digits, so there is always a gradient that moves the whole hand onto the cup.

3. The thumb is folded back against the palm in the recording, yet the thumb-opposition term pays 0.27 of its 0.50
   maximum. It is being satisfied by geometry that does not correspond to an opposed thumb. Opposition should require
   the thumb to be on the far side of the cup from the fingers AND close to the surface, so that a thumb tucked into the
   palm earns nothing.

4. Flexion alone is being rewarded to a level the task does not need. The term for leaving the open posture reached 0.144
   and then stopped mattering, because further flexion moves the fingertips away from the cup rather than around it. That
   term should fade out once the digits are near the cup, handing over to terms that depend on the fingertip-to-surface
   distance, so that late in the approach the only way to earn more is to actually reach the cup.

5. The index finger still never participates (0.001 over the whole round) and the middle finger barely does (0.003),
   while ring and pinky reach 0.013-0.016. The two-weakest-digit term was meant to fix this and did not, because it too
   is satisfied by proximity rather than contact. Once proximity is capped as described in point 1, that term should
   carry most of its value in the contact part.

6. Lifting, carrying and holding were never reached again and cannot be judged. Leave that part of the ladder as it is.

Generated code shown as below:
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 5 of the left-arm stage-2 (manip) track (`cp_l_m05`) ran 3,057 epochs in 4.0 hours, PPO-LSTM with 4,096
environments, from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`). Placement rejection behaved as
designed: 0.417 of resets resampled a placement and none failed. The round ended at the four-hour limit, with the
approach gate at 0.91-0.95 since epoch 1800 and the envelope latch at exactly 0.000 for the whole run.

**The approach is the best it has been in five rounds and everything stays safe.** The approach latch rises 0.577 to
0.926 by epoch 800, dips to 0.808, and holds 0.90-0.95 from epoch 1800 to the end. Cup tilt is 0.03-0.14 degrees, the
largest arm joint velocity per episode is 5.8 rad/s at the end, the no-touch condition holds at 0.99, and the hand never
approaches the table (lowest point 0.252-0.265 m).

**The two new terms worked as designed and immediately became a plateau.** Fine-scale seating reaches 0.634 in the first
200 epochs and sits at 0.697-0.708 (of a 0.80 maximum) for the remaining 2,800; palm-to-cup distance reaches 0.447
immediately and stays at 0.434-0.467 (of 0.60). Together they pay about 1.17 per step from the very first epochs, with
no dependence on the fingers. Every finger term added together is worth 0.07 at the end of the round. The palm-to-cup
gap sits at 0.048-0.058 m and does not close further.

**The thumb learned to oppose.** Thumb opposition rises 0.024 to 0.130 over the round, and the closing command rises
0.190 to 0.325, so the hand is commanding more flexion than in any previous round.

**But contact never happened.** Mean digits in contact is 0.004 at the end and never exceeded 0.009 in any 200-epoch
bucket. Per finger over the whole run: index 0.000, ring 0.000, middle 0.001, pinky 0.001, thumb 0.002. Mean contact
force per link is 0.008 N. Measured hand flexion rose only 0.125 to 0.138 despite the larger command, the per-digit mean
term reached 0.031 (of 1.50) and the two-weakest-digit term 0.002 (of 2.00). The envelope latch, lift, airborne fraction
and success are all exactly 0.000 and the tolerance never left 0.1125.

**Compared with round 4 this is a regression in grasping.** At a comparable epoch, round 4 had per-digit mean 0.142,
digits in contact 0.037 and hand flexion 0.156; round 5 has 0.031, 0.004 and 0.138.

**The recording (final weights, 12 far-start environments, tolerance 0.1125) shows the hand parked beside the cup with
the fingers close to straight.** It arrives at about 4 seconds, sets the palm facing the cup a visible distance away,
and extends the fingers past the cup rather than around it; the thumb sits up near the wrist. That configuration is held
without change to 15 seconds. The cup stays upright and untouched on the table for the whole episode.

To make the code more accurate and train better robot, the feedback for improvement is:
What worked and must be kept: the pre-latch discipline (open hand shaped on the worst joint, direct penalty on pre-latch
flexion), the continuous geometric readiness gate, the restraint penalties, and the tightened thumb-opposition test,
which finally taught the thumb to oppose instead of folding into the palm. The approach latch of 0.95 with cup tilt
under 0.15 degrees and arm speed under 6 rad/s is the best behaviour of the whole track. None of that should be traded
away.

The feedback for improvement is the pattern across the last three rounds: each time a term is added to fix the previous
round's dead end, the policy settles at that term's maximum and stops, because the new term reaches its maximum without
any contact. A flat open hand did it, then a loose claw, now a seated palm. The single rule that would stop this is that
**no term may sit near its maximum while contact is zero.**

1. The seating and palm-distance terms must not be payable on their own. They reached 89 % and 76 % of their maxima
   within 200 epochs and stayed there for 2,800 more, paying about 1.17 per step while every finger term together paid
   0.07. Either multiply them by how many digits are engaged, or cap them at a small fraction of their present value and
   move the freed budget into the contact terms. Position without grip is not progress and should not be priced as if
   it were.

2. Cutting the proximity share from 0.35 to 0.15 removed the on-ramp. Round 4, with the larger share, reached per-digit
   mean 0.142, 0.037 digits in contact and 0.156 flexion; round 5, with the smaller share, reached 0.031, 0.004 and
   0.138. The proximity ramp is what gets the fingers moving at all, so it should be restored to roughly its earlier
   strength - but bounded, so that all five digits at their best proximity are still worth strictly less than one digit
   actually touching.

3. The commanded flexion rose (0.190 to 0.325) while the measured flexion barely moved (0.125 to 0.138). The hand is
   being told to close and is not closing. Rewarding the commanded posture is therefore not reaching the fingers;
   reward what the joints actually do, and check the gap between command and measurement rather than paying for either
   alone.

4. The index and ring fingers have now been at exactly 0.000 for three consecutive rounds while the thumb, middle and
   pinky move. The recording shows why: the fingers are extended past the cup instead of curled around it, so they
   approach the cup along their length rather than closing onto it. The per-digit geometry should be measured from the
   fingertip towards the cup surface along the direction that finger actually closes in, so that a finger lying past
   the cup scores badly and a finger curling onto it scores well.

5. The palm-to-cup gap held at 0.05 m all round and never closed. Whatever pays for the palm should keep increasing all
   the way to contact, with its largest gradient in the final centimetre, in the same way the digit terms were meant to.

6. Lifting, carrying and holding were never reached for the fifth round running. Leave that part of the ladder alone.

Generated code shown as below:
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 6 of the left-arm stage-2 (manip) track (`cp_l_m06b`) ran 2,074 epochs in 3.1 hours, PPO-LSTM with 4,096
environments on the local RTX 5090, from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`). Placement
rejection behaved as designed: 0.42 of resets resampled a placement and none failed. The round was stopped early by the
user after reviewing the latch criteria (see below), not by the four-hour rule. The envelope latch was exactly 0.000 for
the whole run.

**For the first time in six rounds the hand made sustained contact with the cup.** Links in contact rose 0.11 to 0.93 by
epoch 600 and ended at 0.91; the per-digit contact term rose 0.06 to 0.52. The iter_06 principle ("no term may sit near
its maximum while contact is zero") worked: the zero-contact plateaus of rounds 2-5 did not form.

**But only two digits ever touched: the thumb and the pinky.** Over the last 400 epochs: thumb 0.39-0.40, pinky
0.49-0.51, index 0.001, middle 0.001, ring 0.002-0.003. Index, middle and ring never exceeded 0.003 in any 200-epoch
bucket. Palm contact was 0.05 early and 0.007-0.017 afterwards. The envelope needs the palm, the thumb and four digits,
so it was never close.

**The contact came before the approach latch, and the approach collapsed.** The approach latch fell 0.63 to 0.16 by
epoch 1200 and recovered only to 0.31-0.35. Per condition, from the first bucket to the end: along 0.61 to 0.17, gap 0.62
to 0.39, no-touch 0.91 to 0.41, orientation 1.00 to 0.72, and from epoch 1200 the open-hand pose condition also fell,
0.97 to 0.45-0.52. The pre-latch shove penalty grew to -0.39 per step but stayed smaller than the contact reward
(+0.52), so touching the cup with the thumb and the pinky from beside it, before reaching the C pose, became the new
plateau. Travel (0.40) and the approach-latch term (1.07) were still paid alongside it.

**Everything else stayed safe and did not move.** Palm-to-cup gap 0.049-0.065 m, lowest hand point 0.24-0.25 m (no table
contact), cup tilt 1.7-3.3 degrees, lift 0.000-0.003, airborne fraction 0, success 0, tolerance 0.1125 throughout.

**Why the round was stopped: the stage boundary itself is the problem.** Both latches the reward is keyed on are
hand-written, cup-specific templates in the environment (`grasp_fj_t2r/grasp_gates.py`). The approach latch requires a
C pose defined from the cylinder radius and axis (palm plane -1 to +2 cm from the cup side, cup axis within a 25 mm
window ahead of the palm, open hand within 0.3, no finger touching, start orientation within 45 degrees). Stage 1 was
trained to satisfy exactly this (hold_e4280, approach 0.94), and rounds 2-5 kept the approach at 0.91-0.96 while the
envelope stayed at 0. Standing in that template is not a step towards grasping. Neither latch is in the observation,
and once set they stay set even if the hand leaves, so the reward regime they switch cannot be inferred by the policy. The
user's judgement: once the hand is close enough it should learn to close the fingers into a good grasp naturally, and a
cup-shaped template cannot extend to other objects. The same latches were reached by `cup_pick_l_approach_e3400`
(envelope 0.81, thumb/index/middle/pinky and palm 0.54), but that policy learned approach and grasp together in one
reward and never went through a no-contact stage.

**The recording (snapshot epoch 1800, 12 far-start environments, tolerance 0.1125) shows the hand pressing the cup's
lower side with straight fingers.** The hand travels in by about 4 seconds, but instead of setting the palm against the
cup it stops low and in front of it with the fingers extended forward, so the fingertips and the thumb poke the lower
part of the cup from the side. The fingers never curl around the cup and the palm stays a visible distance away. That
configuration is held with little change from 5 to 14 seconds; the cup stays upright on the table and is never lifted.
Video: `our_source/fj_t2r_videos/cp_l_m06b_0922_2302.mp4`.

To make the code more accurate and train better robot, the feedback for improvement is:
What worked and must be kept: making every position term worth little until the fingers are in contact. Round 6 is
the first round in which contact appeared and stayed (links in contact 0.91), so the rule "no term may sit near its
maximum while contact is zero" is right and should stay. The restraint penalties also kept everything safe (no table
contact, cup tilt under 3.3 degrees).

The feedback for improvement is structural and comes from the user, not from tuning: **do not key the reward on the
environment's approach and envelope latches at all.** They are hand-written templates of one grasp on one cylinder
(palm plane -1 to +2 cm from the cup side, cup axis in a 25 mm window, open hand, no touch, start orientation), they are
not in the observation, and once set they stay set. Six rounds show the consequence: when the policy meets the approach
template perfectly (rounds 2-5, approach 0.91-0.96) it has learned to stand still with an open hand, and when the
reward pushes contact (round 6) the contact happens outside the template and the approach latch collapses to 0.16-0.35.
Either way the envelope latch stays at 0. A grasp reward built on these templates also cannot move to other objects.
The latches remain useful as logged metrics only.

1. Replace the stage switch with continuous, object-agnostic quantities computed from the current state. The
   environment now provides (manip_stages `ObjectRewardContext`) the distance from the palm and from every finger link
   to the nearest point of the object's real surface, the direction to that point and the surface normal there. Being
   close is a smooth function of those distances, and closing the fingers is worth more the closer the hand is; there is
   no threshold at which the reward changes regime.

2. Price the grasp by what the hand does to the object, not by a pose: the number of digits in contact, palm contact,
   and opposition, i.e. the thumb's contact pushing against the fingers' contacts from the other side of the object
   (from the surface normals at the contact points). It must rise steeply with the third, fourth and fifth digit, so
   that the thumb-and-pinky touch this round settled on (per-digit mean 0.52 with two digits) is worth little. Index,
   middle and ring stayed at 0.001-0.003 for 2,000 epochs; they need to pay at least as much as the others.

3. Contact before the hand is close and seated must not be profitable. This round the pre-latch shove penalty (-0.39)
   was smaller than the contact reward (+0.52). With continuous proximity this becomes: contact counts in proportion to
   how well the palm faces and approaches the surface, so a fingertip poke from beside the object earns little.

4. Lift and hold stay as outcomes of the object, which are object-agnostic already: rise of the object above its spawn
   height while held, low object speed, upright, then the goal. Holding the closed grasp on the table must not be worth
   staying for.

5. Start from `cup_pick_l_approach_e3400` (user decision): it already closes a four-digit envelope with palm contact
   (0.81) in 87 % of episodes and lifts in 19 %, so the new reward first has to keep that grasp and then teach the lift,
   instead of undoing a no-contact standing habit as in rounds 2-6.

Generated code shown as below:
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 7 of the left-arm stage-2 (manip) track (`cp_l_m07`) ran 2,819 epochs in 4.0 hours, PPO-LSTM with 4,096
environments on the local RTX 5090, from `cup_pick_l_approach_e3400` (approach 0.90, envelope 0.81, lift 0.19). This is
the first round with the object-agnostic reward: no approach/envelope latch, no cylinder template, every term computed
from surface distances, contact forces and object pose. Placement rejection behaved as designed: 0.41 of resets
resampled and none failed. The round ended at the four-hour limit.

**This is the first round that lifts and succeeds.** Far-start lift rose 0.19 (the starting policy) to 0.51 by epoch
200 and 0.86-0.88 from epoch 600 on; the airborne fraction rose to 0.65-0.69. Far-start success went 0.00 to 0.19 by
epoch 1000, 0.38-0.43 by epoch 2600, and the mean successes per episode reached 0.39. That was enough for the
environment to tighten the tolerance for the first time in this track, 0.1125 to 0.101 at about epoch 2700.

**The grasp inherited from e3400 was kept, not undone.** The four digits that touch in the starting policy stay in
contact: thumb 0.75, index 0.75, middle 0.71, pinky 0.73, palm 0.54, about 3.7-3.9 links in contact, palm-to-cup gap
0.044-0.054 m. The ring finger is still exactly 0.000, as it was in e3400 and in every earlier round; the closing
command on its segment sits at 0.203 and never moves.

**The new surface inputs read as expected in training.** Palm surface distance 0.047-0.057 m while grasping, mean
minimum link surface distance 0.028-0.034 m, touching links 0.018 m (the 1.8 cm offset measured in the smoke test).

**Two things are stuck.** The hold-at-the-goal term is exactly 0.000 for all 2,819 epochs and the fraction of steps
near the goal never exceeds 0.045, while goal keypoint distance flattens at 0.216-0.230 m: the hand carries the cup
towards the goal and counts successes in passing, but does not come to rest there. And the cup is carried tilted:
tilt rose 12 degrees to 27-31 degrees and stayed, because the tilt penalty (-0.36) is small against the lift income
(+2.6).

**The largest penalty in the round is squeezing, -0.43 to -0.47 per step**, larger than the tilt penalty and about a
sixth of the lift income. The policy holds the cup harder than the term intends.

**Two transient regressions, both recovered or explained.** Between epochs 800 and 1400, while the carry to the goal
was being learned, contacts fell (digits 0.76 to 0.62, palm 0.49 to 0.32) and lift fell 0.86 to 0.56; by epoch 1600
everything was back (digits 0.75, palm 0.54, lift 0.87). After the tolerance tightened at epoch 2700, grasp fell 0.90
to 0.73 and lift 0.86 to 0.70 while success fell 0.43 to 0.28 - the task simply got harder; mean successes stayed 0.35.

**The environment's own latches are no longer meaningful as a funnel.** Approach latch 0.02-0.03 and envelope latch
0.01-0.02, because the policy no longer passes through the no-touch open-hand template the latch demands. Progress in
this track must be read from grasp / lift / success, not from the latches.

**The recording (snapshot epoch 1800, 12 far-start environments, tolerance 0.101) shows a real pick and carry.** The
hand reaches the cup by about 2.7 s, closes on it and lifts it clear of the table at about 4 s, then raises it high and
holds it up near the robot for the rest of the episode, from 5 s to 14 s. The cup is visibly tilted in the hand for the
whole carry, held against the palm by the thumb and three fingers, and it is never put down or dropped.
Video: `our_source/fj_t2r_videos/cp_l_m07_0923_0331.mp4`.

To make the code more accurate and train better robot, the feedback for improvement is:
**User instruction (2026-09-23): the cup must be carried and lifted upright.** This is the first priority of the next
round; item 2 below is not optional and the reward must make a tilted lift worth clearly less than an upright one.

What worked and must be kept: the whole object-agnostic structure. Dropping the latches and the cylinder template, and
paying continuously for surface proximity, per-digit contact with a steep count ladder, opposition, palm contact and
object height, took this track from six rounds of zero envelope and zero lift to lift 0.87 and far-start success 0.43
in one round, starting from a policy that lifted in 0.19 of episodes. None of that should be traded away, and the next
round should keep the same term structure and the same weights for reach, close, count, digit, links, palm, oppose and
lift.

The feedback for improvement is about the last third of the task, holding the object at the goal, and about how the
object is carried.

1. Holding at the goal never started: that term paid exactly 0.000 for 2,819 epochs, the near-goal step fraction stayed
   under 0.045 and the goal keypoint distance flattened at 0.216-0.230 m while successes were still being counted. The
   policy passes the goal and keeps carrying. Make coming to rest at the goal the dominant income: a continuous,
   distance-shaped term that grows as the object slows near the goal (so it is reachable before the success condition
   is met), worth clearly more per step than holding the object anywhere else, and make the lift income stop growing
   once the object is above the height the goal needs, so there is nothing to gain from lifting higher instead of
   moving to the goal.

2. The cup is carried tilted 27-31 degrees, visible throughout the recording, because the tilt penalty (-0.36 per step)
   is a seventh of the lift income (+2.6). Multiply the lift and goal income by an uprightness factor instead of adding
   a separate penalty, so a tilted carry cannot pay more than an upright one. The environment also requires the object
   upright for a success, so this is on the critical path to the tolerance shrinking further.

3. Squeezing is the largest single penalty of the round, -0.43 to -0.47 per step, larger than the tilt penalty. Either
   the threshold is too low for the force a real hold needs, or the policy is genuinely crushing the cup; the same
   contact set (four digits, palm 0.54) holds fine in e3400 without that cost. Check the threshold against the measured
   holding force and raise it, or soften the penalty, so that holding securely is not taxed as hard as over-gripping.

4. The ring finger has never touched the cup in any round of this track, including in the starting policy, and its
   closing command sits at 0.203 and does not move. The count ladder already pays for a fifth digit, so the reward is
   not the obstacle; before the next round, check whether the ring finger can reach the cup at all in this hand pose
   (joint limits, link geometry, whether its joint is among the 13 movable ones). If it cannot, stop treating five
   digits as the target and make four the full grasp; if it can, add a term that pays only for the ring finger.

5. Keep starting from the best weights of this round rather than from e3400: this round ended with grasp 0.73 and lift
   0.70 right after the tolerance tightened to 0.101, and those numbers were 0.90 and 0.86 just before it. The next
   round should continue from the epoch-1800 snapshot (or the best one) so the tighter tolerance is learned on top of a
   working pick-and-carry.

Generated code shown as below:
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
    # WEIGHTS  (no stage switch, no pose template: every term is a smooth function of the state)
    #
    # Ladder of per-step income (upright, held):
    #   travelling ~0.2  <  seated open hand ~0.4  <  curled near ~0.9  <  two digits ~2.0
    #   <  four digits opposed + palm ~5.6  <  lifted to goal height ~10.8
    #   <  AT REST AT THE GOAL ~20  (+10 on every counted success)
    # A tilted carry (27-31 deg, as in round 7) multiplies lift/carry/hold by ~0.08, so it is worth
    # about as little as not lifting at all.
    #
    # reach / close / count / digit / links / palm / oppose / lift keep round 7's weights: that
    # round took the track from 0 to lift 0.87 and success 0.43 and must not be traded away.
    # =============================================================================================
    W_REACH = 0.50        # palm toward the surface, facing it; x(0.4+0.6*grip): <=0.2 with no contact
    W_CLOSE = 0.60        # measured finger flexion x link nearness x palm seat (closing onto nothing
                          #   is capped well under one real contact)
    W_COUNT = 2.50        # digits in contact, SORTED rank weights: 2 digits 0.20, 3rd +0.26,
                          #   4th +0.39 (four digits = 0.85 = a full grasp), 5th +0.15
    W_DIGIT = 0.80        # per-digit contact mean; index/middle/ring 1.1, thumb 0.9, pinky 0.8
    W_LINKS = 0.40        # more links of each digit on the object (wrapping, not a fingertip)
    W_RING = 0.40         # ring finger only: it has never touched in this track, so if it CAN reach
                          #   it is paid directly; if it cannot, four digits are already a full grasp
    W_PALM = 0.60         # palm contact, mostly together with digits (palm alone 0.25 of it)
    W_OPPOSE = 1.00       # thumb vs fingers and palm vs fingers pushing from opposite sides (normals)
    W_LIFT = 5.00         # height above spawn, x uprightness, CAPPED at the height the goal needs
    W_RISE = 0.80         # upward object speed while held, only until `lifted` latches (no re-farming)
    W_GOAL = 3.00         # held object closing on goal_pos, x uprightness
    W_HOLD = 9.00         # AT REST AT THE GOAL, upright: the dominant income of the whole reward
                          #   (round 7: this term paid exactly 0.000 for 2,819 epochs)
    W_SUCCESS = 10.0      # bonus on the step the environment counts a success
    P_DESCEND = 1.00      # lowering the held object < keeping it still < raising it
    P_STALL = 1.00        # closed grasp parked on the table, growing with episode progress
    P_SETDOWN = 1.50      # object back on the table after it was lifted (no second lift income)
    P_DROP = 1.00         # lifted and then no longer held
    P_TILT = 0.50         # small: uprightness is enforced by MULTIPLYING the lift/goal income
    P_SHOVE = 0.50        # pushing / tipping the object around while it is still on the table
    P_SQUEEZE = 0.50      # softened (0.80): a secure four-digit hold must not be taxed
    P_TABLE = 0.50        # palm / finger links driven into the table
    P_ARM_VEL = 0.12      # smoothness: <=~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.15      # linear object speed while held (a 5 cm/s lift costs ~0.015)
    P_CUP_SPIN = 0.08     # angular object speed while held
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512-point sampling)
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    TILT_DEAD = math.radians(5.0)    # a few degrees of wobble is not a tilted carry
    TILT_SCALE = math.radians(10.0)  # 15 deg -> x0.37, 30 deg -> x0.08 of the lift/goal income
    SQ_TOTAL = 40.0       # round 7 measured -0.45/step at a 15 N threshold with a grip that works
    SQ_LINK = 15.0        # per-link ceiling, likewise raised from 6 N

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
    # UPRIGHTNESS: one factor that multiplies every lift / carry / hold income.
    # Round 7 carried the cup at 27-31 deg because a -0.36 penalty sat against +2.6 of lift income.
    # As a multiplier, 30 deg costs 92 % of the lift instead of 14 %.
    # =============================================================================================
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)   # (N,) in (0,1]

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
    # digits that arrive LAST carry the most: 2 digits = 0.20, 3 = 0.46, 4 = 0.85, 5 = 1.00.
    # Four digits are a full grasp (the ring finger has never touched in this track), but a fifth
    # still adds twice what the first one does.
    rank_w = torch.tensor([0.07, 0.13, 0.26, 0.39, 0.15], device=device, dtype=dtype)
    touch_desc, _ = torch.sort(digit_touch, dim=-1, descending=True)
    r_count = W_COUNT * cq * (touch_desc * rank_w).sum(dim=-1)

    # per-digit mean: index, middle and ring weighted at least as much as thumb and pinky
    digit_w = torch.tensor([0.9, 1.1, 1.1, 1.1, 0.8], device=device, dtype=dtype)
    digit_score = 0.6 * digit_touch + 0.4 * digit_links                                      # (N,5)
    r_digit = W_DIGIT * cq * (digit_score * digit_w).sum(dim=-1) / digit_w.sum()

    # more links per digit on the object = wrapping along the body, not a fingertip
    r_links = W_LINKS * cq * digit_links.mean(dim=-1)

    # ring finger only (index 3): pays if it can reach the object at all; costs nothing if it cannot
    r_ring = W_RING * cq * digit_score[:, 3]

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

    # "held" = palm contact plus opposing digits, full at FOUR digits (smooth).
    # Floors on palm (0.4) and opposition (0.5) keep force/normal flicker from zeroing a real grasp.
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)

    # =============================================================================================
    # LIFT: x uprightness, and CAPPED at the height the goal itself needs
    # =============================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    on_table = 1.0 - torch.tanh(height / 0.02)
    # the goal sits 0.21-0.28 m above the spawn height; read it from the goal itself (object-agnostic)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    h_eff = torch.minimum(height, h_need)          # nothing to gain from lifting past the goal
    # steep over the first 3 cm (breaks the table habit), still rising all the way to goal height
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)
    r_lift = W_LIFT * held * upright * lift_shape
    vz = ctx.cup_lin_vel[:, 2]
    # upward speed pays only until `lifted` latches: putting the object down and raising it again
    # cannot re-earn it
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)
    # lowering the held object costs: descending < still < raising. It fades out at goal height, so
    # descending ONTO the goal from above is not taxed (the lift no longer pays for going higher).
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)
    # a closed grasp parked on the table is not worth staying for (free early, costly late)
    p_stall = P_STALL * held * on_table * ctx.episode_progress
    # object back on the table after a lift, and a lifted object no longer held
    p_setdown = P_SETDOWN * lifted * on_table
    p_drop = P_DROP * lifted * (1.0 - held)

    # =============================================================================================
    # GOAL: carry there, then COME TO REST there, upright -- the dominant income
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    r_goal = W_GOAL * held * upright * (1.0 - torch.tanh(ctx.goal_dist / 0.15))
    # two scales, fixed (never success_tol): the coarse one gives a real gradient from the 0.22 m
    # the previous round stalled at (~1.0/step), the fine one makes the last few cm the big money.
    near_goal = (0.35 * (1.0 - torch.tanh(ctx.goal_dist / 0.25))
                 + 0.65 * torch.exp(-ctx.goal_dist / 0.05))
    still = torch.exp(-cup_speed / 0.08) * torch.exp(-spin / 2.0)
    # arriving near the goal already pays; settling there pays four times more. At rest on the goal
    # this is ~9/step, about twice the whole lift income, so passing through it is never optimal.
    r_hold = W_HOLD * held * upright * near_goal * (0.25 + 0.75 * still)
    r_success = W_SUCCESS * success

    # =============================================================================================
    # SAFETY AND SMOOTHNESS (small next to the stage income)
    # =============================================================================================
    # small explicit tilt cost: uprightness is mainly enforced as a multiplier above
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    p_shove = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                          + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    # squeeze: thresholds raised and weight softened, because the same four-digit grip that holds
    # fine was paying -0.45/step in round 7 -- a sixth of the lift income for gripping securely
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    floor_pen = torch.tanh((0.002 - (ctx.hand_z_min - ctx.table_z)).clamp(min=0.0) / 0.01)
    p_table = P_TABLE * (palm_pen + floor_pen)
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * held) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = held * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                        + P_CUP_SPIN * torch.tanh(spin / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_reach + r_close
        + r_count + r_digit + r_links + r_ring + r_palm + r_oppose
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
        "grasp_ring": r_ring,
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 8 of the left-arm stage-2 (manip) track (`cp_l_m08`) ran 3,229 epochs in 4.6 hours, PPO-LSTM with 4,096
environments on the local RTX 5090, from round 7's epoch-1800 snapshot. The round answered the user's instruction that
the cup be lifted and carried upright: uprightness became a multiplier on the lift, rise, goal and hold income instead
of a small penalty, the lift income was capped at the height the goal needs, holding at the goal was raised to 9.0 and
the squeeze thresholds were raised. Placement rejection behaved as designed: 0.42 of resets resampled and none failed.

**The cup is now carried upright.** Cup tilt fell 28 degrees to 4.3-6.7 degrees and stayed there for the last 1,400
epochs. At 5 degrees the uprightness factor is about 0.95, against 0.15 at the 24 degrees where the round sat until
epoch 1400, so the policy now collects essentially all of the lift and goal income. The change was not gradual: tilt
held at 23-25 degrees from epoch 200 to 1400 and then fell to 10 degrees by epoch 1700 and to 5 degrees by epoch 2000.

**The grasp improved with the upright pose, and the ring finger finally touches.** Ring contact was exactly 0.000 in
every round of this track including the starting policy, and in this round it rose from 0.000 to 0.156 at epoch 2500
and 0.40 for the last 600 epochs. Links in contact rose 3.7 to 4.36. The other digits are at their best values of the
track (thumb 0.79, index 0.77, middle 0.76, pinky 0.77) with palm contact 0.64 and a palm gap of 0.039 m. Grasp
episodes reached 0.94 and lift 0.89. The drop penalty fell to -0.09.

**Success drove the tolerance down twice more.** Far-start success oscillated 0.16-0.55 and the mean successes per
episode reached 0.39, which shrank the goal tolerance 0.101 to 0.082 at about epoch 2300 and to 0.066 by epoch 3100 -
41 % tighter than the 0.1125 this track started from, though still far from the 0.015 m floor the task defines. Each
tightening cost success for a few hundred epochs (0.47 to 0.22 after the last one) and then recovered; grasp and lift
were unaffected.

**Holding at the goal is now the only blocked stage.** That term rose 0.000 (round 7) to 0.53-0.63, but its maximum is
9.0, so the policy earns about 6 % of it. The fraction of steps near the goal is 0.019-0.050 and the goal keypoint
distance flattened at 0.163-0.174 m, roughly 2.5 times the current tolerance. The pattern is unchanged from round 7:
the cup is carried towards the goal and successes are counted in passing, but the hand does not settle there.

**Two of my own calls in this round were wrong and are worth recording.** At epoch 1126 I read the flat 24-degree tilt
as a hard limit of the four-digit grasp; it was a plateau before a transition. At epoch 2539 I read the ring finger's
zero contact as likely structural; it started touching 700 epochs later once the cup was upright. Both were premature
readings of a flat curve.

**The recording (snapshot epoch 3200, 12 far-start environments, tolerance 0.066) shows an upright pick and carry that
never settles.** The hand reaches the cup by about 2.7 s and lifts it at about 4 s, and from then on the cup is held
visibly upright in the hand for the whole episode, gripped against the palm with the fingers around it. The hand keeps
moving it, up near the robot at 5-7 s, back down towards the table at 9-11 s and up again at 12-14 s, without coming to
rest anywhere. The cup is never tipped, put down or dropped.
Video: `our_source/fj_t2r_videos/cp_l_m08_0923_1235.mp4`.

To make the code more accurate and train better robot, the feedback for improvement is:
What worked and must be kept: making uprightness a multiplier on the lift, rise, goal and hold income rather than a
separate penalty. That single change took cup tilt from 27-31 degrees to 4-7 degrees, and with the upright pose the
grasp improved on its own: the ring finger touched for the first time in the whole track (0.000 to 0.40), links in
contact rose to 4.36, palm contact to 0.64, grasp episodes to 0.94 and lift to 0.89. The capped lift income and the
raised squeeze thresholds also behaved as intended (squeeze -0.45 in round 7, -0.15 here). Keep all of this, and keep
the object-agnostic structure: no latches, no cylinder template.

The feedback for improvement is one stage: coming to rest at the goal. It is the only part of the task that has never
worked, in either round of the new reward.

1. The hold term pays 0.53-0.63 out of a maximum of 9.0, the near-goal step fraction is 0.019-0.050, and the goal
   keypoint distance has been flat at 0.163-0.174 m for 1,500 epochs while the tolerance has shrunk to 0.066 m. The
   shape of the term is the problem, not its weight: it was already raised 5.0 to 9.0 this round and the distance did
   not move. Two things to change together. First, make the approach to the goal pay on a scale matched to where the
   policy actually is: the fine ramp is 0.05 m wide, so at 0.17 m it contributes almost nothing and the policy sees a
   flat landscape between 0.17 m and the tolerance. Add a coarse term that keeps rising over the whole 0.0-0.30 m
   range, so there is a gradient to follow from where it sits now. Second, pay for reducing the distance (the change
   in goal_dist between steps) as well as for the distance itself, so moving towards the goal is rewarded even before
   arriving.

2. Carrying the cup around with no destination must be worth less than arriving. The lift income is 2.2-2.6 per step
   while the goal and hold terms together pay about 1.0, so hovering with the cup upright remains the best steady
   state. Now that the cup is reliably upright and held, reduce the standing lift income once the object is above the
   height the goal needs (it is already capped; make it decay with time held, or scale it by proximity to the goal),
   so that the only way to keep earning is to close on the goal.

3. The success bonus is tiny in practice (0.015-0.036 per step averaged) while each success also restarts the episode
   clock. If the intent is that successes are the point, the bonus should be large enough to be visible next to the
   2.5 per step the policy earns for simply holding the cup up.

4. Keep the tolerance curriculum as it is. It has moved 0.1125 to 0.066 on its own and each tightening was absorbed
   within a few hundred epochs. Do not compensate for it in the reward.

5. Start the next round from this round's epoch-3200 snapshot (`cp_l_m08/nn/last_..._ep_3200_rew_4891.019.pth`), which
   holds the upright five-digit grasp at tolerance 0.066.

6. Worth recording for the next reader: two flat curves in this round were read as hard limits and were not. Tilt sat
   at 24 degrees for 1,200 epochs before falling to 5, and the ring finger sat at 0.000 for 2,500 epochs before rising
   to 0.40. Do not conclude that a plateau is structural in this task without a geometric check.

Generated code shown as below:
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 9 of the left-arm stage-2 (manip) track (`cp_l_m09`) ran 1,712 epochs in 4.0 hours, PPO-LSTM with 4,096
environments, this time on the vision-3090 machine (RTX 3090, about 8.0 s per epoch against 4.6 s on the local 5090),
from round 8's epoch-3200 snapshot. Two things changed: the reward budget was moved to the end of the task (reach and
close 0.30 and fading once held, keeping the grasp 0.80, lift 1.00 capped at goal height and multiplied by horizontal
closeness to the goal, transport 2.50, at rest at the goal 10.0, success 100), and the environment now applies an
external wrench to the object once it is lifted and inside a fixed radius of the goal, with an ADR level tied to how
well the policy does there. Placement rejection behaved as designed: 0.42 of resets resampled and none failed.

**The tolerance curriculum went further than in any previous round.** Starting from 0.1125 again (a new run restarts
it), the goal tolerance fell to 0.047 m by epoch 1700, against 0.066 m at the end of round 8 and 0.015 m as the task's
floor. Far-start success held at 0.85-0.87 through the first three tightenings and eased to 0.77 at the tightest
setting; mean successes per episode were 0.39-0.53. In round 8 each tightening cut success roughly in half.

**Time spent at the goal grew steadily, which is what the round was for.** The hold-at-the-goal term rose 0.054 to
0.215, the transport term 0.078 to 0.207, and the time held inside the disturbance radius rose 24 to 41 steps with the
fraction of steps in that radius rising 0.24 to 0.43. The goal keypoint distance fell 0.204 to 0.167 m - the first
sustained fall in this track - but it is still 3.5 times the current tolerance, so the hand still does not come to rest
on the goal.

**The grasp first loosened, then recovered.** Cutting the grasp income from 3.86 to 0.80 per step cost contact
immediately: digits fell 0.75 (round 8) to 0.43-0.47 by epoch 800, palm 0.64 to 0.34, the palm gap opened 0.039 to
0.078 m and the airborne fraction fell 0.67 to 0.32. From epoch 800 on it recovered to digits 0.53-0.56, palm 0.43, gap
0.057 m and airborne 0.46, with the drop penalty easing -0.49 to -0.30. So the reduced grasp income is survivable, but
the grasp is measurably looser than round 8 and the drop penalty is the largest negative term of the round.

**The ring finger lost the contact it had gained.** It was 0.40 at the end of round 8, and here went 0.28 to 0.05 at
epoch 1200 before partly recovering to 0.13. The other four digits strengthened over the same interval, so the hand
re-shaped around a four-digit hold when the wrench started arriving.

**The cup stays upright** at 7.3-8.4 degrees throughout, and the hand never approaches the table (lowest point 0.40 m).

**The recording (snapshot epoch 800, tolerance 0.08, 12 far-start environments) shows repeated successful episodes, not a pick-release loop.** The hand reaches the cup at about 2.7 s, grasps it and lifts it upright by 4.1 s, and at about 4.7 s the scene resets: the arm is back at its start pose and the cup stands at a new place on the table. The same sequence then repeats twice more inside the 15 s recording. Frames at 0.5 s spacing across the transition confirm a reset rather than the hand putting the cup down. The metrics agree: the set-down penalty averages -0.001 (of a 3.0 weight), so deliberate set-downs essentially never happen, and `ctrl/prev_ep_successes_mean` is 3.8-4.0, i.e. episodes end after the policy has collected close to the environment's limit of 5 successes. What the recording shows is the policy finishing the task repeatedly, with the cup upright each time.

**Caveat on the snapshot.** rl_games only writes a snapshot when the mean reward improves, and with the income
concentrated in the success bonus nothing was written after epoch 800. The recording therefore shows an early policy
(tolerance 0.089 era), not the epoch-1712 state with tolerance 0.047. The final weights of this round do not exist.

To make the code more accurate and train better robot, the feedback for improvement is:
What worked and must be kept: moving the income to the end of the task, and the goal-gated wrench. Together they gave
the first sustained fall in goal distance (0.204 to 0.167 m), tripled the hold term, raised time inside the goal radius
from 24 to 41 steps, and carried the tolerance from 0.1125 to 0.047 without the success collapse that every earlier
tightening caused. Keep the ceilings, keep uprightness as a multiplier, keep the wrench.

The feedback for improvement is that the task ends too easily: episodes finish after about four cheap successes at a
goal that never moves, so the policy is never asked to transport the object anywhere or to stay put.

1. Successes are cheap and end the episode early. Episodes finish with 3.8-4.0 successes out of the limit of 5, because the goal sits directly above the spawn point and is re-sampled in the same place, so hovering there collects the bonus again and again until the episode truncates. The reward should pay the large bonus for the first success and pay later ones through the hold term, and the environment's goal sequence should move the goal after a success (`goal_delta_distance` is 0.0 today) so that a second success means transporting somewhere new. Without that, the tolerance curriculum is being driven by a task that ends before it gets hard.

2. The hold term is still not reaching the policy: 0.215 out of 10.0, because at 0.167 m the object is 3.5 tolerances
   away and the fine part of the ramp is narrow. Widen the paying region so that being inside about two tolerances
   already pays a large fraction, and make the term grow as the object stays there (a dwell factor), so that standing
   still at the goal beats arriving repeatedly.

3. The grasp is looser than round 8 (digits 0.55 against 0.75, palm 0.43 against 0.64, gap 0.057 against 0.039 m) and
   the drop penalty is the round's largest negative term. The 0.80 grasp ceiling is close to the lower bound of what
   works; if anything it should go to about 1.2 now that the goal terms are larger, rather than lower.

4. The ring finger fell 0.40 to 0.13 when the wrench arrived. Check whether it can contribute at all under the
   disturbance, or whether the four-digit hold is simply the stable answer; if it is, stop paying a ring-specific term.

5. Next round should start from this round's epoch-800 snapshot, which is the only weight file that exists. Consider
   also saving on a schedule rather than on reward improvement for this reward shape, or the end-of-round weights will
   keep being lost.

6. Machine note: this round ran on vision-3090 at about 8.0 s per epoch, 1.7 times slower than the local 5090, so a
   four-hour round is roughly 1,700 epochs instead of 3,000. Judge plateaus against epochs, not wall clock.

Generated code shown as below:
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
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Object-agnostic and arm-agnostic throughout: only surface distances/normals, contact
    # forces, object pose/velocity and the goal. No approach/envelope latch, no radius/axis/half
    # height, no world-frame direction constant.
    #
    # Per-step ceilings (the budget IS the design):
    #   reach + close    0.60   pre-contact only, x (1 - held): it vanishes once the object is held
    #   keep the grasp   1.20   per-digit contact + count ladder + palm + opposition
    #   lift             1.00   x uprightness x horizontal closeness to the goal, capped at goal height
    #   transport        2.50   1.70 on goal_dist over 0-0.30 m + 0.80 on the RATE it falls (signed)
    #   hold at rest    10.00   real hold, upright, at the goal, still, growing with dwell
    #   first success   60.00   one-off; later successes 2.0, so staying pays, not re-arriving
    #
    # Steady states: hovering upright away from the goal ~1.6/step; at rest on the goal ~10-13/step.
    # =============================================================================================
    W_REACH = 0.35        # palm towards the surface and facing it            } 0.60 total,
    W_CLOSE = 0.25        # measured flexion x link nearness x palm seat      } x (1 - held)
    W_COUNT = 0.50        # digits in contact, sorted rank ladder             }
    W_DIGIT = 0.35        # per-digit contact mean, every digit alike         } 1.20 total: enough
    W_PALM = 0.15         # palm contact, only together with digits           } to build and keep a
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)  } grasp, not to live on
    W_LIFT = 0.80         # height above spawn, capped at what the goal needs }
    W_RISE = 0.20         # upward speed, only until `lifted` latches         } 1.00 total
    W_GOAL = 1.70         # goal_dist, constant slope over the whole 0-0.30 m }
    W_PROGRESS = 0.80     # signed rate at which goal_dist falls              } 2.50 total
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode only (~6 steps of a perfect hold)
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 6.00         # lifted and no longer held: on top of losing up to 10 of hold income
    P_SETDOWN = 3.00      # back on the table after a lift: the lift income is not re-earnable
    P_STALL = 1.00        # closed grasp parked on the table, growing with episode progress
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    P_SQUEEZE = 0.50      # force beyond what a hold needs (thresholds measured in round 7)
    P_TABLE = 0.60        # palm or finger links driven into the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the goal income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: 0.02 at 0.1 m/s against +0.6 of transport income,
    P_CUP_SPIN = 0.06     #   so carrying to the goal always beats standing still
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
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
    # CONTACT AND GRASP QUALITY
    # =============================================================================================
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5) does this digit touch
    digit_links = link_c.mean(dim=-1)                              # (N,5) how many of its links do
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)                              # (N,) 0..5
    finger_any = digit_touch[:, 1:].amax(dim=-1)                   # (N,) any of the four fingers
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    # ONE uprightness multiplier on every lift / transport / hold income. An additive tilt penalty
    # does not keep the object upright; multiplying does (tilt fell 28 deg -> 5 deg when it did).
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)      # (N,) in (0,1]

    # ---- palm against the object's REAL surface (works for any shape, either arm) ----------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    # facing: the palm normal points at the nearest surface point AND against the outward normal
    # there. Both are dot products of context vectors, so nothing is tied to a world axis.
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / 0.02)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face), palm_c)
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

    # "held" = palm plus several OPPOSING digits (smooth, full at four). The floors on palm and
    # opposition keep force/normal flicker from zeroing a real grasp when the wrench arrives.
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)                      # (N,)
    # the 10/step asks for more than the lift does: palm and opposition enter a SECOND time, so a
    # fingertip touch can never collect the dominant income.
    hold_grip = held * (0.5 + 0.5 * palm_c) * (0.5 + 0.5 * opp_sat)

    # =============================================================================================
    # REACH AND CLOSE -- 0.60 total, and it FADES OUT once the object is really held.
    # From random weights this is the only income before contact, so it keeps a long-range pull;
    # but its zero-contact ceiling stays below what one real digit contact pays, which is the rule
    # that finally produced contact in this track.
    # =============================================================================================
    fade = (1.0 - held).clamp(min=0.0, max=1.0)
    reach_far = (1.0 - torch.tanh(palm_d / 0.20)) * (0.4 + 0.6 * face)   # 0.20 m covers 0.15-0.49 m
    reach_near = palm_near * face                                        # the last 2 cm
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near) * (0.45 + 0.55 * grip)

    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)        # saturates at half flexion: no crushing drive
    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)      # (N,5,3)
    near_digit = link_near.amax(dim=-1)                                                     # (N,5)
    # a straight finger lying along the object gets nothing (flex = 0), a curled finger in the air
    # gets nothing (near = 0): only curling ONTO the surface with the palm seated pays.
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total (raised from 0.80: at 0.80 the grasp loosened and the drop
    # penalty became the largest negative term of the round).
    # =============================================================================================
    # sorted rank weights (sum 1.0): 2 digits 0.18, 3 = 0.42, 4 = 0.80, 5 = 1.00. The digits that
    # arrive LAST carry the most, so a two-digit pinch is worth under a quarter of a real envelope.
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    touch_desc, _ = torch.sort(digit_touch, dim=-1, descending=True)
    r_count = W_COUNT * cq * (touch_desc * rank_w).sum(dim=-1)

    # per-digit mean, EVERY digit weighted alike (no digit-specific term: four digits are already a
    # full grasp on the ladder, and a fifth still adds more than the first one did)
    digit_score = 0.6 * digit_touch + 0.4 * digit_links                                     # (N,5)
    r_digit = W_DIGIT * cq * digit_score.mean(dim=-1)

    # palm contact counts mostly together with digits, never on its own (it funded a flat open hand
    # pressed against the object when it did)
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # =============================================================================================
    # LIFT -- 1.00 total, x uprightness, capped at the height the goal needs, and multiplied by
    # HORIZONTAL closeness to the goal: hovering at the right height in the wrong place earns ~0.2.
    # =============================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    on_table = 1.0 - torch.tanh(height / 0.02)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)   # read from the goal
    h_eff = torch.minimum(height, h_need)        # nothing to gain from lifting past the goal
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)     # steep first 3 cm
    d_xy = (ctx.cup_pos[:, :2] - ctx.goal_pos[:, :2]).norm(dim=-1)
    goal_xy = torch.exp(-d_xy / 0.12)
    r_lift = W_LIFT * held * upright * goal_xy * lift_shape
    vz = ctx.cup_lin_vel[:, 2]
    # upward speed pays only until `lifted` latches, so setting the object down and raising it again
    # can never re-earn the lift income
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total: 1.70 on the distance, 0.80 on the RATE at which it falls
    # =============================================================================================
    # a CONSTANT slope over the whole 0-0.30 m range (a narrow ramp left the landscape flat exactly
    # where the policy sat), plus a fine ramp for the last few centimetres
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = W_GOAL * held * upright * (0.55 * goal_far + 0.45 * goal_fine)

    # paying for the REDUCTION of goal_dist between steps. The function must be pure, so the rate is
    # read from the state: the closing speed towards the goal is d(goal_dist)/dt up to the rotation
    # part, which uprightness already handles. SIGNED, so drifting away costs what approaching pays
    # and an out-and-back orbit nets negative (tanh is concave). Faded inside ~3 cm so that jitter at
    # rest, and the environment's pushes, neither pay nor punish.
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = W_PROGRESS * held * upright * approach_scale * torch.tanh(closing / 0.10)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, the dominant income of the whole reward
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    # the paying region is scaled by the CURRENT tolerance, so it always covers where the curriculum
    # is: exp(-d / 4*tol) pays 0.78 at one tolerance and 0.61 at two, and a fixed fine ramp keeps the
    # last centimetres worth crossing as the tolerance shrinks towards 0.015 m.
    tol_scale = (4.0 * ctx.success_tol).clamp(min=0.06)
    near_goal = 0.65 * torch.exp(-ctx.goal_dist / tol_scale) + 0.35 * torch.exp(-ctx.goal_dist / 0.03)
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    # 0.30 + 0.70*still, not `still` alone: the environment pushes and twists the object at the goal,
    # so holding station against a push must stay profitable while drifting off loses all of it.
    settle = 0.30 + 0.70 * still
    # dwell: episode_progress restarts after every success, so at the goal it measures how long the
    # object has been standing there. Settling therefore pays strictly more than arriving.
    dwell = 0.60 + 0.40 * torch.tanh(ctx.episode_progress / 0.25)
    r_hold = W_HOLD * hold_grip * upright * near_goal * settle * dwell

    # the FIRST success of an episode is the big one; later ones decay hard, because the episode
    # counts up to five and the income for staying must come from r_hold, not from re-arriving.
    # The factor is the mean of the two possible conventions for num_successes (whether it already
    # includes this step's success or not), so it decays monotonically under either one.
    n_succ = ctx.num_successes.to(dtype)
    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = success * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING -- each priced above the income it would replace
    # =============================================================================================
    p_drop = P_DROP * lifted * (1.0 - held)             # 6.0 on top of losing up to 10 of hold
    p_setdown = P_SETDOWN * lifted * on_table           # no second lift income after a set-down
    p_stall = P_STALL * held * on_table * ctx.episode_progress   # a closed grasp parked on the table
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
    # object motion while held: 0.02 at 0.1 m/s against +0.6 of transport income at the same speed,
    # so carrying to the goal is always better than standing still
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

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 10 is the first round of the left/right parallel plan: the same byte-identical reward, both arms from random
weights, left (`cp_l_m10`) on vision-3090 and right (`cp_r_m10`) on the local 5090, PPO-LSTM with 4,096 environments.
It ran 11.7 hours: 5,005 epochs on the left (8.6 s/epoch) and 8,270 on the right (5.1 s/epoch). No Traceback, no PhysX
error, placement rejection 0.42 with no failures on either side. The round was stopped on the stall criterion, not on a
clock: the wrench ADR never left level 0 because its gate never opened.

**Both arms learned to reach and to grasp, and nothing after that.** Palm-to-surface distance fell to 0.045 m (left)
and 0.037 m (right), inside the 0.06 m gate. Grasp episodes reached 0.83 left and 0.98 right and then sat flat for
4,000 and 7,000 epochs respectively. Lift is 0.008 left and 0.000 right; the airborne fraction is 0.001 and 0.000.
The object never leaves the table, so goal distance stays at its start value (0.25 m), success is 0, the tolerance
never moved from 0.1125, and the goal-gated wrench never fired: `wrench_near_goal_frac` and `wrench_goal_dwell_ep`
are exactly 0 on both sides and no ADR level was ever logged.

**The cause is income available with the object still on the table.** On the left the hold term pays 0.54-0.69 per
step throughout. The goal sits about 0.25 m above the object's start, and the hold term's coarse ramp is shaped on
goal_dist alone, so a cup resting on the table inside a closed hand already collects most of what the policy can
reliably earn. Lift pays at most 1.00 and only if the policy risks moving. The result is the parking equilibrium seen
on 09.21 in the IKER track: grasp, then stand still. The stall penalty (1.0, scaled by episode progress) was not
enough to break it.

**The two arms converged on different grasps from the identical reward.** Left: thumb 0.49, index 0.38, middle 0.30,
ring 0.39, pinky 0.59, palm 0.38, 2.2-2.6 links in contact - a whole-hand hold. Right: thumb 0.74, index 0.69, ring
0.76, but middle 0.011, pinky 0.002 and palm 0.001 - a three-digit pinch with no palm. Both reach the same grasp
episode score, so the reward cannot tell them apart; the palm term (0.15 of the 1.20 grasp budget) is too small to
make the palm worth acquiring. This is a reward-shape asymmetry, not a mirror defect: the mirror contract tests pass,
the right arm's earlier index-contact deficit resolved by itself (0.05 to 0.69), and left/right start poses are
verified mirrors.

**Cup tilt stayed small on both sides** (3.4 degrees left, 0.6 right), and the drop penalty is negligible, consistent
with an object that is gripped but never moved.

**One logging caveat.** `contact/*_at_success` reads 0.95-1.00 on both runs although no success ever occurred. These
event-EMA channels are not trustworthy until the first real success; do not read them as evidence.

**The recordings confirm the reading: both arms grasp and then stand still.** Left (snapshot epoch 5000) reaches the
cup by about 2.7 s, closes the whole hand around it with the palm against the side, and holds that pose without moving
the cup from 4 s to the end of the episode. Right (snapshot epoch 7200) reaches the cup by about 4 s and closes a
narrower grip on it, again without moving it. In neither recording does the cup leave the table or tip over.
Videos: `our_source/fj_t2r_videos/cp_l_m10_0924_1128.mp4` and `cp_r_m10_0924_1127.mp4`.

**Where on the object the hand grips, measured.** The graspable band of the standard cup spans z 0.232-0.332 in
env-local coordinates (table top 0.205, object origin 0.282) and its rim is at about 0.383. The lowest finger link
averages 0.339 on the left and 0.327 on the right, i.e. the hand hovers at the very top of the band, just under the
rim, and closes there. The palm never seats: palm-to-cup gap 0.070-0.078 m, palm contact 0.38 left and 0.001 right,
with only 2.1-2.5 of five digits and 2.2-2.6 of fifteen links in contact. The user's reading of the recordings is the
same on both arms: the hand cannot get below the rim, so index and thumb close on air beside it.

**This is not a workspace limit.** The 0.27 m palm-box floor printed at boot belongs to the palm-target action mode;
this task commands seven arm joint increments clamped only by joint limits (arm_target_box_rad 0.0), and the lowest
finger link already reaches 0.21 m (table level) in the worst environments. The hand can descend; the reward does not
ask it to. Proximity is measured to the NEAREST surface point, and from a hand coming in above the cup the nearest
point is the rim edge, whose normal points partly upward - so 'bring the palm close and face the surface' is satisfied
at the rim, and nothing pays for sliding down the side wall to where a real envelope grasp lives.

To make the code more accurate and train better robot, the feedback for improvement is:
What worked and must be kept: the first half of the chain now trains from random weights on both arms. Reaching and
grasping were learned in under 1,000 epochs and held for thousands more (palm-to-surface 0.037-0.045 m, grasp episodes
0.83-0.98, four to five digits in contact on the left). Keep the reach and grasp terms as they are, keep uprightness as
a multiplier, keep the object-agnostic and arm-agnostic rule, and keep the first-success-only bonus.

The feedback for improvement is a single structural fault, and one weight.

1. The second half of the task must pay nothing while the object is on the table. Transport, being near the goal, the
   hold term and the success bonus were all shaped on goal_dist alone. Because the goal sits about 0.25 m above the
   object's start, a cup gripped on the table already collected 0.54-0.69 per step of hold income - more than the
   entire lift term could offer (1.00), and without risk. Multiply every second-half term by a smooth function of how
   far the object has actually risen above ctx.cup_spawn_pos[:, 2], so that all of it is zero on the table. This is
   the same parking trap recorded on 09.21: a dense state-proportional term made "hold the pose and stay" optimal.

2. Lifting must be the largest single step up in income. Raise the lift ceiling from 1.00 to about 2.5, rising with
   height to the height the goal needs, and let it be the only door to the goal terms (point 1). Make a closed grasp
   parked on the table actively costly as the episode progresses.

3. Pay the palm properly. From the identical reward the left arm acquired a whole-hand grasp (palm 0.38, five digits)
   and the right arm a three-digit pinch with palm 0.001. The palm term is 0.15 of a 1.20 grasp budget, which is not
   enough to be worth acquiring, and a pinch is a worse starting point for lifting and for surviving the wrench. Give
   the palm a clearly larger share, and make the digit-count ladder require the palm for its top rungs.

4. Nothing is known yet about robustness under the wrench, because the gate (lifted AND near the goal) never opened.
   Once lifting works, the ADR schedule should be watched from the first epoch it fires: level, fire fraction, dwell,
   and whether per-digit contact at success holds up as the level rises.

5. Keep both arms on the identical file and keep running them in parallel; the divergence in grasp style is the most
   useful signal this round produced, and it only shows up when the same reward is trained twice.

6. Make the reward ask for the side wall, not the nearest point. Both arms grip at the rim because the nearest
   surface point from above is the rim edge. Two object-agnostic levers: (a) weight proximity and facing by how
   horizontal the surface normal at the contact point is - a side wall has a horizontal normal, a rim or a lid does
   not, and gravity is shared by both arms so this stays mirror-safe; (b) pay for the number of LINKS of each digit in
   contact rather than for a digit touching at all, since a fingertip on the rim gives one link while a wrap around
   the wall gives two or three. Round 10 sat at 2.2-2.6 links of fifteen.

Generated code shown as below:
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
    """How HORIZONTAL a surface normal is: 1 on a vertical side wall, 0 on a rim face, a lid or
    the table, ~0.3 on a 45 deg rim edge. This is the only place the world frame is used besides
    height, and gravity is shared by the left arm and the mirrored right arm, so it is mirror-safe.
    It is what asks for the WALL instead of the nearest point, which from above is the rim."""
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
    #   reach            0.60   pre-contact only, x (1 - held): it vanishes once the object is held
    #   keep the grasp   1.20   links-per-digit + count ladder (top rungs need the palm) + palm + opposition
    #   lift             2.50   x uprightness x real hold, rising to the height the goal needs
    #   transport        2.50   1.60 on goal_dist over 0-0.30 m + 0.90 on the signed RATE it falls
    #   hold at rest    10.00   real hold, upright, at the goal, still, growing with dwell
    #   first success   60.00   one-off; later successes 2.0, so staying pays, not re-arriving
    #
    # EVERY term from "transport" down is multiplied by `off_table`, a smooth function of the
    # object's rise above its own spawn height: the whole second half of the task pays exactly
    # ZERO while the object sits on the table. That is the fault of round 10 (0.54-0.69/step of
    # hold income with the object never lifted) and it is fixed structurally, not by weights.
    #
    # Steady states: grasp parked on the table 1.2 early and 0.5 late (minus the time pressure);
    # lifted and hovering ~4.5; at rest on the goal ~13; a success spike of 60.
    # =============================================================================================
    W_REACH = 0.38        # palm towards a SIDE WALL of the surface and facing it   } 0.60 total,
    W_CLOSE = 0.22        # measured flexion x link nearness to a wall x palm seat  } x (1 - held)
    W_COUNT = 0.40        # digit-count ladder, sorted; the top rungs need the palm }
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike        } 1.20 total:
    W_PALM = 0.30         # palm contact (2x round 10: a pinch must not score alike)} build and keep
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)        } a grasp, not live on it
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height }
    W_RISE = 0.40         # upward speed, only until `lifted` latches                         } 2.50
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m       } 2.50 total,
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls                    } x off_table
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode (~5 steps of a perfect hold)
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 8.00         # lifted and no longer held: on top of losing up to 10 of hold income
    P_SETDOWN = 4.00      # back on the table after a lift: lift income is never re-earnable
    P_STALL = 0.80        # time pressure while the object is still on the table, late in the episode
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
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5) does this digit touch
    # NUMBER OF LINKS of each digit, made convex so one link is worth far less than two or three:
    # 1 link -> 0.19, 2 links -> 0.54, 3 links -> 1.00. A fingertip on the rim barely scores; a
    # digit wrapped around the wall scores fully. Round 10 sat at 2.2-2.6 links of fifteen.
    digit_links = (link_c.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)                              # (N,) 0..5
    finger_any = digit_touch[:, 1:].amax(dim=-1)                   # (N,) any of the four fingers
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    # ---- palm against the object's real surface, and against a WALL of it -----------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    # facing: the palm normal points at the nearest surface point AND against the outward normal
    # there. Both are dot products of context vectors, so nothing is tied to a world axis.
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])           # 1 at a wall, ~0 at the rim/lid
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / 0.02)
    # seating means: close to the surface, facing it, and the surface there is a WALL. Real palm
    # contact always counts, whatever the geometry read says.
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side, palm_c)
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
    # from round 10 (0.4+0.6 -> 0.3+0.7): the right arm's three-digit pinch with palm 0.001 must
    # collect only ~0.3 of the lift income, so seating the palm is worth acquiring.
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)                      # (N,)
    # the 10/step asks for more than the lift does: palm and opposition enter a SECOND time, so a
    # fingertip touch can never collect the dominant income even under the wrench.
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)

    # =============================================================================================
    # REACH -- 0.60 total, and it FADES OUT once the object is really held.
    # From random weights this is the only income before contact, so it keeps a long-range pull;
    # but its zero-contact ceiling stays below what one real contact pays, the rule that finally
    # produced contact in this track. Both parts are weighted by how horizontal the surface is at
    # the point they measure, so the pull ends at the WALL, not at the rim.
    # =============================================================================================
    fade = (1.0 - held).clamp(min=0.0, max=1.0)
    # far part keeps a floor of 0.35 on the wall weighting: from 0.49 m away the nearest point may
    # well be a rim, and the policy still has to be pulled towards the object at all.
    reach_far = ((1.0 - torch.tanh(palm_d / 0.20))                 # 0.20 m covers the 0.15-0.49 m spawn
                 * (0.4 + 0.6 * face) * (0.35 + 0.65 * palm_side))
    reach_near = palm_near * face * palm_side                      # the last 2 cm: wall only
    r_reach = W_REACH * fade * (0.55 * reach_far + 0.45 * reach_near) * (0.45 + 0.55 * grip)

    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)        # saturates at half flexion: no crushing drive
    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)      # (N,5,3)
    link_side = _sidewall(ctx.link_surface_normal[..., 2])                                  # (N,5,3)
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)                       # (N,5)
    # a straight finger lying along the object gets nothing (flex = 0), a curled finger in the air
    # gets nothing (near = 0), a fingertip hooked over the rim gets a quarter (link_side ~ 0):
    # only curling ONTO a wall with the palm seated pays.
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

    # palm contact is now worth having (0.30 of 1.20, double round 10), but it still counts only
    # together with digits: on its own it funded a flat open hand pressed against the object.
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # A closed grasp parked on the table must become actively unattractive as the episode goes on:
    # the on-table half of the income decays to 45 %. It decays for the reach terms too, so letting
    # go never becomes better than holding on -- only LIFTING restores the full income.
    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_reach = r_reach * table_decay
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
    # time pressure while the object is still on the table. Deliberately NOT gated on `held`, so it
    # cannot make letting go of a grasp look better than keeping it.
    p_stall = P_STALL * on_table * late
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
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 11 was a regression and was stopped after 1.2 hours (left `cp_l_m11` 517 epochs on vision-3090, right
`cp_r_m11` 881 epochs on the local 5090, identical reward file, both from random weights). No Traceback, no PhysX
error, placement rejection 0.42 with no failures.

**Both arms walked away from the object instead of approaching it.** Palm-to-surface distance rose 0.221 to 0.371 m on
the left and sat at 0.357-0.480 m on the right, against 0.10 m by the same epoch in round 10. Contacts are
0.000-0.005 per digit, links in contact 0.003, palm contact 0.000, grasp episodes 0.001 and 0.000. Nothing downstream
happened: lift 0, airborne 0, goal distance unchanged at 0.249 m, tolerance 0.1125, and the wrench never fired.

**The cause is the side-wall weighting applied to the long-range reach term.** The factor (1 - |surface normal z|) was
multiplied into reaching as well as seating. From the start pose the nearest point of an open cup is its rim, whose
normal is nearly vertical, so that factor is close to zero there and approaching the cup paid almost nothing. Meanwhile
the stall penalty is unconditional and grows with episode progress: total reward stayed negative all round (-0.33 to
-0.10 left, -0.34 to -0.36 right) and was dominated by that penalty (-0.28 to -0.32). With no income for approaching
and a standing cost for the object being on the table, retreating was the least bad option, and both arms found it.

**What this round does confirm:** the fault is in the reward, not in the mirror or the machines. Two different GPUs,
two arms and two start seeds produced the same collapse from the same file, and the change that caused it is the only
one that touched the approach term.

To make the code more accurate and train better robot, the feedback for improvement is:
Round 11 changed four things at once and one of them broke the first stage. Keep three, undo one.

1. UNDO: the side-wall factor on the long-range approach. It must pay on plain palm-to-surface distance, or with a
   floor of about 0.3 on the factor, so that closing in on the object is always worth doing. The rim-versus-wall choice
   is not made at 0.3 m away.

2. KEEP but narrow: prefer the side wall only from close in. Within a few centimetres of the surface, weight seating,
   per-link nearness and contact by how horizontal the surface normal is. That is where round 10's rim grip was formed
   (lowest finger link 0.33-0.34 against a wall spanning 0.23-0.33), and where the preference belongs.

3. FIX: the stall penalty must not punish a policy that has not reached the object. Gate it on the hand being at the
   object (in contact, or within a few centimetres) and keep it below the reach income it competes with. In this round
   it was the largest term in the whole reward, for policies that never touched anything.

4. KEEP unchanged: the second half (transport, goal, hold, success) gated on how far the object has risen, so nothing
   pays while it sits on the table; lift raised to 2.5; per-digit scoring by number of links in contact; the doubled
   palm share with the top ladder rungs requiring the palm; first-success-only bonus.

5. Method note for the next rounds: changing the approach term and the stall term together, in the same round that
   also re-scaled lift and the goal gate, is why this took a full round to attribute. The parallel left/right runs did
   their job - the identical collapse on two machines ruled out mirror and hardware in one hour instead of a day.

Generated code shown as below:
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
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 12 ran both arms in parallel from random weights with the identical file (left `cp_l_m12` 2,076 epochs on
vision-3090, right `cp_r_m12` 3,387 epochs on the local 5090, 4.7 hours). It fixed round 11's collapse and then hit a
new wall. No Traceback, no PhysX error, placement rejection 0.42 with no failures.

**The approach came back, exactly as intended.** Palm-to-surface distance fell 0.239 to 0.047 m on the left and 0.212
to 0.062 m on the right, against round 11 where both arms retreated to 0.36-0.37 m. The stall penalty, now gated on
actually holding the object, reads -0.001 instead of -0.30, and total reward turned positive (0.12 left, 0.09 right).
The hand also descended to the side wall: lowest finger link 0.299 left and 0.289 right, inside the 0.232-0.332 band,
against 0.33-0.34 (rim level) in round 10.

**But the grasp never formed.** Only the thumb touches (0.38 left, 0.19 right); index, middle, ring and pinky stay at
0.006 or below. Links in contact are 0.40 of fifteen. **Palm contact is exactly 0.000 on both arms for the whole
round**, the palm-to-cup gap stops at 0.047-0.054 m, and the grasp reward terms (count, per-digit, palm) pay 0.001.
Lift is 0.003, the airborne fraction 0, goal distance unchanged at 0.250 m, tolerance 0.1125, and the wrench never
fired. Both arms flattened out: the last three buckets are identical to within noise.

**The recordings show why.** The hand arrives beside the cup with the fingers already curled and holds that pose; the
thumb tip rests against the wall and the other digits close on air next to it. The user identified this from the video
before the metrics did: approaching with a closed hand makes the wrap impossible.

**The reward has no term that keeps the hand open while approaching.** Closing pays only as flexion x link nearness x
palm seat, so curling early earns nothing - but it costs nothing either, and the policy settles into it. The stage-1
reward that produced every working grasp in this track does have that term: it multiplies the approach income by an
open-pose factor measured on the movable joints, so the hand must stay open to be paid for arriving.

To make the code more accurate and train better robot, the feedback for improvement is:
Round 12 proved two of the three fixes and exposed the missing one.

1. KEEP: the approach shaped on plain palm-to-surface distance, the side-wall preference faded in only within a few
   centimetres with floors, and the stall penalty gated on holding the object. Together these recovered the approach
   (0.24 to 0.047 m) and brought the hand down to the side wall.

2. ADD, and this is the round's finding: **the hand must be paid to stay open while it approaches.** Both arms arrived
   with the fingers curled and could not wrap; palm contact was 0.000 for 2,000-3,400 epochs on each machine. Multiply
   the approach income by an open-pose factor on the movable finger joints (distance of each joint from its default
   open pose, both the mean and the worst joint), the way the stage-1 reward does. Closing should only start paying
   once the links are close to the surface.

3. The palm is the gate to everything downstream and it never arrived. Give palm proximity a steep gradient over the
   0.08 to 0.02 m range - both arms stopped at 0.05 m - and let the lower rungs of the digit ladder pay without the
   palm, so a first partial wrap is worth forming; keep the palm requirement for the top rungs only.

4. Programme note (user decision 09.24): rounds 10-12 showed that a single continuous reward cannot build a wrap from
   random weights in this task. The wrap is now trained first by the verified stage-1 reward, per arm, and this
   continuous reward is applied on top of those weights, as one policy per arm. Stage 1 finished at envelope 0.55
   (left, e2450) and 0.82 (right, e1800); both are archived with their reward file under
   our_source/policy_zoo/cup_pick/<side>/approach. So the next round of this reward starts from a hand that already
   grasps - but items 2 and 3 still apply, because the reward must not undo that grasp.

Generated code shown as below:
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

    It is NEVER applied to the long-range approach. From 0.15-0.49 m away the nearest surface point
    of an open cup is its rim, so multiplying the approach by it (round 11) paid ~0 for closing in
    and both arms retreated to 0.36-0.48 m. It is used only from a few centimetres in, always with a
    floor - that is where the rim-versus-wall choice is actually made."""
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
    #                           = 0.22 long range (PLAIN distance) + 0.24 palm seating + 0.14 closing
    #   keep the grasp   1.20   links-per-digit + count ladder (only the TOP rungs need the palm)
    #                           + palm contact (0.30, worth acquiring) + opposition
    #   lift             2.50   x uprightness x real hold, rising to the height the goal needs
    #   transport        2.50   1.60 on goal_dist over 0-0.30 m + 0.90 on the signed RATE it falls
    #   hold at rest    10.00   real hold, upright, at the goal, still, growing with dwell
    #   first success   60.00   one-off; later successes 2.0, so staying pays, not re-arriving
    #
    # THREE STRUCTURAL RULES, all of them scars from earlier rounds:
    #  (a) Everything from "transport" down is multiplied by `off_table`, a smooth function of the
    #      object's rise above its OWN spawn height: the second half of the task pays exactly ZERO
    #      while the object rests (round 10 collected 0.54-0.69/step of hold income without lifting).
    #  (b) Stage one must ALWAYS be profitable. No factor may zero the long-range approach and no
    #      penalty may exceed the income of the stage the policy is in (round 11 broke both).
    #  (c) The hand must be PAID TO STAY OPEN while it travels: every reach term is multiplied by an
    #      open-pose factor on the movable finger joints, using the mean deviation AND the worst
    #      joint. Round 12 had no such term; both arms arrived curled and palm contact was 0.000 for
    #      2,000-3,400 epochs. Closing only begins to pay once the links are near the surface.
    #
    # Steady states: grasp parked on the table ~1.2 early, ~0.5 late (minus the time pressure);
    # lifted and hovering ~4.5; at rest on the goal ~13; a success spike of 60.
    # =============================================================================================
    W_REACH = 0.22        # long range, PLAIN palm-to-surface distance, x open-pose  } 0.60 total,
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted } x (1 - held)
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat   }
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm }
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike            } 1.20:
    W_PALM = 0.30         # palm contact (2x round 10: a pinch must not score alike)    } build and
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)            } keep a grasp
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height }
    W_RISE = 0.40         # upward speed, only until `lifted` latches                         } 2.50
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m        } 2.50 total,
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls                     } x off_table
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode (~5 steps of a perfect hold)
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 8.00         # lifted and no longer held: on top of losing up to 10 of hold income
    P_SETDOWN = 4.00      # back on the table after a lift: lift income is never re-earnable
    P_STALL = 0.40        # time pressure, gated on actually being ON the object and halved
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

    # WHERE the links touch: a link on a vertical WALL counts fully, one hooked over a rim or lying
    # on a lid counts 0.35. A link in contact is close in by definition, so this is exactly the
    # "only from close in" rule. Used for SCORING only; `digit_touch` / `held` stay on the plain
    # force, so geometry noise can never dissolve a real hold when the wrench arrives.
    link_side = _sidewall(ctx.link_surface_normal[..., 2])                          # (N,5,3)
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side             # (N,5,3)
    link_cw = link_c * link_side_w                                                  # (N,5,3)
    # NUMBER OF LINKS of each digit, made convex so one link is worth far less than two or three:
    # 1 wall link -> 0.19, 2 -> 0.54, 3 -> 1.00; a single fingertip on the rim -> 0.04.
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- the hand's OPEN-POSE factor (rule (c)) --------------------------------------------------
    # hand_q_norm and hand_default_q_norm are oriented the same way for every joint, so the absolute
    # deviation on the movable joints is a direct, contact-free measure of "how curled am I".
    # BOTH statistics are used: the mean, and the worst joint - one finger curled across the palm is
    # enough to block a wrap, and a mean would hide it.
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable               # (N,19)
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))                      # (N,) in (0,1]
    # floored, never zeroed: travelling towards the object must stay profitable whatever the pose.
    open_mul = OPEN_FLOOR + (1.0 - OPEN_FLOOR) * open_pose

    # ---- palm against the object's real surface --------------------------------------------------
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
    # STEEP seating gradient across 0.08 -> 0.02 m: both arms stalled at a 0.05 m palm gap, where
    # this reads 0.22 and rises to 1.0 by 0.02 m. This is where the palm is actually won.
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    # seating means: close to the surface, facing it, and the surface there is a WALL. Real palm
    # contact always counts in full, whatever the geometry read says.
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    # contact quality. Relaxed from round 12's 0.15 + 0.85*seat: a first partial wrap must be worth
    # forming before the palm has arrived, otherwise the grasp terms and the palm terms gate each
    # other and the hand settles where neither pays (round 4's circular equilibrium).
    cq = 0.35 + 0.65 * seat

    # ---- opposition from the surface normals at the touching links ------------------------------
    # opposite outward normals = contacts on opposite sides of the body, whatever its shape.
    normals = ctx.link_surface_normal                                        # (N,5,3,3)
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))           # (N,3)
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))   # (N,3)
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)             # (N,)

    # "held" = palm plus several OPPOSING digits (smooth, full at four). The palm share is large:
    # a three-digit pinch with palm 0.001 collects only ~0.3 of the lift income, so seating the palm
    # is worth acquiring on BOTH arms (round 10 produced a whole hand on one and a pinch on the other
    # from the identical file).
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)                      # (N,)
    # the 10/step asks for more than the lift does: palm and opposition enter a SECOND time, so a
    # fingertip touch can never collect the dominant income even under the wrench.
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)

    # =============================================================================================
    # REACH -- 0.60 total, fading out once the object is really held.
    # The far part is shaped on the PLAIN palm-to-surface distance: no side-wall factor, no
    # late-episode decay, no gate of any kind. All three parts are multiplied by the open-pose
    # factor, so travelling in with a curled hand earns a fraction of travelling in open.
    # =============================================================================================
    fade = (1.0 - held).clamp(min=0.0, max=1.0)
    reach_far = (1.0 - torch.tanh(palm_d / 0.20)) * (0.4 + 0.6 * face)   # 0.20 m covers 0.15-0.49 m
    r_reach = W_REACH * fade * open_mul * reach_far * (0.45 + 0.55 * grip)

    # bringing the palm IN: its own term, with the steep 0.08 -> 0.02 m gradient, the close-in wall
    # preference (floored), and no dependence on the digits, so there is always a gradient that moves
    # the whole hand onto the wall instead of hovering one gap short of it.
    r_palm_in = W_PALM_IN * fade * open_mul * palm_near * (0.3 + 0.7 * face) * palm_side_w

    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)        # saturates at half flexion: no crushing drive
    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)      # (N,5,3)
    # nearness is scored where the link would land: a link a centimetre off a WALL counts fully, one
    # a centimetre off the rim face counts a quarter. This is a close-in term by construction.
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)                       # (N,5)
    # CLOSING ONLY BEGINS TO PAY ONCE THE LINKS ARE NEAR THE SURFACE: a straight finger lying along
    # the object gets nothing (flex = 0), a curled finger in the air gets nothing (near = 0), a
    # fingertip hooked over the rim gets a quarter. Only curling ONTO a wall with the palm seated
    # pays, and this term is NOT multiplied by open_mul -- it is the reward for leaving the open pose
    # at the right moment.
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total. Every digit pays alike; two digits are worth far less than four.
    # =============================================================================================
    # sorted rank weights (sum 1.0): 2 digits 0.18, 3 = 0.42, 4 = 0.80, 5 = 1.00. The digits that
    # arrive LAST carry the most, so a two-digit pinch is worth under a quarter of a real envelope.
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w                                            # (N,5)
    # THE LOWER RUNGS PAY WITHOUT THE PALM (rungs 1-3, 0.42 of the ladder): a first partial wrap must
    # be worth forming. Only the TOP rungs (the 4th and 5th digit, 0.58 of the ladder) need the palm,
    # so a pinch cannot climb to the top however many fingertips it lands.
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))

    # per-digit mean of the LINK COUNT, every digit weighted alike, no palm requirement
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)

    # palm contact is worth having (0.30 of 1.20), but it still counts only together with digits:
    # on its own it funded a flat open hand pressed against the object in round 2.
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # A closed grasp parked on the table must become actively unattractive as the episode goes on:
    # the on-table GRASP income decays to 45 %. The decay never touches the reach terms (rule (b)).
    # Releasing a decayed grasp (>= 0.5) to collect reach income (~0.15 at contact distance) is still
    # a clear loss, so "letting go must never pay" survives the decay.
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
    # is, and it is wide: ~0.56 at one tolerance and ~0.41 at two, so a large fraction is already
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
    # TIME PRESSURE, gated on the hand actually being ON the object: `grip` needs contact, `held`
    # needs palm + opposing digits. A policy that has not reached the object pays NOTHING -- round 11
    # charged it 0.28-0.32/step for existing and taught both arms to retreat. At 0.40 it also stays
    # below the ~1.2 of grasp income that letting go would forfeit, so it makes parking costly
    # without ever making release attractive.
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
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 13 is the first round of the two-phase plan: both arms start from their own stage-1 policy (left `cp_l_a20`
e2450, envelope 0.55; right `cp_r_a20` e1800, envelope 0.82) and train the identical continuous reward on top. Left
`cp_l_m13` reached 963 epochs on vision-3090, right `cp_r_m13` 1,607 on the local 5090, both 2.4 hours, no Traceback,
no PhysX error, placement rejection 0.42.

**The left arm did what the round was for.** Lift episodes rose 0.13 (the starting policy) to 0.46, the airborne
fraction 0.06 to 0.31 and the mean rise 0.010 to 0.047 m. The grasp survived the reward change: after an initial dip to
0.29 it returned to 0.75, with links in contact 2.86 (2.83 at the start), ring contact up 0.20 to 0.52 and palm contact
0.34. Cup tilt is 6-7 degrees. Goal distance is unchanged at 0.250 m, so transport has not started and the wrench never
fired.

**The right arm lost its grasp in the first 200 epochs and never recovered.** Grasp episodes went 0.95 to 0.011 by
epoch 222 and have read 0.000 for the last 1,400 epochs; links in contact 3.65 to 0.05, thumb 0.74 to 0.04, index and
ring exactly 0. Lift is 0.000.

**What the right arm settled into is diagnostic.** Its `hand_curl` is 0.000 - the hand is fully open - while palm
contact is 0.19-0.28 and the palm-to-cup gap is the smallest of either arm at 0.023 m. It parks with the hand flat
against the cup and collects +0.195 per step from the new open-pose factor and the new palm-proximity term without ever
closing a finger. The left arm sits at `hand_curl` 0.31 and keeps the contact income as well.

**The two arms ran the identical file, so this is a property of the reward, not of one arm.** The only difference at
the start was the stage-1 pose: left ended stage 1 at `hand_curl` 0.32, right at 0.20, i.e. the right hand was already
flatter. The new open-pose factor pushes in that direction and, from the flatter start, the right arm slid into the
open-hand basin within 200 epochs. The same reward that teaches the left arm to lift makes the right arm let go.

To make the code more accurate and train better robot, the feedback for improvement is:
What worked and must be kept: applying this reward on top of a stage-1 grasp is the right structure. On the left arm it
lifted the object in 46 % of episodes, up from 13 %, while keeping the wrap it inherited (grasp 0.75, links 2.86, ring
contact 0.52). The gating of the whole second half on how far the object has risen, the lift ceiling of 2.5 and the
palm-free lower rungs all behaved.

The fault is one term, and it is fragile enough to decide the outcome from the initial pose.

1. The open-pose factor must switch off once the hand is at the object. It currently multiplies the reach income at all
   times, so keeping the fingers straight pays even while the hand is touching the cup. The right arm found that
   corner: fully open hand (`hand_curl` 0.000), palm flat on the cup, +0.195 per step, no digit contact, for 1,400
   epochs. Gate it on distance - full weight while the palm is more than about 8 cm from the surface, fading to zero by
   the time contact starts - or multiply it by (1 - held). Being open must be worth something only on the way in.

2. The palm-proximity term (0.24, steep from 0.08 to 0.02 m) is what makes that corner profitable, because it pays on
   proximity alone. Require contact, or at least pair it with a digit-contact factor, so that a flat hand pressed
   against the object without fingers earns much less than a wrap.

3. Both arms must be restarted from their stage-1 weights with the corrected reward (user decision, option 1), so the
   left and right keep using the byte-identical file and the mirror comparison stays clean. The left arm's progress in
   this round (lift 0.46) is discarded deliberately; it is worth one round to keep the pair symmetric.

4. Watch in the next round: `hand_curl` must stay near its stage-1 value (0.2-0.3) rather than collapsing to 0, and
   grasp episodes must not dip below about 0.4 in the first 300 epochs. If either arm shows the flat-hand corner again,
   the open-pose term is still reaching too far.

Generated code shown as below:
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

    It is NEVER applied to the long-range approach. From 0.15-0.49 m away the nearest surface point
    of an open cup is its rim, so multiplying the approach by it (round 11) paid ~0 for closing in
    and both arms retreated to 0.36-0.48 m. It is used only from a few centimetres in, always with a
    floor - that is where the rim-versus-wall choice is actually made."""
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
    #                           = 0.22 long range (PLAIN distance) + 0.24 palm seating + 0.14 closing
    #   keep the grasp   1.20   links-per-digit + count ladder (only the TOP rungs need the palm)
    #                           + palm contact (0.30, worth acquiring) + opposition
    #   lift             2.50   x uprightness x real hold, rising to the height the goal needs
    #   transport        2.50   1.60 on goal_dist over 0-0.30 m + 0.90 on the signed RATE it falls
    #   hold at rest    10.00   real hold, upright, at the goal, still, growing with dwell
    #   first success   60.00   one-off; later successes 2.0, so staying pays, not re-arriving
    #
    # STRUCTURAL RULES carried from earlier rounds:
    #  (a) Everything from "transport" down is multiplied by `off_table`, a smooth function of the
    #      object's rise above its OWN spawn height: the second half of the task pays exactly ZERO
    #      while the object rests (round 10 collected 0.54-0.69/step of hold income without lifting).
    #  (b) Stage one must ALWAYS be profitable. No factor may zero the long-range approach and no
    #      penalty may exceed the income of the stage the policy is in (round 11 broke both).
    #  (c) The hand must be PAID TO STAY OPEN while it travels: every reach term is multiplied by an
    #      open-pose factor on the movable finger joints, using the mean deviation AND the worst
    #      joint. Round 12 had no such term; both arms arrived curled and palm contact was 0.000 for
    #      2,000-3,400 epochs.
    #
    # ROUND 14 FIX (round 13: both arms ran the identical file; the right arm, whose stage-1 hand
    # happened to be flatter, collapsed within 200 epochs into a fully open hand pressed flat against
    # the object - hand_curl 0.000, palm gap 0.023 m, palm contact 0.19-0.28, ZERO digit contact,
    # +0.195/step, unchanged for 1,400 epochs - because the open-pose factor and the palm-seating term
    # both kept paying in full at that pose):
    #  (d) The open-pose factor from rule (c) now ALSO fades with palm-to-surface distance: full
    #      weight beyond ~8 cm, vanishing by ~2 cm where contact starts. It is floored (OPEN_FLOOR),
    #      never zeroed, so long-range reach stays profitable (rule (b)) - staying open only pays on
    #      the way in, exactly as rule (c) intended; the corner was that it kept paying AT the object
    #      too, gated only by a full multi-digit `held`, which a flat pinch-less hand never reaches.
    #  (e) Palm seating (`r_palm_in`) no longer pays on proximity alone. It is paired with a finger
    #      engagement factor built from measured flexion and grip, floored at PALMIN_ENGAGE_FLOOR so
    #      there is still a gradient to bring the palm in before any finger has moved (this is round
    #      4's circular-gate trap, avoided by the floor: palm credit is never fully zero for an
    #      unengaged hand, only reduced). A flat hand parked on the wall now earns a small fraction of
    #      this term instead of all of it.
    # =============================================================================================
    W_REACH = 0.22        # long range, PLAIN palm-to-surface distance, x open-pose  } 0.60 total,
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted } x (1 - held)
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat   }
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm }
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike            } 1.20:
    W_PALM = 0.30         # palm contact (2x round 10: a pinch must not score alike)    } build and
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)            } keep a grasp
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height }
    W_RISE = 0.40         # upward speed, only until `lifted` latches                         } 2.50
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m        } 2.50 total,
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls                     } x off_table
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode (~5 steps of a perfect hold)
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 8.00         # lifted and no longer held: on top of losing up to 10 of hold income
    P_SETDOWN = 4.00      # back on the table after a lift: lift income is never re-earnable
    P_STALL = 0.40        # time pressure, gated on actually being ON the object and halved
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
    OPEN_NEAR = 0.08      # NEW (d): open-pose pays in full at this palm-surface distance or beyond
    OPEN_CONTACT = 0.02   # NEW (d): and its credit is gone by here, where contact starts
    PALMIN_ENGAGE_FLOOR = 0.35   # NEW (e): seating credit an entirely disengaged hand still keeps
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

    # WHERE the links touch: a link on a vertical WALL counts fully, one hooked over a rim or lying
    # on a lid counts 0.35. A link in contact is close in by definition, so this is exactly the
    # "only from close in" rule. Used for SCORING only; `digit_touch` / `held` stay on the plain
    # force, so geometry noise can never dissolve a real hold when the wrench arrives.
    link_side = _sidewall(ctx.link_surface_normal[..., 2])                          # (N,5,3)
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side             # (N,5,3)
    link_cw = link_c * link_side_w                                                  # (N,5,3)
    # NUMBER OF LINKS of each digit, made convex so one link is worth far less than two or three:
    # 1 wall link -> 0.19, 2 -> 0.54, 3 -> 1.00; a single fingertip on the rim -> 0.04.
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- the hand's OPEN-POSE factor (rule (c)) --------------------------------------------------
    # hand_q_norm and hand_default_q_norm are oriented the same way for every joint, so the absolute
    # deviation on the movable joints is a direct, contact-free measure of "how curled am I".
    # BOTH statistics are used: the mean, and the worst joint - one finger curled across the palm is
    # enough to block a wrap, and a mean would hide it. This is a POSE measure only; how far it
    # actually gets to spend is decided below, once the palm's distance to the surface is known.
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable               # (N,19)
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))                      # (N,) in (0,1]

    # ---- measured per-digit flexion (contact-free): used to gate BOTH the palm-seating credit
    # below (FIX (e)) and the close-in credit further down, exactly as rule (c) intends -- an
    # engagement measure that exists before any contact happens, so it cannot recreate round 4's
    # circular "near needs contact, contact needs near" trap.
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)        # saturates at half flexion: no crushing drive

    # ---- palm against the object's real surface --------------------------------------------------
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
    # STEEP seating gradient across 0.08 -> 0.02 m: both arms stalled at a 0.05 m palm gap, where
    # this reads 0.22 and rises to 1.0 by 0.02 m. This is where the palm is actually won.
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    # seating means: close to the surface, facing it, and the surface there is a WALL. Real palm
    # contact always counts in full, whatever the geometry read says.
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    # contact quality. A first partial wrap must be worth forming before the palm has arrived,
    # otherwise the grasp terms and the palm terms gate each other (round 4's equilibrium).
    cq = 0.35 + 0.65 * seat

    # ---- ROUND 14 FIX (d): the open-pose factor now fades with DISTANCE, not only with a full
    # multi-digit `held`. Full weight while the palm is 8 cm or more from the surface, vanishing by
    # 2 cm where contact starts -- "staying open only pays on the way in". Floored by OPEN_FLOOR
    # below, never fully zeroed, so the long-range reach term always stays profitable (rule (b)).
    # This is what closes round 13's corner: at the 0.023 m gap where the right arm parked with a
    # fully open hand, `open_gate` is already close to zero, so open_mul collapses towards the floor
    # instead of paying in full.
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)
    open_mul = OPEN_FLOOR + (1.0 - OPEN_FLOOR) * open_pose * open_gate

    # ---- opposition from the surface normals at the touching links ------------------------------
    # opposite outward normals = contacts on opposite sides of the body, whatever its shape.
    normals = ctx.link_surface_normal                                        # (N,5,3,3)
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))           # (N,3)
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))   # (N,3)
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)             # (N,)

    # "held" = palm plus several OPPOSING digits (smooth, full at four). The palm share is large:
    # a three-digit pinch with palm 0.001 collects only ~0.3 of the lift income, so seating the palm
    # is worth acquiring on BOTH arms (round 10 produced a whole hand on one and a pinch on the other
    # from the identical file).
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)                      # (N,)
    # the 10/step asks for more than the lift does: palm and opposition enter a SECOND time, so a
    # fingertip touch can never collect the dominant income even under the wrench.
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)

    # =============================================================================================
    # REACH -- 0.60 total, fading out once the object is really held.
    # The far part is shaped on the PLAIN palm-to-surface distance: no side-wall factor, no
    # late-episode decay, no gate of any kind. All three parts are multiplied by the (now
    # distance-gated) open-pose factor, so travelling in with a curled hand earns a fraction of
    # travelling in open, and staying open once at the object earns a fraction too.
    # =============================================================================================
    fade = (1.0 - held).clamp(min=0.0, max=1.0)
    reach_far = (1.0 - torch.tanh(palm_d / 0.20)) * (0.4 + 0.6 * face)   # 0.20 m covers 0.15-0.49 m
    r_reach = W_REACH * fade * open_mul * reach_far * (0.45 + 0.55 * grip)

    # ---- ROUND 14 FIX (e): palm seating is now paired with a finger-engagement factor built from
    # measured flexion and grip, floored at PALMIN_ENGAGE_FLOOR. A hand that has not started closing
    # yet still keeps a third of this term (a gradient to bring the palm in survives), but a flat,
    # fully open hand pressed to the wall -- round 13's corner -- now earns a small fraction of it
    # instead of the whole 0.24, on top of the distance-gated open_mul above.
    engage = (0.5 * flex.mean(dim=-1) + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    r_palm_in = (W_PALM_IN * fade * open_mul * palm_near * (0.3 + 0.7 * face) * palm_side_w
                 * seat_engage)

    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)      # (N,5,3)
    # nearness is scored where the link would land: a link a centimetre off a WALL counts fully, one
    # a centimetre off the rim face counts a quarter. This is a close-in term by construction.
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)                       # (N,5)
    # CLOSING ONLY BEGINS TO PAY ONCE THE LINKS ARE NEAR THE SURFACE: a straight finger lying along
    # the object gets nothing (flex = 0), a curled finger in the air gets nothing (near = 0), a
    # fingertip hooked over the rim gets a quarter. Only curling ONTO a wall with the palm seated
    # pays, and this term is NOT multiplied by open_mul -- it is the reward for leaving the open pose
    # at the right moment.
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total. Every digit pays alike; two digits are worth far less than four.
    # =============================================================================================
    # sorted rank weights (sum 1.0): 2 digits 0.18, 3 = 0.42, 4 = 0.80, 5 = 1.00. The digits that
    # arrive LAST carry the most, so a two-digit pinch is worth under a quarter of a real envelope.
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w                                            # (N,5)
    # THE LOWER RUNGS PAY WITHOUT THE PALM (rungs 1-3, 0.42 of the ladder): a first partial wrap must
    # be worth forming. Only the TOP rungs (the 4th and 5th digit, 0.58 of the ladder) need the palm,
    # so a pinch cannot climb to the top however many fingertips it lands.
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))

    # per-digit mean of the LINK COUNT, every digit weighted alike, no palm requirement
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)

    # palm contact is worth having (0.30 of 1.20), but it still counts only together with digits:
    # on its own it funded a flat open hand pressed against the object in round 2.
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # A closed grasp parked on the table must become actively unattractive as the episode goes on:
    # the on-table GRASP income decays to 45 %. The decay never touches the reach terms (rule (b)).
    # Releasing a decayed grasp (>= 0.5) to collect reach income (~0.15 at contact distance) is still
    # a clear loss, so "letting go must never pay" survives the decay.
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
    # is, and it is wide: ~0.56 at one tolerance and ~0.41 at two, so a large fraction is already
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
    # TIME PRESSURE, gated on the hand actually being ON the object: `grip` needs contact, `held`
    # needs palm + opposing digits. A policy that has not reached the object pays NOTHING -- round 11
    # charged it 0.28-0.32/step for existing and taught both arms to retreat. At 0.40 it also stays
    # below the ~1.2 of grasp income that letting go would forfeit, so it makes parking costly
    # without ever making release attractive.
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
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 14 restarted both arms from their stage-1 policies (left `cp_l_a20` e2450, right `cp_r_a20` e1800) with the
open-pose factor limited to the approach and palm seating tied to finger engagement. Left `cp_l_m14` reached 751 epochs
on vision-3090, right `cp_r_m14` 1,317 on the local 5090 (1.9 hours each). No Traceback, no PhysX error, placement
rejection 0.42.

**The round-13 flat-hand corner did not come back.** `hand_curl` stayed at 0.15-0.32 on both arms (0.000 on the right in
round 13), so the open-pose gating did its job.

**The right arm then peaked and collapsed.** By epoch 600-800 it held the grasp at 0.88, lifted in 0.43 of episodes
(airborne 0.28, mean rise 0.064 m), moved the cup towards the goal for the first time (goal distance 0.254 to 0.221 m),
reached far-start success 0.16 with 0.47 successes per episode, and spent up to 5 steps per episode inside the
disturbance radius. From epoch 800 everything fell: grasp 0.61 to 0.27, links 2.39 to 0.62, lift 0.30 to 0.11, success
0.16 to 0.03, and the palm-to-cup gap opened from 0.05 to 0.24-0.29 m - the hand moved away from the cup.

**The reward terms show why.** During the peak the drop penalty was the largest term of the whole reward, -0.75, -0.74,
-0.50 and -0.52 per step across four buckets, larger than lift (+0.33 to +0.41) and hold (+0.32 to +0.38) together
minus nothing else. The penalty is `P_DROP (8.0) x lifted x (1 - held)` with `lifted` the environment's persistent
height latch: once the cup has been lifted, every later step without a full hold is charged 8, until the episode ends.
One slip therefore costs for the rest of the episode. Lifting became a gamble with a large downside, and the policy
found the safe answer - stop lifting, stop grasping, move the hand away - after which the drop penalty fell to -0.10.
The left arm was on the same path when the round was stopped: drop penalty -0.54 to -0.96 per step against lift +0.21
and hold +0.17, total reward -0.77.

**The goal-gated wrench never fired, in this round or any round since iter_09.** The ADR level rose 0.03 to 1.00 (20
N/kg) between epochs 400 and 800 on the right arm, but `dr/wrench_fire_frac` and `dr/wrench_f_norm_max` are exactly
0.000 throughout. The environment gated firing on the parent's `self._lifted`, which this environment family never
updates (the kp-family reward path stores the height latch elsewhere), while the ADR level and the logs used the reward
context's `lifted`. The level climbed on dwell that was measured correctly, but no force was ever applied. This is fixed
in `manip_stages/cup_pick_env.py` (single gate function for firing, ADR and logging) with a contract test that fails if
the two ever diverge again. Everything reported as "under the disturbance" since iter_09 was in fact without one.

To make the code more accurate and train better robot, the feedback for improvement is:
Keep: the open-pose factor limited to the approach and palm seating tied to finger engagement - the flat-hand corner of
round 13 did not return on either arm - and everything else round 14 inherited (object-agnostic surface terms, the
second half gated on how far the object has risen, lift 2.5, link-count scoring, first-success-only bonus).

Fix one term, and re-check the disturbance now that it actually works.

1. No penalty may be proportional to a persistent latch. The drop penalty was 8 x lifted x (1 - held) with `lifted` the
   episode-long height latch, so a single slip charged 8 per step for the rest of the episode. It became the largest
   term in the reward (-0.5 to -0.96 per step on both arms) and made lifting a losing bet: the right arm reached success
   0.16 and then unlearned grasping and moved its hand 0.29 m away from the cup; the left arm was heading the same way.
   Make the cost of losing the object short-lived and bounded: charge it while the object is actually falling or
   resting on the table after having been lifted, based on the current height and velocity, not on the latch; keep its
   size comparable to one step of the lift income, not a multiple of it. The main cost of dropping should be the income
   the policy stops earning. The set-down penalty has the same form (4 x lifted x on_table) and needs the same change.

2. The goal-gated wrench now fires. It was silently disabled from iter_09 through iter_14 by a gating bug in the
   environment (fixed). The next round is therefore the first one in which the policy meets a real disturbance at the
   goal; the ADR level restarts at 0 (5 N/kg). Watch `dr/wrench_fire_frac` and `dr/wrench_f_norm_max` to confirm it
   fires, and watch grasp and drop rate as the level rises.

3. Start points: left from its epoch-450 snapshot (the latest saved, grasp ~0.46, hand_curl 0.25), right from its
   epoch-800 snapshot (the peak before the collapse: grasp 0.88, lift 0.42, success 0.16).

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
