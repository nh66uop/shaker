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

WHAT ROUND 15 GOT RIGHT AND WRONG (measured on both arms):
Round 15 works: both arms lift (0.82-0.96 of episodes), carry to the goal and hold there under the disturbance (far-start success 0.74-0.83 left, 0.68-0.75 right; the wrench fires on 0.47-0.62 of steps and the left held ADR level 1.0 = 20 N/kg for 456 epochs). Nothing collapsed. Keep it whole.
But the grasp both arms converged on is a FOUR-DIGIT grasp, not the five-finger envelope the task asks for. On the right the index finger never touches the object (0.000 over the whole run and on all 8 object species in a deterministic probe, its links 31-44 mm from the surface); at success 3.94 fingers and 5.3 of 15 links touch, palm contact 0.64-0.75 and falling. On the left the index touches in 0.10 of steps, 4.8 links, palm 0.92. Palm contact while carrying falls with object size (0.97 on the smallest to 0.32 on the largest), so the largest object is carried by digits alone.
The reward made this the optimum. All downstream income - lift 2.5, transport 2.5, hold 10 per step - is multiplied by `held`, and `held` saturates at four touching digits. The fifth digit is paid only inside the 1.2 grasp budget (top rung 0.08 + one fifth of the link mean 0.06 = at most 0.14 per step, under 1 % of what four digits unlock), and a deeper wrap likewise only moves the 1.2 budget. Nothing pays for the envelope, so nothing forms it.

FIX IT LIKE THIS:
- Define ONE smooth envelope quality in [0, 1] from quantities that are symmetric across the digits and object-agnostic: the sorted digit-count ladder with the fifth digit as a real step, the mean links per digit (wrap depth), palm contact, thumb/palm opposition. Measure it on the same soft contact forces as `held`.
- NO PER-FINGER TERM. Nothing may name the index or any digit, no finger-specific weight or threshold. The policy must learn that the fifth digit and the deeper wrap are better because the SAME multiplier that pays for lifting, carrying and holding grows with them.
- Multiply the lift, rise, transport, progress, hold-at-goal and success incomes by that envelope quality, floored so a four-digit grasp still earns and still lifts: a four-digit wrap with the palm seated earns about 0.65-0.75 of the downstream income, five digits with two or more links each plus palm and opposition earn 1.0, and a five-fingertip pinch without palm or wrap must not beat a deep four-digit wrap. The missing fifth digit must cost a quarter to a third of everything earned while carrying and holding, on every step.
- Keep `held` (saturating at four digits) for everything that is a GATE rather than an income: the reach fade (1 - held), the drop, stall, drift and descend costs. The disturbance must not be able to turn a real four-digit hold into penalties.
- Rule (b) still binds: with four digits the policy must still clearly prefer lifting and carrying to anything available on the table.
- Everything else from round 15 stays: the falling-only drop and set-down costs, no penalty on a persistent latch, the second half gated on the object's rise, the open-pose factor limited to the approach, palm seating tied to engagement, uprightness as a multiplier, first-success-only bonus, the goal-gated wrench.
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



(Rounds 2-17 are omitted here for length; the round-18 code below already contains everything they established.)

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
    #  (g) ROUND 18: CURL IS A FACTOR OF env_q, NOT A SUMMAND. Round 17 weighted the depth summand by
    #      each digit's inner-link force share; the policy moved the index grip onto its fingertip
    #      (env_q -0.006) and bought it back with palm seating (+0.019), and even a full curl of every
    #      digit was worth only 8-9 % of the downstream income. Now: per digit, curl = share of its
    #      OWN cup force on its two inner links (k=0, k=1) against its fingertip (k=2), full at one
    #      half, x a smooth trust that its INNER links press; a digit that does not touch gives 0 (no
    #      fallback). hand_curl = mean over the five digits; env_q = env_base x (CURL_FLOOR +
    #      (1 - CURL_FLOOR) x hand_curl), so the ladder, palm and opposition can no longer pay back a
    #      fingertip press. The depth summand is round 16's link count again (the force share enters
    #      once, as this factor).
    #  (h) ROUND 18: THE TABLE CHARGE FIRES WHERE A LINK ACTUALLY TOUCHES THE TABLE. hand_z_min is a
    #      point inside the links, so a link resting on the table reads ~1-2 cm; the round-17 band
    #      (2 mm) never fired (pen_table 0.0000) while the hand came down onto the table. The floor
    #      half now starts 2 cm above the top and is full by 1 cm. Same 0.6 cap, same palm half,
    #      current height only (no latch, bounded).
    #
    # Steady states (estimated from the round-17 replay): grasp parked on the table ~1.2 early, ~0.2
    # late; round-17 grasp (hand_curl ~0.27) at rest on the goal ~0.56 of full downstream income
    # (round 17 paid it ~0.73); every further digit that curls +~0.065 of full; five digits curled
    # with the palm -> ~0.82 of full, i.e. all of what env_base allows; a first-success spike <= 60.
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
    P_SQUEEZE = 0.50      # force beyond what a hold needs (kept as is, by instruction)
    P_TABLE = 0.60        # palm or finger links driven onto the table top (cap unchanged)
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
    ENV_W_COUNT = 0.30    # sorted engagement ladder (the fifth digit is its biggest rung)  } sum 1.0
    ENV_W_DEPTH = 0.35    # links in contact per digit (round-16 link count again)         }
    ENV_W_PALM = 0.25     # palm contact                                                    }
    ENV_W_OPP = 0.10      # thumb/palm opposition, same saturation as `held`                }
    ENV_DEPTH_LINKS = 2.0  # "two or more links each" is full, a fingertip alone half
    ENV_NEAR_SHARE = 0.25  # a digit hovering at the surface is worth at most 1/4 of a touching one
    ENV_NEAR_SCALE = 0.01  # m: pre-contact on-ramp, about the last 1-2 cm before a link touches
    # ---- ROUND 18 constants: curl as a factor (rule (g)) and the table band (rule (h)) ---------
    CURL_FULL_SHARE = 0.50   # inner links carrying half or more of the digit's OWN cup force = curled
    CURL_FORCE_REF = 0.30    # N: trust that the INNER links press, tanh(inner force / 0.3): 0.76 at
    #                          0.3 N. On the inner force, not the digit total, so adding fingertip
    #                          force can never raise a digit's curl (c1r repair of c1)
    CURL_FLOOR = 0.50        # env_q keeps half at zero curl. Sized on the round-17 replay: its grasp
    #                          (hand_curl ~0.27) keeps ~3/4 of what it earns now, all five digits
    #                          curled earn all of env_base, each further curled digit ~+6.5 % of the
    #                          full downstream income (more than the 5-8 % the palm buy-back paid)
    FLOOR_BAND_TOP = 0.020   # m above the table top: the floor charge starts here
    FLOOR_BAND_FULL = 0.010  # m: and is full here (a link resting on the table reads ~1-2 cm)

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
    # wrap depth, round 16 again: links in contact per digit, full at two links, mean over five
    count_depth = (link_c.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0)                   # (N,5)
    env_depth = count_depth.mean(dim=-1)                                                   # (N,)
    env_base = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
                + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)       # (N,)

    # ---- ROUND 18 (rule (g)): CURL, a FACTOR of the envelope quality ------------------------------
    # share of each digit's OWN cup force carried by its two inner links (k=0 moved by joint _3,
    # k=1 moved by joint _4) against its fingertip (k=2). Relative, never an absolute force target;
    # pressing harder with the tip lowers it. Same formula for all five digits, no joint angle.
    link_f = ctx.link_cup_force.clamp(min=0.0)                                             # (N,5,3)
    digit_f = link_f.sum(dim=-1)                                                           # (N,5)
    inner_share = ((link_f[..., 0] + link_f[..., 1])
                   / digit_f.clamp(min=1e-6)).clamp(min=0.0, max=1.0)                      # (N,5)
    # smooth trust that the INNER links press: a digit off the object gives exactly 0, never a
    # fallback value, so lifting a finger off the object can never raise the hand's curl; read on
    # the inner force, so extra fingertip force lowers the share and never raises the trust
    inner_f = link_f[..., 0] + link_f[..., 1]                                              # (N,5)
    press = torch.tanh(inner_f / CURL_FORCE_REF)                                           # (N,5)
    digit_curl = (inner_share / CURL_FULL_SHARE).clamp(max=1.0) * press                    # (N,5)
    hand_curl = digit_curl.mean(dim=-1)                                                    # (N,)
    curl_factor = CURL_FLOOR + (1.0 - CURL_FLOOR) * hand_curl                              # [0.5, 1]
    # the ladder, palm and opposition sit INSIDE the factor, so none of them can buy back a digit
    # that moves its grip onto its fingertip
    env_q = env_base * curl_factor                                                         # (N,)
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
    # squeeze unchanged: with rule (g) a harder tip press lowers the curl factor, so this charge no
    # longer competes with the envelope income
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    # palm half unchanged: the palm's own lowest point (a real surface measurement) within 5 mm
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    # ROUND 18 (rule (h)): floor half on a band where a finger link resting on the table actually
    # reads (its point sits inside the link): 0 at 2 cm above the top, full at 1 cm, smoothstep.
    # Current height only, bounded; a hand wrapping the side wall (lowest link ~8-9 cm) pays 0.
    hand_rel = ctx.hand_z_min - ctx.table_z
    floor_t = ((FLOOR_BAND_TOP - hand_rel) / (FLOOR_BAND_TOP - FLOOR_BAND_FULL)).clamp(min=0.0, max=1.0)
    floor_pen = floor_t * floor_t * (3.0 - 2.0 * floor_t)
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
        "pen_table": -p_table,        # round 18: floor band 2 cm -> 1 cm above the table top
        "pen_arm_vel": -p_arm_vel,
        "pen_arm_fast": -p_arm_fast,
        "pen_cup_vel": -p_cup_vel,
        "pen_action": -p_action,
    }
    return reward, components
```

Feed this reward code into the environment, and use the RL algorithm to train the policy. After training, I can see from the robot that:
Round 18 ran left `cp_l_m18` from the round-16 epoch-1000 snapshot with the round-18 reward (curl as a factor of the
envelope quality, table charge band moved to 2 -> 1 cm) for about 1,470 epochs (cumulative about 2,470) on vision-3090.
No Traceback and no PhysX error. It was stopped as a round end after epoch 1250 fell back below epoch 800 in a
fixed-tolerance evaluation; epoch 800 is kept as the best snapshot.

**Fixed-tolerance deterministic evaluation (tolerance 0.0911 m, wrench pinned at 20 N/kg, first episode of 256
environments, seeds 42 and 43: 512 episodes per checkpoint).** Against the epoch-1000 start (success 0.785):
- epoch 450: success 0.811, epoch 800: 0.842 (z +2.3, the best so far), epoch 1250: 0.785 (-0.057 against epoch 800,
  z -2.3). No round-17-style collapse: at epoch 800 round 17 was at 0.703.
- Goal chaining rose and stayed up: three-goal episodes 0.43 -> 0.59 -> 0.66 -> 0.67, successes per episode 2.41 -> 3.36,
  all five goals 0.26 -> 0.56.
- The index curls again: share of its cup force on its two inner links at the success step 0.315 (start), 0.253, 0.338,
  0.307 (round 17: 0.010). Over all five digits 0.188 -> 0.212 at epoch 1250 (z +2.6); ring 0.13 -> 0.19, little finger
  0.01 -> 0.04. Whether this is the curled grip the operator asked for is waiting for the operator's video verdict.
- Early terminations 0.248 -> 0.318, 0.322, 0.336, mostly after the lift. Contact force fell at every checkpoint (mean
  finger-link force per episode 3.52 -> 2.49 -> 2.13 N) while more digits and links touched. Drops stayed rare (0.002)
  and the tilt while held fell (worst 20 -> 16 deg).

**Why more episodes end early: a termination-cause replay** (seed 42, 256 episodes per checkpoint, the environment's
own termination terms recomputed per environment; phase read one step before the terminal step; rates with
per-environment standard errors):
- The whole excess is the object leaving the working area in y after the lift: 0.051 (start) -> 0.105 (epoch 800,
  z 2.3) -> 0.160 (epoch 1250, z 4.1). Arm runaway after the lift 0.078 -> 0.105 -> 0.055 (z 1.1, -1.1), tilt above 60
  deg after the lift 0.023 -> 0.008, object dropped below the table 0, hand into the table 0, arm joint limits 0.
  Terminations before the lift are back at the start's level (0.109 -> 0.125, 0.137; round 17: 0.219).
- Per carried goal the risk did not change. An episode starts one carrying leg per goal it goes for; round 18 chains
  more goals, so it started 930 and 907 legs against 765. Terminations after the lift per leg: 0.054 -> 0.061 -> 0.063
  (z 0.7, 0.8). Only the y exits per leg rose (0.017 -> 0.029 -> 0.045). A second replay of the start snapshot
  reproduced it (0.055 per leg, y exits 0.013).
- The y exits come from the environment's bound. The left arm still uses the right arm's bound (object y <= 0.25 m)
  while the goal box reaches y 0.35 and every new goal is the previous goal moved by up to 0.08 m per axis; a policy
  that chains more goals meets more goals beyond the line. Of the y exits after the lift, the current goal was beyond
  the line in 12 of 27 (epoch 800) and 8 of 41 (epoch 1250), and the object was already within the success tolerance
  of a goal inside the line in 0 and 8. In training the same exits rose from about 0.08-0.11 to 0.17-0.18 of episode
  endings in round 17 and round 18 alike, both of which chained more goals.
- The remaining y exits (15 of 27, 25 of 41) carried the held object past the line while the goal was inside it, most
  of them moving away from the goal at 0.6-0.8 m/s. Fast sideways moves of the held object happen in both directions
  (above 0.5 m/s on 302 and 174 trace steps towards and away from the line at epoch 800, 344 and 287 at epoch 1250,
  239 and 159 at the start); only one side has a bound within reach. Moving away from the goal is already charged by
  the signed progress term and the drift charge; no reward term was found that pays for it.
- It is not a weak hold. In the five steps before a y exit or a runaway after the lift all five digits touched with
  60-95 N (median) of grip; lifted episodes that terminated held with the same force as those that did not (60 vs 55 N
  at epoch 800, 51 vs 50 N at epoch 1250); the wrench fired on 2-3 % of the last ten steps. Runaways jumped from below
  10 rad/s to above 20 rad/s in one step in 26 of 34 and 17 of 22 cases, as at the start (18 of 27).

**The grip got lighter, partly because of the curl factor** (replay, every held step: object lifted and 3 cm up):
- Total finger force on the object 71 N (start) -> 65 -> 59 N. The ring finger moved force off its fingertip (9.2 ->
  5.2 N) and its inner-link share rose 0.02 -> 0.19; the curl factor pays for exactly that (its curl 0.03 -> 0.20).
- The thumb's inner links pressed less (24 -> 18 N), but the thumb's curl was saturated all along (share above one
  half, press trust 0.99), so the curl factor did not pay for that; it matches the squeeze charge above 40 N of total
  force (0.29 -> 0.25 per held step) and the lighter fingers it presses against.
- Hand curl 0.25 -> 0.28, envelope multiplier 0.55 -> 0.57, hold income per held step 0.98 -> 1.14. Palm contact 0.85
  -> 0.91.

**The round-18 table band worked.** The objects knocked over before a grasp with the lowest hand link at 0.217-0.220 m
(13 of 21 in round 17) are gone at epoch 800 (0 of 15) and rare at epoch 1250 (2 of 14).

The operator's direction stands: the fingers should curl around the object rather than press with their tips; the
hold was judged stable.

To make the code more accurate and train better robot, the feedback for improvement is:
Keep the round-18 code whole: no reward change this round. Its two changes did what they were meant to. The index
curls again (0.31-0.34 of its force on the inner links at success, round 17 0.01), the ring and little finger started to
use their inner links, success reached 0.842 at epoch 800 with three-goal episodes up by a half, and the terminations
before a grasp are back to the start's level. Every structural rule stays (second half gated on the object's rise,
envelope quality as ONE multiplier on the downstream income with curl as a factor inside it, no per-finger term,
falling-only drop and set-down costs, open-pose factor only on the approach, palm seating tied to engagement,
uprightness as a multiplier, first-success-only bonus, goal-gated wrench, the table band). Every budget stays,
CURL_FLOOR stays, and no speed or control charge is relaxed.

1. The extra early terminations are not a reward matter. They are the object leaving the working area in y after the
   lift (0.05 -> 0.11 -> 0.16 of episodes); every other cause after the lift is flat or lower, and per carried goal the
   risk of ending the episode did not change (0.054 -> 0.061 -> 0.063 per leg). The left arm's bound is the right
   arm's (object y <= 0.25 m) while the goal box reaches 0.35 m and each new goal is a random step from the last one,
   so a policy that chains more goals meets more goals on or beyond the line, and at the evaluation tolerance the
   success region of any goal above 0.16 m crosses it. That bound is an environment setting and is left to the
   operator. The reward must not name a world axis or a bound, so there is nothing in it to change for this.

2. Do not answer the lighter grip. Finger force on the held object fell 71 -> 59 N: the ring finger moved force off its
   tip, which the curl factor pays for, and the thumb pressed less against the lighter fingers under the unchanged
   squeeze charge. Nothing followed from it: no drops, fewer tilts, the episodes that ended after the lift held as
   firmly as the ones that did not, and the hold income rose. A force floor would be an absolute force target; keep the
   curl factor, its trust on the inner-link force and the squeeze charge as they are.

3. Do not slow the carry to keep the object away from the line. The exits with the goal inside the line are fast
   sideways moves of a firmly held object (0.6-0.8 m/s, most of them away from the goal); such moves happen in every
   direction and only one side has a bound within reach. Moving away from the goal is already charged by the signed
   progress term and the drift charge. Raising the speed charges would tax every transport for an exit that exists only
   because of where the bound sits, and could make standing still competitive again.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.607, 0.637, 0.68, 0.693, 0.681, 0.735, 0.738, 0.731, 0.72]  min 0 · mean 0.689 · max 0.777
contact/finger_index_at_success: [-1, 0.844, 0.876, 0.945, 0.959, 0.968, 0.993, 0.977, 0.978, 0.994]  min -1 · mean 0.933 · max 1
contact/finger_middle: [0, 0.721, 0.745, 0.765, 0.749, 0.74, 0.75, 0.761, 0.754, 0.745]  min 0 · mean 0.747 · max 0.828
contact/finger_middle_at_success: [-1, 0.967, 0.961, 0.97, 0.963, 0.957, 0.962, 0.967, 0.956, 0.991]  min -1 · mean 0.96 · max 1
contact/finger_pinky: [0, 0.71, 0.743, 0.764, 0.751, 0.755, 0.765, 0.778, 0.761, 0.734]  min 0 · mean 0.749 · max 0.837
contact/finger_pinky_at_success: [-1, 0.965, 0.962, 0.982, 0.967, 0.986, 0.986, 0.992, 0.986, 0.983]  min -1 · mean 0.972 · max 1
contact/finger_ring: [0, 0.703, 0.738, 0.76, 0.75, 0.751, 0.751, 0.766, 0.762, 0.741]  min 0 · mean 0.745 · max 0.855
contact/finger_ring_at_success: [-1, 0.979, 0.973, 0.972, 0.984, 0.99, 0.98, 0.994, 0.996, 0.997]  min -1 · mean 0.97 · max 1
contact/finger_thumb: [0, 0.743, 0.777, 0.795, 0.78, 0.777, 0.784, 0.796, 0.782, 0.761]  min 0 · mean 0.775 · max 0.885
contact/finger_thumb_at_success: [-1, 1, 1, 1, 0.998, 1, 1, 1, 1, 1]  min -1 · mean 0.99 · max 1
contact/fingers_touching: [0, 3.48, 3.64, 3.77, 3.72, 3.7, 3.79, 3.84, 3.79, 3.7]  min 0 · mean 3.7 · max 4.12
contact/fingers_touching_at_success: [-1, 4.75, 4.77, 4.87, 4.87, 4.9, 4.92, 4.93, 4.92, 4.97]  min -1 · mean 4.84 · max 5
contact/link_force_mean: [0, 3.66, 3.88, 4.13, 3.77, 3.74, 3.53, 3.85, 4.09, 3.56]  min 0 · mean 3.94 · max 129
contact/links_touching: [0, 4.48, 4.61, 4.88, 4.85, 4.8, 5, 5.06, 4.9, 4.79]  min 0 · mean 4.81 · max 5.2
contact/links_touching_at_success: [-1, 6.05, 5.99, 6.18, 6.18, 6.16, 6.4, 6.32, 6.33, 6.36]  min -1 · mean 6.18 · max 6.53
contact/palm_touching: [0, 0.63, 0.665, 0.668, 0.679, 0.685, 0.673, 0.679, 0.63, 0.627]  min 0 · mean 0.654 · max 0.761
contact/palm_touching_at_success: [-1, 0.872, 0.91, 0.892, 0.906, 0.901, 0.898, 0.954, 0.856, 0.882]  min -1 · mean 0.885 · max 1
ctrl/prev_ep_successes_mean: [0, 2.8, 2.36, 2.23, 2.38, 2.01, 2.4, 1.88, 2.24, 2.36]  min 0 · mean 2.23 · max 2.92
done/abnormal: [0, 0, 0.00122, 0.00146, 0.00171, 0.000732, 0.000244, 0.000732, 0.000244, 0.000244]  min 0 · mean 0.000844 · max 0.00317
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.68e-06 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.01e-06 · max 0.000244
done/max_goals: [0, 0, 0.000244, 0.000244, 0, 0.000244, 0.000732, 0.000244, 0.000732, 0.000244]  min 0 · mean 0.000336 · max 0.00195
done/out_xy: [0, 0.000488, 0, 0, 0.000244, 0.000244, 0, 0.000488, 0, 0.000488]  min 0 · mean 0.000271 · max 0.00171
done/tipped: [0, 0, 0, 0, 0.000488, 0, 0.000244, 0, 0, 0.000732]  min 0 · mean 0.000206 · max 0.00122
episode_lengths/step: [2.09, 476, 404, 430, 455, 531, 593, 557, 517, 513]  min 2.09 · mean 485 · max 631
reward/close: [3.08e-06, 0.00659, 0.0072, 0.00789, 0.00667, 0.00606, 0.00697, 0.00683, 0.00829, 0.0076]  min 3.08e-06 · mean 0.00712 · max 0.0101
reward/goal_hold: [0, 0.641, 0.65, 0.681, 0.696, 0.718, 0.717, 0.699, 0.682, 0.706]  min 0 · mean 0.677 · max 0.77
reward/goal_progress: [0, 0.0339, 0.0388, 0.0344, 0.024, 0.0328, 0.0422, 0.0376, 0.0398, 0.0409]  min 0 · mean 0.0349 · max 0.0845
reward/goal_reach: [0, 0.171, 0.189, 0.22, 0.24, 0.247, 0.247, 0.256, 0.268, 0.273]  min 0 · mean 0.231 · max 0.301
reward/grasp_count: [0, 0.0525, 0.0535, 0.0564, 0.057, 0.0562, 0.0586, 0.0585, 0.0552, 0.0544]  min 0 · mean 0.0556 · max 0.0602
reward/grasp_digit: [0, 0.0527, 0.0536, 0.0583, 0.0585, 0.0577, 0.0603, 0.061, 0.0576, 0.0563]  min 0 · mean 0.057 · max 0.062
reward/grasp_oppose: [0, 0.102, 0.105, 0.107, 0.106, 0.106, 0.103, 0.105, 0.106, 0.103]  min 0 · mean 0.104 · max 0.119
reward/grasp_palm: [0, 0.189, 0.199, 0.2, 0.203, 0.205, 0.202, 0.203, 0.189, 0.188]  min 0 · mean 0.196 · max 0.228
reward/lift_height: [0, 0.586, 0.625, 0.664, 0.689, 0.713, 0.713, 0.713, 0.692, 0.688]  min 0 · mean 0.667 · max 0.754
reward/lift_rise: [0, 0.0048, 0.00515, 0.00424, 0.00458, 0.00453, 0.00451, 0.00208, 0.0037, 0.00419]  min 0 · mean 0.00408 · max 0.0237
reward/palm_in: [0.000301, 0.00203, 0.00208, 0.00225, 0.00193, 0.0018, 0.002, 0.00196, 0.00246, 0.0023]  min 0.000301 · mean 0.00211 · max 0.00284
reward/pen_action: [-0.023, -0.0235, -0.0238, -0.0235, -0.0234, -0.0234, -0.0231, -0.023, -0.0229, -0.0226]  min -0.0247 · mean -0.0233 · max -0.0214
reward/pen_arm_fast: [0, -0.00468, -0.00646, -0.00516, -0.00462, -0.00507, -0.00386, -0.00643, -0.0057, -0.00308]  min -0.0113 · mean -0.00518 · max 0
reward/pen_arm_vel: [-0.0263, -0.0561, -0.0583, -0.059, -0.0586, -0.0576, -0.0588, -0.058, -0.0567, -0.0532]  min -0.0685 · mean -0.0573 · max -0.0253
reward/pen_cup_vel: [0, -0.0348, -0.0365, -0.0395, -0.0395, -0.0389, -0.0394, -0.0384, -0.0376, -0.0336]  min -0.0518 · mean -0.0374 · max 0
reward/pen_descend: [0, -0.00829, -0.007, -0.0101, -0.00817, -0.00906, -0.00848, -0.0102, -0.00796, -0.00808]  min -0.02 · mean -0.00876 · max 0
reward/pen_drift: [0, -0.0608, -0.0675, -0.0846, -0.0983, -0.0884, -0.0899, -0.0902, -0.0947, -0.0892]  min -0.117 · mean -0.0852 · max 0
reward/pen_drop: [-0.000609, -0.0213, -0.0178, -0.0177, -0.0165, -0.0156, -0.0151, -0.0156, -0.0171, -0.0139]  min -0.0349 · mean -0.0178 · max -6.47e-05
reward/pen_setdown: [-0.000145, -0.00284, -0.00322, -0.00229, -0.00255, -0.00263, -0.00245, -0.00255, -0.00267, -0.00272]  min -0.00778 · mean -0.00278 · max -1.57e-05
reward/pen_shove: [-0.00773, -0.0153, -0.0127, -0.0113, -0.0138, -0.0118, -0.0128, -0.0117, -0.011, -0.0167]  min -0.0452 · mean -0.0131 · max -0.00678
reward/pen_squeeze: [0, -0.202, -0.225, -0.222, -0.218, -0.215, -0.217, -0.217, -0.23, -0.212]  min -0.251 · mean -0.216 · max 0
reward/pen_stall: [0, -0.000156, -0.000241, -0.000116, -0.000145, -0.000118, -0.000125, -0.000102, -0.000104, -0.000193]  min -0.00156 · mean -0.000145 · max 0
reward/pen_table: [0, -0.000203, -0.000261, 0, -5.54e-05, 0, -0.000263, -0.00106, -2.92e-05, -0.000469]  min -0.00361 · mean -0.000474 · max 0
reward/pen_tilt: [-0.000538, -0.042, -0.0386, -0.0343, -0.0356, -0.0331, -0.0348, -0.035, -0.0326, -0.0316]  min -0.0657 · mean -0.0364 · max -0.00013
reward/reach: [0.0168, 0.00685, 0.00634, 0.00612, 0.00598, 0.00598, 0.00582, 0.00558, 0.00614, 0.00584]  min 0.00436 · mean 0.00618 · max 0.0218
reward/success_bonus: [0, 0.0588, 0.0436, 0.0382, 0.0376, 0.0739, 0.0282, 0.0332, 0.0331, 0.0318]  min 0 · mean 0.0495 · max 0.265
reward/total: [-0.0413, 1.43, 1.48, 1.57, 1.61, 1.73, 1.68, 1.67, 1.62, 1.67]  min -0.0413 · mean 1.59 · max 1.85
rewards/step: [-1.11, 677, 497, 642, 718, 904, 1.01e+03, 865, 899, 892]  min -4.69 · mean 771 · max 1.05e+03
task/lifted_frac: [0, 0.694, 0.727, 0.752, 0.737, 0.734, 0.743, 0.76, 0.736, 0.71]  min 0 · mean 0.729 · max 0.822
task/successes_mean: [0, 1.48, 1.38, 1.3, 1.35, 1.22, 1.36, 1.23, 1.23, 1.15]  min 0 · mean 1.31 · max 1.81
task/tilt_deg: [0.082, 5.34, 4.94, 4.35, 4.51, 4.17, 4.38, 4.45, 4.11, 3.97]  min 0.0159 · mean 4.62 · max 8.5
task/tol: [0.1, 0.1, 0.09, 0.081, 0.0729, 0.0729, 0.0729, 0.0656, 0.059, 0.059]  min 0.0531 · mean 0.075 · max 0.1

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
