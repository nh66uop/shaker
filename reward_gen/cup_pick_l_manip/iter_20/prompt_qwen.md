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



(Rounds 2-18 are omitted here for length; the round-19 code below already contains everything they established.)

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
Round 19 runs left `cp_l_m19` from the round-18 epoch-800 snapshot with the round-18 reward unchanged and one
environment change made by the operator: the left arm's object y bound is now the mirror of the right arm's. At the time
of writing it had run about 430 epochs on vision-3090, no Traceback and no PhysX error, and it keeps running: this
round is about the grip, not a round end. So far the bound change did what it was for. Object-out-of-area episode
endings fell from about 0.16-0.18 of episodes in round 18 to about 0.01-0.04 (training, done/out_xy x episode length),
successes per finished episode are 3.2 (round 18: 2.1-2.4), the tolerance went 0.10 -> 0.081, the wrench stays at level
1.0, five digits and 6.3 links touch at success with the palm on 0.90 of successes.

**The operator's verdict on the round-18 grip: still a fingertip press.** The operator compared the round-18 epoch-800
video with the round-16 epoch-1000 video on the largest cup, side by side while the cup is carried: in both the index,
middle, ring and little fingers press the object with their fingertips; neither curls or rolls the fingers around it.
The rise of the index's inner-link force share (0.338 at success, the round-16 level) does not count as curl. The curl
signal has to be redesigned.

**What the hand actually does** (geometry replay: seed 42, first episode of 256 environments, tolerance 0.0911 m,
wrench pinned at 20 N/kg, both snapshots; plus a replay of the exact 12-environment videos the operator judged). Each
digit was measured at five points - the start of the first phalanx next to the palm, the three link_pos points, and the
end of the fingertip pad - against the object's own outer wall, and its angle around the object axis from the palm:
- Every digit keeps its first joint next to the palm almost straight (0.00-0.17 of its range: thumb, index, middle, ring,
  little finger alike) and folds a later joint to the limit (last joint 0.85-1.00; the index folds its middle joint
  instead part of the time). The middle joint points of the fingers stand 1.8-3.9 cm off the wall while the fingertip is
  on it (0.6-1.1 cm). 93-98 % of the fingertip force is along the wall normal and the last link points into the wall.
  That is the press the operator sees: straight bars across the object with the tips driven into it. 66 % of all held
  digit-steps are this hook.
- Reaching around is not what is missing. The fingertip pads sit 115-148 degrees around the object from the palm and are
  past the object's widest point on 86-100 % of held steps, in both snapshots.
- The only digit that really rolls - every joint bent part of the way, every point within about 1.5 cm of the wall - is
  the ring finger at epoch 800 (32 % of its held steps, up from 0). The index alternates between the fingertip hook
  (40-63 %) and a second hook: first joint straight, middle joint folded to its limit, last joint straight, so the last
  link lies flat on the wall (25-33 %).
- The frames the operator judged (largest cup, 5 s): every finger touches only with its tip (0.1-0.9 cm from the wall)
  while at least one of its middle joint points is 2.5-3.8 cm off. Epoch 800 seats the palm (14 N); epoch 1000 does not
  (2.6 cm gap).

**Why the round-18 curl factor paid for a press.** It measured, per digit, the share of the digit's force on links
k=0 and k=1 against the fingertip link k=2. On this hand k=1 is the last link itself and the fingertip body is fixed
rigidly onto that same link, with its pad overlapping the last link's own collision surface. So "inner against tip"
splits one rigid segment, and a last link laid flat on the wall loads k=1: share 1, full curl.
- The index's flat-last-link hook scores 0.91 and its fingertip hook 0.00. The index's "recovery" in round 18 is a switch
  between these two hooks (mostly on the mid-sized cups and the closed shaker), not a curl.
- The thumb's fingertip hook scores 0.88: its last link is 5 cm long and carries the press. The thumb alone supplies
  about 0.17 of the 0.25-0.27 hand curl, whatever pose it takes.
- A real roll scores 0.16: the rolled ring finger still presses mostly with its tip.
- The share saturates at one half, so a little force on k=1 is enough for full credit.
- On the largest cup, the one the operator watched, the factor rated epoch 1000 above epoch 800 (hand curl 0.28 against
  0.20, 32 episodes each; the index 0.46 against 0.02).
- The first phalanx next to the palm, whose bending is the first step of any curl, has no contact sensor and no
  link_pos entry. The force share cannot see it.

**What separates a press from a curl, measured.** Whether the digit follows the wall: the spread of its three
link_surface_dist values (farthest minus closest link point of the same digit). A rolled digit 1.1 cm (thumb 0.6 cm), the
flat-last-link hook 1.9 cm, the fingertip hook 2.0-2.5 cm; the true radial distances give the same order. It reads only
context fields every digit shares. Three other candidates do not separate: how far around the object the fingertip
reaches (past the widest point in both snapshots), the number of links in contact (the rolled ring finger presses mostly
with its tip), and normal against tangential fingertip force (almost all normal in every pose).

The operator's direction stands and is now specific: the fingers should bend at every joint so that the whole digit
lies along the object, instead of straight bars with hooked tips. The hold itself was judged stable.

To make the code more accurate and train better robot, the feedback for improvement is:
One change: replace the per-digit measure inside the round-18 curl factor. Keep everything else of the round-18 code
exactly as it is. The factor stays where it is (env_q = env_base x curl_factor, curl_factor = CURL_FLOOR + (1 -
CURL_FLOOR) x hand_curl, hand_curl = mean over the five digits, CURL_FLOOR 0.50); only what a digit's curl measures
changes. Round 18's measure (share of the digit's force on links k=0, k=1 against k=2) paid for two fingertip presses -
the index's last link laid flat on the wall (0.91) and the thumb's hooked tip (0.88) - and almost nothing for the one
digit that really rolled (0.16), because k=1 is the last link itself and the fingertip body is fixed onto it.

1. Curl = the digit FOLLOWS THE WALL. Per digit, from ctx.link_surface_dist of its three link points:
       lift_off = link_surface_dist.amax(dim=-1) - link_surface_dist.amin(dim=-1)                 # (N,5)
       follow   = exp(-(lift_off - FOLLOW_DEAD).clamp(min=0) / FOLLOW_SCALE)                      # (N,5) in (0, 1]
       digit_curl = digit_touch * follow                                                          # (N,5)
   with FOLLOW_DEAD = 0.010 m and FOLLOW_SCALE = 0.006 m, and digit_touch the soft contact the code already computes
   (a digit that does not touch gives exactly 0, no fallback). lift_off is how far the digit's farthest link point
   stands off the wall compared with its own closest one. It is relative (the same offset added to all three points
   changes nothing), reads no force and no joint angle, and treats every digit alike. Measured on the replayed held
   steps: digits bent at every joint along the wall 0.6-1.1 cm (curl 0.7-0.9), the last link laid flat with the other
   joints straight 1.9 cm (0.2), the hooked fingertip 2.0-2.5 cm (0.1-0.2). FOLLOW_DEAD is the resolution of the
   surface distance (it reads up to about 1 cm high at contact, and a joint point sits deeper inside its link than the
   fingertip point), not a pose target. Pressing harder buys nothing; moving the farthest link further off never raises
   the curl.

   Remove inner_share, press, CURL_FULL_SHARE and CURL_FORCE_REF. The inner-link force share must not come back in any
   form: not as a second factor, not as a summand, not as a gate. Do not read hand_q, hand_q_norm or any joint angle for
   this, do not add a force target, and do not name a digit.

   Size, measured by recomputing env_base with the round-18 code on the replayed held steps: at the switch today's
   policy loses 3-6 % of its downstream income (env_mult 0.565 -> 0.549 at epoch 800, 0.558 -> 0.523 at epoch 1000),
   because it earns about as much curl as before, now for the right digits. On the same held states a hand whose touching
   digits all follow the wall earns 78 % more than one where none does (curl_factor 1.0 against 0.55 for five hooked
   tips), and each digit that turns from a fingertip hook into a roll adds about 0.06 to curl_factor, about 8 % of
   everything earned while lifting, carrying and holding. The grasp, lift and hold chain keeps its order, because
   env_mult multiplies all of it alike and `held` / `hold_grip` do not see curl.

2. Keep everything else. Every structural rule stays: second half gated on the object's rise, the envelope quality as
   ONE multiplier on the downstream income with curl as a factor inside it, the ladder, palm and opposition inside
   env_base, no per-finger term, falling-only drop and set-down costs, no penalty on a persistent latch, open-pose factor
   only on the approach, palm seating tied to engagement, uprightness as a multiplier, first-success-only bonus, the
   goal-gated wrench, the round-18 table band. Every budget stays, CURL_FLOOR stays, the squeeze charge stays, and no
   speed or control charge is relaxed.

3. Do not answer the other measurements. Reaching around the object is not missing (the fingertip pads are past the
   widest point on 86-100 % of held steps), so do not add an arc or reach-around term. Do not pay more for links in
   contact than env_depth already does (the rolled ring finger presses mostly with its tip). The y exits of round 18 were
   the environment's bound and the operator has changed it; nothing in the reward names a world axis or a bound.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.629, 0.653, 0.684, 0.712, 0.761, 0.718, 0.714, 0.709, 0.748]  min 0 · mean 0.687 · max 0.771
contact/finger_index_at_success: [-1, 0.969, 0.969, 0.969, 0.973, 0.976, 0.981, 0.981, 0.983, 0.991]  min -1 · mean 0.944 · max 1
contact/finger_middle: [0, 0.659, 0.675, 0.711, 0.737, 0.779, 0.74, 0.729, 0.724, 0.758]  min 0 · mean 0.709 · max 0.783
contact/finger_middle_at_success: [-1, 0.96, 0.98, 0.96, 0.978, 0.974, 0.969, 0.981, 0.98, 0.971]  min -1 · mean 0.943 · max 1
contact/finger_pinky: [0, 0.671, 0.669, 0.699, 0.72, 0.772, 0.733, 0.724, 0.72, 0.754]  min 0 · mean 0.702 · max 0.781
contact/finger_pinky_at_success: [-1, 0.98, 0.975, 0.967, 0.983, 0.984, 0.986, 0.97, 0.98, 0.987]  min -1 · mean 0.947 · max 1
contact/finger_ring: [0, 0.663, 0.669, 0.699, 0.719, 0.778, 0.724, 0.729, 0.731, 0.764]  min 0 · mean 0.704 · max 0.78
contact/finger_ring_at_success: [-1, 0.971, 0.99, 0.981, 0.987, 0.983, 0.984, 0.987, 0.988, 0.988]  min -1 · mean 0.955 · max 1
contact/finger_thumb: [0, 0.7, 0.699, 0.737, 0.763, 0.807, 0.772, 0.761, 0.759, 0.788]  min 0 · mean 0.738 · max 0.815
contact/finger_thumb_at_success: [-1, 1, 1, 0.999, 1, 1, 1, 1, 1, 1]  min -1 · mean 0.968 · max 1
contact/fingers_touching: [0, 3.32, 3.36, 3.53, 3.65, 3.9, 3.69, 3.66, 3.64, 3.81]  min 0 · mean 3.54 · max 3.92
contact/fingers_touching_at_success: [-1, 4.88, 4.91, 4.88, 4.92, 4.92, 4.92, 4.92, 4.93, 4.94]  min -1 · mean 4.82 · max 5
contact/link_force_mean: [0, 2.71, 3.24, 3.29, 3.43, 3.69, 3.54, 3.24, 3.31, 3.45]  min 0 · mean 3.23 · max 3.84
contact/links_touching: [0, 4.27, 4.34, 4.64, 4.74, 5.06, 4.82, 4.76, 4.76, 4.98]  min 0 · mean 4.6 · max 5.1
contact/links_touching_at_success: [-1, 6.16, 6.15, 6.26, 6.24, 6.23, 6.26, 6.22, 6.32, 6.33]  min -1 · mean 6.11 · max 6.39
contact/palm_touching: [0, 0.62, 0.58, 0.592, 0.631, 0.657, 0.62, 0.618, 0.626, 0.657]  min 0 · mean 0.611 · max 0.733
contact/palm_touching_at_success: [-1, 0.947, 0.892, 0.84, 0.871, 0.854, 0.857, 0.882, 0.898, 0.924]  min -1 · mean 0.845 · max 0.956
ctrl/prev_ep_successes_mean: [0, 2.28, 3.11, 3.15, 3.35, 3.22, 3.19, 3.26, 3.26, 3.22]  min 0 · mean 2.93 · max 3.39
done/abnormal: [0, 0.000488, 0.000977, 0.000244, 0.000244, 0.000244, 0.000732, 0.000488, 0.000244, 0.000244]  min 0 · mean 0.000495 · max 0.0022
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.11e-06 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.54e-07 · max 0.000244
done/max_goals: [0, 0.000977, 0.000977, 0.000732, 0.000732, 0.000244, 0.000244, 0.000732, 0.000244, 0.000977]  min 0 · mean 0.000659 · max 0.00366
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.27e-05 · max 0.000488
done/tipped: [0, 0, 0, 0, 0.000244, 0, 0.000244, 0, 0, 0.000244]  min 0 · mean 0.000183 · max 0.00122
episode_lengths/step: [22, 316, 498, 453, 510, 607, 629, 553, 624, 586]  min 22 · mean 514 · max 669
reward/close: [2.6e-06, 0.00519, 0.00662, 0.00829, 0.00706, 0.0082, 0.00863, 0.00818, 0.0077, 0.0078]  min 2.6e-06 · mean 0.00742 · max 0.00969
reward/goal_hold: [0, 0.684, 0.622, 0.662, 0.698, 0.746, 0.699, 0.69, 0.712, 0.708]  min 0 · mean 0.671 · max 0.758
reward/goal_progress: [0, 0.0453, 0.0453, 0.0492, 0.0538, 0.042, 0.0494, 0.0564, 0.0611, 0.0528]  min 0 · mean 0.0491 · max 0.0931
reward/goal_reach: [0, 0.188, 0.17, 0.181, 0.195, 0.222, 0.214, 0.208, 0.217, 0.232]  min 0 · mean 0.198 · max 0.239
reward/grasp_count: [0, 0.0507, 0.0491, 0.0514, 0.0541, 0.0569, 0.0543, 0.0531, 0.0538, 0.0561]  min 0 · mean 0.052 · max 0.0579
reward/grasp_digit: [0, 0.0513, 0.0509, 0.0547, 0.0556, 0.0592, 0.0565, 0.0556, 0.056, 0.0586]  min 0 · mean 0.054 · max 0.0598
reward/grasp_oppose: [0, 0.0924, 0.0924, 0.097, 0.101, 0.107, 0.101, 0.0995, 0.0989, 0.104]  min 0 · mean 0.0973 · max 0.108
reward/grasp_palm: [0, 0.185, 0.174, 0.177, 0.189, 0.197, 0.186, 0.185, 0.188, 0.197]  min 0 · mean 0.183 · max 0.217
reward/lift_height: [0, 0.616, 0.579, 0.612, 0.656, 0.708, 0.663, 0.656, 0.667, 0.67]  min 0 · mean 0.629 · max 0.714
reward/lift_rise: [0, 0.00498, 0.00622, 0.00665, 0.00668, 0.00401, 0.00541, 0.00643, 0.00556, 0.00447]  min 0 · mean 0.00577 · max 0.0218
reward/palm_in: [0.000317, 0.00162, 0.00208, 0.00246, 0.00217, 0.0024, 0.00258, 0.00246, 0.00234, 0.00234]  min 0.000317 · mean 0.00227 · max 0.00286
reward/pen_action: [-0.0219, -0.0236, -0.0232, -0.0232, -0.0229, -0.023, -0.0229, -0.0229, -0.023, -0.0229]  min -0.0243 · mean -0.023 · max -0.0208
reward/pen_arm_fast: [0, -0.00394, -0.00516, -0.00481, -0.00405, -0.00609, -0.00612, -0.00426, -0.00588, -0.00559]  min -0.00854 · mean -0.00515 · max 0
reward/pen_arm_vel: [-0.0271, -0.0534, -0.0541, -0.0539, -0.0575, -0.0573, -0.0576, -0.0571, -0.0574, -0.0591]  min -0.0665 · mean -0.0557 · max -0.0259
reward/pen_cup_vel: [0, -0.0346, -0.0342, -0.0339, -0.0379, -0.0383, -0.0383, -0.0375, -0.0379, -0.0399]  min -0.0499 · mean -0.0362 · max 0
reward/pen_descend: [0, -0.00879, -0.0102, -0.00948, -0.0108, -0.0126, -0.0123, -0.0129, -0.0126, -0.0171]  min -0.0196 · mean -0.0123 · max 0
reward/pen_drift: [0, -0.0612, -0.0543, -0.0563, -0.0609, -0.072, -0.0706, -0.0631, -0.0668, -0.0772]  min -0.0848 · mean -0.0634 · max 0
reward/pen_drop: [-0.000609, -0.0118, -0.0157, -0.0163, -0.0191, -0.0214, -0.0201, -0.0201, -0.0221, -0.0206]  min -0.0284 · mean -0.0187 · max -6.47e-05
reward/pen_setdown: [-0.000145, -0.00283, -0.00289, -0.00297, -0.00335, -0.00336, -0.00308, -0.0035, -0.00384, -0.00506]  min -0.00601 · mean -0.00359 · max -2.35e-05
reward/pen_shove: [-0.00782, -0.0176, -0.0133, -0.0155, -0.0144, -0.0119, -0.0136, -0.0138, -0.0135, -0.0128]  min -0.0447 · mean -0.0147 · max -0.00722
reward/pen_squeeze: [0, -0.169, -0.191, -0.195, -0.194, -0.215, -0.199, -0.19, -0.184, -0.196]  min -0.216 · mean -0.186 · max 0
reward/pen_stall: [0, -0.000218, -0.000158, -0.000176, -0.000194, -0.000138, -0.000165, -0.000217, -0.000219, -0.000286]  min -0.00129 · mean -0.000213 · max 0
reward/pen_table: [0, -3.18e-06, 0, -2.82e-05, -0.000275, -0.000378, -0.000458, -0.000319, -0.000657, -0.000213]  min -0.00281 · mean -0.00038 · max 0
reward/pen_tilt: [-0.000554, -0.0319, -0.0354, -0.0368, -0.035, -0.0352, -0.0349, -0.0369, -0.0344, -0.0386]  min -0.0516 · mean -0.0351 · max -0.000249
reward/reach: [0.0159, 0.00593, 0.00718, 0.00705, 0.0068, 0.00637, 0.00759, 0.0073, 0.00718, 0.00669]  min 0.0047 · mean 0.0072 · max 0.0196
reward/success_bonus: [0, 0.0591, 0.0791, 0.0824, 0.0357, 0.0212, 0.048, 0.0498, 0.118, 0.0454]  min 0 · mean 0.0642 · max 0.272
reward/total: [-0.042, 1.57, 1.44, 1.54, 1.6, 1.68, 1.62, 1.61, 1.73, 1.65]  min -0.0423 · mean 1.57 · max 1.84
rewards/step: [-21.4, 347, 725, 673, 764, 968, 1.03e+03, 905, 1.05e+03, 976]  min -23.7 · mean 793 · max 1.11e+03
task/lifted_frac: [0, 0.646, 0.64, 0.677, 0.694, 0.758, 0.719, 0.698, 0.705, 0.735]  min 0 · mean 0.678 · max 0.761
task/successes_mean: [0, 1.36, 1.35, 1.46, 1.45, 1.48, 1.41, 1.42, 1.42, 1.37]  min 0 · mean 1.36 · max 1.6
task/tilt_deg: [0.084, 4.08, 4.48, 4.68, 4.45, 4.48, 4.44, 4.66, 4.37, 4.88]  min 0.0303 · mean 4.46 · max 6.71
task/tol: [0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09, 0.09, 0.09, 0.081]  min 0.081 · mean 0.0929 · max 0.1

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
