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

I want it to fulfil the following task: Pick up an object with a five-fingered hand and hold it at a goal. This reward is applied to a policy that already reaches the object and closes an envelope grasp around it in most episodes (palm plus thumb, index, middle and pinky in contact), but rarely lifts it. Keep that grasp and teach the lift, the carry to the goal and the hold. The episode still begins with the hand raised away from the object, so reaching and closing must stay worth doing.

Write the reward so that it works for any object shape: use only the object's actual surface as the environment reports it (palm_surface_dist, link_surface_dist, link_surface_dir, link_surface_normal, palm_surface_dir, palm_surface_normal), the contact forces between the hand and the object (palm_cup_force, link_cup_force), the object pose and velocity, and the goal. Do NOT use ctx.approach_done or ctx.envelope_done, and do not use cup_radius, cup_axis, cup_half_height, or any pose template: there is no stage switch. Every term must be a smooth function of the current state.

Reach: pay for bringing the palm close to the object's surface (palm_surface_dist), with the palm facing the surface (palm_normal against palm_surface_normal). This must be worth clearly less than holding the object.

Grasp: the closer the palm and the fingers are to the surface, the more closing the fingers onto it is worth. Pay for contact per digit (thumb, index, middle, ring, pinky; a digit is in contact when any of its links has contact force above about 0.1 N), counted separately and rising steeply with the number of digits: two digits must be worth little, and the fourth and fifth digit must each add clearly more than the first ones. Index, middle and ring contact must be paid at least as much as thumb and pinky. Pay for palm contact. Pay for opposition: the thumb's contact and the fingers' contacts should push on the object from opposite sides (use the surface normals at the nearest points of the touching links; opposing normals mean opposition). A fingertip poke from beside the object, with the palm far from the surface, must earn little.

Lift: once the object is held (palm contact plus opposing digits), pay for the height of the object above its spawn height (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]), rising with height, and for upward motion while held. Holding the closed grasp on the table must not by itself be worth staying for; lowering the held object towards the table must be worth less than keeping it still and clearly less than raising it.

Goal: carry the object to ctx.goal_pos and hold it there, upright and still, so the environment keeps counting successes (ctx.goal_dist <= ctx.success_tol for ctx.success_hold_steps steps). Coming to rest at the goal must be worth more than lifting repeatedly; putting the object down again must not re-earn the lift income. You may add a bonus on ctx.success.

Throughout: keep the object upright, do not drop or knock it over, do not push the hand into the table, lift smoothly. Penalise fast arm motion and fast object motion while holding, but keep these small enough that a slow lift is always better than no lift.

The object must stay upright while it is picked up, carried and held: keep its axis vertical (ctx.cup_tilt is the angle between the object axis and world +z). Lifting or carrying it tilted must be worth clearly less than doing the same upright, so make the lift and goal income scale with uprightness rather than relying on a separate small penalty. In the previous round the object was carried tilted by 27-31 degrees for the whole episode.
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

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
