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
    # ---- proximal finger links: the first segment of each digit, next to the palm ----
    # Each digit is a chain of four rigid segments from the palm outwards: the proximal link (below), then the three
    # links in link_pos: [:, f, 0] the middle link, [:, f, 1] the distal link and [:, f, 2] the fingertip pad, which is
    # rigidly attached to the distal link (so a pad pressed flat on a surface usually loads the distal link too).
    # A digit wrapped around an object touches it with its proximal and middle links as well as its distal link;
    # a digit pressing with its fingertip touches only with the pad and the distal link.
    prox_pos: torch.Tensor             # (N,5,3) position of each digit's proximal link (same finger order as link_pos) [m]; for the index, middle and ring fingers this link is flexed by the finger's root joint; for the thumb and the pinky the root joint is fixed in this hand, so their proximal link moves only with the palm
    prox_cup_force: torch.Tensor       # (N,5) contact force magnitude between each proximal link and the object only [N] (0 = not touching the object), measured the same way as link_cup_force
    prox_surface_dist: torch.Tensor    # (N,5) distance from prox_pos to the nearest point of the object's surface [m], measured like link_surface_dist; prox_pos is a point inside the link, so a touching proximal link reads about 1-3 cm (as for the other finger links) rather than 0
    prox_surface_dir: torch.Tensor     # (N,5,3) unit vector from each prox_pos towards its nearest surface point
    prox_surface_normal: torch.Tensor  # (N,5,3) outward unit normal of the object's surface at that nearest point
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

I want it to fulfil the following task: Pick an object up with a five-fingered hand, carry it to the goal, and hold it there against disturbances, repeatedly. The policy starts from RANDOM WEIGHTS: this one continuous reward, with no stage switch and no checkpoint, has to teach the whole task - travel to the object with an open hand, seat the palm against its side wall and wrap all five digits around it, lift it straight up, carry it to ctx.goal_pos, and stay at rest there while the environment pushes and twists the object.

THE GRASP WE WANT. Each finger rolls around the object: its links follow the wall one after another, from the base of the finger to the tip, with the palm seated against the side wall and the thumb opposing the fingers. The grasp we do NOT want, and which every previous grasp turned into after lifting, is fingertip pressing: the finger roots stay straight, the last joint folds fully, the middle of the finger stands 2.4-3.9 cm off the wall and only the fingertip pad touches it. The grasp must stay rolled while the object is carried and held, not only while it is still on the table.

Per-step ceilings:
- travelling to the object with the fingers open: about 0.6, fading to near zero once the object is held.
- closing and keeping the grasp (digit count with the fifth digit a real step, palm contact, thumb-finger opposition): about 1.2.
- lifting the held object: about 2.5, rising with height up to the height the goal needs (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).
- moving towards the goal: about 2.5, gated on the object being off the table.
- at rest at the goal, upright, really held: about 10, the dominant income, gated on the object being off the table.
- success: a large bonus for the first success of an episode; later successes worth little.

Keep the object upright by multiplying the lift, transport and hold income by uprightness from ctx.cup_tilt. Losing the object must cost a short-lived, bounded amount charged while it is actually falling or back on the table, never an amount proportional to a latch that stays set for the rest of the episode. Lowering a held object must be worth less than keeping it still. Do not push the hand into the table. Penalise fast arm and object motion while holding, small enough that lifting and carrying always beat standing still.

Use no object dimension, no joint index and no per-finger weight: every term must treat the five digits alike and work for any object shape through the surface fields of ctx.
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


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """0 off the object, ~0.6 at 0.1 N, ->1 above a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate and extra force buys nothing."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe normalisation: a zero vector stays zero (its dot products are then 0)."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _sidewall(normal_z: torch.Tensor) -> torch.Tensor:
    """How HORIZONTAL a surface normal is: 1 on a vertical side wall, 0 on a rim face, a lid or the
    table, ~0.3 on a 45 deg rim edge. Used only from a few centimetres in, always with a floor;
    NEVER on the long-range approach (lesson 2)."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Per-step ceilings:
    #   reach            ~0.60  pre-contact only, x (1 - held)
    #                           = 0.35 PLAIN long-range reach (lesson 1 form, NO open-hand factor)
    #                           + 0.24 palm seating (NO open-hand factor) + 0.14 closing
    #                           + 0.04 ADDITIVE open-hand term (far from the wall only)
    #   keep the grasp   1.20   links-per-digit + count ladder + palm contact + opposition
    #   lift             2.50   x uprightness x real hold (`held`); NOT x env_mult (hypothesis B)
    #   transport        2.50   x off_table x env_mult
    #   hold at rest    10.00   real hold, upright, at the goal, still, x env_mult
    #   first success   60.00   one-off; later successes 2.0, x env_mult
    # =============================================================================================
    W_REACH = 0.35        # long range, PLAIN palm-to-surface distance (lesson 1), fades once held
    W_OPEN = 0.04         # ADDITIVE open-pose income on the way in: below the first digit-contact
    #                       increment (~0.05 at the wall) and far below the 0.195 of lesson 4
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike
    W_PALM = 0.30         # palm contact: a pinch must not score like a whole-hand wrap
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height
    W_RISE = 0.40         # upward speed, only until `lifted` latches (income gate, not penalty)
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 2.00         # per step, ONLY while the object is falling without a hold
    P_SETDOWN = 1.00      # per step, ONLY while the object is coming down onto the table
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    P_SQUEEZE = 0.50      # force beyond what a hold needs
    P_TABLE = 0.60        # palm or finger links driven onto the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: small, so lifting and carrying beat standing still
    P_CUP_SPIN = 0.06
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    PALM_STEEP = 0.02     # seating scale
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link
    REACH_FAR = 0.20      # far scale of the plain reach: covers the 0.15-0.49 m start distances
    REACH_NEAR = 0.05     # near scale of the plain reach (last few centimetres)
    OPEN_MEAN_SCALE = 0.12   # mean deviation from the default open pose
    OPEN_MAX_SCALE = 0.30    # and the WORST joint, which is what actually blocks a wrap
    OPEN_NEAR = 0.08      # open-pose term pays in full at this palm-surface distance or beyond
    OPEN_CONTACT = 0.02   # and is zero by here, where contact starts
    PALMIN_ENGAGE_FLOOR = 0.35   # seating credit an entirely disengaged hand still keeps
    TABLE_RISE = 0.04     # smooth "off the table" scale
    TILT_DEAD = math.radians(5.0)
    TILT_SCALE = math.radians(10.0)
    SQ_TOTAL = 40.0
    SQ_LINK = 15.0
    FALL_DEAD = 0.03
    FALL_SCALE = 0.20
    SET_DEAD = 0.02
    SET_SCALE = 0.05
    SET_LIFTOFF = 0.005
    SET_BAND = 0.04
    GOAL_TOL_MULT = 3.0
    GOAL_TOL_MIN = 0.03
    GOAL_FINE = 0.04
    SETTLE_FLOOR = 0.40
    DWELL_BASE = 0.70
    DWELL_SCALE = 1.5
    ENV_FLOOR = 0.10
    ENV_W_COUNT = 0.30
    ENV_W_DEPTH = 0.35
    ENV_W_PALM = 0.25
    ENV_W_OPP = 0.10
    ENV_DEPTH_LINKS = 2.0
    ENV_NEAR_SHARE = 0.25
    ENV_NEAR_SCALE = 0.01
    CURL_FLOOR = 0.50        # env_q keeps half at zero curl
    FOLLOW_DEAD = 0.010      # m: resolution of the surface distance, NOT a pose target
    FOLLOW_SCALE = 0.006     # m: 1.1 cm -> 0.85, 1.9 cm -> 0.22, 2.0-2.5 cm -> 0.19-0.08
    FLOOR_BAND_TOP = 0.020
    FLOOR_BAND_FULL = 0.010

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    n_succ = ctx.num_successes.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    n_movable = movable.sum().clamp(min=1.0)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0
    digit_joint[1, 4:7] = 1.0
    digit_joint[2, 8:11] = 1.0
    digit_joint[3, 12:15] = 1.0
    digit_joint[4, 17:19] = 1.0
    n_digit_joint = digit_joint.sum(-1)

    # ---- height and vertical velocity ------------------------------------------------------------
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    off_table = torch.tanh(height / TABLE_RISE)
    on_table = 1.0 - off_table
    vz = ctx.cup_lin_vel[:, 2]
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)

    # ---- contact and grasp quality ---------------------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5)
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)
    finger_any = digit_touch[:, 1:].amax(dim=-1)
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    link_side = _sidewall(ctx.link_surface_normal[..., 2])
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side
    link_cw = link_c * link_side_w
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- open-pose measure (mean AND worst movable joint) ----------------------------------------
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))

    # ---- measured per-digit flexion --------------------------------------------------------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint
    flex = (closure / 0.5).clamp(max=1.0)

    # ---- palm against the object's real surface --------------------------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    cq = 0.35 + 0.65 * seat

    # open-pose gate: full at 8 cm or more, zero by 2 cm (used ONLY by the additive open term)
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)

    # ---- opposition from the surface normals at the touching links -------------------------------
    normals = ctx.link_surface_normal
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)

    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # ---- envelope quality (multiplier on transport, hold and success only) ------------------------
    digit_gap = (ctx.link_surface_dist.amin(dim=-1) - LINK_DEAD).clamp(min=0.0)
    digit_eng = torch.maximum(digit_touch, ENV_NEAR_SHARE * torch.exp(-digit_gap / ENV_NEAR_SCALE))
    env_rank = torch.tensor([0.02, 0.02, 0.06, 0.15, 0.75], device=device, dtype=dtype)
    eng_desc, _ = torch.sort(digit_eng, dim=-1, descending=True)
    env_count = (eng_desc * env_rank).sum(dim=-1)
    count_depth = (link_c.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0)
    env_depth = count_depth.mean(dim=-1)
    env_base = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
                + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)

    # wall-following curl (lesson 13), a FACTOR of env_q, same formula for every digit
    link_dist = ctx.link_surface_dist
    lift_off = link_dist.amax(dim=-1) - link_dist.amin(dim=-1)
    follow = torch.exp(-(lift_off - FOLLOW_DEAD).clamp(min=0.0) / FOLLOW_SCALE)
    digit_curl = digit_touch * follow
    hand_curl = digit_curl.mean(dim=-1)
    curl_factor = CURL_FLOOR + (1.0 - CURL_FLOOR) * hand_curl
    env_q = env_base * curl_factor
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- fading out once the object is really held
    # =============================================================================================
    fade = loose
    # PLAIN reach of lesson 1: no open-hand factor, no face factor, no side-wall factor (lesson 2)
    reach_far = 1.0 - torch.tanh(palm_d / REACH_FAR)
    reach_near = 1.0 - torch.tanh(palm_d / REACH_NEAR)
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near) * (0.45 + 0.55 * grip)

    # ADDITIVE open-hand income: only while the palm is far from the surface (full at 8 cm, zero by
    # 2 cm) and zero once held, so it can never become the resting point at the wall (lesson 4)
    r_open = W_OPEN * fade * open_gate * open_pose

    # palm seating paired with finger engagement (no open-hand factor any more)
    engage = (0.5 * flex.mean(dim=-1) + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    r_palm_in = W_PALM_IN * fade * palm_near * (0.3 + 0.7 * face) * palm_side_w * seat_engage

    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total
    # =============================================================================================
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_oppose = r_oppose * table_decay

    # =============================================================================================
    # LIFT -- 2.50 total. Hypothesis B: gated by `held` and uprightness only, NOT by env_mult, so the
    # first lift is not throttled by an envelope the policy has not formed yet.
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)
    r_lift = W_LIFT * held * upright * lift_shape
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, x off_table, keeps env_mult
    # =============================================================================================
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = (W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)
              * env_mult)
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10) * env_mult)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, x off_table, keeps env_mult
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded, no latch
    # =============================================================================================
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    set_band = torch.tanh(height / SET_LIFTOFF) * torch.exp(-height / SET_BAND)
    p_setdown = P_SETDOWN * set_band * torch.tanh(((-vz) - SET_DEAD).clamp(min=0.0) / SET_SCALE)
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS
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
        r_reach + r_open + r_palm_in + r_close
        + r_count + r_digit + r_palm + r_oppose
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_drift - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero

    components = {
        "reach": r_reach,
        "open_hand": r_open,
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
iter_00 trained only Hypothesis B (iter_00/selection.json); Hypothesis A was not trained.

Left arm, 5,460 epochs from random weights, measured in 500-epoch windows (tensorboard). The right arm ran the byte-identical reward on the mirrored scene at the same time.

- The approach was learned: palm-to-object gap 0.162 -> 0.085 m by epoch 1,000 and 0.063-0.067 m from epoch 2,500 on, reach episodes 0.81-0.95.
- The hand settled into a two-digit pinch and never left it: from epoch 2,500 to 5,460 the thumb touches the object in 0.73 of steps and the ring finger in 0.58, while the index finger touches in 0.0007, the middle 0.002, the pinky 0.012 and the palm 0.0016. Fingers touching 1.3 on average, links touching 1.4.
- The palm stays 0.040 m from the surface (palm surface distance, the same value the right arm had just before it formed its grasp) and never makes contact.
- Grasp episodes 0.05-0.15, lift 0.000-0.006, envelope 0, successes 0 for the whole run; the disturbance never engaged.
- The hand actions are saturated at +-1 in 0.95 of steps (right arm 0.57-0.66 over the same phase); the distal finger segments are commanded 0.85 closed and the proximal ones 0.14 to 0.30.
- The largest cost in this state is the shove penalty (-0.048 per step), larger than every grasp income term (grasp count 0.004, digit 0.007, palm 0.0003, opposition 0.017); palm seating pays 0.113 and reach 0.147.
- Table and termination costs are the same as on the right arm (table penalty -0.0008, no terminations), so nothing charges the left arm for moving down or in.
- The right arm went through the same state (palm contact 0.007, gap 0.063 m, fingers 0.07 at epochs 1,500-2,000) and left it at epochs 2,000-2,500 when the index finger (0.11) and the palm (0.03) started touching; it then reached grasp 0.92, lift 0.90, envelope 0.79, far-start success 0.61 with palm contact 0.65 and the index finger in 0.81 of steps.

To make the code more accurate and train better robot, the feedback for improvement is:
Change only what Hypothesis A says. Everything else stays, including every term that the right arm used to learn the whole task. The change must be written so that it applies identically to the mirrored arm.

## Hypothesis A
- Before a hold exists, seating the palm against the object must be worth clearly more than adding a second or third digit to a pinch: with one or two digits touching, palm contact must raise the per-step grasp income by more than the next digit would, so that the thumb-and-one-finger pinch (thumb 0.73, ring 0.58, palm 0.0016 for 3,000 epochs) is no longer the best state reachable without lifting.
- The object motion that seating the palm causes while digits already touch must not be charged as shoving: the shove cost must stay for pushing or tipping the object with an open or empty hand, but must not exceed the palm-seating gain when the palm is closing in on a pinched object (it is now -0.048 per step, the largest cost in the pinch state).

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
    table, ~0.3 on a 45 deg rim edge. Used only from a few centimetres in, always with a floor;
    NEVER on the long-range approach (lesson 2)."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Per-step ceilings:
    #   reach            ~0.60  pre-contact only, x (1 - held)
    #                           = 0.35 PLAIN long-range reach (lesson 1 form, NO open-hand factor)
    #                           + 0.24 palm seating (NO open-hand factor) + 0.14 closing
    #                           + 0.04 ADDITIVE open-hand term (far from the wall only)
    #   keep the grasp   1.20   links-per-digit + count ladder + palm contact + opposition
    #                    +0.20  Hypothesis A: pre-hold palm-seat bonus (any digit touching + palm
    #                           contact); halves at a full hold on the table, gone once a held
    #                           object is off the table -> ~1.30 on the table, 1.20 when carried
    #   lift             2.50   x uprightness x real hold (`held`); NOT x env_mult (hypothesis B)
    #   transport        2.50   x off_table x env_mult
    #   hold at rest    10.00   real hold, upright, at the goal, still, x env_mult
    #   first success   60.00   one-off; later successes 2.0, x env_mult
    # =============================================================================================
    W_REACH = 0.35        # long range, PLAIN palm-to-surface distance (lesson 1), fades once held
    W_OPEN = 0.04         # ADDITIVE open-pose income on the way in: below the first digit-contact
    #                       increment (~0.05 at the wall) and far below the 0.195 of lesson 4
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike
    W_PALM = 0.30         # palm contact: a pinch must not score like a whole-hand wrap
    W_PALM_SEAT = 0.20    # Hypothesis A: palm contact while >=1 digit touches, before a held lift.
    #                       With 1 digit: palm adds 0.15 + 0.20 = 0.35 > next digit (<= ~0.24 incl.
    #                       opposition); with 2 digits: 0.225 + 0.20 = 0.425 > next digit (<= ~0.16)
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height
    W_RISE = 0.40         # upward speed, only until `lifted` latches (income gate, not penalty)
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 2.00         # per step, ONLY while the object is falling without a hold
    P_SETDOWN = 1.00      # per step, ONLY while the object is coming down onto the table
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    SHOVE_RELIEF = 0.80   # Hypothesis A: share of the shove cost waived while a digit already
    #                       touches AND the palm is closing in (max 0.10/step at palm contact,
    #                       below the 0.20 palm-seat bonus); full cost with an open or empty hand
    P_SQUEEZE = 0.50      # force beyond what a hold needs
    P_TABLE = 0.60        # palm or finger links driven onto the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: small, so lifting and carrying beat standing still
    P_CUP_SPIN = 0.06
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    PALM_STEEP = 0.02     # seating scale
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link
    REACH_FAR = 0.20      # far scale of the plain reach: covers the 0.15-0.49 m start distances
    REACH_NEAR = 0.05     # near scale of the plain reach (last few centimetres)
    OPEN_MEAN_SCALE = 0.12   # mean deviation from the default open pose
    OPEN_MAX_SCALE = 0.30    # and the WORST joint, which is what actually blocks a wrap
    OPEN_NEAR = 0.08      # open-pose term pays in full at this palm-surface distance or beyond
    OPEN_CONTACT = 0.02   # and is zero by here, where contact starts
    PALMIN_ENGAGE_FLOOR = 0.35   # seating credit an entirely disengaged hand still keeps
    TABLE_RISE = 0.04     # smooth "off the table" scale
    TILT_DEAD = math.radians(5.0)
    TILT_SCALE = math.radians(10.0)
    SQ_TOTAL = 40.0
    SQ_LINK = 15.0
    FALL_DEAD = 0.03
    FALL_SCALE = 0.20
    SET_DEAD = 0.02
    SET_SCALE = 0.05
    SET_LIFTOFF = 0.005
    SET_BAND = 0.04
    GOAL_TOL_MULT = 3.0
    GOAL_TOL_MIN = 0.03
    GOAL_FINE = 0.04
    SETTLE_FLOOR = 0.40
    DWELL_BASE = 0.70
    DWELL_SCALE = 1.5
    ENV_FLOOR = 0.10
    ENV_W_COUNT = 0.30
    ENV_W_DEPTH = 0.35
    ENV_W_PALM = 0.25
    ENV_W_OPP = 0.10
    ENV_DEPTH_LINKS = 2.0
    ENV_NEAR_SHARE = 0.25
    ENV_NEAR_SCALE = 0.01
    CURL_FLOOR = 0.50        # env_q keeps half at zero curl
    FOLLOW_DEAD = 0.010      # m: resolution of the surface distance, NOT a pose target
    FOLLOW_SCALE = 0.006     # m: 1.1 cm -> 0.85, 1.9 cm -> 0.22, 2.0-2.5 cm -> 0.19-0.08
    FLOOR_BAND_TOP = 0.020
    FLOOR_BAND_FULL = 0.010

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    n_succ = ctx.num_successes.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    n_movable = movable.sum().clamp(min=1.0)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0
    digit_joint[1, 4:7] = 1.0
    digit_joint[2, 8:11] = 1.0
    digit_joint[3, 12:15] = 1.0
    digit_joint[4, 17:19] = 1.0
    n_digit_joint = digit_joint.sum(-1)

    # ---- height and vertical velocity ------------------------------------------------------------
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    off_table = torch.tanh(height / TABLE_RISE)
    on_table = 1.0 - off_table
    vz = ctx.cup_lin_vel[:, 2]
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)

    # ---- contact and grasp quality ---------------------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5)
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)
    finger_any = digit_touch[:, 1:].amax(dim=-1)
    any_digit = digit_touch.amax(dim=-1)                           # (N,) any of the five digits, alike
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    link_side = _sidewall(ctx.link_surface_normal[..., 2])
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side
    link_cw = link_c * link_side_w
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- open-pose measure (mean AND worst movable joint) ----------------------------------------
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))

    # ---- measured per-digit flexion --------------------------------------------------------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint
    flex = (closure / 0.5).clamp(max=1.0)

    # ---- palm against the object's real surface --------------------------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    cq = 0.35 + 0.65 * seat

    # open-pose gate: full at 8 cm or more, zero by 2 cm (used ONLY by the additive open term)
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)

    # ---- opposition from the surface normals at the touching links -------------------------------
    normals = ctx.link_surface_normal
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)

    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # ---- envelope quality (multiplier on transport, hold and success only) ------------------------
    digit_gap = (ctx.link_surface_dist.amin(dim=-1) - LINK_DEAD).clamp(min=0.0)
    digit_eng = torch.maximum(digit_touch, ENV_NEAR_SHARE * torch.exp(-digit_gap / ENV_NEAR_SCALE))
    env_rank = torch.tensor([0.02, 0.02, 0.06, 0.15, 0.75], device=device, dtype=dtype)
    eng_desc, _ = torch.sort(digit_eng, dim=-1, descending=True)
    env_count = (eng_desc * env_rank).sum(dim=-1)
    count_depth = (link_c.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0)
    env_depth = count_depth.mean(dim=-1)
    env_base = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
                + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)

    # wall-following curl (lesson 13), a FACTOR of env_q, same formula for every digit
    link_dist = ctx.link_surface_dist
    lift_off = link_dist.amax(dim=-1) - link_dist.amin(dim=-1)
    follow = torch.exp(-(lift_off - FOLLOW_DEAD).clamp(min=0.0) / FOLLOW_SCALE)
    digit_curl = digit_touch * follow
    hand_curl = digit_curl.mean(dim=-1)
    curl_factor = CURL_FLOOR + (1.0 - CURL_FLOOR) * hand_curl
    env_q = env_base * curl_factor
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- fading out once the object is really held
    # =============================================================================================
    fade = loose
    # PLAIN reach of lesson 1: no open-hand factor, no face factor, no side-wall factor (lesson 2)
    reach_far = 1.0 - torch.tanh(palm_d / REACH_FAR)
    reach_near = 1.0 - torch.tanh(palm_d / REACH_NEAR)
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near) * (0.45 + 0.55 * grip)

    # ADDITIVE open-hand income: only while the palm is far from the surface (full at 8 cm, zero by
    # 2 cm) and zero once held, so it can never become the resting point at the wall (lesson 4)
    r_open = W_OPEN * fade * open_gate * open_pose

    # palm seating paired with finger engagement (no open-hand factor any more)
    engage = (0.5 * flex.mean(dim=-1) + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    r_palm_in = W_PALM_IN * fade * palm_near * (0.3 + 0.7 * face) * palm_side_w * seat_engage

    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total (+ the Hypothesis A pre-hold palm-seat bonus)
    # =============================================================================================
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # Hypothesis A: before a hold exists, seating the palm on an object already touched by any digit
    # pays a flat bonus that does NOT depend on how many digits touch, so with one or two digits palm
    # contact is worth more than the next digit. It halves at a full hold on the table (a third or
    # fourth digit is still a net gain: r_palm alone rises more) and vanishes once a held object is
    # off the table, where the lift income (>= 1.05 x tanh(h/0.03) x held) outgrows the fade.
    pre_hold = (1.0 - held * (0.5 + 0.5 * off_table)).clamp(min=0.0, max=1.0)
    r_palm_seat = W_PALM_SEAT * palm_c * any_digit * pre_hold

    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_palm_seat = r_palm_seat * table_decay
    r_oppose = r_oppose * table_decay

    # =============================================================================================
    # LIFT -- 2.50 total. Hypothesis B: gated by `held` and uprightness only, NOT by env_mult, so the
    # first lift is not throttled by an envelope the policy has not formed yet.
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)
    r_lift = W_LIFT * held * upright * lift_shape
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, x off_table, keeps env_mult
    # =============================================================================================
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = (W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)
              * env_mult)
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10) * env_mult)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, x off_table, keeps env_mult
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded, no latch
    # =============================================================================================
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    set_band = torch.tanh(height / SET_LIFTOFF) * torch.exp(-height / SET_BAND)
    p_setdown = P_SETDOWN * set_band * torch.tanh(((-vz) - SET_DEAD).clamp(min=0.0) / SET_SCALE)
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    shove_raw = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                            + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    # Hypothesis A: the palm closing in on an object that a digit already touches (palm facing it
    # inside 8 cm, full by 2 cm, or already in palm contact) is seating, not shoving. The relief needs
    # a touching digit, so pushing or tipping with an open or empty hand keeps the full cost.
    palm_closing = torch.maximum((1.0 - open_gate) * (0.5 + 0.5 * face), palm_c)
    shove_relief = SHOVE_RELIEF * any_digit * palm_closing
    p_shove = shove_raw * (1.0 - shove_relief)
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
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
        r_reach + r_open + r_palm_in + r_close
        + r_count + r_digit + r_palm + r_palm_seat + r_oppose
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_drift - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero

    components = {
        "reach": r_reach,
        "open_hand": r_open,
        "palm_in": r_palm_in,
        "close": r_close,
        "grasp_count": r_count,
        "grasp_digit": r_digit,
        "grasp_palm": r_palm,
        "grasp_palm_seat": r_palm_seat,
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
iter_01 trained only Hypothesis A (iter_01/selection.json) on the left arm; the right arm trained the iter_00 reward, which differs only in the two iter_01 changes.

Both arms learned the whole task from random weights (left 7,211 epochs, right 13,679 epochs; last 500 epochs):
- right: grasp 0.90, lift 0.88, far-start success 0.58 at tolerance 0.023, envelope 0.81, palm contact 0.73, thumb/index/middle/ring 0.84-0.86, pinky 0.33, disturbance at its maximum (20 N/kg).
- left: grasp 0.81, lift 0.77, far-start success 0.34 at tolerance 0.026, envelope 0.66, palm 0.70. Around epoch 6,000 it had grasp 0.94, lift 0.92 and success 0.61; in its last 1,200 epochs the pinky started touching (0.005 -> 0.43) and grasp, lift and success fell (0.94 -> 0.76, 0.92 -> 0.70, 0.61 -> 0.23) at a constant tolerance.

What the operator saw in the final videos (deterministic replay, 12 environments): on both arms the last finger joints (joint 4) are still curled far too much; the right hand approaches naturally and lifts; the left hand approaches naturally and then lifts abruptly.

Measured shape of the grasp:
- Closure of the last finger joints is 0.72 (right) and 0.55 (left) of their range, against 0.24 and 0.31 for the finger root joints; commanded and measured values are equal, so the policy commands this shape, it is not blocked into it. The shape is already there before the lift (right 0.68 at epochs 2,500-3,000).
- In a replay of an earlier policy of the same task (256 episodes), whenever the four fingers held the object, the middle finger link touched it in 0.00 of steps and the fingertip pad in 0.90-1.00; even steps whose joint pattern looked rolled were carried by the pad (0.90). The middle links stood 3.3 cm (median) off the wall.
- The environment now reports each digit's proximal link as well (the segment next to the palm): its position, its contact force with the object and its distance, direction and normal to the object's surface. Nothing in the reward has used those links, the middle link or the proximal link so far.
- The left arm lifts faster than the right: object speed while lifted 0.19 vs 0.11 m/s, arm joint speed 99th percentile while lifted 1.63 vs 0.92 rad/s. Link contact force 4.2 N (right 2.7 N), squeeze cost -0.25 per step (right -0.16).

How the current reward scores a digit: a digit counts as touching when any of its three sensed links touches, so a fingertip pad alone gives full digit contact; wrap depth counts touching links and is full at two, which the pad and the distal link reach together; the wall-following curl factor sits inside the envelope quality with a floor of 0.5.

To make the code more accurate and train better robot, the feedback for improvement is:
Change only what Hypothesis A says. Keep every other term, including the two changes of the previous round (the pre-hold palm-seat bonus and the shove relief), which made the left arm learn the whole task. The reward must stay identical for the mirrored arm (no digit or side is named).

## Hypothesis A
- A digit must count as holding the object by its finger segments, not by its fingertip: wherever the reward counts touching digits or their links (the grasp ladder, the hold gate, the envelope quality, the wrap depth), a digit whose proximal, middle or distal link touches counts fully, while a digit touching only with its fingertip pad counts as a small fraction of that. Use the proximal link fields that the environment now provides.
- Because the middle and proximal links never touch today (they stand about 3 cm off the wall), give those segments a smooth pull towards the object's surface for each digit that already touches it, based on their surface distances, so that there is a gradient from the fingertip grasp the policy knows towards a grasp carried by the finger segments. This pull must be smaller than the lift and hold income so that it never becomes a reason to keep the object on the table.

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
    table, ~0.3 on a 45 deg rim edge. Used only from a few centimetres in, always with a floor;
    NEVER on the long-range approach (lesson 2)."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Per-step ceilings:
    #   reach            ~0.60  pre-contact only, x (1 - held)
    #                           = 0.35 PLAIN long-range reach (lesson 1 form, NO open-hand factor)
    #                           + 0.24 palm seating (NO open-hand factor) + 0.14 closing
    #                           + 0.04 ADDITIVE open-hand term (far from the wall only)
    #   keep the grasp   1.20   links-per-digit + count ladder + palm contact + opposition
    #                    +0.20  pre-hold palm-seat bonus (previous round, kept)
    #                    +0.15  Hypothesis A: segment pull (proximal + middle link towards the surface
    #                           on digits that already touch); not height-gated, so it pays the same
    #                           on the table and in the air and never favours staying on the table
    #   lift             2.50   x uprightness x real hold (`held`); NOT x env_mult
    #   transport        2.50   x off_table x env_mult
    #   hold at rest    10.00   real hold, upright, at the goal, still, x env_mult
    #   first success   60.00   one-off; later successes 2.0, x env_mult
    #
    # Hypothesis A: a digit is counted by its FINGER SEGMENTS (proximal, middle, distal link) at full
    # weight and by its fingertip pad alone only at PAD_SHARE, wherever digits or links are counted
    # (grasp ladder, hold gate, envelope quality, wrap depth).
    # =============================================================================================
    W_REACH = 0.35        # long range, PLAIN palm-to-surface distance (lesson 1), fades once held
    W_OPEN = 0.04         # ADDITIVE open-pose income on the way in
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm
    W_DIGIT = 0.30        # per-digit SEGMENT COUNT, every digit weighted alike
    W_PALM = 0.30         # palm contact: a pinch must not score like a whole-hand wrap
    W_PALM_SEAT = 0.20    # palm contact while >=1 digit touches, before a held lift (previous round)
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)
    W_SEG_PULL = 0.15     # Hypothesis A: pull of proximal/middle links towards the wall; far below the
    #                       lift (2.5) and hold (10) income
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height
    W_RISE = 0.40         # upward speed, only until `lifted` latches (income gate, not penalty)
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 2.00         # per step, ONLY while the object is falling without a hold
    P_SETDOWN = 1.00      # per step, ONLY while the object is coming down onto the table
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    SHOVE_RELIEF = 0.80   # share of the shove cost waived while a digit touches AND the palm closes in
    P_SQUEEZE = 0.50      # force beyond what a hold needs
    P_TABLE = 0.60        # palm or finger links driven onto the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: small, so lifting and carrying beat standing still
    P_CUP_SPIN = 0.06
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    PAD_SHARE = 0.25      # Hypothesis A: a digit touching ONLY with its fingertip pad counts this much
    PULL_DEAD = 0.012     # m: a touching link point reads 1-3 cm (link points sit inside the links)
    PULL_SCALE = 0.02     # m: 1.8 cm -> 0.74, 3.3 cm -> 0.35, 5 cm -> 0.15 (gradient from today's gap)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    PALM_STEEP = 0.02     # seating scale
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link
    REACH_FAR = 0.20      # far scale of the plain reach: covers the 0.15-0.49 m start distances
    REACH_NEAR = 0.05     # near scale of the plain reach (last few centimetres)
    OPEN_MEAN_SCALE = 0.12   # mean deviation from the default open pose
    OPEN_MAX_SCALE = 0.30    # and the WORST joint, which is what actually blocks a wrap
    OPEN_NEAR = 0.08      # open-pose term pays in full at this palm-surface distance or beyond
    OPEN_CONTACT = 0.02   # and is zero by here, where contact starts
    PALMIN_ENGAGE_FLOOR = 0.35   # seating credit an entirely disengaged hand still keeps
    TABLE_RISE = 0.04     # smooth "off the table" scale
    TILT_DEAD = math.radians(5.0)
    TILT_SCALE = math.radians(10.0)
    SQ_TOTAL = 40.0
    SQ_LINK = 15.0
    FALL_DEAD = 0.03
    FALL_SCALE = 0.20
    SET_DEAD = 0.02
    SET_SCALE = 0.05
    SET_LIFTOFF = 0.005
    SET_BAND = 0.04
    GOAL_TOL_MULT = 3.0
    GOAL_TOL_MIN = 0.03
    GOAL_FINE = 0.04
    SETTLE_FLOOR = 0.40
    DWELL_BASE = 0.70
    DWELL_SCALE = 1.5
    ENV_FLOOR = 0.10
    ENV_W_COUNT = 0.30
    ENV_W_DEPTH = 0.35
    ENV_W_PALM = 0.25
    ENV_W_OPP = 0.10
    ENV_DEPTH_LINKS = 2.0
    ENV_NEAR_SHARE = 0.25
    ENV_NEAR_SCALE = 0.01
    CURL_FLOOR = 0.50        # env_q keeps half at zero curl
    FOLLOW_DEAD = 0.010      # m: resolution of the surface distance, NOT a pose target
    FOLLOW_SCALE = 0.006     # m: 1.1 cm -> 0.85, 1.9 cm -> 0.22, 2.0-2.5 cm -> 0.19-0.08
    FLOOR_BAND_TOP = 0.020
    FLOOR_BAND_FULL = 0.010

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    n_succ = ctx.num_successes.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    n_movable = movable.sum().clamp(min=1.0)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0
    digit_joint[1, 4:7] = 1.0
    digit_joint[2, 8:11] = 1.0
    digit_joint[3, 12:15] = 1.0
    digit_joint[4, 17:19] = 1.0
    n_digit_joint = digit_joint.sum(-1)

    # ---- height and vertical velocity ------------------------------------------------------------
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    off_table = torch.tanh(height / TABLE_RISE)
    on_table = 1.0 - off_table
    vz = ctx.cup_lin_vel[:, 2]
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)

    # ---- contact and grasp quality ---------------------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3) middle, distal, pad
    prox_c = _soft_contact(ctx.prox_cup_force)                     # (N,5)   proximal link
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)

    # Hypothesis A: the four segments of each digit, from the palm outwards:
    # [proximal, middle, distal, fingertip pad]. Segments count fully, the pad alone only PAD_SHARE.
    seg_c = torch.cat([prox_c.unsqueeze(-1), link_c], dim=-1)      # (N,5,4)
    seg_w = torch.tensor([1.0, 1.0, 1.0, PAD_SHARE], device=device, dtype=dtype)
    seg_cw = seg_c * seg_w                                         # (N,5,4)
    digit_touch = seg_cw.amax(dim=-1)                              # (N,5) held by finger segments
    n_touch = digit_touch.sum(dim=-1)
    finger_any = digit_touch[:, 1:].amax(dim=-1)

    # plain "touching at all" (any segment, the pad included at full weight): used only where the
    # previous rounds' terms ask whether a digit touches (reach grip, palm terms, shove relief) and
    # for the segment pull
    touch_any = seg_c.amax(dim=-1)                                 # (N,5)
    n_any = touch_any.sum(dim=-1)
    any_digit = touch_any.amax(dim=-1)                             # (N,)
    grip = (0.8 * n_any / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    link_side = _sidewall(ctx.link_surface_normal[..., 2])         # (N,5,3)
    prox_side = _sidewall(ctx.prox_surface_normal[..., 2])         # (N,5)
    seg_side = torch.cat([prox_side.unsqueeze(-1), link_side], dim=-1)
    seg_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * seg_side
    # full at three touching segments on a wall; a pad alone gives (0.25/3)^1.5 ~ 0.02
    digit_links = ((seg_cw * seg_side_w).sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)  # (N,5)

    # ---- open-pose measure (mean AND worst movable joint) ----------------------------------------
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))

    # ---- measured per-digit flexion --------------------------------------------------------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint
    flex = (closure / 0.5).clamp(max=1.0)

    # ---- palm against the object's real surface --------------------------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    cq = 0.35 + 0.65 * seat

    # open-pose gate: full at 8 cm or more, zero by 2 cm (used by the additive open term and the
    # shove relief)
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)

    # ---- opposition from the surface normals at the touching segments ----------------------------
    # Hypothesis A: segment-weighted (proximal link included, pad at PAD_SHARE), part of the hold gate
    seg_normals = torch.cat([ctx.prox_surface_normal.unsqueeze(2), ctx.link_surface_normal], dim=2)
    thumb_n = _unit((seg_cw[:, 0, :].unsqueeze(-1) * seg_normals[:, 0]).sum(dim=1))
    finger_n = _unit((seg_cw[:, 1:, :].unsqueeze(-1) * seg_normals[:, 1:]).sum(dim=(1, 2)))
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)

    # hold gate: counts digits held by their segments (a pad alone is PAD_SHARE of a digit)
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # ---- envelope quality (multiplier on transport, hold and success only) ------------------------
    # Hypothesis A: the "near" share is measured on the finger segments (proximal, middle, distal),
    # not on the pad, and the digit count / wrap depth use the segment-weighted contacts
    seg_dist = torch.cat([ctx.prox_surface_dist.unsqueeze(-1), ctx.link_surface_dist[..., :2]], dim=-1)
    digit_gap = (seg_dist.amin(dim=-1) - LINK_DEAD).clamp(min=0.0)
    digit_eng = torch.maximum(digit_touch, ENV_NEAR_SHARE * torch.exp(-digit_gap / ENV_NEAR_SCALE))
    env_rank = torch.tensor([0.02, 0.02, 0.06, 0.15, 0.75], device=device, dtype=dtype)
    eng_desc, _ = torch.sort(digit_eng, dim=-1, descending=True)
    env_count = (eng_desc * env_rank).sum(dim=-1)
    count_depth = (seg_cw.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0)   # full at two segments
    env_depth = count_depth.mean(dim=-1)
    env_base = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
                + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)

    # wall-following curl (lesson 13), a FACTOR of env_q, same formula for every digit
    link_dist = ctx.link_surface_dist
    lift_off = link_dist.amax(dim=-1) - link_dist.amin(dim=-1)
    follow = torch.exp(-(lift_off - FOLLOW_DEAD).clamp(min=0.0) / FOLLOW_SCALE)
    digit_curl = digit_touch * follow
    hand_curl = digit_curl.mean(dim=-1)
    curl_factor = CURL_FLOOR + (1.0 - CURL_FLOOR) * hand_curl
    env_q = env_base * curl_factor
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- fading out once the object is really held
    # =============================================================================================
    fade = loose
    reach_far = 1.0 - torch.tanh(palm_d / REACH_FAR)
    reach_near = 1.0 - torch.tanh(palm_d / REACH_NEAR)
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near) * (0.45 + 0.55 * grip)

    r_open = W_OPEN * fade * open_gate * open_pose

    engage = (0.5 * flex.mean(dim=-1) + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    r_palm_in = W_PALM_IN * fade * palm_near * (0.3 + 0.7 * face) * palm_side_w * seat_engage

    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total (+ pre-hold palm-seat bonus + Hypothesis A segment pull)
    # =============================================================================================
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_any / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # pre-hold palm-seat bonus (previous round, unchanged)
    pre_hold = (1.0 - held * (0.5 + 0.5 * off_table)).clamp(min=0.0, max=1.0)
    r_palm_seat = W_PALM_SEAT * palm_c * any_digit * pre_hold

    # Hypothesis A: for every digit that already touches the object (any segment, pad included),
    # pull its proximal and middle links towards the object's surface. Smooth in the surface distance,
    # so there is a gradient from today's ~3 cm gap; same formula for every digit; not gated on height,
    # so it pays the same while the object is carried and held as on the table.
    prox_near = torch.exp(-(ctx.prox_surface_dist - PULL_DEAD).clamp(min=0.0) / PULL_SCALE)       # (N,5)
    mid_near = torch.exp(-(ctx.link_surface_dist[:, :, 0] - PULL_DEAD).clamp(min=0.0) / PULL_SCALE)
    seg_pull = touch_any * 0.5 * (prox_near + mid_near)                                           # (N,5)
    r_seg_pull = W_SEG_PULL * seg_pull.mean(dim=-1)

    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_palm_seat = r_palm_seat * table_decay
    r_oppose = r_oppose * table_decay
    r_seg_pull = r_seg_pull * table_decay   # decays on the table late in the episode, never in the air

    # =============================================================================================
    # LIFT -- 2.50 total, gated by `held` and uprightness only
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)
    r_lift = W_LIFT * held * upright * lift_shape
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, x off_table, keeps env_mult
    # =============================================================================================
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = (W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)
              * env_mult)
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10) * env_mult)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, x off_table, keeps env_mult
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded, no latch
    # =============================================================================================
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    set_band = torch.tanh(height / SET_LIFTOFF) * torch.exp(-height / SET_BAND)
    p_setdown = P_SETDOWN * set_band * torch.tanh(((-vz) - SET_DEAD).clamp(min=0.0) / SET_SCALE)
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    shove_raw = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                            + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    # shove relief (previous round, unchanged): seating the palm on a touched object is not shoving
    palm_closing = torch.maximum((1.0 - open_gate) * (0.5 + 0.5 * face), palm_c)
    shove_relief = SHOVE_RELIEF * any_digit * palm_closing
    p_shove = shove_raw * (1.0 - shove_relief)
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
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
        r_reach + r_open + r_palm_in + r_close
        + r_count + r_digit + r_palm + r_palm_seat + r_oppose + r_seg_pull
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_drift - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero

    components = {
        "reach": r_reach,
        "open_hand": r_open,
        "palm_in": r_palm_in,
        "close": r_close,
        "grasp_count": r_count,
        "grasp_digit": r_digit,
        "grasp_palm": r_palm,
        "grasp_palm_seat": r_palm_seat,
        "grasp_oppose": r_oppose,
        "grasp_seg_pull": r_seg_pull,
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
iter_02 trained only Hypothesis A (iter_02/selection.json), on both arms with the identical file (right = cup_grasp_r/iter_01).

Right arm (4,828 epochs): grasp 0.95-0.97, lift 0.92-0.95, far-start success 0.69 at tolerance 0.039. The grasp is now carried by finger segments: 3.5-3.6 digits touch with a segment, the index and middle fingers touch with their proximal links (0.66, 0.68), palm contact 0.66-0.71. Lift started at epochs 1,500-2,000, exactly when the segment-weighted digit count passed 2 (1.24 -> 2.02).

Left arm (4,342 epochs, stopped): grasp 0.85-0.90 from epoch 1,000 on, but lift 0.005-0.017 for 3,300 epochs; success 0; envelope 0.03. Only some digits touch: middle 0.56-0.66, pinky 0.43-0.48, thumb 0.34-0.40 (with its proximal link, which does not flex in this hand), ring 0.00 until epoch 2,000 then 0.30, index 0.03-0.07. Segment-weighted digit count stayed 1.13-1.40. Palm contact 0.39-0.57, palm gap 0.067-0.074 m. Hand actions saturated at +-1 in 0.82 of steps (right 0.73); bounds loss 27.6 (right 16.4).

Measured on hand-built states with the iter_02 code (object on the table, palm seated, middle and pinky touching with segments, thumb touching with its proximal link):
- closing the ring finger onto the object with a segment changes the per-step reward from +0.954 to +0.670 (-0.285); closing ring and index gives +0.705 (-0.250); lifting the object 3 cm instead gives +1.945 (+0.990).
- what falls when the third digit closes: reach 0.138 -> 0, palm seating 0.100 -> 0, closing 0.036 -> 0 (all multiplied by one minus the hold gate), pre-hold palm-seat bonus 0.140 -> 0.093, stall cost -0.029 -> -0.046; what rises: grasp ladder, digit and segment pull together +0.052.
- the previous round's code (iter_01) had the same drop, smaller: -0.140.
So on the table, completing the grasp costs reward unless the object is lifted in the same moment; the right arm found lifting first, the left arm did not and stayed with two to three digits.

To make the code more accurate and train better robot, the feedback for improvement is:
Change only what Hypothesis A says. Keep the segment-based digit counting, the segment pull, the pre-hold palm-seat bonus and the shove relief. The reward must stay identical for the mirrored arm.

## Hypothesis A
- The incomes that pay for getting to the object and seating the hand (the long-range reach, the palm seating and the closing term) must not disappear when the grasp completes on the table. Fade them with how far the object has actually risen above its start height (or with the object being off the table), not with the hold gate, so that on the table closing one more digit onto the object always raises the per-step reward. Today closing a third digit costs -0.285 per step.
- Neither the stall cost nor the pre-hold palm-seat bonus may turn a more complete grasp on the table into a loss: the stall cost must not grow with the number of digits holding the object, and the bonus must not shrink when more digits hold. Lifting must remain the only way to increase income clearly once the grasp is complete.

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
