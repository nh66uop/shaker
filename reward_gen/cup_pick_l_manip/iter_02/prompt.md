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

I want it to fulfil the following task: Stage 2 of a two-stage task, and the only stage this reward is for: from the pre-grasp pose beside the cup, close the hand around it, lift it, carry it to the goal and hold it there. A first policy, trained with a different reward, already brings the hand to the cup; this reward is applied to that policy's weights, so the episode still begins with the hand at the raised start and the first thing it does is travel to the cup. Do not undo that: keep paying enough for reaching the pre-grasp pose in the default open hand pose with the start orientation (ctx.approach_done, the gate the environment latches) that the policy keeps doing it, but that part is solved and must not be where the value sits.

Once ctx.approach_done is set, close the hand into an envelope grasp. The palm goes against the side of the cup and all five digits wrap the cup body: the thumb on the near side and the four fingers around the far side, each finger's links in contact with the cup. The environment sets ctx.envelope_done once the palm, the thumb and at least four digits have held the cup for several consecutive steps. Count the digits separately: a grasp with three digits in contact must be worth clearly less than one with five, and the reward for closing must keep rising as the fourth and the fifth digit arrive rather than saturating once a few links touch. In the previous attempt the ring finger never reached the cup while the thumb, middle and pinky did, so contact on the index and the ring finger has to pay as much as contact on the others. Reward the finger links for wrapping around the cup body along its surface, not for pressing towards its axis.

Then lift. With the envelope closed, raise the cup straight up off the table: pay for the height of the cup above the height it started at (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]), rising with that height, and for upward motion right after the envelope closes so that lifting is learned quickly. Holding the closed grasp on the table must not by itself be worth staying for, and moving the hand or the cup downward towards the table while holding it must be worth less than keeping it still and clearly less than raising it.

Then carry the cup to ctx.goal_pos and hold it there, upright and still, so the environment keeps counting successes: a success needs ctx.goal_dist <= ctx.success_tol for ctx.success_hold_steps consecutive steps with the fingers wrapped. Coming to rest at the goal must be worth more than lifting the cup repeatedly; putting the cup down after reaching the goal must not be a way to earn the lifting income again. You may add a bonus on ctx.success.

Throughout: keep the cup upright, do not drop it, do not knock it over, do not push the hand or the palm into the table, and lift smoothly — penalise fast arm motion and fast cup motion while the cup is held, but keep those penalties small enough that a slow lift is always better than no lift.
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

Re-imagine which steps is missed or wrong.
Show me the improved code as below:
