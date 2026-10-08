# 1. What the task means and which stages the robot must go through

## What is actually being asked

This is a **pure approach / pre-grasp stage**. The policy must move a 7-DOF arm carrying an open, frozen-shaped hand from a fixed raised start pose to a cup whose position is randomised in a 30 cm x 30 cm patch, place the hand in the exact pre-grasp configuration that the environment checks for `approach_done`, and then **stop and stay there for the rest of the episode**, having never touched the cup and without having moved it by a millimetre.

Everything that belongs to the later stage (closing the fingers, contact with the cup, lifting, `goal_dist`, `success`, `lifted`, `envelope_done`) is explicitly *not* rewarded here. Contact is not merely unrewarded, it is a failure mode: it must be penalised.

### The target pose, expressed exactly as the environment measures it

The environment sets `approach_done` on the first step on which all of the following hold at once. The reward should therefore shape *these same quantities*, not some proxy:

With `to_cup = cup_pos - palm_pos`,

| quantity | definition | env window | reward target |
|---|---|---|---|
| normal offset `n` | `to_cup · palm_normal` (palm to cup axis along the palm normal) | `n - cup_radius ∈ [-0.01, 0.02]` | `n = cup_radius + 0.005` |
| forward offset `f` | `to_cup · palm_finger_dir` (cup axis ahead of the palm, towards the fingers) | `f - cup_radius ∈ [-0.005, 0.02]` | `f = cup_radius + 0.0075` |
| axial offset `h` | `(palm_pos - cup_pos) · cup_axis` | `|h| ≤ cup_half_height` | `h = 0` (centre of the band) |
| orientation | `palm_normal ≈ -y`, `palm_finger_dir ≈ +x` | both within ~45 deg | both aligned, 40 deg used as the internal gate |
| hand shape | every movable joint within `0.3` of `hand_default_q_norm` | max deviation < 0.3 | deviation 0 (`hand_default_q_norm`) |
| contact | no finger/thumb link above 0.1 N on the cup | force < 0.1 N | force = 0 |

Because `palm_normal`, `palm_side`, `palm_finger_dir` are orthonormal and, in the nominal pose, `palm_normal = -ŷ`, `palm_finger_dir = +x̂`, `palm_side = -ẑ`, the triple `(n, f, h)` is an orthogonal decomposition of the palm-to-cup vector near the goal pose. So `d_pose = ||(e_n, e_f, e_h)||` is a genuine metric distance to the pre-grasp pose (about 0.15-0.49 m at reset, 0.31 m on average) and is the natural single shaping distance. Requiring `n > 0` with `palm_normal ≈ -y` automatically puts the palm on the cup's **+y side**, with the cup in front of the palm, further along the fingers than the thumb, which is exactly the geometry the task asks for.

### Stages the robot must go through

**Stage A — transit (roughly the first 1-3 s).** From the raised start pose beside the table, drive the palm towards the pre-grasp pose next to wherever the cup happens to be. During this phase `d_pose` falls from ~0.3 m to ~0.01 m. The hand shape must not change at all (the fingers are already in the default open pose, so the "hand" part of the task is a *do-nothing* requirement, but it is a requirement the policy can easily violate because it commands absolute finger targets every step). The wrist may have to turn by up to ~40 deg for far corners, so the orientation term must be a soft preference rather than a hard requirement.

**Stage B — fine placement.** Zero out the three offsets at the centimetre and then millimetre scale while keeping the fingers open and the palm off the table. This is where a single coarse `tanh` shaping term saturates, so a second, sharper shaping scale is needed to produce gradient below ~2 cm.

**Stage C — arrest and hold.** The moment the pose is reached the arm must come to rest and stay there until the 900-step episode ends. Two things follow from the specification:
- The hold reward must be a **constant per-step bonus** — no `episode_progress` factor, no decay, no one-shot bonus. Reaching the pose at step 150 and holding for 750 steps must pay 750 times the hold rate.
- The bonus must be gated on the pose being *currently* satisfied, not on the latched `approach_done` flag alone. A latched flag would pay the policy to touch the pose once and then wander off.

**Throughout — the four things that must not happen.** Contact with the cup (any of the 15 measured finger links or the palm); movement of the cup (displacement from `cup_spawn_pos`, tilt, linear/angular velocity); the palm or the fingers descending onto the table (`palm_clearance`, `hand_z_min - table_z`); thrashing (action rate, arm joint speed).

### Design decisions that follow

1. **No goal/lift/success term of any kind.** `goal_pos`, `goal_dist`, `success`, `num_successes`, `lifted`, `envelope_done` are never read.
2. **Contact is penalised, never gated into the positive terms in a way that would make the state unrecoverable.** The reach/orientation/hand terms stay available while touching so that the policy always has a gradient back out of a bad state, but the hold bonus is switched off entirely by any contact above 0.05 N (stricter than the env's 0.1 N).
3. **Action regularisation is asymmetric.** `actions[0:7]` are arm joint *increments*, so zero means "hold"; penalising their magnitude is meaningful. `actions[7:26]` are *absolute* finger targets, so zero means "middle of the range", i.e. half-closed. Penalising their magnitude would be actively wrong. Instead the finger channels are regularised by pulling `hand_target_norm` towards `hand_default_q_norm`, which is the correct "do nothing" command, and by a shared action-rate term.
4. **Reward stays net-positive in the normal holding state** (≈ +3.3 per step at the pose, ≈ +1.6 even with a knocked cup) so that the policy never prefers to end the episode early by dropping a finger through the table.
5. Penalties are bounded (`tanh` or clamped hinges) so no single term can dominate the shaping.

---

# 2. Reward function

```python
import torch
import math


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stage 1: reach the pre-grasp pose beside the cup, then hold it, without ever
    touching or moving the cup. Nothing about grasping, lifting or the goal is rewarded."""
    eps = 1e-8
    dtype = ctx.palm_pos.dtype
    device = ctx.palm_pos.device
    n_envs = ctx.palm_pos.shape[0]

    # movable finger joints only; the six "locked" joints (range <= 0.05 rad) are ignored,
    # exactly as the environment's approach_done check ignores them.
    movable = torch.as_tensor(
        [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 18], dtype=torch.long, device=device
    )

    # ---------------------------------------------------------------- pre-grasp geometry
    # Same three quantities the environment uses to latch approach_done. Near the goal pose
    # palm_normal / cup_axis / palm_finger_dir are mutually orthogonal, so (e_n, e_f, e_h)
    # is a proper orthogonal decomposition of the palm-to-pose error.
    to_cup = ctx.cup_pos - ctx.palm_pos                                   # (N,3)
    n_off = (to_cup * ctx.palm_normal).sum(-1)                            # palm -> cup axis
    f_off = (to_cup * ctx.palm_finger_dir).sum(-1)                        # cup ahead of palm
    h_off = ((ctx.palm_pos - ctx.cup_pos) * ctx.cup_axis).sum(-1)         # axial, band centre

    e_n = n_off - (ctx.cup_radius + 0.005)      # env window [r-0.01, r+0.02]  -> centre r+0.005
    e_f = f_off - (ctx.cup_radius + 0.0075)     # env window [r-0.005, r+0.02] -> centre r+0.0075
    e_h = h_off                                 # centre of the graspable band

    d_pose = torch.sqrt(e_n * e_n + e_f * e_f + e_h * e_h + eps)          # (N,)

    # Two scales: the coarse one carries the 0.15-0.49 m transit, the sharp one keeps a
    # gradient alive below ~2 cm where the coarse tanh has already saturated.
    r_reach = 0.6 * (1.0 - torch.tanh(3.0 * d_pose)) + 0.4 * (1.0 - torch.tanh(30.0 * d_pose))

    # ---------------------------------------------------------------- wrist orientation
    # Start orientation: palm_normal along -y (facing the cup), palm_finger_dir along +x.
    # Linear in the cosine, so a 40-45 deg turn for a far corner costs ~25-30 % of the term
    # instead of being forbidden.
    cos_n = -ctx.palm_normal[:, 1]
    cos_f = ctx.palm_finger_dir[:, 0]
    r_orient = 0.5 * torch.clamp(cos_n, 0.0, 1.0) + 0.5 * torch.clamp(cos_f, 0.0, 1.0)

    # ---------------------------------------------------------------- hand stays open
    # hand_q_norm / hand_default_q_norm are both "0 = straight, 1 = most flexed" for every
    # joint, so a plain absolute deviation is the right measure on this left hand too.
    dev_q = (ctx.hand_q_norm[:, movable] - ctx.hand_default_q_norm[:, movable]).abs()
    dev_q_mean = dev_q.mean(-1)
    dev_q_max = dev_q.amax(-1)
    r_hand_open = 1.0 - torch.tanh(4.0 * dev_q_mean)   # ~0.62 at 0.1 mean deviation

    # The finger channels are ABSOLUTE targets (a = 0 is half closed, not open), so the
    # correct action regulariser for them is "command the default pose", not "command zero".
    # This also stops a blocked finger from building up PD force against the cup.
    dev_cmd = (ctx.hand_target_norm[:, movable] - ctx.hand_default_q_norm[:, movable]).abs().mean(-1)
    p_hand_cmd = torch.tanh(4.0 * dev_cmd)

    # ---------------------------------------------------------------- nothing may touch the cup
    link_force_max = ctx.link_cup_force.reshape(n_envs, -1).amax(-1)
    force_max = torch.maximum(link_force_max, ctx.palm_cup_force)
    force_sum = ctx.link_cup_force.sum(dim=(1, 2)) + ctx.palm_cup_force
    # tanh keeps a force spike from blowing up the return; the 0.15 N scale sits just above
    # the env's 0.1 N "touching" threshold so a light graze already costs most of the term.
    p_contact = 0.7 * torch.tanh(force_max / 0.15) + 0.3 * torch.tanh(force_sum)

    # ---------------------------------------------------------------- cup must not move at all
    # Persistent (not one-shot): a cup that has been nudged keeps costing, so the policy that
    # never disturbs it beats the one that nudges it and recovers.
    cup_disp = torch.linalg.norm(ctx.cup_pos - ctx.cup_spawn_pos, dim=-1)
    cup_speed = torch.linalg.norm(ctx.cup_lin_vel, dim=-1)
    cup_spin = torch.linalg.norm(ctx.cup_ang_vel, dim=-1)
    p_cup = torch.clamp(
        0.5 * torch.tanh(cup_disp / 0.01)                       # slid / pushed (1 cm scale)
        + 0.3 * torch.tanh(ctx.cup_tilt / 0.05)                 # tipped (~3 deg scale)
        + 0.2 * torch.tanh((cup_speed + 0.1 * cup_spin) / 0.02),  # being knocked right now
        max=1.0,
    )

    # ---------------------------------------------------------------- keep off the table
    # Hinges: free above 2 cm of clearance, 1.0 at table level. The finger check mirrors the
    # episode-ending floor check (table_z - 0.005); palm_clearance is the measured palm gap.
    p_floor = torch.clamp((0.02 - (ctx.hand_z_min - ctx.table_z)) / 0.02, 0.0, 1.0)
    p_palm = torch.clamp((0.02 - ctx.palm_clearance) / 0.02, 0.0, 1.0)
    p_table = 0.5 * (p_floor + p_palm)

    # ---------------------------------------------------------------- "in the pose right now"
    # Tolerances deliberately tighter than the env's, so in_pose implies approach_done.
    cos40 = math.cos(math.radians(40.0))
    in_pose = (
        (e_n.abs() < 0.010)                              # env allows +/-0.015
        & (e_f.abs() < 0.009)                            # env allows +/-0.0125
        & (e_h.abs() < 0.8 * ctx.cup_half_height)        # inside the graspable band
        & (cos_n > cos40) & (cos_f > cos40)              # env allows ~45 deg
        & (dev_q_max < 0.25)                             # env allows 0.30
        & (force_max < 0.05)                             # env threshold is 0.10 N
    )
    # AND with the env's own latched flag: the bonus is paid only while the recognised
    # pre-grasp configuration is actually being held, never merely because it once happened.
    hold_gate = (in_pose & ctx.approach_done).to(dtype)

    # ---------------------------------------------------------------- stop and stay stopped
    arm_speed = torch.linalg.norm(ctx.arm_qd, dim=-1)
    hand_speed = torch.linalg.norm(ctx.hand_qd, dim=-1)
    r_still = 1.0 - torch.tanh((arm_speed + 0.5 * hand_speed) / 0.3)
    # Constant per step: no episode_progress, no decay, no one-shot term. Holding from step
    # 150 to step 900 pays 750x the per-step rate, which is the point of this stage.
    # The 0.5 floor pays for being in the pose at all; the rest pays for being still in it.
    r_hold = hold_gate * (0.5 + 0.5 * r_still)

    # ---------------------------------------------------------------- action regularisation
    # Arm channels are joint INCREMENTS, so zero = hold: penalising their magnitude is
    # meaningful and cheap. Finger channels are absolute targets and are handled by p_hand_cmd.
    p_arm_action = (ctx.actions[:, :7] ** 2).mean(-1)
    # Smoothness over all 26 channels. Clamped because prev_actions is zeros on the first
    # step after a reset, which would otherwise show up as a one-step spike.
    p_rate = torch.clamp(((ctx.actions - ctx.prev_actions) ** 2).mean(-1), max=1.0)

    # ---------------------------------------------------------------- weighted sum
    # Positive budget at the held pose: 1.0 + 0.4 + 0.4 + 1.5 = 3.3/step.
    # Worst realistic penalty (touching + knocked cup + palm down): ~ -3.2/step, and the
    # shaping terms stay available in that state, so the return never rewards ending the
    # episode early by dropping a hand link through the table.
    w_reach, w_orient, w_hand, w_hold = 1.0, 0.4, 0.4, 1.5
    w_contact, w_cup, w_table = 1.2, 1.5, 0.5
    w_hand_cmd, w_arm_action, w_rate = 0.2, 0.02, 0.05

    c_reach = w_reach * r_reach
    c_orient = w_orient * r_orient
    c_hand_open = w_hand * r_hand_open
    c_hold = w_hold * r_hold
    c_contact = -w_contact * p_contact
    c_cup = -w_cup * p_cup
    c_table = -w_table * p_table
    c_hand_cmd = -w_hand_cmd * p_hand_cmd
    c_arm_action = -w_arm_action * p_arm_action
    c_rate = -w_rate * p_rate

    reward = (
        c_reach + c_orient + c_hand_open + c_hold
        + c_contact + c_cup + c_table + c_hand_cmd + c_arm_action + c_rate
    ).to(dtype)

    components = {
        # weighted contributions (these sum exactly to reward)
        "approach": c_reach,
        "orient": c_orient,
        "hand_open": c_hand_open,
        "hold_still": c_hold,
        "contact_pen": c_contact,
        "cup_disturb_pen": c_cup,
        "table_pen": c_table,
        "hand_cmd_pen": c_hand_cmd,
        "arm_action_pen": c_arm_action,
        "action_rate_pen": c_rate,
        # unweighted diagnostics
        "diag_pose_dist": d_pose,
        "diag_offset_normal": e_n,
        "diag_offset_finger": e_f,
        "diag_offset_axial": e_h,
        "diag_hand_dev_max": dev_q_max,
        "diag_cup_disp": cup_disp,
        "diag_contact_force_max": force_max,
        "diag_in_pose": in_pose.to(dtype),
        "diag_arm_speed": arm_speed,
    }
    return reward, components
```
