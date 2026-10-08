

### Step-by-Step Task Analysis & Stage Decomposition

The pouring task is a sequential manipulation problem that can be cleanly decomposed into six logical stages. The reward function must guide the policy through these stages without allowing it to exploit state-proportional rewards (e.g., holding cups high without pouring).

1. **Approach & Grasp** (Stage A): Both hands approach their cups and establish a stable fingertip grasp. The policy is rewarded for reducing fingertip-to-rim distance and applying gentle contact force. Once grasped, this term decays to avoid double-counting.
2. **Lift** (Stage B): Both cups are lifted off the table. The source cup requires sufficient height to enable pouring later. The receiver cup must be lifted to receive beads. This is gated by grasp.
3. **Placement & Alignment** (Stage C): The receiver cup mouth is positioned directly under the source cup mouth. The source cup is tilted toward the receiver. This stage is gated by grasp and lift, and rewarded as a difference term (progress toward alignment).
4. **Aim & Tilt Preparation** (Stage D): The source cup's lip is aligned over the receiver's opening. A small directional signal encourages tilting toward the receiver without imposing a hard angle target.
5. **Pouring (Outcome)** (Stage E): Beads flow from source to receiver. The policy is rewarded for *increments* of beads transferred (`d_in_target`) and penalized for *increments* spilled (`d_spill`). This term is heavily gated on grasp, lift, placement, and tilt to ensure it only pays during actual pouring.
6. **Control & Safety** (Stage F): Smooth actions, joint limit margins, and collision/foreign-contact penalties ensure safe, deployable behavior. A constant `alive` term and a `success_bonus` finalize the reward.

By gating outcome rewards on the full pouring state and keeping state-proportional terms bounded and small, the policy is forced to complete the pouring sequence rather than "holding still" at intermediate states.

---

```python
import torch
import math


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Staged reward for bimanual bead pouring.
    Returns (N,) reward tensor and dict of named component tensors.
    """
    dtype = ctx.src_cup_tilt.dtype
    device = ctx.src_cup_tilt.device
    N = ctx.src_cup_tilt.shape[0]

    # ------------------------------------------------------------------ Constants & Weights
    # Stage A: Grasp
    W_GRASP_APPROACH = 2.0   # Reward for approaching & making contact
    W_GRASP_HOLD = 0.5       # Small maintenance reward once grasped
    # Stage B: Lift
    W_LIFT_SRC = 1.5         # Source needs height for pouring
    W_LIFT_RCV = 1.0         # Receiver needs to be off-table
    # Stage C: Placement & Alignment
    W_PLACE = 3.0            # Horizontal alignment of cup mouths
    W_HEIGHT_DIFF = 2.0      # Source higher than receiver
    # Stage D: Tilt Direction
    W_TILT_DIR = 2.0         # Encourage tilt towards receiver (no hard target)
    # Stage E: Pouring Outcome (Incremental)
    W_POUR = 15.0            # Reward for beads entering receiver
    W_SPILL = 30.0           # Penalty for beads lost
    # Stage F: Control & Safety
    W_CTRL = 0.05            # Action magnitude smoothing
    W_COLLISION = 5.0        # Cup-cup contact force
    W_FOREIGN = 5.0          # Hand touching non-cup objects
    W_SUCCESS = 50.0         # Task completion bonus
    W_ALIVE = 1.0            # Constant survival floor

    # ------------------------------------------------------------------ Masks & Flags
    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    both_grasped = g_src * g_rcv

    # Heights above spawn
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2], min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2], min=0.0)

    # Mouth positions
    src_mouth = ctx.src_cup_mouth_pos
    rcv_mouth = ctx.rcv_cup_mouth_pos

    # ------------------------------------------------------------------ Stage A: Grasp
    # Approach: distance from fingertips to cup rim center (approximated by cup pos + mouth_z)
    src_rim = src_mouth.clone()
    src_rim[:, 2] = src_mouth[:, 2] + ctx.cup_mouth_z
    rcv_rim = rcv_mouth.clone()
    rcv_rim[:, 2] = rcv_mouth[:, 2] + ctx.cup_mouth_z

    src_approach_dist = torch.norm(ctx.src_tips_pos[:, 0] - src_rim.unsqueeze(1), dim=-1).min(dim=1).values
    rcv_approach_dist = torch.norm(ctx.rcv_tips_pos[:, 0] - rcv_rim.unsqueeze(1), dim=-1).min(dim=1).values

    # Staged: pay for approach while not grasped, small hold while grasped
    grasp_src = torch.where(g_src, W_GRASP_HOLD * torch.tanh(g_src), W_GRASP_APPROACH * torch.exp(-src_approach_dist / 0.03))
    grasp_rcv = torch.where(g_rcv, W_GRASP_HOLD * torch.tanh(g_rcv), W_GRASP_APPROACH * torch.exp(-rcv_approach_dist / 0.03))

    # ------------------------------------------------------------------ Stage B: Lift
    # Gated by grasp. Bounded smooth progress to target height.
    lift_src = torch.where(g_src, W_LIFT_SRC * torch.tanh(h_src / 0.15), torch.zeros_like(h_src))
    lift_rcv = torch.where(g_rcv, W_LIFT_RCV * torch.tanh(h_rcv / 0.15), torch.zeros_like(h_rcv))

    # ------------------------------------------------------------------ Stage C: Placement & Alignment
    # Horizontal distance between mouths
    dx = src_mouth[:, 0] - rcv_mouth[:, 0]
    dy = src_mouth[:, 1] - rcv_mouth[:, 1]
    place_dist = torch.sqrt(dx ** 2 + dy ** 2)
    
    # Vertical difference (positive = source higher)
    dz = src_mouth[:, 2] - rcv_mouth[:, 2]
    height_diff = torch.tanh(dz / 0.05)

    # Gated by both grasped. Decay if not lifting.
    place_gate = both_grasped * torch.tanh(h_src / 0.1) * torch.tanh(h_rcv / 0.1)
    place_reward = place_gate * W_PLACE * torch.exp(-place_dist / 0.04) * torch.clamp(height_diff, 0.0, 1.0)

    # ------------------------------------------------------------------ Stage D: Tilt Direction
    # Source up-axis horizontal component vs direction to receiver
    src_up = ctx.src_cup_up
    tilt_dir_h = src_up[:, :2]
    to_rcv_h = rcv_mouth[:, :2] - src_mouth[:, :2]
    to_rcv_h_norm = torch.norm(to_rcv_h, dim=1, keepdim=True) + 1e-6
    cos_align = (tilt_dir_h * (to_rcv_h / to_rcv_h_norm)).sum(dim=1)
    tilt_align = 0.5 * (1.0 + torch.clamp(cos_align, -1.0, 1.0))  # 1.0 when aligned

    # Gated by placement state. Encourage tilt magnitude without target.
    tilt_gate = place_gate * torch.exp(-place_dist / 0.06)
    tilt_reward = tilt_gate * W_TILT_DIR * torch.tanh(ctx.src_cup_tilt / 0.6) * tilt_align

    # ------------------------------------------------------------------ Stage E: Pouring Outcome (Incremental)
    d_in = ctx.d_in_target
    d_spill = ctx.d_spill
    
    # Strong gate: must be grasped, lifted, placed, and tilted towards receiver
    pour_gate = place_gate * tilt_align * torch.tanh(ctx.src_cup_tilt / 0.4)
    pour_delta = pour_gate * W_POUR * torch.clamp(d_in, max=0.5)
    spill_delta = pour_gate * (-W_SPILL * torch.clamp(d_spill, max=0.5))

    # ------------------------------------------------------------------ Stage F: Control & Safety
    ctrl = W_CTRL * (ctx.actions ** 2).mean(dim=1)
    collision = W_COLLISION * torch.tanh(ctx.cup_cup_force / 5.0)
    foreign_src = W_FOREIGN * torch.tanh(ctx.src_hand_foreign_force / 5.0)
    foreign_rcv = W_FOREIGN * torch.tanh(ctx.rcv_hand_foreign_force / 5.0)
    success_bonus = W_SUCCESS * ctx.success.to(dtype)
    alive = W_ALIVE * torch.ones(N, dtype=dtype, device=device)

    # ------------------------------------------------------------------ Compose Reward
    reward = (alive 
              + grasp_src + grasp_rcv 
              + lift_src + lift_rcv 
              + place_reward + tilt_reward 
              + pour_delta + spill_delta 
              - ctrl - collision - foreign_src - foreign_rcv 
              + success_bonus)

    # ------------------------------------------------------------------ Components Dict
    comps = {
        "alive": alive,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "place_rcv": place_reward,
        "tilt_toward": tilt_reward,
        "pour_delta": pour_delta,
        "spill_delta": spill_delta,
        "ctrl_smooth": -ctrl,
        "collision_penalty": -collision,
        "foreign_penalty_src": -foreign_src,
        "foreign_penalty_rcv": -foreign_rcv,
        "success_bonus": success_bonus
    }

    return reward, comps
```