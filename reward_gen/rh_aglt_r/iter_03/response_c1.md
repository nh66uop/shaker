**Task and stages.** The task has six stages, in order:
1. Travel from the far start (palm 0.20-0.42 m from the cup) with the hand open.
2. Bring the palm beside the side wall, about 1.5 r from the axis at origin + 0.15 x mouth height. Both palm axes are horizontal, the thumb is above the little finger, and the palm faces the cup.
3. Close the hand. The thumb goes on the far side and the four fingers on the near side, with the first links on the cup too.
4. Lift the cup straight up while keeping it upright.
5. Carry it to ctx.goal_pos.
6. Hold it there, still and gripped. Up to 5 goals, and only the first success pays a large bonus.

In iter_02 the grasp never appeared. The measured cause is the ratio between a sloppy first grasp (0.586) and parking beside the cup (0.394): 1.49x. The 0.14 approach_orient income is fully paid at the pre-grasp spot, while grasp_hold scales almost everything by hand_up and the quality terms. This iteration makes the one change the prompt asks for:
- **(a)** The grasp income gets a fixed base of 0.65 for any grasp (flag on, times uprightness). The quality terms are paid on top of it, and only they are multiplied by hand_up.
- **(b)** approach_orient fades in the last few centimetres (times 1 - 0.5 near_k).

```python
import torch
import math


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _axial_radial(p: torch.Tensor, cup_pos: torch.Tensor, up: torch.Tensor):
    # cylinder coordinates of point(s) p around the cup axis (broadcasts over a finger dim)
    v = p - cup_pos
    h = (v * up).sum(-1)
    rv = v - h.unsqueeze(-1) * up
    return h, rv


def _horizontal(v: torch.Tensor) -> torch.Tensor:
    # world-horizontal part of a (N,3) vector (z component removed)
    return torch.stack([v[:, 0], v[:, 1], torch.zeros_like(v[:, 0])], dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    up = _unit(ctx.cup_up)
    held = ctx.grasped.float()
    free = 1.0 - held                                   # approach income fades once held

    # ---------------- palm vs. the measured grasp ring ----------------
    h_p, rv_p = _axial_radial(ctx.palm_pos, ctx.cup_pos, up)
    r_p = rv_p.norm(dim=-1)
    h_target = 0.15 * ctx.cup_mouth_z                   # grasp height: origin + 0.15 x mouth
    d_ring = torch.sqrt((r_p - 1.5 * ctx.cup_radius) ** 2 + (h_p - h_target) ** 2 + 1e-12)

    far_k = 1.0 - torch.tanh(d_ring / 0.25)             # long-range pull from the far start
    near_k = 1.0 - torch.tanh(d_ring / 0.05)            # last few cm
    at_pose = torch.exp(-d_ring / 0.015)                # essentially at the grasp pose

    # ---------------- hand orientation quality (average ramp, not a min/product) ----------------
    n = _unit(ctx.palm_axes[:, 0:3])
    f = _unit(ctx.palm_axes[:, 3:6])
    n_ax = (n * up).sum(-1)
    f_ax = (f * up).sum(-1)
    horiz = ((1.0 - n_ax ** 2) * (1.0 - f_ax ** 2)).clamp(0.0, 1.0)   # both axes horizontal
    w = 0.67 * n + 0.73 * f                              # measured palm->axis direction
    w_h = _unit(w - (w * up).sum(-1, keepdim=True) * up)
    to_axis = _unit(-rv_p)
    align = 0.5 * (1.0 + (to_axis * w_h).sum(-1))
    pos_q = torch.exp(-d_ring / 0.02)
    pose_q = (horiz + align + pos_q) / 3.0               # vertical pinch from above scores low

    # thumb tip above the little-finger tip (world z).
    # 1 once the thumb tip is >= 2 cm above, 0 once it is 1 cm below ("a few centimetres").
    thumb_dz = ctx.tips_pos[:, 0, 2] - ctx.tips_pos[:, 4, 2]
    thumb_up = ((thumb_dz + 0.01) / 0.03).clamp(0.0, 1.0)

    # ---------------- contacts (all five fingers treated by the same rule) ----------------
    c = (ctx.finger_force / 1.0).clamp(0.0, 1.0)        # 1 N = env grasp threshold
    thumb_c = c[:, 0]
    fing_c = c[:, 1:].mean(-1)                           # four fingers alike, more is better
    wrap = thumb_c * fing_c                              # needs both sides -> no one-sided push income

    # first-link contact of the four fingers, treated alike.
    # 0.5 N counts a first link as touching (a finger's links share the ~1 N force stop).
    pc = (ctx.finger_prox_force / 0.5).clamp(0.0, 1.0)
    envelope = pc[:, 1:].mean(-1)                        # 0 = fingertip pads only, 1 = all first links on the cup

    h_t, rv_t = _axial_radial(ctx.tips_pos, ctx.cup_pos.unsqueeze(1), up.unsqueeze(1))
    u = _unit(rv_t)
    u_thumb = u[:, 0]
    u_fing = _unit(u[:, 1:].mean(1))
    oppose = 0.5 * (1.0 - (u_thumb * u_fing).sum(-1))   # 1 = thumb on the far side of the fingers
    # the thumb tip must be on the side wall, not over the rim.
    # ramps to 0 over the top 20 % of the mouth height (ctx value, no cup constant)
    mouth = max(float(ctx.cup_mouth_z), 1e-3)
    below_rim = ((mouth - h_t[:, 0]) / (0.2 * mouth)).clamp(0.0, 1.0)
    oppose_q = oppose * below_rim

    closure = ctx.hand_closure.clamp(0.0, 1.0)
    open_ = 1.0 - closure

    # uprightness (same factor as lift / transport / hold)
    upright = torch.cos(ctx.cup_tilt).clamp(min=0.0) ** 2

    # ---------------- travel (no multiplicative factor on approach) ----------------
    approach = free * 0.35 * (0.6 * far_k + 0.4 * near_k)
    open_hand = free * 0.08 * open_ * near_k             # only near the cup -> no far parking spot
    pre_pose = free * 0.07 * near_k * pose_q
    close_at_pose = free * 0.10 * closure * at_pose * pose_q   # closing is not a loss at the pose

    # approach orientation, ADDED (never multiplied onto approach, G2), not held only.
    # (a) palm normal perpendicular to gravity, (b) palm normal pointing at the cup horizontally,
    # (c) thumb tip above the little-finger tip. Ceiling 0.14 < the ~0.3 the approach pays for
    # closing the distance, so getting closer always pays more than turning the hand.
    level = 1.0 - n[:, 2].abs()                          # 1 = horizontal palm normal (0.2 at normal z 0.8)
    to_cup_h = _unit(_horizontal(ctx.cup_pos - ctx.palm_pos))
    face_cos = (_horizontal(n) * to_cup_h).sum(-1)       # >0 facing the cup, <0 facing away
    facing = 0.5 * (1.0 + torch.tanh(face_cos / 0.2))    # ~1 facing the cup, ~0 facing away
    # CHANGED (item 1): fades to half in the last few cm (near_k), where the grasp should take over.
    # Unchanged at halfway (near_k ~ 0); x0.83 at the 4 cm pre-grasp spot (0.14 -> ~0.12 per step).
    orient_fade = 1.0 - 0.5 * near_k
    approach_orient = free * orient_fade * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    # no flipped grasp; 1 with the thumb >= 2 cm above the little finger, floor 0.25 below
    hand_up = 0.25 + 0.75 * thumb_up

    # ---------------- grasp (ceiling ~1.33, >= 2x the pre-grasp step) ----------------
    # grasp income multiplied by cup uprightness (20 deg -> x0.88) and hand uprightness
    grasp_contact = 0.10 * upright * hand_up * wrap      # bridges 0 -> 1 N before the grasp flag
    # CHANGED (item 1): base 0.65 is paid for ANY grasp (flag on, cup upright) and is not shrunk
    # by the quality terms; quality is paid ON TOP and only the quality part is scaled by hand_up:
    # wrap 0.08, pose 0.10, opposition below the rim 0.20, first-link envelope 0.20.
    # Sloppy first grasp ~0.81 vs parked ~0.37 (>= 2x); envelope grasp ~1.33;
    # fingertip-only / thumb over the rim ~0.85x, thumb below the little finger ~0.62x of it.
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * oppose_q + 0.20 * envelope
    grasp_hold = held * upright * (0.65 + hand_up * grasp_quality)

    # ---------------- lift / transport / hold ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)               # cup off the table
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)

    # lowering a held cup lowers this state income -> worth less than keeping still
    lift = 2.5 * held * upright * (height.clamp(min=0.0) / goal_h).clamp(max=1.0)

    carry_gate = held * off * upright                    # no goal income while on the table
    kp = ctx.goal_kp_dist
    transport = 2.5 * carry_gate * (0.6 * torch.exp(-kp / 0.10) + 0.4 * torch.exp(-kp / 0.03))

    tol = max(float(ctx.goal_tol), 1e-3)
    prox = torch.exp(-kp / tol)                          # sharpens as the curriculum shrinks tol
    cup_v = ctx.cup_lin_vel.norm(dim=-1)
    still = torch.exp(-cup_v / 0.05)
    goal_hold = 10.0 * carry_gate * prox * (0.5 + 0.5 * still)   # dominant income

    first = (ctx.goal_success & (ctx.goal_successes <= 1)).float()
    later = (ctx.goal_success & (ctx.goal_successes > 1)).float()
    success_bonus = 60.0 * first + 3.0 * later

    # ---------------- short-lived, bounded penalties ----------------
    cup_w = ctx.cup_ang_vel.norm(dim=-1)
    # tipping: charged only when the cup is tilted beyond a few degrees AND its axis is actually
    # moving (faster than resting solver jitter), weighted up while the tilt is increasing.
    # A standing cup (tilt ~0) earns exactly 0 here regardless of its angular-velocity jitter.
    tilt = ctx.cup_tilt
    tilt_gate = torch.tanh((tilt - 0.05).clamp(min=0.0) / 0.1)        # ~3 deg deadband, ~1 by 20 deg
    axis_dot = torch.cross(ctx.cup_ang_vel, up, dim=-1)               # d(cup axis)/dt
    axis_rate = axis_dot.norm(dim=-1)
    axis_move = torch.tanh((axis_rate - 0.2).clamp(min=0.0) / 0.3)    # 0.2 rad/s: above resting jitter
    tilt_rate = -axis_dot[:, 2] / torch.sin(tilt).clamp(min=1e-3)     # d(tilt)/dt from d(up_z)/dt
    rising = torch.tanh(tilt_rate.clamp(min=0.0) / 0.3)
    tip_penalty = -0.6 * free * (1.0 - off) * tilt_gate * axis_move * (0.5 + 0.5 * rising)
    # charged only while a lifted, released cup is falling (not proportional to `lifted` itself)
    fall_speed = (-ctx.cup_lin_vel[:, 2]).clamp(min=0.0)
    drop_penalty = -1.0 * ctx.lifted.float() * free * torch.tanh(fall_speed / 0.5)
    # small: lifting / carrying must always beat standing still
    motion_penalty = -held * (0.10 * torch.tanh(cup_v / 0.5)
                              + 0.05 * torch.tanh(cup_w / 2.0)
                              + 0.10 * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0))
    action_rate = -0.01 * ((ctx.actions - ctx.prev_actions) ** 2).mean(-1)
    foreign_contact = -0.10 * torch.tanh(ctx.hand_foreign_force / 5.0)

    comps = {
        "approach": approach,
        "open_hand": open_hand,
        "pre_pose": pre_pose,
        "close_at_pose": close_at_pose,
        "approach_orient": approach_orient,
        "grasp_contact": grasp_contact,
        "grasp_hold": grasp_hold,
        "lift": lift,
        "transport": transport,
        "goal_hold": goal_hold,
        "success_bonus": success_bonus,
        "tip_penalty": tip_penalty,
        "drop_penalty": drop_penalty,
        "motion_penalty": motion_penalty,
        "action_rate": action_rate,
        "foreign_contact": foreign_contact,
    }
    reward = torch.zeros_like(approach)
    for v in comps.values():
        reward = reward + v
    return reward, comps
```

**근거 메모** (brief에 가설 줄이 없어 prompt 전체를 한 후보로 구현했다. 바꾼 것은 Required change 1 하나)

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| grasp_hold 기본값 0.65 (held x upright, hand_up·품질과 무관) | 어설픈 첫 파지 >= 주차 2배 요구. iter_02 는 1.49배에 그쳤고 파지가 나오지 않았다(iter_01 2.3배에서는 나왔다) | prompt.md:100, 386, 390, 396 |
| 품질 항 hand_up x (0.08 wrap + 0.10 pose_q + 0.20 oppose_q + 0.20 envelope), 기본값 위에 가산 | "quality terms ... paid ON TOP". 손끝만·림 위 엄지는 약 0.85배, 엄지 아래는 약 0.62배(모두 90 % 미만). 기울기 20도는 upright 로 0.88배 | prompt.md:100, 396 |
| approach_orient x (1 - 0.5 near_k) | "approach-orientation income fade in the last few centimetres". 중간 지점은 near_k 가 약 0 이라 그대로, 4 cm 지점은 0.83배. 주차 약 0.394 에서 0.371 로 준다. 반대를 향한 손은 올바른 자세의 약 0.86배 | prompt.md:388, 396 |
| 추정 수치 | 주차 약 0.371, 어설픈 첫 파지 약 0.81(약 2.2배), 인벨롭 파지 약 1.33, 파지 직전 약 0.385(파지가 2배 넘음). 원장 값 0.254, 0.394, 0.586 을 역산해 주차 지점 d_ring 약 0.04 m, 어설픈 파지 pose_q 약 1 로 잡았다 (추정) | prompt.md:390 |
| 그 밖의 모든 항 | iter_02 그대로 둠("Keep every other part") | prompt.md:199-376, 398 |

**확인이 필요한 곳:**
- 인벨롭 파지 상한이 약 1.33 이다. prompt 의 "about 1.2" 보다 조금 높다. 2배 조건과 90 % 판별 여유를 함께 맞추려고 받아들였다.
- 기울기 20도 행은 upright 인자 하나로 0.883배다. iter_02 와 같은 값이라 여유가 얇다.
- 반대를 향한 손 행은 페이드 때문에 약 0.84배에서 약 0.86배로 여유가 줄었다.
- 이 코드는 실행하지 않았다. 위 수치는 모두 손 계산이고, 드라이런 게이트에서 판정된다.

파일은 쓰지 않았다. 검사할 prompt 는 /home/user/rl_ws/hdgp/reward_gen/rh_aglt_r/iter_03/prompt.md 다.
