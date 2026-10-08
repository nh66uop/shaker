The brief had no hypothesis id line, so this candidate implements the whole prompt: the two required changes in `## Required changes for this iteration` (prompt.md:367-375). Everything else in iter_01 c1 is unchanged. Nothing was run; the ledger margins below are my own calculations from the ledger values and assumed test poses.

## Task and stages (prompt.md:75, 90)

The policy starts from random weights and has one continuous reward. It must learn these stages:
1. Travel from the far start with the hand open. The palm normal stays horizontal and faces the cup, and the thumb stays above the little finger.
2. Bring the palm beside the side wall of the cup (1.5 r from the axis, origin + 0.15 x mouth height), with the thumb on the far side and the fingers on the near side.
3. Close the hand around the cup. The four fingers are treated alike, first links touch, and the thumb is below the rim.
4. Lift the cup straight up to the goal height.
5. Carry it to `ctx.goal_pos`.
6. Hold it upright and still at the goal. The first success earns a large bonus.

Every step forward must pay more than staying where you are.

## Code

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

    # NEW (item 1 / item 2): thumb tip above the little-finger tip (world z).
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

    # NEW (item 1): approach orientation, ADDED (never multiplied onto approach, G2), not held only.
    # (a) palm normal perpendicular to gravity, (b) palm normal pointing at the cup horizontally,
    # (c) thumb tip above the little-finger tip. Ceiling 0.14 < the ~0.3 the approach pays for
    # closing the distance, so getting closer always pays more than turning the hand.
    level = 1.0 - n[:, 2].abs()                          # 1 = horizontal palm normal (0.2 at normal z 0.8)
    to_cup_h = _unit(_horizontal(ctx.cup_pos - ctx.palm_pos))
    face_cos = (_horizontal(n) * to_cup_h).sum(-1)       # >0 facing the cup, <0 facing away
    facing = 0.5 * (1.0 + torch.tanh(face_cos / 0.2))    # ~1 facing the cup, ~0 facing away
    approach_orient = free * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    # NEW (item 2): no flipped grasp; 1 with the thumb >= 2 cm above the little finger, floor 0.25 below
    hand_up = 0.25 + 0.75 * thumb_up

    # ---------------- grasp (ceiling ~1.25, >= 2x the pre-grasp step) ----------------
    # grasp income multiplied by cup uprightness (20 deg -> x0.88) and hand uprightness (item 2)
    grasp_contact = 0.10 * upright * hand_up * wrap      # bridges 0 -> 1 N before the grasp flag
    # base 0.30; wrap 0.15, pose 0.20, opposition (below rim) 0.25, first-link envelope 0.25
    grasp_hold = held * upright * hand_up * (0.30 + 0.15 * wrap + 0.20 * pose_q
                                             + 0.25 * oppose_q + 0.25 * envelope)

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

## 근거 메모

가설: 지정 없음 (brief 에 가설 id 줄이 없어 prompt 전체를 한 후보로 구현). 변경은 필수 변경 2개뿐이고 나머지는 iter_01 c1 과 같다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| approach_orient.level (0.04 x (1-\|n_z\|)) | 접근 중 우팔 \|n_z\| 0.52 로 상승 중. 절반 지점 손바닥 위(n_z 0.8) 0.0851 대 0.0852 | prompt.md:363, 365, 371 |
| approach_orient.facing (0.06, tanh(cos/0.2)) | 사전 파지에서 손바닥이 컵 반대쪽을 볼 때 0.264 대 0.267 | prompt.md:365, 371 |
| approach_orient.thumb_up (0.04) | 절반 지점에서 엄지가 새끼손가락 아래여도 같은 0.0852 | prompt.md:365, 371 |
| approach_orient 는 더하기만 하고 free 로 게이트, 상한 0.14 | 접근에 곱한 인자 때문에 물러났던 사례(G2). 가까워지는 이득이 손 돌리기보다 커야 함 | prompt.md:134, 371 |
| hand_up = 0.25 + 0.75 x thumb_up 을 grasp_contact, grasp_hold 에 곱함 | 테이블 파지에서 엄지가 새끼손가락 아래여도 같은 1.2425. 뒤집힌 파지는 바닥까지 낮춤 | prompt.md:365, 373 |
| 나머지 14항은 iter_01 c1 그대로 | "Keep every other part" | prompt.md:375 |

## 원장 여유 (직접 계산, 실행하지 않음)

- **절반 지점, 손바닥 위:** level 손실 0.032 이상. 필요한 값은 10 % 약 0.022.
- **절반 지점, 엄지 아래(dz<0):** thumb_up 이 0.33 이하라 손실 0.027 이상. 필요한 값은 약 0.022.
- **사전 파지, 반대쪽을 봄:** facing 손실 약 0.05. 필요한 값은 약 0.041. 가장 여유가 적은 항목이다. 사전 파지 자리에서 손바닥에서 컵 방향이 법선과 이루는 cos 이 0.2 이하로 작으면 넘지 못할 수 있다.
- **파지 대 직전 비율:** 약 1.24 / 0.41 = 3.0 배로, 2 배 이상 규칙을 유지한다.
- **뒤집힌 테이블 파지:** hand_up 이 0.5 이하라 90 % 보다 훨씬 낮다.

## 확인이 필요한 점

- **엄지 높이 가정 (가설):** 원장의 바른 인벨롭 파지에서 엄지 끝이 새끼손가락 끝보다 2 cm 이상 위에 있다고 가정했다. 그보다 낮으면 hand_up 이 1보다 작아져 파지 수입이 준다.
- **travel 상한 초과:** approach_orient 를 더해 travel 상한이 약 0.64~0.69 가 됐다. prompt 가 준 "about 0.6" 보다 조금 높다.

prompt 파일: /home/user/rl_ws/hdgp/reward_gen/rh_aglt_r/iter_02/prompt.md
