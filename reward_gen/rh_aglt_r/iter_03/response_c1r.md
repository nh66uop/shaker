brief 에 가설 id 줄이 없어 prompt 전체를 한 후보로 구현했다(iter_03 에는 가설 A/B 표기가 없고 필요한 변경은 하나다). Review items 3(a), 3(b)만 고쳤다. 나머지 항은 이전 응답과 같다.

1. 과제 단계: 먼 출발에서 편 손으로 접근 -> 손바닥을 컵 옆벽 1.5 r 에 수평으로 두고 엄지는 먼 쪽, 손가락은 가까운 쪽 -> 감싸 쥐기(첫마디까지) -> 곧게 들기 -> goal_pos 로 이송 -> 직립, 정지 상태로 유지하며 반복 성공. 이번 수정 범위: 엄지 끝이 벽을 지나쳐 넘어가는 파지와 손바닥 평면이 기운 파지는 품질 수입을 잃는다. 어떤 파지에나 주는 고정 기반 0.65 는 그대로 둔다.

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
    rad = max(float(ctx.cup_radius), 1e-3)

    # ---------------- palm vs. the measured grasp ring ----------------
    h_p, rv_p = _axial_radial(ctx.palm_pos, ctx.cup_pos, up)
    r_p = rv_p.norm(dim=-1)
    h_target = 0.15 * ctx.cup_mouth_z                   # grasp height: origin + 0.15 x mouth
    d_ring = torch.sqrt((r_p - 1.5 * rad) ** 2 + (h_p - h_target) ** 2 + 1e-12)

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

    # CHANGED (review 3b): palm-plane level factor for the held grasp quality.
    # s = combined out-of-plane component of palm normal and finger direction (vs the cup axis).
    # Full credit up to s = 0.2 (e.g. fingers ~10 deg down); 25 deg tilt of both axes
    # (s ~ 0.59) -> ~0.21; fingers pointing straight down (s >= 0.7) -> 0.
    s_tilt = torch.sqrt(n_ax ** 2 + f_ax ** 2 + 1e-12)
    plane_q = (1.0 - (s_tilt - 0.2) / 0.5).clamp(0.0, 1.0)

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
    # CHANGED (review 3a): the thumb TIP must be on the wall: its distance from the cup axis
    # close to ctx.cup_radius. Full credit within 0.5 r of the wall, 0 at 1.0 r off the wall
    # (a thumb tip at 2.2 r, past the wall, earns 0).
    r_thumb = rv_t[:, 0].norm(dim=-1)
    on_wall = (1.0 - ((r_thumb - rad).abs() - 0.5 * rad) / (0.5 * rad)).clamp(0.0, 1.0)
    oppose_q = oppose * below_rim * on_wall

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
    # fades to half in the last few cm (near_k), where the grasp should take over.
    orient_fade = 1.0 - 0.5 * near_k
    approach_orient = free * orient_fade * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    # no flipped grasp; 1 with the thumb >= 2 cm above the little finger, floor 0.25 below
    hand_up = 0.25 + 0.75 * thumb_up

    # ---------------- grasp (ceiling ~1.33, >= 2x the pre-grasp step) ----------------
    grasp_contact = 0.10 * upright * hand_up * wrap      # bridges 0 -> 1 N before the grasp flag
    # base 0.65 is paid for ANY grasp (flag on, cup upright) and is not shrunk by the quality
    # terms; quality is paid ON TOP: wrap 0.08, pose 0.10, opposition (thumb tip on the wall,
    # below the rim) 0.20, first-link envelope 0.20.
    # CHANGED (review 3b): the quality part is also scaled by the palm-plane level factor, so a
    # 25 deg tilted palm plane loses ~0.46 of ~0.58 quality (< 90 % of the envelope grasp).
    # The fixed base is untouched (sloppy first grasp vs parked unchanged).
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * oppose_q + 0.20 * envelope
    grasp_hold = held * upright * (0.65 + hand_up * plane_q * grasp_quality)

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

근거 메모: 가설 id 없음(brief 에 가설 id 줄이 없어 prompt 전체를 한 후보로 구현, 재시도 범위는 Review items 3(a)(b) 한정)

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| on_wall (oppose_q 에 곱함) | 엄지 끝이 2.2 r 로 벽을 지나친 파지가 인벨롭 대비 0.99x, 이를 90% 미만으로 만들어야 함. 엄지 끝의 축 거리가 cup_radius 근처일 때만 대립 품질을 줌. 2.2 r 이면 0 이 되어 약 0.20 이 빠지고, 1.305 에서 약 1.105 로 내려가 약 0.83x (추정) | prompt_c1_repair.md:590-592 |
| plane_q (grasp_quality 에 곱함) | 손바닥 평면 25도 기울임(법선 z 0.42, 손가락 z -0.42)이 0.99x, 10% 넘게 깎여야 함. s 가 약 0.59 이면 plane_q 약 0.21 이 되어 품질 약 0.46 이 빠지고 약 0.65x (추정). 손가락이 거의 수직 아래를 향한 자세(프레임 판독)도 품질 수입이 0 이 됨 | prompt_c1_repair.md:590-592, 77 |
| grasp_hold 기반 0.65 유지 | 엉성한 첫 파지가 주차 상태의 2배 이상이어야 함. 기반은 곱하지 않으므로 이전 비(약 0.81 / 0.37)를 그대로 둠. 엉성한 파지는 이미 oppose_q 가 약 0 이라 on_wall 영향 없음 | prompt_c1_repair.md:100, 396, 592 |
| s 불감대 0.2, on_wall 전액 구간 0.5 r | 측정 인벨롭 파지 기하(법선·손가락 수평, 손바닥 1.5 r)의 인벨롭 행 값을 낮추지 않아, 손끝만·림 위 엄지·20도 기운 컵·뒤집힌 손 행의 90% 판별을 유지하려는 여유폭 (추정) | prompt_c1_repair.md:77, 100, 592 |
| 그 밖의 모든 항 | 이전 응답 그대로. 접근 항, 접근 방향 항, 전도·낙하·운동 벌점 변경 없음 | prompt_c1_repair.md:398, 592 |

주의: 두 기준(엄지 끝 0.5 r, 불감대 s 0.2)은 인벨롭 원장 행의 엄지 끝 반지름과 손가락 기울기를 알지 못한 채 정했다. 인벨롭 행에서 엄지 끝이 1.5 r 을 넘거나 두 축이 약 11도 넘게 기울어 있으면 기준값이 낮아진다. 그러면 손끝만 파지 행의 비율(이전 약 0.85x)이 90% 기준에 가까워질 수 있으니 드라이런에서 확인해야 한다(가설).

prompt 파일: /home/user/rl_ws/hdgp/reward_gen/rh_aglt_r/iter_03/prompt_c1_repair.md
