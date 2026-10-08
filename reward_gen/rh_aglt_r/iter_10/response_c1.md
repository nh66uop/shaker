brief 에 가설 id 줄이 없어서 prompt 전체를 한 후보로 구현했다. 구현 범위는 prompt 의 "Required changes" 1번 하나다. 나머지는 iter_09 c1 그대로 두었다(joint-room 인자 포함).

Task understanding (prompt 75, 441-450): the hand travels open from the far start, reaches the side-wall grasp ring (palm at 1.5 r, thumb on the far side, fingers on the near side), closes into a horizontal envelope, lifts the cup, carries it to ctx.goal_pos and holds it there, up to 5 times. This round changes only the finger handling. In iter_09 the ring and little fingers were never used: right arm contact ring 0.02 / little 0.00, and three fingers earned 79 % of four. Two changes follow the sibling-track method.
(a) A per-fingertip outer-wall closeness term for the four fingers, combined as a squared mean. It pays before and while grasping, so each finger keeps its own pull toward the wall even when it is not touching.
(b) A convex finger factor 0.8 + 0.2 * s^3 multiplies post_f, which scales exactly grasp_hold, lift, transport and goal_hold. Here s is the mean four-finger tanh(F / 0.7 N) score, counting outer-wall contact only.

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
    mouth = max(float(ctx.cup_mouth_z), 1e-3)

    # ---------------- joint room (unchanged) ----------------
    m = ctx.arm_limit_margin.clamp(0.0, 1.0)
    room_j = (m / 0.05).clamp(0.0, 1.0)                 # (N,7)
    room = 0.5 * room_j.min(-1).values + 0.5 * room_j.mean(-1)
    room_pre = 0.5 + 0.5 * room
    room_post = 0.8 + 0.2 * room

    # ---------------- palm vs. the measured grasp ring (unchanged) ----------------
    h_p, rv_p = _axial_radial(ctx.palm_pos, ctx.cup_pos, up)
    r_p = rv_p.norm(dim=-1)
    h_target = 0.15 * ctx.cup_mouth_z                   # grasp height: origin + 0.15 x mouth
    d_ring = torch.sqrt((r_p - 1.5 * rad) ** 2 + (h_p - h_target) ** 2 + 1e-12)

    far_k = 1.0 - torch.tanh(d_ring / 0.25)             # long-range pull from the far start
    near_k = 1.0 - torch.tanh(d_ring / 0.05)            # last few cm
    at_pose = torch.exp(-d_ring / 0.015)                # essentially at the grasp pose

    # ---------------- hand orientation quality (unchanged) ----------------
    n = _unit(ctx.palm_axes[:, 0:3])
    f = _unit(ctx.palm_axes[:, 3:6])
    n_ax = (n * up).sum(-1)
    f_ax = (f * up).sum(-1)
    horiz = ((1.0 - n_ax ** 2) * (1.0 - f_ax ** 2)).clamp(0.0, 1.0)
    w = 0.67 * n + 0.73 * f                              # measured palm->axis direction
    w_h = _unit(w - (w * up).sum(-1, keepdim=True) * up)
    to_axis = _unit(-rv_p)
    align = 0.5 * (1.0 + (to_axis * w_h).sum(-1))
    pos_q = torch.exp(-d_ring / 0.02)
    pose_q = (horiz + align + pos_q) / 3.0

    s_tilt = torch.sqrt(n_ax ** 2 + f_ax ** 2 + 1e-12)
    plane_q = (1.0 - (s_tilt - 0.2) / 0.5).clamp(0.0, 1.0)
    plane_level = (1.0 - (s_tilt - 0.2) / 0.3).clamp(0.0, 1.0)
    level_pre = 0.5 + 0.5 * plane_level

    thumb_dz = ctx.tips_pos[:, 0, 2] - ctx.tips_pos[:, 4, 2]
    thumb_up = ((thumb_dz + 0.01) / 0.03).clamp(0.0, 1.0)

    # ---------------- fingertips in cup cylinder coordinates ----------------
    h_t, rv_t = _axial_radial(ctx.tips_pos, ctx.cup_pos.unsqueeze(1), up.unsqueeze(1))
    r_t = rv_t.norm(dim=-1)                              # (N,5)
    r_thumb = r_t[:, 0]
    h_thumb = h_t[:, 0]

    in_r = ((1.1 * rad - r_thumb) / (0.1 * rad)).clamp(0.0, 1.0)
    in_h = ((h_thumb - 0.4 * mouth) / (0.3 * mouth)).clamp(0.0, 1.0)
    thumb_inside = in_r * in_h
    thumb_out = 1.0 - thumb_inside

    # ---------------- NEW (iter_10): four fingers onto the OUTER wall ----------------
    # same "inside the mouth" rule as the thumb: radially inside the wall AND above 0.4 x mouth -> counts as far
    f_r = r_t[:, 1:]                                     # (N,4) index, middle, ring, little alike
    f_h = h_t[:, 1:]
    f_in_r = ((1.1 * rad - f_r) / (0.1 * rad)).clamp(0.0, 1.0)
    f_in_h = ((f_h - 0.4 * mouth) / (0.3 * mouth)).clamp(0.0, 1.0)
    f_outside = 1.0 - f_in_r * f_in_h
    # radial gap to the outer wall + distance outside the height window [-0.7, 0.8] x mouth
    gap_r = (f_r - rad).clamp(min=0.0)
    gap_h = (-0.7 * mouth - f_h).clamp(min=0.0) + (f_h - 0.8 * mouth).clamp(min=0.0)
    d_wall = torch.sqrt(gap_r ** 2 + gap_h ** 2 + 1e-12)
    wall_close = torch.exp(-d_wall / rad) * f_outside   # 1 = tip on the outer wall inside the window
    wall = (wall_close ** 2).mean(-1)                   # convex (squared) mean: each finger has its own gradient

    # post-grasp finger factor: cube of the mean four-finger outer-wall contact score
    f_score = (torch.tanh(ctx.finger_force[:, 1:].clamp(min=0.0) / 0.7) * f_outside).mean(-1)
    # floor 0.8: a firm 3-finger grasp keeps ~0.82 (still far above letting go); the 4th finger adds the most
    finger_f = 0.8 + 0.2 * f_score ** 3

    # ---------------- contacts (unchanged) ----------------
    c = (ctx.finger_force / 1.0).clamp(0.0, 1.0)
    thumb_c = c[:, 0] * thumb_out
    fing_c = c[:, 1:].mean(-1)
    wrap = thumb_c * fing_c

    pc = (ctx.finger_prox_force / 0.5).clamp(0.0, 1.0)
    envelope = pc[:, 1:].mean(-1)

    u = _unit(rv_t)
    u_thumb = u[:, 0]
    u_fing = _unit(u[:, 1:].mean(1))
    oppose = 0.5 * (1.0 - (u_thumb * u_fing).sum(-1))

    dr_box = (r_thumb - 1.45 * rad).clamp(min=0.0) + (1.15 * rad - r_thumb).clamp(min=0.0)
    dh_box = (h_thumb - 0.8 * mouth).clamp(min=0.0) + (0.45 * mouth - h_thumb).clamp(min=0.0)
    d_box = torch.sqrt(dr_box ** 2 + dh_box ** 2 + 1e-12)
    box_attract = torch.exp(-d_box / rad)

    outer_wall = ((r_thumb - 1.0 * rad) / (0.1 * rad)).clamp(0.0, 1.0)
    not_above = ((mouth - h_thumb) / (0.08 * mouth)).clamp(0.0, 1.0)
    not_far = ((1.8 * rad - r_thumb) / (0.35 * rad)).clamp(0.0, 1.0)
    thumb_place = outer_wall * not_above * not_far
    thumb_f = 0.25 + 0.75 * thumb_place

    deep = ((2.3 * rad - r_p) / (0.5 * rad)).clamp(0.0, 1.0)
    proj = (rv_t[:, 1:] * to_axis.unsqueeze(1)).sum(-1)
    past = (0.5 + proj / (0.5 * rad)).clamp(0.0, 1.0).mean(-1)
    fc = c[:, 1:]
    fingers_all = 0.5 * fc.mean(-1) + 0.5 * fc.min(-1).values
    parts = torch.stack([fingers_all, plane_level, past], dim=-1)
    blend = 0.5 * parts.mean(-1) + 0.5 * parts.min(-1).values
    shape_f = (0.3 + 0.7 * deep) * (0.6 + 0.4 * blend)

    inside_f = 1.0 - 0.5 * thumb_inside
    # scales grasp_hold, lift, transport, goal_hold; NEW: x finger_f
    post_f = shape_f * inside_f * thumb_f * finger_f

    closure = ctx.hand_closure.clamp(0.0, 1.0)
    open_ = 1.0 - closure

    upright = torch.cos(ctx.cup_tilt).clamp(min=0.0) ** 2

    # ---------------- travel (unchanged) ----------------
    approach = room_pre * free * 0.35 * (0.6 * far_k + 0.4 * level_pre * near_k)
    open_hand = room_pre * free * level_pre * 0.08 * open_ * near_k
    pre_pose = room_pre * free * level_pre * 0.07 * near_k * pose_q
    close_at_pose = room_pre * free * level_pre * 0.10 * closure * at_pose * pose_q

    level = 1.0 - n[:, 2].abs()
    to_cup_h = _unit(_horizontal(ctx.cup_pos - ctx.palm_pos))
    face_cos = (_horizontal(n) * to_cup_h).sum(-1)
    facing = 0.5 * (1.0 + torch.tanh(face_cos / 0.2))
    orient_fade = 1.0 - 0.5 * near_k
    approach_orient = room_pre * free * level_pre * orient_fade * (0.04 * level + 0.06 * facing + 0.04 * thumb_up)

    thumb_steer = room_pre * free * level_pre * 0.04 * near_k * box_attract * (0.4 + 0.6 * oppose)

    hand_up = 0.25 + 0.75 * thumb_up

    # NEW: fingertips onto the outer wall, before AND while grasping (not gated by free, so the
    # ring/little fingers keep a pull toward the wall while the cup is held); small (0.10) and
    # gated by the palm being at the grasp ring so it cannot create a parking spot away from the cup
    finger_wall = 0.10 * room_pre * level_pre * near_k * pose_q * hand_up * upright * wall

    # ---------------- grasp (unchanged form; post_f now includes finger_f) ----------------
    grasp_contact = 0.10 * upright * hand_up * wrap * thumb_f
    grasp_quality = 0.08 * wrap + 0.10 * pose_q + 0.20 * deep * past + 0.20 * envelope
    grasp_hold = room_post * held * upright * post_f * (1.05 + hand_up * plane_q * grasp_quality)

    # ---------------- lift / transport / hold (unchanged form) ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)

    lift = room_post * 2.5 * held * upright * post_f * (height.clamp(min=0.0) / goal_h).clamp(max=1.0)

    carry_gate = held * off * upright
    kp = ctx.goal_kp_dist
    transport = room_post * 2.5 * carry_gate * post_f * (0.6 * torch.exp(-kp / 0.10) + 0.4 * torch.exp(-kp / 0.03))

    tol = max(float(ctx.goal_tol), 1e-3)
    prox = torch.exp(-kp / tol)
    cup_v = ctx.cup_lin_vel.norm(dim=-1)
    still = torch.exp(-cup_v / 0.05)
    goal_hold = room_post * 10.0 * carry_gate * post_f * prox * (0.5 + 0.5 * still)   # dominant income

    first = (ctx.goal_success & (ctx.goal_successes <= 1)).float()
    later = (ctx.goal_success & (ctx.goal_successes > 1)).float()
    success_bonus = 60.0 * first + 3.0 * later

    # ---------------- short-lived, bounded penalties (unchanged) ----------------
    cup_w = ctx.cup_ang_vel.norm(dim=-1)
    tilt = ctx.cup_tilt
    tilt_gate = torch.tanh((tilt - 0.05).clamp(min=0.0) / 0.1)
    axis_dot = torch.cross(ctx.cup_ang_vel, up, dim=-1)
    axis_rate = axis_dot.norm(dim=-1)
    axis_move = torch.tanh((axis_rate - 0.2).clamp(min=0.0) / 0.3)
    tilt_rate = -axis_dot[:, 2] / torch.sin(tilt).clamp(min=1e-3)
    rising = torch.tanh(tilt_rate.clamp(min=0.0) / 0.3)
    tip_penalty = -0.6 * free * (1.0 - off) * tilt_gate * axis_move * (0.5 + 0.5 * rising)
    fall_speed = (-ctx.cup_lin_vel[:, 2]).clamp(min=0.0)
    drop_penalty = -1.0 * ctx.lifted.float() * free * torch.tanh(fall_speed / 0.5)
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
        "thumb_steer": thumb_steer,
        "finger_wall": finger_wall,
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
        "joint_room": room,          # diagnostic only (not summed)
        "finger_factor": finger_f,   # diagnostic only (not summed)
    }
    diag = ("joint_room", "finger_factor")
    reward = torch.zeros_like(approach)
    for k, v in comps.items():
        if k in diag:
            continue
        reward = reward + v
    return reward, comps
```

## 근거 메모

가설: 없음(brief 에 가설 id 줄 없음). prompt 의 변경 1 전체를 한 후보로 구현했다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| finger_wall(새 항, 가중치 0.10): 네 손가락 끝마다 바깥벽까지의 반경 간격 max(r_tip - r, 0)와 높이 창 [-0.7, 0.8] x 입구 밖 거리를 합친 거리를 exp(-d/r)로 바꾼다. 입구 안 판정은 엄지와 같은 규칙을 쓴다. 손가락별 값의 제곱 평균이다 | "at least the fingertips should touch". 우 ring 팁은 -0.34, little 팁은 -0.86 높이에 있고, 1.5~1.8 r, 명령 폐쇄 0이다. little 팁 1.8 r에서 1.45 r로 옮기면 보상이 올라야 한다 | prompt.md:434, 437, 446, 448, 100 |
| finger_wall 을 free 로 게이트하지 않음 | "Before and while grasping". 쥔 동안 닿지 않은 손가락은 cube 인자에서 기울기가 0이므로 벽 끌림이 따로 있어야 한다 | prompt.md:446 |
| finger_wall 의 near_k, pose_q, hand_up, upright 게이트 | 주차 자리 금지. 손바닥 반대, 엄지 아래, 기울어진 컵 행이 90 % 미만이어야 하므로 희석을 막는다 | prompt.md:100, 153(R7), 136(G4) |
| finger_f = 0.8 + 0.2 s^3 을 post_f 에 곱함(grasp_hold, lift, transport, goal_hold). s 는 네 손가락 tanh(F/0.7) 평균이고 바깥벽 접촉만 센다 | 세 손가락 상태가 네 손가락 대비 70 % 미만이어야 한다(iter_09 는 79 %). 0.5 N 가벼운 접촉은 10 % 이상 더 받아야 한다. 4번째 손가락이 가장 많이 더한다 | prompt.md:447, 448, 439 |
| 바닥값 0.8 선택 | 추산(추정): 세 손가락/네 손가락 약 0.66(< 0.70), 가벼운 접촉/없음 약 1.2 배. 엉성한 첫 파지는 약 0.8 배로 줄지만, 주차 상태 대비 2 배 이상을 유지한다고 봤다(약 2.1~2.2 배, 여유가 작다) | prompt.md:100, 447 |
| 나머지 항 전부 | iter_09 c1 그대로 | prompt.md:450 |

위험 메모(추정): 엉성한 첫 파지 >= 2 x 주차 행과 세 손가락 < 70 % 행은 바닥값 a 를 두고 서로 반대로 당긴다. a=0.8 에서 추산 여유는 각각 약 8 %와 약 5 %다. 게이트에서 둘 중 하나가 떨어지면 재시도는 a 를 0.78~0.82 안에서 조정하는 것으로 충분할 것으로 본다(가설).

prompt 파일: /home/user/rl_ws/hdgp/reward_gen/rh_aglt_r/iter_10/prompt.md