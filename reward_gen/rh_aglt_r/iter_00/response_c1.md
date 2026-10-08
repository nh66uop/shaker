가설 지정 없음: brief 에 가설 id 줄이 없어서 prompt 전체를 후보 하나로 구현했다. 파일은 쓰지 않았다. 호출자가 경로를 검사한 뒤 기록한다.

읽은 prompt: /home/user/rl_ws/hdgp/reward_gen/rh_aglt_r/iter_00/prompt.md (1~188행 전부)

## 과제 해석과 단계 (prompt 1번 요구)

1. 이동: 먼 출발점(손바닥이 약 0.33 m 떨어진 곳)에서 손을 편 채 컵 옆 파지 고리로 간다. 고리는 컵 축에서 1.5 r 떨어져 있고, 높이는 원점 + 0.15 x 입구 높이다.
2. 자세 잡기: 손바닥 법선과 손가락 축을 수평으로 두고, 손바닥에서 컵 축으로 가는 방향을 0.67 n + 0.73 f 에 맞춘다. 엄지는 반대편에 둔다.
3. 닫기: 엄지와 네 손가락을 함께 닫아 양쪽에서 1 N 넘게 누른다(`grasped`). 한쪽만 밀면 컵이 넘어진다(0.3~0.4 N).
4. 들기: 쥔 채 곧게 들어 목표 높이까지 올린다.
5. 이송: 테이블에서 떨어진 상태에서만 `goal_pos` 로 옮긴다.
6. 유지: 목표에서 쥔 채, 직립으로, 정지해 있는다. 가장 큰 수입이다. 달성할 때마다 다음 목표로 간다.

## 코드

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

    # ---------------- contacts (all five fingers treated by the same rule) ----------------
    c = (ctx.finger_force / 1.0).clamp(0.0, 1.0)        # 1 N = env grasp threshold
    thumb_c = c[:, 0]
    fing_c = c[:, 1:].mean(-1)                           # four fingers alike, more is better
    wrap = thumb_c * fing_c                              # needs both sides -> no one-sided push income

    _, rv_t = _axial_radial(ctx.tips_pos, ctx.cup_pos.unsqueeze(1), up.unsqueeze(1))
    u = _unit(rv_t)
    u_thumb = u[:, 0]
    u_fing = _unit(u[:, 1:].mean(1))
    oppose = 0.5 * (1.0 - (u_thumb * u_fing).sum(-1))   # 1 = thumb on the far side of the fingers

    closure = ctx.hand_closure.clamp(0.0, 1.0)
    open_ = 1.0 - closure

    # ---------------- travel (ceiling ~0.5, no multiplicative factor on approach) ----------------
    approach = free * 0.35 * (0.6 * far_k + 0.4 * near_k)
    open_hand = free * 0.08 * open_ * near_k             # only near the cup -> no far parking spot
    pre_pose = free * 0.07 * near_k * pose_q
    close_at_pose = free * 0.10 * closure * at_pose * pose_q   # closing is not a loss at the pose

    # ---------------- grasp (ceiling 1.2, >= 2x the pre-grasp step) ----------------
    grasp_contact = 0.10 * wrap                          # bridges 0 -> 1 N before the grasp flag
    grasp_hold = held * (0.5 + 0.2 * wrap + 0.25 * pose_q + 0.15 * oppose)

    # ---------------- lift / transport / hold ----------------
    latch = max(float(ctx.lift_latch_m), 1e-3)
    height = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    off = (height / latch).clamp(0.0, 1.0)               # cup off the table
    goal_h = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=latch)
    upright = torch.cos(ctx.cup_tilt).clamp(min=0.0) ** 2

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
    # charged only while the cup is actually rotating (tipping), before it is held, near the table
    tip_penalty = -0.6 * free * (1.0 - off) * torch.tanh(cup_w / 0.3) \
        * (0.5 + 0.5 * torch.tanh(ctx.cup_tilt / 0.2))
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

가설 지정 없음(brief 에 가설 id 줄 없음). prompt 전체를 한 후보로 구현했다.

규칙 해석 두 가지:
- "no joint index, no per-finger weight"(89행)와 엄지/손가락 구분(68, 77행)이 부딪친다. 엄지는 [0], 네 손가락은 [1:] 평균으로 같은 규칙에 넣었고 손가락별 가중치는 두지 않았다.
- 1.5, 0.15, 0.67, 0.73 은 prompt 가 준 무차원 측정 비율이다. 치수는 ctx.cup_radius 와 ctx.cup_mouth_z 에서 읽는다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| approach | 곱셈 인자 없는 단순 거리 0.35 x (0.6 far + 0.4 near), (1 - held) 로 사라짐 | prompt.md:80, 133(G2), 177 |
| open_hand | 편 손이면 접근 수입이 조금 더 큼, 근접 커널에만 붙여 주차 자리를 없앰 | prompt.md:179, 134(G3), 153(R7), 100(편 손 > 주먹) |
| pre_pose | 손바닥 법선·손가락 축 수평, 0.67n + 0.73f 방향, 1.5 r, 높이 +0.15 x 입구 | prompt.md:77, 148(R2) |
| close_at_pose | 파지 자세에서 닫기가 손해가 되지 않게 하는 연속 신호. 4 cm 앞 주먹은 작음 | prompt.md:160, 100 |
| grasp_contact | 엄지 AND 손가락이 0 -> 1 N 사이일 때 쥐기로 넘어가게 잇는 항. 한쪽만 밀면 0 | prompt.md:68, 147(R1) |
| grasp_hold | 쥐면 최대 1.2, 직전 스텝의 2배 이상. 손가락 수·엄지 대향·자세 반영, 세로 두 손가락 집기는 작음 | prompt.md:81, 150(R4), 151(R5), 180~181 |
| lift | 쥔 채 목표 높이까지 오를수록 증가, 최대 2.5, 직립 곱. 내리면 수입이 줄어듦 | prompt.md:82, 87 |
| transport | 테이블에서 떨어졌을 때만, goal_kp_dist 로 계산, 최대 2.5, 직립 곱 | prompt.md:83, 136(G5), 183 |
| goal_hold | 쥔 채·직립·정지·목표 근접, 최대 10(가장 큰 수입). goal_tol 커리큘럼을 따름 | prompt.md:84, 71 |
| success_bonus | 첫 성공 60, 이후 3 | prompt.md:85, 132(G1) |
| tip_penalty | 쥐기 전 컵이 회전하는 동안만, 상한 있음. 상태형 밀림 벌점이 아님 | prompt.md:87, 152(R6), 182 |
| drop_penalty | lifted 이고 놓친 컵이 떨어지는 동안만, 상한 있음 | prompt.md:87, 137(G6) |
| motion_penalty | 쥔 동안 팔·컵 속도에 작은 tanh 벌점 | prompt.md:87, 185 |
| action_rate | 액션 변화 평활화(상한 0.04) | (추정) |
| foreign_contact | 손이 테이블·몸에 닿는 힘에 작은 벌점(상한 0.1) | (추정) |

결정론 검사 순서(100행)는 손으로 계산해 봤다. 실행한 것은 아니다.
- 출발 약 0.03 < 중간 약 0.09 < 직전 약 0.28 < 테이블 위 파지 약 1.1~1.2 < 5 cm 들기(약 +0.9 들기 + 이송 + 유지) < 목표 높이 < 목표에서 쥔 상태.
- 파지 지점의 주먹 약 0.007 < 편 손 약 0.03.
- 넘어지는 컵(20도, 약 1 rad/s): 벌점이 약 -0.56 이라 직전 편 손보다 낮다. 회전이 0.3 rad/s 정도로 느리면 여유가 약 0.07 로 줄어든다.
- 세로 두 손가락 집기 약 0.77 < 수평 파지 약 1.18.
- 들었다 떨어뜨린 컵(낙하 중)은 0.5 미만으로 파지보다 낮다.

한계:
- "직전 스텝의 2배 이상"(81행)은 이상적인 수평 파지(약 1.2)일 때만 여유 있게 맞는다. 파지 직전 스텝에 접촉 수입이 이미 일부 있으면 비율이 2배 가까이까지 떨어진다.
- 5 cm 들기 행에서 이송·유지가 전액 나오려면 lift_latch_m 이 5 cm 이하여야 한다. 더 크면 그 두 항만 줄어든다. lift 항은 그대로라 순서는 유지된다.
