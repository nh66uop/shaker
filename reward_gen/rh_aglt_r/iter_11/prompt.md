You are an expert in robotics, reinforcement learning and code generation.

We control one 7-DOF OpenArm robot arm (the right arm) carrying a five-finger Inspire RH56F1 hand, standing at a table (+x points from the robot toward the table, +z up, the table top is at ctx.table_z). The RH56F1 is under-actuated: six motors drive the thumb abduction (thumb_1), the thumb flexion (thumb_2) and one flexion joint per finger (index_1, middle_1, ring_1, pinky_1); the second link of each finger (and the thumb's last two links) follows its motor through a fixed mechanical coupling, so every finger curls as one unit. The fingers are short: the gap between thumb tip and finger tips is 105 mm fully open and 47 mm fully closed.

Every episode starts FAR from the object: the palm is just above the front edge of the table (0.08 m in x, 0.30 m to the side, 0.45 m high), 0.20-0.42 m (mean 0.30 m) from the cup, the whole hand above the top of the cup, the hand open. The palm normal (ctx.palm_axes[:, 0:3], the palm sensor's local +z) is horizontal and faces the body's centre line (+y for this right arm), the fingers point forward (+x) and 10 deg down, and the thumb is 7 cm above the little finger. The other arm stands still in the mirror-image pose. One cup (a 57 mm diameter cylinder, 0.114 m tall, 0.134 kg, friction 1.5-2.5, mass x0.8-1.25) stands upright on the table at a random position 0.10-0.40 m in x and 0.10-0.30 m to the side. The cup tips over when one side is pushed with only 0.3-0.4 N before something supports its other side.

The action space is a normalized `Box(-1, 1, (13,), float32)` with direct joint control. There is no grasp primitive:
    actions[0:7]  = arm joint INCREMENTS: each arm joint target moves by 0.05 * a rad per step, then passes a low-pass filter (factor 0.1), so each joint moves at most about 0.3 rad/s.
    actions[7:13] = hand motor targets in the order (thumb_1, thumb_2, index_1, middle_1, ring_1, pinky_1), each mapped linearly onto its open-to-grip interval: lo + 0.5 (a + 1)(hi - lo) with lo/hi = min/max(open, grip), so a = 0 is a HALF-closed hand (for thumb_1 a = +1 is the open side). The target is low-pass filtered (factor 0.1) and moves at most one full range per second. A finger whose links touch the cup with more than 1 N stops closing (it can still open), exactly like the real hand's force stop.

Now I want you to help me write a reward function for reinforcement learning.

The reward function receives a single argument `ctx`, an instance of this class (all positions env-local, metres; angles rad; forces N):

```python
@dataclass
class RewardContext:
    # ---- 상수 기하 ----
    table_z: float                 # 상판 윗면 z
    cup_radius: float              # 컵 바깥 반지름(가장 넓은 곳)
    cup_mouth_z: float             # 컵 원점 기준 입구 높이(+)
    cup_bottom_z: float            # 컵 원점 기준 바닥 높이(−)
    lift_latch_m: float            # 들기 래치 높이(스폰 대비)
    goal_tol: float                # 현재 성공 허용오차(키포인트 최대거리)
    goal_max: int                  # 에피소드 목표 수 상한
    # ---- 손·팔 ----
    palm_pos: torch.Tensor         # (N,3)
    palm_lin_vel: torch.Tensor     # (N,3)
    palm_axes: torch.Tensor        # (N,6) [법선, 손가락 길이]
    tips_pos: torch.Tensor         # (N,5,3)
    hand_closure: torch.Tensor     # (N,) 구동 관절 평균 폐쇄도 [0,1]
    hand_q_target_closure: torch.Tensor   # (N,6) 관절별 목표 폐쇄도 [0,1]
    finger_force: torch.Tensor     # (N,5) 손가락별 자기 컵 접촉력 합(첫마디 + 손끝)
    finger_prox_force: torch.Tensor   # (N,5) 첫마디(_1) 컵 접촉력
    finger_tip_force: torch.Tensor    # (N,5) 손끝(_sensor) 컵 접촉력
    palm_force: torch.Tensor       # (N,) 손바닥 패드 컵 접촉력
    hand_foreign_force: torch.Tensor   # (N,) 손이 컵 아닌 것(테이블·몸)에 닿는 힘
    grasped: torch.Tensor          # (N,) bool — 엄지 AND (다른 손가락 하나 이상) 컵 접촉 > 1 N
    hand_min_z: torch.Tensor       # (N,) 손 링크 원점 최저 z
    arm_q: torch.Tensor            # (N,7)
    arm_qd: torch.Tensor           # (N,7)
    arm_limit_margin: torch.Tensor # (N,7) 1 = 범위 중앙, 0 = 한계
    # ---- 컵 ----
    cup_pos: torch.Tensor          # (N,3) 원점
    cup_quat: torch.Tensor         # (N,4)
    cup_up: torch.Tensor           # (N,3) 컵 축(+z)
    cup_tilt: torch.Tensor         # (N,) rad
    cup_lin_vel: torch.Tensor      # (N,3)
    cup_ang_vel: torch.Tensor      # (N,3)
    cup_spawn_pos: torch.Tensor    # (N,3)
    lifted: torch.Tensor           # (N,) bool 래치 — 한 번이라도 lift_latch_m 넘게 들렸다
    # ---- 목표 ----
    goal_pos: torch.Tensor         # (N,3)
    goal_kp_dist: torch.Tensor     # (N,) 키포인트 최대거리(컵 ↔ 목표)
    near_goal: torch.Tensor        # (N,) bool kp_dist ≤ goal_tol
    goal_success: torch.Tensor     # (N,) bool 이 스텝에 목표 하나를 달성(쥔 채)
    goal_successes: torch.Tensor   # (N,) 이번 에피소드 달성 수(이 스텝 포함)
    # ---- 시간·액션 ----
    episode_progress: torch.Tensor # (N,) [0,1]
    actions: torch.Tensor          # (N,13) [팔 7, 손 6]
    prev_actions: torch.Tensor     # (N,13)
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N. Return a reward tensor of shape (N,) and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else. Do not keep any state between calls (no globals).
2. Write staged rewards with tensor masks (`torch.where`), not `if`/`else` on tensors.
3. Prefer bounded, smooth shaping (`torch.exp(-k * d)`, `1 - torch.tanh(k * d)`) so no single term dominates.
4. Finger order everywhere is (thumb, index, middle, ring, pinky). `finger_force` = `finger_prox_force` (the finger's first link) + `finger_tip_force` (the fingertip pad link), all measured against the cup only. `ctx.grasped` is the environment's grasp flag: the thumb AND at least one other finger touch the cup with more than 1 N.
5. `ctx.palm_axes[:, 0:3]` is the unit palm normal (pointing out of the palm, towards a held object) and `ctx.palm_axes[:, 3:6]` the unit direction along the fingers. Cylinder geometry: for a point p, `v = p - ctx.cup_pos`, axial coordinate `h = (v * ctx.cup_up).sum(-1)`, radial vector `v - h.unsqueeze(-1) * ctx.cup_up`; the side wall is at radius `ctx.cup_radius`, the mouth at `h = ctx.cup_mouth_z` and the bottom at `h = ctx.cup_bottom_z`.
6. Height: `ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]` is how far the cup has been raised (0 on the table). `ctx.lifted` becomes True once that exceeds `ctx.lift_latch_m` and stays True for the rest of the episode, even if the cup is dropped.
7. Goal and success are computed by the environment and cannot be redefined by the reward. The first goal of an episode is 0.10-0.18 m above the cup's start position and at most 5 cm away horizontally; after each success the next goal is up to 8 cm away from the previous one. `ctx.goal_kp_dist` is the largest distance between four points on the cup axis and the same points at the goal pose (0 = exactly at the goal, upright). A success is counted when the cup has spent `goal_success_steps` = 10 steps with `goal_kp_dist <= ctx.goal_tol` and, on that step, is near the goal AND `ctx.grasped`. `ctx.goal_tol` starts at 0.10 m and shrinks to 0.02 m as the policy improves. The episode ends after 5 successes.
8. An episode lasts at most 900 steps (15 s). It ends early when the cup falls off the table, leaves the work area, tilts more than 60 degrees, any hand link goes 5 mm below the table top (`ctx.hand_min_z`), an arm joint moves faster than 20 rad/s, the cup moves faster than 3 m/s or 30 rad/s, or a hand contact exceeds 200 N. Do not add penalty terms for these terminations.
9. Name components meaningfully (e.g. "approach", "lift", "success_bonus"); each one is logged during training and shown back to you after training.

I want it to fulfil the following task: reach a cup from far away, grasp it, lift it and carry it to the goal, and hold it there against disturbances, repeatedly. The policy starts from RANDOM WEIGHTS: this one continuous reward, with no stage switch and no checkpoint, has to teach the whole task - travel to the cup with an open hand, bring the palm beside its side wall with the thumb on the far side and the fingers on the near side, close the fingers around it, lift it straight up, carry it to ctx.goal_pos, and stay at rest there.

THE GRASP WE WANT (measured on this hand and this cup): the palm normal and the finger direction both horizontal; the palm about 1.5 cup radii from the cup axis (1.4 cm from the wall), at the height of the cup origin plus 0.15 x the mouth height; the hand LOW on the cup, well below the mouth (the cup will later be poured from, so the hand must leave the upper part free): thumb tip about 0.40 x the mouth height on the outer wall, the four fingertips at about 0.3, -0.1, -0.55 and -0.85 x the mouth height (index, middle, ring, little), ALL FIVE fingers touching the cup; the horizontal direction from the palm to the cup axis about 0.67 x palm normal + 0.73 x finger direction; the thumb on the opposite side of the cup from the fingers. The grasp we do NOT want: a vertical pinch from above (fingers pointing down), and a two-finger pinch - both slipped out when lifted.

Per-step ceilings:
- travelling to the cup with the hand open: about 0.6, fading to near zero once the cup is held.
- closing and keeping the grasp (more fingers on the cup is better, thumb opposing the fingers, palm oriented as above): about 1.2, and at least twice what the step just before the grasp earns.
- lifting the held cup: about 2.5, rising with height up to the height the goal needs (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).
- moving towards the goal: about 2.5, gated on the cup being off the table.
- at rest at the goal, upright, really held: about 10, the dominant income, gated on the cup being off the table.
- success: a large bonus for the first success of an episode; later successes worth little.

Keep the cup upright by multiplying the lift, transport and hold income by uprightness from ctx.cup_tilt. Knocking the cup over before it is held must cost a short-lived, bounded amount charged only while it is actually tipping (its tilt or angular speed rising), never a state penalty that keeps charging after the push. Losing a lifted cup must cost a bounded amount only while it is actually falling, never an amount proportional to `lifted`. Lowering a held cup must be worth less than keeping it still. Penalise fast arm and cup motion while holding, small enough that lifting and carrying always beat standing still.

Use no cup dimension, no scene coordinate, no joint index and no per-finger weight: use ctx.cup_radius, ctx.cup_mouth_z, ctx.table_z and ctx.goal_pos, and treat the four fingers alike. About 15 terms.
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {"component_name": component_tensor, ...}
```
3. Take care of tensor shapes and types; never access a field that is not listed in the class definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` and contains only the function (plus small helper functions if needed). Add short comments explaining the weights.

The deterministic check that will be run on your code evaluates these states of one episode and requires each row to earn MORE per step than the one before: start (far, open) < halfway (open) < just before the grasp (open, 4 cm from the grasp pose) < grasped on the table < lifted 5 cm < lifted to goal height < at the goal held. It also requires: a closed fist at the pre-grasp spot < the open hand there; an open hand that is tipping the cup (tilt 20 deg, spinning) < the open hand at the pre-grasp spot; a vertical two-finger pinch on the table < the horizontal grasp on the table; a cup dropped after lifting (falling) < grasped on the table; a resting cup that only jitters (0.094 rad/s, no tilt) within +-0.01 of the same state without jitter; and NEW: the same horizontal grasp on the table but touching with the fingertip pads only (first links 0 N), with the thumb tip over the cup's rim on the fingers' side, or with the held cup tilted 20 deg, must each earn less than 90 % of the envelope grasp on the table; and NEW: the halfway state with the palm turned up (normal z 0.8), the halfway state with the thumb tip below the little-finger tip, and the pre-grasp state with the palm facing away from the cup must each earn less than 90 % of the correctly oriented state, and the table grasp with the thumb tip below the little-finger tip less than 90 % of the upright-hand grasp. NEW: a sloppy first grasp (grasp flag on, only thumb and index tips at 1.2 N, first links 0 N, thumb on the fingers' side, hand half closed) must earn at least 2x the parked state (open hand at the pre-grasp spot, half closed). NEW (joint limits, ctx.arm_limit_margin 1 = mid-range, 0 = at the limit): the start with the real start margins (smallest 0.075) must earn at least 98 % of the start; the start with two joints at their limits and the halfway state with one joint at its limit must each earn less than 75 % of the same state with joint room; the halfway state must earn more at margin 0.02 than at 0.0 and more at 0.05 than at 0.02; the cup held at the goal with one joint at its limit, and with the measured good-carry margins (wrist 0.00), must each earn 70-97 % of the held-at-goal row. NEW (four fingers): the cup held at the goal with the ring and little fingers off (thumb, index, middle touching) must earn less than 70 % of the same state with all four fingers touching; with the ring and little fingertips touching lightly (0.5 N) at least 10 % more than with them off; and before the grasp (no grasp flag, no contact, hand half closed at the grasp spot) moving the little fingertip from 1.8 to 1.45 cup radii from the axis at -0.3 of the mouth height must raise the reward. NEW (iter_11, five fingers and a low grasp; the reference grasp rows now have the fingertips at the low heights above): held at the goal with the ring finger alone off (thumb, index, middle, little touching) must earn less than 80 % of all touching, on the table too; the ring fingertip touching lightly (0.5 N) at least 10 % more than off; held at the goal with the ring finger not touching, moving its tip from 1.8 to 1.25 cup radii from the axis must raise the reward by at least 3 %; before the grasp, the ring tip and the little tip at -0.85 of the mouth height moved from 1.8 to 1.45 cup radii must raise the reward; the same held and table grasps with the whole hand 0.26 x the mouth height higher (palm 0.41, the grasp the last policy learned) must earn less than 85 % of the low grasp. On real policy trajectories, moving the thumb tip 5 mm toward 1.3 cup radii at 0.40 x the mouth height before the grasp must raise the reward.

# Design brief (verified facts from the sibling tracks and this track; Korean)

# rh_aglt 보상 설계 브리프 (2026-09-29 판 — iter_00 전)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트 맨 앞에 통째로 들어간다**
(메모리 `reward-design-uses-full-history`: 직전 런만 패치하지 말고 전 이력을 종합한다). 좌우 트랙(`rh_aglt_r`·`rh_aglt_l`)이
이 한 파일을 쓴다(`reward_gen/rh_aglt_l/DESIGN_BRIEF.md` 는 이 파일로의 링크). 좌우 보상은 바이트 동일로 시작한다.

## 1. 과제와 고정 조건 (보상이 바꿀 수 없다)

- RH56F1 한 팔이 먼 출발(손바닥이 상판 앞 가장자리 위 (0.08, ∓0.30, 0.45), 손바닥이 반대 손 쪽·수평, 엄지가 새끼보다 위, 손은 모든 컵 윗면 위)에서 테이블 위 57 mm 컵(0.134 kg)에 다가가
  잡고, 들어서 `ctx.goal_pos` 로 옮기고, 거기서 쥔 채 유지한다. 목표는 에피소드당 최대 5개(달성마다 ±8 cm 이동).
- 성공(env): 키포인트 최대거리 ≤ `ctx.goal_tol` 인 스텝 10 누적 **그리고 그 스텝에 목표 근처에서 쥐고 있음**(엄지 AND 다른 손가락 > 1 N).
  첫 목표는 스폰보다 0.10~0.18 m 위라 들지 않고는 닿지 않는다. tol 0.10 → 0.02 커리큘럼.
- 액션 13 = [팔 관절 증분 7(0.05 rad/스텝 → EMA 0.1, 0.3 rad/s), 손 구동 6(편 자세~쥔 자세 구간 선형, a = 0 은 반쯤 쥔 손, 1 N 넘게 닿은 손가락은 닫기 정지)] — 09.29 2차 사용자 결정(cup_grasp 팔 · pour_bi_rh 손).
- 컵 소환 x 0.10–0.40 · y 폭 0.20 · 손·컵 마찰 1.5–2.5 · 컵 질량 ×0.8–1.25(ADR ×0.5–2.5).
- 종료(env, 벌점 항 두지 않는다): 컵 낙하·상자 이탈·60° 넘게 기울어짐 · 팔 qd > 20 · mimic 폭주 · 손이 상판 5 mm 아래 ·
  컵 > 3 m/s·30 rad/s · 손 접촉 > 200 N 2스텝.
- 제어 주기 1/60 s, 에피소드 900 스텝(hold 10 스텝은 보상 0).

## 2. 단계 원칙

접근 → 편 손으로 컵 옆 → 감싸 쥐기 → 들기 → 목표로 이송 → 유지. **한 단계 앞으로 가는 것이 그 자리에 머무는 것보다 항상 스텝당
이득**이어야 하고, 머물기만 해도 벌리는 자리(주차)가 없어야 한다. 파지를 하고 들려고 할 때 보상이 떨어지면 안 된다(pour_bi_rh 사용자 원칙 09-28).

## 3. 측정으로 확정된 사실 — 형제 트랙 두 개

### 3-1. cup_grasp (DG5F-M-S, 같은 구조의 먼 출발 무작위 컵 → 목표 이송, 처음부터 완주)
출처 `reward_gen/cup_grasp_r/DESIGN_BRIEF.md` 1~14, `CUP_GRASP_REWARD_LEDGER.json`, LOOP_STATE.
| # | 사실 |
|---|---|
| G1 | 처음부터 학습해 완주한 보상(r_iter_00): 접근 수입이 `(1−held)` 로 사라지고 `held` 뒤에 들기 2.5 · 이송 2.5 · 목표 유지 10 · 첫 성공 60 을 얹는 구조. 우 grasp 0.91 lift 0.89 성공 0.48(tol 0.023), 좌(l_iter_01) 성공 0.60(tol 0.031) |
| G2 | 먼 거리 접근은 **곱셈 인자 없는** 단순 손바닥 거리(0.35 × (0.6 far + 0.4 near))일 때 배웠다. 접근 수입을 줄이는 인자(옆벽 인자, 편 손 인자 바닥 0.15)를 곱하면 물러나거나 주차했다(손바닥 간격 0.22 → 0.37 m, 접촉 36 → 1.5) |
| G3 | 편 손 신호가 없으면 쥔 손으로 도착했다(파지 에피소드 0.008) |
| G4 | 손가락 접촉 없이 지불되는 편 손·손바닥 근접 수입은 벽에 평평한 손을 눌러 붙였다(+0.195/스텝, 손가락 접촉 0, 1,400 epoch) |
| G5 | 물체가 테이블 위에 있는 동안 목표형 수입이 나오면 테이블에 둔 채 머물렀다(0.54~0.69/스텝, 들기 0.007) |
| G6 | 에피소드 내내 서는 래치에 비례한 낙하 벌점이 가장 큰 항(−0.75/스텝)이 되어 들기를 잊었다. 떨어지는 동안만 물리면 붕괴가 없었다 |
| G7 | 하류 게이트가 4지에서 포화하면 4지 파지가 최적이었다(5번째 손가락 가치 < 하류 수입 1 %) |
| G8 | 하류 수입에 인벨롭 품질을 곱하면 우팔 인벨롭 0.86, 좌팔은 접촉력 ×2.6·팔 폭주 종료 5 → 42 %·파지 0.86 → 0.38 |
| G9 | 사용자 판정(09-27): 4번 관절 과굽힘 손끝 파지, 좌팔은 급하게 들어 올림. 파지 중 가운데 마디 접촉 0.00·손끝 패드 0.90~1.00 |
| G10 | 같은 보상이 거울 팔에서 다른 파지를 냈다(좌 손바닥 0.38 전체 파지, 우 손바닥 0.001 세 손가락 집기) |

### 3-2. pour_bi_rh (RH56F1, 같은 손·같은 컵·같은 제어)
출처 `reward_gen/pour_bi_rh/DESIGN_BRIEF.md` 3·4절, pour_fabric_mimic CLAUDE.md.
| # | 사실 |
|---|---|
| R1 | 컵은 한쪽 접촉 0.3~0.4 N 에 넘어진다(바닥 지름 3.8 cm 테이퍼) — 반대편이 받치기 전에 한 손가락이 밀면 쓰러진다 |
| R2 | 인벨롭 파지 기하(이 손·이 컵, 34k 스텝 중앙): 손바닥-컵 축 1.5 r(표면에서 1.4 cm), 높이 원점 +0.15 × 입구, 손바닥 법선·손가락 축 수평, 손바닥→컵 축 수평 방향 = 0.67 × 법선 + 0.73 × 손가락 축, 손바닥 접촉 0 |
| R3 | 관절 제어에서 손바닥 **속도 차분** 접근은 신호가 되지 않았다(f00: 접근 −0.001~−0.018, 손바닥-컵 21~43 cm 왕복). 접근·편 손은 상태형으로 |
| R4 | 닫기·파지 수입이 손 방향과 무관하면 위에서 세로 집기를 산다(f03: 손가락 축 z −0.95). 두 손가락 집기는 들 때 10곳 중 9곳 미끄러졌다(힘 2~3.5 → 0.3~0.8 N), 쥔 채 든 곳은 2.6 → 3.0 N |
| R5 | 파지 직전 상태 수입이 크면(파지/직전 1.38 배) 파지 판정 전에 머문다 |
| R6 | 상태형 밀림 벌점(`(1−lifted)·tanh(밀린 거리)`)은 한 번 민 컵에서 매 스텝 비용을 내 파지 시도를 포기하게 했다(i09 파지 0.11 → 0.02) |
| R7 | 상태형 "편 손" 가산이 컵 12 cm 앞 주차 자리를 만들었다(i08, 0.13~0.16/스텝) |
| R8 | 손 마찰 2.0 에서만 직립·양쪽 파지가 들기로 이어졌다(쌍 마찰 0.97: 1/73, 2.0: 25/86) — env 에 반영됨 |
| R9 | 관절 한계에 직접 강벌점을 주면 안 된다(i15, corr +0.90). 증분 팔은 한계 96 % 에 갇혔다(f03) — 이 트랙은 사용자 결정으로 증분 팔(cup_grasp 법칙) + bounds_loss 0.005 |

### 3-3. 이 트랙 실측
| # | 사실 |
|---|---|
| A1 | 서 있는 컵(손 접촉 없음, 반발 0)의 솔버 떨림: 각속도 p50 0.017 · p99 0.094 rad/s, 선속도 p50 0.6 mm/s, 기울기 < 0.04° (probe_rh_aglt_cup_rest 256 env). 반발 1.0 이면 0.10 rad/s·2.7 cm/s |
| A3 | 사용자 영상 판정(09.29, aglt_r_i00b ep1600 · aglt_l_i00b ep850): 두 팔 모두 컵까지 접근은 확실히 잘 한다. 왼팔은 컵을 들었지만 **인벨롭 파지가 아니다**. [관찰] 두 팔 모두 컵이 20° 이상 기울고(왼팔은 들면서 거의 수평), 엄지로 보이는 손가락이 컵 윗가장자리 쪽에 있다. 로그(쥔 스텝 근사): 닿은 손가락 3.1/3.8 개 중 첫마디까지 닿은 손가락 0.9/0.8 개, 손 폐쇄도 0.33/0.37, 전 스텝 평균 컵 기울기 12–13° |
| A2 | aglt_r_i00(반발 1.0, e0–202): `tip_penalty`(각속도 노름 기반) −0.115/스텝이 접촉 없이 첫 epoch 부터, `approach` +0.03 — 손바닥↔컵 0.36 → 0.42 m |

### 3-4. 워크스페이스 공통 원칙(메모리)
- 보상은 보편적이고 단순하게: 자산 고유 각도·치수 하드코딩 금지, 항 15개 내외.
- 곱 사다리는 앞 인자가 작으면 뒤 칸 기울기가 0 — 끈끈한 단계 지불의 합 + 게이트 없는 얕은 쉐이핑.
- 물체별 하드코딩 단계 래치 금지 — 가까우면 손가락을 닫는 연속 신호.
- 상태 비례 조밀 보상은 "자세 잡고 가만히 있기"를 최적해로 만든다 — 다음 단계 자리가 늘 더 비싸야 한다.
- 최솟값 기준 항은 동률이면 평평하다 — 평균 램프로.

## 4. 이 트랙이 앞 두 트랙과 다른 점 (보상 설계에 영향)

| 항목 | cup_grasp | pour_bi_rh | rh_aglt |
|---|---|---|---|
| 손 | DG5F 20 관절 | RH56F1 구동 6 | RH56F1 구동 6 (pour_bi_rh 와 같은 사상) |
| 시작 | 먼 출발 0.15–0.49 m | 근거리 16 cm | 먼 출발, 손이 컵 위 9 cm 이상 |
| 컵 | 8종 105–161 mm | 57 mm | 57 mm |
| 목표 | keypoint 5개 | 붓기 | keypoint 5개(cup_grasp 와 같은 구조) |
| 성공 파지 조건 | wrap_frac 커리큘럼 | 두 컵 파지 | env grasp 플래그(엄지 AND 다른 손가락 > 1 N) |
| ctx 표면 필드 | 링크별 표면 거리 | 없음 | 없음 — 컵이 원통 하나라 원점·축·반지름으로 계산 |

## 5. iter_00 요구

1. 뼈대는 G1 구조: 접근(상태형, 곱셈 인자 없음, `(1−held)` 로 소멸) → 파지 유지 → 들기(들린 높이, 목표 높이까지) →
   이송(테이블에서 떨어졌을 때만, 목표 거리) → 목표 유지(쥔 채·직립·정지, 가장 큰 수입) → 첫 성공 큰 보너스.
2. 편 손 접근: 손이 펴져 있을 때 접근 수입이 조금 더 크되 **곱셈으로 접근을 줄이지 않는다**(G2·G3·R7).
3. 파지 수입은 손 방향과 묶는다: 손바닥 법선(`palm_axes[:, 0:3]`)이 컵 축을 수평으로 향할 때(R2·R4). 세로 집기에는 작게.
4. 파지 판정 `ctx.grasped` 뒤 수입이 직전 상태 수입의 2배 이상(R5).
5. 컵 전도 방지는 **짧게 물리는** 벌점: 컵이 들리기 전 기울어지는 각속도·기울기 증가에만(R1·R6·G6). 상태형 밀림 벌점 금지.
6. 테이블 위 컵에는 목표형 수입 없음(G5). 들기 래치 비례 벌점 없음(G6).
7. 5개 손가락을 같게 다룬다: 손가락별 가중치·관절 인덱스·컵 치수 상수 금지(`ctx.cup_radius` 등 ctx 값만).
8. 팔·컵 빠른 움직임 벌점은 작게 — 들고 옮기는 것이 가만히 있는 것보다 늘 이득.
9. 항 15개 내외.


## The previous reward function (iter_10, selected c1)

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

## What happened when the policy was trained with it

iter_10 trained c1 (reward_gen/rh_aglt_r/iter_10/selection.json); no competing hypothesis. Command delay switched off (stage 1).

Runs (seed 42):
- aglt_r_i10 (right, fresh, 4096 envs, epochs 0-4600): goal successes rose with tolerance 0.0229 m at the end; no collapse.
- aglt_l_i10m (left, continued from the mirrored right epoch 3000, 2048 envs): ring contact rose to 0.52 of held steps by epoch 1800, but index fell to 0.39, little to 0.38 and the cup tilt grew to 18 deg median (p90 41 deg); the deterministic replay reached only 0.03 goal successes per episode at tolerance 0.02.

Deterministic replay of the right epoch 4600 (64 envs x 1800 steps, held steps, medians):
- Tolerance 0.0229: lifted in 1.00 of episodes, goal successes 2.30 per episode (0.88 at tolerance 0.02), no drops; the same weights mirrored onto the left arm give 2.02 / 0.83 with the same grasp.
- Grasp: palm 1.73-1.77 cup radii from the axis, palm at 0.41 of the mouth height (32 mm below the mouth, 82 mm above the bottom); fingertip heights / mouth height: thumb 0.64, index 0.55, middle 0.14, ring -0.30, little -0.59; thumb never inside the mouth; cup tilt 8.6 deg (p90 12.9).
- Contact > 0.5 N: thumb 1.00, index 0.99, middle 1.00, ring 0.00, little 0.94; median forces thumb 3.9, index 1.9, middle 1.4, ring 0.0, little 1.2 N; first links: thumb 1.00, index 0.99, middle 0.18, ring 0, little 0. Wrist joint 6 at its limit 0.15-0.18 of steps.
- The ring finger is commanded FULLY OPEN: its pre-clip policy mean is -1.37 (the action is clipped at -1), policy sigma 0.27, so almost no sample even starts closing it.
- Re-computing the iter_10 reward on these held-and-lifted steps: adding 1.2 N on the ring finger would raise the reward by 29 % (finger factor 0.875 -> 0.975, plus the min-finger part of the grasp-shape factor). But with the ring finger NOT touching, moving its tip from 1.8 to 1.25 cup radii (toward the wall) changes the held-at-goal reward by only +0.08 %: the only pre-contact pull is finger_wall (weight 0.10, a squared mean over four tips, logged at 0.007 per step) against a held income of about 12-16 per step. The income for the ring finger therefore appears only on contact, and the policy, whose ring finger is pinned open, never finds it.
- With the real-robot command delay switched on the same weights lift the cup in only 0.78-0.85 of episodes (handled in the separate stage 2, not part of this round).

User verdicts (10.01): "i10 is better, except that the ring finger alone does not touch"; "with iter11 I want all five fingers to take part"; "thinking of the work after pouring, it must grasp lower than now, below the cup mouth".

## Required changes for this iteration

Two changes, nothing else:

1. All five fingers on the cup (user: "all five fingers take part"). The ring finger is the one left out: its income appears only on contact, and before contact closing it does not raise the dominant held income (measured +0.08 % for moving its tip 1.8 -> 1.25 cup radii while the cup is held at the goal). Make each of the four fingers' score in the post-grasp finger factor CONTINUOUS from approach to contact: per finger, the larger of its outer-wall contact score tanh(F / 0.7 N) and a part (about 0.5-0.6) of its fingertip's closeness to the outer wall (the same per-tip outer-wall closeness used by finger_wall, with the four fingers' height window widened to about -1.0 to 0.6 of the mouth height so that the lower grasp of item 2 keeps the little finger inside it; tips inside the mouth count as far). Then combine the four finger scores so that ONE missing finger costs a lot (for example a mix of the mean and the minimum of the four scores, or the cube of the mean), with a floor that keeps a firm grasp with one finger missing clearly better than letting go. Treat the four fingers alike.
   Ledger rows (new): held at the goal with the ring finger alone off (thumb, index, middle and little touching) must earn less than 80 % of all touching (6Q, also on the table 3Q); the ring fingertip touching lightly (0.5 N) at least 10 % more than off (6Qt); held at the goal with the ring finger not touching, moving its tip from 1.8 to 1.25 cup radii must raise the reward by at least 3 % (6Rn vs 6Rf); before the grasp, the ring tip and the low little tip (-0.85 of the mouth height) moved 1 cm toward the outer wall must raise the reward (2Rn, 2Ln).

2. Grasp lower (user: "for the work after pouring it must grasp lower, below the cup mouth"). The approach already aims the palm at the cup origin + 0.15 x the mouth height, but the trained policy holds the palm at 0.41 x the mouth height (thumb tip 0.64), because nothing after the grasp depends on the palm height and the thumb target box is 0.45-0.8 of the mouth height. (a) Multiply the post-grasp income (grasp_hold, lift, transport, goal_hold) by a smooth grasp-height factor of the palm's axial height: 1 within about 0.1 of the mouth height around the target 0.15 x mouth, falling continuously above it so that the measured high grasp (palm 0.41) earns at most 85 % of the same state at 0.15; do not punish lower than the target more than mildly (the hand must not hit the table; the env already ends the episode if a hand link goes below the table). (b) Move the thumb target box and the thumb placement terms down to match: thumb tip about 0.25-0.6 of the mouth height (centre 0.40), still on the outer wall, never inside the mouth.
   Ledger rows (new): held at the goal with the whole hand 0.26 x mouth height higher (the measured i10 grasp) must earn less than 85 % of the low grasp (6H), the same on the table (3H). The ledger's reference grasp now has the fingertips at the low heights (thumb 0.38, index 0.29, middle -0.12, ring -0.56, little -0.85 of the mouth height), and the real-trace steering check moves the thumb toward 0.40 (was 0.62) of the mouth height.

All previous ledger rows (joint limits, four-finger rows, envelope and parking rows) and the real-trace checks must still pass. Keep every other part of the previous reward (including the joint-room factor); change only what these two items require.
