You are an expert in robotics, reinforcement learning and code generation.

We control a bimanual robot: two 7-DOF OpenArm arms, each carrying an Inspire RH56F1 five-finger \
hand, standing at a table. The RH56F1 is an UNDER-ACTUATED hand: only 6 joints per hand are driven \
(thumb abduction, thumb flexion, and one flexion drive for each of the index, middle, ring and pinky \
fingers); the remaining finger joints follow the driven ones through fixed mechanical couplings, so a \
finger always curls as a whole. The RIGHT arm is the "source" arm (src_*), the LEFT arm is the \
"receiver" arm (rcv_*). Two identical cups stand on the table: the source cup (in front of the right \
hand, filled with 20 small beads) and the receiver cup (in front of the left hand, empty). \
Positions are in metres in a frame whose origin is at the robot base; +z is up, +x is forward (away \
from the robot), +y is to the robot's left. The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (24,), float32)`:
  actions[0:6]   = source palm 6-DoF target offset (xyz + yaw/pitch/roll) — the arm is moved \
by a geometric-fabrics controller toward this target
  actions[6:12]  = source hand closure commands, one per driven joint \
(thumb abduction, thumb flexion, index, middle, ring, pinky; 0 = open, 1 = closed)
  actions[12:18] = receiver palm 6-DoF target offset
  actions[18:24] = receiver hand closure commands
The hand controller stops a finger automatically once one of its links touches its own cup \
(contact freeze); opening is always allowed. Fingers can only close when the palm is near its cup. \
The "cups" are slim shaker bodies about 6 cm in diameter and 11 cm tall, and the hand is small: a \
FINGERTIP (precision) grasp — thumb tip on one side, the tips of some fingers on the other — is the \
expected way to hold the cup; a full wrap-around power grasp is NOT required and usually not \
possible. The cups are light, so a finger brushing the cup easily knocks it over: approach slowly \
and touch it only with the fingertips.

Now I want you to help me write a reward function for reinforcement learning.

Typically, the reward function of a manipulation task consists of these parts (some are optional — include them only if really necessary):
1. the distance between the robot's hand and the target object (one term per arm)
2. the difference between the object's current state and its goal state
3. regularization of the robot's action
4. [optional] extra constraints on the target objects implied by the task (e.g. keep the receiver cup upright, do not spill)
5. [optional] extra constraints on the robot implied by the task

The reward function receives a single argument `ctx`, an instance of this class (all positions env-local, metres; angles rad; forces N):

```python
@dataclass(frozen=True)
class RewardContext:
    # ---- 상수 (에피소드 불변, python float) ---------------------------------------
    table_z: float            # 테이블 상면 높이 [m]
    cup_radius: float         # 컵 안쪽 반경 [m]
    cup_mouth_z: float        # 컵 원점(무게중심 부근) → 림(개구) 높이 [m]
    cup_bottom_z: float       # 컵 원점 → 바닥 높이 [m] (음수)
    num_beads: int            # 소스 컵에 든 비드 수

    # ---- 소스 팔 (비드가 든 컵을 잡아 붓는 팔) ------------------------------------
    src_palm_pos: torch.Tensor          # (N,3) 손바닥 위치
    src_palm_axes: torch.Tensor         # (N,6) 손바닥 회전행렬의 열 0(손바닥 법선)·열 1
    src_tips_pos: torch.Tensor          # (N,F,3) 손끝 위치 (F=손가락 수, 0번=엄지)
    src_hand_closure: torch.Tensor      # (N,) 실측 평균 손 폐쇄도 [0,1] (0=펴짐, 1=완전 파지)
    src_finger_force: torch.Tensor      # (N,F) 손가락별 소스 컵 접촉력 [N]
    src_palm_force: torch.Tensor        # (N,) 손바닥의 소스 컵 접촉력 [N]
    src_grasped: torch.Tensor           # (N,) bool 엄지 AND 다른 손가락이 동시에 소스 컵에 접촉
    src_arm_qd: torch.Tensor            # (N,A) 팔 관절 속도 [rad/s]
    src_arm_limit_margin: torch.Tensor  # (N,A) 팔 관절이 한계까지 남긴 여유 [0,1] (1=범위 중앙, 0=한계에 닿음)
    src_arm_qdd: torch.Tensor           # (N,A) 팔 관절 가속도 [rad/s^2] (흔들림·급변 신호)
    # ---- 리시버 팔 (빈 컵을 잡아 받는 팔) -------------------------------------------
    rcv_palm_pos: torch.Tensor          # (N,3)
    rcv_palm_axes: torch.Tensor         # (N,6)
    rcv_tips_pos: torch.Tensor          # (N,F,3)
    rcv_hand_closure: torch.Tensor      # (N,)
    rcv_finger_force: torch.Tensor      # (N,F) 리시버 컵 접촉력
    rcv_palm_force: torch.Tensor        # (N,)
    rcv_grasped: torch.Tensor           # (N,) bool
    rcv_arm_qd: torch.Tensor            # (N,A)
    rcv_arm_limit_margin: torch.Tensor  # (N,A) 리시버 팔의 관절 한계 여유 [0,1]
    rcv_arm_qdd: torch.Tensor           # (N,A) 리시버 팔의 관절 가속도 [rad/s^2]

    # ---- 소스 컵 (비드가 든 컵) ---------------------------------------------------
    src_cup_pos: torch.Tensor           # (N,3) 컵 원점 위치
    src_cup_quat: torch.Tensor          # (N,4) 자세 (w,x,y,z)
    src_cup_up: torch.Tensor            # (N,3) 컵의 위 방향 단위벡터 (직립이면 (0,0,1))
    src_cup_tilt: torch.Tensor          # (N,) 직립 대비 기울기 [rad] (0=직립, π/2=수평)
    src_cup_mouth_pos: torch.Tensor     # (N,3) 컵 개구(림) 중심 위치
    src_cup_lin_vel: torch.Tensor       # (N,3) [m/s]
    src_cup_ang_vel: torch.Tensor       # (N,3) [rad/s]
    src_cup_spawn_pos: torch.Tensor     # (N,3) 에피소드 시작 시 컵 위치(테이블 위)
    # ---- 리시버 컵 (빈 컵) --------------------------------------------------------
    rcv_cup_pos: torch.Tensor           # (N,3)
    rcv_cup_quat: torch.Tensor          # (N,4)
    rcv_cup_up: torch.Tensor            # (N,3)
    rcv_cup_tilt: torch.Tensor          # (N,)
    rcv_cup_mouth_pos: torch.Tensor     # (N,3)
    rcv_cup_lin_vel: torch.Tensor       # (N,3)
    rcv_cup_ang_vel: torch.Tensor       # (N,3)
    rcv_cup_spawn_pos: torch.Tensor     # (N,3)

    # ---- 비드 ----------------------------------------------------------------------
    bead_in_source_frac: torch.Tensor   # (N,) 소스 컵 안에 있는 비드 비율 [0,1]
    bead_in_target_frac: torch.Tensor   # (N,) 리시버 컵 안에 있고 소스 컵 밖인 비드 비율 [0,1] (소스 컵째 끼워 넣은 비드는 0)
    bead_spill_frac: torch.Tensor       # (N,) 어느 컵에도 없이 바닥/테이블로 떨어진 비율 [0,1]
    bead_centroid: torch.Tensor         # (N,3) 비드 무게중심 위치
    d_in_target: torch.Tensor           # (N,) 이번 스텝 bead_in_target_frac 증분 (Δ, 음수 가능)
    d_spill: torch.Tensor               # (N,) 이번 스텝 bead_spill_frac 증분

    # ---- 충돌 (실기 안전 — 09.14) ---------------------------------------------------------
    cup_cup_force: torch.Tensor         # (N,) 두 컵이 서로 부딪히는 접촉력 [N] (0 = 안 닿음)
    src_hand_foreign_force: torch.Tensor  # (N,) 소스 손이 **자기 컵 외**(상대 손·상대 컵·테이블)에 닿는 힘 [N]
    rcv_hand_foreign_force: torch.Tensor  # (N,) 리시버 손이 자기 컵 외에 닿는 힘 [N]

    # ---- 과제 판정 / 시간 ------------------------------------------------------------
    cups_nested: torch.Tensor           # (N,) bool 두 컵 원점 거리 < 9 cm — 소스 컵이 리시버 컵에 끼워져 있음(붓기가 아니라 성공 무효)
    success: torch.Tensor               # (N,) bool 성공 조건 충족 (env 가 판정, 보상이 바꿀 수 없음; cups_nested 면 항상 False)
    episode_progress: torch.Tensor      # (N,) 에피소드 진행도 [0,1]

    # ---- 액션 ------------------------------------------------------------------------
    actions: torch.Tensor               # (N,Dact) 이번 스텝 액션 [-1,1]
    prev_actions: torch.Tensor          # (N,Dact) 직전 스텝 액션
```

Additional knowledge:
1. Everything is BATCHED: every tensor field has a leading dimension N (number of parallel environments). The function must return a reward tensor of shape (N,) — never a Python float — and a dict of named component tensors, each of shape (N,). Use only `torch` and `math`; do not import anything else.
2. Write staged rewards with tensor masks, not `if`/`else` on tensors: `stage2 = torch.where(ctx.src_grasped, r_lift, torch.zeros_like(r_lift))`.
3. Prefer bounded, smooth shaping: `torch.exp(-k * dist)` or `1 - torch.tanh(k * dist)` instead of raw negative distances, so that no single term dominates.
4. `ctx.src_grasped` / `ctx.rcv_grasped` are True when the thumb AND another finger both press on that hand's own cup — this is the grasp-established signal. `ctx.*_hand_closure` is the measured closure (0 = open, 1 = fully closed).
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). The source cup holds a fixed number of beads in this track. The first bead leaves at about 70° with the cup full, and a full cup has lost a fifth of its beads by 80° and half by 86°. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — and the cups NOT nested). You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air.
8. Height above the table: `ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]` is how far the source cup has been lifted (0 while it rests on the table).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and shown back to you after training, so name them meaningfully (e.g. "approach_src", "grasp_rcv", "pour_delta").
11. This policy will be deployed on the real robot, so collisions are a safety hazard: `ctx.cup_cup_force` is the contact force between the two cups, and `ctx.src_hand_foreign_force` / `ctx.rcv_hand_foreign_force` are the forces each hand exerts on anything other than its own cup (the other hand, the other cup, the table). During a correct pour the cups do not touch each other and the hands touch only their own cup; penalise these forces (bounded, e.g. `torch.tanh(f / 5.0)`) so the policy learns to keep clearance. Cup poses seen by the policy are delayed and noisy as on the real perception system.

I want it to fulfil the following task: Using both arms, grasp each cup (a fingertip grasp is fine, no wrap-around needed): the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the mouth of the source cup over the mouth of the receiver cup, and tilt the source cup so that the beads pour into the receiver cup. Keep the receiver cup upright (tilted no more than 20 degrees), do not drop either cup, and spill as few beads as possible. The task is finished when at least 80% of the beads are inside the receiver cup, at most 25% have been spilled, both cups are still held and the receiver cup is upright, and this holds for 20 control steps.
1. Please think step by step and tell me what this task means and which stages the robot must go through.
2. Then write a function with exactly this signature:
```python
def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, {"component_name": component_tensor, ...}
```
   where `reward` has shape (N,) and every dict value has shape (N,).
3. Take care of tensor shapes and types; never call methods that are not listed in the class definition above.
4. Put the final code in ONE ```python fenced block that starts with `import torch` and `import math` and contains only the function (plus small helper functions if needed). Add short comments explaining weights, e.g. `# weight 0.5 works together with the pour term 5.0 — pouring must dominate once grasped`.

Observations from watching the trained policy:
# pour_bi_rh 보상 설계 브리프 (2026-09-26 판 — iter_00 재시작용, 이전 24 라운드 전체 종합)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트에 통째로 들어간다.**

이 트랙은 두 번 리셋했다. 1~12 라운드는 `archive_r1-12`, 13~24 라운드는 `archive_r13-24` 에 있다.
지금은 세 번째 사이클의 iter_00 이다. **패치가 아니라 다시 설계한다.** 이전 코드(iter_24, 약 900 줄)는 프롬프트에 넣지 않는다.

## 1. 과제와 고정 조건 (바꿀 수 없다)

- 오른팔(소스, `src_*`)이 비드 20 개가 든 컵을, 왼팔(리시버, `rcv_*`)이 빈 컵을 잡아 든다. 소스를 기울여 비드를 리시버 컵에 옮긴다.
- 성공(env 가 계산, 보상이 재정의 못 함): 담김 ≥ 0.80 · 흘림 ≤ 0.25 · 두 컵 파지 유지 · 받는 컵 ≤ 20° · 20 스텝 유지 · 컵이 포개지지 않음.
- 액션 24 = 소스 palm 6 · 소스 손 3(엄지 외전·엄지 굽힘·4지 굽힘) · 리시버 palm 6 · 리시버 손 3 (RH56F1 은 손당 3 채널).
  palm 델타 박스(앵커 기준, 도): 소스 yaw **±45** · pitch ±45 · roll **−150 ~ +45** / 리시버 ±30 전부.
  위치 박스: 소스 x ±0.15 · y −0.10~+0.40 · z −0.18~+0.25.
- 스폰(컵 원점, env-local): 소스 (0.382, −0.159, 0.268) · 리시버 (0.378, +0.165, 0.268) — 리시버가 소스의 **정왼쪽(+y)**.
- r = `ctx.cup_radius` = 0.028 m. 비드 지름 0.012 m. 소스 컵 원점 → 림 0.054 m. 제어 주기 1/60 s, 에피소드 900 스텝.

## 2. 측정으로 확정된 물리·기구학 제약

| # | 사실 | 근거 |
|---|---|---|
| K1 | 컵이 기우는 방향은 **손바닥 yaw 에 묶여 있다**. (기우는 방위 − palm yaw) 중앙 −4°, 표준편차 19°. 방향을 바꾸려면 palm yaw 를 돌려야 하고 명령 범위는 ±45°. | i18 ep800 trace |
| K2 | 60° 이상 기울 때 컵이 기우는 방위 중앙 **+68°**(IQR +59~+81) — 소스가 부을 수 있는 쪽은 **자기 앞-왼쪽**이다. 스폰 배치와 방향이 맞는다. | i18 ep800 trace |
| K3 | **높이가 없으면 명령한 회전이 안 난다.** 같은 회전 명령의 오차: palm z 0.29~0.42 → 143~151°, 0.53 → 38~52°, **0.69~0.76 → 10~23°**. 낮으면 어깨 j2 가 먼저 하한에 닿고 손목 j5·j6·j7 이 범위를 다 쓴다(j6 93.6 % · j7 98.8 % 대 j2 22.6 %). | 도달성 프로브 · i17g trace |
| K4 | 붓는 각 자체는 각도 수입 없이도 학습이 찾는다: i18 결정론 env별 최대 기울기 중앙 **87°**, 비드 방출 순간 **91°**. | i18 ep800 |
| K5 | 받는 컵 입구 반지름 0.028 m. 주둥이가 입구 위에 있다고 할 폭은 컵 반지름 규모다. | 자산 |
| K6 | 기울이면 주둥이가 `r·sinθ` 만큼 내려간다 — 높이 조건은 컵 중심이 아니라 **주둥이**로 본다. | 자산 |
| K7 | i24 후반 소스 손목 **j7 이 하한에 붙어 있는 비율 91 %(ep400) · 41 %(ep500)**, j2 하한 10~31 %. 같은 구간에서 소스가 기울 수 있는 방위는 +48° / +31° 로 좁아졌다. | i24 trace |
| K8 | 리시버 손은 5 지 접촉(1.00·1.00·0.98·0.95·0.92), 소스 손은 3 지(1.00·0.99·0.99·약지 0.28~0.56·새끼 0.00). 소스 새끼 손끝은 컵 축에서 4.2 cm(컵 벽 밖)이고 힘 0 이다. ep500 에서 60° 를 넘긴 env 2 개는 둘 다 소스 파지를 잃었다(표본 2). | i24 ep400·ep500 trace |

## 3. 라운드별 교훈 (무엇이 먹혔고 무엇이 실패했나)

| 라운드 | 변경 | 결과 | 교훈 |
|---|---|---|---|
| i08(1차) | 방향 부호가 있는 기울기 도입 | 방향이 잡혔다 | **방향 신호는 반드시 있어야 한다** |
| i09 | 기울기 연동 주둥이 높이 허용 | 첫 성공 | 기울이면 주둥이가 내려간다(K6)를 보상이 반영해야 한다 |
| i10~12 | 흘림 벌점 4 배·조준 벌점 강화 | 초반 정점 뒤 **단조 붕괴**(흘림/유출 0.18 → 0.43) | 조준 척도가 60 cm 까지 열려 멀리서 기울여도 수입이 났다 |
| i13 | 접근(차분)과 방출 허가(컵 반지름 창) 분리 | **붕괴 없음**, 결정론 담김 0.906 · 성공 0.480 · 효율 0.759(트랙 최고) | 접근은 차분, 허가는 좁은 창 — 이 골격은 유지한다 |
| i14 | 전량 이송(꼬리·비움) | 짧은 구간에서 담김 0.53 | 절반만 넘기고 끝나는 구조를 깼다 |
| i15 | **관절 한계 직접 벌점** | **실패**: 한계 접촉 0.21 → 0.43(벌점을 주는데 늘었다), 붓기만 감소. corr(한계 접촉, 담김) **+0.90** | 손목 한계는 결과지 원인이 아니다(K3). 직접 벌점으로 누르면 붓기가 먼저 죽는다 |
| i16w | 만남 통로 + 일반 웜스타트 | 리시버가 건너갔다 되돌아옴 3 회 | 웜스타트는 옛 자세를 통째로 물려받는다 |
| i16g | **파지 단계 ckpt** + 같은 보상 | 옛 습관 없이 만남 지점 하강(1.047 → 0.6), 결정론 87.6° | **파지 단계에서 시작하는 것이 옳다** |
| i17 | 각도 처방 전면 제거(방향 벌점까지 삭제) | 붓기 소멸(결정론 47°, 배달 0) | 결과 전용 보상은 **첫 한 알**이 나가야 경사가 생긴다 |
| i17g | 같은 보상, 파지 단계 ckpt | 자력으로 30° 까지 갔다가 22~26° 로 내려앉음 | 과들기 세금이 기울이기를 막았다 |
| i18 | 들기 허용 0.24 → 0.40 | 결정론 최대 87°·방출 91°(K4). 그러나 **방향 127° 어긋남**, 두 컵 높이 같음, 주둥이 수평 15.7 cm | 각도는 스스로 찾는다. 방향·배치·높이가 남는다 |
| i20 | 브리프 기반 전면 재설계(67 항 → 25 항) | 파지 0.69/0.74 → 0.011/0.013(100 epoch) | 진행형 항의 **비대칭 clamp**(−4,+1)가 탐색 잡음을 순손실로 만들었다 |
| i21 | clamp 대칭(−1,+1), 제어 벌점의 파지 게이트 제거, W_GRASP 1.5 | 파지는 살고 기울기 중앙 48° 에서 정지. 146 알 중 2 알 | 기울임에 대한 수입이 없으면 들고 서 있는다(주차) |
| i22 | tilt_open(창 안·방향 가중·차분) 신설 | **ep250 배달 13 알(트랙 최고)** → ep500/600/700 배달 1 / 2 / 0 | 조준·배치 항이 컵의 **축 회전**(몸체 점 속도)까지 지불해 배달과 무관한 회전을 키웠다 |
| i23 | 조준·배치를 차분형 퍼텐셜로 재작성 | 결정론 4 회 배달 0, 기울기 중앙 45.6 → 40.1° | 차분형은 주차는 막지만 **도달한 자리를 지킬 힘이 없다** |
| i24 | 스텝 캐시로 차분 정확화, 파지 값에 접촉 손가락 수 반영 | 결정론 ep300~600 배달 1/0/0/0, 기울기 46.8 → 39.4°, 방향 오차 26 → 66° | 옮기는 동안만 기울이고(2 s 에 33°, 방향 오차 5°) 끝나면 16° 로 세워 버틴다 |

### i24 에서 측정으로 **반증된** 가설 — 되살리지 말 것

1. "tilt_open 창이 좁아서 기울여도 돈이 안 된다" → 후반 자세에서 리시버 쪽으로 60° 기울였다고 계산하면 **지금의 1r~3r 창에서도 창 값 1.0** 이다. 창 폭은 제약이 아니다.
2. "기울이면 손가락이 떨어져 파지 값이 준다" → 같은 시각끼리 비교하면 엄지 외 접촉 1.61 대 1.87 로 거의 같다. 세운 뒤 손가락이 느는 것은 **시간 효과**다.

## 4. 지금 막고 있는 것 세 가지 (2026-09-26 사용자 판단 포함)

1. **주차 함정.** 상태에 비례하는 수입(파지·들기·운반·정렬·수렴)이 스텝당 12~15 였다. 900 스텝이면 약 13,000.
   반면 배달은 한 알 7.5 × 20 = 150. **아무것도 붓지 않고 자세만 잡아도 거의 전부를 번다.**
   ★사용자 지시(09-26): **회전을 통한 비드 이송이 더 큰 보상이어야 한다.**
2. **각도 딜레마.** 각도를 전부 지우면(i17) 첫 한 알 전까지 경사가 없어 붓기가 사라지고, 예전처럼 각도 사다리·대역·상한을 깔면 자산에 갇힌다.
   ★사용자 지시(09-26): **각도에 대한 것을 일부 보상으로 풀게 한다.** 즉 각도 관련 수입을 전면 금지하던 이전 원칙을 완화한다.
   다만 목표 각도·대역·상한을 처방하는 방식이 아니라, **비드가 실제로 나가기 시작하는 영역**(컵이 가득 찼을 때 첫 알이 약 70°,
   80° 에 1/5, 86° 에 절반이 빠진다)으로 가는 회전에 수입을 붙이는 방식이어야 한다. 그 각도에서 **방향과 주둥이 위치가 맞아야만** 지불한다.
3. **관절 한계 포화.** ★사용자 판단(09-26): 영상으로 보면 **배치 자체는 도달 가능**하다. 문제는 학습이 j7 같은 관절을 한계값까지 밀어
   쓰는 자세를 익힌다는 것이다(K7). i15 처럼 한계에 직접 강한 벌점을 걸면 붓기가 먼저 죽는다(corr +0.90).
   따라서 **한계에 기대지 않고도 같은 기울기가 나오는 자세**(K3: 손이 높을 때)를 보상이 더 싸게 만들어야 한다.

## 5. iter_00 설계 요구

### 5-1. 예산 규칙 (가장 중요)
- **회전으로 비드를 옮기는 수입이 나머지 전부의 합보다 커야 한다.** 성공 궤적에서 배달·회전 관련 수입이 스텝당 총수입의 **절반 이상**이 되게 한다.
- 상태 비례 수입(파지·들기·운반·정렬)은 **상한을 낮게** 두거나 배달이 시작되면(담김 > 0) 줄어들게 한다.
- 접근·배치처럼 위치를 맞추는 항은 차분(진행도) 형태로 주되, **도달한 자리를 지키는 힘이 필요한 항은 작은 상태형 유지 수입 또는 이탈 비용**을 둔다(i23·i24 교훈).
- 살아 있기(alive)·자세 유지만으로 에피소드 총수입의 큰 몫을 벌 수 없어야 한다.

### 5-2. 항의 역할 (겹치지 않게, 목표 25 항 이하)

| 단계 | 역할 |
|---|---|
| A 파지 | 양손 파지 성립·유지. 파지 단계 ckpt 가 이미 안다 — 작게 |
| B 들기 | 소스는 **높게**(K3), 리시버는 낮게. 과들기 세금은 충분히 높은 곳에서만 |
| C 배치 | 리시버 컵 입구를 소스 기준 **부을 수 있는 쪽**(K2: 앞-왼쪽)·소스보다 낮게·중앙선 근처에. 소스 기준 상대 좌표, 컵 반지름 배수 |
| D 조준 | 소스 주둥이를 받는 컵 입구 **위**(lip_dz > 0)·**수평으로 입구 안**(lip_xy 가 r 배수 창 안) |
| E 방향 | 기우는 방향이 받는 컵 쪽이어야 한다. 비용 = 기울기 크기 × 방향 어긋남 |
| F 회전·배달 | ★이 라운드의 주역. 창 안에서 방향이 맞을 때의 회전, 그리고 실제로 옮겨진 비드 |
| G 자세 여유 | 같은 기울기를 관절 한계에 기대지 않고 내는 자세(K3·K7). `ctx.src_arm_limit_margin` 은 팔별 관절 여유다 |
| H 제어·안전 | 속도·스핀·act_flip·palm 포화·컵끼리 접촉·남의 것 건드리기 |

### 5-3. 하지 말 것
- 목표 각도·대역·잔량 연동 방출각 사다리·각도 상한 벌점(overtilt·early_tilt) 같은 **각도 처방**. (각도 수입은 5-1·4-2 의 형태로만)
- 스폰 기준 만남 통로(리시버를 소스 너머로 끈다 — i18 실패).
- 관절 한계에 강한 직접 벌점(i15 실패).
- 새 절대 거리 상수. 조준·배치는 `ctx.cup_radius` 배수로 쓴다(높이 허용만 예외).
- 컵·손 자산 고유 치수 하드코딩.

## 6. 평가 (라운드 판정)

- 학습 곡선의 `task/*` 평균이 아니라 **`ep/*`(끝난 에피소드 기준)와 결정론 자동 평가**(100 epoch 마다 64 env·900 스텝)로 판정한다.
- 핵심 지표: `ep/toward_max`(받는 컵 쪽 최대 기울기) · `ep/dir_err_at_max` · `ep/lipdz_at_max` · `ep/in_target_max` · `ep/spill_end` ·
  소스 손목 여유 · 배달 비드 수.
- 성공 판정: 결정론 평가에서 담김 ≥ 0.8 · 최대 기울기 순간 방향 오차 < 30° · 주둥이가 입구보다 위.

## 7. 웜스타트

- **파지 단계 ckpt rh2_i07-r3 ep400**(`~/rl_ws/ckpt_in/pour_bi_rh/grasp_stage_i07r3_ep400.pth`, entropy ≈ 22, 양손 파지 0.71/0.75, 붓기 자세 없음).
  붓기 습관을 물려받지 않으면서 파지 학습 8~10 시간을 아낀다(i16g 에서 확인).
# iter_00 (세 번째 사이클) 운영 노트 — 2026-09-26

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.
앞의 DESIGN_BRIEF 가 이 트랙 24 라운드 전체의 측정·교훈이다. 이 노트는 **이번 라운드에 무엇을 다르게 하라는지**만 적는다.

## 왜 리셋하는가

iter_24 까지 오면서 파지(양손 1.00/0.99 유지), 들기, 기울이기(결정론 최대 87°)는 각각 성립했지만
비드 배달은 결정론 평가 기준 최고 13 알(1280 알 중, i22 ep250)이고 마지막 네 번은 1/0/0/0 이다.
보상이 24 라운드 동안 패치로 커져(iter_24 약 900 줄) 서로 싸우는 항이 쌓였다. 그래서 코드를 이어받지 않고 다시 쓴다.

## 이번 라운드의 세 가지 요구 (사용자 지시 2026-09-26)

### 1. 회전으로 비드를 옮기는 것이 가장 큰 수입이어야 한다
- 측정: i24 에서 소스를 기울인 채 운반하는 2 s 구간의 aim+place 지급이 스텝당 +1.2 였고, 운반이 끝나면 0 이 된다.
  반면 기울임 수입(tilt_open)은 에피소드 합 평균 0.44, 최대 6.1 이었다(상한 약 94). 배달은 4 라운드 연속 0~1 알이다.
- 요구: 성공 궤적에서 **회전·배달 수입이 스텝당 총수입의 절반 이상**이 되게 예산을 짠다.
  자세·파지·정렬만으로 버는 수입에는 상한을 두거나, 담김이 오르면 줄어들게 한다.

### 2. 각도에 대한 수입을 일부 허용한다
- 배경: iter_17 이 각도 처방을 전부 지웠을 때 붓기가 소멸했다(결정론 47°, 배달 0). 첫 한 알이 나가기 전에는 결과 항의 경사가 0 이다.
- 요구: 각도 관련 수입을 다시 쓴다. 단 **목표 각도·대역·상한을 처방하지 않는다.**
  허용하는 형태는 "비드가 실제로 빠져나가기 시작하는 회전"에 붙이는 수입이다
  (환경 설명: 컵이 가득 찼을 때 첫 알이 약 70°, 80° 에 1/5, 86° 에 절반이 빠진다).
  그리고 그 수입은 **주둥이가 받는 컵 입구 위 창 안에 있고 기우는 방향이 받는 컵 쪽일 때만** 지불한다
  (창 밖 회전은 흘림이고, i22 에서 방향과 무관한 축 회전에 돈이 나간 적이 있다).

### 3. 관절 한계에 기대지 않는 자세로 기울이게 한다
- 사용자 판단: 영상으로 보면 두 컵의 배치 자체는 도달 가능하다. 문제는 학습이 j7 같은 관절을 한계까지 밀어 쓰는 자세를 익힌다는 것이다.
- 측정: i24 후반 소스 j7 이 하한에 붙어 있는 비율 91 %(ep400) · 41 %(ep500), j2 하한 10~31 %.
  같은 구간에서 소스가 기울 수 있는 방위가 +48° / +31° 로 좁아졌다.
  도달성 프로브: 같은 회전 명령의 오차가 palm z 0.29~0.42 에서 143~151°, 0.69~0.76 에서 10~23° 다. **높이가 여유를 만든다.**
- 요구: `ctx.src_arm_limit_margin` 으로 **한계 여유를 유지하며 기울인 경우를 더 값싸게** 만든다.
  i15 처럼 한계 접촉에 강한 직접 벌점을 거는 방식은 금지한다(한계 접촉 0.21 → 0.43 으로 늘고 붓기만 줄었다. corr(한계, 담김) +0.90).

## 유지하는 것

- 접근은 차분(진행도), 방출 허가는 `ctx.cup_radius` 배수의 좁은 창(i13 에서 붕괴가 멈춘 골격).
- 방향 신호는 반드시 둔다(i17 에서 지웠다가 i18 이 127° 어긋난 쪽으로 부었다).
- 흘림·컵끼리 접촉·남의 것 건드리기·속도·명령 요동 벌점은 유지한다. 속도·제어 벌점을 풀어 효율을 얻지 않는다(사용자 원칙).
- 양팔 작업은 y ≈ 0 중심선 부근에서 한다.
- 진행형(차분) 항의 clamp 는 대칭으로 둔다(i20 의 비대칭 clamp 가 파지를 붕괴시켰다).

## 아는 위험 (설계할 때 피할 것)

- **곱 사다리**: 앞 인자가 0 이면 뒤 칸의 경사도 0 이 된다. 게이트를 곱으로 길게 쌓지 않는다.
- **차분형의 빈틈**: 목표 자리에 도달한 뒤 유지할 이유가 사라진다(i23·i24 의 배치 실패).
- **되돌림 펌프**: 차분 수입은 닫힌 경로에서 합이 0 이하가 되어야 한다.
- 소스 손은 3 지 파지다(새끼 접촉 0, 손끝이 컵 축에서 4.2 cm). 5 지 파지를 요구하는 항은 이번에 넣지 않는다 — 도달 가능성이 아직 확인되지 않았다.
