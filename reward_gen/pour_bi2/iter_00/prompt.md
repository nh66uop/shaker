You are an expert in robotics, reinforcement learning and code generation.

We control a bimanual robot: two 7-DOF OpenArm arms, each carrying a 20-DOF five-finger Tesollo DG-5F hand, standing at a table. The RIGHT arm is the "source" arm (src_*), the LEFT arm is the "receiver" arm (rcv_*). Two identical cups stand on the table: the source cup (in front of the right hand, filled with 20 small beads) and the receiver cup (in front of the left hand, empty). Positions are in metres in a frame whose origin is at the robot base; +z is up, +x is forward (away from the robot), +y is to the robot's left. The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, (18,), float32)`:
  actions[0:6]   = source palm 6-DoF target offset (xyz + yaw/pitch/roll) — the arm is moved by a geometric-fabrics controller toward this target
  actions[6]     = source thumb opposition command
  actions[7]     = source thumb closure command
  actions[8]     = source four-finger closure command (index, middle, ring and pinky close together; all flexion joints of a finger share this one command)
  actions[9:15]  = receiver palm 6-DoF target offset
  actions[15]    = receiver thumb opposition command
  actions[16]    = receiver thumb closure command
  actions[17]    = receiver four-finger closure command
Hand commands: -1 = open, +1 = fully closed. The hand controller stops a finger joint automatically once that finger link touches its own cup (contact freeze), so a closing command produces a wrap-around power grasp; opening is always allowed. Fingers can only close when the palm is near its cup. The environment low-pass filters the palm commands (exponential moving average) before they reach the arm controller, so switching a palm command between extremes from one step to the next barely moves the arm. `ctx.actions` / `ctx.prev_actions` are the raw policy outputs before that filter.

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
    adr_progress: float       # 난이도 커리큘럼(ADR) 진행도 [0,1]. 0 = 학습 초반(순간 성공률 0.3 을 아직 못 넘김), 성공률이 넘을 때마다 1/30 씩 오른다. 탐색을 죽이는 벌점(액션 부호 반전·변화율 가중)을 학습 초반에 끄고 과제를 배운 뒤 켜는 게이트로 쓴다

    # ---- 소스 팔 (비드가 든 컵을 잡아 붓는 팔) ------------------------------------
    src_palm_pos: torch.Tensor          # (N,3) 손바닥 위치
    src_palm_axes: torch.Tensor         # (N,6) 손바닥 회전행렬의 열 0(손바닥 법선)·열 1
    src_tips_pos: torch.Tensor          # (N,F,3) 손끝 위치 (F=손가락 수, 0번=엄지)
    src_hand_closure: torch.Tensor      # (N,) 실측 평균 손 폐쇄도 [0,1] (0=펴짐, 1=완전 파지)
    src_finger_force: torch.Tensor      # (N,F) 손가락별 소스 컵 접촉력 [N]
    src_palm_force: torch.Tensor        # (N,) 손바닥의 소스 컵 접촉력 [N]
    src_grasped: torch.Tensor           # (N,) bool 엄지 AND 다른 손가락이 동시에 소스 컵에 접촉
    src_arm_qd: torch.Tensor            # (N,A) 팔 관절 속도 [rad/s]
    # ---- 리시버 팔 (빈 컵을 잡아 받는 팔) -------------------------------------------
    rcv_palm_pos: torch.Tensor          # (N,3)
    rcv_palm_axes: torch.Tensor         # (N,6)
    rcv_tips_pos: torch.Tensor          # (N,F,3)
    rcv_hand_closure: torch.Tensor      # (N,)
    rcv_finger_force: torch.Tensor      # (N,F) 리시버 컵 접촉력
    rcv_palm_force: torch.Tensor        # (N,)
    rcv_grasped: torch.Tensor           # (N,) bool
    rcv_arm_qd: torch.Tensor            # (N,A)

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
    bead_fill_level: torch.Tensor       # (N,) 에피소드 시작 시 소스 컵이 부피로 얼마나 찼는지 [0,1] (정착 후 실측, 에피소드 동안 고정; 비드 양은 에피소드마다 다르다)

    # ---- 충돌 (실기 안전 — 09.14) ---------------------------------------------------------
    cup_cup_force: torch.Tensor         # (N,) 두 컵이 서로 부딪히는 접촉력 [N] (0 = 안 닿음)
    src_hand_foreign_force: torch.Tensor  # (N,) 소스 손이 **자기 컵 외**(상대 손·상대 컵·테이블)에 닿는 힘 [N]
    rcv_hand_foreign_force: torch.Tensor  # (N,) 리시버 손이 자기 컵 외에 닿는 힘 [N]

    # ---- 과제 판정 / 시간 ------------------------------------------------------------
    cups_nested: torch.Tensor           # (N,) bool 두 컵 원점 거리 < 9 cm — 소스 컵이 리시버 컵에 끼워져 있음(붓기가 아니라 성공 무효)
    premature_tilt: torch.Tensor        # (N,) bool 에피소드 래치 — **잡은** 소스 컵이 src_pour_lip_pos 가 리시버 입구 중심에서 xy 8.1 cm 보다 멀 때 premature_tilt_limit 을 넘은 적이 있음(성공 무효, 리셋 전까지 유지; 잡지 않은 채 넘어진 컵은 아님)
    src_pour_lip_pos: torch.Tensor      # (N,3) 소스 컵의 **붓는 쪽 림 점** — 비드가 나가는 지점. 방향 d̂(소스 컵 원점→리시버 입구, 수평)는 컵 자세와 무관해서 직립에서도 정의됨: 입구 중심 + r·(cos θ·d̂ − sin θ·ẑ), r = 4.1 cm, θ = src_tilt_toward_rcv
    src_tilt_toward_rcv: torch.Tensor   # (N,) 리시버 쪽으로 기운 부호 있는 각도 θ [rad] — (d̂, z) 수직면 성분만(옆으로 기운 성분은 0, 반대로 기울면 음수)
    premature_tilt_limit: torch.Tensor  # (N,) 조준 없이 허용되는 src_cup_tilt 상한 [rad] = 채움별 첫 유출각 − 20° (가득 52°, 채움 0.5 에서 69°); 에피소드 동안 일정
    cup_hit: torch.Tensor               # (N,) bool 에피소드 래치 — hold 이후 두 컵이 한 번이라도 부딪혔다(cup_cup_force > 0.5 N). 성공 무효
    pour_dir_xy: torch.Tensor           # (N,2) 붓는 방향 d̂(소스 컵 원점→리시버 입구, 수평 단위벡터). x>0 = 몸 바깥, 로봇 몸통은 −x. x > 0.3 이면 성공 무효
    rcv_side_margin: torch.Tensor       # (N,) 리시버 컵이 자기 팔 쪽으로 중심선에서 떨어진 거리 [m] (스폰 ≈ +0.16). < 0.03 이면(상대 팔 쪽으로 넘어감) 성공 무효
    src_wrap_count: torch.Tensor        # (N,) 소스 손에서 중간/원위 마디가 컵에 닿은 손가락 수 [0,5] — 팁만 닿은 손가락은 0(손끝 집기 ≠ 감싸 쥐기)
    rcv_wrap_count: torch.Tensor        # (N,) 리시버 손의 같은 값
    success: torch.Tensor               # (N,) bool 성공 조건 충족 (env 가 판정, 보상이 바꿀 수 없음; cups_nested·premature_tilt 면 항상 False)

    # ---- 목표량 붓기(09.23 사용자): 정해진 개수만 붓고 소스 컵을 다시 세운다(s2r 에서 여러 컵에 반복) ----
    pour_target_beads: torch.Tensor     # (N,) 이번 에피소드에 옮겨야 할 비드 개수(에피소드 고정). 관측에도 비율로 들어간다
    pour_target_frac: torch.Tensor      # (N,) 목표 개수 / 활성 개수 [0,1]
    bead_active_count: torch.Tensor     # (N,) 이번 에피소드의 활성 비드 개수 — 개수 = frac × 이 값

    # ---- 파지 기준(09.22): 처음 잡아 들어 올린 순간의 손-컵 관계를 에피소드마다 한 번 기록 ------------------
    #   기록 시점 = 그 손이 잡은 채 컵이 스폰보다 3 cm 이상 들렸고 컵 기울기 < 20° 이고 손 접촉력 합 > 0.5 N 인 첫 스텝.
    #   "컵 좌표" = 컵 원점 기준·컵 자세로 되돌린 좌표 — 손이 컵을 그대로 쥐고 있으면 컵이 아무리 돌아도 변하지 않는다.
    #   힘 비율 = [손바닥, 엄지, 검지, 중지, 약지, 소지] 접촉력 / 합 (자기 컵 접촉만, 합 1; 접촉이 없으면 0 벡터).
    src_palm_in_cup: torch.Tensor       # (N,3) 지금 손바닥 위치, 소스 컵 좌표 [m]
    src_tips_in_cup: torch.Tensor       # (N,F,3) 지금 손끝 위치, 소스 컵 좌표 [m] (0번=엄지)
    src_force_share: torch.Tensor       # (N,F+1) 지금 소스 손 힘 비율
    src_force_total: torch.Tensor       # (N,) 지금 소스 손 자기 컵 접촉력 합 [N]
    src_grasp_ref_valid: torch.Tensor   # (N,) bool 이번 에피소드에 소스 손 기준이 기록됐는가 (들어 올리기 전에는 False)
    src_ref_palm_in_cup: torch.Tensor   # (N,3) 기준 시점 손바닥 위치, 컵 좌표 (valid 가 False 면 0)
    src_ref_tips_in_cup: torch.Tensor   # (N,F,3) 기준 시점 손끝 위치, 컵 좌표
    src_ref_force_share: torch.Tensor   # (N,F+1) 기준 시점 힘 비율
    src_ref_force_total: torch.Tensor   # (N,) 기준 시점 접촉력 합 [N]
    rcv_palm_in_cup: torch.Tensor       # (N,3) 리시버 손의 같은 값들 (리시버 컵 좌표)
    rcv_tips_in_cup: torch.Tensor       # (N,F,3)
    rcv_force_share: torch.Tensor       # (N,F+1)
    rcv_force_total: torch.Tensor       # (N,)
    rcv_grasp_ref_valid: torch.Tensor   # (N,) bool
    rcv_ref_palm_in_cup: torch.Tensor   # (N,3)
    rcv_ref_tips_in_cup: torch.Tensor   # (N,F,3)
    rcv_ref_force_share: torch.Tensor   # (N,F+1)
    rcv_ref_force_total: torch.Tensor   # (N,)
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
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). The amount of beads in the source cup changes every episode and `ctx.bead_fill_level` (0 = empty, 1 = full to the rim, measured once the beads have settled, constant during the episode) tells the policy how full the cup is by volume. Beads start leaving the cup at a tilt that depends on that fill (24 mm beads, measured 09.23 with a scripted pour): the first bead leaves at about 91° with 20 beads (fill 0.85), 98° with 14 (fill 0.69) and 92° with 8 (fill 0.48), and 80 % of them are out about 5° later. ★Nothing pours below ~90°: a target angle or an over-tilt charge placed near 72° (the old 30 mm figure) makes the task impossible. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres. `ctx.src_pour_lip_pos` is the point on the source rim where the beads leave: it is built from the horizontal direction d from the source cup's origin to the receiver's mouth, which does not depend on the cup's attitude, so it is defined for an upright cup and does not jump when the cup wobbles or leans the wrong way (rim centre + r·(cos θ·d − sin θ·z), r = 4.1 cm). `ctx.src_tilt_toward_rcv` is θ, the signed tilt toward the receiver in the vertical plane through d (0 for a sideways lean, negative for a lean away). Because the rim centre itself moves toward the receiver by (cup height × sin θ) as the cup tilts, a cup that must tilt further (few beads) is aimed with its body further away — aim the LIP, not the rim centre, at the receiver's mouth.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment: the number of beads transferred is within ONE bead of `ctx.pour_target_beads` (the amount requested for this episode, also given to the actor as `ctx.pour_target_frac`), the source cup is back DOWN to at most 20° from upright, BOTH cups are still grasped at that moment, and the old safety conditions hold (little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — the cups NOT nested, and NO premature tilt: `ctx.premature_tilt` latches for the rest of the episode as soon as the GRASPED source cup's tilt `ctx.src_cup_tilt` (any direction) exceeds `ctx.premature_tilt_limit` — the fill-dependent release tilt minus 20°, i.e. 70° for a full cup and 74° for a half-full one (24 mm beads, measured 09.23; the first bead leaves near 90°, NOT 72°) — while `ctx.src_pour_lip_pos` is still more than 8.1 cm (xy) from the receiver's mouth centre (a cup knocked over without being grasped does not latch), and a latched episode can never succeed). Below that limit the source cup may be tilted anywhere, including while it is still approaching the receiver. You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air. Three more conditions void success: `ctx.cup_hit` (episode latch: the two cups touched each other with more than 0.5 N at any time after the settle phase — keep the cups apart on the table, while lifting and while pouring), the pouring direction `ctx.pour_dir_xy[:, 0] > 0.3` (the robot body is at -x; the source cup must pour sideways or toward the body, not away from it, so the receiver mouth must NOT be placed further from the body than the source cup), and `ctx.rcv_side_margin < 0.03` (the receiver cup must stay on its own arm's side of the centre line, about where it spawned at +0.16; the SOURCE arm brings its cup over). `ctx.src_wrap_count` / `ctx.rcv_wrap_count` count the fingers whose middle or distal link touches the cup (fingertip-only contact counts 0): an enveloping grasp has 3 or more, a fingertip pinch has 0-1 and collapses when the cup is tilted.
8. Height above the table: `ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]` is how far the source cup has been lifted (0 while it rests on the table).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and shown back to you after training, so name them meaningfully (e.g. "approach_src", "grasp_rcv", "pour_delta").
11. This policy will be deployed on the real robot, so collisions are a safety hazard: `ctx.cup_cup_force` is the contact force between the two cups, and `ctx.src_hand_foreign_force` / `ctx.rcv_hand_foreign_force` are the forces each hand exerts on anything other than its own cup (the other hand, the other cup, the table). During a correct pour the cups do not touch each other and the hands touch only their own cup; penalise these forces (bounded, e.g. `torch.tanh(f / 5.0)`) so the policy learns to keep clearance. Cup poses seen by the policy are delayed and noisy as on the real perception system.

I want it to fulfil the following task: Using both arms, grasp each cup: the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the pouring lip of the source cup over the mouth of the receiver cup, and tilt the source cup so that beads pour into the receiver cup. Pour the amount that was requested for this episode (ctx.pour_target_beads, within one bead), then bring the source cup back down to upright while both hands keep holding their cups. Keep the receiver cup upright, do not drop either cup, do not let the cups touch, and spill as few beads as possible. The environment decides success; this round's priority is that beads actually leave the source cup and land in the receiver, because the last three rounds poured nothing at all.
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
5. ★KEEP IT GENERAL. The reward must stay correct when the cup, the beads or the table change. Never hardcode an absolute angle, distance or size that belongs to the current assets — no "the cup releases at 72°", no fixed pour angle, no fixed cup radius. Measured numbers in the knowledge section are there to tell you what the task looks like, NOT to be copied into the code as targets. When you need such a quantity, take it from `ctx` (it is computed by the environment from the current assets) or, better, reward the OUTCOME that the quantity was a proxy for — beads actually leaving the source cup, beads arriving in the receiver, the requested amount being reached — and let the policy discover the angle by itself. Thresholds that describe the ROBOT or the reward's own shape (contact force floor, smoothing scales, ramp widths) are fine.
6. ★KEEP IT SIMPLE. Few terms, each with one clear job, beat many overlapping ones: a simpler reward generalises better and is far easier to diagnose from the training curves. Aim for about 15 components and do not exceed 25. Do not add a term that only repeats what another term already pays for, and do not add a penalty for a failure mode that has not been observed.

Observations from watching the trained policy:
# pour_bi 보상 설계 브리프 (09.26 종합판 — i16 ~ i28 전체)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.
**이 문서는 매 라운드 생성기 노트(notes_iterN.md) 맨 앞에 통째로 들어간다.** 라운드 노트는 이 브리프 뒤에 직전 런 진단만 덧붙인다.

## 0. 왜 이 문서가 필요한가 (사용자 지시 09.26)

- "보상설계 에이전트가 보상을 설계할 때 T2R 과 기존 학습 진행 결과를 종합적으로 보고 만들고 보고해."
- 지금까지 생성기는 직전 보상 코드 + 직전 런 곡선 + 그 라운드 노트만 봤다. iter_26~28 은 모두 직전 런만 고쳤고,
  25 에서 붓기를 가능하게 했던 조건(큰 회전 수입)을 26 에서 지운 뒤 그 영향을 다시 보지 않았다.
  결과: 25 는 94° 를 넘겨 부었는데 26r·27·28 은 55° 근처에서 멈춰 한 알도 못 부었다.

## 1. 과제와 고정 조건 (사용자 결정)

- 양팔로 컵을 잡아(5지 인벨롭 유지) actor obs 의 **요청량(±1 알)** 만 붓고, 소스 컵을 **세운 뒤 틸팅 전 자세로 복귀**.
  (USER_OBJECTIVES_20260925.md, taskspecs/drafts/pour_bi.yaml user_decisions: Q2 요청량+세우기, Q3 자세 복귀, Q5 K=5 로깅 먼저, M=0)
- 성공은 env 가 판정한다(pour_fabric_env.py:662-672). 보상은 판정하지 않는다.
- 원칙: 보상은 보편·단순(항 15개 내외), 각도 처방·자산 고유 치수 금지(창·거리는 ctx.cup_radius 배수),
  속도·제어 벌점을 풀어 효율을 얻지 않는다, 작업은 y≈0 중심선, 관측은 sim2real 가능한 것만.

## 2. 측정으로 확정된 물리 사실

| # | 사실 | 근거 |
|---|---|---|
| P1 | 첫 비드는 총 기울기 약 90 + 8·(1 − fill·in_source)° 이상에서 나온다(fill 0.55 에서 93.6°). | 24 mm 비드 프로브(09.23), notes_iter28 |
| P2 | premature 벌점은 총 기울기가 premature_tilt_limit(p50 73.6°)의 0.7 배, 즉 **약 52~58°** 부터 켜지고, spout 가 입구에서 2r(8.1 cm) 밖이면 env 래치가 성공을 무효로 만든다. | iter_25~28 PRE_LO 0.7, env cfg |
| P3 | env 는 컵이 20° 미만으로 서 있을 때만 들기 기준(파지 자세·힘)을 기록한다. 기울인 채 들면 기록이 안 된다. | pour_fabric_env._grasp_frame |
| P4 | 들기 순간 파지 기준 유지(iter_24 결합)는 효과가 있다: i24 드리프트 1.08~1.56 cm, 힘비 2.4×. | notes_round25 |
| P5 | 붓기 자세(주둥이가 입구 1 cm 위, 방출각)는 i27 plateau 상태에서 손목 델타 박스 안으로 99~100 % 도달 가능하다. 막은 것은 기구학이 아니라 보상이다. | notes_iter28 WHY 1 |
| P6 | ADR 은 순간 성공률 0.30 에서 오른다. 성공 0 이면 ADR 0 이고, ADR 에 묶인 떨림·액션 벌점은 계속 꺼져 있다. | cfg adr_trigger_threshold, i27 tremor 0 |

## 3. 라운드별 결과와 교훈

| 런 | 보상 요지 | 결과 | 교훈 |
|---|---|---|---|
| i16 (옛 목표) | 기울기 연동 높이 허용 | ep5600 ADR30 성공 92 % | 붓기는 가능하다 |
| i24 (옛 목표) | 들기 기준 파지 유지 | ADR30 성공 0.72, 최고 116° | 파지 결합 유지 |
| i25 | 새 목표, TILT_W 3 | 43° 정체, 붓기 0 | 회전 수입이 약하면 안 돈다 |
| i25b | TILT_W 15 웜 | 28~45° 그대로 | 좁아진 탐색은 웜으로 안 풀린다 |
| i25f | TILT_W 15 fresh | 93°, ep1147 첫 붓기, 5.5 중 3.3 붓고 기울인 채 버팀, 세우기 0 | 큰 연속 회전 수입(스텝당 ~+6)이 94° 를 넘게 한다 |
| i25j | + 세우기 인계 | ep1500 128°, 흘림 에피소드 98 %·6.9 알, 세우기 0 | 흘린 비드가 부은 양에 안 들어가 want 가 안 꺼짐 → 과회전. 기울인 채 버티기(10.6~14.5) > 세우기(≤5) |
| i26 | 회전 수입을 첫 비드 이탈 시 종료, need=min(요청−부음, 잔량) | e47 에서 교체 | 판단 전 종료 |
| i26r | fresh, 회전 수입을 입구 0.5r 안 속도형으로 | 55° → 38°, 붓기 0 | 게이트가 방문 상태에서 1e-17 이면 수입 0, 벌점만 남는다(−76~−125/에피소드) |
| i27 | 속도형 수입을 far=0 으로, 리시버 앵커 이동 | 조준은 좋아짐(spout 8.5 cm), 최고 56°(리시버 방향 47°) 벽, 붓기 0, 들기 기준 회피 28→100 %, 리시버 5지 0.26→0.13, flip 0.45 | premature 벽 앞에서 멈춤. 기준 회피가 자세 보상 2.2/스텝을 벌게 했다 |
| i28 웜(ep800) | 반경 2r, 기준 회피 차단, 5지 /F | e56 기준 기록 0.04 → 종료 | 없애려는 습관을 가진 체크포인트에서 웜하지 않는다 |
| i28f | 같은 보상 fresh | 학습 중(09.26 13:33) | — |

## 4. 누적 교훈 (새 보상은 이 목록에 대해 하나씩 답해야 한다)

- L1 **회전 구동력의 크기**: 25 는 연속 수입(스텝당 ~+6)으로 94° 를 넘었고, 26~28 의 속도형(상한 +1/스텝, TILT_W 5)은 55° 벽을 못 넘었다.
  새 회전 항은 최신 trace 의 plateau 상태에서 **"10°·20°·40° 더 기울일 때 스텝당 손익(회전 수입 − premature − 흘림 위험)"** 을 수치로 제시한다.
- L2 **흘림 회계**: 흘린 비드로 요청이 도달 불가가 되면 want 가 꺼지고 마무리가 열려야 한다(iter_26 need=min 유지).
- L3 **게이트 값 점검**: 새 게이트·폭은 최신 trace 방문 상태에서 분포(p10/p50/p90)를 보이고, 0 에 붙어 있지 않음을 확인한다.
- L4 **상태 수입 < 결과 수입**: 기울인 채 버티기, 서 있기, 들기 전 자세 같은 상태 수입이 붓기·세우기·성공보다 작아야 한다. 숫자로 비교한다.
- L5 **웜스타트**: 새 보상이 없애려는 습관이 체크포인트에 있으면 웜하지 않는다(i25b, i28). 웜 지점은 trace 로 확인한다.
- L6 **종료 시점**: fresh 는 첫 붓기까지 ~e1150(i25f). 기대 epoch 전에는 붕괴가 아니면 끊지 않는다.
- L7 **떨림**: 성공 0 동안 ADR 이 0 이라 떨림 벌점이 꺼져 있다. [사용자 결정 대기: ADR 무관하게 켤지]
- L8 **보상 해킹**: env 기록 조건(P3)을 피해 더 버는 경로가 없는지 확인한다.

## 5. 라운드 보고 형식 (생성 후 사용자에게)

1. 이번에 본 근거: 이 브리프의 어떤 교훈(L#)·어떤 trace·TB 구간을 봤는가.
2. 바꾼 것 / 유지한 것과 이유.
3. 최신 trace 상태에서의 수치: L1 손익표, L3 게이트 분포, L4 상태 수입 vs 결과 수입.
4. 위험과 게이트: 몇 epoch 에 무엇을 보고, 실패하면 무엇을 하는가.
5. 웜/fresh 선택 근거(L5).

---

# 새 계보 pour_bi2 · iter_00 노트 (2026-09-26)

## A. 왜 새로 시작하는가

`reward_gen/pour_bi/iter_00 ~ iter_28` 은 여기서 끝냈다. 마지막 세 라운드(i26r, i27, i28)가 연속으로 한 알도
못 부었다. 사용자 결정: iter_00 부터 다시 설계한다. 학습 기록과 교훈은
`~/rl_ws/our_source/pour_bi_runs_0926/` (INDEX.md, ROUNDS.md) 와 위 DESIGN_BRIEF 에 있다.
같은 폴더의 `ROUNDS_pour_bi_i00_i28.md` 가 라운드별 표와 반복 실패 원인 10가지다.

## B. 사용자 결정 (2026-09-26)

1. 목표 단계화: **붓기를 먼저 뚫는다.** 세우기와 자세 복귀는 다음 라운드에 올린다.
   env 의 성공 판정은 그대로다(요청량 ±1 + 소스 20° 이하 + 양손 파지 유지). 보상만 단계화한다.
2. 94도의 벽을 **양쪽에서** 뚫는다. 조급한 기울기 청구 시작점을 env 한계각 가까이로 올리고,
   동시에 55~94도 구간에 연속 회전 수입을 준다.
3. 학습은 server GPU0. 2026-09-27 08:00 까지 라운드 자동 진행(사용자 승인).

## C. 이번 iter_00 이 반드시 지켜야 하는 것 (REQUIREMENTS for the generator)

The reward must satisfy every item below. State the numbers you used in your answer.

R1. **Staged goal.** Beads leaving the source and landing in the receiver is the headline income.
    Uprighting and returning to the pre-tilt posture get only a small terminal bonus in this round.
    Do not make the finish branch pay more per step than pouring does.

R2. **Rotation drive that reaches the states the policy actually visits.** The last three rounds paid
    exactly 0 for rotation because the payout gate evaluated to 1e-17 or 1e-7 in every visited state.
    Gate the rotation income on the pouring lip being within `2 * ctx.cup_radius` (8.1 cm, the same
    radius the environment uses for its premature latch), never on a tighter precision window, and on
    the tilt being toward the receiver. At a state with a lip offset of 9 to 14 cm and a tilt of 55
    degrees the rotation income must still be clearly positive per step, not a rounding error.

R3. **Move the premature charge to the wall, not in front of it.** The environment latches a failure
    only when the grasped source cup passes `ctx.premature_tilt_limit` WHILE the lip is still more
    than 8.1 cm (xy) from the receiver mouth. Charge for premature tilt only near that limit (at or
    above 0.95 of it) and only while the lip is outside that radius. Inside the radius there is no
    tilt charge at all: the first bead does not leave until about 90 degrees, so charging from 55
    degrees makes the whole 55 to 94 degree band a pure loss.

R4. **State income below outcome income.** The sum of every per-step income that a policy can collect
    by just holding a posture (approach, aim, lift, posture, keep, home) must be smaller than the
    income for pouring one bead. Four rounds were lost to parking on state income.

R5. **Spill accounting.** Use `need = min(target - poured, beads still in the source)`. When `need`
    reaches zero, the rotation income stops and the finish branch opens. Beads that spilled must not
    keep the request open, or the policy over-rotates (i25j reached 128 degrees and spilled in 98
    percent of episodes).

R6. **Keep the two guards that were bought with previous rounds.**
    (a) The iter_24 grip-keep principle: pay for similarity to the grasp recorded at lift (relative
    hand-cup pose, finger force distribution, total force), never for an absolute force target.
    (b) The environment records that reference only when the cup is below 20 degrees at lift-off.
    A policy that tilts before lifting must not be able to earn the pre-lift posture income instead.

R7. **Exploration.** Charges that suppress exploration (action flip, hand flip, palm saturation,
    carry-phase rate) stay ramped by `ctx.adr_progress` as in iter_22 and later, because applying
    them at full weight from step 0 of a fresh run taught the policy not to touch the cups (i21, e26).
    A small always-on action-rate charge at the iter_18 level is fine.

R8. **No hardcoded angles or asset dimensions.** Windows and distances are multiples of
    `ctx.cup_radius`; angles come from `ctx.premature_tilt_limit` and the bead release relation.
    A reward that hardcodes 72 or 86 degrees broke when the bead size changed.

R9. Keep the term count near 15 to 22. Simple and general beats clever.

## D. 이 라운드에 답해야 하는 숫자 (answer these in the response text)

1. At tilt 55 degrees with the lip 10 cm from the mouth, what is the per-step rotation income and the
   per-step premature charge?
2. What is the net per-step profit of tilting 10, 20 and 40 degrees further from that state?
3. What is the total per-step state income (R4) and what does one poured bead pay?
