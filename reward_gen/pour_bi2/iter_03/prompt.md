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

The previous reward function was:
```python
import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.sqrt((d * d).sum(dim=-1) + 1e-12)


def _dist_xy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a[:, :2] - b[:, :2]
    return torch.sqrt((d * d).sum(dim=-1) + 1e-12)


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bimanual pour: grasp both cups, lift, bring the cups together, aim the pouring
    LIP over the receiver mouth and tilt until the requested number of beads has left
    the source cup and landed in the receiver.

    Changes of this iteration (exactly the two the round asks for):
      C1  AIMING NOW RAISES THE ROTATION INCOME and never switches it off.
          iter_01 poured with the lip ~3*cup_radius (0.123 m) beside the mouth: the old
          `pour_rot` had no aim factor at all, so tilting without aiming paid the same as
          tilting over the opening.  The aim factor is a BONUS ON A FLOOR
          (FLOOR = 0.3), never a precision gate - iter_26/27 multiplied by a narrow
          gaussian that evaluated to 1e-17 in the visited states and three rounds poured
          nothing.  Teeth are on the OUTCOME side instead: the spill charge per bead is
          raised 20 -> 40 (> the 30 paid for a bead that arrives), so pouring while badly
          aimed loses money while the rotation drive stays alive.
      C2  TREMOR AND ACTION-REVERSAL CHARGES ARE ALIVE FROM STEP 0 (no adr factor),
          small and weighted more on the receiver arm (its qd RMS ran 1.2-1.3x the source
          arm's, and the flip rate sat at 0.49 all run while adr_progress stayed 0.000).
          The palm-saturation charge stays ramped by adr_progress (R7).
      Everything else is unchanged in value: converge_cups, aim_lip, lip_gate, the
      rotation cap, the premature charge at the wall, grip-keep, safety, bookkeeping.
      Only logging was compacted (approach_src+approach_rcv, premature_charge+latched,
      cup_cup+nested merged) to keep the component count at 22 (R9).

    Design rules: every length is a multiple of ctx.cup_radius, every angle a fraction of
    ctx.premature_tilt_limit -> no asset dimension or absolute angle is hardcoded (R8).
    """
    eps = 1e-6
    zeros = torch.zeros_like(ctx.src_cup_tilt)

    # ---- shared scales ---------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # env latch radius for the lip (xy)
    wide = 10.0 * r_cup                      # WIDE kernel: covers the visited 0.2-0.45 m band
    lim = torch.clamp(ctx.premature_tilt_limit, min=1e-3)   # (N,) rad, from env
    n_fing = max(int(ctx.src_tips_pos.shape[1]), 1)

    src_g = ctx.src_grasped.float()
    rcv_g = ctx.rcv_grasped.float()
    both_g = src_g * rcv_g

    # Pre-lift / posture income must not be earnable by tilting before lifting (R6b):
    # the env only records the grasp reference while the cup is well below its limit,
    # so fade that income out with the cup's own tilt, using 0.3 * limit as the scale.
    low_src = 1.0 - torch.tanh(torch.clamp(ctx.src_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))
    low_rcv = 1.0 - torch.tanh(torch.clamp(ctx.rcv_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))

    # ---- 1. approach both hands (0.4 each; dies out once that cup is grasped) ---
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_both = (0.4 * (1.0 - src_g) * low_src * (1.0 - torch.tanh(d_src / two_r))
                     + 0.4 * (1.0 - rcv_g) * low_rcv * (1.0 - torch.tanh(d_rcv / two_r)))

    # ---- 2-3. enveloping grasp (weight 0.5 each) -------------------------------
    # wrap counts are normalised by the number of fingers of THIS hand, not by 5.
    wrap_src = _sat(ctx.src_wrap_count / float(n_fing))
    wrap_rcv = _sat(ctx.rcv_wrap_count / float(n_fing))
    grasp_src = 0.5 * src_g * (0.4 + 0.6 * wrap_src)
    grasp_rcv = 0.5 * rcv_g * (0.4 + 0.6 * wrap_rcv)

    # ---- 4. lift both cups (weight 0.5, saturates at a 2*cup_radius lift) ------
    h_src = _sat((ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]) / two_r)
    h_rcv = _sat((ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]) / two_r)
    lift_both = 0.5 * both_g * 0.5 * (h_src * low_src + h_rcv * low_rcv)
    lifted = torch.minimum(h_src, h_rcv)

    lip_d = _dist_xy(ctx.src_pour_lip_pos, ctx.rcv_cup_mouth_pos)
    dz_lip = ctx.src_pour_lip_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]

    # ---- 5. bring the two cups together (weight 1.2, WIDE kernel) --------------
    # Unchanged: this is what woke the pour up in iter_01 (1.0 per step, cups 0.21 m).
    converge_cups = 1.2 * both_g * lifted * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- 6. aim the LIP (not the rim centre) over the receiver mouth (0.8) -----
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * (0.6 * xy_score + 0.4 * h_score)

    # ---- 7. keep the grasp recorded at lift-off (iter_24 guard, weight 1.0) ----
    # Similarity to the reference relation only; never an absolute force target (R6a).
    s_len = 0.5 * r_cup

    def _grip(palm, ref_palm, tips, ref_tips, share, ref_share, tot, ref_tot, valid, gr):
        p_err = _dist(palm, ref_palm)
        t_err = torch.sqrt(((tips - ref_tips) ** 2).sum(dim=-1) + 1e-12).mean(dim=-1)
        s_err = torch.abs(share - ref_share).mean(dim=-1)
        f_err = torch.abs(tot - ref_tot) / torch.clamp(ref_tot, min=0.5)  # 0.5 N force floor
        score = (0.40 * (1.0 - torch.tanh(p_err / s_len))
                 + 0.35 * (1.0 - torch.tanh(t_err / s_len))
                 + 0.15 * (1.0 - torch.tanh(s_err / 0.15))
                 + 0.10 * (1.0 - torch.tanh(f_err)))
        return valid.float() * gr * score

    grip_src = _grip(ctx.src_palm_in_cup, ctx.src_ref_palm_in_cup,
                     ctx.src_tips_in_cup, ctx.src_ref_tips_in_cup,
                     ctx.src_force_share, ctx.src_ref_force_share,
                     ctx.src_force_total, ctx.src_ref_force_total,
                     ctx.src_grasp_ref_valid, src_g)
    grip_rcv = _grip(ctx.rcv_palm_in_cup, ctx.rcv_ref_palm_in_cup,
                     ctx.rcv_tips_in_cup, ctx.rcv_ref_tips_in_cup,
                     ctx.rcv_force_share, ctx.rcv_ref_force_share,
                     ctx.rcv_force_total, ctx.rcv_ref_force_total,
                     ctx.rcv_grasp_ref_valid, rcv_g)
    grip_keep = 1.0 * 0.5 * (grip_src + grip_rcv)

    # ---- bead bookkeeping: need = min(target - poured, still in source) (R5) ---
    active = torch.clamp(ctx.bead_active_count, min=1.0)
    remaining = torch.clamp(ctx.pour_target_frac - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    need_beads = need_frac * active
    # `want` tapers inside the last bead (env tolerance +-1 bead) and is 0 once spill
    # made the request unreachable -> the rotation income stops, the finish branch opens.
    want = _sat(need_beads)

    # ---- 8. rotation drive: the term that has to cross the ~90 deg wall (8.0) --
    # Gate (kept from iter_01, it is what finally paid): lip near the mouth with the
    # SAME radius the env latch uses, soft 2*cup_radius tail so 9-14 cm still pays.
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    # Progress measured against 1.4 * premature_tilt_limit (a multiple of a ctx angle):
    # reaches past the bead-release tilt for every fill and caps the income at 8.
    rot_prog = _sat(torch.clamp(ctx.src_tilt_toward_rcv, min=0.0) / (1.4 * lim))
    # NEW aim factor: a BONUS ON A FLOOR, not a gate. Both parts are smooth and their
    # tails stay clearly non-zero in the states iter_01 visited (lip 3 * cup_radius,
    # drop 0.03 m): xy part 0.24, height part 0.68 -> factor 0.58 (not 1e-7).
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)      # 1 inside 1r
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))  # low drop
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    # monotone in lip_d at fixed tilt (both lip_gate and aim_xy shrink with lip_d)
    pour_rot = 8.0 * src_g * ctx.src_grasp_ref_valid.float() * want * lip_gate * aim_factor * rot_prog

    # ---- 9. headline income: beads that actually arrive (30 per bead) ----------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 10. pouring past the request (15 per bead beyond the +-1 tolerance) ---
    over_beads = _sat(torch.clamp(ctx.bead_in_target_frac - ctx.pour_target_frac, min=0.0) * active - 1.0,
                      0.0, 5.0)
    pour_overshoot = -15.0 * over_beads

    # ---- 11. spill: 40 per lost bead (was 20) - the teeth for bad aim ----------
    # A bead that leaves the source and misses the receiver now costs MORE (40) than a
    # bead that arrives pays (30), so pouring beside the mouth is a net loss over the
    # episode while the rotation drive itself stays untouched. Charged on the increment.
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 12. finish branch: small, and per step below the pour income (R1) -----
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(ctx.src_cup_tilt / (0.3 * lim)))

    # ---- 13. terminal success bonus (env decides success, we only pay it) ------
    success_bonus = 20.0 * ctx.success.float()

    # ---- 14. premature tilt: charge AT the wall only, plus the latch reminder --
    # ramps in above 0.95 * limit over the last 0.05 * limit and only while the lip is
    # outside 2*cup_radius; inside that radius there is NO tilt charge at all (R3).
    pre_x = torch.clamp(ctx.src_cup_tilt - 0.95 * lim, min=0.0) / (0.05 * lim)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate - 0.4 * ctx.premature_tilt.float()

    # ---- 15. real-robot safety: cups must never touch, nesting is never a pour -
    cup_clash = -(2.0 * torch.tanh(ctx.cup_cup_force / 5.0)
                  + 0.5 * ctx.cup_hit.float()
                  + 3.0 * ctx.cups_nested.float())

    # ---- 16. hands touch only their own cup ------------------------------------
    hand_foreign = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                           + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 17. pouring direction must not point away from the body --------------
    # dimensionless unit-vector component (env voids at 0.3); ramp starts at half of it
    dir_ramp = _sat((ctx.pour_dir_xy[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 18. receiver cup: upright and on its own side of the centre line -----
    side_lo = 0.75 * r_cup                      # env voids below ~0.75 * cup_radius
    side_bad = _sat((side_lo - ctx.rcv_side_margin) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * lim))) - 1.5 * side_bad

    # ---- 19. always-on action rate (iter_18 level, small) --------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)

    # ---- 20. NEW: action reversal, alive from step 0 (weight 0.30) ------------
    # measured flip rate 0.49 with |a| ~ 0.5 -> about -0.04 per step (-0.15 if the
    # actions are saturated): ~5 % of the per-step state income, far below iter_21
    # where full-weight charges taught the policy not to touch the cups.
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip

    # ---- 21. NEW: arm tremor, alive from step 0, receiver arm weighted more ---
    # the receiver arm ran 1.2-1.3x shakier than the source arm; scale 1 rad/s so the
    # measured RMS 0.46 / 0.42 sits on the slope, total about -0.15 per step.
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))

    # ---- 22. palm saturation stays ramped by ADR (R7) -------------------------
    adr = float(ctx.adr_progress)
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -adr * 0.3 * sat

    components = {
        "approach_both": approach_both,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_both": lift_both,
        "converge_cups": converge_cups,
        "aim_lip": aim_lip,
        "grip_keep": grip_keep,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "pour_overshoot": pour_overshoot,
        "spill": spill,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "premature": premature,
        "cup_clash": cup_clash,
        "hand_foreign": hand_foreign,
        "pour_dir": pour_dir,
        "rcv_keep": rcv_keep,
        "action_rate": action_rate,
        "action_flip": action_flip,
        "arm_tremor": arm_tremor,
        "action_reg_adr": action_reg_adr,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 2.71e-05, 0, 0, 0, 2.44e-05, 3.05e-05, 0, 0, 0]  min 0 · mean 3.93e-06 · max 9.77e-05
bead/spill: [0, 0.000831, 0, 0, 4.25e-05, 0.000415, 9.16e-05, 2.03e-05, 0.000139, 0.000138]  min 0 · mean 0.000161 · max 0.0014
done/drop: [0, 0.0132, 0.0171, 0.0181, 0.0146, 0.0115, 0.0183, 0.0161, 0.0154, 0.0178]  min 0 · mean 0.0143 · max 0.0251
episode_lengths/step: [45, 80.2, 74.5, 65.2, 65.9, 67.8, 66.2, 65.8, 67.3, 68.1]  min 45 · mean 70.7 · max 120
reward/action_flip: [-0.0479, -0.0345, -0.0249, -0.0243, -0.0264, -0.0249, -0.0214, -0.0222, -0.0213, -0.0211]  min -0.0479 · mean -0.0257 · max -0.0189
reward/action_rate: [-0.043, -0.0334, -0.0254, -0.0258, -0.0274, -0.0267, -0.0241, -0.0248, -0.024, -0.024]  min -0.043 · mean -0.0271 · max -0.0222
reward/action_reg_adr: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/aim_lip: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.17e-09 · max 1.13e-06
reward/approach_both: [0.0285, 0.0215, 0.0224, 0.0256, 0.0285, 0.0268, 0.0267, 0.0273, 0.0271, 0.0262]  min 0.0178 · mean 0.0257 · max 0.0305
reward/arm_tremor: [-0.000619, -0.109, -0.097, -0.0776, -0.0683, -0.0744, -0.0737, -0.0663, -0.0681, -0.057]  min -0.222 · mean -0.0749 · max -0.000549
reward/converge_cups: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.03e-09 · max 6.79e-06
reward/cup_clash: [0, -0.00122, 0, -0.00281, 0, -0.00549, -0.000244, -0.00061, -0.00134, -0.000122]  min -0.497 · mean -0.0185 · max 0
reward/finish_upright: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0, 0, 0, 0, 7.81e-05, 0, 0, 0, 0]  min 0 · mean 2.99e-05 · max 0.000811
reward/grasp_src: [0, 9.28e-05, 0, 0, 0, 0.000156, 9.28e-05, 0, 0, 0]  min 0 · mean 1.71e-05 · max 0.000249
reward/grip_keep: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.52e-06 · max 0.000202
reward/hand_foreign: [-0.000509, -0.0629, -0.0325, -0.0336, -0.0288, -0.0246, -0.0243, -0.0234, -0.0266, -0.0212]  min -0.299 · mean -0.0315 · max 0
reward/lift_both: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.93e-09 · max 4.24e-06
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.59e-06 · max 0.00732
reward/pour_dir: [0, -0.00117, -0.00037, -4.17e-05, -8.47e-05, -0.000593, -7.3e-05, -0.000145, -3.25e-05, -5.33e-05]  min -0.0121 · mean -0.000632 · max 0
reward/pour_overshoot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pour_rot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.74e-07 · max 0.00034
reward/premature: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00312 · mean -4.18e-05 · max 0
reward/rcv_keep: [0, 0, 0, 0, 0, -0.000219, 0, 0, -0.000619, 0]  min -0.266 · mean -0.00308 · max 0.000114
reward/spill: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0684 · mean -0.00266 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [0, -0.201, -0.146, -0.122, -0.102, -0.111, -0.102, -0.0936, -0.098, -0.0819]  min -0.99 · mean -0.14 · max 0
rewards/step: [-0.0577, -15.2, -10.7, -7.51, -6.61, -6.69, -6.07, -6.09, -6.18, -5.6]  min -99.4 · mean -10.7 · max -0.0577
task/aim_dist: [0.319, 0.334, 0.298, 0.29, 0.292, 0.295, 0.29, 0.295, 0.295, 0.301]  min 0.204 · mean 0.299 · max 0.487
task/cup_collision_rate: [0, 0.000488, 0, 0.000977, 0, 0.00122, 0, 0.000244, 0.000488, 0]  min 0 · mean 0.00361 · max 0.0989
task/cups_center_dist: [0.32, 0.345, 0.305, 0.301, 0.303, 0.304, 0.301, 0.305, 0.306, 0.312]  min 0.225 · mean 0.308 · max 0.496
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0, 0, 0, 0, 0.000732, 0, 0, 0, 0]  min 0 · mean 0.0026 · max 0.0674
task/rcv_cup_lift: [9.03e-05, 0.000819, 0.00123, 0.000568, 0.000519, 0.000805, 0.000594, 0.000689, 0.000423, 0.000311]  min -0.000819 · mean 0.000821 · max 0.0188
task/rcv_grasped: [0, 0, 0, 0, 0, 0.000244, 0, 0, 0, 0]  min 0 · mean 8.85e-05 · max 0.00244
task/rcv_hand_foreign_rate: [0, 0.0156, 0.00391, 0.00146, 0.00146, 0.00195, 0.000488, 0, 0.000732, 0.00171]  min 0 · mean 0.00573 · max 0.306
task/src_cup_lift: [0.000122, 0.00717, -0.00192, -0.00119, -0.00119, -0.00101, -0.00126, -0.00105, 0.00125, -0.000677]  min -0.00197 · mean 0.000603 · max 0.0244
task/src_grasped: [0, 0.000244, 0, 0, 0, 0.000488, 0.000244, 0, 0, 0]  min 0 · mean 5.19e-05 · max 0.000732
task/src_hand_foreign_rate: [0.00122, 0.0854, 0.0513, 0.0469, 0.0422, 0.0359, 0.0364, 0.0347, 0.0381, 0.0325]  min 0 · mean 0.0424 · max 0.127
task/src_tilt_deg: [0.291, 15.8, 15.2, 10.9, 10.3, 11.6, 11.6, 10.1, 10.9, 10.9]  min 0.154 · mean 11.5 · max 23.1
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

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


---

## E. iter_01 학습 결과 (t2r2_i01, e1350 / 9.5 h, fresh, server GPU0) — measured facts and the user's video verdict

The converge change worked: the cups came together, the rotation income woke up, and the policy
poured. Then it regressed. The user watched three replays and gave the verdict below; the user's
judgement is the reference.

**User video verdict (09-27):** "In all three videos the aim is off and the beads spill sideways.
Beads come out one at a time and every one of them goes beside the receiver. The right arm is steady
but the left arm shakes a lot."

| measurement | e700 | e900 | e1100 | e1300 |
|---|---|---|---|---|
| task/aim_dist (lip to mouth, xy, while pouring) | 0.129 m | 0.123 m | 0.124 m | 0.129 m |
| task/pour_lip_dist (episode mean) | 0.071 | 0.065 | 0.063 | 0.074 |
| diag/ep_peak_tilt_deg_tgt_mid | 96.7 | 96.1 | 95.2 | 98.6 |
| bead/in_target | 0.115 | 0.108 | 0.140 | 0.030 |
| bead/spill | 0.115 | 0.088 | 0.073 | 0.265 |
| task/premature_tilt_rate | 0.082 | 0.082 | 0.108 | 0.329 |
| diag/ep_amount_on_target | 0.189 | 0.281 | 0.202 | 0.019 |
| diag/ep_amount_err_beads | -3.04 | -2.25 | -2.68 | -4.63 |
| diag/ctrl_arm_qd_rms_rcv (receiver arm) | 0.465 | 0.454 | 0.467 | 0.444 |
| diag/ctrl_arm_qd_rms_src (source arm) | 0.419 | 1.514 | 0.378 | 0.365 |
| diag/ctrl_flip_rate | 0.490 | 0.486 | 0.489 | 0.489 |
| reward/action_reg_adr (paid) | 0.000 | 0.000 | 0.000 | 0.000 |
| adr/progress | 0.000 | 0.000 | 0.000 | 0.000 |
| diag/ep_end_upright_held | 0.000 | 0.000 | 0.000 | 0.004 |
| diag/pose_meet_y_pour | 0.024 | 0.029 | 0.047 | 0.049 |

Two causes, both confirmed by the numbers:

1. **The rotation income is paid while the lip is NOT over the mouth.** During the pouring phase the
   lip is 0.123 to 0.129 m from the receiver mouth in xy. The cup radius is 0.0405 m and the env
   latch radius is 0.081 m, so the beads leave the lip about three cup radii beside the opening.
   `pour_rot` uses `lip_gate = 1 - tanh(max(lip_d - 2r, 0) / 2r)`, which still pays about 0.23 of
   full value there, and nothing in the term asks for the lip to be ABOVE the opening. Tilting
   without aiming is therefore profitable.

2. **The tremor and action-flip charges were never applied.** They are multiplied by
   `ctx.adr_progress`, which stayed 0.000 for the whole run because ADR only rises after the success
   rate passes 0.30. The action flip rate sat at 0.49 (half of all steps reverse sign) and the
   receiver arm ran 1.2 to 1.3 times shakier than the source arm.

Unchanged and working, do not touch: grasp 0.87 both hands, wrap 3.3/4.4 fingers, grip reference
recorded (`diag/grip_src_ref_valid` 0.89), converge_cups 1.0 per step, premature charge at the wall.

## F. 이번 라운드의 변경 (two changes, decided by the user 09-27)

**Change 1 — aiming must RAISE the rotation income, and must never switch it off.**

Read this constraint before writing the term. In the old lineage, iter_26 and iter_27 multiplied the
rotation income by a tight precision window (`fine = exp(-(offset / (0.5 * cup_radius))**2)`). In the
states the policy actually visited that factor evaluated to 1e-17 and 1e-7, so rotation was never
paid, the cup stopped at 38 to 56 degrees and three rounds in a row poured nothing. Do not repeat
that construction.

The aim factor must be a bonus on top of a floor, not a gate:

    aim_quality in [0, 1]   built from the pouring lip relative to the receiver mouth:
        horizontal offset measured in units of ctx.cup_radius (about 1 cup radius = full,
        falling to 0 at the env latch radius 2 * cup_radius), combined with a low, controlled
        drop height above the mouth
    rotation income = W * rot_prog * lip_gate * (FLOOR + (1 - FLOOR) * aim_quality)   with FLOOR about 0.3

State these numbers in your answer and make them true:
  (a) At the state this policy actually occupied while pouring (lip offset 3 * cup_radius = 0.123 m,
      tilt 75 degrees, lip 0.03 m above the mouth) the rotation income must still be at least
      1.5 per step and at least 30 percent of its value at perfect aim.
  (b) At perfect aim (offset 0, low drop) the rotation income must be clearly larger than at
      3 * cup_radius, so that aiming is what pays.
  (c) The income must rise monotonically as the offset shrinks at a fixed tilt.

Put the teeth on the OUTCOME side instead of the gate: a bead that leaves the source and lands
outside the receiver is already charged by the spill term. Raise that charge so that pouring while
badly aimed loses money over the episode, while the rotation drive itself stays alive. Keep
`lip_gate`, the rotation cap, the premature charge at the wall, `converge_cups` and every other term
as they are.

**Change 2 — charge tremor and action reversal regardless of `ctx.adr_progress`.** These charges
were multiplied by `ctx.adr_progress`, which stayed 0.000 for the whole run, so they were never
applied: the action flip rate sat at 0.49 and the receiver arm ran 1.2 to 1.3 times shakier than the
source arm. Make them alive from step 0, small, and weighted more on the receiver arm than on the
source arm. Keep the total of these charges well under the approach and posture incomes: in iter_21
of the old lineage the same charges at full weight taught the policy not to touch the cups at all
and the run died at epoch 26. State the per-step size of these charges at the measured flip rate
0.49 and receiver joint-speed RMS 0.46 in your answer.

Constraints that still hold: the sum of per-step income a policy can collect by holding a posture
must stay below the income for pouring one bead (30); every window and distance is a multiple of
`ctx.cup_radius`; every angle is a fraction of `ctx.premature_tilt_limit`.


---

## G. iter_02 학습 결과 (t2r2_i02l, LSTM agent, horizon 16, fresh, e926 / 2.2 h) — measured facts

The run never learned to grasp. The agent was switched to the LSTM configuration for this run by
user decision, so read the reward numbers together with the note on the learning rate below.

| measurement | e20 | e200 | e400 | e800 | e920 |
|---|---|---|---|---|---|
| task/src_grasped, task/rcv_grasped | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 / 0.001 |
| task/src_closure (hand closing) | 0.022 | 0.012 | 0.008 | 0.009 | 0.009 |
| reward/total | -0.667 | -0.143 | -0.103 | -0.093 | -0.084 |
| reward/approach_both (the only positive income) | +0.023 | +0.023 | +0.028 | +0.027 | +0.025 |
| reward/arm_tremor | -0.108 | -0.096 | -0.069 | -0.066 | -0.048 |
| reward/action_flip | -0.043 | -0.025 | -0.026 | -0.021 | -0.021 |
| reward/hand_foreign | -0.104 | -0.029 | -0.027 | -0.022 | -0.024 |
| diag/ctrl_arm_qd_rms_rcv | 0.386 | 0.424 | 0.219 | 0.194 | 0.178 |

60.7 million environment steps were collected. The MLP run with the iter_01 reward reached a grasp
rate of 0.8 after about 26 million steps.

Cause on the reward side: in a fresh run the policy has not grasped anything yet, so the approach,
grasp, lift and posture incomes are essentially zero, and the always-on tremor and action-reversal
charges added in iter_02 were the largest signal the policy received. The total reward stayed
negative for the whole run, and the policy reduced the charge by moving less (receiver arm joint
speed 0.44 -> 0.18) and by not closing its hands. This is the same failure as iter_21 of the old
lineage. The iter_02 property check was wrong: it compared the size of these charges with the
posture income of a state in which both cups are already grasped and lifted, a state the fresh
policy never reaches.

(Agent side, handled outside the reward by the operator: the LSTM configuration used an adaptive
learning-rate schedule that cut the actor rate to 1.8e-5 at the start and kept it between 6e-5 and
2e-4, and a critic learning rate of 5e-5. The next run uses the same learning rates as the MLP runs.
Do not compensate for this in the reward.)

## H. 이번 라운드의 변경 (one change, decided by the user 09-27)

**Return the tremor and action-reversal charges to the ADR gate, as they were in iter_00 and iter_01.**
Multiply `arm_tremor` and `action_flip` by `ctx.adr_progress`, so they are exactly zero while the
curriculum has not started and grow as the policy starts to succeed. Inside that gate keep the iter_02
sizes and the heavier weight on the receiver arm. The user decided this ordering: the policy must
first learn the task, and only then pay for shaking.

Keep every other term, weight and gate of iter_02 exactly as it is, including the aim factor on a
floor, the spill charge of 40 per bead and the premature charge at the wall.

State in your answer: the per-step total of the tremor and reversal charges (a) with
`adr_progress = 0` and (b) with `adr_progress = 1` at the measured flip rate 0.49 and receiver
joint-speed RMS 0.46.
