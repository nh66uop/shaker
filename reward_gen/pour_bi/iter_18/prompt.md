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
5. Cup tilt: `ctx.src_cup_tilt` is the angle between the cup's up axis and world +z (0 = upright, π/2 = horizontal). The amount of beads in the source cup changes every episode and `ctx.bead_fill_level` (0 = empty, 1 = full to the rim, measured once the beads have settled, constant during the episode) tells the policy how full the cup is by volume. Beads start leaving the cup at a tilt that depends on that fill (measured): the first bead leaves at about 70° when the cup is full and at about 90° when it holds only a few beads, and a full cup has lost a fifth of its beads by 80° and half by 86°. `ctx.src_cup_mouth_pos` and `ctx.rcv_cup_mouth_pos` are the rim centres. `ctx.src_pour_lip_pos` is the point on the source rim where the beads leave: it is built from the horizontal direction d from the source cup's origin to the receiver's mouth, which does not depend on the cup's attitude, so it is defined for an upright cup and does not jump when the cup wobbles or leans the wrong way (rim centre + r·(cos θ·d − sin θ·z), r = 4.1 cm). `ctx.src_tilt_toward_rcv` is θ, the signed tilt toward the receiver in the vertical plane through d (0 for a sideways lean, negative for a lean away). Because the rim centre itself moves toward the receiver by (cup height × sin θ) as the cup tilts, a cup that must tilt further (few beads) is aimed with its body further away — aim the LIP, not the rim centre, at the receiver's mouth.
6. Bead bookkeeping: `bead_in_target_frac` rises as beads land in the receiver cup; `bead_spill_frac` counts beads lost outside both cups (permanent). `d_in_target` and `d_spill` are this step's increments — reward INCREMENTS of beads transferred rather than the level, otherwise the policy is paid for standing still with a filled cup.
7. `ctx.success` is computed by the environment (enough beads in the receiver cup, little spill, cups close together, the receiver cup held nearly upright — `ctx.rcv_cup_tilt` at most 20° — the cups NOT nested, and NO premature tilt: `ctx.premature_tilt` latches for the rest of the episode as soon as the GRASPED source cup's tilt `ctx.src_cup_tilt` (any direction) exceeds `ctx.premature_tilt_limit` — the fill-dependent release tilt minus 20°, i.e. 52° for a full cup and 69° for a half-full one — while `ctx.src_pour_lip_pos` is still more than 8.1 cm (xy) from the receiver's mouth centre (a cup knocked over without being grasped does not latch), and a latched episode can never succeed). Below that limit the source cup may be tilted anywhere, including while it is still approaching the receiver. You may add a bonus on it but you cannot redefine it. Beads only count as "in the receiver" once they have LEFT the source cup — pushing the source cup into the receiver cup (`ctx.cups_nested`) transfers nothing and is never a success; the beads must fall out of the tilted source cup through the air. Three more conditions void success: `ctx.cup_hit` (episode latch: the two cups touched each other with more than 0.5 N at any time after the settle phase — keep the cups apart on the table, while lifting and while pouring), the pouring direction `ctx.pour_dir_xy[:, 0] > 0.3` (the robot body is at -x; the source cup must pour sideways or toward the body, not away from it, so the receiver mouth must NOT be placed further from the body than the source cup), and `ctx.rcv_side_margin < 0.03` (the receiver cup must stay on its own arm's side of the centre line, about where it spawned at +0.16; the SOURCE arm brings its cup over). `ctx.src_wrap_count` / `ctx.rcv_wrap_count` count the fingers whose middle or distal link touches the cup (fingertip-only contact counts 0): an enveloping grasp has 3 or more, a fingertip pinch has 0-1 and collapses when the cup is tilted.
8. Height above the table: `ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]` is how far the source cup has been lifted (0 while it rests on the table).
9. Do not keep any state between calls (no globals, no attributes); the function must be pure.
10. Each component you return is logged separately during training and shown back to you after training, so name them meaningfully (e.g. "approach_src", "grasp_rcv", "pour_delta").
11. This policy will be deployed on the real robot, so collisions are a safety hazard: `ctx.cup_cup_force` is the contact force between the two cups, and `ctx.src_hand_foreign_force` / `ctx.rcv_hand_foreign_force` are the forces each hand exerts on anything other than its own cup (the other hand, the other cup, the table). During a correct pour the cups do not touch each other and the hands touch only their own cup; penalise these forces (bounded, e.g. `torch.tanh(f / 5.0)`) so the policy learns to keep clearance. Cup poses seen by the policy are delayed and noisy as on the real perception system.

I want it to fulfil the following task: Using both arms, grasp each cup: the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the mouth of the source cup over the mouth of the receiver cup, and tilt the source cup so that the beads pour into the receiver cup. Keep the receiver cup upright, do not drop either cup, and spill as few beads as possible. The task is finished when at least half of the beads are inside the receiver cup with little spill.
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

The previous reward function was:
```python
import torch
import math


# --- geometry constant: the pouring lip sits this far from the rim centre (see ctx doc, r = 4.1 cm) ----
_LIP_R = 0.041


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    # x already normalised; clamp to [0,1] then 3x^2 - 2x^3
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap_quality(finger_force: torch.Tensor, palm_force: torch.Tensor, closure: torch.Tensor,
                  palm_axes: torch.Tensor, cup_up: torch.Tensor, cup_pos: torch.Tensor,
                  palm_pos: torch.Tensor) -> torch.Tensor:
    # Power-grasp quality in [0,1]. Weighted MEAN so each ingredient keeps its own gradient.
    FINGER_ON = 0.3      # [N]
    PALM_SCALE = 3.0     # [N]
    LEVEL_SIGMA = 0.05   # [m]
    CLOS_LO, CLOS_HI = 0.25, 0.40
    dt = closure.dtype
    finger_frac = torch.mean((finger_force > FINGER_ON).to(dt), dim=-1)
    palm_c = torch.tanh(torch.clamp(palm_force, min=0.0) / PALM_SCALE)
    normal = palm_axes[:, 0:3]
    side = 1.0 - torch.clamp(torch.abs(torch.sum(normal * cup_up, dim=-1)), 0.0, 1.0)
    axial = torch.sum((cup_pos - palm_pos) * cup_up, dim=-1)
    level = torch.exp(-(axial / LEVEL_SIGMA) ** 2)
    clos = torch.clamp((closure - CLOS_LO) / (CLOS_HI - CLOS_LO), 0.0, 1.0)
    return 0.30 * finger_frac + 0.30 * palm_c + 0.20 * side + 0.10 * level + 0.10 * clos


def _carry_phase(grasped_f: torch.Tensor, h: torch.Tensor,
                 grasp_lo: float, grasp_hi: float, free_lo: float, free_hi: float) -> torch.Tensor:
    # (N,) in [0,1]: 0 on the table, 1 once grasped and lifted; far above the table the flag is not needed
    held_ramp = grasped_f * _smoothstep((h - grasp_lo) / (grasp_hi - grasp_lo))
    high_ramp = _smoothstep((h - free_lo) / (free_hi - free_lo))
    return torch.maximum(held_ramp, high_ramp)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    N = ctx.src_palm_pos.shape[0]
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ------------------------------------------------------------------ constants
    LIFT_TARGET_SRC = 0.28    # iter_16: the lip must clear the receiver rim at ~80 deg -> ask for more lift
    LIFT_TARGET_RCV = 0.10
    RCV_LIFT_FREE = 0.15
    RCV_LIFT_SIGMA = 0.08
    LIP_DEADBAND = 0.02       # [m]
    APPR_FULL = 0.08          # [m] approach zone full up to the latch radius
    APPR_ZERO = 0.24
    # iter_16: widen the release gate a little (round 16 parked the lip at 5.2 cm, `safe` only ~0.8).
    SAFE_FULL = 0.050         # [m]
    SAFE_ZERO = 0.075         # [m] still 0.6 cm inside the env latch radius 8.1 cm
    LIP_Z_LO = 0.0
    LIP_Z_HI = 0.02
    # iter_16 (main fix a): the height budget GROWS with the tilt. At 80 deg the lip is r*sin(theta) = 4 cm
    # below the rim centre, so the cup must ride ~6 cm higher than at 45 deg; the old fixed decay taxed
    # exactly that motion and made the release pose unreachable (pour_pose was exactly 0.0 for 812 epochs).
    LIP_Z_FREE0 = 0.07
    LIP_Z_SIGMA = 0.10
    LIP_ABOVE_FLOOR = 0.35    # soft gate: tilt income no longer collapses the instant the lip dips
    HOVER_LO = 0.06
    HOVER_HI = 0.11
    HOVER_FREE0 = 0.10
    HOVER_SIGMA = 0.08
    RAISE_LO = -0.04
    # iter_16: carry plateau roughly halved (was ~20 of the 22.9 total) so the pour channel can dominate
    RAISE_W = 0.8
    CONV_FAR = 0.40
    CONV_W = 2.0
    BRING_W = 1.5
    BT_K = 10.0
    AIM_K = 15.0
    AIM_W = 2.5
    MEET_RANGE = 0.30
    MEET_RCV_W = 0.5
    MEET_SRC_W = 0.5
    APPROACH_W = 0.6
    GRASP_W = 0.8
    WRAP_SRC_W = 1.0
    WRAP_RCV_W = 0.4
    LIFT_W = 1.5
    BOTH_W = 0.8
    SRC_TILT_FREE = 0.20
    PRE_MARGIN = 0.17         # [rad] theta_pre = limit - 10 deg
    SRC_TILT_SCALE = 0.20
    SRC_TILT_SIGMA = 0.25
    SRC_UP_W_TABLE = 1.0
    SRC_UP_W_CARRY = 1.5
    LATCH_W = 1.0             # iter_16: tilt is pushed much harder, so latching must cost more
    RELEASE_ABOVE_LIMIT = math.radians(20.0)
    BAND_BELOW = 0.10
    BAND_ABOVE = 0.40
    # iter_16 (main fix b): split the tilt income again, with the weight on the FAR side. Round 16 paid
    # ~5.4/rad uniformly, so the last 35 deg (45 -> 80) was worth only ~+3.3 against a ~20 carry plateau.
    # Now 0 -> theta_pre is worth 6 and theta_pre -> release another 20, i.e. ~+7.6 for the last 30 deg,
    # plus a 2.0 overshoot ramp so the policy keeps turning once beads start falling.
    TILT_PRE_W = 6.0
    TILT_REL_W = 20.0
    TILT_OVER_W = 2.0
    TILT_OVER_RANGE = 0.45    # [rad] extra rotation past the release angle that still earns
    PRE_NEAR_FLOOR = 0.25
    PRE_NEAR_K = 10.0
    POSE_W = 8.0              # iter_16: was 3.0 and logged exactly 0.0 — the pose it asked for was unreachable
    ROT_SPEED_FREE = 1.0
    DROP_DEPTH = 0.03
    RCV_TOPPLED = 1.2
    NEAR_DIST = 0.15
    CLOSE_V_FREE = 0.15
    CLOSE_V_SCALE = 0.20
    FORCE_SCALE = 5.0
    FOREIGN_DEADBAND = 1.0
    CLEAN_DEADBAND = 0.5
    FOREIGN_W_FREE = 0.3
    FOREIGN_W_HELD = 1.5
    CLEAN_FLOOR = 0.5
    GRIP_FLOOR = 0.5          # iter_16: 0.3 -> 0.5; wrap quality dips while the cup rotates and was taxing
                              # every carry income at exactly the moment we want rotation
    RCV_TILT_FREE = 0.12
    RCV_TILT_SIGMA = 0.20
    RCV_PEN_SCALE = 0.25
    RCV_GATE_FLOOR = 0.3
    CARRY_GRASP_LO = 0.015
    CARRY_GRASP_HI = 0.05
    CARRY_FREE_LO = 0.06
    CARRY_FREE_HI = 0.10
    RCV_UP_W = 1.0
    RCV_UP_POUR_W = 1.0
    RCV_STILL_FREE = 0.05
    RCV_STILL_SCALE = 0.15
    RCV_STILL_W = 0.25
    RCV_STILL_NEAR_W = 0.25
    RCV_STILL_POUR_W = 0.50
    POST_POUR_FRAC = 0.5      # iter_16: task succeeds at half the beads, so "after the pour" starts there
    SETTLE_FRAC = 0.05
    POUR_DELTA_W = 120.0      # increments stay far above the level term below
    TRANSFER_W = 5.0          # iter_16: new, level term — holding a partly filled receiver is worth something
    SPILL_W = 50.0
    REACH_FAR = 0.26
    REACH_NEAR = 0.14
    REACH_W = 0.3
    CLOSE_W = 0.8
    CLOSE_NORM = 0.35
    PALM_RATE_W = 0.01
    HAND_RATE_W = 0.005
    SAT_LO = 0.7
    SAT_W = 0.15
    # ---- iter_17: the four video defects (cup collision, receiver crossing, pinch grasp, outward pour)
    WRAP_FULL = 3.0           # fingers with middle/distal contact that count as an envelope
    WRAP_FLOOR = 0.15         # a pinch keeps 15 % of lift/carry/tilt income: gradient survives, exploit does not
    WRAP_CNT_W = 1.2          # direct pay per hand for the envelope; comparable to LIFT_W 1.5 so it is worth closing properly
    DIR_FULL = 0.0            # pour_dir x <= 0 (sideways / toward the body): full income
    DIR_ZERO = 0.28           # ~0 just before the env void threshold 0.3
    DIR_W = 1.5               # direct shaping, same size as bring_together
    DIR_PEN_W = 1.0
    SIDE_VOID = 0.03          # env voids success below this margin
    SIDE_OK = 0.10            # required margin
    HOME_K = 10.0             # 1/m, receiver cup xy distance from its spawn
    HOME_W = 2.0              # receiver income for holding lifted + at home + still; larger than any travel income it had
    SIDE_PEN_W = 2.0
    PRELIFT_SAFE = 0.14       # [m] origin distance; nothing may pay below ~0.12 before both cups are lifted
    PRELIFT_ZERO = 0.10
    PRELIFT_W = 2.0           # exceeds grasp+wrap income of one hand, so pushing the cups together on the table loses
    CONTACT_W = 2.0           # was 1.0; per-step force penalty, active in every phase
    CONTACT_SCALE = 2.0       # [N] sharper than FORCE_SCALE: 0.5 N (the latch threshold) already costs 0.5
    HIT_EVENT_W = 5.0         # charge on the steps where the contact exceeds the latch threshold
    HIT_FORCE = 0.5           # [N] env latch threshold
    HIT_LATCH_W = 0.3         # small persistent reminder (like premature_latch); success is already void
    SUCCESS_W = 20.0          # iter_16: 10 -> 20, the plateau shrank so the terminal payoff must grow

    # ------------------------------------------------------------------ clearance factors
    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    clean_soft = CLEAN_FLOOR + (1.0 - CLEAN_FLOOR) * clean
    not_nested_f = (~ctx.cups_nested).to(dt)

    # ------------------------------------------------------------------ grasp flags / heights
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]

    # ------------------------------------------------------------------ carry phases
    rcv_carry = _carry_phase(g_rcv, h_rcv, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    src_carry = _carry_phase(g_src, h_src, CARRY_GRASP_LO, CARRY_GRASP_HI, CARRY_FREE_LO, CARRY_FREE_HI)
    # held state = grasp flag OR carried, so a thumb contact that flickers while the cup rotates is not fatal
    held_src = torch.maximum(g_src, src_carry)
    held_rcv = torch.maximum(g_rcv, rcv_carry)
    carry_raw = src_carry * rcv_carry      # geometric fact: both cups are off the table
    carry_both = carry_raw                 # re-defined below once the wrap factors exist

    # ------------------------------------------------------------------ iter_17 factors
    # (c) envelope factor per hand: wrap_count/3, floored so a pinch still sees where the income is
    wc_src = torch.clamp(ctx.src_wrap_count.to(dt) / WRAP_FULL, 0.0, 1.0)
    wc_rcv = torch.clamp(ctx.rcv_wrap_count.to(dt) / WRAP_FULL, 0.0, 1.0)
    wrapf_src = WRAP_FLOOR + (1.0 - WRAP_FLOOR) * wc_src
    wrapf_rcv = WRAP_FLOOR + (1.0 - WRAP_FLOOR) * wc_rcv
    # (d) pouring direction: 1 for x <= 0, ~0 for x >= 0.28
    dir_x = ctx.pour_dir_xy[:, 0].to(dt)
    dir_ok = 1.0 - _smoothstep((dir_x - DIR_FULL) / (DIR_ZERO - DIR_FULL))
    # (b) receiver stays on its own side, near its spawn xy
    side_ok = _smoothstep((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    rcv_home = torch.exp(-HOME_K * d_home)
    # income gate for every carry-stage term: both lifted AND both enveloped AND receiver on its side
    carry_both = carry_raw * wrapf_src * wrapf_rcv * (0.2 + 0.8 * side_ok)

    # ------------------------------------------------------------------ receiver upright gate
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up

    # ------------------------------------------------------------------ pouring-lip geometry
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    lip_off = torch.clamp(lip_xy - LIP_DEADBAND, min=0.0)
    approach_gate = 1.0 - _smoothstep((lip_xy - APPR_FULL) / (APPR_ZERO - APPR_FULL))
    safe = 1.0 - _smoothstep((lip_xy - SAFE_FULL) / (SAFE_ZERO - SAFE_FULL))   # lip close enough to release

    # ------------------------------------------------------------------ tilt angles
    tilt_mag = ctx.src_cup_tilt
    limit = ctx.premature_tilt_limit.to(dt)
    release_tilt = limit + RELEASE_ABOVE_LIMIT
    theta_pre = torch.clamp(limit - PRE_MARGIN, min=0.0)
    to_rcv = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    to_rcv_n = to_rcv / (torch.norm(to_rcv, dim=-1, keepdim=True) + 1e-6)
    a_own = torch.sum(ctx.src_cup_up[:, :2] * to_rcv_n, dim=-1)
    theta_own = torch.atan2(a_own, ctx.src_cup_up[:, 2])
    theta = torch.maximum(ctx.src_tilt_toward_rcv.to(dt), theta_own)
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), tilt_mag)

    # ------------------------------------------------------------------ source tilt corridor
    # allowed: 0.20 rad far away -> theta_pre in the approach zone -> unlimited once the lip is within 5 cm
    allowed = SRC_TILT_FREE + approach_gate * torch.clamp(theta_pre - SRC_TILT_FREE, min=0.0)
    tilt_over = torch.clamp(tilt_mag - allowed, min=0.0) * (1.0 - safe)
    src_up = torch.exp(-(tilt_over / SRC_TILT_SIGMA) ** 2)

    # ------------------------------------------------------------------ stage 1: approach + reach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = APPROACH_W * torch.exp(-4.0 * d_src)
    approach_rcv = APPROACH_W * torch.exp(-4.0 * d_rcv)
    reach_src = REACH_W * _smoothstep((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _smoothstep((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ------------------------------------------------------------------ stage 2: grasp (held state)
    prox_src = torch.exp(-6.0 * d_src)
    prox_rcv = torch.exp(-6.0 * d_rcv)
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = GRASP_W * held_src + CLOSE_W * prox_src * clos_src
    grasp_rcv = GRASP_W * held_rcv + CLOSE_W * prox_rcv * clos_rcv
    both_grasped = BOTH_W * held_src * held_rcv

    wq_src = _wrap_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                           ctx.src_palm_axes, ctx.src_cup_up, ctx.src_cup_pos, ctx.src_palm_pos)
    wq_rcv = _wrap_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                           ctx.rcv_palm_axes, ctx.rcv_cup_up, ctx.rcv_cup_pos, ctx.rcv_palm_pos)
    wrap_src = WRAP_SRC_W * held_src * wq_src + WRAP_CNT_W * held_src * wc_src
    # ^ iter_17: plus direct pay for middle/distal link contacts
    #  # wq still falls if fingers slip, so wrap keeps its gradient
    wrap_rcv = WRAP_RCV_W * held_rcv * wq_rcv + WRAP_CNT_W * held_rcv * wc_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_src = LIFT_W * held_src * grip * lift_frac_src * src_up * wrapf_src
    lift_rcv = LIFT_W * held_rcv * lift_frac_rcv * rcv_band * (0.5 + 0.5 * rcv_up) * wrapf_rcv
    both_lifted = BOTH_W * carry_both

    # ------------------------------------------------------------------ heights (budget grows with tilt)
    # iter_16 main fix (a): every height allowance is shifted up by the lip drop r*sin(theta), so rotating
    # and raising together is rewarded instead of taxed.
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    hover_free = HOVER_FREE0 + _LIP_R * sin_t
    lip_free = LIP_Z_FREE0 + _LIP_R * sin_t
    dz_eq = ctx.src_cup_pos[:, 2] + ctx.cup_mouth_z - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_eq - hover_free, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _smoothstep((dz_eq - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _smoothstep((dz_eq - RAISE_LO) / (HOVER_HI - RAISE_LO)) * hover_decay
    lip_above_raw = torch.clamp((lip_dz - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(lip_dz - lip_free, min=0.0) / LIP_Z_SIGMA) ** 2)
    # soft version used to GATE income: keeps a floor so the tilt reward survives the moment the lip dips
    lip_above = LIP_ABOVE_FLOOR + (1.0 - LIP_ABOVE_FLOOR) * lip_above_raw
    aim_z = lip_above_raw * lip_high
    z_ok = torch.maximum(hover, safe * aim_z)

    # ------------------------------------------------------------------ stage 4: raise, converge, bring, aim
    raise_src = RAISE_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide
    conv_xy = torch.clamp(1.0 - lip_off / CONV_FAR, 0.0, 1.0)
    converge = CONV_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy
    bring_together = BRING_W * carry_both * (0.3 + 0.7 * dir_ok) * not_nested_f * clean_soft * src_up * torch.exp(-BT_K * lip_off) \
        * z_ok * rcv_up_soft
    stack_gate = carry_both * not_nested_f * clear * clean_soft
    aim_xy = torch.exp(-AIM_K * lip_off)
    aim_pose = aim_xy * z_ok
    aim = AIM_W * stack_gate * grip * aim_pose * rcv_up_soft * (0.3 + 0.7 * dir_ok)
    # (d) direct direction shaping inside the approach zone, plus a bounded penalty toward the void line
    pour_dir = DIR_W * carry_both * not_nested_f * conv_xy * dir_ok
    pour_dir_pen = -DIR_PEN_W * carry_raw * approach_gate * _smoothstep((dir_x - 0.10) / 0.25)

    # ------------------------------------------------------------------ stage 5: tilt + pour pose
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    near_pre = PRE_NEAR_FLOOR + (1.0 - PRE_NEAR_FLOOR) * torch.exp(-PRE_NEAR_K * lip_off)
    tgt = torch.clamp(release_tilt, min=0.5)
    tilt_gate = carry_both * not_nested_f * clean_soft * grip * beads_left * rcv_up_soft * dir_ok * clear
    # ^ iter_17: x dir_ok (~0 at x >= 0.28) and x clear (no tilt income while the cups touch)
    # part below theta_pre: cannot spill or latch, open in the whole approach zone. Weight 6.
    part_pre = TILT_PRE_W * (torch.minimum(theta_eff, theta_pre) / tgt) * near_pre * approach_gate * hover_wide
    # part above theta_pre: only with the lip inside the safe radius. Weight 20 — this is the stage that
    # round 16 never entered, and it must outweigh the risk of losing the carry income.
    part_rel = TILT_REL_W * (torch.clamp(torch.minimum(theta_eff, tgt) - theta_pre, min=0.0) / tgt) \
        * safe * lip_above
    # keep turning past the release angle (beads keep coming out)
    part_over = TILT_OVER_W * torch.clamp((theta_eff - tgt) / TILT_OVER_RANGE, 0.0, 1.0) * safe * lip_above
    tilt = tilt_gate * (part_pre + part_rel + part_over)

    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    # iter_16: gates softened (aim_xy + lip_above instead of aim_pose + lip_above_raw) so the release pose is
    # actually attainable; weight 8 makes holding it clearly better than parking at 45 deg.
    pour_pose = POSE_W * stack_gate * grip * beads_left * safe * lip_above * aim_xy * pour_band * rcv_up * dir_ok

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer
    pour_delta = POUR_DELTA_W * torch.clamp(ctx.d_in_target, -1.0, 1.0)   # increments dominate the level term
    transferred = TRANSFER_W * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0) * not_nested_f * rcv_up
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_delta = -SPILL_W * settled * torch.clamp(ctx.d_spill, min=0.0)

    # ------------------------------------------------------------------ stage 7: hold after the pour
    post_pour = _smoothstep(ctx.bead_in_target_frac / POST_POUR_FRAC)
    hold_src = 1.5 * post_pour * held_src * wq_src
    hold_rcv = 0.5 * post_pour * held_rcv * wq_rcv * rcv_up

    # ------------------------------------------------------------------ workspace: receiver stays home (iter_17 b)
    # meet_src / meet_rcv (midline meeting) are REMOVED: they paid the receiver arm to cross over.
    v_rcv_h = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_f = 1.0 - torch.tanh(torch.clamp(v_rcv_h - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)
    home_rcv = HOME_W * rcv_carry * wrapf_rcv * rcv_up * rcv_band * side_ok * rcv_home * still_f
    side_margin = -SIDE_PEN_W * held_rcv * (1.0 - side_ok)

    # ------------------------------------------------------------------ collision / clearance
    cup_contact = -CONTACT_W * torch.tanh(ctx.cup_cup_force / CONTACT_SCALE)   # all phases, ungated
    hit_now = (ctx.cup_cup_force > HIT_FORCE).to(dt)
    # stateless stand-in for a one-time charge: paid only on the contact steps, then a small latch reminder
    cup_hit = -HIT_EVENT_W * settled * hit_now - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    w_for_src = FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src
    w_for_rcv = FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv
    hand_foreign_src = -w_for_src * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -w_for_rcv * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos
    cup_dist = torch.norm(rel, dim=-1)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_rel = ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel
    v_close = torch.sum(v_rel * u, dim=-1)
    near_f = (cup_dist < NEAR_DIST).to(dt)
    carry_any = torch.maximum(src_carry, rcv_carry)
    closing_speed = -1.0 * carry_any * near_f \
        * torch.tanh(torch.clamp(v_close - CLOSE_V_FREE, min=0.0) / CLOSE_V_SCALE)
    nested = -0.5 * ctx.cups_nested.to(dt)
    # (a) before both cups are lifted, origin distance below 0.14 m costs (full at 0.10 m)
    prelift_near = -PRELIFT_W * (1.0 - carry_raw) * _smoothstep((PRELIFT_SAFE - cup_dist) / (PRELIFT_SAFE - PRELIFT_ZERO))

    # ------------------------------------------------------------------ constraints
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src \
        * torch.tanh(tilt_over / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)

    fin_active = _smoothstep((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.1))
    pour_phase = torch.maximum(carry_both * safe * fin_active, post_pour)
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)

    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_w = RCV_STILL_W + RCV_STILL_NEAR_W * approach_gate * src_carry + RCV_STILL_POUR_W * pour_phase
    rcv_still = -still_w * rcv_carry \
        * torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)

    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ------------------------------------------------------------------ command smoothness + saturation
    da = ctx.actions - ctx.prev_actions
    palm_sq_src = torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_sq_rcv = torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_sq = torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1)
    palm_rate_src = -PALM_RATE_W * palm_sq_src
    palm_rate_rcv = -PALM_RATE_W * palm_sq_rcv
    hand_rate = -HAND_RATE_W * hand_sq
    a_abs = torch.abs(ctx.actions)
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    palm_sat_src = -SAT_W * sat_src
    palm_sat_rcv = -SAT_W * sat_rcv

    # ------------------------------------------------------------------ success bonus
    success = SUCCESS_W * ctx.success.to(dt) * clear * clean * (0.6 + 0.4 * held_src * wq_src) \
        * (0.6 + 0.4 * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "wrap_src": wrap_src,
        "wrap_rcv": wrap_rcv,
        "both_grasped": both_grasped,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "raise_src": raise_src,
        "converge": converge,
        "bring_together": bring_together,
        "aim": aim,
        "tilt": tilt,
        "pour_pose": pour_pose,
        "rot_speed": rot_speed,
        "pour_delta": pour_delta,
        "transferred": transferred,
        "spill_delta": spill_delta,
        "hold_src": hold_src,
        "hold_rcv": hold_rcv,
        "home_rcv": home_rcv,
        "side_margin": side_margin,
        "pour_dir": pour_dir,
        "pour_dir_pen": pour_dir_pen,
        "cup_hit": cup_hit,
        "prelift_near": prelift_near,
        "cup_contact": cup_contact,
        "hand_foreign_src": hand_foreign_src,
        "hand_foreign_rcv": hand_foreign_rcv,
        "closing_speed": closing_speed,
        "nested": nested,
        "upright_src": upright_src,
        "premature_latch": premature_latch,
        "upright_rcv": upright_rcv,
        "rcv_still": rcv_still,
        "drop": drop,
        "palm_rate_src": palm_rate_src,
        "palm_rate_rcv": palm_rate_rcv,
        "palm_sat_src": palm_sat_src,
        "palm_sat_rcv": palm_sat_rcv,
        "hand_rate": hand_rate,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.08e-06 · max 0.000147
bead/spill: [0.0011, 0.00721, 0.00892, 0.0105, 0.011, 0.0106, 0.0122, 0.0124, 0.0135, 0.0133]  min 0.0011 · mean 0.0107 · max 0.0145
done/drop: [0.000244, 0.000488, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000248 · max 0.0125
episode_lengths/step: [123, 784, 884, 883, 876, 871, 879, 890, 893, 895]  min 101 · mean 846 · max 898
reward/aim: [0, 0, 0, 1.85e-14, 0, 0, 0, 1.35e-08, 0, 0]  min 0 · mean 2.23e-08 · max 1.19e-05
reward/approach_rcv: [0.313, 0.394, 0.391, 0.393, 0.389, 0.385, 0.391, 0.388, 0.392, 0.39]  min 0.287 · mean 0.388 · max 0.398
reward/approach_src: [0.302, 0.321, 0.333, 0.337, 0.336, 0.335, 0.338, 0.34, 0.342, 0.343]  min 0.279 · mean 0.334 · max 0.345
reward/both_grasped: [0, 0.371, 0.621, 0.648, 0.676, 0.678, 0.683, 0.68, 0.685, 0.688]  min 0 · mean 0.607 · max 0.704
reward/both_lifted: [0, 0.00015, 0.000854, 0.00264, 0.00309, 0.00363, 0.00261, 0.00248, 0.00404, 0.00252]  min 0 · mean 0.00251 · max 0.00743
reward/bring_together: [0, 0, 0, 1.56e-22, 0, 0, 0, 1.65e-08, 0, 0]  min 0 · mean 2.61e-08 · max 1.67e-05
reward/closing_speed: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000221 · mean -2.68e-06 · max 0
reward/converge: [0, 4.71e-09, 1.68e-06, 7.31e-07, 8.51e-06, 8.75e-06, 8.85e-06, 3.48e-06, 5.34e-06, 8.77e-06]  min 0 · mean 4.28e-06 · max 0.000239
reward/cup_contact: [-0.0507, -0.00195, 0, 0, 0, 0, 0, -0.000488, 0, 0]  min -0.231 · mean -0.00199 · max 0
reward/cup_hit: [-0.144, -0.00518, 0, 0, -7.32e-05, 0, -7.32e-05, -0.00129, 0, -7.32e-05]  min -0.728 · mean -0.00594 · max 0
reward/drop: [-0.000122, -0.00354, -0.00061, -0.000244, -0.000244, -0.000244, -0.000366, -0.000244, 0, -0.000244]  min -0.024 · mean -0.0012 · max 0
reward/grasp_rcv: [0.0445, 1.02, 1.08, 1.09, 1.08, 1.07, 1.09, 1.09, 1.1, 1.1]  min 0.0133 · mean 1.05 · max 1.12
reward/grasp_src: [0.0414, 0.662, 0.928, 0.971, 0.993, 0.99, 1, 0.996, 1, 1.01]  min 0.0132 · mean 0.912 · max 1.03
reward/hand_foreign_rcv: [-0.0237, -0.0442, -0.0198, -0.0233, -0.0195, -0.0236, -0.0213, -0.0256, -0.0263, -0.0367]  min -0.0696 · mean -0.0275 · max -0.0036
reward/hand_foreign_src: [-0.0276, -0.0403, -0.0455, -0.0831, -0.053, -0.0643, -0.0766, -0.0611, -0.0645, -0.0536]  min -0.0972 · mean -0.0565 · max -0.00672
reward/hand_rate: [-0.0311, -0.0228, -0.0196, -0.0186, -0.018, -0.0172, -0.0167, -0.017, -0.0166, -0.0163]  min -0.0311 · mean -0.0187 · max -0.0156
reward/hold_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.54e-07 · max 1.6e-05
reward/hold_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.01e-07 · max 2.75e-05
reward/home_rcv: [4.55e-08, 0.0158, 0.101, 0.168, 0.222, 0.358, 0.534, 0.577, 0.655, 0.728]  min 0 · mean 0.379 · max 0.821
reward/lift_rcv: [9.27e-05, 0.165, 0.833, 1.05, 1.09, 1.09, 1.13, 1.15, 1.15, 1.12]  min 3.42e-05 · mean 0.926 · max 1.18
reward/lift_src: [1.2e-05, 0.00132, 0.00246, 0.00933, 0.00855, 0.00874, 0.00863, 0.00844, 0.00875, 0.00809]  min 1.2e-05 · mean 0.00655 · max 0.0104
reward/nested: [0, -0.000122, 0, 0, 0, 0, 0, -0.000122, 0, 0]  min -0.0208 · mean -0.000219 · max 0
reward/palm_rate_rcv: [-0.0625, -0.0469, -0.0385, -0.0357, -0.0339, -0.0311, -0.029, -0.0278, -0.0261, -0.0248]  min -0.0625 · mean -0.0335 · max -0.0227
reward/palm_rate_src: [-0.0613, -0.0463, -0.0408, -0.0352, -0.0324, -0.03, -0.0276, -0.0252, -0.0224, -0.0205]  min -0.0618 · mean -0.0321 · max -0.0193
reward/palm_sat_rcv: [-0.0598, -0.0679, -0.0652, -0.0638, -0.0613, -0.0583, -0.0502, -0.0466, -0.0419, -0.0382]  min -0.0705 · mean -0.0536 · max -0.0322
reward/palm_sat_src: [-0.0586, -0.0725, -0.068, -0.0657, -0.0596, -0.0573, -0.0578, -0.0522, -0.0508, -0.0527]  min -0.0727 · mean -0.0587 · max -0.0468
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/pour_dir: [0, 7.05e-07, 3.96e-06, 1.34e-05, 6.06e-06, 2.87e-05, 9.03e-07, 7.52e-06, 0.000147, 0.000393]  min 0 · mean 3.11e-05 · max 0.0004
reward/pour_dir_pen: [0, 0, 0, 0, 0, 0, -1.86e-05, 0, 0, 0]  min -0.000235 · mean -4.27e-06 · max 0
reward/pour_pose: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/prelift_near: [-0.00591, -0.00146, 0, 0, 0, 0, 0, -0.000488, 0, 0]  min -0.166 · mean -0.00174 · max 0
reward/premature_latch: [0, -0.158, -0.0254, -0.0117, -0.000732, -0.00122, -0.000244, -0.000488, -0.000488, -0.0022]  min -0.199 · mean -0.0175 · max 0
reward/raise_src: [0, 5.54e-08, 1.04e-05, 7.32e-06, 7.27e-05, 4.41e-05, 4.8e-05, 2.27e-05, 2.43e-05, 2.97e-05]  min 0 · mean 2.73e-05 · max 0.000248
reward/rcv_still: [-5.99e-05, -0.027, -0.121, -0.118, -0.105, -0.094, -0.0905, -0.0807, -0.08, -0.0756]  min -0.124 · mean -0.0824 · max -1.61e-05
reward/reach_rcv: [0.254, 0.294, 0.297, 0.297, 0.296, 0.296, 0.297, 0.296, 0.296, 0.296]  min 0.202 · mean 0.294 · max 0.298
reward/reach_src: [0.233, 0.271, 0.289, 0.29, 0.291, 0.291, 0.292, 0.292, 0.294, 0.295]  min 0.175 · mean 0.285 · max 0.296
reward/rot_speed: [-3.1e-05, -0.0391, -0.0595, -0.0597, -0.0514, -0.0517, -0.0464, -0.0483, -0.0485, -0.0395]  min -0.0665 · mean -0.0465 · max -3.1e-05
reward/side_margin: [0, -0.00143, -0.0013, -0.000389, -0.00148, -0.00204, -0.00221, -0.000787, -0.00206, -0.00239]  min -0.0152 · mean -0.00154 · max 0
reward/spill_delta: [0, -0.00471, -0.00365, -0.00224, 0, 0, 0, 0, -0.00477, -0.00136]  min -0.0149 · mean -0.00209 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.16e-11 · max 3.23e-08
reward/total: [0.662, 3.99, 5.91, 6.79, 7.22, 7.35, 7.67, 7.69, 7.8, 7.91]  min -0.864 · mean 6.69 · max 8.19
reward/transferred: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.29e-06 · max 0.000615
reward/upright_rcv: [-0.000182, -0.0664, -0.0743, -0.0643, -0.0485, -0.0592, -0.0247, -0.0261, -0.0325, -0.0327]  min -0.104 · mean -0.0472 · max -7.66e-05
reward/upright_src: [-1.29e-05, -0.39, -0.637, -0.247, -0.104, -0.0732, -0.0765, -0.0608, -0.0684, -0.0628]  min -0.672 · mean -0.183 · max 0
reward/wrap_rcv: [0.000123, 1.07, 1.24, 1.24, 1.24, 1.24, 1.26, 1.26, 1.27, 1.28]  min 0.000123 · mean 1.19 · max 1.3
reward/wrap_src: [4.59e-05, 0.496, 1.06, 1.17, 1.23, 1.22, 1.2, 1.14, 1.14, 1.17]  min 4.59e-05 · mean 1.04 · max 1.26
rewards/step: [-67, 3e+03, 5.15e+03, 5.9e+03, 6.29e+03, 6.43e+03, 6.71e+03, 6.95e+03, 6.98e+03, 7.09e+03]  min -119 · mean 5.81e+03 · max 7.27e+03
task/aim_dist: [0.21, 0.456, 0.475, 0.485, 0.477, 0.468, 0.46, 0.454, 0.455, 0.41]  min 0.21 · mean 0.486 · max 1.42
task/cup_collision_rate: [0.0256, 0.000977, 0, 0, 0, 0, 0, 0.000244, 0, 0]  min 0 · mean 0.000982 · max 0.114
task/cups_center_dist: [0.256, 0.42, 0.455, 0.487, 0.483, 0.476, 0.469, 0.459, 0.46, 0.416]  min 0.25 · mean 0.484 · max 1.41
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.000244, 0, 0, 0, 0, 0, 0.000244, 0, 0]  min 0 · mean 0.000438 · max 0.0415
task/rcv_cup_lift: [0.00411, 0.0334, 0.0833, 0.11, 0.114, 0.117, 0.111, 0.1, 0.11, 0.0874]  min -0.000193 · mean 0.0989 · max 0.344
task/rcv_grasped: [0, 0.777, 0.846, 0.842, 0.839, 0.844, 0.861, 0.866, 0.87, 0.875]  min 0 · mean 0.822 · max 0.896
task/rcv_hand_foreign_rate: [0.152, 0.0715, 0.0366, 0.0356, 0.0342, 0.0371, 0.0344, 0.0396, 0.0415, 0.0532]  min 0.0212 · mean 0.0461 · max 0.152
task/src_cup_lift: [0.003, 0.00116, 0.00283, 0.0043, 0.00358, 0.00355, 0.00353, 0.00352, 0.00357, 0.00335]  min 8.21e-05 · mean 0.00337 · max 0.0332
task/src_grasped: [0, 0.487, 0.786, 0.828, 0.859, 0.86, 0.865, 0.855, 0.86, 0.867]  min 0 · mean 0.773 · max 0.887
task/src_hand_foreign_rate: [0.164, 0.117, 0.0952, 0.115, 0.0808, 0.0925, 0.116, 0.095, 0.102, 0.0869]  min 0.0425 · mean 0.0975 · max 0.164
task/src_tilt_deg: [17.2, 26.5, 28.2, 13.5, 8, 7.16, 7.65, 7.44, 7.77, 7.72]  min 4.1 · mean 12.1 · max 33.1
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 18 (t2r_i17, 4096 env, FRESH start, no warm start, 727 epochs / 5.5 h): the source cup is never lifted. Measured, three ticks in a row: task/src_cup_lift 0.003 m, reward/lift_src 0.007, task/src_tilt_deg 7, episode_success 0, bead/in_target 0, adr/progress 0. In the same run the receiver IS lifted (task/rcv_cup_lift 0.09, reward/lift_rcv 1.14), both hands grasp (task/src_grasped 0.87, rcv 0.88), reward/wrap_src 1.12 of 1.2 and wrap_rcv 1.28, cup_collision_rate 0, reward/cup_hit 0, reward/cup_contact 0. The only term still growing is reward/home_rcv (0.32 -> 0.61): the policy settles into "wrap both cups, lift only the receiver a little and hold it still at home".
FACT about lift history (read from the code): iter_11, the last FRESH run, used lift_src = 2.0 * g_src * grip * lift_frac_src * src_up with LIFT_TARGET_SRC 0.25 and from scratch reached task/src_cup_lift 0.22 (max 0.30) with src_grasped 0.86. iter_16 lowered the weight to LIFT_W 1.5 and raised LIFT_TARGET_SRC to 0.28 while training from a warm-started policy that already lifted; iter_17 kept those values and multiplied one more gate (wrapf_src), and was the first fresh run with them.
THE NEXT RUN IS ALSO FRESH (random policy, no checkpoint). REQUIRED: the source-cup lift must be learnable from scratch. Restore at least the iter_11 strength for the source lift (weight >= 2.0, LIFT_TARGET_SRC 0.25) and make sure the very first centimetres of source height pay clearly more than standing still with the cup wrapped on the table - no product of many sub-1 gates on the early lift, and no stationary income (wrap, home_rcv, grasp bonuses) that is larger than what lifting the source adds. home_rcv must not pay more for an idle receiver than the source side earns by lifting. Keep everything that worked in iter_17: cup_contact / cup_hit penalties (collision stayed 0), receiver staying on its own side (side_margin), wrap pay, pour-direction gate, and the iter_16 tilt / pour_pose / pour_delta structure.
