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
    LIFT_TARGET_SRC = 0.25    # iter_18: back to the iter_11 value that was learnable from scratch
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
    LIFT_W = 1.5              # receiver lift only
    LIFT_SRC_W = 3.0          # iter_18: source lift, stronger than iter_11 (2.0) and with no gate product
    LIFTOFF_H = 0.03          # [m] e-fold height of the early lift-off part: first cm pay the most
    LIFT_SRC_UP_FLOOR = 0.5   # tilt may halve the source lift income, never zero it
    LIFT_WRAP_BONUS = 0.25    # envelope adds up to +25 % on the lift, it no longer gates it
    TABLE_WRAP_FRAC = 0.4     # share of wrap pay available with the cup still on the table
    HOME_SRC_FLOOR = 0.2      # share of home_rcv available while the source cup is not carried
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
    HOME_W = 0.8              # iter_18: 2.0 -> 0.8, and scaled by the source carry state (see home_rcv)
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
    # iter_18: only TABLE_WRAP_FRAC of the wrap pay is stationary income; the rest needs the cup off the table
    wrap_lift_src = TABLE_WRAP_FRAC + (1.0 - TABLE_WRAP_FRAC) * src_carry
    wrap_lift_rcv = TABLE_WRAP_FRAC + (1.0 - TABLE_WRAP_FRAC) * rcv_carry
    wrap_src = (WRAP_SRC_W * held_src * wq_src + WRAP_CNT_W * held_src * wc_src) * wrap_lift_src
    # ^ iter_17: plus direct pay for middle/distal link contacts
    #  # wq still falls if fingers slip, so wrap keeps its gradient
    wrap_rcv = (WRAP_RCV_W * held_rcv * wq_rcv + WRAP_CNT_W * held_rcv * wc_rcv) * wrap_lift_rcv
    grip = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * wq_src

    # ------------------------------------------------------------------ stage 3: lift
    lift_frac_src = torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    lift_frac_rcv = torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    # iter_18: no product of sub-1 gates. Half of the income is a steep lift-off part (63 % at 3 cm),
    # half is linear to LIFT_TARGET_SRC so going higher keeps paying.
    h_src_pos = torch.clamp(h_src, min=0.0)
    liftoff_src = 1.0 - torch.exp(-h_src_pos / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * lift_frac_src
    src_up_soft = LIFT_SRC_UP_FLOOR + (1.0 - LIFT_SRC_UP_FLOOR) * src_up
    lift_src = LIFT_SRC_W * held_src * lift_shape_src * src_up_soft * (1.0 + LIFT_WRAP_BONUS * wc_src)
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
    home_src_f = HOME_SRC_FLOOR + (1.0 - HOME_SRC_FLOOR) * src_carry
    home_rcv = HOME_W * home_src_f * rcv_carry * wrapf_rcv * rcv_up * rcv_band * side_ok * rcv_home * still_f
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

adr/progress: [0, 0, 0, 0, 0, 0, 0.2, 0.4, 0.633, 0.833]  min 0 · mean 0.26 · max 1
bead/in_target: [0, 0, 0, 0, 6.1e-05, 0.506, 0.649, 0.69, 0.699, 0.726]  min 0 · mean 0.368 · max 0.747
bead/spill: [0.00126, 0.00974, 0.0122, 0.0125, 0.0125, 0.0356, 0.0441, 0.0455, 0.0309, 0.0357]  min 0.00126 · mean 0.0299 · max 0.0868
done/drop: [0.000244, 0, 0, 0, 0, 0, 0, 0, 0, 0.000244]  min 0 · mean 0.0001 · max 0.0105
episode_lengths/step: [123, 885, 898, 897, 883, 890, 897, 883, 887, 887]  min 101 · mean 877 · max 899
reward/aim: [0, 0.00606, 0.659, 1.53, 1.68, 1.54, 1.55, 1.5, 1.43, 1.56]  min 0 · mean 1.19 · max 1.72
reward/approach_rcv: [0.313, 0.388, 0.376, 0.373, 0.376, 0.378, 0.381, 0.367, 0.361, 0.364]  min 0.287 · mean 0.374 · max 0.403
reward/approach_src: [0.302, 0.401, 0.4, 0.396, 0.389, 0.371, 0.372, 0.374, 0.378, 0.376]  min 0.287 · mean 0.38 · max 0.406
reward/both_grasped: [0, 0.681, 0.685, 0.699, 0.69, 0.691, 0.691, 0.685, 0.69, 0.694]  min 0 · mean 0.677 · max 0.707
reward/both_lifted: [0, 0.65, 0.641, 0.675, 0.667, 0.66, 0.66, 0.64, 0.645, 0.655]  min 0 · mean 0.628 · max 0.682
reward/bring_together: [0, 0.0202, 0.547, 1.06, 1.14, 1.1, 1.08, 1.03, 1.02, 1.1]  min 0 · mean 0.846 · max 1.16
reward/closing_speed: [0, 0, -0.00452, -0.00442, -0.00142, -0.000393, -5.52e-05, -0.000146, -0.000105, -0.000386]  min -0.00828 · mean -0.00112 · max 0
reward/converge: [0, 0.187, 1.2, 1.55, 1.57, 0.806, 0.485, 0.349, 0.393, 0.272]  min 0 · mean 0.659 · max 1.62
reward/cup_contact: [-0.0477, 0, -0.0268, -0.0604, -0.0279, -0.0176, -0.0108, -0.00734, -0.00901, -0.00613]  min -0.236 · mean -0.0144 · max 0
reward/cup_hit: [-0.134, 0, -0.145, -0.357, -0.249, -0.15, -0.0881, -0.0484, -0.081, -0.0546]  min -0.749 · mean -0.112 · max 0
reward/drop: [-0.000122, -0.00061, -0.000366, -0.000244, 0, 0, 0, -0.000122, -0.000244, -0.000244]  min -0.0242 · mean -0.000386 · max 0
reward/grasp_rcv: [0.0445, 1.07, 1.05, 1.07, 1.06, 1.06, 1.07, 1.04, 1.04, 1.05]  min 0.0139 · mean 1.05 · max 1.11
reward/grasp_src: [0.0413, 1.09, 1.1, 1.1, 1.08, 1.05, 1.06, 1.06, 1.07, 1.07]  min 0.0135 · mean 1.06 · max 1.12
reward/hand_foreign_rcv: [-0.0278, -0.0107, -0.0126, -0.00855, -0.0124, -0.0102, -0.00764, -0.0161, -0.0151, -0.0155]  min -0.0816 · mean -0.0143 · max -0.00445
reward/hand_foreign_src: [-0.0253, -0.0118, -0.0291, -0.0269, -0.0279, -0.0161, -0.0107, -0.0182, -0.0217, -0.013]  min -0.0997 · mean -0.0209 · max -0.00561
reward/hand_rate: [-0.0306, -0.0189, -0.0197, -0.0199, -0.0174, -0.0178, -0.0128, -0.0119, -0.0128, -0.0143]  min -0.0311 · mean -0.0167 · max -0.0113
reward/hold_rcv: [0, 0, 0, 0, 2.82e-05, 0.204, 0.264, 0.235, 0.226, 0.248]  min 0 · mean 0.131 · max 0.273
reward/hold_src: [0, 0, 0, 0, 9.34e-05, 0.578, 0.788, 0.863, 0.809, 0.845]  min 0 · mean 0.435 · max 0.895
reward/home_rcv: [1.9e-08, 0.158, 0.2, 0.252, 0.29, 0.296, 0.283, 0.281, 0.253, 0.283]  min 1.9e-08 · mean 0.242 · max 0.307
reward/lift_rcv: [1.04e-05, 1.17, 1.23, 1.28, 1.23, 1.21, 1.14, 1.14, 1.14, 1.22]  min 1.04e-05 · mean 1.16 · max 1.29
reward/lift_src: [0.000111, 2.93, 3.09, 3.11, 3.14, 3.15, 3.14, 3.16, 3.16, 3.19]  min 0.000111 · mean 2.98 · max 3.24
reward/nested: [-0.000122, 0, 0, 0, 0, -0.000122, 0, -0.000122, -0.000122, 0]  min -0.0173 · mean -9.77e-05 · max 0
reward/palm_rate_rcv: [-0.0618, -0.0393, -0.0307, -0.0288, -0.0256, -0.0204, -0.0174, -0.0167, -0.0171, -0.0185]  min -0.0618 · mean -0.025 · max -0.016
reward/palm_rate_src: [-0.0616, -0.0361, -0.0296, -0.0213, -0.0154, -0.00701, -0.00338, -0.00279, -0.00237, -0.00187]  min -0.0622 · mean -0.0149 · max -0.00149
reward/palm_sat_rcv: [-0.0592, -0.0652, -0.0548, -0.0506, -0.0482, -0.0499, -0.0493, -0.0508, -0.0536, -0.0504]  min -0.068 · mean -0.0527 · max -0.0466
reward/palm_sat_src: [-0.0593, -0.074, -0.0717, -0.0783, -0.0915, -0.0982, -0.108, -0.105, -0.11, -0.114]  min -0.121 · mean -0.094 · max -0.0585
reward/pour_delta: [0, 0, 0, 0, 0, 0.116, 0.148, 0.127, 0.0915, 0.106]  min 0 · mean 0.0706 · max 0.196
reward/pour_dir: [0, 0.16, 0.922, 1.2, 1.22, 1.2, 1.2, 1.17, 1.17, 1.2]  min 0 · mean 0.993 · max 1.24
reward/pour_dir_pen: [0, 0, -0.00264, -0.000417, -0.000796, -0.000167, -0.000878, -0.000457, -0.000658, -0.00052]  min -0.00612 · mean -0.000831 · max 0
reward/pour_pose: [0, 0, 0, 0, 0.00113, 3.44, 4.15, 4.22, 4.05, 4.49]  min 0 · mean 2.27 · max 4.65
reward/prelift_near: [-0.00385, 0, -0.000455, -6.62e-06, -3.85e-08, -0.000365, 0, -0.00139, -0.000491, -3.84e-05]  min -0.151 · mean -0.000816 · max 0
reward/premature_latch: [0, -0.000244, -0.000488, -0.000244, -0.00146, -0.0332, -0.0786, -0.0591, -0.0713, -0.0491]  min -0.0925 · mean -0.0335 · max 0
reward/raise_src: [0, 0.478, 0.591, 0.647, 0.638, 0.335, 0.202, 0.147, 0.165, 0.114]  min 0 · mean 0.32 · max 0.658
reward/rcv_still: [-1.73e-05, -0.116, -0.151, -0.149, -0.158, -0.293, -0.318, -0.312, -0.326, -0.335]  min -0.354 · mean -0.24 · max -1.73e-05
reward/reach_rcv: [0.255, 0.297, 0.296, 0.297, 0.297, 0.297, 0.297, 0.295, 0.294, 0.296]  min 0.203 · mean 0.296 · max 0.298
reward/reach_src: [0.232, 0.297, 0.298, 0.297, 0.297, 0.296, 0.297, 0.296, 0.297, 0.296]  min 0.201 · mean 0.295 · max 0.298
reward/rot_speed: [-1.9e-07, -0.0486, -0.0282, -0.0257, -0.0327, -0.0366, -0.0384, -0.0422, -0.0398, -0.0354]  min -0.127 · mean -0.0396 · max -1.9e-07
reward/side_margin: [-2.78e-05, -0.00364, -0.0177, -0.0253, -0.0101, -0.00129, -0.00353, -0.0015, -0.004, -0.00177]  min -0.0409 · mean -0.00745 · max 0
reward/spill_delta: [-0.00153, 0, -0.00275, 0, 0, -0.0063, -0.00292, -0.00638, -0.00479, -0.00546]  min -0.0174 · mean -0.00323 · max 0
reward/success: [0, 0, 0, 0, 0, 4.77, 8.16, 10.6, 8.68, 10.4]  min 0 · mean 4.69 · max 11.3
reward/tilt: [0, 0, 0.0855, 0.752, 1.72, 5.5, 6.11, 6.11, 6.16, 6.36]  min 0 · mean 3.61 · max 6.55
reward/total: [0.673, 12.4, 15.6, 18.4, 19.7, 33.6, 38.8, 41, 38.9, 41.8]  min -0.861 · mean 28.2 · max 42.9
reward/transferred: [0, 0, 0, 0, 0.000282, 2.53, 3.23, 3.37, 3.45, 3.59]  min 0 · mean 1.82 · max 3.69
reward/upright_rcv: [-5.44e-05, -0.0618, -0.0388, -0.0378, -0.0235, -0.0364, -0.0367, -0.0967, -0.045, -0.0531]  min -0.292 · mean -0.0553 · max -5.44e-05
reward/upright_src: [0, -0.021, -0.00657, -0.00395, -0.00332, -0.031, -0.0678, -0.0735, -0.0722, -0.0492]  min -0.672 · mean -0.0569 · max 0
reward/wrap_rcv: [2.05e-06, 1.25, 1.23, 1.27, 1.26, 1.26, 1.27, 1.21, 1.21, 1.24]  min 2.05e-06 · mean 1.22 · max 1.29
reward/wrap_src: [9.34e-06, 1.7, 1.72, 1.77, 1.71, 1.63, 1.67, 1.67, 1.64, 1.64]  min 9.34e-06 · mean 1.62 · max 1.77
rewards/step: [-44.7, 1.1e+04, 1.39e+04, 1.63e+04, 1.74e+04, 2.99e+04, 3.53e+04, 3.63e+04, 3.49e+04, 3.69e+04]  min -72.3 · mean 2.49e+04 · max 3.88e+04
task/aim_dist: [0.211, 0.44, 0.205, 0.158, 0.149, 0.131, 0.129, 0.135, 0.127, 0.136]  min 0.121 · mean 0.198 · max 2.26
task/cup_collision_rate: [0.0251, 0, 0.0142, 0.031, 0.0144, 0.00928, 0.00537, 0.00391, 0.00488, 0.00342]  min 0 · mean 0.00749 · max 0.115
task/cups_center_dist: [0.257, 0.439, 0.209, 0.178, 0.185, 0.234, 0.252, 0.265, 0.261, 0.274]  min 0.176 · mean 0.274 · max 2.26
task/episode_success: [0, 0, 0, 0, 0, 0.273, 0.457, 0.59, 0.49, 0.59]  min 0 · mean 0.267 · max 0.658
task/nested_rate: [0.000244, 0, 0, 0, 0, 0.000244, 0, 0.000244, 0.000244, 0]  min 0 · mean 0.000195 · max 0.0347
task/rcv_cup_lift: [0.00424, 0.137, 0.114, 0.106, 0.103, 0.125, 0.134, 0.128, 0.13, 0.118]  min -0.000362 · mean 0.12 · max 0.548
task/rcv_grasped: [0, 0.84, 0.842, 0.868, 0.856, 0.86, 0.862, 0.854, 0.858, 0.864]  min 0 · mean 0.85 · max 0.882
task/rcv_hand_foreign_rate: [0.164, 0.0251, 0.0298, 0.0212, 0.0288, 0.0239, 0.0237, 0.0361, 0.032, 0.0308]  min 0.0178 · mean 0.0321 · max 0.164
task/src_cup_lift: [0.00285, 0.181, 0.202, 0.199, 0.211, 0.293, 0.327, 0.334, 0.334, 0.339]  min -1.02e-05 · mean 0.26 · max 0.354
task/src_grasped: [0, 0.862, 0.863, 0.871, 0.86, 0.86, 0.867, 0.855, 0.864, 0.859]  min 0 · mean 0.848 · max 0.888
task/src_hand_foreign_rate: [0.147, 0.0278, 0.0618, 0.0564, 0.0425, 0.0242, 0.0156, 0.0264, 0.0332, 0.0256]  min 0.011 · mean 0.0389 · max 0.234
task/src_tilt_deg: [16.7, 7.43, 5.26, 13.6, 28.8, 77.1, 94.4, 99.5, 102, 109]  min 4.27 · mean 61 · max 114
task/success_now: [0, 0, 0, 0, 0, 0.28, 0.466, 0.6, 0.498, 0.598]  min 0 · mean 0.271 · max 0.666

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 19 (t2r_i18, 4096 env, FRESH, epoch 2500+): episode_success recent ~0.54 at ADR 30, in_target ~0.70. The user ACCEPTED the behaviour in the play video except ONE defect: arm tremor. Keep the iter_18 structure and all its weights; change only what is listed below.

Measured on a 64-env deterministic trace of ep_2500 at ADR 30 (900 steps, 60 Hz policy):
- Action layout is 18 dims: 0-5 source palm 6D (xyz, 3 rotations), 6-8 source hand, 9-14 receiver palm 6D, 15-17 receiver hand. Dims 8, 9, 10, 13, 14, 17 flip sign between consecutive steps on 29-49 % of steps during steps 200-850; all other dims under 1 % (dims 6, 7, 16 are 4-10 %).
- Receiver palm command second difference rms is 0.07-0.16 (normalised units) vs 0.02-0.03 for the source arm. Receiver arm joint speed rms while it should only hold (steps 400-850) is 0.2-0.5 rad/s, about twice the source arm.
- Current smoothness terms are small next to the success income (+9.3): rcv_still recent -0.34, palm_rate_rcv -0.017, palm_rate_src -0.002.
- Source tilt peaks at ~155 deg for every fill level (6-20 beads) and first bead entry happens at 124-140 deg, although ctx.premature_tilt_limit is 52-63 deg: the policy flips the cup over instead of metering the tilt. Spill is 0.10-0.11 in the 6-8 and 16-20 bead bins, 0.03-0.04 in between.

ENV CHANGE for the next run (already in code): after the palm EMA (alpha 0.25) the palm command is slew-limited to 0.03 normalised units per step. ctx.actions / ctx.prev_actions are still the RAW policy actions, so the reward can see the chatter the filter hides.

REQUIRED in the new reward:
(a) A sign-alternation penalty on raw actions: charge relu(-(actions * prev_actions)) (product negative = sign flip), summed over all 18 dims, with a weight large enough that the measured chatter (dims with |a| step delta rms 0.4-0.7) costs on the order of 1-2 per step, i.e. comparable to a carry-phase income term, not 0.02.
(b) Raise the action-rate penalties (palm_rate_src, palm_rate_rcv and the hand dims) by roughly 10x and apply them in all phases including after success; the receiver must be charged while holding still.
(c) Saturate tilt income: no additional tilt income beyond ctx.premature_tilt_limit + 0.35 rad (20 deg), and a mild penalty for tilt beyond that + 0.35 rad more. Do not reduce income below that angle.
Everything else (cup_contact, cup_hit, prelift_near, side_margin, wrap, pour_dir gate, lift, home_rcv, pour_pose, pour_delta, transferred, success) stays as in iter_18. This run WARM-STARTS from t2r_i18 ep_2500, so keep reward scale of unchanged terms identical.
