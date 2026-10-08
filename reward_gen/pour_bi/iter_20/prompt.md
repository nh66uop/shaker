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
    # iter_19: x10, applied in all phases; sized like a carry-phase income term (~1e-2..1e-1 per step of chatter)
    PALM_RATE_W = 0.1
    HAND_RATE_W = 0.05
    # iter_19: no tilt income past limit + 0.35 rad; mild penalty for rotating beyond it (cup flip)
    TILT_CAP_ABOVE = 0.35
    OVERTILT_W = 2.0
    FLIP_W = 1.0              # 6 dims alternating +-0.5 -> 6*0.25*1.0 = 1.5 per step
    THETA_REL_BASE = 1.25
    THETA_REL_SPAN = 0.60
    TILT_PEAK_ABOVE = 0.10
    OVERTILT_ABOVE = 0.25
    FLOW_CAP = 0.02
    FLOW_W = 20.0
    OVERTILT_SCALE = 0.3
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
    f_rem = torch.clamp(ctx.bead_fill_level.to(dt) * ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    theta_rel = THETA_REL_BASE + (1.0 - f_rem) * THETA_REL_SPAN
    tilt_peak = theta_rel + TILT_PEAK_ABOVE          # income peaks here, flat beyond
    theta_pk = torch.minimum(theta_eff, tilt_peak)
    part_rel = TILT_REL_W * (torch.clamp(torch.minimum(theta_pk, tgt) - theta_pre, min=0.0) / tgt) \
        * safe * lip_above
    # keep turning past the release angle (beads keep coming out)
    tilt_cap = torch.maximum(tilt_peak, tgt + 0.05)
    over_range = torch.clamp(tilt_cap - tgt, min=0.05)
    part_over = TILT_OVER_W * torch.clamp((theta_pk - tgt) / over_range, 0.0, 1.0) \
        * safe * lip_above
    overtilt = -OVERTILT_W * torch.tanh(torch.clamp(tilt_mag - (theta_rel + OVERTILT_ABOVE), min=0.0) / OVERTILT_SCALE)
    tilt = tilt_gate * (part_pre + part_rel + part_over)

    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    # iter_16: gates softened (aim_xy + lip_above instead of aim_pose + lip_above_raw) so the release pose is
    # actually attainable; weight 8 makes holding it clearly better than parking at 45 deg.
    pour_pose = POSE_W * stack_gate * grip * beads_left * safe * lip_above * aim_xy * pour_band * rcv_up * dir_ok

    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ------------------------------------------------------------------ stage 6: bead transfer
    d_in = ctx.d_in_target.to(dt)
    flow_excess = -FLOW_W * torch.relu(d_in - FLOW_CAP)
    pour_delta = POUR_DELTA_W * torch.clamp(d_in, -1.0, FLOW_CAP)   # increments dominate the level term
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
    act_flip = -FLIP_W * torch.sum(torch.relu(-(ctx.actions * ctx.prev_actions)), dim=-1)
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
        "overtilt": overtilt,
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
        "act_flip": act_flip,
        "flow_excess": flow_excess,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0.0333, 0.0667, 0.1, 0.133, 0.167, 0.2, 0.267, 0.3, 0.333]  min 0 · mean 0.178 · max 0.367
bead/in_target: [0, 0.718, 0.737, 0.717, 0.743, 0.697, 0.696, 0.73, 0.756, 0.743]  min 0 · mean 0.695 · max 0.767
bead/spill: [0.00313, 0.0464, 0.0364, 0.0559, 0.0389, 0.0557, 0.0436, 0.0469, 0.0358, 0.0419]  min 0.00094 · mean 0.0454 · max 0.0862
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000233 · max 0.0493
episode_lengths/step: [128, 864, 891, 895, 885, 881, 880, 889, 888, 896]  min 128 · mean 864 · max 899
reward/act_flip: [-0.25, -0.473, -0.46, -0.427, -0.418, -0.389, -0.388, -0.363, -0.357, -0.346]  min -0.529 · mean -0.4 · max -0.25
reward/aim: [0, 1.45, 1.52, 1.56, 1.6, 1.55, 1.52, 1.66, 1.64, 1.66]  min 0 · mean 1.52 · max 1.72
reward/approach_rcv: [0.354, 0.369, 0.38, 0.372, 0.375, 0.37, 0.369, 0.37, 0.373, 0.378]  min 0.332 · mean 0.371 · max 0.382
reward/approach_src: [0.349, 0.369, 0.371, 0.37, 0.367, 0.374, 0.376, 0.378, 0.37, 0.373]  min 0.338 · mean 0.37 · max 0.38
reward/both_grasped: [0, 0.691, 0.709, 0.697, 0.71, 0.7, 0.717, 0.7, 0.702, 0.701]  min 0 · mean 0.677 · max 0.717
reward/both_lifted: [0, 0.621, 0.662, 0.655, 0.671, 0.651, 0.634, 0.648, 0.653, 0.659]  min 0 · mean 0.626 · max 0.674
reward/bring_together: [0, 1.03, 1.09, 1.11, 1.14, 1.09, 1.07, 1.13, 1.13, 1.14]  min 0 · mean 1.06 · max 1.17
reward/closing_speed: [0, -0.0015, -0.000803, -0.00146, -0.00142, -0.000763, -0.00178, -0.000288, -0.000229, -0.000461]  min -0.004 · mean -0.000873 · max 0
reward/converge: [0, 0.365, 0.419, 0.428, 0.363, 0.376, 0.394, 0.44, 0.439, 0.423]  min 0 · mean 0.381 · max 0.498
reward/cup_contact: [0, -0.0117, -0.00803, -0.00719, -0.0103, -0.00183, -0.0069, -0.00626, -0.00572, -0.00603]  min -0.0243 · mean -0.00707 · max 0
reward/cup_hit: [0, -0.0941, -0.0834, -0.0601, -0.0781, -0.0542, -0.0892, -0.0915, -0.109, -0.12]  min -0.133 · mean -0.0824 · max 0
reward/drop: [0, -0.000122, 0, -0.000244, -0.000122, -0.000122, 0, 0, -0.000122, 0]  min -0.0249 · mean -0.000214 · max 0
reward/flow_excess: [0, -0.0116, -0.0131, -0.0106, -0.012, -0.0104, -0.014, -0.0122, -0.0213, -0.0159]  min -0.0698 · mean -0.0165 · max 0
reward/grasp_rcv: [0.0854, 1.05, 1.09, 1.06, 1.08, 1.07, 1.08, 1.07, 1.08, 1.09]  min 0.0854 · mean 1.04 · max 1.1
reward/grasp_src: [0.0818, 1.08, 1.08, 1.06, 1.07, 1.07, 1.09, 1.08, 1.07, 1.07]  min 0.0818 · mean 1.05 · max 1.09
reward/hand_foreign_rcv: [0, -0.0116, -0.00948, -0.00892, -0.0107, -0.0126, -0.0112, -0.0156, -0.0132, -0.0159]  min -0.0296 · mean -0.0127 · max 0
reward/hand_foreign_src: [-6.75e-05, -0.0418, -0.0125, -0.013, -0.0151, -0.0118, -0.0144, -0.0109, -0.015, -0.0131]  min -0.199 · mean -0.0215 · max -6.75e-05
reward/hand_rate: [-0.081, -0.125, -0.12, -0.113, -0.11, -0.106, -0.108, -0.102, -0.0993, -0.0954]  min -0.138 · mean -0.108 · max -0.081
reward/hold_rcv: [0, 0.259, 0.279, 0.261, 0.269, 0.244, 0.238, 0.252, 0.258, 0.268]  min 0 · mean 0.247 · max 0.288
reward/hold_src: [0, 0.807, 0.815, 0.831, 0.845, 0.851, 0.841, 0.92, 0.923, 0.92]  min 0 · mean 0.828 · max 0.946
reward/home_rcv: [0, 0.218, 0.227, 0.243, 0.242, 0.236, 0.233, 0.255, 0.25, 0.261]  min 0 · mean 0.234 · max 0.276
reward/lift_rcv: [0, 1.07, 1.12, 1.14, 1.13, 1.13, 1.11, 1.16, 1.15, 1.17]  min 0 · mean 1.1 · max 1.21
reward/lift_src: [0, 3.21, 3.24, 3.19, 3.25, 3.2, 3.25, 3.23, 3.22, 3.22]  min 0 · mean 3.13 · max 3.28
reward/nested: [0, 0, -0.000122, 0, -0.000122, 0, 0, 0, 0, 0]  min -0.0022 · mean -0.000105 · max 0
reward/overtilt: [0, -0.788, -0.763, -0.774, -0.9, -0.768, -0.642, -0.441, -0.53, -0.59]  min -0.968 · mean -0.675 · max 0
reward/palm_rate_rcv: [-0.0722, -0.101, -0.104, -0.0992, -0.0991, -0.096, -0.0955, -0.0925, -0.0931, -0.0918]  min -0.118 · mean -0.0958 · max -0.0722
reward/palm_rate_src: [-0.0498, -0.0166, -0.0162, -0.0171, -0.0131, -0.0165, -0.0178, -0.0191, -0.0187, -0.0206]  min -0.0498 · mean -0.0182 · max -0.013
reward/palm_sat_rcv: [-0.0465, -0.0467, -0.043, -0.0474, -0.0443, -0.0445, -0.0446, -0.0453, -0.0443, -0.0432]  min -0.0524 · mean -0.0451 · max -0.0402
reward/palm_sat_src: [-0.0122, -0.112, -0.112, -0.112, -0.118, -0.114, -0.112, -0.107, -0.107, -0.106]  min -0.118 · mean -0.109 · max -0.0122
reward/pour_delta: [0, 0.0104, 0.0182, 0.011, 0.017, 0.0129, 0.0199, 0.017, 0.0246, 0.0211]  min 0 · mean 0.0208 · max 0.085
reward/pour_dir: [0, 1.14, 1.21, 1.21, 1.24, 1.2, 1.16, 1.2, 1.21, 1.22]  min 0 · mean 1.15 · max 1.24
reward/pour_dir_pen: [0, -0.00148, -0.000611, -0.00119, -0.000951, -0.000446, -0.000745, -0.000698, -0.000488, -0.00136]  min -0.0038 · mean -0.000798 · max 0
reward/pour_pose: [0, 4.08, 4.27, 4.11, 4.54, 4.27, 4.24, 4.54, 4.64, 4.7]  min 0 · mean 4.22 · max 4.92
reward/prelift_near: [0, -0.00193, -0.00065, -0.00156, -0.00061, -0.000287, -8.36e-05, -1.12e-06, -0.000345, -4.69e-07]  min -0.0311 · mean -0.00156 · max 0
reward/premature_latch: [0, -0.1, -0.0825, -0.0688, -0.072, -0.0693, -0.0554, -0.0486, -0.063, -0.0618]  min -0.113 · mean -0.0675 · max 0
reward/raise_src: [0, 0.149, 0.172, 0.176, 0.149, 0.155, 0.162, 0.178, 0.178, 0.172]  min 0 · mean 0.155 · max 0.204
reward/rcv_still: [0, -0.395, -0.412, -0.388, -0.387, -0.387, -0.374, -0.364, -0.38, -0.376]  min -0.412 · mean -0.369 · max 0
reward/reach_rcv: [0.3, 0.288, 0.296, 0.296, 0.296, 0.296, 0.296, 0.295, 0.296, 0.297]  min 0.24 · mean 0.293 · max 0.3
reward/reach_src: [0.298, 0.296, 0.296, 0.296, 0.296, 0.296, 0.297, 0.297, 0.296, 0.296]  min 0.271 · mean 0.295 · max 0.298
reward/rot_speed: [0, -0.0373, -0.0382, -0.035, -0.0346, -0.0372, -0.0458, -0.0322, -0.031, -0.0311]  min -0.0699 · mean -0.0357 · max 0
reward/side_margin: [0, -0.00622, -0.00217, -0.00229, -0.00293, -0.0023, -0.00152, -0.000742, -0.000704, -0.000102]  min -0.0304 · mean -0.00206 · max 0
reward/spill_delta: [-0.0106, -0.00529, -9.77e-05, -0.00122, -0.00552, -0.00892, -0.00618, -0.00153, -0.00503, -0.00111]  min -0.0265 · mean -0.0047 · max 0
reward/success: [0, 8.52, 8.92, 10.1, 9.77, 9.36, 8.32, 8.9, 8.03, 7.3]  min 0 · mean 8.53 · max 10.6
reward/tilt: [0, 5.78, 6.29, 6.06, 6.36, 6.01, 6.15, 6.55, 6.7, 6.71]  min 0 · mean 6.07 · max 6.88
reward/total: [0.945, 36.6, 38.6, 39.3, 39.9, 38.6, 37.8, 39.9, 39.3, 38.6]  min 0.945 · mean 37.4 · max 41.1
reward/transferred: [0, 3.51, 3.63, 3.54, 3.65, 3.43, 3.43, 3.61, 3.73, 3.67]  min 0 · mean 3.42 · max 3.78
reward/upright_rcv: [0, -0.106, -0.0716, -0.0757, -0.0778, -0.0698, -0.0599, -0.0464, -0.0611, -0.0524]  min -0.154 · mean -0.0687 · max 0
reward/upright_src: [0, -0.0698, -0.06, -0.0474, -0.0508, -0.0423, -0.0312, -0.0314, -0.0434, -0.044]  min -0.163 · mean -0.0485 · max 0
reward/wrap_rcv: [0, 1.2, 1.28, 1.24, 1.28, 1.23, 1.22, 1.23, 1.24, 1.25]  min 0 · mean 1.19 · max 1.28
reward/wrap_src: [0, 1.64, 1.65, 1.64, 1.66, 1.67, 1.7, 1.71, 1.69, 1.7]  min 0 · mean 1.62 · max 1.73
rewards/step: [89.6, 2.96e+04, 3.42e+04, 3.49e+04, 3.5e+04, 3.46e+04, 3.43e+04, 3.49e+04, 3.31e+04, 3.47e+04]  min 89.6 · mean 3.26e+04 · max 3.72e+04
task/aim_dist: [0.327, 0.154, 0.12, 0.13, 0.121, 0.128, 0.129, 0.134, 0.131, 0.13]  min 0.117 · mean 0.135 · max 0.327
task/cup_collision_rate: [0, 0.00635, 0.00415, 0.00366, 0.00537, 0.000977, 0.00366, 0.00317, 0.00317, 0.00317]  min 0 · mean 0.00366 · max 0.0127
task/cups_center_dist: [0.323, 0.289, 0.258, 0.262, 0.261, 0.259, 0.255, 0.26, 0.263, 0.261]  min 0.245 · mean 0.264 · max 0.369
task/episode_success: [0, 0.495, 0.511, 0.581, 0.558, 0.532, 0.466, 0.495, 0.444, 0.409]  min 0 · mean 0.483 · max 0.608
task/nested_rate: [0, 0, 0.000244, 0, 0.000244, 0, 0, 0, 0, 0]  min 0 · mean 0.000211 · max 0.00439
task/rcv_cup_lift: [0.00218, 0.155, 0.141, 0.134, 0.142, 0.13, 0.128, 0.13, 0.135, 0.134]  min 0.00218 · mean 0.13 · max 0.17
task/rcv_grasped: [0, 0.84, 0.875, 0.86, 0.88, 0.871, 0.888, 0.875, 0.876, 0.879]  min 0 · mean 0.843 · max 0.894
task/rcv_hand_foreign_rate: [0, 0.0349, 0.022, 0.0222, 0.0283, 0.0256, 0.0239, 0.0308, 0.0286, 0.0344]  min 0 · mean 0.0307 · max 0.104
task/src_cup_lift: [0.00132, 0.344, 0.347, 0.339, 0.355, 0.337, 0.335, 0.336, 0.343, 0.341]  min 0.00132 · mean 0.332 · max 0.359
task/src_grasped: [0, 0.889, 0.883, 0.867, 0.882, 0.873, 0.891, 0.874, 0.873, 0.875]  min 0 · mean 0.855 · max 0.898
task/src_hand_foreign_rate: [0.000488, 0.0481, 0.0271, 0.0261, 0.0256, 0.0273, 0.032, 0.0229, 0.0237, 0.0222]  min 0.000488 · mean 0.033 · max 0.182
task/src_tilt_deg: [3.24, 105, 106, 102, 110, 104, 100, 96.6, 100, 102]  min 3.24 · mean 101 · max 112
task/success_now: [0, 0.5, 0.52, 0.587, 0.564, 0.539, 0.473, 0.502, 0.452, 0.413]  min 0 · mean 0.491 · max 0.616

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Round 20 (t2r_i19, 4096 env, warm start from t2r_i18 ep_2500, epoch 545): the user REJECTED iter_19 in the play video: the right (source) hand grasp degraded, the left arm still trembles, and some episodes fail to grasp at all. The next run WARM-STARTS FROM t2r_i18 ep_2500 AGAIN (not from iter_19), so START FROM THE iter_18 REWARD CODE (previous_code is iter_19: remove what iter_19 added unless listed below) and keep every iter_18 weight that is not listed.

Measured facts (TFEvents of t2r_i19 and 64-env deterministic traces at ADR 30, 900 steps, 60 Hz policy):
- iter_19 applied palm/hand action-rate penalties x10 and act_flip (weight 1.0) in EVERY phase. Result at epoch 545: episode_success recent 0.456 (iter_18 was ~0.54), src_grasped 0.87 (iter_18 0.95 in trace), reward/act_flip stayed at -0.34/step, i.e. the policy paid the penalty instead of removing the flips. Trace sign-flip rate over all action dims: iter_18 0.375, iter_19 0.343 (worst dims 0.62 -> 0.55). The tremor was NOT solved and the grasp got worse.
- Tilt did NOT become fill dependent. Episode peak source tilt by initial fill bin (low / mid / high): iter_18 152 / 154 / 150 deg, iter_19 150 / 144 / 147 deg. The first-bead release angle is 72 deg for a full cup and 106 deg for a nearly empty one, so 106 deg is enough for any fill. iter_19 overtilt was -2.0*tanh (saturating, recent -0.62/step) against tilt + pour_pose income of roughly +10/step, and tilt income was FLAT beyond its peak, so flipping the cup over cost almost nothing.
- Source cup linear speed after step 300: median 0.03 m/s, p95 0.17 (iter_18) and 0.20 m/s (iter_19).
- in_target 0.72, spill 0.046, cup_collision_rate 0.003, premature_tilt_rate low: keep these channels as they are in iter_18.

ENV (already in code, unchanged from the iter_19 run): palm command EMA 0.25 then slew limit 0.03 per step. ctx.actions / ctx.prev_actions are the RAW policy outputs.

User goals for this round, in priority order. Change ONLY what serves them:
1. Stable control (tremor, speed) WITHOUT hurting the grasp.
   - Action penalties must be ZERO until both cups are grasped and lifted (multiply by the existing carry-both factor); the reach/grasp/lift phases must be exactly as in iter_18.
   - Use moderate weights inside that gate: palm action-rate about 0.05, hand action-rate about 0.02, sign-flip about 0.5 (iter_19 used 0.1 / 0.05 / 1.0 everywhere).
   - Add physical smoothness inside the same gate, from ctx.src_cup_lin_vel / src_cup_ang_vel / rcv_cup_lin_vel / rcv_cup_ang_vel: penalise source cup linear speed above ~0.15 m/s, source cup angular speed above ~1.5 rad/s, and any receiver cup motion above ~0.05 m/s / 0.5 rad/s while the source cup is tilted past the premature limit. Soft (tanh or clamp) shapes, each at most about -1/step.
2. Accurate posture: keep the iter_18 envelope (wrap) and pour-direction gates unchanged. In addition multiply the tilt and pour_pose income by a soft factor of the SOURCE wrap count (0.4 at 0-1 fingers up to 1.0 at >= 3 fingers) so that tilting with a fingertip pinch earns less than tilting with an envelope grasp.
3. Fill dependent tilt angle: theta_rel = 1.25 + (1 - clamp(bead_fill_level * bead_in_source_frac, 0, 1)) * 0.60 rad.
   - Tilt income rises up to theta_rel + 0.10 rad and then DECLINES linearly (not flat), reaching zero at theta_rel + 0.60 rad.
   - overtilt penalty is LINEAR and unsaturated: -6.0 * clamp(src_cup_tilt - (theta_rel + 0.25), min=0) per rad, clamped at -6.
   - Keep the iter_19 flow limit (pour_delta income clamped at FLOW_CAP 0.02, flow_excess weight 20); transferred and the success bonus unchanged.
The environment now logs diag/* metrics (flip rate, cup speeds while pouring, wrap-ok rates, tilt minus remaining-fill release angle, per-fill-bin episode peak tilt and success); the reward does not need to log anything extra.
