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


# ---------------------------------------------------------------------------------------------
# geometry constant: the pouring lip sits this far from the rim centre (ctx doc, r = 4.1 cm)
_LIP_R = 0.041


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (N,3),(N,3) -> (N,)
    return torch.norm(a - b, dim=-1)


def _step(x: torch.Tensor) -> torch.Tensor:
    # smoothstep on an already normalised argument
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _carry_phase(grasped_f: torch.Tensor, h: torch.Tensor,
                 grasp_lo: float, grasp_hi: float, free_lo: float, free_hi: float) -> torch.Tensor:
    # (N,) in [0,1]: 0 on the table, 1 once the cup is grasped AND off the table.
    # high above the table the (flickering) grasp flag is no longer needed.
    held_ramp = grasped_f * _step((h - grasp_lo) / (grasp_hi - grasp_lo))
    high_ramp = _step((h - free_lo) / (free_hi - free_lo))
    return torch.maximum(held_ramp, high_ramp)


def _grasp_posture(finger_force: torch.Tensor, palm_force: torch.Tensor,
                   wrap_count: torch.Tensor, dt: torch.dtype):
    """Posture quality of ONE hand on its own cup, in [0,1], plus its two defect measures.

    This is the heart of round 21. It is built only from quantities that describe the shape of
    the contact, so it stays meaningful while the cup rotates:
      * wrap   - fingers whose middle/distal link touches the cup (an envelope needs >= 3)
      * palm   - palm contact force (the traces show it collapsing 5.5 N -> 1.25 N during the pour)
      * thumb  - thumb contact force (the thumb is the finger that gets pushed open when the cup
                 slides out of the palm toward the fingertips)
      * tip_f  - fingers touching with the fingertip ONLY = (contacting fingers) - (wrapped fingers);
                 a pinch multiplies the whole quality by as little as 0.4
    """
    wrap_n = wrap_count.to(dt)
    wrap_q = torch.clamp(wrap_n / 3.0, 0.0, 1.0)                       # 1.0 at >= 3 wrapped fingers
    n_contact = torch.sum((finger_force > 0.3).to(dt), dim=-1)          # 0.3 N = contact threshold
    tip_only = torch.clamp(n_contact - wrap_n, min=0.0)
    tip_f = torch.clamp(tip_only / 2.0, 0.0, 1.0)                       # 2 tip-only fingers = full defect
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)         # 0.76 at the required 3 N
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    q = (0.45 * wrap_q + 0.30 * palm_q + 0.25 * thumb_q) * (1.0 - 0.6 * tip_f)
    return q, tip_f, wrap_q


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dev = ctx.src_palm_pos.device
    dt = ctx.src_palm_pos.dtype
    N = ctx.src_palm_pos.shape[0]
    zeros = torch.zeros(N, device=dev, dtype=dt)

    # ======================================================================== weights / constants
    # --- stage 1-3 (reach, grasp, lift): deliberately at the iter_18 level. iter_17 cut this
    #     income and the source cup was then never lifted at all, so nothing here gets weaker.
    APPROACH_W = 0.6
    APPROACH_K = 4.0
    REACH_W = 0.3
    REACH_FAR, REACH_NEAR = 0.26, 0.14
    GRASP_W = 0.8
    CLOSE_W = 0.8
    CLOSE_NORM = 0.35
    BOTH_W = 0.8
    LIFT_SRC_W = 3.0          # iter_18 value that lifted reliably
    LIFT_RCV_W = 1.5
    LIFT_TARGET_SRC = 0.25
    LIFT_TARGET_RCV = 0.10
    LIFTOFF_H = 0.03          # [m] e-fold of the lift-off part: the first cm pays the most
    RCV_LIFT_FREE, RCV_LIFT_SIGMA = 0.15, 0.08

    # --- goal 1: grasp posture that survives rotation. Paid EVERY held step, phase independent,
    #     and amplified (not replaced) during the pour, so losing a wrapped finger at step 600
    #     costs more than at step 200 instead of less.
    POSTURE_SRC_W = 3.0       # comparable with lift_src 3.0: the posture is a first-class income
    POSTURE_RCV_W = 1.2
    POSTURE_POUR_BONUS = 0.6  # x1.6 while pouring / after the pour
    TIP_W = 1.2               # explicit charge for fingertip-only holding (logged separately)
    TIP_POUR_BONUS = 1.0      # x2.0 during the pour
    PALM_SLIP_W = 1.5         # charge for the palm force collapsing below PALM_MIN while held
    PALM_MIN = 3.0            # [N] acceptance target for the pour/late phases
    GRIP_FLOOR = 0.15         # rotating with a ruined grasp keeps only 15 % of the tilt income

    # --- transport / aiming
    CONV_W = 2.0
    CONV_FAR = 0.40
    BRING_W = 1.5
    BT_K = 10.0
    AIM_W = 2.5
    AIM_K = 15.0
    LIP_DEADBAND = 0.02
    APPR_FULL, APPR_ZERO = 0.08, 0.24       # lip xy window in which tilting may start to pay
    SAFE_FULL, SAFE_ZERO = 0.050, 0.075     # release window, inside the env latch radius 8.1 cm
    HOVER_LO, HOVER_HI = 0.06, 0.11
    HOVER_FREE0, HOVER_SIGMA = 0.10, 0.08
    LIP_Z_LO, LIP_Z_HI = 0.0, 0.02
    LIP_Z_FREE0, LIP_Z_SIGMA = 0.07, 0.10
    LIP_ABOVE_FLOOR = 0.35
    DIR_FULL, DIR_ZERO = 0.0, 0.28          # pour_dir_xy[:,0] > 0.3 voids success
    DIR_W, DIR_PEN_W = 1.5, 1.0
    SIDE_VOID, SIDE_OK = 0.03, 0.10
    SIDE_PEN_W = 2.0
    HOME_K, HOME_W = 10.0, 0.8
    HOME_SRC_FLOOR = 0.2

    # --- goal 3: fill dependent pour angle
    THETA_REL_BASE = 1.25     # [rad] first bead leaves a FULL cup at ~72 deg
    THETA_REL_SPAN = 0.60     # ... and a nearly empty one at ~106 deg
    TILT_PEAK_ABOVE = 0.10    # income peaks at theta_rel + 0.10 rad
    TILT_ZERO_ABOVE = 0.60    # ... and is zero at theta_rel + 0.60 rad
    OVERTILT_ABOVE = 0.25     # linear, unsaturated penalty starts here
    OVERTILT_W, OVERTILT_MAX = 6.0, 6.0
    TILT_PRE_W = 4.0          # below the premature limit, inside the approach zone
    TILT_MAIN_W = 20.0        # the real pour income: only with the lip over the receiver mouth
    POSE_W = 6.0              # holding the release pose
    PRE_MARGIN = 0.17         # [rad] theta_pre = premature limit - 10 deg
    SRC_TILT_FREE = 0.20      # [rad] tilt that is free far from the receiver
    SRC_TILT_SCALE = 0.20
    SRC_TILT_SIGMA = 0.25
    SRC_UP_W_TABLE, SRC_UP_W_CARRY = 1.0, 1.5
    LATCH_W = 1.0
    BAND_BELOW, BAND_ABOVE = 0.10, 0.40

    # --- beads
    POUR_DELTA_W = 120.0
    FLOW_CAP = 0.02           # per-step cap on the increment that is paid
    FLOW_W = 20.0
    TRANSFER_W = 5.0
    SPILL_W = 50.0
    SETTLE_FRAC = 0.05
    POST_POUR_FRAC = 0.5      # the task succeeds at half the beads
    HOLD_SRC_W, HOLD_RCV_W = 1.5, 0.5

    # --- receiver constraints
    RCV_TILT_FREE, RCV_TILT_SIGMA = 0.12, 0.20
    RCV_PEN_SCALE = 0.25
    RCV_GATE_FLOOR = 0.3
    RCV_UP_W, RCV_UP_POUR_W = 1.0, 1.0
    RCV_STILL_FREE, RCV_STILL_SCALE = 0.05, 0.15
    RCV_STILL_W, RCV_STILL_NEAR_W, RCV_STILL_POUR_W = 0.25, 0.25, 0.50
    RCV_TOPPLED = 1.2
    DROP_DEPTH = 0.03

    # --- safety / collisions (real robot)
    FORCE_SCALE = 5.0
    CONTACT_W, CONTACT_SCALE = 2.0, 2.0
    HIT_FORCE, HIT_EVENT_W, HIT_LATCH_W = 0.5, 5.0, 0.3
    FOREIGN_DEADBAND, CLEAN_DEADBAND = 1.0, 0.5
    FOREIGN_W_FREE, FOREIGN_W_HELD = 0.3, 1.5
    CLEAN_FLOOR = 0.5
    PRELIFT_SAFE, PRELIFT_ZERO, PRELIFT_W = 0.14, 0.10, 2.0
    NEAR_DIST, CLOSE_V_FREE, CLOSE_V_SCALE = 0.15, 0.15, 0.20

    # --- goal 2: stable control. Base rates are the iter_18 values (iter_19 multiplied them by 10
    #     in every phase and made both the grasp and the success rate worse). Everything stronger
    #     is gated on the carry phase or on the hand channels only.
    PALM_RATE_W, HAND_RATE_W = 0.01, 0.005
    PALM_RATE_CARRY_W, HAND_RATE_CARRY_W = 0.05, 0.02
    SAT_LO, SAT_W = 0.7, 0.15
    ADR_RAMP_FULL = 0.2       # control charges reach full weight at ADR progress 0.2 (6/30)
    FLIP_CARRY_W = 0.6        # sign flip of consecutive raw palm+hand commands, carry gate only
    HAND_FLIP_W = 0.4         # flip of the 6 hand channels while a cup is held: it directly
                              # opens/closes the fingers and is what ruins the posture
    SRC_V_FREE, SRC_V_SCALE = 0.15, 0.15
    SRC_W_FREE, SRC_W_SCALE = 1.5, 1.5
    RCV_V_FREE, RCV_V_SCALE = 0.05, 0.10
    RCV_W_FREE, RCV_W_SCALE = 0.5, 1.0
    ROT_SPEED_FREE = 1.0

    SUCCESS_W = 20.0

    # ======================================================================== basic state
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    src_carry = _carry_phase(g_src, h_src, 0.015, 0.05, 0.06, 0.10)
    rcv_carry = _carry_phase(g_rcv, h_rcv, 0.015, 0.05, 0.06, 0.10)
    held_src = torch.maximum(g_src, src_carry)     # a thumb contact that flickers is not fatal
    held_rcv = torch.maximum(g_rcv, rcv_carry)
    carry_raw = src_carry * rcv_carry               # geometric fact: both cups are off the table
    not_nested_f = (~ctx.cups_nested).to(dt)
    settled = _step(ctx.episode_progress / SETTLE_FRAC)

    clear = 1.0 - torch.tanh(ctx.cup_cup_force / FORCE_SCALE)
    f_foreign = torch.maximum(ctx.src_hand_foreign_force, ctx.rcv_hand_foreign_force)
    clean = 1.0 - torch.tanh(torch.clamp(f_foreign - CLEAN_DEADBAND, min=0.0) / FORCE_SCALE)
    clean_soft = CLEAN_FLOOR + (1.0 - CLEAN_FLOOR) * clean

    # ======================================================================== posture quality
    q_src, tip_src, wrapq_src = _grasp_posture(ctx.src_finger_force, ctx.src_palm_force,
                                               ctx.src_wrap_count, dt)
    q_rcv, tip_rcv, wrapq_rcv = _grasp_posture(ctx.rcv_finger_force, ctx.rcv_palm_force,
                                               ctx.rcv_wrap_count, dt)
    # envelope factor per hand: 0 at <= 1 wrapped finger, 1 at >= 3
    env_src = _step((ctx.src_wrap_count.to(dt) - 1.0) / 2.0)
    env_rcv = _step((ctx.rcv_wrap_count.to(dt) - 1.0) / 2.0)
    # the gate every rotation / pour income is multiplied by: a degrading grasp earns much less
    grip_src = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * (0.5 * q_src + 0.5 * env_src)
    grip_rcv = GRIP_FLOOR + (1.0 - GRIP_FLOOR) * (0.5 * q_rcv + 0.5 * env_rcv)

    # ======================================================================== receiver upright gate
    rcv_tilt_excess = torch.clamp(ctx.rcv_cup_tilt - RCV_TILT_FREE, min=0.0)
    rcv_up_raw = torch.exp(-(rcv_tilt_excess / RCV_TILT_SIGMA) ** 2)
    rcv_up = 1.0 - rcv_carry * (1.0 - rcv_up_raw)
    rcv_up_soft = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_up
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    rcv_home = torch.exp(-HOME_K * d_home)
    dir_x = ctx.pour_dir_xy[:, 0].to(dt)
    dir_ok = 1.0 - _step((dir_x - DIR_FULL) / (DIR_ZERO - DIR_FULL))

    # carry gate of every transport-stage income: both cups lifted AND both ENVELOPED AND the
    # receiver on its own side. The envelope factors are what keep the posture alive after pickup.
    carry_both = carry_raw * (0.25 + 0.75 * env_src) * (0.25 + 0.75 * env_rcv) * (0.2 + 0.8 * side_ok)

    # ======================================================================== pouring geometry
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    lip_off = torch.clamp(lip_xy - LIP_DEADBAND, min=0.0)
    approach_gate = 1.0 - _step((lip_xy - APPR_FULL) / (APPR_ZERO - APPR_FULL))
    safe = 1.0 - _step((lip_xy - SAFE_FULL) / (SAFE_ZERO - SAFE_FULL))

    tilt_mag = ctx.src_cup_tilt
    limit = ctx.premature_tilt_limit.to(dt)
    release_tilt = limit + math.radians(20.0)
    theta_pre = torch.clamp(limit - PRE_MARGIN, min=0.0)
    # signed tilt toward the receiver, never larger than the total tilt
    theta_eff = torch.minimum(torch.clamp(ctx.src_tilt_toward_rcv.to(dt), min=0.0), tilt_mag)

    # height budget grows with the tilt: the lip drops r*sin(theta) below the rim centre, so a cup
    # that must turn further has to ride higher. Rotating and raising together must not be taxed.
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    hover_free = HOVER_FREE0 + _LIP_R * sin_t
    lip_free = LIP_Z_FREE0 + _LIP_R * sin_t
    dz_eq = ctx.src_cup_pos[:, 2] + ctx.cup_mouth_z - ctx.rcv_cup_mouth_pos[:, 2]
    hover_decay = torch.exp(-(torch.clamp(dz_eq - hover_free, min=0.0) / HOVER_SIGMA) ** 2)
    hover = _step((dz_eq - HOVER_LO) / (HOVER_HI - HOVER_LO)) * hover_decay
    hover_wide = _step((dz_eq + 0.04) / (HOVER_HI + 0.04)) * hover_decay
    lip_above_raw = torch.clamp((lip_dz - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(lip_dz - lip_free, min=0.0) / LIP_Z_SIGMA) ** 2)
    lip_above = LIP_ABOVE_FLOOR + (1.0 - LIP_ABOVE_FLOOR) * lip_above_raw
    z_ok = torch.maximum(hover, safe * lip_above_raw * lip_high)

    # fill dependent release angle and the phase switch used by the posture income
    f_rem = torch.clamp(ctx.bead_fill_level.to(dt) * ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    theta_rel = THETA_REL_BASE + (1.0 - f_rem) * THETA_REL_SPAN
    theta_peak = theta_rel + TILT_PEAK_ABOVE
    post_pour = _step(ctx.bead_in_target_frac / POST_POUR_FRAC)
    turning = _step((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.1))
    pour_phase = torch.maximum(carry_raw * safe * turning, post_pour)

    # ======================================================================== stage 1: approach
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = APPROACH_W * torch.exp(-APPROACH_K * d_src)
    approach_rcv = APPROACH_W * torch.exp(-APPROACH_K * d_rcv)
    reach_src = REACH_W * _step((REACH_FAR - d_src) / (REACH_FAR - REACH_NEAR))
    reach_rcv = REACH_W * _step((REACH_FAR - d_rcv) / (REACH_FAR - REACH_NEAR))

    # ======================================================================== stage 2: grasp
    clos_src = torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    clos_rcv = torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_src = GRASP_W * held_src + CLOSE_W * torch.exp(-6.0 * d_src) * clos_src
    grasp_rcv = GRASP_W * held_rcv + CLOSE_W * torch.exp(-6.0 * d_rcv) * clos_rcv
    both_grasped = BOTH_W * held_src * held_rcv

    # ---- GOAL 1 ---- posture income: unconditional on phase, amplified while pouring ------------
    posture_src = POSTURE_SRC_W * held_src * q_src * (1.0 + POSTURE_POUR_BONUS * pour_phase)
    posture_rcv = POSTURE_RCV_W * held_rcv * q_rcv
    tip_only_src = -TIP_W * held_src * tip_src * (1.0 + TIP_POUR_BONUS * pour_phase)
    tip_only_rcv = -TIP_W * held_rcv * tip_rcv
    # the palm force must not collapse: this is exactly the i18 defect (5.5 N -> 1.25 N)
    palm_slip_src = -PALM_SLIP_W * held_src * (0.3 + 0.7 * pour_phase) \
        * torch.clamp(1.0 - torch.clamp(ctx.src_palm_force, min=0.0) / PALM_MIN, 0.0, 1.0)

    # ======================================================================== stage 3: lift
    h_src_pos = torch.clamp(h_src, min=0.0)
    lift_shape_src = 0.5 * (1.0 - torch.exp(-h_src_pos / LIFTOFF_H)) \
        + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, 0.0, 1.0)
    tilt_over = torch.clamp(tilt_mag - (SRC_TILT_FREE + approach_gate
                                        * torch.clamp(theta_pre - SRC_TILT_FREE, min=0.0)), min=0.0) * (1.0 - safe)
    src_up = torch.exp(-(tilt_over / SRC_TILT_SIGMA) ** 2)
    # the lift pays at the iter_18 strength; the tilt corridor may halve it but never zero it,
    # and the envelope adds up to +25 % instead of gating the lift away
    lift_src = LIFT_SRC_W * held_src * lift_shape_src * (0.5 + 0.5 * src_up) * (1.0 + 0.25 * env_src)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = LIFT_RCV_W * held_rcv * torch.clamp(h_rcv / LIFT_TARGET_RCV, 0.0, 1.0) \
        * rcv_band * (0.5 + 0.5 * rcv_up) * (0.25 + 0.75 * env_rcv)
    both_lifted = BOTH_W * carry_both

    # ======================================================================== stage 4: transport
    conv_xy = torch.clamp(1.0 - lip_off / CONV_FAR, 0.0, 1.0)
    converge = CONV_W * carry_both * not_nested_f * src_up * rcv_up_soft * hover_wide * conv_xy
    bring_together = BRING_W * carry_both * (0.3 + 0.7 * dir_ok) * not_nested_f * clean_soft \
        * src_up * torch.exp(-BT_K * lip_off) * z_ok * rcv_up_soft
    stack_gate = carry_both * not_nested_f * clear * clean_soft
    aim_xy = torch.exp(-AIM_K * lip_off)
    aim = AIM_W * stack_gate * grip_src * aim_xy * z_ok * rcv_up_soft * (0.3 + 0.7 * dir_ok)
    pour_dir = DIR_W * carry_both * not_nested_f * conv_xy * dir_ok
    pour_dir_pen = -DIR_PEN_W * carry_raw * approach_gate * _step((dir_x - 0.10) / 0.25)
    home_src_f = HOME_SRC_FLOOR + (1.0 - HOME_SRC_FLOOR) * src_carry
    v_rcv = torch.norm(ctx.rcv_cup_lin_vel, dim=-1)
    still_f = 1.0 - torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)
    home_rcv = HOME_W * home_src_f * rcv_carry * grip_rcv * rcv_up * rcv_band * side_ok * rcv_home * still_f
    side_margin = -SIDE_PEN_W * held_rcv * (1.0 - side_ok)

    # ======================================================================== stage 5: pour angle
    # ---- GOAL 3 ---- one bell per episode: rises to theta_rel + 0.10 rad, zero at + 0.60 rad.
    # There is NO flat income above the peak, which is what pinned every earlier run at 150 deg.
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    rise = torch.clamp(theta_eff / torch.clamp(theta_peak, min=0.3), 0.0, 1.0)
    decline = torch.clamp((theta_rel + TILT_ZERO_ABOVE - tilt_mag) / (TILT_ZERO_ABOVE - TILT_PEAK_ABOVE),
                          0.0, 1.0)
    bell = rise * decline
    # every rotation income is multiplied by the CURRENT posture, so turning the cup with a
    # degrading grasp is worth a fraction of turning it with an envelope
    tilt_gate = carry_both * not_nested_f * clean_soft * clear * beads_left * rcv_up_soft \
        * dir_ok * grip_src
    # (a) below the premature limit and inside the approach zone: safe, cannot latch, gives the
    #     gradient that starts the rotation once the cup is high and on its way
    tilt_pre = TILT_PRE_W * tilt_gate * (torch.minimum(theta_eff, theta_pre)
                                         / torch.clamp(theta_peak, min=0.3)) * approach_gate * hover_wide
    # (b) the real pour: only with the lip inside the release radius and above the receiver rim
    tilt_main = TILT_MAIN_W * tilt_gate * torch.clamp(bell - torch.minimum(theta_eff, theta_pre)
                                                      / torch.clamp(theta_peak, min=0.3), min=0.0) \
        * safe * lip_above
    # holding the release pose (lip over the mouth, angle in the release band, receiver upright)
    pour_band = _step((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = POSE_W * stack_gate * grip_src * beads_left * safe * lip_above * aim_xy \
        * pour_band * decline * rcv_up * dir_ok
    # linear, unsaturated charge past theta_rel + 0.25 rad, only while the cup is carried
    overtilt = -torch.clamp(OVERTILT_W * torch.clamp(tilt_mag - (theta_rel + OVERTILT_ABOVE), min=0.0),
                            max=OVERTILT_MAX) * src_carry
    # keep the cup near upright until it is lifted AND near the receiver
    upright_src = -(SRC_UP_W_TABLE + SRC_UP_W_CARRY * src_carry) * held_src \
        * torch.tanh(tilt_over / SRC_TILT_SCALE)
    premature_latch = -LATCH_W * ctx.premature_tilt.to(dt)
    src_ang_speed = torch.norm(ctx.src_cup_ang_vel, dim=-1)
    rot_speed = -0.3 * held_src * torch.tanh(torch.clamp(src_ang_speed - ROT_SPEED_FREE, min=0.0) / 1.5)

    # ======================================================================== stage 6: beads
    d_in = ctx.d_in_target.to(dt)
    pour_delta = POUR_DELTA_W * torch.clamp(d_in, -1.0, FLOW_CAP)   # increments, per-step cap
    flow_excess = -FLOW_W * torch.relu(d_in - FLOW_CAP)             # dumping everything at once
    transferred = TRANSFER_W * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0) * not_nested_f * rcv_up
    spill_delta = -SPILL_W * settled * torch.clamp(ctx.d_spill, min=0.0)
    hold_src = HOLD_SRC_W * post_pour * held_src * q_src
    hold_rcv = HOLD_RCV_W * post_pour * held_rcv * q_rcv * rcv_up

    # ======================================================================== receiver constraints
    upright_rcv = -(RCV_UP_W + RCV_UP_POUR_W * pour_phase) * rcv_carry \
        * torch.tanh(rcv_tilt_excess / RCV_PEN_SCALE)
    still_w = RCV_STILL_W + RCV_STILL_NEAR_W * approach_gate * src_carry + RCV_STILL_POUR_W * pour_phase
    rcv_still = -still_w * rcv_carry * torch.tanh(torch.clamp(v_rcv - RCV_STILL_FREE, min=0.0) / RCV_STILL_SCALE)
    src_dropped = (h_src < -DROP_DEPTH).to(dt)
    rcv_dropped = ((h_rcv < -DROP_DEPTH) | (ctx.rcv_cup_tilt > RCV_TOPPLED)).to(dt)
    drop = -0.5 * (src_dropped + rcv_dropped)

    # ======================================================================== safety / collisions
    cup_contact = -CONTACT_W * torch.tanh(ctx.cup_cup_force / CONTACT_SCALE)
    hit_now = (ctx.cup_cup_force > HIT_FORCE).to(dt)
    cup_hit = -HIT_EVENT_W * settled * hit_now - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_src = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    f_rcv = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEADBAND, min=0.0)
    hand_foreign_src = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) \
        * torch.tanh(f_src / FORCE_SCALE)
    hand_foreign_rcv = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) \
        * torch.tanh(f_rcv / FORCE_SCALE)
    rel = ctx.rcv_cup_pos - ctx.src_cup_pos
    cup_dist = torch.norm(rel, dim=-1)
    u = rel / (cup_dist.unsqueeze(-1) + 1e-6)
    v_close = torch.sum((ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel) * u, dim=-1)
    closing_speed = -1.0 * torch.maximum(src_carry, rcv_carry) * (cup_dist < NEAR_DIST).to(dt) \
        * torch.tanh(torch.clamp(v_close - CLOSE_V_FREE, min=0.0) / CLOSE_V_SCALE)
    nested = -0.5 * ctx.cups_nested.to(dt)
    prelift_near = -PRELIFT_W * (1.0 - carry_raw) * _step((PRELIFT_SAFE - cup_dist) / (PRELIFT_SAFE - PRELIFT_ZERO))

    # ======================================================================== goal 2: stable control
    da = ctx.actions - ctx.prev_actions
    # exploration-killing charges are off for a policy that has not learned the task yet
    # (adr_progress stays 0 until the running success rate first passes 0.3) and are fully on
    # from ADR 6/30. The iter_18 base rates stay on always.
    ramp = _step(torch.full_like(carry_raw, float(ctx.adr_progress) / ADR_RAMP_FULL))
    palm_w = PALM_RATE_W + (PALM_RATE_CARRY_W - PALM_RATE_W) * carry_raw * ramp
    hand_w = HAND_RATE_W + (HAND_RATE_CARRY_W - HAND_RATE_W) * carry_raw * ramp
    palm_rate_src = -palm_w * torch.sum(da[:, 0:6] ** 2, dim=-1)
    palm_rate_rcv = -palm_w * torch.sum(da[:, 9:15] ** 2, dim=-1)
    hand_rate = -hand_w * (torch.sum(da[:, 6:9] ** 2, dim=-1) + torch.sum(da[:, 15:18] ** 2, dim=-1))
    flips = torch.relu(-(ctx.actions * ctx.prev_actions))
    act_flip = -FLIP_CARRY_W * ramp * carry_raw * torch.sum(flips, dim=-1)
    # the 6 finger channels are charged for flipping whenever a cup is held: an open/close
    # oscillation is exactly what pushes the thumb out and lets the cup creep to the fingertips
    hand_flip = -HAND_FLIP_W * ramp * torch.maximum(held_src, held_rcv) \
        * (torch.sum(flips[:, 6:9], dim=-1) + torch.sum(flips[:, 15:18], dim=-1))
    a_abs = torch.abs(ctx.actions)
    palm_sat_src = -SAT_W * ramp * torch.mean(_step((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    palm_sat_rcv = -SAT_W * ramp * torch.mean(_step((a_abs[:, 9:15] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    v_src = torch.norm(ctx.src_cup_lin_vel, dim=-1)
    src_speed = -carry_raw * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = -carry_raw * torch.tanh(torch.clamp(src_ang_speed - SRC_W_FREE, min=0.0) / SRC_W_SCALE)
    w_rcv = torch.norm(ctx.rcv_cup_ang_vel, dim=-1)
    rcv_motion = -0.5 * carry_raw * pour_phase * (
        torch.tanh(torch.clamp(v_rcv - RCV_V_FREE, min=0.0) / RCV_V_SCALE)
        + torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE))

    # ======================================================================== success
    # the terminal payoff itself is scaled by the posture: a successful pour held with a pinch
    # is worth half of one held with an envelope
    success = SUCCESS_W * ctx.success.to(dt) * clear * clean \
        * (0.5 + 0.5 * held_src * q_src) * (0.6 + 0.4 * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "both_grasped": both_grasped,
        "posture_src": posture_src,
        "posture_rcv": posture_rcv,
        "tip_only_src": tip_only_src,
        "tip_only_rcv": tip_only_rcv,
        "palm_slip_src": palm_slip_src,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "converge": converge,
        "bring_together": bring_together,
        "aim": aim,
        "pour_dir": pour_dir,
        "pour_dir_pen": pour_dir_pen,
        "home_rcv": home_rcv,
        "side_margin": side_margin,
        "tilt_pre": tilt_pre,
        "tilt_main": tilt_main,
        "pour_pose": pour_pose,
        "overtilt": overtilt,
        "upright_src": upright_src,
        "premature_latch": premature_latch,
        "rot_speed": rot_speed,
        "pour_delta": pour_delta,
        "flow_excess": flow_excess,
        "transferred": transferred,
        "spill_delta": spill_delta,
        "hold_src": hold_src,
        "hold_rcv": hold_rcv,
        "upright_rcv": upright_rcv,
        "rcv_still": rcv_still,
        "drop": drop,
        "cup_contact": cup_contact,
        "cup_hit": cup_hit,
        "hand_foreign_src": hand_foreign_src,
        "hand_foreign_rcv": hand_foreign_rcv,
        "closing_speed": closing_speed,
        "nested": nested,
        "prelift_near": prelift_near,
        "palm_rate_src": palm_rate_src,
        "palm_rate_rcv": palm_rate_rcv,
        "hand_rate": hand_rate,
        "act_flip": act_flip,
        "hand_flip": hand_flip,
        "palm_sat_src": palm_sat_src,
        "palm_sat_rcv": palm_sat_rcv,
        "src_speed": src_speed,
        "src_spin": src_spin,
        "rcv_motion": rcv_motion,
        "success": success,
    }

    reward = zeros
    for v in components.values():
        reward = reward + v

    return reward, components
```

Observations from watching the trained policy:
FULL REWRITE (round 21). The operator stopped t2r_i20 and asked for the reward to be written again from
the task description, using the facts below, instead of another patch on iter_20. Every earlier reward
(iter_16 .. iter_20) drifted into the same failure: the grasp is fine at pickup and then DEGRADES while
the arm rotates the cup. Write the reward around "establish an envelope grasp, then KEEP that grasp
posture through lift, carry, rotation and pour".

MEASURED (64-env deterministic traces at ADR 30, 900 steps, 60 Hz; contact threshold 0.5 N;
"wrap" = fingers whose middle OR distal link touches the cup, 0..5; "tip-only" = fingers touching with
the fingertip only; thumb1 = thumb abduction joint, commanded 0 rad for the whole episode):

t2r_i18 ep_2500 (the operator's current S2R candidate), phase means over grasped steps:
  phase            wrap  tip-only  palm N  thumb1 deg  cup tilt deg  lift m
  grasp   60-120   2.21    0.00     0.13      3.9         14.6       0.109
  lift   120-200   3.96    0.46     3.26     13.3         78.7       0.305
  carry  200-260   4.11    0.59     5.50     34.5        142.8       0.406
  tiltup 260-340   4.10    0.73     4.44     53.4        140.9       0.404
  pour   340-500   3.96    0.93     2.62     65.1        135.3       0.405
  late   500-850   3.61    1.27     1.25     63.5        116.0       0.395
  -> wrap peaks at 4.11 in carry and decays to 3.61; tip-only contacts rise 0.00 -> 1.27;
     palm contact force rises to 5.50 N then COLLAPSES to 1.25 N; thumb1 is pushed monotonically
     from 4 deg to 65 deg although its command is 0 the whole time. The cup slides out of the palm
     toward the fingertips as the thumb is forced open. This is the defect to fix.
  -> the policy also tilts far too early: mean cup tilt is already 78.7 deg during the LIFT window
     and 142.8 deg in carry, i.e. it flips the cup while raising it.

Cross-run summary (succ / in_target / spill, then wrap before-pour -> during-pour -> last 30 % of
episode, tip-only before -> during, median peak tilt, thumb1 deviation during pour, action sign-flip rate):
  i16 ep5600  0.95 0.91 0.07 | 2.20 -> 2.55 -> 2.29 | 0.50 -> 1.74 | 104 deg | 37.3 deg | 0.38
  i18 ep2500  0.83 0.91 0.07 | 3.12 -> 3.84 -> 3.55 | 0.74 -> 1.04 | 155 deg | 61.1 deg | 0.38
  i18 ep2900  0.55 0.58 0.06 | 2.65 -> 3.55 -> 3.37 | 2.01 -> 1.32 | 152 deg | 54.7 deg | 0.35
  i19 ep500   0.70 0.92 0.08 | 2.68 -> 3.47 -> 3.42 | 0.85 -> 1.33 | 151 deg | 66.5 deg | 0.35
  i17 ep900   0.00 0.00 0.02 | 1.93 (never poured: the source cup was never lifted)  | 0.51
  -> i16 poured at 104 deg with only 2.2-2.8 wrapped fingers and 2.08 tip-only fingers late: a
     fingertip pinch, which the operator rejected on video. i18 reaches ~4 wrapped fingers but still
     loses them, and it over-rotates to 155 deg (a full cup releases its first bead near 72 deg,
     a nearly empty one near 106 deg, so anything past ~110 deg is pure over-rotation).
  -> iter_19 raised action-rate penalties x10 and added a sign-flip penalty in every phase: the
     sign-flip rate barely moved (0.38 -> 0.35) while success and the grasp got worse. Do NOT repeat
     large ungated action penalties.
  -> iter_17 removed all stationary income with penalties active from step 0 and the source cup was
     then never lifted at all (task/src_cup_lift stuck at 0.003 m for 727 epochs). Do NOT make the
     approach/grasp/lift phases pay less than iter_18 does.

WHAT THE REWARD MUST DO (operator's three goals, in priority order):
1. Grasp posture that SURVIVES rotation. Pay for the envelope while the cup is held, not only when it
   is picked up: use ctx.src_wrap_count / ctx.rcv_wrap_count (fingers with middle/distal contact) and
   ctx.src_palm_force / ctx.rcv_palm_force. Require >= 3 wrapped fingers and a palm contact that does
   not collapse; charge fingertip-only holding. The income must be paid EVERY step the cup is held,
   including during the pour, so that losing a wrapped finger at step 600 costs as much as at step 200.
   Gate the tilt and pour income on the current wrap quality so that rotating with a degrading grasp
   earns less than rotating with a good one.
2. Stable control: keep the iter_18-level action-rate penalties in reach/grasp/lift, and only inside
   the both-cups-carried phase add a moderate smoothness cost (cup linear speed above ~0.15 m/s,
   cup angular speed above ~1.5 rad/s, receiver cup motion while the source is tilted).
3. ADAPTIVE pour (operator 09.21, replaces the earlier "bell at one angle" idea). Every video so far
   shows the cup moved to ONE angle and frozen there, for every fill level. The operator wants: carry,
   then rotate SLOWLY and CONTINUOUSLY from a fill-dependent start angle up to a fixed end angle.
     start angle  theta_start = 1.25 + (1 - clamp(ctx.bead_fill_level, 0, 1)) * 0.60 rad
                  (measured first-release angle: 72 deg full, 89 deg half, 100 deg at 20 %, 106 deg empty)
     end angle    theta_end   = 2.09 rad (120 deg), the same for every fill.
   OBSERVABILITY CONSTRAINT - this is why the previous design froze: the actor is a memoryless MLP and
   sees ONLY the initial fill (ctx.bead_fill_level, constant for the whole episode) and the current cup
   tilt. It does NOT see ctx.bead_in_source_frac. A target angle defined from bead_in_source_frac is
   invisible to the policy, which then settles on one compromise angle. So theta_start MUST be a function
   of ctx.bead_fill_level alone, and the desired motion must be expressible as "rotation speed as a
   function of (current tilt, initial fill)". bead_in_source_frac / d_in_target may still be used for
   the bead-transfer income and as gates, but NOT to define the angle the policy must be at.
   PAY FOR THE SWEEP, NOT FOR THE ANGLE. A reward for "being at angle X" is what produces jump-and-hold.
   Instead pay for forward tilt PROGRESS inside the window [theta_start, theta_end]:
     - income per step proportional to the positive tilt increment (tilt rate toward the receiver * dt,
       or the clamped window-progress difference), only while the tilt is inside the window, the lip is
       over the receiver mouth and above its rim, the pour direction is valid, and the grasp-posture
       gate of goal 1 is high;
     - multiplied by a SLOW-ROTATION band on the source cup angular speed: full pay for roughly
       0.05-0.35 rad/s, falling to zero by ~0.8 rad/s, so that a fast flip through the window collects
       almost nothing while a slow sweep collects the whole integral (the window is 14-48 deg wide, so a
       correct sweep lasts roughly 1-8 s);
     - a small stall charge while the cup is inside the window with beads still in the source and the
       tilt rate near zero, so freezing part-way is not a resting point; it must stay far smaller than
       the posture income so it never pays to rush;
     - below theta_start (and once lifted and near the receiver) a modest, non-repeatable gradient toward
       theta_start so the cup gets to the window; nothing may be paid for tilting during the lift;
     - past theta_end: no income at all, plus a linear unsaturated penalty. The traces show every run
       pinned at 150+ deg because flat income made flipping free.
   The progress income must not be farmable by rocking back and forth: pay only NEW progress (compare
   against a best-so-far that you can rebuild from ctx, or make backward motion cost as much as forward
   motion earns). ctx has no persistent storage of yours, so prefer the symmetric-cost form.
   The per-step cap on ctx.d_in_target income stays: a controlled stream must beat a dump.
   Also do not tilt during the lift: keep the cup near upright until it is lifted and near the
   receiver (ctx.premature_tilt_limit already latches success away, but the reward must also not pay).
KEEP what works in iter_18 and is unchanged: the success/nesting/cup_hit/pour-direction/receiver-side
rules, the bead transfer income on ctx.d_in_target increments with a per-step cap, the spill penalty,
and the lift/approach shaping strengths (iter_18 lifted reliably; iter_17 did not).
The environment now logs diag/* for all of this (wrap during pour, thumb abduction drift, flip rate,
cup speeds, per-fill-bin peak tilt), so the reward does not need to log anything.

ACCEPTANCE (operator 09.21: "even with fresh training, if it looks wrong in sim it cannot be good on the
real robot"). Success rate alone does not pass this round. The run is judged on the physical behaviour
the traces measure, against the i18 ep_2500 numbers quoted above:
  - wrap in the last 30 % of the episode >= 3.5 fingers (i18: 3.55) AND tip-only contacts <= 0.6
    (i18: 1.27) AND source palm contact force in the pour/late phases >= 3 N (i18 collapses 5.50 -> 1.25).
  - thumb abduction drift during the pour <= 15 deg (i18: 61.1 deg).
  - ADAPTIVE SWEEP (operator 09.21): per fill bin, the tilt at which the pour phase begins must
    differ - high-fill bin starts >= 10 deg lower than the low-fill bin (target ~75-90 deg at 80 %,
    ~100 deg at 20 %) - and every bin ends at 110-125 deg (i18: all bins 142-155 deg, i20: 142/146/146,
    i.e. no dependence on fill at all). Source cup angular speed while beads are leaving: median
    <= 0.4 rad/s, and the time from theta_start to peak tilt >= 1 s. One fixed angle held for the whole
    pour is a FAIL even if success is high. The cup stays near upright until it is lifted and close to
    the receiver.
  - action sign-flip rate <= 0.20 (every run so far: 0.35-0.38).
  - success >= 0.5 and spill <= 0.08 at ADR 30, with cup collisions, nesting and receiver-side
    violations at zero as in iter_18.
A reward that hits the success number while any of the posture numbers stays at the iter_18 level is a
failed round: the operator rejects it on video and it cannot transfer.

ROUND 22 ADDENDUM (operator 09.21: "set it up properly first, then train FRESH").
iter_21 was launched FRESH with its control penalties live from step 0. With a random policy the raw
commands flip sign on ~50 % of steps, so hand_flip (HAND_FLIP_W 0.4, gated only on "a cup is held") and
palm_sat charge the very first accidental grasps and teach "do not touch the cup". New scalar
ctx.adr_progress (0 until the running success rate first passes 0.3, then +1/30 per increment) exists for
this. RULE: every exploration-killing control charge - act_flip, hand_flip, palm saturation, and the
CARRY increment of the action-rate weights - is multiplied by ramp = smoothstep(adr_progress / 0.2)
(zero for a policy that has not learned the task, full from ADR 6/30). The iter_18 base action-rate
weights (0.01 / 0.005) stay on always - i18 learned FRESH with them. Task-shaping terms (posture, sweep
window, cup speed, overtilt, spill, premature tilt) are NOT ramped: they define the behaviour and must
be present from the start so the habit never forms. Everything else identical to iter_21.

ROUND 24 ADDENDUM (t2r_i22 measured, 2413 epochs / 18.5 h, FRESH, then stopped).
iter_22 reached the pouring geometry and STOPPED there: grasp 0.874/0.867, lift 0.215/0.111 m,
pour_lip_dist 0.053 m (inside the 0.050-0.075 safe window, safe=0.96, approach_gate=1.0), cup
collisions 0.005, wrap_ok 0.97/0.97 - every precondition for the pour is met - yet src_tilt_deg sat at
19-22 deg for 1180 epochs (1233 -> 2413) while entropy decayed 12.0 -> 8.0. in_target, episode_success
and adr stayed exactly 0. reward/tilt_main was EXACTLY 0.000 for the whole run.
ROOT CAUSE (algebraic, verified by evaluating the iter_22 expression):
    tilt_main = 20 * gate * clamp(bell - min(theta,theta_pre)/theta_peak, 0) * safe * lip_above
    bell = rise * decline,  rise = theta/theta_peak,  decline = 1 until theta_rel + 0.10 rad.
For every theta <= theta_pre the decline factor is still 1, so bell == rise == min(theta,theta_pre)/
theta_peak and the subtraction cancels the whole bell: tilt_main is identically zero below theta_pre
(50.4 deg). The only rotation income below 50 deg was tilt_pre (weight 4), so turning the cup from
22 deg to 50 deg paid +1.3 out of a total of 17.8 (7 %), and all the real income sat behind a 50 deg
cliff. iter_18, which DID pour from fresh, had one continuous tilt term: 0.93 at 16 deg, 1.53 at 26 deg,
1.79 at 30 deg, 3.60 at 50 deg, 5.33 at 74 deg - no cliff, and it reached 30 deg at epoch 1158 and its
first bead at 1243.
FIX FOR iter_23 (single change, nothing else touched): the subtraction exists only to avoid paying
twice for the same rotation, so it must remove the PRE-STAGE PAYMENT, not the whole bell:
    tilt_main = TILT_MAIN_W * gate * clamp(bell - (TILT_PRE_W/TILT_MAIN_W) * min(theta,theta_pre)
                                           / theta_peak, 0) * safe * lip_above
Resulting tilt income (fill 0.765, theta_rel 80 deg, gates = 1): 2.3 at 10 deg, 5.2 at 22 deg, 7.0 at
30 deg, 11.7 at 50 deg, 16.4 at 70 deg, peak 19.6 at 86 deg, 9.8 at 100 deg, 0 at 114 deg. Continuous
and monotone up to the fill-dependent peak, then falling - the adaptive sweep the operator asked for,
with no cliff. Weights, gates, overtilt, posture, control ramps are UNCHANGED from iter_22.
The run is FRESH again: the iter_22 policy has entropy 8.0 and a 22 deg habit baked in.
