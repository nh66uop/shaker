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

The previous reward function was:
```python
import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.sqrt((d * d).sum(dim=-1) + 1e-12)


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기. DG-5F pour_bi2 iter_02 구조를 이식하고 파지 항만 RH(iter_00) 형태를 쓴다.
    모든 길이는 ctx.cup_radius 배수(들기 높이 허용만 절대값 예외), 각도는 LIM 배수."""
    tilt_src = ctx.src_cup_tilt
    zeros = torch.zeros_like(tilt_src)
    dt = tilt_src.dtype

    # ---- 공통 척도 ---------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # lip 수평 허가 반경
    wide = 10.0 * r_cup                      # WIDE 커널 (DG-5F 그대로)
    # premature_tilt_limit 대체 스칼라: RH 실측 방출 101° - 20° (DG-5F 는 유출 93.6° 대 한계 p50 73.6°)
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))   # bead_active_count 대체 (RH 20 개 고정)

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    ones = torch.ones_like(tilt_src)

    in_frac = _sat(ctx.bead_in_target_frac)
    state_decay = 1.0 - 0.5 * in_frac        # iter_00 파지 항의 감쇠 (담김이 오르면 상태 수입 축소)

    # ---- RH 에 없는 DG-5F 필드를 RH ctx 로 계산 (env pour_rules.py 와 같은 식) -----
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)                                   # pour_dir_xy
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])    # src_tilt_toward_rcv (부호 있음)
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)    # src_pour_lip_pos
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]

    # ---- A. 접근 (RH iter_00 형태, 팔당 0.4: fresh 학습이라 DG-5F 자릿수) ---------
    d_src = 0.5 * (_dist(ctx.src_palm_pos, ctx.src_cup_pos) + _dist(ctx.src_tips_pos[:, 0, :], ctx.src_cup_pos))
    d_rcv = 0.5 * (_dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos) + _dist(ctx.rcv_tips_pos[:, 0, :], ctx.rcv_cup_pos))
    near_src = 1.0 - torch.tanh(d_src / (4.0 * r_cup))
    near_rcv = 1.0 - torch.tanh(d_rcv / (4.0 * r_cup))
    approach_src = 0.4 * torch.where(ctx.src_grasped, ones, near_src) * state_decay
    approach_rcv = 0.4 * torch.where(ctx.rcv_grasped, ones, near_rcv) * state_decay

    # ---- A. 파지 유지 (팔당 0.5, DG-5F grasp 자릿수) ------------------------------
    grasp_hold = 0.5 * (src_g + rcv_g) * state_decay

    # ---- B. 들기 (RH iter_00 형태; 합 상한 0.5 = DG-5F lift_both 규모) -------------
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # 소스는 높게(K3), 과들기 세금은 충분히 높은 곳에서만 (높이 허용은 절대값 예외)
    lift_src = (0.3 * _sat(h_src / 0.28) - 0.15 * _sat((h_src - 0.45) / 0.20)) * src_g * state_decay
    # 리시버는 떼기만, 낮게
    lift_rcv = 0.2 * _sat(h_rcv / two_r) * torch.exp(-torch.clamp(h_rcv - 0.15, min=0.0) / 0.10) \
        * rcv_g * state_decay
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- C. 두 컵 수렴 (1.2, WIDE 커널, DG-5F 그대로) -----------------------------
    converge_cups = 1.2 * both_g * lifted * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- D. lip 조준 (0.8, DG-5F 그대로) ------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부: pour_target_frac = 1.0 (RH 는 전량 이송) -----------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F. 회전 구동 (8.0): 0° 부터 선형, 1.4*LIM(약 113°)에서 포화 ---------------
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3                          # 바닥 위 가산, 정밀 게이트 아님
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    # grasp_ref_valid 는 RH 에 없어 곱 인자에서 뺀다
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog

    # ---- F. 실제 도착한 비드 (알당 30) --------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 흘림 (알당 40 > 배달 30: 조준 불량 붓기는 순손실) ------------------------
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 마무리: 다 부은 뒤 소스를 세우고 쥐고 있기 (1.5 < 회전 수입 8) ------------
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # ---- 성공 보너스 (env 판정, 재정의 못 함; 기울인 채 버티기 최대 8 보다 크게) ----
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- 조기 기울임: 벽(0.95~1.0 LIM)에서만, lip 이 2r 밖일 때만 (래치 부분 제외) -
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- 실기 안전: 컵끼리 접촉, 포개기 (cup_hit 은 RH 에 없어 뺀다) ----------------
    cup_clash = -(2.0 * torch.tanh(ctx.cup_cup_force / 5.0)
                  + 3.0 * ctx.cups_nested.to(dt))

    # ---- 손은 자기 컵만 (self-collision ON 에서 팔끼리 접촉도 여기로 잡힌다) --------
    hand_foreign = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                           + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 붓는 방향이 몸 바깥(+x)을 향하지 않게 ------------------------------------
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 리시버 컵: 직립, 중심선 기준 자기 쪽(+y) ---------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)   # rcv_side_margin = env-local y
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- G. 관절 여유: 붓는 중에만 작은 가산 (최대 0.3, 회전 구동에 곱하지 않음) ----
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # ---- 제어 정규화 (DG-5F 그대로) ------------------------------------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))
    # adr_progress 없음: 상수 0.1 포화 벌점 (DG-5F 원형대로 전 차원 평균)
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -0.1 * sat

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_hold": grasp_hold,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "converge_cups": converge_cups,
        "aim_lip": aim_lip,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "spill": spill,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "premature": premature,
        "cup_clash": cup_clash,
        "hand_foreign": hand_foreign,
        "pour_dir": pour_dir,
        "rcv_keep": rcv_keep,
        "posture_margin": posture_margin,
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

Observations from watching the trained policy:
# pour_bi_rh 보상 설계 브리프 (2026-09-26 판 — iter_00 재시작용, 이전 24 라운드 전체 종합)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트에 통째로 들어간다.**

이 트랙은 두 번 리셋했다. 1~12 라운드는 `archive_r1-12`, 13~24 라운드는 `archive_r13-24` 에 있다.
지금은 세 번째 사이클의 iter_00 이다. **패치가 아니라 다시 설계한다.** 이전 코드(iter_24, 약 900 줄)는 프롬프트에 넣지 않는다.

## 1. 과제와 고정 조건 (바꿀 수 없다)

- 오른팔(소스, `src_*`)이 비드 20 개가 든 컵을, 왼팔(리시버, `rcv_*`)이 빈 컵을 잡아 든다. 소스를 기울여 비드를 리시버 컵에 옮긴다.
- 성공(env 가 계산, 보상이 재정의 못 함): 담김 ≥ 0.80 · 흘림 ≤ 0.25 · 두 컵 파지 유지 · 받는 컵 ≤ 20° · 20 스텝 유지 · 컵이 포개지지 않음.
- 액션 24 = 소스 palm 6 · 소스 손 6 · 리시버 palm 6 · 리시버 손 6. 손 6 채널 = 구동 관절 하나씩(엄지 외전·엄지 굽힘·검지·중지·약지·새끼, 0 = 폄 · 1 = 쥠). ★09-27 정정: 이전 판의 '손당 3 채널'은 DG-5F 트랙 문구를 잘못 옮긴 것이다.
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
# iter_02 운영 노트 — 2026-09-27 (세 번째 사이클 2 라운드)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다.

## 사용자 지시 (09-27)
- "접근·파지·리프트가 **fresh 에서도 무조건** 되게 해야 한다."
- 이번 사이클 결정 유지: LSTM(horizon 16), self-collision ON, 붓기 구조는 DG-5F pour_bi2 iter_02 이식(iter_01 c5).

## iter_01 (t2r_rh4_i01, fresh LSTM, c5 보상) 결과 — e1436 에서 사용자 지시로 종료
| 지표 (20 epoch 평균) | e0 | e1100 | e1400 |
|---|---|---|---|
| 소스 / 리시버 파지 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| approach_src (최대 0.4) | 0.016 | 0.040 | 0.037 |
| 소스 손 컵 근처 비율 | 0.008 | 0.000 | 0.000 |
| 리시버 손 컵 근처 비율 | 0.009 | 0.085 | 0.032 |
| 손 닫힘 src / rcv | 0.08 / 0.10 | 0.03 / 0.03 | 0.03 / 0.03 |
| 흘림 / 컵 충돌 / 손 접촉 벌점 | −0.131 / −0.102 / −0.075 | −0.006 / −0.013 / −0.004 | −0.002 / −0.006 / −0.002 |

- 9,400 만 프레임 동안 한 번도 컵을 잡지 못했다. 손을 편 채 컵에서 멀어졌고, 보상 상승은 전부 벌점 회피였다.
- 원인(코드 대조): c5 의 파지 쪽은 `approach`(손바닥·엄지와 컵 중심 거리, 0.4) + `grasp_hold`(성립해야만 0.5) 뿐이다.
  **손끝을 컵 벽에 대는 중간 수입, 손을 닫는 수입, 성립 전 접촉에 대한 단계적 수입이 없다.**
  반면 흘림은 파지 전부터 **비드당 −40** 이다. 컵은 한쪽 접촉 0.3~0.4 N 에 넘어진다(09.17 프로브).
  "근처에 가서 건드리면 큰 손해, 닫아도 이득 없음" 이라 가지 않는 쪽이 최적이었다.
- DG-5F iter_02 는 파지하는 체크포인트에서 이어받는 전제였다(t2r2_i02 = i01 ep250 웜). fresh 에는 파지 항이 따로 필요하다.

## 이 트랙에서 파지를 실제로 배운 보상 (측정)
| 런 | 시작 | 소스 파지 첫 0.05 / 0.3 | 리시버 파지 | 끝 src / rcv |
|---|---|---|---|---|
| t2r_rh_i00_r1 (1 차 사이클) | fresh | e248 / e474 | 없음 | 0.37 / 0.00 |
| t2r_rh_i03 (1 차 사이클) | fresh | e556 / e831 | 없음 | 0.27 / 0.00 |
| t2r_rh2_i03 | rh2_i01 ep1250(fresh, 접근 0.93·파지 0) 이어받음 | +e1753 / +e2154 | 없음 | 0.84 / 0.00 |
| t2r_rh2_i04 | rh2_i03 ep2000 | 곧바로 | **e363** | 0.52 / 0.55 |

- 위 런은 모두 MLP · horizon 64 · self-collision OFF 였다. 이번은 LSTM · horizon 16 · self-collision ON 이다.
- rh2_i01(fresh)은 접근 0.93 에서 **엄지 힘 0 N 으로 주차**했다. rh2_i03 은 이것을 "한쪽 면(엄지 / 나머지)마다 따로 점수를 매기고,
  **접촉력이 생겨야 그 면이 완성**" 되게 바꿔 소스 파지 0.84 를 만들었다.
- 리시버는 fresh 에서 한 번도 파지하지 못했다. rh2_i03 에서는 **소스 들기 수입이 켜지자(e2000→2500) 리시버가 컵을 버렸다**
  (근처 비율 0.87 → 0.05). rh2_i04 는 **소스의 큰 수입(들기 이후)을 리시버 진행도로 게이트**하고, 리시버 근처 벌점 합을
  파지 자세 수입보다 확실히 작게 줄여 리시버 파지 0.55 를 만들었다.
- 컵 사실: 바닥 지름 37.8 mm, 0.134 kg, **한쪽 접촉 0.3~0.4 N 에 넘어진다. 양쪽 동시 접촉은 넘어지지 않는다.**

## iter_02 에서 할 것 (접근·파지·들기를 fresh 에서 확실히)
1. **파지 블록은 rh2_i03 의 형태를 그대로 쓴다**(아래 원문): `_grip_geometry`(손끝 컵 축 기준 높이·반경·맞은편 배치·깔때기),
   `_hand_cup_terms` 의 reach(먼 깔때기 + 면별 손끝-벽, 힘으로 완성) · close(파지 자세에서 닫힘) · grasp(성립 전 접촉도 단계적) ·
   lift / held / up_w · slide · one_sided · palm_push · rush · fling. 가중 W_REACH 1 · W_CLOSE 0.3 · W_GRASP 2 · W_LIFT 3 의
   **자릿수와 사다리 순서(접근 < 파지 < 들기)를 유지**한다. TIP_R 같은 손 치수 상수는 rh2_i03 주석의 측정값을 그대로 쓰고 출처를 적는다.
2. **리시버를 버리지 않게 한다**(rh2_i04 교훈): 소스의 들기 이후 수입(들기·수렴·조준·회전)은 **리시버 진행도**(리시버 reach → 파지 → 들기)로
   게이트한다(곱이 아니라 바닥이 있는 계수로 — 곱 사다리 금지). 리시버의 컵 근처 벌점 합은 리시버 파지 자세 수입보다 확실히 작게 둔다.
3. **파지 전 벌점은 rh2_i03 수준으로 낮춘다.** 흘림은 파지 성립 전 비드당 약 −3(rh2_i03 W_SPILL 60), 두 컵을 쥔 뒤에는 c5 의 −40.
   컵 충돌·손 접촉도 파지 전에는 rh2_i03 의 크기(cup_collision 1.0 · hand_foreign 0.5)를 넘지 않게.
4. **붓기 구조는 iter_01 c5(아래 원문)를 그대로** 두 컵을 쥐고 든 뒤에 적용한다: converge_cups · aim_lip · pour_rot(0° 부터 선형,
   조준 바닥 0.3) · pour_delta 30 · spill −40 · success 20 · finish_upright · premature · pour_dir · rcv_keep. c5 의
   approach_src/rcv · grasp_hold · lift_src/lift_rcv 는 1 번의 rh2_i03 블록으로 **대체**한다(중복 금지).
5. **주차 금지**(memory): 상태 비례 수입은 컵에 손을 댈 때까지 커지지 않게 한다 — rh2_i01 이 reach 0.93 에서 주차했다.
   alive 를 쓰면 rh2_i03 처럼 1.0 이하로 두고, alive + reach 만으로 버는 수입이 파지 자세 수입보다 작아야 한다.
6. 항 수 30 개 안팎까지 허용한다(파지 블록이 팔별로 늘어나므로). 새 ctx 필드를 가정하지 않는다.

## rh2_i04 가 받은 운영자 요구 (원문, 참고)
## What the operator wants from iter_04
1. The receiver must keep approaching and must grasp. Make receiver progress a precondition for the big source pay: e.g. cap or gate lift_src (and later align/tilt) by receiver stage (reach_rcv, then rcv grasp), so that lifting the source while the receiver is away is worth little.
2. Near-cup penalties for the receiver (rush, slide, tilt, palm push, one-sided) must in total stay clearly below what the receiver earns by being in the grasp pose, and leaving the cup must cost clearly more than staying. Do not remove knock-over protection; rescale it.
3. Keep the source-hand reach/grasp/lift formulation that worked.
4. After both cups are held, there must be a gradient that brings the source mouth over the receiver (cups_center_dist stayed 0.34 m with align 0).



## 참고 코드 1 — rh2_i03 보상 원문 (파지 블록을 여기서 가져온다)
```text
import torch
import math


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (0/336 knock-overs at +45 mm) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm, inside the 0-25 mm lift group)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin (no lift with thumb >= 50 mm)

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    # gap is zero only at the measured contact radius (tip frame 26-28 mm, cup outer 28.5 mm);
    # the old +10 mm pad made 33-37 mm count as "on the wall" and killed the gradient
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)        # (N,)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values   # nearest non-thumb tip, (N,)

    # opposition: 1 when the thumb and the other fingers sit on opposite sides of the cup axis
    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # (N,)

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup_terms(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok, lift_target):
    """Unweighted per-arm terms, each in [0, 1]. pose_ok (N,) = cup attitude is acceptable for this stage."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: long-range funnel term + per-side fingertip-on-wall term.
    #    Each side is scored on its own (sum form) so the thumb keeps its full gradient while the finger
    #    side is already done, and measured force completes a side: reach saturates only with contact
    #    force on BOTH sides (the parked pose of i01 had thumb 0.00 N and reach 0.97).
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: balanced two-sided pinch. Before the flag, finger-first touch earns 0.15 and any thumb force
    #    added to it can only raise the value (0.5 N -> +0.19); thumb-only still earns nothing.
    #    Flag boundary (1 N / 1 N): pre_grasp 0.355 -> flagged 0.75, a step up, never a valley.
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

    # closure shaping: measured grasps close to 0.68-0.98, the parked hand sat at 0.19 with the closure
    # channels driven open; pays only once the pinch centre is at the grasp pose (contact freeze makes
    # closing there harmless), saturates at 0.7 so over-squeezing is not rewarded
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift: measured cup elevation only, clamped, and only while really held (explosion-proof)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)                # 5 mm dead band: tipping on the rim is not lifting
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * torch.exp(-(geo["d_grip"] / 0.08) ** 2) * calm
    lift_prog = 0.5 * torch.tanh(h_pos / 0.02) + 0.5 * torch.clamp(h_pos / lift_target, max=1.0)
    lift = held * pose_ok * lift_prog
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)            # "held and at least 5 cm up"

    # penalties (unweighted, bounded)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)       # 1 N dead band: a slightly stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)             # finger-only contact: adding thumb force lowers this
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "lift": lift, "up_w": up_w, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def _pour_geometry(ctx, outer_r):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    align_gate = torch.exp(-(d_align / 0.06) ** 2)                       # wide: tilting is allowed here
    pour_gate = torch.exp(-(d_align / max(ctx.cup_radius, 0.015)) ** 2)  # narrow: beads would land inside
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Stage ladder (per step, cumulative): reach 1 < grasp 2 < lift 3 per arm, then align 4 < tilt 5.
    # A later stage keeps every earlier term, so progress never lowers the reward.
    W_REACH, W_GRASP, W_LIFT = 1.0, 2.0, 3.0
    W_CLOSE = 0.3         # closure shaping at the grasp pose; small next to grasp 2.0, only a gradient helper
    W_ALIGN, W_TILT = 4.0, 5.0
    W_POUR = 150.0        # x d_in_target: 7.5 per bead with 20 beads, one-time, dominates the dense terms at that step
    W_SPILL = 60.0        # x d_spill: 3 per bead, below the pour pay so an imperfect pour is still worth trying
    W_SUCCESS = 20.0      # per step while the env reports success
    W_ALIVE = 1.0         # with reach it outweighs all persistent penalties, so ending the episode never pays
    W_SLIDE, W_TILT_PEN = 0.5, 0.3
    W_ONE_SIDED, W_PALM, W_RUSH, W_FLING = 0.3, 0.2, 0.2, 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05     # x mean squared action change (<= 4), regularisation only
    CUP_OUTER_R = 0.0285  # cup wall, used for the pouring lip
    TIP_R = 0.027         # tip-frame radius at measured two-sided contact (thumb 27.9 mm, nearest finger 26.2 mm)
    UPRIGHT_TOL = 0.35    # rad, = the 20 deg success limit on the receiver cup
    TILT_FULL = math.radians(100.0)   # half the beads are out by 86 deg, reward saturates a little past that
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.15, 0.08   # source ends above the receiver

    dtype = ctx.src_cup_tilt.dtype
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R)
    upright_src = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = align_gate + (1.0 - align_gate) * upright_src   # source may tilt only over the receiver
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                          ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up,
                          ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_src, LIFT_TARGET_SRC)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                          ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                          ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_rcv, LIFT_TARGET_RCV)

    nested = ctx.cups_nested.to(dtype)
    both_up = src["up_w"] * rcv["up_w"] * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(4.0 * d_align)) + 0.5 * torch.exp(-25.0 * d_align))
    pour_tilt = both_up * pour_gate * torch.clamp(ctx.src_cup_tilt / TILT_FULL, max=1.0)

    tilt_pen_src = (1.0 - align_gate) * torch.tanh(ctx.src_cup_tilt / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "reach_src": W_REACH * src["reach"],
        "reach_rcv": W_REACH * rcv["reach"],
        "close_src": W_CLOSE * src["close"],
        "close_rcv": W_CLOSE * rcv["close"],
        "grasp_src": W_GRASP * src["grasp"],
        "grasp_rcv": W_GRASP * rcv["grasp"],
        "lift_src": W_LIFT * src["lift"],
        "lift_rcv": W_LIFT * rcv["lift"],
        "align_mouths": W_ALIGN * align,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
        "contact_push": -(W_ONE_SIDED * (src["one_sided"] + rcv["one_sided"])
                          + W_PALM * (src["palm_push"] + rcv["palm_push"])),
        "rush_near_cup": -W_RUSH * (src["rush"] + rcv["rush"]),
        "cup_fling": -W_FLING * (src["fling"] + rcv["fling"]),
        "cup_collision": -W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0),
        "hand_foreign": -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                      + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)),
        "nested": -W_NESTED * nested,
        "action_rate": -W_ACT_RATE * act_rate,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    return reward, comps

```

## 참고 코드 2 — iter_01 c5 원문 (붓기 구조를 여기서 가져온다)
```text
import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.sqrt((d * d).sum(dim=-1) + 1e-12)


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기. DG-5F pour_bi2 iter_02 구조를 이식하고 파지 항만 RH(iter_00) 형태를 쓴다.
    모든 길이는 ctx.cup_radius 배수(들기 높이 허용만 절대값 예외), 각도는 LIM 배수."""
    tilt_src = ctx.src_cup_tilt
    zeros = torch.zeros_like(tilt_src)
    dt = tilt_src.dtype

    # ---- 공통 척도 ---------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # lip 수평 허가 반경
    wide = 10.0 * r_cup                      # WIDE 커널 (DG-5F 그대로)
    # premature_tilt_limit 대체 스칼라: RH 실측 방출 101° - 20° (DG-5F 는 유출 93.6° 대 한계 p50 73.6°)
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))   # bead_active_count 대체 (RH 20 개 고정)

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    ones = torch.ones_like(tilt_src)

    in_frac = _sat(ctx.bead_in_target_frac)
    state_decay = 1.0 - 0.5 * in_frac        # iter_00 파지 항의 감쇠 (담김이 오르면 상태 수입 축소)

    # ---- RH 에 없는 DG-5F 필드를 RH ctx 로 계산 (env pour_rules.py 와 같은 식) -----
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)                                   # pour_dir_xy
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])    # src_tilt_toward_rcv (부호 있음)
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)    # src_pour_lip_pos
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]

    # ---- A. 접근 (RH iter_00 형태, 팔당 0.4: fresh 학습이라 DG-5F 자릿수) ---------
    d_src = 0.5 * (_dist(ctx.src_palm_pos, ctx.src_cup_pos) + _dist(ctx.src_tips_pos[:, 0, :], ctx.src_cup_pos))
    d_rcv = 0.5 * (_dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos) + _dist(ctx.rcv_tips_pos[:, 0, :], ctx.rcv_cup_pos))
    near_src = 1.0 - torch.tanh(d_src / (4.0 * r_cup))
    near_rcv = 1.0 - torch.tanh(d_rcv / (4.0 * r_cup))
    approach_src = 0.4 * torch.where(ctx.src_grasped, ones, near_src) * state_decay
    approach_rcv = 0.4 * torch.where(ctx.rcv_grasped, ones, near_rcv) * state_decay

    # ---- A. 파지 유지 (팔당 0.5, DG-5F grasp 자릿수) ------------------------------
    grasp_hold = 0.5 * (src_g + rcv_g) * state_decay

    # ---- B. 들기 (RH iter_00 형태; 합 상한 0.5 = DG-5F lift_both 규모) -------------
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # 소스는 높게(K3), 과들기 세금은 충분히 높은 곳에서만 (높이 허용은 절대값 예외)
    lift_src = (0.3 * _sat(h_src / 0.28) - 0.15 * _sat((h_src - 0.45) / 0.20)) * src_g * state_decay
    # 리시버는 떼기만, 낮게
    lift_rcv = 0.2 * _sat(h_rcv / two_r) * torch.exp(-torch.clamp(h_rcv - 0.15, min=0.0) / 0.10) \
        * rcv_g * state_decay
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- C. 두 컵 수렴 (1.2, WIDE 커널, DG-5F 그대로) -----------------------------
    converge_cups = 1.2 * both_g * lifted * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- D. lip 조준 (0.8, DG-5F 그대로) ------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부: pour_target_frac = 1.0 (RH 는 전량 이송) -----------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F. 회전 구동 (8.0): 0° 부터 선형, 1.4*LIM(약 113°)에서 포화 ---------------
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3                          # 바닥 위 가산, 정밀 게이트 아님
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    # grasp_ref_valid 는 RH 에 없어 곱 인자에서 뺀다
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog

    # ---- F. 실제 도착한 비드 (알당 30) --------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 흘림 (알당 40 > 배달 30: 조준 불량 붓기는 순손실) ------------------------
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 마무리: 다 부은 뒤 소스를 세우고 쥐고 있기 (1.5 < 회전 수입 8) ------------
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # ---- 성공 보너스 (env 판정, 재정의 못 함; 기울인 채 버티기 최대 8 보다 크게) ----
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- 조기 기울임: 벽(0.95~1.0 LIM)에서만, lip 이 2r 밖일 때만 (래치 부분 제외) -
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- 실기 안전: 컵끼리 접촉, 포개기 (cup_hit 은 RH 에 없어 뺀다) ----------------
    cup_clash = -(2.0 * torch.tanh(ctx.cup_cup_force / 5.0)
                  + 3.0 * ctx.cups_nested.to(dt))

    # ---- 손은 자기 컵만 (self-collision ON 에서 팔끼리 접촉도 여기로 잡힌다) --------
    hand_foreign = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                           + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 붓는 방향이 몸 바깥(+x)을 향하지 않게 ------------------------------------
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 리시버 컵: 직립, 중심선 기준 자기 쪽(+y) ---------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)   # rcv_side_margin = env-local y
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- G. 관절 여유: 붓는 중에만 작은 가산 (최대 0.3, 회전 구동에 곱하지 않음) ----
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # ---- 제어 정규화 (DG-5F 그대로) ------------------------------------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))
    # adr_progress 없음: 상수 0.1 포화 벌점 (DG-5F 원형대로 전 차원 평균)
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -0.1 * sat

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_hold": grasp_hold,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "converge_cups": converge_cups,
        "aim_lip": aim_lip,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "spill": spill,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "premature": premature,
        "cup_clash": cup_clash,
        "hand_foreign": hand_foreign,
        "pour_dir": pour_dir,
        "rcv_keep": rcv_keep,
        "posture_margin": posture_margin,
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
