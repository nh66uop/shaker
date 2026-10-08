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


def _norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(N,K) -> (N,) L2 노름 (0 에서 안전)."""
    return torch.sqrt((v * v).sum(dim=1) + eps)


def compute_reward(ctx) -> "tuple[torch.Tensor, dict[str, torch.Tensor]]":
    eps = 1e-8
    r = ctx.cup_radius                      # 조준·배치의 유일한 길이 스케일 (컵 반지름 배수만 쓴다)

    # ---- 가중치 (예산 규칙 5-1) -------------------------------------------------
    # 상태 비례 수입 합 상한 ≈ 1.4/step, 회전·배달 수입은 붓는 동안 4~20/step.
    W_APPROACH = 0.15    # 팔별 접근 (파지 ckpt 가 이미 안다 — 작게)
    W_GRASP = 0.15       # 팔별 파지 유지
    W_LIFT_SRC = 0.20    # 소스는 높게 (K3: 높이가 회전 여유를 만든다)
    W_OVERLIFT = 0.10    # 과들기 세금은 충분히 높은 곳에서만
    W_LIFT_RCV = 0.10    # 리시버는 테이블에서 떼기만, 낮게
    W_PLACE = 0.25       # 배치 유지 (차분형의 빈틈 보완, i23·i24)
    W_AIM_PROG = 1.5     # 조준 차분(진행도), 대칭 clamp ±0.3 m/s → ±0.45
    W_AIM_HOLD = 0.25    # 도달한 조준 자리를 지키는 작은 상태형 수입
    W_DIR = 0.8          # 방향 비용 = 기울기 크기 × 방향 어긋남 (i17 에서 지웠다가 실패)
    W_POUR = 8.0         # ★ 방출 영역 회전 수입 — 나머지 전부의 합보다 크게
    W_POUR_RATE = 1.0    # ★ 받는 컵 쪽으로 도는 속도 (첫 한 알 전의 경사)
    W_MARGIN = 1.0       # 한계 여유를 지키며 기울인 자세를 더 값싸게 (K3·K7, i15 식 직접 벌점 금지)
    W_DELIVER = 300.0    # ★ 실제 이송 증분: 비드 1 알 = 0.05 → 15
    W_IN_LEVEL = 1.0     # 담긴 양 유지 (20 스텝 유지 조건 보조, 상한 1.0)
    W_SUCCESS = 3.0      # 성공 유지 보너스 (env 판정, 재정의 못 함)
    W_SPILL = 120.0      # 흘림 증분: 1 알 = 6 < 배달 1 알 = 15 (i10~12 과벌점 붕괴 회피)
    W_RCV_UP = 0.8       # 받는 컵 직립 (성공 조건)
    W_NESTED = 1.5       # 컵 포개기는 이송이 아니다
    W_FALL = 1.0         # 컵을 떨어뜨림
    W_CUPCUP = 1.0       # 실기 안전: 컵끼리 접촉
    W_FOREIGN = 0.75     # 실기 안전: 남의 것 건드리기 (팔별)
    W_CUPVEL = 0.25      # 컵 병진 속도 (팔별)
    W_SPIN = 0.25        # 컵 자기축 회전 — 붓기에 무용 (i22)
    W_QD = 0.25          # 팔 관절 속도 (팔별)
    W_QDD = 0.15         # 팔 관절 가속 (팔별)
    W_ACT_RATE = 0.3     # 명령 요동
    W_ACT_SAT = 0.3      # palm 포화
    W_ACT_FLIP = 0.3     # 부호 뒤집기

    T_REL = math.radians(70.0)   # 환경 설명: 가득 찬 컵에서 첫 알이 약 70°
    T_REL_W = math.radians(12.0) # 80° 1/5 · 86° 절반 을 덮는 폭 (상한·벌점 없음)

    # ---- 기본 량 ----------------------------------------------------------------
    tilt = ctx.src_cup_tilt                                  # (N,)
    zeros = torch.zeros_like(tilt)
    ones = torch.ones_like(tilt)

    src_pos = ctx.src_cup_pos
    rcv_pos = ctx.rcv_cup_pos
    lip = ctx.src_cup_mouth_pos                              # 소스 주둥이(림) 중심
    mouth = ctx.rcv_cup_mouth_pos                            # 받는 컵 입구 중심

    src_held = ctx.src_grasped.to(tilt.dtype)
    rcv_held = ctx.rcv_grasped.to(tilt.dtype)
    both_held = src_held * rcv_held
    valid = 1.0 - ctx.cups_nested.to(tilt.dtype)             # 포개짐은 이송 무효
    src_frac = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    in_frac = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)

    # 상태 비례 수입은 담김이 오르면 줄어든다 (5-1)
    state_decay = 1.0 - 0.5 * in_frac

    # ---- D 조준: 주둥이가 입구 위(lip_dz>0) · 수평으로 입구 안 (r 배수 창) -------
    e_xy = (mouth - lip)[:, :2]
    lip_xy = torch.sqrt((e_xy * e_xy).sum(dim=1) + eps)      # 주둥이-입구 수평 거리
    lip_dz = lip[:, 2] - mouth[:, 2]                         # 기울면 r·sinθ 만큼 내려감(K6)이 이미 반영됨
    gate_xy = torch.exp(-(lip_xy / (2.0 * r)) ** 2)          # 좁은 방출 허가 창 (i13 골격)
    gate_dz = torch.clamp(lip_dz / r, 0.0, 1.0)              # 입구보다 위일 때만
    aim_gate = gate_xy * gate_dz

    # ---- E 방향: 컵 up 축의 수평 성분이 기우는 방위 -----------------------------
    up_xy = ctx.src_cup_up[:, :2]
    up_xy_n = torch.sqrt((up_xy * up_xy).sum(dim=1) + eps)
    to_rcv_xy = (mouth - src_pos)[:, :2]
    to_rcv_n = torch.sqrt((to_rcv_xy * to_rcv_xy).sum(dim=1) + eps)
    dir_cos = (up_xy * to_rcv_xy).sum(dim=1) / (up_xy_n * to_rcv_n)   # 직립 근처에서는 잡음 → 아래서 tilt 로 가중
    align = torch.clamp(dir_cos, 0.0, 1.0)

    # ---- A 파지 ------------------------------------------------------------------
    d_src = 0.5 * (_norm(ctx.src_palm_pos - src_pos) + _norm(ctx.src_tips_pos[:, 0, :] - src_pos))
    d_rcv = 0.5 * (_norm(ctx.rcv_palm_pos - rcv_pos) + _norm(ctx.rcv_tips_pos[:, 0, :] - rcv_pos))
    near_src = 1.0 - torch.tanh(d_src / (4.0 * r))
    near_rcv = 1.0 - torch.tanh(d_rcv / (4.0 * r))
    approach_src = W_APPROACH * torch.where(ctx.src_grasped, ones, near_src) * state_decay
    approach_rcv = W_APPROACH * torch.where(ctx.rcv_grasped, ones, near_rcv) * state_decay
    grasp_hold = W_GRASP * (src_held + rcv_held) * state_decay

    # ---- B 들기 (높이 허용만 절대 상수 예외) ------------------------------------
    h_src = src_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = rcv_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_src = W_LIFT_SRC * torch.clamp(h_src / 0.28, 0.0, 1.0) \
        - W_OVERLIFT * torch.clamp((h_src - 0.45) / 0.20, 0.0, 1.0)
    lift_src = lift_src * src_held * state_decay
    lift_rcv = W_LIFT_RCV * torch.clamp(h_rcv / (2.0 * r), 0.0, 1.0) \
        * torch.exp(-torch.clamp(h_rcv - 0.15, min=0.0) / 0.10) * rcv_held * state_decay

    # ---- C 배치: 소스 기준 상대 좌표, 컵 반지름 배수 (합 형태 — 곱 사다리 회피) ---
    p_z = torch.clamp((src_pos[:, 2] - mouth[:, 2]) / (2.0 * r), 0.0, 1.0)          # 받는 컵이 더 낮게
    d_cup_xy = torch.sqrt(((src_pos - rcv_pos)[:, :2] ** 2).sum(dim=1) + eps)
    p_xy = torch.exp(-((d_cup_xy - 3.0 * r) / (2.0 * r)) ** 2)                      # 붓기 가능한 간격, 포개짐 아님
    y_mid = 0.5 * (src_pos[:, 1] + rcv_pos[:, 1])
    p_mid = torch.exp(-(y_mid / (4.0 * r)) ** 2)                                    # 양팔 작업은 중심선 근처
    place_hold = W_PLACE * (0.4 * p_z + 0.4 * p_xy + 0.2 * p_mid) * both_held * state_decay

    # 조준 차분(진행도): -lip_xy 의 시간미분 ≈ 상대속도의 접근 성분, clamp 대칭(i20)
    dir_hat = e_xy / lip_xy.unsqueeze(1)
    v_rel_xy = (ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel)[:, :2]
    closing = (v_rel_xy * dir_hat).sum(dim=1)
    aim_progress = W_AIM_PROG * torch.clamp(closing, -0.3, 0.3) * both_held
    aim_hold = W_AIM_HOLD * aim_gate * both_held * state_decay

    # ---- E 방향 비용: 기울기 크기 × 방향 어긋남 ---------------------------------
    tilt_norm = torch.clamp(tilt / (0.5 * math.pi), 0.0, 1.0)
    dir_cost = -W_DIR * tilt_norm * (0.5 * (1.0 - dir_cos)) * src_frac

    # ---- F 회전·배달 (주역) -----------------------------------------------------
    # 방출 영역: 70° 중심 단조 증가, 목표각·대역·상한을 처방하지 않는다.
    rel = 0.5 * (1.0 + torch.tanh((tilt - T_REL) / T_REL_W))
    pour_gate = both_held * valid * aim_gate * align          # 창 안 + 방향 일치일 때만 지불
    pour_open = W_POUR * pour_gate * rel * src_frac

    # 받는 컵 쪽으로 기울이는 각속도 성분: 축 = ẑ × t̂ (t̂ = 받는 컵 수평 방향).
    # 대칭 clamp 로 되돌림 펌프가 닫힌 경로에서 이득이 안 되게 한다.
    t_hat = to_rcv_xy / to_rcv_n.unsqueeze(1)
    w = ctx.src_cup_ang_vel
    tilt_rate = -w[:, 0] * t_hat[:, 1] + w[:, 1] * t_hat[:, 0]
    pour_rate = W_POUR_RATE * torch.clamp(tilt_rate, -2.0, 2.0) * both_held * valid * aim_gate * src_frac

    deliver_delta = W_DELIVER * torch.clamp(ctx.d_in_target, -1.0, 1.0) * valid
    in_target_level = W_IN_LEVEL * in_frac * valid
    success_bonus = W_SUCCESS * ctx.success.to(tilt.dtype)

    # ---- G 자세 여유: 한계에 기대지 않은 기울임을 더 값싸게 (직접 벌점 아님) ------
    m_src = ctx.src_arm_limit_margin
    margin = 0.5 * (m_src.mean(dim=1) + torch.min(m_src, dim=1).values)
    posture_margin = W_MARGIN * rel * aim_gate * both_held * torch.clamp(margin, 0.0, 1.0)

    # ---- H 흘림·안전·제어 --------------------------------------------------------
    spill_delta = -W_SPILL * torch.clamp(ctx.d_spill, 0.0, 1.0)
    rcv_upright = -W_RCV_UP * torch.tanh(ctx.rcv_cup_tilt / math.radians(20.0))
    nested_pen = -W_NESTED * ctx.cups_nested.to(tilt.dtype)
    fell = ((src_pos[:, 2] < ctx.src_cup_spawn_pos[:, 2] - 0.5 * r).to(tilt.dtype)
            + (rcv_pos[:, 2] < ctx.rcv_cup_spawn_pos[:, 2] - 0.5 * r).to(tilt.dtype))
    cup_fall = -W_FALL * fell
    cup_cup = -W_CUPCUP * torch.tanh(ctx.cup_cup_force / 5.0)
    foreign = -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    spin_src = torch.abs((ctx.src_cup_ang_vel * ctx.src_cup_up).sum(dim=1))   # 자기축 회전은 붓기에 무용
    spin_rcv = torch.abs((ctx.rcv_cup_ang_vel * ctx.rcv_cup_up).sum(dim=1))
    cup_motion = -W_CUPVEL * (torch.tanh(_norm(ctx.src_cup_lin_vel) / 0.8)
                              + torch.tanh(_norm(ctx.rcv_cup_lin_vel) / 0.8)) \
        - W_SPIN * (torch.tanh(spin_src / 4.0) + torch.tanh(spin_rcv / 4.0))

    qd_s = torch.sqrt((ctx.src_arm_qd ** 2).mean(dim=1) + eps)
    qd_r = torch.sqrt((ctx.rcv_arm_qd ** 2).mean(dim=1) + eps)
    qdd_s = torch.sqrt((ctx.src_arm_qdd ** 2).mean(dim=1) + eps)
    qdd_r = torch.sqrt((ctx.rcv_arm_qdd ** 2).mean(dim=1) + eps)
    arm_motion = -W_QD * (torch.tanh(qd_s / 3.0) + torch.tanh(qd_r / 3.0)) \
        - W_QDD * (torch.tanh(qdd_s / 40.0) + torch.tanh(qdd_r / 40.0))

    da = ctx.actions - ctx.prev_actions
    act_rate = -W_ACT_RATE * da.abs().mean(dim=1)
    sat = (torch.clamp(ctx.actions.abs() - 0.85, min=0.0) / 0.15).mean(dim=1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=1)
    act_sat = -W_ACT_SAT * sat - W_ACT_FLIP * flip

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_hold": grasp_hold,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "place_hold": place_hold,
        "aim_progress": aim_progress,
        "aim_hold": aim_hold,
        "dir_cost": dir_cost,
        "pour_open": pour_open,
        "pour_rate": pour_rate,
        "posture_margin": posture_margin,
        "deliver_delta": deliver_delta,
        "in_target_level": in_target_level,
        "success_bonus": success_bonus,
        "spill_delta": spill_delta,
        "rcv_upright": rcv_upright,
        "nested_pen": nested_pen,
        "cup_fall": cup_fall,
        "cup_cup": cup_cup,
        "foreign": foreign,
        "cup_motion": cup_motion,
        "arm_motion": arm_motion,
        "act_rate": act_rate,
        "act_sat": act_sat,
    }

    reward = torch.zeros_like(tilt)
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
# iter_01 운영 노트 — 2026-09-27

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. 앞의 DESIGN_BRIEF 가 이 트랙 전체의 측정·교훈이다.

## iter_00 이 한 것과 남긴 것 (t2r_rh4_i00, server GPU1, 파지 단계 ckpt 웜스타트)

★**이 트랙에서 처음으로 붓기가 성립했다.** 결정론 평가 ep1300(64 env · 900 스텝 · 동일 조건):
도착 env 64/64 · 도착 비드 1089/1280(85 %) · 리시버 담김 최대 중앙 0.900 · 방향 오차 중앙 12° ·
전 env 가 90° 를 넘겨 기울임 · 소스 손목 자세는 한계에 붙지 않았다.
직전 라운드(i24)의 최고가 1280 알 중 1 알이었다.

### 그런데 비드를 다시 가져온다 (측정)

ep1300 trace(64 env) 시간축 중앙값:

| t | 담김 | 소스 잔량 | 소스 기울기 | 리시버 기울기 |
|---|---|---|---|---|
| 300 | 0.000 | 1.000 | 93° | 18° |
| 375 | 0.250 | 0.650 | 99° | 3° |
| 450 | 0.725 | 0.075 | 105° | 4° |
| 525 | 0.375 | 0.375 | 56° | 4° |
| 600 | 0.000 | 1.000 | 31° | 4° |
| 750 | 0.000 | 1.000 | 82° | 6° |

에피소드 끝 담김 0.135(최대 0.900). 붓고-되받기 사이클 env 당 중앙 1 회(최대 2 회). 끝 흘림 0.000 —
비드는 밖으로 새는 게 아니라 **소스 컵으로 되돌아간다**.

### 왜 되받는가 (에피소드 수입, ep1300 trace, env 중앙)

| 항 | 에피소드 합 | 비중 |
|---|---|---|
| pour_open | 1787 | 65.7 % |
| deliver_delta | 300 | 11.0 % |
| success_bonus | 237 | 8.7 % |
| rcv_upright | −227 | −8.4 % |
| in_target_level | 147 | 5.4 % |

- `pour_open = W_POUR · 창 · 방향 · rel(기울기) · **src_frac**` 는 **상태형**이고 소스 잔량에 비례한다.
  소스가 비면 스텝당 최대 8 짜리 수입이 끊긴다 → 계속 벌려면 비드를 되받아야 한다.
- `deliver_delta` 는 양(+) 합 300 · 음(−) 합 **0** 이다. env 의 `d_in_target` 은 "건너간 비드"의 단조 증분이라
  되받아도 음수 지급이 없다. 되받기에 비용이 전혀 없다.
- 되돌아간 비드 비율 중앙 0.95(들어간 총량 1.00).

### 그 뒤 400~750 epoch 동안 일어난 일 (학습 곡선 20점 평균)

| 지표 | e1330 | e1710 | e2084 |
|---|---|---|---|
| 배달 비드 비율 | 0.338 | 0.297 | **0.114** |
| 담김 최대 | 0.694 | 0.638 | **0.414** |
| success_bonus | 0.807 | 0.733 | **0.181** |
| 흘림(끝) | 0.153 | 0.174 | **0.235** |
| pour_open | 1.730 | 1.956 | **3.326** |
| 총보상 | 3.315 | 3.461 | **4.244** |
| entropy | 1.23 | −1.26 | −2.73 |

결정론 평가도 같은 방향이다. ep1700: 도착 env 56/64 · 도착 비드 828/1280(65 %) · 담김 최대 중앙 0.750 ·
끝 담김 0.095. 같은 구간에 소스 파지 유지는 0.70 → 0.97 로 **좋아졌다** —
정책은 "컵을 더 단단히 쥐고 더 오래 기울이기"로 갔고, 그것이 `pour_open` 을 최대로 만든다.

**보상은 28 % 오르는데 배달은 66 % 줄었다.** 보상이 과제를 대변하지 못한다는 직접 증거다.

### 마지막 상태 (ep3100 결정론, 런 종료 시점 e3254)

| | ep1300 | ep1700 | ep3100 |
|---|---|---|---|
| 도착 비드 | 1089/1280 (85 %) | 828 (65 %) | **178 (14 %)** |
| 도착 env | 64/64 | 56/64 | 44/64 |
| 담김 최대 중앙 | 0.900 | 0.750 | 0.100 |
| 에피소드 끝 담김 | 0.135 | 0.095 | 0.000 |
| 방향 오차 중앙 | 12° | 14° | **6°** |
| 최대 기울기 중앙 | 119.8° | 119.0° | 89.5° |
| 소스 파지 유지 | 0.70 | 0.97 | **0.99** |

ep3100 에서 정책이 하는 일: 두 컵을 거의 완벽하게 쥐고(0.99/1.00), 주둥이를 받는 컵 입구 위
**수평 1.0 cm · 높이 +7.4 cm** 에 맞추고, 방향 오차 **6°** 로, **89.5°** 까지 기울인 자세를
전체 스텝의 **79 %** 동안 유지한다. 그런데 **소스 컵 잔량 중앙이 0.900** — 비드가 나가지 않는다.

즉 되받기(ep1300)에서 한 발 더 나아가 **아예 붓지 않는 자세**로 수렴했다.
`pour_open = W_POUR · 창 · 방향 · rel(기울기) · src_frac` 에서 `src_frac` 은 소스에 남은 비드라,
**비드를 흘리지 않을수록 이 수입이 오래 유지된다.** 붓는 순간 수입이 사라지므로 "붓기 직전 자세"가 최적해다.
[가설] 89.5° 에서도 비드가 안 나가는 물리적 메커니즘(컵 안쪽 벽에 눌려 있는지, 미세한 위쪽 피치인지)은
영상 프레임으로 확인해야 한다 — `our_source/pour_videos/t2r_rh4_i00_ep3100.mp4`.

조준·방향·파지·자세는 이 트랙 역대 최고다(방향 오차 6°, 주둥이 수평 1.0 cm). **붓기만 하지 않는다.**

## iter_01 변경 (이 2 개만, 나머지는 iter_00 그대로)

1. **되받아 다시 붓는 왕복이 이득이 되지 않게 한다.**
   - 측정으로 확인된 것: env 의 `d_in_target` 은 **건너간 비드의 단조 증분**이라 되받아도 음수 지급이 없고,
     같은 비드를 다시 부어도 배달 수입이 다시 붙지 않는다(ep1300 trace: 양 합 300 = 20 알 × 15, 음 합 0).
     반면 `bead_in_target_frac`·`bead_in_source_frac` 은 **현재 상태**라 되받으면 원래대로 돌아간다.
     따라서 "아직 안 건너간 비율"을 상태값으로는 만들 수 없다 — 무상태 함수에서 쓸 수 있는 단조 신호는 `d_in_target` 뿐이다.
   - 요구: 왕복(붓기 → 되받기 → 다시 붓기)의 스텝당 평균 수입이 **같은 길이 동안 부어 놓고 유지**하는 수입보다 낮아야 한다.
     `pour_open` 의 가중·형태를 낮추거나, 소스 잔량 대신 배달 증분에 더 무게를 싣거나, 담김이 높을 때 기울임 수입이
     꺼지게 하는 등 형태는 생성기가 정한다. 창·방향·파지 게이트와 `rel`(70° 중심 단조)은 유지한다.
2. **부어 놓고 유지하는 쪽이 더 크게 한다.** `in_target_level`(지금 W 1.0, 에피소드 합 147)과
   `success_bonus`(지금 W 3.0, 합 237)의 가중을 올려, 담김이 높은 상태를 유지하는 수입이
   기울임 수입보다 커지게 한다. 성공 판정(담김 ≥ 0.80 · 흘림 ≤ 0.25 · 20 스텝)은 env 가 계산하며 바꾸지 않는다.

3. ★**"붓지 않고 기울인 자세만 유지"가 이득이 되지 않아야 한다**(ep3100 에서 확인된 새 최적해).
   기울임 수입은 **비드가 실제로 나가는 것과 연결**되어야 한다. 예를 들어 기울인 채 오래 머무르는 것보다
   방출이 일어나는 쪽이 커지도록, 또는 기울임 수입 자체를 낮춰 유지·배달 수입과의 격차를 벌린다.
   지금 후보(c1)의 스텝당 수입은 **붓지 않고 기울이기 9.52 · 붓는 중 23.46 · 부어 놓고 유지 14.09** 로
   순서는 맞지만 기울이기와 유지의 격차가 1.48 배뿐이다. 탐색이 붓기에 닿기 전에 9.5 짜리 고원에
   눌러앉을 위험이 있다 — 이 격차를 2 배 이상으로 벌리는 것이 이번 변경의 목표다.
   단 i17 에서 회전 신호를 약하게 했더니 붓기가 소멸한 전례가 있으므로, 회전 신호를 **없애지는 않는다**.

## 검사에 새로 넣을 것 (iter_00 검사가 놓친 지점)

iter_00 의 `check_iter00.py` 는 **붓는 순간**의 수입 비율만 쟀다. 다음 두 가지를 추가한다.

- **다 붓고 난 뒤 회전 수입이 꺼지는가**: 같은 자세·같은 기울기에서 담김 0.0 일 때와 0.9 일 때의
  스텝당 수입을 비교한다. 후자가 확실히 낮아야 한다.
- **되받기가 이득이 아닌가**: 담김 0.9 에서 비드를 소스로 되돌린 상태(담김 0.0 · 소스 1.0)로 만들었을 때
  회전 수입이 복구되면 FAIL 이다.
- **부어 놓고 유지**: 담김 0.9 를 유지하는 스텝당 수입이, 같은 조건에서 기울여 회전 수입을 받는 것보다 커야 한다.

## 유지하는 것

- 세 가지 사용자 지시(회전·배달 우선 예산 / 각도 부분 허용 / 관절 여유 자세)는 그대로다.
  iter_00 에서 각도 수입과 `posture_margin` 은 의도대로 동작했다(방향 오차 12°, 손목 한계에 붙지 않음).
- 접근 차분 + 컵 반지름 창 + 방향 비용 + 흘림·안전·제어 벌점, 파지 단계 ckpt 웜스타트.
- 항 수는 25 개 안팎을 유지한다. 새 항을 만들지 말고 위 2 개의 형태·가중만 바꾼다.


## 2026-09-27 오전 추가 — 사용자 영상 판정과 측정 (이 절이 위 내용보다 우선한다)

### 사용자 판정
- ep1300: "조인트가 과도하게 꺾여 있다는 문제는 있지만 제대로 틸팅되는 게 보인다."
- ep1300·ep3100 공통: **과도한 관절 포화, 그리고 두 팔이 서로 겹치는 이상한 자세.** ep3100 이 가장 나쁘다.
- "제대로 컵을 잡고 틸팅 각도는 만들 수 있다. 지금 보상이 그걸 못 알려 주는 것이다."

### 측정 1 — 관절 포화 (후반 2/3, 관절 한계 5 % 안에 머무는 스텝 비율)

| | j2 | j5 | j6 | j7 | 한 관절이라도 |
|---|---|---|---|---|---|
| ep1300 소스 | 0.07 | 0.56 | 0.00 | 0.60 | 0.66 |
| ep1300 리시버 | 0.51 | 0.00 | 0.00 | 0.00 | 0.51 |
| ep3100 소스 | 0.88 | 0.85 | 0.00 | **0.99** | **0.99** |
| ep3100 리시버 | 0.48 | 0.00 | 0.63 | 0.00 | 0.82 |

두 손바닥 거리 중앙: ep1300 0.159 m(최소 0.086) · ep3100 **0.121 m**(최소 0.087). 두 팔이 중심선에 몰려 있다.

### 측정 2 — 두 팔이 서로 통과하고 있었다 (env 설정)
- i00 런 `env.yaml`: `enable_self_collisions: false`. 양팔은 한 articulation 이라 이 값이 false 면
  **왼팔과 오른팔 링크 사이에 충돌이 생기지 않는다.** 영상의 "겹치는 팔"은 실제 관통이다.
- 접촉력이 생기지 않으니 `foreign`(손이 남의 것을 건드림) 벌점도 이것을 보지 못했다.
- 꺼 둔 이유(09.14, 시작 자세 손가락 자기접촉 226 N)는 09.15 자산 재빌드로 해결됐다. 학습이 쓴 USD 는
  재빌드본(09-15 23:16)이고 충돌 필터 46 쌍은 모두 한 손 안의 손가락 겹침, **좌-우 팔 사이 필터는 0 개**다.
- ★iter_01 부터 `env.enable_self_collisions=true` 로 학습한다(사용자 승인 09-27). 팔끼리 관통하는 자세는 물리적으로 막힌다.

### 측정 3 — 비드가 실제로 나오는 각은 70° 가 아니라 약 100° 다
두 런의 trace 에서 비드가 소스 컵을 떠난 순간의 기울기:

| | 사건 수 | p10 | p50 | p90 |
|---|---|---|---|---|
| ep1300 | 1657 | 94° | **101°** | 108° |
| ep3100 | 277 | 82° | 86° | 90° |

기울기 구간별 소스 잔량(ep1300): 60~70° 1.000 · 70~80° 1.000 · 80~90° 1.000 · **90~100° 1.000** · 100~110° 0.175 · 110~130° 0.000.
컵 내부 100.8 mm · 안지름 56.2 mm · 비드 20 개 더미 높이 29.9 mm(채움 0.30).

- 환경 설명의 "가득 찬 컵에서 첫 알이 약 70°" 는 **이 트랙의 컵에는 맞지 않는다**(비드는 가득 차 있지 않다).
- iter_00 의 `rel = 0.5·(1 + tanh((θ − 70°)/12°))` 는 86° 에서 이미 만점이다. ep3100 정책은 기울기 중앙 82.8° 로
  **11.6 초 연속** 버텼고 그동안 소스 잔량은 1.000 이었다. **보상이 비드가 안 나오는 각도에 만점을 주었다.**
  사용자 표현대로 "보상이 제대로 된 틸팅을 알려 주지 못했다."

## iter_01 변경 (최종 — 위의 변경 1·2·3 을 대체한다)

1. **회전 수입을 비드가 실제로 나오는 조건에 맞춘다.**
   70° 같은 각도 상수를 쓰지 않는다(100° 도 이 컵 고유 값이라 박지 않는다). 대신 ctx 의 컵 기하
   (`cup_mouth_z`, `cup_bottom_z`, `cup_radius`, `src_cup_pos`, `src_cup_up`, `src_cup_mouth_pos`)와
   현재 잔량(`bead_in_source_frac`)으로 **림의 가장 낮은 점이 소스 안 비드 더미 높이보다 아래로 내려갔는가**를 계산해,
   그 조건에 가까워질수록(그리고 넘을수록) 회전 수입이 오르게 한다. 더미 높이를 잔량으로 추정할 때 쓰는
   "처음 채움 비율 0.30" 은 이 트랙의 측정값으로 명시해도 된다(주석에 출처를 적을 것).
   창·방향·파지 게이트는 유지한다.
2. **되받기·붓지 않고 버티기를 막는다**(c2 에서 검사로 확인된 구조를 유지).
   회전 수입은 컵에서 나간 비율(담김 + 흘림, 흘림은 영구)이 오르면 꺼지고, 담김 유지(`in_target_level`)와
   성공(`success_bonus`)이 기울임 수입보다 커야 한다.
3. **관절 여유를 지키며 붓는 자세를 가르친다.** 지금 `posture_margin` 은 같은 자세에서 0.19 차이뿐이라
   신호가 거의 없다. i15 처럼 한계에 직접 강한 벌점을 걸지는 말고(붓기가 먼저 죽었다), **붓는 수입을 받을 때
   관절 여유가 클수록 더 받는** 형태로 차이를 키운다. 사용자 판단: 제대로 잡으면 한계에 기대지 않고도 틸팅 각을 만들 수 있다.
4. self-collision 이 켜지면 두 팔이 부딪칠 때 접촉력이 생긴다. `ctx` 에 팔-팔 접촉 필드는 없다 —
   `src_hand_foreign_force`·`rcv_hand_foreign_force`(손이 자기 컵 밖의 것을 누르는 힘)만 있다. 새 필드를 가정하지 말 것.

## 검사 (check_iter01.py 에 이미 있는 [6][7][8] + 새 항목)
- [9] 비드가 나오지 않는 기울기(림 최저점이 더미 위)에서는 회전 수입이 만점에 이르지 않아야 한다.
  같은 방향·창에서 기울기 85° 와 105° 를 비교해 105° 쪽이 확실히 커야 한다(잔량 1.0 기준).

### 측정 4 — self-collision ON 프로브 (파지 단계 ckpt, 64 env · 600 스텝, server GPU1)

| 지표 | OFF | ON |
|---|---|---|
| 소스 파지 유지(후반 2/3) | 0.988 | 0.971 |
| 리시버 파지 유지 | 0.990 | 0.960 |
| 리셋 직후 손가락 힘 최대(자기접촉 상주 여부) | 0.000 N | 0.000 N |
| 손 foreign 힘 p95 | 0 N | 0 N |
| 두 손바닥 거리 중앙 | 0.373 m | 0.462 m |

켜도 손가락 자기접촉이 생기지 않고 파지는 2~3 % 만 줄었다. 두 팔은 더 떨어져 움직인다.
