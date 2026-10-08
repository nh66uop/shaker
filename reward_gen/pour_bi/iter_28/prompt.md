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
5. ★KEEP IT GENERAL. The reward must stay correct when the cup, the beads or the table change. Never hardcode an absolute angle, distance or size that belongs to the current assets — no "the cup releases at 72°", no fixed pour angle, no fixed cup radius. Measured numbers in the knowledge section are there to tell you what the task looks like, NOT to be copied into the code as targets. When you need such a quantity, take it from `ctx` (it is computed by the environment from the current assets) or, better, reward the OUTCOME that the quantity was a proxy for — beads actually leaving the source cup, beads arriving in the receiver, the requested amount being reached — and let the policy discover the angle by itself. Thresholds that describe the ROBOT or the reward's own shape (contact force floor, smoothing scales, ramp widths) are fine.
6. ★KEEP IT SIMPLE. Few terms, each with one clear job, beat many overlapping ones: a simpler reward generalises better and is far easier to diagnose from the training curves. Aim for about 15 components and do not exceed 25. Do not add a term that only repeats what another term already pays for, and do not add a penalty for a failure mode that has not been observed.

The previous reward function was:
```python
import torch
import math


def _step(x: torch.Tensor) -> torch.Tensor:
    """smoothstep on an already normalised argument -> [0,1]"""
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


def _grasp_quality(finger_force: torch.Tensor, palm_force: torch.Tensor,
                   wrap_count: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """[0,1] quality of one hand on its own cup BEFORE the lift reference exists. Only the SHAPE of
    the contact enters (wrapped fingers, palm contact, thumb contact), no cup dimension.
    iter_27 change 3: wrapped fingers are counted out of ALL fingers of the hand (F from the tensor
    shape) instead of saturating at 3, so the lift reference records a five-finger envelope."""
    n_fingers = float(max(int(finger_force.shape[1]), 1))
    wrap_q = torch.clamp(wrap_count.to(dt) / n_fingers, 0.0, 1.0)
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)      # [N] force sensor scale
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    return 0.5 * wrap_q + 0.25 * palm_q + 0.25 * thumb_q


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """iter_24, KEPT UNCHANGED: how closely the hand still holds the cup the way it did at the
    instant it lifted it (hand pose in CUP coordinates, force distribution and force level RELATIVE
    to that instant). Never an absolute force target. Before the reference exists the factor is 1."""
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = torch.clamp(0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim, 0.0, 1.0)
    return torch.where(valid, keep, torch.ones_like(keep))


def _fingers_kept(ref_share, ref_total, finger_force, valid, on_share, keep_frac):
    """iter_27 change 3 (BOTH hands): share of the fingers that touched the cup AT LIFT that still
    press with at least keep_frac of their OWN lift force (ref share * ref total). Purely relative to
    the lift hold: no absolute force floor, no force target. 1 before the reference exists or if no
    finger touched at lift. share = [palm, thumb..pinky], so the fingers are share[:, 1:]."""
    ref_f = torch.clamp(ref_share[:, 1:], min=0.0) * ref_total[:, None]     # [N] each finger at lift
    ref_on = (ref_share[:, 1:] > on_share).to(finger_force.dtype)
    need = torch.clamp(keep_frac * ref_f, min=1e-6)                         # numeric guard only
    touch = _step(torch.clamp(finger_force, min=0.0) / need)
    n_ref = torch.sum(ref_on, dim=-1)
    kept = torch.sum(ref_on * touch, dim=-1) / torch.clamp(n_ref, min=1.0)
    kept = torch.where(n_ref > 0.5, kept, torch.ones_like(kept))
    return torch.where(valid, kept, torch.ones_like(kept))


def _spout(lip, rcv_mouth, cup_r, geo):
    """iter_26 spout geometry, unchanged: (horizontal lip offset, fine, height).
    geo = (precision width, below-rim width, drop width, drop floor); widths in cup radii."""
    prec_r, below_r, drop_r, drop_floor = geo
    delta = lip - rcv_mouth
    lip_xy = torch.norm(delta[:, :2], dim=-1)
    lip_dz = delta[:, 2]
    fine = torch.exp(-(lip_xy / (prec_r * cup_r)) ** 2)                 # spout centred in the mouth
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (below_r * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (drop_r * cup_r)) ** 2)
    height = over_rim * (drop_floor + (1.0 - drop_floor) * low_drop)
    return lip_xy, fine, height


def _turned(vec, ang_vel, tau):
    """vec carried by a body that turns at ang_vel, as it was tau seconds EARLIER (Rodrigues)."""
    wn = torch.norm(ang_vel, dim=-1, keepdim=True)
    k = ang_vel / torch.clamp(wn, min=1e-9)
    phi = -wn * tau
    return vec * torch.cos(phi) + torch.cross(k, vec, dim=-1) * torch.sin(phi) \
        + k * torch.sum(k * vec, dim=-1, keepdim=True) * (1.0 - torch.cos(phi))


def _lean_spout(src_pos, src_up, src_mouth, rcv_mouth, lip_r, cup_r, geo):
    """G = spout * lean of ONE state, rebuilt with the ctx formulas: d = horizontal unit vector
    source cup origin -> receiver mouth, theta = signed tilt toward d in the (d, z) plane,
    lip = source mouth + r (cos theta d - sin theta z), lean = max(up . d, 0)."""
    h = rcv_mouth - src_pos
    hn = torch.clamp(torch.sqrt(h[:, 0] ** 2 + h[:, 1] ** 2), min=1e-6)
    dx, dy = h[:, 0] / hn, h[:, 1] / hn
    toward = src_up[:, 0] * dx + src_up[:, 1] * dy                      # axis component toward d
    theta = torch.atan2(toward, src_up[:, 2])
    c, s = torch.cos(theta), torch.sin(theta)
    lip = src_mouth + lip_r[:, None] * torch.stack([c * dx, c * dy, -s], dim=-1)
    _, fine, height = _spout(lip, rcv_mouth, cup_r, geo)
    return fine * height * torch.clamp(toward, min=0.0)


def _lean_spout_rate(ctx, lip_r, cup_r, geo, tau, dt):
    """(G(both cups moved ahead by tau) - G(both cups moved back by tau)) / (2 tau) [1/s], moved
    along their lin/ang velocities. The receiver mouth, d, theta and the lip are all rebuilt at both
    states, so this is the FULL rate of G. c1r: central, not one-sided - the one-sided error
    -(tau/2) v'Hv is positive at the concave spout peak and paid a jiggle there (+0.07/step at
    gates 1, check_tilt_rate.py A10). (hypothesis, as in iter_26: velocities are in the env frame
    of the positions; ctx lin_vel is the COM velocity, taken as the cup origin's)"""
    sp, su = ctx.src_cup_pos.to(dt), ctx.src_cup_up.to(dt)
    sm, rp, rm = ctx.src_cup_mouth_pos.to(dt), ctx.rcv_cup_pos.to(dt), ctx.rcv_cup_mouth_pos.to(dt)
    vs, ws = ctx.src_cup_lin_vel.to(dt), ctx.src_cup_ang_vel.to(dt)
    vr, wr = ctx.rcv_cup_lin_vel.to(dt), ctx.rcv_cup_ang_vel.to(dt)
    sp_f, sp_b = sp + tau * vs, sp - tau * vs
    g_fwd = _lean_spout(sp_f, _turned(su, ws, -tau), sp_f + _turned(sm - sp, ws, -tau),
                        rp + tau * vr + _turned(rm - rp, wr, -tau), lip_r, cup_r, geo)
    g_back = _lean_spout(sp_b, _turned(su, ws, tau), sp_b + _turned(sm - sp, ws, tau),
                         rp - tau * vr + _turned(rm - rp, wr, tau), lip_r, cup_r, geo)
    return (g_fwd - g_back) / (2.0 * tau)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dt = ctx.src_palm_pos.dtype

    # =============================== weights (iter_26 unless marked iter_27) =================
    APPROACH_W, APPROACH_K = 0.6, 4.0      # reach income, small: it must not compete with pouring
    GRASP_W, CLOSE_W, CLOSE_NORM = 0.8, 0.8, 0.35
    LIFT_SRC_W, LIFT_RCV_W = 3.0, 1.5      # iter_24 levels: these lifted reliably, do not weaken
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03   # carry heights: workspace, not cup sizes
    POSTURE_W_SRC, POSTURE_W_RCV = 3.0, 1.2    # before the lift: build the envelope (now all fingers)
    KEEP_W_SRC, KEEP_W_RCV = 1.5, 1.5          # after the lift: keep THAT grasp (iter_24 design)
    KEEP_POS, KEEP_MAG = 0.02, 0.7             # [m] drift and |log force ratio| at which sim = 0.37
    FINGER_ON_SHARE = 0.02     # "touched at lift" = force share of the lift hold above this
    # iter_27 change 3: a lift finger still counts if it presses with >= half of its OWN lift force
    # (replaces the absolute 0.2 N floor). Losing 1 of 5 costs KEEP_W * keep * 0.2 (~0.19 per step).
    KEEP_FORCE_FRAC = 0.5
    AIM_W, AIM_K, AIM_FAR = 5.0, 10.0, 0.40    # workspace-scale guidance toward the receiver
    AIM_PREC_R = 0.5       # gaussian width of the spout offset inside the receiver mouth [cup radii]
    DROP_R = 1.5           # spout height above the receiver rim at which low_drop = 0.37
    BELOW_R = 1.0          # spout below the receiver rim
    DROP_FLOOR = 0.4       # a high spout keeps 40 % of the aim income (collision margin)
    # iter_27 change 1: rotation help = TILT_W * rate of Phi = latches * spout * lean [1/s], summed
    # per step. Pay gated by rate-free factors and capped at +TILT_PAY_CAP per step, charge in full.
    TILT_W, TILT_PAY_CAP = 5.0, 1.0
    # c1r: half-width of the central difference [s], a numerical step only (policy step is 1/60 s:
    # sim.dt 1/120 x decimation 2). A sampled rate does not telescope; closed cycles up to 5 Hz sum
    # to <= 0 in check_tilt_rate.py, aliasing remains only at the 30 Hz reward Nyquist (A7n).
    TILT_TAU = 1.0 / 120.0
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 60.0, 0.03, -0.05   # paid per unit of bead fraction that ARRIVES
    OVER_DELTA_W = 120.0       # 2x FLOW_W: a bead past the request costs more than it ever paid
    OVER_LEVEL_W = 4.0         # sustained charge for sitting on an overshoot
    SPILL_W = 50.0             # per REQUEST-worth of spilled beads, floored at OVER_DELTA_W
    FINISH_W = 10.0            # sustained: request closed + cup upright at the pre-tilt pose + held
    FIN_UP_FLOOR, FIN_POSE_FLOOR = 0.5, 0.4
    POSE_TOL_R, POSE_SIG_R = 2.0, 2.0   # plateau beyond ctx.cup_mouth_z / gaussian falloff [cup radii]
    SUCCESS_W = 12.0           # env verdict on top of it
    HOME_W = 1.0
    CENTER_W, CENTER_BAND, CENTER_SCALE = 1.5, 0.05, 0.10   # source side centre-line term
    SIDE_VOID, SIDE_OK = 0.03, 0.08     # env voids success below 3 cm receiver side margin
    DIR_ZERO = 0.25                     # env voids success at pour_dir_xy[:,0] > 0.3
    PRE_W, PRE_LO, LATCH_W = 2.0, 0.7, 1.0     # premature tilt: FRACTION of ctx.premature_tilt_limit
    CONTACT_W, HIT_W, HIT_LATCH_W = 2.0, 5.0, 0.3
    FOREIGN_DEAD, FOREIGN_W_FREE, FOREIGN_W_HELD = 1.0, 0.3, 1.5
    RATE_W_BASE, RATE_W_FULL, FLIP_W = 0.005, 0.03, 0.4
    RCV_CONTROL_MULT = 2.0     # receiver arm trembled more -> charge it harder
    TREMOR_SRC_W, TREMOR_RCV_W = 0.3, 0.9      # on the measured arm joint speeds
    TREMOR_FREE, TREMOR_SCALE = 0.20, 0.50     # [rad/s]
    ADR_RAMP_FULL = 0.2        # control charges reach full weight at adr_progress 0.2
    SETTLE_FRAC = 0.05

    # =============================== basic state (unchanged) =================================
    g_src = ctx.src_grasped.to(dt)
    g_rcv = ctx.rcv_grasped.to(dt)
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    off_src = _step((h_src - 0.01) / 0.04)          # lift-off ramp (reward shape, not an asset size)
    off_rcv = _step((h_rcv - 0.01) / 0.04)
    held_src = torch.maximum(g_src, off_src)        # a flickering contact flag must not kill income
    held_rcv = torch.maximum(g_rcv, off_rcv)
    carry_src = held_src * off_src
    carry_rcv = held_rcv * off_rcv
    carry_both = carry_src * carry_rcv
    not_nested = (~ctx.cups_nested).to(dt)
    settled = _step(ctx.episode_progress.to(dt) / SETTLE_FRAC)
    up_src = torch.clamp(ctx.src_cup_up[:, 2], 0.0, 1.0)             # 1 = upright, 0 = horizontal
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], 0.0, 1.0)
    cup_r = max(float(ctx.cup_radius), 1e-3)                         # geometry unit from ctx
    geo = (AIM_PREC_R, BELOW_R, DROP_R, DROP_FLOOR)

    # grasp quality / grasp maintenance
    q_src = _grasp_quality(ctx.src_finger_force, ctx.src_palm_force, ctx.src_wrap_count, dt)
    q_rcv = _grasp_quality(ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_wrap_count, dt)
    keep_src = _grip_keep(ctx.src_palm_in_cup, ctx.src_tips_in_cup, ctx.src_ref_palm_in_cup,
                          ctx.src_ref_tips_in_cup, ctx.src_force_share, ctx.src_ref_force_share,
                          ctx.src_force_total, ctx.src_ref_force_total, ctx.src_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    keep_rcv = _grip_keep(ctx.rcv_palm_in_cup, ctx.rcv_tips_in_cup, ctx.rcv_ref_palm_in_cup,
                          ctx.rcv_ref_tips_in_cup, ctx.rcv_force_share, ctx.rcv_ref_force_share,
                          ctx.rcv_force_total, ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    # iter_27 change 3: fingers-kept on BOTH hands, relative to each finger's own lift force
    kept_src = _fingers_kept(ctx.src_ref_force_share.to(dt), ctx.src_ref_force_total.to(dt),
                             ctx.src_finger_force.to(dt), ctx.src_grasp_ref_valid,
                             FINGER_ON_SHARE, KEEP_FORCE_FRAC)
    kept_rcv = _fingers_kept(ctx.rcv_ref_force_share.to(dt), ctx.rcv_ref_force_total.to(dt),
                             ctx.rcv_finger_force.to(dt), ctx.rcv_grasp_ref_valid,
                             FINGER_ON_SHARE, KEEP_FORCE_FRAC)
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    grip_gate = 0.3 + 0.7 * keep_src        # a ruined grasp earns a fraction of the pour income

    # =============================== requested amount (unchanged) ============================
    poured = torch.clamp(ctx.bead_in_target_frac.to(dt), 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac.to(dt), 0.0, 1.0)
    in_src = torch.clamp(ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    tol = 1.0 / torch.clamp(ctx.bead_active_count.to(dt), min=1.0)   # env tolerance = ONE bead
    d_in = ctx.d_in_target.to(dt)
    poured_prev = torch.clamp(poured - d_in, min=0.0)
    deficit_prev = torch.clamp(target - poured_prev, min=0.0)        # what was still requested
    err = torch.abs(poured - target)
    on_target = 1.0 - _step((err - tol) / (4.0 * tol))                # 1 within one bead, 0 at five
    need = torch.clamp(torch.minimum(target - poured, in_src), min=0.0)   # spilled beads are gone
    want = _step((need - tol) / tol)          # more than one bead still requested AND available
    closed = 1.0 - want
    spilled = torch.clamp(ctx.bead_spill_frac.to(dt), 0.0, 1.0)
    released = _step((poured + spilled) / tol)   # the first bead has LEFT the source (cumulative)

    # =============================== stage 1-3: reach, grasp, lift ===========================
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = APPROACH_W * torch.exp(-APPROACH_K * d_src)
    approach_rcv = APPROACH_W * torch.exp(-APPROACH_K * d_rcv)
    grasp_src = GRASP_W * held_src + CLOSE_W * torch.exp(-6.0 * d_src) \
        * torch.clamp(ctx.src_hand_closure / CLOSE_NORM, 0.0, 1.0)
    grasp_rcv = GRASP_W * held_rcv + CLOSE_W * torch.exp(-6.0 * d_rcv) \
        * torch.clamp(ctx.rcv_hand_closure / CLOSE_NORM, 0.0, 1.0)
    lift_shape_src = 0.5 * (1.0 - torch.exp(-torch.clamp(h_src, min=0.0) / LIFTOFF_H)) \
        + 0.5 * torch.clamp(h_src / LIFT_SRC_H, 0.0, 1.0)
    lift_src = LIFT_SRC_W * held_src * lift_shape_src
    lift_rcv = LIFT_RCV_W * held_rcv * torch.clamp(h_rcv / LIFT_RCV_H, 0.0, 1.0) * up_rcv
    # posture: build the envelope before the lift (change 3: out of all F fingers), KEEP the lift
    # grasp afterwards (iter_24) times the share of lift fingers still pressing (change 3: both hands)
    posture_src = held_src * ((1.0 - v_src_ref) * POSTURE_W_SRC * q_src
                              + v_src_ref * KEEP_W_SRC * keep_src * kept_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * POSTURE_W_RCV * q_rcv
                              + v_rcv_ref * KEEP_W_RCV * keep_rcv * kept_rcv)

    # =============================== stage 4: aim the LIP over the receiver (unchanged) ======
    lip_xy, fine, height = _spout(ctx.src_pour_lip_pos.to(dt), ctx.rcv_cup_mouth_pos.to(dt),
                                  cup_r, geo)
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)     # x > 0.3 voids success
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    aim_xy = 0.2 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) \
        + 0.3 * torch.exp(-AIM_K * lip_xy) + 0.5 * fine
    aim_geo = aim_xy * height
    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_geo * want                            # fades once the request closes

    # =============================== stage 4b: rotation potential (iter_27 change 1) =========
    # Phi = latches * spout * lean. lean = max(up . d, 0) is how far the cup axis leans TOWARD the
    # receiver (no angle constant); spout = fine * height as in the aim. latches = settled *
    # (1 - released) * want only move one way before the first bead (settled rises while the cups
    # stand apart at spawn where spout ~ 0, released is the intended switch-off, want is fixed until
    # a bead leaves), so they scale the rate of spout * lean. The rate is the FULL rate: both cups
    # are moved ahead and back by TILT_TAU and the mouth, d, theta and lip are rebuilt. Holding
    # any pose earns 0; a tilt made while spout = 0 pays 0 and costs 0; a lean away or sideways
    # has lean = 0.
    # The pay gate (aim_gate: carry, not_nested, dir_ok, side_ok, up_rcv, grip_gate) has no rate and
    # is NOT inside Phi: it can only shrink the pay, never the charge.
    lip_r = torch.norm(ctx.src_pour_lip_pos - ctx.src_cup_mouth_pos, dim=-1).to(dt)   # r from ctx
    latches = settled * (1.0 - released) * want
    rate = latches * _lean_spout_rate(ctx, lip_r, cup_r, geo, TILT_TAU, dt)
    tilt_rate = torch.where(rate > 0.0,
                            torch.clamp(TILT_W * aim_gate * rate, max=TILT_PAY_CAP),
                            TILT_W * rate)

    # =============================== stage 5: the flow IS the pour (unchanged) ===============
    useful = torch.clamp(torch.minimum(d_in, deficit_prev), FLOW_FLOOR, FLOW_CAP)
    pour_flow = FLOW_W * useful
    overshoot = -OVER_DELTA_W * torch.clamp(d_in - deficit_prev, min=0.0) \
        - OVER_LEVEL_W * torch.clamp(poured - target, min=0.0)
    spill_w = torch.clamp(SPILL_W / torch.maximum(target, tol), min=OVER_DELTA_W)
    spill = -spill_w * settled * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # =============================== stage 6: finish upright at the pre-tilt pose (unchanged) =
    body_xy = torch.norm(ctx.src_cup_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    pose_tol = max(float(ctx.cup_mouth_z), 0.0) + POSE_TOL_R * cup_r
    pre_tilt_pose = not_nested * torch.exp(
        -(torch.clamp(body_xy - pose_tol, min=0.0) / (POSE_SIG_R * cup_r)) ** 2)
    finish = FINISH_W * closed * (0.3 + 0.7 * on_target) \
        * (FIN_UP_FLOOR + (1.0 - FIN_UP_FLOOR) * up_src) \
        * (FIN_POSE_FLOOR + (1.0 - FIN_POSE_FLOOR) * pre_tilt_pose) \
        * carry_both * up_rcv * (0.4 + 0.6 * keep_src)
    success = SUCCESS_W * ctx.success.to(dt)

    # =============================== workspace constraints ===================================
    center_src = -CENTER_W * held_src * torch.tanh(
        torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)
    # iter_27 change 2: home anchor = receiver spawn x at side margin SIDE_OK (was: its spawn).
    # rcv_side_margin moves 1:1 with the receiver's lateral position, so the lateral offset from the
    # anchor is (margin - SIDE_OK). Form, HOME_W, carry_rcv, up_rcv, side_ok, still_rcv unchanged.
    dx_home = ctx.rcv_cup_pos[:, 0] - ctx.rcv_cup_spawn_pos[:, 0]
    dm_home = ctx.rcv_side_margin.to(dt) - SIDE_OK
    d_home = torch.sqrt(dx_home ** 2 + dm_home ** 2 + 1e-12)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05,
                                             min=0.0) / 0.15)
    home_rcv = HOME_W * carry_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv
    limit = torch.clamp(ctx.premature_tilt_limit.to(dt), min=1e-3)
    risk = _step((ctx.src_cup_tilt.to(dt) / limit - PRE_LO) / (1.0 - PRE_LO))
    premature = -PRE_W * held_src * (1.0 - torch.exp(-AIM_K * lip_xy)) * risk \
        - LATCH_W * ctx.premature_tilt.to(dt)

    # =============================== safety (real robot, unchanged) ==========================
    f_cup = ctx.cup_cup_force
    cup_collision = -CONTACT_W * torch.tanh(f_cup / 2.0) \
        - HIT_W * settled * (f_cup > 0.5).to(dt) - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_s = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    f_r = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    hand_foreign = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) \
        * torch.tanh(f_s / 5.0) \
        - (FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) * torch.tanh(f_r / 5.0)

    # =============================== control (ramped on adr_progress, unchanged) =============
    ramp = _step(torch.full_like(up_src, float(ctx.adr_progress) / ADR_RAMP_FULL))
    rate_w = RATE_W_BASE + (RATE_W_FULL - RATE_W_BASE) * ramp
    da = ctx.actions - ctx.prev_actions
    flips = torch.relu(-(ctx.actions * ctx.prev_actions))
    action_rate = -rate_w * (torch.sum(da[:, 0:9] ** 2, dim=-1)
                             + RCV_CONTROL_MULT * torch.sum(da[:, 9:18] ** 2, dim=-1)) \
        - FLIP_W * ramp * (torch.sum(flips[:, 0:9], dim=-1)
                           + RCV_CONTROL_MULT * torch.sum(flips[:, 9:18], dim=-1)) \
        * torch.maximum(held_src, held_rcv)
    qd_s = torch.sqrt(torch.mean(ctx.src_arm_qd ** 2, dim=-1) + 1e-8)
    qd_r = torch.sqrt(torch.mean(ctx.rcv_arm_qd ** 2, dim=-1) + 1e-8)
    tremor = -ramp * (TREMOR_SRC_W * torch.tanh(torch.clamp(qd_s - TREMOR_FREE, min=0.0) / TREMOR_SCALE)
                      + TREMOR_RCV_W * torch.tanh(torch.clamp(qd_r - TREMOR_FREE, min=0.0) / TREMOR_SCALE))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "posture_src": posture_src,
        "posture_rcv": posture_rcv,
        "aim": aim,
        "tilt_rate": tilt_rate,
        "pour_flow": pour_flow,
        "overshoot": overshoot,
        "spill": spill,
        "finish": finish,
        "success": success,
        "center_src": center_src,
        "home_rcv": home_rcv,
        "premature": premature,
        "cup_collision": cup_collision,
        "hand_foreign": hand_foreign,
        "action_rate": action_rate,
        "tremor": tremor,
    }

    reward = torch.zeros_like(approach_src)
    for v in components.values():
        reward = reward + v

    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.21e-07 · max 5.27e-05
bead/spill: [0, 0.00018, 0, 0, 0.000114, 0, 0, 0.000217, 0.00103, 0.000312]  min 0 · mean 0.0002 · max 0.00118
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.25e-05 · max 0.000488
episode_lengths/step: [77.4, 836, 864, 870, 887, 881, 884, 882, 857, 879]  min 77.4 · mean 852 · max 899
reward/action_rate: [-0.0297, -0.0585, -0.056, -0.031, -0.0446, -0.0401, -0.0312, -0.034, -0.0327, -0.0282]  min -0.0867 · mean -0.0405 · max -0.0262
reward/aim: [0, 0.286, 0.251, 0.0825, 0.524, 0.519, 0.246, 0.578, 0.622, 0.46]  min 0 · mean 0.464 · max 0.952
reward/approach_rcv: [0.346, 0.399, 0.408, 0.37, 0.402, 0.4, 0.378, 0.393, 0.392, 0.378]  min 0.319 · mean 0.392 · max 0.419
reward/approach_src: [0.316, 0.416, 0.42, 0.329, 0.414, 0.412, 0.357, 0.38, 0.386, 0.381]  min 0.316 · mean 0.397 · max 0.425
reward/center_src: [0, -0.000161, -0.00105, -0.00114, -0.000486, -0.000702, -0.000243, -0.00133, -0.00236, -0.00361]  min -0.01 · mean -0.0013 · max 0
reward/cup_collision: [0, -0.00178, -0.00294, -0.000732, -0.00306, -0.00278, -0.00242, -0.0081, -0.00719, -0.0118]  min -0.0244 · mean -0.00555 · max 0
reward/finish: [0, 0, 0, 0, 0, 0, 0, 0.000456, 0.00124, 0]  min 0 · mean 3.15e-05 · max 0.00129
reward/grasp_rcv: [0.0815, 1.21, 1.23, 0.363, 1.2, 1.19, 0.579, 1.14, 1.13, 0.817]  min 0.0733 · mean 1.08 · max 1.26
reward/grasp_src: [0.0704, 1.25, 1.25, 0.343, 1.23, 1.21, 0.552, 1.15, 1.14, 0.788]  min 0.065 · mean 1.1 · max 1.28
reward/hand_foreign: [-0.0389, -0.0206, -0.0171, -0.0248, -0.0325, -0.0361, -0.0372, -0.0422, -0.0273, -0.0559]  min -0.26 · mean -0.036 · max -0.00126
reward/home_rcv: [0, 0.452, 0.54, 0.115, 0.535, 0.527, 0.222, 0.514, 0.612, 0.372]  min 0 · mean 0.484 · max 0.637
reward/lift_rcv: [0, 1.25, 1.39, 0.306, 1.37, 1.35, 0.543, 1.35, 1.29, 0.805]  min 0 · mean 1.19 · max 1.46
reward/lift_src: [0.000999, 2.67, 2.89, 0.608, 2.78, 2.77, 1.05, 2.68, 2.66, 1.65]  min 0.000999 · mean 2.37 · max 2.92
reward/overshoot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/posture_rcv: [0, 0.743, 0.751, 0.168, 0.694, 0.576, 0.259, 0.628, 0.636, 0.47]  min 0 · mean 0.643 · max 1.03
reward/posture_src: [0, 0.955, 0.878, 0.211, 0.963, 0.837, 0.398, 1.79, 2.23, 1.44]  min 0 · mean 1.26 · max 2.44
reward/pour_flow: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/premature: [0, -5.63e-05, -0.00624, -8.24e-05, -1.09e-05, -0.000397, -0.00152, -0.0121, -0.00755, -0.00176]  min -0.0305 · mean -0.00312 · max 0
reward/spill: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00651 · mean -0.000116 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_rate: [0, -3.32e-11, -4.52e-06, 2.08e-06, 0.000562, -0.0125, -0.000177, -0.0014, -6.38e-05, -0.000302]  min -0.0179 · mean -0.000642 · max 0.00384
reward/total: [0.745, 9.55, 9.91, 2.82, 10, 9.69, 4.49, 10.5, 11, 7.42]  min 0.737 · mean 9.27 · max 11.5
reward/tremor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
rewards/step: [36, 6.99e+03, 7.42e+03, 7.61e+03, 7.92e+03, 7.82e+03, 7.81e+03, 8.15e+03, 8.69e+03, 9.17e+03]  min 36 · mean 7.83e+03 · max 9.65e+03
task/aim_dist: [0.321, 0.262, 0.24, 0.292, 0.17, 0.162, 0.249, 0.178, 0.201, 0.219]  min 0.144 · mean 0.213 · max 0.434
task/cup_collision_rate: [0, 0.000244, 0.000244, 0, 0.000244, 0, 0, 0.000244, 0.000732, 0.000732]  min 0 · mean 0.000375 · max 0.00244
task/cups_center_dist: [0.32, 0.257, 0.233, 0.291, 0.172, 0.173, 0.256, 0.186, 0.227, 0.241]  min 0.158 · mean 0.22 · max 0.43
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0, 0, 0, 0, 0, 0, 0.000244, 0.000244, 0]  min 0 · mean 0.00013 · max 0.00415
task/rcv_cup_lift: [0.000507, 0.092, 0.106, 0.0266, 0.104, 0.106, 0.0447, 0.105, 0.0952, 0.0598]  min 0.000507 · mean 0.0947 · max 0.205
task/rcv_grasped: [0, 0.963, 0.947, 0.213, 0.945, 0.892, 0.341, 0.723, 0.783, 0.589]  min 0 · mean 0.808 · max 0.974
task/rcv_hand_foreign_rate: [0.256, 0.0347, 0.019, 0.136, 0.05, 0.0547, 0.125, 0.0586, 0.0408, 0.106]  min 0.00195 · mean 0.053 · max 0.666
task/src_cup_lift: [0.00051, 0.162, 0.207, 0.0453, 0.186, 0.19, 0.0684, 0.176, 0.178, 0.111]  min 0.00051 · mean 0.157 · max 0.223
task/src_grasped: [0, 0.984, 0.973, 0.215, 0.969, 0.959, 0.387, 0.94, 0.929, 0.589]  min 0 · mean 0.872 · max 0.998
task/src_hand_foreign_rate: [0.00391, 0.0149, 0.0203, 0.0117, 0.0244, 0.0247, 0.019, 0.0381, 0.021, 0.0432]  min 0.00195 · mean 0.0347 · max 0.209
task/src_tilt_deg: [0.861, 29.3, 30.4, 11.5, 28.9, 24.1, 13.6, 40.7, 41.2, 27.9]  min 0.319 · mean 30 · max 43.2
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
ROUND 28 NOTES (09.26, diagnosis of t2r_i27 = iter_27 c1r trained WARM from t2r_i26r ep800). Facts only. The
three changes below are the only changes for this round; everything else in iter_27 stays as it is.

TASK DEFINITION (operator, 09.25, unchanged): grasp both cups with both hands and keep the envelope (all
five fingers on the cup, grip held), pour the REQUESTED amount (the target amount that is in the actor
observation), then bring the source cup back upright and return to the pose it had before tilting.

WHAT t2r_i27 DID (deterministic replay, ADR 0, 64 envs x 900 steps; 10 active beads p50, request 5.5 p50;
premature_tilt_limit 73.6 deg p50; r = ctx.cup_radius = 0.041 m)
| metric | ep300 | ep600 | ep800 |
|---|---|---|---|
| success / any bead poured / any bead spilled | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| both cups held at the end | 0.94 | 0.73 | 0.89 |
| episode peak TOTAL tilt p50 (p10-p90) | 42 (38-53) deg | 55 (45-65) deg | 56 (53-58) deg |
| peak total tilt / premature_tilt_limit p50; episodes with peak >= 0.7 of the limit | 0.57; 11% | 0.75; 57% | 0.76; 92% |
| episode peak tilt TOWARD the receiver p50 (p90) | 21 (28) deg | 34 (46) deg | 47 (52) deg |
| cup axis lean OUT of the pour plane (d, z) while carrying p50 | 33 deg | 35 deg | 29 deg |
| source tilt when the cup first passes 3 cm of lift p50 | 11.5 deg | 24 deg | 32 deg |
| source lift reference recorded (episodes) | 64 / 64 | 10 / 63 | 4 / 64 |
| spout to receiver mouth, horizontal: at the tilt peak / end / episode minimum (p50) | 0.112 / 0.086 / 0.069 m | 0.134 / 0.157 / 0.092 m | 0.105 / 0.099 / 0.071 m |
| source cup body to receiver mouth at the end p50 | 0.119 m (2.9 r) | 0.216 m (5.3 r) | 0.180 m (4.4 r) |
| source y / receiver y at the end p50 | +0.029 / +0.137 | -0.105 / +0.098 | -0.084 / +0.084 |
| source 5-finger contact while carrying (1 N) | 0.48 | 0.69 | 0.86 |
| receiver 5-finger contact while carrying (1 N) / 3 or more fingers | 0.26 / 0.93 | 0.05 / 0.74 | 0.13 / 0.87 |
| receiver fingers loaded at its lift reference (share > 0.02) mean / all five > 1 N at lift | 4.2 / 0.06 | 3.9 / 0.03 | 3.9 / 0.02 |
| receiver lift-reference total force p50 | 11.3 N | 9.1 N | 7.9 N |
| tilt_rate per episode, paid / charged p50 | 0.0005 / -0.0007 | 0.0000 / -0.0000 | 0.0013 / -0.0014 |
| premature per episode p50 (p10) | 0.0 (-0.4) | -0.5 (-34) | -3.7 (-22) |
Tensorboard (21-epoch windows at e300 / e500 / e600 / e700 / e810): diag/grip_src_ref_valid 0.86 / 0.86 /
0.41 / 0.04 / 0.04 (receiver 0.88-0.92 throughout); posture_src 0.84 / 0.75 / 1.42 / 2.06 / 2.10; aim 0.40 /
0.47 / 0.47 / 0.55 / 0.68; premature -0.001 / -0.001 / -0.007 / -0.006 / -0.004; tilt_rate -0.0001 to
-0.0004; cup_collision -0.004 / -0.005 / -0.009 / -0.008 / -0.014; episode peak tilt (mid fill) 42 / 46 /
51 / 54 / 53.5 deg; task/pour_lip_dist 0.13 / 0.15 / 0.16 / 0.15 / 0.16 m; pour_flow, finish, in_target 0
throughout; adr_progress 0. A dip every ~100 epochs (e590, e689, e801) recovers within 2 epochs.

WHY (verified by recomputing the iter_27 terms offline, float64, on the trace states; counterfactual poses
move the source cup and hand rigidly, forces held fixed)
1. The rotation stops at the premature ramp, not for lack of room. premature charges from PRE_LO = 0.7 of
   premature_tilt_limit on the TOTAL tilt, times (1 - exp(-AIM_K * spout offset)). At ep800, 92% of the
   episode peaks sit at 0.72-0.79 of the limit. premature per step by total tilt (ep800): 50-55 deg -0.01,
   55-60 deg -0.15, 60-70 deg -0.59 (ep600: -0.02 / -0.23 / -0.78, 70-90 deg -1.74). Arm joints stay >= 0.5
   rad from their limits at the plateau (p10, ep600/ep800). With this grasp the palm PITCH slot is the one
   that turns the cup toward the receiver (+1.06 deg per deg). At ep800 the pitch command sits at its +45 deg
   delta-box edge in 91% of the plateau steps (ep600: 8%, same 0.75 stall ratio). It is not a hard limit.
   With the grasp held in the palm, palm orientations inside the delta box reach >= 93.6 deg toward the
   receiver (out-of-plane < 15 deg) in 99-100% of the plateau states, through yaw and roll.
2. What 10-40 deg more toward the receiver would earn or cost, per step, at the ep800 plateau states (pivot
   at the cup origin .. at the palm): +10 deg: premature -0.31..-0.33, aim +0.16..+0.18, net -0.17..-0.21;
   +20 deg: net -0.64..-1.01; +30 deg: net -0.98..-1.93 and 51-62% of the states cross the env latch
   (tilt > limit while the spout is more than 8.1 cm = 2.0 r from the mouth: success void, -LATCH_W every
   step to the end); +40 deg: net -1.36..-2.42, latch 83-97%. tilt_rate pays 0 for all of them.
   Speed and control charges are negligible here (action_rate -0.005/step, tremor 0 at adr_progress 0) and
   the grip terms fall only slightly with tilt (posture_src -0.04 per 10 deg within an episode above 40 deg).
3. The rotation help never reaches the states the policy visits. Phi = spout * lean with spout = fine *
   height and fine = exp(-(offset / (0.5 r))^2): at the plateau the spout is 0.087-0.137 m away (2.1-3.3 r,
   ep800 p10-p90), so Phi <= 1e-7 (p90, median 6e-11) and every move there pays and costs ~0.
4. Why no bead: a bead leaves only above about 90 + 8 * (1 - fill * in_source) deg of total tilt (93.6 deg
   at fill 0.55). The wall is at 52-58 deg, and at the wall the spout is outside the mouth (r) and mostly
   outside the latch radius (2.0 r), so going on would latch premature and put any bead on the table
   (spill at least -120 / n_active per bead).
5. The source dodges its lift reference. The env records the lift reference only if the cup is tilted less
   than 20 deg when it first rises 3 cm. From e500 to e700 the policy learned to tilt the source cup before
   that (tilted > 20 deg before 3 cm: 28% / 84% / 100% of episodes at ep300 / 600 / 800), so the reference is
   never recorded. Then posture_src pays the PRE-lift branch POSTURE_W_SRC * q_src for the whole carry (2.2 /
   2.7 per step at ep600 / ep800, max 3.0) instead of the lift-relative keep (<= KEEP_W_SRC = 1.5; 0.95-1.65
   where the reference exists), and keep_src = 1 by default, so grip_gate = 1 and kept_src = 1. The iter_24
   coupling and the iter_27 fingers-kept factor are switched off for the source. This tilt at lift is 22-31
   deg OUT of the pour plane (toward the robot, only 7 deg toward the receiver), and the out-of-plane lean
   stays during the carry (30-37 deg in episodes without a reference, 15-29 deg with one). Because premature
   uses the total tilt, this lean uses up most of the budget below 0.7 of the limit.
6. The pour pose is reachable and worth a lot, but the path pays almost nothing. At the ep800 plateau states
   the pour pose (lean toward d at the release angle, spout 1 cm over the mouth centre) earns +3.7 per step
   in aim alone. It needs the source body about 0.10 m (2.4 r) from the mouth, 4-10 cm closer and 4-8 cm
   higher than now (p10-p90). Source y then is -0.01..+0.02 (on the centre line) and the cup centres are 0.15-0.18 m
   apart (not nested). Stage by stage, cumulative per step: (a) turn the existing lean into the pour plane at
   the same total tilt +0.35 (no premature change); (b) raise 5 cm -0.13 (aim drop factor; lift_src is
   already saturated at h = LIFT_SRC_H); (c) approach to 0.104 m +1.43; (d) tilt to the limit with the spout
   centred +2.72; (e) to the release angle +3.66. Under iter_27, tilt_rate pays ~0 until the spout is within
   about 1 r of the mouth centre (Phi 2.6e-4 at 0.057 m after (a), 0.05 at 0.029 m). At the current height the source index fingertip is the closest point to the receiver and sits at
   its rim height; moving 4 / 6 cm closer without raising puts a source fingertip within 1 cm of the
   receiver wall below its rim in 21% / 46% of the states (outer wall radius assumed 4.5 cm), after a 5 cm
   raise in 0-4%.
7. The receiver five-finger loss is not visible to its term. The receiver lift reference is recorded in
   every episode, but the lift hold became lighter and uses fewer fingers (total 11.3 -> 7.9 N; middle
   finger > 1 N at lift 0.83 -> 0.19; pinky > 1 N at lift 0.03-0.11). A finger counts as "on at lift" at a
   force share > 0.02 (about 0.2 N) and as kept at half of its own lift force, and kept_rcv divides by the
   number of fingers on at lift. So kept_rcv is 0.96 (ep300), 0.79 (ep600), 0.96 (ep800) while the 1 N
   five-finger carry is 0.26 / 0.05 / 0.13. The pre-lift window is short (36-41 steps from grasp to the
   reference), and there each wrapped receiver finger pays 0.12 per step. A link to the receiver moving
   toward the centre line is not supported (correlation of its displacement from spawn with its loaded
   finger count -0.32 / +0.21 / -0.14).

REQUIRED CHANGES FOR ITER_28 (these three only)
1. Rotation help that reaches the pour approach. This is the pour-onset blocker. Keep tilt_rate as it is in
   iter_27: the exact FULL rate of Phi (central difference, both cups moved ahead and back by TILT_TAU,
   mouth, d, theta and lip rebuilt), rate-free pay gates (aim_gate) outside Phi, pay capped at TILT_PAY_CAP
   per step, negative rate charged in full and uncapped, TILT_W, and the latches (settled, 1 - released,
   want). Change only the geometry inside Phi:
   - Phi = latches * reach * over_rim * lean, lean = max(src_cup_up . d, 0) as now.
   - reach = exp(-(spout offset / (REACH_R * cup_r))^2) with REACH_R = 2.0. That is the env's premature latch
     radius (8.1 cm = 2.0 cup radii), in place of the 0.5 r precision width.
   - over_rim = the iter_26 factor for a spout below the receiver rim (BELOW_R). Do NOT put the drop factor in
     Phi. The drop height stays in aim, where it already is, so raising the source above the receiver rim
     before tilting is not charged by the rate.
   Effect on the ep800 plateau states (one-shot TILT_W * dPhi, gates 1): Phi 0.12 at the plateau (now
   6e-11); turning the lean into the pour plane +1.7, raising 0, approaching +1.5, tilting to the limit with
   the spout centred +0.85, to the release angle +0.19. Holding any pose still earns 0, and a tilt made with
   the spout 40 cm away still pays and costs ~0 (reach e^-24). premature and the env latch stay as they are,
   so tilting far from the mouth is still charged.
   - Check deterministically, as iter_27/check_tilt_rate.py did (adapt its Phi to this one): (a) every closed
     cycle sums to <= 0, including spout in/out at a fixed lean, jiggle around the (now wider) spout peak with
     all pay gates 1, and cycles that toggle the pay gates or ramp the latches; (b) a tilt made while the
     spout is far pays ~0 and costs ~0; (c) a lean away from or sideways to the receiver earns nothing and
     holding earns 0.
2. No grip income from dodging the lift reference (both hands). The pre-lift posture branch pays only
   while the cup is still at the table: multiply it by (1 - off), with off the existing lift-off ramp
   step((h - 0.01) / 0.04). A cup that is carried without a lift reference counts as keep = 0 wherever keep
   is used outside _grip_keep: keep_eff = valid * keep + (1 - valid) * (1 - off), used in grip_gate, in the
   finish factor and in the posture keep branch. _grip_keep itself stays byte-identical (iter_24 coupling).
   An upright lift keeps the pre-lift income until the env records the reference at 3 cm. A lift tilted past
   the env's record condition loses it by 5 cm and carries with posture 0 and grip_gate 0.3. On the trace
   states (changes 2 and 3 together) this costs the dodging carry steps 2.8 (ep600) / 3.6 (ep800) per step,
   and the ep300 carry steps, where the reference was recorded at lift, 0.18 per step.
3. Five-finger envelope counted out of ALL fingers after the lift (both hands; the queued receiver grip
   intent). In _fingers_kept divide by F = finger_force.shape[1], not by the number of fingers on at lift. A
   finger that was not on at the lift reference (share <= FINGER_ON_SHARE) counts as not kept. Each finger
   still counts relative to its OWN lift force (KEEP_FORCE_FRAC of ref share * ref total). There is no
   absolute force floor and no force target. A 4-finger lift then keeps at most 0.8: about -0.2 to -0.3 per
   step for the whole carry, which pays for loading all five fingers at lift. On the trace states (changes 2
   and 3 together): posture_rcv -0.14 (ep300) to -0.26 (ep800) per step on average.

KEEP (unchanged)
- No prescribed tilt angle, angle band, angle target or angle cap; pouring is driven by the flow of beads.
  Windows and distances in ctx.cup_radius multiples. Work near the y ~ 0 centre line.
- premature exactly as it is (ramp from PRE_LO of premature_tilt_limit, the lip factor, LATCH_W), aim
  exactly as it is (including the spout precision fine, the drop factor, DROP_FLOOR and aim_gate).
- The iter_27 receiver home anchor at side margin SIDE_OK, side_ok, still_rcv, CENTER_BAND on the source.
- The iter_27 pre-lift posture out of all fingers (wrap_count / F), and _fingers_kept relative to each
  finger's own lift force (only its denominator changes, change 3).
- The iter_24 lift-grasp coupling exactly as it is (hand pose, force share and total force AT LIFT).
- The iter_26 rules: need = min(requested - poured, beads still in the source); finish at the env tolerance
  with the body-based pre-tilt pose; spill per request-worth floored at OVER_DELTA_W.
- Control penalties (action_rate, flips, tremor) stay ramped on adr_progress; do not relax any speed or
  control penalty, including still_rcv and the stronger receiver-arm charge.
- approach, grasp, lift, pour_flow, overshoot, spill, finish, success, cup_collision and hand_foreign as in
  iter_27.
- Budget: state income must stay below outcome income at success. Change 1 adds no state income (holding
  earns 0). Change 2 removes up to 3.0 per step of posture income. Change 3 can only lower posture. At
  success: finish + success = 22 per step vs state income <= about 13 (aim fades with want).

WARM START: t2r_i27 ep300, not ep800/850 and not fresh.
- ep800/850 dodge the source reference in 100% of episodes, and change 2 charges exactly that lift (-3.6 per
  step on its carry states). A warm policy keeps its lift-time grasp habit and its lift reference, so ep800
  would first have to unlearn a tilted lift.
- ep300 records the source reference in 64/64 episodes (tilt at 3 cm 11.5 deg). It has no pour habit (0
  poured, 0 spilled), holds both cups at the end (0.94), has the closest approach of the three traces (body
  2.9 r, spout 8.6 cm at the end), the best receiver five-finger carry (0.26), and it predates the out-of-plane
  lift tilt. Its receiver lift hold loads 4.2 of 5 fingers (pinky share > 0.02 in 39%), so change 3 charges
  it from the first epoch and its pre-lift grasp has to add the missing fingers.
- ep500 had the reference in training (diag/grip_src_ref_valid 0.86) but has no deterministic trace.
- A fresh start re-spends about 300 epochs on grasp-lift.
Gate after 300 epochs (tensorboard + deterministic trace): diag/grip_src_ref_valid >= 0.8; source tilt at 3 cm
lift p50 < 20 deg; out-of-plane lean at the tilt peak p50 < 15 deg; spout offset at the tilt peak p50 < 2 r;
peak tilt toward the receiver p50 > 50 deg or a first bead in; receiver fingers on at lift >= 4.5 of 5 and
receiver 5-finger carry >= 0.5. If the gate fails, restart fresh with iter_28.
