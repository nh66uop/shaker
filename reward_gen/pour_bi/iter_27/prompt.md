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
    """[0,1] quality of one hand on its own cup. Only the SHAPE of the contact enters here
    (wrapped fingers, palm contact, thumb contact) - no cup dimension, so it survives a cup swap."""
    wrap_q = torch.clamp(wrap_count.to(dt) / 3.0, 0.0, 1.0)          # >= 3 wrapped fingers = envelope
    palm_q = torch.tanh(torch.clamp(palm_force, min=0.0) / 3.0)      # [N] force sensor scale
    thumb_q = torch.tanh(torch.clamp(finger_force[:, 0], min=0.0) / 0.5)
    return 0.5 * wrap_q + 0.25 * palm_q + 0.25 * thumb_q


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """iter_24, KEPT UNCHANGED (operator requirement 4): how closely the hand still holds the cup
    the way it did at the instant it lifted it - hand pose in CUP coordinates, force distribution
    and force level RELATIVE to that instant. Never an absolute force target. Before the reference
    exists the factor is 1 (nothing to keep yet)."""
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = torch.clamp(0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim, 0.0, 1.0)
    return torch.where(valid, keep, torch.ones_like(keep))


def _fingers_kept(ref_share, finger_force, valid, on_share, touch_n):
    """iter_26 change 5: share of the fingers that touched the cup AT LIFT that still touch it now.
    Purely relative to the lift hold; force enters only through a contact floor (a robot constant),
    never as a target. 1 before the reference exists or if no finger touched at lift."""
    ref_on = (ref_share[:, 1:] > on_share).to(finger_force.dtype)   # share = [palm, thumb..pinky]
    touch = _step(torch.clamp(finger_force, min=0.0) / touch_n)
    n_ref = torch.sum(ref_on, dim=-1)
    kept = torch.sum(ref_on * touch, dim=-1) / torch.clamp(n_ref, min=1.0)
    kept = torch.where(n_ref > 0.5, kept, torch.ones_like(kept))
    return torch.where(valid, kept, torch.ones_like(kept))


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dt = ctx.src_palm_pos.dtype

    # =============================== weights ==================================================
    APPROACH_W, APPROACH_K = 0.6, 4.0      # reach income, small: it must not compete with pouring
    GRASP_W, CLOSE_W, CLOSE_NORM = 0.8, 0.8, 0.35
    LIFT_SRC_W, LIFT_RCV_W = 3.0, 1.5      # iter_24 levels: these lifted reliably, do not weaken
    LIFT_SRC_H, LIFT_RCV_H, LIFTOFF_H = 0.20, 0.10, 0.03   # carry heights: workspace, not cup sizes
    POSTURE_W_SRC, POSTURE_W_RCV = 3.0, 1.2    # before the lift: build an enveloping grasp
    KEEP_W_SRC = 1.5                           # after the lift: keep THAT grasp (iter_24 design)
    # change 5: receiver keep 0.8 -> 1.5 (same as the source) and multiplied by the share of the
    # lift-time fingers still touching: losing one of five fingers costs 1.5 * 0.2 = 0.3 per step
    KEEP_W_RCV = 1.5
    KEEP_POS, KEEP_MAG = 0.02, 0.7             # [m] drift and |log force ratio| at which sim = 0.37
    FINGER_ON_SHARE, FINGER_TOUCH_N = 0.02, 0.2    # "touched at lift" force share / contact floor [N]
    AIM_W, AIM_K, AIM_FAR = 5.0, 10.0, 0.40    # workspace-scale guidance toward the receiver
    # change 4: spout precision and drop height, all in multiples of ctx.cup_radius
    AIM_PREC_R = 0.5       # gaussian width of the spout offset inside the receiver mouth
    DROP_R = 1.5           # spout height above the receiver rim at which low_drop = 0.37
    BELOW_R = 1.0          # spout below the receiver rim
    DROP_FLOOR = 0.4       # a high spout keeps 40 % of the aim income (collision margin)
    # change 1: rotation help as a potential RATE, only until the first bead has left the cup
    # c1r note: summed per step, not multiplied by dt: one rotation totals about TILT_W * dPhi / dt
    TILT_W, TILT_PAY_CAP = 5.0, 1.0            # pay capped at +1 per step, charge uncapped
    FLOW_W, FLOW_CAP, FLOW_FLOOR = 60.0, 0.03, -0.05   # paid per unit of bead fraction that ARRIVES
    OVER_DELTA_W = 120.0       # 2x FLOW_W: a bead past the request costs more than it ever paid
    OVER_LEVEL_W = 4.0         # sustained charge for sitting on an overshoot
    # change 4: spill charged per REQUEST-worth of beads (spilling as many beads as were requested
    # costs SPILL_W). target <= 1, so a spilled bead is never cheaper than in iter_25.
    # c1r M2: the per-fraction weight is floored at OVER_DELTA_W (2x FLOW_W), see the spill term
    SPILL_W = 50.0
    FINISH_W = 10.0            # sustained: request closed + cup upright at the pre-tilt pose + held
    # c1r H1: the pose factor is measured on the source cup BODY (tilt-independent), so the finish
    # rises monotonically as the cup comes upright at any pose and peaks upright at the pre-tilt pose.
    # tilted handover: FINISH_W * FIN_UP_FLOOR * (FIN_POSE_FLOOR + 0.6 * pose) >= AIM_W * geo, since
    # pose = 1 at the pour pose (inside the plateau) and 10 * 0.5 * 0.4 = 2.0 >= AIM_W * geo if geo <= 0.4
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

    # =============================== basic state ==============================================
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
    settled = _step(ctx.episode_progress / SETTLE_FRAC)

    # cup attitude as a dimensionless measure - NO angle constant anywhere
    up_src = torch.clamp(ctx.src_cup_up[:, 2], 0.0, 1.0)             # 1 = upright, 0 = horizontal
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], 0.0, 1.0)
    cup_r = max(float(ctx.cup_radius), 1e-3)                         # change 4: geometry unit from ctx

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
    kept_rcv = _fingers_kept(ctx.rcv_ref_force_share, ctx.rcv_finger_force,
                             ctx.rcv_grasp_ref_valid, FINGER_ON_SHARE, FINGER_TOUCH_N)
    v_src_ref = ctx.src_grasp_ref_valid.to(dt)
    v_rcv_ref = ctx.rcv_grasp_ref_valid.to(dt)
    grip_gate = 0.3 + 0.7 * keep_src        # a ruined grasp earns a fraction of the pour income

    # =============================== requested amount =========================================
    # everything below is expressed in FRACTIONS taken from ctx, so it follows any bead count
    poured = torch.clamp(ctx.bead_in_target_frac.to(dt), 0.0, 1.0)
    target = torch.clamp(ctx.pour_target_frac.to(dt), 0.0, 1.0)
    in_src = torch.clamp(ctx.bead_in_source_frac.to(dt), 0.0, 1.0)
    tol = 1.0 / torch.clamp(ctx.bead_active_count.to(dt), min=1.0)   # env tolerance = ONE bead
    d_in = ctx.d_in_target.to(dt)
    poured_prev = torch.clamp(poured - d_in, min=0.0)
    deficit_prev = torch.clamp(target - poured_prev, min=0.0)        # what was still requested
    err = torch.abs(poured - target)
    on_target = 1.0 - _step((err - tol) / (4.0 * tol))                # 1 within one bead, 0 at five
    # change 2: need = min(requested - poured, beads still in the source) - spilled beads are gone
    need = torch.clamp(torch.minimum(target - poured, in_src), min=0.0)
    # changes 2 + 3: want is on while MORE than one bead is still requested AND still available.
    # It turns off inside the env's one-bead tolerance or when the source can no longer supply the
    # request; exactly then the finishing income opens (closed = 1 - want).
    want = _step((need - tol) / tol)
    closed = 1.0 - want
    # change 1 (c1r M1): has the first bead LEFT the source cup? Built from cumulative bead events
    # (arrived + spilled), so a bead that falls back into the source cannot re-arm the tilt help
    spilled = torch.clamp(ctx.bead_spill_frac.to(dt), 0.0, 1.0)
    released = _step((poured + spilled) / tol)

    # =============================== stage 1-3: reach, grasp, lift ============================
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
    # posture: build the envelope before the lift, KEEP the lift grasp afterwards (iter_24).
    # change 5: the receiver keep also requires every finger that touched at lift to still touch,
    # during the whole carry and pour (held_rcv), relative to the lift hold only.
    posture_src = held_src * ((1.0 - v_src_ref) * POSTURE_W_SRC * q_src
                              + v_src_ref * KEEP_W_SRC * keep_src)
    posture_rcv = held_rcv * ((1.0 - v_rcv_ref) * POSTURE_W_RCV * q_rcv
                              + v_rcv_ref * KEEP_W_RCV * keep_rcv * kept_rcv)

    # =============================== stage 4: aim the LIP over the receiver ===================
    lip_delta = ctx.src_pour_lip_pos - ctx.rcv_cup_mouth_pos
    lip_xy = torch.norm(lip_delta[:, :2], dim=-1)
    lip_dz = lip_delta[:, 2]
    dir_ok = 1.0 - _step(ctx.pour_dir_xy[:, 0].to(dt) / DIR_ZERO)     # x > 0.3 voids success
    side_ok = _step((ctx.rcv_side_margin.to(dt) - SIDE_VOID) / (SIDE_OK - SIDE_VOID))
    # change 4: spill grew with the spout offset INSIDE the mouth and with the drop height, so the
    # aim now has a precision part and a height part, both in cup radii (no angle)
    fine = torch.exp(-(lip_xy / (AIM_PREC_R * cup_r)) ** 2)            # spout centred in the mouth
    over_rim = torch.exp(-(torch.clamp(-lip_dz, min=0.0) / (BELOW_R * cup_r)) ** 2)
    low_drop = torch.exp(-(torch.clamp(lip_dz, min=0.0) / (DROP_R * cup_r)) ** 2)
    height = over_rim * (DROP_FLOOR + (1.0 - DROP_FLOOR) * low_drop)
    aim_xy = 0.2 * torch.clamp(1.0 - lip_xy / AIM_FAR, 0.0, 1.0) \
        + 0.3 * torch.exp(-AIM_K * lip_xy) + 0.5 * fine
    aim_geo = aim_xy * height                 # spout geometry the aim pays for (c1r: not the finish)
    spout = fine * height                     # precise, low spout: gate for the rotation help
    aim_gate = carry_both * not_nested * dir_ok * side_ok * up_rcv * grip_gate
    aim = AIM_W * aim_gate * aim_geo * want                            # fades once the request closes

    # =============================== stage 4b: rotation help as a RATE (change 1) =============
    # potential phi = (1 - up_z) / 2 (0 upright, 1 inverted; the cup's own axis, no angle constant).
    # exact rate of the same expression: d(up)/dt = omega x up
    #   -> d(phi)/dt = 0.5 * (w_y * up_x - w_x * up_y)   [1/s]
    # (hypothesis: src_cup_ang_vel is in the same world/env frame as src_cup_up)
    # Holding any tilt earns 0. Pay is capped at +1 per step and its gate is <= the charge gate;
    # the charge is uncapped, so a closed tilt cycle before the first bead sums to <= 0 (step
    # sampling aside). The term switches off for good once the first bead has LEFT the source cup:
    # from then on only arriving beads pay (pour_flow) and turning the cup back costs nothing here.
    u = ctx.src_cup_up.to(dt)
    w = ctx.src_cup_ang_vel.to(dt)
    dphi = 0.5 * (w[:, 1] * u[:, 0] - w[:, 0] * u[:, 1])
    g_charge = settled * (1.0 - released) * want        # constant while no bead has left
    g_pay = g_charge * aim_gate * spout                  # <= g_charge everywhere
    tilt_rate = torch.where(dphi > 0.0,
                            torch.clamp(TILT_W * g_pay * dphi, max=TILT_PAY_CAP),
                            TILT_W * g_charge * dphi)

    # =============================== stage 5: the flow IS the pour ============================
    # income for every bead fraction that ARRIVES while it is still requested, capped per step so
    # that dumping the cup at once pays less than a controlled trickle
    useful = torch.clamp(torch.minimum(d_in, deficit_prev), FLOW_FLOOR, FLOW_CAP)
    pour_flow = FLOW_W * useful
    # charge for the part of this step's arrival that went PAST the request, plus a sustained charge
    # for staying overshot: pouring everything is strictly worse than stopping on time
    overshoot = -OVER_DELTA_W * torch.clamp(d_in - deficit_prev, min=0.0) \
        - OVER_LEVEL_W * torch.clamp(poured - target, min=0.0)
    # change 4: spilled beads measured in units of the REQUEST (the actor cannot observe spill);
    # c1r M2: floored at OVER_DELTA_W so one spilled bead always costs more than one arriving pays
    spill_w = torch.clamp(SPILL_W / torch.maximum(target, tol), min=OVER_DELTA_W)
    spill = -spill_w * settled * torch.clamp(ctx.d_spill.to(dt), min=0.0)

    # =============================== stage 6: finish upright at the pre-tilt pose =============
    # change 3: opens at the env tolerance (|error| <= 1 bead) or when the request became
    # unreachable (closed = 1 - want). Pays for standing the source cup upright AND returning to
    # the pre-tilt pose, both cups still held.
    # c1r H1: the reward has no memory and the lip moves with the tilt, so the pre-tilt pose is
    # measured on the source cup BODY: horizontal distance from ctx.src_cup_pos to the receiver
    # mouth. Full credit up to cup_mouth_z + POSE_TOL_R cup radii (covers the upright side-by-side
    # pose and the body position while pouring), gaussian falloff beyond, none while nested (the
    # aim never paid a nested pose). This factor does not depend on the tilt, so the up factor makes
    # the finish rise monotonically as the cup comes upright; the maximum is upright at that pose.
    body_xy = torch.norm(ctx.src_cup_pos[:, :2] - ctx.rcv_cup_mouth_pos[:, :2], dim=-1)
    pose_tol = max(float(ctx.cup_mouth_z), 0.0) + POSE_TOL_R * cup_r
    pre_tilt_pose = not_nested * torch.exp(
        -(torch.clamp(body_xy - pose_tol, min=0.0) / (POSE_SIG_R * cup_r)) ** 2)
    finish = FINISH_W * closed * (0.3 + 0.7 * on_target) \
        * (FIN_UP_FLOOR + (1.0 - FIN_UP_FLOOR) * up_src) \
        * (FIN_POSE_FLOOR + (1.0 - FIN_POSE_FLOOR) * pre_tilt_pose) \
        * carry_both * up_rcv * (0.4 + 0.6 * keep_src)
    success = SUCCESS_W * ctx.success.to(dt)

    # =============================== workspace constraints ====================================
    # the source cup must not do its work deep on the receiver's side; with side_ok on the
    # receiver this makes the two cups meet near the centre line
    center_src = -CENTER_W * held_src * torch.tanh(
        torch.clamp(ctx.src_cup_pos[:, 1] - CENTER_BAND, min=0.0) / CENTER_SCALE)
    d_home = torch.norm(ctx.rcv_cup_pos[:, :2] - ctx.rcv_cup_spawn_pos[:, :2], dim=-1)
    still_rcv = 1.0 - torch.tanh(torch.clamp(torch.norm(ctx.rcv_cup_lin_vel, dim=-1) - 0.05,
                                             min=0.0) / 0.15)
    home_rcv = HOME_W * carry_rcv * up_rcv * side_ok * torch.exp(-4.0 * d_home) * still_rcv
    # premature tilt: a FRACTION of the env's own limit, charged only while the lip is not yet over
    # the receiver, plus the episode latch itself (unchanged)
    limit = torch.clamp(ctx.premature_tilt_limit.to(dt), min=1e-3)
    risk = _step((ctx.src_cup_tilt.to(dt) / limit - PRE_LO) / (1.0 - PRE_LO))
    premature = -PRE_W * held_src * (1.0 - torch.exp(-AIM_K * lip_xy)) * risk \
        - LATCH_W * ctx.premature_tilt.to(dt)

    # =============================== safety (real robot) ======================================
    f_cup = ctx.cup_cup_force
    cup_collision = -CONTACT_W * torch.tanh(f_cup / 2.0) \
        - HIT_W * settled * (f_cup > 0.5).to(dt) - HIT_LATCH_W * ctx.cup_hit.to(dt)
    f_s = torch.clamp(ctx.src_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    f_r = torch.clamp(ctx.rcv_hand_foreign_force - FOREIGN_DEAD, min=0.0)
    hand_foreign = -(FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_src) \
        * torch.tanh(f_s / 5.0) \
        - (FOREIGN_W_FREE + (FOREIGN_W_HELD - FOREIGN_W_FREE) * held_rcv) * torch.tanh(f_r / 5.0)

    # =============================== control (ramped on adr_progress, unchanged) ==============
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
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.09e-08 · max 3.05e-05
bead/spill: [0, 0.000232, 0.000638, 0.000106, 6.88e-05, 0, 0.000183, 0, 0.000316, 0]  min 0 · mean 0.000201 · max 0.0025
done/drop: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000114 · max 0.00928
episode_lengths/step: [124, 854, 894, 898, 887, 875, 888, 875, 896, 886]  min 106 · mean 867 · max 899
reward/action_rate: [-0.138, -0.0947, -0.0846, -0.0793, -0.0705, -0.0633, -0.0568, -0.0552, -0.0524, -0.046]  min -0.139 · mean -0.0694 · max -0.0434
reward/aim: [0, 0.00762, 0.00543, 0.0197, 0.0359, 0.0848, 0.102, 0.152, 0.14, 0.234]  min 0 · mean 0.0973 · max 0.363
reward/approach_rcv: [0.308, 0.391, 0.387, 0.402, 0.395, 0.391, 0.391, 0.394, 0.391, 0.399]  min 0.3 · mean 0.392 · max 0.404
reward/approach_src: [0.302, 0.392, 0.403, 0.404, 0.403, 0.405, 0.406, 0.405, 0.404, 0.406]  min 0.299 · mean 0.4 · max 0.407
reward/center_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.000171]  min -0.000591 · mean -2.64e-05 · max 0
reward/cup_collision: [-0.0124, 0, 0, 0, -7.32e-05, 0, -7.32e-05, 0, -0.00022, 0]  min -0.15 · mean -0.00156 · max 0
reward/finish: [0, 0, 7.03e-05, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.19e-05 · max 0.000531
reward/grasp_rcv: [0.0413, 1.07, 1.09, 1.1, 1.09, 1.09, 1.09, 1.1, 1.09, 1.1]  min 0.0357 · mean 1.08 · max 1.13
reward/grasp_src: [0.0401, 1.04, 1.1, 1.1, 1.1, 1.11, 1.11, 1.11, 1.11, 1.11]  min 0.0335 · mean 1.08 · max 1.13
reward/hand_foreign: [-0.0458, -0.0915, -0.0364, -0.0255, -0.0318, -0.026, -0.0295, -0.0364, -0.0341, -0.0262]  min -0.132 · mean -0.0384 · max -0.0147
reward/home_rcv: [2.79e-05, 0.179, 0.25, 0.285, 0.381, 0.404, 0.455, 0.458, 0.48, 0.529]  min 9.54e-06 · mean 0.38 · max 0.559
reward/lift_rcv: [0.000243, 1.12, 1.24, 1.22, 1.15, 1.17, 1.18, 1.16, 1.21, 1.21]  min 0.000243 · mean 1.16 · max 1.25
reward/lift_src: [0.00304, 2.04, 2.24, 2.36, 2.24, 2.32, 2.3, 2.15, 2.28, 2.24]  min 0.000542 · mean 2.17 · max 2.42
reward/overshoot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/posture_rcv: [3.86e-06, 0.635, 0.587, 0.679, 0.741, 0.719, 0.697, 0.642, 0.701, 0.701]  min 3.86e-06 · mean 0.667 · max 0.762
reward/posture_src: [1.65e-05, 0.728, 0.879, 0.838, 0.864, 0.781, 0.831, 0.91, 0.812, 0.823]  min 1.65e-05 · mean 0.82 · max 0.939
reward/pour_flow: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/premature: [0, -0.0142, -0.0156, -0.00404, -0.00252, -0.00108, -0.00197, -0.00175, -0.00399, -0.000732]  min -0.171 · mean -0.0119 · max 0
reward/spill: [0, 0, -0.00293, 0, 0, 0, 0, 0, 0, 0]  min -0.0183 · mean -0.000206 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_rate: [-0.0325, -0.267, -0.262, -0.238, -0.189, -0.214, -0.202, -0.158, -0.201, -0.168]  min -0.372 · mean -0.21 · max -0.0325
reward/total: [0.465, 7.11, 7.75, 8.03, 8.07, 8.14, 8.25, 8.2, 8.3, 8.48]  min 0.0632 · mean 7.88 · max 8.77
reward/tremor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
rewards/step: [26.1, 6.04e+03, 6.92e+03, 7.23e+03, 7.24e+03, 7.14e+03, 7.28e+03, 7.12e+03, 7.38e+03, 7.62e+03]  min 23.7 · mean 6.92e+03 · max 7.81e+03
task/aim_dist: [0.254, 0.45, 0.576, 0.433, 0.414, 0.376, 0.358, 0.332, 0.333, 0.409]  min 0.234 · mean 0.399 · max 0.67
task/cup_collision_rate: [0.00146, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000131 · max 0.0117
task/cups_center_dist: [0.285, 0.449, 0.573, 0.426, 0.412, 0.375, 0.356, 0.336, 0.33, 0.403]  min 0.231 · mean 0.396 · max 0.665
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.62e-05 · max 0.00146
task/rcv_cup_lift: [0.0027, 0.144, 0.264, 0.127, 0.0886, 0.0906, 0.091, 0.0898, 0.0945, 0.155]  min -0.000135 · mean 0.118 · max 0.287
task/rcv_grasped: [0, 0.846, 0.836, 0.832, 0.839, 0.834, 0.843, 0.825, 0.845, 0.847]  min 0 · mean 0.825 · max 0.871
task/rcv_hand_foreign_rate: [0.14, 0.0205, 0.0134, 0.0154, 0.0186, 0.0293, 0.0332, 0.0537, 0.0476, 0.0559]  min 0.0107 · mean 0.0341 · max 0.14
task/src_cup_lift: [0.00232, 0.117, 0.14, 0.164, 0.146, 0.149, 0.155, 0.129, 0.147, 0.135]  min -0.000141 · mean 0.138 · max 0.19
task/src_grasped: [0, 0.808, 0.854, 0.851, 0.856, 0.864, 0.867, 0.87, 0.872, 0.858]  min 0 · mean 0.834 · max 0.879
task/src_hand_foreign_rate: [0.146, 0.104, 0.0596, 0.0479, 0.0415, 0.0432, 0.0583, 0.061, 0.0623, 0.0325]  min 0.0193 · mean 0.058 · max 0.173
task/src_tilt_deg: [10.6, 32.1, 35.5, 31.2, 26.9, 29.2, 28.3, 24.2, 26.9, 21.8]  min 5.79 · mean 27.9 · max 38.1
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
ROUND 27 NOTES (09.26, diagnosis of t2r_i26r = iter_26 c1r trained FRESH). Facts only. The three changes
below are the only changes for this round; everything else in iter_26 stays as it is.

TASK DEFINITION (operator, 09.25, unchanged): grasp both cups with both hands and keep the envelope (all
five fingers on the cup, grip held), pour the REQUESTED amount (the target amount that is in the actor
observation), then bring the source cup back upright and return to the pose it had before tilting.

WHAT t2r_i26r DID (deterministic replay, ADR 0, 64 envs x 900 steps; 10 active beads p50, request 5.5 p50)
| metric | ep300 | ep600 | ep800 |
|---|---|---|---|
| success / any bead poured / any bead spilled | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| both cups held at the end | 0.98 | 0.94 | 0.98 |
| episode peak TOTAL tilt p50 (p10-p90) | 41 (38-50) deg | 48 (45-52) deg | 33.5 (27-40) deg |
| episode peak tilt TOWARD the receiver (src_tilt_toward_rcv) p50 / p90 | -2 / 13.5 deg | 5.8 / 10.9 deg | 6.5 / 10.0 deg |
| spout (src_pour_lip_pos) to receiver mouth, horizontal, end p50 | 0.41 m | 0.33 m | 0.19 m |
| source cup body to receiver mouth, horizontal, end p50 | 0.44 m | 0.37 m | 0.23 m (5.6 cup radii) |
| source cup y / receiver cup y at the end p50 | -0.195 / +0.242 | -0.180 / +0.189 | -0.064 / +0.160 |
| receiver cup distance from its spawn, end p50 | 8.8 cm | 5.3 cm | 2.8 cm |
| source 5-finger contact while carrying (1 N) / pinky alone | 0.93 / 0.99 | 0.98 / 0.99 | 0.38 / 0.39 |
| receiver 5-finger contact while carrying (1 N) / pinky alone | 0.73 / 0.87 | 0.31 / 0.59 | 0.17 / 0.19 |
| tilt_rate per episode, paid / charged | 0 / -125 | 0 / -82 | 0 / -76 |
Tensorboard per-step means (21-epoch windows at e100 / e300 / e600 / e790, then the last 20 epochs to
e866): tilt_rate -0.29 / -0.22 / -0.16 / -0.17 / -0.16 (the largest charge; action_rate -0.04), aim
0.005 / 0.02 / 0.15 / 0.26 / 0.34, task/aim_dist 0.52 / 0.43 / 0.40 / 0.29 / 0.24 m, home_rcv 0.20 /
0.36 / 0.46 / 0.54 / 0.51, source wrap count 2.51 / 2.44 / 2.39 / 2.03 / 1.65, finish and pour_flow 0
throughout, adr_progress 0.

WHY (verified by recomputing the iter_26 terms offline on the trace states)
1. The rotation term never paid; it only charged. Its pay gate contains spout = fine * height, and the
   spout never came near the mouth: spout <= 1e-17 on every active step of all three traces. So every
   increase of (1 - up_z)/2 paid 0 and every decrease was charged in full (g_charge = 1 before the first
   bead). That makes it a tax on holding any tilt and on moving the cup. Charge per step by total tilt
   (ep300): 0-10 deg -0.03, 30-40 deg -0.16, 40-50 deg -0.22. By source cup speed at 25-35 deg (ep800):
   < 3 cm/s -0.06, 6-10 cm/s -0.12, > 10 cm/s -0.22. Moving the source cup 1 cm closer earns only +0.025
   aim per step at the ep800 states, so a quick approach or an exploratory tilt costs more than it earns.
2. No pouring rotation ever started. The "peak tilt" above is a backward lean of the source cup toward
   the torso (cup axis x component -0.51 to -0.71 at the end, toward-receiver component -0.18 to +0.04)
   that grows with lift height (correlation 0.61-0.93). The first bead needs about 90-98 deg total tilt.
   At the same lift height (0.16-0.20 m) the lean fell from 36.6 deg (ep600) to 28.1 deg (ep800).
3. The receiver does not come to meet the source, and the meeting point is off the centre line. At any
   tilt the spout is at most cup_mouth_z * sin + r * cos <= 0.108 m (2.6 cup radii) from the source cup
   axis, so the source body must come within about 3.6 cup radii of the receiver mouth; at ep800 it is
   5.6 cup radii away. home_rcv pays exp(-4 * distance from the receiver's SPAWN). On the ep800 states,
   moving the receiver 2 / 4 / 8 / 12 cm toward the source changes aim + home_rcv by +0.027 / +0.035 /
   -0.042 / -0.97 per step. So the receiver stays within 3-4 cm of spawn (y ~ +0.16), the source does
   about 0.2 m of the travel alone, and the implied pour point is on the receiver's side (source body at
   y ~ +0.02..+0.06), not at y ~ 0.
4. The source five-finger loss is the pinky (thumb, index, middle and ring stay >= 0.985). Pinky contact
   at the lift reference fell from 97% (ep600) to 59% (ep800) of episodes. Where the reference had it, it
   stayed on for 43% of the carry steps, and less as the source moves toward the centre (50% at y -0.15
   to -0.10, 24% at y > -0.05). The reward does not ask for it: the pre-lift posture saturates at 3
   wrapped fingers (wrap_q = min(wrap_count / 3, 1)) and the source keep has no fingers-kept factor.
   Losing the pinky lowers keep_src from 0.644 to 0.617, i.e. -0.04 per step. It is not the tilt charge:
   at equal tilt the charge is slightly larger with the pinky off (-0.116 vs -0.099 per step).
5. The receiver five-finger loss is invisible to its iter_26 term. A finger counts as "still touching"
   at 0.2 N, and only if its force share at lift was > 0.02. The receiver lifts with the pinky barely on
   (1 N at lift in 6-20% of episodes, share > 0.02 in 39-72%), so kept_rcv stays 0.95-0.99 while the
   1 N five-finger contact falls from 0.73 to 0.17.
6. finish never opened (want = 1 on every step because nothing was poured), so the iter_26 finish,
   spill and want changes are still untested.

REQUIRED CHANGES FOR ITER_27 (these three only)
1. Rotation help as the exact rate of a potential. This replaces the iter_26 tilt_rate and keeps its
   name, TILT_W, TILT_PAY_CAP and the switch-off once the first bead has LEFT the source cup.
   - Potential Phi = latches * spout * lean. lean = max(src_cup_up . d, 0), with d the horizontal pour
     direction (ctx.pour_dir_xy), is how far the cup axis leans TOWARD the receiver (no angle constant).
     spout = the iter_26 spout geometry (fine * height, cup-radius widths). latches = settled *
     (1 - released) * want; these only move one way before the first bead.
   - Pay TILT_W * dPhi/dt, where dPhi/dt is the FULL rate of Phi, including the change of spout and d
     caused by both cups' motion. One way is (Phi(now) - Phi(state moved back by a small time tau along
     ctx.src/rcv_cup_lin_vel and _ang_vel)) / tau, recomputing mouth, d and lip with the formulas in the
     ctx comments. Holding any pose earns 0.
   - A positive rate may be multiplied by factors in [0, 1] that have no rate (carry, grip_gate,
     not_nested, dir_ok, side_ok, up_rcv) and capped at TILT_PAY_CAP per step. A negative rate is charged
     in full and uncapped. None of these factors may sit inside Phi (an uncharged gate change is a pump).
   - Check deterministically, as iter_26/check_tilt_rate.py did: (a) every closed cycle sums to <= 0,
     including cycles that move the tilted spout in and out of the mouth and cycles that toggle the
     factors; (b) a tilt made while spout = 0 pays 0 and costs 0, so there is no net charge; (c) a lean
     away from or sideways to the receiver earns nothing.
2. Meet near the centre line. home_rcv keeps its form (HOME_W, carry_rcv, up_rcv, side_ok and still_rcv
   unchanged), but its anchor moves from the receiver's spawn to its spawn x at side margin SIDE_OK
   (ctx.rcv_side_margin = SIDE_OK, the smallest margin that still has full side_ok; the env voids success
   below SIDE_VOID). The receiver then comes about 8 cm toward the centre, and the pour happens near
   y ~ 0 (receiver near +0.08, source body near -0.02) without the source crossing CENTER_BAND.
3. Five-finger envelope on BOTH hands, relative to the lift hold (task definition: all five fingers).
   - Before the lift reference exists, the posture term counts wrapped fingers out of ALL fingers
     (wrap_count / F, F = ctx.src_finger_force.shape[1]) instead of saturating at 3, so that the lift
     reference records a five-finger hold.
   - After the lift, the SOURCE keep gets the same fingers-kept factor as the receiver. For both hands a
     finger counts as still touching relative to its OWN force at the lift reference (ref share * ref
     total, e.g. at least half of it), not at an absolute 0.2 N floor. No absolute force target.

KEEP (unchanged)
- No prescribed tilt angle, angle band, angle target or angle cap; pouring is driven by the flow of
  beads. Windows and distances in ctx.cup_radius multiples.
- The iter_26 rules: need = min(requested - poured, beads still in the source); finish at the env
  tolerance with the body-based pre-tilt pose; spill per request-worth floored at OVER_DELTA_W; spout
  precision and drop height in aim; the receiver fingers-kept factor (made relative by change 3).
- The iter_24 lift-grasp coupling exactly as it is (hand pose, force share and total force AT LIFT).
- Centre-line term for the source side (CENTER_BAND); side_ok on the receiver; the stronger tremor and
  control charge on the receiver arm.
- Control penalties (action_rate, flips, tremor) stay ramped on adr_progress; do not relax any speed or
  control penalty, including still_rcv. tilt_rate is the rotation potential, not a control penalty.
- approach, grasp, lift, aim, pour_flow, overshoot, spill, finish, success, premature, cup_collision
  and hand_foreign as in iter_26.
- Budget: state income must stay below outcome income at success. Change 1 adds no state income,
  because holding a tilt earns 0.

WARM START: t2r_i26r ep800, not fresh. The policy has no pour habit to unlearn (0 poured, 0 spilled, 0
premature in all three traces), grasp-lift-carry is solid (both cups held at the end 0.98), and its
action std has not collapsed (0.32-0.93 per dimension, vs 0.58-0.85 at ep300). It is also the furthest
along the approach (spout 0.19 m). The three changes act on states it already visits: change 1 removes
a -0.16/step charge there (spout ~ 0 there, so the new term is ~ 0), change 2 moves the receiver anchor by 8 cm, and change 3 asks for the pinky
that the ep600 policy still had. ep600 has the better source pinky (0.99) but the largest backward lean
(48 deg), a source cup that drifted away late (y -0.18, spout 0.33 m), and receiver 5-finger contact
already at 0.31. A fresh start re-spends about 300 epochs on grasp-lift (the i26r milestone came at
ep300). Gate after 300 epochs (deterministic trace): peak tilt toward the receiver p50 > 20 deg, spout
to mouth at the end p50 < 0.12 m, receiver side margin p50 < 0.12 m, source 5-finger carry >= 0.8,
receiver >= 0.5. If the gate fails, restart fresh with iter_27.
