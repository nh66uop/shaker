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


def _grip_keep(palm_in, tips_in, ref_palm, ref_tips, share, ref_share, total, ref_total, valid,
               pos_scale, mag_scale):
    """How closely the hand still holds the cup the way it did at the moment it lifted it, in [0,1].

    All three parts are RELATIVE to that moment (operator 09.22: no absolute force target):
      * pose  - palm + fingertips in cup coordinates; a hand that keeps its grip has them fixed no matter
                how the cup is rotated (i18/i23 traces: they drift 2.4-4.2 cm while tilting)
      * share - palm/finger force DISTRIBUTION (i18/i23: about half of it moves while tilting)
      * mag   - total grip force relative to the lift moment (i18/i23: 3-5x harder while tilting)
    Before the reference exists (cup not lifted yet) the factor is 1 - nothing to keep yet.
    """
    drift = torch.cat([torch.norm(palm_in - ref_palm, dim=-1, keepdim=True),
                       torch.norm(tips_in - ref_tips, dim=-1)], dim=1).mean(dim=1)
    pose_sim = torch.exp(-drift / pos_scale)
    share_sim = 1.0 - 0.5 * torch.sum(torch.abs(share - ref_share), dim=-1)
    mag_sim = torch.exp(-torch.abs(torch.log((total + 0.5) / (ref_total + 0.5))) / mag_scale)
    keep = 0.4 * pose_sim + 0.3 * share_sim + 0.3 * mag_sim
    return torch.where(valid, keep, torch.ones_like(keep))


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
    POSTURE_KEEP_W = 1.5      # after the lift: "still the grasp you lifted with" (half the pre-lift weight)
    POSTURE_KEEP_RCV_W = 0.8
    SIDE_W = 1.5              # pour side: palm away from the receiver while carrying toward it
    GEO_PALM_FROM = 0.06      # [m] palm this far UNDER the tilted cup = geometry 0 (i23 sits at +6 cm)
    GEO_TIP_FROM = 0.04       # [m] fingertip this far ON TOP of the tilted cup = 0 (i23 sits at -4 cm)
    POSTURE_POUR_BONUS = 0.6  # x1.6 while pouring / after the pour
    TIP_W = 1.2               # explicit charge for fingertip-only holding (logged separately)
    TIP_POUR_BONUS = 1.0      # x2.0 during the pour
    PALM_SLIP_W = 1.5         # charge for the palm force collapsing below PALM_MIN while held
    PALM_MIN = 3.0            # [N] acceptance target for the pour/late phases
    GRIP_FLOOR = 0.15         # rotating with a ruined grasp keeps only 15 % of the tilt income
    HOLD_FLOOR = 0.20         # iter_25: rotation income kept when the hand geometry is wrong
    GEO_MARGIN = 0.03         # [m] palm this far ABOVE / fingertips this far BELOW the cup axis = full credit
    GEO_TILT_LO = math.radians(15.0)   # geometry requirement starts here ...
    GEO_TILT_HI = math.radians(45.0)   # ... and is fully required from here on
    KEEP_POS = 0.02           # [m] hand-in-cup drift from the lift moment at which pose similarity is 0.37
    KEEP_MAG = 0.7            # |log(force / force at lift)| at which force similarity is 0.37 (2x -> 0.37)

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
    # ---- iter_25 ---- hand geometry while the cup is tilted (operator 09.22): the palm must be on the
    # UPPER side of the tilted cup and the other fingers UNDER it, carrying the cup. i23 did the
    # opposite (palm 6 cm on the gravity side, the 4 fingers 4 cm on top, no finger underneath).
    # Grasp maintenance is what ENABLES the rotation income: wrong geometry keeps only 20 %.
    u_src = ctx.src_cup_up.to(dt)
    down = torch.zeros_like(u_src)
    down[:, 2] = -1.0
    d_perp = down - torch.sum(down * u_src, dim=-1, keepdim=True) * u_src          # gravity across the axis
    d_low = d_perp / torch.clamp(torch.norm(d_perp, dim=-1, keepdim=True), min=1e-6)
    palm_low = torch.sum((ctx.src_palm_pos - ctx.src_cup_pos) * d_low, dim=-1)       # + = under the cup
    tips_low = torch.sum((ctx.src_tips_pos - ctx.src_cup_pos.unsqueeze(1)) * d_low.unsqueeze(1), dim=-1)
    # continuous over the whole range the hand can be in, so a policy that starts with the palm UNDER
    # the cup (i23: +6 cm) still sees which way is better: 0 at palm 6 cm under -> 1 at 3 cm above,
    # and per finger 0 at 4 cm on top -> 1 at 3 cm under (touching)
    palm_top = _step((GEO_PALM_FROM - palm_low) / (GEO_PALM_FROM + GEO_MARGIN))
    under = _step((tips_low[:, 1:] + GEO_TIP_FROM) / (GEO_TIP_FROM + GEO_MARGIN)) \
        * (ctx.src_finger_force[:, 1:] > 0.3).to(dt)
    fingers_under = torch.clamp(torch.sum(under, dim=-1) / 2.0, 0.0, 1.0)             # >= 2 fingers carry it
    geo_src = palm_top * fingers_under
    geo_on = _step((tilt_mag - GEO_TILT_LO) / (GEO_TILT_HI - GEO_TILT_LO))
    # pour side, defined while the cup is still upright: the palm must sit on the side of the source cup
    # AWAY from the receiver so that tipping toward the receiver turns the palm on top. i23 carried the
    # source cup past the receiver and tipped toward the palm (cos +0.77). Linear, so there is a
    # gradient from any arrangement: cos +1 -> 0, cos -0.6 or less -> 1.
    palm_side = (ctx.src_palm_pos - ctx.src_cup_pos)[:, :2]
    palm_side = palm_side / torch.clamp(torch.norm(palm_side, dim=-1, keepdim=True), min=1e-6)
    side_cos = torch.sum(palm_side * ctx.pour_dir_xy.to(dt), dim=-1)
    side_lin = torch.clamp((1.0 - side_cos) / 1.6, 0.0, 1.0)
    # operator 09.22: no absolute force target - the grasp must stay as it was when the cup was first
    # lifted (hand pose in cup coordinates, force distribution, force level), and keeping it is what
    # unlocks the rotation income together with the palm-on-top / fingers-under geometry
    keep_src = _grip_keep(ctx.src_palm_in_cup, ctx.src_tips_in_cup, ctx.src_ref_palm_in_cup,
                          ctx.src_ref_tips_in_cup, ctx.src_force_share, ctx.src_ref_force_share,
                          ctx.src_force_total, ctx.src_ref_force_total, ctx.src_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    keep_rcv = _grip_keep(ctx.rcv_palm_in_cup, ctx.rcv_tips_in_cup, ctx.rcv_ref_palm_in_cup,
                          ctx.rcv_ref_tips_in_cup, ctx.rcv_force_share, ctx.rcv_ref_force_share,
                          ctx.rcv_force_total, ctx.rcv_ref_force_total, ctx.rcv_grasp_ref_valid,
                          KEEP_POS, KEEP_MAG)
    rot_hold_src = HOLD_FLOOR + (1.0 - HOLD_FLOOR) * env_src \
        * ((1.0 - geo_on) + geo_on * geo_src * keep_src)

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
    # before the lift the envelope quality q builds the grasp; from the lift on the posture income is
    # "the grasp is still the one you lifted with", and once tilted it also needs the pour geometry
    v_src = ctx.src_grasp_ref_valid.to(dt)
    v_rcv = ctx.rcv_grasp_ref_valid.to(dt)
    q_hold_src = (1.0 - v_src) * q_src + v_src * keep_src
    q_hold_rcv = (1.0 - v_rcv) * q_rcv + v_rcv * keep_rcv
    # after the lift the keep income is smaller than the pre-lift envelope income and is NOT multiplied
    # by the pour geometry: holding a lifted cup still must never out-earn turning it (local minimum),
    # the geometry is enforced through the rotation income only
    posture_src = held_src * ((1.0 - v_src) * POSTURE_SRC_W * q_src + v_src * POSTURE_KEEP_W * keep_src) \
        * (1.0 + POSTURE_POUR_BONUS * pour_phase)
    posture_rcv = held_rcv * ((1.0 - v_rcv) * POSTURE_RCV_W * q_rcv + v_rcv * POSTURE_KEEP_RCV_W * keep_rcv)
    # iter_24: no pour-phase amplification - tilting further must never raise a grasp penalty by itself
    tip_only_src = -TIP_W * held_src * tip_src
    tip_only_rcv = -TIP_W * held_rcv * tip_rcv

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
    # bring the source cup in from the side that will put the palm on top (paid while carrying toward
    # the receiver, before and during the tilt)
    pour_side = SIDE_W * carry_both * not_nested_f * conv_xy * side_lin
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
        * dir_ok * rot_hold_src
    # (a) below the premature limit and inside the approach zone: safe, cannot latch, gives the
    #     gradient that starts the rotation once the cup is high and on its way
    tilt_pre = TILT_PRE_W * tilt_gate * (torch.minimum(theta_eff, theta_pre)
                                         / torch.clamp(theta_peak, min=0.3)) * approach_gate * hover_wide
    # (b) the real pour: only with the lip inside the release radius and above the receiver rim
    # ---- iter_23 FIX ---- the subtraction exists only so the rotation below theta_pre is not paid
    # twice (tilt_pre already pays it at TILT_PRE_W). Subtracting the FULL fraction cancelled the
    # whole bell for every theta <= theta_pre (decline is still 1 there), so tilt_main was
    # identically 0 below 50 deg and iter_22 stalled at 22 deg for 1180 epochs. Remove only what
    # tilt_pre actually paid, scaled into tilt_main units.
    tilt_main = TILT_MAIN_W * tilt_gate * torch.clamp(
        bell - (TILT_PRE_W / TILT_MAIN_W) * torch.minimum(theta_eff, theta_pre)
        / torch.clamp(theta_peak, min=0.3), min=0.0) \
        * safe * lip_above
    # holding the release pose (lip over the mouth, angle in the release band, receiver upright)
    pour_band = _step((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = POSE_W * stack_gate * rot_hold_src * beads_left * safe * lip_above * aim_xy \
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
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "both_lifted": both_lifted,
        "converge": converge,
        "bring_together": bring_together,
        "aim": aim,
        "pour_dir": pour_dir,
        "pour_side": pour_side,
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

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0.0333, 0.133, 0.233]  min 0 · mean 0.0531 · max 0.333
bead/in_target: [0, 0, 0, 0, 0, 0, 0.0532, 0.565, 0.654, 0.682]  min 0 · mean 0.229 · max 0.699
bead/spill: [0.00104, 0.00675, 0.00822, 0.0102, 0.011, 0.0111, 0.0186, 0.0208, 0.0198, 0.0138]  min 0.00104 · mean 0.014 · max 0.0394
done/drop: [0, 0, 0.000488, 0.000244, 0, 0, 0, 0, 0, 0.000244]  min 0 · mean 9.24e-05 · max 0.0112
episode_lengths/step: [123, 875, 887, 894, 894, 880, 896, 896, 889, 884]  min 102 · mean 876 · max 899
reward/act_flip: [0, 0, 0, 0, 0, 0, 0, -0.0416, -0.383, -0.475]  min -0.504 · mean -0.107 · max 0
reward/aim: [0, 8.36e-05, 0.00663, 0.325, 1.03, 1.49, 1.55, 1.46, 1.47, 1.28]  min 0 · mean 0.932 · max 1.59
reward/approach_rcv: [0.311, 0.392, 0.378, 0.386, 0.376, 0.374, 0.374, 0.362, 0.359, 0.348]  min 0.304 · mean 0.371 · max 0.403
reward/approach_src: [0.302, 0.399, 0.397, 0.405, 0.405, 0.397, 0.397, 0.398, 0.387, 0.386]  min 0.302 · mean 0.394 · max 0.406
reward/both_grasped: [0, 0.682, 0.671, 0.685, 0.678, 0.672, 0.688, 0.683, 0.674, 0.682]  min 0 · mean 0.666 · max 0.698
reward/both_lifted: [0, 0.25, 0.538, 0.585, 0.601, 0.633, 0.654, 0.636, 0.644, 0.635]  min 0 · mean 0.547 · max 0.66
reward/bring_together: [0, 0.000612, 0.0132, 0.303, 0.794, 1.08, 1.13, 1.07, 1.08, 0.957]  min 0 · mean 0.694 · max 1.15
reward/closing_speed: [0, 0, 0, -0.00497, -0.00584, -0.00212, -0.000474, -0.00247, -0.000637, -0.00155]  min -0.0114 · mean -0.00172 · max 0
reward/converge: [0, 0.000529, 0.194, 0.913, 1.29, 1.46, 1.47, 0.866, 0.84, 1.08]  min 0 · mean 0.837 · max 1.51
reward/cup_contact: [-0.00338, 0, -0.000488, -0.0219, -0.0806, -0.0328, -0.0303, -0.00882, -0.0168, -0.0287]  min -0.106 · mean -0.0192 · max 0
reward/cup_hit: [-0.00906, 0, -0.00129, -0.156, -0.435, -0.318, -0.282, -0.127, -0.164, -0.215]  min -0.508 · mean -0.161 · max 0
reward/drop: [0, -0.00134, -0.000854, -0.00159, -0.000366, -0.000732, -0.00122, 0, -0.000244, -0.000366]  min -0.0216 · mean -0.000543 · max 0
reward/flow_excess: [0, 0, 0, 0, 0, 0, -0.00548, -0.0196, -0.0163, -0.0171]  min -0.0292 · mean -0.00672 · max 0
reward/grasp_rcv: [0.0428, 1.08, 1.04, 1.07, 1.04, 1.03, 1.05, 1.03, 1.01, 1]  min 0.0348 · mean 1.02 · max 1.1
reward/grasp_src: [0.0403, 1.12, 1.11, 1.12, 1.11, 1.09, 1.11, 1.1, 1.08, 1.09]  min 0.0305 · mean 1.09 · max 1.14
reward/hand_flip: [0, 0, 0, 0, 0, 0, 0, -0.0202, -0.196, -0.251]  min -0.262 · mean -0.0552 · max 0
reward/hand_foreign_rcv: [-0.0281, -0.0158, -0.0135, -0.0178, -0.0309, -0.00906, -0.0114, -0.0214, -0.0166, -0.0621]  min -0.111 · mean -0.0221 · max -0.00618
reward/hand_foreign_src: [-0.0285, -0.0076, -0.00889, -0.035, -0.069, -0.0201, -0.0308, -0.0181, -0.0194, -0.022]  min -0.105 · mean -0.0253 · max -0.00473
reward/hand_rate: [-0.0308, -0.0238, -0.024, -0.0253, -0.0243, -0.0235, -0.0228, -0.0249, -0.0605, -0.0708]  min -0.0735 · mean -0.034 · max -0.0211
reward/hold_rcv: [0, 0, 0, 0, 0, 0, 0.0271, 0.169, 0.199, 0.205]  min 0 · mean 0.0717 · max 0.232
reward/hold_src: [0, 0, 0, 0, 0, 0, 0.0778, 0.527, 0.629, 0.641]  min 0 · mean 0.22 · max 0.693
reward/home_rcv: [0, 0.0331, 0.0863, 0.117, 0.147, 0.185, 0.186, 0.174, 0.167, 0.155]  min 0 · mean 0.133 · max 0.207
reward/lift_rcv: [0, 0.362, 0.977, 1.04, 1.09, 1.14, 1.16, 1.09, 1.18, 1.19]  min 0 · mean 0.984 · max 1.22
reward/lift_src: [0.000366, 3.04, 2.96, 3.02, 2.97, 3.07, 3.15, 3.13, 3.14, 3.16]  min 0.000366 · mean 2.94 · max 3.2
reward/nested: [0, 0, 0, 0, -0.000122, 0, 0, 0, 0, 0]  min -0.000732 · mean -2.28e-05 · max 0
reward/overtilt: [0, 0, 0, 0, 0, 0, -0.00124, -0.0514, -0.069, -0.0719]  min -0.1 · mean -0.0227 · max 0
reward/palm_rate_rcv: [-0.0625, -0.0433, -0.0324, -0.0289, -0.0254, -0.0225, -0.0204, -0.0225, -0.0586, -0.0662]  min -0.0719 · mean -0.0373 · max -0.0171
reward/palm_rate_src: [-0.0605, -0.0348, -0.0289, -0.024, -0.0181, -0.0127, -0.00936, -0.00952, -0.0231, -0.0255]  min -0.0623 · mean -0.0226 · max -0.00767
reward/palm_sat_rcv: [0, 0, 0, 0, 0, 0, 0, -0.00326, -0.033, -0.0518]  min -0.0529 · mean -0.0111 · max 0
reward/palm_sat_src: [0, 0, 0, 0, 0, 0, 0, -0.00662, -0.069, -0.0966]  min -0.0979 · mean -0.0212 · max 0
reward/posture_rcv: [0, 0.517, 0.392, 0.389, 0.353, 0.297, 0.311, 0.294, 0.294, 0.296]  min 0 · mean 0.345 · max 0.523
reward/posture_src: [0, 0.748, 0.841, 0.878, 0.887, 0.708, 0.869, 0.898, 0.876, 0.984]  min 0 · mean 0.856 · max 1.13
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0.0082, 0.0234, 0.0223, 0.0252]  min -0.0304 · mean 0.00917 · max 0.0375
reward/pour_dir: [0, 0.000134, 0.0394, 0.679, 0.999, 1.15, 1.21, 1.17, 1.18, 1.16]  min 0 · mean 0.808 · max 1.22
reward/pour_dir_pen: [0, 0, -0.00842, -0.0371, -0.013, -0.00275, -0.000769, -0.00108, -0.00049, -0.00074]  min -0.0863 · mean -0.00783 · max 0
reward/pour_pose: [0, 0, 0, 0, 0, 1.56e-05, 0.328, 1.39, 1.57, 1.37]  min 0 · mean 0.546 · max 1.63
reward/pour_side: [0, 0.00199, 0.179, 0.773, 0.977, 0.798, 0.722, 0.602, 0.476, 0.514]  min 0 · mean 0.516 · max 1.02
reward/prelift_near: [0, 0, 0, -0.000249, -0.000529, -2.69e-05, -7.33e-05, -6.86e-05, -0.00022, 0]  min -0.0107 · mean -0.000234 · max 0
reward/premature_latch: [0, -0.000244, -0.000244, 0, 0, -0.00171, -0.0139, -0.0413, -0.053, -0.0688]  min -0.0898 · mean -0.0208 · max 0
reward/rcv_motion: [0, 0, 0, 0, 0, -0.0221, -0.231, -0.313, -0.341, -0.336]  min -0.357 · mean -0.138 · max 0
reward/rcv_still: [0, -0.052, -0.105, -0.197, -0.189, -0.203, -0.288, -0.313, -0.343, -0.344]  min -0.364 · mean -0.217 · max 0
reward/reach_rcv: [0.253, 0.297, 0.293, 0.295, 0.296, 0.296, 0.297, 0.296, 0.296, 0.293]  min 0.238 · mean 0.295 · max 0.297
reward/reach_src: [0.236, 0.298, 0.297, 0.297, 0.297, 0.297, 0.297, 0.297, 0.297, 0.297]  min 0.234 · mean 0.297 · max 0.298
reward/rot_speed: [-3.35e-05, -0.0404, -0.0396, -0.0582, -0.0499, -0.0485, -0.0472, -0.0606, -0.0603, -0.0558]  min -0.0865 · mean -0.0517 · max -3.35e-05
reward/side_margin: [0, -0.0007, -0.0227, -0.00989, -0.0251, -0.00505, -0.00361, -0.0131, -0.00203, -0.00552]  min -0.0413 · mean -0.00893 · max 0
reward/spill_delta: [0, -0.00203, -0.000326, 0, -0.00136, 0, -0.00276, -0.00399, 0, 0]  min -0.0148 · mean -0.00204 · max 0
reward/src_speed: [0, -0.0204, -0.092, -0.15, -0.115, -0.0791, -0.0606, -0.0933, -0.0927, -0.0856]  min -0.178 · mean -0.0832 · max 0
reward/src_spin: [0, -0.021, -0.0372, -0.0803, -0.0668, -0.062, -0.0634, -0.0895, -0.0917, -0.0854]  min -0.107 · mean -0.063 · max 0
reward/success: [0, 0, 0, 0, 0, 0, 0.0604, 4.72, 5.12, 4.04]  min 0 · mean 1.71 · max 7.98
reward/tilt_main: [0, 0, 0, 0.0511, 1.7, 3.66, 5.08, 4.97, 5.27, 4.71]  min 0 · mean 2.82 · max 5.56
reward/tilt_pre: [0, 0, 0.000135, 0.0444, 0.419, 0.829, 0.808, 0.384, 0.351, 0.46]  min 0 · mean 0.345 · max 0.874
reward/tip_only_rcv: [0, -0.551, -0.511, -0.521, -0.402, -0.332, -0.337, -0.366, -0.322, -0.322]  min -0.736 · mean -0.399 · max 0
reward/tip_only_src: [0, -0.63, -0.321, -0.28, -0.246, -0.222, -0.281, -0.37, -0.376, -0.42]  min -0.76 · mean -0.362 · max 0
reward/total: [0.961, 7.54, 8.99, 11.5, 15.5, 19.1, 21.4, 28.3, 28.9, 26.8]  min 0.209 · mean 18.5 · max 31.9
reward/transferred: [0, 0, 0, 0, 0, 0, 0.264, 2.79, 3.22, 3.29]  min 0 · mean 1.12 · max 3.37
reward/upright_rcv: [0, -0.107, -0.0595, -0.153, -0.099, -0.0538, -0.0476, -0.0406, -0.0522, -0.128]  min -0.183 · mean -0.0733 · max 0
reward/upright_src: [-0.00061, -0.0761, -0.0651, -0.0308, -0.00737, -0.0107, -0.00789, -0.0282, -0.0261, -0.0491]  min -0.279 · mean -0.0411 · max -0.00061
rewards/step: [52.8, 6.39e+03, 7.91e+03, 1.01e+04, 1.34e+04, 1.68e+04, 1.92e+04, 2.54e+04, 2.52e+04, 2.41e+04]  min 34.9 · mean 1.63e+04 · max 2.89e+04
task/aim_dist: [0.247, 0.65, 0.397, 0.344, 0.167, 0.148, 0.125, 0.122, 0.12, 0.109]  min 0.106 · mean 0.229 · max 1.45
task/cup_collision_rate: [0.00171, 0, 0.000244, 0.012, 0.043, 0.0173, 0.0159, 0.00464, 0.00854, 0.0151]  min 0 · mean 0.01 · max 0.0557
task/cups_center_dist: [0.283, 0.65, 0.404, 0.354, 0.189, 0.197, 0.205, 0.227, 0.234, 0.222]  min 0.181 · mean 0.285 · max 1.44
task/episode_success: [0, 0, 0, 0, 0, 0, 0.00366, 0.297, 0.324, 0.267]  min 0 · mean 0.109 · max 0.492
task/nested_rate: [0, 0, 0, 0, 0.000244, 0, 0, 0, 0, 0]  min 0 · mean 4.56e-05 · max 0.00146
task/rcv_cup_lift: [0.003, 0.101, 0.108, 0.233, 0.0987, 0.0934, 0.0869, 0.0832, 0.0916, 0.112]  min -0.000336 · mean 0.0901 · max 0.324
task/rcv_grasped: [0, 0.851, 0.817, 0.816, 0.72, 0.727, 0.84, 0.844, 0.834, 0.84]  min 0 · mean 0.803 · max 0.863
task/rcv_hand_foreign_rate: [0.171, 0.032, 0.0305, 0.0332, 0.0625, 0.0232, 0.0239, 0.0334, 0.0317, 0.0962]  min 0.0156 · mean 0.0426 · max 0.24
task/src_cup_lift: [0.00245, 0.2, 0.191, 0.2, 0.191, 0.205, 0.215, 0.246, 0.26, 0.266]  min 0.000216 · mean 0.212 · max 0.313
task/src_grasped: [0, 0.882, 0.877, 0.882, 0.876, 0.867, 0.884, 0.876, 0.867, 0.882]  min 0 · mean 0.867 · max 0.901
task/src_hand_foreign_rate: [0.169, 0.0205, 0.0234, 0.0762, 0.125, 0.0554, 0.0691, 0.051, 0.0435, 0.0513]  min 0.0159 · mean 0.0592 · max 0.234
task/src_tilt_deg: [12.1, 8.72, 8.07, 12.8, 17.6, 34.9, 55.6, 79.5, 84.6, 82.8]  min 5.81 · mean 42.8 · max 86.3
task/success_now: [0, 0, 0, 0, 0, 0, 0.00439, 0.305, 0.33, 0.273]  min 0 · mean 0.111 · max 0.5

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
NEW TASK (round 25). The operator changed the goal: pour a REQUESTED AMOUNT, then bring the source cup
back upright and keep holding both cups. In the real setup the robot will pour into several cups in a
row, so "tip the cup over and leave it there" is no longer an acceptable ending. The env was changed
for this (commit c6708ec7) and the success rule is env-judged, not reward-judged:

  success = |poured_beads - pour_target_beads| <= 1
            AND source cup tilt <= 20 deg
            AND both cups still grasped
            AND (old safety conditions: spill <= 0.40, cup xy < 0.20, not nested, no premature-tilt
                 latch, no cup-cup collision, pour direction ok, receiver upright <= 20 deg)

  New ctx fields: pour_target_beads (how many beads to transfer this episode, sampled per episode),
  pour_target_frac (that count / active beads, also given to the actor as one observation channel),
  bead_active_count (how many beads exist in the source cup this episode).
  Observation is now 224 (actor) / 294 (critic); the policy therefore KNOWS the requested amount.

MEASURED, t2r_i24 best checkpoint, deterministic replay at ADR 30, 64 envs, 900 steps
(the run the operator accepted on video: palm on the upper side of the tilted cup, fingers supporting
from below, no tremor in the pouring arm):

  transfer success (>= half moved)      0.938
  env success (all conditions)          0.719
  in_target median                      1.00      spill median 0.00, p90 0.111
  peak source tilt median               116 deg
  grip drift from the lift reference    1.08 cm (20-50 deg) / 1.46 (50-90) / 1.56 (90-160)
  force-share difference                0.27 / 0.25 / 0.25
  force ratio vs lift                   2.40x / 2.56x / 2.46x
  -> the lift-grasp-keeping design of iter_24 WORKED and must be kept. i23 for comparison:
     3.6 cm drift, 0.51 share difference, 5.0x force.

  corr(fill level, peak tilt)          -0.05 at ADR 30, -0.19 at ADR 0
  -> NO fill-dependent tilt. The policy rotates to the same angle whatever the cup holds.
     Structural cause measured today with a dedicated probe: with 30 mm beads the fill measure
     saturates (12 active beads still read 0.94, 20 read 1.00), so "a less full cup" barely exists.

  source arm action flip rate           0.294      receiver arm 0.445
  source arm joint-speed RMS            0.252      receiver arm 0.341
  -> the RECEIVER arm trembles ~1.5x more than the pouring arm. The operator saw this on video.

  source cup y travel                   -0.162 -> +0.156 (peak +0.193)
  receiver cup y                        +0.161 -> +0.222
  -> the whole pour happens on the left side. The operator requires the work to stay near the
     centre line (y ~ 0) because the real policy will be driven from the head camera view.
     The receiver side already has side_margin / home_rcv terms; the source side has no such term.

  of 60 episodes that transferred beads, 14 were not env successes:
  premature-tilt latch 21 %, cup xy distance over the limit 14 %, receiver tilt was fine (6 deg).

BEAD SIZE — CHANGED TO 24 mm (measured today, spawn count still 20 so the physics load is the same).

Why the old 30 mm beads had to go: with 30 mm the fill measure saturates (12 active beads read 0.94,
20 read 1.00), so "a less full cup" never existed and the policy had nothing to adapt to.

A second defect was found and fixed while measuring (commit addace1c): the cup is tapered - the inner
radius is 26.7 mm at the bottom and 42.6 mm at the rim - but the bead spawn layout sized every ring
from the RIM radius, so bottom-layer beads were spawned outside the wall and PhysX ejected them while
they settled, with no hand touching the cup. Raising solver iterations (8/16 -> 32), the cup SDF
resolution (64 -> 128 -> 256) and the bead contact offset (0 -> 2 mm -> 5 mm) all changed nothing;
spawning inside the tapered wall fixed it completely.

  beads still in the cup after settling (32 envs, no hand contact):
    diameter   before fix   after fix
     24 mm       0.805        1.000
     27 mm       0.938        1.000
     30 mm       0.998        0.905   <- 20 x 30 mm no longer fit in a lattice inside this cup

  24 mm after the fix (16 envs, scripted pour):
    active beads   fill signal   first bead out   80 % out
        20            0.853          90.6 deg      95.9 deg
        14            0.688          97.6 deg     102.0 deg
         8            0.483          91.9 deg      96.2 deg
    27 mm at 20 active reads fill 1.00 again, so it does not solve the saturation.

  Applied in the ENVIRONMENT (this is where asset-specific numbers belong, and `ctx` carries them
  into the reward): bead_diameter_m 0.030 -> 0.024 (mass follows the density, 15.6 g -> 8.0 g),
  bead_active_range (6, 20) -> (8, 20) so one bead is at most 12.5 % of the request,
  premature_release_full_deg 72 -> 90 and premature_release_span_deg 34 -> 8.
  ★These numbers moved once already and will move again with another cup or bead. The previous
  reward hardcoded the old ones (a pour target at 72 deg and an over-tilt charge from 86 deg) and
  would now charge the policy for reaching the angle where beads actually leave. Read angles from
  `ctx`, or reward the outcome instead.

REQUIRED OF THE NEW REWARD (operator):
  1. Income for approaching pour_target_beads, penalty for going past it. Pouring everything must no
     longer be the safe choice.
  2. Pouring must be driven by the FLOW, not by a prescribed angle. The operator's rule for this
     round: "a reward must be general and simple, or neither the learning nor the policy
     generalises". So do not write a target tilt in degrees; pay for beads actually leaving the
     source cup and arriving in the receiver, and let the policy find the angle - which will differ
     with the amount left in the cup, and would differ again with another cup or another bead size.
  3. After the target amount is reached, bringing the cup back upright and holding both cups must pay.
  4. Keep the iter_24 lift-grasp coupling exactly as it is: the reference is the hand pose, force
     share and total force AT LIFT, never an absolute force target.
  5. The source side needs a centre-line term (work near y ~ 0), symmetric to the receiver side.
  6. Stronger tremor penalty on the RECEIVER arm than on the source arm.
  7. Control penalties stay ramped on adr_progress (fresh run starts with them off).
