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

FULL-JOINT CONTROL (no Cartesian/palm controller, no geometric fabric). The action space is a \
normalized `Box(-1, 1, (26,), float32)`, 13 numbers per arm:
  actions[0:7]   = source ARM joint targets (joints 1..7, shoulder to wrist), ABSOLUTE and relative to the \
start pose: a = 0 is the start (home) pose of that joint, a = +1 its upper joint limit, a = -1 its lower joint limit \
(piecewise linear on each side of home). The joint target moves toward that point through an exponential moving \
average (alpha 0.1) and at most 0.3 rad/s; the PD controller gets a position target only. Holding a pose means \
repeating the same action; a = 0 brings the arm back to its start pose. Because a = 0 is the home pose, a cost on \
the size of the arm action would pull the arm back to its start pose and must not be used; a cost on the change of \
the action between steps (smoothness) is fine.
  actions[7:13]  = source HAND joint targets, one per driven joint, in this order: thumb abduction, \
thumb flexion, index, middle, ring, pinky. Each value is an ABSOLUTE target over the range between the \
open pose and the nominal grasp pose of that joint (a = -1 -> the lower of the two, a = +1 -> the higher), \
smoothed by an EMA (alpha 0.1) and limited to the vendor speed (full joint range in 1 s). The fingers cannot \
be driven past the grasp pose. Like the real RH56F1 firmware force stop, a finger whose links touch its \
own cup with more than 1 N does not close any further (opening is always allowed). There is no coupling \
between fingers and no proximity gate. In action units the open hand is thumb abduction +1, every other \
drive -1; the grasp pose is thumb abduction -1, every other drive +1; a = 0 is half way (closure 0.5).
  actions[13:20] = receiver ARM joint targets (same law as the source arm)
  actions[20:26] = receiver HAND joint targets (same law as the source hand)
`*_hand_closure` is measured from the joint positions: 0 = open pose, 1 = nominal grasp pose (clipped). \
Because hand commands are absolute joint targets, |a| close to 1 on a hand drive is a normal command \
(fully open or fully at the grasp pose), not a saturation fault; on an arm drive |a| close to 1 means \
"go to that joint limit".
The episode is terminated (reset) when the physics breaks: a cup moving faster than 3 m/s or spinning faster \
than 30 rad/s, any hand contact above 200 N for 2 steps (including the hand hitting the robot's own body), a cup \
more than 0.5 m from its spawn, a cup below the table, or a finger coupling failure. A terminated episode simply \
stops paying; no reward term is needed for it.
The "cups" are slim shaker bodies about 6 cm in diameter and 11 cm tall. The target grasp is an \
ENVELOPE grasp from the side, with the hand level with the table: the palm plane is upright and faces \
the cup, the finger length axis is horizontal (parallel to the table), and the cup axis lies in front of \
the palm between the palm normal and the finger direction, so that curling the fingers wraps them around \
the cup. Measured on this robot in an earlier run in which the left hand wrapped five fingers around the \
cup and lifted it (median over 34k steps): palm about 1.5 cup radii from the cup axis (1.4 cm from the \
surface), palm about 0.15 x mouth height above the cup origin, palm normal and finger axis both \
horizontal (z components -0.01 and -0.11), and the horizontal direction from the palm to the cup axis = \
0.67 x palm normal + 0.73 x finger axis; five fingers in contact, no palm contact. \
`*_palm_axes[:, 0:3]` is the palm normal (pointing to the grasp side, the thumb tip lies about 7 cm along \
it) and `*_palm_axes[:, 3:6]` is the finger length axis (pointing from the palm to the fingertips). \
The cups are light, so a finger or palm brushing the cup easily knocks it over: approach slowly and \
close the fingers only when the cup is inside the hand.

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
    src_palm_lin_vel: torch.Tensor      # (N,3) 손바닥 선속도 [m/s] (env-local 축) — 차분형(진척) 보상용
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
    rcv_palm_lin_vel: torch.Tensor      # (N,3) [m/s]
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

I want it to fulfil the following task: Using both arms, grasp each cup with an envelope grasp from the side, the hand level with the table (palm upright facing the cup, fingers horizontal and curling around it): the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the mouth of the source cup over the mouth of the receiver cup, and tilt the source cup so that the beads pour into the receiver cup. Keep the receiver cup upright (tilted no more than 20 degrees), do not drop either cup, and spill as few beads as possible. The task is finished when at least 80% of the beads are inside the receiver cup, at most 25% have been spilled, both cups are still held and the receiver cup is upright, and this holds for 20 control steps.
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


def _sat(x, lo: float = 0.0, hi: float = 1.0):
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def _sabs(x: torch.Tensor, eps: float) -> torch.Tensor:
    """매끄러운 |x| (0 에서 꺾이지 않음)."""
    return torch.sqrt(x * x + eps * eps) - eps


def _close_in(d, r):
    """손끝-바깥 벽 거리 성형 (이전 코드 그대로, 척도는 컵 반지름 배수)."""
    return 0.5 * torch.exp(-d / (1.2 * r)) + 0.5 * torch.exp(-d / (0.22 * r))


def _limit_cost(margin: torch.Tensor) -> torch.Tensor:
    """관절 한계 근접 연속 비용 (관절 번호 없음, 유계 [0,1)). 이전 코드 그대로 (쥔 뒤 joint_margin 용)."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    return torch.tanh(c.sum(dim=-1))


def _margin_fac(margin: torch.Tensor) -> torch.Tensor:
    """iter_02 [5-6 1] 쥐기 전 접근 단계 수입에 곱하는 관절 여유 인자 [0.5, 1] (관절 번호 없음, 유계, 바닥 0.5).
    관절별 c = (1 - sat(여유 / 0.25))^2, q = prod_j (1 - 0.5 c_j), 인자 = 0.5 + 0.5 q.
    참 궤적(여유 0.5)에서는 정확히 1. 이전 코드 그대로."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    q = torch.prod(1.0 - 0.5 * c, dim=-1)
    return 0.5 + 0.5 * q


def _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r):
    """컵 좌표(축 = cup_up) 손끝-바깥 벽 거리. 입구 안 손끝은 '먼 것'(2 r)으로 친다 (f07 교훈).
    입구 안 판정 높이 하한 = 입구 - 1 r (_surround 와 같은 값). 이전 코드 그대로."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)                                   # (N,F)
    rad = (rel - h_tip.unsqueeze(-1) * up).norm(dim=-1)              # (N,F)
    gap_raw = torch.clamp(rad - r, min=0.0)
    w_rad = _sat((0.9 * r - rad) / (0.3 * r))
    w_h = _sat((h_tip - (mouth_z - r)) / (0.5 * r)) * _sat((mouth_z + 1.5 * r - h_tip) / (0.5 * r))
    w_in = w_rad * w_h                                               # 1 = 입구 안
    gap = (1.0 - w_in) * gap_raw + w_in * (2.0 * r)
    h_t = h_tip[:, 0]
    h_err_t = torch.clamp(-h_t, min=0.0) + torch.clamp(h_t - 0.5 * mouth_z, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2 + 1e-12)      # (N,)
    h_o = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * mouth_z - h_o, min=0.0) + torch.clamp(h_o - 0.8 * mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2 + 1e-12)       # (N,F-1)
    return d_thumb, d_oth, 1.0 - w_in


def _surround(tips_pos, palm_pos, cup_pos, cup_up, mouth_z, r):
    """iter_01 [5-5 1] 손가락이 컵을 둘러쌌다는 기하 [0,1]. 이전 코드 그대로.
    = (서로 반대쪽) x (네 손가락 끝이 벽 가까이) x (엄지 끝이 벽 가까이)."""
    EPS = 1e-6
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h = (rel * up).sum(dim=-1)                                       # (N,F)
    rad_v = rel - h.unsqueeze(-1) * up                               # (N,F,3)
    rad = rad_v.norm(dim=-1)                                         # (N,F)
    w_in = _sat((0.9 * r - rad) / (0.3 * r)) * _sat((h - (mouth_z - r)) / (0.5 * r))
    gap = (1.0 - w_in) * torch.clamp(rad - r, min=0.0) + w_in * (2.0 * r)
    herr = torch.clamp(-0.4 * mouth_z - h, min=0.0) + torch.clamp(h - 0.8 * mouth_z, min=0.0)
    d = torch.sqrt(gap * gap + herr * herr + 1e-12)
    cl = torch.exp(-(d / (0.4 * r)) ** 2)                            # (N,F)
    cl_t = cl[:, 0]
    cl_f = cl[:, 1:].mean(dim=1)

    u = rad_v / (rad.unsqueeze(-1) + EPS)
    p_rel = palm_pos - cup_pos
    p_h = p_rel - (p_rel * cup_up).sum(dim=-1, keepdim=True) * cup_up
    u_p = p_h / (p_h.norm(dim=-1, keepdim=True) + EPS)
    near = u_p + u[:, 0, :]
    near_hat = near / (near.norm(dim=-1, keepdim=True) + EPS)
    cos_f = (u[:, 1:, :] * near_hat.unsqueeze(1)).sum(dim=-1).mean(dim=1)
    opp = _sat((0.5 - cos_f) / 0.8)
    return opp * cl_f * cl_t


def _hand_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                arm_qd, arm_limit_margin, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                other_cup_pos, r, mouth_z, lift_target, lift_relief):
    """팔별 무가중 항. 두 팔 같은 식 (한 팔에만 붙는 부호·상수 없음). 모든 거리 척도는 컵 반지름 배수.
    iter_03 변경 (DESIGN_BRIEF 5-7, 한 가설·두 변경):
      [5-7 1] 손 방향 점수 level 에 '손바닥이 위를 보는' 쪽만 깎는 비대칭 인자(바닥 0.2)를 곱하고,
              접근 수입에 거리와 무관한 손 방향 몫(0.2 x level)을 더한다 -> 컵에서 먼 시작 자세에서도 방향 기울기.
      [5-7 2] 판정 전 접촉 사전 수입 pre 는 손바닥 잡는 자리(pocket) x 수평 손(level) 에서만 (같은 단계 인자).
    나머지는 iter_02 c1 그대로."""
    EPS = 1e-6
    dt = cup_pos.dtype
    g_raw = grasped.to(dt)

    # ---- 손바닥 기하 (컵 좌표) : K7 자리 = 축에서 1.5 r, 높이 원점 + 0.15 x 입구
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    rad_vec = to_cup - h_c * cup_up
    rad_p = rad_vec.norm(dim=-1)
    dh = -h_c.squeeze(-1) - 0.15 * mouth_z
    dr = rad_p - 1.5 * r
    n_dot = (n_dir * cup_up).sum(dim=-1)
    f_dot = (f_dir * cup_up).sum(dim=-1)
    e_in = rad_vec / (rad_p.unsqueeze(-1) + EPS)
    tgt = 0.67 * n_dir + 0.73 * f_dir                                 # K7 손바닥 -> 컵 축 방향
    tgt_h = tgt - (tgt * cup_up).sum(dim=-1, keepdim=True) * cup_up
    tgt_hat = tgt_h / (tgt_h.norm(dim=-1, keepdim=True) + EPS)
    cos_env = (e_in * tgt_hat).sum(dim=-1)

    # ---- 손끝 기하 / 바깥 접촉 (입구 안 제외)
    d_thumb, d_oth, out_w = _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r)
    f_raw = torch.clamp(finger_force, min=0.0)
    f_out = f_raw * out_w
    touch = torch.tanh(f_out / 0.7)                                   # (N,F) 손가락별 접촉 점수
    touch_o = touch[:, 1:]
    oth_ok = _sat(touch_o.max(dim=1).values / math.tanh(1.0 / 0.7))
    g_eff = g_raw * out_w[:, 0] * oth_ok                              # 쥠 판정 x 바깥 접촉 (같은 단계 인자)
    free = 1.0 - g_eff                                                # 쥐기 전 게이트

    # ---- iter_02 [5-6 1] 쥐기 전 관절 여유 인자 (접근 단계 안에서만; 쥔 뒤 1)
    j_pre = 1.0 - free * (1.0 - _margin_fac(arm_limit_margin))

    # ---- 손 방향 점수 (수평 손)
    s_fl = 0.5 * torch.exp(-(f_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(f_dot))
    s_nl = 0.5 * torch.exp(-(n_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(n_dot))
    #   iter_03 [5-7 1] 뒤집힌 손(법선 z > 0 쪽으로 크게)은 아래로 기운 손보다 싸게: 위쪽으로만 깎는 인자.
    #   문턱 0.25 / 폭 0.45 는 쥔 뒤 roll 비용과 같은 값. 참 궤적(법선 z -0.01, K7)에서는 1 이라 불변.
    #   바닥 있는 곱은 (0.4 + 0.6 s_nl) 와 이것 두 개까지 (5-0).
    #   f02 오른손 표본(법선 z +0.75, 손가락 z -0.65): level 0.086 -> 0.017, 같은 기울기로 아래로 기운 손 0.086 (계산).
    up_ok = 1.0 - _sat((n_dot - 0.25) / 0.45)
    level = s_fl * (0.4 + 0.6 * s_nl) * (0.2 + 0.8 * up_ok)
    level_fac = 0.3 + 0.7 * level                                     # 바닥 0.3 인 수평 손 인자

    # ---- 1 접근 거리 (높이 x2 로 먼저 내려오기, 앞·중심선 쪽 가산은 쥐기 전에만)
    far_side = torch.clamp(palm_pos[:, 0] - cup_pos[:, 0], min=0.0) * free
    to_other = other_cup_pos[:, :2] - cup_pos[:, :2]
    e_other = to_other / (to_other.norm(dim=-1, keepdim=True) + EPS)
    lat = ((palm_pos[:, :2] - cup_pos[:, :2]) * e_other).sum(dim=-1)
    cross = torch.clamp(lat - 0.5 * r, min=0.0) * free
    dr_eff = torch.clamp(dr, min=0.0) + 3.0 * torch.clamp(-dr, min=0.0)
    dh_abs = _sabs(dh, 0.1 * r)
    side_d = dr_eff + far_side + cross
    d_app = side_d + 2.0 * dh_abs
    reach_far = _sat(1.0 - d_app / (30.0 * r))
    reach_near = (1.0 - torch.tanh(d_app / (2.0 * r))) * level_fac
    reach = 0.5 * reach_far + 0.5 * reach_near

    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    near_w = torch.exp(-(d_app / (4.0 * r)) ** 2)
    orient = level * (0.4 + 0.6 * near_w * dir_env)

    # ---- 손 안 판정: 손바닥 자리(pocket) x 손가락 둘러싸기(surround) (iter_01 [5-5 1])
    pocket = _sat((3.2 * r - rad_p) / r) * torch.exp(-(dh / (2.0 * r)) ** 2) * _sat((cos_env + 0.3) / 0.6)
    surround = _surround(tips_pos, palm_pos, cup_pos, cup_up, mouth_z, r)
    in_hand = _sat(pocket / 0.4) * surround                           # 같은 단계(컵이 손 안) 인자끼리의 곱
    release = torch.maximum(_sat(in_hand / 0.5), g_eff)

    # ---- 편 손 인자 (iter_01 [5-5 1]) 그대로
    shut = _sat((hand_closure - 0.1) / 0.4)
    k_open = 0.15 + 0.75 * (1.0 - near_w)
    open_fac = release + (1.0 - release) * (1.0 - k_open * shut)
    #   iter_03 [5-7 1] 거리와 무관한 손 방향 몫 0.2 x level (편 손 인자·관절 여유 인자 안, 접근 수입의 일부).
    #   컵에서 먼 시작 자세(near_w ~ 0, reach_near ~ 0)의 방향 계수 = W_APP x (0.35 x 0.4 + 0.2) = 0.68
    #   (iter_02 는 0.28 -> [61] 수평 - 시작 자세 +0.18). 시작 자세 level 은 iter_02 측정치로 역산한 약 0.36 (가설)
    #   -> 수평 - 시작 자세 약 +0.44, 수평 - 뒤집힘 약 +0.67 (계산, 목표 +0.3 / +0.5).
    approach = j_pre * open_fac * (0.65 * reach + 0.35 * orient + 0.2 * level)

    # ---- 1' 먼저 내려오기: 잡는 높이 점수 (편 손 인자 + iter_02 관절 여유 인자)
    low = j_pre * open_fac * (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 : 손끝별 바깥 벽 거리 볼록 평균(제곱) + 손가락이 둘러쌌을 때만 닫기
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    wrap = j_pre * part * part * (0.15 + 0.85 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)
    close = j_pre * in_hand * level_fac * _sat(hand_closure / 0.7)

    # ---- 3 파지 : 판정 뒤 계단(>= 0.5) + 판정 전 접촉 사전 수입(<= 0.4)
    f_t = f_out[:, 0]
    f_o = f_out[:, 1:].sum(dim=1)
    pre_raw = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    #   iter_03 [5-7 2] 사전 수입은 손바닥이 잡는 자리(pocket) 가까이 x 손이 수평(level) 일 때만 (같은 단계 인자).
    #   K7 자리·수평 손(level >= 0.6)이면 1 이라 참 궤적의 사전 수입은 그대로.
    #   f02 뒤집힌 손 새끼 밀기(1.7 N): iter_02 W_GRASP x 0.14 = 0.42 -> level 0.017 이면 0.01 이하 (계산).
    pre_site = _sat(pocket / 0.4) * _sat(level / 0.6)
    pre = pre_raw * pre_site
    part_o = touch_o.mean(dim=1)
    post = 0.5 + part_o * part_o * part_o                             # 세제곱: 5지째 몫이 가장 크다
    grasp = g_eff * post + (1.0 - g_eff) * pre

    # ---- 4 인벨롭 모양 점수 + 손바닥 굴림 비용
    s_h = torch.exp(-(dh / (0.6 * r)) ** 2)
    s_r = torch.exp(-(dr / (0.5 * r)) ** 2)
    s_f = torch.exp(-(f_dot / 0.3) ** 2)
    s_n = torch.exp(-(n_dot / 0.3) ** 2)
    q = 0.30 * s_h + 0.25 * s_r + 0.30 * s_f + 0.15 * s_n
    q2 = q * q
    roll = _sat((torch.abs(n_dot) - 0.25) / 0.45)

    # ---- 5 들기 : 모양과 무관, 처음 몇 cm 가 가장 가파르고 목표 위는 이득 없음
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.2 * r, min=0.0)
    lin = _sat(h_pos / lift_target)
    lin = lin + (1.0 - lin) * lift_relief
    prog = 0.7 * torch.tanh(h_pos / (0.7 * r)) + 0.3 * lin
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    lift = g_eff * calm * prog
    lifted_w = _sat(h_pos / (0.7 * r))

    # ---- 교란
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    topple = (1.0 - g_eff) * torch.tanh(cup_ang_vel[:, :2].norm(dim=-1) / 2.0)
    f_t_raw = f_raw[:, 0]
    f_o_raw = f_raw[:, 1:].sum(dim=1)
    f_o_max = f_raw[:, 1:].max(dim=1).values
    thumb_excess = torch.clamp(f_t_raw - f_o_raw - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o_max - f_t_raw, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    rush = near_w * free * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    jam = _sat((1.3 * r - rad_p) / (0.3 * r))
    disturb = 0.5 * slide + 0.3 * topple + 0.3 * one_sided + 0.2 * palm_push + 0.3 * fling \
        + 0.06 * rush + 0.6 * jam

    return {"approach": approach, "low": low, "wrap": wrap, "close": close, "grasp": grasp,
            "g_eff": g_eff, "q2": q2, "roll": roll, "prog": prog, "lift": lift,
            "lifted_w": lifted_w, "disturb": disturb}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 5번째 사이클 iter_03: 단계별 합 사다리 (곱 사다리 없음). 두 팔 같은 식·같은 무게.
    iter_03 변경 (DESIGN_BRIEF 5-7, 한 가설·두 변경): _hand_terms 의 손 방향 비대칭 인자 + 먼 곳 방향 몫,
    판정 전 접촉 사전 수입의 자리·수평 게이트. 나머지 항·가중은 iter_02 c1 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (iter_02 c1 그대로) ----------------------------------------------------------------
    W_ALIVE = 0.5
    W_APP = 2.0
    W_LOW = 1.2
    W_WRAP = 1.2
    W_CLOSE = 1.3
    W_GRASP = 3.0
    W_SHAPE = 1.5
    W_SHAPE_LIFT = 5.0
    W_ROLL = 1.5
    W_LIFT = 2.5
    W_AIM = 2.5
    W_POUR_H = 4.0
    W_ROT = 23.0          # weight 23 with delivery 30/bead: 회전+배달 몫 약 52 % (추정)
    W_DELIVER = 30.0      # 알당 30, 부호 있는 증분 (i20: 진행형 비대칭 clamp 금지)
    W_SPILL = 40.0
    W_SUCCESS = 20.0
    W_FINISH = 1.5
    W_MARGIN = 0.8        # 쥔 뒤 관절 여유 비용 (5-6: 그대로)
    W_CARRY = W_ROT + W_POUR_H
    TAIL = 0.25
    UPRIGHT_TOL = math.radians(20.0)
    LIM = math.radians(81.0)

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    cup_h = mouth_z - float(ctx.cup_bottom_z)          # 컵 높이 (자산 속성에서 계산)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    active = float(max(int(ctx.num_beads), 1))
    LIFT_RCV = 3.0 * r_cup

    src_g_raw = ctx.src_grasped.to(dt)
    rcv_g_raw = ctx.rcv_grasped.to(dt)
    both_g_raw = src_g_raw * rcv_g_raw
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) --------------------------------------------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    above_rim = _sat(1.0 + dz_lip / r_cup)
    window = lip_gate * above_rim
    theta_pos = torch.clamp(theta, min=0.0)

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    clear_src = torch.clamp(h_rcv, 0.0, LIFT_RCV) + cup_h + r_cup
    lift_relief_src = window * _sat(theta_pos / (0.5 * LIM))

    src = _hand_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                      ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped, ctx.src_arm_qd,
                      ctx.src_arm_limit_margin,
                      ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                      ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, clear_src, lift_relief_src)
    rcv = _hand_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                      ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                      ctx.rcv_arm_limit_margin,
                      ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                      ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, LIFT_RCV, zeros)

    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    alive = W_ALIVE * ones

    # ---- 1~3 접근(+먼저 내려오기) / 감싸기 전 / 파지 ----
    #   iter_03: approach 안에 먼 곳 손 방향 몫 (W_APP x 0.2 x level, 팔당 최대 +0.4) 포함
    approach_src = W_APP * src["approach"] + W_LOW * src["low"]
    approach_rcv = W_APP * rcv["approach"] + W_LOW * rcv["low"]
    prewrap = W_WRAP * (src["wrap"] + rcv["wrap"]) + W_CLOSE * (src["close"] + rcv["close"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- 4 인벨롭 모양 보너스 - 손바닥 굴림 비용 ---------------------------------------------------------
    envelope = src_g * (src["q2"] * (W_SHAPE + W_SHAPE_LIFT * src["prog"]) - W_ROLL * src["roll"]) \
        + rcv_g * (rcv["q2"] * (W_SHAPE + W_SHAPE_LIFT * rcv["prog"]) - W_ROLL * rcv["roll"])

    # ---- 5 들기 (모양 무관) --------------------------------------------------------------------------
    lift_src = W_LIFT * src["lift"]
    lift_rcv = W_LIFT * rcv["lift"]

    # ---- 6 조준 -----------------------------------------------------------------------------------
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))
    h_err = torch.abs(dz_lip - 0.25 * r_cup)
    hz = 0.3 * (1.0 - torch.tanh(h_err / (1.5 * r_cup))) + 0.7 * _sat(1.0 - h_err / (12.0 * r_cup))
    xy = 0.5 * (1.0 - torch.tanh(lip_d / wide)) + 0.5 * (1.0 - torch.tanh(lip_d / two_r))
    aim_carry = W_AIM * both_g * lifted * (0.6 * xy + 0.4 * hz)

    # ---- 비드 장부 --------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac / TAIL)
    spill_frac = _sat(ctx.bead_spill_frac)

    # ---- 7 붓기: 창 안 회전 진척 x want ---------------------------------------------------------------
    rot_prog = _sat(theta_pos / (1.4 * LIM))            # 95 도까지 단조 증가
    tilt_share = want * window * rot_prog
    pour_rot = W_ROT * both_g * tilt_share
    pour_h = W_POUR_H * both_g * tilt_share * hz
    aim = aim_carry + pour_h

    # ---- 7 배달 (주역, 부호 있는 증분) ---------------------------------------------------------------
    pour_delta = W_DELIVER * ctx.d_in_target * active

    # ---- 다 부으면 끝: 마지막 몫 이어받기 + 세우기 / 성공 ---------------------------------
    upright_src = 1.0 - torch.tanh(tilt_src / (0.3 * LIM))
    finish_upright = both_g * (1.0 - want) * (W_CARRY * (1.0 - spill_frac) + W_FINISH * upright_src)
    success_bonus = W_SUCCESS * ctx.success.to(dt)

    # ---- 흘림 / 창 밖 조기 기울임 -----------------------------------------------------------------------
    post = torch.maximum(src_g_raw, src["lifted_w"])
    spill = -W_SPILL * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g_raw * pre_ramp * out_gate

    # ---- 충돌 (실기 안전, 유계) ------------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g_raw) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g_raw) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                                + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- 컵 교란 -------------------------------------------------------------------------------------
    tilt_pen_src = src_g * (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = rcv_g * (1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2))
    cup_disturb = -(src["disturb"] + rcv["disturb"] + 0.3 * tilt_pen_src + 0.3 * tilt_pen_rcv)

    # ---- 리시버 세움·중심선 쪽 --------------------------------------------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- 관절 여유 비용 (쥔 동안, 5-6: 그대로) -------------------------------------------------------------
    joint_margin = -W_MARGIN * (src_g * _limit_cost(ctx.src_arm_limit_margin)
                                + rcv_g * _limit_cost(ctx.rcv_arm_limit_margin))

    # ---- 제어 비용 (cup_grasp 수준; 팔 액션 크기 비용 없음, 손 채널 포화 벌점 없음) -----------------------------
    a = ctx.actions
    pa = ctx.prev_actions
    da_arm = torch.cat([a[:, 0:7] - pa[:, 0:7], a[:, 13:20] - pa[:, 13:20]], dim=-1)
    da_hand = torch.cat([a[:, 7:13] - pa[:, 7:13], a[:, 20:26] - pa[:, 20:26]], dim=-1)
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (
        torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
        + torch.tanh(torch.clamp(qd_rcv.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0))
    rate_hand = 0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)
    ctrl = -(rate_arm + speed_arm + runaway + rate_hand)

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "prewrap": prewrap,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "envelope": envelope,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "aim": aim,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "spill": spill,
        "premature": premature,
        "collision": collision,
        "cup_disturb": cup_disturb,
        "rcv_keep": rcv_keep,
        "joint_margin": joint_margin,
        "ctrl": ctrl,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0, 0, 0, 1.22e-05, 0, 0, 0, 0, 0]  min 0 · mean 1.4e-06 · max 2.44e-05
bead/spill: [0, 0, 0.00094, 0.0015, 0.000208, 4.88e-05, 0, 7.32e-05, 0.000134, 1.22e-05]  min 0 · mean 0.000408 · max 0.00491
done/drop: [0, 0, 0, 0.000244, 0, 0.000244, 0.000244, 0, 0, 0]  min 0 · mean 2.36e-05 · max 0.00171
done/mimic_runaway: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.96e-05 · max 0.000732
episode_lengths/step: [284, 714, 692, 593, 844, 760, 864, 885, 882, 890]  min 200 · mean 804 · max 899
reward/aim: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.05e-09 · max 2.15e-06
reward/alive: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  min 0.5 · mean 0.5 · max 0.5
reward/approach_rcv: [0.269, 0.0402, 0.0453, 0.611, 1.09, 1.36, 1.57, 1.67, 1.56, 1.47]  min 0.0339 · mean 1 · max 1.69
reward/approach_src: [0.273, 0.101, 0.934, 1.39, 1.51, 1.55, 1.61, 1.66, 1.53, 1.64]  min 0.0239 · mean 1.28 · max 1.73
reward/collision: [0, -0.00245, -0.103, -0.0168, -0.101, -0.0652, -0.0385, -0.029, -0.052, -0.0169]  min -0.234 · mean -0.0423 · max 0
reward/ctrl: [-0.0259, -0.0419, -0.0411, -0.0414, -0.0401, -0.0413, -0.0403, -0.0388, -0.0383, -0.038]  min -0.0455 · mean -0.0401 · max -0.0259
reward/cup_disturb: [-0.21, -0.209, -0.303, -0.42, -0.327, -0.396, -0.401, -0.372, -0.369, -0.345]  min -0.496 · mean -0.337 · max -0.153
reward/envelope: [0, 0, 0, 0.00046, -3.05e-05, -0.000959, 0.000137, 0, 5.04e-05, 8.79e-05]  min -0.00795 · mean -0.000148 · max 0.00102
reward/finish_upright: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0, 0, 0.00981, 0.00106, 0.00883, 0.00691, 0.000108, 0.00206, 0.0015]  min 0 · mean 0.00197 · max 0.0421
reward/grasp_src: [0, 3.33e-06, 0.00306, 0.00167, 0.000627, 0.000406, 0.000314, 0.000167, 0.0001, 0.00642]  min 0 · mean 0.00164 · max 0.0187
reward/joint_margin: [0, 0, 0, -5.13e-06, -6.58e-06, -2.68e-05, 0, 0, 0, -0.000208]  min -0.00198 · mean -6.28e-05 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.31e-06 · max 0.000525
reward/lift_src: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 6.15e-07 · max 0.000455
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7e-06 · max 0.00732
reward/pour_rot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.73e-11 · max 1.02e-07
reward/premature: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/prewrap: [4.7e-06, 0.000133, 0.0131, 0.0171, 0.012, 0.014, 0.0232, 0.0104, 0.0138, 0.0205]  min 3.78e-06 · mean 0.0128 · max 0.0362
reward/rcv_keep: [0, 0, 0, 0.0012, 0.000117, 0.00164, 0.000238, 0, -0.000278, 0.000207]  min -0.00146 · mean 0.000172 · max 0.0111
reward/spill: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0488 · mean -0.000206 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [0, 0.388, 1.04, 2.02, 2.63, 2.91, 3.2, 3.38, 3.13, 3.22]  min 0 · mean 2.35 · max 3.44
rewards/step: [82, 271, 568, 1.02e+03, 2.09e+03, 2.08e+03, 2.79e+03, 2.87e+03, 2.75e+03, 2.98e+03]  min 79.7 · mean 1.95e+03 · max 2.98e+03
task/aim_dist: [0.321, 0.32, 0.318, 0.319, 0.319, 0.317, 0.318, 0.316, 0.313, 0.31]  min 0.306 · mean 0.316 · max 0.321
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.97e-06 · max 0.000244
task/cups_center_dist: [0.321, 0.32, 0.32, 0.319, 0.319, 0.318, 0.319, 0.318, 0.316, 0.314]  min 0.311 · mean 0.318 · max 0.321
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.34e-06 · max 0.000244
task/rcv_cup_lift: [4.29e-06, 4.24e-06, 5.3e-06, 0.000439, 0.000366, 0.00042, 0.000179, 0.000359, 0.000941, 0.00123]  min 3.88e-06 · mean 0.000447 · max 0.00175
task/rcv_grasped: [0, 0, 0, 0.00439, 0.000244, 0.00366, 0.000488, 0, 0.000244, 0.000488]  min 0 · mean 0.000657 · max 0.0249
task/rcv_hand_foreign_rate: [0, 0.00293, 0.0161, 0.0198, 0.223, 0.175, 0.123, 0.0505, 0.118, 0.0437]  min 0 · mean 0.0625 · max 0.354
task/src_cup_lift: [3.16e-06, 7.98e-06, 0.000513, 0.000298, 7.16e-05, 0.000195, 0.000524, 0.000681, 0.000108, 8.51e-05]  min -0.000181 · mean 0.000233 · max 0.00139
task/src_grasped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000105 · max 0.00439
task/src_hand_foreign_rate: [0, 0.00415, 0.433, 0.0408, 0.177, 0.0918, 0.02, 0.0466, 0.052, 0.0168]  min 0 · mean 0.0919 · max 0.772
task/src_tilt_deg: [0.0184, 0.0443, 1.99, 0.84, 0.241, 0.708, 1.86, 2.46, 0.39, 0.293]  min 0.0177 · mean 0.87 · max 5.1
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# pour_bi_rh 보상 설계 브리프 (2026-10-01 판 — 5번째 사이클 iter_04, full-joint 제어·팔 절대 목표·컵 스폰 −12 cm·실기 제어 응답)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트에 통째로 들어간다.**

이 트랙은 네 번 리셋했다. 1~12 라운드 `archive_r1-12`, 13~24 `archive_r13-24`, 3번째 사이클 `archive_r25-34`, 4번째 사이클(full-joint) iter_00~09 는 `archive_r35-44`.
지금은 5번째 사이클의 iter_01 이다. iter_00(09-30 사용자 "학습 보상 다시 세팅")은 **단계별 합 사다리**로 새로 짰다(5절). iter_01 은 그 위에서 2-1 절(참 궤적 최적)과 5-5 절 두 가지를 고쳤다. iter_02 는 iter_01 c1r2 위에 5-6 절(쥐기 전 관절 여유)을 더했다. iter_03 은 iter_02 c1 위에 5-7 절(먼 곳 손 방향·접촉 게이트)을 더했다. iter_04 는 iter_03 c1 위에서 5-8 절만 고친다.

## 1. 과제와 고정 조건 (바꿀 수 없다)

- 오른팔(소스, `src_*`)이 비드 20 개가 든 컵을, 왼팔(리시버, `rcv_*`)이 빈 컵을 잡아 든다. 소스를 기울여 비드를 리시버 컵에 옮긴다.
- 성공(env 가 계산, 보상이 재정의 못 함): 담김 ≥ 0.80 · 흘림 ≤ 0.25 · 두 컵 파지 유지 · 받는 컵 ≤ 20° · 20 스텝 유지 · 컵이 포개지지 않음.
- **액션 26 = 팔마다 13** — [팔 관절 절대 목표 7(a=0 시작 자세·±1 관절 한계, 0.3 rad/s 상한 — iter_03 부터), 손 구동 관절 절대 목표 6]. 로봇 설명 절이 정의한다.
- **물리 이상은 env 가 종료한다(iter_03 부터)**: 컵 > 3 m/s·> 30 rad/s, 손 접촉 > 200 N 2 스텝, 컵이 스폰에서 0.5 m 밖, 컵 낙하, mimic 폭주. 종료 벌점 항은 두지 않는다.
- 스폰(컵 원점, env-local): **09-30 부터 x −0.12**(사용자 A, env `object_spawn_dx`) — 소스 (0.262, −0.159, 0.268) · 리시버 (0.258, +0.165, 0.268), ±2 cm. 시작 손바닥 (0.31, ∓0.30, 0.42): 컵에서 수평 약 15 cm·위 약 15 cm. 보상은 컵 위치를 스폰·컵 기준 상대값으로만 쓴다.
- r = `ctx.cup_radius` = 0.028 m. 소스 컵 원점 → 림 0.054 m. 제어 주기 1/60 s, 에피소드 900 스텝.
- **실기 제어 응답(09-30 부터 env, 사용자 "실제 제어 로직으로 학습")**: 손 관절 속도 상한 2.1 rad/s(엄지 굽힘 0.56) · 네 손가락은 0.065 rad 까지만 펴진다(편 손의 `hand_closure` 최소 약 0.05). **명령 지연(손 4 스텝·팔 9~12 스텝)은 iter_01 부터 끈다 — 사용자 10-01 "초반 학습이 느려질 수 있으니 이송 작업이 성공하고 나서 켠다".** 손 액션 a = 0 은 반쯤 쥔 손(닫힘 0.5)이다 — 편 손은 정책이 −1 쪽을 내야 한다.

## 2. 단계 원칙 (사용자, 09-28)

접근 → 손가락 사이로 컵 → 파지 → 들기 → 붓기 순서로 **gradient 가 다음 단계로 이어져야 한다.
파지를 하고 들려고 할 때 보상이 떨어지면 안 된다.** 한 단계 앞으로 가는 것이 그 자리에 머무는 것보다 항상 스텝당 이득이어야 하고,
머물기만 해도 벌리는 자리(주차)가 없어야 한다. 검사는 `reward_gen/pour_bi_rh/stage_ledger.py`(결정론 상태 원장)로 한다.

### 2-1. 참 궤적 최적 (사용자, 10-01)
"왼손이든 오른손이든 보상함수가 무조건 테스크 성공을 보장해야 한다. 에피소드가 진행되는 동안 **손 펴고 접근 → 인벨롭 그립 → 리프트 → 이송**이라는 트레젝토리 참값을 찾을 수 있게."
검사는 `reward_gen/pour_bi_rh/traj_ledger.py`: 두 팔이 900 스텝 참 궤적(A 시작 자세에서 손을 편 채 잡는 높이로 먼저 내리며 손을 수평으로 · B 컵 뒤 12 cm 로 · C 수평 진입해 K7 자리 · D 감싸 닫고 다섯 손가락 1.5 N · E 두 컵 8 cm 들기 · F 소스를 리시버 입구 옆으로 이송 · G 리시버 쪽 105° 기울여 20 알 전부 붓기 · H 세우고 유지)을 따라가는 상태열에서 스텝마다 보상을 잰다(gamma 0.99, 900 뒤 마지막 상태 유지).
- T1 참 궤적 위 한 스텝 하락 ≤ 0.05(배달 사건 수입 제외) — 단계 경계 절벽 없음. **다 부은 뒤(붓기 → 세우기)도 포함.**
- T2 H 전 어느 시각에서든 '참 궤적대로 계속' 할인 가치 ≥ '그 자리에 멈춤' + 1.
- T3 한 팔만 이탈했을 때 분기 시점 할인 가치가 참 궤적보다 2 이상 낮다: 반쯤 쥔 손 접근 · 위에서 덮는 접근 · 12 cm 편 손 주차 · 10 cm 반쯤 쥔 손 주차 · 쥐고 테이블 유지 · 입구 쪽 비스듬한 파지 · 엄지·중지 두 손가락 파지 · 들고 유지 · 이송 뒤 안 붓기 · **80 % 붓고(성공 문턱) 80° 로 남은 비드를 안은 채 유지**.
- T4 접근 구간(A~C) **매 스텝** 편 손(닫힘 0.05)이 반쯤 쥔 손(닫힘 0.5, 손끝 위치는 f10 재생 실측)보다 팔당 스텝당 +0.15 이상.
- T5 들기까지 두 팔 접근·파지 성분이 같다(스텝당 0.05 안).

## 3. 측정으로 확정된 물리·기구학 사실 (제어와 무관)

| # | 사실 | 근거 |
|---|---|---|
| K1 | 컵이 기우는 방향은 손바닥 yaw 에 묶인다((기우는 방위 − palm yaw) 중앙 −4°, SD 19°). | i18 ep800 |
| K2 | 60° 이상 기울 때 기우는 방위 중앙 +68° — 소스가 부을 수 있는 쪽은 자기 앞-왼쪽. | i18 ep800 |
| K3 | 손이 낮으면 손목 관절이 먼저 한계에 닿아 회전이 안 난다. palm z 0.29~0.42 에서 회전 오차 143~151°, 0.69~0.76 에서 10~23°. | 도달성 프로브 |
| K4 | 붓는 각은 각도 처방 없이도 찾는다(결정론 최대 87°·방출 91°). 가득 찬 컵은 첫 알 약 70°, 86° 에 절반. | i18 |
| K5 | 기울이면 주둥이가 r·sinθ 만큼 내려간다 — 높이 조건은 주둥이로 본다. | 자산 |
| K6 | 컵은 한쪽 접촉 0.3~0.4 N 에 넘어진다(바닥 지름 3.8 cm 테이퍼, 0.134 kg). | 프로브 |
| K7 | **인벨롭 파지 기하(이 로봇)**: 왼손이 5 손가락으로 감싸 들어 올린 34k 스텝 중앙 — 손바닥-컵 축 1.5 r(표면에서 1.4 cm), 높이 원점 +0.15 × 입구, 법선·손가락 축 수평(z −0.01·−0.11), 손바닥→컵 축 수평 방향 = 0.67 × 법선 + 0.73 × 손가락 축, 손바닥 접촉 0. | i24 ep500 재생 |

## 4. 3·4번째 사이클에서 확정된 교훈 (f00~f10 = 4번째 사이클 런)

| 라운드 | 사실 | 교훈 |
|---|---|---|
| rh5_f03 (5번째 iter_03 c1, fresh, 지연 끔) | e1000 종료(사용자 "1번"). 학습 곡선: 두 손 컵 10 cm 안 0.92(e450~) · 파지 0/0 · 컵 근처 쥐기 전 닫힘 0.71/0.76(f02 같은 epoch 0.11/0.16) · 쥐기 전 손바닥 수직 성분 0.38/0.27 · 총보상 e650 부터 3.1~3.4 평평. **ep1000 결정론 재생(server GPU1 64 env)**: 두 손 0/64. t0 닫힘 0.04 → **t60(컵 9~10 cm)에 이미 0.70~0.72**. 컵 옆 머묾(t>300): 오른손 1.9 r·높이 0.20 × 입구·법선 z 0.33·손가락 z −0.34·손가락 축이 컵 쪽 −0.05·닫힘 0.69·**엄지 끝 2.9 r 네 손끝과 같은 방위(cos 1.0)**·손 액션(엄지 외전·엄지 굽힘·검지·중지·약지·새끼) −1·+1·−1·+1·+1·−1. 왼손 1.46 r·0.23·법선 z 0.29·손가락 z −0.62·닫힘 0.76·엄지 3.2 r 같은 방위. 접촉 0. iter_03 c1 코드 검사: **[62] 이 머묾 자세에서 닫힘 0.75 → 0.6 → 0.5 수입 변화 0.000(평평), 0.5 → 0.1 은 0.1 당 +0.07~0.11** · **[63] 엄지를 컵 축 둘레로 0° → 90° → 180° 옮겨도 0.000** · 원장 T4b 접근 구간 닫힘 0.5 > 0.7 칸 **−0.296(쥔 쪽이 더 받음)**, 0.3 > 0.5 +0.026 | **편 손 배수 `shut = sat((닫힘 − 0.1)/0.4)` 가 닫힘 0.5 위에서 평평하다** — 반쯤 이상 쥔 손은 조금 펴서는 아무것도 얻지 못한다. 손끝이 벽에 가까워지는 감싸기 전 수입(`wrap`)은 편 손 인자 밖이라 손가락을 말아 벽에 대면 오히려 오른다. 엄지 대향은 손이 둘러싼 뒤(`surround`)에만 보상돼, 쥐기 전에 엄지를 반대쪽으로 옮길 기울기가 0. 원장 T4(0.05 대 0.5 두 점 비교)가 이 평평한 구간을 놓쳤다 → T4b 추가 |
| rh5_f02 (5번째 iter_02 c1, fresh, 지연 끔) | e3000 종료(사용자 "오른손 회복 가능성 없는 거지? 중단"). 학습 곡선: 팔 관절 한계 0.00~0.02(f01 0.12~0.31 → 풀림) · 왼손 파지 e1100 0.09 → e2481 0.87·컵 15~20 cm 들림·인벨롭 보상 4.9 · 오른손 파지 e1400 0.28 최고 뒤 e1600~3000 0 · 총보상 e2350 부터 13 평평. **ep3000 결정론 재생(server GPU1 64 env)**: 왼손 64/64 쥠(t>150 의 96 %)·들림 중앙 16.6 cm·손바닥 1.44 r·높이 0.16 × 입구·법선 z −0.16·손가락 z +0.05·힘 엄지 5.8·검지 4.6·중지 0.0·약지 1.1·새끼 0.9 N. 오른손 0/64 — 컵 옆 2.6 r 에 **손바닥을 위로 뒤집은 채(법선 z +0.75·손가락 z −0.65)** 편 손(액션 전부 −1), 네 손끝은 컵 원점보다 1 × 입구 아래(테이블 높이)·엄지 4.6 r, **새끼만 1.7 N 으로 컵 밑동을 밂**(컵 2 cm 밀림), 팔 관절 여유 모두 15° 이상. 시간순: 두 손 모두 시작 자세 손가락 z −0.42 그대로 t60 에 컵 옆 6~8 cm 도착(손가락 z −0.6) → 왼손은 t120~150 에 손가락을 수평으로(−0.49 → 0.0), 오른손은 손바닥을 위로(법선 z 0.29 → 0.75, j5 26° → 58°·j7 −24° → −59°). iter_02 c1 코드 검사: [60] 같은 컵에서 수평 인벨롭 편 손 +1.42, 뒤집힌 손의 새끼 밀기 접촉 몫 **+0.24/스텝** · [61] 시작 자세(컵 21 cm)에서 손을 수평으로 돌리는 이득 **+0.18**, 수평 − 뒤집힌 손 +0.26 | **보상은 두 손 대칭이고 왼손은 목표 자세(수평 인벨롭·입구 아래·16 cm 들기)에 도달했다 — 사용자 영상 판정 "왼손은 확실한 인벨롭 파지".** 두 손 모두 손가락을 아래로 한 채 컵 옆까지 와서 거기서 손 방향을 고치는데, 컵에서 멀 때 방향 신호(+0.18)가 작아 방향을 늦게 정한다. 컵 옆에서 방향을 고치다 오른손은 뒤집힌 쪽으로 갔고, 그 자세에서 한 손가락 밀기가 판정 전 접촉 수입(+0.24)을 받아 머문다 |
| rh5_f01 (5번째 iter_01 c1r2, fresh, 지연 끔) | e988 종료(사용자 "재진행 — 왼팔도 영상 보니 이상한 자세로 접근"). 학습 곡선: 컵 10 cm 안 0.95/0.95(e250/e550 부터, f00·f10 보다 빠름) · 20 cm 밖 닫힘 0.46/0.11 · 파지 e400 오른손 0.04 뒤 0 · 총보상 e600 부터 3.9 평평. **ep800 결정론 재생(arm4090 64 env)**: 두 손 모두 쥔 스텝 0, 컵 옆 제자리에 머묾 — 오른손 손바닥 축 1.53 r·높이 0.10 × 입구(자리·높이는 인벨롭), 그러나 손바닥 법선 z 0.58(위로 35°)·손가락 축이 컵 방향 0.14(옆으로)·네 손끝 축 2.9 r(컵 지나침)·엄지 끝 높이 1.54 × 입구·엄지와 네 손끝 같은 방위(cos 1.0)·**손목 j6 한계(여유 0.5°) 100 %**. 왼손 손바닥 2.0 r·법선 z 0.78(50°)·엄지 1.84 × 입구·네 손끝 4.3 r·**j3 97 %·j4 94 %·j6 100 % 한계** — 접근하며 j4(팔꿈치) 91° → 15°, j3 −22° → −82°(한계 −90°), j6 −20° → −45°(한계). 오른팔은 j4 91° → 19°, j6 20° → 45°(한계). iter_01 c1r2 코드 결정론 검사 [59]: 같은 팔을 수평 인벨롭 자리로 바꾸면 +1.12/+1.22(이미 이득), 기운 자세 그대로 관절 여유만 0.5 로 바꾸면 **+0.000**(쥐기 전 관절 여유 신호 없음) | **자리·높이·편 손 접근은 iter_01 로 풀렸다.** 수평 인벨롭 자세가 스텝당 +1.1 더 받는데도, 손이 기운 채 손목(j6)이 한계에 붙으면 거기서 벗어나지 못한다. 관절 여유 비용은 쥔 뒤에만(`W_MARGIN × g_eff`) 있어 접근 중에는 팔이 한계로 흘러가도 손해가 0 — 왼팔은 j3·j4·j6 세 관절이 한계. 도달성 프로브(컵 −12 cm)는 수평 인벨롭 자세가 관절 여유 8.7° 이상으로 닿는다고 했으니, 그 팔 자세는 있다 |
| rh5_f00 (5번째 iter_00 c1r2, fresh, 실기 제어 응답 켬) | e753 에 종료(사용자 "접근할 때 손을 안 펴면 파지하기 어렵다" → 1번). 컵 10 cm 안 비율 0.62/0.65(f10 같은 epoch 0.93/0.86) · 손바닥-컵 10.3/10.2 cm · 파지 0.001/0.005 · 들기 0 · **컵에서 20 cm 밖 손 닫힘 0.42/0.52(f10 0.12/0.19)**, 10 cm 안 0.52/0.59 · e150~450 은 컵 옆 15~21 cm 에 머묾. traj_ledger(iter_00 c1r2 코드): 편 손 − 반쯤 쥔 손 스텝당 A +0.094 · B +0.076 · **C −0.377(최소 −1.11, 컵 앞 진입 구간은 반쯤 쥔 손이 더 번다)**, 에피소드 할인 가치 차 +5.8 뿐 · **80 % 붓고 80° 유지가 끝까지 붓고 세우기보다 할인 가치 +1269 이득**(남은 비드가 1 알이라도 있으면 기울임 수입 want = 1) · 다 부은 스텝에 pour_rot −10.6 절벽 | **편 손 인자가 접근 거리 항에만(0.6~1.0) 곱해지고 높이 항(`low`, 가중 1.2)에는 없다** — f10 코드(4번째 iter_09)는 접근+높이 전체에 0.45~1.0. 손 안(pocket) 판정이 손바닥 위치·방향만 보고 손가락이 컵을 둘러쌌는지 안 봐서, 반쯤 쥔 손도 컵 앞에서 닫기 수입(`close`)을 받는다. 네 손가락 끝의 손바닥 기준 위치(f10 ep1800 재생, 쥐기 전): 닫힘 <0.1 손가락 축 10.0·법선 0.3 cm, 0.4~0.6 7.1·2.3 cm, 0.7~0.9 3.8·4.5 cm. 사용자: 참 궤적 최적(2-1)이 보상의 조건 |
| i03 | 표면 거리 접근은 입구 원판을 표면으로 쳐서, 손바닥이 입구 위에 있으면 거리 1.7 cm·접근 수입 0.75 | 접근은 **컵 원점 기준**, 옆에서(수평 성분 가중) |
| i04 | 깔때기 목표가 "잡는 높이보다 4.5 cm 위에서 도착 후 내려감"이라 위에서 끼우는 동작을 가르쳤다, 파지 0 | 손바닥을 먼저 내리고 테이블과 수평으로 접근 |
| i07 | 손을 쥔 채(폐쇄도 0.63) 다가갔다 | 다가가는 동안 손을 펴게 하는 항이 필요 |
| i08 | 상태형 "편 손" 가산이 컵 12 cm 앞 주차 자리를 만들었다(0.13~0.16/스텝) | 편 손 가산도 **차분형**(편 채 다가갈 때만) |
| f00 (4번째 사이클 첫 iter_00, full-joint) | e0~470: 접근 보상(손바닥 속도 차분) −0.001~−0.018, 손바닥-컵 21~43 cm 왕복, 컵 근처 0(i09 같은 epoch 0.41), 손 폐쇄도 0.59~0.89 유지, 팔 제어 벌점 −0.14~−0.19(가장 큰 항), 팔 목표가 관절 한계 2 % 안 0.02→0.17. 결정론 검사: 시작 자세 6 cm 다가감 +0.002, 15→6 cm −0.122, 편 손−쥔 손 0, 팔 0.3 rad/s 비용 0.378 | **관절 제어에서는 속도 차분 접근이 신호가 되지 않는다** — 상태형 접근과 작은 제어 벌점(cup_grasp) |
| f10 (4번째 iter_09 c1r, fresh, 컵 −12 cm) | e2069 종료. ep800·ep1800 결정론 재생: 오른손 방향은 인벨롭에 가까워짐(ep800 손바닥 1.9 r·+0.35 × 입구·손가락 수평) 그러나 ep1800 에 다시 입구 쪽(2.4 r·+0.67·손가락 −0.34), 엄지 17~20 N·나머지 0.4~3.5 N. 64 env 모두 쥐지만 쥔 뒤 2 s 에 손바닥 +0.4 cm — 들지 않는다(2 cm 이상 7/64). 왼손은 2000 epoch 동안 쥐지 않음: 손바닥 2.5~2.8 r, 손끝 3~4 r, 닫힘 0.25(손 액션 −0.4~−0.9), 엄지 끝 입구 위. 관절 한계 0.00~0.07 | **컵 −12 cm 로 관절 한계는 풀렸다.** 파지 품질 인자를 들기 수입에 **곱하면** 초기 파지(모양 미완)의 들기 이득이 반으로 준다(같은 상태 3 cm: iter_08 +1.43 → iter_09 +0.61) — 쥐고 멈춘다. 손바닥이 컵 옆에 떠 있을 때 인벨롭 자리로 가는 이득 +0.08 뿐 — 왼손이 닫지 않는다. 사용자 영상 판정: "양손 모두 못 들어" |
| f09 (iter_08 c1, f08 ep12200 이어받음) | e3314 종료. ep1600 결정론 재생: 주둥이 높이 차 5.8 cm(f08 21.3)·수평 12 cm, 최대 기울기 평균 80°(64/64 가 45° 초과), j7 여유 7.6°, j2 가 붓는 스텝 96 % 한계. 학습 곡선 ep1500 이후 평평, 비드 타깃 0.1~0.4 %·성공 0~0.001. 두 손 모두 입구 쪽 비스듬한 파지(손바닥 입구 +0.57~+0.92 × 입구·축 2.1~2.5 r·손가락·컵 축 −0.4~−0.7) — f08 도 같음 | **붓는 높이는 풀렸다.** 파지 모양이 인벨롭이 아니다. 들기·조준·붓기 수입이 파지 모양과 무관(비스듬한 파지 테이블→8 cm 들기 이득 +15.9 > 인벨롭 +10.0). 도달성 프로브: 컵 x 0.38 에서 인벨롭은 팔꿈치 j4 여유 1.2°(팔 다 폄), x −0.12 에서 최소 여유 8.7° → 사용자 A: 컵 스폰 x −0.12 |
| f08 (iter_07 c1r, fresh) | e11546 첫 성공 0.001. ep11800 결정론 재생 64 env: 두 손 다섯 손가락 파지(오른손 엄지 1.00·검지 0.93·중지 0.93·약지 0.81·새끼 0.68), 최대 기울기 평균 60°. 두 손 쥐고 기울기 > 30° 스텝: 소스 주둥이가 리시버 입구보다 21.3 cm 위(수평 4.6 cm), 소스 컵 30.6 cm 들림(리시버 7.1 cm), 소스팔 j6 여유 1.4°·j7 여유 2.4°(한계 10° 안 97 %·90 %) | **입구 안 제외·옆 접근·손 방향(iter_07)으로 다섯 손가락 인벨롭과 기울이기가 나왔다.** 그러나 소스 들기 목표(28 cm)가 높이를 끌어올리고 조준 높이 항은 좁아(주둥이 21 → 5.6 cm 내려와도 합 −0.18) 높은 곳에서 붓는다. 관절 여유 항은 붓는 중에만·작아서(0.5 → 0.03 에 +0.04, 기울이기 전 0) 손목이 한계에 붙는다(사용자 영상 판정 09-30) |
| f07 (iter_06 c1, f05 ep500 이어받음) | e1141: 왼손 닿은 손가락 4.17(새끼 0.49), 들기 1.4. ep1000 트레이스: 왼손 다섯 손가락 모두 벽 바깥(새끼 68 %)·컵 19 cm·엄지 외전 +0.93. 오른손 엄지 축 0.57 r(입구 안)·4.7 cm·12.2 N, 중지 −0.46 반쯤 편 채 2.1 r, 엄지 외전 −0.25, 컵 안 들림 | **볼록 다섯 손가락 인자·손가락별 감싸기는 작동한다(왼손).** 손끝-벽 거리를 max(축 거리 − r, 0) 으로 재면 입구 안 손끝도 '벽에 닿음', 입구 안벽 접촉력도 파지로 인정 — 엄지 걸기가 반복된다(f05 e750·f07 e1000). 걸기에서 엄지를 빼는 순간 −1.47(골) |
| f06 (iter_05 c1, f05 ep500 이어받음) | e108: 근처 0.86/0.90 유지(이어받기 성공), 쥠 0.02/0.01, 쥔 스텝 닿은 손가락 2.0/2.0 | **다섯 손가락 인자가 평균(선형)이면 손가락 하나마다 몫이 같아 어려운 마지막 손가락을 건너뛰기 쉽다(사용자 지적).** 닿기 전 손가락별 거리 신호도 없다 |
| f05 (iter_04 c1) | e2457: 두 손 파지(0.70/0.78)·들기(소스 25~30 cm·리시버 9 cm)·기울기 최대 17°·회전 수입 0.27~0.55·비드 0. 좌우 미러 관절 차 ≤0.24 rad(e750) — f04 의 j5 굴림 없음. ep2350 트레이스: 오른손 엄지 1.5 N·중지 1.2 N 두 손가락만(검지 2.1 r 떨어짐), 손바닥 축 2.8 r, 들고 나면 법선 z −0.67, 에피소드당 놓침 19 회(왼손 5). 왼손은 엄지·검지·중지·약지 4 손가락 | **env 파지 판정(엄지 + 다른 손가락 하나)과 min(엄지, 나머지 합) 집기 수입은 두 손가락 집기를 다섯 손가락과 같게 친다(원장 20.84 대 20.36).** 들린 뒤 방향 인자가 풀리면 손이 컵 기준으로 틀어져도 손해가 없다(기울기 45° 차 1.07) |
| f04 (iter_03 c1r2, 팔 절대 목표) | e2256: 왼손 파지 0.85·컵 8 cm·들기 4.1 — 파지→들기 경계에서 총보상이 오르며 이어짐(8.2→10.8→14.2), bounds_loss ≤0.6·팔 한계 ≤0.1·물리 종료 0. ep2050 트레이스: 왼손 손바닥 컵 원점 위 3.4 cm·손가락 축 z −0.57(엄지 4.0·검지 2.1·중지 1.6 N). 오른손 파지 0.22·들기 0 — j5(아래팔 롤) +0.46→+1.50(한계 1.57)로 왼손을 미러한 방향(−0.87)과 반대로 굴려 손바닥이 위(법선 z +0.72)를 보는 받침 자세 | **보상은 좌우 미러 대칭이어도(원장 미러 자세 팔별 차 0) 정책은 한쪽만 다른 해로 갈 수 있다.** 받침 자세 불이익(두 손 합 1.25)이 작으면 그 자리에 머문다. 사용자 영상 판정: 왼손은 더 낮게, 오른손은 한 바퀴 회전 |
| f03 (iter_02, 증분 팔) | e5414: 왼손 근처 0.86·쥠 0.66~0.75 — 파지 자세는 위에서 세로 집기(손가락 축 z −0.95, 엄지 6.0·검지 3.3 N, 손바닥 0). 들기 보상 e3300~3500 0.53 뒤 0: 트레이스에서 든 10 env 중 9 곳이 집기 미끄러짐(총 2~3.5→0.3~0.8), 쥔 채 든 1 곳은 2.61→3.04. 오른손: ep1750 옆 접촉에 컵 16.7° 기울고 교란 −0.24→−0.38 뒤 물러남, ep5400 팔 6/7 관절 |mu| 1.4~2.4 로 한계 96 % 고착 | **닫기·파지가 손 방향과 무관하면 방향 인자를 입구에만 걸어도 세로 집기를 산다.** 두 손가락 집기는 들 때 미끄러진다. 파지 직전 상태 수입이 크면(파지/직전 1.38 배) 파지 판정 전에 머문다 — 2번째 사이클 iter_04 는 1.86 배 |
| i09 | e300~450 오른손 컵 근처 0.43→0.54, **파지 0.06→0.11** 뒤 e450~500 붕괴(근처 0.15·파지 0.02). 같은 구간 grasp+close +0.23 대 cup_disturb −0.26·contact_push −0.08·rush −0.06, 스텝 보상 0.60→0.49 | **파지 시도 벌점이 파지 수입을 넘으면 컵을 떠난다.** `cup_disturb` 의 slide 는 (1−lifted)·tanh(스폰에서 밀린 거리/3 cm) 상태형이라, 한 번 민 컵은 손을 떼도 들기 전까지 매 스텝 비용을 낸다(원장 0p: 0.51 → 0.22) |

이전 사이클 핵심(유지): 접근은 차분, 붓기 허가는 컵 반지름 창(i13) · 방향 신호 필수(i08·1차) · 관절 한계 직접 강벌점 금지(i15, corr +0.90) ·
진행형 항의 비대칭 clamp 금지(i20) · 기울임 수입이 없으면 들고 서 있는다(i21) · 조준 항이 축 회전까지 지불하면 안 된다(i22) ·
주차 함정: 상태 비례 수입이 배달보다 크면 붓지 않는다(회전·배달 수입이 성공 궤적 스텝 수입의 절반 이상).

## 5. 5번째 사이클 iter_00 설계 (09-30 사용자 승인 "진행")

### 5-0. 원칙: 단계 사다리는 **곱이 아니라 합**
4번째 사이클의 반복 실패는 인자 곱에서 나왔다: 한 인자가 작으면 다음 단계 기울기가 사라진다(f10 들기 이득 반감, f05 방향 인자 들린 뒤 풀림, i09 파지 시도 벌점).
각 단계 수입을 **더하고**, 단계마다 "다음 단계 자리 > 이 단계 자리"가 되도록 크기를 정한다. 곱은 한 단계 안에서 같은 뜻의 인자끼리만(예: 쥠 판정 × 접촉 품질) 쓰고,
다른 단계의 모양·방향 인자를 다음 단계 수입에 곱하지 않는다. 바닥 있는 곱도 두 개 넘게 잇지 않는다.

### 5-1. 단계 (두 팔 같은 식·같은 무게, 소스는 y 미러)
| 단계 | 수입 | 형태 |
|---|---|---|
| 1 접근 | 손바닥 → 컵 옆 잡는 자리(원점 기준 K7 자리: 축에서 1.5 r·원점 +0.15 × 입구, 접근 방위는 로봇 쪽)의 거리. 높이 차를 수평보다 무겁게(먼저 내려오기). 손 방향(법선·손가락 축 수평) 점수를 **더한다** | 연속, 먼 척도 + 가까운 척도, 시작 자세(15 cm·15 cm 위)에서도 기울기 |
| 2 감싸기 전 | 손끝별 바깥 벽 거리(입구 안 손끝 = 먼 것) 볼록 평균. 손바닥이 잡는 자리 가까이면 닫기 허용(편 손 요구를 풀어 준다), 멀면 편 손 | 연속 |
| 3 파지 | env 쥠 판정 뒤: 손가락별 **바깥** 접촉(입구 안 제외) 볼록 평균(다섯 손가락일수록 마지막 몫이 크게) | 판정 뒤 계단 + 판정 전 접촉 사전 수입 |
| 4 인벨롭 모양 | 쥔 동안 모양 점수(손바닥 축 거리·높이·법선·손가락 수평)를 **더하는** 보너스 — 들기·조준·붓기에 곱하지 않는다 | 쥔 동안 |
| 5 들기 | 쥔(바깥 접촉) 채 스폰 대비 높이 진척. 소스는 리시버 입구를 넘을 여유(리시버 높이 + 컵 높이 + r)까지만, 그 위는 이득 없음. 처음 몇 cm 가 가장 가파르게 | 모양과 무관 |
| 6 조준(두 손) | 두 컵이 들린 뒤 소스 주둥이 → 리시버 입구 수평·높이(주둥이 기준, 넓은 받침: 21 cm 위·12 cm 옆에서도 기울기) | 연속 |
| 7 붓기 | 조준 창 안에서 회전 진척(방향 있는 기울기) + **비드 도착 증분이 주역**, 다 부으면 세우기 | 증분 |

나머지 항: 흘림 벌점, 성공 보너스, 컵 교란(쥐기 전 밀림·쓰러뜨림은 속도로만, 튕김), 쥔 동안 관절 여유 비용(유계, 관절 번호 없음), 작은 제어 비용, 리시버 세움·중심선 쪽.
합 15 개 안팎(25 개 이하).

### 5-2'. 크기(원장으로 확인)
- 붓는 상태 회전·배달 몫 ≥ 총수입의 절반, 붓기 > 들고 서 있기 + 1.
- 인벨롭 − 입구 쪽 비스듬한 파지(실측): 들린 상태 두 손 합 ≥ 12. 비스듬한 파지의 테이블 → 8 cm 들기 이득 ≤ 인벨롭 이득의 0.5 — **들기 시작(3 cm) 이득은 파지 모양과 무관하게 ≥ +1.4**(f10 실측 쥔 자세).
- 원장 단조: 0 < 1 < 1b < 2 < 3 < 4 < 5 < 6(3 cm) < 6(8 cm) < 7 < 8 < 9(0°·45°·70°·95°) < 10, 골 0.

### 5-2. 제어 벌점은 cup_grasp 수준 (매끄러움은 EMA·속도 상한이 맡는다)
- cup_grasp: 액션 변화율 0.02·tanh(평균 변화²/0.1) + 팔 속도 0.12·tanh(|qd|/1 rad/s) + 3 rad/s 초과 폭주 비용.
- **iter_03 부터 팔 액션 크기 비용 금지**: 팔 액션이 절대 목표(a=0 = 시작 자세)라 |a| 비용은 시작 자세로 끌어당기는 항이 된다.
- 기준: 무작위 ±1 액션 비용 ≤ 0.06/스텝, 팔 0.3 rad/s 비용 ≤ 0.08/스텝. 손 채널 ±1 은 비용 0(절대 목표의 정상 명령).
- 액션 반전(flip)·|a|>0.9 포화 벌점은 관절 증분에서는 탐색을 억누른다 — 두지 않는다.

### 5-4. 하지 말 것
- 손바닥 속도에만 의존하는 접근 수입, 목표 각도·대역·각도 상한 처방, 스폰 기준 만남 통로, 관절 한계 강한 직접 벌점, 새 절대 거리 상수(높이 허용과
  접근 척도는 컵 반지름 배수 또는 명시 근거), 자산 치수 하드코딩, 곱 사다리, 손 채널 포화 벌점.


### 5-5. iter_01 에서 고칠 것 (사용자 10-01 "1번", 나머지 5-1~5-4 는 그대로)
1. **접근 중 편 손**: 컵이 손가락 사이에 들어오기 전(파지 전)에는 접근 수입 **전체(거리·높이·방향)** 가 손을 편 만큼 커야 한다 — 접근 구간 매 스텝 편 손 − 반쯤 쥔 손 ≥ 팔당 +0.15(T4). 편 손 요구를 풀어 주는 조건과 닫기 수입(`close`)은 손바닥 자리만이 아니라 **손가락이 컵을 둘러쌌다는 기하**(네 손가락 끝과 엄지 끝이 컵 벽 바깥 서로 반대쪽, 손끝 바깥 벽 거리 작음)에서만 켠다 — 손바닥이 K7 자리(축 1.5 r)에 오기 전에 손가락이 말리면 손끝이 컵 앞쪽에 머물러 이 조건을 못 채운다. 벌점이 아니라 곱하는 인자(상점)로(진행 세금 금지).
2. **다 부으면 끝**: 성공 문턱(0.8)을 넘은 뒤 남은 비드를 안고 기울인 채 머무는 것이 끝까지 붓고 세우는 것보다 이득이면 안 된다(T3 마지막 행), 다 부은 순간 수입 절벽이 없어야 한다(T1). 기울임 수입은 남은 비드 비율에 비례해 줄고(한 알 남았다고 전액이 아니게), 세우기·성공 수입이 그 자리를 이어받는다.

### 5-6. iter_02 에서 고칠 것 (사용자 10-01 "재진행", 한 가설·한 변경, 나머지 iter_01 c1r2 그대로)
1. **쥐기 전에도 관절 여유**: 팔 관절이 한계 근처(`ctx.*_arm_limit_margin` 작음)면 접근 단계 수입(접근 거리·높이·방향, 감싸기 전)이 줄어야 한다 — 곱하는 인자(상점, 바닥 있음, 관절 번호 없음, 유계)로. 벌점을 더하지 않는다(진행 세금 금지). 쥔 뒤의 `joint_margin` 비용은 그대로.
   검사 [59](f01 실측 머묾 표본): 같은 기운 자세에서 관절 여유만 0.5 로 바꾸면 팔당 스텝당 +0.2 이상, 수평 인벨롭 자리(여유 0.5)는 +0.5 이상. traj_ledger T1~T5 와 [1]~[58] 은 그대로 통과해야 한다(참 궤적은 여유 0.5 로 잰다).

### 5-7. iter_03 에서 고칠 것 (사용자 10-01 "1번", 한 가설·두 변경, 나머지 iter_02 c1 그대로)
가설: 손 방향을 컵 옆에 와서야 고치게 두면 한쪽 손이 틀린 쪽(손바닥 위)으로 굴러 한 손가락 접촉에 머문다 — 컵에서 멀 때 먼저 수평으로 돌리게 하고, 둘러싸지 않은 한 손가락 접촉에는 수입을 주지 않는다.
1. **먼 곳에서도 손 방향**: 접근 수입의 손 방향 몫이 컵에서 먼 시작 자세(21 cm)에서도 충분해야 한다 — 같은 손바닥 위치에서 수평 인벨롭 방향이 시작 자세(손가락 아래)보다 팔당 스텝당 +0.3, 손바닥이 위를 보는 뒤집힌 손보다 +0.5(검사 [61]). 뒤집힌 손(법선 z > 0 쪽으로 크게)은 아래로 기운 손보다 싸야 한다. 참 궤적(A 구간에서 내려오며 수평으로 돌림)의 T1~T5 는 그대로 통과.
2. **둘러싸지 않은 접촉은 무수입**: 판정 전 접촉 사전 수입(`pre`)은 손바닥이 잡는 자리 가까이·손이 수평일 때만(같은 단계 인자) — f02 오른손처럼 뒤집힌 손의 새끼 하나가 컵을 미는 접촉 몫은 스텝당 0.1 이하(검사 [60]), 같은 컵에서 수평 인벨롭 편 손은 뒤집힌 손 머묾보다 +1.0 이상.

### 5-8. iter_04 에서 고칠 것 (사용자 10-01 "진행", 한 가설·두 변경, 나머지 iter_03 c1 그대로)
가설: 접근 중 손을 편 손에서 멀어질수록(닫힘 0.05 → 1) 접근 단계 수입이 매끄럽게 줄어야 하고, 쥐기 전에 엄지를 네 손가락 반대쪽으로 둘 기울기가 있어야 컵 옆 반쯤 쥔 손 주차에서 빠져나온다.
1. **편 손 배수의 전 범위 기울기**: 닫힘 0.05~1.0 어디서든 손을 펴면 접근 단계 수입이 오른다(평평한 구간 없음). 손끝을 말아 벽에 대는 감싸기 전 수입(`wrap`)도 손이 컵을 둘러싸기 전에는 편 손 인자 안에 둔다. 검사 [62](f03 머묾 자세 닫힘 0.75 → 0.1, 0.1 칸마다 +0.03 이상), 원장 T4b(접근 구간 매 스텝 닫힘 0.05 > 0.3 > 0.5 > 0.7 칸마다 팔당 +0.05 이상), T1(닫는 D 구간 한 스텝 하락 0.05 이하)은 그대로.
2. **쥐기 전 엄지 대향 기울기**: 손바닥이 잡는 자리 가까이 있을 때, 엄지 끝이 네 손끝과 컵 축 둘레로 반대쪽에 가까울수록 작은 수입(접근 단계 안, 상점). 검사 [63](f03 머묾 자세 엄지 0° → 90° → 180° 칸마다 +0.05 이상). 참 궤적·T3 이탈 손해는 유지.

## 6. 평가

- 틱은 단계 관점: near_rate → cup_in_pocket·oppose → closure → grasped → cup_lift → converge·aim → pour. 단계가 오를 때 보상·에피소드 길이가 떨어지면 골.
- full-joint 지표: `joint/{src,rcv}_arm_limit_frac`, `_arm_track_err_deg`, `_hand_track_err_deg`.
- 성공 판정은 결정론 평가(담김 ≥ 0.8·최대 기울기 순간 방향 오차 < 30°·주둥이가 입구보다 위).
- 후보 선택 전 `traj_ledger.py` T1~T5 전부 통과(2-1) + 검사 [1]~[63]·원장 T4b. [25]·[31]·[57] 은 10-01 은퇴(엄지가 둘러싸기 전 닫기 이득을 요구해 5-5 1 과 충돌) — 참고 출력만.

## 7. 시작

fresh(보상이 새 행동 — 편 손 접근 — 을 요구). 웜스타트 없음. seed 42. env `object_spawn_dx −0.12`·팔 절대 목표·물리 이상 종료·bounds 0.005·손 속도 상한·편 한계 그대로, **명령 지연 끔**(이송 성공 뒤 켠다, 사용자 10-01).
# 5번째 사이클 iter_04 운영 노트 — 2026-10-01

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다(특히 2-1 참 궤적 최적, 4절 rh5_f03 행, 5-8 고칠 것).

## 사용자 결정 (10-01)
- f03 e941 틱(두 손 반쯤 쥔 손 머묾·파지 0) → "1번": f03 종료(ep1000), server GPU1 결정론 재생.
- 재생 보고 뒤 "진행": iter_04 = 편 손 배수 전 범위 기울기 + 쥐기 전 엄지 대향 기울기, 보상만 바꾸고 fresh.
- (앞서) f02 왼손 "확실한 인벨롭 파지" — 목표 자세: 손바닥 1.44 r·높이 0.16 × 입구·법선 z −0.16·손가락 z +0.05.

## 측정 사실 (rh5_f03, iter_03 c1, fresh, 지연 끔, e1000 종료)
- 학습 곡선 e941(10 epoch 평균, 오른손/왼손): 컵 10 cm 안 0.92/0.92 · 파지 0/0 · 컵 근처 쥐기 전 닫힘 0.71/0.76 · 20 cm 밖 닫힘 0.53/0.50 · 쥐기 전 손바닥 수직 성분 0.38/0.27 · 총보상 3.24 · entropy 25.0.
- ep1000 결정론 재생: DESIGN_BRIEF 4절 rh5_f03 행 그대로.
- iter_03 c1 코드 검사(check_iter04 기준선): [62] 오른손 0.75:+5.87 → 0.6:+5.87 → 0.5:+5.87 → 0.4:+5.97 → 0.1:+6.30, 왼손 같은 모양(0.5 위 0.000) / [63] 엄지 0°·90°·180° 모두 같은 값 / T4b 0.05>0.3 +0.171 · 0.3>0.5 +0.026 · 0.5>0.7 −0.296.

## iter_04 에서 할 것 (DESIGN_BRIEF 5-8, 한 가설·두 변경)
1. 편 손 배수 전 범위 기울기, 둘러싸기 전 wrap 도 편 손 인자 안([62], T4b, T1).
2. 쥐기 전 엄지 대향 기울기([63]).
3. 나머지 iter_03 c1 그대로. 검사: check_iter04 [1]~[63](은퇴 [25][31][57] 제외) + traj_ledger T1~T5·T4b.


## Previous response

```python
import torch
import math


def _sat(x, lo: float = 0.0, hi: float = 1.0):
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def _sabs(x: torch.Tensor, eps: float) -> torch.Tensor:
    """매끄러운 |x| (0 에서 꺾이지 않음)."""
    return torch.sqrt(x * x + eps * eps) - eps


def _close_in(d, r):
    """손끝-바깥 벽 거리 성형 (이전 코드 그대로, 척도는 컵 반지름 배수)."""
    return 0.5 * torch.exp(-d / (1.2 * r)) + 0.5 * torch.exp(-d / (0.22 * r))


def _limit_cost(margin: torch.Tensor) -> torch.Tensor:
    """관절 한계 근접 연속 비용 (관절 번호 없음, 유계 [0,1)). 이전 코드 그대로 (쥔 뒤 joint_margin 용)."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    return torch.tanh(c.sum(dim=-1))


def _margin_fac(margin: torch.Tensor) -> torch.Tensor:
    """iter_02 [5-6 1] 쥐기 전 접근 단계 수입에 곱하는 관절 여유 인자 [0.5, 1] (관절 번호 없음, 유계, 바닥 0.5).
    관절별 c = (1 - sat(여유 / 0.25))^2, q = prod_j (1 - 0.5 c_j), 인자 = 0.5 + 0.5 q.
    참 궤적(여유 0.5)에서는 정확히 1. 이전 코드 그대로."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    q = torch.prod(1.0 - 0.5 * c, dim=-1)
    return 0.5 + 0.5 * q


def _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r):
    """컵 좌표(축 = cup_up) 손끝-바깥 벽 거리. 입구 안 손끝은 '먼 것'(2 r)으로 친다 (f07 교훈).
    입구 안 판정 높이 하한 = 입구 - 1 r (_surround 와 같은 값). 이전 코드 그대로."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)                                   # (N,F)
    rad = (rel - h_tip.unsqueeze(-1) * up).norm(dim=-1)              # (N,F)
    gap_raw = torch.clamp(rad - r, min=0.0)
    w_rad = _sat((0.9 * r - rad) / (0.3 * r))
    w_h = _sat((h_tip - (mouth_z - r)) / (0.5 * r)) * _sat((mouth_z + 1.5 * r - h_tip) / (0.5 * r))
    w_in = w_rad * w_h                                               # 1 = 입구 안
    gap = (1.0 - w_in) * gap_raw + w_in * (2.0 * r)
    h_t = h_tip[:, 0]
    h_err_t = torch.clamp(-h_t, min=0.0) + torch.clamp(h_t - 0.5 * mouth_z, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2 + 1e-12)      # (N,)
    h_o = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * mouth_z - h_o, min=0.0) + torch.clamp(h_o - 0.8 * mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2 + 1e-12)       # (N,F-1)
    return d_thumb, d_oth, 1.0 - w_in


def _surround(tips_pos, palm_pos, cup_pos, cup_up, mouth_z, r):
    """iter_01 [5-5 1] 손가락이 컵을 둘러쌌다는 기하 [0,1]. 이전 코드 그대로.
    = (서로 반대쪽) x (네 손가락 끝이 벽 가까이) x (엄지 끝이 벽 가까이)."""
    EPS = 1e-6
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h = (rel * up).sum(dim=-1)                                       # (N,F)
    rad_v = rel - h.unsqueeze(-1) * up                               # (N,F,3)
    rad = rad_v.norm(dim=-1)                                         # (N,F)
    w_in = _sat((0.9 * r - rad) / (0.3 * r)) * _sat((h - (mouth_z - r)) / (0.5 * r))
    gap = (1.0 - w_in) * torch.clamp(rad - r, min=0.0) + w_in * (2.0 * r)
    herr = torch.clamp(-0.4 * mouth_z - h, min=0.0) + torch.clamp(h - 0.8 * mouth_z, min=0.0)
    d = torch.sqrt(gap * gap + herr * herr + 1e-12)
    cl = torch.exp(-(d / (0.4 * r)) ** 2)                            # (N,F)
    cl_t = cl[:, 0]
    cl_f = cl[:, 1:].mean(dim=1)

    u = rad_v / (rad.unsqueeze(-1) + EPS)
    p_rel = palm_pos - cup_pos
    p_h = p_rel - (p_rel * cup_up).sum(dim=-1, keepdim=True) * cup_up
    u_p = p_h / (p_h.norm(dim=-1, keepdim=True) + EPS)
    near = u_p + u[:, 0, :]
    near_hat = near / (near.norm(dim=-1, keepdim=True) + EPS)
    cos_f = (u[:, 1:, :] * near_hat.unsqueeze(1)).sum(dim=-1).mean(dim=1)
    opp = _sat((0.5 - cos_f) / 0.8)
    return opp * cl_f * cl_t


def _thumb_oppose(tips_pos, cup_pos, cup_up, r):
    """iter_04 [5-8 2] 엄지 끝과 네 손끝의 컵 축 둘레 방위 반대 정도 [0,1] (0 = 같은 방위, 1 = 정반대).
    repair (T1 B 구간 -0.045/팔): 단위 방위 벡터는 손끝이 컵 축 가까이를 지날 때 한 스텝에 뒤집힌다 (추정 원인).
    -> 부드러운 정규화 u = v / sqrt(|v|^2 + r^2) (척도 = 컵 반지름). 축 근처 손끝은 길이가 0 으로 줄어
    cos 이 0(대향 0.5) 쪽으로 매끄럽게 옮겨 가고, 축에서 먼 손끝(>= 2 r)은 단위 벡터와 거의 같다.
    네 손끝은 다시 정규화하지 않고 평균만 낸다 (평균 길이 <= 1 -> cos 은 [-1, 1])."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h = (rel * up).sum(dim=-1, keepdim=True)
    rad_v = rel - h * up                                             # (N,F,3)
    u = rad_v / torch.sqrt((rad_v * rad_v).sum(dim=-1, keepdim=True) + r * r)
    u_f = u[:, 1:, :].mean(dim=1)                                    # 네 손끝 방위 평균 (길이 <= 1)
    cos_tf = (u[:, 0, :] * u_f).sum(dim=-1)
    return _sat(0.5 * (1.0 - cos_tf))


def _hand_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                arm_qd, arm_limit_margin, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                other_cup_pos, r, mouth_z, lift_target, lift_relief):
    """팔별 무가중 항. 두 팔 같은 식 (한 팔에만 붙는 부호·상수 없음). 모든 거리 척도는 컵 반지름 배수.
    iter_04 변경 (DESIGN_BRIEF 5-8, 한 가설·두 변경):
      [5-8 1] 편 손 배수 shut 이 닫힘 0.05~1.0 전 범위에서 기울기를 갖게 하고, 감싸기 전 수입 wrap 도
              편 손 인자 안 (둘러싸면 release 로 풀림).
      [5-8 2] 접근 수입 안에 엄지 대향 몫(손바닥이 잡는 자리 가까이 x 엄지-네 손끝 반대 방위).
    repair: shut 모양(선형 0.7 + 오목 0.3), 근처 k_open 0.3 -> 0.35, wrap 전용 편 손 계수 k_wrap(근처 0.6),
            엄지 대향 방위 부드러운 정규화 + 몫 0.2 -> 0.25. 나머지는 iter_03 c1 그대로."""
    EPS = 1e-6
    dt = cup_pos.dtype
    g_raw = grasped.to(dt)

    # ---- 손바닥 기하 (컵 좌표) : K7 자리 = 축에서 1.5 r, 높이 원점 + 0.15 x 입구
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    rad_vec = to_cup - h_c * cup_up
    rad_p = rad_vec.norm(dim=-1)
    dh = -h_c.squeeze(-1) - 0.15 * mouth_z
    dr = rad_p - 1.5 * r
    n_dot = (n_dir * cup_up).sum(dim=-1)
    f_dot = (f_dir * cup_up).sum(dim=-1)
    e_in = rad_vec / (rad_p.unsqueeze(-1) + EPS)
    tgt = 0.67 * n_dir + 0.73 * f_dir                                 # K7 손바닥 -> 컵 축 방향
    tgt_h = tgt - (tgt * cup_up).sum(dim=-1, keepdim=True) * cup_up
    tgt_hat = tgt_h / (tgt_h.norm(dim=-1, keepdim=True) + EPS)
    cos_env = (e_in * tgt_hat).sum(dim=-1)

    # ---- 손끝 기하 / 바깥 접촉 (입구 안 제외)
    d_thumb, d_oth, out_w = _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r)
    f_raw = torch.clamp(finger_force, min=0.0)
    f_out = f_raw * out_w
    touch = torch.tanh(f_out / 0.7)                                   # (N,F) 손가락별 접촉 점수
    touch_o = touch[:, 1:]
    oth_ok = _sat(touch_o.max(dim=1).values / math.tanh(1.0 / 0.7))
    g_eff = g_raw * out_w[:, 0] * oth_ok                              # 쥠 판정 x 바깥 접촉 (같은 단계 인자)
    free = 1.0 - g_eff                                                # 쥐기 전 게이트

    # ---- iter_02 [5-6 1] 쥐기 전 관절 여유 인자 (접근 단계 안에서만; 쥔 뒤 1)
    j_pre = 1.0 - free * (1.0 - _margin_fac(arm_limit_margin))

    # ---- 손 방향 점수 (수평 손)
    s_fl = 0.5 * torch.exp(-(f_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(f_dot))
    s_nl = 0.5 * torch.exp(-(n_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(n_dot))
    #   iter_03 [5-7 1] 뒤집힌 손(법선 z > 0 쪽으로 크게)은 아래로 기운 손보다 싸게: 위쪽으로만 깎는 인자 (그대로).
    up_ok = 1.0 - _sat((n_dot - 0.25) / 0.45)
    level = s_fl * (0.4 + 0.6 * s_nl) * (0.2 + 0.8 * up_ok)
    level_fac = 0.3 + 0.7 * level                                     # 바닥 0.3 인 수평 손 인자

    # ---- 1 접근 거리 (높이 x2 로 먼저 내려오기, 앞·중심선 쪽 가산은 쥐기 전에만)
    far_side = torch.clamp(palm_pos[:, 0] - cup_pos[:, 0], min=0.0) * free
    to_other = other_cup_pos[:, :2] - cup_pos[:, :2]
    e_other = to_other / (to_other.norm(dim=-1, keepdim=True) + EPS)
    lat = ((palm_pos[:, :2] - cup_pos[:, :2]) * e_other).sum(dim=-1)
    cross = torch.clamp(lat - 0.5 * r, min=0.0) * free
    dr_eff = torch.clamp(dr, min=0.0) + 3.0 * torch.clamp(-dr, min=0.0)
    dh_abs = _sabs(dh, 0.1 * r)
    side_d = dr_eff + far_side + cross
    d_app = side_d + 2.0 * dh_abs
    reach_far = _sat(1.0 - d_app / (30.0 * r))
    reach_near = (1.0 - torch.tanh(d_app / (2.0 * r))) * level_fac
    reach = 0.5 * reach_far + 0.5 * reach_near

    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    near_w = torch.exp(-(d_app / (4.0 * r)) ** 2)
    orient = level * (0.4 + 0.6 * near_w * dir_env)

    # ---- 손 안 판정: 손바닥 자리(pocket) x 손가락 둘러싸기(surround) (iter_01 [5-5 1])
    pocket = _sat((3.2 * r - rad_p) / r) * torch.exp(-(dh / (2.0 * r)) ** 2) * _sat((cos_env + 0.3) / 0.6)
    surround = _surround(tips_pos, palm_pos, cup_pos, cup_up, mouth_z, r)
    in_hand = _sat(pocket / 0.4) * surround                           # 같은 단계(컵이 손 안) 인자끼리의 곱
    release = torch.maximum(_sat(in_hand / 0.5), g_eff)

    # ---- 편 손 인자 (iter_04 [5-8 1]: 전 범위 기울기, repair)
    #   x = sat((c - 0.05)/0.95) (편 손 최소 약 0.05, 브리프 1절), shut = 0.7 x + 0.3 (1 - (1 - x)^2).
    #   기울기(/닫힘 단위): 0.05 에서 1.37, 0.5 에서 약 1.07, 0.75 에서 약 0.90, 1.0 에서 0.74 (평평한 구간 없음).
    #   값: 0.3 -> 0.32, 0.5 -> 0.55, 0.7 -> 0.75, 1.0 -> 1.
    #   이전 응답 shut 은 0.5 위 기울기 0.53 이라 [62]·T4b 0.5>0.7 실패 (Review items).
    #   근처 k_open 0.3 -> 0.35: 근처 최대 실효 기울기 0.35 x 1.37 = 0.48 로 이전 응답(0.3 x 1.64 = 0.49) 이하
    #   -> D 구간(닫기) 한 스텝 하락 상한이 이전 응답보다 커지지 않는다 (계산).
    #   먼 곳 0.9 그대로: T4(시작 자세 0.05 대 0.5) 차 = 0.9 x 0.55 x M (M 약 0.38, A@0 총보상으로 역산, 가설) = 약 0.19.
    C_OPEN = 0.05
    x_c = _sat((hand_closure - C_OPEN) / (1.0 - C_OPEN))
    shut = 0.7 * x_c + 0.3 * (1.0 - (1.0 - x_c) ** 2)
    k_open = 0.35 + 0.55 * (1.0 - near_w)
    open_fac = release + (1.0 - release) * (1.0 - k_open * shut)
    #   wrap 전용 편 손 계수 (같은 편 손 인자의 더 강한 형태, 둘러싸면 release 로 풀림):
    #   T4b t=305 (손바닥 3.16 r, 닫힘 0.7 손끝 벽 0.98 r) 에서 손끝을 말아 벽에 댄 wrap 상승(이전 +0.193)을 깎는다.
    #   추정 순차: 접근·높이 차 약 +0.26, wrap 차 약 -0.14 -> 0.5>0.7 칸 약 +0.12 (계산, 가설; 목표 +0.05).
    k_wrap = 0.6 + 0.3 * (1.0 - near_w)
    open_wrap = release + (1.0 - release) * (1.0 - k_wrap * shut)

    # ---- iter_04 [5-8 2] 엄지 대향 몫 (접근 단계 안, 상점): 손바닥이 잡는 자리 가까이 x 엄지-네 손끝 반대 방위.
    #   편 손 인자·관절 여유 인자 안 (대향을 위해 손을 쥐는 쪽으로 가지 않게). free 게이트 없음 -> 쥠 판정 순간 절벽 없음.
    #   repair: 부드러운 정규화로 축 근처 방위 뒤집힘 제거. 손끝 1 mm/스텝 이동이면 대향 변화 약 0.018/스텝 이하
    #   -> 팔당 하락 약 0.01 이하 (계산). 정규화로 [63] 칸 차가 줄어드는 몫(약 0.66~0.85 배)을 몫 0.2 -> 0.25 로 보전.
    #   [63] 추정: 머묾 자세(근처 open_fac 약 0.74) 90 도 칸마다 W_APP x 0.25 x 0.33~0.42 x 0.74 = 약 0.12~0.16 x 자리 인자.
    oppose = _thumb_oppose(tips_pos, cup_pos, cup_up, r)
    opp_site = _sat(pocket / 0.4) * oppose

    #   iter_03 [5-7 1] 거리와 무관한 손 방향 몫 0.2 x level (그대로) + iter_04 엄지 대향 몫 0.25 x opp_site.
    approach = j_pre * open_fac * (0.65 * reach + 0.35 * orient + 0.2 * level + 0.25 * opp_site)

    # ---- 1' 먼저 내려오기: 잡는 높이 점수 (편 손 인자 + iter_02 관절 여유 인자) (그대로)
    low = j_pre * open_fac * (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 : 손끝별 바깥 벽 거리 볼록 평균(제곱) + 손가락이 둘러쌌을 때만 닫기
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    #   iter_04 [5-8 1] wrap 은 편 손 인자(wrap 전용 계수) 안: 둘러싸기 전 손끝을 말아 벽에 대는 수입은 손을 쥔 만큼 깎이고,
    #   둘러싸면(release -> 1) 원래대로.
    wrap = j_pre * open_wrap * part * part * (0.15 + 0.85 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)
    close = j_pre * in_hand * level_fac * _sat(hand_closure / 0.7)

    # ---- 3 파지 : 판정 뒤 계단(>= 0.5) + 판정 전 접촉 사전 수입(<= 0.4) (그대로)
    f_t = f_out[:, 0]
    f_o = f_out[:, 1:].sum(dim=1)
    pre_raw = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    pre_site = _sat(pocket / 0.4) * _sat(level / 0.6)
    pre = pre_raw * pre_site
    part_o = touch_o.mean(dim=1)
    post = 0.5 + part_o * part_o * part_o                             # 세제곱: 5지째 몫이 가장 크다
    grasp = g_eff * post + (1.0 - g_eff) * pre

    # ---- 4 인벨롭 모양 점수 + 손바닥 굴림 비용 (그대로)
    s_h = torch.exp(-(dh / (0.6 * r)) ** 2)
    s_r = torch.exp(-(dr / (0.5 * r)) ** 2)
    s_f = torch.exp(-(f_dot / 0.3) ** 2)
    s_n = torch.exp(-(n_dot / 0.3) ** 2)
    q = 0.30 * s_h + 0.25 * s_r + 0.30 * s_f + 0.15 * s_n
    q2 = q * q
    roll = _sat((torch.abs(n_dot) - 0.25) / 0.45)

    # ---- 5 들기 : 모양과 무관, 처음 몇 cm 가 가장 가파르고 목표 위는 이득 없음 (그대로)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.2 * r, min=0.0)
    lin = _sat(h_pos / lift_target)
    lin = lin + (1.0 - lin) * lift_relief
    prog = 0.7 * torch.tanh(h_pos / (0.7 * r)) + 0.3 * lin
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    lift = g_eff * calm * prog
    lifted_w = _sat(h_pos / (0.7 * r))

    # ---- 교란 (그대로)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    topple = (1.0 - g_eff) * torch.tanh(cup_ang_vel[:, :2].norm(dim=-1) / 2.0)
    f_t_raw = f_raw[:, 0]
    f_o_raw = f_raw[:, 1:].sum(dim=1)
    f_o_max = f_raw[:, 1:].max(dim=1).values
    thumb_excess = torch.clamp(f_t_raw - f_o_raw - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o_max - f_t_raw, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    rush = near_w * free * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    jam = _sat((1.3 * r - rad_p) / (0.3 * r))
    disturb = 0.5 * slide + 0.3 * topple + 0.3 * one_sided + 0.2 * palm_push + 0.3 * fling \
        + 0.06 * rush + 0.6 * jam

    return {"approach": approach, "low": low, "wrap": wrap, "close": close, "grasp": grasp,
            "g_eff": g_eff, "q2": q2, "roll": roll, "prog": prog, "lift": lift,
            "lifted_w": lifted_w, "disturb": disturb}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 5번째 사이클 iter_04: 단계별 합 사다리 (곱 사다리 없음). 두 팔 같은 식·같은 무게.
    iter_04 변경 (DESIGN_BRIEF 5-8, 한 가설·두 변경): _hand_terms 의 편 손 배수 전 범위 기울기(+ wrap 을 편 손 인자 안으로),
    쥐기 전 엄지 대향 몫(접근 수입 안). 나머지 항·가중은 iter_03 c1 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (iter_03 c1 그대로) ----------------------------------------------------------------
    W_ALIVE = 0.5
    W_APP = 2.0           # 접근 몫 최대: reach 0.65 + orient 0.35 + level 0.2 + 엄지 대향 0.25 (iter_04) -> 팔당 최대 2.9
    W_LOW = 1.2
    W_WRAP = 1.2
    W_CLOSE = 1.3
    W_GRASP = 3.0
    W_SHAPE = 1.5
    W_SHAPE_LIFT = 5.0
    W_ROLL = 1.5
    W_LIFT = 2.5
    W_AIM = 2.5
    W_POUR_H = 4.0
    W_ROT = 23.0          # weight 23 with delivery 30/bead: 회전+배달 몫 약 52 % (추정)
    W_DELIVER = 30.0      # 알당 30, 부호 있는 증분 (i20: 진행형 비대칭 clamp 금지)
    W_SPILL = 40.0
    W_SUCCESS = 20.0
    W_FINISH = 1.5
    W_MARGIN = 0.8        # 쥔 뒤 관절 여유 비용 (5-6: 그대로)
    W_CARRY = W_ROT + W_POUR_H
    TAIL = 0.25
    UPRIGHT_TOL = math.radians(20.0)
    LIM = math.radians(81.0)

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    cup_h = mouth_z - float(ctx.cup_bottom_z)          # 컵 높이 (자산 속성에서 계산)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    active = float(max(int(ctx.num_beads), 1))
    LIFT_RCV = 3.0 * r_cup

    src_g_raw = ctx.src_grasped.to(dt)
    rcv_g_raw = ctx.rcv_grasped.to(dt)
    both_g_raw = src_g_raw * rcv_g_raw
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) --------------------------------------------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    above_rim = _sat(1.0 + dz_lip / r_cup)
    window = lip_gate * above_rim
    theta_pos = torch.clamp(theta, min=0.0)

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    clear_src = torch.clamp(h_rcv, 0.0, LIFT_RCV) + cup_h + r_cup
    lift_relief_src = window * _sat(theta_pos / (0.5 * LIM))

    src = _hand_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                      ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped, ctx.src_arm_qd,
                      ctx.src_arm_limit_margin,
                      ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                      ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, clear_src, lift_relief_src)
    rcv = _hand_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                      ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                      ctx.rcv_arm_limit_margin,
                      ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                      ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, LIFT_RCV, zeros)

    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    alive = W_ALIVE * ones

    # ---- 1~3 접근(+먼저 내려오기) / 감싸기 전 / 파지 ----
    #   iter_04: approach 안에 엄지 대향 몫 (W_APP x 0.25 x opp_site, 팔당 최대 +0.5) 포함, wrap 은 편 손 인자 안
    approach_src = W_APP * src["approach"] + W_LOW * src["low"]
    approach_rcv = W_APP * rcv["approach"] + W_LOW * rcv["low"]
    prewrap = W_WRAP * (src["wrap"] + rcv["wrap"]) + W_CLOSE * (src["close"] + rcv["close"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- 4 인벨롭 모양 보너스 - 손바닥 굴림 비용 ---------------------------------------------------------
    envelope = src_g * (src["q2"] * (W_SHAPE + W_SHAPE_LIFT * src["prog"]) - W_ROLL * src["roll"]) \
        + rcv_g * (rcv["q2"] * (W_SHAPE + W_SHAPE_LIFT * rcv["prog"]) - W_ROLL * rcv["roll"])

    # ---- 5 들기 (모양 무관) --------------------------------------------------------------------------
    lift_src = W_LIFT * src["lift"]
    lift_rcv = W_LIFT * rcv["lift"]

    # ---- 6 조준 -----------------------------------------------------------------------------------
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))
    h_err = torch.abs(dz_lip - 0.25 * r_cup)
    hz = 0.3 * (1.0 - torch.tanh(h_err / (1.5 * r_cup))) + 0.7 * _sat(1.0 - h_err / (12.0 * r_cup))
    xy = 0.5 * (1.0 - torch.tanh(lip_d / wide)) + 0.5 * (1.0 - torch.tanh(lip_d / two_r))
    aim_carry = W_AIM * both_g * lifted * (0.6 * xy + 0.4 * hz)

    # ---- 비드 장부 --------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac / TAIL)
    spill_frac = _sat(ctx.bead_spill_frac)

    # ---- 7 붓기: 창 안 회전 진척 x want ---------------------------------------------------------------
    rot_prog = _sat(theta_pos / (1.4 * LIM))            # 95 도까지 단조 증가
    tilt_share = want * window * rot_prog
    pour_rot = W_ROT * both_g * tilt_share
    pour_h = W_POUR_H * both_g * tilt_share * hz
    aim = aim_carry + pour_h

    # ---- 7 배달 (주역, 부호 있는 증분) ---------------------------------------------------------------
    pour_delta = W_DELIVER * ctx.d_in_target * active

    # ---- 다 부으면 끝: 마지막 몫 이어받기 + 세우기 / 성공 ---------------------------------
    upright_src = 1.0 - torch.tanh(tilt_src / (0.3 * LIM))
    finish_upright = both_g * (1.0 - want) * (W_CARRY * (1.0 - spill_frac) + W_FINISH * upright_src)
    success_bonus = W_SUCCESS * ctx.success.to(dt)

    # ---- 흘림 / 창 밖 조기 기울임 -----------------------------------------------------------------------
    post = torch.maximum(src_g_raw, src["lifted_w"])
    spill = -W_SPILL * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g_raw * pre_ramp * out_gate

    # ---- 충돌 (실기 안전, 유계) ------------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g_raw) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g_raw) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                                + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- 컵 교란 -------------------------------------------------------------------------------------
    tilt_pen_src = src_g * (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = rcv_g * (1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2))
    cup_disturb = -(src["disturb"] + rcv["disturb"] + 0.3 * tilt_pen_src + 0.3 * tilt_pen_rcv)

    # ---- 리시버 세움·중심선 쪽 --------------------------------------------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- 관절 여유 비용 (쥔 동안, 5-6: 그대로) -------------------------------------------------------------
    joint_margin = -W_MARGIN * (src_g * _limit_cost(ctx.src_arm_limit_margin)
                                + rcv_g * _limit_cost(ctx.rcv_arm_limit_margin))

    # ---- 제어 비용 (cup_grasp 수준; 팔 액션 크기 비용 없음, 손 채널 포화 벌점 없음) -----------------------------
    a = ctx.actions
    pa = ctx.prev_actions
    da_arm = torch.cat([a[:, 0:7] - pa[:, 0:7], a[:, 13:20] - pa[:, 13:20]], dim=-1)
    da_hand = torch.cat([a[:, 7:13] - pa[:, 7:13], a[:, 20:26] - pa[:, 20:26]], dim=-1)
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (
        torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
        + torch.tanh(torch.clamp(qd_rcv.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0))
    rate_hand = 0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)
    ctrl = -(rate_arm + speed_arm + runaway + rate_hand)

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "prewrap": prewrap,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "envelope": envelope,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "aim": aim,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "spill": spill,
        "premature": premature,
        "collision": collision,
        "cup_disturb": cup_disturb,
        "rcv_keep": rcv_keep,
        "joint_margin": joint_margin,
        "ctrl": ctrl,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
```

## Review items

```
[36] 원장 0 1.34 < 1 2.37 < 1b 5.61 < 2 8.65 < 3 8.58 < 4 12.44 < 5 25.34 < 6 35.77 · 5/4 2.04 · 5v 12.90 · 3p 8.58 · 0p 1.34 · 1h 2.91
    ✗ 원장 단조가 아니다(골)
[T1] 참 궤적 스텝당 보상: A@0 +1.25 · B@150 +4.50 · C@270 +5.74 · D@350 +8.49 · E@410 +25.33 · F@500 +39.67 · G@640 +41.74 · H@760 +89.18 · 끝 +90.04
     한 스텝 최대 하락 -0.051 (t=366→367, 단계 D)
    ✗ 참 궤적에 절벽: approach_rcv -0.030, approach_src -0.030, aim +0.000
    ✗ 참 궤적 원장 T1~T5 중 실패가 있다
```

원장 [36] 2·3 행 정의(stage_ledger.py 원문, hand(m, cp, gap, closure, f_t, f_o, grasped)):
```
def hand(m, cp, gap, closure, f_t, f_o, grasped, palm_h=None, palm_rad=None, n=64, axes="envelope"):
add("2 옆·잡는 높이, 손끝 5 cm", scene(m, hand(m, base_s, 0.05, 0.0, 0, 0, False), hand(m, base_r, 0.05, 0.0, 0, 0, False), base_s, base_r))
add("3 컵이 손가락 사이, 손끝 1 cm", scene(m, hand(m, base_s, 0.01, 0.2, 0, 0, False), hand(m, base_r, 0.01, 0.2, 0, 0, False), base_s, base_r))
```
