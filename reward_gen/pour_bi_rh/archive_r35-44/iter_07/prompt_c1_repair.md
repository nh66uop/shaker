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


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d, r):
    """손끝-벽 거리 성형 (iter_03 c1r2 그대로, 척도는 컵 반지름 배수: 1.2 r, 0.22 r)."""
    return 0.5 * torch.exp(-d / (1.2 * r)) + 0.5 * torch.exp(-d / (0.22 * r))


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r, other_cup_pos):
    """컵 좌표(축 = cup_up)에서의 손끝 기하 (iter_03 c1r2 그대로).
    iter_06: 엄지 외 네 손가락 각각의 손끝-벽 거리 d_each (N,4) 도 돌려준다 (d_oth = d_each.min)."""
    EPS = 1e-6
    H_W = 2.0
    grasp_h = 0.25 * cup_mouth_z
    thumb_hi = 0.5 * cup_mouth_z

    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)
    rad_vec = rel - h_tip.unsqueeze(-1) * up
    rad = rad_vec.norm(dim=-1)
    gap = torch.clamp(rad - tip_r, min=0.0)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_each = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2)          # (N,4) 손가락별 손끝-벽 거리
    d_oth = d_each.min(dim=1).values

    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))

    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2 + 1e-12)

    to_other = other_cup_pos[:, :2] - cup_pos[:, :2]
    e_other = to_other / (to_other.norm(dim=-1, keepdim=True) + EPS)
    lat = (rel_m[:, :2] * e_other).sum(dim=-1)
    cross_d = torch.clamp(lat - tip_r, min=0.0)

    d_app = rad_m + H_W * torch.abs(h_m - grasp_h) + H_W * cross_d
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.7 * tip_r)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "d_each": d_each, "opp": opp, "d_grip": d_grip,
            "d_app": d_app, "thumb_band": thumb_band}


def _envelope_score(palm_pos, palm_axes, cup_pos, cup_up):
    """K7 인벨롭 방향 점수 (iter_04 그대로). 손가락 축 수평 인자(hand_level)도 함께 돌려준다."""
    EPS = 1e-6
    ENV_N, ENV_F = 0.67, 0.73
    NORMAL_TOL = 0.5
    NORMAL_PEN = 0.3
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    ax_dot = (f_dir * cup_up).sum(dim=-1)
    hand_level = _sat(1.0 - ax_dot * ax_dot)
    n_dot = (n_dir * cup_up).sum(dim=-1)
    normal_level = torch.exp(-(n_dot / NORMAL_TOL) ** 2)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    tgt = ENV_N * n_dir + ENV_F * f_dir
    tgt_h = tgt - (tgt * cup_up).sum(dim=-1, keepdim=True) * cup_up
    tgt_hat = tgt_h / (tgt_h.norm(dim=-1, keepdim=True) + EPS)
    cos_env = (e_in * tgt_hat).sum(dim=-1)
    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    score = hand_level * (0.25 * normal_level + 0.75 * dir_env) - NORMAL_PEN * (1.0 - normal_level)
    return _sat(score), hand_level


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos, other_cup_pos,
                    cup_r, cup_mouth_z, pose_ok, lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (두 팔 같은 식, 한 팔에만 붙는 항·부호·상수 없음).
    iter_06 변경 1: 다섯 손가락 참여 인자를 볼록하게 (엄지 외 손가락별 연속 접촉 평균의 제곱, 바닥 0.4 -> 0.3).
      각 1.5 N 기준 인자: 엄지+1 0.34 · +2 0.47 · +3 0.67 · +4(다섯) 0.96
      -> 한 손가락 더 붙일 때 몫 0.125 · 0.207 · 0.290 (마지막 손가락 몫이 가장 크다, 0 이 되지 않는다).
    iter_06 변경 2: 컵이 손 안(in_hand 또는 판정)에 있을 때 엄지 외 네 손가락 각각의 손끝-벽 거리로
      연속 점수(닿으면 접촉력 점수)를 만들고 같은 볼록 모양(평균의 제곱)으로 준다 -> "wrap".
    iter_05 변경 1·2 (다섯 손가락 인자를 판정 뒤 파지·파지 자세 유지·들기 인정에 곱함, 컵 좌표 파지 자세 유지) 그대로."""
    ORIENT_FLOOR = 0.3
    APP_FLOOR = 0.5
    APP_LIN_K = 20.0
    APP_NEAR_K = 3.0
    PALM_H_K = 0.15       # K7: 손바닥 높이 = 컵 원점 + 0.15 x cup_mouth_z
    SIDE_K = 1.5          # K7: 손바닥-컵 축 1.5 r
    LOW_H_K = 1.0         # 높이·축 거리 허용 척도 1 r
    LOW_SIDE_K = 3.0
    LOW_SIDE_FLOOR = 0.3
    LEVEL_FLOOR = 0.4
    LOW_W = 0.4
    FINGER_F = 0.7        # 손가락별 접촉 척도 [N]: 1.5 N -> 0.97, 1 N(env 판정 문턱) -> 0.89
    # iter_06 변경 1: 볼록 인자 바닥 0.4 -> 0.3 (두 손가락 몫 0.55 -> 0.34, 0 은 아님)
    FIVE_FLOOR = 0.3
    # iter_06 변경 2: 닿기 전 손끝-벽 점수. 상한 0.8 < 접촉 점수 0.97 -> 닿는 편이 항상 더 받는다.
    #   척도 0.5 r: 40 mm -> 0.05, 20 mm -> 0.19, 5 mm -> 0.56 (r = cup_radius 배수, 형상 상수 없음)
    WRAP_CAP = 0.8
    WRAP_K = 0.5
    HOLD_ANG_TOL = 0.4    # 축 내적 척도 (무차원). K7 손가락 축 z -0.11 -> 0.93, f04 -0.57 -> 0.13, 45 도 0.71 -> 0.04

    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r, other_cup_pos)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f = torch.clamp(finger_force, min=0.0)
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)
    f_o_max = f[:, 1:].max(dim=1).values

    # ---- iter_06 변경 1: 볼록 다섯 손가락 참여 (손가락별 연속 접촉 평균의 제곱)
    touch_each = torch.tanh(f[:, 1:] / FINGER_F)          # (N,4)
    part = touch_each.mean(dim=1)
    finger_fac = FIVE_FLOOR + (1.0 - FIVE_FLOOR) * part * part

    # ---- 높이
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)
    t_up = torch.tanh(h_pos / 0.02)
    lift_hold = g_f * t_up

    # ---- K7 방향 인자 (바닥 있음). 들린 뒤에는 1 로 풀림 -> 들기 경계 연속 (iter_04 그대로)
    env_score, hand_level = _envelope_score(palm_pos, palm_axes, cup_pos, cup_up)
    orient_fac = ORIENT_FLOOR + (1.0 - ORIENT_FLOOR) * env_score
    orient_app = APP_FLOOR + (1.0 - APP_FLOOR) * env_score
    o_grip = orient_fac + (1.0 - orient_fac) * lift_hold
    o_app = orient_app + (1.0 - orient_app) * lift_hold

    # ---- A. 접근 (iter_03 c1r2 그대로)
    d_app = geo["d_app"]
    reach_lin = _sat(1.0 - d_app / (APP_LIN_K * cup_r))
    reach_near = 1.0 - torch.tanh(d_app / (APP_NEAR_K * cup_r))
    reach_far = 0.7 * reach_lin + 0.3 * reach_near
    side_t = torch.maximum(_close_in(geo["d_thumb"], cup_r), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"], cup_r), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach_raw = 0.6 * reach_far + 0.4 * reach_tips

    # ---- A'. 잡는 높이로 내려오기 (손바닥 기준, 컵 좌표)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    rad_p = (to_cup - h_c * cup_up).norm(dim=-1)
    h_palm = -h_c.squeeze(-1)
    palm_h_err = torch.abs(h_palm - PALM_H_K * cup_mouth_z)
    palm_h_ok = torch.exp(-(palm_h_err / (LOW_H_K * cup_r)) ** 2)
    side_gap = torch.clamp(rad_p - SIDE_K * cup_r, min=0.0)
    side_gate = LOW_SIDE_FLOOR + (1.0 - LOW_SIDE_FLOOR) * torch.exp(-(side_gap / (LOW_SIDE_K * cup_r)) ** 2)
    level_fac = LEVEL_FLOOR + (1.0 - LEVEL_FLOOR) * hand_level
    level_fac = level_fac + (1.0 - level_fac) * lift_hold
    low_score = level_fac * palm_h_ok * side_gate

    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w
    near_gate = _sat((2.0 * cup_r - geo["d_grip"]) / cup_r)
    release = torch.maximum(torch.maximum(_sat(in_hand / 0.5), near_gate), g_f)
    openness = _sat(1.0 - hand_closure)
    open_fac = release + (1.0 - release) * (0.45 + 0.55 * openness)
    approach = open_fac * (o_app * reach_raw + LOW_W * low_score)

    # ---- B. 닫기 (그대로)
    near_grip = torch.exp(-(geo["d_grip"] / (1.1 * cup_r)) ** 2)
    close = o_grip * pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # ---- B'. iter_06 변경 2: 손 안 감싸기 (닿기 전 손가락별 기울기, 같은 볼록 모양)
    #      손가락별 점수 = max(0.8 exp(-d_i / 0.5 r), tanh(F_i / 0.7 N)). 손 안(in_hand 또는 판정)일 때만.
    #      o_grip 을 곱해 세로 집기로 감싸기 수입을 사지 못하게 한다 (f03 교훈).
    tip_near = WRAP_CAP * torch.exp(-geo["d_each"] / (WRAP_K * cup_r))
    per_finger = torch.maximum(tip_near, touch_each)
    wrap_part = per_finger.mean(dim=1)
    in_gate = torch.maximum(_sat(in_hand / 0.5), g_f)
    wrap = in_gate * o_grip * pose_ok * wrap_part * wrap_part

    # ---- C. 파지: 판정 뒤 수입에 다섯 손가락 참여 인자. 판정 순간 max(판정 뒤, 사전) -> 오르는 계단 유지
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    pre_q = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    post_q = torch.maximum((0.4 + 0.6 * pinch) * finger_fac, pre_q)
    contact_q = torch.where(g, post_q, pre_q)
    grasp = o_grip * pose_ok * contact_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C'. 컵 좌표 파지 자세 유지 (들거나 기울여도 풀지 않는다) x 다섯 손가락 참여 (iter_05 그대로)
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    n_perp = torch.exp(-((n_dir * cup_up).sum(dim=-1) / HOLD_ANG_TOL) ** 2)
    f_perp = torch.exp(-((f_dir * cup_up).sum(dim=-1) / HOLD_ANG_TOL) ** 2)
    side_ok = torch.exp(-((rad_p - SIDE_K * cup_r) / (LOW_H_K * cup_r)) ** 2)
    hold_pose = 0.35 * n_perp + 0.35 * f_perp + 0.15 * palm_h_ok + 0.15 * side_ok
    grip_pose = g_f * pose_ok * finger_fac * hold_pose

    # ---- D. 들기: 들기 인정(held)에 다섯 손가락 참여 인자. 방향 인자 없음
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * finger_fac * torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2) * calm
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- G. 교란 (유계, 래치 없음). finger_excess 는 가장 센 한 손가락 기준
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o_max - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / (3.5 * cup_r)) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "reach_raw": reach_raw, "close": close, "wrap": wrap, "grasp": grasp,
            "grip_pose": grip_pose,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 4번째 사이클 iter_06 (full-joint 26, 팔 절대 목표).
    iter_05 c1 위의 변경:
      1. 다섯 손가락 참여 인자를 볼록하게(평균의 제곱, 바닥 0.3): 마지막 손가락 몫이 가장 크다.
      2. 손 안(in_hand 또는 판정)일 때 엄지 외 네 손가락 각각의 손끝-벽 거리 연속 점수를 같은 볼록 모양으로 (finger_wrap).
      3. 그 밖은 iter_05 c1 그대로. 두 팔 같은 식·같은 무게. 항 25 개."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 -------------------------------------------------------------------------------------
    W_APPROACH = 3.0
    W_CLOSE = 0.8
    # iter_06: 감싸기 1.2 (두 팔 합 한 성분). 엄지·검지·중지 1.5 N 쥔 상태에서 약지·새끼 40 -> 20 mm 약 +0.19,
    # 20 -> 5 mm 약 +0.60 (두 손 합, o_grip = 1 가정, 추정). 닿기 전 상한 0.8^2 x 1.2 = 0.77/손 < 닿은 뒤 1.14/손.
    W_WRAP = 1.2
    W_GRASP = 4.0          # 판정 뒤 크게
    W_GRIP_POSE = 2.5      # iter_05 그대로 (볼록 인자와 함께 한 손가락 몫을 키운다)
    W_LIFT = 5.5
    W_RUSH = 0.06
    W_ALIVE = 0.5
    UPRIGHT_TOL = math.radians(20.0)
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20   # 사용자: 소스 들기 높이 그대로
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) ----------------------------------------------------------------
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

    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    # ---- 팔별 항 (두 팔 같은 식·같은 무게) -----------------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, pose_ok_src,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, pose_ok_rcv,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones

    # ---- A 접근 / B 닫기 / B' 감싸기 / C 파지 / C' 파지 자세 -------------------------------------------
    approach_src = W_APPROACH * src["approach"]
    approach_rcv = W_APPROACH * rcv["approach"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    finger_wrap = W_WRAP * (src["wrap"] + rcv["wrap"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
    grip_pose_src = W_GRIP_POSE * src["grip_pose"]
    grip_pose_rcv = W_GRIP_POSE * rcv["grip_pose"]

    # ---- D 들기 (소스 높게, 리시버 낮게) ------------------------------------------------------------
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- E 배치 ------------------------------------------------------------------------------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad
    place = converge_cups + pour_dir + rcv_keep

    # ---- E 조준 ------------------------------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 --------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F 회전 (8.0): 창 안·조준 품질 x 회전 진척. 각도 처방 없음 -----------------------------------
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog * rcv_gate

    # ---- F 배달 (알당 30, 증분만): 주역 ------------------------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- G 흘림 ------------------------------------------------------------------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill = -40.0 * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- G 충돌 (실기 안전) -----------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- G 컵 교란: 두 팔 같은 무게 ---------------------------------------------------------------
    tilt_pen_src = (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = 1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    dist_src = 0.5 * src["slide"] + 0.3 * tilt_pen_src + 0.3 * src["one_sided"] \
        + 0.2 * src["palm_push"] + 0.3 * src["fling"] + W_RUSH * src["rush"]
    dist_rcv = 0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv + 0.3 * rcv["one_sided"] \
        + 0.2 * rcv["palm_push"] + 0.3 * rcv["fling"] + W_RUSH * rcv["rush"]
    cup_disturb = -(dist_src + dist_rcv)

    # ---- G 제어 정규화 (cup_grasp 수준). 팔 액션 크기 비용 없음 (a=0 = 시작 자세) ----------------------
    a = ctx.actions
    pa = ctx.prev_actions
    a_arm = torch.cat([a[:, 0:7], a[:, 13:20]], dim=-1)
    pa_arm = torch.cat([pa[:, 0:7], pa[:, 13:20]], dim=-1)
    a_hand = torch.cat([a[:, 7:13], a[:, 20:26]], dim=-1)
    pa_hand = torch.cat([pa[:, 7:13], pa[:, 20:26]], dim=-1)
    da_arm = a_arm - pa_arm
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (
        torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
        + torch.tanh(torch.clamp(qd_rcv.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0))
    ctrl_arm = -(rate_arm + speed_arm + runaway)
    da_hand = a_hand - pa_hand
    ctrl_hand = -0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "finger_wrap": finger_wrap,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grip_pose_src": grip_pose_src,
        "grip_pose_rcv": grip_pose_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "place": place,
        "aim": aim,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "posture_margin": posture_margin,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "spill": spill,
        "premature": premature,
        "collision": collision,
        "cup_disturb": cup_disturb,
        "ctrl_arm": ctrl_arm,
        "ctrl_hand": ctrl_hand,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
```

Observations from watching the trained policy:
# pour_bi_rh 보상 설계 브리프 (2026-09-29 판 — 4번째 사이클 iter_03, full-joint 제어·팔 절대 목표)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트에 통째로 들어간다.**

이 트랙은 세 번 리셋했다. 1~12 라운드 `archive_r1-12`, 13~24 라운드 `archive_r13-24`, 3번째 사이클 iter_00~09 는 `archive_r25-34`.
지금은 4번째 사이클의 iter_00 이다. **바뀐 것은 제어다**(palm 지령 + fabric → 양팔 full-joint). 보상은 3번째 사이클 마지막 코드(iter_09)를
바탕으로 구조를 다시 짠다 — 새로 발명하지 않는다. 사용자 판단(09-28): "제어만 바뀌지 reward 문제라고 생각하진 않는다."

## 1. 과제와 고정 조건 (바꿀 수 없다)

- 오른팔(소스, `src_*`)이 비드 20 개가 든 컵을, 왼팔(리시버, `rcv_*`)이 빈 컵을 잡아 든다. 소스를 기울여 비드를 리시버 컵에 옮긴다.
- 성공(env 가 계산, 보상이 재정의 못 함): 담김 ≥ 0.80 · 흘림 ≤ 0.25 · 두 컵 파지 유지 · 받는 컵 ≤ 20° · 20 스텝 유지 · 컵이 포개지지 않음.
- **액션 26 = 팔마다 13** — [팔 관절 절대 목표 7(a=0 시작 자세·±1 관절 한계, 0.3 rad/s 상한 — iter_03 부터), 손 구동 관절 절대 목표 6]. 로봇 설명 절이 정의한다.
- **물리 이상은 env 가 종료한다(iter_03 부터)**: 컵 > 3 m/s·> 30 rad/s, 손 접촉 > 200 N 2 스텝, 컵이 스폰에서 0.5 m 밖, 컵 낙하, mimic 폭주. 종료 벌점 항은 두지 않는다.
- 스폰(컵 원점, env-local): 소스 (0.382, −0.159, 0.268) · 리시버 (0.378, +0.165, 0.268). 시작 손바닥은 컵 원점에서 수평 약 16 cm·위 약 15 cm.
- r = `ctx.cup_radius` = 0.028 m. 소스 컵 원점 → 림 0.054 m. 제어 주기 1/60 s, 에피소드 900 스텝.

## 2. 단계 원칙 (사용자, 09-28)

접근 → 손가락 사이로 컵 → 파지 → 들기 → 붓기 순서로 **gradient 가 다음 단계로 이어져야 한다.
파지를 하고 들려고 할 때 보상이 떨어지면 안 된다.** 한 단계 앞으로 가는 것이 그 자리에 머무는 것보다 항상 스텝당 이득이어야 하고,
머물기만 해도 벌리는 자리(주차)가 없어야 한다. 검사는 `reward_gen/pour_bi_rh/stage_ledger.py`(결정론 상태 원장)로 한다.

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

## 4. 3번째 사이클(iter_00~09)에서 확정된 교훈

| 라운드 | 사실 | 교훈 |
|---|---|---|
| i03 | 표면 거리 접근은 입구 원판을 표면으로 쳐서, 손바닥이 입구 위에 있으면 거리 1.7 cm·접근 수입 0.75 | 접근은 **컵 원점 기준**, 옆에서(수평 성분 가중) |
| i04 | 깔때기 목표가 "잡는 높이보다 4.5 cm 위에서 도착 후 내려감"이라 위에서 끼우는 동작을 가르쳤다, 파지 0 | 손바닥을 먼저 내리고 테이블과 수평으로 접근 |
| i07 | 손을 쥔 채(폐쇄도 0.63) 다가갔다 | 다가가는 동안 손을 펴게 하는 항이 필요 |
| i08 | 상태형 "편 손" 가산이 컵 12 cm 앞 주차 자리를 만들었다(0.13~0.16/스텝) | 편 손 가산도 **차분형**(편 채 다가갈 때만) |
| f00 (4번째 사이클 첫 iter_00, full-joint) | e0~470: 접근 보상(손바닥 속도 차분) −0.001~−0.018, 손바닥-컵 21~43 cm 왕복, 컵 근처 0(i09 같은 epoch 0.41), 손 폐쇄도 0.59~0.89 유지, 팔 제어 벌점 −0.14~−0.19(가장 큰 항), 팔 목표가 관절 한계 2 % 안 0.02→0.17. 결정론 검사: 시작 자세 6 cm 다가감 +0.002, 15→6 cm −0.122, 편 손−쥔 손 0, 팔 0.3 rad/s 비용 0.378 | **관절 제어에서는 속도 차분 접근이 신호가 되지 않는다** — 상태형 접근과 작은 제어 벌점(cup_grasp) |
| f07 (iter_06 c1, f05 ep500 이어받음) | e1141: 왼손 닿은 손가락 4.17(새끼 0.49), 들기 1.4. ep1000 트레이스: 왼손 다섯 손가락 모두 벽 바깥(새끼 68 %)·컵 19 cm·엄지 외전 +0.93. 오른손 엄지 축 0.57 r(입구 안)·4.7 cm·12.2 N, 중지 −0.46 반쯤 편 채 2.1 r, 엄지 외전 −0.25, 컵 안 들림 | **볼록 다섯 손가락 인자·손가락별 감싸기는 작동한다(왼손).** 손끝-벽 거리를 max(축 거리 − r, 0) 으로 재면 입구 안 손끝도 '벽에 닿음', 입구 안벽 접촉력도 파지로 인정 — 엄지 걸기가 반복된다(f05 e750·f07 e1000). 걸기에서 엄지를 빼는 순간 −1.47(골) |
| f06 (iter_05 c1, f05 ep500 이어받음) | e108: 근처 0.86/0.90 유지(이어받기 성공), 쥠 0.02/0.01, 쥔 스텝 닿은 손가락 2.0/2.0 | **다섯 손가락 인자가 평균(선형)이면 손가락 하나마다 몫이 같아 어려운 마지막 손가락을 건너뛰기 쉽다(사용자 지적).** 닿기 전 손가락별 거리 신호도 없다 |
| f05 (iter_04 c1) | e2457: 두 손 파지(0.70/0.78)·들기(소스 25~30 cm·리시버 9 cm)·기울기 최대 17°·회전 수입 0.27~0.55·비드 0. 좌우 미러 관절 차 ≤0.24 rad(e750) — f04 의 j5 굴림 없음. ep2350 트레이스: 오른손 엄지 1.5 N·중지 1.2 N 두 손가락만(검지 2.1 r 떨어짐), 손바닥 축 2.8 r, 들고 나면 법선 z −0.67, 에피소드당 놓침 19 회(왼손 5). 왼손은 엄지·검지·중지·약지 4 손가락 | **env 파지 판정(엄지 + 다른 손가락 하나)과 min(엄지, 나머지 합) 집기 수입은 두 손가락 집기를 다섯 손가락과 같게 친다(원장 20.84 대 20.36).** 들린 뒤 방향 인자가 풀리면 손이 컵 기준으로 틀어져도 손해가 없다(기울기 45° 차 1.07) |
| f04 (iter_03 c1r2, 팔 절대 목표) | e2256: 왼손 파지 0.85·컵 8 cm·들기 4.1 — 파지→들기 경계에서 총보상이 오르며 이어짐(8.2→10.8→14.2), bounds_loss ≤0.6·팔 한계 ≤0.1·물리 종료 0. ep2050 트레이스: 왼손 손바닥 컵 원점 위 3.4 cm·손가락 축 z −0.57(엄지 4.0·검지 2.1·중지 1.6 N). 오른손 파지 0.22·들기 0 — j5(아래팔 롤) +0.46→+1.50(한계 1.57)로 왼손을 미러한 방향(−0.87)과 반대로 굴려 손바닥이 위(법선 z +0.72)를 보는 받침 자세 | **보상은 좌우 미러 대칭이어도(원장 미러 자세 팔별 차 0) 정책은 한쪽만 다른 해로 갈 수 있다.** 받침 자세 불이익(두 손 합 1.25)이 작으면 그 자리에 머문다. 사용자 영상 판정: 왼손은 더 낮게, 오른손은 한 바퀴 회전 |
| f03 (iter_02, 증분 팔) | e5414: 왼손 근처 0.86·쥠 0.66~0.75 — 파지 자세는 위에서 세로 집기(손가락 축 z −0.95, 엄지 6.0·검지 3.3 N, 손바닥 0). 들기 보상 e3300~3500 0.53 뒤 0: 트레이스에서 든 10 env 중 9 곳이 집기 미끄러짐(총 2~3.5→0.3~0.8), 쥔 채 든 1 곳은 2.61→3.04. 오른손: ep1750 옆 접촉에 컵 16.7° 기울고 교란 −0.24→−0.38 뒤 물러남, ep5400 팔 6/7 관절 |mu| 1.4~2.4 로 한계 96 % 고착 | **닫기·파지가 손 방향과 무관하면 방향 인자를 입구에만 걸어도 세로 집기를 산다.** 두 손가락 집기는 들 때 미끄러진다. 파지 직전 상태 수입이 크면(파지/직전 1.38 배) 파지 판정 전에 머문다 — 2번째 사이클 iter_04 는 1.86 배 |
| i09 | e300~450 오른손 컵 근처 0.43→0.54, **파지 0.06→0.11** 뒤 e450~500 붕괴(근처 0.15·파지 0.02). 같은 구간 grasp+close +0.23 대 cup_disturb −0.26·contact_push −0.08·rush −0.06, 스텝 보상 0.60→0.49 | **파지 시도 벌점이 파지 수입을 넘으면 컵을 떠난다.** `cup_disturb` 의 slide 는 (1−lifted)·tanh(스폰에서 밀린 거리/3 cm) 상태형이라, 한 번 민 컵은 손을 떼도 들기 전까지 매 스텝 비용을 낸다(원장 0p: 0.51 → 0.22) |

이전 사이클 핵심(유지): 접근은 차분, 붓기 허가는 컵 반지름 창(i13) · 방향 신호 필수(i08·1차) · 관절 한계 직접 강벌점 금지(i15, corr +0.90) ·
진행형 항의 비대칭 clamp 금지(i20) · 기울임 수입이 없으면 들고 서 있는다(i21) · 조준 항이 축 회전까지 지불하면 안 된다(i22) ·
주차 함정: 상태 비례 수입이 배달보다 크면 붓지 않는다(회전·배달 수입이 성공 궤적 스텝 수입의 절반 이상).

## 5. iter_00 요구 — full-joint 에 맞춘 재편 (09-28 두 번째 판)

사용자(09-28): "이전과 달리 full-joint 방식이니까 reward 를 재개편해." 첫 iter_00(t2r_rh4_f00)은 palm 지령 시절의 형태를 그대로 옮겨 실패했다(4절 f00 행).

### 5-0. 원칙: 관절 제어에서는 **상태에 기울기가 있어야** 한다
- palm 지령 시절에는 액션 = 손바닥 속도라서 "컵 쪽 속도" 같은 차분(속도) 보상이 액션과 곧바로 이어졌다. full-joint 에서는 손바닥 속도가
  관절 7개의 비선형 합이고 EMA 로 약 10 스텝 늦다. 무작위 탐색에서 다가감과 멀어짐이 상쇄돼 기댓값이 0 이다.
- 그래서 접근·편 손은 **상태형**(현재 자세의 함수)으로 준다: 가까울수록, 옆일수록, 손이 펴져 있을수록 스텝당 더 받는다.
- 주차(i08)는 차분형으로 막지 않고 **단계 원장**으로 막는다: 멈춰 있는 어느 자리도 다음 단계 자리보다 싸야 한다.

> **iter_03(09-29 사용자 결정 B)**: 5-1 의 손바닥 원점 기준 접근·입구 대신, 2번째 사이클 iter_04(양손 파지 첫 성공)의 손끝 기준 파지 사다리를 쓴다. 세부는 뒤의 iter_03 노트.

### 5-1. 접근·입구 (같은 관절 제어로 컵 집기에 성공한 cup_grasp 구조를 따른다)
- 원점 기준 거리 보상: 먼 척도 + 가까운 척도의 합(예: 1−tanh(d/먼 척도), 1−tanh(d/가까운 척도)), 높이 차는 수평의 2 배로 센다(옆·잡는 높이 선호,
  위에서 내려오는 경로는 덜 받는다). 무게는 들기 전 합 예산 안에서. 파지가 서면 사라진다.
- 편 손: 컵에서 떨어져 있는 동안 거리 보상에 손 펴짐 인자를 곱한다(예: 0.45 + 0.55 × 펴짐). 손가락 사이에 컵이 오면 인자를 풀어 닫기를 막지 않는다.
- 손가락 방향·손가락 사이(엄지 대향)·잡는 높이로 내려오기는 원거리부터 연속으로 오르는 상태형(곱 사슬 금지).
- 컵 근처에서 붙는 비용(입구 위 정지·손바닥 밀어 넣기·급한 팔)이 **다가가는 수입보다 작아야** 한다 — 15 cm → 6 cm 가 스텝당 오른다.
- 원장 통과 조건: 0 < 1 < 1b(옆 12 cm 편 손 정지) < 2 < 3 < 4 < 5 ... (단조), 3p ≥ 3, 0p ≥ 0, 1h(반쯤 쥔 손) < 1b.

### 5-2. 제어 벌점은 cup_grasp 수준 (매끄러움은 EMA·속도 상한이 맡는다)
- cup_grasp: 액션 변화율 0.02·tanh(평균 변화²/0.1) + 팔 속도 0.12·tanh(|qd|/1 rad/s) + 3 rad/s 초과 폭주 비용.
- **iter_03 부터 팔 액션 크기 비용 금지**: 팔 액션이 절대 목표(a=0 = 시작 자세)라 |a| 비용은 시작 자세로 끌어당기는 항이 된다.
- 기준: 무작위 ±1 액션 비용 ≤ 0.06/스텝, 팔 0.3 rad/s 비용 ≤ 0.08/스텝. 손 채널 ±1 은 비용 0(절대 목표의 정상 명령).
- 액션 반전(flip)·|a|>0.9 포화 벌점은 관절 증분에서는 탐색을 억누른다 — 두지 않는다.

### 5-3. 유지하는 것 (첫 iter_00 에서 검증됨)
- 교란 벌점은 컵이 **움직이는 속도**로만(밀린 상태 자체는 비용 없음). 파지 전 작은 기울기는 싸게.
- 파지(닫기 1.0 → 양쪽 접촉 2.0), 들기(소스 높게·리시버 낮게, W 5.5), 배치·조준·회전·배달(붓는 중 회전·배달 몫 ≥ 절반), 흘림·조기 기울임·충돌.
- 들기 전 상태 합 < 들기 빠른 성분(골 없음). 항 25 개 이하.

### 5-4. 하지 말 것
- 손바닥 속도에만 의존하는 접근 수입, 목표 각도·대역·각도 상한 처방, 스폰 기준 만남 통로, 관절 한계 강한 직접 벌점, 새 절대 거리 상수(높이 허용과
  접근 척도는 컵 반지름 배수 또는 명시 근거), 자산 치수 하드코딩, 곱 사다리, 손 채널 포화 벌점.

## 6. 평가

- 틱은 단계 관점: near_rate → cup_in_pocket·oppose → closure → grasped → cup_lift → converge·aim → pour. 단계가 오를 때 보상·에피소드 길이가 떨어지면 골.
- full-joint 지표: `joint/{src,rcv}_arm_limit_frac`, `_arm_track_err_deg`, `_hand_track_err_deg`.
- 성공 판정은 결정론 평가(담김 ≥ 0.8·최대 기울기 순간 방향 오차 < 30°·주둥이가 입구보다 위).

## 7. 시작

fresh(액션·관측 의미가 바뀌었다). 웜스타트 체크포인트 없음. seed 42.
# iter_07 (4번째 사이클) 운영 노트 — 2026-09-29

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다(K7, f03~f07 행).

## 사용자 결정 (09-29)
- "A 를 하면 다시 돌아올 가능성은? 답습할 것 같은데" → "B 로 진행": f05 ep500(대칭 접근 완료, 파지 습관 전)에서 새 보상으로 다시 시작.
- 유지: 다섯 손가락 인벨롭·기울이는 동안 자세 유지·두 팔 대칭·소스 들기 높이 그대로.
- 이어서 사용자: "접근을 잘못해서도 이유가 될 텐데, 옆으로 접근해야 하는데 위에서 아래로 접근해서" · "palm sensor 로컬 +z 가 중력과 90 도여야 하고, 인벨롭 되게 충분히 낮게 내려간 뒤 접근해야지".

## 측정 사실 (f07 ep1000 결정론 재생 64 env, 쥔 동안 중앙값)
- 왼손: 엄지 축 1.53 r·4.4 cm·5.5 N(벽 바깥), 검지·중지·약지 99 %·새끼 68 % 접촉, 컵 19.3 cm 들고 유지. 볼록 인자·감싸기 항이 의도대로 작동.
- 오른손: 엄지 축 0.57 r(입구 안)·높이 4.7 cm(입구 5.4 cm)·12.2 N, 검지 0.96 r·1.8 N, 약지 0.53, 중지 0.01(반쯤 편 채 2.07 r), 새끼 0.01. 컵 들림 0 %.
  손 액션 중앙: 엄지 외전 −0.25(왼손 +0.93), 중지 −0.46.
- 같은 엄지 걸기가 f05 e750 에도 있었다(엄지 축 0.58 r·6.2 cm).
- 코드: 손끝-벽 거리 gap = max(축 거리 − 벽 반경, 0) — 입구 안(축 거리 < 반경) 손끝도 벽에 닿은 것(0)으로 친다. env 파지 판정·집는 힘·다섯 손가락 인자·감싸기는 자기 컵 접촉력이면 입구 안벽 접촉도 인정한다.
- iter_06 원장: 걸기 10.97 → 엄지를 빼서 놓음 9.49(−1.47, 골) → 엄지 바깥으로 같은 세 손가락 다시 쥠 14.41 → 다섯 손가락 인벨롭 24.67 (두 손 합).

## 측정 사실 — 접근 경로 (f07 ep1000·f05 ep500 결정론 재생 64 env, 쥐기 전)
- 시작 손바닥: 컵 원점에서 수평 15~20 cm·높이 +15 cm(입구 +10 cm). 로컬 +z(손바닥 법선) z 성분 0.00, 손가락 길이 축(로컬 +y) z −0.42(아래로 25 도).
- 손끝 하나가 처음 컵 축 1.3 r 안에 든 순간: 손바닥이 입구보다 +3.3 cm(f07 오른손)·+8.0(f07 왼손)·+7.1(f05 오른손)·+4.8(f05 왼손) 위, 엄지는 입구보다 +3.7~+10.4 cm 위, 64 env 전부 엄지가 입구 위. 손바닥 수평 거리 11~13 cm.
- 이때 로컬 +z 의 z 성분 |0.03~0.13|(거의 수평), 손가락 길이 축 z −0.3~−0.6(아래로 17~37 도). 쥔 뒤 오른손 로컬 +z z −0.18(f07)·−0.31(f05).
- 쥐기 전 스텝 가운데 손끝이 입구 위(축 1 r 안·입구보다 높음)에 있는 비율: f07 오른손 0.19·왼손 0.25, f05 오른손 0.44·왼손 0.09 — f05 ep500(이어받기 기준점)도 위에서 덮는다.
- iter_06 코드로 잰 진입 순간 자세(check_iter07 [47]~[49], 두 손 합): 같은 수평 위치에서 잡는 높이로 내리면 +1.8, 손가락 축까지 수평이면 +2.3. 그런데 잡는 높이·손가락 수평에서 로컬 +z 를 17 도 아래로 굴리면 오히려 +0.37(f07)·+0.06(f05) 더 받는다. 60 스텝 경로 원장 [23]: 먼저 내리고 수평 진입 − 대각선 = +8.9(스텝당 0.15).
- iter_06 코드의 손바닥 높이 점수는 exp(−(높이 오차/1 r)²) — 잡는 높이에서 5 cm 넘게 위면 0 에 가깝다(시작 높이는 14 cm 위). 접근 거리 d_app 은 손끝 중점 높이를 쓰므로 손가락을 아래로 늘어뜨리면 손바닥이 높아도 맞춰진다.

## iter_07 에서 할 것 (이전 코드 iter_06 c1 위의 변경)
1. 손끝-벽 거리는 컵 바깥 면 기준: 손끝이 입구 안(축 거리 < 벽 반경이면서 입구 높이 근처)에 있으면 벽에서 먼 것으로 친다(접근·감싸기·엄지 대역).
2. 입구 안에 든 손끝의 컵 접촉력은 파지 수입·다섯 손가락 인자·파지 자세 유지·들기 인정에서 세지 않는다(연속 가중, 바깥·입구 아래일수록 1).
3. 옆 접근(사용자): 손바닥이 잡는 높이(인벨롭 높이)까지 먼저 내려간 뒤 수평으로 들어가게 한다. 손바닥 높이 점수가 시작 높이(14 cm 위)에서도 기울기를 갖게 하고, 컵 근처 접근 수입(손끝-벽·손 안 판정)은 손바닥이 높을수록 연속으로 줄인다. 두 팔 같은 식.
4. 손 방향(사용자): 손바닥 센서 로컬 +z(법선)는 중력과 90 도, 손가락 길이 축도 수평. 굴린 법선(z −0.3)이 수평 법선보다 받는 일이 없게, 손가락 길이 축 −25~−37 도 → 수평의 이득이 분명하게.
5. 검사 [46]: 걸기 → 엄지를 빼서 놓음이 −0.05 보다 나빠지지 않고(골 없음), 엄지 바깥 다시 쥠이 걸기보다 3.0 이상. [45] 걸고 3 cm 든 파지가 인벨롭 들기보다 12 이상 낮음.
6. 검사 [47]~[49]·[23] (f07·f05 진입 순간 실측 자세): 잡는 높이로 내림이 기록 자세보다 두 손 합 0.4 이상(손가락 수평까지 0.6), 13→9→5 cm→잡는 높이 각 칸 +0.10 이상, 잡는 높이에서 손가락 축 수평 +0.4 이상, 법선 수평이 17 도 굴린 것보다 +0.2 이상, 먼저 내리는 60 스텝 경로가 대각선보다 15 이상.
7. 이전 성질 유지([36]·[38]~[44]·[15]~[37]), 두 팔 같은 식·같은 무게. 그 밖은 iter_06 c1 그대로.

## Previous response

```python
import torch
import math


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d, r):
    """손끝-벽 거리 성형 (iter_03 c1r2 그대로, 척도는 컵 반지름 배수: 1.2 r, 0.22 r)."""
    return 0.5 * torch.exp(-d / (1.2 * r)) + 0.5 * torch.exp(-d / (0.22 * r))


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r, other_cup_pos):
    """컵 좌표(축 = cup_up)에서의 손끝 기하.
    iter_07 변경 1: 손끝-벽 거리는 컵 바깥 면 기준. 입구 안(축 거리 < 벽 반경, 입구 높이 근처) 손끝은
      벽에서 먼 것(2 r)으로 친다 -> 접근(d_thumb·d_oth)·감싸기(d_each)·엄지 대역에 모두 적용.
      입구 안 가중 w_in 을 손가락별로 돌려주고, 바깥 가중 out_w = 1 - w_in 은 접촉력 가중(변경 2)에 쓴다."""
    EPS = 1e-6
    H_W = 2.0
    grasp_h = 0.25 * cup_mouth_z
    thumb_hi = 0.5 * cup_mouth_z
    # iter_07 변경 1: 입구 안 판정 (전부 컵 반지름 배수)
    #   축 거리: 0.9 r 부터 0.6 r 까지 0 -> 1 (가설: 손끝 좌표가 바깥 벽 접촉에서도 r 을 약간 밑돌 수 있어 여유 0.1 r)
    #   높이: 입구 - 1.5 r ~ 입구 + 1.0 r 에서 1, 양끝 0.5 r 에 걸쳐 0 (가설: 아래쪽 테이퍼 몸통 오판 방지)
    #   f07 오른손 엄지(0.57 r, 입구 -0.7 cm)·f05 엄지(0.58 r, 입구 +0.8 cm) 둘 다 w_in = 1
    IN_RAD_LO, IN_RAD_W = 0.9, 0.3
    IN_H_LO, IN_H_HI, IN_H_W = 1.5, 1.0, 0.5
    IN_FAR = 2.0          # 입구 안 손끝의 유효 벽 거리 = 2 r (_close_in 0.09, 감싸기 0.015)

    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)
    rad_vec = rel - h_tip.unsqueeze(-1) * up
    rad = rad_vec.norm(dim=-1)
    gap_raw = torch.clamp(rad - tip_r, min=0.0)

    w_rad = _sat((IN_RAD_LO * tip_r - rad) / (IN_RAD_W * tip_r))
    w_h = _sat((h_tip - (cup_mouth_z - (IN_H_LO + IN_H_W) * tip_r)) / (IN_H_W * tip_r)) \
        * _sat((cup_mouth_z + (IN_H_HI + IN_H_W) * tip_r - h_tip) / (IN_H_W * tip_r))
    w_in = w_rad * w_h                                             # (N,F) 1 = 입구 안
    gap = (1.0 - w_in) * gap_raw + w_in * (IN_FAR * tip_r)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_each = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2)          # (N,4) 손가락별 손끝-바깥 벽 거리
    d_oth = d_each.min(dim=1).values

    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))

    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2 + 1e-12)

    to_other = other_cup_pos[:, :2] - cup_pos[:, :2]
    e_other = to_other / (to_other.norm(dim=-1, keepdim=True) + EPS)
    lat = (rel_m[:, :2] * e_other).sum(dim=-1)
    cross_d = torch.clamp(lat - tip_r, min=0.0)

    d_app = rad_m + H_W * torch.abs(h_m - grasp_h) + H_W * cross_d
    # iter_07 변경 1: 엄지가 입구 안이면 엄지 대역 0 (연속)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.7 * tip_r) * (1.0 - w_in[:, 0])
    return {"d_thumb": d_thumb, "d_oth": d_oth, "d_each": d_each, "opp": opp, "d_grip": d_grip,
            "d_app": d_app, "thumb_band": thumb_band, "out_w": 1.0 - w_in}


def _envelope_score(palm_pos, palm_axes, cup_pos, cup_up):
    """K7 인벨롭 방향 점수.
    iter_07 변경 4: 손가락 길이 축 수평 인자를 1 - dot^2 에서 exp(-(dot/0.4)^2) 로
      (K7 z -0.11 -> 0.93, -25 도 0.42 -> 0.33, -37 도 0.60 -> 0.11; 이전 0.99/0.82/0.64),
      법선 허용 0.5 -> 0.3 (z -0.29 굴림 -> 0.39, 이전 0.71). 법선 수평 인자도 돌려준다."""
    EPS = 1e-6
    ENV_N, ENV_F = 0.67, 0.73
    NORMAL_TOL = 0.3      # iter_07: 0.5 -> 0.3 (굴린 법선이 수평 법선보다 받지 못하게)
    NORMAL_PEN = 0.3
    FINGER_TOL = 0.4      # iter_07: 손가락 축 z 성분 허용 (HOLD_ANG_TOL 과 같은 척도)
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    ax_dot = (f_dir * cup_up).sum(dim=-1)
    hand_level = torch.exp(-(ax_dot / FINGER_TOL) ** 2)
    n_dot = (n_dir * cup_up).sum(dim=-1)
    normal_level = torch.exp(-(n_dot / NORMAL_TOL) ** 2)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    tgt = ENV_N * n_dir + ENV_F * f_dir
    tgt_h = tgt - (tgt * cup_up).sum(dim=-1, keepdim=True) * cup_up
    tgt_hat = tgt_h / (tgt_h.norm(dim=-1, keepdim=True) + EPS)
    cos_env = (e_in * tgt_hat).sum(dim=-1)
    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    score = hand_level * (0.25 * normal_level + 0.75 * dir_env) - NORMAL_PEN * (1.0 - normal_level)
    return _sat(score), hand_level, normal_level


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos, other_cup_pos,
                    cup_r, cup_mouth_z, pose_ok, lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (두 팔 같은 식, 한 팔에만 붙는 항·부호·상수 없음).
    iter_07 변경 2: 입구 안 손끝의 접촉력은 파지 수입·다섯 손가락 인자·파지 자세 유지·들기 인정에서 빼고
      (손가락별 바깥 가중 out_w 를 곱함), 판정 인정 g_eff = 판정 x 엄지 바깥 가중 x (바깥 손가락 1 N 접촉) 으로 연속화.
    iter_07 변경 3: 손바닥 높이 점수에 선형 원거리 성분(6 r) -> 시작 높이(14 cm 위)에서도 기울기.
      손끝-벽 접근 수입과 손 안 판정은 손바닥이 잡는 높이보다 높을수록 연속으로 줄인다(high_fac).
    iter_07 변경 4: 낮추기 점수의 수평 인자에 법선 수평도 곱한다(hand_level x normal_level).
    그 밖은 iter_06 c1 그대로."""
    ORIENT_FLOOR = 0.3
    APP_FLOOR = 0.5
    APP_LIN_K = 20.0
    APP_NEAR_K = 3.0
    PALM_H_K = 0.15       # K7: 손바닥 높이 = 컵 원점 + 0.15 x cup_mouth_z
    SIDE_K = 1.5          # K7: 손바닥-컵 축 1.5 r
    LOW_H_K = 1.0         # 높이·축 거리 허용 척도 1 r (근거리)
    # iter_07 변경 3: 원거리 선형 척도 6 r (= 16.8 cm): 13 -> 9 -> 5 -> 0 cm 위 각 칸 +0.12 이상 (0.5 가중 뒤)
    LOW_H_FAR_K = 6.0
    LOW_SIDE_K = 3.0
    LOW_SIDE_FLOOR = 0.3
    LEVEL_FLOOR = 0.4
    LOW_W = 0.4
    # iter_07 변경 3: 잡는 높이보다 위인 손바닥 -> 근처 접근 수입 감쇠. 1.5 r 척도, 바닥 0.2 (기울기 유지)
    #   f07·f05 진입 순간(입구 +3.3~+8.0 cm = 잡는 높이 +7.9~+12.6 cm) -> 0.2 + 0.8 x (0.03 이하)
    HIGH_K = 1.5
    HIGH_FLOOR = 0.2
    FINGER_F = 0.7        # 손가락별 접촉 척도 [N]: 1.5 N -> 0.97, 1 N(env 판정 문턱) -> 0.89
    FIVE_FLOOR = 0.3
    WRAP_CAP = 0.8
    WRAP_K = 0.5
    HOLD_ANG_TOL = 0.4
    GRASP_TOUCH = math.tanh(1.0 / FINGER_F)   # 1 N 바깥 접촉 점수 -> 판정 인정 1

    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r, other_cup_pos)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    out_w = geo["out_w"]                                   # (N,F) 1 = 바깥·입구 아래
    f_raw = torch.clamp(finger_force, min=0.0)
    # ---- iter_07 변경 2: 입구 안 손끝 접촉력 제외 (연속 가중)
    f = f_raw * out_w
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)

    touch_each = torch.tanh(f[:, 1:] / FINGER_F)          # (N,4) 바깥 접촉만
    part = touch_each.mean(dim=1)
    finger_fac = FIVE_FLOOR + (1.0 - FIVE_FLOOR) * part * part
    oth_ok = _sat(touch_each.max(dim=1).values / GRASP_TOUCH)
    g_eff = g_f * out_w[:, 0] * oth_ok                    # 엄지 걸기(입구 안) 판정은 0 에 가깝다

    # ---- 높이
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)
    t_up = torch.tanh(h_pos / 0.02)
    lift_hold = g_eff * t_up

    # ---- K7 방향 인자 (바닥 있음). 들린 뒤에는 1 로 풀림
    env_score, hand_level, normal_level = _envelope_score(palm_pos, palm_axes, cup_pos, cup_up)
    orient_fac = ORIENT_FLOOR + (1.0 - ORIENT_FLOOR) * env_score
    orient_app = APP_FLOOR + (1.0 - APP_FLOOR) * env_score
    o_grip = orient_fac + (1.0 - orient_fac) * lift_hold
    o_app = orient_app + (1.0 - orient_app) * lift_hold

    # ---- 손바닥 높이 (컵 좌표)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    rad_p = (to_cup - h_c * cup_up).norm(dim=-1)
    h_palm = -h_c.squeeze(-1)
    palm_h_err = torch.abs(h_palm - PALM_H_K * cup_mouth_z)
    palm_h_near = torch.exp(-(palm_h_err / (LOW_H_K * cup_r)) ** 2)
    palm_h_far = _sat(1.0 - palm_h_err / (LOW_H_FAR_K * cup_r))
    palm_h_ok = 0.5 * palm_h_near + 0.5 * palm_h_far       # iter_07 변경 3
    palm_hi = torch.clamp(h_palm - PALM_H_K * cup_mouth_z, min=0.0)
    high_fac = HIGH_FLOOR + (1.0 - HIGH_FLOOR) * torch.exp(-(palm_hi / (HIGH_K * cup_r)) ** 2)

    # ---- A. 접근. 손끝-벽 수입은 손바닥이 높을수록 줄인다 (iter_07 변경 3)
    d_app = geo["d_app"]
    reach_lin = _sat(1.0 - d_app / (APP_LIN_K * cup_r))
    reach_near = 1.0 - torch.tanh(d_app / (APP_NEAR_K * cup_r))
    reach_far = 0.7 * reach_lin + 0.3 * reach_near
    side_t = torch.maximum(_close_in(geo["d_thumb"], cup_r), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"], cup_r), torch.tanh(f_o / 0.5))
    reach_tips = high_fac * 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach_raw = 0.6 * reach_far + 0.4 * reach_tips

    # ---- A'. 잡는 높이로 먼저 내려오기 (iter_07: 원거리 기울기, 법선 수평도 곱함)
    side_gap = torch.clamp(rad_p - SIDE_K * cup_r, min=0.0)
    side_gate = LOW_SIDE_FLOOR + (1.0 - LOW_SIDE_FLOOR) * torch.exp(-(side_gap / (LOW_SIDE_K * cup_r)) ** 2)
    level_fac = LEVEL_FLOOR + (1.0 - LEVEL_FLOOR) * hand_level * normal_level
    level_fac = level_fac + (1.0 - level_fac) * lift_hold
    low_score = level_fac * palm_h_ok * side_gate

    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w * high_fac            # iter_07 변경 3: 손 안 판정도 높이 감쇠
    near_gate = _sat((2.0 * cup_r - geo["d_grip"]) / cup_r)
    release = torch.maximum(torch.maximum(_sat(in_hand / 0.5), near_gate), g_f)
    openness = _sat(1.0 - hand_closure)
    open_fac = release + (1.0 - release) * (0.45 + 0.55 * openness)
    approach = open_fac * (o_app * reach_raw + LOW_W * low_score)

    # ---- B. 닫기 (그대로)
    near_grip = torch.exp(-(geo["d_grip"] / (1.1 * cup_r)) ** 2)
    close = o_grip * pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # ---- B'. 손 안 감싸기 (손끝-바깥 벽 거리·바깥 접촉만)
    tip_near = WRAP_CAP * torch.exp(-geo["d_each"] / (WRAP_K * cup_r))
    per_finger = torch.maximum(tip_near, touch_each)
    wrap_part = per_finger.mean(dim=1)
    in_gate = torch.maximum(_sat(in_hand / 0.5), g_f)
    wrap = in_gate * o_grip * pose_ok * wrap_part * wrap_part

    # ---- C. 파지: 판정 뒤 수입은 g_eff 로 연속 혼합 (post_q >= pre_q -> 오르는 계단 유지)
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    pre_q = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    post_q = torch.maximum((0.4 + 0.6 * pinch) * finger_fac, pre_q)
    contact_q = g_eff * post_q + (1.0 - g_eff) * pre_q
    grasp = o_grip * pose_ok * contact_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C'. 컵 좌표 파지 자세 유지 (손바닥 높이는 iter_06 근거리 점수 그대로)
    n_dir = palm_axes[:, 0:3]
    f_dir = palm_axes[:, 3:6]
    n_perp = torch.exp(-((n_dir * cup_up).sum(dim=-1) / HOLD_ANG_TOL) ** 2)
    f_perp = torch.exp(-((f_dir * cup_up).sum(dim=-1) / HOLD_ANG_TOL) ** 2)
    side_ok = torch.exp(-((rad_p - SIDE_K * cup_r) / (LOW_H_K * cup_r)) ** 2)
    hold_pose = 0.35 * n_perp + 0.35 * f_perp + 0.15 * palm_h_near + 0.15 * side_ok
    grip_pose = g_eff * pose_ok * finger_fac * hold_pose

    # ---- D. 들기: 들기 인정(held)도 g_eff
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_eff * finger_fac * torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2) * calm
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- G. 교란 (유계, 래치 없음). 벌점 쪽은 실측 접촉력 전체(f_raw)로 본다 (보수적)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    f_t_raw = f_raw[:, 0]
    f_o_raw = f_raw[:, 1:].sum(dim=1)
    f_o_max = f_raw[:, 1:].max(dim=1).values
    thumb_excess = torch.clamp(f_t_raw - f_o_raw - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o_max - f_t_raw, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / (3.5 * cup_r)) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "reach_raw": reach_raw, "close": close, "wrap": wrap, "grasp": grasp,
            "grip_pose": grip_pose, "g_eff": g_eff,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 4번째 사이클 iter_07 (full-joint 26, 팔 절대 목표).
    iter_06 c1 위의 변경:
      1. 손끝-벽 거리는 컵 바깥 면 기준 (입구 안 손끝 = 먼 것).
      2. 입구 안 손끝 접촉력은 파지·다섯 손가락·파지 자세·들기 인정에서 제외 (g_eff 연속 인정, 뒤 단계 게이트도 g_eff).
      3. 옆 접근: 손바닥 높이 점수 원거리 기울기, 높은 손바닥의 근처 접근 수입 감쇠.
      4. 손 방향: 손가락 축 수평 가우스(0.4), 법선 허용 0.3, 낮추기 점수에 법선 수평.
      그 밖은 iter_06 c1 그대로. 두 팔 같은 식·같은 무게. 항 25 개."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (iter_06 그대로) -------------------------------------------------------------------------
    W_APPROACH = 3.0
    W_CLOSE = 0.8
    W_WRAP = 1.2
    W_GRASP = 4.0          # 판정 뒤 크게
    W_GRIP_POSE = 2.5
    W_LIFT = 5.5
    W_RUSH = 0.06
    W_ALIVE = 0.5
    UPRIGHT_TOL = math.radians(20.0)
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20   # 사용자: 소스 들기 높이 그대로
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))

    # 벌점 게이트용 원 판정 (보수적으로 원 판정 유지)
    src_g_raw = ctx.src_grasped.to(dt)
    rcv_g_raw = ctx.rcv_grasped.to(dt)
    both_g_raw = src_g_raw * rcv_g_raw
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) ----------------------------------------------------------------
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

    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    # ---- 팔별 항 (두 팔 같은 식·같은 무게) -----------------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, pose_ok_src,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, pose_ok_rcv,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # iter_07 변경 2: 뒤 단계 수입 게이트는 바깥 접촉 기준 인정(g_eff) -> 엄지 걸기로 든 컵은 배치·조준·붓기 수입 없음
    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones

    # ---- A 접근 / B 닫기 / B' 감싸기 / C 파지 / C' 파지 자세 -------------------------------------------
    approach_src = W_APPROACH * src["approach"]
    approach_rcv = W_APPROACH * rcv["approach"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    finger_wrap = W_WRAP * (src["wrap"] + rcv["wrap"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
    grip_pose_src = W_GRIP_POSE * src["grip_pose"]
    grip_pose_rcv = W_GRIP_POSE * rcv["grip_pose"]

    # ---- D 들기 (소스 높게, 리시버 낮게) ------------------------------------------------------------
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- E 배치 ------------------------------------------------------------------------------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad
    place = converge_cups + pour_dir + rcv_keep

    # ---- E 조준 ------------------------------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 --------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F 회전 (8.0): 창 안·조준 품질 x 회전 진척. 각도 처방 없음 -----------------------------------
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog * rcv_gate

    # ---- F 배달 (알당 30, 증분만): 주역 ------------------------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- G 흘림 (벌점 게이트는 원 판정) ----------------------------------------------------------------
    post = torch.maximum(both_g_raw, src["lifted_w"])
    spill = -40.0 * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g_raw * pre_ramp * out_gate

    # ---- G 충돌 (실기 안전) -----------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g_raw) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g_raw) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                                + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- G 컵 교란: 두 팔 같은 무게 ---------------------------------------------------------------
    tilt_pen_src = (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = 1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    dist_src = 0.5 * src["slide"] + 0.3 * tilt_pen_src + 0.3 * src["one_sided"] \
        + 0.2 * src["palm_push"] + 0.3 * src["fling"] + W_RUSH * src["rush"]
    dist_rcv = 0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv + 0.3 * rcv["one_sided"] \
        + 0.2 * rcv["palm_push"] + 0.3 * rcv["fling"] + W_RUSH * rcv["rush"]
    cup_disturb = -(dist_src + dist_rcv)

    # ---- G 제어 정규화 (cup_grasp 수준). 팔 액션 크기 비용 없음 (a=0 = 시작 자세) ----------------------
    a = ctx.actions
    pa = ctx.prev_actions
    a_arm = torch.cat([a[:, 0:7], a[:, 13:20]], dim=-1)
    pa_arm = torch.cat([pa[:, 0:7], pa[:, 13:20]], dim=-1)
    a_hand = torch.cat([a[:, 7:13], a[:, 20:26]], dim=-1)
    pa_hand = torch.cat([pa[:, 7:13], pa[:, 20:26]], dim=-1)
    da_arm = a_arm - pa_arm
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (
        torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
        + torch.tanh(torch.clamp(qd_rcv.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0))
    ctrl_arm = -(rate_arm + speed_arm + runaway)
    da_hand = a_hand - pa_hand
    ctrl_hand = -0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "finger_wrap": finger_wrap,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "grip_pose_src": grip_pose_src,
        "grip_pose_rcv": grip_pose_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "place": place,
        "aim": aim,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "posture_margin": posture_margin,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "spill": spill,
        "premature": premature,
        "collision": collision,
        "cup_disturb": cup_disturb,
        "ctrl_arm": ctrl_arm,
        "ctrl_hand": ctrl_hand,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
```

## Review items

```
[23] 60 스텝 할인합: 대각선 +112.20 · 먼저 내리고 수평 진입 +118.72 · 수평으로 먼저 가서 위에서 내림 +106.97
     인벨롭 축 손: 대각선 +141.10 · 먼저 내리고 수평 진입 +155.71 (+14.61) · 위에서 내림 +129.66
    ✗ 인벨롭 축 손: '먼저 내리고 수평 진입'이 대각선보다 15(스텝당 0.25) 이상, '위에서 내림'보다 +1.0 이상 크지 않다
```
