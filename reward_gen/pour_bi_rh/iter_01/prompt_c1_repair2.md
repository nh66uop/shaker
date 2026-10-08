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
    """관절 한계 근접 연속 비용 (관절 번호 없음, 유계 [0,1)). 이전 코드 그대로."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    return torch.tanh(c.sum(dim=-1))


def _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r):
    """컵 좌표(축 = cup_up) 손끝-바깥 벽 거리. 입구 안 손끝은 '먼 것'(2 r)으로 친다 (f07 교훈)."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)                                   # (N,F)
    rad = (rel - h_tip.unsqueeze(-1) * up).norm(dim=-1)              # (N,F)
    gap_raw = torch.clamp(rad - r, min=0.0)
    w_rad = _sat((0.9 * r - rad) / (0.3 * r))
    w_h = _sat((h_tip - (mouth_z - 2.0 * r)) / (0.5 * r)) * _sat((mouth_z + 1.5 * r - h_tip) / (0.5 * r))
    w_in = w_rad * w_h                                               # 1 = 입구 안
    gap = (1.0 - w_in) * gap_raw + w_in * (2.0 * r)
    h_t = h_tip[:, 0]
    h_err_t = torch.clamp(-h_t, min=0.0) + torch.clamp(h_t - 0.5 * mouth_z, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2 + 1e-12)      # (N,)
    h_o = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * mouth_z - h_o, min=0.0) + torch.clamp(h_o - 0.8 * mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2 + 1e-12)       # (N,F-1)
    return d_thumb, d_oth, 1.0 - w_in


def _hand_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos, other_cup_pos,
                r, mouth_z, lift_target, lift_relief):
    """팔별 무가중 항. 두 팔 같은 식 (한 팔에만 붙는 부호·상수 없음). 모든 거리 척도는 컵 반지름 배수.
    재시도 2 변경 (Review items):
      [46] 쥐기 전 게이트(앞·중심선 가산, 편 손 해제, rush)를 원 판정(grasped) 대신 바깥 접촉 판정 g_eff 로.
           엄지 걸기는 g_eff 가 이미 약 0 이라 엄지를 빼는 순간 게이트가 뛰지 않는다.
      [25] 편 손 해제 = max(sat(pocket / 0.4), g_eff): 손 안이면 닫아도 접근 수입이 줄지 않는다.
      [31] 가까운 척도 몫 0.3 -> 0.5: 손가락 사이 자세가 12 cm 편 손 정지보다 더 이득.
      [34] 가까운 척도·감싸기 바닥·닫기에 수평 손 인자(같은 단계, 바닥 있음)를 건다: 수직 손의 입구 수입을 줄인다."""
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
    free = 1.0 - g_eff                                                # 재시도 2 [46]: 쥐기 전 게이트

    # ---- 손 방향 점수 (수평 손): 넓은 받침(1 - |dot|) 을 섞어 수직 손에서도 기울기 ([34][35][49])
    s_fl = 0.5 * torch.exp(-(f_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(f_dot))
    s_nl = 0.5 * torch.exp(-(n_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(n_dot))
    level = s_fl * (0.4 + 0.6 * s_nl)
    level_fac = 0.3 + 0.7 * level                                     # 재시도 2 [34]: 바닥 0.3 인 수평 손 인자

    # ---- 1 접근 거리 (L1: 수평 성분이 높이에 묻히지 않게 [24], 높이 x2 로 먼저 내려오기)
    #   컵보다 앞(+x)·상대 컵 쪽(중심선 쪽, [14])으로 나간 만큼 가산 - 쥐기 전에만
    far_side = torch.clamp(palm_pos[:, 0] - cup_pos[:, 0], min=0.0) * free
    to_other = other_cup_pos[:, :2] - cup_pos[:, :2]
    e_other = to_other / (to_other.norm(dim=-1, keepdim=True) + EPS)
    lat = ((palm_pos[:, :2] - cup_pos[:, :2]) * e_other).sum(dim=-1)
    cross = torch.clamp(lat - 0.5 * r, min=0.0) * free
    #   축 쪽으로 밀려 들어간 손바닥은 3 배로 친다 ([26])
    dr_eff = torch.clamp(dr, min=0.0) + 3.0 * torch.clamp(-dr, min=0.0)
    dh_abs = _sabs(dh, 0.1 * r)
    side_d = dr_eff + far_side + cross
    d_app = side_d + 2.0 * dh_abs
    #   먼 척도 30 r (시작 자세에서도 기울기) + 가까운 척도 2 r (재시도 2 [31]: 몫 0.3 -> 0.5,
    #   [34]: 가까운 척도는 수평 손일수록 크게)
    reach_far = _sat(1.0 - d_app / (30.0 * r))
    reach_near = (1.0 - torch.tanh(d_app / (2.0 * r))) * level_fac
    reach = 0.5 * reach_far + 0.5 * reach_near

    dir_env = 0.3 * _sat(cos_env) + 0.7 * torch.exp(-((1.0 - cos_env) / 0.2) ** 2)
    near_w = torch.exp(-(d_app / (4.0 * r)) ** 2)
    orient = level * (0.4 + 0.6 * near_w * dir_env)

    # ---- 손 안 판정 (pocket): 손바닥이 축 3.2 r 안·잡는 높이 근처·K7 방위 쪽 ([28])
    pocket = _sat((3.2 * r - rad_p) / r) * torch.exp(-(dh / (2.0 * r)) ** 2) * _sat((cos_env + 0.3) / 0.6)
    #   재시도 2 [25]: pocket 0.4 이상이면 편 손 요구를 완전히 푼다 (닫기가 접근 수입을 깎지 않음)
    #   [46]: 원 판정 대신 g_eff (엄지 걸기에서 빼는 순간 해제가 사라지지 않음)
    release = torch.maximum(_sat(pocket / 0.4), g_eff)
    openness = _sat(1.0 - hand_closure)
    open_fac = release + (1.0 - release) * (0.6 + 0.4 * openness)    # 편 손 요구는 손 안이 아닐 때만 (i07·i08)
    approach = open_fac * (0.65 * reach + 0.35 * orient)

    # ---- 1' 먼저 내려오기: 잡는 높이 점수 (수평이 멀어도 바닥 0.6) ([23][24])
    low = (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 : 손끝별 바깥 벽 거리 볼록 평균(제곱), 자리·수평 손일수록 크게 + 손 안이면 닫기
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    #   재시도 2 [34]: 바닥 0.3 -> 0.15 (자리·수평 손이 아니면 감싸기 수입이 작다)
    wrap = part * part * (0.15 + 0.85 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)
    #   재시도 2 [34]: 닫기도 수평 손 인자 (바닥 0.3)
    close = pocket * level_fac * _sat(hand_closure / 0.7)

    # ---- 3 파지 : 판정 뒤 계단(>= 0.5) + 판정 전 접촉 사전 수입(<= 0.4)
    f_t = f_out[:, 0]
    f_o = f_out[:, 1:].sum(dim=1)
    pre = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    part_o = touch_o.mean(dim=1)
    post = 0.5 + part_o * part_o * part_o                             # 세제곱: 5지째 몫이 가장 크다 ([43])
    grasp = g_eff * post + (1.0 - g_eff) * pre

    # ---- 4 인벨롭 모양 점수 (가중 평균의 제곱, 컵 좌표라 기울여도 불변) + 손바닥 굴림(법선 기울기) 비용
    s_h = torch.exp(-(dh / (0.6 * r)) ** 2)
    s_r = torch.exp(-(dr / (0.5 * r)) ** 2)
    s_f = torch.exp(-(f_dot / 0.3) ** 2)
    s_n = torch.exp(-(n_dot / 0.3) ** 2)
    q = 0.30 * s_h + 0.25 * s_r + 0.30 * s_f + 0.15 * s_n
    q2 = q * q
    roll = _sat((torch.abs(n_dot) - 0.25) / 0.45)                     # 받침 자세(법선 z +0.72) -> 1 ([39])

    # ---- 5 들기 : 모양과 무관, 처음 몇 cm 가 가장 가파르고 목표 위는 이득 없음
    #   붓기 창 안에서 기울였으면 목표 높이 부족분을 면제 ([50])
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.2 * r, min=0.0)
    lin = _sat(h_pos / lift_target)
    lin = lin + (1.0 - lin) * lift_relief
    prog = 0.7 * torch.tanh(h_pos / (0.7 * r)) + 0.3 * lin
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    lift = g_eff * calm * prog
    lifted_w = _sat(h_pos / (0.7 * r))

    # ---- 교란 (쥐기 전 밀림·쓰러뜨림은 속도로만, i09) + 손바닥을 컵 축 1.3 r 안으로 밀어 넣기 ([26])
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
    rush = near_w * free * torch.tanh(_rms(arm_qd) / 1.0)             # 재시도 2 [46]: g_eff 게이트
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
    """RH 양팔 붓기, 5번째 사이클 iter_00 재시도 2: 단계별 합 사다리 (곱 사다리 없음). 두 팔 같은 식·같은 무게.
    재시도 2 변경은 _hand_terms 의 접근·감싸기 전·닫기·쥐기 전 게이트와 W_CLOSE 뿐 (항 수 그대로)."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 -----------------------------------------------------------------------------------
    W_ALIVE = 0.5         # 종료가 이득이 되지 않도록 작은 양의 기반
    W_APP = 2.0           # 팔당 접근 최대 2
    W_LOW = 1.2           # 먼저 내려오기 ([23], 추정)
    W_WRAP = 1.2
    W_CLOSE = 1.3         # 재시도 2 [25]: 0.8 -> 1.3. 손 안(pocket)에서만: 닫힘 0.12 -> 0.4 약 +0.2, -> 0.7 약 +0.4 (pocket 0.4 이상, 추정)
    W_GRASP = 3.0         # 판정 뒤 >= 1.5 > 판정 전 사전 수입 <= 1.2 -> 파지 판정이 계단으로 이득; 5지 1.5 x 3 = 4.5
    W_SHAPE = 1.5         # 쥔 동안 인벨롭 모양 보너스 (테이블 위)
    W_SHAPE_LIFT = 5.0    # 모양 보너스가 들린 만큼 커짐 (더하는 항; 들기 기본 수입에는 곱하지 않음)
    W_ROLL = 1.5          # 쥔 채 손바닥 굴림(받침 자세) 비용, 팔당 최대 1.5 ([39])
    W_LIFT = 2.5          # 3 cm 이득 약 1.6 (모양 무관, [56] >= 1.4)
    W_AIM = 2.5
    W_POUR_H = 4.0        # 기울인 채 주둥이 높이: 21.3 -> 5.6 cm 합 약 +1.1 ([50], 추정)
    W_ROT = 23.0          # weight 23 with delivery 30/bead: 회전+배달 몫 약 52 % (추정)
    W_DELIVER = 30.0      # 알당 30, 부호 있는 증분 (i20: 진행형 비대칭 clamp 금지)
    W_SPILL = 40.0
    W_SUCCESS = 20.0
    W_FINISH = 1.5
    W_MARGIN = 0.8        # 쥔 동안 관절 여유 비용, 팔당 최대 0.8 (유계, i15)
    UPRIGHT_TOL = math.radians(20.0)
    LIM = math.radians(81.0)

    r_cup = float(ctx.cup_radius)
    mouth_z = float(ctx.cup_mouth_z)
    cup_h = mouth_z - float(ctx.cup_bottom_z)          # 컵 높이 (자산 속성에서 계산)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    active = float(max(int(ctx.num_beads), 1))
    LIFT_RCV = 3.0 * r_cup                             # 리시버는 낮게 (반지름 배수)

    src_g_raw = ctx.src_grasped.to(dt)
    rcv_g_raw = ctx.rcv_grasped.to(dt)
    both_g_raw = src_g_raw * rcv_g_raw
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (K5: 주둥이 기준) -----------------------------------------------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])     # 리시버 쪽으로 기운 부호 있는 각
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    above_rim = _sat(1.0 + dz_lip / r_cup)
    window = lip_gate * above_rim                       # 붓기 창 (같은 단계 인자)
    theta_pos = torch.clamp(theta, min=0.0)

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    clear_src = torch.clamp(h_rcv, 0.0, LIFT_RCV) + cup_h + r_cup      # 리시버 입구를 넘을 여유까지만
    #   창 안에서 기울였으면 들기 목표 부족분 면제
    lift_relief_src = window * _sat(theta_pos / (0.5 * LIM))

    src = _hand_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                      ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped, ctx.src_arm_qd,
                      ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                      ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, clear_src, lift_relief_src)
    rcv = _hand_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                      ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                      ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                      ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, LIFT_RCV, zeros)

    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    alive = W_ALIVE * ones

    # ---- 1~3 접근(+먼저 내려오기) / 감싸기 전 / 파지 ------------------------------------------------------
    approach_src = W_APP * src["approach"] + W_LOW * src["low"]
    approach_rcv = W_APP * rcv["approach"] + W_LOW * rcv["low"]
    prewrap = W_WRAP * (src["wrap"] + rcv["wrap"]) + W_CLOSE * (src["close"] + rcv["close"])
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- 4 인벨롭 모양 보너스 (쥔 동안 더함) - 손바닥 굴림 비용 ----------------------------------------------
    envelope = src_g * (src["q2"] * (W_SHAPE + W_SHAPE_LIFT * src["prog"]) - W_ROLL * src["roll"]) \
        + rcv_g * (rcv["q2"] * (W_SHAPE + W_SHAPE_LIFT * rcv["prog"]) - W_ROLL * rcv["roll"])

    # ---- 5 들기 (모양 무관) ------------------------------------------------------------------------
    lift_src = W_LIFT * src["lift"]
    lift_rcv = W_LIFT * rcv["lift"]

    # ---- 6 조준 (두 컵이 들린 뒤, 주둥이 기준, 넓은 받침: 21 cm 위·12 cm 옆에서도 기울기) ----------------
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))
    h_err = torch.abs(dz_lip - 0.25 * r_cup)
    hz = 0.3 * (1.0 - torch.tanh(h_err / (1.5 * r_cup))) + 0.7 * _sat(1.0 - h_err / (12.0 * r_cup))
    xy = 0.5 * (1.0 - torch.tanh(lip_d / wide)) + 0.5 * (1.0 - torch.tanh(lip_d / two_r))
    aim_carry = W_AIM * both_g * lifted * (0.6 * xy + 0.4 * hz)

    # ---- 비드 장부 ----------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- 7 붓기: 창 안 회전 진척 (방향 있는 기울기) + 기울이는 동안 주둥이 높이 -------------------------
    rot_prog = _sat(theta_pos / (1.4 * LIM))            # 95 도까지 단조 증가
    pour_rot = W_ROT * both_g * want * window * rot_prog
    pour_h = W_POUR_H * both_g * want * window * rot_prog * hz
    aim = aim_carry + pour_h

    # ---- 7 배달 (주역, 부호 있는 증분) -------------------------------------------------------------------
    pour_delta = W_DELIVER * ctx.d_in_target * active

    # ---- 다 부으면 세우기 / 성공 --------------------------------------------------------------------------
    finish_upright = W_FINISH * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))
    success_bonus = W_SUCCESS * ctx.success.to(dt)

    # ---- 흘림 (영구, 소스를 쥐었거나 든 뒤) / 창 밖 조기 기울임 ------------------------------------------------
    post = torch.maximum(src_g_raw, src["lifted_w"])
    spill = -W_SPILL * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g_raw * pre_ramp * out_gate

    # ---- 충돌 (실기 안전, 유계) ----------------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g_raw) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g_raw) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                                + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- 컵 교란: 속도형 + 쥔 컵의 기울기(소스는 붓기 창 밖에서만) --------------------------------------------
    tilt_pen_src = src_g * (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = rcv_g * (1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2))
    cup_disturb = -(src["disturb"] + rcv["disturb"] + 0.3 * tilt_pen_src + 0.3 * tilt_pen_rcv)

    # ---- 리시버 세움·중심선 쪽 ----------------------------------------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- 관절 여유 비용 (쥔 동안) -------------------------------------------------------------------
    joint_margin = -W_MARGIN * (src_g * _limit_cost(ctx.src_arm_limit_margin)
                                + rcv_g * _limit_cost(ctx.rcv_arm_limit_margin))

    # ---- 제어 비용 (cup_grasp 수준; 팔 액션 크기 비용 없음, 손 채널 포화 벌점 없음) ------------------------
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
bead/in_target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.23e-06 · max 2.44e-05
bead/spill: [0, 0.00151, 0, 0, 0, 0, 0, 0.00117, 0, 1.22e-05]  min 0 · mean 0.000156 · max 0.00181
done/drop: [0, 0.00122, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.00039 · max 0.00488
done/mimic_runaway: [0, 0, 0, 0.000488, 0, 0.00122, 0, 0, 0, 0]  min 0 · mean 0.000108 · max 0.0022
episode_lengths/step: [352, 321, 641, 522, 890, 881, 827, 823, 891, 848]  min 224 · mean 657 · max 899
reward/aim: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/alive: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  min 0.5 · mean 0.5 · max 0.5
reward/approach_rcv: [0.409, 0.874, 0.96, 0.97, 1.04, 1.26, 1.07, 1.3, 1.5, 1.26]  min 0.368 · mean 1.09 · max 1.64
reward/approach_src: [0.419, 0.909, 0.347, 0.908, 0.599, 1.04, 1.1, 1.27, 1.11, 1.35]  min 0.175 · mean 0.927 · max 1.53
reward/collision: [0, -0.0698, -0.282, -0.644, -0.08, -0.314, -0.194, -0.102, -0.0732, -0.0691]  min -0.733 · mean -0.187 · max 0
reward/ctrl: [-0.026, -0.0412, -0.0451, -0.0669, -0.0417, -0.0426, -0.0418, -0.0415, -0.0395, -0.0418]  min -0.0828 · mean -0.0454 · max -0.0259
reward/cup_disturb: [-0.208, -0.393, -0.212, -0.219, -0.218, -0.219, -0.219, -0.208, -0.217, -0.275]  min -0.526 · mean -0.266 · max -0.206
reward/envelope: [0, -0.00416, 0, 0, 0, 0, -9.77e-05, 0, 0, -0.000178]  min -0.00964 · mean -0.000287 · max 0.000421
reward/finish_upright: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/grasp_rcv: [0, 0.00383, 0, 0.000546, 0.00409, 0.000217, 0, 0.00272, 0.00505, 0.00204]  min 0 · mean 0.00747 · max 0.0535
reward/grasp_src: [0, 0.0812, 0, 0.000103, 0, 0, 0.00134, 0.000572, 0, 0.0233]  min 0 · mean 0.00903 · max 0.141
reward/joint_margin: [0, -1.7e-06, 0, 0, 0, 0, -2.93e-05, 0, 0, -0.000434]  min -0.00307 · mean -0.000139 · max 0
reward/lift_rcv: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.72e-07 · max 0.000113
reward/lift_src: [0, 1.74e-06, 0, 0, 0, 0, 2.4e-08, 0, 0, 5.04e-08]  min 0 · mean 4.74e-06 · max 0.000472
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.71e-06 · max 0.00732
reward/pour_rot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/premature: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00246 · mean -9.73e-06 · max 0
reward/prewrap: [3.84e-06, 0.0985, 0.000243, 0.00138, 0.00707, 0.0302, 0.0122, 0.0341, 0.132, 0.135]  min 3.84e-06 · mean 0.0884 · max 0.403
reward/rcv_keep: [0, 9.04e-05, 0, -0.00059, 0, 0, 0, 0, -0.000366, 0]  min -0.00161 · mean -5.17e-05 · max 0.00163
reward/spill: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0488 · mean -0.000263 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [0, 1.84, 1.22, 1.39, 1.79, 2.21, 2.19, 2.75, 2.88, 2.85]  min 0 · mean 2.07 · max 3.16
rewards/step: [489, 656, 703, 754, 1.55e+03, 2e+03, 1.9e+03, 2.35e+03, 2.68e+03, 2.35e+03]  min 283 · mean 1.39e+03 · max 2.71e+03
task/aim_dist: [0.32, 0.311, 0.321, 0.319, 0.32, 0.32, 0.321, 0.317, 0.317, 0.316]  min 0.304 · mean 0.317 · max 0.321
task/cup_collision_rate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3.89e-06 · max 0.000488
task/cups_center_dist: [0.32, 0.313, 0.321, 0.319, 0.32, 0.32, 0.321, 0.317, 0.317, 0.318]  min 0.309 · mean 0.318 · max 0.321
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.000244, 0, 0, 0, 0, 0, 0.000488, 0.000244, 0]  min 0 · mean 7.19e-05 · max 0.000977
task/rcv_cup_lift: [4.37e-06, 8.53e-05, 9.57e-06, 2.11e-05, 2.24e-05, 3.77e-05, 1.39e-05, 2.84e-05, 1.33e-05, 2.38e-05]  min -0.000131 · mean 8.52e-05 · max 0.00431
task/rcv_grasped: [0, 0.000244, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000267 · max 0.00659
task/rcv_hand_foreign_rate: [0, 0.155, 0.626, 0.616, 0.168, 0.285, 0.0662, 0.0786, 0.185, 0.0686]  min 0 · mean 0.226 · max 0.931
task/src_cup_lift: [3.01e-06, -0.000248, 7.07e-06, 1.52e-05, 6.5e-06, 1.76e-05, -5.05e-06, 9.94e-06, 5.67e-06, 0.000404]  min -0.00107 · mean 7.57e-05 · max 0.000841
task/src_grasped: [0, 0.0144, 0, 0, 0, 0, 0.000244, 0, 0, 0.000732]  min 0 · mean 0.000907 · max 0.0295
task/src_hand_foreign_rate: [0, 0.0654, 0, 0.78, 0.0317, 0.605, 0.511, 0.321, 0.043, 0.184]  min 0 · mean 0.25 · max 0.837
task/src_tilt_deg: [0.0178, 6.92, 0.0186, 0.0739, 0.019, 0.0565, 0.306, 0.131, 0.0203, 2.37]  min 0.0178 · mean 1.55 · max 12.1
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# pour_bi_rh 보상 설계 브리프 (2026-10-01 판 — 5번째 사이클 iter_01, full-joint 제어·팔 절대 목표·컵 스폰 −12 cm·실기 제어 응답)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트에 통째로 들어간다.**

이 트랙은 네 번 리셋했다. 1~12 라운드 `archive_r1-12`, 13~24 `archive_r13-24`, 3번째 사이클 `archive_r25-34`, 4번째 사이클(full-joint) iter_00~09 는 `archive_r35-44`.
지금은 5번째 사이클의 iter_01 이다. iter_00(09-30 사용자 "학습 보상 다시 세팅")은 **단계별 합 사다리**로 새로 짰다(5절). iter_01 은 그 위에서 2-1 절(참 궤적 최적)과 5-5 절 두 가지만 고친다.

## 1. 과제와 고정 조건 (바꿀 수 없다)

- 오른팔(소스, `src_*`)이 비드 20 개가 든 컵을, 왼팔(리시버, `rcv_*`)이 빈 컵을 잡아 든다. 소스를 기울여 비드를 리시버 컵에 옮긴다.
- 성공(env 가 계산, 보상이 재정의 못 함): 담김 ≥ 0.80 · 흘림 ≤ 0.25 · 두 컵 파지 유지 · 받는 컵 ≤ 20° · 20 스텝 유지 · 컵이 포개지지 않음.
- **액션 26 = 팔마다 13** — [팔 관절 절대 목표 7(a=0 시작 자세·±1 관절 한계, 0.3 rad/s 상한 — iter_03 부터), 손 구동 관절 절대 목표 6]. 로봇 설명 절이 정의한다.
- **물리 이상은 env 가 종료한다(iter_03 부터)**: 컵 > 3 m/s·> 30 rad/s, 손 접촉 > 200 N 2 스텝, 컵이 스폰에서 0.5 m 밖, 컵 낙하, mimic 폭주. 종료 벌점 항은 두지 않는다.
- 스폰(컵 원점, env-local): **09-30 부터 x −0.12**(사용자 A, env `object_spawn_dx`) — 소스 (0.262, −0.159, 0.268) · 리시버 (0.258, +0.165, 0.268), ±2 cm. 시작 손바닥 (0.31, ∓0.30, 0.42): 컵에서 수평 약 15 cm·위 약 15 cm. 보상은 컵 위치를 스폰·컵 기준 상대값으로만 쓴다.
- r = `ctx.cup_radius` = 0.028 m. 소스 컵 원점 → 림 0.054 m. 제어 주기 1/60 s, 에피소드 900 스텝.
- **실기 제어 응답(09-30 부터 env, 사용자 "실제 제어 로직으로 학습")**: 손 명령 4 스텝 지연 · 팔 명령 9~12 스텝 지연(env 마다) · 손 관절 속도 상한 2.1 rad/s(엄지 굽힘 0.56) · 네 손가락은 0.065 rad 까지만 펴진다(편 손의 `hand_closure` 최소 약 0.05). 손 액션 a = 0 은 반쯤 쥔 손(닫힘 0.5)이다 — 편 손은 정책이 −1 쪽을 내야 한다.

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

## 6. 평가

- 틱은 단계 관점: near_rate → cup_in_pocket·oppose → closure → grasped → cup_lift → converge·aim → pour. 단계가 오를 때 보상·에피소드 길이가 떨어지면 골.
- full-joint 지표: `joint/{src,rcv}_arm_limit_frac`, `_arm_track_err_deg`, `_hand_track_err_deg`.
- 성공 판정은 결정론 평가(담김 ≥ 0.8·최대 기울기 순간 방향 오차 < 30°·주둥이가 입구보다 위).
- 후보 선택 전 `traj_ledger.py` T1~T5 전부 통과(2-1) + 검사 [1]~[57].

## 7. 시작

fresh(보상이 새 행동 — 편 손 접근 — 을 요구). 웜스타트 없음. seed 42. env `object_spawn_dx −0.12`·팔 절대 목표·물리 이상 종료·bounds 0.005·실기 제어 응답 그대로.
# 5번째 사이클 iter_01 운영 노트 — 2026-10-01

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다(특히 2-1 참 궤적 최적, 4절 rh5_f00 행, 5-5 고칠 것).

## 사용자 결정 (10-01)
- 틱 보고(rh5_f00 e753, 컵에서 20 cm 밖 손 닫힘 0.42/0.52) 뒤: "접근할 때 손을 안 펴면 파지하기 어려울 텐데?" → 1번(f00 종료, iter_01 에서 편 손 접근을 고친다, fresh).
- "왼손이든 오른손이든 보상함수가 무조건 테스크 성공을 보장해야 한다. 에피소드 동안 손 펴고 접근, 인벨롭 그립, 리프트, 이송이라는 트레젝토리 참값을 찾을 수 있게." → DESIGN_BRIEF 2-1, 검사 traj_ledger.py T1~T5.

## 측정 사실 (rh5_f00, iter_00 c1r2, fresh, 실기 제어 응답 켬, e753 종료)
- 학습 곡선 e753(10 epoch 평균, 오른손/왼손): 컵 10 cm 안 비율 0.62/0.65 · 손바닥-컵 10.3/10.2 cm · 파지 0.001/0.005 · 들기 0 · 컵 20 cm 밖 손 닫힘 0.42/0.52 · 10 cm 안 쥐기 전 닫힘 0.52/0.59 · 오른손 손 추종 오차 16.8° · collision −0.24.
- 같은 epoch f10(iter_09 c1r, 실기 제어 응답 없음): 10 cm 안 0.93/0.86 · 파지 0.062/0 · 20 cm 밖 닫힘 0.12/0.19.
- 편 손 한계(0.065 rad)가 만드는 닫힘 최소는 약 0.05 — 0.42~0.52 는 정책이 손 액션을 0 근처(반쯤 쥔 손)로 둔 값이다.

## iter_00 c1r2 코드 결정론 검사 (traj_ledger.py, 두 컵 y 거울 자리)
- T4 편 손(0.05) − 반쯤 쥔 손(0.5), 팔당 스텝당: 내리기 구간 A +0.094 · 컵 뒤로 도는 B +0.076 · 수평 진입 C −0.377(최소 −1.11). 한 팔 반쯤 쥔 손 접근의 에피소드 할인 가치 손해 +5.8 뿐(위에서 덮는 접근 +49.7, 12 cm 주차 +481).
- T3 소스가 80 % 붓고 80° 로 남은 비드를 안은 채 유지: 끝까지 붓고 세우기보다 할인 가치 +1268.6 이득(`want = sat(need_frac × 20)` 가 한 알만 남아도 1).
- T1 다 부은 스텝에 pour_rot −10.65·aim −1.33 한 스텝 절벽.
- T2 전진 최적·T5 대칭은 통과.
- 코드 원인: 편 손 인자 `open_fac = release + (1 − release)(0.6 + 0.4 × openness)` 가 `approach`(거리·방향)에만 곱해지고 `low`(높이, 가중 1.2)는 편 손과 무관. `release = max(sat(pocket / 0.4), g_eff)` 의 pocket 은 손바닥 위치(축 3.2 r 안·잡는 높이·K7 방위)만 보고 손가락이 컵을 둘러쌌는지 보지 않는다. `close = pocket × level_fac × sat(closure / 0.7)` 도 같은 pocket.
- 4번째 iter_09 c1r(f10) 는 `open_fac = release + (1 − release)(0.45 + 0.55 × openness)` 를 거리+높이 전체에 곱했다.

## iter_01 에서 할 것 (DESIGN_BRIEF 5-5, 한 가설, 두 변경)
1. 접근 중 편 손: 파지 전 접근 수입 전체(거리·높이·방향)에 편 손 인자. 편 손 요구 해제와 닫기 수입은 손가락이 컵을 둘러싼 기하에서만. T4 매 스텝 ≥ +0.15.
2. 다 부으면 끝: 기울임 수입이 남은 비드 비율에 비례해 줄고 세우기·성공이 이어받는다. T3 '80 % 붓고 80° 유지' 손해 ≥ 2, T1 절벽 없음.
3. 나머지 항·가중은 iter_00 c1r2 그대로. 검사: check_iter00 [1]~[57] + traj_ledger T1~T5 전부.


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
    """관절 한계 근접 연속 비용 (관절 번호 없음, 유계 [0,1)). 이전 코드 그대로."""
    c = (1.0 - _sat(margin / 0.25)) ** 2
    return torch.tanh(c.sum(dim=-1))


def _tip_geometry(tips_pos, cup_pos, cup_up, mouth_z, r):
    """컵 좌표(축 = cup_up) 손끝-바깥 벽 거리. 입구 안 손끝은 '먼 것'(2 r)으로 친다 (f07 교훈). 이전 코드 그대로."""
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)                                   # (N,F)
    rad = (rel - h_tip.unsqueeze(-1) * up).norm(dim=-1)              # (N,F)
    gap_raw = torch.clamp(rad - r, min=0.0)
    w_rad = _sat((0.9 * r - rad) / (0.3 * r))
    w_h = _sat((h_tip - (mouth_z - 2.0 * r)) / (0.5 * r)) * _sat((mouth_z + 1.5 * r - h_tip) / (0.5 * r))
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
    """iter_01 [5-5 1] 손가락이 컵을 둘러쌌다는 기하 [0,1] (재시도: [T1][25][57] 대응으로 다시 짬).
    (a) 손끝 바깥 벽 거리 작음: 다섯 손끝 평균 exp(-(d / 0.3 r)^2). 엄지 곱 대신 평균이라 한 손끝이 멀어도 0 이 되지 않는다.
        입구 안 판정은 '림 아래 1 r 보다 높은' 손끝만. 그보다 낮은 곳에서 반경 안에 든 손끝은 벽에 닿은 것으로 본다
        (가설: D 단계 닫힘 0.7 이상 재생 손끝이 벽 안쪽으로 찍혀 '입구 안 = 먼 것'이 되고 한 스텝 절벽을 만들었다, T1 t=378).
    (b) 서로 반대쪽: 네 손가락 끝 방위가 '손바닥 + 엄지' 방위(가까운 쪽 턱)의 반대쪽.
        손바닥이 K7 자리에 오기 전에 손가락이 말리면 손끝이 손바닥과 같은 앞쪽에 모여 0 에 가깝다.
        (가설: 손바닥을 가까운 쪽 턱에 넣어 엄지 끝 위치 오차에 덜 민감하게 한다.)"""
    EPS = 1e-6
    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h = (rel * up).sum(dim=-1)                                       # (N,F)
    rad_v = rel - h.unsqueeze(-1) * up                               # (N,F,3)
    rad = rad_v.norm(dim=-1)                                         # (N,F)
    w_in = _sat((0.9 * r - rad) / (0.3 * r)) * _sat((h - (mouth_z - r)) / (0.5 * r))
    gap = (1.0 - w_in) * torch.clamp(rad - r, min=0.0) + w_in * (2.0 * r)
    h_t = h[:, 0]
    herr_t = torch.clamp(-h_t, min=0.0) + torch.clamp(h_t - 0.5 * mouth_z, min=0.0)
    h_o = h[:, 1:]
    herr_o = torch.clamp(-0.4 * mouth_z - h_o, min=0.0) + torch.clamp(h_o - 0.8 * mouth_z, min=0.0)
    herr = torch.cat([herr_t.unsqueeze(1), herr_o], dim=1)
    d = torch.sqrt(gap * gap + herr * herr + 1e-12)
    #   척도 0.3 r: 벽 0.15 r -> 0.78, 반쯤 쥔 손 K7 손끝(약 0.45 r, 추정) -> 0.1 (T4 C 구간 보호)
    closeness = torch.exp(-(d / (0.3 * r)) ** 2).mean(dim=1)

    u = rad_v / (rad.unsqueeze(-1) + EPS)
    p_rel = palm_pos - cup_pos
    p_h = p_rel - (p_rel * cup_up).sum(dim=-1, keepdim=True) * cup_up
    u_p = p_h / (p_h.norm(dim=-1, keepdim=True) + EPS)
    near = u_p + u[:, 0, :]
    near_hat = near / (near.norm(dim=-1, keepdim=True) + EPS)
    cos_f = (u[:, 1:, :] * near_hat.unsqueeze(1)).sum(dim=-1).mean(dim=1)
    opp = _sat((0.3 - cos_f) / 0.8)                                  # 반대쪽(cos <= -0.5) -> 1, 같은 쪽 -> 0
    return opp * closeness


def _hand_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos, other_cup_pos,
                r, mouth_z, lift_target, lift_relief):
    """팔별 무가중 항. 두 팔 같은 식 (한 팔에만 붙는 부호·상수 없음). 모든 거리 척도는 컵 반지름 배수.
    iter_01 변경 (DESIGN_BRIEF 5-5 1):
      - 편 손 인자 open_fac 를 접근 거리·방향(approach)과 높이(low) 전체에 곱한다 (벌점 아님, 곱하는 상점).
      - 편 손 요구 해제(release)와 닫기(close)는 손바닥 자리(pocket) x 손가락 둘러싸기 기하(surround) 에서만.
    재시도 (Review items):
      [T4] 시작 자세(접근 수입이 작은 곳)에서도 편 손 이득 >= 0.15: 편 손 요구 세기 k 를 컵에서 멀수록 크게(0.9),
           잡는 자리 근처는 작게(0.2) -> 컵 옆에서 손을 닫아 가는 참 궤적(D)의 스텝당 하락도 작다 ([T1]).
      [T1][25][57] surround 를 다섯 손끝 평균 + 손바닥·엄지 대 네 손가락 반대쪽으로 (엄지 곱·입구 안 오판정 제거)."""
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

    # ---- 손 방향 점수 (수평 손)
    s_fl = 0.5 * torch.exp(-(f_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(f_dot))
    s_nl = 0.5 * torch.exp(-(n_dot / 0.3) ** 2) + 0.5 * _sat(1.0 - torch.abs(n_dot))
    level = s_fl * (0.4 + 0.6 * s_nl)
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

    # ---- 편 손 인자 (iter_01 [5-5 1], 재시도 [T4][T1])
    #   openness: 닫힘 0.05(편 손 최소) -> 1, 0.5 -> 0.5. 제곱이라 반쯤 쥔 손 결손 몫 0.75.
    #   k: 컵에서 멀면 0.9 (시작 자세 접근 합 약 0.38 에서도 편 손 - 반쯤 쥔 손 약 +0.26, 추정),
    #      잡는 자리 근처 0.2 (접근 합 약 3.2 에서 반쯤 쥔 손 결손 약 0.4 -> D 단계 닫기 스텝당 하락 < 0.025/팔, 추정)
    openness = _sat(1.0 - (hand_closure - 0.05) / 0.9)
    k_open = 0.2 + 0.7 * (1.0 - near_w)
    open_fac = release + (1.0 - release) * (1.0 - k_open * (1.0 - openness * openness))
    approach = open_fac * (0.65 * reach + 0.35 * orient)

    # ---- 1' 먼저 내려오기: 잡는 높이 점수 - iter_01: 편 손 인자를 여기에도 곱한다 (f00 원인)
    low = open_fac * (0.5 * torch.exp(-(dh / r) ** 2) + 0.5 * _sat(1.0 - dh_abs / (6.0 * r))) \
        * (0.6 + 0.4 * torch.exp(-(side_d / (4.0 * r)) ** 2))

    # ---- 2 감싸기 전 : 손끝별 바깥 벽 거리 볼록 평균(제곱) + 손가락이 둘러쌌을 때만 닫기
    near_tip = torch.cat([_close_in(d_thumb, r).unsqueeze(1), _close_in(d_oth, r)], dim=1)
    per_tip = torch.maximum(0.8 * near_tip, touch)
    part = per_tip.mean(dim=1)
    wrap = part * part * (0.15 + 0.85 * torch.exp(-(d_app / (3.0 * r)) ** 2) * level)
    #   iter_01: pocket 대신 in_hand (손바닥 자리만으로는 닫기 수입 없음)
    close = in_hand * level_fac * _sat(hand_closure / 0.7)

    # ---- 3 파지 : 판정 뒤 계단(>= 0.5) + 판정 전 접촉 사전 수입(<= 0.4)
    f_t = f_out[:, 0]
    f_o = f_out[:, 1:].sum(dim=1)
    pre = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
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
    """RH 양팔 붓기, 5번째 사이클 iter_01 (재시도): 단계별 합 사다리 (곱 사다리 없음). 두 팔 같은 식·같은 무게.
    iter_01 변경 (DESIGN_BRIEF 5-5): [1] 편 손 접근·둘러싸기 기하 해제 (_hand_terms),
    [2] 기울임 수입을 남은 비드 비율에 비례. 재시도 [7]: 나간 비드 몫의 이어받기를 회전 수입(pour_rot)·높이 수입(aim)
    에서 빼서 세우기 수입(finish_upright)으로 옮긴다 -> 다 부은 뒤 같은 기울기의 회전 수입 0. 나머지 항·가중 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 (iter_00 c1r2 그대로) ----------------------------------------------------------------
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
    W_MARGIN = 0.8
    #   재시도 [7]: 이어받기 무게 = 회전 + 주둥이 높이 무게의 합 (새 크기 아님). 비드가 떠날 때 잃는 기울임 수입
    #   (알당 <= (23 + 4) / 20 x window x rot_prog) 을 알당 27 / 20 로 이어받아 한 스텝 하락이 없다 (T1).
    W_CARRY = W_ROT + W_POUR_H
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
                      ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                      ctx.src_cup_spawn_pos, ctx.rcv_cup_pos, r_cup, mouth_z, clear_src, lift_relief_src)
    rcv = _hand_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                      ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                      ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                      ctx.rcv_cup_spawn_pos, ctx.src_cup_pos, r_cup, mouth_z, LIFT_RCV, zeros)

    src_g = src["g_eff"]
    rcv_g = rcv["g_eff"]
    both_g = src_g * rcv_g

    alive = W_ALIVE * ones

    # ---- 1~3 접근(+먼저 내려오기) / 감싸기 전 / 파지 ---------------------------------------------------
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

    # ---- 비드 장부 (iter_01 [5-5 2]) -------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    #   기울임 수입은 남은 비드 비율에 비례 (이전: sat(need x 20) 은 한 알만 남아도 1 -> 80 % 붓고 머묾 +1269)
    want = _sat(need_frac)
    #   잘 나간 몫 = 소스를 떠났고 흘리지 않은 비드 (리시버 안 + 공중). 흘린 비드는 넣지 않는다.
    out_ok = _sat(1.0 - need_frac - torch.clamp(ctx.bead_spill_frac, min=0.0))

    # ---- 7 붓기: 창 안 회전 진척 x 남은 비율 (재시도 [7]: 이어받기 몫은 여기서 뺀다) ------------------------------
    rot_prog = _sat(theta_pos / (1.4 * LIM))            # 95 도까지 단조 증가
    tilt_share = want * window * rot_prog
    pour_rot = W_ROT * both_g * tilt_share               # 다 부으면 0 (같은 95 도 자세에서도)
    pour_h = W_POUR_H * both_g * tilt_share * hz
    aim = aim_carry + pour_h

    # ---- 7 배달 (주역, 부호 있는 증분) ---------------------------------------------------------------
    pour_delta = W_DELIVER * ctx.d_in_target * active

    # ---- 다 부으면 끝: 나간 비드 몫 이어받기 + 세우기 / 성공 (재시도 [7], 5-5 2) ------------------------------
    #   이어받기(W_CARRY x out_ok)는 기울기와 무관 -> 다 부은 뒤 세우는 동안 떨어지지 않고, 세울수록 W_FINISH 몫이 오른다 (T1).
    #   80 % 붓고 80 도 유지: 기울임 23 x 0.2 x 0.71 + 높이 약 0.6 + 이어받기 27 x 0.8 = 약 25.4
    #   < 끝까지 붓고 세우기 27 + 1.5 = 28.5 (스텝당 약 -3, 남은 4 알 배달 수입 별도) (T3, 추정)
    upright_src = 1.0 - torch.tanh(tilt_src / (0.3 * LIM))
    finish_upright = both_g * (W_CARRY * out_ok + W_FINISH * (1.0 - want) * upright_src)
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

    # ---- 관절 여유 비용 (쥔 동안) -----------------------------------------------------------------------
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
[1] 붓는 상태 스텝당 총수입 +87.320 · 회전·배달 몫 +39.888 (46 %)
    주차(두 컵 들고 정지) 스텝당 총수입 +36.817
    ✗ 회전·배달 수입이 총수입의 절반 미만
[25] 왼손 손가락 사이 자세에서 닫기: 원래(닫힘 0.12) +2.600 · 0.4 -0.130 · 0.7 -0.011
     prewrap +0.075 → +0.159 → +0.317
    ✗ 컵이 손가락 사이에 있을 때 닫는 기울기가 약하다(0.4 에서 +0.1, 0.7 에서 +0.3 이상)
[31] 왼손 12 cm 편 손 정지 +2.314 · 손가락 사이 자세 +2.600 (차 +0.285) · approach_rcv +1.606→+1.761
    ✗ 손가락 사이 자세가 12 cm 편 손 정지보다 스텝당 0.5 이상 이득이 아니다
[57] f10 왼손(편 손, 손끝 3~4 r) +5.49 · 손바닥을 인벨롭 자리로 +6.14 (+0.648) · 거기서 손끝 벽 1.15 r·닫힘 0.6 +6.33 (+0.188) · 제자리에서 손끝만 벽으로·닫힘 0.6 +5.57 (+0.079)
    ✗ 떠 있는 편 손에서 인벨롭 자리로(+0.3)·벽으로 손끝을 대고 닫기(+0.3)·제자리 닫기(≥ 0) 기울기가 부족
[T1] 참 궤적 스텝당 보상: A@0 +1.25 · B@150 +3.70 · C@270 +4.94 · D@350 +6.89 · E@410 +23.79 · F@500 +38.12 · G@640 +40.19 · H@760 +87.63 · 끝 +88.49
     한 스텝 최대 하락 -0.563 (t=386→387, 단계 D)
    ✗ 참 궤적에 절벽: prewrap -0.545, grasp_rcv -0.009, grasp_src -0.009
[T4] src 편 손(0.05) − 반쯤 쥔 손(0.5), 접근 구간 스텝당: 최소 -0.989 · A +0.553 · B +0.668 · C +0.093
     최악 t=319: 손바닥 축 2.41 r · 반쯤 쥔 손 손끝 축 거리(엄지·검지·중지·약지·새끼) [2.28, 1.09, 1.09, 1.09, 1.09] r · 편 손 − 반쯤 쥔 손 성분 prewrap -0.989, aim +0.000, alive +0.000
    ✗ src: 접근 중 손을 펴는 이득이 스텝당 0.15 미만인 구간이 있다(f00 반쯤 쥔 손 접근)
[T4] rcv 편 손(0.05) − 반쯤 쥔 손(0.5), 접근 구간 스텝당: 최소 -0.989 · A +0.553 · B +0.668 · C +0.093
    ✗ rcv: 접근 중 손을 펴는 이득이 스텝당 0.15 미만인 구간이 있다(f00 반쯤 쥔 손 접근)
    ✗ 참 궤적 원장 T1~T5 중 실패가 있다
```
