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
  actions[0:6]   = source palm 6-DoF command (xyz + yaw/pitch/roll) — the arm is moved \
by a geometric-fabrics controller toward a palm target. From iter_08 the env runs the INCREMENT mode: \
the (EMA-filtered) command is the palm target VELOCITY as a fraction of a max speed \
(source 0.6 m/s·180 deg/s, receiver 0.4 m/s·120 deg/s); the target accumulates, stays inside a box and \
within a few cm of the measured palm. So +-1 means "max speed", 0 means "hold still" (before iter_08: absolute target offset)
  actions[6:12]  = source hand closure commands, one per driven joint \
(thumb abduction, thumb flexion, index, middle, ring, pinky; 0 = open, 1 = closed)
  actions[12:18] = receiver palm 6-DoF command (same mode as the source palm)
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


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """cup_grasp 형태: 0 N -> 0, 0.1 N -> ~0.6, 수십 분의 1 N 위에서 포화 (힘 스파이크가 지배하지 못함)."""
    return torch.tanh(torch.clamp(force, min=0.0) / scale)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    rh2_i03 블록 재사용. 높이 오차(h_err_t, h_err_o)를 추가로 돌려준다 (접촉력 대체 경로의 높이 게이트용).
    iter_05 변경 2 유지: 깔때기 목표는 처음부터 잡는 높이다 (d_funnel == d_grip, ARRIVE_RISE 없음)."""
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
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

    # pinch centre: 목표는 잡는 높이의 컵 축 (위에서 도착하는 rise 없음)
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    d_funnel = d_grip
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_funnel": d_funnel, "d_grip": d_grip,
            "thumb_band": thumb_band, "h_err_t": h_err_t, "h_err_o": h_err_o}


def _hand_cup_terms(palm_pos, palm_lin_vel, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                    cup_r, cup_mouth_z, cup_bottom_z, pose_ok, cup_upright, progress,
                    lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (iter_03 cup_grasp 골격 + iter_04 손가락 방향 + iter_05 수평 접근 + iter_07 닫기·밀려 들어감
    + iter_08 차분형 접근(approach)·접근 중 손 펴기(open)).
    두 팔 사이 곱 없음. 들기 전 수입(finger_dir·descend·close)은 held x 컵 높이로 꺼진다.
    reach_raw 는 더 이상 보상으로 지급하지 않고 리시버 진행도 게이트(rcv_stage)에만 쓴다.
    cup_bottom_z 는 쓰지 않는다 (시그니처만 유지)."""
    EPS = 1e-6
    SIDE_K = 2.5       # 옆 파지 자세의 손바닥-컵 축 수평거리 = 컵 반지름의 약 2.5 배 (i03 ep5900 왼손 7.0 cm)
    H_W = 2.0          # 근거리 성분에서만 높이 오차를 수평 간격의 2 배로 센다
    V_REF_K = 10.0     # iter_08 1: 접근 기준 속도 = 컵 반지름 10 배 /s (DG-5F t2r2 와 같은 척도)
    OPEN_R_K = 5.4     # iter_08 2: 손 펴기 가산의 반경 15 cm 를 컵 반지름 배수로 (0.15 / 0.028 = 5.36)
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r)   # 접촉 반경 = ctx.cup_radius
    g_f = grasped.to(cup_pos.dtype)
    f = torch.clamp(finger_force, min=0.0)                     # (N,F)
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)

    # ---- 연속 held (cup_grasp 3 번 형태, RH 판): 엄지 x 다른 손가락, 손가락 수·손바닥 접촉이 오를수록 더
    c = _soft_contact(f)                                       # (N,F)
    c_t = c[:, 0]
    c_o = c[:, 1:].max(dim=1).values
    n_touch = c.sum(dim=1)
    palm_c = _soft_contact(palm_force)
    pinch_c = c_t * c_o                                        # 양쪽 접촉 (없으면 0: 한쪽 접촉은 held 아님)
    count_sat = torch.clamp((n_touch - 2.0) / 2.0, 0.0, 1.0)   # 2 지 -> 0, 4 지 -> 1 (K8: 소스 3 지)
    held = pinch_c * (0.6 + 0.4 * count_sat) * (0.85 + 0.15 * palm_c)
    firm = torch.tanh(torch.minimum(f_t, f_o) / 1.5)           # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 1.0 - torch.maximum(held, g_f)                 # 파지(grasped) 또는 연속 held 가 서면 0

    # ---- 높이 --------------------------------------------------------------------------------------
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)               # 5 mm dead band: tipping on the rim is not lifting
    t_up = torch.tanh(h_pos / 0.02)
    fade = 1.0 - held * t_up                                   # 들기 전 수입 끄기: 컵 높이 기준 (테이블 위 = 1)
    on_table = 1.0 - torch.tanh(h_pos / 0.04)
    late = torch.clamp((progress - 0.2) / 0.8, 0.0, 1.0)
    table_decay = 1.0 - 0.55 * late * on_table                 # cup_grasp 5 번: 후반 테이블 위 파지 수입 감쇠

    # ---- 손바닥의 컵 좌표 (컵 축 기준): 수평거리 rad_p, 높이 h_palm -----------------------------------
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up                            # 컵 축에 수직인 성분 (직립이면 수평)
    rad_p = to_axis.norm(dim=-1)                               # (N,) 손바닥-컵 축 수평거리
    h_palm = -h_c.squeeze(-1)                                  # (N,) 컵 origin 위 손바닥 높이 (컵 축 방향)
    side_d = SIDE_K * cup_r
    grasp_h = 0.25 * cup_mouth_z                               # 잡는 높이

    # ---- reach_raw: 지급하지 않음. 리시버 진행도 게이트(rcv_stage)용으로만 유지 (iter_07 과 같은 식) -----
    d3 = to_cup.norm(dim=-1)                                   # (N,) 손바닥-컵 origin 3D 거리
    side_d3 = (side_d ** 2 + grasp_h ** 2) ** 0.5
    gap3 = torch.clamp(d3 - side_d3, min=0.0)
    reach_far = 1.0 - torch.tanh(gap3 / (7.0 * cup_r))
    gap_xy = torch.clamp(rad_p - side_d, min=0.0)
    h_err_p = torch.abs(h_palm - grasp_h)
    d_eff = torch.sqrt(gap_xy ** 2 + (H_W * h_err_p) ** 2 + 1e-12)
    reach_near = 1.0 - torch.tanh(d_eff / (2.0 * cup_r))
    reach_raw = 0.6 * reach_far + 0.4 * reach_near

    # ---- A. 접근 (iter_08 1): 차분형 진척. 손바닥 속도의 컵 origin 방향 성분 / (10 r /s)
    # 선형·홀함수 (다가가면 +, 같은 속도로 멀어지면 같은 크기 -, 정지 0, 왕복 합 0). 대칭 clamp ±1 (i21: 비대칭 clamp 금지).
    # 옆 접근 가중 = (손바닥->origin 방향의 수평 성분)^2: 입구 바로 위에서 내려오면 0, 옆에서 다가가면 1.
    # 옆 파지 거리(2.5r) 안쪽에서는 지급 0 (더 밀어 넣을 이유 없음, 0.5r 폭으로 부드럽게). 파지 전만, 컵이 서 있을 때만.
    e_c = to_cup / (d3.unsqueeze(-1) + EPS)
    v_to = (palm_lin_vel * e_c).sum(dim=-1)
    prog = torch.clamp(v_to / (V_REF_K * cup_r), -1.0, 1.0)
    side_w = (e_c[:, :2] * e_c[:, :2]).sum(dim=-1)
    out_w = _sat((rad_p - side_d) / (0.5 * cup_r))
    approach = pre_grasp * cup_upright * side_w * out_w * prog

    # ---- A1. 손가락 방향 (iter_04 2 + iter_05 4) ------------------------------------------------------
    f_dir = palm_axes[:, 3:6]
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    dir_dot = _sat((f_dir * e_in).sum(dim=-1))
    level = _sat(1.0 - torch.abs((f_dir * cup_up).sum(dim=-1)))
    near_w = torch.exp(-(d_eff / (2.0 * cup_r)) ** 2)
    finger_dir = fade * near_w * dir_dot * level

    # ---- A3. 입구 위 정지 비용 (iter_05 3): 파지 전만, 래치 없음, 유계 [0,1].
    inside_w = _sat((side_d - rad_p) / (0.5 * cup_r))
    above_w = _sat((h_palm - cup_mouth_z) / (0.1 * cup_mouth_z))
    hover = (1.0 - held) * inside_w * above_w

    # ---- A4. 손바닥 밀려 들어감 비용 (iter_07 3): 옆 파지 거리 안쪽 AND 입구 높이 아래. 파지 전만.
    push_in = pre_grasp * inside_w * (1.0 - above_w)

    # ---- A2. 잡는 높이로 와서 양쪽에서 감싸기 (rh2_i03 기하). 접촉력 대체 경로는 높이 대역 안에서만
    funnel = 0.5 * (1.0 - torch.tanh(4.0 * geo["d_funnel"])) + 0.5 * torch.exp(-(geo["d_funnel"] / 0.03) ** 2)
    band_t = torch.exp(-(geo["h_err_t"] / 0.02) ** 2)          # (N,)
    band_o = torch.exp(-(geo["h_err_o"] / 0.02) ** 2)          # (N,F-1)
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t * band_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh((f[:, 1:] * band_o).sum(dim=1) / 0.5))
    tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    descend = fade * (0.4 * funnel + 0.6 * tips)

    # ---- 닫기 (iter_07 2, 재시도 [25]): 컵이 손가락 사이에 있을 때만, 닫힘이 오를수록 확실히 오른다 ------------
    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w
    clos_eff = torch.maximum(torch.clamp(hand_closure / 0.7, 0.0, 1.0), pinch_c)
    touch_q = 0.7 + 0.05 * (c_t + c_o) + 0.2 * pinch_c
    close = fade * table_decay * pose_ok * in_hand * clos_eff * touch_q   # 무가중 최대 1.0

    # ---- 접근 중 손 펴기 (iter_08 2): 컵이 손가락 사이가 아닐 때(1 - in_hand) 닫힘이 낮을수록 가산.
    # 손바닥-origin 이 OPEN_R_K x r (약 15 cm) 밖이면 0 (r 폭으로 부드럽게), 파지 후 0.
    # 골 없음: 닫힘에 대한 기울기 close 쪽 >= W_CLOSE x in_hand x touch_q/0.7 >= in_hand,
    #   open 쪽 = W_OPEN x (1 - in_hand) = 0.2 x (1 - in_hand) -> in_hand > 1/6 이면 닫는 쪽이 이득.
    open_near = _sat((OPEN_R_K * cup_r - d3) / cup_r)
    open_hand = pre_grasp * (1.0 - in_hand) * open_near * _sat(1.0 - hand_closure)

    # ---- B. 파지: 성립 전 접촉부터 단계적 (한 손가락 < 양쪽 < 여러 손가락·단단히) ---------------------
    grasp_q = 0.2 * c_o + 0.3 * pinch_c + 0.5 * held * (0.4 + 0.6 * firm)
    grasp = table_decay * pose_ok * grasp_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C. 들기: 품질로 곱하지 않는다 (held x 직립만). 빠른 성분 0.7 x W_LIFT >= 들기 전 수입 합 -> 골짜기 없음
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)   # 과들기 세금: over_h 위에서만
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- 벌점 (무가중, 유계, 래치 없음) --------------------------------------------------------------
    v_lin = cup_lin_vel.norm(dim=-1)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "open": open_hand, "reach_raw": reach_raw, "finger_dir": finger_dir,
            "descend": descend, "close": close, "grasp": grasp, "hover": hover, "push_in": push_in,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기 iter_08. iter_07 위에 두 가지를 바꾼다 (env: palm 증분(속도) 방식):
    (1) 상태형 reach_* 제거 -> 차분형 approach_* (손바닥 속도의 자기 컵 방향 성분 / 10r/s, 선형·홀함수,
        옆 접근 가중 수평^2, 파지 전·컵 직립일 때만, 옆 파지 거리 안쪽 0). 가중 0.4/손.
    (2) open_*: 컵이 손가락 사이가 아닐 때 손을 펼수록 작은 가산 (최대 0.2/손, 약 15 cm 밖 0, 파지 후 0).
    finger_dir, descend, mouth_hover, palm_inside, close, grasp, lift(W_LIFT 5.5), 파지 전 흘림 0, 붓기(c5),
    제어 정규화는 iter_07 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 ------------------------------------------------------------------------------------
    # 들기 전 팔별 상태형 최대 = W_FINGER_DIR 0.4 + W_DESCEND 0.7 + W_CLOSE 1.0 (+ W_OPEN 0.2, close 와 in_hand 로 상보)
    #   <= 2.3 < 들기 빠른 성분 0.7 x W_LIFT 5.5 = 3.85 (iter_07 의 2.9 보다 여유가 크다: 상태형 reach 0.8 제거).
    #   approach(0.4) 는 차분형이라 정지 상태에서 0 -> 합을 늘리지 않는다 (iter_08 노트 4).
    # W_APPROACH 0.4: DG-5F t2r2 와 같은 크기/손. W_OPEN 0.2: close 최대 1.0 보다 작아 컵이 손가락 사이면 닫는 쪽이 이득.
    W_DESCEND, W_CLOSE, W_GRASP, W_LIFT = 0.7, 1.0, 2.0, 5.5
    W_APPROACH = 0.4
    W_OPEN = 0.2
    W_FINGER_DIR = 0.4
    W_HOVER = 0.3
    W_PUSH_IN = 0.3
    W_ALIVE = 0.5          # 상수: 경사를 만들지 않는다
    W_RCV_CROSS = 0.3      # 작은 비용: 리시버 파지 전 손바닥이 자기 컵보다 중심선 쪽으로 넘어감
    UPRIGHT_TOL = math.radians(20.0)   # 받는 컵 성공 한계 20 deg 와 같은 척도
    # 들기 목표·세금 높이 (높이 허용만 절대값 예외): 소스는 높게(K3), 리시버는 낮게
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25  # 바닥 있는 리시버 진행도 게이트 (소스의 들기 이후 수입에만)
    RCV_PEN_SCALE = 0.5    # 리시버 근처 벌점 축소 (rh2_i04 교훈)

    # ---- 공통 척도 (c5) ----------------------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (c5 원문) -------------------------------------------------------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)    # K6
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)

    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    # ---- A/B/C. 팔별 접근·손가락 방향·파지·들기 ----------------------------------------------------
    # 접근(approach)의 "컵이 서 있을 때만" 게이트는 자기 컵 직립도(upright_*)를 쓴다.
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_lin_vel, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_src, upright_src, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_lin_vel, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_rcv, upright_rcv, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 (reach -> 파지 -> 들기), 바닥 있는 계수. reach_raw 는 지급하지 않고 이 게이트에만 쓴다 (붓기 부분 c5 유지)
    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones
    approach_src = W_APPROACH * src["approach"]   # 차분형: 다가가면 +, 물러나면 -, 정지 0 (iter_08 1)
    approach_rcv = W_APPROACH * rcv["approach"]
    open_src = W_OPEN * src["open"]               # 컵이 손가락 사이가 아닐 때 손 펴기 (iter_08 2)
    open_rcv = W_OPEN * rcv["open"]
    finger_dir_src = W_FINGER_DIR * src["finger_dir"]
    finger_dir_rcv = W_FINGER_DIR * rcv["finger_dir"]
    descend_src = W_DESCEND * src["descend"]
    descend_rcv = W_DESCEND * rcv["descend"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
    mouth_hover_src = -W_HOVER * src["hover"]
    mouth_hover_rcv = -W_HOVER * rcv["hover"]
    palm_inside_src = -W_PUSH_IN * src["push_in"]
    palm_inside_rcv = -W_PUSH_IN * rcv["push_in"]
    # 소스: 리시버 목표 높이까지는 팔별(게이트 없음), 그 위 높이(K3 붓기용)·세금만 리시버 진행도 게이트
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    # 리시버 손바닥이 자기 컵보다 중심선 쪽(y 가 컵 y 보다 작은 쪽)으로 넘어가면 작은 비용 (1r 여유, 3r 에서 포화)
    cross_x = _sat((ctx.rcv_cup_pos[:, 1] - ctx.rcv_palm_pos[:, 1] - r_cup) / two_r)
    rcv_cross = -W_RCV_CROSS * (1.0 - rcv["held"]) * cross_x

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- C. 두 컵 수렴 (c5 1.2) --------------------------------------------------------------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- D. lip 조준 (c5 0.8) ----------------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 (c5: 전량 이송) -----------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F. 회전 구동 (c5 8.0) ---------------------------------------------------------------------
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog * rcv_gate

    # ---- F. 실제 도착한 비드 (c5 알당 30, 증분만) ---------------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 흘림 (iter_07 1): 파지 전 0, 두 컵을 쥔 뒤 또는 소스를 든 뒤 알당 40 ------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill_w = 40.0 * post
    spill = -spill_w * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 마무리 (c5 1.5 < 회전 수입 8) -------------------------------------------------------------
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # ---- 성공 보너스 (env 판정, c5 20) -------------------------------------------------------------
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- 조기 기울임 (c5) ---------------------------------------------------------------------------
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- 실기 안전 (c5 / rh2_i03 크기) -------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 붓는 방향이 몸 바깥(+x)을 향하지 않게 (c5) -------------------------------------------------
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 리시버 컵: 직립, 중심선 기준 자기 쪽(+y) (c5) ----------------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- G. 관절 여유: 붓는 중에만 작은 가산 (c5 최대 0.3) -----------------------------------------
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # ---- 컵 교란 벌점 (넘어뜨림 방지용, 작게; 리시버 쪽 RCV_PEN_SCALE) -----------------------------
    tilt_pen_src = (1.0 - lip_gate) * torch.tanh(tilt_src / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    cup_disturb_src = -(0.5 * src["slide"] + 0.3 * tilt_pen_src)
    cup_disturb_rcv = -RCV_PEN_SCALE * (0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv)
    contact_push = -(0.3 * (src["one_sided"] + RCV_PEN_SCALE * rcv["one_sided"])
                     + 0.2 * (src["palm_push"] + RCV_PEN_SCALE * rcv["palm_push"]))
    rush_near_cup = -0.2 * (src["rush"] + RCV_PEN_SCALE * rcv["rush"])
    cup_fling = -0.3 * (src["fling"] + RCV_PEN_SCALE * rcv["fling"])

    # ---- 제어 정규화 (c5, 게이트 없음; 사용자 원칙: 줄이지 않는다) ----------------------------------
    # 증분 방식에서 action_reg_adr(|a|>0.9) 는 "최대 속도로 계속 미는 것"에 대한 비용이다.
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -0.1 * sat

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "open_src": open_src,
        "open_rcv": open_rcv,
        "finger_dir_src": finger_dir_src,
        "finger_dir_rcv": finger_dir_rcv,
        "mouth_hover_src": mouth_hover_src,
        "mouth_hover_rcv": mouth_hover_rcv,
        "palm_inside_src": palm_inside_src,
        "palm_inside_rcv": palm_inside_rcv,
        "descend_src": descend_src,
        "descend_rcv": descend_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "rcv_cross": rcv_cross,
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
        "cup_disturb_src": cup_disturb_src,
        "cup_disturb_rcv": cup_disturb_rcv,
        "contact_push": contact_push,
        "rush_near_cup": rush_near_cup,
        "cup_fling": cup_fling,
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
# iter_09 운영 노트 — 2026-09-28

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다. iter_08 노트가 이 노트 뒤에 이어진다.

## 사용자 지시 (09-28)
- 틱 보고의 (a)·(b) 둘 다 필요: (a) 편 손 보상도 차분형으로, (b) 컵 근처에서 손가락 사이로 들어가는 기울기 강화.
- 단계 원칙: 접근 → 손가락 사이로 컵 → 파지 → 리프트 → 붓기, 다음 단계로 갈 때 보상이 떨어지면 안 된다. 주차(가만히 있어도 받는 수입) 자리 금지.

## iter_08 결과 — t2r_rh4_i08 (env palm 증분 지령, fresh)
- e296: 오른손 컵 근처(손바닥-컵 origin < 10 cm) 0.44 — 같은 epoch 의 i07·i06 은 0. 오른손 손 닫힘 0.56 → 0.28(편 손 접근).
  왼손 손바닥-컵 38 → 22 cm. bounds_loss 0.17 (절대 방식 i04 같은 시기 1.47).
- e350~e696(50 epoch 단위, 20 epoch 평균) 오른손 컵 근처 비율: 0.43, 0.43, 0.11, **0.98**, 0.61, 0.07, 0.24, 0.14. 손바닥-컵 거리 8~15 cm 사이 왕복.
  - 가장 가까이 간 e500(손바닥-컵 8.1 cm): `cup_disturb_src` −0.078, `palm_inside_src` −0.045.
  - 물러난 e600~696(11.7~12.1 cm): `cup_disturb_src` −0.019~−0.021, `palm_inside_src` 0.
  - 그 동안 `open_src` 0.14~0.15/스텝(상한 0.2)이 계속 들어왔다. `open` 은 손바닥-컵 origin 이 약 12 cm(4.4 x 컵 반지름) 안이면
    정지해 있어도 전액이 나오는 상태형 가산이다. `approach_src` 는 0.00~0.02(차분형이라 멈추면 0).
  - [가설] "편 손으로 컵 12 cm 앞에 멈춰 있기"가 벌점 없이 버는 주차 자리가 됐다.
- 왼손도 같은 양상: 컵 근처 0.56~0.82 → 0.28~0.32, `open_rcv` 0.13~0.15.
- 다음 단계 신호 0: 손가락 방향(`finger_dir_*`) 0.000, 엄지 대향 0, 컵이 손가락 사이 0, 닫기 0, 손끝 간격 80~90 mm. 파지 0.
- entropy 22.6(e694), 조기 종료 없음.

## 손가락 제어 사실 (env, 이번 라운드는 바꾸지 않는다)
- 손 액션 6 채널 = 닫힘 목표 (a+1)/2. 스텝당 변화 상한 `synergy_close_speed` 0.005(60 Hz → 0 에서 1 까지 3.3 s).
  네 손가락 채널은 평균으로 묶인다(`couple_four_fingers`). 컵 근처에서만 닫을 수 있다(`close_gate`). 1 N 접촉에서 그 손가락은 멈춘다.
- i06 ep950 왼손: 엄지 명령 +1 을 계속 냈는데 닫힘이 t=60 0.15, t=120 0.45, t=200 0.85. 손가락 사이 자세가 유지된 구간(t=60~100) 동안
  닫힘은 최대 0.3 정도만 오를 수 있었다.

## iter_09 에서 할 것 (iter_08 코드 위의 변경, 이름 유지, 새 ctx 필드 없음)
1. **(a) 편 손 가산을 차분형으로**: `open_*` 를 "정지해 있어도 받는 상태형"에서 "손을 편 채 컵 쪽으로 움직일 때만 받는" 형태로 바꾼다
   (예: 차분형 접근 진척이 양수일 때 x 폄 정도, 또는 접근 진척 자체에 폄 정도를 곱한 가산). 멈추면 0, 물러나면 0 이하.
   컵이 손가락 사이에 있으면 0(닫기가 맡음), 파지 후 0. 이름 `open_src`·`open_rcv` 유지.
2. **(b) 손가락 사이로 들어가는 기울기를 옆 파지 거리 밖(약 12 cm)부터 받게 한다**: 지금 `finger_dir`·`descend` 의 손끝-벽·깔때기 항은
   손바닥-컵 12 cm 에서 거의 0 이다. 옆 파지 거리 밖에서도 "손가락이 컵 축 쪽을 향함"·"엄지와 다른 손가락이 컵 양쪽으로 벌어져
   컵을 사이에 둘 수 있는 배치"가 조금씩 오르게, 가까울수록 커지게(연속, 곱 사슬 금지). 이것들은 상태형이어도 되지만 크기는
   다음 단계(손가락 사이 → 닫기 → 파지)의 수입보다 작아야 한다 — 주차 자리가 되지 않게 그 자리의 합이 "손가락 사이" 자세보다 확실히 낮게.
3. 옆 파지 거리 안으로 들어갈 때 붙는 비용(`palm_inside` 0.3, `cup_disturb` 의 미끄러짐)이 손가락 사이 진입 수입보다 크면 안 된다.
   측정: 8 cm 까지 들어간 e500 에 두 비용 합 −0.12/스텝. 손가락 사이 진입(컵이 엄지-손가락 사이, 손끝 벽 1~2 cm) 자세의 수입이
   12 cm 편 손 정지 자세보다 스텝당 0.5 이상 크게.
4. 그 밖(approach, close, grasp, lift W_LIFT 5.5, mouth_hover, 파지 전 흘림 0, 붓기 c5, 제어 벌점)은 iter_08 그대로.
# iter_08 운영 노트 — 2026-09-28

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다. iter_07 노트가 이 노트 뒤에 이어진다.

## 사용자 지시 (09-28)
- "t2r dg5f-m-s 의 증분 방식으로 고치는 건 어떤지" → 1(env 증분 palm 지령 이식), 2(보상 iter_08) 진행.
- "매번 파지에서 막히니까 LSTM 바꾸는 의미가 없어진다." — 병목은 파지 단계.
- 단계 원칙 유지: 접근 → 손가락 사이로 컵 → 파지 → 리프트 → 붓기, 다음 단계로 갈 때 보상이 떨어지면 안 된다.

## env 변경 (이번 라운드부터, launch 가 `env.palm_action_mode=increment` 를 준다)
- palm 6D 액션을 EMA(α 0.25)로 거른 뒤 **손바닥 목표 속도**(최대 속도 비율)로 읽는다. 최대 속도 소스 0.6 m/s·180°/s,
  리시버 0.4 m/s·120°/s. 목표는 누적되고 델타·palm 박스 안, 현재 실제 palm 근처(소스 6 cm·30°, 리시버 5 cm·15°)로 묶인다.
  → **액션 ±1 은 "박스 끝 자세"가 아니라 "최대 속도"다.** 정지 = 액션 0.
- 새 ctx 필드: `src_palm_lin_vel`, `rcv_palm_lin_vel` (N,3) 손바닥 선속도 [m/s] (env-local 축).
- 근거: 절대 방식에서 palm 명령이 한계에 붙었다 — i04 ep850 오른손 x·z 100 %·roll 59 %, 명령−실제 간극 z 7 cm·roll 0.9 rad;
  i06 ep950 왼손 z·roll 98~100 %.

## iter_07 결과 — t2r_rh4_i07 (fresh, e~700 대에서 교체)
- e694: 오른손(소스) 컵 근처 0.38(이 트랙에서 처음으로 초반 접근; i06 같은 epoch 0), `reach_src` 0.31 → 0.49.
  그러나 **손 닫힘 0.63 으로 쥔 채 다가간다**: 컵이 손가락 사이 0, 손끝 간격 39 mm. 왼손(리시버)은 컵 근처 0, 손바닥-컵 31 cm.
- 파지 0. entropy 25.9. 조기 종료 없음.

## DG-5F 트랙(pour_bi2)에서 확인한 것 (t2r2_i06·i07 코드)
- 접근은 **차분형(진척)**: 손바닥 속도의 컵 origin 방향 성분 / v_ref(컵 반지름 10 배 /s), 속도에 대해 선형·홀함수
  (다가가면 +, 같은 속도로 멀어지면 같은 크기 −, 정지 0, 왕복 합 0). 옆 접근 가중 = (손바닥→origin 방향의 수평 성분)^2
  — 입구 바로 위에서 내려오면 0, 옆에서 다가가면 1. 파지하면 0. 가중 0.4/손.
- 같은 LSTM 구조로 소스 파지·들기 도달 epoch: t2r2_i03m e199/e250, t2r2_i06 e407/e819.

## iter_08 에서 할 것 (iter_07 코드 위의 변경)
1. **접근 점수를 차분형으로 바꾼다**: 상태형 `reach_src`·`reach_rcv`(3D·근거리 거리 점수)를 없애고
   `approach_src`·`approach_rcv` = 손바닥 속도의 자기 컵 origin 방향 성분 / (10 x 컵 반지름 /s), 선형·홀함수,
   옆 접근 가중 (수평 성분)^2, 파지 전만(파지 또는 연속 held 가 서면 0), 컵이 서 있을 때만. 가중 0.4/손.
   옆 파지 거리(컵 반지름 약 2.5 배) 안쪽으로 더 들어가는 진척은 지급하지 않는다(그 안에서는 밀어 넣을 이유가 없다).
2. **다가가는 동안 손을 펴게 한다**(i07 오른손: 닫힘 0.63 으로 접근). 컵이 손가락 사이에 있지 않을 때(iter_07 `close` 의
   in_hand 조건의 여집합) 손 닫힘이 낮을수록 작은 가산(최대 0.2/손), 컵에서 먼 곳(손바닥-origin 15 cm 밖)은 0,
   파지 후 0. 이름 `open_src`·`open_rcv`. 컵이 손가락 사이에 들어온 뒤에는 닫기(`close_*`, 최대 1.0)가 이 가산보다 커서
   닫는 쪽이 늘 이득이어야 한다(골 금지).
3. 그 밖(finger_dir, descend, mouth_hover, palm_inside, close, grasp, lift W_LIFT 5.5, 파지 전 흘림 0, 붓기 c5,
   제어 정규화 action_rate·action_flip·arm_tremor·action_reg_adr)은 iter_07 그대로. 제어 벌점은 줄이지 않는다(사용자 원칙).
   증분 방식에서 액션은 속도라 `action_reg_adr`(|a|>0.9 포화)는 "최대 속도로 계속 미는 것"에 대한 비용이 된다.
4. 들기 전 합 < 0.7 x W_LIFT 는 그대로 지킨다(차분형 접근은 정지 상태에서 0 이라 합을 늘리지 않는다).


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


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """cup_grasp 형태: 0 N -> 0, 0.1 N -> ~0.6, 수십 분의 1 N 위에서 포화 (힘 스파이크가 지배하지 못함)."""
    return torch.tanh(torch.clamp(force, min=0.0) / scale)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    rh2_i03 블록 재사용. 높이 오차(h_err_t, h_err_o)를 추가로 돌려준다 (접촉력 대체 경로의 높이 게이트용).
    iter_05 변경 2 유지: 깔때기 목표는 처음부터 잡는 높이다 (d_funnel == d_grip, ARRIVE_RISE 없음).
    iter_09 (b): straddle = 컵 축이 엄지 손끝과 다른 손가락 손끝(평균) 사이에 끼는 배치 (컵 축에 수직인 평면에서),
    그리고 그 사이 폭이 컵 지름 이상. 손이 컵에서 떨어져 있어도 정의된다 (거리와 무관)."""
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
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

    # iter_09 (b) straddle: 엄지 손끝(a) -> 다른 손가락 손끝 평균(b) 선분 위에서 컵 축(원점)에 가장 가까운 점의 매개변수 t.
    # t 가 선분 가운데(0.25~0.75)면 컵 축이 두 손끝 사이에 있다. 폭 |b-a| 가 컵 지름(2 x tip_r) 이상이면 컵이 들어갈 수 있다.
    a_t = rad_vec[:, 0]
    b_o = o_vec
    ab = b_o - a_t
    ab2 = (ab * ab).sum(dim=-1)
    t_par = -(a_t * ab).sum(dim=-1) / (ab2 + EPS)
    width_ok = _sat(torch.sqrt(ab2 + 1e-12) / (2.0 * tip_r))
    straddle = _band(t_par, 0.25, 0.75, 0.15) * width_ok      # (N,) [0,1]

    # pinch centre: 목표는 잡는 높이의 컵 축 (위에서 도착하는 rise 없음)
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    d_funnel = d_grip
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_funnel": d_funnel, "d_grip": d_grip,
            "thumb_band": thumb_band, "h_err_t": h_err_t, "h_err_o": h_err_o, "straddle": straddle}


def _hand_cup_terms(palm_pos, palm_lin_vel, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                    cup_r, cup_mouth_z, cup_bottom_z, pose_ok, cup_upright, progress,
                    lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (iter_03 cup_grasp 골격 + iter_04 손가락 방향 + iter_05 수평 접근 + iter_07 닫기·밀려 들어감
    + iter_08 차분형 접근(approach) + iter_09 차분형 편 손(open)·원거리 손가락 사이 배치(finger_dir 원거리 성분)).
    두 팔 사이 곱 없음. 들기 전 수입(finger_dir·descend·close)은 held x 컵 높이로 꺼진다.
    reach_raw 는 보상으로 지급하지 않고 리시버 진행도 게이트(rcv_stage)에만 쓴다.
    cup_bottom_z 는 쓰지 않는다 (시그니처만 유지)."""
    EPS = 1e-6
    SIDE_K = 2.5       # 옆 파지 자세의 손바닥-컵 축 수평거리 = 컵 반지름의 약 2.5 배 (i03 ep5900 왼손 7.0 cm)
    H_W = 2.0          # 근거리 성분에서만 높이 오차를 수평 간격의 2 배로 센다
    V_REF_K = 10.0     # iter_08 1: 접근 기준 속도 = 컵 반지름 10 배 /s (DG-5F t2r2 와 같은 척도)
    OPEN_R_K = 5.4     # iter_08 2: 손 펴기 가산의 반경 15 cm 를 컵 반지름 배수로 (0.15 / 0.028 = 5.36)
    FAR_K = 4.0        # iter_09 (b): 원거리 손가락 배치 성분의 거리 척도 = 컵 반지름 4 배 (d_eff 5 cm 에서 약 0.58)
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r)   # 접촉 반경 = ctx.cup_radius
    g_f = grasped.to(cup_pos.dtype)
    f = torch.clamp(finger_force, min=0.0)                     # (N,F)
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)

    # ---- 연속 held (cup_grasp 3 번 형태, RH 판): 엄지 x 다른 손가락, 손가락 수·손바닥 접촉이 오를수록 더
    c = _soft_contact(f)                                       # (N,F)
    c_t = c[:, 0]
    c_o = c[:, 1:].max(dim=1).values
    n_touch = c.sum(dim=1)
    palm_c = _soft_contact(palm_force)
    pinch_c = c_t * c_o                                        # 양쪽 접촉 (없으면 0: 한쪽 접촉은 held 아님)
    count_sat = torch.clamp((n_touch - 2.0) / 2.0, 0.0, 1.0)   # 2 지 -> 0, 4 지 -> 1 (K8: 소스 3 지)
    held = pinch_c * (0.6 + 0.4 * count_sat) * (0.85 + 0.15 * palm_c)
    firm = torch.tanh(torch.minimum(f_t, f_o) / 1.5)           # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 1.0 - torch.maximum(held, g_f)                 # 파지(grasped) 또는 연속 held 가 서면 0

    # ---- 높이 --------------------------------------------------------------------------------------
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)               # 5 mm dead band: tipping on the rim is not lifting
    t_up = torch.tanh(h_pos / 0.02)
    fade = 1.0 - held * t_up                                   # 들기 전 수입 끄기: 컵 높이 기준 (테이블 위 = 1)
    on_table = 1.0 - torch.tanh(h_pos / 0.04)
    late = torch.clamp((progress - 0.2) / 0.8, 0.0, 1.0)
    table_decay = 1.0 - 0.55 * late * on_table                 # cup_grasp 5 번: 후반 테이블 위 파지 수입 감쇠

    # ---- 손바닥의 컵 좌표 (컵 축 기준): 수평거리 rad_p, 높이 h_palm -----------------------------------
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up                            # 컵 축에 수직인 성분 (직립이면 수평)
    rad_p = to_axis.norm(dim=-1)                               # (N,) 손바닥-컵 축 수평거리
    h_palm = -h_c.squeeze(-1)                                  # (N,) 컵 origin 위 손바닥 높이 (컵 축 방향)
    side_d = SIDE_K * cup_r
    grasp_h = 0.25 * cup_mouth_z                               # 잡는 높이

    # ---- reach_raw: 지급하지 않음. 리시버 진행도 게이트(rcv_stage)용으로만 유지 (iter_07 과 같은 식) -----
    d3 = to_cup.norm(dim=-1)                                   # (N,) 손바닥-컵 origin 3D 거리
    side_d3 = (side_d ** 2 + grasp_h ** 2) ** 0.5
    gap3 = torch.clamp(d3 - side_d3, min=0.0)
    reach_far = 1.0 - torch.tanh(gap3 / (7.0 * cup_r))
    gap_xy = torch.clamp(rad_p - side_d, min=0.0)
    h_err_p = torch.abs(h_palm - grasp_h)
    d_eff = torch.sqrt(gap_xy ** 2 + (H_W * h_err_p) ** 2 + 1e-12)
    reach_near = 1.0 - torch.tanh(d_eff / (2.0 * cup_r))
    reach_raw = 0.6 * reach_far + 0.4 * reach_near

    # ---- A. 접근 (iter_08 1): 차분형 진척. 손바닥 속도의 컵 origin 방향 성분 / (10 r /s)
    # 선형·홀함수 (다가가면 +, 같은 속도로 멀어지면 같은 크기 -, 정지 0, 왕복 합 0). 대칭 clamp ±1 (i21: 비대칭 clamp 금지).
    # 옆 접근 가중 = (손바닥->origin 방향의 수평 성분)^2: 입구 바로 위에서 내려오면 0, 옆에서 다가가면 1.
    # 옆 파지 거리(2.5r) 안쪽에서는 지급 0 (더 밀어 넣을 이유 없음, 0.5r 폭으로 부드럽게). 파지 전만, 컵이 서 있을 때만.
    e_c = to_cup / (d3.unsqueeze(-1) + EPS)
    v_to = (palm_lin_vel * e_c).sum(dim=-1)
    prog = torch.clamp(v_to / (V_REF_K * cup_r), -1.0, 1.0)
    side_w = (e_c[:, :2] * e_c[:, :2]).sum(dim=-1)
    out_w = _sat((rad_p - side_d) / (0.5 * cup_r))
    approach_core = cup_upright * side_w * out_w * prog         # 부호 있음 (접근 +, 이탈 -)
    approach = pre_grasp * approach_core

    # ---- A1. 손가락 방향 (iter_04 2 + iter_05 4) + iter_09 (b) 원거리 성분 --------------------------------
    f_dir = palm_axes[:, 3:6]
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    dir_dot = _sat((f_dir * e_in).sum(dim=-1))
    level = _sat(1.0 - torch.abs((f_dir * cup_up).sum(dim=-1)))
    near_w = torch.exp(-(d_eff / (2.0 * cup_r)) ** 2)
    near_part = near_w * dir_dot * level                       # iter_08 그대로의 근거리 성분
    # iter_09 (b): 옆 파지 거리 밖(약 12 cm)부터 연속으로 오르는 성분. 거리 가중은 1 - tanh(d_eff / 4r)
    # (d_eff 0 -> 1, 5 cm -> ~0.58, 30 cm -> ~0.01). 배치 품질은 곱 사슬이 아니라 합:
    # 0.5 x 손가락이 컵 축을 향함 + 0.5 x 컵 축이 엄지와 다른 손가락 사이(straddle).
    far_w = 1.0 - torch.tanh(d_eff / (FAR_K * cup_r))
    far_part = far_w * (0.5 * dir_dot + 0.5 * geo["straddle"])
    # 근거리 0.7 + 원거리 0.3: 손가락 사이 자세(d_eff≈0, 방향·straddle 1)에서 합 1.0 (iter_08 최대와 같음).
    # 12 cm 정지 자세에서는 최대 약 0.7 x 0.45 + 0.3 x 0.58 ≈ 0.49 (가중 후 ≈ 0.2) -> 손가락 사이 자세보다 확실히 낮다.
    finger_dir = fade * (0.7 * near_part + 0.3 * far_part)

    # ---- A3. 입구 위 정지 비용 (iter_05 3): 파지 전만, 래치 없음, 유계 [0,1].
    inside_w = _sat((side_d - rad_p) / (0.5 * cup_r))
    above_w = _sat((h_palm - cup_mouth_z) / (0.1 * cup_mouth_z))
    hover = (1.0 - held) * inside_w * above_w

    # ---- A4. 손바닥 밀려 들어감 비용 (iter_07 3): 옆 파지 거리 안쪽 AND 입구 높이 아래. 파지 전만.
    push_in = pre_grasp * inside_w * (1.0 - above_w)

    # ---- A2. 잡는 높이로 와서 양쪽에서 감싸기 (rh2_i03 기하). 접촉력 대체 경로는 높이 대역 안에서만
    funnel = 0.5 * (1.0 - torch.tanh(4.0 * geo["d_funnel"])) + 0.5 * torch.exp(-(geo["d_funnel"] / 0.03) ** 2)
    band_t = torch.exp(-(geo["h_err_t"] / 0.02) ** 2)          # (N,)
    band_o = torch.exp(-(geo["h_err_o"] / 0.02) ** 2)          # (N,F-1)
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t * band_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh((f[:, 1:] * band_o).sum(dim=1) / 0.5))
    tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    descend = fade * (0.4 * funnel + 0.6 * tips)

    # ---- 닫기 (iter_07 2, 재시도 [25]): 컵이 손가락 사이에 있을 때만, 닫힘이 오를수록 확실히 오른다 ------------
    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w
    clos_eff = torch.maximum(torch.clamp(hand_closure / 0.7, 0.0, 1.0), pinch_c)
    touch_q = 0.7 + 0.05 * (c_t + c_o) + 0.2 * pinch_c
    close = fade * table_decay * pose_ok * in_hand * clos_eff * touch_q   # 무가중 최대 1.0

    # ---- 편 손 접근 (iter_09 (a)): 차분형. 접근 진척(approach_core, 부호 있음) x 폄 정도.
    # 손을 편 채 컵 쪽으로 움직일 때만 +, 멈추면 0, 물러나면 0 이하 (같은 홀함수 prog 를 쓰므로 왕복 합 약 0).
    # 컵이 손가락 사이면 (1 - in_hand) 로 0 (닫기가 맡음), 파지 후 0, 약 15 cm(OPEN_R_K x r) 밖 0 (iter_08 와 같은 범위).
    # 정지 상태에서 0 이므로 12 cm 앞 편 손 정지 자세의 주차 수입이 사라진다 (iter_08 e600~696 open 0.14~0.15/스텝).
    open_near = _sat((OPEN_R_K * cup_r - d3) / cup_r)
    openness = _sat(1.0 - hand_closure)
    open_hand = pre_grasp * (1.0 - in_hand) * open_near * openness * approach_core

    # ---- B. 파지: 성립 전 접촉부터 단계적 (한 손가락 < 양쪽 < 여러 손가락·단단히) ---------------------
    grasp_q = 0.2 * c_o + 0.3 * pinch_c + 0.5 * held * (0.4 + 0.6 * firm)
    grasp = table_decay * pose_ok * grasp_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C. 들기: 품질로 곱하지 않는다 (held x 직립만). 빠른 성분 0.7 x W_LIFT >= 들기 전 수입 합 -> 골짜기 없음
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)   # 과들기 세금: over_h 위에서만
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- 벌점 (무가중, 유계, 래치 없음) --------------------------------------------------------------
    v_lin = cup_lin_vel.norm(dim=-1)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "open": open_hand, "reach_raw": reach_raw, "finger_dir": finger_dir,
            "descend": descend, "close": close, "grasp": grasp, "hover": hover, "push_in": push_in,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기 iter_09. iter_08 위에 두 가지를 바꾼다 (env: palm 증분(속도) 방식, 그대로):
    (a) open_*: 상태형 편 손 가산 -> 차분형 (접근 진척 x 폄 정도; 멈추면 0, 물러나면 0 이하,
        컵이 손가락 사이면 0, 파지 후 0). 가중 0.2/손 유지.
    (b) finger_dir_*: 옆 파지 거리 밖(약 12 cm)부터 연속으로 오르는 원거리 성분
        (손가락이 컵 축을 향함 + 컵 축이 엄지와 다른 손가락 사이) 추가. 무가중 최대 1.0 유지.
    approach, descend, mouth_hover, palm_inside, close, grasp, lift(W_LIFT 5.5), 파지 전 흘림 0, 붓기(c5),
    제어 정규화는 iter_08 그대로."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 ------------------------------------------------------------------------------------
    # 들기 전 팔별 상태형 최대 = W_FINGER_DIR 0.4 + W_DESCEND 0.7 + W_CLOSE 1.0 = 2.1
    #   < 들기 빠른 성분 0.7 x W_LIFT 5.5 = 3.85. iter_09 에서 open(0.2) 은 차분형이 되어 정지 합에서 빠졌다.
    #   approach(0.4)·open(0.2) 는 차분형이라 정지 상태에서 0 -> 합을 늘리지 않는다.
    # 손가락 사이 진입 자세 (추정 계산): finger_dir ~0.4 + descend ~0.4 + close(편 손) ~0.1 ≈ 0.9/스텝.
    # 12 cm 편 손 정지 자세 (추정 계산): finger_dir <= ~0.2 + descend ~0.1 + open 0 ≈ 0.3/스텝 -> 차이 >= 0.5.
    # 옆 파지 거리 안 진입 비용 (e500 측정 합 -0.12/스텝) 은 이 차이보다 작다.
    W_DESCEND, W_CLOSE, W_GRASP, W_LIFT = 0.7, 1.0, 2.0, 5.5
    W_APPROACH = 0.4
    W_OPEN = 0.2           # 차분형 편 손 가산: approach 0.4 보다 작은 보조 (편 손으로 다가갈 때 최대 +0.2/손)
    W_FINGER_DIR = 0.4
    W_HOVER = 0.3
    W_PUSH_IN = 0.3
    W_ALIVE = 0.5          # 상수: 경사를 만들지 않는다
    W_RCV_CROSS = 0.3      # 작은 비용: 리시버 파지 전 손바닥이 자기 컵보다 중심선 쪽으로 넘어감
    UPRIGHT_TOL = math.radians(20.0)   # 받는 컵 성공 한계 20 deg 와 같은 척도
    # 들기 목표·세금 높이 (높이 허용만 절대값 예외): 소스는 높게(K3), 리시버는 낮게
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25  # 바닥 있는 리시버 진행도 게이트 (소스의 들기 이후 수입에만)
    RCV_PEN_SCALE = 0.5    # 리시버 근처 벌점 축소 (rh2_i04 교훈)

    # ---- 공통 척도 (c5) ----------------------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup
    wide = 10.0 * r_cup
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    nested = ctx.cups_nested.to(dt)

    # ---- 붓기 기하 (c5 원문) -------------------------------------------------------------------------
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)    # K6
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)

    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    # ---- A/B/C. 팔별 접근·손가락 방향·파지·들기 ----------------------------------------------------
    # 접근(approach)·편 손(open)의 "컵이 서 있을 때만" 게이트는 자기 컵 직립도(upright_*)를 쓴다.
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_lin_vel, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_src, upright_src, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_lin_vel, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_rcv, upright_rcv, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 (reach -> 파지 -> 들기), 바닥 있는 계수. reach_raw 는 지급하지 않고 이 게이트에만 쓴다 (붓기 부분 c5 유지)
    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones
    approach_src = W_APPROACH * src["approach"]   # 차분형: 다가가면 +, 물러나면 -, 정지 0 (iter_08 1)
    approach_rcv = W_APPROACH * rcv["approach"]
    open_src = W_OPEN * src["open"]               # 차분형: 편 손으로 다가갈 때만 +, 정지 0 (iter_09 (a))
    open_rcv = W_OPEN * rcv["open"]
    finger_dir_src = W_FINGER_DIR * src["finger_dir"]   # 근거리 0.7 + 원거리 0.3 (iter_09 (b))
    finger_dir_rcv = W_FINGER_DIR * rcv["finger_dir"]
    descend_src = W_DESCEND * src["descend"]
    descend_rcv = W_DESCEND * rcv["descend"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
    mouth_hover_src = -W_HOVER * src["hover"]
    mouth_hover_rcv = -W_HOVER * rcv["hover"]
    palm_inside_src = -W_PUSH_IN * src["push_in"]
    palm_inside_rcv = -W_PUSH_IN * rcv["push_in"]
    # 소스: 리시버 목표 높이까지는 팔별(게이트 없음), 그 위 높이(K3 붓기용)·세금만 리시버 진행도 게이트
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    # 리시버 손바닥이 자기 컵보다 중심선 쪽(y 가 컵 y 보다 작은 쪽)으로 넘어가면 작은 비용 (1r 여유, 3r 에서 포화)
    cross_x = _sat((ctx.rcv_cup_pos[:, 1] - ctx.rcv_palm_pos[:, 1] - r_cup) / two_r)
    rcv_cross = -W_RCV_CROSS * (1.0 - rcv["held"]) * cross_x

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- C. 두 컵 수렴 (c5 1.2) --------------------------------------------------------------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- D. lip 조준 (c5 0.8) ----------------------------------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 (c5: 전량 이송) -----------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F. 회전 구동 (c5 8.0) ---------------------------------------------------------------------
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog * rcv_gate

    # ---- F. 실제 도착한 비드 (c5 알당 30, 증분만) ---------------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 흘림 (iter_07 1): 파지 전 0, 두 컵을 쥔 뒤 또는 소스를 든 뒤 알당 40 ------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill_w = 40.0 * post
    spill = -spill_w * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 마무리 (c5 1.5 < 회전 수입 8) -------------------------------------------------------------
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # ---- 성공 보너스 (env 판정, c5 20) -------------------------------------------------------------
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- 조기 기울임 (c5) ---------------------------------------------------------------------------
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- 실기 안전 (c5 / rh2_i03 크기) -------------------------------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 붓는 방향이 몸 바깥(+x)을 향하지 않게 (c5) -------------------------------------------------
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 리시버 컵: 직립, 중심선 기준 자기 쪽(+y) (c5) ----------------------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- G. 관절 여유: 붓는 중에만 작은 가산 (c5 최대 0.3) -----------------------------------------
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # ---- 컵 교란 벌점 (넘어뜨림 방지용, 작게; 리시버 쪽 RCV_PEN_SCALE) -----------------------------
    tilt_pen_src = (1.0 - lip_gate) * torch.tanh(tilt_src / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    cup_disturb_src = -(0.5 * src["slide"] + 0.3 * tilt_pen_src)
    cup_disturb_rcv = -RCV_PEN_SCALE * (0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv)
    contact_push = -(0.3 * (src["one_sided"] + RCV_PEN_SCALE * rcv["one_sided"])
                     + 0.2 * (src["palm_push"] + RCV_PEN_SCALE * rcv["palm_push"]))
    rush_near_cup = -0.2 * (src["rush"] + RCV_PEN_SCALE * rcv["rush"])
    cup_fling = -0.3 * (src["fling"] + RCV_PEN_SCALE * rcv["fling"])

    # ---- 제어 정규화 (c5, 게이트 없음; 사용자 원칙: 줄이지 않는다) ----------------------------------
    # 증분 방식에서 action_reg_adr(|a|>0.9) 는 "최대 속도로 계속 미는 것"에 대한 비용이다.
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -0.1 * sat

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "open_src": open_src,
        "open_rcv": open_rcv,
        "finger_dir_src": finger_dir_src,
        "finger_dir_rcv": finger_dir_rcv,
        "mouth_hover_src": mouth_hover_src,
        "mouth_hover_rcv": mouth_hover_rcv,
        "palm_inside_src": palm_inside_src,
        "palm_inside_rcv": palm_inside_rcv,
        "descend_src": descend_src,
        "descend_rcv": descend_rcv,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "rcv_cross": rcv_cross,
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
        "cup_disturb_src": cup_disturb_src,
        "cup_disturb_rcv": cup_disturb_rcv,
        "contact_push": contact_push,
        "rush_near_cup": rush_near_cup,
        "cup_fling": cup_fling,
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

## Review items

[31] 왼손 12 cm 편 손 정지 +0.812 · 손가락 사이 자세 +1.140 (차 +0.328) · finger_dir +0.123→+0.218
    ✗ 손가락 사이 자세가 12 cm 편 손 정지보다 스텝당 0.5 이상 이득이 아니다

(검사 정의: [31] 은 iter_06 ep950 왼손 실제 재생 자세(t=60~100: 손바닥 수평 7.7 cm·높이 +2.4 cm, 손가락이 컵 축 쪽 0.91,
손끝-벽 0~1.5 cm, 엄지 접촉, 닫힘 0.12)와, 같은 자세를 손바닥이 컵 축에서 12 cm 가 되게 수평으로 옮기고 손을 편(닫힘 0, 접촉 0)
정지 자세를 비교한다. 두 손바닥 속도 0. 노트 iter_09 3 번의 "0.5 이상" 기준.)
