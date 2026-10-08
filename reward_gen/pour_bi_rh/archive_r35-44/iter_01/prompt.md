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
  actions[0:7]   = source ARM joint increments (joints 1..7, shoulder to wrist). Each policy step the \
arm joint target moves by 0.05 rad x a, is smoothed by an exponential moving average (alpha 0.1) and \
clamped to the joint limits; the PD controller gets a position target only. The effective maximum \
joint speed is about 0.3 rad/s. a = 0 means "hold the current joint target", not "go home".
  actions[7:13]  = source HAND joint targets, one per driven joint, in this order: thumb abduction, \
thumb flexion, index, middle, ring, pinky. Each value is an ABSOLUTE target over the range between the \
open pose and the nominal grasp pose of that joint (a = -1 -> the lower of the two, a = +1 -> the higher), \
smoothed by an EMA (alpha 0.1) and limited to the vendor speed (full joint range in 1 s). The fingers cannot \
be driven past the grasp pose. Like the real RH56F1 firmware force stop, a finger whose links touch its \
own cup with more than 1 N does not close any further (opening is always allowed). There is no coupling \
between fingers and no proximity gate. In action units the open hand is thumb abduction +1, every other \
drive -1; the grasp pose is thumb abduction -1, every other drive +1; a = 0 is half way (closure 0.5).
  actions[13:20] = receiver ARM joint increments (same law as the source arm)
  actions[20:26] = receiver HAND joint targets (same law as the source hand)
`*_hand_closure` is measured from the joint positions: 0 = open pose, 1 = nominal grasp pose (clipped). \
Because hand commands are absolute joint targets, |a| close to 1 on a hand drive is a normal command \
(fully open or fully at the grasp pose), not a saturation fault; on an arm drive |a| close to 1 means \
"move that joint at full speed".
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
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """0 N -> 0, 0.1 N -> ~0.6, 수십 분의 1 N 위에서 포화 (힘 스파이크가 지배하지 못함)."""
    return torch.tanh(torch.clamp(force, min=0.0) / scale)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """컵 좌표(축 = cup_up)에서의 손끝 기하. 이전 코드와 같다.
    straddle = 컵 축이 엄지 손끝과 다른 손가락 손끝(평균) 사이에 끼는 배치 (거리와 무관하게 정의)."""
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # 잡는 높이 (컵 원점 위)
    thumb_hi = 0.5 * cup_mouth_z        # 엄지 높이 대역 상한

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F)
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values

    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # 엄지 대향 [0,1]

    a_t = rad_vec[:, 0]
    ab = o_vec - a_t
    ab2 = (ab * ab).sum(dim=-1)
    t_par = -(a_t * ab).sum(dim=-1) / (ab2 + EPS)
    width_ok = _sat(torch.sqrt(ab2 + 1e-12) / (2.0 * tip_r))
    straddle = _band(t_par, 0.25, 0.75, 0.15) * width_ok

    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_funnel": d_grip, "d_grip": d_grip,
            "thumb_band": thumb_band, "h_err_t": h_err_t, "h_err_o": h_err_o, "straddle": straddle}


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped,
                    arm_qd, cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                    cup_r, cup_mouth_z, pose_ok, cup_upright, progress,
                    lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항.
    iter_00 (재편 두 번째 판) + 재시도 변경:
      - 접근 유효거리: L2 -> L1 (수평 틈 + 2 x 높이 오차). L2 에서는 높이 오차가 클 때 수평 기울기가 사라졌다([24] 수평 6cm +0.006).
      - 접근 먼 척도: 1 - tanh -> 선형 (18 r 까지 일정한 기울기). 가까운 척도 3 r tanh 는 유지.
      - 편 손 인자 해제: in_hand 외에 손바닥이 옆 목표(2.5 r) 안이면 해제 ([25] 닫기 수입 상쇄 방지).
      - push_in 은 손바닥이 2 r 안으로 들어올 때만 ([15] 6 cm 옆에서 비용이 접근 수입을 깎음).
      - 닫기 touch_q 바닥 0.7 -> 0.85 (최대 1.0 유지)."""
    EPS = 1e-6
    SIDE_K = 2.5          # 손바닥 목표 = 컵 축에서 2.5 r 옆
    H_W = 2.0             # 높이 차는 수평의 2 배로 센다 (브리프 5-1)
    APP_LIN_K = 18.0      # 접근 선형 척도 18 r: 시작 자세(L1 유효거리 약 13 r)에서도 일정한 기울기
    APP_NEAR_K = 3.0      # 접근 가까운 척도 3 r: 15 cm -> 6 cm 구간에서 수입이 가파르게 오름
    FAR_K = 4.0
    NEAR_K = 1.5
    FUNNEL_FAR_K = 2.0
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, cup_r)
    g_f = grasped.to(cup_pos.dtype)
    f = torch.clamp(finger_force, min=0.0)
    f_t = f[:, 0]
    f_o = f[:, 1:].sum(dim=1)

    # ---- 연속 held: 엄지 x 다른 손가락 (한쪽 접촉은 held 아님)
    c = _soft_contact(f)
    c_t = c[:, 0]
    c_o = c[:, 1:].max(dim=1).values
    n_touch = c.sum(dim=1)
    palm_c = _soft_contact(palm_force)
    pinch_c = c_t * c_o
    count_sat = torch.clamp((n_touch - 2.0) / 2.0, 0.0, 1.0)
    held = pinch_c * (0.6 + 0.4 * count_sat) * (0.85 + 0.15 * palm_c)
    firm = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    pre_grasp = 1.0 - torch.maximum(held, g_f)

    # ---- 높이
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)
    t_up = torch.tanh(h_pos / 0.02)
    fade = 1.0 - held * t_up                                   # 들기 전 수입 끄기 (lift 와 같은 held x t_up)
    on_table = 1.0 - torch.tanh(h_pos / 0.04)
    late = torch.clamp((progress - 0.2) / 0.8, 0.0, 1.0)
    table_decay = 1.0 - 0.55 * late * on_table

    # ---- 손바닥의 컵 좌표 (원점 기준)
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up
    rad_p = to_axis.norm(dim=-1)
    h_palm = -h_c.squeeze(-1)
    side_d = SIDE_K * cup_r
    grasp_h = 0.25 * cup_mouth_z

    # ---- 원점 기준 유효 거리 (손가락 방향·리시버 게이트용 L2, 이전과 같음)
    d3 = to_cup.norm(dim=-1)
    gap_xy = torch.clamp(rad_p - side_d, min=0.0)
    h_err_p = torch.abs(h_palm - grasp_h)
    d_eff = torch.sqrt(gap_xy ** 2 + (H_W * h_err_p) ** 2 + 1e-12)

    # ---- reach_raw: 지급하지 않음 (리시버 진행도 게이트 전용, 이전 코드와 같음)
    side_d3 = (side_d ** 2 + grasp_h ** 2) ** 0.5
    gap3 = torch.clamp(d3 - side_d3, min=0.0)
    reach_far = 1.0 - torch.tanh(gap3 / (7.0 * cup_r))
    reach_near = 1.0 - torch.tanh(d_eff / (2.0 * cup_r))
    reach_raw = 0.6 * reach_far + 0.4 * reach_near

    # ---- 컵이 손가락 사이에 있는 정도 (편 손 인자 해제 + 닫기 게이트)
    opp_w = _sat((geo["opp"] - 0.4) / 0.3)
    d_tip_max = torch.maximum(geo["d_thumb"], geo["d_oth"])
    tip_w = torch.exp(-(d_tip_max / (1.5 * cup_r)) ** 2)
    grip_w = torch.exp(-(geo["d_grip"] / (3.0 * cup_r)) ** 2)
    in_hand = opp_w * tip_w * grip_w

    # ---- A. 접근 + 편 손: 상태형 (브리프 5-0, 5-1). 파지가 서면 pre_grasp 로 사라진다.
    # L1 유효거리: 수평 틈과 높이 오차(x2)가 각자 기울기를 가진다 (높이가 커도 수평 6 cm 가 값을 가짐).
    d_app = gap_xy + H_W * h_err_p
    app_lin = _sat(1.0 - d_app / (APP_LIN_K * cup_r))          # 먼 척도: 일정 기울기
    app_near = 1.0 - torch.tanh(d_app / (APP_NEAR_K * cup_r))  # 가까운 척도
    app_state = 0.6 * app_lin + 0.4 * app_near
    openness = _sat(1.0 - hand_closure)
    # 해제: 컵이 손가락 사이(in_hand) 이거나 손바닥이 옆 목표(2.5 r) 안이면 편 손 인자를 푼다.
    #   2.5 r + 1 r 밖(12 cm 옆 1b, 10 cm 비교)에서는 인자가 그대로 -> 편 손 > 쥔 손.
    near_gate = _sat((side_d + cup_r - rad_p) / cup_r)
    release = torch.maximum(_sat(in_hand / 0.5), near_gate)
    open_fac = release + (1.0 - release) * (0.45 + 0.55 * openness)
    approach = pre_grasp * cup_upright * app_state * open_fac

    # ---- B. 손가락 방향 (근거리 0.7 + 원거리 0.3, 곱 사슬 아닌 합)
    f_dir = palm_axes[:, 3:6]
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    dir_dot = _sat((f_dir * e_in).sum(dim=-1))
    level = _sat(1.0 - torch.abs((f_dir * cup_up).sum(dim=-1)))
    near_w = torch.exp(-(d_eff / (NEAR_K * cup_r)) ** 2)
    near_part = near_w * dir_dot * level
    far_w = 1.0 - torch.tanh(d_eff / (FAR_K * cup_r))
    far_part = far_w * (0.5 * dir_dot + 0.5 * geo["straddle"])
    finger_dir = fade * (0.7 * near_part + 0.3 * far_part)

    # ---- B 비용: 입구 위 정지 / 손바닥 밀려 들어감
    inside_w = _sat((side_d - rad_p) / (0.5 * cup_r))
    above_w = _sat((h_palm - cup_mouth_z) / (0.1 * cup_mouth_z))
    hover = (1.0 - held) * inside_w * above_w
    # push_in: 손바닥이 컵 축에서 2 r 안(벽에 거의 붙음)일 때만. 옆 목표(2.5 r) 부근 접근은 비용 없음.
    push_w = _sat((2.0 * cup_r - rad_p) / (0.5 * cup_r))
    push_in = pre_grasp * push_w * (1.0 - above_w)

    # ---- B. 잡는 높이로 와서 양쪽에서 감싸기
    d_fn = geo["d_funnel"]
    funnel = 0.5 * (1.0 - torch.tanh(d_fn / (FUNNEL_FAR_K * cup_r))) + 0.5 * torch.exp(-(d_fn / 0.03) ** 2)
    band_t = torch.exp(-(geo["h_err_t"] / 0.02) ** 2)
    band_o = torch.exp(-(geo["h_err_o"] / 0.02) ** 2)
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t * band_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh((f[:, 1:] * band_o).sum(dim=1) / 0.5))
    tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    descend = fade * (0.3 * funnel + 0.7 * tips)

    # ---- C. 닫기: 컵이 손가락 사이에 있을 때만, 폐쇄도·손끝 접촉에 지불
    # touch_q 바닥 0.85 (최대 1.0 은 이전과 같음): 접촉 전에도 닫는 기울기가 충분하게.
    clos_eff = torch.maximum(torch.clamp(hand_closure / 0.7, 0.0, 1.0), pinch_c)
    touch_q = 0.85 + 0.05 * (c_t + c_o) + 0.05 * pinch_c
    close = fade * table_decay * pose_ok * in_hand * clos_eff * touch_q

    # ---- C. 파지: 한 손가락 < 양쪽 < 여러 손가락·단단히
    grasp_q = 0.2 * c_o + 0.3 * pinch_c + 0.5 * held * (0.4 + 0.6 * firm)
    grasp = table_decay * pose_ok * grasp_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- D. 들기 (held x 자세만, 과들기 세금은 over_h 위에서만)
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)

    # ---- G. 교란 (무가중, 유계, 래치 없음). slide 는 미는 속도(차분)만 (브리프 5-3 유지).
    v_lin = cup_lin_vel.norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * torch.tanh(v_xy / 0.05)
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(torch.clamp(palm_force, min=0.0) / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    # 급한 팔: 관절 속도 rms / 1 rad/s
    rush = near * (1.0 - g_f) * torch.tanh(_rms(arm_qd) / 1.0)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"approach": approach, "reach_raw": reach_raw, "finger_dir": finger_dir,
            "descend": descend, "close": close, "grasp": grasp, "hover": hover, "push_in": push_in,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기, 4번째 사이클 iter_00 재편 두 번째 판 재시도 (full-joint 26 = 팔마다 [팔 증분 7, 손 절대 목표 6]).
    이전 응답 위의 변경 (Review items [15]·[24]·[25]):
    1. 접근: L1 유효거리 + 선형 먼 척도(18 r) + 가까운 척도(3 r), W 0.5 -> 0.7.
    2. 편 손 인자: 손바닥이 옆 목표(2.5 r) 안이거나 컵이 손가락 사이면 해제.
    3. push_in 은 손바닥 2 r 안에서만. 닫기 touch_q 바닥 0.85.
    4. 들기 전 상태 합 = 0.7 + 0.4 + 1.6 + 1.0 = 3.7 < 들기 빠른 성분 0.7 x 5.5 = 3.85 (골 없음). 항 25 개."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 ------------------------------------------------------------------------------------
    # 들기 전 팔별 상태형 최대 = W_APPROACH 0.7 + W_FINGER_DIR 0.4 + W_DESCEND 1.6 + W_CLOSE 1.0 = 3.7
    #   < 들기 빠른 성분 0.7 x W_LIFT 5.5 = 3.85 -> 들기로 넘어갈 때 골짜기 없음.
    #   approach 는 파지 때 사라지고 grasp(W 2.0) 가 이어받는다.
    W_DESCEND, W_CLOSE, W_GRASP, W_LIFT = 1.6, 1.0, 2.0, 5.5
    W_APPROACH = 0.7       # 상태형 접근: 시작 자세 수평 6 cm 에서 약 +0.05 (선형 척도 0.6 x 0.06/18r x 0.7)
    W_FINGER_DIR = 0.4
    W_HOVER = 0.2          # 컵 근처 비용 < 접근 수입 (브리프 5-1)
    W_PUSH_IN = 0.15
    W_RUSH = 0.1
    W_ALIVE = 0.5          # 상수: 경사를 만들지 않는다
    W_RCV_CROSS = 0.3
    UPRIGHT_TOL = math.radians(20.0)   # 받는 컵 성공 한계 20 deg 와 같은 척도
    # 들기 목표·세금 높이 (높이 허용만 절대값 예외, 이전 코드 유지): 소스는 높게(K3), 리시버는 낮게
    LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC = 0.28, 0.45, 0.20
    LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV = 0.08, 0.15, 0.10
    RCV_GATE_FLOOR = 0.25
    RCV_PEN_SCALE = 0.5

    # ---- 공통 척도 ----------------------------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
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
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)   # 컵 반지름 창

    upright_src = torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = lip_gate + (1.0 - lip_gate) * upright_src
    pose_ok_rcv = upright_rcv

    # ---- 팔별 항 ------------------------------------------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos,
                          ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped,
                          ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, r_cup, ctx.cup_mouth_z,
                          pose_ok_src, upright_src, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos,
                          ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped,
                          ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, r_cup, ctx.cup_mouth_z,
                          pose_ok_rcv, upright_rcv, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 게이트 (바닥 있음): 소스의 들기 이후 수입에만
    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones

    # ---- A 접근: 상태형 (원점 기준 L1 거리 x 편 손 인자, 파지 후 소멸) ---------------------------------
    approach_src = W_APPROACH * src["approach"]
    approach_rcv = W_APPROACH * rcv["approach"]

    # ---- B 입구: 손가락 방향(원거리부터 연속) + 잡는 높이로 내려와 감싸기 ------------------------------
    mouth_src = W_FINGER_DIR * src["finger_dir"] + W_DESCEND * src["descend"]
    mouth_rcv = W_FINGER_DIR * rcv["finger_dir"] + W_DESCEND * rcv["descend"]

    # B 비용 하나로: 입구 위 정지 + 손바닥 밀려 들어감 + 컵 근처 급한 팔 + 리시버 손바닥 중심선 넘어감
    cross_x = _sat((ctx.rcv_cup_pos[:, 1] - ctx.rcv_palm_pos[:, 1] - r_cup) / two_r)
    mouth_cost = -(W_HOVER * (src["hover"] + rcv["hover"])
                   + W_PUSH_IN * (src["push_in"] + rcv["push_in"])
                   + W_RUSH * (src["rush"] + RCV_PEN_SCALE * rcv["rush"])
                   + W_RCV_CROSS * (1.0 - rcv["held"]) * cross_x)

    # ---- C 파지: 닫기(W 1.0) -> 양쪽 접촉 파지(W 2.0 이 이어받음) ------------------------------------
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]

    # ---- D 들기: 소스는 리시버 목표 높이 위(K3 붓기용)·세금만 리시버 진행도 게이트 ---------------------
    lift_src = W_LIFT * (src["lift_lo"] + src["lift_hi"] * rcv_gate)
    lift_rcv = W_LIFT * (rcv["lift_lo"] + rcv["lift_hi"])

    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- E 배치: 두 컵 수렴 (1.2) + 붓는 방향 (-1.5) + 리시버 직립·자기 쪽 (0.5 / -1.5) -------------
    converge_cups = 1.2 * both_g * lifted * rcv_gate * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad
    place = converge_cups + pour_dir + rcv_keep

    # ---- E 조준: 소스 주둥이 lip 을 입구 위 창 안에 (0.8) ----------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim = 0.8 * both_g * lifted * rcv_gate * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부 --------------------------------------------------------------------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F 회전 (8.0): 창 안·조준 품질 x 회전 진척. 각도 처방 없음 ------------------------------------
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

    # 관절 여유: 붓는 중에만 작은 가산 (최대 0.3, 강한 직접 벌점 아님)
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # 마무리 (1.5 < 회전 수입 8)
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # 성공 보너스 (env 판정)
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- G 흘림: 파지 전 0, 두 컵 쥔 뒤 또는 소스 든 뒤 알당 40 ----------------------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill = -40.0 * post * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # 창 밖 조기 기울임
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- G 충돌 (실기 안전): 컵-컵 + 포개짐 + 남의 것 건드리기 ----------------------------------------
    cup_clash = -((1.0 + 1.0 * both_g) * torch.tanh(ctx.cup_cup_force / 5.0) + 3.0 * nested)
    hand_foreign = -(0.5 + 0.5 * both_g) * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    collision = cup_clash + hand_foreign

    # ---- G 컵 교란 (넘어뜨림 방지, 리시버 쪽 RCV_PEN_SCALE) --------------------------------------------
    # 기울기 벌점은 1 - exp(-(tilt/20deg)^2): 작은 기울기(5 deg -> 0.06)는 싸고 넘어짐(90 deg)은 1.
    tilt_pen_src = (1.0 - lip_gate) * (1.0 - torch.exp(-(tilt_src / UPRIGHT_TOL) ** 2))
    tilt_pen_rcv = 1.0 - torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    dist_src = 0.5 * src["slide"] + 0.3 * tilt_pen_src + 0.3 * src["one_sided"] \
        + 0.2 * src["palm_push"] + 0.3 * src["fling"]
    dist_rcv = 0.5 * rcv["slide"] + 0.3 * tilt_pen_rcv + 0.3 * rcv["one_sided"] \
        + 0.2 * rcv["palm_push"] + 0.3 * rcv["fling"]
    cup_disturb = -(dist_src + RCV_PEN_SCALE * dist_rcv)

    # ---- G 제어 정규화 (cup_grasp 수준, 브리프 5-2, 이전 응답과 같음) ------------------------------------
    # 팔 채널 = [0:7] (소스), [13:20] (리시버): 관절 증분. 손 채널 = [7:13], [20:26]: 절대 목표 (|a|~1 은 정상, 크기 벌점 없음).
    # 반전(flip)·|a|>0.9 포화 벌점은 관절 증분의 탐색을 억누르므로 두지 않는다.
    a = ctx.actions
    pa = ctx.prev_actions
    a_arm = torch.cat([a[:, 0:7], a[:, 13:20]], dim=-1)
    pa_arm = torch.cat([pa[:, 0:7], pa[:, 13:20]], dim=-1)
    a_hand = torch.cat([a[:, 7:13], a[:, 20:26]], dim=-1)
    pa_hand = torch.cat([pa[:, 7:13], pa[:, 20:26]], dim=-1)
    da_arm = a_arm - pa_arm
    rate_arm = 0.02 * torch.tanh((da_arm * da_arm).mean(dim=-1) / 0.1)
    mag_arm = 0.01 * (a_arm * a_arm).mean(dim=-1)
    qd_src = ctx.src_arm_qd
    qd_rcv = ctx.rcv_arm_qd
    speed_arm = 0.12 * 0.5 * (torch.tanh(_rms(qd_src) / 1.0) + torch.tanh(_rms(qd_rcv) / 1.0))
    runaway = 0.1 * 0.5 * (
        torch.tanh(torch.clamp(qd_src.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0)
        + torch.tanh(torch.clamp(qd_rcv.abs().max(dim=-1).values - 3.0, min=0.0) / 1.0))
    ctrl_arm = -(rate_arm + mag_arm + speed_arm + runaway)
    da_hand = a_hand - pa_hand
    ctrl_hand = -0.005 * torch.tanh((da_hand * da_hand).mean(dim=-1) / 0.1)   # 손 목표 변경은 싸게

    components = {
        "alive": alive,
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "mouth_src": mouth_src,
        "mouth_rcv": mouth_rcv,
        "mouth_cost": mouth_cost,
        "close_src": close_src,
        "close_rcv": close_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
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
# pour_bi_rh 보상 설계 브리프 (2026-09-28 판 — 4번째 사이클 iter_00, full-joint 제어)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트에 통째로 들어간다.**

이 트랙은 세 번 리셋했다. 1~12 라운드 `archive_r1-12`, 13~24 라운드 `archive_r13-24`, 3번째 사이클 iter_00~09 는 `archive_r25-34`.
지금은 4번째 사이클의 iter_00 이다. **바뀐 것은 제어다**(palm 지령 + fabric → 양팔 full-joint). 보상은 3번째 사이클 마지막 코드(iter_09)를
바탕으로 구조를 다시 짠다 — 새로 발명하지 않는다. 사용자 판단(09-28): "제어만 바뀌지 reward 문제라고 생각하진 않는다."

## 1. 과제와 고정 조건 (바꿀 수 없다)

- 오른팔(소스, `src_*`)이 비드 20 개가 든 컵을, 왼팔(리시버, `rcv_*`)이 빈 컵을 잡아 든다. 소스를 기울여 비드를 리시버 컵에 옮긴다.
- 성공(env 가 계산, 보상이 재정의 못 함): 담김 ≥ 0.80 · 흘림 ≤ 0.25 · 두 컵 파지 유지 · 받는 컵 ≤ 20° · 20 스텝 유지 · 컵이 포개지지 않음.
- **액션 26 = 팔마다 13** — [팔 관절 증분 7, 손 구동 관절 절대 목표 6]. 로봇 설명 절이 정의한다. 손은 결합·게이트·접촉 동결이 없다.
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

## 4. 3번째 사이클(iter_00~09)에서 확정된 교훈

| 라운드 | 사실 | 교훈 |
|---|---|---|
| i03 | 표면 거리 접근은 입구 원판을 표면으로 쳐서, 손바닥이 입구 위에 있으면 거리 1.7 cm·접근 수입 0.75 | 접근은 **컵 원점 기준**, 옆에서(수평 성분 가중) |
| i04 | 깔때기 목표가 "잡는 높이보다 4.5 cm 위에서 도착 후 내려감"이라 위에서 끼우는 동작을 가르쳤다, 파지 0 | 손바닥을 먼저 내리고 테이블과 수평으로 접근 |
| i07 | 손을 쥔 채(폐쇄도 0.63) 다가갔다 | 다가가는 동안 손을 펴게 하는 항이 필요 |
| i08 | 상태형 "편 손" 가산이 컵 12 cm 앞 주차 자리를 만들었다(0.13~0.16/스텝) | 편 손 가산도 **차분형**(편 채 다가갈 때만) |
| f00 (4번째 사이클 첫 iter_00, full-joint) | e0~470: 접근 보상(손바닥 속도 차분) −0.001~−0.018, 손바닥-컵 21~43 cm 왕복, 컵 근처 0(i09 같은 epoch 0.41), 손 폐쇄도 0.59~0.89 유지, 팔 제어 벌점 −0.14~−0.19(가장 큰 항), 팔 목표가 관절 한계 2 % 안 0.02→0.17. 결정론 검사: 시작 자세 6 cm 다가감 +0.002, 15→6 cm −0.122, 편 손−쥔 손 0, 팔 0.3 rad/s 비용 0.378 | **관절 제어에서는 속도 차분 접근이 신호가 되지 않는다** — 상태형 접근과 작은 제어 벌점(cup_grasp) |
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

### 5-1. 접근·입구 (같은 관절 제어로 컵 집기에 성공한 cup_grasp 구조를 따른다)
- 원점 기준 거리 보상: 먼 척도 + 가까운 척도의 합(예: 1−tanh(d/먼 척도), 1−tanh(d/가까운 척도)), 높이 차는 수평의 2 배로 센다(옆·잡는 높이 선호,
  위에서 내려오는 경로는 덜 받는다). 무게는 들기 전 합 예산 안에서. 파지가 서면 사라진다.
- 편 손: 컵에서 떨어져 있는 동안 거리 보상에 손 펴짐 인자를 곱한다(예: 0.45 + 0.55 × 펴짐). 손가락 사이에 컵이 오면 인자를 풀어 닫기를 막지 않는다.
- 손가락 방향·손가락 사이(엄지 대향)·잡는 높이로 내려오기는 원거리부터 연속으로 오르는 상태형(곱 사슬 금지).
- 컵 근처에서 붙는 비용(입구 위 정지·손바닥 밀어 넣기·급한 팔)이 **다가가는 수입보다 작아야** 한다 — 15 cm → 6 cm 가 스텝당 오른다.
- 원장 통과 조건: 0 < 1 < 1b(옆 12 cm 편 손 정지) < 2 < 3 < 4 < 5 ... (단조), 3p ≥ 3, 0p ≥ 0, 1h(반쯤 쥔 손) < 1b.

### 5-2. 제어 벌점은 cup_grasp 수준 (매끄러움은 EMA·속도 상한이 맡는다)
- cup_grasp: 액션 변화율 0.02·tanh(평균 변화²/0.1) + 팔 액션 크기 0.01·평균(a_arm²) + 팔 속도 0.12·tanh(|qd|/1 rad/s) + 3 rad/s 초과 폭주 비용.
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
# iter_01 (4번째 사이클) 운영 노트 — 2026-09-28

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다.

## 사용자 지시 (09-28)
- 영상 판정: "핸드가 수평되게 컵에 접근하지도 않고, 왼손은 손가락들이 말려 있고, 오른팔은 j7 이 꺾여서 핸드가 테이블에 수직으로 박힌다."
- "1번, 2.a, 3번 진행" — 1 = 손가락 힘 멈춤(env), 2.a = 손 범위 편 자세~쥔 자세(env), 3 = 입구 보상 전체에 손 수평 인자(이 라운드의 보상 변경).

## env 변경 (이번 라운드부터, 로봇 설명 절에 반영)
- 손 액션은 관절 한계 전체가 아니라 편 자세~쥔 자세 범위의 절대 목표. 컵 접촉 1 N 을 넘은 손가락은 더 닫지 않는다(펴기 허용).

## t2r_rh4_f01 (iter_00 c1r 보상, PPO cup_grasp 설정) 에서 측정한 것
- e700 부터 오른손 접근·손가락 사이·파지(e2163 오른손 파지 0.18·파지 보상 0.54, 왼손 근처 0.82·손가락 사이 0.71). 들기 보상은 e1850 0.031 → e2163 0.002.
- ep1650 결정론 재생(64 env): 오른손은 컵 근처(수평 10 cm 안) 스텝의 손가락 축 z 중앙 −0.93 — 87 % 가 손가락이 거의 수직으로 아래,
  수평 자세는 0 스텝. j7 −0.86 rad. 이 수직 자세의 입구 보상(mouth_src) 0.63/스텝.
- 이전 코드에서 mouth = 0.4 × finger_dir + 1.6 × descend. 손가락 수평(level) 인자는 finger_dir 의 근거리 부분(0.7 × near × dir × level)에만 있고,
  descend(잡는 높이로 와서 양쪽에서 감싸기)는 손끝 높이·대향만 보고 손 방향을 보지 않는다.
- 같은 재생에서 100 N 넘는 손끝 힘의 99~100 % 가 수직 손이었고, 그때 엄지 바디(thumb_3·thumb_sensor)가 컵 몸통 안(축 거리 < r)에 있었다
  (그 스텝의 49~60 % 는 손바닥이 입구보다 위 — 수직 손이 엄지를 컵 입구로 밀어 넣는다).
- 결정론 검사 [34](새): 위 수직 손 재생 자세 512 개에서 이전 코드 mouth_src 0.642, 수평 '컵이 손가락 사이' 자세 1.391 (비 0.46).

## iter_01 에서 할 것 (이전 코드 위의 변경 하나)
1. 입구 보상 전체(손가락 방향 + 내려와 감싸기)에 손가락 수평 인자를 건다: 손가락 길이 축(palm_axes[:, 3:6])이 컵 축(cup_up)에 수직일수록 1,
   컵 축과 나란하면(손가락이 아래나 위) 작게. [34] 기준: 수직 손 재생 자세의 입구 보상이 수평 '손가락 사이' 자세의 35 % 이하.
   원장(0 < 1 < 1b < 2 < 3 < 4 < 5 …, 3p·0p·1h)과 이전 검사([15][21][24][25][28][31][33])는 그대로 통과해야 한다.
2. 그 밖(접근·편 손·닫기·파지·들기·붓기·제어 벌점)은 이전 코드 그대로.
