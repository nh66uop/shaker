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


def _palm_surface_dist(palm_pos, cup_pos, cup_up, cup_r, cup_mouth_z, cup_bottom_z):
    """손바닥 -> 자기 컵 원통 표면 거리 (컵 축 기준: 반경 간극 + 바닥~림 높이 밖 간극). 치수는 ctx 에서."""
    rel = palm_pos - cup_pos
    h = (rel * cup_up).sum(dim=-1)
    rad = (rel - h.unsqueeze(-1) * cup_up).norm(dim=-1)
    gap_r = torch.clamp(rad - cup_r, min=0.0)
    gap_h = torch.clamp(h - cup_mouth_z, min=0.0) + torch.clamp(cup_bottom_z - h, min=0.0)
    return torch.sqrt(gap_r * gap_r + gap_h * gap_h + 1e-12)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    rh2_i03 블록 재사용. 높이 오차(h_err_t, h_err_o)를 추가로 돌려준다 (접촉력 대체 경로의 높이 게이트용)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (높이 허용 = 절대값 예외) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
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

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_funnel": d_funnel, "d_grip": d_grip,
            "thumb_band": thumb_band, "h_err_t": h_err_t, "h_err_o": h_err_o}


def _hand_cup_terms(palm_pos, tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd,
                    cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                    cup_r, cup_mouth_z, cup_bottom_z, pose_ok, progress,
                    lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (iter_03: cup_grasp 골격으로 다시 씀). 두 팔 사이 곱 없음.
    들기 전 수입(reach·descend·close)은 held x 컵 높이로 꺼진다 (테이블 위에서는 끄지 않는다)."""
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

    # ---- 높이 --------------------------------------------------------------------------------------
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)               # 5 mm dead band: tipping on the rim is not lifting
    t_up = torch.tanh(h_pos / 0.02)
    fade = 1.0 - held * t_up                                   # 들기 전 수입 끄기: 컵 높이 기준 (테이블 위 = 1)
    on_table = 1.0 - torch.tanh(h_pos / 0.04)
    late = torch.clamp((progress - 0.2) / 0.8, 0.0, 1.0)
    table_decay = 1.0 - 0.55 * late * on_table                 # cup_grasp 5 번: 후반 테이블 위 파지 수입 감쇠

    # ---- A. 접근: 손바닥-컵 표면 거리, 두 척도(20 cm / 5 cm), 곱 인자 없음 --------------------------
    d_palm = _palm_surface_dist(palm_pos, cup_pos, cup_up, cup_r, cup_mouth_z, cup_bottom_z)
    reach_raw = 0.6 * (1.0 - torch.tanh(d_palm / 0.20)) + 0.4 * (1.0 - torch.tanh(d_palm / 0.05))
    reach = fade * reach_raw

    # ---- A2. 잡는 높이로 내려와 양쪽에서 감싸기 (rh2_i03 기하). 접촉력 대체 경로는 높이 대역 안에서만
    funnel = 0.5 * (1.0 - torch.tanh(4.0 * geo["d_funnel"])) + 0.5 * torch.exp(-(geo["d_funnel"] / 0.03) ** 2)
    band_t = torch.exp(-(geo["h_err_t"] / 0.02) ** 2)          # (N,)
    band_o = torch.exp(-(geo["h_err_o"] / 0.02) ** 2)          # (N,F-1)
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t * band_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh((f[:, 1:] * band_o).sum(dim=1) / 0.5))
    tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    descend = fade * (0.4 * funnel + 0.6 * tips)

    # ---- 닫기: 잡는 자리에서만, 0.7 에서 포화 --------------------------------------------------------
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = fade * table_decay * pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # ---- B. 파지: 성립 전 접촉부터 단계적 (한 손가락 < 양쪽 < 여러 손가락·단단히) ---------------------
    grasp_q = 0.2 * c_o + 0.3 * pinch_c + 0.5 * held * (0.4 + 0.6 * firm)
    grasp = table_decay * pose_ok * grasp_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C. 들기: 품질로 곱하지 않는다 (held x 직립만). 빠른 성분 0.7 >= 들기 전 수입 합 -> 골짜기 없음
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)   # 과들기 세금: over_h 위에서만
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)              # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)           # "held and at least 5 cm up"

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
    return {"reach": reach, "reach_raw": reach_raw, "descend": descend, "close": close, "grasp": grasp,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기 iter_03. 접근·파지·들기 = cup_grasp 골격(팔별 합, 곱 사슬 없음, 들기 전 수입은 컵 높이로 끔),
    붓기 = iter_02 c5 부분 그대로(이름 유지)."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 ------------------------------------------------------------------------------------
    # 들기 전 팔별 최대 = W_REACH 1.0 + W_DESCEND 0.7 + W_CLOSE 0.3 = 2.0 < 들기 빠른 성분 0.7 x W_LIFT = 2.1
    # -> 컵을 드는 순간 끊기는 수입보다 들기 수입이 늘 크다 (골짜기 없음)
    W_REACH, W_DESCEND, W_CLOSE, W_GRASP, W_LIFT = 1.0, 0.7, 0.3, 2.0, 3.0
    W_ALIVE = 0.5          # 상수: 경사를 만들지 않는다, 서 있기 < 접근(20 cm -> 2 cm 에 +0.58/스텝)
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

    # ---- A/B/C. 팔별 접근·파지·들기 ---------------------------------------------------------------
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force,
                          ctx.src_hand_closure, ctx.src_grasped, ctx.src_arm_qd,
                          ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_src, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force,
                          ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                          ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_rcv, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 (reach -> 파지 -> 들기), 바닥 있는 계수. reach 는 끄기 전 값(들고 나서 진행도가 줄지 않게)
    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones
    reach_src = W_REACH * src["reach"]           # 소스·리시버 같은 크기로 독립 지급
    reach_rcv = W_REACH * rcv["reach"]
    descend_src = W_DESCEND * src["descend"]
    descend_rcv = W_DESCEND * rcv["descend"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
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

    # ---- 흘림: 파지 전 알당 3, 두 컵을 쥔 뒤 또는 소스를 든 뒤 알당 40 (c5) --------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill_w = 3.0 + 37.0 * post
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

    # ---- 제어 정규화 (c5, 게이트 없음) --------------------------------------------------------------
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
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
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
# iter_04 운영 노트 — 2026-09-28 (세 번째 사이클 4 라운드)

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다. iter_03 노트(cup_grasp 골격)는 이 노트 뒤에 그대로 이어진다.

## 사용자 지시 (09-26 ~ 09-28)
- "접근·파지·리프트가 fresh 에서도 **빨리** 되어야 후속 작업(붓기)을 할 수 있다."
- 관절·명령을 한계까지 밀어붙이는 자세로 학습하는 것을 원하지 않는다(09-26 "J7 과 같은 관절을 한계값까지 가도록 학습함").
- 유지: LSTM(→LN→MLP, concat 끔), horizon 16, self-collision ON, 붓기 구조(c5), iter_03 의 cup_grasp 골격.

## iter_03 보상 결과 — t2r_rh4_i03 (fresh LSTM, e6164 에서 사용자 지시로 종료)
학습 설정: mini_epochs 2, lr 1e-4 adaptive(kl 0.016), e_clip 0.1, critic 1e-4, entropy_coef 0.0005. epoch 당 65,536 프레임.
NaN·OOM·학습 붕괴 없음. entropy 34 → 22.7(e1000, 6,550만 프레임; i02m 은 같은 프레임에 12.7) → 8.7(e5600).

| 지표 (20 epoch 평균) | e1000 | e3500 | e4500 | e5600 | e5950~6050 | e6164 |
|---|---|---|---|---|---|---|
| 왼손(리시버) 컵 근처 비율 | 0.95 | 0.90 | 0.90 | 0.93 | | |
| 오른손(소스) 컵 근처 비율 | 0.00 | 0.00 | 0.92 | 0.92 | | |
| 왼손 엄지 대향(근처 env) | 0.00 | 0.01 | 0.17 | 0.60 | | |
| 왼손 컵이 엄지-손가락 사이 | 0.00 | 0.27 | 0.32 | 0.92 | | |
| 오른손 엄지 대향 / 컵 사이 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | | |
| 왼손 파지 플래그 | 0 | 0.001 | 0.004 | 0.001 | 0.12 → 0.23 | 0.0007 |
| 왼손 컵 들림(rcv_cup_lift) | | | | | 0.13 → 0.28 | 0.03 |
| 에피소드 길이(최대 900) | 835 | 893 | 890 | 887 | 812 → 678 | 873 |

- **오른손은 컵 근처까지 오는 데 e4000(2.6억 프레임)이 걸렸다.** 왼손은 e600~1000.
  결정론 재생 시작 자세는 두 손이 대칭이다(손바닥이 자기 컵에서 21.8 cm, 로봇 쪽 −7 cm·바깥쪽 14 cm·위 15 cm).
- **왼손 파지는 e5800~6050 에 처음 나왔다가(플래그 0.23, 들림 0.28) e6164 에 다시 사라졌다.**
  같은 시기에 조기 종료가 늘었다(스텝당 비율, e5500 → e6050): 팔 관절 속도 폭주(runaway) 5e-6 → 2.2e-4,
  mimic 폭주 5e-6 → 1.1e-4, 컵 떨어뜨림(drop) 5e-6 → 1.1e-4, mimic 속도 튐 5e-5 → 6.9e-4.
  에피소드 900 스텝 기준 폭주 + mimic + drop 이면 에피소드의 약 40 % 가 일찍 끝난 셈이다. 그 뒤 파지가 사라지고 에피소드 길이가 873 으로 돌아왔다.
  [가설] 파지 시도가 조기 종료로 이어져 남은 에피소드 수입을 잃으므로 정책이 파지에서 물러났다.

## ep5900 결정론 재생 (64 env, 900 스텝, 300 스텝 이후 중앙값)
- **왼손(리시버)**: 로봇 쪽·바깥에서 다가옴(손바닥이 컵에 대해 x −0.060, y +0.034). 손끝이 컵 축에서 2.7~3.7 cm(컵 반지름 2.8 cm)로 벽에 붙음.
  검지·중지 끝 접촉(0.17 N·0.10 N), 손 닫힘 0.27, 파지 유지 0.01. 파지 중 접촉 손가락 수는 2 개 87 %, 3 개 13 %.
- **오른손(소스)**: 컵 **먼 쪽(로봇 반대, +x)·바깥쪽** 에 멈춤(손바닥이 컵에 대해 x +0.025, y −0.036, 입구 높이).
  손바닥은 컵 축에서 4.5 cm, 손끝은 7.5~13 cm 로 아래·바깥으로 펼침(엄지-검지 간격 90 mm). 접촉 0, 엄지 대향 0.
  이 자세에서 접근 수입 0.75, 내려오기 0.10 이 매 스텝 들어온다(`reach_src` 0.76, `descend_src` 0.10, 로그 e4000~6000).
- **오른손 손바닥 명령이 한계에 붙어 있다**: palm 명령 채널 |a| > 0.95 비율 x 0.67 · z 0.84 · yaw 0.96 · roll 0.95.
  명령과 실제 손바닥 자세의 차 yaw 1.1 rad · roll 1.8 rad · 위치 4~6 cm. 팔 관절은 한계에 붙지 않았다(모든 관절 여유 > 0.06).
  왼손은 z 0.97 만 한계(명령 z 가 손바닥 상자 바닥 0.26 m), 나머지 채널 0.03~0.47.
  현재 포화 벌점 `action_reg_adr` 는 스텝당 −0.044(24 채널 평균이라 한 손 palm 4 채널 포화는 약 −0.017).
- 손바닥 법선(`*_palm_axes` 앞 3 칸)의 방향을 재생으로 확인했다: 엄지 끝이 법선 쪽 +7.7 cm, 손가락을 닫으면 손끝이 법선 쪽으로 온다(+0.02 m).
  즉 **법선은 손바닥 안쪽(쥐는 쪽)을 가리킨다.** iter_03 보상은 `*_palm_axes` 를 쓰지 않는다.
  오른손은 이미 손바닥 법선이 컵 쪽(법선·(컵−손바닥) 0.88)이라, 손바닥 방향 항만으로는 이 주차를 설명하지 못한다.

## 사용자 검토 (09-28, 1차 후보 c1 뒤)
- "오른손은 j7 쪽이 컵에 붙어 있는 느낌" → 재생으로 확인(아래 사실).
- "표면보다는 물체 origin 이 좋을 것 같다" → 접근 거리를 컵 origin 기준으로.
- 1차 후보의 `far_cross`·`palm_sat` 두 벌점은 "검토 필요" — 아래 가설 A/B 로 나눠 비교한다.

## 손 기하 사실 (URDF·ep5900 재생)
- `*_palm_pos` 는 `r_hl_palm_sensor`(왼손 `l_hl_palm_sensor`) 링크다. 손 부착점(j7 끝 플랜지) 기준 손 길이 방향 10.4 cm,
  손바닥 면 쪽 1.6 cm. 엄지 뿌리 7.1 cm, 네 손가락 뿌리 13.6 cm, 손끝 약 21 cm. j7 축은 이 점에서 약 16 cm 뒤.
- `*_palm_axes` 의 앞 3 칸 = 손바닥 법선(손바닥 안쪽, 쥐는 쪽). **뒤 3 칸 = 손가락 길이 방향**(손바닥에서 손끝 쪽).
  재생에서 손끝이 이 축을 따라 손바닥에서 +0.10 m 에 있다(네 손가락 0.093~0.107).
- 오른손(ep5900, 실패): 손가락 길이 방향이 컵 축에서 **바깥쪽 0.76 · 아래 0.64**. 손바닥 뒤꿈치~손 부착점(손바닥에서 손목 쪽 3~9 cm)이
  컵 입구 원판 바로 위(컵 벽보다 안쪽, 입구보다 1.7~5.5 cm 위)에 걸쳐 있다. 손끝은 벽에서 4.6~10 cm.
- 왼손(ep5900, 파지 근처): 손가락 길이 방향이 컵 축 쪽 **안쪽 0.88**, 거의 수평(아래 0.17). 손끝은 벽에서 0~0.9 cm, 손바닥은 벽에서 4.2 cm.
- iter_03 접근 거리(`_palm_surface_dist`)는 입구 원판을 표면으로 친다: 손바닥이 입구 위(반지름 안)에 있으면 거리 = 입구 위 높이.
  그래서 오른손 자세의 손바닥 표면 거리는 1.7 cm, 접근 수입 0.75(최대 1.0)였다.
- 손바닥-컵 origin 거리는 두 자세가 거의 같다: 오른손(실패) 6.9 cm, 왼손(좋음) 7.1 cm. **origin 거리만으로는 두 자세를 가르지 못한다.**
  두 자세를 가르는 것은 손가락 방향이다.

## iter_04 에서 할 것 (iter_03 코드 위의 변경)
공통(가설 A·B 모두):
1. **접근 거리를 손바닥 → 자기 컵 origin 거리로 바꾼다**(`*_cup_pos`). 입구 위에 손을 얹어 접근 수입을 받는 길을 없앤다.
   두 척도(원거리·근거리) 형태는 유지하되, 옆에서 제대로 쥔 자세의 손바닥-origin 거리(약 7 cm, 컵 반지름 + 손 두께)가
   근거리 척도에서 충분히 높은 값을 받도록 척도를 컵 반지름 배수로 잡는다. 성분 이름 `reach_src`·`reach_rcv` 는 유지.
2. **손가락 방향 항을 새로 둔다**(`finger_dir_src`, `finger_dir_rcv`). 손가락 길이 방향(`*_palm_axes[:, 3:6]`)이
   손바닥에서 자기 컵 축을 향하는 수평 방향과 이루는 정도(내적, 0 아래는 0)를 준다. 컵에 가까울 때만(손바닥-origin 거리로
   켜지는 연속 가중), 파지 전 수입과 같이 컵 높이로 끈다(`fade`). 크기는 접근 최대(1.0)의 절반 안팎.
   측정 기준: 실패 자세(바깥 0.76)는 0, 좋은 자세(안쪽 0.88)는 거의 전액.

가설 A — 두 벌점을 뺀다: iter_03 에 1·2 만 더한다(`far_cross`·`palm_sat` 없음).
  근거: 먼 쪽 주차 자세는 손가락이 바깥을 향한 자세이고 2 가 그 수입을 직접 없앤다. `far_cross` 는 로봇 전방(+x) 방향을 가정한다.
  `palm_sat` 는 ep5900 에서 좋은 쪽 왼손에도 스텝당 −0.16 을 물린다(명령 z 가 손바닥 상자 바닥에 붙음).
가설 B — 두 벌점을 둔다: 1·2 에 1차 후보의 `far_cross`(파지 전 손바닥이 자기 컵보다 +x 로 0.5r 넘으면 1.5r 에서 포화, 가중 0.5, 두 손)와
  `palm_sat`(팔별 palm 6 채널 |a| 0.9→1 램프 평균, 가중 0.5, 손 채널 제외)를 그대로 더한다.
  근거: 오른손 palm 명령 4 채널이 67~96 % 시간 한계에 붙어 있었다. 사용자 원칙(명령·관절을 한계까지 밀지 않는다).

그 밖에는 iter_03 코드를 바꾸지 않는다. 이름을 바꾸지 않는다. 새 ctx 필드를 가정하지 않는다. 컵 치수는 ctx 에서.

# iter_03 운영 노트 — 2026-09-27 (세 번째 사이클 3 라운드)

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다.

## 사용자 지시 (09-27)
- "접근·파지·리프트가 fresh 에서도 **빨리** 되어야 후속 작업(붓기)을 할 수 있다."
- "왼손이 이상하게 오른손 컵으로 가는 느낌" — 측정으로 확인(아래).
- "다음 실험은 에피소드를 생각해서 엔트로피를 적절히 유지" — 학습 설정 쪽(보상 밖)에서 따로 다룬다.
- 유지: LSTM(→LN→MLP, concat 끔), horizon 16, self-collision ON, 붓기 구조(c5 = DG-5F iter_02 이식).

## iter_02 보상으로 돌린 두 런 (fresh LSTM, 같은 보상)
| | t2r_rh4_i02 (원래 LSTM yaml) | t2r_rh4_i02m (LSTM→LN→MLP, concat 끔, lr 3e-4 linear) |
|---|---|---|
| 종료 epoch | 1739 | 1588 |
| 오른손 컵 근처 비율 최대 | 0.000 | **0.858**(e1011) → 0.015(e1371) |
| 왼손 컵 근처 비율 최대 | 0.68(e1250) → 0.31 | 0.000 |
| 파지 플래그 | 0 | 1.2e-5 |
| entropy (e1011 / e1371) | 27.4 / — | 12.5 / 8.3 |

i02m ep1200 결정론 재생(64 env, 900 스텝):
- 오른손: 2 s 에 손끝이 컵 벽(축에서 2.4~3.4 cm, 컵 반지름 2.8 cm)에 붙어 10 s 유지. **손끝 높이 4~6 cm**(잡는 높이 약 1.3 cm 보다 높다), 손 닫힘 0.45 에서 멈춤, 엄지 5.6 %·중지 2.1 % 접촉. 한 번이라도 파지한 env 3.1 %.
- 왼손: 1 s 에 자기 컵 5.7 cm 까지 왔다가 **자기 컵(y +0.165)을 지나 중심선 쪽 y +0.074** 로 가서 멈춤.
  자기 컵까지 12~13 cm 로 그대로, 소스 컵까지 46 → 25 cm. 손은 닫힌(0.47) 채 공중.
- 두 손 모두 "다가가서 반쯤 닫고 멈춤". 내려가서 쥐고 드는 단계가 없다.

## 같은 저장소 cup_grasp 트랙 — fresh LSTM 으로 접근·파지·들기를 끝냈다 (09-27 조사)
- 오른팔 iter_00: grasp_ep ≥ 0.5 e2392(3.2 h), lift ≥ 0.5 e2789, 최종 grasp 0.90 · lift 0.88. 왼팔 iter_01: grasp 0.5 e1157(1.6 h), lift 0.5 e1421.
- 학습 설정: LSTM 1024 → LN → MLP [1024,1024,512,512], horizon 16, seq 16, minibatch 16384, **mini_epochs 2, lr 1e-4 adaptive(kl 0.016), e_clip 0.1, entropy_coef 0**, critic lr 1e-4, gamma 0.99.
  entropy 36.7 → 파지 직전 18.6(e2250) → 12.3(e5000). 우리 i02m 은 e1011 에 이미 12.5.
- 손은 DG-5F(19 관절 절대 명령)라 손 모양 항(링크별 접촉·curl)은 RH 에 못 옮긴다. **옮길 수 있는 골격**:
  1. **곱 게이트 없는 원거리 접근**: 손바닥-컵 표면 거리 d, `0.35 · fade · (0.6(1−tanh(d/0.20)) + 0.4(1−tanh(d/0.05)))`.
     무작위 초기에서 접근 수입이 충분히 커야 한다(할인 수익: 서 있기 +1.14 대 접근 +8.11).
     접근에 다른 인자를 곱하면(옆벽·편손 바닥 0.15) 손바닥 간극이 0.22 → 0.37 m 로 물러났다.
  2. **편 손은 작은 가산항**(0.04), 손바닥 8 cm 밖에서 전액·2 cm 에서 0·쥐면 0. 편 손 주차 수입 < 한 손가락 접촉 수입.
  3. **연속 held**: `clip((n_touch−2)/2) · (0.3+0.7·palm) · (0.5+0.5·opp)` — 이진 래치가 아니다.
  4. **들기는 품질로 곱하지 않는다**(2.1 · held · upright · (0.5 tanh(h/0.03) + 0.5 h/h_need) + 상승 속도 0.4). 들기가 파지보다 먼저 나왔다.
  5. **테이블 위에서는 들기 이후 수입 0**(off_table), 에피소드 후반 테이블 위 파지 수입 감쇠 · stall 비용 — 테이블 위에서 쥐고 버티기 방지.
  6. **래치 없는 벌점**: drop 은 떨어지는 동안·안 쥔 상태에서만. 높이 래치 비례 벌점은 들기를 지웠다.
  7. 기동 전 **결정론 원장**: L1 서 있기 < 접근, L2 편 손 주차 < 한 손가락, L3 테이블 < 3 cm < 8 cm, L4 누름 < 감기.
  8. (왼팔 iter_03) 들기 전 수입을 held 가 아니라 **컵 높이로 끈다** — held 로 끄면 테이블 위에서 파지를 완성하는 것이 손해가 됐다.
- 주의: 이 컵(pour_mimic)은 한쪽 접촉 0.3~0.4 N 에 넘어진다. cup_grasp 의 shove 면제·손바닥 안착 보너스는 그대로 옮기지 않는다.

## iter_03 에서 할 것
1. **팔별 접근·파지·들기를 cup_grasp 골격으로 다시 쓴다**(팔별 합, 두 팔 사이 곱 사슬 금지).
   - 접근: 손바닥-자기 컵 **표면** 거리(컵 반지름 빼기), 두 척도(20 cm · 5 cm), 곱 인자 없음, 들면 꺼짐.
     **무작위 초기(손바닥 15~16 cm)에서도 경사가 커야 한다** — iter_02 는 20 cm→6 cm 수입 차가 +0.05/스텝이었다.
   - 파지 자세로 내려오기: RH 손끝(`*_tips_pos`)과 컵 축 기준 높이·반경으로 "잡는 높이까지 내려와 양쪽에서 감싸기"를
     단계적으로 준다(iter_02 rh2_i03 블록의 grip_geometry 를 재사용해도 된다). 측정상 오른손은 벽에 붙고도 손끝이 입구 높이에 머물렀다.
   - 닫기·접촉: RH 는 손당 6 채널(0 폄·1 쥠)이고 손가락 모양은 고를 수 없다. `*_hand_closure`, `*_finger_force`(5), `*_palm_force` 만 쓴다.
     성립 전 접촉을 단계적으로, 엄지+다른 손가락 양쪽 접촉과 손바닥 접촉이 오를수록 더(cup_grasp 연속 held 형태).
   - 들기: 품질로 곱하지 않는다. 테이블 위 수입(들기 이후)은 0. 들기 전 수입은 컵 높이로 끈다.
2. **왼손(리시버)이 자기 컵으로 가게 한다.** 측정: 왼손은 자기 컵을 지나 중심선 쪽 9 cm 에 머물렀다.
   리시버 접근을 소스와 같은 크기로 독립 지급하고, 파지 전 리시버 손바닥이 자기 컵보다 중심선 쪽(y 가 컵 y 보다 작은 쪽)으로
   넘어가면 작은 비용을 둔다(손바닥 y 기준, 컵 반지름 배수). 소스의 들기 이후 수입은 리시버 진행도 게이트(바닥 있는)를 유지한다.
3. 파지 전 벌점은 작게: 흘림 약 3/알, 컵 건드림(미끄러짐·기울임)은 넘어뜨림 방지용으로만. 래치 금지.
4. **붓기 구조는 iter_02 의 c5 부분을 그대로**(두 컵을 쥐고 든 뒤: converge·aim·pour_rot·pour_delta 30·spill 40·success 20 ·
   finish·premature·pour_dir·rcv_keep·posture_margin·제어 정규화). 이름을 바꾸지 않는다(로그 비교).
5. 항 수는 30 개 안팎 허용. 새 ctx 필드를 가정하지 않는다. 컵 치수는 ctx 에서 읽는다.

## 참고 1 — cup_grasp 설계 브리프 원문(20 라운드 교훈)
```text
# Verified lessons from 20 earlier rounds of the same task (cup_pick, both arms, 09.20-09.26)

Every line is a measurement from a finished run.

1. From random weights the approach was learned when the long-range reach was shaped on plain palm-to-surface distance (weight 0.35, 0.6 far + 0.4 near, times (0.45 + 0.55 grip), fading once held): palm gap 0.045 / 0.037 m and grasp episodes 0.83 / 0.98 (left / right).
2. Any factor that shrinks the long-range reach income at random initialisation made the policy retreat or park. A side-wall factor on the far reach: palm gap rose 0.22 -> 0.37 m. An open-hand factor with floor 0.15 multiplied onto the reach with weight 0.22: reach income 0.003-0.007 per step against 0.037 for the plain reach in the same epoch window, while the action and arm-speed penalties stayed at -0.023 and -0.013; contacts fell 36 -> 1.5 and the palm gap froze at 0.20-0.23 m.
3. Without any open-hand signal the hand arrived already curled: grasp episodes 0.008, palm contact 0 after 2,075 epochs.
4. An open-hand or palm-proximity income that paid without digit contact produced a flat open hand pressed on the wall: +0.195 per step, hand curl 0.000, no digit contact for 1,400 epochs.
5. While digit contact was zero, any term sitting near its maximum became the resting point (seat + palm terms 1.17 per step against 0.07 for the fingers, contact 0.004).
6. When the object could earn income while still on the table, the policy held it there: 0.54-0.69 per step of goal-shaped income with the object on the table, lift episodes 0.007 after 5,000 epochs.
7. A drop penalty proportional to the episode-long height latch became the largest term (-0.75 per step) and the policy unlearned lifting; charging it only while the object falls removed the collapse.
8. When the downstream gate saturated at four touching digits, a four-digit grasp became optimal: the fifth digit was worth at most 0.14 per step, under 1 % of the downstream income.
9. Multiplying the downstream income by an envelope quality raised the right arm's envelope to 0.86; on the left arm the same reward came with contact force x2.6, arm runaway terminations 5 % -> 42 % and grasp 0.86 -> 0.38.
10. Fingertip pressing, measured by replay on 256 episodes: finger-root joints at 0.00-0.17 of their range, last joints at 0.85-1.00, 66 % of held digit-steps are root-straight / tip-folded, 93-98 % of the fingertip force is normal to the wall. Before lift the last joints sit at 0.11-0.63 and this shape is 4-31 %; after lift it is 47-100 % per digit.
11. A curl measure based on the share of force on the "inner" links paid for pressing, because the second sensed point of each digit is the distal link itself and the fingertip pad is fixed to it: a pad lying on the wall scores full curl; a genuinely rolled finger scored 0.16. The finger's first link (moved by its root joint) has no contact sensor and no ctx point.
12. A link-count wrap depth also scores fingertip pressing as full depth when the distal link and the pad touch together.
13. The spread of each digit's three surface distances (max - min of ctx.link_surface_dist over its three points) separates the shapes in replay: rolled 1.1 cm (thumb 0.6), middle-joint hook 1.9 cm, fingertip press 2.0-2.5 cm.
14. The same reward on the mirrored arm gave different grasps: left full-hand grasp with palm 0.38, right three-digit pinch with palm 0.001.

```

## 참고 2 — cup_grasp_r iter_00 보상 원문(DG-5F 손 — 손 모양 항은 RH 에 없는 필드를 쓴다. 골격만 가져올 것)
```text
import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """0 off the object, ~0.6 at 0.1 N, ->1 above a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate and extra force buys nothing."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe normalisation: a zero vector stays zero (its dot products are then 0)."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _sidewall(normal_z: torch.Tensor) -> torch.Tensor:
    """How HORIZONTAL a surface normal is: 1 on a vertical side wall, 0 on a rim face, a lid or the
    table, ~0.3 on a 45 deg rim edge. Used only from a few centimetres in, always with a floor;
    NEVER on the long-range approach (lesson 2)."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Per-step ceilings:
    #   reach            ~0.60  pre-contact only, x (1 - held)
    #                           = 0.35 PLAIN long-range reach (lesson 1 form, NO open-hand factor)
    #                           + 0.24 palm seating (NO open-hand factor) + 0.14 closing
    #                           + 0.04 ADDITIVE open-hand term (far from the wall only)
    #   keep the grasp   1.20   links-per-digit + count ladder + palm contact + opposition
    #   lift             2.50   x uprightness x real hold (`held`); NOT x env_mult (hypothesis B)
    #   transport        2.50   x off_table x env_mult
    #   hold at rest    10.00   real hold, upright, at the goal, still, x env_mult
    #   first success   60.00   one-off; later successes 2.0, x env_mult
    # =============================================================================================
    W_REACH = 0.35        # long range, PLAIN palm-to-surface distance (lesson 1), fades once held
    W_OPEN = 0.04         # ADDITIVE open-pose income on the way in: below the first digit-contact
    #                       increment (~0.05 at the wall) and far below the 0.195 of lesson 4
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike
    W_PALM = 0.30         # palm contact: a pinch must not score like a whole-hand wrap
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height
    W_RISE = 0.40         # upward speed, only until `lifted` latches (income gate, not penalty)
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 2.00         # per step, ONLY while the object is falling without a hold
    P_SETDOWN = 1.00      # per step, ONLY while the object is coming down onto the table
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    P_SQUEEZE = 0.50      # force beyond what a hold needs
    P_TABLE = 0.60        # palm or finger links driven onto the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: small, so lifting and carrying beat standing still
    P_CUP_SPIN = 0.06
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    PALM_STEEP = 0.02     # seating scale
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link
    REACH_FAR = 0.20      # far scale of the plain reach: covers the 0.15-0.49 m start distances
    REACH_NEAR = 0.05     # near scale of the plain reach (last few centimetres)
    OPEN_MEAN_SCALE = 0.12   # mean deviation from the default open pose
    OPEN_MAX_SCALE = 0.30    # and the WORST joint, which is what actually blocks a wrap
    OPEN_NEAR = 0.08      # open-pose term pays in full at this palm-surface distance or beyond
    OPEN_CONTACT = 0.02   # and is zero by here, where contact starts
    PALMIN_ENGAGE_FLOOR = 0.35   # seating credit an entirely disengaged hand still keeps
    TABLE_RISE = 0.04     # smooth "off the table" scale
    TILT_DEAD = math.radians(5.0)
    TILT_SCALE = math.radians(10.0)
    SQ_TOTAL = 40.0
    SQ_LINK = 15.0
    FALL_DEAD = 0.03
    FALL_SCALE = 0.20
    SET_DEAD = 0.02
    SET_SCALE = 0.05
    SET_LIFTOFF = 0.005
    SET_BAND = 0.04
    GOAL_TOL_MULT = 3.0
    GOAL_TOL_MIN = 0.03
    GOAL_FINE = 0.04
    SETTLE_FLOOR = 0.40
    DWELL_BASE = 0.70
    DWELL_SCALE = 1.5
    ENV_FLOOR = 0.10
    ENV_W_COUNT = 0.30
    ENV_W_DEPTH = 0.35
    ENV_W_PALM = 0.25
    ENV_W_OPP = 0.10
    ENV_DEPTH_LINKS = 2.0
    ENV_NEAR_SHARE = 0.25
    ENV_NEAR_SCALE = 0.01
    CURL_FLOOR = 0.50        # env_q keeps half at zero curl
    FOLLOW_DEAD = 0.010      # m: resolution of the surface distance, NOT a pose target
    FOLLOW_SCALE = 0.006     # m: 1.1 cm -> 0.85, 1.9 cm -> 0.22, 2.0-2.5 cm -> 0.19-0.08
    FLOOR_BAND_TOP = 0.020
    FLOOR_BAND_FULL = 0.010

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    n_succ = ctx.num_successes.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    n_movable = movable.sum().clamp(min=1.0)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0
    digit_joint[1, 4:7] = 1.0
    digit_joint[2, 8:11] = 1.0
    digit_joint[3, 12:15] = 1.0
    digit_joint[4, 17:19] = 1.0
    n_digit_joint = digit_joint.sum(-1)

    # ---- height and vertical velocity ------------------------------------------------------------
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    off_table = torch.tanh(height / TABLE_RISE)
    on_table = 1.0 - off_table
    vz = ctx.cup_lin_vel[:, 2]
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)

    # ---- contact and grasp quality ---------------------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5)
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)
    finger_any = digit_touch[:, 1:].amax(dim=-1)
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    link_side = _sidewall(ctx.link_surface_normal[..., 2])
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side
    link_cw = link_c * link_side_w
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- open-pose measure (mean AND worst movable joint) ----------------------------------------
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))

    # ---- measured per-digit flexion --------------------------------------------------------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint
    flex = (closure / 0.5).clamp(max=1.0)

    # ---- palm against the object's real surface --------------------------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    cq = 0.35 + 0.65 * seat

    # open-pose gate: full at 8 cm or more, zero by 2 cm (used ONLY by the additive open term)
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)

    # ---- opposition from the surface normals at the touching links -------------------------------
    normals = ctx.link_surface_normal
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)

    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # ---- envelope quality (multiplier on transport, hold and success only) ------------------------
    digit_gap = (ctx.link_surface_dist.amin(dim=-1) - LINK_DEAD).clamp(min=0.0)
    digit_eng = torch.maximum(digit_touch, ENV_NEAR_SHARE * torch.exp(-digit_gap / ENV_NEAR_SCALE))
    env_rank = torch.tensor([0.02, 0.02, 0.06, 0.15, 0.75], device=device, dtype=dtype)
    eng_desc, _ = torch.sort(digit_eng, dim=-1, descending=True)
    env_count = (eng_desc * env_rank).sum(dim=-1)
    count_depth = (link_c.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0)
    env_depth = count_depth.mean(dim=-1)
    env_base = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
                + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)

    # wall-following curl (lesson 13), a FACTOR of env_q, same formula for every digit
    link_dist = ctx.link_surface_dist
    lift_off = link_dist.amax(dim=-1) - link_dist.amin(dim=-1)
    follow = torch.exp(-(lift_off - FOLLOW_DEAD).clamp(min=0.0) / FOLLOW_SCALE)
    digit_curl = digit_touch * follow
    hand_curl = digit_curl.mean(dim=-1)
    curl_factor = CURL_FLOOR + (1.0 - CURL_FLOOR) * hand_curl
    env_q = env_base * curl_factor
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- fading out once the object is really held
    # =============================================================================================
    fade = loose
    # PLAIN reach of lesson 1: no open-hand factor, no face factor, no side-wall factor (lesson 2)
    reach_far = 1.0 - torch.tanh(palm_d / REACH_FAR)
    reach_near = 1.0 - torch.tanh(palm_d / REACH_NEAR)
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near) * (0.45 + 0.55 * grip)

    # ADDITIVE open-hand income: only while the palm is far from the surface (full at 8 cm, zero by
    # 2 cm) and zero once held, so it can never become the resting point at the wall (lesson 4)
    r_open = W_OPEN * fade * open_gate * open_pose

    # palm seating paired with finger engagement (no open-hand factor any more)
    engage = (0.5 * flex.mean(dim=-1) + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    r_palm_in = W_PALM_IN * fade * palm_near * (0.3 + 0.7 * face) * palm_side_w * seat_engage

    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total
    # =============================================================================================
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_oppose = r_oppose * table_decay

    # =============================================================================================
    # LIFT -- 2.50 total. Hypothesis B: gated by `held` and uprightness only, NOT by env_mult, so the
    # first lift is not throttled by an envelope the policy has not formed yet.
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)
    r_lift = W_LIFT * held * upright * lift_shape
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, x off_table, keeps env_mult
    # =============================================================================================
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = (W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)
              * env_mult)
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10) * env_mult)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, x off_table, keeps env_mult
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded, no latch
    # =============================================================================================
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    set_band = torch.tanh(height / SET_LIFTOFF) * torch.exp(-height / SET_BAND)
    p_setdown = P_SETDOWN * set_band * torch.tanh(((-vz) - SET_DEAD).clamp(min=0.0) / SET_SCALE)
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    p_shove = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                          + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    hand_rel = ctx.hand_z_min - ctx.table_z
    floor_t = ((FLOOR_BAND_TOP - hand_rel) / (FLOOR_BAND_TOP - FLOOR_BAND_FULL)).clamp(min=0.0, max=1.0)
    floor_pen = floor_t * floor_t * (3.0 - 2.0 * floor_t)
    p_table = P_TABLE * 0.5 * (palm_pen + floor_pen)
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * held) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = held * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                        + P_CUP_SPIN * torch.tanh(spin / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_reach + r_open + r_palm_in + r_close
        + r_count + r_digit + r_palm + r_oppose
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_drift - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero

    components = {
        "reach": r_reach,
        "open_hand": r_open,
        "palm_in": r_palm_in,
        "close": r_close,
        "grasp_count": r_count,
        "grasp_digit": r_digit,
        "grasp_palm": r_palm,
        "grasp_oppose": r_oppose,
        "lift_height": r_lift,
        "lift_rise": r_rise,
        "goal_reach": r_goal,
        "goal_progress": r_progress,
        "goal_hold": r_hold,
        "success_bonus": r_success,
        "pen_drop": -p_drop,
        "pen_setdown": -p_setdown,
        "pen_stall": -p_stall,
        "pen_drift": -p_drift,
        "pen_descend": -p_descend,
        "pen_tilt": -p_tilt,
        "pen_shove": -p_shove,
        "pen_squeeze": -p_squeeze,
        "pen_table": -p_table,
        "pen_arm_vel": -p_arm_vel,
        "pen_arm_fast": -p_arm_fast,
        "pen_cup_vel": -p_cup_vel,
        "pen_action": -p_action,
    }
    return reward, components

```


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
    rh2_i03 블록 재사용. 높이 오차(h_err_t, h_err_o)를 추가로 돌려준다 (접촉력 대체 경로의 높이 게이트용)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (높이 허용 = 절대값 예외) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
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

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp, "d_funnel": d_funnel, "d_grip": d_grip,
            "thumb_band": thumb_band, "h_err_t": h_err_t, "h_err_o": h_err_o}


def _hand_cup_terms(palm_pos, palm_axes, tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd,
                    cup_pos, cup_up, cup_lin_vel, cup_ang_vel, cup_spawn_pos,
                    cup_r, cup_mouth_z, cup_bottom_z, pose_ok, progress,
                    lift_split, lift_target, over_h, over_span):
    """팔별 무가중 항 (iter_03 cup_grasp 골격 + iter_04 가설 A: origin 접근 거리 · 손가락 방향). 두 팔 사이 곱 없음.
    들기 전 수입(reach·descend·close·finger_dir)은 held x 컵 높이로 꺼진다 (테이블 위에서는 끄지 않는다).
    cup_bottom_z 는 iter_03 의 표면 거리에서만 쓰였고 iter_04 에서는 쓰지 않는다 (시그니처만 유지)."""
    EPS = 1e-6
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

    # ---- 높이 --------------------------------------------------------------------------------------
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)               # 5 mm dead band: tipping on the rim is not lifting
    t_up = torch.tanh(h_pos / 0.02)
    fade = 1.0 - held * t_up                                   # 들기 전 수입 끄기: 컵 높이 기준 (테이블 위 = 1)
    on_table = 1.0 - torch.tanh(h_pos / 0.04)
    late = torch.clamp((progress - 0.2) / 0.8, 0.0, 1.0)
    table_decay = 1.0 - 0.55 * late * on_table                 # cup_grasp 5 번: 후반 테이블 위 파지 수입 감쇠

    # ---- A. 접근 (iter_04 공통 1): 손바닥 -> 자기 컵 origin 거리. 입구 위에 손을 얹어 받던 수입을 없앤다.
    # 옆에서 쥔 자세의 손바닥-origin 거리(약 7 cm = 컵 반지름 + 손 두께)가 높은 값을 받도록 2r 을 빼고 잰다.
    # 척도는 컵 반지름 배수: 원거리 7r (~20 cm), 근거리 2r (~5.6 cm). 곱 인자 없음 (cup_grasp 교훈 2).
    # 값 예: 7 cm -> 0.86, 10 cm -> 0.61, 21.8 cm(시작 자세) -> 0.19.
    d_org = (palm_pos - cup_pos).norm(dim=-1)
    d_eff = torch.clamp(d_org - 2.0 * cup_r, min=0.0)
    reach_raw = 0.6 * (1.0 - torch.tanh(d_eff / (7.0 * cup_r))) + 0.4 * (1.0 - torch.tanh(d_eff / (2.0 * cup_r)))
    reach = fade * reach_raw

    # ---- A1. 손가락 방향 (iter_04 공통 2): 손가락 길이 방향(palm_axes[:,3:6])과 손바닥 -> 컵 축 수평 방향의 내적.
    # 0 아래는 0 (측정: 실패 자세 바깥 0.76 -> 0, 좋은 자세 안쪽 0.88 -> 0.88). 컵 근처에서만 켠다 (origin 거리의
    # 연속 가중, 7 cm -> 0.94, 14 cm -> ~0.05). 들기 전 수입이므로 fade 로 끈다.
    f_dir = palm_axes[:, 3:6]
    to_cup = cup_pos - palm_pos
    h_c = (to_cup * cup_up).sum(dim=-1, keepdim=True)
    to_axis = to_cup - h_c * cup_up                            # 컵 축에 수직인 성분 (직립이면 수평)
    e_in = to_axis / (to_axis.norm(dim=-1, keepdim=True) + EPS)
    dir_dot = _sat((f_dir * e_in).sum(dim=-1))
    near_w = torch.exp(-(d_eff / (2.0 * cup_r)) ** 2)
    finger_dir = fade * near_w * dir_dot

    # ---- A2. 잡는 높이로 내려와 양쪽에서 감싸기 (rh2_i03 기하). 접촉력 대체 경로는 높이 대역 안에서만
    funnel = 0.5 * (1.0 - torch.tanh(4.0 * geo["d_funnel"])) + 0.5 * torch.exp(-(geo["d_funnel"] / 0.03) ** 2)
    band_t = torch.exp(-(geo["h_err_t"] / 0.02) ** 2)          # (N,)
    band_o = torch.exp(-(geo["h_err_o"] / 0.02) ** 2)          # (N,F-1)
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t * band_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh((f[:, 1:] * band_o).sum(dim=1) / 0.5))
    tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    descend = fade * (0.4 * funnel + 0.6 * tips)

    # ---- 닫기: 잡는 자리에서만, 0.7 에서 포화 --------------------------------------------------------
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = fade * table_decay * pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # ---- B. 파지: 성립 전 접촉부터 단계적 (한 손가락 < 양쪽 < 여러 손가락·단단히) ---------------------
    grasp_q = 0.2 * c_o + 0.3 * pinch_c + 0.5 * held * (0.4 + 0.6 * firm)
    grasp = table_decay * pose_ok * grasp_q * (0.1 + 0.9 * geo["thumb_band"])

    # ---- C. 들기: 품질로 곱하지 않는다 (held x 직립만). 빠른 성분 0.7 >= 들기 전 수입 합 -> 골짜기 없음
    lin_lo = torch.clamp(h_pos, max=lift_split) / lift_target
    lin = torch.clamp(h_pos / lift_target, max=1.0)
    lin_hi = torch.clamp(lin - lin_lo, min=0.0)
    tax = 0.5 * torch.clamp((h_pos - over_h) / over_span, 0.0, 1.0)   # 과들기 세금: over_h 위에서만
    hold_up = held * pose_ok
    lift_lo = hold_up * (0.7 * t_up + 0.3 * lin_lo)
    lift_hi = hold_up * (0.3 * lin_hi - tax)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)              # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)           # "held and at least 5 cm up"

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
    return {"reach": reach, "reach_raw": reach_raw, "finger_dir": finger_dir, "descend": descend,
            "close": close, "grasp": grasp,
            "lift_lo": lift_lo, "lift_hi": lift_hi, "held": held, "up_w": up_w, "lifted_w": lifted_w,
            "slide": slide, "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기 iter_04 가설 A. iter_03 (cup_grasp 골격 + c5 붓기) 위에 두 가지만 바꾼다:
    (1) 접근 거리 = 손바닥 -> 자기 컵 origin, (2) 손가락 방향 항 finger_dir_src/rcv. far_cross·palm_sat 없음."""
    tilt_src = ctx.src_cup_tilt
    dt = tilt_src.dtype
    zeros = torch.zeros_like(tilt_src)
    ones = torch.ones_like(tilt_src)

    # ---- 가중 ------------------------------------------------------------------------------------
    # iter_03: 들기 전 팔별 최대 = W_REACH 1.0 + W_DESCEND 0.7 + W_CLOSE 0.3 = 2.0 < 들기 빠른 성분 0.7 x W_LIFT = 2.1
    # iter_04: W_FINGER_DIR 0.4 (접근 최대 1.0 의 절반 안팎 중 보수적인 쪽) 를 더하면 이론상 최대 2.4.
    #   실제 옆 파지 자세의 reach 는 약 0.86 이고 모든 들기 전 항이 동시에 최대가 되지 않는다 (들기 전 합 약 2.2 안팎).
    W_REACH, W_DESCEND, W_CLOSE, W_GRASP, W_LIFT = 1.0, 0.7, 0.3, 2.0, 3.0
    W_FINGER_DIR = 0.4
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
    src = _hand_cup_terms(ctx.src_palm_pos, ctx.src_palm_axes, ctx.src_tips_pos, ctx.src_finger_force,
                          ctx.src_palm_force, ctx.src_hand_closure, ctx.src_grasped, ctx.src_arm_qd,
                          ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel, ctx.src_cup_ang_vel,
                          ctx.src_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_src, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_SRC, OVER_H_SRC, OVER_S_SRC)
    rcv = _hand_cup_terms(ctx.rcv_palm_pos, ctx.rcv_palm_axes, ctx.rcv_tips_pos, ctx.rcv_finger_force,
                          ctx.rcv_palm_force, ctx.rcv_hand_closure, ctx.rcv_grasped, ctx.rcv_arm_qd,
                          ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel,
                          ctx.rcv_cup_spawn_pos, r_cup, ctx.cup_mouth_z, ctx.cup_bottom_z,
                          pose_ok_rcv, ctx.episode_progress,
                          LIFT_T_RCV, LIFT_T_RCV, OVER_H_RCV, OVER_S_RCV)

    # 리시버 진행도 (reach -> 파지 -> 들기), 바닥 있는 계수. reach 는 끄기 전 값(들고 나서 진행도가 줄지 않게)
    rcv_stage = _sat(0.4 * rcv["reach_raw"] + 0.3 * rcv_g + 0.3 * rcv["up_w"])
    rcv_gate = RCV_GATE_FLOOR + (1.0 - RCV_GATE_FLOOR) * rcv_stage

    alive = W_ALIVE * ones
    reach_src = W_REACH * src["reach"]           # 소스·리시버 같은 크기로 독립 지급
    reach_rcv = W_REACH * rcv["reach"]
    finger_dir_src = W_FINGER_DIR * src["finger_dir"]   # 손가락이 컵 축 쪽을 향할 때만 (바깥 향한 주차 자세 = 0)
    finger_dir_rcv = W_FINGER_DIR * rcv["finger_dir"]
    descend_src = W_DESCEND * src["descend"]
    descend_rcv = W_DESCEND * rcv["descend"]
    close_src = W_CLOSE * src["close"]
    close_rcv = W_CLOSE * rcv["close"]
    grasp_src = W_GRASP * src["grasp"]
    grasp_rcv = W_GRASP * rcv["grasp"]
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

    # ---- 흘림: 파지 전 알당 3, 두 컵을 쥔 뒤 또는 소스를 든 뒤 알당 40 (c5) --------------------------
    post = torch.maximum(both_g, src["lifted_w"])
    spill_w = 3.0 + 37.0 * post
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

    # ---- 제어 정규화 (c5, 게이트 없음) --------------------------------------------------------------
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
        "reach_src": reach_src,
        "reach_rcv": reach_rcv,
        "finger_dir_src": finger_dir_src,
        "finger_dir_rcv": finger_dir_rcv,
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

[11] src 파지 경로: 멀리(20cm) +1.33 → 깔때기(6cm) +1.83 → 손끝이 벽(1cm) +2.71 → 벽에 닿음·닫힘 0.7 +3.25 → 한쪽 접촉 0.5N +3.49 → 양쪽 1.5N·파지 +4.98 → 파지·5cm 들기 +4.89
    ✗ src: 다음 단계에서 수입이 오르지 않는다 — 파지·5cm 들기
[16] src 들기 사다리: 테이블 +9.12 · 3cm +9.01 · 8cm +9.11
    ✗ src: 테이블 위 파지 대비 들기 이득이 작다(cup_grasp L3)
[16] rcv 들기 사다리: 테이블 +9.12 · 3cm +9.18 · 8cm +9.62
    ✗ rcv: 테이블 위 파지 대비 들기 이득이 작다(cup_grasp L3)
