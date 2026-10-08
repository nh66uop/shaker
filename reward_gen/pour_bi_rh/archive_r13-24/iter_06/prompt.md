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

I want it to fulfil the following task: Using both arms, grasp each cup (a fingertip grasp is fine, no wrap-around needed): the right hand grasps the source cup (which contains the beads) and the left hand grasps the empty receiver cup. Lift both cups off the table, bring the mouth of the source cup over the mouth of the receiver cup, and tilt the source cup so that the beads pour into the receiver cup. Keep the receiver cup upright, do not drop either cup, and spill as few beads as possible. The task is finished when at least half of the beads are inside the receiver cup with little spill.
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


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    iter_05: unchanged from iter_04 — the grasp works (4/4 envs, thumb 3-5 N, fingers 2.5-5 N)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (0/336 knock-overs at +45 mm) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm, inside the 0-25 mm lift group)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin (no lift with thumb >= 50 mm)

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
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup_terms(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok):
    """Unweighted per-arm terms in [0, 1]. Reach / grasp / close are iter_04 VERBATIM (they work:
    both hands grasp in 4/4 deterministic envs). iter_05 only removes the lift PAY from here — it is
    assembled in compute_reward so that no product of sub-1 gates sits on the early lift."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: long-range funnel term + per-side fingertip-on-wall term (unchanged).
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: balanced two-sided pinch (unchanged). Flag boundary 1 N / 1 N: 0.355 -> 0.75, never a valley.
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # trace: thumb 3-5 N vs fingers 2.5-5 N -> ~0.98
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

    # closure shaping (unchanged): measured closure 0.55-0.65 on both hands while holding
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift STATE only. 5 mm dead band: tipping on the rim is not lifting.
    h_pos = torch.clamp(cup_pos[:, 2] - cup_spawn_pos[:, 2] - 0.005, min=0.0)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    v_lin = cup_lin_vel.norm(dim=-1)

    # penalties (unweighted, bounded) — all unchanged from iter_04
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)        # 1 N dead band: a stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "h_pos": h_pos, "lifted_w": lifted_w,
            "held": g_f, "pinch": pinch, "disp": disp, "v_lin": v_lin, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling,
            "far": torch.tanh(4.0 * geo["d_funnel"])}


def _pour_geometry(ctx, outer_r, pour_sigma):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    align_gate = torch.exp(-(d_align / 0.06) ** 2)                       # wide: tilting is allowed here
    # iter_05: sigma 0.041 -> ~0.04-0.05. With aim_dist measured at 0.32-1.16 m the old gate was exp(-60),
    # which is exactly the logged pour_tilt of 1e-38..1e-8. The wide lead-in now lives in tilt_lead.
    pour_gate = torch.exp(-(d_align / pour_sigma) ** 2)
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift per arm, then carry < align < tilt < pour.
    # iter_05 fixes three measured failures: (1) the source lift was strangled by a gate product,
    # (2) the receiver over-lifted and dragged its cup, (3) the tilt stage had no reachable gradient.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (it kept the receiver at its cup:
    W_AWAY_RCV = 0.8                       # rcv_near_rate 0.84 vs 0.05 in i03)
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (4/4 envs grasp)
    # --- source lift (iter_18 shape) -------------------------------------------------------------
    W_LIFT_SRC = 4.0        # was 3.0 and shared with the receiver; the receiver channel drops to 1.5
    LIFTOFF_H = 0.03        # e-fold of the lift-off half: 3 cm -> 0.57 of it, so the first cm pay most
    LIFT_TARGET_SRC = 0.15  # unchanged; the trace only ever reached 20-76 mm
    LIFT_OVER_W = 0.15      # keeps paying past the target instead of stopping dead at 0.15 m
    LIFT_OVER_RANGE = 0.10
    LIFT_FLOOR = 0.60       # was 0.15. Measured: the gate opened 0.70 -> 0.72 while lift_src stayed at
                            # 0.20 for 330 epochs, so it was not the blocker; keep the coupling, cap the loss
    SRC_UP_FLOOR = 0.5      # iter_18 floor: attitude may halve the lift income, never zero it (it was 0.11
                            # at the measured cup_up_z 0.87)
    LIFT_PINCH_BONUS = 0.2  # grip quality is a BONUS, not a gate (iter_18 lesson)
    # --- receiver lift band ----------------------------------------------------------------------
    W_LIFT_RCV = 1.5        # was 3.0: measured lift_rcv 0.614 vs lift_src 0.209 — the easy motion paid 3x
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10    # 0.08 target + 2 cm margin
    RCV_LIFT_SIGMA = 0.05   # 0.15 m -> 0.45, 0.19 m -> 0.06: the measured 120-190 mm overshoot loses the pay
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7      # measured reach_rcv away / near the cup
    # --- carry / align / tilt --------------------------------------------------------------------
    W_CARRY, W_ALIGN = 2.0, 4.0            # unchanged weights
    W_TILT_LEAD = 1.5       # new: the wide, safe part of the tilt (<= 45 deg, no bead can leave)
    W_TILT = 6.0            # was 5.0; this is the stage the policy never entered (pour_tilt 0.000)
    TILT_PRE = math.radians(45.0)   # first bead leaves at ~70 deg with a full cup -> 45 deg cannot spill
    TILT_FULL = math.radians(100.0)
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # same Gaussian width as iter_04, now applied to the corridor EXCESS
    TILT_LIFT_H = 0.03      # tilt corridor and tilt_lead are fully open at 3 cm — the policy already
                            # reaches 20-76 mm, so the gradient is live where the policy lives
    LEAD_RANGE = 0.60       # aim_dist measured 0.32 m -> lead factor 0.63 (was exp(-60) = 0)
    # --- receiver discipline ---------------------------------------------------------------------
    W_HOME_RCV = 0.8        # measured drag 210-270 mm with slide switched off above 2.5 cm
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12       # income factor: 0.21 m -> 0.05, so converging by dragging the receiver fails
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5   # 0.3 base, 0.8 while the source is carried
    STILL_FREE, STILL_SCALE = 0.05, 0.15   # 0.2 m/s -> tanh(1.0) = 0.76
    # --- task / penalties (all unchanged from iter_04) --------------------------------------------
    W_POUR = 150.0          # x d_in_target: 7.5 per bead with 20 beads
    W_SPILL = 60.0
    W_SUCCESS = 20.0
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_TILT_PEN = 0.3
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05
    CUP_OUTER_R = 0.0285
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35

    dtype = ctx.src_cup_tilt.dtype
    pour_sigma = max(1.2 * ctx.cup_radius, 0.04)   # beads must land inside the receiver mouth
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R, pour_sigma)

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)

    # ---- source attitude CORRIDOR (replaces the fixed upright gate) --------------------------------
    # free tilt: 0.15 rad on the table -> 45 deg once grasped and 3 cm up -> 100 deg when aligned.
    # A toppled cup on the table is 1.57 rad against 0.15 rad, so src_up ~ 0 exactly as in iter_04,
    # but a deliberate carried pour no longer deletes grasp_src (0.71 measured) and the lift pay.
    carry_soft_src = g_src * torch.clamp(h_src / TILT_LIFT_H, max=1.0)
    tilt_free = TILT_FREE_TABLE + (TILT_PRE - TILT_FREE_TABLE) * carry_soft_src
    tilt_free = tilt_free + (TILT_FULL - tilt_free) * align_gate
    tilt_excess_src = torch.clamp(ctx.src_cup_tilt - tilt_free, min=0.0)
    src_up = torch.exp(-(tilt_excess_src / TILT_TOL) ** 2)
    pose_ok_src = src_up
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                          ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up,
                          ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_src)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                          ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                          ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_rcv)

    # ---- receiver progress gate on the source lift -------------------------------------------------
    up_w_src = g_src * torch.clamp(h_src / 0.05, max=1.0)      # "held and at least 5.5 cm up"
    up_w_rcv = g_rcv * torch.clamp(h_rcv / 0.05, max=1.0)
    prog_reach = torch.clamp((rcv["reach"] - REACH_AWAY) / (REACH_NEAR - REACH_AWAY), 0.0, 1.0)
    prog_grasp = torch.clamp(rcv["grasp"] / 0.4, max=1.0)
    rcv_prog = PROG_REACH * prog_reach + PROG_GRASP * torch.maximum(prog_grasp, up_w_rcv)
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # 0.60 .. 1.00 (measured rcv_prog 0.64 -> 0.86)

    # ---- source lift: iter_18 shape, single floored attitude factor, grip as a bonus ---------------
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_over = torch.clamp((h_src - LIFT_TARGET_SRC) / LIFT_OVER_RANGE, 0.0, 1.0)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0) \
        + LIFT_OVER_W * lift_over                              # 3 cm -> 0.37, 15 cm -> 0.98, 25 cm -> 1.15
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])

    # ---- receiver lift: banded around its 0.08 m target, over-lifting loses the pay ----------------
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * rcv_band * (0.5 + 0.5 * upright_rcv)

    # ---- receiver stays where it spawned and stays still --------------------------------------------
    d_home = rcv["disp"]
    home_f = 0.3 + 0.7 * torch.exp(-(d_home / HOME_SIGMA) ** 2)   # income factor for carry / align
    rcv_home_pen = rcv["held"] * torch.tanh(torch.clamp(d_home - HOME_FREE, min=0.0) / HOME_SCALE)
    still_w = W_STILL_RCV + W_STILL_POUR * up_w_src               # stricter while the source is carried
    rcv_still_pen = still_w * rcv["held"] \
        * torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)

    # ---- carry / align: the SOURCE must close the gap, not the receiver ----------------------------
    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * home_f * (1.0 - nested)
    carry = carry_w * (0.5 * (1.0 - torch.tanh(2.5 * d_align)) + 0.5 * torch.clamp(1.0 - d_align / 0.45, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * home_f * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(4.0 * d_align)) + 0.5 * torch.exp(-25.0 * d_align))

    # ---- tilt: a wide safe lead-in, then the narrow aligned pour -----------------------------------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    lead_near = torch.clamp(1.0 - d_align / LEAD_RANGE, 0.0, 1.0)
    tilt_lead = carry_soft_src * beads_left * (1.0 - nested) \
        * torch.clamp(ctx.src_cup_tilt / TILT_PRE, max=1.0) * (0.3 + 0.7 * lead_near)
    tilt_beyond = torch.clamp(torch.clamp(ctx.src_cup_tilt, max=TILT_FULL) - TILT_PRE, min=0.0) \
        / (TILT_FULL - TILT_PRE)
    pour_tilt = both_up * pour_gate * beads_left * upright_rcv * tilt_beyond

    tilt_pen_src = torch.tanh(tilt_excess_src / TILT_TOL)       # corridor excess only
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "reach_src": W_REACH_SRC * src["reach"],
        "reach_rcv": W_REACH_RCV * rcv["reach"],
        "away_rcv": -W_AWAY_RCV * rcv["far"],
        "close_src": W_CLOSE_SRC * src["close"],
        "close_rcv": W_CLOSE_RCV * rcv["close"],
        "grasp_src": W_GRASP * src["grasp"],
        "grasp_rcv": W_GRASP * rcv["grasp"],
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "carry_src": W_CARRY * carry,
        "align_mouths": W_ALIGN * align,
        "tilt_lead": W_TILT_LEAD * tilt_lead,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE_SRC * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE_RCV * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
        "rcv_home": -W_HOME_RCV * rcv_home_pen,
        "rcv_still": -rcv_still_pen,
        "contact_push": -(W_ONE_SIDED_SRC * src["one_sided"] + W_ONE_SIDED_RCV * rcv["one_sided"]
                          + W_PALM_SRC * src["palm_push"] + W_PALM_RCV * rcv["palm_push"]),
        "rush_near_cup": -(W_RUSH_SRC * src["rush"] + W_RUSH_RCV * rcv["rush"]),
        "cup_fling": -W_FLING * (src["fling"] + rcv["fling"]),
        "cup_collision": -W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0),
        "hand_foreign": -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                      + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)),
        "nested": -W_NESTED * nested,
        "action_rate": -W_ACT_RATE * act_rate,
        "rcv_progress": torch.zeros_like(rcv_prog),
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    comps["rcv_progress"] = rcv_prog   # diagnostic only (added after the sum, not part of the reward)
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 4.88e-05, 0, 0, 0, 4.88e-05, 9.77e-05, 4.88e-05, 9.77e-05, 0]  min 0 · mean 4.91e-05 · max 0.000488
bead/spill: [0, 0.0198, 0.0232, 0.028, 0.0251, 0.0262, 0.027, 0.0262, 0.0398, 0.024]  min 0 · mean 0.0258 · max 0.043
done/drop: [0, 0, 0.00195, 0.00195, 0.000977, 0.00195, 0, 0, 0.000977, 0.00293]  min 0 · mean 0.000963 · max 0.00488
done/mimic_runaway: [0, 0, 0, 0.000977, 0, 0.00293, 0, 0, 0.00195, 0]  min 0 · mean 0.000652 · max 0.00391
episode_lengths/step: [97, 483, 486, 487, 376, 433, 436, 498, 447, 473]  min 97 · mean 450 · max 604
reward/action_rate: [-0.0399, -0.0338, -0.034, -0.0337, -0.0332, -0.0337, -0.0342, -0.0334, -0.0337, -0.0335]  min -0.0399 · mean -0.0336 · max -0.0314
reward/align_mouths: [0, 0.0377, 0.0387, 0.0351, 0.0545, 0.114, 0.157, 0.144, 0.135, 0.208]  min 0 · mean 0.105 · max 0.231
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/away_rcv: [-0.07, -0.142, -0.134, -0.138, -0.13, -0.139, -0.145, -0.149, -0.149, -0.153]  min -0.161 · mean -0.143 · max -0.07
reward/carry_src: [0, 0.0452, 0.0573, 0.0578, 0.0742, 0.115, 0.141, 0.165, 0.162, 0.251]  min 0 · mean 0.122 · max 0.275
reward/close_rcv: [0.0754, 0.102, 0.112, 0.105, 0.105, 0.0863, 0.0765, 0.0686, 0.0659, 0.0684]  min 0.0604 · mean 0.0879 · max 0.141
reward/close_src: [0.0458, 0.0888, 0.0955, 0.0931, 0.0942, 0.0971, 0.0937, 0.0992, 0.0881, 0.0884]  min 0.0458 · mean 0.0942 · max 0.113
reward/contact_push: [-0.00868, -0.0631, -0.0596, -0.0511, -0.0448, -0.0363, -0.0326, -0.0358, -0.0299, -0.0292]  min -0.142 · mean -0.0437 · max -0.00868
reward/cup_collision: [0, -0.000971, 0, -0.000532, 0, 0, -0.000957, -0.000977, -0.00389, -0.00293]  min -0.0128 · mean -0.0012 · max 0
reward/cup_disturb_rcv: [-0.0291, -0.112, -0.116, -0.114, -0.128, -0.136, -0.136, -0.122, -0.11, -0.0999]  min -0.145 · mean -0.121 · max -0.0291
reward/cup_disturb_src: [-0.0642, -0.218, -0.204, -0.182, -0.185, -0.134, -0.113, -0.0988, -0.103, -0.102]  min -0.332 · mean -0.149 · max -0.0642
reward/cup_fling: [-0.000454, -0.0181, -0.0176, -0.0187, -0.0227, -0.0253, -0.0223, -0.0199, -0.0272, -0.0207]  min -0.0328 · mean -0.0205 · max -0.000454
reward/grasp_rcv: [0.00166, 0.74, 0.742, 0.72, 0.66, 0.558, 0.504, 0.496, 0.486, 0.521]  min 0.00166 · mean 0.604 · max 1.21
reward/grasp_src: [0.00657, 0.865, 0.941, 0.9, 0.977, 1.07, 1.08, 1.15, 1.04, 1.12]  min 0.00657 · mean 1.04 · max 1.48
reward/hand_foreign: [-0.000324, -0.00337, -0.00137, -0.00346, -0.00322, -0.00196, -0.000607, -0.000853, -0.00316, -0.00428]  min -0.0195 · mean -0.00288 · max -6.1e-05
reward/lift_rcv: [0, 0.324, 0.381, 0.449, 0.408, 0.24, 0.137, 0.0617, 0.0497, 0.0306]  min 0 · mean 0.221 · max 0.758
reward/lift_src: [0, 0.524, 0.769, 0.948, 1.12, 1.72, 1.97, 2.25, 2.1, 2.25]  min 0 · mean 1.53 · max 2.53
reward/nested: [0, -0.00195, 0, -0.00195, 0, 0, 0, -0.00195, -0.00586, -0.00195]  min -0.0117 · mean -0.00144 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean 0.000218 · max 0.0146
reward/pour_tilt: [0, 1.68e-36, 2.25e-22, 1.04e-10, 1.7e-14, 5.92e-11, 1.75e-11, 8.77e-08, 1.01e-18, 7.83e-08]  min 0 · mean 9.35e-07 · max 0.000317
reward/rcv_home: [0, -0.228, -0.195, -0.134, -0.151, -0.164, -0.139, -0.15, -0.139, -0.12]  min -0.411 · mean -0.165 · max 0
reward/rcv_progress: [0.263, 0.639, 0.652, 0.631, 0.647, 0.628, 0.615, 0.598, 0.582, 0.599]  min 0.263 · mean 0.622 · max 0.862
reward/rcv_still: [0, -0.144, -0.162, -0.173, -0.206, -0.219, -0.228, -0.222, -0.207, -0.202]  min -0.249 · mean -0.197 · max 0
reward/reach_rcv: [0.863, 0.924, 0.946, 0.928, 0.969, 0.931, 0.897, 0.851, 0.847, 0.847]  min 0.82 · mean 0.905 · max 1.02
reward/reach_src: [0.603, 0.707, 0.726, 0.706, 0.732, 0.736, 0.716, 0.736, 0.693, 0.709]  min 0.603 · mean 0.722 · max 0.829
reward/rush_near_cup: [-0.24, -0.0855, -0.079, -0.0844, -0.0758, -0.0741, -0.072, -0.0727, -0.0766, -0.0705]  min -0.24 · mean -0.0749 · max -0.0458
reward/spill_delta: [0, 0.00293, -0.0176, -0.00586, -0.0615, -0.0586, -0.0234, -0.0322, -0.0586, -0.00293]  min -0.135 · mean -0.0291 · max 0.0264
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_lead: [0, 0.113, 0.137, 0.134, 0.222, 0.344, 0.297, 0.317, 0.274, 0.315]  min 0 · mean 0.237 · max 0.356
reward/total: [2.14, 4.36, 4.85, 5.07, 5.3, 5.91, 6.04, 6.32, 5.93, 6.48]  min 2.14 · mean 5.61 · max 6.78
rewards/step: [111, 2.12e+03, 2.51e+03, 2.61e+03, 1.87e+03, 2.59e+03, 2.63e+03, 3.25e+03, 2.7e+03, 2.97e+03]  min 111 · mean 2.52e+03 · max 3.51e+03
task/aim_dist: [0.32, 0.317, 0.81, 0.397, 0.409, 0.322, 0.281, 0.323, 0.936, 0.579]  min 0.263 · mean 0.655 · max 2.76
task/cup_collision_rate: [0, 0.000977, 0, 0.000977, 0, 0, 0.000977, 0.000977, 0.00391, 0.00293]  min 0 · mean 0.00127 · max 0.0127
task/cups_center_dist: [0.32, 0.32, 0.813, 0.401, 0.414, 0.33, 0.286, 0.33, 0.944, 0.584]  min 0.269 · mean 0.66 · max 2.77
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.000977, 0, 0.000977, 0, 0, 0, 0.000977, 0.00293, 0.000977]  min 0 · mean 0.000719 · max 0.00586
task/rcv_cup_lift: [8.58e-05, 0.0806, 0.0768, 0.0612, 0.082, 0.0912, 0.121, 0.156, 0.167, 0.205]  min 8.58e-05 · mean 0.133 · max 0.583
task/rcv_grasped: [0, 0.485, 0.499, 0.487, 0.505, 0.467, 0.468, 0.446, 0.431, 0.441]  min 0 · mean 0.471 · max 0.716
task/rcv_hand_foreign_rate: [0.000977, 0.00977, 0.00293, 0.00586, 0.00781, 0.00586, 0.00195, 0.00195, 0.00684, 0.00977]  min 0 · mean 0.00742 · max 0.0713
task/src_cup_lift: [0.000191, 0.0157, 0.0899, 0.0319, 0.0393, 0.0886, 0.0876, 0.134, 0.411, 0.229]  min 0.000191 · mean 0.178 · max 1.19
task/src_grasped: [0, 0.451, 0.491, 0.469, 0.537, 0.554, 0.554, 0.579, 0.525, 0.562]  min 0 · mean 0.538 · max 0.786
task/src_hand_foreign_rate: [0, 0.000977, 0.00195, 0.00586, 0.000977, 0.000977, 0, 0.00195, 0.00488, 0.00195]  min 0 · mean 0.00145 · max 0.00684
task/src_tilt_deg: [0.672, 12.9, 14.5, 15.3, 20.3, 22.8, 19, 19.5, 19.3, 18.1]  min 0.672 · mean 17.7 · max 23.6
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Operator notes for round 6 (iter_05 result). Every number is measured. Hypotheses are marked as such. Env unchanged: hand-cup friction 2.0, contact-freeze threshold 1.0 N, grasp flag = thumb > 1.0 N AND another finger > 1.0 N.

## What iter_05 fixed (run t2r_rh2_i05, 1024 envs, warm from i04 epoch-1650, 569 epochs)
50-epoch means from the start of the run to its end:
- reward/lift_src 0.511 -> 2.18 (iter_04 ended at 0.117 after a 330-epoch ceiling at 0.2).
- task/src_cup_lift 0.029 -> 0.35 m (iter_04 deterministic trace: 20-76 mm).
- reward/tilt_lead 0.113 -> 0.31 and task/src_tilt_deg 12.9 -> 19 deg. In iter_04 there was no tilt channel at all and the source cup never left cup_up_z 0.87-0.96.
- reward/carry_src 0.043 -> 0.239, reward/align_mouths 0.039 -> 0.201.
- task/src_grasped 0.486 -> 0.570, rewards/iter 1874 -> 2986.
So the lift rebuild (iter_18 shape, attitude corridor, LIFT_FLOOR 0.60) and tilt_lead both worked.

## What iter_05 did not fix, with the measured cause
reward/pour_tilt was 0.000 at every epoch and task/episode_success 0.000 at every epoch.

1. THE SOURCE LIFT HAS NO UPPER BOUND. `lift_shape_src = 0.5*(1-exp(-h/0.03)) + 0.5*clamp(h/0.15) + 0.15*clamp((h-0.15)/0.10)` keeps paying to 0.25 m and never charges for going higher. Measured result: task/src_cup_lift 50-epoch means 0.144, 0.335, 0.298, 0.350, 0.260 m against LIFT_TARGET_SRC 0.15 m, with per-epoch values ranging 0.18-0.57 m.
2. THE TWO CUPS SEPARATED. task/cups_center_dist 50-epoch means 0.379 -> 0.875 m (spawn separation is about 0.32 m); last 10 epochs 0.514 m. The logged aim_dist went 0.30 -> 0.75 m.
3. THAT SEPARATION SWITCHES BOTH TILT CHANNELS OFF (read from iter_05 code, checked against the logs):
   - `tilt_lead` multiplies `(0.3 + 0.7 * clamp(1 - d_align / LEAD_RANGE))` with LEAD_RANGE 0.60. It was sized on the then-measured aim_dist 0.32 m, which gives 0.63. At the d_align 0.87-0.97 m that the run actually reached, `lead_near` is 0 and the factor collapses to its floor 0.3.
   - `pour_tilt = both_up * pour_gate * beads_left * upright_rcv * tilt_beyond`, and `pour_gate` is a narrow Gaussian on d_align. At ~0.9 m it is numerically zero. This is why pour_tilt stayed 0.000 for the whole run even though the source cup did tilt to about 19 deg.
   - So the 45 deg cap in tilt_lead is NOT what stopped the tilt at 19 deg. The distance condition is.
4. THE INCENTIVE RATIO FAVOURS LIFTING OVER CONVERGING. Measured at the end: lift_src 2.18 against carry_src 0.239 and align_mouths 0.201, i.e. about 10 to 1 against closing the gap.
5. THE RECEIVER DISCIPLINE ERODED ITS GRASP. reward/grasp_rcv 50-epoch means 0.734, 0.722, 0.738, 0.747, 0.620, 0.576, 0.557, 0.503, 0.495, 0.428, 0.515. task/rcv_grasped 0.497 -> 0.404, recovering to 0.435. reward/close_rcv 0.115 -> 0.065, reward/reach_rcv 0.964 -> 0.848.
   The receiver lift band itself behaves as designed but was simply abandoned: task/rcv_cup_lift 50-epoch means 0.091 -> 0.221 m against RCV_LIFT_FREE 0.10 m, and reward/lift_rcv collapsed 0.307 -> 0.024 as a result. The policy accepted losing that income rather than staying in the band.
   reward/rcv_still stayed at -0.15 to -0.23 and reward/rcv_home at -0.12 to -0.25 throughout.
6. losses/entropy rose monotonically 25.5 -> 26.9 over the whole run; it has never fallen in any run of this track.

## What the operator wants from iter_06
1. Bound the source lift. It must stop paying, or start costing, above a height from which pouring is possible. Measured overshoot: 0.35 m mean, 0.57 m peak, against a 0.15 m target.
2. Make closing the gap between the cups worth clearly more than lifting higher. Measured ratio is about 10:1 the wrong way (lift_src 2.18 vs carry 0.239 + align 0.201). The pour cannot start while the cups sit 0.5-0.9 m apart.
3. Size the tilt distance conditions on the distance the policy actually reaches, or make them a consequence of convergence rather than a hard gate. Measured: LEAD_RANGE 0.60 was set for 0.32 m and the run lives at 0.5-0.97 m; pour_gate is numerically zero there.
4. Stop the receiver grasp erosion (grasp_rcv 0.73 -> 0.43-0.52, rcv_grasped 0.50 -> 0.40-0.44, close_rcv 0.115 -> 0.065) while keeping the anti-drag intent that did work. Staying inside the receiver band must be worth more than abandoning it.
5. Keep what works: the grasp formulation for both hands, the iter_18-shape lift-off gradient, the attitude corridor, and tilt_lead as a channel (it moved the tilt from nothing to about 19 deg).

## Launch plan
iter_06 starts warm from the i05 epoch-550 checkpoint, 1024 envs, same env physics.
