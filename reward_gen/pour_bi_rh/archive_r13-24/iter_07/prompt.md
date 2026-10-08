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
    iter_06: unchanged since iter_04 — the grasp works (4/4 envs, thumb 3-5 N, fingers 2.5-5 N)."""
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
    both hands grasp in 4/4 deterministic envs, and iter_05 measured grasp_src 1.04 mean). The lift
    PAY is assembled in compute_reward so that no product of sub-1 gates sits on the early lift."""
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


def _pour_geometry(ctx, outer_r, pour_sigma, align_sigma):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    # iter_06: both sigmas widened. The corridor opener was 0.06 m and the pour gate 0.04 m, i.e.
    # exp(-43) at the closest distance the whole i05 run ever reached (aim_dist min 0.263 m) — that is
    # the logged pour_tilt of exactly 0.000 for 569 epochs, not a hard task.
    align_gate = torch.exp(-(d_align / align_sigma) ** 2)
    pour_gate = torch.exp(-(d_align / pour_sigma) ** 2)
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt < pour.
    # iter_06 breaks the measured i05 loop: an unbounded source lift (0.35 m mean, 0.57 m peak against
    # a 0.15 m target) -> cups 0.5-0.97 m apart (spawn 0.32 m) -> both tilt channels switched off by
    # distance (pour_tilt 0.000, success 0.000 for 569 epochs) -> receiver chases the height and leaves
    # its band (rcv_cup_lift 0.091 -> 0.221 m, lift_rcv 0.307 -> 0.024, grasp_rcv 0.73 -> 0.43).
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.85-0.97 measured:
    W_AWAY_RCV = 0.8                       # it keeps the left hand at its own cup)
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (4/4 envs grasp)
    # --- source lift: iter_18 lift-off shape, now inside a BAND -----------------------------------
    W_LIFT_SRC = 3.0        # was 4.0. Measured lift_src 2.18 vs carry 0.239 + align 0.201 = 10:1 the
                            # wrong way; the lift is solved (src_cup_lift 0.029 -> 0.35 m), convergence is not
    LIFTOFF_H = 0.03        # unchanged iter_18 lift-off e-fold: the first cm pay most
    LIFT_TARGET_SRC = 0.15  # unchanged: shape saturates here
    LIFT_FREE_SRC = 0.24    # pourable height from the env geometry (cup_scale 0.65): receiver rim at its
                            # 0.10 m band edge 0.1539 + 0.05 clearance + 0.0286 lip drop = 0.233 m
    LIFT_SIGMA_SRC = 0.08   # band 1.00 at 0.21 m, 0.15 at the measured 0.35 m mean, 4e-8 at the 0.57 m peak
    W_OVER_LIFT = 1.5       # the band alone is FLAT at 0.35-0.57 m; this ramp gives a live gradient there
    OVER_LIFT_RANGE = 0.20  # linear 0.24 -> 0.44 m: -0.82 at the 0.35 m mean, -1.50 at the 0.57 m peak
    LIFT_FLOOR = 0.60       # unchanged. Measured: the gate opened 0.70 -> 0.72 while lift_src was flat,
    SRC_UP_FLOOR = 0.5      # so it is not the blocker. Attitude may halve the lift income, never zero it.
    LIFT_PINCH_BONUS = 0.2  # grip quality is a BONUS, not a gate (iter_18 lesson)
    # --- receiver lift band -----------------------------------------------------------------------
    W_LIFT_RCV = 2.0        # was 1.5: the band was ABANDONED (lift_rcv 0.307 -> 0.024), not over-paid
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10    # 0.08 target + 2 cm margin (unchanged)
    RCV_LIFT_SIGMA = 0.07   # was 0.05 -> exp(-5.85) = 0.003 at the measured 0.221 m, a dead escape.
                            # 0.07 gives 0.050 at 0.221 m, 0.60 at 0.15 m, 0.92 at 0.12 m: a way back.
    W_HOLD_RCV = 1.2        # new: direct pay for grasped + in band + at home + still. The receiver lost
                            # ~0.63/step by drifting (grasp_rcv -0.30, lift_rcv -0.28, close_rcv -0.05)
    RCV_DISC_FLOOR = 0.35   # discipline factor floor: the source keeps a convergence gradient during the
                            # warm start, when the receiver still sits at the measured 0.205 m
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7      # measured reach_rcv away / near the cup
    # --- convergence: the stage that never happened ------------------------------------------------
    W_CONVERGE = 3.0        # new, on the cup-cup xy distance — the quantity the env's success test uses
    CONV_FAR = 1.00         # measured cups_center_dist 50-epoch means to 0.875 m, per-epoch 0.94-0.97 m
    CONV_NEAR = 0.16        # under the success threshold 0.20 m, above the nested threshold 0.09 m
    CONV_SIGMA = 0.25       # close-in half: 0.38 at 0.50 m, 0.74 at the 0.32 m spawn, 0.96 at 0.20 m
    W_CARRY, W_ALIGN = 3.0, 5.0            # were 2.0 / 4.0; measured carry 0.239 + align 0.201 vs lift 2.18
    CARRY_K, CARRY_RANGE = 1.2, 1.00       # was tanh(2.5*d) + ramp 0.45 -> 0.011 at the measured 0.9 m
    ALIGN_K, ALIGN_SIGMA = 2.0, 0.15       # was tanh(4.0*d) + exp(-25*d) -> 0.03 at 0.30 m
    W_TILT_LEAD = 1.5       # channel and weight kept: it moved the tilt from nothing to 19 deg
    LEAD_FLOOR = 0.25       # the distance factor is now convergence, not a 0.60 m gate pinned at its floor
    W_TILT = 8.0            # was 6.0; still the stage the policy has never entered (pour_tilt 0.000)
    TILT_PRE = math.radians(45.0)   # first bead leaves at ~70 deg with a full cup -> 45 deg cannot spill
    TILT_FULL = math.radians(100.0)
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # Gaussian width on the corridor EXCESS (unchanged)
    TILT_LIFT_H = 0.03      # corridor and tilt_lead fully open at 3 cm — the policy lives well above it
    ALIGN_GATE_SIGMA = 0.12  # was 0.06: the 100 deg corridor must be open wherever pour_tilt pays
                             # (0.21 -> 57 deg free at d_align 0.15 m, 0.64 -> 80 deg at 0.08 m)
    # --- receiver discipline -----------------------------------------------------------------------
    W_HOME_RCV = 0.8        # unchanged anti-drag penalty
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12       # unchanged income factor on the drag
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5   # 0.3 base, 0.8 while the source is carried (unchanged)
    STILL_FREE, STILL_SCALE = 0.05, 0.15   # 0.2 m/s -> tanh(1.0) = 0.76
    # --- task / penalties (all unchanged) ----------------------------------------------------------
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
    CUP_OUTER_R = 0.0285    # 0.044 x cup_scale 0.65
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35

    dtype = ctx.src_cup_tilt.dtype
    # 4 x inner radius (0.0281 m at cup_scale 0.65) with a 0.15 m floor: 0.046 at the measured best
    # aim_dist 0.263 m instead of exp(-43), 0.37 at 0.15 m, 0.018 at 0.30 m (too small to pay for a dump).
    pour_sigma = max(4.0 * ctx.cup_radius, 0.15)
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R, pour_sigma, ALIGN_GATE_SIGMA)

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)

    # ---- source attitude CORRIDOR (kept from iter_05) ----------------------------------------------
    # free tilt: 0.15 rad on the table -> 45 deg once grasped and 3 cm up -> 100 deg when aligned.
    # A toppled cup on the table is 1.57 rad against 0.15 rad, so src_up ~ 0 exactly as in iter_04.
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

    # ---- receiver progress gate on the source lift (unchanged) -------------------------------------
    up_w_src = g_src * torch.clamp(h_src / 0.05, max=1.0)      # "held and at least 5.5 cm up"
    up_w_rcv = g_rcv * torch.clamp(h_rcv / 0.05, max=1.0)
    prog_reach = torch.clamp((rcv["reach"] - REACH_AWAY) / (REACH_NEAR - REACH_AWAY), 0.0, 1.0)
    prog_grasp = torch.clamp(rcv["grasp"] / 0.4, max=1.0)
    rcv_prog = PROG_REACH * prog_reach + PROG_GRASP * torch.maximum(prog_grasp, up_w_rcv)
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # 0.60 .. 1.00 (measured rcv_prog 0.60-0.65)

    # ---- source lift: iter_18 shape INSIDE A BAND, plus a linear cost above it ---------------------
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_lift_band = torch.exp(-(torch.clamp(h_src - LIFT_FREE_SRC, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_lift_band * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])
    # ungated by the grasp flag on purpose: throwing the cup up must not dodge the cost
    over_lift = torch.clamp((h_src - LIFT_FREE_SRC) / OVER_LIFT_RANGE, 0.0, 1.0)

    # ---- receiver lift: band around its 0.08 m target, now climbable from the measured 0.221 m -----
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * rcv_band * (0.5 + 0.5 * upright_rcv)

    # ---- receiver discipline: home, still, and a single factor on all convergence income -----------
    d_home = rcv["disp"]
    home_near = torch.exp(-(d_home / HOME_SIGMA) ** 2)
    still_f = 1.0 - torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    # iter_05 scaled carry/align by the drag only, so the receiver could still buy alignment by LIFTING
    # (measured 0.091 -> 0.221 m). Leaving the band now costs the same income as dragging.
    rcv_disc = RCV_DISC_FLOOR + (1.0 - RCV_DISC_FLOOR) * rcv_band * home_near
    rcv_home_pen = rcv["held"] * torch.tanh(torch.clamp(d_home - HOME_FREE, min=0.0) / HOME_SCALE)
    still_w = W_STILL_RCV + W_STILL_POUR * up_w_src               # stricter while the source is carried
    rcv_still_pen = still_w * rcv["held"] \
        * torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    # positive pay for doing exactly the right thing, scaled by the source carry state so an idle
    # receiver cannot out-earn the source lifting (iter_18 home_src_f lesson)
    hold_rcv = W_HOLD_RCV * g_rcv * rcv_band * home_near * still_f * (0.4 + 0.6 * carry_soft_src)

    # ---- convergence: the SOURCE closes the gap, sized on the distance the policy reaches ----------
    d_cups_xy = (ctx.src_cup_pos[:, :2] - ctx.rcv_cup_pos[:, :2]).norm(dim=-1)
    conv_lin = torch.clamp((CONV_FAR - d_cups_xy) / (CONV_FAR - CONV_NEAR), 0.0, 1.0)
    conv_close = torch.exp(-(torch.clamp(d_cups_xy - CONV_NEAR, min=0.0) / CONV_SIGMA) ** 2)
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close     # 0.06 at 0.90 m, 0.38 at 0.50 m, 0.96 at 0.20 m
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.44 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * (1.0 - nested) * conv_shape

    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * (1.0 - nested)
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- tilt: the safe lead-in is now pulled by convergence, the pour gate is reachable -----------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_lead = carry_soft_src * beads_left * (1.0 - nested) \
        * torch.clamp(ctx.src_cup_tilt / TILT_PRE, max=1.0) * (LEAD_FLOOR + (1.0 - LEAD_FLOOR) * conv_shape)
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
        "over_lift_src": -W_OVER_LIFT * over_lift,
        "hold_rcv": hold_rcv,
        "converge": converge,
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
bead/in_target: [0, 9.77e-05, 0, 9.77e-05, 4.88e-05, 4.88e-05, 0.000146, 9.77e-05, 0, 9.77e-05]  min 0 · mean 0.00013 · max 0.000586
bead/spill: [4.88e-05, 0.0347, 0.0266, 0.04, 0.0218, 0.027, 0.0357, 0.0278, 0.025, 0.0329]  min 4.88e-05 · mean 0.0307 · max 0.0477
done/drop: [0, 0, 0, 0.000977, 0.000977, 0.000977, 0.000977, 0, 0, 0.000977]  min 0 · mean 0.000923 · max 0.00684
done/mimic_runaway: [0, 0, 0, 0.000977, 0, 0, 0, 0.000977, 0, 0]  min 0 · mean 0.000608 · max 0.00586
episode_lengths/step: [92.9, 487, 459, 464, 506, 479, 455, 481, 504, 514]  min 92.9 · mean 480 · max 621
reward/action_rate: [-0.0388, -0.0339, -0.0345, -0.0339, -0.0338, -0.0336, -0.0345, -0.034, -0.0324, -0.0329]  min -0.0388 · mean -0.0337 · max -0.0319
reward/align_mouths: [0, 0.367, 0.346, 0.334, 0.308, 0.413, 0.691, 0.629, 0.872, 0.757]  min 0 · mean 0.529 · max 0.989
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/away_rcv: [-0.0884, -0.163, -0.161, -0.156, -0.168, -0.159, -0.161, -0.172, -0.167, -0.17]  min -0.178 · mean -0.163 · max -0.0884
reward/carry_src: [0, 0.278, 0.225, 0.227, 0.519, 0.543, 0.613, 0.608, 0.705, 0.604]  min 0 · mean 0.469 · max 0.756
reward/close_rcv: [0.0475, 0.0722, 0.0683, 0.0715, 0.067, 0.0747, 0.0684, 0.0677, 0.0652, 0.0611]  min 0.0475 · mean 0.0689 · max 0.096
reward/close_src: [0.0394, 0.0993, 0.0894, 0.0817, 0.0864, 0.0829, 0.0859, 0.0926, 0.0941, 0.0905]  min 0.0394 · mean 0.0889 · max 0.113
reward/contact_push: [-0.0262, -0.0319, -0.0286, -0.0367, -0.0191, -0.027, -0.0283, -0.0286, -0.0252, -0.0278]  min -0.147 · mean -0.0274 · max -0.00662
reward/converge: [0, 0.492, 0.449, 0.46, 0.796, 0.82, 1.05, 1.09, 1.22, 1.06]  min 0 · mean 0.821 · max 1.3
reward/cup_collision: [0, -0.000977, -0.00488, -0.00375, -0.00195, -0.000977, 0, -0.000455, 0, -0.00101]  min -0.0164 · mean -0.00271 · max 0
reward/cup_disturb_rcv: [-0.0231, -0.125, -0.145, -0.136, -0.137, -0.123, -0.124, -0.151, -0.142, -0.141]  min -0.198 · mean -0.134 · max -0.0231
reward/cup_disturb_src: [-0.11, -0.0916, -0.0956, -0.113, -0.104, -0.125, -0.106, -0.119, -0.106, -0.101]  min -0.174 · mean -0.105 · max -0.0433
reward/cup_fling: [-0.000646, -0.0198, -0.0249, -0.0196, -0.0202, -0.0205, -0.0168, -0.022, -0.0201, -0.0172]  min -0.0449 · mean -0.0209 · max -0.000646
reward/grasp_rcv: [8.42e-05, 0.701, 0.569, 0.631, 0.778, 0.743, 0.654, 0.766, 0.814, 0.7]  min 8.42e-05 · mean 0.703 · max 0.826
reward/grasp_src: [0.0241, 1.19, 1.11, 1.12, 1.17, 1.07, 1.09, 1.12, 1.2, 1.11]  min 0.0241 · mean 1.14 · max 1.6
reward/hand_foreign: [-0.0038, -0.0116, -0.00615, -0.0142, -0.0106, -0.00892, -0.00884, -0.00999, -0.00836, -0.0108]  min -0.032 · mean -0.00917 · max -0.00152
reward/hold_rcv: [0, 0.00153, 0.00214, 0.00232, 0.0798, 0.0883, 0.104, 0.104, 0.118, 0.104]  min 0 · mean 0.0627 · max 0.141
reward/lift_rcv: [0, 0.0583, 0.0427, 0.0743, 0.585, 0.675, 0.762, 0.735, 0.859, 0.769]  min 0 · mean 0.494 · max 0.94
reward/lift_src: [0, 1.49, 1.4, 1.25, 1.7, 1.39, 1.3, 1.47, 1.42, 1.41]  min 0 · mean 1.45 · max 1.81
reward/nested: [0, 0, -0.00391, -0.00586, -0.00586, 0, -0.00391, -0.00195, 0, -0.00586]  min -0.0215 · mean -0.00418 · max 0
reward/over_lift_src: [0, -0.0281, -0.0308, -0.0148, -0.023, -0.0116, -0.013, -0.0165, -0.0136, -0.015]  min -0.0609 · mean -0.018 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean 0.000148 · max 0.0293
reward/pour_tilt: [0, 0.00223, 0.000639, 0.000331, 0.000253, 0.000485, 0.000566, 0.00869, 0.0136, 0.00147]  min 0 · mean 0.00199 · max 0.0136
reward/rcv_home: [0, -0.175, -0.185, -0.148, -0.126, -0.119, -0.0849, -0.0904, -0.0918, -0.101]  min -0.334 · mean -0.124 · max 0
reward/rcv_progress: [0.211, 0.63, 0.597, 0.628, 0.652, 0.628, 0.606, 0.657, 0.678, 0.619]  min 0.211 · mean 0.639 · max 0.765
reward/rcv_still: [0, -0.265, -0.228, -0.244, -0.286, -0.28, -0.284, -0.313, -0.329, -0.281]  min -0.363 · mean -0.283 · max 0
reward/reach_rcv: [0.751, 0.845, 0.848, 0.866, 0.814, 0.828, 0.818, 0.824, 0.824, 0.803]  min 0.751 · mean 0.835 · max 0.951
reward/reach_src: [0.6, 0.735, 0.698, 0.693, 0.708, 0.703, 0.69, 0.722, 0.726, 0.707]  min 0.6 · mean 0.707 · max 0.798
reward/rush_near_cup: [-0.236, -0.0623, -0.0669, -0.0672, -0.0641, -0.0681, -0.0678, -0.0623, -0.0616, -0.0699]  min -0.236 · mean -0.0649 · max -0.0411
reward/spill_delta: [-0.00293, -0.0234, -0.0234, -0.00586, -0.0205, -0.0264, -0.0176, -0.0381, -0.0703, -0.0146]  min -0.126 · mean -0.0286 · max 0.041
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_lead: [0, 0.272, 0.243, 0.318, 0.464, 0.484, 0.423, 0.556, 0.563, 0.431]  min 0 · mean 0.399 · max 0.596
reward/total: [1.93, 6.52, 5.96, 6.07, 8, 7.84, 8.33, 8.67, 9.39, 8.56]  min 1.93 · mean 7.68 · max 9.84
rewards/step: [75.2, 3.15e+03, 2.94e+03, 2.92e+03, 3.93e+03, 3.99e+03, 3.92e+03, 4.11e+03, 4.57e+03, 4.7e+03]  min 75.2 · mean 3.69e+03 · max 5.42e+03
task/aim_dist: [0.318, 0.663, 0.898, 0.521, 0.337, 0.312, 0.391, 0.425, 0.435, 0.572]  min 0.241 · mean 0.622 · max 3.75
task/cup_collision_rate: [0, 0.000977, 0.00488, 0.00391, 0.00195, 0.000977, 0, 0.000977, 0, 0.000977]  min 0 · mean 0.00287 · max 0.0176
task/cups_center_dist: [0.319, 0.665, 0.902, 0.524, 0.353, 0.329, 0.402, 0.44, 0.449, 0.581]  min 0.249 · mean 0.631 · max 3.75
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0, 0.00195, 0.00293, 0.00293, 0, 0.00195, 0.000977, 0, 0.00293]  min 0 · mean 0.00209 · max 0.0107
task/rcv_cup_lift: [4.48e-05, 0.194, 0.18, 0.19, 0.056, 0.043, 0.0445, 0.0416, 0.0604, 0.0472]  min 4.48e-05 · mean 0.113 · max 1.74
task/rcv_grasped: [0, 0.523, 0.485, 0.514, 0.545, 0.535, 0.537, 0.599, 0.64, 0.556]  min 0 · mean 0.553 · max 0.652
task/rcv_hand_foreign_rate: [0.0117, 0.0352, 0.0176, 0.0459, 0.0332, 0.0254, 0.0264, 0.0352, 0.0225, 0.0322]  min 0 · mean 0.0282 · max 0.0977
task/src_cup_lift: [0.000511, 0.287, 0.132, 0.0662, 0.126, 0.089, 0.112, 0.14, 0.143, 0.0865]  min 0.000511 · mean 0.182 · max 0.657
task/src_grasped: [0, 0.604, 0.565, 0.577, 0.604, 0.564, 0.563, 0.59, 0.62, 0.566]  min 0 · mean 0.587 · max 0.821
task/src_hand_foreign_rate: [0, 0.000977, 0.00195, 0, 0.00293, 0.00293, 0.000977, 0, 0.00195, 0.000977]  min 0 · mean 0.00161 · max 0.00684
task/src_tilt_deg: [1.79, 14.9, 14.6, 16.9, 25, 27.4, 21.2, 28, 27.5, 20.5]  min 1.79 · mean 20.8 · max 28.7
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Operator notes for round 7 (iter_06 result). Every number is measured. Hypotheses are marked as such. Env unchanged: hand-cup friction 2.0, contact-freeze threshold 1.0 N, grasp flag = thumb > 1.0 N AND another finger > 1.0 N.

## What iter_06 fixed (run t2r_rh2_i06, 1024 envs, warm from i05 epoch-550, 800 epochs)
Every stage before the pour now works, and all of it held to the end of the run:
- task/cups_center_dist 0.875 -> 0.358 m at the end (50-epoch means fell 0.749 -> 0.524). The cups reached their 0.32 m spawn separation. reward/converge, the new term on the cup-cup xy distance, went 0 -> 1.11 and reward/align_mouths 0.201 -> 0.848.
- task/src_cup_lift 0.35 -> 0.106 m, inside the 0.15 m target. The new band plus over_lift ramp held it there for 800 epochs; reward/over_lift_src stayed near -0.01, i.e. the policy sits inside the band rather than being dragged back into it.
- task/rcv_cup_lift 0.221 -> 0.054 m, inside its 0.10 m band (50-epoch means 0.049). reward/hold_rcv 0 -> 0.112 and reward/lift_rcv 0.024 -> 0.878, so the new receiver income is being collected.
- task/src_grasped 0.57 -> 0.59 and task/rcv_grasped 0.435 -> 0.60, both the best of any run on this track. reward/grasp_rcv recovered 0.43 -> 0.748.
- rewards/iter 2986 -> 4417.
Note on a wrong operator call during this run: at epoch 217 the receiver was stuck at 0.189 m and hold_rcv paid 0.0018, and the operator concluded the incentive could not reach it. That was wrong - by epoch 342 the receiver had come down to 0.057 m on its own. It was a 130-epoch learning delay, not a dead incentive.

## What iter_06 did not fix
reward/pour_tilt peaked at a 50-epoch mean of 0.0070 and ended at 0.0016; task/episode_success and bead/in_target were 0.000 at every epoch; reward/pour_delta was 0 in the last 12 epochs (all-time max 0.0293).
task/src_tilt_deg 50-epoch means: 23.3, 22.8, 26.2, 24.8, 22.2, 26.7, 24.9, 24.4. The tilt has not left the 21-27 deg band for 380 epochs, and reward/tilt_lead fell 0.508 -> 0.450 over the same stretch.
This time the distance excuse is gone: the cups are at 0.358 m, both lift bands hold, and both grasps are at their best. The tilt is what is stuck.

## A LEARNING-DYNAMICS FAULT THAT THE REWARD CANNOT FIX (measured this round, applies to every run on this track)
The policy's action noise has been growing without bound for about 3000 epochs across i04, i05 and i06, and it is large enough to corrupt the very metrics above.
- losses/entropy rose monotonically 19.5 -> 28.1 across i04 -> i05 -> i06 and never fell once.
- The agent config has `fixed_sigma: True`, so sigma is a state-independent learned parameter. For a diagonal Gaussian, entropy = sum(ln sigma_i) + 24 * 1.4189 = sum(ln sigma_i) + 34.05. Computing that from the i06 checkpoints gives 26.84 at epoch 50 and 28.17 at epoch 750, against logged values of 26.94 and 28.06. The logged entropy IS the policy sigma; no environment randomness is involved.
- Per-dimension sigma from the i06 checkpoints (the action space is Box(-1, 1, (24,))), epoch 50 -> epoch 750:
  - arm / palm 6-DoF, both arms: 0.19-0.76 -> 0.19-0.77 (stable and tight)
  - four-finger flexion, both hands: 0.90-1.26 -> 0.94-1.31
  - THUMB ABDUCTION: source 1.66 -> 1.97, receiver 1.69 -> 2.13
  A sigma of 2.13 on a [-1, 1] box puts most of the sampled mass outside the box, where it clips. That channel is effectively a random number every step, and it is still growing.
- The agent config already documents this exact failure at a higher setting: "08.21 fab_test8: 0.002 -> 0.0005. 0.002 was this repo's best MLP value, and after ep2100 the policy sigma inflated 0.76 -> 1.30 and the total reward regressed 3017 -> 2045." The present value 0.0005 is still inflating it, to 2.13.
- Likely mechanism (HYPOTHESIS, not verified): the hand controller freezes a finger once one of its links touches its own cup, so those action dimensions have no effect in exactly the states where the grasp matters. With no gradient either way, the entropy bonus inflates them for free.
- Consequence for reading every metric in this file: the deterministic play trace of i04 grasped with both hands in 4 of 4 envs while the training curve for the same policy read task/src_grasped 0.52. The plateaus reported from training curves are measurements of a policy that is being drowned in its own action noise. Tilting the source cup past 45 deg requires holding a grasp while the wrist rotates, and the thumb abduction command is random at every step.
- Learning rate and KL, for completeness: info/kl sat at 0.017-0.020 for the whole run, above the configured kl_threshold 0.013, but `lr_schedule: linear` selects LinearScheduler, which decays by epoch and ignores KL (verified in the rl_games source). The threshold is inert. info/last_lr was 2.79e-4 of 3e-4 at epoch 745, only 7 % decayed against a 10000-epoch horizon. An adaptive schedule would cut only above KL 0.026 and raise only below 0.0065, so it would sit in its dead band here; the learning rate is not the problem.

## Operator changes to the agent config for the iter_07 run (not part of the reward)
These are applied at launch, alongside the new reward. They are listed so the reward is not written to compensate for them.
- `entropy_coef` 0.0005 -> 0.0, to stop the free inflation of sigma.
- The run starts warm from the i06 epoch-750 checkpoint with the policy sigma reset to 0.5 for all 24 dimensions (train.py `--sigma`, which the stock rl_games in use applies on restore). This pulls the thumb channels down from 2.13 and costs the arm channels a little, which are at 0.19-0.77.
- `lr_schedule` is left at linear for the reason given above.

## What the operator wants from iter_07
1. The source cup must tilt past 45 deg. Measured: it has sat at 21-27 deg for 380 epochs with the cups already converged at 0.358 m, and `tilt_lead` saturates at 45 deg by construction (TILT_PRE), so above that angle only `pour_tilt` has any slope.
2. `pour_tilt` is gated by `pour_gate`, a Gaussian on d_align with sigma 0.15 m. At the measured working distance that gate is small, so the only term that pays past 45 deg is nearly switched off exactly where the policy lives. Size it on the distance the run actually holds, or make the gate a consequence of convergence rather than a separate hard gate. The regenerator of iter_06 flagged this in advance as the next lever.
3. Keep everything that now works and do not re-tune it: the source lift band and over_lift ramp, `converge`, `carry_src`, `align_mouths`, the receiver band with `hold_rcv` and `rcv_disc`, the receiver anti-drag penalties, and the grasp formulation for both hands.
4. Assume the action-noise fault above is being fixed at the agent level. Do not add reward terms that try to suppress action noise.

## Launch plan
iter_07 starts warm from the i06 epoch-750 checkpoint with sigma reset to 0.5 and entropy_coef 0, 1024 envs, same env physics.
