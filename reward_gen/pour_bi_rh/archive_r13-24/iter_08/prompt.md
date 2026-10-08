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
    iter_07: unchanged since iter_04 — the grasp works (src_grasped 0.59, rcv_grasped 0.60, both best-ever)."""
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
    """Unweighted per-arm terms in [0, 1]. Reach / grasp / close / lift-state and every penalty are
    iter_06 VERBATIM — they are on the keep list (grasp_src 1.11, grasp_rcv 0.70-0.81 measured, both
    lift bands held for 800 epochs)."""
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
    # NOTE: pose_ok = src_up rides on the attitude CORRIDOR. iter_07 widens that corridor to 65 deg once
    # the cup is carried, which is what stops tilting from silently taxing this term (see compute_reward).

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


def _pour_geometry(ctx, outer_r, aim_k, aim_sigma):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre,
    plus the SINGLE smooth aim factor that iter_07 uses both to open the attitude corridor and to pay the
    release tilt. Geometry (lip, clearance, half-weighted height error) is iter_06 verbatim.

    iter_06 used two Gaussians here and both were numerically dead at the distance the run holds
    (measured d_align 0.30 m from cups_center_dist 0.358 m + src lift 0.106 m + rcv lift 0.054 m):
    pour_gate exp(-(0.30/0.15)^2) = 0.018 and align_gate exp(-(0.30/0.12)^2) = 0.002. That is the whole
    reason pour_tilt never left 0.002 and the corridor stayed pinned at 45 deg."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    # 0.33 at the measured 0.30 m, 0.55 at 0.20 m, 0.80 at 0.10 m — a live gradient over the exact
    # interval the run occupies; still only 0.12 at 0.45 m and 0.04 at 0.60 m, so dumping beads from
    # far away is never bought.
    aim_shape = 0.5 * (1.0 - torch.tanh(aim_k * d_align)) + 0.5 * torch.exp(-(d_align / aim_sigma) ** 2)
    return d_align, aim_shape


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt < pour.
    # iter_07 is a focused change to the TILT stage only. Measured: every earlier stage held for 800
    # epochs (cups 0.875 -> 0.358 m, src lift 0.35 -> 0.106 m, rcv lift 0.221 -> 0.054 m, both grasps
    # best-ever) while task/src_tilt_deg sat in 21-27 deg for 380 epochs. At the d_align 0.30 m that this
    # state implies, iter_06's pour_gate was 0.018 and its corridor opener align_gate 0.002, so past 45 deg
    # the only paying term was switched off AND the corridor charged grasp/lift income for the excess.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.80-0.87 measured:
    W_AWAY_RCV = 0.8                       # it keeps the left hand at its own cup)
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (best-ever grasp rates)
    # --- source lift: band + over-lift ramp. KEEP LIST — held 0.106 m for 800 epochs ----------------
    W_LIFT_SRC = 3.0
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.15
    LIFT_FREE_SRC = 0.24
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5       # measured over_lift_src ~ -0.01: the policy sits inside the band, not dragged
    OVER_LIFT_RANGE = 0.20
    LIFT_FLOOR = 0.60
    SRC_UP_FLOOR = 0.5
    LIFT_PINCH_BONUS = 0.2
    # --- receiver lift band — KEEP LIST (rcv lift 0.221 -> 0.054 m, lift_rcv 0.024 -> 0.878) --------
    W_LIFT_RCV = 2.0
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10
    RCV_LIFT_SIGMA = 0.07
    W_HOLD_RCV = 1.2        # measured hold_rcv 0 -> 0.112: the receiver income is being collected
    RCV_DISC_FLOOR = 0.35
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7
    # --- convergence / carry / align — KEEP LIST (cups 0.875 -> 0.358 m, align 0.201 -> 0.848) ------
    W_CONVERGE = 3.0
    CONV_FAR = 1.00
    CONV_NEAR = 0.16
    CONV_SIGMA = 0.25
    W_CARRY, W_ALIGN = 3.0, 5.0
    CARRY_K, CARRY_RANGE = 1.2, 1.00
    ALIGN_K, ALIGN_SIGMA = 2.0, 0.15
    # --- TILT LADDER: three segments, one continuous slope 0 -> 100 deg ----------------------------
    # Slope per degree, evaluated at the measured operating point (cups 0.36 m, d_align 0.30 m):
    #   lead 0.033  ->  rise 0.093  ->  pour 0.054 (and 0.137 once d_align reaches 0.10 m).
    # Nowhere in 0..100 deg is the tilt income flat, which is the whole point of this round.
    W_TILT_LEAD = 2.0       # was 1.5. At the 45 deg cap the ENTIRE tilt stage was worth 1.11/step at the
                            # measured distance — the same as grasp_src (1.11) and less than lift_src (1.41)
    TILT_PRE = math.radians(45.0)    # end of the lead-in segment (unchanged angle)
    W_TILT_RISE = 2.5       # NEW 45 -> 65 deg bridge. Same gates as tilt_lead, so nothing switches at 45 deg
    TILT_SAFE = math.radians(65.0)   # first bead leaves at ~70 deg with a full cup -> 5 deg of margin:
                                     # this whole segment is spill-free at ANY distance
    W_TILT = 8.0            # unchanged weight, now spanning 65 -> 100 deg and paid by aim_shape
    TILT_FULL = math.radians(100.0)
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # Gaussian width on the corridor EXCESS (unchanged)
    TILT_LIFT_H = 0.03      # corridor and tilt_lead fully open at 3 cm — the policy lives at 0.106 m
    LEAD_FLOOR = 0.25       # distance factor of lead+rise is convergence, not a gate (unchanged form)
    AIM_K = 3.0             # far tail: 0.12 at 0.45 m, 0.04 at 0.60 m over the visited aim_dist range
    AIM_SIGMA = 0.30        # = the measured operating d_align. Was an 0.15 m Gaussian (0.018 there) for the
                            # pay and an 0.12 m one (0.002) for the corridor — both numerically dead.
    # --- receiver discipline — KEEP LIST -----------------------------------------------------------
    W_HOME_RCV = 0.8
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5
    STILL_FREE, STILL_SCALE = 0.05, 0.15
    # --- task / penalties (all unchanged) ----------------------------------------------------------
    W_POUR = 150.0          # x d_in_target: 7.5 per bead with 20 beads
    W_SPILL = 60.0          # -3 per bead spilled; beads_left also deletes the whole tilt income
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
    d_align, aim_shape = _pour_geometry(ctx, CUP_OUTER_R, AIM_K, AIM_SIGMA)

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)

    # ---- source attitude CORRIDOR: free to 65 deg once carried, to 100 deg once aimed --------------
    # 0.15 rad on the table -> 65 deg when grasped and 3 cm up -> 100 deg when the lip is over the
    # receiver. The 45 -> 65 widening is bought by the measured spill threshold (first bead at ~70 deg):
    # below 65 deg nothing can leave the cup, so charging for it only taxed grasp_src and lift_src.
    # A toppled cup on the table is still 1.57 rad against 0.15 rad, so src_up ~ 0 exactly as before.
    carry_soft_src = g_src * torch.clamp(h_src / TILT_LIFT_H, max=1.0)
    tilt_free = TILT_FREE_TABLE + (TILT_SAFE - TILT_FREE_TABLE) * carry_soft_src
    tilt_free = tilt_free + (TILT_FULL - TILT_SAFE) * aim_shape   # 77 deg at 0.30 m, 89 at 0.15, 93 at 0.10
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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # 0.60 .. 1.00 (measured rcv_prog 0.62-0.68)

    # ---- source lift: band + linear cost above it (unchanged — measured 0.106 m for 800 epochs) ----
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_lift_band = torch.exp(-(torch.clamp(h_src - LIFT_FREE_SRC, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_lift_band * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])
    # ungated by the grasp flag on purpose: throwing the cup up must not dodge the cost
    over_lift = torch.clamp((h_src - LIFT_FREE_SRC) / OVER_LIFT_RANGE, 0.0, 1.0)

    # ---- receiver lift band (unchanged — measured 0.054 m, inside the 0.10 m band) -----------------
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * rcv_band * (0.5 + 0.5 * upright_rcv)

    # ---- receiver discipline: home, still, and one factor on all convergence income (unchanged) ----
    d_home = rcv["disp"]
    home_near = torch.exp(-(d_home / HOME_SIGMA) ** 2)
    still_f = 1.0 - torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    rcv_disc = RCV_DISC_FLOOR + (1.0 - RCV_DISC_FLOOR) * rcv_band * home_near
    rcv_home_pen = rcv["held"] * torch.tanh(torch.clamp(d_home - HOME_FREE, min=0.0) / HOME_SCALE)
    still_w = W_STILL_RCV + W_STILL_POUR * up_w_src               # stricter while the source is carried
    rcv_still_pen = still_w * rcv["held"] \
        * torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    hold_rcv = W_HOLD_RCV * g_rcv * rcv_band * home_near * still_f * (0.4 + 0.6 * carry_soft_src)

    # ---- convergence (unchanged): the SOURCE closes the gap, on the quantity the success test uses --
    d_cups_xy = (ctx.src_cup_pos[:, :2] - ctx.rcv_cup_pos[:, :2]).norm(dim=-1)
    conv_lin = torch.clamp((CONV_FAR - d_cups_xy) / (CONV_FAR - CONV_NEAR), 0.0, 1.0)
    conv_close = torch.exp(-(torch.clamp(d_cups_xy - CONV_NEAR, min=0.0) / CONV_SIGMA) ** 2)
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close     # 0.65 at the measured 0.358 m, 0.96 at 0.20 m
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.60 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * (1.0 - nested) * conv_shape

    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * (1.0 - nested)
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- TILT: three segments of ONE slope. Only the density changes at 45 deg, never the gates -----
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    # beads_left is the anti-spill interlock: beads on the table delete the whole ladder (3.3-5.2 / step
    # at the measured distance) for the rest of the episode, on top of spill_delta's -3 per bead.
    tilt_hold = carry_soft_src * beads_left * (1.0 - nested)          # "grasped, carried, still full"
    lead_dist = LEAD_FLOOR + (1.0 - LEAD_FLOOR) * conv_shape          # 0.74 at the measured 0.358 m
    lead_frac = torch.clamp(ctx.src_cup_tilt / TILT_PRE, max=1.0)                       # 0 -> 45 deg
    rise_frac = torch.clamp((ctx.src_cup_tilt - TILT_PRE) / (TILT_SAFE - TILT_PRE), 0.0, 1.0)  # 45 -> 65
    pour_frac = torch.clamp((ctx.src_cup_tilt - TILT_SAFE) / (TILT_FULL - TILT_SAFE), 0.0, 1.0)  # 65 -> 100
    tilt_lead = tilt_hold * lead_frac * lead_dist
    # the 45-65 deg bridge: identical gating to tilt_lead (spill-free by geometry, so no distance gate),
    # 2.7x the per-degree density so the ladder steepens exactly where the policy has been stuck
    tilt_rise = tilt_hold * rise_frac * lead_dist
    # the release segment: now a consequence of convergence (aim_shape) instead of a dead 0.15 m Gaussian
    pour_tilt = both_up * aim_shape * beads_left * upright_rcv * (1.0 - nested) * pour_frac

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
        "tilt_rise": W_TILT_RISE * tilt_rise,
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
        "aim_gate": torch.zeros_like(aim_shape),
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    # diagnostics only (added AFTER the sum, not part of the reward): rcv_progress as before, and the
    # aim factor itself so the next round can tell "not tilting" from "gate closed" without guessing.
    comps["rcv_progress"] = rcv_prog
    comps["aim_gate"] = aim_shape
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 9.77e-05, 4.88e-05, 0.000195, 4.88e-05, 0, 0, 9.77e-05, 9.77e-05, 0.000195]  min 0 · mean 9.79e-05 · max 0.000684
bead/spill: [0, 0.0227, 0.0164, 0.0152, 0.00415, 0.0121, 0.0126, 0.0253, 0.0248, 0.0345]  min 0 · mean 0.019 · max 0.0563
done/drop: [0, 0, 0.000977, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000256 · max 0.00391
done/mimic_runaway: [0, 0, 0, 0.000977, 0, 0, 0, 0, 0, 0]  min 0 · mean 0.000209 · max 0.00391
episode_lengths/step: [105, 640, 652, 770, 783, 735, 802, 734, 783, 736]  min 105 · mean 733 · max 870
reward/action_rate: [-0.0313, -0.0244, -0.0276, -0.0275, -0.0285, -0.0271, -0.0258, -0.0226, -0.0225, -0.0217]  min -0.0313 · mean -0.0251 · max -0.0207
reward/aim_gate: [0.291, 0.506, 0.441, 0.559, 0.578, 0.619, 0.652, 0.683, 0.72, 0.73]  min 0.291 · mean 0.616 · max 0.79
reward/align_mouths: [0, 0.882, 0.882, 1.47, 1.7, 1.85, 2.17, 1.92, 2.51, 2.62]  min 0 · mean 1.81 · max 2.82
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/away_rcv: [-0.0762, -0.163, -0.161, -0.166, -0.166, -0.175, -0.174, -0.179, -0.183, -0.182]  min -0.189 · mean -0.172 · max -0.0762
reward/carry_src: [0, 0.942, 0.996, 1.3, 1.53, 1.61, 1.73, 1.62, 1.73, 1.68]  min 0 · mean 1.41 · max 1.89
reward/close_rcv: [0.0737, 0.0706, 0.0751, 0.0737, 0.0749, 0.0762, 0.0712, 0.0649, 0.0703, 0.074]  min 0.0573 · mean 0.0718 · max 0.0854
reward/close_src: [0.0452, 0.0833, 0.0816, 0.0745, 0.0602, 0.0594, 0.0467, 0.0494, 0.0482, 0.0458]  min 0.0391 · mean 0.0607 · max 0.114
reward/contact_push: [-0.0229, -0.0118, -0.00781, -0.00495, -0.0025, -0.00203, -0.00377, -0.00491, -0.00379, -0.00445]  min -0.133 · mean -0.00611 · max -0.000895
reward/converge: [0, 1.42, 1.46, 1.79, 2.07, 2.16, 2.28, 2.12, 2.16, 2.15]  min 0 · mean 1.91 · max 2.34
reward/cup_collision: [0, 0, 0, 0, 0, -0.000957, 0, -0.000655, -0.000785, 0]  min -0.00989 · mean -0.00121 · max 0
reward/cup_disturb_rcv: [-0.0273, -0.111, -0.107, -0.0921, -0.086, -0.0885, -0.0838, -0.0986, -0.0765, -0.0962]  min -0.293 · mean -0.0958 · max -0.0273
reward/cup_disturb_src: [-0.103, -0.0557, -0.0508, -0.0388, -0.029, -0.0321, -0.0279, -0.0399, -0.0259, -0.034]  min -0.335 · mean -0.0399 · max -0.0182
reward/cup_fling: [-0.00103, -0.0138, -0.0108, -0.00968, -0.00821, -0.0112, -0.00702, -0.0112, -0.0105, -0.013]  min -0.0301 · mean -0.0118 · max -0.00103
reward/grasp_rcv: [0.000106, 0.914, 1, 1.07, 1.2, 1.31, 1.27, 1.17, 1.25, 1.25]  min 0.000106 · mean 1.11 · max 1.34
reward/grasp_src: [0.0277, 1.39, 1.31, 1.4, 1.5, 1.46, 1.17, 1.27, 1.51, 1.51]  min 0.0277 · mean 1.39 · max 1.72
reward/hand_foreign: [-0.00901, -0.00325, -0.00432, -0.0031, -0.00309, -0.00399, -0.00193, -0.00277, -0.00265, -0.00488]  min -0.0545 · mean -0.00416 · max -0.00042
reward/hold_rcv: [0, 0.152, 0.204, 0.291, 0.374, 0.392, 0.407, 0.346, 0.424, 0.408]  min 0 · mean 0.314 · max 0.479
reward/lift_rcv: [0, 1.03, 1.2, 1.32, 1.47, 1.33, 1.42, 1.09, 1.41, 1.37]  min 0 · mean 1.25 · max 1.53
reward/lift_src: [0, 2.18, 2.11, 2.4, 2.64, 2.65, 2.69, 2.53, 2.71, 2.69]  min 0 · mean 2.46 · max 2.85
reward/nested: [0, 0, 0, -0.00195, -0.00195, -0.00586, 0, -0.00391, -0.00391, 0]  min -0.0254 · mean -0.00392 · max 0
reward/over_lift_src: [0, -0.0219, -0.00951, -0.0138, -0.00399, -0.00766, -0.00041, -0.0123, -0.0114, -0.0193]  min -0.0597 · mean -0.0139 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean 0.0002 · max 0.0146
reward/pour_tilt: [0, 0, 0.000104, 0.000433, 0, 0.00553, 0.0439, 0.113, 0.156, 0.141]  min 0 · mean 0.0577 · max 0.297
reward/rcv_home: [0, -0.144, -0.0756, -0.0549, -0.0202, -0.0385, -0.0129, -0.0225, -0.0152, -0.0126]  min -0.273 · mean -0.0449 · max 0
reward/rcv_progress: [0.248, 0.729, 0.747, 0.753, 0.778, 0.776, 0.76, 0.732, 0.741, 0.748]  min 0.248 · mean 0.738 · max 0.823
reward/rcv_still: [0, -0.357, -0.347, -0.336, -0.34, -0.334, -0.34, -0.344, -0.289, -0.3]  min -0.44 · mean -0.328 · max 0
reward/reach_rcv: [0.833, 0.828, 0.835, 0.815, 0.809, 0.802, 0.789, 0.772, 0.766, 0.779]  min 0.747 · mean 0.799 · max 0.879
reward/reach_src: [0.602, 0.697, 0.696, 0.674, 0.613, 0.609, 0.597, 0.589, 0.572, 0.557]  min 0.533 · mean 0.619 · max 0.749
reward/rush_near_cup: [-0.239, -0.0411, -0.0462, -0.0394, -0.0307, -0.0251, -0.0241, -0.0315, -0.0271, -0.0292]  min -0.239 · mean -0.0348 · max -0.0192
reward/spill_delta: [0, -0.0176, -0.0117, -0.0205, -0.0146, 6.98e-10, -0.00293, 0.00293, -0.00586, -0.0146]  min -0.0938 · mean -0.00977 · max 0.0439
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_lead: [0, 0.4, 0.519, 0.768, 1.13, 1.54, 1.61, 1.5, 1.58, 1.55]  min 0 · mean 1.16 · max 1.66
reward/tilt_rise: [0, 0.000584, 0.00523, 0.00327, 0.0112, 1.02, 1.53, 1.54, 1.73, 1.67]  min 0 · mean 0.827 · max 1.88
reward/total: [2.07, 11, 11.5, 13.6, 15.4, 17.1, 18.1, 16.9, 18.9, 18.7]  min 2.07 · mean 15.4 · max 20.2
rewards/step: [126, 6.94e+03, 7.38e+03, 1.08e+04, 1.17e+04, 1.21e+04, 1.42e+04, 1.25e+04, 1.49e+04, 1.36e+04]  min 126 · mean 1.15e+04 · max 1.67e+04
task/aim_dist: [0.318, 0.431, 0.308, 0.276, 0.229, 0.2, 0.171, 0.219, 0.153, 0.17]  min 0.148 · mean 0.31 · max 1.66
task/cup_collision_rate: [0, 0, 0, 0, 0, 0.000977, 0, 0.00195, 0.000977, 0]  min 0 · mean 0.00143 · max 0.00977
task/cups_center_dist: [0.32, 0.434, 0.311, 0.286, 0.243, 0.207, 0.181, 0.229, 0.169, 0.189]  min 0.167 · mean 0.321 · max 1.67
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0, 0, 0.000977, 0.000977, 0.00293, 0, 0.00195, 0.00195, 0]  min 0 · mean 0.00196 · max 0.0127
task/rcv_cup_lift: [7.84e-05, 0.0506, 0.0795, 0.126, 0.0873, 0.0578, 0.0672, 0.0436, 0.0797, 0.0816]  min 7.84e-05 · mean 0.0728 · max 0.735
task/rcv_grasped: [0, 0.702, 0.729, 0.761, 0.81, 0.809, 0.806, 0.767, 0.768, 0.772]  min 0 · mean 0.746 · max 0.839
task/rcv_hand_foreign_rate: [0.0225, 0.00977, 0.0146, 0.00879, 0.00684, 0.00781, 0.00293, 0.00586, 0.00293, 0.00684]  min 0 · mean 0.00963 · max 0.194
task/src_cup_lift: [0.000447, 0.157, 0.132, 0.154, 0.143, 0.139, 0.132, 0.144, 0.161, 0.178]  min 0.000447 · mean 0.181 · max 0.791
task/src_grasped: [0, 0.722, 0.684, 0.745, 0.809, 0.833, 0.848, 0.8, 0.829, 0.817]  min 0 · mean 0.776 · max 0.873
task/src_hand_foreign_rate: [0, 0.000977, 0.000977, 0.000977, 0.00391, 0.00391, 0.00293, 0.00488, 0.00684, 0.0107]  min 0 · mean 0.00358 · max 0.0186
task/src_tilt_deg: [1.57, 14.1, 16.9, 21.3, 28.6, 46.5, 51.6, 51.9, 54.4, 54]  min 1.57 · mean 37.3 · max 56.8
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 8 operator notes — t2r_rh2_i07 (iter_07) 종료 관찰

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 이번 라운드의 결론 한 줄

붓기 전 단계(파지·들기·모으기·조준)는 전부 성립했다. 막힌 것은 **기울임의 방향**이다.
보상이 방향 없는 크기 `ctx.src_cup_tilt` 만 쓰고, env 가 제공하는 방향 필드
(`src_tilt_toward_rcv`, `pour_dir_xy`, `src_pour_lip_pos`, `premature_tilt`)를 한 번도 쓰지 않았다.

## 1. 사용자 직접 관찰 (ep1950 영상, 증거 우선순위 2)

- 소스 컵의 **회전 방향이 잘못됐다**.
- 컵 입구가 **몸쪽을 향해야 하는데 world +x 를 바라보고 있다**.

`context.py` 정의상 `pour_dir_xy` 의 x>0 은 몸 바깥이고 로봇 몸통은 −x 다. 즉 관찰된 자세는
붓는 방향이 리시버 반대쪽을 향한 상태다.

## 2. 보상 코드 사실 (직접 확인)

- `reward_gen/pour_bi_rh/iter_00 ~ iter_07` 8개 전부에서
  `src_tilt_toward_rcv` · `pour_dir_xy` · `premature_tilt` · `src_pour_lip_pos` 참조 횟수 = **0**.
- 같은 저장소의 성공한 형제 트랙 `reward_gen/pour_bi/iter_16~19` 는 같은 필드를 **4~5회** 쓴다.
  (`pour_bi` i16 ep5600 은 최대 난이도 결정적 재생 성공 92 %·in_target 91 % 로 영상 확인됨)
- iter_07 은 `tilt_lead` · `tilt_rise` · `pour_tilt` · 자세 통로 전부를 `ctx.src_cup_tilt` 로 만든다.
  이 값은 `context.py` 주석대로 **직립 대비 기울기 크기**(0=직립, π/2=수평)이며 **방향이 없다**.
  어느 쪽으로 기울여도 같은 보상이 나온다.
- iter_07 은 붓는 쪽 림 점을 `ctx.src_pour_lip_pos` 대신 직접 만든다(L124-127):
  `lip = src_cup_mouth_pos + outer_r * (src_cup_up * src_cup_up[:,2:3] - ez)`.
  이것은 림에서 **기하학적으로 가장 낮은 점**이라 기울인 방향이 어디든 정의된다. 방향 제약이 아니다.

## 3. env 가 이미 주는 방향 필드 (context.py 정의, 그대로 인용)

- `src_tilt_toward_rcv` (N,): 리시버 쪽으로 기운 **부호 있는** 각도 θ [rad].
  (d̂, z) 수직면 성분만 본다. **옆으로 기운 성분은 0, 반대로 기울면 음수.**
- `pour_dir_xy` (N,2): 붓는 방향 d̂ 의 수평 단위벡터. **x>0 = 몸 바깥, 로봇 몸통은 −x.
  x > 0.3 이면 성공 무효.**
- `src_pour_lip_pos` (N,3): 비드가 실제로 나가는 림 점. 컵이 직립이어도 정의된다.
- `premature_tilt` (N,) bool: **에피소드 래치**. 잡은 소스 컵의 `src_pour_lip_pos` 가 리시버 입구 중심에서
  xy 8.1 cm 보다 먼 상태에서 `src_cup_tilt` 가 `premature_tilt_limit` 을 넘은 적이 있으면 참.
  **리셋 전까지 유지되고 성공이 영구 무효**가 된다.
- `premature_tilt_limit` (N,): 채움별 첫 유출각 − 20°. **가득 채운 컵에서 52°**, 채움 0.5 에서 69°.
- `cup_hit`, `rcv_side_margin` (< 0.03 이면 성공 무효), `cups_nested` 도 같은 성격의 무효 조건이다.

## 4. 이번 런 수치 (t2r_rh2_i07-r3, e1978 에서 종료, 12.08 h, Traceback 0)

마지막 10 epoch 평균:

- `task/src_tilt_deg` **54.8°**. 50-epoch 평균이 400 epoch 동안 54 ± 0.7 로 평평.
  마지막 200/400 epoch 선형 회귀 기울기 +0.71 / +0.34 deg per 100 epoch.
- `task/episode_success` · `task/success_now` · `reward/success_bonus` · `reward/pour_delta`
  = **런 전 구간 0.0**.
- `bead/in_target` 0.0002 (최대 0.0007), `bead/crossed` 0.0002 (최대 0.0021),
  `bead/spill` 0.0319 (최대 0.0563), `bead/in_source` 0.963.
- `task/src_grasped` 0.828 · `task/rcv_grasped` 0.762 (둘 다 런 최고 수준).
  `task/src_closure` 0.437 · `task/rcv_closure` 0.491.
- `reward/aim_gate` 0.739, `task/aim_dist` 정상 구간 0.17~0.25 m (최소 0.149).
- `reward/tilt_lead` 1.565 · `tilt_rise` 1.732 · `pour_tilt` 0.202.
- `task/nested_rate` 0.0015 · `task/cup_collision_rate` 0.0018 · `done/drop` 0.0006 — 전부 낮다.
- `losses/entropy` 17.62 로 단조 하락 지속(entropy_coef 0 의 효과, 커밋 3b02f403).

## 5. 결정론 재생 (ep1950, 4 env, step 600 스냅샷 1개)

파지 1.00/1.00 · 들기 0.211 m · **src_cup_tilt 65.4°** · aim_dist 0.128 m ·
**in_source 1.000 · in_target 0.000 · spill 0.000 · success 0.000**.

컵을 65° 까지 기울이고 입구를 0.128 m 까지 붙였는데 비드가 **한 알도 나가지 않았다.**
(주의: 에피소드 최대각이 아니라 step 600 한 시점 값이다.)

## 6. 산술적으로 확인된 것

- iter_07 의 `pour_tilt` 는 65° 부터 시작하는데 런 평균은 54.8° 라 평균에서 `pour_frac` = 0 이다.
  그런데 로그값은 0.202 다. 따라서 이 값은 **65° 를 넘긴 소수 env 몫**이고 분포가 넓다.
- `tilt_rise` 는 65° 에서 `rise_frac` 이 1.0 으로 포화해 경사가 0 이 되고,
  `pour_tilt` 는 65→100° 를 35도에 펴 놔서 65.4° 에서 `pour_frac` 이 0.011 뿐이다.
  정책이 멈춘 각도가 두 항의 교대 지점과 겹친다.
- 런 평균 54.8° 와 재생 65.4° 는 **가득 찬 컵의 `premature_tilt_limit` 52° 를 넘는다.**

## 7. [가설] premature_tilt 래치

`bead/in_source` 가 0.96 으로 컵이 가득 차 있으므로 한계는 52° 다. 평균 기울임이 54.8° 이고
붓는 방향이 리시버 반대쪽(사용자 관찰)이면, 림 점이 리시버 입구에서 8.1 cm 안에 들어올 수 없으므로
`premature_tilt` 가 래치되어 그 에피소드의 성공이 영구 무효가 된다.
이것이 사실이면 `episode_success` 가 런 내내 정확히 0 인 것과 맞는다.

**미검증이다.** env 는 `premature_tilt` · `pour_dir_xy` 를 extras 로 기록하지 않아
tfevents 로는 래치율을 볼 수 없다(기록되는 task/ 키 31개를 전부 확인했다).

## 8. 다음 보상이 반드시 해야 할 것

1. 기울임 단계 전체(`tilt_lead` · `tilt_rise` · `pour_tilt` · 자세 통로)를 방향 없는
   `ctx.src_cup_tilt` 대신 **부호 있는 `ctx.src_tilt_toward_rcv`** 기준으로 다시 세운다.
   `ctx.src_cup_tilt` 는 과기울임 **벌점**에만 남긴다.
2. `ctx.pour_dir_xy[:,0]` 로 방향 게이트를 만든다. env 의 무효 기준이 x > 0.3 이므로
   그보다 앞에서 0 이 되어야 한다.
3. `ctx.premature_tilt` 에 명시적 벌점을 준다. 래치는 되돌릴 수 없으므로 진입 자체를 막아야 한다.
4. 기울임 허용각을 `ctx.premature_tilt_limit` 에 묶고, **림 점이 리시버 입구에 가까워졌을 때만**
   그 위로 열어 준다. 거리 기준은 손으로 만든 lip 이 아니라 `ctx.src_pour_lip_pos` 로 잰다.
5. 형제 트랙 `reward_gen/pour_bi/iter_18/compute_reward.py` 의 L216-217(dir_ok),
   L232-237(lip 기하·approach_gate·safe), L240-255(theta·theta_eff·allowed 통로),
   L402(premature_latch) 가 위 네 가지를 이미 구현한 참조 구현이다. 로봇이 다르니
   그대로 복사하지 말고 구조만 가져온다.

## 9. 건드리지 말 것

파지·들기·모으기·조준은 이번 런에서 성립했다(§4). 해당 항의 가중·형태를 크게 바꾸면 회귀한다.
`entropy_coef` 는 0 으로 두고 sigma 는 손 12축만 다루는 기존 방침을 유지한다.
