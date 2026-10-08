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
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted)."""
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
    # gap is zero only at the measured contact radius (tip frame 26-28 mm, cup outer 28.5 mm);
    # the old +10 mm pad made 33-37 mm count as "on the wall" and killed the gradient
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
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok, lift_target):
    """Unweighted per-arm terms, each in [0, 1]. pose_ok (N,) = cup attitude is acceptable for this stage."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: long-range funnel term + per-side fingertip-on-wall term.
    #    Each side is scored on its own (sum form) so the thumb keeps its full gradient while the finger
    #    side is already done, and measured force completes a side: reach saturates only with contact
    #    force on BOTH sides (the parked pose of i01 had thumb 0.00 N and reach 0.97).
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: balanced two-sided pinch. Before the flag, finger-first touch earns 0.15 and any thumb force
    #    added to it can only raise the value (0.5 N -> +0.19); thumb-only still earns nothing.
    #    Flag boundary (1 N / 1 N): pre_grasp 0.355 -> flagged 0.75, a step up, never a valley.
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

    # closure shaping: measured grasps close to 0.68-0.98, the parked hand sat at 0.19 with the closure
    # channels driven open; pays only once the pinch centre is at the grasp pose (contact freeze makes
    # closing there harmless), saturates at 0.7 so over-squeezing is not rewarded
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift: measured cup elevation only, clamped, and only while really held (explosion-proof)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)                # 5 mm dead band: tipping on the rim is not lifting
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * torch.exp(-(geo["d_grip"] / 0.08) ** 2) * calm
    lift_prog = 0.5 * torch.tanh(h_pos / 0.02) + 0.5 * torch.clamp(h_pos / lift_target, max=1.0)
    lift = held * pose_ok * lift_prog
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)            # "held and at least 5 cm up"

    # penalties (unweighted, bounded)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)       # 1 N dead band: a slightly stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)             # finger-only contact: adding thumb force lowers this
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "lift": lift, "up_w": up_w, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling,
            "held": held, "far": torch.tanh(4.0 * geo["d_funnel"])}


def _pour_geometry(ctx, outer_r):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    align_gate = torch.exp(-(d_align / 0.06) ** 2)                       # wide: tilting is allowed here
    pour_gate = torch.exp(-(d_align / max(ctx.cup_radius, 0.015)) ** 2)  # narrow: beads would land inside
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Stage ladder: reach < grasp < lift per arm, then carry < align < tilt. New in iter_04: the source
    # lift pay is scaled by receiver progress, and the receiver pays for being away from its cup.
    W_REACH_SRC, W_GRASP, W_LIFT = 1.0, 2.0, 3.0   # source formulation and weights unchanged (grasp_src 1.61, lift_src 2.09)
    W_REACH_RCV = 1.5     # near 0.7 vs away 0.2 measured: staying now earns 0.75 more from reach alone (was 0.5)
    W_CLOSE_SRC = 0.3
    W_CLOSE_RCV = 0.6     # close_rcv peaked at 0.10 of 0.3 with the tip gap stuck at 66-77 mm; doubled
    W_AWAY_RCV = 0.8      # x tanh(4 d): ~0.16 at the grasp pose, ~0.66 at 0.3 m -> leaving costs a further 0.5
    LIFT_FLOOR = 0.15     # source lift pay with the receiver away: 2.09 -> ~0.31, about the old 0.3 net loss of leaving
    PROG_REACH, PROG_GRASP = 0.35, 0.65   # parked-near receiver unlocks 0.45 of lift_src, a grasping one all of it
    REACH_AWAY, REACH_NEAR = 0.2, 0.7     # measured reach_rcv after the retreat / while near the cup
    W_CARRY = 2.0         # new stage between lift (3) and align (4): source mouth toward the receiver
    W_ALIGN, W_TILT = 4.0, 5.0
    W_POUR = 150.0        # x d_in_target: 7.5 per bead with 20 beads
    W_SPILL = 60.0
    W_SUCCESS = 20.0
    W_ALIVE = 1.0
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25  # rcv halved: cup_disturb_rcv was -0.13 against grasp_rcv 0.01
    W_TILT_PEN = 0.3                      # knock-over protection kept for both cups (tips at 0.3-0.4 N one-sided)
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05    # rush was the largest near-cup cost (-0.23 summed at e2000)
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05
    CUP_OUTER_R = 0.0285
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35
    TILT_FULL = math.radians(100.0)
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.15, 0.08

    dtype = ctx.src_cup_tilt.dtype
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R)
    upright_src = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = align_gate + (1.0 - align_gate) * upright_src
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                          ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up,
                          ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_src, LIFT_TARGET_SRC)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                          ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                          ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_rcv, LIFT_TARGET_RCV)

    # receiver progress in [0, 1]: 0 at the retreated pose, 0.35 parked at the cup, 1 with a flagged grasp
    # (grasp term is >= 0.4 once flagged with the thumb in its band)
    prog_reach = torch.clamp((rcv["reach"] - REACH_AWAY) / (REACH_NEAR - REACH_AWAY), 0.0, 1.0)
    prog_grasp = torch.clamp(rcv["grasp"] / 0.4, max=1.0)
    rcv_prog = PROG_REACH * prog_reach + PROG_GRASP * torch.maximum(prog_grasp, rcv["up_w"])
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog

    nested = ctx.cups_nested.to(dtype)
    # carry: needs the source held and up and the receiver merely HELD (not yet lifted), so the gradient
    # exists at the measured 0.34 m: linear part keeps a constant slope out to 0.45 m
    carry_w = src["up_w"] * rcv["held"] * pose_ok_rcv * (1.0 - nested)
    carry = carry_w * (0.5 * (1.0 - torch.tanh(2.5 * d_align)) + 0.5 * torch.clamp(1.0 - d_align / 0.45, 0.0, 1.0))
    both_up = src["up_w"] * rcv["up_w"] * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(4.0 * d_align)) + 0.5 * torch.exp(-25.0 * d_align))
    pour_tilt = both_up * pour_gate * torch.clamp(ctx.src_cup_tilt / TILT_FULL, max=1.0)

    tilt_pen_src = (1.0 - align_gate) * torch.tanh(ctx.src_cup_tilt / UPRIGHT_TOL)
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
        "lift_src": W_LIFT * lift_gate * src["lift"],
        "lift_rcv": W_LIFT * rcv["lift"],
        "carry_src": W_CARRY * carry,
        "align_mouths": W_ALIGN * align,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE_SRC * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE_RCV * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
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
bead/in_target: [0, 0, 0, 4.88e-05, 0, 4.88e-05, 4.88e-05, 0, 0, 0]  min 0 · mean 2.06e-05 · max 0.000244
bead/spill: [0, 0.00288, 0.00552, 0.0137, 0.0159, 0.0125, 0.0153, 0.0113, 0.0251, 0.0208]  min 0 · mean 0.0144 · max 0.0331
done/drop: [0, 0, 0, 0.00195, 0, 0.00195, 0, 0.000977, 0.000977, 0.000977]  min 0 · mean 0.000764 · max 0.00586
done/mimic_runaway: [0, 0, 0.000977, 0.000977, 0, 0, 0.00195, 0.000977, 0.000977, 0.000977]  min 0 · mean 0.000733 · max 0.00586
episode_lengths/step: [89.2, 815, 725, 344, 487, 430, 375, 419, 424, 438]  min 89.2 · mean 503 · max 873
reward/action_rate: [-0.0332, -0.0349, -0.0354, -0.0349, -0.0346, -0.0357, -0.0352, -0.0354, -0.0339, -0.0348]  min -0.038 · mean -0.0349 · max -0.0313
reward/align_mouths: [0, 0, 0, 0, 0.000405, 0.000802, 0.0035, 0.0113, 0.0764, 0.0794]  min 0 · mean 0.0211 · max 0.111
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/away_rcv: [-0.0576, -0.0659, -0.0701, -0.106, -0.114, -0.115, -0.112, -0.122, -0.142, -0.135]  min -0.147 · mean -0.107 · max -0.0366
reward/carry_src: [0, 0, 0, 0.000251, 0.000477, 0.000934, 0.0044, 0.0139, 0.0622, 0.0642]  min 0 · mean 0.018 · max 0.0932
reward/close_rcv: [0.0785, 0.23, 0.227, 0.148, 0.121, 0.112, 0.12, 0.111, 0.089, 0.101]  min 0.0785 · mean 0.142 · max 0.258
reward/close_src: [0.0413, 0.15, 0.128, 0.0821, 0.0705, 0.0593, 0.0631, 0.0728, 0.0676, 0.0673]  min 0.0413 · mean 0.0859 · max 0.163
reward/contact_push: [-0.00739, -0.0589, -0.124, -0.117, -0.123, -0.0945, -0.106, -0.0849, -0.0712, -0.0619]  min -0.154 · mean -0.0915 · max -0.00739
reward/cup_collision: [0, 0, 0, -0.000977, -0.00195, -0.000798, -0.000514, -0.000565, -0.000975, -0.00391]  min -0.0112 · mean -0.000897 · max 0
reward/cup_disturb_rcv: [-0.0445, -0.121, -0.139, -0.15, -0.161, -0.161, -0.145, -0.129, -0.135, -0.127]  min -0.188 · mean -0.141 · max -0.0362
reward/cup_disturb_src: [-0.0618, -0.249, -0.296, -0.277, -0.301, -0.298, -0.295, -0.289, -0.275, -0.258]  min -0.375 · mean -0.287 · max -0.0618
reward/cup_fling: [-0.000591, -0.004, -0.00678, -0.0141, -0.0143, -0.0158, -0.0187, -0.0172, -0.0202, -0.0224]  min -0.0295 · mean -0.0143 · max -0.000169
reward/grasp_rcv: [0.00474, 0.12, 0.118, 0.17, 0.229, 0.29, 0.463, 0.608, 0.655, 0.664]  min 0.00349 · mean 0.376 · max 0.86
reward/grasp_src: [0.00697, 0.564, 0.962, 0.806, 0.914, 0.831, 0.882, 0.87, 0.747, 0.678]  min 0.00697 · mean 0.79 · max 1.16
reward/hand_foreign: [0, -0.00144, -0.000486, -0.000732, -0.00135, -0.00534, -0.00226, -0.0076, -0.00638, -0.0037]  min -0.0115 · mean -0.0027 · max 0
reward/lift_rcv: [0, 0, 0, 0.00288, 0.0946, 0.218, 0.409, 0.622, 0.634, 0.614]  min 0 · mean 0.293 · max 0.932
reward/lift_src: [0, 9.94e-05, 0.000267, 0.00365, 0.00718, 0.0104, 0.0423, 0.12, 0.187, 0.209]  min 0 · mean 0.0675 · max 0.283
reward/nested: [0, 0, 0, -0.00195, -0.00391, -0.00195, -0.00391, -0.00391, -0.00391, -0.00195]  min -0.0195 · mean -0.00253 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean 3.08e-05 · max 0.0146
reward/pour_tilt: [0, 0, 0, 0, 4.07e-14, 4.26e-34, 1.65e-26, 8.23e-38, 1.15e-09, 2.39e-08]  min 0 · mean 6.59e-09 · max 7.68e-07
reward/rcv_progress: [0.279, 0.4, 0.402, 0.388, 0.399, 0.444, 0.524, 0.608, 0.623, 0.636]  min 0.252 · mean 0.5 · max 0.699
reward/reach_rcv: [0.89, 1.16, 1.12, 1.01, 0.985, 1.04, 1.03, 1.04, 0.979, 0.977]  min 0.863 · mean 1.04 · max 1.23
reward/reach_src: [0.587, 0.817, 0.807, 0.709, 0.693, 0.674, 0.681, 0.729, 0.724, 0.723]  min 0.587 · mean 0.735 · max 0.888
reward/rush_near_cup: [-0.224, -0.135, -0.104, -0.0996, -0.0934, -0.0895, -0.0852, -0.0737, -0.0764, -0.0763]  min -0.224 · mean -0.0946 · max -0.0636
reward/spill_delta: [0, 0, -0.00586, -0.00879, -0.0967, -0.0264, 0.00879, -0.0146, -0.0176, 0.00293]  min -0.132 · mean -0.0173 · max 0.0352
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/total: [2.18, 3.33, 3.54, 3.04, 3.11, 3.32, 3.82, 4.33, 4.36, 4.37]  min 2.18 · mean 3.72 · max 4.83
rewards/step: [99.6, 2.68e+03, 2.54e+03, 1.01e+03, 1.54e+03, 1.44e+03, 1.33e+03, 1.82e+03, 1.79e+03, 2.01e+03]  min 99.6 · mean 1.84e+03 · max 2.89e+03
task/aim_dist: [0.319, 0.389, 0.516, 1.16, 1.07, 0.902, 0.386, 0.344, 0.705, 0.324]  min 0.284 · mean 0.68 · max 3.58
task/cup_collision_rate: [0, 0, 0, 0.000977, 0.00195, 0.000977, 0.000977, 0.000977, 0.000977, 0.00391]  min 0 · mean 0.001 · max 0.0117
task/cups_center_dist: [0.32, 0.388, 0.516, 1.16, 1.07, 0.896, 0.383, 0.342, 0.704, 0.323]  min 0.283 · mean 0.678 · max 3.58
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0, 0, 0.000977, 0.00195, 0.000977, 0.00195, 0.00195, 0.00195, 0.000977]  min 0 · mean 0.00127 · max 0.00977
task/rcv_cup_lift: [0.000783, 0.0125, 0.0124, 0.111, 0.138, 0.0731, 0.0575, 0.0557, 0.0787, 0.0587]  min 0.000271 · mean 0.0715 · max 0.482
task/rcv_grasped: [0, 0.0293, 0.0176, 0.0459, 0.127, 0.2, 0.318, 0.445, 0.479, 0.49]  min 0 · mean 0.244 · max 0.587
task/rcv_hand_foreign_rate: [0, 0.00684, 0.000977, 0.000977, 0.00391, 0.0137, 0.00684, 0.0225, 0.0176, 0.0107]  min 0 · mean 0.00741 · max 0.042
task/src_cup_lift: [0.000143, 0.00618, 0.0394, 0.0427, 0.336, 0.00408, 0.00933, 0.00936, 0.16, 0.02]  min -0.0137 · mean 0.0518 · max 0.659
task/src_grasped: [0, 0.156, 0.438, 0.425, 0.479, 0.481, 0.495, 0.538, 0.51, 0.512]  min 0 · mean 0.438 · max 0.586
task/src_hand_foreign_rate: [0, 0, 0, 0.000977, 0, 0.00488, 0, 0.00195, 0.00293, 0.00195]  min 0 · mean 0.00147 · max 0.00781
task/src_tilt_deg: [0.539, 4.22, 3.97, 6.74, 6.45, 7.97, 7.49, 8.18, 10.9, 11]  min 0.539 · mean 7.53 · max 13.9
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
Operator notes for round 5 (iter_04 result). Every number is measured. Hypotheses are marked as such. Env unchanged: hand-cup friction 2.0, contact-freeze threshold 1.0 N, grasp flag = thumb > 1.0 N AND another finger > 1.0 N.

## Outcome of t2r_rh2_i04 (iter_04 reward, 1024 envs, started from the i03 epoch-2000 checkpoint, 1650 epochs)
iter_04 fixed what it was written for: the receiver no longer abandons its cup (task/rcv_near_rate 0.84 at e1583 vs 0.05 in i03), and task/rcv_grasped rose 0.00 -> 0.52. But task/episode_success, reward/pour_tilt and reward/success_bonus were 0.000 at every epoch, and the source lift stalled.

reward/lift_src, 50-epoch means from e1000: 0.047, 0.062, 0.090, 0.126, 0.154, 0.214, 0.196, 0.201, 0.227, 0.228, 0.202. It rose to 0.2 by e1250 and was flat for the next 330 epochs.
Over those same 330 epochs reward/rcv_progress rose 0.641 -> 0.665, so lift_gate = 0.15 + 0.85 * rcv_prog went from 0.70 to 0.72. THE GATE IS NOT THE BLOCKER: it opened while lift_src stayed flat.
At e1574-1583 (10-epoch means): reward/carry_src 0.068, reward/align_mouths 0.079, reward/pour_tilt 0.000, task/src_grasped 0.54, task/rcv_grasped 0.52, reward/grasp_src 0.71, losses/entropy 25.19 (monotonically rising for 1500 epochs), rewards/iter 2176.

## Deterministic play trace of the epoch-1550 checkpoint (play --trace_steps 900, 4 envs)
Both hands grasp in 4 of 4 envs and hold it:
- Source grasped from step 100-200, receiver from step 200, both still grasped at step 800. Source finger forces: thumb 3-5 N, other fingers summed 2.5-5 N. Receiver: thumb 2-4 N, others 3-7 N. Hand closure 0.55-0.65 on both hands. Palm force 0 on both.
- SOURCE LIFT: cup height reaches only 20-76 mm (LIFT_TARGET_SRC is 0.15 m). Two of four envs peak at 55-61 mm around step 500 and sink back to 20 mm by step 800.
- SOURCE TILT: cup_up_z stays 0.87-0.96 the whole episode. There is no pouring tilt at all, in any env, at any step.
- RECEIVER: lifts its cup to 120-190 mm (LIFT_TARGET_RCV is 0.08 m) and drags it sideways 210-270 mm, still growing at step 800. Its cup_up_z drifts 1.00 -> 0.88-0.93.
- Because the receiver cup keeps moving, term grasp_rcv decays 1.6-2.0 -> 0.3-0.6, and term carry_src peaks at 0.31-0.61 near step 400 and decays to 0.09-0.26 by step 800. align_mouths swings 0.35-0.97 without settling.
- No cup ejection in the deterministic trace: the largest per-step cup motion (84-285 mm) occurs at step 897, which is the episode reset.

So the failure is not grasping. It is: the source is lifted only a third of the way and then sinks, the source never tilts, and the receiver drags its cup away so the meeting point never settles.

## Cross-track fact: the same source-lift stall was solved on pour_bi (DG-5F) in iter_18
Read from reward_gen/pour_bi/notes_round18.md and iter_18/compute_reward.py, and its live status.json at epoch 324.
- Their run t2r_i17 (727 epochs) had task/src_cup_lift 0.003 m and reward/lift_src 0.007 while the receiver was lifted and both hands grasped - the same shape of failure.
- Their recorded diagnosis: the early lift was multiplied by several sub-1 gates, and stationary income (wrap, home_rcv) was larger than what lifting the source added.
- After iter_18 their measured values at epoch 324 are task/src_cup_lift 0.187 m, reward/lift_src 2.98, reward/both_lifted 0.65, reward/converge 0.21, reward/raise_src 0.49. The source lift became learnable.
- What iter_18 does structurally (verified in its code):
  - `lift_shape_src = 0.5 * (1 - exp(-h_src / LIFTOFF_H)) + 0.5 * clamp(h_src / LIFT_TARGET_SRC, 0, 1)` with the comment "no product of sub-1 gates". Half the income is a steep lift-off part that pays in the first centimetres, half is linear so going higher keeps paying.
  - `src_up_soft = FLOOR + (1 - FLOOR) * src_up`: the uprightness gate has a floor instead of multiplying the lift to zero.
  - `lift_src = LIFT_SRC_W * held_src * lift_shape_src * src_up_soft * (1 + 0.25 * wc_src)`: the envelope quality is a bonus, not a gate.
  - `rcv_band = exp(-(clamp(h_rcv - RCV_LIFT_FREE, min=0) / RCV_LIFT_SIGMA)**2)`: the receiver is paid to lift only within a band, so over-lifting loses pay.
  - `rcv_still = -still_w * rcv_carry * tanh(clamp(|v_rcv| - RCV_STILL_FREE, min=0) / RCV_STILL_SCALE)`, with still_w increased while the source approaches and while pouring.
  - `home_rcv = HOME_W * home_src_f * rcv_carry * ... * rcv_home * still_f` where `home_src_f = HOME_SRC_FLOOR + (1 - HOME_SRC_FLOOR) * src_carry`: the receiver's stationary income is scaled by whether the source is being carried, so an idle receiver cannot out-earn the source lifting.
  - Staged terms after both cups are up: both_lifted, raise_src, converge, bring_together, aim.
- Limit of this cross-track evidence: at epoch 324 iter_18 still has reward/tilt 0.000, reward/pour_pose 0.000 and reward/aim 0.007. Only its lift and convergence lessons are demonstrated; its pour stage is not.

## What the operator wants from iter_05
1. Make the source lift pay from the first centimetres and keep paying past the target, in the iter_18 shape. No product of several sub-1 gates on the early lift. The receiver-progress gate on lift_src may stay only if its floor is high enough that the source lift can still grow; the measured evidence is that the gate at 0.70-0.72 was not what held lift_src at 0.2.
2. Bound the receiver: pay its lift only inside a band near LIFT_TARGET_RCV (measured overshoot: 120-190 mm against a 0.08 m target) and penalise receiver cup speed, so it stops dragging the cup 210-270 mm. The receiver's stationary income must not exceed what the source earns by lifting.
3. Give the source cup a tilt gradient that can start before a full-height lift. Measured: pour_tilt was 0.000 for the entire run and the source cup never left 0.87-0.96 of upright.
4. Keep the grasp formulation for both hands unchanged - it grasps in 4 of 4 envs deterministically with 2-7 N per side - and keep receiver reach.

## Launch plan
iter_05 starts warm from the i04 epoch-1550 checkpoint (both hands already grasp), 1024 envs, same env physics.
