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


def _smoothstep(x):
    """3t^2 - 2t^3 on a pre-normalised argument; flat outside [0, 1]."""
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    iter_09: VERBATIM since iter_04 — the grasp works (src_grasped 0.798, rcv_grasped 0.770)."""
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
    iter_06 VERBATIM — on the keep list (grasp_src 1.49, grasp_rcv 1.22 measured over 374 epochs,
    both lift bands held). iter_09 changes NOTHING in this function."""
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

    # closure shaping (unchanged): measured closure 0.44 / 0.49 on the two hands while holding
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


def _pour_geometry(ctx, lip_r, mouth_z, clearance, sep_min):
    """Everything about the pour measured on the DIRECTED spout (iter_08), plus the iter_09 LIP DROP.

      d_hat    : horizontal unit vector src cup origin -> receiver mouth. Independent of the cup's
                 attitude, so it is defined for an upright cup and never flips while rotating.
      theta    : SIGNED tilt toward the receiver = atan2(up . d_hat, up_z). A purely sideways tilt
                 gives 0, tilting away gives a negative angle.
      lip      : the rim point the beads leave from = mouth centre + r*(cos(theta)*d_hat
                 - sin(theta)*z). Drops as the cup turns toward the receiver, RISES when it turns away.
      lip_drop : ★iter_09 — how far that lip sits BELOW where it would be with the same cup origin and
                 an upright cup:  r*sin(theta) (the rim point walking down the circle)
                 + mouth_z*(1 - cos(theta)) (the whole rim plane dropping as it rotates about the
                 origin). 0.067 m at 75 deg, 0.082 at 90, 0.095 at 105 with this cup
                 (r = cup_radius = 0.0281 m, mouth_z = cup_mouth_z = 0.0539 m from the env cfg).
                 This is the height a turning cup MUST buy back, and every height budget below adds it.
                 The sibling track only needs the r*sin term because its budget is measured on an
                 upright-equivalent mouth height; here the budgets are measured on the cup lift, so the
                 cos term belongs in as well.
                 It rides on theta_eff, not on src_cup_tilt: a sideways tilt needs no extra altitude to
                 pour, so it buys none."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]          # (N,2)
    sep = v.norm(dim=-1)                                               # (N,)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / sep_min, max=1.0)   # the azimuth is meaningless when the cups are coaxial

    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])                     # signed, negative = away
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    # what is tilted but NOT toward the receiver: 0 for a perfect pour attitude, ~= src_cup_tilt for a
    # sideways or backwards tilt.
    mis = torch.clamp(ctx.src_cup_tilt - theta_eff, min=0.0)

    # sin saturates at 90 deg, cos keeps running to 180, so lip_drop never decreases with the angle
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    cos_t = torch.cos(torch.clamp(theta_eff, 0.0, math.pi))
    lip_drop = lip_r * sin_t + mouth_z * (1.0 - cos_t)

    lip_xy_vec = ctx.src_cup_mouth_pos[:, :2] \
        + (lip_r * torch.cos(theta)).unsqueeze(-1) * d_hat - ctx.rcv_cup_mouth_pos[:, :2]
    lip_xy = lip_xy_vec.norm(dim=-1)
    lip_dz = ctx.src_cup_mouth_pos[:, 2] - lip_r * torch.sin(theta) - ctx.rcv_cup_mouth_pos[:, 2]
    # aim distance to a point `clearance` above the receiver rim; the height error counts half, exactly
    # as in iter_06/07/08 so carry_src and align_mouths keep their measured scale (align 2.84)
    d_align = torch.sqrt(lip_xy ** 2 + (0.5 * (lip_dz - clearance)) ** 2)
    return {"d_hat_x": d_hat[:, 0], "sep_ok": sep_ok, "theta": theta, "theta_eff": theta_eff,
            "mis": mis, "lip_xy": lip_xy, "lip_dz": lip_dz, "d_align": d_align,
            "sin_t": sin_t, "lip_drop": lip_drop}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt(SIGNED) < pour.
    # iter_10 — GOAL CHANGE: spill-free transfer. iter_09 already pours (replay: success 81 %, 70 % of
    # beads delivered) but 28 % of the beads go to the floor and action/step_delta sat at 0.44 all run.
    # Everything that built the pour (grasp, lifts, converge, align, signed tilt ladder, tilt-linked
    # height budget, corridor) is VERBATIM iter_09. iter_10 only ADDS / RE-PRICES:
    #   1. spill is the most expensive event: -12 per bead (was -3), and the per-step success and
    #      transfer incomes are scaled by (1 - spill/0.40), so one spilled bead also deletes 1/8 of the
    #      success income for the rest of the episode. Delivering a bead pays +7.5 once.
    #   2. flow limit: pour_delta pays at most 1 bead per step, every extra bead in the same step is
    #      unpaid AND charged (the sibling's flow_excess, re-expressed in beads because this track has
    #      20 beads = 0.05 per bead, above the sibling's 0.02 cap)
    #   3. unaimed_release: tilt magnitude past the release angle while the spout is NOT over the mouth
    #   4. overtilt (linear past release + 20 deg), cup speed/spin limits, receiver spin limit,
    #      command sign-flip and carry-phase rate charges. There is no adr_progress in this context, so
    #      all control charges are gated on BOTH cups grasped and lifted (up_w_src * up_w_rcv), the
    #      strongest ones additionally on the pour phase; weights are ~half the sibling's.
    # Measured in iter_09 and KEPT: grasp 0.736/0.775, diag_mis 1.03 deg, src tilt 72 deg.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.76 measured)
    W_AWAY_RCV = 0.8
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (best-ever grasp rates)
    # --- source lift: band + over-lift ramp. KEEP LIST for shape and weights; only the FREE HEIGHT
    #     moves, and it moves to 0.24 + lip_drop, i.e. it is identical at theta = 0 ------------------
    W_LIFT_SRC = 3.0
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.15
    LIFT_FREE_SRC0 = 0.24   # ★ base of the moving budget: 0.24 upright, 0.307 at 75 deg, 0.335 at 105
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    LIFT_FLOOR = 0.60
    SRC_UP_FLOOR = 0.5
    LIFT_PINCH_BONUS = 0.2
    # --- receiver lift band — KEEP LIST (rcv lift 0.075 m, lift_rcv 1.33) ---------------------------
    W_LIFT_RCV = 2.0
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10
    RCV_LIFT_SIGMA = 0.07
    W_HOLD_RCV = 1.2
    RCV_DISC_FLOOR = 0.35
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7
    # --- convergence / carry / align — KEEP LIST, unchanged ----------------------------------------
    W_CONVERGE = 3.0
    CONV_FAR = 1.00
    CONV_NEAR = 0.16
    CONV_SIGMA = 0.25
    W_CARRY, W_ALIGN = 3.0, 5.0
    CARRY_K, CARRY_RANGE = 1.2, 1.00
    ALIGN_K, ALIGN_SIGMA = 2.0, 0.15
    POUR_CLEARANCE = 0.04   # aim point 4 cm above the receiver rim; with the new lift budget a 75 deg
                            # pose already scores d_align 0.070 against 0.096 today, so aligning
                            # improves while turning and needs no tilt subsidy of its own
    SEP_MIN = 0.03          # below 3 cm of xy separation the pouring azimuth is numerically unstable
    # --- lip geometry gates ------------------------------------------------------------------------
    # cup_radius = 0.0432 * cup_scale 0.65 = 0.0281 m, cup_mouth_z = 0.0829 * 0.65 = 0.0539 m (env cfg).
    SAFE_FULL, SAFE_ZERO = 0.030, 0.060    # lip over the receiver mouth (inner radius 0.028 m)
    # ★ the distance factor that iter_08 did not have. Fitted to the trace: late lip_xy mean 0.0894 m,
    # 26.3 % of samples below 0.06, per-env minima 0.014-0.125 m. 1 at 0.05, 0.70 at 0.089, 0.12 at 0.12
    # — a live gradient exactly across the band the policy lives in. It replaces the old approach_gate
    # (0.08/0.16), which was 0.97 at the operating point and therefore blind.
    LEAD_FULL, LEAD_ZERO = 0.050, 0.140
    LIP_DEADBAND = 0.02
    AIM_XY_K = 12.0
    LIP_Z_LO, LIP_Z_HI = 0.0, 0.02         # lip ABOVE the receiver rim. Deliberately NOT tilt-linked:
                                           # lip_dz already contains -r*sin(theta), so adding it to the
                                           # thresholds too would demand 2*r*sin(theta) of clearance.
                                           # Beads leaving below the rim never land in the cup — this
                                           # gate is physics; what had to move is the lift budget.
    LIP_Z_FREE0, LIP_Z_SIGMA = 0.12, 0.10  # ★ + cup_radius*sin(theta): the sibling's own form, and it
                                           # exactly cancels the r*sin already inside lip_dz, so the
                                           # "lip too high" charge is measured upright-equivalent
    LIP_ABOVE_FLOOR = 0.35                 # soft when GATING income, hard when PAYING the pose
    # --- TILT LADDER on the SIGNED angle -----------------------------------------------------------
    # release = first-bead angle. Measured for this cup: ~70 deg full, a fifth gone by 80, half by 86;
    # the angle rises as the cup empties, so the target follows bead_in_source_frac.
    RELEASE_FULL = math.radians(75.0)
    RELEASE_SPAN = math.radians(25.0)
    PRE_MARGIN = math.radians(20.0)
    OVER_RANGE = math.radians(30.0)
    W_TILT_AIM = 2.0        # 3.0 -> 2.0: this was free income at any distance (lead factor 0.985) and
                            # the policy banked it at 45 deg and stopped. Now it is distance-gated.
    W_TILT_NEAR = 6.0       # 4.0 -> 6.0: theta_pre -> release, needs the spout inside the lead zone
    W_TILT_POUR = 10.0      # 7.0 -> 10.0: release -> release+30 deg, needs the spout OVER the mouth
    W_POUR_POSE = 5.0       # 3.0 -> 5.0: plateau at the release attitude; success needs a 10-step hold
    BAND_BELOW, BAND_ABOVE = 0.10, 0.40
    LEAD_FLOOR = 0.15       # 0.25 -> 0.15: more of tilt_aim now depends on actually being close
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # Gaussian width on the corridor EXCESS (unchanged)
    TILT_LIFT_H = 0.03      # corridor and tilt_aim fully open at 3 cm — the policy lives at 0.226 m
    W_EARLY_TILT = 0.8      # stateless stand-in for a premature-tilt latch (this track has no latch):
                            # charged every step the MAGNITUDE exceeds what the current aim allows
    MIS_FREE = math.radians(5.0)    # ★ refit: the trace has 53.4 % above 20 deg and 36.8 % above 40.
    MIS_SPAN = math.radians(60.0)   # linear and UNSATURATED to 60 deg — the old tanh(.../0.40) was at
                                    # 0.86 by 40 deg, i.e. flat exactly where most of the mass sits
    W_MISDIR = 1.6          # 1.2 -> 1.6; the true charge was -0.58, five times the logged -0.11
    DIR_FULL, DIR_ZERO = 0.0, 0.28   # d_hat_x > 0 = pouring away from the torso
    W_DIR_PEN = 1.0
    NEST_SAFE, NEST_ZERO = 0.115, 0.09
    W_NEAR_NEST = 1.0
    # --- receiver discipline — KEEP LIST -----------------------------------------------------------
    W_HOME_RCV = 0.8
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5
    STILL_FREE, STILL_SCALE = 0.05, 0.15
    # --- task / penalties --------------------------------------------------------------------------
    W_POUR = 150.0          # x d_in_target: 7.5 per bead with 20 beads — two orders above the level term
    W_TRANSFER = 4.0        # level term: success needs in_target >= 0.5 HELD for 10 steps
    W_SPILL = 240.0         # ★ 60 -> 240: -12 per bead spilled, above the +7.5 a delivered bead earns;
                            # beads_left also deletes the whole tilt income for that bead
    SETTLE_FRAC = 0.03      # ★ spill charge off for the first 3 % of the episode (27 steps): beads
                            # settling at spawn are not the policy's doing
    SPILL_ZERO = 0.40       # ★ success/transferred scaled by 1 - spill/0.40: the env's success spill
                            # limit is 0.40, so at that point the success income is worth nothing
    W_SUCCESS = 20.0
    # ★ flow limit (sibling iter_23 flow_excess, re-expressed in BEADS per step: 1 bead = 1/num_beads)
    FLOW_CAP_BEADS = 1.0    # paid per step at most 1 bead (60 beads/s at 60 Hz, so a trickle is free)
    W_FLOW = 40.0           # x excess fraction: -2 per extra bead in the same step (and it is unpaid)
    # ★ release while not aimed: tilt MAGNITUDE (beads do not care about direction) past the release
    # angle while the spout is outside the receiver mouth or below its rim
    UNAIMED_BELOW = math.radians(5.0)
    UNAIMED_SPAN = math.radians(10.0)
    W_UNAIMED = 1.5
    # ★ overtilt: linear, unsaturated past release + 20 deg (pour_tilt still pays up to +30 deg, so the
    # net slope stays positive to +30 and turns negative beyond it); capped
    OVERTILT_ABOVE = math.radians(20.0)
    W_OVERTILT, OVERTILT_MAX = 6.0, 3.0
    # ★ cup motion limits (sibling iter_23 thresholds, weights halved — warm start, no ADR ramp)
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, ROT_POUR_SCALE, W_ROT_POUR = 1.0, 1.5, 1.0   # stricter while pouring
    RCV_W_FREE, RCV_W_SCALE, W_RCV_SPIN = 0.5, 1.0, 0.5         # rcv linear speed: rcv_still (kept)
    # ★ command smoothness (sibling FLIP_CARRY_W 0.6 / HAND_FLIP_W 0.4 over 18 dims; here 12 + 12 dims)
    W_FLIP_PALM = 0.3       # x sum over the 12 palm dims of relu(-a*a_prev)
    W_FLIP_HAND = 0.2       # x sum over the 12 hand dims, while that hand holds its cup
    W_RATE_PALM_CARRY = 0.03   # x sum over 12 palm dims of da^2, carry phase only (base action_rate kept)
    W_RATE_HAND_CARRY = 0.01
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_TILT_PEN_RCV = 0.3
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05
    # ★ palm action saturation. src palm y sat 97.6 % of steps (mean +0.993), pitch 92.0 % (+0.977),
    # with the arm joints settled at 0.28-0.37 rad/s — the arm is stopped, not following. There is no
    # adr_progress in this context, so the charge is gated on both cups grasped AND lifted, which a
    # fresh policy does not reach for hundreds of epochs. Worst case -0.30 against a total of ~14.5.
    SAT_LO, W_PALM_SAT = 0.7, 0.15
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35

    dtype = ctx.src_cup_tilt.dtype
    geo = _pour_geometry(ctx, ctx.cup_radius, ctx.cup_mouth_z, POUR_CLEARANCE, SEP_MIN)
    d_align = geo["d_align"]
    theta_eff = geo["theta_eff"]

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)
    not_nested = 1.0 - nested

    # ---- spout gates: how close the pouring lip is to the receiver mouth --------------------------
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL) / (LEAD_ZERO - LEAD_FULL))
    safe = 1.0 - _smoothstep((geo["lip_xy"] - SAFE_FULL) / (SAFE_ZERO - SAFE_FULL))
    aim_xy = torch.exp(-AIM_XY_K * torch.clamp(geo["lip_xy"] - LIP_DEADBAND, min=0.0))
    # ★ the height budget follows the tilt. lip_free cancels the r*sin already inside lip_dz, so this
    # charge asks "is the cup absurdly high for an UPRIGHT pour", not "did you dare to rotate".
    lip_free = LIP_Z_FREE0 + ctx.cup_radius * geo["sin_t"]
    lip_above_raw = torch.clamp((geo["lip_dz"] - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(geo["lip_dz"] - lip_free, min=0.0) / LIP_Z_SIGMA) ** 2)
    lip_above = LIP_ABOVE_FLOOR + (1.0 - LIP_ABOVE_FLOOR) * lip_above_raw     # gating (keeps a floor)
    z_ok = lip_above_raw * lip_high                                          # paying (no floor)
    dir_ok = 1.0 - _smoothstep((geo["d_hat_x"] - DIR_FULL) / (DIR_ZERO - DIR_FULL))
    dir_soft = 0.3 + 0.7 * dir_ok       # soft: this env does not void on direction, but the real robot
                                        # must pour across or toward the torso, never away from it

    # ---- release angle: rises as the cup empties (first bead ~70 deg full, ~86 deg at half) --------
    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    release_tilt = RELEASE_FULL + RELEASE_SPAN * (1.0 - fill)
    theta_pre = torch.clamp(release_tilt - PRE_MARGIN, min=0.30)
    theta_max = release_tilt + OVER_RANGE

    # ---- source attitude CORRIDOR, opened by the SPOUT DISTANCE and charged on the MAGNITUDE -------
    # 0.15 rad on the table -> theta_pre (55 deg full) once carried -> release (75 deg) once the spout
    # is inside the lead zone -> release+30 deg once it is over the mouth itself. iter_08 used the old
    # approach_gate here, which was 0.97 at lip_xy 0.089 m, so the corridor already allowed 74 deg from
    # 9 cm away and 11 % of samples tilted past 75 deg from there — that is where the 12 % spill came
    # from. With lead_gate the corridor tops out at ~69 deg at 9 cm (the observed ceiling, now for an
    # explicit and closable reason), reaches 75 deg at lip_xy ~0.055 and past 100 deg once safe opens.
    carry_soft_src = g_src * torch.clamp(h_src / TILT_LIFT_H, max=1.0)
    allowed = TILT_FREE_TABLE + carry_soft_src * torch.clamp(theta_pre - TILT_FREE_TABLE, min=0.0)
    allowed = allowed + carry_soft_src * lead_gate * torch.clamp(release_tilt - theta_pre, min=0.0)
    allowed = allowed + carry_soft_src * safe * (theta_max - release_tilt)
    tilt_excess_src = torch.clamp(ctx.src_cup_tilt - allowed, min=0.0)
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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # 0.60 .. 1.00 (measured rcv_prog 0.74)

    # ---- source lift: band + linear cost above it. ★THE iter_09 FIX: the free height is no longer a
    # constant 0.24 m but 0.24 + lip_drop(theta_eff). The trace shows the policy buying that height by
    # itself (cup dz 0.152 m at 40-60 deg -> 0.205 m above 75 deg) and being charged twice for it, by
    # src_lift_band cutting lift_src and by over_lift. Identical to iter_08 at theta = 0. -----------
    lift_free_src = LIFT_FREE_SRC0 + geo["lip_drop"]
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_lift_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_lift_band * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])
    # ungated by the grasp flag on purpose: throwing the cup up must not dodge the cost
    over_lift = torch.clamp((h_src - lift_free_src) / OVER_LIFT_RANGE, 0.0, 1.0)

    # ---- receiver lift band (unchanged — measured 0.075 m, inside the 0.10 m band) -----------------
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
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close     # 0.99 at the measured 0.11 m — saturated, which
                                                       # is why it can no longer be the tilt lead factor
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.77 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * not_nested * conv_shape

    # carry / align keep their iter_06 shape and weights on the DIRECTED spout (align 2.84 measured)
    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * not_nested
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * not_nested
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- TILT: three segments of the SIGNED angle, each with a stricter SPOUT requirement -----------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    # beads_left is the anti-spill interlock: beads on the table delete the whole ladder for the rest
    # of the episode, on top of spill_delta's -3 per bead.
    tilt_hold = carry_soft_src * beads_left * not_nested * geo["sep_ok"] * dir_soft
    lead_dist = LEAD_FLOOR + (1.0 - LEAD_FLOOR) * lead_gate     # ★ the real spout distance now, not a
                                                                # cup-xy factor pinned at 0.985
    aim_frac = torch.clamp(theta_eff / theta_pre, 0.0, 1.0)                                  # 0 -> 55 deg
    near_frac = torch.clamp((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.05),
                            0.0, 1.0)                                                        # 55 -> 75
    pour_frac = torch.clamp((theta_eff - release_tilt) / (theta_max - release_tilt), 0.0, 1.0)  # 75 -> 105
    tilt_aim = tilt_hold * aim_frac * lead_dist
    tilt_near = tilt_hold * near_frac * lead_gate * lip_above
    # the release segment: only with the spout actually over the receiver mouth and above its rim
    pour_tilt = both_up * beads_left * dir_soft * upright_rcv * safe * lip_above * pour_frac
    # plateau at the release attitude so holding the pose beats swinging through it (10-step streak)
    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = both_up * beads_left * dir_ok * upright_rcv * safe * z_ok * aim_xy * pour_band

    # ---- directional / prematurity penalties -------------------------------------------------------
    early_tilt = torch.tanh(tilt_excess_src / TILT_TOL)      # magnitude above what the aim allows
    # ★ linear and unsaturated to 60 deg: 0.25 at 20, 0.58 at 40, 0.92 at 60. The old
    # tanh(clamp(mis - 10 deg)/0.40) was already 0.86 at 40 deg, i.e. flat over the 37 % of samples
    # that sit above it, which is why a -0.58 charge showed up as -0.11 in the epoch means.
    mis_pen = torch.clamp((geo["mis"] - MIS_FREE) / MIS_SPAN, 0.0, 1.0)
    mis_tilt = carry_soft_src * mis_pen
    # charged whenever the source cup is carried: the relative placement of the two cups is decided
    # during convergence, long before the spout is inside the aim radius
    pour_dir_pen = carry_soft_src * _smoothstep((geo["d_hat_x"] - 0.10) / 0.25)
    cup_center_d = (ctx.src_cup_pos - ctx.rcv_cup_pos).norm(dim=-1)
    near_nest = _smoothstep((NEST_SAFE - cup_center_d) / (NEST_SAFE - NEST_ZERO))   # 0 at the pour pose
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    # ★ palm saturation. Action layout of THIS track: [0:6] src palm 6-DoF, [6:12] src hand,
    # [12:18] rcv palm, [18:24] rcv hand. Gated on both cups grasped AND lifted — the stand-in for the
    # sibling's adr_progress ramp, which does not exist in this context. Off for the whole early run.
    a_abs = ctx.actions.abs()
    sat_gate = up_w_src * up_w_rcv
    sat_src_raw = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv_raw = torch.mean(_smoothstep((a_abs[:, 12:18] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)

    # ================================================================ ★ iter_10: spill-free transfer
    # control gate WITHOUT adr_progress: both cups grasped and >= 5.5 cm up (the same gate the palm
    # saturation charge used in iter_09 without hurting the pour). pour_phase additionally needs the
    # spout inside the lead zone and the cup turning toward the receiver.
    ctrl_gate = up_w_src * up_w_rcv
    pour_phase = ctrl_gate * lead_gate * aim_frac

    # ---- beads: flow-limited increments, spill priced above everything it can buy ------------------
    n_beads = float(max(int(ctx.num_beads), 1))
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)              # 1 bead per step paid in full
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)         # the rest of a dump: unpaid + charged
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_step = settled * torch.clamp(ctx.d_spill, min=0.0)
    spill_lvl = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    clean_f = torch.clamp(1.0 - spill_lvl / SPILL_ZERO, 0.0, 1.0)   # 1 clean, 0 at the success limit

    # ---- aiming at the moment of release: tilt MAGNITUDE past (release - 5 deg) while the spout is
    #      not over the mouth (safe) or not above the rim. 0 for a correct pour pose. -----------------
    has_beads = torch.clamp(fill / 0.10, max=1.0)               # >= 2 beads left
    releasing = _smoothstep((ctx.src_cup_tilt - (release_tilt - UNAIMED_BELOW)) / UNAIMED_SPAN)
    unaimed = carry_soft_src * has_beads * releasing * (1.0 - safe * lip_above_raw)

    # ---- overtilt: linear past release + 20 deg, carried cup only ----------------------------------
    overtilt = carry_soft_src * torch.clamp(
        W_OVERTILT * torch.clamp(ctx.src_cup_tilt - (release_tilt + OVERTILT_ABOVE), min=0.0),
        max=OVERTILT_MAX)

    # ---- cup motion limits ------------------------------------------------------------------------
    v_src = src["v_lin"]
    w_src = ctx.src_cup_ang_vel.norm(dim=-1)
    w_rcv = ctx.rcv_cup_ang_vel.norm(dim=-1)
    src_speed = ctrl_gate * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = ctrl_gate * torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE)
    rot_pour = pour_phase * torch.tanh(torch.clamp(w_src - ROT_POUR_FREE, min=0.0) / ROT_POUR_SCALE)
    rcv_spin = ctrl_gate * torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE)

    # ---- command smoothness. Layout: [0:6] src palm, [6:12] src hand, [12:18] rcv palm,
    #      [18:24] rcv hand. --------------------------------------------------------------------------
    da = ctx.actions - ctx.prev_actions
    flips = torch.clamp(-(ctx.actions * ctx.prev_actions), min=0.0)
    flip_palm = flips[:, 0:6].sum(dim=-1) + flips[:, 12:18].sum(dim=-1)
    flip_hand = flips[:, 6:12].sum(dim=-1) + flips[:, 18:24].sum(dim=-1)
    rate_palm = (da[:, 0:6] ** 2).sum(dim=-1) + (da[:, 12:18] ** 2).sum(dim=-1)
    rate_hand = (da[:, 6:12] ** 2).sum(dim=-1) + (da[:, 18:24] ** 2).sum(dim=-1)

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
        "tilt_aim": W_TILT_AIM * tilt_aim,
        "tilt_near": W_TILT_NEAR * tilt_near,
        "pour_tilt": W_TILT_POUR * pour_tilt,
        "pour_pose": W_POUR_POSE * pour_pose,
        "pour_delta": W_POUR * pour_paid,
        "flow_excess": -W_FLOW * flow_excess,
        "transferred": W_TRANSFER * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0) * not_nested * upright_rcv
        * clean_f,
        "spill_delta": -W_SPILL * spill_step,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype) * clean_f,
        "unaimed_release": -W_UNAIMED * unaimed,
        "overtilt": -overtilt,
        "src_speed": -W_SRC_SPEED * src_speed,
        "src_spin": -W_SRC_SPIN * src_spin,
        "rot_pour": -W_ROT_POUR * rot_pour,
        "rcv_spin": -W_RCV_SPIN * rcv_spin,
        "act_flip": -W_FLIP_PALM * ctrl_gate * flip_palm,
        "hand_flip": -W_FLIP_HAND * ctrl_gate * flip_hand,
        "carry_rate": -ctrl_gate * (W_RATE_PALM_CARRY * rate_palm + W_RATE_HAND_CARRY * rate_hand),
        "early_tilt": -W_EARLY_TILT * early_tilt,
        "mis_tilt": -W_MISDIR * mis_tilt,
        "pour_dir_pen": -W_DIR_PEN * pour_dir_pen,
        "near_nested": -W_NEAR_NEST * near_nest,
        "cup_disturb_src": -W_SLIDE_SRC * src["slide"],
        "cup_disturb_rcv": -(W_SLIDE_RCV * rcv["slide"] + W_TILT_PEN_RCV * tilt_pen_rcv),
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
        "palm_sat_src": -W_PALM_SAT * sat_gate * sat_src_raw,
        "palm_sat_rcv": -W_PALM_SAT * sat_gate * sat_rcv_raw,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    # diagnostics only (added AFTER the sum, not part of the reward). One per fix, so the next round
    # can tell which of the four moved: the moving lift budget, the new distance factor, the raw palm
    # saturation, and the signed / misdirected split of the tilt.
    comps["rcv_progress"] = rcv_prog
    comps["aim_gate"] = safe
    comps["diag_theta_deg"] = theta_eff * (180.0 / math.pi)
    comps["diag_mis_deg"] = geo["mis"] * (180.0 / math.pi)
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lift_budget"] = lift_free_src
    comps["diag_lead_gate"] = lead_gate
    comps["diag_palm_sat"] = sat_src_raw
    # iter_10 diagnostics: beads landing per step, spill cleanliness, control gates, raw cup speeds
    comps["diag_flow_beads"] = torch.clamp(d_in, min=0.0) * n_beads
    comps["diag_clean"] = clean_f
    comps["diag_ctrl_gate"] = ctrl_gate
    comps["diag_pour_phase"] = pour_phase
    comps["diag_src_w"] = w_src
    comps["diag_src_v"] = v_src
    comps["diag_flip_palm"] = flip_palm
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0.0333, 0.0667, 0.1, 0.133, 0.133, 0.133, 0.133, 0.133]  min 0 · mean 0.0932 · max 0.133
bead/in_target: [0, 0.405, 0.457, 0.451, 0.46, 0.349, 0.294, 0.24, 0.196, 0.184]  min 0 · mean 0.325 · max 0.476
bead/spill: [0, 0.172, 0.142, 0.14, 0.146, 0.154, 0.181, 0.17, 0.179, 0.193]  min 0 · mean 0.169 · max 0.368
done/drop: [0, 0.000488, 0.000244, 0, 0, 0.000244, 0.000488, 0.000244, 0, 0]  min 0 · mean 0.000399 · max 0.00146
done/mimic_runaway: [0, 0.000488, 0.000488, 0, 0.000732, 0.000244, 0.000488, 0.000244, 0.000244, 0.000244]  min 0 · mean 0.000366 · max 0.00146
episode_lengths/step: [120, 582, 677, 671, 719, 568, 620, 605, 633, 549]  min 120 · mean 595 · max 727
reward/act_flip: [0, -0.252, -0.27, -0.273, -0.277, -0.229, -0.223, -0.214, -0.203, -0.204]  min -0.358 · mean -0.231 · max 0
reward/action_rate: [-0.0161, -0.0213, -0.0216, -0.0215, -0.0217, -0.0201, -0.0198, -0.0191, -0.0185, -0.018]  min -0.0241 · mean -0.0199 · max -0.0161
reward/aim_gate: [0, 0.451, 0.49, 0.483, 0.48, 0.39, 0.314, 0.352, 0.329, 0.377]  min 0 · mean 0.404 · max 0.56
reward/align_mouths: [0, 1.72, 1.93, 2.06, 2.21, 1.99, 1.88, 1.87, 1.83, 1.94]  min 0 · mean 1.91 · max 2.76
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/away_rcv: [-0.043, -0.155, -0.158, -0.156, -0.157, -0.157, -0.163, -0.16, -0.164, -0.162]  min -0.167 · mean -0.159 · max -0.043
reward/carry_rate: [0, -0.167, -0.175, -0.174, -0.177, -0.147, -0.142, -0.135, -0.131, -0.132]  min -0.213 · mean -0.15 · max 0
reward/carry_src: [0, 0.908, 1.03, 1.15, 1.27, 1.22, 1.18, 1.19, 1.15, 1.18]  min 0 · mean 1.12 · max 1.66
reward/close_rcv: [0.112, 0.0755, 0.0747, 0.0771, 0.0776, 0.078, 0.0793, 0.0864, 0.085, 0.0839]  min 0.0643 · mean 0.0793 · max 0.112
reward/close_src: [0.0132, 0.0153, 0.0174, 0.016, 0.015, 0.0137, 0.0144, 0.0145, 0.0155, 0.0167]  min 0.0131 · mean 0.0155 · max 0.0205
reward/contact_push: [-0.00492, -0.00829, -0.00805, -0.00837, -0.00775, -0.0089, -0.00774, -0.00855, -0.00811, -0.00868]  min -0.0386 · mean -0.00837 · max -0.0014
reward/converge: [0, 1.31, 1.45, 1.55, 1.68, 1.54, 1.5, 1.51, 1.5, 1.56]  min 0 · mean 1.5 · max 1.97
reward/cup_collision: [0, -0.0243, -0.0203, -0.0159, -0.0165, -0.00644, -0.00343, -0.00286, -0.00198, -0.00168]  min -0.0801 · mean -0.0101 · max 0
reward/cup_disturb_rcv: [-0.0288, -0.127, -0.124, -0.117, -0.112, -0.107, -0.105, -0.106, -0.11, -0.111]  min -0.277 · mean -0.116 · max -0.0288
reward/cup_disturb_src: [-0.0829, -0.0333, -0.0304, -0.0346, -0.0304, -0.0354, -0.0375, -0.036, -0.0418, -0.0449]  min -0.104 · mean -0.0371 · max -0.0104
reward/cup_fling: [-0.000463, -0.0162, -0.0156, -0.0159, -0.0149, -0.0158, -0.0155, -0.0153, -0.0172, -0.0164]  min -0.0705 · mean -0.0164 · max -0.000463
reward/diag_clean: [1, 0.655, 0.712, 0.716, 0.699, 0.69, 0.653, 0.677, 0.676, 0.666]  min 0.342 · mean 0.676 · max 1
reward/diag_ctrl_gate: [0, 0.656, 0.669, 0.653, 0.676, 0.605, 0.607, 0.597, 0.594, 0.61]  min 0 · mean 0.622 · max 0.753
reward/diag_flip_palm: [0.419, 1.05, 1.1, 1.12, 1.14, 1.01, 1, 0.976, 0.919, 0.919]  min 0.419 · mean 1 · max 1.45
reward/diag_flow_beads: [0, 0.0195, 0.0203, 0.0173, 0.0178, 0.0193, 0.0154, 0.0105, 0.00928, 0.01]  min 0 · mean 0.0152 · max 0.083
reward/diag_lead_gate: [0, 0.668, 0.689, 0.679, 0.684, 0.622, 0.611, 0.618, 0.597, 0.626]  min 0 · mean 0.639 · max 0.776
reward/diag_lift_budget: [0.24, 0.306, 0.307, 0.308, 0.309, 0.304, 0.303, 0.302, 0.3, 0.302]  min 0.24 · mean 0.304 · max 0.318
reward/diag_lip_xy: [0.291, 0.345, 0.215, 0.332, 0.187, 0.482, 0.481, 0.396, 0.395, 0.28]  min 0.127 · mean 0.424 · max 1.12
reward/diag_mis_deg: [0.512, 2.49, 1.81, 1.63, 1.86, 1.84, 1.68, 1.39, 1.41, 1.51]  min 0.512 · mean 1.78 · max 11.8
reward/diag_palm_sat: [0.343, 0.529, 0.526, 0.523, 0.534, 0.528, 0.51, 0.518, 0.517, 0.532]  min 0.343 · mean 0.523 · max 0.588
reward/diag_pour_phase: [0, 0.581, 0.605, 0.584, 0.597, 0.534, 0.518, 0.517, 0.503, 0.53]  min 0 · mean 0.547 · max 0.672
reward/diag_src_v: [0.0418, 0.456, 0.292, 0.328, 0.254, 0.414, 0.322, 0.327, 0.396, 0.373]  min 0.0418 · mean 3.56 · max 829
reward/diag_src_w: [0.55, 2.18, 2.09, 2.01, 2, 2.29, 2, 2.16, 2.05, 2.09]  min 0.55 · mean 34.4 · max 4.88e+03
reward/diag_theta_deg: [0.705, 73.2, 74.6, 75.2, 77, 70.8, 70.6, 69.1, 66.9, 69.1]  min 0.705 · mean 71.3 · max 86.4
reward/early_tilt: [-0.00209, -0.127, -0.106, -0.133, -0.134, -0.156, -0.163, -0.165, -0.171, -0.174]  min -0.234 · mean -0.151 · max -0.00209
reward/flow_excess: [0, -0.00586, -0.00195, -0.00293, -0.00195, -0.00195, -0.000488, -0.000488, -1.49e-09, -1.56e-09]  min -0.0146 · mean -0.00162 · max 0
reward/grasp_rcv: [0.000298, 0.861, 0.893, 0.96, 1.01, 0.988, 1.06, 1.1, 1.08, 1.1]  min 0.000298 · mean 0.99 · max 1.18
reward/grasp_src: [0.000117, 1.12, 1.26, 1.17, 1.24, 1.06, 1.13, 1.14, 1.15, 1.16]  min 0.000117 · mean 1.15 · max 1.45
reward/hand_flip: [0, -0.0723, -0.0698, -0.0643, -0.072, -0.0609, -0.0592, -0.0559, -0.0552, -0.0543]  min -0.088 · mean -0.0617 · max 0
reward/hand_foreign: [-0.0147, -0.00461, -0.00387, -0.00495, -0.0055, -0.00522, -0.00617, -0.00594, -0.00555, -0.00499]  min -0.019 · mean -0.00546 · max -0.00299
reward/hold_rcv: [0, 0.142, 0.173, 0.208, 0.238, 0.225, 0.195, 0.2, 0.192, 0.195]  min 0 · mean 0.193 · max 0.25
reward/lift_rcv: [0, 0.738, 0.855, 1.01, 1.16, 1.19, 1.23, 1.21, 1.19, 1.2]  min 0 · mean 1.07 · max 1.48
reward/lift_src: [0, 1.61, 1.77, 1.97, 2.1, 2.09, 2.21, 2.2, 2.06, 2.06]  min 0 · mean 1.98 · max 2.68
reward/mis_tilt: [0, -0.0181, -0.0153, -0.01, -0.0115, -0.00948, -0.00585, -0.00516, -0.00621, -0.00689]  min -0.104 · mean -0.0104 · max 0
reward/near_nested: [0, -0.027, -0.0245, -0.0138, -0.0102, -0.00859, -0.00844, -0.00591, -0.00495, -0.00939]  min -0.0768 · mean -0.0124 · max 0
reward/nested: [0, -0.00293, -0.000488, -0.00293, -0.000977, -0.00342, -0.00244, -0.00391, -0.000977, -0.00195]  min -0.00732 · mean -0.00256 · max 0
reward/over_lift_src: [0, -0.271, -0.229, -0.15, -0.124, -0.0675, -0.0347, -0.0396, -0.0744, -0.088]  min -0.492 · mean -0.124 · max 0
reward/overtilt: [0, -0.0324, -0.0358, -0.0377, -0.0188, -0.00713, -0.00438, -0.00223, -0.00223, -0.00453]  min -0.241 · mean -0.016 · max 0
reward/palm_sat_rcv: [0, -0.0397, -0.0404, -0.0387, -0.0408, -0.037, -0.0375, -0.0394, -0.038, -0.038]  min -0.0486 · mean -0.0386 · max 0
reward/palm_sat_src: [0, -0.0581, -0.0589, -0.0569, -0.0597, -0.0549, -0.0533, -0.0531, -0.0533, -0.0559]  min -0.0713 · mean -0.0555 · max 0
reward/pour_delta: [0, 0.119, 0.143, 0.108, 0.126, 0.13, 0.108, 0.0696, 0.0439, 0.0696]  min -0.00183 · mean 0.0995 · max 0.52
reward/pour_dir_pen: [0, -0.236, -0.314, -0.23, -0.201, -0.0661, -0.0332, -0.0286, -0.0334, -0.0468]  min -0.427 · mean -0.132 · max 0
reward/pour_pose: [0, 0.183, 0.211, 0.253, 0.27, 0.323, 0.287, 0.322, 0.271, 0.282]  min 0 · mean 0.264 · max 0.851
reward/pour_tilt: [0, 0.355, 0.406, 0.459, 0.452, 0.478, 0.403, 0.424, 0.364, 0.376]  min 0 · mean 0.405 · max 1.19
reward/rcv_home: [0, -0.079, -0.0709, -0.0643, -0.0746, -0.132, -0.182, -0.176, -0.162, -0.151]  min -0.185 · mean -0.117 · max 0
reward/rcv_progress: [0.277, 0.749, 0.747, 0.756, 0.77, 0.749, 0.749, 0.758, 0.743, 0.752]  min 0.277 · mean 0.749 · max 0.826
reward/rcv_spin: [0, -0.138, -0.137, -0.127, -0.141, -0.107, -0.111, -0.109, -0.118, -0.129]  min -0.192 · mean -0.124 · max 0
reward/rcv_still: [0, -0.345, -0.351, -0.349, -0.357, -0.332, -0.352, -0.355, -0.354, -0.368]  min -0.464 · mean -0.35 · max 0
reward/reach_rcv: [0.893, 0.822, 0.816, 0.819, 0.817, 0.818, 0.808, 0.818, 0.811, 0.815]  min 0.805 · mean 0.817 · max 0.893
reward/reach_src: [0.542, 0.526, 0.52, 0.519, 0.509, 0.518, 0.504, 0.505, 0.505, 0.505]  min 0.492 · mean 0.513 · max 0.626
reward/rot_pour: [0, -0.196, -0.216, -0.197, -0.214, -0.17, -0.158, -0.157, -0.164, -0.171]  min -0.421 · mean -0.18 · max 0
reward/rush_near_cup: [-0.221, -0.038, -0.036, -0.0376, -0.0353, -0.0442, -0.04, -0.041, -0.0418, -0.0437]  min -0.221 · mean -0.0408 · max -0.0258
reward/spill_delta: [0, -0.117, -0.0908, -0.117, -0.082, -0.132, -0.217, -0.199, -0.24, -0.185]  min -0.838 · mean -0.178 · max 0
reward/src_speed: [0, -0.104, -0.111, -0.108, -0.11, -0.0952, -0.0916, -0.0875, -0.0977, -0.0992]  min -0.275 · mean -0.102 · max 0
reward/src_spin: [0, -0.087, -0.0902, -0.0824, -0.0932, -0.0735, -0.0705, -0.0694, -0.0748, -0.0757]  min -0.185 · mean -0.0788 · max 0
reward/success_bonus: [0, 5, 6.54, 6.42, 6.36, 4.15, 3, 2.25, 1.67, 1.47]  min 0 · mean 3.86 · max 6.84
reward/tilt_aim: [0, 0.69, 0.668, 0.706, 0.728, 0.795, 0.803, 0.819, 0.77, 0.787]  min 0 · mean 0.734 · max 1.31
reward/tilt_near: [0, 1.57, 1.56, 1.74, 1.86, 2.04, 1.96, 2.09, 1.95, 2.05]  min 0 · mean 1.84 · max 3.23
reward/total: [2.15, 16.4, 19.3, 20.4, 21.3, 18.8, 17.3, 16.7, 15.3, 15.5]  min 2.15 · mean 17.4 · max 21.5
reward/transferred: [0, 0.659, 0.924, 0.956, 0.956, 0.652, 0.495, 0.376, 0.278, 0.254]  min 0 · mean 0.575 · max 0.999
reward/unaimed_release: [0, -0.143, -0.102, -0.106, -0.116, -0.113, -0.121, -0.119, -0.145, -0.162]  min -0.702 · mean -0.125 · max 0
rewards/step: [143, 9.38e+03, 1.31e+04, 1.28e+04, 1.67e+04, 1.08e+04, 1.11e+04, 1.06e+04, 1.04e+04, 8.27e+03]  min 143 · mean 1.04e+04 · max 1.71e+04
task/aim_dist: [0.319, 0.527, 0.392, 0.434, 0.336, 0.611, 0.614, 0.53, 0.561, 0.38]  min 0.191 · mean 0.578 · max 1.41
task/cup_collision_rate: [0, 0.0498, 0.0359, 0.0303, 0.0288, 0.0112, 0.00537, 0.00366, 0.00244, 0.00269]  min 0 · mean 0.0188 · max 0.158
task/cups_center_dist: [0.32, 0.581, 0.447, 0.491, 0.394, 0.666, 0.669, 0.584, 0.613, 0.434]  min 0.202 · mean 0.632 · max 1.46
task/episode_success: [0, 0.413, 0.473, 0.46, 0.48, 0.348, 0.263, 0.213, 0.155, 0.136]  min 0 · mean 0.31 · max 0.494
task/nested_rate: [0, 0.00146, 0.000244, 0.00146, 0.000488, 0.00171, 0.00122, 0.00195, 0.000488, 0.000977]  min 0 · mean 0.00128 · max 0.00366
task/rcv_cup_lift: [0.000147, 0.113, 0.216, 0.0975, 0.0921, 0.0806, 0.0675, 0.0807, 0.073, 0.0701]  min 0.000147 · mean 0.0929 · max 0.224
task/rcv_grasped: [0, 0.76, 0.765, 0.78, 0.799, 0.765, 0.774, 0.778, 0.759, 0.772]  min 0 · mean 0.768 · max 0.856
task/rcv_hand_foreign_rate: [0.000488, 0.00317, 0.00146, 0.00269, 0.00342, 0.000977, 0.00171, 0.00293, 0.00293, 0.00244]  min 0.000488 · mean 0.00259 · max 0.00806
task/src_cup_lift: [0.000349, 0.448, 0.49, 0.328, 0.369, 0.339, 0.397, 0.353, 0.398, 0.259]  min 0.000349 · mean 0.397 · max 0.776
task/src_grasped: [0, 0.718, 0.739, 0.721, 0.741, 0.682, 0.696, 0.693, 0.686, 0.694]  min 0 · mean 0.702 · max 0.801
task/src_hand_foreign_rate: [0.0396, 0.0117, 0.011, 0.0125, 0.0139, 0.0166, 0.0168, 0.0164, 0.0186, 0.0154]  min 0.00781 · mean 0.0156 · max 0.0571
task/src_tilt_deg: [1.22, 75.7, 76.4, 76.9, 78.8, 72.7, 72.3, 70.5, 68.3, 70.6]  min 1.22 · mean 73.1 · max 87.7
task/success_now: [0, 0.423, 0.485, 0.467, 0.49, 0.358, 0.271, 0.219, 0.161, 0.14]  min 0 · mean 0.319 · max 0.505

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 11 operator notes — 안정 속도는 지키고, 조준을 틸팅보다 먼저 끝내게 한다

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 목표와 원칙 (사용자)

- 목표(09.22): **흘림 없는 이송** — 안정 제어·정밀 조준·비드 이송.
- ★원칙(09.22): "SIM 에서는 속도를 빠르게 해서 틸팅하는 게 좋다고 나오지만, 실제 제어로 갈 때는 안정적인 속도로 틸팅하는 게 좋은 것."
  **속도 벌점·제어 벌점을 풀어서 SIM 효율을 되찾는 설계는 금지다.** 판정 기준은 "안정 속도에서 흘림이 얼마나 되는가"다.

## 1. iter_10 결과 (t2r_rh3_i10, server GPU0, 4096 env, e387 · 101 M frames 에서 종료)

체크포인트 주변 ±25 epoch 평균:

| ckpt | 흘림/유출 | 이송 효율 | success | in_source | action/step_delta | 회전 초과 | entropy |
|---|---|---|---|---|---|---|---|
| ep50 | 0.257 | 0.711 | 0.445 | 0.389 | 0.437 | 0.210 | 8.78 |
| **ep100** | **0.224** | **0.746** | **0.453** | 0.405 | 0.435 | 0.193 | 8.72 |
| ep150 | 0.242 | 0.727 | 0.451 | 0.397 | 0.427 | 0.197 | 8.62 |
| ep200 | 0.321 | 0.635 | 0.319 | 0.489 | 0.409 | 0.160 | 8.58 |
| ep250 | 0.369 | 0.581 | 0.245 | 0.536 | 0.397 | 0.161 | 8.38 |
| ep300 | 0.431 | 0.510 | 0.176 | 0.580 | 0.383 | 0.163 | 8.17 |
| ep350 | 0.471 | 0.465 | 0.148 | 0.583 | 0.365 | 0.172 | 8.07 |

(흘림/유출 = spill/(1−in_source), 이송 효율 = in_target/(1−in_source), 회전 초과 = −reward/rot_pour)

### 잘 된 것
- **제어 안정이 꾸준히 좋아졌다**: action/step_delta 0.437 → **0.365**. iter_09 는 런 내내 0.44 평평이었다.
- **컵 회전이 느려졌다**: 회전 초과 0.210 → 0.16~0.17. 사용자 원칙대로다.
- 컵 충돌: 초반 0.047 까지 올랐다가 **0.002** 로 해소.
- 초반 100 epoch 은 흘림/유출 **0.224**·효율 0.746 으로 트랙 최고였다(iter_09 는 0.27~0.36).

### 무너진 것
- ep100 이후 **흘림/유출 0.224 → 0.471 로 4배 가까이 악화**, 효율 0.746 → 0.465, success 0.453 → 0.148.
- in_source 0.405 → 0.583 — 덜 붓는다.
- entropy 8.72 → 8.07 — 성능 하락과 함께 급락.

## 2. ★원인: 속도가 아니라 조준이다 (측정)

**같은 회전 속도 대역으로 맞춰 런 전반(e0~187)과 후반(e187~375)을 비교했다**(컵 거리 0.9 m 이상 폭주 epoch 제외):

| 회전 초과 대역 | 흘림/유출 전반 | 흘림/유출 후반 | 이송 효율 |
|---|---|---|---|
| 0.16~0.18 | **0.259** (21 ep) | **0.412** (89 ep) | 0.702 → 0.532 |
| 0.18~0.20 | **0.248** (71 ep) | **0.480** (6 ep) | 0.720 → 0.457 |

**속도를 맞춰 놓고도 흘림이 1.6~1.9배**다. 속도로는 설명되지 않는다.

조준이 무너진 증거:
- `diag_lip_xy`(붓는 림 점과 리시버 입구의 xy 거리) 50-ep 0.30 → **0.49 m**
- `unaimed_release` 벌점 -0.103 → **-0.143** — 조준 없이 기울이는 빈도가 늘었다

[가설] 느리게 기울이면 비드가 기울기 **도중부터** 나오기 시작한다. 빠르게 돌릴 때는 원심력으로 비드를 붙잡았다가 방출각에서 한꺼번에 내보냈으므로
조준이 늦어도 됐다(사용자 가설, epoch 평균으로는 확정 안 됨 — 빠른 회전 epoch 이 이송 효율이 높았다: 상위 25 % 0.679 vs 하위 25 % 0.610).
**안정 속도에서는 조준이 틸팅보다 먼저 끝나 있어야 한다.**

## 3. iter_10 의 조준 선행 장치가 약한 이유 (코드 확인)

`reward_gen/pour_bi_rh/iter_10/compute_reward.py`:

```python
UNAIMED_BELOW = math.radians(5.0)      # L290
UNAIMED_SPAN  = math.radians(10.0)     # L291
W_UNAIMED     = 1.5                    # L292
releasing = _smoothstep((ctx.src_cup_tilt - (release_tilt - UNAIMED_BELOW)) / UNAIMED_SPAN)   # L499
unaimed   = carry_soft_src * has_beads * releasing * (1.0 - safe * lip_above_raw)            # L500
```

1. **창이 늦다.** `release_tilt` 는 가득 찬 컵에서 75° 이므로 벌점은 **70° 에서 시작해 80° 에서 최대**다. 조준 요구가 틸팅 끝에서야 켜진다.
2. **경사가 없다.** 조준 여부를 `safe`(lip_xy 0.03~0.06 m) 로만 판정한다. 실제 운영점 lip_xy 는 0.3~0.5 m 라 `safe` 는 사실상 항상 0 이고,
   벌점은 "조준했냐/안 했냐"의 이진값이 된다. **주둥이를 입구에 가깝게 가져가도 벌점이 줄지 않는다** — 조준 쪽으로 움직일 경사가 없다.
3. **가중이 작다.** 1.5 는 success_bonus(블록 평균 최대 6.5) 대비 작다.

참고: `lead_gate`(L336, LEAD_FULL/ZERO 0.050/0.140)는 이미 `tilt_near` 수입에 거리 경사를 준다. 문제는 벌점 쪽이다.

## 4. 다음 보상이 해야 할 것

1. ★**조준 요구를 틸팅 앞으로 당긴다.** 벌점 창을 방출각 −5° 가 아니라 **theta_pre**(가득 찬 컵 55°) 부근부터 켠다.
   비드가 나오기 시작하는 각(가득 찬 컵 약 70°)에 닿기 **전에** 조준이 끝나 있어야 한다.
2. ★**조준에 경사를 준다.** 이진 `safe` 대신 lip_xy 에 연속으로 반응하게 해서, 주둥이가 입구에 가까워질수록 벌점이 줄게 한다.
   거리 척도는 실측 운영점(0.2~0.5 m)에서 경사가 살아 있어야 한다.
3. 가중을 올리되, **붓기 자체를 포기하게 만들지 않는다.** iter_10 에서 in_source 가 0.40 → 0.58 로 올라갔다. 조준 없이 기울이는 것이
   비싸지는 만큼 **조준하고 기울이는 것**의 수입이 확실히 커야 한다.
4. **속도·제어 벌점(src_speed·src_spin·rot_pour·rcv_spin·act_flip·hand_flip·carry_rate·overtilt·flow_excess·spill 240)은 그대로 둔다.** 사용자 원칙.

## 5. 건드리지 말 것

- 파지·들기·converge·align·방향(signed tilt)·기울기 연동 높이 예산 — 성립했다.
- 제어 안정 장치 — iter_10 에서 step_delta 를 0.437 → 0.365 로 내린 게 이것들이다.
- env 무변경. `entropy_coef` 0 유지. 트랙 전용 컨텍스트에 없는 필드(adr_progress 등) 사용 금지.
- 웜스타트는 **iter_10 ep100**(흘림/유출 0.224·효율 0.746, 이 라운드 최저 흘림 지점)이다.
