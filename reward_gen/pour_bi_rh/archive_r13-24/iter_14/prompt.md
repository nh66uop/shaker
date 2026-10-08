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
    # iter_13: the lip point in 3-D and the xy unit vector receiver mouth -> lip, for the closing rate
    lip_u = lip_xy_vec / torch.clamp(lip_xy, min=1e-6).unsqueeze(-1)
    lip_pt = torch.cat([ctx.rcv_cup_mouth_pos[:, :2] + lip_xy_vec,
                        (ctx.src_cup_mouth_pos[:, 2] - lip_r * torch.sin(theta)).unsqueeze(-1)], dim=-1)
    return {"d_hat_x": d_hat[:, 0], "sep_ok": sep_ok, "theta": theta, "theta_eff": theta_eff,
            "mis": mis, "lip_xy": lip_xy, "lip_dz": lip_dz, "d_align": d_align,
            "sin_t": sin_t, "lip_drop": lip_drop, "lip_u": lip_u, "lip_pt": lip_pt}


def _lip_closing_rate(ctx, lip_u, lip_pt):
    """iter_13: d(lip_xy)/dt from the rigid-body velocities (stateless potential difference).
    The lip is taken as a point fixed on the source cup, the receiver mouth as a point fixed on the
    receiver cup. Negative = the spout is closing on the receiver mouth. [m/s]"""
    v_lip = ctx.src_cup_lin_vel + torch.cross(ctx.src_cup_ang_vel, lip_pt - ctx.src_cup_pos, dim=-1)
    v_mouth = ctx.rcv_cup_lin_vel + torch.cross(ctx.rcv_cup_ang_vel,
                                                ctx.rcv_cup_mouth_pos - ctx.rcv_cup_pos, dim=-1)
    return ((v_lip[:, :2] - v_mouth[:, :2]) * lip_u).sum(dim=-1)


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
    # iter_11 — AIM BEFORE TILT. iter_10 got the slower, smoother control the real robot needs
    # (step_delta 0.437 -> 0.365) but after ep100 spill/outflow went 0.224 -> 0.471 AT THE SAME rotation
    # speed; diag_lip_xy drifted 0.30 -> 0.49 m. A slowly turning cup starts losing beads part-way
    # through the turn, so the aim has to be finished before the first-bead angle (~70 deg full).
    # Only the unaimed_release block changes, plus one paired income (aimed_tilt). EVERY speed /
    # control charge (src_speed, src_spin, rot_pour, rcv_spin, act_flip, hand_flip, carry_rate,
    # overtilt, flow_excess) and spill_delta W 240 are VERBATIM iter_10.
    #   a. window moves from [release-5, release+5] (70-80 deg full) to [theta_pre-5, theta_pre+10]
    #      (50-65 deg full): it saturates BEFORE the first bead leaves
    #   b. "aimed" is no longer the binary safe gate (0.03-0.06 m, ~always 0 at the 0.2-0.5 m operating
    #      point) but a continuous aim_q(lip_xy) whose slope is alive from the mouth out to 0.6 m
    #   c. W_UNAIMED 1.5 -> 3.0, and aimed_tilt pays W 2.5 * aim_q over the same window, so going from
    #      unaimed to aimed while tilted is worth 5.5 per step, and aimed tilting pays MORE than iter_10
    # iter_12 — CONTROL THAT THE ARM CAN FOLLOW. Trace (i11 ep150/ep350): src wrist joints inside the
    # lower 5 % of their range 46-62 % of steps, src palm rotation error 89-98 deg, src palm |a|>0.95 on
    # 43 % of dim-steps (y 0.77); rcv palm y flips sign on 72-77 % of steps. palm_sat (-0.05/episode)
    # and act_flip (ctrl_gate only) never reached these. There are no joint-position fields, so the
    # handles are palm action saturation and palm sign alternation. Aim, speed, flow, spill, tilt,
    # grasp and lift terms are VERBATIM iter_11 (user rule: never buy SIM efficiency with control).
    #   a. palm_sat_src / palm_sat_rcv: gate = that hand's grasp only (was both grasped AND lifted),
    #      W 0.15 -> 3.0 (src) / 1.5 (rcv). src op point raw 0.52 -> about -1.1 per step, a fifth of
    #      the success income (5.2 mean), removable almost for free by backing the command off the rail.
    #   b. palm_bang_src / palm_bang_rcv (new): fraction of the 6 palm dims whose command flips sign
    #      with |a| > 0.1 on both steps (the trace's own definition), gated on that hand's grasp, W 1.5.
    #      rcv op point 0.375 -> -0.56 per step.
    #   c. act_flip and the palm half of carry_rate: same weights, per-arm grasp gate instead of
    #      ctrl_gate, so bang-bang before the lift / while only one cup is held is charged too.
    # iter_13 — APPROACH AND PERMISSION SEPARATED; the window is cup geometry, precision is the outcome.
    # i10/i11/i12 all collapsed the same way: rotation speed unchanged, lip_xy 0.19 -> 0.44 m, success
    # 0.48 -> 0.18, and aimed_tilt income ROSE while the aim got worse (0.249 -> 0.364), because aim_q's
    # 0.60 m linear tail still paid 0.38 at 0.2 m. One scale was doing two jobs (pull toward the mouth
    # AND licence to tilt). Now:
    #   a. approach_lip (new): stateless potential difference on lip_xy = -d(lip_xy)/dt from the
    #      rigid-body velocities, tilt-independent, flat (0) once the lip is inside the window. Standing
    #      still pays 0, so it cannot become a parking reward; back-and-forth sums to ~0.
    #   b. ONE permission window perm(lip_xy) in multiples of ctx.cup_radius (1 r full, 3 r zero) times
    #      lip_above_raw gates EVERY income paid past theta_pre (aimed_tilt, tilt_near, pour_tilt,
    #      pour_pose) and the corridor's release stage. It is FLAT inside 1 r: no mm-shaped preference
    #      inside the mouth. aim_q, AIM_FREE/NEAR/FAR/Z_FLOOR, AIM_XY_K, LIP_DEADBAND and SAFE_* are gone.
    #   c. unaimed_release charges (1 - perm) instead of (1 - aim_q): tilting toward release outside the
    #      window now costs the full W 3.0 at 0.2 m (was ~0.6 x 3.0).
    #   d. precision inside the window is left to the bead outcome: the bead-releasing incomes are
    #      scaled by clean_f (1 - spill/0.40, the same factor success/transferred already use) instead
    #      of beads_left (1 - spill). One spilled bead now deletes 1/8 of them for the rest of the episode.
    #   e. lead_gate re-expressed in cup radii (1.75 r / 5 r = 0.049 / 0.140 m, numerically the old
    #      0.05 / 0.14), still only for tilt_aim / corridor pre-release stage / pour_phase.
    # Grasp, lifts, converge, carry, align, signed tilt, lift budget, every speed / control charge,
    # flow_excess, overtilt, spill 240, success / transferred and the iter_12 per-arm control terms are
    # VERBATIM iter_12.
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
    # ★ iter_13: every lip-distance window is written in RECEIVER-MOUTH RADII (ctx.cup_radius), not mm.
    # PERMISSION window for everything paid past theta_pre: 1 at lip_xy <= 1 r (the lip is over the
    # mouth opening itself), 0 at >= 3 r (one full mouth diameter outside the rim). Deliberately
    # generous and FLAT inside: where inside it the beads actually land is taught by pour_delta /
    # spill_delta / clean_f, not by a hand-tuned mm profile. Replaces SAFE_* (0.03/0.06 m) and aim_q.
    PERM_IN_R, PERM_OUT_R = 1.0, 3.0
    # lead zone for the PRE-release stage only (tilt_aim floor, corridor theta_pre -> release,
    # pour_phase). 1.75 r / 5 r = 0.049 / 0.140 m: the iter_08-12 values 0.05 / 0.14 re-expressed.
    LEAD_FULL_R, LEAD_ZERO_R = 1.75, 5.0
    # ★ iter_13 approach: potential difference on lip_xy, paid per step as closing speed / SRC_V_FREE
    # (clamped to +-1, so closing faster than the free cup speed buys nothing and src_speed starts
    # charging). W 1.0 = the alive income at the free speed; below tilt_near (2.1 measured) and
    # far below success (20), so arriving can never outbid pouring. Being still pays exactly 0.
    W_APPROACH = 1.0
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
    # ★ iter_11: window anchored on theta_pre (release - 20 deg = 55 deg full), not on release.
    # On at theta_pre - 5 (50 deg full), saturated at theta_pre + 10 (65 deg full) — 5 deg before the
    # first bead leaves a full cup (~70 deg). Follows fill exactly as theta_pre does.
    UNAIMED_BELOW = math.radians(5.0)
    UNAIMED_SPAN = math.radians(15.0)
    W_UNAIMED = 3.0         # unchanged weight; iter_13 charges x (1 - perm) instead of x (1 - aim_q), so
                            # at 0.2 m it is the full 3.0 (was 0.62 x 3.0) and 0 inside 1 r
    # iter_13: aim_q (AIM_FREE 0.025 / AIM_NEAR 0.06 / AIM_FAR 0.60 / AIM_Z_FLOOR 0.3) removed — its
    # 0.60 m tail paid aimed_tilt 0.38 at 0.2 m, and aimed_tilt income rose while lip_xy doubled.
    W_AIMED_TILT = 2.5      # unchanged; now x perm x clean_f. Below tilt_near 6 / pour_tilt 10 so the
                            # 65 deg plateau does not beat actually pouring
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
    W_FLIP_PALM = 0.3       # x sum over the 6 palm dims of relu(-a*a_prev), per arm, gated on THAT
                            # arm's grasp (iter_12; was ctrl_gate over both arms)
    W_FLIP_HAND = 0.2       # x sum over the 12 hand dims, ctrl_gate (unchanged)
    W_RATE_PALM_CARRY = 0.03   # x sum over 6 palm dims of da^2, per arm, gated on that arm's grasp
                               # (iter_12; was ctrl_gate). Base action_rate kept.
    # ★ iter_12 palm sign alternation (bang-bang). Rate in [0, 1] = fraction of the arm's 6 palm dims
    # whose command changes sign with |a| > BANG_DEAD on both steps. Normalised on the success state:
    # rcv ep150 per-dim rates [0.45 0.77 0.27 0.13 0.46 0.17] -> 0.375 mean.
    BANG_DEAD = 0.1
    W_BANG_SRC, W_BANG_RCV = 1.5, 1.5   # -0.56 per step at the rcv operating point while holding;
                                        # a quarter of the pour_tilt + tilt_near income, so smoothing
                                        # is worth doing but releasing the cup never is
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
    # ★ iter_12: that version stayed flat (diag_palm_sat 0.52, -0.05 per episode). Gate is now the
    # arm's own grasp (warm start: the grasp already exists, a fresh policy still cannot be frozen
    # before it grasps), and the weight is sized against the success income (5.2 per step mean).
    # SAT_LO 0.7 kept: the operating point is 0.9-1.0, so the smoothstep slope is live there.
    SAT_LO = 0.7
    W_PALM_SAT_SRC = 3.0    # 0.15 -> 3.0: raw 0.52 x grasp 0.73 -> about -1.1 per step
    W_PALM_SAT_RCV = 1.5    # 0.15 -> 1.5: rcv raw ~0.39 x grasp 0.78 -> about -0.46 per step
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
    # ★ iter_13: both windows in receiver-mouth radii (r = ctx.cup_radius), no absolute mm constants
    r_mouth = ctx.cup_radius
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r_mouth)
                                  / ((LEAD_ZERO_R - LEAD_FULL_R) * r_mouth))
    perm_xy = 1.0 - _smoothstep((geo["lip_xy"] - PERM_IN_R * r_mouth)
                                / ((PERM_OUT_R - PERM_IN_R) * r_mouth))
    # ★ the height budget follows the tilt. lip_free cancels the r*sin already inside lip_dz, so this
    # charge asks "is the cup absurdly high for an UPRIGHT pour", not "did you dare to rotate".
    lip_free = LIP_Z_FREE0 + ctx.cup_radius * geo["sin_t"]
    lip_above_raw = torch.clamp((geo["lip_dz"] - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(geo["lip_dz"] - lip_free, min=0.0) / LIP_Z_SIGMA) ** 2)
    lip_above = LIP_ABOVE_FLOOR + (1.0 - LIP_ABOVE_FLOOR) * lip_above_raw     # gating (keeps a floor)
    z_ok = lip_above_raw * lip_high                                          # paying (no floor)
    # ★ iter_13 PERMISSION: lip over the mouth (within the r-window) AND above the receiver rim. Hard
    # (no floor): a bead that leaves below the rim or outside 3 r cannot land in the cup.
    perm = perm_xy * lip_above_raw
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
    # iter_13: the last stage opens on perm_xy (1 r .. 3 r) instead of safe (0.03 .. 0.06 m).
    carry_soft_src = g_src * torch.clamp(h_src / TILT_LIFT_H, max=1.0)
    allowed = TILT_FREE_TABLE + carry_soft_src * torch.clamp(theta_pre - TILT_FREE_TABLE, min=0.0)
    allowed = allowed + carry_soft_src * lead_gate * torch.clamp(release_tilt - theta_pre, min=0.0)
    allowed = allowed + carry_soft_src * perm_xy * (theta_max - release_tilt)
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
    # ★ iter_13 outcome link: everything paid past theta_pre (where beads can leave) is scaled by
    # clean_f = 1 - spill/0.40 (the factor success/transferred already carry) instead of beads_left.
    # Inside the permission window, WHERE the lip should be is decided by where beads actually landed.
    spill_lvl = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    clean_f = torch.clamp(1.0 - spill_lvl / SPILL_ZERO, 0.0, 1.0)   # 1 clean, 0 at the success limit
    release_hold = carry_soft_src * clean_f * not_nested * geo["sep_ok"] * dir_soft
    lead_dist = LEAD_FLOOR + (1.0 - LEAD_FLOOR) * lead_gate     # ★ the real spout distance now, not a
                                                                # cup-xy factor pinned at 0.985
    aim_frac = torch.clamp(theta_eff / theta_pre, 0.0, 1.0)                                  # 0 -> 55 deg
    near_frac = torch.clamp((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.05),
                            0.0, 1.0)                                                        # 55 -> 75
    pour_frac = torch.clamp((theta_eff - release_tilt) / (theta_max - release_tilt), 0.0, 1.0)  # 75 -> 105
    # 0 -> theta_pre: no bead can leave, the lead zone (5 r) is enough and it keeps its floor
    tilt_aim = tilt_hold * aim_frac * lead_dist
    # ★ theta_pre -> release (the first bead leaves inside this segment): PERMISSION only
    tilt_near = release_hold * near_frac * perm
    # the release segment: only with the spout actually over the receiver mouth and above its rim
    pour_tilt = both_up * clean_f * dir_soft * upright_rcv * perm * pour_frac
    # plateau at the release attitude so holding the pose beats swinging through it (10-step streak).
    # iter_13: aim_xy (exp(-12 (lip_xy - 0.02))) removed — flat inside the window, the outcome decides.
    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = both_up * clean_f * dir_ok * upright_rcv * perm_xy * z_ok * pour_band

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
    # [12:18] rcv palm, [18:24] rcv hand. ★ iter_12: each arm is gated on ITS OWN grasp flag (no lift
    # needed) — the saturated commands and the joint-limit parking happen from the grasp onward.
    a_abs = ctx.actions.abs()
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
    # (spill_lvl / clean_f are computed above, next to the tilt ladder that now uses them)

    # ---- aiming at the moment of release: tilt MAGNITUDE past (release - 5 deg) while the spout is
    #      not inside the r-window (perm) or not above the rim. 0 for a correct pour pose. -----------------
    has_beads = torch.clamp(fill / 0.10, max=1.0)               # >= 2 beads left
    # charge on the tilt MAGNITUDE (beads do not care about direction), window before the first bead.
    # ★ iter_13: outside the permission window (1 - perm), not (1 - aim_q) with its 0.60 m tail.
    pre_release = _smoothstep((ctx.src_cup_tilt - (theta_pre - UNAIMED_BELOW)) / UNAIMED_SPAN)
    unaimed = carry_soft_src * has_beads * pre_release * (1.0 - perm)
    # paired income on the SIGNED angle, same window, both cups held and lifted: aimed tilting is paid,
    # a sideways tilt only pays the charge. ★ iter_13: x perm x clean_f (was x aim_q).
    aimed_win = _smoothstep((theta_eff - (theta_pre - UNAIMED_BELOW)) / UNAIMED_SPAN)
    aimed_tilt = both_up * has_beads * clean_f * dir_soft * upright_rcv * aimed_win * perm

    # ---- ★ iter_13 approach: stateless potential difference on lip_xy, independent of the tilt ------
    # Phi = max(lip_xy, 1 r); paid = -dPhi/dt / SRC_V_FREE, clamped to [-1, 1]. Positive while the spout
    # closes on the mouth, negative while it backs away, 0 when still and 0 once inside the window.
    lip_rate = _lip_closing_rate(ctx, geo["lip_u"], geo["lip_pt"])
    outside_win = (geo["lip_xy"] > PERM_IN_R * r_mouth).to(dtype)
    approach_gate = carry_soft_src * rcv["held"] * not_nested * has_beads
    approach_lip = approach_gate * outside_win * torch.clamp(-lip_rate / SRC_V_FREE, -1.0, 1.0)

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
    flip_palm_src = flips[:, 0:6].sum(dim=-1)
    flip_palm_rcv = flips[:, 12:18].sum(dim=-1)
    flip_palm = flip_palm_src + flip_palm_rcv
    flip_hand = flips[:, 6:12].sum(dim=-1) + flips[:, 18:24].sum(dim=-1)
    rate_palm_src = (da[:, 0:6] ** 2).sum(dim=-1)
    rate_palm_rcv = (da[:, 12:18] ** 2).sum(dim=-1)
    rate_hand = (da[:, 6:12] ** 2).sum(dim=-1) + (da[:, 18:24] ** 2).sum(dim=-1)
    # ★ iter_12 bang-bang rate: sign change with |a| > BANG_DEAD on both steps, fraction of 6 dims
    both_big = (a_abs > BANG_DEAD) & (ctx.prev_actions.abs() > BANG_DEAD)
    bang = (both_big & (ctx.actions * ctx.prev_actions < 0.0)).to(dtype)     # (N, 24)
    bang_src = bang[:, 0:6].mean(dim=-1)
    bang_rcv = bang[:, 12:18].mean(dim=-1)

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
        "aimed_tilt": W_AIMED_TILT * aimed_tilt,
        "approach_lip": W_APPROACH * approach_lip,
        "overtilt": -overtilt,
        "src_speed": -W_SRC_SPEED * src_speed,
        "src_spin": -W_SRC_SPIN * src_spin,
        "rot_pour": -W_ROT_POUR * rot_pour,
        "rcv_spin": -W_RCV_SPIN * rcv_spin,
        "act_flip": -W_FLIP_PALM * (g_src * flip_palm_src + g_rcv * flip_palm_rcv),
        "hand_flip": -W_FLIP_HAND * ctrl_gate * flip_hand,
        "carry_rate": -(W_RATE_PALM_CARRY * (g_src * rate_palm_src + g_rcv * rate_palm_rcv)
                        + W_RATE_HAND_CARRY * ctrl_gate * rate_hand),
        "palm_bang_src": -W_BANG_SRC * g_src * bang_src,
        "palm_bang_rcv": -W_BANG_RCV * g_rcv * bang_rcv,
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
        "palm_sat_src": -W_PALM_SAT_SRC * g_src * sat_src_raw,
        "palm_sat_rcv": -W_PALM_SAT_RCV * g_rcv * sat_rcv_raw,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    # diagnostics only (added AFTER the sum, not part of the reward). One per fix, so the next round
    # can tell which of the four moved: the moving lift budget, the new distance factor, the raw palm
    # saturation, and the signed / misdirected split of the tilt.
    comps["rcv_progress"] = rcv_prog
    comps["aim_gate"] = perm_xy          # iter_13: the r-window (was safe 0.03/0.06 m)
    comps["diag_perm"] = perm            # window x lip above the rim (replaces diag_aim_q)
    comps["diag_pre_release"] = pre_release
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
    # iter_12 diagnostics: raw (ungated) control quality per arm
    comps["diag_palm_sat_rcv"] = sat_rcv_raw
    comps["diag_bang_src"] = bang_src
    comps["diag_bang_rcv"] = bang_rcv
    # iter_13 diagnostics: is the tilt toward release happening INSIDE the window, and how fast does
    # the spout close on the mouth (m/s, + = closing, clamped to 1 m/s against velocity spikes)
    comps["diag_perm_at_tilt"] = pre_release * perm
    comps["diag_unperm_at_tilt"] = pre_release * (1.0 - perm)
    comps["diag_close_speed"] = torch.clamp(-lip_rate, -1.0, 1.0)
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0.0667, 0.167, 0.233, 0.333, 0.433, 0.533, 0.6, 0.7, 0.8]  min 0 · mean 0.43 · max 0.9
bead/in_target: [0, 0.322, 0.382, 0.343, 0.471, 0.386, 0.423, 0.431, 0.391, 0.376]  min 0 · mean 0.385 · max 0.554
bead/spill: [0, 0.155, 0.135, 0.147, 0.139, 0.141, 0.132, 0.116, 0.116, 0.112]  min 0 · mean 0.133 · max 0.227
done/drop: [0, 0.000244, 0.000244, 0.000488, 0, 0.000488, 0, 0, 0.000244, 0.000488]  min 0 · mean 0.000391 · max 0.0022
done/mimic_runaway: [0, 0.000244, 0, 0.000732, 0, 0.000244, 0.000488, 0.000244, 0.000244, 0.000732]  min 0 · mean 0.000369 · max 0.00171
episode_lengths/step: [120, 573, 585, 631, 668, 571, 613, 630, 632, 599]  min 120 · mean 615 · max 706
reward/act_flip: [0, -0.191, -0.176, -0.146, -0.144, -0.146, -0.151, -0.147, -0.121, -0.132]  min -0.292 · mean -0.153 · max 0
reward/action_rate: [-0.0147, -0.0172, -0.016, -0.0151, -0.0152, -0.0156, -0.0158, -0.0162, -0.0148, -0.0156]  min -0.0213 · mean -0.0159 · max -0.0144
reward/aim_gate: [0, 0.465, 0.557, 0.5, 0.558, 0.497, 0.584, 0.583, 0.564, 0.59]  min 0 · mean 0.544 · max 0.693
reward/aimed_tilt: [0, 0.267, 0.266, 0.306, 0.165, 0.241, 0.232, 0.225, 0.252, 0.264]  min 0 · mean 0.244 · max 0.611
reward/align_mouths: [0, 2.09, 2.46, 2.4, 2.6, 2.34, 2.46, 2.36, 2.44, 2.35]  min 0 · mean 2.35 · max 2.7
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/approach_lip: [0, 0.0185, 0.021, 0.0232, 0.0219, 0.0179, 0.0216, 0.00946, -0.00456, -0.00585]  min -0.0124 · mean 0.0163 · max 0.277
reward/away_rcv: [-0.0542, -0.162, -0.163, -0.164, -0.163, -0.163, -0.172, -0.17, -0.175, -0.173]  min -0.177 · mean -0.166 · max -0.0542
reward/carry_rate: [0, -0.134, -0.129, -0.112, -0.116, -0.112, -0.117, -0.117, -0.102, -0.107]  min -0.195 · mean -0.117 · max 0
reward/carry_src: [0, 1.22, 1.3, 1.31, 1.37, 1.28, 1.4, 1.33, 1.4, 1.34]  min 0 · mean 1.31 · max 1.56
reward/close_rcv: [0.1, 0.0835, 0.0824, 0.0835, 0.0836, 0.088, 0.079, 0.0828, 0.0781, 0.0782]  min 0.0712 · mean 0.0818 · max 0.1
reward/close_src: [0.0156, 0.0173, 0.0194, 0.021, 0.0191, 0.0195, 0.0206, 0.0211, 0.021, 0.0219]  min 0.0143 · mean 0.0197 · max 0.0249
reward/contact_push: [-0.00552, -0.00881, -0.0083, -0.00815, -0.00759, -0.00941, -0.0105, -0.00887, -0.0091, -0.0094]  min -0.0323 · mean -0.00883 · max -0.00188
reward/converge: [0, 1.63, 1.74, 1.74, 1.83, 1.7, 1.75, 1.69, 1.72, 1.66]  min 0 · mean 1.69 · max 1.94
reward/cup_collision: [0, -0.00121, -0.0109, -0.0046, -0.00984, -0.00268, -0.00337, -0.00364, -0.00985, -0.00673]  min -0.0428 · mean -0.00529 · max 0
reward/cup_disturb_rcv: [-0.02, -0.112, -0.116, -0.112, -0.11, -0.106, -0.0972, -0.0978, -0.0919, -0.0929]  min -0.311 · mean -0.104 · max -0.02
reward/cup_disturb_src: [-0.116, -0.0359, -0.0387, -0.0461, -0.037, -0.0434, -0.0445, -0.044, -0.0407, -0.039]  min -0.116 · mean -0.041 · max -0.0118
reward/cup_fling: [-0.000617, -0.0189, -0.0187, -0.0202, -0.0156, -0.0185, -0.0172, -0.0157, -0.0194, -0.019]  min -0.0596 · mean -0.0178 · max -0.000617
reward/diag_bang_rcv: [0.21, 0.259, 0.245, 0.231, 0.221, 0.23, 0.227, 0.23, 0.212, 0.217]  min 0.202 · mean 0.233 · max 0.347
reward/diag_bang_src: [0.0722, 0.0889, 0.104, 0.0959, 0.0936, 0.086, 0.0726, 0.066, 0.0581, 0.0636]  min 0.0517 · mean 0.081 · max 0.125
reward/diag_clean: [1, 0.697, 0.729, 0.702, 0.706, 0.707, 0.718, 0.755, 0.762, 0.768]  min 0.546 · mean 0.726 · max 1
reward/diag_close_speed: [0.0381, 0.00341, 0.0115, 0.00856, 0.00421, -0.00116, 0.0119, 0.0102, 0.00439, 0.0054]  min -0.00858 · mean 0.00925 · max 0.136
reward/diag_ctrl_gate: [0, 0.626, 0.674, 0.649, 0.709, 0.633, 0.674, 0.656, 0.654, 0.64]  min 0 · mean 0.651 · max 0.777
reward/diag_flip_palm: [0.386, 0.736, 0.682, 0.58, 0.556, 0.577, 0.584, 0.583, 0.49, 0.531]  min 0.386 · mean 0.601 · max 1.07
reward/diag_flow_beads: [0, 0.0137, 0.0142, 0.0142, 0.0205, 0.0181, 0.021, 0.0249, 0.0156, 0.0142]  min 0 · mean 0.0164 · max 0.0554
reward/diag_lead_gate: [0, 0.623, 0.674, 0.641, 0.707, 0.653, 0.728, 0.725, 0.713, 0.72]  min 0 · mean 0.687 · max 0.823
reward/diag_lift_budget: [0.241, 0.302, 0.305, 0.303, 0.312, 0.301, 0.303, 0.303, 0.302, 0.3]  min 0.241 · mean 0.303 · max 0.314
reward/diag_lip_xy: [0.288, 0.34, 0.182, 0.573, 0.428, 0.618, 0.204, 0.628, 0.311, 0.352]  min 0.114 · mean 0.419 · max 1.18
reward/diag_mis_deg: [0.928, 1.47, 1.69, 2.3, 1.31, 1.58, 1.82, 1.74, 1.91, 1.75]  min 0.928 · mean 1.81 · max 12.5
reward/diag_palm_sat: [0.328, 0.477, 0.431, 0.407, 0.428, 0.399, 0.427, 0.457, 0.496, 0.502]  min 0.322 · mean 0.454 · max 0.557
reward/diag_palm_sat_rcv: [0.364, 0.396, 0.367, 0.351, 0.356, 0.343, 0.348, 0.337, 0.352, 0.349]  min 0.324 · mean 0.356 · max 0.451
reward/diag_perm: [0, 0.459, 0.501, 0.478, 0.503, 0.491, 0.57, 0.564, 0.514, 0.55]  min 0 · mean 0.517 · max 0.638
reward/diag_perm_at_tilt: [0, 0.449, 0.489, 0.463, 0.493, 0.441, 0.505, 0.492, 0.444, 0.461]  min 0 · mean 0.471 · max 0.625
reward/diag_pour_phase: [0, 0.548, 0.608, 0.575, 0.642, 0.569, 0.617, 0.599, 0.592, 0.579]  min 0 · mean 0.586 · max 0.724
reward/diag_pre_release: [0, 0.647, 0.685, 0.654, 0.718, 0.598, 0.637, 0.636, 0.631, 0.606]  min 0 · mean 0.642 · max 0.825
reward/diag_src_v: [0.0486, 0.352, 0.311, 0.498, 0.349, 0.459, 0.317, 0.438, 0.46, 0.509]  min 0.0486 · mean 1.25 · max 388
reward/diag_src_w: [1.01, 2.53, 2.22, 2.41, 2.62, 2.34, 2.3, 2.27, 5.56, 5.53]  min 1.01 · mean 29 · max 1.23e+04
reward/diag_theta_deg: [1.99, 69, 72.6, 70, 79.8, 68.3, 70.4, 70.8, 68.9, 67.1]  min 1.99 · mean 69.9 · max 81.2
reward/diag_unperm_at_tilt: [0, 0.198, 0.195, 0.191, 0.224, 0.158, 0.133, 0.144, 0.187, 0.145]  min 0 · mean 0.171 · max 0.416
reward/early_tilt: [-0.00338, -0.108, -0.114, -0.136, -0.115, -0.125, -0.108, -0.11, -0.102, -0.0959]  min -0.21 · mean -0.114 · max -0.00338
reward/flow_excess: [0, -0.00146, -0.00146, -0.00195, -0.00439, -0.00391, -0.00391, -0.00586, -0.00146, -0.00146]  min -0.0083 · mean -0.00212 · max 0
reward/grasp_rcv: [0.000249, 1.04, 1.04, 1.05, 1.07, 1.08, 1.19, 1.19, 1.23, 1.19]  min 0.000249 · mean 1.1 · max 1.27
reward/grasp_src: [0.000179, 1.17, 1.28, 1.25, 1.32, 1.15, 1.24, 1.2, 1.19, 1.18]  min 0.000179 · mean 1.21 · max 1.38
reward/hand_flip: [0, -0.0634, -0.0618, -0.0536, -0.0624, -0.0577, -0.0626, -0.0695, -0.0648, -0.0698]  min -0.0964 · mean -0.0631 · max 0
reward/hand_foreign: [-0.0108, -0.00393, -0.00372, -0.00415, -0.00386, -0.00309, -0.00473, -0.00353, -0.00368, -0.0038]  min -0.0244 · mean -0.00413 · max -0.00206
reward/hold_rcv: [0, 0.191, 0.192, 0.198, 0.217, 0.222, 0.238, 0.242, 0.227, 0.224]  min 0 · mean 0.22 · max 0.276
reward/lift_rcv: [0, 1.17, 1.21, 1.2, 1.16, 1.18, 1.27, 1.26, 1.29, 1.26]  min 0 · mean 1.21 · max 1.43
reward/lift_src: [0, 2.28, 2.39, 2.34, 2.49, 2.21, 2.33, 2.28, 2.34, 2.32]  min 0 · mean 2.31 · max 2.66
reward/mis_tilt: [0, -0.0111, -0.0148, -0.0188, -0.0107, -0.00905, -0.015, -0.0128, -0.0137, -0.0131]  min -0.135 · mean -0.0134 · max 0
reward/near_nested: [0, -0.00403, -0.0151, -0.00834, -0.00988, -0.00551, -0.00833, -0.0111, -0.0146, -0.0169]  min -0.0725 · mean -0.0108 · max 0
reward/nested: [0, -0.00488, -0.00732, -0.00586, -0.00635, -0.00635, -0.00879, -0.00977, -0.00635, -0.0103]  min -0.0195 · mean -0.00716 · max 0
reward/over_lift_src: [0, -0.0422, -0.0236, -0.0289, -0.0189, -0.0469, -0.0433, -0.0454, -0.0115, -0.0148]  min -0.139 · mean -0.0317 · max 0
reward/overtilt: [0, -0.0124, -0.00991, -0.0117, -0.0296, -0.0264, -0.0248, -0.0331, -0.0192, -0.0341]  min -0.0666 · mean -0.0208 · max 0
reward/palm_bang_rcv: [0, -0.32, -0.305, -0.279, -0.275, -0.278, -0.279, -0.277, -0.257, -0.262]  min -0.466 · mean -0.285 · max 0
reward/palm_bang_src: [0, -0.0953, -0.12, -0.11, -0.113, -0.0957, -0.0842, -0.0689, -0.0567, -0.0638]  min -0.155 · mean -0.0889 · max 0
reward/palm_sat_rcv: [0, -0.444, -0.424, -0.396, -0.425, -0.379, -0.403, -0.376, -0.399, -0.388]  min -0.479 · mean -0.403 · max 0
reward/palm_sat_src: [0, -1.14, -1.02, -0.946, -1.05, -0.873, -0.987, -1.04, -1.14, -1.14]  min -1.48 · mean -1.05 · max 0
reward/pour_delta: [0, 0.0952, 0.0952, 0.0897, 0.135, 0.114, 0.132, 0.165, 0.108, 0.097]  min 0 · mean 0.109 · max 0.403
reward/pour_dir_pen: [0, -0.0549, -0.0692, -0.049, -0.0678, -0.062, -0.123, -0.135, -0.116, -0.115]  min -0.262 · mean -0.0947 · max 0
reward/pour_pose: [0, 0.479, 0.459, 0.435, 0.573, 0.484, 0.469, 0.526, 0.495, 0.491]  min 0 · mean 0.459 · max 1
reward/pour_tilt: [0, 0.608, 0.571, 0.548, 0.769, 0.667, 0.654, 0.751, 0.652, 0.688]  min 0 · mean 0.612 · max 1.48
reward/rcv_home: [0, -0.11, -0.077, -0.0651, -0.0348, -0.0466, -0.0628, -0.0747, -0.0793, -0.0908]  min -0.16 · mean -0.0744 · max 0
reward/rcv_progress: [0.279, 0.743, 0.752, 0.743, 0.759, 0.736, 0.75, 0.75, 0.742, 0.738]  min 0.279 · mean 0.745 · max 0.839
reward/rcv_spin: [0, -0.132, -0.156, -0.146, -0.156, -0.126, -0.129, -0.113, -0.132, -0.125]  min -0.16 · mean -0.132 · max 0
reward/rcv_still: [0, -0.376, -0.401, -0.392, -0.4, -0.36, -0.362, -0.341, -0.363, -0.346]  min -0.461 · mean -0.363 · max 0
reward/reach_rcv: [0.897, 0.813, 0.815, 0.813, 0.815, 0.821, 0.8, 0.809, 0.792, 0.798]  min 0.786 · mean 0.81 · max 0.914
reward/reach_src: [0.526, 0.525, 0.513, 0.53, 0.518, 0.541, 0.536, 0.539, 0.53, 0.542]  min 0.503 · mean 0.53 · max 0.599
reward/rot_pour: [0, -0.192, -0.245, -0.221, -0.244, -0.208, -0.24, -0.211, -0.222, -0.196]  min -0.269 · mean -0.214 · max 0
reward/rush_near_cup: [-0.223, -0.0374, -0.0346, -0.0372, -0.0324, -0.0407, -0.0346, -0.0374, -0.0374, -0.0383]  min -0.223 · mean -0.0371 · max -0.0194
reward/spill_delta: [0, -0.155, -0.132, -0.138, -0.141, -0.152, -0.158, -0.0879, -0.0645, -0.103]  min -0.275 · mean -0.129 · max 0
reward/src_speed: [0, -0.0972, -0.129, -0.117, -0.132, -0.102, -0.107, -0.0918, -0.0958, -0.0892]  min -0.203 · mean -0.105 · max 0
reward/src_spin: [0, -0.0863, -0.109, -0.101, -0.104, -0.0911, -0.101, -0.0883, -0.095, -0.0822]  min -0.119 · mean -0.0923 · max 0
reward/success_bonus: [0, 3.94, 5.23, 4.24, 6.63, 5.06, 5.93, 6.42, 5.81, 5.66]  min 0 · mean 5.34 · max 7.17
reward/tilt_aim: [0, 0.862, 0.936, 0.896, 0.971, 0.851, 0.858, 0.852, 0.91, 0.899]  min 0 · mean 0.878 · max 1.02
reward/tilt_near: [0, 1.57, 1.71, 1.59, 1.68, 1.46, 1.54, 1.6, 1.55, 1.52]  min 0 · mean 1.54 · max 2
reward/total: [2.09, 17.3, 19.7, 18.6, 22.2, 19.5, 21, 21.6, 20.9, 20.4]  min 2.09 · mean 19.7 · max 23.2
reward/transferred: [0, 0.603, 0.795, 0.634, 0.996, 0.768, 0.954, 1.03, 0.992, 0.955]  min 0 · mean 0.844 · max 1.11
reward/unaimed_release: [0, -0.173, -0.221, -0.154, -0.133, -0.0802, -0.0753, -0.106, -0.171, -0.153]  min -1.15 · mean -0.145 · max 0
rewards/step: [134, 1.01e+04, 1.17e+04, 1.21e+04, 1.44e+04, 1.23e+04, 1.25e+04, 1.34e+04, 1.31e+04, 1.25e+04]  min 134 · mean 1.21e+04 · max 1.65e+04
task/aim_dist: [0.316, 0.449, 0.278, 0.786, 0.502, 0.865, 0.363, 0.854, 0.384, 0.484]  min 0.175 · mean 0.563 · max 1.46
task/cup_collision_rate: [0, 0.00146, 0.0181, 0.00659, 0.0149, 0.00342, 0.00464, 0.00513, 0.0144, 0.00928]  min 0 · mean 0.00791 · max 0.0825
task/cups_center_dist: [0.318, 0.501, 0.333, 0.839, 0.563, 0.916, 0.415, 0.906, 0.436, 0.533]  min 0.22 · mean 0.616 · max 1.51
task/episode_success: [0, 0.317, 0.376, 0.352, 0.491, 0.391, 0.44, 0.442, 0.384, 0.372]  min 0 · mean 0.389 · max 0.554
task/nested_rate: [0, 0.00244, 0.00366, 0.00293, 0.00317, 0.00317, 0.00439, 0.00488, 0.00317, 0.00513]  min 0 · mean 0.00358 · max 0.00977
task/rcv_cup_lift: [3.33e-05, 0.074, 0.0835, 0.0883, 0.158, 0.0807, 0.0864, 0.147, 0.0804, 0.128]  min -0.0274 · mean 0.0881 · max 0.208
task/rcv_grasped: [0, 0.755, 0.769, 0.754, 0.781, 0.745, 0.777, 0.763, 0.768, 0.75]  min 0 · mean 0.76 · max 0.891
task/rcv_hand_foreign_rate: [0, 0.000244, 0.00293, 0.00269, 0.00415, 0.00293, 0.00415, 0.00366, 0.00513, 0.00439]  min 0 · mean 0.00357 · max 0.0186
task/src_cup_lift: [0.000809, 0.315, 0.278, 0.542, 0.262, 0.576, 0.366, 0.627, 0.249, 0.39]  min 0.000809 · mean 0.389 · max 0.747
task/src_grasped: [0, 0.716, 0.738, 0.727, 0.768, 0.695, 0.735, 0.716, 0.719, 0.713]  min 0 · mean 0.722 · max 0.833
task/src_hand_foreign_rate: [0.0261, 0.012, 0.00952, 0.0103, 0.00732, 0.00684, 0.0115, 0.00806, 0.00684, 0.00928]  min 0.00391 · mean 0.00964 · max 0.0725
task/src_tilt_deg: [2.92, 70.5, 74.3, 72.3, 81.1, 69.8, 72.2, 72.5, 70.8, 68.9]  min 2.92 · mean 71.7 · max 82.8
task/success_now: [0, 0.325, 0.385, 0.363, 0.503, 0.403, 0.451, 0.456, 0.394, 0.383]  min 0 · mean 0.399 · max 0.562

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 14 operator notes — 절반에서 멈추는 구조를 깬다: 전량 이송과 마무리 안정화에 값을 친다

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 목표와 원칙 (사용자)

- 목표: **흘림 없는 이송**. 09.23 사용자 추가: **전량 이송되고 안정화될수록 좋다**는 보상이 없어서 개선이 멈춘 것 아닌가 — 이 라운드가 그 답이다.
- ★원칙: 실기는 안정 속도 틸팅. **속도·제어 벌점을 풀어 SIM 효율을 되찾는 설계 금지.**
- 조준 허용 범위는 mm 상수로 박지 않고 `ctx.cup_radius` 배수로 쓴다(iter_13 규약 유지).

## 1. iter_13 결과 (t2r_rh3_i13, server GPU0, 4096 env, e1250 에서 종료)

40-epoch 블록 평균:

| 구간 | success | 흘림/유출 | 이송 효율 | in_source | in_target | 기울기 | entropy |
|---|---|---|---|---|---|---|---|
| e0~40 | 0.371 | 0.265 | 0.696 | 0.474 | — | 70.0° | 8.39 |
| e120~160(저점) | 0.288 | 0.332 | 0.619 | 0.521 | — | 68.0° | 7.92 |
| **e520~560(정점)** | **0.480** | **0.211** | **0.759** | 0.390 | 0.463 | 78.5° | 6.81 |
| e1210~1250 | 0.38 대 | 0.22 | 0.75 | 0.483 | 0.386 | 69.5° | 4.99 |

- ★**iter_10~12 의 단조 붕괴를 처음으로 막았다.** 저점 뒤 11 블록 연속 상승해 트랙 최고(이송 효율 0.759)를 찍었다.
- 정점 뒤 600 epoch 동안 50-ep 이동평균 0.482 를 넘지 못하고 0.38~0.45 로 내려앉았다. entropy 는 8.39 → 4.99 로 단조 하락.
- 조준 규율은 끝까지 유지됐다: 창 밖에서 기울이는 비율 0.208 → 0.12~0.16, 창 안 0.46~0.52.
- 컵 겹침 0.005·컵 충돌 0.010 으로 낮다.

## 2. ★왜 절반에서 멈추나 (측정 + 코드)

성공 판정은 env 가 한다(`pour_fabric_env.py:625`): in_target ≥ 0.50 ∧ 흘림 ≤ 0.40 ∧ 두 컵 xy < 0.20 m ∧ 끼워넣기 아님, 그리고 10 스텝 유지.

iter_13 실측 수입(스텝당 평균, e1210~1250 / 정점):

| 항목 | 값 | 성격 |
|---|---|---|
| success_bonus (W 20 × success × clean_f) | 5.77 / 6.71 | 임계 통과 시 스텝당 |
| transferred (W **4.0** × in_target × …) | 0.97 / 1.03 | 상태 비례 |
| pour_delta (W 150 × Δin_target) | 0.105 / 0.122 | 한 알 +7.5, 순간 |
| spill_delta (W 240 × Δspill) | −0.113 / −0.111 | 한 알 −12 |

- **in_target 을 0.5 에서 1.0 으로 올려도 늘어나는 상시 수입은 transferred 의 +2/스텝뿐**이다. 반대로 더 부으면 한 알당 −12 의 흘림 위험과, 흘림이 한도를 넘어 성공이 꺼질 때의 −20/스텝을 진다.
- 실제로 **in_source 가 0.48 에서 멈췄다 — 비드 절반이 소스 컵에 남는다.** in_target 은 0.386~0.463 에서 평평하다.
- 마무리(전량 이송 후 두 컵을 직립·정지로 유지)에 대한 별도 수입은 없다. success_bonus 가 유지에 값을 치지만 임계가 0.5 라 절반만 옮겨도 계속 나온다.

## 3. env 변경 (사용자 결정 09.23 "리워드/env 갱신 진행")

`pour_fabric_env_cfg.py` 성공 판정:

| 필드 | 이전 | 이번 | 뜻 |
|---|---|---|---|
| `success_fill_ratio` | 0.50 | **0.80** | 비드 20개 중 16개 이상 |
| `success_spill_max` | 0.40 | **0.25** | 흘림 5개 이하 |
| `success_hold_steps` | 10 | **20** | 1/3 s 유지(마무리 안정화) |

- ★**success/episode_success 는 이전 런과 직접 비교 불가**하다. 라운드 간 비교는 `bead/in_target`·흘림/유출·이송 효율로 한다.
- 컵 자산·물리·관측·액션은 그대로다(두꺼운 바닥 `shaker_closed_thick_rl` 유지).
- 첫 틱 주의: 임계가 올라가 **success 가 0.48 → 0 근처에서 시작**하는 것이 정상이다. 붕괴로 오판하지 말 것. 판정 지표는 in_target 과 in_source 다.

## 4. 다음 보상이 해야 할 것

1. ★**전량 이송에 값을 친다.** in_target 상태항(현재 W 4.0)을 올리고, **끝으로 갈수록 가팔라지게** 한다.
   소스가 비는 정도(1 − in_source)에도 수입을 줘서 "절반 붓고 멈추기"보다 "끝까지 붓기"가 확실히 이득이 되게 한다.
   새 상수는 실측 운영점으로 정한다(정점 in_target 0.463·in_source 0.390, 현재 0.386/0.483).
2. ★**마무리 안정화에 값을 친다.** 전량에 가까운 상태에서 **두 컵이 직립이고 정지**이며 손이 잡고 있는 구간에 수입을 준다.
   상태 비례로만 주면 "자세 잡고 멈춰 있기"가 최적해가 되므로(형제 트랙 교훈) 전량 조건과 곱하고, 기울기 복귀·정지에 한정한다.
3. **성공 임계가 0.80/0.25 로 올라간 것을 보상 쪽에도 반영한다.** `clean_f`(현재 1 − spill/0.40)의 기준을 새 흘림 한도에 맞춘다.
4. **그대로 둘 것:** iter_13 의 접근/허가 분리(approach_lip·perm 창, cup_radius 배수)·조준 규율·속도 벌점(src_speed·src_spin·rot_pour·rcv_spin)·
   flow_excess·overtilt·spill 240·팔별 palm_sat/palm_bang/act_flip/carry_rate. 이것들이 붕괴를 막고 제어를 개선했다.
5. 절대 거리 상수를 새로 만들지 않는다(조준·허가 관련은 `ctx.cup_radius` 배수).

## 5. 웜스타트

- **i13 ep550**(정점: success 0.480·흘림/유출 0.211·효율 0.759). 영상 확인 결과는 이 노트 작성 시점에 아직 없다.

## 6. ★영상·결정론 재생 실측 (09.23, 16 env · 900 스텝 · server GPU0)

| 체크포인트 | step 600 지표 |
|---|---|
| ep550(정점) | in_target **0.906** · in_source **0.000** · spill 0.094 · 기울기 127.0° · aim 0.095 m · 성공 0.938 |
| ep1250(마지막) | in_target **0.906** · in_source **0.000** · spill 0.084 · 기울기 127.8° · aim 0.071 m · 성공 1.000 |

★**중요한 정정:** "절반만 붓고 멈춘다"는 해석은 **학습 곡선(stochastic)에만 해당**한다. 결정론 재생에서는 두 체크포인트 모두
**소스 컵을 완전히 비우고(in_source 0.000) 90.6 % 를 받는 컵에 넣는다.** 새 성공 기준(0.80/0.25)으로도 성공이다.
즉 정책은 이미 전량 이송을 할 줄 알며, 학습 곡선의 in_source 0.48 은 **탐색 잡음(sigma)에서 나오는 실패분**이다.
또한 ep1250 은 ep550 과 같거나 더 낫다 — 곡선 정체는 정책 열화가 아니다.

**따라서 이 라운드의 표적은 "전량 이송을 가르치기"가 아니라 "탐색 잡음이 있어도 무너지지 않게(견고성)"다.**
- 전량 이송·안정화 수입은 여전히 필요하다(임계 0.5 뒤 평평한 구간을 없앤다).
- 다만 **결정론 성능은 이미 90 % 대**이므로, 새 항이 이 동작을 흔들지 않게 기존 구조(접근/허가 분리·조준 규율)를 유지한다.
