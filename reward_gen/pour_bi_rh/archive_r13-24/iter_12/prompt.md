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
    # ★ iter_11: window anchored on theta_pre (release - 20 deg = 55 deg full), not on release.
    # On at theta_pre - 5 (50 deg full), saturated at theta_pre + 10 (65 deg full) — 5 deg before the
    # first bead leaves a full cup (~70 deg). Follows fill exactly as theta_pre does.
    UNAIMED_BELOW = math.radians(5.0)
    UNAIMED_SPAN = math.radians(15.0)
    W_UNAIMED = 3.0         # 1.5 -> 3.0 (charge is now continuous, see aim_q)
    # ★ iter_11: continuous aim quality on lip_xy. d = lip_xy - AIM_FREE;
    # miss = 0.5*(1 - exp(-d/AIM_NEAR)) + 0.5*clamp(d/AIM_FAR, 0, 1)
    # -> 0 at <= 2.5 cm, 0.28 at 6.5 cm, 0.49 at 12.5 cm, 0.62 at 0.2 m, 0.72 at 0.3 m, 0.90 at 0.5 m, 1 at 0.625 m.
    # The linear half keeps a slope of 0.83 /m across the measured 0.2-0.5 m operating band; the
    # exponential half is steep over the last few cm into the mouth.
    AIM_FREE = 0.025        # [guess] just inside the receiver inner radius 0.0281 m
    AIM_NEAR = 0.06
    AIM_FAR = 0.60          # [guess] chosen so the 0.2-0.5 m band (operator measurement) is unsaturated
    AIM_Z_FLOOR = 0.3       # lip below the receiver rim keeps 30 % of the aim credit (xy slope survives)
    W_AIMED_TILT = 2.5      # paid over the same window x aim_q; below tilt_near 6 / pour_tilt 10 so the
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
    # ★ iter_11: continuous aim quality (1 = spout over the mouth and above the rim)
    aim_d = torch.clamp(geo["lip_xy"] - AIM_FREE, min=0.0)
    aim_miss = 0.5 * (1.0 - torch.exp(-aim_d / AIM_NEAR)) + 0.5 * torch.clamp(aim_d / AIM_FAR, 0.0, 1.0)
    aim_q = (1.0 - aim_miss) * (AIM_Z_FLOOR + (1.0 - AIM_Z_FLOOR) * lip_above_raw)
    # charge on the tilt MAGNITUDE (beads do not care about direction), window before the first bead
    pre_release = _smoothstep((ctx.src_cup_tilt - (theta_pre - UNAIMED_BELOW)) / UNAIMED_SPAN)
    unaimed = carry_soft_src * has_beads * pre_release * (1.0 - aim_q)
    # paired income on the SIGNED angle, same window, both cups held and lifted: aimed tilting is paid,
    # a sideways tilt only pays the charge
    aimed_win = _smoothstep((theta_eff - (theta_pre - UNAIMED_BELOW)) / UNAIMED_SPAN)
    aimed_tilt = both_up * has_beads * dir_soft * upright_rcv * aimed_win * aim_q

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
        "aimed_tilt": W_AIMED_TILT * aimed_tilt,
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
    comps["diag_aim_q"] = aim_q
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
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0.0333, 0.0667, 0.1, 0.133, 0.167, 0.2, 0.233, 0.267, 0.3]  min 0 · mean 0.165 · max 0.333
bead/in_target: [0, 0.445, 0.486, 0.45, 0.416, 0.414, 0.388, 0.329, 0.312, 0.297]  min 0 · mean 0.391 · max 0.543
bead/spill: [0, 0.136, 0.119, 0.1, 0.108, 0.147, 0.169, 0.204, 0.201, 0.187]  min 0 · mean 0.149 · max 0.213
done/drop: [0, 0.000732, 0.000244, 0.000244, 0, 0, 0, 0.000244, 0, 0.000244]  min 0 · mean 0.000286 · max 0.00146
done/mimic_runaway: [0, 0.000977, 0.00122, 0.000244, 0.000244, 0, 0.000244, 0, 0, 0.000244]  min 0 · mean 0.000335 · max 0.00195
episode_lengths/step: [119, 634, 674, 651, 566, 666, 711, 732, 712, 618]  min 119 · mean 638 · max 771
reward/act_flip: [0, -0.25, -0.257, -0.232, -0.214, -0.215, -0.198, -0.167, -0.158, -0.13]  min -0.323 · mean -0.2 · max 0
reward/action_rate: [-0.015, -0.0213, -0.0215, -0.0204, -0.02, -0.0197, -0.0186, -0.0172, -0.0169, -0.0163]  min -0.0232 · mean -0.019 · max -0.015
reward/aim_gate: [0, 0.481, 0.501, 0.491, 0.465, 0.494, 0.499, 0.395, 0.395, 0.384]  min 0 · mean 0.46 · max 0.691
reward/aimed_tilt: [0, 0.196, 0.156, 0.196, 0.212, 0.23, 0.278, 0.313, 0.293, 0.25]  min 0 · mean 0.237 · max 1.15
reward/align_mouths: [0, 1.94, 2.1, 2.1, 2.09, 2.23, 2.19, 2.26, 2.2, 2.12]  min 0 · mean 2.12 · max 2.83
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/away_rcv: [-0.0454, -0.16, -0.155, -0.154, -0.152, -0.163, -0.172, -0.171, -0.17, -0.167]  min -0.174 · mean -0.162 · max -0.0454
reward/carry_rate: [0, -0.163, -0.169, -0.155, -0.142, -0.148, -0.141, -0.126, -0.121, -0.104]  min -0.202 · mean -0.139 · max 0
reward/carry_src: [0, 1.11, 1.18, 1.2, 1.21, 1.29, 1.23, 1.3, 1.3, 1.28]  min 0 · mean 1.22 · max 1.73
reward/close_rcv: [0.109, 0.0787, 0.0753, 0.0791, 0.0828, 0.0841, 0.0742, 0.073, 0.0786, 0.0796]  min 0.0674 · mean 0.0786 · max 0.109
reward/close_src: [0.016, 0.0148, 0.0149, 0.0147, 0.0149, 0.015, 0.0142, 0.0145, 0.0157, 0.0191]  min 0.0136 · mean 0.0159 · max 0.0252
reward/contact_push: [-0.00455, -0.0074, -0.00656, -0.0075, -0.00968, -0.00812, -0.00734, -0.00555, -0.00789, -0.00882]  min -0.0212 · mean -0.00782 · max -0.000806
reward/converge: [0, 1.48, 1.57, 1.58, 1.59, 1.7, 1.66, 1.79, 1.79, 1.75]  min 0 · mean 1.64 · max 2.03
reward/cup_collision: [0, -0.00492, -0.0164, -0.0222, -0.0164, -0.00774, -0.00251, -0.000277, -0.000244, 0]  min -0.13 · mean -0.00796 · max 0
reward/cup_disturb_rcv: [-0.0275, -0.116, -0.119, -0.117, -0.119, -0.115, -0.118, -0.123, -0.125, -0.119]  min -0.268 · mean -0.119 · max -0.0275
reward/cup_disturb_src: [-0.102, -0.032, -0.0319, -0.038, -0.0408, -0.0364, -0.0326, -0.0274, -0.0333, -0.0443]  min -0.102 · mean -0.0349 · max -0.00944
reward/cup_fling: [-0.000303, -0.0172, -0.015, -0.0136, -0.0168, -0.0131, -0.0115, -0.0142, -0.0135, -0.0159]  min -0.0676 · mean -0.0153 · max -0.000303
reward/diag_aim_q: [0.0866, 0.647, 0.644, 0.603, 0.604, 0.659, 0.686, 0.671, 0.662, 0.637]  min 0.0866 · mean 0.645 · max 0.738
reward/diag_clean: [1, 0.726, 0.75, 0.791, 0.777, 0.701, 0.662, 0.612, 0.617, 0.645]  min 0.593 · mean 0.703 · max 1
reward/diag_ctrl_gate: [0, 0.633, 0.651, 0.629, 0.598, 0.66, 0.687, 0.695, 0.668, 0.628]  min 0 · mean 0.645 · max 0.737
reward/diag_flip_palm: [0.38, 1.07, 1.07, 0.99, 0.947, 0.92, 0.83, 0.723, 0.705, 0.625]  min 0.38 · mean 0.869 · max 1.28
reward/diag_flow_beads: [0, 0.0232, 0.0176, 0.0239, 0.0203, 0.0144, 0.0193, 0.0134, 0.0154, 0.0149]  min 0 · mean 0.0176 · max 0.146
reward/diag_lead_gate: [0, 0.655, 0.688, 0.657, 0.644, 0.682, 0.697, 0.687, 0.673, 0.633]  min 0 · mean 0.665 · max 0.767
reward/diag_lift_budget: [0.241, 0.304, 0.307, 0.305, 0.305, 0.308, 0.309, 0.311, 0.31, 0.308]  min 0.241 · mean 0.307 · max 0.316
reward/diag_lip_xy: [0.29, 0.217, 0.214, 0.495, 0.235, 0.141, 0.209, 0.17, 0.215, 0.376]  min 0.101 · mean 0.3 · max 0.829
reward/diag_mis_deg: [0.613, 2.51, 2.01, 1.78, 1.74, 1.68, 1.57, 1.39, 1.24, 1.56]  min 0.613 · mean 1.77 · max 11.3
reward/diag_palm_sat: [0.339, 0.537, 0.52, 0.507, 0.506, 0.534, 0.541, 0.528, 0.523, 0.502]  min 0.317 · mean 0.52 · max 0.562
reward/diag_pour_phase: [0, 0.565, 0.588, 0.571, 0.546, 0.6, 0.625, 0.613, 0.594, 0.553]  min 0 · mean 0.579 · max 0.668
reward/diag_pre_release: [0, 0.688, 0.704, 0.689, 0.679, 0.719, 0.732, 0.754, 0.734, 0.71]  min 0 · mean 0.707 · max 0.799
reward/diag_src_v: [0.0457, 0.382, 0.358, 0.342, 0.331, 0.276, 0.27, 0.276, 0.257, 0.305]  min 0.0457 · mean 0.542 · max 31.4
reward/diag_src_w: [0.67, 2.16, 2.93, 1.83, 2.21, 1.89, 1.8, 1.76, 1.9, 2.22]  min 0.67 · mean 9.1 · max 1.32e+03
reward/diag_theta_deg: [1.03, 71.3, 74.3, 72.7, 72.1, 75.1, 76.3, 78.1, 77.2, 75.6]  min 1.03 · mean 74.3 · max 83.8
reward/early_tilt: [-0.00128, -0.133, -0.13, -0.128, -0.151, -0.133, -0.119, -0.152, -0.151, -0.174]  min -0.209 · mean -0.14 · max -0.00128
reward/flow_excess: [0, -0.00439, -0.00195, -0.00391, -0.00195, -0.000488, -0.00146, -0.00146, -0.00195, -0.00195]  min -0.022 · mean -0.00199 · max 0
reward/grasp_rcv: [0.000369, 0.977, 0.914, 0.925, 0.948, 1.14, 1.14, 1.15, 1.16, 1.1]  min 0.000369 · mean 1.04 · max 1.18
reward/grasp_src: [5.74e-05, 1.13, 1.19, 1.13, 1.11, 1.24, 1.25, 1.24, 1.25, 1.27]  min 5.74e-05 · mean 1.2 · max 1.38
reward/hand_flip: [0, -0.0679, -0.0707, -0.0675, -0.0641, -0.0706, -0.073, -0.0723, -0.0678, -0.0646]  min -0.0762 · mean -0.0684 · max 0
reward/hand_foreign: [-0.00437, -0.00397, -0.00481, -0.00525, -0.00527, -0.00407, -0.00569, -0.00708, -0.00589, -0.00757]  min -0.01 · mean -0.00553 · max -0.00334
reward/hold_rcv: [0, 0.202, 0.212, 0.232, 0.222, 0.232, 0.212, 0.216, 0.213, 0.184]  min 0 · mean 0.209 · max 0.246
reward/lift_rcv: [0, 1.02, 1.08, 1.13, 1.17, 1.22, 1.12, 1.23, 1.23, 1.16]  min 0 · mean 1.14 · max 1.42
reward/lift_src: [0, 1.82, 2.07, 2.17, 2.13, 2.19, 2.1, 2.19, 2.32, 2.34]  min 0 · mean 2.15 · max 2.62
reward/mis_tilt: [0, -0.0167, -0.0145, -0.00791, -0.00796, -0.009, -0.0136, -0.00901, -0.0074, -0.00936]  min -0.0927 · mean -0.0122 · max 0
reward/near_nested: [0, -0.0111, -0.0169, -0.0316, -0.0188, -0.0111, -0.00399, -0.00128, -0.000411, -0.000434]  min -0.173 · mean -0.0111 · max 0
reward/nested: [0, -0.00391, -0.00195, -0.00488, -0.00439, -0.00342, -0.00195, -0.000977, -0.000488, 0]  min -0.0171 · mean -0.00257 · max 0
reward/over_lift_src: [0, -0.189, -0.129, -0.074, -0.062, -0.091, -0.138, -0.121, -0.067, -0.0218]  min -0.326 · mean -0.0932 · max 0
reward/overtilt: [0, -0.0303, -0.0262, -0.0227, -0.0157, -0.00808, -0.0112, -0.00758, -0.00695, -0.00629]  min -0.109 · mean -0.0159 · max 0
reward/palm_sat_rcv: [0, -0.0365, -0.0387, -0.0366, -0.0359, -0.0397, -0.0406, -0.0414, -0.0384, -0.0376]  min -0.044 · mean -0.0381 · max 0
reward/palm_sat_src: [0, -0.0577, -0.0564, -0.0532, -0.0515, -0.0585, -0.0615, -0.0608, -0.0586, -0.0528]  min -0.065 · mean -0.0561 · max 0
reward/pour_delta: [0, 0.145, 0.0861, 0.159, 0.132, 0.103, 0.132, 0.0897, 0.0952, 0.0952]  min 0 · mean 0.115 · max 0.983
reward/pour_dir_pen: [0, -0.214, -0.165, -0.11, -0.109, -0.116, -0.101, -0.0637, -0.0464, -0.0857]  min -0.367 · mean -0.116 · max 0
reward/pour_pose: [0, 0.255, 0.249, 0.289, 0.325, 0.34, 0.444, 0.432, 0.411, 0.361]  min 0 · mean 0.347 · max 1.14
reward/pour_tilt: [0, 0.411, 0.409, 0.477, 0.514, 0.497, 0.626, 0.605, 0.58, 0.548]  min 0 · mean 0.521 · max 1.65
reward/rcv_home: [0, -0.0888, -0.0871, -0.0997, -0.104, -0.107, -0.105, -0.108, -0.122, -0.114]  min -0.137 · mean -0.106 · max 0
reward/rcv_progress: [0.282, 0.753, 0.755, 0.753, 0.753, 0.771, 0.762, 0.777, 0.773, 0.751]  min 0.282 · mean 0.759 · max 0.824
reward/rcv_spin: [0, -0.119, -0.128, -0.111, -0.112, -0.122, -0.136, -0.145, -0.137, -0.137]  min -0.188 · mean -0.127 · max 0
reward/rcv_still: [0, -0.338, -0.351, -0.331, -0.344, -0.369, -0.38, -0.415, -0.414, -0.417]  min -0.453 · mean -0.372 · max 0
reward/reach_rcv: [0.905, 0.818, 0.822, 0.826, 0.833, 0.816, 0.792, 0.799, 0.808, 0.809]  min 0.791 · mean 0.815 · max 0.905
reward/reach_src: [0.547, 0.512, 0.508, 0.511, 0.519, 0.504, 0.498, 0.501, 0.508, 0.506]  min 0.495 · mean 0.511 · max 0.628
reward/rot_pour: [0, -0.181, -0.198, -0.17, -0.173, -0.188, -0.19, -0.193, -0.191, -0.19]  min -0.331 · mean -0.188 · max 0
reward/rush_near_cup: [-0.224, -0.0394, -0.0366, -0.0387, -0.0428, -0.0364, -0.0338, -0.0312, -0.0322, -0.036]  min -0.224 · mean -0.0374 · max -0.0277
reward/spill_delta: [0, -0.144, -0.229, -0.085, -0.146, -0.0996, -0.152, -0.158, -0.182, -0.179]  min -0.413 · mean -0.144 · max 0
reward/src_speed: [0, -0.11, -0.101, -0.0892, -0.0887, -0.103, -0.103, -0.11, -0.101, -0.103]  min -0.252 · mean -0.102 · max 0
reward/src_spin: [0, -0.0797, -0.0841, -0.0713, -0.0726, -0.0767, -0.0762, -0.0804, -0.0802, -0.0819]  min -0.133 · mean -0.0794 · max 0
reward/success_bonus: [0, 6.42, 7.42, 7.11, 6.37, 5.55, 4.62, 3.24, 2.95, 2.94]  min 0 · mean 5.17 · max 8.12
reward/tilt_aim: [0, 0.699, 0.797, 0.856, 0.811, 0.831, 0.861, 0.867, 0.858, 0.756]  min 0 · mean 0.81 · max 1.33
reward/tilt_near: [0, 1.71, 1.93, 2.06, 2.04, 2.2, 2.42, 2.47, 2.45, 2.15]  min 0 · mean 2.14 · max 3.71
reward/total: [2.15, 20.1, 22, 22.6, 21.7, 21.7, 20.7, 19.7, 19.4, 18.6]  min 2.15 · mean 20.6 · max 23.6
reward/transferred: [0, 0.963, 1.13, 1.1, 0.967, 0.83, 0.673, 0.475, 0.432, 0.445]  min 0 · mean 0.778 · max 1.26
reward/unaimed_release: [0, -0.132, -0.165, -0.231, -0.213, -0.162, -0.0944, -0.106, -0.116, -0.139]  min -1.05 · mean -0.151 · max 0
rewards/step: [140, 1.35e+04, 1.41e+04, 1.52e+04, 1.28e+04, 1.44e+04, 1.4e+04, 1.49e+04, 1.29e+04, 1.27e+04]  min 140 · mean 1.32e+04 · max 1.76e+04
task/aim_dist: [0.318, 0.382, 0.438, 0.612, 0.384, 0.233, 0.306, 0.278, 0.362, 0.496]  min 0.184 · mean 0.434 · max 1.04
task/cup_collision_rate: [0, 0.00928, 0.0312, 0.042, 0.0286, 0.0132, 0.00391, 0.000244, 0.000244, 0]  min 0 · mean 0.0143 · max 0.251
task/cups_center_dist: [0.319, 0.434, 0.493, 0.667, 0.439, 0.29, 0.363, 0.337, 0.42, 0.553]  min 0.192 · mean 0.49 · max 1.09
task/episode_success: [0, 0.452, 0.506, 0.46, 0.427, 0.428, 0.391, 0.307, 0.285, 0.274]  min 0 · mean 0.391 · max 0.566
task/nested_rate: [0, 0.00195, 0.000977, 0.00244, 0.0022, 0.00171, 0.000977, 0.000488, 0.000244, 0]  min 0 · mean 0.00128 · max 0.00854
task/rcv_cup_lift: [7.54e-05, 0.0956, 0.0936, 0.0843, 0.073, 0.0902, 0.0919, 0.0832, 0.0689, 0.0679]  min 7.54e-05 · mean 0.0846 · max 0.187
task/rcv_grasped: [0, 0.772, 0.781, 0.771, 0.764, 0.796, 0.802, 0.824, 0.808, 0.77]  min 0 · mean 0.783 · max 0.855
task/rcv_hand_foreign_rate: [0, 0.00171, 0.00513, 0.00488, 0.00366, 0.00146, 0.00439, 0.00391, 0.00269, 0.00757]  min 0 · mean 0.00353 · max 0.00854
task/src_cup_lift: [0.000475, 0.372, 0.449, 0.358, 0.338, 0.244, 0.289, 0.296, 0.356, 0.32]  min 0.000475 · mean 0.346 · max 0.81
task/src_grasped: [0, 0.692, 0.721, 0.708, 0.69, 0.742, 0.751, 0.778, 0.771, 0.751]  min 0 · mean 0.729 · max 0.791
task/src_hand_foreign_rate: [0.0115, 0.0112, 0.0112, 0.0122, 0.0125, 0.0125, 0.0149, 0.0186, 0.0176, 0.0156]  min 0.00732 · mean 0.0145 · max 0.0288
task/src_tilt_deg: [1.64, 73.8, 76.3, 74.5, 73.9, 76.8, 77.8, 79.5, 78.4, 77.1]  min 1.64 · mean 76.1 · max 85.2
task/success_now: [0, 0.465, 0.514, 0.469, 0.437, 0.438, 0.399, 0.315, 0.295, 0.284]  min 0 · mean 0.401 · max 0.576

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 12 operator notes — 제어가 관절 한계·뱅뱅으로 무너지는 것을 막는다(조준·속도·흘림 장치는 그대로)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 목표와 원칙 (사용자)

- 목표(09.22): **흘림 없는 이송** — 안정 제어·정밀 조준·비드 이송.
- ★원칙: SIM 에서는 빠른 회전 틸팅이 효율이 좋지만 실기는 안정 속도 틸팅이 맞다. **속도·제어 벌점을 풀어 SIM 효율을 되찾는 설계 금지.**
- 사용자 영상 관찰(i11 ep150·ep350): "play 된 팔들이 자연스럽지 않음 — 관절 한계까지 밀려 있어 보이고, 흔들림도 심함."

## 1. iter_11 결과 (t2r_rh3_i11, server GPU0, 4096 env, e~500 에서 종료)

- 흘림/유출(= spill/(1−in_source)) 50-ep 블록: 0.183 → 0.228 → 0.238 → 0.291 → 0.353 → 0.366 (마지막 10 epoch 0.378).
- episode_success 0.467 → 0.290. 기울기 75.2° → 79.8°. action/step_delta 는 계속 좋아져 0.370.
- 최고 지점 = **ep150**: 결정론 재생 16/16 성공, in_target 86.9 %, spill 11.9 %.
- iter_11 의 조준 선행 장치(unaimed_release 창 50~65°, 연속 aim_q, aimed_tilt)는 초반 150 epoch 동안 흘림/유출을 0.183 까지 내렸다(iter_10 최저 0.224).

## 2. trace 측정 (ep150·ep350 16 env + ep150 64 env, 900 스텝, 결정론)

액션 배치: [0:6] 소스 palm, [6:12] 소스 손, [12:18] 리시버 palm, [18:24] 리시버 손.

### 2-1. 소스 팔: palm 명령이 도달 불가 쪽에 붙어 있다
- 소스 손목 관절 r_aj_5 는 스텝의 58~59 %, r_aj_7 은 46~62 % 가 하한 5 % 안이다. 7관절 중 하나라도 한계 5 % 안인 스텝 59~66 %.
- 소스 palm 회전 오차(명령 − 실제) 중앙 89~98°. 명령한 자세를 팔이 따라가지 못한다.
- 소스 palm 액션 |a|>0.95 포화 비율(차원 0..5): ep150 [0.19 0.77 0.28 0.24 0.56 0.52] · ep350 [0.35 0.74 0.37 0.21 0.42 0.77].
  평균: ep150 [−0.49 +0.89 +0.32 −0.08 +0.56 −0.72]. y(1)·회전 4·5 가 한쪽 끝에 붙어 있다.
- 소스 손목에 ~6 Hz 방향 반전 진동.
- iter_11 의 palm_sat_src(SAT_LO 0.7, W 0.15, 양손 파지·들기 게이트)는 diag_palm_sat 0.527 로 평평했고 에피소드 합 약 −0.05(성공 보너스의 1/100 수준)라 행동을 못 바꿨다.

### 2-2. 리시버 팔: 뱅뱅 명령
- 리시버 palm 액션 부호 교대(연속 두 스텝 |a|>0.1 이면서 부호 반대) 비율(차원 12..17): ep150 [0.45 0.77 0.27 0.13 0.46 0.17] · ep350 [0.53 0.72 0.18 0.07 0.36 0.02].
  **y(13) 는 스텝의 72~77 % 에서 부호가 뒤집힌다.** 같은 차원 포화 32~41 %.
- iter_11 의 act_flip(W 0.3, relu(−a·a_prev) 합)은 ctrl_gate(양손 파지 AND 두 컵 5.5 cm 들기) 안에서만 켜진다.
  [가설] 리시버 뱅뱅이 게이트 밖(들기 전·리시버만 쥐고 있을 때)에서도 나서 벌점이 닿지 않는다.

### 2-3. 컨텍스트 제약 (코드 확인: tasks/pour_fabric_mimic/t2r/context.py)
- **관절 위치·관절 한계 필드가 없다.** 팔은 src_arm_qd·rcv_arm_qd(관절 속도)만 있다. 관절 한계를 직접 벌할 수 없으므로
  palm 액션 포화(actions[:,0:6]·[12:18])와 액션 부호 교대·변화율, arm_qd 로 다룬다.
- adr_progress·src_tilt_toward_rcv·pour_dir_xy·premature_tilt·src_pour_lip_pos 없음(쓰면 static_check 실패).

## 3. env 변경 (사용자 결정 09.22, 이번 라운드부터)

- 컵 자산 `shaker_closed_rl.usd` → **`shaker_closed_thick_rl.usd`**: 바닥에 convexHull 16 mm 를 쌓아(scale 1) 바닥 윗면 −0.0881 → −0.0721.
  scale 0.65 에서 바닥이 10.4 mm 높아지고 내부 깊이가 그만큼 준다. 외곽 프로파일은 같다.
- 근거: i11 ep150 64 env trace 에서 받는 컵에 들어간 비드의 2~3 % 가 바닥을 뚫고 빠졌다(흘림의 약 10 %). CCD 로는 거의 안 줄어 되돌렸다.
- 비드 판정의 inside_z_min 이 새 바닥 윗면으로 올라갔다(흘림·in_target 판정). 비드 초기 높이도 새 바닥 기준.
  ctx.cup_bottom_z 는 컵 외곽 바닥(원점 → −object_origin_offset_z)이라 **바뀌지 않는다**.
- 관측·액션 차원 불변.

## 4. 다음 보상이 해야 할 것

1. ★**소스 palm 포화를 실제로 비싸게.** 포화는 "팔이 따라갈 수 없는 명령"이고 관절 한계 근접의 원인이다. 현재 −0.05/에피소드는 무의미하다.
   게이트를 파지만(들기 불요)으로 넓히고, 가중을 성공 보상과 견줄 크기로 올린다. SAT_LO 는 0.7 유지 가능 — 실측 운영점이 0.9~1.0 이라 경사는 있다.
2. ★**리시버 palm 뱅뱅을 게이트 밖에서도 벌한다.** 리시버가 컵을 쥐고 있는 동안(rcv_grasped) 리시버 palm 12 차원의 부호 교대·변화율을 벌한다.
   소스 쪽도 같은 형태로 파지 중이면 켠다.
3. 액션 벌점은 fresh 정책을 얼리지 않게 **파지 게이트** 뒤에 둔다(웜스타트이므로 파지는 이미 성립).
4. **그대로 둘 것:** 파지·들기·converge·방향(signed tilt)·기울기 연동 높이 예산·조준 선행(unaimed_release 창·aim_q·aimed_tilt)·
   속도 벌점(src_speed·src_spin·rot_pour·rcv_spin)·flow_excess·overtilt·spill 240·success/transferred clamp.
5. 새 항은 **성공 상태에서 실측한 값으로 정규화**한다(ep150 운영점: 소스 palm 포화 평균 약 0.43, 리시버 y 부호 교대 0.77).

## 5. 웜스타트

- **i11 ep150**(흘림/유출 0.183 구간, 결정론 16/16 성공). 컵 바닥이 바뀌어 초반 지표가 흔들릴 수 있다.
