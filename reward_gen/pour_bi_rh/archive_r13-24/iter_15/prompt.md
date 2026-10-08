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


def _late_ramp(x, lo, hi):
    """iter_14: monotone ramp that is NEVER flat above `lo` and gets steeper toward `hi`.
    0.5*t + 0.5*t^2 on t = clamp((x - lo)/(hi - lo)): marginal value per unit x is 0.5/(hi-lo) at the
    bottom and 1.5/(hi-lo) at the top, so the last beads are worth three times the first ones.
    Deliberately NOT a smoothstep: smoothstep has zero slope at t = 0, i.e. no gradient exactly at the
    operating point the policy sits on today."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    iter_14: VERBATIM since iter_04 - the grasp works (src_grasped 0.722, rcv_grasped 0.760)."""
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
    iter_06 VERBATIM - on the keep list (grasp_src 1.21, grasp_rcv 1.10 measured over the iter_13 run,
    both lift bands held). iter_14 changes NOTHING in this function."""
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

    # penalties (unweighted, bounded) - all unchanged from iter_04
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
    iter_14: VERBATIM iter_13 (the aim discipline held all run: tilting outside the window fell from
    0.208 to 0.12-0.16 and never came back).

      d_hat    : horizontal unit vector src cup origin -> receiver mouth. Independent of the cup's
                 attitude, so it is defined for an upright cup and never flips while rotating.
      theta    : SIGNED tilt toward the receiver = atan2(up . d_hat, up_z). A purely sideways tilt
                 gives 0, tilting away gives a negative angle.
      lip      : the rim point the beads leave from = mouth centre + r*(cos(theta)*d_hat
                 - sin(theta)*z). Drops as the cup turns toward the receiver, RISES when it turns away.
      lip_drop : how far that lip sits BELOW where it would be with the same cup origin and an upright
                 cup: r*sin(theta) + mouth_z*(1 - cos(theta)). 0.067 m at 75 deg, 0.082 at 90.
                 Measured budget diag_lift_budget 0.300-0.312 all run, i.e. the term is live."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]          # (N,2)
    sep = v.norm(dim=-1)                                               # (N,)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / sep_min, max=1.0)   # the azimuth is meaningless when the cups are coaxial

    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])                     # signed, negative = away
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    # what is tilted but NOT toward the receiver: 0 for a perfect pour attitude (diag_mis_deg 1.8 mean)
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
    # as in iter_06..13 so carry_src and align_mouths keep their measured scale (align 2.35)
    d_align = torch.sqrt(lip_xy ** 2 + (0.5 * (lip_dz - clearance)) ** 2)
    # the lip point in 3-D and the xy unit vector receiver mouth -> lip, for the closing rate
    lip_u = lip_xy_vec / torch.clamp(lip_xy, min=1e-6).unsqueeze(-1)
    lip_pt = torch.cat([ctx.rcv_cup_mouth_pos[:, :2] + lip_xy_vec,
                        (ctx.src_cup_mouth_pos[:, 2] - lip_r * torch.sin(theta)).unsqueeze(-1)], dim=-1)
    return {"d_hat_x": d_hat[:, 0], "sep_ok": sep_ok, "theta": theta, "theta_eff": theta_eff,
            "mis": mis, "lip_xy": lip_xy, "lip_dz": lip_dz, "d_align": d_align,
            "sin_t": sin_t, "lip_drop": lip_drop, "lip_u": lip_u, "lip_pt": lip_pt}


def _lip_closing_rate(ctx, lip_u, lip_pt):
    """iter_13 (kept): d(lip_xy)/dt from the rigid-body velocities (stateless potential difference).
    The lip is taken as a point fixed on the source cup, the receiver mouth as a point fixed on the
    receiver cup. Negative = the spout is closing on the receiver mouth. [m/s]"""
    v_lip = ctx.src_cup_lin_vel + torch.cross(ctx.src_cup_ang_vel, lip_pt - ctx.src_cup_pos, dim=-1)
    v_mouth = ctx.rcv_cup_lin_vel + torch.cross(ctx.rcv_cup_ang_vel,
                                                ctx.rcv_cup_mouth_pos - ctx.rcv_cup_pos, dim=-1)
    return ((v_lip[:, :2] - v_mouth[:, :2]) * lip_u).sum(dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt(SIGNED) < pour,
    # and NEW in iter_14: < full transfer < finish (empty source, both cups upright and still).
    #
    # iter_13 RESULT. It stopped the monotone collapse of iter_10-12 (11 rising blocks after the dip,
    # peak transfer efficiency 0.759) and the aim discipline held to the end (tilting outside the
    # permission window 0.208 -> 0.12-0.16). But the 50-epoch mean never passed 0.482 for 600 epochs.
    # Deterministic replay of BOTH ep550 and ep1250 empties the source (in_source 0.000) and delivers
    # 0.906, so the policy already knows the full pour; the learning-curve in_source 0.48 is exploration
    # noise. What is missing is a REASON to go past half: above in_target 0.5 the only growing income
    # was transferred (+2.0 total from 0.5 to 1.0), against -12 per spilled bead.
    #
    # iter_14 changes EXACTLY four things and keeps everything else verbatim:
    #   1. clean_f follows the env: SPILL_ZERO 0.40 -> 0.25 (success_spill_max is now 0.25). The tilt
    #      ladder uses clean_pour = 0.10 + 0.90*clean_f so a spoiled episode keeps a tenth of the pour
    #      income instead of going dark; success / transferred keep the hard, env-matched factor.
    #   2. transfer_tail (new) + W_TRANSFER 4.0 -> 5.0: a ramp that is 0 below the measured operating
    #      point (in_target 0.386 mean) and full at the demonstrated 0.906, steepening toward the end.
    #   3. src_emptied (new): the same ramp on 1 - bead_in_source_frac, 0 at the measured in_source
    #      0.483 and full at the demonstrated 0.000. Gated on clean_f and on beads actually delivered,
    #      so emptying the cup onto the table pays nothing.
    #   4. settle_finish (new): income only in the END state - near-full transfer AND near-empty source,
    #      both cups still grasped, both upright inside UPRIGHT_TOL, both still. This is what the env's
    #      new 20-step hold asks for. It cannot become a parking reward because it is ~0 (0.052) at the
    #      current operating point and only opens after the beads have been delivered.
    # KEEP LIST (user note 4.4), all verbatim iter_13: approach/permission split (approach_lip, perm in
    # cup_radius multiples), aim discipline, src_speed / src_spin / rot_pour / rcv_spin, flow_excess,
    # overtilt, spill 240, per-arm palm_sat / palm_bang / act_flip / carry_rate, grasp / lift / converge
    # / carry / align. No new ABSOLUTE distance constant is introduced anywhere.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.81 measured)
    W_AWAY_RCV = 0.8
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (best-ever grasp rates)
    # --- source lift: band + over-lift ramp. KEEP LIST (diag_lift_budget 0.300 measured) -------------
    W_LIFT_SRC = 3.0
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.15
    LIFT_FREE_SRC0 = 0.24   # base of the moving budget: 0.24 upright, 0.307 at 75 deg, 0.335 at 105
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    LIFT_FLOOR = 0.60
    SRC_UP_FLOOR = 0.5
    LIFT_PINCH_BONUS = 0.2
    # --- receiver lift band - KEEP LIST (rcv lift 0.088 m, lift_rcv 1.21) ---------------------------
    W_LIFT_RCV = 2.0
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10
    RCV_LIFT_SIGMA = 0.07
    W_HOLD_RCV = 1.2
    RCV_DISC_FLOOR = 0.35
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7
    # --- convergence / carry / align - KEEP LIST, unchanged ----------------------------------------
    W_CONVERGE = 3.0
    CONV_FAR = 1.00
    CONV_NEAR = 0.16
    CONV_SIGMA = 0.25
    W_CARRY, W_ALIGN = 3.0, 5.0
    CARRY_K, CARRY_RANGE = 1.2, 1.00
    ALIGN_K, ALIGN_SIGMA = 2.0, 0.15
    POUR_CLEARANCE = 0.04   # aim point 4 cm above the receiver rim (unchanged since iter_06)
    SEP_MIN = 0.03          # below 3 cm of xy separation the pouring azimuth is numerically unstable
    # --- lip geometry gates - KEEP LIST, every window in RECEIVER-MOUTH RADII (ctx.cup_radius) ------
    PERM_IN_R, PERM_OUT_R = 1.0, 3.0
    LEAD_FULL_R, LEAD_ZERO_R = 1.75, 5.0
    W_APPROACH = 1.0
    LIP_Z_LO, LIP_Z_HI = 0.0, 0.02
    LIP_Z_FREE0, LIP_Z_SIGMA = 0.12, 0.10
    LIP_ABOVE_FLOOR = 0.35
    # --- TILT LADDER on the SIGNED angle - KEEP LIST ------------------------------------------------
    RELEASE_FULL = math.radians(75.0)
    RELEASE_SPAN = math.radians(25.0)
    PRE_MARGIN = math.radians(20.0)
    OVER_RANGE = math.radians(30.0)
    W_TILT_AIM = 2.0
    W_TILT_NEAR = 6.0
    W_TILT_POUR = 10.0
    W_POUR_POSE = 5.0
    BAND_BELOW, BAND_ABOVE = 0.10, 0.40
    LEAD_FLOOR = 0.15
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # Gaussian width on the corridor EXCESS (unchanged)
    TILT_LIFT_H = 0.03
    W_EARLY_TILT = 0.8
    MIS_FREE = math.radians(5.0)
    MIS_SPAN = math.radians(60.0)
    W_MISDIR = 1.6
    DIR_FULL, DIR_ZERO = 0.0, 0.28   # d_hat_x > 0 = pouring away from the torso
    W_DIR_PEN = 1.0
    NEST_SAFE, NEST_ZERO = 0.115, 0.09
    W_NEAR_NEST = 1.0
    # --- receiver discipline - KEEP LIST -----------------------------------------------------------
    W_HOME_RCV = 0.8
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5
    STILL_FREE, STILL_SCALE = 0.05, 0.15
    # --- task / penalties ---------------------------------------------------------------------------
    W_POUR = 150.0          # x d_in_target: 7.5 per bead with 20 beads (unchanged)
    W_SPILL = 240.0         # -12 per bead spilled (unchanged, keep list)
    SETTLE_FRAC = 0.03      # spill charge off for the first 3 % of the episode (beads settling at spawn)
    W_SUCCESS = 20.0        # env thresholds MOVED to 0.80 fill / 0.25 spill / 20-step hold; the weight
                            # is unchanged, so the same success income now costs a real finish
    # ★ iter_14 (1): the spill limit of the SUCCESS TEST is now 0.25, so the income is worth nothing
    # there, exactly as SPILL_ZERO 0.40 mirrored the old limit. At the measured spill 0.133 clean_f
    # goes 0.667 -> 0.468; at the deterministic 0.09 it goes 0.775 -> 0.64.
    SPILL_ZERO = 0.25
    # the tilt ladder rides on a FLOORED version: an episode that has already blown the budget must
    # still learn where the remaining beads go, at a tenth of the clean income (0.10 x W, i.e. 1.0 of
    # pour_tilt's 10.0) instead of going completely dark and parking on align/converge.
    CLEAN_POUR_FLOOR = 0.10
    # ★ iter_14 (2) full-transfer income. Constants are the MEASURED operating point and the MEASURED
    # deterministic ceiling, nothing else:
    #   TAIL_LO 0.35 - just below bead/in_target 0.386 mean (range 0.376-0.471) so the slope is alive
    #                  where the policy lives today instead of starting flat above it
    #   TAIL_HI 0.90 - the deterministic replay value 0.906 (ep550 AND ep1250), i.e. proven reachable
    W_TRANSFER = 5.0        # 4.0 -> 5.0: keeps the DEMONSTRATED state's income under the stricter
                            # clean_f (4.0x0.906x0.775 = 2.81 -> 5.0x0.906x0.64 = 2.90) while the
                            # half-poured state drops (1.17 -> 0.99)
    TAIL_LO, TAIL_HI = 0.35, 0.90
    W_TAIL = 4.0            # 0 at in_target 0.35, +2.6/step at 0.906 after clean_f: half the measured
                            # success income (5.34), so finishing pays well but never outbids success
    # ★ iter_14 (3) emptying income. EMPTY_LO 0.45 = just below the measured emptied 0.517
    # (in_source 0.483 at the end of the run; 0.610 at the peak block), full at 1.0 = the deterministic
    # in_source 0.000. Beads that left the cup onto the TABLE must not pay: x clean_f x fill_ok.
    EMPTY_LO = 0.45
    W_EMPTY = 3.0           # +1.9/step at an empty source after clean_f
    # ★ iter_14 (4) finish stabilisation. finish_q is the MEAN of the two ramps, not their min: a
    # min-based term is flat whenever the two are tied and gives no gradient on either.
    W_SETTLE = 3.0          # up to +3/step ONLY in the finished pose (both cups held, upright inside
                            # UPRIGHT_TOL and still). 0.052 -> ~0.1/step at today's operating point, so
                            # it cannot be farmed before the beads are delivered.
    FLOW_CAP_BEADS = 1.0    # paid per step at most 1 bead (unchanged)
    W_FLOW = 40.0           # -2 per extra bead in the same step, and it is unpaid (unchanged)
    UNAIMED_BELOW = math.radians(5.0)
    UNAIMED_SPAN = math.radians(15.0)
    W_UNAIMED = 3.0         # x (1 - perm), unchanged (keep list: the aim discipline held all run)
    W_AIMED_TILT = 2.5
    OVERTILT_ABOVE = math.radians(20.0)
    W_OVERTILT, OVERTILT_MAX = 6.0, 3.0
    # ★ cup motion limits - KEEP LIST (user principle: never buy SIM efficiency by loosening these)
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, ROT_POUR_SCALE, W_ROT_POUR = 1.0, 1.5, 1.0   # stricter while pouring
    RCV_W_FREE, RCV_W_SCALE, W_RCV_SPIN = 0.5, 1.0, 0.5
    W_FLIP_PALM = 0.3
    W_FLIP_HAND = 0.2
    W_RATE_PALM_CARRY = 0.03
    BANG_DEAD = 0.1
    W_BANG_SRC, W_BANG_RCV = 1.5, 1.5
    W_RATE_HAND_CARRY = 0.01
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_TILT_PEN_RCV = 0.3
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05
    SAT_LO = 0.7
    W_PALM_SAT_SRC = 3.0    # measured -1.05/step at diag_palm_sat 0.454: still live, keep
    W_PALM_SAT_RCV = 1.5    # measured -0.40/step at diag_palm_sat_rcv 0.356
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35      # rad (20 deg) - the env's own receiver-upright limit; iter_14 reuses it
                            # for the SOURCE in the finish pose, so no new attitude constant appears

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

    # ---- spout gates: how close the pouring lip is to the receiver mouth (VERBATIM iter_13) --------
    r_mouth = ctx.cup_radius
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r_mouth)
                                  / ((LEAD_ZERO_R - LEAD_FULL_R) * r_mouth))
    perm_xy = 1.0 - _smoothstep((geo["lip_xy"] - PERM_IN_R * r_mouth)
                                / ((PERM_OUT_R - PERM_IN_R) * r_mouth))
    lip_free = LIP_Z_FREE0 + ctx.cup_radius * geo["sin_t"]
    lip_above_raw = torch.clamp((geo["lip_dz"] - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(geo["lip_dz"] - lip_free, min=0.0) / LIP_Z_SIGMA) ** 2)
    lip_above = LIP_ABOVE_FLOOR + (1.0 - LIP_ABOVE_FLOOR) * lip_above_raw     # gating (keeps a floor)
    z_ok = lip_above_raw * lip_high                                          # paying (no floor)
    perm = perm_xy * lip_above_raw
    dir_ok = 1.0 - _smoothstep((geo["d_hat_x"] - DIR_FULL) / (DIR_ZERO - DIR_FULL))
    dir_soft = 0.3 + 0.7 * dir_ok

    # ---- release angle: rises as the cup empties (first bead ~70 deg full, ~86 deg at half) --------
    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    release_tilt = RELEASE_FULL + RELEASE_SPAN * (1.0 - fill)
    theta_pre = torch.clamp(release_tilt - PRE_MARGIN, min=0.30)
    theta_max = release_tilt + OVER_RANGE

    # ---- source attitude CORRIDOR, opened by the SPOUT DISTANCE and charged on the MAGNITUDE -------
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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # measured rcv_progress 0.745

    # ---- source lift: band + linear cost above it, free height tied to lip_drop (unchanged) --------
    lift_free_src = LIFT_FREE_SRC0 + geo["lip_drop"]
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_lift_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_lift_band * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])
    # ungated by the grasp flag on purpose: throwing the cup up must not dodge the cost
    over_lift = torch.clamp((h_src - lift_free_src) / OVER_LIFT_RANGE, 0.0, 1.0)

    # ---- receiver lift band (unchanged - measured 0.088 m, inside the 0.10 m band) -----------------
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
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.76 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * not_nested * conv_shape

    # carry / align keep their iter_06 shape and weights on the DIRECTED spout (align 2.35 measured)
    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * not_nested
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * not_nested
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- TILT: three segments of the SIGNED angle, each with a stricter SPOUT requirement -----------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_hold = carry_soft_src * beads_left * not_nested * geo["sep_ok"] * dir_soft
    # ★ iter_14: SPILL_ZERO is the env's NEW spill limit (0.25). clean_f is the hard, env-matched factor
    # used by success / transferred; clean_pour keeps a 0.10 floor for the tilt ladder so a spoiled
    # episode still learns where the remaining beads should go instead of abandoning the pour.
    spill_lvl = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    clean_f = torch.clamp(1.0 - spill_lvl / SPILL_ZERO, 0.0, 1.0)   # 1 clean, 0 at the success limit
    clean_pour = CLEAN_POUR_FLOOR + (1.0 - CLEAN_POUR_FLOOR) * clean_f
    release_hold = carry_soft_src * clean_pour * not_nested * geo["sep_ok"] * dir_soft
    lead_dist = LEAD_FLOOR + (1.0 - LEAD_FLOOR) * lead_gate
    aim_frac = torch.clamp(theta_eff / theta_pre, 0.0, 1.0)                                  # 0 -> 55 deg
    near_frac = torch.clamp((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.05),
                            0.0, 1.0)                                                        # 55 -> 75
    pour_frac = torch.clamp((theta_eff - release_tilt) / (theta_max - release_tilt), 0.0, 1.0)  # 75 -> 105
    # 0 -> theta_pre: no bead can leave, the lead zone (5 r) is enough and it keeps its floor
    tilt_aim = tilt_hold * aim_frac * lead_dist
    # theta_pre -> release (the first bead leaves inside this segment): PERMISSION only
    tilt_near = release_hold * near_frac * perm
    # the release segment: only with the spout actually over the receiver mouth and above its rim
    pour_tilt = both_up * clean_pour * dir_soft * upright_rcv * perm * pour_frac
    # plateau at the release attitude so holding the pose beats swinging through it
    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = both_up * clean_pour * dir_ok * upright_rcv * perm_xy * z_ok * pour_band

    # ---- directional / prematurity penalties -------------------------------------------------------
    early_tilt = torch.tanh(tilt_excess_src / TILT_TOL)      # magnitude above what the aim allows
    mis_pen = torch.clamp((geo["mis"] - MIS_FREE) / MIS_SPAN, 0.0, 1.0)
    mis_tilt = carry_soft_src * mis_pen
    pour_dir_pen = carry_soft_src * _smoothstep((geo["d_hat_x"] - 0.10) / 0.25)
    cup_center_d = (ctx.src_cup_pos - ctx.rcv_cup_pos).norm(dim=-1)
    near_nest = _smoothstep((NEST_SAFE - cup_center_d) / (NEST_SAFE - NEST_ZERO))   # 0 at the pour pose
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    # palm saturation. Action layout: [0:6] src palm, [6:12] src hand, [12:18] rcv palm, [18:24] rcv hand
    a_abs = ctx.actions.abs()
    sat_src_raw = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv_raw = torch.mean(_smoothstep((a_abs[:, 12:18] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)

    # control gate WITHOUT adr_progress: both cups grasped and >= 5.5 cm up (measured 0.651)
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

    # ---- aiming at the moment of release (VERBATIM iter_13) ----------------------------------------
    has_beads = torch.clamp(fill / 0.10, max=1.0)               # >= 2 beads left
    pre_release = _smoothstep((ctx.src_cup_tilt - (theta_pre - UNAIMED_BELOW)) / UNAIMED_SPAN)
    unaimed = carry_soft_src * has_beads * pre_release * (1.0 - perm)
    aimed_win = _smoothstep((theta_eff - (theta_pre - UNAIMED_BELOW)) / UNAIMED_SPAN)
    aimed_tilt = both_up * has_beads * clean_pour * dir_soft * upright_rcv * aimed_win * perm

    # ---- approach: stateless potential difference on lip_xy, independent of the tilt ---------------
    lip_rate = _lip_closing_rate(ctx, geo["lip_u"], geo["lip_pt"])
    outside_win = (geo["lip_xy"] > PERM_IN_R * r_mouth).to(dtype)
    approach_gate = carry_soft_src * rcv["held"] * not_nested * has_beads
    approach_lip = approach_gate * outside_win * torch.clamp(-lip_rate / SRC_V_FREE, -1.0, 1.0)

    # ---- overtilt: linear past release + 20 deg, carried cup only ----------------------------------
    overtilt = carry_soft_src * torch.clamp(
        W_OVERTILT * torch.clamp(ctx.src_cup_tilt - (release_tilt + OVERTILT_ABOVE), min=0.0),
        max=OVERTILT_MAX)

    # ---- cup motion limits (KEEP LIST: these are what the real robot needs) ------------------------
    v_src = src["v_lin"]
    w_src = ctx.src_cup_ang_vel.norm(dim=-1)
    w_rcv = ctx.rcv_cup_ang_vel.norm(dim=-1)
    src_speed = ctrl_gate * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = ctrl_gate * torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE)
    rot_pour = pour_phase * torch.tanh(torch.clamp(w_src - ROT_POUR_FREE, min=0.0) / ROT_POUR_SCALE)
    rcv_spin = ctrl_gate * torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE)

    # ================================================== ★ iter_14: FULL TRANSFER AND FINISH =========
    # The flat region above in_target 0.5 is what stopped iter_13: the only growing income there was
    # transferred (+2.0 total from half to full) against -12 per spilled bead. Both ramps below are 0
    # at the MEASURED stochastic operating point and 1 at the MEASURED deterministic replay, and both
    # get steeper toward the end, so the last beads are worth three times the first ones.
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    emptied = torch.clamp(1.0 - fill, 0.0, 1.0)
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)      # 0.035 at 0.386 (today), 0.744 at 0.80, 1 at 0.90
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)        # 0.068 at in_source 0.483, 1 at in_source 0
    # a source emptied ONTO THE TABLE must pay nothing: clean_f is already 0 at the 0.25 spill limit,
    # and fill_ok additionally requires the beads to have arrived somewhere useful
    fill_ok = torch.clamp(in_t / TAIL_LO, max=1.0)
    # MEAN of the two ramps, not min: a min-based term is flat whenever the two are tied
    finish_q = 0.5 * (fill_tail + empty_q)

    # finish pose: both cups STILL, reusing the existing speed thresholds (no new constant). Mean of
    # four one-sided ramps so every axis keeps its own slope instead of one zero killing the product.
    v_still_src = 1.0 - torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    w_still_src = 1.0 - torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE)
    w_still_rcv = 1.0 - torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE)
    settle_still = 0.25 * (v_still_src + w_still_src + still_f + w_still_rcv)
    upright_src = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)   # the source RETURNS after pouring
    # HARD grasp product on purpose: dropping the cups must never buy stillness. Gated on finish_q, so
    # this is unreachable before the transfer is essentially done - it is the 20-step hold, not parking.
    settle_fin = finish_q * g_src * g_rcv * not_nested * upright_src * upright_rcv * settle_still

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
        # level term, now steeper and stricter: 5.0 x in_target x clean_f(0.25)
        "transferred": W_TRANSFER * in_t * not_nested * upright_rcv * clean_f,
        # ★ iter_14: the missing slope above half. 0 at the measured 0.386, full at the demonstrated 0.906
        "transfer_tail": W_TAIL * fill_tail * not_nested * upright_rcv * clean_f,
        # ★ iter_14: emptying the SOURCE pays too, so "half and stop" is strictly worse than finishing
        "src_emptied": W_EMPTY * empty_q * fill_ok * not_nested * clean_f,
        # ★ iter_14: the finish the env's 20-step hold now asks for (both cups held, upright, still)
        "settle_finish": W_SETTLE * settle_fin,
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
    # diagnostics only (added AFTER the sum, not part of the reward)
    comps["rcv_progress"] = rcv_prog
    comps["aim_gate"] = perm_xy          # the r-window
    comps["diag_perm"] = perm            # window x lip above the rim
    comps["diag_pre_release"] = pre_release
    comps["diag_theta_deg"] = theta_eff * (180.0 / math.pi)
    comps["diag_mis_deg"] = geo["mis"] * (180.0 / math.pi)
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lift_budget"] = lift_free_src
    comps["diag_lead_gate"] = lead_gate
    comps["diag_palm_sat"] = sat_src_raw
    comps["diag_flow_beads"] = torch.clamp(d_in, min=0.0) * n_beads
    comps["diag_clean"] = clean_f
    comps["diag_ctrl_gate"] = ctrl_gate
    comps["diag_pour_phase"] = pour_phase
    comps["diag_src_w"] = w_src
    comps["diag_src_v"] = v_src
    comps["diag_flip_palm"] = flip_palm
    comps["diag_palm_sat_rcv"] = sat_rcv_raw
    comps["diag_bang_src"] = bang_src
    comps["diag_bang_rcv"] = bang_rcv
    comps["diag_perm_at_tilt"] = pre_release * perm
    comps["diag_unperm_at_tilt"] = pre_release * (1.0 - perm)
    comps["diag_close_speed"] = torch.clamp(-lip_rate, -1.0, 1.0)
    # ★ iter_14 diagnostics: one per new mechanism, so the next round can tell which moved
    comps["diag_fill_tail"] = fill_tail          # how far past the old plateau the transfer got
    comps["diag_empty_q"] = empty_q              # how empty the source ends up
    comps["diag_finish_q"] = finish_q            # the combined finish state (mean ramp)
    comps["diag_settle_still"] = settle_still    # were both cups actually stopped at the end
    comps["diag_emptied"] = emptied              # raw 1 - in_source, for comparison across rounds
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0.244, 0.512, 0.539, 0.558, 0.242, 0.468, 0.536, 0.585, 0.601]  min 0 · mean 0.431 · max 0.601
bead/spill: [0, 0.117, 0.223, 0.222, 0.222, 0.0885, 0.155, 0.167, 0.174, 0.177]  min 0 · mean 0.156 · max 0.225
done/drop: [0, 0.000244, 0.000244, 0.000732, 0.00122, 0.000244, 0.000488, 0.000244, 0.000244, 0.000488]  min 0 · mean 0.000394 · max 0.00122
done/mimic_runaway: [0, 0.000488, 0, 0.000977, 0.000244, 0.000488, 0.000732, 0.000244, 0, 0]  min 0 · mean 0.000354 · max 0.00122
episode_lengths/step: [121, 269, 377, 450, 590, 432, 555, 597, 644, 692]  min 121 · mean 487 · max 703
reward/act_flip: [0, -0.189, -0.171, -0.166, -0.167, -0.115, -0.16, -0.166, -0.166, -0.166]  min -0.216 · mean -0.155 · max 0
reward/action_rate: [-0.0133, -0.0156, -0.0157, -0.0154, -0.0152, -0.0152, -0.0153, -0.0152, -0.0154, -0.015]  min -0.0158 · mean -0.0153 · max -0.0133
reward/aim_gate: [0, 0.731, 0.64, 0.627, 0.62, 0.306, 0.635, 0.647, 0.624, 0.65]  min 0 · mean 0.571 · max 0.756
reward/aimed_tilt: [0, 0.421, 0.0984, 0.102, 0.0957, 0.0787, 0.175, 0.157, 0.0939, 0.103]  min 0 · mean 0.185 · max 1.25
reward/align_mouths: [0, 2.18, 2.76, 2.83, 2.86, 1.3, 2.5, 2.87, 2.94, 3.02]  min 0 · mean 2.46 · max 3.06
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/approach_lip: [0, 0.0184, 0.0121, 0.0104, 0.0132, 0.0971, 0.0255, 0.0168, 0.0101, 0.00733]  min -0.02 · mean 0.0227 · max 0.258
reward/away_rcv: [-0.0682, -0.158, -0.161, -0.163, -0.162, -0.164, -0.164, -0.163, -0.162, -0.162]  min -0.166 · mean -0.156 · max -0.0682
reward/carry_rate: [0, -0.136, -0.132, -0.129, -0.129, -0.0876, -0.123, -0.128, -0.131, -0.13]  min -0.143 · mean -0.118 · max 0
reward/carry_src: [0, 1.23, 1.48, 1.47, 1.47, 0.81, 1.33, 1.52, 1.55, 1.55]  min 0 · mean 1.32 · max 1.65
reward/close_rcv: [0.0794, 0.0969, 0.0871, 0.0817, 0.0796, 0.0841, 0.0885, 0.0866, 0.0828, 0.0812]  min 0.0775 · mean 0.0852 · max 0.0979
reward/close_src: [0.0279, 0.0209, 0.0224, 0.0219, 0.0215, 0.0159, 0.017, 0.0176, 0.0182, 0.0185]  min 0.0145 · mean 0.0197 · max 0.0279
reward/contact_push: [-0.00288, -0.00676, -0.00586, -0.00401, -0.00463, -0.0282, -0.00893, -0.00633, -0.0051, -0.00463]  min -0.0303 · mean -0.0071 · max -0.00111
reward/converge: [0, 1.47, 1.91, 1.96, 2, 1.32, 1.7, 1.99, 2.05, 2.08]  min 0 · mean 1.74 · max 2.09
reward/cup_collision: [0, -0.0102, -0.00382, -0.00318, -0.00381, -0.00161, -0.00411, -0.00277, -0.00165, -0.0022]  min -0.0102 · mean -0.00303 · max 0
reward/cup_disturb_rcv: [-0.0275, -0.0903, -0.109, -0.112, -0.119, -0.191, -0.102, -0.108, -0.111, -0.115]  min -0.216 · mean -0.11 · max -0.0275
reward/cup_disturb_src: [-0.213, -0.0276, -0.0271, -0.0251, -0.0252, -0.0733, -0.0366, -0.0274, -0.0252, -0.0236]  min -0.213 · mean -0.0401 · max -0.0149
reward/cup_fling: [-0.000649, -0.0189, -0.0209, -0.0189, -0.0184, -0.0492, -0.0172, -0.0147, -0.0148, -0.0133]  min -0.0927 · mean -0.0202 · max -0.000649
reward/diag_bang_rcv: [0.165, 0.244, 0.227, 0.218, 0.224, 0.197, 0.22, 0.22, 0.219, 0.219]  min 0.165 · mean 0.221 · max 0.289
reward/diag_bang_src: [0.0574, 0.0768, 0.0967, 0.0958, 0.103, 0.0975, 0.0908, 0.0981, 0.102, 0.0984]  min 0.0574 · mean 0.0942 · max 0.105
reward/diag_clean: [1, 0.671, 0.431, 0.425, 0.419, 0.757, 0.578, 0.543, 0.526, 0.52]  min 0.417 · mean 0.586 · max 1
reward/diag_close_speed: [0.074, 0.0106, -0.00755, -0.00635, -0.00478, 0.133, 0.00764, 0.00861, -0.00156, 0.00411]  min -0.0137 · mean 0.0188 · max 0.214
reward/diag_ctrl_gate: [0, 0.736, 0.756, 0.773, 0.777, 0.363, 0.715, 0.78, 0.791, 0.804]  min 0 · mean 0.675 · max 0.804
reward/diag_emptied: [0, 0.472, 0.756, 0.78, 0.796, 0.35, 0.646, 0.722, 0.773, 0.791]  min 0 · mean 0.608 · max 0.806
reward/diag_empty_q: [0, 0.229, 0.727, 0.755, 0.774, 0.328, 0.615, 0.691, 0.752, 0.769]  min 0 · mean 0.573 · max 0.784
reward/diag_fill_tail: [0, 0.0718, 0.428, 0.456, 0.475, 0.207, 0.414, 0.475, 0.529, 0.542]  min 0 · mean 0.369 · max 0.542
reward/diag_finish_q: [0, 0.15, 0.577, 0.606, 0.625, 0.268, 0.514, 0.583, 0.641, 0.656]  min 0 · mean 0.471 · max 0.656
reward/diag_flip_palm: [0.233, 0.691, 0.62, 0.594, 0.596, 0.527, 0.592, 0.598, 0.595, 0.592]  min 0.233 · mean 0.587 · max 0.768
reward/diag_flow_beads: [0, 0.146, 0.00879, 0.0115, 0.0139, 0.00952, 0.0161, 0.0208, 0.012, 0.0176]  min 0 · mean 0.0188 · max 0.146
reward/diag_lead_gate: [0, 0.786, 0.765, 0.781, 0.793, 0.371, 0.732, 0.783, 0.782, 0.811]  min 0 · mean 0.688 · max 0.811
reward/diag_lift_budget: [0.244, 0.316, 0.317, 0.319, 0.319, 0.283, 0.313, 0.319, 0.32, 0.322]  min 0.244 · mean 0.309 · max 0.322
reward/diag_lip_xy: [0.277, 0.148, 0.219, 0.329, 0.262, 0.281, 0.285, 0.272, 0.38, 0.219]  min 0.131 · mean 0.267 · max 0.411
reward/diag_mis_deg: [1.35, 1.78, 1.84, 1.77, 1.42, 4.2, 1.24, 1.19, 1.27, 0.824]  min 0.824 · mean 1.92 · max 13.6
reward/diag_palm_sat: [0.306, 0.444, 0.407, 0.399, 0.385, 0.314, 0.4, 0.397, 0.386, 0.383]  min 0.215 · mean 0.388 · max 0.495
reward/diag_palm_sat_rcv: [0.357, 0.397, 0.349, 0.351, 0.349, 0.394, 0.381, 0.358, 0.35, 0.351]  min 0.341 · mean 0.36 · max 0.397
reward/diag_perm: [0, 0.664, 0.628, 0.615, 0.608, 0.298, 0.613, 0.632, 0.612, 0.63]  min 0 · mean 0.556 · max 0.751
reward/diag_perm_at_tilt: [0, 0.651, 0.611, 0.596, 0.585, 0.287, 0.596, 0.615, 0.598, 0.616]  min 0 · mean 0.54 · max 0.715
reward/diag_pour_phase: [0, 0.697, 0.697, 0.717, 0.722, 0.332, 0.669, 0.728, 0.736, 0.755]  min 0 · mean 0.628 · max 0.755
reward/diag_pre_release: [0, 0.762, 0.771, 0.786, 0.787, 0.37, 0.724, 0.784, 0.794, 0.818]  min 0 · mean 0.686 · max 0.818
reward/diag_settle_still: [0.942, 0.612, 0.591, 0.586, 0.582, 0.436, 0.622, 0.608, 0.596, 0.597]  min 0.221 · mean 0.596 · max 0.942
reward/diag_src_v: [0.0596, 0.31, 0.345, 0.425, 0.321, 0.511, 0.384, 0.284, 0.33, 0.246]  min 0.0596 · mean 0.468 · max 3.74
reward/diag_src_w: [1.49, 2.69, 2.43, 3.29, 2.29, 3.75, 2.5, 2.02, 2.07, 2.07]  min 1.49 · mean 7.11 · max 141
reward/diag_theta_deg: [7.78, 84.1, 85.7, 87, 87.2, 50, 81, 87.8, 88.7, 91.1]  min 5.78 · mean 76.5 · max 91.1
reward/diag_unperm_at_tilt: [0, 0.111, 0.16, 0.19, 0.201, 0.0835, 0.128, 0.169, 0.196, 0.202]  min 0 · mean 0.146 · max 0.206
reward/early_tilt: [-0.0685, -0.108, -0.116, -0.108, -0.106, -0.169, -0.109, -0.104, -0.0999, -0.0953]  min -0.186 · mean -0.108 · max -0.0685
reward/flow_excess: [0, -0.021, -0.00146, -0.000488, -0.00391, -0.00293, -0.00195, -0.00488, -0.00195, -0.00391]  min -0.021 · mean -0.00313 · max 0
reward/grasp_rcv: [0.00108, 1.36, 1.24, 1.19, 1.16, 0.831, 1.17, 1.21, 1.2, 1.18]  min 0.00108 · mean 1.09 · max 1.36
reward/grasp_src: [0.00432, 1.43, 1.38, 1.37, 1.38, 1.21, 1.26, 1.38, 1.4, 1.45]  min 0.00432 · mean 1.28 · max 1.45
reward/hand_flip: [0, -0.057, -0.0693, -0.0712, -0.069, -0.0317, -0.0632, -0.0682, -0.0751, -0.0707]  min -0.0751 · mean -0.0597 · max 0
reward/hand_foreign: [-0.000244, -0.00551, -0.00469, -0.00373, -0.0038, -0.00766, -0.00369, -0.0033, -0.00258, -0.00208]  min -0.0117 · mean -0.00385 · max -0.000244
reward/hold_rcv: [0, 0.16, 0.248, 0.232, 0.234, 0.152, 0.211, 0.245, 0.244, 0.249]  min 0 · mean 0.208 · max 0.263
reward/lift_rcv: [0, 0.915, 1.34, 1.37, 1.4, 0.653, 1.17, 1.36, 1.4, 1.42]  min 0 · mean 1.18 · max 1.48
reward/lift_src: [0, 2.64, 2.57, 2.61, 2.63, 1.65, 2.48, 2.67, 2.69, 2.77]  min 0 · mean 2.38 · max 2.77
reward/mis_tilt: [0, -0.0108, -0.0105, -0.00784, -0.0105, -0.0256, -0.00677, -0.00731, -0.00589, -0.00424]  min -0.156 · mean -0.013 · max 0
reward/near_nested: [0, -0.0204, -0.00698, -0.00679, -0.00683, -0.00339, -0.00672, -0.00384, -0.00428, -0.00363]  min -0.0204 · mean -0.00582 · max 0
reward/nested: [0, -0.0083, -0.00684, -0.00781, -0.00732, -0.00195, -0.00684, -0.00244, -0.00391, -0.00244]  min -0.0083 · mean -0.00427 · max 0
reward/over_lift_src: [0, -0.0188, -0.0455, -0.0422, -0.0419, -0.0155, -0.0278, -0.0285, -0.0316, -0.0186]  min -0.0474 · mean -0.0273 · max 0
reward/overtilt: [0, -0.0798, -0.0192, -0.028, -0.0302, -0.0223, -0.0345, -0.0514, -0.0382, -0.0469]  min -0.0798 · mean -0.0355 · max 0
reward/palm_bang_rcv: [0, -0.325, -0.304, -0.299, -0.305, -0.2, -0.285, -0.296, -0.298, -0.297]  min -0.398 · mean -0.276 · max 0
reward/palm_bang_src: [0, -0.0916, -0.12, -0.121, -0.134, -0.114, -0.11, -0.126, -0.131, -0.13]  min -0.135 · mean -0.114 · max 0
reward/palm_sat_rcv: [0, -0.516, -0.45, -0.46, -0.46, -0.353, -0.469, -0.462, -0.457, -0.461]  min -0.516 · mean -0.421 · max 0
reward/palm_sat_src: [0, -1.11, -1.02, -1, -0.968, -0.723, -0.959, -1.01, -0.987, -0.99]  min -1.22 · mean -0.918 · max 0
reward/pour_delta: [0, 0.992, 0.0586, 0.0714, 0.0842, 0.0586, 0.112, 0.137, 0.0824, 0.115]  min -0.00366 · mean 0.123 · max 0.992
reward/pour_dir_pen: [0, -0.191, -0.121, -0.106, -0.136, -0.055, -0.154, -0.114, -0.128, -0.148]  min -0.249 · mean -0.113 · max 0
reward/pour_pose: [0, 0.637, 0.426, 0.429, 0.387, 0.242, 0.502, 0.622, 0.545, 0.52]  min 0 · mean 0.509 · max 1.8
reward/pour_tilt: [0, 1.07, 0.589, 0.598, 0.557, 0.357, 0.75, 0.913, 0.794, 0.793]  min 0 · mean 0.741 · max 2.46
reward/rcv_home: [0, -0.101, -0.0675, -0.0701, -0.0741, -0.0409, -0.0763, -0.0669, -0.0654, -0.0707]  min -0.101 · mean -0.0634 · max 0
reward/rcv_progress: [0.29, 0.822, 0.808, 0.81, 0.813, 0.705, 0.775, 0.806, 0.807, 0.815]  min 0.29 · mean 0.765 · max 0.822
reward/rcv_spin: [0, -0.145, -0.156, -0.168, -0.176, -0.0773, -0.143, -0.165, -0.177, -0.178]  min -0.195 · mean -0.147 · max 0
reward/rcv_still: [0, -0.385, -0.412, -0.437, -0.441, -0.27, -0.384, -0.427, -0.442, -0.447]  min -0.453 · mean -0.384 · max 0
reward/reach_rcv: [0.919, 0.845, 0.821, 0.816, 0.816, 0.833, 0.82, 0.82, 0.815, 0.816]  min 0.815 · mean 0.829 · max 0.919
reward/reach_src: [0.518, 0.55, 0.529, 0.525, 0.525, 0.559, 0.519, 0.515, 0.51, 0.511]  min 0.509 · mean 0.527 · max 0.591
reward/rot_pour: [0, -0.241, -0.272, -0.278, -0.277, -0.13, -0.248, -0.258, -0.282, -0.293]  min -0.319 · mean -0.241 · max 0
reward/rush_near_cup: [-0.232, -0.0298, -0.0255, -0.0228, -0.0223, -0.0398, -0.0309, -0.0231, -0.023, -0.0211]  min -0.232 · mean -0.0396 · max -0.0205
reward/settle_finish: [0, 0.0002, 0.000187, 0.000282, 0.000481, 3.44e-05, 0.000101, 0.000101, 0.000125, 6.5e-05]  min 0 · mean 0.000199 · max 0.000655
reward/spill_delta: [0, -0.653, -0.0732, -0.108, -0.0967, -0.193, -0.185, -0.111, -0.0469, -0.0762]  min -0.653 · mean -0.138 · max 0
reward/src_emptied: [0, 0.26, 0.587, 0.626, 0.653, 0.315, 0.69, 0.786, 0.895, 0.924]  min 0 · mean 0.582 · max 0.924
reward/src_speed: [0, -0.104, -0.123, -0.128, -0.126, -0.0636, -0.113, -0.118, -0.119, -0.124]  min -0.203 · mean -0.113 · max 0
reward/src_spin: [0, -0.0968, -0.114, -0.116, -0.114, -0.0573, -0.102, -0.104, -0.116, -0.118]  min -0.118 · mean -0.0985 · max 0
reward/success_bonus: [0, 0.107, 3.69, 3.96, 4.14, 1.95, 4.33, 4.99, 5.79, 5.96]  min 0 · mean 3.61 · max 5.96
reward/tilt_aim: [0, 0.752, 0.869, 0.916, 0.895, 0.46, 0.837, 0.978, 0.954, 0.956]  min 0 · mean 0.822 · max 1.41
reward/tilt_near: [0, 1.82, 1.24, 1.25, 1.18, 0.678, 1.46, 1.61, 1.47, 1.48]  min 0 · mean 1.38 · max 3.82
reward/total: [1.92, 15.6, 20.1, 20.6, 20.8, 12, 20.6, 23.5, 24.4, 24.9]  min 1.92 · mean 19.4 · max 24.9
reward/transfer_tail: [0, 0.171, 0.618, 0.645, 0.66, 0.321, 0.748, 0.843, 0.961, 0.97]  min 0 · mean 0.609 · max 0.97
reward/transferred: [0, 0.69, 0.77, 0.805, 0.821, 0.409, 0.932, 1.06, 1.18, 1.2]  min 0 · mean 0.785 · max 1.2
reward/unaimed_release: [0, -0.249, -0.0463, -0.034, -0.0278, -0.0316, -0.0633, -0.0491, -0.0432, -0.0418]  min -0.249 · mean -0.0543 · max 0
rewards/step: [113, 2.27e+03, 4.59e+03, 6.61e+03, 9.12e+03, 6.81e+03, 1.01e+04, 1.14e+04, 1.33e+04, 1.59e+04]  min 113 · mean 8.23e+03 · max 1.59e+04
task/aim_dist: [0.304, 0.211, 0.326, 0.466, 0.404, 0.359, 0.399, 0.428, 0.513, 0.308]  min 0.211 · mean 0.375 · max 0.576
task/cup_collision_rate: [0, 0.0166, 0.00488, 0.00439, 0.00513, 0.0022, 0.00562, 0.00366, 0.00195, 0.00317]  min 0 · mean 0.00434 · max 0.0166
task/cups_center_dist: [0.312, 0.273, 0.389, 0.531, 0.468, 0.397, 0.459, 0.493, 0.579, 0.376]  min 0.224 · mean 0.432 · max 0.643
task/episode_success: [0, 0, 0.338, 0.357, 0.378, 0.161, 0.325, 0.39, 0.452, 0.463]  min 0 · mean 0.296 · max 0.463
task/nested_rate: [0, 0.00415, 0.00342, 0.00391, 0.00366, 0.000977, 0.00342, 0.00122, 0.00195, 0.00122]  min 0 · mean 0.00213 · max 0.00415
task/rcv_cup_lift: [8.46e-05, 0.127, 0.104, 0.104, 0.101, 0.0506, 0.102, 0.102, 0.0967, 0.106]  min 8.46e-05 · mean 0.0912 · max 0.127
task/rcv_grasped: [0, 0.848, 0.856, 0.867, 0.872, 0.6, 0.803, 0.854, 0.862, 0.872]  min 0 · mean 0.775 · max 0.875
task/rcv_hand_foreign_rate: [0, 0.00317, 0.00366, 0.00342, 0.00366, 0.00195, 0.00171, 0.00171, 0.00195, 0.00122]  min 0 · mean 0.00208 · max 0.00366
task/src_cup_lift: [0.00215, 0.275, 0.327, 0.399, 0.396, 0.218, 0.348, 0.423, 0.419, 0.283]  min 0.00215 · mean 0.319 · max 0.487
task/src_grasped: [0, 0.794, 0.795, 0.803, 0.812, 0.742, 0.759, 0.82, 0.825, 0.843]  min 0 · mean 0.747 · max 0.843
task/src_hand_foreign_rate: [0.000488, 0.0154, 0.0125, 0.011, 0.00903, 0.0227, 0.0107, 0.00928, 0.0061, 0.00586]  min 0.000488 · mean 0.0107 · max 0.0325
task/src_tilt_deg: [9.13, 85.9, 87.5, 88.7, 88.6, 54.2, 82.3, 89, 90, 91.9]  min 9.13 · mean 78.4 · max 91.9
task/success_now: [0, 0.00684, 0.347, 0.371, 0.39, 0.17, 0.354, 0.408, 0.463, 0.476]  min 0 · mean 0.309 · max 0.476

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 15 operator notes — 손목을 한계에서 떼고 팔 떨림을 잡는다 (새 ctx 필드 사용)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 목표와 원칙 (사용자)

- 목표: **흘림 없는 이송** + 실기 이식 가능한 **안정 제어**.
- 09.23 사용자 관찰(영상): "관절을 극단적으로 꺾는 것과 팔들이 안정화되지 않고 흔들리는 것이 문제다."
- ★원칙: 속도·제어 벌점을 풀어 SIM 효율을 되찾는 설계 금지. 조준 창은 절대 거리 상수 대신 `ctx.cup_radius` 배수.

## 1. ★env 변경 — 보상이 관절을 직접 볼 수 있게 했다 (커밋 96297818)

`RewardContext` 에 팔당 두 필드를 **추가**했다(기존 필드·관측·액션 불변, 이전 iter 보상도 그대로 동작):

| 필드 | 뜻 |
|---|---|
| `src_arm_limit_margin`, `rcv_arm_limit_margin` | (N,7) soft limit 기준 여유 [0,1]. 1 = 범위 중앙, **0 = 한계에 닿음** |
| `src_arm_qdd`, `rcv_arm_qdd` | (N,7) 관절 가속도 [rad/s²] = (qd − 이전 qd)/스텝. 떨림 신호 |

실기에서도 관절 각도·속도는 읽히므로 s2r 계약을 깨지 않는다. 앞선 라운드는 팔 관절 속도(`arm_qd`)만 볼 수 있었고,
관절 한계는 **손바닥 액션 포화라는 간접 신호**로만 눌렀다.

## 2. ★i13 ep550 결정론 trace 실측 (64 env × 900 스텝, 09.23)

**소스 팔은 손목이 하한에 박혀 있다.**

| 관절 | 하한 5 % 안 스텝 비율 | 여유 중앙값 |
|---|---|---|
| r_aj_2 | 0.23 | 0.113 |
| r_aj_3 | 0.19 | 0.124 |
| **r_aj_5** | **0.42** | 0.115 |
| **r_aj_7** | **0.41** | 0.117 |
| 한 관절이라도 한계 5 % 안 | **0.464** | 최소 관절 여유 중앙 **0.104** |

- **리시버 팔은 깨끗하다: 어느 관절도 한계 5 % 안에 들어가지 않는다(0.000).** 문제는 소스 팔 한쪽이다.
- 소스 palm 회전 명령 대 실제 오차 중앙 **90.5°**(p90 124.5°) — 팔이 따라갈 수 없는 자세를 계속 명령한다.
- i11 과 비교: 한 관절이라도 한계 안인 비율 0.59~0.66 → **0.464**. iter_12 의 팔별 포화 벌점이 일부 줄였지만 절반 가까이 남았다.

**떨림은 소스 손목에 남아 있다.**

| 지표 | src | rcv |
|---|---|---|
| palm 액션 |Δa| 평균 | 0.149 | **0.429** |
| palm 액션 부호 교대 | 3.4 %/스텝 | **19.9 %/스텝** |
| 관절속도 방향 반전(관절 5·6·7) | **5.2 · 6.2 · 3.8 Hz** | 3.0 · 5.0 · 3.4 Hz |
| 관절 가속도 |qdd| 중앙 / p90 | 2.5 / **12.8** rad/s² | 2.4 / 9.3 |
| 관절속도 RMS | 0.522 rad/s | 0.342 |

- i11 대비 리시버 부호 교대 31~38 % → 19.9 %, 소스는 크게 개선(3.4 %). **남은 것은 소스 손목의 5~6 Hz 진동과 리시버 palm 의 큰 명령 변화량**이다.
- 소스 컵 각속도 최대 562 rad/s — 64 env 중 41 env 에서 한 번 이상 20 rad/s 를 넘지만 **전체 스텝의 0.24 %** 뿐인 순간 튐이다(비드 방출 직후 구간).

**비드 경로는 건전하다:** 소스를 떠난 1275 개 중 96 % 가 입구로 나간다. 받는 컵 진입 1059 개 중 입구 아래 진입 1 %.

## 3. 다음 보상이 해야 할 것

1. ★**관절 한계 여유에 직접 벌점.** `min(arm_limit_margin)` 이 작을수록 비싸게. 소스 팔 운영점은 최소 여유 **0.104**,
   한계 5 % 안이 46 % 이므로 경사가 살아 있는 구간은 대략 0~0.3 이다. 리시버는 이미 깨끗하니 같은 항을 걸어도 수입 손실이 없다.
   파지 중에만 켠다(접근 단계의 자유도를 뺏지 않는다).
2. ★**관절 가속도(떨림)에 벌점.** `arm_qdd` 의 크기. 운영점 중앙 2.5·p90 12.8 rad/s² 를 기준으로 정규화하고,
   정상 붓기 동작(기울이기)을 막지 않도록 **속도가 아니라 가속도**에만 건다. 5~6 Hz 진동은 가속도에서 가장 크게 보인다.
3. **리시버 palm 명령 변화량**(|Δa| 0.429)은 기존 `palm_bang_rcv`·`carry_rate` 로 이미 내려가는 중이다 — 가중을 조금만 올린다.
4. **그대로 둘 것:** 접근/허가 분리(approach_lip·perm, cup_radius 배수)·조준 규율·속도 벌점·flow_excess·overtilt·spill 240·
   iter_14 의 전량 이송(transfer_tail·src_emptied)·마무리 안정화(settle_finish)·팔별 palm_sat/palm_bang.
5. 새 항은 **성공 동작을 흔들지 않게** 크기를 잡는다. 결정론 재생 기준선은 in_target 0.906·in_source 0.000·흘림 0.094(ep550)이다.
   새 벌점 합이 스텝당 −1 을 크게 넘지 않게 한다(현재 제어 벌점 합계 약 −2, success_bonus 20/스텝).

## 4. 웜스타트

- **i13 ep550**(결정론 16 env: in_target 0.906·in_source 0.000·성공 0.938). i14 는 e29 에서 멈춰 비교 표본이 없다.
