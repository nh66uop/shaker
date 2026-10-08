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
    bottom and 1.5/(hi-lo) at the top, so the last part is worth three times the first."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _limit_pen(margin, free):
    """iter_15, KEPT but weakened in iter_16: how hard an arm is pressed against its joint limits.
    `margin` is (N, A) with 1 = middle of the range and 0 = touching a limit. The iter_15 run showed
    this penalty CANNOT be the lever: it grew -0.40 -> -0.50 while diag_src_limit_hit went 0.21 ->
    0.43 and the pour collapsed instead (corr(limit_hit, in_target) = +0.90 on first differences).
    The kinematic probe says why - with the rendezvous where it is today there is no posture with
    margin >= 0.30 at all. So the shape stays and only the weight drops (see W_ARM_LIMIT)."""
    d = torch.clamp((free - torch.clamp(margin, 0.0, 1.0)) / free, 0.0, 1.0)      # (N, A)
    worst = d.max(dim=-1).values                                                  # (N,)
    return 0.7 * worst * worst + 0.3 * (d * d).mean(dim=-1)


def _shake_pen(qdd, free_mean, scale_mean, free_peak, scale_peak):
    """iter_15, KEPT but weakened: arm tremor charged on ACCELERATION only, so the deliberate pouring
    rotation (a velocity) is never taxed. Measured operating point unchanged by iter_15
    (diag_src_qdd 9.1-10.2 all run), which is the second reason the weight drops in iter_16."""
    q = qdd.abs()
    mean_part = torch.tanh(torch.clamp(q.mean(dim=-1) - free_mean, min=0.0) / scale_mean)
    peak_part = torch.tanh(torch.clamp(q.max(dim=-1).values - free_peak, min=0.0) / scale_peak)
    return 0.65 * mean_part + 0.35 * peak_part


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    VERBATIM since iter_04 - the grasp works (src_grasped 0.70-0.88, rcv_grasped 0.75-0.92)."""
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
    iter_06 VERBATIM and stay so in iter_16 - the grasp side is not what blocks the transfer."""
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

    # penalties (unweighted, bounded) - all unchanged from iter_04. NOTE for iter_16: `slide` is gated
    # on (1 - lifted_w), i.e. it charges dragging a cup that is still on the table and costs nothing
    # once the cup is carried - so it does NOT block the receiver from travelling, and it stays as the
    # anti-tipping protection the operator asked to keep.
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
    """Everything about the pour measured on the DIRECTED spout (iter_08) plus the iter_09 LIP DROP.
    VERBATIM since iter_13 (aim discipline held all run: diag_mis_deg 1.1-2.5 deg).

      d_hat    : horizontal unit vector src cup origin -> receiver mouth. Independent of the cup's
                 attitude, so it is defined for an upright cup and never flips while rotating.
      theta    : SIGNED tilt toward the receiver = atan2(up . d_hat, up_z).
      lip      : the rim point the beads leave from; drops as the cup turns toward the receiver.
      lip_drop : how far that lip sits BELOW the upright-cup lip: r*sin + mouth_z*(1 - cos)."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]          # (N,2)
    sep = v.norm(dim=-1)                                               # (N,)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / sep_min, max=1.0)   # the azimuth is meaningless when the cups are coaxial

    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])                     # signed, negative = away
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    # what is tilted but NOT toward the receiver: 0 for a perfect pour attitude
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
    # as in iter_06..15 so carry_src and align_mouths keep their measured scale
    d_align = torch.sqrt(lip_xy ** 2 + (0.5 * (lip_dz - clearance)) ** 2)
    lip_u = lip_xy_vec / torch.clamp(lip_xy, min=1e-6).unsqueeze(-1)
    lip_pt = torch.cat([ctx.rcv_cup_mouth_pos[:, :2] + lip_xy_vec,
                        (ctx.src_cup_mouth_pos[:, 2] - lip_r * torch.sin(theta)).unsqueeze(-1)], dim=-1)
    return {"d_hat_x": d_hat[:, 0], "sep_ok": sep_ok, "theta": theta, "theta_eff": theta_eff,
            "mis": mis, "lip_xy": lip_xy, "lip_dz": lip_dz, "d_align": d_align,
            "sin_t": sin_t, "lip_drop": lip_drop, "lip_u": lip_u, "lip_pt": lip_pt}


def _lip_closing_rate(ctx, lip_u, lip_pt):
    """iter_13 (kept): d(lip_xy)/dt from the rigid-body velocities (stateless potential difference).
    Negative = the spout is closing on the receiver mouth. [m/s]"""
    v_lip = ctx.src_cup_lin_vel + torch.cross(ctx.src_cup_ang_vel, lip_pt - ctx.src_cup_pos, dim=-1)
    v_mouth = ctx.rcv_cup_lin_vel + torch.cross(ctx.rcv_cup_ang_vel,
                                                ctx.rcv_cup_mouth_pos - ctx.rcv_cup_pos, dim=-1)
    return ((v_lip[:, :2] - v_mouth[:, :2]) * lip_u).sum(dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt(SIGNED) < pour
    # < full transfer < finish. NEW in iter_16: WHERE the two cups meet is now part of the reward.
    #
    # iter_15 RESULT. The whole run went backwards: bead/in_target 0.665 -> 0.235, episode_success
    # 0.536 -> 0.125, src tilt 97 -> 63 deg, in_source 0.36 -> 0.60. The two iter_15 additions are the
    # reason, and the measurement says exactly why:
    #   - arm_limit_src grew -0.40 -> -0.50 while diag_src_limit_hit went UP 0.21 -> 0.43. First
    #     difference correlation corr(limit_hit, in_target) = +0.90: in TODAY'S geometry the policy
    #     can only pour by pinning its wrist, so charging the pin charges the pour.
    #   - arm_shake did nothing at all: diag_src_qdd stayed 9.1-10.2 the whole run.
    #   - kinematic probe (URDF FK, verified against the trace to 0.0 mm): with the spout at the point
    #     it actually uses, (0.37, +0.046, 0.55) pouring toward +y, there is NO arm posture with wrist
    #     margin >= 0.30. At (0.37, -0.05, 0.45) the best margin is 0.500, at (0.37, -0.15, 0.45)
    #     it is 0.628. The real operating point measures 0.089-0.103.
    #   - trace of the division of labour: the SOURCE cup travels 0.352 m and crosses the midline
    #     (y -0.159 -> +0.046, z 0.268 -> 0.550) while the RECEIVER cup moves 0.057 m in xy and stays
    #     home. The wrist is exhausted because the source arm does all of the travelling.
    #
    # iter_16 therefore changes EXACTLY three things and keeps everything else verbatim:
    #   1. meet_cross / src_cross (new): the rendezvous must happen in the SOURCE arm's own region,
    #      expressed as a fraction of the spawn-to-spawn span so it survives spawn randomisation.
    #   2. the receiver is untied once it carries: home_eff releases home_near inside rcv_disc (which
    #      multiplies converge / carry / align / hold_rcv), rcv_home is gated on "not carrying" and
    #      halved, rcv_still gets the source's own free speed while it transits.
    #   3. W_ARM_LIMIT 1.2 -> 0.4 and W_ARM_SHAKE 1.0 -> 0.4: kept as rudders, no longer a wall.
    # KEEP LIST (operator note 4), all verbatim: approach/permission split (approach_lip, perm in
    # cup_radius multiples), aim discipline, src_speed / src_spin / rot_pour / rcv_spin, flow_excess,
    # overtilt, spill 240, transfer_tail / src_emptied, per-arm palm_sat / palm_bang, grasp / lift /
    # converge / carry / align, and every lift band (the pour HEIGHT is a separate hypothesis).
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.81 measured)
    W_AWAY_RCV = 0.8                       # hand-to-OWN-cup distance only: independent of where the
                                           # cup is carried, so it never blocked the receiver travel
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged
    # --- source lift: band + over-lift ramp. KEEP LIST (diag_lift_budget 0.294 measured) -------------
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
    # --- receiver lift band - KEEP LIST (rcv lift 0.097 m measured, lift_rcv 1.19) ------------------
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
    TILT_LIFT_H = 0.03      # "carrying" height, reused for BOTH arms in iter_16 (no new constant)
    W_EARLY_TILT = 0.8
    MIS_FREE = math.radians(5.0)
    MIS_SPAN = math.radians(60.0)
    W_MISDIR = 1.6
    DIR_FULL, DIR_ZERO = 0.0, 0.28   # d_hat_x > 0 = pouring away from the torso
    W_DIR_PEN = 1.0
    NEST_SAFE, NEST_ZERO = 0.115, 0.09
    W_NEAR_NEST = 1.0
    # --- receiver discipline - RESHAPED in iter_16 (see the block further down) ---------------------
    W_HOME_RCV = 0.4        # 0.8 -> 0.4 and now gated on "not yet carrying": its only remaining job is
                            # to stop the cup being DRAGGED on the table before it is lifted
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5
    STILL_FREE, STILL_SCALE = 0.05, 0.15
    # --- task / penalties ---------------------------------------------------------------------------
    W_POUR = 150.0          # x d_in_target: 7.5 per bead with 20 beads (unchanged)
    W_SPILL = 240.0         # -12 per bead spilled (unchanged, keep list)
    SETTLE_FRAC = 0.03      # spill charge off for the first 3 % of the episode (beads settling at spawn)
    W_SUCCESS = 20.0        # unchanged; every guidance term stays well below it
    SPILL_ZERO = 0.25       # the env's spill limit (unchanged)
    CLEAN_POUR_FLOOR = 0.10
    W_TRANSFER = 5.0
    TAIL_LO, TAIL_HI = 0.35, 0.90
    W_TAIL = 4.0            # keep list: the iter_14 full-transfer pair worked (early in_target 0.47-0.53)
    EMPTY_LO = 0.45
    W_EMPTY = 3.0           # keep list
    W_SETTLE = 3.0          # unchanged; still ~0.0002/step because the source ends at ~90 deg. Left as
                            # is on purpose (one hypothesis per round); diag_finish_q / diag_settle_still
                            # stay so a later round can decide whether the source must come back up.
    FLOW_CAP_BEADS = 1.0    # paid per step at most 1 bead (unchanged)
    W_FLOW = 40.0           # -2 per extra bead in the same step, and it is unpaid (unchanged)
    UNAIMED_BELOW = math.radians(5.0)
    UNAIMED_SPAN = math.radians(15.0)
    W_UNAIMED = 3.0
    W_AIMED_TILT = 2.5
    OVERTILT_ABOVE = math.radians(20.0)
    W_OVERTILT, OVERTILT_MAX = 6.0, 3.0
    # cup motion limits - KEEP LIST (user principle: never buy SIM efficiency by loosening these)
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, ROT_POUR_SCALE, W_ROT_POUR = 1.0, 1.5, 1.0   # stricter while pouring
    RCV_W_FREE, RCV_W_SCALE, W_RCV_SPIN = 0.5, 1.0, 0.5
    W_FLIP_PALM = 0.3
    W_FLIP_HAND = 0.2
    W_RATE_PALM_SRC = 0.03
    W_RATE_PALM_RCV = 0.05
    BANG_DEAD = 0.1
    W_BANG_SRC = 1.5
    W_BANG_RCV = 2.0
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
    W_PALM_SAT_SRC = 3.0    # KEEP LIST (per-arm palm saturation stays exactly as measured)
    W_PALM_SAT_RCV = 1.5
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35      # rad (20 deg) - the env's own receiver-upright limit, reused for the
                            # source in the finish pose, so no new attitude constant appears
    # iter_15 posture terms, KEPT but demoted to rudders (iter_16 change 3). The iter_15 run proved
    # they cannot open the wrist range by themselves: the penalty doubled and limit_hit ROSE 0.21 ->
    # 0.43 while the pour collapsed, corr(limit_hit, in_target) = +0.90. The geometry has to move
    # first; these two then only pick between postures that are already feasible.
    MARGIN_FREE = 0.30
    W_ARM_LIMIT = 0.4       # 1.2 -> 0.4: -0.16/step on the source at the measured margin 0.090
    QDD_FREE, QDD_SCALE = 2.5, 10.3
    QDD_PEAK_FREE, QDD_PEAK_SCALE = 12.8, 12.8
    W_ARM_SHAKE = 0.4       # 1.0 -> 0.4: -0.11/step, since diag_src_qdd never moved under 1.0
    # ============ iter_16 (1): THE RENDEZVOUS CORRIDOR ==============================================
    # All of it is relative to the two SPAWN positions, so nothing here is an absolute coordinate and
    # spawn randomisation (and a mirrored left/right setup) is handled by construction.
    #   span   = rcv_spawn_y - src_spawn_y  (+0.324 m measured: the receiver sits on the +y side)
    #   frac   = (y - src_spawn_y) / span   -> 0 at the SOURCE spawn, 1 at the RECEIVER spawn
    # MEET_FULL = 1/3 is the operator's target band ("as far as the 1/3 point measured from the source
    # spawn"), which at the measured spawns is y = -0.051: exactly the probe point (0.37, -0.05, 0.45)
    # where the best wrist margin is 0.500 instead of "no solution". Further across (y = -0.15,
    # margin 0.628) keeps full credit, which is why the ramp saturates instead of peaking.
    # Measured today: frac_rcv = 0.824 at the pour, frac_src = 0.633.
    SPAN_MIN = 0.05         # numerical floor on |span| only; never reached in this task (0.324 m)
    MEET_FULL = 1.0 / 3.0
    W_MEET = 5.0            # same standing as align_mouths (the strongest guidance term) and far below
                            # success 20. ~0.4/step at today's frac_rcv 0.82, ~2.4/step effective once
                            # the receiver is at the 1/3 point: a slope, not a plateau. Gated hard on
                            # BOTH cups being grasped and lifted, so parking the receiver cup next to a
                            # source cup that is still on the table pays nothing.
    W_SRC_CROSS = 0.8       # the mirror charge: the source cup past the same 1/3 point. -0.36/step at
                            # the measured frac_src 0.633, inside the -1/step budget for new terms, and
                            # it disappears by itself as soon as the rendezvous moves.

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

    # ---- spout gates: how close the pouring lip is to the receiver mouth (VERBATIM iter_13..15) ----
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
    carry_soft_rcv = g_rcv * torch.clamp(h_rcv / TILT_LIFT_H, max=1.0)   # iter_16: same shape, receiver
    allowed = TILT_FREE_TABLE + carry_soft_src * torch.clamp(theta_pre - TILT_FREE_TABLE, min=0.0)
    allowed = allowed + carry_soft_src * lead_gate * torch.clamp(release_tilt - theta_pre, min=0.0)
    allowed = allowed + carry_soft_src * perm_xy * (theta_max - release_tilt)
    tilt_excess_src = torch.clamp(ctx.src_cup_tilt - allowed, min=0.0)
    src_up = torch.exp(-(tilt_excess_src / TILT_TOL) ** 2)
    pose_ok_src = src_up
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_rcv = upright_rcv

    # ============ iter_16 (1): where the two cups meet, in spawn-relative lateral coordinates =======
    span_y = ctx.rcv_cup_spawn_pos[:, 1] - ctx.src_cup_spawn_pos[:, 1]
    span_sgn = torch.where(span_y >= 0.0, torch.ones_like(span_y), -torch.ones_like(span_y))
    span_safe = span_sgn * torch.clamp(span_y.abs(), min=SPAN_MIN)   # signed, never zero
    frac_rcv = (ctx.rcv_cup_pos[:, 1] - ctx.src_cup_spawn_pos[:, 1]) / span_safe   # 1 = its own spawn
    frac_src = (ctx.src_cup_pos[:, 1] - ctx.src_cup_spawn_pos[:, 1]) / span_safe   # 0 = its own spawn
    # late ramp, so the gradient exists at today's 0.82 and TRIPLES as the 1/3 point is approached;
    # it saturates (never peaks) past the 1/3 point, so coming further across is never punished here.
    meet_q = _late_ramp(1.0 - frac_rcv, 0.0, 1.0 - MEET_FULL)
    src_cross = torch.clamp((frac_src - MEET_FULL) / (1.0 - MEET_FULL), 0.0, 1.0)
    # income only when BOTH cups are carried: no parking value for a lone receiver cup
    meet_cross = W_MEET * carry_soft_rcv * carry_soft_src * upright_rcv * not_nested * meet_q

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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # measured rcv_progress 0.75

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

    # ---- receiver lift band (unchanged - measured 0.097 m, inside the 0.10 m band) ------------------
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * rcv_band * (0.5 + 0.5 * upright_rcv)

    # ============ iter_16 (2): the receiver is untied once it is CARRYING ===========================
    # home_near (sigma 0.12 m) multiplies rcv_disc, and rcv_disc multiplies converge + carry_src +
    # align_mouths + hold_rcv. Crossing to the 1/3 point means a displacement of ~0.215 m, at which
    # home_near = 0.04 and rcv_disc falls 0.87 -> 0.38: about -2.8/step. That, not the small rcv_home
    # penalty, is what priced the crossing out of the market. home_eff releases the factor to 1 as
    # soon as the receiver holds its cup 3 cm off the table, so the pour ladder keeps its measured
    # scale while the cup is free to travel. The anti-tipping protection is untouched:
    # cup_disturb_rcv (slide + tilt), rcv_spin, cup_fling and upright_rcv all stay as they were.
    d_home = rcv["disp"]
    home_near = torch.exp(-(d_home / HOME_SIGMA) ** 2)
    home_eff = home_near + (1.0 - home_near) * carry_soft_rcv
    # stillness: full strictness only once the receiver has ARRIVED (meet_q). While it transits, its
    # free speed rises to the source cup's own existing limit SRC_V_FREE - no new speed constant, and
    # no relaxation of any cup speed limit at the moment of the pour.
    still_free_eff = STILL_FREE + (SRC_V_FREE - STILL_FREE) * (1.0 - meet_q)
    still_over_rcv = torch.tanh(torch.clamp(rcv["v_lin"] - still_free_eff, min=0.0) / STILL_SCALE)
    still_f = 1.0 - still_over_rcv
    rcv_disc = RCV_DISC_FLOOR + (1.0 - RCV_DISC_FLOOR) * rcv_band * home_eff
    # rcv_home keeps ONLY its original job: do not drag the cup across the table before lifting it
    rcv_home_pen = rcv["held"] * (1.0 - carry_soft_rcv) \
        * torch.tanh(torch.clamp(d_home - HOME_FREE, min=0.0) / HOME_SCALE)
    still_w = W_STILL_RCV + W_STILL_POUR * up_w_src * meet_q   # strict at the rendezvous, not en route
    rcv_still_pen = still_w * rcv["held"] * still_over_rcv
    hold_rcv = W_HOLD_RCV * g_rcv * rcv_band * home_eff * still_f * (0.4 + 0.6 * carry_soft_src)

    # ---- convergence (unchanged): closing the gap, on the quantity the success test uses -----------
    d_cups_xy = (ctx.src_cup_pos[:, :2] - ctx.rcv_cup_pos[:, :2]).norm(dim=-1)
    conv_lin = torch.clamp((CONV_FAR - d_cups_xy) / (CONV_FAR - CONV_NEAR), 0.0, 1.0)
    conv_close = torch.exp(-(torch.clamp(d_cups_xy - CONV_NEAR, min=0.0) / CONV_SIGMA) ** 2)
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.77 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * not_nested * conv_shape

    # carry / align keep their iter_06 shape and weights on the DIRECTED spout
    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * not_nested
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * not_nested
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- TILT: three segments of the SIGNED angle, each with a stricter SPOUT requirement -----------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_hold = carry_soft_src * beads_left * not_nested * geo["sep_ok"] * dir_soft
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

    # control gate WITHOUT adr_progress: both cups grasped and >= 5.5 cm up (measured 0.64)
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

    # ---- aiming at the moment of release (VERBATIM iter_13..15) ------------------------------------
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

    # ---- arm posture and tremor: kept, demoted to rudders (iter_16 change 3) -----------------------
    margin_pen_src = _limit_pen(ctx.src_arm_limit_margin, MARGIN_FREE)
    margin_pen_rcv = _limit_pen(ctx.rcv_arm_limit_margin, MARGIN_FREE)
    arm_limit_src = g_src * margin_pen_src
    arm_limit_rcv = g_rcv * margin_pen_rcv
    shake_src = g_src * _shake_pen(ctx.src_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)
    shake_rcv = g_rcv * _shake_pen(ctx.rcv_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)

    # ---- iter_14 FULL TRANSFER AND FINISH (kept verbatim: both worked) -----------------------------
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    emptied = torch.clamp(1.0 - fill, 0.0, 1.0)
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)
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
    # this is unreachable before the transfer is essentially done - it is the hold, not parking.
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
        # ★ iter_16 (1): bring the RECEIVER cup into the source arm's own region. Full at the 1/3
        # point of the spawn-to-spawn span (measured y = -0.05, wrist margin 0.500 instead of "no
        # solution"). ~0.4/step today, ~2.4/step effective at the target - below success 20.
        "meet_cross": meet_cross,
        # ★ iter_16 (1): and stop the SOURCE cup from crossing the midline to do it (measured today:
        # frac 0.633 -> -0.36/step). This is the only new charge; the -1/step budget holds.
        "src_cross": -W_SRC_CROSS * carry_soft_src * src_cross,
        "converge": converge,
        "carry_src": W_CARRY * carry,
        "align_mouths": W_ALIGN * align,
        "tilt_aim": W_TILT_AIM * tilt_aim,
        "tilt_near": W_TILT_NEAR * tilt_near,
        "pour_tilt": W_TILT_POUR * pour_tilt,
        "pour_pose": W_POUR_POSE * pour_pose,
        "pour_delta": W_POUR * pour_paid,
        "flow_excess": -W_FLOW * flow_excess,
        "transferred": W_TRANSFER * in_t * not_nested * upright_rcv * clean_f,
        "transfer_tail": W_TAIL * fill_tail * not_nested * upright_rcv * clean_f,
        "src_emptied": W_EMPTY * empty_q * fill_ok * not_nested * clean_f,
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
        # ★ iter_16 (3): 1.2 -> 0.4. Kept as a rudder between feasible postures; it cannot create
        # range that the rendezvous geometry does not give (corr(limit_hit, in_target) = +0.90).
        "arm_limit_src": -W_ARM_LIMIT * arm_limit_src,
        "arm_limit_rcv": -W_ARM_LIMIT * arm_limit_rcv,
        # ★ iter_16 (3): 1.0 -> 0.4. diag_src_qdd never moved at 1.0, so it buys nothing at full price.
        "arm_shake_src": -W_ARM_SHAKE * shake_src,
        "arm_shake_rcv": -W_ARM_SHAKE * shake_rcv,
        "act_flip": -W_FLIP_PALM * (g_src * flip_palm_src + g_rcv * flip_palm_rcv),
        "hand_flip": -W_FLIP_HAND * ctrl_gate * flip_hand,
        "carry_rate": -(W_RATE_PALM_SRC * g_src * rate_palm_src
                        + W_RATE_PALM_RCV * g_rcv * rate_palm_rcv
                        + W_RATE_HAND_CARRY * ctrl_gate * rate_hand),
        "palm_bang_src": -W_BANG_SRC * g_src * bang_src,
        "palm_bang_rcv": -W_BANG_RCV * g_rcv * bang_rcv,
        "early_tilt": -W_EARLY_TILT * early_tilt,
        "mis_tilt": -W_MISDIR * mis_tilt,
        "pour_dir_pen": -W_DIR_PEN * pour_dir_pen,
        "near_nested": -W_NEAR_NEST * near_nest,
        "cup_disturb_src": -W_SLIDE_SRC * src["slide"],
        "cup_disturb_rcv": -(W_SLIDE_RCV * rcv["slide"] + W_TILT_PEN_RCV * tilt_pen_rcv),
        # ★ iter_16 (2): only while the receiver cup is NOT yet carried, and at half weight
        "rcv_home": -W_HOME_RCV * rcv_home_pen,
        # ★ iter_16 (2): strict stillness only at the rendezvous; in transit the free speed is the
        # source cup's own SRC_V_FREE, so no cup speed limit is loosened at the pour itself
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
    comps["diag_fill_tail"] = fill_tail
    comps["diag_empty_q"] = empty_q
    comps["diag_finish_q"] = finish_q
    comps["diag_settle_still"] = settle_still
    comps["diag_emptied"] = emptied
    comps["diag_src_margin_min"] = torch.clamp(ctx.src_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_rcv_margin_min"] = torch.clamp(ctx.rcv_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_src_limit_hit"] = (comps["diag_src_margin_min"] < 0.05).to(dtype)   # i15 run: 0.21->0.43
    comps["diag_rcv_limit_hit"] = (comps["diag_rcv_margin_min"] < 0.05).to(dtype)
    comps["diag_src_qdd"] = ctx.src_arm_qdd.abs().mean(dim=-1)
    comps["diag_rcv_qdd"] = ctx.rcv_arm_qdd.abs().mean(dim=-1)
    comps["diag_src_qdd_max"] = ctx.src_arm_qdd.abs().max(dim=-1).values
    comps["diag_rcv_qdd_max"] = ctx.rcv_arm_qdd.abs().max(dim=-1).values
    comps["diag_rate_palm_rcv"] = rate_palm_rcv
    # ★ iter_16 diagnostics: WHERE the rendezvous happens, in spawn-relative units (0 = source spawn
    # side, 1 = receiver spawn side). Target: diag_frac_rcv <= 0.333, diag_frac_src <= 0.333.
    comps["diag_frac_rcv"] = frac_rcv            # i15 trace at the pour: 0.824
    comps["diag_frac_src"] = frac_src            # i15 trace at the pour: 0.633
    comps["diag_meet_q"] = meet_q                # 0.167 today, 1.0 at the target band
    comps["diag_home_eff"] = home_eff            # 1 once the receiver carries its cup
    comps["diag_rcv_travel"] = d_home            # receiver cup xy displacement [m]: 0.057 today
    comps["diag_src_travel"] = src["disp"]       # source cup xy displacement [m]: 0.21 today
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0.000122, 0.000134, 0.00375, 0.00841, 0.00511, 0.0027, 0.0022, 0.00458, 0.00326]  min 0 · mean 0.00393 · max 0.0222
bead/spill: [1.22e-05, 0.00854, 0.00265, 0.033, 0.0392, 0.0339, 0.0348, 0.0264, 0.0438, 0.0406]  min 1.22e-05 · mean 0.0295 · max 0.077
done/drop: [0, 0, 0, 0, 0, 0.000488, 0.000244, 0.000244, 0.000244, 0]  min 0 · mean 0.000206 · max 0.00195
done/mimic_runaway: [0, 0.000244, 0, 0, 0, 0, 0, 0, 0.000244, 0]  min 0 · mean 9.75e-05 · max 0.000977
episode_lengths/step: [85.1, 819, 856, 848, 819, 762, 759, 773, 796, 780]  min 85.1 · mean 806 · max 891
reward/act_flip: [0, -0.143, -0.0739, -0.108, -0.0824, -0.104, -0.0905, -0.0865, -0.0987, -0.118]  min -0.177 · mean -0.103 · max 0
reward/action_rate: [-0.0322, -0.0217, -0.0153, -0.0146, -0.0125, -0.0124, -0.0115, -0.0107, -0.0118, -0.0126]  min -0.0322 · mean -0.0142 · max -0.0104
reward/aim_gate: [0, 0.409, 0.506, 0.663, 0.585, 0.569, 0.613, 0.61, 0.547, 0.618]  min 0 · mean 0.525 · max 0.706
reward/aimed_tilt: [0, 0, 0.0168, 0.278, 0.402, 0.346, 0.278, 0.307, 0.295, 0.294]  min 0 · mean 0.243 · max 0.472
reward/align_mouths: [0, 2.81, 2.94, 3.18, 2.91, 2.82, 2.86, 3, 2.88, 3.12]  min 0 · mean 2.86 · max 3.34
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/approach_lip: [0, 0.0109, -0.0024, 0.0174, 0.0311, -0.0103, -0.0033, -0.0199, -0.0101, 0.00913]  min -0.197 · mean -0.00336 · max 0.0629
reward/arm_limit_rcv: [0, -0.0196, -0.0342, -0.0354, -0.0181, -0.0224, -0.0249, -0.0184, -0.0226, -0.0217]  min -0.0539 · mean -0.0229 · max 0
reward/arm_limit_src: [0, -0.113, -0.122, -0.113, -0.108, -0.0952, -0.103, -0.103, -0.1, -0.102]  min -0.128 · mean -0.104 · max 0
reward/arm_shake_rcv: [0, -0.0722, -0.0643, -0.0673, -0.0752, -0.077, -0.0815, -0.0714, -0.0634, -0.0592]  min -0.12 · mean -0.0704 · max 0
reward/arm_shake_src: [0, -0.0818, -0.0755, -0.076, -0.0754, -0.0736, -0.0774, -0.0675, -0.074, -0.0816]  min -0.109 · mean -0.0759 · max 0
reward/away_rcv: [-0.104, -0.174, -0.167, -0.175, -0.173, -0.168, -0.162, -0.161, -0.162, -0.166]  min -0.18 · mean -0.167 · max -0.104
reward/carry_rate: [0, -0.189, -0.126, -0.132, -0.112, -0.115, -0.111, -0.101, -0.109, -0.123]  min -0.237 · mean -0.126 · max 0
reward/carry_src: [0, 1.81, 1.9, 1.97, 1.91, 1.82, 1.86, 1.88, 1.83, 1.93]  min 0 · mean 1.83 · max 2.08
reward/close_rcv: [0.0678, 0.0791, 0.0849, 0.0748, 0.0753, 0.0786, 0.0803, 0.0842, 0.0872, 0.0871]  min 0.0674 · mean 0.0823 · max 0.0984
reward/close_src: [0.0422, 0.0987, 0.0699, 0.0702, 0.0745, 0.0825, 0.088, 0.0938, 0.0859, 0.0767]  min 0.0422 · mean 0.0836 · max 0.101
reward/contact_push: [-0.00592, -0.00338, -0.00179, -0.00125, -0.00147, -0.00253, -0.0028, -0.00321, -0.00456, -0.00443]  min -0.114 · mean -0.00309 · max -0.000624
reward/converge: [0, 2.12, 2.22, 2.25, 2.26, 2.14, 2.2, 2.23, 2.17, 2.26]  min 0 · mean 2.18 · max 2.37
reward/cup_collision: [0, -0.000828, -0.000244, -9.59e-05, -0.000301, -0.00106, -0.00163, -0.00214, -0.00292, -0.0026]  min -0.00614 · mean -0.00112 · max 0
reward/cup_disturb_rcv: [-0.0263, -0.0717, -0.067, -0.0665, -0.077, -0.0794, -0.0853, -0.0871, -0.0883, -0.0852]  min -0.234 · mean -0.0814 · max -0.0263
reward/cup_disturb_src: [-0.0564, -0.0275, -0.0173, -0.0174, -0.0168, -0.0216, -0.0224, -0.0211, -0.0242, -0.0233]  min -0.268 · mean -0.0224 · max -0.00725
reward/cup_fling: [-0.000604, -0.00765, -0.00579, -0.00726, -0.011, -0.0125, -0.0115, -0.00906, -0.00983, -0.00912]  min -0.0165 · mean -0.00935 · max -0.000604
reward/diag_bang_rcv: [0.285, 0.211, 0.128, 0.121, 0.113, 0.106, 0.113, 0.0933, 0.0929, 0.101]  min 0.0855 · mean 0.122 · max 0.295
reward/diag_bang_src: [0.192, 0.143, 0.106, 0.132, 0.0993, 0.12, 0.109, 0.104, 0.104, 0.107]  min 0.0759 · mean 0.114 · max 0.192
reward/diag_clean: [1, 0.976, 0.989, 0.882, 0.866, 0.9, 0.889, 0.932, 0.877, 0.869]  min 0.772 · mean 0.914 · max 1
reward/diag_close_speed: [0.00979, -0.00187, 0.00259, 0.0146, 0.0125, 0.000696, 0.00198, -0.00101, 0.00687, 0.011]  min -0.049 · mean 0.00422 · max 0.0201
reward/diag_ctrl_gate: [0, 0.71, 0.725, 0.745, 0.728, 0.693, 0.693, 0.721, 0.712, 0.744]  min 0 · mean 0.716 · max 0.826
reward/diag_emptied: [6.1e-05, 0.0128, 0.00573, 0.0473, 0.0625, 0.0495, 0.0463, 0.0298, 0.0444, 0.0485]  min 6.1e-05 · mean 0.037 · max 0.0997
reward/diag_empty_q: [0, 0.00588, 0.00232, 0.0081, 0.013, 0.0135, 0.0102, 0.00758, 0.00582, 0.00558]  min 0 · mean 0.00811 · max 0.0236
reward/diag_fill_tail: [0, 0, 0, 0, 2.62e-05, 1.21e-05, 0, 0, 8.07e-05, 0]  min 0 · mean 1.15e-05 · max 0.000282
reward/diag_finish_q: [0, 0.00294, 0.00116, 0.00405, 0.00654, 0.00676, 0.0051, 0.00379, 0.00295, 0.00279]  min 0 · mean 0.00406 · max 0.0118
reward/diag_flip_palm: [1.25, 0.612, 0.33, 0.431, 0.329, 0.402, 0.343, 0.32, 0.362, 0.426]  min 0.281 · mean 0.41 · max 1.25
reward/diag_flow_beads: [0, 0, 0.000244, 0, 0, 0.000244, 0.000488, 0.000244, 0, 0.000488]  min 0 · mean 0.000184 · max 0.00195
reward/diag_frac_rcv: [0.999, 0.693, 0.634, 0.575, 0.668, 0.641, 0.612, 0.582, 0.551, 0.431]  min -0.0434 · mean 0.623 · max 1.21
reward/diag_frac_src: [0.00145, -0.185, 0.2, 0.133, 0.12, 0.217, 0.254, 0.19, 0.099, 0.164]  min -0.645 · mean 0.145 · max 1.11
reward/diag_home_eff: [1, 0.977, 0.977, 0.972, 0.982, 0.984, 0.982, 0.986, 0.976, 0.975]  min 0.952 · mean 0.978 · max 1
reward/diag_lead_gate: [0, 0.563, 0.67, 0.753, 0.727, 0.684, 0.724, 0.718, 0.683, 0.73]  min 0 · mean 0.659 · max 0.801
reward/diag_lift_budget: [0.24, 0.254, 0.262, 0.264, 0.265, 0.265, 0.262, 0.263, 0.264, 0.262]  min 0.24 · mean 0.261 · max 0.267
reward/diag_lip_xy: [0.291, 0.254, 0.106, 0.156, 0.179, 0.107, 0.1, 0.109, 0.127, 0.141]  min 0.0798 · mean 0.162 · max 0.601
reward/diag_meet_q: [0.000867, 0.392, 0.465, 0.573, 0.429, 0.475, 0.515, 0.553, 0.604, 0.652]  min 0.000867 · mean 0.497 · max 0.687
reward/diag_mis_deg: [0.535, 3.41, 7.05, 4.74, 6.6, 5.7, 7.24, 5.16, 6.55, 8.78]  min 0.535 · mean 6.37 · max 13.1
reward/diag_palm_sat: [0.442, 0.304, 0.426, 0.373, 0.381, 0.379, 0.359, 0.366, 0.345, 0.341]  min 0.291 · mean 0.372 · max 0.589
reward/diag_palm_sat_rcv: [0.535, 0.411, 0.461, 0.419, 0.432, 0.433, 0.383, 0.425, 0.45, 0.464]  min 0.375 · mean 0.433 · max 0.535
reward/diag_perm: [0, 0.404, 0.504, 0.66, 0.582, 0.567, 0.612, 0.606, 0.545, 0.617]  min 0 · mean 0.522 · max 0.705
reward/diag_perm_at_tilt: [0, 0, 0.0648, 0.157, 0.212, 0.216, 0.18, 0.171, 0.173, 0.173]  min 0 · mean 0.144 · max 0.245
reward/diag_pour_phase: [0, 0.225, 0.365, 0.371, 0.317, 0.339, 0.305, 0.345, 0.313, 0.304]  min 0 · mean 0.294 · max 0.395
reward/diag_pre_release: [0, 0.00747, 0.0976, 0.198, 0.256, 0.245, 0.215, 0.205, 0.223, 0.209]  min 0 · mean 0.178 · max 0.29
reward/diag_rate_palm_rcv: [4.51, 2.76, 1.7, 1.5, 1.34, 1.3, 1.32, 1.08, 1.13, 1.24]  min 1.01 · mean 1.53 · max 4.51
reward/diag_rcv_limit_hit: [0, 0, 0, 0, 0, 0, 0.000244, 0, 0.000244, 0]  min 0 · mean 9.88e-05 · max 0.0022
reward/diag_rcv_margin_min: [0.245, 0.23, 0.199, 0.2, 0.233, 0.222, 0.222, 0.236, 0.233, 0.235]  min 0.174 · mean 0.228 · max 0.336
reward/diag_rcv_qdd: [6.16, 5.3, 5.06, 5.19, 5.5, 5.73, 5.84, 5.59, 5.16, 5.11]  min 4.51 · mean 72.2 · max 1.77e+05
reward/diag_rcv_qdd_max: [14.4, 12.8, 12.4, 12.5, 13.2, 14.1, 14.4, 13.8, 12.9, 12.9]  min 11.3 · mean 213 · max 5.32e+05
reward/diag_rcv_travel: [0.000671, 0.11, 0.125, 0.148, 0.128, 0.133, 0.144, 0.155, 0.16, 0.21]  min 0.000671 · mean 0.147 · max 0.41
reward/diag_settle_still: [0.975, 0.78, 0.796, 0.787, 0.758, 0.75, 0.758, 0.775, 0.76, 0.763]  min 0.549 · mean 0.767 · max 0.975
reward/diag_src_limit_hit: [0, 0.00366, 0.0356, 0.0164, 0.0154, 0.0295, 0.0486, 0.0693, 0.0405, 0.0322]  min 0 · mean 0.0305 · max 0.117
reward/diag_src_margin_min: [0.193, 0.113, 0.108, 0.117, 0.119, 0.128, 0.123, 0.125, 0.127, 0.125]  min 0.104 · mean 0.122 · max 0.193
reward/diag_src_qdd: [6.11, 6.42, 5.91, 5.93, 6.13, 6.5, 6.48, 5.82, 6.25, 6.38]  min 5.43 · mean 6.39 · max 99.6
reward/diag_src_qdd_max: [13.7, 15.5, 13.9, 14.6, 15.8, 17.1, 16.9, 14.7, 15.7, 16.2]  min 13 · mean 16.2 · max 451
reward/diag_src_travel: [0.00119, 0.229, 0.108, 0.168, 0.21, 0.117, 0.122, 0.124, 0.131, 0.108]  min 0.00119 · mean 0.161 · max 0.574
reward/diag_src_v: [0.0359, 0.247, 0.163, 0.195, 0.226, 0.201, 0.189, 0.182, 0.229, 0.19]  min 0.0359 · mean 0.215 · max 6.41
reward/diag_src_w: [0.501, 1.22, 1.13, 1.53, 1.52, 1.55, 1.54, 1.33, 1.49, 1.42]  min 0.501 · mean 1.95 · max 474
reward/diag_theta_deg: [0.487, 20, 28.8, 30.8, 29.5, 31.2, 27.6, 28.5, 29.1, 27.2]  min 0.487 · mean 26.6 · max 33.6
reward/diag_unperm_at_tilt: [0, 0.00747, 0.0328, 0.0406, 0.0441, 0.029, 0.0351, 0.0342, 0.0492, 0.0357]  min 0 · mean 0.0334 · max 0.0686
reward/early_tilt: [-0.00222, -0.0693, -0.065, -0.061, -0.0702, -0.0772, -0.0671, -0.061, -0.0687, -0.062]  min -0.208 · mean -0.0671 · max -0.00222
reward/flow_excess: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00195 · mean -1.12e-05 · max 0
reward/grasp_rcv: [0.000306, 1.35, 1.35, 1.33, 1.31, 1.28, 1.22, 1.25, 1.29, 1.34]  min 0.000306 · mean 1.29 · max 1.48
reward/grasp_src: [0.0043, 1.45, 1.47, 1.5, 1.46, 1.41, 1.44, 1.44, 1.42, 1.47]  min 0.0043 · mean 1.45 · max 1.72
reward/hand_flip: [0, -0.124, -0.0826, -0.0662, -0.0533, -0.043, -0.0387, -0.0377, -0.0464, -0.0554]  min -0.196 · mean -0.0636 · max 0
reward/hand_foreign: [-0.0205, -0.00188, -0.0014, -0.00163, -0.00105, -0.00185, -0.000868, -0.00134, -0.000774, -0.000772]  min -0.0205 · mean -0.00132 · max -0.000205
reward/hold_rcv: [0, 0.612, 0.628, 0.616, 0.617, 0.586, 0.611, 0.622, 0.589, 0.591]  min 0 · mean 0.597 · max 0.689
reward/lift_rcv: [0, 1.45, 1.44, 1.48, 1.42, 1.38, 1.35, 1.38, 1.37, 1.43]  min 0 · mean 1.4 · max 1.57
reward/lift_src: [0, 2.4, 2.34, 2.5, 2.26, 2.24, 2.34, 2.4, 2.38, 2.48]  min 0 · mean 2.36 · max 2.93
reward/meet_cross: [0, 1.68, 1.97, 2.52, 1.83, 2.01, 2.17, 2.34, 2.52, 2.78]  min 0 · mean 2.1 · max 2.92
reward/mis_tilt: [0, -0.0159, -0.0866, -0.0451, -0.0891, -0.069, -0.105, -0.0643, -0.0898, -0.138]  min -0.196 · mean -0.0813 · max 0
reward/near_nested: [0, -0.0211, -0.0253, -0.0049, -0.00433, -0.00779, -0.0204, -0.0164, -0.00332, -0.00501]  min -0.0364 · mean -0.00813 · max 0
reward/nested: [0, -0.00439, -0.00342, -0.000488, -0.000488, -0.00195, -0.00146, -0.00146, -0.00146, -0.000977]  min -0.0176 · mean -0.00177 · max 0
reward/over_lift_src: [0, -0.0132, -0.083, -0.0538, -0.154, -0.109, -0.0861, -0.0684, -0.0712, -0.0649]  min -0.187 · mean -0.0772 · max 0
reward/overtilt: [0, 0, 0, -0.000461, -8.27e-05, 0, -4.77e-06, 0, -0.000634, -0.000459]  min -0.0019 · mean -0.000223 · max 0
reward/palm_bang_rcv: [0, -0.341, -0.196, -0.184, -0.177, -0.164, -0.183, -0.15, -0.148, -0.164]  min -0.421 · mean -0.192 · max 0
reward/palm_bang_src: [0, -0.169, -0.136, -0.177, -0.129, -0.155, -0.14, -0.137, -0.136, -0.146]  min -0.204 · mean -0.145 · max 0
reward/palm_sat_rcv: [0, -0.486, -0.565, -0.512, -0.544, -0.546, -0.482, -0.541, -0.574, -0.597]  min -0.628 · mean -0.539 · max 0
reward/palm_sat_src: [0, -0.643, -0.991, -0.863, -0.876, -0.856, -0.826, -0.833, -0.762, -0.769]  min -1.64 · mean -0.843 · max 0
reward/pour_delta: [0, 0, 0.00183, 0, 0, 0.00183, 0.00366, 0.00183, 0, 0.00366]  min -0.00549 · mean 0.00117 · max 0.0146
reward/pour_dir_pen: [0, -0.0455, -0.0801, -0.0955, -0.0293, -0.182, -0.12, -0.0883, -0.0478, -0.0406]  min -0.901 · mean -0.0998 · max 0
reward/pour_pose: [0, 0, 1.09e-05, 0.248, 0.354, 0.253, 0.229, 0.182, 0.226, 0.217]  min 0 · mean 0.175 · max 0.366
reward/pour_tilt: [0, 0, 0, 0.303, 0.417, 0.302, 0.279, 0.215, 0.278, 0.263]  min 0 · mean 0.211 · max 0.429
reward/rcv_home: [0, -0.000179, -0.0003, -5.03e-05, -0.000536, -0.000807, -0.00135, -0.000476, -0.0015, -0.000463]  min -0.00189 · mean -0.000474 · max 0
reward/rcv_progress: [0.234, 0.777, 0.789, 0.769, 0.784, 0.777, 0.786, 0.787, 0.79, 0.793]  min 0.234 · mean 0.783 · max 0.868
reward/rcv_spin: [0, -0.0926, -0.0913, -0.0872, -0.0977, -0.0968, -0.0949, -0.0954, -0.0889, -0.0874]  min -0.176 · mean -0.0923 · max 0
reward/rcv_still: [0, -0.136, -0.149, -0.175, -0.162, -0.169, -0.168, -0.172, -0.186, -0.203]  min -0.213 · mean -0.168 · max 0
reward/reach_rcv: [0.801, 0.794, 0.812, 0.786, 0.79, 0.801, 0.813, 0.819, 0.828, 0.82]  min 0.781 · mean 0.81 · max 0.902
reward/reach_src: [0.598, 0.762, 0.685, 0.671, 0.677, 0.685, 0.694, 0.71, 0.702, 0.68]  min 0.598 · mean 0.7 · max 0.766
reward/rot_pour: [0, -0.0188, -0.0422, -0.0679, -0.0726, -0.0662, -0.0657, -0.0606, -0.0641, -0.0667]  min -0.0834 · mean -0.0526 · max 0
reward/rush_near_cup: [-0.236, -0.0373, -0.0351, -0.0333, -0.0336, -0.0387, -0.0338, -0.0341, -0.036, -0.033]  min -0.236 · mean -0.0352 · max -0.0146
reward/settle_finish: [0, 0.000149, 1.96e-06, 0.000362, 0.00107, 0.000713, 0.000423, 0.00062, 0.000277, 0.000144]  min 0 · mean 0.00046 · max 0.00341
reward/spill_delta: [0, -0.0498, -0.0146, -0.0352, -0.0498, -0.0498, -0.0557, -0.0234, -0.0234, -0.041]  min -0.155 · mean -0.041 · max 0
reward/src_cross: [0, -0.0411, -0.033, -0.0288, -0.0769, -0.0477, -0.0502, -0.0423, -0.0307, -0.0213]  min -0.409 · mean -0.0506 · max 0
reward/src_emptied: [0, 0, 0, 0, 9.24e-05, 4.46e-05, 2.8e-05, 0, 8.3e-05, 6.54e-05]  min 0 · mean 3.11e-05 · max 0.00073
reward/src_speed: [0, -0.0653, -0.0507, -0.0541, -0.0648, -0.0703, -0.0635, -0.059, -0.0613, -0.0596]  min -0.262 · mean -0.0639 · max 0
reward/src_spin: [0, -0.0269, -0.025, -0.0364, -0.0485, -0.0476, -0.0485, -0.0418, -0.0451, -0.0475]  min -0.0607 · mean -0.0392 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_aim: [0, 0.437, 0.671, 0.646, 0.608, 0.575, 0.555, 0.65, 0.6, 0.556]  min 0 · mean 0.541 · max 0.726
reward/tilt_near: [0, 0, 0.0138, 0.619, 0.955, 0.834, 0.671, 0.681, 0.731, 0.74]  min 0 · mean 0.573 · max 1.13
reward/total: [2.03, 15.5, 16, 18.5, 17.7, 17, 17.2, 17.9, 17.8, 18.6]  min 1.61 · mean 16.9 · max 19.4
reward/transfer_tail: [0, 0, 0, 0, 1.64e-05, 1.91e-05, 0, 0, 0, 0]  min 0 · mean 8.22e-06 · max 0.000431
reward/transferred: [0, 0.000364, 0.000113, 0.00779, 0.0212, 0.015, 0.00565, 0.00522, 0.00814, 0.0124]  min 0 · mean 0.00894 · max 0.0459
reward/unaimed_release: [0, -0.00343, -0.0783, -0.0965, -0.104, -0.0609, -0.0786, -0.0844, -0.137, -0.0889]  min -0.185 · mean -0.0765 · max 0
rewards/step: [82.6, 1.24e+04, 1.37e+04, 1.59e+04, 1.42e+04, 1.25e+04, 1.27e+04, 1.38e+04, 1.42e+04, 1.41e+04]  min 82.6 · mean 1.37e+04 · max 1.69e+04
task/aim_dist: [0.319, 0.304, 0.191, 0.223, 0.225, 0.466, 0.387, 0.44, 0.223, 0.298]  min 0.173 · mean 0.261 · max 0.687
task/cup_collision_rate: [0, 0.000732, 0.000488, 0, 0.000488, 0.00171, 0.00171, 0.00562, 0.00146, 0.00366]  min 0 · mean 0.00169 · max 0.00928
task/cups_center_dist: [0.32, 0.318, 0.209, 0.242, 0.245, 0.486, 0.403, 0.46, 0.244, 0.317]  min 0.193 · mean 0.278 · max 0.706
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.00146, 0.000977, 0.000244, 0, 0, 0.00171, 0.000977, 0.000977, 0.000977]  min 0 · mean 0.000887 · max 0.00879
task/rcv_cup_lift: [8.11e-05, 0.0747, 0.0641, 0.073, 0.0615, 0.0651, 0.0561, 0.0621, 0.0629, 0.0669]  min 8.11e-05 · mean 0.0695 · max 0.294
task/rcv_grasped: [0, 0.796, 0.825, 0.836, 0.813, 0.818, 0.812, 0.82, 0.828, 0.816]  min 0 · mean 0.818 · max 0.906
task/rcv_hand_foreign_rate: [0.0491, 0.00195, 0.00269, 0.000732, 0.0022, 0.000488, 0.0022, 0.000977, 0.000732, 0.00122]  min 0 · mean 0.00214 · max 0.0491
task/src_cup_lift: [0.000279, 0.218, 0.173, 0.209, 0.218, 0.327, 0.209, 0.271, 0.193, 0.217]  min 0.000279 · mean 0.205 · max 0.339
task/src_grasped: [0, 0.755, 0.78, 0.804, 0.76, 0.73, 0.727, 0.751, 0.753, 0.765]  min 0 · mean 0.764 · max 0.911
task/src_hand_foreign_rate: [0.00415, 0.000977, 0.000732, 0.00171, 0.00269, 0.00269, 0.00195, 0.00317, 0.00122, 0.00171]  min 0 · mean 0.00183 · max 0.00732
task/src_tilt_deg: [1.02, 23.2, 34.4, 34.7, 34.1, 34.8, 33.2, 35.8, 36.8, 37]  min 1.02 · mean 32.9 · max 40.2
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 17 operator notes — 각도를 처방하지 않는다: 붓는 각은 학습이 찾게 한다

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 사용자 지시 (09.24)

- ep2100 영상 관찰: **"파지가 살짝 불안하다"**, **"붓다가 비드가 들어가지 않았는데 컵을 세우려는 게 보인다"**.
- ★지시: **"각도에 대한 걸 제거해야 함. 적절한 각도는 학습이 찾게 해야 함."**
- 기존 원칙 유지: 실기는 안정 속도 틸팅(속도·제어 벌점 유지), 조준 창은 `ctx.cup_radius` 배수, 새 절대 거리 상수 금지.

## 1. i16g 결과 (파지 단계 웜스타트 rh2_i07-r3 ep400 + iter_16, 4096 env, e2100+)

- ★**파지 물려받기 성공**: 2100 epoch 내내 파지 0.75/0.82 유지, 들기 0.21~0.28 m.
- ★**만남 지점이 습관 없이 내려왔다**: `diag_frac_rcv` 1.047 → **0.57~0.63** 단조 하강(웜 계열은 0.83~0.91 고착 + 3회 왕복).
  손목 한계 접촉 0.05 이하·여유 0.126(웜 0.16~0.26 / 0.105). **손목 한계는 웜스타트 습관이 만든 것이 맞다.**
- 결정론 재생(16 env, step 600): ep900 기울기 23.4°·in_target 0.000 / **ep2100 기울기 87.6°·in_target 0.150·흘림 0.050·소스 잔량 0.756**.
  학습 곡선의 "기울기 35° 고정"은 탐색 잡음이 섞인 평균이었다 — 정책은 붓기를 배우는 중이다.
- entropy 21.9 → 3.5 로 붕괴. 성공(기준 0.80/0.25/20)은 전 구간 0.

## 2. ★왜 붓다가 컵을 세우나 (코드 + 로그)

- 마무리 자세 항은 원인이 아니다: `settle_finish` 스텝당 **+0.001**, `diag_finish_q` 0.004 — 전량 조건이 안 채워져 꺼져 있다.
- 실제 수입 구성(최근 100 epoch, 스텝당):

| 자세를 유지해서 버는 것 | 값 | 실제로 배달해서 버는 것 | 값 |
|---|---|---|---|
| tilt_near (W 6.0) | +0.731 | pour_delta (W 150 × Δin_target) | +0.001 |
| aimed_tilt (W 2.5) | +0.313 | transferred (W 5.0 × in_target) | +0.009 |
| pour_tilt (W 10) | +0.258 | transfer_tail (W 4.0) | +0.000 |
| **합계 ≈ 1.30** | | **합계 ≈ 0.010** | |

  **자세 수입이 배달 수입의 100 배다.** 비드가 나오지 않아도 각도만 유지하면 돈이 된다.
- 각도 사다리가 잔량에 연동돼 있다(`compute_reward.py:409-412`): `release_tilt = 75° + 25° × (1 − 잔량)`.
  ep2100 실측(잔량 0.756)에서 방출각 ≈ 81°, 실제 87.6° — 대역 위쪽 끝이다. 더 기울이면 `overtilt`(W 6.0, 방출각+여유 초과) 구간이고,
  비드가 안 나오면 대역에 남을 이유가 사라진다. [가설] 이것이 "붓다 마는" 되돌림의 정체다.
- 파지 불안: `close_src` 0.089(손이 파지점 3 cm 안에 머무는 시간이 짧다). 결정론 재생의 `gS 1.00` 과 달리 학습 분포에서는 흔들린다.

## 3. 다음 보상이 해야 할 것

1. ★**각도 처방을 전부 없앤다.** 아래 항의 **각도 목표·대역·사다리**를 제거한다:
   `tilt_aim`(W 2.0) · `tilt_near`(6.0) · `pour_tilt`(10) · `pour_pose`(5.0) · `aimed_tilt`(2.5) · `overtilt`(6.0) ·
   `early_tilt`(0.8, 각도 상한 벌점) · 잔량 연동 `release_tilt`/`theta_pre`/`theta_max` 사다리 전체.
   **붓는 각은 보상이 정하지 않는다. 학습이 찾는다.**
2. ★**수입은 결과로만 준다.** 비드가 받는 컵에 들어가는 흐름(`pour_delta`)과 상태(`transferred`·`transfer_tail`·`src_emptied`),
   그리고 흘림 벌점(`spill_delta` 240)·과다 유출(`flow_excess`)이 전부다. 각도가 아니라 **비드가 어디 있는가**가 값을 만든다.
3. **기하 조건은 허가(게이트)로만 남긴다.** 주둥이가 받는 컵 입구 위(cup_radius 배수 `perm` 창)일 때만 방출 관련 수입이 켜지고,
   창 밖에서 비드를 쏟으면 흘림 벌점이 그대로 물린다. 방향(소스 컵이 리시버 쪽으로 기울었는지)도 **부호 게이트로만** 쓰고 각도 크기에는 값을 주지 않는다.
   `unaimed_release` 는 "창 밖 방출"에 대한 벌점으로 유지하되 **각도 창(theta_pre 기준)이 아니라 실제 방출(비드가 소스를 떠남)** 을 기준으로 건다.
4. **파지 유지에 값을 친다.** 기울이는 동안 파지가 흔들리는 것이 관찰됐다 — `close_src`/파지 항이 **운반·붓기 구간에서도** 살아 있게 하고,
   처음 잡아 올린 접촉 분포를 유지하는 쪽에 수입을 준다(형제 트랙 i24 의 교훈: 절대 힘 기준 금지, 기울이는 내내 유지).
5. **그대로 둘 것:** 접근/허가 분리(`approach_lip`·`perm`, cup_radius 배수)·만남 통로(`meet_cross`·`src_cross`)·
   속도 벌점(src_speed·src_spin·rot_pour·rcv_spin)·제어 벌점(palm_sat·palm_bang·act_flip·carry_rate)·관절 한계/떨림 벌점(W 0.4)·
   리시버 규율(cup_disturb_rcv·upright_rcv)·성공 기준(env 0.80/0.25/20).
6. **탐색:** entropy 3.5 까지 붕괴했다. 각도 수입을 없애면 초반에 자세가 흐트러질 수 있다 — 새 수입(배달)이 즉시 잡히도록
   `pour_delta` 의 한 알당 값(현재 7.5)은 유지하고, 상태 항(`transferred` 5.0)은 각도 수입이 빠진 만큼 조금 올려도 된다.

## 4. 웜스타트

- **i16g ep2100**(결정론: 기울기 87.6°·in_target 0.150·흘림 0.050·파지 1.00/1.00). 이미 기울이는 동작을 갖고 있으므로
  각도 수입을 없애도 부팅 단계에서 붓기 자체를 잃지 않는다. 습관 없는 계보이므로 만남 지점(frac_rcv 0.57~0.63)도 유지된다.
