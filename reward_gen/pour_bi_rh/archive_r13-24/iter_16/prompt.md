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
    bottom and 1.5/(hi-lo) at the top, so the last beads are worth three times the first ones."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _limit_pen(margin, free):
    """★ iter_15 (1): how hard an arm is pressed against its joint limits, in [0, 1].
    `margin` is (N, A) with 1 = middle of the range and 0 = touching a limit. Measured on the i13
    ep550 trace: the SOURCE arm has some joint inside 5 % of a limit on 46.4 % of steps and a median
    worst-joint margin of 0.104, the RECEIVER arm never enters the 5 % zone. So the live band is
    0 .. `free` (= 0.30) and anything above it is free - the approach phase keeps its full freedom.
    The deficit is SQUARED: sitting mid-band is nearly free, being pinned is expensive.
    Worst joint 0.7 + mean of all joints 0.3, so every joint keeps its own slope instead of only the
    single binding one (a pure min term is flat whenever two joints are tied)."""
    d = torch.clamp((free - torch.clamp(margin, 0.0, 1.0)) / free, 0.0, 1.0)      # (N, A)
    worst = d.max(dim=-1).values                                                  # (N,)
    return 0.7 * worst * worst + 0.3 * (d * d).mean(dim=-1)


def _shake_pen(qdd, free_mean, scale_mean, free_peak, scale_peak):
    """★ iter_15 (2): arm tremor, charged on ACCELERATION and never on velocity, so the deliberate
    pouring rotation is not taxed - the measured 5-6 Hz source-wrist oscillation (velocity-sign
    reversals 5.2 / 6.2 / 3.8 Hz on joints 5/6/7) shows up almost entirely in qdd.
    Normalised on the measured operating point: per-joint |qdd| median 2.5 (free) and p90 12.8
    rad/s^2, i.e. tanh hits 0.76 exactly at the p90 step. The second, smaller term looks at the WORST
    joint with the same p90 as its free level, because a mean over 7 joints dilutes a wrist spike."""
    q = qdd.abs()
    mean_part = torch.tanh(torch.clamp(q.mean(dim=-1) - free_mean, min=0.0) / scale_mean)
    peak_part = torch.tanh(torch.clamp(q.max(dim=-1).values - free_peak, min=0.0) / scale_peak)
    return 0.65 * mean_part + 0.35 * peak_part


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    iter_15: VERBATIM since iter_04 - the grasp works (src_grasped 0.843, rcv_grasped 0.872)."""
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
    iter_06 VERBATIM - on the keep list (grasp_src 1.45, grasp_rcv 1.18 measured over the iter_14 run,
    both lift bands held). iter_15 changes NOTHING in this function."""
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
    iter_15: VERBATIM iter_13/14 (the aim discipline held all run: diag_mis_deg 0.82-1.9 deg mean and
    the bead path is clean - 96 % of the beads that leave the source go out through the mouth).

      d_hat    : horizontal unit vector src cup origin -> receiver mouth. Independent of the cup's
                 attitude, so it is defined for an upright cup and never flips while rotating.
      theta    : SIGNED tilt toward the receiver = atan2(up . d_hat, up_z). A purely sideways tilt
                 gives 0, tilting away gives a negative angle.
      lip      : the rim point the beads leave from = mouth centre + r*(cos(theta)*d_hat
                 - sin(theta)*z). Drops as the cup turns toward the receiver, RISES when it turns away.
      lip_drop : how far that lip sits BELOW where it would be with the same cup origin and an upright
                 cup: r*sin(theta) + mouth_z*(1 - cos(theta)). 0.067 m at 75 deg, 0.082 at 90.
                 Measured budget diag_lift_budget 0.309 mean all run, i.e. the term is live."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]          # (N,2)
    sep = v.norm(dim=-1)                                               # (N,)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / sep_min, max=1.0)   # the azimuth is meaningless when the cups are coaxial

    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])                     # signed, negative = away
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    # what is tilted but NOT toward the receiver: 0 for a perfect pour attitude (diag_mis_deg 1.9 mean)
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
    # as in iter_06..14 so carry_src and align_mouths keep their measured scale (align 3.02)
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
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt(SIGNED) < pour
    # < full transfer < finish. NEW in iter_15: an explicit ARM-POSTURE cost on top of that ladder.
    #
    # iter_14 RESULT. The task side is healthy and still climbing at the end of the run:
    # episode_success 0 -> 0.463, in_target -> 0.601, total -> 24.9, and both iter_14 additions are
    # live and rising (transfer_tail 0 -> 0.97, src_emptied 0 -> 0.924). Nothing on the pour ladder is
    # dead or dominating, so the whole pour side is KEPT VERBATIM.
    #
    # What blocks the real-robot transfer is the arm posture, now measurable for the first time
    # (i13 ep550 deterministic trace, 64 env x 900 steps):
    #   - the SOURCE wrist is pinned: some joint inside 5 % of a limit on 46.4 % of steps, median
    #     worst-joint margin 0.104 (joints 5 and 7 alone are inside on 41-42 % of steps). The RECEIVER
    #     arm never enters the 5 % zone, so the same term costs it nothing.
    #   - the SOURCE wrist oscillates at 5-6 Hz: |qdd| median 2.5, p90 12.8 rad/s^2.
    #   - the RECEIVER palm command is the big mover left: |da| 0.429 (src 0.149), sign alternation
    #     19.9 %/step (src 3.4 %).
    # Until now the reward could only see arm_qd and pressed on joint limits indirectly through palm
    # action saturation. The env now exposes *_arm_limit_margin and *_arm_qdd (both readable on the
    # real robot, so the s2r contract is intact) and iter_15 charges them DIRECTLY.
    #
    # iter_15 changes EXACTLY three things and keeps everything else verbatim:
    #   1. arm_limit_src / arm_limit_rcv (new): squared deficit of the joint-limit margin below 0.30,
    #      worst joint 0.7 + mean 0.3, gated on the grasp so the approach keeps its freedom.
    #   2. arm_shake_src / arm_shake_rcv (new): joint ACCELERATION, normalised on the measured median
    #      2.5 and p90 12.8 rad/s^2. Acceleration, not velocity, so the pouring rotation is not taxed.
    #   3. the receiver palm command smoothness is nudged (W_BANG_RCV 1.5 -> 2.0, receiver carry-rate
    #      weight 0.03 -> 0.05, source unchanged). No new mechanism.
    # KEEP LIST (user note 4), all verbatim iter_14: approach/permission split (approach_lip, perm in
    # cup_radius multiples), aim discipline, src_speed / src_spin / rot_pour / rcv_spin, flow_excess,
    # overtilt, spill 240, transfer_tail / src_emptied, settle_finish, per-arm palm_sat / palm_bang,
    # grasp / lift / converge / carry / align. No new ABSOLUTE distance constant is introduced.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.82 measured)
    W_AWAY_RCV = 0.8
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (best-ever grasp rates)
    # --- source lift: band + over-lift ramp. KEEP LIST (diag_lift_budget 0.309 measured) -------------
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
    # --- receiver lift band - KEEP LIST (rcv lift 0.106 m, lift_rcv 1.42) ---------------------------
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
    W_SUCCESS = 20.0        # unchanged; measured success_bonus 5.96/step at success_now 0.476
    SPILL_ZERO = 0.25       # the env's spill limit (unchanged): clean_f 0.52 measured
    CLEAN_POUR_FLOOR = 0.10
    W_TRANSFER = 5.0        # unchanged (transferred 1.20 measured, still rising)
    TAIL_LO, TAIL_HI = 0.35, 0.90
    W_TAIL = 4.0            # unchanged (transfer_tail 0 -> 0.97 over the run: live and climbing)
    EMPTY_LO = 0.45
    W_EMPTY = 3.0           # unchanged (src_emptied 0 -> 0.924: the strongest new income of iter_14)
    W_SETTLE = 3.0          # unchanged and on the KEEP LIST although it measures ~0.0002/step: the
                            # source ends at 91 deg so upright_src ~ 0. Left as is this round on
                            # purpose (one hypothesis per round); diag_finish_q / diag_settle_still
                            # stay so the next round can decide whether the source must come back up.
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
    # ★ iter_15 (3): receiver palm command smoothness nudged only. Measured src |da| 0.149 vs rcv
    # 0.429 and sign alternation 3.4 %/step vs 19.9 %/step, and these two terms are already pulling it
    # down, so the source keeps 0.03 and only the receiver goes 0.03 -> 0.05 (carry_rate -0.13 ->
    # about -0.20/step).
    W_RATE_PALM_SRC = 0.03
    W_RATE_PALM_RCV = 0.05
    BANG_DEAD = 0.1
    W_BANG_SRC = 1.5
    W_BANG_RCV = 2.0        # 1.5 -> 2.0: -0.30 -> about -0.40/step at the measured diag_bang_rcv 0.22
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
    W_PALM_SAT_SRC = 3.0    # KEEP LIST: the indirect limit pressure stays; the new term is direct
    W_PALM_SAT_RCV = 1.5
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35      # rad (20 deg) - the env's own receiver-upright limit, reused for the
                            # source in the finish pose, so no new attitude constant appears
    # ★ iter_15 (1) JOINT LIMIT MARGIN. MARGIN_FREE 0.30 = the top of the band where the trace says a
    # gradient is needed (median worst-joint margin 0.104, 46.4 % of steps inside 5 % of a limit).
    # W 1.2 -> about -0.40/step for the source at today's operating point (0.7*((0.30-0.104)/0.30)^2
    # = 0.30 from the worst joint alone) and up to -1.2/step with a joint pinned; the receiver arm
    # never enters the zone, so it loses no income. Gated on the grasp flag: the approach must keep
    # its full posture freedom.
    MARGIN_FREE = 0.30
    W_ARM_LIMIT = 1.2
    # ★ iter_15 (2) ARM TREMOR. Charged on ACCELERATION only, so the pouring rotation (a velocity) is
    # untouched and the measured 5-6 Hz wrist oscillation is hit where it is largest. QDD_FREE 2.5 =
    # the measured per-joint |qdd| median (today's normal motion is free), QDD_SCALE 10.3 = 12.8 - 2.5
    # so tanh reaches 0.76 exactly at the measured p90; the worst-joint term uses the same 12.8 as its
    # free level because a mean over 7 joints dilutes a single shaking wrist.
    QDD_FREE, QDD_SCALE = 2.5, 10.3
    QDD_PEAK_FREE, QDD_PEAK_SCALE = 12.8, 12.8
    W_ARM_SHAKE = 1.0       # about -0.2/step on the source now. Together with arm_limit the two new
                            # terms are about -0.8/step at the measured operating point: inside the
                            # -1/step budget, against ~-2/step of existing control penalties and a
                            # 20/step success bonus, so the demonstrated pour stays clearly profitable.

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

    # ---- spout gates: how close the pouring lip is to the receiver mouth (VERBATIM iter_13/14) -----
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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # measured rcv_progress 0.815

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

    # ---- receiver lift band (unchanged - measured 0.106 m, inside the 0.10 m band) ------------------
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
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.87 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * not_nested * conv_shape

    # carry / align keep their iter_06 shape and weights on the DIRECTED spout (align 3.02 measured)
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

    # control gate WITHOUT adr_progress: both cups grasped and >= 5.5 cm up (measured 0.804)
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

    # ---- aiming at the moment of release (VERBATIM iter_13/14) -------------------------------------
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

    # ================================================== ★ iter_15: ARM POSTURE AND TREMOR ===========
    # The cup-side motion limits above say nothing about HOW the arm produces that motion. The trace
    # says the source arm produces it by pressing its wrist into the joint limits (46.4 % of steps
    # inside 5 %) and by oscillating at 5-6 Hz. Both are now charged directly, and both are gated on
    # the grasp flag so the free approach posture is untouched.
    margin_pen_src = _limit_pen(ctx.src_arm_limit_margin, MARGIN_FREE)
    margin_pen_rcv = _limit_pen(ctx.rcv_arm_limit_margin, MARGIN_FREE)
    arm_limit_src = g_src * margin_pen_src
    arm_limit_rcv = g_rcv * margin_pen_rcv
    shake_src = g_src * _shake_pen(ctx.src_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)
    shake_rcv = g_rcv * _shake_pen(ctx.rcv_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)

    # ================================================== iter_14: FULL TRANSFER AND FINISH (kept) ====
    # Both ramps are 0 at the iter_13 stochastic operating point and 1 at the iter_13 deterministic
    # replay, and both get steeper toward the end. Measured over the iter_14 run: transfer_tail rose
    # 0 -> 0.97 and src_emptied 0 -> 0.924, i.e. they are doing exactly what they were added for.
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    emptied = torch.clamp(1.0 - fill, 0.0, 1.0)
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)      # 0.19 at in_target 0.60 (today), 1 at 0.90
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)        # 0.77 at the measured emptied 0.79, 1 at empty
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
        # ★ iter_15 (1): keep the wrist off its stops. -0.4/step on the source today, ~0 on the clean
        # receiver arm, up to -1.2/step with a joint pinned. Grasp-gated: free during the approach.
        "arm_limit_src": -W_ARM_LIMIT * arm_limit_src,
        "arm_limit_rcv": -W_ARM_LIMIT * arm_limit_rcv,
        # ★ iter_15 (2): kill the 5-6 Hz wrist oscillation. Acceleration only, so tilting is free.
        "arm_shake_src": -W_ARM_SHAKE * shake_src,
        "arm_shake_rcv": -W_ARM_SHAKE * shake_rcv,
        "act_flip": -W_FLIP_PALM * (g_src * flip_palm_src + g_rcv * flip_palm_rcv),
        "hand_flip": -W_FLIP_HAND * ctrl_gate * flip_hand,
        # ★ iter_15 (3): per-arm palm rate weights, receiver 0.03 -> 0.05 (|da| 0.429 vs src 0.149)
        "carry_rate": -(W_RATE_PALM_SRC * g_src * rate_palm_src
                        + W_RATE_PALM_RCV * g_rcv * rate_palm_rcv
                        + W_RATE_HAND_CARRY * ctrl_gate * rate_hand),
        "palm_bang_src": -W_BANG_SRC * g_src * bang_src,
        "palm_bang_rcv": -W_BANG_RCV * g_rcv * bang_rcv,   # ★ iter_15 (3): 1.5 -> 2.0
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
    comps["diag_fill_tail"] = fill_tail          # how far past the old plateau the transfer got
    comps["diag_empty_q"] = empty_q              # how empty the source ends up
    comps["diag_finish_q"] = finish_q            # the combined finish state (mean ramp)
    comps["diag_settle_still"] = settle_still    # were both cups actually stopped at the end
    comps["diag_emptied"] = emptied              # raw 1 - in_source, for comparison across rounds
    # ★ iter_15 diagnostics: raw joint state, so the next round can read the posture directly
    comps["diag_src_margin_min"] = torch.clamp(ctx.src_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_rcv_margin_min"] = torch.clamp(ctx.rcv_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_src_limit_hit"] = (comps["diag_src_margin_min"] < 0.05).to(dtype)   # trace: 0.464
    comps["diag_rcv_limit_hit"] = (comps["diag_rcv_margin_min"] < 0.05).to(dtype)   # trace: 0.000
    comps["diag_src_qdd"] = ctx.src_arm_qdd.abs().mean(dim=-1)                      # trace: median 2.5
    comps["diag_rcv_qdd"] = ctx.rcv_arm_qdd.abs().mean(dim=-1)                      # trace: median 2.4
    comps["diag_src_qdd_max"] = ctx.src_arm_qdd.abs().max(dim=-1).values            # trace: p90 12.8
    comps["diag_rcv_qdd_max"] = ctx.rcv_arm_qdd.abs().max(dim=-1).values
    comps["diag_rate_palm_rcv"] = rate_palm_rcv
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0.0333, 0.0333, 0.0667, 0.0667, 0.1, 0.1, 0.1, 0.1]  min 0 · mean 0.0663 · max 0.1
bead/in_target: [0, 0.665, 0.524, 0.464, 0.429, 0.434, 0.287, 0.298, 0.255, 0.235]  min 0 · mean 0.371 · max 0.666
bead/spill: [0, 0.178, 0.146, 0.148, 0.166, 0.163, 0.146, 0.142, 0.117, 0.0964]  min 0 · mean 0.134 · max 0.228
done/drop: [0, 0, 0, 0.000488, 0.000244, 0.000488, 0.000488, 0.000977, 0.000488, 0]  min 0 · mean 0.000429 · max 0.00171
done/mimic_runaway: [0, 0, 0.000488, 0.000244, 0.000732, 0, 0.000244, 0.00122, 0.000977, 0.000244]  min 0 · mean 0.000415 · max 0.00171
episode_lengths/step: [120, 749, 658, 610, 644, 653, 550, 569, 537, 615]  min 120 · mean 595 · max 749
reward/act_flip: [0, -0.179, -0.165, -0.161, -0.158, -0.151, -0.134, -0.134, -0.135, -0.142]  min -0.216 · mean -0.149 · max 0
reward/action_rate: [-0.0133, -0.0154, -0.0155, -0.0156, -0.0156, -0.0153, -0.015, -0.015, -0.015, -0.0151]  min -0.0159 · mean -0.0152 · max -0.0133
reward/aim_gate: [0, 0.694, 0.649, 0.576, 0.554, 0.581, 0.542, 0.579, 0.6, 0.648]  min 0 · mean 0.595 · max 0.756
reward/aimed_tilt: [0, 0.0886, 0.178, 0.193, 0.186, 0.195, 0.287, 0.31, 0.352, 0.439]  min 0 · mean 0.276 · max 1.24
reward/align_mouths: [0, 3.18, 2.71, 2.42, 2.31, 2.3, 2.02, 2.13, 2.08, 2.28]  min 0 · mean 2.32 · max 3.33
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/approach_lip: [0, 0.00782, 0.0163, 0.0201, 0.00983, 0.0119, 0.00517, 0.0109, 0.00246, -0.00108]  min -0.0217 · mean 0.0104 · max 0.246
reward/arm_limit_rcv: [0, -0.004, -0.00848, -0.00931, -0.00818, -0.00899, -0.00927, -0.00914, -0.00928, -0.00569]  min -0.0628 · mean -0.00889 · max 0
reward/arm_limit_src: [0, -0.543, -0.459, -0.468, -0.496, -0.533, -0.473, -0.488, -0.47, -0.501]  min -0.596 · mean -0.475 · max 0
reward/arm_shake_rcv: [0, -0.373, -0.295, -0.262, -0.242, -0.248, -0.231, -0.249, -0.233, -0.212]  min -0.397 · mean -0.248 · max 0
reward/arm_shake_src: [0, -0.374, -0.338, -0.3, -0.307, -0.299, -0.265, -0.272, -0.258, -0.231]  min -0.42 · mean -0.287 · max 0
reward/away_rcv: [-0.0682, -0.166, -0.165, -0.166, -0.169, -0.172, -0.169, -0.166, -0.168, -0.169]  min -0.175 · mean -0.167 · max -0.0682
reward/carry_rate: [0, -0.195, -0.179, -0.172, -0.169, -0.165, -0.152, -0.152, -0.152, -0.159]  min -0.212 · mean -0.164 · max 0
reward/carry_src: [0, 1.61, 1.43, 1.31, 1.24, 1.26, 1.12, 1.15, 1.09, 1.18]  min 0 · mean 1.24 · max 1.7
reward/close_rcv: [0.0793, 0.0763, 0.0809, 0.08, 0.0785, 0.0765, 0.0778, 0.0789, 0.0749, 0.0766]  min 0.073 · mean 0.0785 · max 0.0998
reward/close_src: [0.0279, 0.0185, 0.022, 0.0205, 0.0214, 0.0219, 0.0271, 0.0261, 0.0259, 0.024]  min 0.0152 · mean 0.0228 · max 0.0279
reward/contact_push: [-0.00301, -0.00272, -0.00655, -0.00852, -0.00998, -0.00854, -0.0112, -0.0106, -0.0109, -0.0101]  min -0.0291 · mean -0.00906 · max -0.000972
reward/converge: [0, 2.18, 1.9, 1.73, 1.66, 1.67, 1.49, 1.53, 1.48, 1.56]  min 0 · mean 1.64 · max 2.33
reward/cup_collision: [0, -0.00285, -0.00122, -0.00215, -0.00125, -0.00129, -0.00221, -0.0054, -0.00348, -0.00264]  min -0.00687 · mean -0.00224 · max 0
reward/cup_disturb_rcv: [-0.0271, -0.12, -0.112, -0.104, -0.105, -0.105, -0.106, -0.108, -0.114, -0.113]  min -0.207 · mean -0.108 · max -0.0271
reward/cup_disturb_src: [-0.213, -0.015, -0.0316, -0.0375, -0.0409, -0.0381, -0.047, -0.0461, -0.0512, -0.0437]  min -0.213 · mean -0.0419 · max -0.0103
reward/cup_fling: [-0.000477, -0.0135, -0.0161, -0.0151, -0.0153, -0.0149, -0.0197, -0.0204, -0.021, -0.0167]  min -0.0908 · mean -0.0177 · max -0.000477
reward/diag_bang_rcv: [0.164, 0.239, 0.232, 0.237, 0.239, 0.234, 0.232, 0.232, 0.238, 0.252]  min 0.164 · mean 0.237 · max 0.287
reward/diag_bang_src: [0.0575, 0.09, 0.0785, 0.0875, 0.0876, 0.0829, 0.0661, 0.0657, 0.0574, 0.0452]  min 0.0429 · mean 0.072 · max 0.102
reward/diag_clean: [1, 0.528, 0.586, 0.59, 0.562, 0.55, 0.624, 0.63, 0.689, 0.735]  min 0.393 · mean 0.639 · max 1
reward/diag_close_speed: [0.0727, -0.00654, 0.00775, 0.00803, 0.00753, 0.0144, -0.000794, 0.00756, 0.00502, 0.00606]  min -0.0162 · mean 0.00881 · max 0.196
reward/diag_ctrl_gate: [0, 0.847, 0.744, 0.7, 0.678, 0.679, 0.617, 0.641, 0.617, 0.637]  min 0 · mean 0.666 · max 0.885
reward/diag_emptied: [0, 0.851, 0.689, 0.634, 0.616, 0.617, 0.456, 0.462, 0.394, 0.348]  min 0 · mean 0.525 · max 0.904
reward/diag_empty_q: [0, 0.836, 0.655, 0.598, 0.573, 0.569, 0.371, 0.374, 0.301, 0.242]  min 0 · mean 0.46 · max 0.888
reward/diag_fill_tail: [0, 0.611, 0.468, 0.403, 0.352, 0.355, 0.21, 0.221, 0.189, 0.168]  min 0 · mean 0.304 · max 0.611
reward/diag_finish_q: [0, 0.724, 0.561, 0.501, 0.463, 0.462, 0.291, 0.297, 0.245, 0.205]  min 0 · mean 0.382 · max 0.74
reward/diag_flip_palm: [0.234, 0.626, 0.61, 0.606, 0.604, 0.577, 0.522, 0.519, 0.523, 0.545]  min 0.234 · mean 0.567 · max 0.773
reward/diag_flow_beads: [0, 0.00928, 0.0181, 0.0188, 0.0178, 0.0139, 0.012, 0.0154, 0.0115, 0.00952]  min 0 · mean 0.016 · max 0.15
reward/diag_lead_gate: [0, 0.861, 0.77, 0.729, 0.72, 0.728, 0.664, 0.692, 0.688, 0.723]  min 0 · mean 0.713 · max 0.874
reward/diag_lift_budget: [0.244, 0.327, 0.316, 0.311, 0.307, 0.305, 0.295, 0.296, 0.293, 0.294]  min 0.243 · mean 0.303 · max 0.329
reward/diag_lip_xy: [0.277, 0.274, 0.233, 0.298, 0.384, 0.506, 0.635, 0.733, 0.277, 0.368]  min 0.0785 · mean 0.45 · max 0.954
reward/diag_mis_deg: [1.34, 1.1, 1.25, 1.32, 1.15, 1.17, 2.17, 2.3, 2.47, 1.99]  min 0.687 · mean 1.73 · max 13.7
reward/diag_palm_sat: [0.306, 0.437, 0.434, 0.407, 0.392, 0.398, 0.401, 0.404, 0.413, 0.413]  min 0.208 · mean 0.41 · max 0.49
reward/diag_palm_sat_rcv: [0.356, 0.348, 0.352, 0.348, 0.329, 0.311, 0.285, 0.282, 0.274, 0.265]  min 0.255 · mean 0.313 · max 0.404
reward/diag_perm: [0, 0.678, 0.642, 0.573, 0.549, 0.577, 0.538, 0.572, 0.589, 0.637]  min 0 · mean 0.586 · max 0.751
reward/diag_perm_at_tilt: [0, 0.667, 0.615, 0.542, 0.496, 0.519, 0.439, 0.481, 0.485, 0.526]  min 0 · mean 0.523 · max 0.714
reward/diag_pour_phase: [0, 0.804, 0.699, 0.652, 0.634, 0.632, 0.556, 0.582, 0.564, 0.593]  min 0 · mean 0.618 · max 0.826
reward/diag_pre_release: [0, 0.876, 0.76, 0.709, 0.669, 0.664, 0.556, 0.589, 0.561, 0.59]  min 0 · mean 0.647 · max 0.898
reward/diag_rate_palm_rcv: [1.55, 2.98, 2.98, 2.94, 2.94, 2.86, 2.71, 2.7, 2.75, 2.89]  min 1.55 · mean 2.86 · max 3.71
reward/diag_rcv_limit_hit: [0, 0, 0, 0, 0, 0.000244, 0.000977, 0.000244, 0.000244, 0]  min 0 · mean 8.4e-05 · max 0.000977
reward/diag_rcv_margin_min: [0.254, 0.408, 0.366, 0.358, 0.361, 0.362, 0.358, 0.363, 0.362, 0.372]  min 0.196 · mean 0.359 · max 0.412
reward/diag_rcv_qdd: [4.68, 8.69, 7.74, 7.3, 13.3, 6.71, 6.59, 7.44, 6.74, 6.2]  min 4.68 · mean 7.87 · max 262
reward/diag_rcv_qdd_max: [10.9, 22.8, 20, 18.2, 47.7, 16.7, 16.2, 18.9, 16.6, 15.1]  min 10.9 · mean 22.3 · max 1.36e+03
reward/diag_settle_still: [0.944, 0.582, 0.606, 0.624, 0.635, 0.638, 0.642, 0.631, 0.638, 0.661]  min 0.228 · mean 0.63 · max 0.944
reward/diag_src_limit_hit: [0, 0.332, 0.253, 0.316, 0.401, 0.488, 0.415, 0.441, 0.427, 0.473]  min 0 · mean 0.373 · max 0.501
reward/diag_src_margin_min: [0.164, 0.0772, 0.0955, 0.0957, 0.0897, 0.0817, 0.0904, 0.0829, 0.0861, 0.0815]  min 0.0659 · mean 0.0904 · max 0.164
reward/diag_src_qdd: [6.75, 8.95, 9.21, 8.65, 9.27, 8.75, 9.19, 9.65, 9.41, 8.06]  min 6.75 · mean 9.62 · max 71
reward/diag_src_qdd_max: [16.2, 24.8, 26, 23.2, 25.4, 24.1, 25.4, 27.2, 26.1, 21.9]  min 16.2 · mean 26.2 · max 130
reward/diag_src_v: [0.0594, 0.291, 0.376, 0.316, 0.359, 0.358, 0.546, 0.433, 0.441, 0.367]  min 0.0594 · mean 0.456 · max 7.47
reward/diag_src_w: [1.47, 2.14, 2.44, 2.08, 2.54, 2.36, 2.28, 2.48, 2.89, 2.23]  min 1.47 · mean 4.31 · max 337
reward/diag_theta_deg: [7.77, 96.1, 83.6, 78.5, 74.5, 72.4, 61.5, 63.4, 60.2, 61.1]  min 5.4 · mean 70.4 · max 98
reward/diag_unperm_at_tilt: [0, 0.209, 0.144, 0.167, 0.173, 0.145, 0.117, 0.108, 0.0762, 0.0635]  min 0 · mean 0.124 · max 0.266
reward/early_tilt: [-0.0682, -0.0952, -0.104, -0.12, -0.12, -0.111, -0.125, -0.125, -0.131, -0.128]  min -0.184 · mean -0.119 · max -0.0606
reward/flow_excess: [0, -0.00146, -0.00342, -0.00439, -0.00439, -0.00244, -0.00146, -0.00146, -0.000977, -0.000977]  min -0.0225 · mean -0.00198 · max 0
reward/grasp_rcv: [0.00109, 1.24, 1.17, 1.1, 1.11, 1.14, 1.08, 1.1, 1.05, 1.09]  min 0.00109 · mean 1.1 · max 1.38
reward/grasp_src: [0.00431, 1.53, 1.37, 1.29, 1.26, 1.26, 1.13, 1.16, 1.13, 1.15]  min 0.00431 · mean 1.23 · max 1.59
reward/hand_flip: [0, -0.0744, -0.0655, -0.0639, -0.0622, -0.0629, -0.0595, -0.0591, -0.0586, -0.0599]  min -0.0792 · mean -0.0609 · max 0
reward/hand_foreign: [-0.000478, -0.00157, -0.0027, -0.00335, -0.00269, -0.00217, -0.00378, -0.00345, -0.00362, -0.00303]  min -0.00815 · mean -0.00323 · max -0.000478
reward/hold_rcv: [0, 0.26, 0.237, 0.219, 0.216, 0.224, 0.206, 0.205, 0.202, 0.229]  min 0 · mean 0.217 · max 0.276
reward/lift_rcv: [0, 1.46, 1.31, 1.19, 1.15, 1.18, 1.12, 1.16, 1.14, 1.23]  min 0 · mean 1.19 · max 1.54
reward/lift_src: [0, 2.86, 2.55, 2.37, 2.31, 2.23, 2.04, 2.16, 2.15, 2.28]  min 0 · mean 2.29 · max 2.94
reward/mis_tilt: [0, -0.00504, -0.00707, -0.00639, -0.00439, -0.00515, -0.0136, -0.0129, -0.0159, -0.012]  min -0.157 · mean -0.00967 · max 0
reward/near_nested: [0, -0.00517, -0.00357, -0.00395, -0.00465, -0.00235, -0.00791, -0.0109, -0.0162, -0.0151]  min -0.0228 · mean -0.00773 · max 0
reward/nested: [0, -0.00439, -0.00293, -0.00439, -0.00439, -0.00195, -0.00635, -0.0103, -0.0171, -0.0122]  min -0.02 · mean -0.00708 · max 0
reward/over_lift_src: [0, -0.0312, -0.0422, -0.055, -0.054, -0.0922, -0.0952, -0.07, -0.0475, -0.0193]  min -0.113 · mean -0.049 · max 0
reward/overtilt: [0, -0.0231, -0.0325, -0.0372, -0.0298, -0.0237, -0.0156, -0.0165, -0.00888, -0.0039]  min -0.0857 · mean -0.0211 · max 0
reward/palm_bang_rcv: [0, -0.45, -0.407, -0.401, -0.402, -0.394, -0.384, -0.389, -0.397, -0.427]  min -0.526 · mean -0.402 · max 0
reward/palm_bang_src: [0, -0.122, -0.0975, -0.106, -0.105, -0.0986, -0.0715, -0.0681, -0.0593, -0.0443]  min -0.139 · mean -0.0823 · max 0
reward/palm_sat_rcv: [0, -0.478, -0.438, -0.412, -0.382, -0.358, -0.306, -0.313, -0.295, -0.285]  min -0.53 · mean -0.359 · max 0
reward/palm_sat_src: [0, -1.17, -1.08, -0.959, -0.904, -0.916, -0.879, -0.908, -0.908, -0.918]  min -1.2 · mean -0.946 · max 0
reward/pour_delta: [0, 0.0604, 0.119, 0.123, 0.101, 0.0934, 0.0842, 0.0989, 0.0806, 0.0641]  min -0.00732 · mean 0.107 · max 1.04
reward/pour_dir_pen: [0, -0.1, -0.091, -0.0439, -0.0246, -0.0384, -0.0323, -0.0379, -0.0519, -0.0566]  min -0.21 · mean -0.0504 · max 0
reward/pour_pose: [0, 0.671, 0.593, 0.538, 0.413, 0.363, 0.341, 0.369, 0.36, 0.343]  min 0 · mean 0.466 · max 1.86
reward/pour_tilt: [0, 0.885, 0.81, 0.741, 0.565, 0.488, 0.448, 0.481, 0.454, 0.411]  min 0 · mean 0.616 · max 2.59
reward/rcv_home: [0, -0.0585, -0.0735, -0.0714, -0.0791, -0.0741, -0.107, -0.114, -0.123, -0.12]  min -0.126 · mean -0.0907 · max 0
reward/rcv_progress: [0.29, 0.834, 0.787, 0.753, 0.745, 0.747, 0.733, 0.75, 0.738, 0.747]  min 0.29 · mean 0.751 · max 0.851
reward/rcv_spin: [0, -0.199, -0.16, -0.147, -0.133, -0.13, -0.112, -0.121, -0.114, -0.107]  min -0.211 · mean -0.131 · max 0
reward/rcv_still: [0, -0.47, -0.406, -0.371, -0.354, -0.352, -0.321, -0.338, -0.329, -0.326]  min -0.498 · mean -0.355 · max 0
reward/reach_rcv: [0.919, 0.804, 0.812, 0.804, 0.797, 0.794, 0.802, 0.809, 0.805, 0.806]  min 0.791 · mean 0.806 · max 0.919
reward/reach_src: [0.519, 0.498, 0.535, 0.536, 0.538, 0.541, 0.574, 0.572, 0.569, 0.568]  min 0.498 · mean 0.548 · max 0.585
reward/rot_pour: [0, -0.315, -0.271, -0.244, -0.236, -0.241, -0.197, -0.206, -0.19, -0.176]  min -0.33 · mean -0.224 · max 0
reward/rush_near_cup: [-0.232, -0.0156, -0.0274, -0.0345, -0.035, -0.0353, -0.0392, -0.0397, -0.0415, -0.0394]  min -0.232 · mean -0.037 · max -0.00979
reward/settle_finish: [0, 0.000204, 0.000111, 0.000114, 1.29e-05, 0.000478, 0.000377, 0.000279, 7.05e-05, 0.000187]  min 0 · mean 0.000205 · max 0.000657
reward/spill_delta: [0, -0.0615, -0.108, -0.105, -0.129, -0.123, -0.155, -0.158, -0.111, -0.123]  min -0.668 · mean -0.132 · max 0
reward/src_emptied: [0, 1.13, 0.811, 0.662, 0.534, 0.516, 0.329, 0.363, 0.337, 0.326]  min 0 · mean 0.525 · max 1.13
reward/src_speed: [0, -0.125, -0.117, -0.108, -0.104, -0.102, -0.0932, -0.0996, -0.0926, -0.0837]  min -0.199 · mean -0.102 · max 0
reward/src_spin: [0, -0.121, -0.109, -0.1, -0.0961, -0.0981, -0.0893, -0.09, -0.0823, -0.0722]  min -0.13 · mean -0.0931 · max 0
reward/success_bonus: [0, 7.38, 5.15, 4.1, 3.26, 3.11, 1.87, 2.13, 1.9, 1.86]  min 0 · mean 3.19 · max 7.38
reward/tilt_aim: [0, 1.15, 1.01, 1, 0.98, 0.964, 0.902, 0.933, 0.924, 1]  min 0 · mean 0.971 · max 1.4
reward/tilt_near: [0, 1.91, 1.77, 1.59, 1.38, 1.36, 1.36, 1.5, 1.57, 1.74]  min 0 · mean 1.62 · max 3.78
reward/total: [1.92, 26.7, 22, 19.4, 17.3, 16.9, 14.2, 15.2, 14.8, 15.8]  min 1.92 · mean 17.7 · max 26.7
reward/transfer_tail: [0, 1.2, 0.859, 0.693, 0.565, 0.548, 0.344, 0.381, 0.348, 0.364]  min 0 · mean 0.563 · max 1.2
reward/transferred: [0, 1.48, 1.09, 0.894, 0.746, 0.726, 0.507, 0.556, 0.526, 0.583]  min 0 · mean 0.765 · max 1.48
reward/unaimed_release: [0, -0.0313, -0.035, -0.0412, -0.0637, -0.0692, -0.118, -0.114, -0.0877, -0.0848]  min -0.27 · mean -0.0793 · max 0
rewards/step: [96.3, 1.69e+04, 1.44e+04, 1.22e+04, 1.11e+04, 1.14e+04, 7.3e+03, 8.47e+03, 7.71e+03, 1.04e+04]  min 96.3 · mean 1.04e+04 · max 1.92e+04
task/aim_dist: [0.304, 0.421, 0.409, 0.415, 0.555, 0.683, 0.966, 0.986, 0.493, 0.537]  min 0.187 · mean 0.63 · max 1.24
task/cup_collision_rate: [0, 0.00391, 0.00171, 0.00293, 0.00195, 0.00146, 0.00244, 0.00659, 0.00439, 0.00317]  min 0 · mean 0.00312 · max 0.0127
task/cups_center_dist: [0.312, 0.492, 0.471, 0.473, 0.61, 0.736, 1.01, 1.03, 0.536, 0.581]  min 0.217 · mean 0.682 · max 1.29
task/episode_success: [0, 0.536, 0.386, 0.322, 0.268, 0.263, 0.146, 0.161, 0.141, 0.125]  min 0 · mean 0.237 · max 0.536
task/nested_rate: [0, 0.0022, 0.00146, 0.0022, 0.0022, 0.000977, 0.00317, 0.00513, 0.00854, 0.0061]  min 0 · mean 0.00353 · max 0.01
task/rcv_cup_lift: [8.23e-05, 0.109, 0.0956, 0.0935, 0.0927, 0.0922, 0.0882, 0.0927, 0.164, 0.0835]  min 8.23e-05 · mean 0.0974 · max 0.209
task/rcv_grasped: [0, 0.915, 0.827, 0.781, 0.77, 0.774, 0.749, 0.771, 0.754, 0.766]  min 0 · mean 0.775 · max 0.943
task/rcv_hand_foreign_rate: [0, 0.000732, 0.0022, 0.00269, 0.00244, 0.000977, 0.00391, 0.00342, 0.00342, 0.00342]  min 0 · mean 0.00282 · max 0.00708
task/src_cup_lift: [0.00215, 0.435, 0.415, 0.343, 0.428, 0.493, 0.714, 0.573, 0.362, 0.388]  min 0.00215 · mean 0.441 · max 0.749
task/src_grasped: [0, 0.875, 0.793, 0.753, 0.735, 0.735, 0.682, 0.695, 0.68, 0.698]  min 0 · mean 0.725 · max 0.909
task/src_hand_foreign_rate: [0.000977, 0.00562, 0.00732, 0.00708, 0.00635, 0.0061, 0.0083, 0.00879, 0.00879, 0.00684]  min 0.000977 · mean 0.00789 · max 0.0247
task/src_tilt_deg: [9.11, 97.2, 84.8, 79.8, 75.7, 73.5, 63.7, 65.7, 62.7, 63]  min 9.11 · mean 72.2 · max 99
task/success_now: [0, 0.545, 0.396, 0.336, 0.282, 0.274, 0.156, 0.169, 0.146, 0.132]  min 0 · mean 0.248 · max 0.545

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 16 operator notes — 만남 지점을 오른팔 영역으로 옮긴다 (손목 한계의 진짜 원인)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 목표와 원칙 (사용자)

- 목표: **흘림 없는 이송** + 실기 이식 가능한 **안정 제어**.
- 사용자 관찰(09.23): "왼팔이 제자리에만 있고 오른팔 쪽으로 안 넘어가서 오른팔에 여유가 없어진 걸 수도 있다." → **측정으로 확인됐다(아래 2·3).**
- ★원칙: 속도·제어 벌점을 풀어 SIM 효율을 되찾지 않는다. 조준 창은 `ctx.cup_radius` 배수.

## 1. iter_15 결과 (t2r_rh3_i15, 4096 env, e~270 종료)

| 구간 | in_target | in_source | 흘림/유출 | success | 기울기 | limit_hit | qdd |
|---|---|---|---|---|---|---|---|
| e0~40 | 0.474 | 0.365 | 0.224 | 0.347 | 82.0° | 0.211 | 9.16 |
| e40~80 | 0.476 | 0.361 | 0.225 | 0.336 | 81.8° | 0.244 | 9.64 |
| e120~160 | 0.413 | 0.411 | 0.262 | 0.258 | 72.5° | 0.461 | 9.43 |
| e200~219 | **0.260** | **0.597** | 0.300 | 0.141 | 62.6° | 0.426 | 10.18 |

- ★**관절 한계 벌점은 실패했다.** 벌점이 −0.40 → −0.50 으로 커지는 동안 `diag_src_limit_hit` 은 0.21 → 0.43 으로 **올랐고**,
  대신 붓기가 줄었다(기울기 82 → 63°, in_source 0.36 → 0.60). **1차 차분 상관 corr(limit_hit, in_target) = +0.90** —
  지금 구조에서는 손목을 한계까지 써야 더 많이 옮긴다.
- 떨림 벌점(가속도)도 `diag_src_qdd` 9.1~10.2 로 효과가 없었다.
- iter_14 의 전량 이송 항은 유효했다(초반 in_target 0.47~0.53, i13 마지막 0.386 대비 개선).
- ep250 결정론 재생(16 env): in_target 0.800·in_source 0.000·흘림 **0.200**·성공 0.812. i13 ep550(0.906/0.000/0.094)보다 흘림이 나쁘다.

## 2. ★기구학 프로브 — 손목 한계는 "습관"이 아니라 **만남 지점의 결과**다

URDF FK(체인 `r_aj_1..7` → `r_hl_palm_sensor`)를 만들고 trace 와 대조해 **위치 오차 0.0 mm** 로 정합을 확인했다.
관절 한계는 trace meta 의 soft limit 을 썼다. 여유 = 1 − |q − 중앙| / 반범위.

| 검사 | 조건 | 결과 |
|---|---|---|
| 1차 | 정책이 만든 **손바닥 자세 고정**, 다른 관절 조합 탐색 | 여유 ≥ 0.30 해 **0 %** · 최선 중앙 0.103(실제 0.096) |
| 2차 | 붓기 요구만(주둥이 위치 ±2 cm·기울기 75~110°·방위 ±25°), 자세 자유 | 여유 ≥ 0.30 **35 %** |
| 3차 | 주둥이 위치 격자, **방위 자유** | 전 지점 0.36~0.67 (지금 지점에서도 0.43) |
| 4차 | 주둥이 위치 × **실제 붓는 방향(+y, 받는 컵이 왼쪽)** | 아래 표 |

**4차(핵심):**

| 주둥이가 가는 지점 | 최대 손목 여유 |
|---|---|
| **지금 지점 (0.37, +0.05, 0.55)** | **해 없음** |
| 중앙 (0.37, −0.05, 0.45) | **0.500** |
| 오른팔 쪽 (0.37, −0.15, 0.45) | **0.628** |

실제 운영점의 최소 여유는 0.089~0.103 이다. **만남 지점을 y −0.05 ~ −0.15·높이 0.45 m 로 옮기면 여유가 5~7 배 생긴다.**

## 3. ★두 팔의 분담 실측 (i13 ep550 trace, 붓는 스텝 = 기울기 > 70°)

| | 시작 | 붓는 순간 | 이동 |
|---|---|---|---|
| 소스 컵(오른팔) | (0.382, **−0.159**, 0.268) | (0.374, **+0.046**, 0.550) | **0.352 m** |
| 리시버 컵(왼팔) | (0.378, +0.165, 0.268) | (0.373, +0.108, 0.384) | 0.132 m |
| 리시버 손바닥 | (0.311, +0.300, 0.421) | (0.365, +0.152, 0.391) | 0.162 m |

**오른팔이 몸 중앙을 넘어 왼쪽(y +0.046)·높이 0.55 m 까지 간다. 왼팔은 13 cm 만 나온다.** 붓는 순간 두 컵 수평거리는 6.4 cm 다.

원인 후보(코드): `rcv_home`(리시버를 시작 위치 근처에 묶는 항)·`rcv_still`(정지 유지)·`away_rcv`·`cup_disturb_rcv` 가
리시버 팔의 이동을 억제한다. 이 항들은 iter_04 에서 "리시버가 컵을 흔들어 넘어뜨린다"는 문제로 들어왔다.

## 4. 다음 보상이 해야 할 것

1. ★**만남 지점을 소스 팔 영역으로 옮긴다.** 두 컵이 만나는 지점(또는 리시버 컵 위치)이 **소스 컵 시작 위치 쪽(y 음수)** 에 오도록 수입을 준다.
   절대 좌표를 새로 박지 말고 `ctx.src_cup_spawn_pos`·`ctx.rcv_cup_spawn_pos` 기준 상대량으로 쓴다(스폰 랜덤에 견딘다).
   목표 대역: 소스 컵 시작 y 에서 리시버 컵 시작 y 쪽으로 **1/3 지점까지** 리시버가 나오는 정도(실측 기준 y ≈ −0.05 ~ −0.10).
2. ★**리시버를 묶는 항을 파지 후에 푼다.** `rcv_home`·`rcv_still` 은 리시버가 컵을 잡기 전까지만 켜고,
   잡은 뒤에는 **소스 쪽으로 이동하는 것**에 수입을 준다. 컵 전도·흔들기 방지는 `cup_disturb_rcv`·기울기 벌점으로 남긴다.
3. **관절 한계 벌점은 남기되 약하게**(W 1.2 → 0.4 수준). 길이 열린 뒤에는 방향타 역할만 한다. 떨림(가속도) 벌점도 같은 이유로 약하게.
4. **그대로 둘 것:** 접근/허가 분리(approach_lip·perm)·조준 규율·속도 벌점·flow_excess·overtilt·spill 240·
   전량 이송(transfer_tail·src_emptied)·팔별 palm_sat/palm_bang.
5. 새 항 크기는 스텝당 합계 −1 을 넘지 않게 하고, 유도 수입은 success_bonus(20)보다 작게 둔다.

## 5. 기동 (사용자 결정 09.23)

**A/B 두 런을 각 2048 env 로 동시에 돌린다.**
- **fresh**: 체크포인트 없이 처음부터. 만남 지점을 바꾸는 것은 i09 부터 굳은 습관을 깨는 일이라 웜스타트로는 계곡을 못 넘을 수 있다.
- **warm**: i13 ep550(결정론 in_target 0.906·in_source 0.000) 에서 시작.
- 판정: 손목 여유(diag_src_margin_min)·limit_hit·리시버 이동량·in_target·흘림/유출을 같은 epoch 에서 비교한다.
