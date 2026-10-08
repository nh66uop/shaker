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


# =====================================================================================================
# iter_22 = iter_21 design + ONLY the three round-22 operator changes (09.25 evening, user approved),
# repaired after review (fixes 1-5, marked "repair n"):
#   (1) place_rcv : receiver MOUTH target at an ABSOLUTE height (rim of a cup standing on the table, raised
#                   by RCV_POUR_LIFT) - no longer dragged up by the source. Horizontal part judged against the
#                   source SPOUT with a cup_radius window; pour point near the y = 0 centreline; the spout
#                   height gets a ceiling (aim_lip potential and the permission window).
#                   repair 5: until the source BOTTOM is above the receiver RIM the horizontal pull stops at a
#                   body-clearance radius (cup_radius multiples), so it never drags the receiver into the source.
#   (2) tilt_open : NEW small difference-form signal. Potential PHI = win * theta+ * max(cos psi, 0) (repair 2:
#                   only the part of the lean that points at the receiver); rate = exact derivative of that same
#                   potential (repair 2). Payments capped at +1/step, charges ungated and capped only far away
#                   (repair 1) -> closed cycles sum to <= 0. Only before the first bead has LEFT the source
#                   (repair 3). Weight 0.1 -> about one bead over a whole episode (repair 4). No angle target/band.
#   (3) tilt_dir  : strong only OUTSIDE the window (1.5), weak inside (0.3). spill_delta: x0.25 inside the
#                   aimed window for the first bead(s) to leave the source only, at most one bead (repair 3).
# Unchanged on purpose: grasp, lift, pour_sink, all outcome weights, every speed / control / safety penalty.
# Summed terms: iter_21's 25 + tilt_open = 26.
# =====================================================================================================


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _smoothstep(x):
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_slope(x):
    """d smoothstep / dx (exactly 0 outside [0, 1])."""
    t = torch.clamp(x, 0.0, 1.0)
    return 6.0 * t * (1.0 - t)


def _late_ramp(x, lo, hi):
    """0.5*t + 0.5*t^2: never flat above lo, last part worth 3x the first."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _progress(rate_closing, v_scale):
    """Difference-form payment from an exact potential rate, positive = getting closer.
    Symmetric clamp (iter_21): +-1 per step, reached at the stable-speed limit v_scale, so going faster than
    allowed buys nothing and exploration jitter nets zero (i20: an asymmetric clamp made jitter a net loss).
    Used by place_rcv / aim_lip only; tilt_open has its own asymmetric bound (repair 1)."""
    return torch.clamp(rate_closing / v_scale, -1.0, 1.0)


def _limit_pen(margin, free):
    d = torch.clamp((free - torch.clamp(margin, 0.0, 1.0)) / free, 0.0, 1.0)
    worst = d.max(dim=-1).values
    return 0.7 * worst * worst + 0.3 * (d * d).mean(dim=-1)


def _shake_pen(qdd, free_mean, scale_mean, free_peak, scale_peak):
    q = qdd.abs()
    mean_part = torch.tanh(torch.clamp(q.mean(dim=-1) - free_mean, min=0.0) / scale_mean)
    peak_part = torch.tanh(torch.clamp(q.max(dim=-1).values - free_peak, min=0.0) / scale_peak)
    return 0.65 * mean_part + 0.35 * peak_part


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the CUP frame. VERBATIM since iter_04 (grasp 0.75/0.82 measured)."""
    ARRIVE_RISE = 0.045
    FUNNEL_R0 = 0.01
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z
    thumb_hi = 0.5 * cup_mouth_z

    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)
    rad_vec = rel - h_tip.unsqueeze(-1) * up
    rad = rad_vec.norm(dim=-1)
    gap = torch.clamp(rad - tip_r, min=0.0)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values

    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))

    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
              cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok):
    """Per-arm stage-A value in [0, 1] plus the table-disturbance penalties (iter_21 VERBATIM).
    ONE monotone ladder, grasped (>= 0.4) always above not-grasped (<= 0.39)."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach = 0.4 * reach_far + 0.6 * 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])

    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = near_grip * torch.clamp(hand_closure / 0.7, max=1.0)
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    contact_q = (0.4 + 0.6 * pinch) * (0.1 + 0.9 * geo["thumb_band"])
    balance = torch.minimum(f_t, f_o) / (torch.maximum(f_t, f_o) + 1e-6)
    keep = near_grip * (0.3 + 0.7 * balance)
    held_val = 0.4 + 0.3 * contact_q + 0.3 * keep                       # in [0.4, 1.0]
    free_val = 0.25 * reach + 0.10 * close + 0.10 * pre_grasp           # at most 0.39
    value = torch.where(g, held_val, free_val) * pose_ok

    h_pos = torch.clamp(cup_pos[:, 2] - cup_spawn_pos[:, 2] - 0.005, min=0.0)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    v_lin = cup_lin_vel.norm(dim=-1)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"value": value, "h": h_pos, "v_lin": v_lin, "slide": slide, "one_sided": one_sided,
            "palm_push": palm_push, "rush": rush, "fling": fling, "keep": keep, "balance": balance}


def _pour_geometry(ctx, r):
    """Directed-spout geometry, iter_08/09 VERBATIM.
      d_hat : horizontal unit vector source cup origin -> receiver mouth
      theta : SIGNED tilt toward the receiver (locates the lip; never a target)
      lip   : rim point the beads leave from; lip_xy / lip_dz relative to the receiver mouth."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    sep = v.norm(dim=-1)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / 0.03, max=1.0)          # azimuth undefined when coaxial (3 cm floor)
    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    cos_t = torch.cos(torch.clamp(theta_eff, 0.0, math.pi))
    lip_drop = r * sin_t + ctx.cup_mouth_z * (1.0 - cos_t)
    lip_xy_vec = ctx.src_cup_mouth_pos[:, :2] \
        + (r * torch.cos(theta)).unsqueeze(-1) * d_hat - ctx.rcv_cup_mouth_pos[:, :2]
    lip_xy = lip_xy_vec.norm(dim=-1)
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r * torch.sin(theta)
    lip_dz = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_pt = torch.cat([ctx.rcv_cup_mouth_pos[:, :2] + lip_xy_vec, lip_z.unsqueeze(-1)], dim=-1)
    return {"d_hat": d_hat, "sep_ok": sep_ok, "theta": theta, "lip_drop": lip_drop,
            "lip_xy_vec": lip_xy_vec, "lip_xy": lip_xy, "lip_dz": lip_dz, "lip_pt": lip_pt}


def _point_vel(lin_vel, ang_vel, origin, point):
    return lin_vel + torch.cross(ang_vel, point - origin, dim=-1)


def _tilt_direction_error(ctx, d_hat):
    """iter_19 VERBATIM: psi = angle between the cup's lean (horizontal part of its up axis) and d_hat.
    dir_err = (1 - cos psi)/2 is 0 when leaning at the receiver at ANY tilt - no magnitude is valued."""
    u_xy = ctx.src_cup_up[:, :2]
    u_h = u_xy.norm(dim=-1)
    cos_psi = torch.clamp((u_xy * d_hat).sum(dim=-1) / torch.clamp(u_h, min=1e-6), -1.0, 1.0)
    return 0.5 * (1.0 - cos_psi), torch.acos(cos_psi)


def _spout_window(lip_xy, lip_dz, dlip_xy, dlip_dz, r_in, r_out, z_lo, z_hi, z_fade):
    """iter_22: spout window in [0, 1] and its exact time derivative (same state function both ways).
    horizontal: 1 inside r_in, smoothstep to 0 at r_out (iter_13 window, unchanged);
    height    : 0 -> 1 linearly over [0, z_lo] (iter_17 floor, unchanged), 1 up to the NEW ceiling z_hi,
                smoothstep to 0 at z_hi + z_fade."""
    t_xy = (lip_xy - r_in) / (r_out - r_in)
    w_xy = 1.0 - _smoothstep(t_xy)
    dw_xy = -_smoothstep_slope(t_xy) * dlip_xy / (r_out - r_in)
    w_lo = torch.clamp(lip_dz / z_lo, 0.0, 1.0)
    dw_lo = ((lip_dz > 0.0) & (lip_dz < z_lo)).to(lip_dz.dtype) * dlip_dz / z_lo
    t_hi = (lip_dz - z_hi) / z_fade
    w_hi = 1.0 - _smoothstep(t_hi)
    dw_hi = -_smoothstep_slope(t_hi) * dlip_dz / z_fade
    w_z = w_lo * w_hi
    dw_z = dw_lo * w_hi + w_lo * dw_hi
    return {"w_xy": w_xy, "w_z": w_z, "win": w_xy * w_z, "dwin": dw_xy * w_z + w_xy * dw_z}


def _tilt_open_potential(ctx, geo, r, v_mouth, du, r_in, r_out, z_lo, z_hi, z_fade):
    """iter_22 repair 2: potential PHI = win * theta+ * max(cos psi, 0) and its EXACT time derivative.
    Every rate is the derivative of the very formula that gives the value: theta and psi from the cup up axis
    u and d_hat (d_hat turns when either cup moves), the lip point of _pour_geometry and the window of
    _spout_window. theta+ alone reaches pi/2 for a cup lying on its side at ANY azimuth within 90 deg of d_hat;
    the cos psi factor keeps only the part of the lean that points at the receiver. No angle target, no band.
    du = d(u)/dt = w x u (assumption (가설): ang_vel in the world frame, as _point_vel already assumes)."""
    up = ctx.src_cup_up
    d_hat = geo["d_hat"]
    theta = geo["theta"]
    dtype = theta.dtype
    # d_hat = v / |v|, v = receiver mouth - source origin (horizontal)
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    sep = torch.clamp(v.norm(dim=-1), min=1e-4)
    dv = v_mouth[:, :2] - ctx.src_cup_lin_vel[:, :2]
    dd_hat = (dv - d_hat * (d_hat * dv).sum(dim=-1, keepdim=True)) / sep.unsqueeze(-1)
    # theta = atan2(a, b), a = u_xy . d_hat, b = u_z
    a = (up[:, :2] * d_hat).sum(dim=-1)
    da = (du[:, :2] * d_hat).sum(dim=-1) + (up[:, :2] * dd_hat).sum(dim=-1)
    b = up[:, 2]
    dtheta = (b * da - a * du[:, 2]) / torch.clamp(a * a + b * b, min=1e-3)
    # cos psi = a / |u_xy| (same formula as _tilt_direction_error)
    h = torch.clamp(up[:, :2].norm(dim=-1), min=1e-6)
    dh = (up[:, :2] * du[:, :2]).sum(dim=-1) / h
    cos_psi = torch.clamp(a / h, -1.0, 1.0)
    dcos = (da - cos_psi * dh) / h
    # lip: xy = src mouth + r cos(theta) d_hat - rcv mouth, z = src mouth z - r sin(theta) (see _pour_geometry)
    v_m = _point_vel(ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_pos, ctx.src_cup_mouth_pos)
    dlip_vec = v_m[:, :2] - (r * torch.sin(theta) * dtheta).unsqueeze(-1) * d_hat \
        + (r * torch.cos(theta)).unsqueeze(-1) * dd_hat - v_mouth[:, :2]
    dlip_xy = (geo["lip_xy_vec"] * dlip_vec).sum(dim=-1) / torch.clamp(geo["lip_xy"], min=1e-4)
    dlip_dz = v_m[:, 2] - r * torch.cos(theta) * dtheta - v_mouth[:, 2]
    wnd = _spout_window(geo["lip_xy"], geo["lip_dz"], dlip_xy, dlip_dz, r_in, r_out, z_lo, z_hi, z_fade)
    # PHI and dPHI/dt (product rule over the three factors)
    th_p = torch.clamp(theta, min=0.0)
    c_p = torch.clamp(cos_psi, min=0.0)
    dth_p = (theta > 0.0).to(dtype) * dtheta
    dc_p = (cos_psi > 0.0).to(dtype) * dcos
    win, dwin = wnd["win"], wnd["dwin"]
    phi = win * th_p * c_p
    phi_rate = dwin * th_p * c_p + win * dth_p * c_p + win * th_p * dc_p
    return {"phi": phi, "phi_rate": phi_rate, "dtheta": dtheta, "wnd": wnd}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ------------------------------------------------------------------ constants and their sources
    W_ALIVE = 1.0            # constant survival floor, identical in every state (not A-D income)
    # A grasp (iter_21 verbatim)
    W_GRASP_SRC, W_GRASP_RCV = 1.5, 1.5
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35       # rad, the ENV's own receiver-upright success limit (20 deg)
    # B lift (iter_21 verbatim; brief 5-1 B keeps the raised source allowance)
    W_LIFT_SRC = 1.5
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.30
    LIFT_FREE_SRC0 = 0.40
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    W_LIFT_RCV = 0.5
    LIFT_TARGET_RCV = 0.08
    CARRY_H = 0.03
    # C placement - iter_22 change 1: WORLD-FIXED height for the receiver mouth, horizontal vs the SPOUT
    RCV_POUR_LIFT = 0.15     # m, height allowance (the only absolute constant brief 5-3 permits). Mouth target
                             # = rim of a cup standing on the table + 0.15 m. i21 rcv lift reached 0.182 m max.
                             # With the spout window 0.7 r..4 r above that mouth, the source origin sits at
                             # lift ~0.23-0.34 m while pouring (70-90 deg), inside the lift_src paid range
                             # (LIFT_TARGET_SRC 0.30), so placement and lift_src do not fight.
    Z_TOL_R = 1.0            # +-1 r free band around the mouth height target
    PLACE_WIN_R = 1.0        # horizontal: receiver mouth pulled until the spout is within 1 r (= PERM_IN_R)
    BODY_CLEAR_R = 1.0       # repair 5: until the source BOTTOM is BODY_CLEAR_R*r above the receiver RIM the pull
                             # stops at lip_xy = (PLACE_WIN_R + BODY_CLEAR_R) r = 2 r. Upright source: its lip is
                             # 1 r off its axis, so the two axes stay 3 r apart (about 2.4 cm between the walls
                             # of the ~6 cm cups). The free radius shrinks linearly (slope 1) over that 1 r.
    PLACE_AZ = math.radians(68.0)       # K2 pourable side, kept from iter_21 as a GUARD only
    PLACE_AZ_TOL = math.radians(45.0)   # K1 palm-yaw box +-45 deg: 23..113 deg free, outside costs distance
    CENTER_FREE_R = 2.0      # |y| of the receiver mouth (= pour point) free up to 2 r (y ~= 0 centreline)
    W_PLACE = 1.0
    # D aim
    W_AIM = 1.5
    V_PROG = 0.15            # m/s: progress terms saturate at the stable cup speed limit (SRC_V_FREE)
    LIP_Z_HI_R = 0.7         # spout floor above the rim, 0.7 r ~= 2 cm (iter_17 permission height)
    LIP_Z_MAX_R = 4.0        # iter_22 change 1: spout CEILING 4 r ~= 11 cm above the rim (i21 hovered at
                             # +13.8 cm). Window height 3.3 r > lip drop at a quarter turn (r + mouth_z ~= 2.9 r
                             # with K5/K6), so a spout at the ceiling upright is still above the floor when level.
    LIP_Z_FADE_R = 2.0       # window fades out between 4 r and 6 r
    SINK_KEEP_R = 1.0        # pour_sink geometry (iter_19)
    # E direction - iter_22 change 3
    W_TILT_DIR_OUT = 1.5     # outside the window: iter_19/21 strength (1.03/step at the i18 failure state)
    W_TILT_DIR_IN = 0.3      # inside the window: a fifth, the spout is already over the mouth
    # tilt_open - iter_22 change 2 (repaired)
    W_TILT_OPEN = 0.1        # repair 4. Per step at most +0.1. Whole-episode income is bounded by
                             # W_TILT_OPEN * PHI_max / (TILT_RATE_SCALE * dt): payments never exceed the exact
                             # rate and first_out only falls. PHI_max = theta when the first bead leaves, ~70 deg
                             # = 1.22 rad with a full cup (prompt knowledge 5). dt is not given; at dt = 1/60 s
                             # (가설) the bound is 0.1 * 1.22 * 60 ~= 7.3 ~= one bead (+7.5), 9.4 if the cup
                             # reaches 90 deg before any bead has left.
    OPEN_BEADS = 1.0         # repair 3: active until the first bead has LEFT the source (not until one landed)
    OPEN_CHARGE_CAP = 100.0  # repair 1: charges are not gated and capped only at 100x the payment cap
                             # (-10 reward/step) against physics glitches (i21 diag_src_w max 2.02e3 rad/s).
                             # A real fall of PHI is charged in full while |rate| <= 100 /s.
    # H height keep while pouring (iter_19 verbatim)
    W_POUR_SINK = 1.0
    V_Z_SCALE = 0.04
    # F outcome - unchanged weights (per-bead units with 20 beads)
    W_POUR = 150.0           # +7.5 per bead delivered
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0            # -2 per extra bead dumped in the same step
    W_SPILL = 240.0          # -12 per bead spilled
    SPILL_RELIEF = 0.25      # iter_22 change 3 (repair 3): x0.25 (-3/bead) inside the AIMED window, only for the
                             # first bead(s) to leave the source and at most one bead's worth
    SETTLE_FRAC = 0.03
    W_UNAIMED = 60.0         # -3 per bead released outside the permission window
    W_TRANSFER = 8.0         # level term: 0.9 in, 0.05 spill -> 5.76/step (> A+B, brief 5-2 (3))
    W_TAIL = 4.0
    TAIL_LO, TAIL_HI = 0.10, 0.90
    EMPTY_LO = 0.20
    W_SUCCESS = 20.0
    SPILL_ZERO = 0.25        # env spill limit
    PERM_IN_R, PERM_OUT_R = 1.0, 3.0      # spout window in cup radii (iter_13, kept)
    LEAD_FULL_R, LEAD_ZERO_R = 1.75, 5.0  # pour-phase window for the stricter spin limit (kept)
    # G control - KEEP LIST values (unchanged, user principle: no speed / control relaxation)
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, W_ROT_POUR = 1.0, 1.0
    RCV_W_FREE, RCV_W_SCALE, W_RCV_SPIN = 0.5, 1.0, 0.5
    STILL_FREE, STILL_SCALE, W_STILL_POUR = 0.05, 0.15, 0.5
    W_TILT_PEN_RCV = 0.3
    W_FLIP_PALM, W_FLIP_HAND = 0.3, 0.2
    W_RATE_PALM_SRC, W_RATE_PALM_RCV, W_RATE_HAND = 0.03, 0.05, 0.01
    W_ACT_RATE = 0.05
    BANG_DEAD, W_BANG_SRC, W_BANG_RCV = 0.1, 1.5, 2.0
    SAT_LO, W_PALM_SAT_SRC, W_PALM_SAT_RCV = 0.7, 3.0, 1.5
    MARGIN_FREE, W_ARM_LIMIT = 0.30, 0.4
    QDD_FREE, QDD_SCALE = 2.5, 10.3
    QDD_PEAK_FREE, QDD_PEAK_SCALE = 12.8, 12.8
    W_ARM_SHAKE = 0.4
    TILT_RATE_SCALE = ROT_POUR_FREE   # rad/s: tilt_open payment saturates at the pour-phase spin limit
    # safety / disturbance (unchanged)
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0

    dtype = ctx.src_cup_tilt.dtype
    r = ctx.cup_radius
    geo = _pour_geometry(ctx, r)
    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    nested = ctx.cups_nested.to(dtype)
    not_nested = 1.0 - nested

    # ------------------------------------------------------------------ A. grasp (iter_21 verbatim)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    h_src_raw = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    carry_src = g_src * torch.clamp(h_src_raw / CARRY_H, max=1.0)
    src_table_upright = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = carry_src + (1.0 - carry_src) * src_table_upright
    src = _hand_cup(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                    ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel,
                    ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z, TIP_R, pose_ok_src)
    rcv = _hand_cup(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                    ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel,
                    ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z, TIP_R, upright_rcv)
    h_src, h_rcv = src["h"], rcv["h"]
    carry_rcv = g_rcv * torch.clamp(h_rcv / CARRY_H, max=1.0)
    carry_both = carry_src * carry_rcv * not_nested
    up_w_src = g_src * torch.clamp(h_src / 0.05, max=1.0)
    up_w_rcv = g_rcv * torch.clamp(h_rcv / 0.05, max=1.0)
    ctrl_gate = up_w_src * up_w_rcv

    # ------------------------------------------------------------------ B. lift (iter_21 verbatim)
    lift_free_src = LIFT_FREE_SRC0 + geo["lip_drop"]
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    over_lift = torch.clamp((h_src - lift_free_src) / OVER_LIFT_RANGE, 0.0, 1.0)
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_band - W_OVER_LIFT * over_lift
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * (0.5 + 0.5 * upright_rcv)

    # ------------------------------------------------------------------ bead bookkeeping (gates)
    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    has_beads = torch.clamp(fill / 0.10, max=1.0)        # >= 2 beads left
    stage_cd = carry_both * has_beads                     # placement / aim live while there is a pour ahead
    n_beads = float(max(int(ctx.num_beads), 1))
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    released = 1.0 - fill                                 # beads no longer in the source (receiver, air, spill)
    first_out = torch.clamp(OPEN_BEADS - n_beads * released, 0.0, 1.0)   # repair 3: 1 until the first bead LEFT

    # velocities of the two points that matter, and the spout rate relative to the mouth (place / aim, as before)
    v_mouth = _point_vel(ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_pos, ctx.rcv_cup_mouth_pos)
    v_lip = _point_vel(ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_pos, geo["lip_pt"])
    dv = v_lip - v_mouth
    dlip_xy = (geo["lip_xy_vec"] * dv[:, :2]).sum(dim=-1) / torch.clamp(geo["lip_xy"], min=1e-4)
    du_src = torch.cross(ctx.src_cup_ang_vel, ctx.src_cup_up, dim=-1)   # d(up)/dt (가설: world-frame ang_vel)

    # ------------------------------------------------------------------ spout window (a GATE, never a state income)
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    z_lo, z_hi = LIP_Z_HI_R * r, LIP_Z_MAX_R * r
    opn = _tilt_open_potential(ctx, geo, r, v_mouth, du_src, r_in, r_out, z_lo, z_hi, LIP_Z_FADE_R * r)
    wnd = opn["wnd"]
    win = wnd["win"]                                      # position window (horizontal x height)
    perm_xy = wnd["w_xy"]
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = win * toward                                   # aimed window: position + leaning toward
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r) / ((LEAD_ZERO_R - LEAD_FULL_R) * r))

    # ------------------------------------------------------------------ C. placement (iter_22 change 1, repair 5)
    # Receiver mouth: world-fixed height (not tied to the source), horizontally until the SPOUT is within
    # PLACE_WIN_R*r, |y| near the centreline, and not outside the pourable side of the source (guard).
    # Repair 5: the horizontal pull reaches PLACE_WIN_R*r only once the source BOTTOM centre is BODY_CLEAR_R*r
    # above the receiver RIM centre; below that it stops at (PLACE_WIN_R + BODY_CLEAR_R)*r, so it never drags
    # the receiver into the source body. Potential = norm of the excesses; paid as its exact rate -> 0 when parked.
    z_mouth_tgt = ctx.table_z - ctx.cup_bottom_z + ctx.cup_mouth_z + RCV_POUR_LIFT
    dz_m = ctx.rcv_cup_mouth_pos[:, 2] - z_mouth_tgt
    ez = torch.sign(dz_m) * torch.clamp(dz_m.abs() - Z_TOL_R * r, min=0.0)
    z_bot = ctx.src_cup_pos[:, 2] + ctx.cup_bottom_z * ctx.src_cup_up[:, 2]      # source bottom centre
    clear = z_bot - ctx.rcv_cup_mouth_pos[:, 2]                                   # height above the receiver rim
    dclear = ctx.src_cup_lin_vel[:, 2] + ctx.cup_bottom_z * du_src[:, 2] - v_mouth[:, 2]
    ramp = BODY_CLEAR_R * r
    r_free = PLACE_WIN_R * r + ramp - torch.clamp(clear, 0.0, ramp)
    dr_free = -((clear > 0.0) & (clear < ramp)).to(dtype) * dclear
    eh = torch.clamp(geo["lip_xy"] - r_free, min=0.0)
    y_m = ctx.rcv_cup_mouth_pos[:, 1]
    ey = torch.clamp(y_m.abs() - CENTER_FREE_R * r, min=0.0)
    rel = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    drel = v_mouth[:, :2] - ctx.src_cup_lin_vel[:, :2]
    a_lo, a_hi = PLACE_AZ - PLACE_AZ_TOL, PLACE_AZ + PLACE_AZ_TOL
    n_lo_x, n_lo_y = math.sin(a_lo), -math.cos(a_lo)      # outward normal of the 23 deg edge
    n_hi_x, n_hi_y = -math.sin(a_hi), math.cos(a_hi)      # outward normal of the 113 deg edge
    es_lo = torch.clamp(rel[:, 0] * n_lo_x + rel[:, 1] * n_lo_y, min=0.0)
    es_hi = torch.clamp(rel[:, 0] * n_hi_x + rel[:, 1] * n_hi_y, min=0.0)
    des_lo = drel[:, 0] * n_lo_x + drel[:, 1] * n_lo_y
    des_hi = drel[:, 0] * n_hi_x + drel[:, 1] * n_hi_y
    d_place = torch.sqrt(eh ** 2 + ez ** 2 + ey ** 2 + es_lo ** 2 + es_hi ** 2)
    dd_place = (eh * (dlip_xy - dr_free) + ez * v_mouth[:, 2] + ey * torch.sign(y_m) * v_mouth[:, 1]
                + es_lo * des_lo + es_hi * des_hi) / torch.clamp(d_place, min=1e-4)
    place = stage_cd * _progress(-dd_place, V_PROG)

    # ------------------------------------------------------------------ D. aim (3D, difference form)
    # Potential: horizontal spout->mouth distance + how far the spout is BELOW its floor + (iter_22) how far
    # it is ABOVE its ceiling. Tilting toward the receiver lowers the spout, so a spout hovering too high is
    # brought into the window either by descending or by tilting - no angle is named.
    below = torch.clamp(z_lo - geo["lip_dz"], min=0.0)
    above = torch.clamp(geo["lip_dz"] - z_hi, min=0.0)
    d_aim = torch.sqrt(geo["lip_xy"] ** 2 + below ** 2 + above ** 2)
    dd_aim = ((geo["lip_xy_vec"] * dv[:, :2]).sum(dim=-1) + (above - below) * dv[:, 2]) \
        / torch.clamp(d_aim, min=1e-4)
    aim = stage_cd * _progress(-dd_aim, V_PROG)

    # ------------------------------------------------------------------ tilt_open (iter_22 change 2, repaired)
    # PHI = win * theta+ * max(cos psi, 0); phi_rate is its exact derivative (repair 2). Per step:
    #   first_out * ( carry_both * sep_ok * clamp(rate/scale, 0, +1) + clamp(rate/scale, -OPEN_CHARGE_CAP, 0) )
    # Payments <= rate (gates <= 1, cap +1) and charges = rate (no grasp / sep gate), so while first_out is
    # constant every closed cycle (in / out of the window, tilt / untilt, grasp flicker) sums to <= 0, up to
    # time discretisation (repair 1). first_out only falls, so the episode total is bounded (see W_TILT_OPEN).
    rate_n = opn["phi_rate"] / TILT_RATE_SCALE
    open_pay = carry_both * geo["sep_ok"] * torch.clamp(rate_n, 0.0, 1.0)
    open_charge = torch.clamp(rate_n, -OPEN_CHARGE_CAP, 0.0)
    tilt_open = first_out * (open_pay + open_charge)

    # ------------------------------------------------------------------ E. direction (cost only, iter_22 change 3)
    dir_err, psi = _tilt_direction_error(ctx, geo["d_hat"])
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)          # magnitude only as a multiplier, no target
    dir_w = W_TILT_DIR_IN * win + W_TILT_DIR_OUT * (1.0 - win)   # strong only outside the window
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err * dir_w

    # ------------------------------------------------------------------ H. keep height while pouring (verbatim)
    org_vz = ctx.src_cup_lin_vel[:, 2] - v_mouth[:, 2]
    h_org = ctx.src_cup_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    org_low = (h_org < (LIP_Z_HI_R + SINK_KEEP_R) * r).to(dtype)
    pour_sink = ctrl_gate * not_nested * org_low * torch.clamp(org_vz / V_Z_SCALE, -1.0, 1.0)

    # ------------------------------------------------------------------ F. outcome (the beads)
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    # iter_22 change 3 (repair 3): a spilled bead has necessarily LEFT the source, so first_out is already 0 when
    # its spill is counted. The relief gate uses the same count WITHOUT the beads counted as spilled in this step
    # ("no other bead left before them") and relieves at most one bead's worth: x0.25 (-3 instead of -12, still
    # a loss vs +7.5 delivered). Every later spill, and spill outside the aimed window, keeps the full -12/bead.
    spill_now = torch.clamp(ctx.d_spill, min=0.0)
    first_spill = torch.clamp(OPEN_BEADS - n_beads * (released - spill_now), 0.0, 1.0)
    relieved = perm * first_spill * torch.clamp(spill_now, max=OPEN_BEADS / n_beads)
    spill_step = settled * (spill_now - (1.0 - SPILL_RELIEF) * relieved)
    spill_lvl = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    clean_f = torch.clamp(1.0 - spill_lvl / SPILL_ZERO, 0.0, 1.0)
    emptied = 1.0 - fill
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)
    fill_ok = torch.clamp(in_t / TAIL_LO, max=1.0)       # emptying onto the table pays nothing
    finish_q = 0.5 * (fill_tail + empty_q * fill_ok)
    release_step = torch.clamp(d_in, min=0.0) + torch.clamp(ctx.d_spill, min=0.0)
    unaimed = settled * release_step * (1.0 - perm)
    out_gate = not_nested * upright_rcv * clean_f

    # ------------------------------------------------------------------ G. control (iter_21 verbatim)
    v_src = src["v_lin"]
    w_src = ctx.src_cup_ang_vel.norm(dim=-1)
    w_rcv = ctx.rcv_cup_ang_vel.norm(dim=-1)
    pour_phase = ctrl_gate * lead_gate * perm_xy
    src_speed = ctrl_gate * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = W_SRC_SPIN * ctrl_gate * torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE) \
        + W_ROT_POUR * pour_phase * torch.tanh(torch.clamp(w_src - ROT_POUR_FREE, min=0.0) / SRC_W_SCALE)
    rcv_still_over = torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    rcv_steady = W_TILT_PEN_RCV * torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL) \
        + W_RCV_SPIN * ctrl_gate * torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE) \
        + W_STILL_POUR * ctrl_gate * perm_xy * rcv_still_over

    a_abs = ctx.actions.abs()
    da = ctx.actions - ctx.prev_actions
    flips = torch.clamp(-(ctx.actions * ctx.prev_actions), min=0.0)
    flip_palm_src = flips[:, 0:6].sum(dim=-1)
    flip_palm_rcv = flips[:, 12:18].sum(dim=-1)
    flip_hand = flips[:, 6:12].sum(dim=-1) + flips[:, 18:24].sum(dim=-1)
    act_flip = W_FLIP_PALM * (flip_palm_src + flip_palm_rcv) \
        + W_FLIP_HAND * flip_hand
    both_big = (a_abs > BANG_DEAD) & (ctx.prev_actions.abs() > BANG_DEAD)
    bang = (both_big & (ctx.actions * ctx.prev_actions < 0.0)).to(dtype)
    bang_src = bang[:, 0:6].mean(dim=-1)
    bang_rcv = bang[:, 12:18].mean(dim=-1)
    palm_bang = W_BANG_SRC * bang_src + W_BANG_RCV * bang_rcv
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 12:18] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    palm_sat = W_PALM_SAT_SRC * sat_src + W_PALM_SAT_RCV * sat_rcv
    cmd_rate = W_RATE_PALM_SRC * (da[:, 0:6] ** 2).sum(dim=-1) \
        + W_RATE_PALM_RCV * (da[:, 12:18] ** 2).sum(dim=-1) \
        + W_RATE_HAND * ((da[:, 6:12] ** 2).sum(dim=-1) + (da[:, 18:24] ** 2).sum(dim=-1)) \
        + W_ACT_RATE * (da ** 2).mean(dim=-1)
    arm_posture = W_ARM_LIMIT * (_limit_pen(ctx.src_arm_limit_margin, MARGIN_FREE)
                                 + _limit_pen(ctx.rcv_arm_limit_margin, MARGIN_FREE)) \
        + W_ARM_SHAKE * (_shake_pen(ctx.src_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)
                         + _shake_pen(ctx.rcv_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE))

    # ------------------------------------------------------------------ safety / disturbance (verbatim)
    cup_disturb = W_SLIDE_SRC * src["slide"] + W_SLIDE_RCV * rcv["slide"] \
        + W_ONE_SIDED_SRC * src["one_sided"] + W_ONE_SIDED_RCV * rcv["one_sided"] \
        + W_PALM_SRC * src["palm_push"] + W_PALM_RCV * rcv["palm_push"] \
        + W_RUSH_SRC * src["rush"] + W_RUSH_RCV * rcv["rush"] \
        + W_FLING * (src["fling"] + rcv["fling"])
    safety = W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0) \
        + W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)) \
        + W_NESTED * nested

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        # A
        "grasp_src": W_GRASP_SRC * src["value"],
        "grasp_rcv": W_GRASP_RCV * rcv["value"],
        # B
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        # C / D / E / H / open: difference forms and a cost - none of them pays for standing still
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": -tilt_dir,
        "tilt_open": W_TILT_OPEN * tilt_open,
        "pour_sink": W_POUR_SINK * pour_sink,
        # F
        "pour_delta": W_POUR * pour_paid - W_FLOW * flow_excess,
        "transferred": W_TRANSFER * in_t * out_gate,
        "transfer_tail": W_TAIL * finish_q * out_gate,
        "spill_delta": -W_SPILL * spill_step,
        "unaimed_release": -W_UNAIMED * unaimed,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype) * clean_f,
        # G
        "src_speed": -W_SRC_SPEED * src_speed,
        "src_spin": -src_spin,
        "rcv_steady": -rcv_steady,
        "act_flip": -act_flip,
        "palm_bang": -palm_bang,
        "palm_sat": -palm_sat,
        "cmd_rate": -cmd_rate,
        "arm_posture": -arm_posture,
        # safety
        "cup_disturb": -cup_disturb,
        "safety": -safety,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value

    # diagnostics only (added AFTER the sum)
    comps["diag_place_dist_r"] = d_place / r
    comps["diag_place_az_deg"] = torch.atan2(rel[:, 1], rel[:, 0]) * (180.0 / math.pi)
    comps["diag_side_excess_r"] = (es_lo + es_hi) / r
    comps["diag_rcv_mouth_dz_r"] = dz_m / r
    comps["diag_rcv_mouth_y"] = y_m
    comps["diag_body_clear_r"] = clear / r
    comps["diag_place_free_r"] = r_free / r
    comps["diag_aim_dist_r"] = d_aim / r
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_lip_dz_r"] = geo["lip_dz"] / r
    comps["diag_h_org"] = h_org
    comps["diag_win"] = win
    comps["diag_perm"] = perm
    comps["diag_toward"] = toward
    comps["diag_theta_rate"] = opn["dtheta"]
    comps["diag_open_phi"] = opn["phi"]
    comps["diag_open_rate"] = opn["phi_rate"]
    comps["diag_first_out"] = first_out
    comps["diag_first_spill"] = first_spill
    comps["diag_relieved_beads"] = settled * relieved * n_beads
    comps["diag_dir_err_deg"] = psi * (180.0 / math.pi)
    comps["diag_src_tilt_deg"] = ctx.src_cup_tilt * (180.0 / math.pi)
    comps["diag_lift_budget"] = lift_free_src
    comps["diag_ctrl_gate"] = ctrl_gate
    comps["diag_stage_cd"] = stage_cd
    comps["diag_release_beads"] = release_step * n_beads
    comps["diag_unaimed_beads"] = unaimed * n_beads
    comps["diag_clean"] = clean_f
    comps["diag_finish_q"] = finish_q
    comps["diag_grip_keep_src"] = src["keep"]
    comps["diag_balance_src"] = src["balance"]
    comps["diag_palm_sat_src"] = sat_src
    comps["diag_src_w"] = w_src
    comps["diag_src_v"] = v_src
    comps["diag_src_margin_min"] = torch.clamp(ctx.src_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_state_income"] = comps["grasp_src"] + comps["grasp_rcv"] + comps["lift_src"] + comps["lift_rcv"]
    comps["diag_outcome_income"] = comps["transferred"] + comps["transfer_tail"] + comps["success_bonus"]
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 3.66e-05, 0.000122, 4.88e-05, 0.00011, 0.000159, 0.000208, 8.54e-05, 0.000146, 0.000159]  min 0 · mean 0.000104 · max 0.00033
bead/spill: [0, 0.0192, 0.0163, 0.0137, 0.0107, 0.0154, 0.0122, 0.0163, 0.0163, 0.0133]  min 0 · mean 0.0148 · max 0.024
done/drop: [0, 0.000244, 0.000244, 0.000732, 0.000244, 0.000244, 0, 0.000488, 0.000244, 0.000732]  min 0 · mean 0.000295 · max 0.00146
done/mimic_runaway: [0, 0.000244, 0.000488, 0.000244, 0, 0.000244, 0.000244, 0.000244, 0, 0]  min 0 · mean 0.000128 · max 0.000977
episode_lengths/step: [88.4, 778, 762, 769, 793, 802, 791, 813, 764, 767]  min 88.4 · mean 771 · max 839
reward/act_flip: [-0.654, -0.365, -0.24, -0.18, -0.142, -0.126, -0.109, -0.0932, -0.0892, -0.0746]  min -0.654 · mean -0.17 · max -0.0639
reward/aim_lip: [0, 0.0213, 0.0526, 0.0862, 0.0948, 0.0419, 0.068, 0.0944, 0.121, 0.166]  min -0.235 · mean 0.0826 · max 0.197
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/arm_posture: [-0.256, -0.359, -0.385, -0.378, -0.382, -0.39, -0.397, -0.385, -0.426, -0.42]  min -0.436 · mean -0.392 · max -0.256
reward/cmd_rate: [-0.415, -0.262, -0.19, -0.157, -0.131, -0.118, -0.104, -0.092, -0.0854, -0.0736]  min -0.415 · mean -0.141 · max -0.0671
reward/cup_disturb: [-0.32, -0.134, -0.111, -0.0952, -0.0725, -0.079, -0.077, -0.0772, -0.0823, -0.0826]  min -0.604 · mean -0.0904 · max -0.0273
reward/diag_aim_dist_r: [10.4, 7.32, 6.36, 7.43, 7.98, 7.88, 6.14, 7.4, 7.23, 7.32]  min 4.46 · mean 7.99 · max 20.2
reward/diag_balance_src: [0, 0.571, 0.596, 0.626, 0.649, 0.637, 0.66, 0.649, 0.624, 0.639]  min 0 · mean 0.628 · max 0.732
reward/diag_body_clear_r: [-4.04, -0.165, 0.101, 0.88, 0.966, 1.02, 0.0282, 1.59, 1.13, 0.591]  min -4.04 · mean 0.347 · max 8.72
reward/diag_clean: [1, 0.959, 0.963, 0.969, 0.975, 0.963, 0.968, 0.963, 0.961, 0.967]  min 0.949 · mean 0.966 · max 1
reward/diag_ctrl_gate: [0, 0.407, 0.468, 0.541, 0.67, 0.656, 0.679, 0.657, 0.604, 0.616]  min 0 · mean 0.593 · max 0.834
reward/diag_dir_err_deg: [82.9, 78.3, 53.2, 34.3, 34.5, 35.4, 39.1, 42.1, 44, 45.8]  min 32.1 · mean 46.6 · max 99.6
reward/diag_finish_q: [0, 0.000111, 0.000187, 5.11e-05, 0.000163, 0.00035, 0.000214, 6.16e-05, 6.7e-05, 0.000118]  min 0 · mean 0.000153 · max 0.000572
reward/diag_first_out: [1, 0.933, 0.934, 0.943, 0.952, 0.932, 0.943, 0.937, 0.933, 0.942]  min 0.913 · mean 0.939 · max 1
reward/diag_first_spill: [1, 0.933, 0.934, 0.943, 0.952, 0.932, 0.943, 0.937, 0.933, 0.942]  min 0.913 · mean 0.939 · max 1
reward/diag_grip_keep_src: [0.212, 0.435, 0.459, 0.405, 0.366, 0.337, 0.337, 0.344, 0.323, 0.358]  min 0.212 · mean 0.371 · max 0.492
reward/diag_h_org: [-0.0537, 0.0506, 0.0544, 0.0726, 0.0778, 0.0798, 0.0535, 0.0963, 0.0795, 0.0623]  min -0.0626 · mean 0.06 · max 0.293
reward/diag_lift_budget: [0.4, 0.404, 0.413, 0.422, 0.418, 0.418, 0.415, 0.415, 0.421, 0.423]  min 0.4 · mean 0.417 · max 0.426
reward/diag_lip_dz: [-2.24e-05, 0.0994, 0.0933, 0.103, 0.112, 0.115, 0.0915, 0.133, 0.111, 0.0915]  min -0.0298 · mean 0.0959 · max 0.324
reward/diag_lip_dz_r: [-0.000796, 3.54, 3.32, 3.67, 4, 4.1, 3.26, 4.74, 3.95, 3.26]  min -1.06 · mean 3.42 · max 11.5
reward/diag_lip_xy: [0.291, 0.195, 0.171, 0.193, 0.207, 0.203, 0.155, 0.181, 0.181, 0.191]  min 0.114 · mean 0.207 · max 0.487
reward/diag_open_phi: [0, 0.000708, 0.0307, 0.094, 0.111, 0.0958, 0.0782, 0.177, 0.179, 0.205]  min 0 · mean 0.119 · max 0.306
reward/diag_open_rate: [0, 0.000519, -0.00881, 0.0123, -0.00465, -0.06, -0.0342, -0.0405, 0.0215, 0.126]  min -2.02 · mean 0.00567 · max 0.922
reward/diag_outcome_income: [0, 9.69e-05, 8.75e-05, 0.000156, 0.000405, 0.00028, 0.000295, 0.000225, 0.000291, 0.000651]  min 0 · mean 0.000212 · max 0.00154
reward/diag_palm_sat_src: [0.439, 0.42, 0.352, 0.277, 0.22, 0.203, 0.191, 0.198, 0.187, 0.175]  min 0.165 · mean 0.259 · max 0.588
reward/diag_perm: [0, 0.00634, 0.066, 0.124, 0.216, 0.174, 0.199, 0.375, 0.284, 0.297]  min 0 · mean 0.198 · max 0.397
reward/diag_place_az_deg: [90.1, 70.5, 76.8, 84.9, 84.2, 83, 85.8, 91.9, 91.5, 95.9]  min 35.6 · mean 83.8 · max 97.7
reward/diag_place_dist_r: [10.1, 8.29, 6.98, 7.8, 7.44, 7.71, 5.71, 7.16, 6.26, 6.58]  min 3.96 · mean 8.07 · max 20.1
reward/diag_place_free_r: [2, 1.44, 1.46, 1.55, 1.69, 1.7, 1.68, 1.6, 1.65, 1.59]  min 1.37 · mean 1.6 · max 2
reward/diag_rcv_mouth_dz_r: [-5.34, -3.83, -3.78, -3.54, -3.05, -3.13, -2.84, -2.82, -2.71, -2.68]  min -5.34 · mean -2.93 · max 0.757
reward/diag_rcv_mouth_y: [0.16, 0.171, 0.15, 0.136, 0.105, 0.101, 0.0873, 0.0866, 0.0806, 0.0619]  min 0.015 · mean 0.108 · max 0.252
reward/diag_release_beads: [0, 0.00903, 0.00586, 0.00342, 0.0022, 0.00269, 0.0022, 0.00293, 0.00366, 0.0022]  min 0 · mean 0.00381 · max 0.0122
reward/diag_relieved_beads: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.75e-10]  min 0 · mean 6.87e-06 · max 0.000363
reward/diag_side_excess_r: [0, 0.715, 0.287, 1.44, 0.0412, 1.63, 0.379, 1.37, 0.126, 0.272]  min 0 · mean 1.16 · max 9.34
reward/diag_src_margin_min: [0.194, 0.116, 0.101, 0.0953, 0.0957, 0.0965, 0.089, 0.0897, 0.0791, 0.076]  min 0.0692 · mean 0.0931 · max 0.194
reward/diag_src_tilt_deg: [0.997, 16.8, 24.6, 32.2, 27.8, 27, 24.3, 26.1, 32, 34.3]  min 0.997 · mean 27.4 · max 38.3
reward/diag_src_v: [0.0362, 0.217, 0.22, 0.191, 0.208, 0.241, 0.183, 0.242, 0.211, 0.215]  min 0.0362 · mean 0.225 · max 8.69
reward/diag_src_w: [0.497, 1.55, 1.87, 1.27, 1.18, 3.04, 1.28, 1.24, 1.49, 1.36]  min 0.497 · mean 1.84 · max 133
reward/diag_stage_cd: [0, 0.506, 0.572, 0.631, 0.709, 0.7, 0.704, 0.693, 0.653, 0.652]  min 0 · mean 0.648 · max 0.854
reward/diag_state_income: [0.46, 2.66, 2.84, 2.92, 3, 2.96, 3.03, 3.03, 2.94, 2.97]  min 0.46 · mean 2.93 · max 3.38
reward/diag_theta_rate: [0.114, 0.0829, 0.314, 0.127, -0.00223, -0.293, -0.0996, -0.031, -0.00436, 0.0222]  min -40.5 · mean -0.0654 · max 10.9
reward/diag_toward: [0.555, 0.597, 0.813, 0.916, 0.914, 0.915, 0.915, 0.917, 0.891, 0.883]  min 0.388 · mean 0.856 · max 0.934
reward/diag_unaimed_beads: [0, 0.00903, 0.00563, 0.00342, 0.00216, 0.00269, 0.0022, 0.00291, 0.00366, 0.00195]  min 0 · mean 0.00375 · max 0.0122
reward/diag_win: [0, 0.0124, 0.0736, 0.127, 0.217, 0.175, 0.202, 0.376, 0.285, 0.301]  min 0 · mean 0.2 · max 0.398
reward/grasp_rcv: [0.217, 0.748, 0.792, 0.796, 0.796, 0.788, 0.798, 0.8, 0.777, 0.727]  min 0.217 · mean 0.775 · max 0.845
reward/grasp_src: [0.243, 0.895, 0.951, 0.963, 0.964, 0.948, 0.959, 0.954, 0.927, 0.965]  min 0.243 · mean 0.948 · max 1.13
reward/lift_rcv: [0, 0.235, 0.262, 0.284, 0.33, 0.323, 0.339, 0.337, 0.322, 0.311]  min 0 · mean 0.306 · max 0.401
reward/lift_src: [0, 0.785, 0.833, 0.876, 0.909, 0.904, 0.932, 0.943, 0.918, 0.962]  min 0 · mean 0.898 · max 1.09
reward/palm_bang: [-0.845, -0.55, -0.398, -0.355, -0.299, -0.259, -0.224, -0.197, -0.171, -0.135]  min -0.878 · mean -0.3 · max -0.11
reward/palm_sat: [-2.12, -1.91, -1.61, -1.29, -1.05, -0.954, -0.908, -0.911, -0.893, -0.853]  min -2.42 · mean -1.2 · max -0.84
reward/place_rcv: [0, 0.0222, 0.0424, 0.0713, 0.0843, 0.0669, 0.0813, 0.0853, 0.083, 0.0968]  min -0.228 · mean 0.0668 · max 0.119
reward/pour_delta: [0, 0, -0.00183, 0, 0, 0, 0, 0, 0, 0]  min -0.00183 · mean 0.000124 · max 0.00366
reward/pour_sink: [0, 0.00457, 0.0204, 0.0201, 0.0357, 0.0369, 0.0382, 0.0234, 0.0671, 0.0705]  min 0 · mean 0.0394 · max 0.106
reward/rcv_steady: [-0.00403, -0.167, -0.195, -0.201, -0.223, -0.201, -0.199, -0.214, -0.19, -0.204]  min -0.273 · mean -0.2 · max -0.00403
reward/safety: [-0.0191, -0.00563, -0.00585, -0.00734, -0.00246, -0.0035, -0.00466, -0.00259, -0.00336, -0.00681]  min -0.0219 · mean -0.00526 · max -0.00102
reward/spill_delta: [0, -0.108, -0.0703, -0.041, -0.0264, -0.0322, -0.0264, -0.0352, -0.0439, -0.0264]  min -0.146 · mean -0.0453 · max 0
reward/src_speed: [0, -0.054, -0.0565, -0.0433, -0.0433, -0.045, -0.0455, -0.0361, -0.0424, -0.0442]  min -0.26 · mean -0.0481 · max 0
reward/src_spin: [0, -0.0218, -0.0331, -0.0304, -0.0362, -0.0454, -0.0467, -0.0562, -0.0617, -0.0594]  min -0.0752 · mean -0.0444 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_dir: [0, -0.0791, -0.044, -0.018, -0.0165, -0.0167, -0.0196, -0.0233, -0.0302, -0.0372]  min -0.111 · mean -0.0341 · max 0
reward/tilt_open: [0, -8.19e-05, -0.00428, -0.00583, -0.00564, -0.00721, -0.00674, -0.0083, -0.00762, -0.00466]  min -0.0121 · mean -0.00546 · max 4.31e-05
reward/total: [-3.17, -0.282, 0.609, 1.29, 1.78, 1.83, 2.05, 2.09, 2.06, 2.26]  min -3.26 · mean 1.44 · max 2.37
reward/transfer_tail: [0, 0, 8.95e-06, 0, 1.17e-06, 0, 9.51e-06, 3.07e-06, 1.45e-05, 3.37e-05]  min 0 · mean 1.34e-05 · max 0.000387
reward/transferred: [0, 9.69e-05, 7.85e-05, 0.000156, 0.000403, 0.00028, 0.000286, 0.000222, 0.000277, 0.000618]  min 0 · mean 0.000198 · max 0.00122
reward/unaimed_release: [0, -0.0271, -0.0169, -0.0103, -0.00647, -0.00806, -0.00659, -0.00873, -0.011, -0.00586]  min -0.0366 · mean -0.0112 · max 0
rewards/step: [-191, -175, 453, 932, 1.29e+03, 1.43e+03, 1.66e+03, 1.66e+03, 1.6e+03, 1.64e+03]  min -835 · mean 1.11e+03 · max 1.89e+03
task/aim_dist: [0.319, 0.27, 0.243, 0.265, 0.282, 0.28, 0.235, 0.278, 0.261, 0.264]  min 0.184 · mean 0.284 · max 0.627
task/cup_collision_rate: [0, 0.000732, 0.00146, 0.000977, 0, 0.000488, 0.00122, 0.000244, 0.000732, 0.00171]  min 0 · mean 0.000866 · max 0.00317
task/cups_center_dist: [0.32, 0.272, 0.255, 0.287, 0.3, 0.297, 0.251, 0.292, 0.28, 0.283]  min 0.205 · mean 0.3 · max 0.648
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.000732, 0.000977, 0.00269, 0.000732, 0.000732, 0.00122, 0.000488, 0.000732, 0.0022]  min 0 · mean 0.00134 · max 0.0061
task/rcv_cup_lift: [7.95e-05, 0.043, 0.0443, 0.0511, 0.0647, 0.0626, 0.0707, 0.0711, 0.0741, 0.0751]  min 7.95e-05 · mean 0.0682 · max 0.172
task/rcv_grasped: [0, 0.737, 0.797, 0.803, 0.811, 0.799, 0.799, 0.79, 0.781, 0.755]  min 0 · mean 0.783 · max 0.909
task/rcv_hand_foreign_rate: [0.051, 0.01, 0.0061, 0.00269, 0.00146, 0.00269, 0.00244, 0.00195, 0.00171, 0.00293]  min 0.000244 · mean 0.00409 · max 0.0598
task/src_cup_lift: [0.000278, 0.147, 0.152, 0.177, 0.196, 0.196, 0.178, 0.221, 0.207, 0.191]  min 0.000278 · mean 0.182 · max 0.4
task/src_grasped: [0, 0.682, 0.726, 0.75, 0.765, 0.762, 0.773, 0.764, 0.747, 0.77]  min 0 · mean 0.75 · max 0.927
task/src_hand_foreign_rate: [0.00195, 0.000977, 0.00122, 0.00244, 0.00146, 0.0022, 0.00244, 0.0022, 0.00269, 0.00195]  min 0 · mean 0.00189 · max 0.00513
task/src_tilt_deg: [0.997, 16.8, 24.6, 32.2, 27.8, 27, 24.3, 26.1, 32, 34.3]  min 0.997 · mean 27.4 · max 38.3
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# pour_bi_rh 보상 설계 브리프 (09.25 종합판 — iter_09 ~ iter_19 전체를 한 문서로)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트에 통째로 들어간다.**

## 0. 왜 이 문서가 필요한가 (사용자 지적 09.25)

- 생성기는 매 라운드 **직전 보상 코드 1개 + 그 런의 학습 곡선 표(태그별 10개 표본점) + 그 라운드 노트**만 본다.
  "10" 은 항 개수 제한이 아니라 **태그당 표본점 수**다(태그 ~126 개 전부 들어간다). 문제는 다른 데 있다:
  1. **이전 라운드들의 교훈이 누적되지 않는다.** 예: iter_17 이 각도 처방을 지울 때 방향 벌점(mis_tilt)까지 지워
     i18 이 받는 컵 반대쪽(127° 어긋남)으로 부었다. 방향이 필요하다는 교훈은 iter_08 에서 이미 얻었었다.
  2. **학습 곡선 표가 실제 동작과 크게 달랐다.** `task/src_tilt_deg` 는 전 env·전 스텝 평균이라 접근·파지 구간이 섞여
     i18 에서 곡선 34° 대 결정론 재생 env 별 최대 중앙 87° 였다. 09.25 부터 에피소드 지표(`ep/*`)와 결정론 자동 평가가 추가됐다.
  3. **보상이 19 라운드 동안 패치로만 커졌다.** iter_19 는 822 줄·67 항이다. 서로 싸우는 항(자세 유지 수입 대 배달)이 쌓였다.
- 따라서 **iter_20 은 패치가 아니라 이 브리프 전체를 근거로 다시 설계**한다.

## 1. 과제와 고정 조건

- 오른팔(소스)이 비드 20 개가 든 컵을 잡아 들고, 왼팔(리시버)이 빈 컵을 잡아 든다. 소스를 기울여 비드를 리시버 컵에 옮긴다.
- 목표: **흘림 없는 이송**(성공 = in_target ≥ 0.80 · spill ≤ 0.25 · 20 스텝 유지, env 판정).
- ★사용자 원칙: **붓는 각도의 크기는 보상이 정하지 않는다 — 학습이 찾는다.** 방향·위치(과제 기하)는 보상이 알려줘도 된다.
- ★사용자 원칙: 실기는 안정 속도 틸팅. 속도·제어 벌점을 풀어 효율을 얻지 않는다.
- ★사용자 원칙: 양팔 작업은 **y ≈ 0 중심선 부근**(FPP 시야). 조준·허가 창은 `ctx.cup_radius` 배수로 쓴다.
- 액션 24 = 소스 palm 6 · 소스 손 6 · 리시버 palm 6 · 리시버 손 6. palm 델타 박스(앵커 기준, 도):
  소스 yaw **±45** · pitch ±45 · roll **−150 ~ +45** / 리시버 ±30 전부. 위치는 소스 x ±0.15 · y −0.10~+0.40 · z −0.18~+0.25.
- 스폰(컵 원점, env-local): 소스 (0.382, −0.159, 0.268) · 리시버 (0.378, +0.165, 0.268) — **리시버는 소스의 정왼쪽(+y)**.

## 2. 측정으로 확정된 물리·기구학 제약

| # | 사실 | 근거 |
|---|---|---|
| K1 | **컵이 기우는 방향은 손바닥 yaw 에 묶여 있다.** (기우는 방위 − palm yaw) 중앙 −4°, 표준편차 19°. 기울기 방향을 바꾸려면 palm yaw 를 돌려야 하고 명령 범위는 ±45°. | i18 ep800 trace |
| K2 | 기울기 60° 이상일 때 컵이 기우는 방위 중앙 **+68°**(앞쪽 왼편, IQR +59~+81). 즉 **소스가 부을 수 있는 쪽은 자기 왼쪽 앞**이다. 스폰 배치(리시버가 정왼쪽)와 일치한다. | i18 ep800 trace |
| K3 | **손은 높은 곳에서만 명령한 회전을 만든다.** 같은 회전 명령에서 오차: palm z 0.29~0.42 → 143~151°, 0.53 → 38~52°, **0.69~0.76 → 10~23°**. 낮으면 어깨 j2 가 하한(−0.17)에 먼저 닿고 손목 j5·j6·j7 이 범위를 다 쓴다(j6 93.6 %·j7 98.8 % 대 j2 22.6 %). | 도달성 프로브·i17g trace |
| K4 | 붓는 각 자체는 **각도 수입 없이도 학습이 찾는다**: i18 결정론 env 별 최대 기울기 중앙 87°, 비드 방출 순간 중앙 91°. | i18 ep800 |
| K5 | 받는 컵 입구 반지름 0.028 m, 비드 지름 0.012 m. 주둥이가 입구 위에 있다고 할 폭은 컵 반지름 규모다. | 자산 |
| K6 | 소스 컵 원점 → 림 0.054 m(scale 0.65). 기울이면 주둥이가 `r·sinθ` 만큼 내려간다 — 높이 조건은 **주둥이**로 본다. | 자산 |

## 3. 라운드별 교훈 (무엇이 먹혔고 무엇이 실패했나)

| 라운드 | 변경 | 결과 | 교훈 |
|---|---|---|---|
| i08 | 방향 부호 있는 기울기(theta) 도입 | 방향은 잡혔다 | **방향은 반드시 보상에 있어야 한다** |
| i09 | 기울기 연동 높이 예산(lip_drop) | 첫 성공 | 기울이면 주둥이가 내려가므로 높이 허용이 따라가야 한다 |
| i10~12 | 흘림 240·조준 벌점 강화 | 초반 정점 후 **단조 붕괴**(흘림/유출 0.18 → 0.43) | 조준 척도가 너무 넓어(60 cm 까지 점수) 멀리서 기울여도 수입 |
| i13 | 접근(차분)·허가(컵 반지름 창) 분리 | **붕괴 없음**, 결정론 in_target 0.906 | 접근은 차분 형태, 허가는 좁은 창 — 유지할 것 |
| i14 | 전량 이송(transfer_tail·src_emptied) | 짧은 구간에서 in_target 0.53 까지 상승 | 절반만 넘기면 끝나는 구조를 깼다 |
| i15 | 관절 한계 벌점 | **실패**: 한계 접촉 0.21 → 0.43, 붓기만 감소. corr(limit, in_target) +0.90 | 손목 한계는 결과지 원인이 아니다(K3) — 벌점으로 못 고친다 |
| i16w | 만남 통로(스폰 기준) + 웜스타트 | 리시버가 건너갔다 되돌아옴 3 회 | 웜스타트는 옛 자세를 통째로 물려받는다 |
| i16g | 파지 단계 ckpt + 같은 보상 | 리시버 습관 없이 건너옴(frac 1.05 → 0.6), 결정론 87.6°·in_target 0.15 | **파지 단계에서 시작하는 것이 옳다** |
| i17 | 각도 처방 전면 제거, 탐색 죽은 ckpt | 붓기 소멸(결정론 47°·배달 0) | 결과 전용 보상은 **첫 한 알**이 나가야 경사가 생긴다 — 탐색이 필요 |
| i17g | 같은 보상, 파지 단계 ckpt | 자력으로 30° 까지 → 22° 로 내려앉음 | **과들기 세금**이 기울이기를 막았다(i18 에서 확인) |
| i18 | 들기 허용 0.24 → 0.40 | 과들기 벌점 −0.09~−0.15 → −0.003, 결정론 최대 87°·방출 91° | 붓는 각은 찾았다. 그러나 **방향 127° 어긋남·받는 컵과 같은 높이·주둥이 수평거리 15.7 cm** |

## 4. i18 이 실패한 구조적 이유 (종합)

1. **배치가 뒤집혔다.** 만남 통로가 리시버를 **스폰 기준**으로 오른쪽으로 끌어, 소스를 지나쳐 **뒤쪽 오른편**(방위 −163°)에 세웠다.
   소스가 부을 수 있는 쪽은 K2 의 +68° ± 45° 이므로 **도달 불가능한 배치**다. 이 상태에서 방향 벌점을 주면 i15 처럼 붓기만 줄어든다.
2. **방향 신호가 없었다.** iter_17 이 mis_tilt 를 지웠다. 방향은 허가 게이트에만 남아 틀린 방향 자체에는 비용이 없었다.
3. **높이를 버리고 기울였다.** 0.62 m 까지 들었다가 기울이면서 0.36 m 로 내려왔고(K3 와 반대), 손목 j5·j7 을 −1.3 rad 로 꺾었다.
4. **보상 예산이 배달을 압도한다.** 파지·들기·운반·정렬·수렴 같은 **상태 비례 수입이 스텝당 약 12~15**(i17g e1000: 총 15.3,
   그중 align 2.48·lift 2.42·meet 2.13·converge 2.12·carry 1.66·grasp 2.67·alive 1.0). 에피소드 900 스텝이면 약 13,000.
   반면 배달은 한 알 +7.5 × 20 = 150 이다. **아무것도 붓지 않고 자세만 잡아도 거의 전부를 번다** — 형제 트랙에서 확인된 "주차 함정" 구조다.

## 5. iter_20 설계 요구 (다시 설계 — 패치 금지)

### 5-1. 구조: 단계별, 항 수를 줄인다
각 항은 **한 가지 역할**만 한다. 목표 25 항 이하. 역할이 겹치는 항은 합친다. iter_19 의 검증된 기하 헬퍼(`_grip_geometry`, 방향 theta, lip_dz, lip_xy)는 재사용한다.

| 단계 | 역할 | 형태 |
|---|---|---|
| A 파지 | 양손 파지 성립·유지(기존 grasp·grip_hold) | 상태(작게) — 파지 단계 ckpt 가 이미 안다 |
| B 들기 | 소스는 **높이**(K3: 손바닥 0.6 m 이상이 회전에 유리), 리시버는 낮게 | 소스 들기 허용 상향 유지(무료 상한 컵 z ≈ 0.67), 과들기 벌점은 그 위에서만 |
| C 배치 | ★리시버 컵 입구를 **소스 컵 기준 부을 수 있는 쪽**(K2: 소스의 왼쪽 앞, 방위 +68° ± 45°)·**소스보다 낮게**·중앙선 근처에 둔다 | **소스 기준 상대 좌표**, 컵 반지름 배수, **차분(진행도) 형태**. 스폰 기준 만남 통로(meet_cross·src_cross)는 제거 |
| D 조준 | 소스 주둥이를 받는 컵 입구 **위**(lip_dz > 0)·**수평으로 입구 안**(lip_xy < 1~3 r)으로 | 차분 형태 접근(approach 를 수평+높이 3D 로), 허가 창 perm 유지 |
| E 방향 | 기우는 방향이 받는 컵 쪽이어야 한다 | 비용 = 기울기 크기 × 방향 어긋남(iter_19 tilt_dir). **각도 크기 목표는 없다** |
| F 결과 | 비드가 받는 컵에 들어감 = 수입, 흘림 = 벌점 | pour_delta·transferred·transfer_tail·src_emptied − spill·flow_excess·unaimed_release |
| G 제어 | 안정 속도·부드러운 명령 | 속도·스핀·act_flip·palm_bang·palm_sat·carry_rate(유지), 관절 한계/떨림은 약하게(0.4) |
| H 높이 유지 | 붓는 동안 가라앉지 않기 | pour_sink(차분) — iter_19 유지 |

### 5-2. ★예산 규칙 (주차 함정 방지)
- **상태 비례 수입(A~D)의 합이 성공 상태의 결과 수입(F)보다 작아야 한다.** 목표: 성공 상태(in_target 0.9, 두 컵 파지·조준)에서
  F 가 스텝당 수입의 **절반 이상**. 지금은 F ≈ 0 인데 A~D ≈ 12~15/스텝이다.
- 방법: (1) C·D 는 **차분 형태**로 준다(가까워질 때만 값, 머물면 0). (2) 운반·정렬·수렴 같은 상태 수입은 **상한을 낮추거나**
  배달이 시작되면(in_target > 0) 줄어들게 한다. (3) transferred 의 스텝당 값을 A~D 합보다 크게 둔다.
- **탐색 경사:** 첫 한 알 전에도 방향을 알려줄 신호가 필요하다(i17 교훈). E(방향)·D(조준 접근)가 그 역할을 한다.
  비드 자체의 움직임(소스 안 비드가 입구 쪽으로 쏠림)에 작은 신호를 주는 것도 허용된다 — 각도 처방이 아니다.

### 5-3. 하지 말 것
- 기울기 크기 목표·대역·잔량 연동 방출각 사다리(release_tilt)·각도 상한 벌점(overtilt·early_tilt).
- 스폰 기준 만남 통로(리시버를 소스 너머로 끈다).
- 관절 한계를 강한 벌점으로 누르기(i15 실패).
- 새 절대 거리 상수(조준·배치는 `ctx.cup_radius` 배수, 높이 허용만 예외).

## 6. 평가 (라운드 판정 기준)

- 학습 곡선의 `task/*` 평균이 아니라 **`ep/*`(끝난 에피소드 기준)와 결정론 자동 평가(`<label>_deteval.tsv`)**로 판정한다.
- 핵심 지표: `ep/toward_max`(받는 컵 쪽 최대 기울기) · `ep/dir_err_at_max`(최대 기울기 순간 방향 오차) ·
  `ep/lipdz_at_max`(그 순간 주둥이가 입구보다 위인가) · `ep/in_target_max` · `ep/spill_end`.
- 성공 판정: 결정론 평가에서 in_target_max ≥ 0.8 · dir_err_at_max < 30° · lipdz_at_max > 0.

## 7. 웜스타트

- **파지 단계 ckpt rh2_i07-r3 ep400**(entropy ~22, 양손 파지 0.71/0.75, 붓기 자세 없음). 붓기 습관을 물려받지 않는다.


# round 23 operator notes: i22 는 붓기 직전 자리(주둥이가 입구 위 0.9 r)에 머물며 속도 판독으로 번다. 그 두 경로만 막는다

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. 수치 출처: 결정론 trace(64 env · 900 스텝 · ADR 0) ep250·500·600·700 과
`iter_22/compute_reward.py`. r = `ctx.cup_radius` = 0.028 m.

## i22 결과 (t2r_rh3_i22, 파지 단계 ckpt, ep700 에서 라운드 끝)

| ep | env별 최대 기울기 중앙 | >60° env | 최대 기울기 순간 주둥이 수평 / 높이 | 소스를 떠난 비드 | 조용한 붓기* → 들어감 | 회전 급등 중 방출 → 들어감 |
|---|---|---|---|---|---|---|
| 250 | 61.6° | 33/64 | 2.46 r / 2.37 r | 58 | 17 → 11 | 41 → 2 |
| 500 | 50.6° | 1/64 | 1.99 r / 1.25 r | 28 | 0 | 28 → 1 |
| 600 | 67.7° | 43/64 | 1.94 r / 1.19 r | 157 | 0 | 157 → 2 |
| 700 | 70.5° | 48/64 | 1.48 r / 0.90 r | 152 | 1 → 0 | 151 → 0 |

*조용한 붓기 = 방출 직전 5 스텝 파지 유지, 소스 |ω| < 5 rad/s. 주둥이 = 보상의 lip(`diag_lip_xy`, `diag_lip_dz`).

- ep250 의 조용한 붓기 17 개: 기울기 92° 부근, 주둥이 수평 1.19 r, 입구 위 2.2 r, 방향 오차 11.5°, palm z 0.54 m, 창 0.97.
  그중 11 개(65 %)가 들어갔다. **pour_delta 는 기하상 도달 가능하다.**
- ep700 은 주둥이를 입구 가까이(1.48 r) 가져왔지만 **입구 위 0.9 r 로 낮고 70° 부근에서 멈춘다.** 기울기 >85° 스텝은 64 env 전체에서 12 개.
- ep700 의 방출 152 개 중 151 개는 붓기가 아니다. 방출 ±3 스텝 안 소스 |ω| 최대 중앙 65 rad/s(자기 축 성분 78 %),
  손가락 힘 최대 중앙 78 N(>20 N 이 91 %). 이 비드는 받는 컵 축에서 중앙 11 r 떨어진 곳을 지나 떨어졌다.
  "방출 순간 방향 오차·주둥이 거리"는 이런 사고 방출이 섞인 값이다.

보상 항 합(env 당 첫 에피소드, trace 의 `term_*`)과, 같은 항을 **위치 차분**으로 다시 계산한 값
(`diag_aim_dist_r`·`diag_place_dist_r`·`diag_h_org` 의 스텝 차분에 같은 무게·게이트·±1 clamp):

| ep | aim_lip 지불 / 위치 차분 | place_rcv | pour_sink | 초과 합 |
|---|---|---|---|---|
| 250 | 106 / 105 | 92 / 65 | 49 / −10 | +87 |
| 500 | 81 / 82 | 81 / 55 | 124 / −18 | +167 |
| 600 | 124 / 96 | 77 / 61 | 143 / 7 | +180 |
| 700 | 201 / 92 | 97 / 54 | 153 / 18 | **+288** |

- ep700 의 결과 수입(pour_delta·transferred·transfer_tail·success)은 0 이다. 에피소드 합은 ep250 2630 → ep700 2960 으로 올랐다.

## 원인 (코드: iter_22/compute_reward.py)

1. **[확인: 코드+trace] aim_lip·place_rcv 가 컵 자기 축 회전을 주둥이의 접근으로 센다.** 두 항의 변화율은
   `v_lip = _point_vel(src_cup_lin_vel, src_cup_ang_vel, src_cup_pos, lip_pt)` 에서 온다(409·411·453·464 줄). lip 은 "d_hat 쪽 림 점"이라는
   **기하 점**이다. 컵이 자기 축으로 돌면 재질은 움직여도 lip 은 제자리인데, 이 공식은 ω_축 × (lip − 원점) 을 lip 의 속도로 센다.
   - aim 초과 지불은 (축 회전 부호 × lip 의 d_hat 기준 좌우 치우침 부호) = −1 인 배치에서만 난다: +0.196/스텝(반대 배치 −0.022).
   - ep700 은 이 배치에 91 %(ep250 71 %), 좌우 치우침 중앙 0.85 r(ep250 0.56 r). 즉 **좌우로 비켜 선 주둥이에 돈을 준다.**
     방향 오차 증가(16° → 28°)와 같은 방향이다. tilt_dir 가 창 안 25° 에 매기는 값은 0.01/스텝으로 이 초과의 1/20 이다.
   - 기울인 채 창 안에서 축 회전 판독은 +0.42 rad/s, env 별 부호 유지 91 % 다. 같은 구간 컵 안 비드 방위는 그만큼 돌지 않는다
     (적분 1.17 rad 대 비드 0.15 rad) [가설: 판독이 실제 회전보다 크다].
2. **[확인: trace] pour_sink 가 실제 높이 변화 없이 번다.** ep700 +153/에피소드 대 위치 차분 +18. 초과는 축 회전 부호와 무관하다
   (+0.32 / +0.36 /스텝). [가설] ctx 선속도의 수직 성분이 위치 변화와 맞지 않는다(접촉 보정 속도).
3. **[확인: 기하+trace] pour_sink 의 켜짐 구역이 붓는 자세와 겹치지 않는다.** pour_sink 는 소스 원점이 받는 컵 입구 위 1.7 r 안
   (`org_low`)일 때만 켜진다. h_org − lip_dz = r·sinθ − mouth_z·cosθ 이다(trace 확인: 70° +0.27 r, 80° +0.57 r, 90° +1.02 r).
   - 90° 에서 org_low 안이면 lip_dz < 0.7 r, 곧 창 바닥 아래다. 겹치는 구간은 70° 부근 lip_dz 0.7 ~ 1.4 r 뿐이다.
   - ep700: 기울기 60 ~ 80° 스텝의 87 ~ 98 % 가 org_low 안, 주둥이 0.9 r(80° 에서 0.45 r), pour_sink +0.30 ~ +0.41/스텝.
     더 기울이면 창 바닥 아래로 가거나 org_low 를 떠나 이 수입을 잃는다 → **70° 부근 정지.**
   - ep250 의 92° 붓기는 주둥이가 입구 위 2.2 r, org_low 밖이었다(기울기 >60° 스텝의 2 % 만 org_low).
4. **[확인: trace] 비드 손실은 회전 급등 사고다.** 두 컵을 든 뒤 소스 |ω| > 5 rad/s 스텝: ep700 297(ep250 125), 자기 축 성분 중앙 0.77.
   그중 38 % 에서 손가락 힘 > 20 N(조용한 스텝 0 %). 그때 손 명령은 검지·중지·약지·새끼 +1.0(완전 굽힘), 약지·새끼 접촉률 37 %·3 %.
   컵이 손에서 튀어 나가기도 한다(예: ep700 env 61, 4.8 m/s). [가설] 굽혀 둔 손가락에 도는 컵이 걸려 접촉 해석이 튄다. 원인 1 의 축 회전 지불이 이 회전을 부추겼을 수 있다 [가설].
5. 이번 원인이 아닌 것: 창 폭(붓기 자리에서 창 0.9 ~ 1.0), tilt_dir 창 안 0.3(방향 오차는 원인 1 이 돈을 준 것),
   흘림 비용(ep700 흘림 −19/에피소드, 비드가 흐르기 전에 멈춘다).
   참고 계산: 조용한 붓기 적중 65 % 에서 비드당 pour_delta+spill 기대 +0.7. transferred·transfer_tail 은 흘림 5 개째에 clean_f = 0 이 되어 0 이다.

## iter_23 변경 (이것만, 나머지 설계는 iter_22 그대로)

1. **aim_lip·place_rcv 변화율을 주둥이 기하의 정확한 미분으로.** `_tilt_open_potential` 이 이미 계산하는 dlip_xy·dlip_dz
   (소스 입구 = 축 위 점의 속도 + dθ/dt + d(d_hat)/dt. du = ω × up 이라 자기 축 회전은 0)를 돌려받아
   place(dlip_xy)·aim(dlip_xy, dlip_dz)에 쓴다. `v_lip = _point_vel(..., geo["lip_pt"])` 는 쓰지 않는다.
   무게·창·대칭 clamp(±1)·V_PROG 는 그대로.
2. **pour_sink 를 합에서 뺀다(항 제거, 합산 25 항).** 높이 하한은 조준 창 바닥(aim 의 `below`, 창의 w_lo)이 이미 맡는다.
   하강 비용만 남기는 안은 쓰지 않는다: 비대칭 clamp 는 잡음을 순손실로 만들고, ctrl_gate(파지 플래그) 곱이 있어 컵을 놓으면 0 이 된다(둘 다 i20 교훈).
   ep700 에서 이 항은 순수입 +153/에피소드였다. 제거는 벌점 완화가 아니라 수입 제거다.
3. **진단만 추가(합산 뒤, 보상 불변).** `diag_src_axial_w` = src_cup_ang_vel · src_cup_up,
   `diag_src_n_contact`·`diag_rcv_n_contact` = 힘 > 1.0 N 인 손가락 수(사용자 결정: 5지 파지는 로깅 먼저).

지키는 것(사용자 원칙): 각도 처방 금지(각도 목표·대역·상한 없음) · 속도·제어·안전 벌점(G·safety 전부) 값과 게이트 그대로 ·
창은 `ctx.cup_radius` 배수(허가 1 ~ 3 r, 높이 바닥 0.7 r·천장 4 r, 배치 창 그대로) · y ≈ 0 중심선 ·
예산 규칙(브리프 5-2, 상태 수입 < 결과 수입) · tilt_open·tilt_dir(0.3 / 1.5)·첫 비드 흘림 완화·배치 목표·결과 무게는 iter_22 그대로 ·
변경은 위 3 개뿐.

사용자 성공 정의(09-25 결정. 판정 쪽 결정이고 이번 보상 변경이 아니다): in_target ≥ 0.80 · spill ≤ 0.25 · 20 스텝 유지 ·
받는 컵 기울기 ≤ 20° · 첫 들기 뒤 파지 끊김 0 · 5지 파지는 로깅 먼저.
참고: 지금 파지 플래그는 첫 들기 뒤 끊김이 ep700 소스 354 회/64 env(63 % 가 1 스텝)다. 끊김 0 판정은 이 깜빡임 처리부터 정해야 한다.

## 웜스타트 (메인 결정용)

- **ep250 권고, ep700 아님.** ep700 은 원인 1·3 의 균형점이다: 기울기 >60° 스텝의 89 % 가 org_low 안, 좌우 치우침 배치 91 %,
  조용한 붓기 1 회. iter_23 은 그 수입을 없애므로 ep700 에서 시작하면 낮은 자리 습관부터 버려야 한다.
  ep250 은 높은 주둥이(2.2 r)·92° 붓기 모드가 살아 있고(17 회 → 11 개), 초과 지불이 +87 로 가장 작다.
  ckpt: 서버 `t2r_rh3_i22/nn/last_open-rh_b_pour_fab_mimic_ep_250_rew_921.32996.pth`.
- 단 TaskSpec REQ-grasp-stage-warmstart 는 파지 단계 ckpt 만 허용한다(붓기 습관을 가진 ckpt 금지). ep250 은 붓기 자세를 가진 ckpt 라
  **사용자 확인이 필요하다.** 확인이 없으면 파지 단계 ckpt rh2_i07-r3 ep400(i21·i22 와 같음).
