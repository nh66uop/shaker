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
# iter_23 = iter_22 design + ONLY the three round-23 operator changes (09.26):
#   (1) aim_lip / place_rcv rates = the EXACT time derivative of the spout geometry. dlip_xy / dlip_dz come from
#       _tilt_open_potential (velocity of the source MOUTH, an on-axis point, + d theta/dt + d d_hat/dt, with
#       du = w x u), so spin of the cup about its own axis leaves the geometric lip where it is and pays exactly 0.
#       The body-point velocity v_lip = _point_vel(..., lip_pt) is no longer used (i22 ep700: aim paid 201 vs a
#       position difference of 92 per episode, the excess from axial spin with the spout offset sideways).
#       Weights, windows, the symmetric +-1 clamp and V_PROG unchanged.
#   (2) pour_sink removed from the sum (25 summed terms). It paid +153/episode at ep700 against +18 of real height
#       change, and its active zone (source origin < 1.7 r above the receiver rim) overlaps the aimed window only
#       near 70 deg, which parked the pour there. The window floor (aim `below`, window w_lo) keeps the spout high.
#   (3) diagnostics only, after the sum: diag_src_axial_w, diag_src_n_contact, diag_rcv_n_contact.
# Unchanged on purpose: grasp, lift, placement target, tilt_open, tilt_dir (0.3 / 1.5), first-bead spill relief,
# every outcome weight, every speed / control / safety penalty and its gate.
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
    Used by place_rcv / aim_lip only; tilt_open has its own asymmetric bound (iter_22 repair 1)."""
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
      lip   : rim point the beads leave from; lip_xy / lip_dz relative to the receiver mouth.
    iter_23: lip_pt is still returned but no longer differentiated as a body point (change 1)."""
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
    height    : 0 -> 1 linearly over [0, z_lo] (iter_17 floor, unchanged), 1 up to the ceiling z_hi,
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
    du = d(u)/dt = w x u (assumption (가설): ang_vel in the world frame, as _point_vel already assumes).
    iter_23 change 1: also returns dlip_xy = d|lip_xy|/dt and dlip_dz = d(lip_dz)/dt. They are built from the
    source MOUTH velocity (an on-axis point), dtheta and dd_hat only, so spin about the cup's own axis gives 0."""
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
    return {"phi": phi, "phi_rate": phi_rate, "dtheta": dtheta, "wnd": wnd,
            "dlip_xy": dlip_xy, "dlip_dz": dlip_dz}


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
    # C placement (iter_22 verbatim): WORLD-FIXED height for the receiver mouth, horizontal vs the SPOUT
    RCV_POUR_LIFT = 0.15     # m, height allowance (the only absolute constant brief 5-3 permits). Mouth target
                             # = rim of a cup standing on the table + 0.15 m. With the spout window 0.7 r..4 r
                             # above that mouth, the source origin sits at lift ~0.23-0.34 m while pouring
                             # (70-90 deg), inside the lift_src paid range, so placement and lift_src do not fight.
    Z_TOL_R = 1.0            # +-1 r free band around the mouth height target
    PLACE_WIN_R = 1.0        # horizontal: receiver mouth pulled until the spout is within 1 r (= PERM_IN_R)
    BODY_CLEAR_R = 1.0       # until the source BOTTOM is BODY_CLEAR_R*r above the receiver RIM the pull stops at
                             # lip_xy = (PLACE_WIN_R + BODY_CLEAR_R) r = 2 r, so it never drags the receiver into
                             # the source body. The free radius shrinks linearly (slope 1) over that 1 r.
    PLACE_AZ = math.radians(68.0)       # K2 pourable side, kept as a GUARD only
    PLACE_AZ_TOL = math.radians(45.0)   # K1 palm-yaw box +-45 deg: 23..113 deg free, outside costs distance
    CENTER_FREE_R = 2.0      # |y| of the receiver mouth (= pour point) free up to 2 r (y ~= 0 centreline)
    W_PLACE = 1.0
    # D aim
    W_AIM = 1.5
    V_PROG = 0.15            # m/s: progress terms saturate at the stable cup speed limit (SRC_V_FREE)
    LIP_Z_HI_R = 0.7         # spout floor above the rim, 0.7 r ~= 2 cm (iter_17 permission height). With
                             # pour_sink gone (iter_23 change 2) this floor alone keeps the spout above the mouth.
    LIP_Z_MAX_R = 4.0        # spout CEILING 4 r ~= 11 cm above the rim (iter_22)
    LIP_Z_FADE_R = 2.0       # window fades out between 4 r and 6 r
    # E direction (iter_22 verbatim)
    W_TILT_DIR_OUT = 1.5     # outside the window: iter_19/21 strength
    W_TILT_DIR_IN = 0.3      # inside the window: a fifth, the spout is already over the mouth
    # tilt_open (iter_22 verbatim)
    W_TILT_OPEN = 0.1        # per step at most +0.1; whole-episode income bounded by
                             # W_TILT_OPEN * PHI_max / (TILT_RATE_SCALE * dt) ~= one bead (dt = 1/60 s, 가설)
    OPEN_BEADS = 1.0         # active until the first bead has LEFT the source
    OPEN_CHARGE_CAP = 100.0  # charges ungated, capped only at 100x the payment cap (physics glitches)
    # F outcome - unchanged weights (per-bead units with 20 beads)
    W_POUR = 150.0           # +7.5 per bead delivered
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0            # -2 per extra bead dumped in the same step
    W_SPILL = 240.0          # -12 per bead spilled
    SPILL_RELIEF = 0.25      # x0.25 (-3/bead) inside the AIMED window, first bead(s) only, at most one bead
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
    # diagnostics only (iter_23 change 3)
    CONTACT_F = 1.0          # N, a finger counts as touching its cup above this force (user: log 5-finger grasp)

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
    first_out = torch.clamp(OPEN_BEADS - n_beads * released, 0.0, 1.0)   # 1 until the first bead LEFT

    # receiver mouth velocity (an on-axis point, so its own axial spin adds nothing) and d(up)/dt of the source
    v_mouth = _point_vel(ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_pos, ctx.rcv_cup_mouth_pos)
    du_src = torch.cross(ctx.src_cup_ang_vel, ctx.src_cup_up, dim=-1)   # d(up)/dt (가설: world-frame ang_vel)

    # ------------------------------------------------------------------ spout window (a GATE, never a state income)
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    z_lo, z_hi = LIP_Z_HI_R * r, LIP_Z_MAX_R * r
    opn = _tilt_open_potential(ctx, geo, r, v_mouth, du_src, r_in, r_out, z_lo, z_hi, LIP_Z_FADE_R * r)
    # iter_23 change 1: the spout rates used by place / aim are the exact derivatives of the lip GEOMETRY
    # (the same ones the window and tilt_open already use). Axial spin of the source cup contributes exactly 0.
    dlip_xy = opn["dlip_xy"]                              # d|lip_xy|/dt, relative to the receiver mouth
    dlip_dz = opn["dlip_dz"]                              # d(lip_dz)/dt
    wnd = opn["wnd"]
    win = wnd["win"]                                      # position window (horizontal x height)
    perm_xy = wnd["w_xy"]
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = win * toward                                   # aimed window: position + leaning toward
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r) / ((LEAD_ZERO_R - LEAD_FULL_R) * r))

    # ------------------------------------------------------------------ C. placement (iter_22 target, iter_23 rate)
    # Receiver mouth: world-fixed height (not tied to the source), horizontally until the SPOUT is within
    # PLACE_WIN_R*r, |y| near the centreline, and not outside the pourable side of the source (guard).
    # The horizontal pull reaches PLACE_WIN_R*r only once the source BOTTOM centre is BODY_CLEAR_R*r above the
    # receiver RIM centre. Potential = norm of the excesses; paid as its exact rate -> 0 when parked.
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
    # Potential: horizontal spout->mouth distance + how far the spout is BELOW its floor + how far it is ABOVE
    # its ceiling. iter_23 change 1: its rate is the exact derivative of that potential,
    #   d d_aim/dt = (lip_xy * d|lip_xy|/dt + (above - below) * d(lip_dz)/dt) / d_aim,
    # so spinning the cup about its own axis (the lip stays put) pays nothing.
    below = torch.clamp(z_lo - geo["lip_dz"], min=0.0)
    above = torch.clamp(geo["lip_dz"] - z_hi, min=0.0)
    d_aim = torch.sqrt(geo["lip_xy"] ** 2 + below ** 2 + above ** 2)
    dd_aim = (geo["lip_xy"] * dlip_xy + (above - below) * dlip_dz) / torch.clamp(d_aim, min=1e-4)
    aim = stage_cd * _progress(-dd_aim, V_PROG)

    # ------------------------------------------------------------------ tilt_open (iter_22 verbatim)
    # PHI = win * theta+ * max(cos psi, 0); phi_rate is its exact derivative. Per step:
    #   first_out * ( carry_both * sep_ok * clamp(rate/scale, 0, +1) + clamp(rate/scale, -OPEN_CHARGE_CAP, 0) )
    # Payments <= rate and charges = rate, so every closed cycle sums to <= 0 while first_out is constant.
    rate_n = opn["phi_rate"] / TILT_RATE_SCALE
    open_pay = carry_both * geo["sep_ok"] * torch.clamp(rate_n, 0.0, 1.0)
    open_charge = torch.clamp(rate_n, -OPEN_CHARGE_CAP, 0.0)
    tilt_open = first_out * (open_pay + open_charge)

    # ------------------------------------------------------------------ E. direction (cost only, iter_22 verbatim)
    dir_err, psi = _tilt_direction_error(ctx, geo["d_hat"])
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)          # magnitude only as a multiplier, no target
    dir_w = W_TILT_DIR_IN * win + W_TILT_DIR_OUT * (1.0 - win)   # strong only outside the window
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err * dir_w

    # ------------------------------------------------------------------ H. removed (iter_23 change 2)
    # pour_sink is no longer summed: it paid for vertical velocity readings without height change and its zone
    # parked the pour near 70 deg. The spout floor of the aim potential / window keeps the height instead.
    h_org = ctx.src_cup_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]   # kept for the existing diagnostic only

    # ------------------------------------------------------------------ F. outcome (the beads, iter_22 verbatim)
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    # A spilled bead has necessarily LEFT the source; the relief gate uses the count WITHOUT this step's spill and
    # relieves at most one bead's worth: x0.25 (-3 instead of -12, still a loss vs +7.5 delivered).
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
        # C / D / E / open: difference forms and a cost - none of them pays for standing still or for axial spin
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": -tilt_dir,
        "tilt_open": W_TILT_OPEN * tilt_open,
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
    # iter_23 change 3: axial spin reading of the source cup and the number of fingers touching each own cup
    comps["diag_src_axial_w"] = (ctx.src_cup_ang_vel * ctx.src_cup_up).sum(dim=-1)
    comps["diag_src_n_contact"] = (ctx.src_finger_force > CONTACT_F).to(dtype).sum(dim=-1)
    comps["diag_rcv_n_contact"] = (ctx.rcv_finger_force > CONTACT_F).to(dtype).sum(dim=-1)
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0.000122, 8.54e-05, 7.32e-05, 4.88e-05, 6.1e-05, 6.1e-05, 3.66e-05, 6.1e-05, 0]  min 0 · mean 6.48e-05 · max 0.00022
bead/spill: [0.00011, 0.0164, 0.0129, 0.018, 0.018, 0.0153, 0.0136, 0.013, 0.0102, 0.00734]  min 0.00011 · mean 0.0132 · max 0.0245
done/drop: [0.000244, 0.000244, 0.000244, 0.000244, 0.000244, 0, 0.000244, 0.000488, 0.000732, 0.000244]  min 0 · mean 0.00024 · max 0.00122
done/mimic_runaway: [0, 0, 0.000244, 0, 0, 0, 0, 0.000244, 0.000244, 0]  min 0 · mean 0.000112 · max 0.000977
episode_lengths/step: [95.7, 790, 790, 781, 761, 802, 797, 815, 842, 853]  min 95.7 · mean 789 · max 887
reward/act_flip: [-0.662, -0.396, -0.275, -0.215, -0.175, -0.144, -0.125, -0.111, -0.0958, -0.081]  min -0.662 · mean -0.188 · max -0.0682
reward/aim_lip: [0, 0.0123, 0.0139, 0.00154, 0.0362, 0.0488, 0.0566, 0.0752, 0.121, 0.136]  min -0.225 · mean 0.0607 · max 0.169
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/arm_posture: [-0.255, -0.36, -0.391, -0.389, -0.406, -0.395, -0.398, -0.404, -0.411, -0.399]  min -0.427 · mean -0.396 · max -0.255
reward/cmd_rate: [-0.417, -0.277, -0.215, -0.177, -0.151, -0.132, -0.12, -0.107, -0.096, -0.0862]  min -0.417 · mean -0.155 · max -0.0761
reward/cup_disturb: [-0.319, -0.112, -0.0996, -0.089, -0.0913, -0.0942, -0.0733, -0.072, -0.0594, -0.0586]  min -0.609 · mean -0.0831 · max -0.0336
reward/diag_aim_dist_r: [10.4, 6.42, 8.27, 5.68, 6.34, 8.21, 5.82, 5.29, 5.35, 4.78]  min 4.66 · mean 6.95 · max 17.1
reward/diag_balance_src: [0, 0.61, 0.629, 0.627, 0.615, 0.625, 0.677, 0.671, 0.711, 0.734]  min 0 · mean 0.656 · max 0.756
reward/diag_body_clear_r: [-4.03, -0.155, 0.0496, 0.0609, -0.821, 0.136, -0.89, -0.645, 0.321, 0.189]  min -4.13 · mean -0.177 · max 3.74
reward/diag_clean: [1, 0.963, 0.971, 0.961, 0.959, 0.966, 0.97, 0.971, 0.977, 0.983]  min 0.946 · mean 0.971 · max 1
reward/diag_ctrl_gate: [0, 0.537, 0.56, 0.548, 0.584, 0.582, 0.669, 0.712, 0.769, 0.75]  min 0 · mean 0.645 · max 0.824
reward/diag_dir_err_deg: [84.1, 74.9, 40.3, 37.2, 40.8, 45.2, 49, 57.4, 66.1, 60.3]  min 34.9 · mean 53.9 · max 99.5
reward/diag_finish_q: [0, 0.000173, 0.000111, 0.000243, 4.79e-05, 0.000184, 6.51e-05, 0, 0.0002, 0]  min 0 · mean 0.000117 · max 0.000501
reward/diag_first_out: [0.999, 0.936, 0.944, 0.931, 0.931, 0.939, 0.95, 0.949, 0.958, 0.971]  min 0.905 · mean 0.949 · max 0.999
reward/diag_first_spill: [0.999, 0.936, 0.944, 0.932, 0.931, 0.939, 0.95, 0.95, 0.958, 0.971]  min 0.905 · mean 0.949 · max 0.999
reward/diag_grip_keep_src: [0.212, 0.442, 0.46, 0.38, 0.393, 0.391, 0.374, 0.357, 0.337, 0.42]  min 0.212 · mean 0.392 · max 0.48
reward/diag_h_org: [-0.0534, 0.0519, 0.0566, 0.056, 0.0319, 0.06, 0.0315, 0.0378, 0.0648, 0.0615]  min -0.0574 · mean 0.0506 · max 0.161
reward/diag_lift_budget: [0.4, 0.405, 0.41, 0.412, 0.41, 0.408, 0.407, 0.407, 0.406, 0.406]  min 0.4 · mean 0.408 · max 0.413
reward/diag_lip_dz: [0.000288, 0.101, 0.0993, 0.0969, 0.0746, 0.105, 0.0773, 0.0834, 0.111, 0.108]  min -0.00776 · mean 0.0956 · max 0.207
reward/diag_lip_dz_r: [0.0103, 3.59, 3.54, 3.45, 2.66, 3.74, 2.75, 2.97, 3.96, 3.84]  min -0.276 · mean 3.41 · max 7.37
reward/diag_lip_xy: [0.292, 0.172, 0.225, 0.152, 0.161, 0.195, 0.157, 0.142, 0.137, 0.127]  min 0.124 · mean 0.184 · max 0.452
reward/diag_open_phi: [0, 0.00161, 0.0181, 0.0175, 0.0102, 0.00786, 0.0291, 0.0277, 0.0273, 0.0335]  min 0 · mean 0.0165 · max 0.0434
reward/diag_open_rate: [0, -0.00339, -0.0092, -0.0255, -0.0137, -0.0134, -0.0439, -0.0199, -0.0317, -0.0352]  min -0.215 · mean -0.0176 · max 0.573
reward/diag_outcome_income: [0, 8.87e-05, 0.000122, 6.66e-05, 5.13e-05, 9.71e-05, 0.000214, 0.000249, 0, 0]  min 0 · mean 0.000104 · max 0.000529
reward/diag_palm_sat_src: [0.447, 0.416, 0.301, 0.258, 0.227, 0.209, 0.174, 0.15, 0.147, 0.146]  min 0.136 · mean 0.238 · max 0.586
reward/diag_perm: [0, 0.0146, 0.0754, 0.0816, 0.0485, 0.0597, 0.172, 0.156, 0.228, 0.274]  min 0 · mean 0.11 · max 0.301
reward/diag_place_az_deg: [90.1, 69.2, 88.2, 80.9, 79.1, 74.6, 79.7, 90.5, 98.3, 102]  min 36.3 · mean 84.8 · max 111
reward/diag_place_dist_r: [10.2, 7.21, 8.55, 5.86, 5.72, 7.25, 5.52, 4.47, 4.47, 4.05]  min 3.85 · mean 6.73 · max 18.3
reward/diag_place_free_r: [2, 1.46, 1.53, 1.46, 1.73, 1.8, 1.73, 1.71, 1.57, 1.6]  min 1.36 · mean 1.63 · max 2
reward/diag_rcv_mouth_dz_r: [-5.34, -3.49, -3.44, -3.37, -3.2, -3.18, -3.04, -2.6, -2.14, -2.44]  min -5.34 · mean -2.83 · max -0.386
reward/diag_rcv_mouth_y: [0.161, 0.168, 0.136, 0.129, 0.112, 0.107, 0.0862, 0.076, 0.0657, 0.0673]  min 0.0231 · mean 0.108 · max 0.239
reward/diag_rcv_n_contact: [0.0103, 2.37, 2.72, 2.88, 3.07, 3.11, 3.12, 2.98, 3.19, 3.38]  min 0.0103 · mean 2.99 · max 3.56
reward/diag_release_beads: [0.000732, 0.00439, 0.00439, 0.00269, 0.00708, 0.00513, 0.00244, 0.0061, 0.00171, 0.00244]  min 0 · mean 0.00331 · max 0.00952
reward/diag_relieved_beads: [0, 5.58e-09, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.3e-06 · max 0.000244
reward/diag_side_excess_r: [0, 0.212, 0.449, 0.0237, 0.0137, 1.1, 0.353, 0.11, 0.388, 0.219]  min 0 · mean 0.671 · max 7.49
reward/diag_src_axial_w: [-0.0116, -0.0201, -0.174, -0.159, -0.179, -0.141, -0.0939, -0.0345, 0.136, 0.0607]  min -222 · mean -0.764 · max 18.7
reward/diag_src_margin_min: [0.194, 0.116, 0.109, 0.106, 0.0984, 0.096, 0.0907, 0.0914, 0.0898, 0.0879]  min 0.0841 · mean 0.0984 · max 0.194
reward/diag_src_n_contact: [0.0354, 2.01, 2.12, 2.17, 2.15, 2.13, 2.23, 2.28, 2.36, 2.37]  min 0.0354 · mean 2.21 · max 2.5
reward/diag_src_tilt_deg: [0.935, 15.1, 18.8, 20.7, 18.6, 15.9, 15.4, 16.5, 17.6, 16.8]  min 0.935 · mean 17.5 · max 22.3
reward/diag_src_v: [0.0411, 0.22, 0.213, 0.171, 0.189, 0.197, 0.148, 0.154, 0.155, 0.131]  min 0.0411 · mean 0.206 · max 5.57
reward/diag_src_w: [0.489, 1.29, 1.28, 1.29, 1.29, 1.13, 1.07, 1.21, 1.19, 1.06]  min 0.489 · mean 2.17 · max 238
reward/diag_stage_cd: [0, 0.601, 0.635, 0.633, 0.653, 0.654, 0.722, 0.723, 0.776, 0.755]  min 0 · mean 0.69 · max 0.839
reward/diag_state_income: [0.461, 2.79, 2.9, 2.91, 2.84, 2.86, 3.05, 3.07, 3.25, 3.26]  min 0.461 · mean 3.01 · max 3.45
reward/diag_theta_rate: [0.102, -0.162, 0.0727, -0.0427, -0.0275, -0.101, -0.0897, -0.035, -0.031, -0.0908]  min -59.6 · mean 0.0784 · max 67
reward/diag_toward: [0.547, 0.618, 0.893, 0.91, 0.894, 0.865, 0.863, 0.829, 0.794, 0.86]  min 0.393 · mean 0.829 · max 0.923
reward/diag_unaimed_beads: [0.000732, 0.00439, 0.00439, 0.00269, 0.00708, 0.00513, 0.00244, 0.0061, 0.00171, 0.00244]  min 0 · mean 0.00327 · max 0.00952
reward/diag_win: [0, 0.0238, 0.0762, 0.0819, 0.0497, 0.0641, 0.186, 0.174, 0.274, 0.29]  min 0 · mean 0.121 · max 0.317
reward/grasp_rcv: [0.218, 0.753, 0.783, 0.808, 0.803, 0.82, 0.851, 0.807, 0.813, 0.793]  min 0.218 · mean 0.802 · max 0.86
reward/grasp_src: [0.244, 0.917, 0.962, 0.923, 0.922, 0.924, 0.964, 0.965, 1, 1.04]  min 0.244 · mean 0.961 · max 1.12
reward/lift_rcv: [0, 0.284, 0.296, 0.294, 0.309, 0.309, 0.337, 0.353, 0.37, 0.369]  min 0 · mean 0.329 · max 0.399
reward/lift_src: [-0.000366, 0.833, 0.856, 0.883, 0.808, 0.803, 0.9, 0.946, 1.06, 1.06]  min -0.000366 · mean 0.914 · max 1.15
reward/palm_bang: [-0.844, -0.585, -0.493, -0.404, -0.356, -0.268, -0.225, -0.218, -0.194, -0.176]  min -0.87 · mean -0.332 · max -0.145
reward/palm_sat: [-2.15, -1.89, -1.42, -1.27, -1.1, -1.04, -0.918, -0.822, -0.795, -0.76]  min -2.41 · mean -1.16 · max -0.731
reward/place_rcv: [0, 0.0164, 0.0407, 0.0393, 0.0588, 0.0771, 0.0841, 0.0821, 0.116, 0.12]  min -0.213 · mean 0.0696 · max 0.139
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0.00183, -0.00183, -0.00183, 0]  min -0.00366 · mean 6.33e-05 · max 0.00183
reward/rcv_steady: [-0.00431, -0.196, -0.19, -0.18, -0.182, -0.17, -0.191, -0.207, -0.238, -0.208]  min -0.27 · mean -0.197 · max -0.00431
reward/safety: [-0.0179, -0.00593, -0.0102, -0.00675, -0.0134, -0.00493, -0.00274, -0.00553, -0.00292, -0.00103]  min -0.0183 · mean -0.00495 · max -7.46e-05
reward/spill_delta: [-0.00879, -0.0527, -0.0527, -0.0322, -0.085, -0.0615, -0.0264, -0.0732, -0.0205, -0.0293]  min -0.114 · mean -0.0395 · max 0
reward/src_speed: [0, -0.0755, -0.0701, -0.0429, -0.0372, -0.0251, -0.0294, -0.0361, -0.0372, -0.0267]  min -0.261 · mean -0.0439 · max 0
reward/src_spin: [0, -0.0296, -0.0314, -0.028, -0.0258, -0.022, -0.0286, -0.0358, -0.0422, -0.0284]  min -0.0564 · mean -0.03 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_dir: [0, -0.0665, -0.0176, -0.0159, -0.02, -0.0194, -0.0241, -0.0363, -0.0554, -0.0438]  min -0.108 · mean -0.038 · max 0
reward/tilt_open: [0, -0.000353, -0.00276, -0.00332, -0.00213, -0.0021, -0.00505, -0.00297, -0.00355, -0.00393]  min -0.00671 · mean -0.0026 · max 5.8e-05
reward/total: [-3.22, -0.2, 0.693, 1.11, 1.28, 1.59, 2.02, 2.07, 2.42, 2.59]  min -3.28 · mean 1.47 · max 2.86
reward/transfer_tail: [0, 1.58e-06, 0, 0, 0, 0, 6.21e-06, 0, 0, 0]  min 0 · mean 5.74e-06 · max 0.000179
reward/transferred: [0, 8.71e-05, 0.000122, 6.66e-05, 5.13e-05, 9.71e-05, 0.000208, 0.000249, 0, 0]  min 0 · mean 9.78e-05 · max 0.000516
reward/unaimed_release: [-0.0022, -0.0132, -0.0132, -0.00806, -0.0212, -0.0154, -0.00732, -0.0183, -0.00513, -0.00732]  min -0.0286 · mean -0.00982 · max 0
rewards/step: [-246, -175, 495, 878, 973, 1.33e+03, 1.55e+03, 1.78e+03, 2e+03, 2.22e+03]  min -886 · mean 1.17e+03 · max 2.51e+03
task/aim_dist: [0.321, 0.247, 0.297, 0.23, 0.231, 0.283, 0.225, 0.21, 0.224, 0.208]  min 0.193 · mean 0.258 · max 0.514
task/cup_collision_rate: [0, 0.000244, 0.000244, 0.000244, 0.000732, 0, 0.000488, 0.000244, 0.000488, 0.000488]  min 0 · mean 0.000409 · max 0.00391
task/cups_center_dist: [0.321, 0.249, 0.307, 0.242, 0.24, 0.291, 0.233, 0.216, 0.228, 0.214]  min 0.202 · mean 0.265 · max 0.513
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.000977, 0.00415, 0.00171, 0.00562, 0.00171, 0.000732, 0.00171, 0.000488, 0]  min 0 · mean 0.00139 · max 0.00659
task/rcv_cup_lift: [7.25e-05, 0.0526, 0.0539, 0.0558, 0.0608, 0.0612, 0.065, 0.0774, 0.0905, 0.0817]  min 7.25e-05 · mean 0.071 · max 0.14
task/rcv_grasped: [0, 0.768, 0.791, 0.796, 0.813, 0.814, 0.832, 0.808, 0.829, 0.808]  min 0 · mean 0.809 · max 0.9
task/rcv_hand_foreign_rate: [0.051, 0.0107, 0.00464, 0.0061, 0.00293, 0.00342, 0.000977, 0.00464, 0.0022, 0.000977]  min 0 · mean 0.00447 · max 0.0593
task/src_cup_lift: [0.000588, 0.158, 0.164, 0.165, 0.146, 0.175, 0.15, 0.169, 0.209, 0.197]  min 0.000588 · mean 0.175 · max 0.308
task/src_grasped: [0, 0.712, 0.733, 0.725, 0.72, 0.721, 0.769, 0.768, 0.812, 0.814]  min 0 · mean 0.757 · max 0.919
task/src_hand_foreign_rate: [0.000977, 0.00122, 0.000732, 0.00342, 0.00195, 0.000977, 0.00122, 0.000977, 0.00244, 0.000977]  min 0 · mean 0.00137 · max 0.00391
task/src_tilt_deg: [0.935, 15.1, 18.8, 20.7, 18.6, 15.9, 15.4, 16.5, 17.6, 16.8]  min 0.935 · mean 17.5 · max 22.3
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


# round 24 operator notes: i23 는 두 컵을 붙여 두고(주둥이 수평 1.9 r) 기울이지 않는다. 기울이면 소스 파지가 끊겼고, 첫 비드 전 기울이기 신호는 속도 판독 때문에 세금이 됐다

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. 수치 출처: 결정론 trace(64 env · 900 스텝 · ADR 0) ep300·400·500·600,
학습 곡선(t2r_rh3_i23 summaries), `iter_23/compute_reward.py`. r = `ctx.cup_radius` = 0.028 m. 제어 주기 dt = 1/60 s(trace meta dt_env).

## i23 결과 (t2r_rh3_i23, 파지 단계 ckpt, ep600 에서 라운드 끝: 3 회 연속 배달 0 · 기울기 후퇴)

| ep | env별 최대 기울기 중앙 | >60° env | 후반* 주둥이 수평 / 높이 | 후반 기울기 / 방향 오차 | 후반 소스 기우는 방위 / 리시버 방위 | 떠난 비드 → 들어감 | env 에피소드 합 중앙 |
|---|---|---|---|---|---|---|---|
| 300 | 45.6° | 11/64 | 4.5 r / 3.0 r | 11° / 40° | +63° / +76° | 233 → 4 | 2595 |
| 400 | 38.5° | 4/64 | 2.1 r / 2.7 r | 15° / 41° | +28° / +65° | 110 → 0 | 2803 |
| 500 | 37.9° | 0/64 | 2.0 r / 4.0 r | 21° / 81° | +21° / +96° | 18 → 0 | 2905 |
| 600 | 40.1° | 0/64 | 1.85 r / 4.5 r | 23° / 89° | +22° / +107° | 0 → 0 | 2994 |

*후반 = t ≥ 450 스텝의 env별 중앙의 중앙. 주둥이 = 보상의 lip(`diag_lip_xy`, `diag_lip_dz_r`).

- **컵은 떨어져 있지 않다.** 요약기의 "수평 29 cm · 소스 y −0.15 / 리시버 y +0.16" 은 기울기 > 45° 스텝(0.1 ~ 0.3 %)만의 값이다.
  후반 전 스텝으로는 ep400 부터 55 ~ 62/64 env 가 주둥이 수평 < 3 r, ep600 배치 거리 0.87 r, 소스 y −0.02 · 리시버 y +0.05(중심선).
- **주둥이가 천장(4 r) 위에 있다.** ep600 후반 4.5 r(50/64 env 가 > 4 r). 소스 원점 z 0.415 → 0.532 m, lift_src 스텝당 1.0 → 1.38.
  학습 `diag_body_clear_r` −1.6 → +1.1 r. 직립 소스는 바닥이 리시버 림 높이일 때 이미 주둥이가 4.05 r 위다(입구 − 바닥 0.114 m).
  받는 쪽으로 90° 기울이면 주둥이가 r + mouth_z = 2.9 r 내려가고 수평으로 0.9 r 다가간다. 원점이 그대로면 지금 자리에서 약 1.7 r 위 · 약 1 r 로
  창 안이다(lip 식으로 계산).
- **방향이 벌어졌다.** 파지 · 기울기 > 30° 스텝의 기우는 방위: ep300 45 ~ 90° 가 88 % → ep600 0 ~ 45° 가 96 %.
  리시버 방위: ep300 45 ~ 90° 95 % → ep600 90° 이상 91 %, 배치 가드 113° 밖 28 %. 학습 `diag_place_az_deg` 54 → 103°, `diag_side_excess_r` 0.43 → 1.01 r.
  같은 기간 파지 · 기울기 > 25° 스텝의 소스 palm pitch 명령 중앙 +0.82 → +0.40, roll −0.78 → −0.65(|a| > 0.7 이 palm_sat 구간).
  기우는 방위 회귀: pitch +49°/단위, roll −72°/단위(R² 0.44).
  리시버는 목표 높이로 오르며(z 0.32 → 0.40) 뒤로(x 0.40 → 0.31) 갔다.
- 소스 팔: ep600 후반 r_aj_2 가 한계까지 범위의 5 % 안에 있는 스텝 100 %, r_aj_7 은 87 %. palm y 명령 +0.94(> 0.7 이 98 %)인데 palm y 는 −0.04 에 머문다.

기울기와 파지:

- **ep300·400 에서 최대 기울기 > 60° 인 env 15 개 전부 소스 파지를 잃었다.** 60° 를 넘은 뒤 120 스텝 파지 평균 0.00 ~ 0.18(1 개는 0.75 뒤 끝에 0).
  이 env 들의 에피소드 합 1348 대 나머지 2478(ep300), 1476 대 2743(ep400). 차이 −1130 은 상태 수입 −624, 흘림 · 조준 밖 −226, 제어 · 교란 약 −350(palm_sat −254), aim+place +71.
- 떠난 비드(ep300 ~ 500) 344 개 중 337 개가 회전 급등(±3 스텝 |ω| > 5 rad/s) 또는 파지 끊김 중에 나갔고 그중 5 개가 들어갔다.
  조용한 붓기는 7 개(ep400 1 env, 주둥이 5.7 r 위 · 수평 3.0 r, 창 밖) → 0 개. **i23 에는 조종된 붓기가 한 번도 없다**(i22 ep250 은 17 회 → 11 개).
- 긴 끊김(≥ 30 스텝) 26/8/2/0 회. 끊긴 순간 기울기 중앙 40° · 45°(ep300 · 400), 엄지 힘 ≤ 1 N 이 77 ~ 100 %.
  예: ep300 env51 은 44 → 76° 동안 엄지 벌림 명령 1.0 → 0.02, 엄지 힘 4.8 → 0.6 N, 약지 · 새끼 29 · 48 N, 이어 중지 179 N 로 컵이 튕겨 나갔다.
  env32 는 33 → 57° 를 약 25 스텝에 천천히 기울였고 엄지 1.2 ~ 2.5 N 로 버티다 52 ~ 54° 에서 끊겼다.
- **소스는 3 지 파지다.** 파지 중 접촉 손가락 수 3 개 62 ~ 85 %, 4 개 11 ~ 29 %, 5 개 0 %. 약지 접촉률 0.12 ~ 0.31, 새끼 0.0. 리시버는 5 개 82 ~ 90 %.
- 파지 플래그 끊김(첫 들기 뒤) 717/567/433/245 회/64 env, 55 ~ 59 % 가 1 스텝. 접촉 손가락 수별로 2 스텝 뒤 끊겨 있을 확률:
  리시버 2 개 11.5 % · 3 개 2.8 % · 4 개 0.57 % · 5 개 0.30 %, 소스 2 개 6.2 % · 3 개 0.43 % · 4 개 0.79 %
  (소스 4 개 상태에는 기울 때 컵이 약지에 부딪힌 순간이 섞여 있다 [가설]).

스텝당 수입(생존 +1.0 제외, ep300 ~ 600 합침, trace 의 `term_*` 그대로):

| 상태 | 스텝 수 | 파지+들기 | aim+place | tilt_dir+open | 흘림·조준 밖 | 제어 G | 합 |
|---|---|---|---|---|---|---|---|
| S1 파지 · 떨어짐(주둥이 > 3 r, 기울기 < 45°) | 70045 | +3.80 | +0.48 | −0.06 | −0.01 | −1.39 | +2.81 |
| S2 파지 · 붙음(≤ 3 r, < 45°) = 지금 주차 | 107789 | +3.98 | −0.02 | −0.10 | 0.00 | −1.31 | +2.55 |
| S3 파지 · 받는 쪽으로 ≥ 45°(방향 < 45°) | 1571 | +3.72 | −0.12 | −0.02 | −0.18 | −1.67 | +1.72 |
| S4 들기 뒤 소스 파지 잃음 | 8946 | +1.00 | 0.00 | 0.00 | −0.48 | −1.61 | −1.36 |

S3 − S2 = −0.84/스텝: palm_sat −0.25, 흘림·조준 밖 −0.18, lift_src −0.13, lift_rcv −0.13, src_speed+spin −0.09, arm_posture −0.07, aim −0.07
(tilt_dir +0.07, rcv_steady +0.05). S4 는 교란(cup_disturb) −0.25 포함.

rate 항 지불 대 같은 퍼텐셜의 위치 차분(env 당 에피소드 합, 같은 무게 · 게이트 · clamp):

| ep | aim_lip 지불 / 차분 | place_rcv 지불 / 차분 | tilt_open 지불 / ΔΦ 차분 |
|---|---|---|---|
| 300 | 30 / 79 | 46 / 53 | −1.2 / +0.1 |
| 400 | 3 / 106 | 72 / 72 | −7.2 / 0.0 |
| 500 | 65 / 123 | 78 / 82 | −6.1 / 0.0 |
| 600 | 94 / 125 | 97 / 87 | −2.0 / 0.0 |

- 후반 주차(t ≥ 450)에서 aim 지불 −61(ep400) · −27(ep600) 대 차분 +8 · −0.5. 가만히 있는데 스텝당 −0.05 ~ −0.15 를 뗀다.
- 소스 컵 각속도 판독: 후반 파지 구간 θ 변화율(식이 쓰는 ω × up) 적분 −2.1 · −2.4 rad/env 대 실제 θ 변화 −0.09 · −0.04 rad(ep400 · 600).
  |ω × up| 평균 0.52 · 0.42 대 |d up/dt| 0.21 · 0.10 rad/s. 이 판독에 입구 팔 길이(mouth_z 0.054 m)를 곱하면 1.6 ~ 2.8 cm/s 로, aim 의 가짜 멀어짐과 크기가 맞는다.
- tilt_open 을 trace 필드로 iter_23 식 그대로 다시 계산하면 −1.22/−7.19/−6.13/−2.00(trace −1.21/−7.17/−6.13/−2.00).
  지급 · 청구에 같은 게이트와 대칭 clamp 를 써도 −0.9/−6.4/−5.7/−2.0 이다. **세금은 clamp 가 아니라 속도 판독에서 온다.**

붓기 한 번의 값(iter_23 결과 식 · 무게 그대로, 450 스텝부터 한 스텝에 한 알, 파지 유지, 계산):

| 결과 | 결과 수입 합 |
|---|---|
| 20 in / 0 흘림 | +14127 |
| 18 / 2 | +8262 |
| 16 / 4 | +2701 |
| 13 / 7 (i22 조용한 붓기 적중 65 %) | +49 |
| 8 / 12 | −64 |
| 기울이다 파지를 잃음(ep300 실측 에피소드 합 차이) | −1130 |

- 학습 곡선: reward/total −0.77 → +2.52(e1-50 → e580-620). 증가 +3.29 중 제어 · 교란 벌점 감소 +2.52(palm_sat +1.41, palm_bang +0.47,
  act_flip +0.37, cmd_rate +0.22), 상태 수입 +0.56, aim+place +0.23. tilt_open 은 e250 이후 모든 구간에서 음수(−0.002 ~ −0.004).
  ep/toward_max 49.0°(e250-300) → 35.9°, ep/dir_err_at_max 30 → 55°, ep/in_source_min 0.82 → 0.89, ep/spill_end 0.12 → 0.07.

## 원인 (코드: iter_23/compute_reward.py)

1. **[확인: trace] 기울이면 소스 파지가 끊겼고, 정책은 기울이기를 버렸다.** > 60° 15/15 끊김, 에피소드 −1130, 떠난 비드 98 % 가 사고.
   보상 쪽 사실: 파지 값 `held_val = 0.4 + 0.3·contact_q + 0.3·keep`(132 줄)은 엄지와 네 손가락 합의 min(pinch, 128 줄)과 균형(keep)만 본다.
   엄지 + 손가락 1 ~ 2 개로 최대치에 닿고, 닿은 손가락 수는 값에 없다.
   [가설] 약지 · 새끼가 컵을 잡지 않고 굽혀만 있어, 기울일 때 컵이 그쪽으로 돌며 엄지 접촉이 빠진다(env51 · env32).
2. **[확인: 코드+trace] 첫 비드 전에 "받는 쪽으로 기울이기" 가 받을 돈이 없다.** tilt_open 상한은 W·Φ/(scale·dt) = 0.1 · π · 60 = 19(90° 에서 9)이고,
   실제로는 에피소드당 −1 ~ −7 인 세금이었다(원인 3). 전이 비용 S3 − S2 = −0.84/스텝 × 23 → 60° 를 1 rad/s(ROT_POUR_FREE)로 ≥ 39 스텝 ≈ −33.
   붓기가 깨끗하면 +2700 ~ +14000 이지만 i23 에는 그 근처에 간 적이 없다(조용한 붓기 0).
3. **[확인: trace] rate 항 3 개가 움직임이 아니라 속도 판독에 돈을 준다.** aim_lip · place_rcv · tilt_open 의 변화율은 ctx 선 · 각속도
   (`root_lin_vel_w` · `root_ang_vel_w`, pour_fabric_env.py:602·606)로 만든 정확한 미분이다(212 ~ 256, 396 ~ 406, 441, 453, 460 줄).
   식은 맞지만 손에 든 컵의 판독이 실제의 2 ~ 4 배이고 치우쳐 있다. aim 은 접근에서 덜 받고(ep400 3 대 106) 주차에서 뗀다(−27 ~ −61),
   tilt_open 은 기울기 신호가 세금이 된다. i20(비대칭 clamp) · i22(축 회전 · pour_sink) · i23(각속도 헛읽음)으로 같은 계열이 네 라운드째다.
4. **[확인: trace, 기구는 가설] 방향이 벌어졌다.** palm_sat 구간 명령(pitch 0.82)을 줄이면서 기우는 방위가 71 → 25° 로 돌았고,
   리시버는 높이 목표로 오르며 뒤로 가서 107° 에 섰다. 배치 가드(PLACE_AZ 68 ± 45°)는 차분 퍼텐셜이라 붙잡는 힘이 없다.
   [가설] 오른팔은 r_aj_2 한계에 붙어 +y 로 못 가고, 왼팔은 목표 높이에서 뒤쪽 자세가 편하다. palm_sat 는 제어 벌점이라 풀지 않는다.
   이번에는 바꾸지 않는다(기구가 가설 단계). 변경 1 · 3 뒤에는 방향을 받는 쪽으로 돌리는 것이 tilt_open 으로 정직하게 지급된다.
5. 이번 원인이 아닌 것: 주둥이 과고도(직립 소스의 몸통 여유 1 r + 컵 높이 4.05 r 때문이고, 받는 쪽으로 기울이면 창 안으로 내려온다),
   창 폭 · 천장 · 배치 목표 높이, 흘림 무게(기울이지 않아 흘림이 0 으로 줄었다).
6. 메인 가설 판정: "총보상이 상태 수입과 벌점 감소로 컸다" 확인(+3.29 중 +3.08). "붓기 쪽 항이 10 배 작다" 확인(aim+place 0.24 대 상태 3.28/스텝).
   "안전하게 들고 기울이지 않는 쪽으로 수렴" 확인, 이유는 원인 1 · 2. "pour_sink 제거가 컵을 모으고 기울이던 마지막 유인을 없앴다" 는
   절반: pour_sink 가 빠진 뒤 소스가 12 cm 올랐고(연결은 [가설]. lift_src 는 h 0.30 m 까지 cm 당 스텝 +0.025 를 준다) 기울기 수입도 사라졌지만,
   컵은 모였다. "결정론에서 컵이 29 cm 벌어진다" 는 반박(요약기의 기울기 > 45° 부분집합).

## iter_24 변경 (이것만, 나머지 설계는 iter_23 그대로)

1. **aim_lip · place_rcv · tilt_open 의 변화율을 직전 제어 스텝과의 위치 차분으로.** 퍼텐셜 식은 그대로 둔다:
   d_aim(452 줄), d_place(440 줄), PHI = win · θ+ · max(cos ψ, 0)(253 줄).
   - 모듈 수준 캐시(dict)에 env 별 직전 값 d_aim · d_place · PHI · episode_progress 를 둔다(detach · clone, import 는 torch · math 만).
     rate = (이번 − 직전) / CONTROL_DT, CONTROL_DT = 1/60 s(trace meta dt_env 0.016667 = sim dt 1/120 × decimation 2).
   - 차분을 0 으로 두는 경우: 첫 호출, env 수 · 장치가 바뀐 호출(validator 드라이런), episode_progress 가 직전보다 작아진 env(새 에피소드).
     캐시는 게이트와 무관하게 매 호출 갱신한다. 떨어뜨린 컵이 스폰 자리로 되돌아가는 스텝의 퍼텐셜 튐은 clamp 가 한 스텝으로 막는다.
   - 무게 · 게이트 · clamp 그대로: aim · place = stage_cd · clamp(−Δd / (V_PROG · dt), −1, +1),
     tilt_open = first_out · (carry_both · sep_ok · clamp(ΔPHI / (scale · dt), 0, +1) + clamp(ΔPHI / (scale · dt), −100, 0)).
   - ctx 선 · 각속도로 만든 미분(v_mouth, du_src, dd_hat, dlip_xy · dlip_dz, dwin, phi_rate, dclear, dr_free, des_lo · des_hi)은 보상 합에서 빠진다.
     `diag_open_rate` · `diag_theta_rate` 는 차분 값으로 바꾼다. 제어 · 안전 벌점의 속도 사용(src_speed · src_spin · rcv_steady · fling)은 그대로.
2. **파지 값에 자기 컵에 닿은 손가락 수(사용자 결정 Q9: K=5, 로깅은 i23 에서 끝).** `_hand_cup` 의 held_val 을
   `0.4 + 0.2·contact_q + 0.2·keep + 0.2·(n_on / 4)` 로. n_on = 엄지를 뺀 네 손가락 중 힘 > 1.0 N 인 수
   (env 파지 플래그 · `diag_*_n_contact` 와 같은 임계). 두 손 같은 함수. 최대 1.0, 잡힘 ≥ 0.4 > 안 잡힘 ≤ 0.39 사다리, W_GRASP 1.5 그대로
   → 상태 수입 상한 불변. 3 지(n_on 2)는 0.1 낮다(× 1.5 = 스텝당 0.15), 5 지가 최대. free_val 은 그대로.
3. **W_TILT_OPEN 0.1 → 0.5(변경 1 이 있어야 의미가 있다).** 에피소드 상한 0.5 · π · 60 = 94(90° 에서 47), 60° 에서 약 31 로 전이 비용 −33 과 비슷하다.
   비드 20 개 pour_delta 150 과 깨끗한 붓기 +8262 보다 작다. 첫 비드가 나가면 꺼지고, 위치 차분에서는 지급 상한 +1 · 청구 무게이트라
   닫힌 순환 합 ≤ 0 이다. 기울인 채 멈추면 더 받을 것이 없고 되돌리면 청구되므로, 창 안에서 기울인 뒤 남는 수입은 비드뿐이다.

지키는 것(사용자 원칙): 각도 처방 금지(각도 목표 · 대역 · 상한 없음. PHI 는 목표 없는 단조 퍼텐셜) · 속도 · 제어 · 안전 벌점(G · safety 전부)
값과 게이트 그대로(palm_sat 포함) · 창은 `ctx.cup_radius` 배수(허가 1 ~ 3 r, 높이 바닥 0.7 r · 천장 4 r · 소멸 6 r, 배치 창 · 몸통 여유 1 r 그대로) ·
y ≈ 0 중심선(CENTER_FREE_R 2 r) · 예산 규칙(브리프 5-2, 상태 수입 < 결과 수입: 상태 수입 상한 불변, tilt_open 은 차분형) ·
aim · place 대칭 clamp(±1) · tilt_dir(0.3 / 1.5) · 첫 비드 흘림 완화 · 배치 목표 · 들기 · 결과 무게는 iter_23 그대로 ·
reward-audit 는 쓰지 않는다(t2r) · 변경은 위 3 개뿐.

사용자 성공 정의(09-25 결정. 판정 쪽 결정이고 이번 보상 변경은 변경 2 만 관련): in_target ≥ 0.80 · spill ≤ 0.25 · 20 스텝 유지 ·
받는 컵 기울기 ≤ 20° · 5지 파지 K=5 · 첫 들기 뒤 파지 끊김 M=0.
참고: 로깅 결과 소스 5지 0 %, 파지 플래그 끊김 ep600 245 회/64 env(55 % 가 1 스텝). 끊김 0 판정의 깜빡임 처리는 사용자 결정 대기(d_rh_grasp_flicker).

## 웜스타트 (메인 결정용)

- **권고: 파지 단계 ckpt rh2_i07-r3 ep400**(서버 `~/rl_ws/ckpt_in/pour_bi_rh/grasp_stage_i07r3_ep400.pth`, TaskSpec REQ-grasp-stage-warmstart, i21 ~ i23 과 같음).
  변경 2 가 파지 사다리를 바꾸므로 파지부터 다시 배우는 시작점이 맞다.
- 선택지(사용자 승인 필요): i23 ep300 = 서버 `~/rl_ws/ckpt_in/pour_bi_rh/rh3_i23_ep300.pth`. 받는 쪽으로 기우는 자세가 살아 있던 시점이다
  (>45° 순간 기우는 방위 +70° 대 리시버 +65°, >60° 11/64, 리시버가 붓기 쪽 65 ~ 76°). 그러나 그 11 개는 모두 파지를 잃었고(3 지 파지를 물려받는다),
  주둥이 수평 4.5 r 로 접근이 덜 됐으며, 붓기 습관을 가진 ckpt 라 TaskSpec 이 금지한다. i22 ep250 안(d_rh_warm_ep250)도 아직 결정 대기다.
