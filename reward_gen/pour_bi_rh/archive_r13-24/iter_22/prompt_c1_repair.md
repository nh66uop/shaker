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
# iter_20 - REDESIGN (operator brief 09.25, sections 0-7). 25 summed terms, one role each, staged A-H.
#   A grasp      : grasp_src, grasp_rcv                                   (small state)
#   B lift       : lift_src, lift_rcv                                     (small state)
#   C placement  : place_rcv        (source-relative target, difference form)
#   D aim        : aim_lip          (3D spout -> above receiver mouth, difference form)
#   E direction  : tilt_dir         (cost = tilt magnitude x direction error; no angle target)
#   H height     : pour_sink        (difference form on the source origin height, iter_19 verbatim)
#   F outcome    : pour_delta, transferred, transfer_tail, spill_delta, unaimed_release, success_bonus
#   G control    : src_speed, src_spin, rcv_steady, act_flip, palm_bang, palm_sat, cmd_rate, arm_posture
#   safety       : cup_disturb, safety
#   constant     : alive
# Removed on purpose: meet_cross / src_cross (spawn-relative rendezvous), converge, carry_src,
# align_mouths, hold_rcv, rcv_home, pour_dir_pen, settle_finish - all STATE income for a pose (brief 4-4,
# 5-2) or spawn-relative (brief 4-1, 5-3). No tilt target, band, release ladder or angle ceiling exists.
# =====================================================================================================


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _smoothstep(x):
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _late_ramp(x, lo, hi):
    """0.5*t + 0.5*t^2: never flat above lo, last part worth 3x the first."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _progress(rate_closing, v_scale):
    """Difference-form payment from an exact potential rate [m/s], positive = getting closer.
    ASYMMETRIC clamp: closing pays at most +1/step (reached at the stable-speed limit v_scale), but
    opening is charged up to -4/step, so "back off fast, come back slowly" can never net income."""
    # iter_21: symmetric (was -4..+1). Under exploration noise an asymmetric clamp makes jitter itself a net
    # loss (i20: grasp 0.69 -> 0.01 in 100 epochs while episode reward rose -384 -> +712).
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
    """Per-arm stage-A value in [0, 1] plus the table-disturbance penalties.
    Reach / pinch / close / grip_keep formulas are the validated iter_04..19 ones; only the way they
    are combined is new: ONE monotone ladder, grasped (>= 0.4) always above not-grasped (<= 0.39)."""
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
    # keep the grasp that lifted the cup: RATIO of the two contact sides (no absolute force - i24 rule)
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
    """Directed-spout geometry, iter_08/09 VERBATIM (minus the old d_align).
      d_hat : horizontal unit vector source cup origin -> receiver mouth
      theta : SIGNED tilt toward the receiver (used to LOCATE the lip only, never as a target)
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


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ------------------------------------------------------------------ constants and their sources
    W_ALIVE = 1.0            # constant survival floor, identical in every state (not A-D income)
    # A grasp. Parked value ~0.8/arm (contact_q ~0.85 at thumb 3-5 N vs fingers 2.5-5 N, keep ~0.45).
    W_GRASP_SRC, W_GRASP_RCV = 1.5, 1.5   # iter_21: 1.0 -> 1.5, holding must clearly out-earn letting go
    TIP_R = 0.027            # verbatim grip geometry constant
    UPRIGHT_TOL = 0.35       # rad, the ENV's own receiver-upright success limit (20 deg)
    # B lift. KEEP (operator item 5): LIFT_FREE_SRC0 0.40 / LIFT_TARGET_SRC 0.30 (K3 reachability probe:
    # palm z 0.69-0.76 -> roll error 10-23 deg). Weight 3.0 -> 1.5 so A+B stays under the outcome.
    W_LIFT_SRC = 1.5
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.30
    LIFT_FREE_SRC0 = 0.40
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    # receiver: a small "it is off the table" state only. The old 0.10 m ceiling is gone because the
    # placement below (receiver mouth 1.7 r under a source origin at z ~0.57) needs h_rcv ~0.2 m.
    W_LIFT_RCV = 0.5
    LIFT_TARGET_RCV = 0.08   # measured rcv lift 0.073 m
    CARRY_H = 0.03           # "carried" height, both arms (a height, not an angle)
    # C placement - SOURCE-relative target for the receiver MOUTH (brief 5-1 C, K2, K5, K6).
    PLACE_AZ = math.radians(68.0)   # K2: median lean azimuth at tilt > 60 deg (IQR 59-81). The palm
                                    # yaw box (+-45 deg, K1) is what makes +-45 around it reachable.
    PLACE_R_R = 2.0          # horizontal source-origin -> receiver-mouth distance = 2 r = 0.056 m:
                             # where the lip sits at a quarter turn (K6 mouth_z 0.054 ~= 1.9 r)
    LIP_Z_HI_R = 0.7         # spout clearance above the rim, 0.7 r ~= 2 cm (iter_17 permission height)
    SINK_KEEP_R = 1.0        # lip falls r*sin + mouth_z*(1-cos); iter_19 derivation: origin must sit
                             # (0.7 + 1.0) r = 1.7 r above the receiver rim to clear it when horizontal
    PLACE_LOW_W = 0.5        # receiver lower than the target height costs half (the aim term decides
                             # the final spout height; being too HIGH is the dangerous side)
    CENTER_FREE_R = 2.0      # |y of the cup-pair midpoint| free up to 2 r (y ~= 0 centreline, FPP view)
    W_PLACE = 1.0
    # D aim - spout onto a point LIP_Z_HI_R*r above the receiver mouth (horizontal + height, 3D)
    W_AIM = 1.5
    V_PROG = 0.15            # m/s: progress terms saturate at the stable cup speed limit (SRC_V_FREE),
                             # so going faster than allowed buys nothing
    # E direction (iter_19): (1.344/1.571)*0.801 at the i18 failure state -> 1.03/step
    W_TILT_DIR = 1.5
    # H height keep (iter_19 verbatim): i18 sank 0.617 -> 0.359 m in 6.7 s = 0.039 m/s
    W_POUR_SINK = 1.0
    V_Z_SCALE = 0.04
    # F outcome - unchanged weights (per-bead units with 20 beads)
    W_POUR = 150.0           # +7.5 per bead delivered
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0            # -2 per extra bead dumped in the same step
    W_SPILL = 240.0          # -12 per bead spilled
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
    # G control - KEEP LIST values
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, W_ROT_POUR = 1.0, 1.0  # stricter limit while the spout is at the mouth
    RCV_W_FREE, RCV_W_SCALE, W_RCV_SPIN = 0.5, 1.0, 0.5
    STILL_FREE, STILL_SCALE, W_STILL_POUR = 0.05, 0.15, 0.5
    W_TILT_PEN_RCV = 0.3
    W_FLIP_PALM, W_FLIP_HAND = 0.3, 0.2
    W_RATE_PALM_SRC, W_RATE_PALM_RCV, W_RATE_HAND = 0.03, 0.05, 0.01
    W_ACT_RATE = 0.05
    BANG_DEAD, W_BANG_SRC, W_BANG_RCV = 0.1, 1.5, 2.0
    SAT_LO, W_PALM_SAT_SRC, W_PALM_SAT_RCV = 0.7, 3.0, 1.5
    MARGIN_FREE, W_ARM_LIMIT = 0.30, 0.4  # weak rudder (i15: a strong limit penalty failed)
    QDD_FREE, QDD_SCALE = 2.5, 10.3
    QDD_PEAK_FREE, QDD_PEAK_SCALE = 12.8, 12.8
    W_ARM_SHAKE = 0.4
    # safety / disturbance
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

    # ------------------------------------------------------------------ A. grasp
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    h_src_raw = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    carry_src = g_src * torch.clamp(h_src_raw / CARRY_H, max=1.0)
    # knock-over guard only while the source cup is still on the table; no attitude opinion in the air
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

    # ------------------------------------------------------------------ B. lift
    lift_free_src = LIFT_FREE_SRC0 + geo["lip_drop"]     # geometric allowance, never an angle target
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    over_lift = torch.clamp((h_src - lift_free_src) / OVER_LIFT_RANGE, 0.0, 1.0)   # ungated on purpose
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_band - W_OVER_LIFT * over_lift
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * (0.5 + 0.5 * upright_rcv)

    # ------------------------------------------------------------------ bead bookkeeping (gates)
    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    has_beads = torch.clamp(fill / 0.10, max=1.0)        # >= 2 beads left
    stage_cd = carry_both * has_beads                     # placement / aim live while there is a pour ahead

    # velocities of the two points that matter
    v_mouth = _point_vel(ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_pos, ctx.rcv_cup_mouth_pos)
    v_lip = _point_vel(ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_pos, geo["lip_pt"])

    # ------------------------------------------------------------------ C. placement (source-relative)
    # Target for the receiver MOUTH: 2 r out from the source origin at azimuth +68 deg (front-left, the
    # only side the source can pour to - K1/K2), 1.7 r BELOW the source origin (spout clears the rim by
    # 0.7 r even at a quarter turn). The target moves with the source cup, so either arm may do the work.
    # Potential = distance to it (+ centreline excess); paid as its exact rate -> 0 when parked anywhere.
    off = torch.tensor([PLACE_R_R * math.cos(PLACE_AZ), PLACE_R_R * math.sin(PLACE_AZ),
                        -(LIP_Z_HI_R + SINK_KEEP_R)], dtype=dtype, device=ctx.src_cup_pos.device) * r
    e = ctx.rcv_cup_mouth_pos - (ctx.src_cup_pos + off.unsqueeze(0))
    de = v_mouth - ctx.src_cup_lin_vel
    k_z = torch.where(e[:, 2] > 0.0, torch.ones_like(e[:, 2]), PLACE_LOW_W * torch.ones_like(e[:, 2]))
    ez = k_z * e[:, 2]
    d_place = torch.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2 + ez ** 2)
    dd_place = (e[:, 0] * de[:, 0] + e[:, 1] * de[:, 1] + ez * k_z * de[:, 2]) \
        / torch.clamp(d_place, min=1e-4)
    y_mid = 0.5 * (ctx.src_cup_pos[:, 1] + ctx.rcv_cup_pos[:, 1])
    vy_mid = 0.5 * (ctx.src_cup_lin_vel[:, 1] + ctx.rcv_cup_lin_vel[:, 1])
    off_center = (y_mid.abs() > CENTER_FREE_R * r).to(dtype)
    dd_center = off_center * torch.sign(y_mid) * vy_mid
    place = stage_cd * _progress(-(dd_place + dd_center), V_PROG)

    # ------------------------------------------------------------------ D. aim (3D, difference form)
    # Potential: horizontal lip->mouth distance plus how far the lip is BELOW its clearance height.
    # Above that height nothing is charged, so pouring from higher is never taxed here.
    z_tgt = LIP_Z_HI_R * r
    below = torch.clamp(z_tgt - geo["lip_dz"], min=0.0)
    d_aim = torch.sqrt(geo["lip_xy"] ** 2 + below ** 2)
    dv = v_lip - v_mouth
    dd_aim = ((geo["lip_xy_vec"] * dv[:, :2]).sum(dim=-1) - below * dv[:, 2]) / torch.clamp(d_aim, min=1e-4)
    aim = stage_cd * _progress(-dd_aim, V_PROG)

    # permission (a GATE, never paid as a state): spout inside the r-window, above the rim, leaning toward
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    perm_xy = 1.0 - _smoothstep((geo["lip_xy"] - r_in) / (r_out - r_in))
    lip_above = torch.clamp(geo["lip_dz"] / z_tgt, 0.0, 1.0)
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = perm_xy * lip_above * toward
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r) / ((LEAD_ZERO_R - LEAD_FULL_R) * r))

    # ------------------------------------------------------------------ E. direction (cost only)
    dir_err, psi = _tilt_direction_error(ctx, geo["d_hat"])
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)          # magnitude only as a multiplier, no target
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err

    # ------------------------------------------------------------------ H. keep height while pouring
    org_vz = ctx.src_cup_lin_vel[:, 2] - v_mouth[:, 2]
    h_org = ctx.src_cup_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    org_low = (h_org < (LIP_Z_HI_R + SINK_KEEP_R) * r).to(dtype)
    pour_sink = ctrl_gate * not_nested * org_low * torch.clamp(org_vz / V_Z_SCALE, -1.0, 1.0)

    # ------------------------------------------------------------------ F. outcome (the beads)
    n_beads = float(max(int(ctx.num_beads), 1))
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_step = settled * torch.clamp(ctx.d_spill, min=0.0)
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
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

    # ------------------------------------------------------------------ G. control
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
        + W_STILL_POUR * ctrl_gate * perm_xy * rcv_still_over   # still only once the spout is over it

    a_abs = ctx.actions.abs()
    da = ctx.actions - ctx.prev_actions
    flips = torch.clamp(-(ctx.actions * ctx.prev_actions), min=0.0)
    flip_palm_src = flips[:, 0:6].sum(dim=-1)
    flip_palm_rcv = flips[:, 12:18].sum(dim=-1)
    flip_hand = flips[:, 6:12].sum(dim=-1) + flips[:, 18:24].sum(dim=-1)
    # iter_21: control penalties no longer multiplied by the grasp flags - in i20 releasing the cups removed
    # every one of them, so letting go out-earned holding. They now cost the same whether or not a cup is held.
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

    # ------------------------------------------------------------------ safety / disturbance
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
        # C / D / E / H: difference forms and a cost - none of them pays for standing still
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": -W_TILT_DIR * tilt_dir,
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
    comps["diag_place_az_deg"] = torch.atan2(ctx.rcv_cup_mouth_pos[:, 1] - ctx.src_cup_pos[:, 1],
                                             ctx.rcv_cup_mouth_pos[:, 0] - ctx.src_cup_pos[:, 0]) * (180.0 / math.pi)
    comps["diag_y_mid"] = y_mid
    comps["diag_aim_dist_r"] = d_aim / r
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_h_org"] = h_org
    comps["diag_perm"] = perm
    comps["diag_toward"] = toward
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
bead/in_target: [0, 3.66e-05, 6.1e-05, 0.000122, 0.000195, 6.1e-05, 0.000171, 0.000122, 0.000171, 0.000122]  min 0 · mean 8.78e-05 · max 0.000232
bead/spill: [0, 0.0161, 0.0169, 0.0143, 0.0157, 0.0142, 0.017, 0.0157, 0.0181, 0.0137]  min 0 · mean 0.0152 · max 0.0269
done/drop: [0, 0.000244, 0.000732, 0.000244, 0.000732, 0.000244, 0.000244, 0, 0.000732, 0]  min 0 · mean 0.000344 · max 0.00146
done/mimic_runaway: [0, 0, 0.000244, 0, 0.000244, 0, 0, 0, 0.000244, 0.000244]  min 0 · mean 0.00013 · max 0.00122
episode_lengths/step: [102, 767, 796, 736, 743, 775, 755, 712, 762, 737]  min 102 · mean 747 · max 830
reward/act_flip: [-0.648, -0.426, -0.334, -0.259, -0.209, -0.177, -0.152, -0.137, -0.121, -0.119]  min -0.648 · mean -0.224 · max -0.103
reward/aim_lip: [0, 0.00545, 0.0175, 0.0379, 0.0718, 0.0931, 0.0958, 0.114, 0.123, 0.174]  min -0.233 · mean 0.091 · max 0.208
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/arm_posture: [-0.256, -0.361, -0.397, -0.42, -0.46, -0.488, -0.488, -0.48, -0.482, -0.478]  min -0.501 · mean -0.446 · max -0.256
reward/cmd_rate: [-0.413, -0.294, -0.245, -0.203, -0.171, -0.15, -0.133, -0.123, -0.113, -0.107]  min -0.413 · mean -0.176 · max -0.0996
reward/cup_disturb: [-0.323, -0.103, -0.1, -0.091, -0.0986, -0.0906, -0.0908, -0.0916, -0.0824, -0.0899]  min -0.61 · mean -0.0958 · max -0.0322
reward/diag_aim_dist_r: [10.4, 7.08, 5.76, 6.69, 12.7, 4.94, 15.2, 6.14, 5.31, 6.14]  min 4.73 · mean 7.73 · max 21.3
reward/diag_balance_src: [0, 0.6, 0.604, 0.627, 0.604, 0.633, 0.629, 0.631, 0.635, 0.623]  min 0 · mean 0.621 · max 0.723
reward/diag_clean: [1, 0.965, 0.964, 0.968, 0.964, 0.966, 0.963, 0.965, 0.961, 0.97]  min 0.944 · mean 0.966 · max 1
reward/diag_ctrl_gate: [0, 0.577, 0.541, 0.6, 0.599, 0.629, 0.619, 0.614, 0.635, 0.649]  min 0 · mean 0.608 · max 0.825
reward/diag_dir_err_deg: [82.7, 87.6, 83.6, 62.9, 60.5, 53, 46.3, 49.2, 43.8, 44.6]  min 39.7 · mean 57.8 · max 100
reward/diag_finish_q: [0, 2.06e-05, 5.67e-05, 0.000185, 0.000234, 0.000118, 0.000432, 0.000247, 0.000246, 0.000183]  min 0 · mean 0.000155 · max 0.000566
reward/diag_grip_keep_src: [0.21, 0.4, 0.437, 0.459, 0.418, 0.398, 0.402, 0.39, 0.395, 0.425]  min 0.21 · mean 0.411 · max 0.467
reward/diag_h_org: [-0.0537, 0.0474, 0.0776, 0.0868, 0.0735, 0.0739, 0.139, 0.0774, 0.0783, 0.0478]  min -0.0724 · mean 0.0709 · max 0.211
reward/diag_lift_budget: [0.4, 0.403, 0.406, 0.409, 0.409, 0.408, 0.408, 0.408, 0.409, 0.41]  min 0.4 · mean 0.408 · max 0.412
reward/diag_lip_dz: [-2.78e-05, 0.0973, 0.126, 0.131, 0.118, 0.119, 0.183, 0.122, 0.122, 0.0898]  min -0.0235 · mean 0.116 · max 0.256
reward/diag_lip_xy: [0.291, 0.196, 0.16, 0.186, 0.356, 0.137, 0.424, 0.171, 0.148, 0.17]  min 0.131 · mean 0.212 · max 0.596
reward/diag_outcome_income: [0, 0, 0.000141, 0.000123, 0.000205, 0.000163, 9.05e-05, 0.000252, 0.000345, 0.000158]  min 0 · mean 0.000149 · max 0.000912
reward/diag_palm_sat_src: [0.441, 0.455, 0.358, 0.28, 0.229, 0.201, 0.183, 0.16, 0.158, 0.165]  min 0.152 · mean 0.254 · max 0.585
reward/diag_perm: [0, 0.00295, 0.0346, 0.0373, 0.0639, 0.301, 0.325, 0.312, 0.426, 0.445]  min 0 · mean 0.228 · max 0.485
reward/diag_place_az_deg: [90.1, 61.4, 85.3, 86.6, 85.6, 88, 91.8, 92.2, 96.1, 104]  min 36 · mean 87 · max 105
reward/diag_place_dist_r: [10.2, 6.54, 5.56, 6.67, 12.6, 5.05, 15.6, 6.24, 5.68, 6.66]  min 5.03 · mean 7.83 · max 21.8
reward/diag_release_beads: [0, 0.00562, 0.0061, 0.00244, 0.00537, 0.00684, 0.00391, 0.00366, 0.00464, 0.00317]  min 0 · mean 0.00409 · max 0.012
reward/diag_src_margin_min: [0.194, 0.114, 0.115, 0.104, 0.0917, 0.0756, 0.0697, 0.0706, 0.0653, 0.0663]  min 0.0607 · mean 0.0863 · max 0.194
reward/diag_src_tilt_deg: [1.04, 16.5, 17.1, 18.1, 17.7, 16.3, 17.1, 16.8, 18.4, 21]  min 1.04 · mean 17.7 · max 23.6
reward/diag_src_v: [0.0385, 0.19, 0.231, 0.21, 0.248, 0.188, 0.244, 0.219, 0.177, 0.175]  min 0.0385 · mean 0.405 · max 74.4
reward/diag_src_w: [0.568, 1.25, 1.53, 1.43, 1.91, 1.54, 1.49, 1.25, 1.26, 1.43]  min 0.568 · mean 7.69 · max 2.02e+03
reward/diag_stage_cd: [0, 0.62, 0.617, 0.652, 0.637, 0.671, 0.66, 0.651, 0.68, 0.675]  min 0 · mean 0.65 · max 0.835
reward/diag_state_income: [0.459, 2.78, 2.91, 2.97, 2.93, 3.05, 3.03, 3.02, 3.1, 3.04]  min 0.459 · mean 2.97 · max 3.33
reward/diag_toward: [0.562, 0.499, 0.557, 0.718, 0.746, 0.841, 0.891, 0.877, 0.909, 0.905]  min 0.381 · mean 0.785 · max 0.92
reward/diag_unaimed_beads: [0, 0.00562, 0.00587, 0.00244, 0.00537, 0.00684, 0.00364, 0.00366, 0.00449, 0.00317]  min 0 · mean 0.00401 · max 0.0117
reward/diag_y_mid: [0.000239, 0.0752, 0.0445, 0.00286, -0.108, -0.0254, -0.167, -0.0333, -0.0305, -0.0361]  min -0.172 · mean -0.0125 · max 0.17
reward/grasp_rcv: [0.217, 0.762, 0.782, 0.796, 0.777, 0.815, 0.823, 0.811, 0.853, 0.833]  min 0.217 · mean 0.798 · max 0.878
reward/grasp_src: [0.242, 0.89, 0.928, 0.955, 0.924, 0.946, 0.941, 0.94, 0.951, 0.954]  min 0.242 · mean 0.939 · max 1.11
reward/lift_rcv: [0, 0.302, 0.294, 0.321, 0.318, 0.331, 0.328, 0.324, 0.336, 0.342]  min 0 · mean 0.32 · max 0.399
reward/lift_src: [0, 0.828, 0.909, 0.896, 0.905, 0.956, 0.935, 0.946, 0.959, 0.914]  min 0 · mean 0.912 · max 1.08
reward/palm_bang: [-0.842, -0.61, -0.551, -0.472, -0.4, -0.298, -0.23, -0.204, -0.176, -0.149]  min -0.862 · mean -0.35 · max -0.142
reward/palm_sat: [-2.13, -2, -1.63, -1.35, -1.22, -1.18, -1.14, -1.08, -1.07, -1.08]  min -2.42 · mean -1.34 · max -1.03
reward/place_rcv: [0, 0.00191, 0.0352, 0.0419, 0.0515, 0.0437, 0.0625, 0.0838, 0.0852, 0.0815]  min -0.413 · mean 0.0566 · max 0.102
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0.00183, 0, 0]  min -0.00549 · mean 9.41e-05 · max 0.00366
reward/pour_sink: [0, 0.00742, 0.00648, 0.0014, 0.00778, -0.000137, 0.00642, 0.00388, 0.005, 0.0159]  min -0.0141 · mean 0.00844 · max 0.0834
reward/rcv_steady: [-0.00432, -0.183, -0.207, -0.188, -0.181, -0.232, -0.208, -0.187, -0.195, -0.193]  min -0.274 · mean -0.199 · max -0.00432
reward/safety: [-0.018, -0.00844, -0.00518, -0.00657, -0.00497, -0.0035, -0.00505, -0.00333, -0.0049, -0.00543]  min -0.0249 · mean -0.0053 · max -0.000806
reward/spill_delta: [0, -0.0674, -0.0732, -0.0293, -0.0615, -0.082, -0.0469, -0.041, -0.0557, -0.0381]  min -0.141 · mean -0.0487 · max 0
reward/src_speed: [0, -0.0653, -0.0843, -0.0742, -0.0606, -0.048, -0.0462, -0.0338, -0.0308, -0.0349]  min -0.27 · mean -0.0537 · max 0
reward/src_spin: [0, -0.0228, -0.0384, -0.0315, -0.0311, -0.0493, -0.0448, -0.0397, -0.0429, -0.0548]  min -0.0651 · mean -0.0411 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_dir: [0, -0.0937, -0.0813, -0.0434, -0.0403, -0.0294, -0.0246, -0.0258, -0.0279, -0.0357]  min -0.109 · mean -0.0425 · max 0
reward/total: [-3.17, -0.408, 0.245, 0.9, 1.12, 1.35, 1.57, 1.77, 1.89, 1.92]  min -3.27 · mean 1.1 · max 2.07
reward/transfer_tail: [0, 0, 6.56e-06, 0, 3.23e-06, 1.44e-06, 4.38e-06, 2.12e-05, 0, 6.38e-06]  min 0 · mean 1.16e-05 · max 0.000369
reward/transferred: [0, 0, 0.000135, 0.000123, 0.000202, 0.000162, 8.61e-05, 0.00023, 0.000345, 0.000152]  min 0 · mean 0.000138 · max 0.000543
reward/unaimed_release: [0, -0.0168, -0.0176, -0.00732, -0.0161, -0.0205, -0.0109, -0.011, -0.0135, -0.00952]  min -0.0352 · mean -0.012 · max 0
rewards/step: [-267, -358, 163, 472, 795, 1.09e+03, 1.22e+03, 1.15e+03, 1.38e+03, 1.25e+03]  min -879 · mean 802 · max 1.62e+03
task/aim_dist: [0.319, 0.265, 0.255, 0.278, 0.439, 0.237, 0.543, 0.265, 0.25, 0.256]  min 0.223 · mean 0.307 · max 0.729
task/cup_collision_rate: [0, 0.000732, 0.00122, 0.000977, 0.000488, 0.000732, 0.000488, 0.000488, 0.00122, 0.00293]  min 0 · mean 0.00127 · max 0.00732
task/cups_center_dist: [0.32, 0.265, 0.259, 0.286, 0.447, 0.245, 0.551, 0.273, 0.259, 0.265]  min 0.233 · mean 0.313 · max 0.737
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.00122, 0.000732, 0.00171, 0.000977, 0.000977, 0.00122, 0.000732, 0.00122, 0.000977]  min 0 · mean 0.00095 · max 0.00439
task/rcv_cup_lift: [8.5e-05, 0.0583, 0.0557, 0.0657, 0.0664, 0.0614, 0.0598, 0.0595, 0.0593, 0.0702]  min 8.5e-05 · mean 0.0677 · max 0.182
task/rcv_grasped: [0, 0.761, 0.791, 0.79, 0.767, 0.804, 0.8, 0.78, 0.815, 0.795]  min 0 · mean 0.782 · max 0.9
task/rcv_hand_foreign_rate: [0.0498, 0.0142, 0.00659, 0.00513, 0.00562, 0.00171, 0.00537, 0.00342, 0.00269, 0.00269]  min 0.000977 · mean 0.00581 · max 0.0818
task/src_cup_lift: [0.000307, 0.159, 0.187, 0.206, 0.193, 0.189, 0.252, 0.191, 0.191, 0.172]  min 0.000307 · mean 0.192 · max 0.327
task/src_grasped: [0, 0.702, 0.71, 0.723, 0.709, 0.739, 0.739, 0.741, 0.751, 0.743]  min 0 · mean 0.733 · max 0.912
task/src_hand_foreign_rate: [0.00195, 0.00195, 0.00195, 0.00195, 0.00195, 0.00122, 0.00171, 0.000732, 0.00122, 0.00195]  min 0 · mean 0.00198 · max 0.00635
task/src_tilt_deg: [1.04, 16.5, 17.1, 18.1, 17.7, 16.3, 17.1, 16.8, 18.4, 21]  min 1.04 · mean 17.7 · max 23.6
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

# round 22 operator notes — iter_21 은 "들고 가만히 있기"로 수렴했다. 기울기 수입 부재와 배치 불가 두 원인만 고친다

확인된 사실만 적는다. 사용자 승인(09-25 저녁): i21 종료, 아래 세 변경 모두 반영.

## i21 결과 (t2r_rh3_i21, 파지 단계 ckpt, e519 에서 종료)

학습 곡선(e519): 소스/리시버 파지 0.76 / 0.78(최대 0.91 / 0.90).

결정론 평가 ep400(64 env · 96 에피소드):

| 항목 | 값 |
|---|---|
| 파지(소스/리시버) | 1.00 / 1.00 |
| 소스 컵 z | 0.47 m |
| env 별 최대 기울기 중앙 | 48° (>60° 13/64, >90° 2/64) |
| 기울였을 때 방향 오차 | 43° |
| 주둥이 − 받는 컵 입구 | 높이 +13.8 cm, 수평 23.4 cm |
| 받는 컵 z | 0.31 ~ 0.37 m |
| 두 컵 y | 소스 −0.17, 리시버 +0.10 (약 27 cm 떨어짐) |
| 비드 | 소스를 떠난 146 개 중 받는 컵에 들어간 것 2 개 |

- **파지·들기까지는 된다. 받는 컵이 붓는 자리로 오지 않고, 소스는 거의 기울이지 않는다.**

## 원인 (코드: iter_21/compute_reward.py)

1. **기울기를 양(+)으로 주는 항이 없다.** 기울기로 얻는 수입은 pour_delta(비드가 들어갈 때 한 알 +7.5)뿐이다. 기울이면
   tilt_dir(기울기 × 방향 오차)·src_spin·spill(한 알 −12)·unaimed_release(한 알 −3) 비용만 붙는다.
2. **배치가 안 되므로 비드가 들어갈 수 없다.** place_rcv 의 목표가 소스 기준 상대 좌표라서 소스를 들면 목표도 같이 올라간다.
   받는 컵 입구가 주둥이 아래로 오지 않는다(수평 23 cm). 들어갈 수 없으니 기울이는 것은 항상 손해다.
3. lift_src 는 0.40 + lip_drop 까지 지불해서 높이 든 채 머무는 것이 싸다.

## iter_22 변경 (이것만, 나머지 설계는 iter_21 그대로)

1. **배치 목표를 소스에 끌려 올라가지 않게.** 받는 컵 입구 목표를 **절대 높이** 기준으로 두고, 수평은 소스 주둥이 기준
   `ctx.cup_radius` 배수 창으로 본다. 주둥이 높이에는 상한을 둔다(입구보다 위이되 너무 높지 않게). 창은 y ≈ 0 중심선 부근.
2. **창 안에서만 기울이는 쪽으로 작은 신호.** 주둥이가 허가 창 안에 있을 때만, 받는 컵 쪽으로 기울기가 **늘어나는 속도**에
   작은 차분 보상을 준다. **각도 목표·각도 대역은 두지 않는다**(사용자 원칙). 상한을 둬서 이 신호가 배달 수입보다 커지지 않게.
3. **방향 벌점과 흘림 비용의 배치.** tilt_dir 는 창 **밖**에서만 강하게 부과한다. 첫 비드가 나가기 전, 창 **안**에서의 흘림 비용은
   완화한다(첫 한 알이 나가야 경사가 생긴다 — 브리프 i17 교훈).

지키는 것(사용자 원칙): 각도 처방 금지 · 속도·제어 벌점 완화 금지 · 창은 `ctx.cup_radius` 배수 · y ≈ 0 중심선 ·
예산 규칙(브리프 5-2, 상태 수입 < 결과 수입).

## 웜스타트
- 파지 단계 ckpt rh2_i07-r3 ep400(i21 과 같음).


---

# REPAIR ROUND (reviewer request_changes on your previous answer)

Your previous answer is below. It passed the static check and dry run, but a code review found defects in the NEW term tilt_open and the spill relief. Return the COMPLETE corrected answer in the same format. Keep everything else exactly as in your previous answer (all 25 iter_21 terms, weights, control/safety penalties unchanged; only the three operator changes of round 22).

## Required fixes
1. tilt_open must not be farmable by cycling in and out of the window. The symmetric clamp(-1,+1) on the potential rate breaks telescoping: fast exits are charged at most -1/step while entries are paid. Use an asymmetric bound for tilt_open only: payments capped at +1/step, charges not capped (or capped very large, e.g. -10). Any closed cycle must sum to <= 0.
2. tilt_open must be direction-aware. theta = atan2(u.d_hat, u_z) reaches pi/2 at a horizontal cup even when leaning sideways (psi < 90 deg). Weight the potential by the lean direction, e.g. potential = win * theta_pos * max(cos psi, 0), and derive the rate from the same potential (no separate un-derived rate).
3. Gate both tilt_open and the spill relief on "before the first bead has LEFT the source" (operator note wording), not "before the first bead landed in the receiver". For example first_out = clamp(1 - n_beads * (released beads so far), 0, 1) where released = beads no longer in the source (use ctx.bead_in_source_frac: released = 1 - bead_in_source_frac).
4. Bound the cumulative tilt_open income well below delivery: total over an episode should be at most about one bead (+7.5). Lower W_TILT_OPEN accordingly (e.g. 0.1) and state the resulting cumulative bound in a comment.
5. Placement must not pull the receiver into the source cup body. With the source upright, lip_xy <= r implies the cup bodies overlap horizontally; only apply the horizontal placement pull (eh) once the source bottom is above the receiver rim (use ctx.cup_mouth_z, ctx.cup_bottom_z geometry, no new absolute constants), or add a body-clearance term in cup-radius multiples.
Do not change anything else. Do not add angle targets or bands. Do not relax any speed/control penalty.

## Your previous answer

## 1. What the task means and the stages the robot goes through

Two arms, two cups. The right hand (source) takes the cup that holds 20 beads. The left hand (receiver) takes the empty cup. The task runs in these stages:

- **A. Grasp.** Each hand makes a fingertip grasp on its own cup.
- **B. Lift.** Both cups come off the table. The source goes high, because the wrist only produces the commanded rotation when the palm is high (K3).
- **C. Place.** The receiver mouth goes under the point where the source spout will be. That point is on the source's front-left side (K2), near the y = 0 centreline.
- **D. Aim.** The source spout ends up horizontally within a few cup radii of the receiver mouth, and above it but not far above it.
- **E. Direction.** The source leans toward the receiver.
- **F. Outcome.** Beads fall through the air into the receiver. Spill is a cost. Success is judged by the environment.
- **G. Control and H. Height.** Tilting stays at a stable speed with smooth commands, and the source does not sink while it pours.

## Diagnosis of i21 (from the prompt only)

The policy grasps and lifts, then stops.

- **Grasp and lift work:** src_grasped is about 0.74, rcv_grasped about 0.80, and state income is about 3.0 per step.
- **Nothing is delivered:** outcome income is about 1.5e-4, in_target is about 1e-4, and there are no successes.
- **The cups never meet:** the spout is about 17 cm from the receiver mouth horizontally, and place_rcv and aim_lip only pay about 0.08 and 0.17 per step.
- **The source barely tilts:** the curve shows about 20 deg; the deterministic replay shows a median maximum of 48 deg against about 70 deg for the first bead.
- **Why:** no term pays for tilting, only costs attach to it (operator cause 1). The placement target was tied to the source, so it rose whenever the source rose (cause 2).

iter_22 applies only the three operator changes and keeps the rest of iter_21 unchanged:

1. **Placement (place_rcv).** The receiver mouth target is now a fixed world height: the rim of a cup standing on the table plus 0.15 m. Three more rules come with it:
   - The receiver mouth is pulled horizontally until the spout is within 1 r of it.
   - The spout gets a height ceiling of 4 r above the rim. It sits in the aim potential and in the permission window.
   - The receiver mouth may be at most 2 r from y = 0.
   - I kept iter_21's pourable-side knowledge (K2) as a guard. It costs nothing inside 68 ± 45 deg around the source.
2. **New term tilt_open.** It pays for tilting toward the receiver only while the spout is inside the window, and only until the first bead has landed.
   - It is the exact rate of the potential `win * theta+`, so moving in and out of the window earns nothing.
   - There is no angle target and no band.
   - It is capped at 0.5 per step, which saturates at the pour-phase spin limit of 1 rad/s.
3. **Direction cost and spill cost.**
   - tilt_dir keeps weight 1.5 outside the window and drops to 0.3 inside it.
   - Inside the aimed window, spill costs x0.25 (-3 per bead) until the first bead has landed in the receiver.

All speed, control and safety penalties, lift_src, the outcome weights, and the budget structure are unchanged. There are now 26 summed terms: iter_21's 25 plus tilt_open.

## 2. Reward function

```python
import torch
import math


# =====================================================================================================
# iter_22 = iter_21 design + ONLY the three round-22 operator changes (09.25 evening, user approved):
#   (1) place_rcv : receiver MOUTH target at an ABSOLUTE height (rim of a cup standing on the table, raised
#                   by RCV_POUR_LIFT) - no longer dragged up by the source. Horizontal part judged against the
#                   source SPOUT with a cup_radius window; pour point near the y = 0 centreline; the spout
#                   height gets a ceiling (aim_lip potential and the permission window).
#   (2) tilt_open : NEW small difference-form signal: rate of tilt TOWARD the receiver, paid only while the
#                   spout is inside the window and until the first bead has landed in the receiver. It is the
#                   exact rate of the potential win * theta+, so moving in / out of the window cannot farm it.
#                   No angle target, no band, capped at W_TILT_OPEN per step.
#   (3) tilt_dir  : strong only OUTSIDE the window (1.5), weak inside (0.3). spill_delta: x0.25 inside the
#                   aimed window until the first bead has landed (i17 lesson: the first bead must get out).
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
    allowed buys nothing and exploration jitter nets zero (i20: an asymmetric clamp made jitter a net loss)."""
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


def _tilt_rate_toward(up, ang_vel, d_hat):
    """iter_22: d(theta)/dt [rad/s] of the signed tilt toward d_hat, theta = atan2(u . d_hat, u_z), from
    du/dt = w x u (d_hat held fixed over one step). A RATE only - no angle value is used as a target."""
    du = torch.cross(ang_vel, up, dim=-1)
    u_d = (up[:, :2] * d_hat).sum(dim=-1)
    du_d = (du[:, :2] * d_hat).sum(dim=-1)
    den = torch.clamp(u_d * u_d + up[:, 2] * up[:, 2], min=1e-3)
    return (up[:, 2] * du_d - u_d * du[:, 2]) / den


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
    # tilt_open - iter_22 change 2
    W_TILT_OPEN = 0.5        # <= 0.5/step: far below one bead (+7.5) and transferred (~7/step at 0.9 in)
    OPEN_BEADS = 1.0         # active until the first bead has landed in the receiver, then pour_delta takes over
    # H height keep while pouring (iter_19 verbatim)
    W_POUR_SINK = 1.0
    V_Z_SCALE = 0.04
    # F outcome - unchanged weights (per-bead units with 20 beads)
    W_POUR = 150.0           # +7.5 per bead delivered
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0            # -2 per extra bead dumped in the same step
    W_SPILL = 240.0          # -12 per bead spilled
    SPILL_RELIEF = 0.25      # iter_22 change 3: x0.25 (-3/bead) inside the AIMED window before the first landing
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
    TILT_RATE_SCALE = ROT_POUR_FREE   # rad/s: tilt_open saturates at the pour-phase spin limit
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
    first_open = torch.clamp(OPEN_BEADS - n_beads * in_t, 0.0, 1.0)   # 1 until the first bead has landed

    # velocities of the two points that matter, and the spout rates relative to the mouth
    v_mouth = _point_vel(ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_pos, ctx.rcv_cup_mouth_pos)
    v_lip = _point_vel(ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_pos, geo["lip_pt"])
    dv = v_lip - v_mouth
    dlip_xy = (geo["lip_xy_vec"] * dv[:, :2]).sum(dim=-1) / torch.clamp(geo["lip_xy"], min=1e-4)
    dlip_dz = dv[:, 2]

    # ------------------------------------------------------------------ spout window (a GATE, never a state income)
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    z_lo, z_hi = LIP_Z_HI_R * r, LIP_Z_MAX_R * r
    wnd = _spout_window(geo["lip_xy"], geo["lip_dz"], dlip_xy, dlip_dz, r_in, r_out, z_lo, z_hi,
                        LIP_Z_FADE_R * r)
    win = wnd["win"]                                      # position window (horizontal x height)
    perm_xy = wnd["w_xy"]
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = win * toward                                   # aimed window: position + leaning toward
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r) / ((LEAD_ZERO_R - LEAD_FULL_R) * r))

    # ------------------------------------------------------------------ C. placement (iter_22 change 1)
    # Receiver mouth: world-fixed height (not tied to the source), horizontally until the SPOUT is within
    # PLACE_WIN_R*r, |y| near the centreline, and not outside the pourable side of the source (guard).
    # Potential = norm of the excesses; paid as its exact rate -> 0 when parked anywhere.
    z_mouth_tgt = ctx.table_z - ctx.cup_bottom_z + ctx.cup_mouth_z + RCV_POUR_LIFT
    dz_m = ctx.rcv_cup_mouth_pos[:, 2] - z_mouth_tgt
    ez = torch.sign(dz_m) * torch.clamp(dz_m.abs() - Z_TOL_R * r, min=0.0)
    eh = torch.clamp(geo["lip_xy"] - PLACE_WIN_R * r, min=0.0)
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
    dd_place = (eh * dlip_xy + ez * v_mouth[:, 2] + ey * torch.sign(y_m) * v_mouth[:, 1]
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

    # ------------------------------------------------------------------ tilt_open (iter_22 change 2)
    # Exact rate of the potential win * theta+ : pays for tilting TOWARD the receiver only inside the window,
    # charges untilting inside it and carrying a tilted spout OUT of it, so no in/out cycle nets income.
    # Off once the first bead has landed (pour_delta takes over) or the cup is empty. No target, no band.
    theta_rate = _tilt_rate_toward(ctx.src_cup_up, ctx.src_cup_ang_vel, geo["d_hat"])
    theta_pos = torch.clamp(geo["theta"], min=0.0)
    open_rate = wnd["dwin"] * theta_pos + win * toward * theta_rate
    tilt_open = carry_both * geo["sep_ok"] * has_beads * first_open * _progress(open_rate, TILT_RATE_SCALE)

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
    # iter_22 change 3: spill inside the AIMED window before the first landed bead costs x0.25 (-3/bead,
    # still a loss vs +7.5 delivered); spill while carrying / outside the window keeps the full -12/bead.
    spill_w = 1.0 - (1.0 - SPILL_RELIEF) * perm * first_open
    spill_step = settled * torch.clamp(ctx.d_spill, min=0.0) * spill_w
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
    comps["diag_aim_dist_r"] = d_aim / r
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_lip_dz_r"] = geo["lip_dz"] / r
    comps["diag_h_org"] = h_org
    comps["diag_win"] = win
    comps["diag_perm"] = perm
    comps["diag_toward"] = toward
    comps["diag_theta_rate"] = theta_rate
    comps["diag_open_rate"] = open_rate
    comps["diag_first_open"] = first_open
    comps["diag_spill_w"] = spill_w
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

## 근거 메모

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 줄) |
|---|---|---|
| place_rcv: 받는 컵 입구 목표를 절대 높이(테이블 위 컵의 림 + 0.15 m)에 둠 | i21 목표가 소스를 따라 올라감. 받는 컵 z 0.31~0.37, 수평 23.4 cm | 860-861, 866, 849-850 |
| place_rcv: 수평은 주둥이가 입구에서 1 r 안에 들 때까지 끌어당김 | lip_xy 0.15~0.42 m, place_dist 5~15 r, place_rcv 0.08/스텝 | 866-867, 670, 675, 692 |
| place_rcv: 받는 컵 입구 \|y\| 가 2 r 를 넘으면 비용 | 두 컵 y: 소스 -0.17, 리시버 +0.10. y_mid 는 -0.03 이라 i21 무료 구간 안에 있었음 | 750, 851, 685, 867 |
| place_rcv: 부을 수 있는 쪽 가드(68±45°) | place_az 가 61°에서 104°로 커짐(K2 범위 가장자리) | 760, 759, 674, 801 |
| aim_lip: 주둥이 천장 4 r(높이 창 3.3 r > 1/4 회전 때 주둥이 하강 약 2.9 r) | 주둥이가 입구보다 +13.8 cm 위. lip_dz 평균 0.12 m. lift_src 가 높이 머물기에 값을 줌 | 849, 669, 862, 867, 763-764 |
| RCV_POUR_LIFT 0.15 m(높이 허용만 예외) | i21 리시버 들기 최대 0.182 m. 이 높이면 소스가 붓는 동안 들기 0.23~0.34 m 로 LIFT_TARGET_SRC 0.30 과 겹침 | 712, 820, 761, 800 |
| tilt_open(신규): 창 안에서만 받는 컵 쪽 기울기 속도에 차분 보상, 스텝당 0.5 이하, 첫 알이 들어가면 꺼짐 | 기울기에 양의 수입이 없었음. 최대 기울기 중앙 48°, 곡선 약 20° | 858-859, 868-869, 847, 678 |
| tilt_open 을 win*theta+ 의 정확한 변화율로 계산 | 창 들락날락 반복으로 돈을 버는 것 방지. 각도 목표·대역 없음 | 748, 817, 869 |
| tilt_dir: 창 밖 1.5, 창 안 0.3 | 방향 오차 44°, toward 0.9, perm 0.45 | 870, 664, 683, 673, 770 |
| spill_delta: 겨냥한 창 안에서 첫 알이 들어가기 전 x0.25(한 알 -3, 배달 +7.5 보다 여전히 손해) | 소스를 떠난 146 알 중 2 알만 들어감. 첫 한 알이 나가야 경사가 생김 | 852, 870-871, 778, 813 |
| perm 에 천장 반영(너무 높은 곳에서 떨어진 비드는 unaimed) | 창은 cup_radius 배수로 유지 | 750, 773, 867 |
| G 제어·안전·lift_src·결과 가중치는 그대로 | 속도·제어 벌점 완화 금지. 나머지 설계는 iter_21 그대로 | 749, 864, 873-874, 125 |
| 예산: 새 상태 수입 없음(C·D·tilt_open 모두 차분 형태) | 상태 수입 약 3.0/스텝, 결과 수입 약 0 | 808-812, 682, 671 |

Assumptions (가설):
- The step dt is not given in the prompt, so the episode total of each difference-form term (at most the per-step cap) cannot be converted to absolute numbers.
- src_cup_ang_vel is assumed to be in the world frame, the same assumption iter_21's _point_vel already makes.
- Whether the receiver arm can actually lift to 0.15 m and move toward the centreline has not been checked.

There are 26 summed terms, one over the brief's target of 25. The extra one is tilt_open, which the operator asked for (change 2).
