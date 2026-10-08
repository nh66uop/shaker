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
    """Monotone ramp that is NEVER flat above `lo` and gets steeper toward `hi`.
    0.5*t + 0.5*t^2 on t = clamp((x - lo)/(hi - lo)): the marginal value per unit x is 0.5/(hi-lo) at
    the bottom and 1.5/(hi-lo) at the top, so the last part is worth three times the first."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _limit_pen(margin, free):
    """How hard an arm is pressed against its joint limits. `margin` is (N, A) with 1 = middle of the
    range and 0 = touching a limit. KEEP LIST, weight 0.4 (a rudder, not a wall): iter_15 proved it
    cannot open wrist range by itself, and i16g showed the range comes from the rendezvous instead
    (diag_src_limit_hit <= 0.05 with diag_frac_rcv at 0.57-0.63)."""
    d = torch.clamp((free - torch.clamp(margin, 0.0, 1.0)) / free, 0.0, 1.0)      # (N, A)
    worst = d.max(dim=-1).values                                                  # (N,)
    return 0.7 * worst * worst + 0.3 * (d * d).mean(dim=-1)


def _shake_pen(qdd, free_mean, scale_mean, free_peak, scale_peak):
    """Arm tremor charged on ACCELERATION only, so the deliberate pouring rotation (a velocity) is
    never taxed. KEEP LIST, weight 0.4. Measured diag_src_qdd 5.4-6.5 all through i16g."""
    q = qdd.abs()
    mean_part = torch.tanh(torch.clamp(q.mean(dim=-1) - free_mean, min=0.0) / scale_mean)
    peak_part = torch.tanh(torch.clamp(q.max(dim=-1).values - free_peak, min=0.0) / scale_peak)
    return 0.65 * mean_part + 0.35 * peak_part


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the CUP frame (axis = cup_up), so every quantity here stays valid while
    the cup is tilted or upside down. VERBATIM since iter_04 - the grasp works
    (src_grasped 0.75, rcv_grasped 0.82 through all of i16g)."""
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
    iter_06 VERBATIM. iter_17 adds ONE output, `grip_keep`: the grasp the cup was lifted with, scored
    on the RELATIVE force distribution (never an absolute force level - sibling track i24 rule) and on
    the pinch geometry in the cup frame, so it is still meaningful with the cup upside down."""
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

    # closure shaping. iter_17: `pose_ok` for the source is now 1 whenever the cup is carried, so this
    # term no longer fades out while the cup turns (measured close_src 0.087/step in i16g).
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # iter_17 NEW: keep the grasp you lifted with. `balance` is a RATIO of the two contact sides, so
    # no absolute force level is prescribed; the grasp flag already proves there IS contact.
    balance = torch.minimum(f_t, f_o) / (torch.maximum(f_t, f_o) + 1e-6)
    grip_keep = g_f * near_grip * (0.3 + 0.7 * balance)

    # C. lift STATE only. 5 mm dead band: tipping on the rim is not lifting.
    h_pos = torch.clamp(cup_pos[:, 2] - cup_spawn_pos[:, 2] - 0.005, min=0.0)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    v_lin = cup_lin_vel.norm(dim=-1)

    # penalties (unweighted, bounded) - all unchanged from iter_04. `slide` is gated on (1 - lifted_w),
    # i.e. it charges dragging a cup that is still on the table and costs nothing once it is carried.
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
            "grip_keep": grip_keep, "balance": balance,
            "far": torch.tanh(4.0 * geo["d_funnel"])}


def _pour_geometry(ctx, lip_r, mouth_z, clearance, sep_min):
    """WHERE the pouring lip is, measured on the DIRECTED spout. VERBATIM since iter_08/09.
    iter_17 note: `theta` is still computed, but from here on it is used ONLY to locate the lip in
    space (and for the sign gate / diagnostics). No reward term reads it as a target or a band.

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
    # what is tilted but NOT toward the receiver: diagnostic only from iter_17 on
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
    # as in iter_06..16 so carry_src and align_mouths keep their measured scale
    d_align = torch.sqrt(lip_xy ** 2 + (0.5 * (lip_dz - clearance)) ** 2)
    lip_u = lip_xy_vec / torch.clamp(lip_xy, min=1e-6).unsqueeze(-1)
    lip_pt = torch.cat([ctx.rcv_cup_mouth_pos[:, :2] + lip_xy_vec,
                        (ctx.src_cup_mouth_pos[:, 2] - lip_r * torch.sin(theta)).unsqueeze(-1)], dim=-1)
    return {"d_hat_x": d_hat[:, 0], "sep_ok": sep_ok, "theta": theta, "theta_eff": theta_eff,
            "mis": mis, "lip_xy": lip_xy, "lip_dz": lip_dz, "d_align": d_align,
            "sin_t": sin_t, "lip_drop": lip_drop, "lip_u": lip_u, "lip_pt": lip_pt}


def _lip_closing_rate(ctx, lip_u, lip_pt):
    """d(lip_xy)/dt from the rigid-body velocities (stateless potential difference).
    Negative = the spout is closing on the receiver mouth. [m/s]  KEEP LIST, verbatim."""
    v_lip = ctx.src_cup_lin_vel + torch.cross(ctx.src_cup_ang_vel, lip_pt - ctx.src_cup_pos, dim=-1)
    v_mouth = ctx.rcv_cup_lin_vel + torch.cross(ctx.rcv_cup_ang_vel,
                                                ctx.rcv_cup_mouth_pos - ctx.rcv_cup_pos, dim=-1)
    return ((v_lip[:, :2] - v_mouth[:, :2]) * lip_u).sum(dim=-1)


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift per arm, then converge < carry < align < PERMISSION < beads.
    #
    # ============================ iter_17: NO ANGLE IS PRESCRIBED =====================================
    # i16g measurement (2100 epochs, warm start rh2_i07-r3 ep400):
    #   attitude income  tilt_aim 0.556 + tilt_near 0.731 + pour_tilt 0.258 + pour_pose 0.217
    #                    + aimed_tilt 0.313  =  +2.07 / step
    #   delivery income  pour_delta 0.001 + transferred 0.009 + transfer_tail 0.000 + src_emptied 0.000
    #                    =  +0.010 / step
    # Holding an attitude paid 200x more than delivering a bead, and task/episode_success was 0 for
    # the whole run. Worse, the fill-coupled ladder release_tilt = 75 deg + 25 deg * (1 - fill) put the
    # measured pour (87.6 deg at fill 0.756, i.e. a prescribed 81 deg) at the TOP of the paid band, so
    # when no bead came out the reward maximiser was to come back DOWN - which is exactly the "starts
    # pouring, then rights the cup" that the operator saw at ep2100.
    #
    # So iter_17 DELETES every angle target, band, ladder and angle-magnitude penalty:
    #   tilt_aim, tilt_near, pour_tilt, pour_pose, aimed_tilt, overtilt, early_tilt, mis_tilt,
    #   release_tilt / theta_pre / theta_max, the `allowed` attitude corridor, src_up / src_up_soft,
    #   and upright_src inside settle_finish.
    # What remains to pay for a pour: the BEADS (pour_delta, transferred, transfer_tail, src_emptied)
    # minus spill and over-flow. Geometry survives only as PERMISSION, in cup_radius multiples, and
    # direction only as a SIGN gate. The pour angle is the policy's to find.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.82 measured)
    W_AWAY_RCV = 0.8                       # hand-to-OWN-cup distance only: independent of where the
                                           # cup is carried, so it never blocks the receiver travel
    W_CLOSE_SRC, W_CLOSE_RCV = 0.6, 0.6    # iter_17: src 0.3 -> 0.6, matching the receiver (no new
                                           # constant). close_src measured 0.087/step and the operator
                                           # reports the grasp looks unstable during the tilt.
    W_GRASP = 2.0                          # grasp formulation and weight unchanged
    # --- iter_17 NEW: keep the grasp that lifted the cup, for as long as the cup is carried ---------
    W_GRIP_HOLD_SRC = 1.2   # ~0.58/step at the measured hold (grasped 0.76, near_grip ~0.8,
                            # balance ~0.7 from thumb 3-5 N vs fingers 2.5-5 N). Inside the +-1/step
                            # budget for new terms and far below the delivery income it must not outbid.
    W_GRIP_HOLD_RCV = 0.6   # half: the receiver is not the hand that turns upside down
    # --- source lift: band + over-lift ramp. KEEP LIST (diag_lift_budget 0.262 measured) ------------
    W_LIFT_SRC = 3.0
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.15
    LIFT_FREE_SRC0 = 0.24   # base of the moving budget; the tilt-linked part is lip_drop (a geometric
                            # allowance, not an angle target: it only says how far the lip has dropped)
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    LIFT_FLOOR = 0.60
    LIFT_PINCH_BONUS = 0.2
    # --- receiver lift band - KEEP LIST (rcv lift 0.073 m measured, lift_rcv 1.43) ------------------
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
    # --- PERMISSION GATES - every window in cup_radius multiples (KEEP LIST) ------------------------
    PERM_IN_R, PERM_OUT_R = 1.0, 3.0        # the spout window, unchanged
    LEAD_FULL_R, LEAD_ZERO_R = 1.75, 5.0    # the lead-in window, unchanged
    W_APPROACH = 1.0
    # iter_17: the vertical half of the window is now in radii too. 0.7 * cup_radius is 0.021 m at the
    # 0.030 m nominal inner radius of these 6 cm shakers and 0.0196 m at 0.028 m, i.e. the SAME ~2 cm
    # window that LIP_Z_HI = 0.02 gave in iter_06..16, but scale- and spawn-independent.
    LIP_Z_HI_R = 0.7
    TILT_LIFT_H = 0.03      # "carrying" height, both arms. A HEIGHT, not an angle.
    W_MEET = 5.0            # KEEP LIST (rendezvous corridor). i16g: diag_frac_rcv 1.047 -> 0.57-0.63
    W_SRC_CROSS = 0.8       # KEEP LIST: the mirror charge on the source cup crossing the same point
    SPAN_MIN = 0.05         # numerical floor on |span| only; never reached in this task (0.324 m)
    MEET_FULL = 1.0 / 3.0
    W_DIR_PEN = 1.0         # pouring AWAY from the torso is a workspace charge on d_hat_x, not on any
                            # cup attitude - kept
    NEST_SAFE, NEST_ZERO = 0.115, 0.09
    W_NEAR_NEST = 1.0
    # --- receiver discipline - KEEP LIST, unchanged from iter_16 -------------------------------------
    W_HOME_RCV = 0.4
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5
    STILL_FREE, STILL_SCALE = 0.05, 0.15
    # --- THE BEADS: the only thing that pays for a pour ---------------------------------------------
    W_POUR = 150.0          # x d_in_target: 7.5 per bead with 20 beads. UNCHANGED (operator note 6).
    W_SPILL = 240.0         # -12 per bead spilled. UNCHANGED (keep list).
    SETTLE_FRAC = 0.03      # spill charge off for the first 3 % of the episode (beads settling at spawn)
    W_SUCCESS = 20.0        # unchanged; every guidance term stays well below it
    SPILL_ZERO = 0.25       # the env's spill limit (unchanged)
    W_TRANSFER = 8.0        # iter_17: 5.0 -> 8.0. At the ep2100 deterministic replay (in_target 0.150,
                            # spill 0.050 -> clean_f 0.80) this pays 0.96/step, about half of the 2.07
                            # of attitude income just deleted - and unlike it, zero unless beads are
                            # actually inside the receiver.
    TAIL_LO, TAIL_HI = 0.10, 0.90
                            # iter_17: TAIL_LO 0.35 -> 0.10. diag_fill_tail was 0.0000 for the entire
                            # i16g run because 0.35 is above anything reached; 0.10 sits BELOW the
                            # measured 0.150 so the ramp is live (+0.13/step) from the first step.
                            # TAIL_HI 0.90 unchanged, still above the env's 0.80 success threshold.
    W_TAIL = 4.0
    EMPTY_LO = 0.20         # iter_17: 0.45 -> 0.20. Measured emptied = 1 - 0.756 = 0.244 at ep2100, so
                            # 0.45 was unreachable (diag_empty_q 0.006). 0.20 is just under it.
    W_EMPTY = 3.0
    W_SETTLE = 3.0
    FLOW_CAP_BEADS = 1.0    # paid per step at most 1 bead (unchanged)
    W_FLOW = 40.0           # -2 per extra bead in the same step, and it is unpaid (unchanged)
    W_UNAIMED = 60.0        # iter_17: 3.0 -> 60.0 because the quantity changed from a dimensionless
                            # angle window to a BEAD FRACTION. In per-bead units (20 beads) this is
                            # -3 per bead released with the spout outside the permission window,
                            # against pour +7.5 and spill -12: releasing unaimed is never profitable,
                            # and it stays well under the spill charge so "where the bead ended up"
                            # remains the dominant signal.
    # cup motion limits - KEEP LIST (user principle: never buy SIM efficiency by loosening these)
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, ROT_POUR_SCALE, W_ROT_POUR = 1.0, 1.5, 1.0   # stricter while the spout is aimed
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
    UPRIGHT_TOL = 0.35      # rad (20 deg) - the ENV's own receiver-upright success limit. iter_17
                            # reuses it for the SOURCE only while the source cup is still ON THE TABLE
                            # (a knock-over guard). Once the cup is carried the reward has no opinion
                            # about its attitude at all.
    MARGIN_FREE = 0.30
    W_ARM_LIMIT = 0.4       # KEEP LIST, rudder
    QDD_FREE, QDD_SCALE = 2.5, 10.3
    QDD_PEAK_FREE, QDD_PEAK_SCALE = 12.8, 12.8
    W_ARM_SHAKE = 0.4       # KEEP LIST, rudder

    dtype = ctx.src_cup_tilt.dtype
    geo = _pour_geometry(ctx, ctx.cup_radius, ctx.cup_mouth_z, POUR_CLEARANCE, SEP_MIN)
    d_align = geo["d_align"]

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)
    not_nested = 1.0 - nested
    carry_soft_src = g_src * torch.clamp(h_src / TILT_LIFT_H, max=1.0)
    carry_soft_rcv = g_rcv * torch.clamp(h_rcv / TILT_LIFT_H, max=1.0)

    # ---- PERMISSION, all of it in cup_radius multiples ---------------------------------------------
    r_mouth = ctx.cup_radius
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r_mouth)
                                  / ((LEAD_ZERO_R - LEAD_FULL_R) * r_mouth))
    perm_xy = 1.0 - _smoothstep((geo["lip_xy"] - PERM_IN_R * r_mouth)
                                / ((PERM_OUT_R - PERM_IN_R) * r_mouth))
    lip_above_raw = torch.clamp(geo["lip_dz"] / (LIP_Z_HI_R * r_mouth), 0.0, 1.0)
    # direction as a SIGN GATE ONLY: tilted toward the receiver or not. Turning FURTHER toward it buys
    # nothing - there is no value on the magnitude anywhere in this function.
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = perm_xy * lip_above_raw * toward

    # ---- attitude: the ONLY remaining use is the knock-over guard on a cup still on the table -------
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)   # env success criterion, keep list
    pose_ok_rcv = upright_rcv
    src_table_upright = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    # == 1 as soon as the source cup is carried: no attitude prescription in the air, ever.
    pose_ok_src = carry_soft_src + (1.0 - carry_soft_src) * src_table_upright

    # ---- rendezvous corridor (KEEP LIST, verbatim iter_16) -----------------------------------------
    span_y = ctx.rcv_cup_spawn_pos[:, 1] - ctx.src_cup_spawn_pos[:, 1]
    span_sgn = torch.where(span_y >= 0.0, torch.ones_like(span_y), -torch.ones_like(span_y))
    span_safe = span_sgn * torch.clamp(span_y.abs(), min=SPAN_MIN)   # signed, never zero
    frac_rcv = (ctx.rcv_cup_pos[:, 1] - ctx.src_cup_spawn_pos[:, 1]) / span_safe   # 1 = its own spawn
    frac_src = (ctx.src_cup_pos[:, 1] - ctx.src_cup_spawn_pos[:, 1]) / span_safe   # 0 = its own spawn
    meet_q = _late_ramp(1.0 - frac_rcv, 0.0, 1.0 - MEET_FULL)
    src_cross = torch.clamp((frac_src - MEET_FULL) / (1.0 - MEET_FULL), 0.0, 1.0)
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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # measured rcv_progress 0.79

    # ---- source lift: band + linear cost above it, free height tied to the LIP DROP ----------------
    # iter_17: the `src_up_soft` attitude factor is REMOVED - lifting is no longer scored against an
    # attitude corridor. lip_drop stays: it is a geometric height allowance (how far the spout has
    # fallen), not a target for the angle.
    lift_free_src = LIFT_FREE_SRC0 + geo["lip_drop"]
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_lift_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_lift_band * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])
    # ungated by the grasp flag on purpose: throwing the cup up must not dodge the cost
    over_lift = torch.clamp((h_src - lift_free_src) / OVER_LIFT_RANGE, 0.0, 1.0)

    # ---- receiver lift band (unchanged - measured 0.073 m, inside the 0.10 m band) ------------------
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * rcv_band * (0.5 + 0.5 * upright_rcv)

    # ---- receiver discipline, released once it is CARRYING (iter_16, KEEP LIST) ---------------------
    d_home = rcv["disp"]
    home_near = torch.exp(-(d_home / HOME_SIGMA) ** 2)
    home_eff = home_near + (1.0 - home_near) * carry_soft_rcv
    still_free_eff = STILL_FREE + (SRC_V_FREE - STILL_FREE) * (1.0 - meet_q)
    still_over_rcv = torch.tanh(torch.clamp(rcv["v_lin"] - still_free_eff, min=0.0) / STILL_SCALE)
    still_f = 1.0 - still_over_rcv
    rcv_disc = RCV_DISC_FLOOR + (1.0 - RCV_DISC_FLOOR) * rcv_band * home_eff
    rcv_home_pen = rcv["held"] * (1.0 - carry_soft_rcv) \
        * torch.tanh(torch.clamp(d_home - HOME_FREE, min=0.0) / HOME_SCALE)
    still_w = W_STILL_RCV + W_STILL_POUR * up_w_src * meet_q   # strict at the rendezvous, not en route
    rcv_still_pen = still_w * rcv["held"] * still_over_rcv
    hold_rcv = W_HOLD_RCV * g_rcv * rcv_band * home_eff * still_f * (0.4 + 0.6 * carry_soft_src)

    # ---- convergence / carry / align (KEEP LIST, unchanged) ----------------------------------------
    d_cups_xy = (ctx.src_cup_pos[:, :2] - ctx.rcv_cup_pos[:, :2]).norm(dim=-1)
    conv_lin = torch.clamp((CONV_FAR - d_cups_xy) / (CONV_FAR - CONV_NEAR), 0.0, 1.0)
    conv_close = torch.exp(-(torch.clamp(d_cups_xy - CONV_NEAR, min=0.0) / CONV_SIGMA) ** 2)
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.82 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * not_nested * conv_shape

    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * not_nested
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * not_nested
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- iter_17 NEW: hold the grasp you lifted with, all through the carry and the pour ------------
    grip_hold_src = W_GRIP_HOLD_SRC * carry_soft_src * src["grip_keep"]
    grip_hold_rcv = W_GRIP_HOLD_RCV * carry_soft_rcv * rcv["grip_keep"]

    # ---- THE BEADS: flow-limited increments, spill priced above everything it can buy ---------------
    n_beads = float(max(int(ctx.num_beads), 1))
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)              # 1 bead per step paid in full
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)        # the rest of a dump: unpaid + charged
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_step = settled * torch.clamp(ctx.d_spill, min=0.0)

    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    spill_lvl = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    clean_f = torch.clamp(1.0 - spill_lvl / SPILL_ZERO, 0.0, 1.0)   # 1 clean, 0 at the success limit
    emptied = torch.clamp(1.0 - fill, 0.0, 1.0)
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)
    # a source emptied ONTO THE TABLE must pay nothing: clean_f is already 0 at the 0.25 spill limit,
    # and fill_ok additionally requires the beads to have arrived somewhere useful
    fill_ok = torch.clamp(in_t / TAIL_LO, max=1.0)
    # MEAN of the two ramps, not min: a min-based term is flat whenever the two are tied
    finish_q = 0.5 * (fill_tail + empty_q)
    has_beads = torch.clamp(fill / 0.10, max=1.0)               # >= 2 beads left

    # ---- iter_17: "unaimed release" is now triggered by an ACTUAL RELEASE, not by an angle window ---
    # Beads that left the source this step, counted by where they arrived. If they left while the
    # spout was outside the permission window, charge -3 per bead on top of whatever spill costs.
    release_step = torch.clamp(d_in, min=0.0) + torch.clamp(ctx.d_spill, min=0.0)
    unaimed = settled * release_step * (1.0 - perm)

    # ---- approach: stateless potential difference on lip_xy, independent of the tilt (KEEP LIST) ----
    lip_rate = _lip_closing_rate(ctx, geo["lip_u"], geo["lip_pt"])
    outside_win = (geo["lip_xy"] > PERM_IN_R * r_mouth).to(dtype)
    approach_gate = carry_soft_src * rcv["held"] * not_nested * has_beads
    approach_lip = approach_gate * outside_win * torch.clamp(-lip_rate / SRC_V_FREE, -1.0, 1.0)

    # ---- cup motion limits (KEEP LIST: these are what the real robot needs) ------------------------
    v_src = src["v_lin"]
    w_src = ctx.src_cup_ang_vel.norm(dim=-1)
    w_rcv = ctx.rcv_cup_ang_vel.norm(dim=-1)
    ctrl_gate = up_w_src * up_w_rcv                      # both cups grasped and >= 5.5 cm up
    # iter_17: rot_pour's thresholds and weight are untouched; only its GATE changed, because the old
    # gate (aim_frac) was a fraction of a prescribed angle. It is now purely the spout window.
    pour_phase = ctrl_gate * lead_gate * perm_xy
    src_speed = ctrl_gate * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = ctrl_gate * torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE)
    rot_pour = pour_phase * torch.tanh(torch.clamp(w_src - ROT_POUR_FREE, min=0.0) / ROT_POUR_SCALE)
    rcv_spin = ctrl_gate * torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE)

    # ---- arm posture and tremor: kept as rudders (KEEP LIST) ---------------------------------------
    margin_pen_src = _limit_pen(ctx.src_arm_limit_margin, MARGIN_FREE)
    margin_pen_rcv = _limit_pen(ctx.rcv_arm_limit_margin, MARGIN_FREE)
    arm_limit_src = g_src * margin_pen_src
    arm_limit_rcv = g_rcv * margin_pen_rcv
    shake_src = g_src * _shake_pen(ctx.src_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)
    shake_rcv = g_rcv * _shake_pen(ctx.rcv_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)

    # ---- finish: both cups still and still held, AFTER the transfer --------------------------------
    # iter_17: `upright_src` is REMOVED from this product - how the source ends up pointing is not the
    # reward's business. upright_rcv stays because the ENV's own success test requires it (<= 20 deg).
    v_still_src = 1.0 - torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    w_still_src = 1.0 - torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE)
    w_still_rcv = 1.0 - torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE)
    settle_still = 0.25 * (v_still_src + w_still_src + still_f + w_still_rcv)
    # HARD grasp product on purpose: dropping the cups must never buy stillness. Gated on finish_q, so
    # it is unreachable before the transfer is essentially done - it is the hold, not parking.
    settle_fin = finish_q * g_src * g_rcv * not_nested * upright_rcv * settle_still

    # ---- directional / nesting penalties (no cup-attitude magnitude among them any more) ------------
    pour_dir_pen = carry_soft_src * _smoothstep((geo["d_hat_x"] - 0.10) / 0.25)
    cup_center_d = (ctx.src_cup_pos - ctx.rcv_cup_pos).norm(dim=-1)
    near_nest = _smoothstep((NEST_SAFE - cup_center_d) / (NEST_SAFE - NEST_ZERO))
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)   # receiver discipline, keep list

    # ---- command smoothness. Layout: [0:6] src palm, [6:12] src hand, [12:18] rcv palm,
    #      [18:24] rcv hand. (KEEP LIST, all verbatim) ------------------------------------------------
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)
    a_abs = ctx.actions.abs()
    sat_src_raw = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv_raw = torch.mean(_smoothstep((a_abs[:, 12:18] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
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
        # ★ iter_17: close_src is no longer multiplied by an attitude corridor, and its weight matches
        # the receiver's. It now stays alive through the rotation (measured close_src 0.087/step).
        "close_src": W_CLOSE_SRC * src["close"],
        "close_rcv": W_CLOSE_RCV * rcv["close"],
        "grasp_src": W_GRASP * src["grasp"],
        "grasp_rcv": W_GRASP * rcv["grasp"],
        # ★ iter_17 NEW: keep the contact DISTRIBUTION that lifted the cup, for as long as it is
        # carried. Ratio-based, so no absolute force level is prescribed anywhere (i24 rule).
        "grip_hold_src": grip_hold_src,
        "grip_hold_rcv": grip_hold_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "over_lift_src": -W_OVER_LIFT * over_lift,
        "hold_rcv": hold_rcv,
        "meet_cross": meet_cross,
        "src_cross": -W_SRC_CROSS * carry_soft_src * src_cross,
        "converge": converge,
        "carry_src": W_CARRY * carry,
        "align_mouths": W_ALIGN * align,
        # ★ iter_17: pouring income. There is NOTHING between align_mouths and these four terms any
        # more - no angle ladder, no attitude plateau. The policy is paid when beads move and for
        # where the beads are, and for nothing else.
        "pour_delta": W_POUR * pour_paid,                                 # +7.5 per bead delivered
        "flow_excess": -W_FLOW * flow_excess,                             # -2 per bead dumped at once
        "transferred": W_TRANSFER * in_t * not_nested * upright_rcv * clean_f,
        "transfer_tail": W_TAIL * fill_tail * not_nested * upright_rcv * clean_f,
        "src_emptied": W_EMPTY * empty_q * fill_ok * not_nested * clean_f,
        "settle_finish": W_SETTLE * settle_fin,
        "spill_delta": -W_SPILL * spill_step,                             # -12 per bead spilled
        "success_bonus": W_SUCCESS * ctx.success.to(dtype) * clean_f,
        # ★ iter_17: fires on an ACTUAL release outside the permission window (was: on an angle
        # window). -3 per bead, so releasing unaimed can never pay.
        "unaimed_release": -W_UNAIMED * unaimed,
        "approach_lip": W_APPROACH * approach_lip,
        "src_speed": -W_SRC_SPEED * src_speed,
        "src_spin": -W_SRC_SPIN * src_spin,
        "rot_pour": -W_ROT_POUR * rot_pour,
        "rcv_spin": -W_RCV_SPIN * rcv_spin,
        "arm_limit_src": -W_ARM_LIMIT * arm_limit_src,
        "arm_limit_rcv": -W_ARM_LIMIT * arm_limit_rcv,
        "arm_shake_src": -W_ARM_SHAKE * shake_src,
        "arm_shake_rcv": -W_ARM_SHAKE * shake_rcv,
        "act_flip": -W_FLIP_PALM * (g_src * flip_palm_src + g_rcv * flip_palm_rcv),
        "hand_flip": -W_FLIP_HAND * ctrl_gate * flip_hand,
        "carry_rate": -(W_RATE_PALM_SRC * g_src * rate_palm_src
                        + W_RATE_PALM_RCV * g_rcv * rate_palm_rcv
                        + W_RATE_HAND_CARRY * ctrl_gate * rate_hand),
        "palm_bang_src": -W_BANG_SRC * g_src * bang_src,
        "palm_bang_rcv": -W_BANG_RCV * g_rcv * bang_rcv,
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
    # diagnostics only (added AFTER the sum, not part of the reward). The tilt angles are LOGGED so
    # the round can be read, but nothing above reads them as a target.
    comps["rcv_progress"] = rcv_prog
    comps["aim_gate"] = perm_xy          # the r-window
    comps["diag_perm"] = perm            # window x above the rim x tilted-toward sign
    comps["diag_toward"] = toward
    comps["diag_src_tilt_deg"] = ctx.src_cup_tilt * (180.0 / math.pi)   # FREE from iter_17 on
    comps["diag_theta_deg"] = geo["theta_eff"] * (180.0 / math.pi)
    comps["diag_mis_deg"] = geo["mis"] * (180.0 / math.pi)
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_lift_budget"] = lift_free_src
    comps["diag_lead_gate"] = lead_gate
    comps["diag_palm_sat"] = sat_src_raw
    comps["diag_flow_beads"] = torch.clamp(d_in, min=0.0) * n_beads
    # ★ iter_17: the two numbers this round is judged on - beads leaving the source per step, and how
    # many of them left with the spout outside the permission window.
    comps["diag_release_beads"] = release_step * n_beads
    comps["diag_unaimed_beads"] = unaimed * n_beads
    comps["diag_clean"] = clean_f
    comps["diag_ctrl_gate"] = ctrl_gate
    comps["diag_pour_phase"] = pour_phase
    comps["diag_src_w"] = w_src
    comps["diag_src_v"] = v_src
    comps["diag_flip_palm"] = flip_palm
    comps["diag_palm_sat_rcv"] = sat_rcv_raw
    comps["diag_bang_src"] = bang_src
    comps["diag_bang_rcv"] = bang_rcv
    comps["diag_close_speed"] = torch.clamp(-lip_rate, -1.0, 1.0)
    comps["diag_fill_tail"] = fill_tail
    comps["diag_empty_q"] = empty_q
    comps["diag_finish_q"] = finish_q
    comps["diag_settle_still"] = settle_still
    comps["diag_emptied"] = emptied
    # ★ iter_17: grasp survival through the pour (the operator's "the grasp looks unstable")
    comps["diag_grip_keep_src"] = src["grip_keep"]
    comps["diag_grip_keep_rcv"] = rcv["grip_keep"]
    comps["diag_balance_src"] = src["balance"]
    comps["diag_src_margin_min"] = torch.clamp(ctx.src_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_rcv_margin_min"] = torch.clamp(ctx.rcv_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_src_limit_hit"] = (comps["diag_src_margin_min"] < 0.05).to(dtype)   # i16g: <= 0.05
    comps["diag_rcv_limit_hit"] = (comps["diag_rcv_margin_min"] < 0.05).to(dtype)
    comps["diag_src_qdd"] = ctx.src_arm_qdd.abs().mean(dim=-1)
    comps["diag_rcv_qdd"] = ctx.rcv_arm_qdd.abs().mean(dim=-1)
    comps["diag_src_qdd_max"] = ctx.src_arm_qdd.abs().max(dim=-1).values
    comps["diag_rcv_qdd_max"] = ctx.rcv_arm_qdd.abs().max(dim=-1).values
    comps["diag_rate_palm_rcv"] = rate_palm_rcv
    # rendezvous, in spawn-relative units (0 = source spawn side, 1 = receiver spawn side)
    comps["diag_frac_rcv"] = frac_rcv            # i16g at ep2100: 0.57-0.63
    comps["diag_frac_src"] = frac_src            # i16g at ep2100: 0.11-0.16
    comps["diag_meet_q"] = meet_q
    comps["diag_home_eff"] = home_eff
    comps["diag_rcv_travel"] = d_home
    comps["diag_src_travel"] = src["disp"]
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0.00011, 0.000232, 0.000513, 0.000488, 0.000208, 0.000378, 0.000281, 0.000269, 0.000378]  min 0 · mean 0.000303 · max 0.000977
bead/spill: [1.22e-05, 0.0167, 0.0185, 0.0286, 0.0169, 0.00811, 0.0345, 0.0205, 0.0201, 0.0197]  min 1.22e-05 · mean 0.0196 · max 0.0422
done/drop: [0, 0.000488, 0.000488, 0.000244, 0, 0.000244, 0, 0.000488, 0.000244, 0.000488]  min 0 · mean 0.000337 · max 0.00171
done/mimic_runaway: [0, 0.000244, 0, 0.000244, 0, 0, 0.000488, 0, 0.000244, 0]  min 0 · mean 0.00014 · max 0.00122
episode_lengths/step: [90.4, 764, 764, 727, 751, 850, 728, 705, 688, 722]  min 90.4 · mean 750 · max 872
reward/act_flip: [0, -0.156, -0.115, -0.0868, -0.0343, -0.0279, -0.0213, -0.0257, -0.0196, -0.0124]  min -0.179 · mean -0.0584 · max 0
reward/action_rate: [-0.0319, -0.0216, -0.0168, -0.0148, -0.0125, -0.0109, -0.011, -0.00958, -0.00795, -0.0066]  min -0.0319 · mean -0.0128 · max -0.00596
reward/aim_gate: [0, 0.436, 0.196, 0.375, 0.337, 0.705, 0.461, 0.456, 0.548, 0.471]  min 0 · mean 0.405 · max 0.709
reward/align_mouths: [0, 2.61, 2.43, 2.41, 2.67, 3.53, 2.68, 2.65, 2.61, 2.75]  min 0 · mean 2.66 · max 3.53
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/approach_lip: [0, 0.0146, 0.0457, 0.0687, 0.0925, 0.0682, 0.106, 0.0656, 0.0568, 0.0293]  min -0.187 · mean 0.0541 · max 0.117
reward/arm_limit_rcv: [0, -0.0138, -0.0455, -0.0622, -0.0749, -0.0823, -0.0922, -0.0616, -0.0692, -0.0826]  min -0.0963 · mean -0.0614 · max 0
reward/arm_limit_src: [0, -0.107, -0.0868, -0.0614, -0.104, -0.133, -0.126, -0.0969, -0.0984, -0.105]  min -0.136 · mean -0.101 · max 0
reward/arm_shake_rcv: [0, -0.0905, -0.0871, -0.0814, -0.102, -0.115, -0.0984, -0.0836, -0.0723, -0.0735]  min -0.126 · mean -0.0898 · max 0
reward/arm_shake_src: [0, -0.0958, -0.0707, -0.0706, -0.0716, -0.0727, -0.0808, -0.0635, -0.0581, -0.056]  min -0.114 · mean -0.0709 · max 0
reward/away_rcv: [-0.103, -0.178, -0.169, -0.18, -0.177, -0.172, -0.172, -0.164, -0.162, -0.153]  min -0.185 · mean -0.168 · max -0.103
reward/carry_rate: [0, -0.197, -0.154, -0.13, -0.0914, -0.0821, -0.0735, -0.0685, -0.0567, -0.0431]  min -0.24 · mean -0.102 · max 0
reward/carry_src: [0, 1.69, 1.57, 1.65, 1.87, 2.11, 1.77, 1.77, 1.8, 1.77]  min 0 · mean 1.77 · max 2.13
reward/close_rcv: [0.0685, 0.0774, 0.0743, 0.0785, 0.0823, 0.084, 0.0862, 0.0901, 0.0945, 0.0874]  min 0.0666 · mean 0.0835 · max 0.0989
reward/close_src: [0.0827, 0.212, 0.242, 0.204, 0.172, 0.138, 0.133, 0.173, 0.151, 0.161]  min 0.0827 · mean 0.174 · max 0.259
reward/contact_push: [-0.00637, -0.0023, -0.00133, -0.00156, -0.00156, -0.00197, -0.00267, -0.00302, -0.00276, -0.0031]  min -0.118 · mean -0.00256 · max -0.000717
reward/converge: [0, 2.05, 2.04, 2.07, 2.26, 2.46, 2.18, 2.17, 2.13, 2.18]  min 0 · mean 2.17 · max 2.47
reward/cup_collision: [0, -0.000354, -0.00122, -0.00265, -0.0063, -0.00625, -0.00519, -0.00182, -0.00388, -0.00434]  min -0.0201 · mean -0.00352 · max 0
reward/cup_disturb_rcv: [-0.026, -0.0831, -0.0946, -0.087, -0.0768, -0.0716, -0.0857, -0.0881, -0.0827, -0.0902]  min -0.228 · mean -0.0836 · max -0.026
reward/cup_disturb_src: [-0.0561, -0.0206, -0.0227, -0.0191, -0.0139, -0.0107, -0.014, -0.0166, -0.0175, -0.0216]  min -0.261 · mean -0.0194 · max -0.00716
reward/cup_fling: [-0.000473, -0.0107, -0.0103, -0.0131, -0.0137, -0.00778, -0.0142, -0.0124, -0.0146, -0.013]  min -0.0169 · mean -0.0119 · max -0.000473
reward/diag_balance_src: [0, 0.653, 0.619, 0.615, 0.654, 0.698, 0.585, 0.593, 0.602, 0.61]  min 0 · mean 0.626 · max 0.725
reward/diag_bang_rcv: [0.273, 0.236, 0.21, 0.169, 0.0939, 0.0733, 0.0704, 0.0791, 0.0668, 0.0434]  min 0.0321 · mean 0.115 · max 0.297
reward/diag_bang_src: [0.193, 0.147, 0.107, 0.0835, 0.0619, 0.0628, 0.044, 0.0389, 0.0323, 0.033]  min 0.0273 · mean 0.0716 · max 0.193
reward/diag_clean: [1, 0.968, 0.957, 0.939, 0.959, 0.983, 0.923, 0.948, 0.952, 0.952]  min 0.906 · mean 0.956 · max 1
reward/diag_close_speed: [0.0124, 0.00261, 0.00363, 0.00622, 0.0259, 0.0238, 0.021, 0.0222, 0.013, 0.0138]  min -0.0456 · mean 0.0135 · max 0.0303
reward/diag_ctrl_gate: [0, 0.716, 0.696, 0.658, 0.699, 0.817, 0.688, 0.674, 0.645, 0.705]  min 0 · mean 0.703 · max 0.822
reward/diag_emptied: [3.66e-05, 0.0202, 0.0244, 0.0366, 0.0228, 0.0109, 0.0429, 0.0272, 0.0273, 0.0264]  min 3.66e-05 · mean 0.0251 · max 0.0536
reward/diag_empty_q: [0, 0.0139, 0.0152, 0.0242, 0.013, 0.00722, 0.0256, 0.0148, 0.0161, 0.0152]  min 0 · mean 0.0153 · max 0.0328
reward/diag_fill_tail: [0, 8.11e-06, 1.72e-05, 8.49e-05, 4.34e-05, 0, 0, 8.11e-06, 0, 8.11e-06]  min 0 · mean 1.44e-05 · max 0.000297
reward/diag_finish_q: [0, 0.00696, 0.00759, 0.0122, 0.00654, 0.00361, 0.0128, 0.0074, 0.00803, 0.00761]  min 0 · mean 0.00767 · max 0.0164
reward/diag_flip_palm: [1.21, 0.663, 0.488, 0.373, 0.168, 0.126, 0.108, 0.118, 0.0971, 0.0656]  min 0.05 · mean 0.262 · max 1.21
reward/diag_flow_beads: [0, 0, 0, 0, 0, 0, 0, 0.000244, 0, 0]  min 0 · mean 6.56e-05 · max 0.000977
reward/diag_frac_rcv: [0.999, 0.741, 0.604, 0.596, 0.427, 0.359, 0.433, 0.396, 0.412, 0.385]  min -0.229 · mean 0.514 · max 1.42
reward/diag_frac_src: [0.00137, -0.246, -0.524, -0.535, 0.144, 0.221, 0.188, -0.323, -0.34, -0.104]  min -1.8 · mean 0.041 · max 14.5
reward/diag_grip_keep_rcv: [0, 0.134, 0.133, 0.126, 0.137, 0.142, 0.147, 0.174, 0.18, 0.18]  min 0 · mean 0.15 · max 0.207
reward/diag_grip_keep_src: [0, 0.416, 0.407, 0.329, 0.292, 0.239, 0.208, 0.264, 0.227, 0.243]  min 0 · mean 0.292 · max 0.464
reward/diag_home_eff: [1, 0.973, 0.971, 0.957, 0.974, 0.983, 0.965, 0.969, 0.954, 0.972]  min 0.937 · mean 0.973 · max 1
reward/diag_lead_gate: [0, 0.571, 0.413, 0.529, 0.606, 0.779, 0.657, 0.649, 0.712, 0.676]  min 0 · mean 0.585 · max 0.781
reward/diag_lift_budget: [0.24, 0.248, 0.256, 0.253, 0.244, 0.247, 0.257, 0.251, 0.249, 0.246]  min 0.24 · mean 0.249 · max 0.259
reward/diag_lip_dz: [-2.63e-05, 0.158, 0.246, 0.233, 0.115, 0.111, 0.119, 0.193, 0.17, 0.254]  min -2.09 · mean 0.147 · max 0.471
reward/diag_lip_xy: [0.291, 0.406, 0.465, 0.465, 0.122, 0.0801, 0.123, 0.39, 0.277, 0.266]  min 0.0767 · mean 0.235 · max 4.97
reward/diag_meet_q: [0.000863, 0.328, 0.544, 0.507, 0.754, 0.829, 0.76, 0.778, 0.774, 0.781]  min 0.000863 · mean 0.638 · max 0.836
reward/diag_mis_deg: [0.535, 4.99, 6.32, 5.92, 12.3, 9.3, 13.3, 5.16, 9.36, 15.5]  min 0.535 · mean 9.59 · max 20.1
reward/diag_palm_sat: [0.44, 0.336, 0.379, 0.409, 0.338, 0.306, 0.351, 0.325, 0.323, 0.319]  min 0.282 · mean 0.348 · max 0.58
reward/diag_palm_sat_rcv: [0.532, 0.346, 0.381, 0.385, 0.437, 0.484, 0.45, 0.42, 0.44, 0.464]  min 0.324 · mean 0.429 · max 0.532
reward/diag_perm: [0, 0.35, 0.192, 0.365, 0.143, 0.477, 0.408, 0.433, 0.432, 0.311]  min 0 · mean 0.297 · max 0.526
reward/diag_pour_phase: [0, 0.414, 0.187, 0.342, 0.301, 0.695, 0.398, 0.409, 0.46, 0.427]  min 0 · mean 0.371 · max 0.695
reward/diag_rate_palm_rcv: [4.38, 2.89, 2.39, 1.97, 1.21, 0.921, 0.945, 0.991, 0.842, 0.587]  min 0.496 · mean 1.45 · max 4.38
reward/diag_rcv_limit_hit: [0, 0, 0.00342, 0.000732, 0.00513, 0.00781, 0.00415, 0.00513, 0.00635, 0.012]  min 0 · mean 0.00485 · max 0.0251
reward/diag_rcv_margin_min: [0.245, 0.244, 0.19, 0.167, 0.152, 0.147, 0.136, 0.164, 0.156, 0.145]  min 0.131 · mean 0.171 · max 0.322
reward/diag_rcv_qdd: [6.26, 5.96, 6.23, 6.16, 6.77, 7.18, 6.81, 6.16, 5.76, 5.74]  min 5.41 · mean 10.3 · max 9.13e+03
reward/diag_rcv_qdd_max: [14.7, 14.3, 14.6, 14.9, 16.1, 17.1, 16.3, 14.4, 13.9, 13.9]  min 13 · mean 25.7 · max 2.29e+04
reward/diag_rcv_travel: [0.000662, 0.118, 0.145, 0.15, 0.22, 0.263, 0.223, 0.226, 0.219, 0.211]  min 0.000662 · mean 0.199 · max 0.473
reward/diag_release_beads: [0, 0.00415, 0.00635, 0.00708, 0.00317, 0.00293, 0.00879, 0.00464, 0.00439, 0.00879]  min 0 · mean 0.00473 · max 0.0132
reward/diag_settle_still: [0.976, 0.755, 0.776, 0.752, 0.748, 0.796, 0.745, 0.763, 0.745, 0.76]  min 0.546 · mean 0.759 · max 0.976
reward/diag_src_limit_hit: [0, 0.00708, 0.0176, 0.00366, 0.0146, 0.0129, 0.0952, 0.0403, 0.0381, 0.0364]  min 0 · mean 0.0323 · max 0.141
reward/diag_src_margin_min: [0.193, 0.121, 0.134, 0.156, 0.122, 0.102, 0.099, 0.122, 0.121, 0.116]  min 0.0956 · mean 0.121 · max 0.193
reward/diag_src_qdd: [6.11, 7.07, 6.32, 6.79, 6.39, 5.55, 7.33, 6.01, 6.1, 5.64]  min 5.06 · mean 7.17 · max 906
reward/diag_src_qdd_max: [13.8, 17.5, 16.5, 18, 16.9, 14.6, 19.5, 15.9, 16, 15.2]  min 12.9 · mean 19 · max 2.32e+03
reward/diag_src_tilt_deg: [1, 16.7, 28.4, 23.6, 17.9, 20, 35.5, 21.6, 22.5, 24.1]  min 1 · mean 21.9 · max 36.9
reward/diag_src_travel: [0.00115, 0.392, 0.399, 0.443, 0.126, 0.176, 0.14, 0.384, 0.291, 0.264]  min 0.00115 · mean 0.229 · max 4.97
reward/diag_src_v: [0.0359, 0.285, 0.351, 0.287, 0.188, 0.145, 0.216, 0.293, 0.27, 0.217]  min 0.0359 · mean 0.257 · max 23.9
reward/diag_src_w: [0.499, 1.4, 1.31, 1.5, 1.31, 0.983, 1.73, 1.38, 1.51, 1.45]  min 0.499 · mean 2.67 · max 452
reward/diag_theta_deg: [0.466, 11.7, 22.1, 17.7, 5.59, 10.7, 22.2, 16.4, 13.1, 8.58]  min 0.466 · mean 12.3 · max 25.4
reward/diag_toward: [0.555, 0.765, 0.876, 0.846, 0.427, 0.624, 0.745, 0.858, 0.728, 0.552]  min 0.326 · mean 0.687 · max 0.91
reward/diag_unaimed_beads: [0, 0.00391, 0.00635, 0.00639, 0.00211, 0.00282, 0.00683, 0.00384, 0.00371, 0.00797]  min 0 · mean 0.00413 · max 0.0122
reward/flow_excess: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.000977 · mean -3e-06 · max 0
reward/grasp_rcv: [0.000579, 1.32, 1.12, 1.28, 1.35, 1.39, 1.33, 1.26, 1.3, 1.11]  min 0.000579 · mean 1.26 · max 1.48
reward/grasp_src: [0.00486, 1.44, 1.42, 1.41, 1.49, 1.59, 1.45, 1.4, 1.31, 1.35]  min 0.00486 · mean 1.41 · max 1.71
reward/grip_hold_rcv: [0, 0.0798, 0.0789, 0.0735, 0.0802, 0.0845, 0.0865, 0.103, 0.103, 0.107]  min 0 · mean 0.0887 · max 0.123
reward/grip_hold_src: [0, 0.495, 0.487, 0.394, 0.349, 0.286, 0.249, 0.316, 0.272, 0.29]  min 0 · mean 0.348 · max 0.553
reward/hand_flip: [0, -0.108, -0.0703, -0.0559, -0.0635, -0.0649, -0.0551, -0.0398, -0.0305, -0.0262]  min -0.195 · mean -0.0599 · max 0
reward/hand_foreign: [-0.0186, -0.00136, -0.00157, -0.00265, -0.00425, -0.00319, -0.00659, -0.00243, -0.00485, -0.00548]  min -0.0186 · mean -0.00348 · max -0.000416
reward/hold_rcv: [0, 0.589, 0.547, 0.56, 0.519, 0.645, 0.542, 0.539, 0.505, 0.55]  min 0 · mean 0.556 · max 0.699
reward/lift_rcv: [0, 1.38, 1.32, 1.25, 1.3, 1.57, 1.31, 1.27, 1.27, 1.34]  min 0 · mean 1.35 · max 1.58
reward/lift_src: [0, 2.3, 2.41, 2.36, 2.59, 2.82, 2.5, 2.46, 2.48, 2.37]  min 0 · mean 2.46 · max 2.87
reward/meet_cross: [0, 1.35, 2.17, 2.02, 3.24, 3.72, 3.09, 3.17, 3.14, 3.18]  min 0 · mean 2.66 · max 3.75
reward/near_nested: [0, -0.00431, -0.00765, -0.00822, -0.00934, -0.0035, -0.0142, -0.00574, -0.00964, -0.00979]  min -0.0305 · mean -0.00775 · max 0
reward/nested: [0, -0.00195, -0.00537, -0.00293, -0.0112, -0.00195, -0.0166, -0.00439, -0.0117, -0.0122]  min -0.0244 · mean -0.00706 · max 0
reward/over_lift_src: [0, -0.11, -0.0503, -0.0575, -0.0221, -0.00966, -0.0315, -0.039, -0.0282, -0.0736]  min -0.243 · mean -0.0464 · max 0
reward/palm_bang_rcv: [0, -0.392, -0.341, -0.266, -0.132, -0.107, -0.0962, -0.115, -0.096, -0.0575]  min -0.419 · mean -0.175 · max 0
reward/palm_bang_src: [0, -0.167, -0.117, -0.0897, -0.0638, -0.074, -0.0387, -0.0316, -0.024, -0.0275]  min -0.204 · mean -0.0747 · max 0
reward/palm_sat_rcv: [0, -0.39, -0.437, -0.445, -0.544, -0.635, -0.554, -0.514, -0.536, -0.57]  min -0.645 · mean -0.521 · max 0
reward/palm_sat_src: [0, -0.764, -0.874, -0.956, -0.829, -0.789, -0.844, -0.787, -0.758, -0.753]  min -1.61 · mean -0.824 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0.00183, 0, 0]  min -0.00366 · mean 0.000321 · max 0.00732
reward/pour_dir_pen: [0, -0.12, -0.00629, -0.0821, -0.0129, -0.117, -0.0131, -0.0167, -0.152, -0.302]  min -0.898 · mean -0.129 · max 0
reward/rcv_home: [0, -1.73e-05, -0.000561, -0.00248, -0.00268, -0.000296, -0.00279, -0.00253, -0.0099, -0.00138]  min -0.0129 · mean -0.00172 · max 0
reward/rcv_progress: [0.234, 0.77, 0.754, 0.753, 0.766, 0.789, 0.768, 0.764, 0.768, 0.774]  min 0.234 · mean 0.767 · max 0.875
reward/rcv_spin: [0, -0.105, -0.0852, -0.0912, -0.101, -0.119, -0.0993, -0.0906, -0.0993, -0.0902]  min -0.179 · mean -0.0999 · max 0
reward/rcv_still: [0, -0.121, -0.174, -0.168, -0.263, -0.225, -0.233, -0.235, -0.251, -0.233]  min -0.274 · mean -0.207 · max 0
reward/reach_rcv: [0.801, 0.789, 0.794, 0.78, 0.787, 0.797, 0.798, 0.813, 0.817, 0.836]  min 0.762 · mean 0.802 · max 0.906
reward/reach_src: [0.598, 0.754, 0.76, 0.743, 0.688, 0.598, 0.658, 0.762, 0.762, 0.759]  min 0.596 · mean 0.72 · max 0.781
reward/rot_pour: [0, -0.0358, -0.0104, -0.0363, -0.047, -0.076, -0.076, -0.0555, -0.0719, -0.0669]  min -0.0936 · mean -0.0491 · max 0
reward/rush_near_cup: [-0.236, -0.0355, -0.0348, -0.0371, -0.0318, -0.0233, -0.0337, -0.0342, -0.0322, -0.0307]  min -0.236 · mean -0.0331 · max -0.0149
reward/settle_finish: [0, 0.000612, 0.00118, 0.00239, 0.00121, 0.000789, 0.00198, 0.00204, 0.00194, 0.00216]  min 0 · mean 0.00141 · max 0.00384
reward/spill_delta: [0, -0.0498, -0.0762, -0.085, -0.0381, -0.0352, -0.105, -0.0527, -0.0527, -0.105]  min -0.155 · mean -0.056 · max 0
reward/src_cross: [0, -0.0889, -0.00589, -0.0513, -0.0284, -0.0191, -0.0356, -0.0165, -0.0244, -0.0196]  min -0.415 · mean -0.0443 · max 0
reward/src_emptied: [0, 2.3e-05, 0.000183, 2.39e-05, 6.52e-05, 0.000175, 1.52e-05, 6.25e-05, 8.18e-05, 2.43e-06]  min 0 · mean 7.33e-05 · max 0.00107
reward/src_speed: [0, -0.0944, -0.0664, -0.0688, -0.0633, -0.0449, -0.0684, -0.0542, -0.0545, -0.0593]  min -0.259 · mean -0.0649 · max 0
reward/src_spin: [0, -0.0346, -0.0343, -0.0381, -0.0365, -0.0312, -0.0435, -0.0384, -0.0482, -0.0503]  min -0.0648 · mean -0.04 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.4e-06 · max 0.00195
reward/total: [2.08, 14.5, 15.2, 15, 17.4, 19.7, 16.8, 17.1, 16.8, 16.6]  min 1.52 · mean 16.3 · max 19.8
reward/transfer_tail: [0, 9.22e-06, 0, 2.88e-06, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.93e-06 · max 0.000196
reward/transferred: [0, 0.000186, 0.000263, 0.000345, 0.000293, 0.000294, 0.000423, 0.000289, 0.000398, 0.000494]  min 0 · mean 0.000382 · max 0.00145
reward/unaimed_release: [0, -0.0117, -0.019, -0.0192, -0.00634, -0.00847, -0.0205, -0.0115, -0.0111, -0.0239]  min -0.0367 · mean -0.0124 · max 0
rewards/step: [75.3, 1.09e+04, 1.17e+04, 1.08e+04, 1.34e+04, 1.66e+04, 1.2e+04, 1.19e+04, 1.17e+04, 1.2e+04]  min 75.3 · mean 1.22e+04 · max 1.7e+04
task/aim_dist: [0.319, 0.518, 0.592, 0.587, 0.209, 0.177, 0.225, 0.508, 0.394, 0.429]  min 0.173 · mean 0.343 · max 5.64
task/cup_collision_rate: [0, 0.000488, 0.00122, 0.00317, 0.00879, 0.00952, 0.00732, 0.00293, 0.00562, 0.00708]  min 0 · mean 0.00492 · max 0.03
task/cups_center_dist: [0.32, 0.525, 0.607, 0.597, 0.21, 0.18, 0.24, 0.517, 0.402, 0.434]  min 0.174 · mean 0.35 · max 5.64
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 3e-07 · max 0.000244
task/nested_rate: [0, 0.000977, 0.00269, 0.00146, 0.00562, 0.000977, 0.0083, 0.0022, 0.00586, 0.0061]  min 0 · mean 0.00353 · max 0.0122
task/rcv_cup_lift: [8.07e-05, 0.0966, 0.0804, 0.0699, 0.056, 0.0796, 0.0591, 0.0552, 0.0565, 0.0601]  min 8.07e-05 · mean 0.0704 · max 0.233
task/rcv_grasped: [0, 0.807, 0.787, 0.787, 0.816, 0.845, 0.803, 0.796, 0.803, 0.803]  min 0 · mean 0.804 · max 0.911
task/rcv_hand_foreign_rate: [0.05, 0.00171, 0.000977, 0.00269, 0.00415, 0.00122, 0.00391, 0.00146, 0.00244, 0.00293]  min 0 · mean 0.00276 · max 0.061
task/src_cup_lift: [0.000275, 0.263, 0.344, 0.316, 0.174, 0.198, 0.199, 0.26, 0.237, 0.321]  min -2.02 · mean 0.227 · max 0.538
task/src_grasped: [0, 0.76, 0.745, 0.74, 0.783, 0.84, 0.758, 0.747, 0.744, 0.75]  min 0 · mean 0.76 · max 0.91
task/src_hand_foreign_rate: [0.0022, 0.0022, 0.00342, 0.00464, 0.0083, 0.00806, 0.0156, 0.00635, 0.0129, 0.0151]  min 0 · mean 0.00779 · max 0.0232
task/src_tilt_deg: [1, 16.7, 28.4, 23.6, 17.9, 20, 35.5, 21.6, 22.5, 24.1]  min 1 · mean 21.9 · max 36.9
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.5e-07 · max 0.000244

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 18 operator notes — 붓기 높이를 열어 준다(각도는 계속 학습이 찾는다)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 사용자 원칙

- ★"각도에 대한 걸 제거해야 함. 적절한 각도는 학습이 찾게 해야 함" — iter_17 의 각도 처방 제거를 **유지**한다.
- 실기는 안정 속도 틸팅(속도·제어 벌점 유지). 조준·허가 창은 `ctx.cup_radius` 배수. 새 절대 거리 상수는 최소화.
- 사용자 지적(09.25): "j1,2,3 을 활용 못해서 말단이 포화되는 것 아닌가" → 측정으로 확인됨(아래 2).

## 1. iter_17 결과 두 갈래

- **t2r_rh3_i17**(i16g ep2100 웜스타트, entropy 3.5): 각도 수입 제거 후 붓기 **소멸**. 결정론 ep300 = 기울기 46.7°·in_target 0.000·잔량 1.000.
- **t2r_rh3_i17g**(파지 단계 rh2_i07-r3 ep400, entropy 22): 기울기가 스스로 15 → 30°까지 올랐다가 22°로 내려앉음.
  결정론 ep2900 = 파지 1.00/1.00·들기 0.220 m·**기울기 26.1°·in_target 0.000·잔량 1.000**.
  같은 구간 i16g(각도 수입 있음)는 기울기 35°·in_target 0.0036~0.0096 으로 배달 시작.
- 교훈: **탐색이 살아 있으면 각도 처방 없이도 기울인다.** 다만 지금은 다른 벽에 막혀 있다(아래).

## 2. ★측정 — 벽은 보상 설계가 아니라 **작업 높이**다

**(a) i17g trace(ep2800, 64 env × 900 스텝)**
- 소스 palm 회전 명령이 **박스 한계 −150° 에 중앙값으로 붙어 있는데** 컵 기울기는 중앙 24.2°.
  palm 회전 오차 중앙 98°(yaw 94.5°), 위치 오차 7.0 cm.
- 관절 사용률: r_aj_6 **93.6 %**, r_aj_7 **98.8 %**(말단 포화) 대 r_aj_2 **22.6 %**(속도 RMS 0.214).
- 최소 여유 관절 분포: **r_aj_2 가 전체 92.9 %·기울임 55°↑ 85.5 %** — j2 가 하한(−0.17)까지 내려가 박힌다.
- 기울기 75° 초과는 컵이 중앙을 넘어간 y +0.06~+0.20 구간에서만 26 % 나온다(다른 구간 0.6~1.5 %).

**(b) 손바닥 자세 도달성 프로브(학습 없음, 빈 손, 같은 회전 명령)**

| 손바닥 높이 z | 회전 오차 | j2 여유 | 병목 관절 |
|---|---|---|---|
| 0.29~0.42 | **143~151°** | 0.07~0.16 | r_aj_6 / r_aj_4 |
| 0.53 | 38~52° | 0.09~0.18 | r_aj_5 |
| **0.69~0.76** | **10~23°** | 0.21~0.33 | r_aj_5 / r_aj_7 |

  **높이가 회전 도달성을 15배 가까이 바꾼다.** 가로 위치 효과는 그보다 작다.
- ★정정: fabric 기본자세가 j2 를 묶는다는 앞선 해석은 틀렸다. 이 트랙은 `side_rig.setup_fabric` 이
  `fabric.default_config.copy_(fabric_q)` 로 **env 리셋 자세**를 널스페이스 목표로 덮어쓰고, 그 자세의 j2 여유는 0.341 이다.
  j2 는 자세 끌림이 아니라 **낮은 곳에서 palm 목표를 맞추려다** 스스로 하한까지 내려간다(palm_attractor gain 80 대 cspace 1).

**(c) 보상이 높이를 막고 있다(코드)**
- `LIFT_FREE_SRC0 = 0.24` + `over_lift`(W 1.5, 폭 0.20). 컵 스폰 z 0.268 → **무료 구간 상한 컵 z ≈ 0.51**.
- i17g 결정론 재생의 컵 높이 0.478~0.50 — 정확히 그 천장이다.
- 기울기 연동 추가 허용(`lip_drop`)은 기울기가 26° 라 거의 0 이다. **못 기울여 높이가 안 열리고, 낮아서 못 기울이는 순환.**

## 3. 다음 보상이 해야 할 것

1. ★**소스 들기 무료 구간을 올린다.** `LIFT_FREE_SRC0` 0.24 → **0.40** 부근(컵 z ≈ 0.67, 프로브에서 회전 오차 10~23° 대역).
   과들기 벌점은 그 위에서만 걸어 컵을 던지는 것은 계속 막는다(`OVER_LIFT_RANGE` 유지).
2. **들기 목표(`LIFT_TARGET_SRC` 0.15)도 함께 올린다** — 목표가 낮으면 수입 정점이 낮은 곳에 남는다. 0.30 부근을 권한다.
3. **기울기 연동 허용(`lip_drop`)은 유지**한다(기하 보정이지 각도 처방이 아니다).
4. **각도 처방은 계속 넣지 않는다.** 붓는 각은 학습이 정한다. 배달·비움·흘림이 값을 만든다.
5. **그대로 둘 것:** 접근/허가 분리·만남 통로(meet_cross·src_cross)·grip_hold·속도 벌점·제어 벌점·관절 한계/떨림(W 0.4)·
   리시버 규율·성공 기준(env 0.80/0.25/20).
6. 리시버 들기 밴드(0.08/0.10)는 그대로 둔다 — 받는 컵은 낮아도 된다. 다만 소스가 높아지면 두 컵 높이차가 커지므로
   `lip_dz`·`perm` 의 높이 조건이 새 높이에서도 도달 가능한지 확인한다.

## 4. 웜스타트

- **파지 단계 rh2_i07-r3 ep400**(entropy ~22, 파지 0.71/0.75, 붓기 자세 없음). i17g 와 같은 출발점이라 **높이 허용만의 효과**를 가른다.
