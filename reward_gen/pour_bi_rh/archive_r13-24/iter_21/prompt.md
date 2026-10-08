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


def _vertical_rates(ctx, lip_pt):
    """iter_19 NEW. Two vertical rates relative to the RECEIVER MOUTH [m/s], positive = going up.
      lip_rate : d(lip_dz)/dt of the pouring lip (rigid-body velocity at lip_pt, so turning the cup
                 toward the receiver - which drops the lip - shows up here as a descent).
      org_rate : d/dt of the source cup ORIGIN height. The origin sits near the centre of mass and
                 ~13 mm below the pinch, so turning the cup barely moves it: this is the height of
                 the cup itself, independent of how far it is tilted."""
    v_lip = ctx.src_cup_lin_vel + torch.cross(ctx.src_cup_ang_vel, lip_pt - ctx.src_cup_pos, dim=-1)
    v_mouth = ctx.rcv_cup_lin_vel + torch.cross(ctx.rcv_cup_ang_vel,
                                                ctx.rcv_cup_mouth_pos - ctx.rcv_cup_pos, dim=-1)
    lip_rate = v_lip[:, 2] - v_mouth[:, 2]
    org_rate = ctx.src_cup_lin_vel[:, 2] - v_mouth[:, 2]
    return lip_rate, org_rate


def _tilt_direction_error(ctx):
    """iter_19 NEW. How far the cup leans AWAY from the receiver, weighted by how far it leans.
    The horizontal part of the cup's up axis, u_xy, has length sin(tilt) and points where the mouth
    (and so the pour) goes. psi = angle between u_xy and d_hat (source origin -> receiver mouth, the
    same d_hat _pour_geometry uses for theta). dir_err = (1 - cos psi) / 2: 0 toward, 0.5 sideways,
    1 straight away. No angle MAGNITUDE is valued: dir_err is 0 at every tilt when aimed.
    Why not `mis`: at tilt >= 90 deg theta saturates at 90 deg for ANY psi < 90 deg, so mis reads 0
    with an 89 deg sideways error - blind exactly where i18 released its beads (median 91 deg)."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_hat = v / torch.clamp(v.norm(dim=-1), min=1e-6).unsqueeze(-1)
    u_xy = ctx.src_cup_up[:, :2]
    u_h = u_xy.norm(dim=-1)
    cos_psi = torch.clamp((u_xy * d_hat).sum(dim=-1) / torch.clamp(u_h, min=1e-6), -1.0, 1.0)
    dir_err = 0.5 * (1.0 - cos_psi)
    return dir_err, torch.acos(cos_psi)


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
    #
    # ============================ iter_18: ONE CHANGE - THE SOURCE MAY WORK HIGHER ====================
    # iter_17's deletion of every angle target STANDS. Nothing about the pour angle is put back here:
    # the release angle is still the policy's to find, and only beads pay for it.
    # i17g measurement (same warm start as this round, rh2_i07-r3 ep400, entropy 22, 2900 epochs):
    #   grasp 1.00/1.00, source lift 0.220 m, tilt 26.1 deg, in_target 0.000, source residue 1.000.
    #   The source palm ROLL COMMAND sat at the box limit (-150 deg) while the cup reached only ~24 deg,
    #   with a median palm rotation error of 98 deg: the commanded roll was not realisable. Joint use
    #   r_aj_6 93.6 % / r_aj_7 98.8 % against r_aj_2 22.6 % - the wrist was saturated while the shoulder
    #   was idle. A scripted probe (no learning, empty hand, the same roll command) measured how that
    #   depends on HEIGHT alone:
    #       palm z 0.29-0.42  ->  roll error 143-151 deg   (bottleneck r_aj_6 / r_aj_4, j2 margin 0.07-0.16)
    #       palm z 0.53       ->  roll error  38-52  deg   (bottleneck r_aj_5)
    #       palm z 0.69-0.76  ->  roll error  10-23  deg   (j2 margin 0.21-0.33)
    #   Height changes rotational reachability by a factor of ~15; horizontal position matters far less.
    # The reward itself was holding the arm down there: LIFT_FREE_SRC0 = 0.24 on top of a 0.268 m spawn
    # and the 0.005 m dead band puts the free ceiling at cup z ~= 0.51, and the i17g deterministic
    # replay sat at cup z 0.478-0.50 - exactly on that ceiling. The tilt-linked part of the budget
    # (lip_drop) was ~0 because the cup only reached 26 deg, so the loop closed on itself: too low to
    # turn, and not turning so no extra height. iter_18 opens the height and changes NOTHING else.
    #
    # ============================ iter_19: WHICH WAY and FROM WHERE - never HOW FAR ===================
    # i18 ep800 deterministic replay (64 env): the tilt was found by the policy itself (per-env max tilt
    # median 87.2 deg, release at a median 91 deg) - iter_17/18 were right about the angle. But at tilt
    # > 60 deg the cup leaned a median 127 deg AWAY from the receiver direction, and the source cup was
    # only +0.016 m above the receiver (both z ~= 0.36; mouths 0.157 m apart): it poured BESIDE the
    # receiver at its own height. in_target: per-env max median 0.000.
    # The trace also shows the cup lifted to z 0.617 and then SINKING to 0.359 while the tilt grew
    # 25 -> 77 deg, wrist j5/j7 both at -1.3 rad: height traded for tilt.
    # Cause: when iter_17 deleted the angle terms, mis_tilt went too, so direction survived only as the
    # `toward` sign GATE - a gate switches income off but puts no price on the wrong way. Height above
    # the rim was likewise only a gate (lip_above_raw), with no gradient to climb it.
    # iter_19 adds three terms, everything else verbatim:
    #   tilt_dir   : cost = tilt magnitude x direction error, 0 whenever the lean points at the receiver
    #   lip_rise   : difference form on lip_dz (spout vs receiver rim), live only below the permission
    #                height LIP_Z_HI_R * r; climbing pays, sinking (or tipping the lip down) costs
    #   pour_sink  : difference form on the source ORIGIN height vs the receiver rim, live below the
    #                height that keeps the lip at LIP_Z_HI_R * r even at a quarter turn; tilt-free,
    #                so it charges losing height and never charges turning
    # No angle target, band or fill-coupled release ladder comes back. The angle stays the policy's.
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
    # --- source lift: band + over-lift ramp. TWO CONSTANTS MOVE HERE AND NOWHERE ELSE --------------
    W_LIFT_SRC = 3.0        # unchanged: the lift term keeps exactly the scale it was measured at
    LIFTOFF_H = 0.03        # unchanged
    LIFT_TARGET_SRC = 0.30  # iter_18: 0.15 -> 0.30. A budget ceiling only permits height; the INCOME
                            # peak has to move too, or the best-paid place stays low. h = 0.30 is cup
                            # z ~= 0.573 (spawn 0.268 + 0.005 dead band), between the probe's 0.53 row
                            # (38-52 deg error) and its 0.69 row (10-23 deg), and like the old 0.15
                            # against 0.24 it still saturates BELOW the free ceiling, so full lift
                            # income is collected inside the band and never by pushing past it.
    LIFT_FREE_SRC0 = 0.40   # iter_18: 0.24 -> 0.40. Free ceiling cup z ~= 0.673, i.e. the probe's
                            # 0.69-0.76 palm band where the roll error is 10-23 deg instead of the
                            # 143-151 deg measured at 0.29-0.42. Still the base of a MOVING budget:
                            # the tilt-linked part is lip_drop, a geometric allowance (how far the
                            # spout has already fallen), never an angle target.
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
    # --- iter_19 NEW: direction and height (geometry of the task, not an angle prescription) ---------
    W_TILT_DIR = 1.5        # cost = W * (tilt / 90 deg) * (1 - cos psi) / 2. At the i18 failure state
                            # (tilt 77 deg, psi 127 deg): (1.344/1.571) * 0.801 = 0.686 -> 1.03/step,
                            # i.e. the +-1/step budget for a new term; x 37.9 % of ~800 steps spent above
                            # 60 deg that is ~-300 per episode. A 45 deg aiming error at 90 deg costs only
                            # 0.22/step, so the last few degrees of aim are left to the beads.
    V_Z_SCALE = 0.04        # m/s. i18 trace: the cup sank 0.617 -> 0.359 m over steps ~175 -> ~575,
                            # 400 steps at 60 Hz = 6.7 s -> 0.039 m/s. Sinking at that rate saturates
                            # both height terms (-W per step each).
    W_LIP_RISE = 1.0        # same weight as approach_lip, its horizontal twin
    W_POUR_SINK = 1.0       # same scale; both are exact potential rates, so an up-down cycle nets 0
    SINK_KEEP_R = 1.0       # the lip falls by r*sin + mouth_z*(1 - cos): at a quarter turn (i18's median
                            # release was 91 deg) that is r + mouth_z. The origin must therefore sit
                            # (LIP_Z_HI_R + SINK_KEEP_R) * r = 1.7 r ~= 48 mm above the receiver rim for the
                            # lip to still clear it by LIP_Z_HI_R * r when horizontal. i18 at release: the
                            # origin was ~-0.038 m vs the rim (0.359 vs 0.397), 0.086 m short.

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
    # iter_18: the FORMULA is untouched - only LIFT_FREE_SRC0 (0.24 -> 0.40) and LIFT_TARGET_SRC
    # (0.15 -> 0.30) moved. The over-lift charge therefore still begins exactly at the top of the free
    # band (now h = 0.40 + lip_drop) and still saturates OVER_LIFT_RANGE = 0.20 m above it, so
    # throwing the cup upward costs the same as before, just measured from the new ceiling.
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

    # ---- iter_19 NEW: direction and height, both while BOTH cups are carried ------------------------
    carry_both = carry_soft_src * carry_soft_rcv * not_nested
    # (1) direction: tilt magnitude x direction error. Zero when leaning at the receiver, at ANY angle.
    #     sep_ok fades it where the azimuth is undefined (source origin within 3 cm of the mouth axis).
    dir_err, psi = _tilt_direction_error(ctx)
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)                # magnitude only as a multiplier, no target
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err
    # (2) spout above the receiver mouth: exact rate of the potential min(lip_dz, LIP_Z_HI_R * r).
    #     Below the permission height every mm the lip climbs pays and every mm it loses costs -
    #     whether it was lost by sinking or by tipping the lip down there. Above it: nothing, so being
    #     high is never paid for as a state (no parking) and pouring from up there is never taxed.
    lip_vz, org_vz = _vertical_rates(ctx, geo["lip_pt"])
    lip_tgt = LIP_Z_HI_R * r_mouth
    lip_low = (geo["lip_dz"] < lip_tgt).to(dtype)
    lip_rise = carry_both * lip_low * torch.clamp(lip_vz / V_Z_SCALE, -1.0, 1.0)
    # (3) do not trade height for tilt: exact rate of min(h_org, keep) on the source ORIGIN height
    #     above the receiver rim, over the pouring stage (both cups held and >= 5.5 cm up). The origin
    #     barely moves when the cup turns, so this charges losing height and never charges turning.
    h_org = ctx.src_cup_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    keep_h = (LIP_Z_HI_R + SINK_KEEP_R) * r_mouth
    org_low = (h_org < keep_h).to(dtype)
    pour_sink = up_w_src * up_w_rcv * not_nested * org_low \
        * torch.clamp(org_vz / V_Z_SCALE, -1.0, 1.0)

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
        # ★ iter_19 NEW: which way (cost, 0 when aimed at any angle) and from where (difference form)
        "tilt_dir": -W_TILT_DIR * tilt_dir,
        "lip_rise": W_LIP_RISE * lip_rise,
        "pour_sink": W_POUR_SINK * pour_sink,
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
    # ★ iter_19: the numbers this round is judged on (i18 replay: psi 127 deg, origin -0.038 m vs rim)
    comps["diag_dir_err_deg"] = psi * (180.0 / math.pi)
    comps["diag_dir_err_w"] = tilt_n * dir_err
    comps["diag_h_org"] = h_org
    comps["diag_lip_low"] = lip_low
    comps["diag_org_low"] = org_low
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
bead/in_target: [0, 8.54e-05, 0.000269, 0.000439, 0.000964, 0.0014, 0.000964, 0.000415, 0.000317, 0.000146]  min 0 · mean 0.000517 · max 0.00249
bead/spill: [1.22e-05, 0.0159, 0.0235, 0.0245, 0.0233, 0.0224, 0.019, 0.0145, 0.0159, 0.00791]  min 1.22e-05 · mean 0.0178 · max 0.0367
done/drop: [0, 0, 0.000977, 0.000732, 0.000977, 0.000244, 0.000244, 0.000488, 0.000488, 0]  min 0 · mean 0.000339 · max 0.00171
done/mimic_runaway: [0.000244, 0.000244, 0, 0.000244, 0, 0, 0, 0.000244, 0, 0.000244]  min 0 · mean 0.000103 · max 0.000732
episode_lengths/step: [84.1, 759, 778, 755, 771, 807, 826, 755, 788, 793]  min 84.1 · mean 782 · max 863
reward/act_flip: [0, -0.123, -0.12, -0.0979, -0.0844, -0.0772, -0.0643, -0.0537, -0.0546, -0.049]  min -0.18 · mean -0.0837 · max 0
reward/action_rate: [-0.032, -0.024, -0.0214, -0.0205, -0.0185, -0.018, -0.0163, -0.0153, -0.0143, -0.0136]  min -0.032 · mean -0.0182 · max -0.0134
reward/aim_gate: [0, 0.148, 0.556, 0.359, 0.257, 0.158, 0.143, 0.0669, 0.0495, 0.0334]  min 0 · mean 0.187 · max 0.59
reward/align_mouths: [0, 2.08, 2.76, 2.47, 2.53, 2.57, 2.66, 2.19, 2.25, 2.19]  min 0 · mean 2.39 · max 2.86
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/approach_lip: [0, -0.00298, 0.0133, 0.0349, 0.0553, 0.0682, 0.0771, 0.0631, 0.0473, 0.0524]  min -0.165 · mean 0.0464 · max 0.0954
reward/arm_limit_rcv: [0, -0.00162, -0.00451, -0.0289, -0.0387, -0.0427, -0.0438, -0.0501, -0.0526, -0.0484]  min -0.0564 · mean -0.034 · max 0
reward/arm_limit_src: [0, -0.093, -0.102, -0.0989, -0.109, -0.121, -0.125, -0.116, -0.126, -0.13]  min -0.131 · mean -0.115 · max 0
reward/arm_shake_rcv: [0, -0.0817, -0.103, -0.109, -0.107, -0.1, -0.0865, -0.0868, -0.0957, -0.0901]  min -0.116 · mean -0.0964 · max 0
reward/arm_shake_src: [0, -0.0695, -0.0769, -0.0864, -0.0993, -0.0989, -0.0903, -0.0919, -0.0882, -0.0745]  min -0.108 · mean -0.0849 · max 0
reward/away_rcv: [-0.104, -0.15, -0.176, -0.177, -0.173, -0.174, -0.178, -0.174, -0.174, -0.173]  min -0.18 · mean -0.171 · max -0.104
reward/carry_rate: [0, -0.168, -0.173, -0.153, -0.142, -0.141, -0.124, -0.103, -0.11, -0.106]  min -0.242 · mean -0.14 · max 0
reward/carry_src: [0, 1.5, 1.64, 1.5, 1.64, 1.87, 1.91, 1.59, 1.81, 1.85]  min 0 · mean 1.72 · max 2
reward/close_rcv: [0.0683, 0.0756, 0.0756, 0.0702, 0.0734, 0.071, 0.0747, 0.0728, 0.0757, 0.0748]  min 0.0637 · mean 0.0754 · max 0.1
reward/close_src: [0.083, 0.159, 0.17, 0.155, 0.166, 0.148, 0.178, 0.178, 0.153, 0.177]  min 0.083 · mean 0.168 · max 0.205
reward/contact_push: [-0.0064, -0.00679, -0.00304, -0.00379, -0.00365, -0.00197, -0.00163, -0.00192, -0.00163, -0.00139]  min -0.114 · mean -0.00283 · max -0.000792
reward/converge: [0, 1.89, 2.05, 1.97, 2.1, 2.31, 2.32, 2.06, 2.3, 2.34]  min 0 · mean 2.18 · max 2.41
reward/cup_collision: [0, -0.000244, -0.0016, -0.00395, -0.0027, -0.00118, -0.00222, -0.00519, -0.00186, -0.00136]  min -0.00926 · mean -0.00232 · max 0
reward/cup_disturb_rcv: [-0.0267, -0.0774, -0.094, -0.103, -0.0916, -0.0742, -0.0659, -0.083, -0.0687, -0.0619]  min -0.229 · mean -0.0807 · max -0.0267
reward/cup_disturb_src: [-0.0567, -0.0293, -0.0281, -0.0292, -0.022, -0.0158, -0.0171, -0.0207, -0.0137, -0.0137]  min -0.266 · mean -0.0212 · max -0.00496
reward/cup_fling: [-0.000422, -0.00883, -0.013, -0.0146, -0.0155, -0.00985, -0.00913, -0.0134, -0.0105, -0.00758]  min -0.017 · mean -0.011 · max -0.000422
reward/diag_balance_src: [0, 0.573, 0.606, 0.596, 0.616, 0.672, 0.675, 0.63, 0.678, 0.689]  min 0 · mean 0.649 · max 0.726
reward/diag_bang_rcv: [0.278, 0.216, 0.216, 0.199, 0.176, 0.16, 0.146, 0.144, 0.134, 0.122]  min 0.108 · mean 0.167 · max 0.291
reward/diag_bang_src: [0.191, 0.113, 0.0995, 0.106, 0.0969, 0.0988, 0.0947, 0.0907, 0.0863, 0.0852]  min 0.0688 · mean 0.0971 · max 0.191
reward/diag_clean: [1, 0.968, 0.951, 0.951, 0.95, 0.944, 0.96, 0.962, 0.968, 0.98]  min 0.928 · mean 0.962 · max 1
reward/diag_close_speed: [0.0102, 0.000366, -0.00117, 0.00775, 0.0142, 0.0209, 0.0223, 0.0184, 0.0171, 0.0172]  min -0.04 · mean 0.0132 · max 0.0281
reward/diag_ctrl_gate: [0, 0.609, 0.694, 0.665, 0.7, 0.769, 0.766, 0.675, 0.762, 0.762]  min 0 · mean 0.72 · max 0.824
reward/diag_emptied: [7.32e-05, 0.0175, 0.0286, 0.0296, 0.0311, 0.0343, 0.022, 0.0228, 0.017, 0.0108]  min 7.32e-05 · mean 0.022 · max 0.0443
reward/diag_empty_q: [8.11e-06, 0.0103, 0.0187, 0.0198, 0.0205, 0.0232, 0.0133, 0.015, 0.0101, 0.00668]  min 8.11e-06 · mean 0.0141 · max 0.0303
reward/diag_fill_tail: [0, 0, 0, 0.00014, 9.97e-05, 5.87e-05, 0.000126, 9.54e-05, 1.62e-05, 2.72e-05]  min 0 · mean 8.13e-05 · max 0.000763
reward/diag_finish_q: [4.05e-06, 0.00517, 0.00937, 0.00997, 0.0103, 0.0116, 0.00672, 0.00753, 0.00504, 0.00335]  min 4.05e-06 · mean 0.0071 · max 0.0154
reward/diag_flip_palm: [1.23, 0.688, 0.576, 0.513, 0.418, 0.356, 0.303, 0.275, 0.252, 0.223]  min 0.199 · mean 0.406 · max 1.23
reward/diag_flow_beads: [0, 0.000244, 0, 0, 0, 0.000244, 0, 0.000488, 0, 0]  min 0 · mean 9.68e-05 · max 0.00122
reward/diag_frac_rcv: [0.999, 1, 0.794, 0.641, 0.573, 0.456, 0.484, 0.435, 0.378, 0.398]  min 0.251 · mean 0.596 · max 1.68
reward/diag_frac_src: [0.00136, 0.459, 0.441, 0.255, 0.203, 0.292, 0.273, 0.252, 0.332, 0.226]  min -0.249 · mean 0.315 · max 0.672
reward/diag_grip_keep_rcv: [0, 0.14, 0.142, 0.128, 0.132, 0.125, 0.127, 0.129, 0.133, 0.132]  min 0 · mean 0.137 · max 0.208
reward/diag_grip_keep_src: [0, 0.337, 0.34, 0.309, 0.321, 0.293, 0.342, 0.33, 0.283, 0.33]  min 0 · mean 0.328 · max 0.447
reward/diag_home_eff: [1, 0.988, 0.986, 0.966, 0.971, 0.973, 0.974, 0.921, 0.961, 0.971]  min 0.913 · mean 0.97 · max 1
reward/diag_lead_gate: [0, 0.373, 0.646, 0.546, 0.52, 0.452, 0.451, 0.342, 0.238, 0.203]  min 0 · mean 0.401 · max 0.677
reward/diag_lift_budget: [0.4, 0.404, 0.408, 0.408, 0.412, 0.409, 0.407, 0.409, 0.403, 0.403]  min 0.4 · mean 0.407 · max 0.416
reward/diag_lip_dz: [-2.47e-05, 0.152, 0.178, 0.126, 0.0938, 0.127, 0.0933, 0.135, 0.109, 0.099]  min -2.47e-05 · mean 0.126 · max 0.301
reward/diag_lip_xy: [0.291, 0.231, 0.242, 0.179, 0.184, 0.145, 0.13, 0.229, 0.213, 0.215]  min 0.104 · mean 0.192 · max 0.463
reward/diag_meet_q: [0.000855, 0.0354, 0.257, 0.48, 0.589, 0.763, 0.731, 0.773, 0.823, 0.816]  min 0.000855 · mean 0.57 · max 0.841
reward/diag_mis_deg: [0.541, 7.17, 6.93, 14.6, 18.3, 22, 22.2, 25.1, 28.8, 28.6]  min 0.541 · mean 19.2 · max 33
reward/diag_palm_sat: [0.449, 0.517, 0.473, 0.428, 0.415, 0.397, 0.393, 0.427, 0.407, 0.411]  min 0.377 · mean 0.433 · max 0.587
reward/diag_palm_sat_rcv: [0.537, 0.471, 0.4, 0.374, 0.376, 0.391, 0.385, 0.391, 0.377, 0.377]  min 0.354 · mean 0.396 · max 0.537
reward/diag_perm: [0, 0.123, 0.375, 0.175, 0.125, 0.0651, 0.0381, 0.0156, 0.0215, 0.0119]  min 0 · mean 0.101 · max 0.441
reward/diag_pour_phase: [0, 0.136, 0.532, 0.333, 0.235, 0.144, 0.129, 0.045, 0.0389, 0.0255]  min 0 · mean 0.172 · max 0.56
reward/diag_rate_palm_rcv: [4.47, 3.07, 2.73, 2.43, 2.13, 1.82, 1.62, 1.48, 1.43, 1.36]  min 1.29 · mean 2.03 · max 4.47
reward/diag_rcv_limit_hit: [0, 0, 0.000244, 0, 0.000244, 0, 0, 0, 0.000244, 0.000244]  min 0 · mean 0.000114 · max 0.00122
reward/diag_rcv_margin_min: [0.245, 0.289, 0.281, 0.216, 0.197, 0.188, 0.186, 0.177, 0.175, 0.18]  min 0.169 · mean 0.213 · max 0.323
reward/diag_rcv_qdd: [6.18, 6.17, 6.52, 7.24, 7.06, 6.31, 5.78, 6.59, 6.83, 86.6]  min 5.38 · mean 6.67 · max 113
reward/diag_rcv_qdd_max: [14.5, 14.6, 15.8, 18.3, 17.7, 15.3, 13.8, 16.4, 16.4, 222]  min 13.1 · mean 16.3 · max 310
reward/diag_rcv_travel: [0.000678, 0.0678, 0.111, 0.136, 0.154, 0.191, 0.178, 0.19, 0.217, 0.225]  min 0.000678 · mean 0.167 · max 0.372
reward/diag_release_beads: [0, 0.00513, 0.0061, 0.00464, 0.00659, 0.00415, 0.00293, 0.00439, 0.00415, 0.00171]  min 0 · mean 0.00397 · max 0.0115
reward/diag_settle_still: [0.974, 0.772, 0.727, 0.705, 0.683, 0.706, 0.73, 0.721, 0.75, 0.763]  min 0.552 · mean 0.731 · max 0.974
reward/diag_src_limit_hit: [0, 0.00342, 0.012, 0.0183, 0.0208, 0.0159, 0.00977, 0.00806, 0.00952, 0.00562]  min 0 · mean 0.0124 · max 0.0339
reward/diag_src_margin_min: [0.194, 0.129, 0.123, 0.125, 0.119, 0.113, 0.11, 0.114, 0.11, 0.106]  min 0.103 · mean 0.115 · max 0.194
reward/diag_src_qdd: [6.08, 7.37, 6.57, 7.25, 7.64, 6.78, 6.41, 6.98, 6.54, 5.71]  min 5.69 · mean 6.71 · max 28.1
reward/diag_src_qdd_max: [13.7, 18.6, 16.6, 18.6, 19.6, 16.8, 15.8, 17.7, 16.6, 14.1]  min 13.7 · mean 16.9 · max 92.4
reward/diag_src_tilt_deg: [1.02, 12.8, 17.6, 25.1, 34.5, 33.6, 32, 37.6, 33.4, 32.7]  min 1.02 · mean 28.5 · max 40.6
reward/diag_src_travel: [0.00117, 0.261, 0.316, 0.195, 0.172, 0.139, 0.119, 0.189, 0.19, 0.187]  min 0.00117 · mean 0.193 · max 0.448
reward/diag_src_v: [0.0363, 0.365, 0.253, 0.254, 0.244, 0.228, 0.204, 0.259, 0.219, 0.195]  min 0.0363 · mean 0.235 · max 2.27
reward/diag_src_w: [0.515, 4.39, 1.45, 1.51, 1.69, 1.38, 1.32, 1.4, 1.32, 1.13]  min 0.515 · mean 1.65 · max 38.1
reward/diag_theta_deg: [0.476, 5.61, 10.7, 10.5, 16.2, 11.7, 9.74, 12.4, 4.67, 4.1]  min 0.476 · mean 9.31 · max 20.3
reward/diag_toward: [0.552, 0.635, 0.683, 0.57, 0.592, 0.395, 0.406, 0.494, 0.261, 0.251]  min 0.201 · mean 0.476 · max 0.898
reward/diag_unaimed_beads: [0, 0.00507, 0.0054, 0.00435, 0.00615, 0.00361, 0.0026, 0.00293, 0.00413, 0.00171]  min 0 · mean 0.00368 · max 0.0115
reward/flow_excess: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.00146 · mean -1.4e-05 · max 0
reward/grasp_rcv: [0.000291, 1.05, 1.23, 1.16, 1.23, 1.3, 1.37, 1.18, 1.32, 1.32]  min 0.000291 · mean 1.25 · max 1.4
reward/grasp_src: [0.00502, 1.28, 1.37, 1.34, 1.41, 1.54, 1.54, 1.44, 1.55, 1.55]  min 0.00502 · mean 1.47 · max 1.72
reward/grip_hold_rcv: [0, 0.0826, 0.0844, 0.0765, 0.0784, 0.0746, 0.076, 0.0767, 0.0793, 0.0788]  min 0 · mean 0.0811 · max 0.123
reward/grip_hold_src: [0, 0.39, 0.402, 0.365, 0.382, 0.349, 0.408, 0.394, 0.338, 0.393]  min 0 · mean 0.389 · max 0.529
reward/hand_flip: [0, -0.125, -0.117, -0.108, -0.0994, -0.115, -0.0991, -0.0831, -0.0813, -0.0757]  min -0.196 · mean -0.104 · max 0
reward/hand_foreign: [-0.0182, -0.00211, -0.00208, -0.00398, -0.00362, -0.0021, -0.000465, -0.00335, -0.00286, -0.00129]  min -0.0182 · mean -0.0026 · max -0.000465
reward/hold_rcv: [0, 0.527, 0.519, 0.441, 0.433, 0.452, 0.499, 0.472, 0.542, 0.594]  min 0 · mean 0.509 · max 0.674
reward/lift_rcv: [0, 1.19, 1.38, 1.32, 1.4, 1.51, 1.51, 1.33, 1.51, 1.53]  min 0 · mean 1.42 · max 1.59
reward/lift_src: [0, 1.93, 2.22, 2.08, 2.13, 2.41, 2.21, 1.92, 2.19, 2.03]  min 0 · mean 2.16 · max 2.44
reward/meet_cross: [0, 0.13, 1.04, 1.82, 2.37, 3.32, 3.24, 2.97, 3.49, 3.57]  min 0 · mean 2.39 · max 3.69
reward/near_nested: [0, -0.000973, -0.00469, -0.0155, -0.0194, -0.00858, -0.00584, -0.03, -0.0159, -0.00403]  min -0.0518 · mean -0.00919 · max 0
reward/nested: [0, -0.000488, -0.00439, -0.0132, -0.0132, -0.00586, -0.00342, -0.0205, -0.0171, -0.00244]  min -0.0459 · mean -0.00699 · max 0
reward/over_lift_src: [0, -0.00381, -0.00337, -0.00265, -0.0016, -0.00194, -0.00141, -0.00324, -0.00257, -0.00211]  min -0.00647 · mean -0.00219 · max 0
reward/palm_bang_rcv: [0, -0.293, -0.346, -0.293, -0.268, -0.255, -0.23, -0.208, -0.209, -0.191]  min -0.417 · mean -0.256 · max 0
reward/palm_bang_src: [0, -0.0961, -0.0877, -0.0969, -0.0989, -0.11, -0.106, -0.0963, -0.0978, -0.101]  min -0.146 · mean -0.101 · max 0
reward/palm_sat_rcv: [0, -0.473, -0.45, -0.405, -0.422, -0.474, -0.464, -0.44, -0.457, -0.471]  min -0.631 · mean -0.464 · max 0
reward/palm_sat_src: [0, -1.13, -1.09, -0.933, -0.939, -0.963, -0.955, -0.977, -0.986, -1.01]  min -1.64 · mean -1.02 · max 0
reward/pour_delta: [0, 0.00183, 0, 0, -0.00183, 0.00183, 0, 0.00366, 0, 0]  min -0.00549 · mean 0.000451 · max 0.00916
reward/pour_dir_pen: [0, -0.538, -0.284, -0.244, -0.0563, -0.0298, -0.0174, -0.00479, -0.0111, -0.00111]  min -0.908 · mean -0.147 · max 0
reward/rcv_home: [0, -0.000411, -0.000165, -6.28e-05, -0.000276, -0.000219, -6.9e-05, -9.76e-05, -0.000105, -0.00038]  min -0.00205 · mean -0.000266 · max 0
reward/rcv_progress: [0.234, 0.714, 0.753, 0.737, 0.759, 0.774, 0.772, 0.74, 0.775, 0.784]  min 0.234 · mean 0.767 · max 0.874
reward/rcv_spin: [0, -0.108, -0.125, -0.12, -0.136, -0.128, -0.113, -0.112, -0.109, -0.115]  min -0.176 · mean -0.118 · max 0
reward/rcv_still: [0, -0.0781, -0.144, -0.211, -0.258, -0.314, -0.277, -0.243, -0.262, -0.238]  min -0.327 · mean -0.223 · max 0
reward/reach_rcv: [0.802, 0.815, 0.789, 0.785, 0.794, 0.786, 0.781, 0.797, 0.798, 0.795]  min 0.776 · mean 0.798 · max 0.904
reward/reach_src: [0.599, 0.696, 0.711, 0.704, 0.71, 0.675, 0.707, 0.707, 0.668, 0.697]  min 0.599 · mean 0.702 · max 0.766
reward/rot_pour: [0, -0.0125, -0.0702, -0.0477, -0.0539, -0.0433, -0.0269, -0.0114, -0.0138, -0.00566]  min -0.0805 · mean -0.0292 · max 0
reward/rush_near_cup: [-0.236, -0.0549, -0.0353, -0.0412, -0.0362, -0.0283, -0.0287, -0.0363, -0.0279, -0.0269]  min -0.236 · mean -0.0329 · max -0.0145
reward/settle_finish: [0, 0.000603, 0.000923, 0.0026, 0.00264, 0.00861, 0.00514, 0.00181, 0.00187, 0.00165]  min 0 · mean 0.00245 · max 0.00861
reward/spill_delta: [0, -0.0586, -0.0732, -0.0557, -0.0791, -0.0469, -0.0352, -0.0469, -0.0498, -0.0205]  min -0.138 · mean -0.0465 · max 0
reward/src_cross: [0, -0.264, -0.235, -0.144, -0.0816, -0.0922, -0.0791, -0.0392, -0.111, -0.127]  min -0.401 · mean -0.141 · max 0
reward/src_emptied: [0, 0.000133, 2.43e-06, 0.000263, 0.000354, 0.00016, 4.41e-05, 8.63e-05, 2.43e-06, 0]  min 0 · mean 0.000113 · max 0.00102
reward/src_speed: [0, -0.0753, -0.0873, -0.0922, -0.0972, -0.0877, -0.0884, -0.0726, -0.0671, -0.0684]  min -0.264 · mean -0.081 · max 0
reward/src_spin: [0, -0.0327, -0.0424, -0.0351, -0.0467, -0.0495, -0.0432, -0.0385, -0.0413, -0.0285]  min -0.0569 · mean -0.0387 · max 0
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.98e-05 · max 0.00488
reward/total: [2.08, 10.6, 13.3, 13.5, 14.8, 16.8, 17.1, 15.1, 16.7, 16.9]  min 1.54 · mean 15 · max 17.7
reward/transfer_tail: [0, 0, 0, 5.28e-06, 2.19e-05, 0, 1.92e-05, 0, 0, 0]  min 0 · mean 1.86e-05 · max 0.000434
reward/transferred: [0, 0.000107, 0.00025, 0.000648, 0.000505, 0.000862, 0.00106, 0.000383, 0.000841, 0.000112]  min 0 · mean 0.000537 · max 0.00244
reward/unaimed_release: [0, -0.0152, -0.0162, -0.0131, -0.0184, -0.0108, -0.0078, -0.00878, -0.0124, -0.00513]  min -0.0344 · mean -0.011 · max 0
rewards/step: [58.6, 8.58e+03, 1.04e+04, 1.02e+04, 1.18e+04, 1.34e+04, 1.42e+04, 1.2e+04, 1.29e+04, 1.33e+04]  min 58.6 · mean 1.18e+04 · max 1.52e+04
task/aim_dist: [0.319, 0.294, 0.261, 0.284, 0.218, 0.26, 0.273, 0.229, 0.288, 0.213]  min 0.197 · mean 0.281 · max 0.568
task/cup_collision_rate: [0, 0.000488, 0.00244, 0.0061, 0.00195, 0.0022, 0.00122, 0.0083, 0.00464, 0.00195]  min 0 · mean 0.00323 · max 0.0156
task/cups_center_dist: [0.32, 0.296, 0.266, 0.291, 0.23, 0.266, 0.277, 0.231, 0.281, 0.205]  min 0.196 · mean 0.283 · max 0.574
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.16e-06 · max 0.000244
task/nested_rate: [0, 0.000244, 0.00195, 0.00464, 0.000732, 0.00342, 0.000732, 0.0105, 0.00903, 0.00269]  min 0 · mean 0.0035 · max 0.0229
task/rcv_cup_lift: [8.5e-05, 0.0606, 0.0857, 0.0864, 0.0747, 0.0747, 0.0924, 0.0667, 0.0722, 0.0668]  min 8.5e-05 · mean 0.0736 · max 0.157
task/rcv_grasped: [0, 0.782, 0.819, 0.786, 0.819, 0.795, 0.834, 0.772, 0.811, 0.817]  min 0 · mean 0.805 · max 0.905
task/rcv_hand_foreign_rate: [0.0474, 0.00635, 0.00562, 0.00684, 0.00391, 0.00391, 0.00171, 0.00488, 0.00488, 0.00415]  min 0.000488 · mean 0.00498 · max 0.063
task/src_cup_lift: [0.000285, 0.205, 0.233, 0.25, 0.199, 0.215, 0.204, 0.162, 0.231, 0.154]  min 0.000285 · mean 0.207 · max 0.389
task/src_grasped: [0, 0.752, 0.746, 0.736, 0.769, 0.778, 0.817, 0.781, 0.793, 0.803]  min 0 · mean 0.776 · max 0.915
task/src_hand_foreign_rate: [0.00269, 0.00146, 0.00244, 0.00464, 0.00269, 0.00366, 0.00195, 0.00366, 0.0061, 0.00415]  min 0 · mean 0.00284 · max 0.00879
task/src_tilt_deg: [1.02, 14.1, 18.4, 25.3, 36.1, 31.8, 30.1, 38.7, 35.3, 31.9]  min 1.02 · mean 28.5 · max 40.6
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.45e-06 · max 0.000244

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
# round 21 operator notes — iter_20 의 버그 두 개만 고친다(설계는 그대로)

확인된 사실만 적는다.

## i20 결과 (t2r_rh3_i20, 파지 단계 ckpt, e143 에서 사용자 승인 종료)

| epoch | 소스/리시버 파지 | 들기 max | 에피소드 보상 | dir_err_at_max |
|---|---|---|---|---|
| 0~50 | 0.686 / 0.742 | 0.294 | −384 | 77.4° |
| 50~100 | 0.393 / 0.492 | 0.386 | +211 | 61.3° |
| 100~143 | **0.011 / 0.013** | 0.170 | **+712** | 31.9° |

- **컵을 놓을수록 보상이 올랐다.** 방향 오차는 줄고 있었으나(77 → 32°) 파지가 사라져 의미가 없다.

## 원인 (코드: iter_20/compute_reward.py)

1. **진행형 항이 비대칭이었다.** `_progress` 가 `clamp(rate/v, −4, +1)`, `pour_sink` 도 `clamp(−4, +1)`.
   학습 중 행동 잡음으로 거리가 매 스텝 앞뒤로 흔들리면 기댓값이 음수 — **흔들림 자체가 손실**이다.
2. **제어 벌점이 파지 플래그에만 곱해졌다.** act_flip·palm_bang·palm_sat·cmd_rate(palm)·arm_posture 가 `g_src`/`g_rcv` 를 곱해
   **컵을 놓으면 전부 0** 이 됐다. 파지 수입(W 1.0)보다 파지 중 비용이 커서 놓는 것이 이득이었다.

## iter_21 변경 (이것만)

- `_progress`·`pour_sink` 를 **대칭 clamp(−1, +1)** 로. 잡음의 기댓값이 0 이 된다.
- 제어 벌점을 파지 여부와 무관하게 부과(`g_src`/`g_rcv` 곱 제거). 놓아서 피할 수 없다.
- `W_GRASP_SRC/RCV` 1.0 → 1.5. 들고 있는 상태가 놓은 상태보다 확실히 더 벌게.
- 그 외(배치·조준·방향·높이·결과·들기 허용·안전)는 iter_20 그대로.

## 웜스타트
- 파지 단계 ckpt rh2_i07-r3 ep400.
