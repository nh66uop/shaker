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
    iter_08: unchanged since iter_04 — the grasp works (src_grasped 0.828, rcv_grasped 0.762)."""
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
    iter_06 VERBATIM — they are on the keep list (grasp_src 1.51, grasp_rcv 1.25 measured, both lift
    bands held for the whole 1978-epoch run)."""
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
    # NOTE: pose_ok = src_up rides on the attitude CORRIDOR. In iter_08 that corridor is opened by the
    # POURING LIP being over the receiver mouth, not by a direction-free distance (see compute_reward).

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


def _pour_geometry(ctx, lip_r, clearance, sep_min):
    """THE iter_08 CHANGE. Everything about the pour is measured on the DIRECTED spout, not on the
    direction-free tilt magnitude.

      d_hat  : horizontal unit vector src cup origin -> receiver mouth. Independent of the cup's
               attitude, so it is defined even for an upright cup and never flips while rotating.
      theta  : SIGNED tilt toward the receiver = atan2(up . d_hat, up_z). Only the component in the
               (d_hat, z) plane: a purely sideways tilt gives 0, tilting away gives a negative angle.
               theta == src_cup_tilt only when the mouth leans exactly at the receiver.
      lip    : the rim point the beads actually leave from = mouth centre + r*(cos(theta)*d_hat
               - sin(theta)*z). Drops as the cup turns toward the receiver, RISES when it turns away,
               which is what makes every distance below direction-aware.

    iter_07 used `mouth + r*(up*up_z - ez)` (the geometrically lowest rim point). That is defined for
    ANY rotation, so tilt magnitude, aim distance and the attitude corridor were all blind to the
    azimuth — the run tilted 54.8 deg with the mouth pointing at world +x and was paid in full."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]          # (N,2)
    sep = v.norm(dim=-1)                                               # (N,)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / sep_min, max=1.0)   # the azimuth is meaningless when the cups are coaxial

    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])                     # signed, negative = away
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    # what is tilted but NOT toward the receiver: 0 for a perfect pour attitude, ~= src_cup_tilt for a
    # sideways or backwards tilt. This is the only term with a gradient while theta <= 0.
    mis = torch.clamp(ctx.src_cup_tilt - theta_eff, min=0.0)

    lip_xy_vec = ctx.src_cup_mouth_pos[:, :2] \
        + (lip_r * torch.cos(theta)).unsqueeze(-1) * d_hat - ctx.rcv_cup_mouth_pos[:, :2]
    lip_xy = lip_xy_vec.norm(dim=-1)
    lip_dz = ctx.src_cup_mouth_pos[:, 2] - lip_r * torch.sin(theta) - ctx.rcv_cup_mouth_pos[:, 2]
    # aim distance to a point `clearance` above the receiver rim; the height error counts half, exactly
    # as in iter_06/07 so carry_src and align_mouths keep their measured scale (align 2.62 at the end)
    d_align = torch.sqrt(lip_xy ** 2 + (0.5 * (lip_dz - clearance)) ** 2)
    return {"d_hat_x": d_hat[:, 0], "sep_ok": sep_ok, "theta": theta, "theta_eff": theta_eff,
            "mis": mis, "lip_xy": lip_xy, "lip_dz": lip_dz, "d_align": d_align}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt(SIGNED) < pour.
    # iter_08 rebuilds only the tilt/pour stage and the attitude corridor. Measured in iter_07: grasp
    # 0.828/0.762, lifts 0.178/0.082 m, cups 0.189 m, aim_gate 0.739 — all best-ever and all kept.
    # The single failure is that `ctx.src_cup_tilt` is |acos(up_z)|: it pays the same for tilting at the
    # receiver, sideways, or away. Replay at ep1950: 65.4 deg of tilt, lip 0.128 m out, in_source 1.000.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.78-0.83 measured)
    W_AWAY_RCV = 0.8
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (best-ever grasp rates)
    # --- source lift: band + over-lift ramp. KEEP LIST — held 0.13-0.18 m for 1978 epochs ------------
    W_LIFT_SRC = 3.0
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.15
    LIFT_FREE_SRC = 0.24
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    LIFT_FLOOR = 0.60
    SRC_UP_FLOOR = 0.5
    LIFT_PINCH_BONUS = 0.2
    # --- receiver lift band — KEEP LIST (rcv lift 0.082 m, lift_rcv 1.37) ---------------------------
    W_LIFT_RCV = 2.0
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10
    RCV_LIFT_SIGMA = 0.07
    W_HOLD_RCV = 1.2
    RCV_DISC_FLOOR = 0.35
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7
    # --- convergence / carry / align — KEEP LIST, but now measured on the DIRECTED lip --------------
    W_CONVERGE = 3.0
    CONV_FAR = 1.00
    CONV_NEAR = 0.16
    CONV_SIGMA = 0.25
    W_CARRY, W_ALIGN = 3.0, 5.0
    CARRY_K, CARRY_RANGE = 1.2, 1.00
    ALIGN_K, ALIGN_SIGMA = 2.0, 0.15
    POUR_CLEARANCE = 0.04   # aim point 4 cm above the receiver rim: no cup-cup contact, origins > 9 cm
    SEP_MIN = 0.03          # below 3 cm of xy separation the pouring azimuth is numerically unstable
    # --- lip geometry gates ------------------------------------------------------------------------
    # cup_radius = 0.0432 * cup_scale 0.65 = 0.0281 m, cup_mouth_z = 0.0539 m (env cfg). At a 75 deg
    # pour the lip reaches cup_mouth_z*sin + r*cos = 0.059 m ahead of the source origin and 0.067 m
    # below the source rim, so the pouring pose is cups ~0.06 m apart in xy with dz ~0.11 m: origin
    # distance 0.125 m, safely outside the 0.09 m nesting line.
    SAFE_FULL, SAFE_ZERO = 0.030, 0.060    # lip over the receiver mouth (inner radius 0.028 m)
    # the aim radius is the env rule's own: premature_tilt_now() calls a tilt premature when the lip is
    # further than 8.1 cm from the receiver mouth. Full inside 8 cm, zero at 16 cm; the run currently
    # sits at lip_xy ~ 0.13 m, i.e. 0.16 — so tilting past theta_pre is still slightly downhill there
    # and the policy has to close the last 7 cm first. That distance has a live gradient (align/carry).
    APPR_FULL, APPR_ZERO = 0.080, 0.160
    LIP_DEADBAND = 0.02
    AIM_XY_K = 12.0
    LIP_Z_LO, LIP_Z_HI = 0.0, 0.02         # the lip must be ABOVE the receiver rim to pour into it
    LIP_Z_FREE, LIP_Z_SIGMA = 0.12, 0.10   # ... and not so high that the beads scatter
    LIP_ABOVE_FLOOR = 0.35                 # soft when GATING income, hard when PAYING the pose
    # --- TILT LADDER on the SIGNED angle -----------------------------------------------------------
    # release = first-bead angle. Measured for this cup: ~70 deg full, a fifth gone by 80, half by 86;
    # the angle rises as the cup empties, so the target follows bead_in_source_frac (same shape as the
    # sibling env's premature_tilt_limit_rad). Full cup: theta_pre 55 deg, release 75, keep-turning 105.
    RELEASE_FULL = math.radians(75.0)
    RELEASE_SPAN = math.radians(25.0)
    PRE_MARGIN = math.radians(20.0)
    OVER_RANGE = math.radians(30.0)
    W_TILT_AIM = 3.0        # 0 -> theta_pre, distance-free (cannot spill at any distance): 0.055/deg,
                            # against iter_07's tilt_lead 0.044/deg — a correct rotation loses nothing
    W_TILT_NEAR = 4.0       # theta_pre -> release, needs the spout inside the approach zone
    W_TILT_POUR = 7.0       # release -> release+30 deg, needs the spout OVER the receiver mouth
    W_POUR_POSE = 3.0       # plateau at the release attitude: success needs a 10-step hold
    BAND_BELOW, BAND_ABOVE = 0.10, 0.40
    LEAD_FLOOR = 0.25       # distance factor of tilt_aim is convergence, not a gate (iter_07 form)
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # Gaussian width on the corridor EXCESS (unchanged)
    TILT_LIFT_H = 0.03      # corridor and tilt_aim fully open at 3 cm — the policy lives at 0.178 m
    W_EARLY_TILT = 0.8      # stateless stand-in for a premature-tilt latch: charged every step the
                            # MAGNITUDE exceeds what the current aim allows
    MIS_FREE, MIS_SCALE = math.radians(10.0), 0.40
    W_MISDIR = 1.2          # the term with a live gradient at the stuck point (55 deg, wrong azimuth)
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
    W_SPILL = 60.0          # -3 per bead spilled; beads_left also deletes the whole tilt income
    W_SUCCESS = 20.0
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_TILT_PEN_RCV = 0.3
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35

    dtype = ctx.src_cup_tilt.dtype
    geo = _pour_geometry(ctx, ctx.cup_radius, POUR_CLEARANCE, SEP_MIN)
    d_align = geo["d_align"]
    theta_eff = geo["theta_eff"]

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)
    not_nested = 1.0 - nested

    # ---- spout gates: how close the pouring lip is to the receiver mouth --------------------------
    approach_gate = 1.0 - _smoothstep((geo["lip_xy"] - APPR_FULL) / (APPR_ZERO - APPR_FULL))
    safe = 1.0 - _smoothstep((geo["lip_xy"] - SAFE_FULL) / (SAFE_ZERO - SAFE_FULL))
    aim_xy = torch.exp(-AIM_XY_K * torch.clamp(geo["lip_xy"] - LIP_DEADBAND, min=0.0))
    lip_above_raw = torch.clamp((geo["lip_dz"] - LIP_Z_LO) / (LIP_Z_HI - LIP_Z_LO), 0.0, 1.0)
    lip_high = torch.exp(-(torch.clamp(geo["lip_dz"] - LIP_Z_FREE, min=0.0) / LIP_Z_SIGMA) ** 2)
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

    # ---- source attitude CORRIDOR, now opened by the AIM and charged on the MAGNITUDE --------------
    # 0.15 rad on the table -> theta_pre (55 deg full) once carried -> release (75 deg) once the spout
    # is inside the env's 8 cm aim radius -> release+30 deg once it is over the mouth itself. Each step
    # of the corridor matches the segment of the ladder that pays there, so nothing pays for an angle
    # the corridor charges. The excess is measured on ctx.src_cup_tilt on purpose: tilting sideways
    # or backwards is exactly as dangerous as tilting too early toward the receiver (that is also the
    # env rule for premature tilt in the sibling track — direction-free).
    carry_soft_src = g_src * torch.clamp(h_src / TILT_LIFT_H, max=1.0)
    allowed = TILT_FREE_TABLE + carry_soft_src * torch.clamp(theta_pre - TILT_FREE_TABLE, min=0.0)
    allowed = allowed + carry_soft_src * approach_gate * torch.clamp(release_tilt - theta_pre, min=0.0)
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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # 0.60 .. 1.00 (measured rcv_prog 0.75)

    # ---- source lift: band + linear cost above it (unchanged) -------------------------------------
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_lift_band = torch.exp(-(torch.clamp(h_src - LIFT_FREE_SRC, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_lift_band * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])
    # ungated by the grasp flag on purpose: throwing the cup up must not dodge the cost
    over_lift = torch.clamp((h_src - LIFT_FREE_SRC) / OVER_LIFT_RANGE, 0.0, 1.0)

    # ---- receiver lift band (unchanged — measured 0.082 m, inside the 0.10 m band) -----------------
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
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close     # 0.97 at the measured 0.189 m
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.76 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * not_nested * conv_shape

    # carry / align keep their iter_06 shape and weights, but d_align is now the DIRECTED spout, so a
    # cup aimed backwards no longer collects them (its lip rises and moves away from the target point)
    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * not_nested
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * not_nested
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- TILT: three segments of the SIGNED angle, each with a stricter aim requirement -------------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    # beads_left is the anti-spill interlock: beads on the table delete the whole ladder for the rest
    # of the episode, on top of spill_delta's -3 per bead.
    tilt_hold = carry_soft_src * beads_left * not_nested * geo["sep_ok"] * dir_soft
    lead_dist = LEAD_FLOOR + (1.0 - LEAD_FLOOR) * conv_shape          # 0.98 at the measured 0.189 m
    aim_frac = torch.clamp(theta_eff / theta_pre, 0.0, 1.0)                                  # 0 -> 55 deg
    near_frac = torch.clamp((theta_eff - theta_pre) / torch.clamp(release_tilt - theta_pre, min=0.05),
                            0.0, 1.0)                                                        # 55 -> 75
    pour_frac = torch.clamp((theta_eff - release_tilt) / (theta_max - release_tilt), 0.0, 1.0)  # 75 -> 105
    tilt_aim = tilt_hold * aim_frac * lead_dist
    tilt_near = tilt_hold * near_frac * approach_gate * lip_above
    # the release segment: only with the spout actually over the receiver mouth and above its rim
    pour_tilt = both_up * beads_left * dir_soft * upright_rcv * safe * lip_above * pour_frac
    # plateau at the release attitude so holding the pose beats swinging through it (10-step streak)
    pour_band = _smoothstep((theta_eff - (release_tilt - BAND_BELOW)) / (BAND_BELOW + BAND_ABOVE))
    pour_pose = both_up * beads_left * dir_ok * upright_rcv * safe * z_ok * aim_xy * pour_band

    # ---- directional / prematurity penalties -------------------------------------------------------
    early_tilt = torch.tanh(tilt_excess_src / TILT_TOL)      # magnitude above what the aim allows
    mis_tilt = carry_soft_src * torch.tanh(torch.clamp(geo["mis"] - MIS_FREE, min=0.0) / MIS_SCALE)
    # charged whenever the source cup is carried: the relative placement of the two cups is decided
    # during convergence, long before the spout is inside the aim radius
    pour_dir_pen = carry_soft_src * _smoothstep((geo["d_hat_x"] - 0.10) / 0.25)
    cup_center_d = (ctx.src_cup_pos - ctx.rcv_cup_pos).norm(dim=-1)
    near_nest = _smoothstep((NEST_SAFE - cup_center_d) / (NEST_SAFE - NEST_ZERO))   # 0 at the pour pose
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)

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
        "pour_delta": W_POUR * ctx.d_in_target,
        "transferred": W_TRANSFER * torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0) * not_nested * upright_rcv,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
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
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    # diagnostics only (added AFTER the sum, not part of the reward). The env logs only the
    # direction-free task/src_tilt_deg, so without these the next round cannot tell "not tilting" from
    # "tilting the wrong way": diag_theta_deg is the part of the tilt aimed at the receiver and
    # diag_mis_deg the part that is not. They should separate as the azimuth is learned.
    comps["rcv_progress"] = rcv_prog
    comps["aim_gate"] = safe
    comps["diag_theta_deg"] = theta_eff * (180.0 / math.pi)
    comps["diag_mis_deg"] = geo["mis"] * (180.0 / math.pi)
    comps["diag_lip_xy"] = geo["lip_xy"]
    return reward, comps
```

We trained an RL policy (PPO) using the reward function below and tracked the values of the individual reward components as well as task metrics (grasp rate per hand, cup lift, tilt, beads transferred, spill, success rate, episode length) at 10 evenly spaced points during training, plus the min / mean / max encountered:

adr/progress: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
bead/in_target: [0, 0.000635, 0.00112, 0.00127, 0.00161, 0.0022, 0.00137, 0.000879, 0.000732, 0.000781]  min 0 · mean 0.00115 · max 0.00366
bead/spill: [0, 0.0492, 0.0472, 0.06, 0.0737, 0.0789, 0.059, 0.058, 0.0568, 0.0465]  min 0 · mean 0.0547 · max 0.0882
done/drop: [0, 0, 0, 0.000977, 0, 0, 0, 0, 0, 0.000977]  min 0 · mean 0.000228 · max 0.00293
done/mimic_runaway: [0, 0, 0, 0.000977, 0, 0, 0, 0, 0.000977, 0]  min 0 · mean 0.000176 · max 0.00195
episode_lengths/step: [112, 699, 709, 753, 746, 722, 702, 736, 738, 745]  min 112 · mean 696 · max 860
reward/action_rate: [-0.0265, -0.0208, -0.0208, -0.0205, -0.0209, -0.0197, -0.0205, -0.0196, -0.02, -0.0207]  min -0.0265 · mean -0.0205 · max -0.0189
reward/aim_gate: [0, 0.22, 0.153, 0.177, 0.309, 0.426, 0.442, 0.465, 0.276, 0.419]  min 0 · mean 0.325 · max 0.688
reward/align_mouths: [0, 2.84, 2.44, 2.2, 2.37, 2.59, 2.74, 2.85, 2.66, 2.84]  min 0 · mean 2.58 · max 3.67
reward/alive: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min 1 · mean 1 · max 1
reward/away_rcv: [-0.0604, -0.182, -0.181, -0.19, -0.19, -0.187, -0.187, -0.187, -0.193, -0.187]  min -0.201 · mean -0.186 · max -0.0604
reward/carry_src: [0, 1.61, 1.43, 1.35, 1.34, 1.49, 1.6, 1.65, 1.6, 1.66]  min 0 · mean 1.51 · max 2.24
reward/close_rcv: [0.0902, 0.0665, 0.0676, 0.0682, 0.0654, 0.0676, 0.0698, 0.07, 0.0705, 0.0684]  min 0.0531 · mean 0.0681 · max 0.0902
reward/close_src: [0.0447, 0.0384, 0.0394, 0.0389, 0.0381, 0.0395, 0.0389, 0.0382, 0.0373, 0.0384]  min 0.031 · mean 0.0385 · max 0.0491
reward/contact_push: [-0.00244, -0.00115, -0.00416, -0.00314, -0.00362, -0.00634, -0.00288, -0.00486, -0.00389, -0.00261]  min -0.0413 · mean -0.00451 · max -0.000504
reward/converge: [0, 2.23, 2.04, 1.94, 1.84, 1.91, 2.04, 2.08, 2.03, 2.15]  min 0 · mean 2 · max 2.58
reward/cup_collision: [0, -6.55e-05, -0.00277, -0.00166, -0.000485, -0.000752, 0, -0.00276, -0.000389, -0.000498]  min -0.00411 · mean -0.00118 · max 0
reward/cup_disturb_rcv: [-0.0236, -0.115, -0.11, -0.115, -0.111, -0.103, -0.102, -0.104, -0.1, -0.109]  min -0.233 · mean -0.108 · max -0.0236
reward/cup_disturb_src: [-0.0649, -0.0143, -0.0188, -0.0274, -0.0275, -0.028, -0.0223, -0.026, -0.0217, -0.0206]  min -0.161 · mean -0.0246 · max -0.00174
reward/cup_fling: [-0.000407, -0.0113, -0.015, -0.0146, -0.0159, -0.0156, -0.0162, -0.014, -0.0169, -0.0189]  min -0.146 · mean -0.0175 · max -0.000407
reward/diag_lip_xy: [0.292, 0.625, 0.124, 0.11, 0.101, 0.099, 0.169, 0.197, 0.158, 0.0886]  min 0.0547 · mean 0.209 · max 0.866
reward/diag_mis_deg: [0.372, 8.69, 5.65, 5.54, 7.28, 10.9, 8.82, 8.98, 7.11, 8.26]  min 0.372 · mean 8.35 · max 19.7
reward/diag_theta_deg: [0.355, 51.2, 50.5, 49.1, 48, 41.8, 44.3, 44.1, 46.5, 44.5]  min 0.355 · mean 45.2 · max 54.5
reward/early_tilt: [-0.000898, -0.0548, -0.0528, -0.0692, -0.0666, -0.0688, -0.0647, -0.0689, -0.075, -0.0554]  min -0.251 · mean -0.065 · max -0.000898
reward/grasp_rcv: [0.000629, 1.13, 1.05, 1.04, 1.07, 1.17, 1.22, 1.26, 1.27, 1.22]  min 0.000629 · mean 1.15 · max 1.66
reward/grasp_src: [0.00218, 1.58, 1.48, 1.44, 1.48, 1.43, 1.49, 1.49, 1.49, 1.52]  min 0.00218 · mean 1.46 · max 1.82
reward/hand_foreign: [-0.00172, -0.00402, -0.00499, -0.00583, -0.0104, -0.00683, -0.00294, -0.00579, -0.00477, -0.00725]  min -0.0203 · mean -0.00623 · max -0.000611
reward/hold_rcv: [0, 0.408, 0.335, 0.32, 0.306, 0.347, 0.376, 0.368, 0.359, 0.399]  min 0 · mean 0.347 · max 0.46
reward/lift_rcv: [0, 1.33, 1.14, 1.04, 1.1, 1.23, 1.28, 1.33, 1.32, 1.33]  min 0 · mean 1.22 · max 1.56
reward/lift_src: [0, 2.86, 2.68, 2.52, 2.43, 2.16, 2.35, 2.37, 2.32, 2.23]  min 0 · mean 2.41 · max 3.22
reward/mis_tilt: [0, -0.123, -0.0675, -0.0663, -0.0976, -0.175, -0.134, -0.142, -0.0925, -0.122]  min -0.349 · mean -0.121 · max 0
reward/near_nested: [0, -0.0111, -0.00371, -0.00487, -0.0099, -0.00493, -0.000847, -0.0046, -0.00138, -0.00289]  min -0.0179 · mean -0.00441 · max 0
reward/nested: [0, -0.00977, -0.00391, -0.00586, -0.00391, -0.00586, 0, -0.00586, -0.00195, -0.00391]  min -0.0117 · mean -0.00328 · max 0
reward/over_lift_src: [0, -0.0156, -0.0195, -0.0226, -0.0621, -0.16, -0.134, -0.103, -0.134, -0.184]  min -0.205 · mean -0.0864 · max 0
reward/pour_delta: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -0.0146 · mean 0.000555 · max 0.022
reward/pour_dir_pen: [0, -0.85, -0.805, -0.755, -0.724, -0.611, -0.657, -0.609, -0.684, -0.684]  min -0.904 · mean -0.685 · max 0
reward/pour_pose: [0, 0, 0, 0, 0, 0, 0, 1.69e-08, 0, 0]  min 0 · mean 2.16e-08 · max 1.67e-06
reward/pour_tilt: [0, 0, 0, 5.44e-06, 0.000201, 0.000431, 0.00108, 0.00213, 0.000814, 0.000698]  min 0 · mean 0.000577 · max 0.00432
reward/rcv_home: [0, -0.0143, -0.0133, -0.0174, -0.0248, -0.0191, -0.0148, -0.0152, -0.0209, -0.0181]  min -0.0378 · mean -0.0181 · max 0
reward/rcv_progress: [0.264, 0.756, 0.705, 0.684, 0.685, 0.707, 0.734, 0.739, 0.736, 0.741]  min 0.264 · mean 0.716 · max 0.864
reward/rcv_still: [0, -0.308, -0.286, -0.273, -0.27, -0.288, -0.297, -0.321, -0.321, -0.314]  min -0.426 · mean -0.297 · max 0
reward/reach_rcv: [0.866, 0.766, 0.766, 0.762, 0.75, 0.753, 0.764, 0.762, 0.757, 0.762]  min 0.725 · mean 0.761 · max 0.888
reward/reach_src: [0.643, 0.545, 0.553, 0.543, 0.537, 0.545, 0.54, 0.539, 0.53, 0.539]  min 0.518 · mean 0.543 · max 0.643
reward/rush_near_cup: [-0.24, -0.0205, -0.0306, -0.0319, -0.0368, -0.0363, -0.0315, -0.0346, -0.0274, -0.03]  min -0.24 · mean -0.0331 · max -0.00611
reward/spill_delta: [0, -0.00586, -0.00879, -0.0322, -0.0176, -0.00586, -0.00293, -0.0176, -0.0234, -3.49e-09]  min -0.0615 · mean -0.0139 · max 0.0439
reward/success_bonus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
reward/tilt_aim: [0, 0.685, 0.669, 0.623, 0.597, 0.574, 0.61, 0.65, 0.637, 0.632]  min 0 · mean 0.622 · max 0.754
reward/tilt_near: [0, 0.206, 0.292, 0.289, 0.275, 0.201, 0.222, 0.221, 0.268, 0.242]  min 0 · mean 0.236 · max 0.33
reward/total: [2.23, 15.5, 14.3, 13.5, 13.5, 13.7, 14.6, 15, 14.6, 14.8]  min 2.23 · mean 14.2 · max 18.5
reward/transferred: [0, 0.00204, 0.00359, 0.00349, 0.00521, 0.00716, 0.00469, 0.00282, 0.00246, 0.0022]  min 0 · mean 0.00371 · max 0.0118
rewards/step: [107, 9.86e+03, 9.64e+03, 1.01e+04, 1.01e+04, 9.93e+03, 9.7e+03, 1.07e+04, 1.09e+04, 1.09e+04]  min 107 · mean 9.78e+03 · max 1.29e+04
task/aim_dist: [0.32, 0.684, 0.208, 0.167, 0.159, 0.179, 0.259, 0.283, 0.248, 0.193]  min 0.131 · mean 0.308 · max 0.951
task/cup_collision_rate: [0, 0, 0.00391, 0.00195, 0.000977, 0.000977, 0, 0.00391, 0.000977, 0.000977]  min 0 · mean 0.00158 · max 0.00586
task/cups_center_dist: [0.32, 0.717, 0.243, 0.202, 0.192, 0.208, 0.289, 0.312, 0.281, 0.223]  min 0.166 · mean 0.339 · max 0.983
task/episode_success: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0
task/nested_rate: [0, 0.00488, 0.00195, 0.00293, 0.00195, 0.00293, 0, 0.00293, 0.000977, 0.00195]  min 0 · mean 0.00164 · max 0.00586
task/rcv_cup_lift: [5.42e-05, 0.435, 0.0696, 0.0618, 0.0875, 0.0895, 0.0835, 0.0784, 0.0909, 0.0745]  min 5.42e-05 · mean 0.0971 · max 0.439
task/rcv_grasped: [0, 0.777, 0.678, 0.652, 0.67, 0.735, 0.758, 0.775, 0.775, 0.776]  min 0 · mean 0.725 · max 0.928
task/rcv_hand_foreign_rate: [0.00195, 0.00195, 0.00879, 0.0156, 0.0176, 0.0137, 0.00391, 0.00781, 0.00488, 0.0156]  min 0 · mean 0.01 · max 0.042
task/src_cup_lift: [0.000209, 0.612, 0.218, 0.166, 0.188, 0.213, 0.24, 0.221, 0.234, 0.226]  min 0.000209 · mean 0.255 · max 0.637
task/src_grasped: [0, 0.892, 0.845, 0.81, 0.794, 0.771, 0.795, 0.785, 0.791, 0.809]  min 0 · mean 0.8 · max 0.951
task/src_hand_foreign_rate: [0.00293, 0.00879, 0.00586, 0.00781, 0.0137, 0.00879, 0.00781, 0.0117, 0.00977, 0.0127]  min 0.00195 · mean 0.0106 · max 0.0234
task/src_tilt_deg: [0.727, 59.9, 56.1, 54.6, 55.3, 52.7, 53.1, 53.1, 53.6, 52.8]  min 0.727 · mean 53.5 · max 62.5
task/success_now: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 0 · max 0

Please carefully analyse the policy feedback and provide a new, improved reward function. Some helpful tips:
(1) If a task metric (e.g. success rate) is always near zero, the reward is not giving enough signal for that stage; rewrite it or scale it up.
(2) If a component's value is nearly constant over training, the policy is not optimising it — change its scale, its temperature/sharpness (e.g. the k in exp(-k·d)), or drop it.
(3) If a component's magnitude is much larger than the others, it may be dominating; rescale so the stages the policy has not yet reached are still worth pursuing.
(4) Look for exploits: e.g. high tilt reward with zero beads transferred means the policy tilts an empty/unlifted cup — gate that term on the state that makes it meaningful.
Then write the improved function following the same output rules as before.

Observations from watching the trained policy:
# round 9 operator notes — t2r_rh2_i08 (iter_08) 종료 + 결정론 트레이스 + 형제 트랙 대조

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 결론 한 줄

접근은 이미 충분하다. **막힌 것은 기울임 각도이고, 원인은 주둥이 높이 게이트가 기울기에 연동되지 않아
더 눕힐수록 손해가 되는 구조다.**

## 1. iter_08 이 고친 것과 남은 것

방향 항은 작동했다. `diag_theta_deg`(리시버 쪽 부호 있는 각) 45.3°, `diag_mis_deg` 7.8° 로
i07 의 방향 무구속 상태보다 낫다. 그러나 `pour_pose` 는 런 전체에서 정확히 0.0,
`pour_tilt` 0.0008, `episode_success` 0.0 이다. 런은 e374 에서 사용자 판정으로 종료했다.

## 2. 결정론 트레이스 (ep350, 16 env, 900 스텝, Traceback 0)

### 2-1 접근은 된다
- 컵 xy 거리 0.325 m → **0.107~0.117 m**, 이후 780 스텝 평탄.
- 컵 3D 거리 0.205~0.219 m, 소스가 리시버보다 **+0.16 m 위**.
- `diag_lip_xy` 후반 평균 **0.0894 m**, 전체 표본의 **26.3 %가 0.06 미만**,
  env 별 최솟값 0.014~0.125 m.

### 2-2 기울임이 안 따라온다 — 두 조건이 한 번도 동시에 성립하지 않았다
- `lip_xy < 0.06` 인 2101 표본: `theta` 평균 **56.1°**, `theta > 75°` 비율 **0.0000**.
- 전체 후반에서 `theta > 75°` 는 11.0 % 있는데, 그중 `lip_xy < 0.06` 인 것은 **0 개**.
- `lip_xy < 0.06 AND theta > 75°` 동시 표본 = **0 / 8000**.
- 결과: `pour_tilt` 8000 표본 전부 정확히 0, `pour_pose` 0.
- 비드: `in_target` 900 스텝 내내 **0.000**, `spill` 0 → **0.122**, `in_source` 1.000 → 0.878.
  즉 비드는 멀리서 기울일 때만 나가고 전부 바닥으로 간다.

### 2-3 방향 오차는 평균이 가리고 있었다
학습 로그 `diag_mis_deg` 평균은 7.8° 지만 트레이스 분포로는 **20° 초과 53.4 %, 40° 초과 36.8 %**,
`mis_tilt` 벌점도 -0.58 로 학습 로그(-0.11)의 5 배다.

### 2-4 액션 포화와 팔 추종
- 후반 500 스텝 소스 palm **y 액션 평균 +0.993, |a|>0.95 가 97.6 %**, **pitch 평균 +0.977, 92.0 %**.
- 팔 관절속도는 후반 소스 0.28~0.37 · 리시버 0.20~0.24 rad/s 로 거의 정착. 추종 중이 아니라 멈춘 것이다.
- `fabric/*_palm_err`: 앵커 자세(스텝 30)에서 양팔 **0.002 m · 0.88°** 로 완벽하다가,
  정책이 델타를 주는 스텝 60 부터 **리시버 0.14~0.25 m · 소스 0.05~0.11 m**, 회전 20~35° 로 벌어져
  끝까지 안 줄어든다. i07 도 같았다(소스 0.106~0.175, 리시버 0.192~0.243).
- 이 지표에 상수 오프셋은 없다. 부팅 로그에서 fabric FK 대 USD palm 정합이
  **소스 0.14 mm · 리시버 0.12 mm**, `fab→env` [0, 0, -2] mm 다.

## 3. 형제 트랙 pour_fabric(t2r_i18/i22) 대조

### 3-1 팔 델타 박스는 원인이 아니다 — 확대안 폐기
두 트랙 cfg 가 사실상 같다. 소스 y 상한 **+0.40**, 리시버 y 하한 **-0.40** 이 **동일**하고
z 하한만 -0.12(pour_fabric) vs -0.18(mimic) 로 다르다. 형제 트랙은 **같은 박스로**
`in_target` 0.70 · `episode_success` 0.59 를 냈다. 형제 트랙 `trace_i07_ep350` 기록에도
"작업공간 한계 아님: 소스 palm y 최대 +0.166, 리시버 최소 -0.166 까지 도달 가능" 이 남아 있다.
팔이 지령의 절반에서 멈추는 것은 양쪽 공통이고 그 상태로도 붓기는 성립한다.

### 3-2 거리는 이미 충분하고 벌어지는 것은 각도뿐이다
형제 트랙 i18 첫 이송 성공(epoch 1418) 대 이번 트레이스:

| | pour_fabric i18 e1418 | pour_bi_rh i08 ep350 |
|---|---|---|
| 컵 거리 | 0.233 m | **0.205~0.219 m** |
| 주둥이 거리 | 0.060 m | 0.089 m (26 % 는 0.06 미만) |
| 소스 기울기 | **77°** (수렴 후 100~112°) | **52~69°** |
| in_target | 0.492 | **0.000** |

컵 거리는 이쪽이 더 가깝다. 벌어지는 것은 기울기 하나다.

### 3-3 형제 트랙의 학습 순서는 접근 먼저, 기울임 나중이다
i18 fresh 진행: e879 주둥이 0.069·기울기 14.5° → e1147 주둥이 **0.055**·기울기 27.6°·이송 0
→ e1418 주둥이 0.060·기울기 **77.0**·이송 **0.492**. 주둥이를 먼저 붙이고 기울기가 나중에 뛴다.
iter_08 은 `tilt_aim` 을 거리와 무관하게 줘서 이 순서를 뒤집었다(§4-2).

### 3-4 형제 트랙은 높이 예산을 기울기에 연동한다 ★이번 라운드의 핵심
`reward_gen/pour_bi/iter_22/compute_reward.py` L238-242 주석 그대로:
"height budget grows with the tilt: the lip drops r*sin(theta) below the rim centre, so a cup
that must turn further has to ride higher. **Rotating and raising together must not be taxed.**"
구현은 `hover_free = HOVER_FREE0 + _LIP_R * sin_t`, `lip_free = LIP_Z_FREE0 + _LIP_R * sin_t`.

### 3-5 형제 트랙에는 palm 포화 벌점이 있다
`palm_sat_src` / `palm_sat_rcv` 가 있고 `ramp = _step(adr_progress / ADR_RAMP_FULL)` 로 램프한다.
형제 트랙도 i09 에서 "소스 팔 바깥 포화(py -0.99)" 를 겪고 i10 에 이 항을 추가했다.
★단 fresh 에서 전체 가중으로 켜면 죽는다 — i21 이 epoch 26 에 폐기됐다
("control penalties applied at full weight from step 0 of a fresh run").
★★**mimic 트랙 전용 컨텍스트에는 `adr_progress` 가 없다**(`episode_progress` 만 있다).
같은 램프를 그대로 옮길 수 없다.

## 4. iter_08 코드에서 확인된 원인

### 4-1 주둥이 높이 게이트가 고정 밴드다 ★
- `lip_dz = src_cup_mouth_pos[:,2] - lip_r * sin(theta) - rcv_cup_mouth_pos[:,2]` (L151)
- `LIP_Z_LO, LIP_Z_HI = 0.0, 0.02` (L213), `LIP_Z_FREE, LIP_Z_SIGMA = 0.12, 0.10` (L214) — **전부 상수**.
- `lip_above_raw = clamp((lip_dz - 0.0) / 0.02, 0, 1)` (L280)
- iter_08 에는 `sin(theta)` 가 L151 한 곳에만 있고 높이 허용치에는 없다.

더 기울일수록 주둥이가 `r·sin(theta)` 만큼 내려가 고정 밴드를 벗어나고, `lip_above` 가 죽으면서
`tilt_near`(L381)와 `pour_tilt`(L383)가 같이 꺼진다. **더 눕히는 것 자체가 손해다.**

정책은 이걸 스스로 상쇄하려 한다. 트레이스에서 기울기 구간별 컵 높이차가
theta 0~40° 0.164 m → 40~60° 0.152 m → 60~75° **0.176 m** → 75° 이상 **0.205 m** 로 올라간다.
높이 예산을 벌려고 컵을 올리는데 보상은 고정 밴드로 깎는다.

### 4-2 `tilt_aim` 이 사실상 거리와 무관하다
`lead_dist = LEAD_FLOOR + (1-LEAD_FLOOR) * conv_shape` (L374) 인데 `conv_shape` 는
`d_cups_xy` 기준이고 `CONV_NEAR = 0.16` 에서 포화한다(L356-357). 컵이 이미 0.11 m 라
`lead_dist` ≈ 0.985 로 상수다. 멀리서 기울여도 `tilt_aim` 을 거의 다 받는다.

### 4-3 통로 상한이 관측된 천장과 일치한다
`allowed` (L302-304) 는 `safe` 가 닫힌 상태에서 약 **74°** 에서 멈춘다
(TILT_FREE_TABLE 0.15 rad + theta_pre 55° + approach_gate 0.97 ×(75-55)°).
트레이스 최대 기울기 **69°** 가 이 통로 바로 아래다. `early_tilt` 가 그 위를 벌한다.

## 5. 다음 보상이 반드시 해야 할 것

1. ★**높이 허용치를 기울기에 연동한다.** `LIP_Z_LO`·`LIP_Z_HI`·`LIP_Z_FREE` 에
   `cup_radius · sin(theta_eff)` 를 더해, 더 눕히려고 컵을 올리는 행동에 벌점이 붙지 않게 한다.
   형제 트랙 iter_22 L238-246 이 참조 구현이다. 이것이 69° 천장을 여는 직접 수정이다.
2. `tilt_aim` 에 실제 거리 인자를 물린다. 지금의 `lead_dist` 는 `CONV_NEAR` 0.16 에서 포화해 상수다.
   주둥이 거리(`lip_xy`)를 직접 쓰거나 포화점을 0.06 근처로 내려 **접근 먼저, 기울임 나중** 순서를 만든다.
3. palm 액션 포화 벌점을 넣되 `adr_progress` 가 없으므로 다른 게이트를 쓴다
   (예: 양손 파지·들기 성립 이후에만, 또는 `episode_progress` 기준). fresh 전체 가중은 금지(§3-5).
4. 방향 오차는 평균 7.8° 가 아니라 20° 초과가 절반이다(§2-3). `mis_tilt` 의 `MIS_FREE`·`MIS_SCALE` 을
   이 분포에 맞춰 다시 잡는다.

## 6. 건드리지 말 것

- **env 는 바꾸지 않는다.** 팔 델타 박스는 형제 트랙과 같고 그 박스로 성공한 전례가 있다(§3-1).
- 파지·들기·converge·align 은 이번 런에서 성립했다(파지 src 0.798 · rcv 0.770, 컵 거리 0.21 m).
  가중·형태를 크게 바꾸면 회귀한다.
- `entropy_coef` 는 0 유지, sigma 는 손 12축만 다루는 기존 방침 유지.
- mimic 트랙 전용 컨텍스트에 없는 필드를 쓰면 static_check 에서 죽는다. 공용 컨텍스트의
  `src_tilt_toward_rcv`·`pour_dir_xy`·`premature_tilt`·`src_pour_lip_pos`·`adr_progress` 는 **없다**.
  방향·림 기하는 iter_08 처럼 `src_cup_up`·`rcv_cup_mouth_pos`·`cup_radius` 에서 직접 계산한다.
