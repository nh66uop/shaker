# iter_06 — bound the source lift, pay convergence more than height, size the tilt gates on the distance the policy reaches

## What the measured numbers say

**The causal chain is one loop, not four faults.** `lift_shape_src` had no upper bound
(`+ 0.15 * clamp((h - 0.15) / 0.10)` still paying at 0.25 m and nothing charging above it), so
`task/src_cup_lift` ran to 0.35 m mean / 0.57 m peak against a 0.15 m target while `reward/lift_src`
climbed 0.511 → 2.18. Height is the cheapest thing the source arm can do, and it pays 10× what
converging pays (`carry_src` 0.239 + `align_mouths` 0.201 = 0.44). With the source parked high the cups
separated — `task/cups_center_dist` 0.379 → 0.875 m (last 10 epochs 0.514 m) against a 0.32 m spawn —
and **both** tilt channels are conditioned on that distance:

- `tilt_lead`'s `0.3 + 0.7 * clamp(1 - d_align / 0.60)` is exactly its 0.3 floor for every
  d_align ≥ 0.60 m, i.e. flat over the whole 0.5–0.97 m the run lived in. The 45° cap never bound:
  `task/src_tilt_deg` stalled at 19°.
- `pour_gate = exp(-(d_align / 0.04)^2)` is exp(−43) at the *best* distance the run ever reached
  (`task/aim_dist` min 0.263 m). `reward/pour_tilt` is 0.000 at every one of the 569 epochs and
  `task/episode_success` 0.000. This is a numerically dead gate, not a hard task.

**The receiver was pulled out of its band by the same over-lift.** `d_align` counts half the height
error, so a source lip at 0.35–0.57 m demands a receiver rim near 0.25 m: `task/rcv_cup_lift` went
0.091 → 0.221 m. At 0.221 m the iter_05 band `exp(-(clamp(h-0.10)/0.05)^2)` is exp(−5.85) = 0.003 —
numerically dead in the other direction, so there was no gradient back down and
`reward/lift_rcv` collapsed 0.307 → 0.024. Everything that rides on holding the receiver eroded with
it: `grasp_rcv` 0.734 → 0.428/0.515, `task/rcv_grasped` 0.497 → 0.404, `close_rcv` 0.115 → 0.065,
`reach_rcv` 0.964 → 0.848.

**Geometry (env cfg, `cup_scale` 0.65): the pourable height is about 0.21 m, not 0.35 m.**
`cup_mouth_z` = 0.0829 × 0.65 = 0.0539 m, outer radius 0.0286 m. With the receiver at its 0.08 m
target its rim sits 0.134 m above its origin; the iter_05 pour target is 0.05 m above that rim, and a
horizontal source cup's lip hangs one outer radius (0.0286 m) below its own origin. So the source cup
origin belongs at ≈ 0.08 + 0.0539 + 0.05 + 0.0286 = **0.213 m** of lift, 0.233 m if the receiver sits
at the 0.10 m edge of its band. Everything above that is pure loss of alignment.
The env's success test is `xy(src_cup, rcv_cup) < 0.20 m` and `not nested` (centre distance ≥ 0.09 m),
so the convergence target is an **xy** distance of 0.16–0.20 m, which a 0.13 m height separation keeps
safely clear of nesting.

## Changes vs iter_05 (each constant tied to a measured number)

1. **`lift_src` is banded.** The unbounded over-pay is deleted and replaced by
   `src_lift_band = exp(-(clamp(h_src - 0.24) / 0.08)^2)`. LIFT_FREE_SRC 0.24 m is the computed
   pourable height (0.213 m at the receiver's 0.08 m target, 0.233 m at its 0.10 m band edge). The
   band is 1.00 at 0.21 m, **0.15 at the measured 0.35 m mean and 4e-8 at the 0.57 m peak**.
   `lift_shape_src` keeps the iter_18 shape (`0.5*(1-exp(-h/0.03)) + 0.5*clamp(h/0.15)`) unchanged.
2. **New `over_lift_src` (W 1.5, linear ramp over 0.20 m above 0.24 m).** A Gaussian band alone is
   flat where the policy actually is: at 0.35 m its slope is small and at 0.57 m it is zero. The linear
   ramp gives a constant 7.5/m downward gradient across the whole measured 0.24–0.44 m range
   (−0.82 at the 0.35 m mean, −1.50 at the 0.57 m peak) so the warm-started policy has somewhere to go.
3. **`W_LIFT_SRC` 4.0 → 3.0.** Measured ratio lift_src 2.18 : (carry 0.239 + align 0.201) ≈ 10:1 the
   wrong way. The lift is no longer the unsolved stage — `task/src_cup_lift` went 0.029 → 0.35 m.
4. **New `converge` (W 3.0) on the cup-to-cup xy distance**, the quantity the env's success test uses.
   `0.5 * clamp((1.00 - d_xy)/(1.00 - 0.16)) + 0.5 * exp(-(clamp(d_xy - 0.16)/0.25)^2)`.
   CONV_FAR 1.00 m is the top of the measured range (50-epoch means to 0.875 m, per-epoch 0.94–0.97 m);
   CONV_NEAR 0.16 m sits under the success threshold 0.20 m and above the nested threshold 0.09 m.
   Values: 0.06 at 0.90 m, 0.38 at 0.50 m, 0.74 at the 0.32 m spawn, 0.96 at 0.20 m — a live gradient
   over the exact interval the run occupied, worth +1.8 for closing 0.50 → 0.20 m.
5. **`carry_src` W 2.0 → 3.0 and widened**: `0.5*(1-tanh(1.2*d)) + 0.5*clamp(1 - d/1.00)`. The iter_05
   shape (`tanh(2.5*d)`, ramp 0.45) is 0.011 at the measured 0.9 m, i.e. flat; the new one is 0.15
   there and 0.68 at 0.30 m.
6. **`align_mouths` W 4.0 → 5.0 and widened** to `0.5*(1-tanh(2.0*d)) + 0.5*exp(-(d/0.15)^2)`
   (0.24 at 0.30 m, 0.54 at 0.15 m) instead of `tanh(4.0*d)` + `exp(-25*d)`, which is 0.03 at 0.30 m.
   Points 3–6 invert the incentive ratio: converge + carry + align now pay ≈ 6–8 at a 0.20 m gap
   against a 3.2 maximum for the whole source lift channel.
7. **`pour_sigma` 0.04 → `max(4*cup_radius, 0.15)` = 0.15 m.** At the measured best distance 0.263 m
   the old gate was exp(−43); the new one is 0.046, and 0.37 at 0.15 m. Still narrow enough that
   tilting past 45° while 0.30 m away is worth only 0.018 of the term, so it cannot pay to dump beads
   on the table. `W_TILT` 6.0 → 8.0: this is still the stage the policy has never entered, and the
   surrounding carry/align income grew in 5–6.
8. **`align_gate` (the attitude corridor's opener) 0.06 → 0.12 m**, so the corridor that allows 100° of
   tilt is open wherever `pour_tilt` pays (0.21 at 0.15 m → 57° free, 0.64 at 0.08 m → 80° free).
   The corridor itself, `TILT_FREE_TABLE` 0.15 rad, `TILT_PRE` 45°, `TILT_FULL` 100°, `TILT_TOL` 0.35,
   is unchanged — it is on the keep-list and it worked (tilt 0 → 19°).
9. **`tilt_lead` keeps its channel and weight (1.5); only its distance factor changes** from
   `0.3 + 0.7*clamp(1 - d_align/0.60)` to `0.25 + 0.75*conv_shape`. At the measured 0.9 m the old
   factor was pinned at its 0.3 floor; the new one is 0.30 at 0.90 m, 0.53 at 0.50 m and 0.97 at
   0.20 m, so tilting and converging now reinforce each other instead of the tilt waiting on a gate.
10. **`RCV_LIFT_SIGMA` 0.05 → 0.07.** At the measured 0.221 m the old band was 0.003 (no way back);
    0.07 gives 0.050 at 0.221 m, 0.60 at 0.15 m, 0.92 at 0.12 m — a climbable slope home.
    **`W_LIFT_RCV` 1.5 → 2.0**, because the measured collapse was 0.307 → 0.024, i.e. the band was
    abandoned, not over-paid.
11. **New `rcv_disc` factor replaces `home_f` on every convergence term**:
    `0.35 + 0.65 * rcv_band * exp(-(d_home/0.12)^2)`. iter_05 scaled carry/align by home distance only,
    so the receiver could still buy alignment by *lifting*. Now leaving the band or the home spot costs
    the same income: 0.90 when disciplined (h 0.10 m, drag 0.05 m) against 0.36 at the measured end
    state (h 0.221 m, drag ≈ 0.12 m implied by `rcv_home` −0.12…−0.25 at W 0.8). The 0.35 floor keeps
    the source's convergence gradient alive during the warm start, when the receiver is still at 0.205 m.
12. **New `hold_rcv` (W 1.2)**: `g_rcv * rcv_band * home_near * still_f * (0.4 + 0.6*src_carry)` —
    direct pay for a grasped, in-band, at-home, still receiver. It replaces, with a positive sign, what
    the receiver lost by drifting: `grasp_rcv` −0.30, `lift_rcv` −0.28, `close_rcv` −0.05 ≈ 0.63 per
    step, so staying disciplined is now worth clearly more than abandoning the band. It is scaled by
    the source's carry state (iter_18's `home_src_f` lesson) so an idle receiver cannot out-earn the
    source lifting. `rcv_home` (0.8) and `rcv_still` (0.3 + 0.5) keep the anti-drag intent unchanged.
13. **Unchanged**: the grasp/reach/close formulation for both hands and `W_GRASP` 2.0 (4/4 envs grasp,
    thumb 3–5 N), the `lift_gate` receiver-progress floor 0.60, `SRC_UP_FLOOR` 0.5,
    `LIFT_PINCH_BONUS` 0.2, `pour_delta` 150, `spill_delta` 60, `success_bonus` 20, and every safety
    penalty (slide, tilt, one-sided, palm, rush, fling, cup–cup, foreign, nested, action rate).

**Check on a hand-built context** (both hands grasped with the measured 4 N thumb / 4 N summed fingers,
receiver at 0.08 m and still, cup geometry at `cup_scale` 0.65). Under iter_05 the state
"source 0.15 m, cups 0.30 m apart" totals 15.42 and "source 0.35 m, cups 0.90 m apart" totals 15.24 —
a 0.18 difference, i.e. the policy was genuinely indifferent between the pour pose and the state it
converged to, and 0.57 m / 0.90 m scores the same 15.23. Under iter_06 the same three states are
20.49, 11.99 and 10.59.

Expected metric moves: `task/src_cup_lift` 0.35 → 0.20–0.24 m; `task/cups_center_dist` 0.51–0.88 →
below 0.35 m; `reward/converge` and `reward/carry_src` rising above `reward/lift_src`;
`task/rcv_cup_lift` 0.221 → 0.10 m with `reward/lift_rcv` recovering from 0.024;
`reward/grasp_rcv` back toward 0.73; `reward/pour_tilt` leaving 0.000 once `aim_dist` < 0.30 m.

```python
import torch
import math


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    iter_06: unchanged since iter_04 — the grasp works (4/4 envs, thumb 3-5 N, fingers 2.5-5 N)."""
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
    """Unweighted per-arm terms in [0, 1]. Reach / grasp / close are iter_04 VERBATIM (they work:
    both hands grasp in 4/4 deterministic envs, and iter_05 measured grasp_src 1.04 mean). The lift
    PAY is assembled in compute_reward so that no product of sub-1 gates sits on the early lift."""
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

    # closure shaping (unchanged): measured closure 0.55-0.65 on both hands while holding
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


def _pour_geometry(ctx, outer_r, pour_sigma, align_sigma):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    # iter_06: both sigmas widened. The corridor opener was 0.06 m and the pour gate 0.04 m, i.e.
    # exp(-43) at the closest distance the whole i05 run ever reached (aim_dist min 0.263 m) — that is
    # the logged pour_tilt of exactly 0.000 for 569 epochs, not a hard task.
    align_gate = torch.exp(-(d_align / align_sigma) ** 2)
    pour_gate = torch.exp(-(d_align / pour_sigma) ** 2)
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt < pour.
    # iter_06 breaks the measured i05 loop: an unbounded source lift (0.35 m mean, 0.57 m peak against
    # a 0.15 m target) -> cups 0.5-0.97 m apart (spawn 0.32 m) -> both tilt channels switched off by
    # distance (pour_tilt 0.000, success 0.000 for 569 epochs) -> receiver chases the height and leaves
    # its band (rcv_cup_lift 0.091 -> 0.221 m, lift_rcv 0.307 -> 0.024, grasp_rcv 0.73 -> 0.43).
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.85-0.97 measured:
    W_AWAY_RCV = 0.8                       # it keeps the left hand at its own cup)
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (4/4 envs grasp)
    # --- source lift: iter_18 lift-off shape, now inside a BAND -----------------------------------
    W_LIFT_SRC = 3.0        # was 4.0. Measured lift_src 2.18 vs carry 0.239 + align 0.201 = 10:1 the
                            # wrong way; the lift is solved (src_cup_lift 0.029 -> 0.35 m), convergence is not
    LIFTOFF_H = 0.03        # unchanged iter_18 lift-off e-fold: the first cm pay most
    LIFT_TARGET_SRC = 0.15  # unchanged: shape saturates here
    LIFT_FREE_SRC = 0.24    # pourable height from the env geometry (cup_scale 0.65): receiver rim at its
                            # 0.10 m band edge 0.1539 + 0.05 clearance + 0.0286 lip drop = 0.233 m
    LIFT_SIGMA_SRC = 0.08   # band 1.00 at 0.21 m, 0.15 at the measured 0.35 m mean, 4e-8 at the 0.57 m peak
    W_OVER_LIFT = 1.5       # the band alone is FLAT at 0.35-0.57 m; this ramp gives a live gradient there
    OVER_LIFT_RANGE = 0.20  # linear 0.24 -> 0.44 m: -0.82 at the 0.35 m mean, -1.50 at the 0.57 m peak
    LIFT_FLOOR = 0.60       # unchanged. Measured: the gate opened 0.70 -> 0.72 while lift_src was flat,
    SRC_UP_FLOOR = 0.5      # so it is not the blocker. Attitude may halve the lift income, never zero it.
    LIFT_PINCH_BONUS = 0.2  # grip quality is a BONUS, not a gate (iter_18 lesson)
    # --- receiver lift band -----------------------------------------------------------------------
    W_LIFT_RCV = 2.0        # was 1.5: the band was ABANDONED (lift_rcv 0.307 -> 0.024), not over-paid
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10    # 0.08 target + 2 cm margin (unchanged)
    RCV_LIFT_SIGMA = 0.07   # was 0.05 -> exp(-5.85) = 0.003 at the measured 0.221 m, a dead escape.
                            # 0.07 gives 0.050 at 0.221 m, 0.60 at 0.15 m, 0.92 at 0.12 m: a way back.
    W_HOLD_RCV = 1.2        # new: direct pay for grasped + in band + at home + still. The receiver lost
                            # ~0.63/step by drifting (grasp_rcv -0.30, lift_rcv -0.28, close_rcv -0.05)
    RCV_DISC_FLOOR = 0.35   # discipline factor floor: the source keeps a convergence gradient during the
                            # warm start, when the receiver still sits at the measured 0.205 m
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7      # measured reach_rcv away / near the cup
    # --- convergence: the stage that never happened ------------------------------------------------
    W_CONVERGE = 3.0        # new, on the cup-cup xy distance — the quantity the env's success test uses
    CONV_FAR = 1.00         # measured cups_center_dist 50-epoch means to 0.875 m, per-epoch 0.94-0.97 m
    CONV_NEAR = 0.16        # under the success threshold 0.20 m, above the nested threshold 0.09 m
    CONV_SIGMA = 0.25       # close-in half: 0.38 at 0.50 m, 0.74 at the 0.32 m spawn, 0.96 at 0.20 m
    W_CARRY, W_ALIGN = 3.0, 5.0            # were 2.0 / 4.0; measured carry 0.239 + align 0.201 vs lift 2.18
    CARRY_K, CARRY_RANGE = 1.2, 1.00       # was tanh(2.5*d) + ramp 0.45 -> 0.011 at the measured 0.9 m
    ALIGN_K, ALIGN_SIGMA = 2.0, 0.15       # was tanh(4.0*d) + exp(-25*d) -> 0.03 at 0.30 m
    W_TILT_LEAD = 1.5       # channel and weight kept: it moved the tilt from nothing to 19 deg
    LEAD_FLOOR = 0.25       # the distance factor is now convergence, not a 0.60 m gate pinned at its floor
    W_TILT = 8.0            # was 6.0; still the stage the policy has never entered (pour_tilt 0.000)
    TILT_PRE = math.radians(45.0)   # first bead leaves at ~70 deg with a full cup -> 45 deg cannot spill
    TILT_FULL = math.radians(100.0)
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # Gaussian width on the corridor EXCESS (unchanged)
    TILT_LIFT_H = 0.03      # corridor and tilt_lead fully open at 3 cm — the policy lives well above it
    ALIGN_GATE_SIGMA = 0.12  # was 0.06: the 100 deg corridor must be open wherever pour_tilt pays
                             # (0.21 -> 57 deg free at d_align 0.15 m, 0.64 -> 80 deg at 0.08 m)
    # --- receiver discipline -----------------------------------------------------------------------
    W_HOME_RCV = 0.8        # unchanged anti-drag penalty
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12       # unchanged income factor on the drag
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5   # 0.3 base, 0.8 while the source is carried (unchanged)
    STILL_FREE, STILL_SCALE = 0.05, 0.15   # 0.2 m/s -> tanh(1.0) = 0.76
    # --- task / penalties (all unchanged) ----------------------------------------------------------
    W_POUR = 150.0          # x d_in_target: 7.5 per bead with 20 beads
    W_SPILL = 60.0
    W_SUCCESS = 20.0
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_TILT_PEN = 0.3
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05
    CUP_OUTER_R = 0.0285    # 0.044 x cup_scale 0.65
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35

    dtype = ctx.src_cup_tilt.dtype
    # 4 x inner radius (0.0281 m at cup_scale 0.65) with a 0.15 m floor: 0.046 at the measured best
    # aim_dist 0.263 m instead of exp(-43), 0.37 at 0.15 m, 0.018 at 0.30 m (too small to pay for a dump).
    pour_sigma = max(4.0 * ctx.cup_radius, 0.15)
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R, pour_sigma, ALIGN_GATE_SIGMA)

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)

    # ---- source attitude CORRIDOR (kept from iter_05) ----------------------------------------------
    # free tilt: 0.15 rad on the table -> 45 deg once grasped and 3 cm up -> 100 deg when aligned.
    # A toppled cup on the table is 1.57 rad against 0.15 rad, so src_up ~ 0 exactly as in iter_04.
    carry_soft_src = g_src * torch.clamp(h_src / TILT_LIFT_H, max=1.0)
    tilt_free = TILT_FREE_TABLE + (TILT_PRE - TILT_FREE_TABLE) * carry_soft_src
    tilt_free = tilt_free + (TILT_FULL - tilt_free) * align_gate
    tilt_excess_src = torch.clamp(ctx.src_cup_tilt - tilt_free, min=0.0)
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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # 0.60 .. 1.00 (measured rcv_prog 0.60-0.65)

    # ---- source lift: iter_18 shape INSIDE A BAND, plus a linear cost above it ---------------------
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_lift_band = torch.exp(-(torch.clamp(h_src - LIFT_FREE_SRC, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_lift_band * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])
    # ungated by the grasp flag on purpose: throwing the cup up must not dodge the cost
    over_lift = torch.clamp((h_src - LIFT_FREE_SRC) / OVER_LIFT_RANGE, 0.0, 1.0)

    # ---- receiver lift: band around its 0.08 m target, now climbable from the measured 0.221 m -----
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * rcv_band * (0.5 + 0.5 * upright_rcv)

    # ---- receiver discipline: home, still, and a single factor on all convergence income -----------
    d_home = rcv["disp"]
    home_near = torch.exp(-(d_home / HOME_SIGMA) ** 2)
    still_f = 1.0 - torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    # iter_05 scaled carry/align by the drag only, so the receiver could still buy alignment by LIFTING
    # (measured 0.091 -> 0.221 m). Leaving the band now costs the same income as dragging.
    rcv_disc = RCV_DISC_FLOOR + (1.0 - RCV_DISC_FLOOR) * rcv_band * home_near
    rcv_home_pen = rcv["held"] * torch.tanh(torch.clamp(d_home - HOME_FREE, min=0.0) / HOME_SCALE)
    still_w = W_STILL_RCV + W_STILL_POUR * up_w_src               # stricter while the source is carried
    rcv_still_pen = still_w * rcv["held"] \
        * torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    # positive pay for doing exactly the right thing, scaled by the source carry state so an idle
    # receiver cannot out-earn the source lifting (iter_18 home_src_f lesson)
    hold_rcv = W_HOLD_RCV * g_rcv * rcv_band * home_near * still_f * (0.4 + 0.6 * carry_soft_src)

    # ---- convergence: the SOURCE closes the gap, sized on the distance the policy reaches ----------
    d_cups_xy = (ctx.src_cup_pos[:, :2] - ctx.rcv_cup_pos[:, :2]).norm(dim=-1)
    conv_lin = torch.clamp((CONV_FAR - d_cups_xy) / (CONV_FAR - CONV_NEAR), 0.0, 1.0)
    conv_close = torch.exp(-(torch.clamp(d_cups_xy - CONV_NEAR, min=0.0) / CONV_SIGMA) ** 2)
    conv_shape = 0.5 * conv_lin + 0.5 * conv_close     # 0.06 at 0.90 m, 0.38 at 0.50 m, 0.96 at 0.20 m
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.44 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * (1.0 - nested) * conv_shape

    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * (1.0 - nested)
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- tilt: the safe lead-in is now pulled by convergence, the pour gate is reachable -----------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_lead = carry_soft_src * beads_left * (1.0 - nested) \
        * torch.clamp(ctx.src_cup_tilt / TILT_PRE, max=1.0) * (LEAD_FLOOR + (1.0 - LEAD_FLOOR) * conv_shape)
    tilt_beyond = torch.clamp(torch.clamp(ctx.src_cup_tilt, max=TILT_FULL) - TILT_PRE, min=0.0) \
        / (TILT_FULL - TILT_PRE)
    pour_tilt = both_up * pour_gate * beads_left * upright_rcv * tilt_beyond

    tilt_pen_src = torch.tanh(tilt_excess_src / TILT_TOL)       # corridor excess only
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
        "tilt_lead": W_TILT_LEAD * tilt_lead,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE_SRC * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE_RCV * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
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
        "rcv_progress": torch.zeros_like(rcv_prog),
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    comps["rcv_progress"] = rcv_prog   # diagnostic only (added after the sum, not part of the reward)
    return reward, comps
```
