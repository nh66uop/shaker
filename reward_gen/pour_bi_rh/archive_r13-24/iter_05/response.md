# iter_05 — unblock the source lift (iter_18 shape), bound the receiver, open a tilt gradient

## What the measured numbers say

**The source lift was not blocked by the receiver gate — it was blocked by the gate product on top of it.**
`lift_src = 3.0 * lift_gate * g * exp(-(d_grip/0.08)^2) * calm * pose_ok_src * lift_prog`.

- `pose_ok_src` is the dominant loss. The trace has the source cup at `cup_up_z` 0.87–0.96 the whole
  episode, i.e. tilt 0.28–0.52 rad, and `pose_ok_src = exp(-(tilt/0.35)^2)` is **0.52 at 0.284 rad and
  0.11 at 0.516 rad**. The attitude gate alone removed 48–89 % of the lift pay while the hand held the
  cup perfectly well (thumb 3–5 N, other fingers 2.5–5 N, both hands grasped in 4/4 envs).
- `exp(-(d_grip/0.08)^2)` and `calm` cut it again. `d_grip` is redundant here: `src_grasped` already
  requires thumb > 1 N **and** another finger > 1 N on that cup, which the trace measures at 2.5–5 N.
- `lift_gate` went 0.70 → 0.72 over the 330 epochs in which `reward/lift_src` sat flat at 0.20
  (50-epoch means 0.214, 0.196, 0.201, 0.227, 0.228, 0.202). Confirmed: **the gate is not the blocker.**
- Even ungated the shape is weak at the height the policy actually reaches. `lift_prog` is 0.368 at
  h = 20 mm and 0.736 at h = 76 mm, so the whole measured lift range was worth
  3.0 × 0.72 × (0.736 − 0.368) ≈ 0.8/step before the gates, ≈ 0.2 after them. Two of four envs peak at
  55–61 mm and sink back to 20 mm because that costs almost nothing.

**The source never tilts because there is literally no gradient.** `pour_tilt` needs
`pour_gate = exp(-(d_align/0.041)^2)` with `task/aim_dist` measured at 0.32–1.16 m: `exp(-60)`. The log
agrees — `reward/pour_tilt` is 1e-38 … 1e-8 for the entire run, and the cup never leaves 0.87–0.96 of
upright. On top of that, `pose_ok_src` makes tilting *expensive*: at the 90° pour attitude it is 0, so
tilting would delete `grasp_src` (0.71 measured) and `close_src` as well.

**The receiver out-earns the source and drags the meeting point away.** `reward/lift_rcv` 0.614 vs
`reward/lift_src` 0.209 — three times the pay for the easier motion, and nothing stops the overshoot:
the trace lifts to 120–190 mm against `LIFT_TARGET_RCV` 0.08. That also breaks alignment, because
`d_align` counts half the height error: a receiver rim at 190 mm demands a source cup near 0.24 m.
Sideways the cup is dragged 210–270 mm and still growing at step 800, and `slide` is switched off by
`(1 - lifted_w)` as soon as the cup leaves the table, so the drag has **no direct cost** at all — it is
only paid for indirectly by `grasp_rcv` decaying 1.6–2.0 → 0.3–0.6 and `carry_src` 0.31–0.61 → 0.09–0.26.
(Hypothesis, not verified: the drag is partly *paid for* by `align_mouths` 4.0 and `carry_src` 2.0, which
depend on relative geometry and so can be earned by moving either cup.)

## Changes vs iter_04 (each constant tied to a measured number)

1. **`lift_src` rebuilt in the iter_18 shape, no product of sub-1 gates.**
   `lift_shape_src = 0.5*(1-exp(-h/0.03)) + 0.5*clamp(h/0.15) + 0.15*clamp((h-0.15)/0.10)`, and
   `lift_src = 4.0 * g_src * lift_shape_src * src_up_soft * lift_gate * (1 + 0.2*pinch)`.
   `d_grip` and `calm` dropped (see above; `cup_fling` still covers ejection, and the trace shows no
   ejection — the 84–285 mm step is the reset at step 897). 3 cm now pays 1.49 where iter_04 measured 0.20.
2. **`src_up_soft = 0.5 + 0.5*src_up`** (iter_18 floor). At the measured tilt 0.375 rad the old gate was
   0.32; the floor makes the worst case 0.5 instead of 0.11.
3. **Attitude becomes a corridor instead of a fixed gate.** Free tilt is 0.15 rad on the table, ramps to
   45° once the cup is grasped and 3 cm up, and to 100° when aligned. Knock-over protection is unchanged
   where it matters: a toppled cup on the table is 1.57 rad against a 0.15 rad allowance, so `src_up` ≈ 0
   exactly as in iter_04. `W_LIFT_SRC` 3.0 → 4.0 (the receiver channel is cut to 1.5, see 6).
4. **`lift_gate` floor 0.15 → 0.60.** Measured gate was 0.70–0.72 and was not the blocker, so the
   receiver-progress coupling is kept but can no longer remove more than 40 % of the lift pay.
5. **New `tilt_lead` (W 1.5)**: pays `min(tilt, 45°)/45°` once the source is grasped and ≥3 cm up (the
   policy already lives at 20–76 mm), scaled by `0.3 + 0.7*clamp(1 - d_align/0.60)` — at the measured
   `aim_dist` 0.32 m that factor is 0.63, i.e. a live gradient where `pour_tilt` was exp(-60). Capped at
   45° because the first bead only leaves at ~70°, so the lead-in cannot spill.
6. **`pour_tilt` reshaped (W 5.0 → 6.0)**: it now pays only the part **above** 45°, with
   `pour_gate = exp(-(d_align/max(1.2*cup_radius, 0.04))^2)` instead of `/0.041`, and is multiplied by
   `beads_left` so tilting an emptied cup earns nothing (prompt tip 4).
7. **Receiver bounded.** `W_LIFT_RCV` 3.0 → 1.5 (measured lift_rcv 0.614 vs lift_src 0.209) and a band
   `exp(-(clamp(h-0.10)/0.05)^2)`: free to 0.10 m (0.08 target + margin), 0.45 at 0.15 m, 0.06 at 0.19 m —
   the measured 120–190 mm overshoot loses almost all of it (whole term: 1.45 at 8 cm, 0.09 at 18 cm).
8. **New `rcv_still` (0.3, rising to 0.8 while the source is carried)**: `tanh(clamp(|v| - 0.05)/0.15)`.
   At 0.2 m/s that is −0.61.
9. **New `rcv_home`** (W 0.8): `tanh(clamp(d_xy - 0.05)/0.10)` on the receiver cup's xy displacement from
   spawn, active whenever the cup is held — this is the hole `slide` left, since `(1 - lifted_w)` zeroes
   `slide` above 2.5 cm. At the measured 210–270 mm drag it is −0.74 to −0.79.
10. **`align_mouths` and `carry_src` are scaled by `0.3 + 0.7*exp(-(d_home/0.12)^2)`**, so at a 210–270 mm
    drag they keep only 0.33–0.30 of their value: converging by moving the receiver no longer pays.
11. Grasp/reach/close formulation, `pour_delta` 150, `spill_delta` 60, `success_bonus` 20, and every
    safety penalty (rush, one-sided, palm, fling, cup-cup, foreign, nested, action rate) are unchanged.

Expected metric moves: `task/src_cup_lift` up from 0.02–0.08 toward 0.15; `reward/lift_src` from 0.2
toward 2–4; `task/src_tilt_deg` off its 11° ceiling; `task/rcv_cup_lift` down from 0.12–0.19 toward 0.08;
receiver xy drag down from 0.21–0.27 m.

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
    iter_05: unchanged from iter_04 — the grasp works (4/4 envs, thumb 3-5 N, fingers 2.5-5 N)."""
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
    both hands grasp in 4/4 deterministic envs). iter_05 only removes the lift PAY from here — it is
    assembled in compute_reward so that no product of sub-1 gates sits on the early lift."""
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


def _pour_geometry(ctx, outer_r, pour_sigma):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    align_gate = torch.exp(-(d_align / 0.06) ** 2)                       # wide: tilting is allowed here
    # iter_05: sigma 0.041 -> ~0.04-0.05. With aim_dist measured at 0.32-1.16 m the old gate was exp(-60),
    # which is exactly the logged pour_tilt of 1e-38..1e-8. The wide lead-in now lives in tilt_lead.
    pour_gate = torch.exp(-(d_align / pour_sigma) ** 2)
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Ladder: reach < grasp < lift per arm, then carry < align < tilt < pour.
    # iter_05 fixes three measured failures: (1) the source lift was strangled by a gate product,
    # (2) the receiver over-lifted and dragged its cup, (3) the tilt stage had no reachable gradient.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (it kept the receiver at its cup:
    W_AWAY_RCV = 0.8                       # rcv_near_rate 0.84 vs 0.05 in i03)
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (4/4 envs grasp)
    # --- source lift (iter_18 shape) -------------------------------------------------------------
    W_LIFT_SRC = 4.0        # was 3.0 and shared with the receiver; the receiver channel drops to 1.5
    LIFTOFF_H = 0.03        # e-fold of the lift-off half: 3 cm -> 0.57 of it, so the first cm pay most
    LIFT_TARGET_SRC = 0.15  # unchanged; the trace only ever reached 20-76 mm
    LIFT_OVER_W = 0.15      # keeps paying past the target instead of stopping dead at 0.15 m
    LIFT_OVER_RANGE = 0.10
    LIFT_FLOOR = 0.60       # was 0.15. Measured: the gate opened 0.70 -> 0.72 while lift_src stayed at
                            # 0.20 for 330 epochs, so it was not the blocker; keep the coupling, cap the loss
    SRC_UP_FLOOR = 0.5      # iter_18 floor: attitude may halve the lift income, never zero it (it was 0.11
                            # at the measured cup_up_z 0.87)
    LIFT_PINCH_BONUS = 0.2  # grip quality is a BONUS, not a gate (iter_18 lesson)
    # --- receiver lift band ----------------------------------------------------------------------
    W_LIFT_RCV = 1.5        # was 3.0: measured lift_rcv 0.614 vs lift_src 0.209 — the easy motion paid 3x
    LIFT_TARGET_RCV = 0.08
    RCV_LIFT_FREE = 0.10    # 0.08 target + 2 cm margin
    RCV_LIFT_SIGMA = 0.05   # 0.15 m -> 0.45, 0.19 m -> 0.06: the measured 120-190 mm overshoot loses the pay
    PROG_REACH, PROG_GRASP = 0.35, 0.65
    REACH_AWAY, REACH_NEAR = 0.2, 0.7      # measured reach_rcv away / near the cup
    # --- carry / align / tilt --------------------------------------------------------------------
    W_CARRY, W_ALIGN = 2.0, 4.0            # unchanged weights
    W_TILT_LEAD = 1.5       # new: the wide, safe part of the tilt (<= 45 deg, no bead can leave)
    W_TILT = 6.0            # was 5.0; this is the stage the policy never entered (pour_tilt 0.000)
    TILT_PRE = math.radians(45.0)   # first bead leaves at ~70 deg with a full cup -> 45 deg cannot spill
    TILT_FULL = math.radians(100.0)
    TILT_FREE_TABLE = 0.15  # rad, ~9 deg of free tilt while the cup is not lifted (knock-over protection)
    TILT_TOL = 0.35         # same Gaussian width as iter_04, now applied to the corridor EXCESS
    TILT_LIFT_H = 0.03      # tilt corridor and tilt_lead are fully open at 3 cm — the policy already
                            # reaches 20-76 mm, so the gradient is live where the policy lives
    LEAD_RANGE = 0.60       # aim_dist measured 0.32 m -> lead factor 0.63 (was exp(-60) = 0)
    # --- receiver discipline ---------------------------------------------------------------------
    W_HOME_RCV = 0.8        # measured drag 210-270 mm with slide switched off above 2.5 cm
    HOME_FREE, HOME_SCALE = 0.05, 0.10
    HOME_SIGMA = 0.12       # income factor: 0.21 m -> 0.05, so converging by dragging the receiver fails
    W_STILL_RCV, W_STILL_POUR = 0.3, 0.5   # 0.3 base, 0.8 while the source is carried
    STILL_FREE, STILL_SCALE = 0.05, 0.15   # 0.2 m/s -> tanh(1.0) = 0.76
    # --- task / penalties (all unchanged from iter_04) --------------------------------------------
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
    CUP_OUTER_R = 0.0285
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35

    dtype = ctx.src_cup_tilt.dtype
    pour_sigma = max(1.2 * ctx.cup_radius, 0.04)   # beads must land inside the receiver mouth
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R, pour_sigma)

    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    h_src = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    h_rcv = torch.clamp(ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    nested = ctx.cups_nested.to(dtype)

    # ---- source attitude CORRIDOR (replaces the fixed upright gate) --------------------------------
    # free tilt: 0.15 rad on the table -> 45 deg once grasped and 3 cm up -> 100 deg when aligned.
    # A toppled cup on the table is 1.57 rad against 0.15 rad, so src_up ~ 0 exactly as in iter_04,
    # but a deliberate carried pour no longer deletes grasp_src (0.71 measured) and the lift pay.
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

    # ---- receiver progress gate on the source lift -------------------------------------------------
    up_w_src = g_src * torch.clamp(h_src / 0.05, max=1.0)      # "held and at least 5.5 cm up"
    up_w_rcv = g_rcv * torch.clamp(h_rcv / 0.05, max=1.0)
    prog_reach = torch.clamp((rcv["reach"] - REACH_AWAY) / (REACH_NEAR - REACH_AWAY), 0.0, 1.0)
    prog_grasp = torch.clamp(rcv["grasp"] / 0.4, max=1.0)
    rcv_prog = PROG_REACH * prog_reach + PROG_GRASP * torch.maximum(prog_grasp, up_w_rcv)
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # 0.60 .. 1.00 (measured rcv_prog 0.64 -> 0.86)

    # ---- source lift: iter_18 shape, single floored attitude factor, grip as a bonus ---------------
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_over = torch.clamp((h_src - LIFT_TARGET_SRC) / LIFT_OVER_RANGE, 0.0, 1.0)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0) \
        + LIFT_OVER_W * lift_over                              # 3 cm -> 0.37, 15 cm -> 0.98, 25 cm -> 1.15
    src_up_soft = SRC_UP_FLOOR + (1.0 - SRC_UP_FLOOR) * src_up
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_up_soft * lift_gate \
        * (1.0 + LIFT_PINCH_BONUS * src["pinch"])

    # ---- receiver lift: banded around its 0.08 m target, over-lifting loses the pay ----------------
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    rcv_band = torch.exp(-(torch.clamp(h_rcv - RCV_LIFT_FREE, min=0.0) / RCV_LIFT_SIGMA) ** 2)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * rcv_band * (0.5 + 0.5 * upright_rcv)

    # ---- receiver stays where it spawned and stays still --------------------------------------------
    d_home = rcv["disp"]
    home_f = 0.3 + 0.7 * torch.exp(-(d_home / HOME_SIGMA) ** 2)   # income factor for carry / align
    rcv_home_pen = rcv["held"] * torch.tanh(torch.clamp(d_home - HOME_FREE, min=0.0) / HOME_SCALE)
    still_w = W_STILL_RCV + W_STILL_POUR * up_w_src               # stricter while the source is carried
    rcv_still_pen = still_w * rcv["held"] \
        * torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)

    # ---- carry / align: the SOURCE must close the gap, not the receiver ----------------------------
    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * home_f * (1.0 - nested)
    carry = carry_w * (0.5 * (1.0 - torch.tanh(2.5 * d_align)) + 0.5 * torch.clamp(1.0 - d_align / 0.45, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * home_f * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(4.0 * d_align)) + 0.5 * torch.exp(-25.0 * d_align))

    # ---- tilt: a wide safe lead-in, then the narrow aligned pour -----------------------------------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    lead_near = torch.clamp(1.0 - d_align / LEAD_RANGE, 0.0, 1.0)
    tilt_lead = carry_soft_src * beads_left * (1.0 - nested) \
        * torch.clamp(ctx.src_cup_tilt / TILT_PRE, max=1.0) * (0.3 + 0.7 * lead_near)
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
