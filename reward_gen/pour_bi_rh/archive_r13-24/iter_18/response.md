# iter_18 — open the working height; the pour angle stays the policy's to find

## 1. What the task means and which stages the robot must go through

Twenty beads must travel **through the air** from the cup in front of the RIGHT (source) hand into the
cup in front of the LEFT (receiver) hand. Pushing one cup into the other transfers nothing
(`cups_nested` invalidates the success), so the beads have to leave a tilted source cup and land in an
upright receiver cup, with little spill and neither cup dropped.

Stages:

1. **Approach (both arms, in parallel).** Each hand brings its fingertips onto its own cup, thumb on
   one side and the other tips on the other, slowly: a 0.3-0.4 N side push tips these shakers over.
   The hand descends through a funnel, arriving ~45 mm above the pinch height and coming down only
   once the pinch centre is on the cup axis.
2. **Fingertip grasp (both arms).** Thumb AND at least one other finger pressing on the cup
   (`*_grasped`), the two sides balanced so the cup is squeezed rather than shoved.
3. **Lift (both arms).** Both cups leave the table. The receiver only needs a small steady lift. The
   source needs enough height for two different reasons: its rim must still clear the receiver rim
   after the lip drops during the tilt, **and — this is what this round is about — the arm can only
   realise a commanded palm roll when the palm is high.**
4. **Rendezvous.** The cups meet horizontally, inside the source arm's own region of the workspace so
   the source wrist still has range left at the moment of the pour.
5. **Aim.** The pouring lip (the rim point the beads leave from, which moves and drops as the cup
   turns) is brought over the receiver mouth and above its rim.
6. **Pour.** The source cup turns until beads leave it and fall into the receiver. **How far it must
   turn is still not something this reward knows or says.** The release angle depends on the fill, the
   bead pile, where the lip sits and how fast the cup is turning; the only thing measurable without
   guessing is whether beads came out and where they went. The angle is the policy's business, the
   beads are the reward's.
7. **Hold the grasp all the way.** The pinch that lifted the cup has to survive the rotation.
8. **Finish.** The source empties, the receiver holds what it got, both cups stop, nothing is dropped.

## 2. What i17g says: the wall is the working HEIGHT, not the reward shape

iter_17 removed every angle prescription. Two runs came back:

* **t2r_rh3_i17** (warm from i16g ep2100, entropy 3.5): pouring vanished — deterministic ep300 gave
  tilt 46.7 deg, `in_target` 0.000, residue 1.000.
* **t2r_rh3_i17g** (warm from the grasp stage rh2_i07-r3 ep400, entropy 22, the same start this round
  uses): tilt rose **by itself** 15 -> 30 deg and settled at 22 deg. Deterministic ep2900: grasp
  1.00/1.00, source lift 0.220 m, **tilt 26.1 deg, `in_target` 0.000, residue 1.000**.

So exploration does tilt the cup without being told an angle. It then stops, and the trace says why:

| measurement (i17g ep2800 trace, 64 env x 900 steps) | value |
|---|---|
| source palm roll COMMAND | pinned at the box limit -150 deg (median) |
| cup tilt actually achieved | 24.2 deg (median) |
| palm rotation error | 98 deg median (yaw 94.5) |
| joint usage r_aj_6 / r_aj_7 | 93.6 % / 98.8 % (saturated) |
| joint usage r_aj_2 | 22.6 % (idle), yet it holds the minimum margin in 92.9 % of steps |

The policy is *asking* for the roll and the arm cannot deliver it. The scripted probe (no learning,
empty hand, identical roll command) isolates the cause as height, not habit:

| palm z | roll error | j2 margin | bottleneck |
|---|---|---|---|
| 0.29-0.42 | **143-151 deg** | 0.07-0.16 | r_aj_6 / r_aj_4 |
| 0.53 | 38-52 deg | 0.09-0.18 | r_aj_5 |
| **0.69-0.76** | **10-23 deg** | 0.21-0.33 | r_aj_5 / r_aj_7 |

Height changes rotational reachability by roughly 15x; horizontal position matters far less.

**And the reward was the thing holding the arm down there.** `LIFT_FREE_SRC0 = 0.24`, on a 0.268 m
spawn with the 0.005 m dead band, puts the free ceiling at cup z ~= 0.51; the i17g deterministic
replay sat at cup z 0.478-0.50, i.e. exactly on the ceiling, with `diag_lift_budget` flat at
0.240-0.259 all run (the tilt-linked `lip_drop` part contributes ~0 at 26 deg). The loop closed on
itself: too low to turn, and not turning so no extra budget. Everything downstream is consistent with
a cup that is simply in the wrong place to rotate — `diag_release_beads` 0.005 beads/step,
`bead/in_target` 0.0004, `task/episode_success` 0.

Nothing in the bead economy is broken: `pour_delta` (150), `spill_delta` (-240), `unaimed_release`
(-60) never got a chance to fire because no bead was ever released. Re-tuning them would be guessing.

## 3. The one change

**Only the source lift allowance moves.** Two constants, both from the probe table:

| constant | old | new | why exactly this number |
|---|---|---|---|
| `LIFT_FREE_SRC0` | 0.24 | **0.40** | free ceiling cup z = 0.268 + 0.005 + 0.40 = **0.673**, landing the palm in the probe's 0.69-0.76 band where the roll error is 10-23 deg instead of 143-151 deg |
| `LIFT_TARGET_SRC` | 0.15 | **0.30** | the income peak has to move with the ceiling, or the best-paid height stays low. h = 0.30 is cup z 0.573, and 0.30 < 0.40 keeps the same ordering the old pair had (0.15 < 0.24): the lift shape saturates *inside* the free band, never by pushing past it |

Everything about the over-lift charge is kept: `W_OVER_LIFT = 1.5`, `OVER_LIFT_RANGE = 0.20`, ungated
by the grasp flag. It simply starts at the new ceiling, so throwing the cup upward is still charged,
now from h = 0.40 + `lip_drop` and saturating 0.20 m above it. The net per-step lift income
(`lift_src` + `over_lift_src`, at the nominal gates) moves like this:

```
h [m]   0.00  0.10  0.20  0.30  0.40  0.50  0.60  0.70
old     0.00  2.45  3.00  1.26 -1.15 -1.50 -1.50 -1.50
new     0.00  1.95  2.50  3.00  3.00 -0.12 -1.49 -1.50
```

The peak is now a plateau across 0.30-0.40 (cup z 0.573-0.673) and the cliff has moved up with it;
the slightly lower payment at h = 0.10 is the same ramp, just stretched, so the gradient still points
up from the table.

`lip_drop` stays in the budget (`lift_free_src = LIFT_FREE_SRC0 + lip_drop`). It is a geometric
allowance — how far the spout has already fallen — not an angle target, and it now rides on top of a
ceiling the arm can actually use.

**Checked, per operator note 6: is the permission window still reachable from up there?**
`lip_above_raw = clamp(lip_dz / (0.7 * cup_radius))` saturates at `lip_dz >= 0.021 m`; the measured
`diag_lip_dz` is 0.11-0.25 (max 0.471), so raising the source only pushes this factor further into
saturation — `perm` cannot be closed by the extra height, only by `perm_xy` and the sign gate, as
before. The one real cost is `d_align`, whose height half is `0.5 * (lip_dz - 0.04)`: +0.16 m of cup
height adds ~0.08 m to that half. About half of it is repaid by the tilt this round is trying to
unlock, because `lip_dz = (upright lip height) - lip_drop` and `lip_drop` reaches
`cup_radius + cup_mouth_z ~= 0.085 m` at 90 deg. `carry_src` and `align_mouths` keep their broad
components (`1 - tanh(1.2 d)`, `clamp(1 - d/1.00)`), so they still pay while the policy learns to
close the rest. `POUR_CLEARANCE` is deliberately NOT touched: that would be a second change, and
`diag_lip_dz` / `task/aim_dist` will say next round whether it needs one.

**Not touched, verbatim:** no angle target, band, ladder or angle-magnitude penalty anywhere (the
tilt appears only as `diag_src_tilt_deg` / `diag_theta_deg` diagnostics and as the `toward` sign gate);
the approach/permission split in `cup_radius` multiples (`PERM_IN_R/OUT_R`, `LEAD_FULL_R/ZERO_R`,
`LIP_Z_HI_R`); the rendezvous corridor (`meet_cross`, `src_cross`, `MEET_FULL`); `grip_hold_src/rcv`;
the receiver lift band (0.08 / 0.10) and receiver discipline (`rcv_home`, `rcv_still`, `hold_rcv`);
the cup speed and spin limits; every control/smoothness penalty; joint-limit and tremor rudders at
0.4; and the whole bead economy including the success bonus.

**Expected metric movement.** `task/src_cup_lift` 0.22 -> 0.35-0.40 and `diag_lift_budget` 0.25 ->
0.40+; `reward/over_lift_src` stays small (it must not become a large negative — if it does, the
policy is climbing past the new ceiling and `OVER_LIFT_RANGE` is the thing to look at). Then the ones
this round is judged on: `task/src_tilt_deg` above the 26 deg it froze at, `diag_release_beads` above
0.005/step, and `bead/in_target` off 0.0004. `diag_src_margin_min` / `diag_src_limit_hit` should
improve (the probe shows more j2 margin higher up); if `diag_lip_xy` and `task/aim_dist` grow instead,
the height was bought at the cost of the aim and `POUR_CLEARANCE` becomes next round's single change.

## 4. The reward function

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
