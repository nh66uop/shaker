# iter_14 — 전량 이송·마무리 안정화에 값을 친다 (구조는 iter_13 유지)

## 1. What the task means and which stages the robot must go through

The task is a two-arm liquid-style transfer of 20 beads between two slim, light shaker cups.
It decomposes into stages that must be *ordered* by the reward, because every later stage is only
meaningful once the earlier one holds:

1. **Reach, per arm.** Each hand must bring its fingertips to its own cup without brushing it:
   the cups tip over at 0.3-0.4 N of one-sided contact. The thumb has to arrive on one side and the
   other fingertips on the opposite side (opposition), at pinch height on the cup axis.
2. **Fingertip grasp, per arm.** Thumb AND another finger pressing the cup at the same time
   (`src_grasped` / `rcv_grasped`), with balanced two-sided force so the cup is pinched, not pushed.
3. **Lift, per arm.** Both cups leave the table. The source needs enough altitude that its rim can
   later sit above the receiver rim *after* it has rotated (a turning cup loses height:
   `r*sin(theta) + mouth_z*(1 - cos(theta))`), the receiver only needs a small, stable lift.
4. **Converge and carry.** The source cup travels over to the receiver while the receiver stays
   near its home spot, upright and still. The cups must not touch and must not nest.
5. **Aim.** The *lip* (the rim point the beads will actually leave from, on the side facing the
   receiver) must be brought over the receiver mouth and above its rim, BEFORE the tilt reaches the
   first-bead angle (about 70 deg with a full cup). Approaching and being allowed to tilt are two
   different things and are priced separately (iter_13).
6. **Pour.** Tilt the source toward the receiver, in a controlled, real-robot-executable way, so that
   beads leave through the air and land in the receiver. Every bead that misses is permanent loss.
7. **Finish.** ★ This is the stage iter_13 did not price: empty the source completely, then bring both
   cups back to an upright, stationary, still-grasped pose and hold it (the env now wants 20 steps).

## 2. What iter_13 measured, and what this round changes

### 2.1 The diagnosis

Deterministic replay (note section 6) shows the policy **already empties the source cup**
(`in_source` 0.000, `in_target` 0.906 at both ep550 and ep1250). The learning-curve `in_source` 0.48
is exploration noise, not a missing skill. So the target of this round is robustness plus removing
the flat region, not teaching the pour.

The flat region is real and measurable. Per-step income in iter_13 at the operating point:

| term | value | behaviour above in_target 0.5 |
|---|---|---|
| `success_bonus` W 20 x success x clean_f | 5.34 mean | **flat** (threshold was 0.50) |
| `transferred` W 4.0 x in_target | 0.84 mean | +2.0 total for 0.5 -> 1.0 |
| `pour_delta` W 150 x d_in_target | 0.11 mean | +7.5 once per bead |
| `spill_delta` W 240 x d_spill | -0.13 mean | -12 per bead, permanent |

Pouring the second half is worth about +2/step of standing income while risking -12 per bead and the
loss of the whole success income. With the env thresholds now at 0.80 fill / 0.25 spill / 20-step
hold, the reward must make "finish it" strictly better than "stop at half".

### 2.2 The four changes

**(a) `clean_f` follows the new spill limit: `SPILL_ZERO` 0.40 -> 0.25.** The env's
`success_spill_max` is now 0.25, so at 0.25 the success income is worth exactly nothing, which is the
same rule iter_13 used against 0.40. Consequence at the measured spill 0.133: `clean_f` 0.667 -> 0.468
(deterministic spill 0.09: 0.775 -> 0.64). Because that factor also scales the whole tilt ladder, the
ladder would lose about a third of its income, so the ladder uses
`clean_pour = 0.10 + 0.90 * clean_f`: an episode that has already blown the spill budget keeps a tenth
of the pour income (the beads still in the cup must go somewhere and it must keep learning where),
while a clean episode is worth ten times as much. `success_bonus` and `transferred` keep the hard,
env-matched factor with no floor.

**(b) Full-transfer income, steepening toward the end.** `W_TRANSFER` 4.0 -> 5.0 plus a new
`transfer_tail`. Every constant comes from the measurements in the notes:

- `TAIL_LO = 0.35`: just below the stochastic operating point (`bead/in_target` 0.386 mean,
  0.376-0.471 over the run), so the slope is alive where the policy actually lives instead of starting
  flat above it.
- `TAIL_HI = 0.90`: the deterministic replay value (0.906 at both checkpoints), i.e. the value the
  policy has already demonstrated. The tail is worth full price exactly at what is provably reachable.
- shape `0.5*ramp + 0.5*ramp^2`: monotone, never flat above `TAIL_LO`, and the marginal value per bead
  roughly doubles from the low end to the high end (W x 1.27 per unit in_target near 0.4, W x 2.7 near
  0.9). This is the "끝으로 갈수록 가팔라지게" the note asks for, without a dead zone.
- `W_TRANSFER 5.0`: sized so the DEMONSTRATED state keeps its old income under the stricter clean_f
  (4.0 x 0.906 x 0.775 = 2.81 before, 5.0 x 0.906 x 0.64 = 2.90 now) while the half-poured state loses
  income (4.0 x 0.45 x 0.65 = 1.17 -> 5.0 x 0.45 x 0.44 = 0.99). The gap widens on both sides.
- `W_TAIL = 4.0`: 0 at in_target 0.35, +2.6/step at the demonstrated 0.906 after clean_f. That is half
  the measured `success_bonus` income, so finishing is worth pursuing but never outbids the success
  state itself.

**(c) Emptying income (`src_emptied`).** `1 - bead_in_source_frac` with `EMPTY_LO = 0.45`
(current emptied 0.517 at in_source 0.483, 0.610 at the peak block's 0.390 - again just below the
operating point) and full at 1.0 (the deterministic in_source 0.000). Same ramp shape.
`W_EMPTY = 3.0` -> +1.9/step at an empty source after clean_f.
Exploit guard: beads can also leave the source by falling on the table, so this term is multiplied by
`clean_f` (0 at the 0.25 spill limit) AND by `fill_ok = in_target / TAIL_LO` - an empty source with
nothing delivered pays nothing.

**(d) Finish stabilisation (`settle_finish`).** Pays only in the state the task ends in:
near-full transfer AND near-empty source (`finish_q`, a MEAN ramp of the two, not a min - a min-based
term is flat whenever the two are tied), both cups still grasped (`g_src * g_rcv`, hard: dropping a
cup must not buy stillness), both cups upright within the same `UPRIGHT_TOL = 0.35` rad the receiver
already uses and the env's success test uses, and both cups still. Stillness reuses the EXISTING
speed constants (`SRC_V_FREE`, `SRC_W_FREE`, `RCV_W_FREE`, `STILL_FREE`) as a mean of four one-sided
ramps, so every axis keeps its own slope and no new threshold is invented.
`W_SETTLE = 3.0`: up to +3/step, only in the finished pose.
Anti-parking: the sibling track's lesson is that a state-proportional income becomes "pose and freeze".
Here freezing is impossible before the beads are delivered (`finish_q` is ~0 at the current operating
point: 0.5*(0.035 + 0.068) = 0.052, i.e. about 0.1/step), and freezing AFTER the transfer is exactly
the 20-step hold the env now demands.

Net effect: the demonstrated finished state gains about +7.5/step over the half-poured plateau
(tail 2.6 + emptied 1.9 + settle up to 3.0), against a total reward of about 20. At the current
stochastic operating point the new terms sum to roughly 0.2/step, so nothing is displaced and the
existing ladder keeps its measured balance.

### 2.3 Kept verbatim (note section 4.4)

Approach/permission split (`approach_lip`, `perm` window, every aim window in `ctx.cup_radius`
multiples), the aim discipline (`lead_gate`, `unaimed_release`, `aimed_tilt`, corridor), all speed and
control charges (`src_speed`, `src_spin`, `rot_pour`, `rcv_spin`, per-arm `palm_sat` / `palm_bang` /
`act_flip` / `carry_rate`), `flow_excess`, `overtilt`, `W_SPILL 240`, and the whole grasp / lift /
converge / carry / align stack. No new absolute distance constant is introduced.

Expected metric movement: `bead/in_target` up from 0.39 toward 0.6+ with `in_source` down from 0.48;
`bead/spill` flat or lower (clean_f is stricter); `task/episode_success` restarts near 0 because the
env thresholds moved (0.80 / 0.25 / 20 steps) and must be read against `in_target` / `in_source`, not
against the previous run.

## 3. Reward function

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
    bottom and 1.5/(hi-lo) at the top, so the last beads are worth three times the first ones.
    Deliberately NOT a smoothstep: smoothstep has zero slope at t = 0, i.e. no gradient exactly at the
    operating point the policy sits on today."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted).
    iter_14: VERBATIM since iter_04 - the grasp works (src_grasped 0.722, rcv_grasped 0.760)."""
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
    iter_06 VERBATIM - on the keep list (grasp_src 1.21, grasp_rcv 1.10 measured over the iter_13 run,
    both lift bands held). iter_14 changes NOTHING in this function."""
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
    iter_14: VERBATIM iter_13 (the aim discipline held all run: tilting outside the window fell from
    0.208 to 0.12-0.16 and never came back).

      d_hat    : horizontal unit vector src cup origin -> receiver mouth. Independent of the cup's
                 attitude, so it is defined for an upright cup and never flips while rotating.
      theta    : SIGNED tilt toward the receiver = atan2(up . d_hat, up_z). A purely sideways tilt
                 gives 0, tilting away gives a negative angle.
      lip      : the rim point the beads leave from = mouth centre + r*(cos(theta)*d_hat
                 - sin(theta)*z). Drops as the cup turns toward the receiver, RISES when it turns away.
      lip_drop : how far that lip sits BELOW where it would be with the same cup origin and an upright
                 cup: r*sin(theta) + mouth_z*(1 - cos(theta)). 0.067 m at 75 deg, 0.082 at 90.
                 Measured budget diag_lift_budget 0.300-0.312 all run, i.e. the term is live."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]          # (N,2)
    sep = v.norm(dim=-1)                                               # (N,)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / sep_min, max=1.0)   # the azimuth is meaningless when the cups are coaxial

    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])                     # signed, negative = away
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    # what is tilted but NOT toward the receiver: 0 for a perfect pour attitude (diag_mis_deg 1.8 mean)
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
    # as in iter_06..13 so carry_src and align_mouths keep their measured scale (align 2.35)
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
    # Ladder: reach < grasp < lift(band) per arm, then converge < carry < align < tilt(SIGNED) < pour,
    # and NEW in iter_14: < full transfer < finish (empty source, both cups upright and still).
    #
    # iter_13 RESULT. It stopped the monotone collapse of iter_10-12 (11 rising blocks after the dip,
    # peak transfer efficiency 0.759) and the aim discipline held to the end (tilting outside the
    # permission window 0.208 -> 0.12-0.16). But the 50-epoch mean never passed 0.482 for 600 epochs.
    # Deterministic replay of BOTH ep550 and ep1250 empties the source (in_source 0.000) and delivers
    # 0.906, so the policy already knows the full pour; the learning-curve in_source 0.48 is exploration
    # noise. What is missing is a REASON to go past half: above in_target 0.5 the only growing income
    # was transferred (+2.0 total from 0.5 to 1.0), against -12 per spilled bead.
    #
    # iter_14 changes EXACTLY four things and keeps everything else verbatim:
    #   1. clean_f follows the env: SPILL_ZERO 0.40 -> 0.25 (success_spill_max is now 0.25). The tilt
    #      ladder uses clean_pour = 0.10 + 0.90*clean_f so a spoiled episode keeps a tenth of the pour
    #      income instead of going dark; success / transferred keep the hard, env-matched factor.
    #   2. transfer_tail (new) + W_TRANSFER 4.0 -> 5.0: a ramp that is 0 below the measured operating
    #      point (in_target 0.386 mean) and full at the demonstrated 0.906, steepening toward the end.
    #   3. src_emptied (new): the same ramp on 1 - bead_in_source_frac, 0 at the measured in_source
    #      0.483 and full at the demonstrated 0.000. Gated on clean_f and on beads actually delivered,
    #      so emptying the cup onto the table pays nothing.
    #   4. settle_finish (new): income only in the END state - near-full transfer AND near-empty source,
    #      both cups still grasped, both upright inside UPRIGHT_TOL, both still. This is what the env's
    #      new 20-step hold asks for. It cannot become a parking reward because it is ~0 (0.052) at the
    #      current operating point and only opens after the beads have been delivered.
    # KEEP LIST (user note 4.4), all verbatim iter_13: approach/permission split (approach_lip, perm in
    # cup_radius multiples), aim discipline, src_speed / src_spin / rot_pour / rcv_spin, flow_excess,
    # overtilt, spill 240, per-arm palm_sat / palm_bang / act_flip / carry_rate, grasp / lift / converge
    # / carry / align. No new ABSOLUTE distance constant is introduced anywhere.
    W_ALIVE = 1.0
    W_REACH_SRC, W_REACH_RCV = 1.0, 1.5    # receiver reach unchanged (reach_rcv 0.81 measured)
    W_AWAY_RCV = 0.8
    W_CLOSE_SRC, W_CLOSE_RCV = 0.3, 0.6
    W_GRASP = 2.0                          # grasp formulation and weight unchanged (best-ever grasp rates)
    # --- source lift: band + over-lift ramp. KEEP LIST (diag_lift_budget 0.300 measured) -------------
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
    # --- receiver lift band - KEEP LIST (rcv lift 0.088 m, lift_rcv 1.21) ---------------------------
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
    W_SUCCESS = 20.0        # env thresholds MOVED to 0.80 fill / 0.25 spill / 20-step hold; the weight
                            # is unchanged, so the same success income now costs a real finish
    # ★ iter_14 (1): the spill limit of the SUCCESS TEST is now 0.25, so the income is worth nothing
    # there, exactly as SPILL_ZERO 0.40 mirrored the old limit. At the measured spill 0.133 clean_f
    # goes 0.667 -> 0.468; at the deterministic 0.09 it goes 0.775 -> 0.64.
    SPILL_ZERO = 0.25
    # the tilt ladder rides on a FLOORED version: an episode that has already blown the budget must
    # still learn where the remaining beads go, at a tenth of the clean income (0.10 x W, i.e. 1.0 of
    # pour_tilt's 10.0) instead of going completely dark and parking on align/converge.
    CLEAN_POUR_FLOOR = 0.10
    # ★ iter_14 (2) full-transfer income. Constants are the MEASURED operating point and the MEASURED
    # deterministic ceiling, nothing else:
    #   TAIL_LO 0.35 - just below bead/in_target 0.386 mean (range 0.376-0.471) so the slope is alive
    #                  where the policy lives today instead of starting flat above it
    #   TAIL_HI 0.90 - the deterministic replay value 0.906 (ep550 AND ep1250), i.e. proven reachable
    W_TRANSFER = 5.0        # 4.0 -> 5.0: keeps the DEMONSTRATED state's income under the stricter
                            # clean_f (4.0x0.906x0.775 = 2.81 -> 5.0x0.906x0.64 = 2.90) while the
                            # half-poured state drops (1.17 -> 0.99)
    TAIL_LO, TAIL_HI = 0.35, 0.90
    W_TAIL = 4.0            # 0 at in_target 0.35, +2.6/step at 0.906 after clean_f: half the measured
                            # success income (5.34), so finishing pays well but never outbids success
    # ★ iter_14 (3) emptying income. EMPTY_LO 0.45 = just below the measured emptied 0.517
    # (in_source 0.483 at the end of the run; 0.610 at the peak block), full at 1.0 = the deterministic
    # in_source 0.000. Beads that left the cup onto the TABLE must not pay: x clean_f x fill_ok.
    EMPTY_LO = 0.45
    W_EMPTY = 3.0           # +1.9/step at an empty source after clean_f
    # ★ iter_14 (4) finish stabilisation. finish_q is the MEAN of the two ramps, not their min: a
    # min-based term is flat whenever the two are tied and gives no gradient on either.
    W_SETTLE = 3.0          # up to +3/step ONLY in the finished pose (both cups held, upright inside
                            # UPRIGHT_TOL and still). 0.052 -> ~0.1/step at today's operating point, so
                            # it cannot be farmed before the beads are delivered.
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
    W_RATE_PALM_CARRY = 0.03
    BANG_DEAD = 0.1
    W_BANG_SRC, W_BANG_RCV = 1.5, 1.5
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
    W_PALM_SAT_SRC = 3.0    # measured -1.05/step at diag_palm_sat 0.454: still live, keep
    W_PALM_SAT_RCV = 1.5    # measured -0.40/step at diag_palm_sat_rcv 0.356
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35      # rad (20 deg) - the env's own receiver-upright limit; iter_14 reuses it
                            # for the SOURCE in the finish pose, so no new attitude constant appears

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

    # ---- spout gates: how close the pouring lip is to the receiver mouth (VERBATIM iter_13) --------
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
    lift_gate = LIFT_FLOOR + (1.0 - LIFT_FLOOR) * rcv_prog     # measured rcv_progress 0.745

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

    # ---- receiver lift band (unchanged - measured 0.088 m, inside the 0.10 m band) -----------------
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
    held_rcv_soft = 0.4 + 0.6 * rcv["held"]            # the grasp flag flickers (rcv_grasped 0.76 mean)
    converge = W_CONVERGE * carry_soft_src * held_rcv_soft * rcv_disc * not_nested * conv_shape

    # carry / align keep their iter_06 shape and weights on the DIRECTED spout (align 2.35 measured)
    carry_w = up_w_src * rcv["held"] * pose_ok_rcv * rcv_disc * not_nested
    carry = carry_w * (0.5 * (1.0 - torch.tanh(CARRY_K * d_align))
                       + 0.5 * torch.clamp(1.0 - d_align / CARRY_RANGE, 0.0, 1.0))
    both_up = up_w_src * up_w_rcv * rcv_disc * not_nested
    align = both_up * (0.5 * (1.0 - torch.tanh(ALIGN_K * d_align))
                       + 0.5 * torch.exp(-(d_align / ALIGN_SIGMA) ** 2))

    # ---- TILT: three segments of the SIGNED angle, each with a stricter SPOUT requirement -----------
    beads_left = torch.clamp(ctx.bead_in_source_frac + ctx.bead_in_target_frac, 0.0, 1.0)
    tilt_hold = carry_soft_src * beads_left * not_nested * geo["sep_ok"] * dir_soft
    # ★ iter_14: SPILL_ZERO is the env's NEW spill limit (0.25). clean_f is the hard, env-matched factor
    # used by success / transferred; clean_pour keeps a 0.10 floor for the tilt ladder so a spoiled
    # episode still learns where the remaining beads should go instead of abandoning the pour.
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

    # control gate WITHOUT adr_progress: both cups grasped and >= 5.5 cm up (measured 0.651)
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

    # ---- aiming at the moment of release (VERBATIM iter_13) ----------------------------------------
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

    # ================================================== ★ iter_14: FULL TRANSFER AND FINISH =========
    # The flat region above in_target 0.5 is what stopped iter_13: the only growing income there was
    # transferred (+2.0 total from half to full) against -12 per spilled bead. Both ramps below are 0
    # at the MEASURED stochastic operating point and 1 at the MEASURED deterministic replay, and both
    # get steeper toward the end, so the last beads are worth three times the first ones.
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    emptied = torch.clamp(1.0 - fill, 0.0, 1.0)
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)      # 0.035 at 0.386 (today), 0.744 at 0.80, 1 at 0.90
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)        # 0.068 at in_source 0.483, 1 at in_source 0
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
        # level term, now steeper and stricter: 5.0 x in_target x clean_f(0.25)
        "transferred": W_TRANSFER * in_t * not_nested * upright_rcv * clean_f,
        # ★ iter_14: the missing slope above half. 0 at the measured 0.386, full at the demonstrated 0.906
        "transfer_tail": W_TAIL * fill_tail * not_nested * upright_rcv * clean_f,
        # ★ iter_14: emptying the SOURCE pays too, so "half and stop" is strictly worse than finishing
        "src_emptied": W_EMPTY * empty_q * fill_ok * not_nested * clean_f,
        # ★ iter_14: the finish the env's 20-step hold now asks for (both cups held, upright, still)
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
        "act_flip": -W_FLIP_PALM * (g_src * flip_palm_src + g_rcv * flip_palm_rcv),
        "hand_flip": -W_FLIP_HAND * ctrl_gate * flip_hand,
        "carry_rate": -(W_RATE_PALM_CARRY * (g_src * rate_palm_src + g_rcv * rate_palm_rcv)
                        + W_RATE_HAND_CARRY * ctrl_gate * rate_hand),
        "palm_bang_src": -W_BANG_SRC * g_src * bang_src,
        "palm_bang_rcv": -W_BANG_RCV * g_rcv * bang_rcv,
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
    # ★ iter_14 diagnostics: one per new mechanism, so the next round can tell which moved
    comps["diag_fill_tail"] = fill_tail          # how far past the old plateau the transfer got
    comps["diag_empty_q"] = empty_q              # how empty the source ends up
    comps["diag_finish_q"] = finish_q            # the combined finish state (mean ramp)
    comps["diag_settle_still"] = settle_still    # were both cups actually stopped at the end
    comps["diag_emptied"] = emptied              # raw 1 - in_source, for comparison across rounds
    return reward, comps
```
