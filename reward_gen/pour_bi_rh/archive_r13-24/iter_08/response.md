# iter_08 — the tilt stage is rebuilt on a SIGNED toward-receiver angle and on the real pouring lip

## 0. One blocking fact first: this track's `RewardContext` has no direction fields

The round-8 notes (§3, §8) ask for `ctx.src_tilt_toward_rcv`, `ctx.pour_dir_xy`, `ctx.premature_tilt`,
`ctx.src_pour_lip_pos`, `ctx.premature_tilt_limit`. **Those fields do not exist in the contract this
track runs against**, so a reward that names them fails ingest before it ever reaches training:

- `scripts/reward_gen/t2r_rh.py:33-34` imports `prompts`/`validator` from
  `openarm.agnostic.tasks.pour_fabric_mimic.t2r` — the **track-local** t2r, not `modules/t2r`.
- `pour_fabric_mimic/t2r/context.py` has 48 tensor fields and stops at `success` / `episode_progress`.
  It contains none of `bead_fill_level`, `premature_tilt`, `src_pour_lip_pos`, `src_tilt_toward_rcv`,
  `premature_tilt_limit`, `cup_hit`, `pour_dir_xy`, `rcv_side_margin`, `src_wrap_count`,
  `rcv_wrap_count`. That is also why the prompt's own class stub above (the rendered
  `context_stub_source()`) does not list them.
- The split is deliberate, not drift:
  `pour_fabric_mimic/tests/test_mimic_contract.py::test_t2r_contract_is_track_local_not_shared`
  asserts `"modules.t2r" not in _ENV` and `for gone in ("bead_fill_level", "premature_tilt"): assert
  gone not in ctx_src` — the shared module once added required fields and killed this env at startup
  (commit `3dd15927`).
- `pour_fabric_mimic/pour_fabric_env.py::_build_context` never computes a pouring lip, a pour
  direction or a premature latch, and its `_success_now` (L625-628) is only
  `in_target ≥ 0.50 & spill ≤ 0.40 & cups_xy < 0.20 & ~nested`. There is **no** premature-tilt void
  and **no** outward-pour void in this track. `validator.static_check` would reject
  `ctx.src_tilt_toward_rcv` with "RewardContext 에 없는 필드".

So notes §7 ("[가설] premature_tilt 래치가 성공을 무효화한다") is **false for this track**: the latch
does not exist here, which is consistent with the env logging no `task/premature_tilt_rate` key.
`episode_success` is exactly 0 for the simpler reason below.

**What I did instead.** Every quantity the notes ask for is *computable* from fields this track does
have, with the env's own formulas (`pour_fabric/pour_rules.py`), so iter_08 implements all four
requirements without naming a missing field:

| notes §8 asks for | iter_08 uses instead | identical to |
|---|---|---|
| `ctx.src_tilt_toward_rcv` | `theta = atan2(src_cup_up[:, :2]·d̂, src_cup_up[:, 2])` | `pour_rules.pour_lip` L63-64, exactly |
| `ctx.pour_dir_xy` | `d̂ = normalize(rcv_cup_mouth_pos[:, :2] − src_cup_pos[:, :2])` | `pour_rules.pour_dir_update` L46-48 |
| `ctx.src_pour_lip_pos` | `lip = src_cup_mouth_pos + r·cosθ·d̂ − r·sinθ·ẑ`, `r = ctx.cup_radius` | `pour_rules.pour_lip` L65-68 |
| `ctx.premature_tilt` (latch) | per-step `early_tilt` cost on `src_cup_tilt` above the aim-dependent corridor | `pour_rules.premature_tilt_now` L84, minus the latch |
| `ctx.premature_tilt_limit` | `release − 20°`, `release = 75° + 25°·(1 − bead_in_source_frac)` | `premature_tilt_limit_rad` L27-38, same shape |

`r` is not a guessed number: `ctx.cup_radius` is a real field of this contract and resolves to
`0.0432 × cup_scale 0.65 = 0.0281 m` (`pour_fabric_env_cfg.CUP_GEOM_UNIT` + `resolve_cfg`), and
`ctx.cup_mouth_z = 0.0539 m` the same way.

If the operator wants the env-side fields (and the real latch / void rules), that is an env change in
`pour_fabric_mimic`, not a reward change, and it should be a separate round.

## 1. What the task is and which stages the robot must go through

1. **Reach** each cup with its own hand, slowly (the cups topple at ~0.3 N of one-sided contact).
2. **Pinch** thumb against the opposite fingertips — fingertip grasp, the hand cannot wrap a 57 mm body.
3. **Lift** both cups clear of the table; the source higher than the receiver.
4. **Converge**: bring the two cups to within ~6 cm in xy with the source ~11 cm higher, so the
   source's pouring rim hangs over the receiver's mouth while the two *origins* stay more than
   `cups_nested_dist = 9 cm` apart in 3D (the Δz is what buys that margin).
5. **Aim**: rotate the source cup so its lowest rim point — the point beads actually leave from —
   sits over the receiver's mouth centre, above the rim.
6. **Pour**: keep turning past the release angle so the beads fall through the air into the receiver,
   receiver upright, without the cups touching.
7. **Hold** the pose until the env's 10-step success streak completes.

## 2. Why iter_07 stopped at 54.8°: the tilt had no direction

Everything before the pour is solved and stays untouched (notes §9): `task/src_grasped` 0.828,
`task/rcv_grasped` 0.762, `src_cup_lift` 0.178 m inside its band, `rcv_cup_lift` 0.082 m,
`cups_center_dist` 0.875 → 0.189 m, `reward/align_mouths` 2.62. The whole ladder up to `align` is
carried over verbatim.

The tilt stage is rebuilt, because every one of its inputs was **direction-free**:

- `ctx.src_cup_tilt` is `acos(up_z)` (`pour_fabric_env.py::_build_context::_tilt`). Tilting the mouth
  toward the receiver, sideways, or backwards all produce the same number, so `tilt_lead` 1.565 and
  `tilt_rise` 1.732 were paid in full for a rotation that can never pour. The user's ep1950 observation
  (mouth facing world +x, i.e. away) is exactly the configuration that maximised iter_07's tilt income.
- iter_07's lip, `src_cup_mouth_pos + r·(up·up_z − ẑ)`, is the geometrically lowest rim point. It is
  defined for *any* rotation, so `d_align`, `aim_shape`, `carry_src` and `align_mouths` were all blind
  to the azimuth too — the "aim" the policy learned (`aim_gate` 0.739, `aim_dist` 0.17 m) was an aim of
  the cup body, not of the spout.
- Hence the deterministic replay: 65.4° of tilt, lip 0.128 m from the mouth, `in_source` **1.000**.
  Nothing came out because (a) 65° is below the ~70° first-bead angle of a full cup and (b) the
  component of that 65° pointing at the receiver was ≈ 0. The 3.2 % `bead/spill` is the small fraction
  that did leave — on the wrong side.

## 3. Changes vs iter_07

**Kept verbatim** (keep list, notes §9): `_band`, `_close_in`, `_grip_geometry`, `_hand_cup_terms`
(reach / grasp / close / lift-state and every contact penalty), `lift_src` + band + `over_lift_src`,
`lift_rcv`, `hold_rcv`, `rcv_progress` → `lift_gate`, `rcv_home`, `rcv_still`, `converge`, and the
weights of `carry_src` (3.0) and `align_mouths` (5.0).

1. **`_pour_geometry` now returns the real spout.** `d̂`, `theta` (signed, receiver-ward, negative when
   the cup leans away), `lip_xy`, `lip_dz` and `d_align` are built with the env's own lip formula and
   `r = ctx.cup_radius`. `carry_src` and `align_mouths` keep their shape and weights but are now
   measured to *this* lip, so they stop paying for a cup that is aimed backwards. `d_align` targets
   4 cm above the receiver rim (was 5 cm) with the same half-weighted z error.
2. **The three tilt segments run on `theta_eff = min(clamp(theta, 0), src_cup_tilt)`**, never on
   `src_cup_tilt`. A cup tilted 55° sideways or backwards now scores `theta_eff = 0` and earns nothing
   in any of them.
   - `tilt_aim` (W 3.0, 0 → `theta_pre` = release − 20° ≈ 55° full): distance-free income, spill-free by
     geometry, gated on carried + beads inside + convergence — this replaces `tilt_lead` + `tilt_rise`
     and has nearly the same per-degree density (0.055/deg vs 0.044 and 0.125), so a policy that turns
     the *right* way loses nothing relative to iter_07.
   - `tilt_near` (W 4.0, `theta_pre` → release ≈ 75°): additionally needs the lip inside the aim
     radius (`approach_gate`, full at 8 cm — the env rule's own `lip_max_m` 8.1 cm — and 0 at 16 cm)
     and above the receiver rim.
   - `pour_tilt` (W 7.0, release → release + 30°): needs `safe` (lip within 3 cm of the mouth centre,
     0 at 6 cm ≈ the rim), the lip above the rim, receiver upright, both carried, not nested. This is
     the only place where a large tilt is worth more than the carry plateau.
   - `release = 75° + 25°·(1 − bead_in_source_frac)`, same shape as the env's `premature_tilt_limit_rad`:
     the measured first-bead angle of a full cup is ~70°, and the angle rises as the cup empties, so a
     fixed target would go flat exactly when the last beads need a steeper pour.
3. **`pour_pose` (W 3.0), new.** A plateau around the release angle with the lip over the mouth
   (`aim_xy · z_ok · pour_band`), so holding the pouring pose for the 10-step success streak beats
   oscillating through it.
4. **The attitude corridor is now an aim corridor**, and each of its three steps matches the ladder
   segment that pays there, so no term ever pays for an angle another term charges: `allowed` = 0.15 rad
   on the table → `theta_pre` (55°) once carried → release (75°) as the lip enters the 8 cm aim radius
   → release + 30° as `safe` rises. The excess is measured on the **magnitude**
   `src_cup_tilt` (notes §8.1: magnitude keeps the penalty job), it multiplies `grasp_src` and half of
   `lift_src` exactly as before, and it is charged once more as the new `early_tilt` component
   (W 0.8) — the stateless stand-in for the `premature_tilt` latch: over-tilting before the spout is
   over the receiver costs every step it lasts instead of once per episode.
5. **`mis_tilt` (W 1.2), new.** `clamp(src_cup_tilt − theta_eff, min=0)` beyond a 10° free band, scaled
   by the carry state. This is the term that has a live gradient in the exact state the run is stuck
   in: it falls continuously as the wrist rotates the mouth from +x toward the receiver, which
   `theta_eff` alone cannot provide while `theta < 0` (clamped flat).
6. **`pour_dir_pen` (W 1.0), new.** `d̂ₓ` is the body-outward component of the pouring direction
   (`+x` = away from the torso). Penalty ramps in from 0.10 and is full by 0.35, and a soft factor
   `0.3 + 0.7·dir_ok` (1 at `d̂ₓ ≤ 0`, 0 at 0.28) multiplies the tilt income — zero well before the
   0.3 line that voids success in the sibling env. Soft rather than a hard gate, because this track's
   env does not void on it and the cups are separated mostly in y (`d̂ₓ ≈ 0` at spawn).
7. **`near_nested` (W 1.0), new.** The new aim target brings the cups to ~6 cm in xy; the 3D origin
   distance is kept off the 9 cm nesting line by a ramp that starts at 11.5 cm. At the intended pose
   (xy 0.059 m, Δz 0.11 m → 0.125 m) it is exactly 0, so it costs nothing when the pour is done right.
8. **`transferred` (W 4.0), new, level term.** `pour_delta` (150 × Δ, 7.5 per bead) still dominates by
   two orders of magnitude, but success needs `in_target ≥ 0.5` held for 10 steps, so keeping beads in
   must be worth something.
9. **Diagnostics after the sum, not part of the reward** (so the next round does not have to guess
   again, given the env logs only the direction-free `task/src_tilt_deg`): `diag_theta_deg` (the signed
   toward-receiver angle in degrees), `diag_mis_deg` (the misdirected part), `diag_lip_xy` and
   `aim_gate`. If `diag_theta_deg` rises while `task/src_tilt_deg` stays near it, the rotation is
   finally going the right way; if they diverge, the azimuth is still wrong.

## 4. The function evaluated on hand-built contexts (not estimated)

Both cups grasped with the measured 4 N thumb / 4 N summed fingers, receiver upright and lifted
0.08 m, full cup, totals per step:

| source cup state | tilt 40° | 55° | 65° | 75° | 85° | 100° |
|---|---|---|---|---|---|---|
| **aimed**, spout over the mouth (cups xy 0.06, Δz 0.11) | — | **28.39** | 30.45 | 32.80 | 36.50 | **40.73** |
| **aimed**, cups xy 0.12 (lip 6 cm out) | — | 27.70 | 29.72 | 31.69 | 30.38 | — |
| **aimed**, at iter_07's operating point (cups xy 0.19, lift 0.178) | 25.16 | 26.06 | 26.49 | 25.27 | — | — |
| **sideways** (azimuth 90°), same operating point | 21.25 | 20.83 | 19.49 | 17.77 | — | — |

Three things this shows, and they are the whole round:

1. **The azimuth is now worth more than the angle.** At the exact state the run is stuck in
   (lip 0.13 m out, 55° of tilt), rotating the mouth from sideways to the receiver is **+5.2 per step**
   and needs no translation at all — `mis_tilt` releases 1.15 and `tilt_aim` pays 2.94. That gradient
   is available at the first epoch from the loaded checkpoint.
2. **With the spout over the mouth the ladder is monotone from 0° to 100°** (24.88 → 40.73), so there
   is no angle at which turning further is a loss — the failure mode of iter_06 and iter_07.
3. **Tilting past `theta_pre` while still far is mildly downhill** (26.06 → 25.27 at 0.13 m), which is
   the intended instruction "close the last 7 cm first", and it is not iter_07's trap: the gate is on
   the lip distance, which `align_mouths` / `carry_src` / `converge` actively pull down and which the
   run has already driven from 0.875 to 0.19 m.

Expected metric moves: `reward/tilt_*` drops sharply at the first epochs (the current 54.8° is
misdirected and no longer paid), `diag_mis_deg` falls, `diag_theta_deg` rises from ≈ 0, `diag_lip_xy`
falls from ≈ 0.13 m toward 0.03 m, `cups_center_dist` falls from 0.19 to ≈ 0.12 m with
`nested_rate` flat, and `bead/in_target` leaves 0 only after `diag_theta_deg` passes ~70°.

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
