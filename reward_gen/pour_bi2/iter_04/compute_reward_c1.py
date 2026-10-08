import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.sqrt((d * d).sum(dim=-1) + 1e-12)


def _dist_xy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a[:, :2] - b[:, :2]
    return torch.sqrt((d * d).sum(dim=-1) + 1e-12)


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bimanual pour: grasp both cups, lift, bring the cups together, aim the pouring
    LIP over the receiver mouth, tilt until the requested number of beads (+-1) has left
    the source cup and landed in the receiver, then stand the source cup back up.

    Change of this iteration (exactly the one the round asks for, section K):
      K1  Rotation income stops once the remaining request is within the env tolerance
          (one bead): want_rot = sat(need_beads - 1), need = min(target - poured, in source).
      K2  Real finish pays big: poured within +-1 bead of ctx.pour_target_beads (and at
          least one bead actually arrived) -> standing the source cup up while both hands
          hold pays up to 12 per step, above the maximum rotation income (8).
      K3  Dumping never pays: when spill made the request unreachable, rotation stops and
          the finish pays at most the old 1.5 per step.
      Everything else is identical to iter_03 (converge, aim floor, spill 40 per bead,
      premature charge at the wall, grip-keep, ADR-gated tremor/flip).  22 components (R9).

    Design rules: every length is a multiple of ctx.cup_radius, every angle a fraction of
    ctx.premature_tilt_limit -> no asset dimension or absolute angle is hardcoded (R8).
    """
    eps = 1e-6
    zeros = torch.zeros_like(ctx.src_cup_tilt)

    # ---- shared scales ---------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # env latch radius for the lip (xy)
    wide = 10.0 * r_cup                      # WIDE kernel: covers the visited 0.2-0.45 m band
    lim = torch.clamp(ctx.premature_tilt_limit, min=1e-3)   # (N,) rad, from env
    n_fing = max(int(ctx.src_tips_pos.shape[1]), 1)
    adr = float(ctx.adr_progress)            # 0 until the success rate passes the ADR trigger

    src_g = ctx.src_grasped.float()
    rcv_g = ctx.rcv_grasped.float()
    both_g = src_g * rcv_g

    # Pre-lift / posture income must not be earnable by tilting before lifting (R6b).
    low_src = 1.0 - torch.tanh(torch.clamp(ctx.src_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))
    low_rcv = 1.0 - torch.tanh(torch.clamp(ctx.rcv_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))

    # ---- 1. approach both hands (0.4 each; dies out once that cup is grasped) ---
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_both = (0.4 * (1.0 - src_g) * low_src * (1.0 - torch.tanh(d_src / two_r))
                     + 0.4 * (1.0 - rcv_g) * low_rcv * (1.0 - torch.tanh(d_rcv / two_r)))

    # ---- 2-3. enveloping grasp (weight 0.5 each) -------------------------------
    wrap_src = _sat(ctx.src_wrap_count / float(n_fing))
    wrap_rcv = _sat(ctx.rcv_wrap_count / float(n_fing))
    grasp_src = 0.5 * src_g * (0.4 + 0.6 * wrap_src)
    grasp_rcv = 0.5 * rcv_g * (0.4 + 0.6 * wrap_rcv)

    # ---- 4. lift both cups (weight 0.5, saturates at a 2*cup_radius lift) ------
    h_src = _sat((ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]) / two_r)
    h_rcv = _sat((ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]) / two_r)
    lift_both = 0.5 * both_g * 0.5 * (h_src * low_src + h_rcv * low_rcv)
    lifted = torch.minimum(h_src, h_rcv)

    lip_d = _dist_xy(ctx.src_pour_lip_pos, ctx.rcv_cup_mouth_pos)
    dz_lip = ctx.src_pour_lip_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]

    # ---- 5. bring the two cups together (weight 1.2, WIDE kernel, unchanged) ---
    converge_cups = 1.2 * both_g * lifted * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- 6. aim the LIP (not the rim centre) over the receiver mouth (0.8) -----
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * (0.6 * xy_score + 0.4 * h_score)

    # ---- 7. keep the grasp recorded at lift-off (iter_24 guard, weight 1.0) ----
    # Similarity to the reference relation only; never an absolute force target (R6a).
    s_len = 0.5 * r_cup

    def _grip(palm, ref_palm, tips, ref_tips, share, ref_share, tot, ref_tot, valid, gr):
        p_err = _dist(palm, ref_palm)
        t_err = torch.sqrt(((tips - ref_tips) ** 2).sum(dim=-1) + 1e-12).mean(dim=-1)
        s_err = torch.abs(share - ref_share).mean(dim=-1)
        f_err = torch.abs(tot - ref_tot) / torch.clamp(ref_tot, min=0.5)  # 0.5 N force floor
        score = (0.40 * (1.0 - torch.tanh(p_err / s_len))
                 + 0.35 * (1.0 - torch.tanh(t_err / s_len))
                 + 0.15 * (1.0 - torch.tanh(s_err / 0.15))
                 + 0.10 * (1.0 - torch.tanh(f_err)))
        return valid.float() * gr * score

    grip_src = _grip(ctx.src_palm_in_cup, ctx.src_ref_palm_in_cup,
                     ctx.src_tips_in_cup, ctx.src_ref_tips_in_cup,
                     ctx.src_force_share, ctx.src_ref_force_share,
                     ctx.src_force_total, ctx.src_ref_force_total,
                     ctx.src_grasp_ref_valid, src_g)
    grip_rcv = _grip(ctx.rcv_palm_in_cup, ctx.rcv_ref_palm_in_cup,
                     ctx.rcv_tips_in_cup, ctx.rcv_ref_tips_in_cup,
                     ctx.rcv_force_share, ctx.rcv_ref_force_share,
                     ctx.rcv_force_total, ctx.rcv_ref_force_total,
                     ctx.rcv_grasp_ref_valid, rcv_g)
    grip_keep = 1.0 * 0.5 * (grip_src + grip_rcv)

    # ---- bead bookkeeping: need = min(target - poured, still in source) (R5) ---
    active = torch.clamp(ctx.bead_active_count, min=1.0)
    remaining = torch.clamp(ctx.pour_target_frac - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    need_beads = need_frac * active
    # `want` (unchanged) gates the per-bead arrival income.
    want = _sat(need_beads)

    # K1/K3: counts in beads, compared with the env tolerance of +-1 bead.
    target_b = torch.clamp(ctx.pour_target_beads.float(), min=0.0)
    poured_b = torch.clamp(ctx.bead_in_target_frac, min=0.0) * active
    spill_b = torch.clamp(ctx.bead_spill_frac, min=0.0) * active
    # Request unreachable when the permanently spilled beads exceed the slack
    # (active - target) plus the one-bead tolerance. Beads in flight are not counted.
    unreach = _sat(spill_b - (active - target_b + 1.0))
    # K1: rotation income only while more than one requested bead is still missing
    # and the request is still reachable.
    want_rot = _sat(need_beads - 1.0) * (1.0 - unreach)

    # ---- 8. rotation drive (8.0, unchanged except want_rot) --------------------
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    rot_prog = _sat(torch.clamp(ctx.src_tilt_toward_rcv, min=0.0) / (1.4 * lim))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)      # 1 inside 1r
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))  # low drop
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    pour_rot = 8.0 * src_g * ctx.src_grasp_ref_valid.float() * want_rot * lip_gate * aim_factor * rot_prog

    # ---- 9. headline income: beads that actually arrive (30 per bead) ----------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 10. pouring past the request (15 per bead beyond the +-1 tolerance) ---
    over_beads = _sat(torch.clamp(ctx.bead_in_target_frac - ctx.pour_target_frac, min=0.0) * active - 1.0,
                      0.0, 5.0)
    pour_overshoot = -15.0 * over_beads

    # ---- 11. spill: 40 per lost bead (kept) -------------------------------------
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 12. finish branch (K2, K3) --------------------------------------------
    # Real finish: poured within +-1 bead of the request (1 inside, 0 beyond +-2 beads),
    # at least one bead actually arrived, request not voided by spill.
    in_tol = _sat(2.0 - torch.abs(poured_b - target_b))
    real = in_tol * _sat(poured_b) * (1.0 - unreach)
    # Upright score: the sharp kernel (old) pays near upright; the linear ramp over the
    # same 1.4 * limit span as rot_prog gives a gradient from the pouring tilt downward.
    up_sharp = 1.0 - torch.tanh(ctx.src_cup_tilt / (0.3 * lim))
    up_lin = _sat(1.0 - ctx.src_cup_tilt / (1.4 * lim))
    # weight 12 > max pour_rot 8: standing up after a real finish beats holding tilted.
    finish_real = 12.0 * real * (0.5 * up_lin + 0.5 * up_sharp)
    # Request closed without a real finish (spill / overshoot / emptied): small, <= 1.5.
    closed_bad = (1.0 - want_rot) * (1.0 - real)
    finish_small = 1.5 * closed_bad * up_sharp
    finish_upright = both_g * (finish_real + finish_small)

    # ---- 13. terminal success bonus (env decides success, we only pay it) ------
    success_bonus = 20.0 * ctx.success.float()

    # ---- 14. premature tilt: charge AT the wall only, plus the latch reminder --
    pre_x = torch.clamp(ctx.src_cup_tilt - 0.95 * lim, min=0.0) / (0.05 * lim)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate - 0.4 * ctx.premature_tilt.float()

    # ---- 15. real-robot safety: cups must never touch, nesting is never a pour -
    cup_clash = -(2.0 * torch.tanh(ctx.cup_cup_force / 5.0)
                  + 0.5 * ctx.cup_hit.float()
                  + 3.0 * ctx.cups_nested.float())

    # ---- 16. hands touch only their own cup ------------------------------------
    hand_foreign = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                           + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 17. pouring direction must not point away from the body --------------
    dir_ramp = _sat((ctx.pour_dir_xy[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 18. receiver cup: upright and on its own side of the centre line -----
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_side_margin) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * lim))) - 1.5 * side_bad

    # ---- 19. always-on action rate (iter_18 level, small; allowed by R7) ------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)

    # ---- 20. action reversal, behind the ADR gate (R7) -------------------------
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -adr * 0.30 * flip

    # ---- 21. arm tremor, receiver weighted more, ADR-gated (R7) ----------------
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -adr * (0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))

    # ---- 22. palm saturation stays ramped by ADR (R7) -------------------------
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -adr * 0.3 * sat

    components = {
        "approach_both": approach_both,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_both": lift_both,
        "converge_cups": converge_cups,
        "aim_lip": aim_lip,
        "grip_keep": grip_keep,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "pour_overshoot": pour_overshoot,
        "spill": spill,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "premature": premature,
        "cup_clash": cup_clash,
        "hand_foreign": hand_foreign,
        "pour_dir": pour_dir,
        "rcv_keep": rcv_keep,
        "action_rate": action_rate,
        "action_flip": action_flip,
        "arm_tremor": arm_tremor,
        "action_reg_adr": action_reg_adr,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
