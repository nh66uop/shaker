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


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bimanual pour: grasp both cups, lift, aim the pouring lip, tilt until the
    requested number of beads has left the source cup and landed in the receiver.

    Design rules used (all lengths are multiples of ctx.cup_radius, all angles are
    fractions of ctx.premature_tilt_limit -> no asset dimension is hardcoded):
      - headline income = beads arriving (pour_delta, 30 per bead)
      - rotation drive (<= 8 / step) is the only large per-step state income and it is
        switched off as soon as `need` reaches zero
      - all posture/approach income summed is < 5 / step, i.e. below one poured bead
      - the premature-tilt charge only lives at the wall (>= 0.95 * limit) and only
        while the lip is outside 2 * cup_radius, the same radius the env latch uses
    """
    eps = 1e-6
    zeros = torch.zeros_like(ctx.src_cup_tilt)

    # ---- shared scales ---------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # env latch radius for the lip (xy)
    lim = torch.clamp(ctx.premature_tilt_limit, min=1e-3)   # (N,) rad, from env
    n_fing = max(int(ctx.src_tips_pos.shape[1]), 1)

    src_g = ctx.src_grasped.float()
    rcv_g = ctx.rcv_grasped.float()
    both_g = src_g * rcv_g

    # Pre-lift / posture income must not be earnable by tilting before lifting (R6b):
    # the env only records the grasp reference while the cup is well below its limit,
    # so fade that income out with the cup's own tilt, using 0.3 * limit as the scale.
    low_src = 1.0 - torch.tanh(torch.clamp(ctx.src_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))
    low_rcv = 1.0 - torch.tanh(torch.clamp(ctx.rcv_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))

    # ---- 1-2. approach (weight 0.4 each; dies out once the cup is grasped) -----
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_src = 0.4 * (1.0 - src_g) * low_src * (1.0 - torch.tanh(d_src / two_r))
    approach_rcv = 0.4 * (1.0 - rcv_g) * low_rcv * (1.0 - torch.tanh(d_rcv / two_r))

    # ---- 3-4. enveloping grasp (weight 0.5 each) -------------------------------
    # wrap counts are normalised by the number of fingers of THIS hand, not by 5.
    wrap_src = _sat(ctx.src_wrap_count / float(n_fing))
    wrap_rcv = _sat(ctx.rcv_wrap_count / float(n_fing))
    grasp_src = 0.5 * src_g * (0.4 + 0.6 * wrap_src)
    grasp_rcv = 0.5 * rcv_g * (0.4 + 0.6 * wrap_rcv)

    # ---- 5. lift both cups (weight 0.5, saturates at a 2*cup_radius lift) ------
    h_src = _sat((ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]) / two_r)
    h_rcv = _sat((ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]) / two_r)
    lift_both = 0.5 * both_g * 0.5 * (h_src * low_src + h_rcv * low_rcv)
    lifted = torch.minimum(h_src, h_rcv)

    # ---- 6. aim the LIP (not the rim centre) over the receiver mouth (0.8) -----
    lip_d = _dist_xy(ctx.src_pour_lip_pos, ctx.rcv_cup_mouth_pos)
    dz_lip = ctx.src_pour_lip_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
    # a little above the mouth: target 0.25*cup_radius, tolerance 0.5*cup_radius
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    aim_lip = 0.8 * both_g * lifted * (0.6 * (1.0 - torch.tanh(lip_d / two_r)) + 0.4 * h_score)

    # ---- 7. keep the grasp recorded at lift-off (iter_24 guard, weight 1.0) ----
    # Similarity to the reference relation only; never an absolute force target.
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
    # `want` tapers inside the last bead (the env tolerance is +-1 bead) and is 0
    # once spill made the request unreachable -> the finish branch opens.
    want = _sat(need_beads)

    # ---- 8. rotation drive: the term that has to cross the ~90 deg wall (8.0) --
    # Gate 1: lip near the receiver mouth, using the SAME radius as the env latch,
    #         with a soft 2*cup_radius tail so that 9-14 cm still pays clearly.
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    # Gate 2: tilt must be TOWARD the receiver (sideways / away pays nothing).
    # Progress is measured against 1.4 * premature_tilt_limit (a multiple of a ctx
    # angle, not an absolute angle): that reaches past the bead release tilt for
    # every fill, and caps the income so the policy cannot profit by over-rotating.
    rot_prog = _sat(torch.clamp(ctx.src_tilt_toward_rcv, min=0.0) / (1.4 * lim))
    # Gate 3: only after the env has recorded the lift grasp reference (blocks the
    # iter_27 "avoid the reference, collect posture income" path) and while needed.
    pour_rot = 8.0 * src_g * ctx.src_grasp_ref_valid.float() * want * lip_gate * rot_prog

    # ---- 9. headline income: beads that actually arrive (30 per bead) ----------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 10. pouring past the request (15 per bead beyond the +-1 tolerance) ---
    over_beads = _sat(torch.clamp(ctx.bead_in_target_frac - ctx.pour_target_frac, min=0.0) * active - 1.0,
                      0.0, 5.0)
    pour_overshoot = -15.0 * over_beads

    # ---- 11. spill (20 per lost bead, on the increment so it is charged once) --
    spill = -20.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 12. finish branch: small, and per step below the pour income (R1) -----
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(ctx.src_cup_tilt / (0.3 * lim)))

    # ---- 13. terminal success bonus (env decides success, we only pay it) ------
    success_bonus = 20.0 * ctx.success.float()

    # ---- 14. premature tilt charge AT the wall only (R3) ----------------------
    # charge ramps in above 0.95 * limit over the last 0.05 * limit, and only while
    # the lip is still outside 2*cup_radius; inside that radius there is NO charge.
    pre_x = torch.clamp(ctx.src_cup_tilt - 0.95 * lim, min=0.0) / (0.05 * lim)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature_charge = -12.0 * src_g * pre_ramp * out_gate

    # ---- 15. the latch itself voids success: small steady reminder ------------
    premature_latched = -0.4 * ctx.premature_tilt.float()

    # ---- 16-18. real-robot safety: no cup-cup contact, no foreign contact -----
    cup_cup = -(2.0 * torch.tanh(ctx.cup_cup_force / 5.0) + 0.5 * ctx.cup_hit.float())
    hand_foreign = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                           + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))
    nested = -3.0 * ctx.cups_nested.float()

    # ---- 19. pouring direction must not point away from the body -------------
    # dimensionless unit-vector component (env voids at 0.3); ramp starts at half of it
    dir_ramp = _sat((ctx.pour_dir_xy[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 20. receiver cup: upright and on its own side of the centre line -----
    side_lo = 0.75 * r_cup                      # env voids below ~0.75 * cup_radius
    side_bad = _sat((side_lo - ctx.rcv_side_margin) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * lim))) - 1.5 * side_bad

    # ---- 21. always-on action rate (iter_18 level, small) --------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)

    # ---- 22. exploration-suppressing charges, ramped by ADR (R7) -------------
    adr = float(ctx.adr_progress)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    qd2 = 0.5 * ((ctx.src_arm_qd ** 2).mean(dim=-1) + (ctx.rcv_arm_qd ** 2).mean(dim=-1))
    action_reg_adr = -adr * (0.5 * flip + 0.3 * sat + 0.4 * torch.tanh(qd2 / 25.0))

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_both": lift_both,
        "aim_lip": aim_lip,
        "grip_keep": grip_keep,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "pour_overshoot": pour_overshoot,
        "spill": spill,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "premature_charge": premature_charge,
        "premature_latched": premature_latched,
        "cup_cup": cup_cup,
        "hand_foreign": hand_foreign,
        "nested": nested,
        "pour_dir": pour_dir,
        "rcv_keep": rcv_keep,
        "action_rate": action_rate,
        "action_reg_adr": action_reg_adr,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
