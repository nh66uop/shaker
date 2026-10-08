브리프에 가설 id 줄이 없어 prompt 전체를 한 후보로 구현했다. 읽은 파일: /home/user/rl_ws/hdgp/reward_gen/pour_bi2/iter_02/prompt.md (693줄 전체, 2회 분할 읽기). 파일은 쓰지 않았다.

## 최종 코드

```python
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
    LIP over the receiver mouth and tilt until the requested number of beads has left
    the source cup and landed in the receiver.

    Changes of this iteration (exactly the two the round asks for):
      C1  AIMING NOW RAISES THE ROTATION INCOME and never switches it off.
          iter_01 poured with the lip ~3*cup_radius (0.123 m) beside the mouth: the old
          `pour_rot` had no aim factor at all, so tilting without aiming paid the same as
          tilting over the opening.  The aim factor is a BONUS ON A FLOOR
          (FLOOR = 0.3), never a precision gate - iter_26/27 multiplied by a narrow
          gaussian that evaluated to 1e-17 in the visited states and three rounds poured
          nothing.  Teeth are on the OUTCOME side instead: the spill charge per bead is
          raised 20 -> 40 (> the 30 paid for a bead that arrives), so pouring while badly
          aimed loses money while the rotation drive stays alive.
      C2  TREMOR AND ACTION-REVERSAL CHARGES ARE ALIVE FROM STEP 0 (no adr factor),
          small and weighted more on the receiver arm (its qd RMS ran 1.2-1.3x the source
          arm's, and the flip rate sat at 0.49 all run while adr_progress stayed 0.000).
          The palm-saturation charge stays ramped by adr_progress (R7).
      Everything else is unchanged in value: converge_cups, aim_lip, lip_gate, the
      rotation cap, the premature charge at the wall, grip-keep, safety, bookkeeping.
      Only logging was compacted (approach_src+approach_rcv, premature_charge+latched,
      cup_cup+nested merged) to keep the component count at 22 (R9).

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

    src_g = ctx.src_grasped.float()
    rcv_g = ctx.rcv_grasped.float()
    both_g = src_g * rcv_g

    # Pre-lift / posture income must not be earnable by tilting before lifting (R6b):
    # the env only records the grasp reference while the cup is well below its limit,
    # so fade that income out with the cup's own tilt, using 0.3 * limit as the scale.
    low_src = 1.0 - torch.tanh(torch.clamp(ctx.src_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))
    low_rcv = 1.0 - torch.tanh(torch.clamp(ctx.rcv_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))

    # ---- 1. approach both hands (0.4 each; dies out once that cup is grasped) ---
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_both = (0.4 * (1.0 - src_g) * low_src * (1.0 - torch.tanh(d_src / two_r))
                     + 0.4 * (1.0 - rcv_g) * low_rcv * (1.0 - torch.tanh(d_rcv / two_r)))

    # ---- 2-3. enveloping grasp (weight 0.5 each) -------------------------------
    # wrap counts are normalised by the number of fingers of THIS hand, not by 5.
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

    # ---- 5. bring the two cups together (weight 1.2, WIDE kernel) --------------
    # Unchanged: this is what woke the pour up in iter_01 (1.0 per step, cups 0.21 m).
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
    # `want` tapers inside the last bead (env tolerance +-1 bead) and is 0 once spill
    # made the request unreachable -> the rotation income stops, the finish branch opens.
    want = _sat(need_beads)

    # ---- 8. rotation drive: the term that has to cross the ~90 deg wall (8.0) --
    # Gate (kept from iter_01, it is what finally paid): lip near the mouth with the
    # SAME radius the env latch uses, soft 2*cup_radius tail so 9-14 cm still pays.
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    # Progress measured against 1.4 * premature_tilt_limit (a multiple of a ctx angle):
    # reaches past the bead-release tilt for every fill and caps the income at 8.
    rot_prog = _sat(torch.clamp(ctx.src_tilt_toward_rcv, min=0.0) / (1.4 * lim))
    # NEW aim factor: a BONUS ON A FLOOR, not a gate. Both parts are smooth and their
    # tails stay clearly non-zero in the states iter_01 visited (lip 3 * cup_radius,
    # drop 0.03 m): xy part 0.24, height part 0.68 -> factor 0.58 (not 1e-7).
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)      # 1 inside 1r
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))  # low drop
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    # monotone in lip_d at fixed tilt (both lip_gate and aim_xy shrink with lip_d)
    pour_rot = 8.0 * src_g * ctx.src_grasp_ref_valid.float() * want * lip_gate * aim_factor * rot_prog

    # ---- 9. headline income: beads that actually arrive (30 per bead) ----------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 10. pouring past the request (15 per bead beyond the +-1 tolerance) ---
    over_beads = _sat(torch.clamp(ctx.bead_in_target_frac - ctx.pour_target_frac, min=0.0) * active - 1.0,
                      0.0, 5.0)
    pour_overshoot = -15.0 * over_beads

    # ---- 11. spill: 40 per lost bead (was 20) - the teeth for bad aim ----------
    # A bead that leaves the source and misses the receiver now costs MORE (40) than a
    # bead that arrives pays (30), so pouring beside the mouth is a net loss over the
    # episode while the rotation drive itself stays untouched. Charged on the increment.
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 12. finish branch: small, and per step below the pour income (R1) -----
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(ctx.src_cup_tilt / (0.3 * lim)))

    # ---- 13. terminal success bonus (env decides success, we only pay it) ------
    success_bonus = 20.0 * ctx.success.float()

    # ---- 14. premature tilt: charge AT the wall only, plus the latch reminder --
    # ramps in above 0.95 * limit over the last 0.05 * limit and only while the lip is
    # outside 2*cup_radius; inside that radius there is NO tilt charge at all (R3).
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
    # dimensionless unit-vector component (env voids at 0.3); ramp starts at half of it
    dir_ramp = _sat((ctx.pour_dir_xy[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 18. receiver cup: upright and on its own side of the centre line -----
    side_lo = 0.75 * r_cup                      # env voids below ~0.75 * cup_radius
    side_bad = _sat((side_lo - ctx.rcv_side_margin) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * lim))) - 1.5 * side_bad

    # ---- 19. always-on action rate (iter_18 level, small) --------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)

    # ---- 20. NEW: action reversal, alive from step 0 (weight 0.30) ------------
    # measured flip rate 0.49 with |a| ~ 0.5 -> about -0.04 per step (-0.15 if the
    # actions are saturated): ~5 % of the per-step state income, far below iter_21
    # where full-weight charges taught the policy not to touch the cups.
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip

    # ---- 21. NEW: arm tremor, alive from step 0, receiver arm weighted more ---
    # the receiver arm ran 1.2-1.3x shakier than the source arm; scale 1 rad/s so the
    # measured RMS 0.46 / 0.42 sits on the slope, total about -0.15 per step.
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))

    # ---- 22. palm saturation stays ramped by ADR (R7) -------------------------
    adr = float(ctx.adr_progress)
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
```

## 요구 숫자 (생성기 답변 요약)

- (a) iter_01 점유 상태(오차 3r = 0.123 m, 75도, 낙하 0.03 m): pour_rot 1.80/스텝, 완전 조준 5.82 의 31 %.
- (b) 완전 조준 5.82/스텝, 3r 의 3.2배. (c) 기울기 고정 시 오차가 줄면 단조 증가.
- 변경 2 크기: action_flip 약 -0.04/스텝(포화 시 -0.15), arm_tremor 약 -0.15/스텝, 합 -0.18~-0.29/스텝 = 상태 수입 2.85 의 6~10 %.
- 흘림 -40/알 > 도달 +30/알.
- 충돌 처리 메모: "3r 에서 30 % 이상"과 "2r 에서 0" 은 동시에 성립하지 않아 꼬리를 살린 tanh 창(2r 0.54, 3r 0.24)을 택했다.
- 로그 성분 합침(값 불변): approach_src+approach_rcv → approach_both, premature_charge+premature_latched → premature, cup_cup+nested → cup_clash.
