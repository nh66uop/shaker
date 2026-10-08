brief 에 가설 id 줄이 없어서 prompt 전체를 한 후보로 구현했다. 이번 라운드 변경은 섹션 N(prompt.md:656-678)이고, 섹션 N 이 "나머지는 iter_04 그대로"라고 정했으므로 그 밖의 항, 가중치, 게이트는 그대로 두었다.

## 1. 과제와 단계

1. 접근: 오른손은 소스 컵으로, 왼손은 리시버 컵으로 간다.
2. 파지: 두 손 모두 손가락 3개 이상으로 감싸 쥔다(wrap). 손끝으로만 집으면 컵을 기울일 때 놓친다.
3. 들기: 컵을 20° 미만으로 세운 채 들어 올린다. 이 순간 env 가 파지 기준을 기록한다(P3).
4. 모으기와 조준: 두 컵이 닿지 않게 가까이 가져온다. 리시버 입구 위에 맞추는 것은 소스 컵의 림 중심이 아니라 붓는 립이다.
5. 붓기: 리시버 쪽으로 기울여 비드를 공중으로 떨어뜨린다. 요청량 ±1 알까지 붓는다.
6. 마무리: 소스 컵을 20° 이하로 다시 세우고, 그동안 두 손은 계속 쥐고 있다. 리시버 컵은 직립으로 유지한다.

iter_04 는 붓기 도중 멈췄다. 정책은 10 알 중 약 6.5 알을 붓고 최고 98° 에서 머물렀다. 원인은 rot_prog 가 약 103° 에서 포화된 것이다(prompt.md:651-654).

## 2. 요구받은 숫자

가정은 셋이다. cup_radius 0.0405 m(2r = 8.1 cm), premature_tilt_limit 73.6°(P2 의 p50), 손은 감싸 쥔 상태(wrap 3.35 이상, hold 1)다.

**립이 입구 위에 있을 때 회전 수입** (lip_gate 1, aim_factor 1, want_rot 1, pour_rot = 8·(1 − cos θ)/2)

| 기울기 | 스텝당 수입 |
|---|---|
| 60° | 2.00 |
| 100° | 4.69 |
| 140° | 7.06 |
| 180° | 8.00 |

100° 와 140° 사이에서도 +2.37 만큼 계속 오른다. 예전에는 103° 에서 포화됐다. 98° 에서의 기울기(gradient)는 예전 0.078/°, 지금 0.069/° 로 비슷하다.

**D1: 55°, 립이 입구에서 10 cm 떨어졌을 때**
- 게이트: lip_gate 0.770, aim_factor 0.715.
- 회전 수입: 스텝당 +0.94.
- premature 벌점: 0. 55° 는 0.95·limit(69.9°)보다 낮다.

**D2: 같은 상태에서 더 기울였을 때 스텝당 순손익 (55° 대비)**

| 추가 기울기 | 립 10 cm 밖 | 립이 입구 위 |
|---|---|---|
| +10° (65°) | +0.33 | +0.60 |
| +20° (75°) | −5.29 (수입 1.63, 벌점 −5.58, 래치 −0.4) | +1.26 |
| +40° (95°) | −4.58, 여기에 흘림 위험 알당 −40 | +2.64 |

- 입구 위에서는 기울기 벌점이 전혀 없다.
- 10 cm 떨어진 55° 상태에서는 립을 안으로 넣는 쪽이 이득이다. 게이트가 0.55 에서 1 로 오르면 +0.77 이다.

**D3 (R4): 자세만 유지해서 얻는 상태 수입의 합**
- 항목: grasp 1.0, lift 0.5, converge 1.2, aim 0.8, grip_keep 1.0, rcv_keep 0.5.
- 합은 스텝당 5.0 이다. approach 는 파지하면 0 이 된다.
- 비드 한 알은 pour_delta 로 30 이다. 따라서 5.0 < 30 이다.
- 제대로 끝냈을 때의 finish 12/스텝은 여전히 회전 수입 최대 8 보다 크다.

## 3. 보상 코드

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


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bimanual pour: grasp both cups, lift, bring the cups together, aim the pouring
    LIP over the receiver mouth, tilt until the requested number of beads (+-1) has left
    the source cup and landed in the receiver, then stand the source cup back up.

    Changes of this iteration (section N, "no angle prescription"):
      N1  Rotation progress = (1 - cos(theta_toward)) / 2 on the cup's natural scale
          (upright 0, horizontal 0.5, inverted 1). No cap, no target angle; it stops only
          through the outcome want_rot (remaining request within the env tolerance).
          lip_gate, aim floor factor and grasp hold factor unchanged.
      N2  Upright scores (finish branch, receiver upright, pre-lift fade) use the cup
          up-vector z component (clamped at 0) instead of tilt / (k * premature_tilt_limit).
      N3  Premature charge kept as is (it only announces the env latch rule).
      Everything else identical to iter_04.  23 components.

    Design rules: lengths are multiples of ctx.cup_radius; ctx.premature_tilt_limit is used
    only by the premature charge that mirrors the env latch; no absolute angle is hardcoded.
    """
    zeros = torch.zeros_like(ctx.src_cup_tilt)

    # ---- shared scales ---------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # env latch radius for the lip (xy)
    wide = 10.0 * r_cup                      # WIDE kernel: covers the visited 0.2-0.45 m band
    lim = torch.clamp(ctx.premature_tilt_limit, min=1e-3)   # (N,) rad, env rule; premature charge only
    n_fing = max(int(ctx.src_tips_pos.shape[1]), 1)
    adr = float(ctx.adr_progress)            # 0 until the success rate passes the ADR trigger

    src_g = ctx.src_grasped.float()
    rcv_g = ctx.rcv_grasped.float()
    both_g = src_g * rcv_g

    # ---- N2: upright scores from the cup up-vector (1 upright, 0 horizontal) ---
    up_src = torch.clamp(ctx.src_cup_up[:, 2], min=0.0, max=1.0)
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], min=0.0, max=1.0)
    # sharpness exponent of the "tight" upright score: a reward-shape constant, not an angle
    UP_SHARP = 8.0
    up_tight_src = up_src ** UP_SHARP
    up_tight_rcv = up_rcv ** UP_SHARP

    # Pre-lift / posture income must not be earnable by tilting before lifting (R6b).
    low_src = up_src
    low_rcv = up_rcv

    # ---- 1. approach both hands (0.4 each; dies out once that cup is grasped) ---
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_both = (0.4 * (1.0 - src_g) * low_src * (1.0 - torch.tanh(d_src / two_r))
                     + 0.4 * (1.0 - rcv_g) * low_rcv * (1.0 - torch.tanh(d_rcv / two_r)))

    # ---- 2-3. enveloping grasp (weight 0.5 each, unchanged) --------------------
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
    # pour income keeps paying for the last bead (taper inside the last bead)
    want = _sat(need_beads)
    # rotation income stops once the remaining need is within the env tolerance (1 bead):
    # 0 at need <= 1 bead, 1 at need >= 2 beads. This is the ONLY stop of the rotation income.
    want_rot = _sat(need_beads - 1.0)

    # poured amount within +-1 bead of the request (env tolerance); soft edge 1.0 -> 1.5 bead
    poured_beads = ctx.bead_in_target_frac * active
    amount_err = torch.abs(poured_beads - ctx.pour_target_beads)
    on_tol = _sat((1.5 - amount_err) / 0.5)
    # a "real" finish: nothing latched that voids the env success
    clean = ((1.0 - ctx.premature_tilt.float())
             * (1.0 - ctx.cups_nested.float())
             * (1.0 - ctx.cup_hit.float()))
    done = on_tol * clean

    # ---- 8. rotation drive (8.0): N1 natural-scale progress, no cap -----------
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    theta_toward = torch.clamp(ctx.src_tilt_toward_rcv, min=0.0, max=math.pi)
    # upright 0, horizontal 0.5, fully inverted 1; monotone over [0, pi], no target angle
    rot_prog = 0.5 * (1.0 - torch.cos(theta_toward))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)      # 1 inside 1r
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))  # low drop
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    # grasp hold factor: envelope 0 at <= 1 wrapped finger, 1 at >= 3; floor 0.3 (not a gate)
    envelope = _smoothstep((ctx.src_wrap_count.float() - 1.0) / 2.0)
    HOLD_FLOOR = 0.3
    hold = HOLD_FLOOR + (1.0 - HOLD_FLOOR) * envelope
    pour_rot = (8.0 * src_g * ctx.src_grasp_ref_valid.float() * want_rot
                * lip_gate * aim_factor * hold * rot_prog)

    # ---- 9. headline income: beads that actually arrive (30 per bead) ----------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 10. pouring past the request (15 per bead beyond the +-1 tolerance) ---
    over_beads = _sat(torch.clamp(ctx.bead_in_target_frac - ctx.pour_target_frac, min=0.0) * active - 1.0,
                      0.0, 5.0)
    pour_overshoot = -15.0 * over_beads

    # ---- 11. spill: 40 per lost bead (kept) -----------------------------------
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 12. finish branch (big real finish / small fallback), N2 upright score -
    up_score = 0.4 * up_src + 0.6 * up_tight_src
    # weight 12 > max rotation income 8: standing up after a correct pour beats holding tilted
    finish_big = 12.0 * done * both_g * up_score
    # weight 1.5: a spill-unreachable request may never unlock the big finish
    finish_small = 1.5 * (1.0 - want_rot) * (1.0 - done) * both_g * up_tight_src
    finish_upright = finish_big + finish_small

    # ---- 13. terminal success bonus (env decides success, we only pay it) ------
    success_bonus = 20.0 * ctx.success.float()

    # ---- 14. premature tilt: charge AT the env wall only, plus the latch reminder (N3 kept)
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

    # ---- 18. receiver cup: upright (N2) and on its own side of the centre line -
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_side_margin) / side_lo)
    rcv_keep = 0.5 * rcv_g * up_tight_rcv - 1.5 * side_bad

    # ---- 19. fingertip-only holding on the source cup -------------------------
    # fingers pressing the cup (0.5 N contact floor) minus fingers wrapped with a middle /
    # distal link; 0.6 per tip-only finger.
    n_contact = (ctx.src_finger_force > 0.5).float().sum(dim=-1)
    tip_only = torch.clamp(n_contact - ctx.src_wrap_count.float(), min=0.0, max=float(n_fing))
    src_tip_only = -0.6 * src_g * tip_only

    # ---- 20. always-on action rate (iter_18 level, small; allowed by R7) ------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)

    # ---- 21. action reversal, behind the ADR gate (R7) ------------------------
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -adr * 0.30 * flip

    # ---- 22. arm tremor, receiver weighted more, ADR-gated (R7) ---------------
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -adr * (0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))

    # ---- 23. palm saturation stays ramped by ADR (R7) -------------------------
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
        "src_tip_only": src_tip_only,
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

## 4. 근거 메모

가설: 없음. brief 에 가설 id 줄이 없어서 prompt 전체(이번 라운드 변경인 섹션 N)를 한 후보로 구현했다.

**규칙끼리의 충돌**
- R9 는 항 수를 15~22 로 권한다(prompt.md:617). 섹션 N 은 나머지 항을 iter_04 그대로 두라고 한다(prompt.md:673-675).
- 어느 쪽이 더 보수적인지 판단하기 어렵다. 그래서 범위 밖 변경을 하지 않는 쪽을 택했다. 항 수는 23 으로 유지했고, 상한 25 이내다(prompt.md:162).

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| pour_rot: rot_prog = (1 − cos θ)/2, 상한 없음 | 1.4·limit 에서 포화되어 98° 에서 정체, tilt_over_release −6.6° | prompt.md:641, 651-654, 661-666 |
| pour_rot: lip_gate, aim_factor, hold, want_rot 유지 | N1 의 "exactly as they are", 중단 조건은 결과로만 | prompt.md:664-666 |
| finish_upright 의 up_score, up_tight (up_z, up_z^8) | N2: 업벡터 z 로 교체 | prompt.md:667-669 |
| rcv_keep 직립 항 (up_z^8) | N2, 리시버 기울기 20° 이하가 성공 조건 | prompt.md:667-669, 144 |
| low_src / low_rcv = up_z (들기 전 fade) | N2, R6b 기울인 채 들면 들기 전 수입을 받지 못하게 | prompt.md:605-606, 667-669 |
| premature 유지 (0.95·limit, 립이 2r 밖일 때) | N3, R3 | prompt.md:586-591, 670-671 |
| 나머지 항과 가중치 그대로 (finish 12/1.5, spill 40, converge, aim, grip_keep, tip_only, ADR 게이트) | "Keep every other term" | prompt.md:673-675 |
| UP_SHARP = 8 지수 | 보상 모양의 날카로움 상수이며 각도 목표가 아님 | prompt.md:161 (추정: 옛 up_tight 의 조임 정도를 맞춘 값) |
| 60/100/140/180° 수입 수치 | 답에 포함하라는 요구 | prompt.md:677-678 |
| D1~D3 수치 | 답해야 할 숫자 | prompt.md:619-624 |

**위험 (가설)**
- up_z 를 0 에서 자르므로 90° 를 넘으면 finish_big 의 기울기(gradient)가 0 이다. 98° 에서 마무리하려면 90° 아래까지 신호 없이 되돌아와야 한다. 다만 done 이 되면 want_rot 이 0 이 되어 계속 기울여 둘 수입도 사라진다.
- 98° 에서의 회전 수입이 예전 약 7.6 에서 약 4.6 으로 준다. iter_04 체크포인트에서 웜스타트하면 초기 가치 추정이 흔들릴 수 있다.
- 회전 수입이 끝까지 오르므로 과회전해서 흘릴 수 있다. 이 위험을 막는 것은 R5 의 need = min 과 spill −40/알뿐이다.
