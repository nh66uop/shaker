가설 id 없음 처리: brief 에 가설 id 줄이 없고 prompt 에도 Hypothesis A/B 표기가 없다. 그래서 규칙대로 prompt 전체(섹션 N 과 Review items)를 한 후보로 구현했다. 재시도 응답이다. 기록 경로는 호출자가 정한다(예: /home/user/rl_ws/hdgp/reward_gen/pour_bi2/iter_05/response_c1r.md).

## 1. 과제와 단계

1. 오른손이 소스 컵, 왼손이 리시버 컵을 감싸 쥔다(5지 인벨롭).
2. 두 컵을 기울이지 않고 든다. 컵이 20° 아래일 때만 env 가 파지 기준을 기록한다(P3).
3. 소스 컵의 붓는 입술(lip)을 리시버 입구 위로 가져간다. 두 컵은 서로 닿지 않고, 리시버는 자기 쪽에 둔다.
4. 리시버 쪽으로 기울여 요청량(±1 알)만큼 붓는다. 첫 비드는 약 90° 이상에서 나온다.
5. 소스 컵을 다시 세운다. 양손은 계속 잡고 있어야 한다.

## 충돌 메모 (코드 앞)

- **N2 와 Review item 1:** N2 는 직립 점수를 `clamp(up_z, min=0)` 으로 만들라고 한다. Review item 1 은 마무리 분기의 직립 점수가 완전 뒤집힘부터 직립까지 전 구간에서 단조 증가하고 기울기가 0 이 아니어야 한다고 한다. 둘이 충돌한다.
  - 해결: 마무리 분기(finish_big, finish_small)의 넓은 점수만 up-vector 로 만든 전 구간 점수 `up_full = 1 - acos(up_z)/pi` 로 바꿨다(뒤집힘 0, 수평 0.5, 직립 1). 각도 상수와 premature_tilt_limit 은 쓰지 않는다.
  - 리시버 직립(rcv_keep)과 들기 전 페이드(low_*)는 지적되지 않았으므로 N2 그대로 둔다.
- **R9 와 N:** R9 는 항 수 15~22 를 권하고, N 은 나머지 항을 모두 유지하라고 한다. N 을 따라 23 항을 유지했다(25 상한 이내).

## 이번 재시도에서 바꾼 것

- **Review 1:** 마무리 점수를 `up_score = 0.4*up_full + 0.6*up_src^8` 로 바꿨다. finish_small 도 up_tight_src 대신 이 up_score 를 쓴다.
- finish_big 값(done=1, 양손 파지): 180° 0.0, 130° 1.33, 115° 1.73, 100° 2.13, 95° 2.27, 90° 2.40, 80° 2.67, 60° 3.23, 20° 8.64, 5° 11.65, 0° 12.0.
  - 전 구간에서 단조 증가한다.
  - 기울기는 180° 끝점을 포함해 어디서나 0 이 아니다(넓은 항이 각도에 선형, 0.4·12/pi ≈ 1.53 per rad).
- **Review 2:** 가중치와 floor 는 바꾸지 않았다. 보고 수치만 옮긴다.
  - 75°, lip 12.3 cm: 회전 수입 +0.889/step (iter_04 +1.745).
  - 완전 조준: +2.735/step (iter_04 +5.372).
  - 조준 비율은 32 % 로 그대로다.

## 요구 수치

**N: lip 이 입구 위, 인벨롭 파지일 때 회전 수입** (8·rot_prog, 나머지 계수 1)

| 리시버 쪽 기울기 | 60° | 100° | 140° | 180° |
|---|---|---|---|---|
| 회전 수입 / step | 2.00 | 4.69 | 7.06 | 8.00 |

- 100° 에서 140° 사이도 +2.37 올라간다.
- 수입이 멈추는 곳은 결과 쪽 하나뿐이다: 남은 요청이 1 알 이내가 되면(want_rot) 멈춘다.

**D1: 55°, lip 10 cm 인 상태**

- 가정: r = 4.05 cm, dz_lip = 0.25r, wrap ≥ 3, lim p50 73.6°.
- 계수: lip_gate 0.770, aim_factor 0.715, rot_prog 0.213.
- 회전 수입: +0.94/step.
- premature 청구: 0. 55° 는 0.95·lim(69.9°) 아래다.

**D2: 그 상태에서 더 기울일 때 스텝당 순손익** (lip 10 cm 그대로)

| 더 기울인 양 | 회전 수입 | premature 청구 | 순손익(55° 대비) |
|---|---|---|---|
| +10° (65°) | 1.27 | 0 | +0.33 |
| +20° (75°) | 1.63 | -5.58 (out_gate 0.47) | 약 -4.9, 래치 -0.4 와 성공 무효 추가 |
| +40° (95°) | 2.39 | -5.63 - 0.4 | 약 -4.6 |

- +40° 에는 흘림 위험(-40/알)이 더 붙는다.
- lip 을 2r(8.1 cm) 안으로 넣으면 청구가 0 이 된다. 그러면 95° 순손익은 +1.45 이상이고 조준이 좋아질수록 커진다.

**D3: 상태 수입과 결과 수입 (R4)**

- 상태 수입 최대값의 합은 약 5.0/step 이다: 파지 1.0 + lift 0.5 + converge 1.2 + aim 0.8 + grip_keep 1.0 + rcv_keep 0.5.
- 이전 런에서 실측한 합은 약 3.1 이다.
- 한 알을 부으면 30 을 받는다. 상태 수입보다 결과 수입이 크다.

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
      N2  Upright scores (receiver upright, pre-lift fade) use the cup up-vector z component
          (clamped at 0) instead of tilt / (k * premature_tilt_limit).
          Finish branch (review item 1): the broad upright score is built from the up-vector
          over the WHOLE range, up_full = 1 - acos(up_z) / pi (inverted 0, horizontal 0.5,
          upright 1), so standing the cup back up pays a nonzero gradient from any tilt,
          including tilts beyond horizontal.
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
    # full-range upright score for the finish branch (review item 1): from the up-vector,
    # inverted 0 -> horizontal 0.5 -> upright 1, linear in the cup's own attitude, so its
    # slope is nonzero everywhere between fully inverted and upright
    up_full_src = 1.0 - torch.acos(torch.clamp(ctx.src_cup_up[:, 2], -1.0, 1.0)) / math.pi

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

    # ---- 12. finish branch (big real finish / small fallback) ------------------
    # broad part: full-range up-vector score (nonzero slope from inverted to upright);
    # tight part: sharp peak at upright (up_z ** 8), unchanged from N2
    up_score = 0.4 * up_full_src + 0.6 * up_tight_src
    # weight 12 > max rotation income 8: standing up after a correct pour beats holding tilted
    finish_big = 12.0 * done * both_g * up_score
    # weight 1.5: a spill-unreachable request may never unlock the big finish; same
    # full-range score so the fallback also leads back up from beyond horizontal
    finish_small = 1.5 * (1.0 - want_rot) * (1.0 - done) * both_g * up_score
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

## 근거 메모

가설: 지정 없음(brief 에 가설 id 줄이 없음). prompt 전체(섹션 N 과 Review items)를 한 후보로 구현했다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| finish_upright (up_full_src 를 넣은 up_score, finish_small 도 같은 점수) | 90° 이상에서 finish 가 0 이고 기울기도 0 이었다. i04 는 98° 에서 정점을 찍는다 | prompt_c1_repair.md:956, 643-648 |
| pour_rot (rot_prog = (1-cos)/2, 상한 없음, want_rot 로만 멈춤) | 1.4·lim 포화 탓에 약 103° 이상에서 수입이 없었고 6.6° 모자라게 멈췄다 | :651-654, :661-666 |
| pour_rot 가중치, aim/hold floor 유지 | Review 2: 수치만 보고하고 바꾸지 말 것 | :957, :666 |
| rcv_keep, low_src/low_rcv (clamp 한 up_z) | N2 직립 점수 | :667-669 |
| premature (0.95·lim, lip 이 2r 밖일 때만) | N3 유지, R3 | :670-671, :586-591 |
| pour_delta 30/알, spill 40/알, need=min | R1 결과 수입, R5 흘림 회계 | :575-577, :597-600, :674-675 |
| grasp/grip_keep/src_tip_only/hold | i04 에서 인벨롭 3.35 로 회복, 유지 지시 | :636, :645, :673-675 |
| ADR 게이트 벌점(flip, tremor, sat) | R7 탐색 보호, ADR 0 | :608-611, :435 |
| 항 수 23 | N "나머지 항 유지"와 R9(15~22)가 충돌해 N 을 따름 | :617, :673 |
