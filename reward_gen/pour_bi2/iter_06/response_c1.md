가설 지정 없음: brief 에 가설 id 줄이 없어서 prompt 전체를 후보 하나로 구현했습니다. 이번 라운드 변경은 섹션 Q(prompt.md:741-753) 하나입니다.

## 1. 과제 해석과 단계

1. 접근: 오른손은 소스 컵으로, 왼손은 리시버 컵으로 다가갑니다.
2. 감싸 쥐기: 손가락 3개 이상의 중간/원위 마디가 컵에 닿는 인벨롭 파지입니다.
3. 들기: 컵이 20° 미만으로 서 있을 때 들어야 env 가 파지 기준을 기록합니다.
4. 모으기와 조준: 리시버는 자기 쪽(+y)에 두고, 소스 팔이 붓는 쪽 림(lip)을 리시버 입구 위로 가져옵니다.
5. 기울이기: 리시버 쪽으로 기울입니다. 비드는 약 90° 이상에서 나오고, 요청량(±1 알)에서 멈춥니다.
6. 마무리: 소스 컵을 다시 세우고 양손 파지를 유지합니다. 이번 라운드에서 이 단계는 작은 보상만 받습니다.

- 전 과정에 걸친 안전 조건: 컵끼리 부딪히지 않기, 컵 겹쳐 넣기 금지, 손은 자기 컵에만 닿기, 몸 바깥쪽(+x)으로 붓지 않기, 흘림 최소화.
- 제어 쪽 변화: 이번 라운드부터 매끄러움은 fabrics 속도 명령 제어기가 맡습니다(P3, prompt.md:737-738). 그래서 액션/관절속도 벌점 4개를 뺐습니다.

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


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bimanual pour: grasp both cups, lift, bring the cups together, aim the pouring
    LIP over the receiver mouth, tilt until the requested number of beads (+-1) has left
    the source cup and landed in the receiver, then stand the source cup back up.

    Change of this iteration (section Q, user decision 09-28):
      Smoothness is handled by the fabrics controller (palm command = velocity), not by
      the reward. The four action / motion penalty terms of iter_05 are removed:
      action_rate, action_flip, arm_tremor, action_reg_adr. No term reads ctx.actions,
      ctx.prev_actions or arm joint velocities.
      Every other term, weight and gate is identical to iter_05.  19 components.

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

    src_g = ctx.src_grasped.float()
    rcv_g = ctx.rcv_grasped.float()
    both_g = src_g * rcv_g

    # ---- upright scores from the cup up-vector (1 upright, 0 horizontal) -------
    up_src = torch.clamp(ctx.src_cup_up[:, 2], min=0.0, max=1.0)
    up_rcv = torch.clamp(ctx.rcv_cup_up[:, 2], min=0.0, max=1.0)
    # sharpness exponent of the "tight" upright score: a reward-shape constant, not an angle
    UP_SHARP = 8.0
    up_tight_src = up_src ** UP_SHARP
    up_tight_rcv = up_rcv ** UP_SHARP
    # full-range upright score for the finish branch: inverted 0 -> horizontal 0.5 -> upright 1,
    # nonzero slope everywhere between fully inverted and upright
    up_full_src = 1.0 - torch.acos(torch.clamp(ctx.src_cup_up[:, 2], -1.0, 1.0)) / math.pi

    # Pre-lift / posture income must not be earnable by tilting before lifting (R6b).
    low_src = up_src
    low_rcv = up_rcv

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

    # ---- 5. bring the two cups together (weight 1.2, WIDE kernel) --------------
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

    # ---- 8. rotation drive (8.0): natural-scale progress, no cap ---------------
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

    # ---- 11. spill: 40 per lost bead -------------------------------------------
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 12. finish branch (big real finish / small fallback) ------------------
    up_score = 0.4 * up_full_src + 0.6 * up_tight_src
    # weight 12 > max rotation income 8: standing up after a correct pour beats holding tilted
    finish_big = 12.0 * done * both_g * up_score
    # weight 1.5: a spill-unreachable request may never unlock the big finish
    finish_small = 1.5 * (1.0 - want_rot) * (1.0 - done) * both_g * up_score
    finish_upright = finish_big + finish_small

    # ---- 13. terminal success bonus (env decides success, we only pay it) ------
    success_bonus = 20.0 * ctx.success.float()

    # ---- 14. premature tilt: charge AT the env wall only, plus the latch reminder
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
    rcv_keep = 0.5 * rcv_g * up_tight_rcv - 1.5 * side_bad

    # ---- 19. fingertip-only holding on the source cup -------------------------
    # fingers pressing the cup (0.5 N contact floor) minus fingers wrapped with a middle /
    # distal link; 0.6 per tip-only finger.
    n_contact = (ctx.src_finger_force > 0.5).float().sum(dim=-1)
    tip_only = torch.clamp(n_contact - ctx.src_wrap_count.float(), min=0.0, max=float(n_fing))
    src_tip_only = -0.6 * src_g * tip_only

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
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components
```

## 근거 메모

가설 지정 없음 (prompt 전체 구현, 이번 라운드 변경은 섹션 Q)

**남은 항 (19개).** approach_both, grasp_src, grasp_rcv, lift_both, converge_cups, aim_lip, grip_keep, pour_rot, pour_delta, pour_overshoot, spill, finish_upright, success_bonus, premature, cup_clash, hand_foreign, pour_dir, rcv_keep, src_tip_only

**삭제한 항.** action_rate, action_flip, arm_tremor, action_reg_adr. 쓰지 않게 된 `adr` 변수와 `_rms` 헬퍼도 같이 지웠습니다. 이제 코드는 ctx.actions, ctx.prev_actions, 관절속도를 읽지 않습니다.

**파지, 들기, 조준까지 됐지만 붓지 않는 상태의 스텝당 양의 수입 (최대).**
- 컵을 세운 상태: grasp 0.5+0.5, lift 0.5, converge 1.2, aim 0.8, grip_keep 1.0, rcv_keep 0.5 를 합쳐 5.0 입니다.
  - 이때 pour_rot 는 0 입니다(rot_prog=0).
  - finish_small 도 0 입니다(요청이 2알 이상 남아 want_rot=1).
- 붓지 못한 채 기울여 버티는 경우: pour_rot 까지 더해 최대 5.0 + 8.0 = 13.0 입니다(완전 역전일 때).
- 두 경우 모두 비드 1알 수입 30 보다 작아 R4 를 만족합니다.

**회전 수입.** 조건은 lip 이 입구 위, 인벨롭 파지, aim_factor=1, want_rot=1 입니다. 식은 8·(1−cos θ)/2 입니다.

| θ (리시버 쪽) | 회전 수입 / 스텝 |
|---|---|
| 60° | 2.00 |
| 100° | 4.69 |
| 140° | 7.06 |
| 180° | 8.00 |

100° 에서 140° 사이에서도 계속 오릅니다.

**참고 (D1).** 조건은 55°, lip 10 cm, dz 이상적, r=4.1 cm 입니다.
- 회전 수입: lip_gate 0.770 × aim_factor 0.717 × rot_prog 0.213 × 8 ≈ 0.94 / 스텝입니다.
- premature 청구: 0.95·limit(p50 기준 약 70°) 미만이라 0 입니다.

**규칙 충돌.** R7(prompt.md:622-625)은 iter_18 수준의 상시 action-rate 청구를 허용합니다. 섹션 Q(prompt.md:743-745)는 사용자 결정으로 이를 삭제하고, 액션을 읽는 벌점 추가도 금지합니다. 가장 최근 사용자 결정인 Q 를 따랐고, 벌점을 더 넣지 않는 쪽이라 Q 가 더 보수적입니다.

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| action_rate, action_flip, arm_tremor, action_reg_adr 삭제 | 매끄러움은 fabrics 제어기가 맡음(속도 명령 모드) | prompt.md:737-738, 743-745 |
| pour_rot (8.0, 상한 없는 (1−cos θ)/2) | i04 에서 103° 포화로 멈춤, 각도 처방 금지 | prompt.md:665-680, 747-748 |
| pour_delta (30/알), want=min 회계 | 결과 수입이 헤드라인, 흘림 회계 | prompt.md:589-591, 611-614 |
| finish_upright (12 / 1.5, up_full + up_tight) | 마무리는 작게, 전 범위 세우기 기울기 | prompt.md:687-688, 748 |
| premature (0.95·limit 이상 + lip 2r 밖) | 벽 앞이 아니라 벽에서 청구 | prompt.md:600-605, 684-685 |
| grip_keep | iter_24 파지 기준 유지 원칙 | prompt.md:616-620 |
| approach, lift 의 up_z 페이드 | 들기 전 기울이기로 자세 수입 벌기 차단 | prompt.md:619-620, 681-683 |
| converge_cups, aim_lip, lip_gate 2r | 방문 상태에서 게이트가 0 에 붙지 않게 | prompt.md:593-598 |
| cup_clash, hand_foreign, pour_dir, rcv_keep, src_tip_only | 안전과 성공 무효 조건, iter_05 그대로 유지 | prompt.md:747-750 |
| 상태 수입 5.0 (기울여 버티면 최대 13) < 1알 30 | R4, Q 요구 수치 | prompt.md:607-609, 752-753 |

파일은 쓰지 않았습니다. 호출자가 경로를 검사한 뒤 기록하면 됩니다. 읽은 prompt 는 /home/user/rl_ws/hdgp/reward_gen/pour_bi2/iter_06/prompt.md 입니다.
