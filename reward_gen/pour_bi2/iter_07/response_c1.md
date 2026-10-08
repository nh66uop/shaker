brief 에 가설 id 줄이 없어서 prompt 전체를 한 후보로 구현했다. 최신 라운드 절 S(prompt.md:743-761)를 적용했고, 나머지 항과 가중치는 iter_06 그대로다(절 Q, prompt.md:706-718). 파일은 쓰지 않았다. 호출자가 `/home/user/rl_ws/hdgp/reward_gen/pour_bi2/iter_07/response_c1.md` 로 저장하면 된다. 수치는 모두 손으로 계산했고 코드는 실행하지 않았다.

## 과제 단계 (prompt 요구 1)
1. 두 손이 각자 컵의 원점 높이에서 옆으로 접근한다.
2. 감싸 쥐기(wrap 3개 이상)로 잡는다.
3. 각 컵을 들어 올린다. 이때 env 가 파지 기준을 기록한다.
4. 붓는 립(lip)을 리시버 입구 2r 안으로 가져간다.
5. 리시버 쪽으로 기울여 요청량(±1 알)을 붓는다.
6. 소스 컵을 다시 세운다. 그동안 컵끼리 닿지 않게 하고, 손은 자기 컵만 만지고, 흘림을 줄인다.

iter_06 에서는 소스 손이 컵을 쥔 채 손바닥을 아래로 명령해 컵을 테이블에 눌렀다(prompt.md:732-738). 이번 변경은 두 가지다.
- 접근: 상태 비례 항을 손바닥 속도 기반 진척 항으로 바꿨다.
- 들기: 한 손만 컵을 쥐고 있어도 그 컵의 들기에 보상을 준다.

## 코드

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


def _approach_progress(palm_pos, cup_pos, palm_vel, v_ref, retreat_frac):
    """Difference-form approach toward the cup ORIGIN from the SIDE.
    progress = palm velocity component toward the cup origin (positive only while closing in),
    squashed by tanh(v / v_ref); retreating costs retreat_frac of the same scale; a still palm earns 0.
    side_w = (horizontal part of the palm->origin direction)^2: 1 beside the origin, 0 straight above
    the mouth, so an approach from above earns less than one from the side at the same distance."""
    d = cup_pos - palm_pos
    d_norm = torch.sqrt((d * d).sum(dim=-1) + 1e-12)
    d_xy = torch.sqrt((d[:, :2] * d[:, :2]).sum(dim=-1) + 1e-12)
    u = d / d_norm.unsqueeze(-1)
    v_toward = (palm_vel * u).sum(dim=-1)
    prog = torch.tanh(v_toward / v_ref)
    prog = torch.clamp(prog, min=0.0) + retreat_frac * torch.clamp(prog, max=0.0)
    side_w = _sat(d_xy / d_norm) ** 2
    return prog * side_w


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bimanual pour: grasp both cups, lift, bring the cups together, aim the pouring
    LIP over the receiver mouth, tilt until the requested number of beads (+-1) has left
    the source cup and landed in the receiver, then stand the source cup back up.

    Change of this iteration (section S, user decision 09-28):
      1. approach_src / approach_rcv: difference-form progress of the palm toward its cup ORIGIN,
         weighted toward a side approach, fading once that hand grasps (replaces the state-proportional
         approach_both). A still palm earns 0, so it cannot be parked on.
      2. lift_src / lift_rcv: each cup's lift is paid when THAT hand holds it (0.5 per cup, max 1.0),
         instead of only when both hands hold.
      Everything else is identical to iter_06 (no action / motion penalties). 21 components.

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
    # full-range upright score for the finish branch: inverted 0 -> horizontal 0.5 -> upright 1
    up_full_src = 1.0 - torch.acos(torch.clamp(ctx.src_cup_up[:, 2], -1.0, 1.0)) / math.pi

    # Pre-lift / posture income must not be earnable by tilting before lifting (R6b).
    low_src = up_src
    low_rcv = up_rcv

    # ---- 1-2. approach progress per hand (0.4 each; 0 once that hand grasps) ---
    # speed scale 10 cup radii per second (reward-shape constant); retreat costs half the scale.
    # A palm without velocity data (older env: field is None) earns 0 here.
    v_ref = 10.0 * r_cup
    RETREAT = 0.5
    src_vel = ctx.src_palm_lin_vel if ctx.src_palm_lin_vel is not None else torch.zeros_like(ctx.src_palm_pos)
    rcv_vel = ctx.rcv_palm_lin_vel if ctx.rcv_palm_lin_vel is not None else torch.zeros_like(ctx.rcv_palm_pos)
    # weight 0.4 per hand: progress income, cannot be collected by standing still (R4)
    approach_src = 0.4 * (1.0 - src_g) * low_src * _approach_progress(
        ctx.src_palm_pos, ctx.src_cup_pos, src_vel, v_ref, RETREAT)
    approach_rcv = 0.4 * (1.0 - rcv_g) * low_rcv * _approach_progress(
        ctx.rcv_palm_pos, ctx.rcv_cup_pos, rcv_vel, v_ref, RETREAT)

    # ---- 3-4. enveloping grasp (weight 0.5 each) -------------------------------
    wrap_src = _sat(ctx.src_wrap_count / float(n_fing))
    wrap_rcv = _sat(ctx.rcv_wrap_count / float(n_fing))
    grasp_src = 0.5 * src_g * (0.4 + 0.6 * wrap_src)
    grasp_rcv = 0.5 * rcv_g * (0.4 + 0.6 * wrap_rcv)

    # ---- 5-6. lift per hand (0.5 per cup, saturates at a 2*cup_radius lift) ----
    h_src = _sat((ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]) / two_r)
    h_rcv = _sat((ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]) / two_r)
    # weight 0.5 per cup (max 1.0 per step total) keeps state income far below one bead (30)
    lift_src = 0.5 * src_g * h_src * low_src
    lift_rcv = 0.5 * rcv_g * h_rcv * low_rcv
    lifted = torch.minimum(h_src, h_rcv)

    lip_d = _dist_xy(ctx.src_pour_lip_pos, ctx.rcv_cup_mouth_pos)
    dz_lip = ctx.src_pour_lip_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]

    # ---- 7. bring the two cups together (weight 1.2, WIDE kernel) --------------
    converge_cups = 1.2 * both_g * lifted * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- 8. aim the LIP (not the rim centre) over the receiver mouth (0.8) -----
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * (0.6 * xy_score + 0.4 * h_score)

    # ---- 9. keep the grasp recorded at lift-off (iter_24 guard, weight 1.0) ----
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
    # rotation income stops once the remaining need is within the env tolerance (1 bead)
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

    # ---- 10. rotation drive (8.0): natural-scale progress, no cap --------------
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

    # ---- 11. headline income: beads that actually arrive (30 per bead) ---------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 12. pouring past the request (15 per bead beyond the +-1 tolerance) ---
    over_beads = _sat(torch.clamp(ctx.bead_in_target_frac - ctx.pour_target_frac, min=0.0) * active - 1.0,
                      0.0, 5.0)
    pour_overshoot = -15.0 * over_beads

    # ---- 13. spill: 40 per lost bead -------------------------------------------
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 14. finish branch (big real finish / small fallback) ------------------
    up_score = 0.4 * up_full_src + 0.6 * up_tight_src
    # weight 12 > max rotation income 8: standing up after a correct pour beats holding tilted
    finish_big = 12.0 * done * both_g * up_score
    # weight 1.5: a spill-unreachable request may never unlock the big finish
    finish_small = 1.5 * (1.0 - want_rot) * (1.0 - done) * both_g * up_score
    finish_upright = finish_big + finish_small

    # ---- 15. terminal success bonus (env decides success, we only pay it) ------
    success_bonus = 20.0 * ctx.success.float()

    # ---- 16. premature tilt: charge AT the env wall only, plus the latch reminder
    pre_x = torch.clamp(ctx.src_cup_tilt - 0.95 * lim, min=0.0) / (0.05 * lim)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate - 0.4 * ctx.premature_tilt.float()

    # ---- 17. real-robot safety: cups must never touch, nesting is never a pour -
    cup_clash = -(2.0 * torch.tanh(ctx.cup_cup_force / 5.0)
                  + 0.5 * ctx.cup_hit.float()
                  + 3.0 * ctx.cups_nested.float())

    # ---- 18. hands touch only their own cup ------------------------------------
    hand_foreign = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                           + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 19. pouring direction must not point away from the body --------------
    dir_ramp = _sat((ctx.pour_dir_xy[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 20. receiver cup: upright and on its own side of the centre line -----
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_side_margin) / side_lo)
    rcv_keep = 0.5 * rcv_g * up_tight_rcv - 1.5 * side_bad

    # ---- 21. fingertip-only holding on the source cup -------------------------
    # fingers pressing the cup (0.5 N contact floor) minus fingers wrapped with a middle /
    # distal link; 0.6 per tip-only finger.
    n_contact = (ctx.src_finger_force > 0.5).float().sum(dim=-1)
    tip_only = torch.clamp(n_contact - ctx.src_wrap_count.float(), min=0.0, max=float(n_fing))
    src_tip_only = -0.6 * src_g * tip_only

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_src": grasp_src,
        "grasp_rcv": grasp_rcv,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
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

## prompt 이 요구한 수치
계산 가정: r = 0.0405 m(prompt 의 "2r = 8.1 cm"), premature_tilt_limit = 73.6°(p50), wrap 3개 이상(hold = 1), 립 높이 차 dz_lip = 0.25r(aim_h = 1).

**절 S (a) 접근 수입.** 속도 기준 v_ref = 10r = 0.405 m/s 이다.

| 상태 | 계산 | 스텝당 수입 |
|---|---|---|
| 옆에서 컵 원점으로 0.3 m/s 접근 | 0.4 x tanh(0.741) | 0.252 |
| 입구 바로 위에서 0.3 m/s 하강 | side_w = 0 | 0 |
| 같은 속도, 45° 위에서 접근 | side_w = 0.5 | 0.126 |
| 컵 옆에 가만히 있는 손바닥 | 속도 0 | 0 |

**절 S (b) 들기 수입.** 소스 손만 컵을 쥐고 2r 들었을 때 lift_src = 0.5/스텝이다. 두 컵 합의 최대는 1.0/스텝이다.

**절 Q, R4 상태 수입.** 잡고, 들고, 조준했지만 붓지 않는 상태의 합은 grasp 1.0 + lift 1.0 + converge 1.2 + aim 0.8 + grip_keep 1.0 + rcv_keep 0.5 = 5.5/스텝이다. want_rot = 0 이면 finish_small 1.5 가 더해져 최대 7.0 이다. 한 알을 부으면 30 이므로 상태 수입이 더 작다.

**절 D.** 기울기 55°, 립이 입구에서 10 cm 떨어진 상태:

| 항목 | 값 |
|---|---|
| lip_gate | 0.770 |
| aim_factor | 0.715 |
| pour_rot (55°) | 0.94/스텝 |
| premature (55°) | 0. 벽 0.95 x lim = 69.9° 보다 아래 |

같은 10 cm 에서 더 기울일 때:

| 추가 기울기 | pour_rot | premature | 55° 대비 순손익 |
|---|---|---|---|
| +10° (65°) | 1.27 | 0 | +0.33 |
| +20° (75°) | 1.63 | 약 -5.6 | 약 -4.9 |
| +40° (95°) | 2.39 | -5.6 + 래치 -0.4 | 약 -4.6 |

즉 립을 2r 안으로 넣기 전에는 벽을 넘지 않는다. 2r 안에서는 premature 가 0 이다.

**절 N.** 립이 입구 위(1r 안)에 있을 때 pour_rot 은 60° 2.00, 100° 4.69, 140° 7.06, 180° 8.00 이다. 100°에서 140° 사이에서도 계속 오른다.

## 근거 메모
가설 id 없음(brief 에 가설 id 줄이 없어 prompt 전체를 구현).

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| approach_src / approach_rcv (새로 씀) | 증분 명령에서 손바닥이 random walk 한다. 손 위치를 알려 주는 항이 없다. 옆에서 원점으로 가는 차분형 진척, 정지 시 0 | prompt.md:735-741, 745-753, 137-140 |
| lift_src / lift_rcv (새로 씀) | 소스 컵 들기가 e800 에야 시작됐다. both_g 게이트 탓에 최대 0.25/스텝이었다. 손별 0.5 | prompt.md:726-730, 737-738, 754-756 |
| grasp_src / grasp_rcv, grip_keep | R6a iter_24 파지 기준 유지, 감싸 쥐기 | prompt.md:581-585, 624 |
| converge_cups, aim_lip | task/aim_dist 0.33~0.71 m 대역에 맞춘 넓은 커널, 립 조준 | prompt.md:444, 148 |
| pour_rot (그대로) | 절 N 의 상한 없는 회전 수입, lip_gate 2r(R2) | prompt.md:558-563, 640-645 |
| pour_delta, pour_overshoot, spill | 결과 수입 30/알, need = min 회계(R5), 흘림 40 | prompt.md:572-579, 149 |
| finish_upright, success_bonus | 12 / 1.5 마무리, up-vector 점수 | prompt.md:646-648, 712-714 |
| premature | 벽(0.95 x lim)과 2r 밖에서만 청구(R3) | prompt.md:565-570, 649-650 |
| cup_clash, hand_foreign, pour_dir, rcv_keep, src_tip_only | 실기 안전과 성공 무효 조건 | prompt.md:150, 154 |
| 액션·관절속도 벌점 없음 | 절 Q, 부드러움은 fabrics 제어기가 맡는다 | prompt.md:702-710, 757 |
| palm_lin_vel 이 None 일 때 속도 0 으로 대체 | 필드 기본값이 None 인 env 가 있다 | prompt.md:136 |

주의할 점과 설계 선택:
- 후퇴 비용 비율을 0.5 로 정했다. prompt 는 후퇴가 "조금" 손해이면 된다고만 적었다. 접근은 벌고 후퇴는 덜 잃는 비대칭 구조라 왕복 운동으로 보상을 버는 경로가 이론상 있다(가설). 쥐기 전에만 켜지는 항이고 가중치 0.4 이므로 한계는 약 0.2/스텝이다.
- 로그 연속성을 위해 approach 와 lift 를 손별로 나눴다. 그래서 항 수가 19에서 21로 늘었다(상한 25 안).
