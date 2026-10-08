brief 에 가설 id 줄이 없어서 prompt 전체를 한 후보로 구현했다. 이번은 재시도다. Review item 1 에 따라 approach 진척 항만 고쳤고 나머지 코드는 이전 응답과 같다. 파일은 쓰지 않았다. 원래 경로 규칙대로라면 호출자가 `/home/user/rl_ws/hdgp/reward_gen/pour_bi2/iter_07/response_c1r.md` 에 쓴다. 다만 가설 id 가 없으니 파일 이름은 호출자가 확인해야 한다.

---

**과제와 단계.** 양손이 각자 컵으로 다가가 옆에서 감싸 쥔다. 두 컵을 들어 올리고, 붓는 립을 리시버 입구 위로 가져간다. 리시버 쪽으로 기울여 요청량(±1 알)만 붓는다. 소스 컵을 다시 세우고, 그동안 양손은 계속 쥐고 있는다. 수입의 중심은 실제로 옮겨진 비드(pour_delta, 알당 30)다.

**이번 수정 (Review item 1).** 이전 approach 항은 `tanh` 뒤에 RETREAT = 0.5 비대칭을 붙였다. 그래서 컵 옆에서 왕복하면 두 스텝마다 +0.126 이 남았다. 이번에는 진척을 손바닥 속도에 대해 선형인 홀함수로 바꿨다: `clamp(v_toward / v_ref, -3, 3)`. 멀어질 때 드는 비용과 다가갈 때 받는 수입이 정확히 같다.
- 같은 경로를 되짚는 왕복은 속도 프로파일과 무관하게 정확히 0 이다. 수입이 위치의 함수 × 속도라서 선적분이 역방향에서 부호만 바뀌기 때문이다.
- `tanh` 도 뺐다. `tanh` 는 느린 접근과 빠른 후퇴의 조합에 순수익을 준다.
- 클립 ±3 은 수치 보호용 보상 형상 상수다. 3·v_ref ≈ 1.2 m/s 로, 명령 최대 속도보다 커서 정상 동작에서는 걸리지 않는다.
- side_w, 가중 0.4, 파지 후 소멸, low_* 는 그대로 두었다.

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


def _approach_progress(palm_pos, cup_pos, palm_vel, v_ref, v_clip):
    """Difference-form approach toward the cup ORIGIN from the SIDE.
    progress = palm velocity component toward the cup origin, divided by v_ref.
    It is LINEAR and ODD in the velocity: moving away costs exactly what moving closer pays,
    so a back-and-forth motion over the same path nets exactly 0 (review item 1), and a still
    palm earns 0. The symmetric clip +-v_clip is only a numeric guard far above reachable palm speeds.
    side_w = (horizontal part of the palm->origin direction)^2: 1 beside the origin, 0 straight above
    the mouth, so an approach from above earns less than one from the side at the same distance."""
    d = cup_pos - palm_pos
    d_norm = torch.sqrt((d * d).sum(dim=-1) + 1e-12)
    d_xy = torch.sqrt((d[:, :2] * d[:, :2]).sum(dim=-1) + 1e-12)
    u = d / d_norm.unsqueeze(-1)
    v_toward = (palm_vel * u).sum(dim=-1)
    prog = torch.clamp(v_toward / v_ref, -v_clip, v_clip)   # symmetric: no retreat discount
    side_w = _sat(d_xy / d_norm) ** 2
    return prog * side_w


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bimanual pour: grasp both cups, lift, bring the cups together, aim the pouring
    LIP over the receiver mouth, tilt until the requested number of beads (+-1) has left
    the source cup and landed in the receiver, then stand the source cup back up.

    Change of this iteration (section S, user decision 09-28, plus review item 1):
      1. approach_src / approach_rcv: difference-form progress of the palm toward its cup ORIGIN,
         weighted toward a side approach, fading once that hand grasps. The progress is linear and
         symmetric in the palm velocity (retreat costs exactly what approach pays), so neither a
         still palm nor a back-and-forth palm earns anything.
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
    # speed scale 10 cup radii per second (reward-shape constant). Symmetric linear progress:
    # retreat costs exactly what approach pays (review item 1). Clip +-3 v_ref is a numeric guard only.
    # A palm without velocity data (older env: field is None) earns 0 here.
    v_ref = 10.0 * r_cup
    V_CLIP = 3.0
    src_vel = ctx.src_palm_lin_vel if ctx.src_palm_lin_vel is not None else torch.zeros_like(ctx.src_palm_pos)
    rcv_vel = ctx.rcv_palm_lin_vel if ctx.rcv_palm_lin_vel is not None else torch.zeros_like(ctx.rcv_palm_pos)
    # weight 0.4 per hand: progress income, cannot be collected by standing still or oscillating (R4)
    approach_src = 0.4 * (1.0 - src_g) * low_src * _approach_progress(
        ctx.src_palm_pos, ctx.src_cup_pos, src_vel, v_ref, V_CLIP)
    approach_rcv = 0.4 * (1.0 - rcv_g) * low_rcv * _approach_progress(
        ctx.rcv_palm_pos, ctx.rcv_cup_pos, rcv_vel, v_ref, V_CLIP)

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

**요구 수치.** 계산 가정: r = 4.05 cm (2r = 8.1 cm), v_ref = 0.405 m/s.

- **S(a) approach**
  - 옆에서 컵 원점 쪽으로 0.3 m/s: +0.296/step. 같은 속도로 멀어지면 -0.296/step 이라 왕복 한 쌍은 0 이다. 이전 응답은 +0.126 이었다.
  - 입구 바로 위에서 원점 쪽으로 0.3 m/s: 0 (side_w = 0). 45° 위에서는 +0.148.
  - 컵 옆에 가만히 있는 손바닥: 0.
- **S(b) lift:** 소스 손만 쥐고 소스 컵을 2r 들었을 때 +0.5/step.
- **Q/R4 상태 수입:** 쥐고, 들고, 조준했지만 붓지 않는 상태의 수입은 grasp 1.0 + lift 1.0 + converge 1.2 + aim 0.8 + grip_keep 1.0 + rcv_keep 0.5 = 약 5.5/step 이다. 비드 1 알(30)보다 작다.
- **N 회전 수입:** 립이 입구 위에 있고 5지로 감싼 상태. 60° 2.00, 100° 4.69, 140° 7.06, 180° 8.00. 100° 에서 140° 사이에서도 계속 오른다.
- **D, 틸트 55° / 립 10 cm**
  - 55° 에서: pour_rot 약 0.94/step, premature 0. 계산 가정은 limit p50 73.6°, 청구 시작 0.95·lim = 69.9°.
  - +10° (65°): +1.27, 순증 +0.33.
  - +20° (75°): +1.63, premature -5.58, 순손익 약 -3.95.
  - +40° (95°): +2.39, premature -5.6 에 래치 -0.4 가 더해져 순손익 약 -3.6.
  - 립이 2r 안에 있으면 premature 청구가 0 이다.

재시도 (가설 id 미지정: prompt 전체를 한 후보로 구현, Review item 1 만 수정)

| 보상 항 | 대응하는 prompt 관찰·지표 | 근거(prompt 파일:줄) |
|---|---|---|
| approach_src / approach_rcv: 선형 대칭 진척 (RETREAT 제거, tanh 제거, ±3 v_ref 수치 클립) | 왕복 떨림이 수입을 얻음 (+0.252 / -0.126) | prompt_c1_repair.md:1040 |
| approach side_w, 가중 0.4, 파지 후 소멸 유지 | 옆 접근이 위 접근보다 더 받음, 차분형 | prompt_c1_repair.md:745-753 |
| 액션·속도 벌점 없음 유지 | 매끄러움은 제어기 몫 | prompt_c1_repair.md:702-710 |
| lift_src / lift_rcv 0.5 씩 유지 | 손별 들기 | prompt_c1_repair.md:754-756 |
| 나머지 19 항은 이전 응답과 동일 | "Keep everything else of the response unchanged" | prompt_c1_repair.md:1040 |

잔여 위험 (가설): side_w 가 위치의 함수라서 진척 장은 보존장이 아니다. 같은 경로를 되짚는 왕복은 정확히 0 이다. 하지만 옆에서 반경 방향으로 들어가고 위쪽으로 나오는 닫힌 원운동은 이론상 순수입을 남길 수 있다. Review 범위 밖이라 고치지 않았다. 학습에서 파지 전 원운동이 보이면 다음 라운드 후보로 삼는다.
