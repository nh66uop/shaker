# pour_bi 보상 설계 브리프 (09.26 종합판 — i16 ~ i28 전체)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.
**이 문서는 매 라운드 생성기 노트(notes_iterN.md) 맨 앞에 통째로 들어간다.** 라운드 노트는 이 브리프 뒤에 직전 런 진단만 덧붙인다.

## 0. 왜 이 문서가 필요한가 (사용자 지시 09.26)

- "보상설계 에이전트가 보상을 설계할 때 T2R 과 기존 학습 진행 결과를 종합적으로 보고 만들고 보고해."
- 지금까지 생성기는 직전 보상 코드 + 직전 런 곡선 + 그 라운드 노트만 봤다. iter_26~28 은 모두 직전 런만 고쳤고,
  25 에서 붓기를 가능하게 했던 조건(큰 회전 수입)을 26 에서 지운 뒤 그 영향을 다시 보지 않았다.
  결과: 25 는 94° 를 넘겨 부었는데 26r·27·28 은 55° 근처에서 멈춰 한 알도 못 부었다.

## 1. 과제와 고정 조건 (사용자 결정)

- 양팔로 컵을 잡아(5지 인벨롭 유지) actor obs 의 **요청량(±1 알)** 만 붓고, 소스 컵을 **세운 뒤 틸팅 전 자세로 복귀**.
  (USER_OBJECTIVES_20260925.md, taskspecs/drafts/pour_bi.yaml user_decisions: Q2 요청량+세우기, Q3 자세 복귀, Q5 K=5 로깅 먼저, M=0)
- 성공은 env 가 판정한다(pour_fabric_env.py:662-672). 보상은 판정하지 않는다.
- 원칙: 보상은 보편·단순(항 15개 내외), 각도 처방·자산 고유 치수 금지(창·거리는 ctx.cup_radius 배수),
  속도·제어 벌점을 풀어 효율을 얻지 않는다, 작업은 y≈0 중심선, 관측은 sim2real 가능한 것만.

## 2. 측정으로 확정된 물리 사실

| # | 사실 | 근거 |
|---|---|---|
| P1 | 첫 비드는 총 기울기 약 90 + 8·(1 − fill·in_source)° 이상에서 나온다(fill 0.55 에서 93.6°). | 24 mm 비드 프로브(09.23), notes_iter28 |
| P2 | premature 벌점은 총 기울기가 premature_tilt_limit(p50 73.6°)의 0.7 배, 즉 **약 52~58°** 부터 켜지고, spout 가 입구에서 2r(8.1 cm) 밖이면 env 래치가 성공을 무효로 만든다. | iter_25~28 PRE_LO 0.7, env cfg |
| P3 | env 는 컵이 20° 미만으로 서 있을 때만 들기 기준(파지 자세·힘)을 기록한다. 기울인 채 들면 기록이 안 된다. | pour_fabric_env._grasp_frame |
| P4 | 들기 순간 파지 기준 유지(iter_24 결합)는 효과가 있다: i24 드리프트 1.08~1.56 cm, 힘비 2.4×. | notes_round25 |
| P5 | 붓기 자세(주둥이가 입구 1 cm 위, 방출각)는 i27 plateau 상태에서 손목 델타 박스 안으로 99~100 % 도달 가능하다. 막은 것은 기구학이 아니라 보상이다. | notes_iter28 WHY 1 |
| P6 | ADR 은 순간 성공률 0.30 에서 오른다. 성공 0 이면 ADR 0 이고, ADR 에 묶인 떨림·액션 벌점은 계속 꺼져 있다. | cfg adr_trigger_threshold, i27 tremor 0 |

## 3. 라운드별 결과와 교훈

| 런 | 보상 요지 | 결과 | 교훈 |
|---|---|---|---|
| i16 (옛 목표) | 기울기 연동 높이 허용 | ep5600 ADR30 성공 92 % | 붓기는 가능하다 |
| i24 (옛 목표) | 들기 기준 파지 유지 | ADR30 성공 0.72, 최고 116° | 파지 결합 유지 |
| i25 | 새 목표, TILT_W 3 | 43° 정체, 붓기 0 | 회전 수입이 약하면 안 돈다 |
| i25b | TILT_W 15 웜 | 28~45° 그대로 | 좁아진 탐색은 웜으로 안 풀린다 |
| i25f | TILT_W 15 fresh | 93°, ep1147 첫 붓기, 5.5 중 3.3 붓고 기울인 채 버팀, 세우기 0 | 큰 연속 회전 수입(스텝당 ~+6)이 94° 를 넘게 한다 |
| i25j | + 세우기 인계 | ep1500 128°, 흘림 에피소드 98 %·6.9 알, 세우기 0 | 흘린 비드가 부은 양에 안 들어가 want 가 안 꺼짐 → 과회전. 기울인 채 버티기(10.6~14.5) > 세우기(≤5) |
| i26 | 회전 수입을 첫 비드 이탈 시 종료, need=min(요청−부음, 잔량) | e47 에서 교체 | 판단 전 종료 |
| i26r | fresh, 회전 수입을 입구 0.5r 안 속도형으로 | 55° → 38°, 붓기 0 | 게이트가 방문 상태에서 1e-17 이면 수입 0, 벌점만 남는다(−76~−125/에피소드) |
| i27 | 속도형 수입을 far=0 으로, 리시버 앵커 이동 | 조준은 좋아짐(spout 8.5 cm), 최고 56°(리시버 방향 47°) 벽, 붓기 0, 들기 기준 회피 28→100 %, 리시버 5지 0.26→0.13, flip 0.45 | premature 벽 앞에서 멈춤. 기준 회피가 자세 보상 2.2/스텝을 벌게 했다 |
| i28 웜(ep800) | 반경 2r, 기준 회피 차단, 5지 /F | e56 기준 기록 0.04 → 종료 | 없애려는 습관을 가진 체크포인트에서 웜하지 않는다 |
| i28f | 같은 보상 fresh | 학습 중(09.26 13:33) | — |

## 4. 누적 교훈 (새 보상은 이 목록에 대해 하나씩 답해야 한다)

- L1 **회전 구동력의 크기**: 25 는 연속 수입(스텝당 ~+6)으로 94° 를 넘었고, 26~28 의 속도형(상한 +1/스텝, TILT_W 5)은 55° 벽을 못 넘었다.
  새 회전 항은 최신 trace 의 plateau 상태에서 **"10°·20°·40° 더 기울일 때 스텝당 손익(회전 수입 − premature − 흘림 위험)"** 을 수치로 제시한다.
- L2 **흘림 회계**: 흘린 비드로 요청이 도달 불가가 되면 want 가 꺼지고 마무리가 열려야 한다(iter_26 need=min 유지).
- L3 **게이트 값 점검**: 새 게이트·폭은 최신 trace 방문 상태에서 분포(p10/p50/p90)를 보이고, 0 에 붙어 있지 않음을 확인한다.
- L4 **상태 수입 < 결과 수입**: 기울인 채 버티기, 서 있기, 들기 전 자세 같은 상태 수입이 붓기·세우기·성공보다 작아야 한다. 숫자로 비교한다.
- L5 **웜스타트**: 새 보상이 없애려는 습관이 체크포인트에 있으면 웜하지 않는다(i25b, i28). 웜 지점은 trace 로 확인한다.
- L6 **종료 시점**: fresh 는 첫 붓기까지 ~e1150(i25f). 기대 epoch 전에는 붕괴가 아니면 끊지 않는다.
- L7 **떨림**: 성공 0 동안 ADR 이 0 이라 떨림 벌점이 꺼져 있다. [사용자 결정 대기: ADR 무관하게 켤지]
- L8 **보상 해킹**: env 기록 조건(P3)을 피해 더 버는 경로가 없는지 확인한다.

## 5. 라운드 보고 형식 (생성 후 사용자에게)

1. 이번에 본 근거: 이 브리프의 어떤 교훈(L#)·어떤 trace·TB 구간을 봤는가.
2. 바꾼 것 / 유지한 것과 이유.
3. 최신 trace 상태에서의 수치: L1 손익표, L3 게이트 분포, L4 상태 수입 vs 결과 수입.
4. 위험과 게이트: 몇 epoch 에 무엇을 보고, 실패하면 무엇을 하는가.
5. 웜/fresh 선택 근거(L5).

# 새 계보 pour_bi2 · iter_00 노트 (2026-09-26)

## A. 왜 새로 시작하는가

`reward_gen/pour_bi/iter_00 ~ iter_28` 은 여기서 끝냈다. 마지막 세 라운드(i26r, i27, i28)가 연속으로 한 알도
못 부었다. 사용자 결정: iter_00 부터 다시 설계한다. 학습 기록과 교훈은
`~/rl_ws/our_source/pour_bi_runs_0926/` (INDEX.md, ROUNDS.md) 와 위 DESIGN_BRIEF 에 있다.
같은 폴더의 `ROUNDS_pour_bi_i00_i28.md` 가 라운드별 표와 반복 실패 원인 10가지다.

## B. 사용자 결정 (2026-09-26)

1. 목표 단계화: **붓기를 먼저 뚫는다.** 세우기와 자세 복귀는 다음 라운드에 올린다.
   env 의 성공 판정은 그대로다(요청량 ±1 + 소스 20° 이하 + 양손 파지 유지). 보상만 단계화한다.
2. 94도의 벽을 **양쪽에서** 뚫는다. 조급한 기울기 청구 시작점을 env 한계각 가까이로 올리고,
   동시에 55~94도 구간에 연속 회전 수입을 준다.
3. 학습은 server GPU0. 2026-09-27 08:00 까지 라운드 자동 진행(사용자 승인).

## C. 이번 iter_00 이 반드시 지켜야 하는 것 (REQUIREMENTS for the generator)

The reward must satisfy every item below. State the numbers you used in your answer.

R1. **Staged goal.** Beads leaving the source and landing in the receiver is the headline income.
    Uprighting and returning to the pre-tilt posture get only a small terminal bonus in this round.
    Do not make the finish branch pay more per step than pouring does.

R2. **Rotation drive that reaches the states the policy actually visits.** The last three rounds paid
    exactly 0 for rotation because the payout gate evaluated to 1e-17 or 1e-7 in every visited state.
    Gate the rotation income on the pouring lip being within `2 * ctx.cup_radius` (8.1 cm, the same
    radius the environment uses for its premature latch), never on a tighter precision window, and on
    the tilt being toward the receiver. At a state with a lip offset of 9 to 14 cm and a tilt of 55
    degrees the rotation income must still be clearly positive per step, not a rounding error.

R3. **Move the premature charge to the wall, not in front of it.** The environment latches a failure
    only when the grasped source cup passes `ctx.premature_tilt_limit` WHILE the lip is still more
    than 8.1 cm (xy) from the receiver mouth. Charge for premature tilt only near that limit (at or
    above 0.95 of it) and only while the lip is outside that radius. Inside the radius there is no
    tilt charge at all: the first bead does not leave until about 90 degrees, so charging from 55
    degrees makes the whole 55 to 94 degree band a pure loss.

R4. **State income below outcome income.** The sum of every per-step income that a policy can collect
    by just holding a posture (approach, aim, lift, posture, keep, home) must be smaller than the
    income for pouring one bead. Four rounds were lost to parking on state income.

R5. **Spill accounting.** Use `need = min(target - poured, beads still in the source)`. When `need`
    reaches zero, the rotation income stops and the finish branch opens. Beads that spilled must not
    keep the request open, or the policy over-rotates (i25j reached 128 degrees and spilled in 98
    percent of episodes).

R6. **Keep the two guards that were bought with previous rounds.**
    (a) The iter_24 grip-keep principle: pay for similarity to the grasp recorded at lift (relative
    hand-cup pose, finger force distribution, total force), never for an absolute force target.
    (b) The environment records that reference only when the cup is below 20 degrees at lift-off.
    A policy that tilts before lifting must not be able to earn the pre-lift posture income instead.

R7. **Exploration.** Charges that suppress exploration (action flip, hand flip, palm saturation,
    carry-phase rate) stay ramped by `ctx.adr_progress` as in iter_22 and later, because applying
    them at full weight from step 0 of a fresh run taught the policy not to touch the cups (i21, e26).
    A small always-on action-rate charge at the iter_18 level is fine.

R8. **No hardcoded angles or asset dimensions.** Windows and distances are multiples of
    `ctx.cup_radius`; angles come from `ctx.premature_tilt_limit` and the bead release relation.
    A reward that hardcodes 72 or 86 degrees broke when the bead size changed.

R9. Keep the term count near 15 to 22. Simple and general beats clever.

## D. 이 라운드에 답해야 하는 숫자 (answer these in the response text)

1. At tilt 55 degrees with the lip 10 cm from the mouth, what is the per-step rotation income and the
   per-step premature charge?
2. What is the net per-step profit of tilting 10, 20 and 40 degrees further from that state?
3. What is the total per-step state income (R4) and what does one poured bead pay?


---

## E. iter_01 학습 결과 (t2r2_i01, e1350 / 9.5 h, fresh, server GPU0) — measured facts and the user's video verdict

The converge change worked: the cups came together, the rotation income woke up, and the policy
poured. Then it regressed. The user watched three replays and gave the verdict below; the user's
judgement is the reference.

**User video verdict (09-27):** "In all three videos the aim is off and the beads spill sideways.
Beads come out one at a time and every one of them goes beside the receiver. The right arm is steady
but the left arm shakes a lot."

| measurement | e700 | e900 | e1100 | e1300 |
|---|---|---|---|---|
| task/aim_dist (lip to mouth, xy, while pouring) | 0.129 m | 0.123 m | 0.124 m | 0.129 m |
| task/pour_lip_dist (episode mean) | 0.071 | 0.065 | 0.063 | 0.074 |
| diag/ep_peak_tilt_deg_tgt_mid | 96.7 | 96.1 | 95.2 | 98.6 |
| bead/in_target | 0.115 | 0.108 | 0.140 | 0.030 |
| bead/spill | 0.115 | 0.088 | 0.073 | 0.265 |
| task/premature_tilt_rate | 0.082 | 0.082 | 0.108 | 0.329 |
| diag/ep_amount_on_target | 0.189 | 0.281 | 0.202 | 0.019 |
| diag/ep_amount_err_beads | -3.04 | -2.25 | -2.68 | -4.63 |
| diag/ctrl_arm_qd_rms_rcv (receiver arm) | 0.465 | 0.454 | 0.467 | 0.444 |
| diag/ctrl_arm_qd_rms_src (source arm) | 0.419 | 1.514 | 0.378 | 0.365 |
| diag/ctrl_flip_rate | 0.490 | 0.486 | 0.489 | 0.489 |
| reward/action_reg_adr (paid) | 0.000 | 0.000 | 0.000 | 0.000 |
| adr/progress | 0.000 | 0.000 | 0.000 | 0.000 |
| diag/ep_end_upright_held | 0.000 | 0.000 | 0.000 | 0.004 |
| diag/pose_meet_y_pour | 0.024 | 0.029 | 0.047 | 0.049 |

Two causes, both confirmed by the numbers:

1. **The rotation income is paid while the lip is NOT over the mouth.** During the pouring phase the
   lip is 0.123 to 0.129 m from the receiver mouth in xy. The cup radius is 0.0405 m and the env
   latch radius is 0.081 m, so the beads leave the lip about three cup radii beside the opening.
   `pour_rot` uses `lip_gate = 1 - tanh(max(lip_d - 2r, 0) / 2r)`, which still pays about 0.23 of
   full value there, and nothing in the term asks for the lip to be ABOVE the opening. Tilting
   without aiming is therefore profitable.

2. **The tremor and action-flip charges were never applied.** They are multiplied by
   `ctx.adr_progress`, which stayed 0.000 for the whole run because ADR only rises after the success
   rate passes 0.30. The action flip rate sat at 0.49 (half of all steps reverse sign) and the
   receiver arm ran 1.2 to 1.3 times shakier than the source arm.

Unchanged and working, do not touch: grasp 0.87 both hands, wrap 3.3/4.4 fingers, grip reference
recorded (`diag/grip_src_ref_valid` 0.89), converge_cups 1.0 per step, premature charge at the wall.

## F. 이번 라운드의 변경 (two changes, decided by the user 09-27)

**Change 1 — aiming must RAISE the rotation income, and must never switch it off.**

Read this constraint before writing the term. In the old lineage, iter_26 and iter_27 multiplied the
rotation income by a tight precision window (`fine = exp(-(offset / (0.5 * cup_radius))**2)`). In the
states the policy actually visited that factor evaluated to 1e-17 and 1e-7, so rotation was never
paid, the cup stopped at 38 to 56 degrees and three rounds in a row poured nothing. Do not repeat
that construction.

The aim factor must be a bonus on top of a floor, not a gate:

    aim_quality in [0, 1]   built from the pouring lip relative to the receiver mouth:
        horizontal offset measured in units of ctx.cup_radius (about 1 cup radius = full,
        falling to 0 at the env latch radius 2 * cup_radius), combined with a low, controlled
        drop height above the mouth
    rotation income = W * rot_prog * lip_gate * (FLOOR + (1 - FLOOR) * aim_quality)   with FLOOR about 0.3

State these numbers in your answer and make them true:
  (a) At the state this policy actually occupied while pouring (lip offset 3 * cup_radius = 0.123 m,
      tilt 75 degrees, lip 0.03 m above the mouth) the rotation income must still be at least
      1.5 per step and at least 30 percent of its value at perfect aim.
  (b) At perfect aim (offset 0, low drop) the rotation income must be clearly larger than at
      3 * cup_radius, so that aiming is what pays.
  (c) The income must rise monotonically as the offset shrinks at a fixed tilt.

Put the teeth on the OUTCOME side instead of the gate: a bead that leaves the source and lands
outside the receiver is already charged by the spill term. Raise that charge so that pouring while
badly aimed loses money over the episode, while the rotation drive itself stays alive. Keep
`lip_gate`, the rotation cap, the premature charge at the wall, `converge_cups` and every other term
as they are.

**Change 2 — charge tremor and action reversal regardless of `ctx.adr_progress`.** These charges
were multiplied by `ctx.adr_progress`, which stayed 0.000 for the whole run, so they were never
applied: the action flip rate sat at 0.49 and the receiver arm ran 1.2 to 1.3 times shakier than the
source arm. Make them alive from step 0, small, and weighted more on the receiver arm than on the
source arm. Keep the total of these charges well under the approach and posture incomes: in iter_21
of the old lineage the same charges at full weight taught the policy not to touch the cups at all
and the run died at epoch 26. State the per-step size of these charges at the measured flip rate
0.49 and receiver joint-speed RMS 0.46 in your answer.

Constraints that still hold: the sum of per-step income a policy can collect by holding a posture
must stay below the income for pouring one bead (30); every window and distance is a multiple of
`ctx.cup_radius`; every angle is a fraction of `ctx.premature_tilt_limit`.
