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

## M. iter_04 학습 결과 (t2r2_i04, warm from i03m ep3900, e4152 / 8 h) — measured facts

The environment now asks for the WHOLE cup at ADR 0 (target curriculum). ADR never rose, so every
episode asked for the whole cup.

| measurement (window means) | e1-100 | e400-1000 | e1000-2000 | e2000-3000 | e3000-4152 |
|---|---|---|---|---|---|
| task/src_wrap_count | 1.56 | 2.97 | 3.32 | 3.32 | 3.35 |
| diag/pose_src_wrap_ok_pour | 0.18 | 0.84 | 0.95 | 0.97 | 0.98 |
| diag/ep_amount_err_beads (short at the end, of about 10) | -9.0 | -5.7 | -4.0 | -3.6 | -3.5 |
| diag/ep_amount_on_target | 0.008 | 0.072 | 0.102 | 0.088 | 0.085 |
| peak tilt | 62 | 95 | 96 | 97 | 98 |
| diag/tilt_over_release_deg (tilt minus the release angle of the beads left) | -8.7 | -8.0 | -7.1 | -6.6 | -6.6 |
| reward/pour_rot (per step) | 2.5 | 4.5 | 5.0 | 5.4 | 5.5 |
| diag/ep_end_upright_held, task/episode_success, adr/progress | 0 | 0 | 0 | 0 | 0 |

The iter_04 grasp change worked: the source envelope came back from 1.5 to 3.35 fingers.

The whole-cup stage is not solved: the policy pours about 6.5 of 10 beads, peaks at 98 degrees, stays
6.6 degrees below the angle at which the remaining beads would flow, trickles until the episode ends,
and never reaches the +-1 bead window that opens the big finish.

Cause, read from the code: the rotation income uses `rot_prog = tilt_toward_rcv / (1.4 * premature_tilt_limit)`,
which saturates near 103 degrees. Tilting further than that pays nothing more, so the policy never
tilts far enough to empty the cup. The user's rule is that the reward must not prescribe angles; a
multiple of `premature_tilt_limit` is still an angle prescription.

## N. 이번 라운드의 보상 변경 (one principle, decided by the user 09-28): no angle prescription

Remove every angle target, cap and angle-scaled window from the reward. Replace them with the cup's
own orientation and with outcomes:

1. **Rotation income:** measure progress with the source cup's orientation toward the receiver on its
   natural scale (upright 0, horizontal 0.5, fully inverted 1, for example `(1 - cos(theta_toward)) / 2`
   with `theta_toward = ctx.src_tilt_toward_rcv` clamped at 0). No cap and no target angle. The rotation
   income must keep rising as the cup turns further, up to fully inverted. It stops ONLY through the
   outcome that already exists: once the remaining request is within the env tolerance (`want_rot`).
   Keep `lip_gate`, the aim floor factor and the grasp hold factor exactly as they are.
2. **Upright scores** (finish branch, receiver kept upright, pre-lift fade): use the cup's up-vector
   z component (`ctx.src_cup_up[:, 2]`, `ctx.rcv_cup_up[:, 2]`, 1 upright and 0 horizontal, clamp
   negative to 0) instead of tilt divided by a multiple of `premature_tilt_limit`.
3. **Premature charge:** keep it. The environment itself voids success when the grasped cup passes
   `premature_tilt_limit` with the lip away from the mouth, so the charge only announces an env rule.

Keep every other term, weight and gate of iter_04 exactly as it is: the finish threshold at +-1 bead
(12 per step real finish, <= 1.5 otherwise), the grasp hold factor and the fingertip-only charge, the
spill charge of 40 per bead, converge, aim floor, grip-keep, and the ADR-gated tremor and flip charges.

State in your answer: the rotation income at 60, 100, 140 and 180 degrees toward the receiver with
the lip over the mouth and the grasp enveloped, and confirm it is still rising between 100 and 140.

## O. iter_05 학습 결과 (t2r2_i05, warm from i04 ep4450, e1350 / 2.4 h, stopped by the user) — measured facts

| measurement (20-epoch means) | e130 | e458 | e723 | e964 | e1226 |
|---|---|---|---|---|---|
| peak tilt (deg) | 97.3 | 96.7 | 98.7 | 98.6 | 99.2 |
| diag/tilt_over_release_deg | -4.5 | -6.6 | -8.0 | -6.4 | -6.9 |
| beads short at episode end | 3.3 | 3.1 | 2.9 | 2.8 | 2.6 |
| diag/ep_amount_on_target | 0.11 | 0.17 | 0.17 | 0.17 | 0.23 |
| src / rcv grasped | 0.68/0.65 | 0.72/0.72 | 0.95/0.95 | 0.77/0.76 | 0.93/0.93 |
| bead/spill | 0.004 | 0.017 | 0.014 | 0.008 | 0.021 |
| upright held, success, adr | 0 | 0 | 0 | 0 | 0 |

Removing the rotation cap did not raise the peak tilt: it stayed at 96-99 degrees for 1,100 epochs.

Trajectory comparison (64 deterministic episodes each, source = right arm; old lineage i18 and i24 used the
same fabric controller settings, robot asset and start pose as now):

| run | peak tilt (median) | j5 / j7 change from 20 deg to peak (rad) | pour direction of the cup mouth (azimuth) | source y minus receiver y at peak | receiver moved |
|---|---|---|---|---|---|
| i18 ep2500 | 155 | -1.54 / +0.01 | 166 deg | -1.4 cm | 4.1 cm |
| i24 ep1150 | 122 | -1.23 / +0.51 | 155 deg | -3.4 cm | 5.4 cm |
| t2r2_i01 ep1000 | 94 | -1.00 / -0.11 | 142 deg | -8.5 cm | 9.3 cm |
| t2r2_i03m ep3800 | 96 | -0.37 / +0.66 | -142 deg | +8.6 cm | 13.1 cm |
| t2r2_i05 ep1350 | 98 | -0.45 / +0.34 | -166 deg | -0.7 cm | 12.6 cm |

- The deep pours rotate the cup with the forearm roll j5 and tip the mouth toward +y (the receiver side).
  The stuck runs tip the mouth toward -y and rotate with the wrist j7, which ends 11 degrees from its
  joint limit at about 96-99 degrees of cup tilt.
- In the stuck runs the receiver palm was pulled back toward the body (receiver palm delta x at the
  box edge -0.15 m); in the deep pours it stayed forward (delta x -0.04 to +0.15 m, y -0.12 to +0.04 m).
- Palm command vs actual palm after lift: the command speed median sat at the per-step cap
  (0.72 m/s, about 270 deg/s) while the palm moved at 0.02-0.07 m/s; 42-71 % of source palm outputs were
  beyond |0.95| (pinned at the box edge).

## P. Environment changes for this round (user decisions 09-28, code commit 9d300ef4) — facts

1. **Palm command = velocity (increment mode).** The filtered palm action is now read as the palm TARGET
   VELOCITY (fraction of a max speed: source 0.6 m/s and 180 deg/s, receiver 0.4 m/s and 120 deg/s). The
   target accumulates inside the palm box and is tied to the measured palm (source 6 cm / 30 deg, receiver
   5 cm / 15 deg). Action +-1 means "move at max speed", not "go to the box edge". Action 0 means "hold".
   `ctx.actions` / `ctx.prev_actions` are therefore palm velocity commands (plus the 3 grip channels).
2. **Receiver position limit.** The receiver palm may move only -0.05..+0.15 m in x and -0.12..+0.04 m in y
   from its start pose (was -0.15..+0.15 and -0.40..+0.10).
3. **Smoothness is handled by the controller, not by the reward (user principle 09-28):** "with fabrics,
   remove the penalty terms and manage it in the control part; that is why we use fabrics."
4. The run starts FRESH (no checkpoint), LSTM-MLP network as before.

## Q. 이번 라운드의 보상 변경 (decided by the user 09-28)

Remove the four action/motion penalty terms from the iter_05 reward: `action_rate`, `action_flip`,
`arm_tremor`, `action_reg_adr`. Do not add any other term that reads `ctx.actions`, `ctx.prev_actions`
or arm joint velocities as a penalty.

Keep every other term, weight and gate of iter_05 exactly as it is (uncapped natural-scale rotation
income, the up-vector finish score with the full-range part `1 - acos(up_z)/pi`, finish 12 / 1.5,
premature charge, spill 40, converge, aim floor, grip-keep, fingertip-only charge, receiver keep, cup
clash, hand foreign, pour direction).

State in your answer: the list of terms that remain, and the per-step positive income of a grasped,
lifted, aimed but not pouring state (it must stay below one bead of pour income, 30).

## R. iter_06 학습 결과 (t2r2_i06, fresh, incremental palm command, e821 / 1.7 h) — measured facts

| measurement (20-epoch means) | e254 | e359 | e586 | e821 |
|---|---|---|---|---|
| src / rcv grasped | 0.52 / 0.81 | 0.79 / 0.84 | 0.81 / 0.85 | 0.79 / 0.84 |
| src wrap fingers | 2.1 | 3.1 | 3.6 | 2.7 |
| src / rcv cup lift (m) | 0.003 / 0.113 | 0.004 / 0.104 | 0.005 / 0.087 | 0.101 / 0.115 |
| fabric src / rcv palm error (m) | 0.056 / 0.028 | 0.057 / 0.026 | 0.057 / 0.029 | 0.057 / 0.021 |

- The source grasp reached 0.8 at epoch 407; the previous LSTM run with absolute palm commands (t2r2_i03m)
  reached it at epoch 199 and had lifted the source cup 20 cm by epoch 250. i06 lifted the source cup only
  from about epoch 800.
- Trajectory at epoch 600 (64 deterministic episodes): the source hand held the cup (grasped 94-98 %, contact on
  the middle and distal links, palm contact 0) but commanded the palm DOWN (mean z action -0.73, 43 % at -1).
  The palm target sat 5 cm below the palm and the cup was pressed onto the table (max lift median 1.5 cm).
- With incremental commands, action 0 means "stay where you are", so exploration noise integrates into a
  random walk of the palm target instead of staying around the start pose (which is next to the cup). The
  reward has no term that tells the hand WHERE to be relative to the cup, and the lift income is paid only
  when BOTH hands hold their cups (`both_g`), at most 0.25 per step.
- New context fields this round: `ctx.src_palm_lin_vel`, `ctx.rcv_palm_lin_vel` (N,3) palm linear velocity
  [m/s], env-local. The velocity component toward a point is positive only while moving closer, so it pays
  nothing for standing still (see lesson on dense state income -> parking).

## S. 이번 라운드의 보상 변경 (decided by the user 09-28)

1. **Approach as progress toward the cup ORIGIN from the side (difference form), per hand, before that hand
   grasps its cup.** Replace the state-proportional `approach_both` with a progress term: the palm velocity
   component toward the cup origin (`ctx.*_cup_pos`), i.e. `v_palm . (cup_pos - palm_pos)/|cup_pos - palm_pos|`,
   clipped so that retreating costs a little and approaching pays; zero when the palm is still. The approach
   must come from the SIDE at the height of the cup origin (user rule 09-27: the real perception gives only the
   object origin): a palm the same distance away above the mouth must earn less than a palm beside the origin
   (for example weight the progress by how horizontal the approach direction is, or measure progress toward
   the origin on the horizontal plane plus a height-match factor). Use only cup_radius multiples, no asset
   dimensions. It fades to 0 once that hand has grasped.
2. **Lift per hand.** Pay each cup's lift when THAT hand holds it (source: `src_grasped` with `h_src`;
   receiver: `rcv_grasped` with `h_rcv`), not only when both hands hold. Weight 0.5 per cup (max 1.0 per step
   in total), keep the upright fade `low_*`. The sum of state incomes must stay below one bead of pour income (30).
3. Everything else stays exactly as iter_06 (no action/motion penalties; smoothness is the controller's job).

State in your answer: (a) the approach income per step for a palm moving at 0.3 m/s toward the cup origin from
the side, the same palm moving at 0.3 m/s toward the origin from above the mouth, and a still palm next to the
cup; (b) the per-step lift income of a source cup lifted 2*cup_radius with only the source hand grasping.
