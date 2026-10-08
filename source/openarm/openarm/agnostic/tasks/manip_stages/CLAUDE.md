# manip_stages — 단계별 보상 커리큘럼 프레임워크 (정책 1개 · 보상 단계별)

> 과제 목록·단계 목록·정책 창고: [`reward_gen/STAGES.md`](../../../../../../reward_gen/STAGES.md)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · env 는 [`../grasp_fj_t2r`](../grasp_fj_t2r/CLAUDE.md) 상속 — `CupPickEnv` 는 **보상 입력만** 더한다(09.22)

## 왜 따로 있나 (09.20 사용자 결정)
- "기존 grasping 정책 폴더와 pth 는 냅두고 새 디렉토리" — `grasp_fj_t2r` 의 leaf·로그·체크포인트는 **sim2real 대상이라 동결**한다.
  이 패키지는 cfg 를 상속해 gym id 만 새로 등록한다(`open-short_{r,l}_cup_pick-lstm`) → 로그 `log/rl_games/open-short/<side>/cup-pick/`.
- "③ 단일 정책 + 보상 2트랙" — 정책은 **하나**다. 보상만 단계로 갈아 끼우며 같은 가중치를 이어 받는다.
  단계별 최종 .pth 가 그 단계의 정책이다. 정책을 둘로 나누지 않으므로 **인계 분포·전환 신호·LSTM 예열 문제가 없다**(s2r 안전).

## 두 보상 (근거: 09.20 iter 분석)
| 트랙 | 보상 출처 | 실측 |
|---|---|---|
| `reward_gen/cup_pick_l_approach` | 우 `grasp_fj_rand/iter_00`(= `grasp_fj_stage/iter_08`) 좌측 미러 | 우팔에서 **처음부터** 접근 0.91(e500) · 인벨롭 0.59(e750) · 검지 0.32 · 약지 0.25. 리프트는 0.07 에서 막힘 |
| `reward_gen/cup_pick_l_manip` | 우 `grasp_fj_rand/iter_01` 좌측 미러 | 우팔에서 파지하는 정책 위에 얹어 먼 출발 성공 0.85(e5000). **처음부터는 안 된다** — 좌팔에서 1500 epoch 동안 검지·약지 0 · 인벨롭 0 |

구조 차이: 1단계는 닫기 사다리를 `g23`(인벨롭 전·후 모두) 지급하고 1단계 체류 감쇠(`dwell1`)가 있다. 2단계는 닫기를 `g2`(래치 전)까지만
지급하고 값을 들기·목표로 옮기며 체류 감쇠를 뺐다. 그래서 2단계는 "이미 감싸 쥐는 정책" 위에서만 성립한다.

## 순서
1. 1단계 런(`cp_l_a00`) — 좌팔 접근 0.9 가중치에서 이어받아 인벨롭을 세운다. 목표 인벨롭 ≥ 0.5 가 200 epoch 유지.
2. 그 가중치로 2단계 런(`cp_l_m00`) — 리프트·목표. 목표는 우팔과 같은 먼 출발 성공 0.8 이상.
3. 라운드 안에서 보상을 더 고칠 일이 생기면 각 트랙의 t2r 루프(관찰·피드백 → 새 iter)로 돈다.

## 물체 모양과 무관한 보상 입력 (09.22 사용자 결정)
- `cup_pick` gym id 의 entry = `cup_pick_env.CupPickEnv`: 부모 ctx + `palm/link_surface_{dist,dir,normal}` (`object_context.ObjectRewardContext`).
  표면 = 뱅크 종별 충돌 형상에서 512 점(`object_surface`, 강체 actor 프레임 = cup_pos/cup_quat, 오프셋 없음) · 질의 `surface_query`.
  관측·액션·종료·리셋·성공은 부모 그대로(계약 테스트) → 기존 체크포인트 그대로 로드. 로그 `task/palm_surface_dist` 등.
- 생성·ingest 는 `scripts/reward_gen/t2r_ms.py`(t2r_fj.py 를 import, 스텁·검증기만 교체). t2r_fj.py 트랙은 무영향.
- 테스트: `PYTHONPATH=source/openarm /usr/bin/python3 -m pytest source/openarm/openarm/agnostic/tasks/manip_stages/tests -q`(pxr 필요).

## cup_grasp — 첫마디 입력 + critic 접촉 관측 (09.27 사용자 결정 "1,3을 같이")
- 왜: cup_grasp iter_00·01 로 두 팔 모두 무작위 초기에서 완주했지만 영상은 4번 관절 과굽힘 손끝 파지였다. m18 재생에서 네 손가락이
  쥔 스텝의 가운데 마디 접촉 0.00·손끝 패드 0.90~1.00. 첫마디(`_2`)는 센서·ctx 점이 없었고 actor·critic 모두 접촉을 몰랐다.
- gym id `open-short_{r,l}_cup_grasp-lstm` → 로그 `log/rl_games/open-short/<side>/cup-grasp/`. entry `cup_grasp_env.CupGraspEnv(CupPickEnv)`.
  · 보상 입력 `grasp_context.GraspRewardContext`: `prox_pos/prox_cup_force/prox_surface_{dist,dir,normal}` (첫마디 5).
  · critic 전용 +21 = tanh(접촉력/5 N) (첫마디 5 + 마디·손끝 15 + 손바닥 1). critic 157 → 178. **actor obs 133·action 26 불변**(s2r 계약 그대로).
  · 로그 `contact/prox_touching`·`contact/prox_<finger>`·`contact/phalanx_digits`·`task/prox_surface_dist_min_mean`.
- cup_pick gym id·체크포인트는 그대로(critic 차원이 달라 섞지 않는다). 4번 관절 범위 축소는 s2r 불리로 하지 않았다.
- 생성·ingest: `t2r_ms.py render --context grasp`, ingest·reflect 는 iter meta.json 의 context 로 자동 선택. 검증기 `grasp_validator`.
- 부팅 스모크(09.27, 우 64 env 6 epoch): 첫마디 센서 5 · state_space 178 · 새 태그 기록 확인.

## cup_grasp 손 액션 0 = 편 손 (09.28 사용자 결정 "1,2 모두 필요")
- 왜: 부모 사상 `raw = lo + ½(a+1)(hi−lo)` 는 액션 0 이 관절 범위 가운데다. 리셋(편 손) 자세는 대부분 하한 = 액션 −1 이라
  무작위 정책이 굽힘 ~0.5 로 시작하고, 편 손을 유지하려면 손가락 평균을 액션 한계에 붙여야 했다(편 손 자세 조건 전 라운드 0.01,
  왼팔 bounds_loss 1.5 → 681 붕괴 iter_06 e800·iter_07 e650).
- 무엇: `cfg.hand_action_zero_open=True` 면 `hand_action.zero_open_raw` — 관절마다 a=0 → 리셋 자세, a≥0 은 리셋→상한, a<0 은 리셋→하한.
  범위·EMA·관측·차원 불변. **기본 False**(이전 런 재생 호환). 새 런은 launcher EXTRA 에 `env.hand_action_zero_open=true`.
- ★액션 의미가 바뀐다 — 이 플래그로 학습한 정책을 실기에 올리려면 실기 제어기(`robot_control/`·`sim2real/`, 사용자 영역)도 같은 사상을 써야 한다.
- 스모크(09.28, 우 64 env 6 epoch): env.yaml 기록 확인, 초기 굽힘 0.41~0.49 → 0.30~0.37(양수 노이즈가 여전히 굽힌다).

## cup_grasp 실기 지연·긴 유지 (09.28 사용자 "실기 지연도 학습에서 배워둬야", SIME2REAL 인계 cg_l_i01)
- 근거: `sim2real/docs/HANDOFF_cg_l_i01_sim_vs_real_2026-09-28.md` — 실기에서 들고 8 s 부터 스스로 놓음. sim 에피소드는 목표 5개 뒤
  6.5~7.4 s 에 끝나 긴 유지를 본 적 없음. 실기 지연: 팔 로봇 자체 150~200 ms(sim 40~50), 손 300 ms(sim 70~170), 물체 FP++ 0.4 s + 끊김.
- `real_delays.LagQueue`: env 별 **고정** 지연(에피소드마다 재추첨)·구간별(팔 7/손 19)·물체 끊김(마지막 값 유지). 부모 큐와 같은 인터페이스라
  `CupGraspEnv._install_real_delays` 가 큐 객체만 바꿔 끼운다. cfg `real_arm_lag_steps`·`real_hand_lag_steps`·`real_obj_lag_steps`·
  `real_obj_dropout_p`·`real_obj_dropout_steps`(스텝 = 16.7 ms). 팔·손은 **추가** 지연, 배포 pd 속도 제한분(~190 ms)은 넣지 않는다.
- `goal_hold_after` > 0: 그만큼 성공한 뒤 목표 고정(마지막 목표 유지). 조기 종료를 막으려면 `goal_max` 를 키운다.
- 모두 기본값이면 부모와 같다(이전 런 재생 호환). 실기 배포 쪽 규칙(목표 고정 등)과 맞춰 쓴다.

## 하지 않는 것
- env/cfg 값 수정(과제가 s2r leaf 와 달라진다 — 테스트가 막는다) · 기존 트랙 로그·체크포인트 접촉 · 학습 중 같은 GPU 에서 play.
