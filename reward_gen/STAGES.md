# 단계 보상 라이브러리 (manip_stages 프레임워크)

09.20 사용자 방향: **작업에 맞는 reward 구조를 축적하고 policy 도 축적한다.** 왼팔로 구조를 확정하고, 오른팔을 같은 구조로 만든 뒤
reward design 을 확정한다. 이 파일이 "어떤 보상이 무엇을 가르치고 무엇을 전제하는지"의 단일 목록이다.

## 규약

| 것 | 자리 |
|---|---|
| 과제 cfg·gym id | `source/openarm/openarm/agnostic/tasks/manip_stages/` → `open-short_{r,l}_<과제>-lstm` |
| 단계 보상 이력 | `reward_gen/<과제>_<팔>_<단계>/iter_NN/` (t2r 루프 그대로: prompt·response·compute_reward·observation·improvement) |
| 학습 로그 | `log/rl_games/open-short/{left,right}/<과제-하이픈>/<label>/` |
| 정책 | `our_source/policy_zoo/<과제>/<팔>/<단계>/<이름>.pth` + 같은 이름 `.json`(인터페이스·환경·보상·지표·계보) |
| 좌우 미러 | 원본 한 벌 + `scripts/reward_gen/mirror_reward.py` 로 생성. 손으로 고치지 않는다 |

단계 이름: `approach`(먼 출발 → 접근 래치 → 인벨롭 파지) · `manip`(리프트 → 목표 도달 → 유지). 정책은 **하나**이고 단계마다 보상을
갈아 끼우며 같은 가중치를 이어 받는다(사용자 결정 ③). 단계별 최종 `.pth` 가 그 단계의 정책이다.

## 확정된 단계 보상

### `approach` 계열 — 접근과 인벨롭을 가르친다
- 원본: `reward_gen/grasp_fj_rand/iter_00`(= `grasp_fj_stage/iter_08`) · 좌측 미러 `reward_gen/cup_pick_l_approach/iter_00`
- **측정된 효과**(우팔, 처음부터 학습): 먼 출발 접근 0.91(e500) · 인벨롭 래치 0.59(e750) · 검지 0.32 · 약지 0.25 · 닿은 손가락 3.1(e2000)
- **막히는 곳**: 리프트 0.07 · 들린 비율 0.02 · 성공 0.003. 쥔 채 손이 테이블로 내려간다
- **전제**: 없음(처음부터 된다)
- **구조 특징**: 닫기 사다리를 `g23`(래치 전·후 모두) 지급 · 1단계 체류 감쇠 `dwell1` 있음

### `manip` 계열 — 리프트와 목표 도달을 가르친다
- 원본: `reward_gen/grasp_fj_rand/iter_01` · 좌측 미러 `reward_gen/cup_pick_l_manip/iter_00`
- **측정된 효과**(우팔, 접근·파지 정책 위에서): 들린 비율 0.02 → 0.63 · 먼 출발 성공 0.85(e5000, 공차 0.019) · 컵 기울기 4.3° · 들 때 팔 속도 p99 1.1 rad/s
- **막히는 곳**: 공차 바닥 0.015 이후 후반 열화(성공 0.85 → 0.61, e8700). 마지막 가중치를 쓰지 않는다
- **전제**: **이미 감싸 쥐는 정책**. 처음부터 쓰면 실패한다 — 좌팔에서 1500 epoch 동안 검지·약지 접촉 0 · 인벨롭 0
- **구조 특징**: 닫기를 `g2`(래치 전)까지만 지급하고 값을 들기·목표로 옮김 · `dwell1` 제거

## 정책 창고

| 정책 | 단계 | 지표 | 비고 |
|---|---|---|---|
| `policy_zoo/cup_pick/left/approach/seed_fj_randL_i00b_e800.pth` | 접근만 | 접근 0.88 · 인벨롭 0 | 1단계 학습의 출발점 |
| `our_source/fj_t2r_init/fj_rand_i01_best_ep5000.pth` (s2r 동결, 이동 금지) | 전 과정(우팔) | 먼 출발 성공 0.85 · 공차 0.019 | sim2real 후보 |

## 새 과제를 추가할 때

1. `manip_stages/<과제>_cfg.py` 에 cfg 한 쌍(좌·우) — 기존 env 를 상속만 한다.
2. `config/__init__.py` 에 `("short_r", "<과제>")`, `("short_l", "<과제>")` 두 줄.
3. `t2r_fj_round.py TRACKS` 에 `<과제>_<팔>_<단계>` 네 줄.
4. 보상은 `approach` 계열을 복사해 시작한다(접근·파지는 물체가 바뀌어도 대개 같다). 바뀌는 것은 대개 `manip` 계열이다.
5. 결과를 이 파일에 한 항목으로 적는다 — 효과·막히는 곳·전제·구조 특징 네 가지.
