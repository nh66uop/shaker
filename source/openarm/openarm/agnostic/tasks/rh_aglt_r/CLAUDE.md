# rh_aglt_r / rh_aglt_l — RH56F1 한 팔 먼 출발 접근·잡기·들기·이송 (09.29 신설)

> 상위: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) · 보상 설계 근거: [`reward_gen/rh_aglt_r/DESIGN_BRIEF.md`](../../../../../../reward_gen/rh_aglt_r/DESIGN_BRIEF.md)
> 이 파일이 좌우 두 패키지의 규칙 정본이다. `rh_aglt_l/` 은 다른 점만 `profile.py`·cfg 주석에 둔다.

## 정체성 (사용자 결정 09.29)
- env 부터 디렉토리까지 새로 만든다 — `tasks/rh_aglt_{r,l}`. 다른 과제 패키지(grasp_fj·grasp_fj_rh·manip_stages·pour_fabric_mimic)는
  import 하지 않는다. 중립 `modules/`(프로필·keypoint_goal·adr·object_wrench·perception_delay)만 쓴다. 테스트가 잠근다.
- 손마다 컵 1개 · 무작위 keypoint 목표 · 좌우 따로 학습 · **양팔 자산 유지**(쓰지 않는 팔은 휴식 자세 PD 고정).
- 09.29 2차(사용자): 컵 x 0.10–0.40·y 폭 0.20 · **팔 관절 증분(cup_grasp)** · **손 = pour_bi_rh 와 같은 사상** · 관측 잡음·지연 켬(컵 회전 잡음 제외) · 컵 질량·손/컵 마찰 다양화(마찰 1.5–2.5).
- gym id `open-rh_{r,l}_aglt-lstm` (+`-play-lstm`) → 로그 `log/rl_games/open-rh/{right,left}/aglt/<label>/`.
- 보상은 여기 없다: `reward_gen/rh_aglt_{r,l}/iter_NN/compute_reward.py` 를 `env.reward_code_path` 로 준다(t2r 루프, reward-audit 안 씀).

## 차원
| | 값 | 구성 |
|---|---|---|
| action | **13** | 팔 7(관절 증분) + 손 6(구동, grip 구간 절대 목표) |
| actor | **96** | `layout.actor_layout` — 팔 q·qd·q*, 손 q·목표오차, palm pos·R(열0·1), 컵−palm, 컵 축, 손끝−컵, kp−목표kp 12, 목표−palm, 촉각 tanh(F/3 N) 5, 직전 액션 |
| critic | **128** | actor(참값) + 손 qd · 컵 선/각속도 · palm 선속도 · 첫마디/손끝/손바닥 접촉 tanh(F/5 N) · lifted · grasped · 진행 · 성공수 · kp 거리 · 근접 누적 |
actor 에 손 관절속도 없음(실기 드라이버 velocity ≠ 관절속도, 09.07). 잡음: q 0.002 · qd 0.05 · FK 0.005 · 촉각 0.1 N ·
컵 pose 지연 0→3 스텝·위치 1.5 cm(ADR), 회전 잡음 0(축대칭, 사용자 09.29). ★RH pour 트랙은 09.28 에 "성공 전 잡음 끔"을 정했다 — 이 트랙은 사용자 결정으로 켠다.

## 제어
- **팔 관절 증분**(cup_grasp·실기 joint_decoder 와 같은 법칙): q_raw = clamp(q* + 0.05·a) → q* = 0.1·q_raw + 0.9·q*,
  포화 slew 0.3 rad/s, 위치 목표만.
- **손**(pour_bi_rh PourFJ 와 같다): 관절별 [min(open,grip), max(open,grip)] 에 선형 lo + ½(a+1)(hi−lo) — a=0 은 반쯤 쥔 손,
  thumb_1 만 +1 이 편 쪽. EMA 0.1 · 스텝당 ≤ 관절 전 범위 × dt(벤더 1 s) · 컵 접촉 > 1 N 인 손가락은 닫기만 정지.
- 언더액추: 종속 6(엄지는 thumb_4←thumb_3←thumb_2 연쇄) 0/0 · 부팅 URDF↔USD gearing 대조 · 종속 한계 ±1.5 rad 확장.

## 장면 실측 (09.29, URDF FK/IK — `tests` 가 다시 잰다)
```
먼 출발(09.29 2차, 사용자 "cup_grasp 처럼 테이블 앞 바깥") palm_sensor (0.080, −0.300, 0.450) — 상판 앞 가장자리 x 0.07 위
  관절 (−1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646)  한계 여유 10.5°(손목 r_aj_6)  손 링크 최저 z 0.400
  ★방향 규칙(사용자): palm_sensor 국소 +z(손바닥 중심 방향)는 중력에 수직, 두 손바닥이 마주 봄(우 +y·좌 −y),
    엄지 쪽이 새끼 쪽보다 확실히 위(뒤집힘 방지) — 엄지 끝이 새끼 끝보다 71 mm 위. 손가락은 수평에서 −10°.
    x 0.05 이하로 더 물리면 손목 여유가 6° 아래라 0.08 에 둔다.
  ★쉬는 팔 = 활성 팔 시작 자세의 y 거울, 손은 같은 편 손(사용자 09.29 "왼팔 오른팔 모두 대칭적으로"). 프로필 옛 휴식 자세는
    작업면 가운데(y ≈ 0)로 뻗어 반대쪽 컵 소환 범위와 겹쳤다. (이전 출발 (0.22, −0.41, 0.48) 은 aglt_*_i00b 까지)
컵 57 mm(shaker_closed_thick × 0.65) 0.134 kg · 원점 z 0.265 · 윗면 0.319 → 시작 손이 모든 컵 위(손 그림자 회피 불필요)
소환 x 0.10–0.40 · y −0.30–−0.10 (좌: y +0.10–+0.30) · 상판 구멍 (0.135, ∓0.30)·(0.27, ∓0.30) 에 걸리면 재추첨(여유 1 cm)
  구멍 = env_v1 상판 카운터보어 반경 30.5 mm — 그 위에 놓인 컵은 기울거나 빠진다(grasp_fj_t2r 09.17 점검 이상 4.8 %)
좌 거울: 4지 손끝 0.00 mm · 엄지 5.5 mm · 손바닥 2.7 mm(벤더 자산 비대칭, 09.14 기록과 같다)
```
IK 격자(손 방향 고정, 파지 자세 = 컵 + (−0.071, −0.034, +0.018)): 소환 박스 안은 도달. 목표 박스 모서리 높은 곳
(x ≤ 0.32, +0.2 m 위)은 손 방향을 고정하면 한계에 닿는다 — 정책은 손 방향을 바꿀 수 있어 막는 조건은 아니다.

## 목표·성공·종료
- 첫 목표 = 컵 + xy U(±0.05) + z U(+0.10, +0.18). 성공마다 ±0.08 이동(박스 = 소환 박스, z +0.08–+0.22). 에피소드 목표 5개.
- **성공** = kp 최대거리 ≤ tol 인 스텝이 10 쌓이고 **그 스텝에 목표 근처에서 쥐고 있음**(엄지 AND 다른 손가락 > 1 N). tol 0.10 → ×0.9 → 0.02
  (3000 프레임마다 직전 에피소드 성공수 평균 ≥ 2). play 는 tol_floor 고정.
- 종료: 컵 낙하 · 컵 상자 이탈 · 컵 60° 넘게 기울어짐 · 팔 qd > 20 · mimic 폭주(qd 100 × 3스텝 / 오차 3 rad / 비유한) ·
  손이 상판 5 mm 아래 · 컵 > 3 m/s·30 rad/s · 손 접촉 > 200 N 2스텝. truncate = 15 s 또는 목표 5개.
- DR: 손·컵 마찰 U(1.5, 2.5)(처음부터, 런타임 확장 불가) · 컵 질량 ×0.8–1.25 에서 ADR 로 ×0.5–2.5 · 관절 게인 ×1 → ×0.5–2.0(ADR) ·
  들린 컵 외란 0→5 N/kg(ADR). ADR 지표 = 직전 에피소드 성공수/5 > 0.4.

## 파지 모양 로그 (09.29 사용자 "로그로 확인되게", 로그 전용 — 관측·보상 무관, `shape_metrics.py`)
쥔 스텝(env grasp 플래그)만 평균한다. `shape/palm_normal_z`·`finger_dir_z`(수평 ≈ 0, 위에서 집기 ≈ −1) · `fingers_touching` ·
`prox_touching`(첫마디까지 감싼 손가락 수) · 손가락별 `shape/<finger>` · `palm_axis_r`(인벨롭 실측 1.5) · `grasp_h_mouth`(실측 +0.15) ·
`thumb_oppose` · `cup_tilt_deg`, 그리고 편 손 접근 확인용 `closure_near_open`(10 cm 안·아직 안 쥠)·`closure_far`(20 cm 밖).
aglt_{r,l}_i00b 는 이 지표 전(20f05c35)에 떴다 — 다음 기동부터 기록된다.

## 기동
```bash
cd ~/rl_ws/hdgp
PYTHONPATH=source/openarm /usr/bin/python3 -m pytest source/openarm/openarm/agnostic/tasks/rh_aglt_r/tests source/openarm/openarm/agnostic/tasks/rh_aglt_l/tests -q
# 부팅 스모크(영 보상)
./train.sh open-rh_r_aglt-lstm rh_aglt_smoke --num_envs 16 --max_iterations 5 --headless \
  agent.params.config.minibatch_size=256 agent.params.config.central_value_config.minibatch_size=256
# 본 학습
./train.sh open-rh_r_aglt-lstm <label> --num_envs 4096 --headless env.reward_code_path=$PWD/reward_gen/rh_aglt_r/iter_NN/compute_reward.py
```
★minibatch 16384 는 num_envs × horizon 16 의 약수여야 한다 → num_envs 는 1024 의 배수(스모크는 minibatch 를 덮는다).
★env `__init__` 이 `resolve_cfg` 를 다시 불러 robot·컵 cfg 를 프로필에서 새로 만든다 — `env.robot_cfg.*` hydra 오버라이드는 무시된다.
부팅 로그에서 볼 것: `[rh_aglt] 시작 palm … 손바닥↔컵 거리 평균` 이 대역 안, `종속 mimic 6`, action 13 · obs 96 · critic 128.
