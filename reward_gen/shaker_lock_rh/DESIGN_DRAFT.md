# shaker_lock_rh — 셰이커 리드 바요넷 잠금 text2reward 트랙 설계안 (초안, 2026-10-06)

> 상태: **설계 초안** — 코드·환경은 아직 없다. 열린 결정(§7)을 사용자가 정한 뒤 구현한다.
> 형상: `assets/cocktail_rh56/` rev3 (컵 이빨 4개 + 리드 내벽 V 리드인·30° 슬롯·잠금 턱, 딸깍 돌기 없음, 홀더 없음)

## 1. 기존 자산 중 재사용하는 것

| 재사용 | 위치 | 이 트랙에서 |
|---|---|---|
| RH56F1 양손 Fabrics 환경 골격 | `tasks/pour_fabric_mimic/` (gym `open-rh_b_pour_fab_mimic`) | 사본으로 새 태스크 `tasks/shaker_lock_mimic/` 생성 (원본 무변경 — pour 트랙 규칙과 동일) |
| RH56F1 프로필·좌손 미러·mimic 폭주 리셋 | `pour_fabric_mimic/robot_profiles.py`, `mimic_runaway_dep_qd` | 그대로 |
| 쥔 상태 시작 뱅크 | `pour_fabric_mimic/start_bank.py` | **핵심** — "다른 작업자가 파지 완료한 상태에서 시작"을 그대로 구현. 수집 대상만 리드/컵으로 바꿈 |
| 보상 계약 (`RewardContext` dataclass → 프롬프트 자동 렌더) | `pour_fabric_mimic/t2r/{context,loader,prompts,validator}.py` | 구조 그대로, **필드만 셰이커용으로 새로 정의** (§3) |
| t2r CLI (render / ingest / reflect) | `scripts/reward_gen/t2r_rh.py` | 트랙 전용 `t2r_lock.py` 로 복제 (import 경로만 교체) |
| 라운드 판정기·루프 프롬프트 | `t2r_rh_round.py`, `LOOP_PROMPT_rh.md` | 복제 후 KEY_TAGS·성공 지표만 교체 |
| 설계 브리프(매 라운드 프롬프트에 통째로 투입) | `reward_gen/pour_bi_rh/DESIGN_BRIEF.md` 형식 | `reward_gen/shaker_lock_rh/DESIGN_BRIEF.md` 신설 |
| 단계 원장 검사 | `reward_gen/pour_bi_rh/stage_ledger.py`, `traj_ledger.py` | 잠금 참 궤적용으로 새로 작성 (§5) |

## 2. 태스크 정의

**시작 상태(뱅크)**: 오른손이 리드 노브를 쥐고 컵 위에 들고 있음 · 컵은 (A) 왼손이 쥠 또는 (B) 테이블 위 — §7-1.
**목표**: 하강 → V 리드인이 이빨을 홈으로 유도(자가 정렬) → 바닥까지 안착 → 시계방향 30° 비틀기 → 잠금 → (선택) 노브 놓고 후퇴.

**성공 판정 (env 계산, 보상이 재정의 불가)** — 컵 프레임 기준 리드 상대 자세로:
- 리드 상대 yaw 가 잠금각(−30° mod 90°) ± 4° · 상대 z 가 잠금 높이 ± 1.5 mm · 리드 축 기울기 < 3°
- 컵 직립(기울기 < 10°)·낙하 없음 · 20 제어 스텝 유지
- 잠금 판정은 **기하로**: 이빨 하면이 잠금 턱 위(턱 겹침 > 0)인지 — 상대 yaw·z 로 정확히 계산 가능

## 3. RewardContext 필드 초안 (이 필드 이름·주석이 곧 LLM 환경 설명)

```
상수:  table_z, lock_twist (=π/6), lug_pitch (=π/2), seat_dz_m(안착 z), lock_dz_m(잠금 z), vlead_height_m
손(리드 쥔 쪽 lid_*):  palm_pos/lin_vel/axes, tips_pos, hand_closure, finger_force, palm_force,
                       grasped(엄지+손가락 노브 접촉), arm_qd/qdd/limit_margin
손(컵 쪽 cup_*, A안만):  같은 구성
리드:  lid_pos, lid_quat, lid_up, lid_lin_vel, lid_ang_vel
컵:    cup_pos, cup_quat, cup_up, cup_tilt, cup_lin_vel, cup_ang_vel, cup_spawn_pos
상대(컵 프레임, env 가 계산 — 가장 중요):
  rel_axis_dist   (N,) 리드 축↔컵 축 수평 거리 [m]
  rel_axis_tilt   (N,) 두 축 사이 각 [rad]
  rel_dz          (N,) 리드 하강량: 0 = 안착, + = 위 [m]
  rel_yaw_mod     (N,) 리드 상대 yaw 를 90° 주기로 접은 값 (0 = 홈 입구 정렬) [rad]
  twist_progress  (N,) 안착 후 비틀린 정도 0..1 (부호 있는 방향 — 반대로 돌리면 음수 → 방향 없는 항 착취 방지)
  lug_in_groove   (N,) bool 이빨이 V 리드인/채널 안 (기하 판정)
  seated, locked  (N,) bool
  lug_wall_force  (N,) 이빨↔홈 벽 접촉력 [N] (재밍 신호)
충돌:  hand_foreign_force, lid_table_force
판정/시간:  success, episode_progress · actions, prev_actions
```

★pour i00~i07 교훈: 기울기 항을 **방향 없는 크기**로만 줘서 엉뚱한 방향으로 기울여도 보상이 나왔다(LOOP_STATE round8).
→ 비틀기는 처음부터 **부호 있는** `twist_progress` 로만 노출한다.

## 4. 보상 단계 골격 (프롬프트 '추가 지식'에 넣을 내용 — LLM 이 코드화)

1. 정렬: `exp(-k·rel_axis_dist)`, 축 기울기 벌점 — 하강 전 축 맞추기
2. 하강: `rel_dz` 감소의 **증분**(Δ) 보상 — 높이 자체를 주면 공중 정지에 수입 (pour 의 `d_in_target` 교훈과 동일)
3. 자가 정렬 허용: `lug_in_groove` 이후 yaw 오차는 V 가 해결 — 손으로 yaw 를 억지로 맞추는 항은 V 구간에서 끔(손목 비틀며 재밍)
4. 비틀기: `seated` 게이트 뒤 `twist_progress` 증분 보상, 역회전 벌점
5. 잠금 보너스: `locked` · `success`
6. 규율: 재밍(`lug_wall_force` 큰데 진척 없음) 벌점, 리드 놓침(`grasped` 해제) 벌점, 컵 기울기/이동 벌점, 팔 관절 한계 여유, 액션 변화율

## 5. 학습 전 검증 게이트 (pour 트랙 규약 그대로 + 잠금 전용)

1. **물리 프로브(대본)**: 안착 위치에서 리드를 놓으면 중력으로 V 를 타고 홈에 들어가는가 (μ < tan40° = 0.84 조건, sim 마찰값 확인)
   → 들어가지 않으면 보상 문제가 아니라 접촉 모델 문제. 리드는 **SDF 콜라이더 필수**(convex hull 은 홈을 메움 — 컵 트랙 선례).
2. 대본 비틀기: 안착 후 손목으로 30° 회전 → `locked` True, 들어 올림 시 이빨이 턱에 걸림
3. `stage_ledger`: 참 궤적(정렬→하강→안착→비틀기→잠금)에서 스텝 보상 절벽 없음 · 공중 정지/안착 후 정지/역회전/억지 비틀기(재밍)가 참 궤적보다 낮음
4. `t2r_lock.py ingest` PASS

## 6. 단계별 일정 제안

| 단계 | 내용 | 산출 |
|---|---|---|
| 0 | 열린 결정 확정 (§7) | — |
| 1 | 셰이커 sim USD(SDF) 변환 + 물리 프로브 1·2 | 프로브 로그, 마찰·간격 확정 |
| 2 | `tasks/shaker_lock_mimic/` 환경 (pour 사본, 비드 제거, 상대 자세 계산) + 테스트 | gym id, boot 프로브 |
| 3 | 시작 뱅크 수집 (리드 노브 파지 상태) | `.npz` |
| 4 | `RewardContext`·프롬프트·validator·`t2r_lock.py` + DESIGN_BRIEF | iter_00 prompt.md |
| 5 | iter_00 생성 → ingest → 학습 → reflect 루프 | iter_NN |

## 7. 열린 결정 (사용자)

1. **컵을 누가 잡나** — 홀더를 폐기했으므로 30° 비틀 때 반작용 토크를 받을 곳이 필요하다.
   - A. 왼손이 컵을 쥔다(양손, pour 트랙 구조 그대로 · 액션 24) — 실기와 같은 조건이면 이쪽
   - B. 컵은 테이블 위, 오른손만(액션 12) — 컵이 같이 돌아갈 위험. 테이블 마찰/고정 지그 필요
2. **시작 뱅크 출처** — 다른 작업자의 파지 정책 체크포인트를 받을 수 있나? 없으면 대본으로 노브 핀치 파지 상태를 만들어 수집
3. **노브 파지 방식** — 지금 scene 은 상단 핀치(노브 십자 36 mm 를 엄지·검지로 집기). 다른 작업자 정책의 파지 형태와 맞춰야 함
4. **종료 조건** — 잠금까지만? 노브 놓고 후퇴까지?
5. **학습 호스트** — 5090(local) / server GPU / vision-3090
