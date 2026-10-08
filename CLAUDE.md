# hdgp — 학습 실험 저장소 공통 규칙

> **이 저장소는 일반 SW 개발이 아니라 실험 루프다.** 코드 품질보다 가설 검증이 우선.
>
> **도메인 해석·진단 방법론·reward 구조·핵심 지표는 각 프로젝트 전용 CLAUDE.md를 기준으로 한다.**
> - 해당 태스크 폴더의 CLAUDE.md (주로 `source/openarm/openarm/agnostic/tasks/<task>/CLAUDE.md`)
>
> hdgp는 **모든 학습 프로젝트 공통**만 담는다. 포괄적 지표·프로젝트 도메인 규칙은 여기 두지 않는다.

Isaac Lab / Isaac Sim 기반. OpenArm(7 DOF) + Teosllo(20 DOF) 조작.

---

## 핵심 규칙: 로그 먼저

**로그(TFEvents) 수치 근거 없이 코드를 수정하지 않는다.**

```
1. TFEvents + 체크포인트에서 수치 추출
2. 이전 test와 지표 비교
3. 가설 주도로 primary bottleneck 특정
   (고정 체크리스트 ✗ — 프로젝트 전용 CLAUDE.md의 진단 방법론을 따른다)
4. 최소 수정 → reward-audit 통과 (사람이 쓰는 보상만. t2r·IKER 루프 보상은 제외)
5. record_test_snapshot(train.sh 자동) → analysis.md 누적 → 주제 완료 시 Notion
```

---

## 증거 우선순위

하위 항목이 상위 항목을 뒤집을 수 없다.

```
1 TFEvents 스칼라 (parse_tfevents.py)
2 play.py 렌더링 / 사용자의 직접 관찰
3 analysis.md 이전 분석 이력
4 test_history.md (git diff 기반)
5 소스 코드 직접 읽기
6 일반 RL 직관
```
TFEvents 수치와 모순되는 일반 RL 조언은 무시한다.

---

## 분석 원칙

- **단일 로깅값·단일 시점 판단 금지.** 표면 지표(reward·entropy·성공률·frac)는 **출발점일 뿐**이다.
- **가설 주도 인과 추적**: 표면에서 멈추지 말고 메커니즘 — action 추종(raw→gate→applied), gate 경로,
  물리 좌표, joint 한계, reward 구성비 — 까지 직접 측정으로 검증한다.
- **봐야 할 지표는 가설마다 다르다.** 고정 N축 체크리스트만으론 RL 실험의 *원인*을 알 수 없다.
- 프로젝트별 진단 절차·핵심 지표·인과 모델은 **해당 프로젝트 CLAUDE.md 참조.**

---

## 실행 환경 (분산)

- **로컬** (`user@10.102.101.234`, RTX 5090): 분석·코드 수정·모니터링.
- **server** (`oem@10.102.101.240`, RTX PRO 6000 Blackwell ×2, 98GB): 학습.
- **코드 동기화**: 로컬 수정 → `git push` → server `git pull` (branch `main`).
- **학습**: SSH + conda(miniforge3 `proj-hdgp-py311`) →
  `CUDA_VISIBLE_DEVICES=N NOTE="" ./train.sh <task> <label> --num_envs 2048 --headless` (setsid 백그라운드).
- **모니터링**: 로컬서 `sync_server.sh`(60s rsync) → `watch_pour.sh`.

---

## 코드 수정 규칙

```
1. 한 번에 하나의 가설만 검증 (사용자가 통합 명시 시 예외).
2. obs/action 차원 변경 금지 (명시 요청 없이).
3. 관련 없는 파일 수정 금지 — 요청된 프로젝트 폴더만.
4. 새 reward term 전에 기존 weight 조정으로 해결 가능한지 먼저 확인.
5. 사람이 손으로 바꾸는 reward/gate/weight 는 변경 전 reward-audit 통과 (~/.claude/skills/reward-audit/).
   t2r·IKER 루프의 생성 보상에는 쓰지 않는다.
6. 변경 후 예상 지표 이동 방향 명시.
```

---

## 실험 규칙

- 보상이 새 행동을 요구하거나 obs·액션의 의미·차원이 바뀌면 fresh 로 시작한다. 앞 단계 정책 위에 다음 단계를 쌓는 단계 커리큘럼과 고원 탈출용 warm restart 만 예외이고, 그때는 기동 보고에 이유를 적는다.
- seed 는 통일한다. seed 를 바꿔 결과를 고치지 않는다. seed 에 따라 성공이 갈리면 보상 설계 문제로 본다.
- 종료·지속은 트랙의 단계별 기대 epoch 로 판단한다(표는 트랙 CLAUDE.md, t2r 은 라운드 판정 규칙). 기대 안에서는 붕괴(NaN, 지표 동결, 급락)가 아니면 끊지 않는다. 기대를 넘겼는데 다음 단계 신호가 없으면 연장을 권하지 않고 중단 후보로 보고한다. 기대 epoch 가 없으면 중단·연장을 권하기 전에 사용자에게 묻는다.

---

## 기록 규칙

- **학습 전**: `record_test_snapshot.py` (train.sh가 자동 기록).
- **분석 후**: `analysis.md` 누적.
- **주제 완료 시 Notion**: Notion MCP 로 해당 섹션을 읽은 뒤 추가만 한다(구조 변경은 허락 뒤). 6단계 구조: **질문 → 문제 → 원인 → 변경 → 근거 → 결과**.

---

## 도구

| 도구 | 위치 | 용도 |
|-----|------|------|
| `parse_tfevents.py` | `scripts/tools/` | TFEvents 바이너리 파싱 (tensorflow 불필요) |
| `monitor_pour.py` / `watch_pour.sh` | `rl_ws/scripts/tools/` | 여러 실험 통합 비교표 |
| `sync_server.sh` | `rl_ws/scripts/tools/` | server TFEvents → 로컬 mirror (60s) |
| `record_test_snapshot.py` | `scripts/tools/` | 학습 전 git diff + 파라미터 스냅샷 |
| `openarm_fk.py` | `scripts/tools/` | OpenArm FK (numpy only) |
| `train.sh` | 루트 | 스냅샷 자동 기록 + 학습 실행 |

> **HTML 보고 페이지**(eli5·분석 리포트·아티팩트)를 만들 때는 [.claude/rules/ui-pages.md](.claude/rules/ui-pages.md)
> 를 따른다 — ui-ux-pro-max 검색 스크립트 조회, 내보내기 전 헤드리스 렌더 확인, 로컬 사본 charset 래핑.

---

## 보고는 끝난 뒤 한 번, 요약으로 (09.17 사용자 지시)

**현상.** 매 턴 "필요한 것 목록(비공개 점검)", "필요(비공개 점검)" 같은 **내부 점검 목록**을 출력하고
어떤 파일을 왜 여는지·무엇이 무엇에 의존하는지 단계마다 중계했다.
사용자 평: *"자꾸 보여줘서 뭐가 뭔지 파악이 안돼."* 정보가 아니라 소음이고 토큰만 쓴다.

### 금지

**아래 문구로 시작하는 문단은 언어·표기를 불문하고 금지한다.** (09.17 재발 — 한국어를 영어로 바꿔 쓰며 계속 위반했다)

```
필요한 것 / 필요한 것 목록 / 필요(비공개 점검) / 필요한 것 정리
Needed next / What I need next / Next I need / 다음에 필요한 것
"독립인 N·M 을 함께 요청합니다" / "…에 의존하므로 나중에"
```

- **계획 단계는 내부에서만 한다.** "(비공개)"라고 적어도 화면에 나오면 위반이다. 무엇을 조회할지·무엇이 무엇에 의존하는지는 **쓰지 않는다**.
- 파일별 편집 경과, 중간 단계 나열, 시도했다 버린 접근의 서술 금지.
- 도구 호출 앞 설명은 **한 줄 이하**. 자명하면 생략한다.
- **같은 산출물을 두 번 보고하지 않는다.** 영상·시트·이미지를 파일로 보냈으면 캡션이 설명이다 — 다음 메시지에서 다시 묘사하지 않는다. 관찰 결과가 필요하면 최종 요약에 **한 줄**로 넣는다.
- 틱·백그라운드 작업 대기 중에 "기다리는 중"이라는 메시지를 따로 쓰지 않는다. 알림이 오면 그때 결과만 말한다.

### 기본형

조사·수정·기동·복구가 **전부 끝난 뒤 한 번** 요약한다. 네 가지만 적는다.

```
무엇이 바뀌었나 · 왜(근거 수치 또는 오류 한 줄) · 지금 상태 · 다음에 볼 것
```

수치·커밋 해시·경로는 결론을 뒷받침할 때만 넣는다. 같은 사실을 두 번 쓰지 않는다.

### 예외 — 즉시 알린다

- 사용자가 답해야 다음 행동이 갈리는 **결정** (선택지와 근거만, 경과는 빼고)
- 크래시·데이터 손실·다른 세션 작업을 덮어쓸 위험
- 실기 동작 승인(memory `real-robot-motion-needs-permission`)
- 루프 틱 보고는 기존 형식 유지 — 한 줄 지표 + 추세 해석 + 다음 틱 예고
