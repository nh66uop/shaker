# iter_01 운영 노트 — 2026-09-27

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. 앞의 DESIGN_BRIEF 가 이 트랙 전체의 측정·교훈이다.


## iter_00 이 한 것과 남긴 것 (t2r_rh4_i00, server GPU1, 파지 단계 ckpt 웜스타트)

★**이 트랙에서 처음으로 붓기가 성립했다.** 결정론 평가 ep1300(64 env · 900 스텝 · 동일 조건):
도착 env 64/64 · 도착 비드 1089/1280(85 %) · 리시버 담김 최대 중앙 0.900 · 방향 오차 중앙 12° ·
전 env 가 90° 를 넘겨 기울임 · 소스 손목 자세는 한계에 붙지 않았다.
직전 라운드(i24)의 최고가 1280 알 중 1 알이었다.

### 그런데 비드를 다시 가져온다 (측정)

ep1300 trace(64 env) 시간축 중앙값:

| t | 담김 | 소스 잔량 | 소스 기울기 | 리시버 기울기 |
|---|---|---|---|---|
| 300 | 0.000 | 1.000 | 93° | 18° |
| 375 | 0.250 | 0.650 | 99° | 3° |
| 450 | 0.725 | 0.075 | 105° | 4° |
| 525 | 0.375 | 0.375 | 56° | 4° |
| 600 | 0.000 | 1.000 | 31° | 4° |
| 750 | 0.000 | 1.000 | 82° | 6° |

에피소드 끝 담김 0.135(최대 0.900). 붓고-되받기 사이클 env 당 중앙 1 회(최대 2 회). 끝 흘림 0.000 —
비드는 밖으로 새는 게 아니라 **소스 컵으로 되돌아간다**.

### 왜 되받는가 (에피소드 수입, ep1300 trace, env 중앙)

| 항 | 에피소드 합 | 비중 |
|---|---|---|
| pour_open | 1787 | 65.7 % |
| deliver_delta | 300 | 11.0 % |
| success_bonus | 237 | 8.7 % |
| rcv_upright | −227 | −8.4 % |
| in_target_level | 147 | 5.4 % |

- `pour_open = W_POUR · 창 · 방향 · rel(기울기) · **src_frac**` 는 **상태형**이고 소스 잔량에 비례한다.
  소스가 비면 스텝당 최대 8 짜리 수입이 끊긴다 → 계속 벌려면 비드를 되받아야 한다.
- `deliver_delta` 는 양(+) 합 300 · 음(−) 합 **0** 이다. env 의 `d_in_target` 은 "건너간 비드"의 단조 증분이라
  되받아도 음수 지급이 없다. 되받기에 비용이 전혀 없다.
- 되돌아간 비드 비율 중앙 0.95(들어간 총량 1.00).

### 그 뒤 400~750 epoch 동안 일어난 일 (학습 곡선 20점 평균)

| 지표 | e1330 | e1710 | e2084 |
|---|---|---|---|
| 배달 비드 비율 | 0.338 | 0.297 | **0.114** |
| 담김 최대 | 0.694 | 0.638 | **0.414** |
| success_bonus | 0.807 | 0.733 | **0.181** |
| 흘림(끝) | 0.153 | 0.174 | **0.235** |
| pour_open | 1.730 | 1.956 | **3.326** |
| 총보상 | 3.315 | 3.461 | **4.244** |
| entropy | 1.23 | −1.26 | −2.73 |

결정론 평가도 같은 방향이다. ep1700: 도착 env 56/64 · 도착 비드 828/1280(65 %) · 담김 최대 중앙 0.750 ·
끝 담김 0.095. 같은 구간에 소스 파지 유지는 0.70 → 0.97 로 **좋아졌다** —
정책은 "컵을 더 단단히 쥐고 더 오래 기울이기"로 갔고, 그것이 `pour_open` 을 최대로 만든다.

**보상은 28 % 오르는데 배달은 66 % 줄었다.** 보상이 과제를 대변하지 못한다는 직접 증거다.

### 마지막 상태 (ep3100 결정론, 런 종료 시점 e3254)

| | ep1300 | ep1700 | ep3100 |
|---|---|---|---|
| 도착 비드 | 1089/1280 (85 %) | 828 (65 %) | **178 (14 %)** |
| 도착 env | 64/64 | 56/64 | 44/64 |
| 담김 최대 중앙 | 0.900 | 0.750 | 0.100 |
| 에피소드 끝 담김 | 0.135 | 0.095 | 0.000 |
| 방향 오차 중앙 | 12° | 14° | **6°** |
| 최대 기울기 중앙 | 119.8° | 119.0° | 89.5° |
| 소스 파지 유지 | 0.70 | 0.97 | **0.99** |

ep3100 에서 정책이 하는 일: 두 컵을 거의 완벽하게 쥐고(0.99/1.00), 주둥이를 받는 컵 입구 위
**수평 1.0 cm · 높이 +7.4 cm** 에 맞추고, 방향 오차 **6°** 로, **89.5°** 까지 기울인 자세를
전체 스텝의 **79 %** 동안 유지한다. 그런데 **소스 컵 잔량 중앙이 0.900** — 비드가 나가지 않는다.

즉 되받기(ep1300)에서 한 발 더 나아가 **아예 붓지 않는 자세**로 수렴했다.
`pour_open = W_POUR · 창 · 방향 · rel(기울기) · src_frac` 에서 `src_frac` 은 소스에 남은 비드라,
**비드를 흘리지 않을수록 이 수입이 오래 유지된다.** 붓는 순간 수입이 사라지므로 "붓기 직전 자세"가 최적해다.
[가설] 89.5° 에서도 비드가 안 나가는 물리적 메커니즘(컵 안쪽 벽에 눌려 있는지, 미세한 위쪽 피치인지)은
영상 프레임으로 확인해야 한다 — `our_source/pour_videos/t2r_rh4_i00_ep3100.mp4`.

조준·방향·파지·자세는 이 트랙 역대 최고다(방향 오차 6°, 주둥이 수평 1.0 cm). **붓기만 하지 않는다.**


## 2026-09-27 오전 추가 — 사용자 영상 판정과 측정 (이 절이 위 내용보다 우선한다)

### 사용자 판정
- ep1300: "조인트가 과도하게 꺾여 있다는 문제는 있지만 제대로 틸팅되는 게 보인다."
- ep1300·ep3100 공통: **과도한 관절 포화, 그리고 두 팔이 서로 겹치는 이상한 자세.** ep3100 이 가장 나쁘다.
- "제대로 컵을 잡고 틸팅 각도는 만들 수 있다. 지금 보상이 그걸 못 알려 주는 것이다."

### 측정 1 — 관절 포화 (후반 2/3, 관절 한계 5 % 안에 머무는 스텝 비율)

| | j2 | j5 | j6 | j7 | 한 관절이라도 |
|---|---|---|---|---|---|
| ep1300 소스 | 0.07 | 0.56 | 0.00 | 0.60 | 0.66 |
| ep1300 리시버 | 0.51 | 0.00 | 0.00 | 0.00 | 0.51 |
| ep3100 소스 | 0.88 | 0.85 | 0.00 | **0.99** | **0.99** |
| ep3100 리시버 | 0.48 | 0.00 | 0.63 | 0.00 | 0.82 |

두 손바닥 거리 중앙: ep1300 0.159 m(최소 0.086) · ep3100 **0.121 m**(최소 0.087). 두 팔이 중심선에 몰려 있다.

### 측정 2 — 두 팔이 서로 통과하고 있었다 (env 설정)
- i00 런 `env.yaml`: `enable_self_collisions: false`. 양팔은 한 articulation 이라 이 값이 false 면
  **왼팔과 오른팔 링크 사이에 충돌이 생기지 않는다.** 영상의 "겹치는 팔"은 실제 관통이다.
- 접촉력이 생기지 않으니 `foreign`(손이 남의 것을 건드림) 벌점도 이것을 보지 못했다.
- 꺼 둔 이유(09.14, 시작 자세 손가락 자기접촉 226 N)는 09.15 자산 재빌드로 해결됐다. 학습이 쓴 USD 는
  재빌드본(09-15 23:16)이고 충돌 필터 46 쌍은 모두 한 손 안의 손가락 겹침, **좌-우 팔 사이 필터는 0 개**다.
- ★iter_01 부터 `env.enable_self_collisions=true` 로 학습한다(사용자 승인 09-27). 팔끼리 관통하는 자세는 물리적으로 막힌다.

### 측정 3 — 비드가 실제로 나오는 각은 70° 가 아니라 약 100° 다
두 런의 trace 에서 비드가 소스 컵을 떠난 순간의 기울기:

| | 사건 수 | p10 | p50 | p90 |
|---|---|---|---|---|
| ep1300 | 1657 | 94° | **101°** | 108° |
| ep3100 | 277 | 82° | 86° | 90° |

기울기 구간별 소스 잔량(ep1300): 60~70° 1.000 · 70~80° 1.000 · 80~90° 1.000 · **90~100° 1.000** · 100~110° 0.175 · 110~130° 0.000.
컵 내부 100.8 mm · 안지름 56.2 mm · 비드 20 개 더미 높이 29.9 mm(채움 0.30).

- 환경 설명의 "가득 찬 컵에서 첫 알이 약 70°" 는 **이 트랙의 컵에는 맞지 않는다**(비드는 가득 차 있지 않다).
- iter_00 의 `rel = 0.5·(1 + tanh((θ − 70°)/12°))` 는 86° 에서 이미 만점이다. ep3100 정책은 기울기 중앙 82.8° 로
  **11.6 초 연속** 버텼고 그동안 소스 잔량은 1.000 이었다. **보상이 비드가 안 나오는 각도에 만점을 주었다.**
  사용자 표현대로 "보상이 제대로 된 틸팅을 알려 주지 못했다."


### 측정 4 — self-collision ON 프로브 (파지 단계 ckpt, 64 env · 600 스텝, server GPU1)

| 지표 | OFF | ON |
|---|---|---|
| 소스 파지 유지(후반 2/3) | 0.988 | 0.971 |
| 리시버 파지 유지 | 0.990 | 0.960 |
| 리셋 직후 손가락 힘 최대(자기접촉 상주 여부) | 0.000 N | 0.000 N |
| 손 foreign 힘 p95 | 0 N | 0 N |
| 두 손바닥 거리 중앙 | 0.373 m | 0.462 m |

켜도 손가락 자기접촉이 생기지 않고 파지는 2~3 % 만 줄었다. 두 팔은 더 떨어져 움직인다.


## 2026-09-27 — DG-5F 세션(pour_bi2 iter_02) 보상과의 비교 (사용자 지시 "보상이 이상하다, 비교해 봐")

같은 붓기 과제를 DG-5F short 손으로 푸는 세션이 i16~i28 과 pour_bi2 iter_00~02 에서 얻은 교훈
(`reward_gen/pour_bi2/DESIGN_BRIEF.md` L1~L8)에 c4 를 대 보았다.

| 항목 | DG-5F iter_02 | RH c4 | 판정 |
|---|---|---|---|
| 회전 구동 | `8.0 · want · lip_gate · aim_factor · rot_prog`, rot_prog = 받는 쪽 기울기 / (1.4·premature_limit) **0° 부터 선형** | `4.0 · aim_gate · align · rel · remain`, rel = 방출 조건 tanh — **0~85° 기울기 0.02~0.07** | L1 위반: i26~28 이 속도형·벽형 구동으로 55° 를 못 넘었다 |
| 조준 게이트 | **바닥 0.3** + 품질 가산, 2r 꼬리 | `exp(-(lip_xy/2r)²) · clamp(dz/r)` 곱, 바닥 없음 | L3 위반 — 아래 측정 |
| 배달 / 흘림 단가 | 30 / **−40** (흘림 > 배달) | 15 / −6 (흘림 < 배달) | c4 는 흘리며 부어도 이득 |
| 성공 보너스 | 20 /스텝 | 5 /스텝 | |
| 요청량·want | `min(요청−부음, 잔량)` | 없음(전량) | RH ctx 에 pour_target_frac 없음 |
| 관절 여유 | 없음(P5: 붓기 자세는 박스 안 99~100 % 도달 가능, 막은 건 보상) | 회전 수입의 60 % 에 곱 | 사용자 판단(P5 와 같음)과 방향 일치, 단 게이트 0 을 물려받음 |
| 방출각 | ctx.premature_tilt_limit 배수 | FILL0=0.30 + 더미 모양 [가설] | RH ctx 에는 premature_tilt_limit 가 없다 |

**측정 (L3 게이트 분포, 두 컵 파지 스텝, p10/p50/p90)**

| 상태 | 주둥이-입구 수평 거리 / r | c4 aim_gate | 0.1 초과 비율 |
|---|---|---|---|
| 파지 단계 ckpt = i01 시작점 (self-col ON 프로브) | 9.8 / 12.4 / 14.7 | **0.000 / 0.000 / 0.000** | **0.000** |
| i00 ep1300 (붓기 성립 정책) | 0.2 / 0.6 / 5.6 | 0.000 / 0.909 / 0.992 | 0.818 |

→ c4 로 시작하면 **회전·자세 수입 전부(pour_open·posture_margin·pour_rate)가 첫 스텝부터 0** 이다.
정책이 주둥이를 입구 2r 안으로 먼저 가져오기 전까지 기울일 이유가 전혀 없고, 가져온 뒤에도 85° 까지 기울기 경사가 없다.
DG-5F 세션 L1·L3 가 "세 라운드 동안 한 알도 못 부은" 원인으로 적은 구조와 같다.



## 2026-09-27 최종 지시 — DG-5F iter_02 구조 이식 (이 절이 위의 모든 "iter_01 변경" 절을 대체한다)

사용자: "파지 쪽 빼고는 (DG-5F 보상을) 가져와도 될 것 같아. 둘의 차이는 손 차이뿐이니까."
사용자: "horizon length 를 16 으로 하고, LSTM 모듈을 사용할 거야."

### 무엇을 하라는가
- 아래 **DG-5F 세션 pour_bi2 iter_02 의 compute_reward 를 기준 코드로 삼아 RH 로 옮긴다.**
  항 이름·가중·형태(회전 구동 rot_prog 선형, 조준 바닥 0.3, lip_gate 2r 꼬리, 수렴 WIDE 커널, 배달 30 · 흘림 −40,
  성공 20, finish_upright, premature 벽 부과, cup_clash, hand_foreign, pour_dir, rcv_keep, action_rate·flip, arm_tremor)를
  그대로 유지한다. **바꾸는 것은 RH env 에 없는 ctx 필드를 RH ctx 로 계산하는 부분뿐이다.**
- **파지 관련 항은 가져오지 않는다**: `approach_both`, `grasp_src`, `grasp_rcv`(wrap_count 사용), `grip_keep`(파지 기준 필드 사용).
  대신 RH 의 파지 항(아래 "RH 쪽 참고 코드" = iter_00 의 approach_src / approach_rcv / grasp_hold / lift_src / lift_rcv)을 쓴다.
  ★이번 런은 LSTM 네트워크라 파지 단계 체크포인트(MLP)를 이어받을 수 없어 **처음부터(fresh)** 학습한다.
  그러므로 파지·접근 항은 파지를 0 에서 배울 만큼의 크기여야 한다 — DG-5F 의 approach 0.4/팔 · grasp 0.5/팔 과 같은 자릿수로 둔다.
- `grasp_ref_valid`(DG-5F `pour_rot` 의 곱 인자)는 RH 에 없다. 곱 인자에서 빼고 `src_grasped` 만 쓴다.
- `low_src` / `low_rcv`(기울인 채 들기 수입 차단)는 파지 기준 기록과 관련된 장치지만 기울기만 쓰므로 유지해도 된다.

### RH ctx 에 없는 DG-5F 필드 → RH 에서 계산하는 법 (env `pour_fabric/pour_rules.py` 와 같은 식)
| DG-5F 필드 | RH 계산 |
|---|---|
| `pour_dir_xy` d̂ | `normalize(rcv_cup_mouth_pos[:, :2] − src_cup_pos[:, :2])` (수평 단위벡터) |
| `src_tilt_toward_rcv` θ | `atan2((src_cup_up[:, :2] · d̂), src_cup_up[:, 2])` — 받는 컵 쪽 부호 있는 기울기 |
| `src_pour_lip_pos` | `src_cup_mouth_pos` 에 xy 로 `r·cosθ·d̂` 를 더하고 z 에서 `r·sinθ` 를 뺀다 (r = ctx.cup_radius) |
| `premature_tilt_limit` | RH 에는 없다. **측정값으로 정한 스칼라 `LIM` 을 쓴다.** DG-5F 는 유출각 − 약 20° 가 한계였다(유출 93.6°, 한계 p50 73.6°). RH 실측 유출 중앙 101°(p10 94°) → `LIM = radians(81)`. 주석에 "RH 실측 방출 101° − 20°" 로 출처를 적는다. rot_prog 는 1.4·LIM ≈ 113° 에서 포화해 실측 p90 108° 를 넘는다. |
| `bead_active_count` | `ctx.num_beads` (RH 는 20 개 고정) |
| `pour_target_frac` | RH 과제는 전량 이송(성공 = 담김 ≥ 0.80). `pour_target_frac = 1.0` 로 둔다. `pour_overshoot` 항은 쓰지 않는다(목표 초과가 없다). |
| `rcv_side_margin` | 리시버는 왼팔(+y). `ctx.rcv_cup_pos[:, 1]` (env-local y). |
| `cup_hit` | RH 에 없다. `cup_cup_force` 항으로 충분하다 — 빼도 된다. |
| `premature_tilt` (래치 플래그) | RH 에 없다. premature 항의 래치 부분(−0.4·flag)만 빼고 벽 부과(−12·...)는 계산값으로 유지. |
| `adr_progress` | RH 에 없다. `action_reg_adr` 는 작은 상수 가중(0.1)의 palm 포화 벌점으로 둔다. |

### RH 쪽에서 추가로 지킬 것
- **self-collision ON**(env.enable_self_collisions=true)으로 학습한다. 팔끼리 닿으면 `src/rcv_hand_foreign_force` 가 오른다
  (측정: ep1300 정책을 ON 에서 돌리면 34/64 env 에서 10 N 초과). DG-5F 의 `hand_foreign` 을 그대로 쓴다.
- 관절 여유: DG-5F 에는 항이 없다(그쪽 P5: 붓기 자세는 박스 안에서 도달 가능, 막은 것은 보상). 사용자 판단도 같다.
  RH 에서는 `ctx.src_arm_limit_margin` 으로 **작은 가산**(최대 스텝당 0.3 정도, 붓는 중에만)을 하나 둬도 된다. 회전 구동에 곱하지 않는다.
- 되받기: `pour_delta` 는 `d_in_target` 단조 증분이라 되받아 다시 부어도 재지급이 없다. 부어 놓고 유지(성공 20)가
  기울인 채 버티기(최대 8)보다 크게 두면 된다 — DG-5F 가중이 이미 그렇다.

### DG-5F 세션 기준 코드 (reward_gen/pour_bi2/iter_02/compute_reward.py 원문)

```text
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


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bimanual pour: grasp both cups, lift, bring the cups together, aim the pouring
    LIP over the receiver mouth and tilt until the requested number of beads has left
    the source cup and landed in the receiver.

    Changes of this iteration (exactly the two the round asks for):
      C1  AIMING NOW RAISES THE ROTATION INCOME and never switches it off.
          iter_01 poured with the lip ~3*cup_radius (0.123 m) beside the mouth: the old
          `pour_rot` had no aim factor at all, so tilting without aiming paid the same as
          tilting over the opening.  The aim factor is a BONUS ON A FLOOR
          (FLOOR = 0.3), never a precision gate - iter_26/27 multiplied by a narrow
          gaussian that evaluated to 1e-17 in the visited states and three rounds poured
          nothing.  Teeth are on the OUTCOME side instead: the spill charge per bead is
          raised 20 -> 40 (> the 30 paid for a bead that arrives), so pouring while badly
          aimed loses money while the rotation drive stays alive.
      C2  TREMOR AND ACTION-REVERSAL CHARGES ARE ALIVE FROM STEP 0 (no adr factor),
          small and weighted more on the receiver arm (its qd RMS ran 1.2-1.3x the source
          arm's, and the flip rate sat at 0.49 all run while adr_progress stayed 0.000).
          The palm-saturation charge stays ramped by adr_progress (R7).
      Everything else is unchanged in value: converge_cups, aim_lip, lip_gate, the
      rotation cap, the premature charge at the wall, grip-keep, safety, bookkeeping.
      Only logging was compacted (approach_src+approach_rcv, premature_charge+latched,
      cup_cup+nested merged) to keep the component count at 22 (R9).

    Design rules: every length is a multiple of ctx.cup_radius, every angle a fraction of
    ctx.premature_tilt_limit -> no asset dimension or absolute angle is hardcoded (R8).
    """
    eps = 1e-6
    zeros = torch.zeros_like(ctx.src_cup_tilt)

    # ---- shared scales ---------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # env latch radius for the lip (xy)
    wide = 10.0 * r_cup                      # WIDE kernel: covers the visited 0.2-0.45 m band
    lim = torch.clamp(ctx.premature_tilt_limit, min=1e-3)   # (N,) rad, from env
    n_fing = max(int(ctx.src_tips_pos.shape[1]), 1)

    src_g = ctx.src_grasped.float()
    rcv_g = ctx.rcv_grasped.float()
    both_g = src_g * rcv_g

    # Pre-lift / posture income must not be earnable by tilting before lifting (R6b):
    # the env only records the grasp reference while the cup is well below its limit,
    # so fade that income out with the cup's own tilt, using 0.3 * limit as the scale.
    low_src = 1.0 - torch.tanh(torch.clamp(ctx.src_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))
    low_rcv = 1.0 - torch.tanh(torch.clamp(ctx.rcv_cup_tilt - 0.3 * lim, min=0.0) / (0.3 * lim))

    # ---- 1. approach both hands (0.4 each; dies out once that cup is grasped) ---
    d_src = _dist(ctx.src_palm_pos, ctx.src_cup_pos)
    d_rcv = _dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos)
    approach_both = (0.4 * (1.0 - src_g) * low_src * (1.0 - torch.tanh(d_src / two_r))
                     + 0.4 * (1.0 - rcv_g) * low_rcv * (1.0 - torch.tanh(d_rcv / two_r)))

    # ---- 2-3. enveloping grasp (weight 0.5 each) -------------------------------
    # wrap counts are normalised by the number of fingers of THIS hand, not by 5.
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
    # Unchanged: this is what woke the pour up in iter_01 (1.0 per step, cups 0.21 m).
    converge_cups = 1.2 * both_g * lifted * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- 6. aim the LIP (not the rim centre) over the receiver mouth (0.8) -----
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * (0.6 * xy_score + 0.4 * h_score)

    # ---- 7. keep the grasp recorded at lift-off (iter_24 guard, weight 1.0) ----
    # Similarity to the reference relation only; never an absolute force target (R6a).
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
    # `want` tapers inside the last bead (env tolerance +-1 bead) and is 0 once spill
    # made the request unreachable -> the rotation income stops, the finish branch opens.
    want = _sat(need_beads)

    # ---- 8. rotation drive: the term that has to cross the ~90 deg wall (8.0) --
    # Gate (kept from iter_01, it is what finally paid): lip near the mouth with the
    # SAME radius the env latch uses, soft 2*cup_radius tail so 9-14 cm still pays.
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    # Progress measured against 1.4 * premature_tilt_limit (a multiple of a ctx angle):
    # reaches past the bead-release tilt for every fill and caps the income at 8.
    rot_prog = _sat(torch.clamp(ctx.src_tilt_toward_rcv, min=0.0) / (1.4 * lim))
    # NEW aim factor: a BONUS ON A FLOOR, not a gate. Both parts are smooth and their
    # tails stay clearly non-zero in the states iter_01 visited (lip 3 * cup_radius,
    # drop 0.03 m): xy part 0.24, height part 0.68 -> factor 0.58 (not 1e-7).
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)      # 1 inside 1r
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))  # low drop
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    # monotone in lip_d at fixed tilt (both lip_gate and aim_xy shrink with lip_d)
    pour_rot = 8.0 * src_g * ctx.src_grasp_ref_valid.float() * want * lip_gate * aim_factor * rot_prog

    # ---- 9. headline income: beads that actually arrive (30 per bead) ----------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 10. pouring past the request (15 per bead beyond the +-1 tolerance) ---
    over_beads = _sat(torch.clamp(ctx.bead_in_target_frac - ctx.pour_target_frac, min=0.0) * active - 1.0,
                      0.0, 5.0)
    pour_overshoot = -15.0 * over_beads

    # ---- 11. spill: 40 per lost bead (was 20) - the teeth for bad aim ----------
    # A bead that leaves the source and misses the receiver now costs MORE (40) than a
    # bead that arrives pays (30), so pouring beside the mouth is a net loss over the
    # episode while the rotation drive itself stays untouched. Charged on the increment.
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 12. finish branch: small, and per step below the pour income (R1) -----
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(ctx.src_cup_tilt / (0.3 * lim)))

    # ---- 13. terminal success bonus (env decides success, we only pay it) ------
    success_bonus = 20.0 * ctx.success.float()

    # ---- 14. premature tilt: charge AT the wall only, plus the latch reminder --
    # ramps in above 0.95 * limit over the last 0.05 * limit and only while the lip is
    # outside 2*cup_radius; inside that radius there is NO tilt charge at all (R3).
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
    # dimensionless unit-vector component (env voids at 0.3); ramp starts at half of it
    dir_ramp = _sat((ctx.pour_dir_xy[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 18. receiver cup: upright and on its own side of the centre line -----
    side_lo = 0.75 * r_cup                      # env voids below ~0.75 * cup_radius
    side_bad = _sat((side_lo - ctx.rcv_side_margin) / side_lo)
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * lim))) - 1.5 * side_bad

    # ---- 19. always-on action rate (iter_18 level, small) --------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)

    # ---- 20. NEW: action reversal, alive from step 0 (weight 0.30) ------------
    # measured flip rate 0.49 with |a| ~ 0.5 -> about -0.04 per step (-0.15 if the
    # actions are saturated): ~5 % of the per-step state income, far below iter_21
    # where full-weight charges taught the policy not to touch the cups.
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip

    # ---- 21. NEW: arm tremor, alive from step 0, receiver arm weighted more ---
    # the receiver arm ran 1.2-1.3x shakier than the source arm; scale 1 rad/s so the
    # measured RMS 0.46 / 0.42 sits on the slope, total about -0.15 per step.
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))

    # ---- 22. palm saturation stays ramped by ADR (R7) -------------------------
    adr = float(ctx.adr_progress)
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -adr * 0.3 * sat

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
        "action_rate": action_rate,
        "action_flip": action_flip,
        "arm_tremor": arm_tremor,
        "action_reg_adr": action_reg_adr,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components

```
