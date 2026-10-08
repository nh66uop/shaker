# round 10 operator notes — 목표 전환: 흘림 없는 이송

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 목표가 바뀌었다 (사용자 09.22)

> "pour-fab iter23처럼 이제는 안정적인 제어와 정밀조준 및 비드 이송이 중요해지고 있음. 목표는 흘림없는 이송임."

iter_09 는 **붓기를 성공시키는** 데까지 왔다. 이번 라운드의 목표는 **흘리지 않고 옮기는 것**이다.
세 가지가 핵심이다 — **안정적인 제어**(명령이 떨리지 않고, 컵을 휘두르지 않는다), **정밀 조준**(주둥이를 리시버 입구 위에 정확히 둔다),
**비드 이송**(천천히, 조금씩, 전부 받는 컵에 넣는다).

성공 판정은 `spill ≤ 0.40` 이라 흘려도 성공률은 오른다. **성공률만 올리는 보상은 이번 목표에 맞지 않는다.**

## 1. 지금 상태 (t2r_rh3_i09, server GPU0, 4096 env, e~915 · 240 M frames, 보상 iter_09)

학습 곡선 50-epoch 블록, 최근 6개:

| | 블록 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| episode_success | 0.368 | 0.405 | **0.461** | 0.447 | 0.364 | 0.410 |
| bead/spill | 0.232 | 0.206 | **0.171** | 0.178 | 0.216 | 0.198 |
| **흘림/유출** = spill/(1−in_source) | 0.366 | 0.326 | **0.270** | 0.281 | 0.355 | 0.321 |
| 이송 효율 = in_target/(1−in_source) | 0.596 | 0.642 | **0.699** | 0.685 | 0.601 | 0.641 |
| action/step_delta | 0.445 | 0.440 | 0.439 | 0.440 | 0.439 | 0.444 |

- **컵을 떠난 비드의 약 3분의 1이 여전히 바닥으로 간다.** 흘림/유출 비율은 0.270 까지 내려갔다가 0.32~0.36 으로 되돌아왔다. 줄어드는 추세가 아니다.
- **제어 안정은 런 내내 개선이 없다.** `action/step_delta`(스텝당 액션 변화량)가 0.44 에서 평평하고 `reward/action_rate` 는 -0.021 로 불변이다.
- 성공률은 0.36~0.46 에서 진동하고 **0.5 를 넘은 epoch 은 런 전체에 하나도 없다**(최대 0.4856).
- 조준은 좋다: `diag_mis_deg` **1.03°**(트랙 최저), `diag_lip_xy` 0.21 m, `src_tilt_deg` 72°.
- 파지 src 0.736 · rcv 0.775. nested 0.0008 · collision 0.0016. entropy 8.5~8.7.
- `fabric/src_rot_err_deg` 79~95°(소스 팔 자세 지령과 실제가 크게 어긋난다).

결정론 재생(ep650, 16 env, 900 스텝): 파지 1.00/1.00 · 기울임 101.1° · **도달 70.0 % · 흘림 28.4 %** · 성공 81.2 %.
**재생에서도 비드의 28 % 가 바닥으로 간다.** 탐색 노이즈를 빼도 흘림은 남는다.

## 2. iter_09 에 이미 있는 것

- `spill_delta = -W_SPILL(60) * ctx.d_spill` — 비드 1개당 -3. **게이트 없음.**
- `transferred = W_TRANSFER(4.0) * in_target_frac * ...`
- `action_rate = -W_ACT_RATE(0.05) * mean((a - a_prev)^2)`
- `palm_sat_src/rcv = -W_PALM_SAT(0.15) * sat_gate * ...` (sat_gate = 양손 파지·들기 이후)
- `pour_delta` 는 있으나 **스텝당 상한이 없다**.

## 3. 형제 트랙 pour_fabric iter_23 에 있고 iter_09 에 없는 것 (코드 확인)

`reward_gen/pour_bi/iter_23/compute_reward.py`:

1. **`flow_excess`** (L358) — `pour_delta = POUR_DELTA_W * clamp(d_in, -1, FLOW_CAP)`,
   `flow_excess = -FLOW_W * relu(d_in - FLOW_CAP)`, **FLOW_CAP 0.02**(스텝당 비드 2 %), **FLOW_W 20**.
   한 스텝에 많이 넘어가면 그 초과분은 보상이 아니라 벌점이다. **한꺼번에 쏟지 못하게 막는 장치.**
2. **`rot_speed`** (L353) — `-0.3 * held_src * tanh(clamp(|src_cup_ang_vel| - 1.0, 0) / 1.5)`. 소스 컵 각속도 1.0 rad/s 초과 벌점.
3. **`src_speed` · `src_spin` · `rcv_motion`** (L406-412) — 소스 선속도(0.15 m/s)·각속도(1.5 rad/s), 리시버 선속도(0.05)·각속도(0.5) 한계.
4. **`act_flip` · `hand_flip`** — 연속 명령 부호가 뒤집히는 떨림 벌점(`relu(-(a * a_prev))`). FLIP_CARRY_W 0.6, HAND_FLIP_W 0.4. 운반 중에만.
5. **`palm_rate` · `hand_rate`** — 운반 중 가중 상향(PALM_RATE_CARRY_W 0.05, HAND_RATE_CARRY_W 0.02).
6. **`overtilt`** — 목표각 초과 선형·비포화 벌점(OVERTILT_ABOVE 0.25 rad, W 6.0).
7. `spill_delta = -SPILL_W(50) * settled * clamp(d_spill, min=0)` — **`settled` 게이트**가 있다.
8. 위 제어 벌점 전부를 `ramp = _step(adr_progress / 0.2)` 로 서서히 켠다.

## 4. ★제약

- **이 트랙 전용 컨텍스트(`tasks/pour_fabric_mimic/t2r/context.py`)에는 `adr_progress` 가 없다.**
  §3-8 의 램프를 그대로 옮길 수 없다. 다른 게이트가 필요하다(예: 양손 파지·들기 성립 이후, 또는 붓기 단계에서만).
  `premature_tilt` · `src_tilt_toward_rcv` · `pour_dir_xy` · `src_pour_lip_pos` 도 없다 — iter_09 처럼 `src_cup_up`·`rcv_cup_mouth_pos`·`cup_radius` 로 직접 계산한다.
- 이 트랙에 있는 것: `d_in_target` · `d_spill` · `src_cup_ang_vel` · `src_cup_lin_vel` · `rcv_cup_ang_vel` · `rcv_cup_lin_vel` · `actions` · `prev_actions`.
- ★**fresh 에서 제어 벌점을 전체 가중으로 켜면 죽는다** — 형제 트랙 i21 이 epoch 26 에 폐기됐다. 이번은 성공률 0.4 인 정책에서 웜스타트하므로
  조건은 다르지만, 새 벌점이 기존 붓기를 한 번에 무너뜨리지 않도록 가중을 단계적으로 잡는다.
- 액션 24 = src palm 6 [0:6] + src hand 6 [6:12] + rcv palm 6 [12:18] + rcv hand 6 [18:24]. 형제 트랙과 슬라이스가 다르다.

## 5. [가설] 흘림의 원인

재생에서 도달 70 % · 흘림 28 % 인데 기울기는 101° 로 목표를 넘는다. 원인 후보는 세 가지이고 **아직 측정하지 않았다**(trace 없음):
(가) 한 번에 많이 쏟는다 — 유량 제한이 없다(§2 `pour_delta` 상한 없음).
(나) 컵을 빠르게 휘두른다 — 각속도 벌점이 없다.
(다) 명령이 떨려 주둥이가 흔들린다 — `action/step_delta` 0.44 평평이 이를 시사한다.
세 가지 모두 iter_23 의 장치(§3-1, 2~3, 4~5)가 각각 겨냥한다.

## 6. 다음 보상이 해야 할 것

1. ★**유량을 제한한다** — `pour_delta` 에 스텝당 상한을 두고 초과분에 벌점(`flow_excess`). 천천히 붓는 것이 빨리 쏟는 것보다 이득이어야 한다.
2. ★**흘림 벌점을 1순위로 올린다** — 흘림이 성공 보너스·이송 보상보다 비싸야 한다. 지금은 성공률만 올리면 흘림이 있어도 이득이다.
3. **컵 운동을 부드럽게** — 소스 컵 각속도·선속도 한계(`rot_speed`·`src_speed`·`src_spin`), 리시버 정지(`rcv_motion`).
4. **명령 떨림을 줄인다** — `act_flip`·`hand_flip`, 운반 중 `palm_rate` 상향.
5. **과기울임 억제** — `overtilt`. 재생 기울기 101° 가 필요 이상인지 목표각과 비교한다.
6. 위 3~5 는 §4 의 제약대로 `adr_progress` 없이 게이트한다.

## 7. 건드리지 말 것

- 파지·들기·converge·align·방향(`diag_mis` 1.03°)·기울기 연동 높이 예산은 iter_09 에서 성립했다. 가중·형태를 크게 바꾸면 회귀한다.
- env 무변경. `entropy_coef` 0 유지.
