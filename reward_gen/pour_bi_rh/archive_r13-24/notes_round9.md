# round 9 operator notes — t2r_rh2_i08 (iter_08) 종료 + 결정론 트레이스 + 형제 트랙 대조

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

## 0. 결론 한 줄

접근은 이미 충분하다. **막힌 것은 기울임 각도이고, 원인은 주둥이 높이 게이트가 기울기에 연동되지 않아
더 눕힐수록 손해가 되는 구조다.**

## 1. iter_08 이 고친 것과 남은 것

방향 항은 작동했다. `diag_theta_deg`(리시버 쪽 부호 있는 각) 45.3°, `diag_mis_deg` 7.8° 로
i07 의 방향 무구속 상태보다 낫다. 그러나 `pour_pose` 는 런 전체에서 정확히 0.0,
`pour_tilt` 0.0008, `episode_success` 0.0 이다. 런은 e374 에서 사용자 판정으로 종료했다.

## 2. 결정론 트레이스 (ep350, 16 env, 900 스텝, Traceback 0)

### 2-1 접근은 된다
- 컵 xy 거리 0.325 m → **0.107~0.117 m**, 이후 780 스텝 평탄.
- 컵 3D 거리 0.205~0.219 m, 소스가 리시버보다 **+0.16 m 위**.
- `diag_lip_xy` 후반 평균 **0.0894 m**, 전체 표본의 **26.3 %가 0.06 미만**,
  env 별 최솟값 0.014~0.125 m.

### 2-2 기울임이 안 따라온다 — 두 조건이 한 번도 동시에 성립하지 않았다
- `lip_xy < 0.06` 인 2101 표본: `theta` 평균 **56.1°**, `theta > 75°` 비율 **0.0000**.
- 전체 후반에서 `theta > 75°` 는 11.0 % 있는데, 그중 `lip_xy < 0.06` 인 것은 **0 개**.
- `lip_xy < 0.06 AND theta > 75°` 동시 표본 = **0 / 8000**.
- 결과: `pour_tilt` 8000 표본 전부 정확히 0, `pour_pose` 0.
- 비드: `in_target` 900 스텝 내내 **0.000**, `spill` 0 → **0.122**, `in_source` 1.000 → 0.878.
  즉 비드는 멀리서 기울일 때만 나가고 전부 바닥으로 간다.

### 2-3 방향 오차는 평균이 가리고 있었다
학습 로그 `diag_mis_deg` 평균은 7.8° 지만 트레이스 분포로는 **20° 초과 53.4 %, 40° 초과 36.8 %**,
`mis_tilt` 벌점도 -0.58 로 학습 로그(-0.11)의 5 배다.

### 2-4 액션 포화와 팔 추종
- 후반 500 스텝 소스 palm **y 액션 평균 +0.993, |a|>0.95 가 97.6 %**, **pitch 평균 +0.977, 92.0 %**.
- 팔 관절속도는 후반 소스 0.28~0.37 · 리시버 0.20~0.24 rad/s 로 거의 정착. 추종 중이 아니라 멈춘 것이다.
- `fabric/*_palm_err`: 앵커 자세(스텝 30)에서 양팔 **0.002 m · 0.88°** 로 완벽하다가,
  정책이 델타를 주는 스텝 60 부터 **리시버 0.14~0.25 m · 소스 0.05~0.11 m**, 회전 20~35° 로 벌어져
  끝까지 안 줄어든다. i07 도 같았다(소스 0.106~0.175, 리시버 0.192~0.243).
- 이 지표에 상수 오프셋은 없다. 부팅 로그에서 fabric FK 대 USD palm 정합이
  **소스 0.14 mm · 리시버 0.12 mm**, `fab→env` [0, 0, -2] mm 다.

## 3. 형제 트랙 pour_fabric(t2r_i18/i22) 대조

### 3-1 팔 델타 박스는 원인이 아니다 — 확대안 폐기
두 트랙 cfg 가 사실상 같다. 소스 y 상한 **+0.40**, 리시버 y 하한 **-0.40** 이 **동일**하고
z 하한만 -0.12(pour_fabric) vs -0.18(mimic) 로 다르다. 형제 트랙은 **같은 박스로**
`in_target` 0.70 · `episode_success` 0.59 를 냈다. 형제 트랙 `trace_i07_ep350` 기록에도
"작업공간 한계 아님: 소스 palm y 최대 +0.166, 리시버 최소 -0.166 까지 도달 가능" 이 남아 있다.
팔이 지령의 절반에서 멈추는 것은 양쪽 공통이고 그 상태로도 붓기는 성립한다.

### 3-2 거리는 이미 충분하고 벌어지는 것은 각도뿐이다
형제 트랙 i18 첫 이송 성공(epoch 1418) 대 이번 트레이스:

| | pour_fabric i18 e1418 | pour_bi_rh i08 ep350 |
|---|---|---|
| 컵 거리 | 0.233 m | **0.205~0.219 m** |
| 주둥이 거리 | 0.060 m | 0.089 m (26 % 는 0.06 미만) |
| 소스 기울기 | **77°** (수렴 후 100~112°) | **52~69°** |
| in_target | 0.492 | **0.000** |

컵 거리는 이쪽이 더 가깝다. 벌어지는 것은 기울기 하나다.

### 3-3 형제 트랙의 학습 순서는 접근 먼저, 기울임 나중이다
i18 fresh 진행: e879 주둥이 0.069·기울기 14.5° → e1147 주둥이 **0.055**·기울기 27.6°·이송 0
→ e1418 주둥이 0.060·기울기 **77.0**·이송 **0.492**. 주둥이를 먼저 붙이고 기울기가 나중에 뛴다.
iter_08 은 `tilt_aim` 을 거리와 무관하게 줘서 이 순서를 뒤집었다(§4-2).

### 3-4 형제 트랙은 높이 예산을 기울기에 연동한다 ★이번 라운드의 핵심
`reward_gen/pour_bi/iter_22/compute_reward.py` L238-242 주석 그대로:
"height budget grows with the tilt: the lip drops r*sin(theta) below the rim centre, so a cup
that must turn further has to ride higher. **Rotating and raising together must not be taxed.**"
구현은 `hover_free = HOVER_FREE0 + _LIP_R * sin_t`, `lip_free = LIP_Z_FREE0 + _LIP_R * sin_t`.

### 3-5 형제 트랙에는 palm 포화 벌점이 있다
`palm_sat_src` / `palm_sat_rcv` 가 있고 `ramp = _step(adr_progress / ADR_RAMP_FULL)` 로 램프한다.
형제 트랙도 i09 에서 "소스 팔 바깥 포화(py -0.99)" 를 겪고 i10 에 이 항을 추가했다.
★단 fresh 에서 전체 가중으로 켜면 죽는다 — i21 이 epoch 26 에 폐기됐다
("control penalties applied at full weight from step 0 of a fresh run").
★★**mimic 트랙 전용 컨텍스트에는 `adr_progress` 가 없다**(`episode_progress` 만 있다).
같은 램프를 그대로 옮길 수 없다.

## 4. iter_08 코드에서 확인된 원인

### 4-1 주둥이 높이 게이트가 고정 밴드다 ★
- `lip_dz = src_cup_mouth_pos[:,2] - lip_r * sin(theta) - rcv_cup_mouth_pos[:,2]` (L151)
- `LIP_Z_LO, LIP_Z_HI = 0.0, 0.02` (L213), `LIP_Z_FREE, LIP_Z_SIGMA = 0.12, 0.10` (L214) — **전부 상수**.
- `lip_above_raw = clamp((lip_dz - 0.0) / 0.02, 0, 1)` (L280)
- iter_08 에는 `sin(theta)` 가 L151 한 곳에만 있고 높이 허용치에는 없다.

더 기울일수록 주둥이가 `r·sin(theta)` 만큼 내려가 고정 밴드를 벗어나고, `lip_above` 가 죽으면서
`tilt_near`(L381)와 `pour_tilt`(L383)가 같이 꺼진다. **더 눕히는 것 자체가 손해다.**

정책은 이걸 스스로 상쇄하려 한다. 트레이스에서 기울기 구간별 컵 높이차가
theta 0~40° 0.164 m → 40~60° 0.152 m → 60~75° **0.176 m** → 75° 이상 **0.205 m** 로 올라간다.
높이 예산을 벌려고 컵을 올리는데 보상은 고정 밴드로 깎는다.

### 4-2 `tilt_aim` 이 사실상 거리와 무관하다
`lead_dist = LEAD_FLOOR + (1-LEAD_FLOOR) * conv_shape` (L374) 인데 `conv_shape` 는
`d_cups_xy` 기준이고 `CONV_NEAR = 0.16` 에서 포화한다(L356-357). 컵이 이미 0.11 m 라
`lead_dist` ≈ 0.985 로 상수다. 멀리서 기울여도 `tilt_aim` 을 거의 다 받는다.

### 4-3 통로 상한이 관측된 천장과 일치한다
`allowed` (L302-304) 는 `safe` 가 닫힌 상태에서 약 **74°** 에서 멈춘다
(TILT_FREE_TABLE 0.15 rad + theta_pre 55° + approach_gate 0.97 ×(75-55)°).
트레이스 최대 기울기 **69°** 가 이 통로 바로 아래다. `early_tilt` 가 그 위를 벌한다.

## 5. 다음 보상이 반드시 해야 할 것

1. ★**높이 허용치를 기울기에 연동한다.** `LIP_Z_LO`·`LIP_Z_HI`·`LIP_Z_FREE` 에
   `cup_radius · sin(theta_eff)` 를 더해, 더 눕히려고 컵을 올리는 행동에 벌점이 붙지 않게 한다.
   형제 트랙 iter_22 L238-246 이 참조 구현이다. 이것이 69° 천장을 여는 직접 수정이다.
2. `tilt_aim` 에 실제 거리 인자를 물린다. 지금의 `lead_dist` 는 `CONV_NEAR` 0.16 에서 포화해 상수다.
   주둥이 거리(`lip_xy`)를 직접 쓰거나 포화점을 0.06 근처로 내려 **접근 먼저, 기울임 나중** 순서를 만든다.
3. palm 액션 포화 벌점을 넣되 `adr_progress` 가 없으므로 다른 게이트를 쓴다
   (예: 양손 파지·들기 성립 이후에만, 또는 `episode_progress` 기준). fresh 전체 가중은 금지(§3-5).
4. 방향 오차는 평균 7.8° 가 아니라 20° 초과가 절반이다(§2-3). `mis_tilt` 의 `MIS_FREE`·`MIS_SCALE` 을
   이 분포에 맞춰 다시 잡는다.

## 6. 건드리지 말 것

- **env 는 바꾸지 않는다.** 팔 델타 박스는 형제 트랙과 같고 그 박스로 성공한 전례가 있다(§3-1).
- 파지·들기·converge·align 은 이번 런에서 성립했다(파지 src 0.798 · rcv 0.770, 컵 거리 0.21 m).
  가중·형태를 크게 바꾸면 회귀한다.
- `entropy_coef` 는 0 유지, sigma 는 손 12축만 다루는 기존 방침 유지.
- mimic 트랙 전용 컨텍스트에 없는 필드를 쓰면 static_check 에서 죽는다. 공용 컨텍스트의
  `src_tilt_toward_rcv`·`pour_dir_xy`·`premature_tilt`·`src_pour_lip_pos`·`adr_progress` 는 **없다**.
  방향·림 기하는 iter_08 처럼 `src_cup_up`·`rcv_cup_mouth_pos`·`cup_radius` 에서 직접 계산한다.
