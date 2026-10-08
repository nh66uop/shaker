# round 8 operator notes — t2r_rh2_i07 (iter_07) 종료 관찰

확인된 사실만 적는다. 추정은 [가설] 로 표시한다.

> ## ★정정 (작성 직후 확인, 09.21)
>
> 아래 §3 과 §7 은 **틀렸다.** 내가 공용 컨텍스트
> `source/openarm/openarm/agnostic/modules/t2r/context.py` 를 읽고 썼는데,
> 이 트랙은 **트랙 전용** 컨텍스트
> `source/openarm/openarm/agnostic/tasks/pour_fabric_mimic/t2r/context.py` 를 쓴다
> (`scripts/reward_gen/t2r_rh.py:33-34` 가 거기서 import 한다).
>
> 트랙 전용 컨텍스트에는 `src_tilt_toward_rcv` · `pour_dir_xy` · `premature_tilt` ·
> `premature_tilt_limit` · `src_pour_lip_pos` · `cup_hit` · `rcv_side_margin` ·
> `*_wrap_count` 가 **하나도 없다**(참조 횟수 0). 분리는
> `pour_fabric_mimic/tests/test_mimic_contract.py::test_t2r_contract_is_track_local_not_shared`
> (커밋 3dd15927)가 의도적으로 강제한다. 이 이름을 쓰면 학습 전에 static_check 에서 실패한다.
>
> mimic env 에는 **premature-tilt 래치도, 바깥쪽 붓기 무효 규칙도 없다.**
> 성공 판정은 `pour_fabric_env.py:625` 의 네 조건뿐이다:
> in_target ≥ success_fill_ratio ∧ spill ≤ success_spill_max ∧ 컵 xy < success_xy_thresh ∧ ¬nested.
> `task/premature_tilt_rate` 키가 없는 것도 그래서다.
>
> **살아남는 것**: §1 사용자 관찰, §2 의 "보상이 방향 없는 `ctx.src_cup_tilt` 만 쓴다",
> §4·§5·§6 수치, §8 의 네 가지 요구. 방향을 쓰려면 기성 필드가 아니라
> `src_cup_up` 과 `rcv_cup_mouth_pos` 에서 **직접 계산**해야 한다.
> 무효화 래치가 없으므로 §7 의 "성공이 영구 무효" 대신 단순한 사실이 남는다 —
> 엉뚱한 방향으로 기울이면 비드가 리시버에 안 들어가고, 그래서 in_target 과 성공이 0 이다.

## 0. 이번 라운드의 결론 한 줄

붓기 전 단계(파지·들기·모으기·조준)는 전부 성립했다. 막힌 것은 **기울임의 방향**이다.
보상이 방향 없는 크기 `ctx.src_cup_tilt` 만 쓰고, env 가 제공하는 방향 필드
(`src_tilt_toward_rcv`, `pour_dir_xy`, `src_pour_lip_pos`, `premature_tilt`)를 한 번도 쓰지 않았다.

## 1. 사용자 직접 관찰 (ep1950 영상, 증거 우선순위 2)

- 소스 컵의 **회전 방향이 잘못됐다**.
- 컵 입구가 **몸쪽을 향해야 하는데 world +x 를 바라보고 있다**.

`context.py` 정의상 `pour_dir_xy` 의 x>0 은 몸 바깥이고 로봇 몸통은 −x 다. 즉 관찰된 자세는
붓는 방향이 리시버 반대쪽을 향한 상태다.

## 2. 보상 코드 사실 (직접 확인)

- `reward_gen/pour_bi_rh/iter_00 ~ iter_07` 8개 전부에서
  `src_tilt_toward_rcv` · `pour_dir_xy` · `premature_tilt` · `src_pour_lip_pos` 참조 횟수 = **0**.
- 같은 저장소의 성공한 형제 트랙 `reward_gen/pour_bi/iter_16~19` 는 같은 필드를 **4~5회** 쓴다.
  (`pour_bi` i16 ep5600 은 최대 난이도 결정적 재생 성공 92 %·in_target 91 % 로 영상 확인됨)
- iter_07 은 `tilt_lead` · `tilt_rise` · `pour_tilt` · 자세 통로 전부를 `ctx.src_cup_tilt` 로 만든다.
  이 값은 `context.py` 주석대로 **직립 대비 기울기 크기**(0=직립, π/2=수평)이며 **방향이 없다**.
  어느 쪽으로 기울여도 같은 보상이 나온다.
- iter_07 은 붓는 쪽 림 점을 `ctx.src_pour_lip_pos` 대신 직접 만든다(L124-127):
  `lip = src_cup_mouth_pos + outer_r * (src_cup_up * src_cup_up[:,2:3] - ez)`.
  이것은 림에서 **기하학적으로 가장 낮은 점**이라 기울인 방향이 어디든 정의된다. 방향 제약이 아니다.

## 3. env 가 이미 주는 방향 필드 (context.py 정의, 그대로 인용)

- `src_tilt_toward_rcv` (N,): 리시버 쪽으로 기운 **부호 있는** 각도 θ [rad].
  (d̂, z) 수직면 성분만 본다. **옆으로 기운 성분은 0, 반대로 기울면 음수.**
- `pour_dir_xy` (N,2): 붓는 방향 d̂ 의 수평 단위벡터. **x>0 = 몸 바깥, 로봇 몸통은 −x.
  x > 0.3 이면 성공 무효.**
- `src_pour_lip_pos` (N,3): 비드가 실제로 나가는 림 점. 컵이 직립이어도 정의된다.
- `premature_tilt` (N,) bool: **에피소드 래치**. 잡은 소스 컵의 `src_pour_lip_pos` 가 리시버 입구 중심에서
  xy 8.1 cm 보다 먼 상태에서 `src_cup_tilt` 가 `premature_tilt_limit` 을 넘은 적이 있으면 참.
  **리셋 전까지 유지되고 성공이 영구 무효**가 된다.
- `premature_tilt_limit` (N,): 채움별 첫 유출각 − 20°. **가득 채운 컵에서 52°**, 채움 0.5 에서 69°.
- `cup_hit`, `rcv_side_margin` (< 0.03 이면 성공 무효), `cups_nested` 도 같은 성격의 무효 조건이다.

## 4. 이번 런 수치 (t2r_rh2_i07-r3, e1978 에서 종료, 12.08 h, Traceback 0)

마지막 10 epoch 평균:

- `task/src_tilt_deg` **54.8°**. 50-epoch 평균이 400 epoch 동안 54 ± 0.7 로 평평.
  마지막 200/400 epoch 선형 회귀 기울기 +0.71 / +0.34 deg per 100 epoch.
- `task/episode_success` · `task/success_now` · `reward/success_bonus` · `reward/pour_delta`
  = **런 전 구간 0.0**.
- `bead/in_target` 0.0002 (최대 0.0007), `bead/crossed` 0.0002 (최대 0.0021),
  `bead/spill` 0.0319 (최대 0.0563), `bead/in_source` 0.963.
- `task/src_grasped` 0.828 · `task/rcv_grasped` 0.762 (둘 다 런 최고 수준).
  `task/src_closure` 0.437 · `task/rcv_closure` 0.491.
- `reward/aim_gate` 0.739, `task/aim_dist` 정상 구간 0.17~0.25 m (최소 0.149).
- `reward/tilt_lead` 1.565 · `tilt_rise` 1.732 · `pour_tilt` 0.202.
- `task/nested_rate` 0.0015 · `task/cup_collision_rate` 0.0018 · `done/drop` 0.0006 — 전부 낮다.
- `losses/entropy` 17.62 로 단조 하락 지속(entropy_coef 0 의 효과, 커밋 3b02f403).

## 5. 결정론 재생 (ep1950, 4 env, step 600 스냅샷 1개)

파지 1.00/1.00 · 들기 0.211 m · **src_cup_tilt 65.4°** · aim_dist 0.128 m ·
**in_source 1.000 · in_target 0.000 · spill 0.000 · success 0.000**.

컵을 65° 까지 기울이고 입구를 0.128 m 까지 붙였는데 비드가 **한 알도 나가지 않았다.**
(주의: 에피소드 최대각이 아니라 step 600 한 시점 값이다.)

## 6. 산술적으로 확인된 것

- iter_07 의 `pour_tilt` 는 65° 부터 시작하는데 런 평균은 54.8° 라 평균에서 `pour_frac` = 0 이다.
  그런데 로그값은 0.202 다. 따라서 이 값은 **65° 를 넘긴 소수 env 몫**이고 분포가 넓다.
- `tilt_rise` 는 65° 에서 `rise_frac` 이 1.0 으로 포화해 경사가 0 이 되고,
  `pour_tilt` 는 65→100° 를 35도에 펴 놔서 65.4° 에서 `pour_frac` 이 0.011 뿐이다.
  정책이 멈춘 각도가 두 항의 교대 지점과 겹친다.
- 런 평균 54.8° 와 재생 65.4° 는 **가득 찬 컵의 `premature_tilt_limit` 52° 를 넘는다.**

## 7. [가설] premature_tilt 래치

`bead/in_source` 가 0.96 으로 컵이 가득 차 있으므로 한계는 52° 다. 평균 기울임이 54.8° 이고
붓는 방향이 리시버 반대쪽(사용자 관찰)이면, 림 점이 리시버 입구에서 8.1 cm 안에 들어올 수 없으므로
`premature_tilt` 가 래치되어 그 에피소드의 성공이 영구 무효가 된다.
이것이 사실이면 `episode_success` 가 런 내내 정확히 0 인 것과 맞는다.

**미검증이다.** env 는 `premature_tilt` · `pour_dir_xy` 를 extras 로 기록하지 않아
tfevents 로는 래치율을 볼 수 없다(기록되는 task/ 키 31개를 전부 확인했다).

## 8. 다음 보상이 반드시 해야 할 것

1. 기울임 단계 전체(`tilt_lead` · `tilt_rise` · `pour_tilt` · 자세 통로)를 방향 없는
   `ctx.src_cup_tilt` 대신 **부호 있는 `ctx.src_tilt_toward_rcv`** 기준으로 다시 세운다.
   `ctx.src_cup_tilt` 는 과기울임 **벌점**에만 남긴다.
2. `ctx.pour_dir_xy[:,0]` 로 방향 게이트를 만든다. env 의 무효 기준이 x > 0.3 이므로
   그보다 앞에서 0 이 되어야 한다.
3. `ctx.premature_tilt` 에 명시적 벌점을 준다. 래치는 되돌릴 수 없으므로 진입 자체를 막아야 한다.
4. 기울임 허용각을 `ctx.premature_tilt_limit` 에 묶고, **림 점이 리시버 입구에 가까워졌을 때만**
   그 위로 열어 준다. 거리 기준은 손으로 만든 lip 이 아니라 `ctx.src_pour_lip_pos` 로 잰다.
5. 형제 트랙 `reward_gen/pour_bi/iter_18/compute_reward.py` 의 L216-217(dir_ok),
   L232-237(lip 기하·approach_gate·safe), L240-255(theta·theta_eff·allowed 통로),
   L402(premature_latch) 가 위 네 가지를 이미 구현한 참조 구현이다. 로봇이 다르니
   그대로 복사하지 말고 구조만 가져온다.

## 9. 건드리지 말 것

파지·들기·모으기·조준은 이번 런에서 성립했다(§4). 해당 항의 가중·형태를 크게 바꾸면 회귀한다.
`entropy_coef` 는 0 으로 두고 sigma 는 손 12축만 다루는 기존 방침을 유지한다.
