# pour_bi_rh 보상 설계 브리프 (2026-09-29 판 — 4번째 사이클 iter_03, full-joint 제어·팔 절대 목표)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. **이 문서는 매 라운드 생성기 프롬프트에 통째로 들어간다.**

이 트랙은 세 번 리셋했다. 1~12 라운드 `archive_r1-12`, 13~24 라운드 `archive_r13-24`, 3번째 사이클 iter_00~09 는 `archive_r25-34`.
지금은 4번째 사이클의 iter_00 이다. **바뀐 것은 제어다**(palm 지령 + fabric → 양팔 full-joint). 보상은 3번째 사이클 마지막 코드(iter_09)를
바탕으로 구조를 다시 짠다 — 새로 발명하지 않는다. 사용자 판단(09-28): "제어만 바뀌지 reward 문제라고 생각하진 않는다."

## 1. 과제와 고정 조건 (바꿀 수 없다)

- 오른팔(소스, `src_*`)이 비드 20 개가 든 컵을, 왼팔(리시버, `rcv_*`)이 빈 컵을 잡아 든다. 소스를 기울여 비드를 리시버 컵에 옮긴다.
- 성공(env 가 계산, 보상이 재정의 못 함): 담김 ≥ 0.80 · 흘림 ≤ 0.25 · 두 컵 파지 유지 · 받는 컵 ≤ 20° · 20 스텝 유지 · 컵이 포개지지 않음.
- **액션 26 = 팔마다 13** — [팔 관절 절대 목표 7(a=0 시작 자세·±1 관절 한계, 0.3 rad/s 상한 — iter_03 부터), 손 구동 관절 절대 목표 6]. 로봇 설명 절이 정의한다.
- **물리 이상은 env 가 종료한다(iter_03 부터)**: 컵 > 3 m/s·> 30 rad/s, 손 접촉 > 200 N 2 스텝, 컵이 스폰에서 0.5 m 밖, 컵 낙하, mimic 폭주. 종료 벌점 항은 두지 않는다.
- 스폰(컵 원점, env-local): 소스 (0.382, −0.159, 0.268) · 리시버 (0.378, +0.165, 0.268). 시작 손바닥은 컵 원점에서 수평 약 16 cm·위 약 15 cm.
- r = `ctx.cup_radius` = 0.028 m. 소스 컵 원점 → 림 0.054 m. 제어 주기 1/60 s, 에피소드 900 스텝.

## 2. 단계 원칙 (사용자, 09-28)

접근 → 손가락 사이로 컵 → 파지 → 들기 → 붓기 순서로 **gradient 가 다음 단계로 이어져야 한다.
파지를 하고 들려고 할 때 보상이 떨어지면 안 된다.** 한 단계 앞으로 가는 것이 그 자리에 머무는 것보다 항상 스텝당 이득이어야 하고,
머물기만 해도 벌리는 자리(주차)가 없어야 한다. 검사는 `reward_gen/pour_bi_rh/stage_ledger.py`(결정론 상태 원장)로 한다.

## 3. 측정으로 확정된 물리·기구학 사실 (제어와 무관)

| # | 사실 | 근거 |
|---|---|---|
| K1 | 컵이 기우는 방향은 손바닥 yaw 에 묶인다((기우는 방위 − palm yaw) 중앙 −4°, SD 19°). | i18 ep800 |
| K2 | 60° 이상 기울 때 기우는 방위 중앙 +68° — 소스가 부을 수 있는 쪽은 자기 앞-왼쪽. | i18 ep800 |
| K3 | 손이 낮으면 손목 관절이 먼저 한계에 닿아 회전이 안 난다. palm z 0.29~0.42 에서 회전 오차 143~151°, 0.69~0.76 에서 10~23°. | 도달성 프로브 |
| K4 | 붓는 각은 각도 처방 없이도 찾는다(결정론 최대 87°·방출 91°). 가득 찬 컵은 첫 알 약 70°, 86° 에 절반. | i18 |
| K5 | 기울이면 주둥이가 r·sinθ 만큼 내려간다 — 높이 조건은 주둥이로 본다. | 자산 |
| K6 | 컵은 한쪽 접촉 0.3~0.4 N 에 넘어진다(바닥 지름 3.8 cm 테이퍼, 0.134 kg). | 프로브 |
| K7 | **인벨롭 파지 기하(이 로봇)**: 왼손이 5 손가락으로 감싸 들어 올린 34k 스텝 중앙 — 손바닥-컵 축 1.5 r(표면에서 1.4 cm), 높이 원점 +0.15 × 입구, 법선·손가락 축 수평(z −0.01·−0.11), 손바닥→컵 축 수평 방향 = 0.67 × 법선 + 0.73 × 손가락 축, 손바닥 접촉 0. | i24 ep500 재생 |

## 4. 3번째 사이클(iter_00~09)에서 확정된 교훈

| 라운드 | 사실 | 교훈 |
|---|---|---|
| i03 | 표면 거리 접근은 입구 원판을 표면으로 쳐서, 손바닥이 입구 위에 있으면 거리 1.7 cm·접근 수입 0.75 | 접근은 **컵 원점 기준**, 옆에서(수평 성분 가중) |
| i04 | 깔때기 목표가 "잡는 높이보다 4.5 cm 위에서 도착 후 내려감"이라 위에서 끼우는 동작을 가르쳤다, 파지 0 | 손바닥을 먼저 내리고 테이블과 수평으로 접근 |
| i07 | 손을 쥔 채(폐쇄도 0.63) 다가갔다 | 다가가는 동안 손을 펴게 하는 항이 필요 |
| i08 | 상태형 "편 손" 가산이 컵 12 cm 앞 주차 자리를 만들었다(0.13~0.16/스텝) | 편 손 가산도 **차분형**(편 채 다가갈 때만) |
| f00 (4번째 사이클 첫 iter_00, full-joint) | e0~470: 접근 보상(손바닥 속도 차분) −0.001~−0.018, 손바닥-컵 21~43 cm 왕복, 컵 근처 0(i09 같은 epoch 0.41), 손 폐쇄도 0.59~0.89 유지, 팔 제어 벌점 −0.14~−0.19(가장 큰 항), 팔 목표가 관절 한계 2 % 안 0.02→0.17. 결정론 검사: 시작 자세 6 cm 다가감 +0.002, 15→6 cm −0.122, 편 손−쥔 손 0, 팔 0.3 rad/s 비용 0.378 | **관절 제어에서는 속도 차분 접근이 신호가 되지 않는다** — 상태형 접근과 작은 제어 벌점(cup_grasp) |
| f03 (iter_02, 증분 팔) | e5414: 왼손 근처 0.86·쥠 0.66~0.75 — 파지 자세는 위에서 세로 집기(손가락 축 z −0.95, 엄지 6.0·검지 3.3 N, 손바닥 0). 들기 보상 e3300~3500 0.53 뒤 0: 트레이스에서 든 10 env 중 9 곳이 집기 미끄러짐(총 2~3.5→0.3~0.8), 쥔 채 든 1 곳은 2.61→3.04. 오른손: ep1750 옆 접촉에 컵 16.7° 기울고 교란 −0.24→−0.38 뒤 물러남, ep5400 팔 6/7 관절 |mu| 1.4~2.4 로 한계 96 % 고착 | **닫기·파지가 손 방향과 무관하면 방향 인자를 입구에만 걸어도 세로 집기를 산다.** 두 손가락 집기는 들 때 미끄러진다. 파지 직전 상태 수입이 크면(파지/직전 1.38 배) 파지 판정 전에 머문다 — 2번째 사이클 iter_04 는 1.86 배 |
| i09 | e300~450 오른손 컵 근처 0.43→0.54, **파지 0.06→0.11** 뒤 e450~500 붕괴(근처 0.15·파지 0.02). 같은 구간 grasp+close +0.23 대 cup_disturb −0.26·contact_push −0.08·rush −0.06, 스텝 보상 0.60→0.49 | **파지 시도 벌점이 파지 수입을 넘으면 컵을 떠난다.** `cup_disturb` 의 slide 는 (1−lifted)·tanh(스폰에서 밀린 거리/3 cm) 상태형이라, 한 번 민 컵은 손을 떼도 들기 전까지 매 스텝 비용을 낸다(원장 0p: 0.51 → 0.22) |

이전 사이클 핵심(유지): 접근은 차분, 붓기 허가는 컵 반지름 창(i13) · 방향 신호 필수(i08·1차) · 관절 한계 직접 강벌점 금지(i15, corr +0.90) ·
진행형 항의 비대칭 clamp 금지(i20) · 기울임 수입이 없으면 들고 서 있는다(i21) · 조준 항이 축 회전까지 지불하면 안 된다(i22) ·
주차 함정: 상태 비례 수입이 배달보다 크면 붓지 않는다(회전·배달 수입이 성공 궤적 스텝 수입의 절반 이상).

## 5. iter_00 요구 — full-joint 에 맞춘 재편 (09-28 두 번째 판)

사용자(09-28): "이전과 달리 full-joint 방식이니까 reward 를 재개편해." 첫 iter_00(t2r_rh4_f00)은 palm 지령 시절의 형태를 그대로 옮겨 실패했다(4절 f00 행).

### 5-0. 원칙: 관절 제어에서는 **상태에 기울기가 있어야** 한다
- palm 지령 시절에는 액션 = 손바닥 속도라서 "컵 쪽 속도" 같은 차분(속도) 보상이 액션과 곧바로 이어졌다. full-joint 에서는 손바닥 속도가
  관절 7개의 비선형 합이고 EMA 로 약 10 스텝 늦다. 무작위 탐색에서 다가감과 멀어짐이 상쇄돼 기댓값이 0 이다.
- 그래서 접근·편 손은 **상태형**(현재 자세의 함수)으로 준다: 가까울수록, 옆일수록, 손이 펴져 있을수록 스텝당 더 받는다.
- 주차(i08)는 차분형으로 막지 않고 **단계 원장**으로 막는다: 멈춰 있는 어느 자리도 다음 단계 자리보다 싸야 한다.

> **iter_03(09-29 사용자 결정 B)**: 5-1 의 손바닥 원점 기준 접근·입구 대신, 2번째 사이클 iter_04(양손 파지 첫 성공)의 손끝 기준 파지 사다리를 쓴다. 세부는 뒤의 iter_03 노트.

### 5-1. 접근·입구 (같은 관절 제어로 컵 집기에 성공한 cup_grasp 구조를 따른다)
- 원점 기준 거리 보상: 먼 척도 + 가까운 척도의 합(예: 1−tanh(d/먼 척도), 1−tanh(d/가까운 척도)), 높이 차는 수평의 2 배로 센다(옆·잡는 높이 선호,
  위에서 내려오는 경로는 덜 받는다). 무게는 들기 전 합 예산 안에서. 파지가 서면 사라진다.
- 편 손: 컵에서 떨어져 있는 동안 거리 보상에 손 펴짐 인자를 곱한다(예: 0.45 + 0.55 × 펴짐). 손가락 사이에 컵이 오면 인자를 풀어 닫기를 막지 않는다.
- 손가락 방향·손가락 사이(엄지 대향)·잡는 높이로 내려오기는 원거리부터 연속으로 오르는 상태형(곱 사슬 금지).
- 컵 근처에서 붙는 비용(입구 위 정지·손바닥 밀어 넣기·급한 팔)이 **다가가는 수입보다 작아야** 한다 — 15 cm → 6 cm 가 스텝당 오른다.
- 원장 통과 조건: 0 < 1 < 1b(옆 12 cm 편 손 정지) < 2 < 3 < 4 < 5 ... (단조), 3p ≥ 3, 0p ≥ 0, 1h(반쯤 쥔 손) < 1b.

### 5-2. 제어 벌점은 cup_grasp 수준 (매끄러움은 EMA·속도 상한이 맡는다)
- cup_grasp: 액션 변화율 0.02·tanh(평균 변화²/0.1) + 팔 속도 0.12·tanh(|qd|/1 rad/s) + 3 rad/s 초과 폭주 비용.
- **iter_03 부터 팔 액션 크기 비용 금지**: 팔 액션이 절대 목표(a=0 = 시작 자세)라 |a| 비용은 시작 자세로 끌어당기는 항이 된다.
- 기준: 무작위 ±1 액션 비용 ≤ 0.06/스텝, 팔 0.3 rad/s 비용 ≤ 0.08/스텝. 손 채널 ±1 은 비용 0(절대 목표의 정상 명령).
- 액션 반전(flip)·|a|>0.9 포화 벌점은 관절 증분에서는 탐색을 억누른다 — 두지 않는다.

### 5-3. 유지하는 것 (첫 iter_00 에서 검증됨)
- 교란 벌점은 컵이 **움직이는 속도**로만(밀린 상태 자체는 비용 없음). 파지 전 작은 기울기는 싸게.
- 파지(닫기 1.0 → 양쪽 접촉 2.0), 들기(소스 높게·리시버 낮게, W 5.5), 배치·조준·회전·배달(붓는 중 회전·배달 몫 ≥ 절반), 흘림·조기 기울임·충돌.
- 들기 전 상태 합 < 들기 빠른 성분(골 없음). 항 25 개 이하.

### 5-4. 하지 말 것
- 손바닥 속도에만 의존하는 접근 수입, 목표 각도·대역·각도 상한 처방, 스폰 기준 만남 통로, 관절 한계 강한 직접 벌점, 새 절대 거리 상수(높이 허용과
  접근 척도는 컵 반지름 배수 또는 명시 근거), 자산 치수 하드코딩, 곱 사다리, 손 채널 포화 벌점.

## 6. 평가

- 틱은 단계 관점: near_rate → cup_in_pocket·oppose → closure → grasped → cup_lift → converge·aim → pour. 단계가 오를 때 보상·에피소드 길이가 떨어지면 골.
- full-joint 지표: `joint/{src,rcv}_arm_limit_frac`, `_arm_track_err_deg`, `_hand_track_err_deg`.
- 성공 판정은 결정론 평가(담김 ≥ 0.8·최대 기울기 순간 방향 오차 < 30°·주둥이가 입구보다 위).

## 7. 시작

fresh(액션·관측 의미가 바뀌었다). 웜스타트 체크포인트 없음. seed 42.
# iter_03 (4번째 사이클) 운영 노트 — 2026-09-29

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다(K7 인벨롭 기하, f03 행).

## 사용자 결정 (09-29)
- 보상 B: 2번째 사이클 iter_04(런 t2r_rh2_i04, 이 트랙에서 양손 파지가 처음 선 보상)의 손끝 기준 파지 구조로 돌아가고,
  K7 인벨롭 방향과 속도형 밀림 벌점을 더한다.
- 팔 액션은 절대 목표(a=0 시작 자세·±1 관절 한계, 0.3 rad/s) — 로봇 설명 절. PPO bounds_loss 0.005. 물리 이상은 env 가 종료한다.

## 측정 사실
- t2r_rh2_i04(iter_04, 손바닥 지령·시너지 손, i03 ep2000 이어받음): 왼손 파지 0 → 0.54, 양손 0.52/0.54(e1550). 결정론 4/4 env 양손 파지 유지,
  엄지 3~5 N · 나머지 합 2.5~7 N · 손바닥 0.
- 같은 원장(손끝도 손바닥과 같이 이동)에서 iter_04: 파지(양쪽 1.5 N, 판정 켜짐) 7.82 / 벽에 닿고 닫힘 0.7 4.20 = 1.86 배.
  iter_02: 9.96 / 7.23 = 1.38 배. iter_04 는 손 방향을 보지 않는다(2f·2v = 2, 2.15).
- iter_04 의 깔때기는 파지점보다 4.5 cm 위에 먼저 도착하게 한다(ARRIVE_RISE): 원장 1 절반 접근 1.10 → 1b 옆 12 cm·잡는 높이 0.86 (골).
  3번째 사이클 i04 에서도 같은 형태의 깔때기가 '위에서 끼우기'를 가르쳤다(브리프 4절).
- iter_04 의 slide 는 스폰에서 밀린 거리 상태형이라 원장 3p 2.62 < 3 2.90 (골). i09 파지 붕괴와 같은 형태(브리프 4절).
- f03(iter_02): 왼손 세로 집기(손가락 축 z −0.95, 엄지·검지만), 들 때 10 곳 중 9 곳 미끄러짐. 오른손은 옆 접촉 뒤 물러남.
- iter_04 원문(파지 전·들기 부분, 붓기 부분 제외):

```python
def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (0/336 knock-overs at +45 mm) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm, inside the 0-25 mm lift group)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin (no lift with thumb >= 50 mm)

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    # gap is zero only at the measured contact radius (tip frame 26-28 mm, cup outer 28.5 mm);
    # the old +10 mm pad made 33-37 mm count as "on the wall" and killed the gradient
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)        # (N,)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values   # nearest non-thumb tip, (N,)

    # opposition: 1 when the thumb and the other fingers sit on opposite sides of the cup axis
    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # (N,)

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup_terms(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok, lift_target):
    """Unweighted per-arm terms, each in [0, 1]. pose_ok (N,) = cup attitude is acceptable for this stage."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: long-range funnel term + per-side fingertip-on-wall term.
    #    Each side is scored on its own (sum form) so the thumb keeps its full gradient while the finger
    #    side is already done, and measured force completes a side: reach saturates only with contact
    #    force on BOTH sides (the parked pose of i01 had thumb 0.00 N and reach 0.97).
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: balanced two-sided pinch. Before the flag, finger-first touch earns 0.15 and any thumb force
    #    added to it can only raise the value (0.5 N -> +0.19); thumb-only still earns nothing.
    #    Flag boundary (1 N / 1 N): pre_grasp 0.355 -> flagged 0.75, a step up, never a valley.
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

    # closure shaping: measured grasps close to 0.68-0.98, the parked hand sat at 0.19 with the closure
    # channels driven open; pays only once the pinch centre is at the grasp pose (contact freeze makes
    # closing there harmless), saturates at 0.7 so over-squeezing is not rewarded
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift: measured cup elevation only, clamped, and only while really held (explosion-proof)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)                # 5 mm dead band: tipping on the rim is not lifting
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * torch.exp(-(geo["d_grip"] / 0.08) ** 2) * calm
    lift_prog = 0.5 * torch.tanh(h_pos / 0.02) + 0.5 * torch.clamp(h_pos / lift_target, max=1.0)
    lift = held * pose_ok * lift_prog
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)            # "held and at least 5 cm up"

    # penalties (unweighted, bounded)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)       # 1 N dead band: a slightly stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)             # finger-only contact: adding thumb force lowers this
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "lift": lift, "up_w": up_w, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling,
            "held": held, "far": torch.tanh(4.0 * geo["d_funnel"])}
```

## iter_03 에서 할 것 (이전 코드 iter_02 위의 변경)
1. 팔마다 들기 전 구조를 iter_04 형태로 바꾼다: 손끝 기준 접근(집는 중심 → 파지점 + 엄지·나머지 손끝-벽 거리, 한쪽씩 합, 힘이 닿으면 완성, 대향 가중),
   파지는 **env 파지 판정(엄지 > 1 N AND 다른 손가락 > 1 N) 뒤에** 크게 주고 판정 전은 작은 사전 항(판정 경계는 오르는 계단, 골 없음),
   닫기는 파지점 근처 × 폐쇄(작은 무게), 들기는 held = 판정 × 파지점 근처 × 컵 속도 차분.
2. 깔때기는 **옆·잡는 높이로 도착**한다(파지점 위 도착 없음). 접근 거리는 컵 원점 기준·높이 차 가중(브리프 5-1 원칙).
3. K7 인벨롭 방향 점수(iter_02 의 env_score)를 **접근·닫기·파지 모두에** 바닥 있는 인자로 건다(들기에는 걸지 않는다 — 들기 경계 연속).
4. 밀림 벌점은 속도형만(iter_02 형태). iter_04 의 밀린 거리 상태형은 쓰지 않는다.
5. 두 팔의 들기 전 구조·무게는 같게(역할별 떠남 벌점 없음). 붓기 이후(배치·조준·회전·배달·흘림·충돌)는 iter_02 그대로.
6. 팔 액션 크기 비용은 두지 않는다(로봇 설명 절). 변화율·팔 속도 비용은 유지.
7. 검사 통과 조건: 원장 단조(0 < 1 < 1b < 2 < 3 < 4 < 5 < 6), 파지 5 / 벽에 닿음 4 ≥ 1.6, 5v(같은 파지 힘·세로 손) < 5 − 1.0,
   2f·2v < 2, 3p ≥ 3, 0p ≥ 0, 1h < 1b, [35] 인벨롭 > 손끝 겨눔·수직 손, [37] 팔 액션 크기 비용 0, [15] 원거리 접근 기울기 > 0, [33] 제어 비용.
