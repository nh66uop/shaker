"""목표에서만 터지는 외란 + 성공 연동 ADR — 순수 torch 함수(isaaclab·isaac 금지, CPU 단위테스트 가능).

09.23 사용자 결정. `cp_l_m08` 정책은 컵을 들고 똑바로 세워 나르지만 **목표에 안착하지 않고 든 채로 맴돈다**.
"목표에 머무르기"가 배울 값어치가 있는 기술이 되려면 그 자리에 외란이 있어야 한다 — 그래야 머무르는 데
견고한 파지가 필요해진다. 외란은 정책이 잘할수록 커진다(ADR).

★게이트 = 부모의 **높이 래치**(`_lifted`) ∧ **목표 근방**. 높이 래치를 빼면 안 된다(`fj_core_cfg` 의 경고):
  접촉 래치나 무게이트로 켜면 컵이 아직 상판에 있는 접근 구간에 외란이 발화해 `cup_disp` 가 approach 순벌점을
  만들고 `disp_at_latch` 오염으로 lift·transfer·success_bonus 의 감쇠 계수를 통째로 망친다.

★★회피 실패모드(`fj_core_cfg` 가 지목한 것)의 수입 비교 — **목표를 피하는 것이 이득이 되면 안 된다**.
  외란이 목표에서만 터지므로 "목표에 안 가면 외란 없음" 이라는 우회로가 생긴다. 다음 보상 라운드의
  스텝 수입은 목표에 정지해 있을 때 약 10/step(`cup_pick_l_manip/iter_08` 의 `goal_hold` 9 · `goal_reach` 3 ·
  `lift_height` 5 가 겹치는 구간) 대 그 밖에서 약 1/step 이다. 즉 회피는 **스텝당 ~9 를 버리는** 선택이고,
  외란은 W2 에서도 질량정규화 20 N/kg · 평균 발화율 ~2.6 % 라 기대 손실이 그 근처에도 못 간다.
  (W1 실측: 컵 0.134 kg 에서 |F| 기댓값 ≈ 1.07 N = 컵 무게의 0.82배.)
  같은 이유로 **목표에 있는 것 자체를 벌하는 항이 보상에 있으면 안 된다** — iter_08 을 확인했다:
  `pen_descend` 는 목표 높이에서 0 으로 사라지고, `pen_stall` 은 `on_table` 게이트라 들린 상태에서 0,
  `pen_cup_vel`·`pen_squeeze` 는 움직임·과압만 벌한다. 목표 체류에 비례해 커지는 벌점은 없다.
"""

from __future__ import annotations

import torch

#: 외란이 터지는 "목표 근방" 반경 [m] — `ctx.goal_dist`(컵 키포인트 ↔ 목표 자세 키포인트 최대거리) 기준.
#:   왜 `k·success_tol` 이 아니라 **고정 반경**인가: `success_tol` 은 0.1125 → 0.015 m 로 줄어드는
#:   커리큘럼 값이라, 거기에 묶으면 정책이 잘할수록 외란 구역이 7.5배 좁아진다 — 외란이 가장 필요한
#:   시점(정확히 안착할 수 있게 된 때)에 사라지는 역방향 커브다. 0.15 m 는 가장 느슨한 공차(0.1125)보다
#:   커서 어느 커리큘럼 단계에서도 성공 구역을 **항상 포함**하고, 물체 치수가 전혀 들어가지 않는다.
#:   실측 정합(`cp_l_m08` e3200): `task/kp_dist` 평균 0.165 · 에피소드 최소 평균 0.148 — 0.15 는
#:   "목표까지 갔던 에피소드"만 걸러내는 자리다(무차별 발화도, 영원히 0 도 아니다).
NEAR_GOAL_RADIUS_M = 0.15

#: W1(레벨 0) / W2(레벨 1) 외란 크기 [N/kg · N·m/kg] — `fj_core_cfg` 의 두 단계 그대로.
W1_FORCE_SCALE, W1_TORQUE_SCALE = 5.0, 0.5
W2_FORCE_SCALE, W2_TORQUE_SCALE = 20.0, 2.0

#: ADR 한 번에 움직이는 폭. 0→1 을 다 가려면 20회 갱신이 필요하다(아래 갱신 주기와 곱해 ~40 epoch).
ADR_LEVEL_STEP = 0.05
#: 승급 기준 — 끝난 에피소드의 "들고 목표 근방에 있던 스텝 비율" EMA 가 이 위면 올린다.
#:   근거: 현행 정책은 목표를 지나치기만 하므로(성공 공차 안 체류 `task/near_goal` 0.019) 0.15 반경
#:   체류가 에피소드의 20 % 를 넘는다는 것은 "지나치는" 것이 아니라 **머무르고 있다**는 뜻이다.
ADR_RAISE_DWELL = 0.20
#: 강등 기준 — 이 아래로 떨어지면 외란이 정책을 목표에서 쫓아낸 것이므로 되돌린다.
#:   0.08 은 승급선의 0.4배 — 히스테리시스 폭을 충분히 두어 레벨이 매 갱신마다 흔들리지 않게 한다.
ADR_LOWER_DWELL = 0.08
#: 레벨을 한 칸 움직이기 전에 모아야 할 **완료 에피소드** 수. 4096 env · ~500 스텝 에피소드에서
#:   스텝당 ~8개가 끝나므로 ~64 스텝(≈2 epoch)에 한 번 — 0→1 전 구간이 ~40 epoch 이다(느린 ADR).
ADR_EPISODES_PER_UPDATE = 512
#: 체류율 EMA 계수(완료 에피소드 묶음 단위) — 부모 `_event_ema` 와 같은 0.05.
ADR_DWELL_EMA = 0.05
#: "아직 끝난 에피소드가 없다" 센티널(부모 `_event_ema` 규약과 동일).
NO_EVENT = -1.0


def near_goal_mask(goal_dist: torch.Tensor, radius: float = NEAR_GOAL_RADIUS_M) -> torch.Tensor:
    """(N,) bool — 물체가 목표 근방인가. `goal_dist` 는 ctx 의 키포인트 거리 [m]."""
    if goal_dist.dim() != 1:
        raise ValueError(f"goal_dist: expected (N,), got {tuple(goal_dist.shape)}")
    return goal_dist <= float(radius)


def wrench_gate(lifted: torch.Tensor, goal_dist: torch.Tensor,
                radius: float = NEAR_GOAL_RADIUS_M) -> torch.Tensor:
    """(N,) bool — 외란 발화 조건 = **높이 래치 ∧ 목표 근방**. 둘 중 하나만으로는 터지지 않는다."""
    if lifted.dtype != torch.bool or lifted.shape != goal_dist.shape:
        raise ValueError(f"lifted: expected bool {tuple(goal_dist.shape)}, got {tuple(lifted.shape)} {lifted.dtype}")
    return lifted & near_goal_mask(goal_dist, radius)


def wrench_scales(level):
    """ADR 레벨 → (force_scale, torque_scale) [N/kg, N·m/kg]. W1 ↔ W2 선형보간.

    `level` 은 파이썬 float 또는 0차원 텐서(호스트 동기화를 피하려 env 는 텐서로 굴린다).
    """
    if isinstance(level, torch.Tensor):
        lv = level.clamp(0.0, 1.0)
    else:
        lv = min(max(float(level), 0.0), 1.0)
    return (W1_FORCE_SCALE + (W2_FORCE_SCALE - W1_FORCE_SCALE) * lv,
            W1_TORQUE_SCALE + (W2_TORQUE_SCALE - W1_TORQUE_SCALE) * lv)


def episode_ema(prev: torch.Tensor, val: torch.Tensor, mask: torch.Tensor,
                alpha: float = ADR_DWELL_EMA) -> torch.Tensor:
    """끝난 에피소드(`mask`)의 평균으로 EMA 를 한 칸 민다. 끝난 게 없으면 그대로, 첫 이벤트는 그 값으로.

    부모 `GraspFJEnv._event_ema` 와 같은 규약이되 계수를 인자로 받는다(단위테스트용). 새 텐서를 돌려준다.
    """
    n = mask.sum()
    m = (val * mask.to(val.dtype)).sum() / n.clamp(min=1).to(val.dtype)
    upd = torch.where(prev < 0.0, m, (1.0 - alpha) * prev + alpha * m)
    return torch.where(n > 0, upd, prev)


def next_adr_level(level: torch.Tensor, dwell_ema: torch.Tensor, *,
                   step: float = ADR_LEVEL_STEP,
                   raise_above: float = ADR_RAISE_DWELL,
                   lower_below: float = ADR_LOWER_DWELL) -> torch.Tensor:
    """성공 연동 ADR 한 칸 — 목표 체류율이 높으면 +step, 낮으면 −step, 사이면 유지. [0,1] 클램프.

    `dwell_ema` 가 센티널(음수 = 아직 완료 에피소드 없음)이면 아무것도 하지 않는다. 새 텐서.
    """
    up = (dwell_ema > raise_above).to(level.dtype)
    down = (dwell_ema < lower_below).to(level.dtype)
    moved = (level + step * (up - down)).clamp(0.0, 1.0)
    return torch.where(dwell_ema < 0.0, level, moved)
