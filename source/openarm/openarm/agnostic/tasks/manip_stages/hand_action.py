"""손 액션 → 관절 목표, **액션 0 = 리셋(편 손) 자세** 판 (09.28 사용자 결정 "1,2 모두 필요").

왜: 부모(`grasp_fj_env._hand_targets`, hand_direct)는 `raw = lo + ½(a+1)(hi−lo)` 라 액션 0 이 관절 범위 가운데다.
  리셋 자세(편 손)는 대부분 관절 하한 = 액션 −1 에 있다. 그래서 무작위 초기 정책은 손가락 굽힘 ~0.5 로 시작하고
  (cup_grasp iter_07 e134 실측 0.49~0.54), 손을 펴 두려면 손가락 13 차원 평균을 액션 한계 −1 에 붙여야 한다.
  그 결과 편 손 자세 조건이 전 라운드 0.01 이었고, 왼팔은 bounds_loss 폭증(1.5 → 681) 뒤 두 번 무너졌다(iter_06 e800, iter_07 e650).

무엇: 관절마다 리셋 자세 q0 를 액션 0 에 두는 연속 구간 선형 사상.
    a ≥ 0 :  raw = q0 + a·(hi − q0)
    a < 0 :  raw = q0 + a·(q0 − lo)
  a = −1 → lo, 0 → q0, +1 → hi 로 범위는 부모와 같다. q0 = lo 인 관절은 a < 0 쪽이 모두 lo 다.
  EMA·clamp·`_syn_close` 는 부모와 같다. 액션·관측 차원은 불변이지만 **액션의 의미가 바뀐다** — 실기 제어기도 같은 사상을 써야 한다.
"""

from __future__ import annotations

import torch


def zero_open_raw(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
    """(N,J) 액션 [-1,1] → 관절 목표. lo/hi/q0 는 (1,J) 또는 (J,) 로 브로드캐스트. q0 는 [lo,hi] 안이어야 한다."""
    a = a.clamp(-1.0, 1.0)
    up = q0 + a.clamp(min=0.0) * (hi - q0)
    down = q0 + a.clamp(max=0.0) * (q0 - lo)
    return torch.where(a >= 0.0, up, down)
