"""팔 지령 — 관절 증분 + EMA (cup_grasp · grasp_fj Track B 와 같은 법칙). torch 만 쓴다(Isaac 없이 테스트).

    q_raw = clamp(q*_{t−1} + k·a, lo, hi)
    q*_t  = clamp(α·q_raw + (1−α)·q*_{t−1}, lo, hi)          포화 slew = α·k·60 = 0.3 rad/s (k 0.05 · α 0.1)

사용자 결정(09.29): "팔 액션은 cup_grasp 처럼 관절 증분(full-joint)". 실기 `sim2real/.../joint_decoder.py` 와 같은 법칙이다.
"""

from __future__ import annotations

import torch


def step_arm_increment(q_prev: torch.Tensor, a: torch.Tensor, *, k: float, alpha: float,
                       q_lo: torch.Tensor, q_hi: torch.Tensor) -> torch.Tensor:
    """증분 한 스텝. a 는 [−1, 1] 로 자른다. 새 텐서를 돌려준다."""
    if k <= 0.0:
        raise ValueError(f"k 는 양수: {k}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha 는 (0, 1]: {alpha}")
    q_raw = torch.maximum(torch.minimum(q_prev + float(k) * a.clamp(-1.0, 1.0), q_hi), q_lo)
    q_new = float(alpha) * q_raw + (1.0 - float(alpha)) * q_prev
    return torch.maximum(torch.minimum(q_new, q_hi), q_lo)
