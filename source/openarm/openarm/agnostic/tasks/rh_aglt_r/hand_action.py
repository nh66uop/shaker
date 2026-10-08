"""손 지령 — RH56F1 구동 6관절(종속 6은 PhysX mimic). torch 만 쓴다(Isaac 없이 테스트).

사용자 결정(09.29): "손 액션은 pour_bi_rh 와 동일하게". pour_fabric_mimic `direct_hand_targets`(PourFJEnvCfg:
hand_direct_range=grip · hand_direct_contact_freeze=True) 와 같은 식이다.

    [lo, hi] = [min(open, grip), max(open, grip)]    관절별 — 편 자세~쥔 자세 구간
    raw  = lo + ½(a+1)(hi−lo)                        a = 0 은 구간 가운데(반쯤 쥔 손)
    q*_t = α·raw + (1−α)·q*_{t−1}                    EMA
    Δ    = clamp(q*_t − q*_{t−1}, ±관절 한계 전 범위 × dt / full_range_s)   벤더 속도(전 범위 1 s)
    컵 접촉력 > 임계인 손가락은 닫는 방향 Δ = 0         실기 RH56F1 펌웨어 힘 멈춤(펴기는 허용)
    q*_t = clamp(q*_{t−1} + Δ, lo, hi)
★thumb_1(외전)은 open 1.57 > grip 1.20 이라 이 관절만 a = +1 이 편 쪽이다(pour_bi_rh 와 같다).
"""

from __future__ import annotations

import torch


def grip_range(q_open: torch.Tensor, q_grip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.minimum(q_open, q_grip), torch.maximum(q_open, q_grip)


def linear_grip_raw(a: torch.Tensor, q_open: torch.Tensor, q_grip: torch.Tensor) -> torch.Tensor:
    """액션 (N,H) → 관절 목표 (N,H). −1 → min(open,grip), +1 → max(open,grip)."""
    lo, hi = grip_range(q_open, q_grip)
    return lo.unsqueeze(0) + 0.5 * (a.clamp(-1.0, 1.0) + 1.0) * (hi - lo).unsqueeze(0)


def step_hand_target(prev: torch.Tensor, raw: torch.Tensor, *, alpha: float, cap: torch.Tensor,
                     freeze: torch.Tensor, q_open: torch.Tensor, q_grip: torch.Tensor) -> torch.Tensor:
    """EMA → 스텝당 변화 상한 → 접촉 손가락 닫기 정지 → [open, grip] 구간 clamp. 새 텐서.

    cap (H,) 은 관절별 스텝당 최대 변화, freeze (N,H) bool 은 닫기를 멈출 관절이다.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha 는 (0, 1]: {alpha}")
    ema = float(alpha) * raw + (1.0 - float(alpha)) * prev
    delta = (ema - prev).clamp(-cap.unsqueeze(0), cap.unsqueeze(0))
    close_dir = torch.sign(q_grip - q_open).unsqueeze(0)
    closing = (delta * close_dir) > 0.0
    delta = torch.where(freeze & closing, torch.zeros_like(delta), delta)
    lo, hi = grip_range(q_open, q_grip)
    return torch.maximum(torch.minimum(prev + delta, hi.unsqueeze(0)), lo.unsqueeze(0))


def closure(q: torch.Tensor, q_open: torch.Tensor, q_grip: torch.Tensor) -> torch.Tensor:
    """관절별 폐쇄도 (N,H) ∈ [0,1] — 0 = open, 1 = grip."""
    span = (q_grip - q_open).unsqueeze(0)
    safe = torch.where(span.abs() > 1e-6, span, torch.ones_like(span))
    return ((q - q_open.unsqueeze(0)) / safe).clamp(0.0, 1.0)
