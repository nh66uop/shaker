"""컵 소환 — 직사각형 균등 + 상판 구멍 회피 재추첨. torch 만 쓴다(Isaac 없이 테스트).

상판(env_v1) 구멍: 카운터보어 반경 0.0305 m 다섯 개(grasp_fj_t2r 09.17 collision 메시 실측). 컵 원(가장 넓은
반경)이 구멍 원 + 여유와 겹치면 그 env 만 같은 범위에서 다시 뽑는다. 시작 자세의 손은 모든 컵 윗면보다 위라
(이 트랙 CLAUDE.md 실측) 손 그림자 회피는 두지 않는다.
"""

from __future__ import annotations

import torch


def _hits_hole(xy: torch.Tensor, holes: tuple, cup_radius: float, margin: float) -> torch.Tensor:
    hit = torch.zeros(xy.shape[0], dtype=torch.bool, device=xy.device)
    for cx, cy, r in holes:
        d = (xy - torch.tensor((float(cx), float(cy)), device=xy.device)).norm(dim=-1)
        hit |= d < float(r) + float(cup_radius) + float(margin)
    return hit


def sample_cup_xy(n: int, center: tuple, half: tuple, holes: tuple, cup_radius: float,
                  margin: float, device, max_tries: int = 16) -> tuple[torch.Tensor, int, int]:
    """(xy (n,2), 다시 뽑은 수, 끝까지 걸린 수). 끝까지 걸린 env 는 중심으로 둔다(중심이 구멍 밖인지는 cfg 가 검사)."""
    ctr = torch.tensor((float(center[0]), float(center[1])), device=device)
    hw = torch.tensor((float(half[0]), float(half[1])), device=device)

    def draw(k: int) -> torch.Tensor:
        return ctr + (torch.rand(k, 2, device=device) * 2.0 - 1.0) * hw

    xy = draw(n)
    bad = _hits_hole(xy, holes, cup_radius, margin)
    resampled = int(bad.sum())
    for _ in range(int(max_tries)):
        if not bool(bad.any()):
            break
        k = int(bad.sum())
        xy[bad] = draw(k)
        bad = _hits_hole(xy, holes, cup_radius, margin)
    failed = int(bad.sum())
    if failed:
        xy[bad] = ctr
    return xy, resampled, failed
