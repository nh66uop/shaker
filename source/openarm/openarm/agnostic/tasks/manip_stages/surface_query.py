"""점 → 물체 표면 최근접 질의 (torch, 순수 함수) — `object_surface` 점군을 매 스텝 물체 자세로 옮겨 쓴다.

★질의는 점을 물체 프레임으로 되돌려(q_b = Rᵀ(q − p)) 종별 점군과 비교한 뒤 결과 방향·법선만 월드로 돌린다
  — 점군 (N,K,3) 을 매 스텝 월드로 옮기지 않는다. env 청크 단위로 거리 행렬을 만들어 4096 env × 16 질의 × 512 점도
  한 번에 (청크,16,512) 만 잡는다.
★뒷면 건너뛰기: 질의점이 표본 면의 **뒤**(접평면 아래 `BACKFACE_TOL_M` 초과)에 있으면 그 표본은 후보에서 밀린다.
  얇은 벽(shaker 0.8 mm · cup_big 5.8 mm)의 안쪽 면이 바깥에서 온 손가락의 "가장 가까운 점"이 되어 법선이 뒤집히는 것을
  막는다. 앞을 보는 표본이 하나도 없으면(물체 속 깊이) 그냥 가장 가까운 표본이다.
"""

from __future__ import annotations

import torch

#: 이만큼까지는 면 뒤에 있어도 앞면으로 친다 [m] — 표본 접평면 오차·얕은 관통 여유.
BACKFACE_TOL_M = 0.005
#: 뒷면 표본에 더하는 벌점 [m] — 어떤 실제 거리보다 크다.
_BACKFACE_PENALTY_M = 10.0
#: 한 번에 거리 행렬을 만드는 env 수.
DEFAULT_CHUNK = 1024


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(w,x,y,z) 쿼터니언 (N,4) 로 벡터 (N,...,3) 회전 — 새 텐서."""
    shape = v.shape
    qv = q.view(q.shape[0], *([1] * (v.dim() - 2)), 4).expand(*shape[:-1], 4)
    w, u = qv[..., :1], qv[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(torch.cat([q[:, :1], -q[:, 1:]], dim=1), v)


def nearest_surface(query: torch.Tensor, obj_pos: torch.Tensor, obj_quat: torch.Tensor,
                    clouds: torch.Tensor, cloud_normals: torch.Tensor, species: torch.Tensor,
                    *, chunk: int = DEFAULT_CHUNK, backface_tol: float = BACKFACE_TOL_M
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """질의점 (N,Q,3) → (거리 (N,Q), 표면점으로의 단위 방향 (N,Q,3), 그 점의 바깥 법선 (N,Q,3)). 전부 월드(질의와 같은) 프레임.

    obj_pos (N,3)·obj_quat (N,4, wxyz) = 점군 프레임(강체 actor)의 자세, query 와 같은 원점. clouds/cloud_normals (S,K,3) 는
    종별 점군, species (N,) long 은 env 별 종 번호.
    """
    n, nq, _ = query.shape
    dist = torch.empty(n, nq, device=query.device, dtype=query.dtype)
    vec_b = torch.empty(n, nq, 3, device=query.device, dtype=query.dtype)
    nrm_b = torch.empty(n, nq, 3, device=query.device, dtype=query.dtype)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        qb = quat_rotate_inverse(obj_quat[s:e], query[s:e] - obj_pos[s:e, None, :])    # (C,Q,3)
        pts, nrm = clouds[species[s:e]], cloud_normals[species[s:e]]                   # (C,K,3)
        diff = pts[:, None, :, :] - qb[:, :, None, :]                                   # (C,Q,K,3) q → p
        d = diff.norm(dim=-1)
        behind = (diff * nrm[:, None, :, :]).sum(-1) > backface_tol                     # (q−p)·n < −tol
        idx = torch.argmin(d + behind.to(d.dtype) * _BACKFACE_PENALTY_M, dim=-1)        # (C,Q)
        dist[s:e] = torch.gather(d, 2, idx[..., None]).squeeze(-1)
        vec_b[s:e] = torch.gather(diff, 2, idx[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        nrm_b[s:e] = torch.gather(nrm, 1, idx[..., None].expand(-1, -1, 3))
    dir_w = quat_rotate(obj_quat, vec_b / dist[..., None].clamp(min=1e-9))
    return dist, dir_w, quat_rotate(obj_quat, nrm_b)


__all__ = ["nearest_surface", "quat_rotate", "quat_rotate_inverse", "BACKFACE_TOL_M"]
