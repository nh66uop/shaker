"""생성 보상 검증 — `GraspRewardContext` 판. `object_validator` 와 같은 방식으로 부모 검증기를 **그대로** 돌린다.

  · 정적 검사: 부모 결과에서 "RewardContext 에 없는 필드" 오류 중 표면 필드와 첫마디 필드만 지운다.
  · 드라이런: `object_validator.make_fake_context` + 첫마디 필드(거리 0–0.3 m · 힘 0–5 N · 단위 벡터).
t2r_ms.py 는 이 모듈을 import 하지만 `--context grasp`(또는 grasp meta)일 때만 검증기로 쓴다. cup_pick 트랙(기본 object)은 쓰지 않는다.
"""

from __future__ import annotations

import contextlib
from unittest import mock

import torch

from ..grasp_fj_t2r.t2r import validator as _V
from . import object_validator as _OV
from .grasp_context import GRASP_TENSOR_FIELDS, GraspRewardContext, from_object
from .object_context import OBJECT_TENSOR_FIELDS

ValidationReport = _V.ValidationReport
_BASE_STATIC_CHECK = _V.static_check
_UNKNOWN_FIELD_MSG = "RewardContext 에 없는 필드: ctx."


def static_check(src: str) -> ValidationReport:
    rep = _BASE_STATIC_CHECK(src)
    ours = {f"{_UNKNOWN_FIELD_MSG}{name}" for name in OBJECT_TENSOR_FIELDS + GRASP_TENSOR_FIELDS}
    rep.errors = [e for e in rep.errors if e not in ours]
    rep.ok = not rep.errors
    return rep


def make_fake_context(n: int = 16, *, device: str = "cpu", seed: int = 0) -> GraspRewardContext:
    """표면 가짜 ctx + 첫마디 필드. 물리적 정합은 보장하지 않는다(드라이런 전용)."""
    base = _OV.make_fake_context(n, device=device, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(20_000 + seed)

    def unit(*shape):
        v = torch.rand(*shape, generator=g) * 2.0 - 1.0
        return (v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)).to(device)

    return from_object(base,
                       prox_pos=(base.palm_pos[:, None, :] + torch.rand(n, 5, 3, generator=g).to(device) * 0.05),
                       prox_cup_force=(torch.rand(n, 5, generator=g) * 5.0).to(device),
                       prox_surface_dist=(torch.rand(n, 5, generator=g) * 0.3).to(device),
                       prox_surface_dir=unit(n, 5, 3), prox_surface_normal=unit(n, 5, 3))


@contextlib.contextmanager
def _grasp_context_in_base_validator():
    with mock.patch.object(_V, "static_check", static_check), mock.patch.object(_V, "make_fake_context", make_fake_context):
        yield


def dry_run(path: str, *, n: int = 64, device: str = "cpu") -> ValidationReport:
    with _grasp_context_in_base_validator():
        return _V.dry_run(path, n=n, device=device)


def validate(path: str, devices: tuple[str, ...] | None = None) -> ValidationReport:
    with _grasp_context_in_base_validator():
        return _V.validate(path, devices)


cuda_usable = _V.cuda_usable

__all__ = ["ValidationReport", "static_check", "make_fake_context", "dry_run", "validate", "cuda_usable"]
