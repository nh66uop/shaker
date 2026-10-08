"""생성 보상 검증 — `ObjectRewardContext` 판. grasp_fj_t2r 검증기(s2r 동결)를 **그대로** 돌리고 두 곳만 바꾼다.

  · 정적 검사: 부모 결과에서 "RewardContext 에 없는 필드" 오류 중 표면 필드(`OBJECT_TENSOR_FIELDS`)만 지운다.
    그 밖의 오류(환각 필드·금지 import·ctx 대입 …)는 부모 판정 그대로다.
  · 드라이런: 부모 가짜 ctx + 표면 필드(거리 0–0.3 m · 단위 벡터)로 채운 `ObjectRewardContext`.
★부모 모듈 전역(`static_check`·`make_fake_context`)을 **이 호출 동안만** 바꿔 부모 `validate` 를 돌린다 — 부모 코드 경로가
  한 줄도 갈라지지 않는다. 다른 트랙(t2r_fj.py)은 이 모듈을 import 하지 않으므로 영향이 없다.
"""

from __future__ import annotations

import contextlib
from unittest import mock

import torch

from ..grasp_fj_t2r.t2r import validator as _V
from .object_context import OBJECT_TENSOR_FIELDS, ObjectRewardContext, from_base

ValidationReport = _V.ValidationReport
_BASE_STATIC_CHECK = _V.static_check
_BASE_FAKE = _V.make_fake_context
_UNKNOWN_FIELD_MSG = "RewardContext 에 없는 필드: ctx."


def static_check(src: str) -> ValidationReport:
    rep = _BASE_STATIC_CHECK(src)
    ours = {f"{_UNKNOWN_FIELD_MSG}{name}" for name in OBJECT_TENSOR_FIELDS}
    rep.errors = [e for e in rep.errors if e not in ours]
    rep.ok = not rep.errors
    return rep


def make_fake_context(n: int = 16, *, device: str = "cpu", seed: int = 0) -> ObjectRewardContext:
    """부모 가짜 ctx + 표면 필드. 물리적 정합은 보장하지 않는다(드라이런 전용)."""
    base = _BASE_FAKE(n, device=device, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(10_000 + seed)

    def dist(*shape):
        return (torch.rand(*shape, generator=g) * 0.3).to(device)

    def unit(*shape):
        v = torch.rand(*shape, generator=g) * 2.0 - 1.0
        return (v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)).to(device)

    return from_base(base, palm_surface_dist=dist(n), palm_surface_dir=unit(n, 3), palm_surface_normal=unit(n, 3),
                     link_surface_dist=dist(n, 5, 3), link_surface_dir=unit(n, 5, 3, 3),
                     link_surface_normal=unit(n, 5, 3, 3))


@contextlib.contextmanager
def _object_context_in_base_validator():
    with mock.patch.object(_V, "static_check", static_check), mock.patch.object(_V, "make_fake_context", make_fake_context):
        yield


def dry_run(path: str, *, n: int = 64, device: str = "cpu") -> ValidationReport:
    with _object_context_in_base_validator():
        return _V.dry_run(path, n=n, device=device)


def validate(path: str, devices: tuple[str, ...] | None = None) -> ValidationReport:
    with _object_context_in_base_validator():
        return _V.validate(path, devices)


cuda_usable = _V.cuda_usable

__all__ = ["ValidationReport", "static_check", "make_fake_context", "dry_run", "validate", "cuda_usable"]
