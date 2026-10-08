"""ObjectRewardContext — `RewardContext`(grasp_fj_t2r, s2r 동결) + 물체 모양과 무관한 **표면 거리** 입력 (09.22 사용자 결정).

★기존 필드는 한 글자도 바꾸지 않는다(래치 approach_done·envelope_done 도 로그용으로 남는다). 새 필드만 뒤에 붙는다.
★새 필드 주석이 곧 LLM 이 읽는 환경 설명이다(`context_stub_source` 가 부모 스텁 뒤에 이 클래스의 필드 블록을 그대로 붙인다).
  주석의 수치 출처: `object_surface` 표본(K 512 · FPS 덮개 반경 cup_big s100 14 mm · s130 17 mm · shaker 11 mm) ·
  `tests/test_object_surface.py` 의 브루트포스 메시 거리 대조(09.22). 손가락 마디 원점↔표면 접촉 거리는 스모크 실측.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields

import torch

from ..grasp_fj_t2r.t2r.context import RewardContext
from ..grasp_fj_t2r.t2r.context import context_stub_source as _base_stub_source


@dataclass(frozen=True)
class ObjectRewardContext(RewardContext):
    # ---- object surface: measured on the object's actual collision surface, the same way for any object shape ----
    palm_surface_dist: torch.Tensor    # (N,) distance from palm_pos to the nearest point of the object's surface [m]; the surface is represented by 512 points spread evenly over the object's collision geometry (about 1.1-1.7 cm apart), so this reads up to about 1 cm more than the true distance when the point is at the surface and is within a few mm of it beyond 3 cm
    palm_surface_dir: torch.Tensor     # (N,3) unit vector from palm_pos towards that nearest surface point
    palm_surface_normal: torch.Tensor  # (N,3) outward unit normal of the object's surface at that nearest point; palm_surface_dir . palm_surface_normal is close to -1 when palm_pos is outside the object, facing it. Surface points seen from behind are skipped, so next to a thin-walled cup the nearest point is on the outer side of the wall
    link_surface_dist: torch.Tensor    # (N,5,3) the same distance for each finger link position in link_pos (same finger and link order) [m]; link_pos are points inside the finger links, not on their skin, so a link that touches the object (link_cup_force > 0) reads about LINK_TOUCH_DIST rather than 0
    link_surface_dir: torch.Tensor     # (N,5,3,3) unit vector from each finger link position towards its nearest surface point
    link_surface_normal: torch.Tensor  # (N,5,3,3) outward unit normal of the object's surface at each link's nearest surface point

    # ------------------------------------------------------------------


OBJECT_TENSOR_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(ObjectRewardContext) if f.name not in {g.name for g in fields(RewardContext)})
TENSOR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ObjectRewardContext) if f.type == "torch.Tensor")
SCALAR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ObjectRewardContext) if f.type in ("float", "int"))

#: 스텁의 자리표시 → 스모크 실측값(09.22, `open-short_l_cup_pick-lstm` 256 env · approach e3400 가중치 211 epoch:
#:   컵 접촉력 > 0.1 N 인 마디의 표면 거리 평균 17.9 mm · 2 cm 초과 27.8 %).
STUB_VALUES: dict[str, str] = {"LINK_TOUCH_DIST": "1-3 cm (1.8 cm on average)"}


def object_fields_source() -> str:
    """`ObjectRewardContext` 의 필드 블록(섹션 머리말 포함) — 부모 스텁 뒤에 붙는다."""
    src = inspect.getsource(ObjectRewardContext)
    start = src.index("    # ---- object surface")
    end = src.index("    # ----------", start)
    body = src[start:end].rstrip() + "\n"
    for key, val in STUB_VALUES.items():
        body = body.replace(key, val)
    return body


def context_stub_source(side: str = "r") -> str:
    """프롬프트 재료: 부모 `class RewardContext` 스텁(좌/우 문구 포함) + 표면 필드. LLM 에게는 한 클래스로 보인다."""
    return _base_stub_source(side) + "\n" + object_fields_source()


def from_base(base: RewardContext, **extra: torch.Tensor) -> ObjectRewardContext:
    """부모 ctx 의 모든 필드 + 새 필드 → ObjectRewardContext (복사 없이 같은 텐서를 넘긴다)."""
    missing = set(OBJECT_TENSOR_FIELDS) - set(extra)
    unknown = set(extra) - set(OBJECT_TENSOR_FIELDS)
    if missing or unknown:
        raise KeyError(f"표면 필드 불일치 — 없음 {sorted(missing)} · 모름 {sorted(unknown)}")
    return ObjectRewardContext(**{f.name: getattr(base, f.name) for f in fields(RewardContext)}, **extra)


__all__ = ["ObjectRewardContext", "OBJECT_TENSOR_FIELDS", "TENSOR_FIELDS", "SCALAR_FIELDS",
           "context_stub_source", "object_fields_source", "from_base"]
