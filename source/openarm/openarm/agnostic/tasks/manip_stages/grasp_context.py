"""GraspRewardContext — `ObjectRewardContext` + 각 손가락 **첫마디**(뿌리 관절이 움직이는 링크) 입력 (09.27 사용자 결정).

왜: cup_pick·cup_grasp 20여 라운드 내내 정책은 손끝 패드로만 쥐었다. m18 재생(256 에피소드)에서 네 손가락이 컵을 쥔 스텝의
가운데 마디(`_3`) 접촉률은 0.00, 손끝 패드 0.90~1.00 이었다. 첫마디(`_2`)에는 센서도 ctx 점도 없어서 뿌리부터 감싸는 파지를
보상이 볼 수 없었다. 사용자 결정(09.27 "1,3을 같이"): 보상은 마디 접촉을 기준으로 하고, env 는 첫마디 입력을 더한다.

★부모(`ObjectRewardContext`) 필드는 한 글자도 바꾸지 않는다. 새 필드만 뒤에 붙는다 — cup_pick 트랙의 보상 코드는 그대로 돈다.
★새 필드 주석이 곧 LLM 이 읽는 환경 설명이다(`context_stub_source` 가 부모 스텁 뒤에 이 블록을 붙인다).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields

import torch

from .object_context import ObjectRewardContext
from .object_context import context_stub_source as _object_stub_source

#: critic 접촉 관측 차원 = 첫마디 5 + (가운데·끝마디·손끝) 15 + 손바닥 1 — env(블록)와 cfg(state_space)의 단일 출처.
CRITIC_CONTACT_DIM = 5 + 15 + 1
#: critic 접촉 관측의 힘 척도 [N] — tanh(F/5): 0.5 N → 0.10, 5 N → 0.76, 20 N → 1.0. 스파이크(수백 N)는 1 에서 포화.
CRITIC_FORCE_REF_N = 5.0

#: 첫마디 링크 이름 접미사 — 뿌리 굽힘 관절 `_2` 가 움직이는 링크(`<side>_hl_<finger>_2`).
PROXIMAL_SUFFIX = "_2"


@dataclass(frozen=True)
class GraspRewardContext(ObjectRewardContext):
    # ---- proximal finger links: the first segment of each digit, next to the palm ----
    # Each digit is a chain of four rigid segments from the palm outwards: the proximal link (below), then the three
    # links in link_pos: [:, f, 0] the middle link, [:, f, 1] the distal link and [:, f, 2] the fingertip pad, which is
    # rigidly attached to the distal link (so a pad pressed flat on a surface usually loads the distal link too).
    # A digit wrapped around an object touches it with its proximal and middle links as well as its distal link;
    # a digit pressing with its fingertip touches only with the pad and the distal link.
    prox_pos: torch.Tensor             # (N,5,3) position of each digit's proximal link (same finger order as link_pos) [m]; for the index, middle and ring fingers this link is flexed by the finger's root joint; for the thumb and the pinky the root joint is fixed in this hand, so their proximal link moves only with the palm
    prox_cup_force: torch.Tensor       # (N,5) contact force magnitude between each proximal link and the object only [N] (0 = not touching the object), measured the same way as link_cup_force
    prox_surface_dist: torch.Tensor    # (N,5) distance from prox_pos to the nearest point of the object's surface [m], measured like link_surface_dist; prox_pos is a point inside the link, so a touching proximal link reads about PROX_TOUCH_DIST rather than 0
    prox_surface_dir: torch.Tensor     # (N,5,3) unit vector from each prox_pos towards its nearest surface point
    prox_surface_normal: torch.Tensor  # (N,5,3) outward unit normal of the object's surface at that nearest point

    # ------------------------------------------------------------------


GRASP_TENSOR_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(GraspRewardContext) if f.name not in {g.name for g in fields(ObjectRewardContext)})

#: 자리표시 → 값. 첫마디 원점↔표면 접촉 거리는 부모 마디(1-3 cm) 와 같은 방법으로 잰다 — 스모크 실측 뒤 갱신한다.
STUB_VALUES: dict[str, str] = {"PROX_TOUCH_DIST": "1-3 cm (as for the other finger links)"}


def grasp_fields_source() -> str:
    """`GraspRewardContext` 의 필드 블록(섹션 머리말·설명 주석 포함) — 표면 필드 뒤에 붙는다."""
    src = inspect.getsource(GraspRewardContext)
    start = src.index("    # ---- proximal finger links")
    end = src.index("    # ----------", start)
    body = src[start:end].rstrip() + "\n"
    for key, val in STUB_VALUES.items():
        body = body.replace(key, val)
    return body


def context_stub_source(side: str = "r") -> str:
    """프롬프트 재료: 부모 스텁 + 표면 필드 + 첫마디 필드. LLM 에게는 한 클래스로 보인다."""
    return _object_stub_source(side) + grasp_fields_source()


def from_object(base: ObjectRewardContext, **extra: torch.Tensor) -> GraspRewardContext:
    """ObjectRewardContext 의 모든 필드 + 첫마디 필드 → GraspRewardContext (복사 없이 같은 텐서를 넘긴다)."""
    missing = set(GRASP_TENSOR_FIELDS) - set(extra)
    unknown = set(extra) - set(GRASP_TENSOR_FIELDS)
    if missing or unknown:
        raise KeyError(f"첫마디 필드 불일치 — 없음 {sorted(missing)} · 모름 {sorted(unknown)}")
    return GraspRewardContext(**{f.name: getattr(base, f.name) for f in fields(ObjectRewardContext)}, **extra)


__all__ = ["GraspRewardContext", "GRASP_TENSOR_FIELDS", "PROXIMAL_SUFFIX", "CRITIC_CONTACT_DIM", "CRITIC_FORCE_REF_N", "context_stub_source",
           "grasp_fields_source", "from_object"]
