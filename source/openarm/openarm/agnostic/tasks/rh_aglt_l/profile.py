"""RH56F1 **왼팔** 프로필 — `modules/robot_profiles.RH56F1_RIGHT` 의 거울상(모듈 레지스트리에는 넣지 않는다:
grasp 계열 config 가 그 dict 로 gym id 를 찍는다).

거울 규칙(09.29 URDF FK 재확인, pour_fabric_mimic 09.14 와 같은 결과):
  · 팔 부호 (−1, −1, −1, +1, −1, −1, −1) — 먼 출발 자세에서 손바닥 2.7 mm · 4지 손끝 0.00 mm · 엄지 5.5 mm
    (엄지·손바닥 차이는 벤더 자산의 엄지 마운트 비대칭, 부호 오류면 수 cm).
  · 손 6관절 전부 +1 — 좌 URDF 가 엄지 축(thumb_1·thumb_2 z)을 이미 뒤집어 두었고 4지 축·한계·mimic 배율은 좌우 같다.
  · 손바닥 프레임: 좌 R = 반사 뒤 열 0·1 부호가 반대(오른손 좌표계 유지), 열 2(손바닥 법선)는 거울 그대로 → 보상 입력
    palm_axes(열 2·1)의 법선 의미는 좌우 같다.
  · 쉬는 우팔 = 우 프로필의 쉬는 좌팔 × 부호(fabric `_ARM_REST_R` 과 같은 값), 쉬는 우손 = 0.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from openarm.agnostic.modules import robot_profiles as _rp

ARM_SIGN_L = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
_FLEX = ("index", "middle", "ring", "pinky")
_HAND_ALL = ("thumb_1", "thumb_2", "thumb_3", "thumb_4", *(f"{f}_1" for f in _FLEX), *(f"{f}_2" for f in _FLEX))
_R = _rp.RH56F1_RIGHT


def _to_left(name: str) -> str:
    return "l_" + name[2:] if name.startswith("r_") else name


def _mirror_init(init: dict) -> dict:
    out = {k: v for k, v in init.items() if not k.startswith(("r_", "l_"))}      # head 등
    for i, s in enumerate(ARM_SIGN_L, start=1):
        out[f"l_aj_{i}"] = s * init[f"r_aj_{i}"]
        out[f"r_aj_{i}"] = s * init[f"l_aj_{i}"]
    for n in _HAND_ALL:
        out[f"l_hj_{n}"] = init[f"r_hj_{n}"]
        out[f"r_hj_{n}"] = 0.0
    return out


RH56F1_LEFT = _replace(
    _R,
    name="rh56f1_left",
    arm_joint_regex="l_aj_[1-7]",
    hand_joint_regex="l_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)",
    palm_body=_to_left(_R.palm_body),
    hand_joint_names=tuple(_to_left(n) for n in _R.hand_joint_names),
    finger_sensor_bodies={f: tuple(_to_left(b) for b in bs) for f, bs in _R.finger_sensor_bodies.items()},
    fingertip_bodies=tuple(_to_left(b) for b in _R.fingertip_bodies),
    palm_box_min=(_R.palm_box_min[0], -_R.palm_box_max[1], _R.palm_box_min[2]),
    palm_box_max=(_R.palm_box_max[0], -_R.palm_box_min[1], _R.palm_box_max[2]),
    init_joint_pos=_mirror_init(_R.init_joint_pos),
    object_spawn_center=(_R.object_spawn_center[0], -_R.object_spawn_center[1]),
)
