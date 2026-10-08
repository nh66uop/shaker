"""파지 모양 로그 지표 — rh_aglt_l/shape_metrics.py(커밋 bba5e96a·9ef620ef) 사본(09.29 사용자: "이식은 해두고, 다음 실험에서 보이게").
트랙끼리 파일을 공유하지 않으려고 복사했다. 이 트랙은 두 손이라 env 가 키를 shape/{src,rcv}_* 로 바꿔 기록한다.
보상·관측과 무관한 로그 전용.

파지 모양 로그 지표 — 보상·관측과 무관한 로그 전용. torch 만 쓴다(Isaac 없이 테스트).

사용자 09.29 "파지 모양을 로그로 확인되게": 전 스텝 평균(`task/grasped`, `contact/n_fingers`)은 접근 스텝이 섞여 쥔 모양을
가린다. 여기 지표는 **쥔 스텝(env grasp 플래그)만** 평균한다. 쥔 env 가 하나도 없으면 그 스텝은 기록하지 않는다.

  shape/palm_normal_z     손바닥 법선의 z. 옆잡기(수평) ≈ 0, 위에서 내려 집기 ≈ −1
  shape/finger_dir_z      손가락 길이 방향의 z. 수평 ≈ 0, 손끝이 아래 ≈ −1
  shape/fingers_touching  컵에 1 N 넘게 닿은 손가락 수(0–5)
  shape/prox_touching     첫마디(_1)가 컵에 1 N 넘게 닿은 손가락 수 — 감싸기 정도(손끝만이면 0)
  shape/<finger>          그 손가락이 닿은 비율(thumb, index, middle, ring, pinky)
  shape/palm_axis_r       손바닥 ↔ 컵 축 거리 / 컵 반지름 (인벨롭 실측 1.5)
  shape/grasp_h_mouth     손바닥 높이(컵 원점 기준 축방향) / 입구 높이 (인벨롭 실측 +0.15)
  shape/thumb_oppose      엄지 끝과 4지 끝 평균이 컵 축 반대편에 있는 정도(1 = 정반대)
  shape/cup_tilt_deg      쥔 동안의 컵 기울기
  shape/closure_near_open 손바닥이 컵 원점 10 cm 안이고 아직 쥐지 않은 env 의 손 폐쇄도(편 손 접근 확인)
  shape/closure_far       손바닥이 컵 원점 20 cm 밖인 env 의 손 폐쇄도
  접근 중(아직 안 쥔 env) 손 방향 — 사용자 09.29 방향 규칙(손바닥 +z 수평·마주 봄, 엄지가 새끼보다 위, 뒤집힘 방지):
  shape/approach_palm_z_abs   |손바닥 법선 z| (0 = 수평)
  shape/approach_palm_to_mid  손바닥 법선의 중심선 방향 성분(우 +y·좌 −y, 1 = 정확히 마주 봄)
  shape/approach_thumb_up_mm  엄지 끝 z − 새끼 끝 z [mm] (양수 = 엄지가 위)
  shape/approach_flipped      엄지 끝이 새끼 끝보다 아래인 비율(뒤집힘)
"""

from __future__ import annotations

import torch

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor | None:
    n = m.float().sum()
    if float(n) < 1.0:
        return None
    return (x * m.float()).sum() / n


def grasp_shape(*, grasped: torch.Tensor, palm_pos: torch.Tensor, palm_normal: torch.Tensor,
                finger_dir: torch.Tensor, tips_pos: torch.Tensor, finger_force: torch.Tensor,
                prox_force: torch.Tensor, cup_pos: torch.Tensor, cup_up: torch.Tensor, cup_radius: float,
                cup_mouth_z: float, closure: torch.Tensor, thr: float, mid_sign: float = 1.0) -> dict[str, torch.Tensor]:
    """쥔 스텝 평균 지표 dict(값은 0-dim 텐서). 해당 env 가 없는 키는 빠진다."""
    out: dict[str, torch.Tensor] = {}
    g = grasped
    up = _unit(cup_up)
    v = palm_pos - cup_pos
    h = (v * up).sum(-1)
    r = (v - h.unsqueeze(-1) * up).norm(dim=-1)
    touch = finger_force > thr
    rel = tips_pos - cup_pos.unsqueeze(1)
    rv = rel - (rel * up.unsqueeze(1)).sum(-1, keepdim=True) * up.unsqueeze(1)
    d = _unit(rv)
    oppose = (-(d[:, 0] * _unit(d[:, 1:].mean(dim=1))).sum(-1)).clamp(0.0, 1.0)
    tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
    vals = {
        "shape/palm_normal_z": _unit(palm_normal)[:, 2],
        "shape/finger_dir_z": _unit(finger_dir)[:, 2],
        "shape/fingers_touching": touch.float().sum(-1),
        "shape/prox_touching": (prox_force > thr).float().sum(-1),
        "shape/palm_axis_r": r / max(float(cup_radius), 1e-6),
        "shape/grasp_h_mouth": h / max(float(cup_mouth_z), 1e-6),
        "shape/thumb_oppose": oppose,
        "shape/cup_tilt_deg": tilt,
        **{f"shape/{f}": touch[:, i].float() for i, f in enumerate(FINGERS)},
    }
    for k, x in vals.items():
        m = _masked_mean(x, g)
        if m is not None:
            out[k] = m
    free = ~g
    nz = _unit(palm_normal)
    thumb_up = tips_pos[:, 0, 2] - tips_pos[:, -1, 2]
    for k, x in (("shape/approach_palm_z_abs", nz[:, 2].abs()), ("shape/approach_palm_to_mid", nz[:, 1] * float(mid_sign)),
                 ("shape/approach_thumb_up_mm", 1000.0 * thumb_up), ("shape/approach_flipped", (thumb_up < 0.0).float())):
        m = _masked_mean(x, free)
        if m is not None:
            out[k] = m
    dist = v.norm(dim=-1)
    near = _masked_mean(closure, (dist < 0.10) & ~g)
    far = _masked_mean(closure, dist > 0.20)
    if near is not None:
        out["shape/closure_near_open"] = near
    if far is not None:
        out["shape/closure_far"] = far
    return out
