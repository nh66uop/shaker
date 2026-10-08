#!/usr/bin/env python3
"""shaker_closed_rl 의 바닥을 두껍게 한 사본 — pour_fabric_mimic 전용(09.22 사용자 결정).

문제(09.22 i11 ep150 trace, 64 env): 받는 컵에 들어간 비드의 2~3 % 가 바닥을 뚫고 빠졌다.
바닥은 SDF 관 메시 1.2 mm + convexHull `bottom_plug` 4 mm(scale 0.65 에서 0.78 / 2.6 mm)이고,
비드(반지름 6 mm)는 떨어질 때 물리 스텝당 ~12 mm 움직인다. CCD(씬+강체)로는 0.028 → 0.020 뿐이었다.

해결: 기존 plug 위에 convexHull 원기둥 `floor_fill` 을 하나 더 쌓아 바닥 두께를 늘린다.
반경은 채움 구간 **하단**의 관 내경(테이퍼라 위로 갈수록 넓어진다)이라 외곽 프로파일을 바꾸지 않는다.
상단에 남는 틈은 내경 증가분(수 mm × scale)으로 비드 지름보다 훨씬 작다.

원본 shaker_closed_rl.usd 는 다른 트랙(grasp 등)이 쓰므로 제자리 수정하지 않는다.

실행:
  /usr/bin/python3 scripts/tools/make_thick_floor_shaker_asset.py [--fill 0.016]
산출:
  assets/cup/shaker_closed_thick_rl.usd
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

_HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_HDGP_ROOT, "assets", "cup", "shaker_closed_rl.usd")
_DST = os.path.join(_HDGP_ROOT, "assets", "cup", "shaker_closed_thick_rl.usd")
_BASE_LINK = "/object_shaker_body/baseLink"
_COLLISIONS = f"{_BASE_LINK}/collisions"
_PLUG = f"{_BASE_LINK}/bottom_plug"
_FILL = f"{_BASE_LINK}/floor_fill"


def _inner_radius_at(points: np.ndarray, z: float) -> float:
    """관 내벽 반경(z 층별 최소 반경)을 z 에서 선형 보간한다."""
    zs = np.round(points[:, 2], 4)
    r = np.linalg.norm(points[:, :2], axis=1)
    levels = np.array(sorted(set(zs)))
    r_in = np.array([r[zs == lv].min() for lv in levels])
    return float(np.interp(z, levels, r_in))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", type=float, default=0.016, help="plug 위에 더할 두께 [m, scale 1]")
    a = ap.parse_args()

    src = Usd.Stage.Open(_SRC)
    plug = UsdGeom.Cylinder(src.GetPrimAtPath(_PLUG))
    plug_h = float(plug.GetHeightAttr().Get())
    plug_cz = float(plug.GetPrim().GetAttribute("xformOp:translate").Get()[2])
    plug_top = plug_cz + 0.5 * plug_h
    pts = np.asarray(src.GetPrimAtPath(_COLLISIONS).GetAttribute("points").Get(), dtype=float)
    fill_r = _inner_radius_at(pts, plug_top)
    top_r = _inner_radius_at(pts, plug_top + a.fill)

    shutil.copyfile(_SRC, _DST)
    stage = Usd.Stage.Open(_DST)
    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(_FILL))
    cyl.CreateRadiusAttr(fill_r)
    cyl.CreateHeightAttr(a.fill)
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([(-fill_r, -fill_r, -0.5 * a.fill), (fill_r, fill_r, 0.5 * a.fill)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set((0.0, 0.0, plug_top + 0.5 * a.fill))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    cyl.GetPrim().CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
    stage.GetRootLayer().Save()

    chk = Usd.Stage.Open(_DST)
    if not chk.GetPrimAtPath(_FILL).HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("floor_fill CollisionAPI 미적용")
    if chk.GetPrimAtPath(_COLLISIONS).GetAttribute("physics:approximation").Get() != "sdf":
        raise RuntimeError("관 콜라이더 SDF 설정 유실")
    print(f"[산출] {_DST}")
    print(f"  plug top {plug_top:+.4f} → 새 바닥 {plug_top + a.fill:+.4f} (총 두께 {(plug_h + a.fill)*1000:.1f} mm, scale 1)")
    print(f"  floor_fill r {fill_r*1000:.2f} mm · 상단 틈 {(top_r - fill_r)*1000:.2f} mm")


if __name__ == "__main__":
    main()
