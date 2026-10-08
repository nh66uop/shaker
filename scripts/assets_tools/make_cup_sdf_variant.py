#!/usr/bin/env python3
"""cup_big_sdf 의 SDF 해상도만 바꾼 사본 — 비드 누출 원인 분리용(09.23).

왜: 컵 벽은 1~4 mm 인데 SDF 격자는 최대 변 0.1776 m / sdfResolution 64 = 2.8 mm 라
    벽 한 장이 격자 한 칸보다 얇다. 지름 24~27 mm 비드가 손에 눌릴 때 벽을 통과한다
    (솔버 반복수 8/16 → 32 에서 변화 없음: 0.788 → 0.788).
    해상도만 올리면 **형상은 그대로** 두고(내경·외경·파지 조건 불변) 벽 표현만 촘촘해진다.

실행:
  /usr/bin/python3 scripts/assets_tools/make_cup_sdf_variant.py --resolution 256
산출:
  assets/cup/cup_big_sdf_r<해상도>.usd
"""
from __future__ import annotations

import argparse
import os
import shutil

from pxr import Usd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "assets", "cup", "cup_big_sdf.usd")
_MESH = "/cup_big/cup_big"
_ATTR = "physxSDFMeshCollision:sdfResolution"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True)
    a = ap.parse_args()
    dst = os.path.join(_ROOT, "assets", "cup", f"cup_big_sdf_r{a.resolution}.usd")
    shutil.copyfile(_SRC, dst)
    stage = Usd.Stage.Open(dst)
    attr = stage.GetPrimAtPath(_MESH).GetAttribute(_ATTR)
    if not attr:
        raise SystemExit(f"{_ATTR} 없음 — 원본이 sdf 콜라이더가 아니다")
    before = attr.Get()
    attr.Set(int(a.resolution))
    stage.GetRootLayer().Save()
    print(f"{os.path.relpath(dst, _ROOT)}: sdfResolution {before} -> {a.resolution}")


if __name__ == "__main__":
    main()
