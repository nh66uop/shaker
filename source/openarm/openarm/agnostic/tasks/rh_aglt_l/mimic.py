"""RH56F1 언더액추 계약 — 종속 6관절은 PhysX mimic 제약이 몬다(grasp_fj_rh 09.07 · pour_fabric_mimic 이식).

부팅에서 세 가지를 한다:
  1. 자산 URDF `<mimic>` 표에서 종속 → (리더, 배율)을 읽는다(코드에 표를 적지 않는다 — 자산과 갈린다)
  2. USD prim 의 `physxMimicJoint:rotZ:gearing` 이 −배율인지 대조한다(09.02: headless 빌드가 12개를 잃었다)
  3. 종속관절 한계를 ±margin 넓힌다(없으면 접촉이 구동관절을 한계 밖으로 역구동할 때 mimic 제약과
     한계가 동시 불만족이 되어 폭주한다: rh_b1 222 epoch 중 118 에서 오차 400~916 rad → 확장 뒤 0)
`parse_mimic_pairs` 는 순수 파이썬(테스트용), 나머지는 Isaac 이 떠 있을 때만 부른다.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

GEARING_ATTR = "physxMimicJoint:rotZ:gearing"


def find_asset_urdf(usd_abs_path: str) -> str:
    """USD 와 같은 폴더의 URDF 하나. 없거나 둘 이상이면 실패."""
    d = os.path.dirname(usd_abs_path)
    cands = sorted(glob.glob(os.path.join(d, "*.urdf")))
    if len(cands) != 1:
        raise RuntimeError(f"[rh_aglt] 자산 URDF 를 특정 못했다({d}): {cands}")
    return cands[0]


def parse_mimic_pairs(urdf_path: str, hand_prefix: str) -> dict[str, tuple[str, float]]:
    """`<prefix>_hj_*` 종속관절 → (리더, 배율). hand_prefix 는 'r' | 'l'."""
    pairs: dict[str, tuple[str, float]] = {}
    for j in ET.parse(urdf_path).getroot().iter("joint"):
        m = j.find("mimic")
        dep = j.get("name") or ""
        if m is None or not dep.startswith(f"{hand_prefix}_hj_"):
            continue
        pairs[dep] = (str(m.get("joint")), float(m.get("multiplier", 1.0)))
    if not pairs:
        raise RuntimeError(f"[rh_aglt] URDF 에 {hand_prefix} 손 mimic 이 없다 ({urdf_path}) — 언더액추 손 전용 과제")
    return pairs


def assert_usd_gearing(robot_prim_env0: str, pairs: dict[str, tuple[str, float]]) -> None:
    """USD 에 mimic 제약이 gearing 값까지 맞게 붙었는지. 규약 q_dep + gearing·q_ref = 0 → gearing = −배율."""
    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    found: dict[str, float] = {}
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith(robot_prim_env0):
            continue
        a = p.GetAttribute(GEARING_ATTR)
        if a and a.Get() is not None:
            found[p.GetName()] = float(a.Get())
    bad = []
    for dep, (_lead, mult) in pairs.items():
        got = found.get(dep)
        if got is None:
            bad.append(f"{dep}: mimic 제약 없음")
        elif abs(got + mult) > 1e-3:
            bad.append(f"{dep}: gearing {got:.4f} ≠ −배율 {-mult:.4f}")
    if bad:
        raise RuntimeError("[rh_aglt] USD 언더액추 결합이 깨졌다 — " + " · ".join(bad))
