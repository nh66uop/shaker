"""좌우 미러 계약 — cup_pick 의 좌·우 보상 파일은 **바이트가 같아야** 한다 (09.23 사용자 결정).

왜 복사본을 놓고 비교하나: iter_07 부터 보상이 프레임 무관해졌다(방향은 전부 ctx 벡터끼리의 내적).
그래서 옛 `mirror_reward.py` 부호 반전이 필요 없고, **같은 파일을 양팔이 그대로 쓴다.**
한쪽만 손대면 좌우 정책이 서로 다른 과제를 배우므로 그것을 이 테스트가 막는다.

★프레임 무관 규칙도 같이 지킨다: 보상 코드에 월드 축 상수(+y·-y 등)가 있으면 미러가 깨진다.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[7] / "reward_gen"
_LEFT, _RIGHT = _ROOT / "cup_pick_l_manip", _ROOT / "cup_pick_r_manip"
#: 좌우 공용 보상은 iter_10(양팔 병렬 fresh 학습)부터다. 그 이전 iter 는 좌팔 전용 이력.
_SHARED_FROM = 10


def _shared_iters() -> list[str]:
    return sorted(p.name for p in _RIGHT.glob("iter_*")
                  if p.is_dir() and int(p.name.split("_")[1]) >= _SHARED_FROM)


def test_right_rewards_are_byte_identical_to_the_left_ones():
    names = _shared_iters()
    assert names, "우팔 공용 iter 가 하나도 없다 — 복사 단계를 건너뛰었다"
    for name in names:
        r = (_RIGHT / name / "compute_reward.py").read_bytes()
        l = (_LEFT / name / "compute_reward.py").read_bytes()
        assert r == l, f"{name}: 좌우 보상이 다르다 — 한쪽만 수정했다"


def test_shared_rewards_hold_no_world_axis_constant():
    """월드 축 상수는 미러를 깬다. ctx 벡터끼리의 내적만 허용한다."""
    # 월드 축 상수 = 성분이 전부 0/±1 인 **3원소** 텐서(예: tensor([0.0, -1.0, 0.0])).
    # 5원소 가중치 벡터(rank_w 등)는 축이 아니므로 걸리면 안 된다 — 09.23 오탐 1회.
    _u = r"-?[01](?:\.0*)?"
    banned = re.compile(rf"""tensor\(\s*[\[(]\s*{_u}\s*,\s*{_u}\s*,\s*{_u}\s*[\])]"""
                        r"""|\[:,\s*1\]\s*[<>*+-]|ctx\.cup_axis|ctx\.cup_radius|ctx\.cup_half_height"""
                        r"""|approach_done|envelope_done""")
    for name in _shared_iters():
        src = (_LEFT / name / "compute_reward.py").read_text(encoding="utf-8")
        hits = [m.group(0) for m in banned.finditer(src)]
        assert not hits, f"{name}: 미러를 깨는 표현 {hits}"
