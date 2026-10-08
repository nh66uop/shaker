#!/usr/bin/env python3
"""보상 파일을 좌↔우로 **기계적으로** 미러한다 — 손으로 고치면 두 벌이 조용히 갈라진다.

★09.20 사용자 "왼팔로 구조를 확정하고 오른쪽도 동일하게 · reward 구조를 축적". 같은 단계 보상이 팔마다 한 벌씩 생기는데,
  실제로 다른 것은 **시작 자세의 손바닥 법선 방향 한 줄**뿐이다(우 +y · 좌 −y, short-tl URDF FK 실측 — 위치 미러 오차 0.0003 mm).
  ctx 의 나머지는 env 가 이미 측(side)에 맞춰 준다: 손 정규화는 두 손 모두 "0 = 곧음 · 1 = 가장 굽힘",
  접근 방향 조건은 `grasp_fj_t2r_env` 가 `cfg.hand_side` 로 고른다.

  그래서 원본은 **한 벌만** 두고 반대쪽은 이 스크립트가 만든다. 계약(그 한 줄 외 완전 동일)은
  `tasks/manip_stages/tests/test_manip_stages_contract.py` 가 잠근다.

    python3 scripts/reward_gen/mirror_reward.py --src reward_gen/cup_pick_r_approach/iter_00/compute_reward.py \
        --dst reward_gen/cup_pick_l_approach/iter_00/compute_reward.py --to l
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: 우 → 좌 치환(코드 한 줄 + 바로 위 주석). 좌 → 우는 역방향으로 쓴다.
RULES: tuple[tuple[str, str], ...] = (
    ("    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))",
     "    ang_n = torch.acos(torch.clamp(-n[:, 1], -1.0, 1.0))"),
    ("# start orientation kept (palm_normal +y, palm_finger_dir +x)",
     "# start orientation kept (palm_normal -y on this left hand, palm_finger_dir +x)"),
)


def mirror(src_text: str, to: str) -> str:
    """`to` 가 "l" 이면 우 → 좌, "r" 이면 좌 → 우. 규칙마다 원문이 정확히 한 번 있어야 한다(없으면 조용히 어긋난다)."""
    if to not in ("l", "r"):
        raise SystemExit(f"[mirror_reward] --to 는 l | r — got {to!r}")
    out = src_text
    for right, left in RULES:
        a, b = (right, left) if to == "l" else (left, right)
        if out.count(a) != 1:
            raise SystemExit(f"[mirror_reward] 원문이 {out.count(a)}번 있다(1번이어야): {a!r}\n"
                             "  보상 코드가 이 방향 관용구를 안 쓰면 RULES 를 갱신하고 테스트도 같이 고칠 것")
        out = out.replace(a, b)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--to", required=True, choices=["l", "r"], help="만들 쪽")
    a = ap.parse_args()
    text = mirror(a.src.read_text(encoding="utf-8"), a.to)
    a.dst.parent.mkdir(parents=True, exist_ok=True)
    a.dst.write_text(text, encoding="utf-8")
    print(f"[mirror_reward] {a.src} → {a.dst} ({'우→좌' if a.to == 'l' else '좌→우'}, 바뀐 줄 {len(RULES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
