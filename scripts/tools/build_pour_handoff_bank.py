#!/usr/bin/env python3
"""rh_aglt 쥔 상태 두 개(우 = 소스, 비드 든 컵 · 좌 = 리시버, 빈 컵)를 짝지어 pour_fabric_mimic 시작 뱅크로 만든다.

10.01 사용자: "둘로 나누고 100 % 뱅크로 · aglt 할 때 미리 비드를 넣어 둔 상태로 모으기".
  수집: rh_aglt_{r,l} play + `env.grasp_capture_path=… env.bead_count=20(우만)` → GraspCapture npz.
  여기: 우 항목마다 좌 항목을 무작위로 골라 두 컵·두 손바닥이 충분히 떨어진 짝만 남긴다.
  관절 상태는 같은 자산(openarm_rh56f1_bi_rl)이라 이름으로 합친다 — r_* 는 우 수집, l_* 는 좌 수집.
출력 필드는 pour_fabric_mimic/start_bank.py `_KEYS` + 이름 배열(start_bank 가 env 순서와 대조).

사용:
  python3 scripts/tools/build_pour_handoff_bank.py --right r.npz --left l.npz --out bank.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _names(z, key: str) -> list[str]:
    return [str(x) for x in z[key]]


def _side_mask(names: list[str], prefix: str) -> np.ndarray:
    return np.array([n.startswith(prefix) for n in names])


def build(r, l, *, min_cup_sep: float, min_palm_sep: float, tries: int, seed: int) -> dict[str, np.ndarray]:
    jn = _names(r, "joint_names")
    if jn != _names(l, "joint_names"):
        raise SystemExit("우·좌 수집의 joint_names 가 다르다 — 같은 자산이 아니다")
    m_r, m_l = _side_mask(jn, "r_"), _side_mask(jn, "l_")
    other = [n for n, a, b in zip(jn, m_r, m_l) if not (a or b)]
    if [n for n in other if not n.startswith("head_")]:
        raise SystemExit(f"r_/l_/head_ 가 아닌 관절이 있다(어느 쪽 값을 쓸지 정해야 한다): {other}")
    # head_* 는 두 aglt env 모두 같은 고정 자세 — 우 수집 값을 쓴다
    if "beads" not in r.files:
        raise SystemExit("우(소스) 수집에 beads 가 없다 — env.bead_count 로 모아야 한다")
    rng = np.random.default_rng(seed)
    nr, nl = len(r["joint_pos"]), len(l["joint_pos"])
    pick_r, pick_l = [], []
    for i in range(nr):
        cand = rng.integers(0, nl, size=tries)
        d_cup = np.linalg.norm(r["cup_root"][i, :2] - l["cup_root"][cand, :2], axis=1)
        d_palm = np.linalg.norm(r["palm_pos"][i] - l["palm_pos"][cand], axis=1)
        ok = np.nonzero((d_cup >= min_cup_sep) & (d_palm >= min_palm_sep))[0]
        if len(ok):
            pick_r.append(i)
            pick_l.append(int(cand[ok[0]]))
    if not pick_r:
        raise SystemExit("거리 조건을 만족하는 짝이 없다")
    ir, il = np.array(pick_r), np.array(pick_l)
    take_l = m_l[None]                                           # l_* 만 좌 수집, 나머지(r_*, head_*)는 우 수집
    jp = np.where(take_l, l["joint_pos"][il], r["joint_pos"][ir])
    jv = np.where(take_l, l["joint_vel"][il], r["joint_vel"][ir])
    return {
        "joint_pos": jp.astype(np.float32), "joint_vel": jv.astype(np.float32),
        "src_root": r["cup_root"][ir], "rcv_root": l["cup_root"][il], "beads": r["beads"][ir],
        "src_arm_q_target": r["arm_q_target"][ir], "rcv_arm_q_target": l["arm_q_target"][il],
        "src_syn_target": r["hand_target"][ir], "rcv_syn_target": l["hand_target"][il],
        "src_spawn": r["spawn"][ir], "rcv_spawn": l["spawn"][il],
        "joint_names": np.array(jn),
        "src_arm_names": r["arm_names"], "rcv_arm_names": l["arm_names"],
        "src_hand_names": r["hand_names"], "rcv_hand_names": l["hand_names"],
        "src_idx": ir, "rcv_idx": il,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--right", required=True, help="우(소스) GraspCapture npz — 비드 포함")
    ap.add_argument("--left", required=True, help="좌(리시버) GraspCapture npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_cup_sep", type=float, default=0.12, help="두 컵 xy 거리 하한 [m] (지름 57 mm × 2 + 여유)")
    ap.add_argument("--min_palm_sep", type=float, default=0.15, help="두 손바닥 거리 하한 [m]")
    ap.add_argument("--tries", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    r, l = np.load(a.right), np.load(a.left)
    out = build(r, l, min_cup_sep=a.min_cup_sep, min_palm_sep=a.min_palm_sep, tries=a.tries, seed=a.seed)
    np.savez_compressed(a.out, **out)
    n = len(out["joint_pos"])
    print(f"뱅크 {n} 개(우 {len(r['joint_pos'])} 중) → {a.out} · 비드 {out['beads'].shape[1]} · "
          f"소스 들림 {np.median(r['lift'][out['src_idx']]) * 100:.1f} cm · 리시버 들림 "
          f"{np.median(l['lift'][out['rcv_idx']]) * 100:.1f} cm (중앙값)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
