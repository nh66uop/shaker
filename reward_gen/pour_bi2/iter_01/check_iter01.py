#!/usr/bin/env python3
"""pour_bi2 iter_00 결정론 성질 검사.

생성된 보상을 float64 CPU 에서 합성 상태로 직접 평가해, notes_iter00.md 의 요구 R2·R3·R4 를 잰다.

  python3 reward_gen/pour_bi2/iter_00/check_iter00.py reward_gen/pour_bi2/iter_00/compute_reward_c1.py

검사 항목
  A. 회전 수입이 방문 상태에서 살아 있나 (R2)  — 기울기 55도, 립 오프셋 10 cm 에서 회전 항 > 0
  B. 가까울수록 회전 수입이 크다              — 오프셋 6 cm 수입 > 10 cm 수입
  C. 래치 반경 안에서 더 기울이면 이득 (R2+R3) — 오프셋 6 cm 에서 55 → 65 → 75 → 94도 총보상 단조 증가
  D. 래치 반경 안에는 조급한 기울기 청구가 없다 (R3)
  E. 래치 반경 밖 청구는 한계각 0.95 아래에서 0 에 가깝다 (R3)
  F. 상태 수입 합 < 비드 한 알 수입 (R4)
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

_HDGP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HDGP / "source" / "openarm"))
from openarm.agnostic.modules.t2r.validator import make_fake_context  # noqa: E402

torch.set_default_dtype(torch.float64)

CUP_R = 0.0405           # env cfg 의 컵 안쪽 반경 [m] (래치 반경 2r = 8.1 cm)
LIMIT_DEG = 73.6         # premature_tilt_limit p50 (가득 70도, 절반 74도)
N = 1


def load(path: str):
    spec = importlib.util.spec_from_file_location("gen_reward", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward


def state(tilt_deg: float, lip_off_m: float, *, poured: float = 0.0, spilled: float = 0.0,
          lifted: bool = True, adr: float = 0.0):
    """합성 상태 하나. 양손 파지, 기준 기록 유효, 기준과 현재 파지가 같다(keep = 1)."""
    ctx = make_fake_context(N, device="cpu", seed=0)
    one = torch.ones(N)
    zero = torch.zeros(N)
    tilt = math.radians(tilt_deg)

    rcv_mouth = torch.tensor([[0.45, 0.16, 0.95]]).repeat(N, 1)
    lip = rcv_mouth + torch.tensor([[lip_off_m, 0.0, 0.03]]).repeat(N, 1)
    src_cup = lip - torch.tensor([[0.0, 0.0, 0.04]]).repeat(N, 1)
    up = torch.tensor([[math.sin(tilt), 0.0, math.cos(tilt)]]).repeat(N, 1)

    kw = dict(
        cup_radius=CUP_R, adr_progress=adr,
        src_cup_tilt=one * tilt, src_tilt_toward_rcv=one * tilt,
        src_cup_up=up, src_cup_pos=src_cup, src_cup_mouth_pos=lip,
        src_pour_lip_pos=lip, rcv_cup_mouth_pos=rcv_mouth,
        rcv_cup_pos=rcv_mouth - torch.tensor([[0.0, 0.0, 0.06]]).repeat(N, 1),
        rcv_cup_up=torch.tensor([[0.0, 0.0, 1.0]]).repeat(N, 1), rcv_cup_tilt=zero.clone(),
        premature_tilt_limit=one * math.radians(LIMIT_DEG),
        premature_tilt=torch.zeros(N, dtype=torch.bool),
        cups_nested=torch.zeros(N, dtype=torch.bool), cup_hit=torch.zeros(N, dtype=torch.bool),
        success=torch.zeros(N, dtype=torch.bool),
        src_grasped=torch.ones(N, dtype=torch.bool), rcv_grasped=torch.ones(N, dtype=torch.bool),
        src_grasp_ref_valid=one.clone(), rcv_grasp_ref_valid=one.clone(),
        src_wrap_count=one * 5.0, rcv_wrap_count=one * 5.0,
        bead_in_target_frac=one * poured, bead_spill_frac=one * spilled,
        bead_in_source_frac=one * max(0.0, 1.0 - poured - spilled),
        d_in_target=zero.clone(), d_spill=zero.clone(),
        bead_fill_level=one * 0.55, bead_active_count=one * 12.0,
        pour_target_frac=one * 0.5, pour_target_beads=one * 6.0,
        cup_cup_force=zero.clone(), src_hand_foreign_force=zero.clone(),
        rcv_hand_foreign_force=zero.clone(),
        pour_dir_xy=torch.tensor([[-0.2, -0.98]]).repeat(N, 1),
        rcv_side_margin=one * 0.10, episode_progress=one * 0.4,
        actions=torch.zeros(N, 18), prev_actions=torch.zeros(N, 18),
        src_arm_qd=torch.zeros(N, 7), rcv_arm_qd=torch.zeros(N, 7),
        src_cup_lin_vel=torch.zeros(N, 3), src_cup_ang_vel=torch.zeros(N, 3),
        rcv_cup_lin_vel=torch.zeros(N, 3), rcv_cup_ang_vel=torch.zeros(N, 3),
    )
    ctx = replace(ctx, **kw)
    # 들기: 스폰보다 위, 손바닥은 컵 옆
    ctx = replace(ctx,
                  src_cup_spawn_pos=ctx.src_cup_pos - torch.tensor([[0.0, 0.0, 0.25 if lifted else 0.0]]),
                  rcv_cup_spawn_pos=ctx.rcv_cup_pos - torch.tensor([[0.0, 0.0, 0.20 if lifted else 0.0]]),
                  src_palm_pos=ctx.src_cup_pos + torch.tensor([[0.0, -0.05, 0.0]]),
                  rcv_palm_pos=ctx.rcv_cup_pos + torch.tensor([[0.0, 0.05, 0.0]]))
    # 파지 기준 = 현재(유지 만점)
    ctx = replace(ctx,
                  src_ref_palm_in_cup=ctx.src_palm_in_cup, src_ref_tips_in_cup=ctx.src_tips_in_cup,
                  src_ref_force_share=ctx.src_force_share, src_ref_force_total=ctx.src_force_total,
                  rcv_ref_palm_in_cup=ctx.rcv_palm_in_cup, rcv_ref_tips_in_cup=ctx.rcv_tips_in_cup,
                  rcv_ref_force_share=ctx.rcv_force_share, rcv_ref_force_total=ctx.rcv_force_total)
    return ctx


def ev(fn, **kw):
    total, terms = fn(state(**kw))
    return float(total[0]), {k: float(v[0]) for k, v in terms.items()}


def pick(terms: dict, *keys) -> float:
    return sum(v for k, v in terms.items() if any(s in k.lower() for s in keys))


def main() -> int:
    fn = load(sys.argv[1])
    ok = True
    print("== 상태별 총보상 (립 오프셋 x 기울기)")
    offs = [0.02, 0.04, 0.06, 0.081, 0.10, 0.14, 0.25, 0.37, 0.45]
    tilts = [0, 20, 40, 55, 65, 75, 85, 94, 110, 130]
    print("off\\tilt " + " ".join(f"{t:>8d}" for t in tilts))
    grid = {}
    for o in offs:
        row = []
        for t in tilts:
            tot, terms = ev(fn, tilt_deg=t, lip_off_m=o)
            grid[(o, t)] = (tot, terms)
            row.append(f"{tot:+8.3f}")
        print(f"{o*100:5.1f}cm " + " ".join(row))

    rot_keys = ("tilt", "rot", "pour_pose", "sweep")
    pre_keys = ("premature", "pretilt_pen", "early")
    print("\n== A/B 회전 수입 (항 이름에 tilt/rot 포함)")
    for o in (0.06, 0.10, 0.14):
        v = pick(grid[(o, 55)][1], *rot_keys)
        print(f"  55도 · 오프셋 {o*100:.0f} cm : {v:+.4f}")
    a = pick(grid[(0.10, 55)][1], *rot_keys)
    b = pick(grid[(0.06, 55)][1], *rot_keys)
    if not (abs(a) > 1e-4):
        print("  A FAIL: 오프셋 10 cm 에서 회전 수입이 사실상 0 (i26·i27 의 실패 모드)"); ok = False
    if not (b >= a):
        print("  B FAIL: 가까울수록 회전 수입이 커야 한다"); ok = False

    print("\n== C 래치 반경 안에서 더 기울일 때 순이익 (오프셋 6 cm, 55도 기준)")
    base = grid[(0.06, 55)][0]
    for t in (65, 75, 94):
        d = grid[(0.06, t)][0] - base
        print(f"  +{t-55:>2d}도 : {d:+.4f}")
        if d <= 0:
            print(f"  C FAIL: {t}도로 더 기울일 때 손해다"); ok = False

    print("\n== D/E 조급한 기울기 청구")
    for o in (0.04, 0.06):
        v = min(pick(grid[(o, t)][1], *pre_keys) for t in tilts)
        print(f"  오프셋 {o*100:.0f} cm 최소 청구 : {v:+.4f}")
        if v < -1e-3:
            print("  D FAIL: 래치 반경 안에서 청구가 있다"); ok = False
    for t in (40, 55, 65):
        v = pick(grid[(0.14, t)][1], *pre_keys)
        print(f"  오프셋 14 cm · {t}도 청구 : {v:+.4f}")
        if t <= 55 and v < -0.05:
            print("  E FAIL: 한계각의 0.95 아래에서 청구가 크다"); ok = False

    print("\n== F 상태 수입 합 vs 비드 한 알")
    park, park_terms = ev(fn, tilt_deg=20, lip_off_m=0.06)
    state_income = sum(v for k, v in park_terms.items() if v > 0)
    ctx_a = state(tilt_deg=94, lip_off_m=0.06)
    ctx_b = replace(ctx_a, d_in_target=torch.ones(N) * (1.0 / 12.0),
                    bead_in_target_frac=torch.ones(N) * (1.0 / 12.0),
                    bead_in_source_frac=torch.ones(N) * (11.0 / 12.0))
    bead_pay = float(fn(ctx_b)[0][0] - fn(ctx_a)[0][0])
    print(f"  주차 상태(20도·6 cm) 양의 수입 합 : {state_income:+.3f} / 스텝")
    print(f"  비드 한 알이 들어갈 때 총보상 증가 : {bead_pay:+.3f}")
    if bead_pay <= state_income:
        print("  F FAIL: 한 알 수입이 상태 수입 합보다 작다 (주차가 최적해가 된다)"); ok = False

    print("\n== G 먼 거리에서 두 컵을 모을 유인 (iter_01 의 변경)")
    prev = None
    for o in (0.45, 0.37, 0.25, 0.14, 0.081):
        tot = grid[(o, 20)][0]
        conv = pick(grid[(o, 20)][1], "converge", "aim")
        print(f"  오프셋 {o*100:5.1f} cm · 20도 : 총 {tot:+7.3f} · 접근계 수입 {conv:+6.3f}")
        if prev is not None and tot <= prev + 1e-3:
            print("  G FAIL: 가까이 갈 때 총보상이 늘지 않는다"); ok = False
        prev = tot
    slope = (grid[(0.25, 20)][0] - grid[(0.37, 20)][0]) / 0.12
    print(f"  0.37 m -> 0.25 m 기울기 : {slope:+.3f} / m")
    if slope < 0.5:
        print("  G FAIL: 방문 거리에서 접근 기울기가 너무 작다(iter_00 의 실패 모드)"); ok = False

    print("\n== 결과:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
