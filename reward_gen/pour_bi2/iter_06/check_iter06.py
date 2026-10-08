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
    offs = [0.0, 0.02, 0.04, 0.06, 0.081, 0.10, 0.123, 0.14, 0.25, 0.37, 0.45]
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

    print("\n== H 조준이 회전 수입을 올리되 끄지는 않는가 (iter_02 의 변경 1)")
    rot_keys2 = ("tilt", "rot", "pour_pose", "sweep")
    perfect = pick(grid[(0.0, 75)][1], *rot_keys2)
    visited = pick(grid[(0.123, 75)][1], *rot_keys2)
    print(f"  75도 · 완전 조준(0 cm)      : {perfect:+.3f} / 스텝")
    print(f"  75도 · 실측 방문(12.3 cm)   : {visited:+.3f} / 스텝  ({(visited/perfect*100 if perfect else 0):.0f} % of perfect)")
    if visited < 1.5:
        print("  H FAIL: 방문 상태에서 회전 수입이 1.5 미만 — i26·i27 의 게이트 0 실패 모드"); ok = False
    if perfect <= 0 or visited < 0.30 * perfect:
        print("  H FAIL: 방문 상태 수입이 완전 조준의 30 % 미만"); ok = False
    if perfect <= visited:
        print("  H FAIL: 조준을 잘해도 수입이 늘지 않는다"); ok = False
    prev = None
    mono = True
    for o in (0.14, 0.123, 0.10, 0.081, 0.06, 0.04, 0.02, 0.0):
        v = pick(grid[(o, 75)][1], *rot_keys2)
        if prev is not None and v < prev - 1e-6:
            mono = False
        prev = v
    print(f"  오프셋이 줄 때 단조 증가 : {'예' if mono else '아니오'}")
    if not mono:
        print("  H FAIL: 오프셋이 줄어드는데 회전 수입이 줄어드는 구간이 있다"); ok = False

    print("\n== I 떨림·부호반전 청구가 ADR 0 에서 정확히 0 인가 (iter_03 의 변경)")
    a = torch.full((N, 18), 0.5); pa = a.clone(); pa[:, :9] *= -1
    for adr in (0.0, 1.0):
        c = state(tilt_deg=20, lip_off_m=0.30, lifted=False, adr=adr)
        c = replace(c, actions=a, prev_actions=pa,
                    rcv_arm_qd=torch.full((N, 7), 0.46), src_arm_qd=torch.full((N, 7), 0.42),
                    src_grasped=torch.zeros(N, dtype=torch.bool), rcv_grasped=torch.zeros(N, dtype=torch.bool))
        _, t = fn(c)
        ch = sum(float(v[0]) for k, v in t.items() if any(s in k for s in ("tremor", "flip")))
        pos = sum(float(v[0]) for v in t.values() if float(v[0]) > 0)
        print(f"  adr={adr:.0f} · 파지 전 : 떨림+부호반전 {ch:+.4f} / 양의 수입 {pos:+.4f}")
        if adr == 0.0 and abs(ch) > 1e-9:
            print("  I FAIL: ADR 0 인데 떨림·부호반전 청구가 있다 — i02l 의 정지 수렴 원인"); ok = False

    print("\n== K 세우기 문턱 (iter_04 의 변경): 목표 5 알, 활성 10 알")
    def fin(tilt, poured, spilled, in_src, target=5.0):
        c = state(tilt_deg=tilt, lip_off_m=0.05)
        c = replace(c, pour_target_beads=torch.ones(N) * target, pour_target_frac=torch.ones(N) * target / 10.0,
                    bead_active_count=torch.ones(N) * 10.0, bead_in_target_frac=torch.ones(N) * poured / 10.0,
                    bead_spill_frac=torch.ones(N) * spilled / 10.0, bead_in_source_frac=torch.ones(N) * in_src / 10.0)
        tot, terms = fn(c)
        return float(tot[0]), float(terms["pour_rot"][0]), float(terms["finish_upright"][0])
    rows = {
        "a 한 알 모자란 채 95도 버티기": fin(95, 4, 0, 6),
        "a2 두 알 모자란 채 95도 (계속 부어야)": fin(95, 3, 0, 7),
        "b 공차 안(4알) 세움 5도": fin(5, 4, 0, 6),
        "b2 정확히 5알 세움 5도": fin(5, 5, 0, 5),
        "c 전부 쏟고 세움 5도": fin(5, 0, 10, 0),
        "d 붓기 전 컵 세워 들고 있기": fin(5, 0, 0, 10),
    }
    for k, (t, r, f) in rows.items():
        print(f"  {k:32s} 총 {t:+7.3f}  회전 {r:+6.3f}  세우기 {f:+6.3f}")
    if not rows["b 공차 안(4알) 세움 5도"][0] > rows["a 한 알 모자란 채 95도 버티기"][0] + 3:
        print("  K FAIL: 공차 안에서 세우는 쪽이 기울여 버티기보다 확실히 이득이어야 한다"); ok = False
    if not rows["a2 두 알 모자란 채 95도 (계속 부어야)"][1] > 1.0:
        print("  K FAIL: 두 알 이상 모자라면 회전 수입이 살아 있어야 한다"); ok = False
    if not rows["c 전부 쏟고 세움 5도"][2] <= 1.5 + 1e-6:
        print("  K FAIL: 쏟고 세우기가 1.5 넘게 받는다"); ok = False
    if rows["d 붓기 전 컵 세워 들고 있기"][2] > 1e-6:
        print("  K FAIL: 붓기 전 세워 들고 있기만 해도 세우기 수입이 나온다"); ok = False

    print("\n== L 소스 파지 감쌈이 회전 수입을 올리되 끄지 않는가 (iter_04 의 변경 2)")
    rots = {}
    for w in (1.0, 1.5, 3.0, 5.0):
        c = state(tilt_deg=95, lip_off_m=0.0)
        c = replace(c, src_wrap_count=torch.ones(N) * w, pour_target_beads=torch.ones(N) * 8.0,
                    pour_target_frac=torch.ones(N) * 0.8, bead_active_count=torch.ones(N) * 10.0,
                    bead_in_target_frac=torch.ones(N) * 0.2, bead_in_source_frac=torch.ones(N) * 0.8)
        _, t = fn(c)
        rots[w] = float(t["pour_rot"][0])
        tip = sum(float(v[0]) for k, v in t.items() if "tip" in k)
        print(f"  감쌈 {w:3.1f}지 : 회전 수입 {rots[w]:+.3f}  손끝 청구 {tip:+.3f}")
    if not (rots[1.5] < rots[3.0] <= rots[5.0] + 1e-9):
        print("  L FAIL: 감쌈이 늘 때 회전 수입이 늘어야 한다"); ok = False
    if rots[1.5] < 0.3 * rots[5.0]:
        print("  L FAIL: 1.5지에서 회전 수입이 0 에 가깝다(게이트가 되면 안 된다)"); ok = False

    print("\n== N 회전 수입에 각도 상한이 없는가 (iter_05 의 변경)")
    prev = None; vals = {}
    for a in (60, 100, 120, 140, 160, 180):
        c = state(tilt_deg=a, lip_off_m=0.0)
        c = replace(c, src_wrap_count=torch.ones(N) * 5.0, pour_target_beads=torch.ones(N) * 10.0,
                    pour_target_frac=torch.ones(N), bead_active_count=torch.ones(N) * 10.0,
                    bead_in_target_frac=torch.ones(N) * 0.5, bead_in_source_frac=torch.ones(N) * 0.5)
        _, t = fn(c)
        vals[a] = float(t["pour_rot"][0])
        print(f"  {a:3d}도 : 회전 수입 {vals[a]:+.3f}")
    if not (vals[140] > vals[100] + 0.3 and vals[180] > vals[140]):
        print("  N FAIL: 100도 이후에도 뒤집을수록 회전 수입이 올라야 한다(각도 상한 금지)"); ok = False
    import inspect, re
    src = open(sys.argv[1]).read()
    hits = [l.strip() for l in src.splitlines() if "premature_tilt_limit" in l or re.search(r"\blim\b", l)]
    print("  premature_tilt_limit / lim 을 쓰는 줄:")
    for h in hits: print("    ", h[:110])

    print("\n== O 요청량을 채운 뒤 90도 넘게 기울어 있어도 세우기 신호가 있는가 (iter_05 재검토)")
    tilts = (150, 130, 115, 100, 90, 60, 20, 5)
    fins = {a: fin(a, 5, 0, 5)[2] for a in tilts}
    for a in tilts:
        print(f"  {a:3d}도 : 세우기 수입 {fins[a]:+.3f}")
    if not all(fins[b] > fins[a] for a, b in zip(tilts, tilts[1:])):
        print("  O FAIL: 뒤집힌 쪽에서 세울수록 세우기 수입이 매 구간 올라야 한다"); ok = False

    print("\n== P 동작 벌점이 없는가 (iter_06 의 변경: 떨림은 제어 쪽에서)")
    a = torch.full((N, 18), 0.9); pa = -a
    for adr in (0.0, 1.0):
        c = state(tilt_deg=20, lip_off_m=0.05, adr=adr)
        c1 = replace(c, actions=a, prev_actions=pa, rcv_arm_qd=torch.full((N, 7), 2.0), src_arm_qd=torch.full((N, 7), 2.0))
        c0 = replace(c, actions=torch.zeros(N, 18), prev_actions=torch.zeros(N, 18),
                     rcv_arm_qd=torch.zeros(N, 7), src_arm_qd=torch.zeros(N, 7))
        d = float(fn(c1)[0][0] - fn(c0)[0][0])
        print(f"  adr={adr:.0f} : 액션 반전·관절속도 2 rad/s 가 총보상을 바꾼 양 {d:+.6f}")
        if abs(d) > 1e-6:
            print("  P FAIL: 액션·관절속도가 보상에 들어 있다"); ok = False
    src = open(sys.argv[1]).read()
    names = [k for k in ("action_rate", "action_flip", "arm_tremor", "action_reg_adr") if f'"{k}"' in src]
    print("  남은 벌점 항 이름:", names or "없음")
    if names:
        ok = False

    print("\n== 결과:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
