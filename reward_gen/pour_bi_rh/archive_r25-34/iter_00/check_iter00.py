"""iter_00 후보의 결정론 성질 검사 (float64 CPU, 학습 불요).

이번 라운드가 노린 세 가지를 그대로 잰다.
  1. 예산 — 붓는 상태에서 회전·배달 수입이 스텝당 총수입의 절반 이상인가.
     그리고 "두 컵 들고 가만히 있기"(주차)의 스텝당 수입이 붓기보다 확실히 낮은가.
  2. 각도 수입 — 같은 기울기라도 방향이 반대이거나 주둥이가 창 밖이면 지불이 사라지는가.
     기울기 0 에서 70° 로 갈 때 수입이 단조로 늘고, 목표 각을 지나쳐도 벌점이 튀지 않는가(각도 처방 금지).
  3. 관절 여유 — 같은 기울기·같은 위치에서 팔 관절 여유만 다를 때 여유가 큰 쪽이 더 많이 받는가.
  4. 공통 — 유한성, 상태 무보존(같은 입력 두 번 = 같은 값), N·dtype 바뀌어도 동작, 닫힌 경로 차분 합 ≤ 0.

사용법: python3 check_iter00.py [후보.py]     기본 compute_reward_c1.py
"""
import math
import sys
import types

import torch

torch.set_default_dtype(torch.float64)
BASE = "/home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/"
CAND = sys.argv[1] if len(sys.argv) > 1 else BASE + "iter_00/compute_reward_c1.py"
R, MZ, BZ, TABLE_Z = 0.02808, 0.053885, -0.0599, 0.205   # RH56F1 shaker x0.65 (env cfg)
DT = 1.0 / 60.0
POUR_AZ = math.radians(68)          # K2: 소스가 부을 수 있는 방위
N = 1024


def load(path):
    ns = {"RewardContext": object}
    exec(open(path).read(), ns)
    return ns


NS = load(CAND)


def reset_cache():
    """후보가 모듈 수준 dict 캐시를 쓰면 비운다(이름 규약 `_..._CACHE` 류)."""
    for k, v in list(NS.items()):
        if k.startswith("_") and not k.startswith("__") and isinstance(v, dict):
            v.clear()


def make_ctx(ps, us, pr, ur, vs=None, ws=None, prog=0.5, margin=0.5, in_t=0.0, d_in=0.0,
             fingers=None, grasped=True, spill=0.0, d_spill=0.0):
    n = ps.shape[0]
    c = types.SimpleNamespace()
    z0, o = torch.zeros(n), torch.ones(n)
    vs = torch.zeros(n, 3) if vs is None else vs
    ws = torch.zeros(n, 3) if ws is None else ws
    c.src_cup_pos, c.src_cup_up, c.rcv_cup_pos, c.rcv_cup_up = ps, us, pr, ur
    c.src_cup_mouth_pos, c.rcv_cup_mouth_pos = ps + MZ * us, pr + MZ * ur
    c.src_cup_lin_vel, c.rcv_cup_lin_vel = vs, torch.zeros(n, 3)
    c.src_cup_ang_vel, c.rcv_cup_ang_vel = ws, torch.zeros(n, 3)
    c.src_cup_tilt = torch.acos(us[:, 2].clamp(-1, 1))
    c.rcv_cup_tilt = torch.acos(ur[:, 2].clamp(-1, 1))
    c.cup_mouth_z, c.cup_radius, c.cup_bottom_z, c.table_z, c.num_beads = MZ, R, BZ, TABLE_Z, 20
    ring = torch.tensor([[math.cos(a), math.sin(a), 0.0]
                         for a in torch.linspace(0, 2 * math.pi, 6)[:5].tolist()])
    for side, p, u in (("src", ps, us), ("rcv", pr, ur)):
        setattr(c, f"{side}_cup_spawn_pos",
                torch.stack([p[:, 0], p[:, 1], torch.full((n,), TABLE_Z - BZ)], -1))
        setattr(c, f"{side}_tips_pos", p.unsqueeze(1) + 0.3 * MZ * u.unsqueeze(1) + R * ring.unsqueeze(0))
        setattr(c, f"{side}_palm_pos", p - 1.2 * R * u)      # 손바닥은 컵 옆(축에서 한 반지름 남짓)
        setattr(c, f"{side}_finger_force", (2.0 * torch.ones(n, 5)) if fingers is None else fingers)
        setattr(c, f"{side}_palm_force", z0.clone())
        setattr(c, f"{side}_hand_closure", 0.8 * o)
        setattr(c, f"{side}_grasped", torch.full((n,), bool(grasped)))
        setattr(c, f"{side}_arm_qd", torch.zeros(n, 7))
        setattr(c, f"{side}_arm_qdd", torch.zeros(n, 7))
        m = margin if torch.is_tensor(margin) else margin * torch.ones(n, 7)
        setattr(c, f"{side}_arm_limit_margin", m.clone())
        setattr(c, f"{side}_hand_foreign_force", z0.clone())
    c.bead_in_target_frac = in_t * o if not torch.is_tensor(in_t) else in_t
    c.bead_spill_frac = spill * o if not torch.is_tensor(spill) else spill
    c.bead_in_source_frac = (1.0 - c.bead_in_target_frac - c.bead_spill_frac).clamp(min=0.0)
    c.d_in_target = d_in * o if not torch.is_tensor(d_in) else d_in
    c.d_spill = d_spill * o if not torch.is_tensor(d_spill) else d_spill
    c.cup_cup_force = z0.clone()
    c.episode_progress = prog * o if not torch.is_tensor(prog) else prog
    c.cups_nested = torch.zeros(n, dtype=torch.bool)
    c.success = torch.zeros(n, dtype=torch.bool)
    c.actions, c.prev_actions = torch.zeros(n, 24), torch.zeros(n, 24)
    return c


def run(c):
    r, comps = NS["compute_reward"](c)
    return r, {k: v for k, v in comps.items()}


def pour_state(n, tilt_deg, az=POUR_AZ, lip_dxy=0.4 * R, lip_dz=2.0 * R, src_z=0.62, lean_az=None):
    """받는 컵 입구 위에 소스 주둥이를 두고 tilt 만큼 기운 상태.

    `az` 는 두 컵의 배치 방위(받는 컵에서 소스 주둥이로 가는 방향)이고,
    `lean_az` 는 컵이 기우는 방위다. 기본은 둘이 같다(= 받는 컵 쪽으로 기움).
    방향 검사는 배치를 그대로 두고 `lean_az` 만 돌린다.
    """
    ang = math.radians(tilt_deg) * torch.ones(n)
    azt = az * torch.ones(n)
    lat = (az if lean_az is None else lean_az) * torch.ones(n)
    us = torch.stack([torch.sin(ang) * torch.cos(lat), torch.sin(ang) * torch.sin(lat), torch.cos(ang)], -1)
    ur = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
    pr = torch.tensor([0.40, 0.0, TABLE_Z - BZ + 0.10]).repeat(n, 1)
    mouth_r = pr + MZ * ur
    lip = mouth_r + torch.stack([lip_dxy * torch.cos(azt), lip_dxy * torch.sin(azt),
                                 lip_dz * torch.ones(n)], -1)
    ps = lip - MZ * us
    ps[:, 2] = src_z - MZ * us[:, 2]
    return ps, us, pr, ur


def state_center(n, tilt_deg, gap=0.052, lean_off=0.0, az=POUR_AZ, dz=3.0 * R, src_z=0.62):
    """컵 **중심**을 받는 컵 입구에서 gap 만큼 떨어진 곳에 고정하고 기울인다.

    실제 팔은 컵 중심을 잡고 있으므로 기울이면 주둥이가 `MZ·sinθ` 만큼 휘돌아 나간다.
    `lean_off` = 0 이면 받는 컵 쪽으로 기울이고(주둥이가 입구 위로 온다), π 면 반대쪽으로 기운다.
    """
    ang = math.radians(tilt_deg) * torch.ones(n)
    u_az = torch.tensor([math.cos(az), math.sin(az)])
    lean = az + math.pi + lean_off                       # 중심 → 입구 방향이 '받는 컵 쪽'
    us = torch.stack([torch.sin(ang) * math.cos(lean), torch.sin(ang) * math.sin(lean), torch.cos(ang)], -1)
    ur = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
    pr = torch.tensor([0.40, 0.0, TABLE_Z - BZ + 0.10]).repeat(n, 1)
    mouth_r = pr + MZ * ur
    ps = mouth_r.clone()
    ps[:, :2] += gap * u_az
    ps[:, 2] = src_z
    return ps, us, pr, ur


def share(comps, keys):
    tot = sum(float(v.mean()) for v in comps.values())
    part = sum(float(comps[k].mean()) for k in comps if any(s in k for s in keys))
    return part, tot


def main() -> int:
    ok = True
    names = None

    # --- 1. 예산: 붓는 상태 대 주차 상태 -------------------------------------------------
    reset_cache()
    ps, us, pr, ur = pour_state(N, 75.0)
    c_pour = make_ctx(ps, us, pr, ur, prog=0.5, in_t=0.35, d_in=1.0 / 20.0)   # 스텝마다 한 알 들어감
    r_pour, comps_pour = run(c_pour)
    names = sorted(comps_pour)
    reset_cache()
    ps0, us0, pr0, ur0 = pour_state(N, 8.0, lip_dxy=3.0 * R, lip_dz=4.0 * R)
    c_park = make_ctx(ps0, us0, pr0, ur0, prog=0.5, in_t=0.0, d_in=0.0)
    r_park, comps_park = run(c_park)
    pour_i, tot_pour = share(comps_pour, ("pour", "tilt", "deliver", "transfer", "rot", "open", "in_target"))
    print(f"[1] 붓는 상태 스텝당 총수입 {float(r_pour.mean()):+.3f} · 회전·배달 몫 {pour_i:+.3f} "
          f"({pour_i / tot_pour * 100 if tot_pour else float('nan'):.0f} %)")
    print(f"    주차(두 컵 들고 정지) 스텝당 총수입 {float(r_park.mean()):+.3f}")
    if not (pour_i > 0.5 * abs(tot_pour)):
        print("    ✗ 회전·배달 수입이 총수입의 절반 미만")
        ok = False
    if not (float(r_pour.mean()) > float(r_park.mean()) + 1.0):
        print("    ✗ 붓기가 주차보다 확실히 크지 않다")
        ok = False

    # --- 2. 각도 수입: 방향·창 조건 -------------------------------------------------------
    rows = []
    for deg in (0, 20, 40, 55, 70, 85, 100):
        reset_cache()
        ps, us, pr, ur = state_center(N, float(deg))
        r, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
        rows.append((deg, float(r.mean())))
    print("[2] 기울기별 스텝당 수입(방향·창 맞음): " + " · ".join(f"{d}°:{v:+.2f}" for d, v in rows))
    rise = [rows[i + 1][1] - rows[i][1] for i in range(len(rows) - 1)]
    print("    구간 증분: " + " · ".join(f"{rows[i][0]}→{rows[i+1][0]}:{rise[i]:+.3f}" for i in range(len(rise))))
    if min(rise[:4]) < -0.05:                         # 0 → 70° 구간은 사실상 단조 증가여야 한다
        print("    ✗ 0~70° 구간에서 수입이 줄어드는 지점이 있다")
        ok = False
    drop_after = rows[4][1] - rows[6][1]              # 70° → 100°
    if drop_after > 0.5 * max(1e-9, rows[4][1]):
        print(f"    ⚠ 70°를 지나면 수입이 {drop_after:+.2f} 줄어든다(각도 상한 처방 의심)")

    reset_cache()
    ps, us, pr, ur = state_center(N, 75.0, lean_off=math.pi)    # 중심 고정, 받는 컵 반대쪽으로 기울임
    r_wrong, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
    reset_cache()
    ps, us, pr, ur = state_center(N, 75.0, gap=8.0 * R)         # 두 컵이 멀다(창 밖)
    r_far, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
    reset_cache()
    ps, us, pr, ur = state_center(N, 75.0)
    r_right, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
    print(f"    같은 75°: 방향 맞음 {float(r_right.mean()):+.3f} · 반대 방향 {float(r_wrong.mean()):+.3f} "
          f"· 창 밖 {float(r_far.mean()):+.3f}")
    if not (float(r_right.mean()) > float(r_wrong.mean()) + 0.2):
        print("    ✗ 방향이 반대인데 수입이 줄지 않는다")
        ok = False
    if not (float(r_right.mean()) > float(r_far.mean()) + 0.2):
        print("    ✗ 창 밖인데 수입이 줄지 않는다")
        ok = False

    # --- 3. 관절 여유 ---------------------------------------------------------------------
    reset_cache()
    ps, us, pr, ur = state_center(N, 70.0)
    r_hi, _ = run(make_ctx(ps, us, pr, ur, prog=0.5, margin=0.6))
    reset_cache()
    m_lo = 0.6 * torch.ones(N, 7)
    m_lo[:, 6] = 0.01            # j7 이 한계에 붙음
    m_lo[:, 1] = 0.05            # j2 도 거의 한계
    r_lo, _ = run(make_ctx(ps, us, pr, ur, prog=0.5, margin=m_lo))
    d = float(r_hi.mean() - r_lo.mean())
    print(f"[3] 같은 70° 자세에서 관절 여유 큼 {float(r_hi.mean()):+.3f} · j7 한계 {float(r_lo.mean()):+.3f} (차 {d:+.3f})")
    if d <= 1e-6:
        print("    ✗ 관절 여유를 구분하지 않는다(이번 라운드 요구 3 미충족)")
        ok = False

    # --- 4. 공통 성질 ---------------------------------------------------------------------
    reset_cache()
    ps, us, pr, ur = pour_state(64, 60.0)
    c = make_ctx(ps, us, pr, ur, prog=0.5)
    r1, _ = run(c)
    r2, _ = run(c)                                    # 같은 입력 두 번
    same = torch.allclose(r1, r2)
    finite = bool(torch.isfinite(r1).all())
    reset_cache()
    ps, us, pr, ur = pour_state(7, 60.0)              # N 이 바뀌어도 동작
    try:
        r3, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
        nchange = bool(torch.isfinite(r3).all())
    except Exception as e:                            # noqa: BLE001
        nchange, _ = False, print(f"    ✗ N 변경에서 예외: {e}")
    print(f"[4] 유한 {finite} · 같은 입력 재현 {same} · N 변경 {nchange}")
    if not (finite and nchange):
        ok = False
    if not same:
        print("    ⚠ 같은 입력이 다른 값을 낸다(스텝 캐시가 있으면 정상일 수 있다 — 차분 항 확인 필요)")

    # 닫힌 경로: 기울였다 되돌리면 차분 수입 합이 0 이하
    reset_cache()
    cyc = list(range(0, 76, 5)) + list(range(70, -1, -5))
    tot = 0.0
    for i, deg in enumerate(cyc):
        ps, us, pr, ur = state_center(N, float(deg))
        r, _ = run(make_ctx(ps, us, pr, ur, prog=0.3 + i * DT))
        tot += float(r.mean())
    base = 0.0
    reset_cache()
    for i in range(len(cyc)):
        ps, us, pr, ur = state_center(N, 0.0)
        r, _ = run(make_ctx(ps, us, pr, ur, prog=0.3 + i * DT))
        base += float(r.mean())
    print(f"[5] 기울였다 되돌린 닫힌 경로 합 {tot:+.2f} (같은 길이 정지 {base:+.2f})")
    if tot > base + 2.0 * len(cyc):
        print("    ⚠ 되돌림 펌프 의심(왕복만으로 정지보다 스텝당 2 이상 더 번다)")

    print(f"\n항 {len(names)}개: {', '.join(names)}")
    print("판정:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
