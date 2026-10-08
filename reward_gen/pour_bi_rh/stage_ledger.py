"""pour_bi_rh 단계 원장: 접근 → 손가락 사이에 컵 → 파지 → 들기 → 모으기 → 기울이기 → 붓기 순서로
결정론 상태를 만들어 스텝당 보상이 단계마다 오르는지(골·절벽이 없는지) 본다.

사용법: python3 stage_ledger.py <compute_reward.py> [<compute_reward.py> ...]
float64 CPU. 상태는 check_iter06.py 의 make_ctx/pour_state 를 쓴다(컵 치수·테이블 높이 동일).
"""
import importlib.util
import math
import sys

import torch

HERE = "/home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/"


def load_check(cand):
    sys.argv = ["x", cand]
    spec = importlib.util.spec_from_file_location("chk", HERE + "archive_r25-34/iter_08/check_iter08.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ★09.29 인벨롭 기준 자세(i24 ep500 왼손 5지 감싸 들기 34k 스텝 중앙): 손바닥-컵 축 1.5 r, 높이 원점 +0.15 x 입구,
#   법선·손가락 축 수평, (손바닥→컵 축 수평 방향) = 0.67 x 법선 + 0.73 x 손가락 축. 손바닥은 컵의 -x 쪽이라 그 방향 = +x.
ENV_AXES = {"envelope": ([0.67, -0.742, 0.0], [0.742, 0.67, 0.0]),     # (법선, 손가락 축)
            "point": ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),                 # 손끝이 컵 축을 겨눔(이전 원장 자세)
            "vertical": ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0])}             # 손바닥은 컵을 보고 손가락이 아래(f01 오른손)


def _side_sign(cp):
    """리시버(왼손, 컵 y>0) = +1, 소스(오른손, 컵 y<0) = −1. 원장 자세는 리시버 기준으로 만들고 소스는 y 를 뒤집는다."""
    return -1.0 if float(cp[:, 1].mean()) < 0.0 else 1.0


def hand(m, cp, gap, closure, f_t, f_o, grasped, palm_h=None, palm_rad=None, n=64, axes="envelope"):
    """cup_p 둘레 손: 엄지 +x 쪽, 네 손가락 맞은편, 손바닥은 로봇 쪽(-x) 옆. axes = ENV_AXES 키(기본 인벨롭)."""
    R, MZ = m.R, m.MZ
    h = 0.25 * MZ
    rr = R + 0.0005 + gap
    sy = _side_sign(cp)                  # ★09.29 좌우 미러: 소스(컵 y<0)는 y 를 뒤집는다(사용자 '둘 다 대칭')
    def ring(a):
        return cp + torch.tensor([rr * math.cos(a), sy * rr * math.sin(a), h])
    tips = torch.stack([ring(math.pi / 2)] + [ring(-math.pi / 2 + da) for da in (-0.26, -0.09, 0.09, 0.26)], 1)
    prad = 1.5 * R if palm_rad is None else palm_rad
    ph = 0.15 * MZ if palm_h is None else palm_h
    palm = cp + torch.tensor([-prad, 0.0, ph])
    nv, fv = ENV_AXES[axes]
    axes = torch.tensor([nv[0], sy * nv[1], nv[2], fv[0], sy * fv[1], fv[2]]).repeat(n, 1)
    ff = torch.zeros(n, 5); ff[:, 0] = f_t; ff[:, 1:] = f_o / 4.0
    return tips, palm, axes, ff, closure * torch.ones(n), torch.full((n,), bool(grasped))


def hook_hand(m, cp, n=64, thumb_rad=0.57, thumb_h=0.047, f_thumb=12.2, grasped=True):
    """★09.29 f07 ep1000 오른손 실측(엄지 입구 안 걸기): 엄지 축 0.57 r·높이 4.7 cm·12.2 N, 검지 0.96 r·3.2 cm·1.8 N, 약지 1.10 r·1.0 cm·0.7 N,
    중지 2.07 r·새끼 1.48 r 은 안 닿음, 손바닥 축 2.83 r·높이 4.3 cm. 리시버 기준으로 만들고 소스는 y 미러."""
    R = m.R
    sy = _side_sign(cp)
    def at(rad, ang, h):
        return cp + torch.tensor([rad * R * math.cos(ang), sy * rad * R * math.sin(ang), h])
    fingers = ((0.96, 0.032, 1.8), (2.07, 0.008, 0.0), (1.10, 0.010, 0.7), (1.48, -0.006, 0.0))
    tips = [at(thumb_rad, math.pi / 2, thumb_h)] + [at(rr, -math.pi / 2 + da, hh) for (rr, hh, _f), da in zip(fingers, (-0.26, -0.09, 0.09, 0.26))]
    tips = torch.stack(tips, 1)
    palm = cp + torch.tensor([-2.83 * R, 0.0, 0.043])
    nv, fv = ENV_AXES["envelope"]
    ax = torch.tensor([nv[0], sy * nv[1], nv[2], fv[0], sy * fv[1], fv[2]]).repeat(n, 1)
    ff = torch.zeros(n, 5); ff[:, 0] = f_thumb
    for k, (_r, _h, fk) in enumerate(fingers):
        ff[:, k + 1] = fk
    return tips, palm, ax, ff, 0.7 * torch.ones(n), torch.full((n,), bool(grasped))


def start_hand(m, cp, n=64):
    """시작 자세: 손바닥이 컵 축에서 수평 15.8 cm(로봇 쪽), 위 +15.1 cm, 손가락 아래(중립), 손 편 채."""
    palm = cp + torch.tensor([-0.158, 0.0, 0.151])
    tips = (palm + torch.tensor([0.0, 0.0, -0.10])).unsqueeze(1).repeat(1, 5, 1)
    axes = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]).repeat(n, 1)
    return tips, palm, axes, torch.zeros(n, 5), torch.zeros(n), torch.zeros(n, dtype=torch.bool)


NUM_ACTIONS = 24   # ★09.28 full-joint(pour_fj) 후보는 --actions 26


def scene(m, src_h, rcv_h, ps, pr, us=None, in_t=0.0, d_in=0.0, n=64, lift_s=0.0, lift_r=0.0, push=None):
    """push = (dx, dy): 컵이 스폰에서 수평으로 밀린 거리 [m](들지 않은 채). 스폰은 밀기 전 자리로 둔다."""
    up = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
    us = up if us is None else us
    c = m.make_ctx(ps.clone(), us.clone(), pr.clone(), up.clone(), prog=0.3, in_t=in_t, d_in=d_in)
    c.actions, c.prev_actions = torch.zeros(n, NUM_ACTIONS), torch.zeros(n, NUM_ACTIONS)
    for side, cp, hh, lf in (("src", ps, src_h, lift_s), ("rcv", pr, rcv_h, lift_r)):
        spawn = cp.clone(); spawn[:, 2] = m.TABLE_Z - m.BZ
        if push is not None:
            spawn[:, 0] -= push[0]; spawn[:, 1] -= push[1]
        setattr(c, f"{side}_cup_spawn_pos", spawn)
        if hh is None:
            continue                                     # make_ctx 기본 손(쥔 상태) 유지
        t, pa, ax, ff, cl, gr = hh
        setattr(c, f"{side}_tips_pos", t); setattr(c, f"{side}_palm_pos", pa); setattr(c, f"{side}_palm_axes", ax)
        setattr(c, f"{side}_finger_force", ff); setattr(c, f"{side}_hand_closure", cl); setattr(c, f"{side}_grasped", gr)
        setattr(c, f"{side}_palm_force", torch.zeros(n))
    return c


def ledger(cand):
    m = load_check(cand)
    NS = m.NS
    n = 64
    base_s = torch.tensor([0.382, -0.159, m.TABLE_Z - m.BZ]).repeat(n, 1)
    base_r = torch.tensor([0.378, 0.165, m.TABLE_Z - m.BZ]).repeat(n, 1)
    rows = []

    def add(name, c):
        m.reset_cache()
        r, comps = NS["compute_reward"](c)
        rows.append((name, float(r.mean()), {k: float(v.mean()) for k, v in comps.items()}))

    # 1~6: 두 손이 같은 단계로 함께 나아간다
    add("0 시작(두 손 22 cm·위 15 cm)", scene(m, start_hand(m, base_s), start_hand(m, base_r), base_s, base_r))
    # ★09.29 손바닥을 옮기면 손끝도 같은 만큼 옮긴다 — 전에는 손끝이 시작 자리에 남아 손끝 기준 보상(2번째 사이클 iter_04)이 0~1b 에서 평평했다
    def moved(cp, palm_off, cl=None):
        t, pa, ax, ff, c0, gr = start_hand(m, cp)
        new_pa = cp + torch.tensor(palm_off)
        t = t + (new_pa - pa).unsqueeze(1)
        return t, new_pa, ax, ff, (c0 if cl is None else cl * torch.ones(n)), gr
    mid = lambda cp: moved(cp, [-0.10, 0.0, 0.06])
    add("1 절반 접근(수평 10·위 6 cm)", scene(m, mid(base_s), mid(base_r), base_s, base_r))
    # ★09.28 full-joint 재편: 상태형 접근의 주차 후보 — 옆 12 cm·잡는 높이에 편 손으로 정지(i08 주차 자리). 2 보다 낮아야 한다.
    # ★09.29 주차 자리는 테이블과 수평인 손(인벨롭 축, 손끝은 손가락 축 +8 cm·엄지는 법선 +7 cm — f01 실측)으로 둔다.
    #   예전처럼 시작 자세(손가락 아래)를 잡는 높이로 내리면 손끝이 테이블 아래 8.7 cm 로 들어가 손끝 기준 보상이 물리적으로 없는 자세를 본다.
    def park(cp, cl):
        sy = _side_sign(cp)
        nv, fv = (torch.tensor([v[0], sy * v[1], v[2]]) for v in ENV_AXES["envelope"])
        pa = cp + torch.tensor([-0.12, 0.0, 0.25 * m.MZ])
        tips = torch.stack([pa + 0.07 * nv] + [pa + 0.08 * fv + dz * torch.tensor([0.0, 0.0, 1.0]) for dz in (0.015, 0.005, -0.005, -0.015)], 1)
        ax = torch.cat([nv, fv]).repeat(n, 1)
        return tips, pa, ax, torch.zeros(n, 5), cl * torch.ones(n), torch.zeros(n, dtype=torch.bool)
    add("1b 옆 12 cm·잡는 높이, 편 손 정지", scene(m, park(base_s, 0.0), park(base_r, 0.0), base_s, base_r))
    add("2 옆·잡는 높이, 손끝 5 cm", scene(m, hand(m, base_s, 0.05, 0.0, 0, 0, False), hand(m, base_r, 0.05, 0.0, 0, 0, False), base_s, base_r))
    # 비교 자세(골 판정 아님): 같은 위치·손끝에서 손 방향만 다르다 — 2(인벨롭)보다 낮아야 한다
    add("2f 손끝 겨눔(비교)", scene(m, hand(m, base_s, 0.05, 0.0, 0, 0, False, axes="point"), hand(m, base_r, 0.05, 0.0, 0, 0, False, axes="point"), base_s, base_r))
    add("2v 수직 손(비교)", scene(m, hand(m, base_s, 0.05, 0.0, 0, 0, False, axes="vertical"), hand(m, base_r, 0.05, 0.0, 0, 0, False, axes="vertical"), base_s, base_r))
    add("3 컵이 손가락 사이, 손끝 1 cm", scene(m, hand(m, base_s, 0.01, 0.2, 0, 0, False), hand(m, base_r, 0.01, 0.2, 0, 0, False), base_s, base_r))
    add("4 벽에 닿음·닫힘 0.7", scene(m, hand(m, base_s, 0.0, 0.7, 0, 0, False), hand(m, base_r, 0.0, 0.7, 0, 0, False), base_s, base_r))
    add("5 다섯 손가락 1.5 N 파지", scene(m, hand(m, base_s, 0.0, 0.8, 1.5, 6.0, True), hand(m, base_r, 0.0, 0.8, 1.5, 6.0, True), base_s, base_r))
    # ★09.29 f03 왼손 실측: 위에서 세로 집기(손가락 축 아래, 손바닥 입구 높이·축에서 2.8 r). 같은 파지 힘이면 5 보다 1.0 이상 낮아야 한다
    add("5v 세로 집기 파지(비교)", scene(m, hand(m, base_s, 0.0, 0.8, 1.5, 6.0, True, palm_h=0.95 * m.MZ, palm_rad=2.8 * m.R, axes="vertical"),
                                    hand(m, base_r, 0.0, 0.8, 1.5, 6.0, True, palm_h=0.95 * m.MZ, palm_rad=2.8 * m.R, axes="vertical"), base_s, base_r))
    # ★09.29 사용자 "인벨롭이라 5지 전부 참여": f05 오른손 실측(엄지·중지 두 손가락 끝). 같은 자리·같은 방향에서 두 손가락만 닿으면 5 보다 1.0 이상 낮아야 한다
    def two_finger(cp):
        t, pa, ax, ff, cl, gr = hand(m, cp, 0.0, 0.8, 1.5, 6.0, True)
        ff = torch.zeros_like(ff); ff[:, 0] = 1.5; ff[:, 2] = 1.5
        return t, pa, ax, ff, cl, gr
    add("5t 엄지·중지 두 손가락 파지(비교)", scene(m, two_finger(base_s), two_finger(base_r), base_s, base_r))
    # ★09.29 사용자 "2,3 개만 붙이고 다른 건 안 붙여도 된다고 생각할 것": 손가락 수별 파지(엄지 + 검지부터 차례로) — 마지막 손가락 몫이 가장 커야 한다(검사 [43])
    def k_finger(cp, k):
        t, pa, ax, ff, cl, gr = hand(m, cp, 0.0, 0.8, 1.5, 6.0, True)
        ff = torch.zeros_like(ff); ff[:, 0] = 1.5; ff[:, 1:1 + k] = 1.5
        return t, pa, ax, ff, cl, gr
    for k in (2, 3):
        add(f"5_{k + 1} 엄지 + {k} 손가락 파지(비교)", scene(m, k_finger(base_s, k), k_finger(base_r, k), base_s, base_r))
    add("5h 엄지 입구 안 걸기 파지(비교)", scene(m, hook_hand(m, base_s), hook_hand(m, base_r), base_s, base_r))
    # ★09.28 i09 실측 골: 파지 시도 중 컵이 밀린 상태(들기 전). 밀린 파지도 3(손가락 사이)보다 낮으면 안 된다.
    PUSH = (0.02, 0.01)
    push_s = base_s + torch.tensor([PUSH[0], PUSH[1], 0.0]); push_r = base_r + torch.tensor([PUSH[0], -PUSH[1], 0.0])
    add("3p 손가락 사이·컵 2.2 cm 밀림", scene(m, hand(m, push_s, 0.01, 0.2, 0, 0, False), hand(m, push_r, 0.01, 0.2, 0, 0, False), push_s, push_r, push=PUSH))
    add("1h 옆 12 cm, 반쯤 쥔 손(0.87)", scene(m, park(base_s, 0.87), park(base_r, 0.87), base_s, base_r))
    # 손은 밀린 컵 기준 같은 상대 위치(시작 자세)에 둔다 — 컵이 멀어진 기하 차이를 빼고 '밀린 상태' 비용만 잰다(상태형 접근, 09.28)
    add("0p 물러남(시작 자리)·컵 2.2 cm 밀림", scene(m, start_hand(m, push_s), start_hand(m, push_r), push_s, push_r, push=PUSH))
    add("4p 닫힘 0.7·컵 2.2 cm 밀림", scene(m, hand(m, push_s, 0.0, 0.7, 0, 0, False), hand(m, push_r, 0.0, 0.7, 0, 0, False), push_s, push_r, push=PUSH))
    add("5p 파지·컵 2.2 cm 밀림(들기 전)", scene(m, hand(m, push_s, 0.0, 0.8, 1.5, 6.0, True), hand(m, push_r, 0.0, 0.8, 1.5, 6.0, True), push_s, push_r, push=PUSH))
    for lift in (0.03, 0.08):
        ps = base_s + torch.tensor([0.0, 0.0, lift]); pr = base_r + torch.tensor([0.0, 0.0, lift])
        add(f"6 두 컵 {int(lift*100)} cm 들기", scene(m, hand(m, ps, 0.0, 0.8, 1.5, 6.0, True), hand(m, pr, 0.0, 0.8, 1.5, 6.0, True), ps, pr))
    # ★09.29 7 행부터도 쥔 손은 6 행과 같은 인벨롭 파지(양쪽 1.5 N)를 컵과 함께 옮긴다 — 예전에는 옛 검사 모듈의 기본 손
    #   (손가락 아래·손바닥 축 아래)으로 바뀌어, 들기 전 수입을 쥔 채 유지하는 보상에서 6 → 7 이 손 자세 차이만큼 떨어졌다.
    def carried(cp, u=None):
        t, pa, ax, ff, cl, gr = hand(m, cp, 0.0, 0.8, 1.5, 6.0, True)
        if u is None:
            return t, pa, ax, ff, cl, gr
        ez = torch.tensor([0.0, 0.0, 1.0]).expand_as(u)
        k = torch.cross(ez, u, dim=-1); sn = k.norm(dim=-1, keepdim=True); cs = (ez * u).sum(-1, keepdim=True)
        k = k / sn.clamp(min=1e-9)
        def rot(v):   # Rodrigues, 행마다 ez → u
            return v * cs + torch.cross(k, v, dim=-1) * sn + k * (k * v).sum(-1, keepdim=True) * (1.0 - cs)
        t2 = cp.unsqueeze(1) + torch.stack([rot(t[:, i] - cp) for i in range(t.shape[1])], 1)
        pa2 = cp + rot(pa - cp)
        ax2 = torch.cat([rot(ax[:, :3]), rot(ax[:, 3:])], dim=-1)
        return t2, pa2, ax2, ff, cl, gr
    # pour_state 는 리시버를 y=0 에 둔다 -> rcv_keep(리시버 컵 y >= 0.75r) 벌점이 붙는다. 리시버를 자기 쪽 +5 cm 로 옮긴다.
    YOFF = torch.tensor([0.0, 0.05, 0.0])
    # 7~: 붓기 쪽 (make_ctx 기본 손 = 쥔 상태)
    ps = base_s + torch.tensor([0.0, 0.0, 0.28]); pr = base_r + torch.tensor([0.0, 0.0, 0.08])
    add("7 소스 28 cm·리시버 8 cm, 제자리", scene(m, carried(ps), carried(pr), ps, pr))
    ps2 = torch.tensor([0.39, 0.02, m.TABLE_Z - m.BZ + 0.28]).repeat(n, 1)
    add("8 소스를 리시버 쪽으로 반쯤", scene(m, carried(ps2), carried(pr), ps2, pr))
    for tilt in (0, 45, 70, 95):
        ps3, us3, pr3, ur3 = m.pour_state(n, tilt)
        ps3 = ps3 + YOFF; pr3 = pr3 + YOFF
        add(f"9 입구 위 조준·기울기 {tilt}°", scene(m, carried(ps3, us3), carried(pr3), ps3, pr3, us=us3))
        if tilt == 45:   # ★09.29 사용자 "틸팅 중에도 그 자세 유지": 컵은 45° 기울었는데 손은 세운 자세 그대로(컵 기준 파지가 틀어짐)
            add("9r 기울기 45°, 손이 컵과 함께 돌지 않음(비교)", scene(m, carried(ps3), carried(pr3), ps3, pr3, us=us3))
    ps3, us3, pr3, ur3 = m.pour_state(n, 100)
    ps3 = ps3 + YOFF; pr3 = pr3 + YOFF
    add("10 붓는 중(스텝당 0.05 도착)", scene(m, carried(ps3, us3), carried(pr3), ps3, pr3, us=us3, d_in=0.05))
    return rows


if __name__ == "__main__":
    cands = sys.argv[1:]
    if cands and cands[0] == "--actions":
        NUM_ACTIONS = int(cands[1]); cands = cands[2:]
    all_rows = [ledger(c) for c in cands]
    print("단계".ljust(34) + "".join(f"{c.split('/')[-2]:>12}" for c in cands))
    for i, row in enumerate(all_rows[0]):
        vals = [rows[i][1] for rows in all_rows]
        # 밀린 변형(4p·5p)은 3(손가락 사이)과 비교한다 — 밀린 파지가 그 앞 단계보다 싸면 컵을 떠난다(i09)
        # 0p 는 0(시작)과 비교: 밀린 뒤 물러나도 비용이 남으면 한 번 민 값이 에피소드 끝까지 간다
        if row[0][:2] in ("2f", "2v"):   # 비교 자세: 2(인벨롭)보다 낮아야 정상 — 높거나 같으면 ✗
            r2 = next(j for j, r in enumerate(all_rows[0]) if r[0].startswith("2 "))
            vals_c = [rows[i][1] for rows in all_rows]; ref2 = [rows[r2][1] for rows in all_rows]
            print(row[0].ljust(34) + "".join(f"{v:9.2f}{'  ✗' if v >= q - 1e-6 else '   '}" for v, q in zip(vals_c, ref2)))
            continue
        if row[0][:2] == "5_":           # 손가락 수별 파지: 값만 싣는다(판정은 check_iter06 [43])
            print(row[0].ljust(34) + "".join(f"{v:9.2f}   " for v in vals))
            continue
        if row[0][:2] == "9r":           # 기울일 때 손 자세 틀어짐: 같은 45° 행보다 낮아야 정상
            r45 = next(j for j, r in enumerate(all_rows[0]) if r[0].startswith("9 ") and "45" in r[0])
            ref45 = [rows[r45][1] for rows in all_rows]
            print(row[0].ljust(34) + "".join(f"{v:9.2f}{'  ✗' if v >= q - 1e-6 else '   '}" for v, q in zip(vals, ref45)))
            continue
        if row[0][:2] in ("5v", "5t", "5h"):   # 세로 집기·두 손가락 집기: 5(인벨롭 파지)보다 1.0 이상 낮아야 정상
            r5 = next(j for j, r in enumerate(all_rows[0]) if r[0].startswith("5 "))
            ref5 = [rows[r5][1] for rows in all_rows]
            print(row[0].ljust(34) + "".join(f"{v:9.2f}{'  ✗' if v > q - 1.0 else '   '}" for v, q in zip(vals, ref5)))
            continue
        key = "0 " if row[0].startswith("0p") else "3 "
        ref = next((j for j, r in enumerate(all_rows[0]) if r[0].startswith(key)), None) if row[0][1:2] == "p" else None
        if row[0].startswith("1h"):      # 반쯤 쥔 손은 편 손(1b)보다 낮아야 정상 — 표에 값만 싣고 골 판정은 하지 않는다
            ref = i
        prev = [rows[ref if ref is not None else i - 1][1] if i else None for rows in all_rows]
        # 밀린 변형은 컵이 2 cm 옮겨져 손-컵 기하 항이 조금 바뀐다 — 0.02/스텝 안은 기하 차이로 본다(비용 여부만 가른다)
        tol = 0.02 if row[0][1:2] == "p" else 1e-6
        cells = "".join(f"{v:9.2f}{'' if p is None else ('  ✗' if v < p - tol else '   ')}" for v, p in zip(vals, prev))
        print(row[0].ljust(34) + cells)
    print("✗ = 앞 단계보다 스텝당 수입이 낮다(골). 4p·5p 는 3(손가락 사이)과 비교, 2f·2v 는 2(인벨롭)보다 낮아야 하고 5v 는 5 보다 1.0 이상 낮아야 한다")
