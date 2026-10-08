"""Deterministic float64 check of iter_23 change 1 (aim_lip / place_rcv rates from the exact spout geometry).

Runs the WHOLE compute_reward of a candidate on synthetic rigid-body states (stage_cd = 1, receiver upright) and reads
comps["aim_lip"], comps["place_rcv"], diag_aim_dist_r, diag_place_dist_r. Compared against iter_22.
  1. exact rate vs central finite difference of the potential (d_aim, d_place, lip_xy, lip_dz)
  2. pure axial spin of the source cup pays exactly 0
  3. smooth closed cycles (tilt + translate + orbit + axial spin) at dt = 1/60 sum to ~0 (<= +0.05 per loop)
  4. residual probes (not iter_23 changes): symmetric-clamp speed asymmetry, coaxial crossing, and the Isaac Lab
     velocity frame (ctx lin_vel = root_lin_vel_w = COM velocity, while cup_pos is the link origin).
Usage: python3 check_aim_place.py [candidate.py]   (default compute_reward_c1.py)
"""
import math
import sys
import types
import torch

torch.set_default_dtype(torch.float64)
BASE = "/home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/"
CAND = sys.argv[1] if len(sys.argv) > 1 else BASE + "iter_23/compute_reward_c1.py"
R, MZ, BZ, TABLE_Z = 0.02808, 0.053885, -0.0599, 0.205      # RH56F1 shaker x0.65 (env cfg)
COM_C = -0.0324   # COM along the cup axis from the link origin, uniform density over the USD colliders (추정)
DT, W_AIM, W_PLACE, V_PROG = 1.0 / 60.0, 1.5, 1.0, 0.15
M = torch.tensor([0.40, 0.0, TABLE_Z - BZ + MZ + 0.15])     # receiver mouth at its placement height target


def load(path):
    ns = {"RewardContext": object}
    exec(open(path).read(), ns)
    return ns


NEW, OLD = load(CAND), load(BASE + "iter_22/compute_reward.py")


def rot(u, w, s):
    """Rotate u about the (per-row) axis w by |w|*s (Rodrigues)."""
    th = w.norm(dim=-1, keepdim=True) * s
    k = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return u * torch.cos(th) + torch.cross(k, u, dim=-1) * torch.sin(th) + k * (k * u).sum(-1, keepdim=True) * (1 - torch.cos(th))


def make_ctx(ps, us, pr, ur, vs, ws, vr, wr, com_c=0.0):
    """vs / vr are LINK-ORIGIN velocities. com_c != 0 feeds the COM velocity the env really reports."""
    n = ps.shape[0]
    c = types.SimpleNamespace()
    c.src_cup_pos, c.src_cup_up, c.rcv_cup_pos, c.rcv_cup_up = ps, us, pr, ur
    c.src_cup_mouth_pos, c.rcv_cup_mouth_pos = ps + MZ * us, pr + MZ * ur
    c.src_cup_lin_vel = vs + com_c * torch.cross(ws, us, dim=-1)
    c.rcv_cup_lin_vel = vr + com_c * torch.cross(wr, ur, dim=-1)
    c.src_cup_ang_vel, c.rcv_cup_ang_vel = ws, wr
    c.src_cup_tilt = torch.acos(us[:, 2].clamp(-1, 1))
    c.rcv_cup_tilt = torch.acos(ur[:, 2].clamp(-1, 1))
    c.cup_mouth_z, c.cup_radius, c.cup_bottom_z, c.table_z, c.num_beads = MZ, R, BZ, TABLE_Z, 20
    z0, o = torch.zeros(n), torch.ones(n)
    ring = torch.tensor([[math.cos(a), math.sin(a), 0.0] for a in torch.linspace(0, 2 * math.pi, 6)[:5].tolist()])
    for side, p, u in (("src", ps, us), ("rcv", pr, ur)):
        setattr(c, f"{side}_cup_spawn_pos", torch.stack([p[:, 0], p[:, 1], torch.full((n,), TABLE_Z - BZ)], -1))
        setattr(c, f"{side}_tips_pos", p.unsqueeze(1) + 0.3 * MZ * u.unsqueeze(1) + R * ring.unsqueeze(0))
        setattr(c, f"{side}_finger_force", 2.0 * torch.ones(n, 5))
        setattr(c, f"{side}_palm_force", z0.clone())
        setattr(c, f"{side}_hand_closure", 0.8 * o)
        setattr(c, f"{side}_grasped", torch.ones(n, dtype=torch.bool))
        setattr(c, f"{side}_arm_qd", torch.zeros(n, 7))
        setattr(c, f"{side}_arm_qdd", torch.zeros(n, 7))
        setattr(c, f"{side}_arm_limit_margin", 0.5 * torch.ones(n, 7))
        setattr(c, f"{side}_hand_foreign_force", z0.clone())
    c.bead_in_source_frac, c.bead_in_target_frac, c.bead_spill_frac = o.clone(), z0.clone(), z0.clone()
    c.d_in_target, c.d_spill, c.cup_cup_force, c.episode_progress = z0.clone(), z0.clone(), z0.clone(), 0.5 * o
    c.cups_nested, c.success = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    c.actions, c.prev_actions = torch.zeros(n, 24), torch.zeros(n, 24)
    return c


def terms(ns, c):
    _, comps = ns["compute_reward"](c)
    return comps


def advance(st, s, com_c=0.0):
    ps, us, pr, ur, vs, ws, vr, wr = st
    return make_ctx(ps + vs * s, rot(us, ws, s), pr + vr * s, rot(ur, wr, s), vs, ws, vr, wr, com_c)


def q(x, p):
    return float(x.quantile(p)) if x.numel() else float("nan")


# ------------------------------------------------------------------ states near the pour window (iter_22 check style)
torch.manual_seed(0)
N = 4096
pr = (M - torch.tensor([0, 0, MZ])).repeat(N, 1) + 0.02 * torch.randn(N, 3)
ur = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0]) + 0.05 * torch.randn(N, 3), dim=-1)
ang = torch.rand(N) * 1.6
az = math.radians(68) + 0.6 * torch.randn(N)
us = torch.stack([torch.sin(ang) * torch.cos(az), torch.sin(ang) * torch.sin(az), torch.cos(ang)], -1)
ps = pr + MZ * ur - torch.stack([0.05 * torch.cos(az), 0.05 * torch.sin(az), torch.zeros(N)], -1) \
    + torch.tensor([0, 0, 0.04]) + 0.03 * torch.randn(N, 3)
vs, ws = 0.02 * torch.randn(N, 3), 0.3 * torch.randn(N, 3)
vr, wr = 0.01 * torch.randn(N, 3), 0.1 * torch.randn(N, 3)
ST = (ps, us, pr, ur, vs, ws, vr, wr)


def rate_check(ns, name):
    h = 1e-6
    t0 = terms(ns, advance(ST, 0.0))
    tp, tm = terms(ns, advance(ST, h)), terms(ns, advance(ST, -h))
    assert float(t0["diag_stage_cd"].min()) > 0.999, "stage_cd must be 1 in this check"
    out = {}
    for key, pot, w in (("aim_lip", "diag_aim_dist_r", W_AIM), ("place_rcv", "diag_place_dist_r", W_PLACE)):
        pay = t0[key]
        fd = -(tp[pot] - tm[pot]) * R / (2 * h) / V_PROG * w          # what an exact rate would pay (unclamped)
        ok = (pay.abs() < 0.999 * w) & (fd.abs() < 0.999 * w) & (t0[pot] * R > 1e-3)
        err = (pay - fd).abs()[ok]
        out[key] = (int(ok.sum()), q(err, 0.5), q(err, 0.99), float((err > 1e-3).double().mean()))
    print(f"[1] {name}: exact rate vs finite difference (reward units per step, W included)")
    for k, (n, med, p99, frac) in out.items():
        print(f"    {k:9s} n={n}  |pay-fd| median {med:.2e}  p99 {p99:.2e}  frac>1e-3 {frac:.4f}")
    return out


def lip_rate_check():
    h = 1e-6

    def lip(c):
        geo = NEW["_pour_geometry"](c, R)
        vm = NEW["_point_vel"](c.rcv_cup_lin_vel, c.rcv_cup_ang_vel, c.rcv_cup_pos, c.rcv_cup_mouth_pos)
        du = torch.cross(c.src_cup_ang_vel, c.src_cup_up, dim=-1)
        o = NEW["_tilt_open_potential"](c, geo, R, vm, du, R, 3 * R, 0.7 * R, 4 * R, 2 * R)
        return geo, o
    g0, o0 = lip(advance(ST, 0.0))
    gp, _ = lip(advance(ST, h))
    gm, _ = lip(advance(ST, -h))
    res = {}
    for k, dk in (("lip_xy", "dlip_xy"), ("lip_dz", "dlip_dz")):
        fd = (gp[k] - gm[k]) / (2 * h)
        e = (o0[dk] - fd).abs()
        res[k] = (q(e, 0.5), q(e, 0.99), q(o0[dk].abs(), 0.5))
        print(f"[1] {dk}: |exact-fd| median {res[k][0]:.2e} p99 {res[k][1]:.2e} m/s  (|rate| median {res[k][2]:.2e})")
    return res


def axial_check(ns, name):
    s = 3.0 * (2 * torch.rand(N) - 1)                                 # -3..3 rad/s about the cup's own axis
    st = (ps, us, pr, ur, torch.zeros(N, 3), s.unsqueeze(-1) * us, torch.zeros(N, 3), torch.zeros(N, 3))
    t = terms(ns, advance(st, 0.0))
    tp = terms(ns, advance(st, 1e-6))
    geo_move = float((tp["diag_aim_dist_r"] - t["diag_aim_dist_r"]).abs().max()) * R
    a, p = t["aim_lip"], t["place_rcv"]
    # iter_23 notes reading: +0.42 rad/s axial while tilted in the window
    st2 = (ps, us, pr, ur, torch.zeros(N, 3), 0.42 * us, torch.zeros(N, 3), torch.zeros(N, 3))
    t2 = terms(ns, advance(st2, 0.0))
    inw = t2["diag_win"] > 0.5
    pos = (t2["aim_lip"] + t2["place_rcv"])[inw]
    print(f"[2] {name}: pure axial spin  max|aim| {a.abs().max():.2e}  max|place| {p.abs().max():.2e}  "
          f"(geometry moved {geo_move:.1e} m)  at 0.42 rad/s in window: mean {pos.mean():+.4f}, "
          f"mean of positive side {pos.clamp(min=0).sum() / max(int((pos > 0).sum()), 1):+.4f}/step")
    return float(a.abs().max()), float(p.abs().max()), float(pos.clamp(min=0).sum() / max(int((pos > 0).sum()), 1))


def loop_state(t, base, prm, com_c):
    """Closed smooth loop: tilt about a horizontal axis, translation along the lean, orbit about the receiver
    mouth, axial spin. All periodic in T*DT. Returns ctx with exact analytic velocities."""
    p0, u0, rcv_p, rcv_u, k_ax, e_lean, Om, A_t, A_x, A_o, S, ph = base + prm
    a = A_t * math.sin(Om * t)
    da = A_t * Om * math.cos(Om * t)
    x = A_x * math.sin(Om * t + ph[0])
    dx = A_x * Om * math.cos(Om * t + ph[0])
    b = A_o * math.sin(Om * t + ph[1])
    db = A_o * Om * math.cos(Om * t + ph[1])
    rel = p0[:, :2] - M[:2]
    cb, sb = math.cos(b), math.sin(b)
    rel_b = torch.stack([cb * rel[:, 0] - sb * rel[:, 1], sb * rel[:, 0] + cb * rel[:, 1]], -1)
    drel_b = db * torch.stack([-rel_b[:, 1], rel_b[:, 0]], -1)
    p = torch.cat([M[:2] + rel_b, p0[:, 2:]], -1) + x * e_lean
    v = torch.cat([drel_b, torch.zeros(p.shape[0], 1)], -1) + dx * e_lean
    u = rot(u0, k_ax, a)
    w = k_ax * da + S * math.cos(Om * t) * u                           # tilt rate + axial spin (spin angle periodic)
    n = p.shape[0]
    return make_ctx(p, u, rcv_p, rcv_u, v, w, torch.zeros(n, 3), torch.zeros(n, 3), com_c)


def cycle_check(ns, name, com_c=0.0, spin=0.0, T=240, n=512, sub=1, unclamped=False):
    """Loop period T*DT (4 s). sub > 1 samples it sub times finer (payment scaled by 1/sub, same integral).
    'regular' loops stay >= 1 cm from the receiver axis (d_hat defined) and never lie sideways (a^2+b^2 >= 1e-3,
    where _tilt_open_potential clamps the atan2 denominator): the two sets where the geometry itself is singular."""
    Om = 2 * math.pi / (T * DT)
    p0, u0 = ps[:n], us[:n]
    lean = torch.nn.functional.normalize(torch.cat([u0[:, :2], torch.zeros(n, 1)], -1), dim=-1)
    k_ax = torch.nn.functional.normalize(torch.cross(torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1), lean, dim=-1), dim=-1)
    base = (p0, u0, pr[:n], ur[:n], k_ax, lean)
    tot, clamped, pen = torch.zeros(n), 0, torch.zeros(n)
    minsep, minab = torch.full((n,), 9.0), torch.full((n,), 9.0)
    saved = ns["_progress"]
    if unclamped:
        ns["_progress"] = lambda rate, v: rate / v
    try:
        for i in range(T * sub):
            c = loop_state(i * DT / sub, base, (Om, 0.5, 0.02, 0.25, spin, (0.9, 2.1)), com_c)
            t = terms(ns, c)
            tot += (t["aim_lip"] + t["place_rcv"]) / sub
            pen += (t["src_speed"] + t["src_spin"]) / sub
            clamped += int(((t["aim_lip"].abs() > 0.999 * W_AIM) | (t["place_rcv"].abs() > 0.999 * W_PLACE)).sum())
            geo = ns["_pour_geometry"](c, R)
            minsep = torch.minimum(minsep, (c.rcv_cup_mouth_pos[:, :2] - c.src_cup_pos[:, :2]).norm(dim=-1))
            a = (c.src_cup_up[:, :2] * geo["d_hat"]).sum(-1)
            minab = torch.minimum(minab, a * a + c.src_cup_up[:, 2] ** 2)
    finally:
        ns["_progress"] = saved
    reg = (minsep > 0.01) & (minab > 1e-3)
    tr = tot[reg]
    tag = f"{name} com_c={com_c:+.4f} spin={spin:.1f} sub={sub}{' unclamped' if unclamped else ''}"
    print(f"[3] {tag}: all {n}: max {tot.max():+.3f} min {tot.min():+.3f} | regular {int(reg.sum())}: "
          f"max {tr.max():+.4f} mean {tr.mean():+.4f} min {tr.min():+.4f} frac>+0.05 {(tr > 0.05).double().mean():.4f}"
          f" | clamped steps {clamped / (T * sub * n):.4f} | control pen/loop {pen.mean():+.2f}")
    return {"all_max": float(tot.max()), "regular_n": int(reg.sum()), "regular_max": float(tr.max()),
            "regular_mean": float(tr.mean()), "regular_frac_gt_0.05": float((tr > 0.05).double().mean())}


def path_sum(ns, pts, ups, com_c=0.0):
    """Per-step sum of aim+place along a discrete path of (pos, up) samples; velocities by exact central
    difference of the analytic path, sampled at the same DT the env uses."""
    tot = pen = 0.0
    for i in range(len(pts)):
        pm, pp = pts[i - 1], pts[(i + 1) % len(pts)]
        um, up_ = ups[i - 1], ups[(i + 1) % len(ups)]
        v = (pp - pm) / (2 * DT)
        du = (up_ - um) / (2 * DT)
        w = torch.cross(ups[i], du, dim=-1)                            # minimal w with w x u = du
        c = make_ctx(pts[i], ups[i], (M - torch.tensor([0, 0, MZ])).unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]]),
                     v, w, torch.zeros(1, 3), torch.zeros(1, 3), com_c)
        t = terms(ns, c)
        tot += float(t["aim_lip"] + t["place_rcv"])
        pen += float(t["src_speed"] + t["src_spin"])
    return tot, pen


def tilted_up(theta, az):
    return torch.tensor([[math.sin(theta) * math.cos(az), math.sin(theta) * math.sin(az), math.cos(theta)]])


def probe_speed_asym(ns):
    """Pre-existing symmetric clamp: approach at 0.5 V_PROG, retreat at 3 V_PROG along the lean (same path)."""
    u = tilted_up(math.radians(60), 0.0)
    far = torch.tensor([[M[0] - 0.14, 0.0, M[2] + 0.06]])
    near = torch.tensor([[M[0] - 0.08, 0.0, M[2] + 0.06]])
    L = float((near - far).norm())
    n_in = int(round(L / (0.5 * V_PROG * DT)))
    n_out = int(round(L / (3.0 * V_PROG * DT)))
    pts = [far + (near - far) * (i / n_in) for i in range(n_in)] + [near + (far - near) * (i / n_out) for i in range(n_out)]
    return path_sum(ns, pts, [u] * len(pts)), len(pts)


def probe_coaxial(ns, miss=0.002, v_cross=0.05, rho=0.03):
    """Source origin crosses the receiver axis (miss distance `miss`) at a slow body speed, returns on a half
    circle. The spout jumps ~2 r cos(theta) while d_hat flips, so a symmetric +-1 clamp truncates one leg."""
    u = tilted_up(math.radians(45), 0.0)
    z = M[2] + 0.10
    n1 = int(round(2 * rho / (v_cross * DT)))
    leg = [torch.tensor([[M[0] - rho + 2 * rho * i / n1, M[1] + miss, z]]) for i in range(n1)]
    n2 = int(round(math.pi * rho / (0.5 * V_PROG * DT)))
    arc = [torch.tensor([[M[0] + rho * math.cos(math.pi * i / n2), M[1] + miss + rho * math.sin(math.pi * i / n2), z]])
           for i in range(n2)]
    fwd = path_sum(ns, leg + arc, [u] * (n1 + n2))
    rev = path_sum(ns, list(reversed(leg + arc)), [u] * (n1 + n2))
    return fwd, rev, n1 + n2


def probe_com(ns, com_c):
    """Tilt forward 80 deg at A (spout short of the mouth), slide 2.5 r toward the receiver, untilt, slide back.
    Slow everywhere. Exact rates net 0; the COM-velocity reading adds com_c * grad(d) . du."""
    az, th_max, n = 0.0, math.radians(80), 90
    pa = torch.tensor([[M[0] - 0.10, 0.0, M[2] + 0.05]])
    pb = pa + torch.tensor([[2.5 * R, 0.0, 0.0]])
    pts, ups = [], []
    for i in range(n):
        pts.append(pa); ups.append(tilted_up(th_max * i / n, az))
    for i in range(n):
        pts.append(pa + (pb - pa) * (i / n)); ups.append(tilted_up(th_max, az))
    for i in range(n):
        pts.append(pb); ups.append(tilted_up(th_max * (1 - i / n), az))
    for i in range(n):
        pts.append(pb + (pa - pb) * (i / n)); ups.append(tilted_up(0.0, az))
    f = path_sum(ns, pts, ups, com_c)
    r = path_sum(ns, list(reversed(pts)), list(reversed(ups)), com_c)
    return f, r, len(pts)


if __name__ == "__main__":
    print(f"candidate: {CAND}")
    r_new = rate_check(NEW, "iter_23")
    r_old = rate_check(OLD, "iter_22")
    lip = lip_rate_check()
    ax_new = axial_check(NEW, "iter_23")
    ax_old = axial_check(OLD, "iter_22")
    for nm, ns in (("iter_23", NEW), ("iter_22", OLD)):
        cycle_check(ns, nm, 0.0, 0.0)                         # env dt, as trained
        cycle_check(ns, nm, 0.0, 3.0)                         # + axial spin 3 rad/s amplitude
        cycle_check(ns, nm, 0.0, 3.0, sub=10, unclamped=True)  # exactness: fine dt, no clamp -> ~0 if exact
        cycle_check(ns, nm, COM_C, 0.0)                       # env velocity frame (COM velocity), 추정 offset
    for nm, ns in (("iter_23", NEW), ("iter_22", OLD)):
        (s, p), n = probe_speed_asym(ns)
        print(f"[4a] {nm}: slow-in / fast-out loop ({n} steps): aim+place {s:+.3f}, src_speed+spin {p:+.3f}")
        (f, pf), (b, pb), n = probe_coaxial(ns)
        print(f"[4b] {nm}: coaxial crossing loop ({n} steps): aim+place fwd {f:+.3f} rev {b:+.3f}, "
              f"control pen {pf:+.3f}/{pb:+.3f}")
        for cc in (0.0, COM_C):
            (f, _), (b, _), n = probe_com(ns, cc)
            print(f"[4c] {nm}: tilt-slide-untilt loop ({n} steps) com_c={cc:+.4f}: aim+place fwd {f:+.3f} rev {b:+.3f}")
