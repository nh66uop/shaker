"""Deterministic float64 check of the iter_27 tilt_rate (rotation potential rate), full compute_reward.

usage: python3 check_tilt_rate.py [compute_reward_<id>.py]     (default: compute_reward.py)

Phi = latches * spout * lean, lean = max(up . d, 0), spout = fine * height (iter_26 geometry),
d = horizontal unit vector source cup origin -> receiver mouth, lip = mouth + r (cos t d - sin t z),
t = atan2(up . d, up_z)  (env: pour_rules.pour_dir_update / pour_lip). Phi is re-implemented here
independently of the candidate and every trajectory is a rigid motion of both cups with exact
velocities, sampled at the policy step (sim.dt 1/120 x decimation 2 = 1/60 s, pour_fabric_env_cfg).
Checks
 D  rate vs exact derivative (central finite difference of Phi along the exact rigid motion, h 1e-6)
 A  closed cycles sum to <= 0: tilt at the mouth, spout in/out and up/down with a fixed lean, receiver
    moving, random both-cup Fourier cycles, fast jiggle at the mouth, jiggle around the concave spout
    peak with all pay gates 1 (A10: the pump of a one-sided difference), pay-gate toggles (side_ok,
    carry, keep) set from the rate sign, settled ramp (pump tests). A7n (info, not a pass criterion):
    a 0.1 s cycle with 10-30 Hz harmonics aliases at the 60 Hz reward Nyquist; any memoryless sampled
    rate does this (iter_26 too); the palm command is EMA 0.25 + slew 0.03/step filtered
 B  a tilt made while spout = 0 (lip 40 cm away) pays 0 and costs 0; a lean made far away and then
    carried in and out again nets <= 0
 C  a lean away from or sideways to the receiver earns nothing; holding still earns exactly 0
 S  sensitivity: ctx lin_vel is the COM velocity (IsaacLab root_lin_vel_w = root_com_lin_vel_w);
    COM 2 cm off the cup origin along the axis
"""
import dataclasses
import math
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "source" / "openarm"))
from openarm.agnostic.modules.t2r.loader import call_reward_fn, load_reward_fn  # noqa: E402
from openarm.agnostic.modules.t2r.validator import make_fake_context  # noqa: E402

MZ, RIM, CUP_R = 0.10, 0.041, 0.041
DT = 1.0 / 60.0
TILT_W, CAP = 5.0, 1.0
PREC_R, BELOW_R, DROP_R, FLOOR = 0.5, 1.0, 1.5, 0.4
RCV0 = torch.tensor([0.35, 0.08, 0.30])          # receiver origin at side margin 0.08
FN, _ = load_reward_fn(str(HERE / (sys.argv[1] if len(sys.argv) > 1 else "compute_reward.py")))
TPL = make_fake_context(1, seed=0)


def skew(a):
    z = torch.zeros_like(a[:, 0])
    return torch.stack([torch.stack([z, -a[:, 2], a[:, 1]], -1), torch.stack([a[:, 2], z, -a[:, 0]], -1),
                        torch.stack([-a[:, 1], a[:, 0], z], -1)], 1)


def rotm(axis, ang):
    k = skew(axis)
    eye = torch.eye(3).expand(axis.shape[0], 3, 3)
    return eye + torch.sin(ang)[:, None, None] * k + (1 - torch.cos(ang))[:, None, None] * (k @ k)


def geom(ps, rs, pr, rr):
    up, rup = rs[:, :, 2], rr[:, :, 2]
    sm, rm = ps + MZ * up, pr + MZ * rup
    h = rm[:, :2] - ps[:, :2]
    d = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    tw = (up[:, :2] * d).sum(-1)
    th = torch.atan2(tw, up[:, 2])
    lip = sm.clone()
    lip[:, :2] += RIM * torch.cos(th)[:, None] * d
    lip[:, 2] -= RIM * torch.sin(th)
    return up, rup, sm, rm, d, th, lip, tw


def phi_geo(ps, rs, pr, rr):
    """spout * lean, independent re-implementation"""
    _, _, _, rm, _, _, lip, tw = geom(ps, rs, pr, rr)
    dl = lip - rm
    fine = torch.exp(-(dl[:, :2].norm(dim=-1) / (PREC_R * CUP_R)) ** 2)
    over = torch.exp(-(torch.clamp(-dl[:, 2], min=0) / (BELOW_R * CUP_R)) ** 2)
    low = torch.exp(-(torch.clamp(dl[:, 2], min=0) / (DROP_R * CUP_R)) ** 2)
    return fine * over * (FLOOR + (1 - FLOOR) * low) * torch.clamp(tw, min=0)


def ctx_of(s, prog=None, side=None, carry=None, keep_mag=None, com=0.0):
    n = s["ps"].shape[0]
    up, rup, sm, rm, d, th, lip, _ = geom(s["ps"], s["rs"], s["pr"], s["rr"])
    kw = {}
    for f in dataclasses.fields(TPL):
        v = getattr(TPL, f.name)
        if isinstance(v, torch.Tensor):
            v = v.expand(n, *v.shape[1:]).clone()
            v = v if v.dtype == torch.bool else v.double()
        kw[f.name] = v
    one, zero = torch.ones(n), torch.zeros(n)
    carry = one if carry is None else carry
    lift = 0.25 * carry                                   # carry 0 -> cup on its spawn height
    vs = s["vs"] + torch.cross(s["ws"], com * up, dim=-1)  # COM velocity if COM is off the origin
    kw.update(src_cup_pos=s["ps"], src_cup_up=up, src_cup_mouth_pos=sm, src_cup_lin_vel=vs,
              src_cup_ang_vel=s["ws"], src_cup_tilt=torch.acos(up[:, 2].clamp(-1, 1)),
              rcv_cup_pos=s["pr"], rcv_cup_up=rup, rcv_cup_mouth_pos=rm, rcv_cup_lin_vel=s["vr"],
              rcv_cup_ang_vel=s["wr"], rcv_cup_tilt=torch.acos(rup[:, 2].clamp(-1, 1)),
              src_pour_lip_pos=lip, src_tilt_toward_rcv=th, pour_dir_xy=d,
              rcv_side_margin=s["pr"][:, 1] if side is None else side,
              src_cup_spawn_pos=s["ps"] - torch.stack([zero, zero, lift], -1),
              rcv_cup_spawn_pos=s["pr"] - torch.stack([zero, zero, lift], -1),
              src_grasped=carry > 0.5, rcv_grasped=carry > 0.5,
              cups_nested=(s["ps"] - s["pr"]).norm(dim=-1) < 0.09, premature_tilt=zero > 1,
              cup_hit=zero > 1, success=zero > 1, bead_in_source_frac=one, bead_in_target_frac=zero,
              bead_spill_frac=zero, d_in_target=zero, d_spill=zero, pour_target_frac=0.5 * one,
              bead_active_count=10 * one, episode_progress=one if prog is None else prog,
              src_grasp_ref_valid=one > 0, rcv_grasp_ref_valid=one > 0, adr_progress=0.0)
    for sd in ("src", "rcv"):                             # keep = 1 unless keep_mag lowers it
        kw[f"{sd}_palm_in_cup"] = kw[f"{sd}_ref_palm_in_cup"].clone()
        kw[f"{sd}_tips_in_cup"] = kw[f"{sd}_ref_tips_in_cup"].clone()
        kw[f"{sd}_force_share"] = kw[f"{sd}_ref_force_share"].clone()
        kw[f"{sd}_ref_force_total"] = 10.0 * one
        kw[f"{sd}_force_total"] = 10.0 * one if keep_mag is None else keep_mag
        kw[f"{sd}_finger_force"] = 10.0 * kw[f"{sd}_ref_force_share"][:, 1:]
    kw["cup_radius"], kw["cup_mouth_z"] = CUP_R, MZ
    return type(TPL)(**kw)


def tilt(s, **g):
    _, t = call_reward_fn(FN, ctx_of(s, **g))
    return t["tilt_rate"]


# ------------------------------------------------------------------ trajectories (rigid, exact velocities)
def rot_traj(a1, th1, dth1, a2, th2, dth2, r0):
    r1 = rotm(a1, th1)
    return r1 @ rotm(a2, th2) @ r0, dth1[:, None] * a1 + dth2[:, None] * (r1 @ a2[:, :, None])[:, :, 0]


def pour_pose(n, tilt_deg, dz=0.01, dxy=0.0, dy_axis=None):
    """source origin such that the lip sits dxy (toward d) and dz above the receiver rim at this tilt"""
    t = math.radians(tilt_deg)
    rm = RCV0 + torch.tensor([0, 0, MZ])
    # tilt toward +y (d ~ +y): up = (0, sin t, cos t); mouth = p + MZ up; lip = mouth + r(cos t d - sin t z)
    off = torch.tensor([0.0, MZ * math.sin(t) + RIM * math.cos(t), MZ * math.cos(t) - RIM * math.sin(t)])
    ps = rm - off + torch.tensor([0.0, dxy, dz])
    return ps.expand(n, 3).clone()


def run(traj, T, adapt=None, **g):
    """sum of tilt_rate over one closed trajectory sampled at k*DT, k = 0..T-1 (T*DT = period).
    adapt = (key, value_when_rate_up, value_when_rate_down): a pay-gate factor set from the sign of
    the rate at that step (the factors have no rate, so the sign does not depend on them)"""
    tot, pos, neg = 0.0, 0.0, 0.0
    for k in range(T):
        s = traj(k * DT)
        gk = {key: (v(k * DT, s) if callable(v) else v) for key, v in g.items()}
        if adapt is not None:
            key, hi, lo = adapt
            up_now = tilt(s, **gk) > 0
            gk[key] = torch.where(up_now, torch.full_like(up_now, hi, dtype=torch.float64),
                                  torch.full_like(up_now, lo, dtype=torch.float64))
        r = tilt(s, **gk)
        tot = tot + r
        pos = pos + torch.clamp(r, min=0)
        neg = neg + torch.clamp(r, max=0)
    return tot, pos, neg


def still(n):
    z = torch.zeros(n, 3)
    return dict(vs=z, ws=z, vr=z, wr=z, pr=RCV0.expand(n, 3).clone(), rr=torch.eye(3).expand(n, 3, 3).clone())


N = 64
torch.manual_seed(0)
ex = torch.tensor([1.0, 0, 0]).expand(N, 3).clone()      # rotation about -x tilts the cup toward +y
ez = torch.tensor([0, 0, 1.0]).expand(N, 3).clone()
ey = torch.tensor([0, 1.0, 0]).expand(N, 3).clone()
res = {}


def cyc_tilt(amp_deg, per, base_deg, dz, dxy):
    """rotate about the mouth-side axis: tilt base -> base+amp -> base, source origin fixed"""
    A = torch.tensor(amp_deg).double() * math.pi / 180
    w0 = 2 * math.pi / per
    ps = torch.cat([pour_pose(1, b + a / 2, z, x) for a, b, z, x in zip(amp_deg, base_deg, dz, dxy)])
    B = torch.tensor(base_deg).double() * math.pi / 180

    def f(t):
        th = B + A * math.sin(w0 * t / 2) ** 2
        dth = A * math.sin(w0 * t) * w0 / 2 * torch.ones_like(B)
        rs, ws = rot_traj(-ex[:len(B)], th, dth, ez[:len(B)], 0 * th, 0 * th, torch.eye(3).expand(len(B), 3, 3))
        s = still(len(B))
        s.update(ps=ps, rs=rs, ws=ws)
        return s
    return f


def cyc_translate(tilt_deg, per, amp, direction, dz0=0.01):
    """fixed lean, lip moved along `direction` by amp*(1-cos)/2 (in and out of the mouth)"""
    n = len(tilt_deg)
    t0 = torch.tensor(tilt_deg).double() * math.pi / 180
    rs = rotm(-ex[:n], t0)
    w0 = 2 * math.pi / per
    base = torch.cat([pour_pose(1, td, dz0, -a / 2 * dd[1]) for td, a, dd in zip(tilt_deg, amp, direction)])
    D = torch.tensor(direction).double()
    A = torch.tensor(amp).double()

    def f(t):
        s = still(n)
        s.update(ps=base + (A * (1 - math.cos(w0 * t)) / 2)[:, None] * D, rs=rs,
                 vs=(A * math.sin(w0 * t) * w0 / 2)[:, None] * D, ws=torch.zeros(n, 3))
        return s
    return f


def cyc_random(n, per, seed, jiggle=False):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    K = 3
    base_t = 30 + 70 * rnd(n)
    ps0 = torch.cat([pour_pose(1, float(b)) for b in base_t])
    sc = 0.004 if jiggle else 0.03
    ap, ar, aa = sc * (rnd(n, K, 3) - 0.5), sc * (rnd(n, K, 3) - 0.5), (0.1 if jiggle else 0.6) * (rnd(n, K, 3) - 0.5)
    ph = 2 * math.pi * rnd(n, K, 6)
    kk = torch.arange(1, K + 1).double()
    w0 = 2 * math.pi / per

    def four(amp, phase, t):          # closed: sum_k amp (cos(k w t + p) - cos p)
        c = torch.cos(kk[None, :, None] * w0 * t + phase) - torch.cos(phase)
        dc = -torch.sin(kk[None, :, None] * w0 * t + phase) * kk[None, :, None] * w0
        return (amp * c).sum(1), (amp * dc).sum(1)

    def f(t):
        dp, vp = four(ap, ph[:, :, :3], t)
        dr, vr = four(ar, ph[:, :, 3:], t)
        ang, dang = four(aa, ph[:, :, :3] + 1.0, t)
        a1 = torch.stack([-torch.ones(n), 0 * base_t, 0 * base_t], -1)
        th1 = base_t * math.pi / 180 + ang[:, 0]
        rs, ws = rot_traj(a1, th1, dang[:, 0], ey[:n], ang[:, 1], dang[:, 1], torch.eye(3).expand(n, 3, 3))
        rr, wr = rot_traj(ex[:n], 0.3 * ang[:, 2], 0.3 * dang[:, 2], ez[:n], 0 * th1, 0 * th1,
                          torch.eye(3).expand(n, 3, 3))
        return dict(ps=ps0 + dp, vs=vp, rs=rs, ws=ws, pr=RCV0 + dr, vr=vr, rr=rr, wr=wr)
    return f


# ------------------------------------------------------------------ D: derivative
f = cyc_random(N, 2.0, 1)
errs = []
for k in range(0, 120, 7):
    s = f(k * DT)
    r = tilt(s, prog=torch.ones(N), side=torch.full((N,), 0.2))
    up, rup, sm, rm, d, th, lip, tw = geom(s["ps"], s["rs"], s["pr"], s["rr"])
    tt = torch.clamp(d[:, 0] / 0.25, 0, 1)
    gate = (1 - tt * tt * (3 - 2 * tt)) * torch.clamp(rup[:, 2], 0, 1) \
        * ((s["ps"] - s["pr"]).norm(dim=-1) >= 0.09).double()        # dir_ok * up_rcv * not_nested
    h = 1e-6

    def shifted(sg):
        return dict(ps=s["ps"] + sg * h * s["vs"], rs=rotm(torch.nn.functional.normalize(s["ws"] + 1e-30, dim=-1),
                    sg * h * s["ws"].norm(dim=-1)) @ s["rs"], pr=s["pr"] + sg * h * s["vr"],
                    rr=rotm(torch.nn.functional.normalize(s["wr"] + 1e-30, dim=-1), sg * h * s["wr"].norm(dim=-1)) @ s["rr"])
    exact = TILT_W * (phi_geo(**shifted(1)) - phi_geo(**shifted(-1))) / (2 * h)
    exact = torch.where(exact > 0, gate * exact, exact)
    m = exact < 0.9 * CAP                                  # uncapped regime
    errs.append(((r - exact)[m]).abs() / exact[m].abs().clamp(min=0.05))
e = torch.cat(errs)
res["D_rel_err_p50_p99_max"] = [float(e.quantile(0.5)), float(e.quantile(0.99)), float(e.max())]
print(f"D  rate vs exact dPhi/dt: rel err p50 {e.quantile(0.5):.2e}  p99 {e.quantile(0.99):.2e}  max {e.max():.2e}")

# ------------------------------------------------------------------ A: closed cycles
worst = {}
cases = {"A1 tilt 0->92->0 at the mouth, 1-8 s": [1, 2, 4, 8],
         "A2 tilt 40->80->40 at the mouth, 0.5-4 s": [0.5, 1, 2, 4]}
for name, pers in cases.items():
    w = -1e9
    for per in pers:
        T = int(round(per / DT))
        fn = cyc_tilt([92.0 if "92" in name else 40.0] * 4, per, [0.0 if "92" in name else 40.0] * 4,
                      [0.0, 0.01, 0.03, 0.06], [0.0, 0.005, -0.01, 0.02])
        tot, pos, neg = run(fn, T)
        w = max(w, float(tot.max()))
    worst[name] = w
for name, dirs in {"A3 lean 60 fixed, lip in/out along d (8 cm), 0.5-4 s": [(0, 1.0, 0)],
                   "A4 lean 60 fixed, lip in/out sideways (6 cm), 0.5-4 s": [(1.0, 0, 0)],
                   "A5 lean 60 fixed, lip up/down (10 cm), 0.5-4 s": [(0, 0, 1.0)]}.items():
    w = -1e9
    for per in [0.5, 1, 2, 4]:
        amp = 0.08 if "along" in name else (0.06 if "side" in name else 0.10)
        fn = cyc_translate([60.0, 75.0, 90.0], per, [amp] * 3, dirs * 3)
        tot, _, _ = run(fn, int(round(per / DT)))
        w = max(w, float(tot.max()))
    worst[name] = w
for name, jig, pers in [("A6 random both-cup Fourier cycles (64 x 3 seeds), 1-4 s", False, [1, 2, 4]),
                        ("A7 fast jiggle at the mouth (4 mm, 0.1 rad), 0.2-0.5 s", True, [0.2, 0.3, 0.5]),
                        ("A7n reward-Nyquist aliasing: jiggle with 0.1 s period, 10-30 Hz harmonics (info)", True, [0.1])]:
    w = -1e9
    for per in pers:
        for seed in range(3):
            tot, _, _ = run(cyc_random(N, per, 10 * seed + int(per * 10), jiggle=jig), int(round(per / DT)))
            w = max(w, float(tot.max()))
    worst[name] = w

# jiggle of the lip around the spout peak (mouth centre, rim height = concave maximum), all pay gates 1:
# the pump a one-sided (backward) difference makes, since its error -(tau/2) v'Hv is > 0 where H < 0
w = -1e9
for tdeg in (60.0, 90.0):
    for dirn in ((0, 1.0, 0), (1.0, 0, 0), (0, 0, 1.0)):
        for amp in (0.002, 0.005, 0.01):
            for fq in (1.0, 2.0, 3.0, 5.0):
                D = torch.tensor(dirn).double()
                base, rs = pour_pose(1, tdeg, dz=0.0), rotm(-ex[:1], torch.full((1,), math.radians(tdeg)))

                def jig(t, D=D, base=base, rs=rs, amp=amp, fq=fq):
                    s = still(1)
                    s.update(ps=base + amp * math.sin(2 * math.pi * fq * t) * D, rs=rs, ws=torch.zeros(1, 3),
                             vs=(amp * 2 * math.pi * fq * math.cos(2 * math.pi * fq * t) * D)[None, :].clone())
                    return s
                tot, _, _ = run(jig, int(round(60 / fq)), side=torch.full((1,), 0.2))
                w = max(w, float(tot.max()))
worst["A10 jiggle 2-10 mm, 1-5 Hz around the spout peak, all pay gates 1"] = w
# pump: pay-gate factors toggled from the rate sign (open while Phi rises, closed while it falls, and back)
w = -1e9
for per in [0.5, 1, 2, 4]:
    for fn in (cyc_tilt([92.0] * 4, per, [0.0] * 4, [0.0, 0.01, 0.03, 0.06], [0.0, 0.005, -0.01, 0.02]),
               cyc_translate([60.0, 75.0, 90.0], per, [0.08] * 3, [(0, 1.0, 0)] * 3),
               cyc_random(N, per, 99)):
        for ad in (("side", 0.2, 0.0), ("side", 0.0, 0.2), ("keep_mag", 10.0, 1.0),
                   ("carry", 1.0, 0.0), ("carry", 0.0, 1.0)):
            tot, _, _ = run(fn, int(round(per / DT)), adapt=ad)
            w = max(w, float(tot.max()))
worst["A8 pay-gate toggles (side_ok, keep, carry) in phase with the rate sign"] = w
# settled ramp: cycle starts at episode progress 0, 0.02, 0.2 (settled latch grows during the cycle)
w = -1e9
for p0 in (0.0, 0.02, 0.2):
    fn = cyc_tilt([92.0] * 4, 1.0, [0.0] * 4, [0.01] * 4, [0.0] * 4)
    tot, _, _ = run(fn, 60, prog=lambda t, s, p0=p0: torch.full((4,), p0 + t / 15.0))
    w = max(w, float(tot.max()))
worst["A9 settled ramps during the cycle (start 0 / 0.02 / 0.2)"] = w
for k, v in worst.items():
    print(f"{k:72s} max sum {v:+.6f}")
res["A_closed_cycle_max_sum"] = worst

# ------------------------------------------------------------------ B: spout = 0
far = torch.tensor([0.0, -0.40, 0.0])
fn = cyc_tilt([120.0] * 4, 2.0, [0.0] * 4, [0.01] * 4, [0.0] * 4)
mx = 0.0
for k in range(120):
    s = fn(k * DT)
    s["ps"] = s["ps"] + far
    mx = max(mx, float(tilt(s).abs().max()))
res["B_far_tilt_abs_max_per_step"] = mx
print(f"B  tilt 0->120->0 with the lip 40 cm away: |tilt_rate| max per step {mx:.3e}")
# lean 70 made far away, carried in (lip over the mouth), then out again, lean kept
n = 4
rs = rotm(-ex[:n], torch.full((n,), math.radians(70)))
p_in = pour_pose(n, 70.0)
tot = torch.zeros(n)
for per in (1.0, 3.0):
    T = int(round(per / DT))
    for k in range(T):
        t = k * DT
        a = (1 - math.cos(2 * math.pi * t / per)) / 2           # 0 far -> 1 at the mouth -> 0 far
        da = math.sin(2 * math.pi * t / per) * math.pi / per
        s = still(n)
        s.update(ps=p_in + (1 - a) * far, vs=(-da * far).expand(n, 3).clone(), rs=rs, ws=torch.zeros(n, 3))
        tot = tot + tilt(s)
res["B_lean_far_carry_in_out_sum_max"] = float(tot.max())
print(f"B  lean 70 made far away, carried in and out: sum max {tot.max():+.6f}")

# ------------------------------------------------------------------ C: away / sideways / still
n = 3
mx_c = 0.0
for axis, lab in ((ex[:n], "away"), (ey[:n], "sideways")):
    for k in range(60):
        t = k * DT
        th = torch.full((n,), math.radians(92) * math.sin(math.pi * t / 1.0) ** 2)
        dth = torch.full((n,), math.radians(92) * math.sin(2 * math.pi * t / 1.0) * math.pi / 1.0)
        rs, ws = rot_traj(axis, th, dth, ez[:n], 0 * th, 0 * th, torch.eye(3).expand(n, 3, 3))
        s = still(n)
        s.update(ps=pour_pose(n, 0.0), rs=rs, ws=ws)
        mx_c = max(mx_c, float(tilt(s).abs().max()))
res["C_away_sideways_abs_max"] = mx_c
s = still(N)
s.update(ps=pour_pose(N, 80.0), rs=rotm(-ex, torch.full((N,), math.radians(80))))
res["C_hold_still_abs_max"] = float(tilt(s).abs().max())
print(f"C  lean away / sideways at the mouth: |tilt_rate| max {mx_c:.3e};  holding 80 deg over the mouth: "
      f"{res['C_hold_still_abs_max']:.3e}")

# ------------------------------------------------------------------ S: COM offset sensitivity
w = -1e9
for com in (-0.02, 0.02):
    for per in (1, 2, 4):
        fn = cyc_tilt([92.0] * 4, per, [0.0] * 4, [0.0, 0.01, 0.03, 0.06], [0.0, 0.005, -0.01, 0.02])
        tot, _, _ = run(fn, int(round(per / DT)), com=com)
        w = max(w, float(tot.max()))
res["S_com_offset_2cm_max_sum"] = w
print(f"S  COM 2 cm off the origin (lin_vel = COM velocity), tilt cycles: max sum {w:+.6f}")
bad = [k for k, v in res["A_closed_cycle_max_sum"].items() if v > 1e-9 and "(info)" not in k]
print("RESULT", "PASS" if not bad and res["B_far_tilt_abs_max_per_step"] < 1e-9 and mx_c < 1e-9
      and res["C_hold_still_abs_max"] == 0.0 else f"FAIL {bad}")
