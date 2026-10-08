"""Deterministic float64 check of iter_24 change 1-3 (step-difference rates, finger-count grasp, W_TILT_OPEN 0.5).

Runs the WHOLE compute_reward of a candidate on synthetic rigid-body states and compares with iter_23.
  0. everything except the changed terms is bit-identical to iter_23 (potentials d_aim / d_place / PHI included);
     grasp equals the notes formula 0.4 + 0.2 cq + 0.2 keep + 0.2 n_on/4; ladder held >= 0.4 > free <= 0.39
  1. aim_lip / place_rcv / tilt_open = the notes formula on the iter_23 potentials at t and t-1 (exact), and
     against the iter_23 exact-derivative payment with TRUE (link-origin) velocities (O(dt) agreement)
  2. standing still with arbitrary velocity readings pays exactly 0 (iter_23 paid / charged there)
  3. closed cycles at env dt: aim+place telescopes to 0 without the clamp; tilt_open <= 0 on every cycle
  4. cache semantics: first call, N / dtype change, episode_progress not increasing -> delta 0
  5. residual probes: symmetric-clamp speed asymmetry, coaxial crossing, COM-velocity frame (now irrelevant),
     first_out flicker (a bead pokes out at the top of a tilt and falls back), grasp-flag flicker gating
Usage: python3 check_step_diff.py [candidate.py]   (default compute_reward_c1.py)
"""
import math
import sys
import types
import torch

torch.set_default_dtype(torch.float64)
BASE = "/home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/"
CAND = sys.argv[1] if len(sys.argv) > 1 else BASE + "iter_24/compute_reward_c1.py"
R, MZ, BZ, TABLE_Z = 0.02808, 0.053885, -0.0599, 0.205      # RH56F1 shaker x0.65 (env cfg, iter_23 check)
COM_C = -0.0324   # COM along the cup axis from the link origin (iter_23 estimate, 추정)
DT, W_AIM, W_PLACE, W_OPEN_NEW, V_PROG = 1.0 / 60.0, 1.5, 1.0, 0.5, 0.15
M = torch.tensor([0.40, 0.0, TABLE_Z - BZ + MZ + 0.15])     # receiver mouth at its placement height target
CHANGED = {"place_rcv", "aim_lip", "tilt_open", "grasp_src", "grasp_rcv", "diag_theta_rate", "diag_open_rate",
           "diag_state_income"}


def load(path):
    ns = {"RewardContext": object}
    exec(open(path).read(), ns)
    return ns


NEW, OLD = load(CAND), load(BASE + "iter_23/compute_reward.py")


def reset_cache(ns):
    if "_STEP_CACHE" in ns:
        ns["_STEP_CACHE"].clear()


def rot(u, w, s):
    th = w.norm(dim=-1, keepdim=True) * s
    k = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return u * torch.cos(th) + torch.cross(k, u, dim=-1) * torch.sin(th) + k * (k * u).sum(-1, keepdim=True) * (1 - torch.cos(th))


def make_ctx(ps, us, pr, ur, vs, ws, vr, wr, com_c=0.0, prog=0.5, first_out=1.0, fingers=None, grasped=None):
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
        setattr(c, f"{side}_finger_force", (2.0 * torch.ones(n, 5)) if fingers is None else fingers[side])
        setattr(c, f"{side}_palm_force", z0.clone())
        setattr(c, f"{side}_hand_closure", 0.8 * o)
        setattr(c, f"{side}_grasped", torch.ones(n, dtype=torch.bool) if grasped is None else grasped[side])
        setattr(c, f"{side}_arm_qd", torch.zeros(n, 7))
        setattr(c, f"{side}_arm_qdd", torch.zeros(n, 7))
        setattr(c, f"{side}_arm_limit_margin", 0.5 * torch.ones(n, 7))
        setattr(c, f"{side}_hand_foreign_force", z0.clone())
    fill = 1.0 - (1.0 - first_out) / 20.0                     # first_out 0 <=> one bead out of the source
    c.bead_in_source_frac, c.bead_in_target_frac, c.bead_spill_frac = fill * o, z0.clone(), z0.clone()
    c.d_in_target, c.d_spill, c.cup_cup_force = z0.clone(), z0.clone(), z0.clone()
    c.episode_progress = (prog * o) if not torch.is_tensor(prog) else prog
    c.cups_nested, c.success = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    c.actions, c.prev_actions = torch.zeros(n, 24), torch.zeros(n, 24)
    return c


def terms(ns, c):
    _, comps = ns["compute_reward"](c)
    return comps


def q(x, p):
    return float(x.quantile(p)) if x.numel() else float("nan")


# ------------------------------------------------------------------ states near the pour window (iter_23 style)
torch.manual_seed(0)
N = 2048
pr = (M - torch.tensor([0, 0, MZ])).repeat(N, 1) + 0.02 * torch.randn(N, 3)
ur = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0]) + 0.05 * torch.randn(N, 3), dim=-1)
ang = torch.rand(N) * 1.6
az = math.radians(68) + 0.6 * torch.randn(N)
us = torch.stack([torch.sin(ang) * torch.cos(az), torch.sin(ang) * torch.sin(az), torch.cos(ang)], -1)
ps = pr + MZ * ur - torch.stack([0.05 * torch.cos(az), 0.05 * torch.sin(az), torch.zeros(N)], -1) \
    + torch.tensor([0, 0, 0.04]) + 0.03 * torch.randn(N, 3)


# ------------------------------------------------------------------ 0. unchanged terms, potentials, grasp formula
def check_identity():
    g = torch.Generator().manual_seed(1)
    f = {s: torch.rand(N, 5, generator=g) * 3.0 for s in ("src", "rcv")}           # 0..3 N: n_on varies 0..4
    gr = {s: torch.rand(N, generator=g) > 0.3 for s in ("src", "rcv")}
    c = make_ctx(ps, us, pr, ur, 0.05 * torch.randn(N, 3), 0.5 * torch.randn(N, 3), 0.02 * torch.randn(N, 3),
                 0.2 * torch.randn(N, 3), fingers=f, grasped=gr)
    c.actions, c.prev_actions = 2 * torch.rand(N, 24, generator=g) - 1, 2 * torch.rand(N, 24, generator=g) - 1
    c.bead_in_source_frac = torch.rand(N, generator=g)
    c.bead_in_target_frac = (1 - c.bead_in_source_frac) * torch.rand(N, generator=g)
    c.bead_spill_frac = 1 - c.bead_in_source_frac - c.bead_in_target_frac
    c.d_in_target, c.d_spill = 0.1 * torch.rand(N, generator=g), 0.1 * torch.rand(N, generator=g)
    reset_cache(NEW)
    tn, to = terms(NEW, c), terms(OLD, c)
    only_old = sorted(set(to) - set(tn))
    only_new = sorted(set(tn) - set(to))
    worst = max(((k, float((tn[k] - to[k]).abs().max())) for k in to if k in tn and k not in CHANGED),
                key=lambda kv: kv[1])
    zero_first = max(float(tn[k].abs().max()) for k in ("aim_lip", "place_rcv", "tilt_open"))
    # grasp: independent formula (iter_23 geometry, notes change 2)
    grasp_err, ladder = 0.0, []
    for side, pose in (("src", None), ("rcv", None)):
        geo = OLD["_grip_geometry"](getattr(c, f"{side}_tips_pos"), getattr(c, f"{side}_cup_pos"),
                                    getattr(c, f"{side}_cup_up"), MZ, 0.027)
        ff = getattr(c, f"{side}_finger_force")
        f_t, f_o = ff[:, 0], ff[:, 1:].sum(1)
        pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
        cq = (0.4 + 0.6 * pinch) * (0.1 + 0.9 * geo["thumb_band"])
        near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
        keep = near_grip * (0.3 + 0.7 * torch.minimum(f_t, f_o) / (torch.maximum(f_t, f_o) + 1e-6))
        n_on = (ff[:, 1:] > 1.0).double().sum(1)
        held_new = 0.4 + 0.2 * cq + 0.2 * keep + 0.2 * n_on / 4.0
        held_old = 0.4 + 0.3 * cq + 0.3 * keep
        gmask = getattr(c, f"{side}_grasped")
        # grasp_new - grasp_old = 1.5 * pose_ok * (held_new - held_old) on grasped envs, 0 elsewhere
        diff = tn[f"grasp_{side}"] - to[f"grasp_{side}"]
        pose_ok = torch.where(gmask, to[f"grasp_{side}"] / (1.5 * held_old), torch.zeros(N))
        expect = torch.where(gmask, 1.5 * pose_ok * (held_new - held_old), torch.zeros(N))
        grasp_err = max(grasp_err, float((diff - expect).abs().max()))
        ladder.append((float(held_new.min()), float(held_new.max())))
    si = float(((tn["diag_state_income"] - to["diag_state_income"])
                - (tn["grasp_src"] - to["grasp_src"] + tn["grasp_rcv"] - to["grasp_rcv"])).abs().max())
    print(f"[0] keys only in iter_23: {only_old}  only in candidate: {only_new}")
    print(f"[0] max |cand - iter_23| over unchanged terms and diagnostics: {worst[1]:.1e} ({worst[0]})")
    print(f"[0] first call (empty cache): max |aim|,|place|,|tilt_open| = {zero_first:.1e}")
    print(f"[0] grasp vs notes formula: max err {grasp_err:.1e}; held range src {ladder[0]}, rcv {ladder[1]}; "
          f"state-income diag err {si:.1e}")
    # ladder: grasped value >= 0.4 always above free <= 0.39 (pose_ok = 1 here: carried or upright)
    return {"only_old": only_old, "only_new": only_new, "unchanged_max_abs_diff": worst[1],
            "first_call_max": zero_first, "grasp_formula_err": grasp_err, "held_range": ladder}


# ------------------------------------------------------------------ trajectory helpers
def loop_state(t, base, prm, com_c=0.0, prog=0.5, first_out=1.0):
    """Closed smooth loop (iter_23 check): tilt about a horizontal axis, translation along the lean, orbit about the
    receiver mouth, axial spin. Exact analytic LINK-ORIGIN velocities (what iter_23's derivative expects)."""
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
    w = k_ax * da + S * math.cos(Om * t) * u
    n = p.shape[0]
    return make_ctx(p, u, rcv_p, rcv_u, v, w, torch.zeros(n, 3), torch.zeros(n, 3), com_c, prog, first_out)


def loop_base(n):
    p0, u0 = ps[:n], us[:n]
    lean = torch.nn.functional.normalize(torch.cat([u0[:, :2], torch.zeros(n, 1)], -1), dim=-1)
    k_ax = torch.nn.functional.normalize(torch.cross(torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1), lean, dim=-1), dim=-1)
    return (p0, u0, pr[:n], ur[:n], k_ax, lean)


def ref_pay(t_prev, t_now, c_now):
    """Notes formula on the iter_23 potentials (independent of the candidate)."""
    stage = t_now["diag_stage_cd"]
    dp = (t_now["diag_place_dist_r"] - t_prev["diag_place_dist_r"]) * R
    da = (t_now["diag_aim_dist_r"] - t_prev["diag_aim_dist_r"]) * R
    dphi = t_now["diag_open_phi"] - t_prev["diag_open_phi"]
    place = W_PLACE * stage * torch.clamp(-dp / (V_PROG * DT), -1.0, 1.0)
    aim = W_AIM * stage * torch.clamp(-da / (V_PROG * DT), -1.0, 1.0)
    sep = (c_now.rcv_cup_mouth_pos[:, :2] - c_now.src_cup_pos[:, :2]).norm(dim=-1)
    sep_ok = torch.clamp(sep / 0.03, max=1.0)
    carry = t_now["diag_stage_cd"]                          # has_beads = 1 here, so stage_cd = carry_both
    rn = dphi / DT
    opn = W_OPEN_NEW * t_now["diag_first_out"] * (carry * sep_ok * torch.clamp(rn, 0.0, 1.0) + torch.clamp(rn, -100.0, 0.0))
    return {"place_rcv": place, "aim_lip": aim, "tilt_open": opn}


# ------------------------------------------------------------------ 1. notes formula exactness + vs exact derivative
def check_formula(n=1024, T=240):
    Om = 2 * math.pi / (T * DT)
    base = loop_base(n)
    reset_cache(NEW)
    err = {k: 0.0 for k in ("place_rcv", "aim_lip", "tilt_open")}
    vs_exact = {k: [] for k in ("place_rcv", "aim_lip", "tilt_open")}
    prev_old = None
    prev_c = None
    for i in range(T + 1):
        c = loop_state(i * DT, base, (Om, 0.5, 0.02, 0.25, 3.0, (0.9, 2.1)), prog=(i + 1) / 900.0)
        tn, to = terms(NEW, c), terms(OLD, c)
        if prev_old is not None:
            ref = ref_pay(prev_old, to, c)
            for k in err:
                err[k] = max(err[k], float((tn[k] - ref[k]).abs().max()))
            # iter_23 = exact derivative with TRUE readings; trapezoid of t-1 and t is the matching O(dt^2) reading
            ok = (to["diag_stage_cd"] > 0.999)
            for k, wold in (("place_rcv", 1.0), ("aim_lip", 1.0), ("tilt_open", W_OPEN_NEW / 0.1)):
                trap = 0.5 * (to[k] + prev_old[k]) * wold
                vs_exact[k].append((tn[k] - trap)[ok].abs())
        prev_old, prev_c = to, c
    out = {}
    print("[1] candidate vs notes formula on iter_23 potentials (max abs, reward/step):",
          ", ".join(f"{k} {v:.1e}" for k, v in err.items()))
    for k, lst in vs_exact.items():
        e = torch.cat(lst)
        out[k] = (q(e, 0.5), q(e, 0.99))
        print(f"[1] {k:9s} step difference vs iter_23 exact derivative (true velocities, trapezoid): "
              f"median {out[k][0]:.1e}  p99 {out[k][1]:.1e}")
    return {"formula_max_err": err, "vs_exact_derivative": out}


# ------------------------------------------------------------------ 2. standing still with arbitrary readings
def check_still(steps=60):
    g = torch.Generator().manual_seed(2)
    reset_cache(NEW)
    tot_new = torch.zeros(N)
    tot_old = torch.zeros(N)
    for i in range(steps):
        vs = 0.3 * (2 * torch.rand(N, 3, generator=g) - 1)
        ws = 3.0 * (2 * torch.rand(N, 3, generator=g) - 1)
        vr = 0.1 * (2 * torch.rand(N, 3, generator=g) - 1)
        wr = 1.0 * (2 * torch.rand(N, 3, generator=g) - 1)
        c = make_ctx(ps, us, pr, ur, vs, ws, vr, wr, prog=(i + 1) / 900.0)
        tn, to = terms(NEW, c), terms(OLD, c)
        tot_new += tn["aim_lip"] + tn["place_rcv"] + tn["tilt_open"]
        tot_old += to["aim_lip"] + to["place_rcv"] + to["tilt_open"]
    # a realistic parked reading: the i23 trace's held-cup omega bias (|w x u| ~0.4-0.5 rad/s), no motion
    reset_cache(NEW)
    w_b = torch.nn.functional.normalize(torch.randn(N, 3, generator=g), dim=-1) * 0.45
    tn_b = torch.zeros(N)
    to_b = torch.zeros(N)
    for i in range(steps):
        c = make_ctx(ps, us, pr, ur, torch.zeros(N, 3), w_b, torch.zeros(N, 3), torch.zeros(N, 3), prog=(i + 1) / 900.0)
        tn, to = terms(NEW, c), terms(OLD, c)
        tn_b += tn["aim_lip"] + tn["place_rcv"] + tn["tilt_open"]
        to_b += to["aim_lip"] + to["place_rcv"] + to["tilt_open"]
    print(f"[2] fixed pose, random readings (|v|<=0.3 m/s, |w|<=3 rad/s), {steps} steps: candidate max|sum| "
          f"{float(tot_new.abs().max()):.1e}; iter_23 mean {float(tot_old.mean()):+.2f}, |sum| p99 {q(tot_old.abs(), 0.99):.2f}")
    print(f"[2] fixed pose, 0.45 rad/s omega bias: candidate max|sum| {float(tn_b.abs().max()):.1e}; iter_23 "
          f"mean {float(to_b.mean()) / steps:+.4f}/step (x900 = {float(to_b.mean()) / steps * 900:+.1f}/episode)")
    return {"still_random_max_abs": float(tot_new.abs().max()), "iter23_random_mean": float(tot_old.mean()),
            "still_bias_max_abs": float(tn_b.abs().max()), "iter23_bias_per_step": float(to_b.mean()) / steps}


# ------------------------------------------------------------------ 3. closed cycles
def check_cycles(T=240, n=512, spin=3.0, amp=(0.5, 0.02, 0.25), unclamped=False):
    Om = 2 * math.pi / (T * DT)
    base = loop_base(n)
    reset_cache(NEW)
    saved = NEW["_progress"]
    if unclamped:
        NEW["_progress"] = lambda rate, v: rate / v
    ap, op, ap_old = torch.zeros(n), torch.zeros(n), torch.zeros(n)
    minsep, minab = torch.full((n,), 9.0), torch.full((n,), 9.0)
    try:
        for i in range(T + 1):                 # T+1 samples: state T == state 0, T differences close the loop
            c = loop_state(i * DT, base, (Om, amp[0], amp[1], amp[2], spin, (0.9, 2.1)), prog=(i + 1) / 900.0)
            t = terms(NEW, c)
            ap += t["aim_lip"] + t["place_rcv"]
            op += t["tilt_open"]
            if i > 0:                          # iter_23 (exact rates, true velocities) on the same T states
                to = terms(OLD, c)
                ap_old += to["aim_lip"] + to["place_rcv"]
            geo = OLD["_pour_geometry"](c, R)
            minsep = torch.minimum(minsep, (c.rcv_cup_mouth_pos[:, :2] - c.src_cup_pos[:, :2]).norm(dim=-1))
            a = (c.src_cup_up[:, :2] * geo["d_hat"]).sum(-1)
            minab = torch.minimum(minab, a * a + c.src_cup_up[:, 2] ** 2)
    finally:
        NEW["_progress"] = saved
    reg = (minsep > 0.01) & (minab > 1e-3)     # iter_23 'regular' loops: d_hat defined, not lying sideways
    tag = f"T={T} amp={amp} spin={spin}{' unclamped' if unclamped else ''}"
    print(f"[3] {tag}: aim+place sum max {float(ap.max()):+.2e} min {float(ap.min()):+.2e} | regular {int(reg.sum())}: "
          f"cand max {float(ap[reg].max()):+.3f} mean {float(ap[reg].mean()):+.4f}, iter_23 max {float(ap_old[reg].max()):+.3f} "
          f"mean {float(ap_old[reg].mean()):+.4f} | tilt_open sum max {float(op.max()):+.2e} min {float(op.min()):+.2e}")
    return {"aim_place_max": float(ap.max()), "aim_place_min": float(ap.min()), "tilt_open_max": float(op.max()),
            "regular_n": int(reg.sum()), "regular_cand_max": float(ap[reg].max()),
            "regular_cand_mean": float(ap[reg].mean()), "regular_i23_max": float(ap_old[reg].max()),
            "regular_i23_mean": float(ap_old[reg].mean())}


# ------------------------------------------------------------------ 4. cache semantics
def check_cache():
    base = loop_base(256)
    Om = 2 * math.pi / (240 * DT)
    prm = (Om, 0.5, 0.02, 0.25, 0.0, (0.9, 2.1))
    res = {}
    reset_cache(NEW)
    t0 = terms(NEW, loop_state(0.0, base, prm, prog=0.1))
    res["first_call"] = max(float(t0[k].abs().max()) for k in ("aim_lip", "place_rcv", "tilt_open"))
    t1 = terms(NEW, loop_state(DT * 20, base, prm, prog=0.2))
    moving = float(t1["aim_lip"].abs().max())
    # N change (validator dry run) -> 0
    small = loop_base(64)
    tN = terms(NEW, loop_state(DT * 40, small, prm, prog=0.3))
    res["n_change"] = max(float(tN[k].abs().max()) for k in ("aim_lip", "place_rcv", "tilt_open"))
    # progress not increasing in half the envs (new episode) -> those 0, the others equal the reference
    reset_cache(NEW)
    terms(NEW, loop_state(0.0, base, prm, prog=0.5))
    old_prev = terms(OLD, loop_state(0.0, base, prm, prog=0.5))
    prog = torch.full((256,), 0.6)
    prog[:128] = 1.0 / 900.0
    c = loop_state(DT * 10, base, prm, prog=prog)
    tn, to = terms(NEW, c), terms(OLD, c)
    ref = ref_pay(old_prev, to, c)
    res["new_episode"] = max(float(tn[k][:128].abs().max()) for k in ("aim_lip", "place_rcv", "tilt_open"))
    res["same_episode_err"] = max(float((tn[k][128:] - ref[k][128:]).abs().max()) for k in ("aim_lip", "place_rcv", "tilt_open"))
    # repeated call without time passing -> 0, and the next step is unaffected
    reset_cache(NEW)
    terms(NEW, loop_state(0.0, base, prm, prog=0.5))
    cA = loop_state(DT * 10, base, prm, prog=0.6)
    tA1 = terms(NEW, cA)
    tA2 = terms(NEW, cA)
    res["repeat_call"] = max(float(tA2[k].abs().max()) for k in ("aim_lip", "place_rcv", "tilt_open"))
    cB = loop_state(DT * 20, base, prm, prog=0.7)
    tB = terms(NEW, cB)
    reset_cache(NEW)
    terms(NEW, cA)
    tB_ref = terms(NEW, cB)
    res["after_repeat_err"] = max(float((tB[k] - tB_ref[k]).abs().max()) for k in ("aim_lip", "place_rcv", "tilt_open"))
    # dtype change -> 0
    reset_cache(NEW)
    terms(NEW, loop_state(0.0, base, prm, prog=0.5))
    c32 = loop_state(DT * 10, base, prm, prog=0.6)
    for k, v in vars(c32).items():
        if torch.is_tensor(v) and v.is_floating_point():
            setattr(c32, k, v.float())
    t32 = terms(NEW, c32)
    res["dtype_change"] = max(float(t32[k].abs().max()) for k in ("aim_lip", "place_rcv", "tilt_open"))
    print(f"[4] cache: first call {res['first_call']:.1e} | N change {res['n_change']:.1e} | new-episode envs "
          f"{res['new_episode']:.1e}, other envs vs reference {res['same_episode_err']:.1e} | repeated call "
          f"{res['repeat_call']:.1e}, next step vs no-repeat {res['after_repeat_err']:.1e} | dtype change "
          f"{res['dtype_change']:.1e} | (moving control: |aim| max {moving:.2f})")
    return res


# ------------------------------------------------------------------ 5. residual probes
def path_sum(ns, pts, ups, com_c=0.0, first_out=None, keys=("aim_lip", "place_rcv")):
    reset_cache(ns)
    tot = pen = 0.0
    L = len(pts)
    for i in range(L + 1):                       # last sample == first: the loop closes
        j = i % L
        pm, pp = pts[j - 1], pts[(j + 1) % L]
        um, up_ = ups[j - 1], ups[(j + 1) % L]
        v = (pp - pm) / (2 * DT)
        w = torch.cross(ups[j], (up_ - um) / (2 * DT), dim=-1)
        fo = 1.0 if first_out is None else first_out[j]
        c = make_ctx(pts[j], ups[j], (M - torch.tensor([0, 0, MZ])).unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]]),
                     v, w, torch.zeros(1, 3), torch.zeros(1, 3), com_c, prog=(i + 1) / 2000.0, first_out=fo)
        t = terms(ns, c)
        if i == 0:
            continue                              # sample 0 primes the cache; samples 1..L visit each state once
        tot += float(sum(t[k] for k in keys))
        pen += float(t["src_speed"] + t["src_spin"])
    return tot, pen


def tilted_up(theta, az):
    return torch.tensor([[math.sin(theta) * math.cos(az), math.sin(theta) * math.sin(az), math.cos(theta)]])


def probe_speed_asym(ns):
    u = tilted_up(math.radians(60), 0.0)
    far = torch.tensor([[M[0] - 0.14, 0.0, M[2] + 0.06]])
    near = torch.tensor([[M[0] - 0.08, 0.0, M[2] + 0.06]])
    L = float((near - far).norm())
    n_in, n_out = int(round(L / (0.5 * V_PROG * DT))), int(round(L / (3.0 * V_PROG * DT)))
    pts = [far + (near - far) * (i / n_in) for i in range(n_in)] + [near + (far - near) * (i / n_out) for i in range(n_out)]
    return path_sum(ns, pts, [u] * len(pts)), len(pts)


def probe_coaxial(ns, miss=0.002, v_cross=0.05, rho=0.03):
    u = tilted_up(math.radians(45), 0.0)
    z = M[2] + 0.10
    n1 = int(round(2 * rho / (v_cross * DT)))
    leg = [torch.tensor([[M[0] - rho + 2 * rho * i / n1, M[1] + miss, z]]) for i in range(n1)]
    n2 = int(round(math.pi * rho / (0.5 * V_PROG * DT)))
    arc = [torch.tensor([[M[0] + rho * math.cos(math.pi * i / n2), M[1] + miss + rho * math.sin(math.pi * i / n2), z]])
           for i in range(n2)]
    return path_sum(ns, leg + arc, [u] * (n1 + n2)), path_sum(ns, list(reversed(leg + arc)), [u] * (n1 + n2)), n1 + n2


def probe_com(ns, com_c):
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
    return path_sum(ns, pts, ups, com_c)


def probe_first_out_flicker(ns, th_lo=30.0, th_hi=80.0, rate=1.0, flicker=True):
    """Spout inside the window, lean toward the receiver: tilt th_lo -> th_hi at `rate` rad/s with first_out = 1,
    back at the same rate with first_out = 0 (a bead poked past the rim and fell back in) when flicker=True."""
    # spout 0.4-0.7 r from the mouth horizontally and 0.8-2.6 r above it over 30..80 deg (inside the window)
    az = math.pi                                    # lean toward -x ... receiver on the -x side of the source
    src_p = torch.tensor([[M[0] + 0.07, 0.0, M[2] + 0.04]])
    n = int(round(math.radians(th_hi - th_lo) / (rate * DT)))
    ups = [tilted_up(math.radians(th_lo) + math.radians(th_hi - th_lo) * i / n, az) for i in range(n)] + \
          [tilted_up(math.radians(th_hi) - math.radians(th_hi - th_lo) * i / n, az) for i in range(n)]
    pts = [src_p] * (2 * n)
    fo = [1.0] * n + ([0.0] * n if flicker else [1.0] * n)
    fo[0] = 1.0
    tot, pen = path_sum(ns, pts, ups, first_out=fo, keys=("tilt_open",))
    # window check at the top
    c = make_ctx(src_p, ups[n - 1], (M - torch.tensor([0, 0, MZ])).unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]]),
                 torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3), prog=0.9)
    reset_cache(ns)
    t = terms(ns, c)
    return tot, pen, 2 * n, float(t["diag_win"]), float(t["diag_open_phi"])


def run_probes():
    out = {}
    for nm, ns in (("cand", NEW), ("iter_23", OLD)):
        (s, p), n = probe_speed_asym(ns)
        print(f"[5a] {nm}: slow-in / fast-out loop ({n} steps): aim+place {s:+.3f}, src_speed+spin {p:+.3f}")
        (f, pf), (b, pb), n = probe_coaxial(ns)
        print(f"[5b] {nm}: coaxial crossing loop ({n} steps): aim+place fwd {f:+.3f} rev {b:+.3f}, pen {pf:+.3f}/{pb:+.3f}")
        for cc in (0.0, COM_C):
            f, _ = probe_com(ns, cc)
            print(f"[5c] {nm}: tilt-slide-untilt loop com_c={cc:+.4f}: aim+place {f:+.3f}")
        for fl in (False, True):
            tot, pen, n, win, phi = probe_first_out_flicker(ns, flicker=fl)
            print(f"[5d] {nm}: tilt 30->80->30 deg in window (win {win:.2f}, PHI top {phi:.2f}), {n} steps, "
                  f"first_out flicker={fl}: tilt_open {tot:+.3f}, src_speed+spin {pen:+.3f}")
            out[f"{nm}_flicker_{fl}"] = tot
        out[f"{nm}_speed_asym"] = s
    return out


if __name__ == "__main__":
    print(f"candidate: {CAND}")
    r0 = check_identity()
    r1 = check_formula()
    r2 = check_still()
    r3 = [check_cycles(unclamped=True), check_cycles(), check_cycles(T=60, amp=(0.9, 0.05, 0.6)),
          check_cycles(T=30, amp=(1.2, 0.06, 0.8))]
    r4 = check_cache()
    r5 = run_probes()
