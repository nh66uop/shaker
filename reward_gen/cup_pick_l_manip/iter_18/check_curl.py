"""Deterministic check of the iter_18 curl-as-factor candidates, float64, CPU (like iter_17/check_depth.py).

Two layers, both on each candidate's OWN code:
  A. end-to-end: the candidate's compute_reward on hand-built states. env_mult is read back from
     success_bonus / 62 (success on, no earlier success, object 1 m up, so off_table = 1 and the
     first-success factor = 1): no re-implementation of env_q.
  B. curl term alone: the candidate's curl lines, copied verbatim, for the light-force regime where soft
     contact is not saturated. Layer B is cross-checked against layer A on saturated states.
Properties: tip press < curl; a harder tip press never raises the curl term; near-zero forces go to 0
smoothly (no fallback); bounded; symmetric; a digit taken off never raises env_mult; no new rate term.
Usage: python3 check_curl.py [ids...]   (default: r17 c1 q2 q3 c1r when present)
"""
import math
import sys
import types
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
PROBE = 62.0   # W_SUCCESS_FIRST 60 + W_SUCCESS_LATER 2 at the first success


def load(cid):
    path = HERE.parent / "iter_17" / "compute_reward.py" if cid == "r17" else HERE / f"compute_reward_{cid}.py"
    mod = types.ModuleType(f"cr_{cid}")
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod.compute_reward


def make_ctx(link_f, palm_f, *, hand_rel=0.5, cup_speed=0.0, goal_dist=0.0, n_succ=0.0, success=True):
    """Held, upright, 1 m above spawn, at the goal. Thumb links and palm on the +x side of the object,
    finger links on the -x side (opposition 1). Every link point 0.5 m from the surface, so the
    pre-contact on-ramp terms are 0 and only forces move env_q."""
    n = link_f.shape[0]
    z = lambda *s: torch.zeros(*s)
    e = lambda v: torch.tensor(v).expand(n, 3).clone()
    cup = e([0.30, 0.15, 1.20])
    ex, ey, ez = e([1.0, 0, 0]), e([0, 1.0, 0]), e([0, 0, 1.0])
    side = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0]).view(1, 5, 1, 1)
    link_n = side * torch.tensor([1.0, 0, 0]).view(1, 1, 1, 3) * torch.ones(n, 5, 3, 1)
    link_pos = cup.view(n, 1, 1, 3) + 0.56 * link_n
    return types.SimpleNamespace(
        table_z=0.205, lift_latch_height=0.10, success_hold_steps=10, max_successes=5,
        palm_pos=cup + 0.56 * ex, palm_normal=-ex, palm_side=ey, palm_finger_dir=ez,
        link_pos=link_pos, link_cup_force=link_f, palm_cup_force=palm_f,
        hand_q=z(n, 19), hand_q_norm=torch.full((n, 19), 0.3), hand_target_norm=torch.full((n, 19), 0.3),
        hand_default_q_norm=torch.full((n, 19), 0.1), hand_qd=z(n, 19),
        hand_z_min=torch.full((n,), 0.205 + hand_rel), palm_clearance=torch.full((n,), 0.5),
        arm_q=z(n, 7), arm_qd=z(n, 7),
        cup_pos=cup, cup_quat=torch.tensor([1.0, 0, 0, 0]).expand(n, 4).clone(), cup_axis=ez,
        cup_tilt=z(n), cup_lin_vel=cup_speed * ex, cup_ang_vel=z(n, 3),
        cup_spawn_pos=e([0.30, 0.15, 0.20]), cup_radius=torch.full((n,), 0.06),
        cup_half_height=torch.full((n,), 0.05),
        goal_pos=cup - goal_dist * ex, goal_dist=torch.full((n,), float(goal_dist)),
        success_tol=torch.full((n,), 0.03), lifted=torch.ones(n, dtype=torch.bool),
        success=torch.full((n,), bool(success)), num_successes=torch.full((n,), float(n_succ)),
        episode_progress=torch.full((n,), 0.5),
        approach_done=torch.ones(n, dtype=torch.bool), envelope_done=torch.ones(n, dtype=torch.bool),
        actions=z(n, 26), prev_actions=z(n, 26),
        palm_surface_dist=torch.full((n,), 0.5), palm_surface_dir=-ex, palm_surface_normal=ex,
        link_surface_dist=torch.full((n, 5, 3), 0.5), link_surface_dir=-link_n, link_surface_normal=link_n,
    )


def env_mult(fn, link_f, palm_f=None):
    palm_f = torch.full((link_f.shape[0],), 20.0) if palm_f is None else palm_f
    _, comp = fn(make_ctx(link_f, palm_f))
    return comp["success_bonus"] / PROBE


def hand(*digits):
    """five digits given as (inner0, inner1, tip) or one triple repeated five times."""
    rows = list(digits) * 5 if len(digits) == 1 else list(digits)
    return torch.tensor([rows], dtype=torch.float64)


# ---- layer B: the curl lines of each candidate, verbatim (f: (...,3) raw link forces of ONE digit) ----
def _share(f):
    f = f.clamp(min=0.0)
    d = f.sum(-1)
    return ((f[..., 0] + f[..., 1]) / d.clamp(min=1e-6)).clamp(min=0.0, max=1.0), d


def curl_c1(f):
    s, d = _share(f)
    return (s / 0.50).clamp(max=1.0) * torch.tanh(d / 0.30)


def curl_c1r(f):
    s, _ = _share(f)
    inner = f.clamp(min=0.0)[..., 0] + f.clamp(min=0.0)[..., 1]
    return (s / 0.50).clamp(max=1.0) * torch.tanh(inner / 0.30)


def curl_q2(f):
    s, d = _share(f)
    return s * torch.tanh(d / 0.30)


def curl_q3(f):
    s, d = _share(f)
    return (0.30 + 0.70 * (s / 0.50).clamp(max=1.0)) * torch.tanh(d / 0.30)


CURL = {"c1": (curl_c1, 0.50), "c1r": (curl_c1r, 0.50), "q2": (curl_q2, 0.60), "q3": (curl_q3, 0.75)}


def layer_b(cid):
    curl, floor = CURL[cid]
    g = torch.linspace(0.0, 3.0, 61)
    i0, i1, t = torch.meshgrid(g, g, g, indexing="ij")
    f = torch.stack([i0, i1, t], -1)                        # (61,61,61,3), tip along the last grid axis
    c = curl(f)
    rise = (c[..., 1:] - c[..., :-1]).clamp(min=0.0)        # a harder tip must never raise the curl
    k = int(rise.argmax())
    where = f[..., :-1, :].reshape(-1, 3)[k].tolist()
    eps = torch.logspace(-9, -1, 50).view(-1, 1) * torch.tensor([[0.3, 0.3, 0.4]])
    near = curl(eps)
    gen = torch.Generator().manual_seed(0)
    rnd = torch.rand(200000, 3, generator=gen) * 50 * (torch.rand(200000, 3, generator=gen) > 0.3)
    cr = curl(rnd)
    return {
        "tip_press(0.3,0,5)": curl(torch.tensor([0.3, 0.0, 5.0])).item(),
        "curl(2,2,1)": curl(torch.tensor([2.0, 2.0, 1.0])).item(),
        "tip_only(0,0,3)": curl(torch.tensor([0.0, 0.0, 3.0])).item(),
        "no_contact": curl(torch.zeros(3)).item(),
        "max_rise_on_harder_tip": rise.max().item(), "at(i0,i1,tip)": [round(x, 2) for x in where],
        "near_zero_max(F<=1e-3)": near[:25].max().item(), "near_zero_monotone": bool((near[1:] >= near[:-1] - 1e-15).all()),
        "bounded": bool((cr >= 0).all() and (cr <= 1 + 1e-12).all() and torch.isfinite(cr).all()),
        "floor": floor,
    }


def layer_a(cid, fn):
    out = {}
    tip, crl, tip_only = hand((0.3, 0.0, 5.0)), hand((2.0, 2.0, 1.0)), hand((0.0, 0.0, 3.0))
    out["env_mult tip press(0.3,0,5)"] = env_mult(fn, tip).item()
    out["env_mult curl(2,2,1)"] = env_mult(fn, crl).item()
    out["env_mult tip only(0,0,3)"] = env_mult(fn, tip_only).item()
    # harder tip press, contact saturated (every link >= 1 N or fixed), five digits alike
    worst = 0.0
    for a, b in [(0.3, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 0.0), (5.0, 5.0)]:
        tips = torch.linspace(2.0, 30.0, 57)   # from 2 N: the tip's soft contact is saturated (1 - 5e-12)
        rows = torch.stack([torch.tensor([[a, b, float(t)]] * 5) for t in tips])
        em = env_mult(fn, rows)
        worst = max(worst, (em[1:] - em[:-1]).clamp(min=0.0).max().item())
    out["max rise of env_mult on harder tip (saturated)"] = worst
    # cross-check layer B against layer A on saturated states: env_base = 1 there
    if cid in CURL:
        curl, floor = CURL[cid]
        st = torch.tensor([[[1.0, 1.0, 1.0]] * 5, [[2.0, 2.0, 8.0]] * 5, [[3.0, 1.0, 1.0]] * 5, [[0.5, 0.5, 9.0]] * 5])
        pred_q = floor + (1 - floor) * curl(st).mean(-1)
        pred = (0.1 + 0.9 * pred_q) if cid != "q3" else (0.1 + 0.9 * 1.0) * pred_q
        out["layerB_vs_A max abs err"] = (pred - env_mult(fn, st)).abs().max().item()
    # bounded + finite on random states (spikes to 1000 N, zeros, negatives clamped by the code)
    gen = torch.Generator().manual_seed(1)
    f = torch.rand(4000, 5, 3, generator=gen) * 12 * (torch.rand(4000, 5, 3, generator=gen) > 0.35)
    f[::50] *= 80.0
    f[::97] = -f[::97]
    pf = torch.rand(4000, generator=gen) * 25 * (torch.rand(4000, generator=gen) > 0.3)
    em = env_mult(fn, f, pf)
    out["env_mult range (random)"] = [round(em.min().item(), 4), round(em.max().item(), 4)]
    # symmetry: permuting the four fingers' forces never changes env_mult
    perm = f[:, [0, 2, 4, 1, 3], :]
    out["finger permutation max diff"] = (env_mult(fn, perm, pf) - em).abs().max().item()
    # a digit taken off the object never raises env_mult
    rises = []
    for d in range(5):
        g = f.clone()
        g[:, d, :] = 0.0
        rises.append((env_mult(fn, g, pf) - em).clamp(min=0.0).max().item())
    out["digit removed: max rise"] = max(rises)
    return out


# ---- today's grasp and the round-17 trade, end-to-end ------------------------------------------------
def at_success_e1000():
    """fixed-tol eval, forces at success (summary.md, e1000): inner split evenly over k=0,1."""
    return [(11.6, 11.6, 12.52), (1.31, 1.31, 7.87), (0.20, 0.20, 11.02), (0.215, 0.215, 8.62), (0.04, 0.04, 1.84)]


def sizing(fn):
    base = at_success_e1000()
    em0 = env_mult(fn, hand(*base)).item()
    one = list(base)
    one[2] = (6.0, 6.0, 11.02)                              # middle carries ~half inside
    em1 = env_mult(fn, hand(*one)).item()
    allc = [(max(a, 4.0), max(b, 4.0), c) for a, b, c in base]
    allc = [(max(a, c / 2), max(b, c / 2), c) for a, b, c in allc]
    emf = env_mult(fn, hand(*allc)).item()
    return {"today": round(em0, 4), "+1 digit curled": round(em1 - em0, 4), "all curled": round(emf, 4),
            "+1 digit / all curled": round((em1 - em0) / emf, 4), "today / all curled": round(em0 / emf, 4)}


def r17_trade(fn):
    """replay means while held: index 2.3 N inside / 9.8 N tip (e1000) -> 0.4 / 14.8 (m17 e800), palm soft
    contact 0.825 -> 0.901 (force chosen so tanh(F/0.15) hits the mean). Hold and lift income, B/A."""
    other = [(13.5, 13.5, 12.0), None, (0.3, 0.3, 11.0), (0.4, 0.4, 8.6), (0.05, 0.05, 1.8)]
    a = list(other); a[1] = (0.0, 2.3, 9.8)
    b = list(other); b[1] = (0.0, 0.4, 14.8)
    pa = torch.tensor([0.15 * math.atanh(0.825)])
    pb = torch.tensor([0.15 * math.atanh(0.901)])
    ca = make_ctx(hand(*a), pa, n_succ=3.0, success=False)
    cb = make_ctx(hand(*b), pb, n_succ=3.0, success=False)
    c = list(other); c[1] = (0.0, 2.3, 9.8)                # index as at e1000, palm as at m17
    d = list(other); d[1] = (5.0, 5.0, 9.8)                # index curled (half or more inside), palm as at m17
    _, ta = fn(ca)
    _, tb = fn(cb)
    _, tc = fn(make_ctx(hand(*c), pb, n_succ=3.0, success=False))
    _, td = fn(make_ctx(hand(*d), pb, n_succ=3.0, success=False))
    r = lambda x, y, k: round((x[k] / y[k]).item(), 4)
    return {"hold B/A": r(tb, ta, "goal_hold"), "lift B/A": r(tb, ta, "lift_height"),
            "hold C/B": r(tc, tb, "goal_hold"), "hold D/B": r(td, tb, "goal_hold")}


def cycle(fn):
    """closed radial loop 0.10 -> 0.15 -> 0.10 m from the goal at 0.1 m/s (30 steps each way), 60 Hz.
    Sum of goal_progress + pen_drift over the loop: constant curled grip, and the adversarial loop that
    recedes with a fingertip grip and returns with a curled one (grip modulated in step with the motion)."""
    curled, tipped = (2.0, 2.0, 1.0), (0.3, 0.0, 5.0)
    ds = [0.10 + 0.05 * i / 30 for i in range(30)]
    def leg(grip, sign):
        tot = 0.0
        for dd in (ds if sign > 0 else ds[::-1]):
            _, comp = fn(make_ctx(hand(grip), torch.tensor([20.0]), cup_speed=0.1 * sign, goal_dist=dd,
                                  n_succ=3.0, success=False))
            tot += (comp["goal_progress"] + comp["pen_drift"]).item()
        return tot
    return {"constant grip": round(leg(curled, +1) + leg(curled, -1), 4),
            "recede tipped, return curled": round(leg(tipped, +1) + leg(curled, -1), 4)}


def table_band(fn):
    f = hand((2.0, 2.0, 1.0))
    vals = {}
    for h in (0.03, 0.02, 0.015, 0.01, 0.005, 0.0):
        _, comp = fn(make_ctx(f, torch.tensor([20.0]), hand_rel=h))
        vals[f"{h * 100:.1f}cm"] = round(-comp["pen_table"].item(), 4)
    return vals


def main():
    ids = sys.argv[1:] or [c for c in ("r17", "c1", "q2", "q3", "c1r")
                           if c == "r17" or (HERE / f"compute_reward_{c}.py").exists()]
    for cid in ids:
        fn = load(cid)
        print(f"===== {cid}")
        if cid in CURL:
            for k, v in layer_b(cid).items():
                print(f"  B {k:34s} {v}")
        for k, v in layer_a(cid, fn).items():
            print(f"  A {k:46s} {v}")
        print(f"  sizing (at-success forces, e1000)            {sizing(fn)}")
        print(f"  round-17 trade (index to tip + palm)          {r17_trade(fn)}")
        print(f"  pen_table vs lowest link height               {table_band(fn)}")
        print(f"  closed loop, goal_progress + pen_drift        {cycle(fn)}")
    print("rate/difference terms: no candidate adds one (goal_progress and lift_rise are unchanged from iter_17)")


if __name__ == "__main__":
    main()
