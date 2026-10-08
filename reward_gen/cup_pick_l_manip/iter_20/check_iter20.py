"""Deterministic check of the iter_20 candidates, float64, CPU (reuses iter_18/check_curl.py make_ctx and helpers).

iter_19/improvement.md plans ONE change: the per-digit curl measure inside the round-18 curl factor becomes
    lift_off = link_surface_dist.amax(-1) - link_surface_dist.amin(-1)
    follow   = exp(-(lift_off - FOLLOW_DEAD).clamp(min=0) / FOLLOW_SCALE),  digit_curl = digit_touch * follow
with FOLLOW_DEAD 0.010, FOLLOW_SCALE 0.006; inner_share, press, CURL_FULL_SHARE, CURL_FORCE_REF removed.

  1. identity: md5, AST against iter_19/compute_reward.py and against c1, the changed statements and what they read.
  2. properties, on each candidate's OWN code, two ways:
       end-to-end   compute_reward on hand-built states, env_mult read back as success_bonus / 62
       instrumented the candidate's compute_reward body cut after `env_mult = ...` and `return locals()`,
                    so hand_curl / digit_curl come from the candidate's lines, not a re-implementation
     tip press < curl on the diagnosis' own shapes; a harder tip never raises the curl; a digit removed never
     raises it; bounded; smooth near zero; symmetric; relative; the farthest link moved off never raises it;
     exploit probes (open hand, straight finger tangent to the wall, fingers parallel to the cup axis).
  3. offline replay on the m18_curl traces (DATA rl/eval/cup_pick_l/m18_curl/raw/*.trace.npz, per-step ctx of
     the traced envs, float16 on disk -> float64 here): held steps (lifted and dz >= 3 cm) of e1000 and m18 e800,
     both judged tip press by the operator, and the operator's own frames (video replica, env 11 = cup_big_s130).
Usage: python3 check_iter20.py [ids...]   (default: every compute_reward_<id>.py present)
"""
import ast
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "iter_18"))
import check_curl as CC  # noqa: E402  (sets float64 as the default dtype)

BASE = HERE.parent / "iter_19" / "compute_reward.py"
TRACES = Path("/media/user/DATA/kuku_ai/rl/eval/cup_pick_l/m18_curl/raw")
PROBE = CC.PROBE
FOLLOW_DEAD, FOLLOW_SCALE = 0.010, 0.006
# ctx.link_surface_dist medians of m18_e800 held steps (m18_curl/tools/check_follow.py, cand_eval.py)
SHAPES = {"roll (ring)": (0.018, 0.017, 0.011), "roll (index, rare)": (0.022, 0.012, 0.012),
          "pip_hook (index)": (0.030, 0.012, 0.013), "tip_hook (middle)": (0.034, 0.037, 0.013),
          "tip_hook (thumb)": (0.024, 0.034, 0.012)}
CURLED = {"roll (ring)", "roll (index, rare)"}
FORCES = {"tip press (0.3,0,5)": (0.3, 0.0, 5.0), "inner (2,2,1)": (2.0, 2.0, 1.0), "tip only (0,0,3)": (0.0, 0.0, 3.0)}


# ---------------------------------------------------------------------------------------------- loading
def _ns(path):
    mod = types.ModuleType(f"cr_{path.stem}")
    mod.__dict__["RewardContext"] = object          # the env loader injects it; an unquoted annotation needs it
    return mod.__dict__


def load(path):
    ns = _ns(path)
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns["compute_reward"]


def load_locals(path, stop="env_mult"):
    """the candidate's compute_reward body up to and including `stop = ...`, returning locals()."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_reward")
    idx = next(i for i, s in enumerate(fn.body) if isinstance(s, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == stop for t in s.targets))
    fn.body = fn.body[:idx + 1] + [ast.parse("return locals()").body[0]]
    ast.fix_missing_locations(tree)
    ns = _ns(path)
    exec(compile(tree, str(path), "exec"), ns)
    return ns["compute_reward"]


# ---------------------------------------------------------------------------------------------- 1. identity
def _strip_ann(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.arg):
            n.annotation = None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.returns = None
    return tree


def _body(src):
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compute_reward")
    return [ast.dump(s) for s in fn.body], fn


def identity(ids):
    src0 = BASE.read_text()
    c1 = (HERE / "compute_reward_c1.py").read_text()
    b0, _ = _body(src0)
    out = {}
    for cid in ids:
        src = (HERE / f"compute_reward_{cid}.py").read_text()
        b, fn = _body(src)
        removed = [s for s in b0 if s not in b]
        added = [s for s in b if s not in b0]
        stmts = [s for s in fn.body if ast.dump(s) in added]
        reads = sorted({f"ctx.{n.attr}" for s in stmts for n in ast.walk(s)
                        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "ctx"})
        names = sorted({n.id for s in stmts for n in ast.walk(s) if isinstance(n, ast.Name)})
        const_idx = [ast.unparse(n) for s in stmts for n in ast.walk(s)
                     if isinstance(n, ast.Subscript) and any(isinstance(c, ast.Constant) and isinstance(c.value, int)
                                                             for c in ast.walk(n.slice))]
        out[cid] = {
            "md5": hashlib.md5(src.encode()).hexdigest()[:8],
            "ast_equal_c1": ast.dump(ast.parse(src)) == ast.dump(ast.parse(c1)),
            "ast_equal_c1_wo_ann": ast.dump(_strip_ann(ast.parse(src))) == ast.dump(_strip_ann(ast.parse(c1))),
            "signature_equal_iter19": ast.dump(ast.parse(src).body[-1].args) == ast.dump(ast.parse(src0).body[-1].args),
            "stmts_removed_vs_iter19": len(removed), "stmts_added_vs_iter19": len(added),
            "added": [ast.unparse(s) for s in stmts],
            "added_reads_ctx": reads, "added_names": names, "added_const_index": const_idx,
        }
    return out


# ---------------------------------------------------------------------------------------------- 1b. term diff
DOWNSTREAM = {"lift_height", "lift_rise", "goal_reach", "goal_progress", "goal_hold", "success_bonus"}


def term_diff(fn_a, fn_b):
    """max |a - b| per term on the validator's fake ObjectRewardContext (4 seeds x 256)."""
    sys.path.insert(0, str(HERE.parents[2] / "source" / "openarm"))
    from openarm.agnostic.tasks.manip_stages.object_validator import make_fake_context
    worst = {}
    for seed in range(4):
        ctx = make_fake_context(256, device="cpu", seed=seed)
        ra, ta = fn_a(ctx)
        rb, tb = fn_b(ctx)
        worst["reward"] = max(worst.get("reward", 0.0), (ra - rb).abs().max().item())
        for k in ta:
            worst[k] = max(worst.get(k, 0.0), (ta[k] - tb[k]).abs().max().item())
    return worst


# ---------------------------------------------------------------------------------------------- 2. properties
def make_ctx(link_f, sdist, palm_f=None, **kw):
    """iter_18 make_ctx (held, upright, 1 m up, at the goal, thumb/palm on +x, fingers on -x: opposition 1)
    with the link surface distances set. Pre-contact terms move; env_mult (read back from success_bonus) does
    not see them."""
    n = link_f.shape[0]
    palm_f = torch.full((n,), 20.0) if palm_f is None else palm_f
    ctx = CC.make_ctx(link_f, palm_f, **kw)
    ctx.link_surface_dist = sdist.expand(n, 5, 3).clone() if sdist.dim() == 3 else sdist
    return ctx


def env_mult(fn, ctx):
    return fn(ctx)[1]["success_bonus"] / PROBE


def five(tri, n=1):
    return torch.tensor([[list(tri)] * 5] * n, dtype=torch.float64)


def shapes_table(fn, fl):
    rows = {}
    for sname, sd in SHAPES.items():
        for fname, f in FORCES.items():
            ctx = make_ctx(five(f), five(sd))
            loc = fl(ctx)
            rows[(sname, fname)] = (loc["hand_curl"].item(), loc["curl_factor"].item(), env_mult(fn, ctx).item(),
                                    loc["env_mult"].item())
    return rows


def harder_tip(fn, fl):
    """tip force up with the other two links fixed, every shape: max rise of hand_curl and env_mult.
    From 0.5 N (soft contact 0.9993) and from 2 N (saturated); plus a coupled model where 10 N pushes the tip
    point 1 mm deeper (the only physical way harder pressing changes the geometry)."""
    out = {}
    for lo in (0.0, 0.5, 2.0):
        tips = torch.linspace(lo, 30.0, 300)
        worst_h, worst_m = 0.0, 0.0
        for sd in SHAPES.values():
            for a, b in [(0.0, 0.0), (0.3, 0.0), (1.0, 1.0), (2.0, 2.0), (5.0, 0.0)]:
                f = torch.stack([torch.tensor([[a, b, float(t)]] * 5) for t in tips])
                ctx = make_ctx(f, five(sd))
                h = fl(ctx)["hand_curl"]
                m = env_mult(fn, ctx)
                worst_h = max(worst_h, (h[1:] - h[:-1]).clamp(min=0).max().item())
                worst_m = max(worst_m, (m[1:] - m[:-1]).clamp(min=0).max().item())
        out[f"tip {lo:.1f}->30 N: max rise hand_curl / env_mult"] = (worst_h, worst_m)
    tips = torch.linspace(2.0, 30.0, 300)
    worst = 0.0
    for sd in SHAPES.values():
        f = torch.stack([torch.tensor([[0.0, 0.0, float(t)]] * 5) for t in tips])
        s = five(sd, len(tips))
        s[..., 2] = (s[..., 2] - 1e-4 * tips.view(-1, 1)).clamp(min=0.0)
        h = fl(make_ctx(f, s))["hand_curl"]
        worst = max(worst, (h[1:] - h[:-1]).clamp(min=0).max().item())
    out["coupled, tip 2->30 N and 1 mm deeper per 10 N: max rise hand_curl"] = worst
    return out


def random_states(n=6000, seed=3):
    gen = torch.Generator().manual_seed(seed)
    f = torch.rand(n, 5, 3, generator=gen) * 12 * (torch.rand(n, 5, 3, generator=gen) > 0.35)
    f[::50] *= 80.0
    f[::97] = -f[::97]
    sd = 0.005 + 0.05 * torch.rand(n, 5, 3, generator=gen)
    sd[::41] = -sd[::41] * 0.2                  # a point reading inside the surface
    sd[::53] *= 20.0                             # far off
    pf = torch.rand(n, generator=gen) * 25 * (torch.rand(n, generator=gen) > 0.3)
    return f, sd, pf


def props(fn, fl):
    out = {}
    f, sd, pf = random_states()
    ctx = make_ctx(f, sd, pf)
    loc = fl(ctx)
    em = env_mult(fn, ctx)
    h, cf, dc = loc["hand_curl"], loc["curl_factor"], loc["digit_curl"]
    out["bounded digit_curl [min,max], finite"] = [dc.min().item(), dc.max().item(), bool(torch.isfinite(dc).all())]
    out["bounded curl_factor"] = [cf.min().item(), cf.max().item()]
    out["bounded env_mult (end-to-end)"] = [em.min().item(), em.max().item(), bool(torch.isfinite(em).all())]
    out["instrumented env_mult == end-to-end (max abs)"] = (loc["env_mult"] - em).abs().max().item()
    # removal: a digit's force to 0 (geometry kept), or force 0 and the digit far away
    rh, rm = [], []
    for d in range(5):
        for far in (False, True):
            g, s = f.clone(), sd.clone()
            g[:, d, :] = 0.0
            if far:
                s[:, d, :] = 0.5
            c2 = make_ctx(g, s, pf)
            rh.append((fl(c2)["hand_curl"] - h).clamp(min=0).max().item())
            rm.append((env_mult(fn, c2) - em).clamp(min=0).max().item())
    out["digit removed: max rise hand_curl / env_mult"] = [max(rh), max(rm)]
    # symmetry
    p5 = [3, 0, 4, 1, 2]
    out["5-digit permutation: max |d hand_curl|"] = (fl(make_ctx(f[:, p5], sd[:, p5], pf))["hand_curl"] - h).abs().max().item()
    p4 = [0, 2, 4, 1, 3]
    out["4-finger permutation: max |d env_mult|"] = (env_mult(fn, make_ctx(f[:, p4], sd[:, p4], pf)) - em).abs().max().item()
    # relative: the same offset on a digit's three points
    out["offset +2 cm on all three points: max |d digit_curl|"] = (fl(make_ctx(f, sd + 0.02, pf))["digit_curl"] - dc).abs().max().item()
    far = sd.argmax(dim=-1, keepdim=True)
    s2 = sd.scatter_add(-1, far, torch.full_like(far, 0.01, dtype=sd.dtype))
    out["farthest point 1 cm further: max rise digit_curl"] = (fl(make_ctx(f, s2, pf))["digit_curl"] - dc).max().item()
    mid = sd.argsort(dim=-1)[..., 1:2]
    s3 = sd.scatter(-1, mid, sd.amax(dim=-1, keepdim=True))
    out["middle point out to the farthest: max rise digit_curl"] = (fl(make_ctx(f, s3, pf))["digit_curl"] - dc).max().item()
    # force scaled x3 on touching digits (no force target)
    touching = (f.clamp(min=0) > 0.5).any(-1)
    out["force x3 on touching digits: max |d digit_curl|"] = ((fl(make_ctx(f * 3, sd, pf))["digit_curl"] - dc).abs() * touching).max().item()
    # smooth near zero force: no fallback, linear on-ramp
    eps = torch.logspace(-9, 0, 60)
    fe = torch.stack([torch.tensor([[0.3 * e, 0.3 * e, 0.4 * e]] * 5) for e in eps])
    he = fl(make_ctx(fe, five(SHAPES["roll (ring)"], len(eps))))["hand_curl"]
    out["force->0 (roll geometry): hand_curl at F<=1e-3 N max"] = he[eps <= 1e-3].max().item()
    out["force->0: monotone, 0 at 1e-9 N"] = [bool((he[1:] >= he[:-1] - 1e-15).all()), he[0].item()]
    # continuity in lift_off: steepest step of follow on a 0.1 mm grid, and the flat zone
    lo = torch.linspace(0.0, 0.05, 501)
    s = torch.stack([torch.tensor([[0.012, 0.012 + float(x), 0.012]] * 5) for x in lo])
    hl = fl(make_ctx(five((0.0, 0.0, 3.0), len(lo)), s))["hand_curl"]
    out["lift_off 0->5 cm, 0.1 mm grid: max |step| of hand_curl"] = (hl[1:] - hl[:-1]).abs().max().item()
    out["lift_off <= FOLLOW_DEAD: hand_curl min (flat at touch)"] = hl[lo <= FOLLOW_DEAD].min().item()
    return out


def cycle_geom(fn):
    """iter_18 closed radial loop (0.10 -> 0.15 -> 0.10 m, 0.1 m/s, 30 steps each way) with the curl modulated
    by GEOMETRY in step with the motion: recede with every digit hooked (tip_hook middle), return rolled
    (roll ring). Same forces both ways. Sum of goal_progress + pen_drift (the rate pair) and of the whole reward."""
    f = five((2.0, 2.0, 1.0))
    ds = [0.10 + 0.05 * i / 30 for i in range(30)]

    def leg(shape, sign):
        rate, whole = 0.0, 0.0
        for dd in (ds if sign > 0 else ds[::-1]):
            r, comp = fn(make_ctx(f, five(SHAPES[shape]), torch.tensor([20.0]), cup_speed=0.1 * sign, goal_dist=dd,
                                  n_succ=3.0, success=False))
            rate += (comp["goal_progress"] + comp["pen_drift"]).item()
            whole += r.item()
        return rate, whole
    roll = "roll (ring)"
    hook = "tip_hook (middle)"
    a = [x + y for x, y in zip(leg(roll, +1), leg(roll, -1))]
    b = [x + y for x, y in zip(leg(hook, +1), leg(roll, -1))]
    stay = fn(make_ctx(f, five(SHAPES[roll]), torch.tensor([20.0]), goal_dist=0.10, n_succ=3.0, success=False))[0].item() * 60
    return {"constant roll: rate, whole": [round(x, 3) for x in a],
            "recede hooked, return rolled: rate, whole": [round(x, 3) for x in b],
            "stay rolled at 0.10 m for 60 steps: whole": round(stay, 3)}


def exploit_probes(fn, fl):
    """open hand; straight finger tangent to a cylinder (link spacing measured in the traces:
    k0->k1 3.88 cm, k1->tip 2.55 cm fingers, 3.63 cm thumb/pinky); fingers parallel to the cup axis."""
    out = {}
    ctx = make_ctx(torch.zeros(1, 5, 3), five((0.012, 0.012, 0.012)))
    out["open hand, fingers along the wall but not touching: hand_curl"] = fl(ctx)["hand_curl"].item()
    for r in (0.0345, 0.0398, 0.0458, 0.0518):              # outer-wall radius medians s085, s100, s115, s130
        for where, t in (("tangent at k1", 3.88), ("best tangent (mid-chain)", 3.215), ("tangent at the tip", 6.43)):
            xs = torch.tensor([0.0, 3.88, 6.43]) / 100.0 - t / 100.0
            gap = (torch.sqrt(r * r + xs * xs) - r)
            lift = (gap.max() - gap.min()).item()
            sd = gap + 0.010                                   # the link point sits ~1 cm inside its link
            c = fl(make_ctx(five((0.0, 2.0, 1.0)), five(tuple(sd.tolist()))))["hand_curl"].item()
            out[f"straight distal chain R {r * 100:.2f} cm, {where}: lift_off {lift * 100:.2f} cm -> curl"] = round(c, 3)
    # four fingers straight along a generatrix (parallel to the cup axis), all three links touching, palm on the
    # same side, thumb opposite: the case the diagnosis listed as unverified
    ctx = make_ctx(five((2.0, 2.0, 2.0)), five((0.012, 0.012, 0.012)))
    ctx.palm_surface_normal = -ctx.palm_surface_normal        # palm on the fingers' side
    out["fingers parallel to the axis + thumb opposite: hand_curl, env_mult"] = (
        round(fl(ctx)["hand_curl"].item(), 3), round(env_mult(fn, ctx).item(), 3))
    ctx = make_ctx(five((2.0, 2.0, 1.0)), five(SHAPES["roll (ring)"]))
    out["reference: five rolled digits (ring roll geometry): hand_curl, env_mult"] = (
        round(fl(ctx)["hand_curl"].item(), 3), round(env_mult(fn, ctx).item(), 3))
    return out


# ---------------------------------------------------------------------------------------------- 3. replay
class Trace:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.tr = z["trace"].astype(np.float64)
        self.length = z["length"]
        self.envs = list(z["envs"])
        self.species = [str(s) for s in z["species"]]
        self.base = {}
        for i, c in enumerate(str(c) for c in z["channels"]):
            self.base.setdefault(c.split("[")[0], i)

    def get(self, name, e, n=1):
        i = self.base[name]
        x = self.tr[e, :self.length[e], i:i + n]
        return x[:, 0] if n == 1 else x


def trace_ctx(tr, e, mask):
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a[mask]))  # noqa: E731
    f = t(tr.get("link_f", e, 15)).view(-1, 5, 3)
    ctx = make_ctx(f, t(tr.get("link_sdist", e, 15)).view(-1, 5, 3), t(tr.get("palm_f", e)))
    ctx.link_surface_normal = t(tr.get("link_n", e, 45)).view(-1, 5, 3, 3)
    ctx.palm_surface_normal = t(tr.get("palm_sn", e, 3))
    ctx.palm_surface_dist = t(tr.get("palm_sdist", e))
    ctx.palm_normal = t(tr.get("palm_normal", e, 3))
    ctx.hand_q_norm = t(tr.get("qn", e, 19))
    return ctx


GROUPS = {"s130": ("cup_big_s130",), "shaker": ("shaker_closed",)}


def replay(fn, fl, fn19, fl19, label):
    tr = Trace(TRACES / f"{label}.json.trace.npz")
    acc = {k: [] for k in ("h20", "h19", "m20", "m19", "m20_e2e", "touch", "dc", "g", "jit")}
    for e in range(len(tr.envs)):
        held = (tr.get("lifted", e) > 0.5) & (tr.get("dz", e) >= 0.03)
        if held.sum() == 0:
            continue
        ctx = trace_ctx(tr, e, held)
        l20, l19 = fl(ctx), fl19(ctx)
        acc["h20"].append(l20["hand_curl"]); acc["h19"].append(l19["hand_curl"])
        acc["m20"].append(l20["env_mult"]); acc["m19"].append(l19["env_mult"])
        acc["m20_e2e"].append(env_mult(fn, ctx))
        acc["touch"].append(l20["digit_touch"]); acc["dc"].append(l20["digit_curl"])
        acc["g"] += [tr.species[e]] * int(held.sum())
        acc["jit"].append((l20["digit_curl"][1:] - l20["digit_curl"][:-1]).abs().mean(0, keepdim=True))
    c = {k: (torch.cat(v) if k != "g" else np.array(v)) for k, v in acc.items()}
    h20, touch, dc = c["h20"], c["touch"], c["dc"]
    allf = touch.mean(-1)          # the same states with every touching digit following the wall
    rep = {
        "held steps": len(h20),
        "hand_curl iter_19 -> iter_20 (mean)": (round(c["h19"].mean().item(), 3), round(h20.mean().item(), 3)),
        "hand_curl iter_20 median / p90": (round(h20.median().item(), 3), round(h20.quantile(0.9).item(), 3)),
        "share of held steps hand_curl >= 0.5": round((h20 >= 0.5).double().mean().item(), 4),
        "same states, every touching digit follows: hand_curl": round(allf.mean().item(), 3),
        "iter_20 hand_curl / that": round((h20.mean() / allf.mean()).item(), 3),
        "digit_curl mean thumb..pinky": [round(x, 3) for x in dc.mean(0).tolist()],
        "share of digit-steps digit_curl >= 0.5 thumb..pinky": [round(x, 3) for x in (dc >= 0.5).double().mean(0).tolist()],
        "env_mult iter_19 -> iter_20": (round(c["m19"].mean().item(), 3), round(c["m20"].mean().item(), 3),
                                        f"{(c['m20'].mean() / c['m19'].mean() - 1).item():+.1%}"),
        "instrumented vs end-to-end env_mult (max abs)": (c["m20"] - c["m20_e2e"]).abs().max().item(),
        "step-to-step |d digit_curl| mean over envs": [round(x, 3) for x in c["jit"].mean(0).tolist()],
    }
    for gname, mem in GROUPS.items():
        m = torch.from_numpy(np.isin(c["g"], mem))
        if m.any():
            rep[f"{gname}: hand_curl iter_19 -> iter_20"] = (round(c["h19"][m].mean().item(), 3), round(h20[m].mean().item(), 3))
    return rep


def frames(fl, fl19, label, env=11, times=(1, 3, 5, 7, 9, 11, 13, 14.9)):
    tr = Trace(TRACES / f"{label}.json.trace.npz")
    e = tr.envs.index(env)
    rows = []
    for t in times:
        step = int(round(t * 60))
        if step > tr.length[e]:
            rows.append(f"t {t:>4} s: past the traced first episode ({tr.length[e]} steps)")
            continue
        m = np.zeros(tr.length[e], dtype=bool)
        m[step - 1] = True
        ctx = trace_ctx(tr, e, m)
        l20, l19 = fl(ctx), fl19(ctx)
        held = bool((tr.get("lifted", e)[step - 1] > 0.5) and (tr.get("dz", e)[step - 1] >= 0.03))
        lo = (ctx.link_surface_dist.amax(-1) - ctx.link_surface_dist.amin(-1))[0] * 100
        rows.append(f"t {t:>4} s held {int(held)}: lift_off cm {[round(x, 1) for x in lo.tolist()]} "
                    f"digit_curl {[round(x, 2) for x in l20['digit_curl'][0].tolist()]} "
                    f"hand_curl iter_20 {l20['hand_curl'].item():.3f} (iter_19 {l19['hand_curl'].item():.3f})")
    return rows


def main():
    ids = sys.argv[1:] or sorted(p.stem.split("_")[-1] for p in HERE.glob("compute_reward_*.py"))
    print("== 1. identity against iter_19/compute_reward.py and c1")
    for cid, row in identity(ids).items():
        print(f"  {cid}: " + ", ".join(f"{k} {v}" for k, v in row.items() if k != "added"))
        for s in row["added"]:
            print(f"      + {s}")
    fn19, fl19 = load(BASE), load_locals(BASE)
    fn_c1 = load(HERE / "compute_reward_c1.py")
    d = term_diff(fn19, fn_c1)
    same = sorted(k for k, v in d.items() if v == 0.0)
    diff = {k: round(v, 6) for k, v in d.items() if v != 0.0}
    print(f"  c1 vs iter_19, fake ctx 4x256: terms exactly equal {same}")
    print(f"  c1 vs iter_19, fake ctx 4x256: terms that differ (max abs) {diff}; all downstream x env_mult: "
          f"{set(diff) - {'reward'} <= DOWNSTREAM}")
    for cid in ids:
        if cid != "c1":
            print(f"  {cid} vs c1, fake ctx 4x256: max abs over reward and all terms "
                  f"{max(term_diff(fn_c1, load(HERE / f'compute_reward_{cid}.py')).values())}")
    for cid in ids:
        path = HERE / f"compute_reward_{cid}.py"
        fn, fl = load(path), load_locals(path)
        print(f"===== {cid}")
        print("  -- shapes (five digits alike): hand_curl, curl_factor, env_mult end-to-end, env_mult instrumented")
        for (s, f), v in shapes_table(fn, fl).items():
            print(f"    {s:20s} {f:20s} " + "  ".join(f"{x:.4f}" for x in v))
        for k, v in harder_tip(fn, fl).items():
            print(f"  {k:60s} {v}")
        for k, v in props(fn, fl).items():
            print(f"  {k:60s} {v}")
        for k, v in exploit_probes(fn, fl).items():
            print(f"  X {k:80s} {v}")
        print(f"  iter_18 table band / closed loop (link points 0.5 m off: curl 1 both ways): table {CC.table_band(fn)}, cycle {CC.cycle(fn)}")
        print(f"  closed loop with the curl modulated by geometry: {cycle_geom(fn)}")
        if cid != "c1":
            continue
        print("  -- offline replay, held steps (lifted and dz >= 3 cm), m18_curl traces")
        for label in ("e1000_s42", "m18_e800_s42", "vid_e1000", "vid_m18_e800"):
            print(f"  [{label}]")
            for k, v in replay(fn, fl, fn19, fl19, label).items():
                print(f"    {k:55s} {v}")
        for label in ("vid_e1000", "vid_m18_e800"):
            print(f"  [{label} env 11 cup_big_s130, the operator's frames]")
            for r in frames(fl, fl19, label):
                print(f"    {r}")


if __name__ == "__main__":
    main()
