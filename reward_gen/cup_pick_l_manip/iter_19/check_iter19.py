"""Deterministic check of the iter_19 candidates, float64, CPU (reuses iter_18/check_curl.py).

iter_18/improvement.md plans NO reward change for round 19, so there is no changed term to probe. This script
  1. confirms each candidate against iter_18/compute_reward.py (md5, AST, AST without annotations);
  2. re-runs every iter_18 check_curl property on the iter_19 compute_reward.py and diffs it term by term
     against iter_18 on random states (must be exactly 0);
  3. re-checks the two unchanged RATE terms with closed cycles whose two legs run at DIFFERENT speeds
     (iter_18 only ran a symmetric 0.1 m/s loop): goal_progress (+ pen_drift) on a radial loop around the
     goal, lift_rise (+ pen_descend + pen_setdown) on a vertical loop below the latch. It reports the
     rate-only cycle sum and the whole-reward margin of cycling over staying still at the loop's best point.
Usage: python3 check_iter19.py
"""
import ast
import hashlib
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "iter_18"))
import check_curl as CC  # noqa: E402  (sets float64 as the default dtype)

DT = 1.0 / 60.0          # policy step: sim.dt 1/120 x decimation 2
BASE = HERE.parent / "iter_18" / "compute_reward.py"
CURLED, PALM_F = (2.0, 2.0, 1.0), 20.0


def load(path):
    mod = types.ModuleType(f"cr_{path.stem}")
    mod.__dict__["RewardContext"] = object      # the env loader injects this name; an unquoted annotation needs it
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod.compute_reward


def _strip_ann(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.arg):
            n.annotation = None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.returns = None
    return tree


def identity():
    src0 = BASE.read_text()
    a0 = ast.dump(ast.parse(src0))
    s0 = ast.dump(_strip_ann(ast.parse(src0)))
    out = {}
    for p in sorted(HERE.glob("compute_reward*.py")):
        src = p.read_text()
        row = {"md5": hashlib.md5(src.encode()).hexdigest()[:8], "bytes_equal": src == src0}
        try:
            row["ast_equal"] = ast.dump(ast.parse(src)) == a0
            row["ast_equal_wo_annotations"] = ast.dump(_strip_ann(ast.parse(src))) == s0
        except SyntaxError as e:
            row["syntax_error"] = f"line {e.lineno}: {e.msg}"
        out[p.name] = row
    return out


def random_diff(fn_a, fn_b):
    """term-by-term max |a - b| on the validator's fake ObjectRewardContext (several seeds, n=256)."""
    from openarm.agnostic.tasks.manip_stages.object_validator import make_fake_context
    worst = 0.0
    for seed in range(4):
        ctx = make_fake_context(256, device="cpu", seed=seed)
        ra, ta = fn_a(ctx)
        rb, tb = fn_b(ctx)
        worst = max(worst, (ra - rb).abs().max().item())
        for k in ta:
            worst = max(worst, (ta[k] - tb[k]).abs().max().item())
    return worst


# ---- radial loop around the goal: goal_progress + pen_drift ---------------------------------------
def _radial_leg(fn, d0, d1, speed, tol):
    """move the held object radially from goal_dist d0 to d1 at |speed| m/s; returns (rate sum, total sum, steps)."""
    n = max(1, round(abs(d1 - d0) / (speed * DT)))
    sign = 1.0 if d1 > d0 else -1.0                         # +1 = receding (cup_lin_vel along +x, goal on -x)
    rate = tot = 0.0
    for i in range(n):
        dd = d0 + (d1 - d0) * (i + 0.5) / n
        ctx = CC.make_ctx(CC.hand(CURLED), torch.tensor([PALM_F]), cup_speed=sign * speed, goal_dist=dd,
                          n_succ=3.0, success=False)
        ctx.success_tol = torch.full((1,), tol)
        r, c = fn(ctx)
        rate += (c["goal_progress"] + c["pen_drift"]).item()
        tot += r.item()
    return rate, tot, n


def _stay(fn, dd, tol, steps):
    ctx = CC.make_ctx(CC.hand(CURLED), torch.tensor([PALM_F]), cup_speed=0.0, goal_dist=dd, n_succ=3.0,
                      success=False)
    ctx.success_tol = torch.full((1,), tol)
    r, _ = fn(ctx)
    return r.item() * steps


def radial_cycles(fn, tol=0.0911):
    rows = []
    for d_in, d_out in ((0.03, 0.08), (0.10, 0.15), (0.13, 0.25)):
        for v_in in (0.02, 0.05, 0.10, 0.30):
            for v_out in (0.05, 0.10, 0.30, 0.60, 0.80):
                r1, t1, n1 = _radial_leg(fn, d_in, d_out, v_out, tol)       # recede
                r2, t2, n2 = _radial_leg(fn, d_out, d_in, v_in, tol)        # return
                stay = _stay(fn, d_in, tol, n1 + n2)
                rows.append({"loop": f"{d_in:.2f}<->{d_out:.2f}", "v_in": v_in, "v_out": v_out,
                             "rate_sum": r1 + r2, "whole_minus_stay": (t1 + t2) - stay, "steps": n1 + n2})
    return rows


def recede_cost_per_m(fn, dd=0.13, tol=0.0911):
    """rate-term charge per metre of receding at goal_dist dd, as a function of speed."""
    out = {}
    for v in (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.20):
        ctx = CC.make_ctx(CC.hand(CURLED), torch.tensor([PALM_F]), cup_speed=v, goal_dist=dd, n_succ=3.0,
                          success=False)
        ctx.success_tol = torch.full((1,), tol)
        _, c = fn(ctx)
        per_step = -(c["goal_progress"] + c["pen_drift"]).item()
        per_step_all = -(c["goal_progress"] + c["pen_drift"] + c["pen_cup_vel"]).item()
        out[f"{v:.2f} m/s"] = {"rate/m": round(per_step / (v * DT), 2), "rate+cup_vel/m": round(per_step_all / (v * DT), 2)}
    return out


# ---- vertical loop below the latch: lift_rise + pen_descend + pen_setdown ------------------------------
def _vert_ctx(h, vz):
    ctx = CC.make_ctx(CC.hand(CURLED), torch.tensor([PALM_F]), n_succ=0.0, success=False, goal_dist=0.25)
    ctx.cup_pos = torch.tensor([[0.30, 0.15, 0.20 + h]])
    ctx.goal_pos = torch.tensor([[0.30, 0.15, 0.45]])
    ctx.goal_dist = torch.tensor([0.25 - h])
    ctx.cup_lin_vel = torch.tensor([[0.0, 0.0, vz]])
    ctx.lifted = torch.zeros(1, dtype=torch.bool)
    return ctx


def vertical_cycles(fn):
    rows = []
    keys = ("lift_rise", "pen_descend", "pen_setdown")
    for h0, h1 in ((0.01, 0.05), (0.03, 0.08)):
        for v_up in (0.02, 0.05, 0.10):
            for v_dn in (0.05, 0.10, 0.30, 0.60):
                rate = tot = 0.0
                steps = 0
                for v, a, b, s in ((v_up, h0, h1, 1.0), (v_dn, h1, h0, -1.0)):
                    n = max(1, round((b - a) * s / (v * DT)))
                    steps += n
                    for i in range(n):
                        h = a + (b - a) * (i + 0.5) / n
                        r, c = fn(_vert_ctx(h, s * v))
                        rate += sum(c[k].item() for k in keys)
                        tot += r.item()
                stay = fn(_vert_ctx(h1, 0.0))[0].item() * steps
                rows.append({"loop": f"h {h0:.2f}<->{h1:.2f}", "v_up": v_up, "v_dn": v_dn, "rate_sum": rate,
                             "whole_minus_stay_top": tot - stay, "steps": steps})
    return rows


def main():
    print("== 1. identity vs iter_18/compute_reward.py")
    for k, v in identity().items():
        print(f"  {k:26s} {v}")
    fn18 = load(BASE)
    fn19 = load(HERE / "compute_reward.py")
    print("== 2. iter_18 check_curl properties on iter_19/compute_reward.py")
    for k, v in CC.layer_a("c1r", fn19).items():
        print(f"  A {k:46s} {v}")
    print(f"  sizing (at-success forces, e1000)            {CC.sizing(fn19)}")
    print(f"  round-17 trade (index to tip + palm)          {CC.r17_trade(fn19)}")
    print(f"  pen_table vs lowest link height               {CC.table_band(fn19)}")
    print(f"  closed loop 0.1 m/s both ways (iter_18 test)  {CC.cycle(fn19)}")
    print(f"  max |iter_19 - iter_18| over reward and all terms, fake ctx 4x256: {random_diff(fn18, fn19)}")
    print("== 3a. radial closed loops (goal_progress + pen_drift), tol 0.0911, curled grip, 3 successes")
    rows = radial_cycles(fn19)
    for r in rows:
        print(f"  {r['loop']}  v_in {r['v_in']:.2f}  v_out {r['v_out']:.2f}  rate_sum {r['rate_sum']:+8.3f}"
              f"  whole-stay {r['whole_minus_stay']:+10.2f}  steps {r['steps']}")
    pos = [r for r in rows if r["rate_sum"] > 0]
    farm = [r for r in rows if r["whole_minus_stay"] > 0]
    print(f"  rate_sum > 0 in {len(pos)}/{len(rows)} loops, max {max(r['rate_sum'] for r in rows):+.3f};"
          f" whole reward beats staying at the inner point in {len(farm)}/{len(rows)}")
    print("== 3b. rate-term charge per metre of receding at goal_dist 0.13 m")
    for k, v in recede_cost_per_m(fn19).items():
        print(f"  {k:9s} {v}")
    print("== 3c. vertical closed loops below the latch (lift_rise + pen_descend + pen_setdown)")
    vrows = vertical_cycles(fn19)
    for r in vrows:
        print(f"  {r['loop']}  v_up {r['v_up']:.2f}  v_dn {r['v_dn']:.2f}  rate_sum {r['rate_sum']:+8.3f}"
              f"  whole-stay_top {r['whole_minus_stay_top']:+9.2f}  steps {r['steps']}")
    vpos = [r for r in vrows if r["rate_sum"] > 0]
    vfarm = [r for r in vrows if r["whole_minus_stay_top"] > 0]
    print(f"  rate_sum > 0 in {len(vpos)}/{len(vrows)} loops, max {max(r['rate_sum'] for r in vrows):+.3f};"
          f" whole reward beats staying at the top in {len(vfarm)}/{len(vrows)}")


if __name__ == "__main__":
    main()
