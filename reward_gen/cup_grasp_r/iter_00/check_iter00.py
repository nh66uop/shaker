"""Deterministic check of the cup_grasp iter_00 candidates (float64, CPU).

Per-step income ledger on hand-built states (RoutingReview 09.26: every boundary must satisfy stay < advance):
  L1 random start 0.31 m away: standing still vs moving towards the object (arm 0.5 rad/s, actions 0.3), and the
     reach gradient 0.31 -> 0.25 m
  L2 open hand pressed on the wall, no digit contact  <  one digit touching with one link
  L3 four-digit grasp with palm on the table          <  the same grasp 3 cm up (and 8 cm up)
  L4 at the goal, held: five digits fingertip-pressing (lift_off 2.5 cm)  <  five digits rolled (lift_off 1.1 cm)
  L5 bounded, finite on random fake contexts (object_validator.make_fake_context)
Usage: python3 check_iter00.py [ids...]   (default: every compute_reward_<id>.py present)
"""
import importlib.util, sys, types
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent

def load(cid):
    spec = importlib.util.spec_from_file_location(f"cr_{cid}", HERE / f"compute_reward_{cid}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.compute_reward

def ctx(*, palm_d=0.31, curl=0.0, link_f=None, link_sd=None, palm_f=0.0, h=0.0, lifted=False,
        goal_dist=0.25, arm_qd=0.0, act=0.0, cup_speed=0.0, success=False, n_succ=0.0):
    n = 1
    z = lambda *s: torch.zeros(*s)
    e = lambda v: torch.tensor([v], dtype=torch.float64)
    spawn = e([0.30, 0.15, 0.25]); cup = spawn + e([0, 0, h])
    ex, ey, ez = e([1.0, 0, 0]), e([0, 1.0, 0]), e([0, 0, 1.0])
    r = 0.045
    palm = cup + (r + palm_d) * ex
    side = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0]).view(1, 5, 1, 1)
    link_n = side * torch.tensor([1.0, 0, 0]).view(1, 1, 1, 3) * torch.ones(n, 5, 3, 1)
    lsd = torch.full((n, 5, 3), palm_d + 0.03) if link_sd is None else link_sd
    link_pos = cup.view(n, 1, 1, 3) + (r + lsd).unsqueeze(-1) * link_n
    lf = z(n, 5, 3) if link_f is None else link_f
    default = torch.full((n, 19), 0.1)
    q = default + curl * 0.8
    goal = cup - goal_dist * ex if goal_dist > 0 else cup.clone()
    tilt = z(n)
    return types.SimpleNamespace(
        table_z=0.205, lift_latch_height=0.10, success_hold_steps=10, max_successes=5,
        palm_pos=palm, palm_normal=-ex, palm_side=ey, palm_finger_dir=ez,
        link_pos=link_pos, link_cup_force=lf, palm_cup_force=torch.full((n,), float(palm_f)),
        hand_q=z(n, 19), hand_q_norm=q, hand_target_norm=q.clone(), hand_default_q_norm=default, hand_qd=z(n, 19),
        hand_z_min=torch.full((n,), 0.205 + 0.03 + h), palm_clearance=torch.full((n,), 0.03 + h),
        arm_q=z(n, 7), arm_qd=torch.full((n, 7), arm_qd / 7 ** 0.5),
        cup_pos=cup, cup_quat=torch.tensor([[1.0, 0, 0, 0]]), cup_axis=ez, cup_tilt=tilt,
        cup_lin_vel=cup_speed * ez, cup_ang_vel=z(n, 3), cup_spawn_pos=spawn, cup_radius=torch.full((n,), r),
        cup_half_height=torch.full((n,), 0.07), goal_pos=goal, goal_dist=torch.full((n,), float(goal_dist)),
        success_tol=torch.full((n,), 0.09), lifted=torch.full((n,), bool(lifted)),
        success=torch.full((n,), bool(success)), num_successes=torch.full((n,), float(n_succ)),
        episode_progress=torch.full((n,), 0.3), approach_done=torch.full((n,), palm_d < 0.03),
        envelope_done=torch.zeros(n, dtype=torch.bool),
        actions=torch.full((n, 26), float(act)), prev_actions=z(n, 26),
        palm_surface_dist=torch.full((n,), float(palm_d)), palm_surface_dir=-ex, palm_surface_normal=ex,
        link_surface_dist=lsd, link_surface_dir=-link_n, link_surface_normal=link_n)

def grasp(nd, shape="roll", f=3.0):
    """nd digits in contact (thumb first). roll: all three points on the wall, force on all three;
    press: middle 2.5 cm off, only the distal link + pad carry force."""
    lf = torch.zeros(1, 5, 3); sd = torch.full((1, 5, 3), 0.04)
    for d in range(nd):
        if shape == "roll":
            lf[0, d] = torch.tensor([f, f, f]); sd[0, d] = torch.tensor([0.012, 0.008, 0.001])
        else:
            lf[0, d] = torch.tensor([0.0, f, 2 * f]); sd[0, d] = torch.tensor([0.026, 0.004, 0.001])
    return lf, sd

def R(fn, **kw):
    tot, comp = fn(ctx(**kw)); return float(tot.reshape(-1)[0]), comp

def main(ids):
    ok_all = {}
    for cid in ids:
        fn = load(cid); res = []
        # L1: discounted return (gamma 0.99, 100 steps) from a random-init-like hand (curl 0.35):
        #     stand at 0.31 m for 100 steps  vs  move in 60 steps to 1 cm, then 40 steps with one digit touching
        g = 0.99
        stand = sum(g ** t * R(fn, palm_d=0.31, curl=0.35)[0] for t in range(100))
        path = 0.0
        for t in range(60):
            d = 0.31 - (0.30 * (t + 1) / 60)
            path += g ** t * R(fn, palm_d=d, curl=0.35, arm_qd=0.5, act=0.3)[0]
        lf1 = torch.zeros(1, 5, 3); lf1[0, 1, 2] = 2.0; sd1 = torch.full((1, 5, 3), 0.03); sd1[0, 1, 2] = 0.001
        for t in range(60, 100):
            path += g ** t * R(fn, palm_d=0.01, curl=0.35, link_f=lf1, link_sd=sd1)[0]
        step_stand = R(fn, palm_d=0.31, curl=0.35)[0]; step_close = R(fn, palm_d=0.25, curl=0.35, arm_qd=0.5, act=0.3)[0]
        res.append(("L1 return: approach-and-touch beats standing (curl 0.35)", path > stand,
                    f"stand {stand:+.3f} approach {path:+.3f} | one-step stand {step_stand:+.4f} closer-moving {step_close:+.4f}"))
        flat, _ = R(fn, palm_d=0.005, palm_f=5.0, link_sd=torch.full((1, 5, 3), 0.03))
        lf = torch.zeros(1, 5, 3); lf[0, 1, 2] = 2.0; sd = torch.full((1, 5, 3), 0.03); sd[0, 1, 2] = 0.001
        one, _ = R(fn, palm_d=0.005, palm_f=5.0, curl=0.2, link_f=lf, link_sd=sd)
        res.append(("L2 flat open hand on wall < one digit contact", flat < one, f"flat {flat:+.4f} one-digit {one:+.4f}"))
        lf4, sd4 = grasp(4); table, _ = R(fn, palm_d=0.0, palm_f=10.0, curl=0.5, link_f=lf4, link_sd=sd4, h=0.0)
        up3, _ = R(fn, palm_d=0.0, palm_f=10.0, curl=0.5, link_f=lf4, link_sd=sd4, h=0.03)
        up8, _ = R(fn, palm_d=0.0, palm_f=10.0, curl=0.5, link_f=lf4, link_sd=sd4, h=0.08)
        res.append(("L3 4-digit grasp: table < 3 cm < 8 cm", table < up3 < up8, f"table {table:+.4f} 3cm {up3:+.4f} 8cm {up8:+.4f}"))
        lfp, sdp = grasp(5, "press"); lfr, sdr = grasp(5, "roll")
        kw = dict(palm_d=0.0, palm_f=10.0, curl=0.6, h=0.25, lifted=True, goal_dist=0.02)
        press, _ = R(fn, link_f=lfp, link_sd=sdp, **kw); roll, _ = R(fn, link_f=lfr, link_sd=sdr, **kw)
        res.append(("L4 at goal: fingertip press < rolled", press < roll, f"press {press:+.4f} roll {roll:+.4f} ratio {press / max(roll, 1e-9):.3f}"))
        from openarm.agnostic.tasks.manip_stages.object_validator import make_fake_context
        fk = make_fake_context(64, seed=3); t, _ = fn(fk)
        res.append(("L5 finite on fake contexts", bool(torch.isfinite(t).all()), f"range {float(t.min()):+.3f}..{float(t.max()):+.3f}"))
        print(f"== {cid}")
        for name, ok, info in res: print(f"  {'PASS' if ok else 'FAIL'}  {name}: {info}")
        ok_all[cid] = all(ok for _, ok, _ in res)
    print("SUMMARY", ok_all)

if __name__ == "__main__":
    sys.path.insert(0, "/home/user/rl_ws/hdgp/source/openarm")
    ids = sys.argv[1:] or sorted(p.stem.split("_", 2)[2] for p in HERE.glob("compute_reward_*.py"))
    main(ids)
