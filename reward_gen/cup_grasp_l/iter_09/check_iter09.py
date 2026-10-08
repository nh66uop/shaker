"""Deterministic check of the cup_grasp iter_09 candidates (base iter_02) (GraspRewardContext: proximal links; L1-L6 regressions + L7/L8) (float64, CPU).

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
        goal_dist=0.25, arm_qd=0.0, act=0.0, cup_speed=0.0, success=False, n_succ=0.0, hspeed=0.0, tilt_deg=0.0, prox_f=None, prox_sd=None, up_digits=(), prog=0.3):
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
    tilt = torch.full((n,), tilt_deg * 3.14159265 / 180.0)
    pf = z(n, 5) if prox_f is None else prox_f
    psd = torch.full((n, 5), palm_d + 0.02) if prox_sd is None else prox_sd
    prox_n = side[..., 0, :] * torch.tensor([1.0, 0, 0]).view(1, 1, 3) * torch.ones(n, 5, 1)
    prox_pos = cup.view(n, 1, 3) + (r + psd).unsqueeze(-1) * prox_n
    lsn = link_n.clone(); psn = prox_n.clone()
    for dgt in up_digits:
        lsn[:, dgt, :, :] = torch.tensor([0.0, 0.0, 1.0]); psn[:, dgt, :] = torch.tensor([0.0, 0.0, 1.0])
    return types.SimpleNamespace(
        prox_pos=prox_pos, prox_cup_force=pf, prox_surface_dist=psd, prox_surface_dir=-prox_n, prox_surface_normal=psn,
        table_z=0.205, lift_latch_height=0.10, success_hold_steps=10, max_successes=5,
        palm_pos=palm, palm_normal=-ex, palm_side=ey, palm_finger_dir=ez,
        link_pos=link_pos, link_cup_force=lf, palm_cup_force=torch.full((n,), float(palm_f)),
        hand_q=z(n, 19), hand_q_norm=q, hand_target_norm=q.clone(), hand_default_q_norm=default, hand_qd=z(n, 19),
        hand_z_min=torch.full((n,), 0.205 + 0.03 + h), palm_clearance=torch.full((n,), 0.03 + h),
        arm_q=z(n, 7), arm_qd=torch.full((n, 7), arm_qd / 7 ** 0.5),
        cup_pos=cup, cup_quat=torch.tensor([[1.0, 0, 0, 0]]), cup_axis=ez, cup_tilt=tilt,
        cup_lin_vel=cup_speed * ez + hspeed * ex, cup_ang_vel=z(n, 3), cup_spawn_pos=spawn, cup_radius=torch.full((n,), r),
        cup_half_height=torch.full((n,), 0.07), goal_pos=goal, goal_dist=torch.full((n,), float(goal_dist)),
        success_tol=torch.full((n,), 0.09), lifted=torch.full((n,), bool(lifted)),
        success=torch.full((n,), bool(success)), num_successes=torch.full((n,), float(n_succ)),
        episode_progress=torch.full((n,), float(prog)), approach_done=torch.full((n,), palm_d < 0.03),
        envelope_done=torch.zeros(n, dtype=torch.bool),
        actions=torch.full((n, 26), float(act)), prev_actions=z(n, 26),
        palm_surface_dist=torch.full((n,), float(palm_d)), palm_surface_dir=-ex, palm_surface_normal=ex,
        link_surface_dist=lsd, link_surface_dir=-link_n, link_surface_normal=lsn)

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
        # L6 (iter_01 Hypothesis A): the left pinch state (thumb + one finger, palm 4 cm off) must be worth less than
        #     seating the palm on that pinch (palm contact, object nudged 2 cm/s, tilt 2 deg), and the palm must be
        #     worth more than a third digit added to the pinch without the palm
        lfp = torch.zeros(1, 5, 3); sdp = torch.full((1, 5, 3), 0.04)
        for d in (0, 3):
            lfp[0, d, 1:] = 2.0; sdp[0, d] = torch.tensor([0.025, 0.004, 0.001])
        kwp = dict(curl=0.45, link_f=lfp, link_sd=sdp)
        pinch, _ = R(fn, palm_d=0.04, **kwp)
        seated, _ = R(fn, palm_d=0.004, palm_f=3.0, hspeed=0.02, tilt_deg=2.0, **kwp)
        lf3 = lfp.clone(); sd3 = sdp.clone(); lf3[0, 1, 1:] = 2.0; sd3[0, 1] = torch.tensor([0.025, 0.004, 0.001])
        third, _ = R(fn, palm_d=0.04, curl=0.45, link_f=lf3, link_sd=sd3)
        res.append(("L6 pinch < pinch+palm seated (object nudged)", pinch < seated, f"pinch {pinch:+.4f} seated {seated:+.4f} margin {seated - pinch:+.4f}"))
        res.append(("L6b pinch+third digit < pinch+palm seated", third < seated, f"third digit {third:+.4f} seated {seated:+.4f}"))
        # L7 (iter_02 Hypothesis A): at the goal, held, five digits carried by their segments (proximal + middle + distal
        #     touching, pad too) must clearly beat five digits pressing with the pad (pad + distal link only)
        kw7 = dict(palm_d=0.0, palm_f=10.0, curl=0.6, h=0.25, lifted=True, goal_dist=0.02)
        lfp7, sdp7 = grasp(5, "press"); lfr7, sdr7 = grasp(5, "roll")
        press7, _ = R(fn, link_f=lfp7, link_sd=sdp7, prox_f=torch.zeros(1, 5), prox_sd=torch.full((1, 5), 0.035), **kw7)
        wrap7, _ = R(fn, link_f=lfr7, link_sd=sdr7, prox_f=torch.full((1, 5), 3.0), prox_sd=torch.full((1, 5), 0.015), **kw7)
        res.append(("L7 at goal: segment wrap > fingertip press", wrap7 > press7 * 1.15, f"press {press7:+.4f} wrap {wrap7:+.4f} ratio {press7 / max(wrap7, 1e-9):.3f} (need < 0.87)"))
        # L8: from the tip-only grasp on the table (4 digits, palm seated), pulling the middle and proximal links from
        #     3.3 cm to 1.5 cm off the wall (still no contact) must raise the reward, and touching with them more
        kw8 = dict(palm_d=0.0, palm_f=10.0, curl=0.5, h=0.0)
        lf8, sd8 = grasp(4, "press")
        far8, _ = R(fn, link_f=lf8, link_sd=sd8, prox_sd=torch.full((1, 5), 0.033), **kw8)
        sd8n = sd8.clone(); sd8n[0, :4, 0] = 0.015
        near8, _ = R(fn, link_f=lf8, link_sd=sd8n, prox_sd=torch.full((1, 5), 0.015), **kw8)
        lf8t = lf8.clone(); lf8t[0, :4, 0] = 3.0
        touch8, _ = R(fn, link_f=lf8t, link_sd=sd8n, prox_f=torch.cat([torch.full((1, 4), 3.0), torch.zeros(1, 1)], 1), prox_sd=torch.full((1, 5), 0.015), **kw8)
        res.append(("L8 tip-only -> segments near -> segments touching rises", far8 < near8 < touch8, f"far {far8:+.4f} near {near8:+.4f} touching {touch8:+.4f}"))
        # L9: the segment pull must not make the table state beat lifting (L3 boundary with segments touching)
        up8, _ = R(fn, link_f=lf8t, link_sd=sd8n, prox_f=torch.cat([torch.full((1, 4), 3.0), torch.zeros(1, 1)], 1), prox_sd=torch.full((1, 5), 0.015), palm_d=0.0, palm_f=10.0, curl=0.5, h=0.03)
        res.append(("L9 segments touching: table < 3 cm up", touch8 < up8, f"table {touch8:+.4f} 3cm {up8:+.4f}"))
        # L10 (iter_03 Hypothesis A): on the table, palm seated, middle+pinky by segments and thumb by its proximal
        #      link (the left arm's stalled state): closing the ring, then ring+index, onto the object must RAISE the
        #      per-step reward; and lifting the completed grasp 3 cm must still clearly beat keeping it on the table
        def st(digs, h=0.0):
            lf = torch.zeros(1, 5, 3); sd = torch.full((1, 5, 3), 0.04); pf = torch.zeros(1, 5); psd = torch.full((1, 5), 0.035)
            for d in digs: lf[0, d, 1] = 2.0; sd[0, d] = torch.tensor([0.02, 0.012, 0.004])
            pf[0, 0] = 2.0; psd[0, 0] = 0.015
            return R(fn, palm_d=0.004, palm_f=3.0, curl=0.45, link_f=lf, link_sd=sd, prox_f=pf, prox_sd=psd, h=h)[0]
        s2, s3, s4, s4u = st((2, 4)), st((2, 3, 4)), st((1, 2, 3, 4)), st((1, 2, 3, 4), h=0.03)
        res.append(("L10 table: +ring and +ring+index raise reward", s2 < s3 < s4, f"2+thumb {s2:+.4f} +ring {s3:+.4f} +index {s4:+.4f}"))
        res.append(("L10b completed grasp: lift 3 cm >> stay (gain > 0.5)", s4u - s4 > 0.5, f"table {s4:+.4f} lifted {s4u:+.4f} gain {s4u - s4:+.4f}"))
        # L11 (iter_04 Hypothesis A): palm seated, ONLY index and thumb flexed 1.5 cm from the object (index pad
        #      touching). With their nearest surface on the RIM/top (normal up) the closing credit and the segment pull
        #      must be <= 30 % of the same digits next to the SIDE WALL (iter_03: close 0.078 vs 0.117 = 66 %)
        def st11(up):
            lf = torch.zeros(1, 5, 3); sd = torch.full((1, 5, 3), 0.04); pf = torch.zeros(1, 5); psd = torch.full((1, 5), 0.035)
            sd[0, 2:] = 0.10; psd[0, 2:] = 0.10   # only thumb and index take part: middle/ring/pinky far away
            for dg in (0, 1): sd[0, dg] = torch.tensor([0.015, 0.015, 0.015]); psd[0, dg] = 0.015
            lf[0, 1, 2] = 2.0
            return R(fn, palm_d=0.004, palm_f=3.0, curl=0.7, link_f=lf, link_sd=sd, prox_f=pf, prox_sd=psd, up_digits=up)
        (tw, cw), (tr, cr) = st11(()), st11((0, 1))
        cl_w, cl_r = float(cw["close"]), float(cr["close"])
        sp_w, sp_r = float(cw.get("grasp_seg_pull", torch.zeros(1))), float(cr.get("grasp_seg_pull", torch.zeros(1)))
        res.append(("L11 above rim: close credit <= 30 % of side wall", cl_r <= 0.3 * cl_w + 1e-6, f"close wall {cl_w:.4f} rim {cl_r:.4f}"))
        res.append(("L11b above rim: segment pull <= 30 % of side wall", sp_r <= 0.3 * sp_w + 1e-6, f"pull wall {sp_w:.4f} rim {sp_r:.4f} | total wall {tw:+.4f} rim {tr:+.4f}"))
        # L11c (iter_05 Hypothesis A, rim guard on contact): a digit TOUCHING with its middle link on the rim/top
        #      earns <= 30 % of the same touch on the side wall in segment pull
        def st11c(up):
            lf = torch.zeros(1, 5, 3); sd = torch.full((1, 5, 3), 0.10); pf = torch.zeros(1, 5); psd = torch.full((1, 5), 0.10)
            sd[0, 1] = torch.tensor([0.012, 0.012, 0.012]); psd[0, 1] = 0.012; lf[0, 1, 0] = 2.0
            return R(fn, palm_d=0.004, palm_f=3.0, curl=0.7, link_f=lf, link_sd=sd, prox_f=pf, prox_sd=psd, up_digits=up)[1]
        cwc, crc = st11c(()), st11c((1,))
        pw, pr = float(cwc.get("grasp_seg_pull", torch.zeros(1))), float(crc.get("grasp_seg_pull", torch.zeros(1)))
        res.append(("L11c touching on rim: segment pull <= 30 % of side wall", pr <= 0.3 * pw + 1e-6, f"pull wall {pw:.4f} rim {pr:.4f}"))
        # L12 (operator: smooth hand-over; iter_04 park): parked 3.8 cm in front with fingers curled (no touch) <
        #      2.5 cm closer < first digit touching < two digits < two digits + palm, early AND late in the episode
        def st12(stage, prog):
            lf = torch.zeros(1, 5, 3); pf = torch.zeros(1, 5)
            if stage == 0: pd, ld, prd = 0.038, 0.035, 0.040
            else: pd, ld, prd = 0.025, 0.020, 0.025
            sd = torch.full((1, 5, 3), ld); psd = torch.full((1, 5), prd); palm_f = 0.0
            if stage >= 2: lf[0, 1, 0] = 0.5; sd[0, 1, 0] = 0.012
            if stage >= 3: lf[0, 0, 0] = 0.5; sd[0, 0, 0] = 0.012
            if stage >= 4: palm_f, pd = 1.0, 0.004
            return R(fn, palm_d=pd, palm_f=palm_f, curl=0.45, link_f=lf, link_sd=sd, prox_f=pf, prox_sd=psd, prog=prog)[0]
        for prog in (0.3, 0.8):
            v = [st12(k, prog) for k in range(5)]
            res.append((f"L12 park < closer < 1 touch < 2 touch < +palm (progress {prog})", all(a < b for a, b in zip(v, v[1:])),
                        " ".join(f"{x:+.4f}" for x in v)))
        # L13 continuity: one digit's middle-link force 0 -> 1 N (0.01 N steps) and all links 5 cm -> 1.2 cm (no contact),
        #      late in the episode: never decreasing (tol 1e-4) and no step larger than 0.02 per increment
        def sweep(vals, mk):
            r = [mk(x) for x in vals]; d = [b - a for a, b in zip(r, r[1:])]
            return min(d), max(abs(x) for x in d), r[0], r[-1]
        def f_state(fz):
            lf = torch.zeros(1, 5, 3); lf[0, 1, 0] = fz; sd = torch.full((1, 5, 3), 0.02); sd[0, 1, 0] = 0.012
            return R(fn, palm_d=0.025, curl=0.45, link_f=lf, link_sd=sd, prox_sd=torch.full((1, 5), 0.025), prog=0.8)[0]
        def d_state(dd):
            return R(fn, palm_d=dd + 0.005, curl=0.45, link_sd=torch.full((1, 5, 3), dd), prox_sd=torch.full((1, 5), dd + 0.005), prog=0.8)[0]
        mn, mx, a, b = sweep([i * 0.01 for i in range(101)], f_state)
        res.append(("L13a first-touch force sweep monotone and smooth", mn > -1e-4 and mx < 0.02, f"min step {mn:+.5f} max |step| {mx:.5f} 0N {a:+.4f} 1N {b:+.4f}"))
        mn, mx, a, b = sweep([0.05 - i * 0.00038 for i in range(101)], d_state)
        res.append(("L13b approach sweep 5 -> 1.2 cm monotone and smooth", mn > -1e-4 and mx < 0.02, f"min step {mn:+.5f} max |step| {mx:.5f} 5cm {a:+.4f} 1.2cm {b:+.4f}"))
        # L14: flexion in the air (nothing touching) must not raise the palm-seating income
        p0 = float(R(fn, palm_d=0.03, curl=0.0, link_sd=torch.full((1, 5, 3), 0.03))[1]["palm_in"])
        p1 = float(R(fn, palm_d=0.03, curl=0.45, link_sd=torch.full((1, 5, 3), 0.03))[1]["palm_in"])
        res.append(("L14 air flexion does not raise palm seating", p1 <= p0 + 1e-9, f"open {p0:.4f} curled {p1:.4f}"))
        # L18 (iter_09 Hypothesis A): lifted at the goal, palm seated. Five digits wrapped by segments vs the iter_02
        #      right-arm hold (thumb, ring, pinky wrapped; index and middle off the wall, 3.5 cm) vs index and middle
        #      pressing with the pad only (middle link 2.5 cm off). The partial holds must earn clearly less.
        def held(mode):
            lf = torch.zeros(1, 5, 3); sd = torch.full((1, 5, 3), 0.035); pf = torch.zeros(1, 5); psd = torch.full((1, 5), 0.035)
            for d in range(5):
                wrapped = mode == "five" or d in (0, 3, 4)
                if wrapped:
                    lf[0, d] = torch.tensor([3.0, 3.0, 3.0]); sd[0, d] = torch.tensor([0.012, 0.008, 0.001]); pf[0, d] = 3.0; psd[0, d] = 0.015
                elif mode == "pad":
                    lf[0, d] = torch.tensor([0.0, 3.0, 6.0]); sd[0, d] = torch.tensor([0.025, 0.004, 0.001])
            return R(fn, palm_d=0.0, palm_f=10.0, curl=0.6, h=0.25, lifted=True, goal_dist=0.02, link_f=lf, link_sd=sd, prox_f=pf, prox_sd=psd)[0]
        h5, h3, hp = held("five"), held("three"), held("pad")
        res.append(("L18 held at goal: 3 digits (index+middle off) <= 0.80 x five wrapped", h3 <= 0.80 * h5, f"five {h5:+.3f} three {h3:+.3f} ratio {h3 / max(h5, 1e-9):.2f}"))
        res.append(("L18b held at goal: index+middle pad-press <= 0.75 x five wrapped", hp <= 0.75 * h5, f"five {h5:+.3f} pad {hp:+.3f} ratio {hp / max(h5, 1e-9):.2f}"))
        from openarm.agnostic.tasks.manip_stages.grasp_validator import make_fake_context
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
