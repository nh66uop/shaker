ROUND 27 NOTES (09.26, diagnosis of t2r_i26r = iter_26 c1r trained FRESH). Facts only. The three changes
below are the only changes for this round; everything else in iter_26 stays as it is.

TASK DEFINITION (operator, 09.25, unchanged): grasp both cups with both hands and keep the envelope (all
five fingers on the cup, grip held), pour the REQUESTED amount (the target amount that is in the actor
observation), then bring the source cup back upright and return to the pose it had before tilting.

WHAT t2r_i26r DID (deterministic replay, ADR 0, 64 envs x 900 steps; 10 active beads p50, request 5.5 p50)
| metric | ep300 | ep600 | ep800 |
|---|---|---|---|
| success / any bead poured / any bead spilled | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| both cups held at the end | 0.98 | 0.94 | 0.98 |
| episode peak TOTAL tilt p50 (p10-p90) | 41 (38-50) deg | 48 (45-52) deg | 33.5 (27-40) deg |
| episode peak tilt TOWARD the receiver (src_tilt_toward_rcv) p50 / p90 | -2 / 13.5 deg | 5.8 / 10.9 deg | 6.5 / 10.0 deg |
| spout (src_pour_lip_pos) to receiver mouth, horizontal, end p50 | 0.41 m | 0.33 m | 0.19 m |
| source cup body to receiver mouth, horizontal, end p50 | 0.44 m | 0.37 m | 0.23 m (5.6 cup radii) |
| source cup y / receiver cup y at the end p50 | -0.195 / +0.242 | -0.180 / +0.189 | -0.064 / +0.160 |
| receiver cup distance from its spawn, end p50 | 8.8 cm | 5.3 cm | 2.8 cm |
| source 5-finger contact while carrying (1 N) / pinky alone | 0.93 / 0.99 | 0.98 / 0.99 | 0.38 / 0.39 |
| receiver 5-finger contact while carrying (1 N) / pinky alone | 0.73 / 0.87 | 0.31 / 0.59 | 0.17 / 0.19 |
| tilt_rate per episode, paid / charged | 0 / -125 | 0 / -82 | 0 / -76 |
Tensorboard per-step means (21-epoch windows at e100 / e300 / e600 / e790, then the last 20 epochs to
e866): tilt_rate -0.29 / -0.22 / -0.16 / -0.17 / -0.16 (the largest charge; action_rate -0.04), aim
0.005 / 0.02 / 0.15 / 0.26 / 0.34, task/aim_dist 0.52 / 0.43 / 0.40 / 0.29 / 0.24 m, home_rcv 0.20 /
0.36 / 0.46 / 0.54 / 0.51, source wrap count 2.51 / 2.44 / 2.39 / 2.03 / 1.65, finish and pour_flow 0
throughout, adr_progress 0.

WHY (verified by recomputing the iter_26 terms offline on the trace states)
1. The rotation term never paid; it only charged. Its pay gate contains spout = fine * height, and the
   spout never came near the mouth: spout <= 1e-17 on every active step of all three traces. So every
   increase of (1 - up_z)/2 paid 0 and every decrease was charged in full (g_charge = 1 before the first
   bead). That makes it a tax on holding any tilt and on moving the cup. Charge per step by total tilt
   (ep300): 0-10 deg -0.03, 30-40 deg -0.16, 40-50 deg -0.22. By source cup speed at 25-35 deg (ep800):
   < 3 cm/s -0.06, 6-10 cm/s -0.12, > 10 cm/s -0.22. Moving the source cup 1 cm closer earns only +0.025
   aim per step at the ep800 states, so a quick approach or an exploratory tilt costs more than it earns.
2. No pouring rotation ever started. The "peak tilt" above is a backward lean of the source cup toward
   the torso (cup axis x component -0.51 to -0.71 at the end, toward-receiver component -0.18 to +0.04)
   that grows with lift height (correlation 0.61-0.93). The first bead needs about 90-98 deg total tilt.
   At the same lift height (0.16-0.20 m) the lean fell from 36.6 deg (ep600) to 28.1 deg (ep800).
3. The receiver does not come to meet the source, and the meeting point is off the centre line. At any
   tilt the spout is at most cup_mouth_z * sin + r * cos <= 0.108 m (2.6 cup radii) from the source cup
   axis, so the source body must come within about 3.6 cup radii of the receiver mouth; at ep800 it is
   5.6 cup radii away. home_rcv pays exp(-4 * distance from the receiver's SPAWN). On the ep800 states,
   moving the receiver 2 / 4 / 8 / 12 cm toward the source changes aim + home_rcv by +0.027 / +0.035 /
   -0.042 / -0.97 per step. So the receiver stays within 3-4 cm of spawn (y ~ +0.16), the source does
   about 0.2 m of the travel alone, and the implied pour point is on the receiver's side (source body at
   y ~ +0.02..+0.06), not at y ~ 0.
4. The source five-finger loss is the pinky (thumb, index, middle and ring stay >= 0.985). Pinky contact
   at the lift reference fell from 97% (ep600) to 59% (ep800) of episodes. Where the reference had it, it
   stayed on for 43% of the carry steps, and less as the source moves toward the centre (50% at y -0.15
   to -0.10, 24% at y > -0.05). The reward does not ask for it: the pre-lift posture saturates at 3
   wrapped fingers (wrap_q = min(wrap_count / 3, 1)) and the source keep has no fingers-kept factor.
   Losing the pinky lowers keep_src from 0.644 to 0.617, i.e. -0.04 per step. It is not the tilt charge:
   at equal tilt the charge is slightly larger with the pinky off (-0.116 vs -0.099 per step).
5. The receiver five-finger loss is invisible to its iter_26 term. A finger counts as "still touching"
   at 0.2 N, and only if its force share at lift was > 0.02. The receiver lifts with the pinky barely on
   (1 N at lift in 6-20% of episodes, share > 0.02 in 39-72%), so kept_rcv stays 0.95-0.99 while the
   1 N five-finger contact falls from 0.73 to 0.17.
6. finish never opened (want = 1 on every step because nothing was poured), so the iter_26 finish,
   spill and want changes are still untested.

REQUIRED CHANGES FOR ITER_27 (these three only)
1. Rotation help as the exact rate of a potential. This replaces the iter_26 tilt_rate and keeps its
   name, TILT_W, TILT_PAY_CAP and the switch-off once the first bead has LEFT the source cup.
   - Potential Phi = latches * spout * lean. lean = max(src_cup_up . d, 0), with d the horizontal pour
     direction (ctx.pour_dir_xy), is how far the cup axis leans TOWARD the receiver (no angle constant).
     spout = the iter_26 spout geometry (fine * height, cup-radius widths). latches = settled *
     (1 - released) * want; these only move one way before the first bead.
   - Pay TILT_W * dPhi/dt, where dPhi/dt is the FULL rate of Phi, including the change of spout and d
     caused by both cups' motion. One way is (Phi(now) - Phi(state moved back by a small time tau along
     ctx.src/rcv_cup_lin_vel and _ang_vel)) / tau, recomputing mouth, d and lip with the formulas in the
     ctx comments. Holding any pose earns 0.
   - A positive rate may be multiplied by factors in [0, 1] that have no rate (carry, grip_gate,
     not_nested, dir_ok, side_ok, up_rcv) and capped at TILT_PAY_CAP per step. A negative rate is charged
     in full and uncapped. None of these factors may sit inside Phi (an uncharged gate change is a pump).
   - Check deterministically, as iter_26/check_tilt_rate.py did: (a) every closed cycle sums to <= 0,
     including cycles that move the tilted spout in and out of the mouth and cycles that toggle the
     factors; (b) a tilt made while spout = 0 pays 0 and costs 0, so there is no net charge; (c) a lean
     away from or sideways to the receiver earns nothing.
2. Meet near the centre line. home_rcv keeps its form (HOME_W, carry_rcv, up_rcv, side_ok and still_rcv
   unchanged), but its anchor moves from the receiver's spawn to its spawn x at side margin SIDE_OK
   (ctx.rcv_side_margin = SIDE_OK, the smallest margin that still has full side_ok; the env voids success
   below SIDE_VOID). The receiver then comes about 8 cm toward the centre, and the pour happens near
   y ~ 0 (receiver near +0.08, source body near -0.02) without the source crossing CENTER_BAND.
3. Five-finger envelope on BOTH hands, relative to the lift hold (task definition: all five fingers).
   - Before the lift reference exists, the posture term counts wrapped fingers out of ALL fingers
     (wrap_count / F, F = ctx.src_finger_force.shape[1]) instead of saturating at 3, so that the lift
     reference records a five-finger hold.
   - After the lift, the SOURCE keep gets the same fingers-kept factor as the receiver. For both hands a
     finger counts as still touching relative to its OWN force at the lift reference (ref share * ref
     total, e.g. at least half of it), not at an absolute 0.2 N floor. No absolute force target.

KEEP (unchanged)
- No prescribed tilt angle, angle band, angle target or angle cap; pouring is driven by the flow of
  beads. Windows and distances in ctx.cup_radius multiples.
- The iter_26 rules: need = min(requested - poured, beads still in the source); finish at the env
  tolerance with the body-based pre-tilt pose; spill per request-worth floored at OVER_DELTA_W; spout
  precision and drop height in aim; the receiver fingers-kept factor (made relative by change 3).
- The iter_24 lift-grasp coupling exactly as it is (hand pose, force share and total force AT LIFT).
- Centre-line term for the source side (CENTER_BAND); side_ok on the receiver; the stronger tremor and
  control charge on the receiver arm.
- Control penalties (action_rate, flips, tremor) stay ramped on adr_progress; do not relax any speed or
  control penalty, including still_rcv. tilt_rate is the rotation potential, not a control penalty.
- approach, grasp, lift, aim, pour_flow, overshoot, spill, finish, success, premature, cup_collision
  and hand_foreign as in iter_26.
- Budget: state income must stay below outcome income at success. Change 1 adds no state income,
  because holding a tilt earns 0.

WARM START: t2r_i26r ep800, not fresh. The policy has no pour habit to unlearn (0 poured, 0 spilled, 0
premature in all three traces), grasp-lift-carry is solid (both cups held at the end 0.98), and its
action std has not collapsed (0.32-0.93 per dimension, vs 0.58-0.85 at ep300). It is also the furthest
along the approach (spout 0.19 m). The three changes act on states it already visits: change 1 removes
a -0.16/step charge there (spout ~ 0 there, so the new term is ~ 0), change 2 moves the receiver anchor by 8 cm, and change 3 asks for the pinky
that the ep600 policy still had. ep600 has the better source pinky (0.99) but the largest backward lean
(48 deg), a source cup that drifted away late (y -0.18, spout 0.33 m), and receiver 5-finger contact
already at 0.31. A fresh start re-spends about 300 epochs on grasp-lift (the i26r milestone came at
ep300). Gate after 300 epochs (deterministic trace): peak tilt toward the receiver p50 > 20 deg, spout
to mouth at the end p50 < 0.12 m, receiver side margin p50 < 0.12 m, source 5-finger carry >= 0.8,
receiver >= 0.5. If the gate fails, restart fresh with iter_27.
