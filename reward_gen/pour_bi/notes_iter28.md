ROUND 28 NOTES (09.26, diagnosis of t2r_i27 = iter_27 c1r trained WARM from t2r_i26r ep800). Facts only. The
three changes below are the only changes for this round; everything else in iter_27 stays as it is.

TASK DEFINITION (operator, 09.25, unchanged): grasp both cups with both hands and keep the envelope (all
five fingers on the cup, grip held), pour the REQUESTED amount (the target amount that is in the actor
observation), then bring the source cup back upright and return to the pose it had before tilting.

WHAT t2r_i27 DID (deterministic replay, ADR 0, 64 envs x 900 steps; 10 active beads p50, request 5.5 p50;
premature_tilt_limit 73.6 deg p50; r = ctx.cup_radius = 0.041 m)
| metric | ep300 | ep600 | ep800 |
|---|---|---|---|
| success / any bead poured / any bead spilled | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| both cups held at the end | 0.94 | 0.73 | 0.89 |
| episode peak TOTAL tilt p50 (p10-p90) | 42 (38-53) deg | 55 (45-65) deg | 56 (53-58) deg |
| peak total tilt / premature_tilt_limit p50; episodes with peak >= 0.7 of the limit | 0.57; 11% | 0.75; 57% | 0.76; 92% |
| episode peak tilt TOWARD the receiver p50 (p90) | 21 (28) deg | 34 (46) deg | 47 (52) deg |
| cup axis lean OUT of the pour plane (d, z) while carrying p50 | 33 deg | 35 deg | 29 deg |
| source tilt when the cup first passes 3 cm of lift p50 | 11.5 deg | 24 deg | 32 deg |
| source lift reference recorded (episodes) | 64 / 64 | 10 / 63 | 4 / 64 |
| spout to receiver mouth, horizontal: at the tilt peak / end / episode minimum (p50) | 0.112 / 0.086 / 0.069 m | 0.134 / 0.157 / 0.092 m | 0.105 / 0.099 / 0.071 m |
| source cup body to receiver mouth at the end p50 | 0.119 m (2.9 r) | 0.216 m (5.3 r) | 0.180 m (4.4 r) |
| source y / receiver y at the end p50 | +0.029 / +0.137 | -0.105 / +0.098 | -0.084 / +0.084 |
| source 5-finger contact while carrying (1 N) | 0.48 | 0.69 | 0.86 |
| receiver 5-finger contact while carrying (1 N) / 3 or more fingers | 0.26 / 0.93 | 0.05 / 0.74 | 0.13 / 0.87 |
| receiver fingers loaded at its lift reference (share > 0.02) mean / all five > 1 N at lift | 4.2 / 0.06 | 3.9 / 0.03 | 3.9 / 0.02 |
| receiver lift-reference total force p50 | 11.3 N | 9.1 N | 7.9 N |
| tilt_rate per episode, paid / charged p50 | 0.0005 / -0.0007 | 0.0000 / -0.0000 | 0.0013 / -0.0014 |
| premature per episode p50 (p10) | 0.0 (-0.4) | -0.5 (-34) | -3.7 (-22) |
Tensorboard (21-epoch windows at e300 / e500 / e600 / e700 / e810): diag/grip_src_ref_valid 0.86 / 0.86 /
0.41 / 0.04 / 0.04 (receiver 0.88-0.92 throughout); posture_src 0.84 / 0.75 / 1.42 / 2.06 / 2.10; aim 0.40 /
0.47 / 0.47 / 0.55 / 0.68; premature -0.001 / -0.001 / -0.007 / -0.006 / -0.004; tilt_rate -0.0001 to
-0.0004; cup_collision -0.004 / -0.005 / -0.009 / -0.008 / -0.014; episode peak tilt (mid fill) 42 / 46 /
51 / 54 / 53.5 deg; task/pour_lip_dist 0.13 / 0.15 / 0.16 / 0.15 / 0.16 m; pour_flow, finish, in_target 0
throughout; adr_progress 0. A dip every ~100 epochs (e590, e689, e801) recovers within 2 epochs.

WHY (verified by recomputing the iter_27 terms offline, float64, on the trace states; counterfactual poses
move the source cup and hand rigidly, forces held fixed)
1. The rotation stops at the premature ramp, not for lack of room. premature charges from PRE_LO = 0.7 of
   premature_tilt_limit on the TOTAL tilt, times (1 - exp(-AIM_K * spout offset)). At ep800, 92% of the
   episode peaks sit at 0.72-0.79 of the limit. premature per step by total tilt (ep800): 50-55 deg -0.01,
   55-60 deg -0.15, 60-70 deg -0.59 (ep600: -0.02 / -0.23 / -0.78, 70-90 deg -1.74). Arm joints stay >= 0.5
   rad from their limits at the plateau (p10, ep600/ep800). With this grasp the palm PITCH slot is the one
   that turns the cup toward the receiver (+1.06 deg per deg). At ep800 the pitch command sits at its +45 deg
   delta-box edge in 91% of the plateau steps (ep600: 8%, same 0.75 stall ratio). It is not a hard limit.
   With the grasp held in the palm, palm orientations inside the delta box reach >= 93.6 deg toward the
   receiver (out-of-plane < 15 deg) in 99-100% of the plateau states, through yaw and roll.
2. What 10-40 deg more toward the receiver would earn or cost, per step, at the ep800 plateau states (pivot
   at the cup origin .. at the palm): +10 deg: premature -0.31..-0.33, aim +0.16..+0.18, net -0.17..-0.21;
   +20 deg: net -0.64..-1.01; +30 deg: net -0.98..-1.93 and 51-62% of the states cross the env latch
   (tilt > limit while the spout is more than 8.1 cm = 2.0 r from the mouth: success void, -LATCH_W every
   step to the end); +40 deg: net -1.36..-2.42, latch 83-97%. tilt_rate pays 0 for all of them.
   Speed and control charges are negligible here (action_rate -0.005/step, tremor 0 at adr_progress 0) and
   the grip terms fall only slightly with tilt (posture_src -0.04 per 10 deg within an episode above 40 deg).
3. The rotation help never reaches the states the policy visits. Phi = spout * lean with spout = fine *
   height and fine = exp(-(offset / (0.5 r))^2): at the plateau the spout is 0.087-0.137 m away (2.1-3.3 r,
   ep800 p10-p90), so Phi <= 1e-7 (p90, median 6e-11) and every move there pays and costs ~0.
4. Why no bead: a bead leaves only above about 90 + 8 * (1 - fill * in_source) deg of total tilt (93.6 deg
   at fill 0.55). The wall is at 52-58 deg, and at the wall the spout is outside the mouth (r) and mostly
   outside the latch radius (2.0 r), so going on would latch premature and put any bead on the table
   (spill at least -120 / n_active per bead).
5. The source dodges its lift reference. The env records the lift reference only if the cup is tilted less
   than 20 deg when it first rises 3 cm. From e500 to e700 the policy learned to tilt the source cup before
   that (tilted > 20 deg before 3 cm: 28% / 84% / 100% of episodes at ep300 / 600 / 800), so the reference is
   never recorded. Then posture_src pays the PRE-lift branch POSTURE_W_SRC * q_src for the whole carry (2.2 /
   2.7 per step at ep600 / ep800, max 3.0) instead of the lift-relative keep (<= KEEP_W_SRC = 1.5; 0.95-1.65
   where the reference exists), and keep_src = 1 by default, so grip_gate = 1 and kept_src = 1. The iter_24
   coupling and the iter_27 fingers-kept factor are switched off for the source. This tilt at lift is 22-31
   deg OUT of the pour plane (toward the robot, only 7 deg toward the receiver), and the out-of-plane lean
   stays during the carry (30-37 deg in episodes without a reference, 15-29 deg with one). Because premature
   uses the total tilt, this lean uses up most of the budget below 0.7 of the limit.
6. The pour pose is reachable and worth a lot, but the path pays almost nothing. At the ep800 plateau states
   the pour pose (lean toward d at the release angle, spout 1 cm over the mouth centre) earns +3.7 per step
   in aim alone. It needs the source body about 0.10 m (2.4 r) from the mouth, 4-10 cm closer and 4-8 cm
   higher than now (p10-p90). Source y then is -0.01..+0.02 (on the centre line) and the cup centres are 0.15-0.18 m
   apart (not nested). Stage by stage, cumulative per step: (a) turn the existing lean into the pour plane at
   the same total tilt +0.35 (no premature change); (b) raise 5 cm -0.13 (aim drop factor; lift_src is
   already saturated at h = LIFT_SRC_H); (c) approach to 0.104 m +1.43; (d) tilt to the limit with the spout
   centred +2.72; (e) to the release angle +3.66. Under iter_27, tilt_rate pays ~0 until the spout is within
   about 1 r of the mouth centre (Phi 2.6e-4 at 0.057 m after (a), 0.05 at 0.029 m). At the current height the source index fingertip is the closest point to the receiver and sits at
   its rim height; moving 4 / 6 cm closer without raising puts a source fingertip within 1 cm of the
   receiver wall below its rim in 21% / 46% of the states (outer wall radius assumed 4.5 cm), after a 5 cm
   raise in 0-4%.
7. The receiver five-finger loss is not visible to its term. The receiver lift reference is recorded in
   every episode, but the lift hold became lighter and uses fewer fingers (total 11.3 -> 7.9 N; middle
   finger > 1 N at lift 0.83 -> 0.19; pinky > 1 N at lift 0.03-0.11). A finger counts as "on at lift" at a
   force share > 0.02 (about 0.2 N) and as kept at half of its own lift force, and kept_rcv divides by the
   number of fingers on at lift. So kept_rcv is 0.96 (ep300), 0.79 (ep600), 0.96 (ep800) while the 1 N
   five-finger carry is 0.26 / 0.05 / 0.13. The pre-lift window is short (36-41 steps from grasp to the
   reference), and there each wrapped receiver finger pays 0.12 per step. A link to the receiver moving
   toward the centre line is not supported (correlation of its displacement from spawn with its loaded
   finger count -0.32 / +0.21 / -0.14).

REQUIRED CHANGES FOR ITER_28 (these three only)
1. Rotation help that reaches the pour approach. This is the pour-onset blocker. Keep tilt_rate as it is in
   iter_27: the exact FULL rate of Phi (central difference, both cups moved ahead and back by TILT_TAU,
   mouth, d, theta and lip rebuilt), rate-free pay gates (aim_gate) outside Phi, pay capped at TILT_PAY_CAP
   per step, negative rate charged in full and uncapped, TILT_W, and the latches (settled, 1 - released,
   want). Change only the geometry inside Phi:
   - Phi = latches * reach * over_rim * lean, lean = max(src_cup_up . d, 0) as now.
   - reach = exp(-(spout offset / (REACH_R * cup_r))^2) with REACH_R = 2.0. That is the env's premature latch
     radius (8.1 cm = 2.0 cup radii), in place of the 0.5 r precision width.
   - over_rim = the iter_26 factor for a spout below the receiver rim (BELOW_R). Do NOT put the drop factor in
     Phi. The drop height stays in aim, where it already is, so raising the source above the receiver rim
     before tilting is not charged by the rate.
   Effect on the ep800 plateau states (one-shot TILT_W * dPhi, gates 1): Phi 0.12 at the plateau (now
   6e-11); turning the lean into the pour plane +1.7, raising 0, approaching +1.5, tilting to the limit with
   the spout centred +0.85, to the release angle +0.19. Holding any pose still earns 0, and a tilt made with
   the spout 40 cm away still pays and costs ~0 (reach e^-24). premature and the env latch stay as they are,
   so tilting far from the mouth is still charged.
   - Check deterministically, as iter_27/check_tilt_rate.py did (adapt its Phi to this one): (a) every closed
     cycle sums to <= 0, including spout in/out at a fixed lean, jiggle around the (now wider) spout peak with
     all pay gates 1, and cycles that toggle the pay gates or ramp the latches; (b) a tilt made while the
     spout is far pays ~0 and costs ~0; (c) a lean away from or sideways to the receiver earns nothing and
     holding earns 0.
2. No grip income from dodging the lift reference (both hands). The pre-lift posture branch pays only
   while the cup is still at the table: multiply it by (1 - off), with off the existing lift-off ramp
   step((h - 0.01) / 0.04). A cup that is carried without a lift reference counts as keep = 0 wherever keep
   is used outside _grip_keep: keep_eff = valid * keep + (1 - valid) * (1 - off), used in grip_gate, in the
   finish factor and in the posture keep branch. _grip_keep itself stays byte-identical (iter_24 coupling).
   An upright lift keeps the pre-lift income until the env records the reference at 3 cm. A lift tilted past
   the env's record condition loses it by 5 cm and carries with posture 0 and grip_gate 0.3. On the trace
   states (changes 2 and 3 together) this costs the dodging carry steps 2.8 (ep600) / 3.6 (ep800) per step,
   and the ep300 carry steps, where the reference was recorded at lift, 0.18 per step.
3. Five-finger envelope counted out of ALL fingers after the lift (both hands; the queued receiver grip
   intent). In _fingers_kept divide by F = finger_force.shape[1], not by the number of fingers on at lift. A
   finger that was not on at the lift reference (share <= FINGER_ON_SHARE) counts as not kept. Each finger
   still counts relative to its OWN lift force (KEEP_FORCE_FRAC of ref share * ref total). There is no
   absolute force floor and no force target. A 4-finger lift then keeps at most 0.8: about -0.2 to -0.3 per
   step for the whole carry, which pays for loading all five fingers at lift. On the trace states (changes 2
   and 3 together): posture_rcv -0.14 (ep300) to -0.26 (ep800) per step on average.

KEEP (unchanged)
- No prescribed tilt angle, angle band, angle target or angle cap; pouring is driven by the flow of beads.
  Windows and distances in ctx.cup_radius multiples. Work near the y ~ 0 centre line.
- premature exactly as it is (ramp from PRE_LO of premature_tilt_limit, the lip factor, LATCH_W), aim
  exactly as it is (including the spout precision fine, the drop factor, DROP_FLOOR and aim_gate).
- The iter_27 receiver home anchor at side margin SIDE_OK, side_ok, still_rcv, CENTER_BAND on the source.
- The iter_27 pre-lift posture out of all fingers (wrap_count / F), and _fingers_kept relative to each
  finger's own lift force (only its denominator changes, change 3).
- The iter_24 lift-grasp coupling exactly as it is (hand pose, force share and total force AT LIFT).
- The iter_26 rules: need = min(requested - poured, beads still in the source); finish at the env tolerance
  with the body-based pre-tilt pose; spill per request-worth floored at OVER_DELTA_W.
- Control penalties (action_rate, flips, tremor) stay ramped on adr_progress; do not relax any speed or
  control penalty, including still_rcv and the stronger receiver-arm charge.
- approach, grasp, lift, pour_flow, overshoot, spill, finish, success, cup_collision and hand_foreign as in
  iter_27.
- Budget: state income must stay below outcome income at success. Change 1 adds no state income (holding
  earns 0). Change 2 removes up to 3.0 per step of posture income. Change 3 can only lower posture. At
  success: finish + success = 22 per step vs state income <= about 13 (aim fades with want).

WARM START: t2r_i27 ep300, not ep800/850 and not fresh.
- ep800/850 dodge the source reference in 100% of episodes, and change 2 charges exactly that lift (-3.6 per
  step on its carry states). A warm policy keeps its lift-time grasp habit and its lift reference, so ep800
  would first have to unlearn a tilted lift.
- ep300 records the source reference in 64/64 episodes (tilt at 3 cm 11.5 deg). It has no pour habit (0
  poured, 0 spilled), holds both cups at the end (0.94), has the closest approach of the three traces (body
  2.9 r, spout 8.6 cm at the end), the best receiver five-finger carry (0.26), and it predates the out-of-plane
  lift tilt. Its receiver lift hold loads 4.2 of 5 fingers (pinky share > 0.02 in 39%), so change 3 charges
  it from the first epoch and its pre-lift grasp has to add the missing fingers.
- ep500 had the reference in training (diag/grip_src_ref_valid 0.86) but has no deterministic trace.
- A fresh start re-spends about 300 epochs on grasp-lift.
Gate after 300 epochs (tensorboard + deterministic trace): diag/grip_src_ref_valid >= 0.8; source tilt at 3 cm
lift p50 < 20 deg; out-of-plane lean at the tilt peak p50 < 15 deg; spout offset at the tilt peak p50 < 2 r;
peak tilt toward the receiver p50 > 50 deg or a first bead in; receiver fingers on at lift >= 4.5 of 5 and
receiver 5-finger carry >= 0.5. If the gate fails, restart fresh with iter_28.
