NEW TASK (round 25). The operator changed the goal: pour a REQUESTED AMOUNT, then bring the source cup
back upright and keep holding both cups. In the real setup the robot will pour into several cups in a
row, so "tip the cup over and leave it there" is no longer an acceptable ending. The env was changed
for this (commit c6708ec7) and the success rule is env-judged, not reward-judged:

  success = |poured_beads - pour_target_beads| <= 1
            AND source cup tilt <= 20 deg
            AND both cups still grasped
            AND (old safety conditions: spill <= 0.40, cup xy < 0.20, not nested, no premature-tilt
                 latch, no cup-cup collision, pour direction ok, receiver upright <= 20 deg)

  New ctx fields: pour_target_beads (how many beads to transfer this episode, sampled per episode),
  pour_target_frac (that count / active beads, also given to the actor as one observation channel),
  bead_active_count (how many beads exist in the source cup this episode).
  Observation is now 224 (actor) / 294 (critic); the policy therefore KNOWS the requested amount.

MEASURED, t2r_i24 best checkpoint, deterministic replay at ADR 30, 64 envs, 900 steps
(the run the operator accepted on video: palm on the upper side of the tilted cup, fingers supporting
from below, no tremor in the pouring arm):

  transfer success (>= half moved)      0.938
  env success (all conditions)          0.719
  in_target median                      1.00      spill median 0.00, p90 0.111
  peak source tilt median               116 deg
  grip drift from the lift reference    1.08 cm (20-50 deg) / 1.46 (50-90) / 1.56 (90-160)
  force-share difference                0.27 / 0.25 / 0.25
  force ratio vs lift                   2.40x / 2.56x / 2.46x
  -> the lift-grasp-keeping design of iter_24 WORKED and must be kept. i23 for comparison:
     3.6 cm drift, 0.51 share difference, 5.0x force.

  corr(fill level, peak tilt)          -0.05 at ADR 30, -0.19 at ADR 0
  -> NO fill-dependent tilt. The policy rotates to the same angle whatever the cup holds.
     Structural cause measured today with a dedicated probe: with 30 mm beads the fill measure
     saturates (12 active beads still read 0.94, 20 read 1.00), so "a less full cup" barely exists.

  source arm action flip rate           0.294      receiver arm 0.445
  source arm joint-speed RMS            0.252      receiver arm 0.341
  -> the RECEIVER arm trembles ~1.5x more than the pouring arm. The operator saw this on video.

  source cup y travel                   -0.162 -> +0.156 (peak +0.193)
  receiver cup y                        +0.161 -> +0.222
  -> the whole pour happens on the left side. The operator requires the work to stay near the
     centre line (y ~ 0) because the real policy will be driven from the head camera view.
     The receiver side already has side_margin / home_rcv terms; the source side has no such term.

  of 60 episodes that transferred beads, 14 were not env successes:
  premature-tilt latch 21 %, cup xy distance over the limit 14 %, receiver tilt was fine (6 deg).

BEAD SIZE — CHANGED TO 24 mm (measured today, spawn count still 20 so the physics load is the same).

Why the old 30 mm beads had to go: with 30 mm the fill measure saturates (12 active beads read 0.94,
20 read 1.00), so "a less full cup" never existed and the policy had nothing to adapt to.

A second defect was found and fixed while measuring (commit addace1c): the cup is tapered - the inner
radius is 26.7 mm at the bottom and 42.6 mm at the rim - but the bead spawn layout sized every ring
from the RIM radius, so bottom-layer beads were spawned outside the wall and PhysX ejected them while
they settled, with no hand touching the cup. Raising solver iterations (8/16 -> 32), the cup SDF
resolution (64 -> 128 -> 256) and the bead contact offset (0 -> 2 mm -> 5 mm) all changed nothing;
spawning inside the tapered wall fixed it completely.

  beads still in the cup after settling (32 envs, no hand contact):
    diameter   before fix   after fix
     24 mm       0.805        1.000
     27 mm       0.938        1.000
     30 mm       0.998        0.905   <- 20 x 30 mm no longer fit in a lattice inside this cup

  24 mm after the fix (16 envs, scripted pour):
    active beads   fill signal   first bead out   80 % out
        20            0.853          90.6 deg      95.9 deg
        14            0.688          97.6 deg     102.0 deg
         8            0.483          91.9 deg      96.2 deg
    27 mm at 20 active reads fill 1.00 again, so it does not solve the saturation.

  Applied in the ENVIRONMENT (this is where asset-specific numbers belong, and `ctx` carries them
  into the reward): bead_diameter_m 0.030 -> 0.024 (mass follows the density, 15.6 g -> 8.0 g),
  bead_active_range (6, 20) -> (8, 20) so one bead is at most 12.5 % of the request,
  premature_release_full_deg 72 -> 90 and premature_release_span_deg 34 -> 8.
  ★These numbers moved once already and will move again with another cup or bead. The previous
  reward hardcoded the old ones (a pour target at 72 deg and an over-tilt charge from 86 deg) and
  would now charge the policy for reaching the angle where beads actually leave. Read angles from
  `ctx`, or reward the outcome instead.

REQUIRED OF THE NEW REWARD (operator):
  1. Income for approaching pour_target_beads, penalty for going past it. Pouring everything must no
     longer be the safe choice.
  2. Pouring must be driven by the FLOW, not by a prescribed angle. The operator's rule for this
     round: "a reward must be general and simple, or neither the learning nor the policy
     generalises". So do not write a target tilt in degrees; pay for beads actually leaving the
     source cup and arriving in the receiver, and let the policy find the angle - which will differ
     with the amount left in the cup, and would differ again with another cup or another bead size.
  3. After the target amount is reached, bringing the cup back upright and holding both cups must pay.
  4. Keep the iter_24 lift-grasp coupling exactly as it is: the reference is the hand pose, force
     share and total force AT LIFT, never an absolute force target.
  5. The source side needs a centre-line term (work near y ~ 0), symmetric to the receiver side.
  6. Stronger tremor penalty on the RECEIVER arm than on the source arm.
  7. Control penalties stay ramped on adr_progress (fresh run starts with them off).
