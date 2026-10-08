ROUND 26 OPERATOR NOTES (09.25 evening). Facts only. The operator approved all five changes below.

TASK DEFINITION (operator, 09.25): grasp both cups with both hands and keep the envelope (all five
fingers on the cup, grip held), pour the REQUESTED amount (the target amount that is in the actor
observation), then bring the source cup back upright and return to the pose it had before tilting.

WHAT t2r_i25j DID (deterministic replay, ADR 0, 64 envs x 900 steps, ep1250 = best, ep1500 = degrading)
| metric | ep1250 | ep1500 |
|---|---|---|
| success / upright at the end / ever upright after pouring | 0 / 0 / 0 | 0 / 0 / 0 |
| requested +-1 bead at the end | 25% | 28% |
| mean error (poured - requested, beads) | -2.11 | -2.34 |
| episodes with spill / mean beads spilled | 80% / 4.38 | 98% / 6.86 |
| end tilt p50 (p10-p90) | 94 deg (85-110) | 128 deg (111-137) |
| extra tilt after the first spill p50 / share tilting >10 deg more | +6.5 deg / 27% | +26.4 deg / 94% |
| receiver hand 5-finger contact while carrying | 90.7% | 72.2% |
Source hand keeps 5-finger contact 94-99%. Beads bouncing out of the receiver: 0.3-0.5%.

WHY (verified against iter_25/compute_reward.py and the trace)
1. Spill keeps "want" on and tilt income keeps growing: spilled beads never count as poured, so the
   request can no longer be met; want stays 1 (78-83% of spilled episodes end with want on), and
   tilt_aimed pays more the further the cup turns. Training fed this back into deeper tilts (end tilt
   94 -> 128 deg) and more spill. At the end of an unreachable episode, holding the tilt earns about
   10.6-14.5 while standing the cup up earns at most about 5, so upright never happens (0/64).
2. Income cliff (ep1250): episodes with zero spill still stop 1-3 beads short, holding just below the
   pouring angle and collecting tilt income (8.3) instead of filling (3.5 after the request is met).
3. Not the aim width: 89-99% of spilled beads left the spout inside the receiver mouth radius. Spill
   grows with horizontal spout offset inside the mouth (0-1 cm: 34-46% spilled, 2-3 cm: 83-88%) and
   with drop height (spout > 10 cm above the mouth: 71-85%) and with tilt.
4. The actor does not observe poured or spilled beads (only the request and the fill level), so it
   cannot react to spill inside an episode; the reward must make spilling itself expensive.

REQUIRED CHANGES FOR ITER_26 (all approved; these five only)
1. Tie the rotation income to bead events: tilt income stops once the first bead has LEFT the source
   cup; after that only beads arriving in the receiver pay. No angle target, no angle band, no angle cap.
2. "want" must see what is still reachable: need = min(requested - poured, beads still in the source).
   When spill makes the request unreachable, want turns off and the finishing income opens.
3. Finishing opens at the environment's own success tolerance (|error| <= 1 bead), not only on an
   exact fill. Finishing pays for standing the source cup upright AND returning to the pre-tilt pose
   (the pose held just before tilting began), with both cups still held.
4. Make spill more expensive relative to the request, and give the aim term the spout's horizontal
   precision inside the mouth and the drop height (geometry in ctx.cup_radius multiples; no angle).
5. Reward keeping the RECEIVER hand's five-finger grip while carrying and pouring (grip quality
   relative to the hold at lift, never an absolute force target).

KEEP (operator rules from rounds 24-25, unchanged)
- A reward must be general and simple. No prescribed tilt angle in degrees; read angles from ctx or
  reward the outcome. Pouring is driven by the flow of beads.
- Income for approaching the requested amount, penalty for going past it.
- The iter_24 lift-grasp coupling exactly as it is (hand pose, force share and total force AT LIFT).
- Centre-line term for the source side (work near y ~ 0), stronger tremor penalty on the receiver arm.
- Control penalties stay ramped on adr_progress; do not relax any speed or control penalty.
- Budget: state income must stay below outcome income at success.

WARM START: t2r_i25j ep1250 (best before degradation). A 300-epoch gate (upright > 0, requested +-1 hit
rate rising, share of tilts > 120 deg falling) decides whether to continue or restart fresh.
