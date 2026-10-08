## I can see from the robot that
Round 14 restarted both arms from their stage-1 policies (left `cp_l_a20` e2450, right `cp_r_a20` e1800) with the
open-pose factor limited to the approach and palm seating tied to finger engagement. Left `cp_l_m14` reached 751 epochs
on vision-3090, right `cp_r_m14` 1,317 on the local 5090 (1.9 hours each). No Traceback, no PhysX error, placement
rejection 0.42.

**The round-13 flat-hand corner did not come back.** `hand_curl` stayed at 0.15-0.32 on both arms (0.000 on the right in
round 13), so the open-pose gating did its job.

**The right arm then peaked and collapsed.** By epoch 600-800 it held the grasp at 0.88, lifted in 0.43 of episodes
(airborne 0.28, mean rise 0.064 m), moved the cup towards the goal for the first time (goal distance 0.254 to 0.221 m),
reached far-start success 0.16 with 0.47 successes per episode, and spent up to 5 steps per episode inside the
disturbance radius. From epoch 800 everything fell: grasp 0.61 to 0.27, links 2.39 to 0.62, lift 0.30 to 0.11, success
0.16 to 0.03, and the palm-to-cup gap opened from 0.05 to 0.24-0.29 m - the hand moved away from the cup.

**The reward terms show why.** During the peak the drop penalty was the largest term of the whole reward, -0.75, -0.74,
-0.50 and -0.52 per step across four buckets, larger than lift (+0.33 to +0.41) and hold (+0.32 to +0.38) together
minus nothing else. The penalty is `P_DROP (8.0) x lifted x (1 - held)` with `lifted` the environment's persistent
height latch: once the cup has been lifted, every later step without a full hold is charged 8, until the episode ends.
One slip therefore costs for the rest of the episode. Lifting became a gamble with a large downside, and the policy
found the safe answer - stop lifting, stop grasping, move the hand away - after which the drop penalty fell to -0.10.
The left arm was on the same path when the round was stopped: drop penalty -0.54 to -0.96 per step against lift +0.21
and hold +0.17, total reward -0.77.

**The goal-gated wrench never fired, in this round or any round since iter_09.** The ADR level rose 0.03 to 1.00 (20
N/kg) between epochs 400 and 800 on the right arm, but `dr/wrench_fire_frac` and `dr/wrench_f_norm_max` are exactly
0.000 throughout. The environment gated firing on the parent's `self._lifted`, which this environment family never
updates (the kp-family reward path stores the height latch elsewhere), while the ADR level and the logs used the reward
context's `lifted`. The level climbed on dwell that was measured correctly, but no force was ever applied. This is fixed
in `manip_stages/cup_pick_env.py` (single gate function for firing, ADR and logging) with a contract test that fails if
the two ever diverge again. Everything reported as "under the disturbance" since iter_09 was in fact without one.

## Feedback for improvement
Keep: the open-pose factor limited to the approach and palm seating tied to finger engagement - the flat-hand corner of
round 13 did not return on either arm - and everything else round 14 inherited (object-agnostic surface terms, the
second half gated on how far the object has risen, lift 2.5, link-count scoring, first-success-only bonus).

Fix one term, and re-check the disturbance now that it actually works.

1. No penalty may be proportional to a persistent latch. The drop penalty was 8 x lifted x (1 - held) with `lifted` the
   episode-long height latch, so a single slip charged 8 per step for the rest of the episode. It became the largest
   term in the reward (-0.5 to -0.96 per step on both arms) and made lifting a losing bet: the right arm reached success
   0.16 and then unlearned grasping and moved its hand 0.29 m away from the cup; the left arm was heading the same way.
   Make the cost of losing the object short-lived and bounded: charge it while the object is actually falling or
   resting on the table after having been lifted, based on the current height and velocity, not on the latch; keep its
   size comparable to one step of the lift income, not a multiple of it. The main cost of dropping should be the income
   the policy stops earning. The set-down penalty has the same form (4 x lifted x on_table) and needs the same change.

2. The goal-gated wrench now fires. It was silently disabled from iter_09 through iter_14 by a gating bug in the
   environment (fixed). The next round is therefore the first one in which the policy meets a real disturbance at the
   goal; the ADR level restarts at 0 (5 N/kg). Watch `dr/wrench_fire_frac` and `dr/wrench_f_norm_max` to confirm it
   fires, and watch grasp and drop rate as the level rises.

3. Start points: left from its epoch-450 snapshot (the latest saved, grasp ~0.46, hand_curl 0.25), right from its
   epoch-800 snapshot (the peak before the collapse: grasp 0.88, lift 0.42, success 0.16).
