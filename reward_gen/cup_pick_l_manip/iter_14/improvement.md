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
