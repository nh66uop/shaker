Round 7 of the left-arm stage-2 (manip) track (`cp_l_m07`) ran 2,819 epochs in 4.0 hours, PPO-LSTM with 4,096
environments on the local RTX 5090, from `cup_pick_l_approach_e3400` (approach 0.90, envelope 0.81, lift 0.19). This is
the first round with the object-agnostic reward: no approach/envelope latch, no cylinder template, every term computed
from surface distances, contact forces and object pose. Placement rejection behaved as designed: 0.41 of resets
resampled and none failed. The round ended at the four-hour limit.

**This is the first round that lifts and succeeds.** Far-start lift rose 0.19 (the starting policy) to 0.51 by epoch
200 and 0.86-0.88 from epoch 600 on; the airborne fraction rose to 0.65-0.69. Far-start success went 0.00 to 0.19 by
epoch 1000, 0.38-0.43 by epoch 2600, and the mean successes per episode reached 0.39. That was enough for the
environment to tighten the tolerance for the first time in this track, 0.1125 to 0.101 at about epoch 2700.

**The grasp inherited from e3400 was kept, not undone.** The four digits that touch in the starting policy stay in
contact: thumb 0.75, index 0.75, middle 0.71, pinky 0.73, palm 0.54, about 3.7-3.9 links in contact, palm-to-cup gap
0.044-0.054 m. The ring finger is still exactly 0.000, as it was in e3400 and in every earlier round; the closing
command on its segment sits at 0.203 and never moves.

**The new surface inputs read as expected in training.** Palm surface distance 0.047-0.057 m while grasping, mean
minimum link surface distance 0.028-0.034 m, touching links 0.018 m (the 1.8 cm offset measured in the smoke test).

**Two things are stuck.** The hold-at-the-goal term is exactly 0.000 for all 2,819 epochs and the fraction of steps
near the goal never exceeds 0.045, while goal keypoint distance flattens at 0.216-0.230 m: the hand carries the cup
towards the goal and counts successes in passing, but does not come to rest there. And the cup is carried tilted:
tilt rose 12 degrees to 27-31 degrees and stayed, because the tilt penalty (-0.36) is small against the lift income
(+2.6).

**The largest penalty in the round is squeezing, -0.43 to -0.47 per step**, larger than the tilt penalty and about a
sixth of the lift income. The policy holds the cup harder than the term intends.

**Two transient regressions, both recovered or explained.** Between epochs 800 and 1400, while the carry to the goal
was being learned, contacts fell (digits 0.76 to 0.62, palm 0.49 to 0.32) and lift fell 0.86 to 0.56; by epoch 1600
everything was back (digits 0.75, palm 0.54, lift 0.87). After the tolerance tightened at epoch 2700, grasp fell 0.90
to 0.73 and lift 0.86 to 0.70 while success fell 0.43 to 0.28 - the task simply got harder; mean successes stayed 0.35.

**The environment's own latches are no longer meaningful as a funnel.** Approach latch 0.02-0.03 and envelope latch
0.01-0.02, because the policy no longer passes through the no-touch open-hand template the latch demands. Progress in
this track must be read from grasp / lift / success, not from the latches.

**The recording (snapshot epoch 1800, 12 far-start environments, tolerance 0.101) shows a real pick and carry.** The
hand reaches the cup by about 2.7 s, closes on it and lifts it clear of the table at about 4 s, then raises it high and
holds it up near the robot for the rest of the episode, from 5 s to 14 s. The cup is visibly tilted in the hand for the
whole carry, held against the palm by the thumb and three fingers, and it is never put down or dropped.
Video: `our_source/fj_t2r_videos/cp_l_m07_0923_0331.mp4`.
