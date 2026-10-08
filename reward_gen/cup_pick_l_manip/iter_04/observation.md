Round 4 of the left-arm stage-2 (manip) track (`cp_l_m04`) ran 2,611 epochs in 3.6 hours, PPO-LSTM with 4,096
environments, from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`). Placement rejection behaved as
designed: 0.417 of resets resampled a placement and none failed. The round was ended because the approach gate held at
0.93-0.96 from epoch 2000 while the envelope latch stayed at exactly 0.000 for the whole run.

**Everything the previous round broke is now fixed.** The gate's default-pose condition passes 1.000 of steps for the
first 400 epochs and settles at 0.23-0.26 once the hand starts to flex, but unlike the previous round the approach latch
does not follow it down: it rises 0.582 to 0.891 and ends at 0.957. The no-touch condition holds at 0.957-0.999. Cup
tilt stays at 0.03-0.62 degrees against 12.3 degrees before, the largest arm joint velocity per episode peaks at 12.6
rad/s against 11,591 before, no runaway terminations occur, and the hand never reaches the table (lowest point 0.239 m).
The pre-latch flexion penalty is only -0.02 to -0.06 per step, which means the policy has learned to keep the hand open
until the latch is set and to flex only afterwards - the conditional behaviour the reward was asking for, learned even
though the latch is not in the observation.

**The closing terms rose and then saturated without any contact.** Over the round: leaving the open posture 0.001 to
0.144, per-digit mean 0.001 to 0.142, the two last digits 0.000 to 0.134, wrap 0.000 to 0.114, palm-distance 0.000 to
0.061. All five are flat over the last 800 epochs. Meanwhile mean digits in contact is 0.037, mean contact force per
link is 0.011 N, and per-finger contact is thumb 0.004, index 0.001, middle 0.003, ring 0.013, pinky 0.016. Palm contact
is 0.009 and the palm-to-cup gap is 0.052 m, slightly wider than the 0.038 m it reached at epoch 500. The envelope latch
is 0.000, the lift fraction 0.002 and success 0.000, with the tolerance never leaving 0.1125.

**Hand flexion went backwards.** It rises 0.126 to 0.164 by epoch 1200 and then falls steadily to 0.121, while the
closing command flattens at 0.288. The policy found a flexion level and stopped there.

**The recording (final weights, 12 far-start environments, tolerance 0.1125) shows why.** The hand travels to the cup
and stops beside it in a loose claw: the four fingers are bent at the knuckles, roughly matching the cup's curvature,
but the fingertips end a visible gap short of the cup wall and never advance. The thumb is folded back against the palm
rather than across the cup. The palm also stops short of the cup surface. That pose is reached at about 4 seconds and
held unchanged to 15 seconds, with the cup untouched and upright on the table for the whole episode.
