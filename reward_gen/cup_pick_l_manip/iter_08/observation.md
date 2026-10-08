Round 8 of the left-arm stage-2 (manip) track (`cp_l_m08`) ran 3,229 epochs in 4.6 hours, PPO-LSTM with 4,096
environments on the local RTX 5090, from round 7's epoch-1800 snapshot. The round answered the user's instruction that
the cup be lifted and carried upright: uprightness became a multiplier on the lift, rise, goal and hold income instead
of a small penalty, the lift income was capped at the height the goal needs, holding at the goal was raised to 9.0 and
the squeeze thresholds were raised. Placement rejection behaved as designed: 0.42 of resets resampled and none failed.

**The cup is now carried upright.** Cup tilt fell 28 degrees to 4.3-6.7 degrees and stayed there for the last 1,400
epochs. At 5 degrees the uprightness factor is about 0.95, against 0.15 at the 24 degrees where the round sat until
epoch 1400, so the policy now collects essentially all of the lift and goal income. The change was not gradual: tilt
held at 23-25 degrees from epoch 200 to 1400 and then fell to 10 degrees by epoch 1700 and to 5 degrees by epoch 2000.

**The grasp improved with the upright pose, and the ring finger finally touches.** Ring contact was exactly 0.000 in
every round of this track including the starting policy, and in this round it rose from 0.000 to 0.156 at epoch 2500
and 0.40 for the last 600 epochs. Links in contact rose 3.7 to 4.36. The other digits are at their best values of the
track (thumb 0.79, index 0.77, middle 0.76, pinky 0.77) with palm contact 0.64 and a palm gap of 0.039 m. Grasp
episodes reached 0.94 and lift 0.89. The drop penalty fell to -0.09.

**Success drove the tolerance down twice more.** Far-start success oscillated 0.16-0.55 and the mean successes per
episode reached 0.39, which shrank the goal tolerance 0.101 to 0.082 at about epoch 2300 and to 0.066 by epoch 3100 -
41 % tighter than the 0.1125 this track started from, though still far from the 0.015 m floor the task defines. Each
tightening cost success for a few hundred epochs (0.47 to 0.22 after the last one) and then recovered; grasp and lift
were unaffected.

**Holding at the goal is now the only blocked stage.** That term rose 0.000 (round 7) to 0.53-0.63, but its maximum is
9.0, so the policy earns about 6 % of it. The fraction of steps near the goal is 0.019-0.050 and the goal keypoint
distance flattened at 0.163-0.174 m, roughly 2.5 times the current tolerance. The pattern is unchanged from round 7:
the cup is carried towards the goal and successes are counted in passing, but the hand does not settle there.

**Two of my own calls in this round were wrong and are worth recording.** At epoch 1126 I read the flat 24-degree tilt
as a hard limit of the four-digit grasp; it was a plateau before a transition. At epoch 2539 I read the ring finger's
zero contact as likely structural; it started touching 700 epochs later once the cup was upright. Both were premature
readings of a flat curve.

**The recording (snapshot epoch 3200, 12 far-start environments, tolerance 0.066) shows an upright pick and carry that
never settles.** The hand reaches the cup by about 2.7 s and lifts it at about 4 s, and from then on the cup is held
visibly upright in the hand for the whole episode, gripped against the palm with the fingers around it. The hand keeps
moving it, up near the robot at 5-7 s, back down towards the table at 9-11 s and up again at 12-14 s, without coming to
rest anywhere. The cup is never tipped, put down or dropped.
Video: `our_source/fj_t2r_videos/cp_l_m08_0923_1235.mp4`.
