Round 5 of the left-arm stage-2 (manip) track (`cp_l_m05`) ran 3,057 epochs in 4.0 hours, PPO-LSTM with 4,096
environments, from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`). Placement rejection behaved as
designed: 0.417 of resets resampled a placement and none failed. The round ended at the four-hour limit, with the
approach gate at 0.91-0.95 since epoch 1800 and the envelope latch at exactly 0.000 for the whole run.

**The approach is the best it has been in five rounds and everything stays safe.** The approach latch rises 0.577 to
0.926 by epoch 800, dips to 0.808, and holds 0.90-0.95 from epoch 1800 to the end. Cup tilt is 0.03-0.14 degrees, the
largest arm joint velocity per episode is 5.8 rad/s at the end, the no-touch condition holds at 0.99, and the hand never
approaches the table (lowest point 0.252-0.265 m).

**The two new terms worked as designed and immediately became a plateau.** Fine-scale seating reaches 0.634 in the first
200 epochs and sits at 0.697-0.708 (of a 0.80 maximum) for the remaining 2,800; palm-to-cup distance reaches 0.447
immediately and stays at 0.434-0.467 (of 0.60). Together they pay about 1.17 per step from the very first epochs, with
no dependence on the fingers. Every finger term added together is worth 0.07 at the end of the round. The palm-to-cup
gap sits at 0.048-0.058 m and does not close further.

**The thumb learned to oppose.** Thumb opposition rises 0.024 to 0.130 over the round, and the closing command rises
0.190 to 0.325, so the hand is commanding more flexion than in any previous round.

**But contact never happened.** Mean digits in contact is 0.004 at the end and never exceeded 0.009 in any 200-epoch
bucket. Per finger over the whole run: index 0.000, ring 0.000, middle 0.001, pinky 0.001, thumb 0.002. Mean contact
force per link is 0.008 N. Measured hand flexion rose only 0.125 to 0.138 despite the larger command, the per-digit mean
term reached 0.031 (of 1.50) and the two-weakest-digit term 0.002 (of 2.00). The envelope latch, lift, airborne fraction
and success are all exactly 0.000 and the tolerance never left 0.1125.

**Compared with round 4 this is a regression in grasping.** At a comparable epoch, round 4 had per-digit mean 0.142,
digits in contact 0.037 and hand flexion 0.156; round 5 has 0.031, 0.004 and 0.138.

**The recording (final weights, 12 far-start environments, tolerance 0.1125) shows the hand parked beside the cup with
the fingers close to straight.** It arrives at about 4 seconds, sets the palm facing the cup a visible distance away,
and extends the fingers past the cup rather than around it; the thumb sits up near the wrist. That configuration is held
without change to 15 seconds. The cup stays upright and untouched on the table for the whole episode.
