## I can see from the robot that
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

## Feedback for improvement
What worked and must be kept: the pre-latch discipline (open hand shaped on the worst joint, direct penalty on pre-latch
flexion), the continuous geometric readiness gate, the restraint penalties, and the tightened thumb-opposition test,
which finally taught the thumb to oppose instead of folding into the palm. The approach latch of 0.95 with cup tilt
under 0.15 degrees and arm speed under 6 rad/s is the best behaviour of the whole track. None of that should be traded
away.

The feedback for improvement is the pattern across the last three rounds: each time a term is added to fix the previous
round's dead end, the policy settles at that term's maximum and stops, because the new term reaches its maximum without
any contact. A flat open hand did it, then a loose claw, now a seated palm. The single rule that would stop this is that
**no term may sit near its maximum while contact is zero.**

1. The seating and palm-distance terms must not be payable on their own. They reached 89 % and 76 % of their maxima
   within 200 epochs and stayed there for 2,800 more, paying about 1.17 per step while every finger term together paid
   0.07. Either multiply them by how many digits are engaged, or cap them at a small fraction of their present value and
   move the freed budget into the contact terms. Position without grip is not progress and should not be priced as if
   it were.

2. Cutting the proximity share from 0.35 to 0.15 removed the on-ramp. Round 4, with the larger share, reached per-digit
   mean 0.142, 0.037 digits in contact and 0.156 flexion; round 5, with the smaller share, reached 0.031, 0.004 and
   0.138. The proximity ramp is what gets the fingers moving at all, so it should be restored to roughly its earlier
   strength - but bounded, so that all five digits at their best proximity are still worth strictly less than one digit
   actually touching.

3. The commanded flexion rose (0.190 to 0.325) while the measured flexion barely moved (0.125 to 0.138). The hand is
   being told to close and is not closing. Rewarding the commanded posture is therefore not reaching the fingers;
   reward what the joints actually do, and check the gap between command and measurement rather than paying for either
   alone.

4. The index and ring fingers have now been at exactly 0.000 for three consecutive rounds while the thumb, middle and
   pinky move. The recording shows why: the fingers are extended past the cup instead of curled around it, so they
   approach the cup along their length rather than closing onto it. The per-digit geometry should be measured from the
   fingertip towards the cup surface along the direction that finger actually closes in, so that a finger lying past
   the cup scores badly and a finger curling onto it scores well.

5. The palm-to-cup gap held at 0.05 m all round and never closed. Whatever pays for the palm should keep increasing all
   the way to contact, with its largest gradient in the final centimetre, in the same way the digit terms were meant to.

6. Lifting, carrying and holding were never reached for the fifth round running. Leave that part of the ladder alone.
