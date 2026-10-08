Round 2 of the left-arm stage-2 (manip) track (`cp_l_m02`) ran 1,530 epochs in 2.2 hours, PPO-LSTM with 4,096
environments, starting from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`, which approaches 0.94 of far
starts and never touches the cup). Placement rejection behaved as designed: 0.417 of resets resampled a placement and
none failed.

**The fixes to the approach latch worked and should be kept.** The approach gate now latches on 0.576 of episodes in the
first 200 epochs, 0.893 by epoch 400-600, 0.942 by epoch 1000-1200 and 0.949 at the end. The collapse of round 1 (0.49
falling to 0.06) did not recur: the no-touch condition holds at 0.957-0.982 for the whole round, cup tilt stays at
2.4-3.4 degrees, and the two new pre-latch penalties are barely charged (contact -0.003 to -0.021, disturbance -0.004 to
-0.011), which means the policy stopped pressing the palm into the cup before the latch. The palm-to-cup gap closes from
0.043 to 0.028 m and the hand never reaches the table (lowest hand point 0.236-0.245 m).

**Nothing behind the latch moved.** The envelope latch is exactly 0.000 in every one of the 1,530 epochs. Lift is 0.001,
the airborne fraction 0.000, success 0.000, and the tolerance never left its starting value of 0.1125.

**The fingers never touch the cup.** Per-finger contact over the round: index 0.000, middle 0.000 and ring 0.000 in every
200-epoch bucket; thumb 0.002 rising only to 0.018-0.041; pinky 0.019 falling to 0.000. Mean digits in contact is
0.022-0.043 across the whole round with no trend. Palm contact, by contrast, rises 0.289 to 0.620 to 0.667 and stays
there. The hand is also not closing: the closing command is flat at 0.263 to 0.331 over 1,530 epochs and the grasp
quality measure moves only 0.223 to 0.306.

**The reward decomposition shows why.** The terms that pay without any contact all rose - palm 0.103 to 0.232,
worst-digit surface proximity 0.231 to 0.306 (of a 0.60 maximum), surface wrap 0.182 to 0.252 - while every term that
requires real finger contact stayed dead for the entire round: per-digit mean 0.002-0.005, weakest digit exactly 0.000,
opposition exactly 0.000, closing drive 0.001, envelope latch 0.000, lift 0.000, goal 0.000. Total reward rose 1.83 to
2.11 entirely on the no-contact terms.

**The recording (final weights, 12 far-start environments, tolerance 0.1125) shows one pose held for the whole episode.**
The hand travels from the raised start to the cup and parks beside it with the flat of the palm against the cup's side.
All four fingers stay fully extended, pointing past the cup and down towards the table, and the thumb is folded back
against the palm instead of opposing the fingers. That pose is reached at about 4 seconds and is still held at 15
seconds. The cup is never enclosed, never moves and never leaves the table.
