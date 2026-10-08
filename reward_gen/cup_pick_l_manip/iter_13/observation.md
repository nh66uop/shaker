Round 13 is the first round of the two-phase plan: both arms start from their own stage-1 policy (left `cp_l_a20`
e2450, envelope 0.55; right `cp_r_a20` e1800, envelope 0.82) and train the identical continuous reward on top. Left
`cp_l_m13` reached 963 epochs on vision-3090, right `cp_r_m13` 1,607 on the local 5090, both 2.4 hours, no Traceback,
no PhysX error, placement rejection 0.42.

**The left arm did what the round was for.** Lift episodes rose 0.13 (the starting policy) to 0.46, the airborne
fraction 0.06 to 0.31 and the mean rise 0.010 to 0.047 m. The grasp survived the reward change: after an initial dip to
0.29 it returned to 0.75, with links in contact 2.86 (2.83 at the start), ring contact up 0.20 to 0.52 and palm contact
0.34. Cup tilt is 6-7 degrees. Goal distance is unchanged at 0.250 m, so transport has not started and the wrench never
fired.

**The right arm lost its grasp in the first 200 epochs and never recovered.** Grasp episodes went 0.95 to 0.011 by
epoch 222 and have read 0.000 for the last 1,400 epochs; links in contact 3.65 to 0.05, thumb 0.74 to 0.04, index and
ring exactly 0. Lift is 0.000.

**What the right arm settled into is diagnostic.** Its `hand_curl` is 0.000 - the hand is fully open - while palm
contact is 0.19-0.28 and the palm-to-cup gap is the smallest of either arm at 0.023 m. It parks with the hand flat
against the cup and collects +0.195 per step from the new open-pose factor and the new palm-proximity term without ever
closing a finger. The left arm sits at `hand_curl` 0.31 and keeps the contact income as well.

**The two arms ran the identical file, so this is a property of the reward, not of one arm.** The only difference at
the start was the stage-1 pose: left ended stage 1 at `hand_curl` 0.32, right at 0.20, i.e. the right hand was already
flatter. The new open-pose factor pushes in that direction and, from the flatter start, the right arm slid into the
open-hand basin within 200 epochs. The same reward that teaches the left arm to lift makes the right arm let go.
