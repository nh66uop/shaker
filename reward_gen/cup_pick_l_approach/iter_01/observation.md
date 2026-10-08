Round 2 of the left-arm approach stage (`cp_l_a01`) trained the redesigned stage-1 reward from the e800 seed for 4284
epochs, 5.6 hours, and was stopped at the hour cap. This reward pays only for reaching the pre-grasp pose beside the cup
and holding it without touching the cup; it has no closing, lifting or goal term.

**The stage now stops at its own boundary.** The envelope latch, the lift fraction and the airborne fraction were exactly
0.000 for the whole round. The previous stage-1 reward (round 1, the `grasp_fj_stage/iter_08` lineage) reached an
envelope latch of 0.81, which is why the two policies had no boundary; that is gone.

**The cup is never touched and never moves.** Finger-cup contact and palm-cup contact stayed at 0.000 for the whole
round, and mean cup tilt is 0.008-0.035 degrees against 7 degrees under the previous reward. The recording shows the cup
standing exactly where it was placed at the end of the episode. The seed policy this round started from used to knock the
cup over (visible in its recording at 12.2 s); nothing like that appears now.

**The approach itself improved.** Far-start episodes complete the approach 0.93-0.94 of the time, up from 0.88 at the
seed. The orientation condition passes 1.000 of steps, the default-hand-pose condition 0.9998 and the no-contact
condition 0.9999. The two positioning conditions are the weakest and still rising slowly: the palm-plane gap passes
0.753 -> 0.778 and the along-the-fingers offset 0.757 -> 0.775 over the round. The palm-to-cup gap settles at 0.059 m.

**Holding is learned but is not yet complete.** The hold term rose 0.40 -> 0.52 -> 0.55 -> 0.565 and was still creeping up
when the round ended, so the pose is reached and held for a growing part of the episode rather than for all of it. The
total reward is flat at 2.03-2.12, which says the remaining head-room is in holding longer and in the two positioning
conditions, not anywhere else.

**In the recording** the hand leaves the raised start, travels to the cup, and comes to rest beside it with the fingers
open and the thumb extended on the near side, the cup between them. It stays there for the rest of the fifteen seconds.
The arm makes small corrections but does not drift into the cup. This is the intended stage-1 behaviour.

The hand never rests on the table (lowest hand point 0.24 m), placement rejection resampled 0.42 of resets with no
failures, and there were no physics errors.
