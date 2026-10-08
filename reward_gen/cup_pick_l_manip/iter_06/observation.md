Round 6 of the left-arm stage-2 (manip) track (`cp_l_m06b`) ran 2,074 epochs in 3.1 hours, PPO-LSTM with 4,096
environments on the local RTX 5090, from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`). Placement
rejection behaved as designed: 0.42 of resets resampled a placement and none failed. The round was stopped early by the
user after reviewing the latch criteria (see below), not by the four-hour rule. The envelope latch was exactly 0.000 for
the whole run.

**For the first time in six rounds the hand made sustained contact with the cup.** Links in contact rose 0.11 to 0.93 by
epoch 600 and ended at 0.91; the per-digit contact term rose 0.06 to 0.52. The iter_06 principle ("no term may sit near
its maximum while contact is zero") worked: the zero-contact plateaus of rounds 2-5 did not form.

**But only two digits ever touched: the thumb and the pinky.** Over the last 400 epochs: thumb 0.39-0.40, pinky
0.49-0.51, index 0.001, middle 0.001, ring 0.002-0.003. Index, middle and ring never exceeded 0.003 in any 200-epoch
bucket. Palm contact was 0.05 early and 0.007-0.017 afterwards. The envelope needs the palm, the thumb and four digits,
so it was never close.

**The contact came before the approach latch, and the approach collapsed.** The approach latch fell 0.63 to 0.16 by
epoch 1200 and recovered only to 0.31-0.35. Per condition, from the first bucket to the end: along 0.61 to 0.17, gap 0.62
to 0.39, no-touch 0.91 to 0.41, orientation 1.00 to 0.72, and from epoch 1200 the open-hand pose condition also fell,
0.97 to 0.45-0.52. The pre-latch shove penalty grew to -0.39 per step but stayed smaller than the contact reward
(+0.52), so touching the cup with the thumb and the pinky from beside it, before reaching the C pose, became the new
plateau. Travel (0.40) and the approach-latch term (1.07) were still paid alongside it.

**Everything else stayed safe and did not move.** Palm-to-cup gap 0.049-0.065 m, lowest hand point 0.24-0.25 m (no table
contact), cup tilt 1.7-3.3 degrees, lift 0.000-0.003, airborne fraction 0, success 0, tolerance 0.1125 throughout.

**Why the round was stopped: the stage boundary itself is the problem.** Both latches the reward is keyed on are
hand-written, cup-specific templates in the environment (`grasp_fj_t2r/grasp_gates.py`). The approach latch requires a
C pose defined from the cylinder radius and axis (palm plane -1 to +2 cm from the cup side, cup axis within a 25 mm
window ahead of the palm, open hand within 0.3, no finger touching, start orientation within 45 degrees). Stage 1 was
trained to satisfy exactly this (hold_e4280, approach 0.94), and rounds 2-5 kept the approach at 0.91-0.96 while the
envelope stayed at 0. Standing in that template is not a step towards grasping. Neither latch is in the observation,
and once set they stay set even if the hand leaves, so the reward regime they switch cannot be inferred by the policy. The
user's judgement: once the hand is close enough it should learn to close the fingers into a good grasp naturally, and a
cup-shaped template cannot extend to other objects. The same latches were reached by `cup_pick_l_approach_e3400`
(envelope 0.81, thumb/index/middle/pinky and palm 0.54), but that policy learned approach and grasp together in one
reward and never went through a no-contact stage.

**The recording (snapshot epoch 1800, 12 far-start environments, tolerance 0.1125) shows the hand pressing the cup's
lower side with straight fingers.** The hand travels in by about 4 seconds, but instead of setting the palm against the
cup it stops low and in front of it with the fingers extended forward, so the fingertips and the thumb poke the lower
part of the cup from the side. The fingers never curl around the cup and the palm stays a visible distance away. That
configuration is held with little change from 5 to 14 seconds; the cup stays upright on the table and is never lifted.
Video: `our_source/fj_t2r_videos/cp_l_m06b_0922_2302.mp4`.
