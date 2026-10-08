## I can see from the robot that
Round 3 of the left-arm stage-2 (manip) track (`cp_l_m03`) ran 2,868 epochs in 4.5 hours, PPO-LSTM with 4,096
environments, again starting from the stage-1 approach policy (`cup_pick_l_approach_hold_e4280`). Placement rejection
behaved as designed: 0.416 of resets resampled a placement and none failed. There is no recording for this round: the
playback was started while the training process still held the GPU, the physics engine failed to launch its collision
kernels, and no frames were written. Everything below is read from the training metrics.

**The fingers closed for the first time in three rounds.** Mean digits in contact rose 0.02 (epochs 0-200) to 0.57
(400-600) to 1.15 (800-1000), dipped to 0.61 (1600-1800), then reached 1.59 (2200-2400). Per finger at the peak: thumb
0.605, middle 0.467, ring 0.468, pinky 0.128, index 0.011. The middle finger, which had been exactly 0.000 for two whole
rounds, first touched at epoch 1800 and reached 0.467. The weakest-digit term left zero for the first time (0.001 to
0.116). Contact force per link rose 0.003 to 0.43 N and the closing command rose 0.215 to 0.366.

**The envelope latched for the first time, briefly.** It is 0.000 up to epoch 2000, then 0.115 (2200-2400) and 0.152
(2400-2600), then falls back to 0.046. The lift fraction reached only 0.022 and success stayed 0.000 throughout, with
the tolerance never leaving 0.1125.

**The approach latch then collapsed.** Far-start approach holds 0.93-0.95 until epoch 1800 and then falls: 0.877
(2000-2200), 0.723, 0.539, 0.347, 0.373. Two of the six gate conditions explain it. The default-pose condition, which
requires EVERY movable finger joint to stay within 0.3 of its default angle, passes 0.997 of steps in the first 200
epochs and then 0.213 from epoch 400 onward - it broke as soon as the hand learned to flex, long before the collapse
became visible in the latch. The no-touch condition falls 0.979 to 0.376. The instantaneous pass rates of the other
conditions are also low late in the round (gap 0.22, along 0.29), so the latch depends on a narrow window early in the
episode, and once the hand starts closing before it arrives, that window never opens.

**Closing was done by pushing, not by wrapping.** Cup tilt rises 0.06 degrees to 3.4, then spikes to 12.3 degrees at
epochs 2200-2400 and settles at 4.0. The largest arm joint velocity per episode explodes over the same interval: 11.8,
then 22.9, then 1,901, 2,020, 11,591 rad/s, and abnormal-runaway terminations appear (0.002 of episodes). The envelope
numbers of that interval therefore come from a physically unstable state and cannot be taken at face value.

**The palm left the cup and never came back.** Palm contact stays between 0.07 and 0.16 for the whole round (it was
0.67 in the previous round), the palm reward term is 0.014-0.036, and the palm-to-cup gap grows 0.031 m to 0.082 m as
the fingers engage. Since the envelope latch needs the palm, the thumb and at least four digits together, the grasp was
converging to a fingertip pinch that cannot satisfy it.

**The index finger never participated**, staying between 0.002 and 0.011 for all 2,868 epochs.

## Feedback for improvement
What worked and must be kept: the flexion-gated proximity and wrap credit, the single continuous per-digit ladder from
"curled and near" to "touching", the term that pays for leaving the default open posture, and the thumb-opposition term.
Together they did what two previous rounds could not - the hand closes, the middle and ring fingers engage, and the
envelope latched for the first time. Do not go back to contact-only closing terms.

The feedback for improvement is that closing now destroys the gate that pays for closing.

1. Closing before the latch must cost more than it earns. Every closing term is gated on the approach latch, but there
   is nothing that discourages flexing the hand before that latch, and the only counterweight - holding the default open
   posture before the latch - is worth about one twenty-fifth of what closing pays afterwards. The optimal policy under
   that arrangement is to close as early as possible, which is exactly what happened. Holding the open posture until the
   latch has to be worth roughly as much as closing is afterwards, and flexing the fingers before the latch should be
   penalised directly on the joint angles, not only when it produces contact.

2. The gate's pose condition is a maximum, not an average. It requires every movable finger joint to stay within 0.3 of
   its default angle, so a single finger that flexes past that threshold breaks the whole approach gate. Measured pass
   rate fell from 0.997 to 0.213 at epoch 400 and never recovered. The pre-latch posture term should therefore be shaped
   on the worst joint, matching what the gate actually tests, instead of on an average over the hand.

3. The reward regime switches on a state the policy cannot observe. The observation vector contains joint angles and
   velocities, palm pose, fingertip positions relative to the palm, the commanded hand state, cup and goal keypoints and
   the previous action - it contains neither the approach latch, nor the envelope latch, nor any contact force. The
   policy has to infer from its own memory that its reward regime has changed. Terms that turn on and off hard at the
   latch are therefore weak signals; prefer terms whose value changes smoothly with the same geometric quantities the
   gate measures, so that the gradient points towards the gate rather than depending on knowing which side of it the
   policy is on.

4. The grasp is being made by pushing the cup. Cup tilt reached 12 degrees and the largest arm joint velocity per
   episode reached 11,591 rad/s, with runaway terminations appearing, exactly in the interval where the envelope first
   latched. Closing has to be restrained: excessive contact force, cup displacement and cup rotation while closing
   should cost clearly more than the extra contact they buy, so that the fingers settle onto the cup rather than drive
   through it.

5. The palm must stay on the cup while the fingers close. Palm contact never rose above 0.16 and the palm-to-cup gap
   grew from 0.031 m to 0.082 m as the fingers engaged, so the grasp converged to a fingertip pinch. The envelope needs
   the palm, the thumb and four digits at once. Keeping the palm against the cup should be paid for continuously
   throughout closing, using the distance itself rather than only the binary contact.

6. The index finger still never participates (0.002 to 0.011 over the whole round) while the middle and ring fingers now
   do. Whatever asymmetry keeps it out should be addressed by making the per-digit score of the two digits that arrive
   last worth clearly more than the ones that arrive first.
