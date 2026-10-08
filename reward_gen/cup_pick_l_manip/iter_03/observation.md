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
