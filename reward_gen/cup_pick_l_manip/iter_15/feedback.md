## I can see from the robot that
Round 15 ran left `cp_l_m15` to 922 epochs on vision-3090 (from the round-14 epoch-450 snapshot) and right `cp_r_m15` to
817 epochs on the local 5090 (from the round-14 epoch-800 snapshot). No Traceback, no PhysX error, no collapse.

**Both arms now lift, carry and hold under the disturbance.** Last five buckets: left grasp 0.84-0.89, lift 0.82-0.88,
far-start success 0.74-0.83, 1.42 successes per episode; right grasp 0.94-0.97, lift 0.91-0.96, far-start success
0.68-0.75, 1.25 successes per episode. The goal-gated wrench fires (gate fraction of steps 0.47 left, 0.62 right); on
the left the ADR level reached 1.0 (20 N/kg) at epoch 466 and stayed there to 922 with grasp, lift and success intact;
the right held 20 N/kg for 600 epochs (grasp 0.97, lift 0.95, far-start success 0.83 at its best).
The round-15 drop/set-down change did what it was meant to: no arm unlearned lifting this time.

**The grasp both arms converged on is a four-digit grasp, not an envelope.** Training tags over the whole right run:
`contact/finger_index` 0.000 (index never touches), thumb/middle/ring/pinky 0.77-0.82; at success thumb 1.00,
middle 0.99-1.00, ring 0.95-0.97, pinky 0.99, index 0.000, 3.94 fingers and 5.3-5.4 of the 15 links, palm 0.64-0.75
and falling over the last buckets. The environment's own five-digit envelope measure `stage/envelope_ep` is 0.000 for
the entire right run. Left: index 0.10-0.11 (0.09-0.11 at success), 3.93 fingers, 4.8 links, palm at success 0.91-0.93,
`stage/envelope_ep` 0.51-0.54.

A deterministic probe of the epoch-800 checkpoints (256 envs, 8 object species, 430 complete episodes on the right)
confirms it per species: the right lifts every species (0.89-0.95) with thumb + middle + ring + pinky against the palm
and the index at 0.00 contact on all eight, its links 31-44 mm off the surface; five-digit envelope 0.000 of episodes.
Palm contact while lifted falls with object size, 0.97 on the small cup to 0.32 on the largest, so the largest object
is carried by digits alone. The left forms a five-digit envelope in 0.11-0.39 of episodes depending on species.

**The reward makes the four-digit grasp optimal, and the arithmetic is plain.** Every downstream income - lift 2.10 +
rise 0.40, transport 1.60 + 0.90, hold 10.0, about 15 per step - is multiplied by `held`, and `held` saturates at four
touching digits (`digit_hold = (n_touch - 2) / 2`). The fifth digit is paid only inside the 1.2 grasp budget: the top
rung of the sorted count ladder (0.40 x 0.20 = 0.08) plus one fifth of the per-digit link mean (0.30 / 5 = 0.06), at most
0.14 per step, under 1 % of what four digits already unlock. Deeper wrap (more links per digit) likewise only moves the
1.2 budget. So the policy has nothing to gain from the fifth digit or from a deeper wrap, and the wrench at the goal
is survived with four digits, so nothing pushes it there either.

**Left-only early terminations are an environment box, not the reward.** In the same probe the left ends 0.11-0.24 of
episodes per species with `out_xy` (right 0.00-0.04) and tips 0.08-0.23 (right 0.06-0.13). The left mirror cfg keeps
the right arm's `object_out_y` box (-0.60, 0.25) while the left object spawns at y 0.00-0.30 and its goals reach
y 0.35, so a correctly carried object leaves the box. That is a cfg fix outside this reward.

## Feedback for improvement
Keep round 15 whole: it is the first round in which both arms lift, carry, settle at the goal and survive the
disturbance, and none of its terms collapsed. Every structural rule (a)-(e) stays, the budgets stay, the falling-only
drop and set-down costs stay, the open-pose factor limited to the approach stays, palm seating tied to engagement stays.

Change one thing, structurally: what the reward calls a real hold must keep improving until the hand forms a full
envelope, and the whole downstream income must scale with it.

1. Define ONE smooth envelope quality in [0, 1], computed once from quantities that are symmetric across the digits and
   object-agnostic: the sorted digit-count ladder extended so the fifth digit is a real step, the mean number of links
   per digit (wrap depth), palm contact, and thumb/palm opposition. No term may name a digit. Nothing that says
   "index", no per-finger weight, no finger-specific threshold: the policy must learn that the fifth digit and the
   deeper wrap are worth having because the SAME multiplier that pays for lifting and holding grows with them.

2. Multiply the lift, rise, transport, progress, hold-at-goal and success incomes by that envelope quality (with a
   floor, so a four-digit grasp still earns and still lifts). Target values: a four-digit wrap with the palm seated
   earns about 0.65-0.75 of the downstream income; five digits with two or more links each, palm and opposition earn
   1.0; a five-fingertip pinch without palm or wrap must not beat a deep four-digit wrap. The missing fifth digit then
   costs the policy roughly a quarter to a third of everything it earns while carrying and holding, on every step -
   several units per step instead of the 0.14 it costs today.

3. Keep `held` as it is (saturating at four digits, plain force) for everything that is a GATE rather than an income:
   the reach fade `(1 - held)`, the drop, stall, drift and descend costs. Those must not weaken when the envelope
   quality plateaus, and the disturbance must not be able to dissolve a real four-digit hold into penalties. Measure
   the envelope quality on the same soft contact forces, not on geometry alone.

4. Rule (b) still binds: with four digits the policy must still clearly prefer lifting and carrying to anything
   available on the table. Check the ceilings after the change: four-digit downstream income must remain far above
   the 1.2 grasp budget plus the decaying on-table income.

5. Start points are a user decision for the next round: continue from the round-15 policies (left e922, right e800,
   both already lifting with four digits) or restart from the stage-1 policies. The measure of success for this round
   is `contact/finger_index` rising from 0.000 (right) / 0.10 (left) and `contact/links_touching_at_success` rising
   from 5.4 / 4.8 while grasp, lift and success hold; `stage/envelope_ep` is the environment's own five-digit measure.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.0815, 0.282, 0.293, 0.233, 0.171, 0.00439, 0, 0.000488, 0]  min 0 · mean 0.103 · max 0.351
contact/finger_index_at_success: [-1, 0.0243, 0.235, 0.206, 0.189, 0.149, 0.00378, 4.94e-15, 3.05e-22, 1.26e-44]  min -1 · mean 0.0661 · max 0.51
contact/finger_middle: [0, 0.58, 0.628, 0.61, 0.604, 0.671, 0.461, 0.782, 0.802, 0.781]  min 0 · mean 0.675 · max 0.842
contact/finger_middle_at_success: [-1, 0.909, 0.761, 0.819, 0.877, 0.925, 0.993, 1, 0.999, 0.998]  min -1 · mean 0.908 · max 1
contact/finger_pinky: [0, 0.543, 0.684, 0.699, 0.663, 0.696, 0.449, 0.771, 0.768, 0.766]  min 0 · mean 0.682 · max 0.827
contact/finger_pinky_at_success: [-1, 0.896, 0.923, 0.956, 0.974, 0.979, 0.983, 0.998, 0.989, 0.994]  min -1 · mean 0.889 · max 1
contact/finger_ring: [0, 0.48, 0.507, 0.591, 0.658, 0.671, 0.43, 0.668, 0.708, 0.744]  min 0 · mean 0.62 · max 0.803
contact/finger_ring_at_success: [-1, 0.818, 0.828, 0.813, 0.96, 0.947, 0.936, 0.895, 0.905, 0.982]  min -1 · mean 0.805 · max 1
contact/finger_thumb: [0.00439, 0.646, 0.734, 0.738, 0.699, 0.726, 0.48, 0.779, 0.801, 0.789]  min 0.00366 · mean 0.722 · max 0.849
contact/finger_thumb_at_success: [-1, 0.999, 1, 1, 1, 1, 1, 1, 1, 1]  min -1 · mean 0.964 · max 1
contact/fingers_touching: [0.00439, 2.33, 2.83, 2.93, 2.86, 2.94, 1.82, 3, 3.08, 3.08]  min 0.00366 · mean 2.8 · max 3.32
contact/fingers_touching_at_success: [-1, 3.65, 3.75, 3.79, 4, 4, 3.92, 3.89, 3.89, 3.97]  min -1 · mean 3.7 · max 4.17
contact/link_force_mean: [0.00146, 2.68, 2.48, 2.29, 2.3, 2.48, 1.99, 3.46, 3.47, 3.13]  min 0.00146 · mean 13.5 · max 1.79e+04
contact/links_touching: [0.00464, 3.28, 3.83, 3.81, 3.68, 3.62, 2.47, 4.23, 4.3, 4.18]  min 0.00391 · mean 3.75 · max 4.51
contact/links_touching_at_success: [-1, 5.2, 5.11, 5.22, 5.24, 4.93, 5.29, 5.51, 5.43, 5.48]  min -1 · mean 5.01 · max 5.78
contact/palm_touching: [0, 0.46, 0.571, 0.592, 0.513, 0.604, 0.224, 0.521, 0.507, 0.546]  min 0 · mean 0.487 · max 0.629
contact/palm_touching_at_success: [-1, 0.686, 0.955, 0.84, 0.849, 0.892, 0.524, 0.748, 0.736, 0.823]  min -1 · mean 0.686 · max 1
ctrl/prev_ep_successes_mean: [0, 0.0198, 0.0227, 0.436, 2.18, 2.9, 0.637, 3.2, 2.97, 2.9]  min 0 · mean 1.65 · max 3.42
done/abnormal: [0, 0.000244, 0, 0.000244, 0, 0, 0, 0, 0.000244, 0]  min 0 · mean 0.000114 · max 0.00122
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 8.28e-06 · max 0.000488
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 7.3e-06 · max 0.000732
done/max_goals: [0, 0, 0, 0, 0.000488, 0, 0.000488, 0, 0.000488, 0.000244]  min 0 · mean 0.000303 · max 0.00244
done/out_xy: [0, 0.00122, 0.000244, 0.000732, 0.000488, 0, 0, 0, 0, 0]  min 0 · mean 0.000206 · max 0.00195
done/tipped: [0, 0.00146, 0.000244, 0.000488, 0.000244, 0.000244, 0.000488, 0.000244, 0, 0]  min 0 · mean 0.000283 · max 0.00244
episode_lengths/step: [2.09, 550, 558, 637, 575, 616, 818, 908, 898, 850]  min 2.09 · mean 684 · max 921
reward/close: [9.44e-07, 0.0193, 0.0159, 0.0149, 0.0128, 0.00956, 0.012, 0.018, 0.0179, 0.0135]  min 9.44e-07 · mean 0.0157 · max 0.0256
reward/goal_hold: [0, 0.292, 0.41, 0.583, 0.822, 1, 0.364, 0.865, 0.949, 1.1]  min 0 · mean 0.677 · max 1.16
reward/goal_progress: [0, -0.00438, -0.00903, -0.00423, 0.0238, 0.0543, 0.0243, 0.0456, 0.0553, 0.0961]  min -0.0323 · mean 0.0292 · max 0.11
reward/goal_reach: [0, 0.0637, 0.0883, 0.138, 0.223, 0.266, 0.097, 0.217, 0.256, 0.344]  min 0 · mean 0.182 · max 0.362
reward/grasp_count: [3.65e-06, 0.0301, 0.0386, 0.0397, 0.0372, 0.0395, 0.0218, 0.0408, 0.0427, 0.0436]  min 3.65e-06 · mean 0.0368 · max 0.0451
reward/grasp_digit: [9.12e-06, 0.0378, 0.0431, 0.0428, 0.0404, 0.0399, 0.0273, 0.0506, 0.0512, 0.049]  min 9.12e-06 · mean 0.0423 · max 0.0542
reward/grasp_oppose: [0, 0.0771, 0.0903, 0.0939, 0.0877, 0.0978, 0.0577, 0.0965, 0.0978, 0.0961]  min 0 · mean 0.0894 · max 0.105
reward/grasp_palm: [0, 0.13, 0.169, 0.176, 0.153, 0.181, 0.067, 0.154, 0.151, 0.163]  min 0 · mean 0.144 · max 0.188
reward/lift_height: [0, 0.402, 0.623, 0.786, 0.825, 0.949, 0.393, 0.917, 0.952, 1.03]  min 0 · mean 0.737 · max 1.1
reward/lift_rise: [0, 0.0178, 0.00954, 0.00789, 0.00625, 0.00803, 0.00565, 0.00897, 0.0079, 0.00616]  min 0 · mean 0.01 · max 0.0365
reward/palm_in: [0.000313, 0.00506, 0.00391, 0.0035, 0.00338, 0.00244, 0.00431, 0.0051, 0.00501, 0.00409]  min 0.000313 · mean 0.00438 · max 0.00783
reward/pen_action: [-0.0209, -0.0248, -0.025, -0.0245, -0.0241, -0.024, -0.024, -0.0231, -0.0227, -0.0225]  min -0.0252 · mean -0.0237 · max -0.0203
reward/pen_arm_fast: [0, -0.00581, -0.00637, -0.00756, -0.00624, -0.0057, -0.000401, -0.00039, -0.00105, -0.00101]  min -0.0108 · mean -0.00397 · max 0
reward/pen_arm_vel: [-0.0281, -0.0484, -0.0567, -0.056, -0.0502, -0.0543, -0.0296, -0.0404, -0.0391, -0.041]  min -0.0603 · mean -0.0451 · max -0.0221
reward/pen_cup_vel: [0, -0.0239, -0.0351, -0.0368, -0.0303, -0.0326, -0.0106, -0.0204, -0.0202, -0.0234]  min -0.0403 · mean -0.0251 · max 0
reward/pen_descend: [0, -0.0308, -0.0321, -0.0277, -0.00741, -0.0079, -0.00276, -0.0024, -0.00236, -0.00207]  min -0.0579 · mean -0.0129 · max 0
reward/pen_drift: [0, -0.0225, -0.0363, -0.0454, -0.0582, -0.0592, -0.0133, -0.0265, -0.0304, -0.0317]  min -0.0657 · mean -0.0337 · max 0
reward/pen_drop: [-0.000609, -0.068, -0.0572, -0.0442, -0.0379, -0.0206, -0.00838, -0.00995, -0.0159, -0.0128]  min -0.108 · mean -0.0323 · max -6.47e-05
reward/pen_setdown: [-0.000145, -0.027, -0.0111, -0.00606, -0.00362, -0.00255, -0.00126, -0.00171, -0.00124, -0.00116]  min -0.0799 · mean -0.00799 · max -1.57e-05
reward/pen_shove: [-0.00819, -0.0494, -0.0261, -0.0151, -0.0158, -0.0161, -0.0162, -0.0144, -0.0151, -0.0134]  min -0.187 · mean -0.0235 · max -0.00813
reward/pen_squeeze: [0, -0.169, -0.132, -0.117, -0.134, -0.149, -0.129, -0.239, -0.234, -0.219]  min -0.258 · mean -0.171 · max 0
reward/pen_stall: [0, -0.00301, -0.000685, -0.00052, -0.000203, -0.000213, -0.000148, -0.000351, -0.000267, -0.000223]  min -0.0379 · mean -0.00116 · max 0
reward/pen_table: [0, -5.52e-05, 0, 0, 0, 0, -0.000404, 0, 0, 0]  min -0.00394 · mean -7.99e-05 · max 0
reward/pen_tilt: [-0.00055, -0.0515, -0.0551, -0.042, -0.042, -0.0419, -0.0317, -0.0355, -0.0361, -0.036]  min -0.087 · mean -0.0433 · max -0.000479
reward/reach: [0.0171, 0.0111, 0.00929, 0.00783, 0.00755, 0.00569, 0.0114, 0.00978, 0.00859, 0.00816]  min 0.00387 · mean 0.0089 · max 0.0236
reward/success_bonus: [0, 0, 0, 0.0239, 0.192, 0.0599, 0.017, 0.0501, 0.0569, 0.14]  min 0 · mean 0.0606 · max 0.268
reward/total: [-0.0411, 0.557, 1.02, 1.49, 2.02, 2.3, 0.835, 2.06, 2.23, 2.69]  min -0.177 · mean 1.61 · max 2.78
rewards/step: [-1.1, 268, 470, 784, 998, 1.48e+03, 465, 1.64e+03, 1.91e+03, 1.83e+03]  min -49.8 · mean 1.09e+03 · max 2.33e+03
task/lifted_frac: [0, 0.39, 0.634, 0.677, 0.646, 0.672, 0.411, 0.711, 0.739, 0.734]  min 0 · mean 0.611 · max 0.799
task/successes_mean: [0, 0.00635, 0.0427, 0.274, 1.18, 1.46, 0.503, 1.3, 1.5, 1.42]  min 0 · mean 0.825 · max 1.6
task/tilt_deg: [0.0835, 7.07, 7.34, 5.45, 5.42, 5.37, 4.09, 4.42, 4.48, 4.47]  min 0.0615 · mean 5.67 · max 12.9
task/tol: [0.112, 0.112, 0.112, 0.112, 0.112, 0.101, 0.112, 0.112, 0.101, 0.0911]  min 0.082 · mean 0.107 · max 0.112
