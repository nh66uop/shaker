## I can see from the robot that
Round 16 ran left `cp_l_m16` from the round-15 best snapshot (epoch 981) to epoch 800, then resumed as `cp_l_m16r` from
the epoch-800 snapshot to cumulative epoch 1266 on vision-3090 (rl_games restarts its epoch counter on resume; the
numbers below are cumulative). No Traceback, no PhysX error, no NaN. The run was stopped at 1266 as a round end.

**Training curves (moving average).** grasp 0.68 at 1100 -> 0.63 at 1221 (0.51 in the last 50-epoch bucket); far-start
success 0.54 -> 0.44 (0.35); three-goal episodes 0.28 -> 0.21; envelope peaked at 0.71 around epoch 940 and was 0.60
at 1221 (0.48 in the last bucket). The wrench ADR level stayed at 1.0 (20 N/kg) from about epoch 900 to the end.
lift_ema 0.71, lifted fraction 0.69-0.72, wrench near-goal fraction 0.44-0.47 and total reward 2.04-2.16 were flat.

**Part of that decline is the success tolerance, not the policy.** ctx.success_tol is not stored in the checkpoint:
on resume it restarted at 0.1125 and shrank again, 0.11 (buckets 0-150) -> 0.10 (200-350) -> 0.09 (400-450). The
success and goal-dwell drop (dwell 36 -> 11-14, successes per episode 1.40 -> 1.02) coincides with that shrink.

**The grasp got stronger, and the squeeze charge grew with it.** Over the resumed run pen_squeeze went -0.13 -> -0.20
-> -0.25 monotonically while wrap fraction rose 0.38 -> 0.53, grasp quality at lift 0.29 -> 0.60-0.64 and hand curl
0.27 -> 0.35.

**What the operator saw in the deterministic videos (epoch 800, 1000, 1250; one episode each, largest object):**
the envelope is formed and the object is held stably ("more stable than expected"), BUT the index, middle, ring and
pinky fingers press on the object with their FINGERTIPS instead of curling around it. The operator wants the fingers to
wrap (curl) around the object. The thumb is hidden from this camera.
Per-link contact is available to the reward: ctx.link_cup_force[:, f, k] for k = 0 (link moved by joint _3),
1 (link moved by joint _4), 2 (fingertip). A digit that "presses with the tip" carries most of its cup force on k = 2;
a digit that "wraps" carries force on k = 0 and k = 1 as well.

## Feedback for improvement
Keep round 16 whole: the object is lifted, carried and held stably under the disturbance, and the operator judged the
hold itself good. Every structural rule stays (second half gated on the object's rise, envelope quality as ONE
multiplier on the downstream income, no per-finger term, falling-only drop and set-down costs, open-pose factor only on
the approach, uprightness as a multiplier, first-success-only bonus, goal-gated wrench). The budgets stay.

Change ONE thing: what counts as a deep wrap.

1. Wrap depth must mean "the digit carries its grip on its inner links", not "several links touch". Today a digit
   whose fingertip presses hard while a middle link brushes the wall already scores as a two-link wrap. For each digit,
   measure how its cup force is DISTRIBUTED over its links: the share of the digit's total cup force carried by the two
   inner links (k = 0 and k = 1) against the fingertip (k = 2). Use that share, symmetric across all five digits and
   with no digit named, to weight the wrap-depth part of the envelope quality. A digit touching only with its tip, or
   carrying almost all of its force on the tip, counts as a shallow wrap; a digit with force on its inner links counts
   as deep. This is a RELATIVE measure (a share of the digit's own force), never an absolute force target, and it must
   fall back smoothly to the current link count when a digit's total force is near zero.

2. Do not relax the squeeze charge (P_SQUEEZE, SQ_TOTAL, SQ_LINK stay). With change 1 pressing harder with the tips no
   longer raises the envelope quality, so the squeeze charge stops competing with it.

3. Nothing else changes. The success tolerance restart on resume is an environment matter and is handled at launch.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.462, 0.534, 0.574, 0.589, 0.633, 0.581, 0.699, 0.698, 0.691]  min 0 · mean 0.592 · max 0.71
contact/finger_index_at_success: [-1, 0.819, 0.897, 0.934, 0.897, 0.937, 0.908, 0.961, 0.982, 0.937]  min -1 · mean 0.877 · max 0.989
contact/finger_middle: [0, 0.593, 0.65, 0.671, 0.722, 0.73, 0.71, 0.765, 0.739, 0.745]  min 0 · mean 0.69 · max 0.771
contact/finger_middle_at_success: [-1, 0.954, 0.96, 0.953, 0.947, 0.941, 0.96, 0.975, 0.97, 0.934]  min -1 · mean 0.924 · max 0.99
contact/finger_pinky: [0, 0.599, 0.64, 0.658, 0.708, 0.728, 0.697, 0.749, 0.734, 0.739]  min 0 · mean 0.683 · max 0.752
contact/finger_pinky_at_success: [-1, 0.968, 0.958, 0.964, 0.971, 0.976, 0.979, 0.972, 0.993, 0.976]  min -1 · mean 0.94 · max 0.998
contact/finger_ring: [0, 0.607, 0.654, 0.683, 0.715, 0.734, 0.704, 0.757, 0.726, 0.74]  min 0 · mean 0.69 · max 0.758
contact/finger_ring_at_success: [-1, 0.979, 0.973, 0.978, 0.976, 0.978, 0.988, 0.97, 0.97, 0.965]  min -1 · mean 0.941 · max 1
contact/finger_thumb: [0, 0.642, 0.685, 0.708, 0.753, 0.767, 0.742, 0.792, 0.765, 0.773]  min 0 · mean 0.724 · max 0.796
contact/finger_thumb_at_success: [-1, 1, 1, 1, 1, 1, 0.999, 1, 1, 1]  min -1 · mean 0.967 · max 1
contact/fingers_touching: [0, 2.9, 3.16, 3.29, 3.49, 3.59, 3.43, 3.76, 3.66, 3.69]  min 0 · mean 3.38 · max 3.77
contact/fingers_touching_at_success: [-1, 4.72, 4.79, 4.83, 4.79, 4.83, 4.83, 4.88, 4.91, 4.81]  min -1 · mean 4.71 · max 4.95
contact/link_force_mean: [0, 3.36, 3.61, 3.7, 3.59, 4.15, 4.42, 4.85, 4.3, 4.95]  min 0 · mean 4.08 · max 5.19
contact/links_touching: [0, 3.61, 3.97, 4.14, 4.36, 4.53, 4.29, 4.64, 4.51, 4.6]  min 0 · mean 4.22 · max 4.66
contact/links_touching_at_success: [-1, 5.83, 5.99, 6.01, 5.96, 6.07, 6.05, 6.04, 6.08, 6.03]  min -1 · mean 5.88 · max 6.27
contact/palm_touching: [0, 0.544, 0.584, 0.567, 0.625, 0.644, 0.626, 0.618, 0.636, 0.616]  min 0 · mean 0.599 · max 0.684
contact/palm_touching_at_success: [-1, 0.924, 0.906, 0.844, 0.894, 0.911, 0.902, 0.831, 0.903, 0.875]  min -1 · mean 0.862 · max 0.966
ctrl/prev_ep_successes_mean: [0, 2.68, 3.09, 3.23, 3.17, 2.86, 2.41, 2.39, 2.1, 1.78]  min 0 · mean 2.44 · max 3.28
done/abnormal: [0, 0.000488, 0.000732, 0.000488, 0.000488, 0.000977, 0.000488, 0.000977, 0.000488, 0.000732]  min 0 · mean 0.000684 · max 0.00244
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.95e-07 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 4.95e-07 · max 0.000244
done/max_goals: [0, 0.000977, 0.000732, 0.00195, 0.000977, 0.000488, 0.000244, 0.000488, 0, 0.000488]  min 0 · mean 0.000504 · max 0.00269
done/out_xy: [0, 0.000977, 0.000244, 0.000488, 0, 0, 0.000488, 0, 0, 0]  min 0 · mean 0.000199 · max 0.00122
done/tipped: [0, 0.000244, 0.000488, 0.000244, 0, 0.000244, 0.000244, 0.000488, 0, 0.000488]  min 0 · mean 0.000256 · max 0.00171
episode_lengths/step: [2.09, 310, 405, 468, 453, 551, 568, 471, 498, 494]  min 2.09 · mean 453 · max 590
reward/close: [3.06e-06, 0.00675, 0.00641, 0.00758, 0.00756, 0.00703, 0.00724, 0.0088, 0.00751, 0.00888]  min 3.06e-06 · mean 0.00732 · max 0.0104
reward/goal_hold: [0, 0.788, 0.881, 0.867, 0.917, 0.975, 0.906, 0.971, 0.92, 0.901]  min 0 · mean 0.881 · max 1
reward/goal_progress: [0, 0.0638, 0.0656, 0.0627, 0.0548, 0.0451, 0.039, 0.0381, 0.051, 0.0379]  min 0 · mean 0.0469 · max 0.102
reward/goal_reach: [0, 0.207, 0.222, 0.227, 0.25, 0.262, 0.25, 0.254, 0.266, 0.268]  min 0 · mean 0.239 · max 0.283
reward/grasp_count: [0, 0.0409, 0.0451, 0.0465, 0.0498, 0.0516, 0.0484, 0.0519, 0.0519, 0.0511]  min 0 · mean 0.0477 · max 0.0533
reward/grasp_digit: [0, 0.0409, 0.0453, 0.0469, 0.0498, 0.0521, 0.0487, 0.0516, 0.051, 0.0514]  min 0 · mean 0.0478 · max 0.0538
reward/grasp_oppose: [0, 0.0889, 0.0947, 0.0965, 0.103, 0.106, 0.103, 0.108, 0.105, 0.105]  min 0 · mean 0.0995 · max 0.109
reward/grasp_palm: [0, 0.162, 0.174, 0.17, 0.187, 0.193, 0.187, 0.185, 0.19, 0.184]  min 0 · mean 0.179 · max 0.203
reward/lift_height: [0, 0.687, 0.774, 0.777, 0.846, 0.9, 0.838, 0.92, 0.924, 0.894]  min 0 · mean 0.823 · max 0.938
reward/lift_rise: [0, 0.00817, 0.00697, 0.0068, 0.00809, 0.00755, 0.00733, 0.00524, 0.00726, 0.00665]  min 0 · mean 0.00727 · max 0.0246
reward/palm_in: [0.000346, 0.00226, 0.00216, 0.00245, 0.00235, 0.00224, 0.00222, 0.00257, 0.00229, 0.00258]  min 0.000346 · mean 0.0023 · max 0.00355
reward/pen_action: [-0.0236, -0.024, -0.0238, -0.0238, -0.0238, -0.0237, -0.0234, -0.0236, -0.0235, -0.0236]  min -0.0245 · mean -0.0236 · max -0.0228
reward/pen_arm_fast: [0, -0.00903, -0.00862, -0.00875, -0.00642, -0.00834, -0.00915, -0.00941, -0.00546, -0.00778]  min -0.0197 · mean -0.00808 · max 0
reward/pen_arm_vel: [-0.0268, -0.0533, -0.0551, -0.0562, -0.0572, -0.058, -0.0578, -0.0568, -0.0552, -0.0563]  min -0.0615 · mean -0.0557 · max -0.0253
reward/pen_cup_vel: [0, -0.0336, -0.034, -0.0359, -0.0377, -0.0378, -0.0383, -0.0364, -0.0347, -0.0359]  min -0.0458 · mean -0.0356 · max 0
reward/pen_descend: [0, -0.0152, -0.0132, -0.0158, -0.0153, -0.0141, -0.0155, -0.0134, -0.0139, -0.0157]  min -0.0266 · mean -0.0143 · max 0
reward/pen_drift: [0, -0.0487, -0.0509, -0.0549, -0.065, -0.0667, -0.067, -0.0624, -0.0613, -0.068]  min -0.0769 · mean -0.0608 · max 0
reward/pen_drop: [-0.000609, -0.0202, -0.0196, -0.0265, -0.0236, -0.0224, -0.0259, -0.026, -0.0227, -0.024]  min -0.0394 · mean -0.0223 · max -6.47e-05
reward/pen_setdown: [-0.000145, -0.00549, -0.00482, -0.00447, -0.00455, -0.00406, -0.00501, -0.00571, -0.00375, -0.00378]  min -0.00866 · mean -0.00424 · max -4.01e-05
reward/pen_shove: [-0.00779, -0.0224, -0.0199, -0.0172, -0.0144, -0.014, -0.0151, -0.0144, -0.0119, -0.0127]  min -0.0395 · mean -0.0159 · max -0.00779
reward/pen_squeeze: [0, -0.169, -0.192, -0.199, -0.203, -0.224, -0.213, -0.249, -0.241, -0.255]  min -0.257 · mean -0.211 · max 0
reward/pen_stall: [0, -0.000308, -0.000283, -0.000181, -0.000215, -0.000219, -0.00032, -0.000212, -0.000293, -0.000247]  min -0.00135 · mean -0.000254 · max 0
reward/pen_table: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min -7.2e-05 · mean -4.23e-07 · max 0
reward/pen_tilt: [-0.000539, -0.0332, -0.0365, -0.0361, -0.0389, -0.0381, -0.0385, -0.038, -0.0345, -0.0354]  min -0.0491 · mean -0.0363 · max -0.000504
reward/reach: [0.0162, 0.00929, 0.00785, 0.00842, 0.00728, 0.00669, 0.00717, 0.00642, 0.00635, 0.00661]  min 0.00597 · mean 0.00756 · max 0.0217
reward/success_bonus: [0, 0.112, 0.128, 0.148, 0.0658, 0.0512, 0.0768, 0.0536, 0.0313, 0.0233]  min 0 · mean 0.0832 · max 0.306
reward/total: [-0.0429, 1.78, 1.99, 1.99, 2.06, 2.15, 2.01, 2.12, 2.11, 2]  min -0.0431 · mean 1.98 · max 2.25
rewards/step: [-1.11, 456, 717, 903, 881, 1.16e+03, 1.2e+03, 988, 1.05e+03, 1.01e+03]  min -5.84 · mean 896 · max 1.28e+03
task/lifted_frac: [0, 0.568, 0.62, 0.637, 0.684, 0.701, 0.674, 0.728, 0.715, 0.716]  min 0 · mean 0.657 · max 0.731
task/successes_mean: [0, 1.07, 1.23, 1.22, 1.33, 1.31, 1.17, 1.34, 1.04, 1.02]  min 0 · mean 1.17 · max 1.46
task/tilt_deg: [0.0822, 4.29, 4.68, 4.62, 4.94, 4.87, 4.93, 4.89, 4.34, 4.48]  min 0.0736 · mean 4.65 · max 6.47
task/tol: [0.112, 0.112, 0.112, 0.112, 0.101, 0.101, 0.101, 0.101, 0.0911, 0.0911]  min 0.0911 · mean 0.103 · max 0.112
