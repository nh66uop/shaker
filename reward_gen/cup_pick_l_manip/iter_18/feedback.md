## I can see from the robot that
Round 18 ran left `cp_l_m18` from the round-16 epoch-1000 snapshot with the round-18 reward (curl as a factor of the
envelope quality, table charge band moved to 2 -> 1 cm) for about 1,470 epochs (cumulative about 2,470) on vision-3090.
No Traceback and no PhysX error. It was stopped as a round end after epoch 1250 fell back below epoch 800 in a
fixed-tolerance evaluation; epoch 800 is kept as the best snapshot.

**Fixed-tolerance deterministic evaluation (tolerance 0.0911 m, wrench pinned at 20 N/kg, first episode of 256
environments, seeds 42 and 43: 512 episodes per checkpoint).** Against the epoch-1000 start (success 0.785):
- epoch 450: success 0.811, epoch 800: 0.842 (z +2.3, the best so far), epoch 1250: 0.785 (-0.057 against epoch 800,
  z -2.3). No round-17-style collapse: at epoch 800 round 17 was at 0.703.
- Goal chaining rose and stayed up: three-goal episodes 0.43 -> 0.59 -> 0.66 -> 0.67, successes per episode 2.41 -> 3.36,
  all five goals 0.26 -> 0.56.
- The index curls again: share of its cup force on its two inner links at the success step 0.315 (start), 0.253, 0.338,
  0.307 (round 17: 0.010). Over all five digits 0.188 -> 0.212 at epoch 1250 (z +2.6); ring 0.13 -> 0.19, little finger
  0.01 -> 0.04. Whether this is the curled grip the operator asked for is waiting for the operator's video verdict.
- Early terminations 0.248 -> 0.318, 0.322, 0.336, mostly after the lift. Contact force fell at every checkpoint (mean
  finger-link force per episode 3.52 -> 2.49 -> 2.13 N) while more digits and links touched. Drops stayed rare (0.002)
  and the tilt while held fell (worst 20 -> 16 deg).

**Why more episodes end early: a termination-cause replay** (seed 42, 256 episodes per checkpoint, the environment's
own termination terms recomputed per environment; phase read one step before the terminal step; rates with
per-environment standard errors):
- The whole excess is the object leaving the working area in y after the lift: 0.051 (start) -> 0.105 (epoch 800,
  z 2.3) -> 0.160 (epoch 1250, z 4.1). Arm runaway after the lift 0.078 -> 0.105 -> 0.055 (z 1.1, -1.1), tilt above 60
  deg after the lift 0.023 -> 0.008, object dropped below the table 0, hand into the table 0, arm joint limits 0.
  Terminations before the lift are back at the start's level (0.109 -> 0.125, 0.137; round 17: 0.219).
- Per carried goal the risk did not change. An episode starts one carrying leg per goal it goes for; round 18 chains
  more goals, so it started 930 and 907 legs against 765. Terminations after the lift per leg: 0.054 -> 0.061 -> 0.063
  (z 0.7, 0.8). Only the y exits per leg rose (0.017 -> 0.029 -> 0.045). A second replay of the start snapshot
  reproduced it (0.055 per leg, y exits 0.013).
- The y exits come from the environment's bound. The left arm still uses the right arm's bound (object y <= 0.25 m)
  while the goal box reaches y 0.35 and every new goal is the previous goal moved by up to 0.08 m per axis; a policy
  that chains more goals meets more goals beyond the line. Of the y exits after the lift, the current goal was beyond
  the line in 12 of 27 (epoch 800) and 8 of 41 (epoch 1250), and the object was already within the success tolerance
  of a goal inside the line in 0 and 8. In training the same exits rose from about 0.08-0.11 to 0.17-0.18 of episode
  endings in round 17 and round 18 alike, both of which chained more goals.
- The remaining y exits (15 of 27, 25 of 41) carried the held object past the line while the goal was inside it, most
  of them moving away from the goal at 0.6-0.8 m/s. Fast sideways moves of the held object happen in both directions
  (above 0.5 m/s on 302 and 174 trace steps towards and away from the line at epoch 800, 344 and 287 at epoch 1250,
  239 and 159 at the start); only one side has a bound within reach. Moving away from the goal is already charged by
  the signed progress term and the drift charge; no reward term was found that pays for it.
- It is not a weak hold. In the five steps before a y exit or a runaway after the lift all five digits touched with
  60-95 N (median) of grip; lifted episodes that terminated held with the same force as those that did not (60 vs 55 N
  at epoch 800, 51 vs 50 N at epoch 1250); the wrench fired on 2-3 % of the last ten steps. Runaways jumped from below
  10 rad/s to above 20 rad/s in one step in 26 of 34 and 17 of 22 cases, as at the start (18 of 27).

**The grip got lighter, partly because of the curl factor** (replay, every held step: object lifted and 3 cm up):
- Total finger force on the object 71 N (start) -> 65 -> 59 N. The ring finger moved force off its fingertip (9.2 ->
  5.2 N) and its inner-link share rose 0.02 -> 0.19; the curl factor pays for exactly that (its curl 0.03 -> 0.20).
- The thumb's inner links pressed less (24 -> 18 N), but the thumb's curl was saturated all along (share above one
  half, press trust 0.99), so the curl factor did not pay for that; it matches the squeeze charge above 40 N of total
  force (0.29 -> 0.25 per held step) and the lighter fingers it presses against.
- Hand curl 0.25 -> 0.28, envelope multiplier 0.55 -> 0.57, hold income per held step 0.98 -> 1.14. Palm contact 0.85
  -> 0.91.

**The round-18 table band worked.** The objects knocked over before a grasp with the lowest hand link at 0.217-0.220 m
(13 of 21 in round 17) are gone at epoch 800 (0 of 15) and rare at epoch 1250 (2 of 14).

The operator's direction stands: the fingers should curl around the object rather than press with their tips; the
hold was judged stable.

## Feedback for improvement
Keep the round-18 code whole: no reward change this round. Its two changes did what they were meant to. The index
curls again (0.31-0.34 of its force on the inner links at success, round 17 0.01), the ring and little finger started to
use their inner links, success reached 0.842 at epoch 800 with three-goal episodes up by a half, and the terminations
before a grasp are back to the start's level. Every structural rule stays (second half gated on the object's rise,
envelope quality as ONE multiplier on the downstream income with curl as a factor inside it, no per-finger term,
falling-only drop and set-down costs, open-pose factor only on the approach, palm seating tied to engagement,
uprightness as a multiplier, first-success-only bonus, goal-gated wrench, the table band). Every budget stays,
CURL_FLOOR stays, and no speed or control charge is relaxed.

1. The extra early terminations are not a reward matter. They are the object leaving the working area in y after the
   lift (0.05 -> 0.11 -> 0.16 of episodes); every other cause after the lift is flat or lower, and per carried goal the
   risk of ending the episode did not change (0.054 -> 0.061 -> 0.063 per leg). The left arm's bound is the right
   arm's (object y <= 0.25 m) while the goal box reaches 0.35 m and each new goal is a random step from the last one,
   so a policy that chains more goals meets more goals on or beyond the line, and at the evaluation tolerance the
   success region of any goal above 0.16 m crosses it. That bound is an environment setting and is left to the
   operator. The reward must not name a world axis or a bound, so there is nothing in it to change for this.

2. Do not answer the lighter grip. Finger force on the held object fell 71 -> 59 N: the ring finger moved force off its
   tip, which the curl factor pays for, and the thumb pressed less against the lighter fingers under the unchanged
   squeeze charge. Nothing followed from it: no drops, fewer tilts, the episodes that ended after the lift held as
   firmly as the ones that did not, and the hold income rose. A force floor would be an absolute force target; keep the
   curl factor, its trust on the inner-link force and the squeeze charge as they are.

3. Do not slow the carry to keep the object away from the line. The exits with the goal inside the line are fast
   sideways moves of a firmly held object (0.6-0.8 m/s, most of them away from the goal); such moves happen in every
   direction and only one side has a bound within reach. Moving away from the goal is already charged by the signed
   progress term and the drift charge. Raising the speed charges would tax every transport for an exit that exists only
   because of where the bound sits, and could make standing still competitive again.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.607, 0.637, 0.68, 0.693, 0.681, 0.735, 0.738, 0.731, 0.72]  min 0 · mean 0.689 · max 0.777
contact/finger_index_at_success: [-1, 0.844, 0.876, 0.945, 0.959, 0.968, 0.993, 0.977, 0.978, 0.994]  min -1 · mean 0.933 · max 1
contact/finger_middle: [0, 0.721, 0.745, 0.765, 0.749, 0.74, 0.75, 0.761, 0.754, 0.745]  min 0 · mean 0.747 · max 0.828
contact/finger_middle_at_success: [-1, 0.967, 0.961, 0.97, 0.963, 0.957, 0.962, 0.967, 0.956, 0.991]  min -1 · mean 0.96 · max 1
contact/finger_pinky: [0, 0.71, 0.743, 0.764, 0.751, 0.755, 0.765, 0.778, 0.761, 0.734]  min 0 · mean 0.749 · max 0.837
contact/finger_pinky_at_success: [-1, 0.965, 0.962, 0.982, 0.967, 0.986, 0.986, 0.992, 0.986, 0.983]  min -1 · mean 0.972 · max 1
contact/finger_ring: [0, 0.703, 0.738, 0.76, 0.75, 0.751, 0.751, 0.766, 0.762, 0.741]  min 0 · mean 0.745 · max 0.855
contact/finger_ring_at_success: [-1, 0.979, 0.973, 0.972, 0.984, 0.99, 0.98, 0.994, 0.996, 0.997]  min -1 · mean 0.97 · max 1
contact/finger_thumb: [0, 0.743, 0.777, 0.795, 0.78, 0.777, 0.784, 0.796, 0.782, 0.761]  min 0 · mean 0.775 · max 0.885
contact/finger_thumb_at_success: [-1, 1, 1, 1, 0.998, 1, 1, 1, 1, 1]  min -1 · mean 0.99 · max 1
contact/fingers_touching: [0, 3.48, 3.64, 3.77, 3.72, 3.7, 3.79, 3.84, 3.79, 3.7]  min 0 · mean 3.7 · max 4.12
contact/fingers_touching_at_success: [-1, 4.75, 4.77, 4.87, 4.87, 4.9, 4.92, 4.93, 4.92, 4.97]  min -1 · mean 4.84 · max 5
contact/link_force_mean: [0, 3.66, 3.88, 4.13, 3.77, 3.74, 3.53, 3.85, 4.09, 3.56]  min 0 · mean 3.94 · max 129
contact/links_touching: [0, 4.48, 4.61, 4.88, 4.85, 4.8, 5, 5.06, 4.9, 4.79]  min 0 · mean 4.81 · max 5.2
contact/links_touching_at_success: [-1, 6.05, 5.99, 6.18, 6.18, 6.16, 6.4, 6.32, 6.33, 6.36]  min -1 · mean 6.18 · max 6.53
contact/palm_touching: [0, 0.63, 0.665, 0.668, 0.679, 0.685, 0.673, 0.679, 0.63, 0.627]  min 0 · mean 0.654 · max 0.761
contact/palm_touching_at_success: [-1, 0.872, 0.91, 0.892, 0.906, 0.901, 0.898, 0.954, 0.856, 0.882]  min -1 · mean 0.885 · max 1
ctrl/prev_ep_successes_mean: [0, 2.8, 2.36, 2.23, 2.38, 2.01, 2.4, 1.88, 2.24, 2.36]  min 0 · mean 2.23 · max 2.92
done/abnormal: [0, 0, 0.00122, 0.00146, 0.00171, 0.000732, 0.000244, 0.000732, 0.000244, 0.000244]  min 0 · mean 0.000844 · max 0.00317
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.68e-06 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.01e-06 · max 0.000244
done/max_goals: [0, 0, 0.000244, 0.000244, 0, 0.000244, 0.000732, 0.000244, 0.000732, 0.000244]  min 0 · mean 0.000336 · max 0.00195
done/out_xy: [0, 0.000488, 0, 0, 0.000244, 0.000244, 0, 0.000488, 0, 0.000488]  min 0 · mean 0.000271 · max 0.00171
done/tipped: [0, 0, 0, 0, 0.000488, 0, 0.000244, 0, 0, 0.000732]  min 0 · mean 0.000206 · max 0.00122
episode_lengths/step: [2.09, 476, 404, 430, 455, 531, 593, 557, 517, 513]  min 2.09 · mean 485 · max 631
reward/close: [3.08e-06, 0.00659, 0.0072, 0.00789, 0.00667, 0.00606, 0.00697, 0.00683, 0.00829, 0.0076]  min 3.08e-06 · mean 0.00712 · max 0.0101
reward/goal_hold: [0, 0.641, 0.65, 0.681, 0.696, 0.718, 0.717, 0.699, 0.682, 0.706]  min 0 · mean 0.677 · max 0.77
reward/goal_progress: [0, 0.0339, 0.0388, 0.0344, 0.024, 0.0328, 0.0422, 0.0376, 0.0398, 0.0409]  min 0 · mean 0.0349 · max 0.0845
reward/goal_reach: [0, 0.171, 0.189, 0.22, 0.24, 0.247, 0.247, 0.256, 0.268, 0.273]  min 0 · mean 0.231 · max 0.301
reward/grasp_count: [0, 0.0525, 0.0535, 0.0564, 0.057, 0.0562, 0.0586, 0.0585, 0.0552, 0.0544]  min 0 · mean 0.0556 · max 0.0602
reward/grasp_digit: [0, 0.0527, 0.0536, 0.0583, 0.0585, 0.0577, 0.0603, 0.061, 0.0576, 0.0563]  min 0 · mean 0.057 · max 0.062
reward/grasp_oppose: [0, 0.102, 0.105, 0.107, 0.106, 0.106, 0.103, 0.105, 0.106, 0.103]  min 0 · mean 0.104 · max 0.119
reward/grasp_palm: [0, 0.189, 0.199, 0.2, 0.203, 0.205, 0.202, 0.203, 0.189, 0.188]  min 0 · mean 0.196 · max 0.228
reward/lift_height: [0, 0.586, 0.625, 0.664, 0.689, 0.713, 0.713, 0.713, 0.692, 0.688]  min 0 · mean 0.667 · max 0.754
reward/lift_rise: [0, 0.0048, 0.00515, 0.00424, 0.00458, 0.00453, 0.00451, 0.00208, 0.0037, 0.00419]  min 0 · mean 0.00408 · max 0.0237
reward/palm_in: [0.000301, 0.00203, 0.00208, 0.00225, 0.00193, 0.0018, 0.002, 0.00196, 0.00246, 0.0023]  min 0.000301 · mean 0.00211 · max 0.00284
reward/pen_action: [-0.023, -0.0235, -0.0238, -0.0235, -0.0234, -0.0234, -0.0231, -0.023, -0.0229, -0.0226]  min -0.0247 · mean -0.0233 · max -0.0214
reward/pen_arm_fast: [0, -0.00468, -0.00646, -0.00516, -0.00462, -0.00507, -0.00386, -0.00643, -0.0057, -0.00308]  min -0.0113 · mean -0.00518 · max 0
reward/pen_arm_vel: [-0.0263, -0.0561, -0.0583, -0.059, -0.0586, -0.0576, -0.0588, -0.058, -0.0567, -0.0532]  min -0.0685 · mean -0.0573 · max -0.0253
reward/pen_cup_vel: [0, -0.0348, -0.0365, -0.0395, -0.0395, -0.0389, -0.0394, -0.0384, -0.0376, -0.0336]  min -0.0518 · mean -0.0374 · max 0
reward/pen_descend: [0, -0.00829, -0.007, -0.0101, -0.00817, -0.00906, -0.00848, -0.0102, -0.00796, -0.00808]  min -0.02 · mean -0.00876 · max 0
reward/pen_drift: [0, -0.0608, -0.0675, -0.0846, -0.0983, -0.0884, -0.0899, -0.0902, -0.0947, -0.0892]  min -0.117 · mean -0.0852 · max 0
reward/pen_drop: [-0.000609, -0.0213, -0.0178, -0.0177, -0.0165, -0.0156, -0.0151, -0.0156, -0.0171, -0.0139]  min -0.0349 · mean -0.0178 · max -6.47e-05
reward/pen_setdown: [-0.000145, -0.00284, -0.00322, -0.00229, -0.00255, -0.00263, -0.00245, -0.00255, -0.00267, -0.00272]  min -0.00778 · mean -0.00278 · max -1.57e-05
reward/pen_shove: [-0.00773, -0.0153, -0.0127, -0.0113, -0.0138, -0.0118, -0.0128, -0.0117, -0.011, -0.0167]  min -0.0452 · mean -0.0131 · max -0.00678
reward/pen_squeeze: [0, -0.202, -0.225, -0.222, -0.218, -0.215, -0.217, -0.217, -0.23, -0.212]  min -0.251 · mean -0.216 · max 0
reward/pen_stall: [0, -0.000156, -0.000241, -0.000116, -0.000145, -0.000118, -0.000125, -0.000102, -0.000104, -0.000193]  min -0.00156 · mean -0.000145 · max 0
reward/pen_table: [0, -0.000203, -0.000261, 0, -5.54e-05, 0, -0.000263, -0.00106, -2.92e-05, -0.000469]  min -0.00361 · mean -0.000474 · max 0
reward/pen_tilt: [-0.000538, -0.042, -0.0386, -0.0343, -0.0356, -0.0331, -0.0348, -0.035, -0.0326, -0.0316]  min -0.0657 · mean -0.0364 · max -0.00013
reward/reach: [0.0168, 0.00685, 0.00634, 0.00612, 0.00598, 0.00598, 0.00582, 0.00558, 0.00614, 0.00584]  min 0.00436 · mean 0.00618 · max 0.0218
reward/success_bonus: [0, 0.0588, 0.0436, 0.0382, 0.0376, 0.0739, 0.0282, 0.0332, 0.0331, 0.0318]  min 0 · mean 0.0495 · max 0.265
reward/total: [-0.0413, 1.43, 1.48, 1.57, 1.61, 1.73, 1.68, 1.67, 1.62, 1.67]  min -0.0413 · mean 1.59 · max 1.85
rewards/step: [-1.11, 677, 497, 642, 718, 904, 1.01e+03, 865, 899, 892]  min -4.69 · mean 771 · max 1.05e+03
task/lifted_frac: [0, 0.694, 0.727, 0.752, 0.737, 0.734, 0.743, 0.76, 0.736, 0.71]  min 0 · mean 0.729 · max 0.822
task/successes_mean: [0, 1.48, 1.38, 1.3, 1.35, 1.22, 1.36, 1.23, 1.23, 1.15]  min 0 · mean 1.31 · max 1.81
task/tilt_deg: [0.082, 5.34, 4.94, 4.35, 4.51, 4.17, 4.38, 4.45, 4.11, 3.97]  min 0.0159 · mean 4.62 · max 8.5
task/tol: [0.1, 0.1, 0.09, 0.081, 0.0729, 0.0729, 0.0729, 0.0656, 0.059, 0.059]  min 0.0531 · mean 0.075 · max 0.1
