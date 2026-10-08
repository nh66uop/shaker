## I can see from the robot that
Round 17 ran left `cp_l_m17` from the round-16 epoch-1000 snapshot with the round-17 reward (wrap depth weighted by
each digit's inner-link force share) for 1250 epochs (cumulative 2250) on vision-3090. No Traceback, no PhysX error, no
NaN. It was stopped at 1250 as a round end because a fixed-tolerance evaluation showed a regression against the
snapshot it started from.

**Fixed-tolerance deterministic evaluation (tolerance 0.0911 m, wrench pinned at 20 N/kg, first episode of 256
environments, seeds 42 and 43: 512 episodes per checkpoint).** Epoch 800 against the epoch-1000 start: success 0.785 ->
0.703 (z -2.9). Goal chaining improved (three-goal episodes +0.09, successes per episode +0.31), but early terminations
rose 0.25 -> 0.41 (z 5.6) and lifted episodes fell 0.91 -> 0.79 (z -5.3). At epoch 450 success was still equal (0.760)
while terminations had already risen by 0.12.

**The fingers did not curl; the index moved its grip onto its fingertip.** Replay of both checkpoints (seed 42), every
held step (object lifted and at least 3 cm up):
- index: share of its cup force carried by the two inner links 0.19 -> 0.03 (z -5.4); fingertip-dominant on 0.73 ->
  0.98 of its pressing steps; inner-link contact on 0.26 -> 0.04 of held steps; fingertip force 9.8 -> 14.8 N, inner
  links 2.3 -> 0.4 N.
- middle, ring, pinky: 0.02-0.08 of their force on the inner links at BOTH checkpoints (fingertip-dominant 0.92-0.99).
  The four fingers together: 0.10 -> 0.05.
- thumb: 0.69 -> 0.74 on its inner links, 27 -> 35 N there. Total cup force while held 92 -> 111 N (z 2.8), squeeze
  charge 0.29 -> 0.34 per held step.
- Around the first success of an episode (30 steps before to 30 after): at epoch 1000 the index carried 0.40 of its
  force inside 30 steps before and 0.31 30 steps after (force drifts to the tip during a hold); at epoch 800 it carried
  0.03 throughout, with twice the fingertip force (10-11 N against 5-6 N).

**Why the round-17 depth weighting did not produce curling.** The round-17 reward was evaluated on the replayed states
(its own code, per held step):
- The envelope quality did not fall: env_q 0.692 -> 0.701. Its depth part fell (0.331 -> 0.309, z -3.9), but palm
  contact rose (0.825 -> 0.901, z 2.6), and the palm gain (+0.019 of env_q) is more than twice the depth loss (-0.008).
- Palm contact is paid three times on every held step: as 0.25 of env_q, inside `held` (0.3 + 0.7 palm) which gates
  lift, transport and hold, and again inside `hold_grip` (0.4 + 0.6 palm). The replay fits an index fingertip pushing
  the object into the palm while the thumb pushes harder from the other side: hold income per held step 1.26 -> 1.43,
  `held` 0.87 -> 0.91.
- The depth part is a SUMMAND averaged over five digits. One digit's whole swing from a fingertip press to a full curl
  moves env_q by at most 0.35 x 0.7 / 5 = 0.049; with each finger touching on a single link (link-count depth 0.48-0.52
  on all four fingers at both checkpoints) the index's actual change cost 0.006 of env_q, 0.02-0.03 per held step.
  Curling EVERY digit fully would have added only 0.31-0.38 per held step, 8-9 % of the downstream base (3.7-4.3).
- It is not the trust ramp and not saturation: the force-share trust was 0.92-0.999 on every digit (all digits press far
  above 0.3 N), env_q sat at 0.70, the digit-count ladder at 0.90. Opposition was saturated (0.98-0.99) and did not pay
  for the change (thumb opposition 0.87 -> 0.83, palm opposition 0.28 -> 0.35).

**Why more episodes ended early.** Replay, 256 episodes each, cause recomputed per environment from the environment's
own termination terms: 69 -> 106 terminations, 28 -> 56 of them before the object was lifted.
- Arm joint speed above 20 rad/s doubled, 0.098 -> 0.195 of episodes (z 3.2): the largest cause and the largest
  increase. 10 of them (epoch 1000: 1) happened with NO object contact, before any grasp, with the lowest hand link
  within 2.4 cm of the 0.205 m table top (median 1.4 cm) and the speed on wrist joints 5-7. The other 40 (epoch 1000:
  24) happened with the object grasped, during lift-off (14 against 4) or while carrying (26 against 20); in 17 of the
  50 runaways the object's speed jumps from under 1 m/s to over 1.5 m/s in a single step. Those showed no consistent
  relation to grip force or fingertip use.
- Objects knocked over before any grasp 0.047 -> 0.082; 13 of the 21 (epoch 1000: 5 of 12) with the lowest hand link at
  0.217-0.220 m. During training the lowest hand link across environments slid 0.222 -> 0.213 m while pen_table stayed
  0.0000 for the whole run: the table charge starts 2 mm above the top, but hand-link points sit inside the links (on
  the object a touching link reads 1-3 cm), so a finger resting on the table reads about 1-2 cm and is never charged.
- Out of the working area in y: 0.105 -> 0.117, the same at both checkpoints. The bound (y <= 0.25 m) is the right
  arm's; on the left arm the object spawns up to y 0.30 and the goal box reaches y 0.35, so 8 of 256 episodes end at
  their first step and 15-19 while carrying the object towards a goal near that line. That is an environment matter.
- The wrench fired in the last 10 steps of 0 and 7 of the terminated episodes: it is not the driver.

The operator's direction stands: the index to little finger should curl around the object instead of pressing with
their tips, and the hold itself was judged stable.

## Feedback for improvement
Keep the round-17 code except two things. The object is lifted, carried and held under the disturbance, the operator
judged the hold stable, and goal chaining improved. Every structural rule stays (second half gated on the object's
rise, envelope quality as ONE multiplier on the downstream income, no per-finger term, falling-only drop and set-down
costs, open-pose factor only on the approach, palm seating tied to engagement, uprightness as a multiplier,
first-success-only bonus, goal-gated wrench). Every budget stays, and no speed or control charge is relaxed.

1. Curl must be a FACTOR of the envelope quality, not a summand that palm contact can buy back. Round 17 weighted the
   wrap-depth summand by each digit's inner-link share; the policy paid for moving the index onto its fingertip (-0.006
   of env_q) with palm seating (+0.019), and even a full curl of every digit was worth only 8-9 % of the downstream
   income. Replace that weighting with a direct curl measure:
   - For each digit, curl = the share of that digit's OWN cup force carried by its two inner links (k = 0 and k = 1)
     against its fingertip (k = 2), full at about one half, multiplied by a smooth trust that the digit presses (a few
     tenths of a newton). A digit that does not touch contributes 0, never a fallback value, so taking a finger off the
     object can never raise it. Same formula for all five digits, no digit named, relative (never an absolute force
     target), no joint angle.
   - Hand curl = the mean of the five digit curls.
   - Multiply env_q by (CURL_FLOOR + (1 - CURL_FLOOR) x hand curl), so the digit ladder, palm and opposition can no
     longer pay back a fingertip press. The depth summand returns to round 16's link count: the force share enters
     once, as this factor.
   - Size it from the replay: today's grasp (thumb 0.64-0.74 of its force inside, the four fingers 0.02-0.08, hand curl
     about 0.25) keeps about three quarters of the downstream income it earns now; a hand whose five digits each carry
     half or more of their force inside earns all of it; each further digit that curls is worth about 6 % of all lift,
     transport and hold income, as much as the palm seating the fingertip press bought (env_q +0.019 and `held` 0.87 ->
     0.91, together 5-8 % of that income per held step). The floor keeps rule (b): with today's grasp, lifting and
     carrying still clearly beat anything available on the table.

2. The table charge must fire where a hand link actually touches the table. Its floor half charges only when the
   lowest hand-link point is within 2 mm of the table top, but those points sit inside the links, so it stayed 0.0000
   for the whole run while the hand came down onto the table. Of the new early terminations before any grasp, all 10
   arm runaways without object contact ended with the lowest hand-link point within 2.4 cm of the top (median 1.4 cm),
   and so did 13 of the 21 objects knocked over (1.2-1.5 cm); the other terminations ended with it at a median of
   24-28 cm. Move the band up to where a link resting on the table actually is: start charging about 2 cm above the
   top and reach the full floor charge by about 1 cm. Keep P_TABLE's cap of 0.6 and its palm half, and keep it a
   function of the current height only (no latch, bounded), so a hand wrapping the object's side wall above the band
   is never charged.

3. Nothing else changes. The left arm's out-of-area bound (y <= 0.25 m, inherited from the right arm) ends 3 % of
   episodes at their first step and 6-7 % while carrying; that is an environment matter handled at launch. The
   runaways during lift-off and carrying (the object's speed jumping in one step under a firm grasp) showed no link to
   grip force or fingertip use in the replay and are not a reward matter this round.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.593, 0.654, 0.687, 0.695, 0.719, 0.731, 0.715, 0.732, 0.722]  min 0 · mean 0.69 · max 0.766
contact/finger_index_at_success: [-1, 0.869, 0.953, 0.964, 0.985, 0.963, 0.983, 0.977, 0.976, 0.976]  min -1 · mean 0.943 · max 1
contact/finger_middle: [0, 0.718, 0.718, 0.751, 0.732, 0.744, 0.757, 0.747, 0.752, 0.757]  min 0 · mean 0.739 · max 0.826
contact/finger_middle_at_success: [-1, 0.976, 0.966, 0.99, 0.982, 0.97, 0.954, 0.975, 0.979, 0.975]  min -1 · mean 0.957 · max 1
contact/finger_pinky: [0, 0.706, 0.702, 0.741, 0.723, 0.736, 0.753, 0.746, 0.743, 0.763]  min 0 · mean 0.734 · max 0.834
contact/finger_pinky_at_success: [-1, 0.967, 0.979, 0.971, 0.987, 0.971, 0.981, 0.97, 0.981, 0.981]  min -1 · mean 0.964 · max 1
contact/finger_ring: [0, 0.709, 0.698, 0.745, 0.715, 0.721, 0.73, 0.729, 0.739, 0.761]  min 0 · mean 0.727 · max 0.846
contact/finger_ring_at_success: [-1, 0.97, 0.957, 0.972, 0.979, 0.939, 0.973, 0.968, 0.973, 0.98]  min -1 · mean 0.955 · max 1
contact/finger_thumb: [0, 0.747, 0.743, 0.777, 0.753, 0.765, 0.784, 0.776, 0.78, 0.79]  min 0 · mean 0.767 · max 0.884
contact/finger_thumb_at_success: [-1, 0.996, 0.999, 0.997, 1, 1, 1, 1, 1, 1]  min -1 · mean 0.988 · max 1
contact/fingers_touching: [0, 3.47, 3.52, 3.7, 3.62, 3.69, 3.76, 3.71, 3.75, 3.79]  min 0 · mean 3.66 · max 4.07
contact/fingers_touching_at_success: [-1, 4.78, 4.85, 4.89, 4.93, 4.84, 4.89, 4.89, 4.91, 4.91]  min -1 · mean 4.83 · max 5
contact/link_force_mean: [0, 3.62, 3.91, 4.67, 5.1, 5.01, 4.15, 3.94, 4.24, 3.72]  min 0 · mean 4.13 · max 7.3
contact/links_touching: [0, 4.35, 4.4, 4.75, 4.57, 4.7, 4.76, 4.79, 4.87, 4.93]  min 0 · mean 4.67 · max 5.17
contact/links_touching_at_success: [-1, 5.91, 6.05, 6.14, 6.09, 6.14, 6.05, 6.24, 6.34, 6.37]  min -1 · mean 6.1 · max 6.43
contact/palm_touching: [0, 0.642, 0.623, 0.619, 0.632, 0.672, 0.679, 0.688, 0.673, 0.692]  min 0 · mean 0.656 · max 0.789
contact/palm_touching_at_success: [-1, 0.903, 0.893, 0.873, 0.898, 0.914, 0.924, 0.917, 0.924, 0.919]  min -1 · mean 0.893 · max 1
ctrl/prev_ep_successes_mean: [0, 3.04, 2.65, 2.68, 2.04, 2.36, 1.83, 2.36, 2.47, 2.32]  min 0 · mean 2.35 · max 3.04
done/abnormal: [0, 0.000244, 0.00122, 0.000732, 0.00122, 0.00146, 0.000977, 0.000488, 0.00122, 0]  min 0 · mean 0.000906 · max 0.00293
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 9.14e-07 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.19e-06 · max 0.000244
done/max_goals: [0, 0.000732, 0.000244, 0.000244, 0.000244, 0.000244, 0.000244, 0, 0, 0.000244]  min 0 · mean 0.000426 · max 0.00195
done/out_xy: [0, 0.000488, 0, 0.000244, 0.000244, 0.000732, 0.000244, 0, 0.000244, 0]  min 0 · mean 0.000278 · max 0.00171
done/tipped: [0, 0.000488, 0, 0, 0.000244, 0, 0, 0, 0, 0]  min 0 · mean 0.000205 · max 0.00122
episode_lengths/step: [2.09, 400, 460, 389, 442, 410, 439, 541, 471, 576]  min 2.09 · mean 455 · max 626
reward/close: [3.08e-06, 0.00653, 0.00661, 0.00828, 0.0068, 0.00542, 0.00554, 0.00511, 0.00624, 0.00594]  min 3.08e-06 · mean 0.00632 · max 0.01
reward/goal_hold: [0, 0.898, 0.896, 0.866, 0.852, 0.942, 0.903, 0.946, 0.907, 0.924]  min 0 · mean 0.902 · max 0.999
reward/goal_progress: [0, 0.0522, 0.056, 0.0454, 0.0628, 0.0545, 0.0567, 0.0829, 0.0676, 0.0673]  min 0 · mean 0.0616 · max 0.114
reward/goal_reach: [0, 0.242, 0.266, 0.271, 0.281, 0.325, 0.335, 0.354, 0.353, 0.384]  min 0 · mean 0.313 · max 0.422
reward/grasp_count: [0, 0.051, 0.0506, 0.0536, 0.053, 0.0543, 0.0544, 0.056, 0.0568, 0.0583]  min 0 · mean 0.054 · max 0.0609
reward/grasp_digit: [0, 0.0502, 0.0502, 0.0549, 0.0528, 0.0552, 0.0556, 0.0576, 0.058, 0.0596]  min 0 · mean 0.0549 · max 0.0624
reward/grasp_oppose: [0, 0.101, 0.1, 0.104, 0.104, 0.105, 0.105, 0.104, 0.103, 0.107]  min 0 · mean 0.103 · max 0.12
reward/grasp_palm: [0, 0.192, 0.187, 0.185, 0.189, 0.201, 0.203, 0.206, 0.202, 0.207]  min 0 · mean 0.196 · max 0.236
reward/lift_height: [0, 0.811, 0.819, 0.839, 0.841, 0.884, 0.902, 0.91, 0.894, 0.912]  min 0 · mean 0.865 · max 0.993
reward/lift_rise: [0, 0.00551, 0.0049, 0.00405, 0.00729, 0.00646, 0.00615, 0.0082, 0.00668, 0.00583]  min 0 · mean 0.00635 · max 0.0299
reward/palm_in: [0.000301, 0.00202, 0.00204, 0.00248, 0.00215, 0.0017, 0.00174, 0.00167, 0.00187, 0.00187]  min 0.000301 · mean 0.00197 · max 0.00282
reward/pen_action: [-0.023, -0.0235, -0.0231, -0.0234, -0.0233, -0.0233, -0.0232, -0.0233, -0.0235, -0.0233]  min -0.0246 · mean -0.0233 · max -0.0215
reward/pen_arm_fast: [0, -0.00634, -0.00601, -0.00612, -0.00796, -0.00842, -0.00501, -0.00468, -0.00559, -0.00512]  min -0.0112 · mean -0.00577 · max 0
reward/pen_arm_vel: [-0.0263, -0.0553, -0.0558, -0.057, -0.0584, -0.0587, -0.0577, -0.0591, -0.0607, -0.0592]  min -0.0706 · mean -0.0574 · max -0.0253
reward/pen_cup_vel: [0, -0.0353, -0.036, -0.0351, -0.0379, -0.0378, -0.0361, -0.0378, -0.0393, -0.0391]  min -0.0543 · mean -0.0367 · max 0
reward/pen_descend: [0, -0.00943, -0.0129, -0.0104, -0.0109, -0.0127, -0.00917, -0.0117, -0.00992, -0.00954]  min -0.0202 · mean -0.0107 · max 0
reward/pen_drift: [0, -0.0629, -0.0663, -0.077, -0.0715, -0.0858, -0.0858, -0.0803, -0.0921, -0.105]  min -0.122 · mean -0.0795 · max 0
reward/pen_drop: [-0.000609, -0.0213, -0.0216, -0.0251, -0.0252, -0.0183, -0.0168, -0.0184, -0.0214, -0.0149]  min -0.0302 · mean -0.0191 · max -6.47e-05
reward/pen_setdown: [-0.000145, -0.00371, -0.00377, -0.00261, -0.00433, -0.0033, -0.00301, -0.00465, -0.00429, -0.00243]  min -0.00747 · mean -0.00334 · max -1.57e-05
reward/pen_shove: [-0.00773, -0.0146, -0.0158, -0.0113, -0.016, -0.0125, -0.0126, -0.0137, -0.0129, -0.0124]  min -0.0451 · mean -0.0136 · max -0.00578
reward/pen_squeeze: [0, -0.212, -0.218, -0.246, -0.239, -0.244, -0.225, -0.212, -0.246, -0.206]  min -0.252 · mean -0.224 · max 0
reward/pen_stall: [0, -0.000163, -0.000156, -8.75e-05, -0.000195, -0.00019, -0.000182, -0.000256, -0.000162, -0.000183]  min -0.00178 · mean -0.000191 · max 0
reward/pen_table: [0, 0, 0, 0, 0, 0, 0, -1.18e-05, 0, -4e-05]  min -0.000144 · mean -5.81e-06 · max 0
reward/pen_tilt: [-0.000538, -0.0372, -0.034, -0.0371, -0.0354, -0.0358, -0.0366, -0.0325, -0.0374, -0.0373]  min -0.0644 · mean -0.0354 · max -0.00013
reward/reach: [0.0168, 0.00688, 0.0071, 0.00685, 0.00677, 0.00593, 0.0061, 0.00603, 0.00608, 0.006]  min 0.00403 · mean 0.00653 · max 0.0218
reward/success_bonus: [0, 0.104, 0.126, 0.0211, 0.0262, 0.0574, 0.039, 0.0156, 0.0765, 0.0255]  min 0 · mean 0.0664 · max 0.334
reward/total: [-0.0413, 2.04, 2.08, 1.93, 1.95, 2.16, 2.16, 2.25, 2.18, 2.25]  min -0.0413 · mean 2.13 · max 2.49
rewards/step: [-1.11, 751, 943, 801, 915, 869, 947, 1.19e+03, 1.04e+03, 1.35e+03]  min -3.34 · mean 965 · max 1.45e+03
task/lifted_frac: [0, 0.693, 0.693, 0.734, 0.696, 0.707, 0.72, 0.706, 0.721, 0.733]  min 0 · mean 0.709 · max 0.823
task/successes_mean: [0, 1.48, 1.33, 1.39, 1.05, 1.19, 1.05, 1.11, 1.24, 1.13]  min 0 · mean 1.22 · max 1.79
task/tilt_deg: [0.082, 4.77, 4.35, 4.72, 4.5, 4.51, 4.65, 4.13, 4.76, 4.7]  min 0.0159 · mean 4.5 · max 8.33
task/tol: [0.1, 0.1, 0.09, 0.081, 0.081, 0.0729, 0.0656, 0.0656, 0.059, 0.0531]  min 0.0478 · mean 0.0741 · max 0.1
