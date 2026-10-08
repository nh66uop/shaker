## I can see from the robot that
Round 19 runs left `cp_l_m19` from the round-18 epoch-800 snapshot with the round-18 reward unchanged and one
environment change made by the operator: the left arm's object y bound is now the mirror of the right arm's. At the time
of writing it had run about 430 epochs on vision-3090, no Traceback and no PhysX error, and it keeps running: this
round is about the grip, not a round end. So far the bound change did what it was for. Object-out-of-area episode
endings fell from about 0.16-0.18 of episodes in round 18 to about 0.01-0.04 (training, done/out_xy x episode length),
successes per finished episode are 3.2 (round 18: 2.1-2.4), the tolerance went 0.10 -> 0.081, the wrench stays at level
1.0, five digits and 6.3 links touch at success with the palm on 0.90 of successes.

**The operator's verdict on the round-18 grip: still a fingertip press.** The operator compared the round-18 epoch-800
video with the round-16 epoch-1000 video on the largest cup, side by side while the cup is carried: in both the index,
middle, ring and little fingers press the object with their fingertips; neither curls or rolls the fingers around it.
The rise of the index's inner-link force share (0.338 at success, the round-16 level) does not count as curl. The curl
signal has to be redesigned.

**What the hand actually does** (geometry replay: seed 42, first episode of 256 environments, tolerance 0.0911 m,
wrench pinned at 20 N/kg, both snapshots; plus a replay of the exact 12-environment videos the operator judged). Each
digit was measured at five points - the start of the first phalanx next to the palm, the three link_pos points, and the
end of the fingertip pad - against the object's own outer wall, and its angle around the object axis from the palm:
- Every digit keeps its first joint next to the palm almost straight (0.00-0.17 of its range: thumb, index, middle, ring,
  little finger alike) and folds a later joint to the limit (last joint 0.85-1.00; the index folds its middle joint
  instead part of the time). The middle joint points of the fingers stand 1.8-3.9 cm off the wall while the fingertip is
  on it (0.6-1.1 cm). 93-98 % of the fingertip force is along the wall normal and the last link points into the wall.
  That is the press the operator sees: straight bars across the object with the tips driven into it. 66 % of all held
  digit-steps are this hook.
- Reaching around is not what is missing. The fingertip pads sit 115-148 degrees around the object from the palm and are
  past the object's widest point on 86-100 % of held steps, in both snapshots.
- The only digit that really rolls - every joint bent part of the way, every point within about 1.5 cm of the wall - is
  the ring finger at epoch 800 (32 % of its held steps, up from 0). The index alternates between the fingertip hook
  (40-63 %) and a second hook: first joint straight, middle joint folded to its limit, last joint straight, so the last
  link lies flat on the wall (25-33 %).
- The frames the operator judged (largest cup, 5 s): every finger touches only with its tip (0.1-0.9 cm from the wall)
  while at least one of its middle joint points is 2.5-3.8 cm off. Epoch 800 seats the palm (14 N); epoch 1000 does not
  (2.6 cm gap).

**Why the round-18 curl factor paid for a press.** It measured, per digit, the share of the digit's force on links
k=0 and k=1 against the fingertip link k=2. On this hand k=1 is the last link itself and the fingertip body is fixed
rigidly onto that same link, with its pad overlapping the last link's own collision surface. So "inner against tip"
splits one rigid segment, and a last link laid flat on the wall loads k=1: share 1, full curl.
- The index's flat-last-link hook scores 0.91 and its fingertip hook 0.00. The index's "recovery" in round 18 is a switch
  between these two hooks (mostly on the mid-sized cups and the closed shaker), not a curl.
- The thumb's fingertip hook scores 0.88: its last link is 5 cm long and carries the press. The thumb alone supplies
  about 0.17 of the 0.25-0.27 hand curl, whatever pose it takes.
- A real roll scores 0.16: the rolled ring finger still presses mostly with its tip.
- The share saturates at one half, so a little force on k=1 is enough for full credit.
- On the largest cup, the one the operator watched, the factor rated epoch 1000 above epoch 800 (hand curl 0.28 against
  0.20, 32 episodes each; the index 0.46 against 0.02).
- The first phalanx next to the palm, whose bending is the first step of any curl, has no contact sensor and no
  link_pos entry. The force share cannot see it.

**What separates a press from a curl, measured.** Whether the digit follows the wall: the spread of its three
link_surface_dist values (farthest minus closest link point of the same digit). A rolled digit 1.1 cm (thumb 0.6 cm), the
flat-last-link hook 1.9 cm, the fingertip hook 2.0-2.5 cm; the true radial distances give the same order. It reads only
context fields every digit shares. Three other candidates do not separate: how far around the object the fingertip
reaches (past the widest point in both snapshots), the number of links in contact (the rolled ring finger presses mostly
with its tip), and normal against tangential fingertip force (almost all normal in every pose).

The operator's direction stands and is now specific: the fingers should bend at every joint so that the whole digit
lies along the object, instead of straight bars with hooked tips. The hold itself was judged stable.

## Feedback for improvement
One change: replace the per-digit measure inside the round-18 curl factor. Keep everything else of the round-18 code
exactly as it is. The factor stays where it is (env_q = env_base x curl_factor, curl_factor = CURL_FLOOR + (1 -
CURL_FLOOR) x hand_curl, hand_curl = mean over the five digits, CURL_FLOOR 0.50); only what a digit's curl measures
changes. Round 18's measure (share of the digit's force on links k=0, k=1 against k=2) paid for two fingertip presses -
the index's last link laid flat on the wall (0.91) and the thumb's hooked tip (0.88) - and almost nothing for the one
digit that really rolled (0.16), because k=1 is the last link itself and the fingertip body is fixed onto it.

1. Curl = the digit FOLLOWS THE WALL. Per digit, from ctx.link_surface_dist of its three link points:
       lift_off = link_surface_dist.amax(dim=-1) - link_surface_dist.amin(dim=-1)                 # (N,5)
       follow   = exp(-(lift_off - FOLLOW_DEAD).clamp(min=0) / FOLLOW_SCALE)                      # (N,5) in (0, 1]
       digit_curl = digit_touch * follow                                                          # (N,5)
   with FOLLOW_DEAD = 0.010 m and FOLLOW_SCALE = 0.006 m, and digit_touch the soft contact the code already computes
   (a digit that does not touch gives exactly 0, no fallback). lift_off is how far the digit's farthest link point
   stands off the wall compared with its own closest one. It is relative (the same offset added to all three points
   changes nothing), reads no force and no joint angle, and treats every digit alike. Measured on the replayed held
   steps: digits bent at every joint along the wall 0.6-1.1 cm (curl 0.7-0.9), the last link laid flat with the other
   joints straight 1.9 cm (0.2), the hooked fingertip 2.0-2.5 cm (0.1-0.2). FOLLOW_DEAD is the resolution of the
   surface distance (it reads up to about 1 cm high at contact, and a joint point sits deeper inside its link than the
   fingertip point), not a pose target. Pressing harder buys nothing; moving the farthest link further off never raises
   the curl.

   Remove inner_share, press, CURL_FULL_SHARE and CURL_FORCE_REF. The inner-link force share must not come back in any
   form: not as a second factor, not as a summand, not as a gate. Do not read hand_q, hand_q_norm or any joint angle for
   this, do not add a force target, and do not name a digit.

   Size, measured by recomputing env_base with the round-18 code on the replayed held steps: at the switch today's
   policy loses 3-6 % of its downstream income (env_mult 0.565 -> 0.549 at epoch 800, 0.558 -> 0.523 at epoch 1000),
   because it earns about as much curl as before, now for the right digits. On the same held states a hand whose touching
   digits all follow the wall earns 78 % more than one where none does (curl_factor 1.0 against 0.55 for five hooked
   tips), and each digit that turns from a fingertip hook into a roll adds about 0.06 to curl_factor, about 8 % of
   everything earned while lifting, carrying and holding. The grasp, lift and hold chain keeps its order, because
   env_mult multiplies all of it alike and `held` / `hold_grip` do not see curl.

2. Keep everything else. Every structural rule stays: second half gated on the object's rise, the envelope quality as
   ONE multiplier on the downstream income with curl as a factor inside it, the ladder, palm and opposition inside
   env_base, no per-finger term, falling-only drop and set-down costs, no penalty on a persistent latch, open-pose factor
   only on the approach, palm seating tied to engagement, uprightness as a multiplier, first-success-only bonus, the
   goal-gated wrench, the round-18 table band. Every budget stays, CURL_FLOOR stays, the squeeze charge stays, and no
   speed or control charge is relaxed.

3. Do not answer the other measurements. Reaching around the object is not missing (the fingertip pads are past the
   widest point on 86-100 % of held steps), so do not add an arc or reach-around term. Do not pay more for links in
   contact than env_depth already does (the rolled ring finger presses mostly with its tip). The y exits of round 18 were
   the environment's bound and the operator has changed it; nothing in the reward names a world axis or a bound.

For reference, we trained an RL policy (PPO) with the most recent reward function above and tracked the individual reward components and some task metrics at 10 evenly spaced points during training, plus the min / mean / max encountered. Tags `reward/<name>` are your components (per-step mean over environments; `reward/total` is their sum as returned). The task metrics mean:
- contact/fingers_touching: number of fingers (0-5) with at least one measured link touching the cup (force > 0.1 N), averaged over environments and steps; contact/links_touching: number of measured links touching (0-15); contact/palm_touching: fraction of environments whose palm touches the cup; contact/finger_<name>: fraction of environments where that finger touches; contact/link_force_mean: mean link-cup force [N].
- contact/<metric>_at_success: the same quantity averaged only over the steps where a success was counted (a moving average over recent successes; -1 until the first success).
- ctrl/prev_ep_successes_mean: successes (0-5) reached in each environment's most recently finished episode, averaged over environments; task/successes_mean: successes so far in the running episodes; task/lifted_frac: fraction of environments whose cup has been lifted; task/tol: current success tolerance [m]; task/tilt_deg: cup tilt [deg].
- done/<reason>: fraction of environments ending an episode on a step for that reason (fell, tipped, out_xy, hand_floor, abnormal, max_goals); episode_lengths/step: mean episode length [steps]; rewards/step: mean return.

contact/finger_index: [0, 0.629, 0.653, 0.684, 0.712, 0.761, 0.718, 0.714, 0.709, 0.748]  min 0 · mean 0.687 · max 0.771
contact/finger_index_at_success: [-1, 0.969, 0.969, 0.969, 0.973, 0.976, 0.981, 0.981, 0.983, 0.991]  min -1 · mean 0.944 · max 1
contact/finger_middle: [0, 0.659, 0.675, 0.711, 0.737, 0.779, 0.74, 0.729, 0.724, 0.758]  min 0 · mean 0.709 · max 0.783
contact/finger_middle_at_success: [-1, 0.96, 0.98, 0.96, 0.978, 0.974, 0.969, 0.981, 0.98, 0.971]  min -1 · mean 0.943 · max 1
contact/finger_pinky: [0, 0.671, 0.669, 0.699, 0.72, 0.772, 0.733, 0.724, 0.72, 0.754]  min 0 · mean 0.702 · max 0.781
contact/finger_pinky_at_success: [-1, 0.98, 0.975, 0.967, 0.983, 0.984, 0.986, 0.97, 0.98, 0.987]  min -1 · mean 0.947 · max 1
contact/finger_ring: [0, 0.663, 0.669, 0.699, 0.719, 0.778, 0.724, 0.729, 0.731, 0.764]  min 0 · mean 0.704 · max 0.78
contact/finger_ring_at_success: [-1, 0.971, 0.99, 0.981, 0.987, 0.983, 0.984, 0.987, 0.988, 0.988]  min -1 · mean 0.955 · max 1
contact/finger_thumb: [0, 0.7, 0.699, 0.737, 0.763, 0.807, 0.772, 0.761, 0.759, 0.788]  min 0 · mean 0.738 · max 0.815
contact/finger_thumb_at_success: [-1, 1, 1, 0.999, 1, 1, 1, 1, 1, 1]  min -1 · mean 0.968 · max 1
contact/fingers_touching: [0, 3.32, 3.36, 3.53, 3.65, 3.9, 3.69, 3.66, 3.64, 3.81]  min 0 · mean 3.54 · max 3.92
contact/fingers_touching_at_success: [-1, 4.88, 4.91, 4.88, 4.92, 4.92, 4.92, 4.92, 4.93, 4.94]  min -1 · mean 4.82 · max 5
contact/link_force_mean: [0, 2.71, 3.24, 3.29, 3.43, 3.69, 3.54, 3.24, 3.31, 3.45]  min 0 · mean 3.23 · max 3.84
contact/links_touching: [0, 4.27, 4.34, 4.64, 4.74, 5.06, 4.82, 4.76, 4.76, 4.98]  min 0 · mean 4.6 · max 5.1
contact/links_touching_at_success: [-1, 6.16, 6.15, 6.26, 6.24, 6.23, 6.26, 6.22, 6.32, 6.33]  min -1 · mean 6.11 · max 6.39
contact/palm_touching: [0, 0.62, 0.58, 0.592, 0.631, 0.657, 0.62, 0.618, 0.626, 0.657]  min 0 · mean 0.611 · max 0.733
contact/palm_touching_at_success: [-1, 0.947, 0.892, 0.84, 0.871, 0.854, 0.857, 0.882, 0.898, 0.924]  min -1 · mean 0.845 · max 0.956
ctrl/prev_ep_successes_mean: [0, 2.28, 3.11, 3.15, 3.35, 3.22, 3.19, 3.26, 3.26, 3.22]  min 0 · mean 2.93 · max 3.39
done/abnormal: [0, 0.000488, 0.000977, 0.000244, 0.000244, 0.000244, 0.000732, 0.000488, 0.000244, 0.000244]  min 0 · mean 0.000495 · max 0.0022
done/fell: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 1.11e-06 · max 0.000244
done/hand_floor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 5.54e-07 · max 0.000244
done/max_goals: [0, 0.000977, 0.000977, 0.000732, 0.000732, 0.000244, 0.000244, 0.000732, 0.000244, 0.000977]  min 0 · mean 0.000659 · max 0.00366
done/out_xy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  min 0 · mean 2.27e-05 · max 0.000488
done/tipped: [0, 0, 0, 0, 0.000244, 0, 0.000244, 0, 0, 0.000244]  min 0 · mean 0.000183 · max 0.00122
episode_lengths/step: [22, 316, 498, 453, 510, 607, 629, 553, 624, 586]  min 22 · mean 514 · max 669
reward/close: [2.6e-06, 0.00519, 0.00662, 0.00829, 0.00706, 0.0082, 0.00863, 0.00818, 0.0077, 0.0078]  min 2.6e-06 · mean 0.00742 · max 0.00969
reward/goal_hold: [0, 0.684, 0.622, 0.662, 0.698, 0.746, 0.699, 0.69, 0.712, 0.708]  min 0 · mean 0.671 · max 0.758
reward/goal_progress: [0, 0.0453, 0.0453, 0.0492, 0.0538, 0.042, 0.0494, 0.0564, 0.0611, 0.0528]  min 0 · mean 0.0491 · max 0.0931
reward/goal_reach: [0, 0.188, 0.17, 0.181, 0.195, 0.222, 0.214, 0.208, 0.217, 0.232]  min 0 · mean 0.198 · max 0.239
reward/grasp_count: [0, 0.0507, 0.0491, 0.0514, 0.0541, 0.0569, 0.0543, 0.0531, 0.0538, 0.0561]  min 0 · mean 0.052 · max 0.0579
reward/grasp_digit: [0, 0.0513, 0.0509, 0.0547, 0.0556, 0.0592, 0.0565, 0.0556, 0.056, 0.0586]  min 0 · mean 0.054 · max 0.0598
reward/grasp_oppose: [0, 0.0924, 0.0924, 0.097, 0.101, 0.107, 0.101, 0.0995, 0.0989, 0.104]  min 0 · mean 0.0973 · max 0.108
reward/grasp_palm: [0, 0.185, 0.174, 0.177, 0.189, 0.197, 0.186, 0.185, 0.188, 0.197]  min 0 · mean 0.183 · max 0.217
reward/lift_height: [0, 0.616, 0.579, 0.612, 0.656, 0.708, 0.663, 0.656, 0.667, 0.67]  min 0 · mean 0.629 · max 0.714
reward/lift_rise: [0, 0.00498, 0.00622, 0.00665, 0.00668, 0.00401, 0.00541, 0.00643, 0.00556, 0.00447]  min 0 · mean 0.00577 · max 0.0218
reward/palm_in: [0.000317, 0.00162, 0.00208, 0.00246, 0.00217, 0.0024, 0.00258, 0.00246, 0.00234, 0.00234]  min 0.000317 · mean 0.00227 · max 0.00286
reward/pen_action: [-0.0219, -0.0236, -0.0232, -0.0232, -0.0229, -0.023, -0.0229, -0.0229, -0.023, -0.0229]  min -0.0243 · mean -0.023 · max -0.0208
reward/pen_arm_fast: [0, -0.00394, -0.00516, -0.00481, -0.00405, -0.00609, -0.00612, -0.00426, -0.00588, -0.00559]  min -0.00854 · mean -0.00515 · max 0
reward/pen_arm_vel: [-0.0271, -0.0534, -0.0541, -0.0539, -0.0575, -0.0573, -0.0576, -0.0571, -0.0574, -0.0591]  min -0.0665 · mean -0.0557 · max -0.0259
reward/pen_cup_vel: [0, -0.0346, -0.0342, -0.0339, -0.0379, -0.0383, -0.0383, -0.0375, -0.0379, -0.0399]  min -0.0499 · mean -0.0362 · max 0
reward/pen_descend: [0, -0.00879, -0.0102, -0.00948, -0.0108, -0.0126, -0.0123, -0.0129, -0.0126, -0.0171]  min -0.0196 · mean -0.0123 · max 0
reward/pen_drift: [0, -0.0612, -0.0543, -0.0563, -0.0609, -0.072, -0.0706, -0.0631, -0.0668, -0.0772]  min -0.0848 · mean -0.0634 · max 0
reward/pen_drop: [-0.000609, -0.0118, -0.0157, -0.0163, -0.0191, -0.0214, -0.0201, -0.0201, -0.0221, -0.0206]  min -0.0284 · mean -0.0187 · max -6.47e-05
reward/pen_setdown: [-0.000145, -0.00283, -0.00289, -0.00297, -0.00335, -0.00336, -0.00308, -0.0035, -0.00384, -0.00506]  min -0.00601 · mean -0.00359 · max -2.35e-05
reward/pen_shove: [-0.00782, -0.0176, -0.0133, -0.0155, -0.0144, -0.0119, -0.0136, -0.0138, -0.0135, -0.0128]  min -0.0447 · mean -0.0147 · max -0.00722
reward/pen_squeeze: [0, -0.169, -0.191, -0.195, -0.194, -0.215, -0.199, -0.19, -0.184, -0.196]  min -0.216 · mean -0.186 · max 0
reward/pen_stall: [0, -0.000218, -0.000158, -0.000176, -0.000194, -0.000138, -0.000165, -0.000217, -0.000219, -0.000286]  min -0.00129 · mean -0.000213 · max 0
reward/pen_table: [0, -3.18e-06, 0, -2.82e-05, -0.000275, -0.000378, -0.000458, -0.000319, -0.000657, -0.000213]  min -0.00281 · mean -0.00038 · max 0
reward/pen_tilt: [-0.000554, -0.0319, -0.0354, -0.0368, -0.035, -0.0352, -0.0349, -0.0369, -0.0344, -0.0386]  min -0.0516 · mean -0.0351 · max -0.000249
reward/reach: [0.0159, 0.00593, 0.00718, 0.00705, 0.0068, 0.00637, 0.00759, 0.0073, 0.00718, 0.00669]  min 0.0047 · mean 0.0072 · max 0.0196
reward/success_bonus: [0, 0.0591, 0.0791, 0.0824, 0.0357, 0.0212, 0.048, 0.0498, 0.118, 0.0454]  min 0 · mean 0.0642 · max 0.272
reward/total: [-0.042, 1.57, 1.44, 1.54, 1.6, 1.68, 1.62, 1.61, 1.73, 1.65]  min -0.0423 · mean 1.57 · max 1.84
rewards/step: [-21.4, 347, 725, 673, 764, 968, 1.03e+03, 905, 1.05e+03, 976]  min -23.7 · mean 793 · max 1.11e+03
task/lifted_frac: [0, 0.646, 0.64, 0.677, 0.694, 0.758, 0.719, 0.698, 0.705, 0.735]  min 0 · mean 0.678 · max 0.761
task/successes_mean: [0, 1.36, 1.35, 1.46, 1.45, 1.48, 1.41, 1.42, 1.42, 1.37]  min 0 · mean 1.36 · max 1.6
task/tilt_deg: [0.084, 4.08, 4.48, 4.68, 4.45, 4.48, 4.44, 4.66, 4.37, 4.88]  min 0.0303 · mean 4.46 · max 6.71
task/tol: [0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09, 0.09, 0.09, 0.081]  min 0.081 · mean 0.0929 · max 0.1
