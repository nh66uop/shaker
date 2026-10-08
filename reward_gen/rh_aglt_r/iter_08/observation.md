iter_08 trained c1r (reward_gen/rh_aglt_r/iter_08/selection.json); no competing hypothesis. Both arms trained this reward byte-identically.

Runs (same reward, same seed 42, 4096 envs):
- aglt_r_i08 / aglt_l_i08 (no command delay, epochs 0-423 / 0-226, stopped by the user to add the real-robot response to the environment): right palm within 10 cm 0.61 at epoch ~400.
- aglt_r_i08b / aglt_l_i08b (arm command delay 150-200 ms and hand command delay 50-83 ms from the start, stopped at epochs ~2250 / ~1300): right palm within 10 cm 0.00 at epoch 400 and 0.62 at epoch 2000, grasp flag 0.001-0.003 for ~1000 epochs with an open hand near the cup; left never within 10 cm. The delay is now switched off for this stage and will be added after the task is learned (user decision).
- aglt_r_i08c (no command delay, stopped at epoch ~1440): within 10 cm 0.80 at epoch ~400, grasp flag rising from epoch ~710, best window epochs 1100-1300: grasp flag 0.59-0.70, lifted 0.50-0.62, goal successes 0.34-0.50 per episode (training tolerance 0.10 m); palm 1.8-2.0 cup radii from the axis while held, thumb tip 1.32 cup radii at 0.71-0.81 of the mouth height (outer wall), thumb tip inside the mouth 0.001 of held steps. Then it COLLAPSED within ~100 epochs (epochs 1300-1390): within 10 cm 0.82 -> 0.02, grasp 0.70 -> 0.02, palm-to-cup distance 0.09 -> 0.44 m, with no early terminations (episode length 896 of 900), no NaN, KL 0.016-0.027, entropy 10.8 -> 10.4.
- aglt_l_i08c (no command delay, epoch ~900 when stopped for this round): grasp flag 0.49, lifted 0.42, goal successes 0.77; wrist joint 6 at its limit 0.00-0.10 of steps.

What the collapse was (deterministic replay of the right-arm checkpoints, 64 envs x 1800 steps, pre-clip policy output recorded):
- ctx.arm_limit_margin (1 = mid-range, 0 = at the joint limit). "At the limit" below = margin < 0.05.
- Epoch-1200 checkpoint (best): grasp flag 0.97 and lifted 0.94 in the middle of each episode. While carrying the cup to the goals the wrist joint 6 is at its limit 0.80 of the steps (0.13 at the start of an episode, 0.96 at the end); joints 4 and 5 at their limits 0.30 / 0.24. The pre-clip policy output exceeds +-1 on almost no step.
- Epoch-1400 checkpoint (collapsed): from the first step the shoulder joint 2 is at its limit 0.72 of the steps and the wrist joint 6 0.93, the median policy output pushes both further into the limits (-0.93 and +0.97), and the palm drifts 0.3-0.6 m away from the cup. The pre-clip output still does not exceed +-1.
- Order in the training logs: joint 2 at its limit rose 0.05 -> 0.20 while success was still at its best (epoch 1270-1300), then joint 6 0.41 -> 0.91 and the approach broke.
- With joint-increment control the joint target is clamped at the limit, so any action pushing further has the same outcome; the reward paid nothing for keeping joint range, so nothing pulled the arm back out.
- The good grasp also uses the wrist limit: the mirrored left policy (trace good_mirror) holds the wrist joint 6 at its limit on 0.63 of its held-and-lifted steps (median margin there 0.00; other joints 0.30-0.74). Before the grasp it stays off the limits (median of the smallest margin 0.16).
- The far start pose itself has a smallest margin of 0.075 (joint 1); per joint 0.075, 0.216, 0.584, 0.559, 0.759, 0.263, 0.577.

Deterministic ledger on the iter_08 reward with new joint-limit rows: every old row passes; the reward is identical (1.00x) whether a joint is at its limit or not, at the start, halfway and held at the goal.
