iter_13 trained only Hypothesis A (iter_13/selection.json, candidate c1r). Run cg_l_i13, left arm, vision-3090, fresh, seed 42, 4096 envs, default finger action mapping (the open pose is at the action limit: -1 for the flexion joints, +1 for the thumb), agent bounds_loss_coef 0.0001, entropy_coef 0.

Measured from the training logs (per-joint flexion log, 0 = open default pose):
- The open-hand approach was learned again (all 13 movable joints 0.00-0.01 by epoch 212) and the hand reached the object: reach 0.98 at epoch 610, mean palm-to-surface distance 5.7-5.8 cm at epochs 808-1010, 70 % of steps in the contact phase.
- Flexion in the contact phase stayed 0.00 on all 13 joints and the closing income stayed 0.000 up to epoch 1010; digit contact 0.16, palm contact 0.14, grasp 0, lift 0. iter_12 (same approach reward, no near-object closing credit) ended the same way at epoch 2906 (digit contact 0.43, palm contact 0.39, flexion 0.00, grasp 0).
- The action bounds loss rose 17.8 -> 33 -> 43 -> 48 (epochs 212-1010).

Measured from the checkpoints (policy run on CPU over the 4096 stored observations and LSTM states):
- The finger action means of all 13 movable joints sit far beyond the open-side action limit: flexion joints -2.4 to -3.2, thumb +3.1 (iter_12: -1.8 to -2.8 and +2.1 to +2.2); action noise sigma 0.55-0.67. The probability that a sampled finger action lands inside the range (i.e. actually curls the finger) is 0.04-0.7 % per joint. The spread of these means across the 4096 environments is 0.1-0.3, so the finger command does not depend on the state (far from the object or 5 cm from it).
- Bounds loss recomputed from these means: 46.4 (log 43-48); iter_12: 25.5 (log 24-27).
- Mechanism: the open-hand income and the finger-target cost are maximal exactly at the open pose; samples beyond the limit are clipped to the open pose and never cost anything, samples inside the range cost, so moving the mean further beyond the limit always pays and the bounds loss coefficient barely resists. The near-object closing credit of iter_13 (curling at 5/4/3 cm raises the reward by +0.049/+0.068/+0.095) is therefore never sampled.
