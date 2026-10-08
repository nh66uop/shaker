iter_07 was trained on both arms from random weights (right 1490 epochs, left 1420 epochs, same code, 4096 environments, seed 42) and stopped by the operator. Numbers are means over the last 100 epochs.

| measure | iter_07 right | iter_07 left | completed runs, same window |
|---|---|---|---|
| reached the object (episode) | 0.98 | 0.00 | 0.97-0.98 |
| palm to the object surface, cm | 6.9 | 24 | 3.5-5.2 |
| digits touching | 0.23 | 0.00 | 1.5-2.8 |
| palm touching | 0.00 | 0.00 | 0.33-0.49 |
| grasp (episode) | 0.00 | 0.00 | 0.88-0.97 |
| measured finger curl of the whole hand | 0.55 | 0.37 | 0.40-0.42 |
| open-hand pose condition during the approach | 0.01 | 0.01 | 0.18-0.26 |
| open-hand income (ceiling 0.10) | 0.010 | 0.001 | - |

- The right arm stalled where iter_06 stalled: the palm stays about 7 cm from the object and the hand is not opened; the open-hand income it collects is a tenth of its ceiling.
- The left arm collapsed at epochs 650-700: the policy's action means left the action range (the rl_games bounds loss rose from 1.5 to 681, KL spiked to 0.15, the adaptive learning rate fell from 6e-4 to 3e-5), the arm backed away from 9 cm to 38 cm and did not recover in 700 epochs. The left arm of iter_06 had the same event at epoch 800 (bounds loss up to about 1000) and recovered 500 epochs later.

Cause found in the environment (verified in the code and the logged configuration): each finger joint is commanded as an absolute target, raw = lo + 0.5 (a + 1)(hi - lo), so action 0 is the middle of the joint range, and the reset pose that the open-hand measure compares against (hand_default_q_norm) lies at the joint's lower limit, i.e. at action -1. A random policy therefore starts with the fingers half curled (0.49-0.54 measured at epoch 134), and holding the hand open means pushing the mean of 13 finger actions to the edge of the action range, which is exactly what drives the bounds loss.

Operator decision, applied to the environment for the next round: the finger action is re-centred so that action 0 is the reset (open) pose (per joint: a >= 0 moves from the reset pose to the upper limit, a < 0 from the reset pose to the lower limit; for joints whose reset pose is the lower limit, negative actions all give the open pose). The range, the smoothing and the observations are unchanged. In a short check with this mapping the fingers start at a curl of 0.30-0.37 instead of 0.41-0.49; they do not start at 0 because exploration noise on the positive side still curls them.
