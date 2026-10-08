iter_14 trained only Hypothesis A (iter_14/selection.json, candidate c1). Run cg_l_i14, left arm, local, fresh, seed 42, 4096 envs, default finger action mapping, agent bounds_loss_coef 0.005 (a parallel run with the iter_13 reward and the same coefficient, cg_l_i13b, touched the object but never lifted it in 3586 epochs and was stopped).

Measured from the training logs (last-30-epoch means at epoch ~6000):
- The full task now works from random initial poses: first lift near epoch 3900; lifted fraction of steps 0.67, lift within the episode 0.30, success within the episode 0.21 (goal 1/2/3 reached 0.24/0.20/0.15), successes per episode 0.89. At success: 4.5 digits and 0.58 palm contact. Action bounds loss stayed 0.2-1.2 the whole run; policy entropy 4.4.
- Operator video verdict (best checkpoint, two environments): the grasp is the wanted envelope grip (not a hook), but the fingers wrap the UPPER part of the cup from above; the wrap must sit lower.

Measured from the epoch-6050 checkpoint (4096 stored observations; 1106 environments with the object origin within 12 cm of the palm; heights along the cup axis relative to the object ORIGIN, median and quartiles):
- palm centre +5.6 cm (+4.6 to +6.7), thumb tip +8.4 cm (+7.2 to +9.5), index tip +9.5 cm (+8.1 to +10.9), middle tip +7.2 cm, ring tip +4.9 cm, pinky tip +2.2 cm; palm distance from the axis 8.9 cm.
- For reference (asset data, not available to the reward): the object origin of these cups is 6.6-8.9 cm above the base and the rim is 8.5-11.5 cm above the origin, so the thumb and index tips are at rim height.
- The origin-relative palm-seating income (which prefers the palm level with the origin) is multiplied by (1 - held), so once the object is held nothing in the reward prefers one grasp height over another.
