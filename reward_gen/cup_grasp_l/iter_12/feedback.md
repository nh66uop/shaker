## I can see from the robot that
iter_12 trained only Hypothesis A (iter_12/selection.json, candidate c1r). Run cg_l_i12, left arm, local, fresh, seed 42, 4096 envs, default finger action mapping.

Measured from the training logs (per-joint flexion log, 0 = open default pose, 1 = full flexion; last-30-epoch means):
- The open-hand approach was learned: mean flexion during the approach phase fell 0.37 (epoch 0) -> 0.21 -> 0.06 -> 0.00 by about epoch 300, and all 13 movable finger joints read 0.00 during the approach from then on (iter_11 at epoch 350-565, same code without this change: 0.55-0.57, index joint 4 0.88, ring joint 3 0.86, pinky joint 3 0.82).
- The hand then stops at the object with the fingers open and does not start closing. At epoch 806: reach 0.99, mean palm-to-surface distance 7.1 cm, 59 % of steps in the contact phase (palm within 5 cm of the surface or touching), yet flexion in the contact phase is 0.00 on all 13 joints; digit contact 0.04, palm contact 0.03, grasp 0.00, lift 0.00.
- Policy entropy fell 36.3 -> 28.7 -> 24.4 (epochs 108, 420, 806); open-hand income 0.29 per step (its ceiling is 0.30).
- The action bounds loss rose from 0.5 to 23-25 and stayed there (iter_11: about 2).

Deterministic evaluation of the iter_12 reward (float64, one environment, palm facing the side wall, finger links 2 cm further out than the palm, no contact), total reward for curl 0 / 0.3 / 0.6:
- palm 7 cm from the surface: +0.457 / +0.244 / +0.177 (curling costs up to 0.28)
- 5 cm: +0.478 / +0.300 / +0.244 (costs 0.23)
- 4 cm: +0.490 / +0.349 / +0.306 (costs 0.18)
- 3 cm: +0.503 / +0.424 / +0.406 (costs 0.10)
- 2 cm: +0.517 / +0.550 / +0.579 (pays 0.06)
- 1 cm: +0.532 / +0.586 / +0.633 (pays 0.10)
The loss comes from the open-hand income (0.30 open -> 0.09 at 5 cm with curl 0.6) and the finger-target cost; the closing credit at 5 cm is only 0.003 because it is multiplied by the palm seat measure, which is near zero until the palm is within about 2 cm of the surface.

## Feedback for improvement
Keep the previous code (iter_12) exactly as it is, and change only what Hypothesis A says, identically for the mirrored arm. Do NOT bring back any change from iters 03-08, do NOT change the finger action mapping, and keep the open-hand approach that iter_12 taught.

Operator requirement for the whole reward (unchanged): every income must connect smoothly to the next one, without a flat stretch or a step at any boundary; moving a little towards the next stage must always earn more than staying. Operator: "when the hand is near the object it must be able to grasp" and "if the hand is already gathered while approaching, it can never learn the next stage."

## Hypothesis A
- The hand now arrives open and parks a few centimetres in front of the object: from 7 cm to 3 cm palm-to-surface distance, curling the fingers costs 0.10-0.28 per step, and only inside 2 cm does it pay. Make closing towards the object pay from where the fingers can reach the wall: release the open-hand requirement and the finger-target cost smoothly over a wider band of palm distance (fully released by about 3-4 cm from the surface, still essentially un-released at 10 cm), and let the pre-contact closing credit grow with how near the finger links (proximal and middle segments) are to the side wall, rather than only once the palm is seated, so that it already rises from about 5 cm.
- Targets: at 3, 4 and 5 cm with no contact, curling the fingers towards the wall must never lower the reward (and should raise it at 3-4 cm); at 7 and 10 cm with no contact, the open hand must still beat the measured hooked hand by at least 0.08; flexion in the air far from the object must not pay (the closing credit needs the finger links near the wall).
