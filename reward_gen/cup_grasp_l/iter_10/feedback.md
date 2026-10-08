## I can see from the robot that
iter_10 (the iter_09 code plus the origin-relative approach) was trained on the right arm only (cg_r_i08, local RTX 5090, fresh, seed 42) and stopped by the operator at about 800 epochs.

The origin-relative approach did what it was meant to: the palm is within the graspable height band around the object origin 0.87 of the time at epoch 635-735 (iter_09 and the iter_02 right-arm run: 0.00-0.01 at the same epochs), and in the deterministic replay the hand no longer goes to the mouth of the object.

But the hand travels as a fist. Operator verdict on the replay: the fingers are already curled BEFORE the approach. Frames: from 0.5 s the fingers are closed into a fist; the hand arrives beside the object at body height after about 3 s and stays there with the fist against the side until the end, for both the large and the small cup; nothing is lifted.

Last 100 epochs (iter_10 / iter_02 right-arm run / iter_09, same window): palm to the object surface 8.4 / 5.5 / 5.7 cm, proximal links touching 0.00 / 0.47 / 0.40, digits touching 0.04 / 0.16 / 0.41, grasp 0.00 / 0.07 / 0.25, finger curl of the whole hand 0.61 / 0.43 / 0.52, open-hand pose condition 0.01 / 0.15 / 0.01.

The episode starts with the hand in its open reset pose. With the current finger action mapping, action 0 is the middle of each joint's range, so the fingers curl unless the policy keeps commanding the open side. In the iter_10 code the palm-seating income is multiplied by an engage factor that grows with the flexion of ALL fingers, touching or not (0.5 x mean flexion + 0.5 x grip, floor 0.35), so a fist beside the object earns more palm seating than an open hand at the same place (deterministic check: palm 3 cm from the surface, nothing touching: open 0.033, curled 0.057).

## Feedback for improvement
Keep the previous code (iter_10 = the iter_02 code plus the per-digit wrap envelope plus the origin-relative approach) exactly as it is, and change only what Hypothesis A says, identically for the mirrored arm. Do NOT bring back any other change from iters 03-08.

Operator requirement for the whole reward (unchanged): every income must connect smoothly to the next one, without a flat stretch or a step at any boundary; moving a little towards the next stage must always earn more than staying.

## Hypothesis A
- Finger flexion must not raise the palm-seating income while nothing touches: in the palm-seating engage factor, count the flexion only of digits that are actually touching the object (continuous in their contact), so that curling the fingers in the air before or during the approach earns nothing, and the hand is best off keeping the open start pose until its fingers reach the object.
