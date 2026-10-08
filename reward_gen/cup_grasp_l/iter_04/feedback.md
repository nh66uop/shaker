## I can see from the robot that
iter_04 was trained on both arms from random weights (left 3292 epochs, right 3549 epochs, same code, 4096 environments, seed 42) and stopped by the operator. The arm reaches the object and then parks in front of it without touching it. Numbers are means over the last 100 epochs; the comparison is iter_02, the last round that lifted (same arm, same epoch window).

| measure | iter_04 left | iter_02 left | iter_04 right | iter_02 code on right |
|---|---|---|---|---|
| reached the object (episode) | 0.94 | 0.77 | 0.83 | 0.97 |
| reached but touching nothing now | 0.98 | 0.24 | 0.99 | 0.15 |
| digits touching the object | 0.03 | 1.9 | 0.01 | 2.7 |
| palm touching | 0.00 | 0.57 | 0.00 | 0.63 |
| grasp (episode) | 0.05 | 0.89 | 0.008 | 0.95 |
| lift (episode) | 0.00 | 0.01 | 0.00 | 0.92 |
| nearest link to the surface, cm | 3.7 | 2.5 | 3.3 | 2.4 |
| palm to the surface, cm | 4.0 | 3.5 | 3.8 | 3.5 |
| measured finger flexion (0 open, 1 closed) | 0.46 | 0.31 | 0.45 | 0.37 |

Per-term reward per step, iter_04 (left / right): reach 0.119 / 0.119, palm seating 0.102 / 0.101, open hand 0.007 / 0.005; every contact-gated term (closing, digit count, digit segments, palm contact, palm seat bonus, opposition, segment pull) is below 0.001; stall cost 0.006 / 0.004. So the parked state earns about 0.22 per step with almost no cost. In iter_02 palm seating was 0.03-0.07 and the contact-gated terms together were 0.26-0.63.

The fingers are not open while parked: flexion is higher than in iter_02, i.e. the hand curls in the air 3-4 cm in front of the object.

What changed from iter_02 to iter_04 in the code (both measured runs above use one or the other):
- the closing term now pays only while a digit touches the object; in iter_02 it paid flexion times link nearness to the surface before any contact.
- the stall cost is now charged in full as soon as any single digit or the palm touches (gate = the largest single contact); in iter_02 its gate rose with how much of the hand was on the object (grip and hold).
- close in (inside 3.5 cm), the closing nearness and the segment pull are multiplied by how horizontal the surface normal is, down to 0.05 on a rim or top face.
- the palm-seating income grows with finger flexion whether or not anything touches (unchanged from iter_02, but in iter_04 it is the largest income left near the object).
- the pre-lift incomes fade with the object's height instead of with the hold gate.

Entropy fell from 27-28 to 23-24 over the last 1000 epochs while parked.

## Feedback for improvement
Keep everything else in the previous code (the height-based fade of the pre-lift incomes, segment-based digit counting, palm contact, opposition, the lift, transport and hold incomes, the rim guard's purpose). Change only what Hypothesis A says, and keep it identical for the mirrored arm.

Operator requirement for the whole reward: every income must connect smoothly to the next one, so the gradient carries the policy from travelling, to touching, to closing, to holding, to lifting, to carrying, without a flat stretch or a step at any boundary. No term may switch from zero to full at a single contact or at a threshold; gates must be continuous in the measured quantities (distance, force, flexion, height), and each stage's income must already be rising where the previous stage's income levels off. The state reached at the end of one stage must earn strictly more by moving a little towards the next stage than by staying.

## Hypothesis A
- Make the first touch pay instead of cost: the stall cost must rise gradually with how much of the hand is on the object (as in iter_02), never reach its full size at a single contact; and closing credit must again start before contact, paying flexion times nearness of the links to the object, so that curling the fingers draws them onto the object instead of stopping in front of it. Flexion must not raise the palm-seating income while nothing touches.
- Keep the rim guard only on contact: the side-wall preference applies to segments that are actually touching (a touch on a rim edge or top face earns little closing or pull credit), not to the approach distance band, so that coming the last few centimetres towards any part of the object is never worth less than parking in front of it.
