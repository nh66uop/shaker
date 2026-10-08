## I can see from the robot that
iter_08 was trained with the finger action re-centred (action 0 = open pose) on both arms from random weights (right 1667 epochs, left 1317 epochs) and stopped by the operator. The hand approaches open (finger curl 0.32-0.41, open-hand income 0.06) but the palm stops 6.9-7.0 cm from the object, digits touching 0.01-0.03, grasp 0.00. The operator watched the right-arm replay and judged: the fingers move about constantly, the object never comes in between the thumb and the other fingers, the fingers keep knocking the cup, and the arm only learned to tilt the hand and stay beside it.

Across the whole cup_grasp history, at epoch 1150-1250 (same window for every run):

| run | open hand while approaching | palm to surface | digits touching | grasp (episode) | lift at the end of the run |
|---|---|---|---|---|---|
| right, iter_02 code | 0.20 | 4.1 cm | 1.90 | 0.94 | 0.94 |
| left, iter_01 code | 0.28 | 5.5 cm | 1.06 | 0.73 | 0.70 |
| left, iter_02 code | 0.11 | 4.1 cm | 1.39 | 0.86 | 0.01 |
| iter_04 (both arms) | 0.01-0.07 | 5.0-6.1 cm | 0.02-0.03 | 0.01 | 0 |
| iter_06 | 0.01 | 6.7-8.6 cm | 0.07-0.14 | 0.00 | 0.01 |
| iter_07 | 0.01 | 7.1 cm (left collapsed) | 0.16 | 0.00 | 0 |
| iter_08 | 0.01-0.02 | 6.8-7.1 cm | 0.02 | 0.00 | 0 |

So the task was solved with the iter_02 reward (right arm: straight open-hand approach, grasp by the finger segments, lift, carry, hold) and broke from iter_04 on; iter_05 (which undid the two iter_04 causes) was stopped after 285 epochs and never evaluated, and iters 06-08 each stacked another change on top. The finger action re-centring is removed again for the next round (the environment goes back to the mapping iter_02 was trained with).

The one defect the operator named for the iter_02 reward is fingertip pressing after the lift. Measured in the right-arm run of the iter_02 code, at the moment of success (last 200 epochs): thumb 1.00, ring 0.96, pinky 0.99, palm 0.83 touching, but index 0.014 and middle 0.001 (the middle finger was 0.89 at epoch 1500 and let go later); wall-following of the links at success (1 = links lie along the wall): index 0.19, middle 0.37 against ring 0.88, pinky 0.98. In the iter_02 code, the envelope quality that multiplies the transport, hold and success incomes ranks the digits and gives the best-engaged digit 0.75 of the digit-count weight and the two weakest 0.02 each, so two digits leaving the wall after the lift costs almost nothing.

## Feedback for improvement
START FROM THE iter_02 CODE (shown in the history above as iter_02), not from the iter_08 code: every change made in iters 03-08 is dropped. Keep the iter_02 code exactly as it is, including its approach, closing, stall, fade and open-hand terms, and change only what Hypothesis A says, identically for the mirrored arm.

Operator requirement for the whole reward (unchanged): every income must connect smoothly to the next one, without a flat stretch or a step at any boundary; gates must be continuous in the measured quantities, and moving a little towards the next stage must always earn more than staying.

## Hypothesis A
- Keep the digits wrapped after the lift: while the object is lifted, carried and held, the grasp quality that scales the transport, hold and success incomes must depend on every digit keeping its segments (proximal and middle links, not only the fingertip pad) on the object with its links following the wall, and the weakest digits must count as much as the best one, so that a hold in which one or two digits let go or press only with the pad earns clearly less than the same hold with all five digits wrapped.
