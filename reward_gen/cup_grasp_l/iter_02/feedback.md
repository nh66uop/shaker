## I can see from the robot that
iter_02 trained only Hypothesis A (iter_02/selection.json), on both arms with the identical file (right = cup_grasp_r/iter_01).

Right arm (4,828 epochs): grasp 0.95-0.97, lift 0.92-0.95, far-start success 0.69 at tolerance 0.039. The grasp is now carried by finger segments: 3.5-3.6 digits touch with a segment, the index and middle fingers touch with their proximal links (0.66, 0.68), palm contact 0.66-0.71. Lift started at epochs 1,500-2,000, exactly when the segment-weighted digit count passed 2 (1.24 -> 2.02).

Left arm (4,342 epochs, stopped): grasp 0.85-0.90 from epoch 1,000 on, but lift 0.005-0.017 for 3,300 epochs; success 0; envelope 0.03. Only some digits touch: middle 0.56-0.66, pinky 0.43-0.48, thumb 0.34-0.40 (with its proximal link, which does not flex in this hand), ring 0.00 until epoch 2,000 then 0.30, index 0.03-0.07. Segment-weighted digit count stayed 1.13-1.40. Palm contact 0.39-0.57, palm gap 0.067-0.074 m. Hand actions saturated at +-1 in 0.82 of steps (right 0.73); bounds loss 27.6 (right 16.4).

Measured on hand-built states with the iter_02 code (object on the table, palm seated, middle and pinky touching with segments, thumb touching with its proximal link):
- closing the ring finger onto the object with a segment changes the per-step reward from +0.954 to +0.670 (-0.285); closing ring and index gives +0.705 (-0.250); lifting the object 3 cm instead gives +1.945 (+0.990).
- what falls when the third digit closes: reach 0.138 -> 0, palm seating 0.100 -> 0, closing 0.036 -> 0 (all multiplied by one minus the hold gate), pre-hold palm-seat bonus 0.140 -> 0.093, stall cost -0.029 -> -0.046; what rises: grasp ladder, digit and segment pull together +0.052.
- the previous round's code (iter_01) had the same drop, smaller: -0.140.
So on the table, completing the grasp costs reward unless the object is lifted in the same moment; the right arm found lifting first, the left arm did not and stayed with two to three digits.

## Feedback for improvement
Change only what Hypothesis A says. Keep the segment-based digit counting, the segment pull, the pre-hold palm-seat bonus and the shove relief. The reward must stay identical for the mirrored arm.

## Hypothesis A
- The incomes that pay for getting to the object and seating the hand (the long-range reach, the palm seating and the closing term) must not disappear when the grasp completes on the table. Fade them with how far the object has actually risen above its start height (or with the object being off the table), not with the hold gate, so that on the table closing one more digit onto the object always raises the per-step reward. Today closing a third digit costs -0.285 per step.
- Neither the stall cost nor the pre-hold palm-seat bonus may turn a more complete grasp on the table into a loss: the stall cost must not grow with the number of digits holding the object, and the bonus must not shrink when more digits hold. Lifting must remain the only way to increase income clearly once the grasp is complete.
