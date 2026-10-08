## I can see from the robot that
iter_03 was not trained (its reward was generated and checked, then revised before launch); iter_02 is the last trained round of this arm.

The operator watched the final left-arm videos of iter_02 (deterministic replay, 12 environments, largest cup included) and judged: the index finger and the thumb are higher than the object's rim, so they close on air above the rim. This is the same defect the operator named in an earlier round of this task ("the hand cannot get below the rim, so index and thumb close on air beside it").

In the iter_03 code:
- the segment pull pays for bringing a touching digit's proximal and middle links close to the NEAREST surface point, whatever that point is; when the digit sits above the object, the nearest point is the rim edge or the top face, whose normal points up, and the pull still pays in full.
- the closing term pays flexion times link nearness, and the nearness keeps 0.25 of its value on a rim or top face (it is weighted 0.25 + 0.75 x how horizontal the surface normal is).
- a flexed digit that touches nothing costs nothing.

Earlier rounds of this task found (measured): weighting the long-range reach by how horizontal the surface normal is made the arm retreat (palm gap 0.22 -> 0.37 m), because from the start pose the nearest point of an open cup is its rim; weighting only from a few centimetres in did not cause that.

## Feedback for improvement
Keep everything in the previous code (the height-based fade of the pre-lift incomes, the stall cost and palm-seat bonus that do not depend on digit count, the segment-based digit counting, the segment pull, the palm-seat bonus, the shove relief). Change only what Hypothesis A says, and keep it identical for the mirrored arm.

## Hypothesis A
- Close in only (within a few centimetres of the object, never on the long-range reach): weight the segment pull and the closing term's link nearness by how horizontal the object's surface normal is at each segment's nearest surface point, so that pulling a digit towards a rim edge or a top face pays almost nothing and only the side wall pays.
- A digit that is flexed while its segments' nearest surface is a rim edge or a top face (or while it touches nothing) must earn no closing credit, so that closing the index finger or the thumb above the rim is worth nothing.
