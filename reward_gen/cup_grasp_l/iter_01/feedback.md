## I can see from the robot that
iter_01 trained only Hypothesis A (iter_01/selection.json) on the left arm; the right arm trained the iter_00 reward, which differs only in the two iter_01 changes.

Both arms learned the whole task from random weights (left 7,211 epochs, right 13,679 epochs; last 500 epochs):
- right: grasp 0.90, lift 0.88, far-start success 0.58 at tolerance 0.023, envelope 0.81, palm contact 0.73, thumb/index/middle/ring 0.84-0.86, pinky 0.33, disturbance at its maximum (20 N/kg).
- left: grasp 0.81, lift 0.77, far-start success 0.34 at tolerance 0.026, envelope 0.66, palm 0.70. Around epoch 6,000 it had grasp 0.94, lift 0.92 and success 0.61; in its last 1,200 epochs the pinky started touching (0.005 -> 0.43) and grasp, lift and success fell (0.94 -> 0.76, 0.92 -> 0.70, 0.61 -> 0.23) at a constant tolerance.

What the operator saw in the final videos (deterministic replay, 12 environments): on both arms the last finger joints (joint 4) are still curled far too much; the right hand approaches naturally and lifts; the left hand approaches naturally and then lifts abruptly.

Measured shape of the grasp:
- Closure of the last finger joints is 0.72 (right) and 0.55 (left) of their range, against 0.24 and 0.31 for the finger root joints; commanded and measured values are equal, so the policy commands this shape, it is not blocked into it. The shape is already there before the lift (right 0.68 at epochs 2,500-3,000).
- In a replay of an earlier policy of the same task (256 episodes), whenever the four fingers held the object, the middle finger link touched it in 0.00 of steps and the fingertip pad in 0.90-1.00; even steps whose joint pattern looked rolled were carried by the pad (0.90). The middle links stood 3.3 cm (median) off the wall.
- The environment now reports each digit's proximal link as well (the segment next to the palm): its position, its contact force with the object and its distance, direction and normal to the object's surface. Nothing in the reward has used those links, the middle link or the proximal link so far.
- The left arm lifts faster than the right: object speed while lifted 0.19 vs 0.11 m/s, arm joint speed 99th percentile while lifted 1.63 vs 0.92 rad/s. Link contact force 4.2 N (right 2.7 N), squeeze cost -0.25 per step (right -0.16).

How the current reward scores a digit: a digit counts as touching when any of its three sensed links touches, so a fingertip pad alone gives full digit contact; wrap depth counts touching links and is full at two, which the pad and the distal link reach together; the wall-following curl factor sits inside the envelope quality with a floor of 0.5.

## Feedback for improvement
Change only what Hypothesis A says. Keep every other term, including the two changes of the previous round (the pre-hold palm-seat bonus and the shove relief), which made the left arm learn the whole task. The reward must stay identical for the mirrored arm (no digit or side is named).

## Hypothesis A
- A digit must count as holding the object by its finger segments, not by its fingertip: wherever the reward counts touching digits or their links (the grasp ladder, the hold gate, the envelope quality, the wrap depth), a digit whose proximal, middle or distal link touches counts fully, while a digit touching only with its fingertip pad counts as a small fraction of that. Use the proximal link fields that the environment now provides.
- Because the middle and proximal links never touch today (they stand about 3 cm off the wall), give those segments a smooth pull towards the object's surface for each digit that already touches it, based on their surface distances, so that there is a gradient from the fingertip grasp the policy knows towards a grasp carried by the finger segments. This pull must be smaller than the lift and hold income so that it never becomes a reason to keep the object on the table.
