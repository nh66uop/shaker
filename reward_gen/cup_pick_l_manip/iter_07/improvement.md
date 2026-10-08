**User instruction (2026-09-23): the cup must be carried and lifted upright.** This is the first priority of the next
round; item 2 below is not optional and the reward must make a tilted lift worth clearly less than an upright one.

What worked and must be kept: the whole object-agnostic structure. Dropping the latches and the cylinder template, and
paying continuously for surface proximity, per-digit contact with a steep count ladder, opposition, palm contact and
object height, took this track from six rounds of zero envelope and zero lift to lift 0.87 and far-start success 0.43
in one round, starting from a policy that lifted in 0.19 of episodes. None of that should be traded away, and the next
round should keep the same term structure and the same weights for reach, close, count, digit, links, palm, oppose and
lift.

The feedback for improvement is about the last third of the task, holding the object at the goal, and about how the
object is carried.

1. Holding at the goal never started: that term paid exactly 0.000 for 2,819 epochs, the near-goal step fraction stayed
   under 0.045 and the goal keypoint distance flattened at 0.216-0.230 m while successes were still being counted. The
   policy passes the goal and keeps carrying. Make coming to rest at the goal the dominant income: a continuous,
   distance-shaped term that grows as the object slows near the goal (so it is reachable before the success condition
   is met), worth clearly more per step than holding the object anywhere else, and make the lift income stop growing
   once the object is above the height the goal needs, so there is nothing to gain from lifting higher instead of
   moving to the goal.

2. The cup is carried tilted 27-31 degrees, visible throughout the recording, because the tilt penalty (-0.36 per step)
   is a seventh of the lift income (+2.6). Multiply the lift and goal income by an uprightness factor instead of adding
   a separate penalty, so a tilted carry cannot pay more than an upright one. The environment also requires the object
   upright for a success, so this is on the critical path to the tolerance shrinking further.

3. Squeezing is the largest single penalty of the round, -0.43 to -0.47 per step, larger than the tilt penalty. Either
   the threshold is too low for the force a real hold needs, or the policy is genuinely crushing the cup; the same
   contact set (four digits, palm 0.54) holds fine in e3400 without that cost. Check the threshold against the measured
   holding force and raise it, or soften the penalty, so that holding securely is not taxed as hard as over-gripping.

4. The ring finger has never touched the cup in any round of this track, including in the starting policy, and its
   closing command sits at 0.203 and does not move. The count ladder already pays for a fifth digit, so the reward is
   not the obstacle; before the next round, check whether the ring finger can reach the cup at all in this hand pose
   (joint limits, link geometry, whether its joint is among the 13 movable ones). If it cannot, stop treating five
   digits as the target and make four the full grasp; if it can, add a term that pays only for the ring finger.

5. Keep starting from the best weights of this round rather than from e3400: this round ended with grasp 0.73 and lift
   0.70 right after the tolerance tightened to 0.101, and those numbers were 0.90 and 0.86 just before it. The next
   round should continue from the epoch-1800 snapshot (or the best one) so the tighter tolerance is learned on top of a
   working pick-and-carry.
