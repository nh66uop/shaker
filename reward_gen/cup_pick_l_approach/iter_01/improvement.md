This round produced the stage-1 policy the two-stage split needs, and the reward that produced it should be kept as the
approach stage of the library.

Keep:
- No closing, lifting or goal term at all. The envelope latch and the lift fraction stayed at exactly 0.000, which is what
  gives the two stages a boundary.
- The contact penalty and the cup-disturbance penalty. Cup contact 0.000 and cup tilt 0.03 degrees, against a seed policy
  that knocked the cup over.
- A hold term that does not decay with episode progress. It is what turned "pass through the pose" into "stop there".

Two things to improve if this stage is revisited:

**Hold the pose for the whole episode, not most of it.** The hold term was still rising at the hour cap (0.40 -> 0.565),
so a part of every episode is still spent before or outside the pose. Paying more for the fraction of the episode spent
in the pose, and for arriving earlier, would finish this. The arrival itself is quick, so the head-room is in the tail.

**Tighten the two positioning conditions.** The palm-plane gap and the along-the-fingers offset pass 0.78 of steps, while
orientation, hand pose and no-contact pass essentially always. Those two are what place the cup between the thumb and the
fingers, and they are what the next stage inherits, so shaping them harder (a narrower target band around the values the
env checks) would hand the grasping stage a better starting distribution.

One caution. Success, tolerance and the goal are not part of this stage and stayed at zero the whole round; they must not
be added here to make the numbers move.
