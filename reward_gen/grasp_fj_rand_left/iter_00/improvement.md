Two things went right and must be kept.

- The approach works on the left arm exactly as on the right: the hand leaves the raised start, finds the cup wherever it
  is placed, and arrives beside it in the default open pose with the start orientation, completing the approach in
  0.86-0.94 of far-start episodes and holding a palm-to-cup gap of about 0.037 m.
- The hand keeps its pre-grasp shape until it is beside the cup, and it never presses into the table.

The rest is what has to change.

**Closing must pay per digit, so that a three-finger grip is worth clearly less than a five-finger one.** This is the
whole failure of the round. The thumb, the middle finger and the pinky reach the cup (0.59-0.70 of steps) while the index
and the ring finger essentially never touch it (0.0004, 0.0003). The envelope needs four digits and the palm, so it never
completes and nothing after it is ever reached. A probe with the policy removed shows both hands close all five fingers on
the cup with the same forces, so this is a reward shape the policy settled into, not a limit of the hand. The terms that
pay for closing should count each digit separately and should keep rising as the fourth and fifth digit arrive, rather
than saturating once a few links are in contact; contact on the index and the ring finger has to be worth as much as
contact on the middle finger and the pinky. Income that a three-digit grip can already collect should be small next to
what completing the envelope pays.

**The hand should aim the fingers around the cup, not only at its near side.** The fingers that miss are the ones that
have to wrap to the far side of the cup body. Reward the finger links for closing around the cup along its surface (the
tangential direction), not for pressing towards its axis, and make that term require the palm to be against the cup first,
so it cannot be earned by brushing the cup from outside.

**Approaching must stop paying once the hand is already beside the cup.** The approach terms still earn every step after
the approach latch has fired, and the policy is content to hold the arrived pose: the grasp-pose and palm-reach terms
together pay about 0.27 per step while the whole closing side of the reward pays less than 0.2, and lifting pays nothing
because it is never reached. After approach_done, the approach income should flatten and the value should come from
closing and lifting.

One caution. Success and the tolerance never moved in this round, so nothing can be concluded from them, and the success
condition must not be loosened to make the numbers move. The blockage is the envelope, and it is one specific thing: two
fingers that never reach the cup.
