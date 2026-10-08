What worked and must be kept exactly as it is: holding the open hand until the latch (the strong open-posture term shaped
on the worst joint, plus the direct pre-latch flexion penalty), gating the closing terms on the continuous geometric
readiness instead of the unobservable latch flag, and the restraint penalties on contact force, cup shoving and arm
speed. Together they gave an approach latch of 0.96, a no-touch condition of 0.96, cup tilt under 0.6 degrees, arm speed
under 13 rad/s and no runaway terminations, while the hand still learned to flex only after the latch. None of that
should be weakened to buy contact.

The feedback for improvement is that the hand now stops one gap short of the cup and nothing pays for crossing it.

1. The proximity part of the per-digit score peaks where the finger is near the surface, and the policy has settled
   exactly there. Closing terms saturated (0.13-0.14 each) while contact stayed at 0.037 digits and 0.011 N, and hand
   flexion actually fell back from 0.164 to 0.121 after epoch 1200, so at that pose flexing further is a loss. Credit
   for being near the surface has to be capped well below what a single real contact pays, and the gap between "near"
   and "touching" has to be a large step up rather than a small one, so that the last centimetre is where most of the
   money is.

2. The gating is circular and settles in a low-value equilibrium. Palm credit is multiplied by digit engagement, digit
   credit needs the digits near the surface, and the hand sits at a distance where neither pays much: palm-distance
   credit reached only 0.061 of its 0.60 maximum and the palm-to-cup gap grew from 0.038 m to 0.052 m. Bringing the palm
   in to the cup should pay on the palm-to-cup distance alone once the approach has latched, with no dependence on the
   digits, so there is always a gradient that moves the whole hand onto the cup.

3. The thumb is folded back against the palm in the recording, yet the thumb-opposition term pays 0.27 of its 0.50
   maximum. It is being satisfied by geometry that does not correspond to an opposed thumb. Opposition should require
   the thumb to be on the far side of the cup from the fingers AND close to the surface, so that a thumb tucked into the
   palm earns nothing.

4. Flexion alone is being rewarded to a level the task does not need. The term for leaving the open posture reached 0.144
   and then stopped mattering, because further flexion moves the fingertips away from the cup rather than around it. That
   term should fade out once the digits are near the cup, handing over to terms that depend on the fingertip-to-surface
   distance, so that late in the approach the only way to earn more is to actually reach the cup.

5. The index finger still never participates (0.001 over the whole round) and the middle finger barely does (0.003),
   while ring and pinky reach 0.013-0.016. The two-weakest-digit term was meant to fix this and did not, because it too
   is satisfied by proximity rather than contact. Once proximity is capped as described in point 1, that term should
   carry most of its value in the contact part.

6. Lifting, carrying and holding were never reached again and cannot be judged. Leave that part of the ladder as it is.
