What worked and must be kept: making every position term worth little until the fingers are in contact. Round 6 is
the first round in which contact appeared and stayed (links in contact 0.91), so the rule "no term may sit near its
maximum while contact is zero" is right and should stay. The restraint penalties also kept everything safe (no table
contact, cup tilt under 3.3 degrees).

The feedback for improvement is structural and comes from the user, not from tuning: **do not key the reward on the
environment's approach and envelope latches at all.** They are hand-written templates of one grasp on one cylinder
(palm plane -1 to +2 cm from the cup side, cup axis in a 25 mm window, open hand, no touch, start orientation), they are
not in the observation, and once set they stay set. Six rounds show the consequence: when the policy meets the approach
template perfectly (rounds 2-5, approach 0.91-0.96) it has learned to stand still with an open hand, and when the
reward pushes contact (round 6) the contact happens outside the template and the approach latch collapses to 0.16-0.35.
Either way the envelope latch stays at 0. A grasp reward built on these templates also cannot move to other objects.
The latches remain useful as logged metrics only.

1. Replace the stage switch with continuous, object-agnostic quantities computed from the current state. The
   environment now provides (manip_stages `ObjectRewardContext`) the distance from the palm and from every finger link
   to the nearest point of the object's real surface, the direction to that point and the surface normal there. Being
   close is a smooth function of those distances, and closing the fingers is worth more the closer the hand is; there is
   no threshold at which the reward changes regime.

2. Price the grasp by what the hand does to the object, not by a pose: the number of digits in contact, palm contact,
   and opposition, i.e. the thumb's contact pushing against the fingers' contacts from the other side of the object
   (from the surface normals at the contact points). It must rise steeply with the third, fourth and fifth digit, so
   that the thumb-and-pinky touch this round settled on (per-digit mean 0.52 with two digits) is worth little. Index,
   middle and ring stayed at 0.001-0.003 for 2,000 epochs; they need to pay at least as much as the others.

3. Contact before the hand is close and seated must not be profitable. This round the pre-latch shove penalty (-0.39)
   was smaller than the contact reward (+0.52). With continuous proximity this becomes: contact counts in proportion to
   how well the palm faces and approaches the surface, so a fingertip poke from beside the object earns little.

4. Lift and hold stay as outcomes of the object, which are object-agnostic already: rise of the object above its spawn
   height while held, low object speed, upright, then the goal. Holding the closed grasp on the table must not be worth
   staying for.

5. Start from `cup_pick_l_approach_e3400` (user decision): it already closes a four-digit envelope with palm contact
   (0.81) in 87 % of episodes and lifts in 19 %, so the new reward first has to keep that grasp and then teach the lift,
   instead of undoing a no-contact standing habit as in rounds 2-6.
