Start from the previous code shown above, and change only what one hypothesis below says. Everything not named stays as it is, except that the term count must respect the output rule on the number of terms: merge or drop terms if needed, never add a new per-finger term.

## Hypothesis A
- The long-range reach: restore the plain form of lesson 1 (weight 0.35, 0.6 * far + 0.4 * near, times (0.45 + 0.55 * grip), fading once held) and remove the open-hand multiplier from it and from the palm term.
- Keep the fingers open on the way in with a separate ADDITIVE term: it pays for the open pose only while the palm is far from the surface (full at about 8 cm, zero by 2 cm and zero once held), and it must be smaller than the first digit-contact increment and than the 0.195 per step of lesson 4 when the hand stands still at the wall.
- Keep the wall-following curl (lesson 13) as the curl factor inside the envelope quality.

## Hypothesis B
- Everything in Hypothesis A.
- In addition, do not multiply the lift and rise incomes by the envelope quality: lifting must pay with any real hold (the `held` gate), so that the first lift is not throttled by an envelope the policy has not formed yet. Transport, hold-at-goal and success keep the envelope-quality multiplier.
