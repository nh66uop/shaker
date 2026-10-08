Keep the round-18 code whole: no reward change this round. Its two changes did what they were meant to. The index
curls again (0.31-0.34 of its force on the inner links at success, round 17 0.01), the ring and little finger started to
use their inner links, success reached 0.842 at epoch 800 with three-goal episodes up by a half, and the terminations
before a grasp are back to the start's level. Every structural rule stays (second half gated on the object's rise,
envelope quality as ONE multiplier on the downstream income with curl as a factor inside it, no per-finger term,
falling-only drop and set-down costs, open-pose factor only on the approach, palm seating tied to engagement,
uprightness as a multiplier, first-success-only bonus, goal-gated wrench, the table band). Every budget stays,
CURL_FLOOR stays, and no speed or control charge is relaxed.

1. The extra early terminations are not a reward matter. They are the object leaving the working area in y after the
   lift (0.05 -> 0.11 -> 0.16 of episodes); every other cause after the lift is flat or lower, and per carried goal the
   risk of ending the episode did not change (0.054 -> 0.061 -> 0.063 per leg). The left arm's bound is the right
   arm's (object y <= 0.25 m) while the goal box reaches 0.35 m and each new goal is a random step from the last one,
   so a policy that chains more goals meets more goals on or beyond the line, and at the evaluation tolerance the
   success region of any goal above 0.16 m crosses it. That bound is an environment setting and is left to the
   operator. The reward must not name a world axis or a bound, so there is nothing in it to change for this.

2. Do not answer the lighter grip. Finger force on the held object fell 71 -> 59 N: the ring finger moved force off its
   tip, which the curl factor pays for, and the thumb pressed less against the lighter fingers under the unchanged
   squeeze charge. Nothing followed from it: no drops, fewer tilts, the episodes that ended after the lift held as
   firmly as the ones that did not, and the hold income rose. A force floor would be an absolute force target; keep the
   curl factor, its trust on the inner-link force and the squeeze charge as they are.

3. Do not slow the carry to keep the object away from the line. The exits with the goal inside the line are fast
   sideways moves of a firmly held object (0.6-0.8 m/s, most of them away from the goal); such moves happen in every
   direction and only one side has a bound within reach. Moving away from the goal is already charged by the signed
   progress term and the drift charge. Raising the speed charges would tax every transport for an exit that exists only
   because of where the bound sits, and could make standing still competitive again.
