What is working and must be preserved: the approach latch and the pre-latch discipline. Approach 0.95, no-touch 0.97,
cup tilt 2.6 degrees, cup left where it spawned. Paying the pre-grasp geometry after the latch as well as before it, and
making the latch bonus larger than the terms it switches off, removed the incentive to sit unlatched. Keep all of that.

The feedback for improvement is that the flat open hand is currently the best-paying pose in the whole reward, so the
policy has correctly found it and will not leave it.

1. Proximity and wrap are collected by a straight finger lying alongside the cup. Both terms measure only where a link
   is, not whether the digit is flexed around the body, and an extended finger skimming the surface sits exactly where
   they pay most. Curling would move links off that surface band and lose money. Credit for being near or on the surface
   should require the digit itself to be closing around the cup, so that an extended hand resting against it earns
   nothing.

2. Nothing rewards the thumb opposing the fingers. In the video the thumb is folded back against the palm, yet the
   envelope the task asks for needs the thumb plus at least four digits on opposite sides of the body. The thumb being
   across from the fingers should be paid for in its own right, before any contact happens.

3. The contact terms are an unreachable ramp. Per-digit contact, weakest-digit contact and opposition were exactly zero
   for 1,530 epochs, so they carried no gradient at any point in the round - they cannot teach anything. There must be a
   single continuous path from "approaching" through "nearly touching" to "closed on the cup" that strictly increases at
   every step of closing, and the near part of that path must be clearly worth less than the touching part, so that
   stopping at "near" is a loss rather than a comfortable plateau.

4. Palm-only contact is currently the largest contact income and it rose to 0.67 while the fingers stayed at zero. Palm
   contact should not pay on its own; it should count only together with digits that are also closing, otherwise it
   funds exactly the flat-hand pose seen in the video.

5. The hand's closing command never moved (0.26 to 0.33 over the round), so the policy is not even exploring closure.
   Whatever replaces the current closing drive has to make each additional increment of finger flexion profitable from
   the pose the hand is already in, not only once contact has been made.

6. Lifting, carrying and holding were never reached and cannot be judged from this round. They should stay gated behind
   a real enclosing grasp rather than being loosened, since the failure is upstream of them.

7. The policy this round starts from was trained for 4,280 epochs by a stage-1 reward whose largest standing terms were
   "hold the pre-grasp pose" and "keep the hand in its default open posture", with any cup contact penalised. Holding an
   open hand beside the cup is therefore a deeply reinforced habit, not a random starting point, and the closing command
   did not move at all this round (0.26 to 0.33 over 1,530 epochs). Stage 2 needs a term that pays for leaving the
   default open posture as such - measured on the finger joints, not on contact - so that there is a gradient out of
   that habit from the very first step, before anything touches the cup. It should be smaller than the terms that pay
   for real contact and a real envelope, so closing onto nothing is never as good as closing onto the cup.
