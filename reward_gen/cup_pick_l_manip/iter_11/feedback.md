## I can see from the robot that
Round 11 was a regression and was stopped after 1.2 hours (left `cp_l_m11` 517 epochs on vision-3090, right
`cp_r_m11` 881 epochs on the local 5090, identical reward file, both from random weights). No Traceback, no PhysX
error, placement rejection 0.42 with no failures.

**Both arms walked away from the object instead of approaching it.** Palm-to-surface distance rose 0.221 to 0.371 m on
the left and sat at 0.357-0.480 m on the right, against 0.10 m by the same epoch in round 10. Contacts are
0.000-0.005 per digit, links in contact 0.003, palm contact 0.000, grasp episodes 0.001 and 0.000. Nothing downstream
happened: lift 0, airborne 0, goal distance unchanged at 0.249 m, tolerance 0.1125, and the wrench never fired.

**The cause is the side-wall weighting applied to the long-range reach term.** The factor (1 - |surface normal z|) was
multiplied into reaching as well as seating. From the start pose the nearest point of an open cup is its rim, whose
normal is nearly vertical, so that factor is close to zero there and approaching the cup paid almost nothing. Meanwhile
the stall penalty is unconditional and grows with episode progress: total reward stayed negative all round (-0.33 to
-0.10 left, -0.34 to -0.36 right) and was dominated by that penalty (-0.28 to -0.32). With no income for approaching
and a standing cost for the object being on the table, retreating was the least bad option, and both arms found it.

**What this round does confirm:** the fault is in the reward, not in the mirror or the machines. Two different GPUs,
two arms and two start seeds produced the same collapse from the same file, and the change that caused it is the only
one that touched the approach term.

## Feedback for improvement
Round 11 changed four things at once and one of them broke the first stage. Keep three, undo one.

1. UNDO: the side-wall factor on the long-range approach. It must pay on plain palm-to-surface distance, or with a
   floor of about 0.3 on the factor, so that closing in on the object is always worth doing. The rim-versus-wall choice
   is not made at 0.3 m away.

2. KEEP but narrow: prefer the side wall only from close in. Within a few centimetres of the surface, weight seating,
   per-link nearness and contact by how horizontal the surface normal is. That is where round 10's rim grip was formed
   (lowest finger link 0.33-0.34 against a wall spanning 0.23-0.33), and where the preference belongs.

3. FIX: the stall penalty must not punish a policy that has not reached the object. Gate it on the hand being at the
   object (in contact, or within a few centimetres) and keep it below the reach income it competes with. In this round
   it was the largest term in the whole reward, for policies that never touched anything.

4. KEEP unchanged: the second half (transport, goal, hold, success) gated on how far the object has risen, so nothing
   pays while it sits on the table; lift raised to 2.5; per-digit scoring by number of links in contact; the doubled
   palm share with the top ladder rungs requiring the palm; first-success-only bonus.

5. Method note for the next rounds: changing the approach term and the stall term together, in the same round that
   also re-scaled lift and the goal gate, is why this took a full round to attribute. The parallel left/right runs did
   their job - the identical collapse on two machines ruled out mirror and hardware in one hour instead of a day.
