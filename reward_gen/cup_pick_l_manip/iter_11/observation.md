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
