What worked and must be kept: the first half of the chain now trains from random weights on both arms. Reaching and
grasping were learned in under 1,000 epochs and held for thousands more (palm-to-surface 0.037-0.045 m, grasp episodes
0.83-0.98, four to five digits in contact on the left). Keep the reach and grasp terms as they are, keep uprightness as
a multiplier, keep the object-agnostic and arm-agnostic rule, and keep the first-success-only bonus.

The feedback for improvement is a single structural fault, and one weight.

1. The second half of the task must pay nothing while the object is on the table. Transport, being near the goal, the
   hold term and the success bonus were all shaped on goal_dist alone. Because the goal sits about 0.25 m above the
   object's start, a cup gripped on the table already collected 0.54-0.69 per step of hold income - more than the
   entire lift term could offer (1.00), and without risk. Multiply every second-half term by a smooth function of how
   far the object has actually risen above ctx.cup_spawn_pos[:, 2], so that all of it is zero on the table. This is
   the same parking trap recorded on 09.21: a dense state-proportional term made "hold the pose and stay" optimal.

2. Lifting must be the largest single step up in income. Raise the lift ceiling from 1.00 to about 2.5, rising with
   height to the height the goal needs, and let it be the only door to the goal terms (point 1). Make a closed grasp
   parked on the table actively costly as the episode progresses.

3. Pay the palm properly. From the identical reward the left arm acquired a whole-hand grasp (palm 0.38, five digits)
   and the right arm a three-digit pinch with palm 0.001. The palm term is 0.15 of a 1.20 grasp budget, which is not
   enough to be worth acquiring, and a pinch is a worse starting point for lifting and for surviving the wrench. Give
   the palm a clearly larger share, and make the digit-count ladder require the palm for its top rungs.

4. Nothing is known yet about robustness under the wrench, because the gate (lifted AND near the goal) never opened.
   Once lifting works, the ADR schedule should be watched from the first epoch it fires: level, fire fraction, dwell,
   and whether per-digit contact at success holds up as the level rises.

5. Keep both arms on the identical file and keep running them in parallel; the divergence in grasp style is the most
   useful signal this round produced, and it only shows up when the same reward is trained twice.

6. Make the reward ask for the side wall, not the nearest point. Both arms grip at the rim because the nearest
   surface point from above is the rim edge. Two object-agnostic levers: (a) weight proximity and facing by how
   horizontal the surface normal at the contact point is - a side wall has a horizontal normal, a rim or a lid does
   not, and gravity is shared by both arms so this stays mirror-safe; (b) pay for the number of LINKS of each digit in
   contact rather than for a digit touching at all, since a fingertip on the rim gives one link while a wrap around
   the wall gives two or three. Round 10 sat at 2.2-2.6 links of fifteen.
