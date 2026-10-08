## I can see from the robot that
Round 10 is the first round of the left/right parallel plan: the same byte-identical reward, both arms from random
weights, left (`cp_l_m10`) on vision-3090 and right (`cp_r_m10`) on the local 5090, PPO-LSTM with 4,096 environments.
It ran 11.7 hours: 5,005 epochs on the left (8.6 s/epoch) and 8,270 on the right (5.1 s/epoch). No Traceback, no PhysX
error, placement rejection 0.42 with no failures on either side. The round was stopped on the stall criterion, not on a
clock: the wrench ADR never left level 0 because its gate never opened.

**Both arms learned to reach and to grasp, and nothing after that.** Palm-to-surface distance fell to 0.045 m (left)
and 0.037 m (right), inside the 0.06 m gate. Grasp episodes reached 0.83 left and 0.98 right and then sat flat for
4,000 and 7,000 epochs respectively. Lift is 0.008 left and 0.000 right; the airborne fraction is 0.001 and 0.000.
The object never leaves the table, so goal distance stays at its start value (0.25 m), success is 0, the tolerance
never moved from 0.1125, and the goal-gated wrench never fired: `wrench_near_goal_frac` and `wrench_goal_dwell_ep`
are exactly 0 on both sides and no ADR level was ever logged.

**The cause is income available with the object still on the table.** On the left the hold term pays 0.54-0.69 per
step throughout. The goal sits about 0.25 m above the object's start, and the hold term's coarse ramp is shaped on
goal_dist alone, so a cup resting on the table inside a closed hand already collects most of what the policy can
reliably earn. Lift pays at most 1.00 and only if the policy risks moving. The result is the parking equilibrium seen
on 09.21 in the IKER track: grasp, then stand still. The stall penalty (1.0, scaled by episode progress) was not
enough to break it.

**The two arms converged on different grasps from the identical reward.** Left: thumb 0.49, index 0.38, middle 0.30,
ring 0.39, pinky 0.59, palm 0.38, 2.2-2.6 links in contact - a whole-hand hold. Right: thumb 0.74, index 0.69, ring
0.76, but middle 0.011, pinky 0.002 and palm 0.001 - a three-digit pinch with no palm. Both reach the same grasp
episode score, so the reward cannot tell them apart; the palm term (0.15 of the 1.20 grasp budget) is too small to
make the palm worth acquiring. This is a reward-shape asymmetry, not a mirror defect: the mirror contract tests pass,
the right arm's earlier index-contact deficit resolved by itself (0.05 to 0.69), and left/right start poses are
verified mirrors.

**Cup tilt stayed small on both sides** (3.4 degrees left, 0.6 right), and the drop penalty is negligible, consistent
with an object that is gripped but never moved.

**One logging caveat.** `contact/*_at_success` reads 0.95-1.00 on both runs although no success ever occurred. These
event-EMA channels are not trustworthy until the first real success; do not read them as evidence.

**The recordings confirm the reading: both arms grasp and then stand still.** Left (snapshot epoch 5000) reaches the
cup by about 2.7 s, closes the whole hand around it with the palm against the side, and holds that pose without moving
the cup from 4 s to the end of the episode. Right (snapshot epoch 7200) reaches the cup by about 4 s and closes a
narrower grip on it, again without moving it. In neither recording does the cup leave the table or tip over.
Videos: `our_source/fj_t2r_videos/cp_l_m10_0924_1128.mp4` and `cp_r_m10_0924_1127.mp4`.

**Where on the object the hand grips, measured.** The graspable band of the standard cup spans z 0.232-0.332 in
env-local coordinates (table top 0.205, object origin 0.282) and its rim is at about 0.383. The lowest finger link
averages 0.339 on the left and 0.327 on the right, i.e. the hand hovers at the very top of the band, just under the
rim, and closes there. The palm never seats: palm-to-cup gap 0.070-0.078 m, palm contact 0.38 left and 0.001 right,
with only 2.1-2.5 of five digits and 2.2-2.6 of fifteen links in contact. The user's reading of the recordings is the
same on both arms: the hand cannot get below the rim, so index and thumb close on air beside it.

**This is not a workspace limit.** The 0.27 m palm-box floor printed at boot belongs to the palm-target action mode;
this task commands seven arm joint increments clamped only by joint limits (arm_target_box_rad 0.0), and the lowest
finger link already reaches 0.21 m (table level) in the worst environments. The hand can descend; the reward does not
ask it to. Proximity is measured to the NEAREST surface point, and from a hand coming in above the cup the nearest
point is the rim edge, whose normal points partly upward - so 'bring the palm close and face the surface' is satisfied
at the rim, and nothing pays for sliding down the side wall to where a real envelope grasp lives.

## Feedback for improvement
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
