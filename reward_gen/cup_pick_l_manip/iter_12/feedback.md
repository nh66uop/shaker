## I can see from the robot that
Round 12 ran both arms in parallel from random weights with the identical file (left `cp_l_m12` 2,076 epochs on
vision-3090, right `cp_r_m12` 3,387 epochs on the local 5090, 4.7 hours). It fixed round 11's collapse and then hit a
new wall. No Traceback, no PhysX error, placement rejection 0.42 with no failures.

**The approach came back, exactly as intended.** Palm-to-surface distance fell 0.239 to 0.047 m on the left and 0.212
to 0.062 m on the right, against round 11 where both arms retreated to 0.36-0.37 m. The stall penalty, now gated on
actually holding the object, reads -0.001 instead of -0.30, and total reward turned positive (0.12 left, 0.09 right).
The hand also descended to the side wall: lowest finger link 0.299 left and 0.289 right, inside the 0.232-0.332 band,
against 0.33-0.34 (rim level) in round 10.

**But the grasp never formed.** Only the thumb touches (0.38 left, 0.19 right); index, middle, ring and pinky stay at
0.006 or below. Links in contact are 0.40 of fifteen. **Palm contact is exactly 0.000 on both arms for the whole
round**, the palm-to-cup gap stops at 0.047-0.054 m, and the grasp reward terms (count, per-digit, palm) pay 0.001.
Lift is 0.003, the airborne fraction 0, goal distance unchanged at 0.250 m, tolerance 0.1125, and the wrench never
fired. Both arms flattened out: the last three buckets are identical to within noise.

**The recordings show why.** The hand arrives beside the cup with the fingers already curled and holds that pose; the
thumb tip rests against the wall and the other digits close on air next to it. The user identified this from the video
before the metrics did: approaching with a closed hand makes the wrap impossible.

**The reward has no term that keeps the hand open while approaching.** Closing pays only as flexion x link nearness x
palm seat, so curling early earns nothing - but it costs nothing either, and the policy settles into it. The stage-1
reward that produced every working grasp in this track does have that term: it multiplies the approach income by an
open-pose factor measured on the movable joints, so the hand must stay open to be paid for arriving.

## Feedback for improvement
Round 12 proved two of the three fixes and exposed the missing one.

1. KEEP: the approach shaped on plain palm-to-surface distance, the side-wall preference faded in only within a few
   centimetres with floors, and the stall penalty gated on holding the object. Together these recovered the approach
   (0.24 to 0.047 m) and brought the hand down to the side wall.

2. ADD, and this is the round's finding: **the hand must be paid to stay open while it approaches.** Both arms arrived
   with the fingers curled and could not wrap; palm contact was 0.000 for 2,000-3,400 epochs on each machine. Multiply
   the approach income by an open-pose factor on the movable finger joints (distance of each joint from its default
   open pose, both the mean and the worst joint), the way the stage-1 reward does. Closing should only start paying
   once the links are close to the surface.

3. The palm is the gate to everything downstream and it never arrived. Give palm proximity a steep gradient over the
   0.08 to 0.02 m range - both arms stopped at 0.05 m - and let the lower rungs of the digit ladder pay without the
   palm, so a first partial wrap is worth forming; keep the palm requirement for the top rungs only.

4. Programme note (user decision 09.24): rounds 10-12 showed that a single continuous reward cannot build a wrap from
   random weights in this task. The wrap is now trained first by the verified stage-1 reward, per arm, and this
   continuous reward is applied on top of those weights, as one policy per arm. Stage 1 finished at envelope 0.55
   (left, e2450) and 0.82 (right, e1800); both are archived with their reward file under
   our_source/policy_zoo/cup_pick/<side>/approach. So the next round of this reward starts from a hand that already
   grasps - but items 2 and 3 still apply, because the reward must not undo that grasp.
