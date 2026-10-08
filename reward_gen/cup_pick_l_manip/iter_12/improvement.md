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
