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
