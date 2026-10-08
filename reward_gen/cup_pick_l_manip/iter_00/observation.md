Round 1 of the left-arm track (`fj_randL_i00`, resumed as `fj_randL_i00b`) trained the mirrored right-arm reward from
scratch on the left arm. The environment is the exact mirror of the right one: the cup appears uniformly in x 0.10-0.40 m,
y 0.00-0.30 m every episode and every episode starts far from the cup. The run reached epoch 3000 before the physics
engine failed (an unrelated process took the GPU memory), and was resumed from the best pre-failure weights for another
1533 epochs, 2.1 hours. The two segments behave the same, so they are read as one round.

**The approach was learned quickly and holds.** Far-start episodes complete the approach 0.86-0.94 of the time from
epoch 1000 onward, and the hand keeps its start orientation almost perfectly (the orientation condition passes 0.998 of
steps). The palm-to-cup gap settles at 0.035-0.039 m. On the right arm the same reward, started from a policy that had
already learned to approach, held the approach at about the same level, so the mirrored approach behaves as intended.

**The grasp stops at three digits and the envelope never closes.** The envelope latch is 0.000 for the whole round. The
reason is visible in the per-finger contacts over the last 200 epochs: thumb 0.67, middle 0.70, pinky 0.59, but
**index 0.0004 and ring 0.0003**. The latch needs the palm, the thumb and at least four digits in contact for five
consecutive steps; with the index and the ring never touching, three digits is the ceiling. Mean digits in contact rose
1.45 to 2.00 across the round and then flattened, and palm contact rose 0.14 to 0.48.

**The hand itself can use all five fingers.** With the policy taken out of the loop (cup teleported to the grasp pose in
front of the palm, all finger joints commanded closed), the left hand and the right hand behave the same: all five
fingers touch in every environment, with peak forces thumb 61.7 / index 22.5 / middle 26.4 / ring 23.7 / pinky 11.0 N on
the left against 57.6 / 22.8 / 28.5 / 25.6 / 15.6 N on the right, and the same joint closure fractions. So the missing
index and ring contacts are what the policy does, not a mirror error in the hand's signs, limits or sensors.

**Lifting barely starts.** The lift fraction peaked at 0.055 around epoch 800 and fell back to 0.030, the airborne
fraction is 0.009-0.018, and success stayed at 0.000, so the tolerance never moved from 0.1125. Cup tilt is 7 degrees
and the hand never rests on the table (lowest hand point 0.26 m).

**The recording (epoch 800 weights) shows the same thing.** The hand travels from the raised start to the cup, stops
beside it with the palm against the cup's side, and holds that pose with a partly closed hand for the rest of the
episode. The cup stays on the table and is never enclosed; the fingers that do close reach it from one side only.

**Placement rejection behaved as designed.** 0.42 of resets resampled a placement, none failed, and no cup spawned under
the hand or on a tabletop hole.

One note on the first segment. At epoch 3000 PhysX failed to allocate GPU memory because a video recording was started on
the same GPU; the simulation died and the following 6,700 epochs ran on frozen state (reward -0.012, episode length 337,
every metric constant). That segment is discarded; nothing in this observation comes from it.
