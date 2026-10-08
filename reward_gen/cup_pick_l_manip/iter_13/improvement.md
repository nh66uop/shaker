What worked and must be kept: applying this reward on top of a stage-1 grasp is the right structure. On the left arm it
lifted the object in 46 % of episodes, up from 13 %, while keeping the wrap it inherited (grasp 0.75, links 2.86, ring
contact 0.52). The gating of the whole second half on how far the object has risen, the lift ceiling of 2.5 and the
palm-free lower rungs all behaved.

The fault is one term, and it is fragile enough to decide the outcome from the initial pose.

1. The open-pose factor must switch off once the hand is at the object. It currently multiplies the reach income at all
   times, so keeping the fingers straight pays even while the hand is touching the cup. The right arm found that
   corner: fully open hand (`hand_curl` 0.000), palm flat on the cup, +0.195 per step, no digit contact, for 1,400
   epochs. Gate it on distance - full weight while the palm is more than about 8 cm from the surface, fading to zero by
   the time contact starts - or multiply it by (1 - held). Being open must be worth something only on the way in.

2. The palm-proximity term (0.24, steep from 0.08 to 0.02 m) is what makes that corner profitable, because it pays on
   proximity alone. Require contact, or at least pair it with a digit-contact factor, so that a flat hand pressed
   against the object without fingers earns much less than a wrap.

3. Both arms must be restarted from their stage-1 weights with the corrected reward (user decision, option 1), so the
   left and right keep using the byte-identical file and the mirror comparison stays clean. The left arm's progress in
   this round (lift 0.46) is discarded deliberately; it is worth one round to keep the pair symmetric.

4. Watch in the next round: `hand_curl` must stay near its stage-1 value (0.2-0.3) rather than collapsing to 0, and
   grasp episodes must not dip below about 0.4 in the first 300 epochs. If either arm shows the flat-hand corner again,
   the open-pose term is still reaching too far.
