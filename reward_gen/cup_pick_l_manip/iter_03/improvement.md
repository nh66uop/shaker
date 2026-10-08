What worked and must be kept: the flexion-gated proximity and wrap credit, the single continuous per-digit ladder from
"curled and near" to "touching", the term that pays for leaving the default open posture, and the thumb-opposition term.
Together they did what two previous rounds could not - the hand closes, the middle and ring fingers engage, and the
envelope latched for the first time. Do not go back to contact-only closing terms.

The feedback for improvement is that closing now destroys the gate that pays for closing.

1. Closing before the latch must cost more than it earns. Every closing term is gated on the approach latch, but there
   is nothing that discourages flexing the hand before that latch, and the only counterweight - holding the default open
   posture before the latch - is worth about one twenty-fifth of what closing pays afterwards. The optimal policy under
   that arrangement is to close as early as possible, which is exactly what happened. Holding the open posture until the
   latch has to be worth roughly as much as closing is afterwards, and flexing the fingers before the latch should be
   penalised directly on the joint angles, not only when it produces contact.

2. The gate's pose condition is a maximum, not an average. It requires every movable finger joint to stay within 0.3 of
   its default angle, so a single finger that flexes past that threshold breaks the whole approach gate. Measured pass
   rate fell from 0.997 to 0.213 at epoch 400 and never recovered. The pre-latch posture term should therefore be shaped
   on the worst joint, matching what the gate actually tests, instead of on an average over the hand.

3. The reward regime switches on a state the policy cannot observe. The observation vector contains joint angles and
   velocities, palm pose, fingertip positions relative to the palm, the commanded hand state, cup and goal keypoints and
   the previous action - it contains neither the approach latch, nor the envelope latch, nor any contact force. The
   policy has to infer from its own memory that its reward regime has changed. Terms that turn on and off hard at the
   latch are therefore weak signals; prefer terms whose value changes smoothly with the same geometric quantities the
   gate measures, so that the gradient points towards the gate rather than depending on knowing which side of it the
   policy is on.

4. The grasp is being made by pushing the cup. Cup tilt reached 12 degrees and the largest arm joint velocity per
   episode reached 11,591 rad/s, with runaway terminations appearing, exactly in the interval where the envelope first
   latched. Closing has to be restrained: excessive contact force, cup displacement and cup rotation while closing
   should cost clearly more than the extra contact they buy, so that the fingers settle onto the cup rather than drive
   through it.

5. The palm must stay on the cup while the fingers close. Palm contact never rose above 0.16 and the palm-to-cup gap
   grew from 0.031 m to 0.082 m as the fingers engaged, so the grasp converged to a fingertip pinch. The envelope needs
   the palm, the thumb and four digits at once. Keeping the palm against the cup should be paid for continuously
   throughout closing, using the distance itself rather than only the binary contact.

6. The index finger still never participates (0.002 to 0.011 over the whole round) while the middle and ring fingers now
   do. Whatever asymmetry keeps it out should be addressed by making the per-digit score of the two digits that arrive
   last worth clearly more than the ones that arrive first.
