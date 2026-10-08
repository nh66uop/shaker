What worked and must be kept: moving the income to the end of the task, and the goal-gated wrench. Together they gave
the first sustained fall in goal distance (0.204 to 0.167 m), tripled the hold term, raised time inside the goal radius
from 24 to 41 steps, and carried the tolerance from 0.1125 to 0.047 without the success collapse that every earlier
tightening caused. Keep the ceilings, keep uprightness as a multiplier, keep the wrench.

The feedback for improvement is that the task ends too easily: episodes finish after about four cheap successes at a
goal that never moves, so the policy is never asked to transport the object anywhere or to stay put.

1. Successes are cheap and end the episode early. Episodes finish with 3.8-4.0 successes out of the limit of 5, because the goal sits directly above the spawn point and is re-sampled in the same place, so hovering there collects the bonus again and again until the episode truncates. The reward should pay the large bonus for the first success and pay later ones through the hold term, and the environment's goal sequence should move the goal after a success (`goal_delta_distance` is 0.0 today) so that a second success means transporting somewhere new. Without that, the tolerance curriculum is being driven by a task that ends before it gets hard.

2. The hold term is still not reaching the policy: 0.215 out of 10.0, because at 0.167 m the object is 3.5 tolerances
   away and the fine part of the ramp is narrow. Widen the paying region so that being inside about two tolerances
   already pays a large fraction, and make the term grow as the object stays there (a dwell factor), so that standing
   still at the goal beats arriving repeatedly.

3. The grasp is looser than round 8 (digits 0.55 against 0.75, palm 0.43 against 0.64, gap 0.057 against 0.039 m) and
   the drop penalty is the round's largest negative term. The 0.80 grasp ceiling is close to the lower bound of what
   works; if anything it should go to about 1.2 now that the goal terms are larger, rather than lower.

4. The ring finger fell 0.40 to 0.13 when the wrench arrived. Check whether it can contribute at all under the
   disturbance, or whether the four-digit hold is simply the stable answer; if it is, stop paying a ring-specific term.

5. Next round should start from this round's epoch-800 snapshot, which is the only weight file that exists. Consider
   also saving on a schedule rather than on reward improvement for this reward shape, or the end-of-round weights will
   keep being lost.

6. Machine note: this round ran on vision-3090 at about 8.0 s per epoch, 1.7 times slower than the local 5090, so a
   four-hour round is roughly 1,700 epochs instead of 3,000. Judge plateaus against epochs, not wall clock.
