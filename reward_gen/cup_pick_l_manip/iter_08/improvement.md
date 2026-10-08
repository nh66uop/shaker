What worked and must be kept: making uprightness a multiplier on the lift, rise, goal and hold income rather than a
separate penalty. That single change took cup tilt from 27-31 degrees to 4-7 degrees, and with the upright pose the
grasp improved on its own: the ring finger touched for the first time in the whole track (0.000 to 0.40), links in
contact rose to 4.36, palm contact to 0.64, grasp episodes to 0.94 and lift to 0.89. The capped lift income and the
raised squeeze thresholds also behaved as intended (squeeze -0.45 in round 7, -0.15 here). Keep all of this, and keep
the object-agnostic structure: no latches, no cylinder template.

The feedback for improvement is one stage: coming to rest at the goal. It is the only part of the task that has never
worked, in either round of the new reward.

1. The hold term pays 0.53-0.63 out of a maximum of 9.0, the near-goal step fraction is 0.019-0.050, and the goal
   keypoint distance has been flat at 0.163-0.174 m for 1,500 epochs while the tolerance has shrunk to 0.066 m. The
   shape of the term is the problem, not its weight: it was already raised 5.0 to 9.0 this round and the distance did
   not move. Two things to change together. First, make the approach to the goal pay on a scale matched to where the
   policy actually is: the fine ramp is 0.05 m wide, so at 0.17 m it contributes almost nothing and the policy sees a
   flat landscape between 0.17 m and the tolerance. Add a coarse term that keeps rising over the whole 0.0-0.30 m
   range, so there is a gradient to follow from where it sits now. Second, pay for reducing the distance (the change
   in goal_dist between steps) as well as for the distance itself, so moving towards the goal is rewarded even before
   arriving.

2. Carrying the cup around with no destination must be worth less than arriving. The lift income is 2.2-2.6 per step
   while the goal and hold terms together pay about 1.0, so hovering with the cup upright remains the best steady
   state. Now that the cup is reliably upright and held, reduce the standing lift income once the object is above the
   height the goal needs (it is already capped; make it decay with time held, or scale it by proximity to the goal),
   so that the only way to keep earning is to close on the goal.

3. The success bonus is tiny in practice (0.015-0.036 per step averaged) while each success also restarts the episode
   clock. If the intent is that successes are the point, the bonus should be large enough to be visible next to the
   2.5 per step the policy earns for simply holding the cup up.

4. Keep the tolerance curriculum as it is. It has moved 0.1125 to 0.066 on its own and each tightening was absorbed
   within a few hundred epochs. Do not compensate for it in the reward.

5. Start the next round from this round's epoch-3200 snapshot (`cp_l_m08/nn/last_..._ep_3200_rew_4891.019.pth`), which
   holds the upright five-digit grasp at tolerance 0.066.

6. Worth recording for the next reader: two flat curves in this round were read as hard limits and were not. Tilt sat
   at 24 degrees for 1,200 epochs before falling to 5, and the ring finger sat at 0.000 for 2,500 epochs before rising
   to 0.40. Do not conclude that a plateau is structural in this task without a geometric check.
