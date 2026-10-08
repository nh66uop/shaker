Round 9 of the left-arm stage-2 (manip) track (`cp_l_m09`) ran 1,712 epochs in 4.0 hours, PPO-LSTM with 4,096
environments, this time on the vision-3090 machine (RTX 3090, about 8.0 s per epoch against 4.6 s on the local 5090),
from round 8's epoch-3200 snapshot. Two things changed: the reward budget was moved to the end of the task (reach and
close 0.30 and fading once held, keeping the grasp 0.80, lift 1.00 capped at goal height and multiplied by horizontal
closeness to the goal, transport 2.50, at rest at the goal 10.0, success 100), and the environment now applies an
external wrench to the object once it is lifted and inside a fixed radius of the goal, with an ADR level tied to how
well the policy does there. Placement rejection behaved as designed: 0.42 of resets resampled and none failed.

**The tolerance curriculum went further than in any previous round.** Starting from 0.1125 again (a new run restarts
it), the goal tolerance fell to 0.047 m by epoch 1700, against 0.066 m at the end of round 8 and 0.015 m as the task's
floor. Far-start success held at 0.85-0.87 through the first three tightenings and eased to 0.77 at the tightest
setting; mean successes per episode were 0.39-0.53. In round 8 each tightening cut success roughly in half.

**Time spent at the goal grew steadily, which is what the round was for.** The hold-at-the-goal term rose 0.054 to
0.215, the transport term 0.078 to 0.207, and the time held inside the disturbance radius rose 24 to 41 steps with the
fraction of steps in that radius rising 0.24 to 0.43. The goal keypoint distance fell 0.204 to 0.167 m - the first
sustained fall in this track - but it is still 3.5 times the current tolerance, so the hand still does not come to rest
on the goal.

**The grasp first loosened, then recovered.** Cutting the grasp income from 3.86 to 0.80 per step cost contact
immediately: digits fell 0.75 (round 8) to 0.43-0.47 by epoch 800, palm 0.64 to 0.34, the palm gap opened 0.039 to
0.078 m and the airborne fraction fell 0.67 to 0.32. From epoch 800 on it recovered to digits 0.53-0.56, palm 0.43, gap
0.057 m and airborne 0.46, with the drop penalty easing -0.49 to -0.30. So the reduced grasp income is survivable, but
the grasp is measurably looser than round 8 and the drop penalty is the largest negative term of the round.

**The ring finger lost the contact it had gained.** It was 0.40 at the end of round 8, and here went 0.28 to 0.05 at
epoch 1200 before partly recovering to 0.13. The other four digits strengthened over the same interval, so the hand
re-shaped around a four-digit hold when the wrench started arriving.

**The cup stays upright** at 7.3-8.4 degrees throughout, and the hand never approaches the table (lowest point 0.40 m).

**The recording (snapshot epoch 800, tolerance 0.08, 12 far-start environments) shows repeated successful episodes, not a pick-release loop.** The hand reaches the cup at about 2.7 s, grasps it and lifts it upright by 4.1 s, and at about 4.7 s the scene resets: the arm is back at its start pose and the cup stands at a new place on the table. The same sequence then repeats twice more inside the 15 s recording. Frames at 0.5 s spacing across the transition confirm a reset rather than the hand putting the cup down. The metrics agree: the set-down penalty averages -0.001 (of a 3.0 weight), so deliberate set-downs essentially never happen, and `ctrl/prev_ep_successes_mean` is 3.8-4.0, i.e. episodes end after the policy has collected close to the environment's limit of 5 successes. What the recording shows is the policy finishing the task repeatedly, with the cup upright each time.

**Caveat on the snapshot.** rl_games only writes a snapshot when the mean reward improves, and with the income
concentrated in the success bonus nothing was written after epoch 800. The recording therefore shows an early policy
(tolerance 0.089 era), not the epoch-1712 state with tolerance 0.047. The final weights of this round do not exist.
