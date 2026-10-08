Round 17 ran left `cp_l_m17` from the round-16 epoch-1000 snapshot with the round-17 reward (wrap depth weighted by
each digit's inner-link force share) for 1250 epochs (cumulative 2250) on vision-3090. No Traceback, no PhysX error, no
NaN. It was stopped at 1250 as a round end because a fixed-tolerance evaluation showed a regression against the
snapshot it started from.

**Fixed-tolerance deterministic evaluation (tolerance 0.0911 m, wrench pinned at 20 N/kg, first episode of 256
environments, seeds 42 and 43: 512 episodes per checkpoint).** Epoch 800 against the epoch-1000 start: success 0.785 ->
0.703 (z -2.9). Goal chaining improved (three-goal episodes +0.09, successes per episode +0.31), but early terminations
rose 0.25 -> 0.41 (z 5.6) and lifted episodes fell 0.91 -> 0.79 (z -5.3). At epoch 450 success was still equal (0.760)
while terminations had already risen by 0.12.

**The fingers did not curl; the index moved its grip onto its fingertip.** Replay of both checkpoints (seed 42), every
held step (object lifted and at least 3 cm up):
- index: share of its cup force carried by the two inner links 0.19 -> 0.03 (z -5.4); fingertip-dominant on 0.73 ->
  0.98 of its pressing steps; inner-link contact on 0.26 -> 0.04 of held steps; fingertip force 9.8 -> 14.8 N, inner
  links 2.3 -> 0.4 N.
- middle, ring, pinky: 0.02-0.08 of their force on the inner links at BOTH checkpoints (fingertip-dominant 0.92-0.99).
  The four fingers together: 0.10 -> 0.05.
- thumb: 0.69 -> 0.74 on its inner links, 27 -> 35 N there. Total cup force while held 92 -> 111 N (z 2.8), squeeze
  charge 0.29 -> 0.34 per held step.
- Around the first success of an episode (30 steps before to 30 after): at epoch 1000 the index carried 0.40 of its
  force inside 30 steps before and 0.31 30 steps after (force drifts to the tip during a hold); at epoch 800 it carried
  0.03 throughout, with twice the fingertip force (10-11 N against 5-6 N).

**Why the round-17 depth weighting did not produce curling.** The round-17 reward was evaluated on the replayed states
(its own code, per held step):
- The envelope quality did not fall: env_q 0.692 -> 0.701. Its depth part fell (0.331 -> 0.309, z -3.9), but palm
  contact rose (0.825 -> 0.901, z 2.6), and the palm gain (+0.019 of env_q) is more than twice the depth loss (-0.008).
- Palm contact is paid three times on every held step: as 0.25 of env_q, inside `held` (0.3 + 0.7 palm) which gates
  lift, transport and hold, and again inside `hold_grip` (0.4 + 0.6 palm). The replay fits an index fingertip pushing
  the object into the palm while the thumb pushes harder from the other side: hold income per held step 1.26 -> 1.43,
  `held` 0.87 -> 0.91.
- The depth part is a SUMMAND averaged over five digits. One digit's whole swing from a fingertip press to a full curl
  moves env_q by at most 0.35 x 0.7 / 5 = 0.049; with each finger touching on a single link (link-count depth 0.48-0.52
  on all four fingers at both checkpoints) the index's actual change cost 0.006 of env_q, 0.02-0.03 per held step.
  Curling EVERY digit fully would have added only 0.31-0.38 per held step, 8-9 % of the downstream base (3.7-4.3).
- It is not the trust ramp and not saturation: the force-share trust was 0.92-0.999 on every digit (all digits press far
  above 0.3 N), env_q sat at 0.70, the digit-count ladder at 0.90. Opposition was saturated (0.98-0.99) and did not pay
  for the change (thumb opposition 0.87 -> 0.83, palm opposition 0.28 -> 0.35).

**Why more episodes ended early.** Replay, 256 episodes each, cause recomputed per environment from the environment's
own termination terms: 69 -> 106 terminations, 28 -> 56 of them before the object was lifted.
- Arm joint speed above 20 rad/s doubled, 0.098 -> 0.195 of episodes (z 3.2): the largest cause and the largest
  increase. 10 of them (epoch 1000: 1) happened with NO object contact, before any grasp, with the lowest hand link
  within 2.4 cm of the 0.205 m table top (median 1.4 cm) and the speed on wrist joints 5-7. The other 40 (epoch 1000:
  24) happened with the object grasped, during lift-off (14 against 4) or while carrying (26 against 20); in 17 of the
  50 runaways the object's speed jumps from under 1 m/s to over 1.5 m/s in a single step. Those showed no consistent
  relation to grip force or fingertip use.
- Objects knocked over before any grasp 0.047 -> 0.082; 13 of the 21 (epoch 1000: 5 of 12) with the lowest hand link at
  0.217-0.220 m. During training the lowest hand link across environments slid 0.222 -> 0.213 m while pen_table stayed
  0.0000 for the whole run: the table charge starts 2 mm above the top, but hand-link points sit inside the links (on
  the object a touching link reads 1-3 cm), so a finger resting on the table reads about 1-2 cm and is never charged.
- Out of the working area in y: 0.105 -> 0.117, the same at both checkpoints. The bound (y <= 0.25 m) is the right
  arm's; on the left arm the object spawns up to y 0.30 and the goal box reaches y 0.35, so 8 of 256 episodes end at
  their first step and 15-19 while carrying the object towards a goal near that line. That is an environment matter.
- The wrench fired in the last 10 steps of 0 and 7 of the terminated episodes: it is not the driver.

The operator's direction stands: the index to little finger should curl around the object instead of pressing with
their tips, and the hold itself was judged stable.
