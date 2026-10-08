Round 18 ran left `cp_l_m18` from the round-16 epoch-1000 snapshot with the round-18 reward (curl as a factor of the
envelope quality, table charge band moved to 2 -> 1 cm) for about 1,470 epochs (cumulative about 2,470) on vision-3090.
No Traceback and no PhysX error. It was stopped as a round end after epoch 1250 fell back below epoch 800 in a
fixed-tolerance evaluation; epoch 800 is kept as the best snapshot.

**Fixed-tolerance deterministic evaluation (tolerance 0.0911 m, wrench pinned at 20 N/kg, first episode of 256
environments, seeds 42 and 43: 512 episodes per checkpoint).** Against the epoch-1000 start (success 0.785):
- epoch 450: success 0.811, epoch 800: 0.842 (z +2.3, the best so far), epoch 1250: 0.785 (-0.057 against epoch 800,
  z -2.3). No round-17-style collapse: at epoch 800 round 17 was at 0.703.
- Goal chaining rose and stayed up: three-goal episodes 0.43 -> 0.59 -> 0.66 -> 0.67, successes per episode 2.41 -> 3.36,
  all five goals 0.26 -> 0.56.
- The index curls again: share of its cup force on its two inner links at the success step 0.315 (start), 0.253, 0.338,
  0.307 (round 17: 0.010). Over all five digits 0.188 -> 0.212 at epoch 1250 (z +2.6); ring 0.13 -> 0.19, little finger
  0.01 -> 0.04. Whether this is the curled grip the operator asked for is waiting for the operator's video verdict.
- Early terminations 0.248 -> 0.318, 0.322, 0.336, mostly after the lift. Contact force fell at every checkpoint (mean
  finger-link force per episode 3.52 -> 2.49 -> 2.13 N) while more digits and links touched. Drops stayed rare (0.002)
  and the tilt while held fell (worst 20 -> 16 deg).

**Why more episodes end early: a termination-cause replay** (seed 42, 256 episodes per checkpoint, the environment's
own termination terms recomputed per environment; phase read one step before the terminal step; rates with
per-environment standard errors):
- The whole excess is the object leaving the working area in y after the lift: 0.051 (start) -> 0.105 (epoch 800,
  z 2.3) -> 0.160 (epoch 1250, z 4.1). Arm runaway after the lift 0.078 -> 0.105 -> 0.055 (z 1.1, -1.1), tilt above 60
  deg after the lift 0.023 -> 0.008, object dropped below the table 0, hand into the table 0, arm joint limits 0.
  Terminations before the lift are back at the start's level (0.109 -> 0.125, 0.137; round 17: 0.219).
- Per carried goal the risk did not change. An episode starts one carrying leg per goal it goes for; round 18 chains
  more goals, so it started 930 and 907 legs against 765. Terminations after the lift per leg: 0.054 -> 0.061 -> 0.063
  (z 0.7, 0.8). Only the y exits per leg rose (0.017 -> 0.029 -> 0.045). A second replay of the start snapshot
  reproduced it (0.055 per leg, y exits 0.013).
- The y exits come from the environment's bound. The left arm still uses the right arm's bound (object y <= 0.25 m)
  while the goal box reaches y 0.35 and every new goal is the previous goal moved by up to 0.08 m per axis; a policy
  that chains more goals meets more goals beyond the line. Of the y exits after the lift, the current goal was beyond
  the line in 12 of 27 (epoch 800) and 8 of 41 (epoch 1250), and the object was already within the success tolerance
  of a goal inside the line in 0 and 8. In training the same exits rose from about 0.08-0.11 to 0.17-0.18 of episode
  endings in round 17 and round 18 alike, both of which chained more goals.
- The remaining y exits (15 of 27, 25 of 41) carried the held object past the line while the goal was inside it, most
  of them moving away from the goal at 0.6-0.8 m/s. Fast sideways moves of the held object happen in both directions
  (above 0.5 m/s on 302 and 174 trace steps towards and away from the line at epoch 800, 344 and 287 at epoch 1250,
  239 and 159 at the start); only one side has a bound within reach. Moving away from the goal is already charged by
  the signed progress term and the drift charge; no reward term was found that pays for it.
- It is not a weak hold. In the five steps before a y exit or a runaway after the lift all five digits touched with
  60-95 N (median) of grip; lifted episodes that terminated held with the same force as those that did not (60 vs 55 N
  at epoch 800, 51 vs 50 N at epoch 1250); the wrench fired on 2-3 % of the last ten steps. Runaways jumped from below
  10 rad/s to above 20 rad/s in one step in 26 of 34 and 17 of 22 cases, as at the start (18 of 27).

**The grip got lighter, partly because of the curl factor** (replay, every held step: object lifted and 3 cm up):
- Total finger force on the object 71 N (start) -> 65 -> 59 N. The ring finger moved force off its fingertip (9.2 ->
  5.2 N) and its inner-link share rose 0.02 -> 0.19; the curl factor pays for exactly that (its curl 0.03 -> 0.20).
- The thumb's inner links pressed less (24 -> 18 N), but the thumb's curl was saturated all along (share above one
  half, press trust 0.99), so the curl factor did not pay for that; it matches the squeeze charge above 40 N of total
  force (0.29 -> 0.25 per held step) and the lighter fingers it presses against.
- Hand curl 0.25 -> 0.28, envelope multiplier 0.55 -> 0.57, hold income per held step 0.98 -> 1.14. Palm contact 0.85
  -> 0.91.

**The round-18 table band worked.** The objects knocked over before a grasp with the lowest hand link at 0.217-0.220 m
(13 of 21 in round 17) are gone at epoch 800 (0 of 15) and rare at epoch 1250 (2 of 14).

The operator's direction stands: the fingers should curl around the object rather than press with their tips; the
hold was judged stable.
