iter_10 trained c1 (reward_gen/rh_aglt_r/iter_10/selection.json); no competing hypothesis. Command delay switched off (stage 1).

Runs (seed 42):
- aglt_r_i10 (right, fresh, 4096 envs, epochs 0-4600): goal successes rose with tolerance 0.0229 m at the end; no collapse.
- aglt_l_i10m (left, continued from the mirrored right epoch 3000, 2048 envs): ring contact rose to 0.52 of held steps by epoch 1800, but index fell to 0.39, little to 0.38 and the cup tilt grew to 18 deg median (p90 41 deg); the deterministic replay reached only 0.03 goal successes per episode at tolerance 0.02.

Deterministic replay of the right epoch 4600 (64 envs x 1800 steps, held steps, medians):
- Tolerance 0.0229: lifted in 1.00 of episodes, goal successes 2.30 per episode (0.88 at tolerance 0.02), no drops; the same weights mirrored onto the left arm give 2.02 / 0.83 with the same grasp.
- Grasp: palm 1.73-1.77 cup radii from the axis, palm at 0.41 of the mouth height (32 mm below the mouth, 82 mm above the bottom); fingertip heights / mouth height: thumb 0.64, index 0.55, middle 0.14, ring -0.30, little -0.59; thumb never inside the mouth; cup tilt 8.6 deg (p90 12.9).
- Contact > 0.5 N: thumb 1.00, index 0.99, middle 1.00, ring 0.00, little 0.94; median forces thumb 3.9, index 1.9, middle 1.4, ring 0.0, little 1.2 N; first links: thumb 1.00, index 0.99, middle 0.18, ring 0, little 0. Wrist joint 6 at its limit 0.15-0.18 of steps.
- The ring finger is commanded FULLY OPEN: its pre-clip policy mean is -1.37 (the action is clipped at -1), policy sigma 0.27, so almost no sample even starts closing it.
- Re-computing the iter_10 reward on these held-and-lifted steps: adding 1.2 N on the ring finger would raise the reward by 29 % (finger factor 0.875 -> 0.975, plus the min-finger part of the grasp-shape factor). But with the ring finger NOT touching, moving its tip from 1.8 to 1.25 cup radii (toward the wall) changes the held-at-goal reward by only +0.08 %: the only pre-contact pull is finger_wall (weight 0.10, a squared mean over four tips, logged at 0.007 per step) against a held income of about 12-16 per step. The income for the ring finger therefore appears only on contact, and the policy, whose ring finger is pinned open, never finds it.
- With the real-robot command delay switched on the same weights lift the cup in only 0.78-0.85 of episodes (handled in the separate stage 2, not part of this round).

User verdicts (10.01): "i10 is better, except that the ring finger alone does not touch"; "with iter11 I want all five fingers to take part"; "thinking of the work after pouring, it must grasp lower than now, below the cup mouth".
