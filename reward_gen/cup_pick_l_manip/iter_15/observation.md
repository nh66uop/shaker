Round 15 ran left `cp_l_m15` to 922 epochs on vision-3090 (from the round-14 epoch-450 snapshot) and right `cp_r_m15` to
817 epochs on the local 5090 (from the round-14 epoch-800 snapshot). No Traceback, no PhysX error, no collapse.

**Both arms now lift, carry and hold under the disturbance.** Last five buckets: left grasp 0.84-0.89, lift 0.82-0.88,
far-start success 0.74-0.83, 1.42 successes per episode; right grasp 0.94-0.97, lift 0.91-0.96, far-start success
0.68-0.75, 1.25 successes per episode. The goal-gated wrench fires (gate fraction of steps 0.47 left, 0.62 right); on
the left the ADR level reached 1.0 (20 N/kg) at epoch 466 and stayed there to 922 with grasp, lift and success intact;
the right held 20 N/kg for 600 epochs (grasp 0.97, lift 0.95, far-start success 0.83 at its best).
The round-15 drop/set-down change did what it was meant to: no arm unlearned lifting this time.

**The grasp both arms converged on is a four-digit grasp, not an envelope.** Training tags over the whole right run:
`contact/finger_index` 0.000 (index never touches), thumb/middle/ring/pinky 0.77-0.82; at success thumb 1.00,
middle 0.99-1.00, ring 0.95-0.97, pinky 0.99, index 0.000, 3.94 fingers and 5.3-5.4 of the 15 links, palm 0.64-0.75
and falling over the last buckets. The environment's own five-digit envelope measure `stage/envelope_ep` is 0.000 for
the entire right run. Left: index 0.10-0.11 (0.09-0.11 at success), 3.93 fingers, 4.8 links, palm at success 0.91-0.93,
`stage/envelope_ep` 0.51-0.54.

A deterministic probe of the epoch-800 checkpoints (256 envs, 8 object species, 430 complete episodes on the right)
confirms it per species: the right lifts every species (0.89-0.95) with thumb + middle + ring + pinky against the palm
and the index at 0.00 contact on all eight, its links 31-44 mm off the surface; five-digit envelope 0.000 of episodes.
Palm contact while lifted falls with object size, 0.97 on the small cup to 0.32 on the largest, so the largest object
is carried by digits alone. The left forms a five-digit envelope in 0.11-0.39 of episodes depending on species.

**The reward makes the four-digit grasp optimal, and the arithmetic is plain.** Every downstream income - lift 2.10 +
rise 0.40, transport 1.60 + 0.90, hold 10.0, about 15 per step - is multiplied by `held`, and `held` saturates at four
touching digits (`digit_hold = (n_touch - 2) / 2`). The fifth digit is paid only inside the 1.2 grasp budget: the top
rung of the sorted count ladder (0.40 x 0.20 = 0.08) plus one fifth of the per-digit link mean (0.30 / 5 = 0.06), at most
0.14 per step, under 1 % of what four digits already unlock. Deeper wrap (more links per digit) likewise only moves the
1.2 budget. So the policy has nothing to gain from the fifth digit or from a deeper wrap, and the wrench at the goal
is survived with four digits, so nothing pushes it there either.

**Left-only early terminations are an environment box, not the reward.** In the same probe the left ends 0.11-0.24 of
episodes per species with `out_xy` (right 0.00-0.04) and tips 0.08-0.23 (right 0.06-0.13). The left mirror cfg keeps
the right arm's `object_out_y` box (-0.60, 0.25) while the left object spawns at y 0.00-0.30 and its goals reach
y 0.35, so a correctly carried object leaves the box. That is a cfg fix outside this reward.
