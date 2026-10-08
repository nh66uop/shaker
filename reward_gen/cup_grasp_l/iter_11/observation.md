iter_11 (iter_10 plus touch-only flexion in the palm-seating engage factor) was trained on the right arm (cg_r_i09, local, fresh, seed 42) and stopped by the operator at about 545 epochs so that the new per-joint flexion log could be used. The same reward is now running on the left arm on another host. This note records what the per-joint replay showed.

Per-joint flexion (0 = the open start pose, 1 = joint limit), deterministic replay, 64 environments, 900 steps, split by phase (approach = palm more than 5 cm from the surface and nothing touching):

| | approach | touching | lifted |
|---|---|---|---|
| iter_02 right-arm run (succeeded), best checkpoint | 0.015 (every joint <= 0.06) | 0.25 | 0.35 |
| iter_11 right-arm run at ~500 epochs | 0.61 (index/middle joint 3: 0.84/0.89, ring joints 2 and 4: 0.98) | (1 % of steps) | - |

At epochs 400-500 the iter_02 run also had a half-curled hand (whole-hand curl 0.48, open-hand pose condition 0.01); it opened the hand from about epoch 735 (open-hand pose condition 0.15). The iter_10 run was still closed at epoch 735 (0.01).

Deterministic check of the iter_11 reward with the measured hooked hand put in: with the palm 7-10 cm from the surface (where the policy stays), the open hand earns only 0.03-0.04 more than the hook (the open-hand term, weight 0.04, which also switches off between 8 and 2 cm); the closing term is 0 at that distance. Inside 3 cm the hook earns more through the closing term, as intended.

Why an earlier policy never had this problem: the left-arm cup_pick policy cp_l_m08 was trained on top of a dedicated approach stage (reward_gen/cup_pick_l_approach/iter_01) whose reward paid an always-on open-hand income of 0.4, a 0.2 cost for commanding the fingers away from the open default pose, a 1.2 contact cost and a 1.5 hold bonus only with every joint within 0.25 of the open pose. Its note says: the finger channels are absolute targets, a = 0 is half closed, not open. The approach run learned the open hand from a half-curled start (curl 0.48 -> 0.39, open-hand pose condition 0 -> 0.25). The cup_grasp runs start from random weights, where action 0 half-closes every finger, with a 0.04 open-hand term.
