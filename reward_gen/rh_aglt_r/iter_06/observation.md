iter_06 trained the single candidate c1 (reward_gen/rh_aglt_r/iter_06/selection.json); no competing hypothesis. Right arm only (the left arm kept training the iter_05 reward).

Run: aglt_r_i06 (right, epochs 0-2600, stopped by the user). Fresh, 4096 envs.

Measured (training logs):
- The palm is within 10 cm of the cup from epoch ~440 (0.77-0.83). The grasp flag stayed at 0.000-0.004 until epoch ~1420, then rose to 0.37-0.43, but lifted stayed 0.001 for the remaining ~1100 epochs. Per-finger contact while held: index 0.73-0.87, little finger 0.73-0.91, middle 0.00, ring 0.00; thumb opposite 0.00; finger direction z -0.41 to -0.48. bounds_loss 2.9-3.2 at the end.
- Video ep2000 (user: "the analysis is right"): the hand lies over the top of the cup, fingers stretched horizontally across the mouth, one finger hanging down the far side; the cup is never lifted.

Trajectory ledger (new): deterministic replays in the RIGHT environment, 64 envs x 1800 steps, the full reward context saved every step, then each reward file re-evaluated on those real states. Traces: good_mirror = the left arm's iter_05 policy mirrored to the right arm (the grasp the user wants: four fingers around the side wall, thumb on the outer wall; grasp 0.81, lifted 0.76); bad_i03_rim (right iter_03 policy), bad_i05_thumb_in (right iter_05, thumb tip inside the mouth), bad_i06_hat (right iter_06, hand over the mouth). Step-average reward (about the episode return PPO compares), as a fraction of good_mirror:
- iter_03 reward: rim 0.93, thumb-in 0.85, hat 0.09
- iter_05 reward: rim 0.73, thumb-in 0.67, hat 0.10
- iter_06 reward: rim 0.40, thumb-in 0.37, hat 0.11
So iter_06 already ranks the wanted grasp far above the other converged grasps, yet the fresh right arm still ended in the hat.
- Why the hat is a trap under iter_06: on the table the grasp flag is on 47 % of steps with no lift; on those held steps it earns 0.676 per step (grasp_hold 0.33, approach-side terms ~0.14 on free steps), while hovering level beside the cup earns 0.371 (ledger row 2h) — 1.8x hovering, and leaving it for the side grasp means passing through lower-income states. The grasp flag only needs thumb contact and one other finger above 1 N, which the thumb pressing the rim from above satisfies; the grasp base income (1.05 x the floored shape factor, >= 0.63) is paid whatever the thumb position.
- Grasp geometry on held steps (thumb tip = tip link origin, radii of the cup, height as a fraction of the mouth height): good_mirror thumb 1.39 r / 0.74, palm 2.08 r / 0.58; hat thumb 1.48 r / 1.05 (above the rim), palm 2.12 r / 0.42; thumb-in thumb 0.95 r / 0.90; rim thumb 0.91 r / 0.79. The left arm's own replays: thumb 1.18-1.28 r / 0.50-0.67. The palm position does NOT separate the hat from the good grasp; the thumb tip height and radius do.
- With the same reward (iter_05) the left arm found the side grasp and the right arm the thumb-inside grasp: nothing before the grasp flag steers the thumb to the outer wall below the rim, so which grasp a fresh run meets first is left to chance.
