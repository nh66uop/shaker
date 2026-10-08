iter_00 trained the single candidate c1r (reward_gen/rh_aglt_r/iter_00/selection.json); no competing hypothesis.

Runs: aglt_r_i00b (right arm, 0-1746 epochs) and aglt_l_i00b (left arm, 0-976 epochs), fresh, 4096 envs, stopped by the user. Start pose then was the palm at (0.22, -0.41, 0.48); it is now the table front edge (0.08, -0.30, 0.45) with the idle arm mirrored.

Measured (training logs, last ~30 epochs, all-step means unless stated):
- Approach works: palm within 10 cm of the cup origin on 62 % (right) / 32-75 % (left) of steps; palm-to-cup 0.10-0.15 m from 0.33 m at start.
- Grasp flag (thumb AND another finger > 1 N): 0.56 (right) / 0.47-0.71 (left) of steps. Lifted latch: 0.003 (right) / 0.49 (left).
- On grasped steps (estimated): 3.1 (right) / 3.8 (left) fingers touch the cup, but only 0.9 / 0.8 fingers touch with their first link; the rest touch with the fingertip pad only. Palm contact 0.00. Hand closure 0.33 / 0.37.
- Cup tilt averaged over ALL steps (including the approach at 0 deg) is 12-13 deg.
- User verdict on the deterministic videos (right ep1600, left ep850): "the arm clearly gets close to the cup well; the left arm lifted the cup but it is NOT an envelope grip".
- Frame reading [observation, not a verdict]: both arms grasp from the side (palm towards the cup side wall, not from above); the cup tilts 20-25 deg while held on the table (right) and almost horizontal while lifted (left); the finger identified as the thumb sits at the cup's top rim on the same (camera) side as the other fingers rather than opposite them.
- Deterministic ledger on c1r: a fingertip-only grasp (first links 0 N, same total force) earns exactly the same as the envelope grasp (1.1955 vs 1.1955); a held cup tilted 20 deg earns 1.151 vs 1.1955 upright (4 % less); the thumb on the rim earns 1.05 (12 % less).
