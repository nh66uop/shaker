iter_01 trained the single candidate c1 (reward_gen/rh_aglt_r/iter_01/selection.json); no competing hypothesis.

Runs: aglt_r_i01 (right, 0-406 epochs) and aglt_l_i01 (left, 0-210 epochs), fresh, 4096 envs, start pose at the table front edge with the idle arm mirrored; stopped after ~20 minutes by the user's decision to add approach-orientation keeping.

Measured (training logs, last ~20 epochs):
- Approach: palm-to-cup 0.092 m (right) / 0.117 m (left); within 10 cm on 77 % / 40 % of steps. Grasp flag 0.10 / 0.06, lifted ~0.
- During the approach (not-yet-grasped steps): |palm normal z| (palm sensor local +z) 0.52 on the right and RISING (0.45 -> 0.52); 0.17 on the left. Palm normal component towards the body centre line 0.73 (right) / 0.90 (left). Thumb tip above the little-finger tip by 76-81 mm; flipped hands 0.
- On grasped steps: 2.0 / 2.2 fingers touch, of which 1.6 / 1.3 with their first link (iter_00: 0.8-0.9). Thumb opposition 0.005 / 0.10 (thumb on the same side as the fingers). Palm normal z +0.61 on the right (palm turned upwards while grasping) / +0.19 left. Held cup tilt 4.3 / 8.1 deg.
- Deterministic ledger on iter_01 c1: a halfway-approach state with the palm turned up (normal z 0.8) earns 0.0851 vs 0.0852 horizontal; the same state with the thumb tip below the little-finger tip earns the same 0.0852; the pre-grasp state with the palm facing away from the cup earns 0.264 vs 0.267; the table grasp with the thumb tip below the little-finger tip earns the same 1.2425.
