iter_02 trained the single candidate c1 (reward_gen/rh_aglt_r/iter_02/selection.json); no competing hypothesis.

Runs: aglt_r_i02 (right, 0-1451 epochs) and aglt_l_i02 (left, 0-738 epochs), fresh, 4096 envs; stopped by the user.

Measured (training logs, 100-epoch bins right / 60-epoch bins left):
- The approach is learned: palm within 10 cm of the cup on 0.83 of steps from epoch ~800 (right) and ~300 (left), flat afterwards.
- The grasp never appears: grasp flag at most 0.011 (right, epochs 600-699) and 0.006 (left); 0.000-0.002 for the last ~750 epochs (right). At the same epochs iter_01 was at 0.053 (300-399) and 0.119 (400-413).
- The hand parks beside the cup half closed: closure near the cup (not grasped) 0.66-0.69 (right).
- Approach orientation (logged on not-yet-grasped steps): best at epochs 600-699 (|palm normal z| 0.22, towards the centre line 0.89), then drifting while parked: |n_z| 0.44-0.51, towards the centre line 0.47-0.79. approach_orient earns 0.12 per step throughout. Flipped hands 0.
- bounds_loss oscillates 0.5-1.06 and exceeds 1 in two bins (right).
- Deterministic ledger, new row pair: a sloppy FIRST grasp (grasp flag on, thumb and index tips at 1.2 N, first links 0 N, thumb on the fingers' side, hand half closed) against the parked state (open hand at the pre-grasp spot, half closed): iter_00 c1r 0.839 / 0.254 = 3.3x (grasps appeared early), iter_01 c1 0.586 / 0.254 = 2.3x (grasp 0.12 at epoch 400), iter_02 c1 0.586 / 0.394 = 1.49x (grasp never appeared). The pour track saw the same: parked at 1.38x, grasped at 1.86x.
