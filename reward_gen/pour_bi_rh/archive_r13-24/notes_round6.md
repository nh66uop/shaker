Operator notes for round 6 (iter_05 result). Every number is measured. Hypotheses are marked as such. Env unchanged: hand-cup friction 2.0, contact-freeze threshold 1.0 N, grasp flag = thumb > 1.0 N AND another finger > 1.0 N.

## What iter_05 fixed (run t2r_rh2_i05, 1024 envs, warm from i04 epoch-1650, 569 epochs)
50-epoch means from the start of the run to its end:
- reward/lift_src 0.511 -> 2.18 (iter_04 ended at 0.117 after a 330-epoch ceiling at 0.2).
- task/src_cup_lift 0.029 -> 0.35 m (iter_04 deterministic trace: 20-76 mm).
- reward/tilt_lead 0.113 -> 0.31 and task/src_tilt_deg 12.9 -> 19 deg. In iter_04 there was no tilt channel at all and the source cup never left cup_up_z 0.87-0.96.
- reward/carry_src 0.043 -> 0.239, reward/align_mouths 0.039 -> 0.201.
- task/src_grasped 0.486 -> 0.570, rewards/iter 1874 -> 2986.
So the lift rebuild (iter_18 shape, attitude corridor, LIFT_FLOOR 0.60) and tilt_lead both worked.

## What iter_05 did not fix, with the measured cause
reward/pour_tilt was 0.000 at every epoch and task/episode_success 0.000 at every epoch.

1. THE SOURCE LIFT HAS NO UPPER BOUND. `lift_shape_src = 0.5*(1-exp(-h/0.03)) + 0.5*clamp(h/0.15) + 0.15*clamp((h-0.15)/0.10)` keeps paying to 0.25 m and never charges for going higher. Measured result: task/src_cup_lift 50-epoch means 0.144, 0.335, 0.298, 0.350, 0.260 m against LIFT_TARGET_SRC 0.15 m, with per-epoch values ranging 0.18-0.57 m.
2. THE TWO CUPS SEPARATED. task/cups_center_dist 50-epoch means 0.379 -> 0.875 m (spawn separation is about 0.32 m); last 10 epochs 0.514 m. The logged aim_dist went 0.30 -> 0.75 m.
3. THAT SEPARATION SWITCHES BOTH TILT CHANNELS OFF (read from iter_05 code, checked against the logs):
   - `tilt_lead` multiplies `(0.3 + 0.7 * clamp(1 - d_align / LEAD_RANGE))` with LEAD_RANGE 0.60. It was sized on the then-measured aim_dist 0.32 m, which gives 0.63. At the d_align 0.87-0.97 m that the run actually reached, `lead_near` is 0 and the factor collapses to its floor 0.3.
   - `pour_tilt = both_up * pour_gate * beads_left * upright_rcv * tilt_beyond`, and `pour_gate` is a narrow Gaussian on d_align. At ~0.9 m it is numerically zero. This is why pour_tilt stayed 0.000 for the whole run even though the source cup did tilt to about 19 deg.
   - So the 45 deg cap in tilt_lead is NOT what stopped the tilt at 19 deg. The distance condition is.
4. THE INCENTIVE RATIO FAVOURS LIFTING OVER CONVERGING. Measured at the end: lift_src 2.18 against carry_src 0.239 and align_mouths 0.201, i.e. about 10 to 1 against closing the gap.
5. THE RECEIVER DISCIPLINE ERODED ITS GRASP. reward/grasp_rcv 50-epoch means 0.734, 0.722, 0.738, 0.747, 0.620, 0.576, 0.557, 0.503, 0.495, 0.428, 0.515. task/rcv_grasped 0.497 -> 0.404, recovering to 0.435. reward/close_rcv 0.115 -> 0.065, reward/reach_rcv 0.964 -> 0.848.
   The receiver lift band itself behaves as designed but was simply abandoned: task/rcv_cup_lift 50-epoch means 0.091 -> 0.221 m against RCV_LIFT_FREE 0.10 m, and reward/lift_rcv collapsed 0.307 -> 0.024 as a result. The policy accepted losing that income rather than staying in the band.
   reward/rcv_still stayed at -0.15 to -0.23 and reward/rcv_home at -0.12 to -0.25 throughout.
6. losses/entropy rose monotonically 25.5 -> 26.9 over the whole run; it has never fallen in any run of this track.

## What the operator wants from iter_06
1. Bound the source lift. It must stop paying, or start costing, above a height from which pouring is possible. Measured overshoot: 0.35 m mean, 0.57 m peak, against a 0.15 m target.
2. Make closing the gap between the cups worth clearly more than lifting higher. Measured ratio is about 10:1 the wrong way (lift_src 2.18 vs carry 0.239 + align 0.201). The pour cannot start while the cups sit 0.5-0.9 m apart.
3. Size the tilt distance conditions on the distance the policy actually reaches, or make them a consequence of convergence rather than a hard gate. Measured: LEAD_RANGE 0.60 was set for 0.32 m and the run lives at 0.5-0.97 m; pour_gate is numerically zero there.
4. Stop the receiver grasp erosion (grasp_rcv 0.73 -> 0.43-0.52, rcv_grasped 0.50 -> 0.40-0.44, close_rcv 0.115 -> 0.065) while keeping the anti-drag intent that did work. Staying inside the receiver band must be worth more than abandoning it.
5. Keep what works: the grasp formulation for both hands, the iter_18-shape lift-off gradient, the attitude corridor, and tilt_lead as a channel (it moved the tilt from nothing to about 19 deg).

## Launch plan
iter_06 starts warm from the i05 epoch-550 checkpoint, 1024 envs, same env physics.
