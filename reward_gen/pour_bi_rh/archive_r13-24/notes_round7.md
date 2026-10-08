Operator notes for round 7 (iter_06 result). Every number is measured. Hypotheses are marked as such. Env unchanged: hand-cup friction 2.0, contact-freeze threshold 1.0 N, grasp flag = thumb > 1.0 N AND another finger > 1.0 N.

## What iter_06 fixed (run t2r_rh2_i06, 1024 envs, warm from i05 epoch-550, 800 epochs)
Every stage before the pour now works, and all of it held to the end of the run:
- task/cups_center_dist 0.875 -> 0.358 m at the end (50-epoch means fell 0.749 -> 0.524). The cups reached their 0.32 m spawn separation. reward/converge, the new term on the cup-cup xy distance, went 0 -> 1.11 and reward/align_mouths 0.201 -> 0.848.
- task/src_cup_lift 0.35 -> 0.106 m, inside the 0.15 m target. The new band plus over_lift ramp held it there for 800 epochs; reward/over_lift_src stayed near -0.01, i.e. the policy sits inside the band rather than being dragged back into it.
- task/rcv_cup_lift 0.221 -> 0.054 m, inside its 0.10 m band (50-epoch means 0.049). reward/hold_rcv 0 -> 0.112 and reward/lift_rcv 0.024 -> 0.878, so the new receiver income is being collected.
- task/src_grasped 0.57 -> 0.59 and task/rcv_grasped 0.435 -> 0.60, both the best of any run on this track. reward/grasp_rcv recovered 0.43 -> 0.748.
- rewards/iter 2986 -> 4417.
Note on a wrong operator call during this run: at epoch 217 the receiver was stuck at 0.189 m and hold_rcv paid 0.0018, and the operator concluded the incentive could not reach it. That was wrong - by epoch 342 the receiver had come down to 0.057 m on its own. It was a 130-epoch learning delay, not a dead incentive.

## What iter_06 did not fix
reward/pour_tilt peaked at a 50-epoch mean of 0.0070 and ended at 0.0016; task/episode_success and bead/in_target were 0.000 at every epoch; reward/pour_delta was 0 in the last 12 epochs (all-time max 0.0293).
task/src_tilt_deg 50-epoch means: 23.3, 22.8, 26.2, 24.8, 22.2, 26.7, 24.9, 24.4. The tilt has not left the 21-27 deg band for 380 epochs, and reward/tilt_lead fell 0.508 -> 0.450 over the same stretch.
This time the distance excuse is gone: the cups are at 0.358 m, both lift bands hold, and both grasps are at their best. The tilt is what is stuck.

## A LEARNING-DYNAMICS FAULT THAT THE REWARD CANNOT FIX (measured this round, applies to every run on this track)
The policy's action noise has been growing without bound for about 3000 epochs across i04, i05 and i06, and it is large enough to corrupt the very metrics above.
- losses/entropy rose monotonically 19.5 -> 28.1 across i04 -> i05 -> i06 and never fell once.
- The agent config has `fixed_sigma: True`, so sigma is a state-independent learned parameter. For a diagonal Gaussian, entropy = sum(ln sigma_i) + 24 * 1.4189 = sum(ln sigma_i) + 34.05. Computing that from the i06 checkpoints gives 26.84 at epoch 50 and 28.17 at epoch 750, against logged values of 26.94 and 28.06. The logged entropy IS the policy sigma; no environment randomness is involved.
- Per-dimension sigma from the i06 checkpoints (the action space is Box(-1, 1, (24,))), epoch 50 -> epoch 750:
  - arm / palm 6-DoF, both arms: 0.19-0.76 -> 0.19-0.77 (stable and tight)
  - four-finger flexion, both hands: 0.90-1.26 -> 0.94-1.31
  - THUMB ABDUCTION: source 1.66 -> 1.97, receiver 1.69 -> 2.13
  A sigma of 2.13 on a [-1, 1] box puts most of the sampled mass outside the box, where it clips. That channel is effectively a random number every step, and it is still growing.
- The agent config already documents this exact failure at a higher setting: "08.21 fab_test8: 0.002 -> 0.0005. 0.002 was this repo's best MLP value, and after ep2100 the policy sigma inflated 0.76 -> 1.30 and the total reward regressed 3017 -> 2045." The present value 0.0005 is still inflating it, to 2.13.
- Likely mechanism (HYPOTHESIS, not verified): the hand controller freezes a finger once one of its links touches its own cup, so those action dimensions have no effect in exactly the states where the grasp matters. With no gradient either way, the entropy bonus inflates them for free.
- Consequence for reading every metric in this file: the deterministic play trace of i04 grasped with both hands in 4 of 4 envs while the training curve for the same policy read task/src_grasped 0.52. The plateaus reported from training curves are measurements of a policy that is being drowned in its own action noise. Tilting the source cup past 45 deg requires holding a grasp while the wrist rotates, and the thumb abduction command is random at every step.
- Learning rate and KL, for completeness: info/kl sat at 0.017-0.020 for the whole run, above the configured kl_threshold 0.013, but `lr_schedule: linear` selects LinearScheduler, which decays by epoch and ignores KL (verified in the rl_games source). The threshold is inert. info/last_lr was 2.79e-4 of 3e-4 at epoch 745, only 7 % decayed against a 10000-epoch horizon. An adaptive schedule would cut only above KL 0.026 and raise only below 0.0065, so it would sit in its dead band here; the learning rate is not the problem.

## Operator changes to the agent config for the iter_07 run (not part of the reward)
These are applied at launch, alongside the new reward. They are listed so the reward is not written to compensate for them.
- `entropy_coef` 0.0005 -> 0.0, to stop the free inflation of sigma.
- The run starts warm from the i06 epoch-750 checkpoint with the policy sigma reset to 0.5 for all 24 dimensions (train.py `--sigma`, which the stock rl_games in use applies on restore). This pulls the thumb channels down from 2.13 and costs the arm channels a little, which are at 0.19-0.77.
- `lr_schedule` is left at linear for the reason given above.

## What the operator wants from iter_07
1. The source cup must tilt past 45 deg. Measured: it has sat at 21-27 deg for 380 epochs with the cups already converged at 0.358 m, and `tilt_lead` saturates at 45 deg by construction (TILT_PRE), so above that angle only `pour_tilt` has any slope.
2. `pour_tilt` is gated by `pour_gate`, a Gaussian on d_align with sigma 0.15 m. At the measured working distance that gate is small, so the only term that pays past 45 deg is nearly switched off exactly where the policy lives. Size it on the distance the run actually holds, or make the gate a consequence of convergence rather than a separate hard gate. The regenerator of iter_06 flagged this in advance as the next lever.
3. Keep everything that now works and do not re-tune it: the source lift band and over_lift ramp, `converge`, `carry_src`, `align_mouths`, the receiver band with `hold_rcv` and `rcv_disc`, the receiver anti-drag penalties, and the grasp formulation for both hands.
4. Assume the action-noise fault above is being fixed at the agent level. Do not add reward terms that try to suppress action noise.

## Launch plan
iter_07 starts warm from the i06 epoch-750 checkpoint with sigma reset to 0.5 and entropy_coef 0, 1024 envs, same env physics.
