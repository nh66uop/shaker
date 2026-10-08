Operator notes for round 5 (iter_04 result). Every number is measured. Hypotheses are marked as such. Env unchanged: hand-cup friction 2.0, contact-freeze threshold 1.0 N, grasp flag = thumb > 1.0 N AND another finger > 1.0 N.

## Outcome of t2r_rh2_i04 (iter_04 reward, 1024 envs, started from the i03 epoch-2000 checkpoint, 1650 epochs)
iter_04 fixed what it was written for: the receiver no longer abandons its cup (task/rcv_near_rate 0.84 at e1583 vs 0.05 in i03), and task/rcv_grasped rose 0.00 -> 0.52. But task/episode_success, reward/pour_tilt and reward/success_bonus were 0.000 at every epoch, and the source lift stalled.

reward/lift_src, 50-epoch means from e1000: 0.047, 0.062, 0.090, 0.126, 0.154, 0.214, 0.196, 0.201, 0.227, 0.228, 0.202. It rose to 0.2 by e1250 and was flat for the next 330 epochs.
Over those same 330 epochs reward/rcv_progress rose 0.641 -> 0.665, so lift_gate = 0.15 + 0.85 * rcv_prog went from 0.70 to 0.72. THE GATE IS NOT THE BLOCKER: it opened while lift_src stayed flat.
At e1574-1583 (10-epoch means): reward/carry_src 0.068, reward/align_mouths 0.079, reward/pour_tilt 0.000, task/src_grasped 0.54, task/rcv_grasped 0.52, reward/grasp_src 0.71, losses/entropy 25.19 (monotonically rising for 1500 epochs), rewards/iter 2176.

## Deterministic play trace of the epoch-1550 checkpoint (play --trace_steps 900, 4 envs)
Both hands grasp in 4 of 4 envs and hold it:
- Source grasped from step 100-200, receiver from step 200, both still grasped at step 800. Source finger forces: thumb 3-5 N, other fingers summed 2.5-5 N. Receiver: thumb 2-4 N, others 3-7 N. Hand closure 0.55-0.65 on both hands. Palm force 0 on both.
- SOURCE LIFT: cup height reaches only 20-76 mm (LIFT_TARGET_SRC is 0.15 m). Two of four envs peak at 55-61 mm around step 500 and sink back to 20 mm by step 800.
- SOURCE TILT: cup_up_z stays 0.87-0.96 the whole episode. There is no pouring tilt at all, in any env, at any step.
- RECEIVER: lifts its cup to 120-190 mm (LIFT_TARGET_RCV is 0.08 m) and drags it sideways 210-270 mm, still growing at step 800. Its cup_up_z drifts 1.00 -> 0.88-0.93.
- Because the receiver cup keeps moving, term grasp_rcv decays 1.6-2.0 -> 0.3-0.6, and term carry_src peaks at 0.31-0.61 near step 400 and decays to 0.09-0.26 by step 800. align_mouths swings 0.35-0.97 without settling.
- No cup ejection in the deterministic trace: the largest per-step cup motion (84-285 mm) occurs at step 897, which is the episode reset.

So the failure is not grasping. It is: the source is lifted only a third of the way and then sinks, the source never tilts, and the receiver drags its cup away so the meeting point never settles.

## Cross-track fact: the same source-lift stall was solved on pour_bi (DG-5F) in iter_18
Read from reward_gen/pour_bi/notes_round18.md and iter_18/compute_reward.py, and its live status.json at epoch 324.
- Their run t2r_i17 (727 epochs) had task/src_cup_lift 0.003 m and reward/lift_src 0.007 while the receiver was lifted and both hands grasped - the same shape of failure.
- Their recorded diagnosis: the early lift was multiplied by several sub-1 gates, and stationary income (wrap, home_rcv) was larger than what lifting the source added.
- After iter_18 their measured values at epoch 324 are task/src_cup_lift 0.187 m, reward/lift_src 2.98, reward/both_lifted 0.65, reward/converge 0.21, reward/raise_src 0.49. The source lift became learnable.
- What iter_18 does structurally (verified in its code):
  - `lift_shape_src = 0.5 * (1 - exp(-h_src / LIFTOFF_H)) + 0.5 * clamp(h_src / LIFT_TARGET_SRC, 0, 1)` with the comment "no product of sub-1 gates". Half the income is a steep lift-off part that pays in the first centimetres, half is linear so going higher keeps paying.
  - `src_up_soft = FLOOR + (1 - FLOOR) * src_up`: the uprightness gate has a floor instead of multiplying the lift to zero.
  - `lift_src = LIFT_SRC_W * held_src * lift_shape_src * src_up_soft * (1 + 0.25 * wc_src)`: the envelope quality is a bonus, not a gate.
  - `rcv_band = exp(-(clamp(h_rcv - RCV_LIFT_FREE, min=0) / RCV_LIFT_SIGMA)**2)`: the receiver is paid to lift only within a band, so over-lifting loses pay.
  - `rcv_still = -still_w * rcv_carry * tanh(clamp(|v_rcv| - RCV_STILL_FREE, min=0) / RCV_STILL_SCALE)`, with still_w increased while the source approaches and while pouring.
  - `home_rcv = HOME_W * home_src_f * rcv_carry * ... * rcv_home * still_f` where `home_src_f = HOME_SRC_FLOOR + (1 - HOME_SRC_FLOOR) * src_carry`: the receiver's stationary income is scaled by whether the source is being carried, so an idle receiver cannot out-earn the source lifting.
  - Staged terms after both cups are up: both_lifted, raise_src, converge, bring_together, aim.
- Limit of this cross-track evidence: at epoch 324 iter_18 still has reward/tilt 0.000, reward/pour_pose 0.000 and reward/aim 0.007. Only its lift and convergence lessons are demonstrated; its pour stage is not.

## What the operator wants from iter_05
1. Make the source lift pay from the first centimetres and keep paying past the target, in the iter_18 shape. No product of several sub-1 gates on the early lift. The receiver-progress gate on lift_src may stay only if its floor is high enough that the source lift can still grow; the measured evidence is that the gate at 0.70-0.72 was not what held lift_src at 0.2.
2. Bound the receiver: pay its lift only inside a band near LIFT_TARGET_RCV (measured overshoot: 120-190 mm against a 0.08 m target) and penalise receiver cup speed, so it stops dragging the cup 210-270 mm. The receiver's stationary income must not exceed what the source earns by lifting.
3. Give the source cup a tilt gradient that can start before a full-height lift. Measured: pour_tilt was 0.000 for the entire run and the source cup never left 0.87-0.96 of upright.
4. Keep the grasp formulation for both hands unchanged - it grasps in 4 of 4 envs deterministically with 2-7 N per side - and keep receiver reach.

## Launch plan
iter_05 starts warm from the i04 epoch-1550 checkpoint (both hands already grasp), 1024 envs, same env physics.
