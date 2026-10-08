Round 16 ran left `cp_l_m16` from the round-15 best snapshot (epoch 981) to epoch 800, then resumed as `cp_l_m16r` from
the epoch-800 snapshot to cumulative epoch 1266 on vision-3090 (rl_games restarts its epoch counter on resume; the
numbers below are cumulative). No Traceback, no PhysX error, no NaN. The run was stopped at 1266 as a round end.

**Training curves (moving average).** grasp 0.68 at 1100 -> 0.63 at 1221 (0.51 in the last 50-epoch bucket); far-start
success 0.54 -> 0.44 (0.35); three-goal episodes 0.28 -> 0.21; envelope peaked at 0.71 around epoch 940 and was 0.60
at 1221 (0.48 in the last bucket). The wrench ADR level stayed at 1.0 (20 N/kg) from about epoch 900 to the end.
lift_ema 0.71, lifted fraction 0.69-0.72, wrench near-goal fraction 0.44-0.47 and total reward 2.04-2.16 were flat.

**Part of that decline is the success tolerance, not the policy.** ctx.success_tol is not stored in the checkpoint:
on resume it restarted at 0.1125 and shrank again, 0.11 (buckets 0-150) -> 0.10 (200-350) -> 0.09 (400-450). The
success and goal-dwell drop (dwell 36 -> 11-14, successes per episode 1.40 -> 1.02) coincides with that shrink.

**The grasp got stronger, and the squeeze charge grew with it.** Over the resumed run pen_squeeze went -0.13 -> -0.20
-> -0.25 monotonically while wrap fraction rose 0.38 -> 0.53, grasp quality at lift 0.29 -> 0.60-0.64 and hand curl
0.27 -> 0.35.

**What the operator saw in the deterministic videos (epoch 800, 1000, 1250; one episode each, largest object):**
the envelope is formed and the object is held stably ("more stable than expected"), BUT the index, middle, ring and
pinky fingers press on the object with their FINGERTIPS instead of curling around it. The operator wants the fingers to
wrap (curl) around the object. The thumb is hidden from this camera.
Per-link contact is available to the reward: ctx.link_cup_force[:, f, k] for k = 0 (link moved by joint _3),
1 (link moved by joint _4), 2 (fingertip). A digit that "presses with the tip" carries most of its cup force on k = 2;
a digit that "wraps" carries force on k = 0 and k = 1 as well.
