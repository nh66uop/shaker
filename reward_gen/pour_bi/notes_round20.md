Round 20 (t2r_i19, 4096 env, warm start from t2r_i18 ep_2500, epoch 545): the user REJECTED iter_19 in the play video: the right (source) hand grasp degraded, the left arm still trembles, and some episodes fail to grasp at all. The next run WARM-STARTS FROM t2r_i18 ep_2500 AGAIN (not from iter_19), so START FROM THE iter_18 REWARD CODE (previous_code is iter_19: remove what iter_19 added unless listed below) and keep every iter_18 weight that is not listed.

Measured facts (TFEvents of t2r_i19 and 64-env deterministic traces at ADR 30, 900 steps, 60 Hz policy):
- iter_19 applied palm/hand action-rate penalties x10 and act_flip (weight 1.0) in EVERY phase. Result at epoch 545: episode_success recent 0.456 (iter_18 was ~0.54), src_grasped 0.87 (iter_18 0.95 in trace), reward/act_flip stayed at -0.34/step, i.e. the policy paid the penalty instead of removing the flips. Trace sign-flip rate over all action dims: iter_18 0.375, iter_19 0.343 (worst dims 0.62 -> 0.55). The tremor was NOT solved and the grasp got worse.
- Tilt did NOT become fill dependent. Episode peak source tilt by initial fill bin (low / mid / high): iter_18 152 / 154 / 150 deg, iter_19 150 / 144 / 147 deg. The first-bead release angle is 72 deg for a full cup and 106 deg for a nearly empty one, so 106 deg is enough for any fill. iter_19 overtilt was -2.0*tanh (saturating, recent -0.62/step) against tilt + pour_pose income of roughly +10/step, and tilt income was FLAT beyond its peak, so flipping the cup over cost almost nothing.
- Source cup linear speed after step 300: median 0.03 m/s, p95 0.17 (iter_18) and 0.20 m/s (iter_19).
- in_target 0.72, spill 0.046, cup_collision_rate 0.003, premature_tilt_rate low: keep these channels as they are in iter_18.

ENV (already in code, unchanged from the iter_19 run): palm command EMA 0.25 then slew limit 0.03 per step. ctx.actions / ctx.prev_actions are the RAW policy outputs.

User goals for this round, in priority order. Change ONLY what serves them:
1. Stable control (tremor, speed) WITHOUT hurting the grasp.
   - Action penalties must be ZERO until both cups are grasped and lifted (multiply by the existing carry-both factor); the reach/grasp/lift phases must be exactly as in iter_18.
   - Use moderate weights inside that gate: palm action-rate about 0.05, hand action-rate about 0.02, sign-flip about 0.5 (iter_19 used 0.1 / 0.05 / 1.0 everywhere).
   - Add physical smoothness inside the same gate, from ctx.src_cup_lin_vel / src_cup_ang_vel / rcv_cup_lin_vel / rcv_cup_ang_vel: penalise source cup linear speed above ~0.15 m/s, source cup angular speed above ~1.5 rad/s, and any receiver cup motion above ~0.05 m/s / 0.5 rad/s while the source cup is tilted past the premature limit. Soft (tanh or clamp) shapes, each at most about -1/step.
2. Accurate posture: keep the iter_18 envelope (wrap) and pour-direction gates unchanged. In addition multiply the tilt and pour_pose income by a soft factor of the SOURCE wrap count (0.4 at 0-1 fingers up to 1.0 at >= 3 fingers) so that tilting with a fingertip pinch earns less than tilting with an envelope grasp.
3. Fill dependent tilt angle: theta_rel = 1.25 + (1 - clamp(bead_fill_level * bead_in_source_frac, 0, 1)) * 0.60 rad.
   - Tilt income rises up to theta_rel + 0.10 rad and then DECLINES linearly (not flat), reaching zero at theta_rel + 0.60 rad.
   - overtilt penalty is LINEAR and unsaturated: -6.0 * clamp(src_cup_tilt - (theta_rel + 0.25), min=0) per rad, clamped at -6.
   - Keep the iter_19 flow limit (pour_delta income clamped at FLOW_CAP 0.02, flow_excess weight 20); transferred and the success bonus unchanged.
The environment now logs diag/* metrics (flip rate, cup speeds while pouring, wrap-ok rates, tilt minus remaining-fill release angle, per-fill-bin episode peak tilt and success); the reward does not need to log anything extra.
