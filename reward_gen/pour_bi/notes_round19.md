Round 19 (t2r_i18, 4096 env, FRESH, epoch 2500+): episode_success recent ~0.54 at ADR 30, in_target ~0.70. The user ACCEPTED the behaviour in the play video except ONE defect: arm tremor. Keep the iter_18 structure and all its weights; change only what is listed below.

Measured on a 64-env deterministic trace of ep_2500 at ADR 30 (900 steps, 60 Hz policy):
- Action layout is 18 dims: 0-5 source palm 6D (xyz, 3 rotations), 6-8 source hand, 9-14 receiver palm 6D, 15-17 receiver hand. Dims 8, 9, 10, 13, 14, 17 flip sign between consecutive steps on 29-49 % of steps during steps 200-850; all other dims under 1 % (dims 6, 7, 16 are 4-10 %).
- Receiver palm command second difference rms is 0.07-0.16 (normalised units) vs 0.02-0.03 for the source arm. Receiver arm joint speed rms while it should only hold (steps 400-850) is 0.2-0.5 rad/s, about twice the source arm.
- Current smoothness terms are small next to the success income (+9.3): rcv_still recent -0.34, palm_rate_rcv -0.017, palm_rate_src -0.002.
- Source tilt peaks at ~155 deg for every fill level (6-20 beads) and first bead entry happens at 124-140 deg, although ctx.premature_tilt_limit is 52-63 deg: the policy flips the cup over instead of metering the tilt. Spill is 0.10-0.11 in the 6-8 and 16-20 bead bins, 0.03-0.04 in between.

ENV CHANGE for the next run (already in code): after the palm EMA (alpha 0.25) the palm command is slew-limited to 0.03 normalised units per step. ctx.actions / ctx.prev_actions are still the RAW policy actions, so the reward can see the chatter the filter hides.

REQUIRED in the new reward:
(a) A sign-alternation penalty on raw actions: charge relu(-(actions * prev_actions)) (product negative = sign flip), summed over all 18 dims, with a weight large enough that the measured chatter (dims with |a| step delta rms 0.4-0.7) costs on the order of 1-2 per step, i.e. comparable to a carry-phase income term, not 0.02.
(b) Raise the action-rate penalties (palm_rate_src, palm_rate_rcv and the hand dims) by roughly 10x and apply them in all phases including after success; the receiver must be charged while holding still.
(c) Saturate tilt income: no additional tilt income beyond ctx.premature_tilt_limit + 0.35 rad (20 deg), and a mild penalty for tilt beyond that + 0.35 rad more. Do not reduce income below that angle.
Everything else (cup_contact, cup_hit, prelift_near, side_margin, wrap, pour_dir gate, lift, home_rcv, pour_pose, pour_delta, transferred, success) stays as in iter_18. This run WARM-STARTS from t2r_i18 ep_2500, so keep reward scale of unchanged terms identical.
