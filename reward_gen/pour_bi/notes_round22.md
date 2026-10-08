FULL REWRITE (round 21). The operator stopped t2r_i20 and asked for the reward to be written again from
the task description, using the facts below, instead of another patch on iter_20. Every earlier reward
(iter_16 .. iter_20) drifted into the same failure: the grasp is fine at pickup and then DEGRADES while
the arm rotates the cup. Write the reward around "establish an envelope grasp, then KEEP that grasp
posture through lift, carry, rotation and pour".

MEASURED (64-env deterministic traces at ADR 30, 900 steps, 60 Hz; contact threshold 0.5 N;
"wrap" = fingers whose middle OR distal link touches the cup, 0..5; "tip-only" = fingers touching with
the fingertip only; thumb1 = thumb abduction joint, commanded 0 rad for the whole episode):

t2r_i18 ep_2500 (the operator's current S2R candidate), phase means over grasped steps:
  phase            wrap  tip-only  palm N  thumb1 deg  cup tilt deg  lift m
  grasp   60-120   2.21    0.00     0.13      3.9         14.6       0.109
  lift   120-200   3.96    0.46     3.26     13.3         78.7       0.305
  carry  200-260   4.11    0.59     5.50     34.5        142.8       0.406
  tiltup 260-340   4.10    0.73     4.44     53.4        140.9       0.404
  pour   340-500   3.96    0.93     2.62     65.1        135.3       0.405
  late   500-850   3.61    1.27     1.25     63.5        116.0       0.395
  -> wrap peaks at 4.11 in carry and decays to 3.61; tip-only contacts rise 0.00 -> 1.27;
     palm contact force rises to 5.50 N then COLLAPSES to 1.25 N; thumb1 is pushed monotonically
     from 4 deg to 65 deg although its command is 0 the whole time. The cup slides out of the palm
     toward the fingertips as the thumb is forced open. This is the defect to fix.
  -> the policy also tilts far too early: mean cup tilt is already 78.7 deg during the LIFT window
     and 142.8 deg in carry, i.e. it flips the cup while raising it.

Cross-run summary (succ / in_target / spill, then wrap before-pour -> during-pour -> last 30 % of
episode, tip-only before -> during, median peak tilt, thumb1 deviation during pour, action sign-flip rate):
  i16 ep5600  0.95 0.91 0.07 | 2.20 -> 2.55 -> 2.29 | 0.50 -> 1.74 | 104 deg | 37.3 deg | 0.38
  i18 ep2500  0.83 0.91 0.07 | 3.12 -> 3.84 -> 3.55 | 0.74 -> 1.04 | 155 deg | 61.1 deg | 0.38
  i18 ep2900  0.55 0.58 0.06 | 2.65 -> 3.55 -> 3.37 | 2.01 -> 1.32 | 152 deg | 54.7 deg | 0.35
  i19 ep500   0.70 0.92 0.08 | 2.68 -> 3.47 -> 3.42 | 0.85 -> 1.33 | 151 deg | 66.5 deg | 0.35
  i17 ep900   0.00 0.00 0.02 | 1.93 (never poured: the source cup was never lifted)  | 0.51
  -> i16 poured at 104 deg with only 2.2-2.8 wrapped fingers and 2.08 tip-only fingers late: a
     fingertip pinch, which the operator rejected on video. i18 reaches ~4 wrapped fingers but still
     loses them, and it over-rotates to 155 deg (a full cup releases its first bead near 72 deg,
     a nearly empty one near 106 deg, so anything past ~110 deg is pure over-rotation).
  -> iter_19 raised action-rate penalties x10 and added a sign-flip penalty in every phase: the
     sign-flip rate barely moved (0.38 -> 0.35) while success and the grasp got worse. Do NOT repeat
     large ungated action penalties.
  -> iter_17 removed all stationary income with penalties active from step 0 and the source cup was
     then never lifted at all (task/src_cup_lift stuck at 0.003 m for 727 epochs). Do NOT make the
     approach/grasp/lift phases pay less than iter_18 does.

WHAT THE REWARD MUST DO (operator's three goals, in priority order):
1. Grasp posture that SURVIVES rotation. Pay for the envelope while the cup is held, not only when it
   is picked up: use ctx.src_wrap_count / ctx.rcv_wrap_count (fingers with middle/distal contact) and
   ctx.src_palm_force / ctx.rcv_palm_force. Require >= 3 wrapped fingers and a palm contact that does
   not collapse; charge fingertip-only holding. The income must be paid EVERY step the cup is held,
   including during the pour, so that losing a wrapped finger at step 600 costs as much as at step 200.
   Gate the tilt and pour income on the current wrap quality so that rotating with a degrading grasp
   earns less than rotating with a good one.
2. Stable control: keep the iter_18-level action-rate penalties in reach/grasp/lift, and only inside
   the both-cups-carried phase add a moderate smoothness cost (cup linear speed above ~0.15 m/s,
   cup angular speed above ~1.5 rad/s, receiver cup motion while the source is tilted).
3. ADAPTIVE pour (operator 09.21, replaces the earlier "bell at one angle" idea). Every video so far
   shows the cup moved to ONE angle and frozen there, for every fill level. The operator wants: carry,
   then rotate SLOWLY and CONTINUOUSLY from a fill-dependent start angle up to a fixed end angle.
     start angle  theta_start = 1.25 + (1 - clamp(ctx.bead_fill_level, 0, 1)) * 0.60 rad
                  (measured first-release angle: 72 deg full, 89 deg half, 100 deg at 20 %, 106 deg empty)
     end angle    theta_end   = 2.09 rad (120 deg), the same for every fill.
   OBSERVABILITY CONSTRAINT - this is why the previous design froze: the actor is a memoryless MLP and
   sees ONLY the initial fill (ctx.bead_fill_level, constant for the whole episode) and the current cup
   tilt. It does NOT see ctx.bead_in_source_frac. A target angle defined from bead_in_source_frac is
   invisible to the policy, which then settles on one compromise angle. So theta_start MUST be a function
   of ctx.bead_fill_level alone, and the desired motion must be expressible as "rotation speed as a
   function of (current tilt, initial fill)". bead_in_source_frac / d_in_target may still be used for
   the bead-transfer income and as gates, but NOT to define the angle the policy must be at.
   PAY FOR THE SWEEP, NOT FOR THE ANGLE. A reward for "being at angle X" is what produces jump-and-hold.
   Instead pay for forward tilt PROGRESS inside the window [theta_start, theta_end]:
     - income per step proportional to the positive tilt increment (tilt rate toward the receiver * dt,
       or the clamped window-progress difference), only while the tilt is inside the window, the lip is
       over the receiver mouth and above its rim, the pour direction is valid, and the grasp-posture
       gate of goal 1 is high;
     - multiplied by a SLOW-ROTATION band on the source cup angular speed: full pay for roughly
       0.05-0.35 rad/s, falling to zero by ~0.8 rad/s, so that a fast flip through the window collects
       almost nothing while a slow sweep collects the whole integral (the window is 14-48 deg wide, so a
       correct sweep lasts roughly 1-8 s);
     - a small stall charge while the cup is inside the window with beads still in the source and the
       tilt rate near zero, so freezing part-way is not a resting point; it must stay far smaller than
       the posture income so it never pays to rush;
     - below theta_start (and once lifted and near the receiver) a modest, non-repeatable gradient toward
       theta_start so the cup gets to the window; nothing may be paid for tilting during the lift;
     - past theta_end: no income at all, plus a linear unsaturated penalty. The traces show every run
       pinned at 150+ deg because flat income made flipping free.
   The progress income must not be farmable by rocking back and forth: pay only NEW progress (compare
   against a best-so-far that you can rebuild from ctx, or make backward motion cost as much as forward
   motion earns). ctx has no persistent storage of yours, so prefer the symmetric-cost form.
   The per-step cap on ctx.d_in_target income stays: a controlled stream must beat a dump.
   Also do not tilt during the lift: keep the cup near upright until it is lifted and near the
   receiver (ctx.premature_tilt_limit already latches success away, but the reward must also not pay).
KEEP what works in iter_18 and is unchanged: the success/nesting/cup_hit/pour-direction/receiver-side
rules, the bead transfer income on ctx.d_in_target increments with a per-step cap, the spill penalty,
and the lift/approach shaping strengths (iter_18 lifted reliably; iter_17 did not).
The environment now logs diag/* for all of this (wrap during pour, thumb abduction drift, flip rate,
cup speeds, per-fill-bin peak tilt), so the reward does not need to log anything.

ACCEPTANCE (operator 09.21: "even with fresh training, if it looks wrong in sim it cannot be good on the
real robot"). Success rate alone does not pass this round. The run is judged on the physical behaviour
the traces measure, against the i18 ep_2500 numbers quoted above:
  - wrap in the last 30 % of the episode >= 3.5 fingers (i18: 3.55) AND tip-only contacts <= 0.6
    (i18: 1.27) AND source palm contact force in the pour/late phases >= 3 N (i18 collapses 5.50 -> 1.25).
  - thumb abduction drift during the pour <= 15 deg (i18: 61.1 deg).
  - ADAPTIVE SWEEP (operator 09.21): per fill bin, the tilt at which the pour phase begins must
    differ - high-fill bin starts >= 10 deg lower than the low-fill bin (target ~75-90 deg at 80 %,
    ~100 deg at 20 %) - and every bin ends at 110-125 deg (i18: all bins 142-155 deg, i20: 142/146/146,
    i.e. no dependence on fill at all). Source cup angular speed while beads are leaving: median
    <= 0.4 rad/s, and the time from theta_start to peak tilt >= 1 s. One fixed angle held for the whole
    pour is a FAIL even if success is high. The cup stays near upright until it is lifted and close to
    the receiver.
  - action sign-flip rate <= 0.20 (every run so far: 0.35-0.38).
  - success >= 0.5 and spill <= 0.08 at ADR 30, with cup collisions, nesting and receiver-side
    violations at zero as in iter_18.
A reward that hits the success number while any of the posture numbers stays at the iter_18 level is a
failed round: the operator rejects it on video and it cannot transfer.

ROUND 22 ADDENDUM (operator 09.21: "set it up properly first, then train FRESH").
iter_21 was launched FRESH with its control penalties live from step 0. With a random policy the raw
commands flip sign on ~50 % of steps, so hand_flip (HAND_FLIP_W 0.4, gated only on "a cup is held") and
palm_sat charge the very first accidental grasps and teach "do not touch the cup". New scalar
ctx.adr_progress (0 until the running success rate first passes 0.3, then +1/30 per increment) exists for
this. RULE: every exploration-killing control charge - act_flip, hand_flip, palm saturation, and the
CARRY increment of the action-rate weights - is multiplied by ramp = smoothstep(adr_progress / 0.2)
(zero for a policy that has not learned the task, full from ADR 6/30). The iter_18 base action-rate
weights (0.01 / 0.005) stay on always - i18 learned FRESH with them. Task-shaping terms (posture, sweep
window, cup speed, overtilt, spill, premature tilt) are NOT ramped: they define the behaviour and must
be present from the start so the habit never forms. Everything else identical to iter_21.
