Keep round 15 whole: it is the first round in which both arms lift, carry, settle at the goal and survive the
disturbance, and none of its terms collapsed. Every structural rule (a)-(e) stays, the budgets stay, the falling-only
drop and set-down costs stay, the open-pose factor limited to the approach stays, palm seating tied to engagement stays.

Change one thing, structurally: what the reward calls a real hold must keep improving until the hand forms a full
envelope, and the whole downstream income must scale with it.

1. Define ONE smooth envelope quality in [0, 1], computed once from quantities that are symmetric across the digits and
   object-agnostic: the sorted digit-count ladder extended so the fifth digit is a real step, the mean number of links
   per digit (wrap depth), palm contact, and thumb/palm opposition. No term may name a digit. Nothing that says
   "index", no per-finger weight, no finger-specific threshold: the policy must learn that the fifth digit and the
   deeper wrap are worth having because the SAME multiplier that pays for lifting and holding grows with them.

2. Multiply the lift, rise, transport, progress, hold-at-goal and success incomes by that envelope quality (with a
   floor, so a four-digit grasp still earns and still lifts). Target values: a four-digit wrap with the palm seated
   earns about 0.65-0.75 of the downstream income; five digits with two or more links each, palm and opposition earn
   1.0; a five-fingertip pinch without palm or wrap must not beat a deep four-digit wrap. The missing fifth digit then
   costs the policy roughly a quarter to a third of everything it earns while carrying and holding, on every step -
   several units per step instead of the 0.14 it costs today.

3. Keep `held` as it is (saturating at four digits, plain force) for everything that is a GATE rather than an income:
   the reach fade `(1 - held)`, the drop, stall, drift and descend costs. Those must not weaken when the envelope
   quality plateaus, and the disturbance must not be able to dissolve a real four-digit hold into penalties. Measure
   the envelope quality on the same soft contact forces, not on geometry alone.

4. Rule (b) still binds: with four digits the policy must still clearly prefer lifting and carrying to anything
   available on the table. Check the ceilings after the change: four-digit downstream income must remain far above
   the 1.2 grasp budget plus the decaying on-table income.

5. Start points are a user decision for the next round: continue from the round-15 policies (left e922, right e800,
   both already lifting with four digits) or restart from the stage-1 policies. The measure of success for this round
   is `contact/finger_index` rising from 0.000 (right) / 0.10 (left) and `contact/links_touching_at_success` rising
   from 5.4 / 4.8 while grasp, lift and success hold; `stage/envelope_ep` is the environment's own five-digit measure.
