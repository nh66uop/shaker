Keep the round-17 code except two things. The object is lifted, carried and held under the disturbance, the operator
judged the hold stable, and goal chaining improved. Every structural rule stays (second half gated on the object's
rise, envelope quality as ONE multiplier on the downstream income, no per-finger term, falling-only drop and set-down
costs, open-pose factor only on the approach, palm seating tied to engagement, uprightness as a multiplier,
first-success-only bonus, goal-gated wrench). Every budget stays, and no speed or control charge is relaxed.

1. Curl must be a FACTOR of the envelope quality, not a summand that palm contact can buy back. Round 17 weighted the
   wrap-depth summand by each digit's inner-link share; the policy paid for moving the index onto its fingertip (-0.006
   of env_q) with palm seating (+0.019), and even a full curl of every digit was worth only 8-9 % of the downstream
   income. Replace that weighting with a direct curl measure:
   - For each digit, curl = the share of that digit's OWN cup force carried by its two inner links (k = 0 and k = 1)
     against its fingertip (k = 2), full at about one half, multiplied by a smooth trust that the digit presses (a few
     tenths of a newton). A digit that does not touch contributes 0, never a fallback value, so taking a finger off the
     object can never raise it. Same formula for all five digits, no digit named, relative (never an absolute force
     target), no joint angle.
   - Hand curl = the mean of the five digit curls.
   - Multiply env_q by (CURL_FLOOR + (1 - CURL_FLOOR) x hand curl), so the digit ladder, palm and opposition can no
     longer pay back a fingertip press. The depth summand returns to round 16's link count: the force share enters
     once, as this factor.
   - Size it from the replay: today's grasp (thumb 0.64-0.74 of its force inside, the four fingers 0.02-0.08, hand curl
     about 0.25) keeps about three quarters of the downstream income it earns now; a hand whose five digits each carry
     half or more of their force inside earns all of it; each further digit that curls is worth about 6 % of all lift,
     transport and hold income, as much as the palm seating the fingertip press bought (env_q +0.019 and `held` 0.87 ->
     0.91, together 5-8 % of that income per held step). The floor keeps rule (b): with today's grasp, lifting and
     carrying still clearly beat anything available on the table.

2. The table charge must fire where a hand link actually touches the table. Its floor half charges only when the
   lowest hand-link point is within 2 mm of the table top, but those points sit inside the links, so it stayed 0.0000
   for the whole run while the hand came down onto the table. Of the new early terminations before any grasp, all 10
   arm runaways without object contact ended with the lowest hand-link point within 2.4 cm of the top (median 1.4 cm),
   and so did 13 of the 21 objects knocked over (1.2-1.5 cm); the other terminations ended with it at a median of
   24-28 cm. Move the band up to where a link resting on the table actually is: start charging about 2 cm above the
   top and reach the full floor charge by about 1 cm. Keep P_TABLE's cap of 0.6 and its palm half, and keep it a
   function of the current height only (no latch, bounded), so a hand wrapping the object's side wall above the band
   is never charged.

3. Nothing else changes. The left arm's out-of-area bound (y <= 0.25 m, inherited from the right arm) ends 3 % of
   episodes at their first step and 6-7 % while carrying; that is an environment matter handled at launch. The
   runaways during lift-off and carrying (the object's speed jumping in one step under a firm grasp) showed no link to
   grip force or fingertip use in the replay and are not a reward matter this round.
