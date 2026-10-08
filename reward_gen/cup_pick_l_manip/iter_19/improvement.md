One change: replace the per-digit measure inside the round-18 curl factor. Keep everything else of the round-18 code
exactly as it is. The factor stays where it is (env_q = env_base x curl_factor, curl_factor = CURL_FLOOR + (1 -
CURL_FLOOR) x hand_curl, hand_curl = mean over the five digits, CURL_FLOOR 0.50); only what a digit's curl measures
changes. Round 18's measure (share of the digit's force on links k=0, k=1 against k=2) paid for two fingertip presses -
the index's last link laid flat on the wall (0.91) and the thumb's hooked tip (0.88) - and almost nothing for the one
digit that really rolled (0.16), because k=1 is the last link itself and the fingertip body is fixed onto it.

1. Curl = the digit FOLLOWS THE WALL. Per digit, from ctx.link_surface_dist of its three link points:
       lift_off = link_surface_dist.amax(dim=-1) - link_surface_dist.amin(dim=-1)                 # (N,5)
       follow   = exp(-(lift_off - FOLLOW_DEAD).clamp(min=0) / FOLLOW_SCALE)                      # (N,5) in (0, 1]
       digit_curl = digit_touch * follow                                                          # (N,5)
   with FOLLOW_DEAD = 0.010 m and FOLLOW_SCALE = 0.006 m, and digit_touch the soft contact the code already computes
   (a digit that does not touch gives exactly 0, no fallback). lift_off is how far the digit's farthest link point
   stands off the wall compared with its own closest one. It is relative (the same offset added to all three points
   changes nothing), reads no force and no joint angle, and treats every digit alike. Measured on the replayed held
   steps: digits bent at every joint along the wall 0.6-1.1 cm (curl 0.7-0.9), the last link laid flat with the other
   joints straight 1.9 cm (0.2), the hooked fingertip 2.0-2.5 cm (0.1-0.2). FOLLOW_DEAD is the resolution of the
   surface distance (it reads up to about 1 cm high at contact, and a joint point sits deeper inside its link than the
   fingertip point), not a pose target. Pressing harder buys nothing; moving the farthest link further off never raises
   the curl.

   Remove inner_share, press, CURL_FULL_SHARE and CURL_FORCE_REF. The inner-link force share must not come back in any
   form: not as a second factor, not as a summand, not as a gate. Do not read hand_q, hand_q_norm or any joint angle for
   this, do not add a force target, and do not name a digit.

   Size, measured by recomputing env_base with the round-18 code on the replayed held steps: at the switch today's
   policy loses 3-6 % of its downstream income (env_mult 0.565 -> 0.549 at epoch 800, 0.558 -> 0.523 at epoch 1000),
   because it earns about as much curl as before, now for the right digits. On the same held states a hand whose touching
   digits all follow the wall earns 78 % more than one where none does (curl_factor 1.0 against 0.55 for five hooked
   tips), and each digit that turns from a fingertip hook into a roll adds about 0.06 to curl_factor, about 8 % of
   everything earned while lifting, carrying and holding. The grasp, lift and hold chain keeps its order, because
   env_mult multiplies all of it alike and `held` / `hold_grip` do not see curl.

2. Keep everything else. Every structural rule stays: second half gated on the object's rise, the envelope quality as
   ONE multiplier on the downstream income with curl as a factor inside it, the ladder, palm and opposition inside
   env_base, no per-finger term, falling-only drop and set-down costs, no penalty on a persistent latch, open-pose factor
   only on the approach, palm seating tied to engagement, uprightness as a multiplier, first-success-only bonus, the
   goal-gated wrench, the round-18 table band. Every budget stays, CURL_FLOOR stays, the squeeze charge stays, and no
   speed or control charge is relaxed.

3. Do not answer the other measurements. Reaching around the object is not missing (the fingertip pads are past the
   widest point on 86-100 % of held steps), so do not add an arc or reach-around term. Do not pay more for links in
   contact than env_depth already does (the rolled ring finger presses mostly with its tip). The y exits of round 18 were
   the environment's bound and the operator has changed it; nothing in the reward names a world axis or a bound.
