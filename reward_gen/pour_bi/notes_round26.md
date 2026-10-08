NEXT ROUND REQUIREMENTS (operator 09.23, after watching the t2r_i24 ep1450 video).
VERDICT ON THE CURRENT RUN: the pouring posture is what the operator wants (palm over the tilted cup,
fingers underneath carrying it) and the arm joints are NOT saturated - "apart from the tremor the task
looks successful". My geo metric (palm position along gravity across the cup axis, measured from the hand
body origin) did NOT match what the video shows, so it must not be used as the pour-posture verdict;
judge the posture from the video and from the keep metrics (hand-in-cup drift, force share, force ratio).

1) WORK NEAR THE CENTRE LINE (y ~ 0), inside the visible workspace (operator: later the policy is driven
   from an FPP/head camera, so the manipulation must stay where that camera can see it).
   Measured on trace_view (ep1450, env 0): the SOURCE cup starts at y = -0.168 (its own side) and is
   carried across the centre line to y = +0.166 (max +0.193) during the pour; the receiver cup drifts from
   +0.164 to +0.237. So the whole pour happens on the receiver's side, far to the left.
   Wanted: the pour happens near y ~ 0 - the source arm must not cross the centre line, and the receiver
   should not be pushed further out to make room. The receiver already has side_margin / home_rcv; the
   source side has NO such term. Add the mirrored constraint for the source cup (and keep the existing
   receiver-side rules), or reward both cups being near y ~ 0 at the moment of pouring.
2) TREMOR IS ASYMMETRIC - the receiver (left) arm shakes about twice as much as the source arm, although
   both carry the same action-rate/flip weights. Same trace, sign-flip rate per action dim:
     source arm dims 0-5: 0.02 0.08 0.02 0.10 0.47 0.37  (mean 0.176), action-delta rms 0.133
     receiver  dims 9-14: 0.35 0.41 0.23 0.09 0.38 0.47  (mean 0.323), action-delta rms 0.209
   Arm joint speed rms is similar (0.368 vs 0.390 rad/s), so it is command chatter, not motion.
   Wanted: a heavier smoothness/flip charge on the receiver arm (it only has to hold the cup still), or
   scale the existing charge by how much that arm actually needs to move.
3) KEEP what this round proved: the lift-time grasp reference (keep) and its coupling to the rotation
   income. Trace ep1400: hand-in-cup drift 1.95 cm, share change 0.37, force 4.0x at 90-160 deg, versus
   3.6 cm / 0.51 / 5.0x for i23; source thumb drift 18.9 deg vs 61 deg in i18.
