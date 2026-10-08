Round 19 runs left `cp_l_m19` from the round-18 epoch-800 snapshot with the round-18 reward unchanged and one
environment change made by the operator: the left arm's object y bound is now the mirror of the right arm's. At the time
of writing it had run about 430 epochs on vision-3090, no Traceback and no PhysX error, and it keeps running: this
round is about the grip, not a round end. So far the bound change did what it was for. Object-out-of-area episode
endings fell from about 0.16-0.18 of episodes in round 18 to about 0.01-0.04 (training, done/out_xy x episode length),
successes per finished episode are 3.2 (round 18: 2.1-2.4), the tolerance went 0.10 -> 0.081, the wrench stays at level
1.0, five digits and 6.3 links touch at success with the palm on 0.90 of successes.

**The operator's verdict on the round-18 grip: still a fingertip press.** The operator compared the round-18 epoch-800
video with the round-16 epoch-1000 video on the largest cup, side by side while the cup is carried: in both the index,
middle, ring and little fingers press the object with their fingertips; neither curls or rolls the fingers around it.
The rise of the index's inner-link force share (0.338 at success, the round-16 level) does not count as curl. The curl
signal has to be redesigned.

**What the hand actually does** (geometry replay: seed 42, first episode of 256 environments, tolerance 0.0911 m,
wrench pinned at 20 N/kg, both snapshots; plus a replay of the exact 12-environment videos the operator judged). Each
digit was measured at five points - the start of the first phalanx next to the palm, the three link_pos points, and the
end of the fingertip pad - against the object's own outer wall, and its angle around the object axis from the palm:
- Every digit keeps its first joint next to the palm almost straight (0.00-0.17 of its range: thumb, index, middle, ring,
  little finger alike) and folds a later joint to the limit (last joint 0.85-1.00; the index folds its middle joint
  instead part of the time). The middle joint points of the fingers stand 1.8-3.9 cm off the wall while the fingertip is
  on it (0.6-1.1 cm). 93-98 % of the fingertip force is along the wall normal and the last link points into the wall.
  That is the press the operator sees: straight bars across the object with the tips driven into it. 66 % of all held
  digit-steps are this hook.
- Reaching around is not what is missing. The fingertip pads sit 115-148 degrees around the object from the palm and are
  past the object's widest point on 86-100 % of held steps, in both snapshots.
- The only digit that really rolls - every joint bent part of the way, every point within about 1.5 cm of the wall - is
  the ring finger at epoch 800 (32 % of its held steps, up from 0). The index alternates between the fingertip hook
  (40-63 %) and a second hook: first joint straight, middle joint folded to its limit, last joint straight, so the last
  link lies flat on the wall (25-33 %).
- The frames the operator judged (largest cup, 5 s): every finger touches only with its tip (0.1-0.9 cm from the wall)
  while at least one of its middle joint points is 2.5-3.8 cm off. Epoch 800 seats the palm (14 N); epoch 1000 does not
  (2.6 cm gap).

**Why the round-18 curl factor paid for a press.** It measured, per digit, the share of the digit's force on links
k=0 and k=1 against the fingertip link k=2. On this hand k=1 is the last link itself and the fingertip body is fixed
rigidly onto that same link, with its pad overlapping the last link's own collision surface. So "inner against tip"
splits one rigid segment, and a last link laid flat on the wall loads k=1: share 1, full curl.
- The index's flat-last-link hook scores 0.91 and its fingertip hook 0.00. The index's "recovery" in round 18 is a switch
  between these two hooks (mostly on the mid-sized cups and the closed shaker), not a curl.
- The thumb's fingertip hook scores 0.88: its last link is 5 cm long and carries the press. The thumb alone supplies
  about 0.17 of the 0.25-0.27 hand curl, whatever pose it takes.
- A real roll scores 0.16: the rolled ring finger still presses mostly with its tip.
- The share saturates at one half, so a little force on k=1 is enough for full credit.
- On the largest cup, the one the operator watched, the factor rated epoch 1000 above epoch 800 (hand curl 0.28 against
  0.20, 32 episodes each; the index 0.46 against 0.02).
- The first phalanx next to the palm, whose bending is the first step of any curl, has no contact sensor and no
  link_pos entry. The force share cannot see it.

**What separates a press from a curl, measured.** Whether the digit follows the wall: the spread of its three
link_surface_dist values (farthest minus closest link point of the same digit). A rolled digit 1.1 cm (thumb 0.6 cm), the
flat-last-link hook 1.9 cm, the fingertip hook 2.0-2.5 cm; the true radial distances give the same order. It reads only
context fields every digit shares. Three other candidates do not separate: how far around the object the fingertip
reaches (past the widest point in both snapshots), the number of links in contact (the rolled ring finger presses mostly
with its tip), and normal against tangential fingertip force (almost all normal in every pose).

The operator's direction stands and is now specific: the fingers should bend at every joint so that the whole digit
lies along the object, instead of straight bars with hooked tips. The hold itself was judged stable.
