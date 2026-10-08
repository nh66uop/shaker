# Verified lessons from 20 earlier rounds of the same task (cup_pick, both arms, 09.20-09.26)

Every line is a measurement from a finished run.

1. From random weights the approach was learned when the long-range reach was shaped on plain palm-to-surface distance (weight 0.35, 0.6 far + 0.4 near, times (0.45 + 0.55 grip), fading once held): palm gap 0.045 / 0.037 m and grasp episodes 0.83 / 0.98 (left / right).
2. Any factor that shrinks the long-range reach income at random initialisation made the policy retreat or park. A side-wall factor on the far reach: palm gap rose 0.22 -> 0.37 m. An open-hand factor with floor 0.15 multiplied onto the reach with weight 0.22: reach income 0.003-0.007 per step against 0.037 for the plain reach in the same epoch window, while the action and arm-speed penalties stayed at -0.023 and -0.013; contacts fell 36 -> 1.5 and the palm gap froze at 0.20-0.23 m.
3. Without any open-hand signal the hand arrived already curled: grasp episodes 0.008, palm contact 0 after 2,075 epochs.
4. An open-hand or palm-proximity income that paid without digit contact produced a flat open hand pressed on the wall: +0.195 per step, hand curl 0.000, no digit contact for 1,400 epochs.
5. While digit contact was zero, any term sitting near its maximum became the resting point (seat + palm terms 1.17 per step against 0.07 for the fingers, contact 0.004).
6. When the object could earn income while still on the table, the policy held it there: 0.54-0.69 per step of goal-shaped income with the object on the table, lift episodes 0.007 after 5,000 epochs.
7. A drop penalty proportional to the episode-long height latch became the largest term (-0.75 per step) and the policy unlearned lifting; charging it only while the object falls removed the collapse.
8. When the downstream gate saturated at four touching digits, a four-digit grasp became optimal: the fifth digit was worth at most 0.14 per step, under 1 % of the downstream income.
9. Multiplying the downstream income by an envelope quality raised the right arm's envelope to 0.86; on the left arm the same reward came with contact force x2.6, arm runaway terminations 5 % -> 42 % and grasp 0.86 -> 0.38.
10. Fingertip pressing, measured by replay on 256 episodes: finger-root joints at 0.00-0.17 of their range, last joints at 0.85-1.00, 66 % of held digit-steps are root-straight / tip-folded, 93-98 % of the fingertip force is normal to the wall. Before lift the last joints sit at 0.11-0.63 and this shape is 4-31 %; after lift it is 47-100 % per digit.
11. A curl measure based on the share of force on the "inner" links paid for pressing, because the second sensed point of each digit is the distal link itself and the fingertip pad is fixed to it: a pad lying on the wall scores full curl; a genuinely rolled finger scored 0.16. The finger's first link (moved by its root joint) has no contact sensor and no ctx point.
12. A link-count wrap depth also scores fingertip pressing as full depth when the distal link and the pad touch together.
13. The spread of each digit's three surface distances (max - min of ctx.link_surface_dist over its three points) separates the shapes in replay: rolled 1.1 cm (thumb 0.6), middle-joint hook 1.9 cm, fingertip press 2.0-2.5 cm.
14. The same reward on the mirrored arm gave different grasps: left full-hand grasp with palm 0.38, right three-digit pinch with palm 0.001.
