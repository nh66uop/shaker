We control a bimanual robot: two 7-DOF OpenArm arms, each carrying an Inspire RH56F1 five-finger \
hand, standing at a table. The RH56F1 is an UNDER-ACTUATED hand: only 6 joints per hand are driven \
(thumb abduction, thumb flexion, and one flexion drive for each of the index, middle, ring and pinky \
fingers); the remaining finger joints follow the driven ones through fixed mechanical couplings, so a \
finger always curls as a whole. The RIGHT arm is the "source" arm (src_*), the LEFT arm is the \
"receiver" arm (rcv_*). Two identical cups stand on the table: the source cup (in front of the right \
hand, filled with {num_beads} small beads) and the receiver cup (in front of the left hand, empty). \
Positions are in metres in a frame whose origin is at the robot base; +z is up, +x is forward (away \
from the robot), +y is to the robot's left. The table top is at z = table_z.

The action space is a normalized `Box(-1, 1, ({num_actions},), float32)`:
  actions[0:6]   = source palm 6-DoF command (xyz + yaw/pitch/roll) — the arm is moved \
by a geometric-fabrics controller toward a palm target. From iter_08 the env runs the INCREMENT mode: \
the (EMA-filtered) command is the palm target VELOCITY as a fraction of a max speed \
(source 0.6 m/s·180 deg/s, receiver 0.4 m/s·120 deg/s); the target accumulates, stays inside a box and \
within a few cm of the measured palm. So +-1 means "max speed", 0 means "hold still" (before iter_08: absolute target offset)
  actions[6:12]  = source hand commands, one per driven joint \
(thumb abduction, thumb flexion, index, middle, ring, pinky; the four fingers are averaged into one common command). \
From iter_09 the hand also runs the INCREMENT mode: the command is the closing SPEED as a fraction of the max rate \
(0.0167 closure per step at 60 Hz = 1 s from open to closed, the RH56F1 vendor full-range speed): +1 = close at max speed, -1 = open at max speed, \
0 = hold. The measured closure (`*_hand_closure`) is exactly the integrated command (before iter_09: absolute closure target)
  actions[12:18] = receiver palm 6-DoF command (same mode as the source palm)
  actions[18:24] = receiver hand closure commands
The hand controller stops a finger automatically once one of its links touches its own cup \
(contact freeze); opening is always allowed. Fingers can only close when the palm is near its cup. \
The "cups" are slim shaker bodies about 6 cm in diameter and 11 cm tall, and the hand is small: a \
FINGERTIP (precision) grasp — thumb tip on one side, the tips of some fingers on the other — is the \
expected way to hold the cup; a full wrap-around power grasp is NOT required and usually not \
possible. The cups are light, so a finger brushing the cup easily knocks it over: approach slowly \
and touch it only with the fingertips.
