iter_06 was trained on both arms from random weights (right 5500 epochs, left 5170 epochs, same code, 4096 environments, seed 42) and stopped by the operator. Numbers are means over the last 100 epochs; the comparison is the earlier round that completed the task on each arm (right: iter_02 code, left: iter_01 code), same epoch window.

The origin-relative approach worked: the palm is within the graspable height band around the object origin 0.90 (right) / 0.83 (left) of the time (iter_04: 0.00; completed runs: 0.60-0.78), the palm faces the object 0.98 of the time, and the palm plane is within 2 cm of the object's side 0.66-0.76 of the time.

But the grasp does not form:

| measure | iter_06 right | iter_06 left | completed runs |
|---|---|---|---|
| palm to the object surface, cm | 5.9 | 6.1 | 2.9 |
| digits touching | 0.77 | 0.67 | 2.6-3.3 |
| proximal links touching | 0.00 | 0.04 | 1.6 |
| palm touching | 0.00 | 0.00 | 0.69-0.71 |
| index / middle touching | 0.006 / 0.000 | 0.018 / 0.010 | 0.05-0.81 |
| grasp (episode) | 0.23 | 0.34 | 0.97 |
| lift (episode) | 0.003 | 0.001 | 0.95 |
| measured finger curl of the whole hand | 0.51 | 0.63 | 0.23-0.36 |
| open-hand pose condition during the approach | 0.01 | 0.01 | 0.03-0.11 |
| entropy | 13.1 | 17.4 | 19.2 |

Per-term reward per step (right / left): reach 0.173 / 0.167, palm seating 0.054 / 0.052, closing 0.034 / 0.029 (completed runs 0.009-0.012), open hand 0.001 / 0.001, shove cost -0.020 / -0.022 (completed runs -0.007); every contact term (digit count, digit segments, palm contact, opposition, segment pull) is below 0.006.

In the iter_06 code the closing credit before contact is measured flexion times the nearness of each finger's links to the object surface, multiplied by a palm-seat factor that is not zero while the palm is still several centimetres away. A hand whose fingers are curled into a fist in front of the object brings its links close to the surface and earns closing credit while the palm stays away. The open-hand term is worth at most 0.04 per step.

Frame reading of deterministic replays at epoch 5000 (one environment per arm; an observation, not the operator's verdict): the right hand arrives beside the object at body height after about 3 s and stays there with the fingers curled into a fist against the side; the left hand goes lower, near the base of the object, fingers curled. Neither lifts the object; both open and withdraw at the end.
