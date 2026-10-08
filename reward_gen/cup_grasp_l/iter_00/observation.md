iter_00 trained only Hypothesis B (iter_00/selection.json); Hypothesis A was not trained.

Left arm, 5,460 epochs from random weights, measured in 500-epoch windows (tensorboard). The right arm ran the byte-identical reward on the mirrored scene at the same time.

- The approach was learned: palm-to-object gap 0.162 -> 0.085 m by epoch 1,000 and 0.063-0.067 m from epoch 2,500 on, reach episodes 0.81-0.95.
- The hand settled into a two-digit pinch and never left it: from epoch 2,500 to 5,460 the thumb touches the object in 0.73 of steps and the ring finger in 0.58, while the index finger touches in 0.0007, the middle 0.002, the pinky 0.012 and the palm 0.0016. Fingers touching 1.3 on average, links touching 1.4.
- The palm stays 0.040 m from the surface (palm surface distance, the same value the right arm had just before it formed its grasp) and never makes contact.
- Grasp episodes 0.05-0.15, lift 0.000-0.006, envelope 0, successes 0 for the whole run; the disturbance never engaged.
- The hand actions are saturated at +-1 in 0.95 of steps (right arm 0.57-0.66 over the same phase); the distal finger segments are commanded 0.85 closed and the proximal ones 0.14 to 0.30.
- The largest cost in this state is the shove penalty (-0.048 per step), larger than every grasp income term (grasp count 0.004, digit 0.007, palm 0.0003, opposition 0.017); palm seating pays 0.113 and reach 0.147.
- Table and termination costs are the same as on the right arm (table penalty -0.0008, no terminations), so nothing charges the left arm for moving down or in.
- The right arm went through the same state (palm contact 0.007, gap 0.063 m, fingers 0.07 at epochs 1,500-2,000) and left it at epochs 2,000-2,500 when the index finger (0.11) and the palm (0.03) started touching; it then reached grasp 0.92, lift 0.90, envelope 0.79, far-start success 0.61 with palm contact 0.65 and the index finger in 0.81 of steps.
