# iter_01 (left only) — kuku-experiment-critic card HC1 (09-26 23:0x)
HC1: the symmetric reward has a palm-less local optimum (thumb + one finger); which arm falls in is decided by the basin. Supported by the tag comparison (see selection.json evidence). HC2 (left ctx asymmetry) not supported: palm surface distance, table cost, terminations match the right arm.
Hypothesis A (trained): pre-hold palm-seat bonus independent of digit count + shove relief while a digit touches and the palm closes in.
Judge windows (100-epoch means, advice only): palm_touching > 0.03 and index or middle > 0.05 by e3000 (right arm escaped at e2000-2500); grasp_ep >= 0.5 by e3500; lift_ep > 0.3 by e4000. Falsified if palm contact stays < 0.01 with a two-digit pinch past e3000.
