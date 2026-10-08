"""iter_04 c1 성질 검사 — 손바닥 기울기 스윕(컵 옆 머묾)에서 수입이 단조 감소·바닥 유지, 수평 행은 iter_03 과 같다."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "reward_gen"))
import torch
import t2r_aglt as T
from t2r_aglt import call_reward_fn, load_reward_fn  # noqa: E402  (rh_aglt_r.reward_context 로더)

it = Path(__file__).parent
rows, park, jitter, sloppy = T.ledger_rows()
hover = dict(sloppy[1][1])
res = {}
for tag, path in (("i03", it.parent / "iter_03" / "compute_reward.py"), ("i04", it / "compute_reward_c1.py")):
    fn, _ = load_reward_fn(str(path))
    sweep = []
    for nz in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        a = math.asin(nz)
        sweep.append({**hover, "axes": [0.0, math.cos(a), -math.sin(a), 1.0, 0.0, 0.0]})
    tot, _ = call_reward_fn(fn, T._ctx_rows(sweep))
    lv = T.ledger(path)["rows"]
    res[tag] = ([round(float(v), 4) for v in tot], lv)
    print(tag, "hover |n_z| 0..0.8:", res[tag][0])
s3, s4 = res["i03"][0], res["i04"][0]
mono = all(b <= a + 1e-6 for a, b in zip(s4, s4[1:]))
floor = s4[-1] > 0.5 * s4[0]
same = {k: (res["i03"][1][k], res["i04"][1][k]) for k in res["i03"][1] if abs(res["i03"][1][k] - res["i04"][1][k]) > 1e-4}
print("monotone_in_tilt", mono, "| tilted 0.8 keeps > half of level", floor)
print("rows changed vs iter_03:", same)
