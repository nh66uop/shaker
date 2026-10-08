"""iter_05 c1 성질 검사 — 목표에서 든 상태(6)에서 엄지 반경·새끼 접촉력·손가락 기울기 스윕이 단조이고 바닥이 있는지,
그리고 결함이 있어도 들기 쪽 수입(4→5→6)이 여전히 오르는지."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "reward_gen"))
import torch
import t2r_aglt as T
from t2r_aglt import call_reward_fn, load_reward_fn  # noqa: E402

it = Path(__file__).parent
fn, _ = load_reward_fn(str(it / "compute_reward_c1.py"))
rows, park, jitter, sloppy = T.ledger_rows()
g6 = dict(rows[6][1]); cup = torch.tensor(g6["cup"], dtype=torch.float64)

def run(ctx_rows):
    tot, _ = call_reward_fn(fn, T._ctx_rows(ctx_rows)); return [round(float(v), 3) for v in tot]

# 1) 엄지 끝 반경 스윕(입구 높이 −1 cm): 0.3 r(안) → 1.6 r(바깥)
thumb = []
for rr in [0.3, 0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6]:
    tips = [list(t) for t in g6["tips"]]
    tips[0] = (cup + torch.tensor([rr * T.CUP_R, 0.0, T.CUP_MOUTH - 0.01], dtype=torch.float64)).tolist()
    thumb.append({**g6, "tips": tips})
t_s = run(thumb); print("thumb r/R 0.3..1.6:", t_s)
# 2) 새끼 접촉력 스윕 0 → 2 N
pk = [{**g6, "ff": g6["ff"][:4] + [f], "fp": g6["fp"][:4] + [min(f, 0.8)], "ft": g6["ft"][:4] + [max(f - 0.8, 0)]} for f in [0, 0.25, 0.5, 0.75, 1.0, 1.5]]
p_s = run(pk); print("pinky force 0..1.5 N:", p_s)
# 3) 손가락 아래 기울기 0..40°
tl = [{**g6, "axes": [0.0, 1.0, 0.0, math.cos(math.radians(a)), 0.0, -math.sin(math.radians(a))]} for a in [0, 10, 15, 20, 25, 30, 40]]
d_s = run(tl); print("fingers down 0..40 deg:", d_s)
# 4) 결함 파지로도 들기 사다리가 오르는가(엄지 입구 안)
lad = []
for dz, lifted, dxy, kw in [(0.0, False, (0, 0), {}), (0.05, False, (0, 0), {}), (0.14, True, (0, 0), {}), (0.14, True, (0.03, 0.02), {"near": 1.0})]:
    off = torch.tensor([dxy[0], dxy[1], dz], dtype=torch.float64); c = T.CUP_C + off
    base = dict(rows[3][1]); tips = [list(t) for t in base["tips"]]
    tips = [(torch.tensor(t, dtype=torch.float64) + off).tolist() for t in tips]
    tips[0] = (c + torch.tensor([0.5 * T.CUP_R, 0.0, T.CUP_MOUTH - 0.01], dtype=torch.float64)).tolist()
    palm = (torch.tensor(base["palm"], dtype=torch.float64) + off).tolist()
    lad.append({**base, "tips": tips, "palm": palm, "cup": c.tolist(), "lifted": float(lifted),
                "kp": float((c - T.GOAL).norm()), "hand_min_z": 0.215 + dz, **kw})
l_s = run(lad); print("thumb-inside ladder 3,4,5,6:", l_s)
mono = lambda a, inc: all((b >= x - 1e-6) if inc else (b <= x + 1e-6) for x, b in zip(a, a[1:]))
print("thumb outside-better (r>=1.0 > r<=0.8):", min(t_s[4:]) > max(t_s[:3]),
      "| pinky monotone up:", mono(p_s, True), "| tilt monotone down:", mono(d_s, False),
      "| floor (40deg > 0.6*level):", d_s[-1] > 0.6 * d_s[0], "| defect ladder rises:", mono(l_s, True))
