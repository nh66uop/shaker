"""Deterministic check of iter_17 c1 wrap-depth (force-distribution weighted), float64."""
import torch
torch.set_default_dtype(torch.float64)
FULL, FLOOR, REF, LINKS = 0.50, 0.30, 0.30, 2.0
soft = lambda f: torch.tanh(f.clamp(min=0) / 0.15)
def depth(f):  # f: (N,5,3) link forces [N]
    link_c = soft(f); count = (link_c.sum(-1) / LINKS).clamp(max=1.0)
    d = f.clamp(min=0).sum(-1); share = ((f[..., 0] + f[..., 1]) / d.clamp(min=1e-6)).clamp(0, 1)
    w = FLOOR + (1 - FLOOR) * (share / FULL).clamp(max=1.0); trust = torch.tanh(d / REF)
    return (count * ((1 - trust) + trust * w)).mean(-1)
def digits(inner0, inner1, tip):
    return torch.tensor([[[inner0, inner1, tip]] * 5])
cases = {
 "tip press 5N + inner brush 0.3N (round-16 habit)": digits(0.3, 0.0, 5.0),
 "curl: inner 2N+2N, tip 1N": digits(2.0, 2.0, 1.0),
 "curl light: inner 0.5+0.5, tip 0.3": digits(0.5, 0.5, 0.3),
 "tip only 3N": digits(0.0, 0.0, 3.0),
 "no contact": digits(0.0, 0.0, 0.0),
 "near-zero forces 0.01 each": digits(0.01, 0.01, 0.01),
}
for k, f in cases.items():
    r16 = (soft(f).sum(-1) / LINKS).clamp(max=1.0).mean(-1)
    print(f"{k:48s} round16 {r16.item():.3f} -> iter17 {depth(f).item():.3f}")
# properties on random forces
torch.manual_seed(0); f = torch.rand(20000, 5, 3) * 6 * (torch.rand(20000, 5, 3) > 0.4)
d = depth(f); r16 = (soft(f).sum(-1) / LINKS).clamp(max=1.0).mean(-1)
print("bounded [0,1]:", bool((d >= 0).all() and (d <= 1 + 1e-12).all()), "| never above round16:", bool((d <= r16 + 1e-12).all()))
# monotone: shift force from tip to inner at fixed total -> depth non-decreasing
base = torch.tensor([[[0.5, 0.5, 4.0]] * 5]); shifted = torch.tensor([[[1.5, 1.5, 2.0]] * 5])
print("tip->inner shift raises depth:", depth(shifted).item() > depth(base).item())
# tip pressing harder lowers depth
print("harder tip lowers depth:", depth(digits(0.3, 0.3, 8.0)).item() < depth(digits(0.3, 0.3, 2.0)).item())
