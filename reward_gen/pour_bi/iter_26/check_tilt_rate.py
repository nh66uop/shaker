"""Deterministic check of iter_26 c1 tilt_rate (float64).
1) dphi = 0.5*(w_y*u_x - w_x*u_y) equals d/dt of phi=(1-u_z)/2 under rigid rotation du/dt = w x u.
2) closed tilt cycles before the first bead leaves net <= 0 (pay gate <= charge gate, pay capped),
   including cycles that start while `settled` still ramps (charge gate grows over time)."""
import math, torch
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
N, TILT_W, CAP = 4096, 5.0, 1.0
def rot(u, w, s):
    th = w.norm(dim=-1, keepdim=True) * s; k = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return u*torch.cos(th) + torch.cross(k, u, dim=-1)*torch.sin(th) + k*(k*u).sum(-1, keepdim=True)*(1-torch.cos(th))
phi = lambda u: 0.5*(1.0 - u[:, 2])
dphi = lambda u, w: 0.5*(w[:, 1]*u[:, 0] - w[:, 0]*u[:, 1])
ang = torch.rand(N)*2.5; az = torch.rand(N)*2*math.pi
u = torch.stack([torch.sin(ang)*torch.cos(az), torch.sin(ang)*torch.sin(az), torch.cos(ang)], -1)
w = 2.0*torch.randn(N, 3); h = 1e-6
fd = (phi(rot(u, w, h)) - phi(rot(u, w, -h))) / (2*h); r = dphi(u, w)
rel = (fd - r).abs() / r.abs().clamp(min=1e-3)
print(f"derivative: abs err max {(fd-r).abs().max():.3e}  rel p99 {rel.quantile(0.99):.3e}")
def step(x): t = x.clamp(0, 1); return t*t*(3-2*t)
def cycle(M=2048, T=300, DT=1/60, start_prog=0.0, ep_steps=900, settle=0.05):
    tot = torch.zeros(M); u0 = torch.tensor([0., 0., 1.]).repeat(M, 1)
    axis = torch.nn.functional.normalize(torch.randn(M, 3)*torch.tensor([1., 1., 0.]), dim=-1)
    amp = 0.2 + 2.0*torch.rand(M); nper = torch.randint(1, 4, (M,)).double()
    g_aim = torch.rand(M, T)            # arbitrary aim_gate*spout in [0,1], varies per step
    for k in range(T):
        t = k*DT; om = 2*math.pi*nper/(T*DT)
        th = amp*torch.sin(om*t)**2      # 0 -> amp -> 0, closed
        dth = amp*2*torch.sin(om*t)*torch.cos(om*t)*om
        uk = rot(u0, axis*th[:, None], 1.0); wk = axis*dth[:, None]
        prog = torch.full((M,), start_prog + k/ep_steps)
        g_charge = step(prog/settle)     # released = 0, want = 1 before the first bead
        g_pay = g_charge * g_aim[:, k]
        d = dphi(uk, wk)
        v = torch.where(d > 0, torch.clamp(TILT_W*g_pay*d, max=CAP), TILT_W*g_charge*d)
        tot += v*DT
    return tot
for sp in (0.0, 0.02, 0.2):
    s = cycle(start_prog=sp)
    print(f"closed cycles start_prog={sp}: max sum {s.max():.6f}  frac>1e-9 {(s>1e-9).double().mean():.4f}")
