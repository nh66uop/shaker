"""Deterministic check of iter_22 c1r tilt_open: phi_rate must equal d(phi)/dt, and closed cycles must net <= 0."""
import math, types, torch
src = open('/home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/iter_22/compute_reward_c1r.py').read()
ns = {'RewardContext': object}; exec(src, ns)
torch.set_default_dtype(torch.float64); torch.manual_seed(0); N = 4096; R, MZ, BZ = 0.0281, 0.0539, -0.0599
def rot(u, w, dt):
    if isinstance(dt, float) and w.norm(dim=-1).max() <= 1.0 + 1e-9 and w.norm(dim=-1).min() >= 1.0 - 1e-9 and False: pass
    th = w.norm(dim=-1, keepdim=True) * dt; k = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return u*torch.cos(th) + torch.cross(k, u, dim=-1)*torch.sin(th) + k*(k*u).sum(-1, keepdim=True)*(1-torch.cos(th))
def ctx_of(ps, us, pr, ur, vs, ws, vr, wr):
    c = types.SimpleNamespace()
    c.src_cup_pos, c.src_cup_up, c.rcv_cup_pos, c.rcv_cup_up = ps, us, pr, ur
    c.src_cup_mouth_pos, c.rcv_cup_mouth_pos = ps + MZ*us, pr + MZ*ur
    c.src_cup_lin_vel, c.src_cup_ang_vel, c.rcv_cup_lin_vel, c.rcv_cup_ang_vel = vs, ws, vr, wr
    c.src_cup_tilt = torch.acos(us[:, 2].clamp(-1, 1)); c.cup_mouth_z = MZ; c.cup_radius = R; c.cup_bottom_z = BZ
    return c
def phi_and_rate(c):
    geo = ns['_pour_geometry'](c, R)
    vm = ns['_point_vel'](c.rcv_cup_lin_vel, c.rcv_cup_ang_vel, c.rcv_cup_pos, c.rcv_cup_mouth_pos)
    du = torch.cross(c.src_cup_ang_vel, c.src_cup_up, dim=-1)
    o = ns['_tilt_open_potential'](c, geo, R, vm, du, 1.0*R, 3.0*R, 0.7*R, 4.0*R, 2.0*R)
    return o['phi'], o['phi_rate']
# random states near the pour window: receiver mouth ~ (0.40, 0.03, 0.47), source spout above-beside it, tilted toward it
pr = torch.tensor([0.40, 0.03, 0.47 - MZ]).repeat(N, 1) + 0.01*torch.randn(N, 3)
ur = torch.nn.functional.normalize(torch.tensor([0., 0., 1.]) + 0.05*torch.randn(N, 3), dim=-1)
ang = torch.rand(N)*1.6; az = math.radians(68) + 0.6*torch.randn(N)   # tilt 0..92 deg, lean roughly toward +68 deg
us = torch.stack([torch.sin(ang)*torch.cos(az), torch.sin(ang)*torch.sin(az), torch.cos(ang)], -1)
ps = pr + MZ*ur - torch.stack([0.045*torch.cos(az), 0.045*torch.sin(az), torch.zeros(N)], -1) + torch.tensor([0, 0, 0.03]) + 0.02*torch.randn(N, 3)
vs, ws = 0.05*torch.randn(N, 3), 0.8*torch.randn(N, 3); vr, wr = 0.03*torch.randn(N, 3), 0.2*torch.randn(N, 3)
dt = 1e-6
def adv(s):
    return ctx_of(ps + vs*s, rot(us, ws, s), pr + vr*s, rot(ur, wr, s), vs, ws, vr, wr)
phi0, rate0 = phi_and_rate(adv(0.0)); fd = (phi_and_rate(adv(dt))[0] - phi_and_rate(adv(-dt))[0]) / (2*dt)
m = phi0 > 1e-4
err = (fd - rate0).abs()
print(f"states with phi>0: {int(m.sum())}/{N}")
print(f"|fd-rate| median {err[m].median():.3e}  p99 {err[m].quantile(0.99):.3e}  max {err[m].max():.3e}  |rate| median {rate0[m].abs().median():.3e}")
rel = err[m] / rate0[m].abs().clamp(min=1e-3)
print(f"relative err p99 {rel.quantile(0.99):.3e}  frac rel>1e-2: {(rel>1e-2).float().mean():.4f}")
# closed-cycle check: integrate the per-step tilt_open (payment/charge rule) around random smooth closed loops at dt=1/60
def step_value(rate_n, carry=1.0, sep=1.0):
    return carry*sep*torch.clamp(rate_n, 0.0, 1.0) + torch.clamp(rate_n, -100.0, 0.0)
T = 240; tot = torch.zeros(512); DT = 1/60
w_loop = 2*math.pi/(T*DT)
base_s, base_u = ps[:512], us[:512]
for k in range(T):
    t = k*DT; s = math.sin(w_loop*t)
    # loop: tilt oscillates and source moves out and back along the lean direction (closed path, back at start)
    axis = torch.nn.functional.normalize(torch.cross(torch.tensor([0., 0., 1.]).repeat(512, 1), base_u, dim=-1), dim=-1)
    wv = axis * (1.2*w_loop*math.cos(w_loop*t))
    off = torch.stack([0.06*s*torch.cos(az[:512]), 0.06*s*torch.sin(az[:512]), torch.zeros(512)], -1)
    vel = torch.stack([0.06*w_loop*math.cos(w_loop*t)*torch.cos(az[:512]), 0.06*w_loop*math.cos(w_loop*t)*torch.sin(az[:512]), torch.zeros(512)], -1)
    u_t = rot(base_u, axis, 1.2*math.sin(w_loop*t))
    c = ctx_of(base_s + off, u_t, pr[:512], ur[:512], vel, wv, torch.zeros(512, 3), torch.zeros(512, 3))
    _, r_ = phi_and_rate(c)
    tot += step_value(r_ / 1.0)
print(f"closed loops (512): max cycle sum {tot.max():.4f}  frac > +0.05: {(tot > 0.05).float().mean():.4f}  mean {tot.mean():.4f}")

# ---- localize: finite-difference each factor
def parts(c):
    geo = ns['_pour_geometry'](c, R)
    vm = ns['_point_vel'](c.rcv_cup_lin_vel, c.rcv_cup_ang_vel, c.rcv_cup_pos, c.rcv_cup_mouth_pos)
    du = torch.cross(c.src_cup_ang_vel, c.src_cup_up, dim=-1)
    o = ns['_tilt_open_potential'](c, geo, R, vm, du, 1.0*R, 3.0*R, 0.7*R, 4.0*R, 2.0*R)
    a = (c.src_cup_up[:, :2]*geo['d_hat']).sum(-1); h = c.src_cup_up[:, :2].norm(dim=-1).clamp(min=1e-6)
    return {'theta': geo['theta'], 'win': o['wnd']['win'], 'cos': (a/h).clamp(-1, 1), 'lip_xy': geo['lip_xy'], 'lip_dz': geo['lip_dz']}, o
p0, o0 = parts(adv(0.0)); pp, _ = parts(adv(dt)); pm, _ = parts(adv(-dt))
for k in ('theta', 'win', 'lip_xy', 'lip_dz'):
    fdk = (pp[k]-pm[k])/(2*dt)
    ref = {'theta': o0['dtheta'], 'win': o0['wnd']['dwin']}.get(k)
    if ref is None: print(k, 'fd median |.|', f"{fdk[m].abs().median():.3e}"); continue
    e = (fdk-ref)[m].abs(); print(f"{k}: median err {e.median():.3e} p99 {e.quantile(0.99):.3e} |ref| median {ref[m].abs().median():.3e}")
