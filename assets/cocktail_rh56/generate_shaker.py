#!/usr/bin/env python3
"""RH56F1 bayonet shaker generator (body / lid) — pure Python, no deps.

Design B ("gravity lead-in + bayonet twist-lock"), rev 3 = rev 2 without the detent bump at the
end of the twist slot (too tight in PLA; set real-profile detent=0.8 for rev 2). rev 2 (inverted bayonet):
  body (cup) carries plain collar + 4 outward lugs (teeth); the lid skirt carries the
  V lead-in, twist slot and locking ledge on its inner wall, so nothing ramp-like is
  visible on the cup when the lid is off. The bottom holder is dropped; the cup
  bottom is a plain cylinder.

    python generate_shaker.py                 # sim + real(print) 둘 다 생성 + 검증
    python generate_shaker.py --profile sim   # sim 메시/URDF만

Every part is a "polar solid": outer surface r_out(theta, z) and (optional)
inner cavity r_in(theta, z), both star-shaped about the z axis. Grooves, lugs,
keys, the knob, chamfers are all expressed as radius functions over the
(arc-length s, height) plane, using a smoothed signed-distance of 2-D polygons,
so the mesh is watertight by construction.

Frames (mm in this file, m in URDF):
  body   : origin = bottom centre, +z up, rim at z = body_h.
           lugs centred at theta = 0, 90, 180, 270 deg (+x, +y, -x, -y).
  lid    : origin = mating plane (underside of the top plate) centre, +z up.
           groove V-mouths centred at theta = 0, 90, ... in the lid frame.
           drop-in pose  : lid frame at (0,0,body_h + gasket) in body frame, yaw 0
           locked pose   : same, yaw = -twist_deg (clockwise seen from above),
                           z lowered by `helix` (gasket compression).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
BASE = dict(
    # body (grip zone sized for RH56F1 power grasp — see DESIGN_KR.md §1)
    grip_r=30.0,            # grip OD 60
    wall=2.5,
    body_h=170.0,
    floor=6.0,
    # bayonet collar on body top: plain cylinder + outward lugs (teeth)
    collar_r=32.0,          # collar OD 64 (lug tips at collar_r + groove_depth)
    groove_depth=2.2,       # lug radial projection == lid groove depth
    n_lugs=4,
    lead_angle_deg=40.0,    # V lead-in flank angle from horizontal (needs tan > mu)
    lug_w=6.0,
    lug_h=5.0,
    overhang=4.0,           # slot floor height above the V top (= ledge thickness when lock_ledge)
    lock_ledge=True,        # locking ledge under the twist slot (kept in rev3)
    twist_deg=30.0,
    collar_top_chamfer=1.5,
    collar_floor=3.0,       # plain collar below the lid skirt bottom (locked pose)
    lug_top_ch=1.0,         # lug entry-corner chamfer
    # lid: V lead-in + twist slot + locking ledge cut into the skirt inner wall
    lid_skirt_t=4.5,        # >= groove_depth + 2.3 mm wall behind the groove
    lid_plate=5.0,
    lid_entry_chamfer=3.0,
    lid_lip_min=1.5,        # min wall at the skirt bottom face (printable: >= 3 lines @0.4 nozzle)
    slot_top_margin=3.0,    # plain skirt between twist-slot ceiling and plate underside
    knob_h=22.0,
    knob_span=36.0,
    knob_arm=10.0,
    density_g_mm3=1.27e-3,  # PETG solid; real print mass must be weighed
)

PROFILES = {
    # FDM print: tight fits, face gasket + helix compression, detent bump
    "real": dict(rad_clr=0.4, side_clr=0.5, z_clr=0.6, gasket=1.5, helix=0.9,
                 detent=0.0, smooth=0.35,   # rev3: detent bump removed (too tight in PLA; rev2 = 0.8)
                 n_theta=1440, dz=0.25),
    # PhysX/SDF: generous clearances (contact offsets eat ~1 mm), rigid -> no
    # gasket / helix / detent; coarser mesh
    "sim": dict(rad_clr=1.0, side_clr=1.0, z_clr=1.0, gasket=0.0, helix=0.0,
                detent=0.0, smooth=0.9, n_theta=256, dz=0.6),
}


class P(dict):
    __getattr__ = dict.__getitem__


def params(profile: str) -> P:
    p = P(BASE)
    p.update(PROFILES[profile])
    p.profile = profile
    tanL = math.tan(math.radians(p.lead_angle_deg))
    # --- lid groove layout (s = arc mm at collar_r in the LID frame,
    #     w = mm above the skirt bottom edge, i.e. depth into the lid from its opening)
    p.S = 2 * math.pi * p.collar_r / p.n_lugs
    p.c = p.lug_w / 2 + p.side_clr
    p.Hv = (p.S / 2 - p.c) * tanL           # V lead-in height
    p.t_s0 = p.Hv + p.overhang              # twist-slot floor (= top of locking ledge)
    p.t_s1 = p.t_s0 + p.lug_h + p.z_clr     # twist-slot ceiling
    p.L_tw = p.collar_r * math.radians(p.twist_deg)
    p.skirt_len = p.t_s1 + p.helix + p.slot_top_margin
    p.lid_in_r = p.collar_r + p.rad_clr
    p.lid_out_r = p.lid_in_r + p.lid_skirt_t
    # entry chamfer and groove floor do not stack (inner radius = max of the two), so the
    # lip wall is skirt_t - chamfer; flare the outside only if that drops below lid_lip_min
    p.lid_lip_flare = max(0.0, p.lid_entry_chamfer + p.lid_lip_min - p.lid_skirt_t)
    # --- body lugs (t = mm below body rim). Drop-in: lid underside at rim + gasket,
    #     so w = skirt_len - gasket - t. Lug lower (load) face at w = t_s0 + 0.4 z_clr.
    p.lug_w_lo = p.t_s0 + 0.4 * p.z_clr
    p.lug_t1 = p.skirt_len - p.gasket - p.lug_w_lo      # lug lower face, below rim
    p.lug_t0 = p.lug_t1 - p.lug_h                       # lug upper (entry) face
    # plain collar must reach below the lid skirt in the locked pose
    p.collar_h = p.skirt_len - p.gasket + p.helix + p.collar_floor
    return p


# ---------------------------------------------------------------------------
# 2-D helpers
# ---------------------------------------------------------------------------
def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def poly_sd(poly, x, y):
    """signed distance, positive inside."""
    d = min(_seg_dist(x, y, *poly[i], *poly[(i + 1) % len(poly)]) for i in range(len(poly)))
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return d if inside else -d


def fill(poly, x, y, w):
    """smoothed indicator (0 outside .. 1 inside), boundary at 0.5."""
    sd = poly_sd(poly, x, y)
    if sd >= w / 2:
        return 1.0
    if sd <= -w / 2:
        return 0.0
    a = (sd + w / 2) / w
    return a * a * (3 - 2 * a)


def sector_s(theta, n, r, offset=0.0):
    """arc coordinate in the local sector (centred on offset + k*2pi/n)."""
    sec = 2 * math.pi / n
    loc = (theta - offset + sec / 2) % sec - sec / 2
    return loc * r


def lid_groove_poly(p):
    """(s, w) in the lid frame. Opens at the skirt bottom (w < 0), V lead-in up to
    Hv, vertical channel, then the twist slot running to +s (lid turns -twist, so the
    body lug moves +s relative to the lid) and climbing `helix` (gasket compression)."""
    tanL = math.tan(math.radians(p.lead_angle_deg))
    e = 5.0
    S, c, Hv, h = p.S, p.c, p.Hv, p.helix
    return [(S / 2 + e, -20.0), (-S / 2 - e, -20.0), (-S / 2 - e, -e * tanL),
            (-c, Hv), (-c, p.t_s1),
            (c + p.L_tw, p.t_s1 + h)] + (
        [(c + p.L_tw, p.t_s0 + h), (c, p.t_s0), (c, Hv)] if p.lock_ledge else
        # rev3: no ledge — the slot is open downward; its end wall drops straight onto the V flank
        [(c + p.L_tw, Hv - p.L_tw * tanL)]) + [(S / 2 + e, -e * tanL)]


def detent_poly(p):
    """bump on the lid groove floor just on the unlock side of the locked lug."""
    s1 = p.L_tw - p.lug_w / 2 - 0.3
    return [(s1 - 1.6, p.t_s0 - 1), (s1, p.t_s0 - 1), (s1, p.t_s1 + p.helix + 1), (s1 - 1.6, p.t_s1 + p.helix + 1)]


def lug_poly(p):
    """(s, t) on the body collar, t = mm below rim. Entry (upper) corners chamfered;
    lower face flat = load face against the lid locking ledge."""
    hw, ch = p.lug_w / 2, p.lug_top_ch
    return [(-hw, p.lug_t1), (hw, p.lug_t1), (hw, p.lug_t0 + ch), (hw - ch, p.lug_t0),
            (-hw + ch, p.lug_t0), (-hw, p.lug_t0 + ch)]


# ---------------------------------------------------------------------------
# radius functions
# ---------------------------------------------------------------------------
def body_outer(p):
    L = lug_poly(p)

    def r(th, z):
        t = p.body_h - z
        if t <= p.collar_h:                     # plain collar + outward lugs
            R = p.collar_r
            if t < 2.0:
                R -= p.collar_top_chamfer * (1 - t / 2.0)
            if p.lug_t0 - 1.0 < t < p.lug_t1 + 1.0:
                s = sector_s(th, p.n_lugs, p.collar_r)
                R += p.groove_depth * fill(L, s, t, p.smooth)
            return R
        if t <= p.collar_h + (p.collar_r - p.grip_r):   # 45 deg step
            return p.collar_r - (t - p.collar_h)
        R = p.grip_r                            # plain grip / bottom (no holder keys)
        if z < 0.8:
            R -= 0.8 - z
        return R
    return r


def body_inner(p):
    thin = p.grip_r - p.wall

    def r(th, z):
        return thin
    return r


def plus_r(th, span, arm):
    c, s = abs(math.cos(th)), abs(math.sin(th))
    a, b = span / 2, arm / 2
    r1 = min(a / c if c > 1e-9 else 1e9, b / s if s > 1e-9 else 1e9)
    r2 = min(b / c if c > 1e-9 else 1e9, a / s if s > 1e-9 else 1e9)
    return max(r1, r2)


def lid_outer(p):
    """in u-frame: u = 0 at knob top, increasing downward."""
    def r(th, u):
        if u <= p.knob_h:
            inset = max(0.0, 1.5 - u)
            return plus_r(th, p.knob_span, p.knob_arm) - inset
        if u <= p.knob_h + 1.0:
            return p.lid_out_r - (1.0 - (u - p.knob_h))
        umax = p.knob_h + p.lid_plate + p.skirt_len
        R = p.lid_out_r
        if u > umax - 4.0:
            R += p.lid_lip_flare * (u - (umax - 4.0)) / 4.0
        if u > umax - 0.3:
            R -= u - (umax - 0.3)
        return R
    return r


def lid_inner_w(p):
    """lid inner radius as f(theta_lid, w), w = mm above the skirt bottom edge."""
    G, D = lid_groove_poly(p), detent_poly(p)

    def r(th, w):
        cham = p.lid_entry_chamfer * (4.0 - w) / 4.0 if w < 4.0 else 0.0
        s = sector_s(th, p.n_lugs, p.collar_r)
        g = p.groove_depth * fill(G, s, w, p.smooth)
        if p.detent > 0:
            g -= p.detent * fill(D, s, w, 1.2)
        return p.lid_in_r + max(cham, g)
    return r


def lid_inner(p):
    """in u-frame (u = 0 at knob top, down)."""
    f = lid_inner_w(p)
    umax = p.knob_h + p.lid_plate + p.skirt_len

    def r(th, u):
        return f(th, umax - u)
    return r


# ---------------------------------------------------------------------------
# polar-solid mesher
# ---------------------------------------------------------------------------
def levels(*spans, extra=()):
    """spans: (z0, z1, dz). Returns sorted unique z list."""
    zs = set(round(z, 4) for z in extra)
    for z0, z1, dz in spans:
        n = max(1, int(math.ceil((z1 - z0) / dz)))
        for i in range(n + 1):
            zs.add(round(z0 + (z1 - z0) * i / n, 4))
    return sorted(zs)


def build_solid(n, outer, inner):
    """outer: list of (z, rfun) bottom->top ; inner: list of (z, rfun) top->bottom
    (empty -> solid with top cap). Profile path: bottom centre -> outer rings ->
    rim -> inner rings -> inner-floor centre. Returns (verts, tris)."""
    ths = [2 * math.pi * i / n for i in range(n)]
    V, T = [], []

    def add_ring(z, f):
        i0 = len(V)
        for th in ths:
            r = f(th, z)
            V.append((r * math.cos(th), r * math.sin(th), z))
        return i0

    def add_pt(z):
        V.append((0.0, 0.0, z))
        return ("pt", len(V) - 1)

    path = [add_pt(outer[0][0])]
    for z, f in outer:
        path.append(("ring", add_ring(z, f)))
    if inner:
        for z, f in inner:
            path.append(("ring", add_ring(z, f)))
        path.append(add_pt(inner[-1][0]))
    else:
        path.append(add_pt(outer[-1][0]))

    for (ka, a), (kb, b) in zip(path, path[1:]):
        for i in range(n):
            j = (i + 1) % n
            if ka == "pt":
                T.append((a, b + j, b + i))
            elif kb == "pt":
                T.append((a + i, a + j, b))
            else:
                T.append((a + i, a + j, b + j))
                T.append((a + i, b + j, b + i))
    vol = signed_volume(V, T)
    if vol < 0:
        T = [(a, c, b) for a, b, c in T]
    return V, T


def signed_volume(V, T):
    s = 0.0
    for a, b, c in T:
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = V[a], V[b], V[c]
        s += x1 * (y2 * z3 - y3 * z2) - x2 * (y1 * z3 - y3 * z1) + x3 * (y1 * z2 - y2 * z1)
    return s / 6.0


def mass_props(V, T, rho):
    """volume, COM, inertia about COM (mm, g) via tetrahedra from origin."""
    vol = 0.0
    cx = cy = cz = 0.0
    xx = yy = zz = xy = xz = yz = 0.0
    for a, b, c in T:
        p1, p2, p3 = V[a], V[b], V[c]
        d = (p1[0] * (p2[1] * p3[2] - p3[1] * p2[2]) - p2[0] * (p1[1] * p3[2] - p3[1] * p1[2])
             + p3[0] * (p1[1] * p2[2] - p2[1] * p1[2])) / 6.0
        vol += d
        cx += d * (p1[0] + p2[0] + p3[0]) / 4
        cy += d * (p1[1] + p2[1] + p3[1]) / 4
        cz += d * (p1[2] + p2[2] + p3[2]) / 4

        def q(i, j):
            return (p1[i] * p1[j] + p2[i] * p2[j] + p3[i] * p3[j]
                    + (p1[i] + p2[i] + p3[i]) * (p1[j] + p2[j] + p3[j])) / 20.0
        xx += d * q(0, 0); yy += d * q(1, 1); zz += d * q(2, 2)
        xy += d * q(0, 1); xz += d * q(0, 2); yz += d * q(1, 2)
    com = (cx / vol, cy / vol, cz / vol)
    m = rho * vol
    # second moments about origin -> inertia about COM
    Ixx = rho * (yy + zz) - m * (com[1] ** 2 + com[2] ** 2)
    Iyy = rho * (xx + zz) - m * (com[0] ** 2 + com[2] ** 2)
    Izz = rho * (xx + yy) - m * (com[0] ** 2 + com[1] ** 2)
    return dict(volume_mm3=vol, mass_g=m, com_mm=com, I_gmm2=(Ixx, Iyy, Izz))


def write_stl(path, V, T, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(T)))
        for a, b, c in T:
            p1, p2, p3 = V[a], V[b], V[c]
            ux, uy, uz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
            vx, vy, vz = p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            L = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            f.write(struct.pack("<12fH", nx / L, ny / L, nz / L, *p1, *p2, *p3, 0))


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------
def make_body(p):
    fo, fi = body_outer(p), body_inner(p)
    z_col = p.body_h - p.collar_h
    zo = levels((0.0, 0.8, 0.4), (z_col - 2.5, p.body_h, p.dz), extra=(z_col / 2,))
    zi = [p.floor, p.body_h]
    return build_solid(p.n_theta, [(z, fo) for z in zo], [(z, fi) for z in reversed(zi)])


def make_lid(p):
    fo, fi = lid_outer(p), lid_inner(p)
    umax = p.knob_h + p.lid_plate + p.skirt_len
    u_plate = p.knob_h + p.lid_plate
    uo = levels((0.0, 1.5, 0.5), (p.knob_h, p.knob_h + 1.0, 0.5), (umax - 0.8, umax, 0.4),
                extra=(p.knob_h - 1e-3,))
    # outer path must go "bottom -> top" in the mesher sense: mesher bottom = knob top
    outer = []
    for u in uo:
        if abs(u - (p.knob_h - 1e-3)) < 1e-6:
            continue
        if abs(u - p.knob_h) < 1e-6:   # step: knob section, then plate section at same u
            outer.append((u, lambda th, uu, f=fo: f(th, p.knob_h - 1e-6)))
            outer.append((u, lambda th, uu, f=fo: f(th, p.knob_h + 1e-6)))
            continue
        outer.append((u, fo))
    ui = levels((u_plate, umax, p.dz), extra=(umax - 4.0,))
    inner = [(u, fi) for u in reversed(ui)]
    V, T = build_solid(p.n_theta, outer, inner)
    # u-frame (down) -> lid frame: z = u_plate - u ; mirror flips handedness
    V = [(x, y, u_plate - u) for x, y, u in V]
    T = [(a, c, b) for a, b, c in T]
    if signed_volume(V, T) < 0:
        T = [(a, c, b) for a, b, c in T]
    return V, T


# ---------------------------------------------------------------------------
# kinematic clearance checks (radius-function level, exact w.r.t. design)
# ---------------------------------------------------------------------------
def check_bayonet(p):
    """Sweep the body lugs through drop -> twist relative to the lid and report the min
    radial clearance between lug tip and lid inner surface (negative = interference)."""
    fl = lid_inner_w(p)
    tip_r = p.collar_r + p.groove_depth
    hw = p.lug_w / 2

    def min_clear(yaw, dz_lid):
        """yaw: lid yaw [rad]; dz_lid: lid lowered from the drop-in seat [mm]."""
        worst = 1e9
        for k in range(p.n_lugs):
            for i in range(13):
                s = -hw + 2 * hw * i / 12
                th_l = k * 2 * math.pi / p.n_lugs + s / p.collar_r - yaw
                for j in range(9):
                    t = p.lug_t0 + p.lug_top_ch + 0.2 + (p.lug_h - p.lug_top_ch - 0.4) * j / 8
                    w = p.skirt_len - (t + p.gasket - dz_lid)
                    if w < 0:
                        continue  # below the skirt: no lid material
                    worst = min(worst, fl(th_l, w) - tip_r)
        return worst

    out = {"seated_unlocked": min_clear(0.0, 0.0)}
    out["twist_path_min"] = min(min_clear(-math.radians(p.twist_deg) * k / 30, p.helix * k / 30)
                                for k in range(31))
    out["locked"] = min_clear(-math.radians(p.twist_deg), p.helix)
    out["drop_path_min"] = min(min_clear(0.0, -(p.skirt_len + 5) * (1 - k / 40)) for k in range(41))
    out["note"] = ("radial clearance [mm]; detent interference expected = detent - rad_clr"
                   if p.detent > 0 else "radial clearance [mm]")
    return out


def check_lock_retention(p):
    """Locked pose: lug lower (load) face vs top of the lid locking ledge, in w."""
    w_lug_lo = p.lug_w_lo + p.helix
    w_ledge = p.t_s0 + p.helix
    if not p.lock_ledge:
        return dict(ledge="none (rev3) - lid is not retained axially; held down by gravity only",
                    lug_top_below_slot_ceiling_mm=(p.t_s1 + p.helix) - (w_lug_lo + p.lug_h))
    return dict(lug_face_above_ledge_mm=w_lug_lo - w_ledge,
                ledge_thickness_mm=p.overhang,
                lug_top_below_slot_ceiling_mm=(p.t_s1 + p.helix) - (w_lug_lo + p.lug_h))


# ---------------------------------------------------------------------------
# URDF
# ---------------------------------------------------------------------------
def urdf_link(name, mesh, mp, frame_note):
    m = mp["mass_g"] / 1000.0
    c = [v / 1000.0 for v in mp["com_mm"]]
    I = [v * 1e-9 for v in mp["I_gmm2"]]
    return f"""  <link name="{name}">
    <!-- {frame_note} -->
    <inertial>
      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>
      <mass value="{m:.5f}"/>
      <inertia ixx="{I[0]:.4e}" ixy="0" ixz="0" iyy="{I[1]:.4e}" iyz="0" izz="{I[2]:.4e}"/>
    </inertial>
    <visual><geometry><mesh filename="{mesh}" scale="0.001 0.001 0.001"/></geometry></visual>
    <collision><geometry><mesh filename="{mesh}" scale="0.001 0.001 0.001"/></geometry></collision>
  </link>
"""


def write_urdfs(p, props, outdir, meshrel):
    os.makedirs(outdir, exist_ok=True)
    notes = dict(body="origin = bottom centre, rim at z=%.1f mm" % p.body_h,
                 lid="origin = mating plane (plate underside) centre")
    for part in ("body", "lid"):
        with open(os.path.join(outdir, f"rh56_shaker_{part}.urdf"), "w") as f:
            f.write(f'<?xml version="1.0"?>\n<!-- generated by generate_shaker.py (profile={p.profile}) -->\n')
            f.write(f'<robot name="rh56_shaker_{part}">\n')
            f.write(urdf_link(part, f"{meshrel}/rh56_shaker_{part}.stl", props[part], notes[part]))
            f.write("</robot>\n")
    lz = (p.body_h + p.gasket - p.helix) / 1000.0
    with open(os.path.join(outdir, "rh56_shaker_locked.urdf"), "w") as f:
        f.write(f'<?xml version="1.0"?>\n<!-- body + lid welded in LOCKED pose (shaking-only training). profile={p.profile} -->\n')
        f.write('<robot name="rh56_shaker_locked">\n')
        f.write(urdf_link("body", f"{meshrel}/rh56_shaker_body.stl", props["body"], notes["body"]))
        f.write(urdf_link("lid", f"{meshrel}/rh56_shaker_lid.stl", props["lid"], notes["lid"]))
        f.write(f"""  <joint name="lid_lock" type="fixed">
    <parent link="body"/><child link="lid"/>
    <origin xyz="0 0 {lz:.6f}" rpy="0 0 {-math.radians(p.twist_deg):.6f}"/>
  </joint>
</robot>
""")


# ---------------------------------------------------------------------------
def run(profile):
    p = params(profile)
    print(f"== profile {profile}")
    parts = dict(body=make_body(p), lid=make_lid(p))
    mesh_dir = os.path.join(HERE, "meshes", profile)
    props = {}
    for name, (V, T) in parts.items():
        write_stl(os.path.join(mesh_dir, f"rh56_shaker_{name}.stl"), V, T, f"rh56_shaker_{name}")
        props[name] = mass_props(V, T, p.density_g_mm3)
        mp = props[name]
        print(f"  {name:6s}: {len(T):7d} tris  {mp['volume_mm3']/1000:7.1f} cm3  {mp['mass_g']:6.1f} g"
              f"  COM z {mp['com_mm'][2]:6.1f} mm")
    write_urdfs(p, props, os.path.join(HERE, "urdf", profile), f"../../meshes/{profile}")

    cav = math.pi * (p.grip_r - p.wall) ** 2 * (p.body_h - p.floor)
    tanL = math.tan(math.radians(p.lead_angle_deg))
    report = dict(
        profile=profile,
        params={k: v for k, v in p.items() if isinstance(v, (int, float, str))},
        derived=dict(
            capacity_ml=cav / 1000.0,
            lid_yaw_capture_deg=360.0 / p.n_lugs / 2,
            lid_radial_capture_mm=p.lid_entry_chamfer + p.collar_top_chamfer,
            lead_in_slide_requires_mu_below=tanL,
            v_lead_in_height_mm=p.Hv,
            collar_height_mm=p.collar_h,
            lid_od_mm=2 * p.lid_out_r,
            lid_lip_od_mm=2 * (p.lid_out_r + p.lid_lip_flare),
            lid_lip_wall_mm=p.lid_skirt_t + p.lid_lip_flare - p.lid_entry_chamfer - 0.3,
            lid_wall_behind_groove_mm=p.lid_skirt_t - p.groove_depth,
            lug_tip_od_mm=2 * (p.collar_r + p.groove_depth),
            lug_below_rim_mm=[p.lug_t0, p.lug_t1],
            lid_total_height_mm=p.knob_h + p.lid_plate + p.skirt_len,
        ),
        poses_in_body_frame_m=dict(
            lid_dropin=dict(xyz=[0, 0, (p.body_h + p.gasket) / 1000], yaw_rad=0.0),
            lid_locked=dict(xyz=[0, 0, (p.body_h + p.gasket - p.helix) / 1000],
                            yaw_rad=-math.radians(p.twist_deg)),
            lid_yaw_symmetry_rad=2 * math.pi / p.n_lugs,
        ),
        mass_props={k: dict(mass_kg=v["mass_g"] / 1000, com_m=[c / 1000 for c in v["com_mm"]],
                            inertia_kgm2=[i * 1e-9 for i in v["I_gmm2"]]) for k, v in props.items()},
        checks=dict(bayonet=check_bayonet(p), lock=check_lock_retention(p)),
    )
    with open(os.path.join(HERE, f"shaker_{profile}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("  capacity %.0f mL | Hv %.1f mm | collar %.1f mm | lid OD %.1f mm"
          % (report["derived"]["capacity_ml"], p.Hv, p.collar_h, 2 * p.lid_out_r))
    print("  checks:", json.dumps(report["checks"]))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["sim", "real", "all"], default="all")
    a = ap.parse_args()
    for prof in (["sim", "real"] if a.profile == "all" else [a.profile]):
        run(prof)
