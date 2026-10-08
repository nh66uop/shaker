#!/usr/bin/env python3
"""B-rep / STEP export of the RH56 bayonet shaker (for Fusion 360 / any CAD).

    .venv-cad/Scripts/python assets/cocktail_rh56/generate_step.py            # real profile
    .venv-cad/Scripts/python assets/cocktail_rh56/generate_step.py --profile sim

Same parameters and same groove/lug polygons as generate_shaker.py (the single
source of truth); only the geometry kernel differs. Lid grooves, body lugs and
the detent are "polar prisms": a polygon in the unrolled (arc s, height) plane
mapped exactly onto coaxial cylinders (edges become true helices, groove floors true
cylinders), closed with ruled side faces — the OCC bottle-thread construction.

Outputs (mm) -> step/<profile>/:
  rh56_shaker_body.step, rh56_shaker_lid.step
  rh56_shaker_assembly_locked.step   (body + lid locked, coloured)
  rh56_shaker_assembly_dropin.step   (lid seated, before twist)
"""
import argparse
import json
import math
import os
import sys

import cadquery as cq
from OCP.BRep import BRep_Tool  # noqa: F401
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeWire,
                                BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFill import BRepFill
from OCP.BRepLib import BRepLib
from OCP.GCE2d import GCE2d_MakeSegment
from OCP.Geom import Geom_CylindricalSurface
from OCP.gp import gp, gp_Ax3, gp_Pnt2d
from OCP.ShapeFix import ShapeFix_Solid
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_shaker as g  # noqa: E402


# ---------------------------------------------------------------------------
def polar_prism(poly, r_in, r_out, r_ref, zfun):
    """Solid {(r, th, z): r_in<=r<=r_out, (s, t) in poly} with th = pi + s/r_ref,
    z = zfun(t). Built centred at th = pi (avoids the u-seam); rotate copies."""
    ax = gp_Ax3(gp.Origin_s(), gp.DZ_s())
    pts = [gp_Pnt2d(math.pi + s / r_ref, zfun(t)) for s, t in poly]

    def ring(r):
        surf = Geom_CylindricalSurface(ax, r)
        edges = []
        mk = BRepBuilderAPI_MakeWire()
        for i in range(len(pts)):
            seg = GCE2d_MakeSegment(pts[i], pts[(i + 1) % len(pts)]).Value()
            e = BRepBuilderAPI_MakeEdge(seg, surf).Edge()
            BRepLib.BuildCurves3d_s(e)
            edges.append(e)
            mk.Add(e)
        w = mk.Wire()
        f = BRepBuilderAPI_MakeFace(surf, w, True).Face()
        return f, edges

    f_in, e_in = ring(r_in)
    f_out, e_out = ring(r_out)
    sew = BRepBuilderAPI_Sewing(1e-4)
    sew.Add(f_in)
    sew.Add(f_out)
    for a, b in zip(e_in, e_out):
        sew.Add(BRepFill.Face_s(a, b))
    sew.Perform()
    exp = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    shell = TopoDS.Shell_s(exp.Current())
    solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    fix = ShapeFix_Solid(solid)
    fix.Perform()
    s = cq.Solid(fix.Solid())
    if s.Volume() < 0:
        s = cq.Solid(s.wrapped.Reversed())
    assert BRepCheck_Analyzer(s.wrapped).IsValid(), "invalid polar prism"
    return s


def copies(solid, n, offset_rad):
    """n copies at offset + k*2pi/n (solid is built centred at pi)."""
    out = []
    for k in range(n):
        ang = math.degrees(offset_rad + 2 * math.pi * k / n - math.pi)
        out.append(solid.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), ang))
    return out


def revolve(pts):
    return cq.Workplane("XZ").polyline(pts).close().revolve(360, (0, 0, 0), (0, 1, 0)).val()


def boolean(base, add=(), sub=()):
    s = base
    for a in add:
        s = s.fuse(a)
    for b in sub:
        s = s.cut(b)
    s = s.clean()
    return s


# ---------------------------------------------------------------------------
def make_body(p):
    """plain cup + 4 outward lugs on the collar (no groove, no holder keys)."""
    z_col = p.body_h - p.collar_h
    thin = p.grip_r - p.wall
    prof = [(0, 0), (p.grip_r - 0.8, 0), (p.grip_r, 0.8),
            (p.grip_r, z_col - (p.collar_r - p.grip_r)), (p.collar_r, z_col),
            (p.collar_r, p.body_h - 2.0), (p.collar_r - p.collar_top_chamfer, p.body_h),
            (thin, p.body_h), (thin, p.floor), (0, p.floor)]
    body = revolve(prof)
    zt = lambda t: p.body_h - t  # noqa: E731
    lug = polar_prism(g.lug_poly(p), p.collar_r - 0.5, p.collar_r + p.groove_depth, p.collar_r, zt)
    return boolean(body, add=copies(lug, p.n_lugs, 0.0))


def make_lid(p):
    """knob + plate + skirt; V lead-in / twist slot / locking ledge cut into the skirt."""
    zp = p.lid_plate
    fl = p.lid_lip_flare
    prof = [(0, zp), (p.lid_out_r - 1.0, zp), (p.lid_out_r, zp - 1.0),
            (p.lid_out_r, -p.skirt_len + 4.0), (p.lid_out_r + fl, -p.skirt_len + 0.3),
            (p.lid_out_r + fl - 0.3, -p.skirt_len),
            (p.lid_in_r + p.lid_entry_chamfer, -p.skirt_len), (p.lid_in_r, -p.skirt_len + 4.0),
            (p.lid_in_r, 0), (0, 0)]
    lid = revolve(prof)
    a, b = p.knob_span / 2, p.knob_arm / 2
    plus = [(a, b), (b, b), (b, a), (-b, a), (-b, b), (-a, b), (-a, -b), (-b, -b), (-b, -a), (b, -a), (b, -b), (a, -b)]
    knob = (cq.Workplane("XY").workplane(offset=zp - 0.5).polyline(plus).close().extrude(p.knob_h + 0.5)
            .edges("|Z").fillet(1.2).faces(">Z").edges().chamfer(1.0)).val()
    zw = lambda w: w - p.skirt_len  # noqa: E731
    r_floor = p.lid_in_r + p.groove_depth
    groove = polar_prism(g.lid_groove_poly(p), p.lid_in_r - 1.0, r_floor, p.collar_r, zw)
    lid = boolean(lid, add=[knob], sub=copies(groove, p.n_lugs, 0.0))
    if p.detent > 0:
        bump = polar_prism(g.detent_poly(p), r_floor - p.detent, r_floor + 0.2, p.collar_r, zw)
        lid = boolean(lid, add=copies(bump, p.n_lugs, 0.0))
    return lid


# ---------------------------------------------------------------------------
def run(profile):
    p = g.params(profile)
    out = os.path.join(HERE, "step", profile)
    os.makedirs(out, exist_ok=True)
    parts = dict(body=make_body(p), lid=make_lid(p))

    ref = {}
    jp = os.path.join(HERE, f"shaker_{profile}.json")
    if os.path.exists(jp):
        with open(jp) as f:
            ref = {k: v["mass_kg"] * 1000 / p.density_g_mm3 for k, v in json.load(f)["mass_props"].items()}

    print(f"== {profile}")
    for k, s in parts.items():
        ok = BRepCheck_Analyzer(s.wrapped).IsValid()
        v = s.Volume()
        cmp = f"  mesh {ref[k]:9.0f}  diff {100 * (v / ref[k] - 1):+5.2f}%" if k in ref else ""
        print(f"  {k:6s} valid={ok}  solids={len(s.Solids())}  faces={len(s.Faces()):4d}  vol {v:9.0f} mm3{cmp}")
        cq.exporters.export(s, os.path.join(out, f"rh56_shaker_{k}.step"))

    steel, lugc = cq.Color(0.72, 0.75, 0.78), cq.Color(0.25, 0.45, 0.75)
    for name, yaw, dz in (("locked", -p.twist_deg, p.gasket - p.helix), ("dropin", 0.0, p.gasket)):
        asm = cq.Assembly(name=f"rh56_shaker_{name}")
        asm.add(parts["body"], name="body", color=steel)
        asm.add(parts["lid"], name="lid",
                loc=cq.Location((0, 0, p.body_h + dz), (0, 0, 1), yaw), color=lugc)
        asm.export(os.path.join(out, f"rh56_shaker_assembly_{name}.step"))
    print("  ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["sim", "real", "all"], default="real")
    a = ap.parse_args()
    for prof in (["sim", "real"] if a.profile == "all" else [a.profile]):
        run(prof)
