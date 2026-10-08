#!/usr/bin/env python3
"""Build a single-file interactive viewer (three.js) of the RH56 bayonet shaker.

    python viz/build_viewer.py      -> viz/rh56_shaker_viewer.html

Meshes are regenerated from generate_shaker.py at viewer resolution (sim profile
geometry) and tagged per vertex: 1 = bayonet groove (V lead-in + twist slot) /
lid lugs, 2 = holder keys. Everything is embedded; no external files.
"""
import base64
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import generate_shaker as g  # noqa: E402


def b64(fmt, seq):
    return base64.b64encode(struct.pack("<%d%s" % (len(seq), fmt), *seq)).decode()


def pack(V, T, flags):
    return dict(v=b64("f", [c for v in V for c in v]), i=b64("I", [k for t in T for k in t]),
                f=b64("B", flags), n=len(T))


def main():
    p = g.params("sim")
    p.n_theta = 192
    p.dz = 0.6
    bV, bT = g.make_body(p)
    lV, lT = g.make_lid(p)
    hV, hT = g.make_holder(p)

    def bflag(v):
        x, y, z = v
        r = math.hypot(x, y)
        t = p.body_h - z
        if 2.0 < t < p.collar_h and r < p.collar_r - 0.4 and r > p.collar_r - p.groove_depth - 1:
            return 1
        if 0.8 < z < p.Hr + 1 and p.grip_r - p.key_depth - 0.6 < r < p.grip_r - 0.4:
            return 2
        return 0

    def lflag(v):
        x, y, z = v
        r = math.hypot(x, y)
        return 1 if (-p.skirt_len + 4 < z < -0.01 and r < p.lid_in_r - 0.4) else 0

    def hflag(v):
        x, y, z = v
        r = math.hypot(x, y)
        return 2 if (z > p.holder_floor - 0.5 and z < p.holder_floor + p.key_h + 0.5
                     and r < p.pocket_r - 0.4) else 0

    meshes = dict(body=pack(bV, bT, [bflag(v) for v in bV]),
                  lid=pack(lV, lT, [lflag(v) for v in lV]),
                  holder=pack(hV, hT, [hflag(v) for v in hV]))

    P = {k: getattr(p, k) for k in ("collar_r", "groove_depth", "n_lugs", "lead_angle_deg", "lug_w", "lug_h",
                           "overhang", "twist_deg", "side_clr", "z_clr", "gasket", "helix", "body_h",
                           "collar_h", "S", "c", "Hv", "t_s0", "t_s1", "L_tw", "lug_t0", "lug_t1",
                           "holder_floor", "grip_r", "lid_out_r", "n_keys", "lid_entry_chamfer",
                           "collar_top_chamfer", "holder_funnel")}
    P["groove"] = g.body_groove_poly(p)
    reports = {}
    for prof in ("sim", "real"):
        fp = os.path.join(os.path.dirname(HERE), f"shaker_{prof}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                r = json.load(f)
            reports[prof] = dict(checks=r["checks"], derived=r["derived"],
                                 mass={k: v["mass_kg"] for k, v in r["mass_props"].items()},
                                 clr=dict(rad=r["params"]["rad_clr"], side=r["params"]["side_clr"],
                                          z=r["params"]["z_clr"], gasket=r["params"]["gasket"],
                                          helix=r["params"]["helix"], detent=r["params"]["detent"]))

    with open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__MESHES__", json.dumps(meshes)).replace("__PARAMS__", json.dumps(P)) \
               .replace("__REPORTS__", json.dumps(reports))
    out = os.path.join(HERE, "rh56_shaker_viewer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(out, "%.2f MB" % (os.path.getsize(out) / 1e6),
          {k: v["n"] for k, v in meshes.items()})


if __name__ == "__main__":
    main()
