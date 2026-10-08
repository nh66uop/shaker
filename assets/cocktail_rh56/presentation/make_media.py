#!/usr/bin/env python3
"""Presentation media for the RH56 bayonet shaker (real-profile STEP geometry).

    .venv-cad/Scripts/python assets/cocktail_rh56/presentation/make_media.py

Writes PNG / MP4 / GIF into this folder. Requires: cadquery, pyvista, matplotlib, imageio[-ffmpeg].
"""
import math
import os
import sys

import cadquery as cq
import imageio.v2 as imageio
import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MPoly, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import generate_shaker as g  # noqa: E402

pv.OFF_SCREEN = True
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROFILE = "real"
p = g.params(PROFILE)
STEP = os.path.join(ROOT, "step", PROFILE)
OUT = HERE
W, H = 1600, 1200

C = dict(steel=(196, 202, 208), groove=(226, 116, 34), lid=(160, 178, 200), lug=(42, 104, 188),
         holder=(92, 96, 102), key=(52, 150, 118), bg="white")


def load(name):
    return cq.importers.importStep(os.path.join(STEP, f"rh56_shaker_{name}.step")).val()


BODY, LID, HOLDER = load("body"), load("lid"), load("holder")


# ---------------------------------------------------------------------------
def to_pv(shape, tol=0.04):
    vs, ts = shape.tessellate(tol, 0.15)
    pts = np.array([[v.x, v.y, v.z] for v in vs])
    faces = np.hstack([[3, *t] for t in ts]).astype(np.int64)
    return pv.PolyData(pts, faces)


def classify(mesh, part):
    """per-cell RGB in the part's local frame."""
    cc = mesh.cell_centers().points
    r = np.hypot(cc[:, 0], cc[:, 1])
    z = cc[:, 2]
    col = np.zeros((mesh.n_cells, 3), np.uint8)
    if part == "body":
        col[:] = C["steel"]
        t = p.body_h - z
        groove = (t > 2.0) & (t < p.collar_h) & (r < p.collar_r - 0.3) & (r > p.collar_r - p.groove_depth - 0.6)
        key = (z > 0.8) & (z < p.Hr + 1) & (r < p.grip_r - 0.3) & (r > p.grip_r - p.key_depth - 0.6)
        col[groove] = C["groove"]
        col[key] = C["key"]
    elif part == "lid":
        col[:] = C["lid"]
        lug = (z < -0.01) & (z > -p.skirt_len + 4) & (r < p.lid_in_r - 0.3) & (r > p.lid_in_r - p.groove_depth - 0.6)
        col[lug] = C["lug"]
    else:
        col[:] = C["holder"]
        rib = (z > p.holder_floor + 0.3) & (z < p.holder_floor + p.key_h + 0.5) & (r < p.pocket_r - 0.3) & (r > p.grip_r - p.key_depth - 1.0)
        col[rib] = C["key"]
    return col


def section_shade(mesh, col, yaw_deg, on=True):
    """darken cells lying on the cut planes (theta = yaw and yaw+90)."""
    if not on:
        return col
    cc = mesh.cell_centers().points
    out = col.copy()
    for a in (yaw_deg, yaw_deg + 90):
        n = np.array([-math.sin(math.radians(a)), math.cos(math.radians(a)), 0.0])
        d = np.abs(cc @ n) < 0.05
        out[d] = (out[d] * 0.62).astype(np.uint8)
    return out


def cut_box(yaw_deg, zlo=-50, zhi=400):
    """solid occupying theta in (yaw, yaw+90) — the removed quadrant."""
    b = cq.Workplane("XY").box(300, 300, zhi - zlo, centered=(False, False, False)).translate((0, 0, zlo)).val()
    return b.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), yaw_deg)


_CUT_CACHE = {}


def quadrant_cut(shape, ang):
    """remove theta in (ang, ang+90). OCCT fails silently on some full-height boxes against the lid,
    so cut in z-bands and nudge a band's bounds until its cut actually removes material."""
    key = (id(shape), round(ang, 3))
    if key in _CUT_CACHE:
        return _CUT_CACHE[key]
    bb = shape.BoundingBox()
    edges = list(np.linspace(bb.zmin - 1, bb.zmax + 1, 9))
    s = shape
    for lo, hi in zip(edges[:-1], edges[1:]):
        for k in range(8):
            d = 0.137 * k
            box = cq.Solid.makeBox(150, 150, (hi + d) - (lo - d), cq.Vector(0, 0, lo - d)).rotate(
                cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), ang + 0.01 * k)
            if s.intersect(box).Volume() < 1e-3:
                break  # nothing to remove in this band
            v0 = s.Volume()
            s2 = s.cut(box)
            if s2.Volume() < v0 - 1e-3:
                s = s2
                break
    _CUT_CACHE[key] = s
    return s


def placed(shape, part, z, yaw, cut_yaw=None):
    """shape in local frame -> PolyData in world (rotate yaw, translate z); cut in world quadrant."""
    s = shape
    if cut_yaw is not None:
        s = quadrant_cut(s, cut_yaw - yaw)
    m = to_pv(s)
    col = classify(m, part)
    m = m.rotate_z(yaw, inplace=False).translate((0, 0, z), inplace=False)
    col = section_shade(m, col, cut_yaw, cut_yaw is not None)
    m.cell_data["rgb"] = col
    return m


def plotter(size=(W, H)):
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(C["bg"])
    pl.enable_anti_aliasing("ssaa")
    return pl


def add(pl, m, opacity=1.0):
    pl.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=True, split_sharp_edges=True,
                specular=0.35, specular_power=20, opacity=opacity, show_scalar_bar=False)


def cam(pl, az_deg, el_deg, dist, focus):
    a, e = math.radians(az_deg), math.radians(el_deg)
    pos = (focus[0] + dist * math.cos(e) * math.cos(a), focus[1] + dist * math.cos(e) * math.sin(a),
           focus[2] + dist * math.sin(e))
    pl.camera_position = [pos, focus, (0, 0, 1)]
    pl.camera.view_angle = 28


# ---------------------------------------------------------------------------
# kinematics (same as the web viewer)
tanL = math.tan(math.radians(p.lead_angle_deg))
hw = p.lug_w / 2
D0 = p.lug_t1 + 14
ZB = p.holder_floor  # body origin in world


def lim(t):
    return math.inf if t <= 0 else max(p.side_clr, p.S / 2 - t / tanL - hw)


def ease(u):
    return 2 * u * u if u < .5 else 1 - (-2 * u + 2) ** 2 / 2


def state(u, yaw0):
    s0 = p.collar_r * math.radians(yaw0)
    if u <= 1:
        drop = D0 * (1 - ease(u))
        tb = p.lug_t1 - p.gasket - drop
        m = lim(tb)
        s = max(-m, min(m, s0))
        twist = 0.0
    else:
        drop = 0.0
        sd = max(-p.side_clr, min(p.side_clr, s0))
        twist = ease(min(1.0, u - 1))
        s = sd + twist * (-p.L_tw - sd)
    dz = p.helix * twist
    return dict(yaw=math.degrees(s / p.collar_r), lid_z=ZB + p.body_h + p.gasket + drop - dz,
                s=s, t_top=p.lug_t0 - p.gasket - drop + dz)


# ---------------------------------------------------------------------------
def still(name, meshes, az, el, dist, focus, size=(W, H)):
    pl = plotter(size)
    for m, op in meshes:
        add(pl, m, op)
    cam(pl, az, el, dist, focus)
    img = pl.screenshot(os.path.join(OUT, name) if name else None, return_img=True)
    pl.close()
    return img


def renders():
    lock = state(2, 0)
    zc = ZB + p.body_h
    body_w = placed(BODY, "body", ZB, 0)
    hold_w = placed(HOLDER, "holder", 0, 0)
    lid_lock = placed(LID, "lid", lock["lid_z"], lock["yaw"])
    still("01_assembly_locked.png", [(hold_w, 1), (body_w, 1), (lid_lock, 1)], 35, 22, 560, (0, 0, 95))

    still("02_exploded.png", [(hold_w.translate((0, 0, -45), inplace=False), 1), (body_w, 1),
                              (placed(LID, "lid", ZB + p.body_h + 70, 0), 1)], 35, 18, 680, (0, 0, 110))

    body_only = placed(BODY, "body", 0, 0)
    still("04_body_collar.png", [(body_only, 1)], 20, 28, 210, (0, 0, p.body_h - 15))

    lid_up = to_pv(LID)
    lid_up.cell_data["rgb"] = classify(lid_up, "lid")
    lid_up = lid_up.rotate_x(180, inplace=False)
    still("05_lid_underside.png", [(lid_up, 1)], 30, 38, 230, (0, 0, -5))

    still("06_holder_keys.png", [(placed(HOLDER, "holder", 0, 0), 1),
                                 (placed(BODY, "body", ZB + 55, 0), 1)], 30, 24, 330, (0, 0, 45))

    # cutaway, locked: section follows the lug (theta = yaw)
    pl_c = [(placed(BODY, "body", ZB, 0, cut_yaw=lock["yaw"]), 1),
            (placed(LID, "lid", lock["lid_z"], lock["yaw"], cut_yaw=lock["yaw"]), 1)]
    still("03_cutaway_locked.png", pl_c, lock["yaw"] + 45, 16, 175, (0, 0, zc - 12))
    ghost_frame(2, 0, (W, H), "03b_ghost_locked.png")


def ghost_frame(u, yaw0, size, name=None):
    """body opaque, lid translucent with its lugs opaque; camera faces the lug from outside."""
    st = state(u, yaw0)
    body = placed(BODY, "body", ZB, 0)
    lid = placed(LID, "lid", st["lid_z"], st["yaw"])
    is_lug = np.all(lid.cell_data["rgb"] == np.array(C["lug"], np.uint8), axis=1)
    lugs = lid.extract_cells(np.where(is_lug)[0]).extract_surface()
    shell = lid.extract_cells(np.where(~is_lug)[0]).extract_surface()
    pl = plotter(size)
    pl.enable_depth_peeling(8)
    add(pl, body)
    add(pl, lugs)
    add(pl, shell, 0.22)
    cam(pl, 0, 10, 150, (0, 0, ZB + p.body_h - 14))
    img = pl.screenshot(os.path.join(OUT, name) if name else None, return_img=True)
    pl.close()
    return img, st


def cut_frame(u, yaw0, size, holder=False):
    st = state(u, yaw0)
    ms = [(placed(BODY, "body", ZB, 0, cut_yaw=st["yaw"]), 1),
          (placed(LID, "lid", st["lid_z"], st["yaw"], cut_yaw=st["yaw"]), 1)]
    return still(None, ms, st["yaw"] + 45, 16, 175, (0, 0, ZB + p.body_h - 12), size), st


def sequence(yaw0=28):
    stages = [(0.0, "① 하강 (각도 오차 +28°)"), (0.55, "② V 경사면이 러그를 유도"), (1.0, "③ 안착 (자동 정렬 완료)"),
              (1.5, "④ 30° 비틀어 잠금"), (2.0, "⑤ 잠김 — 러그가 잠금 턱 아래")]
    fig, axs = plt.subplots(1, 5, figsize=(20, 5.2), dpi=110)
    for ax, (u, lab) in zip(axs, stages):
        img, st = ghost_frame(u, yaw0, (900, 900))
        ax.imshow(img)
        ax.set_title(lab, fontsize=13)
        ax.text(0.5, -0.04, f"잠금 위치 대비 높이 {st['lid_z'] - state(2, yaw0)['lid_z']:+.1f} mm · 회전 {st['yaw']:+.1f}°",
                transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color="#444")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "07_mating_sequence.png"), facecolor="white")
    plt.close(fig)


def unrolled(yaw0=28):
    fig, ax = plt.subplots(figsize=(10, 8.4), dpi=150)
    sMin, sMax = -p.S / 2, p.S / 2
    ax.add_patch(Rectangle((sMin, 0), p.S, p.collar_h, color="#c4cad0", alpha=0.6, lw=0))
    poly = np.array(g.body_groove_poly(p))
    gp = MPoly(poly, closed=True, facecolor="#e27422", alpha=0.55, edgecolor="#b8561a", lw=1.6)
    ax.add_patch(gp)
    gp.set_clip_path(Rectangle((sMin, 0), p.S, 100, transform=ax.transData))
    ax.axhline(0, color="k", lw=1.2)
    if p.detent > 0:
        d = np.array(g.detent_poly(p))
        ax.add_patch(MPoly(d, closed=True, facecolor="#7a3b10", alpha=0.8, lw=0))
        ax.annotate(f"딸깍 돌기 {p.detent} mm", xy=(d[:, 0].mean(), p.t_s0 + 2), xytext=(6, p.collar_h - 1.5),
                    fontsize=10.5, arrowprops=dict(arrowstyle="-", color="#555"))
    # lug path + stage outlines
    us = np.linspace(0, 2, 300)
    path = np.array([[state(u, yaw0)["s"], state(u, yaw0)["t_top"] + p.lug_h / 2] for u in us])
    ax.plot(path[:, 0], path[:, 1], "--", color="#2a68bc", lw=1.4)
    for u, a in ((0.0, .25), (0.55, .4), (1.0, .6), (2.0, .95)):
        st = state(u, yaw0)
        ax.add_patch(Rectangle((st["s"] - hw, st["t_top"]), p.lug_w, p.lug_h, facecolor="#2a68bc", alpha=a,
                               edgecolor="#1b4a8a", lw=1))
    ax.annotate(f"V 리드인 {p.lead_angle_deg:.0f}°\n(높이 {p.Hv:.1f} mm)", xy=(-p.S / 2 + 6, p.Hv * 0.3 + 0.5),
                xytext=(-24.5, -8), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate(f"잠금 턱 {p.overhang:.0f} mm", xy=(-p.c - p.L_tw / 2, p.t_s0 - p.overhang / 2),
                xytext=(-24.5, p.t_s0 - 8), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate(f"{p.twist_deg:.0f}° 비틀림 슬롯\n(나선 압착 {p.helix} mm)", xy=(-p.c - 2, p.t_s1 - 1),
                xytext=(6, p.t_s1 - 3), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(sMax - 0.5, -1.2, "림", ha="right", fontsize=10)
    ax.text(path[0, 0] - hw - 1.5, path[0, 1], "러그 시작 위치\n(+28° 어긋남)", ha="right", va="center",
            fontsize=10, color="#1b4a8a")
    ax.set_xlim(sMin - 1, sMax + 1)
    ax.set_ylim(p.collar_h + 2, -(p.lug_h + 17))
    ax.set_aspect("equal")
    ax.set_xlabel(f"원주 방향 거리 s [mm]  (칼라 반지름 {p.collar_r:.0f} mm 기준)")
    ax.set_ylabel("림에서 아래로 깊이 [mm]")
    ax.set_title("바요넷 홈 펼침도 (러그 1개 구간, 바깥에서 본 모습)", fontsize=13)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "08_groove_unrolled.png"), facecolor="white")
    plt.close(fig)


def video_mating(yaw0=28, fps=30):
    us = [0.0] * 12 + list(np.linspace(0, 1, 75)) + [1.0] * 12 + list(np.linspace(1, 2, 50)) + [2.0] * 30
    fn = os.path.join(OUT, "09_mating_ghost.mp4")
    w = imageio.get_writer(fn, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    gif = []
    for i, u in enumerate(us):
        img, _ = ghost_frame(u, yaw0, (1280, 960))
        w.append_data(img)
        if i % 3 == 0:
            gif.append(img[::2, ::2])
        if i % 30 == 0:
            print(f"  mating frame {i}/{len(us)}")
    w.close()
    imageio.mimsave(os.path.join(OUT, "10_mating_ghost.gif"), gif, duration=0.1, loop=0)


def video_turntable(fps=30, n=150):
    lock = state(2, 0)
    ms = [(placed(HOLDER, "holder", 0, 0), 1), (placed(BODY, "body", ZB, 0), 1),
          (placed(LID, "lid", lock["lid_z"], lock["yaw"]), 1)]
    pl = plotter((1280, 960))
    for m, op in ms:
        add(pl, m, op)
    w = imageio.get_writer(os.path.join(OUT, "11_turntable_locked.mp4"), fps=fps, codec="libx264", quality=8,
                           macro_block_size=8)
    for i in range(n):
        cam(pl, 35 + 360 * i / n, 20, 560, (0, 0, 95))
        pl.render()
        w.append_data(pl.screenshot(return_img=True))
    w.close()
    pl.close()


if __name__ == "__main__":
    which = sys.argv[1:] or ["renders", "sequence", "unrolled", "turntable", "mating"]
    for k in which:
        print("==", k)
        dict(renders=renders, sequence=sequence, unrolled=unrolled, turntable=video_turntable,
             mating=video_mating)[k]()
    print("done ->", OUT)
