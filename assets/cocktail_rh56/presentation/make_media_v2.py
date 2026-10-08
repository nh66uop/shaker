#!/usr/bin/env python3
"""Presentation media for the RH56 bayonet shaker — rev 2 (inverted bayonet, no holder).

    .venv-cad/Scripts/python assets/cocktail_rh56/presentation/make_media_v2.py [renders sequence unrolled turntable mating]

rev 2: the cup (body) carries 4 outward lugs on a plain collar; the lid skirt carries the
V lead-in / twist slot / locking ledge on its inner wall. Same shot list as make_media.py
(rev 1), written to ./v2/ with a `_v2` suffix. rev 1 files are left untouched.
Requires: cadquery, pyvista, matplotlib, imageio[-ffmpeg].
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
OUT = os.path.join(HERE, "v2")
os.makedirs(OUT, exist_ok=True)
W, H = 1600, 1200

C = dict(steel=(196, 202, 208), groove=(226, 116, 34), lid=(160, 178, 200), lug=(42, 104, 188), bg="white")


def fn(name):
    base, ext = os.path.splitext(name)
    return os.path.join(OUT, f"{base}_v2{ext}")


def load(name):
    return cq.importers.importStep(os.path.join(STEP, f"rh56_shaker_{name}.step")).val()


BODY, LID = load("body"), load("lid")


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
        lug = (z > p.body_h - p.collar_h) & (r > p.collar_r + 0.3)
        col[lug] = C["lug"]
    else:
        col[:] = C["lid"]
        band = (z < -0.01) & (z > -p.skirt_len - 0.01) & (r > p.lid_in_r + 0.3) & (r < p.lid_in_r + p.groove_depth + 0.6)
        # exclude the conical entry chamfer at the skirt mouth (not part of the groove)
        u = z + p.skirt_len
        r_ch = p.lid_in_r + p.lid_entry_chamfer * (1 - u / 4.0)
        chamfer = (u < 4.0) & (np.abs(r - r_ch) < 0.2)
        col[band & ~chamfer] = C["groove"]
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


_CUT_CACHE = {}


def quadrant_cut(shape, ang):
    """remove theta in (ang, ang+90); cut in z-bands (OCCT occasionally no-ops a full-height box)."""
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
                break
            v0 = s.Volume()
            s2 = s.cut(box)
            if s2.Volume() < v0 - 1e-3:
                s = s2
                break
    _CUT_CACHE[key] = s
    return s


_MESH_CACHE = {}


def local_mesh(shape, part, cut_local=None):
    key = (id(shape), None if cut_local is None else round(cut_local, 3))
    if key not in _MESH_CACHE:
        s = shape if cut_local is None else quadrant_cut(shape, cut_local)
        m = to_pv(s)
        m.cell_data["rgb"] = classify(m, part)
        _MESH_CACHE[key] = m
    return _MESH_CACHE[key]


def placed(shape, part, z, yaw, cut_yaw=None):
    """shape in local frame -> PolyData in world (rotate yaw, translate z); cut in world quadrant."""
    m = local_mesh(shape, part, None if cut_yaw is None else cut_yaw - yaw)
    col = m.cell_data["rgb"]
    m = m.rotate_z(yaw, inplace=False).translate((0, 0, z), inplace=False)
    m.cell_data["rgb"] = section_shade(m, col, cut_yaw, cut_yaw is not None)
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
# kinematics, in the lid-groove frame: s_rel = arc position of the body lug relative to the
# lid groove centre (lid yaw = -s_rel / collar_r), w = height above the skirt bottom.
tanL = math.tan(math.radians(p.lead_angle_deg))
hw = p.lug_w / 2
W_SEAT_TOP = p.skirt_len - p.lug_t0 - p.gasket      # lug top height in the groove at drop-in seat
D0 = W_SEAT_TOP + 10.0                              # start: lug top 10 mm below the skirt mouth


def lim(w):
    return math.inf if w <= 0 else max(p.side_clr, p.S / 2 - w / tanL - hw)


def ease(u):
    return 2 * u * u if u < .5 else 1 - (-2 * u + 2) ** 2 / 2


def state(u, yaw0):
    """u in [0,1]: drop (V lead-in self-aligns), u in [1,2]: twist-lock. yaw0 = initial lid misalignment."""
    s0 = -p.collar_r * math.radians(yaw0)
    if u <= 1:
        drop = D0 * (1 - ease(u))
        m = lim(W_SEAT_TOP - drop)
        s = max(-m, min(m, s0))
        twist = 0.0
    else:
        drop = 0.0
        sd = max(-p.side_clr, min(p.side_clr, s0))
        twist = ease(min(1.0, u - 1))
        s = sd + twist * (p.L_tw - sd)
    dz = p.helix * twist
    return dict(yaw=-math.degrees(s / p.collar_r), lid_z=p.body_h + p.gasket + drop - dz,
                s=s, w_top=W_SEAT_TOP - drop + dz)


# ---------------------------------------------------------------------------
def still(name, meshes, az, el, dist, focus, size=(W, H)):
    pl = plotter(size)
    for m, op in meshes:
        add(pl, m, op)
    cam(pl, az, el, dist, focus)
    img = pl.screenshot(fn(name) if name else None, return_img=True)
    pl.close()
    return img


def renders():
    lock = state(2, 0)
    zc = p.body_h
    body_w = placed(BODY, "body", 0, 0)
    lid_lock = placed(LID, "lid", lock["lid_z"], lock["yaw"])
    still("01_assembly_locked.png", [(body_w, 1), (lid_lock, 1)], 35, 22, 520, (0, 0, 95))

    still("02_exploded.png", [(body_w, 1), (placed(LID, "lid", p.body_h + 70, 0), 1)], 35, 18, 640, (0, 0, 115))

    still("04_body_collar.png", [(body_w, 1)], 20, 28, 210, (0, 0, p.body_h - 15))

    lid_up = local_mesh(LID, "lid").copy()
    lid_up = lid_up.rotate_x(180, inplace=False)
    still("05_lid_underside.png", [(lid_up, 1)], 30, 38, 230, (0, 0, 10))

    # lid with the near half removed: the groove (V / slot / ledge) on the inner wall at theta = 0
    lid_half = quadrant_cut(quadrant_cut(LID, 90), 180)
    m = to_pv(lid_half)
    m.cell_data["rgb"] = section_shade(m, classify(m, "lid"), 90, True)
    still("06_lid_inner_groove.png", [(m, 1)], 180, 8, 190, (0, 0, -p.skirt_len / 2))

    # cutaway, locked: section through the lug centre (theta = 0, fixed on the cup)
    pl_c = [(placed(BODY, "body", 0, 0, cut_yaw=0), 1),
            (placed(LID, "lid", lock["lid_z"], lock["yaw"], cut_yaw=0), 1)]
    still("03_cutaway_locked.png", pl_c, 45, 16, 175, (0, 0, zc - 12))
    ghost_frame(2, 0, (W, H), "03b_ghost_locked.png")


def ghost_frame(u, yaw0, size, name=None):
    """cup opaque (lugs blue), lid translucent with its groove walls opaque; camera faces the lug."""
    st = state(u, yaw0)
    body = placed(BODY, "body", 0, 0)
    lid = placed(LID, "lid", st["lid_z"], st["yaw"])
    is_g = np.all(lid.cell_data["rgb"] == np.array(C["groove"], np.uint8), axis=1)
    groove = lid.extract_cells(np.where(is_g)[0]).extract_surface()
    shell = lid.extract_cells(np.where(~is_g)[0]).extract_surface()
    pl = plotter(size)
    pl.enable_depth_peeling(8)
    add(pl, body)
    add(pl, groove, 0.38)
    add(pl, shell, 0.16)
    cam(pl, 0, 10, 185, (0, 0, p.body_h - 4))
    img = pl.screenshot(fn(name) if name else None, return_img=True)
    pl.close()
    return img, st


def sequence(yaw0=28):
    stages = [(0.0, "① 하강 (각도 오차 +28°)"), (0.55, "② 리드 V 경사면이 이빨을 유도"), (1.0, "③ 안착 (자동 정렬 완료)"),
              (1.5, "④ 30° 비틀어 잠금"), (2.0, "⑤ 잠김 — 이빨이 리드 잠금 턱 위")]
    fig, axs = plt.subplots(1, 5, figsize=(20, 5.2), dpi=110)
    for ax, (u, lab) in zip(axs, stages):
        img, st = ghost_frame(u, yaw0, (900, 900))
        ax.imshow(img)
        ax.set_title(lab, fontsize=13)
        ax.text(0.5, -0.04, f"잠금 위치 대비 높이 {st['lid_z'] - state(2, yaw0)['lid_z']:+.1f} mm · 회전 {st['yaw']:+.1f}°",
                transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color="#444")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fn("07_mating_sequence.png"), facecolor="white")
    plt.close(fig)


def unrolled(yaw0=28):
    fig, ax = plt.subplots(figsize=(10, 8.4), dpi=150)
    sMin, sMax = -p.S / 2, p.S / 2
    ax.add_patch(Rectangle((sMin, 0), p.S, p.skirt_len, color="#a9b6c6", alpha=0.6, lw=0))
    poly = np.array(g.lid_groove_poly(p))
    gp = MPoly(poly, closed=True, facecolor="#e27422", alpha=0.55, edgecolor="#b8561a", lw=1.6)
    ax.add_patch(gp)
    gp.set_clip_path(Rectangle((sMin, 0), p.S, p.skirt_len, transform=ax.transData))
    ax.axhline(0, color="k", lw=1.2)
    ax.axhline(p.skirt_len, color="#555", lw=0.8, ls=":")
    if p.detent > 0:
        d = np.array(g.detent_poly(p))
        ax.add_patch(MPoly(d, closed=True, facecolor="#7a3b10", alpha=0.8, lw=0))
        ax.annotate(f"딸깍 돌기 {p.detent} mm", xy=(d[:, 0].mean(), d[:, 1].mean()), xytext=(10, p.skirt_len - 3),
                    fontsize=10.5, arrowprops=dict(arrowstyle="-", color="#555"))
    us = np.linspace(0, 2, 300)
    path = np.array([[state(u, yaw0)["s"], state(u, yaw0)["w_top"] - p.lug_h / 2] for u in us])
    ax.plot(path[:, 0], path[:, 1], "--", color="#2a68bc", lw=1.4)
    for u, a in ((0.0, .25), (0.55, .4), (1.0, .6), (2.0, .95)):
        st = state(u, yaw0)
        ax.add_patch(Rectangle((st["s"] - hw, st["w_top"] - p.lug_h), p.lug_w, p.lug_h, facecolor="#2a68bc",
                               alpha=a, edgecolor="#1b4a8a", lw=1))
    ax.annotate(f"V 리드인 {p.lead_angle_deg:.0f}°\n(높이 {p.Hv:.1f} mm)", xy=(-p.S / 2 + 7, p.Hv * 0.3 + 1),
                xytext=(-24.5, p.Hv + 4), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate(f"잠금 턱 {p.overhang:.0f} mm", xy=(p.c + p.L_tw / 2, p.t_s0 - p.overhang / 2 + p.helix / 2),
                xytext=(9, p.t_s0 - 9), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate(f"{p.twist_deg:.0f}° 비틀림 슬롯\n(나선 압착 {p.helix} mm)", xy=(p.c + 4, (p.t_s0 + p.t_s1) / 2 + p.helix * 0.2),
                xytext=(-23, p.skirt_len - 3.5), fontsize=11, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(sMax - 0.5, -1.0, "스커트 하단 (입구)", ha="right", va="top", fontsize=10)
    ax.text(sMax - 0.5, p.skirt_len + 0.6, "리드 상판", ha="right", va="bottom", fontsize=10, color="#555")
    ax.text(path[0, 0] - hw - 1.5, path[0, 1], "컵 이빨 시작 위치\n(+28° 어긋남)", ha="right", va="center",
            fontsize=10, color="#1b4a8a")
    ax.set_xlim(sMin - 1, sMax + 1)
    ax.set_ylim(-(D0 - W_SEAT_TOP) - p.lug_h - 3, p.skirt_len + 3)
    ax.set_aspect("equal")
    ax.set_xlabel(f"원주 방향 거리 s [mm]  (칼라 반지름 {p.collar_r:.0f} mm 기준)")
    ax.set_ylabel("스커트 하단에서 위로 높이 [mm]")
    ax.set_title("리드 바요넷 홈 펼침도 v2 (홈 1개 구간, 바깥에서 투시한 모습)", fontsize=13)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fn("08_groove_unrolled.png"), facecolor="white")
    plt.close(fig)


def video_mating(yaw0=28, fps=30):
    us = [0.0] * 12 + list(np.linspace(0, 1, 75)) + [1.0] * 12 + list(np.linspace(1, 2, 50)) + [2.0] * 30
    w = imageio.get_writer(fn("09_mating_ghost.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    gif = []
    for i, u in enumerate(us):
        img, _ = ghost_frame(u, yaw0, (1280, 960))
        w.append_data(img)
        if i % 3 == 0:
            gif.append(img[::2, ::2])
        if i % 30 == 0:
            print(f"  mating frame {i}/{len(us)}", flush=True)
    w.close()
    imageio.mimsave(fn("10_mating_ghost.gif"), gif, duration=0.1, loop=0)


def video_turntable(fps=30, n=150):
    lock = state(2, 0)
    ms = [(placed(BODY, "body", 0, 0), 1), (placed(LID, "lid", lock["lid_z"], lock["yaw"]), 1)]
    pl = plotter((1280, 960))
    for m, op in ms:
        add(pl, m, op)
    w = imageio.get_writer(fn("11_turntable_locked.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    for i in range(n):
        cam(pl, 35 + 360 * i / n, 20, 520, (0, 0, 95))
        pl.render()
        w.append_data(pl.screenshot(return_img=True))
    w.close()
    pl.close()


def verify(yaw0=28):
    """B-rep check of the animated path: lug/lid interference volume along the sequence."""
    worst = 0.0
    for u in np.linspace(0, 2, 41):
        st = state(u, yaw0)
        L = LID.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), st["yaw"]).translate(cq.Vector(0, 0, st["lid_z"]))
        v = BODY.intersect(L).Volume()
        worst = max(worst, v)
        if v > 1e-3:
            print(f"  u={u:.2f} yaw={st['yaw']:+.1f} intersect {v:.3f} mm3")
    print(f"  path max interference {worst:.3f} mm3")


if __name__ == "__main__":
    which = sys.argv[1:] or ["verify", "renders", "sequence", "unrolled", "turntable", "mating"]
    for k in which:
        print("==", k, flush=True)
        dict(verify=verify, renders=renders, sequence=sequence, unrolled=unrolled, turntable=video_turntable,
             mating=video_mating)[k]()
    print("done ->", OUT)
