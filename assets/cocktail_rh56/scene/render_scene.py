#!/usr/bin/env python3
"""Render the one-hand lid-mating scene: stills + mp4/gif.

    .venv-cad/Scripts/python assets/cocktail_rh56/scene/render_scene.py [stills] [video]
"""
import math
import os
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pyvista as pv
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scene as S  # noqa: E402

mm, R = S.mm, S.R
OUT = os.path.join(os.path.dirname(HERE), "presentation", "robot")
os.makedirs(OUT, exist_ok=True)
TABLE_OBJ = os.path.join(os.path.dirname(os.path.dirname(HERE)), "simulation_setting", "env_v1", "meshes", "visual", "env_v1.obj")
PSI = 255.0          # lid yaw offset (4-fold symmetric) chosen by scene.py search
S.GRASP = S.grasp(20.0)  # hand tilted 20 deg about the pinch axis -> all key poses reachable
YAW0 = 28.0
pv.OFF_SCREEN = True
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]

ARM_WHITE, ARM_DARK, HAND, HAND_TIP = (232, 234, 237), (70, 74, 80), (52, 55, 60), (120, 126, 134)


def link_color(name):
    if name.startswith(("r_hl_", "l_hl_")):
        return HAND_TIP if name.endswith(("_tip", "_sensor")) else HAND
    if name.startswith(("r_al_", "l_al_")):
        return ARM_DARK if name.endswith(("_0", "_7")) else ARM_WHITE
    if name.startswith("head"):
        return (60, 64, 70)
    return (150, 156, 164)


# ---------------------------------------------------------------------------
def load_robot():
    out = {}
    for link, vis in R.visuals.items():
        ms = []
        for path, scale, T in vis:
            tm = trimesh.load(path, force="mesh")
            S_ = np.diag(list(scale) + [1.0])
            tm.apply_transform(T @ S_)
            ms.append(pv.wrap(tm))
        if ms:
            m = ms[0] if len(ms) == 1 else ms[0].merge(ms[1:])
            out[link] = m
    return out


def shaker_meshes():
    d = {}
    for name, shape in (("body", mm.BODY), ("lid", mm.LID), ("holder", mm.HOLDER)):
        m = mm.to_pv(shape, tol=0.08)
        m.cell_data["rgb"] = mm.classify(m, name)
        m = m.scale(0.001, inplace=False)
        d[name] = m
    return d


ROBOT = load_robot()
SHAKER = shaker_meshes()
TABLE = pv.wrap(trimesh.load(TABLE_OBJ, force="mesh"))


# ---------------------------------------------------------------------------
# timeline: list of dicts(lid=(z_mm, yaw), hand=(z_mm, yaw, lift_mm), grip=0..1 closed, label)
def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    return t * t * (3 - 2 * t)


def timeline(scale=1.0):
    n = lambda k: max(2, int(k * scale))  # noqa: E731
    st0, st1, st2 = mm.state(0, YAW0), mm.state(1, YAW0), mm.state(2, YAW0)
    F = []
    for i in range(n(40)):                                    # A approach, lid carried
        t = ease(i / (n(40) - 1))
        z = st0["lid_z"] + lerp(70, 0, t)
        F.append(dict(lid=(z, st0["yaw"]), hand=(z, st0["yaw"], 0), grip=1, label="① 접근: 리드를 쥐고 바디 위로 (+28° 어긋남)"))
    for i in range(n(18)):                                    # B release
        t = ease(i / (n(18) - 1))
        F.append(dict(lid=(st0["lid_z"], st0["yaw"]), hand=(st0["lid_z"], st0["yaw"], lerp(0, 4, t)), grip=1 - t,
                      label="② 놓기: 손을 연다"))
    for i in range(n(50)):                                    # C self-align by gravity
        u = i / (n(50) - 1)
        st = mm.state(u, YAW0)
        F.append(dict(lid=(st["lid_z"], st["yaw"]), hand=(st0["lid_z"], st0["yaw"], lerp(4, 20, ease(u))), grip=0,
                      label="③ 중력 정렬: V 경사면이 리드를 홈으로 안내"))
    for i in range(n(30)):                                    # D move above seated knob
        t = ease(i / (n(30) - 1))
        F.append(dict(lid=(st1["lid_z"], st1["yaw"]),
                      hand=(lerp(st0["lid_z"], st1["lid_z"], t), lerp(st0["yaw"], st1["yaw"], t), lerp(20, 15, t)), grip=0,
                      label="④ 다시 잡기"))
    for i in range(n(26)):
        t = ease(i / (n(26) - 1))
        F.append(dict(lid=(st1["lid_z"], st1["yaw"]), hand=(st1["lid_z"], st1["yaw"], lerp(15, 0, min(1, 2 * t))),
                      grip=max(0.0, 2 * t - 1), label="④ 다시 잡기"))
    for i in range(n(45)):                                    # E twist
        u = 1 + i / (n(45) - 1)
        st = mm.state(u, YAW0)
        F.append(dict(lid=(st["lid_z"], st["yaw"]), hand=(st["lid_z"], st["yaw"], 0), grip=1,
                      label="⑤ 30° 비틀어 잠금 (홀더가 반력을 받음)"))
    for i in range(n(36)):                                    # F release & retreat
        t = i / (n(36) - 1)
        F.append(dict(lid=(st2["lid_z"], st2["yaw"]), hand=(st2["lid_z"], st2["yaw"], lerp(0, 50, ease(max(0, 1.6 * t - 0.6)))),
                      grip=1 - ease(min(1, 2.5 * t)), label="⑥ 잠김 완료 → 손 빼기"))
    return F


def _key_seeds(rng):
    """arm solutions at the key poses, chained like the scene.py search (known to converge)."""
    keys = [(mm.state(0, YAW0), 70), (mm.state(0, YAW0), 0), (mm.state(1, YAW0), 0),
            (mm.state(1.5, YAW0), 0), (mm.state(2, YAW0), 0)]
    q, seeds = dict(S.NOMINAL), []
    for k, (st, dz) in enumerate(keys):
        q, _, _ = S.solve(S.lid_world(st["lid_z"] + dz, st["yaw"], PSI) @ S.GRASP, q, rng, tries=3 if k == 0 else 1)
        seeds.append(dict(q))
    return seeds


def _dist(a, b):
    return sum((a.get(n, 0) - b.get(n, 0)) ** 2 for n in S.ARM) ** 0.5


def solve_all(F):
    rng = np.random.default_rng(0)
    seeds = _key_seeds(rng)
    q = dict(seeds[0])
    out = []
    for k, fr in enumerate(F):
        z, yaw, lift = fr["hand"]
        target = S.lid_world(z + lift, yaw, PSI) @ S.GRASP
        cand = [S.solve(target, q, rng, tries=1)]
        ok = lambda c: c[1] < 5e-4 and c[2] < 5e-3  # noqa: E731  position AND orientation must converge
        if not ok(cand[0]):          # previous-frame seed fell into a bad branch: try the key solutions
            cand += [S.solve(target, s, rng, tries=1) for s in seeds]
        good = [c for c in cand if ok(c)]
        q, ep, er = min(good, key=lambda c: _dist(c[0], q)) if good else min(cand, key=lambda c: c[1])
        if ep > 1e-3 or er > 1e-2:
            print(f"  frame {k}: IK err {ep * 1000:.1f} mm / {math.degrees(er):.1f} deg")
        g = fr["grip"]
        h = {key: S.OPEN[key] + (S.CLOSED[key] - S.OPEN[key]) * g for key in S.CLOSED}
        qq = dict(q)
        qq.update(S.hand_q(h))
        out.append(qq)
    return out


# ---------------------------------------------------------------------------
def build(pl, q, fr):
    T = R.fk(q)
    actors = []
    for link, m in ROBOT.items():
        actors.append(pl.add_mesh(m.transform(T[link], inplace=False), color=link_color(link), smooth_shading=True,
                                  split_sharp_edges=True, specular=0.25, specular_power=15))
    H = S.Tz(S.HOLDER_XY[0], S.HOLDER_XY[1], S.TABLE_Z)
    Bm = H @ S.Tz(0, 0, mm.ZB / 1000)
    L = S.lid_world(fr["lid"][0], fr["lid"][1], PSI)
    for name, M in (("holder", H), ("body", Bm), ("lid", L)):
        actors.append(pl.add_mesh(SHAKER[name].transform(M, inplace=False), scalars="rgb", rgb=True,
                                  smooth_shading=True, split_sharp_edges=True, specular=0.35, specular_power=20))
    return actors


def plotter(size):
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    pl.add_mesh(TABLE, color=(206, 208, 211), smooth_shading=True, specular=0.1)
    return pl


CAMS = {
    "wide": [(1.35, -1.05, 1.05), (0.18, -0.08, 0.40), (0, 0, 1)],
    "close": [(0.78, -0.52, 0.62), (0.33, -0.17, 0.41), (0, 0, 1)],
    "side": [(0.33, -0.95, 0.47), (0.33, -0.17, 0.41), (0, 0, 1)],
}


def shot(pl, q, fr, cam, view_angle=30):
    acts = build(pl, q, fr)
    pl.camera_position = CAMS[cam]
    pl.camera.view_angle = view_angle
    img = pl.screenshot(return_img=True)
    for a in acts:
        pl.remove_actor(a)
    return img


def stills(F, Q):
    keys = [(0, "① 접근 (+28° 어긋남)"), (57, "② 손을 연다"), (107, "③ 중력으로 자동 정렬"),
            (162, "④ 노브 다시 잡기"), (206, "⑤ 30° 비틀어 잠금"), (230, "⑥ 잠김 → 손 빼기")]
    pl = plotter((1600, 1200))
    imageio.imwrite(os.path.join(OUT, "r01_overview_locked.png"), shot(pl, Q[206], F[206], "wide", 28))
    imageio.imwrite(os.path.join(OUT, "r02_close_approach.png"), shot(pl, Q[0], F[0], "close", 32))
    imageio.imwrite(os.path.join(OUT, "r03_close_twist_locked.png"), shot(pl, Q[206], F[206], "close", 32))
    imageio.imwrite(os.path.join(OUT, "r04_side_seated.png"), shot(pl, Q[107], F[107], "side", 30))
    pl.close()
    pl = plotter((900, 900))
    fig, axs = plt.subplots(2, 3, figsize=(18, 12.6), dpi=100)
    for ax, (k, lab) in zip(axs.ravel(), keys):
        ax.imshow(shot(pl, Q[k], F[k], "close", 32))
        ax.set_title(lab, fontsize=17)
        ax.axis("off")
    pl.close()
    fig.suptitle("OpenArm + Inspire RH56F1 — 한 손으로 바요넷 리드 결합", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "r05_sequence_6steps.png"), facecolor="white")
    plt.close(fig)


def video(F, Q, cam="close", name="r06_one_hand_mating", size=(1280, 960)):
    pl = plotter(size)
    w = imageio.get_writer(os.path.join(OUT, name + ".mp4"), fps=30, codec="libx264", quality=8, macro_block_size=8)
    gif = []
    for i, (q, fr) in enumerate(zip(Q, F)):
        img = shot(pl, q, fr, cam, 32 if cam == "close" else 28)
        w.append_data(img)
        if i % 3 == 0:
            gif.append(img[::2, ::2])
        if i % 40 == 0:
            print(f"  {name} frame {i}/{len(F)}")
    for _ in range(30):
        w.append_data(img)
    w.close()
    pl.close()
    imageio.mimsave(os.path.join(OUT, name + ".gif"), gif, duration=0.1, loop=0)


# ---------------------------------------------------------------------------
# stills only: solve each key frame independently (hand may tilt about the pinch axis, alpha)
def solve_key(fr, rng, prev=None):
    z, yaw, lift = fr["hand"]
    best = None
    for alpha in range(-40, 41, 10):
        G = S.grasp(alpha)
        tgt = S.lid_world(z + lift, yaw, PSI) @ G
        for t in range(6):
            seed = prev if (prev is not None and t == 0) else S.NOMINAL
            q, ep, er = R.ik("r_hl_base", tgt, S.ARM, seed, iters=600, seed_noise=0.0 if t == 0 else 1.0, rng=rng)
            c = ep * 1000 + math.degrees(er)
            if c < 0.5:
                c += 0.02 * abs(alpha)          # prefer the least tilted feasible grasp
            if best is None or c < best[0]:
                best = (c, q, alpha, ep, er)
    c, q, alpha, ep, er = best
    g = fr["grip"]
    h = {key: S.OPEN[key] + (S.CLOSED[key] - S.OPEN[key]) * g for key in S.CLOSED}
    q = dict(q)
    q.update(S.hand_q(h))
    return q, alpha, ep, er


def stills_keys():
    F = timeline()
    keys = [(0, "① 접근 (+28° 어긋남)"), (57, "② 손을 연다"), (107, "③ 중력으로 자동 정렬"),
            (170, "④ 노브 다시 잡기"), (206, "⑤ 30° 비틀어 잠금"), (232, "⑥ 잠김 → 손 빼기")]
    rng = np.random.default_rng(0)
    Q, prev = {}, None
    for k, lab in keys:
        q, alpha, ep, er = solve_key(F[k], rng, prev)
        prev = q
        Q[k] = q
        T = R.fk(q)
        knob = (S.lid_world(*F[k]["lid"], PSI) @ np.array([0, 0, S.KNOB_MID, 1]))[:3]
        d = [round(np.linalg.norm(T[f"r_hl_{t}_tip"][:3, 3] - knob) * 1000) for t in ("thumb", "index", "middle")]
        print(f"  key {k}: alpha {alpha:+d}  IK {ep * 1000:.2f} mm / {math.degrees(er):.2f} deg  tip-knob mm {d}")
    pl = plotter((1600, 1200))
    imageio.imwrite(os.path.join(OUT, "r01_overview_locked.png"), shot(pl, Q[206], F[206], "wide", 28))
    imageio.imwrite(os.path.join(OUT, "r02_close_approach.png"), shot(pl, Q[0], F[0], "close", 32))
    imageio.imwrite(os.path.join(OUT, "r03_close_twist_locked.png"), shot(pl, Q[206], F[206], "close", 32))
    imageio.imwrite(os.path.join(OUT, "r04_side_seated.png"), shot(pl, Q[107], F[107], "side", 30))
    pl.close()
    pl = plotter((900, 900))
    fig, axs = plt.subplots(2, 3, figsize=(18, 12.6), dpi=100)
    for ax, (k, lab) in zip(axs.ravel(), keys):
        ax.imshow(shot(pl, Q[k], F[k], "close", 32))
        ax.set_title(lab, fontsize=17)
        ax.axis("off")
    pl.close()
    fig.suptitle("OpenArm + Inspire RH56F1 — 한 손으로 바요넷 리드 결합", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "r05_sequence_6steps.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["keys"]
    if which == ["keys"]:
        stills_keys()
        print("done ->", OUT)
        sys.exit()
    F = timeline()
    print("frames", len(F))
    cache = os.path.join(HERE, "q_cache.npy")
    if os.path.exists(cache) and "resolve" not in which:
        Q = list(np.load(cache, allow_pickle=True))
    else:
        Q = solve_all(F)
        np.save(cache, np.array(Q, dtype=object), allow_pickle=True)
    for k in (0, 57, 107, 162, 206, 230):   # grasp sanity: fingertip-to-knob distance
        T = R.fk(Q[k])
        knob = (S.lid_world(*F[k]["lid"], PSI) @ np.array([0, 0, S.KNOB_MID, 1]))[:3]
        d = [np.linalg.norm(T[f"r_hl_{t}_tip"][:3, 3] - knob) * 1000 for t in ("thumb", "index", "middle")]
        print(f"  frame {k}: tip-knob mm", [round(x) for x in d])
    if "stills" in which:
        stills(F, Q)
    if "video" in which:
        video(F, Q, "close", "r06_one_hand_mating")
        video(F, Q, "wide", "r07_one_hand_mating_wide")
    print("done ->", OUT)

