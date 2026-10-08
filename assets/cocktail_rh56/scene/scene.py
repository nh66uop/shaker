"""OpenArm + RH56F1 (right) one-hand bayonet lid mating scene.

Story (one hand, holder takes the reaction torque):
  approach (lid carried, +28 deg yaw error) -> release -> lid self-aligns on the V lead-in
  -> re-grasp knob -> twist 30 deg to lock -> release & retreat.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "presentation"))
import kin  # noqa: E402
import make_media as mm  # noqa: E402  (shaker STEP geometry + bayonet kinematics)

R = kin.Robot()
P = mm.p

TABLE_Z = 0.205                              # hdgp rh_aglt env: table_surface_z
HOLDER_XY = (0.33, -0.17)                    # in front of the right arm
ARM = [f"r_aj_{i}" for i in range(1, 8)]
CLOSED = dict(opp=2.09, th=0.38, f=0.70)     # top pinch across the 36 mm knob span
OPEN = dict(opp=2.09, th=0.08, f=0.30)
KNOB_MID = (P.lid_plate + P.knob_h / 2) / 1000.0


def hand_q(h):
    q = {"r_hj_thumb_1": h["opp"], "r_hj_thumb_2": h["th"]}
    for n in ("index", "middle", "ring", "pinky"):
        q[f"r_hj_{n}_1"] = h["f"]
    return q


def _grasp_T():
    """lid-frame pose of r_hl_base for the closed pinch (fingers down, thumb->fingers along +X_lid)."""
    T = R.fk(hand_q(CLOSED))
    B = np.linalg.inv(T["r_hl_base"])
    tip = {k: (B @ T[f"r_hl_{k}_tip"])[:3, 3] for k in ("thumb", "index", "middle")}
    fm = (tip["index"] + tip["middle"]) / 2
    c_h = (tip["thumb"] + fm) / 2
    a = fm - tip["thumb"]
    a[2] = 0.0
    u1 = a / np.linalg.norm(a)
    u3 = np.array([0, 0, 1.0])
    U = np.column_stack([u1, np.cross(u3, u1), u3])
    v1, v3 = np.array([1.0, 0, 0]), np.array([0, 0, -1.0])
    V = np.column_stack([v1, np.cross(v3, v1), v3])
    Rl = V @ U.T
    G = np.eye(4)
    G[:3, :3] = Rl
    G[:3, 3] = np.array([0, 0, KNOB_MID]) - Rl @ c_h
    return G


GRASP0 = _grasp_T()


def grasp(alpha_deg=0.0):
    """tilt the whole hand about the pinch axis (lid X through the knob centre); tips stay on the knob."""
    a = math.radians(alpha_deg)
    Rx = np.eye(4)
    Rx[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    C = np.eye(4)
    C[2, 3] = KNOB_MID
    return C @ Rx @ np.linalg.inv(C) @ GRASP0


GRASP = GRASP0


def Tz(x, y, z, yaw_deg=0.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = (x, y, z)
    return T


def lid_world(lid_z_mm, yaw_deg, psi):
    """lid pose in world. lid_z_mm is make_media's z (holder base frame, mm)."""
    return Tz(HOLDER_XY[0], HOLDER_XY[1], TABLE_Z + lid_z_mm / 1000.0, yaw_deg + psi)


def solve(target, seed, rng=None, tries=1):
    best = None
    for k in range(tries):
        q, ep, er = R.ik("r_hl_base", target, ARM, seed, seed_noise=0.6 if k else 0.0, rng=rng)
        if best is None or ep + 0.05 * er < best[1] + 0.05 * best[2]:
            best = (q, ep, er)
    return best


NOMINAL = {"r_aj_1": 0.3, "r_aj_2": 0.2, "r_aj_3": 0.0, "r_aj_4": 1.6, "r_aj_5": 0.0, "r_aj_6": 0.0, "r_aj_7": 0.0,
           "l_aj_2": -0.15, "l_aj_4": 0.35}

if __name__ == "__main__":
    import itertools
    import time
    rng = np.random.default_rng(0)
    keys = [(mm.state(0, 28), 70), (mm.state(0, 28), 0), (mm.state(1, 28), 0), (mm.state(1.5, 28), 0), (mm.state(2, 28), 0)]
    t0 = time.time()
    res = []
    for psi, alpha in itertools.product(range(0, 360, 15), range(-40, 41, 10)):
        G = grasp(alpha)
        q = dict(NOMINAL)
        tot, worst = 0.0, 0.0
        for k, (st, dz) in enumerate(keys):
            Th = lid_world(st["lid_z"] + dz, st["yaw"], psi) @ G
            q, ep, er = solve(Th, q, rng, tries=3 if k == 0 else 1)
            c = ep * 1000 + math.degrees(er) * 0.5
            tot += c
            worst = max(worst, c)
        res.append((worst, tot, psi, alpha))
    res.sort()
    print("search %.0fs" % (time.time() - t0))
    for r in res[:10]:
        print(r)
