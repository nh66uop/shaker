"""Minimal URDF kinematics (FK + damped-least-squares IK) for openarm_rh56f1_bi_rl."""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

URDF_ROOT = r"C:\Users\tony0\orca\urdf"
URDF = os.path.join(URDF_ROOT, "generated", "rl", "openarm_rh56f1_bi_rl", "openarm_rh56f1_bi_rl.urdf")
LINUX_PREFIX = "file:///home/user/rl_ws/urdf/"


def rpy_T(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = xyz
    return T


def axang(a, t):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s, C = math.cos(t), math.sin(t), 1 - math.cos(t)
    T = np.eye(4)
    T[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                 [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                 [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
    return T


def _vec(s, d="0 0 0"):
    return [float(v) for v in (s or d).split()]


class Robot:
    def __init__(self, path=URDF):
        root = ET.parse(path).getroot()
        self.joints = {}      # child link -> dict
        self.by_name = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            lim = j.find("limit")
            mim = j.find("mimic")
            d = dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                     child=j.find("child").get("link"),
                     T=rpy_T(_vec(o.get("xyz") if o is not None else None), _vec(o.get("rpy") if o is not None else None)),
                     axis=_vec(ax.get("xyz") if ax is not None else None, "0 0 1"),
                     lo=float(lim.get("lower")) if lim is not None and lim.get("lower") else -math.pi,
                     hi=float(lim.get("upper")) if lim is not None and lim.get("upper") else math.pi,
                     mimic=(mim.get("joint"), float(mim.get("multiplier", 1)), float(mim.get("offset", 0)))
                     if mim is not None else None)
            self.joints[d["child"]] = d
            self.by_name[d["name"]] = d
        self.visuals = {}     # link -> [(path, scale, T_origin)]
        for l in root.findall("link"):
            out = []
            for v in l.findall("visual"):
                m = v.find("geometry/mesh")
                if m is None:
                    continue
                o = v.find("origin")
                p = os.path.join(URDF_ROOT, m.get("filename").replace(LINUX_PREFIX, ""))
                out.append((p, _vec(m.get("scale"), "1 1 1"),
                            rpy_T(_vec(o.get("xyz") if o is not None else None),
                                  _vec(o.get("rpy") if o is not None else None))))
            self.visuals[l.get("name")] = out
        self.links = [l.get("name") for l in root.findall("link")]

    def value(self, name, q):
        j = self.by_name[name]
        if j["mimic"]:
            src, k, off = j["mimic"]
            return k * self.value(src, q) + off
        return q.get(name, 0.0)

    def fk(self, q, base=np.eye(4)):
        T = {}

        def get(link):
            if link in T:
                return T[link]
            j = self.joints.get(link)
            if j is None:
                T[link] = base
                return base
            M = get(j["parent"]) @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                M = M @ axang(j["axis"], self.value(j["name"], q))
            T[link] = M
            return M

        for l in self.links:
            get(l)
        return T

    # ----- IK on a chain of named joints, target = 4x4 pose of `link`
    def ik(self, link, target, chain, q0, iters=400, w_rot=0.25, seed_noise=0.0, rng=None):
        q = dict(q0)
        names = list(chain)
        lo = np.array([self.by_name[n]["lo"] for n in names])
        hi = np.array([self.by_name[n]["hi"] for n in names])
        x = np.array([q.get(n, 0.0) for n in names])
        if seed_noise and rng is not None:
            x = np.clip(x + rng.normal(0, seed_noise, len(x)), lo, hi)

        path = []                      # root -> link joint list (only these matter for the target pose)
        l = link
        while l in self.joints:
            path.append(self.joints[l])
            l = self.joints[l]["parent"]
        path.reverse()

        def fk_link(qq):
            M = np.eye(4)
            for j in path:
                M = M @ j["T"]
                if j["type"] in ("revolute", "continuous"):
                    M = M @ axang(j["axis"], self.value(j["name"], qq))
            return M

        def err(xv):
            qq = dict(q)
            qq.update(zip(names, xv))
            M = fk_link(qq)
            dp = target[:3, 3] - M[:3, 3]
            R = target[:3, :3] @ M[:3, :3].T
            # true log map: the (R - R^T)/2 shortcut is zero at a 180 deg error and lets IK converge flipped
            ang = Rotation.from_matrix(R).as_rotvec()
            return np.concatenate([dp, w_rot * ang])

        lam = 1e-3
        for _ in range(iters):
            e = err(x)
            if np.linalg.norm(e) < 1e-5:
                break
            J = np.zeros((6, len(x)))
            h = 1e-5
            for i in range(len(x)):
                xp = x.copy()
                xp[i] += h
                J[:, i] = (err(xp) - e) / -h
            dx = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            x = np.clip(x + dx, lo, hi)
        q.update(zip(names, x))
        e = err(x)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]) / w_rot)
