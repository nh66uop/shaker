"""물체 표면 점군 — 충돌 형상에서 면적 가중으로 뽑은 K 점 + 바깥 법선 (강체 `baseLink` 프레임, scale 적용).

★왜(09.22 사용자 결정): 보상 입력의 파지 기하(cup_radius·cup_axis·cup_half_height)는 컵 전용이다. "손이 충분히 가까우면
  손가락을 자연스럽게 닫아 좋은 파지로" 를 물체 모양과 무관하게 주려면 손 → **실제 표면** 거리가 필요하다.
  그 표면을 물체 충돌 메시에서 직접 만든다. 순수 numpy(+ 읽기에 pxr) — Isaac 불요, `/usr/bin/python3` 로 테스트한다.

★프레임 규약(09.22 확인)
  · env 의 `ctx.cup_pos`/`cup_quat` = `object.data.root_pos_w/root_quat_w` = 강체(`baseLink`) **actor 프레임**
    (IsaacLab `RigidObjectData.root_pos_w` == `root_link_pos_w`, COM 아님). 그래서 점군은 baseLink 프레임에 둔다.
  · cup_family 두 USD 모두 루트 Xform → `baseLink`(RigidBodyAPI, xformOp 없음) → `collisions` 메시(xformOp 없음)라
    baseLink 원점 = 메시 원점 = 뱅크 `origin_offset_z`(바닥 위 cup_big 0.0773·shaker 0.0921, scale 전) 이다. 별도 오프셋 없음.
  · 스포너는 뱅크 `scale` 을 참조 프림(루트)에 건다 → 균등 scale 이면 baseLink 프레임 좌표 = scale × 파일 좌표.
  · 충돌 프림 = baseLink 아래 `PhysicsCollisionAPI` 가 붙은 Mesh·Cylinder·Cube·Sphere(shaker 는 `bottom_plug` 원기둥 포함).
    메시 근사는 둘 다 `sdf` 라 충돌 형상 = 메시 그 자체다.

★표본: 면적 가중 무작위 후보(K×oversample) → 법선 인지 최원점 표본(FPS, 거리 = |Δp| + λ·|Δn|). 순수 FPS 는 얇은 벽
  (shaker 0.8 mm)의 안·밖 면을 한 점으로 뭉쳐 바깥 면 자리에 안쪽 법선을 남긴다 — 법선 항이 두 면을 따로 덮게 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_K = 512
DEFAULT_SEED = 0
#: 후보 배수 — FPS 가 고를 후보 수 = K × 이 값.
DEFAULT_OVERSAMPLE = 32
#: FPS 거리의 법선 항 가중 [m] — 반대 법선(|Δn| = 2)이 0.1 m 떨어진 것으로 보여 얇은 벽의 두 면이 따로 덮인다.
NORMAL_WEIGHT_M = 0.05
#: 원기둥·구 테셀레이션 분할 수(충돌 원시형 → 삼각형).
PRIM_SEGMENTS = 48


@dataclass(frozen=True)
class SurfaceCloud:
    points: np.ndarray    # (K,3) float32, baseLink 프레임 [m]
    normals: np.ndarray   # (K,3) float32, 바깥 단위 법선
    area: float           # 표본을 뽑은 충돌 표면 전체 넓이 [m²]
    spacing: float        # 덮개 반경 상한 — 후보 → 가장 가까운 표본 거리의 최대(FPS 특징 공간) [m]


# ---------------------------------------------------------------------------
# 순수 기하
# ---------------------------------------------------------------------------
def triangle_normals(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(M,3,3) → (단위 법선 (M,3), 넓이 (M,)). 법선 방향 = 감김(오른손) 방향."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    return n / np.maximum(a2, 1e-20)[:, None], 0.5 * a2


def sample_surface(tris: np.ndarray, k: int = DEFAULT_K, *, seed: int = DEFAULT_SEED,
                   oversample: int = DEFAULT_OVERSAMPLE, normal_weight: float = NORMAL_WEIGHT_M) -> SurfaceCloud:
    """삼각형 표면에서 K 점을 결정론적으로 뽑는다(면적 가중 후보 → 법선 인지 FPS)."""
    tris = np.asarray(tris, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or len(tris) == 0:
        raise ValueError(f"삼각형 배열은 (M,3,3) 이어야 한다 — got {tris.shape}")
    if k < 1:
        raise ValueError(f"k ≥ 1 — got {k}")
    nrm, area = triangle_normals(tris)
    keep = area > 0.0
    tris, nrm, area = tris[keep], nrm[keep], area[keep]
    total = float(area.sum())
    if total <= 0.0:
        raise ValueError("넓이 0 표면")
    rng = np.random.default_rng(seed)
    m = max(k * int(oversample), k)
    face = rng.choice(len(tris), size=m, p=area / total)
    u, v = rng.random(m), rng.random(m)
    flip = u + v > 1.0
    u, v = np.where(flip, 1.0 - u, u), np.where(flip, 1.0 - v, v)
    t = tris[face]
    cand = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    cand_n = nrm[face]
    feat = np.concatenate([cand, normal_weight * cand_n], axis=1)       # FPS 공간(유클리드 근사)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = int(rng.integers(m))
    dmin = np.linalg.norm(feat - feat[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(dmin))
        dmin = np.minimum(dmin, np.linalg.norm(feat - feat[sel[i]], axis=1))
    pts = cand[sel]
    # 덮개 반경 상한: FPS 특징 공간의 최대 최소거리 ≥ 위치 공간의 후보→최근접 표본 거리(특징 거리 ≥ 위치 거리)
    spacing = float(dmin.max())
    return SurfaceCloud(points=pts.astype(np.float32), normals=cand_n[sel].astype(np.float32),
                        area=total, spacing=spacing)


def point_triangle_distance(q: np.ndarray, tris: np.ndarray, chunk: int = 256) -> np.ndarray:
    """(Q,3) 점 → 삼각형 집합까지의 정확한 최소 거리 (Q,) — 테스트·검증용 브루트포스(Ericson, RTCD 5.1.5)."""
    q = np.asarray(q, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.float64)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac = b - a, c - a
    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        p = q[s:s + chunk, None, :]                                   # (C,1,3)
        ap, bp, cp = p - a, p - b, p - c
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
        v_in, w_in = vb / den, vc / den
        closest = a + v_in[..., None] * ab + w_in[..., None] * ac     # 내부
        # 변(ab·ac·bc)과 꼭짓점 영역 — 뒤에서부터 덮어쓴다
        with np.errstate(divide="ignore", invalid="ignore"):
            t_bc = np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0, 1.0)
            t_ac = np.clip(d2 / (d2 - d6), 0.0, 1.0)
            t_ab = np.clip(d1 / (d1 - d3), 0.0, 1.0)
        m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        closest = np.where(m_bc[..., None], b + t_bc[..., None] * (c - b), closest)
        m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        closest = np.where(m_ac[..., None], a + t_ac[..., None] * ac, closest)
        m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        closest = np.where(m_ab[..., None], a + t_ab[..., None] * ab, closest)
        closest = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, closest)
        closest = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, closest)
        closest = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, closest)
        out[s:s + chunk] = np.linalg.norm(p - closest, axis=-1).min(axis=1)
    return out


# ---------------------------------------------------------------------------
# 충돌 원시형 → 삼각형 (감김 = 바깥 법선)
# ---------------------------------------------------------------------------
def _axis_frame(axis: str) -> np.ndarray:
    """USD 원시형 축("X"|"Y"|"Z") → 로컬 z 를 그 축으로 보내는 회전 (3,3)."""
    if axis == "Z":
        return np.eye(3)
    if axis == "X":      # 열 = 로컬 x→Y · y→Z · z→X (순환 치환, det +1)
        return np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if axis == "Y":
        return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    raise ValueError(f"축은 X|Y|Z — got {axis!r}")


def cylinder_triangles(radius: float, height: float, axis: str = "Z", seg: int = PRIM_SEGMENTS) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * math.pi, seg, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * radius
    h = 0.5 * height
    tris = []
    for i in range(seg):
        j = (i + 1) % seg
        p0, p1 = ring[i], ring[j]
        b0, b1 = [p0[0], p0[1], -h], [p1[0], p1[1], -h]
        t0, t1 = [p0[0], p0[1], h], [p1[0], p1[1], h]
        tris += [[b0, b1, t1], [b0, t1, t0]]                           # 옆면(바깥)
        tris += [[[0, 0, h], t0, t1], [[0, 0, -h], b1, b0]]            # 윗·아랫 뚜껑
    t = np.asarray(tris, dtype=np.float64)
    return t @ _axis_frame(axis).T          # 행 = 점: p' = R p


def box_triangles(size: float) -> np.ndarray:
    h = 0.5 * size
    v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in faces:
        tris += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    t = np.asarray(tris, dtype=np.float64)
    nrm, _ = triangle_normals(t)
    ctr = t.mean(axis=1)
    bad = (nrm * ctr).sum(-1) < 0          # 원점에서 바깥을 보게 감김을 맞춘다
    t[bad] = t[bad][:, ::-1]
    return t


def sphere_triangles(radius: float, seg: int = PRIM_SEGMENTS) -> np.ndarray:
    th = np.linspace(0.0, math.pi, seg // 2 + 1)
    ph = np.linspace(0.0, 2.0 * math.pi, seg + 1)
    P = np.stack([np.outer(np.sin(th), np.cos(ph)), np.outer(np.sin(th), np.sin(ph)),
                  np.outer(np.cos(th), np.ones_like(ph))], axis=-1) * radius
    tris = []
    for i in range(len(th) - 1):
        for j in range(len(ph) - 1):
            a, b, c, d = P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]
            tris += [[a, b, c], [a, c, d]]
    t = np.asarray(tris, dtype=np.float64)
    nrm, area = triangle_normals(t)
    t = t[area > 1e-14]
    nrm, _ = triangle_normals(t)
    bad = (nrm * t.mean(axis=1)).sum(-1) < 0
    t[bad] = t[bad][:, ::-1]
    return t


# ---------------------------------------------------------------------------
# USD 읽기 (pxr)
# ---------------------------------------------------------------------------
def _apply(tris: np.ndarray, m) -> np.ndarray:
    """Gf.Matrix4d(행벡터 규약: p' = p · M) 를 (M,3,3) 에 적용."""
    M = np.array(m, dtype=np.float64)
    flat = tris.reshape(-1, 3)
    out = flat @ M[:3, :3] + M[3, :3]
    return out.reshape(tris.shape)


def read_collision_triangles(usd_path: str, rigid_body_name: str = "baseLink") -> np.ndarray:
    """USD 의 강체 아래 충돌 프림을 모두 삼각형으로 — 강체 프레임, scale 전 (M,3,3)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(usd_path)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    named = [p for p in bodies if p.GetName() == rigid_body_name]
    if len(named) != 1:
        raise RuntimeError(f"{usd_path}: RigidBodyAPI 프림 '{rigid_body_name}' 이 {len(named)}개 "
                           f"(있는 것 {[p.GetPath().pathString for p in bodies]})")
    body = named[0]
    cache = UsdGeom.XformCache()
    body_inv = cache.GetLocalToWorldTransform(body).GetInverse()
    out = []
    for p in Usd.PrimRange(body):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if p.HasAttribute("physics:collisionEnabled") and p.GetAttribute("physics:collisionEnabled").Get() is False:
            continue
        if p.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(p)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            cnt = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            tri_idx, off = [], 0
            for c in cnt:                                             # 다각형 → 부채꼴 삼각형
                f = idx[off:off + c]
                tri_idx += [(f[0], f[i], f[i + 1]) for i in range(1, c - 1)]
                off += c
            t = pts[np.asarray(tri_idx, dtype=np.int64)]
            if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
                t = t[:, ::-1]
        elif p.IsA(UsdGeom.Cylinder):
            g = UsdGeom.Cylinder(p)
            t = cylinder_triangles(float(g.GetRadiusAttr().Get()), float(g.GetHeightAttr().Get()),
                                   str(g.GetAxisAttr().Get()))
        elif p.IsA(UsdGeom.Cube):
            t = box_triangles(float(UsdGeom.Cube(p).GetSizeAttr().Get()))
        elif p.IsA(UsdGeom.Sphere):
            t = sphere_triangles(float(UsdGeom.Sphere(p).GetRadiusAttr().Get()))
        else:
            raise RuntimeError(f"{usd_path}: 지원하지 않는 충돌 프림 {p.GetPath()} ({p.GetTypeName()})")
        rel = cache.GetLocalToWorldTransform(p) * body_inv                # 프림 → 강체 프레임
        out.append(_apply(t, rel))
    if not out:
        raise RuntimeError(f"{usd_path}: '{rigid_body_name}' 아래 충돌 프림이 없다")
    return np.concatenate(out, axis=0)


def species_cloud(usd_path: str, scale: tuple, *, rigid_body_name: str = "baseLink", k: int = DEFAULT_K,
                  seed: int = DEFAULT_SEED) -> SurfaceCloud:
    """뱅크 한 종(usd·scale) → 강체 프레임 점군. scale 은 균등이어야 한다(비균등이면 법선 변환이 따로 필요)."""
    s = tuple(float(v) for v in scale)
    if len(s) != 3 or max(s) - min(s) > 1e-9:
        raise ValueError(f"균등 scale 만 지원 — got {scale}")
    tris = read_collision_triangles(usd_path, rigid_body_name) * s[0]
    return sample_surface(tris, k, seed=seed)


__all__ = ["DEFAULT_K", "DEFAULT_SEED", "SurfaceCloud", "sample_surface", "point_triangle_distance",
           "read_collision_triangles", "species_cloud", "cylinder_triangles", "box_triangles", "sphere_triangles",
           "triangle_normals"]
