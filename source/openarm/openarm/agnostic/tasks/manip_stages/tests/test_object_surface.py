"""표면 점군·최근접 질의 — 해석 대조(Isaac 불요, pxr 필요: `/usr/bin/python3` 또는 Isaac python).

    cd hdgp && PYTHONPATH=source/openarm /usr/bin/python3 -m pytest source/openarm/openarm/agnostic/tasks/manip_stages/tests -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from openarm.agnostic.modules import object_bank as ob
from openarm.agnostic.tasks.manip_stages import object_surface as S
from openarm.agnostic.tasks.manip_stages.surface_query import nearest_surface, quat_rotate

pxr = pytest.importorskip("pxr")

_CUP = ob.spec_by_id("cup_big_s100")
_SHAKER = ob.spec_by_id("shaker_closed")


def _winding(q: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """일반화 감김수(van Oosterom–Strackee 입체각 합 / 4π) — 닫힌 메시 안 1 · 밖 0."""
    a, b, c = (tris[None, :, i, :] - q[:, None, :] for i in range(3))
    la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
    num = np.einsum("qtj,qtj->qt", a, np.cross(b, c))
    den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
    return 2.0 * np.arctan2(num, den).sum(axis=1) / (4.0 * np.pi)


@pytest.fixture(scope="module")
def cup_tris():
    return S.read_collision_triangles(_CUP.usd_path, _CUP.rigid_body_name)


def test_collision_geometry_frame_matches_the_bank_origin_convention(cup_tris):
    # baseLink 원점 = 뱅크 원점: 바닥 z = −origin_offset_z, 윗면 z = rim_z (scale 1, 09.01 pxr 실측값과 같은 규약)
    z = cup_tris[..., 2]
    assert z.min() == pytest.approx(-_CUP.origin_offset_z, abs=5e-4)
    assert z.max() == pytest.approx(_CUP.rim_z, abs=5e-4)
    # 축 ≈ z: xy 중심이 원점 근처, 외벽 반경 = 뱅크 outer_radius
    r = np.hypot(cup_tris[..., 0], cup_tris[..., 1])
    assert r.max() == pytest.approx(_CUP.outer_radius_m, abs=1.5e-3)


def test_shaker_includes_its_bottom_plug_collider():
    tris = S.read_collision_triangles(_SHAKER.usd_path, _SHAKER.rigid_body_name)
    # 메시 1218 삼각형 + 원기둥(48 분할 × 4) — plug 이 빠지면 바닥 구멍이 비어 보인다
    assert len(tris) == 1218 + 4 * S.PRIM_SEGMENTS
    assert tris[..., 2].min() == pytest.approx(-_SHAKER.origin_offset_z, abs=2.5e-3)


def test_cloud_is_deterministic_on_the_mesh_with_outward_normals(cup_tris):
    a = S.sample_surface(cup_tris, 512, seed=0)
    b = S.sample_surface(cup_tris, 512, seed=0)
    assert a.points.shape == (512, 3) and a.normals.shape == (512, 3)
    np.testing.assert_array_equal(a.points, b.points)
    assert S.point_triangle_distance(a.points, cup_tris).max() < 1e-6          # 점은 메시 위
    np.testing.assert_allclose(np.linalg.norm(a.normals, axis=1), 1.0, atol=1e-5)
    # 바깥 법선: 법선 쪽으로 1 mm 나간 점은 물체 밖(감김수 0), 반대로 1 mm 들어간 점은 벽 속(감김수 1)
    eps = 1e-3
    assert (_winding(a.points + eps * a.normals, cup_tris) < 0.5).mean() > 0.97
    assert (_winding(a.points - eps * a.normals, cup_tris) > 0.5).mean() > 0.97
    # 균일 덮개: 1 cm 대 간격(cup_big s100 FPS 14 mm)
    assert a.spacing < 0.02


@pytest.mark.parametrize("spec_id", ["cup_big_s100", "cup_big_s130", "shaker_closed"])
def test_cloud_distance_matches_brute_force_mesh_distance(spec_id):
    spec = ob.spec_by_id(spec_id)
    tris = S.read_collision_triangles(spec.usd_path, spec.rigid_body_name) * spec.scale[0]
    cloud = S.sample_surface(tris, S.DEFAULT_K)
    rng = np.random.default_rng(3)
    lo, hi = tris.reshape(-1, 3).min(0) - 0.12, tris.reshape(-1, 3).max(0) + 0.12
    q = rng.uniform(lo, hi, (2000, 3))
    true = S.point_triangle_distance(q, tris)
    est, _, _ = nearest_surface(torch.tensor(q[None], dtype=torch.float32), torch.zeros(1, 3),
                                torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor(cloud.points[None]),
                                torch.tensor(cloud.normals[None]), torch.zeros(1, dtype=torch.long))
    err = est[0].double().numpy() - true
    assert err.min() > -1e-4, "표본점은 표면 위라 참거리보다 작을 수 없다"
    far = (true > 0.01) & (true < 0.15)
    assert err[far].mean() < 3e-3 and np.percentile(err[far], 95) < 5e-3, (err[far].mean(), np.percentile(err[far], 95))
    near = true <= 0.01                     # 표면 바로 옆: 표본 간격만큼 과대(최대 ≈ 덮개 반경)
    assert err[near].max() < cloud.spacing + 1e-3


def test_backface_skip_reports_the_outer_side_of_a_thin_wall():
    # 두께 1 mm 판: 윗면(+z 법선)과 아랫면(−z 법선). 위 5 mm 의 질의는 윗면 표본을 골라야 한다.
    top = np.array([[0.0, 0.0, 0.0005], [0.1, 0.0, 0.0005], [0.0, 0.1, 0.0005]])
    pts = torch.tensor([[[0.0, 0.0, 0.0005], [0.001, 0.0, -0.0005]]])     # 윗면 점 · (조금 더 가까운) 아랫면 점
    nrm = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    q = torch.tensor([[[0.0012, 0.0, 0.009]]])
    d, direction, normal = nearest_surface(q, torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), pts, nrm,
                                           torch.zeros(1, dtype=torch.long), backface_tol=0.005)
    assert normal[0, 0].tolist() == [0.0, 0.0, 1.0]
    assert (direction[0, 0] * normal[0, 0]).sum() < -0.9
    assert d[0, 0] == pytest.approx(math.dist([0.0012, 0.0, 0.009], top[0]), abs=1e-6)


def _quat(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * axis])


def test_nearest_point_follows_a_rotated_and_translated_object(cup_tris):
    cloud = S.sample_surface(cup_tris, 256, seed=1)
    pts, nrm = torch.tensor(cloud.points[None]), torch.tensor(cloud.normals[None])
    rng = np.random.default_rng(0)
    qb = torch.tensor(rng.uniform(-0.15, 0.15, (1, 16, 3)), dtype=torch.float32)       # 물체 프레임 질의
    d0, dir0, n0 = nearest_surface(qb, torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]), pts, nrm,
                                   torch.zeros(1, dtype=torch.long))
    quat = torch.tensor(_quat([0.3, -0.5, 0.8], 1.1)[None], dtype=torch.float32)
    pos = torch.tensor([[0.25, 0.15, 0.3]])
    qw = quat_rotate(quat, qb) + pos[:, None, :]                                      # 같은 질의를 월드로
    d1, dir1, n1 = nearest_surface(qw, pos, quat, pts, nrm, torch.zeros(1, dtype=torch.long))
    torch.testing.assert_close(d1, d0, atol=1e-5, rtol=0)
    torch.testing.assert_close(dir1, quat_rotate(quat, dir0), atol=1e-5, rtol=0)
    torch.testing.assert_close(n1, quat_rotate(quat, n0), atol=1e-5, rtol=0)
    # 표면점 = 질의 + d·dir 은 월드로 옮긴 점군의 한 점이다
    surf = qw + d1[..., None] * dir1
    cloud_w = quat_rotate(quat, pts) + pos[:, None, :]
    gap = (surf[0][:, None, :] - cloud_w[0][None, :, :]).norm(dim=-1).min(dim=1).values
    assert gap.max() < 1e-5


def test_species_are_indexed_per_env_and_chunking_is_exact(cup_tris):
    big = S.sample_surface(cup_tris * 1.3, 128, seed=0)
    small = S.sample_surface(cup_tris * 0.85, 128, seed=0)
    pts = torch.tensor(np.stack([small.points, big.points]))
    nrm = torch.tensor(np.stack([small.normals, big.normals]))
    n = 7
    q = torch.tensor(np.random.default_rng(2).uniform(-0.2, 0.2, (n, 16, 3)), dtype=torch.float32)
    species = torch.tensor([i % 2 for i in range(n)])
    pos, quat = torch.zeros(n, 3), torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4)
    full = nearest_surface(q, pos, quat, pts, nrm, species, chunk=1024)
    chunked = nearest_surface(q, pos, quat, pts, nrm, species, chunk=3)
    for a, b in zip(full, chunked):
        torch.testing.assert_close(a, b)
    ref = (q[1][:, None, :] - pts[1][None]).norm(dim=-1).min(dim=1).values                      # env 1 = 종 1(큰 컵)
    assert (full[0][1] - ref).abs().max() < 0.03


def test_primitive_tessellations_are_closed_and_outward():
    for tris, vol in ((S.cylinder_triangles(0.03, 0.004, "X"), math.pi * 0.03 ** 2 * 0.004),
                      (S.box_triangles(0.02), 0.02 ** 3), (S.sphere_triangles(0.01), 4 / 3 * math.pi * 0.01 ** 3)):
        signed = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0
        assert signed == pytest.approx(vol, rel=0.03)
    cyl = S.cylinder_triangles(0.03, 0.004, "X")
    assert np.ptp(cyl[..., 0]) == pytest.approx(0.004) and np.ptp(cyl[..., 1]) == pytest.approx(0.06, rel=1e-3)
