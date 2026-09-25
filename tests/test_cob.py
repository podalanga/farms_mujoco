"""Accuracy tests for the centre-of-buoyancy kernels (cob.pyx)"""

import numpy as np
import pytest

from farms_mujoco.swimming import cob
from farms_mujoco.swimming.cob_build import (
    SPHERE, ELLIPSOID, CYLINDER, CAPSULE, POLYHEDRON,
    GeomInfo, assemble_cob_geometry, box_triangles, primitive_volume,
    mesh_volume_centroid, points_inside,
)

RNG = np.random.default_rng(1234)


# ---------------------------------------------------------------------------
# Reference implementation: brute force vectorised triangle clipping
# ---------------------------------------------------------------------------

def clip_reference(tris_world, h):
    """Submerged (V, centroid) of a closed mesh below z <= h, computed by
    clipping every triangle (no BVH, no moments)"""
    apex = np.array([*tris_world.reshape(-1, 3)[:, :2].mean(axis=0), h])
    depth = h - tris_world[:, :, 2]
    wet = depth >= 0
    n_wet = wet.sum(axis=1)
    pieces = [tris_world[n_wet == 3]]
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for count, alone_wet in ((1, True), (2, False)):
            mask = (n_wet == count) & (wet[:, i] == alone_wet)
            if not np.any(mask):
                continue
            t = tris_world[mask]
            d = depth[mask]
            a, b, c = t[:, i], t[:, j], t[:, k]
            q0 = a + (d[:, i]/(d[:, i] - d[:, j]))[:, None]*(b - a)
            q1 = a + (d[:, i]/(d[:, i] - d[:, k]))[:, None]*(c - a)
            if alone_wet:
                pieces.append(np.stack([a, q0, q1], axis=1))
            else:
                pieces.append(np.stack([q0, b, c], axis=1))
                pieces.append(np.stack([q0, c, q1], axis=1))
    tris = np.concatenate(pieces) - apex
    vol6 = np.einsum('ij,ij->i', tris[:, 0], np.cross(tris[:, 1], tris[:, 2]))
    volume = vol6.sum()/6
    if abs(volume) < 1e-300:
        return 0.0, apex
    centroid = (vol6[:, None]*tris.sum(axis=1)).sum(axis=0)/(24*volume) + apex
    return volume, centroid


def revolution_mesh(z, r, n_lon):
    """Closed mesh of a solid of revolution around z. The profile must
    start and end on the axis (r=0)"""
    phi = 2*np.pi*np.arange(n_lon)/n_lon
    rings = [
        np.stack([ri*np.cos(phi), ri*np.sin(phi), np.full(n_lon, zi)], axis=1)
        for zi, ri in zip(z[1:-1], r[1:-1])
    ]
    bottom, top = np.array([0, 0, z[0]]), np.array([0, 0, z[-1]])
    tris = []
    nxt = np.roll(np.arange(n_lon), -1)
    ring = rings[0]
    tris.append(np.stack([np.repeat(bottom[None], n_lon, 0), ring[nxt], ring], 1))
    for ring0, ring1 in zip(rings[:-1], rings[1:]):
        tris.append(np.stack([ring0, ring0[nxt], ring1[nxt]], 1))
        tris.append(np.stack([ring0, ring1[nxt], ring1], 1))
    ring = rings[-1]
    tris.append(np.stack([np.repeat(top[None], n_lon, 0), ring, ring[nxt]], 1))
    return np.concatenate(tris)


def fine_mesh(kind, size, n=1024):
    """Finely tessellated reference mesh of an analytic primitive"""
    if kind in (SPHERE, ELLIPSOID):
        theta = np.linspace(np.pi, 0, n//2+1)
        tris = revolution_mesh(np.cos(theta), np.sin(theta), n)
        scale = [size[0]]*3 if kind == SPHERE else size[:3]
        return tris*np.asarray(scale)
    radius, half = size[0], size[1]
    if kind == CYLINDER:
        z = np.array([-half, -half, half, half])
        r = np.array([0, radius, radius, 0])
        z = np.concatenate([[-half], np.linspace(-half, half, 5), [half]])
        r = np.concatenate([[0], np.full(5, radius), [0]])
        return revolution_mesh(z, r, n)
    theta = np.linspace(np.pi, np.pi/2, n//4+1)
    z_bot = -half + radius*np.cos(theta)
    r_bot = radius*np.sin(theta)
    return revolution_mesh(
        np.concatenate([z_bot, -z_bot[::-1]]),
        np.concatenate([r_bot, r_bot[::-1]]),
        n,
    )


def random_rotation():
    """Uniform random rotation matrix"""
    q = RNG.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def model_for(kind, size, tris=None):
    """CobModel with a single geom"""
    geometry = assemble_cob_geometry([GeomInfo(
        geom_id=0, link=0, kind=kind, size=np.pad(size, (0, 3-len(size))),
        pos=np.zeros(3), rot=np.eye(3), tris=tris,
    )], n_links=1)
    return cob.CobModel(geometry)


def special_rotations():
    """Axis-aligned, nearly horizontal and nearly vertical orientations"""
    rotations = [np.eye(3)]
    for angle in (np.pi/2, np.pi/2 + 1e-9, np.pi/2 - 1e-4, 1e-10, 1e-5, 0.3):
        c, s = np.cos(angle), np.sin(angle)
        rotations.append(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]))
    return rotations


PRIMITIVES = [
    (SPHERE, [0.1]),
    (ELLIPSOID, [0.3, 0.1, 0.05]),
    (CYLINDER, [0.05, 0.2]),
    (CYLINDER, [0.2, 0.01]),
    (CAPSULE, [0.05, 0.2]),
    (CAPSULE, [0.1, 0.02]),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kind,size', PRIMITIVES)
def test_primitive_vs_fine_mesh(kind, size):
    """Analytic kernels agree with a finely tessellated mesh"""
    size = np.array(size, dtype=float)
    local = fine_mesh(kind, size)
    mesh_volume, _ = mesh_volume_centroid(local)
    # Tessellation shrinks the volume, compare relative to its own volume
    true_volume = primitive_volume(kind, size)
    assert abs(mesh_volume - true_volume)/true_volume < 2e-5
    extent = np.max(np.abs(local))
    model = model_for(kind, size)
    rotations = special_rotations() + [random_rotation() for _ in range(8)]
    for rot in rotations:
        center = RNG.normal(size=3)
        world = local @ rot.T + center
        for h in center[2] + extent*np.linspace(-1.05, 1.05, 9):
            volume, centroid = model.submerged(0, rot, center, h)
            ref_volume, ref_centroid = clip_reference(world, h)
            assert volume == pytest.approx(
                ref_volume*true_volume/mesh_volume,
                rel=5e-5, abs=5e-5*true_volume,
            )
            if ref_volume > 1e-3*true_volume:
                assert np.allclose(centroid, ref_centroid, atol=5e-5*extent)


@pytest.mark.parametrize('kind,size', PRIMITIVES)
def test_primitive_limits(kind, size):
    """Fully dry, fully wet and half submerged symmetric cases"""
    size = np.array(size, dtype=float)
    center = np.array([0.1, -0.2, 0.3])
    volume = primitive_volume(kind, size)
    for rot in special_rotations():
        v, _ = cob.primitive_submerged(kind, size, rot, center, center[2] - 10)
        assert v == pytest.approx(0, abs=1e-15)
        v, vc = cob.primitive_submerged(kind, size, rot, center, center[2] + 10)
        assert v == pytest.approx(volume, rel=1e-12)
        assert np.allclose(vc/v, center, atol=1e-12)
        # Point symmetric solids are half submerged at their centre
        v, _ = cob.primitive_submerged(kind, size, rot, center, center[2])
        assert v == pytest.approx(0.5*volume, rel=1e-9)


def test_box_axis_aligned_exact():
    """Axis aligned box: volume = area * wet height"""
    half = np.array([0.3, 0.2, 0.1])
    model = model_for(POLYHEDRON, half, box_triangles(half))
    for h in np.linspace(-0.1, 0.1, 11):
        volume, centroid = model.submerged(0, np.eye(3), np.zeros(3), h)
        wet = h + 0.1
        assert volume == pytest.approx(4*half[0]*half[1]*wet, abs=1e-15)
        if wet > 0:
            assert np.allclose(centroid, [0, 0, -0.1 + wet/2], atol=1e-14)


def torus_mesh(major=0.2, minor=0.05, n_major=64, n_minor=32):
    """Non-convex closed torus mesh"""
    u = 2*np.pi*np.arange(n_major)/n_major
    v = 2*np.pi*np.arange(n_minor)/n_minor
    uu, vv = np.meshgrid(u, v, indexing='ij')
    verts = np.stack([
        (major + minor*np.cos(vv))*np.cos(uu),
        (major + minor*np.cos(vv))*np.sin(uu),
        minor*np.sin(vv),
    ], axis=-1)
    i, j = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing='ij')
    i1, j1 = (i + 1) % n_major, (j + 1) % n_minor
    a, b, c, d = verts[i, j], verts[i1, j], verts[i1, j1], verts[i, j1]
    tris = np.concatenate([
        np.stack([a, b, c], -2).reshape(-1, 3, 3),
        np.stack([a, c, d], -2).reshape(-1, 3, 3),
    ])
    if mesh_volume_centroid(tris)[0] < 0:
        tris = tris[:, [0, 2, 1]]
    return tris


@pytest.mark.parametrize('name', ['box', 'torus', 'hull'])
def test_polyhedron_bvh_vs_brute_force(name):
    """BVH moments give the same result as clipping every triangle"""
    if name == 'box':
        tris = box_triangles([0.3, 0.1, 0.05])
    elif name == 'torus':
        tris = torus_mesh()
    else:
        from farms_mujoco.swimming.cob_build import convex_hull_triangles
        tris = convex_hull_triangles(RNG.normal(size=[300, 3]))
    volume, _ = mesh_volume_centroid(tris)
    extent = np.max(np.linalg.norm(tris.reshape(-1, 3), axis=1))
    model = model_for(POLYHEDRON, np.zeros(3), tris)
    for _ in range(20):
        rot = random_rotation()
        center = RNG.normal(size=3)
        world = tris @ rot.T + center
        for h in center[2] + extent*RNG.uniform(-1.1, 1.1, 5):
            v, c = model.submerged(0, rot, center, h)
            ref_v, ref_c = clip_reference(world, h)
            assert v == pytest.approx(ref_v, rel=1e-9, abs=1e-12*volume)
            if ref_v > 1e-6*volume:
                assert np.allclose(c, ref_c, atol=1e-9*extent)


def test_polyhedron_monte_carlo():
    """Independent check of the polyhedron kernel with Monte Carlo"""
    tris = torus_mesh()
    model = model_for(POLYHEDRON, np.zeros(3), tris)
    rot = random_rotation()
    points = RNG.uniform(-0.26, 0.26, size=[400000, 3])
    inside = points_inside(POLYHEDRON, None, points, tris)
    world = points @ rot.T
    box_volume = 0.52**3
    for h in (-0.1, 0.0, 0.05):
        wet = inside & (world[:, 2] <= h)
        mc_volume = box_volume*wet.mean()
        v, c = model.submerged(0, rot, np.zeros(3), h)
        assert v == pytest.approx(mc_volume, rel=0.02)
        assert np.allclose(c, world[wet].mean(axis=0), atol=5e-3)


def test_links_aggregation_and_units():
    """compute() sums geoms per link and handles unit scaling"""
    geoms = [
        GeomInfo(0, 0, SPHERE, np.array([0.1, 0, 0]), np.zeros(3), np.eye(3)),
        GeomInfo(1, 0, CYLINDER, np.array([0.05, 0.1, 0]), np.zeros(3), np.eye(3)),
        GeomInfo(2, 1, POLYHEDRON, np.zeros(3), np.zeros(3), np.eye(3),
                 box_triangles([0.1, 0.1, 0.1])),
    ]
    model = cob.CobModel(assemble_cob_geometry(geoms, n_links=2))
    meters = 10.0
    xpos = np.array([[0, 0, 0], [1, 0, 0], [0, 2, -0.05]])*meters
    xmat = np.tile(np.eye(3).reshape(1, 9), (3, 1))
    out = model.links_submerged(xpos, xmat, [0.0, 0.0], 1/meters)
    sphere_v = 2/3*np.pi*0.1**3
    cyl_v = np.pi*0.05**2*0.1
    assert out[0, 0] == pytest.approx(sphere_v + cyl_v, rel=1e-12)
    assert out[1, 0] == pytest.approx(0.2*0.2*0.15, rel=1e-12)
    assert out[1, 1:3]/out[1, 0] == pytest.approx([0, 2], abs=1e-12)
