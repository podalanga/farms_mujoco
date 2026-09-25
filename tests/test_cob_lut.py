"""Accuracy tests for the centre-of-buoyancy lookup tables (cob_lut.pyx)"""

import numpy as np
import pytest

from farms_mujoco.swimming import cob, cob_lut, cob_lut_build
from farms_mujoco.swimming.cob_build import (
    SPHERE, CAPSULE, CYLINDER, POLYHEDRON,
    GeomInfo, assemble_cob_geometry, box_triangles,
)
from test_cob import torus_mesh


def geom(kind, size, pos=(0, 0, 0), rot=None, tris=None):
    """GeomInfo helper"""
    return GeomInfo(
        0, 0, kind, np.pad(np.array(size, float), (0, 3-len(size))),
        np.array(pos, float), np.eye(3) if rot is None else rot, tris,
    )


def lut_errors(geoms, resolution=(32, 64), n_poses=20, seed=0, voxel=False):
    """Max volume error (fraction of the full volume) and first moment
    error (fraction of full volume times size, i.e. relative buoyancy
    torque error) of the LUT against the exact kernels"""
    rng = np.random.default_rng(seed)
    lut = cob_lut.CobLut([0], [cob_lut_build.build_link_lut(
        geoms, resolution=resolution, disk_cache=False,
        overlap_tolerance=-1 if voxel else 1e-3,
    )], resolution)
    exact = cob.CobModel(assemble_cob_geometry(geoms, n_links=1))
    full = exact.links_submerged(
        np.array([g.pos for g in geoms]),
        np.array([g.rot.reshape(-1) for g in geoms]), [1e3],
    )[0, 0]
    size = max(np.max(np.abs(cob_lut_build.voxelize_link(geoms)[0])), 1e-9)
    err_v = err_c = 0.0
    for _ in range(n_poses):
        quat = rng.normal(size=4)
        w, x, y, z = quat/np.linalg.norm(quat)
        rot = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        pos = rng.normal(size=3)
        xpos = np.array([rot @ g.pos + pos for g in geoms])
        xmat = np.array([(rot @ g.rot).reshape(-1) for g in geoms])
        for h in pos[2] + size*rng.uniform(-1, 1, 5):
            ref = exact.links_submerged(xpos, xmat, [h])[0]
            volume, centroid = lut.submerged(0, rot, pos, h)
            err_v = max(err_v, abs(volume - ref[0])/full)
            # Moment (i.e. buoyancy torque) error relative to V_full*size
            moment = volume*(centroid - pos) if volume > 0 else np.zeros(3)
            ref_moment = ref[1:] - ref[0]*pos
            err_c = max(err_c, np.linalg.norm(moment - ref_moment)/(full*size))
    return err_v, err_c


def test_octahedral_roundtrip():
    """Octahedral encoding inverts the decoding"""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        u, v = cob_lut.oct_encode(n)
        assert np.allclose(cob_lut_build.oct_decode(u, v), n, atol=1e-12)


@pytest.mark.parametrize('voxel', [False, True])
@pytest.mark.parametrize('name', ['sphere', 'box', 'capsule', 'torus'])
def test_lut_accuracy(name, voxel):
    """LUT agrees with the exact kernels within a few percent, whether it
    is sampled with the exact kernels or built from voxels"""
    geoms = {
        'sphere': [geom(SPHERE, [0.1])],
        'box': [geom(POLYHEDRON, [0], tris=box_triangles([0.2, 0.05, 0.03]))],
        'capsule': [geom(CAPSULE, [0.03, 0.1], pos=[0.02, 0, 0])],
        'torus': [geom(POLYHEDRON, [0], tris=torus_mesh())],
    }[name]
    err_v, err_c = lut_errors(geoms, n_poses=40, voxel=voxel)
    print(f'{name} (voxel={voxel}): volume {err_v:.2e}, centroid {err_c:.2e}')
    tolerance = 0.03 if voxel else 0.015
    assert err_v < tolerance
    assert err_c < tolerance


def test_lut_overlap_union():
    """Overlapping geoms are counted once by the LUT"""
    geoms = [geom(SPHERE, [0.1]), geom(SPHERE, [0.1])]
    lut = cob_lut.CobLut([0], [cob_lut_build.build_link_lut(geoms, disk_cache=False)], (32, 64))
    volume, _ = lut.submerged(0, np.eye(3), np.zeros(3), 10.0)
    assert volume == pytest.approx(4/3*np.pi*0.1**3, rel=1e-6)


def test_lut_link_with_cylinder_and_box():
    """Typical robot link made of a cylinder and a box (overlapping)"""
    geoms = [
        geom(CYLINDER, [0.0175, 0.0326]),
        geom(POLYHEDRON, [0], pos=[0, 0.02, 0],
             tris=box_triangles([0.0175, 0.0325, 0.0225])),
    ]
    lut = cob_lut.CobLut([0], [cob_lut_build.build_link_lut(geoms, disk_cache=False)], (32, 64))
    exact = assemble_cob_geometry(geoms, n_links=1, report_overlap=True)
    volume, _ = lut.submerged(0, np.eye(3), np.zeros(3), 10.0)
    # Union is smaller than the sum, and consistent with the overlap estimate
    assert volume < exact.link_volume[0]
    assert volume == pytest.approx(
        exact.link_volume[0]*exact.link_overlap[0], rel=0.05,
    )
