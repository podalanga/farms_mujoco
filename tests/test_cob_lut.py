"""Accuracy tests for the centre-of-buoyancy lookup tables (cob_lut.pyx)"""

import os

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


def test_lut_disk_cache_next_to_script(tmp_path, monkeypatch):
    """Default cache is cob_lut_cache/ next to the running script"""
    script = tmp_path / 'experiment' / 'run_sim.py'
    script.parent.mkdir()
    script.write_text('')
    monkeypatch.delenv(cob_lut_build.CACHE_ENV, raising=False)
    main = cob_lut_build.sys.modules['__main__']
    monkeypatch.setattr(main, '__file__', str(script), raising=False)
    monkeypatch.setattr(main, '__spec__', None, raising=False)
    dirs = cob_lut_build.cache_dirs()
    assert dirs[0] == str(script.parent / cob_lut_build.CACHE_DIRNAME)
    assert cob_lut_build.cache_dirs('luts')[0] == str(script.parent / 'luts')
    assert cob_lut_build.cache_dirs(False) == []
    # `python -m` runs (spec set) use the working directory
    monkeypatch.setattr(main, '__spec__', object())
    monkeypatch.chdir(tmp_path)
    assert cob_lut_build.cache_dirs()[0] == str(tmp_path / cob_lut_build.CACHE_DIRNAME)
    monkeypatch.setenv(cob_lut_build.CACHE_ENV, str(tmp_path / 'env'))
    assert cob_lut_build.cache_dirs()[0] == str(tmp_path / 'env')


def test_lut_disk_cache_roundtrip_and_fallback(tmp_path, monkeypatch):
    """Tables are written once, reloaded, and fall back when read-only"""
    geoms = [geom(SPHERE, [0.1])]
    monkeypatch.setattr(cob_lut_build, '_CACHE', {})
    lut = cob_lut_build.build_link_lut(geoms, resolution=(8, 16), cache_dir=tmp_path / 'a')
    files = list((tmp_path / 'a').glob('*.npz'))
    assert len(files) == 1 and not list((tmp_path / 'a').glob('*.tmp.npz'))
    monkeypatch.setattr(cob_lut_build, '_CACHE', {})
    monkeypatch.setattr(cob_lut_build, '_build_link_lut', None)  # must not rebuild
    loaded = cob_lut_build.build_link_lut(geoms, resolution=(8, 16), cache_dir=tmp_path / 'a')
    assert np.array_equal(loaded[0], lut[0])
    # Read-only primary directory: next writable fallback is used
    monkeypatch.setattr(cob_lut_build, 'cache_dirs', lambda cache: [
        str(tmp_path / 'ro' / 'x'), str(tmp_path / 'b')])
    (tmp_path / 'ro').mkdir(mode=0o500)
    if os.access(tmp_path / 'ro', os.W_OK):  # root ignores permissions
        pytest.skip('directory permissions not enforced')
    with pytest.warns(UserWarning, match='not writable'):
        path = cob_lut_build._save(cob_lut_build.cache_dirs(None), 'key', lut)
    assert path == str(tmp_path / 'b' / 'key.npz')


def test_lut_disk_cache_copies_fallback(tmp_path, monkeypatch):
    """A table found in a fallback directory is copied to the preferred one"""
    geoms = [geom(SPHERE, [0.1])]
    monkeypatch.setattr(cob_lut_build, '_CACHE', {})
    cob_lut_build.build_link_lut(geoms, resolution=(8, 16), cache_dir=tmp_path / 'old')
    monkeypatch.setattr(cob_lut_build, '_CACHE', {})
    monkeypatch.setattr(cob_lut_build, '_build_link_lut', None)  # must not rebuild
    monkeypatch.setattr(cob_lut_build, 'cache_dirs', lambda cache: [
        str(tmp_path / 'new'), str(tmp_path / 'old')])
    cob_lut_build.build_link_lut(geoms, resolution=(8, 16))
    assert len(list((tmp_path / 'new').glob('*.npz'))) == 1
