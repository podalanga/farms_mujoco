"""Load-time construction of the centre-of-buoyancy lookup tables used by
cob_lut.CobLut (see cob_lut.pyx for the runtime).

For every direction n of an octahedral grid on the sphere and every
threshold t in [t_min(n), t_max(n)], the tables store the volume V and
first moment M of the part of the link {x : n.x <= t} (link frame). The
moment is stored as its component along n and its part orthogonal to n,
which interpolate well across directions (the orthogonal part vanishes
for symmetric shapes).

- Links whose geoms do not overlap: tables are sampled with the exact
  kernels of cob.pyx, so only interpolation errors remain.
- Links with overlapping geoms: the union is voxelised, and for every
  direction the voxel projections are binned and prefix-summed, each
  voxel spreading uniformly over its projected width.

Tables are cached in memory and on disk by geometry hash, so episode
resets and new processes reuse them. The disk cache lives next to the
running script (e.g. `experiments/<name>/cob_lut_cache`), see cache_dirs().
"""

import os
import sys
import tempfile
import warnings

import numpy as np

from .cob import CobModel
from .cob_build import (
    SPHERE, ELLIPSOID, CYLINDER, CAPSULE, POLYHEDRON,
    assemble_cob_geometry, estimate_overlap, link_geoms, link_grid,
    geom_inside_grid, geoms_key, primitive_volume, mesh_volume_centroid,
)
from .cob_lut import CobLut

LUT_VERSION = 2
CACHE_ENV = 'FARMS_COB_LUT_CACHE'
CACHE_DIRNAME = 'cob_lut_cache'
_CACHE = {}
_WARNED = set()


# Disk cache location
# ---------------------------------------------------------------------------

def _script_dir():
    """Directory of the running script, or the working directory

    `python -m` modules (e.g. the macOS mjpython re-exec of farms_sim) and
    console scripts (inside the environment, sys.prefix) use the working
    directory instead.
    """
    main = sys.modules.get('__main__')
    script = getattr(main, '__file__', None)
    if script and getattr(main, '__spec__', None) is None:
        script_dir = os.path.dirname(os.path.realpath(script))
        prefixes = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix)}
        if not any(
                os.path.commonpath([script_dir, prefix]) == prefix
                for prefix in prefixes
        ):
            return script_dir
    return os.getcwd()


def _user_cache_dir():
    """$XDG_CACHE_HOME or ~/.cache, None without a home (e.g. Docker --user)"""
    base = os.environ.get('XDG_CACHE_HOME')
    if not base:
        home = os.path.expanduser('~')
        base = os.path.join(home, '.cache') if home != '~' else None
    return os.path.join(base, 'farms_mujoco', 'cob_lut') if base else None


def cache_dirs(cache=None):
    """Disk cache directories, in order of preference

    cache: None for the default, False to disable the disk cache, or a
    directory (relative paths are relative to the running script).
    Default: `$FARMS_COB_LUT_CACHE` if set, else `<script dir>/cob_lut_cache`,
    e.g. `experiments/<name>/cob_lut_cache` for `experiments/<name>/run_sim.py`.
    The user cache and the temporary directory follow as fallbacks for
    read-only locations (e.g. a read-only Docker volume).
    """
    if cache is False:
        return []
    if cache is None or cache is True:
        cache = os.environ.get(CACHE_ENV) or CACHE_DIRNAME
    cache = os.path.expanduser(str(cache))
    if not os.path.isabs(cache):
        cache = os.path.join(_script_dir(), cache)
    dirs = []
    for directory in (
            cache, _user_cache_dir(),
            os.path.join(tempfile.gettempdir(), 'farms_mujoco_cob_lut'),
    ):
        if directory and os.path.abspath(directory) not in dirs:
            dirs.append(os.path.abspath(directory))
    return dirs


def _load(dirs, key):
    """Cached LUT from the first directory holding a valid file

    A table found in a fallback directory (e.g. the user cache of an older
    version) is also copied to the preferred one, so that each experiment
    folder ends up with its own tables.
    """
    for index, directory in enumerate(dirs):
        path = os.path.join(directory, f'{key}.npz')
        if not os.path.isfile(path):
            continue
        try:
            with np.load(path) as data:
                lut = (
                    data['table'], data['t_range'], float(data['volume']),
                    data['centroid'],
                )
        except (OSError, KeyError, ValueError, EOFError):
            continue
        if index > 0:
            _save(dirs[:1], key, lut)
        return lut
    return None


def _save(dirs, key, lut):
    """Atomic write (safe across parallel processes) to the first writable dir"""
    for directory in dirs:
        path = os.path.join(directory, f'{key}.npz')
        tmp = f'{path}.{os.getpid()}.tmp.npz'
        try:
            os.makedirs(directory, exist_ok=True)
            np.savez(tmp, table=lut[0], t_range=lut[1], volume=lut[2], centroid=lut[3])
            os.replace(tmp, path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass
            continue
        if directory != dirs[0] and dirs[0] not in _WARNED:
            _WARNED.add(dirs[0])
            warnings.warn(
                f'CoB LUT cache {dirs[0]} is not writable, using {directory}'
                f' instead (set {CACHE_ENV} or cob_lut_cache to choose)',
                stacklevel=3,
            )
        return path
    return None


def oct_decode(u, v):
    """Octahedral coordinates in [0, 1]^2 to unit vectors (vectorised)"""
    px = 2*np.asarray(u, dtype=float) - 1
    py = 2*np.asarray(v, dtype=float) - 1
    pz = 1 - np.abs(px) - np.abs(py)
    neg = pz < 0
    sx = np.where(px >= 0, 1.0, -1.0)
    sy = np.where(py >= 0, 1.0, -1.0)
    qx = np.where(neg, (1 - np.abs(py))*sx, px)
    qy = np.where(neg, (1 - np.abs(px))*sy, py)
    n = np.stack([qx, qy, pz], axis=-1)
    return n/np.linalg.norm(n, axis=-1, keepdims=True)


def grid_directions(n_dir):
    """Unit directions of the octahedral grid, [n_dir*n_dir, 3]"""
    uu, vv = np.meshgrid(
        np.linspace(0, 1, n_dir), np.linspace(0, 1, n_dir), indexing='ij',
    )
    return oct_decode(uu.reshape(-1), vv.reshape(-1))


def rotation_with_last_row(n):
    """A rotation matrix R with R^T z = n (last row equal to n)"""
    helper = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    a = np.cross(helper, n)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return np.array([a, b, n])


def decompose_moments(table, directions):
    """[V, Mx, My, Mz] -> [V, M.n, M - (M.n) n] per direction"""
    moments = table[..., 1:]
    along = np.einsum('dki,di->dk', moments, directions)
    perp = moments - along[..., None]*directions[:, None, :]
    return np.concatenate([table[..., :1], along[..., None], perp], axis=-1)


# ---------------------------------------------------------------------------
# Exact tables (non-overlapping geoms)
# ---------------------------------------------------------------------------

def geom_support(geom, n):
    """max over the geom of n.x (link frame)"""
    ng = geom.rot.T @ n
    size = geom.size
    if geom.kind == SPHERE:
        extent = size[0]
    elif geom.kind == ELLIPSOID:
        extent = np.linalg.norm(size[:3]*ng)
    elif geom.kind == CYLINDER:
        extent = size[1]*abs(ng[2]) + size[0]*np.hypot(ng[0], ng[1])
    elif geom.kind == CAPSULE:
        extent = size[1]*abs(ng[2]) + size[0]
    else:
        extent = np.max(geom.tris.reshape(-1, 3) @ ng)
    return float(n @ geom.pos + extent)


def build_exact_table(geoms, n_dir, n_depth):
    """Tables sampled with the exact kernels"""
    model = CobModel(assemble_cob_geometry(
        [geom.__class__(**{**geom.__dict__, 'geom_id': i, 'link': 0})
         for i, geom in enumerate(geoms)],
        n_links=1,
    ))
    directions = grid_directions(n_dir)
    table = np.zeros([len(directions), n_depth, 4])
    t_range = np.zeros([len(directions), 2])
    for d_i, n in enumerate(directions):
        rot = rotation_with_last_row(n)
        xpos = np.array([rot @ geom.pos for geom in geoms])
        xmat = np.array([(rot @ geom.rot).reshape(-1) for geom in geoms])
        t_min = -max(geom_support(geom, -n) for geom in geoms)
        t_max = max(geom_support(geom, n) for geom in geoms)
        for k, t in enumerate(np.linspace(t_min, t_max, n_depth)):
            out = model.links_submerged(xpos, xmat, [t])[0]
            table[d_i, k, 0] = out[0]
            table[d_i, k, 1:] = rot.T @ out[1:]
        t_range[d_i] = t_min, t_max
    return table, t_range


# ---------------------------------------------------------------------------
# Voxel tables (overlapping geoms)
# ---------------------------------------------------------------------------

def voxelize_link(geoms, voxels=96):
    """Voxel centres (link frame) of the union of a link's geoms, the
    voxel half size, and the voxel volume scaled so that the union volume
    is the exact sum of volumes times the voxel union fraction"""
    axes, step = link_grid(geoms, cells=voxels)
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    union = np.zeros(len(points), dtype=bool)
    sum_counts = 0
    exact_sum = 0.0
    for geom in geoms:
        inside = geom_inside_grid(geom, axes)
        union |= inside
        sum_counts += np.count_nonzero(inside)
        exact_sum += (
            mesh_volume_centroid(geom.tris)[0]
            if geom.kind == POLYHEDRON
            else primitive_volume(geom.kind, geom.size)
        )
    n_union = np.count_nonzero(union)
    dv = exact_sum*n_union/max(sum_counts, 1)/max(n_union, 1)
    return points[union], np.full(3, 0.5*step), dv


def build_voxel_table(points, half_voxel, dv, n_dir, n_depth, oversample=8,
                      chunk=32):
    """Tables from binned voxel projections, vectorised over batches of
    directions. Each voxel spreads uniformly over its projected width
    [p - w, p + w], so V(t) = dv/(2w) (S(t + w) - S(t - w)) with
    S(tau) = sum_{p <= tau} (tau - p), evaluated from prefix sums over
    fine bins (and likewise for the first moments)."""
    directions = grid_directions(n_dir)
    n_total = len(directions)
    table = np.zeros([n_total, n_depth, 4])
    t_range = np.zeros([n_total, 2])
    n_bins = oversample*n_depth
    n_points = len(points)
    for start in range(0, n_total, chunk):
        dirs = directions[start:start+chunk]
        n_d = len(dirs)
        proj = points @ dirs.T  # [n_points, n_d]
        w = np.abs(dirs) @ half_voxel
        t_min = proj.min(axis=0) - w
        t_max = proj.max(axis=0) + w
        scale = n_bins/(t_max - t_min)
        idx = np.clip(((proj - t_min)*scale).astype(np.int64), 0, n_bins-1)
        flat = (idx + n_bins*np.arange(n_d)).reshape(-1)
        size = n_d*n_bins

        def binned(values):
            return np.bincount(flat, values, size).reshape(n_d, n_bins)

        # Per bin: count, sum(p), sum(x), sum(x*p)
        count = np.bincount(flat, minlength=size).reshape(n_d, n_bins)
        sum_p = binned(proj.reshape(-1))
        sum_x = np.stack([
            binned(np.repeat(points[:, i], n_d)) for i in range(3)
        ], -1)
        sum_xp = np.stack([
            binned((points[:, i:i+1]*proj).reshape(-1)) for i in range(3)
        ], -1)
        zeros = np.zeros([n_d, 1])
        cum_c = np.concatenate([zeros, np.cumsum(count, axis=1)], axis=1)
        cum_p = np.concatenate([zeros, np.cumsum(sum_p, axis=1)], axis=1)
        cum_x = np.concatenate([np.zeros([n_d, 1, 3]), np.cumsum(sum_x, axis=1)], axis=1)
        cum_xp = np.concatenate([np.zeros([n_d, 1, 3]), np.cumsum(sum_xp, axis=1)], axis=1)
        for j in range(n_d):
            edges = np.linspace(t_min[j], t_max[j], n_bins + 1)
            s0 = edges*cum_c[j] - cum_p[j]
            s1 = edges[:, None]*cum_x[j] - cum_xp[j]
            t = np.linspace(t_min[j], t_max[j], n_depth)
            values = []
            for tau in (t + w[j], t - w[j]):
                extra = np.maximum(tau - edges[-1], 0)
                values.append((
                    np.interp(tau, edges, s0) + extra*n_points,
                    np.stack([
                        np.interp(tau, edges, s1[:, i]) + extra*cum_x[j, -1, i]
                        for i in range(3)
                    ], -1),
                ))
            d_i = start + j
            table[d_i, :, 0] = dv*(values[0][0] - values[1][0])/(2*w[j])
            table[d_i, :, 1:] = dv*(values[0][1] - values[1][1])/(2*w[j])
            table[d_i, 0] = 0
            table[d_i, -1, 0] = dv*n_points
            table[d_i, -1, 1:] = dv*cum_x[j, -1]
            t_range[d_i] = t_min[j], t_max[j]
    return table, t_range


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def _build_link_lut(geoms, n_dir, n_depth, voxels, overlap_tolerance):
    """Uncached LUT construction for one link"""
    overlap = estimate_overlap(geoms) if len(geoms) > 1 else 1.0
    if overlap >= 1 - overlap_tolerance:
        table, t_range = build_exact_table(geoms, n_dir, n_depth)
    else:
        points, half_voxel, dv = voxelize_link(geoms, voxels=voxels)
        table, t_range = build_voxel_table(points, half_voxel, dv, n_dir, n_depth)
    full = table[:, -1]
    volume = float(np.mean(full[:, 0]))
    centroid = np.mean(full[:, 1:], axis=0)/volume if volume > 0 else np.zeros(3)
    table = decompose_moments(table, grid_directions(n_dir))
    return table, t_range, volume, centroid


def build_link_lut(geoms, resolution=(32, 64), voxels=96, overlap_tolerance=1e-3,
                   disk_cache=True, cache_dir=None):
    """(table, t_range, volume, centroid) for one link, cached by geometry

    disk_cache: False keeps the tables in memory only.
    cache_dir: disk cache directory, see cache_dirs().
    """
    n_dir, n_depth = resolution
    if not geoms:
        return (
            np.zeros([n_dir*n_dir, n_depth, 5]), np.zeros([n_dir*n_dir, 2]),
            0.0, np.zeros(3),
        )
    key = geoms_key(
        [geom.__class__(**{**geom.__dict__, 'geom_id': 0, 'link': 0}) for geom in geoms],
        LUT_VERSION, *resolution, voxels, overlap_tolerance,
    )
    if key in _CACHE:
        return _CACHE[key]
    dirs = cache_dirs(cache_dir if disk_cache else False)
    lut = _load(dirs, key)
    if lut is None:
        lut = _build_link_lut(geoms, n_dir, n_depth, voxels, overlap_tolerance)
        _save(dirs, key, lut)
    _CACHE[key] = lut
    return lut


def build_cob_lut(model, body_ids, geom_group=2, meters=1.0,
                  resolution=(32, 64), voxels=96, cache_dir=None):
    """CobLut for the given MuJoCo bodies"""
    geoms = link_geoms(model, body_ids, geom_group=geom_group, meters=meters)
    luts = [
        build_link_lut(
            [geom for geom in geoms if geom.link == link_i],
            resolution=resolution, voxels=voxels, cache_dir=cache_dir,
        )
        for link_i in range(len(body_ids))
    ]
    return CobLut(body_ids, luts, resolution)
