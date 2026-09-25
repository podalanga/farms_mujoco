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

Tables are cached in memory and on disk (~/.cache/farms_mujoco/cob_lut)
by geometry hash, so episode resets and new processes reuse them.
"""

import os

import numpy as np

from .cob import CobModel
from .cob_build import (
    SPHERE, ELLIPSOID, CYLINDER, CAPSULE, POLYHEDRON,
    assemble_cob_geometry, estimate_overlap, link_geoms, link_grid,
    geom_inside_grid, geoms_key, primitive_volume, mesh_volume_centroid,
)
from .cob_lut import CobLut

LUT_VERSION = 2
CACHE_DIR = os.path.join(
    os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
    'farms_mujoco', 'cob_lut',
)
_CACHE = {}


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
                   disk_cache=True):
    """(table, t_range, volume, centroid) for one link, cached by geometry"""
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
    path = os.path.join(CACHE_DIR, f'{key}.npz')
    if disk_cache and os.path.isfile(path):
        try:
            data = np.load(path)
            _CACHE[key] = (
                data['table'], data['t_range'], float(data['volume']),
                data['centroid'],
            )
            return _CACHE[key]
        except (OSError, KeyError, ValueError):
            pass
    lut = _build_link_lut(geoms, n_dir, n_depth, voxels, overlap_tolerance)
    _CACHE[key] = lut
    if disk_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = f'{path}.{os.getpid()}.tmp.npz'
            np.savez(tmp, table=lut[0], t_range=lut[1], volume=lut[2], centroid=lut[3])
            os.replace(tmp, path)
        except OSError:
            pass
    return lut


def build_cob_lut(model, body_ids, geom_group=2, meters=1.0,
                  resolution=(32, 64), voxels=96):
    """CobLut for the given MuJoCo bodies"""
    geoms = link_geoms(model, body_ids, geom_group=geom_group, meters=meters)
    luts = [
        build_link_lut(
            [geom for geom in geoms if geom.link == link_i],
            resolution=resolution, voxels=voxels,
        )
        for link_i in range(len(body_ids))
    ]
    return CobLut(body_ids, luts, resolution)
