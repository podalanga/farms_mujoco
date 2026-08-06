from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _tet_vol_and_wcentroid(apex, a, b, c):
    """Vectorized signed volume*6 and volume-weighted centroid for a batch
    of (apex, a, b, c) tetrahedra. a, b, c: (K, 3) arrays. apex: (3,).
    Returns (vol: (K,), weighted_centroid_sum: (3,), vol_sum: scalar)
    packed as (total_vol, weighted_centroid) already reduced -- callers
    just want the reduction, not per-tet values.
    """
    ab = b - apex
    ac = c - apex
    aa = a - apex
    vol6 = np.einsum('ij,ij->i', aa, np.cross(ab, ac))
    vol = vol6 / 6.0
    tet_centroid = (apex + a + b + c) / 4.0
    return vol, tet_centroid


def submerged_volume_and_centroid(vertices_world, faces, water_z, plane_point=None):
    """Same contract as the original: volume (float) and centroid (3,).

    Vectorized: no Python-level loop over triangles. Triangles are bucketed
    by how many of their 3 vertices are wet (0 / 1 / 2 / 3), and within the
    1-wet and 2-wet buckets, further bucketed by *which* vertex is the odd
    one out (3 sub-cases each). That gives at most 8 numpy calls total,
    independent of triangle count, instead of one Python iteration per
    triangle.
    """
    vertices_world = np.asarray(vertices_world, dtype=float)
    faces = np.asarray(faces)
    depths = water_z - vertices_world[:, 2]

    if np.all(depths <= 0.0):
        return 0.0, vertices_world.mean(axis=0)

    if plane_point is None:
        mean_xy = vertices_world[:, :2].mean(axis=0)
        plane_point = np.array([mean_xy[0], mean_xy[1], water_z])
    apex = np.asarray(plane_point, dtype=float)

    tri_verts = vertices_world[faces]        # (M, 3, 3)
    tri_d = depths[faces]                    # (M, 3)
    wet = tri_d >= 0.0                       # (M, 3) bool
    n_wet = wet.sum(axis=1)                  # (M,)

    total_vol = 0.0
    weighted_centroid = np.zeros(3)

    # --- fully wet triangles: no clipping needed ---
    mask3 = n_wet == 3
    if np.any(mask3):
        v = tri_verts[mask3]
        vol, tc = _tet_vol_and_wcentroid(apex, v[:, 0], v[:, 1], v[:, 2])
        total_vol += vol.sum()
        weighted_centroid += (vol[:, None] * tc).sum(axis=0)

    # --- partial triangles (1 or 2 wet vertices) ---
    # For each fixed "special vertex" index i in {0,1,2}, gather the
    # triangles where i is the lone wet vertex (n_wet==1) or the lone dry
    # vertex (n_wet==2), and do the edge-crossing + volume math vectorized
    # across that whole group at once.
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3

        # n_wet == 1, wet vertex is i -> one small wet triangle (vi, pj, pk)
        m1 = (n_wet == 1) & wet[:, i]
        if np.any(m1):
            vi = tri_verts[m1, i]
            vj = tri_verts[m1, j]
            vk = tri_verts[m1, k]
            di, dj, dk = tri_d[m1, i], tri_d[m1, j], tri_d[m1, k]
            pj = _cross_point(vi, vj, di, dj)
            pk = _cross_point(vi, vk, di, dk)
            vol, tc = _tet_vol_and_wcentroid(apex, vi, pj, pk)
            total_vol += vol.sum()
            weighted_centroid += (vol[:, None] * tc).sum(axis=0)

        # n_wet == 2, dry vertex is i -> wet quad (pj, vj, vk, pk) -> 2 tris
        m2 = (n_wet == 2) & (~wet[:, i])
        if np.any(m2):
            vi = tri_verts[m2, i]
            vj = tri_verts[m2, j]
            vk = tri_verts[m2, k]
            di, dj, dk = tri_d[m2, i], tri_d[m2, j], tri_d[m2, k]
            pj = _cross_point(vi, vj, di, dj)
            pk = _cross_point(vi, vk, di, dk)
            vol_a, tc_a = _tet_vol_and_wcentroid(apex, pj, vj, vk)
            vol_b, tc_b = _tet_vol_and_wcentroid(apex, pj, vk, pk)
            total_vol += vol_a.sum() + vol_b.sum()
            weighted_centroid += (vol_a[:, None] * tc_a).sum(axis=0)
            weighted_centroid += (vol_b[:, None] * tc_b).sum(axis=0)

    if abs(total_vol) < _EPS:
        return 0.0, vertices_world.mean(axis=0)

    centroid = weighted_centroid / total_vol
    return total_vol, centroid


def _cross_point(vi, vj, di, dj):
    """Vectorized edge-crossing point where depth hits 0, for batches of
    edges (vi -> vj) with depths (di, dj). Matches the original's
    denom-near-zero -> t=0.5 fallback and t clamped to [0,1]."""
    denom = di - dj
    small = np.abs(denom) < _EPS
    # avoid div-by-zero warnings; result is overwritten by `small` branch
    safe_denom = np.where(small, 1.0, denom)
    t = np.where(small, 0.5, di / safe_denom)
    t = np.clip(t, 0.0, 1.0)
    return vi + t[:, None] * (vj - vi)


def submerged_volume_and_centroid_fast(cache, world_pos, world_rot, water_z):
    """Identical cheap-reject wrapper to the original; only the fallback
    exact-clip path now calls the vectorized function above."""
    world_pos = np.asarray(world_pos, dtype=float)
    centroid_world = world_rot @ cache.centroid_local + world_pos

    depth = water_z - centroid_world[2]
    if depth < -cache.bound_radius:
        return 0.0, centroid_world
    if depth > cache.bound_radius:
        return cache.volume, centroid_world

    verts_world = cache.verts_local @ world_rot.T + world_pos
    return submerged_volume_and_centroid(verts_world, cache.faces, water_z)


def buoyancy_force(volume, rho_fluid, g=9.81):
    return rho_fluid * g * volume
