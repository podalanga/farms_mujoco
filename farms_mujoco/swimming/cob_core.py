from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _clip_triangle(v0, v1, v2, d0, d1, d2):
    """Clip one triangle to the wet half-space (depth >= 0).

    Returns a list of (a, b, c) vertex triples covering the wet portion
    only (0, 1, or 2 triangles). Depths are used purely to find the
    plane-crossing points via linear interpolation along edges.
    """
    verts = (v0, v1, v2)
    d = (d0, d1, d2)
    wet = [x >= 0.0 for x in d]
    n_wet = sum(wet)

    if n_wet == 0:
        return []
    if n_wet == 3:
        return [(v0, v1, v2)]

    def cross_point(i, j):
        # point where edge i->j crosses the plane (depth == 0)
        di, dj = d[i], d[j]
        denom = di - dj
        if abs(denom) < _EPS:
            t = 0.5
        else:
            t = di / denom
        t = min(1.0, max(0.0, t))
        return verts[i] + t * (verts[j] - verts[i])

    if n_wet == 1:
        # one wet vertex -> small wet triangle near it
        i = wet.index(True)
        j, k = (i + 1) % 3, (i + 2) % 3
        pj = cross_point(i, j)
        pk = cross_point(i, k)
        return [(verts[i], pj, pk)]

    # n_wet == 2: dry vertex is the odd one out -> wet quad -> 2 triangles
    i = wet.index(False)
    j, k = (i + 1) % 3, (i + 2) % 3  # both wet
    pj = cross_point(i, j)  # crossing on edge i-j
    pk = cross_point(i, k)  # crossing on edge i-k
    vj, vk = verts[j], verts[k]
    return [(pj, vj, vk), (pj, vk, pk)]


def submerged_volume_and_centroid(vertices_world, faces, water_z, plane_point=None):
    """
    Parameters
    ----------
    vertices_world : (N,3) array, mesh vertices already in world frame
    faces          : (M,3) int array, outward-wound triangle indices
    water_z         : float, world-frame height of the flat water surface
    plane_point     : optional (3,) point on the water plane to use as the
                       tetrahedron apex. Any point with z == water_z works;
                       defaults to the mesh's own vertex centroid projected
                       onto the plane, which keeps the tetrahedron volumes
                       well-conditioned (small) for typical body sizes.

    Returns
    -------
    volume   : float, submerged volume (0.0 if fully dry)
    centroid : (3,) array, center of buoyancy in world frame
               (returns the mesh centroid, unclipped, if volume == 0)
    """
    vertices_world = np.asarray(vertices_world, dtype=float)
    depths = water_z - vertices_world[:, 2]

    # Fast paths: fully dry / fully submerged whole mesh.
    if np.all(depths <= 0.0):
        return 0.0, vertices_world.mean(axis=0)

    if plane_point is None:
        mean_xy = vertices_world[:, :2].mean(axis=0)
        plane_point = np.array([mean_xy[0], mean_xy[1], water_z])
    apex = np.asarray(plane_point, dtype=float)

    total_vol = 0.0
    weighted_centroid = np.zeros(3)

    for tri in faces:
        i0, i1, i2 = tri
        v0, v1, v2 = vertices_world[i0], vertices_world[i1], vertices_world[i2]
        d0, d1, d2 = depths[i0], depths[i1], depths[i2]

        if d0 <= 0 and d1 <= 0 and d2 <= 0:
            continue

        wet_tris = [(v0, v1, v2)] if (d0 >= 0 and d1 >= 0 and d2 >= 0) \
            else _clip_triangle(v0, v1, v2, d0, d1, d2)

        for a, b, c in wet_tris:
            vol6 = np.dot(a - apex, np.cross(b - apex, c - apex))
            vol = vol6 / 6.0
            if abs(vol) < _EPS:
                continue
            tet_centroid = (apex + a + b + c) / 4.0
            total_vol += vol
            weighted_centroid += vol * tet_centroid

    if abs(total_vol) < _EPS:
        return 0.0, vertices_world.mean(axis=0)

    centroid = weighted_centroid / total_vol
    return total_vol, centroid


def submerged_volume_and_centroid_fast(cache, world_pos, world_rot, water_z):
    """cheap-reject wrapper around submerged_volume_and_centroid,
    using the (volume, centroid_local, bound_radius) precomputed once per
    primitive in primitive_meshes.PrimitiveCache.

    Transforms only what's needed for the current classification:
      - centroid_world is always needed (one 3-vector transform).
      - If the whole primitive is provably on one side of the water plane
        (|centroid_world.z - water_z| > bound_radius), we skip clipping
        entirely: fully dry -> (0.0, centroid_world); fully submerged ->
        (cache.volume, centroid_world) since volume is rotation/translation
        invariant.
      - Only when the bounding sphere actually straddles the plane do we
        pay for transforming every vertex and running the real clip.

    Parameters
    ----------
    cache      : primitive_meshes.PrimitiveCache
    world_pos  : (3,) world-frame position of the primitive's local origin
    world_rot  : (3,3) world-frame rotation matrix of the primitive
    water_z    : float, world-frame water height

    Returns
    -------
    volume, centroid_world -- same contract as submerged_volume_and_centroid
    """







    world_pos = np.asarray(world_pos, dtype=float)
    centroid_world = world_rot @ cache.centroid_local + world_pos


    depth = water_z - centroid_world[2]
    if depth < -cache.bound_radius:
        return 0.0, centroid_world
    if depth > cache.bound_radius:
        return cache.volume, centroid_world

    # Straddles the water plane -- fall back to the exact per-triangle clip.
    verts_world = cache.verts_local @ world_rot.T + world_pos
    return submerged_volume_and_centroid(verts_world, cache.faces, water_z)


def buoyancy_force(volume, rho_fluid, g=9.81):
    """Upward buoyancy force magnitude (world +z). Apply at the centroid
    returned by submerged_volume_and_centroid."""
    return rho_fluid * g * volume
