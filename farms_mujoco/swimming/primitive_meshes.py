from __future__ import annotations

import numpy as np


def box_mesh(half_extents):
    """Axis-aligned box, local frame, centered at origin"""
    hx, hy, hz = half_extents
    v = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    f = np.array([
        [0, 3, 2], [0, 2, 1],  # bottom (z-)
        [4, 5, 6], [4, 6, 7],  # top (z+)
        [0, 1, 5], [0, 5, 4],  # y-
        [1, 2, 6], [1, 6, 5],  # x+
        [2, 3, 7], [2, 7, 6],  # y+
        [3, 0, 4], [3, 4, 7],  # x-
    ])
    return v, f


def uv_sphere_mesh(radius, n_lat=12, n_lon=24):
    """UV sphere, local frame, centered at origin"""
    verts = [np.array([0, 0, radius])]
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat  # 0 (pole) .. pi (pole)
        z = radius * np.cos(theta)
        r = radius * np.sin(theta)
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            verts.append(np.array([r * np.cos(phi), r * np.sin(phi), z]))
    verts.append(np.array([0, 0, -radius]))
    verts = np.array(verts)

    faces = []
    top = 0
    bottom = len(verts) - 1
    ring0 = 1
    # top cap
    for j in range(n_lon):
        faces.append([top, ring0 + j, ring0 + (j + 1) % n_lon])
    # middle quads
    for i in range(n_lat - 2):
        r0 = 1 + i * n_lon
        r1 = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            a, b = r0 + j, r0 + (j + 1) % n_lon
            c, d = r1 + j, r1 + (j + 1) % n_lon
            faces.append([a, c, d])
            faces.append([a, d, b])
    '''
    I took the quad to be 

    d ----- c ring n+1
    | \     |
    |   \   |
    b ----- a ring n
    '''
    # bottom cap
    last_ring = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append([bottom, last_ring + (j + 1) % n_lon, last_ring + j])
    return verts, np.array(faces)


def cylinder_mesh(radius, half_height, n_seg=32):
    """Cylinder, local frame, axis along z, centered at origin"""
    verts = []
    for z in (-half_height, half_height):
        for j in range(n_seg):
            phi = 2 * np.pi * j / n_seg
            verts.append([radius * np.cos(phi), radius * np.sin(phi), z])
    verts.append([0, 0, -half_height])  # bottom center
    verts.append([0, 0, half_height])   # top center
    verts = np.array(verts)

    bottom_c = len(verts) - 2 #bottom index
    top_c = len(verts) - 1 #top index
    faces = []
    for j in range(n_seg):
        jn = (j + 1) % n_seg
        b0, b1 = j, jn
        t0, t1 = n_seg + j, n_seg + jn
        # side wall (two tris), outward normal
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
        # bottom cap (normal -z)
        faces.append([bottom_c, b1, b0])
        # top cap (normal +z)
        faces.append([top_c, t0, t1])
    return verts, np.array(faces)


def capsule_mesh(radius, half_length, n_lat=8, n_lon=24):
    """Capsule, local frame, axis along z: cylinder of half_length plus a
    hemisphere of `radius` capping each end (MuJoCo capsule convention:
    `size = (radius, half_length)`, total length along axis = 2*half_length
    + 2*radius).
    """
    verts = []
    faces = []

    def hemisphere(z_center, sign, n_lat_h, n_lon):
        local_v = []
        for i in range(1, n_lat_h):          # Exclude the final pole iteration
            theta = (np.pi / 2) * i / n_lat_h
            z = z_center + sign * radius * np.sin(theta)
            r = radius * np.cos(theta)
            for j in range(n_lon):
                phi = 2 * np.pi * j / n_lon
                local_v.append([r * np.cos(phi), r * np.sin(phi), z])
        local_v.append([0, 0, z_center + sign * radius])
        return local_v

    # equator rings (shared with cylinder ends)
    eq_top = []
    eq_bot = []
    for j in range(n_lon):
        phi = 2 * np.pi * j / n_lon
        eq_top.append([radius * np.cos(phi), radius * np.sin(phi), half_length])
        eq_bot.append([radius * np.cos(phi), radius * np.sin(phi), -half_length])

    idx = {}
    verts.extend(eq_top)
    idx['eq_top'] = list(range(0, n_lon))
    verts.extend(eq_bot)
    idx['eq_bot'] = list(range(n_lon, 2 * n_lon))

    # NOTE: cylindrical side wall between eq_top/eq_bot is built below with
    # the same winding fix as cylinder_mesh (b0,b1,t1 / b0,t1,t0).

    top_h = hemisphere(half_length, +1, n_lat, n_lon)
    base = len(verts)
    verts.extend(top_h)
    idx['top_rings'] = [list(range(base + k * n_lon, base + (k + 1) * n_lon))
                         for k in range(n_lat - 1)]
    idx['top_pole'] = base + (n_lat - 1) * n_lon

    bot_h = hemisphere(-half_length, -1, n_lat, n_lon)
    base = len(verts)
    verts.extend(bot_h)
    idx['bot_rings'] = [list(range(base + k * n_lon, base + (k + 1) * n_lon))
                         for k in range(n_lat - 1)]
    idx['bot_pole'] = base + (n_lat - 1) * n_lon

    verts = np.array(verts)

    # cylindrical side wall
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        b0, b1 = idx['eq_bot'][j], idx['eq_bot'][jn]
        t0, t1 = idx['eq_top'][j], idx['eq_top'][jn]
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    def cap_rings(equator_ring, rings, pole, outward_top):
        rings_full = [equator_ring] + rings
        for k in range(len(rings_full) - 1):
            r0, r1 = rings_full[k], rings_full[k + 1]
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                a, b = r0[j], r0[jn]
                c, d = r1[j], r1[jn]
                if outward_top:
                    faces.append([a, b, d])
                    faces.append([a, d, c])
                else:
                    faces.append([a, d, b])
                    faces.append([a, c, d])
        last = rings_full[-1]
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            if outward_top:
                faces.append([pole, last[j], last[jn]])
            else:
                faces.append([pole, last[jn], last[j]])

    cap_rings(idx['eq_top'], idx['top_rings'], idx['top_pole'], outward_top=True)
    cap_rings(idx['eq_bot'], idx['bot_rings'], idx['bot_pole'], outward_top=False)

    return verts, np.array(faces)


PRIMITIVE_BUILDERS = {
    'box': box_mesh,
    'sphere': lambda size: uv_sphere_mesh(size[0]),
    'cylinder': lambda size: cylinder_mesh(size[0], size[1]),
    'capsule': lambda size: capsule_mesh(size[0], size[1]),
}


def build_local_mesh(geom_type: str, size):
    """size follows MuJoCo geom_size convention for the given geom_type:
    box: half-extents (hx,hy,hz); sphere: (radius,); cylinder/capsule:
    (radius, half_length)
    """
    if geom_type == 'box':
        return box_mesh(size)
    if geom_type not in PRIMITIVE_BUILDERS:
        raise ValueError(f'No primitive mesh builder for geom_type={geom_type!r}')
    return PRIMITIVE_BUILDERS[geom_type](size)


class PrimitiveCache:
    """Everything needed to compute submerged volume/centroid for one
    collision primitive, precomputed once at model-load time in the
    primitive's own LOCAL frame:

    - verts_local, faces : the closed triangle mesh (for the rare
      near-surface exact-clip fallback)
    - volume             : full (unclipped) mesh volume
    - centroid_local      : full (unclipped) mesh centroid, local frame
    - bound_radius        : max distance from centroid_local to any
      vertex -- i.e. radius of the tightest sphere centered on the
      centroid that contains the whole primitive. Used as a cheap
      per-step "is this primitive anywhere near the surface" test.
    """
    __slots__ = ('verts_local', 'faces', 'volume', 'centroid_local', 'bound_radius')

    def __init__(self, verts_local, faces, volume, centroid_local, bound_radius):
        self.verts_local = verts_local
        self.faces = faces
        self.volume = volume
        self.centroid_local = centroid_local
        self.bound_radius = bound_radius


def build_primitive_cache(geom_type: str, size) -> 'PrimitiveCache':
    """Build and cache the local-frame mesh plus its full volume,
    centroid, and bounding radius. Call once per (geom_type, size) at
    model-load time -- never in the per-step hot path.
    """
    # Local import to avoid a hard circular dependency at module load.
    from .cob_core import submerged_volume_and_centroid

    verts, faces = build_local_mesh(geom_type, size)
    # Any water_z strictly above every vertex -> whole mesh counts as
    # "submerged" -> submerged_volume_and_centroid returns the exact
    # full-mesh volume/centroid (this is exercised by the n_wet==3 path,
    # i.e. no clipping happens, only a plain closed-mesh divergence sum).
    water_z_above_all = float(verts[:, 2].max()) + 1.0
    volume, centroid_local = submerged_volume_and_centroid(
        verts, faces, water_z_above_all,
    )
    bound_radius = float(np.max(np.linalg.norm(verts - centroid_local, axis=1)))
    return PrimitiveCache(
        verts_local=verts,
        faces=faces,
        volume=volume,
        centroid_local=centroid_local,
        bound_radius=bound_radius,
    )
