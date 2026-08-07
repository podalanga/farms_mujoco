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


def uv_sphere_mesh(radius, n_lat=16, n_lon=32):
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
    r'''
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


def cylinder_mesh(radius, half_height, n_seg=20):
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


def capsule_mesh(radius, half_length, n_lat=6, n_lon=16):
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


# NOTE on the resolutions above (n_lat/n_lon/n_seg defaults):
# These meshes only exist to integrate volume/centroid for buoyancy --
# never rendered -- so they don't need render-quality tessellation.
# Benchmarked on a 0.05m sphere: the original 30x60 UV sphere (3480
# faces) gives ~0.46% volume error vs the exact analytic sphere volume;
# dropping to 16x32 (960 faces, used above) still holds under ~1.6%
# error while cutting per-step clip cost roughly 3x, because the clip
# math scales with face count. Below ~n_lat=12 error climbs fast (UV
# spheres' polar triangle fans systematically underestimate volume
# there), so don't go lower without checking the tradeoff yourself --
# e.g. copy the loop in test_coarse.py-style: build a cache, compare
# `cache.volume` to `4/3*pi*r**3`, and time the hot call.
PRIMITIVE_BUILDERS = {
    'box': box_mesh,
    'sphere': lambda size, res: uv_sphere_mesh(size[0], n_lat=res.sphere_n_lat, n_lon=res.sphere_n_lon),
    'cylinder': lambda size, res: cylinder_mesh(size[0], size[1], n_seg=res.cylinder_n_seg),
    'capsule': lambda size, res: capsule_mesh(size[0], size[1], n_lat=res.capsule_n_lat, n_lon=res.capsule_n_lon),
}


class MeshResolution:
    """Tessellation knobs for the mesh-clip fallback path, gathered in
    one place so callers (ultimately SwimmingHandler in drag.pyx) can
    expose them as configurable options instead of the fixed defaults
    baked into uv_sphere_mesh/cylinder_mesh/capsule_mesh.

    Defaults match the benchmarked values in the comment above this
    class's old location (~1.6% volume error, ~3x fewer faces than the
    original 30x60 sphere). Only matters for shapes actually going
    through the mesh-clip path -- e.g. with the default 'analytic'
    cob_method, sphere_n_lat/sphere_n_lon are irrelevant for spheres
    (closed-form is used instead) unless you force cob_method='mesh'.
    """
    __slots__ = (
        'sphere_n_lat', 'sphere_n_lon',
        'cylinder_n_seg',
        'capsule_n_lat', 'capsule_n_lon',
    )

    def __init__(
        self,
        sphere_n_lat=16, sphere_n_lon=32,
        cylinder_n_seg=20,
        capsule_n_lat=6, capsule_n_lon=16,
    ):
        self.sphere_n_lat = sphere_n_lat
        self.sphere_n_lon = sphere_n_lon
        self.cylinder_n_seg = cylinder_n_seg
        self.capsule_n_lat = capsule_n_lat
        self.capsule_n_lon = capsule_n_lon


DEFAULT_MESH_RESOLUTION = MeshResolution()

# geom_type -> integer code for PrimitiveCache.analytic_kind. 0 always
# means "no closed form known, use the mesh-clip path". Add an entry
# here (and a matching branch in buoyancy.py's dispatcher + a `_cy`
# solver in buoyancy_cy.pyx) when a new analytic shape is added to
# analytic_shapes.py.
_ANALYTIC_KIND_CODES = {
    'sphere': 1,
}


def build_local_mesh(geom_type: str, size, mesh_resolution: 'MeshResolution' = None):
    """size follows MuJoCo geom_size convention for the given geom_type:
    box: half-extents (hx,hy,hz); sphere: (radius,); cylinder/capsule:
    (radius, half_length)
    """
    if mesh_resolution is None:
        mesh_resolution = DEFAULT_MESH_RESOLUTION
    if geom_type == 'box':
        return box_mesh(size)
    if geom_type not in PRIMITIVE_BUILDERS:
        raise ValueError(f'No primitive mesh builder for geom_type={geom_type!r}')
    return PRIMITIVE_BUILDERS[geom_type](size, mesh_resolution)


class PrimitiveCache:
    """Everything needed to compute submerged volume/centroid for one
    collision primitive, precomputed once at model-load time in the
    primitive's own LOCAL frame:

    - verts_local, faces : the closed triangle mesh (for the mesh-clip
      path -- still built and cached even for shapes with a closed
      form, so cob_method='mesh' can force it for any primitive,
      e.g. for accuracy comparisons or debugging the analytic path)
    - volume             : full (unclipped) mesh volume
    - centroid_local      : full (unclipped) mesh centroid, local frame
    - bound_radius        : max distance from centroid_local to any
      vertex -- i.e. radius of the tightest sphere centered on the
      centroid that contains the whole primitive. Used as a cheap
      per-step "is this primitive anywhere near the surface" test,
      shared by both the mesh-clip and analytic paths.
    - faces_i64           : same triangles as `faces`, pre-cast to a
      contiguous int64 array once here at load time. buoyancy_cy.pyx (the
      Cython hot path) indexes into this directly every step -- casting
      fresh on every call would just move the allocation cost from
      "once at load" to "every step", which defeats the point of
      precomputing this cache in the first place.
    - geom_type, geom_size : the MuJoCo primitive type ('sphere',
      'box', ...) and its geom_size tuple, kept around so a closed-form
      solver (analytic_shapes.py) can be called directly instead of
      going through the mesh -- e.g. a sphere's exact volume needs only
      geom_size[0] (radius) and the world position, no mesh at all.
    - analytic_kind        : precomputed int dispatch code (0 = no
      closed form, use mesh; 1 = sphere; see
      primitive_meshes._ANALYTIC_KIND_CODES). Stored as a plain int
      instead of comparing `geom_type` strings in the hot loop, so the
      Cython dispatcher (buoyancy_cy.pyx) is a cheap int compare rather
      than a Python string comparison on every single step.
    """
    __slots__ = (
        'verts_local', 'faces', 'volume', 'centroid_local', 'bound_radius',
        'faces_i64', 'geom_type', 'geom_size', 'analytic_kind',
    )

    def __init__(self, verts_local, faces, volume, centroid_local, bound_radius,
                 geom_type='mesh', geom_size=()):
        self.verts_local = verts_local
        self.faces = faces
        self.volume = volume
        self.centroid_local = centroid_local
        self.bound_radius = bound_radius
        self.faces_i64 = np.ascontiguousarray(faces, dtype=np.int64)
        self.geom_type = geom_type
        self.geom_size = tuple(float(s) for s in geom_size)
        self.analytic_kind = _ANALYTIC_KIND_CODES.get(geom_type, 0)


def build_primitive_cache(geom_type: str, size, mesh_resolution: 'MeshResolution' = None) -> 'PrimitiveCache':
    """Build and cache the local-frame mesh plus its full volume,
    centroid, and bounding radius. Call once per (geom_type, size) at
    model-load time -- never in the per-step hot path.

    `mesh_resolution` only affects the tessellation used for the
    mesh-clip fallback path (build_local_mesh above); it has no effect
    on shapes with a closed form (currently just 'sphere') beyond that
    fallback mesh still being built and cached so cob_method='mesh' can
    force it if you want to sanity-check the analytic path against it.
    """
    # Local import to avoid a hard circular dependency at module load.
    from .buoyancy import submerged_volume_and_centroid

    verts, faces = build_local_mesh(geom_type, size, mesh_resolution)
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
        geom_type=geom_type,
        geom_size=size,
    )
