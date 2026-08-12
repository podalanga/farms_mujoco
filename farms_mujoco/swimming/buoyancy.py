"""buoyancy.py -- all pure-Python/NumPy buoyancy math: both "how much
force does this volume of displaced water produce" (the original
content of this file) and "how much of this primitive is submerged in
the first place" (formerly cob_core.py, i.e. the center-of-buoyancy /
submerged-volume-and-centroid math).

These two used to be separate files because they grew at different
times, not because they're doing logically different jobs -- both are
"buoyancy math", just at different levels (per-primitive submerged
volume vs. force/torque from that volume). Merged here so there's one
place to read/edit buoyancy physics instead of two.

Compiled Cython twin: buoyancy_cy.pyx (module `buoyancy_cy`, not
`buoyancy.pyx` -- see that file's docstring for why the `_cy` suffix
is load-bearing: a compiled `buoyancy.pyx` would produce a module
literally named `buoyancy`, colliding with this file in the same
package). buoyancy_cy.pyx is the hard-required hot path for the exact
'mesh'/'analytic' cob methods; this file is no longer wired into the
per-step loop for that math (see submerged_volume_and_centroid_fast
below), but IS still called every step for the 'ramp' vs 'exact'
buoyancy-force combination via compute_link_buoyancy -- see that
function's docstring.

NOTE on the import below: buoyancy_cy.pyx imports compute_link_buoyancy
from THIS module at its own top level (it has to -- that's where the
mesh/analytic per-primitive dispatch lives). So this module cannot also
import buoyancy_cy at the top level, or the two would import each other
mid-initialization. submerged_volume_and_centroid_fast_cy is therefore
imported lazily, inside submerged_volume_and_centroid_fast, the one
place it's used -- by the time that function is actually called, both
modules have finished loading.

Closed-form per-shape solvers (currently just the sphere) live in
their own file, analytic_shapes.py, and stay separate from this one:
that file is deliberately a readable derivation/reference + validation
target for buoyancy_cy's typed twin, not runtime-dispatched code, so
mixing it into this file would blur that distinction.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12
DEBUG_BUOYANCY = True
MIN_SUBMERGED_VOLUME = 1e-8 

# buoyancy_cy.pyx (compiled to buoyancy_cy.*.so) is a HARD dependency for
# the per-step hot path, not an optional accelerator. There used to be a
# try/except ImportError here that silently dropped into a pure-NumPy
# reimplementation of the same clip math (plus a separate ANALYTIC_SOLVERS
# dispatch for the analytic-sphere shortcut) whenever the extension
# wasn't built. That meant two independently-maintained copies of the
# same algorithm that could quietly drift apart -- or silently run 20x
# slower -- with no signal beyond "the .so happened to be missing".
# That's exactly the shape of bug that's expensive to track down (see
# the recent link_swimming_info memoryview issue). If buoyancy_cy isn't
# built, this now fails loudly -- the first call to
# submerged_volume_and_centroid_fast raises ImportError -- build it with
# `python setup_hydrodynamics.py build_ext --inplace` -- instead of
# degrading silently. (It can't fail at THIS module's import time
# anymore, since the import is now lazy for the circular-import reason
# above -- but it still fails loudly, just on first use instead of on
# import.)
#
# submerged_volume_and_centroid below (the vectorized NumPy clip) is NOT
# part of that removed fallback -- it's kept because
# primitive_meshes.build_primitive_cache calls it exactly once per
# (geom_type, size) at model-load time, to get each primitive's full
# unclipped volume/centroid. That's a one-off setup call, not the hot
# loop, so plain NumPy is fine there and there's no Cython twin of it to
# keep in sync.


# --- Helper Math Functions (Pure NumPy to avoid farms_core Python/Cython API quirks) ---

def _quat_xyzw_to_matrix(q):
    """Rotation matrix from (x,y,z,w) quaternion."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z),   2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),     1-2*(x*x+y*y)],
    ])

def _quat_mult_xyzw(q1, q2):
    """Hamilton product of two (x,y,z,w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


# --- Submerged volume/centroid ("center of buoyancy") math ---
# Formerly cob_core.py -- see the module docstring above for why it's
# here now.

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
    # Manual cross + dot instead of np.cross/np.einsum: both are generic
    # N-dimensional-axis-aware functions, which pays real overhead
    # (normalize_axis_tuple, moveaxis) on every call even though we
    # always have (K,3) arrays with the vector axis fixed at -1. Writing
    # it out skips that bookkeeping -- confirmed ~1s saved on the
    # zbot profile from this change alone.
    cross_x = ab[:, 1]*ac[:, 2] - ab[:, 2]*ac[:, 1]
    cross_y = ab[:, 2]*ac[:, 0] - ab[:, 0]*ac[:, 2]
    cross_z = ab[:, 0]*ac[:, 1] - ab[:, 1]*ac[:, 0]
    vol6 = aa[:, 0]*cross_x + aa[:, 1]*cross_y + aa[:, 2]*cross_z
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
        wet_verts = vertices_world[depths >= 0.0]
        xy_source = wet_verts if len(wet_verts) > 0 else vertices_world
        mean_xy = xy_source[:, :2].mean(axis=0)
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


_fast_cy = None  # cache for the lazily-imported buoyancy_cy function, set on first call


def submerged_volume_and_centroid_fast(cache, world_pos, world_rot, water_z, force_mesh=False):
    """Per-primitive, per-step submerged volume/centroid -- the function
    compute_buoyancy_mesh below actually calls in the hot loop. Thin
    wrapper around buoyancy_cy.submerged_volume_and_centroid_fast_cy,
    kept so callers import a stable name from this module rather than
    reaching into buoyancy_cy directly, and so world_pos gets the same
    dtype/contiguity normalization every time regardless of what the
    caller passed in.

    Imports buoyancy_cy lazily (see module docstring) -- it's the same
    hard dependency it always was, just checked on first call instead
    of at module-import time.

    All the actual dispatch logic -- the analytic-shape shortcut
    (`cache.analytic_kind`, currently spheres, see analytic_shapes.py
    for the derivation buoyancy_cy.pyx's typed version matches) and the
    cheap fully-above/fully-below bound-sphere reject before falling
    into the per-triangle clip loop -- lives in buoyancy_cy.pyx now.
    This file has no NumPy equivalent of it anymore; see the module-level
    comment above for why that duplication was removed.

    `force_mesh=True` forces the mesh-clip loop even for shapes with a
    closed form -- e.g. to sanity-check the analytic path against the
    mesh path.

    The import itself is cached in `_fast_cy` (module-level, resolved
    once on first call) rather than re-imported every call: the
    circular-import hazard is only at module-init time, so a
    sys.modules lookup + attribute bind on every primitive, every step
    was pure waste once the module graph has finished loading.
    """
    global _fast_cy
    if _fast_cy is None:
        from .buoyancy_cy import submerged_volume_and_centroid_fast_cy
        _fast_cy = submerged_volume_and_centroid_fast_cy
    world_pos = np.asarray(world_pos, dtype=float)
    return _fast_cy(cache, world_pos, world_rot, water_z, force_mesh)


def buoyancy_force(volume, rho_fluid, g=9.81):
    return rho_fluid * g * volume


# --- Buoyancy Methods ---
# Both of these compute ONLY buoyancy (force, torque) -- no drag, no xfrc
# writes, no rotation-frame bookkeeping beyond what buoyancy itself needs.
# hydrodynamics.pyx (via buoyancy_cy.pyx) is responsible for combining
# this with drag and writing results.

def compute_buoyancy_analytic(bound_radius, position, global2urdf, mass, water_density, surface, gravity, density):
    """Fallback method: single-point bounding-sphere ramp.

    NOTE: `bound_radius`/`position` used to be called `height`/`position`
    even though the caller always passes the primitive's bounding radius
    and world z-position -- renamed here for clarity, callers should pass
    the same values as before.
    """
    buoyancy_global = np.zeros(3)
    if mass > 0 and position - bound_radius < surface:
        buoyancy_global[2] = -water_density * mass * gravity / density * min(
            (surface + bound_radius - position) / (2 * bound_radius), 1.0
        )

    R_global2urdf = _quat_xyzw_to_matrix(global2urdf)
    return R_global2urdf @ buoyancy_global, np.zeros(3)


def compute_buoyancy_mesh(primitives, pos_urdf, com_position, urdf2global, global2urdf, water_density, surface, gravity, force_mesh=False):
    """Exact method: Stonefish-style per-primitive submerged volume.

    `force_mesh`: passed straight through to
    submerged_volume_and_centroid_fast per primitive. False (default)
    lets each primitive use its closed-form solution when it has one
    (currently spheres -- see analytic_shapes.py) and the mesh-clip
    path otherwise. True forces every primitive through the mesh-clip
    path regardless -- this is what cob_method='mesh' means (see
    compute_link_buoyancy below): useful for comparing against the
    analytic path, or if you don't trust a closed form yet.
    """
    if not primitives:
        return np.zeros(3), np.zeros(3)

    v_total = 0.0
    cwx, cwy, cwz = 0.0, 0.0, 0.0

    R_urdf2global = _quat_xyzw_to_matrix(urdf2global)
    R_global2urdf = R_urdf2global.T  # Inverse/Transpose for rotation back

    for cache, offset_pos, offset_quat in primitives:
        # 1. Geom world position
        offset_pos_global = R_urdf2global @ offset_pos
        world_pos = pos_urdf + offset_pos_global

        # 2. Geom world orientation
        world_quat = _quat_mult_xyzw(urdf2global, offset_quat)
        world_rot = _quat_xyzw_to_matrix(world_quat)

        # 3. Exact submerged volume & centroid (analytic shortcut if
        # this primitive's geom_type has one and force_mesh is False,
        # otherwise the triangle-mesh clip)
        vol_i, centroid_i = submerged_volume_and_centroid_fast(cache, world_pos, world_rot, surface, force_mesh)

        if not np.isfinite(vol_i) or vol_i <= 0.0:
            continue

        v_total += vol_i
        cwx += vol_i * centroid_i[0]
        cwy += vol_i * centroid_i[1]
        cwz += vol_i * centroid_i[2]

    if v_total <= MIN_SUBMERGED_VOLUME:
        return np.zeros(3), np.zeros(3)

    # Center of Buoyancy (World Frame)
    cob_world = np.array([cwx / v_total, cwy / v_total, cwz / v_total])

    # 4. Upward buoyancy force (World Frame)
    force_global = np.array([0.0, 0.0, -water_density * gravity * v_total])

    # 5. Righting torque about CoM (World Frame) -> tau = (CB - CoM) x F
    # r_vec = cob_world - com_position
    r_vec = cob_world - pos_urdf
    torque_global = np.cross(r_vec, force_global)

    # 6. Rotate both into URDF frame
    force_urdf = R_global2urdf @ force_global
    torque_urdf = R_global2urdf @ torque_global

    return force_urdf, torque_urdf


def compute_link_buoyancy(force_mesh, primitives, pos_urdf, com_position, urdf2global, global2urdf,
                           bound_radius, mass, water_density, surface, gravity, density):
    """Single entry point called by buoyancy_cy.pyx for the "exact" (non-
    ramp) buoyancy methods. Returns (force_urdf, torque_urdf) -- buoyancy
    only. This is only called at all when buoyancy_cy.pyx has already
    decided cob_method != 'ramp' (see SwimmingHandler.use_exact_cob in
    hydrodynamics.pyx) -- the pure 'ramp' method never crosses into
    Python, it stays entirely in compute_buoyancy_analytic_fast in
    buoyancy_cy.pyx.

    `force_mesh`: False (the 'analytic' cob_method) lets each primitive
    use a closed-form solution when its geom_type has one (currently
    spheres) and mesh-clip otherwise. True (the 'mesh' cob_method)
    forces every primitive through mesh-clip regardless of shape.

    Falls back to the analytic ramp if `primitives` is empty even
    though an exact method was requested -- e.g. a link whose collision
    geoms are all unsupported types (see geom_utils.py's degrade-safely
    comment) has nothing to clip against, so this is the same safety
    net that existed before, just no longer gated by a boolean that
    conflated "use exact method" with "exact method available".

    Renamed from `get_buoyancy_forces`: this never "gets" anything that
    already existed, it computes buoyancy from scratch every call.
    """
    if primitives:
        force_urdf, torque_urdf = compute_buoyancy_mesh(
            primitives,
            pos_urdf,
            com_position,
            urdf2global,
            global2urdf,
            water_density,
            surface,
            gravity,
            force_mesh=force_mesh,
        )

        if DEBUG_BUOYANCY:
            force_z = force_urdf[2]

            print(
                f"[BUOYANCY] "
                f"mass={mass:.6f} kg | "
                f"density={density:.2f} kg/m^3 | "
                f"volume={-force_z / (water_density * gravity):.8e} m^3 | "
                f"Fz_urdf={force_z:.6f} N | "
                f"weight={mass * gravity:.6f} N | "
                f"torque={torque_urdf}"
            )

        return force_urdf, torque_urdf
    else:
        return compute_buoyancy_analytic(
            bound_radius, pos_urdf[2], global2urdf, mass, water_density, surface, gravity, density
        )
