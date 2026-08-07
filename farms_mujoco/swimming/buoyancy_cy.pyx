# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""buoyancy_cy.pyx -- all Cython buoyancy math: the per-triangle
submerged-volume/centroid clip loop (formerly cob_fast.pyx) plus the
buoyancy force/torque entry points that were already here.

Same story as buoyancy.py/cob_core.py's merge (see buoyancy.py's module
docstring): these were two files doing "buoyancy math, compiled" at two
different levels, split only because of how they grew, not because
they're logically distinct. Merged here for the same reason.

NOTE ON THE NAME: this project has a `buoyancy.py` (pure Python/NumPy:
compute_buoyancy_analytic, compute_buoyancy_mesh, compute_link_buoyancy,
plus -- since the cob_core.py merge -- submerged_volume_and_centroid,
the vectorized NumPy clip used only at model-load time). Compiling a
Cython file literally named `buoyancy.pyx` would produce a module also
literally named `buoyancy`, which collides with `buoyancy.py` in the
same package -- only one `buoyancy` module can be imported at a time.
So this file is named `buoyancy_cy.pyx` (compiles to module
`buoyancy_cy`) specifically to coexist with `buoyancy.py`. Don't rename
this back to `buoyancy.pyx` without also renaming or removing
`buoyancy.py`, or the import at the bottom of this docstring becomes
ambiguous.

Why the per-triangle clip loop lives here and not as vectorized NumPy:
NumPy vectorization amortizes its per-call dispatch overhead over
*many* elements. That works great for big arrays. It works badly when
you call it thousands of times per second on SMALL arrays (a primitive
mesh has maybe 50-300 triangles) -- which is exactly this workload: one
call to `submerged_volume_and_centroid_fast` per primitive per
simulation step. A typed Cython loop pays overhead once per triangle
instead of once per NumPy call, so it wins once batches are this small.

buoyancy.py imports this module (lazily, from inside
submerged_volume_and_centroid_fast -- see that function's docstring for
why it's lazy) as a hard build requirement for the hot path, not an
optional accelerator. buoyancy.py still keeps
`submerged_volume_and_centroid` (the vectorized NumPy clip), but only
for primitive_meshes.build_primitive_cache's one-off, load-time "full
mesh volume" call; it is no longer invoked per-step and has no
obligation to match this file triangle-for-triangle. If you change the
clip math here, there's nothing else to keep in sync for the hot path.

Contents:
  - submerged_sphere_analytic_cy       : O(1) closed-form sphere COB
  - submerged_volume_and_centroid_fast_cy : per-primitive submerged
                                            volume/centroid, dispatching
                                            to the analytic solver above
                                            or the typed mesh-clip loop
  - compute_buoyancy_analytic_fast     : single-point bounding-sphere
                                            ramp, pure Cython
  - compute_link_buoyancy_fast         : combined per-link buoyancy
                                            entry point hydrodynamics.pyx
                                            calls
"""

import numpy as np
cimport numpy as np
from libc.math cimport fabs

from farms_core.utils.transform cimport quat_rot

# The pure-Python exact-method module -- see the naming note above for
# why this isn't a name clash despite both files being about buoyancy.
from .buoyancy import compute_link_buoyancy

ctypedef np.float64_t DTYPE_t

cdef double _EPS = 1e-12


# --- Submerged volume/centroid ("center of buoyancy") math ---
# Formerly cob_fast.pyx -- see the module docstring above for why it's
# here now. Mirrors buoyancy.py's `submerged_volume_and_centroid`
# EXACTLY (same clip formulas, same edge-crossing fallback, same
# "apex = mean(xy) at water_z" convention) -- this is not a different
# algorithm, it's the same algorithm written so the per-triangle work
# happens in one typed loop instead of ~8 NumPy calls per primitive.

cpdef tuple submerged_sphere_analytic_cy(
    double radius, double cx, double cy, double cz, double water_z,
):
    """Exact closed-form submerged volume + centroid of a sphere, O(1)
    -- no mesh, no clipping, no per-triangle loop at all. Same formula
    and same derivation as analytic_shapes.submerged_sphere; keep the
    two in sync if you ever touch this math (that Python version is the
    documented reference -- this is just its typed twin so spheres
    never pay Python-call overhead in the hot path).
    """
    cdef double bottom = cz - radius
    cdef double h = water_z - bottom
    cdef double volume, d
    cdef np.ndarray[DTYPE_t, ndim=1] centroid

    if h <= 0.0:
        centroid = np.empty(3, dtype=np.float64)
        centroid[0] = cx; centroid[1] = cy; centroid[2] = cz
        return 0.0, centroid
    if h >= 2.0 * radius:
        centroid = np.empty(3, dtype=np.float64)
        centroid[0] = cx; centroid[1] = cy; centroid[2] = cz
        return (4.0 / 3.0) * np.pi * radius * radius * radius, centroid

    volume = np.pi * h * h * (3.0 * radius - h) / 3.0
    d = 3.0 * (2.0 * radius - h) * (2.0 * radius - h) / (4.0 * (3.0 * radius - h))
    centroid = np.empty(3, dtype=np.float64)
    centroid[0] = cx; centroid[1] = cy; centroid[2] = cz - d
    return volume, centroid


cdef inline double _cross_t(double di, double dj) nogil:
    """Edge-crossing parameter t in [0, 1] where depth hits zero.
    Matches buoyancy._cross_point's near-zero-denominator -> t=0.5
    fallback, and its clamp to [0, 1]."""
    cdef double denom = di - dj
    cdef double t
    if fabs(denom) < _EPS:
        return 0.5
    t = di / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t


cdef inline double _tet_vol6(
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
) nogil:
    """6x signed volume of the tetrahedron (apex, a, b, c), given a, b, c
    already as edge vectors *from* apex (i.e. a = vert_a - apex, etc).
    This is det([a b c]) = a . (b x c), written out by hand instead of
    via np.cross + np.einsum -- see buoyancy.py's module docstring for
    why that inlining matters even outside Cython."""
    return (
        ax * (by * cz - bz * cy)
        - ay * (bx * cz - bz * cx)
        + az * (bx * cy - by * cx)
    )


cdef inline void _accumulate_tet(
    double apex_x, double apex_y, double apex_z,
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz,
    double *total_vol, double *cwx, double *cwy, double *cwz,
) noexcept nogil:
    """Add one tet(apex, a, b, c)'s signed volume and volume-weighted
    centroid into the running totals. a, b, c are WORLD-frame points
    (not yet apex-relative) -- this is the one place that offset
    happens, so callers just hand in raw points."""
    cdef double ex = ax - apex_x
    cdef double ey = ay - apex_y
    cdef double ez = az - apex_z
    cdef double fx = bx - apex_x
    cdef double fy = by - apex_y
    cdef double fz = bz - apex_z
    cdef double gx = cx - apex_x
    cdef double gy = cy - apex_y
    cdef double gz = cz - apex_z
    cdef double vol = _tet_vol6(ex, ey, ez, fx, fy, fz, gx, gy, gz) / 6.0
    cdef double tcx = (apex_x + ax + bx + cx) / 4.0
    cdef double tcy = (apex_y + ay + by + cy) / 4.0
    cdef double tcz = (apex_z + az + bz + cz) / 4.0
    total_vol[0] += vol
    cwx[0] += vol * tcx
    cwy[0] += vol * tcy
    cwz[0] += vol * tcz


def submerged_volume_and_centroid_fast_cy(
    object cache,
    np.ndarray[DTYPE_t, ndim=1] world_pos,
    np.ndarray[DTYPE_t, ndim=2] world_rot,
    double water_z,
    bint force_mesh=False,
):
    """Drop-in replacement for
    buoyancy.submerged_volume_and_centroid_fast(cache, world_pos,
    world_rot, water_z) -> (volume, centroid_world).

    Dispatch order:
    1. If `cache.analytic_kind` says this primitive has a closed form
       (currently: spheres) and `force_mesh` is False, use that --
       O(1), exact, no mesh touched at all (skips even the bound-sphere
       reject below, since the analytic path is already cheaper than
       computing that reject would be).
    2. Otherwise, same cheap bound-sphere reject as before
       (fully-above/fully-below early-outs), then -- only when the
       primitive straddles the surface -- the typed per-triangle clip
       loop.

    `force_mesh=True` skips step 1 even for shapes with a closed form
    -- useful for validating the analytic path against the mesh path,
    or for consistently comparing accuracy/timing between methods.
    """
    if not force_mesh and cache.analytic_kind == 1:
        return submerged_sphere_analytic_cy(
            cache.geom_size[0], world_pos[0], world_pos[1], world_pos[2], water_z,
        )

    cdef np.ndarray[DTYPE_t, ndim=1] centroid_local = cache.centroid_local
    cdef np.ndarray[DTYPE_t, ndim=1] centroid_world = world_rot.dot(centroid_local) + world_pos
    cdef double bound_radius = cache.bound_radius
    cdef double depth = water_z - centroid_world[2]

    if depth < -bound_radius:
        return 0.0, centroid_world
    if depth > bound_radius:
        return <double> cache.volume, centroid_world

    cdef np.ndarray[DTYPE_t, ndim=2] verts_local = cache.verts_local
    cdef np.ndarray[np.int64_t, ndim=2] faces = cache.faces_i64
    # verts_world = verts_local @ world_rot.T + world_pos, one NumPy call
    # (cheap -- primitive vertex counts are small and this only runs
    # once per call, not once per triangle).
    cdef np.ndarray[DTYPE_t, ndim=2] verts_world = np.ascontiguousarray(
        verts_local.dot(world_rot.T) + world_pos
    )

    cdef Py_ssize_t n_verts = verts_world.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] depths = water_z - verts_world[:, 2]

    cdef double apex_x = 0.0, apex_y = 0.0
    cdef Py_ssize_t vi
    for vi in range(n_verts):
        apex_x += verts_world[vi, 0]
        apex_y += verts_world[vi, 1]
    apex_x /= n_verts
    apex_y /= n_verts
    cdef double apex_z = water_z

    cdef Py_ssize_t n_faces = faces.shape[0]
    cdef Py_ssize_t fi, i0, i1, i2, ii, ij, ik
    cdef int n_wet
    cdef double d0, d1, d2
    cdef double v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z
    cdef double ivx, ivy, ivz, di_
    cdef double jvx, jvy, jvz, dj_
    cdef double kvx, kvy, kvz, dk_
    cdef double px, py, pz, qx, qy, qz, t
    cdef double total_vol = 0.0
    cdef double cwx = 0.0, cwy = 0.0, cwz = 0.0

    for fi in range(n_faces):
        i0 = faces[fi, 0]
        i1 = faces[fi, 1]
        i2 = faces[fi, 2]
        d0 = depths[i0]
        d1 = depths[i1]
        d2 = depths[i2]

        n_wet = 0
        if d0 >= 0.0:
            n_wet += 1
        if d1 >= 0.0:
            n_wet += 1
        if d2 >= 0.0:
            n_wet += 1

        if n_wet == 0:
            continue

        v0x = verts_world[i0, 0]; v0y = verts_world[i0, 1]; v0z = verts_world[i0, 2]
        v1x = verts_world[i1, 0]; v1y = verts_world[i1, 1]; v1z = verts_world[i1, 2]
        v2x = verts_world[i2, 0]; v2y = verts_world[i2, 1]; v2z = verts_world[i2, 2]

        if n_wet == 3:
            # Fully wet triangle: no clipping, one tet straight to apex.
            _accumulate_tet(
                apex_x, apex_y, apex_z,
                v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z,
                &total_vol, &cwx, &cwy, &cwz,
            )
            continue

        # n_wet in (1, 2): identify the "odd one out" vertex i -- the
        # lone wet vertex when n_wet==1, the lone dry vertex when
        # n_wet==2 -- exactly like buoyancy.py's per-i loop, just
        # resolved directly instead of via three separate boolean masks.
        if n_wet == 1:
            if d0 >= 0.0:
                ii, ij, ik = i0, i1, i2
            elif d1 >= 0.0:
                ii, ij, ik = i1, i2, i0
            else:
                ii, ij, ik = i2, i0, i1
        else:
            if d0 < 0.0:
                ii, ij, ik = i0, i1, i2
            elif d1 < 0.0:
                ii, ij, ik = i1, i2, i0
            else:
                ii, ij, ik = i2, i0, i1

        ivx = verts_world[ii, 0]; ivy = verts_world[ii, 1]; ivz = verts_world[ii, 2]
        jvx = verts_world[ij, 0]; jvy = verts_world[ij, 1]; jvz = verts_world[ij, 2]
        kvx = verts_world[ik, 0]; kvy = verts_world[ik, 1]; kvz = verts_world[ik, 2]
        di_ = depths[ii]; dj_ = depths[ij]; dk_ = depths[ik]

        # Crossing points on edges (i->j) and (i->k).
        t = _cross_t(di_, dj_)
        px = ivx + t * (jvx - ivx); py = ivy + t * (jvy - ivy); pz = ivz + t * (jvz - ivz)
        t = _cross_t(di_, dk_)
        qx = ivx + t * (kvx - ivx); qy = ivy + t * (kvy - ivy); qz = ivz + t * (kvz - ivz)

        if n_wet == 1:
            # Small wet triangle (vi, p_on_ij, p_on_ik).
            _accumulate_tet(
                apex_x, apex_y, apex_z,
                ivx, ivy, ivz, px, py, pz, qx, qy, qz,
                &total_vol, &cwx, &cwy, &cwz,
            )
        else:
            # Wet quad (p_on_ij, vj, vk, p_on_ik) -> two tets, same
            # split as buoyancy.py's n_wet==2 branch.
            _accumulate_tet(
                apex_x, apex_y, apex_z,
                px, py, pz, jvx, jvy, jvz, kvx, kvy, kvz,
                &total_vol, &cwx, &cwy, &cwz,
            )
            _accumulate_tet(
                apex_x, apex_y, apex_z,
                px, py, pz, kvx, kvy, kvz, qx, qy, qz,
                &total_vol, &cwx, &cwy, &cwz,
            )

    if fabs(total_vol) < _EPS:
        return 0.0, verts_world.mean(axis=0)

    cdef np.ndarray[DTYPE_t, ndim=1] centroid = np.empty(3, dtype=np.float64)
    centroid[0] = cwx / total_vol
    centroid[1] = cwy / total_vol
    centroid[2] = cwz / total_vol
    return total_vol, centroid


# --- Buoyancy force/torque entry points ---
# Original content of this file.

cdef void compute_buoyancy_analytic_fast(
    double density,
    double water_density,
    double bound_radius,
    double position,
    DTYPEv1 global2urdf,
    double mass,
    double surface,
    double gravity,
    DTYPEv1 buoyancy,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 tmp,
):
    """Single-point bounding-sphere ramp -- fast Cython path used when
    cob_method='ramp' (or as the safety-net fallback for a link with no
    usable collision primitives even when an exact method is
    requested).

    NOTE: this duplicates the math in buoyancy.compute_buoyancy_analytic.
    That's intentional -- this cdef version stays in Cython so the
    common case (every ramp-method link, every step) never crosses into
    Python. buoyancy.compute_buoyancy_analytic is kept only as the
    Python-callable reference/fallback; if you change the ramp formula,
    change it in BOTH places or delete one.
    """
    cdef unsigned int i
    if mass > 0 and position - bound_radius < surface:
        tmp[0] = 0
        tmp[1] = 0
        tmp[2] = -water_density*mass*gravity/density*min(
            (surface + bound_radius - position) / (2 * bound_radius),
            1,
        )
        quat_rot(tmp, global2urdf, quat_c, tmp4, buoyancy)
    else:
        for i in range(3):
            buoyancy[i] = 0


cdef void compute_link_buoyancy_fast(
    bint use_buoyancy,
    bint use_exact_cob,
    bint force_mesh,
    object primitives,
    double density,
    double water_density,
    double bound_radius,
    double pos_x,
    double pos_y,
    double pos_z,
    DTYPEv1 global2urdf,
    DTYPEv1 urdf2global,
    double mass,
    double surface,
    double gravity,
    DTYPEv1 buoyancy,
    DTYPEv1 buoyancy_torque,
    DTYPEv1 com_position,
    DTYPEv1 pos_urdf,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 tmp,
) except *:
    """Combined per-link buoyancy entry point, called once per link per
    step by hydrodynamics.compute_link_forces. Fills `buoyancy` (force,
    URDF frame) and `buoyancy_torque` (URDF frame).

    Dispatch mirrors cob_method exactly ('ramp' / 'mesh' / 'analytic',
    see SwimmingHandler in hydrodynamics.pyx for where use_exact_cob and
    force_mesh come from): the ramp case never leaves Cython; the exact
    cases call into buoyancy.py's compute_link_buoyancy, which is a
    genuine Python call (mesh clipping / analytic-shape dispatch
    happens there, dropping straight back into this same module's
    submerged_volume_and_centroid_fast_cy for the actual per-primitive
    work) -- that's why this function is declared `except *`, so an
    exception raised on the Python side (e.g. a malformed primitives
    cache) actually propagates instead of being silently swallowed,
    which is what a bare `cdef void` would otherwise risk.

    `com_position` must already be filled by the caller whenever
    use_buoyancy and use_exact_cob and primitives are all true -- this
    function doesn't know how to read sensor data (deliberately; see
    hydrodynamics.pyx, which owns LinkSensorArrayCy). It's ignored
    otherwise.
    """
    cdef unsigned int i

    if not use_buoyancy:
        for i in range(3):
            buoyancy[i] = 0
            buoyancy_torque[i] = 0
        return

    if use_exact_cob and primitives:
        pos_urdf[0] = pos_x
        pos_urdf[1] = pos_y
        pos_urdf[2] = pos_z
        res_force, res_torque = compute_link_buoyancy(
            force_mesh=force_mesh,
            primitives=primitives,
            pos_urdf=pos_urdf,
            com_position=com_position,
            urdf2global=urdf2global,
            global2urdf=global2urdf,
            bound_radius=bound_radius,
            mass=mass,
            water_density=water_density,
            surface=surface,
            gravity=gravity,
            density=density,
        )
        for i in range(3):
            buoyancy[i] = res_force[i]
            buoyancy_torque[i] = res_torque[i]
    else:
        compute_buoyancy_analytic_fast(
            density=density,
            water_density=water_density,
            bound_radius=bound_radius,
            position=pos_z,
            global2urdf=global2urdf,
            mass=mass,
            surface=surface,
            gravity=gravity,
            buoyancy=buoyancy,
            quat_c=quat_c,
            tmp4=tmp4,
            tmp=tmp,
        )
        for i in range(3):
            buoyancy_torque[i] = 0
