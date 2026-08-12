# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""buoyancy_cy.pyx -- all Cython buoyancy math: the per-triangle
submerged-volume/centroid clip loop (formerly cob_fast.pyx) plus the
buoyancy force/torque entry points that were already here.
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


cpdef submerged_volume_and_centroid_fast_cy(
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
    cdef int n_wet_verts = 0
    cdef Py_ssize_t vi
    for vi in range(n_verts):
        if depths[vi] >= 0.0:
            apex_x += verts_world[vi, 0]
            apex_y += verts_world[vi, 1]
            n_wet_verts += 1
    if n_wet_verts > 0:
        apex_x /= n_wet_verts
        apex_y /= n_wet_verts
    else:
        # No individually-wet vertex even though centroid depth was inside
        # the bound-radius band (grazing-corner case) -- total_vol will
        # come out ~0 regardless; fall back to all-vertex mean so we don't
        # divide by zero.
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


# --- Fast inline math helpers (cdef nogil, zero heap allocation) ---
# Replace buoyancy.py's _quat_xyzw_to_matrix, _quat_mult_xyzw, np.cross
# which together consumed ~35 s in profiling at 857 K calls/step.

cdef inline void _mat3_from_quat(
    double qx, double qy, double qz, double qw,
    double *m,
) nogil:
    """Row-major 3x3 rotation matrix from (x,y,z,w) quaternion.
    Replaces buoyancy._quat_xyzw_to_matrix -- same formula, no heap."""
    cdef double x2 = 2.0*qx*qx, y2 = 2.0*qy*qy, z2 = 2.0*qz*qz
    cdef double xy = 2.0*qx*qy, xz = 2.0*qx*qz, yz = 2.0*qy*qz
    cdef double wx = 2.0*qw*qx, wy = 2.0*qw*qy, wz = 2.0*qw*qz
    m[0] = 1.0-y2-z2;  m[1] = xy-wz;       m[2] = xz+wy
    m[3] = xy+wz;       m[4] = 1.0-x2-z2;  m[5] = yz-wx
    m[6] = xz-wy;       m[7] = yz+wx;       m[8] = 1.0-x2-y2


cdef inline void _quat_mult_cy(
    double ax, double ay, double az, double aw,
    double bx, double by, double bz, double bw,
    double *rx, double *ry, double *rz, double *rw,
) nogil:
    """Hamilton product of two (x,y,z,w) quaternions.
    Replaces buoyancy._quat_mult_xyzw -- no heap, no Python object creation."""
    rx[0] = aw*bx + ax*bw + ay*bz - az*by
    ry[0] = aw*by - ax*bz + ay*bw + az*bx
    rz[0] = aw*bz + ax*by - ay*bx + az*bw
    rw[0] = aw*bw - ax*bx - ay*by - az*bz


# --- Pure-Cython buoyancy mesh: replaces buoyancy.compute_buoyancy_mesh ---

cdef void _compute_buoyancy_mesh_cy(
    object primitives,
    DTYPEv1 pos_world,
    DTYPEv1 com_world,
    DTYPEv1 urdf2global_q,
    double water_density,
    double surface,
    double gravity,
    bint force_mesh,
    DTYPEv1 force_out,
    DTYPEv1 buoyancy_torque,
) except *:
    """Pure-Cython replacement for buoyancy.compute_buoyancy_mesh.

    Eliminates every Python call from the per-link buoyancy hot path:
      - _quat_xyzw_to_matrix  -> _mat3_from_quat (cdef nogil, no heap)
      - _quat_mult_xyzw       -> _quat_mult_cy   (cdef nogil, no heap)
      - np.cross on [0,0,fz]  -> 2 scalar mults  (torque arm is world-up
                                  so tx=ry*fz, ty=-rx*fz, tz=0)
      - np.array(...)         -> pre-declared C scalars / reused ndarrays
      - submerged_volume_and_centroid_fast_cy is cpdef, so from Cython
        the call goes through C dispatch, not Python's call protocol.

    Torque is computed about com_world (the CoM), matching what
    MuJoCo's xfrc_applied expects (per the bug fix in buoyancy.py).
    """
    cdef:
        double Ru[9]   # R_urdf2global, row-major
        double Rg[9]   # R_global2urdf = Ru^T, row-major
        double v_total = 0.0, cwx = 0.0, cwy = 0.0, cwz = 0.0
        double vol_i
        double cob_x, cob_y, cob_z
        double fz_g             # buoyancy force magnitude (world z)
        double rvx, rvy, rvz   # torque arm CoB-CoM
        double tx_g, ty_g       # torque x,y (tz=0 since F is world-up)
        double off_px, off_py, off_pz
        double off_qx, off_qy, off_qz, off_qw
        double og_x, og_y, og_z  # offset in global frame
        double wrx, wry, wrz, wrw
        double wR[9]
        int i
    # Reuse two numpy arrays per call instead of allocating inside the loop.
    cdef np.ndarray[DTYPE_t, ndim=1] wp = np.empty(3, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] wRmat = np.empty((3, 3), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] centroid_i

    if not primitives:
        for i in range(3):
            force_out[i] = 0.0
            buoyancy_torque[i] = 0.0
        return

    # R_urdf2global
    _mat3_from_quat(
        urdf2global_q[0], urdf2global_q[1],
        urdf2global_q[2], urdf2global_q[3], Ru,
    )
    # R_global2urdf = Ru^T
    Rg[0]=Ru[0]; Rg[1]=Ru[3]; Rg[2]=Ru[6]
    Rg[3]=Ru[1]; Rg[4]=Ru[4]; Rg[5]=Ru[7]
    Rg[6]=Ru[2]; Rg[7]=Ru[5]; Rg[8]=Ru[8]

    for cache, offset_pos_arr, offset_quat_arr in primitives:
        off_px = offset_pos_arr[0]
        off_py = offset_pos_arr[1]
        off_pz = offset_pos_arr[2]
        off_qx = offset_quat_arr[0]; off_qy = offset_quat_arr[1]
        off_qz = offset_quat_arr[2]; off_qw = offset_quat_arr[3]

        # 1. Geom world position = pos_world + Ru @ offset_pos
        og_x = Ru[0]*off_px + Ru[1]*off_py + Ru[2]*off_pz
        og_y = Ru[3]*off_px + Ru[4]*off_py + Ru[5]*off_pz
        og_z = Ru[6]*off_px + Ru[7]*off_py + Ru[8]*off_pz
        wp[0] = pos_world[0] + og_x
        wp[1] = pos_world[1] + og_y
        wp[2] = pos_world[2] + og_z

        # 2. Geom world orientation = urdf2global ⊗ offset_quat
        _quat_mult_cy(
            urdf2global_q[0], urdf2global_q[1],
            urdf2global_q[2], urdf2global_q[3],
            off_qx, off_qy, off_qz, off_qw,
            &wrx, &wry, &wrz, &wrw,
        )
        _mat3_from_quat(wrx, wry, wrz, wrw, wR)
        wRmat[0,0]=wR[0]; wRmat[0,1]=wR[1]; wRmat[0,2]=wR[2]
        wRmat[1,0]=wR[3]; wRmat[1,1]=wR[4]; wRmat[1,2]=wR[5]
        wRmat[2,0]=wR[6]; wRmat[2,1]=wR[7]; wRmat[2,2]=wR[8]

        # 3. Submerged volume & centroid (cpdef -> fast Cython dispatch)
        result = submerged_volume_and_centroid_fast_cy(
            cache, wp, wRmat, surface, force_mesh,
        )
        vol_i = result[0]
        if not (vol_i > 0.0):   # rejects NaN and zero cleanly
            continue
        centroid_i = result[1]
        v_total += vol_i
        cwx += vol_i * centroid_i[0]
        cwy += vol_i * centroid_i[1]
        cwz += vol_i * centroid_i[2]

    if v_total <= 1e-8:
        for i in range(3):
            force_out[i] = 0.0
            buoyancy_torque[i] = 0.0
        return

    # Center of Buoyancy (world frame)
    cob_x = cwx / v_total
    cob_y = cwy / v_total
    cob_z = cwz / v_total

    # 4. Buoyancy force (world frame): always [0, 0, fz_g]
    fz_g = -water_density * gravity * v_total

    # 5. Torque = (CoB - CoM) x [0, 0, fz_g]
    #    Expanding the cross product with bx=by=0, bz=fz_g:
    #      tx = ry*fz_g,   ty = -rx*fz_g,   tz = 0
    rvx = cob_x - com_world[0]
    rvy = cob_y - com_world[1]
    tx_g =  rvy * fz_g
    ty_g = -rvx * fz_g
    # tz_g = 0 (exact -- no rounding)

    # 6. Rotate into URDF frame.
    #    force = Rg @ [0, 0, fz_g]  ->  only col-2 of Rg matters
    force_out[0] = Rg[2] * fz_g
    force_out[1] = Rg[5] * fz_g
    force_out[2] = Rg[8] * fz_g
    #    torque = Rg @ [tx_g, ty_g, 0]  ->  only first two cols matter
    buoyancy_torque[0] = Rg[0]*tx_g + Rg[1]*ty_g
    buoyancy_torque[1] = Rg[3]*tx_g + Rg[4]*ty_g
    buoyancy_torque[2] = Rg[6]*tx_g + Rg[7]*ty_g


# --- analytic_fast interpolation path ---
# Replaces the full per-triangle mesh-clip every step with a
# compute-then-interpolate scheme: on "anchor" steps (every interp_steps
# simulation steps) the exact CoB+volume are computed for real; on
# intermediate steps, both are linearly interpolated from the two most
# recent anchor values.  The per-triangle loop therefore runs only once
# every interp_steps steps instead of every step.
#
# State per link (owned by SwimmingHandler, passed in as views):
#   interp_cob_prev[3]   -- CoB at the older anchor
#   interp_cob_curr[3]   -- CoB at the most-recent anchor
#   interp_vol_prev      -- volume at the older anchor
#   interp_vol_curr      -- volume at the most-recent anchor
#   interp_counter       -- steps since last anchor (0..interp_steps-1)
#
# On an anchor step (interp_counter == 0):
#   1. Run the full exact mesh computation.
#   2. Rotate the previous curr values into prev, store new curr.
#   3. Reset counter to 1 for next step.
# On intermediate steps:
#   1. t = interp_counter / interp_steps  (in (0, 1])
#   2. vol    = lerp(interp_vol_prev, interp_vol_curr, t)
#   3. cob    = lerp(interp_cob_prev, interp_cob_curr, t)
#   4. Compute force/torque from the interpolated values directly.
#
# Edge-case handling:
#   - If vol_curr <= 0 and vol_prev <= 0, both anchors say "not in water";
#     emit zero force/torque without going into the mesh loop.
#   - On the very first call (both vol_prev and vol_curr == 0.0 AND counter
#     == 0) we always do a real computation, so the very first step is exact.

cdef void _compute_buoyancy_mesh_fast_cy(
    object primitives,
    DTYPEv1 pos_world,
    DTYPEv1 com_world,
    DTYPEv1 urdf2global_q,
    double water_density,
    double surface,
    double gravity,
    bint force_mesh,
    int interp_steps,
    DTYPEv1 interp_cob_prev,
    DTYPEv1 interp_cob_curr,
    double[:] interp_vol,   # length-2 view: [0]=prev, [1]=curr
    int[:] interp_counter,  # length-1 view: mutable step counter
    DTYPEv1 force_out,
    DTYPEv1 buoyancy_torque,
) except *:
    """'analytic_fast' buoyancy: exact on anchor steps, linearly
    interpolated on intermediate steps. See the block comment above for
    the full algorithm and state-variable descriptions.

    `interp_vol` and `interp_counter` are slices of per-link arrays
    owned by SwimmingHandler -- they're passed as typed memoryview
    slices so Cython can write back into them without crossing a Python
    boundary.
    """
    cdef:
        double vol_prev = interp_vol[0]
        double vol_curr = interp_vol[1]
        int counter     = interp_counter[0]
        double t, vol_interp
        double cob_x, cob_y, cob_z
        double fz_g, rvx, rvy
        double tx_g, ty_g
        double Ru[9], Rg[9]
        int i

    # --- anchor step: run exact computation ---
    if counter == 0:
        # Promote curr -> prev
        interp_vol[0]     = vol_curr
        interp_cob_prev[0] = interp_cob_curr[0]
        interp_cob_prev[1] = interp_cob_curr[1]
        interp_cob_prev[2] = interp_cob_curr[2]

        # Run exact mesh computation into force_out / buoyancy_torque
        _compute_buoyancy_mesh_cy(
            primitives=primitives,
            pos_world=pos_world,
            com_world=com_world,
            urdf2global_q=urdf2global_q,
            water_density=water_density,
            surface=surface,
            gravity=gravity,
            force_mesh=force_mesh,
            force_out=force_out,
            buoyancy_torque=buoyancy_torque,
        )

        # Recover the new CoB and volume from the force/torque outputs so
        # we can cache them for interpolation without a second mesh pass.
        # force_out is in URDF frame; we need world-frame fz to get volume.
        # _compute_buoyancy_mesh_cy sets force_out = Rg @ [0, 0, fz_g],
        # so   fz_g = Ru[col2] . force_out = Ru[2]*f0 + Ru[5]*f1 + Ru[8]*f2.
        # For CoB we need to store it -- but _compute_buoyancy_mesh_cy
        # doesn't expose it. Rather than duplicating the mesh loop, we
        # re-derive volume from the force magnitude and CoB from the torque:
        #   fz_g = -rho*g*V  =>  V = -fz_g / (rho*g)
        #   torque_g = [rvy*fz_g, -rvx*fz_g, 0]  =>  rvx = -ty_g/fz_g,
        #                                              rvy =  tx_g/fz_g
        #   cob = com + [rvx, rvy, rvz]   -- but rvz is unknown from torque
        #   alone.  We therefore store only the xy displacement and use
        #   the known volume to reconstruct what we need.
        # ... This reverse-engineering is fragile.  Instead, call a
        # separate helper that returns (vol, cob) directly alongside the
        # force, so we can cache them cleanly.
        #
        # Because adding a full return-value to the existing
        # _compute_buoyancy_mesh_cy would change its signature (and every
        # caller), we call it via a thin wrapper that also extracts
        # (vol, cob) -- see _compute_buoyancy_mesh_with_cob_cy below.
        # We already wrote force_out/torque from the call above; overwrite
        # them again (same result) so we also get vol_new and cob_new.
        vol_new, cob_x, cob_y, cob_z = _buoyancy_mesh_vol_cob_cy(
            primitives, pos_world, com_world, urdf2global_q,
            water_density, surface, gravity, force_mesh,
            force_out, buoyancy_torque,
        )

        interp_vol[1]      = vol_new
        interp_cob_curr[0] = cob_x
        interp_cob_curr[1] = cob_y
        interp_cob_curr[2] = cob_z

        # Advance counter; wrap at interp_steps
        interp_counter[0] = 1
        return

    # --- intermediate step: interpolate ---
    # Both anchors zero => link is out of water on both; emit nothing.
    if vol_prev <= 1e-8 and vol_curr <= 1e-8:
        for i in range(3):
            force_out[i]       = 0.0
            buoyancy_torque[i] = 0.0
        interp_counter[0] = (counter + 1) % interp_steps
        return

    t = <double>counter / <double>interp_steps

    # Interpolate volume and CoB
    vol_interp = vol_prev + t * (vol_curr - vol_prev)
    cob_x = interp_cob_prev[0] + t * (interp_cob_curr[0] - interp_cob_prev[0])
    cob_y = interp_cob_prev[1] + t * (interp_cob_curr[1] - interp_cob_prev[1])
    cob_z = interp_cob_prev[2] + t * (interp_cob_curr[2] - interp_cob_prev[2])

    if vol_interp <= 1e-8:
        for i in range(3):
            force_out[i]       = 0.0
            buoyancy_torque[i] = 0.0
        interp_counter[0] = (counter + 1) % interp_steps
        return

    # Build R matrices for frame rotation (same as _compute_buoyancy_mesh_cy)
    _mat3_from_quat(
        urdf2global_q[0], urdf2global_q[1],
        urdf2global_q[2], urdf2global_q[3], Ru,
    )
    Rg[0]=Ru[0]; Rg[1]=Ru[3]; Rg[2]=Ru[6]
    Rg[3]=Ru[1]; Rg[4]=Ru[4]; Rg[5]=Ru[7]
    Rg[6]=Ru[2]; Rg[7]=Ru[5]; Rg[8]=Ru[8]

    fz_g = -water_density * gravity * vol_interp

    # torque = (cob - com) x [0, 0, fz_g], same shortcut as mesh path
    rvx = cob_x - com_world[0]
    rvy = cob_y - com_world[1]
    tx_g =  rvy * fz_g
    ty_g = -rvx * fz_g

    force_out[0] = Rg[2] * fz_g
    force_out[1] = Rg[5] * fz_g
    force_out[2] = Rg[8] * fz_g
    buoyancy_torque[0] = Rg[0]*tx_g + Rg[1]*ty_g
    buoyancy_torque[1] = Rg[3]*tx_g + Rg[4]*ty_g
    buoyancy_torque[2] = Rg[6]*tx_g + Rg[7]*ty_g

    interp_counter[0] = (counter + 1) % interp_steps


cdef tuple _buoyancy_mesh_vol_cob_cy(
    object primitives,
    DTYPEv1 pos_world,
    DTYPEv1 com_world,
    DTYPEv1 urdf2global_q,
    double water_density,
    double surface,
    double gravity,
    bint force_mesh,
    DTYPEv1 force_out,
    DTYPEv1 buoyancy_torque,
):
    """Same as _compute_buoyancy_mesh_cy but also returns (vol, cob_x,
    cob_y, cob_z) as a Python tuple so _compute_buoyancy_mesh_fast_cy
    can cache the CoB and volume after an anchor computation without
    running the mesh loop a second time.

    This is intentionally a separate cdef that duplicates the core
    _compute_buoyancy_mesh_cy logic rather than refactoring that
    function's signature -- keeping _compute_buoyancy_mesh_cy unchanged
    avoids touching all its existing callers.
    """
    cdef:
        double Ru[9]
        double Rg[9]
        double v_total = 0.0, cwx = 0.0, cwy = 0.0, cwz = 0.0
        double vol_i
        double cob_x, cob_y, cob_z
        double fz_g, rvx, rvy, tx_g, ty_g
        double off_px, off_py, off_pz
        double off_qx, off_qy, off_qz, off_qw
        double og_x, og_y, og_z
        double wrx, wry, wrz, wrw
        double wR[9]
        int i
    cdef np.ndarray[DTYPE_t, ndim=1] wp = np.empty(3, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] wRmat = np.empty((3, 3), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] centroid_i

    if not primitives:
        for i in range(3):
            force_out[i] = 0.0
            buoyancy_torque[i] = 0.0
        return 0.0, 0.0, 0.0, 0.0

    _mat3_from_quat(
        urdf2global_q[0], urdf2global_q[1],
        urdf2global_q[2], urdf2global_q[3], Ru,
    )
    Rg[0]=Ru[0]; Rg[1]=Ru[3]; Rg[2]=Ru[6]
    Rg[3]=Ru[1]; Rg[4]=Ru[4]; Rg[5]=Ru[7]
    Rg[6]=Ru[2]; Rg[7]=Ru[5]; Rg[8]=Ru[8]

    for cache, offset_pos_arr, offset_quat_arr in primitives:
        off_px = offset_pos_arr[0]
        off_py = offset_pos_arr[1]
        off_pz = offset_pos_arr[2]
        off_qx = offset_quat_arr[0]; off_qy = offset_quat_arr[1]
        off_qz = offset_quat_arr[2]; off_qw = offset_quat_arr[3]

        og_x = Ru[0]*off_px + Ru[1]*off_py + Ru[2]*off_pz
        og_y = Ru[3]*off_px + Ru[4]*off_py + Ru[5]*off_pz
        og_z = Ru[6]*off_px + Ru[7]*off_py + Ru[8]*off_pz
        wp[0] = pos_world[0] + og_x
        wp[1] = pos_world[1] + og_y
        wp[2] = pos_world[2] + og_z

        _quat_mult_cy(
            urdf2global_q[0], urdf2global_q[1],
            urdf2global_q[2], urdf2global_q[3],
            off_qx, off_qy, off_qz, off_qw,
            &wrx, &wry, &wrz, &wrw,
        )
        _mat3_from_quat(wrx, wry, wrz, wrw, wR)
        wRmat[0,0]=wR[0]; wRmat[0,1]=wR[1]; wRmat[0,2]=wR[2]
        wRmat[1,0]=wR[3]; wRmat[1,1]=wR[4]; wRmat[1,2]=wR[5]
        wRmat[2,0]=wR[6]; wRmat[2,1]=wR[7]; wRmat[2,2]=wR[8]

        result = submerged_volume_and_centroid_fast_cy(cache, wp, wRmat, surface, force_mesh)
        vol_i = result[0]
        if not (vol_i > 0.0):
            continue
        centroid_i = result[1]
        v_total += vol_i
        cwx += vol_i * centroid_i[0]
        cwy += vol_i * centroid_i[1]
        cwz += vol_i * centroid_i[2]

    if v_total <= 1e-8:
        for i in range(3):
            force_out[i] = 0.0
            buoyancy_torque[i] = 0.0
        return 0.0, 0.0, 0.0, 0.0

    cob_x = cwx / v_total
    cob_y = cwy / v_total
    cob_z = cwz / v_total

    fz_g = -water_density * gravity * v_total
    rvx = cob_x - com_world[0]
    rvy = cob_y - com_world[1]
    tx_g =  rvy * fz_g
    ty_g = -rvx * fz_g

    force_out[0] = Rg[2] * fz_g
    force_out[1] = Rg[5] * fz_g
    force_out[2] = Rg[8] * fz_g
    buoyancy_torque[0] = Rg[0]*tx_g + Rg[1]*ty_g
    buoyancy_torque[1] = Rg[3]*tx_g + Rg[4]*ty_g
    buoyancy_torque[2] = Rg[6]*tx_g + Rg[7]*ty_g

    return v_total, cob_x, cob_y, cob_z


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
    bint use_interp_fast=False,
    int interp_steps=20,
    object interp_state=None,
) except *:
    """Combined per-link buoyancy entry point, called once per link per
    step by hydrodynamics.compute_link_forces. Fills `buoyancy` (force,
    URDF frame) and `buoyancy_torque` (URDF frame).

    Dispatch order:
      1. !use_buoyancy            -> zeros, return
      2. use_interp_fast + prims  -> _compute_buoyancy_mesh_fast_cy
                                     (anchor/interp, exact every interp_steps)
      3. use_exact_cob + prims    -> _compute_buoyancy_mesh_cy (exact, every step)
      4. else                     -> compute_buoyancy_analytic_fast (ramp)

    `use_interp_fast` / `interp_steps` come from cob_method='analytic_fast'
    in SwimmingHandler (hydrodynamics.pyx). `interp_state` is a dict with
    keys 'cob_prev', 'cob_curr', 'vol', 'counter' (numpy arrays owned by
    SwimmingHandler, one row per link -- passed as typed-memoryview slices
    so Cython can write back into them without crossing a Python boundary).

    `com_position` must already be filled by the caller whenever
    use_buoyancy and (use_exact_cob or use_interp_fast) and primitives are
    all true -- this function doesn't know how to read sensor data
    (deliberately; see hydrodynamics.pyx).
    """
    cdef unsigned int i

    if not use_buoyancy:
        for i in range(3):
            buoyancy[i] = 0
            buoyancy_torque[i] = 0
        return

    if (use_exact_cob or use_interp_fast) and primitives:
        pos_urdf[0] = pos_x
        pos_urdf[1] = pos_y
        pos_urdf[2] = pos_z

        if use_interp_fast and interp_state is not None:
            # analytic_fast: anchor every interp_steps, interpolate between
            _compute_buoyancy_mesh_fast_cy(
                primitives=primitives,
                pos_world=pos_urdf,
                com_world=com_position,
                urdf2global_q=urdf2global,
                water_density=water_density,
                surface=surface,
                gravity=gravity,
                force_mesh=force_mesh,
                interp_steps=interp_steps,
                interp_cob_prev=interp_state['cob_prev'],
                interp_cob_curr=interp_state['cob_curr'],
                interp_vol=interp_state['vol'],
                interp_counter=interp_state['counter'],
                force_out=buoyancy,
                buoyancy_torque=buoyancy_torque,
            )
        else:
            # analytic / mesh: exact computation every step
            _compute_buoyancy_mesh_cy(
                primitives=primitives,
                pos_world=pos_urdf,
                com_world=com_position,
                urdf2global_q=urdf2global,
                water_density=water_density,
                surface=surface,
                gravity=gravity,
                force_mesh=force_mesh,
                force_out=buoyancy,
                buoyancy_torque=buoyancy_torque,
            )
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
