from __future__ import annotations
import numpy as np
from .cob_core import submerged_volume_and_centroid_fast

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

# --- Buoyancy Methods ---
# Both of these compute ONLY buoyancy (force, torque) -- no drag, no xfrc
# writes, no rotation-frame bookkeeping beyond what buoyancy itself needs.
# drag.pyx is responsible for combining this with drag and writing results.

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


def compute_buoyancy_mesh(primitives, pos_urdf, com_position, urdf2global, global2urdf, water_density, surface, gravity):
    """Exact method: Stonefish-style per-primitive submerged volume."""
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

        # 3. Exact submerged volume & centroid
        vol_i, centroid_i = submerged_volume_and_centroid_fast(cache, world_pos, world_rot, surface)

        if vol_i <= 0.0:
            continue

        v_total += vol_i
        cwx += vol_i * centroid_i[0]
        cwy += vol_i * centroid_i[1]
        cwz += vol_i * centroid_i[2]

    if v_total <= 0.0:
        return np.zeros(3), np.zeros(3)

    # Center of Buoyancy (World Frame)
    cob_world = np.array([cwx / v_total, cwy / v_total, cwz / v_total])

    # 4. Upward buoyancy force (World Frame)
    force_global = np.array([0.0, 0.0, -water_density * gravity * v_total])

    # 5. Righting torque about CoM (World Frame) -> tau = (CB - CoM) x F
    r_vec = cob_world - com_position
    torque_global = np.cross(r_vec, force_global)

    # 6. Rotate both into URDF frame
    force_urdf = R_global2urdf @ force_global
    torque_urdf = R_global2urdf @ torque_global

    return force_urdf, torque_urdf


def compute_link_buoyancy(use_mesh_cob, primitives, pos_urdf, com_position, urdf2global, global2urdf,
                           bound_radius, mass, water_density, surface, gravity, density):
    """Single entry point called by drag.pyx. Returns (force_urdf, torque_urdf)
    -- buoyancy only. Dispatches to the exact mesh method when primitives
    are available and requested, otherwise the analytic ramp fallback.

    Renamed from `get_buoyancy_forces`: this never "gets" anything that
    already existed, it computes buoyancy from scratch every call.
    """
    if use_mesh_cob and primitives:
        return compute_buoyancy_mesh(
            primitives, pos_urdf, com_position, urdf2global, global2urdf,
            water_density, surface, gravity
        )
    else:
        return compute_buoyancy_analytic(
            bound_radius, pos_urdf[2], global2urdf, mass, water_density, surface, gravity, density
        )
