"""
geom_utils.py -- build the [(PrimitiveCache, offset_pos, offset_quat), ...]
list per link that drag.pyx's SwimmingHandler / compute_buoyancy_mesh
expect (see SwimmingHandler.__init__: self.link_primitives = [
    gather_link_collision_primitives(...) for link in links
]).

Runs once at model-load time, not in the per-step hot path -- same
constraint primitive_meshes.build_primitive_cache already documents.

ASSUMPTIONS -- check these against your actual dm_control/mujoco version
before trusting this blindly:
  - geom_type ints follow the standard MuJoCo mjtGeom enum:
    2=sphere, 3=capsule, 5=cylinder, 6=box (0=plane,1=hfield,4=ellipsoid,
    7=mesh are not primitive-triangulated here and are skipped with a
    warning -- ellipsoid/mesh support would need their own builder in
    primitive_meshes.py).
  - A geom counts as a "collision geom" for buoyancy purposes iff
    geom_group == 2. This MUST match the criterion SwimmingHandler.heights
    uses in drag.pyx (`physics.model.geom_group[geom_i] == 2`) -- if the
    two ever disagree, self.heights (used for the analytic fallback and
    the whole-link early-out bound_radius test) and self.link_primitives
    (used for the mesh COB) silently describe different sets of geoms.
    Do NOT switch this to contype/conaffinity: MuJoCo bodies routinely
    ship with contype=0, conaffinity=0 on their real collision geoms
    specifically to disable self-collision while keeping the geom as the
    physically-meaningful collision shape -- filtering on those flags
    drops exactly the geoms this module needs, with no error, just a
    silent fallback to the old analytic ramp in get_buoyancy_forces.
  - physics.model.geom_quat is MuJoCo's native (w, x, y, z) order; this
    is converted to the (x, y, z, w) order used everywhere else in this
    codebase (see drag.pyx's _quat_xyzw_to_matrix docstring). If your
    farms_core quat utilities actually expect something else, fix
    _wxyz_to_xyzw below -- everything downstream depends on this one
    conversion being right.
  - geom_pos/geom_size are in the model's native length units; the
    caller is responsible for the same `/units.meters` scaling already
    applied to `heights`/`masses` elsewhere in SwimmingHandler. Pass
    `meters` through and this module applies it consistently to
    geom_pos and geom_size.
"""

from __future__ import annotations

import warnings

import numpy as np

from .primitive_meshes import build_primitive_cache

# mjtGeom ints for the shapes primitive_meshes.py knows how to triangulate.
_GEOM_TYPE_NAMES = {
    2: 'sphere',
    3: 'capsule',
    5: 'cylinder',
    6: 'box',
}


def _wxyz_to_xyzw(q_wxyz):
    """Convert MuJoCo's native (w, x, y, z) quaternion to (x, y, z, w)."""
    w, x, y, z = q_wxyz
    return np.array([x, y, z, w], dtype=float)


def gather_link_collision_primitives(physics, link_name, prefix='', meters=1.0, mesh_resolution=None):
    """Build cached primitive meshes for every collision geom (group==2)
    attached to a given link/body, in that link's own frame.

    `mesh_resolution`: optional primitive_meshes.MeshResolution
    controlling tessellation density for the mesh-clip fallback path
    (irrelevant for shapes handled by a closed form -- currently
    spheres -- unless cob_method='mesh' forces the mesh path for them
    too). Defaults to primitive_meshes.DEFAULT_MESH_RESOLUTION if None.

    Returns
    -------
    list of (PrimitiveCache, offset_pos, offset_quat)
        offset_pos  : (3,) geom position relative to the link/body origin,
                      already scaled by `meters` (same convention as
                      SwimmingHandler.heights/masses).
        offset_quat : (4,) geom orientation relative to the link/body
                      origin, in (x, y, z, w) order.
        Empty list if the link has no supported collision primitives
        (mesh/ellipsoid geoms, or none at all) -- compute_buoyancy_mesh
        already treats an empty/falsy primitives list as "contributes
        nothing" (see its `if not primitives:` early return), so this
        degrades safely rather than crashing. get_buoyancy_forces falls
        back to the analytic ramp method in that case -- if that's not
        what you expect for a given link, check geom_group on its
        collision geom first (see module docstring).
    """
    body_row = physics.named.model.body_mass.axes.row
    body_key = prefix + link_name
    try:
        body_id = body_row.convert_key_item(body_key)
    except KeyError:
        warnings.warn(f'gather_link_collision_primitives: no body named '
                       f'{body_key!r}; returning no primitives for it')
        return []

    geom_bodyid = physics.model.geom_bodyid
    geom_group = physics.model.geom_group
    geom_type = physics.model.geom_type
    geom_size = physics.model.geom_size
    geom_pos = physics.model.geom_pos
    geom_quat = physics.model.geom_quat

    primitives = []
    matched_geom_ids = []

    for geom_i in range(len(geom_bodyid)):
        if geom_bodyid[geom_i] != body_id or geom_group[geom_i] != 2:
            continue

        matched_geom_ids.append(int(geom_i))

        gtype = int(geom_type[geom_i])
        type_name = _GEOM_TYPE_NAMES.get(gtype)
        if type_name is None:
            warnings.warn(
                f'gather_link_collision_primitives: link {body_key!r} has a '
                f'collision geom of unsupported type id={gtype} (mesh or '
                f'ellipsoid geoms are not triangulated by primitive_meshes.py '
                f'yet) -- skipping it, buoyancy on this link will be missing '
                f'that geom\'s volume.'
            )
            continue

        size = np.asarray(geom_size[geom_i], dtype=float) / meters
        if type_name == 'box':
            size = size[:3]
        elif type_name == 'sphere':
            size = size[:1]
        else:  # capsule, cylinder
            size = size[:2]

        cache = build_primitive_cache(type_name, size, mesh_resolution)

        offset_pos = np.asarray(geom_pos[geom_i], dtype=float) / meters
        offset_quat = _wxyz_to_xyzw(geom_quat[geom_i])

        primitives.append((cache, offset_pos, offset_quat))

    if not matched_geom_ids:
        warnings.warn(
            f'gather_link_collision_primitives: link {body_key!r} has no '
            f'group==2 collision geoms -- mesh COB will get no primitives '
            f'for this link and get_buoyancy_forces will silently fall '
            f'back to the analytic ramp method for it.'
        )

    return primitives
