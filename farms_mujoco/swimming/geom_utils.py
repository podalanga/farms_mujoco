# """
# geom_utils.py -- build the [(PrimitiveCache, offset_pos, offset_quat), ...]
# list per link that drag.pyx's SwimmingHandler / compute_buoyancy_mesh
# expect (see SwimmingHandler.__init__: self.link_primitives = [
#     gather_link_collision_primitives(...) for link in links
# ]).

# Runs once at model-load time, not in the per-step hot path -- same
# constraint primitive_meshes.build_primitive_cache already documents.

# ASSUMPTIONS -- check these against your actual dm_control/mujoco version
# before trusting this blindly:
#   - geom_type ints follow the standard MuJoCo mjtGeom enum:
#     2=sphere, 3=capsule, 5=cylinder, 6=box (0=plane,1=hfield,4=ellipsoid,
#     7=mesh are not primitive-triangulated here and are skipped with a
#     warning -- ellipsoid/mesh support would need their own builder in
#     primitive_meshes.py).
#   - physics.model.geom_quat is MuJoCo's native (w, x, y, z) order; this
#     is converted to the (x, y, z, w) order used everywhere else in this
#     codebase (see drag.pyx's _quat_xyzw_to_matrix docstring). If your
#     farms_core quat utilities actually expect something else, fix
#     _wxyz_to_xyzw below -- everything downstream depends on this one
#     conversion being right.
#   - geom_pos/geom_size are in the model's native length units; the
#     caller is responsible for the same `/units.meters` scaling already
#     applied to `heights`/`masses` elsewhere in SwimmingHandler. Pass
#     `meters` through and this module applies it consistently to
#     geom_pos and geom_size.
# """

# from __future__ import annotations

# import warnings

# import numpy as np

# from .primitive_meshes import build_primitive_cache

# # mjtGeom ints for the shapes primitive_meshes.py knows how to triangulate.
# _GEOM_TYPE_NAMES = {
#     2: 'sphere',
#     3: 'capsule',
#     5: 'cylinder',
#     6: 'box',
# }


# def _wxyz_to_xyzw(q_wxyz):
#     w, x, y, z = q_wxyz
#     return np.array([x, y, z, w], dtype=float)


# # def gather_link_collision_primitives(physics, link_name, prefix='', meters=1.0):
# #     """Build cached primitive meshes for every collision geom (group==2)
# #     attached to a given link/body, in that link's own frame.

# #     Returns
# #     -------
# #     list of (PrimitiveCache, offset_pos, offset_quat)
# #         offset_pos  : (3,) geom position relative to the link/body origin,
# #                       already scaled by `meters` (same convention as
# #                       SwimmingHandler.heights/masses).
# #         offset_quat : (4,) geom orientation relative to the link/body
# #                       origin, in (x, y, z, w) order.
# #         Empty list if the link has no supported collision primitives
# #         (mesh/ellipsoid geoms, or none at all) -- compute_buoyancy_mesh
# #         already treats an empty/falsy primitives list as "contributes
# #         nothing" (see its `if not primitives:` early return), so this
# #         degrades safely rather than crashing.
# #     """
# #     body_row = physics.named.model.body_mass.axes.row
# #     body_key = prefix + link_name
# #     try:
# #         body_id = body_row.convert_key_item(body_key)
# #     except KeyError:
# #         warnings.warn(f'gather_link_collision_primitives: no body named '
# #                        f'{body_key!r}; returning no primitives for it')
# #         return []

# #     geom_bodyid = physics.model.geom_bodyid
# #     geom_group = physics.model.geom_group
# #     geom_type = physics.model.geom_type
# #     geom_size = physics.model.geom_size
# #     geom_pos = physics.model.geom_pos
# #     geom_quat = physics.model.geom_quat

# #     primitives = []
# #     # for geom_i in range(len(geom_bodyid)):
# #     #     if geom_bodyid[geom_i] != body_id or geom_group[geom_i] != 2:
# #     #         continue

# #     #addition starts here
# #     geom_contype = physics.model.geom_contype
# #     geom_conaffinity = physics.model.geom_conaffinity

# #     for geom_i in range(len(geom_bodyid)):
# #         if geom_bodyid[geom_i] != body_id:
# #             continue
            
# #         # A geom is a collision geom if it has collision properties (contype/conaffinity)
# #         # AND it is not strictly a visual mesh (group 1).
# #         # is_collision = (geom_contype[geom_i] != 0) or (geom_conaffinity[geom_i] != 0)
# #         is_visual_only = (geom_group[geom_i] == 1)
        
# #         if is_visual_only:
# #             continue

# # #addition stops here
# #         gtype = int(geom_type[geom_i])
# #         type_name = _GEOM_TYPE_NAMES.get(gtype)
# #         if type_name is None:
# #             warnings.warn(
# #                 f'gather_link_collision_primitives: link {body_key!r} has a '
# #                 f'collision geom of unsupported type id={gtype} (mesh or '
# #                 f'ellipsoid geoms are not triangulated by primitive_meshes.py '
# #                 f'yet) -- skipping it, buoyancy on this link will be missing '
# #                 f'that geom\'s volume.'
# #             )
# #             continue

# #         size = np.asarray(geom_size[geom_i], dtype=float) / meters
# #         if type_name == 'box':
# #             size = size[:3]
# #         elif type_name == 'sphere':
# #             size = size[:1]
# #         else:  # capsule, cylinder
# #             size = size[:2]

# #         cache = build_primitive_cache(type_name, size)

# #         offset_pos = np.asarray(geom_pos[geom_i], dtype=float) / meters
# #         offset_quat = _wxyz_to_xyzw(geom_quat[geom_i])

# #         primitives.append((cache, offset_pos, offset_quat))

# #     return primitives


# def _primitive_volume(type_name, size):
#     if type_name == 'sphere':
#         r = float(size[0])
#         return (4.0 / 3.0) * np.pi * r**3
#     if type_name == 'box':
#         x, y, z = map(float, size[:3])
#         return x * y * z
#     if type_name == 'cylinder':
#         r, h = map(float, size[:2])
#         return np.pi * r**2 * h
#     if type_name == 'capsule':
#         r, h = map(float, size[:2])
#         return np.pi * r**2 * h + (4.0 / 3.0) * np.pi * r**3
#     return 0.0


# def gather_link_collision_primitives(physics, link_name, prefix='', meters=1.0):
#     """Build cached primitive meshes for every collision geom (group==2)
#     attached to a given link/body, in that link's own frame.
#     """
#     body_row = physics.named.model.body_mass.axes.row
#     body_key = prefix + link_name
#     try:
#         body_id = body_row.convert_key_item(body_key)
#     except KeyError:
#         warnings.warn(
#             f'gather_link_collision_primitives: no body named {body_key!r}; '
#             f'returning no primitives for it'
#         )
#         return []

#     geom_bodyid = physics.model.geom_bodyid
#     geom_group = physics.model.geom_group
#     geom_type = physics.model.geom_type
#     geom_size = physics.model.geom_size
#     geom_pos = physics.model.geom_pos
#     geom_quat = physics.model.geom_quat

#     primitives = []
#     matched_geom_ids = []
#     total_est_volume = 0.0

#     for geom_i in range(len(geom_bodyid)):
#         if geom_bodyid[geom_i] != body_id or geom_group[geom_i] != 2:
#             continue

#         matched_geom_ids.append(int(geom_i))

#         gtype = int(geom_type[geom_i])
#         type_name = _GEOM_TYPE_NAMES.get(gtype)
#         if type_name is None:
#             warnings.warn(
#                 f'gather_link_collision_primitives: link {body_key!r} has a '
#                 f'collision geom of unsupported type id={gtype} '
#                 f'(geom index {geom_i}) -- skipping it'
#             )
#             continue

#         raw_size = np.asarray(geom_size[geom_i], dtype=float)
#         size = raw_size / meters
#         if type_name == 'box':
#             size = size[:3]
#         elif type_name == 'sphere':
#             size = size[:1]
#         else:
#             size = size[:2]

#         est_vol = _primitive_volume(type_name, size)
#         total_est_volume += est_vol

#         warnings.warn(
#             f'gather_link_collision_primitives: body {body_key!r} geom {geom_i} '
#             f'type={type_name} raw_size={raw_size.tolist()} scaled_size={size.tolist()} '
#             f'est_volume={est_vol:.6g}'
#         )

#         cache = build_primitive_cache(type_name, size)

#         offset_pos = np.asarray(geom_pos[geom_i], dtype=float) / meters
#         offset_quat = _wxyz_to_xyzw(geom_quat[geom_i])

#         primitives.append((cache, offset_pos, offset_quat))

#     warnings.warn(
#         f'gather_link_collision_primitives: body {body_key!r} matched '
#         f'{len(matched_geom_ids)} collision geoms, built {len(primitives)} '
#         f'primitive caches, total_est_volume={total_est_volume:.6g}'
#     )

#     if matched_geom_ids:
#         warnings.warn(
#             f'gather_link_collision_primitives: body {body_key!r} geom ids = '
#             f'{matched_geom_ids}'
#         )

#     return primitives



"""
geom_utils.py -- build the [(PrimitiveCache, offset_pos, offset_quat), ...]
list per link that drag.pyx's SwimmingHandler / compute_buoyancy_mesh
expect.

Runs once at model-load time.
"""
from __future__ import annotations
import warnings
import numpy as np
from .primitive_meshes import build_primitive_cache

# MuJoCo geom type integers for the shapes primitive_meshes.py knows how to triangulate.
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

def gather_link_collision_primitives(physics, link_name, prefix='', meters=1.0):
    """Build cached primitive meshes for every collision geom attached to a given link/body.
    
    Returns:
        list of (PrimitiveCache, offset_pos, offset_quat)
    """
    body_row = physics.named.model.body_mass.axes.row
    body_key = prefix + link_name
    
    try:
        body_id = body_row.convert_key_item(body_key)
    except KeyError:
        warnings.warn(f'gather_link_collision_primitives: no body named {body_key!r}; returning no primitives for it')
        return []

    geom_bodyid = physics.model.geom_bodyid
    geom_group = physics.model.geom_group
    geom_type = physics.model.geom_type
    geom_size = physics.model.geom_size
    geom_pos = physics.model.geom_pos
    geom_quat = physics.model.geom_quat
    
    # NEW: Fetch collision flags. In MuJoCo, a geom is a collision geom if 
    # it can collide with anything (contype != 0) or if anything can collide with it (conaffinity != 0).
    geom_contype = physics.model.geom_contype
    geom_conaffinity = physics.model.geom_conaffinity

    primitives = []
    
    print(f"\n--- Scanning geoms for link: {body_key} (body_id={body_id}) ---")
    
    for geom_i in range(len(geom_bodyid)):
        # Only look at geoms attached to this specific body
        if geom_bodyid[geom_i] != body_id:
            continue
            
        gtype = int(geom_type[geom_i])
        group = int(geom_group[geom_i])
        contype = int(geom_contype[geom_i])
        conaffinity = int(geom_conaffinity[geom_i])
        
        # A geom is a collision geom if it has collision properties
        is_collision = (contype != 0) or (conaffinity != 0)
        
        print(f"  Geom {geom_i}: type={gtype} ({_GEOM_TYPE_NAMES.get(gtype, 'unknown')}), "
              f"group={group}, contype={contype}, conaffinity={conaffinity} -> "
              f"{'[INCLUDED]' if is_collision else '[SKIPPED - not a collision geom]'}")

        if not is_collision:
            continue
            
        type_name = _GEOM_TYPE_NAMES.get(gtype)
        if type_name is None:
            warnings.warn(
                f'gather_link_collision_primitives: link {body_key!r} has a '
                f'collision geom of unsupported type id={gtype} -- skipping it.'
            )
            continue
            
        size = np.asarray(geom_size[geom_i], dtype=float) / meters
        if type_name == 'box':
            size = size[:3]
        elif type_name == 'sphere':
            size = size[:1]
        else:  # capsule, cylinder
            size = size[:2]
            
        cache = build_primitive_cache(type_name, size)
        offset_pos = np.asarray(geom_pos[geom_i], dtype=float) / meters
        offset_quat = _wxyz_to_xyzw(geom_quat[geom_i])
        
        primitives.append((cache, offset_pos, offset_quat))
        
    print(f"--- Found {len(primitives)} valid collision primitives for {body_key} ---\n")
    
    return primitives