"""cob_options.py -- configuration for the center-of-buoyancy method.

Read off `arena_options.water` the same permissive way `water_density`
and `water_options.sph` already are elsewhere in this codebase: plain
attribute reads with defaults, so existing config files that don't
mention any of this keep working completely unchanged.

Two config-file shapes are both supported, so you can use whichever
matches how the rest of your config is organized:

1. Flat fields directly on `water` (matches the existing sph/drag/
   buoyancy style):

     water:
       cob_method: analytic
       cob_sphere_n_lat: 16
       cob_sphere_n_lon: 32
       cob_cylinder_n_seg: 20
       cob_capsule_n_lat: 6
       cob_capsule_n_lon: 16

2. A nested `cob` block (takes priority over the flat fields above if
   both are present -- lets you keep the method-switch and its knobs
   grouped together instead of scattered flat fields):

     water:
       cob:
         method: analytic       # 'ramp' | 'mesh' | 'analytic'
         sphere: {n_lat: 16, n_lon: 32}
         cylinder: {n_seg: 20}
         capsule: {n_lat: 6, n_lon: 16}

   `cob` (and its sub-blocks) can be either an object with matching
   attributes or a plain dict -- both work, since YAML configs usually
   deserialize to dicts and dataclass-style configs deserialize to
   objects, and there's no reason this should care which your project
   uses.

method options:
  - 'ramp'     : old single-point bounding-sphere linear ramp. Fast,
                 approximate, no collision primitives needed.
  - 'mesh'     : exact per-primitive submerged volume via triangle-mesh
                 clipping ("tetrahedron method"), for every primitive
                 regardless of shape.
  - 'analytic' (default): exact per-primitive submerged volume, using a
                 closed-form solution where one is known (currently
                 spheres -- O(1), no tessellation error at all -- see
                 analytic_shapes.py) and mesh-clip otherwise. Strictly
                 better than 'mesh' for any scene that includes
                 spheres.

The sphere/cylinder/capsule n_lat/n_lon/n_seg knobs only affect the
mesh-clip fallback (irrelevant for spheres under 'analytic' unless you
also force 'mesh'). Defaults were benchmarked at ~1.6% sphere-volume
error vs. exact, at roughly 3x fewer faces than tessellation this
codebase used to default to -- see primitive_meshes.py's
MeshResolution docstring if you want to trade accuracy for speed
differently.
"""

from __future__ import annotations


def _get(source, name, default):
    """getattr for config objects, __getitem__ for plain dicts, default
    otherwise -- so this works whether a project's config layer
    deserializes YAML into dataclass-style option objects or into plain
    nested dicts, without this module needing to know or care which."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


class CobOptions:
    """Resolved center-of-buoyancy configuration for one
    SwimmingHandler. Build with `CobOptions.from_water_options(...)`
    rather than constructing directly, unless you're calling from
    plain Python/tests."""

    __slots__ = (
        'method',
        'sphere_n_lat', 'sphere_n_lon',
        'cylinder_n_seg',
        'capsule_n_lat', 'capsule_n_lon',
    )

    def __init__(
        self,
        method='analytic',
        sphere_n_lat=16, sphere_n_lon=32,
        cylinder_n_seg=20,
        capsule_n_lat=6, capsule_n_lon=16,
    ):
        if method not in ('ramp', 'mesh', 'analytic'):
            raise ValueError(
                f"cob method={method!r} not recognised -- "
                f"expected one of 'ramp', 'mesh', 'analytic'"
            )
        self.method = method
        self.sphere_n_lat = sphere_n_lat
        self.sphere_n_lon = sphere_n_lon
        self.cylinder_n_seg = cylinder_n_seg
        self.capsule_n_lat = capsule_n_lat
        self.capsule_n_lon = capsule_n_lon

    @classmethod
    def from_water_options(cls, water_options):
        """Build from arena_options.water. See module docstring for the
        two supported config-file shapes (flat cob_* fields, or a
        nested `cob` block which takes priority when present)."""
        cob_block = _get(water_options, 'cob', None)
        sphere_block = _get(cob_block, 'sphere', None)
        cylinder_block = _get(cob_block, 'cylinder', None)
        capsule_block = _get(cob_block, 'capsule', None)

        return cls(
            method=_get(cob_block, 'method',
                        _get(water_options, 'cob_method', 'analytic')),
            sphere_n_lat=_get(sphere_block, 'n_lat',
                               _get(water_options, 'cob_sphere_n_lat', 16)),
            sphere_n_lon=_get(sphere_block, 'n_lon',
                               _get(water_options, 'cob_sphere_n_lon', 32)),
            cylinder_n_seg=_get(cylinder_block, 'n_seg',
                                 _get(water_options, 'cob_cylinder_n_seg', 20)),
            capsule_n_lat=_get(capsule_block, 'n_lat',
                                _get(water_options, 'cob_capsule_n_lat', 6)),
            capsule_n_lon=_get(capsule_block, 'n_lon',
                                _get(water_options, 'cob_capsule_n_lon', 16)),
        )

    def to_mesh_resolution(self):
        """Adapt to primitive_meshes.MeshResolution, the shape
        gather_link_collision_primitives / build_primitive_cache
        actually expect."""
        from .primitive_meshes import MeshResolution
        return MeshResolution(
            sphere_n_lat=self.sphere_n_lat,
            sphere_n_lon=self.sphere_n_lon,
            cylinder_n_seg=self.cylinder_n_seg,
            capsule_n_lat=self.capsule_n_lat,
            capsule_n_lon=self.capsule_n_lon,
        )
