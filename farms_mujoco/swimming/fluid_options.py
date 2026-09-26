"""Fluid model options, read from the arena `water` options.

All fields are optional so existing configurations keep working. They can
be given flat on `water` or grouped in a nested `cob` block (which takes
priority), e.g.:

    water:
      drag: true
      buoyancy: true
      cob_method: exact        # exact | lut | ramp
      cob_geom_group: 2        # 2: collision geoms, 1: visual geoms
      cob_overlap: ignore      # ignore | scale (union of overlapping geoms)
      cob_lut_resolution: [32, 64]  # [directions per side, depths]
      drag_implicit: false     # semi-implicit quadratic drag
      fluid_model: legacy      # legacy | ellipsoid
      ellipsoid_fit: mvee      # mvee | inertia
      ellipsoid_coefficients: [1.0, 1.0, 2.5]  # [form, viscous, rotational]
      dynamic_viscosity: 1.0e-3  # [Pa.s] for the ellipsoid viscous drag
      added_mass: off          # off | explicit | implicit

cob_method:
- exact: exact per-geom submerged volume and centroid (spheres,
  ellipsoids, cylinders and capsules in closed form, boxes and meshes
  with a BVH, see cob.pyx).
- lut: per-link lookup table built at load time from a voxelisation of
  the union of the link's geoms, O(1) per link and overlap-free.
- ramp: legacy bounding-sphere ramp based on the link mass and density.

The legacy values 'analytic', 'analytical', 'analytic_fast' and 'mesh' map
to 'exact'.
"""

from dataclasses import dataclass, field

COB_METHODS = ('exact', 'lut', 'ramp')
COB_ALIASES = {
    'analytic': 'exact', 'analytical': 'exact', 'analytic_fast': 'exact',
    'mesh': 'exact',
}
FLUID_MODELS = ('legacy', 'ellipsoid')
ELLIPSOID_FITS = ('mvee', 'inertia')
ADDED_MASS = ('off', 'explicit', 'implicit')


def _get(source, name, default):
    """Read a field from an options object or a dict"""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _check(name, value, allowed):
    if value not in allowed:
        raise ValueError(f'{name}={value!r} not in {allowed}')
    return value


@dataclass
class FluidOptions:
    """Resolved fluid options"""
    cob_method: str = 'exact'
    cob_geom_group: int = 2
    cob_overlap: str = 'ignore'
    cob_lut_resolution: list = field(default_factory=lambda: [32, 64])
    drag_implicit: bool = False
    fluid_model: str = 'legacy'
    ellipsoid_fit: str = 'mvee'
    ellipsoid_coefficients: list = field(default_factory=lambda: [1.0, 1.0, 2.5])
    dynamic_viscosity: float = 1.0e-3
    added_mass: str = 'off'

    def __post_init__(self):
        self.cob_method = COB_ALIASES.get(self.cob_method, self.cob_method)
        _check('cob_method', self.cob_method, COB_METHODS)
        _check('cob_overlap', self.cob_overlap, ('ignore', 'scale'))
        _check('fluid_model', self.fluid_model, FLUID_MODELS)
        _check('ellipsoid_fit', self.ellipsoid_fit, ELLIPSOID_FITS)
        if self.added_mass is False or self.added_mass is None:
            self.added_mass = 'off'
        _check('added_mass', self.added_mass, ADDED_MASS)
        self.cob_geom_group = int(self.cob_geom_group)
        self.cob_lut_resolution = [int(n) for n in self.cob_lut_resolution]
        self.dynamic_viscosity = float(self.dynamic_viscosity)
        self.ellipsoid_coefficients = [
            float(c) for c in self.ellipsoid_coefficients
        ]

    @classmethod
    def from_water_options(cls, water):
        """Build from arena_options.water (flat fields or `cob` block)"""
        block = _get(water, 'cob', None)
        kwargs = {}
        for name in cls.__dataclass_fields__:
            value = _get(water, name, None)
            if name.startswith('cob_'):
                value = _get(block, name[4:], value)
            if value is not None:
                kwargs[name] = value
        return cls(**kwargs)
