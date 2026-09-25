# Swimming hydrodynamics

Fluid forces (buoyancy, drag, added mass) on the links of swimming animats,
computed in C once per physics step and applied through MuJoCo's
`xfrc_applied`. Everything runs on a single core with no allocation in the
simulation loop, so that many environments can run in parallel (e.g. for
reinforcement learning).

## Configuration

Links opt in with `fluid_interaction: true` (animat config). Water and
model options live in the arena config:

```yaml
water:
  drag: true
  buoyancy: true
  height: 0                    # Surface height [m]
  density: 1000.0              # [kg/m^3]
  viscosity: 1.0               # Scale of the quadratic drag
  velocity: [0, 0, 0]          # Or velocity maps, see extension.py
  # All below are optional (defaults shown)
  cob_method: exact            # exact | lut | ramp
  cob_geom_group: 2            # 2: collision geoms, 1: visual meshes
  cob_overlap: ignore          # ignore | scale
  cob_lut_resolution: [32, 64] # LUT directions per side, depths
  drag_implicit: false         # Semi-implicit legacy quadratic drag
  fluid_model: legacy          # legacy | ellipsoid
  ellipsoid_fit: mvee          # mvee | inertia
  ellipsoid_coefficients: [1.0, 1.0, 2.5]  # form, viscous, rotational
  dynamic_viscosity: 1.0e-3    # [Pa.s] ellipsoid viscous drag
  added_mass: off              # off | implicit | explicit
```

The previous values `cob_method: analytic | analytic_fast | mesh` map to
`exact`. The other previous `cob_*` keys are accepted and ignored.

## Centre of buoyancy (`cob.pyx`, `cob_lut.pyx`)

Buoyancy is `-rho V g` applied at the centre of buoyancy (V: submerged
volume). It is applied as a force plus the torque `(c_b - c_m) x F` at the
link CoM.

`cob_method: exact` computes the submerged volume and centroid of every
buoyant geom exactly, reading the geom poses directly from MuJoCo:

| Geom | Method | Cost when crossing the surface |
|---|---|---|
| sphere | spherical cap, closed form | ~6 ns |
| ellipsoid | affine map to a unit-sphere cap, closed form | ~16 ns |
| cylinder | integral of circular segments along the axis, closed form | ~60 ns |
| capsule | exact cylinder + Gauss-Legendre hemispheres (~1e-10) | ~65 ns |
| box, mesh | BVH with divergence-theorem node moments | ~170 ns (box) |

Fully wet or dry geoms cost ~8 ns. For meshes, the tetrahedra apex is put on
the water plane so only the wet surface is needed. Each BVH node stores
`S0 = sum det(a,b,c)`, `S1 = sum N`, `m = sum det*s` and `M = sum s N^T`
(`N = a x b + b x c + c x a`, `s = a + b + c`). A fully wet node then
contributes `6V = S0 - S1.p` and `24(V c) = m - M p + p (S0 - S1.p)` in
O(1), and only the triangles in leaves crossing the waterline are clipped.
The cost is O(k + log n) for k triangles near the waterline. Any closed
mesh (including non-convex ones) is exact. Non-watertight meshes fall back
to their convex hull.

Irregular bodies: `cob_geom_group: 1` uses the visual meshes of the links
instead of their collision primitives.

`cob_method: lut` uses one lookup table per link, which is O(1) (~60 ns
per link) whatever the number and shape of its geoms. The tables hold V
and the first moment for an octahedral grid of water directions and
depths. They are sampled with the exact kernels, or from a voxelisation of
the union of the geoms when they overlap, so overlapping geoms are only
counted once. Errors are about 1% of the link volume. The tables are cached
in memory and in `~/.cache/farms_mujoco/cob_lut`.

Overlapping collision geoms are counted twice by `exact`
(`cob_overlap: ignore`, as before). `cob_overlap: scale` rescales them by
the union fraction. `lut` uses the union.

`cob_method: ramp` is the legacy bounding-sphere ramp using the link mass
and density.

## Drag and added mass

`fluid_model: legacy`: per-axis quadratic drag in the link frame,
`F_i = viscosity * c_i v_i |v_i|` and `T_i = c'_i w_i |w_i|`, with the
link `drag_coefficients` `[[c], [c']]` (negative). It is optionally
semi-implicit (`drag_implicit`, stable for large timesteps).

`fluid_model: ellipsoid` (`ellipsoid_model.pyx`, `ellipsoid_fit.py`):
each link is approximated by an ellipsoid, either the minimum-volume
enclosing ellipsoid of its geoms (`mvee`, as Stonefish) or its inertia
ellipsoid (`inertia`, as MuJoCo). In the ellipsoid frame:

- form drag `-1/2 rho C_form A_proj(v) |v| v` with
  `A_proj(u) = pi sqrt((bc u_x)^2 + (ac u_y)^2 + (ab u_z)^2)`
- viscous drag `-6 pi mu r v` and `-8 pi mu r^3 w`
- rotational drag `-rho C_rot I_D,i w_i |w|` with
  `I_D,i = 8pi/15 r_i max(r_j, r_k)^4`
- added mass from Lamb's coefficients: `m_i = rho V k_i/(2 - k_i)` and the
  rotational `I_i`. The Kirchhoff terms `(m o v) x w` and
  `(m o v) x v + (I o w) x w` (Munk moment) are always applied. The
  acceleration terms are handled by `added_mass`:
  - `implicit` (recommended): the body mass and inertia are augmented by
    the added mass times the submerged fraction, as Stonefish does. This is
    unconditionally stable, and the extra weight is cancelled.
  - `explicit`: uses filtered finite-difference accelerations. It is
    unstable when the added mass exceeds about half the link mass.

All terms are scaled by the submerged fraction from the CoB computation.

## Files

| File | Content |
|---|---|
| `hydrodynamics.pyx` | `SwimmingHandler` (per-step loop) and water properties |
| `extension.py` | `SwimmingExtension` (MuJoCo task extension), water maps |
| `cob.pyx`, `cob_build.py` | Exact CoB kernels and their load-time geometry |
| `cob_lut.pyx`, `cob_lut_build.py` | CoB lookup tables |
| `drag.pyx` | Legacy per-axis quadratic drag kernels |
| `ellipsoid_model.pyx`, `ellipsoid_fit.py` | Ellipsoid drag and added mass |
| `fluid_options.py` | Options parsing |

Tests: `tests/` (pytest). Benchmarks: `benchmarks/bench_cob.py` (kernels),
`benchmarks/bench_fluid.py` (full simulations) and
`benchmarks/inspect_buoyancy.py` (per-link buoyancy budget of a model).
