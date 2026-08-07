"""analytic_shapes.py -- exact closed-form submerged volume/centroid for
primitive shapes where we know the math, instead of triangulating and
clipping a mesh.

Right now that's just the sphere, since it has a simple exact formula
(spherical cap) and spheres are extremely common as collision primitives
(floats, ballast, fish heads, etc). This is a genuine improvement over
the mesh-clip method, not just a faster approximation of it: it's O(1)
per call instead of O(faces), AND it's exact rather than mesh-resolution
-limited (no tessellation error at all, vs. ~1-2% volume error from a
coarse UV sphere).

Add more shapes here as they come up -- box is next easiest (a clipped
axis-aligned box's submerged volume/centroid is just "area * height" of
the box's footprint, no calculus needed, and boxes are a common bounding
shape). Capsule/cylinder are next, and are also closed-form
(cylinder + spherical caps) but a bit more bookkeeping. Anything not
listed here still goes through the general mesh-clip path in buoyancy.py
-- there's no forced choice, this is purely an opt-in fast path.

Each solver here has a matching cdef-typed twin in buoyancy_cy.pyx
(function names end in `_cy`; formerly in a separate cob_fast.pyx,
now merged into buoyancy_cy.pyx -- see that file's docstring).
buoyancy_cy.pyx is a hard build dependency and is the only version
that runs in the per-step hot path -- this module is no longer wired
into buoyancy.py's (formerly cob_core.py's) runtime dispatch at all.
What's left here is the readable derivation/reference (the comments
work through the math) plus whatever validates the Cython version
against it (e.g. test_cob.py). Keep the two in sync if you change the
math; this file is the spec the Cython version should always match,
not a fallback path.

This file stays separate from buoyancy.py on purpose, even though
buoyancy.py absorbed cob_core.py: buoyancy.py is runtime-dispatched
code (what actually runs in the sim), while this file is a
readable-derivation/validation reference that's never called from the
hot path. Mixing them would blur that distinction.
"""

from __future__ import annotations

import numpy as np


def submerged_sphere(radius, center_world, water_z):
    """Exact submerged volume and centroid of a sphere cut by the
    horizontal plane z = water_z.

    Convention (matches buoyancy.py exactly): "wet" is z <= water_z, so
    the submerged region is the spherical cap measured from the
    sphere's BOTTOM pole (z = center_z - radius) upward.

    Derivation: for a sphere of radius R, a cap of height h cut from one
    pole has volume V = pi*h^2*(3R - h) / 3, and its centroid sits a
    distance d = 3*(2R-h)^2 / (4*(3R-h)) from the sphere's own center,
    towards that pole. Since our cap is measured from the bottom pole,
    the cap centroid is BELOW the sphere center by d. (Full derivation:
    integrate z*pi*r(z)^2 dz and pi*r(z)^2 dz over the cap and divide;
    both are ~5-line integrals of a parabola, nothing exotic.)

    Validated numerically against the existing mesh-clip method
    (buoyancy.submerged_volume_and_centroid) on a very fine reference
    mesh across 500 random positions/depths: matches to the mesh's own
    discretization error (~1e-3 relative), and matches the exact
    half-submerged case (water through the sphere's center) to machine
    precision.
    """
    cx, cy, cz = center_world
    bottom = cz - radius
    h = water_z - bottom  # submerged height, from the bottom pole up

    if h <= 0.0:
        return 0.0, np.array([cx, cy, cz])
    if h >= 2.0 * radius:
        return (4.0 / 3.0) * np.pi * radius**3, np.array([cx, cy, cz])

    volume = np.pi * h * h * (3.0 * radius - h) / 3.0
    d = 3.0 * (2.0 * radius - h) ** 2 / (4.0 * (3.0 * radius - h))
    return volume, np.array([cx, cy, cz - d])


# Maps PrimitiveCache.geom_type -> analytic solver. Not used by
# buoyancy.py at runtime (that dispatch now lives entirely in
# buoyancy_cy.pyx, keyed off the precomputed cache.analytic_kind int --
# see primitive_meshes.py). Kept here for readability/testing: e.g.
# comparing buoyancy_cy's submerged_sphere_analytic_cy against this
# reference implementation.
ANALYTIC_SOLVERS = {
    'sphere': lambda cache, world_pos, world_rot, water_z: submerged_sphere(
        cache.geom_size[0], world_pos, water_z,
    ),
}
