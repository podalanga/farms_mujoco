"""Build script for the three Cython extensions:

    buoyancy_cy.pyx       -- buoyancy only: submerged-volume/centroid
                              (COB) math plus buoyancy force/torque.
                              (note: NOT the same module as buoyancy.py
                              -- see that file's docstring for why the
                              _cy suffix matters)
    drag.pyx               -- drag only
    hydrodynamics.pyx      -- orchestration (SwimmingHandler, WaterProperties,
                               compute_link_forces, apply_swimming_forces)

Usage (from the directory containing all three .pyx files, alongside
their .py/.pxd companions -- primitive_meshes.py, analytic_shapes.py,
buoyancy.py, geom_utils.py, cob_options.py, drag.pxd, buoyancy_cy.pxd):

    python setup_hydrodynamics.py build_ext --inplace

Order in the ext_modules list below doesn't matter -- Cython resolves
cross-module cdef calls (hydrodynamics.pyx calling into drag.pyx and
buoyancy_cy.pyx) via their .pxd declarations at cythonize time, and via
a capsule-based lookup at import time, not via link order. Both
extensions still need to exist and be importable at runtime, though:
don't ship hydrodynamics.*.so without drag.*.so and buoyancy_cy.*.so
next to it.

If your project already has a setup.py that cythonizes other .pyx
files, the cleanest move is folding this file's list into that same
cythonize(...) call rather than keeping a second build script around --
this one exists so you can build and benchmark this set on its own.
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        [
            "drag.pyx",
            "buoyancy_cy.pyx",
            "hydrodynamics.pyx",
        ],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
)
