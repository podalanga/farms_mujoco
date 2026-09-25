#!/usr/bin/env python3
"""Micro-benchmark of the centre-of-buoyancy kernels (ns per geom call).

Heights are spread so that the geom crosses the waterline (worst case),
and separately fully wet/dry (quick reject path).
"""

import json
import sys

import numpy as np

from farms_mujoco.swimming import cob, cob_lut
from farms_mujoco.swimming.cob_lut_build import build_link_lut
from farms_mujoco.swimming.cob_build import (
    SPHERE, ELLIPSOID, CYLINDER, CAPSULE, POLYHEDRON,
    GeomInfo, assemble_cob_geometry, box_triangles,
)

sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/tests')
from test_cob import torus_mesh, random_rotation  # noqa: E402


def main():
    """Main"""
    cases = {
        'sphere': (SPHERE, [0.1, 0, 0], None),
        'ellipsoid': (ELLIPSOID, [0.3, 0.1, 0.05], None),
        'cylinder': (CYLINDER, [0.05, 0.2, 0], None),
        'capsule': (CAPSULE, [0.05, 0.2, 0], None),
        'box (12 tris)': (POLYHEDRON, [0, 0, 0], box_triangles([0.3, 0.1, 0.05])),
        'torus mesh (4096 tris)': (POLYHEDRON, [0, 0, 0], torus_mesh()),
        'torus mesh (65536 tris)': (
            POLYHEDRON, [0, 0, 0], torus_mesh(n_major=256, n_minor=128),
        ),
    }
    rot = random_rotation()
    results = {}
    for name, (kind, size, tris) in cases.items():
        model = cob.CobModel(assemble_cob_geometry([GeomInfo(
            0, 0, kind, np.array(size, float), np.zeros(3), np.eye(3), tris,
        )], n_links=1))
        straddle = np.linspace(-0.04, 0.04, 17)
        ns_cross, _ = cob.benchmark_geom(model, 0, rot, np.zeros(3), straddle, 2000)
        ns_wet, _ = cob.benchmark_geom(model, 0, rot, np.zeros(3), [10.0], 200000)
        geom = GeomInfo(0, 0, kind, np.array(size, float), np.zeros(3), np.eye(3), tris)
        lut = cob_lut.CobLut([0], [build_link_lut([geom], disk_cache=False)], (32, 64))
        ns_lut, _ = cob_lut.benchmark_link(lut, 0, rot, np.zeros(3), straddle, 2000)
        results[name] = {
            'crossing_ns': round(ns_cross, 1), 'wet_ns': round(ns_wet, 1),
            'lut_crossing_ns': round(ns_lut, 1),
        }
        print(
            f'{name:28s} crossing {ns_cross:9.1f} ns   fully wet {ns_wet:6.1f} ns'
            f'   LUT {ns_lut:6.1f} ns'
        )
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=2)


if __name__ == '__main__':
    main()
