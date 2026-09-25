#!/usr/bin/env python3
"""Print the buoyancy geometry of an experiment's animat, per link.

For every fluid-interacting link: mass, volume of the buoyant geoms
(collision group 2 and visual group 1), union volume when geoms overlap,
and the resulting buoyancy/weight ratio in the configured water.

Usage: python inspect_buoyancy.py --experiment <dir> [--group 2]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_fluid import resolve_path  # noqa: E402  pylint: disable=wrong-import-position


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    experiment_dir = os.path.abspath(args.experiment)
    os.chdir(experiment_dir)
    sys.path.insert(0, experiment_dir)

    # pylint: disable=import-outside-toplevel
    from farms_core import pylog
    from farms_core.experiment.options import ExperimentOptions
    from farms_mujoco.simulation.mjcf import setup_mjcf_xml, get_prefix
    from farms_mujoco.swimming.cob_build import build_cob_geometry
    from dm_control import mjcf
    pylog.set_level('error')

    options = ExperimentOptions.load('experiment_config.yaml')
    arena = options.arenas[0]
    arena.sdf = resolve_path(arena.sdf, experiment_dir)
    arena.water.sdf = resolve_path(arena.water.sdf, experiment_dir)
    for animat in options.animats:
        animat.sdf = resolve_path(animat.sdf, experiment_dir)
    mjcf_model, _, _ = setup_mjcf_xml(experiment_options=options)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)
    model = physics.model
    units = options.simulation.units
    animat = options.animats[0]
    links = [link for link in animat.morphology.links if link.fluid_interaction]
    body_ids = np.array([
        model.name2id(get_prefix(0) + link.name, 'body') for link in links
    ])
    rho = float(arena.water.density)
    geometries = {
        group: build_cob_geometry(model, body_ids, geom_group=group,
                                  meters=float(units.meters),
                                  report_overlap=True)
        for group in (2, 1)
    }
    print(f'Water density: {rho} kg/m^3')
    print(
        f'{"link":12s} {"mass[kg]":>9s} {"V_coll[cm3]":>11s} {"union":>7s}'
        f' {"V_vis[cm3]":>10s} {"B/W coll":>8s} {"B/W union":>9s} {"B/W vis":>8s}'
    )
    total = np.zeros(4)
    for i, link in enumerate(links):
        mass = model.body_mass[body_ids[i]]/float(units.kilograms)
        v_coll = geometries[2].link_volume[i]
        union = geometries[2].link_overlap[i]
        v_vis = geometries[1].link_volume[i]
        total += [mass, v_coll, v_coll*union, v_vis]
        print(
            f'{link.name:12s} {mass:9.4f} {1e6*v_coll:11.2f} {union:7.3f}'
            f' {1e6*v_vis:10.2f} {rho*v_coll/mass:8.3f}'
            f' {rho*v_coll*union/mass:9.3f} {rho*v_vis/mass:8.3f}'
        )
    mass, v_coll, v_union, v_vis = total
    print(
        f'{"total":12s} {mass:9.4f} {1e6*v_coll:11.2f} {v_union/v_coll:7.3f}'
        f' {1e6*v_vis:10.2f} {rho*v_coll/mass:8.3f} {rho*v_union/mass:9.3f}'
        f' {rho*v_vis/mass:8.3f}'
    )
    print('B/W > 1 floats, < 1 sinks (fully submerged)')


if __name__ == '__main__':
    main()
