#!/usr/bin/env python3
"""Benchmark and regression harness for the swimming (fluid) extension.

Runs an experiment headless, times the SwimmingExtension.before_step call
separately from the full environment step, and records the base-link
trajectory so that different fluid implementations can be compared.

Usage:
    python bench_fluid.py --experiment <dir with experiment_config.yaml>
        [--iterations N] [--out result.json] [--water key=value ...]
"""

import os
import sys
import json
import time
import argparse
import copy

import numpy as np


def parse_value(text):
    """Parse a CLI override value as YAML scalar/list"""
    import yaml
    return yaml.safe_load(text)


def resolve_path(path, start):
    """Resolve a relative model path by searching upwards from start"""
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    tail = path
    while tail.startswith('../'):
        tail = tail[3:]
    directory = start
    while True:
        candidate = os.path.join(directory, tail)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            return path
        directory = parent


def run(experiment_dir, iterations, water_overrides, n_animats=1):
    """Run experiment headless and return timings and trajectory"""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    experiment_dir = os.path.abspath(experiment_dir)
    os.chdir(experiment_dir)
    if experiment_dir not in sys.path:
        sys.path.insert(0, experiment_dir)

    # pylint: disable=import-outside-toplevel
    from farms_core import pylog
    from farms_core.experiment.options import ExperimentOptions
    from farms_sim.simulation import simulation_setup
    from farms_mujoco.swimming.extension import SwimmingExtension
    from farms_mujoco.simulation.mjcf import get_prefix
    pylog.set_level('warning')

    options = ExperimentOptions.load('experiment_config.yaml')
    runtime = options.simulation.runtime
    runtime.headless = True
    runtime.show_progress = False
    runtime.n_iterations = iterations
    runtime.buffer_size = iterations
    arena = options.arenas[0]
    arena.sdf = resolve_path(arena.sdf, experiment_dir)
    water = arena.water
    water.sdf = resolve_path(water.sdf, experiment_dir)
    for animat in options.animats:
        animat.sdf = resolve_path(animat.sdf, experiment_dir)
    for key, value in water_overrides.items():
        setattr(water, key, value)
    if n_animats > 1:
        base = options.animats[0]
        options.animats = []
        for i in range(n_animats):
            animat = copy.deepcopy(base)
            animat.spawn.pose[0] += 0.5*i
            options.animats.append(animat)

    # Time the swimming extension
    timings = []
    original = SwimmingExtension.before_step

    def timed_before_step(self, task, action, physics):
        tic = time.perf_counter_ns()
        original(self, task, action, physics)
        timings.append(time.perf_counter_ns() - tic)

    SwimmingExtension.before_step = timed_before_step
    try:
        tic = time.perf_counter()
        sim = simulation_setup(options)
        setup_time = time.perf_counter() - tic
        physics = sim.physics
        body_id = physics.model.name2id(
            get_prefix(0) + options.animats[0].morphology.links[0].name,
            'body',
        )
        positions = np.zeros([iterations, 3])
        tic = time.perf_counter()
        for iteration in sim.iterator(show_progress=False):
            positions[iteration] = physics.data.xpos[body_id]
        total_time = time.perf_counter() - tic
    finally:
        SwimmingExtension.before_step = original

    timings = np.array(timings, dtype=float)*1e-3  # [us]
    sim_time = iterations*options.simulation.physics.timestep
    return {
        'experiment': experiment_dir,
        'iterations': iterations,
        'n_animats': n_animats,
        'water_overrides': water_overrides,
        'setup_time_s': setup_time,
        'total_time_s': total_time,
        'real_time_factor': sim_time/total_time,
        'step_us_mean': 1e6*total_time/iterations,
        'swimming_calls': int(len(timings)),
        'swimming_us_mean': float(np.mean(timings)) if len(timings) else 0,
        'swimming_us_median': float(np.median(timings)) if len(timings) else 0,
        'swimming_us_total_per_iteration': (
            float(np.sum(timings))/iterations if len(timings) else 0
        ),
        'trajectory': positions.tolist(),
    }


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--n_animats', type=int, default=1)
    parser.add_argument('--out', default='')
    parser.add_argument('--water', nargs='*', default=[])
    args = parser.parse_args()
    overrides = {}
    for item in args.water:
        key, value = item.split('=', 1)
        overrides[key] = parse_value(value)
    out = os.path.abspath(args.out) if args.out else ''
    result = run(args.experiment, args.iterations, overrides, args.n_animats)
    summary = {k: v for k, v in result.items() if k != 'trajectory'}
    summary['final_position'] = result['trajectory'][-1]
    print(json.dumps(summary, indent=2))
    if out:
        with open(out, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile)


if __name__ == '__main__':
    main()
