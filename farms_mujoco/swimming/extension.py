"""Swimming extension: applies hydrodynamic forces to an animat"""

import os

import numpy as np
from imageio import imread

from farms_core import pylog
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions, ArenaOptions
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.extensions import AnimatExtension
from farms_mujoco.simulation.mjcf import get_prefix
from farms_mujoco.swimming.hydrodynamics import (
    SwimmingHandler,
    WaterPropertiesMaps,
)


def load_water_maps(water_options):
    """Water velocity maps from images.

    water.velocity is [vx_min, vy_min, _, vx_max, vy_max, _, x_min, y_min,
    x_max, y_max] and water.maps the paths to the x and y velocity images.
    """
    water_velocity = water_options.velocity
    paths = [os.path.expandvars(path) for path in water_options.maps]
    for path in paths[:2]:
        assert os.path.isfile(path), (
            f'{path=} is not pointing to an existing file'
            f'\nNote: {water_velocity=}'
        )
    pngs = [np.flipud(imread(path)).T for path in paths[:2]]
    vels = []
    for png_i, png in enumerate(pngs):
        info = np.iinfo(png.dtype)
        vels.append(
            (png.astype(np.double) - info.min)
            * (water_velocity[png_i+3] - water_velocity[png_i])
            / (info.max - info.min)
            + water_velocity[png_i]
        )
    pylog.debug(
        'Water velocities loaded: %s'
        '\nVelX: Min=%s [m/s] Max=%s [m/s]'
        '\nVelY: Min=%s [m/s] Max=%s [m/s]',
        paths, vels[0].min(), vels[0].max(), vels[1].min(), vels[1].max(),
    )
    return {
        'pos_min': np.array(water_velocity[6:8]),
        'pos_max': np.array(water_velocity[8:10]),
        'vel_x': +vels[0],
        'vel_y': -vels[1],
    }


class SwimmingExtension(AnimatExtension):
    """Swimming extension"""

    def __init__(
            self,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
            arena_options: ArenaOptions,
            substep=True,
            water_properties=None,
    ):
        super().__init__(substep=substep)
        self.animat_i = animat_i
        self.animat_data = animat_data
        self.animat_options = animat_options
        self.arena_options = arena_options
        self._handler: SwimmingHandler = None
        self._water_properties = water_properties
        water = arena_options.water
        self.constant_velocity: bool = len(water.velocity) == 3
        if not self.constant_velocity and water_properties is None:
            self.water_maps = load_water_maps(water)
            self._water_properties = WaterPropertiesMaps(
                surface=float(water.height),
                density=float(water.density),
                viscosity=float(water.viscosity),
                vel_x=self.water_maps['vel_x'],
                vel_y=self.water_maps['vel_y'],
                pos_min=self.water_maps['pos_min'],
                pos_max=self.water_maps['pos_max'],
            )

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        """From options"""
        return cls(
            animat_i=animat_i,
            animat_data=animat_data,
            animat_options=animat_options,
            arena_options=experiment_options.arenas[0],
        )

    @property
    def handler(self) -> SwimmingHandler:
        """Swimming handler"""
        return self._handler

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self._handler = SwimmingHandler(
            data=self.animat_data,
            animat_options=self.animat_options,
            arena_options=self.arena_options,
            units=task.units,
            physics=physics,
            water=self._water_properties,
            prefix=get_prefix(self.animat_i),
        )

    def before_step(self, task, action, physics):
        """Compute and apply the fluid forces (written to xfrc_applied)"""
        del action
        self._handler.step(
            physics.time()/task.units.seconds,
            task.iteration % task.buffer_size,
            task.physics_timestep,
        )
