"""Drag and buoyancy forces for swimming links.

Naming convention used throughout this file, so the two force sources
never get conflated again:
  - `compute_drag_*`      -> drag only
  - `compute_buoyancy_*`  -> buoyancy only
  - `compute_link_forces` -> combines both, in URDF frame, does NOT touch
                              data_xfrc (pure computation, easy to test)
  - `apply_swimming_forces` -> calls compute_link_forces and writes the
                              result into data_xfrc (the only function
                              with that side effect)
"""

include 'types.pxd'

import numpy as np
cimport numpy as np

from farms_core.sensors.data_cy cimport LinkSensorArrayCy, XfrcArrayCy
from farms_core.utils.transform cimport quat_conj, quat_mult, quat_rot

# Pure Python buoyancy module for the exact mesh-based COB method.
# Renamed from get_buoyancy_forces: it computes buoyancy from scratch
# every call, it doesn't "get" a precomputed value.
from .buoyancy import compute_link_buoyancy


cdef void link_swimming_info(
    LinkSensorArrayCy data_links,
    unsigned int iteration,
    int sensor_i,
    DTYPEv1 urdf2global,
    DTYPEv1 com2global,
    DTYPEv1 global2urdf,
    DTYPEv1 com2urdf,
    DTYPEv1 urdf2com,
    DTYPEv1 link_lin_velocity,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
):
    """Link swimming information"""
    # Orientations
    urdf2global[:] = data_links.urdf_orientation_cy(iteration, sensor_i)
    com2global[:] = data_links.com_orientation_cy(iteration, sensor_i)
    quat_conj(urdf2global, global2urdf)
    quat_mult(global2urdf, com2global, com2urdf)
    quat_conj(com2urdf, urdf2com)

    # Compute velocity in CoM frame
    quat_rot(
        data_links.com_lin_velocity_cy(iteration, sensor_i),
        global2urdf,
        quat_c,
        tmp4,
        link_lin_velocity,
    )
    quat_rot(
        data_links.com_ang_velocity_cy(iteration, sensor_i),
        global2urdf,
        quat_c,
        tmp4,
        link_ang_velocity,
    )


cdef void compute_drag_force(
    DTYPEv1 force,
    DTYPEv1 link_velocity,
    DTYPEv1 coefficients,
    DTYPEv1 buoyancy,
    double viscosity,
):
    """Quadratic drag force (URDF frame) plus buoyancy already summed in.
    Renamed from compute_force: this is drag, with buoyancy folded in as
    the last step -- the old name gave no hint either force was involved.
    """
    cdef unsigned int i
    for i in range(3):
        force[i] = link_velocity[i]*link_velocity[i]
        if link_velocity[i] < 0:
            force[i] *= -1
        force[i] *= viscosity*coefficients[i]
        force[i] += buoyancy[i]


cdef void compute_drag_torque(
    DTYPEv1 torque,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 coefficients,
):
    """Quadratic drag torque (URDF frame). Renamed from compute_torque
    for the same reason as compute_drag_force above."""
    cdef unsigned int i
    for i in range(3):
        torque[i] = link_ang_velocity[i]*link_ang_velocity[i]
        if link_ang_velocity[i] < 0:
            torque[i] *= -1
        torque[i] *= coefficients[i]


cdef void compute_buoyancy_analytic_fast(
    double density,
    double water_density,
    double bound_radius,
    double position,
    DTYPEv1 global2urdf,
    double mass,
    double surface,
    double gravity,
    DTYPEv1 buoyancy,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 tmp,
):
    """Single-point bounding-sphere ramp -- fast Cython path used when
    the exact mesh-COB method isn't requested/available for this link.

    NOTE: this duplicates the math in buoyancy.compute_buoyancy_analytic.
    That's intentional -- this cdef version stays in Cython so the
    common case (every non-mesh link, every step) never crosses into
    Python. buoyancy.compute_buoyancy_analytic is kept only as the
    Python-callable reference/fallback for compute_link_buoyancy; if you
    change the ramp formula, change it in BOTH places or delete one.

    `bound_radius` was called `height` in the original -- it's actually
    the primitive's bounding-sphere radius (0.5 * geom_rbound), renamed
    here to match what SwimmingHandler and buoyancy.py now call it.
    """
    if mass > 0 and position - bound_radius < surface:
        tmp[0] = 0
        tmp[1] = 0
        tmp[2] = -water_density*mass*gravity/density*min(
            (surface + bound_radius - position) / (2 * bound_radius),
            1,
        )
        quat_rot(tmp, global2urdf, quat_c, tmp4, buoyancy)
    else:
        for i in range(3):
            buoyancy[i] = 0


cdef bint compute_link_forces(
        double time,
        unsigned int iteration,
        LinkSensorArrayCy data_links,
        unsigned int links_index,
        DTYPEv2 coefficients,
        DTYPEv2 z3,
        DTYPEv2 z4,
        WaterProperties water,
        double mass,
        double bound_radius,
        double density,
        double gravity,
        bint use_buoyancy,
        object primitives,
        bint use_mesh_cob,
        DTYPEv1 force_out,
        DTYPEv1 torque_out,
):
    """Pure computation: drag + buoyancy for one link, combined, in the
    link's URDF frame. Writes into `force_out`/`torque_out` (already
    caller-provided scratch, so no allocation) and `urdf2global_out` (so
    the caller can rotate to world frame without recomputing it).

    Returns False if the link's bounding sphere is entirely above the
    water surface (nothing to compute, force/torque left untouched) and
    True otherwise. This used to be smuggled out as a value confusingly
    named `apply_force` by the caller -- it's a "was this link in
    water-interaction range" flag, not a force.
    """
    cdef unsigned int i
    cdef double pos_x = data_links.array[iteration, links_index, 0]
    cdef double pos_y = data_links.array[iteration, links_index, 1]
    cdef double pos_z = data_links.array[iteration, links_index, 2]
    cdef double surface = water.surface(time, pos_x, pos_y)

    if pos_z - bound_radius > surface:
        return False

    cdef DTYPEv1 force=z3[0], torque=z3[1], buoyancy=z3[2], tmp=z3[3]
    cdef DTYPEv1 link_lin_velocity=z3[4], link_ang_velocity=z3[5]
    cdef DTYPEv1 fluid_velocity_urdf=z3[6]
    cdef DTYPEv1 urdf2global=z4[0], com2global=z4[1]
    cdef DTYPEv1 global2urdf=z4[2], urdf2com=z4[3], com2urdf=z4[4]
    cdef DTYPEv1 quat_c=z4[5], tmp4=z4[6]

    cdef DTYPEv1 pos_urdf=z3[7], com_position=z3[8]
    cdef DTYPEv1 buoyancy_torque=z3[9]

    link_swimming_info(
        data_links=data_links,
        iteration=iteration,
        sensor_i=links_index,
        urdf2global=urdf2global,
        com2global=com2global,
        global2urdf=global2urdf,
        com2urdf=com2urdf,
        urdf2com=urdf2com,
        link_lin_velocity=link_lin_velocity,
        link_ang_velocity=link_ang_velocity,
        quat_c=quat_c,
        tmp4=tmp4,
    )

    # --- Buoyancy ---
    buoyancy_torque[0] = 0
    buoyancy_torque[1] = 0
    buoyancy_torque[2] = 0

    if use_buoyancy:
        if use_mesh_cob and primitives:
            pos_urdf[0], pos_urdf[1], pos_urdf[2] = pos_x, pos_y, pos_z
            com_pos_sensor = data_links.com_position_cy(iteration, links_index)
            com_position[0] = com_pos_sensor[0]
            com_position[1] = com_pos_sensor[1]
            com_position[2] = com_pos_sensor[2]

            res_force, res_torque = compute_link_buoyancy(
                use_mesh_cob=use_mesh_cob,
                primitives=primitives,
                pos_urdf=pos_urdf,
                com_position=com_position,
                urdf2global=urdf2global,
                global2urdf=global2urdf,
                bound_radius=bound_radius,
                mass=mass,
                water_density=water.density(time, pos_x, pos_y, pos_z),
                surface=surface,
                gravity=gravity,
                density=density,
            )
            for i in range(3):
                buoyancy[i] = res_force[i]
                buoyancy_torque[i] = res_torque[i]
        else:
            compute_buoyancy_analytic_fast(
                density=density,
                water_density=water.density(time, pos_x, pos_y, pos_z),
                bound_radius=bound_radius,
                position=pos_z,
                global2urdf=global2urdf,
                mass=mass,
                surface=surface,
                gravity=gravity,
                buoyancy=buoyancy,
                quat_c=quat_c,
                tmp4=tmp4,
                tmp=tmp,
            )
    else:
        for i in range(3):
            buoyancy[i] = 0

    # --- Drag (relative to fluid velocity) ---
    quat_rot(
        vector=water.velocity(time, pos_x, pos_y, pos_z),
        quat=global2urdf,
        quat_c=quat_c,
        tmp4=tmp4,
        out=fluid_velocity_urdf,
    )
    link_lin_velocity[0] -= fluid_velocity_urdf[0]
    link_lin_velocity[1] -= fluid_velocity_urdf[1]
    link_lin_velocity[2] -= fluid_velocity_urdf[2]

    compute_drag_force(
        force=force,
        link_velocity=link_lin_velocity,
        coefficients=coefficients[0],
        buoyancy=buoyancy,
        viscosity=water.viscosity(time, pos_x, pos_y, pos_z),
    )
    compute_drag_torque(
        torque=torque,
        link_ang_velocity=link_ang_velocity,
        coefficients=coefficients[1],
    )
    for i in range(3):
        torque[i] += buoyancy_torque[i]

    # Rotate combined force/torque into world frame for the caller.
    quat_rot(force, urdf2global, quat_c, tmp4, force_out)
    quat_rot(torque, urdf2global, quat_c, tmp4, torque_out)

    return True


cpdef bint apply_swimming_forces(
        double time,
        unsigned int iteration,
        LinkSensorArrayCy data_links,
        unsigned int links_index,
        XfrcArrayCy data_xfrc,
        unsigned int xfrc_index,
        DTYPEv2 coefficients,
        DTYPEv2 z3,
        DTYPEv2 z4,
        WaterProperties water,
        double mass,
        double bound_radius,
        double density,
        double gravity,
        bint use_buoyancy,
        object primitives=None,
        bint use_mesh_cob=False,
):
    """Compute this link's drag+buoyancy and write it into data_xfrc.

    Renamed from drag_forces: the old name promised drag only, but it
    also computed buoyancy and (as a side effect not implied by the
    name at all) wrote directly into the xfrc sensor array. Call sites
    used to do `apply_force = drag_forces(...)`, which read like
    `apply_force` held a force vector -- it's actually the boolean this
    function returns: whether the link was close enough to the water
    surface for anything to have been computed and written.
    """
    cdef bint in_water

    # z3 rows 0-9 are scratch used internally by compute_link_forces.
    # force_out/torque_out need their own dedicated rows (10, 11) so
    # they survive untouched across that call -- see SwimmingHandler
    # .__init__ for the z3 sizing (12 rows instead of the original 11).
    cdef DTYPEv1 _force_out = z3[10]
    cdef DTYPEv1 _torque_out = z3[11]

    in_water = compute_link_forces(
        time=time,
        iteration=iteration,
        data_links=data_links,
        links_index=links_index,
        coefficients=coefficients,
        z3=z3,
        z4=z4,
        water=water,
        mass=mass,
        bound_radius=bound_radius,
        density=density,
        gravity=gravity,
        use_buoyancy=use_buoyancy,
        primitives=primitives,
        use_mesh_cob=use_mesh_cob,
        force_out=_force_out,
        torque_out=_torque_out,
    )
    if not in_water:
        return False

    cdef unsigned int i
    for i in range(3):
        data_xfrc.array[iteration, xfrc_index, i] = _force_out[i]
        data_xfrc.array[iteration, xfrc_index, i+3] = _torque_out[i]
    return True


cdef class WaterProperties:
    """Water properties"""
    def __init__(self):
        super(WaterProperties, self).__init__()

    cdef double surface(self, double t, double x, double y):
        return 0

    cdef double density(self, double t, double x, double y, double z):
        return 1000

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):
        return np.array([0, 0, 0])

    cdef double viscosity(self, double t, double x, double y, double z):
        return 1.0


cdef class WaterPropertiesConstant(WaterProperties):
    """Water properties"""
    cdef double _surface
    cdef double _density
    cdef double _viscosity
    cdef DTYPEv1 _velocity

    def __init__(self, surface, density, velocity, viscosity):
        super(WaterPropertiesConstant, self).__init__()
        self._surface = surface
        self._density = density
        self._velocity = velocity
        self._viscosity = viscosity

    cdef double surface(self, double t, double x, double y):
        return self._surface

    cdef double density(self, double t, double x, double y, double z):
        return self._density

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):
        return self._velocity

    cdef double viscosity(self, double t, double x, double y, double z):
        return self._viscosity

    cpdef void set_velocity(self, double t, double vx, double vy, double vz):
        self._velocity[0] = vx
        self._velocity[1] = vy
        self._velocity[2] = vz


cdef class WaterPropertiesExtension(WaterProperties):
    """Water properties"""
    cdef object _surface
    cdef object _density
    cdef object _viscosity
    cdef object _velocity

    def __init__(self, surface, density, velocity, viscosity):
        super(WaterPropertiesExtension, self).__init__()
        self._surface = surface
        self._density = density
        self._velocity = velocity
        self._viscosity = viscosity

    cdef double surface(self, double t, double x, double y):
        return self._surface(t, x, y)

    cdef double density(self, double t, double x, double y, double z):
        return self._density(t, x, y, z)

    cdef DTYPEv1 velocity(self, double t, double x, double y, double z):
        return self._velocity(t, x, y, z)

    cdef double viscosity(self, double t, double x, double y, double z):
        return self._viscosity(t, x, y, z)


cdef class SwimmingHandler:
    """Swimming handler"""
    cdef object links
    cdef object xfrc
    cdef object animat_options
    cdef unsigned int n_links
    cdef bint drag
    cdef bint sph
    cdef bint buoyancy
    cdef bint use_mesh_cob
    cdef object link_primitives
    cdef WaterProperties water
    cdef double meters
    cdef double newtons
    cdef double torques
    cdef int[:] links_swimming
    cdef unsigned int[:] links_indices
    cdef unsigned int[:] xfrc_indices
    cdef DTYPEv1 masses
    cdef DTYPEv1 bound_radii  # renamed from `heights` -- that's what it is
    cdef DTYPEv1 densities
    cdef DTYPEv2 z3
    cdef DTYPEv2 z4
    cdef DTYPEv3 links_coefficients

    def __init__(self, data, animat_options, arena_options, units, physics, water=None, prefix=''):
        super(SwimmingHandler, self).__init__()
        self.animat_options = animat_options
        self.links = data.sensors.links
        self.xfrc = data.sensors.xfrc
        water_options = arena_options.water
        self.drag = bool(water_options.drag)
        self.sph = getattr(water_options, 'sph', False)
        self.buoyancy = bool(water_options.buoyancy)
        self.meters = float(units.meters)
        self.newtons = float(units.newtons)
        self.torques = float(units.torques)
        self.water = WaterPropertiesConstant(
            surface=float(water_options.height),
            density=float(water_options.density),
            velocity=np.array(water_options.velocity, dtype=float),
            viscosity=float(water_options.viscosity),
        ) if water is None else water

        # z3 grew by one row (12 instead of 11) for apply_swimming_forces'
        # dedicated torque_out scratch -- see that function's comment.
        self.z3 = np.zeros([12, 3])
        self.z4 = np.zeros([7, 4])

        self.use_mesh_cob = True

        links = [
            link
            for link in self.animat_options.morphology.links
            if link.fluid_interaction
        ]
        self.n_links = len(links)
        links_row = physics.named.model.body_mass.axes.row
        self.masses = np.array([
            physics.model.body_mass[links_row.convert_key_item(prefix+link.name)]
            for link in links
        ], dtype=float)/units.kilograms

        self.bound_radii = np.array([
            [
                0.5*physics.model.geom_rbound[geom_i]
                for geom_i in range(len(physics.model.geom_bodyid))
                if links_row.names[physics.named.model.geom_bodyid[geom_i]]
                == prefix+link.name
                and physics.model.geom_group[geom_i] == 2
            ][0]
            for link in links
        ], dtype=float)/self.meters

        self.densities = np.array([link.density for link in links])
        self.xfrc_indices = np.array([
            self.xfrc.names.index(link.name)
            for link in links
        ], dtype=np.uintc)
        self.links_indices = np.array([
            self.links.names.index(link.name)
            for link in links
        ], dtype=np.uintc)
        self.links_coefficients = np.array([
            np.array(link.drag_coefficients)
            for link in links
        ])

        # Per-link collision-primitive caches for the mesh COB method.
        if self.use_mesh_cob:
            from .geom_utils import gather_link_collision_primitives
            self.link_primitives = [
                gather_link_collision_primitives(
                    physics=physics,
                    link_name=link.name,
                    prefix=prefix,
                    meters=self.meters,
                )
                for link in links
            ]
        else:
            self.link_primitives = [None]*self.n_links

        if self.sph:
            self.water._surface = 1e8

    cpdef step(self, double time, unsigned int iteration):
        """Swimming step"""
        cdef unsigned int i
        cdef bint in_water
        if self.drag or self.sph or self.buoyancy:
            for i in range(self.n_links):
                if self.drag or self.buoyancy:
                    in_water = apply_swimming_forces(
                        time=time,
                        iteration=iteration,
                        data_links=self.links,
                        links_index=self.links_indices[i],
                        data_xfrc=self.xfrc,
                        xfrc_index=self.xfrc_indices[i],
                        coefficients=self.links_coefficients[i],
                        z3=self.z3,
                        z4=self.z4,
                        water=self.water,
                        mass=self.masses[i],
                        bound_radius=self.bound_radii[i],
                        density=self.densities[i],
                        gravity=-9.81,
                        use_buoyancy=self.buoyancy,
                        primitives=self.link_primitives[i],
                        use_mesh_cob=self.use_mesh_cob,
                    )

    cpdef set_frame(self, int frame):
        """Set frame"""
        self.frame = frame

    cpdef void set_water_velocity(self, DTYPEv1 velocity):
        """Set water velocity"""
        self.water.set_velocity(vx=velocity[0], vy=velocity[1], vz=velocity[2])
