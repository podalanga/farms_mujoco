"""hydrodynamics.pyx -- orchestration layer for swimming forces.

This is the third piece of what used to be a single drag.pyx:
  - drag.pyx          -- drag force/torque only
  - buoyancy_cy.pyx    -- buoyancy force/torque only
  - hydrodynamics.pyx  (this file) -- everything that combines them:
      * link_swimming_info      : per-link kinematic state (shared by
                                   both drag and buoyancy)
      * compute_link_forces     : combines drag + buoyancy for one
                                   link, URDF frame, pure computation
                                   (no side effects, easy to test)
      * apply_swimming_forces   : calls compute_link_forces and writes
                                   the result into data_xfrc (the only
                                   function with that side effect)
      * WaterProperties and subclasses
      * SwimmingHandler          : owns per-simulation config (drag/
                                   buoyancy/sph toggles, cob_method,
                                   mesh resolution, primitive caches)
                                   and drives the per-step loop.

Nothing here does its own physics math -- it calls into drag.pyx's
compute_link_drag_fast and buoyancy_cy.pyx's compute_link_buoyancy_fast
(both cimported below, so these are direct C-level calls with no Python
overhead, same as if the code were still all in one file).
"""

include 'types.pxd'

import numpy as np
cimport numpy as np

from farms_core.sensors.data_cy cimport LinkSensorArrayCy, XfrcArrayCy
from farms_core.utils.transform cimport quat_conj, quat_mult, quat_rot

from .drag cimport compute_link_drag_fast
from .buoyancy_cy cimport compute_link_buoyancy_fast

from .cob_options import CobOptions


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
    """Link swimming information -- kinematic state shared by both the
    drag and buoyancy computations for one link. Unchanged from the
    original single-file drag.pyx."""
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
        bint use_exact_cob,
        bint force_mesh,
        DTYPEv1 force_out,
        DTYPEv1 torque_out,
):
    """Pure computation: drag + buoyancy for one link, combined, in the
    link's URDF frame. Writes into `force_out`/`torque_out` (already
    caller-provided scratch, so no allocation).

    Returns False if the link's bounding sphere is entirely above the
    water surface (nothing to compute, force/torque left untouched) and
    True otherwise.

    `use_exact_cob` / `force_mesh` come from SwimmingHandler's
    `cob_method` ('ramp' / 'mesh' / 'analytic', see CobOptions and
    SwimmingHandler.__init__ below) -- resolved into two bints once at
    setup time rather than compared as a string here, since this runs
    every link every step.

    This function itself now does almost no physics -- it gathers
    kinematic state via link_swimming_info, delegates buoyancy to
    buoyancy_cy.compute_link_buoyancy_fast and drag to
    drag.compute_link_drag_fast, then combines and rotates the result
    to world frame. All three cross-module calls are direct C calls
    (cimported via .pxd), not Python calls.
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

    cdef double water_density = water.density(time, pos_x, pos_y, pos_z)

    # com_position is only meaningful for the exact COB methods (torque
    # is taken about it); only pay for the sensor lookup when it'll
    # actually be used.
    if use_buoyancy and use_exact_cob and primitives:
        com_pos_sensor = data_links.com_position_cy(iteration, links_index)
        com_position[0] = com_pos_sensor[0]
        com_position[1] = com_pos_sensor[1]
        com_position[2] = com_pos_sensor[2]

    compute_link_buoyancy_fast(
        use_buoyancy=use_buoyancy,
        use_exact_cob=use_exact_cob,
        force_mesh=force_mesh,
        primitives=primitives,
        density=density,
        water_density=water_density,
        bound_radius=bound_radius,
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=pos_z,
        global2urdf=global2urdf,
        urdf2global=urdf2global,
        mass=mass,
        surface=surface,
        gravity=gravity,
        buoyancy=buoyancy,
        buoyancy_torque=buoyancy_torque,
        com_position=com_position,
        pos_urdf=pos_urdf,
        quat_c=quat_c,
        tmp4=tmp4,
        tmp=tmp,
    )

    compute_link_drag_fast(
        force=force,
        torque=torque,
        link_lin_velocity=link_lin_velocity,
        link_ang_velocity=link_ang_velocity,
        fluid_velocity_world=water.velocity(time, pos_x, pos_y, pos_z),
        global2urdf=global2urdf,
        quat_c=quat_c,
        tmp4=tmp4,
        fluid_velocity_urdf=fluid_velocity_urdf,
        coefficients=coefficients,
        buoyancy=buoyancy,
        viscosity=water.viscosity(time, pos_x, pos_y, pos_z),
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
        bint use_exact_cob=False,
        bint force_mesh=False,
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
    # force_out/torque_out need their own dedicated rows (10, 11).
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
        use_exact_cob=use_exact_cob,
        force_mesh=force_mesh,
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
    cdef bint use_exact_cob
    cdef bint force_mesh
    cdef object cob_method
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
        # dedicated torque_out scratch.
        self.z3 = np.zeros([12, 3])
        self.z4 = np.zeros([7, 4])

        # --- Center-of-buoyancy method + mesh resolution ---
        # See cob_options.py's module docstring for the config-file
        # shapes this reads (flat `cob_*` fields on water_options, or a
        # nested `cob` block -- both work, nested takes priority).
        # Resolved once here into the two bints compute_link_forces
        # actually branches on every link every step, so that hot path
        # never does a string comparison.
        cob_options = CobOptions.from_water_options(water_options)
        self.cob_method = cob_options.method
        self.use_exact_cob = self.cob_method != 'ramp'
        self.force_mesh = self.cob_method == 'mesh'
        mesh_resolution = cob_options.to_mesh_resolution()

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

        # Per-link collision-primitive caches, needed by both the
        # 'mesh' and 'analytic' cob_methods (analytic still needs the
        # cache per primitive to know its geom_type/geom_size, even
        # when it skips the mesh itself -- see PrimitiveCache in
        # primitive_meshes.py).
        if self.use_exact_cob:
            from .geom_utils import gather_link_collision_primitives
            self.link_primitives = [
                gather_link_collision_primitives(
                    physics=physics,
                    link_name=link.name,
                    prefix=prefix,
                    meters=self.meters,
                    mesh_resolution=mesh_resolution,
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
                        use_exact_cob=self.use_exact_cob,
                        force_mesh=self.force_mesh,
                    )
