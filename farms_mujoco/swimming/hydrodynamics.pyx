"""Hydrodynamics: buoyancy and drag on the links of an animat.

SwimmingHandler.step() runs once per physics step. It is a single C loop
over the fluid-interacting links which:

1. evaluates the water surface at each link,
2. computes the submerged volume and centre of buoyancy of every link
   (cob.CobModel, reading the geom poses directly from MuJoCo),
3. computes buoyancy and drag in the world frame,
4. writes the wrench at the link CoM into MuJoCo's xfrc_applied and into
   the xfrc sensor buffer (for logging and visualisation).
"""

# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

include 'types.pxd'

import numpy as np
cimport numpy as np

from farms_core.sensors.sensor_convention cimport (
    LINK_COM_POSITION_X,
    LINK_URDF_POSITION_X,
    LINK_URDF_ORIENTATION_X,
    LINK_COM_VELOCITY_LIN_X,
    LINK_COM_VELOCITY_ANG_X,
)

from .cob cimport CobModel
from .cob_lut cimport CobLut
from .drag cimport quadratic_drag, quadratic_drag_implicit
from .ellipsoid_model cimport EllipsoidModel, ADDED_MASS_IMPLICIT
from .fluid_options import FluidOptions

np.import_array()

cdef enum:
    COB_EXACT = 0
    COB_LUT = 1
    COB_RAMP = 2


# ---------------------------------------------------------------------------
# Water properties
# ---------------------------------------------------------------------------

cdef class WaterProperties:
    """Water properties (base class: still water at z=0)"""

    cdef double surface(self, double t, double x, double y):
        return 0

    cdef double density(self, double t, double x, double y, double z):
        return 1000

    cdef void velocity(self, double t, double x, double y, double z, double *out):
        out[0] = 0
        out[1] = 0
        out[2] = 0

    cdef double viscosity(self, double t, double x, double y, double z):
        return 1.0


cdef class WaterPropertiesConstant(WaterProperties):
    """Uniform, time invariant water properties"""
    cdef public double _surface
    cdef double _density
    cdef double _viscosity
    cdef double _velocity[3]

    def __init__(self, surface, density, velocity, viscosity):
        super().__init__()
        self._surface = surface
        self._density = density
        self._viscosity = viscosity
        self.set_velocity(0, velocity[0], velocity[1], velocity[2])

    cdef double surface(self, double t, double x, double y):
        return self._surface

    cdef double density(self, double t, double x, double y, double z):
        return self._density

    cdef void velocity(self, double t, double x, double y, double z, double *out):
        out[0] = self._velocity[0]
        out[1] = self._velocity[1]
        out[2] = self._velocity[2]

    cdef double viscosity(self, double t, double x, double y, double z):
        return self._viscosity

    cpdef void set_velocity(self, double t, double vx, double vy, double vz):
        """Set the water velocity"""
        self._velocity[0] = vx
        self._velocity[1] = vy
        self._velocity[2] = vz


cdef class WaterPropertiesMaps(WaterPropertiesConstant):
    """Constant surface, density and viscosity, with a horizontal water
    velocity field given on a regular grid (nearest cell lookup)"""
    cdef double[:, ::1] _vel_x
    cdef double[:, ::1] _vel_y
    cdef double _pos_min[2]
    cdef double _pos_max[2]

    def __init__(self, surface, density, viscosity, vel_x, vel_y, pos_min, pos_max):
        super().__init__(surface, density, [0, 0, 0], viscosity)
        self._vel_x = np.ascontiguousarray(vel_x, dtype=np.double)
        self._vel_y = np.ascontiguousarray(vel_y, dtype=np.double)
        for i in range(2):
            self._pos_min[i] = pos_min[i]
            self._pos_max[i] = pos_max[i]

    cdef void velocity(self, double t, double x, double y, double z, double *out):
        cdef int ix, iy
        cdef double fx, fy
        out[0] = 0
        out[1] = 0
        out[2] = 0
        if not (self._pos_min[0] < x < self._pos_max[0]
                and self._pos_min[1] < y < self._pos_max[1]):
            return
        fx = (x - self._pos_min[0])/(self._pos_max[0] - self._pos_min[0])
        fy = (y - self._pos_min[1])/(self._pos_max[1] - self._pos_min[1])
        ix = <int>(fx*self._vel_x.shape[0] + 0.5)
        iy = <int>(fy*self._vel_x.shape[1] + 0.5)
        ix = min(ix, self._vel_x.shape[0] - 1)
        iy = min(iy, self._vel_x.shape[1] - 1)
        out[0] = self._vel_x[ix, iy]
        out[1] = self._vel_y[ix, iy]

    def velocity_at(self, double x, double y):
        """Python access for tests"""
        cdef double out[3]
        self.velocity(0, x, y, 0, out)
        return np.array([out[0], out[1], out[2]])


cdef class WaterPropertiesExtension(WaterProperties):
    """Water properties given by Python callbacks (flexible but slow)"""
    cdef object _surface
    cdef object _density
    cdef object _viscosity
    cdef object _velocity

    def __init__(self, surface, density, velocity, viscosity):
        super().__init__()
        self._surface = surface
        self._density = density
        self._velocity = velocity
        self._viscosity = viscosity

    cdef double surface(self, double t, double x, double y):
        return self._surface(t, x, y)

    cdef double density(self, double t, double x, double y, double z):
        return self._density(t, x, y, z)

    cdef void velocity(self, double t, double x, double y, double z, double *out):
        vel = self._velocity(t, x, y, z)
        out[0] = vel[0]
        out[1] = vel[1]
        out[2] = vel[2]

    cdef double viscosity(self, double t, double x, double y, double z):
        return self._viscosity(t, x, y, z)


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------

cdef inline void quat2mat(const double *q, double *R) noexcept nogil:
    """Row-major rotation matrix from an (x, y, z, w) quaternion"""
    cdef double x = q[0], y = q[1], z = q[2], w = q[3]
    R[0] = 1 - 2*(y*y + z*z)
    R[1] = 2*(x*y - z*w)
    R[2] = 2*(x*z + y*w)
    R[3] = 2*(x*y + z*w)
    R[4] = 1 - 2*(x*x + z*z)
    R[5] = 2*(y*z - x*w)
    R[6] = 2*(x*z - y*w)
    R[7] = 2*(y*z + x*w)
    R[8] = 1 - 2*(x*x + y*y)


cdef inline void mat_vec(const double *R, const double *v, double *out) noexcept nogil:
    """out = R v"""
    out[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2]
    out[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2]
    out[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2]


cdef inline void mat_t_vec(const double *R, const double *v, double *out) noexcept nogil:
    """out = R^T v"""
    out[0] = R[0]*v[0] + R[3]*v[1] + R[6]*v[2]
    out[1] = R[1]*v[0] + R[4]*v[1] + R[7]*v[2]
    out[2] = R[2]*v[0] + R[5]*v[1] + R[8]*v[2]


cdef inline void cross(const double *a, const double *b, double *out) noexcept nogil:
    """out = a x b"""
    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

cdef class SwimmingHandler:
    """Computes and applies the fluid forces on the links of an animat"""
    cdef readonly object options
    cdef readonly object cob_geometry
    cdef WaterProperties water
    cdef CobModel cob
    cdef CobLut lut
    cdef EllipsoidModel ellipsoid
    cdef bint drag, buoyancy, drag_implicit, ellipsoid_drag, use_exact
    cdef bint implicit_added_mass
    cdef int cob_method
    cdef readonly int n_links
    cdef double inv_meters, newtons, torques, kilograms, inertia_unit
    cdef double gravity[3]
    cdef int[::1] body_ids
    cdef int[::1] links_indices
    cdef int[::1] xfrc_indices
    cdef double[::1] masses, densities, bound_radii
    cdef double[:, :, ::1] coefficients
    cdef double[::1] surfaces
    cdef readonly double[::1] full_volume   # Full buoyant volume per link
    cdef double[::1] body_mass, body_mass0
    cdef double[:, ::1] body_inertia, body_inertia0, inertia_added
    cdef readonly double[:, ::1] submerged  # [V, V*c] per link (SI)
    cdef DTYPEv3 links_array
    cdef DTYPEv3 xfrc_array
    cdef object _physics_arrays
    cdef double[:, ::1] geom_xpos, geom_xmat, xpos, xmat, xfrc_applied

    def __init__(self, data, animat_options, arena_options, units, physics,
                 water=None, prefix=''):
        super().__init__()
        water_options = arena_options.water
        self.options = FluidOptions.from_water_options(water_options)
        self.drag = bool(water_options.drag)
        self.buoyancy = bool(water_options.buoyancy)
        self.drag_implicit = self.options.drag_implicit
        self.ellipsoid_drag = self.options.fluid_model == 'ellipsoid'
        self.cob_method = {
            'exact': COB_EXACT, 'lut': COB_LUT, 'ramp': COB_RAMP,
        }[self.options.cob_method]
        self.inv_meters = 1/float(units.meters)
        self.kilograms = float(units.kilograms)
        self.inertia_unit = self.kilograms*float(units.meters)**2
        self.newtons = float(units.newtons)
        self.torques = float(units.torques)
        gravity = np.array(physics.model.opt.gravity)/float(units.acceleration)
        for i in range(3):
            self.gravity[i] = gravity[i]
        self.water = WaterPropertiesConstant(
            surface=float(water_options.height),
            density=float(water_options.density),
            velocity=np.array(water_options.velocity[:3], dtype=float),
            viscosity=float(water_options.viscosity),
        ) if water is None else water
        if getattr(water_options, 'sph', False):
            self.water._surface = 1e8

        # Links
        links = [
            link for link in animat_options.morphology.links
            if link.fluid_interaction
        ]
        self.n_links = len(links)
        model = physics.model
        self.body_ids = np.array([
            model.name2id(prefix + link.name, 'body') for link in links
        ], dtype=np.intc)
        sensors = data.sensors
        self.links_indices = np.array([
            sensors.links.names.index(link.name) for link in links
        ], dtype=np.intc)
        self.xfrc_indices = np.array([
            sensors.xfrc.names.index(link.name) for link in links
        ], dtype=np.intc)
        self.links_array = sensors.links.array
        self.xfrc_array = sensors.xfrc.array
        # Original body masses and inertias (the implicit added mass
        # modifies them, and handlers are rebuilt at every episode)
        if not hasattr(physics, '_farms_body_mass0'):
            physics._farms_body_mass0 = np.array(model.body_mass)
            physics._farms_body_inertia0 = np.array(model.body_inertia)
        self.body_mass0 = physics._farms_body_mass0
        self.body_inertia0 = physics._farms_body_inertia0
        model.body_mass[:] = physics._farms_body_mass0
        model.body_inertia[:] = physics._farms_body_inertia0
        self.body_mass = model.body_mass
        self.body_inertia = model.body_inertia
        self.masses = np.array([
            model.body_mass[body_id] for body_id in self.body_ids
        ], dtype=float)/float(units.kilograms)
        self.densities = np.array([link.density for link in links], dtype=float)
        self.coefficients = np.ascontiguousarray([
            np.reshape(link.drag_coefficients, [2, 3]) for link in links
        ], dtype=float).reshape(-1, 2, 3)
        self.bound_radii = np.array([
            max([
                np.linalg.norm(model.geom_pos[geom_i]) + model.geom_rbound[geom_i]
                for geom_i in np.flatnonzero(model.geom_bodyid == body_id)
                if model.geom_group[geom_i] == self.options.cob_geom_group
            ] or [0.0])
            for body_id in self.body_ids
        ], dtype=float)*self.inv_meters
        self.surfaces = np.zeros(self.n_links)
        self.submerged = np.zeros([self.n_links, 4])

        # Buoyancy geometry
        self.use_exact = self.cob_method == COB_EXACT or (
            self.ellipsoid_drag and self.cob_method == COB_RAMP
        )
        if self.use_exact:
            from .cob_build import build_cob_geometry
            self.cob_geometry = build_cob_geometry(
                model=model,
                body_ids=np.asarray(self.body_ids),
                geom_group=self.options.cob_geom_group,
                meters=float(units.meters),
                overlap=self.options.cob_overlap,
            )
            self.cob = CobModel(self.cob_geometry)
            self.full_volume = np.array(self.cob_geometry.link_volume, dtype=float)
        if self.cob_method == COB_LUT:
            from .cob_lut_build import build_cob_lut
            self.lut = build_cob_lut(
                model=model,
                body_ids=np.asarray(self.body_ids),
                geom_group=self.options.cob_geom_group,
                meters=float(units.meters),
                resolution=self.options.cob_lut_resolution,
            )
            self.full_volume = np.array(self.lut.volume, dtype=float)
        if self.ellipsoid_drag:
            from .ellipsoid_model import (
                build_ellipsoid_model, implicit_added_inertia,
            )
            self.ellipsoid = build_ellipsoid_model(
                model=model,
                body_ids=np.asarray(self.body_ids),
                options=self.options,
                meters=float(units.meters),
                kilograms=float(units.kilograms),
                water_density=float(water_options.density),
            )
            self.implicit_added_mass = (
                self.ellipsoid.added_mass == ADDED_MASS_IMPLICIT
            )
            self.inertia_added = implicit_added_inertia(
                self.ellipsoid, model, np.asarray(self.body_ids),
            )

        # MuJoCo state (views on the MjData buffers)
        data_mj = physics.data
        self._physics_arrays = (
            data_mj.geom_xpos, data_mj.geom_xmat, data_mj.xpos, data_mj.xmat,
            data_mj.xfrc_applied,
        )
        self.geom_xpos = data_mj.geom_xpos
        self.geom_xmat = data_mj.geom_xmat
        self.xpos = data_mj.xpos
        self.xmat = data_mj.xmat
        self.xfrc_applied = data_mj.xfrc_applied

    cpdef void step(self, double time, unsigned int iteration, double timestep=0):
        """Compute and apply the fluid forces for this physics step"""
        cdef int li, s_i, i
        cdef double x, y, z
        if not (self.drag or self.buoyancy) or self.n_links == 0:
            return

        # Water surface at each link
        for li in range(self.n_links):
            s_i = self.links_indices[li]
            x = self.links_array[iteration, s_i, LINK_URDF_POSITION_X]
            y = self.links_array[iteration, s_i, LINK_URDF_POSITION_X+1]
            self.surfaces[li] = self.water.surface(time, x, y)

        # Submerged volume and centre of buoyancy of every link
        if self.use_exact:
            self.cob.compute(
                &self.geom_xpos[0, 0], &self.geom_xmat[0, 0],
                self.inv_meters, &self.surfaces[0], &self.submerged[0, 0],
            )
        if self.cob_method == COB_LUT:
            self.lut.compute(
                &self.xpos[0, 0], &self.xmat[0, 0], self.inv_meters,
                &self.surfaces[0], &self.submerged[0, 0],
            )

        for li in range(self.n_links):
            self.link_forces(li, time, iteration, timestep)

    cdef void link_forces(self, int li, double time, unsigned int iteration,
                          double timestep):
        """Fluid wrench on one link, written to xfrc_applied and sensors"""
        cdef int s_i = self.links_indices[li], x_i = self.xfrc_indices[li]
        cdef int body = self.body_ids[li], i
        cdef double *state = &self.links_array[iteration, s_i, 0]
        cdef double *pos = state + LINK_URDF_POSITION_X
        cdef double *com = state + LINK_COM_POSITION_X
        cdef double surface = self.surfaces[li]
        cdef double volume = 0, fraction, density, viscosity, ramp
        cdef double force[3]
        cdef double torque[3]
        cdef double buoyancy[3]
        cdef double arm[3]
        cdef double tmp[3]
        cdef double tmp2[3]
        cdef double vel[3]
        cdef double R[9]
        cdef bint in_water

        # Is the link interacting with the water
        if self.cob_method == COB_RAMP:
            in_water = pos[2] - self.bound_radii[li] <= surface
        else:
            volume = self.submerged[li, 0]
            in_water = volume > 0
        if not in_water:
            for i in range(6):
                self.xfrc_array[iteration, x_i, i] = 0
                self.xfrc_applied[body, i] = 0
            if self.implicit_added_mass:
                self.set_added_mass(li, 0, 0)
            return

        density = self.water.density(time, pos[0], pos[1], pos[2])
        for i in range(3):
            force[i] = 0
            torque[i] = 0
            buoyancy[i] = 0

        # Buoyancy: -rho*V*g applied at the centre of buoyancy
        if self.buoyancy:
            if self.cob_method == COB_RAMP:
                if self.masses[li] > 0:
                    ramp = (surface + self.bound_radii[li] - pos[2])/(
                        2*self.bound_radii[li]
                    )
                    ramp = 1 if ramp > 1 else ramp
                    for i in range(3):
                        buoyancy[i] = (
                            -density*self.masses[li]*self.gravity[i]
                            /self.densities[li]*ramp
                        )
            else:
                for i in range(3):
                    buoyancy[i] = -density*volume*self.gravity[i]
                    arm[i] = self.submerged[li, i+1]/volume - com[i]
                cross(arm, buoyancy, torque)
            for i in range(3):
                force[i] = buoyancy[i]

        # Drag
        if self.drag:
            self.water.velocity(time, pos[0], pos[1], pos[2], vel)
            for i in range(3):
                vel[i] = state[LINK_COM_VELOCITY_LIN_X+i] - vel[i]
            if self.ellipsoid_drag:
                volume = self.submerged[li, 0]
                fraction = (
                    volume/self.full_volume[li] if self.full_volume[li] > 0 else 0
                )
                fraction = 1 if fraction > 1 else fraction
                viscosity = self.water.viscosity(time, pos[0], pos[1], pos[2])
                for i in range(3):
                    arm[i] = self.xpos[body, i]*self.inv_meters
                self.ellipsoid.wrench(
                    li, &self.xmat[body, 0], arm, com, vel,
                    state + LINK_COM_VELOCITY_ANG_X,
                    density, viscosity, fraction, timestep, tmp, tmp2,
                )
                for i in range(3):
                    force[i] += tmp[i]
                    torque[i] += tmp2[i]
                if self.implicit_added_mass:
                    ramp = self.set_added_mass(li, fraction, density)
                    # The added mass must not weigh: cancel its gravity
                    for i in range(3):
                        force[i] -= ramp*self.gravity[i]
            else:
                quat2mat(state + LINK_URDF_ORIENTATION_X, R)
                viscosity = self.water.viscosity(time, pos[0], pos[1], pos[2])
                mat_t_vec(R, vel, tmp)  # Relative velocity in link frame
                if self.drag_implicit:
                    mat_t_vec(R, buoyancy, arm)
                    quadratic_drag_implicit(
                        tmp, &self.coefficients[li, 0, 0], viscosity,
                        self.masses[li], timestep, arm, tmp2,
                    )
                else:
                    quadratic_drag(
                        tmp, &self.coefficients[li, 0, 0], viscosity, tmp2,
                    )
                mat_vec(R, tmp2, tmp)
                for i in range(3):
                    force[i] += tmp[i]
                mat_t_vec(R, state + LINK_COM_VELOCITY_ANG_X, tmp)
                quadratic_drag(tmp, &self.coefficients[li, 1, 0], 1.0, tmp2)
                mat_vec(R, tmp2, tmp)
                for i in range(3):
                    torque[i] += tmp[i]

        # Apply at the CoM, in the world frame
        for i in range(3):
            self.xfrc_array[iteration, x_i, i] = force[i]
            self.xfrc_array[iteration, x_i, i+3] = torque[i]
            self.xfrc_applied[body, i] = force[i]*self.newtons
            self.xfrc_applied[body, i+3] = torque[i]*self.torques

    cdef double set_added_mass(self, int li, double fraction, double density):
        """Implicit added mass (Stonefish style): add the mean translational
        added mass and the diagonal added inertia (scaled by the submerged
        fraction) to the body. Returns the added mass [kg]."""
        cdef int body = self.body_ids[li], i
        cdef double added = 0
        if fraction > 0:
            added = fraction*density*(
                self.ellipsoid.mass_added[li, 0]
                + self.ellipsoid.mass_added[li, 1]
                + self.ellipsoid.mass_added[li, 2]
            )/3
        self.body_mass[body] = self.body_mass0[body] + added*self.kilograms
        for i in range(3):
            self.body_inertia[body, i] = self.body_inertia0[body, i] + (
                fraction*density*self.inertia_added[li, i]*self.inertia_unit
            )
        return added

    cpdef void set_water_velocity(self, DTYPEv1 velocity):
        """Set water velocity (constant water properties only)"""
        (<WaterPropertiesConstant>self.water).set_velocity(
            0, velocity[0], velocity[1], velocity[2],
        )

    def links_submerged(self):
        """Last computed [V, V*c] per link (SI units)"""
        return np.asarray(self.submerged)
