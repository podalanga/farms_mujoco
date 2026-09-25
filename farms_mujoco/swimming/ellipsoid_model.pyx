"""Ellipsoid drag and added mass model.

Each link is approximated by an ellipsoid (see ellipsoid_fit.py). In the
ellipsoid frame, with v the velocity of the ellipsoid centre relative to
the water and w the angular velocity, the fluid wrench is (MuJoCo's
ellipsoid model conventions, Stonefish-style added mass):

- form drag        f = -1/2 rho C_form pi sqrt(sum((A_i v_i)^2)) v,
                   A = (bc, ac, ab), i.e. -1/2 rho C_form A_proj(v) |v| v
- viscous drag     f = -6 pi mu r_D C_visc v, g = -8 pi mu r_D^3 C_visc w
- rotational drag  g_i = -rho C_rot I_D,i w_i |w|,
                   I_D,i = 8 pi/15 r_i max(r_j, r_k)^4
- added mass       f = (m_A o v) x w, g = (m_A o v) x v + (I_A o w) x w
                   and, when explicit, -m_A o dv/dt and -I_A o dw/dt from
                   filtered finite differences of the velocities

Quadratic drag terms are scaled by the water viscosity option (as the
legacy per-link drag) and every term by the submerged fraction of the
link. The wrench is returned in the world frame, about the link CoM.
With added_mass: implicit, the acceleration terms are instead added to
the body mass and inertia by the SwimmingHandler (as Stonefish does).
"""

# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import warnings

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, M_PI

from .cob_build import link_geoms, quat2mat
from .ellipsoid_fit import (
    fit_link_ellipsoid, added_mass, angular_drag_inertia,
    projected_area_coefficients,
)

np.import_array()


cdef inline void _cross(const double *a, const double *b, double *out) noexcept nogil:
    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]


cdef class EllipsoidModel:
    """Per-link ellipsoid fluid model"""

    def __init__(self, ellipsoids, coefficients, dynamic_viscosity=1e-3,
                 added_mass_mode='off', filter_alpha=0.5):
        self.ellipsoids = ellipsoids
        self.n_links = len(ellipsoids)
        self.added_mass = {
            'off': ADDED_MASS_OFF,
            'explicit': ADDED_MASS_EXPLICIT,
            'implicit': ADDED_MASS_IMPLICIT,
        }[added_mass_mode]
        self.c_form, self.c_viscous, self.c_rotational = coefficients
        self.dynamic_viscosity = dynamic_viscosity
        self.filter_alpha = filter_alpha
        n = self.n_links
        self.center = np.ascontiguousarray(
            [e.center for e in ellipsoids], dtype=np.double,
        ).reshape(n, 3)
        self.rotation = np.ascontiguousarray(
            [e.rotation.reshape(-1) for e in ellipsoids], dtype=np.double,
        ).reshape(n, 9)
        self.area = np.ascontiguousarray(
            [projected_area_coefficients(e.axes) for e in ellipsoids],
            dtype=np.double,
        ).reshape(n, 3)
        self.radius = np.array([np.mean(e.axes) for e in ellipsoids], dtype=np.double)
        self.drag_inertia = np.ascontiguousarray(
            [angular_drag_inertia(e.axes) for e in ellipsoids], dtype=np.double,
        ).reshape(n, 3)
        masses, inertias = [], []
        for e in ellipsoids:
            mass, inertia = added_mass(e.axes)
            masses.append(mass)
            inertias.append(inertia)
        self.mass_added = np.ascontiguousarray(masses, dtype=np.double).reshape(n, 3)
        self.inertia_added = np.ascontiguousarray(inertias, dtype=np.double).reshape(n, 3)
        self.prev_velocity = np.zeros([n, 6])
        self.acceleration = np.zeros([n, 6])
        self.has_prev = np.zeros(n, dtype=np.intc)

    cdef void wrench(
        self, int link, const double *R_body, const double *p_body,
        const double *com, const double *velocity, const double *omega,
        double density, double drag_scale, double fraction, double dt,
        double *force, double *torque,
    ) noexcept nogil:
        """Fluid wrench (world frame, about the CoM) on a link.

        velocity: CoM velocity relative to the water (world), omega:
        angular velocity (world)."""
        cdef double Re[9]
        cdef double center[3]
        cdef double arm[3]
        cdef double tmp[3]
        cdef double vc[3]
        cdef double v[3]
        cdef double w[3]
        cdef double f[3]
        cdef double g[3]
        cdef double mv[3]
        cdef double iw[3]
        cdef double acc[6]
        cdef const double *Rb = &self.rotation[link, 0]
        cdef const double *area = &self.area[link, 0]
        cdef double speed, spin, r = self.radius[link], mu, alpha
        cdef int i, j
        # Ellipsoid pose in the world
        for i in range(3):
            for j in range(3):
                Re[3*i+j] = (
                    R_body[3*i]*Rb[j] + R_body[3*i+1]*Rb[3+j]
                    + R_body[3*i+2]*Rb[6+j]
                )
            center[i] = p_body[i] + (
                R_body[3*i]*self.center[link, 0]
                + R_body[3*i+1]*self.center[link, 1]
                + R_body[3*i+2]*self.center[link, 2]
            )
            arm[i] = center[i] - com[i]
        # Velocity of the ellipsoid centre, then in the ellipsoid frame
        _cross(omega, arm, tmp)
        for i in range(3):
            vc[i] = velocity[i] + tmp[i]
        for i in range(3):
            v[i] = Re[i]*vc[0] + Re[3+i]*vc[1] + Re[6+i]*vc[2]
            w[i] = Re[i]*omega[0] + Re[3+i]*omega[1] + Re[6+i]*omega[2]

        # Quadratic form drag with the projected area along v
        speed = sqrt(
            (area[0]*v[0])**2 + (area[1]*v[1])**2 + (area[2]*v[2])**2
        )
        spin = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2])
        mu = self.dynamic_viscosity*self.c_viscous
        for i in range(3):
            f[i] = (
                -0.5*density*self.c_form*drag_scale*M_PI*speed*v[i]
                - 6*M_PI*mu*r*v[i]
            )
            g[i] = (
                -density*self.c_rotational*drag_scale
                *self.drag_inertia[link, i]*w[i]*spin
                - 8*M_PI*mu*r*r*r*w[i]
            )

        # Added mass
        if self.added_mass != ADDED_MASS_OFF:
            for i in range(3):
                mv[i] = density*self.mass_added[link, i]*v[i]
                iw[i] = density*self.inertia_added[link, i]*w[i]
            _cross(mv, w, tmp)
            for i in range(3):
                f[i] += tmp[i]
            _cross(mv, v, tmp)
            for i in range(3):
                g[i] += tmp[i]
            _cross(iw, w, tmp)
            for i in range(3):
                g[i] += tmp[i]
            if self.added_mass == ADDED_MASS_EXPLICIT and dt > 0:
                # Filtered finite difference accelerations (world frame)
                alpha = self.filter_alpha
                for i in range(3):
                    acc[i] = vc[i]
                    acc[3+i] = omega[i]
                if self.has_prev[link]:
                    for i in range(6):
                        self.acceleration[link, i] = (
                            alpha*(acc[i] - self.prev_velocity[link, i])/dt
                            + (1 - alpha)*self.acceleration[link, i]
                        )
                self.has_prev[link] = 1
                for i in range(6):
                    self.prev_velocity[link, i] = acc[i]
                for i in range(3):
                    tmp[i] = (
                        Re[i]*self.acceleration[link, 0]
                        + Re[3+i]*self.acceleration[link, 1]
                        + Re[6+i]*self.acceleration[link, 2]
                    )
                    f[i] -= density*self.mass_added[link, i]*tmp[i]
                for i in range(3):
                    tmp[i] = (
                        Re[i]*self.acceleration[link, 3]
                        + Re[3+i]*self.acceleration[link, 4]
                        + Re[6+i]*self.acceleration[link, 5]
                    )
                    g[i] -= density*self.inertia_added[link, i]*tmp[i]

        # Back to the world frame, about the CoM
        for i in range(3):
            force[i] = fraction*(Re[3*i]*f[0] + Re[3*i+1]*f[1] + Re[3*i+2]*f[2])
            torque[i] = fraction*(Re[3*i]*g[0] + Re[3*i+1]*g[1] + Re[3*i+2]*g[2])
        _cross(arm, force, tmp)
        for i in range(3):
            torque[i] += tmp[i]

    def compute_wrench(self, int link, R_body, p_body, com, velocity, omega,
                       double density=1000, double drag_scale=1,
                       double fraction=1, double dt=0):
        """Python access for tests: (force, torque) in the world frame"""
        cdef double[::1] rb = np.ascontiguousarray(R_body, np.double).reshape(-1)
        cdef double[::1] pb = np.ascontiguousarray(p_body, np.double)
        cdef double[::1] cm = np.ascontiguousarray(com, np.double)
        cdef double[::1] vel = np.ascontiguousarray(velocity, np.double)
        cdef double[::1] om = np.ascontiguousarray(omega, np.double)
        cdef double f[3]
        cdef double t[3]
        self.wrench(link, &rb[0], &pb[0], &cm[0], &vel[0], &om[0], density,
                    drag_scale, fraction, dt, f, t)
        return np.array([f[0], f[1], f[2]]), np.array([t[0], t[1], t[2]])


def build_ellipsoid_model(model, body_ids, options, meters=1.0, kilograms=1.0,
                          water_density=1000.0):
    """EllipsoidModel for the given MuJoCo bodies"""
    geoms = link_geoms(
        model, body_ids, geom_group=options.cob_geom_group, meters=meters,
    )
    ellipsoids = []
    for link_i, body_id in enumerate(body_ids):
        ellipsoids.append(fit_link_ellipsoid(
            options.ellipsoid_fit,
            [geom for geom in geoms if geom.link == link_i],
            mass=model.body_mass[body_id]/kilograms,
            inertia=np.array(model.body_inertia[body_id])/(kilograms*meters**2),
            ipos=np.array(model.body_ipos[body_id])/meters,
            iquat=model.body_iquat[body_id],
        ))
    ellipsoid_model = EllipsoidModel(
        ellipsoids,
        coefficients=options.ellipsoid_coefficients,
        dynamic_viscosity=options.dynamic_viscosity,
        added_mass_mode=options.added_mass,
    )
    if options.added_mass == 'explicit':
        ratios = [
            water_density*np.max(ellipsoid_model.mass_added[link_i])
            /(model.body_mass[body_id]/kilograms)
            for link_i, body_id in enumerate(body_ids)
        ]
        if max(ratios, default=0) > 0.5:
            warnings.warn(
                'Explicit added mass is numerically unstable when the added'
                f' mass is large compared to the link mass (max ratio'
                f' {max(ratios):.2f}), consider added_mass: implicit'
            )
    return ellipsoid_model


def implicit_added_inertia(ellipsoid_model, model, body_ids):
    """Per-link diagonal added inertia (per unit density) in the body
    inertial frame, for the implicit added mass"""
    result = []
    for link_i, body_id in enumerate(body_ids):
        ellipsoid = ellipsoid_model.ellipsoids[link_i]
        inertial = quat2mat(model.body_iquat[body_id])
        rot = inertial.T @ ellipsoid.rotation
        tensor = rot @ np.diag(ellipsoid_model.inertia_added[link_i]) @ rot.T
        result.append(np.diag(tensor).copy())
    return np.array(result)
