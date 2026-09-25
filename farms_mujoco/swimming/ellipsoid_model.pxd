"""Ellipsoid drag and added mass model (see ellipsoid_model.pyx)"""

cdef enum:
    ADDED_MASS_OFF = 0
    ADDED_MASS_EXPLICIT = 1
    ADDED_MASS_IMPLICIT = 2


cdef class EllipsoidModel:
    cdef readonly int n_links
    cdef readonly int added_mass
    cdef readonly object ellipsoids
    cdef double c_form, c_viscous, c_rotational, dynamic_viscosity
    cdef double filter_alpha
    cdef double[:, ::1] center        # [link, 3] ellipsoid centre (body frame)
    cdef double[:, ::1] rotation      # [link, 9] ellipsoid axes (body frame)
    cdef double[:, ::1] area          # [link, 3] (bc, ac, ab)
    cdef double[::1] radius           # [link] mean radius (a + b + c)/3
    cdef double[:, ::1] drag_inertia  # [link, 3] 8pi/15 r_i max(r_j, r_k)^4
    cdef readonly double[:, ::1] mass_added     # [link, 3] per unit density
    cdef readonly double[:, ::1] inertia_added  # [link, 3] per unit density
    cdef double[:, ::1] prev_velocity  # [link, 6] world lin/ang (explicit)
    cdef double[:, ::1] acceleration   # [link, 6] filtered, world
    cdef int[::1] has_prev

    cdef void wrench(
        self, int link, const double *R_body, const double *p_body,
        const double *com, const double *velocity, const double *omega,
        double density, double drag_scale, double fraction, double dt,
        double *force, double *torque,
    ) noexcept nogil
