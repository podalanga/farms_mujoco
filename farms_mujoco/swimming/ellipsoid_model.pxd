"""Ellipsoid drag and added mass model (see ellipsoid_model.pyx)"""


cdef class EllipsoidModel:
    cdef readonly int n_links
    cdef readonly double[::1] link_volume

    cdef void wrench(
        self, int link, const double *R_body, const double *velocity,
        const double *angular_velocity, double density, double viscosity,
        double fraction, double *force, double *torque,
    ) noexcept nogil
