"""Ellipsoid drag and added mass model (work in progress)"""

# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np


cdef class EllipsoidModel:
    """Placeholder until the ellipsoid model is implemented"""

    cdef void wrench(
        self, int link, const double *R_body, const double *velocity,
        const double *angular_velocity, double density, double viscosity,
        double fraction, double *force, double *torque,
    ) noexcept nogil:
        cdef int i
        for i in range(3):
            force[i] = 0
            torque[i] = 0


def build_ellipsoid_model(**kwargs):
    """Build the ellipsoid model"""
    raise NotImplementedError('fluid_model: ellipsoid is not implemented yet')
