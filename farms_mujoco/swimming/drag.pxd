"""Drag force kernels (see drag.pyx)"""

cdef void quadratic_drag(
    const double *velocity, const double *coefficients, double scale,
    double *out,
) noexcept nogil

cdef void quadratic_drag_implicit(
    const double *velocity, const double *coefficients, double scale,
    double mass, double dt, const double *other, double *out,
) noexcept nogil
