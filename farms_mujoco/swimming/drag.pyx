"""Drag force kernels.

Per-axis quadratic drag in a link frame, F_i = scale*c_i*v_i*|v_i|, where
the coefficients c_i are negative (drag opposes the motion) and scale is
the fluid viscosity for forces or 1 for torques.
"""

# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

from libc.math cimport fabs, sqrt


cdef void quadratic_drag(
    const double *velocity, const double *coefficients, double scale,
    double *out,
) noexcept nogil:
    """Explicit per-axis quadratic drag"""
    cdef int i
    for i in range(3):
        out[i] = scale*coefficients[i]*velocity[i]*fabs(velocity[i])


cdef void quadratic_drag_implicit(
    const double *velocity, const double *coefficients, double scale,
    double mass, double dt, const double *other, double *out,
) noexcept nogil:
    """Semi-implicit (backward Euler) per-axis quadratic drag.

    Solves m*(v' - v)/dt = -c*v'*|v'| + f per axis for the end-of-step
    velocity v', with c = -scale*coefficient >= 0 and f the other fluid
    forces on that axis, and returns the drag -c*v'*|v'|. Unconditionally
    stable for large timesteps, falls back to explicit drag otherwise.
    """
    cdef int i
    cdef double c, a, b, v_new
    for i in range(3):
        c = -scale*coefficients[i]
        if c > 0 and mass > 0 and dt > 0:
            a = mass/dt
            b = a*velocity[i] + other[i]
            if b >= 0:
                v_new = 2*b/(a + sqrt(a*a + 4*c*b))
            else:
                v_new = 2*b/(a + sqrt(a*a - 4*c*b))
            out[i] = -c*v_new*fabs(v_new)
        else:
            out[i] = scale*coefficients[i]*velocity[i]*fabs(velocity[i])
