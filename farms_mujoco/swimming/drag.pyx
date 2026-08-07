"""drag.pyx -- drag physics only.

Split out of what used to be a single drag.pyx containing drag,
buoyancy, and orchestration all together. Now:
  - drag.pyx          (this file) -- drag force/torque, nothing else
  - buoyancy_cy.pyx    -- buoyancy force/torque, nothing else (note the
                          `_cy` suffix: buoyancy.py already exists as
                          the pure-Python exact-method module, and a
                          compiled `buoyancy.pyx` would produce a
                          module literally named `buoyancy`, colliding
                          with it in the same package -- see that
                          file's docstring)
  - hydrodynamics.pyx  -- orchestration: combines drag + buoyancy per
                          link, water property classes, SwimmingHandler

This file has no knowledge of buoyancy, sensors, or water surfaces --
it only knows "given a relative velocity and some coefficients, what's
the drag force/torque". `compute_link_drag_fast` is the one entry point
hydrodynamics.pyx calls; it does the fluid-relative-velocity rotation
that used to live inline in compute_link_forces, then calls the two
pure force/torque functions below.
"""

from farms_core.utils.transform cimport quat_rot


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


cdef void compute_link_drag_fast(
    DTYPEv1 force,
    DTYPEv1 torque,
    DTYPEv1 link_lin_velocity,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 fluid_velocity_world,
    DTYPEv1 global2urdf,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 fluid_velocity_urdf,
    DTYPEv2 coefficients,
    DTYPEv1 buoyancy,
    double viscosity,
):
    """Combined per-link drag entry point, called once per link per step
    by hydrodynamics.compute_link_forces.

    Rotates the ambient fluid velocity into the link's URDF frame,
    subtracts it from the link's own velocity (drag depends on the
    RELATIVE velocity between link and fluid, not the link's absolute
    velocity), then computes drag force (with buoyancy folded in --
    see compute_drag_force) and drag torque.

    Mutates `link_lin_velocity` in place (becomes the relative
    velocity) -- same as the original inline code did; the caller's
    copy of it is scratch, not needed again after this call.

    `coefficients` is the link's (2, 3) drag-coefficient array:
    coefficients[0] = linear (force) coefficients, coefficients[1] =
    angular (torque) coefficients -- unchanged convention.
    """
    quat_rot(fluid_velocity_world, global2urdf, quat_c, tmp4, fluid_velocity_urdf)
    link_lin_velocity[0] -= fluid_velocity_urdf[0]
    link_lin_velocity[1] -= fluid_velocity_urdf[1]
    link_lin_velocity[2] -= fluid_velocity_urdf[2]

    compute_drag_force(
        force=force,
        link_velocity=link_lin_velocity,
        coefficients=coefficients[0],
        buoyancy=buoyancy,
        viscosity=viscosity,
    )
    compute_drag_torque(
        torque=torque,
        link_ang_velocity=link_ang_velocity,
        coefficients=coefficients[1],
    )
