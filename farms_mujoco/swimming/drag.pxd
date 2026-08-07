include 'types.pxd'


cdef void compute_drag_force(
    DTYPEv1 force,
    DTYPEv1 link_velocity,
    DTYPEv1 coefficients,
    DTYPEv1 buoyancy,
    double viscosity,
)

cdef void compute_drag_torque(
    DTYPEv1 torque,
    DTYPEv1 link_ang_velocity,
    DTYPEv1 coefficients,
)

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
)
