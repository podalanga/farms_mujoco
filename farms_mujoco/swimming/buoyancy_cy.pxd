include 'types.pxd'

# Only the buoyancy force/torque entry points need cdef-level (cimport)
# declarations here -- hydrodynamics.pyx cimports these directly.
# submerged_volume_and_centroid_fast_cy and submerged_sphere_analytic_cy
# (formerly cob_fast.pyx, now merged into buoyancy_cy.pyx) are plain
# `def`/`cpdef` functions reached via a normal Python import from
# buoyancy.py, same as before the merge -- no cdef declaration needed
# for them here.

cdef void compute_buoyancy_analytic_fast(
    double density,
    double water_density,
    double bound_radius,
    double position,
    DTYPEv1 global2urdf,
    double mass,
    double surface,
    double gravity,
    DTYPEv1 buoyancy,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 tmp,
)

cdef void compute_link_buoyancy_fast(
    bint use_buoyancy,
    bint use_exact_cob,
    bint force_mesh,
    object primitives,
    double density,
    double water_density,
    double bound_radius,
    double pos_x,
    double pos_y,
    double pos_z,
    DTYPEv1 global2urdf,
    DTYPEv1 urdf2global,
    double mass,
    double surface,
    double gravity,
    DTYPEv1 buoyancy,
    DTYPEv1 buoyancy_torque,
    DTYPEv1 com_position,
    DTYPEv1 pos_urdf,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 tmp,
) except *
