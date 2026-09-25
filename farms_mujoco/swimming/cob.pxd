"""Centre-of-buoyancy geometry kernels (see cob.pyx)"""

# Geometry kinds handled by the kernels
cdef enum:
    COB_SPHERE = 0
    COB_ELLIPSOID = 1
    COB_CYLINDER = 2
    COB_CAPSULE = 3
    COB_POLYHEDRON = 4
    COB_NODE_MOMENTS = 16  # [S0, S1(3), m(3), M(9)] per BVH node


cdef struct Bvh:
    # Flat bounding volume hierarchy over the triangles of all polyhedra
    const double *center    # [n_nodes, 3] node AABB centre (geom frame)
    const double *half      # [n_nodes, 3] node AABB half extents
    const double *moments   # [n_nodes, 16] divergence theorem moments
    const int *left         # [n_nodes] left child, -1 for leaves
    const int *right        # [n_nodes] right child
    const int *tri_start    # [n_nodes] first triangle (leaves)
    const int *tri_count    # [n_nodes] number of triangles (leaves)
    const double *tris      # [n_tris, 9] triangle vertices (geom frame)


# All kernels write out[0] = submerged volume and out[1:4] = volume
# weighted centroid (V*c) in the world frame, for the half-space z <= h.
# R is a row-major world rotation (MuJoCo xmat) and c the world position.

cdef void sphere_submerged(
    double radius, const double *c, double h, double *out,
) noexcept nogil

cdef void ellipsoid_submerged(
    const double *axes, const double *R, const double *c, double h,
    double *out,
) noexcept nogil

cdef void cylinder_submerged(
    double radius, double half_length, const double *R, const double *c,
    double h, double *out,
) noexcept nogil

cdef void capsule_submerged(
    double radius, double half_length, const double *R, const double *c,
    double h, double *out,
) noexcept nogil

cdef void polyhedron_submerged(
    const Bvh *bvh, int root, const double *R, const double *c, double h,
    double *out,
) noexcept nogil


cdef class CobModel:
    cdef object _geometry
    cdef readonly int n_links
    cdef readonly int n_geoms
    cdef int[::1] geom_kind
    cdef int[::1] geom_link
    cdef readonly int[::1] geom_id
    cdef int[::1] geom_root
    cdef double[:, ::1] geom_size
    cdef double[::1] geom_scale
    cdef double[::1] geom_volume
    cdef double[:, ::1] geom_centroid
    cdef double[::1] geom_rbound
    cdef object _center, _half, _moments, _left, _right
    cdef object _tri_start, _tri_count, _tris
    cdef Bvh bvh

    cdef void _bind_bvh(self)
    cdef void geom_submerged(
        self, int g, const double *R, const double *c, double h,
        double *out,
    ) noexcept nogil
    cdef void compute(
        self,
        const double *geom_xpos,
        const double *geom_xmat,
        double inv_meters,
        const double *surfaces,
        double *out,
    ) noexcept nogil
