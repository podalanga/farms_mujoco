"""Centre-of-buoyancy lookup tables (see cob_lut.pyx)"""


cdef class CobLut:
    cdef readonly int n_links
    cdef readonly int n_dir
    cdef readonly int n_depth
    cdef int[::1] body_ids
    cdef double[:, :, :, ::1] table   # [link, dir, depth, (V, M.n, M_perp)]
    cdef double[:, :, ::1] t_range    # [link, dir, (t_min, t_max)]
    cdef double[::1] volume           # [link] full volume
    cdef double[:, ::1] centroid      # [link, 3] full centroid (body frame)
    cdef double[::1] t_bound          # [link] max |t| of the table range

    cdef void link_submerged(
        self, int link, const double *R, const double *p, double h,
        double *out,
    ) noexcept nogil

    cdef void compute(
        self,
        const double *xpos,
        const double *xmat,
        double inv_meters,
        const double *surfaces,
        double *out,
    ) noexcept nogil
