"""Centre-of-buoyancy lookup tables: O(1) submerged volume and centroid
per link, whatever its shape and number of (overlapping) geoms.

The tables are built once by cob_lut_build.py. For each direction n of an
octahedral grid and threshold t, they hold the volume V and first moment
M = (M.n) n + M_perp of the region {x : n.x <= t} of the link (link frame).

Query: the water half-space z <= h in the link frame is {n.x <= t} with
n = R^T z and t = h - p_z. V, M.n and M_perp are interpolated bilinearly
over the direction grid and linearly over the normalised depth, and the
moment is rebuilt as (M.n) n + M_perp projected orthogonally to n.
"""

# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport fabs, floor

np.import_array()


# ---------------------------------------------------------------------------
# Octahedral direction mapping
# ---------------------------------------------------------------------------

cdef inline double _sign(double x) noexcept nogil:
    return 1.0 if x >= 0 else -1.0


cdef inline void _oct_encode(const double *n, double *u, double *v) noexcept nogil:
    """Unit vector to octahedral coordinates in [0, 1]^2"""
    cdef double l1 = fabs(n[0]) + fabs(n[1]) + fabs(n[2])
    cdef double px = n[0]/l1, py = n[1]/l1, qx
    if n[2] < 0:
        qx = (1 - fabs(py))*_sign(px)
        py = (1 - fabs(px))*_sign(py)
        px = qx
    u[0] = 0.5*(px + 1)
    v[0] = 0.5*(py + 1)


def oct_encode(n):
    """Unit vector to octahedral coordinates (Python, for tests)"""
    cdef double vec[3]
    cdef double u, v
    for i in range(3):
        vec[i] = n[i]
    _oct_encode(vec, &u, &v)
    return u, v


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

cdef class CobLut:
    """Per-link centre-of-buoyancy lookup tables"""

    def __init__(self, body_ids, luts, resolution):
        self.n_links = len(luts)
        self.n_dir, self.n_depth = resolution
        self.body_ids = np.ascontiguousarray(body_ids, dtype=np.intc)
        self.table = np.ascontiguousarray(
            [lut[0] for lut in luts], dtype=np.double,
        ).reshape(self.n_links, self.n_dir*self.n_dir, self.n_depth, 5)
        self.t_range = np.ascontiguousarray(
            [lut[1] for lut in luts], dtype=np.double,
        ).reshape(self.n_links, self.n_dir*self.n_dir, 2)
        self.volume = np.array([lut[2] for lut in luts], dtype=np.double)
        self.centroid = np.ascontiguousarray(
            [lut[3] for lut in luts], dtype=np.double,
        ).reshape(self.n_links, 3)
        self.t_bound = np.array([
            np.max(np.abs(lut[1])) if len(lut[1]) else 0.0 for lut in luts
        ], dtype=np.double)

    cdef void link_submerged(
        self, int link, const double *R, const double *p, double h,
        double *out,
    ) noexcept nogil:
        """Submerged [V, V*c] (world) of a link with pose (R, p)"""
        cdef double n[3]
        cdef double moment[4]
        cdef double acc[5]
        cdef double t, u, v, fu, fv, wu, wv, w, s, t_min, t_max, frac, dot
        cdef int iu, iv, du, dv, d, k, i
        cdef const double *row0
        cdef const double *row1
        n[0] = R[6]
        n[1] = R[7]
        n[2] = R[8]
        t = h - p[2]
        for i in range(5):
            acc[i] = 0
        if t <= -self.t_bound[link]:
            for i in range(4):
                out[i] = 0
            return
        if t >= self.t_bound[link]:
            moment[0] = self.volume[link]
            for i in range(3):
                moment[i+1] = moment[0]*self.centroid[link, i]
        else:
            _oct_encode(n, &u, &v)
            fu = u*(self.n_dir - 1)
            fv = v*(self.n_dir - 1)
            iu = <int>floor(fu)
            iv = <int>floor(fv)
            iu = 0 if iu < 0 else (self.n_dir - 2 if iu > self.n_dir - 2 else iu)
            iv = 0 if iv < 0 else (self.n_dir - 2 if iv > self.n_dir - 2 else iv)
            wu = fu - iu
            wv = fv - iv
            for du in range(2):
                for dv in range(2):
                    w = (wu if du else 1 - wu)*(wv if dv else 1 - wv)
                    if w == 0:
                        continue
                    d = (iu + du)*self.n_dir + iv + dv
                    t_min = self.t_range[link, d, 0]
                    t_max = self.t_range[link, d, 1]
                    s = (t - t_min)/(t_max - t_min)*(self.n_depth - 1)
                    if s <= 0:
                        continue
                    if s >= self.n_depth - 1:
                        k = self.n_depth - 2
                        frac = 1
                    else:
                        k = <int>floor(s)
                        frac = s - k
                    row0 = &self.table[link, d, k, 0]
                    row1 = row0 + 5
                    for i in range(5):
                        acc[i] += w*(row0[i] + frac*(row1[i] - row0[i]))
            # Rebuild the moment: along n plus the part orthogonal to n
            dot = acc[2]*n[0] + acc[3]*n[1] + acc[4]*n[2]
            moment[0] = acc[0]
            for i in range(3):
                moment[i+1] = acc[1]*n[i] + acc[2+i] - dot*n[i]
        out[0] = moment[0]
        for i in range(3):
            out[i+1] = (
                R[3*i]*moment[1] + R[3*i+1]*moment[2] + R[3*i+2]*moment[3]
                + moment[0]*p[i]
            )

    cdef void compute(
        self,
        const double *xpos,
        const double *xmat,
        double inv_meters,
        const double *surfaces,
        double *out,
    ) noexcept nogil:
        """Per-link submerged [V, V*c] (n_links x 4) from body poses"""
        cdef int link, body, i
        cdef double p[3]
        for link in range(self.n_links):
            body = self.body_ids[link]
            for i in range(3):
                p[i] = xpos[3*body+i]*inv_meters
            self.link_submerged(link, xmat + 9*body, p, surfaces[link], out + 4*link)

    def submerged(self, int link, R, p, double h):
        """Python access for tests: (V, centroid)"""
        cdef double[::1] rot = np.ascontiguousarray(R, np.double).reshape(-1)
        cdef double[::1] pos = np.ascontiguousarray(p, np.double).reshape(-1)
        cdef double out[4]
        self.link_submerged(link, &rot[0], &pos[0], h, out)
        centroid = (
            np.array([out[1], out[2], out[3]])/out[0]
            if out[0] > 0 else np.array(p, dtype=np.double)
        )
        return out[0], centroid


def benchmark_link(CobLut lut, int link, R, p, heights, int repeats=1000):
    """Average time [ns] of one link_submerged call (for benchmarks)"""
    import time  # pylint: disable=import-outside-toplevel
    cdef double[::1] rot = np.ascontiguousarray(R, np.double).reshape(-1)
    cdef double[::1] pos = np.ascontiguousarray(p, np.double).reshape(-1)
    cdef double[::1] hs = np.ascontiguousarray(heights, np.double)
    cdef double out[4]
    cdef double checksum = 0.0
    cdef int i, j, n = hs.shape[0]
    tic = time.perf_counter_ns()
    with nogil:
        for i in range(repeats):
            for j in range(n):
                lut.link_submerged(link, &rot[0], &pos[0], hs[j], out)
                checksum += out[0]
    toc = time.perf_counter_ns()
    return (toc - tic)/(repeats*n), checksum
