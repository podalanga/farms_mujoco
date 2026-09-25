"""Centre of buoyancy (CoB) geometry kernels.

Every kernel computes the part of a solid lying below a horizontal water
plane z <= h and returns its volume V and volume-weighted centroid V*c in
the world frame. Accumulating (V, V*c) over several solids and dividing
gives the centre of buoyancy of the union (assuming no overlap).

Methods (all exact up to floating point unless stated):

- Sphere: closed-form spherical cap.
- Ellipsoid: the affine map x = c + R*diag(a)*u sends the unit ball to the
  ellipsoid and planes to planes, so the submerged part is the image of a
  unit-sphere cap. Volumes scale by abc and centroids map through the
  affine transform.
- Cylinder: integral of circular-segment areas along the axis. The signed
  distance of the waterline in each disk varies linearly along the axis,
  so volume and first moments have closed-form antiderivatives.
- Capsule: exact cylinder plus two hemispherical caps integrated slice by
  slice with Gauss-Legendre quadrature on sub-intervals split at the
  kinks where slices become fully wet/dry (relative error ~1e-10).
- Polyhedron (box, mesh): divergence theorem with the tetrahedra apex on
  the water plane, so the waterline cap contributes nothing and only the
  wet part of the surface is needed. A BVH stores, per node, the moments
  S0 = sum(det(a, b, c)), S1 = sum(N), m = sum(det*s), M = sum(s N^T) with
  N = a x b + b x c + c x a and s = a + b + c, which make fully wet nodes
  O(1) regardless of how many triangles they hold:
      6V = S0 - S1.p,  24(V c) = m - M p + p (S0 - S1.p).
  Only triangles in leaves crossing the waterline are clipped, so the
  cost is O(k + log n) for k triangles near the waterline.
"""

# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, acos, asin, cos, fabs, M_PI

np.import_array()

cdef double EPS_AXIS = 1e-9  # Axis considered vertical below this tilt

# Gauss-Legendre nodes/weights on [-1, 1] (12 points)
cdef int GL_N = 12
cdef double GL_X[12]
cdef double GL_W[12]
_gl_x, _gl_w = np.polynomial.legendre.leggauss(12)
for _i in range(12):
    GL_X[_i] = _gl_x[_i]
    GL_W[_i] = _gl_w[_i]


cdef inline double _clamp(double x, double lo, double hi) noexcept nogil:
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# Sphere and ellipsoid
# ---------------------------------------------------------------------------

cdef inline double _unit_cap_volume(double height) noexcept nogil:
    """Volume of a unit-sphere cap of given height (0 <= height <= 2)"""
    return M_PI*height*height*(3.0 - height)/3.0


cdef inline double _unit_cap_centroid(double height) noexcept nogil:
    """Distance from the unit-sphere centre to the centroid of a cap of
    given height, measured towards the cap"""
    return 0.75*(2.0 - height)*(2.0 - height)/(3.0 - height)


cdef void sphere_submerged(
    double radius, const double *c, double h, double *out,
) noexcept nogil:
    """Submerged volume and V*c of a sphere"""
    cdef double height = _clamp((h - c[2])/radius + 1.0, 0.0, 2.0)
    cdef double volume = radius*radius*radius*_unit_cap_volume(height)
    out[0] = volume
    out[1] = volume*c[0]
    out[2] = volume*c[1]
    out[3] = volume*(c[2] - radius*_unit_cap_centroid(height))


cdef void ellipsoid_submerged(
    const double *axes, const double *R, const double *c, double h,
    double *out,
) noexcept nogil:
    """Submerged volume and V*c of an ellipsoid with semi-axes along the
    columns of R"""
    # w = diag(a) R^T z, the plane normal in unit-ball coordinates
    cdef double w0 = axes[0]*R[6], w1 = axes[1]*R[7], w2 = axes[2]*R[8]
    cdef double norm = sqrt(w0*w0 + w1*w1 + w2*w2)
    cdef double height = _clamp((h - c[2])/norm + 1.0, 0.0, 2.0)
    cdef double volume = (
        axes[0]*axes[1]*axes[2]*_unit_cap_volume(height)
    )
    # Centroid offset = -d/|w| R diag(a) w
    cdef double scale = -_unit_cap_centroid(height)/norm
    cdef double d0 = scale*axes[0]*w0
    cdef double d1 = scale*axes[1]*w1
    cdef double d2 = scale*axes[2]*w2
    out[0] = volume
    out[1] = volume*(c[0] + R[0]*d0 + R[1]*d1 + R[2]*d2)
    out[2] = volume*(c[1] + R[3]*d0 + R[4]*d1 + R[5]*d2)
    out[3] = volume*(c[2] + R[6]*d0 + R[7]*d1 + R[8]*d2)


# ---------------------------------------------------------------------------
# Cylinder and capsule (slices along the axis)
# ---------------------------------------------------------------------------
# A disk of radius r, cut by a line at signed distance u from its centre,
# has wet area A(u) = r^2 acos(-u/r) + u sqrt(r^2 - u^2) (clamped to
# [0, pi r^2]) and first moment Q(u) = -2/3 (r^2 - u^2)^(3/2) along the
# direction pointing out of the water. G, H and K are the antiderivatives
# of A, u*A and Q, extended continuously outside [-r, r].

cdef inline double _seg_G(double u, double r) noexcept nogil:
    cdef double r2 = r*r, extra = 0.0, root
    if u <= -r:
        return 0.0
    if u >= r:
        extra = M_PI*r2*(u - r)
        u = r
    root = sqrt(r2 - u*u)
    return r2*(u*acos(-u/r) + root) - root*root*root/3.0 + extra


cdef inline double _seg_H(double u, double r) noexcept nogil:
    cdef double r2 = r*r, extra = 0.0, root, s
    if u >= r:
        extra = 0.5*M_PI*r2*(u*u - r2)
    u = _clamp(u, -r, r)
    root = sqrt(r2 - u*u)
    s = asin(u/r)
    return (
        r2*(0.5*u*u*acos(-u/r) - 0.25*r2*s + 0.25*u*root)
        + 0.125*u*(2.0*u*u - r2)*root + 0.125*r2*r2*s
    ) + extra


cdef inline double _seg_K(double u, double r) noexcept nogil:
    cdef double r2 = r*r, root
    u = _clamp(u, -r, r)
    root = sqrt(r2 - u*u)
    return -(2.0/3.0)*(
        0.125*u*(5.0*r2 - 2.0*u*u)*root + 0.375*r2*r2*asin(u/r)
    )


cdef inline void _disk_segment(
    double u, double rho, double *area, double *moment,
) noexcept nogil:
    """Wet area and first moment of a disk of radius rho cut at u"""
    cdef double root
    if u >= rho:
        area[0] = M_PI*rho*rho
        moment[0] = 0.0
    elif u <= -rho:
        area[0] = 0.0
        moment[0] = 0.0
    else:
        root = sqrt(rho*rho - u*u)
        area[0] = rho*rho*acos(-u/rho) + u*root
        moment[0] = -(2.0/3.0)*root*root*root


cdef inline void _axis_frame(
    const double *R, double *axis, double *perp, double *sigma,
) noexcept nogil:
    """Axis (local z) in world, and the unit vector perp orthogonal to
    the axis along which the world z increases (sigma = its z gain)"""
    cdef double az, inv
    axis[0] = R[2]
    axis[1] = R[5]
    axis[2] = R[8]
    az = axis[2]
    sigma[0] = sqrt(_clamp(1.0 - az*az, 0.0, 1.0))
    if sigma[0] > EPS_AXIS:
        inv = 1.0/sigma[0]
        perp[0] = -az*axis[0]*inv
        perp[1] = -az*axis[1]*inv
        perp[2] = (1.0 - az*az)*inv
    else:
        # Vertical axis: any perpendicular works (its moment vanishes)
        sigma[0] = EPS_AXIS
        perp[0] = R[0]
        perp[1] = R[3]
        perp[2] = R[6]


cdef void _cylinder_moments(
    double r, double L, double delta, double k, double *moments,
) noexcept nogil:
    """Volume, axial and perpendicular moments of the wet part of a
    cylinder where the waterline distance is u(s) = delta - k*s along the
    axis coordinate s in [-L, L]. Slices are fully wet where u >= r,
    dry where u <= -r and partial in between."""
    cdef double area, moment, slope, s_wet, s_dry, lo, hi, mid, half
    cdef double u_lo, u_hi, dg
    moments[0] = 0.0
    moments[1] = 0.0
    moments[2] = 0.0
    if fabs(k)*L < 1e-5*r:
        # Axis parallel to the surface: constant cross-section
        _disk_segment(delta, r, &area, &moment)
        slope = 2.0*sqrt(r*r - delta*delta) if fabs(delta) < r else 0.0
        moments[0] = 2.0*L*area
        moments[1] = -(2.0/3.0)*k*L*L*L*slope
        moments[2] = 2.0*L*moment
        return

    # Axial coordinates where u = r (start of fully wet) and u = -r (dry)
    s_wet = (delta - r)/k
    s_dry = (delta + r)/k

    # Fully wet slices
    if k > 0:
        lo, hi = -L, (s_wet if s_wet < L else L)
    else:
        lo, hi = (s_wet if s_wet > -L else -L), L
    if hi > lo:
        moments[0] += M_PI*r*r*(hi - lo)
        moments[1] += 0.5*M_PI*r*r*(hi*hi - lo*lo)

    # Partial slices, integrated in closed form over u in [-r, r]
    lo = s_wet if s_wet < s_dry else s_dry
    hi = s_dry if s_wet < s_dry else s_wet
    lo = lo if lo > -L else -L
    hi = hi if hi < L else L
    if hi <= lo:
        return
    mid = 0.5*(lo + hi)
    half = 0.5*(hi - lo)
    u_lo = _clamp(delta - k*hi, -r, r)
    u_hi = _clamp(delta - k*lo, -r, r)
    if k < 0:
        u_lo, u_hi = u_hi, u_lo
    dg = _seg_G(u_hi, r) - _seg_G(u_lo, r)
    if k < 0:
        dg = -dg
    moments[0] += dg/k
    moments[2] += (_seg_K(u_hi, r) - _seg_K(u_lo, r))/(k if k > 0 else -k)
    if fabs(k)*half < 1e-3*r:
        # Nearly horizontal axis: expand around the interval midpoint to
        # avoid the cancellation of the closed form
        u_lo = delta - k*mid
        slope = 2.0*sqrt(r*r - u_lo*u_lo) if fabs(u_lo) < r else 0.0
        moments[1] += mid*dg/k - (2.0/3.0)*k*half*half*half*slope
    else:
        # s = (delta - u)/k
        moments[1] += (
            delta*dg - (_seg_H(u_hi, r) - _seg_H(u_lo, r))*(1 if k > 0 else -1)
        )/(k*k)


cdef void cylinder_submerged(
    double radius, double half_length, const double *R, const double *c,
    double h, double *out,
) noexcept nogil:
    """Submerged volume and V*c of a cylinder along local z"""
    cdef double axis[3]
    cdef double perp[3]
    cdef double sigma, moments[3]
    cdef int i
    _axis_frame(R, axis, perp, &sigma)
    _cylinder_moments(
        radius, half_length, (h - c[2])/sigma, axis[2]/sigma, moments,
    )
    out[0] = moments[0]
    for i in range(3):
        out[i+1] = moments[0]*c[i] + moments[1]*axis[i] + moments[2]*perp[i]


cdef void _hemisphere_moments(
    double r, double depth, double az, double sigma, double sign,
    double *moments,
) noexcept nogil:
    """Volume, axial and perpendicular moments of the wet part of a
    hemisphere occupying s in [0, r] (sign=1) or [-r, 0] (sign=-1) along
    the axis from its centre, which lies at the given depth below the
    surface. The waterline distance in the slice at s is
    (depth - s*az)/sigma."""
    cdef double lo = 0.0 if sign > 0 else -r
    cdef double hi = r if sign > 0 else 0.0
    cdef double points[4]
    cdef int n_points = 0, i, j, q
    cdef double disc, root, s0, s1, mid, half, rho
    cdef double theta, s, weight, area, moment, u, tmp
    moments[0] = 0.0
    moments[1] = 0.0
    moments[2] = 0.0

    # Breakpoints where slices switch between dry, partial and wet, i.e.
    # (depth - s*az)^2 = sigma^2 (r^2 - s^2), whose roots are
    # s = depth*az +/- sigma*sqrt(r^2 - depth^2) since az^2 + sigma^2 = 1
    points[n_points] = lo
    n_points += 1
    disc = r*r - depth*depth
    if disc > 0.0:
        root = sigma*sqrt(disc)
        for j in range(2):
            s = depth*az + (2*j - 1)*root
            if lo < s < hi:
                points[n_points] = s
                n_points += 1
    points[n_points] = hi
    n_points += 1

    for i in range(n_points-1):
        s0 = points[i]
        s1 = points[i+1]
        if s1 - s0 <= 0.0:
            continue
        mid = 0.5*(s0 + s1)
        rho = sqrt(_clamp(r*r - mid*mid, 0.0, r*r))
        u = (depth - mid*az)/sigma
        if u >= rho:
            # Fully wet slices: integrate pi*(r^2 - s^2) exactly
            moments[0] += M_PI*(r*r*(s1 - s0) - (s1*s1*s1 - s0*s0*s0)/3.0)
            moments[1] += M_PI*(
                0.5*r*r*(s1*s1 - s0*s0) - 0.25*(s1**4 - s0**4)
            )
        elif u > -rho:
            # Partial slices: Gauss-Legendre with s = mid - half*cos(theta)
            # which removes the square-root behaviour at the breakpoints
            half = 0.5*(s1 - s0)
            for q in range(GL_N):
                theta = 0.5*M_PI*(GL_X[q] + 1.0)
                tmp = cos(theta)
                s = mid - half*tmp
                weight = GL_W[q]*0.5*M_PI*half*sqrt(1.0 - tmp*tmp)
                rho = sqrt(_clamp(r*r - s*s, 0.0, r*r))
                _disk_segment((depth - s*az)/sigma, rho, &area, &moment)
                moments[0] += weight*area
                moments[1] += weight*area*s
                moments[2] += weight*moment


cdef void capsule_submerged(
    double radius, double half_length, const double *R, const double *c,
    double h, double *out,
) noexcept nogil:
    """Submerged volume and V*c of a capsule along local z"""
    cdef double axis[3]
    cdef double perp[3]
    cdef double sigma, k, delta, cap_z, moments[3], cap[3]
    cdef double offset
    cdef int i, end
    _axis_frame(R, axis, perp, &sigma)
    k = axis[2]/sigma
    delta = (h - c[2])/sigma
    _cylinder_moments(radius, half_length, delta, k, moments)
    out[0] = moments[0]
    for i in range(3):
        out[i+1] = moments[0]*c[i] + moments[1]*axis[i] + moments[2]*perp[i]
    for end in range(2):
        offset = half_length if end == 0 else -half_length
        cap_z = c[2] + offset*axis[2]
        if cap_z + radius <= h:
            # Fully wet hemisphere
            moments[0] = 2.0*M_PI*radius*radius*radius/3.0
            moments[1] = (1.0 if end == 0 else -1.0)*0.375*radius*moments[0]
            moments[2] = 0.0
        elif cap_z - radius >= h:
            continue
        else:
            _hemisphere_moments(
                radius, h - cap_z, axis[2], sigma,
                1.0 if end == 0 else -1.0, moments,
            )
        out[0] += moments[0]
        for i in range(3):
            out[i+1] += (
                moments[0]*(c[i] + offset*axis[i])
                + moments[1]*axis[i] + moments[2]*perp[i]
            )


# ---------------------------------------------------------------------------
# Polyhedra: BVH with divergence theorem moments
# ---------------------------------------------------------------------------

cdef inline void _add_tet(
    const double *p, const double *a, const double *b, const double *c,
    double *acc,
) noexcept nogil:
    """Accumulate 6V and 24(V c) of the tetrahedron (p, a, b, c)"""
    cdef double ax = a[0] - p[0], ay = a[1] - p[1], az = a[2] - p[2]
    cdef double bx = b[0] - p[0], by = b[1] - p[1], bz = b[2] - p[2]
    cdef double cx = c[0] - p[0], cy = c[1] - p[1], cz = c[2] - p[2]
    cdef double vol6 = (
        ax*(by*cz - bz*cy) - ay*(bx*cz - bz*cx) + az*(bx*cy - by*cx)
    )
    acc[0] += vol6
    acc[1] += vol6*(a[0] + b[0] + c[0] + p[0])
    acc[2] += vol6*(a[1] + b[1] + c[1] + p[1])
    acc[3] += vol6*(a[2] + b[2] + c[2] + p[2])


cdef inline void _lerp(
    const double *a, const double *b, double da, double db, double *out,
) noexcept nogil:
    """Point where the depth goes to zero on the edge a -> b"""
    cdef double t = da/(da - db)
    out[0] = a[0] + t*(b[0] - a[0])
    out[1] = a[1] + t*(b[1] - a[1])
    out[2] = a[2] + t*(b[2] - a[2])


cdef inline void _clip_triangle(
    const double *tri, const double *n, double t, const double *p,
    double *acc,
) noexcept nogil:
    """Accumulate the wet part of a triangle for the region n.x <= t"""
    cdef const double *v[3]
    cdef double d[3]
    cdef double q0[3]
    cdef double q1[3]
    cdef int i, wet = 0, odd = 0
    v[0] = tri
    v[1] = tri + 3
    v[2] = tri + 6
    for i in range(3):
        d[i] = t - (n[0]*v[i][0] + n[1]*v[i][1] + n[2]*v[i][2])
        if d[i] >= 0.0:
            wet += 1
    if wet == 0:
        return
    if wet == 3:
        _add_tet(p, v[0], v[1], v[2], acc)
        return
    # Rotate so that the odd vertex (alone on its side) comes first,
    # preserving the triangle orientation
    for i in range(3):
        if (d[i] >= 0.0) == (wet == 1):
            odd = i
    cdef const double *a = v[odd]
    cdef const double *b = v[(odd + 1) % 3]
    cdef const double *c = v[(odd + 2) % 3]
    cdef double da = d[odd], db = d[(odd + 1) % 3], dc = d[(odd + 2) % 3]
    _lerp(a, b, da, db, q0)
    _lerp(a, c, da, dc, q1)
    if wet == 1:
        _add_tet(p, a, q0, q1, acc)
    else:
        _add_tet(p, q0, b, c, acc)
        _add_tet(p, q0, c, q1, acc)


cdef inline void _add_node(
    const double *mom, const double *p, double *acc,
) noexcept nogil:
    """Accumulate a fully wet BVH node in O(1) from its moments"""
    cdef double vol6 = mom[0] - (mom[1]*p[0] + mom[2]*p[1] + mom[3]*p[2])
    cdef int i
    acc[0] += vol6
    for i in range(3):
        acc[i+1] += (
            mom[4+i]
            - (mom[7+3*i]*p[0] + mom[8+3*i]*p[1] + mom[9+3*i]*p[2])
            + p[i]*vol6
        )


cdef void polyhedron_submerged(
    const Bvh *bvh, int root, const double *R, const double *c, double h,
    double *out,
) noexcept nogil:
    """Submerged volume and V*c of a closed triangle mesh stored in a BVH
    (vertices in the geom frame, world = R @ local + c)"""
    cdef double n[3]
    cdef double p[3]
    cdef double acc[4]
    cdef int stack[128]
    cdef int top = 0, node, i
    cdef double t, proj, ext, vol
    cdef const double *center
    cdef const double *half
    # Water half-space in the geom frame: n.x <= t, with apex p on it
    n[0] = R[6]
    n[1] = R[7]
    n[2] = R[8]
    t = h - c[2]
    for i in range(3):
        p[i] = t*n[i]
        acc[i] = 0.0
    acc[3] = 0.0
    stack[0] = root
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        center = bvh.center + 3*node
        half = bvh.half + 3*node
        proj = n[0]*center[0] + n[1]*center[1] + n[2]*center[2]
        ext = fabs(n[0])*half[0] + fabs(n[1])*half[1] + fabs(n[2])*half[2]
        if proj - ext >= t:
            continue  # Dry
        if proj + ext <= t:
            _add_node(bvh.moments + COB_NODE_MOMENTS*node, p, acc)
        elif bvh.left[node] < 0:
            for i in range(bvh.tri_start[node],
                           bvh.tri_start[node] + bvh.tri_count[node]):
                _clip_triangle(bvh.tris + 9*i, n, t, p, acc)
        elif top < 126:
            stack[top] = bvh.left[node]
            stack[top+1] = bvh.right[node]
            top += 2
    vol = acc[0]/6.0
    out[0] = vol
    for i in range(3):
        acc[i+1] /= 24.0
    for i in range(3):
        out[i+1] = (
            R[3*i]*acc[1] + R[3*i+1]*acc[2] + R[3*i+2]*acc[3] + vol*c[i]
        )


# ---------------------------------------------------------------------------
# Model: all buoyant geoms of an animat, flattened
# ---------------------------------------------------------------------------

cdef class CobModel:
    """Flattened buoyancy geometry of a set of links (see cob_build.py).

    Geoms are grouped by link (geom_link is sorted). compute() reads the
    MuJoCo geom poses directly and writes, per link, [V, V*cx, V*cy, V*cz]
    in SI units.
    """

    def __init__(self, geometry):
        self._geometry = geometry
        self.n_links = geometry.n_links
        self.n_geoms = len(geometry.geom_kind)
        self.geom_kind = np.ascontiguousarray(geometry.geom_kind, np.intc)
        self.geom_link = np.ascontiguousarray(geometry.geom_link, np.intc)
        self.geom_id = np.ascontiguousarray(geometry.geom_id, np.intc)
        self.geom_root = np.ascontiguousarray(geometry.geom_root, np.intc)
        self.geom_size = np.ascontiguousarray(geometry.geom_size, np.double)
        self.geom_scale = np.ascontiguousarray(geometry.geom_scale, np.double)
        self.geom_volume = np.ascontiguousarray(geometry.geom_volume, np.double)
        self.geom_centroid = np.ascontiguousarray(
            geometry.geom_centroid, np.double,
        )
        self.geom_rbound = np.ascontiguousarray(geometry.geom_rbound, np.double)
        bvh = geometry.bvh
        self._center = np.ascontiguousarray(bvh['center'], np.double).reshape(-1)
        self._half = np.ascontiguousarray(bvh['half'], np.double).reshape(-1)
        self._moments = np.ascontiguousarray(
            bvh['moments'], np.double,
        ).reshape(-1)
        self._left = np.ascontiguousarray(bvh['left'], np.intc)
        self._right = np.ascontiguousarray(bvh['right'], np.intc)
        self._tri_start = np.ascontiguousarray(bvh['tri_start'], np.intc)
        self._tri_count = np.ascontiguousarray(bvh['tri_count'], np.intc)
        self._tris = np.ascontiguousarray(bvh['tris'], np.double).reshape(-1)
        self._bind_bvh()

    cdef void _bind_bvh(self):
        cdef double[::1] center = self._center
        cdef double[::1] half = self._half
        cdef double[::1] moments = self._moments
        cdef int[::1] left = self._left
        cdef int[::1] right = self._right
        cdef int[::1] tri_start = self._tri_start
        cdef int[::1] tri_count = self._tri_count
        cdef double[::1] tris = self._tris
        self.bvh.center = &center[0] if center.shape[0] else NULL
        self.bvh.half = &half[0] if half.shape[0] else NULL
        self.bvh.moments = &moments[0] if moments.shape[0] else NULL
        self.bvh.left = &left[0] if left.shape[0] else NULL
        self.bvh.right = &right[0] if right.shape[0] else NULL
        self.bvh.tri_start = &tri_start[0] if tri_start.shape[0] else NULL
        self.bvh.tri_count = &tri_count[0] if tri_count.shape[0] else NULL
        self.bvh.tris = &tris[0] if tris.shape[0] else NULL

    cdef void geom_submerged(
        self, int g, const double *R, const double *c, double h,
        double *out,
    ) noexcept nogil:
        """Submerged (V, V*c) of geom g given its world pose (SI units)"""
        cdef int kind = self.geom_kind[g], i
        cdef double depth, lc[3]
        cdef double rbound = self.geom_rbound[g]
        cdef const double *size = &self.geom_size[g, 0]
        cdef const double *cl = &self.geom_centroid[g, 0]
        # World centroid of the full geom, used for the quick tests
        for i in range(3):
            lc[i] = c[i] + R[3*i]*cl[0] + R[3*i+1]*cl[1] + R[3*i+2]*cl[2]
        depth = h - lc[2]
        if depth <= -rbound:
            out[0] = 0.0
            out[1] = 0.0
            out[2] = 0.0
            out[3] = 0.0
            return
        if depth >= rbound:
            out[0] = self.geom_volume[g]
            for i in range(3):
                out[i+1] = out[0]*lc[i]
            return
        if kind == COB_SPHERE:
            sphere_submerged(size[0], c, h, out)
        elif kind == COB_ELLIPSOID:
            ellipsoid_submerged(size, R, c, h, out)
        elif kind == COB_CYLINDER:
            cylinder_submerged(size[0], size[1], R, c, h, out)
        elif kind == COB_CAPSULE:
            capsule_submerged(size[0], size[1], R, c, h, out)
        else:
            polyhedron_submerged(&self.bvh, self.geom_root[g], R, c, h, out)

    cdef void compute(
        self,
        const double *geom_xpos,
        const double *geom_xmat,
        double inv_meters,
        const double *surfaces,
        double *out,
    ) noexcept nogil:
        """Per-link submerged [V, V*c] (n_links x 4) for surfaces[link]"""
        cdef int g, link, i, gid
        cdef double c[3]
        cdef double tmp[4]
        cdef double scale
        for i in range(4*self.n_links):
            out[i] = 0.0
        for g in range(self.n_geoms):
            link = self.geom_link[g]
            gid = self.geom_id[g]
            for i in range(3):
                c[i] = geom_xpos[3*gid+i]*inv_meters
            self.geom_submerged(g, geom_xmat + 9*gid, c, surfaces[link], tmp)
            scale = self.geom_scale[g]
            for i in range(4):
                out[4*link+i] += scale*tmp[i]

    def submerged(self, int g, R, c, double h):
        """Python access for tests: (V, centroid) of geom g"""
        cdef double[::1] rot = np.ascontiguousarray(R, np.double).reshape(-1)
        cdef double[::1] pos = np.ascontiguousarray(c, np.double).reshape(-1)
        cdef double out[4]
        self.geom_submerged(g, &rot[0], &pos[0], h, out)
        centroid = (
            np.array([out[1], out[2], out[3]])/out[0]
            if out[0] > 0 else np.array(c, dtype=np.double)
        )
        return out[0], centroid

    def links_submerged(self, geom_xpos, geom_xmat, surfaces, double inv_meters=1):
        """Python access for tests/benchmarks: (n_links, 4) array"""
        cdef double[::1] xpos = np.ascontiguousarray(geom_xpos, np.double).reshape(-1)
        cdef double[::1] xmat = np.ascontiguousarray(geom_xmat, np.double).reshape(-1)
        cdef double[::1] surf = np.ascontiguousarray(surfaces, np.double).reshape(-1)
        result = np.zeros([self.n_links, 4])
        cdef double[:, ::1] res = result
        self.compute(&xpos[0], &xmat[0], inv_meters, &surf[0], &res[0, 0])
        return result


def primitive_submerged(int kind, size, R, c, double h):
    """Direct access to the analytic kernels for tests: (V, V*c)"""
    cdef double[::1] sz = np.ascontiguousarray(size, np.double).reshape(-1)
    cdef double[::1] rot = np.ascontiguousarray(R, np.double).reshape(-1)
    cdef double[::1] pos = np.ascontiguousarray(c, np.double).reshape(-1)
    cdef double out[4]
    if kind == COB_SPHERE:
        sphere_submerged(sz[0], &pos[0], h, out)
    elif kind == COB_ELLIPSOID:
        ellipsoid_submerged(&sz[0], &rot[0], &pos[0], h, out)
    elif kind == COB_CYLINDER:
        cylinder_submerged(sz[0], sz[1], &rot[0], &pos[0], h, out)
    elif kind == COB_CAPSULE:
        capsule_submerged(sz[0], sz[1], &rot[0], &pos[0], h, out)
    else:
        raise ValueError(f'Unknown primitive kind {kind}')
    return out[0], np.array([out[1], out[2], out[3]])


def benchmark_geom(CobModel model, int g, R, c, heights, int repeats=1000):
    """Average time [ns] of one geom_submerged call, looping in C over the
    given water heights (for benchmarks)"""
    import time  # pylint: disable=import-outside-toplevel
    cdef double[::1] rot = np.ascontiguousarray(R, np.double).reshape(-1)
    cdef double[::1] pos = np.ascontiguousarray(c, np.double).reshape(-1)
    cdef double[::1] hs = np.ascontiguousarray(heights, np.double)
    cdef double out[4]
    cdef double checksum = 0.0
    cdef int i, j, n = hs.shape[0]
    tic = time.perf_counter_ns()
    with nogil:
        for i in range(repeats):
            for j in range(n):
                model.geom_submerged(g, &rot[0], &pos[0], hs[j], out)
                checksum += out[0]
    toc = time.perf_counter_ns()
    return (toc - tic)/(repeats*n), checksum
