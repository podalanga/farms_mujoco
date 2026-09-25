"""Load-time ellipsoid approximation of links for the fluid model.

- mvee: minimum-volume enclosing ellipsoid of the link's buoyant geoms
  (Khachiyan's algorithm on the convex hull of surface samples), as in
  Stonefish's ellipsoidal approximation.
- inertia: ellipsoid of uniform density with the same mass and principal
  inertia as the MuJoCo body (as in MuJoCo's fluidshape=ellipsoid).

Added mass coefficients follow Lamb (Hydrodynamics, 1932, §114-115):
kappa_i = abc * int_0^inf dl / ((r_i^2 + l) sqrt((a^2+l)(b^2+l)(c^2+l))),
m_i = rho V kappa_i / (2 - kappa_i) and
I_i = rho V/5 (r_j^2 - r_k^2)^2 (kappa_k - kappa_j)
      / (2 (r_j^2 - r_k^2) + (r_j^2 + r_k^2)(kappa_j - kappa_k)).
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.spatial import ConvexHull, QhullError

from .cob_build import SPHERE, ELLIPSOID, CYLINDER, CAPSULE, quat2mat


@dataclass
class LinkEllipsoid:
    """Ellipsoid in a link frame: x = center + rotation @ diag(axes) @ u"""
    center: np.ndarray
    rotation: np.ndarray
    axes: np.ndarray

    @property
    def volume(self):
        """Ellipsoid volume"""
        return 4/3*np.pi*np.prod(self.axes)


def fibonacci_sphere(n):
    """n quasi-uniform unit vectors"""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n)
    theta = np.pi*(1 + 5**0.5)*i
    return np.stack([
        np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi),
    ], axis=1)


def geom_surface_points(geom, n=256):
    """Points on the surface of a geom, in the link frame"""
    dirs = fibonacci_sphere(n)
    size = geom.size
    if geom.kind == SPHERE:
        points = size[0]*dirs
    elif geom.kind == ELLIPSOID:
        points = size[:3]*dirs
    elif geom.kind in (CYLINDER, CAPSULE):
        radius, half = size[0], size[1]
        phi = 2*np.pi*np.arange(64)/64
        ring = np.stack([radius*np.cos(phi), radius*np.sin(phi)], axis=1)
        if geom.kind == CYLINDER:
            points = np.concatenate([
                np.c_[ring, np.full(64, -half)], np.c_[ring, np.full(64, half)],
            ])
        else:
            caps = radius*dirs
            caps[:, 2] += np.where(caps[:, 2] >= 0, half, -half)
            points = caps
    else:
        points = geom.tris.reshape(-1, 3)
    return points @ geom.rot.T + geom.pos


def mvee(points, tolerance=1e-6, max_iterations=10000):
    """Minimum-volume enclosing ellipsoid (Khachiyan). Returns the centre c
    and matrix A such that (x - c)^T A (x - c) <= 1"""
    try:
        points = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        pass
    n, d = points.shape
    q = np.c_[points, np.ones(n)].T
    u = np.full(n, 1/n)
    for _ in range(max_iterations):
        x = q @ (u[:, None]*q.T)
        m = np.einsum('ij,ji->i', q.T, np.linalg.solve(x, q))
        j = np.argmax(m)
        step = (m[j] - d - 1)/((d + 1)*(m[j] - 1))
        new_u = (1 - step)*u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tolerance:
            u = new_u
            break
        u = new_u
    center = points.T @ u
    cov = points.T @ (u[:, None]*points) - np.outer(center, center)
    return center, np.linalg.inv(cov)/d


def fit_mvee(geoms):
    """Minimum-volume enclosing ellipsoid of a link's geoms"""
    points = np.concatenate([geom_surface_points(geom) for geom in geoms])
    center, matrix = mvee(points)
    eigval, eigvec = np.linalg.eigh(matrix)
    if np.linalg.det(eigvec) < 0:
        eigvec[:, 0] *= -1
    return LinkEllipsoid(center, eigvec, 1/np.sqrt(eigval))


def fit_inertia(mass, inertia, ipos, iquat):
    """Uniform-density ellipsoid with the given mass and principal inertia
    (MuJoCo body_inertia in the frame (body_ipos, body_iquat))"""
    ix, iy, iz = inertia
    axes = np.sqrt(np.maximum(5/(2*mass)*np.array([
        iy + iz - ix, ix + iz - iy, ix + iy - iz,
    ]), 1e-12))
    return LinkEllipsoid(np.array(ipos, float), quat2mat(iquat), axes)


def lamb_kappa(axes):
    """Lamb's shape integrals (alpha0, beta0, gamma0)"""
    a, b, c = axes
    scale = max(axes)**2

    def integrand(l, r):
        l *= scale
        return scale/((r*r + l)*np.sqrt((a*a + l)*(b*b + l)*(c*c + l)))

    return np.array([
        a*b*c*quad(integrand, 0, np.inf, args=(r,), limit=200)[0] for r in axes
    ])


def added_mass(axes, density=1.0):
    """Translational and rotational added mass of an ellipsoid (diagonal,
    ellipsoid frame) per unit fluid density by default"""
    axes = np.asarray(axes, dtype=float)
    kappa = lamb_kappa(axes)
    volume = 4/3*np.pi*np.prod(axes)
    mass = density*volume*kappa/(2 - kappa)
    inertia = np.zeros(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        rj2, rk2 = axes[j]**2, axes[k]**2
        diff = rj2 - rk2
        denominator = 2*diff + (rj2 + rk2)*(kappa[j] - kappa[k])
        if abs(diff) > 1e-9*max(rj2, rk2) and abs(denominator) > 1e-300:
            inertia[i] = (
                density*volume/5*diff**2*(kappa[k] - kappa[j])/denominator
            )
    return mass, inertia


def angular_drag_inertia(axes):
    """MuJoCo's angular drag moments I_D,i = 8pi/15 r_i max(r_j, r_k)^4"""
    return np.array([
        8*np.pi/15*axes[i]*max(axes[(i + 1) % 3], axes[(i + 2) % 3])**4
        for i in range(3)
    ])


def projected_area_coefficients(axes):
    """(bc, ac, ab) such that the projected area along unit u is
    pi*sqrt(sum((coef*u)^2))"""
    a, b, c = axes
    return np.array([b*c, a*c, a*b])


def fit_link_ellipsoid(method, geoms, mass=None, inertia=None, ipos=None, iquat=None):
    """Ellipsoid of a link with the requested method"""
    if method == 'mvee':
        if not geoms:
            raise ValueError('mvee ellipsoid requires buoyant geoms')
        return fit_mvee(geoms)
    if method == 'inertia':
        return fit_inertia(mass, inertia, ipos, iquat)
    raise ValueError(f'Unknown ellipsoid fit {method!r}')
