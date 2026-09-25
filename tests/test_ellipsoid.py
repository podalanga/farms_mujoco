"""Tests for the ellipsoid approximation, added mass and fluid wrench"""

import numpy as np
import pytest

from farms_mujoco.swimming import ellipsoid_fit as ef
from farms_mujoco.swimming.ellipsoid_model import EllipsoidModel
from farms_mujoco.swimming.cob_build import (
    SPHERE, CYLINDER, POLYHEDRON, GeomInfo, box_triangles,
)
from test_cob import random_rotation


def geom(kind, size, pos=(0, 0, 0), rot=None, tris=None):
    """GeomInfo helper"""
    return GeomInfo(
        0, 0, kind, np.pad(np.array(size, float), (0, 3-len(size))),
        np.array(pos, float), np.eye(3) if rot is None else rot, tris,
    )


def contains(ellipsoid, points, tol=1e-3):
    """Whether all points lie in the ellipsoid (with tolerance)"""
    local = (points - ellipsoid.center) @ ellipsoid.rotation/ellipsoid.axes
    return np.all(np.sum(local**2, axis=1) <= 1 + tol)


def test_mvee_sphere_and_box():
    """MVEE of a sphere is itself, of a box the sqrt(3) scaled ellipsoid"""
    e = ef.fit_mvee([geom(SPHERE, [0.1], pos=[1, 2, 3])])
    assert np.allclose(e.axes, 0.1, rtol=2e-3)
    assert np.allclose(e.center, [1, 2, 3], atol=1e-4)
    half = np.array([0.3, 0.2, 0.1])
    rot = random_rotation()
    e = ef.fit_mvee([geom(POLYHEDRON, half, rot=rot, tris=box_triangles(half))])
    assert np.allclose(np.sort(e.axes), np.sort(half*np.sqrt(3)), rtol=1e-3)
    assert contains(e, box_triangles(half).reshape(-1, 3) @ rot.T)


def test_mvee_contains_link_geoms():
    """MVEE of a multi-geom link contains every geom surface point"""
    geoms = [
        geom(CYLINDER, [0.0175, 0.0326], rot=random_rotation()),
        geom(POLYHEDRON, [0.0175, 0.0325, 0.0225], pos=[0, 0.02, 0],
             tris=box_triangles([0.0175, 0.0325, 0.0225])),
    ]
    e = ef.fit_mvee(geoms)
    points = np.concatenate([ef.geom_surface_points(g) for g in geoms])
    assert contains(e, points)
    assert e.volume < 4/3*np.pi*np.max(np.linalg.norm(points - e.center, axis=1))**3


def test_inertia_fit():
    """Inertia ellipsoid recovers the axes of a solid ellipsoid"""
    axes = np.array([0.3, 0.2, 0.1])
    mass = 2.0
    inertia = mass/5*np.array([
        axes[1]**2 + axes[2]**2, axes[0]**2 + axes[2]**2, axes[0]**2 + axes[1]**2,
    ])
    e = ef.fit_inertia(mass, inertia, [0, 0, 0], [1, 0, 0, 0])
    assert np.allclose(e.axes, axes)


def test_added_mass_sphere():
    """Sphere: added mass is half the displaced fluid mass, no inertia"""
    mass, inertia = ef.added_mass([0.1, 0.1, 0.1], density=1000)
    volume = 4/3*np.pi*0.1**3
    assert np.allclose(mass, 0.5*1000*volume, rtol=1e-8)
    assert np.allclose(inertia, 0, atol=1e-12)


@pytest.mark.parametrize('ratio,k1,k2,k_rot', [
    # Lamb coefficients for prolate spheroids, from the closed-form
    # alpha0/beta0 expressions (Lamb 1932, §115)
    (2.0, 0.2100, 0.7042, 0.2394),
    (5.0, 0.0591, 0.8943, 0.7000),
    (10.0, 0.0207, 0.9602, 0.8835),
])
def test_added_mass_prolate_spheroid(ratio, k1, k2, k_rot):
    """Prolate spheroid matches Lamb's tabulated inertia coefficients"""
    a, b = ratio, 1.0
    mass, inertia = ef.added_mass([a, b, b], density=1.0)
    volume = 4/3*np.pi*a*b*b
    assert mass[0]/volume == pytest.approx(k1, abs=2e-3)
    assert mass[1]/volume == pytest.approx(k2, abs=2e-3)
    assert mass[2]/volume == pytest.approx(k2, abs=2e-3)
    # Rotational coefficient relative to the displaced fluid inertia
    fluid_inertia = volume*(a*a + b*b)/5
    assert inertia[1]/fluid_inertia == pytest.approx(k_rot, abs=1e-2)
    assert inertia[0] == pytest.approx(0, abs=1e-9)


def spheroid_model(added_mass='off', coefficients=(1.0, 0.0, 0.0)):
    """Model for a single prolate spheroid along x"""
    e = ef.LinkEllipsoid(np.zeros(3), np.eye(3), np.array([0.2, 0.05, 0.05]))
    return EllipsoidModel([e], coefficients, 1e-3, added_mass), e


def test_form_drag():
    """Form drag = 1/2 rho C A |v| v, with A the projected area"""
    model, e = spheroid_model()
    rot = random_rotation()
    for axis, area in ((0, np.pi*0.05*0.05), (1, np.pi*0.2*0.05)):
        velocity = 0.3*rot[:, axis]
        force, torque = model.compute_wrench(
            0, rot, np.zeros(3), np.zeros(3), velocity, np.zeros(3),
            density=1000,
        )
        assert np.allclose(force, -0.5*1000*area*0.3*velocity, rtol=1e-10)
        assert np.allclose(torque, 0, atol=1e-12)
    # Scaled by the submerged fraction
    force_half, _ = model.compute_wrench(
        0, rot, np.zeros(3), np.zeros(3), velocity, np.zeros(3),
        density=1000, fraction=0.5,
    )
    assert np.allclose(force_half, 0.5*force)


def test_munk_moment():
    """Oblique translation of a spheroid produces the destabilising Munk
    moment (m2 - m1) u v about the transverse axis"""
    model, e = spheroid_model(added_mass='implicit', coefficients=(0, 0, 0))
    mass, _ = ef.added_mass(e.axes, density=1000)
    velocity = np.array([1.0, 0.5, 0.0])
    force, torque = model.compute_wrench(
        0, np.eye(3), np.zeros(3), np.zeros(3), velocity, np.zeros(3),
        density=1000,
    )
    assert np.allclose(force, 0, atol=1e-12)
    assert torque[2] == pytest.approx((mass[0] - mass[1])*1.0*0.5, rel=1e-10)


def test_wrench_about_com():
    """Force at an offset ellipsoid centre gives the lever arm torque"""
    e = ef.LinkEllipsoid(np.array([0.1, 0, 0]), np.eye(3), np.full(3, 0.05))
    model = EllipsoidModel([e], (1.0, 0.0, 0.0), 1e-3, 'off')
    velocity = np.array([0, 0.2, 0])
    force, torque = model.compute_wrench(
        0, np.eye(3), np.zeros(3), np.zeros(3), velocity, np.zeros(3),
    )
    assert np.allclose(torque, np.cross([0.1, 0, 0], force))
