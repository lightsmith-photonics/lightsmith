"""Rasterise profiles and structures onto grids (shared by grid-based solvers).

All functions return NumPy arrays; solvers move them to their device. Fill
fractions are computed by supersampling each cell, giving smooth (subpixel)
permittivity at interfaces.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .problems import Ball, Box, Circle, Ellipse, Polygon, Prism, Rectangle, Structure, Structure2D


def inside_profile(profile, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of points (x, y) inside a 2D profile (arrays broadcast)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if isinstance(profile, Rectangle):
        (cx, cy), (w, h) = profile.center_nm, profile.size_nm
        return (np.abs(x - cx) <= w / 2) & (np.abs(y - cy) <= h / 2)
    if isinstance(profile, Circle):
        cx, cy = profile.center_nm
        return (x - cx) ** 2 + (y - cy) ** 2 <= profile.radius_nm**2
    if isinstance(profile, Ellipse):
        cx, cy = profile.center_nm
        a, b = profile.semi_axes_nm
        t = np.radians(profile.angle_deg)
        u = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
        v = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
        return (u / a) ** 2 + (v / b) ** 2 <= 1
    if isinstance(profile, Polygon):
        vx, vy = np.asarray(profile.vertices_nm, float).T
        inside = np.zeros(x.shape, dtype=bool)
        j = len(vx) - 1
        for i in range(len(vx)):  # even-odd ray casting
            crosses = (vy[i] > y) != (vy[j] > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = (vx[j] - vx[i]) * (y - vy[i]) / (vy[j] - vy[i]) + vx[i]
            inside ^= crosses & (x < xint)
            j = i
        return inside
    raise TypeError(f"unknown profile {type(profile).__name__}")


def profile_bounds(profile) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) enclosing a 2D profile."""
    if isinstance(profile, Rectangle):
        (cx, cy), (w, h) = profile.center_nm, profile.size_nm
        return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    if isinstance(profile, Circle):
        (cx, cy), r = profile.center_nm, profile.radius_nm
        return cx - r, cx + r, cy - r, cy + r
    if isinstance(profile, Ellipse):
        (cx, cy), r = profile.center_nm, max(profile.semi_axes_nm)
        return cx - r, cx + r, cy - r, cy + r
    if isinstance(profile, Polygon):
        vx, vy = np.asarray(profile.vertices_nm, float).T
        return float(vx.min()), float(vx.max()), float(vy.min()), float(vy.max())
    raise TypeError(f"unknown profile {type(profile).__name__}")


def geometry_xy_bounds(geometry) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) enclosing a 3D geometry in plan view."""
    if isinstance(geometry, Box):
        (cx, cy, _), (sx, sy, _) = geometry.center_nm, geometry.size_nm
        return cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    if isinstance(geometry, Ball):
        (cx, cy, _), r = geometry.center_nm, geometry.radius_nm
        return cx - r, cx + r, cy - r, cy + r
    if isinstance(geometry, Prism):
        x0, x1, y0, y1 = profile_bounds(geometry.profile)
        # a negative sidewall angle widens the profile with height
        grow = abs(np.tan(np.radians(geometry.sidewall_angle_deg))) * (geometry.z_max_nm - geometry.z_min_nm)
        return x0 - grow, x1 + grow, y0 - grow, y1 + grow
    raise TypeError(f"unknown geometry {type(geometry).__name__}")


def _offset_profile_inside(profile, x, y, inset: np.ndarray | float):
    """Inside test for a profile shrunk by `inset` (used for sloped sidewalls). Approximate for polygons."""
    if np.all(np.asarray(inset) == 0):
        return inside_profile(profile, x, y)
    if isinstance(profile, Rectangle):
        (cx, cy), (w, h) = profile.center_nm, profile.size_nm
        return (np.abs(x - cx) <= w / 2 - inset) & (np.abs(y - cy) <= h / 2 - inset)
    if isinstance(profile, Circle):
        cx, cy = profile.center_nm
        return np.hypot(x - cx, y - cy) <= profile.radius_nm - inset
    # Polygons/ellipses: scale about the centroid as an approximation.
    return inside_profile(profile, x, y)


def inside_geometry(geometry, x, y, z) -> np.ndarray:
    x, y, z = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, y, z)))
    if isinstance(geometry, Box):
        c, s = np.asarray(geometry.center_nm), np.asarray(geometry.size_nm)
        return (np.abs(x - c[0]) <= s[0] / 2) & (np.abs(y - c[1]) <= s[1] / 2) & (np.abs(z - c[2]) <= s[2] / 2)
    if isinstance(geometry, Ball):
        c = geometry.center_nm
        return (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= geometry.radius_nm**2
    if isinstance(geometry, Prism):
        in_z = (z >= geometry.z_min_nm) & (z <= geometry.z_max_nm)
        inset = np.tan(np.radians(geometry.sidewall_angle_deg)) * np.clip(z - geometry.z_min_nm, 0, None)
        return in_z & _offset_profile_inside(geometry.profile, x, y, inset)
    raise TypeError(f"unknown geometry {type(geometry).__name__}")


def _sub_offsets(spacing: float, n: int) -> np.ndarray:
    return ((np.arange(n) + 0.5) / n - 0.5) * spacing


def fill_fractions_2d(profiles: Sequence, x: np.ndarray, y: np.ndarray, subsamples: int = 4) -> np.ndarray:
    """Fill fraction of each profile in each cell of the tensor grid (x, y) of cell centres.

    Returns (n_profiles, nx, ny). Later profiles override earlier ones (painter's order).
    """
    dx = np.diff(x).mean() if len(x) > 1 else 1.0
    dy = np.diff(y).mean() if len(y) > 1 else 1.0
    X, Y = np.meshgrid(x, y, indexing="ij")
    frac = np.zeros((len(profiles),) + X.shape)
    for ox in _sub_offsets(dx, subsamples):
        for oy in _sub_offsets(dy, subsamples):
            owner = np.full(X.shape, -1)
            for i, p in enumerate(profiles):
                owner[inside_profile(p, X + ox, Y + oy)] = i
            for i in range(len(profiles)):
                frac[i] += owner == i
    return frac / subsamples**2


def fill_fractions_3d(geometries: Sequence, x, y, z, subsamples: int = 2) -> np.ndarray:
    """Like :func:`fill_fractions_2d` for 3D geometries; returns (n_geom, nx, ny, nz)."""
    d = [np.diff(a).mean() if len(a) > 1 else 1.0 for a in (x, y, z)]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    frac = np.zeros((len(geometries),) + X.shape)
    for ox in _sub_offsets(d[0], subsamples):
        for oy in _sub_offsets(d[1], subsamples):
            for oz in _sub_offsets(d[2], subsamples):
                owner = np.full(X.shape, -1)
                for i, g in enumerate(geometries):
                    owner[inside_geometry(g, X + ox, Y + oy, Z + oz)] = i
                for i in range(len(geometries)):
                    frac[i] += owner == i
    return frac / subsamples**3


def permittivity_2d(structures: Sequence[Structure2D], background, x, y, wavelength_nm: float,
                    subsamples: int = 4) -> np.ndarray:
    """Complex relative permittivity on a 2D grid, volume-averaged at interfaces."""
    eps_bg = complex(background.eps(np.array([wavelength_nm]))[0])
    frac = fill_fractions_2d([s.profile for s in structures], x, y, subsamples)
    eps = np.full(frac.shape[1:], eps_bg, dtype=complex)
    for f, s in zip(frac, structures):
        eps += f * (complex(s.material.eps(np.array([wavelength_nm]))[0]) - eps_bg)
    return eps


def permittivity_3d(structures: Sequence[Structure], background, x, y, z, wavelength_nm: float,
                    subsamples: int = 2) -> np.ndarray:
    eps_bg = complex(background.eps(np.array([wavelength_nm]))[0])
    frac = fill_fractions_3d([s.geometry for s in structures], x, y, z, subsamples)
    eps = np.full(frac.shape[1:], eps_bg, dtype=complex)
    for f, s in zip(frac, structures):
        eps += f * (complex(s.material.eps(np.array([wavelength_nm]))[0]) - eps_bg)
    return eps
