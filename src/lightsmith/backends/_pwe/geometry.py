"""Unit-cell geometry for the plane-wave-expansion solver.

Lengths inside the engine are in units of the lattice constant ``a`` (the length
of the first lattice vector), so reciprocal vectors carry the 2 pi:
``b_i . a_j = 2 pi delta_ij``. Shapes keep their nm coordinates and are scaled
here. The permittivity is periodic: every inclusion is replicated on the lattice,
so shapes may straddle the cell boundary.

Two routes to eps:

- analytic: Fourier coefficients of each inclusion's indicator function
  (circles, ellipses, rectangles, polygons, balls, boxes, vertical prisms), exact
  for non-overlapping inclusions;
- raster: supersampled painter's-order raster of the oblique cell (shared inside
  tests from :mod:`lightsmith.raster`), used for overlaps, sloped sidewalls and
  for the real-space smoothed tensors of the iterative solver.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ...problems import Ball, Box, Circle, Ellipse, Polygon, Prism, Rectangle
from ...raster import inside_geometry, inside_profile


@dataclass(frozen=True)
class Lattice:
    """Bravais lattice: rows of `vectors` are a_i in units of a; `a_nm` is the unit."""

    vectors: np.ndarray  # (d, d)
    a_nm: float

    @property
    def dim(self) -> int:
        return self.vectors.shape[0]

    @property
    def reciprocal(self) -> np.ndarray:
        """Rows b_i (units 1/a, including 2 pi)."""
        return 2 * np.pi * np.linalg.inv(self.vectors).T

    @property
    def cell_measure(self) -> float:
        """Area (2D) or volume (3D) of the unit cell in units of a^d."""
        return float(abs(np.linalg.det(self.vectors)))

    @property
    def vectors_nm(self) -> np.ndarray:
        return self.vectors * self.a_nm


@dataclass
class Inclusion:
    shape: object  # 2D profile or 3D geometry from lightsmith.problems (nm)
    eps: float


@dataclass
class Cell:
    lattice: Lattice
    eps_background: float
    inclusions: list[Inclusion] = field(default_factory=list)

    @property
    def eps_values(self) -> list[float]:
        return [self.eps_background] + [i.eps for i in self.inclusions]


# --- analytic Fourier transforms of indicator functions ----------------------


def bessel_j1(x: np.ndarray) -> np.ndarray:
    """J1(x) from Bessel's integral by the (spectrally accurate) periodic trapezoid rule."""
    x = np.asarray(x, dtype=float)
    n = max(64, int(np.abs(x).max(initial=0.0)) + 64)
    tau = (np.arange(n) + 0.5) * (2 * np.pi / n)
    out = np.empty(x.shape)
    flat, res = x.ravel(), out.ravel()
    step = max(1, 4_000_000 // n)
    for s in range(0, flat.size, step):
        res[s:s + step] = np.cos(tau[None, :] - flat[s:s + step, None] * np.sin(tau[None, :])).mean(axis=1)
    return out


def _jinc(x: np.ndarray) -> np.ndarray:
    """2 J1(x) / x, equal to 1 at x = 0."""
    small = np.abs(x) < 1e-8
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x**2 / 8, 2 * bessel_j1(xs) / xs)


def _sinc(x: np.ndarray) -> np.ndarray:
    return np.sinc(np.asarray(x) / np.pi)  # sin(x)/x


def _polygon_ft(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """∫ e^{-i q.r} over a simple polygon (divergence theorem, edge by edge). q: (n, 2)."""
    d = np.roll(v, -1, axis=0) - v  # edges
    mid = v + d / 2
    area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
    sign = 1.0 if area >= 0 else -1.0  # orientation -> outward normal
    q2 = np.sum(q**2, axis=1)
    zero = q2 < 1e-24
    q2s = np.where(zero, 1.0, q2)
    qn = q[:, None, 0] * d[None, :, 1] - q[:, None, 1] * d[None, :, 0]  # q . n_j L_j (CCW)
    ph = np.exp(-1j * (q @ mid.T)) * _sinc((q @ d.T) / 2)
    s = sign * 1j / q2s * np.sum(qn * ph, axis=1)
    return np.where(zero, abs(area), s)


def _profile_ft(p, q: np.ndarray) -> np.ndarray:
    """Fourier transform (nm^2) of a 2D profile indicator at wavevectors q (n, 2) in 1/nm."""
    if isinstance(p, Circle):
        c = np.asarray(p.center_nm)
        return np.pi * p.radius_nm**2 * _jinc(np.linalg.norm(q, axis=1) * p.radius_nm) * np.exp(-1j * q @ c)
    if isinstance(p, Ellipse):
        c = np.asarray(p.center_nm)
        a, b = p.semi_axes_nm
        t = math.radians(p.angle_deg)
        qu = q[:, 0] * math.cos(t) + q[:, 1] * math.sin(t)
        qv = -q[:, 0] * math.sin(t) + q[:, 1] * math.cos(t)
        return np.pi * a * b * _jinc(np.hypot(a * qu, b * qv)) * np.exp(-1j * q @ c)
    if isinstance(p, Rectangle):
        c = np.asarray(p.center_nm)
        w, h = p.size_nm
        return w * h * _sinc(q[:, 0] * w / 2) * _sinc(q[:, 1] * h / 2) * np.exp(-1j * q @ c)
    if isinstance(p, Polygon):
        return _polygon_ft(np.asarray(p.vertices_nm, float), q)
    raise TypeError(f"no analytic transform for {type(p).__name__}")


def _geometry_ft(g, q: np.ndarray) -> np.ndarray:
    """Fourier transform (nm^3) of a 3D geometry indicator at q (n, 3) in 1/nm."""
    if isinstance(g, Ball):
        c, r = np.asarray(g.center_nm), g.radius_nm
        x = np.linalg.norm(q, axis=1) * r
        small = x < 1e-4
        xs = np.where(small, 1.0, x)
        f = np.where(small, 1 - x**2 / 10, 3 * (np.sin(xs) - xs * np.cos(xs)) / xs**3)
        return 4 / 3 * np.pi * r**3 * f * np.exp(-1j * q @ c)
    if isinstance(g, Box):
        c, s = np.asarray(g.center_nm), np.asarray(g.size_nm)
        return np.prod(s) * np.prod(_sinc(q * s / 2), axis=1) * np.exp(-1j * q @ c)
    if isinstance(g, Prism):
        if g.sidewall_angle_deg != 0:
            raise TypeError("no analytic transform for sloped sidewalls")
        h = g.z_max_nm - g.z_min_nm
        zc = (g.z_max_nm + g.z_min_nm) / 2
        return _profile_ft(g.profile, q[:, :2]) * h * _sinc(q[:, 2] * h / 2) * np.exp(-1j * q[:, 2] * zc)
    raise TypeError(f"no analytic transform for {type(g).__name__}")


def shape_ft(shape, q: np.ndarray) -> np.ndarray:
    return _profile_ft(shape, q) if q.shape[1] == 2 else _geometry_ft(shape, q)


def has_analytic_ft(shape) -> bool:
    if isinstance(shape, Prism):
        return shape.sidewall_angle_deg == 0
    return isinstance(shape, (Circle, Ellipse, Rectangle, Polygon, Ball, Box))


# --- rasterisation of the oblique periodic cell ---------------------------------


def _bounds(shape) -> tuple[np.ndarray, float]:
    """Centre (nm) and bounding radius (nm) of a profile or geometry."""
    if isinstance(shape, Circle):
        return np.asarray(shape.center_nm, float), shape.radius_nm
    if isinstance(shape, Ellipse):
        return np.asarray(shape.center_nm, float), max(shape.semi_axes_nm)
    if isinstance(shape, Rectangle):
        return np.asarray(shape.center_nm, float), math.hypot(*shape.size_nm) / 2
    if isinstance(shape, Polygon):
        v = np.asarray(shape.vertices_nm, float)
        c = v.mean(axis=0)
        return c, float(np.linalg.norm(v - c, axis=1).max())
    if isinstance(shape, Ball):
        return np.asarray(shape.center_nm, float), shape.radius_nm
    if isinstance(shape, Box):
        return np.asarray(shape.center_nm, float), float(np.linalg.norm(shape.size_nm)) / 2
    if isinstance(shape, Prism):
        c2, r2 = _bounds(shape.profile)
        h = (shape.z_max_nm - shape.z_min_nm) / 2
        return np.array([c2[0], c2[1], (shape.z_max_nm + shape.z_min_nm) / 2]), math.hypot(r2, h)
    raise TypeError(f"unknown shape {type(shape).__name__}")


def _images(shape, lat: Lattice) -> list[np.ndarray]:
    """Lattice translations (nm) of `shape` that can reach the cell [0, 1)^d (fractional)."""
    c, r = _bounds(shape)
    inv = np.linalg.inv(lat.vectors_nm)
    fc = c[: lat.dim] @ inv
    ext = r * np.linalg.norm(inv, axis=0)  # fractional half-extent along each a_i
    ranges = [range(math.floor(fc[i] - ext[i] - 1), math.ceil(fc[i] + ext[i]) + 1) for i in range(lat.dim)]
    return [np.asarray(n, float) @ lat.vectors_nm for n in itertools.product(*ranges)]


def _inside(shape, pts: np.ndarray) -> np.ndarray:
    if pts.shape[-1] == 2:
        return inside_profile(shape, pts[..., 0], pts[..., 1])
    return inside_geometry(shape, pts[..., 0], pts[..., 1], pts[..., 2])


@dataclass
class Raster:
    grid: tuple[int, ...]
    eps_mean: np.ndarray  # <eps> per pixel
    inv_eps_mean: np.ndarray  # <1/eps> per pixel
    overlap: bool  # some point lies in two inclusions (or two images of one)


def rasterize(cell: Cell, grid: Sequence[int], subsamples: int) -> Raster:
    """Supersampled raster of the periodic cell; pixel centres at fractional j / N_i."""
    lat = cell.lattice
    d = lat.dim
    grid = tuple(int(n) for n in grid)
    u = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing="ij"), axis=-1)  # (*grid, d)
    eps = np.asarray(cell.eps_values, float)
    images = [_images(inc.shape, lat) for inc in cell.inclusions]
    sub = (np.arange(subsamples) + 0.5) / subsamples - 0.5
    eps_sum = np.zeros(grid)
    inv_sum = np.zeros(grid)
    overlap = False
    for off in itertools.product(sub, repeat=d):
        pts = (u + np.asarray(off) / np.asarray(grid)) @ lat.vectors_nm
        owner = np.zeros(grid, dtype=int)  # 0 = background
        count = np.zeros(grid, dtype=np.int16)
        for i, inc in enumerate(cell.inclusions):
            hit = np.zeros(grid, dtype=bool)
            for t in images[i]:
                m = _inside(inc.shape, pts + t)
                count += m
                hit |= m
            owner[hit] = i + 1
        overlap |= bool((count > 1).any())
        eps_sum += eps[owner]
        inv_sum += 1 / eps[owner]
    n = subsamples**d
    return Raster(grid, eps_sum / n, inv_sum / n, overlap)


def _box_filter(idx: np.ndarray, grid: Sequence[int], subsamples: int) -> np.ndarray:
    """Fourier factor of averaging over a pixel with `subsamples` points per axis (Dirichlet kernel)."""
    f = np.ones(idx.shape[0])
    for i, n in enumerate(grid):
        x = np.pi * idx[:, i] / n
        den = subsamples * np.sin(x / subsamples)
        f *= np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(den == 0, 1.0, den))
    return f


# --- the table of Fourier coefficients eps_hat(dG) used by the dense solver ------


@dataclass
class EpsTable:
    """eps_hat at all index differences in the box [-2M_i, 2M_i]; `half` holds 2M_i."""

    values: np.ndarray  # complex, shape (4M_1+1, ..., 4M_d+1)
    half: tuple[int, ...]
    method: str  # "analytic" | "raster"


def _diff_indices(half: Sequence[int]) -> np.ndarray:
    axes = [np.arange(-h, h + 1) for h in half]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(half))


def eps_table(cell: Cell, max_index: Sequence[int], method: str = "auto", resolution: int = 64,
              subsamples: int = 4) -> EpsTable:
    """Fourier coefficients of eps for index differences up to 2 * max_index."""
    lat = cell.lattice
    half = tuple(2 * int(m) for m in max_index)
    didx = _diff_indices(half)
    shape = tuple(2 * h + 1 for h in half)
    analytic_ok = all(has_analytic_ft(i.shape) for i in cell.inclusions)
    if method == "analytic" and not analytic_ok:
        raise ValueError("eps_fourier='analytic' needs circles, ellipses, rectangles, polygons, balls, boxes "
                         "or vertical prisms")
    grid = tuple(max(resolution, 2 * h + 2) for h in half)
    grid = tuple(n + (n % 2) for n in grid)
    if method == "auto":
        if analytic_ok and cell.inclusions:
            # Overlap check on a coarse raster: the analytic sum is only exact for disjoint inclusions.
            coarse = rasterize(cell, tuple(min(n, 48 if lat.dim == 2 else 24) for n in grid), 2)
            method = "raster" if coarse.overlap else "analytic"
        else:
            method = "raster" if cell.inclusions else "analytic"
    if method == "analytic":
        q = (didx @ lat.reciprocal) / lat.a_nm  # 1/nm
        vol_nm = lat.cell_measure * lat.a_nm**lat.dim
        vals = np.where(np.all(didx == 0, axis=1), cell.eps_background, 0.0).astype(complex)
        for inc in cell.inclusions:
            vals += (inc.eps - cell.eps_background) * shape_ft(inc.shape, q) / vol_nm
        return EpsTable(vals.reshape(shape), half, "analytic")
    ras = rasterize(cell, grid, subsamples)
    return EpsTable(table_from_grid(ras.eps_mean, half, subsamples), half, "raster")


def table_from_grid(eps_grid: np.ndarray, half: Sequence[int], subsamples: int | None = None) -> np.ndarray:
    """eps_hat table from a real-space grid (NumPy); deconvolves pixel averaging when `subsamples` is given."""
    grid = eps_grid.shape
    if any(2 * h + 1 > n for h, n in zip(half, grid)):
        raise ValueError(f"grid {grid} too coarse for index differences up to {tuple(half)}")
    didx = _diff_indices(half)
    coef = np.fft.fftn(eps_grid) / eps_grid.size
    vals = coef[tuple((didx % np.asarray(grid)).T)]
    if subsamples:
        vals = vals / _box_filter(didx, grid, subsamples)
    return vals.reshape(tuple(2 * h + 1 for h in half))


# --- smoothed inverse-permittivity tensors for the iterative solver -------------


def smoothed_inverse_eps(cell: Cell, grid: Sequence[int], subsamples: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Kottke/Farjadpour subpixel smoothing on the real-space grid.

    Returns (eta, eps_mean, overlap) where eta is the (d, d, *grid) inverse
    permittivity tensor <1/eps> n n^T + <eps>^-1 (1 - n n^T) with n the local
    interface normal, and eps_mean the arithmetic pixel average (used for E_z in
    2D TM, which is tangential to every interface).
    """
    lat = cell.lattice
    d = lat.dim
    ras = rasterize(cell, grid, subsamples)
    em, im = ras.eps_mean, ras.inv_eps_mean
    inv_a = np.linalg.inv(lat.vectors)
    grad = np.zeros((d,) + em.shape)
    for i in range(d):
        dfi = (np.roll(em, -1, axis=i) - np.roll(em, 1, axis=i)) * (em.shape[i] / 2)
        grad += dfi[None] * inv_a[:, i].reshape((d,) + (1,) * d)
    gn = np.sqrt(np.sum(grad**2, axis=0))
    mixed = (np.abs(em * im - 1) > 1e-12) & (gn > 0)
    n = grad / np.where(gn > 0, gn, 1.0)
    eye = np.eye(d).reshape((d, d) + (1,) * d)
    nn = n[:, None] * n[None, :]
    eta = np.where(mixed, im * nn + (eye - nn) / em, eye / em)
    return eta, em, ras.overlap
