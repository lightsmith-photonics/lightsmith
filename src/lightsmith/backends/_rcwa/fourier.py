"""Fourier coefficients of patterned layers and normal-vector fields for RCWA.

A unit cell is r = u a1 + v a2 with u, v in [-1/2, 1/2) (lattice coordinates).
For 1D lattices the cell is the segment s in [-|a1|/2, |a1|/2) along a1 through
the origin: shapes are sampled along that line (a ridge is, e.g., a Rectangle
whose extent across a1 is anything non-zero).

Every shape s gets a *form factor* Phi_s(m, n): the Fourier coefficient of its
(painter's-order resolved) fill fraction at G = m b1 + n b2. Permittivity and
inverse-permittivity coefficients for any wavelength are then linear
combinations, so geometry is processed once per solve:

- 1D: exact, from the boundaries of every segment (found by bisection to ~1e-12 |a1|).
- 2D "analytic": closed-form transforms of rectangles, circles, ellipses and
  polygons (valid when shapes and their periodic images do not overlap).
- 2D "raster": supersampled fill fractions on a lattice-aligned grid
  (``raster.inside_profile``), transformed exactly as piecewise-constant pixels.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...problems import Circle, Ellipse, Polygon, Rectangle
from ...raster import inside_profile


class Lattice:
    """Direct and reciprocal lattice. a2 is None for 1D gratings (periodic along a1)."""

    def __init__(self, a1, a2=None):
        self.a1 = np.asarray(a1, dtype=float)
        self.a2 = None if a2 is None else np.asarray(a2, dtype=float)
        if self.a2 is None:
            p2 = self.a1 @ self.a1
            if p2 == 0:
                raise ValueError("lattice_a1_nm must be non-zero")
            self.b1, self.b2 = 2 * np.pi * self.a1 / p2, np.zeros(2)
            self.measure = float(np.sqrt(p2))  # period
        else:
            a = np.stack([self.a1, self.a2], axis=1)
            det = np.linalg.det(a)
            if abs(det) < 1e-12 * (self.a1 @ self.a1):
                raise ValueError("lattice vectors are parallel")
            b = 2 * np.pi * np.linalg.inv(a).T
            self.b1, self.b2 = b[:, 0], b[:, 1]
            self.measure = float(abs(det))  # cell area

    @property
    def dim(self) -> int:
        return 1 if self.a2 is None else 2

    def g(self, m, n):
        """Cartesian reciprocal vectors (gx, gy) for integer arrays m, n."""
        m, n = np.asarray(m, float), np.asarray(n, float)
        return m * self.b1[0] + n * self.b2[0], m * self.b1[1] + n * self.b2[1]

    def images(self) -> list[np.ndarray]:
        r = (-1, 0, 1)
        if self.a2 is None:
            return [k * self.a1 for k in r]
        return [i * self.a1 + j * self.a2 for i in r for j in r]


def harmonic_set(dim: int, m_max: int, n_max: int) -> np.ndarray:
    """Integer orders (m, n), shape (N, 2); rectangular truncation |m| <= M, |n| <= N."""
    n_max = 0 if dim == 1 else n_max
    mm, nn = np.meshgrid(np.arange(-m_max, m_max + 1), np.arange(-n_max, n_max + 1), indexing="ij")
    return np.stack([mm.ravel(), nn.ravel()], axis=1)


def toeplitz_index(harm: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """Flat index into a (2k1+1, 2k2+1) coefficient array for entry (i, j) = c[h_i - h_j]."""
    d = harm[:, None, :] - harm[None, :, :]
    return (d[..., 0] + k1) * (2 * k2 + 1) + (d[..., 1] + k2)


# --- closed-form shape transforms -----------------------------------------------


def _sinc(t):
    return np.sinc(t / np.pi)


def bessel_j1(x) -> np.ndarray:
    """J1(x) from its periodic integral representation (trapezoid rule: spectrally accurate)."""
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    n = int(max(64, 1.3 * (np.abs(flat).max() if flat.size else 0) + 64))
    tau = 2 * np.pi * np.arange(n) / n
    s, c = np.sin(tau), tau
    out = np.empty(flat.shape)
    chunk = max(1, 2_000_000 // n)
    for i in range(0, flat.size, chunk):
        out[i:i + chunk] = np.cos(c[None, :] - flat[i:i + chunk, None] * s[None, :]).mean(axis=1)
    return out.reshape(x.shape)


def _airy(q):
    """2 J1(q) / q, with the q -> 0 limit."""
    small = np.abs(q) < 1e-8
    qs = np.where(small, 1.0, q)
    return np.where(small, 1.0, 2 * bessel_j1(qs) / qs)


def shape_transform(profile, gx, gy) -> np.ndarray:
    """Integral of exp(-i g.r) over the profile's area (nm^2)."""
    gx, gy = np.broadcast_arrays(np.asarray(gx, float), np.asarray(gy, float))
    if isinstance(profile, Polygon):
        return _polygon_transform(np.asarray(profile.vertices_nm, float), gx, gy)
    cx, cy = profile.center_nm
    phase = np.exp(-1j * (gx * cx + gy * cy))
    if isinstance(profile, Rectangle):
        w, h = profile.size_nm
        return w * h * _sinc(gx * w / 2) * _sinc(gy * h / 2) * phase
    if isinstance(profile, Circle):
        r = profile.radius_nm
        return np.pi * r * r * _airy(np.hypot(gx, gy) * r) * phase
    if isinstance(profile, Ellipse):
        a, b = profile.semi_axes_nm
        t = np.radians(profile.angle_deg)
        gu, gv = gx * np.cos(t) + gy * np.sin(t), -gx * np.sin(t) + gy * np.cos(t)
        return np.pi * a * b * _airy(np.hypot(a * gu, b * gv)) * phase
    raise TypeError(f"no analytic transform for {type(profile).__name__}")


def _polygon_transform(v: np.ndarray, gx, gy) -> np.ndarray:
    x, y = v[:, 0], v[:, 1]
    area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area2 < 0:  # make counter-clockwise
        v = v[::-1]
    p0, p1 = v, np.roll(v, -1, axis=0)
    e, c = p1 - p0, (p0 + p1) / 2
    g2 = gx**2 + gy**2
    zero = g2 == 0
    g2s = np.where(zero, 1.0, g2)
    total = np.zeros(gx.shape, dtype=complex)
    for k in range(len(v)):
        gn = gx * e[k, 1] - gy * e[k, 0]  # g . (outward normal * edge length)
        ge = gx * e[k, 0] + gy * e[k, 1]
        total += gn * np.exp(-1j * (gx * c[k, 0] + gy * c[k, 1])) * _sinc(ge / 2)
    return np.where(zero, abs(area2) / 2, 1j * total / g2s)


# --- form factors ------------------------------------------------------------------


def _owner(profiles: Sequence, lat: Lattice, x, y) -> np.ndarray:
    """Index of the shape covering each point (painter's order, periodic images), -1 = background."""
    own = np.full(np.shape(x), -1)
    for i, p in enumerate(profiles):
        inside = np.zeros(np.shape(x), dtype=bool)
        for s in lat.images():
            inside |= inside_profile(p, x - s[0], y - s[1])
        own[inside] = i
    return own


def _form_factors_1d(profiles, lat: Lattice, k: int, samples: int = 8192) -> np.ndarray:
    p = lat.measure
    uh = lat.a1 / p
    pts = lambda s: (s * uh[0], s * uh[1])  # noqa: E731
    owner = lambda s: _owner(profiles, lat, *pts(s))  # noqa: E731
    s = -p / 2 + (np.arange(samples) + 0.5) * p / samples
    own = owner(s)
    nxt = np.roll(own, -1)
    idx = np.nonzero(own != nxt)[0]
    out = np.zeros((len(profiles), 2 * k + 1, 1), dtype=complex)
    if idx.size == 0:
        if own[0] >= 0:
            out[own[0], k, 0] = 1.0
        return out
    lo = s[idx]
    hi = lo + p / samples
    left = own[idx]
    for _ in range(48):  # bisection on each boundary
        mid = (lo + hi) / 2
        same = owner(mid) == left
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    b = np.sort((lo + hi) / 2)
    starts, ends = b, np.append(b[1:], b[0] + p)
    seg_owner = owner(((starts + ends) / 2 + p / 2) % p - p / 2)
    m = np.arange(-k, k + 1)
    gm = 2 * np.pi * m / p
    for s0, s1, o in zip(starts, ends, seg_owner):
        if o < 0:
            continue
        g = np.where(m == 0, 1.0, gm)
        val = (np.exp(-1j * g * s0) - np.exp(-1j * g * s1)) / (1j * g * p)
        out[o, :, 0] += np.where(m == 0, (s1 - s0) / p, val)
    return out


def _overlapping(profiles, lat: Lattice, n: int = 96) -> bool:
    u = (np.arange(n) + 0.5) / n - 0.5
    uu, vv = np.meshgrid(u, u, indexing="ij")
    x, y = uu * lat.a1[0] + vv * lat.a2[0], uu * lat.a1[1] + vv * lat.a2[1]
    count = np.zeros(x.shape, dtype=int)
    for p in profiles:
        for s in lat.images():
            count += inside_profile(p, x - s[0], y - s[1])
    return bool(count.max(initial=0) > 1)


def raster_fill(profiles, lat: Lattice, n: int, subsamples: int = 4) -> np.ndarray:
    """Fill fraction of each shape in each pixel of an n x n lattice-aligned grid, (S, n, n)."""
    u = (np.arange(n) + 0.5) / n - 0.5
    frac = np.zeros((len(profiles), n, n))
    off = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) / n
    for du in off:
        for dv in off:
            uu, vv = np.meshgrid(u + du, u + dv, indexing="ij")
            x, y = uu * lat.a1[0] + vv * lat.a2[0], uu * lat.a1[1] + vv * lat.a2[1]
            own = _owner(profiles, lat, x, y)
            for i in range(len(profiles)):
                frac[i] += own == i
    return frac / subsamples**2


def pixel_coefficients(xp, grid, k1: int, k2: int, device=None):
    """Exact Fourier coefficients of a piecewise-constant pixel map on the unit cell.

    grid[..., i, j] is the value on pixel centred at u = (i + 1/2)/Nu - 1/2,
    v = (j + 1/2)/Nv - 1/2. Returns (..., 2k1+1, 2k2+1). Works on any array API
    namespace (differentiable with torch).
    """
    nu, nv = grid.shape[-2], grid.shape[-1]
    d = xp.fft.fft2(grid) if nv > 1 else xp.fft.fft(grid, axis=-2)
    m, n = np.arange(-k1, k1 + 1), np.arange(-k2, k2 + 1)

    def factor(idx, nn):  # pixel shape factor and shift of the cell origin to u = -1/2
        return np.exp(1j * np.pi * idx) * np.exp(-1j * np.pi * idx / nn) * np.sinc(idx / nn) / nn

    d = xp.take(d, xp.asarray(m % nu, device=device), axis=-2)
    d = xp.take(d, xp.asarray(n % nv, device=device), axis=-1)
    f = np.outer(factor(m, nu), factor(n, nv))
    return d * xp.asarray(f, dtype=d.dtype, device=device)


def form_factors(profiles: Sequence, lat: Lattice, k1: int, k2: int, method: str = "auto",
                 raster_points: int = 256, subsamples: int = 4) -> tuple[np.ndarray, str]:
    """Phi[s, m + k1, n + k2] for every shape; returns (Phi, method actually used)."""
    if not profiles:
        return np.zeros((0, 2 * k1 + 1, 2 * (0 if lat.dim == 1 else k2) + 1), dtype=complex), "none"
    if lat.dim == 1:
        return _form_factors_1d(profiles, lat, k1), "analytic-1d"
    if method == "analytic" or (method == "auto" and not _overlapping(profiles, lat)):
        m, n = np.meshgrid(np.arange(-k1, k1 + 1), np.arange(-k2, k2 + 1), indexing="ij")
        gx, gy = lat.g(m, n)
        return np.stack([shape_transform(p, gx, gy) for p in profiles]) / lat.measure, "analytic"
    npix = max(raster_points, 2 * max(k1, k2) + 2)
    frac = raster_fill(profiles, lat, npix, subsamples)
    return pixel_coefficients(np, frac.astype(complex), k1, k2), "raster"


def normal_field_coefficients(label_coef: np.ndarray, lat: Lattice, k1: int, k2: int,
                              blur: float = 0.1) -> np.ndarray:
    """Fourier coefficients of n n^T for a smooth unit normal field, (3, 2k1+1, 2k2+1) = [xx, xy, yy].

    label_coef: Fourier coefficients (2L1+1, 2L2+1) of a field that changes value across
    every material boundary. Its Gaussian-blurred gradient g gives the structure tensor
    J = G * (g g^H) (smoothed again); n is the principal direction of J. Unlike g/|g| this
    stays continuous across ridge/groove centres (for 1D-like cells it reduces exactly to
    the grating normal, i.e. Li's rule) and is radial around circles. Normal-vector
    factorization after Schuster et al., JOSA A 24, 2880 (2007); Goetz et al., Opt.
    Express 16, 17295 (2008). Isotropic points get n n^T = I/2.
    """
    l1, l2 = (label_coef.shape[0] - 1) // 2, (label_coef.shape[1] - 1) // 2
    n1, n2 = 2 * l1, 2 * l2
    m, n = np.meshgrid(np.arange(-l1, l1 + 1), np.arange(-l2, l2 + 1), indexing="ij")
    gx, gy = lat.g(m, n)
    sigma = blur * min(np.linalg.norm(lat.a1), np.linalg.norm(lat.a2))
    kernel = np.exp(-0.5 * sigma**2 * (gx**2 + gy**2))
    pos = (m % n1, n % n2)

    def to_grid(coef):
        buf = np.zeros((n1, n2), dtype=complex)
        np.add.at(buf, pos, coef)
        return np.fft.ifft2(buf) * n1 * n2

    def smooth(field):
        c = np.fft.fft2(field) / (n1 * n2)
        return to_grid(c[pos] * kernel)

    f = label_coef * kernel
    ex, ey = to_grid(1j * gx * f), to_grid(1j * gy * f)
    jxx = np.real(smooth(np.abs(ex) ** 2))
    jyy = np.real(smooth(np.abs(ey) ** 2))
    jxy = np.real(smooth(np.real(ex * np.conj(ey))))
    c, s = jxx - jyy, 2 * jxy
    norm = np.hypot(c, s)
    floor = 1e-12 * norm.max() if norm.max() > 0 else 1.0
    c, s = c / np.maximum(norm, floor), s / np.maximum(norm, floor)
    tensors = [(1 + c) / 2, s / 2, (1 - c) / 2]
    mm, nn = np.arange(-k1, k1 + 1), np.arange(-k2, k2 + 1)
    out = []
    for t in tensors:
        coef = np.fft.fft2(t) / (n1 * n2)
        out.append(coef[np.ix_(mm % n1, nn % n2)])
    return np.stack(out)
