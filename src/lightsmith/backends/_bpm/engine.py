"""Beam propagation through a sliced device: port launch, Padé/Crank-Nicolson steps, port projection.

The device between its two port planes is cut into piecewise-uniform slices
(the EME slicer, with finer staircasing of tapers). In each slice the
transverse operator M (see :mod:`.operator`) is fixed; the reference index is
updated at every slice from the field's Rayleigh quotient,
n_ref^2 = Re<phi, M phi> / <phi, phi> (adaptive reference), and the
Crank-Nicolson factors are LU'd once per slice, so a long uniform section
costs one factorisation plus cheap solves. The physical field is
phi exp(i theta) with theta = sum k0 n_ref dw, accumulated exactly.

The propagated field is the transverse H (see :mod:`.operator`), so slice
boundaries keep H continuous, and each input's power <E, H> (E ~ Q H / n) is
renormalised there to stay continuous: plain field continuity conserves an
n-weighted norm instead, an error of first order in the change of effective
index that accumulates along a taper (cf. the energy-conserving parabolic
equation, Collins & Westwood, J. Acoust. Soc. Am. 89, 1068 (1991)).

Port modes come from the FDE engine on each port window, exactly as in EME;
the field launched from port q is H_t of mode q, and the amplitude of port
mode p at the far plane is <E_p, H> / <E_p, H_p> over the propagated
components (unconjugated: picks one mode of a lossy or lossless guide).
Propagation is one-way: inputs at the min plane give S(max <- min), and a
second run from the max plane gives S(min <- max). Reflections are not
modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .._fde import Grid
from .._fde.modes import bend_transform
from .operator import Operator
from .pade import cn_factors
from .tridiag import BlockTridiag, block_matvec


@dataclass
class Settings:
    comps: tuple[int, ...]
    step_nm: float
    pade_order: int
    pade_rotation_deg: float
    reference_index: float | None  # None = adaptive
    power_continuity: bool
    bend_radius_nm: float | None
    bend_center_nm: float
    subpixel: str
    pml_reflection: float


@dataclass
class RunStats:
    steps: int = 0
    factorizations: int = 0
    n_ref: list = field(default_factory=list)
    peak_factor_bytes: int = 0
    power_rescale: list = field(default_factory=list)


def cell_eps(cell, grid: Grid, wavelength_nm: float, s: Settings):
    raw = cell.owners.eps(cell.section.material_eps(wavelength_nm), s.subpixel)
    n_pml = float(np.sqrt(np.abs(raw[2][-1, -1])))
    if s.bend_radius_nm is None:
        return raw, None, n_pml
    eps, mu = bend_transform(grid, raw, s.bend_radius_nm, s.bend_center_nm)
    return eps, mu, n_pml


def propagate(cells, grid: Grid, wavelength_nm: float, phi, s: Settings, xp, asarray, stats: RunStats,
              n_bounds: tuple[float, float]):
    """Propagate line-layout fields `phi` (n, nb, k) through `cells` in order. Returns (phi, theta)."""
    k0 = 2 * math.pi / wavelength_nm
    theta = 0.0
    n_ref = s.reference_index
    prev = None  # operator of the previous slice
    for cell in cells:
        eps, mu, n_pml = cell_eps(cell, grid, wavelength_nm, s)
        op = Operator(grid, eps, mu, wavelength_nm, s.comps, xp, asarray, n_pml, s.pml_reflection)
        A, B, C = op.blocks()
        f = op.layout.from_lines(xp, phi)
        num, den, flux = op.rayleigh(f)
        n_col = _index(xp, num, den, n_bounds)
        if prev is not None and s.power_continuity:
            # Keep each column's power <E, H> = flux / n continuous across the slice boundary.
            pnum, pden, pflux = prev.rayleigh(f)
            ratio = (pflux / _index(xp, pnum, pden, n_bounds)) / (flux / n_col)
            ratio = xp.where(ratio > 0, ratio, xp.ones_like(ratio))
            phi = phi * xp.astype(xp.sqrt(ratio), phi.dtype)
            stats.power_rescale.append(float(xp.max(xp.abs(ratio - 1))))
        prev = op
        if s.reference_index is None:
            w = xp.abs(flux)
            n_ref = float(xp.sqrt(xp.sum(w * n_col**2) / xp.sum(w))) if float(xp.sum(w)) > 0 else n_bounds[1]
        stats.n_ref.append(n_ref)
        nsteps = max(1, math.ceil(cell.length / s.step_nm - 1e-9))
        dw = cell.length / nsteps
        gain, pairs = cn_factors(s.pade_order, k0 * n_ref * dw, s.pade_rotation_deg)
        n2 = n_ref**2
        eye = asarray(np.eye(op.layout.nb))
        facs = []
        for a, b in pairs:
            fb = None
            if b is not None:
                fb = BlockTridiag(xp, A * (-1 / (b * n2)), eye * (1 + 1 / b) - B * (1 / (b * n2)), C * (-1 / (b * n2)))
                stats.factorizations += 1
            facs.append((a, fb))
        stats.peak_factor_bytes = max(stats.peak_factor_bytes, sum(f.nbytes() for _, f in facs if f is not None))
        for _ in range(nsteps):
            phi = phi * gain
            for a, fb in facs:
                if a is not None:
                    phi = phi * (1 + 1 / a) - block_matvec(xp, A, B, C, phi) * (1 / (a * n2))
                if fb is not None:
                    phi = fb.solve(phi)
            theta += k0 * n_ref * dw
        stats.steps += nsteps
        del facs
    return phi, theta


def _index(xp, num, den, n_bounds):
    """sqrt(num / den) per column, clipped to `n_bounds` (the upper bound where undefined)."""
    ok = (den > 0) & (num > 0)
    n2 = xp.where(ok, num / xp.where(ok, den, xp.ones_like(den)), xp.ones_like(den) * n_bounds[1] ** 2)
    return xp.clip(xp.sqrt(n2), n_bounds[0], n_bounds[1])


def port_fields(ms, comps: tuple[int, ...], shape: tuple[int, int]):
    """Launch fields (M, C, nx, ny) (H of each mode) and measuring weights (M, C, nx, ny) of a port ModeSet.

    Measuring weights w_p satisfy sum(w_p * H_q) = <E_p, H_q> / <E_p, H_p> over the kept H components,
    so sum(w_p * launch_p) = 1 and, for full-vector fields, the amplitude of mode p is <E_p, H>.
    """
    h = {0: ms.hx, 1: ms.hy}
    w = {0: -ms.wb[None, :] * ms.ey, 1: ms.wa[None, :] * ms.ex}
    launch = np.stack([h[c] for c in comps], axis=1)
    meas = np.stack([w[c] for c in comps], axis=1)
    norm = np.sum(launch * meas, axis=2).sum(axis=1)
    meas = meas / norm[:, None, None]
    m = launch.shape[0]
    return launch.reshape(m, len(comps), *shape), meas.reshape(m, len(comps), *shape), norm
