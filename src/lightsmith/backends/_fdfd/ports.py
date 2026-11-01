"""Ports on the 2D grid: geometry, unidirectional mode sources, modal decomposition.

A port is a line normal to x or y. Its reference plane is the grid line ``i0``
nearest to the port centre; ``d = +1/-1`` is the direction *into* the device.

Source (total-field / scattered-field): with f the incident mode on the lines
``i0`` and ``i0 + d`` (amplitude 1 at ``i0``), the source is ``b = A f``
restricted to the lines ``i0 - d`` and ``i0``. The solution is then exactly the
incident mode on the device side of ``i0`` plus scattered fields, and nothing
travels backwards from the source.

Monitor: the modal amplitude p(s) = v . field on the lines ``i0 + d`` and
``i0 + 2d`` (s = 1, 2 cells into the device) is split into
p(s) = a u^s + c u^-s with u = exp(i beta dl): a enters the device, c leaves it.
The outgoing amplitude c, referenced to the port plane, is the S-parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...problems import Port
from .grid import Grid2D
from .modes import SlabMode, solve_slab_modes


@dataclass(frozen=True)
class PortGeom:
    name: str
    axis: int  # normal axis (0 = x, 1 = y)
    d: int  # +1 / -1: direction into the device
    i0: int  # reference line index along `axis`
    lo: int  # transverse window [lo, hi)
    hi: int
    num_modes: int

    def labels(self) -> list[str]:
        return [self.name] if self.num_modes == 1 else [f"{self.name}@{m}" for m in range(self.num_modes)]

    def line(self, arr, i: int):
        """Window of the grid line `i` (along the normal) of a (Nx, Ny, ...) array."""
        return arr[i, self.lo:self.hi] if self.axis == 0 else arr[self.lo:self.hi, i]

    def full_line_index(self, i: int):
        return (i, slice(None)) if self.axis == 0 else (slice(None), i)


def port_geometry(port: Port, grid: Grid2D) -> PortGeom:
    axis = next(k for k, s in enumerate(port.size_nm) if s == 0)
    if axis == 2:
        raise ValueError(f"port {port.name!r}: in 2D a port must be normal to x or y (size_nm[0] or [1] == 0)")
    t = 1 - axis
    d = 1 if port.direction == "+" else -1
    i0 = grid.index(axis, port.center_nm[axis])
    tc = grid.axis(t)
    half = port.size_nm[t] / 2
    inside = np.nonzero(np.abs(tc - port.center_nm[t]) <= half + 1e-9 * grid.dl)[0]
    tlo, thi = grid.interior(t)
    inside = inside[(inside >= tlo) & (inside < thi)]
    if len(inside) < 3:
        raise ValueError(f"port {port.name!r}: transverse window has {len(inside)} cells outside the PML; "
                         "make the port wider or the grid finer")
    lo, hi = grid.interior(axis)
    for i in (i0 - d, i0, i0 + d, i0 + 2 * d):
        if not lo <= i < hi:
            raise ValueError(f"port {port.name!r}: needs 2 cells on each side of its plane outside the PML; "
                             "move it away from the domain edge")
    return PortGeom(name=port.name, axis=axis, d=d, i0=i0, lo=int(inside[0]), hi=int(inside[-1]) + 1,
                    num_modes=port.num_modes)


def port_modes(pg: PortGeom, eps: np.ndarray, pol: str, k0dl: float) -> list[SlabMode]:
    """Modes on the port's reference line, from the node permittivity grid (numpy)."""
    t = 1 - pg.axis
    line = pg.line(eps, pg.i0)
    n_t = eps.shape[t]
    # Permittivity on the transverse edges lo..hi (edge e sits between nodes e-1 and e).
    full = eps[pg.i0, :] if pg.axis == 0 else eps[:, pg.i0]
    padded = np.concatenate([full[:1], full, full[-1:]])
    edges = (padded[:-1] + padded[1:]) / 2  # (n_t + 1,)
    assert len(edges) == n_t + 1
    return solve_slab_modes(line, edges[pg.lo:pg.hi + 1], pol, k0dl, pg.num_modes)


def mode_source(pg: PortGeom, mode: SlabMode, diag_np, matvec) -> np.ndarray:
    """TF/SF source (Nx, Ny) launching `mode` into the device with unit amplitude at the port plane."""
    shape = np.asarray(diag_np.c0).shape
    f = np.zeros(shape, dtype=complex)
    u = np.exp(1j * mode.beta_dl)
    sl0 = pg.full_line_index(pg.i0)
    win = slice(pg.lo, pg.hi)
    if pg.axis == 0:
        f[pg.i0, win] = mode.profile
        f[pg.i0 + pg.d, win] = mode.profile * u
    else:
        f[win, pg.i0] = mode.profile
        f[win, pg.i0 + pg.d] = mode.profile * u
    af = matvec(np, diag_np, f)
    b = np.zeros(shape, dtype=complex)
    b[sl0] = af[sl0]
    sl_back = pg.full_line_index(pg.i0 - pg.d)
    b[sl_back] = af[sl_back]
    return b


def decompose(xp, pg: PortGeom, mode: SlabMode, X, projector):
    """(a, c): amplitudes entering / leaving the device at the port plane, for each RHS of X (Nx, Ny, k)."""
    p1 = xp.sum(projector[:, None] * pg.line(X, pg.i0 + pg.d), axis=0)
    p2 = xp.sum(projector[:, None] * pg.line(X, pg.i0 + 2 * pg.d), axis=0)
    u = complex(np.exp(1j * mode.beta_dl))
    det = 1 / u - u
    a = (p1 / u**2 - p2 / u) / det
    c = (u * p2 - u**2 * p1) / det
    return a, c
