"""Ports on the 3D grid: geometry, total-field/scattered-field mode sources, modal decomposition.

A port is a rectangle normal to one axis ``a``; its reference plane is the grid
plane ``i0`` nearest to the port centre and ``d = +1/-1`` points into the device.
The transverse window is the port rectangle minus the PML.

Source: with ``f`` the incident mode (exact discrete mode, see ``modes``) written on
a few planes around ``i0`` and ``chi`` the indicator of the total-field side
(positions ``p`` with ``(p - i0) d >= 0``; normal components sit at half-integer
positions), the source is ``b = A (chi f) - chi (A f)`` on the planes ``i0 - 2 .. i0 + 2``.
The solution is the incident mode on the device side plus scattered fields, and
nothing travels backwards from the source (as in the 2D solver).

Monitor: the modal amplitude p(s) of the transverse E on the planes ``i0 + d`` and
``i0 + 2d`` is split into ``a u^s + c u^-s`` (a enters the device, c leaves it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...problems import Port
from .grid import Grid3D
from .modes import VectorMode, plane, plane_axes, solve_vector_modes


@dataclass(frozen=True)
class PortGeom3D:
    name: str
    axis: int
    d: int
    i0: int
    window: tuple[int, int, int, int]  # (lo1, hi1, lo2, hi2) along (t1, t2)
    num_modes: int

    def labels(self) -> list[str]:
        return [self.name] if self.num_modes == 1 else [f"{self.name}@{m}" for m in range(self.num_modes)]

    def index(self, i: int) -> tuple:
        """Index tuple into (k, 3, Nx, Ny, Nz) selecting plane i (all components, full plane)."""
        t = [slice(None)] * 3
        t[self.axis] = i
        return (slice(None), slice(None), *t)

    def window_slices(self) -> dict[int, slice]:
        t1, t2 = plane_axes(self.axis)
        lo1, hi1, lo2, hi2 = self.window
        return {t1: slice(lo1, hi1), t2: slice(lo2, hi2)}

    def tangential(self, X, i: int):
        """(k, 2, w1, w2) transverse E (t1, t2 components) on plane i inside the window."""
        t1, t2 = plane_axes(self.axis)
        ws = self.window_slices()
        idx = [None] * 3
        idx[self.axis] = i
        idx[t1], idx[t2] = ws[t1], ws[t2]
        out = [X[(slice(None), c, *idx)] for c in (t1, t2)]
        if t1 > t2:  # remaining axes come out in increasing order; reorder to (t1, t2)
            out = [o.mT for o in out]
        return out


def port_geometry(port: Port, grid: Grid3D) -> PortGeom3D:
    axis = next(k for k, s in enumerate(port.size_nm) if s == 0)
    d = 1 if port.direction == "+" else -1
    i0 = grid.index(axis, port.center_nm[axis])
    win = []
    for t in plane_axes(axis):
        tc = grid.axis(t)
        inside = np.nonzero(np.abs(tc - port.center_nm[t]) <= port.size_nm[t] / 2 + 1e-9 * grid.dl)[0]
        tlo, thi = grid.interior(t)
        inside = inside[(inside >= tlo) & (inside < thi)]
        if len(inside) < 3:
            raise ValueError(f"port {port.name!r}: transverse window has {len(inside)} cells along "
                             f"{'xyz'[t]} outside the PML; make the port larger or the grid finer")
        win += [int(inside[0]), int(inside[-1]) + 1]
    lo, hi = grid.interior(axis)
    for i in (i0 - 3, i0 + 3, i0 + 2 * d):
        if not lo <= i < hi:
            raise ValueError(f"port {port.name!r}: needs 3 cells on each side of its plane outside the PML; "
                             "move it away from the domain edge")
    return PortGeom3D(name=port.name, axis=axis, d=d, i0=i0, window=tuple(win), num_modes=port.num_modes)


def port_modes(pg: PortGeom3D, eps_c: tuple[np.ndarray, ...], k0dl: float, stretch) -> list[VectorMode]:
    """Modes on the port plane from the component permittivities (each (Nx, Ny, Nz), numpy)."""
    a = pg.axis
    t1, t2 = plane_axes(a)
    en, e1, e2 = (plane(eps_c[c], a, pg.i0) for c in (a, t1, t2))
    st = ((stretch[0][t1], stretch[1][t1]), (stretch[0][t2], stretch[1][t2]))
    return solve_vector_modes(en, e1, e2, pg.window, k0dl, pg.num_modes, st)


def incident_field(pg: PortGeom3D, mode: VectorMode, shape, planes: range) -> tuple[np.ndarray, np.ndarray]:
    """(f, chi): the incident mode on `planes` (shape (1, 3, Nx, Ny, Nz)) and the total-field indicator."""
    a = pg.axis
    t1, t2 = plane_axes(a)
    f = np.zeros((1, 3) + tuple(shape), complex)
    chi = np.zeros((1, 3) + tuple(shape))
    u = np.exp(1j * mode.beta_dl)
    lo1, hi1, lo2, hi2 = pg.window

    def put(comp, i, vals2d, pos):
        idx = [slice(None)] * 3
        idx[a] = i
        idx[t1], idx[t2] = slice(lo1, hi1), slice(lo2, hi2)
        v = vals2d if t1 < t2 else vals2d.T
        f[(0, comp, *idx)] = v * u ** ((pos - pg.i0) * pg.d)
        full = [slice(None)] * 3
        full[a] = i
        chi[(0, comp, *full)] = 1.0 if (pos - pg.i0) * pg.d >= 0 else 0.0

    for i in planes:
        put(t1, i, mode.e1, i)
        put(t2, i, mode.e2, i)
        put(a, i, pg.d * mode.kappa * mode.q, i + 0.5)
    return f, chi


def mode_source(pg: PortGeom3D, mode: VectorMode, apply, shape) -> np.ndarray:
    """TF/SF source (1, 3, Nx, Ny, Nz) launching `mode` with unit amplitude at the port plane.

    apply: X -> A X for numpy arrays of shape (k, 3, Nx, Ny, Nz); shape: (Nx, Ny, Nz).
    """
    f, chi = incident_field(pg, mode, shape, range(pg.i0 - 3, pg.i0 + 4))
    b = apply(chi * f) - chi * apply(f)
    keep = np.zeros_like(chi)
    for i in range(pg.i0 - 2, pg.i0 + 3):
        keep[pg.index(i)] = 1
    return b * keep


def decompose(xp, pg: PortGeom3D, mode: VectorMode, X, proj):
    """(a, c) per right-hand side: amplitudes entering / leaving the device at the port plane.

    proj: (2, w1, w2) array (namespace xp) of the mode's projector.
    """
    k = X.shape[0]

    def amp(i):
        e1, e2 = pg.tangential(X, i)
        return xp.sum(xp.reshape(proj[0] * e1 + proj[1] * e2, (k, -1)), axis=1)

    p1, p2 = amp(pg.i0 + pg.d), amp(pg.i0 + 2 * pg.d)
    u = complex(np.exp(1j * mode.beta_dl))
    det = 1 / u - u
    a = (p1 / u**2 - p2 / u) / det
    c = (u * p2 - u**2 * p1) / det
    return a, c
