"""Yee grid on a waveguide cross-section and its finite-difference operators.

Local axes are (x, y) with propagation along +z. Primal nodes sit at
``x_i = x0 + i dx`` (i = 0..nx-1), dual nodes half a cell later. Components
live where the 2D Yee cell puts them (Zhu & Brown, Opt. Express 10, 853 (2002)):

    Ex, Hy: (dual x, primal y)     Ey, Hx: (primal x, dual y)
    Ez:     (primal x, primal y)   Hz:     (dual x, dual y)

so the transverse Poynting pairs (Ex, Hy) and (Ey, Hx) are co-located and mode
overlaps need no interpolation. The wall at the max side of each axis
(``x0 + nx dx``) is PEC. The wall at the min side is PEC, or PMC, which is how
mirror symmetry planes are imposed. A PML (stretched coordinates) can line
either side. An axis with ``n == 1`` is invariant (slab problems).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Wall = Literal["pec", "pmc"]


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int
    bc_min: tuple[Wall, Wall] = ("pec", "pec")
    pml: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))  # cells at (min, max) per axis
    symmetry: tuple[bool, bool] = (False, False)  # min wall is a mirror plane (domain is half the window)
    # non-uniform axes: the n primary widths (node i to node i + 1, the last one ending on the max wall), or None
    # for uniform spacing dx / dy. Forward differences divide by the primary widths, backward ones by the dual
    # widths (p_{i-1} + p_i) / 2 (p_0 at the min wall, whose mirror image is the cell itself).
    widths: tuple = (None, None)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def invariant_y(self) -> bool:
        return self.ny == 1

    def _axis(self, axis: int):
        return (self.x0, self.dx, self.nx) if axis == 0 else (self.y0, self.dy, self.ny)

    def prim_widths(self, axis: int) -> np.ndarray:
        """Primary widths p_i (n,): the distances from primal node i to i + 1 (the last to the max wall)."""
        o, d, n = self._axis(axis)
        w = self.widths[axis]
        if w is None:
            return np.full(n, float(d))
        w = np.asarray(w, dtype=float)
        if len(w) != n:
            raise ValueError(f"grid axis {axis}: {len(w)} widths for {n} cells")
        return w

    def dual_widths(self, axis: int) -> np.ndarray:
        """Dual widths (n,) of the primal nodes: (p_{i-1} + p_i) / 2, and p_0 at the min wall."""
        p = self.prim_widths(axis)
        if self.widths[axis] is None:
            return p
        return np.concatenate([p[:1], 0.5 * (p[:-1] + p[1:])])

    def nodes(self, axis: int, dual: bool) -> np.ndarray:
        o, d, n = self._axis(axis)
        if n == 1 and axis == 1:
            return np.array([o])
        if self.widths[axis] is not None:
            p = self.prim_widths(axis)
            x = o + np.concatenate([[0.0], np.cumsum(p[:-1])])
            return x + 0.5 * p if dual else x
        return o + (np.arange(n) + (0.5 if dual else 0.0)) * d

    def weights(self, axis: int, dual: bool) -> np.ndarray:
        """Quadrature weights (times the spacing) of primal/dual nodes; mirror planes count double."""
        o, d, n = self._axis(axis)
        if n == 1 and axis == 1:
            return np.ones(1)
        w = (self.prim_widths(axis) if dual else self.dual_widths(axis)).copy()
        if not dual:
            w[0] *= 0.0 if self.bc_min[axis] == "pec" else 0.5
        if self.symmetry[axis]:
            w *= 2.0
        return w

    def pml_mask(self, axis: int, dual: bool) -> np.ndarray:
        o, d, n = self._axis(axis)
        lo, hi = self.pml[axis]
        u = self.nodes(axis, dual)
        m = np.zeros(len(u), dtype=bool)
        if lo:
            m |= u < o + lo * d
        if hi:
            m |= u > o + (n - hi) * d
        return m

    def stretch(self, axis: int, dual: bool, k0: float, n_ref: float, reflection: float) -> np.ndarray:
        """Complex coordinate stretch s = 1 + i sigma (cubic grading) for exp(-i omega t)."""
        o, d, n = self._axis(axis)
        u = self.nodes(axis, dual)
        s = np.ones(len(u), dtype=complex)
        for cells, side in zip(self.pml[axis], ("lo", "hi")):
            if not cells:
                continue
            if self.widths[axis] is not None:
                raise ValueError("a PML needs a uniform axis")
            length = cells * d
            sigma_max = 4 * math.log(1 / reflection) / (2 * k0 * n_ref * length)
            depth = (o + length - u) if side == "lo" else (u - (o + n * d - length))
            depth = np.clip(depth / length, 0, 1)
            s += 1j * sigma_max * depth**3
        return s

    def describe(self) -> dict:
        return {"shape": [self.nx, self.ny], "spacing_nm": [self.dx, self.dy if self.ny > 1 else None],
                "origin_nm": [self.x0, self.y0], "min_walls": list(self.bc_min),
                "pml_cells": [list(p) for p in self.pml],
                "symmetry": list(self.symmetry)}


def make_grid(center: tuple[float, float], size: tuple[float, float], spacing: tuple[float, float],
              symmetry: tuple[str | None, str | None] = (None, None), pml_cells: int = 0) -> Grid:
    """Grid covering the window ``center ± size/2``. A zero size makes that axis invariant (y only).

    ``symmetry[a]`` in {"pec", "pmc"} puts a mirror plane at the window centre and keeps only the
    upper half. With ``pml_cells`` > 0, PMLs line the outer walls (inside the window).
    """
    x0s, ns, ds, bcs, pmls, syms = [], [], [], [], [], []
    for a in range(2):
        c, s, d, sym = center[a], size[a], spacing[a], symmetry[a]
        if s <= 0:
            if a == 0:
                raise ValueError("the window must have a nonzero x size")
            x0s.append(c), ns.append(1), ds.append(1.0), bcs.append("pec"), pmls.append((0, 0)), syms.append(False)
            continue
        half = sym in ("pec", "pmc")
        span = s / 2 if half else s
        n = max(2, int(round(span / d)))
        x0s.append(c if half else c - s / 2)
        ns.append(n)
        ds.append(span / n)
        bcs.append(sym if half else "pec")
        p = min(pml_cells, n // 3)
        pmls.append((0 if half else p, p))
        syms.append(half)
    return Grid(x0s[0], x0s[1], ds[0], ds[1], ns[0], ns[1], tuple(bcs), tuple(pmls), tuple(syms))


def sub_grid(grid: Grid, lo: tuple[float, float], hi: tuple[float, float]) -> tuple[Grid, tuple[slice, slice]]:
    """A PEC-walled window of `grid` snapped to its primal nodes, and the index slices it occupies."""
    out, sl = [], []
    for a in range(2):
        o, d, n = grid._axis(a)
        if n == 1 and a == 1:
            out.append((o, d, 1))
            sl.append(slice(0, 1))
            continue
        i0 = int(np.clip(round((lo[a] - o) / d), 0, n - 1))
        i1 = int(np.clip(round((hi[a] - o) / d), i0 + 2, n))
        out.append((o + i0 * d, d, i1 - i0))
        sl.append(slice(i0, i1))
    (x0, dx, nx), (y0, dy, ny) = out
    return Grid(x0, y0, dx, dy, nx, ny), (sl[0], sl[1])
