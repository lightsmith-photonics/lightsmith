"""Eigenmode expansion (EME) engine.

The device between the two port planes is cut into cells along the
propagation axis. Inside each cell the cross-section is uniform, so the
field is a sum of the cell's eigenmodes (from the FDE engine), each picking
up exp(i k0 n L). At the interface between cells, continuity of tangential
E (tested with the right cell's H) and of tangential H (tested with the left
cell's E) gives, with O_ik = <E_i^L, h_k^R> and <E, H> = ∫ (E x H)·z dA,

    R_L = (I + O O^T)^-1 (I - O O^T),     T_LR = O^T (I + R_L),
    R_R = -(I + O^T O)^-1 (I - O^T O),    T_RL = O (I - R_R) = T_LR^T,

with modes normalised to <E_i, H_i> = 1 (unconjugated, so that lossy and
evanescent modes stay biorthogonal). Cells and interfaces are cascaded with
Redheffer star products (stable: only decaying exponentials appear). Port
modes, solved in each port's window at the port plane, couple to the end
cells' modes through the symmetrised overlap ½(<E_p, h_k> + <e_k, H_p>),
which is the same for launching and for measuring.

References: Bienstman & Baets, Opt. Quantum Electron. 33, 327 (2001);
Gallagher & Felici, Proc. SPIE 4987, 69 (2003).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .._fde import CrossSection, Grid, Mode, ModeSolver, OwnerMaps, plane_axes, sub_grid

# --- S-matrix algebra ----------------------------------------------------------------


@dataclass
class SMatrix:
    """Two-sided scattering matrix in modal amplitudes: S11/S22 reflections, S21 (1->2) and S12 transmissions."""

    s11: np.ndarray
    s12: np.ndarray
    s21: np.ndarray
    s22: np.ndarray

    @classmethod
    def propagation(cls, neff: np.ndarray, length_nm: float, wavelength_nm: float) -> "SMatrix":
        k0 = 2 * math.pi / wavelength_nm
        p = np.diag(np.exp(1j * k0 * neff * length_nm))
        z = np.zeros_like(p)
        return cls(z, p, p.copy(), z.copy())

    def star(self, b: "SMatrix") -> "SMatrix":
        """Redheffer product: self (sides 1|2) followed by b (sides 2|3)."""
        a = self
        i1 = np.eye(a.s22.shape[0])
        i2 = np.eye(b.s11.shape[0])
        x = np.linalg.solve(i2 - b.s11 @ a.s22, np.hstack([b.s11 @ a.s21, b.s12]))  # (I - B11 A22)^-1 [...]
        y = np.linalg.solve(i1 - a.s22 @ b.s11, np.hstack([a.s21, a.s22 @ b.s12]))  # (I - A22 B11)^-1 [...]
        n1 = a.s21.shape[1]
        return SMatrix(s11=a.s11 + a.s12 @ x[:, :n1], s12=a.s12 @ x[:, n1:],
                       s21=b.s21 @ y[:, :n1], s22=b.s22 + b.s21 @ y[:, n1:])


@dataclass
class ModeSet:
    """Modes of one cross-section, stacked for fast overlaps: rows are modes."""

    neff: np.ndarray
    ex: np.ndarray  # (M, N) weighted where needed
    ey: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    wa: np.ndarray  # weights at Ex/Hy, (N,)
    wb: np.ndarray  # weights at Ey/Hx
    modes: list = field(default_factory=list, repr=False)

    @classmethod
    def from_modes(cls, modes: list[Mode], grid: Grid, window: tuple[slice, slice] | None = None) -> "ModeSet":
        """Stack modes; modes on a sub-window of `grid` are zero-padded into it."""
        def full(a):
            if window is None:
                return a.ravel()
            out = np.zeros(grid.shape, dtype=complex)
            out[window] = a
            return out.ravel()

        ref = Mode(0j, 0.0, grid, None, None, None)
        wa, wb, _, _ = ref.weights()
        return cls(neff=np.array([m.neff for m in modes]),
                   ex=np.array([full(m.E[0]) for m in modes]), ey=np.array([full(m.E[1]) for m in modes]),
                   hx=np.array([full(m.H[0]) for m in modes]), hy=np.array([full(m.H[1]) for m in modes]),
                   wa=wa.ravel(), wb=wb.ravel(), modes=modes)

    def truncate(self, m: int) -> "ModeSet":
        return ModeSet(self.neff[:m], self.ex[:m], self.ey[:m], self.hx[:m], self.hy[:m], self.wa, self.wb,
                       self.modes[:m])

    def cross(self, other: "ModeSet") -> np.ndarray:
        """O[i, j] = <E_i(self), H_j(other)>."""
        return (self.ex * self.wa) @ other.hy.T - (self.ey * self.wb) @ other.hx.T

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.ex, self.ey, self.hx, self.hy))


def interface(left: ModeSet, right: ModeSet) -> SMatrix:
    """Mode matching with E continuity tested by the right H and H continuity by the left E.

    With O = <E^L, h^R> only: R_L = (I + O O^T)^-1 (I - O O^T), T_LR = O^T (I + R_L),
    R_R = -(I + O^T O)^-1 (I - O^T O), T_RL = O (I - R_R). Reciprocal (T_RL = T_LR^T) for any
    truncation, and passive in the single-mode limit (T = 2o/(1+o^2) <= 1), unlike projecting both
    conditions on one side, which gives T = 1/o.
    """
    O = left.cross(right)  # O[i, k] = <E_i^L, h_k^R>
    il, ir = np.eye(O.shape[0]), np.eye(O.shape[1])
    g = O @ O.T
    r_l = np.linalg.solve(il + g, il - g)
    t_lr = O.T @ (il + r_l)
    gp = O.T @ O
    r_r = -np.linalg.solve(ir + gp, ir - gp)
    t_rl = O @ (ir - r_r)
    return SMatrix(r_l, t_rl, t_lr, r_r)


def coupling(ports: ModeSet, cells: ModeSet) -> np.ndarray:
    """C[p, k] = ½(<E_p, h_k> + <e_k, H_p>): port mode p in terms of cell mode k (either direction)."""
    return 0.5 * (ports.cross(cells) + cells.cross(ports).T)


# --- slicing along the propagation axis -------------------------------------------------


def _extent_along(geometry, axis: int) -> list[float]:
    """Coordinates along `axis` where the cross-section of `geometry` may change."""
    kind = geometry.kind
    if kind == "box":
        c, s = geometry.center_nm[axis], geometry.size_nm[axis]
        return [c - s / 2, c + s / 2]
    if kind == "ball":
        c, r = geometry.center_nm[axis], geometry.radius_nm
        return [c - r, c + r]
    if kind == "prism":
        if axis == 2:
            return [geometry.z_min_nm, geometry.z_max_nm]
        p = geometry.profile
        if p.kind == "rectangle":
            return [p.center_nm[axis] - p.size_nm[axis] / 2, p.center_nm[axis] + p.size_nm[axis] / 2]
        if p.kind == "circle":
            return [p.center_nm[axis] - p.radius_nm, p.center_nm[axis] + p.radius_nm]
        if p.kind == "ellipse":
            r = max(p.semi_axes_nm)
            return [p.center_nm[axis] - r, p.center_nm[axis] + r]
        return [v[axis] for v in p.vertices_nm]
    return []


def _auto_cells(a: OwnerMaps, b: OwnerMaps, grid: Grid, cells_per_step: float = 3.0,
                lo: int = 10, hi: int = 60) -> int:
    """Cells for a varying segment from how far its outline moves: about `cells_per_step` grid cells per step.

    `a` and `b` are the cross-sections at 1/4 and 3/4 of the segment (half its length apart).
    """
    da, db = a.comps[2].base, b.comps[2].base
    diff = da != db
    shift = diff.sum(axis=0).max() * grid.dx
    if not grid.invariant_y:
        shift = max(shift, diff.sum(axis=1).max() * grid.dy)
    d = grid.dx if grid.invariant_y else min(grid.dx, grid.dy)
    return int(np.clip(math.ceil(2 * shift / (cells_per_step * d)), lo, hi))


@dataclass
class Cell:
    start: float
    stop: float
    section: CrossSection
    owners: OwnerMaps
    key: str

    @property
    def length(self) -> float:
        return self.stop - self.start


def slice_device(structures, background, axis: int, lo: float, hi: float, grid: Grid, dimension: int,
                 plane_z_nm: float, taper_cells: int, max_cell_length_nm: float | None, subsamples: int,
                 subpixel: str) -> list[Cell]:
    """Piecewise-uniform cells covering [lo, hi]; continuously varying segments are staircased."""
    fine = 1 if subpixel == "none" else subsamples

    def at(pos):
        sec = CrossSection.from_structures(structures, background, axis, pos, dimension, plane_z_nm)
        own = OwnerMaps.build(sec, grid, fine, 16 * fine)
        return sec, own

    bps = {lo, hi}
    for s in structures:
        bps.update(b for b in _extent_along(s.geometry, axis) if lo < b < hi)
    bps = sorted(bps)
    # Merge breakpoints closer than a nanometre-scale tolerance.
    tol = 1e-6 * max(1.0, hi - lo)
    pts = [bps[0]]
    for b in bps[1:]:
        if b - pts[-1] > tol:
            pts.append(b)
    pts[-1] = hi
    cells: list[Cell] = []
    for a, b in zip(pts[:-1], pts[1:]):
        probes = [at(a + f * (b - a)) for f in (0.25, 0.5, 0.75)]
        uniform = probes[0][1].same_as(probes[1][1]) and probes[1][1].same_as(probes[2][1])
        n = 1
        if not uniform:
            n = taper_cells or _auto_cells(probes[0][1], probes[2][1], grid)
        if max_cell_length_nm:
            n = max(n, math.ceil((b - a) / max_cell_length_nm))
        edges = np.linspace(a, b, n + 1)
        for k in range(n):
            if n == 1 and uniform:
                sec, own = probes[1]
            else:
                sec, own = at(0.5 * (edges[k] + edges[k + 1]))
            key = own.digest() + "|" + sec.material_key()
            if cells and cells[-1].key == key:
                cells[-1].stop = float(edges[k + 1])
            else:
                cells.append(Cell(float(edges[k]), float(edges[k + 1]), sec, own, key))
    return cells


# --- mode cache ------------------------------------------------------------------------


class _ModeCache:
    """Byte-limited LRU of cell mode sets, shared across solves (e.g. length sweeps)."""

    def __init__(self, max_bytes: int = 256 * 2**20):
        self.max_bytes = max_bytes
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0

    def get(self, key):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key, value: ModeSet):
        if key in self._d:
            return
        self._d[key] = value
        self._bytes += value.nbytes()
        while self._bytes > self.max_bytes and len(self._d) > 1:
            _, old = self._d.popitem(last=False)
            self._bytes -= old.nbytes()

    def clear(self):
        self._d.clear()
        self._bytes = 0


MODE_CACHE = _ModeCache()


def cell_modes(cell: Cell, grid: Grid, wavelength_nm: float, num_modes: int, polarization, subpixel: str,
               pml_reflection: float, use_cache: bool = True) -> ModeSet:
    key = (cell.key, repr(grid), float(wavelength_nm), num_modes, polarization, subpixel, pml_reflection)
    if use_cache:
        hit = MODE_CACHE.get(key)
        if hit is not None:
            return hit
    solver = ModeSolver(cell.section, grid, subpixel, pml_reflection=pml_reflection, owners=cell.owners)
    modes = solver.solve(wavelength_nm, num_modes, None, polarization, group_index="none", extra_modes=0)
    ms = ModeSet.from_modes(modes, grid)
    if use_cache:
        MODE_CACHE.put(key, ms)
    return ms


def port_mode_set(section: CrossSection, grid: Grid, lo: tuple[float, float], hi: tuple[float, float],
                  wavelength_nm: float, num_modes: int, polarization, subpixel: str, subsamples: int) -> ModeSet:
    """Modes of a port window (snapped to `grid`), zero-padded onto the full cross-section grid."""
    g, window = sub_grid(grid, lo, hi)
    solver = ModeSolver(section, g, subpixel, subsamples)
    modes = solver.solve(wavelength_nm, num_modes, None, polarization, group_index="none", extra_modes=2)
    return ModeSet.from_modes(modes, grid, window)


__all__ = ["Cell", "MODE_CACHE", "ModeSet", "SMatrix", "cell_modes", "coupling", "interface", "plane_axes",
           "port_mode_set", "slice_device"]
