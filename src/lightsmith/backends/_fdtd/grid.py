"""Non-uniform (graded) tensor-product grids: per-axis node coordinates, refinement regions and grading.

The Yee grid stays a tensor product: along each axis the node planes x_0 < x_1 < ... < x_{n-1} are free, and
the three axes are independent (as Lumerical's override regions and Tidy3D's graded grids). Along an axis,
primary widths p_i = x_{i+1} - x_i separate nodes; E_c sits at the midpoint of a primary cell along c, H_c at
the midpoints along the other two axes. The dual width of node i is d_i = (p_{i-1} + p_i) / 2, the distance
between the midpoints on either side. The H update divides differences of E along a by p (at H's half-integer
index), the E update divides differences of H by d (at E's node index). Weighted by the cell volumes (E_c:
p_c d_a d_b; H_c: d_c p_a p_b) the two curls are adjoint, so the scheme conserves the same discrete energy as on
a uniform grid and is stable below the Courant limit of the smallest cells.

Widths come from a spacing function: the base spacing, refined in each region [lo, hi] to its own spacing s,
and grading back to the base no faster than h(x) = s + (r - 1) dist(x, [lo, hi]), r the growth ratio. Cells
are marched outward from the scene centre with width min h over the cell, so neighbouring cells differ by at
most about r (``max_ratio`` reports the actual bound). Along a mirror axis the spacing function is made
symmetric about the centre, so the half grid is the mirror image of the other half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import dual_widths


@dataclass(frozen=True)
class Region:
    """A refinement box (world nm) with its spacing per axis."""

    lo: np.ndarray
    hi: np.ndarray
    spacing: np.ndarray

    @staticmethod
    def make(lo, hi, spacing) -> "Region":
        return Region(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float),
                      np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy())


def spacing_fn(base: float, regions: list[Region], a: int, ratio: float, center: float | None = None):
    """h(x) along axis a (vectorised); symmetric about `center` when given (a mirror axis)."""
    rs = [(r.lo[a], r.hi[a], r.spacing[a]) for r in regions if r.spacing[a] < base]

    def h(x):
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape, float(base))
        for lo, hi, s in rs:
            dist = np.maximum(np.maximum(lo - x, x - hi), 0.0)
            out = np.minimum(out, s + (ratio - 1) * dist)
        return out

    if center is None:
        return h, rs
    return (lambda x: np.minimum(h(x), h(2 * center - np.asarray(x, dtype=float)))), rs


def march(h, x0: float, direction: int, base: float, beyond: float) -> np.ndarray:
    """Widths of the cells from x0 outward (direction +-1) until every further cell is `base`: the graded zone.

    Each cell takes the minimum of h over itself (a few fixed-point passes), so it never overruns a finer region;
    marching stops once past `beyond` (the last region edge on that side) with a full base cell."""
    widths, x = [], float(x0)
    t = np.linspace(0.0, 1.0, 17)
    while True:
        w = float(h(np.array([x]))[0])
        for _ in range(6):
            w_new = float(np.min(h(x + direction * t * w)))
            if w_new >= w * (1 - 1e-12):
                break
            w = w_new
        if w >= base * (1 - 1e-12) and direction * (x - beyond) >= 0:
            return np.asarray(widths)
        widths.append(min(w, base))
        x += direction * widths[-1]
        if len(widths) > 1_000_000:
            raise ValueError("graded grid: too many cells (check mesh_override spacings)")


@dataclass
class Grid:
    """Node coordinates (world nm) per axis. Index i of axis a is node plane nodes[a][i]."""

    nodes: tuple

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(len(x) for x in self.nodes)

    def prim(self, a: int) -> np.ndarray:
        """Primary widths p_i = x_{i+1} - x_i, (n - 1,)."""
        return np.diff(self.nodes[a])

    def dual(self, a: int, mirror: int = 0) -> np.ndarray:
        """Dual widths d_i = (p_{i-1} + p_i) / 2, (n,); the ends take their one cell, except the ghost plane of a
        mirror axis (index 0, the image of node 2), which takes d_2."""
        return dual_widths(self.prim(a), mirror)

    def uniform_axis(self, a: int, tol: float = 1e-9) -> bool:
        p = self.prim(a)
        return bool(np.ptp(p) <= tol * p.mean())

    @property
    def uniform(self) -> bool:
        return all(self.uniform_axis(a) for a in range(3))

    def widths(self) -> tuple:
        """Per axis the primary widths, or None for a uniform axis (the Engine's ``widths``)."""
        return tuple(None if self.uniform_axis(a) else self.prim(a) for a in range(3))

    @property
    def origin(self) -> np.ndarray:
        return np.array([x[0] for x in self.nodes], dtype=float)

    def min_spacing(self, a: int | None = None) -> float:
        if a is None:
            return min(self.min_spacing(b) for b in range(3))
        return float(self.prim(a).min())

    def max_spacing(self) -> float:
        return max(float(self.prim(a).max()) for a in range(3))

    def max_ratio(self) -> float:
        """Largest width ratio between neighbouring cells along any axis."""
        r = 1.0
        for a in range(3):
            p = self.prim(a)
            if len(p) > 1:
                q = p[1:] / p[:-1]
                r = max(r, float(np.max(np.maximum(q, 1 / q))))
        return r

    def coord(self, a: int, u) -> np.ndarray:
        """World coordinate of fractional index u along axis a (linear within a cell, so half-integers are
        midpoints; the end cells continue beyond the grid)."""
        u = np.asarray(u, dtype=float)
        x, p = self.nodes[a], self.prim(a)
        i = np.clip(np.floor(u + 1e-9).astype(np.int64), 0, len(p) - 1)
        return x[i] + (u - i) * p[i]

    def coords(self, u) -> np.ndarray:
        """World positions (..., 3) of fractional index triples u (..., 3)."""
        u = np.asarray(u, dtype=float)
        return np.stack([self.coord(a, u[..., a]) for a in range(3)], axis=-1)

    def index_of(self, a: int, x) -> np.ndarray:
        """Fractional node index of world coordinate x along axis a (inverse of coord)."""
        x = np.asarray(x, dtype=float)
        nodes, p = self.nodes[a], self.prim(a)
        i = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(p) - 1)
        return i + (x - nodes[i]) / p[i]

    def crop(self, off) -> "Grid":
        """The grid from node off[a] on along every axis."""
        return Grid(tuple(np.asarray(x[int(o):]) for x, o in zip(self.nodes, off)))

    def half_widths(self, a: int, j) -> tuple[np.ndarray, np.ndarray]:
        """(p_{j-1} / 2, p_j / 2): the parts of node j's dual cell below and above it (end cells clamped)."""
        p = self.prim(a)
        j = np.asarray(j, dtype=np.int64)
        return 0.5 * p[np.clip(j - 1, 0, len(p) - 1)], 0.5 * p[np.clip(j, 0, len(p) - 1)]

    def e_cells(self, c: int, idx3) -> tuple[np.ndarray, np.ndarray]:
        """Centre (n, 3) and size (n, 3) of the pixels of E_c nodes idx3 (n, 3): the primary cell along c, the
        dual cell (asymmetric about the node) along the other axes."""
        idx3 = np.asarray(idx3, dtype=np.int64)
        ctr, size = np.empty(idx3.shape), np.empty(idx3.shape)
        for a in range(3):
            j = idx3[:, a]
            if a == c:
                p = self.prim(a)[np.clip(j, 0, self.shape[a] - 2)]
                ctr[:, a], size[:, a] = self.nodes[a][np.clip(j, 0, self.shape[a] - 2)] + 0.5 * p, p
            else:
                lo, hi = self.half_widths(a, j)
                ctr[:, a], size[:, a] = self.nodes[a][j] + 0.5 * (hi - lo), lo + hi
        return ctr, size

    def vertex_cells(self, idx3) -> tuple[np.ndarray, np.ndarray]:
        """Centre and size of the dual cells (cubes on a uniform grid) of vertices idx3 (n, 3)."""
        idx3 = np.asarray(idx3, dtype=np.int64)
        ctr, size = np.empty(idx3.shape), np.empty(idx3.shape)
        for a in range(3):
            lo, hi = self.half_widths(a, idx3[:, a])
            ctr[:, a], size[:, a] = self.nodes[a][idx3[:, a]] + 0.5 * (hi - lo), lo + hi
        return ctr, size

    def node_weights(self, a: int, lo: int, hi: int) -> np.ndarray:
        """Trapezoid weights (nm) of the nodes lo..hi along a (half cells at both ends)."""
        p = self.prim(a)[lo:hi]
        w = np.zeros(hi - lo + 1)
        w[:-1] += 0.5 * p
        w[1:] += 0.5 * p
        return w

    def describe(self) -> dict:
        return {"kind": "graded", "spacing_min_nm": [self.min_spacing(a) for a in range(3)],
                "spacing_max_nm": [float(self.prim(a).max()) for a in range(3)],
                "max_ratio": self.max_ratio()}


def axis_nodes(center: float, base: float, regions: list[Region], a: int, ratio: float, lo_extent: float,
               hi_extent: float, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    """Graded-zone widths below and above `center` along axis a, each ordered outward from the centre."""
    h, rs = spacing_fn(base, regions, a, ratio, center if mirror else None)
    far_hi = max([center + hi_extent] + [hi for _, hi, _ in rs] + ([2 * center - lo for lo, _, _ in rs] if mirror else []))
    far_lo = min([center - lo_extent] + [lo for lo, _, _ in rs] + ([2 * center - hi for _, hi, _ in rs] if mirror else []))
    if not rs:
        return np.zeros(0), np.zeros(0)
    return march(h, center, -1, base, far_lo), march(h, center, 1, base, far_hi)
