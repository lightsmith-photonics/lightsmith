"""Yee grid, stretched-coordinate PML and rasterisation for 2D FDFD.

Grid layout (2D, structures in the xy plane):

- The scalar field (Ez for TM, Hz for TE) lives on the nodes ``(x[i], y[j])``.
- Node ``i`` is flanked by the x-edges ``e = i`` (at ``x[i] - dl/2``) and
  ``e = i + 1`` (at ``x[i] + dl/2``); there are ``Nx + 1`` x-edges. Same in y.
- The field is zero just outside the grid (Dirichlet / PEC for TM, PMC for TE).
- The PML is added *outside* the user's domain: ``pml_cells`` extra cells per side.

SC-PML for exp(-i omega t): s(d) = 1 + i * sigma_max * (d / L)^m, which makes
outgoing waves exp(+i k x) decay inside the layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...problems import Ball, Box, DeviceProblem, Prism
from ...raster import geometry_xy_bounds, inside_geometry


@dataclass(frozen=True)
class Grid2D:
    dl: float  # cell size in nm (uniform, same in x and y)
    x: np.ndarray  # node coordinates (Nx,)
    y: np.ndarray  # node coordinates (Ny,)
    npml: tuple[int, int]  # PML cells on each side, per axis
    z0: float  # cross-section height used to cut 3D geometries

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.x), len(self.y)

    @property
    def size(self) -> int:
        return len(self.x) * len(self.y)

    def axis(self, a: int) -> np.ndarray:
        return self.x if a == 0 else self.y

    def interior(self, a: int) -> tuple[int, int]:
        """Index range [lo, hi) of non-PML nodes along axis `a`."""
        n = len(self.axis(a))
        return self.npml[a], n - self.npml[a]

    def index(self, a: int, coord: float) -> int:
        return int(np.argmin(np.abs(self.axis(a) - coord)))


def make_grid(problem: DeviceProblem, dl: float, npml: tuple[int, int]) -> Grid2D:
    (cx, cy, cz), (lx, ly, _) = problem.domain_center_nm, problem.domain_size_nm
    axes = []
    for c, L, p in ((cx, lx, npml[0]), (cy, ly, npml[1])):
        n = max(int(round(L / dl)), 1) + 2 * p
        axes.append(c + (np.arange(n) - (n - 1) / 2) * dl)
    return Grid2D(dl=float(dl), x=axes[0], y=axes[1], npml=(int(npml[0]), int(npml[1])), z0=float(cz))


def stretch_factors(grid: Grid2D, a: int, k_ref: float, order: int = 3,
                    reflection: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """SC-PML factors along axis `a` at nodes (N,) and at edges (N+1,)."""
    coords = grid.axis(a)
    n, p, dl = len(coords), grid.npml[a], grid.dl
    node_pos = np.arange(n, dtype=float)
    edge_pos = np.arange(n + 1, dtype=float) - 0.5
    if p == 0:
        return np.ones(n, complex), np.ones(n + 1, complex)
    left, right = p - 0.5, n - p - 0.5  # PML interfaces (in index units)
    sigma_max = -(order + 1) * np.log(reflection) / (2 * k_ref * p * dl)

    def s(pos):
        depth = np.maximum(left - pos, 0) + np.maximum(pos - right, 0)
        return 1 + 1j * sigma_max * (depth / p) ** order

    return s(node_pos), s(edge_pos)


def _cross_section(geometry, z0: float) -> bool:
    if isinstance(geometry, Prism):
        return geometry.z_min_nm <= z0 <= geometry.z_max_nm
    if isinstance(geometry, Box):
        return abs(z0 - geometry.center_nm[2]) <= geometry.size_nm[2] / 2
    if isinstance(geometry, Ball):
        return abs(z0 - geometry.center_nm[2]) <= geometry.radius_nm
    return True


def node_fractions(problem: DeviceProblem, grid: Grid2D, subsamples: int = 4) -> np.ndarray:
    """Area fraction of each structure in each node's cell, painter's order: (n_structures, Nx, Ny).

    Later structures override earlier ones, as everywhere in lightsmith. Depends only on the geometry, so a
    solve computes it once. Each structure is tested inside its bounding box only: a layout's structures
    are small against its domain, and a many-vertex bend on the whole grid costs far more than the solve.
    """
    structs = [s for s in problem.structures if _cross_section(s.geometry, grid.z0)]
    shape = (len(grid.x), len(grid.y))
    frac = np.zeros((len(structs),) + shape)
    if not structs:
        return frac
    boxes = []
    for s in structs:
        x0, x1, y0, y1 = geometry_xy_bounds(s.geometry)
        # one cell of margin: the subsamples of a node's cell reach dl/2 from it
        boxes.append((np.s_[int(np.searchsorted(grid.x, x0 - grid.dl)):int(np.searchsorted(grid.x, x1 + grid.dl, side="right"))],
                      np.s_[int(np.searchsorted(grid.y, y0 - grid.dl)):int(np.searchsorted(grid.y, y1 + grid.dl, side="right"))]))
    points = [np.meshgrid(grid.x[bx], grid.y[by], indexing="ij") for bx, by in boxes]
    offs = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * grid.dl
    owner = np.empty(shape, dtype=np.int32)
    for ox in offs:
        for oy in offs:
            owner.fill(-1)
            for i, (s, box, (X, Y)) in enumerate(zip(structs, boxes, points)):
                owner[box][inside_geometry(s.geometry, X + ox, Y + oy, np.full(X.shape, grid.z0))] = i
            for i, box in enumerate(boxes):
                frac[(i, *box)] += owner[box] == i
    return frac / subsamples**2


def permittivity(problem: DeviceProblem, grid: Grid2D, wavelength_nm: float, subsamples: int = 4,
                 fractions: np.ndarray | None = None) -> np.ndarray:
    """Complex relative permittivity at the nodes, volume-averaged by supersampling each cell.

    3D geometries are cut at z = ``grid.z0`` (the domain centre). `fractions` is :func:`node_fractions`, if the
    caller already has it.
    """
    wl = np.array([wavelength_nm])
    eps_bg = complex(problem.background.eps(wl)[0])
    structs = [s for s in problem.structures if _cross_section(s.geometry, grid.z0)]
    eps = np.full((len(grid.x), len(grid.y)), eps_bg, dtype=complex)
    if not structs:
        return eps
    frac = node_fractions(problem, grid, subsamples) if fractions is None else fractions
    for f, s in zip(frac, structs):
        eps += f * (complex(s.material.eps(wl)[0]) - eps_bg)
    return eps
