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
from ...raster import inside_geometry


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


def permittivity(problem: DeviceProblem, grid: Grid2D, wavelength_nm: float, subsamples: int = 4) -> np.ndarray:
    """Complex relative permittivity at the nodes, volume-averaged by supersampling each cell.

    3D geometries are cut at z = ``grid.z0`` (the domain centre). Later structures
    override earlier ones (painter's order), as everywhere in lightsmith.
    """
    wl = np.array([wavelength_nm])
    eps_bg = complex(problem.background.eps(wl)[0])
    structs = [s for s in problem.structures if _cross_section(s.geometry, grid.z0)]
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    eps = np.full(X.shape, eps_bg, dtype=complex)
    if not structs:
        return eps
    offs = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * grid.dl
    frac = np.zeros((len(structs),) + X.shape)
    for ox in offs:
        for oy in offs:
            owner = np.full(X.shape, -1)
            for i, s in enumerate(structs):
                owner[inside_geometry(s.geometry, X + ox, Y + oy, np.full(X.shape, grid.z0))] = i
            for i in range(len(structs)):
                frac[i] += owner == i
    frac /= subsamples**2
    for f, s in zip(frac, structs):
        eps += f * (complex(s.material.eps(wl)[0]) - eps_bg)
    return eps
