"""Uniform 3D Yee grid, SC-PML stretch factors and rasterisation for 3D FDFD.

Nodes sit at ``(x[i], y[j], z[l])``; the E components are half a cell up along
their own axis (see ``operators``). The PML is added outside the user's domain,
``npml[a]`` cells per side, as in the 2D solver (whose stretch-factor function is
reused: it only needs ``axis``, ``npml`` and ``dl``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...problems import Box, DeviceProblem, Prism
from ...raster import fill_fractions_3d, inside_profile
from .._fdfd.grid import stretch_factors

__all__ = ["Grid3D", "make_grid", "node_fractions", "stretch_factors"]


@dataclass(frozen=True)
class Grid3D:
    dl: float  # cell size in nm (cubic cells)
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    npml: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.x), len(self.y), len(self.z)

    @property
    def size(self) -> int:
        return len(self.x) * len(self.y) * len(self.z)

    def axis(self, a: int) -> np.ndarray:
        return (self.x, self.y, self.z)[a]

    def interior(self, a: int) -> tuple[int, int]:
        """Index range [lo, hi) of non-PML nodes along axis `a`."""
        return self.npml[a], len(self.axis(a)) - self.npml[a]

    def index(self, a: int, coord: float) -> int:
        return int(np.argmin(np.abs(self.axis(a) - coord)))


def make_grid(problem: DeviceProblem, dl: float, npml: tuple[int, int, int]) -> Grid3D:
    axes = []
    for c, L, p in zip(problem.domain_center_nm, problem.domain_size_nm, npml):
        n = max(int(round(L / dl)), 1) + 2 * p
        axes.append(c + (np.arange(n) - (n - 1) / 2) * dl)
    return Grid3D(dl=float(dl), x=axes[0], y=axes[1], z=axes[2], npml=tuple(int(p) for p in npml))


def _separable(g) -> bool:
    return isinstance(g, Box) or (isinstance(g, Prism) and g.sidewall_angle_deg == 0)


def node_fractions(problem: DeviceProblem, grid: Grid3D, subsamples: int) -> np.ndarray:
    """Volume fraction of each structure in each node's cell, painter's order: (n_structures, Nx, Ny, Nz).

    Vertical prisms and boxes (planar photonics) are tested in the xy plane once per
    (x, y) subsample and combined with a 1D z test, which keeps many-vertex polygons
    (exported inverse designs) cheap; other geometries use the generic 3D supersampling.
    """
    geoms = [s.geometry for s in problem.structures]
    if not geoms:
        return np.zeros((0,) + grid.shape)
    if not all(_separable(g) for g in geoms):
        return fill_fractions_3d(geoms, grid.x, grid.y, grid.z, subsamples)
    offs = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * grid.dl
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    frac = np.zeros((len(geoms),) + grid.shape)

    def in_xy(g, xs, ys):
        if isinstance(g, Box):
            return (np.abs(xs - g.center_nm[0]) <= g.size_nm[0] / 2) & (np.abs(ys - g.center_nm[1]) <= g.size_nm[1] / 2)
        return inside_profile(g.profile, xs, ys)

    def in_z(g, zs):
        if isinstance(g, Box):
            return np.abs(zs - g.center_nm[2]) <= g.size_nm[2] / 2
        return (zs >= g.z_min_nm) & (zs <= g.z_max_nm)

    zmask = [[in_z(g, grid.z + oz) for oz in offs] for g in geoms]
    for ox in offs:
        for oy in offs:
            xy = [in_xy(g, X + ox, Y + oy) for g in geoms]
            for k in range(len(offs)):
                owner = np.full(grid.shape, -1, dtype=np.int32)
                for i in range(len(geoms)):
                    if xy[i].any() and zmask[i][k].any():
                        owner[xy[i][:, :, None] & zmask[i][k][None, None, :]] = i
                for i in range(len(geoms)):
                    frac[i] += owner == i
    return frac / subsamples**3


def permittivity(problem: DeviceProblem, fractions: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """Node permittivity (Nx, Ny, Nz) from cached fill fractions (volume-averaged at interfaces)."""
    wl = np.array([wavelength_nm])
    eps_bg = complex(problem.background.eps(wl)[0])
    eps = np.full(fractions.shape[1:], eps_bg, dtype=complex)
    for f, s in zip(fractions, problem.structures):
        eps += f * (complex(s.material.eps(wl)[0]) - eps_bg)
    return eps
