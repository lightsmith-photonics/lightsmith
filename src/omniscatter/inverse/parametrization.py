"""Density parametrisation for topology optimisation (torch).

Pipeline per design region (Hammond et al., Opt. Express 29, 23916 (2021)):

    rho (raw, in [0, 1])  --symmetry-->  --conic filter-->  rho_t
    rho_t  --tanh projection(beta, eta)-->  rho_p  --> eps = eps_lo + rho_p (eps_hi - eps_lo)

Minimum feature sizes are imposed with the geometric constraints of Zhou et al.,
Struct. Multidisc. Optim. 52, 1179 (2015), in the form of Hammond et al.:

    g_s = mean(rho_p  * exp(-c |grad rho_t|^2) * min(rho_t - eta_e, 0)^2)   (solid)
    g_v = mean((1 - rho_p) * exp(-c |grad rho_t|^2) * min(eta_d - rho_t, 0)^2) (void)

with eta_e/eta_d from the conic-filter relations (Qian & Sigmund 2013) and c = 64 R^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def conic_kernel(radius_cells: float):
    import torch

    r = max(radius_cells, 1e-9)
    n = max(math.ceil(r) - 1, 0)  # weights vanish at distance >= r
    ax = torch.arange(-n, n + 1, dtype=torch.float64)
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    w = torch.clamp(1 - torch.sqrt(X**2 + Y**2) / r, min=0)
    return w / w.sum()


def conic_filter(rho, radius_cells: float):
    """Normalised conic (hat) filter; renormalised near the region edge (no padding bias)."""
    import torch
    import torch.nn.functional as F

    if radius_cells <= 1:
        return rho
    k = conic_kernel(radius_cells).to(rho.device, rho.dtype)
    pad = k.shape[0] // 2
    num = F.conv2d(rho[None, None], k[None, None], padding=pad)[0, 0]
    den = F.conv2d(torch.ones_like(rho)[None, None], k[None, None], padding=pad)[0, 0]
    return num / den


def tanh_projection(rho, beta: float, eta: float = 0.5):
    """Smoothed Heaviside; beta = inf gives a hard threshold at eta."""
    import torch

    if math.isinf(beta):
        return (rho > eta).to(rho.dtype)
    if beta <= 0:
        return rho
    tb = math.tanh(beta * eta)
    return (tb + torch.tanh(beta * (rho - eta))) / (tb + math.tanh(beta * (1 - eta)))


def eta_from_lengthscale(min_feature_nm: float, filter_radius_nm: float) -> tuple[float, float]:
    """(eta_e, eta_d) for a target minimum length scale with a conic filter of the given radius."""
    r = min_feature_nm / (2 * filter_radius_nm)
    if r <= 0.5:
        eta_e = 0.5 + r**2
    elif r <= 1:
        eta_e = 1 - (1 - r) ** 2
    else:
        raise ValueError("min_feature_nm must be <= 2 * filter_radius_nm; enlarge the filter radius")
    return eta_e, 1 - eta_e


def lengthscale_constraints(rho_t, rho_p, dl_nm: float, filter_radius_nm: float, min_feature_nm: float):
    """(g_solid, g_void): ~0 when every solid/void feature is wider than min_feature_nm."""
    import torch

    eta_e, eta_d = eta_from_lengthscale(min_feature_nm, filter_radius_nm)
    gx = torch.zeros_like(rho_t)
    gy = torch.zeros_like(rho_t)
    gx[1:-1] = (rho_t[2:] - rho_t[:-2]) / (2 * dl_nm)
    gy[:, 1:-1] = (rho_t[:, 2:] - rho_t[:, :-2]) / (2 * dl_nm)
    c = 64 * filter_radius_nm**2
    ind = torch.exp(-c * (gx**2 + gy**2))
    g_s = torch.mean(rho_p * ind * torch.clamp(rho_t - eta_e, max=0) ** 2)
    g_v = torch.mean((1 - rho_p) * ind * torch.clamp(eta_d - rho_t, max=0) ** 2)
    return g_s, g_v


def grayness(rho_p) -> float:
    """Mean 4 rho (1 - rho): 0 for a binary design, 1 for all-0.5."""
    r = rho_p.detach() if hasattr(rho_p, "detach") else np.asarray(rho_p)
    return float((4 * r * (1 - r)).mean())


def binarization(rho_p) -> float:
    """1 - grayness: 1 for a fully binary design."""
    return 1 - grayness(rho_p)


@dataclass
class RegionMap:
    """Where a design region sits on the solver grid (rectangular block of nodes)."""

    name: str
    ix: slice
    iy: slice
    shape: tuple[int, int]
    x_nm: np.ndarray  # node coordinates inside the region
    y_nm: np.ndarray
    dl_nm: float
    eps_lo: np.ndarray  # (n_wl,) complex permittivities of the two phases
    eps_hi: np.ndarray
    filter_radius_nm: float = 0.0
    eta: float = 0.5
    min_feature_nm: float | None = None
    symmetry: tuple[str, ...] = field(default_factory=tuple)

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    def symmetrize(self, rho):
        if "x" in self.symmetry:  # mirror about the region's vertical centre line (x -> -x)
            rho = (rho + rho.flip(0)) / 2
        if "y" in self.symmetry:
            rho = (rho + rho.flip(1)) / 2
        return rho

    def filtered(self, rho):
        return conic_filter(self.symmetrize(rho), self.filter_radius_nm / self.dl_nm)

    def projected(self, rho, beta: float):
        return tanh_projection(self.filtered(rho), beta, self.eta)


def region_map(name: str, center_nm, size_nm, grid, eps_lo, eps_hi, **kw) -> RegionMap:
    """Map a box (center, size; z ignored in 2D) to the block of grid nodes inside it."""
    sl = []
    for a, coords in enumerate((grid.x, grid.y)):
        inside = np.nonzero(np.abs(coords - center_nm[a]) <= size_nm[a] / 2 + 1e-9)[0]
        if len(inside) < 2:
            raise ValueError(f"design region {name!r} spans fewer than 2 grid cells along {'xy'[a]}")
        sl.append(slice(int(inside[0]), int(inside[-1]) + 1))
    shape = (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
    return RegionMap(name=name, ix=sl[0], iy=sl[1], shape=shape, x_nm=grid.x[sl[0]], y_nm=grid.y[sl[1]],
                     dl_nm=float(grid.dl), eps_lo=np.atleast_1d(np.asarray(eps_lo, complex)),
                     eps_hi=np.atleast_1d(np.asarray(eps_hi, complex)), **kw)


def compose_eps(base_eps, regions: list[RegionMap], rhos_p: list):
    """Insert projected densities into the base permittivity (n_wl, Nx, Ny); differentiable."""
    import torch

    eps = base_eps.clone()
    for reg, rp in zip(regions, rhos_p):
        lo = torch.as_tensor(reg.eps_lo, device=eps.device)[:, None, None]
        hi = torch.as_tensor(reg.eps_hi, device=eps.device)[:, None, None]
        eps[:, reg.ix, reg.iy] = lo + rp[None].to(eps.dtype) * (hi - lo)
    return eps
