"""Band computation driver: picks dense or iterative per polarization, loops over k, collects frequencies.

Solver choice for ``solver="auto"`` (measured on rods/holes in a square lattice):

- TM (E_z): dense Ho-Chan-Soukoulis. For TM it is exactly the generalised problem
  |k+G|^2 e = lambda [eps] e and converges very fast (1e-5 at ~300 plane waves).
- TE (H_z) and 3D: matrix-free LOBPCG with the subpixel-smoothed tensor, which
  converges ~quadratically with resolution (~3e-3 at 32/a, ~5e-4 at 64/a). The
  dense HCS rule treats every E component as tangential and converges slowly
  for TE at curved interfaces (~1 % at 1200 plane waves), so it is used for TE
  only when asked for (num_plane_waves or solver="dense").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...device import Device
from .geometry import Cell, eps_table, smoothed_inverse_eps
from .solvers import FFTOperator, dense_bands, eigen_to_freq, lobpcg, plane_wave_set, toeplitz_eps

# Plane waves per unit cell measure (a^d) when num_plane_waves is not given.
DEFAULT_PW_DENSITY = {2: 400, 3: 600}
# Grid points per lattice constant for the iterative solver.
DEFAULT_RESOLUTION = {2: 32, 3: 16}
# Largest dense matrix "auto" accepts before switching to LOBPCG.
DENSE_AUTO_LIMIT = 2500


@dataclass
class Plan:
    solver: str  # "dense" | "lobpcg"
    precision: str
    num_plane_waves: int
    matrix_size: int
    grid: tuple[int, ...] | None
    idx: np.ndarray | None = None


@dataclass
class BandOutput:
    freqs: dict[str, np.ndarray]  # pol -> (nk, nb)
    info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def make_plan(cell: Cell, pol: str, opts, device: Device) -> Plan:
    lat = cell.lattice
    d = lat.dim
    comps = 2 if d == 3 else 1
    n_dense = opts.num_plane_waves or int(math.ceil(DEFAULT_PW_DENSITY[d] * lat.cell_measure))
    res = opts.resolution or DEFAULT_RESOLUTION[d]
    grid = tuple(max(8, int(round(res * float(np.linalg.norm(lat.vectors[i]))))) for i in range(d))
    solver = opts.solver
    if solver == "auto":
        if opts.num_plane_waves is not None and opts.resolution is None:
            solver = "dense"
        elif opts.resolution is not None and opts.num_plane_waves is None:
            solver = "lobpcg"
        elif pol == "tm" or not cell.inclusions:
            solver = "dense" if comps * n_dense <= DENSE_AUTO_LIMIT else "lobpcg"
        else:
            solver = "lobpcg"
    # auto precision: single for dense on a GPU (batched eigvalsh, ~1e-5 accurate); LOBPCG stays double,
    # since its Gram-based orthogonalisation loses too much in single precision.
    single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu and solver == "dense")
    precision = "single" if single else "double"
    if solver == "dense":
        idx = plane_wave_set(lat.reciprocal, n_dense)
        return Plan("dense", precision, len(idx), comps * len(idx), None, idx)
    n = int(np.prod(grid))
    return Plan("lobpcg", precision, n, comps * n, grid)


def estimate_memory(plan: Plan, n_k: int, num_bands: int) -> int:
    csize = 8 if plan.precision == "single" else 16
    if plan.solver == "dense":
        n = plan.num_plane_waves
        # E and eta (double), plus a chunk of k-matrices with eigh workspace (<= 256 MiB by construction).
        return int(2 * 16 * n * n + min(256 * 2**20, 4 * n_k * plan.matrix_size**2 * csize))
    m = num_bands + max(2, num_bands // 4)
    # ~10 blocks of m vectors, 3-component FFT work arrays, and the eps tensor.
    return int(csize * plan.matrix_size * m * 14 + 8 * 9 * plan.num_plane_waves)


def _dense(cell: Cell, plan: Plan, pol: str, k_cart, num_bands, opts, device, cache: dict) -> np.ndarray:
    xp = device.namespace()
    cdtype = xp.complex64 if plan.precision == "single" else xp.complex128
    lat = cell.lattice
    idx = plan.idx
    key = (len(idx), plan.precision)
    if key not in cache:
        mmax = tuple(int(np.abs(idx[:, i]).max()) for i in range(lat.dim))
        sub = opts.subsamples or (4 if lat.dim == 2 else 2)
        tab = eps_table(cell, mmax, opts.eps_fourier, resolution=64 if lat.dim == 2 else 24, subsamples=sub)
        e = toeplitz_eps(device.asarray(tab.values, dtype=xp.complex128), tab.half, idx, xp)
        eta = xp.linalg.inv(e)
        eta = (eta + xp.conj(xp.permute_dims(eta, (1, 0)))) / 2
        cache[key] = (xp.astype(eta, cdtype), tab.method)
    eta, method = cache[key]
    cache["eps_fourier"] = method
    lam = dense_bands(eta, idx, lat.reciprocal, k_cart, pol, num_bands, xp, device, cdtype)
    return Device.to_numpy(eigen_to_freq(lam, xp)).astype(float)


def _iterative(cell: Cell, plan: Plan, pol: str, k_cart, num_bands, opts, device, out: BandOutput) -> np.ndarray:
    xp = device.namespace()
    cdtype = xp.complex64 if plan.precision == "single" else xp.complex128
    lat = cell.lattice
    sub = opts.subsamples or (4 if lat.dim == 2 else 2)
    eta, eps_mean, _ = smoothed_inverse_eps(cell, plan.grid, sub)
    tol = opts.tol or (1e-4 if plan.precision == "single" else 1e-6)
    extra = opts.extra_bands if opts.extra_bands is not None else max(2, num_bands // 4)
    m = min(num_bands + extra, plan.matrix_size)
    rng = np.random.default_rng(0)
    op = FFTOperator(pol, plan.grid, lat.reciprocal, eta, eps_mean, device, cdtype)
    freqs = np.zeros((len(k_cart), num_bands))
    its, worst = [], 0.0
    x, frame = None, None
    for ik, k in enumerate(k_cart):
        op.set_k(np.asarray(k, float))
        # Warm start from the previous k, plus a little noise so that subspaces the previous block
        # lacks (symmetry-decoupled or, in uniform media, exactly invariant) can still enter.
        x0 = op.initial(m, rng) if x is None else op.carry(x, frame) + op.noise(m, rng, 1e-2)
        th, x, info = lobpcg(op.apply, x0, op.precondition, num_bands, xp, tol, opts.max_iter)
        frame = getattr(op, "frame", None)
        its.append(info.iterations)
        if not info.converged:
            # One cold restart before giving up on this k-point.
            th, x, info = lobpcg(op.apply, op.initial(m, rng), op.precondition, num_bands, xp, tol, opts.max_iter)
            its[-1] += info.iterations
            if not info.converged:
                out.warnings.append(f"LOBPCG ({pol}) did not converge at k-point {ik} "
                                    f"(residual {info.max_residual:.1e}); raise max_iter or tol")
        worst = max(worst, info.max_residual)
        freqs[ik] = np.sqrt(np.clip(np.sort(th)[:num_bands], 0, None)) / (2 * np.pi)
    out.info.setdefault("iterations", {})[pol] = its
    out.info.setdefault("max_residual", {})[pol] = worst
    return freqs


def compute_bands(cell: Cell, k_cart: np.ndarray, polarizations: list[str], num_bands: int, opts,
                  device: Device) -> BandOutput:
    out = BandOutput({}, {"solver": {}, "num_plane_waves": {}, "matrix_size": {}, "eps_treatment": {}})
    cache: dict = {}
    for pol in polarizations:
        plan = make_plan(cell, pol, opts, device)
        if num_bands > plan.matrix_size:
            raise ValueError(f"num_bands={num_bands} exceeds the basis size {plan.matrix_size}; add plane waves")
        out.info["precision"] = plan.precision
        out.info["solver"][pol] = plan.solver
        out.info["num_plane_waves"][pol] = plan.num_plane_waves
        out.info["matrix_size"][pol] = plan.matrix_size
        if plan.solver == "dense":
            out.freqs[pol] = _dense(cell, plan, pol, k_cart, num_bands, opts, device, cache)
            out.info["eps_treatment"][pol] = f"ho_chan_soukoulis_inverse ({cache['eps_fourier']} eps_hat)"
        else:
            out.freqs[pol] = _iterative(cell, plan, pol, k_cart, num_bands, opts, device, out)
            out.info["eps_treatment"][pol] = "subpixel_smoothed_tensor"
            out.info.setdefault("grid", list(plan.grid))
    return out
