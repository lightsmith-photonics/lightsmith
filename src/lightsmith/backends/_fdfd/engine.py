"""2D FDFD engine: grid + operator + ports -> S-parameters (plain or differentiable)."""

from __future__ import annotations

import math
import time
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...progress import report
from ...problems import DeviceProblem, wavelength_array
from . import linsolve
from .grid import Grid2D, make_grid, permittivity, stretch_factors
from .modes import SlabMode
from .operators import Diagonals, boundary_masks, helmholtz, matvec
from .ports import PortGeom, decompose, mode_source, port_geometry, port_modes


class FDFDOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    polarization: Literal["te", "tm"] = Field(
        "te", description="2D polarization: 'te' = E in the xy plane (Hz), the usual choice for planar "
        "photonics; 'tm' = E along z (Ez).")
    dl_nm: float | None = Field(None, gt=0, description="Grid cell size. Default: shortest wavelength / "
                                "(cells_per_wavelength * highest refractive index).")
    cells_per_wavelength: float = Field(20.0, gt=4, description="Cells per wavelength in the densest material.")
    pml_cells: int | tuple[int, int] | tuple[int, int, int] = Field(
        16, description="PML thickness in cells, added outside the domain; a pair gives (x, y) (z = y in 3D), "
        "a triple (x, y, z). 0 = PEC/PMC wall on that axis.")
    pml_reflection: float = Field(1e-8, gt=0, lt=1, description="Target normal-incidence PML reflection.")
    solver: Literal["auto", "scipy", "block", "bicgstab", "direct"] = Field(
        "auto", description="2D: auto = SciPy SuperLU on CPU, torch block-tridiagonal LU on GPU. 3D: auto = "
        "SuperLU ('direct') for small grids on CPU, else matrix-free BiCGStab ('bicgstab', CPU or GPU).")
    precision: Literal["double", "single"] = Field("double", description="Factorisation precision (block solver).")
    subsamples: int = Field(4, ge=1, le=16, description="Supersampling per axis for subpixel permittivity.")
    max_cells: int = Field(4_000_000, ge=1, description="Refuse grids larger than this.")
    # 3D (iterative solver) only:
    tol: float = Field(1e-6, gt=0, lt=1, description="3D: relative residual at which BiCGStab stops.")
    max_iterations: int = Field(20_000, ge=1, description="3D: BiCGStab iteration limit per solve.")
    warm_start: bool = Field(True, description="3D: start each differentiable solve from the previous solution "
                             "(forward and adjoint), which speeds up optimisation severalfold.")
    grad_div: bool = Field(False, description="3D: add the grad-div regulariser of Shin & Fan (2013). It does "
                           "not change the solution; in our tests it did not speed up BiCGStab, so it is off.")
    jacobi: bool = Field(False, description="3D: Jacobi-precondition BiCGStab (slower in our tests).")


def _max_index(problem: DeviceProblem, wl: np.ndarray) -> float:
    mats = [problem.background] + [s.material for s in problem.structures]
    return float(max(np.abs(np.real(m.n(wl))).max() for m in mats))


class FDFD2D:
    """Frequency-domain solver for a 2D :class:`DeviceProblem` (structures in the xy plane)."""

    def __init__(self, problem: DeviceProblem, opts: FDFDOptions | None = None, check_size: bool = True):
        if problem.layout is not None:  # GDS layouts become Prism structures
            from ...layout import resolved_device

            problem = resolved_device(problem)
        self.problem = problem
        self.opts = opts = opts or FDFDOptions()
        self.pol = opts.polarization
        self.wavelengths = wavelength_array(problem.wavelengths)
        dl = opts.dl_nm or float(self.wavelengths.min() / (opts.cells_per_wavelength
                                                           * _max_index(problem, self.wavelengths)))
        npml = tuple(opts.pml_cells)[:2] if isinstance(opts.pml_cells, tuple) else (opts.pml_cells, opts.pml_cells)
        self.grid: Grid2D = make_grid(problem, dl, npml)
        if check_size and self.grid.size > opts.max_cells:
            raise ValueError(f"grid {self.grid.shape} has {self.grid.size} cells > max_cells={opts.max_cells}; "
                             "increase dl_nm or max_cells")
        self.ports: list[PortGeom] = [port_geometry(p, self.grid) for p in problem.ports]
        names = [p.name for p in problem.ports]
        if len(set(names)) != len(names):
            raise ValueError("port names must be unique")
        self.outputs: list[tuple[int, int]] = [(k, m) for k, pg in enumerate(self.ports) for m in range(pg.num_modes)]
        self.labels: list[str] = [lab for pg in self.ports for lab in pg.labels()]
        self._eps: dict[float, np.ndarray] = {}
        self.warnings: list[str] = []

    # --- pieces ---------------------------------------------------------------

    def k0dl(self, lam: float) -> float:
        return 2 * math.pi / lam * self.grid.dl

    def base_eps(self, lam: float) -> np.ndarray:
        if lam not in self._eps:
            self._eps[lam] = permittivity(self.problem, self.grid, lam, self.opts.subsamples)
        return self._eps[lam]

    def _stretch(self, lam: float):
        n_bg = float(np.real(self.problem.background.n(np.array([lam]))[0]))
        k_ref = 2 * math.pi / lam * max(n_bg, 1.0)
        return [stretch_factors(self.grid, a, k_ref, reflection=self.opts.pml_reflection) for a in (0, 1)]

    def diagonals(self, xp, eps, lam: float, device=None) -> Diagonals:
        (sxn, sxe), (syn, sye) = self._stretch(lam)
        conv = (lambda a: xp.asarray(a)) if device is None else (lambda a: xp.asarray(a, device=device))
        masks = boundary_masks(xp, self.grid.shape, device=device)
        return helmholtz(xp, eps, self.pol, self.k0dl(lam), (conv(sxn), conv(sxe)), (conv(syn), conv(sye)), masks)

    def modes(self, eps_np: np.ndarray, lam: float) -> list[list[SlabMode]]:
        k0dl = self.k0dl(lam)
        out = []
        for pg in self.ports:
            ms = port_modes(pg, eps_np, self.pol, k0dl)
            for m, md in enumerate(ms):
                if not md.guided:
                    msg = (f"port {pg.name!r} mode {m} at {lam:g} nm is not guided (neff={md.neff.real:.4f}) "
                           "or touches the port window edge; widen the port or check num_modes")
                    if msg not in self.warnings:
                        self.warnings.append(msg)
            out.append(ms)
        return out

    def label_index(self, port: str, mode: int = 0) -> int:
        for idx, (k, m) in enumerate(self.outputs):
            if self.ports[k].name == port and m == mode:
                return idx
        raise KeyError(f"no port {port!r} with mode {mode}; ports: {self.labels}")

    def _sources(self, eps_np, lam, modes, which: list[int]) -> np.ndarray:
        d_np = self.diagonals(np, eps_np, lam)
        cols = [mode_source(self.ports[k], modes[k][m], d_np, matvec) for k, m in (self.outputs[i] for i in which)]
        return np.stack(cols, axis=-1)

    def _smatrix(self, xp, X, modes, which: list[int], device=None):
        """S (n_out, n_src).

        Outgoing amplitudes C (n_out, n_src) are the S-parameters when the only incoming
        wave is the launched mode. When every port mode is excited, the incoming
        amplitudes A are measured as well and S = C A^-1, which also removes the
        (tiny) effect of waves re-entering from imperfect PMLs.
        """
        rows_a, rows_c = [], []
        for k, m in self.outputs:
            pg, md = self.ports[k], modes[k][m]
            proj = xp.asarray(md.projector) if device is None else xp.asarray(md.projector, device=device)
            a, c = decompose(xp, pg, md, X, proj)
            rows_a.append(a)
            rows_c.append(c)
        C = xp.stack(rows_c, axis=0)
        if list(which) != list(range(len(self.outputs))):
            return C
        A = xp.stack(rows_a, axis=0)
        return xp.linalg.solve(A.T, C.T).T  # S A = C

    # --- solves -------------------------------------------------------------------

    def solve_fields(self, lam: float, sources: list[int] | None = None, eps: np.ndarray | None = None):
        """Fields (Nx, Ny, n_src) and S (n_out, n_src) at one wavelength, with NumPy/SciPy."""
        eps = self.base_eps(lam) if eps is None else eps
        which = list(range(len(self.outputs))) if sources is None else list(sources)
        modes = self.modes(eps, lam)
        B = self._sources(eps, lam, modes, which)
        X = linsolve.solve(self.diagonals(np, eps, lam), B, "scipy")
        return X, self._smatrix(np, X, modes, which)

    def solve(self, device=None) -> tuple[np.ndarray, dict[str, Any]]:
        """Full S-matrix (n_wl, n_out, n_in). On a torch device the block solver is used (no gradients)."""
        t0 = time.perf_counter()
        info: dict[str, Any] = {}
        S = []
        use_torch = device is not None and getattr(device, "framework", "numpy") == "torch"
        for i, lam in enumerate(self.wavelengths):
            report("fdfd_solve", 0.95 * i / len(self.wavelengths), wavelength_nm=float(lam))
            if not use_torch and self.opts.solver != "block":
                S.append(self.solve_fields(float(lam))[1])
                info["factor"] = "scipy-superlu"
            else:
                import torch

                with torch.no_grad():
                    tdev = device.native_device() if use_torch else torch.device("cpu")
                    res = self.differentiable(torch.as_tensor(self.base_eps(float(lam)), device=tdev),
                                              tdev, wavelengths=[float(lam)], info=info, with_fields=False)
                    S.append(res["s"][0].cpu().numpy())
        S = np.asarray(S)
        info["runtime_s"] = time.perf_counter() - t0
        return S, info

    def differentiable(self, eps, device=None, sources: list[int] | None = None, wavelengths=None,
                       info: dict | None = None, with_fields: bool = True) -> dict[str, Any]:
        """Torch-differentiable solve.

        eps: torch tensor (Nx, Ny) or (n_wl, Nx, Ny) of node permittivities (may require grad).
        Returns {"s": (n_wl, n_out, n_src) complex, "fields": (n_wl, n_src, Nx, Ny) complex | None,
        "outputs": labels, "inputs": labels}. Port modes are computed from the detached eps
        (ports must not overlap the design region).
        """
        import array_api_compat.torch as xp
        import torch

        tdev = torch.device(device) if device is not None and not isinstance(device, torch.device) else device
        if tdev is None:
            tdev = eps.device
        wls = self.wavelengths if wavelengths is None else np.asarray(wavelengths, float)
        which = list(range(len(self.outputs))) if sources is None else list(sources)
        dtype = torch.complex64 if self.opts.precision == "single" else None
        eps = eps.to(tdev)
        S, F = [], []
        for w, lam in enumerate(wls):
            lam = float(lam)
            e = eps[w] if eps.ndim == 3 else eps
            e = e.to(torch.complex128) if not e.is_complex() else e
            e_np = e.detach().cpu().numpy()
            modes = self.modes(e_np, lam)
            B = torch.as_tensor(self._sources(e_np, lam, modes, which), device=tdev)
            d = self.diagonals(xp, e, lam, device=tdev)
            X = linsolve.solve(d, B, self.opts.solver, dtype, info)
            S.append(self._smatrix(xp, X, modes, which, device=tdev))
            if with_fields:
                F.append(X.permute(2, 0, 1))
        return {"s": torch.stack(S), "fields": torch.stack(F) if with_fields else None,
                "outputs": list(self.labels), "inputs": [self.labels[i] for i in which]}

    def estimate(self, device=None) -> dict[str, Any]:
        nx, ny = self.grid.shape
        n = nx * ny
        bs = min(nx, ny)
        scipy_bytes = int(16 * n * (12 + 3 * math.log2(max(n, 2))))
        block_bytes = int((8 if self.opts.precision == "single" else 16) * max(nx, ny) * bs * bs * 2)
        return {"grid": [nx, ny], "dl_nm": self.grid.dl, "unknowns": n, "pml_cells": list(self.grid.npml),
                "polarization": self.pol, "sources": len(self.outputs), "wavelengths": len(self.wavelengths),
                "precision": self.opts.precision,
                # the solver `solve` will use on this device: block-tridiagonal LU on a GPU, SuperLU on the CPU
                "memory_bytes": block_bytes if device is not None and device.is_gpu else scipy_bytes,
                "memory_bytes_scipy": scipy_bytes, "memory_bytes_block": block_bytes,
                "fits": n <= self.opts.max_cells}
