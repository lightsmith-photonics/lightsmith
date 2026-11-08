"""3D FDFD engine: grid + operator + ports -> S-parameters (plain or differentiable)."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ...progress import report
from ...problems import DeviceProblem, wavelength_array
from . import linsolve
from .grid import Grid3D, make_grid, node_fractions, permittivity, stretch_factors
from .modes import VectorMode
from .operators import Maxwell3D, Stretch, component_eps
from .ports import PortGeom3D, decompose, mode_source, port_geometry, port_modes

#: Default resolution in 3D (cells per wavelength in the densest material), coarser than 2D's 20.
CELLS_PER_WAVELENGTH_3D = 12.0
#: Grids up to this many unknowns are solved directly (SuperLU) on the CPU when solver="auto". SuperLU's
#: fill-in on 3D curl-curl matrices is severe (minutes already at ~2e4 unknowns), so this is kept tiny.
DIRECT_LIMIT = 3_000


def _max_index(problem: DeviceProblem, wl: np.ndarray) -> float:
    mats = [problem.background] + [s.material for s in problem.structures]
    return float(max(np.abs(np.real(m.n(wl))).max() for m in mats))


class FDFD3D:
    """Frequency-domain solver for a 3D :class:`DeviceProblem` (full-vector Maxwell)."""

    def __init__(self, problem: DeviceProblem, opts):
        if problem.layout is not None:
            from ...layout import resolved_device

            problem = resolved_device(problem)
        self.problem = problem
        self.opts = opts
        self.wavelengths = wavelength_array(problem.wavelengths)
        cpw = opts.cells_per_wavelength if "cells_per_wavelength" in opts.model_fields_set else CELLS_PER_WAVELENGTH_3D
        dl = opts.dl_nm or float(self.wavelengths.min() / (cpw * _max_index(problem, self.wavelengths)))
        p = opts.pml_cells
        npml = (p, p, p) if isinstance(p, int) else (p[0], p[1], p[1]) if len(p) == 2 else tuple(p)
        self.grid: Grid3D = make_grid(problem, dl, npml)
        if self.grid.size > opts.max_cells:
            raise ValueError(f"grid {self.grid.shape} has {self.grid.size} cells > max_cells={opts.max_cells}; "
                             "increase dl_nm or max_cells")
        self.subsamples = opts.subsamples
        self.ports: list[PortGeom3D] = [port_geometry(p, self.grid) for p in problem.ports]
        names = [p.name for p in problem.ports]
        if len(set(names)) != len(names):
            raise ValueError("port names must be unique")
        self.outputs: list[tuple[int, int]] = [(k, m) for k, pg in enumerate(self.ports) for m in range(pg.num_modes)]
        self.labels: list[str] = [lab for pg in self.ports for lab in pg.labels()]
        self._frac: np.ndarray | None = None
        self._eps: dict[float, np.ndarray] = {}
        self._warm: dict = {}
        self.warnings: list[str] = []
        self.info: dict[str, Any] = {}

    # --- pieces ---------------------------------------------------------------

    def k0dl(self, lam: float) -> float:
        return 2 * math.pi / lam * self.grid.dl

    def base_eps(self, lam: float) -> np.ndarray:
        if lam not in self._eps:
            if self._frac is None:
                self._frac = node_fractions(self.problem, self.grid, self.subsamples)
            self._eps[lam] = permittivity(self.problem, self._frac, lam)
        return self._eps[lam]

    def stretch(self, lam: float):
        """([s_node per axis], [s_edge per axis]) for the PML at this wavelength."""
        n_bg = float(np.real(self.problem.background.n(np.array([lam]))[0]))
        k_ref = 2 * math.pi / lam * max(n_bg, 1.0)
        pairs = [stretch_factors(self.grid, a, k_ref, reflection=self.opts.pml_reflection) for a in range(3)]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def operator(self, xp, eps, lam: float, device=None, dtype=None) -> Maxwell3D:
        sn, se = self.stretch(lam)
        st = Stretch.build(xp, sn, se, device=device)
        if dtype is not None:
            st = Stretch(tuple(xp.astype(a, dtype) for a in st.node), tuple(xp.astype(a, dtype) for a in st.half))
            eps = xp.astype(eps, dtype)
        return Maxwell3D(xp, eps, self.k0dl(lam), st, grad_div=self.opts.grad_div)

    def modes(self, eps_np: np.ndarray, lam: float) -> list[list[VectorMode]]:
        import array_api_compat.numpy as xnp

        ec = component_eps(xnp, eps_np)
        sn, se = self.stretch(lam)
        half = [s[1:] for s in se]
        out = []
        for pg in self.ports:
            ms = port_modes(pg, ec, self.k0dl(lam), (sn, half))
            for m, md in enumerate(ms):
                if not md.guided:
                    msg = (f"port {pg.name!r} mode {m} at {lam:g} nm is not guided (neff={md.neff.real:.4f}) "
                           "or touches the port window edge; enlarge the port or check num_modes")
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
        import array_api_compat.numpy as xnp

        op = self.operator(xnp, eps_np, lam)
        cols = [mode_source(self.ports[k], modes[k][m], op.apply, self.grid.shape)[0]
                for k, m in (self.outputs[i] for i in which)]
        return np.stack(cols, axis=0)

    def _smatrix(self, xp, X, modes, which: list[int], device=None):
        """S (n_out, n_src); S = C A^-1 when every port mode is excited (as in 2D)."""
        rows_a, rows_c = [], []
        for k, m in self.outputs:
            pg, md = self.ports[k], modes[k][m]
            proj = xp.asarray(md.proj) if device is None else xp.asarray(md.proj, device=device)
            if X.dtype != proj.dtype:
                proj = xp.astype(proj, X.dtype)
            a, c = decompose(xp, pg, md, X, proj)
            rows_a.append(a)
            rows_c.append(c)
        C = xp.stack(rows_c, axis=0)
        if list(which) != list(range(len(self.outputs))):
            return C
        A = xp.stack(rows_a, axis=0)
        return xp.linalg.solve(A.T, C.T).T

    def method(self, on_gpu: bool) -> str:
        s = self.opts.solver
        if s in ("bicgstab", "direct"):
            return s
        if s in ("block", "scipy"):
            raise ValueError(f"solver={s!r} is 2D-only; in 3D use 'auto', 'bicgstab' or 'direct'")
        return "direct" if (not on_gpu and 3 * self.grid.size <= DIRECT_LIMIT) else "bicgstab"

    def solver_kw(self, on_gpu: bool) -> dict[str, Any]:
        return {"method": self.method(on_gpu), "tol": self.opts.tol, "maxiter": self.opts.max_iterations,
                "precondition": self.opts.jacobi, "info": self.info}

    # --- solves ---------------------------------------------------------------

    def solve(self, device=None) -> tuple[np.ndarray, dict[str, Any]]:
        """Full S-matrix (n_wl, n_out, n_in) with the device's array namespace (no gradients)."""
        t0 = time.perf_counter()
        xp = device.namespace() if device is not None else None
        if xp is None:
            import array_api_compat.numpy as xp
        nat = device.native_device() if device is not None and device.framework != "numpy" else None
        on_gpu = device is not None and device.is_gpu
        S = []
        which = list(range(len(self.outputs)))
        dtype = getattr(xp, "complex64") if self.opts.precision == "single" else None
        for i, lam in enumerate(self.wavelengths):
            report("fdfd_solve", 0.95 * i / len(self.wavelengths), wavelength_nm=float(lam))
            lam = float(lam)
            eps = self.base_eps(lam)
            modes = self.modes(eps, lam)
            B = self._sources(eps, lam, modes, which)
            conv = (lambda a: xp.asarray(a)) if nat is None else (lambda a: xp.asarray(a, device=nat))
            op = self.operator(xp, conv(eps), lam, device=nat, dtype=dtype)
            Bx = conv(B) if dtype is None else xp.astype(conv(B), dtype)
            X = linsolve.Solver(op, **self.solver_kw(on_gpu)).solve(Bx)
            S.append(np.asarray(_to_numpy(self._smatrix(xp, X, modes, which, device=nat)), complex))
        self.info["runtime_s"] = time.perf_counter() - t0
        self.info["factor"] = self.method(on_gpu)
        self._collect_warnings()
        return np.asarray(S), self.info

    def _collect_warnings(self):
        for w in self.info.pop("warnings", []):
            if w not in self.warnings:
                self.warnings.append(w)

    def differentiable(self, eps, device=None, sources: list[int] | None = None, wavelengths=None,
                       info: dict | None = None, with_fields: bool = True) -> dict[str, Any]:
        """Torch-differentiable solve; eps (Nx, Ny, Nz) or (n_wl, Nx, Ny, Nz) node permittivity.

        Returns {"s": (n_wl, n_out, n_src), "fields": (n_wl, n_src, 3, Nx, Ny, Nz) | None, "outputs",
        "inputs"}. Forward and adjoint iterations are warm-started from the previous call
        (successive optimisation iterates change little), which cuts their cost severalfold.
        """
        import array_api_compat.torch as xp
        import torch

        tdev = torch.device(device) if device is not None and not isinstance(device, torch.device) else device
        if tdev is None:
            tdev = eps.device
        wls = self.wavelengths if wavelengths is None else np.asarray(wavelengths, float)
        which = list(range(len(self.outputs))) if sources is None else list(sources)
        dtype = torch.complex64 if self.opts.precision == "single" else torch.complex128
        kw = self.solver_kw(tdev.type != "cpu")
        eps = eps.to(tdev)
        S, F = [], []
        for w, lam in enumerate(wls):
            lam = float(lam)
            e = eps[w] if eps.ndim == 4 else eps
            e = e.to(dtype)
            e_np = e.detach().cpu().numpy().astype(complex)
            modes = self.modes(e_np, lam)
            B = torch.as_tensor(self._sources(e_np, lam, modes, which), device=tdev).to(dtype)
            sn, se = self.stretch(lam)
            st = Stretch.build(xp, sn, se, device=tdev)
            st = Stretch(tuple(a.to(dtype) for a in st.node), tuple(a.to(dtype) for a in st.half))
            k0dl, gd = self.k0dl(lam), self.opts.grad_div

            def make_op(ee, st=st, k0dl=k0dl, gd=gd):
                return Maxwell3D(xp, ee, k0dl, st, grad_div=gd)

            warm = self._warm.setdefault((lam, tuple(which), str(tdev)), {}) if self.opts.warm_start else None
            X = linsolve.solve(e, B, make_op, kw, warm)
            S.append(self._smatrix(xp, X, modes, which, device=tdev))
            if with_fields:
                F.append(X)
        self._collect_warnings()
        if info is not None:
            info.update({k: v for k, v in self.info.items() if k != "solves"})
        return {"s": torch.stack(S), "fields": torch.stack(F) if with_fields else None,
                "outputs": list(self.labels), "inputs": [self.labels[i] for i in which]}

    # --- cost model -------------------------------------------------------------

    def estimate(self, device=None) -> dict[str, Any]:
        nx, ny, nz = self.grid.shape
        n = 3 * nx * ny * nz
        on_gpu = device is not None and getattr(device, "is_gpu", False)
        method = self.method(on_gpu)
        b = 8 if self.opts.precision == "single" else 16
        n_src = len(self.outputs)
        # BiCGStab keeps ~9 vectors per right-hand side; a matvec makes ~12 component-sized temporaries.
        mem_it = b * n * (9 * n_src + 6) + 4 * 16 * nx * ny * nz
        # Nested-dissection-like fill of a 3D 7-point-stencil system: ~ 60 n^(4/3) nonzeros.
        mem_direct = int(16 * 60 * n ** (4 / 3))
        # Throughput model: an iteration streams ~130 arrays of n entries (two matvecs + vector updates).
        bw = 3e11 if on_gpu else 1.5e10
        t_iter = 130 * n * b / bw
        lam_med = float(self.wavelengths.min()) / _max_index(self.problem, self.wavelengths)
        span = max(nx, ny, nz) * self.grid.dl / lam_med
        iters = int(min(self.opts.max_iterations, max(200, 60 * span)))
        return {"dimension": 3, "grid": [nx, ny, nz], "dl_nm": self.grid.dl, "unknowns": n,
                "pml_cells": list(self.grid.npml), "sources": n_src, "wavelengths": len(self.wavelengths),
                "solver": method, "precision": self.opts.precision,
                "memory_bytes_iterative": int(mem_it), "memory_bytes_direct": mem_direct,
                "memory_bytes": int(mem_direct if method == "direct" else mem_it),
                "iterations_estimate": iters, "seconds_per_iteration_estimate": t_iter,
                "seconds_per_solve_estimate": (iters * t_iter) if method != "direct" else None,
                "fits": self.grid.size <= self.opts.max_cells,
                "note": "iteration counts are a rough cold-start guess; warm starts during optimisation "
                        "typically need several times fewer"}


def _to_numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
