"""Native eigenmode expansion (EME) for devices that vary along one propagation axis.

The device between its two port planes is sliced into piecewise-uniform
cells (tapers and curved outlines are staircased), each cell's modes come
from the native FDE engine, interface S-matrices from mode overlaps are
cascaded with Redheffer star products, and port modes (FDE at each port
plane, within the port window) project the result onto the ports.

``dimension=3`` uses full-vector 2D cross-sections; ``dimension=2`` uses
slab (1D) modes of the xy-plane device, invariant along z, with the
in-plane ('te', default) or out-of-plane ('tm') E polarisation.
EME is exact for piecewise-uniform devices given enough modes; it suits
long, mostly-paraxial devices (tapers, MMIs, couplers, crossings) and
needs ports on two opposite planes normal to the same axis.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..problems import DeviceProblem, options_for, wavelength_array
from ..results import SParameterResult
from .base import Backend, register


class EMEOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_modes: int | None = Field(None, ge=1, description="Modes per cell. Default 30 in 2D, 10 in 3D.")
    grid_spacing_nm: float | tuple[float, float] | None = Field(
        None, description="Cross-section grid step. Default from points_per_wavelength and cells_per_feature.")
    points_per_wavelength: float | None = Field(None, gt=0, description="Default 40 in 2D, 15 in 3D.")
    cells_per_feature: float = Field(6, gt=0)
    subpixel: Literal["anisotropic", "volume", "none"] = "anisotropic"
    subsamples: int = Field(32, ge=1)
    boundary: Literal["pec", "pml"] = Field("pec", description="Cross-section walls (PEC box modes, or PML).")
    pml_cells: int = Field(10, ge=1)
    pml_reflection: float = Field(1e-6, gt=0, lt=1)
    polarization: Literal["auto", "te", "tm", "any"] = Field(
        "auto", description="auto: 'te' (in-plane E) in 2D, 'any' in 3D. In 3D, 'te'/'tm' filter by te_fraction "
        "(inexact: the basis loses completeness).")
    taper_cells: int | None = Field(
        None, ge=1, description="Cells per continuously varying segment (tapers, curves). Default: enough for the "
        "outline to move ~3 grid cells per step, clipped to 10-60.")
    max_cell_length_nm: float | None = Field(None, gt=0, description="Also split segments longer than this.")
    max_grid_cells: int = Field(400_000, ge=1)
    cache: bool = Field(True, description="Reuse cell modes across solves (byte-limited LRU, 256 MB).")


def _axis_and_planes(problem: DeviceProblem) -> tuple[int, float, float]:
    axes = {p.size_nm.index(0) for p in problem.ports}
    if len(axes) != 1:
        raise ValueError("EME needs every port normal to the same axis (ports on two opposite faces); found normals "
                         + ", ".join(sorted({'xyz'[a] for a in axes})))
    axis = axes.pop()
    if problem.dimension == 2 and axis == 2:
        raise ValueError("in a 2D device ports must face along x or y")
    pos = sorted({round(p.center_nm[axis], 6) for p in problem.ports})
    if len(pos) != 2:
        raise ValueError("EME needs ports on exactly two opposite planes along the "
                         f"{'xyz'[axis]} axis; found port planes at {pos}")
    lo, hi = pos
    for p in problem.ports:
        at_lo = round(p.center_nm[axis], 6) == lo
        if (p.direction == "+") != at_lo:
            raise ValueError(f"port {p.name!r} must point into the device ({'+' if at_lo else '-'} at the "
                             f"{'min' if at_lo else 'max'} plane)")
    return axis, lo, hi


def _resolved(problem: DeviceProblem) -> DeviceProblem:
    if problem.layout is not None:
        from ..layout import resolved_device

        return resolved_device(problem)
    return problem


@register
class NativeEME(Backend):
    name = "omniscatter-eme"
    method = "eme"
    license = "MIT"
    citation = "Bienstman & Baets, Opt. Quantum Electron. 33, 327 (2001); Gallagher & Felici, Proc. SPIE 4987 (2003)"
    problem_kinds = ("device",)

    @classmethod
    def available(cls):
        try:
            import scipy.sparse.linalg  # noqa: F401
        except ImportError:
            return False, "needs scipy (sparse LU + ARPACK); install scipy"
        return True, ""

    @classmethod
    def supports(cls, problem):
        if not isinstance(problem, DeviceProblem):
            return False, "solves device problems only"
        try:
            _axis_and_planes(problem)
            cls._options(problem)
        except ValueError as e:
            return False, str(e)
        if problem.layout is not None:
            try:
                from ..layout import resolved_device  # noqa: F401
            except ImportError:
                return False, "GDS layouts need omniscatter.layout"
        return True, ""

    @classmethod
    def _options(cls, problem) -> EMEOptions:
        return EMEOptions.model_validate(options_for(problem, cls.name, cls.method))

    @classmethod
    def _setup(cls, problem: DeviceProblem):
        from ._fde import default_spacing, make_grid, plane_axes

        opts = cls._options(problem)
        axis, lo, hi = _axis_and_planes(problem)
        dim = problem.dimension
        iu, iv, sv = plane_axes(axis, 2 if dim == 2 else None)
        c, s = problem.domain_center_nm, problem.domain_size_nm
        center = (c[iu], sv * c[iv] if dim == 3 else 0.0)
        size = (s[iu], s[iv] if dim == 3 else 0.0)
        wl = wavelength_array(problem.wavelengths)
        if opts.grid_spacing_nm is not None:
            g = opts.grid_spacing_nm
            spacing = (float(g), float(g)) if np.isscalar(g) else (float(g[0]), float(g[1]))
        else:
            mats = [problem.background, *(st.material for st in problem.structures)]
            n_max = float(max(np.real(m.n(wl)).max() for m in mats))
            ppw = opts.points_per_wavelength or (40 if dim == 2 else 15)
            feats = []
            for st in problem.structures:
                g_ = st.geometry
                if g_.kind == "box":
                    feats += [g_.size_nm[iu]] + ([g_.size_nm[iv]] if dim == 3 else [])
                elif g_.kind == "prism" and dim == 3 and 2 in (iu, iv):
                    feats.append(g_.z_max_nm - g_.z_min_nm)
            feats = [f for f in feats if f > 0]
            d = default_spacing(float(wl.min()), n_max, ppw, min(feats) if feats else None, opts.cells_per_feature)
            spacing = (d, d)
        grid = make_grid(center, size, spacing, (None, None), opts.pml_cells if opts.boundary == "pml" else 0)
        pol = opts.polarization
        if pol == "auto":
            pol = "te" if dim == 2 else "any"
        pol = None if pol == "any" else pol
        num_modes = opts.num_modes or (30 if dim == 2 else 10)
        return opts, axis, lo, hi, (iu, iv, sv), grid, pol, num_modes

    @classmethod
    def estimate(cls, problem, device: Device):
        opts, axis, lo, hi, _, grid, pol, num_modes = cls._setup(_resolved(problem))
        return {"axis": "xyz"[axis], "length_nm": hi - lo, "cross_section_grid": [grid.nx, grid.ny],
                "grid_spacing_nm": [grid.dx, grid.dy if grid.ny > 1 else None], "modes_per_cell": num_modes,
                "unknowns_per_cell": 2 * grid.size, "memory_bytes": int(2 * grid.size * 180 * 16),
                "fits": grid.size <= opts.max_grid_cells}

    @classmethod
    def solve(cls, problem: DeviceProblem, device: Device) -> SParameterResult:
        from ._eme import SMatrix, cell_modes, coupling, interface, port_mode_set, slice_device

        t0 = time.perf_counter()
        problem = _resolved(problem)
        opts, axis, lo, hi, (iu, iv, sv), grid, pol, num_modes = cls._setup(problem)
        if grid.size > opts.max_grid_cells:
            raise ValueError(f"cross-section grid {grid.nx}x{grid.ny} exceeds max_grid_cells={opts.max_grid_cells}; "
                             "increase grid_spacing_nm or raise max_grid_cells")
        dim = problem.dimension
        cells = slice_device(problem.structures, problem.background, axis, lo, hi, grid, dim,
                             problem.domain_center_nm[2], opts.taper_cells, opts.max_cell_length_nm,
                             opts.subsamples, opts.subpixel)
        wl = wavelength_array(problem.wavelengths)

        def window(port):
            c, s = port.center_nm, port.size_nm
            if dim == 2:
                return (c[iu] - s[iu] / 2, 0.0), (c[iu] + s[iu] / 2, 0.0)
            v0, v1 = sorted((sv * (c[iv] - s[iv] / 2), sv * (c[iv] + s[iv] / 2)))
            return (c[iu] - s[iu] / 2, v0), (c[iu] + s[iu] / 2, v1)

        left = [p for p in problem.ports if round(p.center_nm[axis], 6) == lo]
        right = [p for p in problem.ports if round(p.center_nm[axis], 6) != lo]
        names = {p.name: ([p.name] if p.num_modes == 1 else [f"{p.name}@{j}" for j in range(p.num_modes)])
                 for p in problem.ports}
        order = [n for p in problem.ports for n in names[p.name]]
        warnings: list[str] = []
        s_all, n_used, power = [], [], []
        for lam in wl:
            lam = float(lam)
            sets = {}
            for c in cells:
                if c.key not in sets:
                    sets[c.key] = cell_modes(c, grid, lam, num_modes, pol, opts.subpixel, opts.pml_reflection,
                                             opts.cache)
            m = min(len(ms.neff) for ms in sets.values())
            if m < num_modes:
                warnings.append(f"only {m} modes per cell at {lam:g} nm (asked {num_modes})")
            sets = {k: v.truncate(m) for k, v in sets.items()}
            n_used.append(m)
            S = SMatrix.propagation(sets[cells[0].key].neff, cells[0].length, lam)
            joints: dict = {}
            for a, b in zip(cells[:-1], cells[1:]):
                jk = (a.key, b.key)
                if jk not in joints:
                    joints[jk] = interface(sets[a.key], sets[b.key])
                S = S.star(joints[jk]).star(SMatrix.propagation(sets[b.key].neff, b.length, lam))

            def side_coupling(ports, cell):
                if not ports:
                    return np.zeros((0, m), dtype=complex), []
                rows, labels = [], []
                for p in ports:
                    a0, a1 = window(p)
                    ps = port_mode_set(cell.section, grid, a0, a1, lam, p.num_modes, pol, opts.subpixel,
                                       opts.subsamples)
                    if len(ps.neff) < p.num_modes:
                        raise ValueError(f"port {p.name!r} supports only {len(ps.neff)} modes at {lam:g} nm")
                    rows.append(coupling(ps.truncate(p.num_modes), sets[cell.key]))
                    labels += names[p.name]
                return np.vstack(rows), labels

            cl, ll = side_coupling(left, cells[0])
            cr, lr = side_coupling(right, cells[-1])
            blocks = np.block([[cl @ S.s11 @ cl.T, cl @ S.s12 @ cr.T], [cr @ S.s21 @ cl.T, cr @ S.s22 @ cr.T]])
            labels = ll + lr
            idx = [labels.index(n) for n in order]
            s = blocks[np.ix_(idx, idx)]
            col_power = np.sum(np.abs(s) ** 2, axis=0)
            power.append(col_power.tolist())
            if np.any(col_power > 1.05):
                warnings.append(f"sum |S|^2 up to {col_power.max():.3f} > 1 at {lam:g} nm: increase num_modes "
                                "or refine the grid")
            s_all.append(s)
        s_all = np.asarray(s_all)
        provenance = {
            **device.describe(), "axis": "xyz"[axis], "port_planes_nm": [lo, hi], "cells": len(cells),
            "unique_cross_sections": len({c.key for c in cells}),
            "cell_bounds_nm": [[c.start, c.stop] for c in cells] if len(cells) <= 200 else None,
            "cross_section_grid": grid.describe(), "modes_per_cell": n_used,
            "polarization": pol or "any", "transmitted_power_per_input": power,
            "runtime_s": time.perf_counter() - t0,
        }
        return SParameterResult(method=cls.method, backend=cls.name, wavelengths_nm=wl.tolist(), ports=order,
                                s_real=s_all.real.tolist(), s_imag=s_all.imag.tolist(), provenance=provenance,
                                warnings=warnings)
