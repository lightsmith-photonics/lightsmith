"""Native finite-difference beam propagation method (BPM) for long, slowly varying devices.

The transverse magnetic field is marched along the axis normal to the ports,
from one port plane to the other, with the one-way equation
dH_t/dw = i k0 sqrt(N) H_t, where N = P Q is built from the FDE engine's
Yee-grid operators (so FDE/EME port modes are exact eigenvectors of the
propagator in uniform sections and mode conventions match EME). sqrt is
replaced by a Padé (m, m) approximant (wide angle; m = 0 is the paraxial
Fresnel equation), steps are Crank-Nicolson, transverse walls are
stretched-coordinate PMLs, and the reference index adapts to the field in
every slice. H (continuous across every interface) is carried across slice
boundaries, and each input's power is kept continuous there (energy-conserving
BPM), which makes tapers reciprocal and adiabatic to ~1%.

- ``dimension=2``: slab BPM in the xy plane, 'te' (in-plane E, default, as EME)
  or 'tm' (E along z); exact scalar operators.
- ``dimension=3``: semi-vectorial (quasi-TE: Hy, quasi-TM: Hx) or
  full-vectorial (Hx, Hy coupled; default) on the 2D cross-section.

BPM is unidirectional: reflections and back-scattering are neglected. One run
from the min port plane fills S(max <- min); a second run from the max plane
fills S(min <- max); S entries between ports on the same plane are set to 0.
Use EME (bidirectional, exact for piecewise-uniform devices) or FDFD/FDTD when
reflections or abrupt high-contrast junctions matter: at an abrupt junction BPM
transmits by overlap, an error of order the mismatch, and short converging
(wide -> narrow) high-contrast tapers come out low. The ``bend_radius_nm``
option propagates along a circular arc (conformal transformation, as FDE bend
modes). Solves are block-tridiagonal parallel cyclic reduction in the array
API, so the same code runs on NumPy (CPU) and torch (CUDA).
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..problems import DeviceProblem, options_for, wavelength_array
from ..results import SParameterResult
from .base import Backend, register
from .eme import _axis_and_planes, _resolved

UNIDIRECTIONAL = ("BPM is unidirectional: reflections are neglected and S entries between ports on the same "
                  "plane are set to 0")


class BPMOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector: Literal["auto", "semi", "full"] = Field(
        "auto", description="3D only: semi-vectorial (one H component) or full-vectorial (Hx, Hy). auto = full.")
    polarization: Literal["auto", "te", "tm", "any"] = Field(
        "auto", description="2D: 'te' (in-plane E, default) or 'tm'. 3D semi: dominant Ex ('te', default) or Ey "
        "('tm'). 3D full: filters port modes by te_fraction ('any' by default).")
    grid_spacing_nm: float | tuple[float, float] | None = Field(
        None, description="Cross-section grid step. Default from points_per_wavelength and cells_per_feature.")
    points_per_wavelength: float | None = Field(None, gt=0, description="Default 40 in 2D, 15 in 3D.")
    cells_per_feature: float = Field(6, gt=0)
    subpixel: Literal["anisotropic", "volume", "none"] = "anisotropic"
    subsamples: int = Field(32, ge=1)
    pml_cells: int = Field(15, ge=1)
    pml_reflection: float = Field(1e-8, gt=0, lt=1)
    step_nm: float | None = Field(None, gt=0, description="Max propagation step. Default wavelength / 10.")
    pade_order: int = Field(
        2, ge=0, le=6, description="0 = paraxial, m = Padé (m, m) wide-angle; cost is m solves per step. (1, 1) "
        "suffices for straight guides and gentle tapers; tilted guides, branches and bends need (2, 2).")
    reference_index: Literal["adaptive"] | float = Field(
        "adaptive", description="'adaptive' (Rayleigh quotient of the field, per slice) or a fixed value.")
    taper_grid_cells_per_slice: float = Field(
        1.0, gt=0, description="Staircase continuously varying segments so outlines move about this many grid "
        "cells per slice.")
    max_taper_slices: int = Field(400, ge=1, description="Upper bound of slices per varying segment.")
    power_continuity: bool = Field(
        True, description="Rescale the field at slice boundaries so each input's power is continuous (energy-"
        "conserving BPM); False keeps plain H continuity, which conserves an n-weighted norm instead (a taper's "
        "transmission is then off by ~sqrt(n_out / n_in)).")
    directions: Literal["both", "forward"] = Field(
        "both", description="'forward' only propagates from the min plane and fills S(min <- max) by reciprocity.")
    bend_radius_nm: float | None = Field(
        None, description="Propagate along an arc of this radius (bend about the axis normal to the device plane; "
        "positive = centre of curvature towards -u). The device is given unrolled.")
    precision: Literal["auto", "single", "double"] = Field("auto", description="auto: single on GPU, double on CPU.")
    max_grid_cells: int = Field(400_000, ge=1)
    gpu_min_unknowns: int = Field(
        2000, ge=0, description="Cross-sections with fewer unknowns run on the CPU even when a GPU is selected "
        "(2D slabs: per-step kernel launches cost more than the arithmetic). 0 = always use the given device.")


@register
class NativeBPM(Backend):
    name = "lightsmith-bpm"
    method = "bpm"
    options_model = BPMOptions
    license = "MIT"
    citation = ("Hadley, Opt. Lett. 17, 1426 (1992) (wide-angle Padé BPM); Huang & Xu, IEEE J. Quantum "
                "Electron. 29, 2639 (1993) (vectorial BPM)")
    gpu = True
    problem_kinds = ("device",)

    @classmethod
    def available(cls):
        try:
            import scipy.sparse.linalg  # noqa: F401
        except ImportError:
            return False, "needs scipy (FDE port modes); install scipy"
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
        return True, ""

    @classmethod
    def _options(cls, problem) -> BPMOptions:
        return BPMOptions.model_validate(options_for(problem, cls.name, cls.method))

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
        mats = [problem.background, *(st.material for st in problem.structures)]
        n_all = np.array([np.real(m.n(wl)) for m in mats])
        if opts.grid_spacing_nm is not None:
            g = opts.grid_spacing_nm
            spacing = (float(g), float(g)) if np.isscalar(g) else (float(g[0]), float(g[1]))
        else:
            ppw = opts.points_per_wavelength or (40 if dim == 2 else 15)
            feats = []
            for st in problem.structures:
                g_ = st.geometry
                if g_.kind == "box":
                    feats += [g_.size_nm[iu]] + ([g_.size_nm[iv]] if dim == 3 else [])
                elif g_.kind == "prism" and dim == 3 and 2 in (iu, iv):
                    feats.append(g_.z_max_nm - g_.z_min_nm)
            feats = [f for f in feats if f > 0]
            d = default_spacing(float(wl.min()), float(n_all.max()), ppw, min(feats) if feats else None,
                                opts.cells_per_feature)
            spacing = (d, d)
        grid = make_grid(center, size, spacing, (None, None), opts.pml_cells)
        pol = opts.polarization
        vector = opts.vector if dim == 3 else "semi"
        if vector == "auto":
            vector = "full"
        if pol == "auto":
            pol = "any" if vector == "full" else "te"
        if vector == "semi" and pol == "any":
            raise ValueError("semi-vectorial BPM needs polarization 'te' or 'tm'")
        comps = (0, 1) if vector == "full" else ((1,) if pol == "te" else (0,))  # H components: te -> hy
        step = opts.step_nm or float(wl.min()) / 10
        return dict(opts=opts, axis=axis, lo=lo, hi=hi, axes=(iu, iv, sv), grid=grid, wl=wl, comps=comps,
                    vector=vector, pol=None if pol == "any" else pol, step=step,
                    n_range=(float(n_all.min()), float(n_all.max())))

    @classmethod
    def _bytes_per_value(cls, opts, device: Device) -> int:
        single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu)
        return 8 if single else 16

    @classmethod
    def estimate(cls, problem, device: Device):
        st = cls._setup(_resolved(problem))
        g, opts = st["grid"], st["opts"]
        nc = len(st["comps"])
        n_lines, short = max(g.nx, g.ny), min(g.nx, g.ny)
        nb = nc * short
        levels = max(1, math.ceil(math.log2(max(n_lines, 2))))
        bpv = cls._bytes_per_value(opts, device)
        factor = (2 * levels + 1) * n_lines * nb * nb * bpv * max(1, opts.pade_order)
        blocks = 3 * n_lines * nb * nb * bpv
        n_in = sum(p.num_modes for p in problem.ports)
        steps = math.ceil((st["hi"] - st["lo"]) / st["step"])
        runs = 1 if opts.directions == "forward" else 2
        return {"axis": "xyz"[st["axis"]], "length_nm": st["hi"] - st["lo"], "cross_section_grid": [g.nx, g.ny],
                "grid_spacing_nm": [g.dx, g.dy if g.ny > 1 else None], "vector": st["vector"],
                "components": ["Hx", "Hy"] if nc == 2 else ["Hy" if st["comps"] == (1,) else "Hx"],
                "unknowns": nc * g.size, "block_size": nb, "lines": n_lines, "step_nm": st["step"],
                "min_steps_per_run": steps, "runs_per_wavelength": runs,
                "solves_per_step": max(1, opts.pade_order), "pade_order": opts.pade_order,
                "precision": "single" if bpv == 8 else "double",
                "memory_bytes": int(factor + blocks + 4 * n_lines * nb * n_in * bpv),
                "unidirectional": True, "fits": g.size <= opts.max_grid_cells}

    @classmethod
    def solve(cls, problem: DeviceProblem, device: Device) -> SParameterResult:
        from ._bpm.engine import RunStats, Settings, port_fields, propagate
        from ._bpm.operator import Layout
        from ._eme import port_mode_set, slice_device

        t0 = time.perf_counter()
        problem = _resolved(problem)
        st = cls._setup(problem)
        opts, grid, axis, lo, hi = st["opts"], st["grid"], st["axis"], st["lo"], st["hi"]
        iu, iv, sv = st["axes"]
        if grid.size > opts.max_grid_cells:
            raise ValueError(f"cross-section grid {grid.nx}x{grid.ny} exceeds max_grid_cells={opts.max_grid_cells}; "
                             "increase grid_spacing_nm or raise max_grid_cells")
        dim, comps = problem.dimension, st["comps"]
        cells = slice_device(problem.structures, problem.background, axis, lo, hi, grid, dim,
                             problem.domain_center_nm[2], None, None, opts.subsamples, opts.subpixel,
                             cells_per_step=opts.taper_grid_cells_per_slice,
                             taper_cells_range=(1, opts.max_taper_slices))
        device_note = None
        if device.is_gpu and len(comps) * grid.size < opts.gpu_min_unknowns:
            device_note = (f"ran on the CPU: {len(comps) * grid.size} unknowns per cross-section is below "
                           f"gpu_min_unknowns={opts.gpu_min_unknowns}")
            device = Device("numpy", "cpu")
        xp = device.namespace()
        cdt = "complex64" if cls._bytes_per_value(opts, device) == 8 else "complex128"

        def asarray(a):
            return device.asarray(np.asarray(a, dtype=complex), dtype=cdt)

        settings = Settings(comps=comps, step_nm=st["step"], pade_order=opts.pade_order,
                            reference_index=None if opts.reference_index == "adaptive" else float(opts.reference_index),
                            power_continuity=opts.power_continuity,
                            bend_radius_nm=opts.bend_radius_nm, bend_center_nm=grid.x0 + grid.nx * grid.dx / 2,
                            subpixel=opts.subpixel, pml_reflection=opts.pml_reflection)
        n_lo, n_hi = st["n_range"]
        n_bounds = (0.5 * n_lo, 1.5 * n_hi)

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
        warnings: list[str] = [UNIDIRECTIONAL]
        if st["vector"] == "semi" and dim == 3:
            warnings.append("semi-vectorial: polarisation coupling is neglected and port modes are projected on "
                            f"{'Hy' if comps == (1,) else 'Hx'} only")
        stats = RunStats()
        s_all, power = [], []
        for lam in st["wl"]:
            lam = float(lam)

            def side(ports, cell):
                launch, meas, labels = [], [], []
                for p in ports:
                    a0, a1 = window(p)
                    ms = port_mode_set(cell.section, grid, a0, a1, lam, p.num_modes, st["pol"], opts.subpixel,
                                       opts.subsamples)
                    if len(ms.neff) < p.num_modes:
                        raise ValueError(f"port {p.name!r} supports only {len(ms.neff)} modes at {lam:g} nm")
                    l_, m_, _ = port_fields(ms.truncate(p.num_modes), comps, grid.shape)
                    launch.append(l_)
                    meas.append(m_)
                    labels += names[p.name]
                return np.concatenate(launch), np.concatenate(meas), labels

            sl, sr = side(left, cells[0]), side(right, cells[-1])
            labels = sl[2] + sr[2]
            s = np.zeros((len(labels), len(labels)), complex)
            nl = len(sl[2])
            runs = [(sl, sr, cells, slice(nl, None), slice(0, nl))]
            if opts.directions == "both":
                runs.append((sr, sl, cells[::-1], slice(0, nl), slice(nl, None)))
            for src, dst, seq, rows, cols in runs:
                lay = Layout(comps, grid.nx, grid.ny)
                phi = lay.to_lines(xp, asarray(src[0]))
                phi, theta = propagate(seq, grid, lam, phi, settings, xp, asarray, stats, n_bounds)
                out = Device.to_numpy(lay.from_lines(xp, phi)).astype(complex)  # (k, C, nx, ny)
                amp = np.einsum("pcxy,kcxy->pk", dst[1], out) * np.exp(1j * theta)
                s[rows, cols] = amp
            if opts.directions == "forward":
                s[:nl, nl:] = s[nl:, :nl].T
            idx = [labels.index(n) for n in order]
            s = s[np.ix_(idx, idx)]
            col_power = np.sum(np.abs(s) ** 2, axis=0)
            power.append(col_power.tolist())
            if np.any(col_power > 1.02):
                warnings.append(f"sum |S|^2 up to {col_power.max():.3f} > 1 at {lam:g} nm: refine the grid or the "
                                "step, or raise pade_order")
            s_all.append(s)
        s_all = np.asarray(s_all)
        provenance = {
            **device.describe(), "device_note": device_note, "axis": "xyz"[axis], "port_planes_nm": [lo, hi], "slices": len(cells),
            "slice_bounds_nm": [[c.start, c.stop] for c in cells] if len(cells) <= 200 else None,
            "cross_section_grid": grid.describe(), "vector": st["vector"],
            "components": ["Hx", "Hy"] if len(comps) == 2 else ["Hy" if comps == (1,) else "Hx"],
            "polarization": st["pol"] or "any", "pade_order": opts.pade_order,
"max_step_nm": st["step"], "steps": stats.steps,
            "factorizations": stats.factorizations, "peak_factor_bytes": stats.peak_factor_bytes,
            "precision": "single" if cdt == "complex64" else "double", "unidirectional": True,
            "directions": opts.directions, "reference_index_range": [min(stats.n_ref), max(stats.n_ref)],
            "transmitted_power_per_input": power, "runtime_s": time.perf_counter() - t0,
        }
        return SParameterResult(method=cls.method, backend=cls.name, wavelengths_nm=st["wl"].tolist(), ports=order,
                                s_real=s_all.real.tolist(), s_imag=s_all.imag.tolist(), provenance=provenance,
                                warnings=warnings)
