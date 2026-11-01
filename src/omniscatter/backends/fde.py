"""Native full-vector finite-difference eigenmode (FDE) solver for waveguide cross-sections.

A Yee-grid finite-difference discretisation of the transverse E field
(Zhu & Brown, Opt. Express 10, 853 (2002); equivalent to Fallahkhair, Li &
Murphy, J. Lightwave Technol. 26, 1423 (2008)) with:

- subpixel permittivity (adaptive supersampling of cut cells; harmonic/arithmetic
  anisotropic averaging),
- PEC or PML outer walls, PEC/PMC mirror-symmetry planes,
- bends through the exact transformation-optics form of the conformal map
  (`bend_radius_nm`, bend about the y axis, centre of curvature at x_c - R),
- shift-invert Arnoldi (ARPACK) around `target_neff` or the highest index,
- group index from the discrete perturbation formula (exact for the lossless
  discrete system; includes material dispersion) or a wavelength derivative,
- slab (1D) cross-sections when ``window_size_nm[1] == 0``.

The engine lives in ``omniscatter.backends._fde`` and its ``solve_modes`` /
``port_modes`` are reused by the EME backend for device ports.
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..problems import Circle, Ellipse, ModeProblem, Polygon, Rectangle, options_for, wavelength_array
from ..results import ModeData, ModeField, ModeResult
from .base import Backend, register

Sym = Literal["none", "pec", "pmc"]


class FDEOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_spacing_nm: float | tuple[float, float] | None = Field(
        None, description="Grid step (x, y). Default: points_per_wavelength in the densest material and "
        "cells_per_feature across the smallest structure.")
    points_per_wavelength: float = Field(20, gt=0)
    cells_per_feature: float = Field(8, gt=0)
    subpixel: Literal["anisotropic", "volume", "none"] = Field(
        "anisotropic", description="anisotropic: harmonic mean along each E component, arithmetic across "
        "(best for interfaces); volume: fill-fraction average; none: staircase.")
    subsamples: int = Field(32, ge=1, description="Samples per axis in cells cut by an interface "
                            "(16x as many along x for slabs).")
    boundary: Literal["auto", "pec", "pml"] = Field("auto", description="auto = PML for bends, PEC otherwise.")
    pml_cells: int = Field(12, ge=1)
    pml_reflection: float = Field(1e-6, gt=0, lt=1, description="Normal-incidence reflection the PML grading targets.")
    symmetry: tuple[Sym, Sym] = Field(
        ("none", "none"), description="Mirror planes x = window centre and y = window centre. 'pec' zeroes tangential "
        "E on the plane (e.g. x-plane for TE0 of a strip, whose Ex is even), 'pmc' zeroes tangential H.")
    polarization: Literal["any", "te", "tm"] = Field(
        "any", description="Keep only modes with te_fraction > 0.5 ('te', dominant Ex) or < 0.5 ('tm').")
    group_index: Literal["auto", "perturbation", "finite_difference", "none"] = Field(
        "auto", description="auto: perturbation formula without PML, wavelength finite difference with PML.")
    track_modes: bool = Field(
        False, description="Order modes at each wavelength by overlap with the previous wavelength instead of by neff.")
    bend_reference_x_nm: float | None = Field(
        None, description="x where the bend radius is measured (centre of curvature at this x minus R). "
        "Default: the window centre.")
    max_cells: int = Field(1_000_000, ge=1, description="Refuse grids larger than this.")


def _min_feature(structures) -> float | None:
    sizes = []
    for s in structures:
        p = s.profile
        if isinstance(p, Rectangle):
            sizes.append(min(p.size_nm))
        elif isinstance(p, Circle):
            sizes.append(2 * p.radius_nm)
        elif isinstance(p, Ellipse):
            sizes.append(2 * min(p.semi_axes_nm))
        elif isinstance(p, Polygon):
            v = np.asarray(p.vertices_nm)
            sizes.append(float(np.min(v.max(axis=0) - v.min(axis=0))))
    sizes = [s for s in sizes if s > 0]
    return min(sizes) if sizes else None


def _n_max(problem: ModeProblem, wl: np.ndarray) -> float:
    mats = [problem.background, *(s.material for s in problem.structures)]
    return float(max(np.real(m.n(wl)).max() for m in mats))


def _spacing(problem: ModeProblem, opts: FDEOptions) -> tuple[float, float]:
    if opts.grid_spacing_nm is not None:
        g = opts.grid_spacing_nm
        return (float(g), float(g)) if np.isscalar(g) else (float(g[0]), float(g[1]))
    from ._fde import default_spacing

    wl = wavelength_array(problem.wavelengths)
    d = default_spacing(float(wl.min()), _n_max(problem, wl), opts.points_per_wavelength,
                        _min_feature(problem.structures), opts.cells_per_feature)
    return d, d


def _grid(problem: ModeProblem, opts: FDEOptions):
    from ._fde import make_grid

    sym = tuple(None if s == "none" else s for s in opts.symmetry)
    if problem.window_size_nm[1] <= 0 and sym[1]:
        sym = (sym[0], None)
    boundary = opts.boundary if opts.boundary != "auto" else ("pml" if problem.bend_radius_nm else "pec")
    if problem.bend_radius_nm and sym[0]:
        raise ValueError("symmetry about x is incompatible with a bend (bends are not mirror-symmetric in x)")
    return make_grid(problem.window_center_nm, problem.window_size_nm, _spacing(problem, opts), sym,
                     opts.pml_cells if boundary == "pml" else 0), boundary


def _field(mode) -> ModeField:
    x, y, comp = mode.collocated()
    p = mode.power() / (1e3 if mode.grid.invariant_y else 1e6)  # in µm² (µm per µm of y for slabs)
    scale = 1.0 / math.sqrt(p) if p > 1e-30 else 1.0  # ½ Re∫E x H* dA = 1
    return ModeField(x_nm=x.tolist(), y_nm=y.tolist(),
                     components={k: ((scale * v).real.tolist(), (scale * v).imag.tolist()) for k, v in comp.items()})


@register
class NativeFDE(Backend):
    name = "omniscatter-fde"
    method = "fde"
    license = "MIT"
    citation = ("Zhu & Brown, Opt. Express 10, 853 (2002); Fallahkhair, Li & Murphy, J. Lightwave Technol. 26, "
                "1423 (2008)")
    problem_kinds = ("mode",)

    @classmethod
    def available(cls):
        try:
            import scipy.sparse.linalg  # noqa: F401
        except ImportError:
            return False, "needs scipy (sparse LU + ARPACK); install scipy"
        return True, ""

    @classmethod
    def supports(cls, problem):
        if not isinstance(problem, ModeProblem):
            return False, "solves mode problems only"
        if problem.window_size_nm[0] <= 0:
            return False, "window_size_nm[0] must be > 0 (window_size_nm[1] = 0 gives a slab invariant along y)"
        if problem.window_size_nm[1] < 0:
            return False, "window_size_nm[1] must be >= 0"
        try:
            cls._options(problem)
        except Exception as e:  # noqa: BLE001
            return False, f"invalid fde options: {e}"
        return True, ""

    @classmethod
    def _options(cls, problem) -> FDEOptions:
        return FDEOptions.model_validate(options_for(problem, cls.name, cls.method))

    @classmethod
    def estimate(cls, problem, device: Device):
        opts = cls._options(problem)
        grid, boundary = _grid(problem, opts)
        dofs = 2 * grid.size
        return {"grid": [grid.nx, grid.ny], "grid_spacing_nm": [grid.dx, grid.dy if grid.ny > 1 else None],
                "unknowns": dofs, "boundary": boundary,
                "memory_bytes": int(dofs * 180 * 16),  # sparse LU fill ~100-150 nonzeros per unknown
                "solves": len(wavelength_array(problem.wavelengths)) * (3 if opts.group_index == "finite_difference"
                                                                         else 1),
                "fits": grid.size <= opts.max_cells}

    @classmethod
    def solve(cls, problem: ModeProblem, device: Device) -> ModeResult:
        from ._fde import CrossSection, ModeSolver, track

        t0 = time.perf_counter()
        opts = cls._options(problem)
        wl = wavelength_array(problem.wavelengths)
        grid, boundary = _grid(problem, opts)
        if grid.size > opts.max_cells:
            raise ValueError(f"grid {grid.nx}x{grid.ny} exceeds max_cells={opts.max_cells}; "
                             "increase grid_spacing_nm, use symmetry, or raise max_cells")
        section = CrossSection.from_profiles(problem.structures, problem.background)
        x_ref = problem.window_center_nm[0] if opts.bend_reference_x_nm is None else opts.bend_reference_x_nm
        solver = ModeSolver(section, grid, opts.subpixel, opts.subsamples, problem.bend_radius_nm, x_ref,
                            opts.pml_reflection)
        pol = None if opts.polarization == "any" else opts.polarization
        n_bg = float(np.real(problem.background.n(wl)).max())
        warnings: list[str] = []
        out, residuals, prev = [], [], None
        for lam in wl:
            modes = solver.solve(float(lam), problem.num_modes, problem.target_neff, pol, opts.group_index)
            if opts.track_modes and prev is not None:
                order = track(prev, modes)
                if order != list(range(len(modes))):
                    warnings.append(f"modes reordered by overlap tracking at {lam:g} nm")
                modes = [modes[i] for i in order]
            prev = modes
            if len(modes) < problem.num_modes:
                warnings.append(f"only {len(modes)} of {problem.num_modes} modes found at {lam:g} nm")
            row = []
            for j, m in enumerate(modes):
                residuals.append(m.residual)
                if m.neff.real <= n_bg:
                    warnings.append(f"mode {j} at {lam:g} nm has neff {m.neff.real:.4f} <= background index "
                                    f"{n_bg:.4f}: not guided (a window/radiation mode)")
                if m.meta.get("pml_fraction", 0) > 0.2:
                    warnings.append(f"mode {j} at {lam:g} nm has {m.meta['pml_fraction']:.0%} of |E|² in the PML")
                row.append(ModeData(
                    neff_real=float(m.neff.real), neff_imag=float(m.neff.imag),
                    ng=None if m.ng is None or not np.isfinite(m.ng) else float(m.ng),
                    te_fraction=m.te_fraction, loss_db_per_cm=m.loss_db_per_cm,
                    effective_area_um2=m.effective_area_um2,
                    field=_field(m) if problem.include_fields else None))
            out.append(row)
        provenance = {
            **device.describe(), "grid": grid.describe(), "unknowns": 2 * grid.size, "boundary": boundary,
            "subpixel": opts.subpixel, "eigensolver": "scipy splu (MMD_AT_PLUS_A) + ARPACK shift-invert",
            "target_neff": problem.target_neff, "max_residual": float(max(residuals)) if residuals else None,
            "group_index": opts.group_index, "bend_radius_nm": problem.bend_radius_nm,
            "field_normalization": "fields at cell centres; H in units of E (Z0 H); ½Re∫(E×H*)·z dA = 1 with dA in µm²"
                                   + (" (per µm along y for slabs)" if grid.invariant_y else ""),
            "runtime_s": time.perf_counter() - t0,
        }
        if any(grid.symmetry):
            provenance["note"] = "fields cover the computed half/quarter window; integrals include mirror images"
        return ModeResult(method=cls.method, backend=cls.name, wavelengths_nm=wl.tolist(), modes=out,
                          provenance=provenance, warnings=warnings)
