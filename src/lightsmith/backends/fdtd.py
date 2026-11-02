"""Native finite-difference time-domain (FDTD) solver, CPU or GPU.

A 3D Yee grid (Taflove & Hagness, *Computational Electrodynamics*, 3rd ed.)
with convolutional PML (Roden & Gedney, MOTL 27, 334 (2000)), a
total-field/scattered-field plane-wave source fed by a 1D auxiliary grid, and
Drude/Lorentz dispersion by auxiliary differential equations in D-form
(as in Meep, Oskooi et al., CPC 181, 687 (2010)). Interfaces use anisotropic
subpixel smoothing (Farjadpour et al., Opt. Lett. 31, 2972 (2006)), extended to
dispersive media by integrating the harmonic and arithmetic averages as two
parallel ADE paths.

One broadband Gaussian pulse covers all requested wavelengths. Running DFTs give

    C_sca = (outward flux of the scattered field outside the TFSF box) / I_inc
    C_abs = -(net flux of the total field inside the TFSF box) / I_inc
    C_ext = C_sca + C_abs

Periodic structures and planar stacks (``PeriodicProblem``, ``LayerStackProblem``) are
solved in one unit cell: periodic x/y boundaries, CPML along z, non-cubic cells so each period
holds a whole number of cells, a one-sided TFSF plane source, and reflection/transmission planes
whose DFT fields are split into diffraction orders by a 2D FFT (``_fdtd/periodic.py``). Normal
incidence is one broadband real run; oblique and conical incidence use Bloch-periodic boundaries
at fixed in-plane wavevector (complex fields), one run per (angle, wavelength, polarisation) so
that every result is at the requested angle. Scattering problems accept any incidence direction
(the TFSF box interpolates its 1D feed off-axis).

Written against the Python array API: the same code runs on NumPy (CPU) and
PyTorch (CPU/CUDA); on CUDA the time loop is replayed as CUDA graphs.
Tabulated/library materials are fitted with
:func:`lightsmith.materials_fit.fit_drude_lorentz`.
"""

from __future__ import annotations

from ..device import Device
from ..problems import EmissionProblem, LayerStackProblem, PeriodicProblem, options_for
from ..results import EmissionResult, Result, SpectrumResult
from ..spec import Simulation
from ._fdtd import emission, periodic, scattering
from ._fdtd.media import fit_module, needs_fit
from ._fdtd.options import FDTDOptions
from .base import Backend, register, unsupported_outputs

__all__ = ["FDTDOptions", "NativeFDTD"]


@register
class NativeFDTD(Backend):
    name = "lightsmith-fdtd"
    method = "fdtd"
    license = "MIT"
    citation = ("Taflove & Hagness, Computational Electrodynamics (2005); Roden & Gedney, MOTL 27, 334 (2000); "
                "Oskooi et al., CPC 181, 687 (2010)")
    gpu = True
    problem_kinds = ("scattering", "periodic", "layer_stack", "emission")

    @classmethod
    def supports(cls, sim):
        if isinstance(sim, EmissionProblem):
            return emission.supports(sim)
        if isinstance(sim, (PeriodicProblem, LayerStackProblem)):
            ok, why = periodic.supports(sim)
            if ok and any(needs_fit(m) for m in periodic._materials(sim)) and fit_module() is None:
                ok, why = False, "tabulated/library materials need lightsmith.materials_fit"
            return ok, why
        if not isinstance(sim, Simulation):
            return False, "FDTD solves scattering, periodic and layer_stack problems"
        why = unsupported_outputs(sim, near_field=True, far_field=True, resonances=True, gaussian_beam=True,
                                  substrate=True)
        if not why and sim.illumination.kind == "gaussian_beam" and sim.outputs.near_field is not None:
            why = "does not support near_field output with gaussian_beam illumination (yet)"
        if not why and sim.substrate is not None:
            why = scattering.substrate_unsupported(sim)
        if not why and any(needs_fit(m) for p in sim.particles for m in p.region_materials()) \
                and fit_module() is None:
            why = "tabulated/library materials need lightsmith.materials_fit (Drude-Lorentz fitting)"
        return (not why), why

    @classmethod
    def _options(cls, sim) -> FDTDOptions:
        return FDTDOptions.model_validate(options_for(sim, cls.name, cls.method))

    @classmethod
    def estimate(cls, sim, device: Device):
        opts = cls._options(sim)
        single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu)
        if isinstance(sim, EmissionProblem):
            single = emission.uses_single(opts)
            p = emission.plan(sim, opts, single)
            lay = p["layout"]
            return {"grid": list(lay.shape), "cells": lay.cells, "spacing_nm": p["dx"], "dt_fs": p["dt"] / scattering.FS,
                    "steps_estimate": p["est_steps"], "steps_max": p["max_steps"], "runs": p["runs"],
                    "dft_stride": p["stride"], "precision": "single" if single else "double",
                    "memory_bytes": p["memory_bytes"], "fits": lay.cells <= opts.max_cells, "warnings": p["warnings"],
                    "notes": "two runs: the scene, and the same dipole in the medium alone (P0)"}
        if isinstance(sim, (PeriodicProblem, LayerStackProblem)):
            p = periodic.plan(sim, opts, single)
            c = p["cell"]
            out = {"grid": list(c.shape), "real_cells": list(c.n_cells) + [c.shape[2]], "cells": c.cells,
                   "spacing_nm": list(c.spacing), "dt_fs": p["dt"] / scattering.FS, "steps_max": p["max_steps"],
                   "runs": p["runs"], "dft_stride": p["stride"], "precision": "single" if single else "double",
                   "memory_bytes": p["memory_bytes"], "fits": c.cells <= opts.max_cells}
            if p["oblique_runs"]:
                out["oblique_runs"] = p["oblique_runs"]
                out["oblique_run_min_steps"] = p["oblique_run_min_steps"]
                out["notes"] = ("oblique incidence: one fixed-k_par complex-field run per (angle, wavelength, "
                                "polarisation); runs at large angles use narrow-band, longer pulses")
            return out
        p = scattering.plan(sim, opts, single)
        lay = p["layout"]
        return {"grid": list(lay.shape), "cells": lay.cells, "spacing_nm": p["dx"], "dt_fs": p["dt"] / scattering.FS,
                "steps_estimate": p["est_steps"], "steps_max": p["max_steps"], "dft_stride": p["stride"],
                "precision": "single" if single else "double", "memory_bytes": p["memory_bytes"],
                "fits": lay.cells <= opts.max_cells, "warnings": p["warnings"]}

    @classmethod
    def solve(cls, sim, device: Device) -> Result | SpectrumResult | EmissionResult:
        ok, why = cls.supports(sim)
        if not ok:
            raise ValueError(f"{cls.name} {why}")
        if isinstance(sim, EmissionProblem):
            return emission.solve(sim, cls._options(sim), device)
        if isinstance(sim, (PeriodicProblem, LayerStackProblem)):
            return periodic.solve(sim, cls._options(sim), device)
        out = scattering.solve(sim, cls._options(sim), device)
        res = Result.from_arrays(method=cls.method, backend=cls.name, wavelengths_nm=sim.wavelength_array(),
                                 c_ext=out["c_ext"], c_sca=out["c_sca"], c_abs=out["c_abs"],
                                 r_eq=sim.equivalent_radius_nm(), provenance=out["provenance"],
                                 warnings=out["warnings"])
        res.near_field, res.far_field = out["near_field"], out["far_field"]
        res.resonances = out["resonances"]
        return res
