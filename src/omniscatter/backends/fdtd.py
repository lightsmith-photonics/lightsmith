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

Written against the Python array API: the same code runs on NumPy (CPU) and
PyTorch (CPU/CUDA); on CUDA the time loop is replayed as CUDA graphs.
Tabulated/library materials are fitted with
:func:`omniscatter.materials_fit.fit_drude_lorentz`.
"""

from __future__ import annotations

from ..device import Device
from ..results import Result
from ..spec import Simulation
from ._fdtd import scattering
from ._fdtd.media import fit_module, needs_fit
from ._fdtd.options import FDTDOptions
from .base import Backend, register, unsupported_outputs

__all__ = ["FDTDOptions", "NativeFDTD"]


@register
class NativeFDTD(Backend):
    name = "omniscatter-fdtd"
    method = "fdtd"
    license = "MIT"
    citation = ("Taflove & Hagness, Computational Electrodynamics (2005); Roden & Gedney, MOTL 27, 334 (2000); "
                "Oskooi et al., CPC 181, 687 (2010)")
    gpu = True

    @classmethod
    def supports(cls, sim: Simulation):
        why = unsupported_outputs(sim, near_field=True)
        if not why and any(needs_fit(m) for p in sim.particles for m in p.region_materials()) \
                and fit_module() is None:
            why = "tabulated/library materials need omniscatter.materials_fit (Drude-Lorentz fitting)"
        return (not why), why

    @classmethod
    def _options(cls, sim) -> FDTDOptions:
        return FDTDOptions.model_validate(sim.options_for(cls.name, cls.method))

    @classmethod
    def estimate(cls, sim, device: Device):
        opts = cls._options(sim)
        single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu)
        p = scattering.plan(sim, opts, single)
        lay = p["layout"]
        return {"grid": list(lay.shape), "cells": lay.cells, "spacing_nm": p["dx"], "dt_fs": p["dt"] / scattering.FS,
                "steps_estimate": p["est_steps"], "steps_max": p["max_steps"], "dft_stride": p["stride"],
                "precision": "single" if single else "double", "memory_bytes": p["memory_bytes"],
                "fits": lay.cells <= opts.max_cells, "warnings": p["warnings"]}

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        ok, why = cls.supports(sim)
        if not ok:
            raise ValueError(f"{cls.name} {why}")
        out = scattering.solve(sim, cls._options(sim), device)
        res = Result.from_arrays(method=cls.method, backend=cls.name, wavelengths_nm=sim.wavelength_array(),
                                 c_ext=out["c_ext"], c_sca=out["c_sca"], c_abs=out["c_abs"],
                                 r_eq=sim.equivalent_radius_nm(), provenance=out["provenance"],
                                 warnings=out["warnings"])
        res.near_field = out["near_field"]
        return res
