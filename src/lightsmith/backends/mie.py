"""Mie theory for a single homogeneous sphere, via miepython (MIT)."""

from __future__ import annotations

import time

import numpy as np

from ..device import Device
from ..results import Result
from ..spec import Simulation, Sphere
from .base import Backend, register, unsupported_outputs


@register
class MiePython(Backend):
    name = "miepython"
    method = "mie"
    license = "MIT"
    citation = "S. Prahl, miepython, https://github.com/scottprahl/miepython"

    @classmethod
    def available(cls):
        try:
            import miepython  # noqa: F401
        except ImportError:
            return False, "miepython is not installed"
        return True, ""

    @classmethod
    def supports(cls, sim: Simulation):
        if len(sim.particles) != 1:
            return False, "Mie handles exactly one particle; use 'tmatrix', 'bem' or 'dda' for clusters"
        if not isinstance(sim.particles[0].shape, Sphere):
            return False, f"Mie handles spheres only, got {sim.particles[0].shape.kind!r}; use 'bem' or 'dda'"
        why = unsupported_outputs(sim, orientation_average=True)
        return (not why), why

    @classmethod
    def estimate(cls, sim, device):
        return {"cost": "negligible", "memory_bytes": 0}

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        import miepython

        t0 = time.perf_counter()
        p = sim.particles[0]
        wl = sim.wavelength_array()
        n = p.material.n(wl)
        r = p.shape.radius_nm
        # miepython documents n - ik (exp(+i omega t)); we use n + ik, so conjugate.
        qext, qsca, _qback, _g = miepython.efficiencies(np.conj(n), 2 * r, wl, sim.medium_index)
        geo = np.pi * r**2
        c_ext, c_sca = np.asarray(qext) * geo, np.asarray(qsca) * geo
        warnings = [] if device.name == "cpu" else ["Mie always runs on the CPU; device ignored"]
        return Result.from_arrays(
            method=cls.method, backend=cls.name, wavelengths_nm=wl,
            c_ext=c_ext, c_sca=c_sca, c_abs=c_ext - c_sca, r_eq=r,
            provenance={"miepython_version": miepython.__version__, "device": "cpu",
                        "runtime_s": time.perf_counter() - t0},
            warnings=warnings,
        )
