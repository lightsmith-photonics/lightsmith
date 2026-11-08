"""Native photonic circuit solver (S-matrix netlists), CPU or GPU, differentiable.

Compact models and the interconnection engine live in :mod:`lightsmith.circuits`.
Options (``problem.options["circuit"]`` or ``["lightsmith-circuit"]``) are
validated by :class:`CircuitOptions`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..problems import CircuitProblem, options_for, wavelength_array
from ..results import SParameterResult
from .base import Backend, register


class CircuitOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["auto", "dense", "reduce"] = Field(
        "auto", description="dense: one batched (I - S C) solve; reduce: pairwise sub-circuit reduction "
        "(better for very large netlists); auto picks by size.")
    precision: Literal["double", "single"] = "double"
    unconnected: Literal["error", "ignore"] = Field(
        "error", description="Ports that are neither connected nor external: error, or treat as matched ends.")
    base_dir: str | None = Field(None, description="Directory that relative 'sparams:<path>' files are read from.")
    passivity_tol: float = Field(1e-6, ge=0, description="Warn when the largest singular value of S exceeds 1 + tol.")


@register
class NativeCircuit(Backend):
    name = "lightsmith-circuit"
    method = "circuit"
    options_model = CircuitOptions
    license = "MIT"
    citation = ("S-matrix interconnection S_ee + S_ei C (I - S_ii C)^-1 S_ie; pairwise reduction as scikit-rf's "
                "innerconnect (Arsenovic et al., IEEE Microwave Magazine 23, 98 (2022))")
    gpu = True
    differentiable = True
    problem_kinds = ("circuit",)

    @classmethod
    def _options(cls, problem) -> CircuitOptions:
        return CircuitOptions.model_validate(options_for(problem, cls.name, cls.method))

    @classmethod
    def supports(cls, problem):
        if not isinstance(problem, CircuitProblem):
            return False, "solves circuit problems only"
        from ..circuits.models import get_model

        bad = []
        for name, inst in problem.instances.items():
            try:
                get_model(inst.component)
            except KeyError:
                bad.append(f"{name} ({inst.component})")
        if bad:
            from ..circuits.models import model_names

            return False, f"unknown component model(s): {', '.join(bad)}; available: {', '.join(model_names())}"
        return True, ""

    @classmethod
    def estimate(cls, problem, device: Device):
        opts = cls._options(problem)
        n_wl = len(wavelength_array(problem.wavelengths))
        n_int = 2 * len(problem.connections)
        csize = 8 if opts.precision == "single" else 16
        return {"instances": len(problem.instances), "connections": len(problem.connections),
                "external_ports": len(problem.ports), "wavelengths": n_wl, "internal_ports": n_int,
                "precision": opts.precision, "memory_bytes": int(csize * n_wl * (3 * n_int**2 + 4 * n_int *
                                                                                   max(len(problem.ports), 1))),
                "fits": True}

    @classmethod
    def solve(cls, problem: CircuitProblem, device: Device) -> SParameterResult:
        from ..circuits.engine import circuit_sparameters

        t0 = time.perf_counter()
        opts = cls._options(problem)
        sol = circuit_sparameters(problem, device=device, algorithm=opts.algorithm, precision=opts.precision,
                                  unconnected=opts.unconnected,
                                  base_dir=Path(opts.base_dir) if opts.base_dir else None)
        s = sol.numpy().astype(complex)
        warnings = list(sol.warnings)
        if not np.all(np.isfinite(s)):
            warnings.append("S contains non-finite values (a lossless resonance exactly on a grid point?)")
        else:
            sv = np.linalg.svd(s, compute_uv=False).max(axis=-1)
            if sv.max() > 1 + opts.passivity_tol:
                i = int(np.argmax(sv))
                warnings.append(f"S is not passive: largest singular value {sv[i]:.6f} at "
                                f"{sol.wavelengths_nm[i]:.3f} nm (check model settings)")
        provenance = {**device.describe(), **sol.info, "runtime_s": time.perf_counter() - t0,
                      "time_convention": "exp(-i omega t)"}
        return SParameterResult(method=cls.method, backend=cls.name, wavelengths_nm=sol.wavelengths_nm.tolist(),
                                ports=sol.ports, s_real=s.real.tolist(), s_imag=s.imag.tolist(),
                                provenance=provenance, warnings=warnings)
