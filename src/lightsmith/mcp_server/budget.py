"""Estimate-before-solve guardrails.

Every solve is estimated first (``lightsmith.estimate`` never solves). A solve
whose estimated memory or run time exceeds the server's budget is refused with
the estimate and concrete ways to shrink the problem.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel

from ..problems import wavelength_array
from .config import GIB, ServerConfig

# Rough run-time model, the same one Lightsmith Cloud uses (lightsmith_cloud/billing.py): DDA cost
# per lattice cell per solve measured on an RTX 5070; everything else is order-of-magnitude only.
_STARTUP_S = 3.0  # solver process start-up (imports, device init) for jobs run in a subprocess
_DDA_S_PER_CELL_SOLVE = 3.0e-5
_FDTD_S_PER_CELL_STEP = {"gpu": 2.0e-9, "cpu": 4.0e-8}
_INVERSE_S_PER_SOLVE = 1.0

SHRINK_HINTS: dict[str, list[str]] = {
    "dda": ["lower options.dda.dipoles_per_diameter", "use fewer wavelengths",
            "drop orientation_average or lower options.dda.orientations", "run on a GPU (device='gpu')"],
    "fdtd": ["coarsen the grid: raise options.fdtd.spacing_nm or lower options.fdtd.cells_per_wavelength",
             "shrink options.fdtd.padding_nm", "lower options.fdtd.max_time_fs / run_time_fs",
             "use precision='single' on a GPU"],
    "fdfd": ["raise options.fdfd.dl_nm or lower options.fdfd.cells_per_wavelength", "shrink domain_size_nm",
             "use fewer wavelengths", "use dimension=2 where an effective-index model is acceptable"],
    "rcwa": ["lower options.rcwa.harmonics", "use fewer wavelengths",
             "lower options.rcwa.max_batch_bytes to solve fewer wavelengths at once"],
    "pwe": ["lower options.pwe.num_plane_waves or options.pwe.resolution", "lower num_bands or points_per_segment"],
    "fde": ["raise options.fde.grid_spacing_nm", "shrink window_size_nm",
            "use options.fde.symmetry", "use fewer wavelengths"],
    "eme": ["raise options.eme.grid_spacing_nm", "lower options.eme.num_modes", "shrink the cross-section"],
    "inverse": ["lower optimizer.max_iter", "coarsen the device grid (options.fdfd.dl_nm)",
                "use dimension=2", "use fewer wavelengths or objective terms"],
    "tmatrix": ["lower the multipole order", "use fewer particles or wavelengths"],
}
GENERIC_HINTS = ["use fewer wavelengths", "coarsen the discretisation", "pick a cheaper method (see list_backends)"]


def n_wavelengths(problem: BaseModel) -> int:
    for obj in (problem, getattr(problem, "device", None)):
        w = getattr(obj, "wavelengths", None)
        if w is not None:
            try:
                return int(len(wavelength_array(w)))
            except Exception:
                return 1
    return 1


def runtime_estimate(est: dict[str, Any], problem: BaseModel) -> tuple[float, str]:
    """(seconds, basis) from ``estimate()`` output; basis is backend | calibrated | rough."""
    for k in ("estimated_runtime_s", "runtime_s_estimate"):
        if isinstance(est.get(k), (int, float)):
            return float(est[k]), "backend"
    nwl = n_wavelengths(problem)
    if est.get("method") == "dda" and est.get("grid"):
        cells = math.prod(est["grid"])
        return _DDA_S_PER_CELL_SOLVE * cells * max(1, int(est.get("solves", nwl))), "calibrated"
    if est.get("method") == "fdtd" and est.get("cells"):
        steps = est.get("steps_estimate") or est.get("steps_max") or 0
        per = _FDTD_S_PER_CELL_STEP["gpu" if "cuda" in str(est.get("device", "")) else "cpu"]
        return per * float(est["cells"]) * float(steps) * max(1, int(est.get("runs", 1))), "rough"
    if est.get("max_iter"):  # inverse design: forward + adjoint solves per source per iteration
        per_iter = 2.0 * max(1, int(est.get("sources_per_iteration") or 1))
        return _INVERSE_S_PER_SOLVE * per_iter * int(est["max_iter"]), "rough"
    if est.get("cost") == "negligible":
        return 0.01 * nwl, "rough"
    mem = float(est.get("memory_bytes") or 0)
    return 0.002 * nwl + mem / 2e9, "rough"  # ~2 GB/s of working memory touched once


def check(est: dict[str, Any], problem: BaseModel, cfg: ServerConfig) -> dict[str, Any]:
    """Budget verdict for one estimate: {ok, reasons, hints, estimated_runtime_s, ...}."""
    secs, basis = runtime_estimate(est, problem)
    mem = est.get("memory_bytes")
    reasons = []
    if est.get("fits") is False:
        reasons.append(f"the problem exceeds the {est.get('backend')} backend's own size limit (fits=false)")
    if isinstance(mem, (int, float)) and mem > cfg.max_memory_bytes:
        reasons.append(f"estimated memory {mem / GIB:.2f} GiB exceeds the budget of {cfg.max_memory_bytes / GIB:.2f} GiB")
    if secs > cfg.max_runtime_s:
        reasons.append(f"estimated run time ~{secs:.0f} s ({basis}) exceeds the budget of {cfg.max_runtime_s:.0f} s")
    out = {"ok": not reasons, "estimated_runtime_s": round(secs, 3), "runtime_basis": basis,
           "memory_gib": round(mem / GIB, 4) if isinstance(mem, (int, float)) else None,
           "budget": {"max_memory_gib": round(cfg.max_memory_bytes / GIB, 3), "max_runtime_s": cfg.max_runtime_s}}
    if reasons:
        out["reasons"] = reasons
        out["how_to_shrink"] = SHRINK_HINTS.get(str(est.get("method")), GENERIC_HINTS)
        out["note"] = ("The budget is set by whoever runs this server (LIGHTSMITH_MCP_MAX_MEMORY_GB, "
                       "LIGHTSMITH_MCP_MAX_RUNTIME_S); shrink the problem, or ask them to raise it.")
    if basis == "rough":
        out["runtime_note"] = "order-of-magnitude guess; this backend reports no run time of its own"
    return out


class OverBudget(Exception):
    """A solve was refused before it started; ``details`` holds the estimate and the verdict."""

    def __init__(self, details: dict[str, Any]):
        self.details = details
        super().__init__("refused before solving: " + "; ".join(details["budget_check"].get("reasons", [])) +
                         ". How to shrink: " + "; ".join(details["budget_check"].get("how_to_shrink", [])))
