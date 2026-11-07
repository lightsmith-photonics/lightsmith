"""Top-level entry points: solve, estimate, compare, choose_method.

All accept any problem kind (see ``lightsmith.problems``) as a model, dict or
JSON string.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .backends import Backend, all_backends, get_backend
from .device import resolve_device
from .problems import parse_problem
from .results import Result, relative_difference

# Preference order per problem kind when method="auto": exact/cheap first, then more general.
AUTO_ORDER: dict[str, tuple[str, ...]] = {
    # BEM after T-matrix: it takes the non-spherical and mixed clusters neither Mie nor treams handles,
    # converging smoothly for metals where DDA/FDTD staircase; it cannot do substrates (DDA/FDTD then).
    "scattering": ("mie", "tmatrix", "bem", "dda", "fdtd"),
    "layer_stack": ("tmm", "rcwa", "fdtd"),
    "periodic": ("rcwa", "fdfd", "fdtd"),
    "mode": ("fde",),
    # BPM after EME: same port topology, but one-way (no reflections) and approximate at abrupt junctions;
    # before FDFD/FDTD because it is far cheaper for long, slowly varying devices.
    "device": ("eme", "bpm", "fdfd", "fdtd"),
    "bands": ("pwe", "fdfd"),
    "circuit": ("circuit",),
    "optimization": ("inverse",),
    "emission": ("mie", "fdtd"),
}


def _kind(problem: BaseModel) -> str:
    return getattr(problem, "kind", "scattering")


_SIZE_KEYS = ("unknowns", "dipoles", "modes", "cells", "edges")


def _fits(b: type[Backend], problem: BaseModel) -> tuple[bool, str]:
    """Whether `problem` is within the backend's own size limit, from its (cheap, never-solving) estimate.

    Lets `auto` fall through to the next method rather than pick one that would refuse at solve time.
    An estimate that fails says nothing about size, so the backend is kept and `solve` reports the error.
    """
    try:
        est = b.estimate(problem, resolve_device("cpu"))
    except Exception:  # noqa: BLE001
        return True, ""
    if est.get("fits", True) is False:
        size = ", ".join(f"{k}={est[k]}" for k in _SIZE_KEYS if k in est)
        return False, (f"too large for its size limit ({size}); pass method={b.name!r} with a larger "
                       "limit in its options to run it anyway")
    return True, ""


def choose_method(problem: BaseModel | dict | str) -> tuple[type[Backend], list[str]]:
    """Pick the preferred available backend for `problem`; also return why others were skipped.

    A backend is skipped when it is unavailable, does not support the problem, or reports in its
    estimate that the problem exceeds its size limit (BEM unknowns, DDA dipoles, T-matrix modes, ...).
    """
    problem = parse_problem(problem)
    kind = _kind(problem)
    skipped = []
    by_method: dict[str, list[type[Backend]]] = {}
    for b in all_backends():
        if kind in b.problem_kinds:
            by_method.setdefault(b.method, []).append(b)
    order = AUTO_ORDER.get(kind, ())
    for method in order + tuple(m for m in by_method if m not in order):
        for b in by_method.get(method, []):
            ok, why = b.available()
            if ok:
                ok, why = b.supports(problem)
            if ok:
                ok, why = _fits(b, problem)
            if ok:
                return b, skipped
            skipped.append(f"{b.name}: {why}")
    if not by_method:
        raise ValueError(f"no backend is installed for problem kind {kind!r}")
    raise ValueError("no available backend supports this problem: " + "; ".join(skipped))


def _resolve(problem: BaseModel, method: str) -> type[Backend]:
    if method == "auto":
        return choose_method(problem)[0]
    b = get_backend(method, kind=_kind(problem))
    if _kind(problem) not in b.problem_kinds:
        raise ValueError(f"backend {b.name!r} solves {list(b.problem_kinds)}, not {_kind(problem)!r}")
    for ok, why in (b.available(), b.supports(problem)):
        if not ok:
            raise ValueError(f"backend {b.name!r} cannot run this problem: {why}")
    return b


def solve(problem: BaseModel | dict | str, method: str = "auto", device: str = "auto") -> BaseModel:
    """Solve `problem` with one method (or backend name). `device`: auto | cpu | gpu | cuda:N | torch-cpu."""
    problem = parse_problem(problem)
    b = _resolve(problem, method)
    dev = resolve_device(device if b.gpu else "cpu")
    return b.solve(problem, dev)


def estimate(problem: BaseModel | dict | str, method: str = "auto", device: str = "auto") -> dict[str, Any]:
    """Pre-flight: which backend would run, on what device, and roughly how big. Never solves."""
    problem = parse_problem(problem)
    b = _resolve(problem, method)
    dev = resolve_device(device if b.gpu else "cpu")
    return {"backend": b.name, "method": b.method, **dev.describe(), **b.estimate(problem, dev)}


def discretization(problem: BaseModel | dict | str, method: str = "auto") -> dict[str, Any]:
    """What a backend will discretise before the run (grid, mesh, dipole lattice, harmonics, cells, plane waves),
    with its estimate: see :mod:`lightsmith.discretize`. Never solves."""
    from .discretize import discretization as _discretization

    return _discretization(problem, method)


def compare(problem: BaseModel | dict | str, methods: list[str] | None = None, device: str = "auto",
            reference: str | None = None) -> dict[str, Any]:
    """Solve with several methods and report differences against a reference.

    By default runs every available backend that supports `problem`, in ``AUTO_ORDER``. The
    reference is Mie when it ran, otherwise the first method: the one `auto` would pick when
    `methods` is not given, or the first one listed. Scattering results report relative
    differences of the cross sections; the other kinds report absolute differences (R/T/A,
    |S|^2 and S, neff and ng, band frequencies). Methods whose results cannot be compared
    with the reference are listed under ``not_compared``.
    """
    problem = parse_problem(problem)
    kind = _kind(problem)
    if methods is None:
        order = AUTO_ORDER.get(kind, ())
        found = [b for b in all_backends()
                 if kind in b.problem_kinds and b.available()[0] and b.supports(problem)[0]]
        found.sort(key=lambda b: order.index(b.method) if b.method in order else len(order))
        methods = [b.name for b in found]
    if not methods:
        raise ValueError(f"no available backend supports this {kind!r} problem")
    results = {m: solve(problem, m, device) for m in methods}
    if reference is None:
        reference = next((m for m, r in results.items() if getattr(r, "method", None) == "mie"), methods[0])
    ref = results[reference]
    diffs: dict[str, Any] = {}
    not_compared: list[str] = []
    for m, r in results.items():
        if m == reference:
            continue
        if isinstance(r, Result) and isinstance(ref, Result):
            diffs[m] = {q: relative_difference(r, ref, q) for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2")}
        elif hasattr(r, "difference") and type(r) is type(ref):
            diffs[m] = r.difference(ref)
        else:
            not_compared.append(m)
    out = {"reference": reference, "results": results, "relative_difference": diffs}
    if not_compared:
        out["not_compared"] = not_compared
    return out
