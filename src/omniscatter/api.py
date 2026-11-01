"""Top-level entry points: solve, estimate, compare, choose_method.

All accept any problem kind (see ``omniscatter.problems``) as a model, dict or
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
    "scattering": ("mie", "tmatrix", "dda", "fdtd"),
    "layer_stack": ("tmm", "rcwa", "fdtd"),
    "periodic": ("rcwa", "fdfd", "fdtd"),
    "mode": ("fde",),
    "device": ("eme", "fdfd", "fdtd"),
    "bands": ("pwe", "fdfd"),
    "circuit": ("circuit",),
}


def _kind(problem: BaseModel) -> str:
    return getattr(problem, "kind", "scattering")


def choose_method(problem: BaseModel | dict | str) -> tuple[type[Backend], list[str]]:
    """Pick the preferred available backend for `problem`; also return why others were skipped."""
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


def compare(problem: BaseModel | dict | str, methods: list[str] | None = None, device: str = "auto",
            reference: str | None = None) -> dict[str, Any]:
    """Solve with several methods and report relative differences against a reference.

    By default runs every available backend that supports `problem`, using Mie as the
    reference for scattering when it applies (otherwise the first result).
    """
    problem = parse_problem(problem)
    kind = _kind(problem)
    if methods is None:
        methods = [b.name for b in all_backends()
                   if kind in b.problem_kinds and b.available()[0] and b.supports(problem)[0]]
    results = {m: solve(problem, m, device) for m in methods}
    if reference is None:
        reference = next((m for m, r in results.items() if getattr(r, "method", None) == "mie"), methods[0])
    ref = results[reference]
    diffs: dict[str, Any] = {}
    for m, r in results.items():
        if m == reference:
            continue
        if isinstance(r, Result) and isinstance(ref, Result):
            diffs[m] = {q: relative_difference(r, ref, q) for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2")}
        elif hasattr(r, "difference"):
            diffs[m] = r.difference(ref)
    return {"reference": reference, "results": results, "relative_difference": diffs}
