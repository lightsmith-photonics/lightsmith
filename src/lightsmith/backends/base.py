"""Backend interface and registry.

A backend wraps one implementation of one method. Third-party packages can
register more backends (e.g. GPL-licensed solvers kept out of the MIT core)
through the ``lightsmith.backends`` entry-point group, each entry pointing at
a :class:`Backend` subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Any, ClassVar

from ..device import Device
from ..results import Result
from ..spec import Simulation


class Backend(ABC):
    name: ClassVar[str]  # unique implementation id, e.g. "miepython"
    method: ClassVar[str]  # "mie" | "tmatrix" | "dda" | "fdtd"
    license: ClassVar[str]
    citation: ClassVar[str] = ""
    gpu: ClassVar[bool] = False  # can run on a GPU device
    differentiable: ClassVar[bool] = False  # gradients flow through solve() with a torch device
    problem_kinds: ClassVar[tuple[str, ...]] = ("scattering",)  # see lightsmith.problems.PROBLEM_KINDS

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """Whether the backend's dependencies are installed (and why not)."""
        return True, ""

    @classmethod
    @abstractmethod
    def supports(cls, sim: Simulation) -> tuple[bool, str]:
        """Whether this backend can solve `sim` (and why not)."""

    @classmethod
    def estimate(cls, sim: Simulation, device: Device) -> dict[str, Any]:
        """Cheap pre-flight estimate of problem size and memory. Never solves."""
        return {}

    @classmethod
    @abstractmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        ...

    @classmethod
    def info(cls) -> dict[str, Any]:
        ok, why = cls.available()
        return {"name": cls.name, "method": cls.method, "problem_kinds": list(cls.problem_kinds),
                "license": cls.license, "gpu": cls.gpu, "differentiable": cls.differentiable,
                "available": ok, "unavailable_reason": why or None, "citation": cls.citation}


def unsupported_outputs(sim: Simulation, *, near_field: bool = False, far_field: bool = False,
                        substrate: bool = False, orientation_average: bool = False,
                        resonances: bool = False, gaussian_beam: bool = False) -> str:
    """Helper for `supports`: name the requested features a backend lacks ('' if none)."""
    missing = []
    if sim.outputs.near_field is not None and not near_field:
        missing.append("near_field output")
    if sim.outputs.far_field is not None and not far_field:
        missing.append("far_field output")
    if sim.illumination.kind == "gaussian_beam" and not gaussian_beam:
        missing.append("gaussian_beam illumination")
    if sim.outputs.resonances is not None and not resonances:
        missing.append("resonances output (time-domain solvers only)")
    if sim.substrate is not None and not substrate:
        missing.append("substrate")
    if sim.orientation_average and not orientation_average:
        missing.append("orientation_average")
    return "does not support " + ", ".join(missing) if missing else ""


_REGISTRY: dict[str, type[Backend]] = {}
_PLUGINS_LOADED = False


def register(cls: type[Backend]) -> type[Backend]:
    _REGISTRY[cls.name] = cls
    return cls


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for ep in entry_points(group="lightsmith.backends"):
        register(ep.load())


def all_backends() -> list[type[Backend]]:
    _load_plugins()
    return list(_REGISTRY.values())


def get_backend(name_or_method: str, kind: str | None = None) -> type[Backend]:
    """Look up by backend name, or by method name (first available backend for `kind` wins)."""
    _load_plugins()
    if name_or_method in _REGISTRY:
        return _REGISTRY[name_or_method]
    candidates = [b for b in _REGISTRY.values() if b.method == name_or_method]
    if kind is not None:
        candidates = [b for b in candidates if kind in b.problem_kinds] or candidates
    if not candidates:
        known = sorted({b.method for b in _REGISTRY.values()} | set(_REGISTRY))
        raise KeyError(f"unknown method or backend {name_or_method!r}; known: {known}")
    for b in candidates:
        if b.available()[0]:
            return b
    return candidates[0]
