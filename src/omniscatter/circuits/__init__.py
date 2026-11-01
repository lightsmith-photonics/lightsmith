"""Photonic circuit simulation: an S-matrix netlist solver and compact models.

Solve a :class:`~omniscatter.problems.CircuitProblem` with
``omniscatter.solve(problem)`` (backend ``omniscatter-circuit``, method
``circuit``), or call :func:`circuit_sparameters` directly to get the S-matrix
as an array (a PyTorch tensor with gradients on ``device="torch-cpu"``/CUDA)::

    from omniscatter.circuits import circuit_sparameters
    sol = circuit_sparameters(problem, device="torch-cpu", params={"ring": {"neff": neff_tensor}})
    power = abs(sol["out", "in"]) ** 2

:func:`list_models` documents every compact model and its settings. Files
from field solvers (Touchstone ``.sNp`` or ``SParameterResult`` JSON) enter a
circuit as ``"component": "sparams:<path>"``; :func:`waveguide_from_modes`
turns an FDE ``ModeResult`` into a dispersive waveguide instance.
"""

from .engine import (
    CircuitSolution,
    NetlistError,
    check_netlist,
    circuit_sparameters,
    interconnect,
)
from .models import (
    MODELS,
    Ctx,
    compact_model,
    get_model,
    list_models,
    model_names,
    waveguide_from_modes,
)
from .touchstone import read_sparameters, read_touchstone, write_touchstone

__all__ = [
    "MODELS", "CircuitSolution", "Ctx", "NetlistError", "check_netlist", "circuit_sparameters", "compact_model",
    "get_model", "interconnect", "list_models", "model_names", "read_sparameters", "read_touchstone",
    "waveguide_from_modes", "write_touchstone",
]
