"""Backend wrapper so optimisation problems run through ``lightsmith.solve`` / CLI / MCP.

``solve(OptimizationProblem(...), method="inverse")`` runs :func:`lightsmith.inverse.optimize`
and returns an :class:`~lightsmith.inverse.OptimizationResult`. Registered when the FDFD
backend module is imported (i.e. always, with the built-in backends).
"""

from __future__ import annotations

from ..backends.base import Backend, get_backend, register
from ..device import Device
from .spec import OptimizationProblem


@register
class InverseDesign(Backend):
    name = "lightsmith-inverse"
    method = "inverse"
    license = "MIT"
    citation = ("Hammond et al., Opt. Express 29, 23916 (2021); Svanberg, SIAM J. Optim. 12, 555 (2002); "
                "Molesky et al., Nat. Photonics 12, 659 (2018)")
    gpu = True
    differentiable = False
    problem_kinds = ("optimization",)

    @classmethod
    def available(cls):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return False, "inverse design needs PyTorch (`uv sync --extra gpu`)"
        return True, ""

    @classmethod
    def supports(cls, problem):
        if not isinstance(problem, OptimizationProblem):
            return False, "expects an optimization problem"
        try:
            b = get_backend(problem.backend, kind="device")
        except KeyError as e:
            return False, str(e)
        if not getattr(b, "differentiable", False) or not hasattr(b, "differentiable_solve"):
            return False, f"backend {b.name!r} is not differentiable"
        ok, why = b.available()
        if ok:
            ok, why = b.supports(problem.device)
        return ok, why

    @classmethod
    def estimate(cls, problem, device):
        b = get_backend(problem.backend, kind="device")
        est = b.estimate(problem.device, device)
        n_src = len({(t.in_port, t.in_mode) if hasattr(t, "in_port") else (t.source_port, t.source_mode)
                     for t in problem.objective})
        return {**est, "differentiable_backend": b.name, "max_iter": problem.optimizer.max_iter,
                "sources_per_iteration": n_src}

    @classmethod
    def solve(cls, problem, device: Device):
        from .topopt import optimize

        return optimize(problem, device=device)
