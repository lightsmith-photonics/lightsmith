"""Backend wrapper so optimisation problems run through ``lightsmith.solve`` / CLI / MCP.

``solve(OptimizationProblem(...), method="inverse")`` runs :func:`lightsmith.inverse.optimize`
and returns an :class:`~lightsmith.inverse.OptimizationResult`. Registered when the FDFD
backend module is imported (i.e. always, with the built-in backends).
"""

from __future__ import annotations

from ..backends.base import Backend, get_backend, register
from ..device import Device
from .spec import OptimizationProblem


def device_problem(problem: OptimizationProblem):
    """The device problem the optimiser solves: the optimisation-level solver options override the device's,
    key by key per method (as the optimiser merges them)."""
    dev = problem.device
    if not problem.options:
        return dev
    merged = {m: {**dev.options.get(m, {}), **problem.options.get(m, {})} for m in {*dev.options, *problem.options}}
    return dev.model_copy(update={"options": merged})


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
        est = b.estimate(device_problem(problem), device)
        n_src = len({(t.in_port, t.in_mode) if hasattr(t, "in_port") else (t.source_port, t.source_mode)
                     for t in problem.objective})
        out = {**est, "differentiable_backend": b.name, "max_iter": problem.optimizer.max_iter,
               "sources_per_iteration": n_src}
        if est.get("dimension") == 3:
            # Per iteration and wavelength: one forward solve (all sources batched) and one adjoint solve.
            n_wl = est["wavelengths"]
            b_c = 8 if est.get("precision") == "single" else 16
            saved = n_wl * n_src * est["unknowns"] * b_c  # forward fields kept for the backward pass
            per_vec = est["unknowns"] * b_c
            working = per_vec * (9 * n_src + 6)
            out.update({
                "solves_per_iteration": 2 * n_wl,
                "memory_bytes": int(saved + working + 4 * per_vec),
                "memory_bytes_autograd_fields": int(saved),
            })
            t_solve = est.get("seconds_per_solve_estimate")
            if t_solve:
                cold = 2 * n_wl * t_solve
                # Warm starts typically need 3-5x fewer iterations after the first few iterations.
                out["seconds_per_iteration_estimate"] = {"cold": cold, "warm": cold / 4}
                out["runtime_s_estimate"] = cold * min(3, problem.optimizer.max_iter) + cold / 4 * max(
                    problem.optimizer.max_iter - 3, 0)
            out.pop("seconds_per_solve_estimate", None)
        return out

    @classmethod
    def solve(cls, problem, device: Device):
        from .topopt import optimize

        return optimize(problem, device=device)
