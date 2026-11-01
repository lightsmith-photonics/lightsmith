"""Inverse design: density-based topology optimisation on any differentiable backend.

Quick start::

    import lightsmith as ls
    from lightsmith import inverse

    spec = inverse.OptimizationProblem(device=device_problem, design_regions=[...],
                                       objective=[inverse.SParameterTerm(out_port="o2", in_port="o1")])
    res = inverse.optimize(spec, device="cpu", checkpoint="run.npz")
    res.final_device        # DeviceProblem with the exported polygons
    inverse.write_gds("design.gds", inverse.marching_squares(...))  # via lightsmith.layout

The differentiable-backend protocol
-----------------------------------

Inverse design works with any registered backend that sets ``differentiable = True``
and provides these classmethods (the FDFD backend, ``lightsmith-fdfd``, is the
reference implementation)::

    design_grid(problem) -> grid
        The grid on which permittivities are given; must expose node coordinates
        ``grid.x``, ``grid.y`` (nm, 1D arrays), the cell size ``grid.dl`` (nm) and
        ``grid.shape``. (3D backends add ``grid.z``.)

    base_permittivity(problem) -> np.ndarray, shape (n_wl, *grid.shape), complex
        The rasterised permittivity of the problem's own structures.

    differentiable_solve(problem, eps_grid_override, device, sources=None,
                         with_fields=True, **context) -> DifferentiableSolution
        eps_grid_override: torch tensor (n_wl, *grid.shape) or (*grid.shape), which
        may require grad. Returns torch tensors connected to the autograd graph:
        ``.s`` (n_wl, n_out, n_src) complex S[out, src], ``.outputs`` / ``.inputs``
        (port-mode labels, 'name' or 'name@mode'), ``.fields`` (n_wl, n_src, *grid.shape)
        or None, ``.grid``. Gradients must be exact (adjoint), and ports must not
        depend on the override inside design regions.

    prepare(problem) -> dict            (optional)
        Cached context (e.g. a factorisation-ready engine) passed back to
        ``differentiable_solve`` as ``**context`` on every iteration.

Conventions: lengths in nm, vacuum wavelengths, exp(-i omega t) (Im eps > 0 = loss),
power-normalised S-parameters.
"""

from importlib import import_module
from typing import Any, ClassVar, Protocol, runtime_checkable

# Only the (light) spec is imported eagerly, so ``lightsmith.problems`` can include
# OptimizationProblem in its union without an import cycle; the rest loads on first use.
from .spec import (DesignRegion, FieldIntensityTerm, ObjectiveTerm, OptimizationProblem, OptimizerSettings,
                   SParameterTerm)

_LAZY = {
    "export": ("Contour", "marching_squares", "to_gdsfactory", "to_structures", "write_gds"),
    "topopt": ("DesignOutput", "IterationState", "OptimizationResult", "TopologyOptimizer", "check_backend",
                 "load_checkpoint", "optimize", "save_checkpoint"),
    "optimizers": ("MMA", "Adam"),
    "parametrization": ("binarization", "conic_filter", "eta_from_lengthscale", "grayness",
                        "lengthscale_constraints", "tanh_projection"),
}
_WHERE = {name: mod for mod, names in _LAZY.items() for name in names}


def __getattr__(name: str):
    if name in _WHERE:
        return getattr(import_module(f"{__name__}.{_WHERE[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@runtime_checkable
class DifferentiableBackend(Protocol):
    """Structural type of a backend usable by :func:`optimize` (see the module docstring)."""

    name: ClassVar[str]
    differentiable: ClassVar[bool]

    @classmethod
    def design_grid(cls, problem) -> Any: ...

    @classmethod
    def base_permittivity(cls, problem) -> Any: ...

    @classmethod
    def differentiable_solve(cls, problem, eps_grid_override=None, device="cpu", sources=None,
                             with_fields: bool = True, **context) -> Any: ...


__all__ = [
    "MMA", "Adam", "Contour", "DesignOutput", "DesignRegion", "DifferentiableBackend", "FieldIntensityTerm",
    "IterationState", "ObjectiveTerm", "OptimizationProblem", "OptimizationResult", "OptimizerSettings",
    "SParameterTerm", "TopologyOptimizer", "binarization", "check_backend", "conic_filter", "eta_from_lengthscale",
    "grayness", "lengthscale_constraints", "load_checkpoint", "marching_squares", "optimize",
    "save_checkpoint", "tanh_projection", "to_gdsfactory", "to_structures", "write_gds",
]
