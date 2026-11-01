"""Native finite-difference frequency-domain (FDFD) solver for 2D photonic devices.

Solves the scalar 2D Helmholtz equation on a Yee grid with stretched-coordinate
PML (TE: Hz with E in the xy plane; TM: Ez), launches unidirectional port modes
(computed on the same discretisation), and extracts the full, power-normalised
S-matrix by modal decomposition, including multimode ports.

Linear solves are direct: SciPy SuperLU on the CPU, or a block-tridiagonal LU
written in torch on CUDA GPUs (torch lacks cuDSS on this build, so no sparse
direct solver is available there). Gradients with respect to every permittivity
pixel come from one adjoint (transposed) solve reusing the forward factors; see
:meth:`FDFD.differentiable_solve`, the functional API used by
``lightsmith.inverse``.

References: Shin & Fan, J. Comput. Phys. 231, 3406 (2012) (SC-PML FDFD);
Hughes et al., ACS Photonics 6, 3010 (2019) (ceviche, adjoint FDFD);
Molesky et al., Nat. Photonics 12, 659 (2018) (inverse design review).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..device import Device
from ..problems import DeviceProblem, options_for, wavelength_array
from ..results import SParameterResult
from ._fdfd.engine import FDFD2D, FDFDOptions
from .base import Backend, register


@dataclass
class DifferentiableSolution:
    """Output of :meth:`FDFD.differentiable_solve` (torch tensors, autograd-connected)."""

    s: Any  # (n_wl, n_out, n_src) complex: S[out, src] per wavelength
    fields: Any  # (n_wl, n_src, *grid_shape) complex, or None
    outputs: list[str]  # labels of the S rows (all port modes)
    inputs: list[str]  # labels of the excited port modes (S columns)
    wavelengths_nm: np.ndarray
    grid: Any  # backend grid object (has .x, .y node coordinates in nm)

    def sparam(self, out: str, inp: str):
        """S[out <- inp] over wavelengths, e.g. ``sol.sparam('o2', 'o1')``."""
        return self.s[:, self.outputs.index(out), self.inputs.index(inp)]


@register
class FDFD(Backend):
    name = "lightsmith-fdfd"
    method = "fdfd"
    license = "MIT"
    citation = "Shin & Fan, J. Comput. Phys. 231, 3406 (2012); Hughes et al., ACS Photonics 6, 3010 (2019)"
    gpu = True
    differentiable = True
    problem_kinds = ("device",)

    @classmethod
    def available(cls):
        if importlib.util.find_spec("scipy") is None:
            return False, "needs scipy (`uv pip install scipy`)"
        return True, ""

    @classmethod
    def supports(cls, problem):
        if not isinstance(problem, DeviceProblem):
            return False, "FDFD solves device problems"
        if problem.dimension != 2:
            return False, "3D FDFD is not implemented yet; use dimension=2 (structures in the xy plane)"
        for p in problem.ports:
            if p.size_nm[2] == 0:
                return False, f"port {p.name!r} is normal to z; 2D ports must be normal to x or y"
        try:
            cls._options(problem)
        except Exception as e:
            return False, f"invalid fdfd options: {e}"
        return True, ""

    @classmethod
    def _options(cls, problem) -> FDFDOptions:
        return FDFDOptions.model_validate(options_for(problem, cls.name, cls.method))

    @classmethod
    def engine(cls, problem: DeviceProblem) -> FDFD2D:
        """The underlying :class:`FDFD2D` (grid, permittivity, modes, field solves)."""
        return FDFD2D(problem, cls._options(problem))

    @classmethod
    def estimate(cls, problem, device):
        return cls.engine(problem).estimate()

    @classmethod
    def solve(cls, problem: DeviceProblem, device: Device) -> SParameterResult:
        eng = cls.engine(problem)
        S, info = eng.solve(device if device.is_gpu else None)
        warnings = list(eng.warnings)
        if not any(complex(m.n(wavelength_array(problem.wavelengths))[0]).imag > 0
                   for m in [eng.problem.background] + [s.material for s in eng.problem.structures]):
            excess = (np.abs(S) ** 2).sum(axis=1).max() - 1
            if excess > 1e-2:
                warnings.append(f"output power exceeds input by {excess:.2%} in a lossless device; "
                                "refine the grid (dl_nm) or thicken the PML")
        provenance = {**device.describe(), **eng.estimate(), "solver": info.get("factor"),
                      "runtime_s": info.get("runtime_s"),
                      "port_planes_nm": {pg.name: float(eng.grid.axis(pg.axis)[pg.i0]) for pg in eng.ports}}
        provenance.pop("fits", None)
        return SParameterResult(method=cls.method, backend=cls.name,
                                wavelengths_nm=eng.wavelengths.tolist(), ports=eng.labels,
                                s_real=S.real.tolist(), s_imag=S.imag.tolist(),
                                provenance=provenance, warnings=warnings)

    # --- the differentiable functional API (see lightsmith.inverse) ------------------

    @classmethod
    def prepare(cls, problem: DeviceProblem) -> dict[str, Any]:
        """Context reused across optimisation iterations (grid, rasterisation, port setup)."""
        return {"engine": cls.engine(problem)}

    @classmethod
    def design_grid(cls, problem: DeviceProblem):
        """The grid on which `eps_grid_override` is given: node coordinates ``.x``, ``.y`` in nm."""
        return cls.engine(problem).grid

    @classmethod
    def base_permittivity(cls, problem: DeviceProblem) -> np.ndarray:
        """Rasterised permittivity of `problem`, shape (n_wl, Nx, Ny), complex."""
        eng = cls.engine(problem)
        return np.stack([eng.base_eps(float(lam)) for lam in eng.wavelengths])

    @classmethod
    def differentiable_solve(cls, problem: DeviceProblem, eps_grid_override=None, device: Device | str = "cpu",
                             sources: list[str] | None = None, with_fields: bool = True,
                             engine: FDFD2D | None = None) -> DifferentiableSolution:
        """Torch-differentiable S-parameters (and fields) for a permittivity grid.

        eps_grid_override: torch tensor (Nx, Ny) or (n_wl, Nx, Ny) on :meth:`design_grid`;
        None = the problem's own structures. sources: port labels to excite (default all).
        Gradients w.r.t. the override are exact and cost one adjoint solve per wavelength.
        """
        import torch

        from ..device import resolve_device

        dev = resolve_device(device) if isinstance(device, str) else device
        tdev = torch.device(dev.name) if dev.framework == "torch" else torch.device("cpu")
        eng = engine or cls.engine(problem)
        if eps_grid_override is None:
            eps_grid_override = torch.as_tensor(np.stack([eng.base_eps(float(w)) for w in eng.wavelengths]))
        which = None if sources is None else [eng.labels.index(s) for s in sources]
        res = eng.differentiable(eps_grid_override, tdev, which, with_fields=with_fields)
        return DifferentiableSolution(s=res["s"], fields=res["fields"], outputs=res["outputs"], inputs=res["inputs"],
                                      wavelengths_nm=eng.wavelengths, grid=eng.grid)


# Register the inverse-design wrapper (method "inverse", problem kind "optimization").
from ..inverse import backend as _inverse_backend  # noqa: E402,F401
