"""Engine of the native finite-difference frequency-domain (FDFD) solver.

Modules:

- ``grid``: Yee grid, SC-PML stretch factors, rasterisation of a ``DeviceProblem``.
- ``operators``: the 2D Helmholtz operator as five diagonals (array API, so it
  is differentiable when built from torch tensors).
- ``linsolve``: sparse direct solvers (SciPy SuperLU on CPU, a block-tridiagonal
  Schur-complement LU in torch for GPUs) and the adjoint autograd ``Function``.
- ``modes``: a 1D slab mode solver on the same discretisation (exact discrete
  port modes) plus the analytic slab dispersion relation.
- ``ports``: port geometry, unidirectional (TF/SF) mode sources, and modal
  decomposition into forward/backward amplitudes.
- ``engine``: ties these together into S-parameters (plain or differentiable).
"""

from .engine import FDFD2D, FDFDOptions
from .modes import slab_neff_analytic, solve_slab_modes

__all__ = ["FDFD2D", "FDFDOptions", "slab_neff_analytic", "solve_slab_modes"]
