"""Engine of the native 3D finite-difference frequency-domain (FDFD) solver.

Used by the ``lightsmith-fdfd`` backend for ``DeviceProblem(dimension=3)``, and
through it by 3D inverse design (``lightsmith.inverse``).

Why a frequency-domain solver with an iterative linear solve (and not a time-domain
adjoint on the native FDTD): the inverse-design machinery is built on exact
frequency-domain adjoints of S-parameters. A 3D FDFD keeps that contract unchanged:
the forward solution and the adjoint field are two linear solves with the same
operator (A and A^T), the gradient with respect to every permittivity voxel is one
matvec, port modes are exact discrete eigenmodes so S-parameters need no time
gating or normalisation runs, and each wavelength costs one solve rather than a run
long enough to reach steady state. Successive optimisation iterates change little,
so warm-starting the Krylov solver from the previous forward and adjoint fields
cuts its cost severalfold, which a time-domain run cannot exploit. Direct
factorisation, as in 2D, is out of reach for 3D grids of ~10^6 unknowns
(fill-in grows like n^(4/3)), so the system is solved matrix-free with BiCGStab,
in batches of right-hand sides, on the GPU.

Modules:

- ``grid``: uniform cubic Yee grid, SC-PML stretch factors, rasterisation.
- ``operators``: the curl-curl operator (optionally with a grad-div regulariser) and
  its transpose as array-API matvecs, plus a SciPy assembly for small grids.
- ``linsolve``: batched BiCGStab, SuperLU, and the adjoint autograd ``Function``.
- ``modes``: exact discrete full-vector port modes (generalised eigenproblem on the
  port plane).
- ``ports``: port geometry, TF/SF mode sources and modal decomposition.
- ``engine``: S-parameters (plain or differentiable) and the cost model.

References: Shin & Fan, J. Comput. Phys. 231, 3406 (2012) (SC-PML FDFD) and
Opt. Express 21, 22578 (2013) (grad-div acceleration); van der Vorst, SIAM J. Sci.
Stat. Comput. 13, 631 (1992) (BiCGStab); Hughes et al., ACS Photonics 5, 4781
(2018) (adjoint FDFD).
"""

from .engine import FDFD3D

__all__ = ["FDFD3D"]
