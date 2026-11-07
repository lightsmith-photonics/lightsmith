"""Linear solvers for the 3D FDFD operator and its adjoint autograd rule.

- :func:`bicgstab`: Jacobi-preconditioned BiCGStab (van der Vorst 1992), matrix-free and
  batched over right-hand sides (every column has its own scalars and stops on its
  own), written against the array API so it runs on NumPy or torch (CPU/CUDA).
- :class:`DirectFactor`: SciPy SuperLU of the assembled matrix, for small grids
  (tests, exact gradients) on the CPU.

Adjoint rule (torch convention: the backward pass maps grad_X to J^H grad_X).
With A(eps) X = B and mu = A^-T conj(grad_X):

    grad_B   = conj(mu)
    grad_eps = -conj(d/d eps [mu^T A(eps) X])  (X, mu held fixed)

The second line is evaluated by differentiating one matvec with torch autograd, so
it stays exact for whatever the operator contains. One adjoint solve per wavelength
gives the gradient with respect to every permittivity pixel.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


def _dot(xp, a, b):
    """Conjugated inner products per right-hand side: (k,) for arrays of shape (k, ...)."""
    k = a.shape[0]
    return xp.sum(xp.reshape(xp.conj(a) * b, (k, -1)), axis=1)


def _norm(xp, a):
    k = a.shape[0]
    return xp.sqrt(xp.sum(xp.reshape(xp.real(a) ** 2 + xp.imag(a) ** 2, (k, -1)), axis=1))


def bicgstab(xp, matvec, b, x0=None, minv=None, tol: float = 1e-6, maxiter: int = 20000,
             info: dict | None = None, check_every: int = 10):
    """Solve A x = b for b of shape (k, ...) column by column (batched).

    minv: right preconditioner as an array multiplied elementwise (Jacobi: 1 / diag A).
    Stops when every column has ||b - A x|| <= tol ||b||. info gets iterations and residuals.
    """
    t0 = time.perf_counter()
    k = b.shape[0]
    bshape = (k,) + (1,) * (b.ndim - 1)

    def col(v):
        return xp.reshape(v, bshape)

    prec = (lambda v: v) if minv is None else (lambda v: minv * v)
    x = xp.zeros_like(b) if x0 is None else x0 + 0
    r = b - matvec(x) if x0 is not None else b + 0
    bn = _norm(xp, b)
    bn = xp.where(bn == 0, xp.ones_like(bn), bn)
    rhat = r + 0
    one = xp.ones((k,), dtype=b.dtype, device=getattr(b, "device", None)) if hasattr(b, "device") else \
        xp.ones((k,), dtype=b.dtype)
    rho, alpha, omega = one + 0, one + 0, one + 0
    v = xp.zeros_like(b)
    p = xp.zeros_like(b)
    res = _norm(xp, r) / bn
    it = 0
    done = res <= tol
    zero = xp.zeros_like(rho)
    act = xp.astype(~done, b.dtype)
    while it < maxiter and not (it == 0 and bool(xp.all(done))):
        it += 1
        rho_new = _dot(xp, rhat, r)
        safe = (xp.abs(rho) > 0) & (xp.abs(omega) > 0)
        beta = xp.where(safe, (rho_new / xp.where(safe, rho, one)) * (alpha / xp.where(safe, omega, one)), zero)
        p = r + col(beta) * (p - col(omega) * v)
        ph = prec(p)
        v = matvec(ph)
        den = _dot(xp, rhat, v)
        alpha = xp.where(xp.abs(den) > 0, rho_new / xp.where(xp.abs(den) > 0, den, one), zero) * act
        s = r - col(alpha) * v
        sh = prec(s)
        t = matvec(sh)
        tt = _dot(xp, t, t)
        omega = xp.where(xp.abs(tt) > 0, _dot(xp, t, s) / xp.where(xp.abs(tt) > 0, tt, one), zero) * act
        x = x + col(alpha) * ph + col(omega) * sh
        r = s - col(omega) * t
        rho = rho_new
        if it % check_every == 0 or it == maxiter:  # host syncs only every few iterations
            res = _norm(xp, r) / bn
            done = done | (res <= tol)
            act = xp.astype(~done, b.dtype)
            if bool(xp.all(done)) or not bool(xp.all(xp.isfinite(res))):
                break
    if info is not None:
        info["iterations"] = info.get("iterations", 0) + it
        info.setdefault("solves", []).append(
            {"iterations": it, "residual": float(xp.max(res)), "rhs": k, "seconds": time.perf_counter() - t0})
        info["max_residual"] = max(info.get("max_residual", 0.0), float(xp.max(res)))
    return x, res


class DirectFactor:
    """SuperLU of the assembled operator (CPU); exact solves and transposed solves."""

    kind = "scipy-superlu"

    def __init__(self, op):
        from scipy.sparse.linalg import splu

        self.n = 3 * math.prod(op.shape)
        self.lu = splu(op.to_scipy().tocsc(), permc_spec="COLAMD")

    def _run(self, b, trans):
        is_torch = hasattr(b, "detach")
        bn = b.detach().resolve_conj().cpu().numpy() if is_torch else np.asarray(b)
        k = bn.shape[0]
        x = self.lu.solve(np.ascontiguousarray(bn.reshape(k, self.n).T, dtype=complex), trans=trans).T
        x = x.reshape(bn.shape)
        if is_torch:
            import torch

            return torch.as_tensor(x, device=b.device)
        return x

    def solve(self, b):
        return self._run(b, "N")

    def solve_t(self, b):
        return self._run(b, "T")


class Solver:
    """Solves A x = b and A^T x = b for one operator with the configured method."""

    def __init__(self, op, method: str = "bicgstab", tol: float = 1e-6, maxiter: int = 20000,
                 precondition: bool = True, info: dict | None = None):
        self.op = op
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.info = {} if info is None else info
        self._lu = DirectFactor(op) if method == "direct" else None
        self.minv = None
        if method == "bicgstab" and precondition:
            d = op.diagonal()
            self.minv = 1 / d
        self.kind = "scipy-superlu" if method == "direct" else "bicgstab" + ("+jacobi" if self.minv is not None else "")

    def _iterate(self, matvec, b, x0, minv):
        x, res = bicgstab(self.op.xp, matvec, b, x0=x0, minv=minv, tol=self.tol, maxiter=self.maxiter,
                          info=self.info)
        bad = float(self.op.xp.max(res))
        if not bad <= 10 * self.tol:
            self.info.setdefault("warnings", []).append(
                f"BiCGStab stopped at relative residual {bad:.2e} (tol {self.tol:.0e}); raise max_iterations "
                f"(now {self.maxiter}) or check the PML and grid")
        return x

    def solve(self, b, x0=None):
        if self._lu is not None:
            return self._lu.solve(b)
        return self._iterate(self.op.apply, b, x0, self.minv)

    def solve_t(self, b, x0=None):
        if self._lu is not None:
            return self._lu.solve_t(b)
        # diag(A^T) = diag(A): the same Jacobi preconditioner works.
        return self._iterate(self.op.apply_t, b, x0, self.minv)


def _torch_solve_function():
    import torch

    class MaxwellSolve(torch.autograd.Function):
        """X = A(eps)^-1 B; differentiable in eps (nodes) and B via one adjoint solve."""

        @staticmethod
        def forward(ctx, eps, B, make_op, solver_kw, warm):
            from .operators import component_eps

            op = make_op(eps.detach())
            sol = Solver(op, **solver_kw)
            X = sol.solve(B, x0=warm.get("x") if warm is not None else None)
            if warm is not None:
                warm["x"] = X.detach()
            ctx.sol, ctx.make_op, ctx.warm = sol, make_op, warm
            ctx.component_eps = component_eps
            ctx.save_for_backward(eps, X)
            return X

        @staticmethod
        def backward(ctx, gX):
            import array_api_compat.torch as xp

            eps, X = ctx.saved_tensors
            warm = ctx.warm
            mu = ctx.sol.solve_t(gX.conj().resolve_conj(), x0=warm.get("mu") if warm is not None else None)
            if warm is not None:
                warm["mu"] = mu.detach()
            g_eps = None
            if ctx.needs_input_grad[0]:
                with torch.enable_grad():
                    e = eps.detach().requires_grad_(True)
                    ec = tuple(c[None] for c in ctx.component_eps(xp, e))
                    s = (mu * ctx.sol.op.apply(X, eps_c=ec)).sum()
                    (g,) = torch.autograd.grad(s, e, grad_outputs=torch.ones_like(s))
                g_eps = -g
            gB = mu.conj().resolve_conj() if ctx.needs_input_grad[1] else None
            return g_eps, gB, None, None, None

    return MaxwellSolve


_FN = None


def solve(eps, B, make_op, solver_kw: dict[str, Any], warm: dict | None = None):
    """Differentiable X = A(eps)^-1 B for torch tensors eps (Nx, Ny, Nz) and B (k, 3, Nx, Ny, Nz).

    make_op(eps) -> Maxwell3D builds the operator (the grad-div weight is taken from the
    detached eps; it does not change the solution, see ``operators``). warm: dict reused
    across calls to warm-start the forward ("x") and adjoint ("mu") iterations.
    """
    global _FN
    if _FN is None:
        _FN = _torch_solve_function()
    return _FN.apply(eps, B, make_op, solver_kw, warm)
