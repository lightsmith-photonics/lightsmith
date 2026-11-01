"""Direct solvers for the five-diagonal FDFD operator, and its adjoint autograd rule.

- :class:`ScipyFactor`: SciPy SuperLU on the CPU (NumPy or CPU torch tensors).
- :class:`BlockFactor`: block-tridiagonal LU (block Thomas / Schur complement)
  in torch, so the factorisation and solves run on a CUDA GPU without cuDSS.
  The grid is cut into lines along the longer axis; each step inverts one dense
  (n_short x n_short) Schur complement. Cost O(n_long * n_short^3) flops and
  O(n_long * n_short^2) memory, which suits 2D devices up to ~500 cells across.

Both expose ``solve(B)`` for A X = B and ``solve_t(B)`` for A^T X = B with the
same factors, which is all the adjoint method needs.

Adjoint rule (torch complex convention: the backward pass maps grad_X to
J^H grad_X for this holomorphic map). With A X = B:

    grad_B   = A^-H grad_X = conj(A^-T conj(grad_X))
    grad_c   = -sum_rhs conj(dA/dc X) * grad_B   (one product per diagonal)

so a gradient with respect to every permittivity pixel costs one transposed
solve with the factors already computed in the forward pass.
"""

from __future__ import annotations

import numpy as np

from .operators import Diagonals


def _csr(d: Diagonals):
    import scipy.sparse as sp

    c0, cxm, cxp, cym, cyp = (np.ascontiguousarray(np.asarray(v, dtype=complex)) for v in d)
    nx, ny = c0.shape
    n = nx * ny
    diags = [c0.ravel(), cyp.ravel()[:-1], cym.ravel()[1:], cxp.ravel()[: n - ny], cxm.ravel()[ny:]]
    return sp.diags(diags, [0, 1, -1, ny, -ny], shape=(n, n), format="csc")


class ScipyFactor:
    kind = "scipy-superlu"

    def __init__(self, d: Diagonals):
        from scipy.sparse.linalg import splu

        self.shape = tuple(np.asarray(d.c0).shape)
        self.lu = splu(_csr(d), permc_spec="COLAMD")

    def _run(self, b, trans: str):
        is_torch = hasattr(b, "detach")
        bn = b.detach().resolve_conj().resolve_neg().cpu().numpy() if is_torch else np.asarray(b)
        k = bn.shape[2:]
        x = self.lu.solve(np.ascontiguousarray(bn.reshape(self.shape[0] * self.shape[1], -1), dtype=complex),
                          trans=trans).reshape(self.shape + k)
        if is_torch:
            import torch

            return torch.as_tensor(x, device=b.device).to(b.dtype if b.is_complex() else torch.complex128)
        return x

    def solve(self, b):
        return self._run(b, "N")

    def solve_t(self, b):
        return self._run(b, "T")


class BlockFactor:
    """Block-tridiagonal LU in torch (CPU or CUDA)."""

    kind = "torch-block-tridiagonal"

    def __init__(self, d: Diagonals, dtype=None):
        import torch

        c0, cxm, cxp, cym, cyp = (v.detach() for v in d)
        if dtype is not None:
            c0, cxm, cxp, cym, cyp = (v.to(dtype) for v in (c0, cxm, cxp, cym, cyp))
        nx, ny = c0.shape
        self.transposed = ny > nx  # blocks along the shorter axis
        if self.transposed:
            c0, cxm, cxp, cym, cyp = c0.T, cym.T, cyp.T, cxm.T, cxp.T
        # Now: blocks = rows of c0; within-block couplings cym/cyp; between blocks cxm/cxp.
        self.prev, self.next = cxm.contiguous(), cxp.contiguous()
        nb = c0.shape[0]
        self.dtype = c0.dtype
        D = torch.diag_embed(c0) + torch.diag_embed(cyp[:, :-1], 1) + torch.diag_embed(cym[:, 1:], -1)
        G = torch.empty_like(D)
        G[0] = torch.linalg.inv(D[0])
        for i in range(1, nb):
            S = D[i] - self.prev[i][:, None] * G[i - 1] * self.next[i - 1][None, :]
            G[i] = torch.linalg.inv(S)
        self.G = G
        self.nb = nb

    def _prep(self, b):
        squeeze = b.ndim == 2
        if squeeze:
            b = b[..., None]
        if self.transposed:
            b = b.transpose(0, 1)
        return b.to(self.dtype), squeeze

    def _post(self, x, dtype, squeeze):
        if self.transposed:
            x = x.transpose(0, 1)
        x = x.contiguous()
        if squeeze:
            x = x[..., 0]
        return x.to(dtype)

    def solve(self, b):
        import torch

        dtype = b.dtype
        b, sq = self._prep(b)
        G, P, N = self.G, self.prev, self.next
        y = torch.empty_like(b)
        y[0] = G[0] @ b[0]
        for i in range(1, self.nb):
            y[i] = G[i] @ (b[i] - P[i][:, None] * y[i - 1])
        x = torch.empty_like(b)
        x[-1] = y[-1]
        for i in range(self.nb - 2, -1, -1):
            x[i] = y[i] - G[i] @ (N[i][:, None] * x[i + 1])
        return self._post(x, dtype, sq)

    def solve_t(self, b):
        import torch

        dtype = b.dtype
        c, sq = self._prep(b)
        G, P, N = self.G, self.prev, self.next
        w = torch.empty_like(c)
        w[0] = c[0]
        for i in range(1, self.nb):
            w[i] = c[i] - N[i - 1][:, None] * (G[i - 1].T @ w[i - 1])
        z = torch.empty_like(c)
        z[-1] = G[-1].T @ w[-1]
        for i in range(self.nb - 2, -1, -1):
            z[i] = G[i].T @ (w[i] - P[i + 1][:, None] * z[i + 1])
        return self._post(z, dtype, sq)


def factorize(d: Diagonals, solver: str = "auto", dtype=None):
    """Pick a factorisation: SciPy for NumPy/CPU tensors, block LU for CUDA tensors (or when asked)."""
    on_gpu = hasattr(d.c0, "device") and getattr(d.c0.device, "type", "cpu") != "cpu"
    if solver == "block" or (solver == "auto" and on_gpu):
        return BlockFactor(d, dtype)
    if solver not in ("auto", "scipy"):
        raise ValueError(f"unknown solver {solver!r}; use 'auto', 'scipy' or 'block'")
    if hasattr(d.c0, "detach"):
        d = Diagonals(*(v.detach().cpu().numpy() for v in d))
    return ScipyFactor(d)


def _torch_solve_function():
    import torch

    class HelmholtzSolve(torch.autograd.Function):
        """X = A(c0..cyp)^-1 B, differentiable in the diagonals and B via one adjoint solve."""

        @staticmethod
        def forward(ctx, c0, cxm, cxp, cym, cyp, B, solver, dtype, out):
            fact = factorize(Diagonals(c0, cxm, cxp, cym, cyp), solver, dtype)
            X = fact.solve(B)
            ctx.fact = fact
            ctx.save_for_backward(X)
            if out is not None:
                out["factor"] = fact.kind
            return X

        @staticmethod
        def backward(ctx, gX):
            (X,) = ctx.saved_tensors
            gB = ctx.fact.solve_t(gX.conj().resolve_conj()).conj().resolve_conj()
            Xc = X.conj().resolve_conj()
            zx = torch.zeros_like(Xc[:1])
            zy = torch.zeros_like(Xc[:, :1])
            g0 = -(Xc * gB).sum(-1)
            gxm = -(torch.cat([zx, Xc[:-1]], 0) * gB).sum(-1)
            gxp = -(torch.cat([Xc[1:], zx], 0) * gB).sum(-1)
            gym = -(torch.cat([zy, Xc[:, :-1]], 1) * gB).sum(-1)
            gyp = -(torch.cat([Xc[:, 1:], zy], 1) * gB).sum(-1)
            return g0, gxm, gxp, gym, gyp, gB, None, None, None

    return HelmholtzSolve


_FN = None


def solve(d: Diagonals, B, solver: str = "auto", dtype=None, info: dict | None = None):
    """Solve A X = B, B of shape (Nx, Ny, n_rhs). Differentiable when given torch tensors."""
    if not hasattr(B, "detach"):  # NumPy: plain solve
        f = factorize(d, solver, dtype)
        if info is not None:
            info["factor"] = f.kind
        return f.solve(B)
    global _FN
    if _FN is None:
        _FN = _torch_solve_function()
    import torch

    shape = tuple(B.shape[:2])
    d = Diagonals(*(torch.broadcast_to(v, shape).to(torch.complex128) if not v.is_complex() or tuple(v.shape) != shape
                    else v for v in d))
    return _FN.apply(*d, B, solver, dtype, info)
