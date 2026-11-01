"""Linear algebra beyond the array API standard: general eig and differentiable matrix functions.

A patterned layer enters the S-matrix only through two analytic functions of
Omega^2 = P Q (see ``core``): K = sqrt(Omega^2) and E = exp(i d K). Their
gradients are computed with the Daleckii-Krein formula, whose divided differences
f[mu_i, mu_j] are evaluated in closed form, e.g. (sqrt(mu_i) - sqrt(mu_j)) /
(mu_i - mu_j) = 1 / (lam_i + lam_j). This stays exact for degenerate eigenvalues
(symmetric unit cells, normal incidence), where back-propagating through the
eigenvectors themselves is ill-defined.
"""

from __future__ import annotations

import numpy as np


def framework(xp) -> str:
    name = getattr(xp, "__name__", "")
    return "torch" if "torch" in name else "cupy" if "cupy" in name else "numpy"


def is_tensor(a) -> bool:
    """True for torch/cupy arrays (anything array-like that is not NumPy)."""
    return hasattr(a, "shape") and not isinstance(a, (np.ndarray, np.generic, float, int, complex))


# Where GPU eigensolves happen: "auto" = on the host (LAPACK) below this matrix size, on the
# device above it. torch.linalg.eig on CUDA (MAGMA geev) is slow for the batches of
# moderately sized matrices typical of RCWA; everything else stays on the GPU.
EIG_HOST_BELOW = 1000


def eig(xp, a):
    """Eigenvalues and right eigenvectors of a batch of general complex matrices (no autograd)."""
    fw = framework(xp)
    if fw == "torch":
        import torch

        if a.is_cuda and a.shape[-1] < EIG_HOST_BELOW:
            lam, v = np.linalg.eig(a.detach().cpu().numpy())
            return torch.as_tensor(lam, device=a.device), torch.as_tensor(v, device=a.device)
        return torch.linalg.eig(a.detach() if a.requires_grad else a)
    if fw == "cupy":  # no general eig on the GPU in CuPy: round-trip through NumPy
        import cupy

        lam, v = np.linalg.eig(cupy.asnumpy(a))
        return cupy.asarray(lam), cupy.asarray(v)
    return np.linalg.eig(a)


def _branch(xp, q, tol):
    return xp.where(xp.imag(q) + tol * xp.abs(q) < 0, -q, q)


def _sqrt_exp(xp, mu, v, d, tol):
    """lam = sqrt(mu) (Im >= 0), K = V lam V^-1, E = V exp(i d lam) V^-1; d: (B, 1)."""
    lam = _branch(xp, xp.sqrt(mu), tol)
    e = xp.exp(1j * lam * d)
    vinv = xp.linalg.inv(v)
    return lam, e, vinv, v @ (lam[..., :, None] * vinv), v @ (e[..., :, None] * vinv)


def sqrt_exp_functions(xp, m, d, tol: float):
    """K = sqrt(m) and E = exp(i d K) for a batch of matrices m (B, n, n) and d (B, 1).

    Returns (K, E). With torch, gradients flow to m and d.
    """
    if framework(xp) == "torch":
        import torch

        if torch.is_grad_enabled() and (m.requires_grad or (hasattr(d, "requires_grad") and d.requires_grad)):
            d_t = d if isinstance(d, torch.Tensor) else torch.as_tensor(d, dtype=m.real.dtype, device=m.device)
            return _torch_sqrt_exp().apply(m, d_t, tol)
    mu, v = eig(xp, m)
    return _sqrt_exp(xp, mu, v, d, tol)[3:]


_SQRT_EXP = None


def _torch_sqrt_exp():
    global _SQRT_EXP
    if _SQRT_EXP is not None:
        return _SQRT_EXP
    import torch

    class SqrtExp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, m, d, tol):
            import array_api_compat.torch as xp

            mu, v = eig(xp, m)
            lam, e, vinv, k, emat = _sqrt_exp(xp, mu, v, d, tol)
            ctx.save_for_backward(lam, e, v, vinv, d, k, emat)
            return k, emat

        @staticmethod
        def backward(ctx, g_k, g_e):
            lam, e, v, vinv, d, k, emat = ctx.saved_tensors
            li, lj = lam[..., :, None], lam[..., None, :]
            s = li + lj
            tiny = 1e-30
            s = torch.where(torch.abs(s) < tiny, torch.full_like(s, tiny), s)
            diff = li - lj
            dd = d[..., None] if d.ndim == lam.ndim else d
            # Divided differences of sqrt and exp(i d sqrt) on mu = lam^2.
            d_k = 1 / s
            z = 0.5j * dd * diff
            small = torch.abs(z) < 1e-3
            ei, ej = e[..., :, None], e[..., None, :]
            direct = (ei - ej) / torch.where(small, torch.ones_like(diff), diff * s)
            series = 1j * dd * torch.exp(0.5j * dd * s) * (1 + z * z / 6) / s
            d_e = torch.where(small, series, direct)
            vh, vinvh = v.mH, vinv.mH
            inner = torch.zeros_like(v)
            if g_k is not None:
                inner = inner + (vh @ g_k @ vinvh) * d_k.conj()
            if g_e is not None:
                inner = inner + (vh @ g_e @ vinvh) * d_e.conj()
            g_m = vinvh @ inner @ vh
            g_d = None
            if ctx.needs_input_grad[1] and g_e is not None:
                de_dd = 1j * (k @ emat)
                g_d = torch.real(torch.sum(torch.conj(g_e) * de_dd, dim=(-2, -1)))
                g_d = g_d.reshape(d.shape) if g_d.numel() == d.numel() else g_d.sum().reshape(d.shape)
            return g_m, g_d, None

    _SQRT_EXP = SqrtExp
    return SqrtExp
