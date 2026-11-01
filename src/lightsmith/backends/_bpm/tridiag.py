"""Block-tridiagonal solves by parallel cyclic reduction (PCR), written against the array API.

The system is ``A_i x_{i-1} + B_i x_i + C_i x_{i+1} = d_i`` for lines i = 0..n-1,
with dense (nb x nb) blocks (nb = 1: scalar tridiagonal, the 2D case). PCR
eliminates the neighbours at distance s = 1, 2, 4, ... in every line at once,

    K1 = A_i B_{i-s}^-1,  K2 = C_i B_{i+s}^-1,
    A_i <- -K1 A_{i-s},   C_i <- -K2 C_{i+s},   B_i <- B_i - K1 C_{i-s} - K2 A_{i+s},
    d_i <- d_i - K1 d_{i-s} - K2 d_{i+s},

so after ceil(log2 n) levels every line is decoupled and x_i = B_i^-1 d_i. Each
level is a handful of batched (matrix) products over all lines, which suits
GPUs (no sequential sweep over n as in the Thomas algorithm). The multipliers
of every level are kept, so later right-hand sides with the same matrix (all
steps through a uniform section) cost ~2 log2(n) batched products.

Performance note: NumPy inverts a stack of small blocks one LAPACK call at a time,
each multi-threaded by OpenBLAS; on a busy machine that thread contention can
make 3D factorisations ~100x slower. Set OPENBLAS_NUM_THREADS=1, or use
``device="torch-cpu"`` (torch batches the inverses), for 3D on the CPU.

No pivoting across lines: fine for the Crank-Nicolson BPM matrices (diagonally
dominant for paraxial steps, mildly indefinite but strongly damped for Padé
steps); the diagonal blocks themselves are inverted with pivoting.
"""

from __future__ import annotations


def _down(xp, a, s):
    """a shifted to larger line index by s (a_{i-s} at i), zeros where i - s < 0. Lines are axis -3."""
    pad = xp.zeros_like(a[..., :s, :, :])
    return xp.concat([pad, a[..., :-s, :, :]], axis=-3)


def _up(xp, a, s):
    """a_{i+s} at i, zeros where i + s >= n."""
    pad = xp.zeros_like(a[..., :s, :, :])
    return xp.concat([a[..., s:, :, :], pad], axis=-3)


class BlockTridiag:
    """PCR factorisation of a block-tridiagonal matrix; blocks are (..., n, nb, nb) arrays."""

    def __init__(self, xp, A, B, C):
        self.xp = xp
        n, nb = B.shape[-3], B.shape[-1]
        self.scalar = nb == 1
        mm, inv = self._mm, self._inv
        levels = []
        s = 1
        while s < n:
            bi = inv(B)
            k1 = mm(A, _down(xp, bi, s))
            k2 = mm(C, _up(xp, bi, s))
            B = B - mm(k1, _down(xp, C, s)) - mm(k2, _up(xp, A, s))
            A = -mm(k1, _down(xp, A, s))
            C = -mm(k2, _up(xp, C, s))
            levels.append((s, k1, k2))
            s *= 2
        self.levels = levels
        self.binv = inv(B)

    def _mm(self, a, b):
        return a * b if self.scalar else self.xp.matmul(a, b)

    def _inv(self, a):
        return 1.0 / a if self.scalar else self.xp.linalg.inv(a)

    def solve(self, d):
        """x for right-hand sides d of shape (..., n, nb, k)."""
        xp, mm = self.xp, self._mm
        for s, k1, k2 in self.levels:
            d = d - mm(k1, _down(xp, d, s)) - mm(k2, _up(xp, d, s))
        return mm(self.binv, d)

    def nbytes(self) -> int:
        def nb(a):
            return a.numel() * a.element_size() if hasattr(a, "numel") else a.nbytes

        return sum(nb(k1) + nb(k2) for _, k1, k2 in self.levels) + nb(self.binv)


def block_matvec(xp, A, B, C, x):
    """(A x_{i-1} + B x_i + C x_{i+1}) for x of shape (..., n, nb, k)."""
    if B.shape[-1] == 1:
        return A * _down(xp, x, 1) + B * x + C * _up(xp, x, 1)
    return xp.matmul(A, _down(xp, x, 1)) + xp.matmul(B, x) + xp.matmul(C, _up(xp, x, 1))
