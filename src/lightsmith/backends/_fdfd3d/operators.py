"""The 3D Maxwell operator on a Yee grid, with SC-PML and a grad-div term (array API).

Unknowns are the three E components stored in one array of shape
``(k, 3, Nx, Ny, Nz)`` (k right-hand sides). Component ``a`` sits half a cell
up along axis ``a`` from node ``(i, j, l)``: Ex at ``(i + 1/2, j, l)``, Ey at
``(i, j + 1/2, l)``, Ez at ``(i, j, l + 1/2)``. E vanishes outside the grid
(PEC walls behind the PML). With lengths scaled by the cell size dl,

    A E = curl_B curl_F E - (k0 dl)^2 eps E - grad_F (w div_B (eps E))

``F_a`` is the forward difference along ``a`` divided by the stretch factor at
half-integer points, ``B_a`` the backward difference divided by the stretch at
the nodes (Shin & Fan 2012), so ``div_B curl_B = 0`` exactly. The last term
vanishes on every solution of Maxwell's equations with a divergence-free source
(and, for the total-field/scattered-field sources used here, it just changes the
right-hand side consistently, see ``ports``), so it does not change the solution.
It removes the huge null space of curl-curl (the gradient fields), which makes
Krylov solvers converge an order of magnitude faster: in a uniform medium the
operator becomes the vector Helmholtz operator ``-lap - k^2 eps`` (Shin & Fan,
Opt. Express 21, 22578 (2013)). ``w = 1/eps`` at the nodes.

The transpose (needed by the adjoint) uses ``F_a^T = -B'_a`` and ``B_a^T = -F'_a``
with the stretch applied *before* the difference (``B'_a c = bwd(c / s_half)``,
``F'_a c = fwd(c / s_node)``).

Everything is written against the array API, so the same code runs on NumPy and
torch (CPU/CUDA), and autograd can differentiate the matvec with respect to eps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ix(ax: int, sl) -> tuple:
    """Index tuple selecting `sl` along array axis `ax` (arrays of shape (k, Nx, Ny, Nz))."""
    return (slice(None),) * ax + (sl,)


def fwd(xp, c, ax: int):
    """out[i] = c[i+1] - c[i] along array axis `ax`, with c = 0 beyond the last index."""
    out = -c
    out[_ix(ax, slice(None, -1))] = out[_ix(ax, slice(None, -1))] + c[_ix(ax, slice(1, None))]
    return out


def bwd(xp, c, ax: int):
    """out[i] = c[i] - c[i-1] along array axis `ax`, with c = 0 before the first index."""
    out = c + 0
    out[_ix(ax, slice(1, None))] = out[_ix(ax, slice(1, None))] - c[_ix(ax, slice(None, -1))]
    return out


def component_eps(xp, eps):
    """(eps_x, eps_y, eps_z) at the E positions: means of the two adjacent nodes (edge replicated)."""
    out = []
    for a in range(3):
        n = eps.shape[a]
        up = xp.concat([eps[_ix(a, slice(1, n))], eps[_ix(a, slice(n - 1, n))]], axis=a)
        out.append((eps + up) / 2)
    return tuple(out)


@dataclass
class Stretch:
    """SC-PML factors per axis, shaped to broadcast against (k, Nx, Ny, Nz): nodes and half-integer points."""

    node: tuple  # 3 arrays
    half: tuple

    @classmethod
    def build(cls, xp, s_node: list[np.ndarray], s_edge: list[np.ndarray], device=None):
        """s_node[a]: (N_a,) at nodes; s_edge[a]: (N_a + 1,) at i - 1/2 (as the 2D grid returns)."""
        conv = (lambda v: xp.asarray(v)) if device is None else (lambda v: xp.asarray(v, device=device))
        node, half = [], []
        for a in range(3):
            shape = [1, 1, 1, 1]
            shape[a + 1] = -1
            node.append(conv(np.asarray(s_node[a], complex).reshape(shape)))
            half.append(conv(np.asarray(s_edge[a][1:], complex).reshape(shape)))
        return cls(tuple(node), tuple(half))


class Maxwell3D:
    """Matrix-free A (and A^T) for node permittivity `eps` (Nx, Ny, Nz) at one wavelength."""

    def __init__(self, xp, eps, k0dl: float, stretch: Stretch, grad_div: bool = True, w=None):
        self.xp = xp
        self.eps = eps
        self.k2 = k0dl**2
        self.st = stretch
        self.ec = tuple(e[None] for e in component_eps(xp, eps))  # (1, Nx, Ny, Nz) each
        self.grad_div = grad_div
        if grad_div and w is None:
            w = 1 / eps
        self.w = None if w is None else w[None]
        self.shape = tuple(eps.shape)

    # --- difference operators (on component arrays (k, Nx, Ny, Nz)) ---------------

    def F(self, c, a):
        return fwd(self.xp, c, a + 1) / self.st.half[a]

    def B(self, c, a):
        return bwd(self.xp, c, a + 1) / self.st.node[a]

    def Ft(self, c, a):  # = -B_a^T
        return fwd(self.xp, c / self.st.node[a], a + 1)

    def Bt(self, c, a):  # = -F_a^T
        return bwd(self.xp, c / self.st.half[a], a + 1)

    @staticmethod
    def _curl(D, E):
        ex, ey, ez = E
        return (D(ez, 1) - D(ey, 2), D(ex, 2) - D(ez, 0), D(ey, 0) - D(ex, 1))

    # --- products -------------------------------------------------------------------

    def apply(self, X, eps_c=None):
        """A X for X of shape (k, 3, Nx, Ny, Nz). eps_c overrides the component permittivities (autograd)."""
        xp = self.xp
        ec = self.ec if eps_c is None else eps_c
        E = (X[:, 0], X[:, 1], X[:, 2])
        R = list(self._curl(self.B, self._curl(self.F, E)))
        if self.grad_div:
            q = self.w * (self.B(ec[0] * E[0], 0) + self.B(ec[1] * E[1], 1) + self.B(ec[2] * E[2], 2))
            R = [R[a] - self.F(q, a) for a in range(3)]
        return xp.stack([R[a] - self.k2 * ec[a] * E[a] for a in range(3)], axis=1)

    def apply_t(self, X):
        """A^T X."""
        xp = self.xp
        ec = self.ec
        E = (X[:, 0], X[:, 1], X[:, 2])
        R = list(self._curl(self.Bt, self._curl(self.Ft, E)))
        if self.grad_div:
            q = self.w * (self.Bt(E[0], 0) + self.Bt(E[1], 1) + self.Bt(E[2], 2))
            R = [R[a] - ec[a] * self.Ft(q, a) for a in range(3)]
        return xp.stack([R[a] - self.k2 * ec[a] * E[a] for a in range(3)], axis=1)

    def diagonal(self):
        """diag(A) with shape (1, 3, Nx, Ny, Nz) (for Jacobi preconditioning)."""
        xp = self.xp
        st = self.st
        dd = []
        for b in range(3):  # -diag(B_b F_b) along axis b
            sh = st.half[b]
            n = sh.shape[b + 1]
            prev = xp.concat([xp.zeros_like(sh[_ix(b + 1, slice(0, 1))]), 1 / sh[_ix(b + 1, slice(0, n - 1))]],
                             axis=b + 1)
            dd.append((1 / sh + prev) / st.node[b])
        out = []
        for a in range(3):
            d = sum(dd[b] for b in range(3) if b != a) - self.k2 * self.ec[a]
            if self.grad_div:
                n = self.shape[a]
                wn = self.w / st.node[a]
                nxt = xp.concat([wn[_ix(a + 1, slice(1, n))], xp.zeros_like(wn[_ix(a + 1, slice(0, 1))])], axis=a + 1)
                d = d + self.ec[a] * (nxt + wn) / st.half[a]
            out.append(xp.broadcast_to(d, (1,) + self.shape))
        return xp.stack(out, axis=1)

    # --- explicit sparse matrix (SciPy; small grids, tests and direct solves) -------

    def to_scipy(self):
        """A as a SciPy CSR matrix, unknowns ordered like ``X.reshape(3 * N)`` (component-major, C order)."""
        import scipy.sparse as sp

        nx, ny, nz = self.shape
        dims = (nx, ny, nz)
        ident = [sp.identity(n, format="csr", dtype=complex) for n in dims]

        def axis_op(m, a):
            mats = list(ident)
            mats[a] = m
            return sp.kron(mats[0], sp.kron(mats[1], mats[2], format="csr"), format="csr")

        def np1(v):
            return np.asarray(_to_numpy(v)).ravel()

        Fm, Bm = [], []
        for a, n in enumerate(dims):
            f = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], format="csr", dtype=complex)
            b = sp.diags([np.ones(n), -np.ones(n - 1)], [0, -1], format="csr", dtype=complex)
            Fm.append(axis_op(sp.diags(1 / np1(self.st.half[a])) @ f, a))
            Bm.append(axis_op(sp.diags(1 / np1(self.st.node[a])) @ b, a))
        Z = None

        def curl(D):
            return sp.bmat([[Z, -D[2], D[1]], [D[2], Z, -D[0]], [-D[1], D[0], Z]], format="csr")

        A = curl(Bm) @ curl(Fm)
        ec = sp.diags(np.concatenate([np.asarray(_to_numpy(e)).ravel() for e in self.ec]))
        if self.grad_div:
            grad = sp.vstack(Fm, format="csr")
            div = sp.hstack(Bm, format="csr")
            A = A - grad @ sp.diags(np.asarray(_to_numpy(self.w)).ravel()) @ div @ ec
        return (A - self.k2 * ec).tocsr()


def _to_numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
