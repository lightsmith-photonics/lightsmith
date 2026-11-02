"""RCWA numerical core: layer eigenmodes and Redheffer-star S-matrix cascade.

Units are normalised by k0 (k -> k/k0, z -> k0 z) and fields use H~ = eta0 H, so
with exp(-i omega t) Maxwell's equations read curl E = i H~ and
curl H~ = -i eps E. For harmonics exp(i (kx x + ky y)) the transverse fields obey

    d/dz [Ex, Ey] = i P [Hx, Hy],   d/dz [Hx, Hy] = i Q [Ex, Ey],

    P = [[Kx Ezz^-1 Ky, I - Kx Ezz^-1 Kx], [Ky Ezz^-1 Ky - I, -Ky Ezz^-1 Kx]]
    Q = [[-Kx Ky - E_yx, Kx^2 - E_yy], [E_xx - Ky^2, Ky Kx + E_xy]]

where E_ij are the (factorised) Toeplitz permittivity matrices. Modes are
W exp(+-i lam z) with lam^2 = eig(PQ), Im(lam) > 0, and magnetic mode matrix
V = Q W lam^-1; patterned layers only need the matrix functions K = sqrt(PQ) and
E = exp(i d K) (see ``linalg``). Layers are connected through zero-thickness gap media with the
S-matrix formulas of Rumpf (PIER B 35, 241 (2011)); we choose the gap so that it
is never singular (W0 = I, V0 = Q for eps_gap = 1 + Kx^2 + Ky^2, i.e. kz_gap = 1).
All phase factors are exp(i lam d) with |.| <= 1, which keeps the cascade stable.

Every operation is batched over a leading axis (wavelengths, or wavelength x
angle) and written against the array API, so it runs on NumPy or PyTorch (with
autograd).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .linalg import sqrt_exp_functions


def branch(xp, q, tol):
    return xp.where(xp.imag(q) + tol * xp.abs(q) < 0, -q, q)


@dataclass
class PatternedMatrices:
    """Factorised permittivity matrices of one patterned layer, each (B, N, N)."""

    exx: Any
    exy: Any
    eyx: Any
    eyy: Any
    ezz_inv: Any


class Core:
    def __init__(self, xp, kx, ky, cdt, tol: float):
        """kx, ky: normalised in-plane wavevectors of every harmonic, (B, N)."""
        self.xp, self.kx, self.ky, self.cdt, self.tol = xp, kx, ky, cdt, tol
        self.n = kx.shape[-1]
        self.dev = kx.device
        self.eye_n = xp.eye(self.n, dtype=cdt, device=self.dev)
        self.eye = xp.eye(2 * self.n, dtype=cdt, device=self.dev)
        one = xp.ones_like(kx)
        # Gap medium: eps_g = 1 + kx^2 + ky^2 per harmonic, so kz_g = 1, W0 = I, V0 = Q_g.
        self.v0 = self.bdiag(-kx * ky, -one - ky * ky, one + kx * kx, kx * ky)
        self.lu_v0 = None

    # -- helpers ---------------------------------------------------------------------
    def diag(self, d):
        return d[..., :, None] * self.eye_n

    def bdiag(self, d11, d12, d21, d22):
        xp = self.xp
        top = xp.concat([self.diag(d11), self.diag(d12)], axis=-1)
        bot = xp.concat([self.diag(d21), self.diag(d22)], axis=-1)
        return xp.concat([top, bot], axis=-2)

    def blocks(self, a11, a12, a21, a22):
        xp = self.xp
        return xp.concat([xp.concat([a11, a12], axis=-1), xp.concat([a21, a22], axis=-1)], axis=-2)

    def homogeneous(self, eps):
        """kz (B, N) and dense Q (B, 2N, 2N) of a homogeneous medium, eps (B,)."""
        kx, ky, e = self.kx, self.ky, eps[:, None]
        kz = branch(self.xp, self.xp.sqrt(e - kx * kx - ky * ky), self.tol)
        q = self.bdiag(-kx * ky, kx * kx - e, e - ky * ky, kx * ky)
        return kz, q

    def _guard(self, kz):
        """Avoid division by kz = 0 exactly at a Rayleigh anomaly."""
        xp = self.xp
        tiny = 1e-12 if self.cdt == xp.complex128 else 1e-6
        return xp.where(xp.abs(kz) < tiny, xp.ones_like(kz) * (tiny * 1j), kz)

    def v_homogeneous(self, eps):
        kz, q = self.homogeneous(eps)
        kz = self._guard(kz)
        lam = self.xp.concat([kz, kz], axis=-1)
        return q / lam[..., None, :], kz

    # -- S-matrices ------------------------------------------------------------------
    def s_ref(self, eps):
        xp = self.xp
        v, _ = self.v_homogeneous(eps)
        x = xp.linalg.solve(self.v0, v)
        a, b = self.eye + x, self.eye - x
        ai = xp.linalg.inv(a)
        return (-(ai @ b), 2 * ai, 0.5 * (a - b @ ai @ b), b @ ai)

    def s_trn(self, eps):
        xp = self.xp
        v, _ = self.v_homogeneous(eps)
        x = xp.linalg.solve(self.v0, v)
        a, b = self.eye + x, self.eye - x
        ai = xp.linalg.inv(a)
        return (b @ ai, 0.5 * (a - b @ ai @ b), 2 * ai, -(ai @ b))

    def _s_layer(self, a, b, lam, d_k0):
        xp = self.xp
        x = xp.exp(1j * lam * d_k0)  # (B, 2N), |x| <= 1
        ai = xp.linalg.inv(a)
        xb = x[..., :, None] * b
        m = a - xb @ ai @ xb
        s11 = xp.linalg.solve(m, xb @ ai @ (x[..., :, None] * a) - b)
        s12 = xp.linalg.solve(m, x[..., :, None] * (a - b @ ai @ b))
        return (s11, s12, s12, s11)

    def s_uniform(self, eps, d_k0):
        """Homogeneous layer: W = I, lam = kz (doubly degenerate), no eigensolver.

        The eigenmode form needs Q^-1, which is singular for an order grazing inside the layer (kz = 0,
        a Rayleigh anomaly) and cancels catastrophically near it. Such orders (|kz| < 1 and |kz d| < 1)
        use the layer's transfer matrix instead (_s_transfer), which is entire in kz but grows like
        exp(|Im kz| d), so it cannot serve the strongly evanescent orders. A homogeneous layer does not
        couple harmonics, so each order's rows of the S-matrix come from one form, and in the other form
        that order sees a harmless stand-in (the gap medium, or zero thickness).
        """
        xp = self.xp
        kz, q = self.homogeneous(eps)
        graze = (xp.abs(kz) < 1) & (xp.abs(kz * d_k0) < 1)  # (B, N)
        rows = xp.concat([graze, graze], axis=-1)[..., :, None]
        kz_e = xp.where(graze, xp.ones_like(kz), kz)
        lam = xp.concat([kz_e, kz_e], axis=-1)
        vinv_v0 = lam[..., :, None] * xp.linalg.solve(xp.where(rows, self.v0, q), self.v0)  # V^-1 V0 = lam Q^-1 V0
        s_eig = self._s_layer(self.eye + vinv_v0, self.eye - vinv_v0, lam, d_k0)
        if not bool(xp.any(graze)):
            return s_eig
        s_tr = self._s_transfer(eps, kz, q, d_k0, graze)
        return tuple(xp.where(rows, t, e) for t, e in zip(s_tr, s_eig))

    def _s_transfer(self, eps, kz, q, d_k0, graze):
        """Homogeneous layer from its transfer matrix in (E, H), for the orders in graze:

        T = [[cos, -i sin V^-1], [-i V sin, cos]] with V = Q / kz, so V sin = Q sin(kz d) / kz and
        V^-1 sin = adj(Q) sin(kz d) / (e kz) (each 2x2 block of Q has det e kz^2), finite at kz = 0.
        Orders outside graze get cos = 1, sin = 0 (zero thickness).
        """
        xp = self.xp
        kx, ky, e = self.kx, self.ky, eps[:, None]
        theta = kz * d_k0
        tiny = 1e-12 if self.cdt == xp.complex128 else 1e-6
        small = xp.abs(kz) < tiny
        sinc = xp.where(small, d_k0 * xp.ones_like(kz), xp.sin(theta) / xp.where(small, xp.ones_like(kz), kz))
        cos = xp.where(graze, xp.cos(xp.where(graze, theta, xp.zeros_like(theta))), xp.ones_like(theta))
        sinc = xp.where(graze, sinc, xp.zeros_like(sinc))
        c2 = xp.concat([cos] * 2, axis=-1)[..., :, None]
        s2 = xp.concat([sinc] * 2, axis=-1)[..., :, None]
        adj = self.bdiag(kx * ky, e - kx * kx, ky * ky - e, -kx * ky)
        t12 = -1j * s2 * adj / e[..., None]
        t21 = -1j * s2 * q
        # Change to gap modes on both sides, K = M0^-1 T M0 with M0 = [[I, I], [-V0, V0]]; the cos blocks
        # commute with V0 (both block-diagonal per harmonic).
        p = t12 @ self.v0
        r = xp.linalg.solve(self.v0, t21)
        k21 = 0.5 * (r - p)
        k22 = c2 * self.eye + 0.5 * (p + r)
        k22i = xp.linalg.inv(k22)
        s11 = -(k22i @ k21)
        s21 = c2 * self.eye - 0.5 * (p + r) + k21 @ k22i @ k21
        return (s11, k22i, s21, -(k21 @ k22i))

    def pq(self, mats: PatternedMatrices):
        """Omega^2 = P Q and Q of a patterned layer."""
        kx, ky = self.kx, self.ky
        ez = mats.ezz_inv
        kxc, kyc = kx[..., :, None], ky[..., :, None]
        kxr, kyr = kx[..., None, :], ky[..., None, :]
        eye = self.eye_n
        p = self.blocks(kxc * ez * kyr, eye - kxc * ez * kxr, kyc * ez * kyr - eye, -(kyc * ez * kxr))
        q = self.blocks(-self.diag(kx * ky) - mats.eyx, self.diag(kx * kx) - mats.eyy,
                        mats.exx - self.diag(ky * ky), self.diag(kx * ky) + mats.exy)
        return p @ q, q

    def s_patterned(self, mats: PatternedMatrices, d_k0):
        """Patterned layer from the matrix functions K = W lam W^-1 and E = W exp(i lam d) W^-1.

        Rumpf's layer S-matrix is invariant under A -> W^-1 A^, B -> W^-1 B^, X -> E, so it
        only needs A^ = I + K Q^-1 V0, B^ = I - K Q^-1 V0 (no explicit eigenvectors), which
        keeps gradients well defined for degenerate modes.
        """
        xp = self.xp
        omega2, q = self.pq(mats)
        k, e = sqrt_exp_functions(xp, omega2, d_k0, self.tol)
        kv = k @ xp.linalg.solve(q, self.v0)
        a, b = self.eye + kv, self.eye - kv
        ai = xp.linalg.inv(a)
        eb = e @ b
        m = a - eb @ ai @ eb
        s11 = xp.linalg.solve(m, eb @ ai @ e @ a - b)
        s12 = xp.linalg.solve(m, e @ (a - b @ ai @ b))
        return (s11, s12, s12, s11)

    def star(self, sa, sb):
        """Redheffer star product sa (x) sb."""
        xp = self.xp
        a11, a12, a21, a22 = sa
        b11, b12, b21, b22 = sb
        d = xp.linalg.solve((self.eye - b11 @ a22).mT, a12.mT).mT  # a12 (I - b11 a22)^-1
        f = xp.linalg.solve((self.eye - a22 @ b11).mT, b21.mT).mT  # b21 (I - a22 b11)^-1
        return (a11 + d @ b11 @ a21, d @ b12, f @ a21, b22 + f @ a22 @ b12)

    # -- fields and fluxes -----------------------------------------------------------
    def flux(self, e, h):
        """Per-harmonic Poynting flux Re(Ex Hy* - Ey Hx*), (B, N, P) for fields (B, 2N, P)."""
        xp, n = self.xp, self.n
        ex, ey, hx, hy = e[..., :n, :], e[..., n:, :], h[..., :n, :], h[..., n:, :]
        return xp.real(ex * xp.conj(hy) - ey * xp.conj(hx))
