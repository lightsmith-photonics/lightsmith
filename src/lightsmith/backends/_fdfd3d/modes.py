"""Full-vector port modes on the 3D Yee discretisation (exact discrete modes).

For a port normal to axis ``a`` with transverse axes ``(t1, t2)`` (cyclic, so
``(a, t1, t2)`` is right-handed), a mode of the uniform waveguide is
``E(p) = e * u^(d (p - i0))`` with ``u = exp(i beta dl)`` and ``p`` the position
along ``a`` in cells (the normal component sits at half-integer positions).
Every difference along ``a`` then becomes the factor ``kappa = u^(1/2) - u^(-1/2)
= 2i sin(beta dl / 2)``, for nodes and half-points alike. With ``e_n = d kappa q``
the 3D operator of ``operators`` (without the grad-div term, which vanishes on
modes) reduces exactly to the generalized eigenproblem

    [ -(L + k^2 eps_n)  B1               B2              ] [q ]           [ 0    0  0 ] [q ]
    [  0               -B2 F2 - k^2 eps1  B2 F1          ] [e1] = kappa^2 [-F1   1  0 ] [e1]
    [  0                B1 F2           -B1 F1 - k^2 eps2] [e2]           [-F2   0  1 ] [e2]

(``L = B1 F1 + B2 F2``, lengths in cells), restricted to the port window with E = 0
outside it. Its left eigenvectors are ``(-kappa^2 q, e1, e2)``, so the modes are
exactly biorthogonal under ``e_t,m . (N v_n)_t``, the discrete (unconjugated) flux
``E x H``. Port modes are therefore exact solutions of the discrete 3D problem in a
straight section, which makes mode sources unidirectional and modal decomposition
exact up to round-off and the field leaking past the window edge.

Unit power: ``sin(beta dl) * e_t . (N v)_t = 1``, the exactly conserved discrete
Poynting flux up to a constant common to all modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def plane_axes(a: int) -> tuple[int, int]:
    return (a + 1) % 3, (a + 2) % 3


def plane(arr: np.ndarray, a: int, i: int) -> np.ndarray:
    """Slice of a (Nx, Ny, Nz) array at index i along a, indexed [t1, t2]."""
    s = np.take(arr, i, axis=a)
    t1, t2 = plane_axes(a)
    return s if t1 < t2 else s.T


@dataclass
class VectorMode:
    neff: complex
    beta_dl: complex
    kappa: complex  # u^(1/2) - u^(-1/2)
    q: np.ndarray  # (w1, w2): e_n / (d kappa) on the window
    e1: np.ndarray  # (w1, w2) transverse E along t1
    e2: np.ndarray
    proj: np.ndarray  # (2, w1, w2): amplitude = sum(proj * E_t)
    guided: bool

    def te_fraction(self, t: int) -> float:
        """Fraction of transverse |E|^2 along transverse axis t (0 -> t1, 1 -> t2)."""
        p1, p2 = float(np.sum(np.abs(self.e1) ** 2)), float(np.sum(np.abs(self.e2) ** 2))
        return (p1, p2)[t] / (p1 + p2)


def _ops(n1, n2, sn1, sh1, sn2, sh2):
    import scipy.sparse as sp

    def f(n, sh):
        return sp.diags(1 / sh) @ sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], dtype=complex)

    def b(n, sn):
        return sp.diags(1 / sn) @ sp.diags([np.ones(n), -np.ones(n - 1)], [0, -1], dtype=complex)

    I1, I2 = sp.identity(n1, dtype=complex), sp.identity(n2, dtype=complex)
    F1, B1 = sp.kron(f(n1, sh1), I2, format="csr"), sp.kron(b(n1, sn1), I2, format="csr")
    F2, B2 = sp.kron(I1, f(n2, sh2), format="csr"), sp.kron(I1, b(n2, sn2), format="csr")
    return F1, B1, F2, B2


def solve_vector_modes(eps_n, eps_1, eps_2, window, k0dl: float, num_modes: int, stretch=None,
                       extra: int = 6) -> list[VectorMode]:
    """Modes on one grid plane.

    eps_n/1/2: (N1, N2) permittivity of the normal / t1 / t2 E components on the plane.
    window: (lo1, hi1, lo2, hi2) index ranges where E may be nonzero.
    stretch: ((sn1, sh1), (sn2, sh2)) transverse stretch factors at nodes / half-points (default 1).
    Returns modes sorted by decreasing Re(neff), unit power, with fields on the window.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, splu

    n1, n2 = eps_n.shape
    lo1, hi1, lo2, hi2 = window
    if stretch is None:
        stretch = ((np.ones(n1), np.ones(n1)), (np.ones(n2), np.ones(n2)))
    (sn1, sh1), (sn2, sh2) = stretch
    F1, B1, F2, B2 = _ops(n1, n2, sn1, sh1, sn2, sh2)
    k2 = k0dl**2
    L = B1 @ F1 + B2 @ F2
    D = sp.diags
    Z = None
    M = sp.bmat([[-(L + k2 * D(eps_n.ravel())), B1, B2],
                 [Z, -(B2 @ F2) - k2 * D(eps_1.ravel()), B2 @ F1],
                 [Z, B1 @ F2, -(B1 @ F1) - k2 * D(eps_2.ravel())]], format="csr")
    Iw = sp.identity(n1 * n2, dtype=complex)
    N = sp.bmat([[sp.csr_matrix((n1 * n2, n1 * n2), dtype=complex), None, None], [-F1, Iw, None],
                 [-F2, None, Iw]], format="csr")
    i1, i2 = np.meshgrid(np.arange(lo1, hi1), np.arange(lo2, hi2), indexing="ij")
    idx = (i1 * n2 + i2).ravel()
    np_ = n1 * n2
    sel = np.concatenate([idx, idx + np_, idx + 2 * np_])
    Mw = M[sel][:, sel].tocsc()
    Nw = N[sel][:, sel].tocsr()
    nw = len(idx)
    eps_max = float(max(np.real(e[lo1:hi1, lo2:hi2]).max() for e in (eps_n, eps_1, eps_2)))
    sigma = -k2 * eps_max * 1.0001
    lu = splu((Mw - sigma * Nw).tocsc())
    op = LinearOperator(Mw.shape, matvec=lambda v: lu.solve(Nw @ v), dtype=complex)
    k = min(num_modes + extra, 3 * nw - 2)
    try:
        nu, vec = eigs(op, k=k, which="LM", tol=1e-12, maxiter=20 * 3 * nw)
    except ArpackNoConvergence as e:  # pragma: no cover - keep what converged
        nu, vec = e.eigenvalues, e.eigenvectors
    keep = np.abs(nu) > 1e-12
    kap2 = sigma + 1 / nu[keep]
    vec = vec[:, keep]
    K = np.sqrt(-kap2 + 0j)
    K = np.where(K.real < 0, -K, K)
    order = np.argsort(-K.real)
    border = np.zeros((hi1 - lo1, hi2 - lo2), bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    eps_clad = float(max(np.real(e[lo1:hi1, lo2:hi2][border]).max() for e in (eps_1, eps_2)))
    shape_w = (hi1 - lo1, hi2 - lo2)
    modes: list[VectorMode] = []
    for j in order[:num_modes]:
        Kj = complex(K[j])
        if Kj.real <= 0 or abs(Kj.imag) > abs(Kj.real):
            break
        half = Kj / 2
        if half.real >= 1:
            raise ValueError("grid too coarse for this mode (beyond the Nyquist limit); reduce dl_nm")
        beta = complex(2 * np.arcsin(half))
        if beta.imag < 0 and abs(beta.imag) < 1e-12:
            beta = beta.conjugate()
        u_half = np.exp(0.5j * beta)
        kappa = complex(u_half - 1 / u_half)
        v = vec[:, j]
        nv = Nw @ v
        et, nt = v[nw:], nv[nw:]
        i = int(np.argmax(np.abs(et)))
        v = v * (abs(et[i]) / et[i])
        nv = Nw @ v
        et, nt = v[nw:], nv[nw:]
        if np.allclose(v.imag, 0, atol=1e-12 * np.abs(v).max()):
            v, nv = v.real.astype(complex), nv.real.astype(complex)
            et, nt = v[nw:], nv[nw:]
        flux = np.sin(beta) * np.sum(et * nt)
        c = np.sqrt(flux)
        v, nv = v / c, nv / c
        et, nt = v[nw:], nv[nw:]
        proj = (nt / np.sum(et * nt)).reshape((2,) + shape_w)
        e1, e2 = v[nw:2 * nw].reshape(shape_w), v[2 * nw:].reshape(shape_w)
        amp = np.maximum(np.abs(e1), np.abs(e2))
        edge = float(amp[border].max() / amp.max())
        neff = beta / k0dl
        guided = bool(neff.real**2 > eps_clad and edge < 1e-2)
        modes.append(VectorMode(neff=complex(neff), beta_dl=beta, kappa=kappa, q=v[:nw].reshape(shape_w), e1=e1,
                                e2=e2, proj=proj, guided=guided))
    if len(modes) < num_modes:
        raise ValueError(f"only {len(modes)} propagating mode(s) found at this port; requested num_modes={num_modes}")
    return modes
