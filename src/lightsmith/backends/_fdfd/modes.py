"""1D slab mode solver on the FDFD discretisation, plus the analytic slab dispersion.

Port modes are eigenvectors of the *discrete* transverse operator of the 2D
grid, so a port mode is an exact solution of the discrete 2D problem in a
straight, uniform section. This makes mode injection unidirectional and modal
decomposition exact up to round-off (no grid-dispersion mismatch).

With lengths scaled by dl and K the transverse second difference
(``(K h)_j = a_{j+1}(h_{j+1}-h_j) - a_j(h_j-h_{j-1})``, a = 1 for TM, 1/eps_edge for TE):

    TM (Ez):  (K + (k0 dl)^2 diag(eps)) e = lam e
    TE (Hz):  diag(eps) (K + (k0 dl)^2) h = lam h

and lam = (2 sin(beta dl / 2))^2 gives the discrete propagation constant beta.

Unit power: modes are scaled so the discrete Poynting flux sin(beta dl) * sum(w |m|^2)
is 1, with weight w = 1 (TM) or 1/eps (TE). Modes are orthogonal under the
unconjugated, w-weighted product, which is what modal decomposition uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SlabMode:
    neff: complex  # beta / k0 with the discrete-dispersion correction
    beta_dl: complex  # discrete propagation constant times the cell size
    profile: np.ndarray  # (n,) unit-power profile on the window nodes
    weight: np.ndarray  # (n,) projection weight
    guided: bool

    @property
    def projector(self) -> np.ndarray:
        """Vector v with v @ field = modal amplitude (unconjugated, weighted)."""
        wm = self.weight * self.profile
        return wm / np.sum(wm * self.profile)


def _second_difference(a_edge: np.ndarray) -> np.ndarray:
    n = len(a_edge) - 1
    K = np.zeros((n, n), dtype=complex)
    idx = np.arange(n)
    K[idx, idx] = -(a_edge[1:] + a_edge[:-1])
    K[idx[:-1], idx[:-1] + 1] = a_edge[1:-1]
    K[idx[1:], idx[1:] - 1] = a_edge[1:-1]
    return K


def solve_slab_modes(eps_node: np.ndarray, eps_edge: np.ndarray | None, pol: str, k0dl: float,
                     num_modes: int) -> list[SlabMode]:
    """Modes of a 1D permittivity profile on `n` nodes with Dirichlet ends.

    eps_node: (n,) permittivity at the nodes. eps_edge: (n+1,) permittivity at the
    edges between nodes (incl. the two outer edges), used for TE; None = from nodes.
    Returns modes sorted by decreasing Re(neff).
    """
    eps_node = np.asarray(eps_node, dtype=complex)
    n = len(eps_node)
    if n < 3:
        raise ValueError("port window must span at least 3 grid cells")
    if pol == "tm":
        a_edge = np.ones(n + 1, dtype=complex)
        M = _second_difference(a_edge) + k0dl**2 * np.diag(eps_node)
        weight = np.ones(n)
    elif pol == "te":
        if eps_edge is None:
            p = np.concatenate([eps_node[:1], eps_node, eps_node[-1:]])
            eps_edge = (p[:-1] + p[1:]) / 2
        a_edge = 1 / np.asarray(eps_edge, dtype=complex)
        M = eps_node[:, None] * (_second_difference(a_edge) + k0dl**2 * np.eye(n))
        weight = 1 / eps_node
    else:
        raise ValueError(f"unknown polarization {pol!r}")

    lossless = np.all(np.abs(eps_node.imag) == 0) and (pol == "tm" or np.all(np.abs(np.imag(a_edge)) == 0))
    if lossless:
        # Real symmetric tridiagonal form: TM directly; TE via h = D^(1/2) g with D = diag(eps).
        import scipy.linalg

        k = min(num_modes, n)
        if pol == "tm":
            diag, off, scale = np.diag(M).real, np.diag(M, 1).real, np.ones(n)
        else:
            sq = np.sqrt(eps_node.real)
            S = _second_difference(a_edge).real + k0dl**2 * np.eye(n)
            diag, off, scale = sq**2 * np.diag(S), sq[:-1] * sq[1:] * np.diag(S, 1), sq
        lam, g = scipy.linalg.eigh_tridiagonal(diag, off, select="i", select_range=(n - k, n - 1))
        lam, vec = lam.astype(complex), (scale[:, None] * g).astype(complex)
    elif n <= 2500:
        import scipy.linalg

        lam, vec = scipy.linalg.eig(M)
    else:  # pragma: no cover - large windows
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigs

        lam, vec = eigs(sp.csr_matrix(M), k=num_modes + 4, sigma=k0dl**2 * eps_node.real.max())
    order = np.argsort(-lam.real)
    lam, vec = lam[order], vec[:, order]

    eps_clad = max(eps_node[0].real, eps_node[-1].real)
    modes = []
    for k in range(min(num_modes, len(lam))):
        if lam[k].real <= 0:
            raise ValueError(f"only {k} propagating mode(s) exist at this port; requested num_modes={num_modes}")
        half = np.sqrt(lam[k] + 0j) / 2
        if half.real >= 1:
            raise ValueError("grid too coarse for this mode (beyond the Nyquist limit); reduce dl_nm")
        beta_dl = 2 * np.arcsin(half)
        if beta_dl.imag < 0:
            beta_dl = np.conj(beta_dl) if abs(beta_dl.imag) < 1e-12 else beta_dl
        m = vec[:, k].astype(complex)
        i = int(np.argmax(np.abs(m)))
        m = m * (abs(m[i]) / m[i])
        if np.allclose(m.imag, 0, atol=1e-12 * np.abs(m).max()):
            m = m.real.astype(complex)
        norm = np.sin(beta_dl) * np.sum(weight * m * m)
        m = m / np.sqrt(norm)
        neff = beta_dl / k0dl
        edge = max(abs(m[0]), abs(m[-1])) / np.abs(m).max()
        guided = bool(neff.real**2 > eps_clad and edge < 1e-2)
        modes.append(SlabMode(neff=complex(neff), beta_dl=complex(beta_dl), profile=m, weight=weight, guided=guided))
    return modes


def slab_neff_analytic(n_core: float, n_clad: float, width_nm: float, wavelength_nm: float, pol: str = "te",
                       order: int = 0) -> float:
    """Effective index of a symmetric slab (core between two half-spaces of cladding).

    Uses this module's 2D naming: ``"tm"`` = E along z (tangential to the slab
    walls, the classic slab-TE mode), ``"te"`` = E in the plane (the classic
    slab-TM mode). Solves k w/2 = atan(rho g/k) + order*pi/2 by bisection,
    with rho = 1 (tm) or n_core^2/n_clad^2 (te).
    """
    k0 = 2 * np.pi / wavelength_nm
    rho = 1.0 if pol == "tm" else n_core**2 / n_clad**2

    def f(neff):
        kap = k0 * np.sqrt(n_core**2 - neff**2)
        gam = k0 * np.sqrt(neff**2 - n_clad**2)
        return kap * width_nm / 2 - np.arctan(rho * gam / kap) - order * np.pi / 2

    lo, hi = n_clad + 1e-12, n_core - 1e-12
    if f(lo) * f(hi) > 0:
        raise ValueError(f"slab has no guided mode of order {order}")
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
