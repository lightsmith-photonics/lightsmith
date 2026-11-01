"""Padé approximants of the one-way propagator and their Crank-Nicolson factorisation.

With the envelope phi = psi exp(-i beta_ref w) and X = (M - n_ref^2) / n_ref^2, the
one-way equation d psi/dw = i k0 sqrt(M) psi becomes

    d phi/dw = i beta_ref R(X) phi,    R(X) = sqrt(1 + X) - 1.

R is replaced by the Padé recursion R_{k+1} = (X/2) / (1 + R_k/2), R_0 = 0
(Hadley, Opt. Lett. 17, 1426 (1992)): R_1 = X/2 is the paraxial (Fresnel)
equation and R_{2m} the (m, m) Padé approximant. Every R_{2m} is a sum of
terms a X / (1 + b X) with a, b > 0, so Im R >= 0 whenever Im X >= 0 and the
scheme stays dissipative in PMLs and lossy media. Optionally the branch cut is
rotated by alpha (Milinazzo, Zala & Brooke, J. Acoust. Soc. Am. 101, 760
(1997)), which damps evanescent components instead of propagating them.

Crank-Nicolson, D (phi+ - phi) = i beta_ref dw N (phi+ + phi) / 2 with R = N / D, gives

    (D - tau N) phi+ = (D + tau N) phi,    tau = i beta_ref dw / 2,

and both sides factor into first-order polynomials in X (Hadley's multistep
method), so one step is a product of `order` shifted solves
(1 - X / b_j)^-1 (1 - X / a_j), each with the sparsity of M.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from numpy.polynomial import polynomial as P


def pade_rational(order: int, rotation_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Ascending coefficients (N, D) of R(X) ~ sqrt(1 + X) - 1; order 0 = paraxial, m = Padé (m, m)."""
    steps = 1 if order == 0 else 2 * order
    n, d = np.zeros(1, complex), np.ones(1, complex)
    for _ in range(steps):
        n, d = P.polymul([0, 0.5], d), P.polyadd(d, 0.5 * n)
    if rotation_deg == 0:
        return n, d
    # sqrt(1 + X) = e^{i a/2} sqrt(1 + Y), Y = e^{-i a}(1 + X) - 1, then R(X) = e^{i a/2}(1 + N(Y)/D(Y)) - 1.
    a = math.radians(rotation_deg)
    lin = np.array([cmath.exp(-1j * a) - 1, cmath.exp(-1j * a)])  # Y as a polynomial in X

    def compose(c):
        out = np.zeros(1, complex)
        for coef in c[::-1]:
            out = P.polyadd(P.polymul(out, lin), [coef])
        return out

    ny, dy = compose(n), compose(d)
    rot = cmath.exp(0.5j * a)
    return P.polysub(rot * P.polyadd(dy, ny), dy), dy


def _roots(c: np.ndarray) -> tuple[complex, np.ndarray]:
    """c(X) = c(0) prod(1 - X / r_j): returns (c(0), r)."""
    c = np.trim_zeros(np.asarray(c, complex), "b")
    nz = np.abs(c) > 1e-14 * np.abs(c).max()
    c = c[: np.flatnonzero(nz)[-1] + 1]
    if len(c) == 1:
        return complex(c[0]), np.zeros(0, complex)
    return complex(c[0]), P.polyroots(c)


def cn_factors(order: int, beta_ref_dw: float, rotation_deg: float = 0.0
               ) -> tuple[complex, list[tuple[complex | None, complex | None]]]:
    """One Crank-Nicolson step as gain * prod_j (1 - X/b_j)^-1 (1 - X/a_j); pairs (a_j, b_j), None = identity."""
    n, d = pade_rational(order, rotation_deg)
    tau = 0.5j * beta_ref_dw
    m = max(len(n), len(d))
    n, d = np.pad(n, (0, m - len(n))), np.pad(d, (0, m - len(d)))
    a0, ra = _roots(d + tau * n)
    b0, rb = _roots(d - tau * n)
    ra, rb = list(ra), list(rb)
    k = max(len(ra), len(rb))
    ra += [None] * (k - len(ra))
    rb += [None] * (k - len(rb))
    return a0 / b0, list(zip(ra, rb))


def symbol(order: int, beta_ref_dw: float, x: np.ndarray, rotation_deg: float = 0.0) -> np.ndarray:
    """Amplification of one step for eigenvalue X (for tests and step-size checks)."""
    g, pairs = cn_factors(order, beta_ref_dw, rotation_deg)
    out = np.full(np.shape(x), g, complex)
    for a, b in pairs:
        if a is not None:
            out *= 1 - x / a
        if b is not None:
            out /= 1 - x / b
    return out
