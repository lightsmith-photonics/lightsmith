"""Closed-form integrals of the static kernel 1/R over flat triangles.

For an observation point r and a triangle T with unit normal n (Wilton et al.,
IEEE TAP 32, 276 (1984); Graglia, IEEE TAP 41, 1448 (1993)):

    I0   = int_T 1/R dS'
    Irho = int_T (r' - rho)/R dS'          (rho: projection of r onto the plane of T)
    grad = int_T grad_r (1/R) dS' = -int_T (r - r')/R^3 dS'

Each is a sum over the three edges of T. These are k-independent, so they are
evaluated once per mesh (NumPy, at set-up) and reused for every wavelength and
every medium.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-30


def static_integrals(r: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """I0 (...), Irho (..., 3), grad (..., 3) for points r (..., 3) and triangles tri (..., 3, 3).

    `r` and `tri` broadcast against each other (tri's leading dims with r's).
    """
    v0, v1, v2 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    nrm = np.cross(v1 - v0, v2 - v0)
    n = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)
    d = np.sum((r - v0) * n, axis=-1)
    # in-plane points (to round-off): principal value, no solid-angle jump
    d = np.where(np.abs(d) <= 1e-10 * np.sqrt(np.linalg.norm(nrm, axis=-1)), 0.0, d)
    rho = r - d[..., None] * n
    ad = np.abs(d)
    i0 = 0.0
    irho = 0.0
    f_sum = 0.0
    beta = 0.0
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        e = b - a
        le = np.linalg.norm(e, axis=-1, keepdims=True)
        lh = e / le
        u = np.cross(lh, n)  # outward in-plane edge normal
        lp = np.sum((b - rho) * lh, axis=-1)
        lm = np.sum((a - rho) * lh, axis=-1)
        t0 = np.sum((a - rho) * u, axis=-1)  # signed distance from rho to the edge line (>0 inside)
        rp = np.linalg.norm(r - b, axis=-1)
        rm = np.linalg.norm(r - a, axis=-1)
        r02 = t0**2 + d**2
        # ln((R+ + l+)/(R- + l-)), in the cancellation-free form for each side of the edge
        fwd = np.log((rp + lp + _TINY) / (rm + lm + _TINY))
        bwd = np.log((rm - lm + _TINY) / (rp - lp + _TINY))
        f = np.where(lp + lm >= 0, fwd, bwd)
        bk = (np.arctan2(t0 * lp, r02 + ad * rp + _TINY) - np.arctan2(t0 * lm, r02 + ad * rm + _TINY))
        i0 = i0 + t0 * f - ad * bk
        irho = irho + 0.5 * u * (r02 * f + lp * rp - lm * rm)[..., None]
        f_sum = f_sum + u * f[..., None]
        beta = beta + bk
    grad = -f_sum - (np.sign(d) * beta)[..., None] * n
    return i0, irho, grad
