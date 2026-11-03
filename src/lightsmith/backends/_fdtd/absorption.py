"""Power absorbed inside the particles, from the D-form nodes' own fields (used over an absorbing substrate).

Per frequency the Yee update gives curl H = -i Omega D at every E node exactly (Omega = 2 sin(omega dt / 2) / dt),
so the grid absorbs (Omega / 2) Im(E_p^* D_p) dV at node p, and over a set of nodes that the D -> E map does not
couple to lossy nodes outside it, the sum is the net discrete Poynting inflow into that set. Lossy media are always
D-form (ADE poles), so it is enough to monitor E and D on the dispersive nodes whose pixel (for the vertex tensor:
either of its two vertices) holds a particle medium; the vertex tensor couples only nodes of one vertex, so that set
is closed. A node whose pixel also holds the substrate (only where a particle touches it) is split by the media's
share of fraction x Im eps.
"""

from __future__ import annotations

import numpy as np

from .engine import DFTMonitor, TensorMesh
from .layered import discrete_eps


class _GroupD:
    """E_flat for DFTMonitor(source=...): the D array of the engine's dispersive group k (bound after the build)."""

    H_flat = None

    def __init__(self, k: int):
        self.k, self.eng = k, None

    @property
    def E_flat(self):
        return self.eng.groups[self.k].D


def node_fractions(g) -> np.ndarray:
    """(S, n) medium fractions of each node of a DispersiveGroup or TensorMesh (for the tensor: the mean of its
    two slots, each its vertex's fractions or its plain medium)."""
    if not isinstance(g, TensorMesh):
        return np.asarray(g.frac, dtype=float)
    S, nN = len(g.media), len(g.node_comp)
    out = np.zeros((S, nN))
    for k in range(2):
        v = g.slot_vert[:, k]
        t = v >= 0
        out[:, t] += 0.5 * g.frac[:, v[t]]
        out[g.slot_medium[~t, k], np.nonzero(~t)[0]] += 0.5
    return out


class ParticleAbsorption:
    """DFT monitors of E and D on the particles' lossy nodes, and the power they absorb."""

    def __init__(self, groups, particle_media, omega, dt: float):
        self.omega, self.dt = np.asarray(omega, dtype=float), float(dt)
        self.parts = []
        for k, g in enumerate(groups):
            is_p = np.array([any(m is pm for pm in particle_media) for m in g.media])
            if not is_p.any():
                continue
            F = node_fractions(g)
            sel = np.nonzero(F[is_p].sum(axis=0) > 0)[0]
            if not len(sel):
                continue
            im = np.array([np.maximum(discrete_eps(m, self.omega, dt).imag, 0.0) for m in g.media])  # (S, n_omega)
            num = im[is_p].T @ F[is_p][:, sel]
            den = im.T @ F[:, sel]
            w = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)  # (n_omega, n): the particles' share
            if not np.any(w > 0):
                continue
            src = _GroupD(k)
            idx = np.asarray(g.idx, dtype=np.int64)[sel]
            self.parts.append((src, DFTMonitor(self.omega, idx), DFTMonitor(self.omega, sel, source=src), w))

    @property
    def monitors(self) -> list[DFTMonitor]:
        return [m for _, me, md, _ in self.parts for m in (me, md)]

    @property
    def nodes(self) -> int:
        return sum(len(me.e_idx) for _, me, _, _ in self.parts)

    def bind(self, eng):
        for src, *_ in self.parts:
            src.eng = eng

    def power(self, dV: float) -> np.ndarray:
        """Absorbed power per frequency, in the units of the box fluxes."""
        om = 2 * np.sin(self.omega * self.dt / 2) / self.dt
        p = np.zeros(len(self.omega))
        for _, me, md, w in self.parts:
            e, d = me.results()[0], md.results()[0]
            p += np.sum(w * np.imag(np.conj(e) * d), axis=1)
        return 0.5 * om * p * dV
