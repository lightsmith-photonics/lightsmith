"""Focused Gaussian beam (``spec.GaussianBeam``) as a sum of TFSF plane waves.

The beam is its angular spectrum, A(k_t) = exp(-(k_t w0 / 2)^2) over the transverse wavevector k_t,
discretised at fixed directions: Gauss-Legendre rings in k_t on [0, k_t,max] (A(k_t,max) = 1e-4, or a
numerical aperture of 0.95 if the waist is tighter than that allows) and uniform azimuths on each ring.
Ring and azimuth counts follow the largest transverse distance rho from the focus inside the TFSF box, so
the sampled spectrum reproduces the beam there (k_t rho / 2 + 6 rings, 2 k_t rho + 8 azimuths per ring).
Each direction is one ``TFSFPlaneWave`` (its own 1D feed, phase reference at the focus) whose pulse is
scaled by the quadrature weight; its polarisation is the axis polarisation rotated with the direction.

Weights are fixed, not per frequency, so every frequency sees the same set of angles: the beam of a lens
of fixed NA, whose waist scales with the wavelength (waist_nm at waist_wavelength_nm).
"""

from __future__ import annotations

import numpy as np

from .engine import DFTMonitor
from .tfsf import TFSFPlaneWave

A_MIN = 1e-4  # angular spectrum cut-off
KT_CUT = 2 * np.sqrt(np.log(1 / A_MIN))  # k_t w0 at the cut-off (6.07)
NA_MAX = 0.95


def _rotate(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """v rotated by the rotation that takes unit vector a to unit vector b (about a x b)."""
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(a @ b)
    if s < 1e-12:
        return v.copy()
    u = axis / s
    return v * c + np.cross(u, v) * s + u * (u @ v) * (1 - c)


def plane_wave_components(beam, medium_index: float, wl_ref: float, rho_max: float):
    """[(khat, ehat, weight)] with weights summing to 1, and info about the discretisation."""
    k0, e0 = beam.unit_vectors()
    v0 = np.cross(k0, e0)
    k = 2 * np.pi * medium_index / wl_ref
    kt_max = KT_CUT / beam.waist_nm
    truncated = kt_max > NA_MAX * k
    kt_max = min(kt_max, NA_MAX * k)
    n_rings = int(np.ceil(kt_max * rho_max / 2)) + 6
    x, wgl = np.polynomial.legendre.leggauss(n_rings)
    kts, wrs = 0.5 * kt_max * (x + 1), 0.5 * kt_max * wgl
    comps = []
    for kt, wr in zip(kts, wrs):
        n_phi = int(np.ceil(2 * kt * rho_max)) + 8
        a = np.exp(-((kt * beam.waist_nm / 2) ** 2)) * wr * kt * 2 * np.pi / n_phi
        kz = np.sqrt(k**2 - kt**2)
        for phi in 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi:
            khat = (kt * (np.cos(phi) * e0 + np.sin(phi) * v0) + kz * k0) / k
            comps.append((khat, _rotate(e0, k0, khat), a))
    total = sum(w for _, _, w in comps)
    comps = [(kh, eh, w / total) for kh, eh, w in comps]
    return comps, {"plane_waves": len(comps), "rings": n_rings, "kt_max_per_k": kt_max / k,
                   "truncated": bool(truncated), "waist_wavelength_nm": wl_ref}


def beam_field(comps, r: np.ndarray, k: float) -> np.ndarray:
    """E of the discretised beam at points r (n, 3) relative to the focus, for wavenumber k (exp(-i w t))."""
    out = np.zeros((len(r), 3), dtype=complex)
    for khat, ehat, w in comps:
        out += w * np.exp(1j * k * (r @ khat))[:, None] * ehat
    return out


class BeamSource:
    """Engine source for all beam components at once.

    Each component is set up as a TFSFPlaneWave (its 1D feed and face corrections, phase reference at the
    focus); their 1D grids are then stacked into one (N, M) array stepped together, and the corrections of
    every component on the same face slab are merged into one gather-and-sum, so a step costs one set of
    slab updates whatever the number of plane waves (one TFSFPlaneWave each was ~50x slower)."""

    def __init__(self, comps, **tfsf):
        # every component must reach the focus (s = 0, at m0 on its 1D grid) at the same time, or the
        # components are not in phase there: put s = 0 equally far from the 1D source cell in all of them
        parts = [TFSFPlaneWave(khat=khat, ehat=ehat, **tfsf) for khat, ehat, _ in comps]
        m0 = max(p.m0 for p in parts)
        parts = [p if p.m0 == m0 else TFSFPlaneWave(khat=khat, ehat=ehat, m0=m0, **tfsf)
                 for p, (khat, ehat, _) in zip(parts, comps)]
        self.ehats = [eh for _, eh, _ in comps]
        self.weights = np.array([w for _, _, w in comps])
        self.t_passed = max(p.t_passed for p in parts)
        n, M = len(parts), max(p.M for p in parts)
        self.n, self.M = n, M
        # stacked 1D coefficients; cells past a component's own grid stay zero (ca = da = 1, cb = db = 0)
        ca, cb, da, db = np.ones((n, M)), np.zeros((n, M)), np.ones((n, M)), np.zeros((n, M))
        for j, p in enumerate(parts):
            ca[j, :p.M], cb[j, :p.M], da[j, :p.M], db[j, :p.M] = p.ca, p.cb, p.da, p.db
            cb[j, p.M - 1:] = 0  # a shorter grid keeps its PEC end
        self.ca, self.cb, self.da, self.db = ca[:, 1:-1], cb[:, 1:-1], da[:, :-1], db[:, :-1]
        if len({p.m_src for p in parts}) != 1:
            raise ValueError("beam components must share the 1D source cell")
        self.m_src = parts[0].m_src
        self.wave = parts[0].wave
        self.m0 = np.array([p.m0 for p in parts])
        # merge slabs: key -> (kind, comp, region, shape, flat indices (P, S), weights (P, S), coefs (P,))
        groups: dict[tuple, list] = {}
        for j, p in enumerate(parts):
            for kind, c, region, coef, i0, w, shape_ in p.slabs:
                key = (kind, c, tuple((r.start, r.stop) for r in region))
                groups.setdefault(key, [kind, c, region, shape_, [], [], []])
                g = groups[key]
                g[4].append(j * M + i0)
                g[5].append(np.zeros(len(i0)) if w is None else w)
                g[6].append(coef)
        self.slabs = [(kind, c, region, shape_, np.stack(i0s), np.stack(ws), np.asarray(cs))
                      for kind, c, region, shape_, i0s, ws, cs in groups.values()]

    def probe(self):
        """Flat indices (e, h1, h2) of every component's incident-field probe in the stacked 1D arrays."""
        j = np.arange(self.n) * self.M
        return j + self.m0, j + self.m0 - 1, j + self.m0

    def bind(self, eng):
        a = eng.asarray
        self.E1, self.H1 = eng.zeros((self.n, self.M)), eng.zeros((self.n, self.M))
        self.E_flat, self.H_flat = eng.xp.reshape(self.E1, (-1,)), eng.xp.reshape(self.H1, (-1,))
        self._ca, self._cb, self._da, self._db = a(self.ca), a(self.cb), a(self.da), a(self.db)
        self._wave, self._w = a(self.wave), a(self.weights[:, None])
        self._slabs = [(kind, c, region, shape_, eng.index(i0), a(w), a(cs[:, None]))
                       for kind, c, region, shape_, i0, w, cs in self.slabs]

    def _apply(self, eng, kind: str):
        target = eng.E if kind == "E" else eng.H
        src = self.H_flat if kind == "E" else self.E_flat
        for k, c, region, shape_, i0, w, cs in self._slabs:
            if k != kind:
                continue
            lo = src[i0]
            vals = eng.xp.sum(cs * (lo + w * (src[i0 + 1] - lo)), axis=0)
            target[c][region] += eng.xp.reshape(vals, shape_)

    def correct_H(self, eng):
        self._apply(eng, "H")

    def correct_E(self, eng):
        self._apply(eng, "E")

    def advance_H(self, eng):
        E, H = self.E1, self.H1
        H[:, :-1] *= self._da
        H[:, :-1] -= self._db * (E[:, 1:] - E[:, :-1])

    def advance_E(self, eng):
        E, H = self.E1, self.H1
        E[:, 1:-1] *= self._ca
        E[:, 1:-1] -= self._cb * (H[:, 1:-1] - H[:, :-2])
        m = self.m_src
        E[:, m:m + 1] += self._w * self._wave[eng.n]

    def monitor(self, omega) -> DFTMonitor:
        """Incident-field probes at the focus, all components in one monitor."""
        return DFTMonitor(omega, *self.probe(), source=self)

    def focus_field(self, mon: DFTMonitor) -> np.ndarray:
        """Incident E at the focus, (frequencies, 3): each component's 1D amplitude times its polarisation."""
        e = mon.results()[0]  # (frequencies, components)
        return e @ np.asarray(self.ehats)
