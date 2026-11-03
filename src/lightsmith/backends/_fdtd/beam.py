"""Focused Gaussian beam (``spec.GaussianBeam``) as a sum of TFSF plane waves.

The beam is its angular spectrum, A(k_t) = exp(-(k_t w0 / 2)^2) over the transverse wavevector k_t,
discretised at fixed directions: Gauss-Legendre rings in k_t on [0, k_t,max] (A(k_t,max) = 1e-4, or a
numerical aperture of 0.95 if the waist is tighter than that allows) and uniform azimuths on each ring.
Ring and azimuth counts follow the largest transverse distance rho from the focus inside the TFSF box, so
the sampled spectrum reproduces the beam there (k_t rho / 2 + 6 rings, 2 k_t rho + 8 azimuths per ring).
Each direction is one plane wave fed from tables of its exact discrete field (``layered.py`` with a uniform
background, so the TFSF box does not leak), delayed to reach the focus in phase with the others and scaled
by the quadrature weight; its polarisation is the axis polarisation rotated with the direction.

Weights are fixed, not per frequency, so every frequency sees the same set of angles: the beam of a lens
of fixed NA, whose waist scales with the wavelength (waist_nm at waist_wavelength_nm).
"""

from __future__ import annotations

import itertools

import numpy as np

from .engine import DFTMonitor
from .layered import ObliqueLayeredPlaneWave, _View, read_tables
from .setup import flux

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

    Each component is an ObliqueLayeredPlaneWave in the uniform medium, delayed so that it reaches the focus
    t_focus after its unshifted pulse reaches its reference point, the same t_focus for all: the components
    arrive there in phase. Their tables (scaled by the weights) are concatenated, and the corrections of
    every component on the same face slab are read in one gather-and-sum. The view holds every component's
    incident E at its reference point; with the exact unit solutions, that gives each component's amplitude,
    hence the beam at the focus and its background field anywhere in the tables' range."""

    def __init__(self, comps, *, focus, shape, origin, spacing, dt, lo, hi, inv_eps, host, omega_c, omega_half,
                 max_steps, omega_min):
        self.focus = np.asarray(focus, dtype=float)
        sp = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
        lo, hi, origin = np.asarray(lo), np.asarray(hi), np.asarray(origin, dtype=float)
        # no reference point is farther from the focus than the corners of the box widened by 2 cells
        corners = origin + np.array(list(itertools.product(*zip(lo - 2, hi + 2)))) * sp
        t_focus = np.sqrt(host.eps_inf) * float(np.max(np.linalg.norm(corners - self.focus, axis=1)))
        parts = [ObliqueLayeredPlaneWave(shape=shape, origin=origin, spacing=spacing, dt=dt, lo=lo, hi=hi,
                                         khat=khat, ehat=ehat, inv_eps=inv_eps, host=host, omega_c=omega_c,
                                         omega_half=omega_half, max_steps=max_steps, omega_min=omega_min,
                                         focus=self.focus, t_focus=t_focus)
                 for khat, ehat, _ in comps]
        self.parts = parts
        self.khats = np.array([kh for kh, _, _ in comps])
        self.ehats = np.array([eh for _, eh, _ in comps])
        self.weights = np.array([w for _, _, w in comps])
        self.n = len(parts)
        self.m = parts[0].m
        self.t_passed = max(p.t_passed for p in parts)
        self.t_off = max(p.t_off for p in parts)
        off = np.concatenate([[0], np.cumsum([p.tables.size for p in parts])[:-1]])
        off_inc = np.concatenate([[0], np.cumsum([p.inc_table.size for p in parts])[:-1]])
        self.tables = np.concatenate([w * p.tables.reshape(-1) for p, w in zip(parts, self.weights)])
        self.inc_table = np.concatenate([p.inc_table.reshape(-1) for p in parts])
        self.cmax = np.array([[p.cmax] for p in parts])
        self.slabs = []
        for i, (kind, c, region, _, _, _, arr, shape_) in enumerate(parts[0].slabs):
            assert all(p.slabs[i][2] == region for p in parts)
            self.slabs.append((kind, c, region, np.stack([p.slabs[i][3] + o for p, o in zip(parts, off)]),
                               np.stack([p.slabs[i][4] for p in parts]), np.stack([p.slabs[i][5] for p in parts]),
                               arr, shape_))
        self.inc = (np.stack([p.inc[0] + o for p, o in zip(parts, off_inc)]), np.stack([p.inc[1] for p in parts]),
                    np.stack([p.inc[2] for p in parts]))
        for p in parts:  # keep only what the analysis needs (the 1D solutions)
            del p.tables, p.inc_table
            p.slabs = None
        self.view = _View()

    def bind(self, eng):
        self._T, self._Tinc = eng.asarray(self.tables), eng.asarray(self.inc_table)
        del self.tables, self.inc_table
        self._cmax = eng.index(self.cmax)
        self._slabs = [(kind, c, region, eng.index(base), eng.index(i0), eng.asarray(u0), eng.asarray(arr), shape_)
                       for kind, c, region, base, i0, u0, arr, shape_ in self.slabs]
        self._inc = tuple(f(a) for f, a in zip((eng.index, eng.index, eng.asarray), self.inc))
        self.view.E_flat = eng.zeros((self.n * 3,))
        self.view.H_flat = self.view.E_flat

    def _apply(self, eng, kind: str):
        target = eng.E if kind == "E" else eng.H
        xp = eng.xp
        for k, c, region, base, i0, u0, arr, shape_ in self._slabs:
            if k == kind:
                vals = xp.sum(read_tables(xp, self._T, eng.n, self.m, self._cmax, base, i0, u0), axis=0)
                target[c][region] += arr * xp.reshape(vals, shape_)

    def correct_H(self, eng):
        self._apply(eng, "H")

    def correct_E(self, eng):
        self._apply(eng, "E")

    def advance_H(self, eng):
        pass

    def advance_E(self, eng):
        vals = read_tables(eng.xp, self._Tinc, eng.n, self.m, self._cmax, *self._inc)
        self.view.E_flat[...] = eng.xp.reshape(vals, (-1,))

    def monitor(self, omega) -> DFTMonitor:
        """Every component's incident E at its reference point, in one monitor."""
        return DFTMonitor(np.asarray(omega, dtype=float), np.arange(self.n * 3), source=self.view)

    def amplitudes(self, mon: DFTMonitor, omega) -> np.ndarray:
        """(frequencies, components): each component's amplitude against its unit solution."""
        omega = np.asarray(omega, dtype=float)
        e = mon.results()[0].reshape(len(omega), self.n, 3)
        amp = np.zeros((len(omega), self.n), dtype=complex)
        for j, p in enumerate(self.parts):
            unit = p.incident_reference(omega)
            amp[:, j] = np.sum(e[:, j] * np.conj(unit), axis=1) / np.sum(np.abs(unit) ** 2, axis=1)
        return amp

    def continuum(self, amp: np.ndarray, omega, r: np.ndarray) -> np.ndarray:
        """E of the beam with component amplitudes amp (len(omega), components) at points r (..., 3) in grid
        coordinates (nm), as continuum plane waves from each reference point: (len(omega), ..., 3)."""
        omega, r = np.asarray(omega, dtype=float), np.asarray(r, dtype=float)
        n_b = self.parts[0].n_inc
        out = np.zeros((len(omega),) + r.shape, dtype=complex)
        for j, (p, kh, eh, w) in enumerate(zip(self.parts, self.khats, self.ehats, self.weights)):
            ph = np.exp(1j * n_b * omega.reshape((-1,) + (1,) * (r.ndim - 1)) * ((r - p.r_ref) @ kh)[None])
            out += (w * amp[:, j]).reshape((-1,) + (1,) * (r.ndim - 1))[..., None] * ph[..., None] * eh
        return out

    def focus_field(self, amp: np.ndarray, omega) -> np.ndarray:
        """Incident E at the focus, (frequencies, 3), for the component amplitudes amp."""
        return self.continuum(amp, omega, self.focus[None, :])[:, 0]

    def background(self, amp: np.ndarray, omega, flat: np.ndarray, kind: str = "E") -> np.ndarray:
        """The beam's E or H at flat grid indices, (len(omega), n), for component amplitudes amp
        (len(omega), components); NaN beyond any component's table range."""
        out = 0
        for j, (p, w) in enumerate(zip(self.parts, self.weights)):
            out = out + w * amp[:, j:j + 1] * p.background(omega, flat, kind)
        return out

    def box_flux(self, amp: np.ndarray, omega, pts) -> np.ndarray:
        """Net outward flux of the beam alone through a flux_points box (the grid's averaged-H flux of a sum
        of plane waves need not be exactly conserved)."""
        e = self.background(amp, omega, pts[0], "E")
        h = 0.5 * (self.background(amp, omega, pts[1], "H") + self.background(amp, omega, pts[2], "H"))
        return flux(e, h, pts[3])
