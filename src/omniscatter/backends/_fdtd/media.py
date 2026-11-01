"""Material models as FDTD auxiliary-differential-equation (ADE) media.

Normalised units throughout the engine: lengths in nm, c = eps0 = mu0 = 1, so
time is in nm/c and angular frequency in rad per (nm/c). A photon energy E (eV)
is omega = E / (hbar c) with hbar c = 197.327 eV nm.

Every dispersive medium is eps(w) = eps_inf + sum_k sigma_k / (w0_k^2 - w^2 - i g_k w)
(exp(-i w t)), i.e. each pole obeys  P'' + g P' + w0^2 P = sigma E.
Drude is the pole with w0 = 0 and sigma = wp^2.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

from ...materials import (ConstantIndex, ConstantPermittivity, Drude, DrudeLorentz, LibraryMaterial,
                          Tabulated)

HBARC_EV_NM = 197.3269804


@dataclass(frozen=True)
class Pole:
    sigma: float
    omega0: float
    gamma: float


@dataclass(frozen=True)
class Medium:
    eps_inf: float
    poles: tuple[Pole, ...] = ()
    label: str = ""

    @property
    def dispersive(self) -> bool:
        return bool(self.poles)

    def eps(self, omega: np.ndarray) -> np.ndarray:
        w = np.asarray(omega, dtype=float)
        e = self.eps_inf + 0j * w
        for p in self.poles:
            e = e + p.sigma / (p.omega0**2 - w**2 - 1j * p.gamma * w)
        return e

    def describe(self) -> dict:
        return {"label": self.label, "eps_inf": self.eps_inf,
                "poles": [{"sigma": p.sigma, "omega0": p.omega0, "gamma": p.gamma} for p in self.poles]}


def needs_fit(material) -> bool:
    return isinstance(material, (Tabulated, LibraryMaterial))


def fit_module():
    """omniscatter.materials_fit, or None when this installation lacks it."""
    try:
        return importlib.import_module("omniscatter.materials_fit")
    except ModuleNotFoundError as e:
        if e.name not in ("omniscatter.materials_fit", "scipy", "scipy.optimize"):
            raise
        return None


def _from_drude_lorentz(m: DrudeLorentz, label: str) -> Medium:
    poles = []
    if m.wp_ev > 0:
        poles.append(Pole((m.wp_ev / HBARC_EV_NM) ** 2, 0.0, m.gamma_ev / HBARC_EV_NM))
    for p in m.poles:
        w0 = p.omega0_ev / HBARC_EV_NM
        poles.append(Pole(p.delta_eps * w0**2, w0, p.gamma_ev / HBARC_EV_NM))
    return Medium(float(m.eps_inf), tuple(poles), label)


def _matched_constant(eps: complex, omega_c: float, label: str) -> Medium:
    """A constant complex eps as a Drude term matched exactly at omega_c.

    Lossless eps >= 1 is a plain dielectric. Otherwise eps_inf = max(1, eps' + eps''/20):
    for eps' >= 1 this is a conductivity-like Drude term (gamma ~ 20 omega_c, Im eps ~ 1/omega),
    for eps' < 1 a genuine Drude metal."""
    er, ei = eps.real, eps.imag
    if ei == 0 and er >= 1:
        return Medium(er, (), label)
    eps_inf = max(1.0, er + ei / 20)
    delta = eps_inf - er
    gamma = omega_c * ei / delta
    wp2 = omega_c**2 * (delta**2 + ei**2) / delta
    return Medium(eps_inf, (Pole(wp2, 0.0, gamma),), label)


def to_medium(material, wavelengths_nm: np.ndarray, warnings: list[str]) -> Medium:
    """Convert one omniscatter material to an ADE medium valid over `wavelengths_nm`."""
    wl = np.asarray(wavelengths_nm, dtype=float)
    label = getattr(material, "kind", type(material).__name__)
    if isinstance(material, Drude):
        return _from_drude_lorentz(DrudeLorentz(eps_inf=material.eps_inf, wp_ev=material.wp_ev,
                                                gamma_ev=material.gamma_ev), label)
    if isinstance(material, DrudeLorentz):
        m = _from_drude_lorentz(material, label)
        if any(p.sigma < 0 for p in m.poles):
            warnings.append("DrudeLorentz with negative oscillator strength: the FDTD update may be unstable")
        return m
    if isinstance(material, (ConstantIndex, ConstantPermittivity)):
        eps = complex(material.eps(np.array([wl[0]]))[0])
        omega = 2 * np.pi / wl
        omega_c = 0.5 * (omega.min() + omega.max())
        m = _matched_constant(eps, omega_c, label)
        if m.dispersive and wl.size > 1:
            dev = float(np.max(np.abs(m.eps(omega) - eps)) / abs(eps))
            if dev > 1e-3:
                warnings.append(f"{label} eps={eps:.4g} is not causal as a constant; FDTD models it as a Drude term "
                                f"exact at {2 * np.pi / omega_c:.0f} nm (max deviation {dev:.1%} over the band). "
                                "Use a Drude/DrudeLorentz material for broadband accuracy.")
        return m
    if needs_fit(material):
        mod = fit_module()
        if mod is None:
            raise ValueError(
                "FDTD needs a Drude-Lorentz fit for tabulated/library materials, but omniscatter.materials_fit "
                "is not available in this installation. Give the material as Drude or DrudeLorentz instead.")
        dl, info = mod.fit_drude_lorentz(material, wl, return_info=True)
        name = getattr(material, "name", None) or "tabulated"
        warnings.append(f"{name}: fitted with {info.n_poles} Lorentz pole(s){' + Drude' if info.drude else ''}, "
                        f"RMS relative error {info.rms_rel_error:.2%} over {wl.min():g}-{wl.max():g} nm")
        return _from_drude_lorentz(dl, f"{label}:{name} (fit)")
    raise ValueError(f"FDTD does not support material kind {label!r}")
