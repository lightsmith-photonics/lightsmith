"""Quasistatic (dipole) approximation: Rayleigh spheres, coated spheres, Gans spheroids.

Valid for particles much smaller than the wavelength (size parameter x = k a << 1).
Polarizabilities follow Bohren & Huffman ch. 5 (alpha in volume units times 4 pi,
relative permittivity eps = eps_particle / n_medium^2):

    sphere           alpha = 4 pi a^3 (eps - 1) / (eps + 2)
    coated sphere    the same with the shell-averaged eps_eff (BH 5.36), applied
                     recursively for any number of shells
    spheroid (Gans)  alpha_i = V (eps - 1) / (1 + L_i (eps - 1)), V = 4/3 pi a^2 c,
                     with depolarisation factors L_x = L_y = (1 - L_z)/2

and, with k the wavenumber in the medium,

    C_abs = k Im(e.alpha.e),  C_sca = k^4 |alpha.e|^2 / (6 pi),  C_ext = C_abs + C_sca.

The `corrections` option adds radiative damping (``"radiative"``) or the full
modified long-wavelength approximation (``"mlwa"``: dynamic depolarisation plus
radiative damping; Meier & Wokaun, Opt. Lett. 8, 581 (1983); Kelly et al.,
J. Phys. Chem. B 107, 668 (2003)) per principal axis of semi-axis a_i:

    alpha_i -> alpha_i / (1 - k^2 alpha_i / (4 pi a_i) - i k^3 alpha_i / (6 pi)).

For homogeneous spheres, ``"mie_dipole"`` instead uses the Mie electric-dipole
coefficient a_1 with its denominator expanded exactly to O(x^4) (derived from
a_1 = -i beta / (1 - i beta), beta = (2/3) x^3 alpha / (4 pi a^3)):

    4 pi a^3 / alpha = [(eps + 2) - 3/5 (eps - 2) x^2 - 3/350 (eps^2 - 24 eps + 16) x^4] / (eps - 1),

plus radiative damping. MLWA's x^2 coefficient is (eps - 1) where the exact one is
3/5 (eps - 2), so near a sharp plasmon resonance ``"mie_dipole"`` is several times
more accurate.

With any correction the extinction follows the optical theorem,
C_ext = k Im(e.alpha.e), and C_abs = C_ext - C_sca.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..results import FarFieldData, Result
from ..spec import LayeredSphere, Simulation, Sphere, Spheroid
from .base import Backend, register, unsupported_outputs

X_WARN = 0.3


class QuasistaticOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corrections: Literal["none", "radiative", "mlwa", "mie_dipole"] = Field(
        "none", description="none: static dipole. radiative: add radiative damping. "
        "mlwa: modified long-wavelength approximation (dynamic depolarisation + radiative damping). "
        "mie_dipole: Mie dipole term exact to O(x^4) + radiative damping (homogeneous spheres only).")


def depolarization_factors(semi_axis_xy_nm: float, semi_axis_z_nm: float) -> np.ndarray:
    """(L_x, L_y, L_z) of a spheroid with symmetry axis z (BH 5.33-5.34); they sum to 1."""
    a, c = float(semi_axis_xy_nm), float(semi_axis_z_nm)
    if c >= a:  # prolate (or sphere)
        e = np.sqrt(1 - (a / c) ** 2)
        if e < 1e-4:
            lz = 1 / 3 - 2 * e**2 / 15
        else:
            lz = (1 - e**2) / e**2 * (np.arctanh(e) / e - 1)
    else:  # oblate
        e = np.sqrt(1 - (c / a) ** 2)
        if e < 1e-4:
            lz = 1 / 3 + 2 * e**2 / 15
        else:
            lz = (1 - np.sqrt(1 - e**2) / e * np.arcsin(e)) / e**2
    lx = (1 - lz) / 2
    return np.array([lx, lx, lz])


def coated_sphere_eps(radii_nm: list[float], eps: list[np.ndarray]) -> np.ndarray:
    """Effective (quasistatic) permittivity of a multilayer sphere, core first (BH 5.36, applied recursively)."""
    e_in = np.asarray(eps[0], dtype=complex)
    for l in range(1, len(radii_nm)):
        f = (radii_nm[l - 1] / radii_nm[l]) ** 3
        e_s = np.asarray(eps[l], dtype=complex)
        e_in = e_s * (e_in + 2 * e_s + 2 * f * (e_in - e_s)) / (e_in + 2 * e_s - f * (e_in - e_s))
    return e_in


def polarizability(sim: Simulation, wl: np.ndarray, corrections: str = "none") -> tuple[np.ndarray, np.ndarray]:
    """Principal polarizabilities alpha_i (W, 3) in nm^3 (BH units, includes 4 pi) and semi-axes (3,)."""
    p = sim.particles[0]
    shape = p.shape
    n2 = sim.medium_index**2
    k = 2 * np.pi * sim.medium_index / wl
    if isinstance(shape, Spheroid):
        semi = np.array([shape.semi_axis_xy_nm, shape.semi_axis_xy_nm, shape.semi_axis_z_nm])
        L = depolarization_factors(shape.semi_axis_xy_nm, shape.semi_axis_z_nm)
        eps = np.asarray(p.material.eps(wl), dtype=complex) / n2
    else:
        radii = list(shape.radii_nm) if isinstance(shape, LayeredSphere) else [shape.radius_nm]
        semi = np.full(3, radii[-1])
        L = np.full(3, 1 / 3)
        eps = coated_sphere_eps(radii, [np.asarray(m.eps(wl), dtype=complex) / n2 for m in p.region_materials()])
    vol = 4 / 3 * np.pi * np.prod(semi)
    e = eps[:, None]
    kk = k[:, None]
    if corrections == "mie_dipole":
        if not isinstance(shape, Sphere):
            raise ValueError("corrections='mie_dipole' applies to homogeneous spheres only; use 'mlwa'")
        x2 = (kk * semi) ** 2
        den = (e + 2) - 0.6 * (e - 2) * x2 - 3 / 350 * (e**2 - 24 * e + 16) * x2**2
        alpha = 3 * vol * (e - 1) / den
        return alpha / (1 - 1j * kk**3 * alpha / (6 * np.pi)), semi
    alpha = vol * (e - 1) / (1 + L * (e - 1))
    if corrections == "mlwa":
        alpha = alpha / (1 - kk**2 * alpha / (4 * np.pi * semi) - 1j * kk**3 * alpha / (6 * np.pi))
    elif corrections == "radiative":
        alpha = alpha / (1 - 1j * kk**3 * alpha / (6 * np.pi))
    return alpha, semi


@register
class Quasistatic(Backend):
    name = "lightsmith-quasistatic"
    method = "quasistatic"
    license = "MIT"
    citation = ("Bohren & Huffman, Absorption and Scattering of Light by Small Particles (1983), ch. 5; "
                "Meier & Wokaun, Opt. Lett. 8, 581 (1983)")

    @classmethod
    def supports(cls, sim: Simulation):
        if len(sim.particles) != 1:
            return False, "the quasistatic approximation handles exactly one particle"
        shape = sim.particles[0].shape
        if not isinstance(shape, (Sphere, LayeredSphere, Spheroid)):
            return False, f"quasistatic handles spheres, layered spheres and spheroids, got {shape.kind!r}"
        why = unsupported_outputs(sim, far_field=True, orientation_average=True)
        if not why and sim.orientation_average and sim.outputs.far_field:
            why = "far_field output is not available with orientation_average"
        if not why and cls._options(sim).corrections == "mie_dipole" and not isinstance(shape, Sphere):
            why = "corrections='mie_dipole' applies to homogeneous spheres only; use 'mlwa'"
        return (not why), why

    @classmethod
    def _options(cls, sim) -> QuasistaticOptions:
        return QuasistaticOptions.model_validate(sim.options_for(cls.name, cls.method))

    @classmethod
    def estimate(cls, sim, device):
        return {"cost": "negligible", "memory_bytes": 0, "size_parameter": cls._size_parameter(sim)}

    @classmethod
    def _size_parameter(cls, sim) -> float:
        a = float(sim.particles[0].shape.bounding_half_extent().max())
        return float(2 * np.pi * sim.medium_index * a / sim.wavelength_array().min())

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        t0 = time.perf_counter()
        opts = cls._options(sim)
        wl = sim.wavelength_array()
        k = 2 * np.pi * sim.medium_index / wl
        alpha, _ = polarizability(sim, wl, opts.corrections)
        khat, ehat = sim.illumination.unit_vectors()
        if sim.orientation_average:
            ext_like = k * np.imag(alpha).mean(axis=1)
            c_sca = k**4 / (6 * np.pi) * (np.abs(alpha) ** 2).mean(axis=1)
        else:
            e2 = ehat**2
            ext_like = k * (np.imag(alpha) @ e2)
            c_sca = k**4 / (6 * np.pi) * (np.abs(alpha) ** 2 @ e2)
        if opts.corrections == "none":
            c_abs, c_ext = ext_like, ext_like + c_sca
        else:
            c_ext, c_abs = ext_like, ext_like - c_sca

        x = cls._size_parameter(sim)
        warnings = []
        if x > X_WARN:
            warnings.append(f"size parameter x = {x:.2f} > {X_WARN}: the quasistatic approximation is unreliable; "
                            "use method 'mie' (spheres) or 'dda'/'tmatrix' (spheroids)")
        if device.name != "cpu":
            warnings.append("the quasistatic backend always runs on the CPU; device ignored")
        far = cls._far_field(sim, sim.outputs.far_field, opts, khat, ehat) if sim.outputs.far_field else None
        res = Result.from_arrays(
            method=cls.method, backend=cls.name, wavelengths_nm=wl,
            c_ext=c_ext, c_sca=c_sca, c_abs=c_abs, r_eq=sim.equivalent_radius_nm(),
            provenance={"device": "cpu", "corrections": opts.corrections, "size_parameter": x,
                        "runtime_s": time.perf_counter() - t0},
            warnings=warnings,
        )
        res.far_field = far
        return res

    @classmethod
    def _far_field(cls, sim, req, opts, khat, ehat) -> list[FarFieldData]:
        wl = np.asarray(req.wavelengths_nm, dtype=float)
        alpha, _ = polarizability(sim, wl, opts.corrections)
        hhat = np.cross(khat, ehat)
        th = np.radians(np.asarray(req.theta_deg))[:, None]
        ph = np.radians(np.asarray(req.phi_deg))[None, :]
        nhat = (np.cos(th)[..., None] * khat + np.sin(th)[..., None]
                * (np.cos(ph)[..., None] * ehat + np.sin(ph)[..., None] * hhat))  # (T, P, 3)
        out = []
        for lam, a in zip(wl, alpha):
            k = 2 * np.pi * sim.medium_index / lam
            p = a * ehat  # induced dipole (particle axes = lab axes)
            p_perp = p - nhat * np.sum(nhat * p, axis=-1, keepdims=True)
            dcs = k**4 / (16 * np.pi**2) * np.sum(np.abs(p_perp) ** 2, axis=-1)
            out.append(FarFieldData(wavelength_nm=float(lam), theta_deg=list(req.theta_deg),
                                    phi_deg=list(req.phi_deg), dcsca_domega_nm2=dcs.tolist()))
        return out
