"""lightsmith backend for scattnlay (GPL-3.0-or-later): Mie theory for multilayer spheres.

scattnlay (Peña & Pal 2009; Ladutenko et al. 2017) takes size parameters
x_i = 2 pi n_medium r_i / lambda and relative indices m_i = n_i / n_medium, both
listed from the core outwards, and returns efficiencies normalised by the
geometric cross section of the outer sphere, pi r_out^2.

Sign convention, checked against scattnlay 2.4: m = 1.5 + 0.1i gives Qabs > 0
and m = 1.5 - 0.1i gives Qabs < 0, so scattnlay uses exp(-i omega t) like
lightsmith and indices are passed unconjugated.
"""

from __future__ import annotations

import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from lightsmith import FarFieldData, LayeredSphere, Result, Simulation, Sphere
from lightsmith.backends import Backend, unsupported_outputs
from lightsmith.device import Device
from pydantic import BaseModel, ConfigDict, Field


class ScattnlayOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    multiprecision: bool = Field(False, description="Use scattnlay's Boost multiprecision build (scattnlay_mp), "
                                 "if installed. The PyPI 2.4 wheels ship only double precision.")
    n_max: int | None = Field(None, ge=1, description="Number of multipole terms. Default: scattnlay's own choice.")
    pec_layer: int | None = Field(None, ge=0, description="Index (from the core, 0-based) of a perfectly "
                                  "conducting layer, scattnlay's 'pl'.")


def _scattnlay_version() -> str | None:
    try:
        return version("scattnlay")
    except PackageNotFoundError:
        return None


def _has_mp() -> bool:
    try:
        from scattnlay import mie_mp
    except ImportError:
        return False
    return mie_mp is not None


_SHARED = {"n_max"}  # keys of the method-level options["mie"] that scattnlay honours


class ScattnlayBackend(Backend):
    name = "scattnlay"
    method = "mie"
    license = "GPL-3.0-or-later"
    citation = ("O. Peña and U. Pal, Comput. Phys. Commun. 180, 2348 (2009); K. Ladutenko, U. Pal, A. Rivera and "
                "O. Peña-Rodríguez, Comput. Phys. Commun. 214, 225 (2017); https://github.com/ovidiopr/scattnlay")

    @classmethod
    def available(cls):
        try:
            import scattnlay  # noqa: F401
        except ImportError as e:
            return False, f"scattnlay is not importable ({e}); install it with `pip install scattnlay`"
        return True, ""

    @classmethod
    def supports(cls, sim: Simulation):
        if len(sim.particles) != 1:
            return False, "scattnlay handles exactly one (possibly layered) sphere; use 'tmatrix' or 'dda' for clusters"
        shape = sim.particles[0].shape
        if not isinstance(shape, (Sphere, LayeredSphere)):
            return False, f"scattnlay handles spheres and layered spheres only, got {shape.kind!r}"
        # A sphere looks the same from every direction, so orientation averaging is trivial.
        why = unsupported_outputs(sim, far_field=True, orientation_average=True)
        return (not why), why

    @classmethod
    def estimate(cls, sim, device):
        return {"cost": "negligible", "memory_bytes": 0}

    @classmethod
    def options(cls, sim: Simulation) -> tuple[ScattnlayOptions, list[str]]:
        """Strict ``options["scattnlay"]`` over the shared keys of ``options["mie"]`` (others ignored, warned)."""
        shared = sim.options.get(cls.method, {})
        ignored = sorted(k for k in shared if k not in _SHARED)
        warnings = [f"scattnlay ignores mie options {ignored}"] if ignored else []
        mapped = {k: v for k, v in shared.items() if k in _SHARED}
        return ScattnlayOptions.model_validate({**mapped, **sim.options.get(cls.name, {})}), warnings

    @staticmethod
    def layers(sim: Simulation, wavelengths_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Size parameters x and relative indices m, shape (n_wavelengths, n_layers), core first."""
        p = sim.particles[0]
        radii = np.asarray(p.shape.radii_nm if isinstance(p.shape, LayeredSphere) else [p.shape.radius_nm], float)
        n0 = sim.medium_index
        wl = np.asarray(wavelengths_nm, float)
        x = 2 * np.pi * n0 * radii[None, :] / wl[:, None]
        m = np.stack([np.asarray(mat.n(wl), complex) / n0 for mat in p.region_materials()], axis=1)
        return x, m

    @classmethod
    def _run(cls, x, m, opts: ScattnlayOptions, theta=None):
        from scattnlay import scattnlay

        kw = {"nmax": opts.n_max or -1, "pl": -1 if opts.pec_layer is None else opts.pec_layer,
              "mp": opts.multiprecision and _has_mp()}
        if theta is not None:
            kw["theta"] = np.asarray(theta, float)
        return scattnlay(np.ascontiguousarray(x), np.ascontiguousarray(m), **kw)

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        t0 = time.perf_counter()
        opts, warnings = cls.options(sim)
        if device.name != "cpu":
            warnings.append("scattnlay always runs on the CPU; device ignored")
        if opts.multiprecision and not _has_mp():
            warnings.append("scattnlay multiprecision module not installed; used double precision")
        wl = sim.wavelength_array()
        x, m = cls.layers(sim, wl)
        terms, qext, qsca, qabs, *_ = cls._run(x, m, opts)
        r_out = float(sim.particles[0].shape.radius_nm)
        geo = np.pi * r_out**2
        far = None
        if sim.outputs.far_field is not None:
            far = cls._far_field(sim, opts)
        res = Result.from_arrays(
            method=cls.method, backend=cls.name, wavelengths_nm=wl,
            c_ext=np.asarray(qext) * geo, c_sca=np.asarray(qsca) * geo, c_abs=np.asarray(qabs) * geo, r_eq=r_out,
            provenance={"scattnlay_version": _scattnlay_version(), "device": "cpu",
                        "precision": "multiple" if opts.multiprecision and _has_mp() else "double",
                        "terms": np.atleast_1d(terms).astype(int).tolist(), "layers": int(x.shape[1]),
                        "runtime_s": time.perf_counter() - t0},
            warnings=warnings,
        )
        res.far_field = far
        return res

    @classmethod
    def _far_field(cls, sim: Simulation, opts: ScattnlayOptions) -> list[FarFieldData]:
        """dC_sca/dOmega = (|S2|^2 cos^2 phi + |S1|^2 sin^2 phi) / k^2 (Bohren & Huffman 4.74),
        phi measured from the incident polarization, k in the medium."""
        req = sim.outputs.far_field
        wl = np.asarray(req.wavelengths_nm, float)
        theta = np.radians(np.asarray(req.theta_deg, float))
        phi = np.radians(np.asarray(req.phi_deg, float))[None, :]
        x, m = cls.layers(sim, wl)
        out = []
        for i, lam in enumerate(wl):
            *_, s1, s2 = cls._run(x[i], m[i], opts, theta)
            k = 2 * np.pi * sim.medium_index / lam
            d = (np.abs(s2)[:, None] ** 2 * np.cos(phi) ** 2 + np.abs(s1)[:, None] ** 2 * np.sin(phi) ** 2) / k**2
            out.append(FarFieldData(wavelength_nm=float(lam), theta_deg=list(req.theta_deg), phi_deg=list(req.phi_deg),
                                    dcsca_domega_nm2=d.tolist()))
        return out
