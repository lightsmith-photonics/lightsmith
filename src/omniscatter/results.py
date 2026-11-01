"""Solver results: plain, JSON-serialisable, with provenance."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class NearFieldData(BaseModel):
    """Total field on a plane. Arrays are indexed [axis1][axis2] (and [component] first for E)."""

    wavelength_nm: float
    plane: str
    offset_nm: float
    axis1_nm: list[float]
    axis2_nm: list[float]
    enhancement: list[list[float]] = Field(description="|E|^2 / |E_inc|^2; NaN where undefined (e.g. inside a dipole voxel).")
    e_real: list[list[list[float]]] | None = None
    e_imag: list[list[list[float]]] | None = None


class FarFieldData(BaseModel):
    wavelength_nm: float
    theta_deg: list[float]
    phi_deg: list[float]
    dcsca_domega_nm2: list[list[float]] = Field(description="dC_sca/dOmega in nm^2/sr, indexed [theta][phi].")


class Result(BaseModel):
    """Cross sections of a scattering problem."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["cross_sections"] = "cross_sections"
    method: str = Field(description="Numerical method, e.g. 'mie', 'dda'.")
    backend: str = Field(description="Implementation that produced the numbers.")
    wavelengths_nm: list[float]
    c_ext_nm2: list[float]
    c_sca_nm2: list[float]
    c_abs_nm2: list[float]
    equivalent_radius_nm: float
    near_field: list[NearFieldData] | None = None
    far_field: list[FarFieldData] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_arrays(cls, *, method, backend, wavelengths_nm, c_ext, c_sca, c_abs, r_eq, **kw) -> "Result":
        f = lambda a: np.asarray(a, dtype=float).tolist()  # noqa: E731
        return cls(method=method, backend=backend, wavelengths_nm=f(wavelengths_nm),
                   c_ext_nm2=f(c_ext), c_sca_nm2=f(c_sca), c_abs_nm2=f(c_abs),
                   equivalent_radius_nm=float(r_eq), **kw)

    def _q(self, c: list[float]) -> np.ndarray:
        return np.asarray(c) / (np.pi * self.equivalent_radius_nm**2)

    @property
    def q_ext(self) -> np.ndarray:
        return self._q(self.c_ext_nm2)

    @property
    def q_sca(self) -> np.ndarray:
        return self._q(self.c_sca_nm2)

    @property
    def q_abs(self) -> np.ndarray:
        return self._q(self.c_abs_nm2)

    def summary(self) -> dict[str, Any]:
        """Compact digest: peak extinction and where it is."""
        i = int(np.argmax(self.c_ext_nm2))
        return {"method": self.method, "backend": self.backend,
                "n_wavelengths": len(self.wavelengths_nm),
                "peak_ext_wavelength_nm": self.wavelengths_nm[i],
                "peak_q_ext": float(self.q_ext[i]),
                "runtime_s": self.provenance.get("runtime_s")}

    def to_xarray(self):
        import xarray as xr  # optional dependency

        wl = np.asarray(self.wavelengths_nm)
        ds = xr.Dataset(
            {k: ("wavelength_nm", np.asarray(getattr(self, k))) for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2")},
            coords={"wavelength_nm": wl},
            attrs={"method": self.method, "backend": self.backend,
                   "equivalent_radius_nm": self.equivalent_radius_nm, "time_convention": "exp(-i omega t)"},
        )
        return ds


def relative_difference(a: Result, b: Result, quantity: str = "c_ext_nm2") -> dict[str, float]:
    """Max and RMS relative difference of `quantity` between two results on the same wavelengths."""
    if not np.allclose(a.wavelengths_nm, b.wavelengths_nm):
        raise ValueError("results are on different wavelength grids")
    x, y = np.asarray(getattr(a, quantity)), np.asarray(getattr(b, quantity))
    rel = np.abs(x - y) / np.maximum(np.abs(y), 1e-300)
    return {"max": float(rel.max()), "rms": float(np.sqrt(np.mean(rel**2)))}


# --- results of the other problem kinds ------------------------------------


class _ResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    backend: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DiffractionOrder(BaseModel):
    order: tuple[int, int]
    r: dict[str, list[list[float]]] = Field(description="Reflected power fraction per polarization, [angle][wavelength].")
    t: dict[str, list[list[float]]]


class SpectrumResult(_ResultBase):
    """Power reflection/transmission/absorption of layer stacks and periodic structures."""

    kind: Literal["spectrum"] = "spectrum"
    wavelengths_nm: list[float]
    angles_deg: list[float]
    polarizations: list[str]
    r: dict[str, list[list[float]]] = Field(description="Total reflectance per polarization, [angle][wavelength].")
    t: dict[str, list[list[float]]]
    a: dict[str, list[list[float]]]
    orders: list[DiffractionOrder] | None = None
    r_amplitude: dict[str, list[list[tuple[float, float]]]] | None = Field(
        None, description="Complex zeroth-order reflection amplitude (re, im), when available.")
    t_amplitude: dict[str, list[list[tuple[float, float]]]] | None = None


class ModeField(BaseModel):
    x_nm: list[float]
    y_nm: list[float]
    components: dict[str, tuple[list[list[float]], list[list[float]]]] = Field(
        description="'Ex'...'Hz' -> (real, imag), each indexed [x][y].")


class ModeData(BaseModel):
    neff_real: float
    neff_imag: float
    ng: float | None = None
    te_fraction: float | None = Field(None, description="Fraction of transverse E energy in Ex.")
    loss_db_per_cm: float | None = None
    effective_area_um2: float | None = None
    field: ModeField | None = None


class ModeResult(_ResultBase):
    kind: Literal["modes"] = "modes"
    wavelengths_nm: list[float]
    modes: list[list[ModeData]] = Field(description="[wavelength][mode], sorted by decreasing neff_real.")


class SParameterResult(_ResultBase):
    kind: Literal["sparameters"] = "sparameters"
    wavelengths_nm: list[float]
    ports: list[str] = Field(description="Port names, with '@<mode>' suffix for multimode ports, e.g. 'o1@0'.")
    s_real: list[list[list[float]]] = Field(description="[wavelength][out][in].")
    s_imag: list[list[list[float]]]

    def s(self) -> np.ndarray:
        return np.asarray(self.s_real) + 1j * np.asarray(self.s_imag)


class BandResult(_ResultBase):
    kind: Literal["bands"] = "bands"
    k_reduced: list[tuple[float, float, float]]
    k_distance: list[float] = Field(description="Cumulative path length (for plotting).")
    k_labels: dict[int, str] = Field(description="Index into k_reduced -> label, e.g. {0: 'Γ'}.")
    frequencies: dict[str, list[list[float]]] = Field(
        description="Polarization ('te'/'tm'/'all') -> normalised frequency a/λ, [k][band].")
