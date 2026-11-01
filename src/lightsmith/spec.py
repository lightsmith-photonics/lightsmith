"""The problem specification: one description, solved by any backend.

Units: lengths and wavelengths in nanometres (vacuum wavelength).
Conventions: exp(-i*omega*t); cross sections in nm^2; efficiencies are
normalised by the geometric cross section of the volume-equivalent sphere,
pi * r_eq^2, so they are comparable across methods and shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .materials import Material

SCHEMA_VERSION = "0.1"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- shapes (centred at the origin in the particle frame) -------------------


class Sphere(_Model):
    kind: Literal["sphere"] = "sphere"
    radius_nm: float = Field(gt=0)

    def volume(self) -> float:
        return 4 / 3 * np.pi * self.radius_nm**3

    def bounding_half_extent(self) -> np.ndarray:
        return np.full(3, self.radius_nm)

    def contains(self, p: np.ndarray) -> np.ndarray:
        return np.sum(p**2, axis=-1) <= self.radius_nm**2


class Spheroid(_Model):
    """Spheroid with symmetry axis along z."""

    kind: Literal["spheroid"] = "spheroid"
    semi_axis_xy_nm: float = Field(gt=0)
    semi_axis_z_nm: float = Field(gt=0)

    def volume(self) -> float:
        return 4 / 3 * np.pi * self.semi_axis_xy_nm**2 * self.semi_axis_z_nm

    def bounding_half_extent(self) -> np.ndarray:
        a, c = self.semi_axis_xy_nm, self.semi_axis_z_nm
        return np.array([a, a, c])

    def contains(self, p: np.ndarray) -> np.ndarray:
        a, c = self.semi_axis_xy_nm, self.semi_axis_z_nm
        return (p[..., 0] ** 2 + p[..., 1] ** 2) / a**2 + p[..., 2] ** 2 / c**2 <= 1


class Cylinder(_Model):
    """Flat-ended cylinder with axis along z."""

    kind: Literal["cylinder"] = "cylinder"
    radius_nm: float = Field(gt=0)
    length_nm: float = Field(gt=0)

    def volume(self) -> float:
        return np.pi * self.radius_nm**2 * self.length_nm

    def bounding_half_extent(self) -> np.ndarray:
        return np.array([self.radius_nm, self.radius_nm, self.length_nm / 2])

    def contains(self, p: np.ndarray) -> np.ndarray:
        return (p[..., 0] ** 2 + p[..., 1] ** 2 <= self.radius_nm**2) & (
            np.abs(p[..., 2]) <= self.length_nm / 2
        )


class Shell(_Model):
    thickness_nm: float = Field(gt=0)
    material: Material


class LayeredSphere(_Model):
    """Core-shell / multilayer sphere. The particle's `material` is the core;
    `shells` are listed from the inside out."""

    kind: Literal["layered_sphere"] = "layered_sphere"
    core_radius_nm: float = Field(gt=0)
    shells: list[Shell] = Field(min_length=1)

    @property
    def radii_nm(self) -> list[float]:
        r = [self.core_radius_nm]
        for s in self.shells:
            r.append(r[-1] + s.thickness_nm)
        return r

    @property
    def radius_nm(self) -> float:
        return self.radii_nm[-1]

    def volume(self) -> float:
        return 4 / 3 * np.pi * self.radius_nm**3

    def bounding_half_extent(self) -> np.ndarray:
        return np.full(3, self.radius_nm)

    def contains(self, p: np.ndarray) -> np.ndarray:
        return np.sum(p**2, axis=-1) <= self.radius_nm**2

    def region(self, p: np.ndarray) -> np.ndarray:
        """-1 outside, 0 in the core, i in shell i."""
        r = np.sqrt(np.sum(p**2, axis=-1))
        out = np.searchsorted(np.asarray(self.radii_nm), r, side="left")
        return np.where(r <= self.radius_nm, out, -1)


Shape = Annotated[Union[Sphere, Spheroid, Cylinder, LayeredSphere], Field(discriminator="kind")]


class Particle(_Model):
    shape: Shape
    material: Material = Field(description="Particle material (the core for layered spheres).")
    position_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def region_materials(self) -> list:
        """Materials indexed by region: [core/body, shell 1, shell 2, ...]."""
        shells = getattr(self.shape, "shells", [])
        return [self.material, *[s.material for s in shells]]

    def region(self, p: np.ndarray) -> np.ndarray:
        """Region index per point in the particle frame (-1 = outside)."""
        if hasattr(self.shape, "region"):
            return self.shape.region(p)
        return np.where(self.shape.contains(p), 0, -1)


class Substrate(_Model):
    """Semi-infinite half-space z < interface_z_nm with the given material."""

    material: Material
    interface_z_nm: float = 0.0


# --- illumination & wavelengths ---------------------------------------------


class PlaneWave(_Model):
    kind: Literal["plane_wave"] = "plane_wave"
    direction: tuple[float, float, float] = Field((0.0, 0.0, 1.0), description="Propagation direction.")
    polarization: tuple[float, float, float] = Field((1.0, 0.0, 0.0), description="Real E-field direction.")

    @model_validator(mode="after")
    def _orthogonal(self):
        k, e = np.asarray(self.direction), np.asarray(self.polarization)
        if np.linalg.norm(k) == 0 or np.linalg.norm(e) == 0:
            raise ValueError("direction and polarization must be non-zero")
        if abs(k @ e) > 1e-9 * np.linalg.norm(k) * np.linalg.norm(e):
            raise ValueError("polarization must be orthogonal to direction")
        return self

    def unit_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        k, e = np.asarray(self.direction, float), np.asarray(self.polarization, float)
        return k / np.linalg.norm(k), e / np.linalg.norm(e)


class WavelengthRange(_Model):
    start_nm: float = Field(gt=0)
    stop_nm: float = Field(gt=0)
    num: int = Field(ge=1)

    def values(self) -> np.ndarray:
        return np.linspace(self.start_nm, self.stop_nm, self.num)


# --- requested outputs ------------------------------------------------------


class NearFieldRequest(_Model):
    """Total electric field on a square plane through the scene."""

    plane: Literal["xy", "xz", "yz"] = "xz"
    offset_nm: float = Field(0.0, description="Position of the plane along its normal.")
    half_width_nm: float = Field(gt=0)
    points: int = Field(101, ge=2, description="Samples per axis.")
    wavelengths_nm: list[float] = Field(min_length=1, description="Vacuum wavelengths to sample.")


class FarFieldRequest(_Model):
    """Differential scattering cross section dC_sca/dOmega on a theta x phi grid.

    theta is measured from the incident direction; phi is the azimuth around it,
    measured from the incident polarization (phi=0 is the E-plane, 90 the H-plane).
    """

    theta_deg: list[float] = Field(min_length=1)
    phi_deg: list[float] = Field([0.0, 90.0], min_length=1)
    wavelengths_nm: list[float] = Field(min_length=1)


class Outputs(_Model):
    cross_sections: bool = True
    near_field: NearFieldRequest | None = None
    far_field: FarFieldRequest | None = None


# --- the simulation ---------------------------------------------------------


class Simulation(_Model):
    """A complete scattering problem. Serialisable to/from JSON."""

    kind: Literal["scattering"] = "scattering"
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    particles: list[Particle] = Field(min_length=1)
    medium_index: float = Field(1.0, gt=0, description="Real refractive index of the host medium.")
    illumination: PlaneWave = PlaneWave()
    wavelengths: list[float] | WavelengthRange = Field(description="Vacuum wavelengths in nm.")
    substrate: Substrate | None = Field(None, description="Optional half-space below the particles.")
    orientation_average: bool = Field(False, description="Average cross sections over all particle orientations.")
    outputs: Outputs = Outputs()
    options: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Solver options keyed by method or backend name, e.g. {'dda': {'dipoles_per_diameter': 20}}.",
    )

    def options_for(self, backend_name: str, method: str) -> dict[str, Any]:
        """Method-level options overlaid with backend-specific ones."""
        return {**self.options.get(method, {}), **self.options.get(backend_name, {})}

    @field_validator("wavelengths")
    @classmethod
    def _positive(cls, v):
        if isinstance(v, list) and (not v or min(v) <= 0):
            raise ValueError("wavelengths must be a non-empty list of positive values")
        return v

    def wavelength_array(self) -> np.ndarray:
        w = self.wavelengths
        return w.values() if isinstance(w, WavelengthRange) else np.asarray(w, dtype=float)

    def equivalent_radius_nm(self) -> float:
        v = sum(p.shape.volume() for p in self.particles)
        return (3 * v / (4 * np.pi)) ** (1 / 3)

    # convenience I/O
    def to_json(self, **kw) -> str:
        return self.model_dump_json(**kw)

    @classmethod
    def from_json(cls, text: str) -> "Simulation":
        return cls.model_validate_json(text)

    @classmethod
    def load(cls, path: str | Path) -> "Simulation":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def json_schema() -> dict:
    """JSON Schema of :class:`Simulation`, for agents and validators."""
    return Simulation.model_json_schema()


def dumps_schema() -> str:
    return json.dumps(json_schema(), indent=2)
