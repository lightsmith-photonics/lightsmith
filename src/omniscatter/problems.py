"""Problem types beyond single-object scattering.

Every problem is a pydantic model with a ``kind`` discriminator, so any of them
can be passed as JSON to ``solve``/``estimate``/the CLI/MCP. Shared conventions:
lengths in nm, vacuum wavelengths, exp(-i omega t) (Im(eps) > 0 is loss).

Coordinate conventions:
- layer_stack / periodic: layers are stacked along z and listed from the
  superstrate side to the substrate side; light is incident from the
  semi-infinite superstrate and travels towards the semi-infinite substrate
  (+z). R is measured in the superstrate, T in the substrate.
- mode: the waveguide cross-section lies in the xy plane; propagation is along +z.
- device: structures live in 3D (or the xy plane when dimension == 2); ports are
  planes whose normal is the axis where `size_nm` is zero.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .materials import Material
from .spec import Simulation, WavelengthRange

Wavelengths = list[float] | WavelengthRange


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def wavelength_array(w: Wavelengths) -> np.ndarray:
    return w.values() if isinstance(w, WavelengthRange) else np.asarray(w, dtype=float)


# --- 2D profiles and 3D geometry --------------------------------------------


class Rectangle(_Model):
    kind: Literal["rectangle"] = "rectangle"
    center_nm: tuple[float, float] = (0.0, 0.0)
    size_nm: tuple[float, float]


class Circle(_Model):
    kind: Literal["circle"] = "circle"
    center_nm: tuple[float, float] = (0.0, 0.0)
    radius_nm: float = Field(gt=0)


class Ellipse(_Model):
    kind: Literal["ellipse"] = "ellipse"
    center_nm: tuple[float, float] = (0.0, 0.0)
    semi_axes_nm: tuple[float, float]
    angle_deg: float = 0.0


class Polygon(_Model):
    kind: Literal["polygon"] = "polygon"
    vertices_nm: list[tuple[float, float]] = Field(min_length=3)


Profile = Annotated[Union[Rectangle, Circle, Ellipse, Polygon], Field(discriminator="kind")]


class Prism(_Model):
    """A 2D profile in the xy plane extruded from z_min to z_max (with optional sidewall angle)."""

    kind: Literal["prism"] = "prism"
    profile: Profile
    z_min_nm: float
    z_max_nm: float
    sidewall_angle_deg: float = Field(0.0, description="0 = vertical; positive = narrower at the top.")


class Box(_Model):
    kind: Literal["box"] = "box"
    center_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_nm: tuple[float, float, float]


class Ball(_Model):
    kind: Literal["ball"] = "ball"
    center_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_nm: float = Field(gt=0)


Geometry = Annotated[Union[Prism, Box, Ball], Field(discriminator="kind")]


class Structure(_Model):
    """A material region. Later structures override earlier ones where they overlap."""

    name: str | None = None
    geometry: Geometry
    material: Material


class Structure2D(_Model):
    name: str | None = None
    profile: Profile
    material: Material


# --- thin films ---------------------------------------------------------------


class Layer(_Model):
    thickness_nm: float = Field(ge=0)
    material: Material
    coherent: bool = Field(True, description="False for thick incoherent layers (e.g. a 1 mm substrate slab).")


class LayerStackProblem(_Model):
    """Planar multilayer: reflection/transmission/absorption vs wavelength and angle."""

    kind: Literal["layer_stack"] = "layer_stack"
    superstrate: Material
    layers: list[Layer] = Field(default_factory=list, description="From the superstrate side to the substrate side.")
    substrate: Material
    wavelengths: Wavelengths
    angles_deg: list[float] = Field([0.0], description="Angles of incidence in the superstrate.")
    polarizations: list[Literal["s", "p"]] = ["s", "p"]
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


# --- periodic structures (gratings, metasurfaces, photonic-crystal slabs) ----


class PatternedLayer(_Model):
    thickness_nm: float = Field(gt=0)
    background: Material
    shapes: list[Structure2D] = Field(default_factory=list, description="Inclusions in one unit cell (xy).")


class Incidence(_Model):
    theta_deg: float = 0.0
    phi_deg: float = 0.0
    polarizations: list[Literal["s", "p"]] = ["s", "p"]


class PeriodicProblem(_Model):
    """A 1D/2D-periodic stack of patterned layers under plane-wave illumination."""

    kind: Literal["periodic"] = "periodic"
    lattice_a1_nm: tuple[float, float]
    lattice_a2_nm: tuple[float, float] | None = Field(None, description="None for 1D gratings (periodic along a1).")
    superstrate: Material
    layers: list[PatternedLayer] = Field(min_length=1, description="From the superstrate side to the substrate side.")
    substrate: Material
    wavelengths: Wavelengths
    incidence: Incidence = Incidence()
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


# --- waveguide modes ------------------------------------------------------------


class ModeProblem(_Model):
    """Eigenmodes of a waveguide cross-section (xy plane), propagation along +z."""

    kind: Literal["mode"] = "mode"
    structures: list[Structure2D] = Field(min_length=1)
    background: Material
    window_center_nm: tuple[float, float] = (0.0, 0.0)
    window_size_nm: tuple[float, float]
    wavelengths: Wavelengths
    num_modes: int = Field(2, ge=1)
    target_neff: float | None = Field(None, description="Search near this effective index (default: highest index).")
    bend_radius_nm: float | None = Field(None, description="Bend about the y axis; None = straight.")
    include_fields: bool = False
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


# --- devices with ports (S-parameters) -----------------------------------------


class Port(_Model):
    name: str
    center_nm: tuple[float, float, float]
    size_nm: tuple[float, float, float] = Field(
        description="Zero along the port normal. In 2D devices z may also be zero, e.g. (0, w, 0) for an x-normal port.")
    direction: Literal["+", "-"] = Field(description="Direction of propagation *into* the device along the normal.")
    num_modes: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _one_normal(self):
        zeros = [i for i, s in enumerate(self.size_nm) if s == 0]
        if not (len(zeros) == 1 or (len(zeros) == 2 and 2 in zeros)):
            raise ValueError("port size_nm must be zero along exactly one in-plane axis (the normal)")
        return self

    @property
    def normal_axis(self) -> int:
        return next(i for i, s in enumerate(self.size_nm) if s == 0)


class GdsLayout(_Model):
    """Import geometry from a GDS file: each layer/datatype is extruded with a material."""

    path: str
    cell: str | None = None
    layer_map: dict[str, dict[str, Any]] = Field(
        description="'layer/datatype' -> {'material': Material JSON, 'z_min_nm': .., 'z_max_nm': ..}")


class DeviceProblem(_Model):
    """A photonic device (waveguides, couplers, gratings, ...) characterised by its S-matrix."""

    kind: Literal["device"] = "device"
    dimension: Literal[2, 3] = 3
    structures: list[Structure] = Field(default_factory=list)
    layout: GdsLayout | None = None
    background: Material
    domain_center_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_size_nm: tuple[float, float, float]
    ports: list[Port] = Field(min_length=1)
    wavelengths: Wavelengths
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


# --- photonic band structure ------------------------------------------------


class KPoint(_Model):
    label: str
    reduced: tuple[float, float, float] = Field(description="Coordinates in units of the reciprocal lattice vectors.")


class BandProblem(_Model):
    """Band structure of a photonic crystal along a k-path."""

    kind: Literal["bands"] = "bands"
    dimension: Literal[2, 3] = 2
    lattice_vectors_nm: list[tuple[float, float, float]] = Field(min_length=2, max_length=3)
    background: Material
    structures_2d: list[Structure2D] = Field(default_factory=list, description="Unit-cell inclusions for 2D crystals.")
    structures: list[Structure] = Field(default_factory=list, description="Unit-cell inclusions for 3D crystals.")
    k_path: list[KPoint] = Field(min_length=2)
    points_per_segment: int = Field(16, ge=1)
    num_bands: int = Field(8, ge=1)
    polarization: Literal["te", "tm", "both"] = Field("both", description="2D only: TE = E in plane, TM = E along z.")
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


# --- photonic circuits ------------------------------------------------------


class Instance(_Model):
    component: str = Field(description="Registered compact model name, e.g. 'waveguide', 'mmi1x2', or 'sparams:<file>'.")
    settings: dict[str, Any] = Field(default_factory=dict)


class CircuitProblem(_Model):
    """A netlist of components with S-parameter models; solves the circuit S-matrix."""

    kind: Literal["circuit"] = "circuit"
    instances: dict[str, Instance]
    connections: list[tuple[str, str]] = Field(default_factory=list, description="Pairs of 'instance,port'.")
    ports: dict[str, str] = Field(description="External port name -> 'instance,port'.")
    wavelengths: Wavelengths
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)


# --- the union ------------------------------------------------------------------

from .inverse.spec import OptimizationProblem  # noqa: E402  (it wraps DeviceProblem, defined above)

Problem = Annotated[
    Union[Simulation, LayerStackProblem, PeriodicProblem, ModeProblem, DeviceProblem, BandProblem, CircuitProblem,
          OptimizationProblem],
    Field(discriminator="kind"),
]

PROBLEM_KINDS = ("scattering", "layer_stack", "periodic", "mode", "device", "bands", "circuit", "optimization")


class _Envelope(BaseModel):
    problem: Problem


def parse_problem(obj: Any) -> BaseModel:
    """Parse a problem from a model, dict or JSON string. Missing `kind` means scattering (v0.1 specs)."""
    if isinstance(obj, BaseModel):
        return obj
    if isinstance(obj, str):
        import json

        obj = json.loads(obj)
    if isinstance(obj, dict) and "kind" not in obj:
        obj = {**obj, "kind": "scattering"}
    return _Envelope.model_validate({"problem": obj}).problem


def problem_json_schema() -> dict:
    """JSON Schema covering every problem kind."""
    return _Envelope.model_json_schema()


def options_for(problem: BaseModel, backend_name: str, method: str) -> dict[str, Any]:
    opts = getattr(problem, "options", {}) or {}
    return {**opts.get(method, {}), **opts.get(backend_name, {})}
