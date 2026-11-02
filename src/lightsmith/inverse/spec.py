"""JSON-describable inverse-design problems (``kind = "optimization"``).

An :class:`OptimizationProblem` wraps a :class:`~lightsmith.problems.DeviceProblem`
(the fixed parts: waveguides, ports, wavelengths), one or more design regions,
a weighted objective on S-parameters and/or fields, and optimiser settings.
Lengths in nm, vacuum wavelengths, exp(-i omega t), as everywhere in lightsmith.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..materials import Material
from ..problems import DeviceProblem


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DesignRegion(_Model):
    """A box whose permittivity is free: density 0 -> material_low, 1 -> material_high."""

    name: str = "design"
    center_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_nm: tuple[float, float, float] = Field(description="Box size; z is ignored in 2D. In 3D, z is the "
                                                "thickness of an extruded region (e.g. the 220 nm device layer).")
    material_low: Material = Field(description="Material at density 0 (usually the cladding).")
    material_high: Material = Field(description="Material at density 1 (usually the core).")
    init: float | Literal["random"] = Field(0.5, description="Initial density, or 'random' (seeded).")
    seed: int = 0
    filter_radius_nm: float = Field(0.0, ge=0, description="Conic filter radius; 0 = no filter.")
    eta: float = Field(0.5, gt=0, lt=1, description="Projection threshold.")
    min_feature_nm: float | None = Field(
        None, gt=0, description="Minimum solid and void feature size, imposed with lengthscale penalties "
        "(needs filter_radius_nm >= min_feature_nm / 2).")
    symmetry: list[Literal["x", "y", "z"]] = Field(
        default_factory=list, description="Mirror symmetry: 'x' = symmetric under x -> -x about the region "
        "centre, 'y' likewise; 'z' only for volumetric 3D regions.")
    parametrization: Literal["extruded", "volumetric"] = Field(
        "extruded", description="3D devices only (ignored in 2D). 'extruded': one 2D density in x-y, constant "
        "through the region's thickness size_nm[2], i.e. a single vertical etch (planar fabrication; exports to "
        "GDS). 'volumetric': a free 3D density on the grid nodes (filter and min-feature constraints act in 3D; "
        "exported layer by layer).")

    @model_validator(mode="after")
    def _check(self):
        if self.min_feature_nm is not None and self.filter_radius_nm < self.min_feature_nm / 2:
            raise ValueError("min_feature_nm needs filter_radius_nm >= min_feature_nm / 2")
        if "z" in self.symmetry and self.parametrization != "volumetric":
            raise ValueError("symmetry 'z' needs parametrization='volumetric' (an extruded density is uniform in z)")
        return self


Goal = Literal["maximize", "minimize", "target"]


class _Term(_Model):
    weight: float = Field(1.0, gt=0)
    goal: Goal = "maximize"
    target: float | None = Field(None, description="For goal='target': penalise (value - target)^2.")
    aggregate: Literal["mean", "min", "max"] = Field(
        "mean", description="Over wavelengths. 'min' (worst-case for maximize) uses a smooth minimum.")
    wavelength_weights: list[float] | None = Field(None, description="Per-wavelength weights for 'mean'.")

    @model_validator(mode="after")
    def _target(self):
        if self.goal == "target" and self.target is None:
            raise ValueError("goal='target' needs a target value")
        return self


class SParameterTerm(_Term):
    """|S[out <- in]|^2 (power) of port modes, e.g. transmission o1 -> o2 or reflection o1 -> o1."""

    kind: Literal["sparameter"] = "sparameter"
    out_port: str
    in_port: str
    out_mode: int = Field(0, ge=0)
    in_mode: int = Field(0, ge=0)
    quantity: Literal["power", "real", "imag"] = "power"


class FieldIntensityTerm(_Term):
    """Mean |field|^2 (Hz for TE, Ez for TM; |E|^2 in 3D) in a box, with one port mode as the source."""

    kind: Literal["field_intensity"] = "field_intensity"
    source_port: str
    source_mode: int = Field(0, ge=0)
    center_nm: tuple[float, float, float]
    size_nm: tuple[float, float, float] = Field(description="Box size; zero sizes select a line or point.")


ObjectiveTerm = Annotated[Union[SParameterTerm, FieldIntensityTerm], Field(discriminator="kind")]


class OptimizerSettings(_Model):
    method: Literal["mma", "adam", "lbfgsb"] = Field(
        "mma", description="mma = globally convergent MMA (monotone; Svanberg 2002); adam; lbfgsb = SciPy L-BFGS-B.")
    max_iter: int = Field(100, ge=1, description="Total objective/gradient evaluations budget (outer iterations).")
    learning_rate: float = Field(0.02, gt=0, description="Adam step size (densities are in [0, 1]).")
    move_limit: float = Field(0.2, gt=0, le=1, description="MMA max change per iteration.")
    beta_start: float = Field(4.0, ge=0, description="Initial projection sharpness.")
    beta_max: float = Field(64.0, ge=0)
    beta_growth: float = Field(2.0, ge=1, description="beta *= beta_growth every beta_every iterations (continuation).")
    beta_every: int = Field(20, ge=1)
    lengthscale_weight: float = Field(100.0, ge=0, description="Penalty weight of the min-feature constraints.")
    lengthscale_from_beta: float = Field(16.0, ge=0, description="Apply min-feature penalties once beta >= this.")
    ftol: float = Field(0.0, ge=0, description="Stop when the relative improvement over 10 iterations is below this.")


class OptimizationProblem(_Model):
    """Topology optimisation of a device: maximise sum(weight * term) (minimise terms with goal='minimize').

    ``device.dimension`` selects 2D or 3D; in 3D the FDFD backend solves full-vector
    Maxwell with an iterative solver (see ``lightsmith.backends._fdfd3d``).
    """

    kind: Literal["optimization"] = "optimization"
    device: DeviceProblem
    design_regions: list[DesignRegion] = Field(min_length=1)
    objective: list[ObjectiveTerm] = Field(min_length=1)
    optimizer: OptimizerSettings = OptimizerSettings()
    backend: str = Field("lightsmith-fdfd", description="Any backend with differentiable=True and "
                         "differentiable_solve (see lightsmith.inverse).")
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def to_json(self, **kw) -> str:
        return self.model_dump_json(**kw)

    @classmethod
    def from_json(cls, s: str) -> "OptimizationProblem":
        return cls.model_validate_json(s)
