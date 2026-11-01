"""Open technology definitions: layer stack, GDS layers and materials as data.

A :class:`Technology` says which GDS layer/datatype holds which material over
which z range, what surrounds the patterned layers (cladding, buried oxide,
substrate), and the default waveguide width. It is a pydantic model, so a
technology is plain JSON: define your own with ``Technology.model_validate``
(or :func:`load_technology` from a ``.json`` file) and add it with
:func:`register_technology`.

:func:`generic_soi` is a generic 220 nm silicon-on-insulator stack built only
from open data: silicon (Li 1980, 293 K) and fused silica (Malitson 1965) from
the bundled refractiveindex.info library (CC0), and textbook dimensions (220 nm
device layer, 2 um buried oxide, 70 nm shallow etch, 90 nm rib slab). It is not
any foundry's process.

2D (plan-view) simulations use the effective-index method: each patterned
region becomes a material whose index is the TE0 slab effective index of the
vertical stack at that point (:meth:`Technology.eim_material`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..materials import LibraryMaterial, Material, Tabulated


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LayerDef(_Model):
    """One GDS layer of a technology."""

    gds: tuple[int, int] = Field(description="(layer, datatype).")
    kind: Literal["material", "etch", "drawing"] = Field(
        "material", description="material: polygons are filled with `material` over z_min..z_max; etch: polygons "
        "remove the core over z_min..z_max (filled with the cladding); drawing: no optical effect (floorplan, "
        "labels).")
    material: Material | None = None
    z_min_nm: float | None = None
    z_max_nm: float | None = None
    sidewall_angle_deg: float = Field(0.0, description="0 = vertical; positive = narrower at the top.")
    description: str = ""

    @model_validator(mode="after")
    def _check(self):
        if self.kind != "drawing":
            if self.z_min_nm is None or self.z_max_nm is None or self.z_max_nm <= self.z_min_nm:
                raise ValueError("material and etch layers need z_min_nm < z_max_nm")
        if self.kind == "material" and self.material is None:
            raise ValueError("material layers need a material")
        return self


class Technology(_Model):
    """A planar photonic technology: patterned layers on a buried oxide, under a cladding.

    z = 0 is the top of the buried oxide (bottom of the device layer). Layers
    are listed in override order: later layers win where polygons overlap
    (so an etch layer listed after the core removes part of it).
    """

    name: str
    description: str = ""
    layers: dict[str, LayerDef] = Field(description="Layer name -> definition, in override order.")
    core_layer: str = Field("WG", description="Layer of the waveguide core (sets ports and mode solves).")
    cladding: Material = Field(description="Material above and around the patterned layers.")
    box: Material = Field(description="Buried oxide below z = 0.")
    box_thickness_nm: float = Field(2000.0, gt=0)
    substrate: Material | None = Field(None, description="Handle wafer below the buried oxide (None = omit).")
    wg_width_nm: float = Field(500.0, gt=0, description="Default single-mode waveguide width.")
    min_feature_nm: float = Field(100.0, gt=0, description="Smallest drawn width or gap the components use.")
    min_bend_radius_nm: float = Field(2000.0, gt=0)
    polarization: Literal["te", "tm"] = Field("te", description="Mode family: te = E mostly in the wafer plane.")
    sources: list[str] = Field(default_factory=list, description="Where the numbers come from (open data only).")

    @model_validator(mode="after")
    def _core(self):
        if self.core_layer not in self.layers or self.layers[self.core_layer].kind != "material":
            raise ValueError(f"core_layer {self.core_layer!r} must name a material layer")
        return self

    # --- layers ---------------------------------------------------------------

    @property
    def core(self) -> LayerDef:
        return self.layers[self.core_layer]

    def layer(self, name: str) -> LayerDef:
        try:
            return self.layers[name]
        except KeyError:
            raise KeyError(f"technology {self.name!r} has no layer {name!r}; layers: {', '.join(self.layers)}") from None

    def gds_layer(self, name: str) -> tuple[int, int]:
        return self.layer(name).gds

    def layer_map(self) -> dict[tuple[int, int], dict[str, Any]]:
        """GDS layer -> {material, z_min_nm, z_max_nm, ...} for :mod:`lightsmith.layout` (3D extrusion).

        Etch layers are filled with the cladding, which overrides the core where they overlap."""
        out = {}
        for name, ld in self.layers.items():
            if ld.kind == "drawing":
                continue
            mat = ld.material if ld.kind == "material" else self.cladding
            out[ld.gds] = {"material": mat, "z_min_nm": ld.z_min_nm, "z_max_nm": ld.z_max_nm,
                           "sidewall_angle_deg": ld.sidewall_angle_deg, "name": name}
        return out

    def core_thickness_nm(self, layer: str | None = None) -> float:
        """Thickness of core material left where `layer` is drawn (an etch layer thins the core)."""
        core = self.core
        ld = self.layer(layer or self.core_layer)
        if ld.kind == "etch":
            lo, hi = max(core.z_min_nm, ld.z_min_nm), min(core.z_max_nm, ld.z_max_nm)
            return (core.z_max_nm - core.z_min_nm) - max(hi - lo, 0.0)
        if ld.kind == "drawing":
            return 0.0
        return ld.z_max_nm - ld.z_min_nm

    # --- effective-index method ------------------------------------------------

    def _core_material(self, layer: str) -> Material:
        ld = self.layer(layer)
        return ld.material if ld.kind == "material" else self.core.material

    def slab_neff(self, thickness_nm: float, wavelengths_nm, layer: str | None = None, order: int = 0) -> np.ndarray:
        """Effective index of the box / core(thickness) / cladding slab, for this technology's polarization."""
        wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
        mat = self._core_material(layer or self.core_layer)
        n_core = np.real(mat.n(wl))
        n_box, n_clad = np.real(self.box.n(wl)), np.real(self.cladding.n(wl))
        return np.array([asymmetric_slab_neff(a, b, c, thickness_nm, w, self.polarization, order)
                         for a, b, c, w in zip(n_box, n_core, n_clad, wl)])

    def eim_material(self, layer: str, wavelengths_nm, step_nm: float = 2.0) -> Material:
        """2D material for regions drawn on `layer`: the slab effective index, tabulated over the wavelengths."""
        wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
        lo, hi = float(wl.min()), float(wl.max())
        n = max(int(math.ceil((hi - lo) / step_nm)) + 1, 2)
        grid = np.linspace(lo - step_nm, hi + step_nm, n + 2)
        key = (self.fingerprint(), round(self.core_thickness_nm(layer), 6), layer, tuple(np.round(grid, 6)))
        if key not in _SLAB_CACHE:
            _SLAB_CACHE[key] = tuple(float(v) for v in self.slab_neff(key[1], grid, layer))
        neff = _SLAB_CACHE[key]
        return Tabulated(wavelength_nm=grid.tolist(), n_real=list(neff), n_imag=[0.0] * len(grid),
                         source=f"{self.name}: TE0 slab effective index of {self.core_thickness_nm(layer):g} nm core "
                         f"({layer})")

    def eim_cladding(self, wavelengths_nm) -> Material:
        """2D background material: the cladding (no core drawn)."""
        return self.cladding

    def waveguide_neff(self, wavelengths_nm, width_nm: float | None = None, dimension: int = 2, mode: int = 0,
                       layer: str | None = None) -> np.ndarray:
        """Effective index of a straight waveguide of `width_nm` on `layer` (default: the core, default width).

        dimension=2: symmetric slab of the EIM core index in the cladding, E in plane (matches 2D EME/FDFD
        with polarization 'te'); dimension=3: full-vector FDE of the extruded cross-section.
        """
        wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
        width = float(width_nm or self.wg_width_nm)
        layer = layer or self.core_layer
        if dimension == 2:
            from ..backends._fdfd.modes import slab_neff_analytic

            n_core = self.slab_neff(self.core_thickness_nm(layer), wl, layer)
            n_clad = np.real(self.cladding.n(wl))
            pol2d = "te" if self.polarization == "te" else "tm"
            return np.array([slab_neff_analytic(a, b, width, w, pol2d, mode) for a, b, w in zip(n_core, n_clad, wl)])
        from ..api import solve
        from ..problems import ModeProblem, Rectangle, Structure2D

        ld = self.layer(layer)
        t = ld.z_max_nm - ld.z_min_nm
        zc = (ld.z_min_nm + ld.z_max_nm) / 2
        mp = ModeProblem(structures=[Structure2D(profile=Rectangle(center_nm=(-5000, zc - t / 2 - 2500),
                                                                    size_nm=(10_000 + 2 * width, 5000)),
                                                 material=self.box),
                                     Structure2D(profile=Rectangle(center_nm=(0, zc), size_nm=(width, t)),
                                                 material=self._core_material(layer))],
                         background=self.cladding, window_center_nm=(0, zc), window_size_nm=(width + 3000, t + 3000),
                         wavelengths=wl.tolist(), num_modes=mode + 2)
        res = solve(mp)
        return np.array([sorted((m.neff_real for m in ms), reverse=True)[mode] for ms in res.modes])

    def group_index(self, wavelengths_nm, width_nm: float | None = None, dimension: int = 2,
                    layer: str | None = None, dl_nm: float = 1.0) -> np.ndarray:
        """ng = neff - lambda dneff/dlambda (central difference)."""
        wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
        n0 = self.waveguide_neff(wl, width_nm, dimension, layer=layer)
        dn = (self.waveguide_neff(wl + dl_nm, width_nm, dimension, layer=layer)
              - self.waveguide_neff(wl - dl_nm, width_nm, dimension, layer=layer)) / (2 * dl_nm)
        return n0 - wl * dn

    def fingerprint(self) -> str:
        """Stable JSON of the whole definition (used in model cache keys)."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)


_SLAB_CACHE: dict[tuple, tuple[float, ...]] = {}


def asymmetric_slab_neff(n_sub: float, n_core: float, n_clad: float, thickness_nm: float, wavelength_nm: float,
                         pol: str = "te", order: int = 0) -> float:
    """Guided-mode effective index of a three-layer slab (substrate / core / cladding).

    Solves k t = order pi + atan(r_s g_s / k) + atan(r_c g_c / k) by bisection, with
    k = k0 sqrt(n_core^2 - neff^2), g = k0 sqrt(neff^2 - n^2) and r = 1 (TE: E parallel
    to the interfaces) or n_core^2 / n^2 (TM). Marcuse, Theory of Dielectric Optical
    Waveguides (1991), ch. 1.
    """
    k0 = 2 * math.pi / wavelength_nm
    rs, rc = (1.0, 1.0) if pol == "te" else (n_core**2 / n_sub**2, n_core**2 / n_clad**2)

    def f(neff):
        kap = k0 * math.sqrt(max(n_core**2 - neff**2, 0.0))
        gs = k0 * math.sqrt(max(neff**2 - n_sub**2, 0.0))
        gc = k0 * math.sqrt(max(neff**2 - n_clad**2, 0.0))
        return kap * thickness_nm - math.atan2(rs * gs, kap) - math.atan2(rc * gc, kap) - order * math.pi

    lo, hi = max(n_sub, n_clad) + 1e-12, n_core - 1e-12
    if not lo < hi or f(lo) <= 0:
        raise ValueError(f"no guided mode of order {order} in a {thickness_nm:g} nm slab at {wavelength_nm:g} nm")
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- the generic open SOI stack ---------------------------------------------------------

SILICON = LibraryMaterial(name="Si", dataset="Li-293K")
SILICA = LibraryMaterial(name="SiO2", dataset="Malitson")


def generic_soi() -> Technology:
    """Generic 220 nm SOI: Si device layer on 2 um buried oxide with oxide cladding (open data only)."""
    return Technology(
        name="generic_soi",
        description="Generic 220 nm silicon-on-insulator: 220 nm Si on 2 um SiO2, SiO2 cladding, 70 nm shallow "
                    "etch, 90 nm rib slab. Textbook dimensions; not a foundry process.",
        layers={
            "SLAB": LayerDef(gds=(3, 0), material=SILICON, z_min_nm=0.0, z_max_nm=90.0,
                             description="90 nm rib slab (rib waveguides, slab regions)."),
            "WG": LayerDef(gds=(1, 0), material=SILICON, z_min_nm=0.0, z_max_nm=220.0,
                           description="Full-height 220 nm silicon core."),
            "SHALLOW_ETCH": LayerDef(gds=(2, 0), kind="etch", z_min_nm=150.0, z_max_nm=220.0,
                                     description="70 nm partial etch (grating teeth); draw inside WG."),
            "FLOORPLAN": LayerDef(gds=(99, 0), kind="drawing", description="Cell outline."),
        },
        core_layer="WG",
        cladding=SILICA,
        box=SILICA,
        box_thickness_nm=2000.0,
        substrate=SILICON,
        wg_width_nm=500.0,
        min_feature_nm=100.0,
        min_bend_radius_nm=2000.0,
        sources=["Si: H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980), via refractiveindex.info (CC0)",
                 "SiO2: I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965), via refractiveindex.info (CC0)"],
    )


_TECHNOLOGIES: dict[str, Technology] = {}


def register_technology(tech: Technology, overwrite: bool = False) -> Technology:
    """Make `tech` available by name (e.g. in circuit netlists: settings.technology)."""
    if tech.name in _TECHNOLOGIES and not overwrite and _TECHNOLOGIES[tech.name] != tech:
        raise ValueError(f"a different technology named {tech.name!r} is already registered; pass overwrite=True")
    _TECHNOLOGIES[tech.name] = tech
    return tech


def get_technology(tech: str | dict | Technology | None = None) -> Technology:
    """A technology by name, JSON dict, file path (.json) or instance; None = generic_soi."""
    if tech is None:
        tech = "generic_soi"
    if isinstance(tech, Technology):
        return tech
    if isinstance(tech, dict):
        return Technology.model_validate(tech)
    if tech in _TECHNOLOGIES:
        return _TECHNOLOGIES[tech]
    if str(tech).endswith(".json") and Path(tech).is_file():
        return load_technology(tech)
    raise KeyError(f"unknown technology {tech!r}; registered: {', '.join(_TECHNOLOGIES)}")


def load_technology(path: str | Path) -> Technology:
    """Read a technology from a JSON file (the output of ``tech.model_dump_json()``)."""
    return Technology.model_validate_json(Path(path).read_text(encoding="utf-8"))


def list_technologies() -> list[str]:
    return sorted(_TECHNOLOGIES)


register_technology(generic_soi())
