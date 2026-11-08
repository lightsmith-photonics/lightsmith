"""Problem kinds, their schemas and examples, discovered from the library at run time.

New kinds added to ``lightsmith.problems.Problem`` (and new backends registered
for them, including plugins) show up here without changes to the server.
"""

from __future__ import annotations

import json
import typing
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from .. import problems
from ..api import AUTO_ORDER
from ..backends import all_backends

CONVENTIONS = """\
Lightsmith conventions (fixed; changing them needs a schema version bump):
- Lengths in nm. Wavelengths are vacuum wavelengths in nm.
- Time dependence exp(-i omega t): absorbing materials have Im(eps) > 0 and Im(n) > 0.
- Cross sections in nm^2; efficiencies Q = C / (pi r_eq^2), r_eq = volume-equivalent radius of all particles.
- layer_stack / periodic: layers listed from the superstrate (incidence side) to the substrate; light travels +z.
  R is measured in the superstrate, T in the substrate.
- mode: cross-section in the xy plane, propagation along +z.
- device: ports are planes whose normal is the axis where size_nm is zero; direction is into the device.
- problem.options[<method or backend name>] holds solver options, validated by that backend.
- Materials: {"kind": "constant_index", "n_real": .., "n_imag": ..}, {"kind": "library", "name": "Au"},
  drude, drude_lorentz, tabulated, constant_permittivity. FDTD needs dispersive models (fit_drude_lorentz).
- Results carry provenance and warnings: read the warnings before trusting the numbers.
"""

DESCRIPTIONS = {
    "scattering": "Single or multiple particles (spheres, spheroids, cylinders, layered spheres) under plane-wave "
                  "illumination: extinction/scattering/absorption cross sections, near and far fields.",
    "layer_stack": "Planar multilayer thin films: R/T/A vs wavelength, angle and polarization.",
    "periodic": "1D/2D-periodic patterned layers (gratings, metasurfaces, photonic-crystal slabs): R/T/A and "
                "diffraction orders.",
    "mode": "Waveguide cross-section eigenmodes: effective index, group index, loss, mode fields.",
    "device": "Photonic devices with ports (waveguides, couplers, tapers, GDS layouts): S-parameters.",
    "bands": "Photonic-crystal band structure along a k-path.",
    "circuit": "Netlists of compact-model components (waveguides, couplers, rings, Touchstone files): circuit "
               "S-parameters.",
    "optimization": "Inverse design (topology optimisation) of a device's design region against S-parameter or "
                    "field objectives.",
    "emission": "A point dipole emitter near particles: Purcell (decay-rate) enhancement, radiative enhancement, "
                "quantum efficiency of the antenna and radiation pattern.",
}

_SI = {"kind": "constant_index", "n_real": 3.48}
_SIO2 = {"kind": "constant_index", "n_real": 1.44}

# One small, valid problem per kind (each solves in seconds on a CPU, except optimization).
EXAMPLES: dict[str, dict[str, Any]] = {
    "emission": {
        "kind": "emission",
        "particles": [{"shape": {"kind": "sphere", "radius_nm": 40},
                       "material": {"kind": "library", "name": "Au", "dataset": "Johnson-Christy"}}],
        "medium_index": 1.33,
        "dipole": {"position_nm": [0, 0, 50], "orientation": [0, 0, 1]},
        "wavelengths": {"start_nm": 500, "stop_nm": 700, "num": 41},
    },
    "scattering": {
        "kind": "scattering",
        "particles": [{"shape": {"kind": "sphere", "radius_nm": 40},
                       "material": {"kind": "library", "name": "Au", "dataset": "Johnson-Christy"}}],
        "medium_index": 1.33,
        "wavelengths": {"start_nm": 400, "stop_nm": 800, "num": 81},
    },
    "layer_stack": {
        "kind": "layer_stack",
        "superstrate": {"kind": "constant_index", "n_real": 1.0},
        "layers": [{"thickness_nm": 100, "material": {"kind": "constant_index", "n_real": 1.38}}],
        "substrate": {"kind": "constant_index", "n_real": 1.52},
        "wavelengths": {"start_nm": 400, "stop_nm": 800, "num": 41},
        "angles_deg": [0, 45],
    },
    "periodic": {
        "kind": "periodic",
        "lattice_a1_nm": [400, 0], "lattice_a2_nm": [0, 400],
        "superstrate": {"kind": "constant_index", "n_real": 1.0},
        "layers": [{"thickness_nm": 150, "background": {"kind": "constant_index", "n_real": 2.0},
                    "shapes": [{"profile": {"kind": "circle", "radius_nm": 100},
                                "material": {"kind": "constant_index", "n_real": 1.0}}]}],
        "substrate": {"kind": "constant_index", "n_real": 1.45},
        "wavelengths": [560, 600, 640],
        "incidence": {"polarizations": ["s"]},
        "options": {"rcwa": {"harmonics": 3}},
    },
    "mode": {
        "kind": "mode",
        "structures": [{"profile": {"kind": "rectangle", "size_nm": [500, 220]}, "material": _SI}],
        "background": _SIO2,
        "window_size_nm": [2500, 1600],
        "wavelengths": [1550],
        "num_modes": 2,
        "options": {"fde": {"grid_spacing_nm": 40}},
    },
    "device": {
        "kind": "device",
        "dimension": 2,
        "structures": [{"geometry": {"kind": "box", "size_nm": [8000, 500, 1]}, "material": _SI}],
        "background": _SIO2,
        "domain_size_nm": [6000, 4000, 0],  # room for the absorbing layers of FDFD and FDTD outside the ports
        "ports": [{"name": "o1", "center_nm": [-2000, 0, 0], "size_nm": [0, 3000, 0], "direction": "+"},
                  {"name": "o2", "center_nm": [2000, 0, 0], "size_nm": [0, 3000, 0], "direction": "-"}],
        "wavelengths": [1550],
    },
    "bands": {
        "kind": "bands",
        "lattice_vectors_nm": [[500, 0, 0], [0, 500, 0]],
        "background": {"kind": "constant_index", "n_real": 1.0},
        "structures_2d": [{"profile": {"kind": "circle", "radius_nm": 100},
                           "material": {"kind": "constant_permittivity", "eps_real": 8.9}}],
        "k_path": [{"label": "G", "reduced": [0, 0, 0]}, {"label": "X", "reduced": [0.5, 0, 0]},
                   {"label": "M", "reduced": [0.5, 0.5, 0]}, {"label": "G", "reduced": [0, 0, 0]}],
        "points_per_segment": 8, "num_bands": 6, "polarization": "tm",
    },
    "circuit": {
        "kind": "circuit",
        "instances": {
            "split": {"component": "coupler", "settings": {"coupling": 0.5}},
            "combine": {"component": "coupler", "settings": {"coupling": 0.5}},
            "long": {"component": "waveguide", "settings": {"length_nm": 120000, "neff": 2.4, "ng": 4.2}},
            "short": {"component": "waveguide", "settings": {"length_nm": 100000, "neff": 2.4, "ng": 4.2}},
        },
        "connections": [["split,o3", "long,o1"], ["long,o2", "combine,o2"],
                        ["split,o4", "short,o1"], ["short,o2", "combine,o1"]],
        "ports": {"in1": "split,o1", "in2": "split,o2", "out1": "combine,o3", "out2": "combine,o4"},
        "wavelengths": {"start_nm": 1500, "stop_nm": 1600, "num": 201},
    },
    "optimization": {
        "kind": "optimization",
        "device": {
            "kind": "device", "dimension": 2, "background": _SIO2,
            "structures": [
                {"geometry": {"kind": "box", "center_nm": [-5600, 0, 0], "size_nm": [10000, 400, 220]}, "material": _SI},
                {"geometry": {"kind": "box", "center_nm": [0, 5600, 0], "size_nm": [400, 10000, 220]}, "material": _SI}],
            "domain_size_nm": [2600, 2600, 0], "wavelengths": [1550],
            "ports": [{"name": "o1", "center_nm": [-1050, 0, 0], "size_nm": [0, 1500, 1], "direction": "+"},
                      {"name": "o2", "center_nm": [0, 1050, 0], "size_nm": [1500, 0, 1], "direction": "-"}],
            "options": {"fdfd": {"dl_nm": 40, "pml_cells": 10}},
        },
        "design_regions": [{"size_nm": [1200, 1200, 220], "material_low": _SIO2, "material_high": _SI,
                            "filter_radius_nm": 80}],
        "objective": [{"kind": "sparameter", "out_port": "o2", "in_port": "o1"}],
        "optimizer": {"method": "mma", "max_iter": 20},
    },
}


def _union_members(tp) -> list[type[BaseModel]]:
    out: list[type[BaseModel]] = []
    for a in typing.get_args(tp):
        if isinstance(a, type) and issubclass(a, BaseModel):
            out.append(a)
        elif typing.get_args(a):
            out.extend(_union_members(a))
    return out


@lru_cache(maxsize=1)
def problem_models() -> dict[str, type[BaseModel]]:
    """kind -> pydantic model, read from the ``lightsmith.problems.Problem`` union."""
    models: dict[str, type[BaseModel]] = {}
    for m in _union_members(problems.Problem):
        f = m.model_fields.get("kind")
        if f is not None and isinstance(f.default, str):
            models[f.default] = m
    return models


def kinds() -> list[str]:
    """Every problem kind: the union's, in PROBLEM_KINDS order first, then any new ones."""
    found = problem_models()
    order = [k for k in getattr(problems, "PROBLEM_KINDS", ()) if k in found]
    return order + [k for k in found if k not in order]


def check_kind(kind: str) -> str:
    if kind not in problem_models():
        raise ValueError(f"unknown problem kind {kind!r}; known kinds: {', '.join(kinds())}")
    return kind


@lru_cache(maxsize=None)
def kind_schema(kind: str) -> dict[str, Any]:
    return TypeAdapter(problem_models()[check_kind(kind)]).json_schema()


def backends_for(kind: str) -> list[dict[str, Any]]:
    order = AUTO_ORDER.get(kind, ())
    rows = []
    for b in all_backends():
        if kind in b.problem_kinds:
            info = b.info()
            info["auto_rank"] = order.index(b.method) if b.method in order else None
            rows.append(info)
    rows.sort(key=lambda r: (r["auto_rank"] is None, r["auto_rank"] or 0, r["name"]))
    return rows


def kind_summary(kind: str) -> dict[str, Any]:
    model = problem_models()[kind]
    doc = (model.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
    rows = backends_for(kind)
    return {
        "kind": kind,
        "description": DESCRIPTIONS.get(kind, doc),
        "model": model.__name__,
        "auto_method_order": list(AUTO_ORDER.get(kind, ())),
        "methods": sorted({r["method"] for r in rows}),
        "backends_available": [r["name"] for r in rows if r["available"]],
        "has_example": kind in EXAMPLES,
        "schema_uri": f"lightsmith://schema/{kind}",
        "example_uri": f"lightsmith://examples/{kind}" if kind in EXAMPLES else None,
    }


def repo_examples_dir() -> Path | None:
    """``examples/`` of a source checkout (not shipped in wheels)."""
    d = Path(__file__).resolve().parents[3] / "examples"
    return d if d.is_dir() else None


def repo_examples() -> dict[str, Path]:
    d = repo_examples_dir()
    return {p.stem: p for p in sorted(d.glob("*.json"))} if d else {}


def example(kind_or_name: str) -> dict[str, Any]:
    if kind_or_name in EXAMPLES:
        return EXAMPLES[kind_or_name]
    files = repo_examples()
    if kind_or_name in files:
        return json.loads(files[kind_or_name].read_text(encoding="utf-8"))
    raise ValueError(f"no example named {kind_or_name!r}; available: {', '.join([*EXAMPLES, *files])}")


def capability_matrix() -> dict[str, Any]:
    """Backends x problem kinds, with GPU, licence, availability and auto-selection rank."""
    rows = []
    for b in all_backends():
        info = b.info()
        opts = getattr(b, "options_model", None)
        info["options_key"] = f"problem.options[{b.method!r}] or [{b.name!r}]"
        info["options_schema"] = opts.model_json_schema() if isinstance(opts, type) and issubclass(opts, BaseModel) \
            else None
        rows.append(info)
    return {"kinds": kinds(), "auto_order": {k: list(AUTO_ORDER.get(k, ())) for k in kinds()}, "backends": rows}
