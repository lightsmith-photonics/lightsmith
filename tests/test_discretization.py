"""lightsmith.discretization: what each backend will discretise, without solving."""

from __future__ import annotations

import json

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends import all_backends
from lightsmith.cli import main as cli_main
from lightsmith.problems import parse_problem

N = lambda n: {"kind": "constant_index", "n_real": n, "n_imag": 0.0}
SI, SIO2 = N(3.48), N(1.444)
PRISM = lambda w, l: {"kind": "prism", "profile": {"kind": "rectangle", "center_nm": [0, 0], "size_nm": [l, w]},
                      "z_min_nm": -110, "z_max_nm": 110, "sidewall_angle_deg": 0}

SPHERE = {"kind": "scattering", "particles": [{"shape": {"kind": "sphere", "radius_nm": 40}, "material": N(2.0)}],
          "medium_index": 1.0, "wavelengths": {"start_nm": 500, "stop_nm": 800, "num": 7}}
STACK = {"kind": "layer_stack", "superstrate": N(1.0), "layers": [{"thickness_nm": 100, "material": N(2.0)}],
         "substrate": N(1.45), "wavelengths": {"start_nm": 500, "stop_nm": 800, "num": 5}, "angles_deg": [0],
         "polarizations": ["s"]}
GRATING = {"kind": "periodic", "lattice_a1_nm": [600, 0], "superstrate": N(1.0),
           "layers": [{"thickness_nm": 200, "background": N(1.0),
                       "shapes": [{"profile": {"kind": "rectangle", "center_nm": [0, 0], "size_nm": [300, 300]},
                                   "material": N(2.0)}]}],
           "substrate": N(1.45), "wavelengths": {"start_nm": 600, "stop_nm": 800, "num": 5},
           "incidence": {"theta_deg": 0, "phi_deg": 0, "polarizations": ["s"]}}
MODE = {"kind": "mode", "structures": [{"profile": {"kind": "rectangle", "center_nm": [0, 0], "size_nm": [500, 220]},
                                        "material": SI}],
        "background": SIO2, "window_center_nm": [0, 0], "window_size_nm": [2000, 1500], "wavelengths": [1550],
        "num_modes": 2}
DEVICE = {"kind": "device", "dimension": 2, "structures": [{"geometry": PRISM(450, 6000), "material": SI}],
          "background": SIO2, "domain_center_nm": [0, 0, 0], "domain_size_nm": [3000, 2000, 0],
          "ports": [{"name": "o1", "center_nm": [-1100, 0, 0], "size_nm": [0, 1600, 1], "direction": "+"},
                    {"name": "o2", "center_nm": [1100, 0, 0], "size_nm": [0, 1600, 1], "direction": "-"}],
          "wavelengths": [1550], "options": {"fdfd": {"dl_nm": 25.0}, "fdtd": {"spacing_nm": 25, "polarization": "tm"}}}
BANDS = {"kind": "bands", "dimension": 2, "lattice_vectors_nm": [[500, 0, 0], [0, 500, 0]], "background": N(1.0),
         "structures_2d": [{"profile": {"kind": "circle", "center_nm": [0, 0], "radius_nm": 100},
                            "material": {"kind": "constant_permittivity", "eps_real": 8.9, "eps_imag": 0.0}}],
         "k_path": [{"label": "G", "reduced": [0, 0, 0]}, {"label": "X", "reduced": [0.5, 0, 0]}],
         "points_per_segment": 4, "num_bands": 4, "polarization": "tm"}
EMISSION = {"kind": "emission", "particles": SPHERE["particles"], "medium_index": 1.0,
            "dipole": {"position_nm": [0, 0, 60], "orientation": [0, 0, 1]}, "wavelengths": [600, 700]}

EXPECTED = [
    (SPHERE, "lightsmith-dda", "dipole_lattice"),
    (SPHERE, "lightsmith-bem", "surface_mesh"),
    (SPHERE, "lightsmith-mie", "multipoles"),
    (SPHERE, "miepython", "multipoles"),
    (SPHERE, "treams", "multipoles"),
    (SPHERE, "lightsmith-fdtd", "yee_grid"),
    (SPHERE, "lightsmith-quasistatic", "none"),
    (STACK, "lightsmith-tmm", "none"),
    (STACK, "lightsmith-rcwa", "harmonics"),
    (STACK, "lightsmith-fdtd", "unit_cell"),
    (GRATING, "lightsmith-rcwa", "harmonics"),
    (GRATING, "lightsmith-fdtd", "unit_cell"),
    (MODE, "lightsmith-fde", "cross_section_grid"),
    (DEVICE, "lightsmith-eme", "cells"),
    (DEVICE, "lightsmith-bpm", "cells"),
    (DEVICE, "lightsmith-fdfd", "device_grid"),
    (DEVICE, "lightsmith-fdtd", "device_grid"),
    (BANDS, "lightsmith-pwe", "plane_waves"),
    (EMISSION, "lightsmith-fdtd", "yee_grid"),
]


def _available(name: str) -> bool:
    return any(b.name == name and b.available()[0] for b in all_backends())


@pytest.fixture
def no_solves(monkeypatch):
    """Every backend's solve raises: a layout must never run a solver."""
    for b in all_backends():
        monkeypatch.setattr(b, "solve", classmethod(lambda cls, *a, **k: pytest.fail(f"{cls.name}.solve was called")))


@pytest.mark.parametrize("spec,backend,kind", EXPECTED, ids=[f"{s['kind']}-{b}" for s, b, _ in EXPECTED])
def test_every_backend_has_its_layout(spec, backend, kind, no_solves):
    if not _available(backend):
        pytest.skip(f"{backend} not installed")
    out = ls.discretization(spec, backend)
    assert out["backend"] == backend and out["problem_kind"] == spec["kind"]
    assert out["layout_kind"] == kind
    assert out["error"] is None, out["error"]
    assert out["summary"] and isinstance(out["payload"], dict)
    assert out["estimate"]["backend"] == backend
    json.dumps(out, allow_nan=False)  # plain JSON


def test_dda_lattice_is_the_solve_lattice():
    from lightsmith.backends import dda

    out = ls.discretization(SPHERE, "dda")
    p = out["payload"]
    assert p["n_dipoles"] == out["estimate"]["dipoles"]
    pts = np.asarray(p["points_nm"])
    assert np.all(np.linalg.norm(pts, axis=1) <= 40 + 1e-6)
    assert p["spacing_nm"] == pytest.approx(out["estimate"]["dipole_spacing_nm"])
    assert len(pts) == -(-p["n_dipoles"] // p["stride"])
    assert dda  # imported for the private lattice used above


def test_bem_mesh_matches_the_estimate():
    out = ls.discretization(SPHERE, "bem")
    p = out["payload"]
    assert p["n_triangles"] == out["estimate"]["triangles"] and 2 * p["n_edges"] == out["estimate"]["unknowns"]
    v = np.asarray(p["verts_nm"])
    assert np.allclose(np.linalg.norm(v, axis=1), 40, rtol=0.05)


def test_fdtd_boxes_nest_and_mirror_planes():
    out = ls.discretization(SPHERE, "fdtd")
    p = out["payload"]
    box = {b["name"]: b for b in p["boxes"]}
    inside = lambda a, b: all(x >= y - 1e-9 for x, y in zip(a["lo_nm"], b["lo_nm"])) and all(
        x <= y + 1e-9 for x, y in zip(a["hi_nm"], b["hi_nm"]))
    assert inside(box["flux_total"], box["tfsf"]) and inside(box["tfsf"], box["flux_scattered"])
    assert p["symmetry"]["planes_nm"] == {"x": 0.0, "y": 0.0}
    assert out["estimate"]["cells"] < p["grid"]["cells"] / 3.5  # the run steps a quarter (two mirrors)


def test_fdtd_device_ports_and_polarization():
    out = ls.discretization(DEVICE, "fdtd")
    p = out["payload"]
    assert [q["name"] for q in p["ports"]] == ["o1", "o2"]
    assert p["ports"][0]["source_nm"] == pytest.approx(-1100, abs=25)
    assert abs(p["ports"][0]["monitor_nm"] - p["ports"][0]["source_nm"]) == pytest.approx(50, abs=1e-6)
    assert out["summary"]["polarization"].startswith("tm")
    assert p["two_d"] and len(p["nodes_nm"][2]) == 3


def test_fdtd_unit_cell_planes_are_ordered():
    p = ls.discretization(GRATING, "fdtd")["payload"]
    z = p["planes_nm"]
    assert z["reflection monitor"] < z["source"] < z["stack top"] == 0 < z["stack bottom"] == 200 < z["transmission monitor"]
    assert p["n_cells"][1] == 1 and p["invariant"] == [False, True]
    assert len(p["x_nm"]) == p["n_cells"][0] + 1


def test_mie_orders_follow_the_size_parameter():
    p = ls.discretization(SPHERE, "lightsmith-mie")["payload"]
    assert p["n_max"] == sorted(p["n_max"], reverse=True)  # shorter wavelengths need more orders
    if _available("miepython"):
        q = ls.discretization(SPHERE, "miepython")["payload"]
        assert q["rule"] == "miepython's wiscombe_terms" and len(q["n_max"]) == 7


def test_eme_cells_cover_the_device_and_bpm_steps():
    eme = ls.discretization(DEVICE, "eme")["payload"]
    assert eme["cells"][0]["start_nm"] == pytest.approx(eme["lo_nm"]) and eme["cells"][-1]["stop_nm"] == pytest.approx(eme["hi_nm"])
    bpm = ls.discretization(DEVICE, "bpm")["payload"]
    assert all(c["dz_nm"] <= bpm["step_nm"] + 1e-9 for c in bpm["cells"])


def test_rcwa_harmonics_and_pwe_path():
    r = ls.discretization(GRATING, "rcwa")["payload"]
    assert len(r["harmonics"]) == 2 * r["orders_max"][0] + 1
    w = ls.discretization(BANDS, "pwe")["payload"]
    assert len(w["k_reduced"]) == 5 and w["k_labels"] == {"0": "G", "4": "X"}


def test_refusals_go_to_error_with_the_geometry():
    beam = {**SPHERE, "illumination": {"kind": "gaussian_beam", "waist_nm": 1000},
            "options": {"fdtd": {"spacing_nm": 8, "mesh_override": [{"center_nm": [0, 0, 0], "size_nm": [40, 40, 40],
                                                                      "spacing_nm": 2}]}}}
    out = ls.discretization(beam, "fdtd")
    assert "plane-wave illumination only" in out["error"]
    assert out["payload"]["grid"]["kind"] == "graded"


def test_unsupported_backend_raises_like_solve():
    with pytest.raises(ValueError):
        ls.discretization(STACK, "dda")


@pytest.mark.skipif(not _available("lightsmith-inverse"), reason="needs torch")
def test_inverse_design_grid_and_regions_use_the_merged_options():
    dev = {**DEVICE, "options": {"fdfd": {"dl_nm": 50.0}}}
    opt = {"kind": "optimization", "device": dev, "options": {"fdfd": {"dl_nm": 40.0}},
           "design_regions": [{"name": "d", "center_nm": [0, 0, 0], "size_nm": [1000, 1000, 220],
                               "material_low": SIO2, "material_high": SI}],
           "objective": [{"kind": "sparameter", "out_port": "o2", "in_port": "o1"}], "optimizer": {"max_iter": 2}}
    out = ls.discretization(opt)
    assert out["summary"]["cell size (nm)"] == 40.0  # the optimisation's options win, as in the optimiser
    assert out["estimate"]["dl_nm"] == 40.0
    assert all(abs(n - 1000 / 40) <= 1.5 for n in out["payload"]["design_regions"][0]["shape"])  # 1 µm at 40 nm


def test_cli_discretization(tmp_path, capsys):
    f = tmp_path / "s.json"
    f.write_text(json.dumps(MODE))
    assert cli_main(["discretization", str(f)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["layout_kind"] == "cross_section_grid" and out["backend"] == "lightsmith-fde"


def test_the_option_descriptions_name_no_other_product():
    from lightsmith.backends._fdtd.options import FDTDOptions

    text = json.dumps(FDTDOptions.model_json_schema())
    for name in ("Lumerical", "Meep", "Tidy3D", "COMSOL"):
        assert name not in text
    assert parse_problem(SPHERE)
