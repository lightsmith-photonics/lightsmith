import os
import re

import numpy as np
import pytest
from pydantic import ValidationError

import lightsmith as ls
from lightsmith import materials as M
from lightsmith.materials import LibraryMaterial, MaterialRangeError, list_materials, load_material
from lightsmith.materials_fit import DrudeLorentzFit, fit_drude_lorentz, fit_quality

BAND = np.linspace(450.0, 900.0, 91)
NETWORK = os.environ.get("LIGHTSMITH_NETWORK_TESTS") == "1"


# --- bundled library ----------------------------------------------------------


def test_list_materials_rows():
    rows = list_materials()
    names = {r["name"] for r in rows}
    assert {"Au", "Ag", "Al", "Cu", "Si", "TiO2", "SiO2", "H2O"} <= names
    au = [r for r in rows if r["name"] == "Au" and r["default"]]
    assert len(au) == 1 and au[0]["dataset"] == "Johnson-Christy"
    assert au[0]["material_json"] == {"kind": "library", "name": "Au", "dataset": "Johnson-Christy"}
    assert au[0]["doi"] == "10.1103/PhysRevB.6.4370" and au[0]["license"] == "CC0-1.0"
    for r in rows:  # every row is a usable spec with exactly one default per material
        LibraryMaterial.model_validate(r["material_json"])
    assert all(sum(r["default"] for r in rows if r["name"] == n) == 1 for n in names)
    assert {r["name"] for r in list_materials("silver")} == {"Ag"}
    with pytest.raises(ValueError, match="unknown library material"):
        list_materials("unobtainium")


@pytest.mark.parametrize("row", list_materials(), ids=lambda r: f"{r['name']}-{r['dataset']}")
def test_bundled_dataset_matches_manifest(row):
    """Each bundled file loads, covers its stated range, is the cited dataset and carries the CC0 notice."""
    info = M._manifest()["materials"][row["name"]]["datasets"][row["dataset"]]
    text = M._data_dir().joinpath(*info["file"].split("/")).read_text(encoding="utf-8")
    assert "copyright and related rights waived via CC0 1.0" in text
    assert row["doi"] in M._rii_dois(M._parse_rii_yaml(text))
    assert info["source_url"].endswith(f"/database/data/{info['shelf']}/{info['book']}/nk/{info['page']}.yml")
    assert M._manifest()["commit"] in info["source_url"]
    tab = load_material(row["name"], row["dataset"])
    lo, hi = tab.range_nm
    assert lo <= row["range_nm"][0] * (1 + 1e-3) and hi >= row["range_nm"][1] * (1 - 1e-3)
    assert np.all(np.asarray(tab.n_real) > 0) and np.all(np.asarray(tab.n_imag) >= 0)
    assert row["reference"] in tab.source and "refractiveindex.info" in tab.source


def test_aliases_resolve_to_canonical_names():
    m = LibraryMaterial(name="gold", dataset="JC")
    assert (m.name, m.dataset) == ("Au", "Johnson-Christy")
    assert LibraryMaterial(name="AU").dataset == "Johnson-Christy"  # default filled in, case-insensitive
    assert LibraryMaterial(name="silver", dataset="johnson").name == "Ag"
    assert LibraryMaterial(name="water").dataset == "Hale-Querry"
    with pytest.raises(ValidationError, match="unknown dataset"):
        LibraryMaterial(name="Au", dataset="Palik")


def test_library_json_roundtrip_in_simulation():
    sim = ls.Simulation.model_validate({
        "particles": [{"shape": {"kind": "sphere", "radius_nm": 10},
                       "material": {"kind": "library", "name": "gold", "dataset": "JC"}}],
        "medium_index": 1.33, "wavelengths": [500.0, 600.0],
    })
    mat = sim.particles[0].material
    assert isinstance(mat, LibraryMaterial)
    dumped = sim.model_dump(mode="json")["particles"][0]["material"]
    assert dumped == {"kind": "library", "name": "Au", "dataset": "Johnson-Christy"}
    again = ls.Simulation.from_json(sim.to_json())
    assert again == sim
    assert np.allclose(again.particles[0].material.eps([550.0]), mat.eps([550.0]))


def test_spot_values_gold_near_600nm():
    # Rows of database/data/main/Au/nk/Johnson.yml: "0.5821 0.29 2.863" and "0.6168 0.21 3.272" (um, n, k)
    au = LibraryMaterial(name="Au")
    assert au.eps([582.1])[0] == pytest.approx(complex(0.29, 2.863) ** 2, rel=1e-12)
    assert au.n([616.8])[0] == pytest.approx(complex(0.21, 3.272), rel=1e-12)
    f = (600.0 - 582.1) / (616.8 - 582.1)  # linear interpolation in n and k
    n600 = complex(0.29 + f * (0.21 - 0.29), 2.863 + f * (3.272 - 2.863))
    assert au.eps([600.0])[0] == pytest.approx(n600**2, rel=1e-12)
    assert au.eps([600.0])[0].real == pytest.approx(-9.4, abs=0.2)  # JC Au, eps ~ -9.4 + 1.5i at 600 nm
    assert au.eps([600.0])[0].imag == pytest.approx(1.5, abs=0.2)


def test_formula_entries():
    # Malitson fused silica n_D = 1.4585; Daimon water (24 C) n_D = 1.3330; Devore rutile n_o(589) = 2.613
    assert load_material("SiO2").n([589.3])[0].real == pytest.approx(1.4584, abs=2e-4)
    assert load_material("water", "Daimon").n([589.3])[0].real == pytest.approx(1.3330, abs=2e-4)
    assert load_material("TiO2", "Devore").n([589.3])[0].real == pytest.approx(2.613, abs=2e-3)


def test_out_of_range_error_message():
    with pytest.raises(MaterialRangeError) as e:
        LibraryMaterial(name="Au").eps([150.0, 600.0])
    msg = str(e.value)
    assert "Au (Johnson-Christy) has data for 187.9-1937 nm only" in msg and "requested 150-600 nm" in msg
    assert "fetch_refractiveindex" in msg
    with pytest.raises(MaterialRangeError, match=re.escape("Datasets covering it: 'Green-2008' (250-1450 nm)")):
        LibraryMaterial(name="Si", dataset="Aspnes").eps([900.0])


def test_yaml_parser_formats():
    doc = M._parse_rii_yaml(
        "# comment\nREFERENCES: |\n    A. Author.\n    <a href=\"https://doi.org/10.1/x\"><i>J</i> 1 (2000)</a>\n"
        "DATA:\n  -   type: tabulated n\n      data: |\n          0.6 1.5\n          0.5 1.4\n"
        "  - type: tabulated k\n    data: |\n        0.5 0.1\n        0.6 0.2\n")
    wl, n, k = M._rii_nk(doc)
    assert np.allclose(wl, [500, 600]) and np.allclose(n, [1.4, 1.5]) and np.allclose(k, [0.1, 0.2])
    assert M._rii_dois(doc) == ["10.1/x"] and M._rii_reference(doc) == "A. Author. J 1 (2000)"


def test_mie_gold_sphere_plasmon_peak():
    # 20 nm diameter Au sphere in water: dipolar plasmon at ~520-525 nm (JC data)
    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=10.0), material={"kind": "library", "name": "Au"})],
        medium_index=1.333, wavelengths=ls.WavelengthRange(start_nm=450, stop_nm=650, num=201))
    res = ls.solve(sim, method="mie")
    peak = sim.wavelength_array()[np.argmax(res.c_ext_nm2)]
    assert 515.0 <= peak <= 530.0


# --- Drude-Lorentz fit --------------------------------------------------------


@pytest.mark.parametrize("name", ["Au", "Ag"])
def test_fit_reproduces_johnson_christy(name):
    """Stated accuracy over 450-900 nm (JC data, automatic pole count):
    |d eps|/|eps| RMS < 2 %, worst < 5 %; Im(eps) error (rms_imag_error) RMS < 3 %, worst < 8 %."""
    mat = {"kind": "library", "name": name, "dataset": "Johnson-Christy"}
    model, info = fit_drude_lorentz(mat, BAND, return_info=True)
    assert isinstance(model, ls.DrudeLorentz) and isinstance(info, DrudeLorentzFit)
    assert info.model == model and info.n_poles == len(model.poles) <= 5 and info.drude
    q = fit_quality(model, mat, np.linspace(450, 900, 451))  # also between the fitted samples
    assert q["rms_rel_error"] < 0.02 and q["max_rel_error"] < 0.05
    assert q["rms_imag_error"] < 0.03 and q["max_imag_error"] < 0.08
    assert info.rms_rel_error == pytest.approx(fit_quality(model, mat, info.wavelengths_nm)["rms_rel_error"])
    # physically constrained: passive at every frequency, positive parameters
    assert model.eps_inf >= 1 and model.wp_ev > 0 and model.gamma_ev > 0
    assert all(p.delta_eps > 0 and p.omega0_ev > 0 and p.gamma_ev > 0 for p in model.poles)
    assert np.all(model.eps(np.geomspace(100, 1e5, 400)).imag > 0)
    assert ls.Simulation.model_validate({  # usable in a spec as-is
        "particles": [{"shape": {"kind": "sphere", "radius_nm": 10}, "material": model.model_dump()}],
        "wavelengths": [500.0]}).particles[0].material == model


def test_fit_recovers_known_model_and_fixed_pole_count():
    true = ls.DrudeLorentz(eps_inf=1.5, wp_ev=9.0, gamma_ev=0.07, poles=[
        ls.LorentzPole(delta_eps=1.2, omega0_ev=2.9, gamma_ev=0.6),
        ls.LorentzPole(delta_eps=2.0, omega0_ev=4.0, gamma_ev=1.0)])
    wl = np.linspace(400, 1000, 61)
    fit = fit_drude_lorentz(true, wl, n_poles=2)
    assert fit_quality(fit, true, wl)["rms_rel_error"] < 1e-6
    one = fit_drude_lorentz(true, wl, 1)
    assert len(one.poles) == 1
    # dielectrics get no Drude term; a constant index needs no poles
    const = fit_drude_lorentz({"kind": "constant_index", "n_real": 1.5}, wl)
    assert const.wp_ev == 0 and const.poles == [] and const.eps_inf == pytest.approx(2.25)


def test_fit_with_library_model_matches_mie():
    # 20 nm Au sphere in water: the fitted model reproduces Mie extinction of the JC data within 6 %
    wl = np.linspace(450, 650, 81)
    model = fit_drude_lorentz("Au", np.linspace(450, 900, 91))
    res = {}
    for key, mat in (("data", {"kind": "library", "name": "Au"}), ("fit", model)):
        sim = ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=10.0), material=mat)],
                             medium_index=1.333, wavelengths=wl.tolist())
        res[key] = np.asarray(ls.solve(sim, method="mie").c_ext_nm2)
    assert np.max(np.abs(res["fit"] / res["data"] - 1)) < 0.06
    assert abs(wl[res["fit"].argmax()] - wl[res["data"].argmax()]) <= 3.0


def test_fit_rejects_bad_input():
    with pytest.raises(MaterialRangeError):
        fit_drude_lorentz("Au", [150.0, 600.0])
    with pytest.raises(ValueError):
        fit_drude_lorentz("Au", [])
    with pytest.raises(ValueError):
        fit_drude_lorentz("Au", BAND, n_poles=-1)


# --- network ------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(not NETWORK, reason="set LIGHTSMITH_NETWORK_TESTS=1 to run network tests")
def test_fetch_refractiveindex_network(tmp_path):
    commit = M._manifest()["commit"]
    jc = M.fetch_refractiveindex("main", "Au", "Johnson", cache_dir=tmp_path, ref=commit)
    bundled = load_material("Au")
    assert jc.wavelength_nm == bundled.wavelength_nm and jc.n_imag == bundled.n_imag
    assert "Johnson" in jc.source and "CC0" in jc.source
    # the second call is served from the cache
    cached = list(tmp_path.rglob("Johnson.yml"))
    assert len(cached) == 1
    assert M.fetch_refractiveindex("main", "Au", "Johnson", cache_dir=tmp_path, ref=commit) == jc
    # data path with spaces, found through catalog-nk.yml
    org = M.fetch_refractiveindex("organic", "ProSQ-C16", "Bernhardt-rac-o", cache_dir=tmp_path, ref=commit)
    assert org.range_nm[0] < 400 < org.range_nm[1]
    with pytest.raises(LookupError, match="no entry"):
        M.fetch_refractiveindex("main", "Au", "NoSuchPage", cache_dir=tmp_path, ref=commit)
