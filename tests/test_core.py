import json

import numpy as np
import pytest
from pydantic import ValidationError

import omniscatter as osc
from omniscatter.cli import main as cli_main
from omniscatter.device import _torch_cuda


def sphere_sim(material, radius=40.0, wavelengths=(450.0, 550.0, 650.0), npd=12, medium=1.33):
    return osc.Simulation(
        particles=[osc.Particle(shape=osc.Sphere(radius_nm=radius), material=material)],
        medium_index=medium,
        wavelengths=list(wavelengths),
        options={"dda": {"dipoles_per_diameter": npd}},
    )


DIELECTRIC = osc.ConstantIndex(n_real=1.5, n_imag=0.02)


def test_spec_json_roundtrip_and_schema():
    sim = sphere_sim(osc.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07))
    again = osc.Simulation.from_json(sim.to_json())
    assert again == sim
    schema = osc.json_schema()
    assert "particles" in schema["properties"]


def test_spec_rejects_bad_polarization():
    with pytest.raises(ValidationError):
        osc.PlaneWave(direction=(0, 0, 1), polarization=(0, 0, 1))


def test_mie_rayleigh_limit():
    # Small sphere: C_abs = 4 pi k a^3 Im((e-1)/(e+2)), relative permittivity, k in medium.
    eps = complex(-2.0, 1.0) * 1.33**2
    mat = osc.ConstantPermittivity(eps_real=eps.real, eps_imag=eps.imag)
    a, lam = 2.0, 500.0
    res = osc.solve(sphere_sim(mat, radius=a, wavelengths=[lam]), method="mie")
    k = 2 * np.pi * 1.33 / lam
    er = eps / 1.33**2
    expected = 4 * np.pi * k * a**3 * ((er - 1) / (er + 2)).imag
    assert res.c_abs_nm2[0] == pytest.approx(expected, rel=5e-3)


def test_auto_method_selection():
    assert osc.choose_method(sphere_sim(DIELECTRIC))[0].method == "mie"
    sim = osc.Simulation(
        particles=[osc.Particle(shape=osc.Spheroid(semi_axis_xy_nm=20, semi_axis_z_nm=40), material=DIELECTRIC)],
        wavelengths=[500.0],
    )
    backend, skipped = osc.choose_method(sim)
    assert backend.method == "dda"
    assert any("spheres only" in s for s in skipped)


def test_dda_matches_mie_for_dielectric_sphere():
    c = osc.compare(sphere_sim(DIELECTRIC), ["mie", "dda"], device="cpu")
    for q in ("c_ext_nm2", "c_abs_nm2"):
        assert c["relative_difference"]["dda"][q]["max"] < 0.01


def test_dda_lossless_has_no_absorption():
    res = osc.solve(sphere_sim(osc.ConstantIndex(n_real=1.6)), method="dda", device="cpu")
    assert np.max(np.abs(res.c_abs_nm2)) < 1e-6 * np.max(res.c_ext_nm2)


def test_dda_estimate_does_not_solve():
    est = osc.estimate(sphere_sim(DIELECTRIC), method="dda", device="cpu")
    assert est["dipoles"] > 0 and est["memory_bytes"] > 0 and est["fits"]


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
def test_dda_gpu_matches_cpu():
    sim = sphere_sim(DIELECTRIC)
    cpu = osc.solve(sim, "dda", "cpu")
    gpu = osc.solve(sim, "dda", "gpu")
    assert gpu.provenance["framework"] == "torch"
    np.testing.assert_allclose(gpu.c_ext_nm2, cpu.c_ext_nm2, rtol=1e-3)


def test_cli_json(capsys, tmp_path):
    assert cli_main(["schema"]) == 0
    assert "properties" in json.loads(capsys.readouterr().out)
    spec = tmp_path / "s.json"
    spec.write_text(sphere_sim(DIELECTRIC).to_json())
    assert cli_main(["solve", str(spec), "--method", "mie"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "mie" and len(out["c_ext_nm2"]) == 3
    assert cli_main(["solve", str(tmp_path / "missing.json")]) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "FileNotFoundError"
