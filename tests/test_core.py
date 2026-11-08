import json

import numpy as np
import pytest
from pydantic import ValidationError

import lightsmith as ls
from lightsmith.cli import main as cli_main
from lightsmith.device import _torch_cuda


def sphere_sim(material, radius=40.0, wavelengths=(450.0, 550.0, 650.0), npd=12, medium=1.33):
    return ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material)],
        medium_index=medium,
        wavelengths=list(wavelengths),
        options={"dda": {"dipoles_per_diameter": npd}},
    )


DIELECTRIC = ls.ConstantIndex(n_real=1.5, n_imag=0.02)


def test_spec_json_roundtrip_and_schema():
    sim = sphere_sim(ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07))
    again = ls.Simulation.from_json(sim.to_json())
    assert again == sim
    schema = ls.json_schema()
    assert "particles" in schema["properties"]


def test_spec_rejects_bad_polarization():
    with pytest.raises(ValidationError):
        ls.PlaneWave(direction=(0, 0, 1), polarization=(0, 0, 1))


def test_mie_rayleigh_limit():
    # Small sphere: C_abs = 4 pi k a^3 Im((e-1)/(e+2)), relative permittivity, k in medium.
    eps = complex(-2.0, 1.0) * 1.33**2
    mat = ls.ConstantPermittivity(eps_real=eps.real, eps_imag=eps.imag)
    a, lam = 2.0, 500.0
    res = ls.solve(sphere_sim(mat, radius=a, wavelengths=[lam]), method="mie")
    k = 2 * np.pi * 1.33 / lam
    er = eps / 1.33**2
    expected = 4 * np.pi * k * a**3 * ((er - 1) / (er + 2)).imag
    assert res.c_abs_nm2[0] == pytest.approx(expected, rel=5e-3)


def test_auto_method_selection():
    assert ls.choose_method(sphere_sim(DIELECTRIC))[0].method == "mie"
    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=20, semi_axis_z_nm=40), material=DIELECTRIC)],
        wavelengths=[500.0],
    )
    backend, skipped = ls.choose_method(sim)
    assert backend.method == "bem"  # BEM precedes DDA for homogeneous non-spherical particles
    assert any("spheres only" in s for s in skipped)


def test_dda_matches_mie_for_dielectric_sphere():
    c = ls.compare(sphere_sim(DIELECTRIC), ["mie", "dda"], device="cpu")
    for q in ("c_ext_nm2", "c_abs_nm2"):
        assert c["relative_difference"]["dda"][q]["max"] < 0.01


def test_compare_floors_near_zero_absorption():
    """A lossless sphere's C_abs is ~1e-19 of C_ext; dividing by it turned round-off into huge 'errors'."""
    sim = sphere_sim(ls.ConstantIndex(n_real=1.5))
    c = ls.compare(sim, ["lightsmith-mie", "miepython"], device="cpu")
    assert c["reference"] == "lightsmith-mie"
    assert c["relative_difference"]["miepython"]["c_abs_nm2"]["max"] < 1e-9


def test_compare_reference_follows_the_auto_order(monkeypatch):
    """Without Mie and without a method list, the reference is what `auto` would pick (TMM for a layer
    stack), not the alphabetically first backend (FDTD)."""
    import lightsmith.api as api
    from lightsmith.problems import Layer, LayerStackProblem

    p = LayerStackProblem(superstrate=ls.ConstantIndex(n_real=1.0), substrate=ls.ConstantIndex(n_real=1.52),
                          layers=[Layer(thickness_nm=100, material=ls.ConstantIndex(n_real=1.38))],
                          wavelengths=[500.0, 600.0])
    c = ls.compare(p, ["tmm", "rcwa"], device="cpu")
    assert c["reference"] == "tmm" and max(d["max"] for d in c["relative_difference"]["rcwa"].values()) < 1e-6
    real_solve, ran = api.solve, []
    monkeypatch.setattr(api, "solve", lambda prob, m, dev: ran.append(m) or real_solve(prob, "tmm", "cpu"))
    c = ls.compare(p)  # every applicable backend, solves stubbed (no FDTD run)
    assert ran[0] == c["reference"] == "lightsmith-tmm" and "lightsmith-fdtd" in ran


def test_compare_reports_sparameter_differences():
    """Device results used to come back with an empty relative_difference."""
    from lightsmith.problems import Box, DeviceProblem, Port, Structure

    core, clad = ls.ConstantIndex(n_real=2.85), ls.ConstantIndex(n_real=1.44)
    ports = [Port(name="o1", center_nm=(-1500, 0, 0), size_nm=(0, 4000, 1), direction="+"),
             Port(name="o2", center_nm=(1500, 0, 0), size_nm=(0, 4000, 1), direction="-")]
    p = DeviceProblem(dimension=2, structures=[Structure(geometry=Box(size_nm=(5000, 500, 1)), material=core)],
                      background=clad, domain_size_nm=(3000, 4000, 0), wavelengths=[1550], ports=ports,
                      options={"eme": {"num_modes": 10}})
    c = ls.compare(p, ["eme", "bpm"], device="cpu")
    d = c["relative_difference"]["bpm"]
    assert set(d) == {"power", "amplitude"} and d["power"]["max"] < 1e-3
    assert "not_compared" not in c


def test_auto_skips_a_backend_over_its_size_limit():
    """BEM accepts the spheroid but it exceeds max_unknowns: auto must fall through to DDA, not fail in solve."""
    sim = ls.Simulation(particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=40, semi_axis_z_nm=120),
                                               material=ls.LibraryMaterial(name="Au"))],
                        wavelengths=[700.0], options={"bem": {"max_unknowns": 500}})
    b, skipped = ls.choose_method(sim)
    assert b.method == "dda"
    assert any(s.startswith("lightsmith-bem: too large") for s in skipped)
    with pytest.raises(ValueError, match="max_unknowns"):
        ls.solve(sim, "bem", device="cpu")


def test_dda_lossless_has_no_absorption():
    res = ls.solve(sphere_sim(ls.ConstantIndex(n_real=1.6)), method="dda", device="cpu")
    assert np.max(np.abs(res.c_abs_nm2)) < 1e-6 * np.max(res.c_ext_nm2)


def test_dda_estimate_does_not_solve():
    est = ls.estimate(sphere_sim(DIELECTRIC), method="dda", device="cpu")
    assert est["dipoles"] > 0 and est["memory_bytes"] > 0 and est["fits"]


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
def test_dda_gpu_matches_cpu():
    sim = sphere_sim(DIELECTRIC)
    cpu = ls.solve(sim, "dda", "cpu")
    gpu = ls.solve(sim, "dda", "gpu")
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


# --- backend selection, provenance and messages ---------------------------------------


def _crossing_device(**fdfd):
    """Ports on two axes: EME and BPM refuse it, so `auto` reaches FDFD and then FDTD."""
    from lightsmith.problems import DeviceProblem, Port

    si, sio2 = ls.ConstantIndex(n_real=3.48), ls.ConstantIndex(n_real=1.44)
    from lightsmith.problems import Box, Structure

    return DeviceProblem(
        dimension=2, background=sio2, domain_size_nm=(4000, 4000, 0), wavelengths=[1550.0],
        structures=[Structure(geometry=Box(center_nm=(0, 0, 0), size_nm=(5000, 500, 1)), material=si),
                    Structure(geometry=Box(center_nm=(0, 0, 0), size_nm=(500, 5000, 1)), material=si)],
        ports=[Port(name="o1", center_nm=(-1500, 0, 0), size_nm=(0, 2000, 0), direction="+"),
               Port(name="o2", center_nm=(0, 1500, 0), size_nm=(2000, 0, 0), direction="-")],
        options={"fdfd": fdfd} if fdfd else {})


def test_auto_passes_over_a_backend_that_is_too_large():
    backend, skipped = ls.choose_method(_crossing_device(max_cells=100))
    assert backend.method == "fdtd"
    assert any("lightsmith-fdfd" in s and "too large" in s for s in skipped)
    est = ls.estimate(_crossing_device(max_cells=100), "auto", "cpu")
    assert est["method"] == "fdtd" and any("too large" in s for s in est["auto_skipped"])
    assert ls.estimate(_crossing_device(max_cells=100), "fdfd", "cpu")["fits"] is False  # an answer, not an error
    with pytest.raises(ValueError, match="max_cells"):
        ls.solve(_crossing_device(max_cells=100), "fdfd", "cpu")  # solving it is still refused


def test_every_result_records_versions_and_why_auto_passed_over_backends():
    import numpy

    res = ls.solve(ls.Simulation(particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=12),
                                                        material=ls.ConstantIndex(n_real=1.5))],
                                 wavelengths=[600.0], options={"bem": {"mesh_size_nm": 5}}), device="cpu")
    assert res.provenance["versions"]["lightsmith"] == ls.__version__
    assert res.provenance["versions"]["numpy"] == numpy.__version__
    assert res.method == "bem" and any("Mie handles spheres only" in s for s in res.provenance["auto_skipped"])
    forced = ls.solve(ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=20),
                                                           material=ls.ConstantIndex(n_real=1.5))],
                                    wavelengths=[600.0]), "mie", "cpu")
    assert "auto_skipped" not in forced.provenance and "versions" in forced.provenance


@pytest.mark.parametrize("backend", ["miepython", "lightsmith-mie"])
def test_mie_refusal_suggests_backends_that_accept_the_shape(backend):
    sim = ls.Simulation(particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=20),
                                               material=ls.ConstantIndex(n_real=1.5))], wavelengths=[600.0])
    ok, why = ls.get_backend(backend).supports(sim)
    assert not ok
    import re

    named = re.findall(r"'(bem|dda|tmatrix|fdtd)'", why)
    assert named and all(ls.get_backend(m).supports(sim)[0] for m in named), why


def test_cli_usage_errors_are_json_with_exit_status_1(capsys):
    for argv in (["no-such-command"], ["solve"], ["estimate", "x.json", "--no-such-flag"]):
        with pytest.raises(SystemExit) as e:
            cli_main(argv)
        assert e.value.code == 1
        err = json.loads(capsys.readouterr().err)
        assert err["error"] == "UsageError" and "--help" in err["message"]
