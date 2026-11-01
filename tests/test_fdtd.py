"""Native FDTD backend, validated against Mie.

Measured accuracy (RTX 5070, float32; CPU float64 agrees to ~1e-7):
- dielectric n=1.5, r=60 nm in vacuum, 400-800 nm: C_ext error 1.1% at dx=6 nm, 0.8% at 4,
  0.6% at 3, 0.4% at 2 nm (first order: subpixel smoothing acts like n_eff = 1.5 + 0.0005 dx/nm).
- Drude(9.5, 9.0 eV, 0.07 eV), r=40 nm in n=1.33 (Mie peak 534 nm): peak at 532-536 nm for
  dx <= 3 nm; peak height -13% (dx=3), -8% (2.5), -11% (2), -11% (1.6), -5/-6% (1.33). The error is
  excess absorption in interface cells and is not monotonic in dx.
- Au (Johnson-Christy, Drude-Lorentz fit, 1.6% RMS), r=40 nm in water, dx=2 nm: peak 550 nm as Mie,
  height -3.9%.
- layered sphere: 0.6%; near-field |E|^2 vs native Mie: 2.9% RMS; oblique incidence: ~1%.
"""

import numpy as np
import pytest

import omniscatter as osc
from omniscatter.device import _torch_cuda
from omniscatter.materials import LibraryMaterial
from omniscatter.spec import LayeredSphere, NearFieldRequest, Outputs, Shell

needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
DRUDE = osc.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)


def sphere_sim(material, radius, wavelengths, medium=1.0, **fdtd):
    return osc.Simulation(
        particles=[osc.Particle(shape=osc.Sphere(radius_nm=radius), material=material)],
        medium_index=medium, wavelengths=wavelengths, options={"fdtd": fdtd})


def rel(a, b):
    return np.abs(np.asarray(a) / np.asarray(b) - 1)


def test_fdtd_dielectric_sphere_matches_mie_cpu():
    sim = sphere_sim(osc.ConstantIndex(n_real=1.5), 40, [400.0, 550.0, 800.0], spacing_nm=6.0)
    res = osc.solve(sim, "fdtd", "cpu")
    mie = osc.solve(sim, "mie")
    assert res.provenance["framework"] == "numpy" and res.provenance["precision"] == "double"
    assert rel(res.c_ext_nm2, mie.c_ext_nm2).max() < 0.03
    assert rel(res.c_sca_nm2, mie.c_sca_nm2).max() < 0.03
    assert np.abs(res.c_abs_nm2).max() < 1e-3 * max(res.c_ext_nm2)  # lossless: no absorption
    for key in ("grid", "dt_fs", "steps", "runtime_s", "decay_level", "precision", "device"):
        assert key in res.provenance
    assert res.provenance["decay_level"] < 1e-5


def test_fdtd_estimate_does_not_solve():
    sim = sphere_sim(DRUDE, 40, [450.0, 550.0], medium=1.33)
    est = osc.estimate(sim, method="fdtd", device="cpu")
    assert est["backend"] == "omniscatter-fdtd"
    assert est["spacing_nm"] == pytest.approx(2.0)  # 20 cells per radius for a plasmonic particle
    assert est["cells"] == np.prod(est["grid"]) and est["memory_bytes"] > 0
    assert est["steps_max"] >= est["steps_estimate"] > 1000


def test_fdtd_supports_and_options():
    sim = sphere_sim(DRUDE, 40, [500.0])
    assert osc.get_backend("fdtd").supports(sim)[0]
    far = sim.model_copy(update={"outputs": Outputs(far_field={"theta_deg": [0.0], "wavelengths_nm": [500.0]})})
    ok, why = osc.get_backend("fdtd").supports(far)
    assert not ok and "far_field" in why
    bad = sphere_sim(DRUDE, 40, [500.0], bogus=1)
    with pytest.raises(Exception):
        osc.estimate(bad, method="fdtd", device="cpu")


def test_fdtd_lossy_constant_is_exact_at_single_wavelength():
    # A lossy constant eps is modelled as a Drude term matched at the band centre: exact for one wavelength.
    from omniscatter.backends._fdtd.media import to_medium

    for mat in (osc.ConstantIndex(n_real=1.5, n_imag=0.1), osc.ConstantPermittivity(eps_real=-2.0, eps_imag=1.0)):
        warn = []
        m = to_medium(mat, np.array([500.0]), warn)
        assert m.eps(np.array([2 * np.pi / 500.0]))[0] == pytest.approx(complex(mat.eps(np.array([500.0]))[0]))
        assert m.eps_inf >= 1
    warn = []
    to_medium(osc.ConstantIndex(n_real=1.5, n_imag=0.1), np.array([400.0, 800.0]), warn)
    assert warn and "not causal" in warn[0]


@pytest.mark.gpu
@needs_cuda
def test_fdtd_gpu_matches_cpu():
    sim = sphere_sim(DRUDE, 20, [450.0, 520.0, 600.0], medium=1.33, spacing_nm=4.0, run_time_fs=5.0)
    cpu = osc.solve(sim, "fdtd", "cpu")
    gpu = osc.solve(sim, "fdtd", "gpu")
    assert gpu.provenance["framework"] == "torch" and gpu.provenance["precision"] == "single"
    assert gpu.provenance["cuda_graphs"]
    np.testing.assert_allclose(gpu.c_ext_nm2, cpu.c_ext_nm2, rtol=1e-4)
    np.testing.assert_allclose(gpu.c_abs_nm2, cpu.c_abs_nm2, rtol=1e-4)


@pytest.mark.gpu
@needs_cuda
def test_fdtd_drude_sphere_matches_mie():
    """Plasmon peak of a Drude sphere: position within 4 nm, height within 15% at dx = 2 nm (r/20)."""
    wl = osc.WavelengthRange(start_nm=480, stop_nm=600, num=41)
    sim = sphere_sim(DRUDE, 40, wl, medium=1.33)  # default resolution: 20 cells per radius -> 2 nm
    res = osc.solve(sim, "fdtd", "gpu")
    mie = osc.solve(sim, "mie")
    assert res.provenance["spacing_nm"] == pytest.approx(2.0)
    w = np.asarray(res.wavelengths_nm)
    i, j = np.argmax(mie.c_ext_nm2), np.argmax(res.c_ext_nm2)
    assert abs(w[i] - w[j]) <= 4.0
    assert rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]) < 0.15
    assert rel(res.c_sca_nm2, mie.c_sca_nm2).max() < 0.30
    assert min(res.c_abs_nm2) > 0


@pytest.mark.gpu
@needs_cuda
def test_fdtd_layered_sphere_near_field_and_oblique_incidence():
    sim = osc.Simulation(
        particles=[osc.Particle(shape=LayeredSphere(core_radius_nm=35, shells=[
            Shell(thickness_nm=20, material=osc.ConstantIndex(n_real=1.4))]), material=osc.ConstantIndex(n_real=2.0))],
        medium_index=1.33, wavelengths=[450.0, 550.0, 700.0], options={"fdtd": {"spacing_nm": 5.0}},
        illumination=osc.PlaneWave(direction=(1, 0, 1), polarization=(1, 0, -1)),
        outputs=Outputs(near_field=NearFieldRequest(plane="xz", half_width_nm=100, points=21, wavelengths_nm=[550.0])))
    res = osc.solve(sim, "fdtd", "gpu")
    mie = osc.solve(sim, "omniscatter-mie", "cpu")
    assert rel(res.c_ext_nm2, mie.c_ext_nm2).max() < 0.03
    assert np.abs(res.c_abs_nm2).max() < 2e-3 * max(res.c_ext_nm2)
    a, b = np.asarray(res.near_field[0].enhancement), np.asarray(mie.near_field[0].enhancement)
    ok = np.isfinite(b)
    assert np.sqrt(np.mean((a[ok] / b[ok] - 1) ** 2)) < 0.06


# --- slow, tighter checks -----------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_dielectric_sphere_fine_grid():
    sim = sphere_sim(osc.ConstantIndex(n_real=1.5), 60, osc.WavelengthRange(start_nm=400, stop_nm=800, num=9),
                     spacing_nm=2.0)
    res = osc.solve(sim, "fdtd", "gpu")
    assert rel(res.c_ext_nm2, osc.solve(sim, "mie").c_ext_nm2).max() < 0.006  # measured 0.40%


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_drude_sphere_fine_grid():
    wl = osc.WavelengthRange(start_nm=440, stop_nm=620, num=19)
    sim = sphere_sim(DRUDE, 40, wl, medium=1.33, spacing_nm=40 / 30)
    res = osc.solve(sim, "fdtd", "gpu")
    mie = osc.solve(sim, "mie")
    i, j = np.argmax(mie.c_ext_nm2), np.argmax(res.c_ext_nm2)
    assert abs(i - j) <= 1  # 10 nm sampling; Mie's 530 and 540 nm values differ by only 2%
    print("peak", res.wavelengths_nm[j], mie.wavelengths_nm[i], "height error", rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]))
    assert rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]) < 0.07


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_gold_library_material_via_drude_lorentz_fit():
    wl = osc.WavelengthRange(start_nm=480, stop_nm=620, num=29)
    sim = sphere_sim(LibraryMaterial(name="Au", dataset="Johnson-Christy"), 40, wl, medium=1.33)
    res = osc.solve(sim, "fdtd", "gpu")
    mie = osc.solve(sim, "mie")
    w = np.asarray(res.wavelengths_nm)
    i, j = np.argmax(mie.c_ext_nm2), np.argmax(res.c_ext_nm2)
    assert abs(w[i] - w[j]) <= 10.0
    assert rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]) < 0.2
    assert any("fitted" in m for m in res.warnings)
