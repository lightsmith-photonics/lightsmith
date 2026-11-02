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
- off-axis incidence (n=1.5, dx=6 nm): sphere r=60 nm within 1.0-1.2% of Mie for (0,0,1), (1,0,1),
  (1,1,1), (1,2,3), spread between directions 0.35%; spheroid 40x80 nm within 0.5-1.1% of DDA
  (80 dipoles per diameter) for axial and off-axis directions, 0.3-0.5% at dx=4 nm.
"""

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.device import _torch_cuda
from lightsmith.materials import LibraryMaterial
from lightsmith.spec import FarFieldRequest, LayeredSphere, NearFieldRequest, Outputs, ResonanceRequest, Shell

needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
DRUDE = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)


def sphere_sim(material, radius, wavelengths, medium=1.0, **fdtd):
    return ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material)],
        medium_index=medium, wavelengths=wavelengths, options={"fdtd": fdtd})


def rel(a, b):
    return np.abs(np.asarray(a) / np.asarray(b) - 1)


def test_fdtd_dielectric_sphere_matches_mie_cpu():
    sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 40, [400.0, 550.0, 800.0], spacing_nm=6.0)
    res = ls.solve(sim, "fdtd", "cpu")
    mie = ls.solve(sim, "mie")
    assert res.provenance["framework"] == "numpy" and res.provenance["precision"] == "double"
    assert rel(res.c_ext_nm2, mie.c_ext_nm2).max() < 0.03
    assert rel(res.c_sca_nm2, mie.c_sca_nm2).max() < 0.03
    assert np.abs(res.c_abs_nm2).max() < 1e-3 * max(res.c_ext_nm2)  # lossless: no absorption
    for key in ("grid", "dt_fs", "steps", "runtime_s", "decay_level", "precision", "device"):
        assert key in res.provenance
    assert res.provenance["decay_level"] < 1e-5


def test_fdtd_estimate_does_not_solve():
    sim = sphere_sim(DRUDE, 40, [450.0, 550.0], medium=1.33)
    est = ls.estimate(sim, method="fdtd", device="cpu")
    assert est["backend"] == "lightsmith-fdtd"
    assert est["spacing_nm"] == pytest.approx(2.0)  # 20 cells per radius for a plasmonic particle
    assert est["cells"] == np.prod(est["grid"]) and est["memory_bytes"] > 0
    assert est["steps_max"] >= est["steps_estimate"] > 1000


def test_fdtd_supports_and_options():
    sim = sphere_sim(DRUDE, 40, [500.0])
    assert ls.get_backend("fdtd").supports(sim)[0]
    far = sim.model_copy(update={"outputs": Outputs(far_field={"theta_deg": [0.0], "wavelengths_nm": [500.0]})})
    assert ls.get_backend("fdtd").supports(far)[0]
    ok, why = ls.get_backend("fdtd").supports(sim.model_copy(update={"orientation_average": True}))
    assert not ok and "orientation_average" in why
    bad = sphere_sim(DRUDE, 40, [500.0], bogus=1)
    with pytest.raises(Exception):
        ls.estimate(bad, method="fdtd", device="cpu")


def test_fdtd_lossy_constant_is_exact_at_single_wavelength():
    # A lossy constant eps is modelled as a Drude term matched at the band centre: exact for one wavelength.
    from lightsmith.backends._fdtd.media import to_medium

    for mat in (ls.ConstantIndex(n_real=1.5, n_imag=0.1), ls.ConstantPermittivity(eps_real=-2.0, eps_imag=1.0)):
        warn = []
        m = to_medium(mat, np.array([500.0]), warn)
        assert m.eps(np.array([2 * np.pi / 500.0]))[0] == pytest.approx(complex(mat.eps(np.array([500.0]))[0]))
        assert m.eps_inf >= 1
    warn = []
    to_medium(ls.ConstantIndex(n_real=1.5, n_imag=0.1), np.array([400.0, 800.0]), warn)
    assert warn and "not causal" in warn[0]


@pytest.mark.gpu
@needs_cuda
def test_fdtd_gpu_matches_cpu():
    sim = sphere_sim(DRUDE, 20, [450.0, 520.0, 600.0], medium=1.33, spacing_nm=4.0, run_time_fs=5.0)
    cpu = ls.solve(sim, "fdtd", "cpu")
    gpu = ls.solve(sim, "fdtd", "gpu")
    assert gpu.provenance["framework"] == "torch" and gpu.provenance["precision"] == "single"
    assert gpu.provenance["cuda_graphs"]
    np.testing.assert_allclose(gpu.c_ext_nm2, cpu.c_ext_nm2, rtol=1e-4)
    np.testing.assert_allclose(gpu.c_abs_nm2, cpu.c_abs_nm2, rtol=1e-4)


@pytest.mark.gpu
@needs_cuda
def test_fdtd_drude_sphere_matches_mie():
    """Plasmon peak of a Drude sphere: position within 4 nm, height within 15% at dx = 2 nm (r/20)."""
    wl = ls.WavelengthRange(start_nm=480, stop_nm=600, num=41)
    sim = sphere_sim(DRUDE, 40, wl, medium=1.33)  # default resolution: 20 cells per radius -> 2 nm
    res = ls.solve(sim, "fdtd", "gpu")
    mie = ls.solve(sim, "mie")
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
    sim = ls.Simulation(
        particles=[ls.Particle(shape=LayeredSphere(core_radius_nm=35, shells=[
            Shell(thickness_nm=20, material=ls.ConstantIndex(n_real=1.4))]), material=ls.ConstantIndex(n_real=2.0))],
        medium_index=1.33, wavelengths=[450.0, 550.0, 700.0], options={"fdtd": {"spacing_nm": 5.0}},
        illumination=ls.PlaneWave(direction=(1, 0, 1), polarization=(1, 0, -1)),
        outputs=Outputs(near_field=NearFieldRequest(plane="xz", half_width_nm=100, points=21, wavelengths_nm=[550.0])))
    res = ls.solve(sim, "fdtd", "gpu")
    mie = ls.solve(sim, "lightsmith-mie", "cpu")
    assert rel(res.c_ext_nm2, mie.c_ext_nm2).max() < 0.03
    assert np.abs(res.c_abs_nm2).max() < 2e-3 * max(res.c_ext_nm2)
    a, b = np.asarray(res.near_field[0].enhancement), np.asarray(mie.near_field[0].enhancement)
    ok = np.isfinite(b)
    assert np.sqrt(np.mean((a[ok] / b[ok] - 1) ** 2)) < 0.06


def plane_wave(direction, polarization):
    k, e = np.asarray(direction, float), np.asarray(polarization, float)
    return ls.PlaneWave(direction=tuple(k), polarization=tuple(e - (e @ k) / (k @ k) * k))


@pytest.mark.gpu
@needs_cuda
def test_fdtd_oblique_tfsf_sphere_is_direction_independent():
    """Off-axis incidence (interpolated TFSF feed) on a sphere: each direction vs Mie, and the spread
    between directions, which isolates the oblique source from the common grid error."""
    wl = [400.0, 500.0, 650.0, 800.0]
    ref = ls.solve(sphere_sim(ls.ConstantIndex(n_real=1.5), 60, wl), "mie")
    ext = []
    for k, e in (((0, 0, 1), (1, 0, 0)), ((1, 1, 1), (1, -1, 0)), ((1, 2, 3), (3, 0, -1))):
        sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 60, wl, spacing_nm=6.0).model_copy(
            update={"illumination": plane_wave(k, e)})
        res = ls.solve(sim, "fdtd", "gpu")
        assert rel(res.c_ext_nm2, ref.c_ext_nm2).max() < 0.02  # measured 1.0-1.2% (subpixel error at dx=6)
        assert rel(res.c_sca_nm2, ref.c_sca_nm2).max() < 0.02
        assert np.abs(res.c_abs_nm2).max() < 1e-3 * max(res.c_ext_nm2)
        ext.append(res.c_ext_nm2)
    ext = np.asarray(ext)
    assert (ext.max(axis=0) / ext.min(axis=0) - 1).max() < 0.006  # measured 0.35%


@pytest.mark.gpu
@needs_cuda
def test_fdtd_oblique_tfsf_spheroid_matches_dda():
    """A prolate spheroid, whose C_ext changes 2x with direction, at axial and off-axis incidence vs DDA."""
    wl = [400.0, 500.0, 650.0, 800.0]
    for k, e in (((1, 0, 0), (0, 0, 1)), ((1, 0, 1), (1, 0, -1)), ((1, 2, 3), (3, 0, -1))):
        sim = ls.Simulation(
            particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=40, semi_axis_z_nm=80),
                                   material=ls.ConstantIndex(n_real=1.5))],
            wavelengths=wl, illumination=plane_wave(k, e),
            options={"fdtd": {"spacing_nm": 6.0}, "dda": {"dipoles_per_diameter": 80}})
        res, ref = ls.solve(sim, "fdtd", "gpu"), ls.solve(sim, "dda", "gpu")
        assert rel(res.c_ext_nm2, ref.c_ext_nm2).max() < 0.02  # measured 0.5-1.1% (0.5% at dx=4 nm)


@pytest.mark.gpu
@needs_cuda
def test_fdtd_conformal_tensor_smoothing_beats_diagonal():
    """Off-diagonal tensor terms remove the first-order interface error (1.1% -> 0.04% at dx = 6 nm)."""
    wl = ls.WavelengthRange(start_nm=400, stop_nm=800, num=5)
    mie = ls.solve(sphere_sim(ls.ConstantIndex(n_real=1.5), 60, wl), "mie").c_ext_nm2
    err = {}
    for mode in ("staircase", "subpixel", "conformal"):
        res = ls.solve(sphere_sim(ls.ConstantIndex(n_real=1.5), 60, wl, spacing_nm=6.0, mesh_refinement=mode),
                       "fdtd", "gpu")
        assert res.provenance["mesh_refinement"] == mode
        err[mode] = rel(res.c_ext_nm2, mie).max()
    assert res.provenance["offdiagonal_cells"] > 0
    # measured: conformal 0.04%, subpixel 1.1%, staircase 1.0% (staircase errors swing with grid alignment)
    assert err["conformal"] < 0.002 and err["subpixel"] > 0.008 and err["staircase"] > 0.005


@pytest.mark.gpu
@needs_cuda
def test_fdtd_far_field_matches_mie():
    wl = [500.0, 700.0]
    theta = list(np.linspace(0, 180, 37))
    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=60), material=ls.ConstantIndex(n_real=1.5))],
        wavelengths=wl, options={"fdtd": {"spacing_nm": 4.0}},
        outputs=Outputs(far_field=FarFieldRequest(theta_deg=theta, phi_deg=[0.0, 90.0], wavelengths_nm=wl)))
    res = ls.solve(sim, "fdtd", "gpu")
    mie = ls.solve(sim, "lightsmith-mie")
    t = np.radians(theta)
    for i in range(len(wl)):
        a = np.asarray(res.far_field[i].dcsca_domega_nm2)
        b = np.asarray(mie.far_field[i].dcsca_domega_nm2)
        assert np.median(rel(a, b)) < 2e-3 and rel(a, b).max() < 1e-2  # measured 0.03% and 0.4%
        # the phi-averaged pattern integrates to C_sca (exact for the cos^2 phi dependence of a sphere)
        c = 2 * np.pi * np.trapezoid(0.5 * (a[:, 0] + a[:, 1]) * np.sin(t), t)
        assert c == pytest.approx(res.c_sca_nm2[i], rel=5e-3)


@pytest.mark.gpu
@needs_cuda
def test_fdtd_drude_sphere_resonance():
    """The dipole plasmon from the ring-down matches Mie's extinction peak (Q = peak / FWHM = 8.8)."""
    sim = sphere_sim(DRUDE, 40, ls.WavelengthRange(start_nm=440, stop_nm=600, num=9), medium=1.33)
    sim = sim.model_copy(update={"outputs": Outputs(resonances=ResonanceRequest(max_modes=5))})
    res = ls.solve(sim, "fdtd", "gpu")
    dipole = min(res.resonances, key=lambda m: abs(m.wavelength_nm - 534.0))
    assert abs(dipole.wavelength_nm - 534.0) < 3  # measured 534.7 nm
    assert 6 < dipole.q < 10 and dipole.amplitude > 0.5  # measured Q 7.8
    assert res.resonances[0].amplitude == 1.0
    assert all(440 <= m.wavelength_nm <= 600 for m in res.resonances)


# --- slow, tighter checks -----------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_conformal_is_stable_at_late_times():
    """The symmetrised off-diagonal coupling keeps the update stable; the one-sided average grew without bound."""
    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=80), material=ls.ConstantIndex(n_real=3.5))],
        wavelengths=[500.0, 700.0, 900.0],
        options={"fdtd": {"spacing_nm": 6.0, "run_time_fs": 3000, "max_time_fs": 3000, "precision": "double"}})
    res = ls.solve(sim, "fdtd", "gpu")
    assert res.provenance["mesh_refinement"] == "conformal" and res.provenance["steps"] > 250_000
    assert res.provenance["decay_level"] < 1e-12  # measured 2e-19


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_dielectric_sphere_fine_grid():
    sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 60, ls.WavelengthRange(start_nm=400, stop_nm=800, num=9),
                     spacing_nm=2.0)
    res = ls.solve(sim, "fdtd", "gpu")
    assert rel(res.c_ext_nm2, ls.solve(sim, "mie").c_ext_nm2).max() < 0.002  # conformal; diagonal subpixel: 0.40%


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_drude_sphere_fine_grid():
    wl = ls.WavelengthRange(start_nm=440, stop_nm=620, num=19)
    sim = sphere_sim(DRUDE, 40, wl, medium=1.33, spacing_nm=40 / 30)
    res = ls.solve(sim, "fdtd", "gpu")
    mie = ls.solve(sim, "mie")
    i, j = np.argmax(mie.c_ext_nm2), np.argmax(res.c_ext_nm2)
    assert abs(i - j) <= 1  # 10 nm sampling; Mie's 530 and 540 nm values differ by only 2%
    print("peak", res.wavelengths_nm[j], mie.wavelengths_nm[i], "height error", rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]))
    assert rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]) < 0.07


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_gold_library_material_via_drude_lorentz_fit():
    wl = ls.WavelengthRange(start_nm=480, stop_nm=620, num=29)
    sim = sphere_sim(LibraryMaterial(name="Au", dataset="Johnson-Christy"), 40, wl, medium=1.33)
    res = ls.solve(sim, "fdtd", "gpu")
    mie = ls.solve(sim, "mie")
    w = np.asarray(res.wavelengths_nm)
    i, j = np.argmax(mie.c_ext_nm2), np.argmax(res.c_ext_nm2)
    assert abs(w[i] - w[j]) <= 10.0
    assert rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]) < 0.2
    assert any("fitted" in m for m in res.warnings)
