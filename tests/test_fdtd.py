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
- any incidence direction (tables of the exact discrete plane wave; n=1.5, dx=6 nm): sphere r=60 nm within
  0.08% of Mie for (0,0,1), (1,0,1), (1,1,1), (1,2,3), spread between directions 4e-5 (0.35% with the
  former interpolated 1D feed); spheroid 40x80 nm within 0.5-1.1% of DDA (80 dipoles per diameter) for
  axial and off-axis directions, 0.3-0.5% at dx=4 nm.
"""

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.device import _torch_cuda
from lightsmith.materials import LibraryMaterial
from lightsmith.spec import FarFieldRequest, GaussianBeam, LayeredSphere, NearFieldRequest, Outputs, ResonanceRequest, Shell

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
    assert res.provenance["symmetry"]["reduction"] == 4  # x antisymmetric, y symmetric: a quarter of the grid
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


def test_fdtd_cpu_divergence_raises_before_float64_overflow(monkeypatch):
    # A lossy constant index (stiff Drude term) at under 3 cells per wavelength diverges from ~170 steps.
    # In float64 the fields stay finite well past 200 fs; the run used to return C_ext ~ 1e129.
    # time_step now avoids this instability, so force the plain eps_inf Courant step to keep testing the check.
    from lightsmith.backends._fdtd import scattering

    monkeypatch.setattr(scattering, "time_step", lambda dx, courant, media, grid=None: courant * dx / np.sqrt(3))
    sim = sphere_sim(ls.ConstantIndex(n_real=1.2, n_imag=0.2), 191, [500.0, 600.0, 700.0],
                     spacing_nm=191.0, pml_cells=4, run_time_fs=200.0)
    with pytest.raises(FloatingPointError, match="diverged.*finer mesh.*courant"):
        ls.solve(sim, "fdtd", "cpu")


def test_fdtd_time_step_is_stable_for_dispersive_media_on_coarse_grids():
    # Drude/Lorentz poles lower eps below eps_inf at high grid frequencies, so the eps_inf Courant step
    # diverged for the gold fit at dx >= 20 nm and for lossy constants at dx ~ wavelength/2.
    from lightsmith.backends._fdtd.media import to_medium
    from lightsmith.backends._fdtd.setup import _amplification, time_step

    gold = to_medium(LibraryMaterial(name="Au", dataset="Johnson-Christy"), np.linspace(400, 800, 9), [])
    lossy = to_medium(ls.ConstantIndex(n_real=1.2, n_imag=0.05), np.linspace(500, 700, 5), [])
    for m, coarse, fine in ((gold, 40.0, 5.0), (lossy, 382.0, 95.0)):
        for dx in (coarse, fine):
            dt = time_step(dx, 0.95, [m])
            assert _amplification(m, dt, np.linspace(0, 12 / dx**2, 401)) <= 1 + 1e-6
        assert time_step(coarse, 0.95, [m]) < 0.95 * coarse / np.sqrt(3)
        assert time_step(fine, 0.95, [m]) == 0.95 * fine / np.sqrt(3)  # unchanged where already stable

    wl = {"start_nm": 400.0, "stop_nm": 800.0, "num": 9}  # the 400-800 nm fit has a 5.3 eV, nearly lossless pole
    sim = sphere_sim(LibraryMaterial(name="Au", dataset="Johnson-Christy"), 100, wl, spacing_nm=40.0, run_time_fs=40.0)
    res = ls.solve(sim, "fdtd", "cpu")  # diverged after 366 steps before the dispersive limit
    assert np.all(np.isfinite(res.c_ext_nm2)) and min(res.c_ext_nm2) > 0


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
    """Plasmon peak of a Drude sphere at dx = 2 nm (r/20). Vertex-tensor metal interfaces, measured: peak at
    534 nm as Mie, height 2.6 %, C_ext within 4.3 % and C_sca within 8.2 % over 480-600 nm (diagonal smoothing:
    height 5-13 %, C_ext rippling up to 25 %)."""
    wl = ls.WavelengthRange(start_nm=480, stop_nm=600, num=41)
    sim = sphere_sim(DRUDE, 40, wl, medium=1.33)  # default resolution: 20 cells per radius -> 2 nm
    res = ls.solve(sim, "fdtd", "gpu")
    mie = ls.solve(sim, "mie")
    assert res.provenance["spacing_nm"] == pytest.approx(2.0)
    w = np.asarray(res.wavelengths_nm)
    i, j = np.argmax(mie.c_ext_nm2), np.argmax(res.c_ext_nm2)
    assert res.provenance["tensor_vertices"] > 0
    assert abs(w[i] - w[j]) <= 2.0
    assert rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]) < 0.05
    assert rel(res.c_ext_nm2, mie.c_ext_nm2).max() < 0.07
    assert rel(res.c_sca_nm2, mie.c_sca_nm2).max() < 0.12
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


def test_fdtd_oblique_tfsf_does_not_leak_cpu():
    """Off-axis incidence is fed from tables of the grid's exact discrete plane wave (layered.py, here with a
    uniform background), so a particle of the medium itself scatters nothing (measured ~1e-19 nm², where the
    interpolated 1D feed leaked ~1e-3 of the field), and the near field is the plane wave (measured 3e-4,
    the linear interpolation of the output grid)."""
    k, e = np.array([1.0, 2.0, 3.0]), np.array([3.0, 0.0, -1.0])
    sim = sphere_sim(ls.ConstantIndex(n_real=1.33), 10, [400.0, 800.0], medium=1.33, spacing_nm=4.0,
                     precision="double", pml_cells=6, padding_nm=8).model_copy(update={
                         "illumination": plane_wave(k, e),
                         "outputs": Outputs(near_field=NearFieldRequest(plane="xz", half_width_nm=60, points=9,
                                                                        wavelengths_nm=[600.0]),
                                            far_field=FarFieldRequest(theta_deg=[0, 90], wavelengths_nm=[600.0]))})
    res = ls.solve(sim, "fdtd", "cpu")
    assert res.provenance["tfsf_table_stride"] >= 1
    assert np.abs(res.c_sca_nm2).max() < 1e-12 and np.abs(res.c_abs_nm2).max() < 1e-6
    assert np.abs(res.far_field[0].dcsca_domega_nm2).max() < 1e-15
    nf = res.near_field[0]
    E = np.asarray(nf.e_real) + 1j * np.asarray(nf.e_imag)
    X, Z = np.meshgrid(nf.axis1_nm, nf.axis1_nm, indexing="ij")
    kh, eh = k / np.linalg.norm(k), np.asarray(sim.illumination.unit_vectors()[1])
    ana = eh[:, None, None] * np.exp(1j * 1.33 * 2 * np.pi / 600 * (kh[0] * X + kh[2] * Z))
    assert np.abs(E - ana).max() < 1e-3


@pytest.mark.gpu
@needs_cuda
def test_fdtd_oblique_tfsf_sphere_is_direction_independent():
    """Axial and off-axis incidence (tables of the exact discrete plane wave) on a sphere: each direction vs
    Mie, and the spread between directions, which isolates the source from the common grid error."""
    wl = [400.0, 500.0, 650.0, 800.0]
    ref = ls.solve(sphere_sim(ls.ConstantIndex(n_real=1.5), 60, wl), "mie")
    ext = []
    for k, e in (((0, 0, 1), (1, 0, 0)), ((1, 1, 1), (1, -1, 0)), ((1, 2, 3), (3, 0, -1))):
        sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 60, wl, spacing_nm=6.0).model_copy(
            update={"illumination": plane_wave(k, e)})
        res = ls.solve(sim, "fdtd", "gpu")
        assert rel(res.c_ext_nm2, ref.c_ext_nm2).max() < 3e-3  # measured 0.08%
        assert rel(res.c_sca_nm2, ref.c_sca_nm2).max() < 3e-3
        assert np.abs(res.c_abs_nm2).max() < 1e-4 * max(res.c_ext_nm2)  # measured 1e-5
        ext.append(res.c_ext_nm2)
    ext = np.asarray(ext)
    assert (ext.max(axis=0) / ext.min(axis=0) - 1).max() < 5e-4  # measured 4e-5 (interpolated 1D feed: 0.35%)


@pytest.mark.gpu
@needs_cuda
def test_fdtd_small_sphere_matches_mie():
    """A 20 nm sphere, whose C_ext is ~1e-5 of the incident flux through the TFSF box. Measured 0.04-0.08%
    for both directions; the former 1D-fed source gave C_abs = -0.29 C_ext at 800 nm, because its incident
    flux through the box did not cancel (~1e-5 of it, depending on the DFT stride)."""
    wl = [400.0, 500.0, 650.0, 800.0]
    ref = ls.solve(sphere_sim(ls.ConstantIndex(n_real=1.5), 20, wl), "mie")
    for k, e in (((0, 0, 1), (1, 0, 0)), ((1, 2, 3), (3, 0, -1))):
        sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 20, wl, spacing_nm=2.0, precision="double").model_copy(
            update={"illumination": plane_wave(k, e)})
        res = ls.solve(sim, "fdtd", "gpu")
        assert rel(res.c_ext_nm2, ref.c_ext_nm2).max() < 3e-3
        assert np.abs(res.c_abs_nm2).max() < 1e-4 * max(res.c_ext_nm2)


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


def test_fdtd_metal_vertex_tensor_is_symmetric_and_bounded():
    """The vertex tensor of metal interfaces (engine.TensorMesh) is a sum of congruences: its instantaneous
    D -> E map (one step from rest) is symmetric with every Rayleigh quotient between 1/eps_inf,max and 1/eps_min."""
    from lightsmith.backends._fdtd.engine import Engine, PMLSpec, TensorMesh, _Tensor
    from lightsmith.backends._fdtd.media import Medium, Pole
    from lightsmith.backends._fdtd.setup import voxelise
    from lightsmith.device import Device

    media = [Medium(1.77), Medium(9.5, (Pole(0.00208, 0.0, 1.8e-4),))]  # water, Drude metal (wp = 9 eV)
    shape, c, r = (24, 24, 24), np.array([0.3, 0.1, -0.2]), 6.4
    inv_eps, groups, stats, _ = voxelise(lambda p: (np.linalg.norm(p - c, axis=-1) < r).astype(int), media, shape,
                                         np.full(3, -12.0), 1.0, np.full(3, 2), np.full(3, 22), mode="conformal")
    tm = groups[-1]
    assert isinstance(tm, TensorMesh) and stats["tensor_vertices"] > 100
    eng = Engine(Device("numpy", "cpu"), shape=shape, dx=1.0, dt=0.5, single=False, inv_eps=inv_eps, groups=groups,
                 pml=PMLSpec(cells=((0, 0),) * 3), sources=[], monitors=[], dft_stride=1)
    idx = tm.idx

    def apply(x):
        g = _Tensor(eng, tm)
        eng.E_flat[idx] = x
        g.step(eng)
        return np.array(eng.E_flat[idx])

    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, len(idx)))
    mx, my = apply(x), apply(y)
    assert abs(x @ my - y @ mx) < 1e-12 * abs(x @ mx)
    for z in rng.standard_normal((5, len(idx))):
        q = z @ apply(z) / (z @ z)
        assert 1 / 9.5 <= q <= 1 / 1.77


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
    """Dipole and quadrupole plasmons from the ring-down against the exact complex poles of Mie's a_1 and a_2
    (533.8 nm, Q 8.74; 487.5 nm, Q 35.1). Measured: 535.6 nm, Q 8.2; 488.4 nm, Q 31. The broad dipole is the
    harder fit: with only 5 modes allowed it lands at 537.8 nm."""
    sim = sphere_sim(DRUDE, 40, ls.WavelengthRange(start_nm=440, stop_nm=600, num=9), medium=1.33)
    sim = sim.model_copy(update={"outputs": Outputs(resonances=ResonanceRequest(max_modes=10))})
    res = ls.solve(sim, "fdtd", "gpu")
    dipole = min(res.resonances, key=lambda m: abs(m.wavelength_nm - 533.8))
    quad = min(res.resonances, key=lambda m: abs(m.wavelength_nm - 487.5))
    assert abs(dipole.wavelength_nm - 533.8) < 3 and 7 < dipole.q < 10 and dipole.amplitude > 0.5
    assert abs(quad.wavelength_nm - 487.5) < 2 and 28 < quad.q < 40
    assert max(m.amplitude for m in res.resonances) == 1.0
    assert all(440 <= m.wavelength_nm <= 600 for m in res.resonances)


@pytest.mark.gpu
@needs_cuda
def test_fdtd_gaussian_beam_focus_and_profile():
    """C_ext of a small sphere at the focus equals Mie (normalised by the focal intensity); off the axis it
    follows |E|^2 of the discretised beam (measured: 0.04 % for both; 0.13 % and 0.1 % with 1D-fed components)."""
    from lightsmith.backends._fdtd.beam import beam_field, plane_wave_components
    from lightsmith.spec import GaussianBeam

    wl = [500.0, 600.0, 700.0]
    base = sphere_sim(ls.ConstantIndex(n_real=1.5), 40, wl, spacing_nm=4.0)
    mie = np.asarray(ls.solve(base, "mie").c_ext_nm2)
    for off in ((0.0, 0.0, 0.0), (333.0, 0.0, 0.0)):
        beam = GaussianBeam(waist_nm=1000, focus_nm=tuple(-v for v in off))
        res = ls.solve(base.model_copy(update={"illumination": beam}), "fdtd", "gpu")
        comps, _ = plane_wave_components(beam, 1.0, 600.0, 500.0)
        expect = [np.sum(np.abs(beam_field(comps, np.array([off]), 2 * np.pi / lam)) ** 2)
                  / np.sum(np.abs(beam_field(comps, np.zeros((1, 3)), 2 * np.pi / lam)) ** 2) for lam in wl]
        np.testing.assert_allclose(np.asarray(res.c_ext_nm2) / mie, expect, rtol=2e-3)
        assert res.provenance["gaussian_beam"]["plane_waves"] > 50
    assert not ls.get_backend("mie").supports(base.model_copy(update={"illumination": beam}))[0]


@pytest.mark.gpu
@needs_cuda
def test_fdtd_gaussian_beam_near_field():
    """A particle of the medium itself in a tight beam: no scattering, and the near field is the discretised
    beam (normalised at the focus along the polarisation; measured 2e-4)."""
    from lightsmith.backends._fdtd.beam import beam_field, plane_wave_components
    from lightsmith.spec import GaussianBeam

    beam = GaussianBeam(waist_nm=400)
    sim = sphere_sim(ls.ConstantIndex(n_real=1.0), 10, [500.0, 700.0], spacing_nm=4.0).model_copy(update={
        "illumination": beam, "outputs": Outputs(near_field=NearFieldRequest(plane="xz", half_width_nm=100, points=7,
                                                                             wavelengths_nm=[600.0]))})
    res = ls.solve(sim, "fdtd", "gpu")
    assert np.abs(res.c_sca_nm2).max() < 1e-8  # float32; measured 2e-10
    nf = res.near_field[0]
    E = np.moveaxis(np.asarray(nf.e_real) + 1j * np.asarray(nf.e_imag), 0, -1)
    X, Z = np.meshgrid(nf.axis1_nm, nf.axis1_nm, indexing="ij")
    pts = np.stack([X, np.zeros_like(X), Z], axis=-1).reshape(-1, 3)
    comps, _ = plane_wave_components(beam, 1.0, 600.0, 400.0)  # as the solve: wl_ref = band centre
    k = 2 * np.pi / 600
    ref = beam_field(comps, pts, k).reshape(E.shape) / beam_field(comps, np.zeros((1, 3)), k)[0, 0]
    assert np.isfinite(E).all() and np.abs(E - ref).max() < 2e-3


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
def test_fdtd_metal_vertex_tensor_is_stable_at_late_times():
    """The vertex tensor of metal interfaces is passive: node-by-node off-diagonal terms diverged for this sphere.
    Measured at dx = 3 nm over 1.5 ps: 6e-22 (double), 9e-14 (single)."""
    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=40), material=DRUDE)], wavelengths=[500.0, 700.0, 900.0],
        options={"fdtd": {"spacing_nm": 3.0, "run_time_fs": 1500, "max_time_fs": 1500, "precision": "double"}})
    res = ls.solve(sim, "fdtd", "gpu")
    assert res.provenance["tensor_vertices"] > 0 and res.provenance["steps"] > 250_000
    assert res.provenance["decay_level"] < 1e-15


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
    assert rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]) < 0.04  # measured 2.2 % (diagonal smoothing: 7 %)


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
    assert abs(w[i] - w[j]) <= 5.0  # measured: same 5 nm sample (550 nm)
    assert rel(res.c_ext_nm2[j], mie.c_ext_nm2[i]) < 0.03  # measured 0.7 %
    assert rel(res.c_ext_nm2, mie.c_ext_nm2).max() < 0.07  # measured 4.2 %
    assert any("fitted" in m for m in res.warnings)


# --- mirror symmetries --------------------------------------------------------------------------------------


def solve_both(sim, device="cpu"):
    """The same run with mirror symmetry (auto) and without (off)."""
    out = []
    for sym in ("auto", "off"):
        s = sim.model_copy(update={"options": {"fdtd": {**sim.options["fdtd"], "symmetry": sym}}})
        out.append(ls.solve(s, "fdtd", device))
    return out


def assert_same_run(a, b, tol):
    """Cross sections, far fields and near fields of two runs agree to tol relative to their largest value."""
    for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        x, y = np.asarray(getattr(a, k)), np.asarray(getattr(b, k))
        assert np.abs(x - y).max() <= tol * np.abs(y).max(), k
    for fa, fb in zip(a.far_field or [], b.far_field or []):
        x, y = np.asarray(fa.dcsca_domega_nm2), np.asarray(fb.dcsca_domega_nm2)
        assert np.abs(x - y).max() < tol * np.abs(y).max()
    for na, nb in zip(a.near_field or [], b.near_field or []):
        x = np.asarray(na.e_real) + 1j * np.asarray(na.e_imag)
        y = np.asarray(nb.e_real) + 1j * np.asarray(nb.e_imag)
        assert np.array_equal(np.isfinite(x), np.isfinite(y))
        assert np.nanmax(np.abs(x - y)) < tol * np.nanmax(np.abs(y))


MIRROR_CASES = {
    # x-polarised along z: x antisymmetric (PEC), y symmetric (PMC); conformal off-diagonal terms cross both planes
    "sphere, 2 planes, near and far field": (
        ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=30), material=ls.ConstantIndex(n_real=1.5))],
                      wavelengths=[400.0, 600.0], options={"fdtd": {"spacing_nm": 5.0}},
                      outputs=Outputs(far_field=FarFieldRequest(theta_deg=[0, 60, 120], phi_deg=[0, 45, 90],
                                                                wavelengths_nm=[500.0]),
                                      near_field=NearFieldRequest(plane="xy", half_width_nm=50, points=7,
                                                                  wavelengths_nm=[500.0]))),
        {"x": "antisymmetric (PEC)", "y": "symmetric (PMC)"}),
    # oblique incidence in the yz plane, p-polarised: one symmetric plane, x
    "spheroid, oblique, 1 plane": (
        ls.Simulation(particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=25, semi_axis_z_nm=40),
                                             material=ls.ConstantIndex(n_real=2.0))],
                      wavelengths=[450.0, 650.0], options={"fdtd": {"spacing_nm": 5.0}},
                      illumination=ls.PlaneWave(direction=(0, 1, 1), polarization=(0, 1, -1)),
                      outputs=Outputs(near_field=NearFieldRequest(plane="xz", half_width_nm=50, points=7,
                                                                  wavelengths_nm=[500.0]))),
        {"x": "symmetric (PMC)"}),
    # a Drude dimer along x lit with E along y: the vertex tensor of the metal interfaces is cut by both planes
    "Drude dimer, 2 planes": (
        ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=DRUDE, position_nm=(-22, 0, 0)),
                                 ls.Particle(shape=ls.Sphere(radius_nm=20), material=DRUDE, position_nm=(22, 0, 0))],
                      wavelengths=[480.0, 560.0], options={"fdtd": {"spacing_nm": 4.0}},
                      illumination=ls.PlaneWave(direction=(0, 0, 1), polarization=(0, 1, 0))),
        {"x": "symmetric (PMC)", "y": "antisymmetric (PEC)"}),
}


@pytest.mark.parametrize("case", list(MIRROR_CASES))
def test_fdtd_mirror_symmetry_matches_full_run_cpu(case):
    """A run on the half (quarter) grid steps exactly as the symmetric field of the whole one: cross sections,
    far and near fields agree with the full run to 1e-6 in double precision (measured 1e-15 to 1e-11 for the
    dielectrics, 3e-11 for C_ext of the dimer and 7e-9 for its 1e-4 times smaller C_sca). Fixed run time, so
    both runs stop at the same step."""
    sim, mirrors = MIRROR_CASES[case]
    sim = sim.model_copy(update={"options": {"fdtd": {**sim.options["fdtd"], "precision": "double",
                                                       "run_time_fs": 6.0, "pml_cells": 6}}})
    a, b = solve_both(sim)
    sym = a.provenance["symmetry"]
    assert sym["mirrors"] == mirrors and sym["reduction"] == 2 ** len(mirrors)
    assert b.provenance["symmetry"]["mirrors"] == {} and "off" in b.provenance["symmetry"]["skipped"]
    for ax, n in enumerate(b.provenance["grid"]):
        assert a.provenance["grid"][ax] == ((n - 1) // 2 + 2 if "xyz"[ax] in mirrors else n)
    assert_same_run(a, b, 1e-6)


def test_fdtd_mirror_symmetry_detection():
    """Mirror planes are used only when the particles and the plane wave are symmetric; estimate reports them."""
    from lightsmith.backends._fdtd import symmetry
    from lightsmith.backends._fdtd.options import FDTDOptions

    auto = FDTDOptions()

    def find(sim):
        return symmetry.find(sim, auto)[0]

    sphere = sphere_sim(ls.ConstantIndex(n_real=1.5), 30, [500.0])
    assert find(sphere) == (-1, 1, 0)
    assert find(sphere.model_copy(update={"illumination": ls.PlaneWave(polarization=(0, 1, 0))})) == (1, -1, 0)
    assert find(sphere.model_copy(update={"illumination": plane_wave((0, 0, 1), (1, 1, 0))})) == (0, 0, 0)
    assert find(sphere.model_copy(update={"illumination": plane_wave((1, 0, 1), (1, 0, -1))})) == (0, 1, 0)
    side = ls.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1))
    assert find(sphere.model_copy(update={"illumination": side})) == (0, 1, -1)
    assert symmetry.find(sphere, FDTDOptions(symmetry="off"))[0] == (0, 0, 0)
    beam = sphere.model_copy(update={"illumination": GaussianBeam(waist_nm=500)})
    assert find(beam) == (0, 0, 0) and "Gaussian" in symmetry.find(beam, auto)[1]["skipped"]
    moved = sphere.model_copy(update={"particles": [sphere.particles[0].model_copy(update={"position_nm": (5, 0, 0)})]})
    assert find(moved) == (-1, 1, 0)  # the planes pass through the particles' centre

    def pair(m1, m2, xs=(-30, 30), shape=ls.Sphere(radius_nm=20)):
        return sphere.model_copy(update={"particles": [
            ls.Particle(shape=shape, material=m, position_nm=(x, 0, 0)) for m, x in zip((m1, m2), xs)]})

    glass = ls.ConstantIndex(n_real=1.5)
    assert find(pair(glass, glass)) == (-1, 1, 0)
    assert find(pair(glass, DRUDE)) == (0, 1, 0)  # different materials: no x mirror
    three = pair(glass, glass)
    three = three.model_copy(update={"particles": [*three.particles, three.particles[0].model_copy(
        update={"position_nm": (60.0, 0, 0)})]})
    assert find(three) == (0, 1, 0)  # x = -30, 30, 60: no x mirror
    # overlapping core-shell partners: the first listed wins the overlap, which breaks the x mirror
    shell = LayeredSphere(core_radius_nm=10, shells=[Shell(thickness_nm=10, material=DRUDE)])
    assert find(pair(glass, glass, (-12, 12), shell)) == (0, 1, 0)
    assert find(pair(glass, glass, (-25, 25), shell)) == (-1, 1, 0)  # apart: symmetric
    est = ls.estimate(sphere, method="fdtd", device="cpu")
    full = ls.estimate(sphere.model_copy(update={"options": {"fdtd": {"symmetry": "off"}}}), method="fdtd",
                       device="cpu")
    assert est["symmetry"]["reduction"] == 4 and 3.6 < full["cells"] / est["cells"] < 4


def _assert_faster(sim, a, b, factor: float, attempts: int = 3) -> None:
    """Wall-clock speed-up of run `a` (symmetry) over `b` (full). The GPU may be shared, and another job in only one of
    the two runs skews their ratio, so a slow pair is measured again; the best of a few must still reach `factor`."""
    best = b.provenance["time_loop_s"] / a.provenance["time_loop_s"]
    for _ in range(attempts - 1):
        if best > factor:
            break
        a, b = solve_both(sim, "gpu")
        best = max(best, b.provenance["time_loop_s"] / a.provenance["time_loop_s"])
    assert best > factor


@pytest.mark.gpu
@needs_cuda
def test_fdtd_mirror_symmetry_speedup_gpu():
    """Two mirror planes on the GPU (float32, CUDA graphs): the quarter grid gives the full run's cross sections
    and accuracy against Mie, faster. Measured, sphere r = 60 nm, dx = 3 nm: 52 x 52 x 101 against 101^3 cells,
    C_ext 5e-6 from the full run and 0.063 % from Mie for both, time loop 1.55x faster. The speed-up grows with
    the grid as the fixed per-step cost (table interpolation on the TFSF faces, kernel launches) is amortised:
    in steady state 1.6x at 1e6 full cells, 2.5x at 5e6, 3.7x at 1.5e7; a whole run at dx = 2 nm 1.84x."""
    wl = [400.0, 500.0, 650.0, 800.0]
    sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 60, wl, spacing_nm=3.0)
    a, b = solve_both(sim, "gpu")
    mie = ls.solve(sim, "mie")
    assert a.provenance["cuda_graphs"] and a.provenance["symmetry"]["reduction"] == 4
    assert a.provenance["steps"] == b.provenance["steps"]
    assert rel(a.c_ext_nm2, b.c_ext_nm2).max() < 3e-5 and rel(a.c_sca_nm2, b.c_sca_nm2).max() < 3e-5
    assert rel(a.c_ext_nm2, mie.c_ext_nm2).max() < 2e-3
    _assert_faster(sim, a, b, 1.3)


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_mirror_symmetry_speedup_fine_grid_gpu():
    """The same sphere at dx = 1.5 nm (173^3 cells, 88 x 88 x 173 with symmetry): measured 2.3x faster for the whole
    run (38.6 s against 88.2 s of time loop, 16390 steps each), C_ext 7e-6 from the full run, 0.04 % from Mie."""
    sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 60, [400.0, 500.0, 650.0, 800.0], spacing_nm=1.5)
    a, b = solve_both(sim, "gpu")
    assert rel(a.c_ext_nm2, b.c_ext_nm2).max() < 3e-5
    assert rel(a.c_ext_nm2, ls.solve(sim, "mie").c_ext_nm2).max() < 2e-3
    _assert_faster(sim, a, b, 1.8)
