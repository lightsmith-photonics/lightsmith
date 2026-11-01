import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends.mie_native import mie_cross_sections
from lightsmith.device import _torch_cuda
from lightsmith.spec import FarFieldRequest, LayeredSphere, NearFieldRequest, Outputs, Shell

DIELECTRIC = ls.ConstantIndex(n_real=1.5, n_imag=0.02)
DRUDE = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)
SILICA = ls.ConstantIndex(n_real=1.45)
NATIVE = "lightsmith-mie"


def sim_of(shape, material=DRUDE, wavelengths=(520.0,), medium=1.33, **kw):
    return ls.Simulation(particles=[ls.Particle(shape=shape, material=material)], medium_index=medium,
                          wavelengths=list(wavelengths), **kw)


def core_shell(core=DRUDE, shell=SILICA, rc=40.0, t=10.0):
    return LayeredSphere(core_radius_nm=rc, shells=[Shell(thickness_nm=t, material=shell)])


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.max(np.abs(a - b) / np.abs(b))


# --- accuracy ---------------------------------------------------------------


@pytest.mark.parametrize("material", [DIELECTRIC, DRUDE], ids=["dielectric", "drude"])
@pytest.mark.parametrize("radius", [2.0, 25.0, 60.0, 200.0, 1000.0])
def test_matches_miepython(material, radius):
    # 450-900 nm avoids the Drude epsilon ~ 0 point (~425 nm), where miepython's
    # small-particle branch itself is only good to ~1e-5 (see test_hard_cases_against_mpmath).
    sim = sim_of(ls.Sphere(radius_nm=radius), material, np.linspace(450, 900, 19))
    ref = ls.solve(sim, "miepython")
    res = ls.solve(sim, NATIVE, "cpu")
    assert rel(res.c_ext_nm2, ref.c_ext_nm2) < 1e-6
    assert rel(res.c_sca_nm2, ref.c_sca_nm2) < 1e-6
    assert np.max(np.abs(np.subtract(res.c_abs_nm2, ref.c_abs_nm2)) / np.asarray(ref.c_ext_nm2)) < 1e-6


def _mp_riccati():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    psi = lambda n, z: mp.sqrt(mp.pi * z / 2) * mp.besselj(n + mp.mpf(1) / 2, z)  # noqa: E731
    chi = lambda n, z: -mp.sqrt(mp.pi * z / 2) * mp.bessely(n + mp.mpf(1) / 2, z)  # noqa: E731
    dpsi = lambda n, z: psi(n - 1, z) - n * psi(n, z) / z  # noqa: E731
    dchi = lambda n, z: chi(n - 1, z) - n * chi(n, z) / z  # noqa: E731
    return mp, psi, chi, dpsi, dchi


def _mp_efficiencies(ab, x, n_max):
    ext = sca = 0
    for n in range(1, n_max + 1):
        a, b = ab(n)
        ext += (2 * n + 1) * (a + b).real
        sca += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
    return float(2 * ext / x**2), float(2 * sca / x**2)


@pytest.mark.parametrize("n_part, radius, lam", [
    (4.0, 1.0, 300.0),          # x = 0.028, lossless: Re(a_1) ~ |a_1|^2 is tiny
    (4.0, 1500.0, 300.0),       # x = 42, |m x| = 126: needs a deep D_n start
    (None, 10.0, 422.5),        # Drude near eps = 0
])
def test_hard_cases_against_mpmath(n_part, radius, lam):
    mp, psi, chi, dpsi, dchi = _mp_riccati()
    med = 1.33
    eps = complex(DRUDE.eps(np.array([lam]))[0]) if n_part is None else complex(n_part) ** 2
    m, x = mp.mpc(np.sqrt(eps) / med), mp.mpf(2 * np.pi * med * radius / lam)

    def ab(n):
        xi, dxi = psi(n, x) - 1j * chi(n, x), dpsi(n, x) - 1j * dchi(n, x)
        d = dpsi(n, m * x) / psi(n, m * x)
        a = (dpsi(n, x) - d / m * psi(n, x)) / (dxi - d / m * xi)
        b = (dpsi(n, x) - m * d * psi(n, x)) / (dxi - m * d * xi)
        return a, b

    qe, qs = _mp_efficiencies(ab, x, ls.backends.mie_native.n_max_wiscombe(float(x)) + 10)
    ce, cs, _ = mie_cross_sections(radius, eps, np.array([lam]), med)
    g = np.pi * radius**2
    assert ce[0] / g == pytest.approx(qe, rel=1e-11)
    assert cs[0] / g == pytest.approx(qs, rel=1e-11)


@pytest.mark.parametrize("core_eps, shell_eps, lam", [(2.25, None, 600.0), (None, 2.1, 520.0), (12 + 0.5j, None, 500.0)])
def test_coated_sphere_against_mpmath(core_eps, shell_eps, lam):
    """Layered recursion vs. Bohren & Huffman's coated-sphere formulas (BH 8.1) in 40-digit arithmetic."""
    mp, psi, chi, dpsi, dchi = _mp_riccati()
    med, rc, rs = 1.33, 30.0, 42.0
    metal = complex(DRUDE.eps(np.array([lam]))[0])
    ec, es = (metal if core_eps is None else core_eps), (metal if shell_eps is None else shell_eps)
    m1, m2 = mp.mpc(np.sqrt(complex(ec)) / med), mp.mpc(np.sqrt(complex(es)) / med)
    k = 2 * mp.pi * med / lam
    x, y = k * rc, k * rs

    def ab(n):
        A = ((m2 * psi(n, m2 * x) * dpsi(n, m1 * x) - m1 * dpsi(n, m2 * x) * psi(n, m1 * x))
             / (m2 * chi(n, m2 * x) * dpsi(n, m1 * x) - m1 * dchi(n, m2 * x) * psi(n, m1 * x)))
        B = ((m2 * psi(n, m1 * x) * dpsi(n, m2 * x) - m1 * psi(n, m2 * x) * dpsi(n, m1 * x))
             / (m2 * dchi(n, m2 * x) * psi(n, m1 * x) - m1 * dpsi(n, m1 * x) * chi(n, m2 * x)))
        xi, dxi = psi(n, y) - 1j * chi(n, y), dpsi(n, y) - 1j * dchi(n, y)
        fa, ga = dpsi(n, m2 * y) - A * dchi(n, m2 * y), psi(n, m2 * y) - A * chi(n, m2 * y)
        fb, gb = dpsi(n, m2 * y) - B * dchi(n, m2 * y), psi(n, m2 * y) - B * chi(n, m2 * y)
        a = (psi(n, y) * fa - m2 * dpsi(n, y) * ga) / (xi * fa - m2 * dxi * ga)
        b = (m2 * psi(n, y) * fb - dpsi(n, y) * gb) / (m2 * xi * fb - dxi * gb)
        return a, b

    qe, qs = _mp_efficiencies(ab, y, ls.backends.mie_native.n_max_wiscombe(float(y)) + 10)
    ce, cs, _ = mie_cross_sections([rc, rs], [ec, es], np.array([lam]), med)
    g = np.pi * rs**2
    assert ce[0] / g == pytest.approx(qe, rel=1e-11)
    assert cs[0] / g == pytest.approx(qs, rel=1e-11)


def test_shell_of_core_material_reproduces_sphere():
    wl = np.linspace(400, 900, 26)
    ff = FarFieldRequest(theta_deg=[0, 30, 90, 150, 180], phi_deg=[0, 45, 90], wavelengths_nm=[520.0])
    nf = NearFieldRequest(plane="xz", half_width_nm=80, points=15, wavelengths_nm=[520.0])
    out = Outputs(far_field=ff, near_field=nf)
    shells = [Shell(thickness_nm=10, material=DRUDE), Shell(thickness_nm=15, material=DRUDE)]
    layered = ls.solve(sim_of(LayeredSphere(core_radius_nm=30, shells=shells), wavelengths=wl, outputs=out), NATIVE, "cpu")
    plain = ls.solve(sim_of(ls.Sphere(radius_nm=55), wavelengths=wl, outputs=out), NATIVE, "cpu")
    for q in ("c_ext_nm2", "c_sca_nm2"):
        assert rel(getattr(layered, q), getattr(plain, q)) < 1e-10
    assert rel(layered.far_field[0].dcsca_domega_nm2, plain.far_field[0].dcsca_domega_nm2) < 1e-10
    np.testing.assert_allclose(layered.near_field[0].enhancement, plain.near_field[0].enhancement, rtol=1e-7)


# --- far field ----------------------------------------------------------------


@pytest.mark.parametrize("shape", [ls.Sphere(radius_nm=80), core_shell()], ids=["sphere", "core_shell"])
def test_far_field_integrates_to_csca(shape):
    mu, w = np.polynomial.legendre.leggauss(64)
    theta = np.degrees(np.arccos(mu))
    phi = np.arange(0, 360, 10.0)
    sim = sim_of(shape, wavelengths=[500.0, 650.0],
                 illumination=ls.PlaneWave(direction=(1, 2, 2), polarization=(2, 1, -2)),
                 outputs=Outputs(far_field=FarFieldRequest(theta_deg=theta.tolist(), phi_deg=phi.tolist(),
                                                           wavelengths_nm=[500.0, 650.0])))
    res = ls.solve(sim, NATIVE, "cpu")
    for ff, c_sca in zip(res.far_field, res.c_sca_nm2):
        d = np.asarray(ff.dcsca_domega_nm2)
        assert np.sum(w * d.mean(axis=1)) * 2 * np.pi == pytest.approx(c_sca, rel=1e-10)


def test_far_field_angle_convention():
    """phi = 0 is the E-plane: a small sphere (electric dipole) does not radiate along E."""
    sim = sim_of(ls.Sphere(radius_nm=3), DIELECTRIC, wavelengths=[600.0],
                 outputs=Outputs(far_field=FarFieldRequest(theta_deg=[0, 90], phi_deg=[0, 90], wavelengths_nm=[600.0])))
    d = np.asarray(ls.solve(sim, NATIVE, "cpu").far_field[0].dcsca_domega_nm2)
    assert d[1, 0] < 1e-4 * d[1, 1]  # theta=90 in the E-plane vs the H-plane
    assert d[0, 0] == pytest.approx(d[0, 1], rel=1e-12)  # forward direction is phi-independent


# --- near field -----------------------------------------------------------------


def test_near_field_tends_to_one_far_away():
    def max_dev(offset):
        nf = NearFieldRequest(plane="xy", offset_nm=offset, half_width_nm=300, points=5, wavelengths_nm=[530.0])
        res = ls.solve(sim_of(ls.Sphere(radius_nm=50), outputs=Outputs(near_field=nf)), NATIVE, "cpu")
        return np.max(np.abs(np.asarray(res.near_field[0].enhancement) - 1))

    far, very_far = max_dev(5_000.0), max_dev(100_000.0)
    assert very_far < 1e-3 and very_far < far < 0.1


def _field_at(sim, half_width):
    """Total E at the four corners (+-h, 0, +-h) of an xz near-field plane: radius sqrt(2) h, theta = 45/135 deg."""
    nf = NearFieldRequest(plane="xz", half_width_nm=half_width, points=2, wavelengths_nm=[sim.wavelength_array()[0]])
    res = ls.solve(sim.model_copy(update={"outputs": Outputs(near_field=nf)}), NATIVE, "cpu").near_field[0]
    e = np.asarray(res.e_real) + 1j * np.asarray(res.e_imag)  # (3, 2, 2)
    pts = np.array([[[x, 0.0, z] for z in (-half_width, half_width)] for x in (-half_width, half_width)])
    return e, pts / np.linalg.norm(pts, axis=-1, keepdims=True)


@pytest.mark.parametrize("shape, interface, eps_in, eps_out", [
    (ls.Sphere(radius_nm=50), 50.0, "core", "medium"),
    (core_shell(), 40.0, "core", "shell"),
    (core_shell(), 50.0, "shell", "medium"),
])
def test_near_field_boundary_conditions(shape, interface, eps_in, eps_out):
    """Tangential E and normal D are continuous at every interface (checks the internal coefficients)."""
    lam = 520.0
    sim = sim_of(shape, wavelengths=[lam])
    eps = {"core": complex(DRUDE.eps(np.array([lam]))[0]), "shell": complex(SILICA.eps(np.array([lam]))[0]),
           "medium": 1.33**2}
    d = 1e-6
    e_in, nhat = _field_at(sim, interface * (1 - d) / np.sqrt(2))
    e_out, _ = _field_at(sim, interface * (1 + d) / np.sqrt(2))
    nhat = np.moveaxis(nhat, -1, 0)
    en_in, en_out = np.sum(e_in * nhat, axis=0), np.sum(e_out * nhat, axis=0)
    et_in, et_out = e_in - en_in * nhat, e_out - en_out * nhat
    scale = np.max(np.abs(e_out))
    assert np.max(np.abs(et_in - et_out)) < 1e-4 * scale
    assert np.max(np.abs(eps[eps_in] * en_in - eps[eps_out] * en_out)) < 1e-4 * scale * abs(eps[eps_in])


# --- geometry, options, selection -------------------------------------------------


def test_any_incidence_and_orientation_average():
    wl = [450.0, 550.0, 700.0]
    base = ls.solve(sim_of(core_shell(), wavelengths=wl), NATIVE, "cpu")
    tilted = ls.solve(sim_of(core_shell(), wavelengths=wl, illumination=ls.PlaneWave(direction=(0, 1, 1), polarization=(1, 0, 0))),
                       NATIVE, "cpu")
    avg = ls.solve(sim_of(core_shell(), wavelengths=wl, orientation_average=True), NATIVE, "cpu")
    for r in (tilted, avg):
        assert rel(r.c_ext_nm2, base.c_ext_nm2) < 1e-12


def test_auto_selection_and_estimate():
    assert ls.choose_method(sim_of(core_shell()))[0].name == NATIVE
    with_field = sim_of(ls.Sphere(radius_nm=40), outputs=Outputs(
        far_field=FarFieldRequest(theta_deg=[0.0], wavelengths_nm=[520.0])))
    assert ls.choose_method(with_field)[0].name == NATIVE
    est = ls.estimate(sim_of(core_shell(), wavelengths=np.linspace(400, 800, 50)), NATIVE, "cpu")
    assert est["n_max"] > 0 and est["layers"] == 2 and est["memory_bytes"] > 0
    sim = sim_of(ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=20))
    ok, why = ls.get_backend(NATIVE).supports(sim)
    assert not ok and "spheres only" in why


def test_wavelength_chunks_agree():
    sim = sim_of(core_shell(), wavelengths=np.linspace(400, 900, 37))
    one = ls.solve(sim, NATIVE, "cpu")
    chunked = ls.solve(sim.model_copy(update={"options": {"mie": {"wavelength_chunk": 5}}}), NATIVE, "cpu")
    assert chunked.provenance["wavelength_chunks"] == 8
    assert rel(chunked.c_ext_nm2, one.c_ext_nm2) < 1e-12


# --- autodiff ---------------------------------------------------------------------


def test_autodiff_matches_finite_differences():
    torch = pytest.importorskip("torch")
    f64 = torch.float64
    rc = torch.tensor(30.0, dtype=f64, requires_grad=True)
    rs = torch.tensor(42.0, dtype=f64, requires_grad=True)
    er = torch.tensor(-8.0, dtype=f64, requires_grad=True)
    ei = torch.tensor(1.2, dtype=f64, requires_grad=True)
    lam = np.array([500.0, 560.0, 640.0])
    c_ext, c_sca, c_abs = mie_cross_sections([rc, rs], [torch.complex(er, ei), 2.1], torch.tensor(lam, dtype=f64), 1.33)
    (c_ext.sum() + 0.5 * c_abs.sum()).backward()

    def f(a=30.0, b=42.0, re=-8.0, im=1.2):
        e, _, ab = mie_cross_sections([a, b], [complex(re, im), 2.1], lam, 1.33)
        return float(e.sum() + 0.5 * ab.sum())

    h = 1e-5
    fd = {"rc": (f(a=30 + h) - f(a=30 - h)) / (2 * h), "rs": (f(b=42 + h) - f(b=42 - h)) / (2 * h),
          "er": (f(re=-8 + h) - f(re=-8 - h)) / (2 * h), "ei": (f(im=1.2 + h) - f(im=1.2 - h)) / (2 * h)}
    for name, t in (("rc", rc), ("rs", rs), ("er", er), ("ei", ei)):
        assert t.grad.item() == pytest.approx(fd[name], rel=1e-6), name


# --- GPU --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
def test_gpu_matches_cpu():
    out = Outputs(far_field=FarFieldRequest(theta_deg=[0, 60, 120, 180], phi_deg=[0, 90], wavelengths_nm=[520.0]),
                  near_field=NearFieldRequest(plane="xz", half_width_nm=80, points=11, wavelengths_nm=[520.0]))
    sim = sim_of(core_shell(), wavelengths=np.linspace(400, 900, 101), outputs=out)
    cpu = ls.solve(sim, NATIVE, "cpu")
    gpu = ls.solve(sim, NATIVE, "gpu")
    assert gpu.provenance["framework"] == "torch" and gpu.provenance["precision"] == "double"
    assert rel(gpu.c_ext_nm2, cpu.c_ext_nm2) < 1e-12
    assert rel(gpu.far_field[0].dcsca_domega_nm2, cpu.far_field[0].dcsca_domega_nm2) < 1e-12
    np.testing.assert_allclose(gpu.near_field[0].enhancement, cpu.near_field[0].enhancement, rtol=1e-10)
    single = ls.solve(sim.model_copy(update={"options": {"mie": {"precision": "single"}}}), NATIVE, "gpu")
    assert single.provenance["precision"] == "single"
    assert rel(single.c_ext_nm2, cpu.c_ext_nm2) < 1e-3


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
def test_autodiff_on_gpu():
    import torch

    r = torch.tensor(40.0, dtype=torch.float64, device="cuda", requires_grad=True)
    lam = torch.linspace(450, 700, 5, dtype=torch.float64, device="cuda")
    eps = DRUDE.eps(lam.cpu().numpy())  # NumPy inputs mix with CUDA tensors
    c_ext, _, _ = mie_cross_sections(r, eps, lam, 1.33)
    c_ext.sum().backward()
    h = 1e-5
    f = lambda a: mie_cross_sections(a, eps, lam.cpu().numpy(), 1.33)[0].sum()  # noqa: E731
    assert r.grad.item() == pytest.approx((f(40 + h) - f(40 - h)) / (2 * h), rel=1e-6)
