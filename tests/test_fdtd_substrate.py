"""FDTD scattering by particles on a substrate: normal incidence (layered TFSF fed by an auxiliary column) and
oblique incidence (tables of the exact discrete background, layered.py).

Reference: a small sphere as a point dipole (Mie a_1 polarizability) driven by the background wave (Fresnel)
and coupled to its own reflected field through Sommerfeld integrals (G_par, G_perp); exact in the dipole limit.

Measured (RTX 5070, float32), n = 2, r = 20 nm sphere in air above glass (n = 1.5), 400-800 nm:
- centre 40 nm above the glass, from above: C_sca within 0.17 % (dx = 2 nm) and 0.11 % (dx = 1.33 nm);
- centre 100 nm above: 0.23 % and 0.19 % (with 10 PML cells instead of 16: 1.3 % at 800 nm);
- through the glass (direction +z): 0.23 %.
A particle of the host's own index scatters exactly nothing (C_sca = 0 to 1e-6 nm^2); its C_abs is
float32 cancellation noise (~0.004 nm^2 for this box, exactly 0 in double precision).

Oblique incidence, same sphere 40 nm above the glass, dx = 2 nm: from above at 45 deg, s 0.17 %, p 0.33 %;
through the glass at 30 deg (p) 0.20 %; total internal reflection at 60 deg, s 0.47 %, p 0.7-1.7 % (the
evanescent field varies across the sphere, which the dipole model misses). Null particle in double precision:
C_sca <= 2e-14 nm^2 (TIR) and 2e-17 nm^2 (from above), C_abs <= 3e-5 nm^2.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn, spherical_yn

import lightsmith as ls
from lightsmith.device import _torch_cuda
from lightsmith.spec import FarFieldRequest, NearFieldRequest, Outputs, Substrate

needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
GLASS = Substrate(material=ls.ConstantIndex(n_real=1.5))
DOWN = ls.PlaneWave(direction=(0, 0, -1), polarization=(1, 0, 0))


def _a1(m, x):
    def psi(z):
        return z * spherical_jn(1, z)

    def dpsi(z):
        return spherical_jn(1, z) + z * spherical_jn(1, z, derivative=True)

    def h(z, d=False):
        return spherical_jn(1, z, derivative=d) + 1j * spherical_yn(1, z, derivative=d)

    xi, dxi = x * h(x), h(x) + x * h(x, True)
    return (m * psi(m * x) * dpsi(x) - psi(x) * dpsi(m * x)) / (m * psi(m * x) * dxi - xi * dpsi(m * x))


def _sommerfeld(k1, e1, e2, h, fn):
    """int_0^inf fn(s, s_z, r_s, r_p) e^{2 i k1 h s_z} ds, with fn carrying its 1/s_z (s = sin t, then cosh t)."""
    def f(s_, sz):
        kz2 = np.sqrt(complex(e2 / e1 - s_**2))
        rs, rp = (sz - kz2) / (sz + kz2), (e2 * sz - e1 * kz2) / (e2 * sz + e1 * kz2)
        return fn(s_, sz, rs, rp) * np.exp(2j * k1 * h * sz)

    def part(g, a, b):
        return quad(lambda t: g(t).real, a, b, limit=400)[0] + 1j * quad(lambda t: g(t).imag, a, b, limit=400)[0]

    total = part(lambda t: f(np.sin(t), np.cos(t)) * np.cos(t), 0, np.pi / 2)
    return total + part(lambda t: f(np.cosh(t), 1j * np.sinh(t)) * np.sinh(t), 0, np.arcsinh(20 / (k1 * h)))


def dipole_reference(r, n_p, n1, n2, h, wavelengths, theta_deg=0.0, pol="s", from_below=False):
    """C_sca (nm^2) of a sphere centred h above the interface (host n1 above, n2 below), lit in the xz plane at
    theta from the normal of the incidence medium; s: E along y, p: in the xz plane; per unit incident intensity."""
    out = []
    th = np.radians(theta_deg)
    e1, e2 = n1**2, n2**2
    for lam in wavelengths:
        k0 = 2 * np.pi / lam
        k1, k2 = n1 * k0, n2 * k0
        kx = (n2 if from_below else n1) * k0 * np.sin(th)
        k1z, k2z = np.sqrt(complex(k1**2 - kx**2)), np.sqrt(complex(k2**2 - kx**2))
        if from_below:  # transmitted (or evanescent) wave in the host; incident intensity n2 |E|^2 / 2
            if pol == "s":
                E = np.array([0, 2 * k2z / (k2z + k1z) * np.exp(1j * k1z * h), 0])
            else:
                ph = (e2 / e1) * 2 * e1 * k2z / (e1 * k2z + e2 * k1z) * np.exp(1j * k1z * h) / k2
                E = np.array([k1z * ph, 0, -kx * ph])
            norm = n1 / n2
        else:
            dn, up = np.exp(-1j * k1z * h), np.exp(1j * k1z * h)
            if pol == "s":
                E = np.array([0, dn + (k1z - k2z) / (k1z + k2z) * up, 0])
            else:
                rp = (e2 * k1z - e1 * k2z) / (e2 * k1z + e1 * k2z)
                E = np.array([-(k1z / k1) * (dn - rp * up), 0, -(kx / k1) * (dn + rp * up)])
            norm = 1.0
        alpha = 6j * np.pi * _a1(n_p / n1, k1 * r) / k1**3
        g_par = k1**3 / (8 * np.pi) * 1j * _sommerfeld(k1, e1, e2, h, lambda s_, sz, rs, rp: s_ / sz * (rs - sz**2 * rp))
        g_perp = k1**3 / (4 * np.pi) * 1j * _sommerfeld(k1, e1, e2, h, lambda s_, sz, rs, rp: s_**3 / sz * rp)
        a_par, a_perp = alpha / (1 - alpha * g_par), alpha / (1 - alpha * g_perp)
        P = ((abs(a_par * E[0]) ** 2 + abs(a_par * E[1]) ** 2) * (1 + 6 * np.pi / k1**3 * g_par.imag)
             + abs(a_perp * E[2]) ** 2 * (1 + 6 * np.pi / k1**3 * g_perp.imag))
        out.append(norm * k1**4 / (6 * np.pi) * P)
    return np.array(out)


def oblique(theta_deg, pol, from_below=False):
    th = np.radians(theta_deg)
    k = np.array([np.sin(th), 0.0, np.cos(th) if from_below else -np.cos(th)])
    e = np.array([0.0, 1.0, 0.0]) if pol == "s" else np.cross([0.0, 1.0, 0.0], k)
    return ls.PlaneWave(direction=tuple(k), polarization=tuple(e))


def sim_on(sub, n_p=2.0, r=20.0, h=40.0, wl=(400.0, 500.0, 600.0, 800.0), illumination=DOWN, **fdtd):
    return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=r), material=ls.ConstantIndex(n_real=n_p),
                                                position_nm=(0, 0, h))],
                         wavelengths=list(wl), illumination=illumination, substrate=sub, options={"fdtd": fdtd})


def test_dipole_reference_limits():
    wl = [400.0, 600.0]
    k = 2 * np.pi / np.array(wl)
    free = 6 * np.pi * np.abs(_a1(2.0, k * 20)) ** 2 / k**2  # Mie dipole term
    for theta, pol, below in ((0.0, "s", False), (0.0, "s", True), (37.0, "s", False), (37.0, "p", True)):
        np.testing.assert_allclose(dipole_reference(20, 2.0, 1.0, 1.0, 40, wl, theta, pol, below), free, rtol=1e-10)
    # normal incidence: s and p are the same wave
    np.testing.assert_allclose(dipole_reference(20, 2.0, 1.0, 1.5, 40, wl, 0.0, "p"),
                               dipole_reference(20, 2.0, 1.0, 1.5, 40, wl, 0.0, "s"), rtol=1e-12)


def test_discrete_layered_solver_matches_fresnel():
    """The exact discrete background (layered.solve) reflects as continuum Fresnel when dz << lambda, including
    total internal reflection, for s and p (measured |r| within 3e-5 at dz = 2 nm, 0 at 0.5 nm)."""
    from lightsmith.backends._fdtd.layered import layer_eta, solve
    from lightsmith.backends._fdtd.media import Medium

    host, sub, dz, lam = Medium(1.0), Medium(2.25), 1.0, 600.0
    dt = 0.5 * dz / np.sqrt(3)
    z = (np.arange(80) - 40) * dz
    om = np.array([2 * np.pi / lam])
    for from_top, theta in ((True, 30.0), (True, 60.0), (False, 30.0), (False, 60.0)):
        n1, n2 = (1.0, 1.5) if from_top else (1.5, 1.0)
        sx = n1 * np.sin(np.radians(theta))
        Kx, Om = 2 * np.sin(om * sx * dz / 2) / dz, 2 * np.sin(om * dt / 2) / dt
        eta_t = layer_eta(z, False, om, dt, dz, 1e-9, host, sub)
        eta_z = layer_eta(z + dz / 2, True, om, dt, dz, 1e-9, host, sub)
        kz1 = np.sqrt(complex((n1**2 - sx**2))) * om[0]
        kz2 = np.sqrt(complex((n2**2 - sx**2))) * om[0]
        kz2 = kz2 if kz2.imag >= 0 else -kz2
        for pol, e_t, r_ana, sign in (("s", (0.0, 1.0), (kz1 - kz2) / (kz1 + kz2), 1),
                                      ("p", (1.0, 0.0), (n2**2 * kz1 - n1**2 * kz2) / (n2**2 * kz1 + n1**2 * kz2), -1)):
            f = solve(eta_t, eta_z, Kx, np.zeros(1), Om, dz, e_t, from_top)
            end, i = (-1 if from_top else 0), (1 if pol == "s" else 0)
            comp = "Ey" if pol == "s" else "Ex"
            r_num = sign * (f[comp][0, end] / f["inc"][0, i] - 1) * np.exp(-2j * kz1 * abs(z[end]))
            assert abs(r_num - r_ana) < 2e-4, (from_top, theta, pol, r_num, r_ana)


def test_layered_background_meshes_the_plane_exactly():
    from lightsmith.backends._fdtd.media import Medium
    from lightsmith.backends._fdtd.substrate import layered_background

    host, sub = Medium(1.0), Medium(2.25)
    # nodes at z = 0, 1, 2, ...; the plane at z = 2.3 cuts E_x/E_y pixels at node 2 (f = 0.8) and E_z at 1.5 / 2.5
    inv, groups = layered_background((4, 4, 6), np.zeros(3), 1.0, 2.3, host, sub)
    assert groups == []
    assert inv[0][2, 2, 2] == pytest.approx(1 / (0.8 * 2.25 + 0.2))  # tangential: 1/<eps>
    assert inv[2][2, 2, 2] == pytest.approx(0.3 / 2.25 + 0.7)  # normal, pixel [2, 3], f = 0.3: <1/eps>
    assert inv[0][2, 2, 1] == pytest.approx(1 / 2.25) and inv[0][2, 2, 3] == 1.0


def test_substrate_support_and_refusals():
    fdtd = ls.get_backend("fdtd")
    assert fdtd.supports(sim_on(GLASS))[0]
    assert ls.choose_method(sim_on(GLASS))[0].method == "fdtd"
    assert not ls.get_backend("mie").supports(sim_on(GLASS))[0]
    assert fdtd.supports(sim_on(GLASS, illumination=oblique(40, "p")))[0]
    assert fdtd.supports(sim_on(GLASS, illumination=oblique(60, "s", from_below=True)))[0]  # TIR
    graze = ls.PlaneWave(direction=(0.999, 0, -0.0447), polarization=(0, 1, 0))
    assert "graze" in fdtd.supports(sim_on(GLASS, illumination=graze))[1]
    fitted = Substrate(material=ls.DrudeLorentz(eps_inf=2.25, poles=[]))
    assert "constant-index" in fdtd.supports(sim_on(fitted, illumination=oblique(30, "s", from_below=True)))[1]
    far = sim_on(GLASS).model_copy(update={"outputs": Outputs(far_field=FarFieldRequest(theta_deg=[0],
                                                                                         wavelengths_nm=[500]))})
    assert "far_field" in fdtd.supports(far)[1]
    gold = Substrate(material=ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07))
    assert "absorbing" in fdtd.supports(sim_on(gold))[1]


def test_substrate_tfsf_does_not_leak_cpu():
    """A sphere of the host medium on glass: the layered TFSF box must scatter nothing."""
    sim = sim_on(GLASS, n_p=1.0, r=10.0, h=15.0, wl=(400.0, 800.0), spacing_nm=4.0, pml_cells=6, padding_nm=8,
                 pml_gap_nm=8)  # small grid: the check needs no PML quality
    res = ls.solve(sim, "fdtd", "cpu")
    assert np.max(np.abs(res.c_sca_nm2)) < 1e-9
    assert np.max(np.abs(res.c_abs_nm2)) < 1e-6  # float64
    assert res.provenance["substrate"]["incidence"] == "from above"


def test_oblique_substrate_tfsf_does_not_leak_cpu():
    """A sphere of the host medium on glass, lit at 40 degrees (s): the tables are the grid's own background,
    so nothing scatters, and the near field is |e^{-ik_z z} + r_s e^{ik_z z}|^2 above, |1 + r_s|^2 below
    (measured within 2e-4), uniform along x and with the phase k sin(theta) x."""
    lam, th = 600.0, np.radians(40)
    sim = sim_on(GLASS, n_p=1.0, r=10.0, h=15.0, wl=(400.0, 800.0), illumination=oblique(40, "s"), spacing_nm=4.0,
                 pml_cells=6, padding_nm=8, pml_gap_nm=8)
    sim = sim.model_copy(update={"outputs": Outputs(near_field=NearFieldRequest(
        plane="xz", offset_nm=0.0, half_width_nm=60.0, points=31, wavelengths_nm=[lam]))})
    res = ls.solve(sim, "fdtd", "cpu")
    assert np.max(np.abs(res.c_sca_nm2)) < 1e-12
    assert np.max(np.abs(res.c_abs_nm2)) < 1e-4  # measured ~1e-6 (float64)
    assert res.provenance["substrate"]["source"] == "layered tables (oblique)"
    nf = res.near_field[0]
    z, enh = np.asarray(nf.axis2_nm), np.asarray(nf.enhancement)
    k0 = 2 * np.pi / lam
    k1z, k2z = k0 * np.cos(th), k0 * np.sqrt(2.25 - np.sin(th) ** 2)
    rs = (k1z - k2z) / (k1z + k2z)
    ana = np.where(z > 0, np.abs(np.exp(-1j * k1z * z) + rs * np.exp(1j * k1z * z)) ** 2, abs(1 + rs) ** 2)
    ok = np.isfinite(enh[15]) & (np.abs(z) > 4)
    assert ok.sum() > 15
    np.testing.assert_allclose(enh[15][ok], ana[ok], rtol=2e-3)
    np.testing.assert_allclose(enh[:, ok], np.broadcast_to(enh[15][ok], enh[:, ok].shape), rtol=1e-8)
    ey = np.asarray(nf.e_real)[1] + 1j * np.asarray(nf.e_imag)[1]
    x = np.asarray(nf.axis1_nm)
    step = np.angle(ey[1:, 25] / ey[:-1, 25])
    np.testing.assert_allclose(step, k0 * np.sin(th) * (x[1] - x[0]), atol=1e-6)


def test_substrate_near_field_is_the_standing_wave_cpu():
    """Around a sphere of the host medium the near field is the background: |e^{-ikz} + r e^{ikz}|^2 above the
    glass and |t|^2 below (measured within 0.03 % at dx = 2 nm); points beyond the column's range are NaN."""
    lam = 600.0
    sim = sim_on(GLASS, n_p=1.0, r=10.0, h=15.0, wl=(400.0, 800.0), spacing_nm=4.0, pml_cells=6, padding_nm=8,
                 pml_gap_nm=8)
    sim = sim.model_copy(update={"outputs": Outputs(near_field=NearFieldRequest(
        plane="xz", offset_nm=0.0, half_width_nm=60.0, points=31, wavelengths_nm=[lam]))})
    res = ls.solve(sim, "fdtd", "cpu")
    enh = np.asarray(res.near_field[0].enhancement)
    z = np.asarray(res.near_field[0].axis2_nm)
    k1, r = 2 * np.pi / lam, (1 - 1.5) / 2.5
    ana = np.where(z > 0, np.abs(np.exp(-1j * k1 * z) + r * np.exp(1j * k1 * z)) ** 2, (2 / 2.5) ** 2)
    ok = np.isfinite(enh[15]) & (np.abs(z) > 4)
    assert ok.sum() > 15
    np.testing.assert_allclose(enh[15][ok], ana[ok], rtol=5e-3)
    np.testing.assert_allclose(enh[:, ok], np.broadcast_to(enh[15][ok], enh[:, ok].shape), rtol=1e-9)


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("h", [40.0, 100.0])
def test_sphere_above_glass_matches_dipole_reference(h):
    wl = (400.0, 500.0, 600.0, 800.0)
    res = ls.solve(sim_on(GLASS, h=h, wl=wl, spacing_nm=2.0), "fdtd", "gpu")
    ref = dipole_reference(20, 2.0, 1.0, 1.5, h, wl)
    np.testing.assert_allclose(res.c_sca_nm2, ref, rtol=5e-3)  # measured 0.17 % (h 40), 0.23 % (h 100)
    assert res.provenance["pml_cells"] == 16


@pytest.mark.gpu
@needs_cuda
def test_sphere_lit_through_glass_matches_dipole_reference():
    wl = (400.0, 500.0, 600.0, 800.0)
    up = ls.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0))
    res = ls.solve(sim_on(GLASS, wl=wl, illumination=up, spacing_nm=2.0), "fdtd", "gpu")
    np.testing.assert_allclose(res.c_sca_nm2, dipole_reference(20, 2.0, 1.0, 1.5, 40, wl, from_below=True),
                               rtol=5e-3)  # measured 0.23 %
    assert res.provenance["substrate"]["incidence"] == "through the substrate"


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("theta, pol, below, rtol", [(45.0, "s", False, 5e-3), (45.0, "p", False, 8e-3),
                                                     (30.0, "p", True, 5e-3), (60.0, "s", True, 1e-2),
                                                     (60.0, "p", True, 3e-2)])
def test_oblique_sphere_above_glass_matches_dipole_reference(theta, pol, below, rtol):
    """Oblique s and p from above, through the glass, and total internal reflection (60 deg from the glass:
    the sphere sits in the evanescent field; the dipole model is least exact there)."""
    wl = (400.0, 500.0, 600.0, 800.0)
    res = ls.solve(sim_on(GLASS, wl=wl, illumination=oblique(theta, pol, below), spacing_nm=2.0), "fdtd", "gpu")
    np.testing.assert_allclose(res.c_sca_nm2, dipole_reference(20, 2.0, 1.0, 1.5, 40, wl, theta, pol, below),
                               rtol=rtol)
