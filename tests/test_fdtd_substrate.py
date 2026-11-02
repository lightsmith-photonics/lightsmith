"""FDTD scattering by particles on a substrate (normal incidence; layered TFSF fed by an auxiliary column).

Reference: a small sphere as a point dipole (Mie a_1 polarizability) driven by the incident + reflected
wave and coupled to its own reflected field through Sommerfeld integrals; exact in the dipole limit.

Measured (RTX 5070, float32), n = 2, r = 20 nm sphere in air above glass (n = 1.5), 400-800 nm:
- centre 40 nm above the glass, from above: C_sca within 0.17 % (dx = 2 nm) and 0.11 % (dx = 1.33 nm);
- centre 100 nm above: 0.23 % and 0.19 % (with 10 PML cells instead of 16: 1.3 % at 800 nm);
- through the glass (direction +z): 0.23 %.
A particle of the host's own index scatters exactly nothing (C_sca = 0 to 1e-6 nm^2); its C_abs is
float32 cancellation noise (~0.004 nm^2 for this box, exactly 0 in double precision).
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


def _g_parallel(k1, e1, e2, h):
    """Reflected Green function at a parallel dipole, (k^3/8pi) i int (s/s_z)(r_s - s_z^2 r_p) e^{2ikh s_z} ds."""
    def f(s, sz):
        kz2 = np.sqrt(complex(e2 / e1 - s**2))
        rs, rp = (sz - kz2) / (sz + kz2), (e2 * sz - e1 * kz2) / (e2 * sz + e1 * kz2)
        return (rs - sz**2 * rp) * np.exp(2j * k1 * h * sz)

    def part(fn, a, b):
        return quad(lambda t: fn(t).real, a, b, limit=400)[0] + 1j * quad(lambda t: fn(t).imag, a, b, limit=400)[0]

    total = part(lambda t: f(np.sin(t), np.cos(t)) * np.sin(t), 0, np.pi / 2)  # propagating, s = sin t
    total += part(lambda t: -1j * f(np.cosh(t), 1j * np.sinh(t)) * np.cosh(t), 0, np.arcsinh(20 / (k1 * h)))
    return k1**3 / (8 * np.pi) * 1j * total


def dipole_reference(r, n_p, n1, n2, h, wavelengths, from_below=False):
    """C_sca (nm^2) of a sphere centred h above the interface, x polarisation, normal incidence."""
    out = []
    for lam in wavelengths:
        k1 = 2 * np.pi * n1 / lam
        alpha = 6j * np.pi * _a1(n_p / n1, k1 * r) / k1**3
        G = _g_parallel(k1, n1**2, n2**2, h)
        if from_below:  # transmitted wave, per unit incident intensity in the substrate
            e_bg = 2 * n2 / (n1 + n2) * np.sqrt(n1 / n2)
        else:
            e_bg = np.exp(-1j * k1 * h) + (n1 - n2) / (n1 + n2) * np.exp(1j * k1 * h)
        out.append(k1**4 / (6 * np.pi) * abs(alpha / (1 - alpha * G) * e_bg) ** 2 * (1 + 6 * np.pi / k1**3 * G.imag))
    return np.array(out)


def sim_on(sub, n_p=2.0, r=20.0, h=40.0, wl=(400.0, 500.0, 600.0, 800.0), illumination=DOWN, **fdtd):
    return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=r), material=ls.ConstantIndex(n_real=n_p),
                                                position_nm=(0, 0, h))],
                         wavelengths=list(wl), illumination=illumination, substrate=sub, options={"fdtd": fdtd})


def test_dipole_reference_limits():
    wl = [400.0, 600.0]
    k = 2 * np.pi / np.array(wl)
    free = 6 * np.pi * np.abs(_a1(2.0, k * 20)) ** 2 / k**2  # Mie dipole term
    np.testing.assert_allclose(dipole_reference(20, 2.0, 1.0, 1.0, 40, wl), free, rtol=1e-10)
    np.testing.assert_allclose(dipole_reference(20, 2.0, 1.0, 1.0, 40, wl, from_below=True), free, rtol=1e-10)


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
    oblique = sim_on(GLASS, illumination=ls.PlaneWave(direction=(0.6, 0, -0.8), polarization=(0.8, 0, 0.6)))
    assert "normal incidence" in fdtd.supports(oblique)[1]
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
