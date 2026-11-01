import numpy as np
import pytest

import omniscatter as osc
from omniscatter.backends.quasistatic import depolarization_factors
from omniscatter.spec import FarFieldRequest, LayeredSphere, Outputs, Shell

DIELECTRIC = osc.ConstantIndex(n_real=1.5, n_imag=0.02)
DRUDE = osc.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)
SILICA = osc.ConstantIndex(n_real=1.45)
QS = "quasistatic"
WL = np.linspace(400, 800, 81)


def sim_of(shape, material=DRUDE, wavelengths=WL, corrections="none", medium=1.33, **kw):
    return osc.Simulation(particles=[osc.Particle(shape=shape, material=material)], medium_index=medium,
                          wavelengths=list(wavelengths), options={QS: {"corrections": corrections}}, **kw)


def max_rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))))


@pytest.mark.parametrize("corrections", ["none", "radiative", "mlwa", "mie_dipole"])
def test_dielectric_5nm_sphere_matches_mie(corrections):
    sim = sim_of(osc.Sphere(radius_nm=5), DIELECTRIC, corrections=corrections)
    mie, qs = osc.solve(sim, "omniscatter-mie", "cpu"), osc.solve(sim, QS)
    for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        assert max_rel(getattr(qs, q), getattr(mie, q)) < 0.01
    assert not qs.warnings


def test_metal_5nm_sphere_matches_mie():
    """Across a sharp plasmon resonance the O(x^2) shift dominates; the corrections must remove it."""
    sim = sim_of(osc.Sphere(radius_nm=5), DRUDE)
    mie = osc.solve(sim, "omniscatter-mie", "cpu")
    err = {c: max_rel(osc.solve(sim_of(osc.Sphere(radius_nm=5), DRUDE, corrections=c), QS).c_ext_nm2, mie.c_ext_nm2)
           for c in ("none", "mlwa", "mie_dipole")}
    assert err["mie_dipole"] < 0.01
    assert err["mie_dipole"] < err["mlwa"] < err["none"]
    qs = osc.solve(sim, QS)  # even the static dipole puts the peak in the right place
    assert abs(WL[np.argmax(qs.c_ext_nm2)] - WL[np.argmax(mie.c_ext_nm2)]) <= WL[1] - WL[0]


def test_coated_sphere_matches_mie():
    shape = LayeredSphere(core_radius_nm=3, shells=[Shell(thickness_nm=2, material=SILICA)])
    for material, corrections in ((DIELECTRIC, "none"), (DRUDE, "mlwa")):
        sim = sim_of(shape, material, corrections=corrections)
        assert max_rel(osc.solve(sim, QS).c_ext_nm2, osc.solve(sim, "omniscatter-mie", "cpu").c_ext_nm2) < 0.01


def test_coated_sphere_of_one_material_is_a_sphere():
    shape = LayeredSphere(core_radius_nm=3, shells=[Shell(thickness_nm=1, material=DRUDE), Shell(thickness_nm=1, material=DRUDE)])
    for c in ("none", "mlwa"):
        a, b = osc.solve(sim_of(shape, corrections=c), QS), osc.solve(sim_of(osc.Sphere(radius_nm=5), corrections=c), QS)
        assert max_rel(a.c_ext_nm2, b.c_ext_nm2) < 1e-12


def test_depolarization_factors():
    np.testing.assert_allclose(depolarization_factors(7, 7), [1 / 3] * 3, rtol=1e-12)
    # references from numerical quadrature of L_z = abc/2 int dq / ((c^2+q) sqrt((a^2+q)(b^2+q)(c^2+q)))
    assert depolarization_factors(10, 30)[2] == pytest.approx(0.1087094650525864, rel=1e-12)
    assert depolarization_factors(30, 10)[2] == pytest.approx(0.6353888898560048, rel=1e-12)
    for a, c in ((10, 30), (30, 10), (10, 10 + 1e-7), (10, 10 - 1e-7), (10, 10.01)):
        assert depolarization_factors(a, c).sum() == pytest.approx(1.0, abs=1e-14)
    # the near-sphere series joins the closed forms smoothly
    for c in (10 * (1 + 1e-9), 10 * (1 - 1e-9)):
        assert depolarization_factors(10, c)[2] == pytest.approx(1 / 3, abs=1e-8)


def test_spheroid_of_aspect_one_is_a_sphere():
    a = osc.solve(sim_of(osc.Spheroid(semi_axis_xy_nm=5, semi_axis_z_nm=5)), QS)
    b = osc.solve(sim_of(osc.Sphere(radius_nm=5)), QS)
    assert max_rel(a.c_ext_nm2, b.c_ext_nm2) < 1e-12


def _gans_peak(aspect, polarization, direction, corrections="none"):
    r_eq = 5.0
    a = r_eq / aspect ** (1 / 3)  # constant volume, prolate along z
    wl = np.linspace(400, 1000, 601)
    sim = sim_of(osc.Spheroid(semi_axis_xy_nm=a, semi_axis_z_nm=a * aspect), wavelengths=wl, corrections=corrections,
                 illumination=osc.PlaneWave(direction=direction, polarization=polarization))
    return wl[np.argmax(osc.solve(sim, QS).c_ext_nm2)]


@pytest.mark.parametrize("corrections", ["none", "mlwa"])
def test_gans_resonance_red_shifts_with_aspect_ratio(corrections):
    long = [_gans_peak(ar, (0, 0, 1), (1, 0, 0), corrections) for ar in (1.0, 1.5, 2.0, 3.0)]
    assert all(np.diff(long) > 0), long
    trans = [_gans_peak(ar, (1, 0, 0), (0, 0, 1), corrections) for ar in (1.0, 2.0, 3.0)]
    assert all(np.diff(trans) <= 0), trans  # the transverse mode blue-shifts slightly


def test_polarization_relative_to_axis_and_orientation_average():
    shape = osc.Spheroid(semi_axis_xy_nm=3, semi_axis_z_nm=8)
    along_z = osc.solve(sim_of(shape, illumination=osc.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1))), QS)
    along_x = osc.solve(sim_of(shape, illumination=osc.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0))), QS)
    tilted = osc.solve(sim_of(shape, illumination=osc.PlaneWave(direction=(1, 0, 0), polarization=(0, 1, 1))), QS)
    avg = osc.solve(sim_of(shape, orientation_average=True), QS)
    for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        z, x = np.asarray(getattr(along_z, q)), np.asarray(getattr(along_x, q))
        assert max_rel(getattr(tilted, q), (x + z) / 2) < 1e-12  # 45 deg between x-like and z
        assert max_rel(getattr(avg, q), (2 * x + z) / 3) < 1e-12


@pytest.mark.parametrize("corrections", ["none", "mlwa"])
def test_far_field_integrates_to_csca(corrections):
    mu, w = np.polynomial.legendre.leggauss(16)
    sim = sim_of(osc.Spheroid(semi_axis_xy_nm=3, semi_axis_z_nm=8), corrections=corrections, wavelengths=[600.0],
                 illumination=osc.PlaneWave(direction=(1, 0, 1), polarization=(1, 1, -1)),
                 outputs=Outputs(far_field=FarFieldRequest(theta_deg=np.degrees(np.arccos(mu)).tolist(),
                                                           phi_deg=np.arange(0, 360, 15.0).tolist(), wavelengths_nm=[600.0])))
    res = osc.solve(sim, QS)
    d = np.asarray(res.far_field[0].dcsca_domega_nm2)
    assert np.sum(w * d.mean(axis=1)) * 2 * np.pi == pytest.approx(res.c_sca_nm2[0], rel=1e-12)


def test_energy_balance_with_radiative_damping():
    sim = sim_of(osc.Spheroid(semi_axis_xy_nm=4, semi_axis_z_nm=9), SILICA, corrections="mlwa")
    res = osc.solve(sim, QS)
    assert np.max(np.abs(res.c_abs_nm2)) < 1e-12 * np.max(res.c_ext_nm2)  # lossless: extinction is all scattering


def test_size_parameter_warning():
    assert not osc.solve(sim_of(osc.Sphere(radius_nm=5)), QS).warnings
    big = osc.solve(sim_of(osc.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=20)), QS)
    assert any("size parameter" in w for w in big.warnings)


def test_supports_and_auto_selection():
    qs = osc.get_backend(QS)
    ok, why = qs.supports(sim_of(osc.Cylinder(radius_nm=5, length_nm=20)))
    assert not ok and "cylinder" in why
    ok, why = qs.supports(sim_of(osc.Spheroid(semi_axis_xy_nm=3, semi_axis_z_nm=6), corrections="mie_dipole"))
    assert not ok and "mie_dipole" in why
    # quasistatic is never auto-selected over an exact or full-wave method
    assert osc.choose_method(sim_of(osc.Spheroid(semi_axis_xy_nm=3, semi_axis_z_nm=6)))[0].method != QS
