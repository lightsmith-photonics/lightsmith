import numpy as np
import pytest

import lightsmith as ls
from lightsmith.spec import FarFieldRequest, LayeredSphere, NearFieldRequest, Outputs, Shell

DIELECTRIC = ls.ConstantIndex(n_real=1.5, n_imag=0.02)


def sim(shape=None, mat=DIELECTRIC, npd=12, wl=(500.0,), **kw):
    opts = kw.pop("options", {})
    return ls.Simulation(
        particles=[ls.Particle(shape=shape or ls.Sphere(radius_nm=40), material=mat)],
        medium_index=1.33, wavelengths=list(wl),
        options={"dda": {"dipoles_per_diameter": npd, **opts}}, **kw)


def test_far_field_integrates_to_c_sca():
    th = np.linspace(0, 180, 181)
    s = sim(outputs=Outputs(far_field=FarFieldRequest(theta_deg=th.tolist(), phi_deg=list(range(0, 360, 10)),
                                                       wavelengths_nm=[500.0])))
    r = ls.solve(s, "dda", "cpu")
    f = np.asarray(r.far_field[0].dcsca_domega_nm2).mean(axis=1)  # phi-average (uniform periodic grid)
    integral = np.trapezoid(2 * np.pi * f * np.sin(np.radians(th)), np.radians(th))
    assert integral == pytest.approx(r.c_sca_nm2[0], rel=1e-3)


def test_near_field_tends_to_incident_and_masks_inside():
    s = sim(outputs=Outputs(near_field=NearFieldRequest(half_width_nm=400, points=21, wavelengths_nm=[500.0])))
    e = np.asarray(ls.solve(s, "dda", "cpu").near_field[0].enhancement)
    assert np.isnan(e[10, 10])
    assert e[0, 0] == pytest.approx(1.0, abs=0.02)


def test_shell_of_core_material_equals_sphere():
    a = ls.solve(sim(), "dda", "cpu")
    b = ls.solve(sim(shape=LayeredSphere(core_radius_nm=25, shells=[Shell(thickness_nm=15, material=DIELECTRIC)])),
                  "dda", "cpu")
    np.testing.assert_allclose(b.c_ext_nm2, a.c_ext_nm2, rtol=1e-10)


def test_orientation_average_of_sphere():
    a = ls.solve(sim(), "dda", "cpu")
    b = ls.solve(sim(orientation_average=True, options={"orientations": 4}), "dda", "cpu")
    np.testing.assert_allclose(b.c_ext_nm2, a.c_ext_nm2, rtol=1e-3)


def test_spheroid_polarization_anisotropy():
    rod = ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=30)
    gold_like = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)
    wl = np.linspace(500, 900, 9).tolist()
    along = ls.solve(sim(rod, gold_like, wl=wl, illumination=ls.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1))),
                      "dda", "cpu")
    across = ls.solve(sim(rod, gold_like, wl=wl), "dda", "cpu")
    # The longitudinal plasmon of a prolate rod is red-shifted relative to the transverse one.
    assert wl[int(np.argmax(along.c_ext_nm2))] > wl[int(np.argmax(across.c_ext_nm2))]


def test_dimer_runs_and_reports_provenance():
    p = lambda x: ls.Particle(shape=ls.Sphere(radius_nm=20), material=DIELECTRIC, position_nm=(x, 0, 0))  # noqa: E731
    s = ls.Simulation(particles=[p(-25), p(25)], wavelengths=[500.0], options={"dda": {"dipoles_per_diameter": 10}})
    assert ls.choose_method(s)[0].method in ("tmatrix", "bem")  # T-matrix wins for spheres when treams is installed
    r = ls.solve(s, "dda", "cpu")
    assert r.provenance["dipoles"][0] > 0 and r.c_ext_nm2[0] > 0


@pytest.mark.parametrize("npd", [12, 16, 24])
def test_weighted_discretization_matches_mie_on_a_dielectric(npd):
    mie = ls.solve(sim(), "mie", "cpu")
    r = ls.solve(sim(npd=npd, options={"discretization": "weighted"}), "dda", "cpu")
    assert r.c_ext_nm2[0] == pytest.approx(mie.c_ext_nm2[0], rel=5e-3)
    assert r.c_abs_nm2[0] == pytest.approx(mie.c_abs_nm2[0], rel=5e-3)


def test_weighted_lattice_keeps_the_particle_volume():
    from lightsmith.backends.dda import _lattice

    for npd in (12, 24):
        lat = _lattice(sim(npd=npd), npd, "weighted", subsamples=8)
        volume = lat.frac.sum() * lat.d**3
        assert volume == pytest.approx(4 / 3 * np.pi * 40**3, rel=2e-3)
        assert lat.frac.max() == 1.0 and 0 < lat.frac[lat.frac < 1].max() < 1  # full and partial sites


def test_extrapolation_is_exact_on_a_polynomial_in_one_over_npd():
    from lightsmith.backends.dda import _extrapolate

    npds = [12, 16, 24]
    h = 1 / np.array(npds, float)
    values = np.stack([3.0 + 2.0 * h + 5.0 * h**2, -1.0 + 7.0 * h - 4.0 * h**2], axis=1)[:, None, :]
    np.testing.assert_allclose(_extrapolate(npds, values), [[3.0, -1.0]], atol=1e-12)
    linear = (2.0 + 6.0 * h[:2])[:, None, None]  # two values: a straight line
    np.testing.assert_allclose(_extrapolate(npds[:2], linear), [[2.0]], atol=1e-12)


def test_extrapolation_beats_the_finest_lattice_on_a_dielectric():
    mie = ls.solve(sim(), "mie", "cpu")
    finest = ls.solve(sim(npd=24), "dda", "cpu")
    r = ls.solve(sim(options={"extrapolate": [12, 16, 24]}), "dda", "cpu")
    err = abs(r.c_ext_nm2[0] / mie.c_ext_nm2[0] - 1)
    assert err < 3e-4 and err < abs(finest.c_ext_nm2[0] / mie.c_ext_nm2[0] - 1)
    assert any("extrapolated" in w for w in r.warnings)  # says what it did
    assert set(r.provenance["unextrapolated"]) == {"12", "16", "24"}  # and keeps each rung
    assert r.provenance["unextrapolated"]["24"]["c_ext_nm2"][0] == pytest.approx(finest.c_ext_nm2[0], rel=1e-9)
