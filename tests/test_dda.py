import numpy as np
import pytest

import omniscatter as osc
from omniscatter.spec import FarFieldRequest, LayeredSphere, NearFieldRequest, Outputs, Shell

DIELECTRIC = osc.ConstantIndex(n_real=1.5, n_imag=0.02)


def sim(shape=None, mat=DIELECTRIC, npd=12, wl=(500.0,), **kw):
    opts = kw.pop("options", {})
    return osc.Simulation(
        particles=[osc.Particle(shape=shape or osc.Sphere(radius_nm=40), material=mat)],
        medium_index=1.33, wavelengths=list(wl),
        options={"dda": {"dipoles_per_diameter": npd, **opts}}, **kw)


def test_far_field_integrates_to_c_sca():
    th = np.linspace(0, 180, 181)
    s = sim(outputs=Outputs(far_field=FarFieldRequest(theta_deg=th.tolist(), phi_deg=list(range(0, 360, 10)),
                                                       wavelengths_nm=[500.0])))
    r = osc.solve(s, "dda", "cpu")
    f = np.asarray(r.far_field[0].dcsca_domega_nm2).mean(axis=1)  # phi-average (uniform periodic grid)
    integral = np.trapezoid(2 * np.pi * f * np.sin(np.radians(th)), np.radians(th))
    assert integral == pytest.approx(r.c_sca_nm2[0], rel=1e-3)


def test_near_field_tends_to_incident_and_masks_inside():
    s = sim(outputs=Outputs(near_field=NearFieldRequest(half_width_nm=400, points=21, wavelengths_nm=[500.0])))
    e = np.asarray(osc.solve(s, "dda", "cpu").near_field[0].enhancement)
    assert np.isnan(e[10, 10])
    assert e[0, 0] == pytest.approx(1.0, abs=0.02)


def test_shell_of_core_material_equals_sphere():
    a = osc.solve(sim(), "dda", "cpu")
    b = osc.solve(sim(shape=LayeredSphere(core_radius_nm=25, shells=[Shell(thickness_nm=15, material=DIELECTRIC)])),
                  "dda", "cpu")
    np.testing.assert_allclose(b.c_ext_nm2, a.c_ext_nm2, rtol=1e-10)


def test_orientation_average_of_sphere():
    a = osc.solve(sim(), "dda", "cpu")
    b = osc.solve(sim(orientation_average=True, options={"orientations": 4}), "dda", "cpu")
    np.testing.assert_allclose(b.c_ext_nm2, a.c_ext_nm2, rtol=1e-3)


def test_spheroid_polarization_anisotropy():
    rod = osc.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=30)
    gold_like = osc.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)
    wl = np.linspace(500, 900, 9).tolist()
    along = osc.solve(sim(rod, gold_like, wl=wl, illumination=osc.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1))),
                      "dda", "cpu")
    across = osc.solve(sim(rod, gold_like, wl=wl), "dda", "cpu")
    # The longitudinal plasmon of a prolate rod is red-shifted relative to the transverse one.
    assert wl[int(np.argmax(along.c_ext_nm2))] > wl[int(np.argmax(across.c_ext_nm2))]


def test_dimer_runs_and_reports_provenance():
    p = lambda x: osc.Particle(shape=osc.Sphere(radius_nm=20), material=DIELECTRIC, position_nm=(x, 0, 0))  # noqa: E731
    s = osc.Simulation(particles=[p(-25), p(25)], wavelengths=[500.0], options={"dda": {"dipoles_per_diameter": 10}})
    assert osc.choose_method(s)[0].method == "dda"
    r = osc.solve(s, "dda", "cpu")
    assert r.provenance["dipoles"][0] > 0 and r.c_ext_nm2[0] > 0
