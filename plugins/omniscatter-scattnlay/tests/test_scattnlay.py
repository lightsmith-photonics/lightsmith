import numpy as np
import pytest

import omniscatter as osc
from omniscatter.backends import all_backends, get_backend
from omniscatter_scattnlay import ScattnlayBackend

AU = osc.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)
DIELECTRIC = osc.ConstantIndex(n_real=1.5, n_imag=0.02)
WL = osc.WavelengthRange(start_nm=400, stop_nm=800, num=41)


def sphere_sim(material, radius=40.0, medium=1.33, **kw):
    return osc.Simulation(particles=[osc.Particle(shape=osc.Sphere(radius_nm=radius), material=material)],
                          medium_index=medium, wavelengths=WL, **kw)


def layered_sim(core, shells, core_radius=30.0, medium=1.33, **kw):
    shape = osc.LayeredSphere(core_radius_nm=core_radius,
                              shells=[osc.Shell(thickness_nm=t, material=m) for t, m in shells])
    return osc.Simulation(particles=[osc.Particle(shape=shape, material=core)], medium_index=medium,
                          wavelengths=WL, **kw)


def test_entry_point_registers_backend():
    assert "scattnlay" in {b.name for b in all_backends()}
    b = get_backend("scattnlay")
    assert b is ScattnlayBackend and b.available()[0]
    info = b.info()
    assert info["method"] == "mie" and info["license"] == "GPL-3.0-or-later"


def test_supports():
    assert ScattnlayBackend.supports(sphere_sim(AU))[0]
    assert ScattnlayBackend.supports(layered_sim(AU, [(10.0, DIELECTRIC)]))[0]
    assert ScattnlayBackend.supports(sphere_sim(AU, orientation_average=True))[0]
    spheroid = osc.Simulation(particles=[osc.Particle(shape=osc.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=20),
                                                      material=AU)], wavelengths=[500.0])
    ok, why = ScattnlayBackend.supports(spheroid)
    assert not ok and "spheres" in why
    two = spheroid.model_copy(update={"particles": [sphere_sim(AU).particles[0]] * 2})
    assert not ScattnlayBackend.supports(two)[0]


@pytest.mark.parametrize("material", [AU, DIELECTRIC], ids=["drude-gold", "lossy-dielectric"])
def test_matches_miepython_for_plain_sphere(material):
    sim = sphere_sim(material)
    ref = osc.solve(sim, method="miepython")
    res = osc.solve(sim, method="scattnlay")
    assert res.backend == "scattnlay" and res.method == "mie"
    for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        np.testing.assert_allclose(getattr(res, q), getattr(ref, q), rtol=1e-9, err_msg=q)
    assert np.all(np.asarray(res.c_abs_nm2) > 0)  # sign convention: Im(n) > 0 absorbs
    assert res.provenance["scattnlay_version"] and res.provenance["layers"] == 1


def test_layered_sphere_reduces_to_homogeneous():
    same = layered_sim(AU, [(10.0, AU)], core_radius=30.0)
    ref = osc.solve(sphere_sim(AU, radius=40.0), method="miepython")
    res = osc.solve(same, method="scattnlay")
    np.testing.assert_allclose(res.c_ext_nm2, ref.c_ext_nm2, rtol=1e-9)
    assert res.equivalent_radius_nm == pytest.approx(40.0)


def test_core_shell_layer_order():
    # A vanishing core leaves the shell material: layers are passed core first.
    tiny_core = layered_sim(osc.ConstantIndex(n_real=3.0), [(40.0 - 1e-3, DIELECTRIC)], core_radius=1e-3)
    ref = osc.solve(sphere_sim(DIELECTRIC, radius=40.0), method="miepython")
    np.testing.assert_allclose(osc.solve(tiny_core, method="scattnlay").c_ext_nm2, ref.c_ext_nm2, rtol=1e-6)
    # A real silica-coated gold shell red-shifts the plasmon peak compared to bare gold.
    coated = osc.solve(layered_sim(AU, [(10.0, osc.ConstantIndex(n_real=1.45))]), method="scattnlay")
    bare = osc.solve(sphere_sim(AU, radius=30.0), method="scattnlay")
    assert coated.summary()["peak_ext_wavelength_nm"] > bare.summary()["peak_ext_wavelength_nm"]


def test_far_field_integrates_to_csca():
    theta = np.linspace(0, 180, 721)
    phi = np.linspace(0, 360, 145)[:-1]
    sim = sphere_sim(DIELECTRIC, radius=80.0, outputs=osc.Outputs(far_field=osc.FarFieldRequest(
        theta_deg=theta.tolist(), phi_deg=phi.tolist(), wavelengths_nm=[500.0])))
    res = osc.solve(sim.model_copy(update={"wavelengths": [500.0]}), method="scattnlay")
    d = np.asarray(res.far_field[0].dcsca_domega_nm2)  # [theta][phi]
    th = np.radians(theta)
    integral = np.trapezoid(d.mean(axis=1) * 2 * np.pi * np.sin(th), th)
    assert integral == pytest.approx(res.c_sca_nm2[0], rel=1e-4)
    # E-plane (phi=0) and H-plane (phi=90) agree in the forward direction for a sphere.
    assert d[0, 0] == pytest.approx(d[0, 36], rel=1e-12)


def test_options_are_validated():
    with pytest.raises(Exception):
        osc.solve(sphere_sim(AU, options={"scattnlay": {"bogus": 1}}), method="scattnlay")
    res = osc.solve(sphere_sim(AU, options={"scattnlay": {"multiprecision": True}}), method="scattnlay")
    assert res.provenance["precision"] in ("double", "multiple")
    # Method-level Mie options: n_max is honoured, keys of other Mie backends are ignored with a warning.
    opts, warnings = ScattnlayBackend.options(sphere_sim(AU, options={"mie": {"n_max": 30, "precision": "single"}}))
    assert opts.n_max == 30 and any("precision" in w for w in warnings)
