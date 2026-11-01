"""treams T-matrix backend, validated against Mie, analytic limits and its own invariants."""

import numpy as np
import pytest
from pydantic import ValidationError

import lightsmith as ls
from lightsmith.spec import FarFieldRequest, LayeredSphere, NearFieldRequest, Outputs, Shell, Substrate

pytest.importorskip("treams")
# treams 0.4.7 calls scipy.special.sph_harm, deprecated in SciPy 1.15.
pytestmark = pytest.mark.filterwarnings("ignore:`scipy.special.sph_harm` is deprecated:DeprecationWarning")

from lightsmith.backends.tmatrix_treams import TreamsOptions, _Plan, _Solver  # noqa: E402

DIELECTRIC = ls.ConstantIndex(n_real=1.5, n_imag=0.02)
GOLD = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)  # gold-like Drude metal
WL = [450.0, 520.0, 600.0, 700.0]


def particle(material, radius=40.0, pos=(0.0, 0.0, 0.0)):
    return ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material, position_nm=pos)


def sim_of(particles, wavelengths=WL, **kw):
    return ls.Simulation(particles=particles, medium_index=1.33, wavelengths=list(wavelengths), **kw)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def sphere_quadrature(n_theta=40, n_phi=36):
    """Angles and weights integrating over the sphere: Gauss-Legendre in cos(theta), uniform in phi."""
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.degrees(np.arccos(x))
    phi = np.arange(n_phi) * 360.0 / n_phi
    return theta.tolist(), phi.tolist(), w * (2 * np.pi / n_phi)


@pytest.mark.parametrize("material", [DIELECTRIC, GOLD], ids=["dielectric", "drude"])
def test_single_sphere_matches_mie(material):
    # Off-origin and oblique incidence must not matter for a single sphere.
    illum = ls.PlaneWave(direction=(1.0, 1.0, 1.0), polarization=(1.0, -1.0, 0.0))
    sim = sim_of([particle(material, pos=(10.0, -5.0, 30.0))], illumination=illum)
    res = ls.solve(sim, "treams")
    ref = ls.solve(sim, "miepython")
    for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        assert rel(getattr(res, q), getattr(ref, q)) < 1e-6  # measured ~1e-12
    assert res.provenance["treams_version"] and res.provenance["lmax"][0] >= 3
    assert res.provenance["runtime_s"] > 0


def test_layered_sphere_with_core_material_shell_equals_sphere():
    core = ls.Particle(shape=LayeredSphere(core_radius_nm=25.0, shells=[Shell(thickness_nm=15.0, material=GOLD)]),
                        material=GOLD)
    a = ls.solve(sim_of([core]), "treams")
    b = ls.solve(sim_of([particle(GOLD, radius=40.0)]), "miepython")
    assert rel(a.c_ext_nm2, b.c_ext_nm2) < 1e-6
    assert rel(a.c_sca_nm2, b.c_sca_nm2) < 1e-6


def test_small_core_shell_matches_quasistatic_polarizability():
    # Bohren & Huffman eq. 5.36 for a coated sphere, relative permittivities (host 1.33^2).
    r1, r2, lam, nm = 3.0, 4.5, 500.0, 1.33
    shell = ls.ConstantPermittivity(eps_real=-2.0, eps_imag=1.5)
    cs = ls.Particle(shape=LayeredSphere(core_radius_nm=r1, shells=[Shell(thickness_nm=r2 - r1, material=shell)]),
                      material=DIELECTRIC)
    res = ls.solve(sim_of([cs], wavelengths=[lam]), "treams")
    em = nm**2
    e1, e2 = complex(DIELECTRIC.eps(np.array([lam]))[0]), complex(shell.eps(np.array([lam]))[0])
    f = (r1 / r2) ** 3
    alpha = r2**3 * ((e2 - em) * (e1 + 2 * e2) + f * (e1 - e2) * (em + 2 * e2)) / (
        (e2 + 2 * em) * (e1 + 2 * e2) + f * (2 * e2 - 2 * em) * (e1 - e2))
    k = 2 * np.pi * nm / lam
    assert res.c_abs_nm2[0] == pytest.approx(4 * np.pi * k * alpha.imag, rel=1e-2)


def test_distant_pair_doubles_single_sphere():
    # 10 diameters apart, side by side and along k. Extinction and absorption: near-field
    # coupling is gone, measured <= 0.6%, tolerance 1%. Scattering keeps the coherent
    # interference of the two scattered waves, which decays only like 1/(kd): measured
    # 3.2% (side by side) and 3.8% (along k), tolerance 5%.
    single = ls.solve(sim_of([particle(DIELECTRIC)]), "treams")
    for axis in ((800.0, 0.0, 0.0), (0.0, 0.0, 800.0)):
        pair = ls.solve(sim_of([particle(DIELECTRIC), particle(DIELECTRIC, pos=axis)]), "treams")
        for q, tol in (("c_ext_nm2", 1e-2), ("c_abs_nm2", 1e-2), ("c_sca_nm2", 5e-2)):
            assert rel(getattr(pair, q), 2 * np.asarray(getattr(single, q))) < tol
        assert pair.equivalent_radius_nm == pytest.approx(40.0 * 2 ** (1 / 3))
        assert not pair.warnings


def test_cluster_matches_treams_reference_path():
    # Our coupling (half the translations, absorption-based C_sca) vs treams' own
    # TMatrix.cluster(...).interaction.solve() and TMatrix.xs, for unequal orders too.
    import treams

    lam = 520.0
    parts = [particle(GOLD, 20.0, (-22.0, 0.0, 0.0)), particle(DIELECTRIC, 30.0, (10.0, 55.0, 10.0)),
             particle(GOLD, 15.0, (0.0, -10.0, 45.0))]
    sim = sim_of(parts, wavelengths=[lam], illumination=ls.PlaneWave(direction=(0.0, 1.0, 1.0)))
    solver = _Solver(sim, _Plan(sim, TreamsOptions()))
    tm, _, _ = solver.tmatrix(lam)
    k0, emb = 2 * np.pi / lam, treams.Material(1.33**2)
    singles = [treams.TMatrix.sphere(lm, k0, p.shape.radius_nm, [treams.Material(p.material.eps(np.array([lam]))[0]),
                                                                 emb], poltype="parity")
               for p, lm in zip(parts, solver.plan.lmax)]
    ref = treams.TMatrix.cluster(singles, solver.plan.positions).interaction.solve()
    assert np.max(np.abs(np.asarray(tm) - np.asarray(ref))) < 1e-10 * np.max(np.abs(np.asarray(ref)))
    sca, ext = ref.xs(solver.incident(ref))
    res = ls.solve(sim, "treams")
    assert res.c_ext_nm2[0] == pytest.approx(float(ext), rel=1e-9)
    assert res.c_sca_nm2[0] == pytest.approx(float(sca), rel=1e-9)


def test_gold_dimer_redshifts_for_axial_polarization():
    wl = np.arange(480.0, 641.0, 10.0).tolist()
    r, gap = 20.0, 5.0
    pair = [particle(GOLD, r, (-(r + gap / 2), 0.0, 0.0)), particle(GOLD, r, (r + gap / 2, 0.0, 0.0))]
    peak = lambda res: res.wavelengths_nm[int(np.argmax(res.c_ext_nm2))]  # noqa: E731
    mono = ls.solve(sim_of([particle(GOLD, r)], wavelengths=wl), "treams")
    along = ls.solve(sim_of(pair, wavelengths=wl, illumination=ls.PlaneWave(polarization=(1.0, 0.0, 0.0))),
                      "treams")
    across = ls.solve(sim_of(pair, wavelengths=wl, illumination=ls.PlaneWave(polarization=(0.0, 1.0, 0.0))),
                       "treams")
    assert peak(along) >= peak(mono) + 20  # strong bonding-mode red shift
    assert peak(across) <= peak(mono)  # transverse coupling blue-shifts slightly
    assert along.provenance["extra_orders_close"] == [7, 7]  # gap = R/4
    assert along.provenance["convergence"]["rel_change_c_ext"] < 1e-3
    assert not along.warnings


@pytest.mark.parametrize("material", [DIELECTRIC, GOLD], ids=["dielectric", "drude"])
def test_orientation_average_of_sphere_equals_fixed(material):
    fixed = ls.solve(sim_of([particle(material, pos=(5.0, 0.0, 0.0))]), "treams")
    avg = ls.solve(sim_of([particle(material, pos=(5.0, 0.0, 0.0))], orientation_average=True), "treams")
    assert rel(avg.c_ext_nm2, fixed.c_ext_nm2) < 1e-10
    assert rel(avg.c_sca_nm2, fixed.c_sca_nm2) < 1e-10


def test_orientation_average_of_dimer_matches_quadrature():
    # Analytic average (trace of the global T-matrix) vs explicit average over incidences:
    # Gauss-Legendre in cos(theta) x uniform phi x two polarizations.
    lam = 520.0
    sim = sim_of([particle(GOLD, 20.0, (-30.0, 0.0, 0.0)), particle(DIELECTRIC, 30.0, (25.0, 40.0, 10.0))],
                 wavelengths=[lam], orientation_average=True)
    avg = ls.solve(sim, "treams")
    solver = _Solver(sim, _Plan(sim, TreamsOptions()))
    x, w = np.polynomial.legendre.leggauss(10)
    phis = np.arange(20) * 2 * np.pi / 20
    ext = sca = 0.0
    for ct, wt in zip(x, w):
        st = np.sqrt(1 - ct**2)
        for ph in phis:
            k = np.array([st * np.cos(ph), st * np.sin(ph), ct])
            e1 = np.array([ct * np.cos(ph), ct * np.sin(ph), -st])
            for e in (e1, np.cross(k, e1)):
                solver.khat, solver.ehat = k, e
                x_, s = solver.cross_sections(lam, orientation_average=False)  # reuses the T-matrix
                ext += wt * x_ / (2 * 20 * 2)
                sca += wt * s / (2 * 20 * 2)
    assert avg.c_ext_nm2[0] == pytest.approx(ext, rel=1e-6)
    assert avg.c_sca_nm2[0] == pytest.approx(sca, rel=1e-6)
    assert "lmax_global" in avg.provenance


@pytest.mark.parametrize("cluster", [False, True], ids=["sphere", "dimer"])
def test_far_field_integrates_to_c_sca(cluster):
    theta, phi, w = sphere_quadrature()
    parts = ([particle(GOLD, 20.0, (-30.0, 0.0, 0.0)), particle(GOLD, 20.0, (30.0, 0.0, 0.0))] if cluster
             else [particle(GOLD, 40.0, (0.0, 10.0, 0.0))])
    illum = ls.PlaneWave(direction=(0.0, 1.0, 1.0), polarization=(1.0, 0.0, 0.0))
    sim = sim_of(parts, wavelengths=[520.0, 600.0], illumination=illum,
                 outputs=Outputs(far_field=FarFieldRequest(theta_deg=theta, phi_deg=phi, wavelengths_nm=[520.0])))
    res = ls.solve(sim, "treams")
    ff = res.far_field[0]
    d = np.asarray(ff.dcsca_domega_nm2)
    assert d.shape == (len(theta), len(phi))
    assert float(np.sum(d * w[:, None])) == pytest.approx(res.c_sca_nm2[0], rel=1e-6)


def test_far_field_angle_convention():
    # Small sphere = electric dipole along E: no scattering at theta=90 in the E-plane (phi=0),
    # isotropic in the H-plane (phi=90). Incidence is along -x with E along z.
    illum = ls.PlaneWave(direction=(-1.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0))
    sim = sim_of([particle(DIELECTRIC, 5.0)], wavelengths=[600.0], illumination=illum,
                 outputs=Outputs(far_field=FarFieldRequest(theta_deg=[0.0, 90.0, 180.0], phi_deg=[0.0, 90.0],
                                                           wavelengths_nm=[600.0])))
    d = np.asarray(ls.solve(sim, "treams").far_field[0].dcsca_domega_nm2)
    assert d[1, 0] < 1e-3 * d[1, 1]
    assert d[0, 1] == pytest.approx(d[1, 1], rel=1e-2)
    assert d[2, 1] == pytest.approx(d[1, 1], rel=1e-2)


def test_near_field_quasistatic_and_nan_inside():
    # Small sphere: on the polarization axis E = E0 (1 + 2 alpha / d^3), alpha = r^3 (e - em)/(e + 2 em).
    r, lam, nm = 5.0, 600.0, 1.33
    req = NearFieldRequest(plane="xz", half_width_nm=12.0, points=25, wavelengths_nm=[lam])
    sim = sim_of([particle(DIELECTRIC, r, pos=(0.0, 0.0, 0.0))], wavelengths=[lam], outputs=Outputs(near_field=req))
    nf = ls.solve(sim, "treams").near_field[0]
    enh = np.asarray(nf.enhancement)
    u = np.asarray(nf.axis1_nm)
    centre = len(u) // 2
    assert np.isnan(enh[centre, centre])  # inside the sphere
    assert np.isfinite(enh[0, 0])
    e = complex(DIELECTRIC.eps(np.array([lam]))[0])
    alpha = r**3 * (e - nm**2) / (e + 2 * nm**2)
    i = len(u) - 1  # x = 12 nm on the polarization axis (z = 0)
    expected = abs(1 + 2 * alpha / u[i] ** 3) ** 2
    assert enh[i, centre] == pytest.approx(expected, rel=1e-2)
    assert np.asarray(nf.e_real).shape == (3, len(u), len(u))


def test_near_field_phase_references_lab_origin():
    # A shifted particle must not shift the incident phase: far from the sphere, E ~ e^{ikz} x.
    lam = 600.0
    req = NearFieldRequest(plane="xz", half_width_nm=200.0, points=5, wavelengths_nm=[lam])
    sim = sim_of([particle(DIELECTRIC, 2.0, pos=(150.0, 0.0, 150.0))], wavelengths=[lam],
                 outputs=Outputs(near_field=req))
    nf = ls.solve(sim, "treams").near_field[0]
    z = np.asarray(nf.axis2_nm)
    k = 2 * np.pi * 1.33 / lam
    ex = np.asarray(nf.e_real)[0, 0] + 1j * np.asarray(nf.e_imag)[0, 0]  # x = -200 nm, all z
    np.testing.assert_allclose(ex, np.exp(1j * k * z), atol=1e-3)


def test_supports_and_rejections():
    from lightsmith.backends import get_backend

    b = get_backend("treams")
    assert b.method == "tmatrix" and b.license == "MIT" and b.available()[0]
    spheroid = sim_of([ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=20, semi_axis_z_nm=40), material=GOLD)])
    ok, why = b.supports(spheroid)
    assert not ok and "dda" in why
    ok, why = b.supports(sim_of([particle(GOLD), particle(GOLD, pos=(50.0, 0.0, 0.0))]))
    assert not ok and "overlap" in why
    ok, why = b.supports(sim_of([particle(GOLD)], substrate=Substrate(material=DIELECTRIC, interface_z_nm=-50)))
    assert not ok and "substrate" in why
    oa_ff = sim_of([particle(GOLD)], orientation_average=True,
                   outputs=Outputs(far_field=FarFieldRequest(theta_deg=[0.0], wavelengths_nm=[500.0])))
    assert not b.supports(oa_ff)[0]
    assert ls.choose_method(sim_of([particle(GOLD), particle(GOLD, pos=(100.0, 0.0, 0.0))]))[0].method == "tmatrix"


def test_options_and_estimate():
    sim = sim_of([particle(GOLD), particle(GOLD, pos=(100.0, 0.0, 0.0))], options={"tmatrix": {"lmax": 7}})
    est = ls.estimate(sim, method="treams")
    assert est["lmax"] == [7, 7] and est["modes"] == 2 * 2 * 7 * 9 and est["fits"]
    res = ls.solve(sim.model_copy(update={"wavelengths": [520.0]}), "treams")
    assert res.provenance["lmax"] == [7, 7]
    with pytest.raises(ValidationError):
        ls.solve(sim_of([particle(GOLD)], options={"treams": {"lmaxx": 3}}), "treams")
    with pytest.raises(ValueError, match="max_modes"):
        ls.solve(sim_of([particle(GOLD)], options={"treams": {"max_modes": 10}}), "treams")


@pytest.mark.slow
def test_tetrahedral_cluster_converges_and_conserves():
    # Four touching-ish high-index spheres (the treams docs example, in a 1.33 host).
    mat = ls.ConstantPermittivity(eps_real=16.0, eps_imag=0.5)
    pos = (220 / np.sqrt(24)) * np.array([[np.sqrt(8), 0, -1], [-np.sqrt(2), np.sqrt(6), -1],
                                          [-np.sqrt(2), -np.sqrt(6), -1], [0, 0, 3]])
    parts = [particle(mat, r, tuple(p)) for r, p in zip([110.0, 90.0, 80.0, 75.0], pos)]
    theta, phi, w = sphere_quadrature(60, 48)
    sim = sim_of(parts, wavelengths=[1000.0],
                 outputs=Outputs(far_field=FarFieldRequest(theta_deg=theta, phi_deg=phi, wavelengths_nm=[1000.0])))
    res = ls.solve(sim, "treams")
    assert not res.warnings
    d = np.asarray(res.far_field[0].dcsca_domega_nm2)
    assert float(np.sum(d * w[:, None])) == pytest.approx(res.c_sca_nm2[0], rel=1e-5)
    finer = ls.solve(sim.model_copy(update={"outputs": Outputs(), "options": {"tmatrix": {"extra_orders": 4}}}),
                      "treams")
    assert rel(res.c_ext_nm2, finer.c_ext_nm2) < 1e-3
