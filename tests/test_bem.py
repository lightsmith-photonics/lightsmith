"""Native BEM (PMCHWT) backend against Mie, the quasistatic limit and DDA.

Measured accuracy behind the tolerances (Au/Ag Johnson-Christy spheres in water,
default volume correction): C_ext and C_abs within ~0.05 % of Mie at a 12 nm mesh
on a 40 nm sphere, on and off the plasmon resonance. Tolerances below leave a
margin of ~5-10x over those numbers.
"""

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends._bem.mesh import build_mesh, enclosed_volume
from lightsmith.backends._bem.singular import static_integrals
from lightsmith.spec import FarFieldRequest, LayeredSphere, NearFieldRequest, Outputs, Shell

AU = ls.LibraryMaterial(name="Au", dataset="Johnson-Christy")
AG = ls.LibraryMaterial(name="Ag", dataset="Johnson-Christy")
DIEL = ls.ConstantIndex(n_real=1.5, n_imag=0.02)
DRUDE = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)


def sim(shape=None, mat=AU, wl=(520.0,), h=12.0, medium=1.33, particles=None, **kw):
    opts = {"mesh_size_nm": h, **kw.pop("options", {})}
    particles = particles or [ls.Particle(shape=shape or ls.Sphere(radius_nm=40), material=mat)]
    return ls.Simulation(particles=particles, medium_index=medium, wavelengths=list(wl), options={"bem": opts}, **kw)


def rel_err(r, ref, q):
    return np.abs(np.asarray(getattr(r, q)) / np.asarray(getattr(ref, q)) - 1)


def mie(s):
    return ls.solve(s, "lightsmith-mie", "cpu")


# --- building blocks --------------------------------------------------------


def test_static_integrals_match_brute_force():
    tri = np.array([[0.0, 0, 0], [2.0, 0.3, 0.1], [0.4, 1.7, -0.2]])
    n = 600
    u, v = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="ij")
    m = u + v < 1
    pts = tri[0] + u[m][:, None] * (tri[1] - tri[0]) + v[m][:, None] * (tri[2] - tri[0])
    w = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / m.sum()
    for r in (np.array([0.8, 0.6, 0.5]), np.array([3.0, 2.0, -1.0])):
        d = r - pts
        rr = np.linalg.norm(d, axis=1)
        i0, _, grad = static_integrals(r, tri)
        assert i0 == pytest.approx(np.sum(w / rr), rel=2e-3)
        np.testing.assert_allclose(grad, -np.sum(w * d / rr[:, None] ** 3, axis=0), rtol=0, atol=3e-3 * np.abs(grad).max())


@pytest.mark.parametrize("shape", [ls.Sphere(radius_nm=30), ls.Spheroid(semi_axis_xy_nm=15, semi_axis_z_nm=45),
                                   ls.Cylinder(radius_nm=20, length_nm=50),
                                   LayeredSphere(core_radius_nm=20, shells=[Shell(thickness_nm=5, material=DIEL)])])
def test_meshes_are_closed_outward_and_volume_corrected(shape):
    mesh = build_mesh(sim(shape, DIEL, h=5.0), 5.0)
    assert mesh.n_edges == 3 * len(mesh.tris) // 2  # closed manifold
    radii = getattr(shape, "radii_nm", None) or [None]
    for surf, r in zip(mesh.surfaces, radii):
        vol = enclosed_volume(surf.verts, surf.tris)
        true = 4 / 3 * np.pi * r**3 if r else shape.volume()
        assert vol == pytest.approx(true, rel=1e-9)  # positive: outward oriented


# --- spheres against Mie ----------------------------------------------------


def test_dielectric_sphere_matches_mie():
    s = sim(mat=DIEL, wl=(450.0, 600.0))
    r, m = ls.solve(s, "bem", "cpu"), mie(s)
    for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        assert rel_err(r, m, q).max() < 2e-3, q


@pytest.mark.parametrize("mat,wl", [(AU, (520.0, 700.0)), (AG, (400.0, 600.0))], ids=["Au", "Ag"])
def test_metal_sphere_matches_mie_on_and_off_resonance(mat, wl):
    # 520 nm (Au) and 400 nm (Ag) are on the plasmon resonance in water; 700/600 nm are off it,
    # where volume methods overestimate the absorption most.
    s = sim(mat=mat, wl=wl)
    r, m = ls.solve(s, "bem", "cpu"), mie(s)
    assert rel_err(r, m, "c_ext_nm2").max() < 3e-3
    assert rel_err(r, m, "c_abs_nm2").max() < 3e-3
    assert not r.warnings


def test_mesh_convergence_without_volume_correction():
    # The raw inscribed polyhedron converges as h^2.
    errs = []
    for h in (14.0, 9.0):
        s = sim(wl=(650.0,), h=h, options={"volume_correction": False})
        errs.append(rel_err(ls.solve(s, "bem", "cpu"), mie(s), "c_abs_nm2")[0])
    assert errs[1] < 0.7 * errs[0]
    assert errs[1] < 0.03


def test_layered_sphere_matches_mie():
    p = ls.Particle(shape=LayeredSphere(core_radius_nm=30, shells=[Shell(thickness_nm=10, material=ls.ConstantIndex(
        n_real=1.45))]), material=AU)
    s = sim(particles=[p], wl=(550.0, 700.0), h=12.0)
    r, m = ls.solve(s, "bem", "cpu"), mie(s)
    assert rel_err(r, m, "c_ext_nm2").max() < 5e-3
    assert rel_err(r, m, "c_abs_nm2").max() < 5e-3


def test_far_and_near_field_match_mie():
    s = sim(ls.Sphere(radius_nm=30), wl=(530.0,), h=8.0,
            outputs=Outputs(far_field=FarFieldRequest(theta_deg=[0, 60, 90, 120, 180], wavelengths_nm=[530.0]),
                            near_field=NearFieldRequest(half_width_nm=60, points=9, wavelengths_nm=[530.0])))
    r, m = ls.solve(s, "bem", "cpu"), mie(s)
    np.testing.assert_allclose(r.far_field[0].dcsca_domega_nm2, m.far_field[0].dcsca_domega_nm2, rtol=2e-3)
    u = np.asarray(r.near_field[0].axis1_nm)
    rho = np.hypot(*np.meshgrid(u, u, indexing="ij"))
    away = (rho > 36) | (rho < 24)  # outside and inside, at least ~one triangle from the surface
    eb, em = np.asarray(r.near_field[0].enhancement), np.asarray(m.near_field[0].enhancement)
    np.testing.assert_allclose(eb[away], em[away], rtol=0.03)


def test_far_field_integrates_to_c_sca():
    th = np.linspace(0, 180, 91)
    s = sim(mat=DIEL, wl=(500.0,), outputs=Outputs(far_field=FarFieldRequest(
        theta_deg=th.tolist(), phi_deg=list(range(0, 360, 15)), wavelengths_nm=[500.0])))
    r = ls.solve(s, "bem", "cpu")
    f = np.asarray(r.far_field[0].dcsca_domega_nm2).mean(axis=1)
    assert np.trapezoid(2 * np.pi * f * np.sin(np.radians(th)), np.radians(th)) == pytest.approx(r.c_sca_nm2[0], rel=1e-3)


def test_orientation_average_of_sphere():
    a = ls.solve(sim(mat=DIEL, wl=(500.0,)), "bem", "cpu")
    b = ls.solve(sim(mat=DIEL, wl=(500.0,), orientation_average=True, options={"orientations": 3}), "bem", "cpu")
    np.testing.assert_allclose(b.c_ext_nm2, a.c_ext_nm2, rtol=2e-3)


def test_gmres_matches_direct():
    a = ls.solve(sim(), "bem", "cpu")
    b = ls.solve(sim(options={"solver": "gmres", "tol": 1e-10}), "bem", "cpu")
    np.testing.assert_allclose(b.c_ext_nm2, a.c_ext_nm2, rtol=1e-7)
    assert b.provenance["iterations"][0] > 0


# --- non-spherical particles ------------------------------------------------


def test_spheroid_sphere_limit_matches_mie():
    s = sim(ls.Spheroid(semi_axis_xy_nm=40, semi_axis_z_nm=40), wl=(520.0, 700.0))
    r = ls.solve(s, "bem", "cpu")
    m = mie(sim(wl=(520.0, 700.0)))
    assert rel_err(r, m, "c_ext_nm2").max() < 3e-3


def test_spheroid_polarization_anisotropy():
    # The longitudinal plasmon of a prolate gold-like rod is red-shifted from the transverse one.
    rod = ls.Spheroid(semi_axis_xy_nm=8, semi_axis_z_nm=24)
    wl = np.array([520.0, 620.0, 720.0, 820.0])
    along = ls.solve(sim(rod, DRUDE, wl=wl, h=4.0, illumination=ls.PlaneWave(direction=(1, 0, 0),
                                                                              polarization=(0, 0, 1))), "bem", "cpu")
    across = ls.solve(sim(rod, DRUDE, wl=wl, h=4.0), "bem", "cpu")
    assert wl[np.argmax(along.c_abs_nm2)] > wl[np.argmax(across.c_abs_nm2)] + 80


@pytest.mark.parametrize("shape", [ls.Spheroid(semi_axis_xy_nm=20, semi_axis_z_nm=40),
                                   ls.Cylinder(radius_nm=20, length_nm=60)], ids=["spheroid", "cylinder"])
def test_dielectric_shapes_agree_with_dda(shape):
    # Measured at h = 4 nm: DDA approaches BEM from 0.7 % (npd 16) to 0.3 % (npd 48) for the spheroid,
    # and stays within 0.4 % for the cylinder.
    il = ls.PlaneWave(direction=(0, 1, 0), polarization=(0, 0, 1))
    s = sim(shape, DIEL, wl=(450.0,), h=5.0, illumination=il)
    r = ls.solve(s, "bem", "cpu")
    d = ls.solve(s.model_copy(update={"options": {"dda": {"dipoles_per_diameter": 24}}}), "dda", "cpu")
    assert rel_err(r, d, "c_ext_nm2")[0] < 0.03


def test_dimer_couples_and_is_symmetric():
    wl = np.linspace(500, 620, 4)

    def dimer(gap, mats=(AU, AU), wl=wl):
        xs = (-(20 + gap / 2), 20 + gap / 2)
        parts = [ls.Particle(shape=ls.Sphere(radius_nm=20), material=m, position_nm=(x, 0, 0)) for m, x in zip(mats, xs)]
        return sim(particles=parts, wl=wl, h=8.0)

    single = ls.solve(sim(ls.Sphere(radius_nm=20), wl=wl, h=8.0), "bem", "cpu")
    close = ls.solve(dimer(5.0), "bem", "cpu")
    # x-polarised along the dimer axis: the bonding plasmon is red-shifted and enhanced
    assert wl[np.argmax(close.c_ext_nm2)] > wl[np.argmax(single.c_ext_nm2)]
    assert max(close.c_ext_nm2) > 2 * max(single.c_ext_nm2)
    # far apart, the dimer extinguishes like two independent spheres (plus weak far-field coupling)
    far = ls.solve(dimer(1500.0, wl=wl[1:2]), "bem", "cpu")
    assert far.c_ext_nm2[0] == pytest.approx(2 * single.c_ext_nm2[1], rel=0.03)
    # mirror symmetry: swapping two different particles leaves the cross sections unchanged
    a = ls.solve(dimer(5.0, (AU, DRUDE), wl[:1]), "bem", "cpu")
    b = ls.solve(dimer(5.0, (DRUDE, AU), wl[:1]), "bem", "cpu")
    np.testing.assert_allclose(a.c_ext_nm2, b.c_ext_nm2, rtol=1e-6)


# --- interface ----------------------------------------------------------------


def test_estimate_reports_unknowns_and_memory():
    e = ls.estimate(sim(), "bem", "cpu")
    assert e["backend"] == "lightsmith-bem" and e["unknowns"] == 2 * e["edges"] > 0
    assert e["memory_bytes"] > 16 * e["unknowns"] ** 2 and e["fits"]


def test_auto_picks_bem_for_non_spherical_particles():
    assert ls.choose_method(sim(ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=30)))[0].method == "bem"
    assert ls.choose_method(sim(ls.Cylinder(radius_nm=10, length_nm=30)))[0].method == "bem"
    assert ls.choose_method(sim())[0].method == "mie"


def test_rejects_overlaps_and_substrates():
    parts = [ls.Particle(shape=ls.Sphere(radius_nm=20), material=AU, position_nm=(x, 0, 0)) for x in (0, 30)]
    ok, why = ls.get_backend("bem").supports(sim(particles=parts))
    assert not ok and "overlap" in why
    ok, why = ls.get_backend("bem").supports(sim(substrate=ls.Substrate(material=DIEL)))
    assert not ok and "substrate" in why


def _dimer(gap):
    return sim(particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=AU, position_nm=(s * (20 + gap / 2), 0, 0))
                          for s in (-1, 1)], h=4.0)


def test_nearly_touching_particles_are_checked_on_the_solve_mesh():
    """Volume correction pushes each faceted sphere ~0.1 nm outside the exact one (R = 20 nm, default mesh):
    a 0.1 nm gap passes the exact-shape check but the solve meshes would interpenetrate, so it is refused.
    Gaps below the mesh size are accepted with a warning."""
    from lightsmith.backends._bem.mesh import build_mesh

    bem = ls.get_backend("bem")
    ok, why = bem.supports(_dimer(0.1))
    assert not ok and "volume-corrected meshes overlap" in why
    assert bem.supports(_dimer(0.1).model_copy(update={"options": {"bem": {"mesh_size_nm": 4.0, "volume_correction": False}}}))[0]
    assert bem.supports(_dimer(0.5))[0]
    h = 4.0
    assert bem._gap_warnings(_dimer(2.0), build_mesh(_dimer(2.0), h), h)
    assert not bem._gap_warnings(_dimer(10.0), build_mesh(_dimer(10.0), h), h)


# --- GPU ----------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_matches_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    s = sim(wl=(520.0, 700.0))
    a, b = ls.solve(s, "bem", "cpu"), ls.solve(s, "bem", "cuda")
    np.testing.assert_allclose(b.c_ext_nm2, a.c_ext_nm2, rtol=1e-9)
    np.testing.assert_allclose(b.c_abs_nm2, a.c_abs_nm2, rtol=1e-9)
    c = ls.solve(sim(wl=(520.0, 700.0), options={"precision": "single"}), "bem", "cuda")
    np.testing.assert_allclose(c.c_ext_nm2, a.c_ext_nm2, rtol=2e-3)  # measured ~2e-4


# --- slow convergence checks -------------------------------------------------


@pytest.mark.slow
def test_gold_sphere_fine_mesh_converges_below_1e4():
    s = sim(wl=(520.0, 600.0, 700.0, 800.0), h=6.0)
    r, m = ls.solve(s, "bem", "auto"), mie(s)
    assert rel_err(r, m, "c_ext_nm2").max() < 1e-4 * 3
    assert rel_err(r, m, "c_abs_nm2").max() < 1e-4 * 3


@pytest.mark.slow
def test_gold_rod_converges_with_mesh():
    # No exact reference for a gold rod: self-convergence of the band-integrated longitudinal
    # absorption (pointwise values on a sharp resonance mostly measure small peak shifts).
    wl = np.arange(600.0, 841.0, 10.0)
    il = ls.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1))
    area = [np.trapezoid(ls.solve(sim(ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=30), wl=wl, h=h, illumination=il),
                                  "bem", "auto").c_abs_nm2, wl) for h in (4.0, 3.0, 2.0)]
    d0, d1 = abs(area[0] / area[2] - 1), abs(area[1] / area[2] - 1)
    assert d1 < d0 and d1 < 0.03


@pytest.mark.slow
def test_small_rod_approaches_quasistatic_limit():
    # 10 x 30 nm Drude rod, field along the long axis, where the Gans dipole with radiative damping is
    # accurate. Measured: peak at 676 nm for BEM (h = 1.5 nm) and quasistatics, band-integrated
    # C_abs within 1 % (h = 2 nm still puts the peak 4 nm to the blue: the faceted aspect ratio matters).
    wl = np.arange(660.0, 692.1, 4.0)
    s = sim(ls.Spheroid(semi_axis_xy_nm=5, semi_axis_z_nm=15), DRUDE, wl=wl, h=1.5,
            illumination=ls.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1)))
    r = ls.solve(s, "bem", "auto")
    q = ls.solve(s.model_copy(update={"options": {"quasistatic": {"corrections": "radiative"}}}), "quasistatic", "cpu")
    assert wl[np.argmax(r.c_abs_nm2)] == wl[np.argmax(q.c_abs_nm2)]
    assert np.trapezoid(r.c_abs_nm2, wl) == pytest.approx(np.trapezoid(q.c_abs_nm2, wl), rel=0.03)
