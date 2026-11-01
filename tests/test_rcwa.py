import numpy as np
import pytest

import omniscatter as osc
from omniscatter.backends.rcwa import rcwa_tensors
from omniscatter.device import _torch_cuda
from omniscatter.problems import (Circle, Incidence, Layer, LayerStackProblem, PatternedLayer, PeriodicProblem, Polygon,
                                  Rectangle, Structure2D)

CI, CP = osc.ConstantIndex, osc.ConstantPermittivity


def lamellar(period, width, eps_ridge, eps_groove=1.0, h=300.0, wl=(1000.0,), sub=1.5, theta=0.0, phi=0.0,
             a2=None, ridge_len=1.0, **opts):
    ridge = CP(eps_real=complex(eps_ridge).real, eps_imag=complex(eps_ridge).imag)
    return PeriodicProblem(
        lattice_a1_nm=(period, 0), lattice_a2_nm=a2, superstrate=CI(n_real=1.0), substrate=CI(n_real=sub),
        layers=[PatternedLayer(thickness_nm=h, background=CP(eps_real=eps_groove),
                               shapes=[Structure2D(profile=Rectangle(size_nm=(width, ridge_len)), material=ridge)])],
        wavelengths=list(wl), incidence=Incidence(theta_deg=theta, phi_deg=phi), options={"rcwa": opts})


def hole_slab(m, wl, **opts):
    """Square array of air holes in a lossless n=2 slab on glass: a guided-mode resonance near 596 nm."""
    return PeriodicProblem(
        lattice_a1_nm=(400, 0), lattice_a2_nm=(0, 400), superstrate=CI(n_real=1.0), substrate=CI(n_real=1.45),
        layers=[PatternedLayer(thickness_nm=150, background=CI(n_real=2.0),
                               shapes=[Structure2D(profile=Circle(radius_nm=100), material=CI(n_real=1.0))])],
        wavelengths=list(wl), incidence=Incidence(polarizations=["s"]), options={"rcwa": {"harmonics": m, **opts}})


def total(res, q, pol):
    return np.asarray(getattr(res, q)[pol])


def test_energy_conservation_lossless_1d_conical():
    p = lamellar(700, 300, 6.0, h=250, wl=np.linspace(500, 1200, 8), theta=25, phi=35, harmonics=15)
    res = osc.solve(p, "rcwa", "cpu")
    for pol in "sp":
        np.testing.assert_allclose(total(res, "r", pol) + total(res, "t", pol), 1.0, atol=1e-12)
        orders_sum = sum(np.asarray(o.r[pol]) + np.asarray(o.t[pol]) for o in res.orders)
        np.testing.assert_allclose(orders_sum, 1.0, atol=1e-12)  # all power is in propagating orders
    assert len(res.orders) > 2 and not res.warnings


def test_energy_conservation_lossless_2d_hexagonal_lattice():
    a = 500.0
    p = PeriodicProblem(
        lattice_a1_nm=(a, 0), lattice_a2_nm=(a / 2, a * np.sqrt(3) / 2), superstrate=CI(n_real=1.0),
        substrate=CI(n_real=1.45),
        layers=[PatternedLayer(thickness_nm=200, background=CI(n_real=2.2), shapes=[
            Structure2D(profile=Polygon(vertices_nm=[(-120, -80), (130, -60), (0, 140)]), material=CI(n_real=1.0))])],
        wavelengths=[550.0, 900.0], incidence=Incidence(theta_deg=20, phi_deg=10),
        options={"rcwa": {"harmonics": 4}})
    lau = osc.solve(p.model_copy(update={"options": {"rcwa": {"harmonics": 4, "factorization": "laurent"}}}),
                    "rcwa", "cpu")
    nv = {m: osc.solve(p.model_copy(update={"options": {"rcwa": {"harmonics": m}}}), "rcwa", "cpu") for m in (3, 6)}
    for pol in "sp":
        # Laurent matrices are Hermitian for lossless media: exact energy balance.
        np.testing.assert_allclose(total(lau, "a", pol), 0.0, atol=1e-11)
        # The normal-vector factorization is not Hermitian; the small imbalance shrinks with truncation.
        err3, err6 = (np.abs(total(nv[m], "a", pol)).max() for m in (3, 6))
        assert err6 < 2e-3 and err6 < err3


def test_analytic_and_raster_fourier_agree():
    wl = [600.0]
    a = osc.solve(hole_slab(4, wl), "rcwa", "cpu")
    b = osc.solve(hole_slab(4, wl, fourier="raster", raster_points=512), "rcwa", "cpu")
    assert a.provenance["fourier"] == ["analytic"] and b.provenance["fourier"] == ["raster"]
    assert total(a, "r", "s")[0, 0] == pytest.approx(total(b, "r", "s")[0, 0], abs=2e-3)


def test_rytov_effective_medium_benchmark():
    """Published benchmark: Rytov's long-wavelength limit (Sov. Phys. JETP 2, 466 (1956)).

    A lamellar grating with period P << lambda is a uniaxial slab with eps_TE = f e1 + (1-f) e2 and
    1/eps_TM = f/e1 + (1-f)/e2. RCWA results at P = 10 and 20 nm are extrapolated to P -> 0
    (errors are O(P^2)) and compared with TMM on the effective slab. The TM case is where Li's
    inverse rule matters: the Laurent rule converges towards the wrong limit very slowly.
    Tolerance: 2e-5 (TE) and 5e-5 (TM) absolute in R (R ~ 0.14 and 0.016).
    """
    e1, f, h, lam = 12.0, 0.3, 300.0, 1000.0
    r = {P: osc.solve(lamellar(P, f * P, e1, h=h, wl=[lam], harmonics=15), "rcwa", "cpu") for P in (10.0, 20.0)}
    for pol, eps in (("s", f * e1 + (1 - f)), ("p", 1 / (f / e1 + (1 - f)))):
        slab = LayerStackProblem(superstrate=CI(n_real=1.0), substrate=CI(n_real=1.5),
                                 layers=[Layer(thickness_nm=h, material=CP(eps_real=eps))], wavelengths=[lam],
                                 polarizations=[pol])
        ref = osc.solve(slab, "tmm", "cpu").r[pol][0][0]
        r10, r20 = r[10.0].r[pol][0][0], r[20.0].r[pol][0][0]
        extrapolated = (4 * r10 - r20) / 3
        assert extrapolated == pytest.approx(ref, abs=2e-5 if pol == "s" else 5e-5)
        assert abs(r10 - ref) < abs(r20 - ref)  # approaches the limit as P -> 0


def test_li_factorization_converges_for_metallic_grating():
    # Silver-like lamellar grating, TM, period comparable to the wavelength.
    def r_tm(m, fact):
        p = lamellar(600, 300, complex(-20, 1.5), h=100, wl=[800.0], theta=10, harmonics=m, factorization=fact)
        return osc.solve(p, "rcwa", "cpu").r["p"][0][0]

    li = {m: r_tm(m, "li") for m in (10, 20, 40, 80)}
    errs = [abs(li[m] - li[80]) for m in (10, 20, 40)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 4e-3  # monotone convergence
    lau40 = r_tm(40, "laurent")
    assert abs(lau40 - li[80]) > 5 * errs[2]  # the naive rule is much further off


def test_crossed_lattice_of_full_width_ridges_reduces_to_1d_li():
    one = lamellar(600, 300, complex(-20, 1.5), h=100, wl=[800.0], theta=10, harmonics=12)
    two = lamellar(600, 300, complex(-20, 1.5), h=100, wl=[800.0], theta=10, a2=(0, 400), ridge_len=400.0,
                   harmonics=[12, 1])
    a, b = osc.solve(one, "rcwa", "cpu"), osc.solve(two, "rcwa", "cpu")
    for pol in "sp":
        assert total(b, "r", pol)[0, 0] == pytest.approx(total(a, "r", pol)[0, 0], abs=1e-10)


def test_guided_mode_resonance_in_hole_array_converges():
    wl = np.linspace(592, 600, 17)
    peaks = {}
    for m in (3, 5):
        r = total(osc.solve(hole_slab(m, wl), "rcwa", "cpu"), "r", "s")[0]
        i = int(np.argmax(r))
        assert 0 < i < len(wl) - 1 and r[i] > 0.9  # sharp, near-unity reflection peak
        y0, y1, y2 = r[i - 1:i + 2]
        peaks[m] = wl[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (wl[1] - wl[0])
        assert r[0] < 0.4 and r[-1] < 0.4  # off resonance
    assert abs(peaks[5] - peaks[3]) < 1.0  # resonance wavelength converged to < 1 nm


def test_grating_orders_follow_grating_equation():
    p = lamellar(1500, 600, 4.0, h=200, wl=[1000.0], theta=10, harmonics=12)
    res = osc.solve(p, "rcwa", "cpu")
    orders = sorted(o.order[0] for o in res.orders)
    # Propagating in the substrate (n=1.5): |sin(10 deg) + m/1.5| < 1.5.
    expect = [m for m in range(-12, 13) if abs(np.sin(np.radians(10)) + m * 1000 / 1500) < 1.5]
    assert orders == expect


def test_estimate_reports_sizes():
    est = osc.estimate(hole_slab(5, [600.0]), method="rcwa", device="cpu")
    assert est["harmonics"] == 121 and est["matrix_size"] == 242 and est["eigensolves"] == 1


def test_gradients_match_finite_differences():
    torch = pytest.importorskip("torch")
    p = PeriodicProblem(lattice_a1_nm=(500, 0), lattice_a2_nm=(0, 500), superstrate=CI(n_real=1.0),
                        substrate=CI(n_real=1.45), layers=[PatternedLayer(thickness_nm=150, background=CI(n_real=1.0))],
                        wavelengths=[650.0, 700.0], options={"rcwa": {"harmonics": 3}})
    yy, xx = np.meshgrid(np.arange(16) - 7.5, np.arange(16) - 7.5, indexing="ij")
    g0 = torch.tensor(np.where(xx**2 + yy**2 < 30, 4.0, 1.0) + 0j)  # C4-symmetric: degenerate modes

    def loss(g, d=150.0):
        o = rcwa_tensors(p, "torch-cpu", thicknesses=[d], eps_grids={0: g}, grid_normals={0: g0})
        return o["T"]["s"].sum() + o["R"]["p"].sum() + o["T_orders"]["s"][..., 0].sum()

    g = g0.clone().requires_grad_(True)
    d = torch.tensor(150.0, dtype=torch.float64, requires_grad=True)
    loss(g, d).backward()
    scale = float(g.grad.abs().max())
    h = 1e-5
    for i, j in [(7, 7), (3, 8), (0, 0), (5, 10)]:
        gp, gm = g0.clone(), g0.clone()
        gp[i, j] += h
        gm[i, j] -= h
        fd = float((loss(gp) - loss(gm)) / (2 * h))
        assert float(g.grad[i, j].real) == pytest.approx(fd, rel=1e-4, abs=2e-5 * scale)
    fd = float((loss(g0, 150.0 + 1e-4) - loss(g0, 150.0 - 1e-4)) / 2e-4)
    assert float(d.grad) == pytest.approx(fd, rel=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
def test_rcwa_gpu_matches_cpu():
    wl = np.linspace(590, 600, 5)
    cpu, gpu = osc.solve(hole_slab(4, wl), "rcwa", "cpu"), osc.solve(hole_slab(4, wl), "rcwa", "gpu")
    assert gpu.provenance["framework"] == "torch"
    np.testing.assert_allclose(total(gpu, "r", "s"), total(cpu, "r", "s"), atol=1e-9)
    one = lamellar(700, 300, complex(-20, 1.5), h=100, wl=[700.0, 800.0], theta=10, harmonics=20)
    a, b = osc.solve(one, "rcwa", "cpu"), osc.solve(one, "rcwa", "gpu")
    np.testing.assert_allclose(total(b, "r", "p"), total(a, "r", "p"), atol=1e-9)
