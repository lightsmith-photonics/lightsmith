"""FDTD for periodic structures and layer stacks, validated against TMM and RCWA.

Measured at normal incidence (RTX 5070, float32; CPU float64), max absolute differences in R/T:
- dielectric slab (n=2, 300 nm on glass), 400-800 nm, dz=10 nm: 0.006 vs TMM;
- Drude film (30 nm): 0.0025; Ag Johnson-Christy film (40 nm, via Drude-Lorentz fit): 0.004;
- lamellar grating (P=700, w=300, eps=6, h=250 nm): s 0.006 / p 0.003 except at the Rayleigh
  anomaly (lambda = P), where it is 0.012; first transmitted orders within 0.002;
- hole-array slab guided-mode resonance: 596.12 vs 596.03 nm (RCWA) at 30 cells/wavelength.

Oblique incidence (fixed-k_par Bloch runs, one per angle/wavelength/polarisation), max |dR|, |dT|:
- n=2 slab, 450-750 nm, 20 cells/wavelength: 30 deg s 0.0039 / p 0.0033; 60 deg s 0.0018 / p 0.0012.
  The error is the Yee grid's own (FDTD reproduces the discrete-Yee Fresnel coefficient of a bare
  interface to 1e-4) and falls 4x per halving of the cell (60 deg, 400 nm: 0.0095 -> 0.0024).
- Drude film (30 nm), complex fields through the ADE: 30 deg 0.0008, 60 deg 0.0006.
- lamellar grating vs RCWA (40 harmonics), 500-1200 nm away from Rayleigh anomalies: 30 deg s 0.0037 /
  p 0.0031; 60 deg s 0.0032 / p 0.0017; conical (theta 30, phi 30) s 0.0027 / p 0.0021; single
  orders within 0.007.
- square lattice of n=2 disks (P=400 nm) at theta 40, phi 20 vs RCWA (11 harmonics): 0.0015, orders
  within 0.004.
"""

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.device import _torch_cuda
from lightsmith.materials import LibraryMaterial
from lightsmith.problems import (Circle, Incidence, Layer, LayerStackProblem, PatternedLayer, PeriodicProblem,
                                  Rectangle, Structure2D)

CI, CP = ls.ConstantIndex, ls.ConstantPermittivity
needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
WL = ls.WavelengthRange(start_nm=400, stop_nm=800, num=9)


def stack(layer_material, thickness, pols=("s",), **fdtd):
    return LayerStackProblem(superstrate=CI(n_real=1.0), substrate=CI(n_real=1.5),
                             layers=[Layer(thickness_nm=thickness, material=layer_material)], wavelengths=WL,
                             polarizations=list(pols), options={"fdtd": fdtd})


def lamellar(wl, pols=("s", "p"), theta=0.0, phi=0.0, **fdtd):
    return PeriodicProblem(
        lattice_a1_nm=(700, 0), superstrate=CI(n_real=1.0), substrate=CI(n_real=1.5),
        layers=[PatternedLayer(thickness_nm=250, background=CP(eps_real=1.0),
                               shapes=[Structure2D(profile=Rectangle(size_nm=(300, 1.0)), material=CP(eps_real=6.0))])],
        wavelengths=list(wl), incidence=Incidence(theta_deg=theta, phi_deg=phi, polarizations=list(pols)),
        options={"fdtd": fdtd, "rcwa": {"harmonics": 30}})


def max_abs_diff(a, b):
    return max(v["max"] for v in a.difference(b).values())


def test_fdtd_layer_stack_matches_tmm_cpu():
    for prob, tol in ((stack(CI(n_real=2.0), 300), 0.01),
                      (stack(ls.Drude(eps_inf=5, wp_ev=9.0, gamma_ev=0.05), 30, pols=("s", "p")), 0.005)):
        res = ls.solve(prob, "fdtd", "cpu")
        ref = ls.solve(prob, "tmm", "cpu")
        assert res.kind == "spectrum" and res.provenance["precision"] == "double"
        assert max_abs_diff(res, ref) < tol
        assert res.provenance["real_cells"][:2] == [1, 1]
    assert np.allclose(res.r["s"], res.r["p"])  # normal incidence: s and p coincide (one run)


def test_fdtd_periodic_supports_estimate_and_kinds():
    b = ls.get_backend("lightsmith-fdtd")
    assert {"scattering", "periodic", "layer_stack"} <= set(b.problem_kinds)
    p = lamellar([600.0])
    assert b.supports(p)[0]
    assert b.supports(lamellar([600.0], theta=60, phi=30))[0]
    ok, why = b.supports(lamellar([600.0], theta=88))
    assert not ok and "grazing" in why and "rcwa" in why
    ok, why = b.supports(stack(CI(n_real=2.0), 100).model_copy(update={"angles_deg": [0, 89]}))
    assert not ok and "tmm" in why
    skew = p.model_copy(update={"lattice_a1_nm": (600, 100)})
    assert not b.supports(skew)[0]
    est = ls.estimate(p, method="fdtd", device="cpu")
    nx = est["real_cells"][0]
    assert est["spacing_nm"][0] * nx == pytest.approx(700.0)  # whole cells per period (non-cubic)
    assert est["runs"] == 2 and est["memory_bytes"] > 0 and "oblique_runs" not in est
    # oblique: one run per (angle, wavelength, polarisation), complex fields, longer pulses
    est_o = ls.estimate(lamellar([500.0, 600.0, 700.0], theta=60), method="fdtd", device="cpu")
    assert est_o["runs"] == est_o["oblique_runs"] == 6
    assert est_o["memory_bytes"] > est["memory_bytes"] and est_o["oblique_run_min_steps"] > 0
    est_s = ls.estimate(stack(CI(n_real=2.0), 100, pols=("s", "p")).model_copy(update={"angles_deg": [0, 30, 60]}),
                        method="fdtd", device="cpu")
    assert est_s["runs"] == 1 + 2 * 2 * 9  # normal incidence: one run serves s and p


def test_fdtd_oblique_layer_stack_matches_tmm_cpu():
    prob = LayerStackProblem(superstrate=CI(n_real=1.0), substrate=CI(n_real=1.5),
                             layers=[Layer(thickness_nm=300, material=CI(n_real=2.0))], wavelengths=[450.0, 750.0],
                             angles_deg=[0.0, 30.0, 60.0], polarizations=["s", "p"])
    res, ref = ls.solve(prob, "fdtd", "cpu"), ls.solve(prob, "tmm", "cpu")
    assert res.angles_deg == [0.0, 30.0, 60.0] and len(res.provenance["runs"]) == 1 + 2 * 2 * 2
    for q in "rt":
        for pol in "sp":
            d = np.abs(np.asarray(getattr(res, q)[pol]) - np.asarray(getattr(ref, q)[pol]))
            assert d.max() < 0.008, (q, pol, d)  # measured <= 0.0046 (0 deg), 0.0039 (30), 0.0018 (60)
    assert np.abs(np.asarray(res.a["s"])).max() < 2e-3 and not res.warnings
    assert not np.allclose(res.r["s"][2], res.r["p"][2])  # s and p differ off-normal
    assert "fixed-k_par" in res.provenance["notes"]


@pytest.mark.gpu
@needs_cuda
def test_fdtd_layer_stack_gpu_matches_cpu_and_metal_film():
    prob = stack(ls.Drude(eps_inf=5, wp_ev=9.0, gamma_ev=0.05), 30, run_time_fs=20)
    cpu, gpu = ls.solve(prob, "fdtd", "cpu"), ls.solve(prob, "fdtd", "gpu")
    assert gpu.provenance["framework"] == "torch"
    np.testing.assert_allclose(gpu.r["s"], cpu.r["s"], atol=2e-5)
    np.testing.assert_allclose(gpu.t["s"], cpu.t["s"], atol=2e-5)
    ag = stack(LibraryMaterial(name="Ag", dataset="Johnson-Christy"), 40)
    assert max_abs_diff(ls.solve(ag, "fdtd", "gpu"), ls.solve(ag, "tmm", "cpu")) < 0.01


@pytest.mark.gpu
@needs_cuda
def test_fdtd_oblique_drude_film_gpu_matches_cpu_and_tmm():
    prob = LayerStackProblem(superstrate=CI(n_real=1.0), substrate=CI(n_real=1.5),
                             layers=[Layer(thickness_nm=30, material=ls.Drude(eps_inf=5, wp_ev=9.0, gamma_ev=0.05))],
                             wavelengths=[450.0, 600.0, 750.0], angles_deg=[30.0, 60.0], polarizations=["s", "p"])
    gpu = ls.solve(prob, "fdtd", "gpu")
    assert gpu.provenance["framework"] == "torch" and gpu.provenance["precision"] == "single"
    assert max_abs_diff(gpu, ls.solve(prob, "tmm", "cpu")) < 0.003  # measured 0.0008 (30 deg), 0.0006 (60)
    fixed = prob.model_copy(update={"wavelengths": [600.0], "angles_deg": [45.0],
                                    "options": {"fdtd": {"run_time_fs": 20}}})
    cpu, gpu = ls.solve(fixed, "fdtd", "cpu"), ls.solve(fixed, "fdtd", "gpu")  # complex128 vs complex64
    for pol in "sp":
        np.testing.assert_allclose(gpu.r[pol], cpu.r[pol], atol=5e-5)
        np.testing.assert_allclose(gpu.t[pol], cpu.t[pol], atol=5e-5)


def assert_orders_close(res, ref, atol):
    fo, ro = {o.order: o for o in res.orders}, {o.order: o for o in ref.orders}
    assert set(fo) <= set(ro)
    for k in ro:  # every order RCWA finds carrying power is there
        if k not in fo:
            assert max(np.max(ro[k].r[p]) + np.max(ro[k].t[p]) for p in ro[k].r) < 1e-6, k
    for k in fo:
        for pol in fo[k].r:
            np.testing.assert_allclose(fo[k].t[pol], ro[k].t[pol], atol=atol)
            np.testing.assert_allclose(fo[k].r[pol], ro[k].r[pol], atol=atol)


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("theta, phi, wl", [
    (30.0, 0.0, [500.0, 800.0, 1200.0]),  # Rayleigh anomalies at 700, 1050 nm
    pytest.param(60.0, 0.0, [600.0, 1000.0, 1200.0], marks=pytest.mark.slow),  # 444, 1306 nm; narrow pulses
    (30.0, 30.0, [500.0, 800.0, 1200.0]),  # conical
])
def test_fdtd_oblique_grating_matches_rcwa(theta, phi, wl):
    prob = lamellar(wl, theta=theta, phi=phi)
    res, ref = ls.solve(prob, "fdtd", "gpu"), ls.solve(prob, "rcwa", "cpu")
    assert res.angles_deg == [theta]
    assert max_abs_diff(res, ref) < 0.008  # measured <= 0.0043
    assert_orders_close(res, ref, atol=0.012)  # measured <= 0.0071


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_oblique_2d_lattice_conical_matches_rcwa():
    prob = PeriodicProblem(
        lattice_a1_nm=(400, 0), lattice_a2_nm=(0, 400), superstrate=CI(n_real=1.0), substrate=CI(n_real=1.45),
        layers=[PatternedLayer(thickness_nm=150, background=CI(n_real=1.0),
                               shapes=[Structure2D(profile=Circle(radius_nm=120), material=CI(n_real=2.0))])],
        wavelengths=[500.0, 700.0, 900.0], incidence=Incidence(theta_deg=40, phi_deg=20),
        options={"rcwa": {"harmonics": 11}})
    res, ref = ls.solve(prob, "fdtd", "gpu"), ls.solve(prob, "rcwa", "cpu")
    assert max_abs_diff(res, ref) < 0.005  # measured 0.0015
    assert_orders_close(res, ref, atol=0.01)  # measured <= 0.0039 (orders (0,0), (-1,0), (0,-1), (-1,-1))


@pytest.mark.gpu
@needs_cuda
def test_fdtd_1d_grating_matches_rcwa():
    wl = [500.0, 600.0, 800.0, 900.0, 1000.0, 1200.0]  # avoids the Rayleigh anomaly at 700 nm
    prob = lamellar(wl)
    cmp = ls.compare(prob, ["fdtd", "rcwa"], device="gpu", reference="rcwa")
    diff = cmp["relative_difference"]["fdtd"]
    assert max(v["max"] for v in diff.values()) < 0.015
    res, ref = cmp["results"]["fdtd"], cmp["results"]["rcwa"]
    fo, ro = {o.order: o for o in res.orders}, {o.order: o for o in ref.orders}
    assert set(fo) == set(ro)
    for k in fo:
        for pol in "sp":
            np.testing.assert_allclose(fo[k].t[pol], ro[k].t[pol], atol=0.01)
            np.testing.assert_allclose(fo[k].r[pol], ro[k].r[pol], atol=0.01)


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_fdtd_hole_array_guided_mode_resonance_matches_rcwa():
    wl = np.linspace(588, 604, 33)
    prob = PeriodicProblem(
        lattice_a1_nm=(400, 0), lattice_a2_nm=(0, 400), superstrate=CI(n_real=1.0), substrate=CI(n_real=1.45),
        layers=[PatternedLayer(thickness_nm=150, background=CI(n_real=2.0),
                               shapes=[Structure2D(profile=Circle(radius_nm=100), material=CI(n_real=1.0))])],
        wavelengths=list(wl), incidence=Incidence(polarizations=["s"]),
        options={"fdtd": {"cells_per_wavelength": 30}, "rcwa": {"harmonics": 7}})
    res, ref = ls.solve(prob, "fdtd", "gpu"), ls.solve(prob, "rcwa", "cpu")

    def peak(r):
        i = int(np.argmax(r))
        y0, y1, y2 = r[i - 1:i + 2]
        return wl[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (wl[1] - wl[0])

    rf, rr = np.asarray(res.r["s"][0]), np.asarray(ref.r["s"][0])
    assert abs(peak(rf) - peak(rr)) < 0.5  # measured 0.09 nm
    assert rf.max() > 0.9
    assert abs(rf[0] - rr[0]) < 0.01 and abs(rf[-1] - rr[-1]) < 0.02  # off resonance


@needs_cuda
@pytest.mark.parametrize("substrate,tol", [(LibraryMaterial(name="Si"), 0.02), (LibraryMaterial(name="Au"), 0.02)])
def test_absorbing_substrate_transmission_is_taken_at_the_interface(substrate, tol):
    """T is the flux into the substrate at the interface, as in TMM. The monitor sits a few cells inside the
    substrate, so absorption over that distance is undone per order; a metal substrate must not report T = 0.
    Measured max |dT| vs TMM, 80 nm SiO2 on Si / Au, 420-700 nm: 0.013 / 0.008 (Lorentz-fit error ~2%).
    Before the fix: Si 0.071 low at 420 nm, Au T = 0 everywhere."""
    p = LayerStackProblem(superstrate=CI(n_real=1.0), layers=[Layer(thickness_nm=80, material=CI(n_real=1.46))],
                          substrate=substrate, wavelengths=ls.WavelengthRange(start_nm=420, stop_nm=700, num=5))
    a, b = ls.solve(p, "tmm"), ls.solve(p, "fdtd", device="gpu")
    np.testing.assert_allclose(b.t["s"][0], a.t["s"][0], atol=tol)
    np.testing.assert_allclose(b.r["s"][0], a.r["s"][0], atol=tol)
