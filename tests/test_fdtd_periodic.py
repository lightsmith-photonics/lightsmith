"""FDTD for periodic structures and layer stacks (normal incidence), validated against TMM and RCWA.

Measured (RTX 5070, float32; CPU float64), max absolute differences in R/T:
- dielectric slab (n=2, 300 nm on glass), 400-800 nm, dz=10 nm: 0.006 vs TMM;
- Drude film (30 nm): 0.0025; Ag Johnson-Christy film (40 nm, via Drude-Lorentz fit): 0.004;
- lamellar grating (P=700, w=300, eps=6, h=250 nm): s 0.006 / p 0.003 except at the Rayleigh
  anomaly (lambda = P), where it is 0.012; first transmitted orders within 0.002;
- hole-array slab guided-mode resonance: 596.12 vs 596.03 nm (RCWA) at 30 cells/wavelength.
"""

import numpy as np
import pytest

import omniscatter as osc
from omniscatter.device import _torch_cuda
from omniscatter.materials import LibraryMaterial
from omniscatter.problems import (Circle, Incidence, Layer, LayerStackProblem, PatternedLayer, PeriodicProblem,
                                  Rectangle, Structure2D)

CI, CP = osc.ConstantIndex, osc.ConstantPermittivity
needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
WL = osc.WavelengthRange(start_nm=400, stop_nm=800, num=9)


def stack(layer_material, thickness, pols=("s",), **fdtd):
    return LayerStackProblem(superstrate=CI(n_real=1.0), substrate=CI(n_real=1.5),
                             layers=[Layer(thickness_nm=thickness, material=layer_material)], wavelengths=WL,
                             polarizations=list(pols), options={"fdtd": fdtd})


def lamellar(wl, pols=("s", "p"), **fdtd):
    return PeriodicProblem(
        lattice_a1_nm=(700, 0), superstrate=CI(n_real=1.0), substrate=CI(n_real=1.5),
        layers=[PatternedLayer(thickness_nm=250, background=CP(eps_real=1.0),
                               shapes=[Structure2D(profile=Rectangle(size_nm=(300, 1.0)), material=CP(eps_real=6.0))])],
        wavelengths=list(wl), incidence=Incidence(polarizations=list(pols)),
        options={"fdtd": fdtd, "rcwa": {"harmonics": 30}})


def max_abs_diff(a, b):
    return max(v["max"] for v in a.difference(b).values())


def test_fdtd_layer_stack_matches_tmm_cpu():
    for prob, tol in ((stack(CI(n_real=2.0), 300), 0.01),
                      (stack(osc.Drude(eps_inf=5, wp_ev=9.0, gamma_ev=0.05), 30, pols=("s", "p")), 0.005)):
        res = osc.solve(prob, "fdtd", "cpu")
        ref = osc.solve(prob, "tmm", "cpu")
        assert res.kind == "spectrum" and res.provenance["precision"] == "double"
        assert max_abs_diff(res, ref) < tol
        assert res.provenance["real_cells"][:2] == [1, 1]
    assert np.allclose(res.r["s"], res.r["p"])  # normal incidence: s and p coincide (one run)


def test_fdtd_periodic_supports_estimate_and_kinds():
    b = osc.get_backend("omniscatter-fdtd")
    assert {"scattering", "periodic", "layer_stack"} <= set(b.problem_kinds)
    p = lamellar([600.0])
    assert b.supports(p)[0]
    oblique = p.model_copy(update={"incidence": Incidence(theta_deg=10)})
    ok, why = b.supports(oblique)
    assert not ok and "normal incidence" in why
    skew = p.model_copy(update={"lattice_a1_nm": (600, 100)})
    assert not b.supports(skew)[0]
    est = osc.estimate(p, method="fdtd", device="cpu")
    nx = est["real_cells"][0]
    assert est["spacing_nm"][0] * nx == pytest.approx(700.0)  # whole cells per period (non-cubic)
    assert est["runs"] == 2 and est["memory_bytes"] > 0


@pytest.mark.gpu
@needs_cuda
def test_fdtd_layer_stack_gpu_matches_cpu_and_metal_film():
    prob = stack(osc.Drude(eps_inf=5, wp_ev=9.0, gamma_ev=0.05), 30, run_time_fs=20)
    cpu, gpu = osc.solve(prob, "fdtd", "cpu"), osc.solve(prob, "fdtd", "gpu")
    assert gpu.provenance["framework"] == "torch"
    np.testing.assert_allclose(gpu.r["s"], cpu.r["s"], atol=2e-5)
    np.testing.assert_allclose(gpu.t["s"], cpu.t["s"], atol=2e-5)
    ag = stack(LibraryMaterial(name="Ag", dataset="Johnson-Christy"), 40)
    assert max_abs_diff(osc.solve(ag, "fdtd", "gpu"), osc.solve(ag, "tmm", "cpu")) < 0.01


@pytest.mark.gpu
@needs_cuda
def test_fdtd_1d_grating_matches_rcwa():
    wl = [500.0, 600.0, 800.0, 900.0, 1000.0, 1200.0]  # avoids the Rayleigh anomaly at 700 nm
    prob = lamellar(wl)
    cmp = osc.compare(prob, ["fdtd", "rcwa"], device="gpu", reference="rcwa")
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
    res, ref = osc.solve(prob, "fdtd", "gpu"), osc.solve(prob, "rcwa", "cpu")

    def peak(r):
        i = int(np.argmax(r))
        y0, y1, y2 = r[i - 1:i + 2]
        return wl[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (wl[1] - wl[0])

    rf, rr = np.asarray(res.r["s"][0]), np.asarray(ref.r["s"][0])
    assert abs(peak(rf) - peak(rr)) < 0.5  # measured 0.09 nm
    assert rf.max() > 0.9
    assert abs(rf[0] - rr[0]) < 0.01 and abs(rf[-1] - rr[-1]) < 0.02  # off resonance
