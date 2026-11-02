"""3D FDFD (full-vector Yee, SC-PML, BiCGStab) and 3D inverse design: references and gradient checks."""

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.problems import DeviceProblem, Port, Prism, Rectangle, Structure

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

import array_api_compat.numpy as xnp  # noqa: E402

from lightsmith import inverse  # noqa: E402
from lightsmith.backends._fdfd3d.operators import Maxwell3D, Stretch  # noqa: E402
from lightsmith.backends._fdfd3d.ports import incident_field  # noqa: E402
from lightsmith.backends.fdfd import FDFD  # noqa: E402

SI = ls.ConstantIndex(n_real=3.476)
SIO2 = ls.ConstantIndex(n_real=1.444)
SI_STRIP_TE0 = 2.44523  # 500 x 220 nm strip at 1550 nm, converged FDE (tests/test_fde.py)


def wg(cx, cy, sx, sy, mat=SI):
    return Structure(geometry=Prism(profile=Rectangle(center_nm=(cx, cy), size_nm=(sx, sy)), z_min_nm=-110,
                                    z_max_nm=110), material=mat)


def straight(dl=80.0, lx=1920.0, width=400.0, extra=(), ly=1280.0, lz=1120.0, **opts):
    """A straight strip along x between ports o1 (x < 0) and o2 (x > 0), 4 cells inside the domain."""
    xp = lx / 2 - 5 * dl
    return DeviceProblem(
        dimension=3, background=SIO2, structures=[wg(0, 0, 10 * lx, width), *extra],
        domain_size_nm=(lx, ly, lz), wavelengths=[1550.0],
        ports=[Port(name="o1", center_nm=(-xp, 0, 0), size_nm=(0, ly - 80, lz - 80), direction="+"),
               Port(name="o2", center_nm=(xp, 0, 0), size_nm=(0, ly - 80, lz - 80), direction="-")],
        options={"fdfd": {"dl_nm": dl, "pml_cells": 5, "solver": "bicgstab", "tol": 1e-9, **opts}})


# --- operator -------------------------------------------------------------------------


@pytest.mark.parametrize("grad_div", [False, True])
def test_operator_matches_assembled_matrix_and_transpose(grad_div):
    rng = np.random.default_rng(0)
    shape = (5, 6, 7)
    eps = 1 + rng.random(shape) + 0.1j * rng.random(shape)
    sn = [1 + 0.3j * rng.random(n) for n in shape]
    se = [1 + 0.3j * rng.random(n + 1) for n in shape]
    op = Maxwell3D(xnp, eps, 0.4, Stretch.build(xnp, sn, se), grad_div=grad_div)
    A = op.to_scipy()  # assembled independently with Kronecker products
    X = rng.random((2, 3) + shape) + 1j * rng.random((2, 3) + shape)
    flat = X.reshape(2, -1).T
    np.testing.assert_allclose(op.apply(X).reshape(2, -1).T, A @ flat, rtol=0, atol=1e-12)
    np.testing.assert_allclose(op.apply_t(X).reshape(2, -1).T, A.T @ flat, rtol=0, atol=1e-12)
    np.testing.assert_allclose(op.diagonal().ravel(), A.diagonal(), rtol=0, atol=1e-12)


def test_grad_div_term_vanishes_on_solutions():
    """div_B curl_B = 0 on the stretched grid, so curl-curl images are divergence-free."""
    rng = np.random.default_rng(1)
    shape = (6, 5, 7)
    sn = [1 + 0.5j * rng.random(n) for n in shape]
    se = [1 + 0.5j * rng.random(n + 1) for n in shape]
    op = Maxwell3D(xnp, np.ones(shape, complex), 0.3, Stretch.build(xnp, sn, se), grad_div=False)
    H = tuple(rng.random((1,) + shape) for _ in range(3))
    R = op._curl(op.B, H)
    div = sum(op.B(R[a], a) for a in range(3))
    assert np.abs(div).max() < 1e-12 * max(np.abs(r).max() for r in R)


# --- port modes -------------------------------------------------------------------------


def test_port_mode_neff_matches_converged_fde():
    dev = DeviceProblem(dimension=3, background=SIO2, structures=[wg(0, 0, 5000, 500)],
                        domain_size_nm=(80, 1300, 1000), wavelengths=[1550.0],
                        ports=[Port(name="o1", center_nm=(0, 0, 0), size_nm=(0, 1300, 1000), direction="+")],
                        options={"fdfd": {"dl_nm": 10.0, "pml_cells": 4}})
    eng = FDFD.engine(dev)
    m = eng.modes(eng.base_eps(1550.0), 1550.0)[0][0]
    assert m.te_fraction(0) > 0.95  # TE0: E mostly along y (the small window clips the tails at ~1e-2)
    # First-order staircasing of the 220 nm layer: 10 nm cells give ~1e-2 (see the grid convergence below).
    assert m.neff.real == pytest.approx(SI_STRIP_TE0, abs=1.5e-2)


def test_port_modes_are_exact_and_biorthogonal():
    """A f = 0 for the launched mode inside its window, and the projectors are biorthogonal."""
    dev = straight(dl=40.0, width=900.0, lx=1600.0, ly=2000.0)
    dev = dev.model_copy(update={"ports": [p.model_copy(update={"num_modes": 2}) for p in dev.ports]})
    eng = FDFD.engine(dev)
    eps = eng.base_eps(1550.0)
    modes = eng.modes(eps, 1550.0)[0]
    pg = eng.ports[0]
    for grad_div in (False, True):
        op = eng.operator(xnp, eps, 1550.0)
        op.grad_div = grad_div
        if grad_div:
            op.w = (1 / eps)[None]
        for md in modes:
            f, _ = incident_field(pg, md, eng.grid.shape, range(pg.i0 - 3, pg.i0 + 4))
            r = op.apply(f)
            lo1, hi1, lo2, hi2 = pg.window
            b = 2 if grad_div else 0  # grad-div sees the zero field outside the window at its edge rows
            rows = r[0, :, pg.i0 - 2:pg.i0 + 3, lo1 + b:hi1 - b, lo2 + b:hi2 - b]
            scale = np.abs(op.k2 * f).max()
            assert np.abs(rows).max() < 1e-9 * scale
    P = np.array([[np.sum(mi.proj[0] * mj.e1 + mi.proj[1] * mj.e2) for mj in modes] for mi in modes])
    np.testing.assert_allclose(P, np.eye(2), atol=1e-9)


# --- S-parameters -------------------------------------------------------------------------


def test_straight_waveguide_transmits_with_tiny_reflection():
    r = ls.solve(straight(), method="fdfd", device="cpu")
    S = r.s()[0]
    assert abs(S[1, 0]) ** 2 == pytest.approx(1, abs=2e-3)
    assert abs(S[0, 0]) ** 2 < 1e-4
    np.testing.assert_allclose(S[1, 0], S[0, 1], atol=1e-5)  # reciprocity
    assert r.provenance["max_residual"] < 1e-8


def test_reciprocity_and_power_with_an_asymmetric_scatterer():
    bump = wg(120, 160, 240, 200)
    r = ls.solve(straight(extra=[bump]), method="fdfd", device="cpu")
    S = r.s()[0]
    # Radiation from the scatterer crosses the monitor planes 4 cells away and slightly contaminates the
    # two-plane modal decomposition (the same scheme as the 2D solver).
    np.testing.assert_allclose(S[1, 0], S[0, 1], atol=3e-3)
    p = (np.abs(S) ** 2).sum(axis=0)
    assert np.all(p < 1 + 2e-3) and np.all(p > 0.5)  # lossless, but radiation leaves through the PML
    assert abs(S[0, 0] - S[1, 1]) > 1e-4  # the scatterer really is asymmetric


@pytest.mark.gpu
def test_gpu_bicgstab_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    dev = straight(extra=[wg(120, 160, 240, 200)])
    a = ls.solve(dev, method="fdfd", device="cpu").s()
    b = ls.solve(dev, method="fdfd", device="gpu").s()
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.slow
def test_width_step_matches_eme():
    """Independent reference: EME (full-vector FDE cells + mode matching) for an abrupt 450 -> 900 nm step."""
    dl = 25.0
    dev = DeviceProblem(
        dimension=3, background=SIO2, structures=[wg(-1500, 0, 3000, 450), wg(1500, 0, 3000, 900)],
        domain_size_nm=(1600, 2000, 1400), wavelengths=[1550.0],
        ports=[Port(name="o1", center_nm=(-600, 0, 0), size_nm=(0, 1800, 1200), direction="+"),
               Port(name="o2", center_nm=(600, 0, 0), size_nm=(0, 1800, 1200), direction="-", num_modes=2)],
        options={"fdfd": {"dl_nm": dl, "pml_cells": 10},
                 "eme": {"grid_spacing_nm": dl, "num_modes": 20}})
    f = ls.solve(dev, method="fdfd", device="auto").s()[0]
    e = ls.solve(dev, method="eme", device="cpu").s()[0]
    # o2@0 (TE0) is row 1; the symmetric step does not excite the odd TE1 (row 2).
    assert abs(f[1, 0]) ** 2 == pytest.approx(abs(e[1, 0]) ** 2, abs=0.01)
    assert abs(f[2, 0]) ** 2 < 1e-3 and abs(e[2, 0]) ** 2 < 1e-3


def test_estimate_reports_3d_cost():
    est = ls.estimate(straight(), method="fdfd", device="cpu")
    assert est["dimension"] == 3 and est["unknowns"] == 3 * np.prod(est["grid"])
    assert est["memory_bytes_iterative"] > 0 and est["seconds_per_solve_estimate"] > 0
    spec = inverse.OptimizationProblem(device=straight(), design_regions=[region()],
                                       objective=[inverse.SParameterTerm(out_port="o2", in_port="o1")])
    e2 = ls.estimate(spec, method="inverse", device="cpu")
    assert e2["solves_per_iteration"] == 2  # one wavelength: forward + adjoint
    assert e2["memory_bytes"] > e2["memory_bytes_autograd_fields"] >= 16 * est["unknowns"]
    assert e2["runtime_s_estimate"] > 0


# --- inverse design in 3D ----------------------------------------------------------------


def region(parametrization="extruded", filter_radius_nm=0.0, size_nm=(320, 480, 220), **kw):
    return inverse.DesignRegion(size_nm=size_nm, material_low=SIO2, material_high=SI, init="random",
                                filter_radius_nm=filter_radius_nm, parametrization=parametrization, **kw)


def gap_spec(parametrization="extruded", filter_radius_nm=0.0, max_iter=4, tiny=False):
    """Two waveguide stubs with a gap filled by the design region (320 nm, or 200 nm on the tiny grid)."""
    if tiny:  # 100 nm cells, 8k cells: for gradient checks only (the windows clip the mode tails)
        dev, gap = straight(dl=100.0, lx=2000.0, ly=800.0, lz=600.0, tol=1e-10), 200.0
    else:
        dev, gap = straight(ly=960.0, lz=720.0, tol=1e-10), 320.0
    dev = dev.model_copy(update={"structures": [wg(-2500 - gap / 2, 0, 5000, 400), wg(2500 + gap / 2, 0, 5000, 400)]})
    return inverse.OptimizationProblem(
        device=dev, design_regions=[region(parametrization, filter_radius_nm, size_nm=(gap, 480, 220))],
        objective=[inverse.SParameterTerm(out_port="o2", in_port="o1")],
        optimizer=inverse.OptimizerSettings(method="mma", max_iter=max_iter, beta_every=1000))


@pytest.mark.parametrize("param,radius", [("extruded", 0.0), ("volumetric", 120.0)])
def test_3d_adjoint_gradient_matches_finite_differences(param, radius):
    """d(loss)/d(design variables) through filter, projection, extrusion and the 3D adjoint solve.

    Tolerance: 1e-4 relative (solver tolerance 1e-10, central differences with h = 1e-3).
    """
    topt = inverse.TopologyOptimizer(gap_spec(param, radius, tiny=True), device="cpu")
    reg = topt.regions[0]
    assert reg.mode == param and len(reg.shape) == (2 if param == "extruded" else 3)
    x = topt.initial_x()
    f, g, _ = topt.evaluate(x, 4.0)
    rng = np.random.default_rng(3)
    h = 1e-3
    for i in rng.choice(len(x), 3, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (topt.evaluate(xp, 4.0, grad=False)[0] - topt.evaluate(xm, 4.0, grad=False)[0]) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-6 * np.abs(g).max())


def test_3d_extruded_optimisation_improves_and_exports(tmp_path):
    spec = gap_spec(max_iter=4, tiny=True)
    res = inverse.optimize(spec, device="cpu", validate=False)
    hist = [h["objective"] for h in res.history]
    assert hist[-1] > hist[0] + 0.02
    assert res.designs[0].parametrization == "extruded" and res.provenance["grid"][2] > 1
    solids = [s for s in res.structures if "solid" in s.name]
    assert solids and all(s.geometry.z_min_nm == -110 and s.geometry.z_max_nm == 110 for s in solids)
    d = res.designs[0]
    contours = inverse.marching_squares(np.asarray(d.density), np.asarray(d.x_nm), np.asarray(d.y_nm))
    info = inverse.write_gds(str(tmp_path / "d.gds"), contours)
    assert info["polygons"] == len(contours)


def test_volumetric_region_and_z_symmetry_validation():
    with pytest.raises(ValueError):
        inverse.DesignRegion(size_nm=(1, 1, 1), material_low=SIO2, material_high=SI, symmetry=["z"])
    r = inverse.DesignRegion(size_nm=(1, 1, 1), material_low=SIO2, material_high=SI, symmetry=["z"],
                             parametrization="volumetric")
    assert inverse.OptimizationProblem.model_validate_json(
        gap_spec().model_copy(update={"design_regions": [r]}).to_json()).design_regions[0].symmetry == ["z"]


def test_design_region_must_not_touch_3d_ports():
    spec = gap_spec()
    big = spec.design_regions[0].model_copy(update={"size_nm": (1600, 480, 220)})
    with pytest.raises(ValueError, match="overlaps port"):
        inverse.TopologyOptimizer(spec.model_copy(update={"design_regions": [big]}), device="cpu")
