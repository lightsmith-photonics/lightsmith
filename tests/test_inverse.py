"""Inverse design: parametrisation, optimisers, export, and a small end-to-end topology optimisation."""

import math

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.problems import DeviceProblem, Port, Prism, Rectangle, Structure

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from lightsmith import inverse  # noqa: E402
from lightsmith.inverse import parametrization as par  # noqa: E402,I001

SI = ls.ConstantIndex(n_real=3.48)
SIO2 = ls.ConstantIndex(n_real=1.44)


def wg(cx, cy, sx, sy):
    return Structure(geometry=Prism(profile=Rectangle(center_nm=(cx, cy), size_nm=(sx, sy)), z_min_nm=-110,
                                    z_max_nm=110), material=SI)


def bend_spec(method="mma", max_iter=12, dl=40.0, half=600.0, **opt):
    dev = DeviceProblem(
        dimension=2, background=SIO2, structures=[wg(-5000 - half, 0, 10000, 400), wg(0, 5000 + half, 400, 10000)],
        domain_size_nm=(2 * half + 1400, 2 * half + 1400, 0), wavelengths=[1550.0],
        ports=[Port(name="o1", center_nm=(-half - 450, 0, 0), size_nm=(0, 1500, 1), direction="+"),
               Port(name="o2", center_nm=(0, half + 450, 0), size_nm=(1500, 0, 1), direction="-")],
        options={"fdfd": {"dl_nm": dl, "pml_cells": 10}})
    return inverse.OptimizationProblem(
        device=dev,
        design_regions=[inverse.DesignRegion(size_nm=(2 * half, 2 * half, 220), material_low=SIO2, material_high=SI,
                                             filter_radius_nm=80)],
        objective=[inverse.SParameterTerm(out_port="o2", in_port="o1")],
        optimizer=inverse.OptimizerSettings(method=method, max_iter=max_iter, beta_every=1000, **opt))


# --- parametrisation ---------------------------------------------------------------


def test_filter_and_projection():
    rho = torch.full((20, 30), 0.3, dtype=torch.float64)
    np.testing.assert_allclose(par.conic_filter(rho, 3.5).numpy(), 0.3, atol=1e-12)  # constants preserved
    x = torch.linspace(0, 1, 11, dtype=torch.float64)
    p = par.tanh_projection(x, 8.0, 0.5)
    assert abs(float(p[5]) - 0.5) < 1e-12 and float(p[0]) == pytest.approx(0) and float(p[-1]) == pytest.approx(1)
    assert torch.all(torch.diff(p) > 0)
    assert torch.equal(par.tanh_projection(x, math.inf, 0.5), (x > 0.5).double())
    assert par.binarization(torch.tensor([0.0, 1.0, 1.0])) == 1.0 and par.grayness(torch.tensor([0.5])) == 1.0


def test_lengthscale_constraint_flags_thin_features():
    dl, R, b = 10.0, 60.0, 80.0
    thin, wide = torch.zeros(60, 60, dtype=torch.float64), torch.zeros(60, 60, dtype=torch.float64)
    thin[:, 29:31] = 1  # 20 nm line < 80 nm
    wide[:, 20:40] = 1  # 200 nm line
    vals = []
    for rho in (thin, wide):
        rt = par.conic_filter(rho, R / dl)
        rp = par.tanh_projection(rt, 64.0, 0.5)
        vals.append(sum(float(g) for g in par.lengthscale_constraints(rt, rp, dl, R, b)))
    assert vals[0] > 100 * max(vals[1], 1e-12)
    eta_e, eta_d = par.eta_from_lengthscale(80, 60)  # b/(2R) = 2/3 > 1/2
    assert eta_e == pytest.approx(1 - (1 - 2 / 3) ** 2) and eta_d == pytest.approx(1 - eta_e)
    with pytest.raises(ValueError):
        par.eta_from_lengthscale(200, 60)


def test_mma_is_monotone_on_a_quadratic():
    target = np.linspace(0.1, 0.9, 25)

    def fg(x):
        return float(((x - target) ** 2).sum() + 0.3 * np.sin(5 * x).sum()), 2 * (x - target) + 1.5 * np.cos(5 * x)

    opt, x = inverse.MMA(), np.full(25, 0.5)
    f, g = fg(x)
    hist = [f]
    for _ in range(30):
        x, cached = opt.step(x, f, g, fg)
        f, g = cached
        hist.append(f)
        assert np.all((x >= 0) & (x <= 1))
    assert np.all(np.diff(hist) <= 1e-12) and hist[-1] < hist[0] - 0.1


# --- export --------------------------------------------------------------------------


def test_marching_squares_disc_and_ring():
    x = np.arange(-500, 501, 10.0)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    disc = inverse.marching_squares((r < 300).astype(float), x, x)
    assert len(disc) == 1 and disc[0].solid
    v = disc[0].vertices
    area = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) - np.dot(v[:, 1], np.roll(v[:, 0], 1)))
    assert area == pytest.approx(np.pi * 300**2, rel=0.02)
    ring = inverse.marching_squares(((r < 400) & (r > 200)).astype(float), x, x)
    assert sorted(c.depth for c in ring) == [0, 1]
    structs = inverse.to_structures(ring, SI, SIO2, -110, 110, (0, 0), (1010, 1010))
    assert [s.material for s in structs] == [SIO2, SI, SIO2]
    # Rasterising the structures reproduces the ring.
    from lightsmith.raster import inside_geometry

    owner = np.full(X.shape, -1)
    for i, s in enumerate(structs):
        owner[inside_geometry(s.geometry, X, Y, np.zeros_like(X))] = i
    solid = owner == 1
    assert np.mean(solid == ((r < 400) & (r > 200))) > 0.98


def test_gds_roundtrip(tmp_path):
    x = np.arange(0, 400, 10.0)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = ((np.abs(X - 200) < 120) & (np.abs(Y - 200) < 120) & ~((np.abs(X - 200) < 40) & (np.abs(Y - 200) < 40)))
    cs = inverse.marching_squares(rho.astype(float), x, x)
    info = inverse.write_gds(str(tmp_path / "d.gds"), cs, layer=(3, 0))
    from lightsmith.layout import read_gds

    polys = read_gds(info["path"])
    assert info["holes"] == 1 and sorted(polys) == [(3, 0), (3, 1)]
    outer = polys[(3, 0)][0]
    assert outer[:, 0].min() == pytest.approx(85, abs=1) and outer[:, 0].max() == pytest.approx(315, abs=1)


# --- spec ---------------------------------------------------------------------------


def test_spec_json_roundtrip():
    spec = bend_spec()
    again = inverse.OptimizationProblem.from_json(spec.to_json())
    assert again == spec and again.kind == "optimization"
    schema = inverse.OptimizationProblem.model_json_schema()
    assert "design_regions" in schema["properties"]
    with pytest.raises(ValueError):
        inverse.DesignRegion(size_nm=(1, 1, 1), material_low=SIO2, material_high=SI, min_feature_nm=100,
                             filter_radius_nm=20)


def test_backend_protocol():
    b = ls.get_backend("lightsmith-fdfd")
    assert isinstance(b, inverse.DifferentiableBackend)
    inverse.check_backend(b)
    with pytest.raises(ValueError):
        inverse.check_backend(ls.get_backend("dda"))


def test_optimisation_options_merge_per_method_key():
    spec = bend_spec(dl=50.0)
    spec = spec.model_copy(update={"options": {"fdfd": {"pml_reflection": 1e-6}}})
    opt = inverse.TopologyOptimizer(spec)
    assert opt.grid.dl == 50.0 and opt.grid.npml == (10, 10)  # device's grid settings survive
    assert opt.device_problem.options["fdfd"] == {"dl_nm": 50.0, "pml_cells": 10, "pml_reflection": 1e-6}


def test_design_region_must_not_touch_ports():
    spec = bend_spec()
    bad = spec.model_copy(update={"design_regions": [spec.design_regions[0].model_copy(
        update={"size_nm": (2000, 1200, 220)})]})
    with pytest.raises(ValueError, match="overlaps port"):
        inverse.TopologyOptimizer(bad)


# --- end to end -----------------------------------------------------------------------


def test_bend_topology_optimisation_improves_monotonically(tmp_path):
    seen = []
    res = inverse.optimize(bend_spec(max_iter=12), device="cpu", callback=lambda s: seen.append(s.objective),
                           checkpoint=str(tmp_path / "ck.npz"))
    obj = [h["objective"] for h in res.history]
    assert obj == seen and len(obj) == 12
    assert np.all(np.diff(obj) >= -1e-9)  # globally convergent MMA: monotone at fixed beta
    assert obj[0] < 0.1 and obj[-1] > 0.8
    assert res.objective_binary > 0.6
    assert res.structures and res.sparameters is not None
    t_exported = abs(res.sparameters.s()[0, 1, 0]) ** 2
    assert abs(t_exported - res.objective_binary) < 0.15  # polygons reproduce the pixel design
    assert (tmp_path / "ck.npz").exists()
    # The result is plain JSON.
    again = inverse.OptimizationResult.model_validate_json(res.model_dump_json())
    assert again.objective_final == res.objective_final


def test_checkpoint_resume_and_beta_continuation(tmp_path):
    ck = str(tmp_path / "run.npz")
    spec = bend_spec(method="adam", max_iter=3, dl=50.0, learning_rate=0.05)
    first = inverse.TopologyOptimizer(spec).run(checkpoint=ck)
    assert len(first["history"]) == 3
    longer = spec.model_copy(update={"optimizer": spec.optimizer.model_copy(
        update={"max_iter": 5, "beta_every": 2, "beta_growth": 2.0})})
    out = inverse.TopologyOptimizer(longer).run(checkpoint=ck)
    assert len(out["history"]) == 5 and out["history"][:3] == first["history"]
    assert out["beta"] > spec.optimizer.beta_start


def test_field_objective_and_lbfgsb():
    spec = bend_spec(method="lbfgsb", max_iter=4, dl=50.0)
    spec = spec.model_copy(update={"objective": [inverse.FieldIntensityTerm(
        source_port="o1", center_nm=(0, 0, 0), size_nm=(100, 100, 0))]})
    out = inverse.TopologyOptimizer(spec).run()
    obj = [h["objective"] for h in out["history"]]
    assert obj[-1] > obj[0]


def test_solve_dispatches_optimization_problems():
    res = ls.solve(bend_spec(max_iter=2, dl=50.0), method="inverse", device="cpu")
    assert isinstance(res, inverse.OptimizationResult) and len(res.history) == 2
    est = ls.estimate(bend_spec(), method="inverse", device="cpu")
    assert est["differentiable_backend"] == "lightsmith-fdfd" and est["sources_per_iteration"] == 1
