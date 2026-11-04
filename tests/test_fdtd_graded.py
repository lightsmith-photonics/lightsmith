"""Graded (non-uniform) FDTD grids: grid model, consistency with the uniform grid, and Mie validation.

Measured (RTX 5070, float64 unless noted; see docs/solvers/fdtd.md for the full table):
- n = 3.5, r = 20 nm in vacuum, 400-800 nm, grid "auto" (2 nm inside, 6.7 nm base): within 0.11% of a uniform
  2 nm run in a converged domain (whose own error against Mie is 0.66-0.71%) with 153k cells against 242k.
- the TFSF box on a graded grid does not leak: C_sca of an n = 1 sphere 2e-15 nm^2 (axis-aligned incidence; the
  tables solve the graded 1D problem).
- a uniform grid through the graded code path reproduces the uniform run to 1e-13 (dielectric, metal, substrates).
"""

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.device import Device, _torch_cuda
from lightsmith.problems import Dipole, EmissionProblem
from lightsmith.spec import FarFieldRequest, GaussianBeam, NearFieldRequest, Outputs, Substrate

needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
DRUDE = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)


def sphere_sim(material, radius, wavelengths, medium=1.0, **fdtd):
    return ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material)],
        medium_index=medium, wavelengths=wavelengths, options={"fdtd": fdtd})


def rel(a, b):
    return np.abs(np.asarray(a) / np.asarray(b) - 1)


def max_diff(a, b, key="c_ext_nm2"):
    """Largest difference of a cross section between two runs, relative to the largest C_ext."""
    x, y = np.asarray(getattr(a, key)), np.asarray(getattr(b, key))
    return np.abs(x - y).max() / np.abs(np.asarray(b.c_ext_nm2)).max()


def override(size, spacing, center=(0.0, 0.0, 0.0)):
    return [{"center_nm": center, "size_nm": (size,) * 3, "spacing_nm": spacing}]


# --- the grid model ----------------------------------------------------------------------------------------


def test_graded_axis_is_bounded_symmetric_and_refined():
    from lightsmith.backends._fdtd.grid import Grid, Region, axis_nodes

    reg = [Region.make([-11, -5, -11], [11, 5, 11], [0.5, 0.25, 0.5])]
    for a, fine in ((0, 0.5), (1, 0.25)):
        lo, hi = axis_nodes(0.0, 20.0, reg, a, 1.25, 10, 10, True)
        assert np.allclose(lo, hi)  # symmetric about the centre on a mirror axis
        w = np.concatenate([lo[::-1], hi])
        q = w[1:] / w[:-1]
        assert max(q.max(), (1 / q).max()) <= 1.25 * (1 + 1e-9)
        assert hi[0] == pytest.approx(fine) and hi.max() <= 20.0
        edge = reg[0].hi[a]
        assert np.all(hi[:int(np.searchsorted(np.cumsum(hi), edge - 1e-9))] == pytest.approx(fine))
    # asymmetric region, no mirror: only the side that holds it is refined
    lo, hi = axis_nodes(0.0, 10.0, [Region.make([5, 5, 5], [9, 9, 9], 1.0)], 0, 1.3, 1, 1, False)
    assert hi.min() == pytest.approx(1.0) and (len(lo) == 0 or lo.min() > hi.min())
    g = Grid((np.cumsum(np.r_[0, w]), np.arange(5.0), np.arange(4.0)))
    u = np.array([0.0, 0.5, 3.25, len(w) - 0.5])
    assert np.allclose(g.index_of(0, g.coord(0, u)), u)
    assert np.allclose(g.dual(0)[1:-1], 0.5 * (w[1:] + w[:-1]))
    assert not g.uniform and g.widths()[1] is None


def test_graded_layout_boxes_lie_in_uniform_cells():
    """TFSF and flux-box faces, gaps and CPML sit where the grid is uniform along the face normal."""
    from lightsmith.backends._fdtd import scattering
    from lightsmith.backends._fdtd.options import FDTDOptions

    sim = sphere_sim(DRUDE, 10, [450.0, 650.0], medium=1.33, grid="auto")
    opts = FDTDOptions.model_validate(sim.options["fdtd"])
    lay = scattering.plan(sim, opts, False)["layout"]
    g = lay.grid
    assert g is not None and g.max_ratio() <= 1.25 * (1 + 1e-9)
    assert g.min_spacing() == pytest.approx(0.5)  # 20 cells per radius for a metal
    for a in range(3):
        p = g.prim(a)
        for box, k in ((lay.tfsf, 3), (lay.tot, 1), (lay.sca, 1)):  # k uniform cells on either side of a face
            for i in (box[0][a], box[1][a]):
                if i > 0:  # (beyond a mirror plane the low face is the image of the high one)
                    assert np.allclose(p[i - k:i + k], p[-1], rtol=1e-9, atol=0), (a, i)
        assert np.allclose(p[-(lay.pml + 4):], p[-1], rtol=1e-9, atol=0)
    est = ls.estimate(sim, method="fdtd", device="cpu")
    uni = ls.estimate(sphere_sim(DRUDE, 10, [450.0, 650.0], medium=1.33), method="fdtd", device="cpu")
    assert est["cells"] * 3 < uni["cells"]  # 3.6x fewer cells than the uniform 0.5 nm grid
    assert est["graded_grid"]["spacing_min_nm"] == [pytest.approx(0.5)] * 3


def test_graded_options_collapse_to_the_uniform_run():
    """A mesh override no finer than the base spacing leaves the grid uniform: the run is bit for bit today's."""
    sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 30, [450.0, 650.0], spacing_nm=6.0, precision="double",
                     run_time_fs=6.0, pml_cells=6)
    a = ls.solve(sim, "fdtd", "cpu")
    opts = {**sim.options["fdtd"], "mesh_override": override(40.0, 6.0)}
    b = ls.solve(sim.model_copy(update={"options": {"fdtd": opts}}), "fdtd", "cpu")
    assert "graded_grid" not in b.provenance and b.provenance["grid"] == a.provenance["grid"]
    for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        assert np.array_equal(getattr(a, k), getattr(b, k))
    # grid "auto" with nothing finer to resolve than the background: today's uniform grid
    big = sphere_sim(ls.ConstantIndex(n_real=1.2), 150, [500.0, 700.0])
    est_u = ls.estimate(big, method="fdtd", device="cpu")
    est_a = ls.estimate(big.model_copy(update={"options": {"fdtd": {"grid": "auto"}}}), method="fdtd", device="cpu")
    assert "graded_grid" not in est_a and est_a["grid"] == est_u["grid"]


FORCED_CASES = {
    "dielectric, 2 mirrors, near and far field": ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=30), material=ls.ConstantIndex(n_real=2.0))],
        wavelengths=[450.0, 650.0], options={"fdtd": {"spacing_nm": 5.0}},
        outputs=Outputs(far_field=FarFieldRequest(theta_deg=[0, 60, 120], phi_deg=[0, 90], wavelengths_nm=[500.0]),
                        near_field=NearFieldRequest(plane="xz", half_width_nm=40, points=7, wavelengths_nm=[500.0]))),
    "Drude sphere (vertex tensor)": ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=16), material=DRUDE)], medium_index=1.33,
        wavelengths=[480.0, 560.0], options={"fdtd": {"spacing_nm": 4.0}}),
    "on glass, normal incidence (column source)": ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=ls.ConstantIndex(n_real=2.0),
                               position_nm=(0, 0, 20))],
        substrate=Substrate(material=ls.ConstantIndex(n_real=1.5)), wavelengths=[450.0, 650.0],
        options={"fdtd": {"spacing_nm": 5.0}}),
    "on an absorbing substrate (tables, particle absorption, layered far field)": ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=ls.ConstantIndex(n_real=2.0, n_imag=0.2),
                               position_nm=(0, 0, 20))],
        substrate=Substrate(material=ls.ConstantIndex(n_real=1.5, n_imag=0.3)), wavelengths=[500.0, 600.0],
        illumination=ls.PlaneWave(direction=(0, 0, -1), polarization=(1, 0, 0)), options={"fdtd": {"spacing_nm": 5.0}}),
}


@pytest.mark.parametrize("case", list(FORCED_CASES))
def test_graded_code_path_reproduces_uniform_grid_cpu(case, monkeypatch):
    """Forced through the graded code path (every axis given as width arrays), a uniform grid steps as today's
    uniform engine: per-axis primary and dual widths in the curls, CPML, TFSF tables, layered background, flux
    weights, conformal off-diagonal terms and vertex tensor, mirror planes. Agreement to 1e-12 in double precision
    (measured 3e-15 to 6e-14: the same arithmetic in another order)."""
    from lightsmith.backends._fdtd import grid as G

    sim = FORCED_CASES[case]
    fd = {**sim.options["fdtd"], "precision": "double", "run_time_fs": 6.0, "pml_cells": 6}
    a = ls.solve(sim.model_copy(update={"options": {"fdtd": fd}}), "fdtd", "cpu")
    h = fd["spacing_nm"]
    monkeypatch.setattr(G.Grid, "uniform", property(lambda self: False))
    monkeypatch.setattr(G.Grid, "widths", lambda self: tuple(self.prim(ax) for ax in range(3)))
    monkeypatch.setattr(G, "axis_nodes", lambda center, base, regions, ax, *args: (np.full(3, base), np.full(3, base)))
    fd_g = {**fd, "mesh_override": override(1.0, h / 2)}
    b = ls.solve(sim.model_copy(update={"options": {"fdtd": fd_g}}), "fdtd", "cpu")
    assert b.provenance["graded_grid"]["max_ratio"] == pytest.approx(1.0)
    assert b.provenance["grid"] == a.provenance["grid"]
    for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        assert max_diff(b, a, k) < 1e-12, k
    for fa, fb in zip(a.far_field or [], b.far_field or []):
        x, y = np.asarray(fa.dcsca_domega_nm2), np.asarray(fb.dcsca_domega_nm2)
        assert np.abs(x - y).max() < 1e-9 * np.abs(x).max()
    for na, nb in zip(a.near_field or [], b.near_field or []):
        x, y = np.asarray(na.enhancement), np.asarray(nb.enhancement)
        assert np.nanmax(np.abs(x - y)) < 1e-9 * np.nanmax(x)


@pytest.mark.parametrize("direction", [(0, 0, -1), (1, 0, 0)])
def test_graded_tfsf_does_not_leak_cpu(direction):
    """The tables solve the graded 1D problem along the incidence axis, so the box stays leak-free on a graded
    grid: an n = 1 sphere scatters nothing (measured 2e-15 nm^2 against a geometric 1257 nm^2)."""
    pol = (0, 1, 0)
    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=ls.ConstantIndex(n_real=1.0))],
        wavelengths=[400.0, 700.0], illumination=ls.PlaneWave(direction=direction, polarization=pol),
        options={"fdtd": {"precision": "double", "spacing_nm": 10.0, "mesh_override": override(50.0, 2.5)}})
    res = ls.solve(sim, "fdtd", "cpu")
    assert res.provenance["graded_grid"]["spacing_min_nm"] == [pytest.approx(2.5)] * 3
    assert np.abs(res.c_sca_nm2).max() < 1e-10 * np.pi * 20**2
    assert np.abs(res.c_abs_nm2).max() < 1e-7 * np.pi * 20**2


def test_graded_mirror_symmetry_matches_full_run_cpu():
    """The graded grid is symmetric about the mirror planes, so the quarter grid steps as the whole one."""
    sim = sphere_sim(DRUDE, 12, [480.0, 560.0], medium=1.33, spacing_nm=6.0, mesh_override=override(30.0, 2.0),
                     precision="double", run_time_fs=6.0, pml_cells=6)
    out = []
    for sym in ("auto", "off"):
        s = sim.model_copy(update={"options": {"fdtd": {**sim.options["fdtd"], "symmetry": sym}}})
        out.append(ls.solve(s, "fdtd", "cpu"))
    a, b = out
    assert a.provenance["symmetry"]["reduction"] == 4 and "graded_grid" in a.provenance
    for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        assert max_diff(a, b, k) < 1e-6, k


def _graded_widths(shape, ratio=1.4):
    """Primary widths of a strongly graded grid: 1 in the middle, growing by `ratio` per cell up to 4."""

    def axis(n):
        h = 1.0 * ratio ** np.abs(np.arange(n - 1) - (n - 2) / 2)
        return np.minimum(h, 4.0)

    return tuple(axis(n) for n in shape)


def test_graded_engine_conserves_the_discrete_energy():
    """In a lossless PEC box the graded Yee update conserves W = sum V_E eps E^n.E^n + sum V_H H^{n-1/2}.H^{n+1/2}
    (V the primary x dual cell volumes), to rounding over 4000 steps: the two curls are adjoint in this weighted
    inner product, so nothing can grow at grading transitions."""
    from lightsmith.backends._fdtd.engine import Engine, PMLSpec, dual_widths

    shape = (14, 16, 18)
    widths = _graded_widths(shape)
    eps = 2.25
    dt = 0.95 / np.sqrt(sum(1 / w.min() ** 2 for w in widths))
    eng = Engine(Device("numpy", "cpu"), shape=shape, dx=1.0, dt=dt, single=False,
                 inv_eps=np.full((3,) + shape, 1 / eps), groups=[], pml=PMLSpec(cells=((0, 0),) * 3), sources=[],
                 monitors=[], dft_stride=1, widths=widths)
    rng = np.random.default_rng(1)
    for c in range(3):
        eng.E[c][eng.RE[c]] = rng.standard_normal(eng.E[c][eng.RE[c]].shape)
        eng.H[c][eng.RH[c]] = rng.standard_normal(eng.H[c][eng.RH[c]].shape)

    def vol(kind, c):
        out = np.ones(shape)
        for a in range(3):
            prim = np.r_[widths[a], widths[a][-1]]
            v = (prim if (a == c) == (kind == "E") else dual_widths(widths[a]))
            s = [1, 1, 1]
            s[a] = shape[a]
            out = out * v.reshape(s)
        return out

    VE, VH = [vol("E", c) for c in range(3)], [vol("H", c) for c in range(3)]
    W = []
    for _ in range(4000):
        h_old = np.array(eng.H)
        eng._update_H()
        e2 = sum(np.sum(VE[c] * eps * eng.E[c] ** 2) for c in range(3))
        W.append(e2 + sum(np.sum(VH[c] * h_old[c] * eng.H[c]) for c in range(3)))
        eng._update_E()
    W = np.asarray(W)
    assert np.abs(W / W[0] - 1).max() < 1e-10


def test_graded_vertex_tensor_is_symmetric_and_bounded():
    """On a graded grid the vertex tensor's D -> E map is symmetric in the energy inner product (node volumes)
    and its Rayleigh quotients stay between 1/eps_inf,max and 1/eps_min: still a sum of passive congruences."""
    from lightsmith.backends._fdtd.engine import Engine, PMLSpec, TensorMesh, _Tensor
    from lightsmith.backends._fdtd.grid import Grid
    from lightsmith.backends._fdtd.media import Medium, Pole
    from lightsmith.backends._fdtd.setup import node_volumes, voxelise

    media = [Medium(1.77), Medium(9.5, (Pole(0.00208, 0.0, 1.8e-4),))]
    shape = (24, 24, 24)
    widths = _graded_widths(shape, 1.3)
    grid = Grid(tuple(np.r_[0.0, np.cumsum(w)] - np.sum(w) / 2 for w in widths))
    c, r = np.array([0.3, 0.1, -0.2]), 4.4
    inv_eps, groups, stats, _ = voxelise(lambda p: (np.linalg.norm(p - c, axis=-1) < r).astype(int), media, shape,
                                         grid.origin, 1.0, np.full(3, 2), np.full(3, 22), mode="conformal", grid=grid)
    tm = groups[-1]
    assert isinstance(tm, TensorMesh) and stats["tensor_vertices"] > 100
    eng = Engine(Device("numpy", "cpu"), shape=shape, dx=1.0, dt=0.3, single=False, inv_eps=inv_eps, groups=groups,
                 pml=PMLSpec(cells=((0, 0),) * 3), sources=[], monitors=[], dft_stride=1, widths=widths)
    idx = tm.idx
    V = node_volumes(idx, shape, grid)

    def apply(x):
        g = _Tensor(eng, tm)
        eng.E_flat[idx] = x
        g.step(eng)
        return np.array(eng.E_flat[idx])

    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, len(idx)))
    mx, my = apply(x), apply(y)
    assert abs(x @ (V * my) - y @ (V * mx)) < 1e-12 * abs(x @ (V * mx))
    for z in rng.standard_normal((5, len(idx))):
        q = z @ (V * apply(z)) / (z @ (V * z))
        assert 1 / 9.5 <= q <= 1 / 1.77


def test_graded_grid_refusals():
    sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 30, [500.0], grid="auto")
    beam = sim.model_copy(update={"illumination": GaussianBeam(waist_nm=800.0)})
    with pytest.raises(ValueError, match="Gaussian"):
        ls.estimate(beam, method="fdtd", device="cpu")
    oblique = sim.model_copy(update={"illumination": ls.PlaneWave(direction=(1, 0, 1), polarization=(0, 1, 0))})
    with pytest.raises(ValueError, match="oblique incidence needs the grid uniform"):
        ls.estimate(oblique, method="fdtd", device="cpu")
    emission = EmissionProblem(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=30), material=ls.ConstantIndex(n_real=1.5))],
        dipole=Dipole(position_nm=(0, 0, 40)), wavelengths=[500.0],
        options={"fdtd": {"grid": "auto"}})
    with pytest.raises(ValueError, match="scattering problems only"):
        ls.estimate(emission, method="fdtd", device="cpu")
    with pytest.raises(Exception, match="grading_ratio"):
        ls.estimate(sphere_sim(ls.ConstantIndex(n_real=1.5), 30, [500.0], grading_ratio=2.0), method="fdtd",
                    device="cpu")


# --- validation against Mie and the uniform grid (GPU) -------------------------------------------------------

WL = list(np.linspace(400.0, 800.0, 9))


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_graded_high_index_sphere_matches_uniform_and_mie():
    """n = 3.5, r = 20 nm: the graded grid (2 nm inside, 6.7 nm base) agrees with a uniform 2 nm run in a converged
    domain to well under that run's own discretisation error against Mie, with fewer cells. Measured: within 0.11%
    of the uniform run, whose error is 0.66-0.71%; 153k cells against 242k for today's uniform layout."""
    sim = sphere_sim(ls.ConstantIndex(n_real=3.5), 20, WL, precision="double")
    mie = ls.solve(sim, "mie")
    big = {**sim.options["fdtd"], "padding_nm": 50.0, "pml_gap_nm": 50.0, "pml_cells": 20}
    uni = ls.solve(sim.model_copy(update={"options": {"fdtd": big}}), "fdtd", "gpu")
    gr = ls.solve(sim.model_copy(update={"options": {"fdtd": {**sim.options["fdtd"], "grid": "auto"}}}), "fdtd", "gpu")
    est = ls.estimate(sim, method="fdtd", device="gpu")
    assert gr.provenance["graded_grid"]["max_ratio"] <= 1.25 * (1 + 1e-9)
    assert gr.provenance["cells"] < 0.7 * est["cells"]
    err_u = rel(uni.c_ext_nm2, mie.c_ext_nm2)
    assert 0.004 < err_u.min() and err_u.max() < 0.01  # the discretisation error at dx = 2 nm
    assert rel(gr.c_ext_nm2, uni.c_ext_nm2).max() < 0.002
    assert rel(gr.c_sca_nm2, uni.c_sca_nm2).max() < 0.002
    assert np.abs(gr.c_abs_nm2).max() < 1e-4 * max(gr.c_ext_nm2)


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_graded_drude_sphere_matches_uniform_with_fewer_cells():
    """A small metal sphere (Drude, r = 10 nm in water, 0.5 nm cells) on the graded grid: the same spectrum as the
    uniform 0.5 nm grid to well under its discretisation error, with far fewer cells. Measured: 488k cells against
    1.74M (2.1x faster in float64, 1.1x in float32, where the per-step source overhead dominates); the C_ext spectra
    differ by at most 7e-4 of the peak, against the uniform run's own 13% peak error against Mie."""
    sim = sphere_sim(DRUDE, 10, list(np.linspace(450.0, 650.0, 9)), medium=1.33, precision="double")
    mie = ls.solve(sim, "mie")
    uni = ls.solve(sim, "fdtd", "gpu")
    gr = ls.solve(sim.model_copy(update={"options": {"fdtd": {**sim.options["fdtd"], "grid": "auto"}}}), "fdtd", "gpu")
    assert gr.provenance["cells"] * 3 < uni.provenance["cells"]
    peak = max(mie.c_ext_nm2)
    err_u = np.abs(np.asarray(uni.c_ext_nm2) - mie.c_ext_nm2).max() / peak
    assert err_u > 0.05  # the discretisation error at r / 20
    assert np.abs(np.asarray(gr.c_ext_nm2) - uni.c_ext_nm2).max() / peak < 0.05 * err_u


SUBSTRATE_CASES = {
    "glass, from below (column source)": (Substrate(material=ls.ConstantIndex(n_real=1.5)), None),
    "glass, from above (column source)": (Substrate(material=ls.ConstantIndex(n_real=1.5)),
                                         ls.PlaneWave(direction=(0, 0, -1), polarization=(1, 0, 0))),
    "absorbing, from above (tables, particle absorption)": (
        Substrate(material=ls.ConstantIndex(n_real=1.5, n_imag=0.3)),
        ls.PlaneWave(direction=(0, 0, -1), polarization=(1, 0, 0))),
}


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("case", list(SUBSTRATE_CASES))
def test_graded_sphere_on_substrate_matches_uniform(case):
    """An n = 3.5, r = 20 nm sphere touching a substrate (the interface inside the refined zone): the graded run
    agrees with the uniform 2 nm run. Measured (pointwise, against 0.6-0.8% discretisation error): glass from below
    C_ext 0.16%, C_sca 0.13%; from above 0.20%, 0.12%; absorbing substrate 0.34%, 0.05%; C_abs within 1.1e-4 of
    C_ext. 228k cells against 342k."""
    sub, ill = SUBSTRATE_CASES[case]
    kw = {} if ill is None else {"illumination": ill}
    sim = ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=ls.ConstantIndex(n_real=3.5),
                                               position_nm=(0, 0, 20))],
                        substrate=sub, wavelengths=list(np.linspace(450.0, 750.0, 7)),
                        options={"fdtd": {"precision": "double"}}, **kw)
    uni = ls.solve(sim, "fdtd", "gpu")
    gr = ls.solve(sim.model_copy(update={"options": {"fdtd": {"precision": "double", "grid": "auto"}}}), "fdtd", "gpu")
    assert "graded_grid" in gr.provenance and gr.provenance["cells"] < uni.provenance["cells"]
    for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        assert max_diff(gr, uni, k) < 0.005, k


@pytest.mark.gpu
@needs_cuda
def test_graded_far_and_near_fields_match_mie():
    """Far field (near-to-far-field surface weighted by the cells' widths) and near field (world coordinates) of the
    n = 3.5 sphere on the graded grid (2 nm inside, 6.7 nm base)."""
    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=ls.ConstantIndex(n_real=3.5))],
        wavelengths=[450.0, 600.0, 750.0], options={"fdtd": {"precision": "double", "grid": "auto"}},
        outputs=Outputs(far_field=FarFieldRequest(theta_deg=list(range(0, 181, 15)), phi_deg=[0, 45, 90],
                                                  wavelengths_nm=[450.0, 600.0]),
                        near_field=NearFieldRequest(plane="xz", half_width_nm=40, points=41, wavelengths_nm=[600.0])))
    res, mie = ls.solve(sim, "fdtd", "gpu"), ls.solve(sim, "lightsmith-mie")
    assert "graded_grid" in res.provenance
    ff = np.concatenate([rel(a.dcsca_domega_nm2, b.dcsca_domega_nm2).ravel() for a, b in zip(res.far_field, mie.far_field)])
    assert np.median(ff) < 0.01 and ff.max() < 0.012  # measured 0.75% and 0.79% (uniform 2 nm: 0.68%, 0.72%)
    nf = res.near_field[0]
    ne, nm = np.asarray(nf.enhancement), np.asarray(mie.near_field[0].enhancement)
    u, v = np.meshgrid(nf.axis1_nm, nf.axis2_nm, indexing="ij")
    ok = np.isfinite(nm) & (np.hypot(u, v) > 24.0)  # away from the surface, where |E|^2 jumps by eps^2 over a cell
    assert np.isfinite(ne[ok]).all()
    assert np.sqrt(np.mean((ne[ok] / nm[ok] - 1) ** 2)) < 0.02  # measured 0.47%


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("material", ["dielectric", "metal"])
def test_graded_grid_is_stable_at_late_times(material):
    """No growth at grading transitions: a conformal n = 3.5 sphere (3 ps) and a Drude sphere through the vertex
    tensor (1.5 ps) on grids graded 3 -> 10 nm and 3 -> 8 nm at ratio 1.25 decay to rounding: measured 5e-20 after
    547k steps and 9e-21 after 273k steps in double precision."""
    mat, base, T = ((ls.ConstantIndex(n_real=3.5), 10.0, 3000) if material == "dielectric" else (DRUDE, 8.0, 1500))
    sim = sphere_sim(mat, 40, [500.0, 700.0, 900.0], spacing_nm=base, mesh_override=override(100.0, 3.0),
                     run_time_fs=T, max_time_fs=T, precision="double")
    res = ls.solve(sim, "fdtd", "gpu")
    assert res.provenance["graded_grid"]["max_ratio"] <= 1.25 * (1 + 1e-9) and res.provenance["steps"] > 200_000
    assert res.provenance["decay_level"] < 1e-15
