"""Native 2D FDFD backend: ports, S-parameters, modes, adjoint gradients, GPU agreement."""

import math

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.device import _torch_cuda
from lightsmith.problems import DeviceProblem, Polygon, Port, Prism, Rectangle, Structure

pytest.importorskip("scipy")

SI = ls.ConstantIndex(n_real=3.48)
SIO2 = ls.ConstantIndex(n_real=1.44)
SIN = ls.ConstantIndex(n_real=2.0)


def prism(profile, material=SI):
    return Structure(geometry=Prism(profile=profile, z_min_nm=-110, z_max_nm=110), material=material)


def rect(cx, cy, sx, sy, material=SI):
    return prism(Rectangle(center_nm=(cx, cy), size_nm=(sx, sy)), material)


def straight(pol="te", dl=25.0, width=500.0, wavelengths=(1500.0, 1550.0, 1600.0), num_modes=1, length=2400.0):
    return DeviceProblem(
        dimension=2, background=SIO2, structures=[rect(0, 0, 20000, width)],
        domain_size_nm=(length + 600, 2500, 0), wavelengths=list(wavelengths),
        ports=[Port(name="o1", center_nm=(-length / 2, 0, 0), size_nm=(0, 2000, 1), direction="+", num_modes=num_modes),
               Port(name="o2", center_nm=(length / 2, 0, 0), size_nm=(0, 2000, 1), direction="-", num_modes=num_modes)],
        options={"fdfd": {"polarization": pol, "dl_nm": dl}})


class _Gpu:
    """resolve_device stand-in: a GPU for "auto" and "gpu", whatever this machine has."""

    def __init__(self, monkeypatch, free_bytes):
        from lightsmith import api
        from lightsmith.device import Device, resolve_device

        gpu = Device("torch", "cuda:0")
        monkeypatch.setattr(api, "resolve_device", lambda d: resolve_device("cpu") if d == "cpu" else gpu)
        monkeypatch.setattr(Device, "memory_free_bytes", lambda self: free_bytes)
        self.api, self.gpu = api, gpu


def test_auto_leaves_a_gpu_the_problem_does_not_fit(monkeypatch):
    # The block LU that FDFD uses on a GPU can need 20x SuperLU's memory: the example examples/pdk_mzi.py needs
    # 12.9 GB against 0.8 GB, and on a 12 GB GPU under Windows it pages for hours instead of failing.
    prob = straight(wavelengths=(1550.0,))
    backend = ls.get_backend("fdfd")
    free = 1000
    g = _Gpu(monkeypatch, free)
    assert backend.estimate(prob, g.gpu)["memory_bytes"] > free
    dev, note = g.api._select_device(backend, prob, "auto")
    assert dev.name == "cpu" and "ran on the CPU" in note and "cuda:0" in note

    res = ls.solve(prob, "fdfd")
    assert res.provenance["device"] == "cpu" and any("ran on the CPU" in w for w in res.warnings)
    assert abs(res.s()[0, 1, 0]) ** 2 > 0.95


def test_auto_keeps_a_gpu_the_problem_fits_and_naming_a_device_is_obeyed(monkeypatch):
    prob = straight(wavelengths=(1550.0,))
    backend = ls.get_backend("fdfd")
    g = _Gpu(monkeypatch, 2**50)
    assert g.api._select_device(backend, prob, "auto") == (g.gpu, None)

    _Gpu(monkeypatch, 1000)
    assert g.api._select_device(backend, prob, "gpu") == (g.gpu, None)  # asked for by name


def test_auto_stays_on_the_gpu_when_its_free_memory_is_unknown(monkeypatch):
    prob = straight(wavelengths=(1550.0,))
    g = _Gpu(monkeypatch, None)
    assert g.api._select_device(ls.get_backend("fdfd"), prob, "auto") == (g.gpu, None)


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_straight_waveguide_transmits(pol):
    from lightsmith.backends.fdfd import FDFD

    prob = straight(pol)
    res = ls.solve(prob, method="fdfd", device="cpu")
    S = res.s()
    assert res.ports == ["o1", "o2"] and S.shape == (3, 2, 2)
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), 1, atol=1e-4)
    np.testing.assert_allclose(np.abs(S[:, 0, 1]), 1, atol=1e-4)
    assert np.abs(S[:, 0, 0]).max() < 1e-3 and np.abs(S[:, 1, 1]).max() < 1e-3
    # Phase = discrete beta * distance between the port planes.
    eng = FDFD.engine(prob)
    L = (eng.ports[1].i0 - eng.ports[0].i0) * eng.grid.dl
    for k, lam in enumerate(eng.wavelengths):
        beta_dl = eng.modes(eng.base_eps(lam), lam)[0][0].beta_dl.real
        phase = np.angle(S[k, 1, 0] * np.exp(-1j * beta_dl * L / eng.grid.dl))
        assert abs(phase) < 1e-3
    assert res.provenance["solver"] == "scipy-superlu"
    assert not res.warnings


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_slab_mode_matches_analytic_neff(pol):
    from lightsmith.backends._fdfd import slab_neff_analytic
    from lightsmith.backends.fdfd import FDFD

    exact = slab_neff_analytic(3.48, 1.44, 500, 1550, pol)
    errs = []
    for dl in (10.0, 5.0):
        eng = FDFD.engine(straight(pol, dl=dl, wavelengths=[1550.0]))
        neff = eng.modes(eng.base_eps(1550.0), 1550.0)[0][0].neff.real
        errs.append(abs(neff - exact) / exact)
    assert errs[1] < 1e-3
    assert 3 < errs[0] / errs[1] < 5  # second-order convergence


def test_multimode_ports_are_orthogonal():
    res = ls.solve(straight("te", dl=25, width=1100, wavelengths=[1550.0], num_modes=2), method="fdfd", device="cpu")
    assert res.ports == ["o1@0", "o1@1", "o2@0", "o2@1"]
    S = np.abs(res.s()[0])
    np.testing.assert_allclose(S[2, 0], 1, atol=1e-4)  # o1@0 -> o2@0
    np.testing.assert_allclose(S[3, 1], 1, atol=1e-4)  # o1@1 -> o2@1
    assert S[3, 0] < 1e-4 and S[2, 1] < 1e-4  # no mode conversion in a straight guide


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_power_conservation_closed_guide(pol):
    """Parallel-plate guide (no PML across it): all power is in the port modes, so S is unitary.

    Two propagating modes; the first evanescent mode decays within ~200 nm, and the
    ports sit 800 nm from the obstacles, so evanescent tunnelling is negligible.
    """
    prob = DeviceProblem(
        dimension=2, background=SIO2, structures=[rect(100, 150, 300, 400, SIN), rect(-150, -250, 200, 300, SI)],
        domain_size_nm=(2800, 1000, 0), wavelengths=[1200.0, 1300.0],
        ports=[Port(name="a", center_nm=(-1100, 0, 0), size_nm=(0, 1200, 1), direction="+", num_modes=2),
               Port(name="b", center_nm=(1100, 0, 0), size_nm=(0, 1200, 1), direction="-", num_modes=2)],
        options={"fdfd": {"polarization": pol, "dl_nm": 20, "pml_cells": [16, 0]}})
    S = ls.solve(prob, method="fdfd", device="cpu").s()
    for Sk in S:
        np.testing.assert_allclose((np.abs(Sk) ** 2).sum(axis=0), 1, atol=1e-6)
        np.testing.assert_allclose(Sk, Sk.T, atol=1e-8)  # reciprocity
        assert np.abs(Sk[2:, :2]).max() < 0.99 and np.abs(Sk[:2, :2]).max() > 0.05  # real scattering


def _bend(radius, width=450.0, pol="te", dl=30.0, core=SI):
    n = 48
    t = np.linspace(-np.pi / 2, 0, n)
    outer = [((radius + width / 2) * np.cos(a), radius + (radius + width / 2) * np.sin(a)) for a in t]
    inner = [((radius - width / 2) * np.cos(a), radius + (radius - width / 2) * np.sin(a)) for a in t[::-1]]
    lead = 900.0
    x0, x1 = -lead - 400, radius + width / 2 + 500
    y0, y1 = -width / 2 - 500, radius + lead + 400
    return DeviceProblem(
        dimension=2, background=SIO2,
        structures=[rect(-5000, 0, 10000, width, core), prism(Polygon(vertices_nm=outer + inner), core),
                    rect(radius, radius + 5000, width, 10000, core)],
        domain_center_nm=((x0 + x1) / 2, (y0 + y1) / 2, 0),
        domain_size_nm=(x1 - x0, y1 - y0, 0), wavelengths=[1550.0],
        ports=[Port(name="o1", center_nm=(-lead, 0, 0), size_nm=(0, 2000, 1), direction="+"),
               Port(name="o2", center_nm=(radius, radius + lead, 0), size_nm=(2000, 0, 1), direction="-")],
        options={"fdfd": {"polarization": pol, "dl_nm": dl}})


def test_bend_transmission_grows_with_radius():
    t = []
    for r in (400.0, 1500.0):
        S = ls.solve(_bend(r), method="fdfd", device="cpu").s()[0]
        t.append(abs(S[1, 0]) ** 2)
        assert abs(S[1, 0]) ** 2 + abs(S[0, 0]) ** 2 <= 1 + 1e-6  # radiation only removes power
    assert t[1] > t[0] + 0.2 and t[1] > 0.95


def test_y_splitter_is_symmetric():
    w, sep = 500.0, 1000.0
    prob = DeviceProblem(
        dimension=2, background=SIO2,
        structures=[rect(-5000, 0, 10000, w), rect(600, 0, 1200, 2 * sep / 2 + w + 400),
                    rect(5000 + 1200, sep / 2, 10000, w), rect(5000 + 1200, -sep / 2, 10000, w)],
        domain_size_nm=(3600, 3000, 0), domain_center_nm=(600, 0, 0), wavelengths=[1550.0],
        ports=[Port(name="o1", center_nm=(-800, 0, 0), size_nm=(0, 1600, 1), direction="+"),
               Port(name="o2", center_nm=(2000, sep / 2, 0), size_nm=(0, 1000, 1), direction="-"),
               Port(name="o3", center_nm=(2000, -sep / 2, 0), size_nm=(0, 1000, 1), direction="-")],
        options={"fdfd": {"dl_nm": 25}})
    S = ls.solve(prob, method="fdfd", device="cpu").s()[0]
    p2, p3 = abs(S[1, 0]) ** 2, abs(S[2, 0]) ** 2
    assert abs(p2 - p3) < 1e-6
    assert 0.1 < p2 < 0.5 and p2 + p3 + abs(S[0, 0]) ** 2 <= 1 + 1e-6


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_adjoint_gradient_matches_finite_differences(pol):
    torch = pytest.importorskip("torch")
    from lightsmith.backends.fdfd import FDFD

    prob = straight(pol, dl=40.0, wavelengths=[1500.0, 1600.0], length=1200.0)
    ctx = FDFD.prepare(prob)
    base = torch.as_tensor(FDFD.base_permittivity(prob))
    pert = torch.as_tensor(np.random.default_rng(1).uniform(0, 3, base.shape[1:]))
    mask = torch.zeros(base.shape[1:], dtype=torch.float64)
    mask[25:45, 20:50] = 1  # perturb only away from the ports

    def objective(p):
        sol = FDFD.differentiable_solve(prob, base + p * mask, "cpu", sources=["o1"], **ctx)
        return (sol.sparam("o2", "o1").abs() ** 2).sum() + 0.5 * (sol.fields[:, 0, 35, 35].abs() ** 2).sum()

    x = pert.clone().requires_grad_(True)
    objective(x).backward()
    g = x.grad.numpy()
    h = 1e-4
    for i, j in [(30, 30), (35, 37), (40, 25), (28, 45)]:
        xp, xm = pert.clone(), pert.clone()
        xp[i, j] += h
        xm[i, j] -= h
        with torch.no_grad():
            fd = (float(objective(xp)) - float(objective(xm))) / (2 * h)
        assert abs(g[i, j] - fd) <= 1e-3 * abs(fd) + 1e-12, (i, j, g[i, j], fd)


def test_block_solver_matches_scipy_on_cpu():
    torch = pytest.importorskip("torch")
    from lightsmith.backends._fdfd import linsolve
    from lightsmith.backends.fdfd import FDFD

    for size in ((1600, 1000, 0), (1000, 1600, 0)):  # blocks along x, then along y
        prob = DeviceProblem(dimension=2, background=SIO2, structures=[rect(0, 0, 400, 300)],
                             domain_size_nm=size, wavelengths=[1550.0],
                             ports=[Port(name="o1", center_nm=(-300, 0, 0), size_nm=(0, 600, 1), direction="+")],
                             options={"fdfd": {"dl_nm": 50, "pml_cells": 8}})
        eng = FDFD.engine(prob)
        eps = eng.base_eps(1550.0)
        import array_api_compat.torch as xp

        d = eng.diagonals(xp, torch.as_tensor(eps), 1550.0, device=torch.device("cpu"))
        d = linsolve.Diagonals(*(torch.broadcast_to(v, eps.shape).contiguous() for v in d))
        B = torch.as_tensor(np.random.default_rng(0).normal(size=eps.shape + (2,)) + 0j)
        sc, bl = linsolve.ScipyFactor(d), linsolve.BlockFactor(d)
        np.testing.assert_allclose(bl.solve(B).numpy(), sc.solve(B).numpy(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(bl.solve_t(B).numpy(), sc.solve_t(B).numpy(), rtol=1e-9, atol=1e-9)


def test_estimate_and_registration():
    info = ls.estimate(straight(), method="fdfd", device="cpu")
    assert info["backend"] == "lightsmith-fdfd" and info["unknowns"] == info["grid"][0] * info["grid"][1]
    b = ls.get_backend("fdfd", kind="device")
    assert b.differentiable and "device" in b.problem_kinds
    ok, why = b.supports(straight().model_copy(update={"dimension": 3}))
    assert ok, why  # 3D devices go to the full-vector engine (tests/test_fdfd3d.py)


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda(), reason="needs CUDA")
def test_gpu_matches_cpu():
    torch = pytest.importorskip("torch")
    from lightsmith.backends.fdfd import FDFD

    prob = straight("te", dl=30, wavelengths=[1550.0], length=1200.0)
    s_cpu = ls.solve(prob, method="fdfd", device="cpu")
    s_gpu = ls.solve(prob, method="fdfd", device="cuda")
    assert s_gpu.provenance["solver"] == "torch-block-tridiagonal"
    np.testing.assert_allclose(s_gpu.s(), s_cpu.s(), atol=1e-10)

    base = torch.as_tensor(FDFD.base_permittivity(prob))
    grads = []
    for dev in ("cpu", "cuda"):
        x = (base.real * 0 + 0.5).to(dev).requires_grad_(True)
        sol = FDFD.differentiable_solve(prob, base.to(dev) + x, dev, sources=["o1"])
        (sol.sparam("o2", "o1").abs() ** 2).sum().backward()
        grads.append(x.grad.cpu().numpy())
    np.testing.assert_allclose(grads[1], grads[0], rtol=1e-6, atol=1e-12 * np.abs(grads[0]).max())
    assert math.isfinite(float(np.abs(grads[0]).max()))


def test_gds_layout_device(tmp_path):
    """A DeviceProblem defined by a GDS layout (via lightsmith.layout) matches the explicit structures."""
    from lightsmith.layout import write_gds
    from lightsmith.problems import GdsLayout

    path = write_gds(tmp_path / "wg.gds", {"TOP": {"polygons": {
        (1, 0): [np.array([[-10000, -250], [10000, -250], [10000, 250], [-10000, 250]], float)]}}})
    explicit = straight("te", dl=30, wavelengths=[1550.0], length=1200.0)
    layout = explicit.model_copy(update={"structures": [], "layout": GdsLayout(path=str(path), layer_map={
        "1/0": {"material": SI.model_dump(), "z_min_nm": -110, "z_max_nm": 110}})})
    s1 = ls.solve(explicit, method="fdfd", device="cpu").s()
    s2 = ls.solve(layout, method="fdfd", device="cpu").s()
    np.testing.assert_allclose(s2, s1, atol=1e-12)


def test_mcp_device_example_runs_on_fdfd():
    """A 2D port with z size 0, as the Port schema documents and the MCP server's example uses."""
    from lightsmith.mcp_server.kinds import EXAMPLES

    res = ls.solve(EXAMPLES["device"], "fdfd", "cpu")
    assert abs(abs(res.s()[0, 1, 0]) - 1) < 1e-2  # a straight guide


def test_fdfd_refuses_a_port_normal_to_z():
    p = ls.problems.parse_problem(
        {"kind": "device", "dimension": 2, "background": {"kind": "constant_index", "n_real": 1.44},
         "domain_size_nm": [2000, 2000, 0], "wavelengths": [1550],
         "ports": [{"name": "o1", "center_nm": [0, 0, 0], "size_nm": [500, 500, 0], "direction": "+"}]})
    ok, why = ls.get_backend("fdfd").supports(p)
    assert not ok and "normal to z" in why


def test_fdfd_estimate_reports_memory_and_the_budget_refuses_it():
    from lightsmith.mcp_server.budget import check
    from lightsmith.mcp_server.config import ServerConfig
    from lightsmith.mcp_server.kinds import EXAMPLES

    prob = ls.problems.parse_problem(EXAMPLES["device"])
    est = ls.estimate(prob, "fdfd", "cpu")
    assert est["memory_bytes"] == est["memory_bytes_scipy"] > 0 and est["precision"] in ("single", "double")
    verdict = check(est, prob, ServerConfig(max_memory_bytes=1))
    assert not verdict["ok"] and any("memory" in r for r in verdict["reasons"])
