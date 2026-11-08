"""FDTD device problems: exact discrete port modes, S-parameters against FDFD (2D) and physical limits (3D)."""

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends.fdtd import NativeFDTD as fdtd
from lightsmith.device import _torch_cuda
from lightsmith.problems import Box, DeviceProblem, Port, Prism, Rectangle, Structure

pytest.importorskip("scipy")
needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="needs PyTorch with CUDA")

SI = ls.ConstantIndex(n_real=3.48)
SIO2 = ls.ConstantIndex(n_real=1.44)


def rect(cx, cy, sx, sy, material=SI):
    return Structure(geometry=Prism(profile=Rectangle(center_nm=(cx, cy), size_nm=(sx, sy)), z_min_nm=-110,
                                    z_max_nm=110), material=material)


def device_2d(structures, pol, dl, length=2400.0, wavelengths=(1500.0, 1550.0, 1600.0), num_modes=1, window=3600):
    return DeviceProblem(
        dimension=2, background=SIO2, structures=structures, domain_size_nm=(length + 600, window + 400, 0),
        wavelengths=list(wavelengths),
        ports=[Port(name="o1", center_nm=(-length / 2, 0, 0), size_nm=(0, window, 1), direction="+",
                    num_modes=num_modes),
               Port(name="o2", center_nm=(length / 2, 0, 0), size_nm=(0, window, 1), direction="-",
                    num_modes=num_modes)],
        options={"fdfd": {"polarization": pol, "dl_nm": dl},
                 "fdtd": {"spacing_nm": dl, "polarization": pol, "precision": "double"}})


def test_device_supports_and_estimate():
    ok, _ = fdtd.supports(device_2d([rect(0, 0, 20000, 500)], "te", 25))
    assert ok
    p = device_2d([rect(0, 0, 20000, 500)], "te", 25)
    bad = p.model_copy(update={"ports": [Port(name="v", center_nm=(0, 0, 0), size_nm=(500, 500, 0), direction="+")]})
    assert "normal to x or y" in fdtd.supports(bad)[1]
    est = fdtd.estimate(p, ls.resolve_device("cpu"))
    assert est["runs"] == 2 and est["fits"] and est["grid"][2] == 3


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_straight_waveguide_2d_is_exact_on_the_grid_cpu(pol):
    """A straight guide transmits fully with the phase of the grid's own discrete beta: the port modes are exact
    for the FDTD grid, so nothing reflects and no other mode is launched (to the CPML's reflection, ~4e-5)."""
    p = device_2d([rect(0, 0, 20000, 500)], pol, 25.0)
    r = ls.solve(p, "fdtd", "cpu")
    S = r.s()
    assert r.ports == ["o1", "o2"] and not r.warnings
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), 1, atol=1e-4)
    np.testing.assert_allclose(S[:, 1, 0], S[:, 0, 1], atol=1e-6)
    assert np.abs(S[:, 0, 0]).max() < 1e-4
    neff = np.asarray(r.provenance["ports"]["o1"]["neff"])[:, 0]
    k0 = 2 * np.pi / np.asarray(r.wavelengths_nm)
    phase = np.angle(S[:, 1, 0] * np.exp(-1j * neff * k0 * 2400.0))
    assert np.abs(phase).max() < 1e-4


def test_multimode_ports_2d_cpu():
    p = device_2d([rect(0, 0, 20000, 1100)], "te", 25.0, wavelengths=[1550.0], num_modes=2)
    r = ls.solve(p, "fdtd", "cpu")
    S = np.abs(r.s()[0])
    assert r.ports == ["o1@0", "o1@1", "o2@0", "o2@1"]
    np.testing.assert_allclose([S[2, 0], S[3, 1]], 1, atol=1e-4)
    assert max(S[3, 0], S[2, 1], S[0, 0], S[1, 1]) < 1e-4


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_width_step_2d_matches_fdfd(pol):
    """A 500 -> 900 nm silicon width step at dl = 10 nm: |S| within 1e-4 of FDFD on the same grid spacing for TE
    (measured 3e-5) and 5e-4 for TM (measured 3.4e-4 in |S22|, from 1.8e-3 at dl = 20 nm: the two discretisations
    converge to each other). Phases agree once FDFD's conventions are applied: its reference planes sit half a cell before the port
    centres, and its TE amplitudes are those of H_z (the reflections change sign)."""
    p = device_2d([rect(-5000, 0, 10000, 500), rect(5000, 0, 10000, 900)], pol, 10.0)
    res = ls.solve(p, "fdtd", "gpu")
    S = res.s()
    ref = ls.solve(p, "fdfd", "cpu")
    F = ref.s()
    np.testing.assert_allclose(np.abs(S), np.abs(F), atol=1e-4 if pol == "te" else 5e-4)
    # FDFD's S referred to the port centres: outgoing ~ exp(-+i beta x) at a port pointing +-x, incoming the opposite
    k0 = 2 * np.pi / np.asarray(ref.wavelengths_nm)
    shift = []
    for port in p.ports:
        beta = np.asarray(res.provenance["ports"][port.name]["neff"])[:, 0] * k0
        dxp = port.center_nm[0] - ref.provenance["port_planes_nm"][port.name]
        sgn = 1 if port.direction == "+" else -1
        shift.append((np.exp(-1j * sgn * beta * dxp), np.exp(-1j * sgn * beta * dxp)))  # (outgoing, 1 / incoming)
    G = F.copy()
    for o in range(2):
        for i in range(2):
            G[:, o, i] *= shift[o][0] * shift[i][1]
    if pol == "te":
        G[:, [0, 1], [0, 1]] *= -1
    assert np.abs(np.angle(S / G)).max() < 1e-2  # measured 3.5e-3: the leapfrog's dispersion (FDFD has none)


SI3, SIO23 = ls.ConstantIndex(n_real=3.476), ls.ConstantIndex(n_real=1.444)


def wire_3d(structures, length=2000.0, window=(2400.0, 2400.0), num_modes=1, dx=25.0):
    w, h = window
    return DeviceProblem(
        structures=structures, background=SIO23, domain_size_nm=(length, w, h), wavelengths=[1500, 1550, 1600],
        ports=[Port(name="o1", center_nm=(-length / 2, 0, 0), size_nm=(0, w, h), direction="+", num_modes=num_modes),
               Port(name="o2", center_nm=(length / 2, 0, 0), size_nm=(0, w, h), direction="-", num_modes=num_modes)],
        options={"fdtd": {"spacing_nm": dx}, "eme": {"num_modes": 240, "grid_spacing_nm": dx}})


@pytest.mark.gpu
@needs_cuda
def test_straight_wire_3d_is_exact_on_the_grid():
    """500 x 220 nm silicon wire, TE0 and TM0 ports, float32: TE0 transmits with |S21| = 1 and the grid's phase
    (measured 1e-6 rad), reflections and mode conversion at the CPML's level (measured 4e-5). TM0, weakly guided
    (neff 1.77 in silica), loses up to 5e-4 to the window's edges."""
    r = ls.solve(wire_3d([Structure(geometry=Box(size_nm=(3000, 500, 220)), material=SI3)], num_modes=2), "fdtd",
                 "gpu")
    S = r.s()
    assert r.ports == ["o1@0", "o1@1", "o2@0", "o2@1"] and not r.warnings
    np.testing.assert_allclose(np.abs(S[:, 2, 0]), 1, atol=1e-4)
    np.testing.assert_allclose(np.abs(S[:, 3, 1]), 1, atol=1e-3)
    neff = np.asarray(r.provenance["ports"]["o1"]["neff"])
    k0 = 2 * np.pi / np.asarray(r.wavelengths_nm)
    assert np.abs(np.angle(S[:, 2, 0] * np.exp(-1j * neff[:, 0] * k0 * 2000))).max() < 1e-4
    off = np.abs(S.copy())
    off[:, [2, 3, 0, 1], [0, 1, 2, 3]] = 0
    assert off.max() < 2e-4
    assert np.abs(S - np.swapaxes(S, 1, 2)).max() < 1e-4


def _width_step():
    return wire_3d([Structure(geometry=Box(center_nm=(-1000, 0, 0), size_nm=(2000, 500, 220)), material=SI3),
                    Structure(geometry=Box(center_nm=(1000, 0, 0), size_nm=(2000, 700, 220)), material=SI3)],
                   window=(2000.0, 1500.0))


@pytest.mark.gpu
@needs_cuda
def test_width_step_3d_is_reciprocal_passive_and_converged():
    """500 -> 700 nm silicon wire step: |S21| = 0.99753 at 1550 nm and dx = 25 nm (0.99739 at dx = 15 nm; EME tends
    to it slowly, 0.9972 with 240 modes in a 3 um box), reciprocal to 1e-4."""
    S = ls.solve(_width_step(), "fdtd", "gpu").s()
    assert abs(abs(S[1, 1, 0]) - 0.99753) < 5e-4
    assert np.abs(S - np.swapaxes(S, 1, 2)).max() < 2e-4
    assert np.all((np.abs(S) ** 2).sum(axis=1) < 1 + 1e-4)
    assert 0.014 < np.abs(S[:, 0, 0]).min() and np.abs(S[:, 0, 0]).max() < 0.022  # 1.6-1.9 % reflected amplitude


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_pdk_mmi_1x2_in_3d():
    """The PDK's 1x2 MMI on 220 nm SOI (11 M cells at dx = 25 nm, float32): an even split, |S21| = |S31| = 0.7008
    (EME with 20 modes: 0.7012), reciprocal to 3e-4 and passive. Light sent back into an output port reflects off
    the MMI's input facet and returns to both outputs (|S22| = |S32| = 0.155; 20-mode EME misses it, 0.03)."""
    from lightsmith.pdk import generic_soi, get_component

    p = get_component("mmi1x2").device_problem(generic_soi(), [1500.0, 1550.0, 1600.0], dimension=3,
                                               port_margin_nm=800, padding_nm=800)
    p = p.model_copy(update={"options": {**p.options, "fdtd": {"spacing_nm": 25.0}}})
    r = ls.solve(p, "fdtd", "gpu")
    S = r.s()
    assert r.ports == ["o1", "o2", "o3"]
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), np.abs(S[:, 2, 0]), atol=1e-3)
    assert abs(abs(S[1, 1, 0]) - 0.7012) < 3e-3
    assert np.abs(S - np.swapaxes(S, 1, 2)).max() < 5e-4
    assert np.all((np.abs(S) ** 2).sum(axis=1) < 1 + 1e-3)
    assert 0.12 < np.abs(S[1, 1, 1]) < 0.19


# --- graded grids ------------------------------------------------------------------------------------------------

STEP = [rect(-5000, 0, 10000, 500), rect(5000, 0, 10000, 900)]
CORE = {"center_nm": (0, 0, 0), "size_nm": (1e5, 1200, 1e5), "spacing_nm": 10.0}  # across both guides, all along x


def test_graded_device_grid_layout():
    """Ports sit in uniform cells along their normals (here next to an override box that ends between them), the
    CPML is uniform, cells grow by at most grading_ratio, and 'auto' refines silicon to cells_per_wavelength in
    silicon on a base of cells_per_wavelength in silica. Without graded options the grid is the uniform one."""
    from lightsmith.backends._fdtd import device as devices
    from lightsmith.backends._fdtd.options import FDTDOptions

    p = device_2d(STEP, "te", 25.0)
    box = {"center_nm": (0, 0, 0), "size_nm": (600, 1200, 1e5), "spacing_nm": 8.0}
    opts = FDTDOptions.model_validate({**p.options["fdtd"], "mesh_override": [box]})
    g = devices.make_grid(p, opts)
    assert g.graded is not None and g.graded.max_ratio() <= opts.grading_ratio * (1 + 1e-9)
    assert g.min_spacing(0) == pytest.approx(8.0) and g.min_spacing(1) == pytest.approx(8.0)
    for f in devices.port_frames(p, g):  # raises if a port's cells are not uniform
        assert f.ds == pytest.approx(25.0) and f.widths[0] is not None
    for a in (0, 1):
        w = g.graded.prim(a)
        assert np.ptp(w[:g.n_pml + 1]) == 0 and np.ptp(w[-(g.n_pml + 2):]) == 0
        lo, hi = g.interior[a]
        assert g.coord(a, lo) < p.domain_center_nm[a] - p.domain_size_nm[a] / 2
        assert g.coord(a, hi) > p.domain_center_nm[a] + p.domain_size_nm[a] / 2
    auto = devices.make_grid(p, FDTDOptions.model_validate({"grid": "auto"}))
    assert auto.dx == pytest.approx(1500 / (1.44 * 20))
    assert auto.min_spacing(1) == pytest.approx(1500 / (3.48 * 20))
    assert devices.make_grid(p, FDTDOptions.model_validate(p.options["fdtd"])).graded is None
    est = fdtd.estimate(p.model_copy(update={"options": {"fdtd": {**p.options["fdtd"], "mesh_override": [box]}}}),
                        ls.resolve_device("cpu"))
    assert est["graded_grid"]["spacing_min_nm"][0] == pytest.approx(8.0)


def test_graded_straight_waveguide_2d_is_exact_on_the_grid_cpu():
    """On a grid graded from 25 to 10 nm across the guide (y) the port modes are still exact for the grid: |S21| = 1
    with the grid's phase (measured 1e-10 rad) and reflection at the CPML's level (measured 3e-5)."""
    p = device_2d([rect(0, 0, 20000, 500)], "te", 25.0)
    across = {**CORE, "spacing_nm": (25.0, 10.0, 25.0)}  # graded across the guide only
    p = p.model_copy(update={"options": {"fdtd": {**p.options["fdtd"], "mesh_override": [across]}}})
    r = ls.solve(p, "fdtd", "cpu")
    S = r.s()
    assert not r.warnings and r.provenance["graded_grid"]["spacing_min_nm"][1] == pytest.approx(10.0)
    np.testing.assert_allclose(np.abs(S[:, 1, 0]), 1, atol=1e-4)
    assert np.abs(S[:, 0, 0]).max() < 1e-4
    neff = np.asarray(r.provenance["ports"]["o1"]["neff"])[:, 0]
    k0 = 2 * np.pi / np.asarray(r.wavelengths_nm)
    assert np.abs(np.angle(S[:, 1, 0] * np.exp(-1j * neff * k0 * 2400.0))).max() < 1e-4


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_graded_width_step_2d_matches_uniform_fine_grid(pol):
    """The 500 -> 900 nm step with 25 nm cells graded to 10 nm across the guides: |S| within 1e-4 of a uniform
    10 nm run (measured 1.8e-5 TE, 8e-6 TM) with 63 % of its cells; uniform 25 nm cells err by 2e-3."""
    fine = device_2d(STEP, pol, 10.0)
    p = device_2d(STEP, pol, 25.0)
    p = p.model_copy(update={"options": {"fdtd": {**p.options["fdtd"], "mesh_override": [CORE]}}})
    a, b = ls.solve(p, "fdtd", "gpu"), ls.solve(fine, "fdtd", "gpu")
    assert a.provenance["cells"] < 0.7 * b.provenance["cells"]
    np.testing.assert_allclose(np.abs(a.s()), np.abs(b.s()), atol=1e-4)


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_graded_wire_step_3d_matches_uniform_fine_grid():
    """500 -> 700 nm wire step, 40 nm cells graded to 15 nm in a box around the wires (all along x): |S| of the
    uniform 15 nm run (7.3 M cells, measured once: |S21| 0.99666, 0.99737, 0.99768 and |S11| 0.0168, 0.0181,
    0.0193 at 1500, 1550, 1600 nm) within 3e-5 (measured 8e-6) from 3.3 M cells; uniform 25 nm errs by 1e-3."""
    box = {"center_nm": (0, 0, 0), "size_nm": (1e5, 900, 400), "spacing_nm": 15.0}
    st = [Structure(geometry=Box(center_nm=(-1500, 0, 0), size_nm=(3000, 500, 220)), material=SI3),
          Structure(geometry=Box(center_nm=(1500, 0, 0), size_nm=(3000, 700, 220)), material=SI3)]
    p = wire_3d(st, dx=40.0)
    p = p.model_copy(update={"options": {"fdtd": {"spacing_nm": 40.0, "mesh_override": [box]}}})
    r = ls.solve(p, "fdtd", "gpu")
    S = np.abs(r.s())
    assert r.provenance["cells"] < 0.5 * 7293993 and not r.warnings
    np.testing.assert_allclose(S[:, 1, 0], [0.99666, 0.99737, 0.99768], atol=3e-5)
    np.testing.assert_allclose(S[:, 0, 1], S[:, 1, 0], atol=1e-5)
    np.testing.assert_allclose(S[:, 0, 0], [0.0168, 0.0181, 0.0193], atol=2e-4)


def test_device_time_step_respects_the_dispersive_limit():
    """A Drude metal on a coarse grid: the explicit ADE update would be unstable at the Courant step."""
    from lightsmith.backends._fdtd.device import plan
    from lightsmith.backends._fdtd.options import FDTDOptions
    from lightsmith.backends._fdtd.setup import dispersive_dt_limit

    metal = ls.Drude(eps_inf=1.0, wp_ev=16.0, gamma_ev=0.1)
    dx = 60.0
    p = device_2d([rect(0, 0, 20000, 500, metal)], "te", dx)
    opts = FDTDOptions(spacing_nm=dx, polarization="te", precision="double")
    out = plan(p, opts, False)
    courant_step = opts.courant * dx / 2**0.5  # eps_inf = 1 here, so only the dispersive limit can shrink it
    limit = dispersive_dt_limit(out["media"], 8 / dx**2)
    assert limit < courant_step  # the case this test is for
    assert out["dt"] <= limit
    # a dielectric device keeps the Courant step
    assert plan(device_2d([rect(0, 0, 20000, 500)], "te", dx), opts, False)["dt"] < courant_step + 1e-12
