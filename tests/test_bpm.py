"""Beam propagation method: operators against the FDE engine, then devices against FDE, EME and theory."""

import importlib.util

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends._bpm import BlockTridiag, Operator, block_matvec, symbol
from lightsmith.backends._bpm.operator import keep_masks
from lightsmith.backends._fde import make_grid, port_modes
from lightsmith.backends._fde.modes import bend_transform, build_operators
from lightsmith.backends.bpm import NativeBPM
from lightsmith.problems import Box, DeviceProblem, Polygon, Port, Prism, Structure

SI, SIO2 = ls.ConstantIndex(n_real=3.476), ls.ConstantIndex(n_real=1.444)
# 2D devices: core = slab effective index of 220 nm SOI (TE), cladding SiO2 (as in test_eme).
CORE, CLAD = ls.ConstantIndex(n_real=2.85), ls.ConstantIndex(n_real=1.44)


# NumPy inverts block stacks with threaded LAPACK calls, which crawl on a busy machine; torch batches them.
CPU_3D = "torch-cpu" if importlib.util.find_spec("torch") else "cpu"


def port2d(name, x, y=0.0, width=2000.0, direction="+", num_modes=1):
    return Port(name=name, center_nm=(x, y, 0), size_nm=(0, width, 1), direction=direction, num_modes=num_modes)


# --- building blocks ---------------------------------------------------------------------


@pytest.mark.parametrize("shape,pml,sym,bend", [((30, 1), 5, (None, None), None), ((14, 9), 3, (None, None), None),
                                                ((9, 14), 3, ("pmc", None), None), ((12, 8), 3, (None, None), 3e3),
                                                ((10, 8), 0, (None, None), None)])
def test_operator_blocks_and_solves_match_the_fde_matrices(shape, pml, sym, bend):
    """N = P Q of the FDE engine (slab, 3D, PMC mirror, bend, PEC walls); blocks by probing; PCR vs dense LU."""
    import array_api_compat.numpy as xp

    rng = np.random.default_rng(0)
    g = make_grid((0, 0), (shape[0] * 20.0, shape[1] * 20.0 if shape[1] > 1 else 0), (20, 20), sym, pml)
    eps = 1 + 3 * rng.random((3, *g.shape)) + 0.1j * rng.random((3, *g.shape))
    mu = None
    if bend:
        eps, mu = bend_transform(g, eps, bend, 0.0)
    ops = build_operators(g, eps, mu, 1550, 1e-6, 1.3)
    keep = keep_masks(g).ravel().astype(float)
    full = np.diag(keep) @ (ops.P @ ops.Q).toarray() @ np.diag(keep)
    for comps in [(0,), (1,), (0, 1)]:
        op = Operator(g, eps, mu, 1550, comps, xp, lambda a: np.asarray(a, complex), 1.3, 1e-6)
        idx = np.concatenate([np.arange(g.size) + c * g.size for c in comps])
        ref_op = full[np.ix_(idx, idx)]
        f = rng.random((2, len(comps), *g.shape)) + 1j * rng.random((2, len(comps), *g.shape))
        ref = (ref_op @ f.reshape(2, -1).T).T
        scale = np.abs(ref).max()
        assert np.abs(op.apply(f).reshape(2, -1) - ref).max() < 1e-12 * scale
        A, B, C = op.blocks(chunk=5)
        lay = op.layout
        fl = lay.to_lines(xp, f)
        assert np.abs(lay.from_lines(xp, block_matvec(xp, A, B, C, fl)).reshape(2, -1) - ref).max() < 1e-12 * scale
        r = 0.3 + 0.7j
        x = BlockTridiag(xp, -A / r, (1 + 1 / r) * np.eye(lay.nb) - B / r, -C / r).solve(fl)
        xr = np.linalg.solve((1 + 1 / r) * np.eye(len(idx)) - ref_op / r, f.reshape(2, -1).T).T
        assert np.abs(lay.from_lines(xp, x).reshape(2, -1) - xr).max() < 1e-11 * np.abs(xr).max()


def test_pade_steps_approach_the_exact_one_way_propagator():
    """|G| = 1 for real X (lossless), and the phase error of one step falls with the Padé order."""
    x = np.linspace(-0.3, 0.3, 61)
    bdw = 0.05
    exact = np.exp(1j * bdw * (np.sqrt(1 + x) - 1))
    errs = []
    for order in range(4):
        g = symbol(order, bdw, x)
        np.testing.assert_allclose(np.abs(g), 1, atol=1e-12)
        errs.append(np.abs(g - exact).max())
    assert errs[1] < errs[0] / 10 and errs[2] < errs[1] and errs[3] <= errs[2] * 1.01
    lossy = symbol(1, bdw, x + 0.05j)  # Im X > 0 (PML, loss) must decay
    assert np.all(np.abs(lossy) < 1)


# --- straight guides: lossless transmission and the FDE phase ---------------------------------


def straight2d(length=3000.0, pol="te", wavelengths=(1500, 1550)):
    return DeviceProblem(dimension=2, structures=[Structure(geometry=Box(size_nm=(length + 2000, 500, 1)), material=CORE)],
                         background=CLAD, domain_size_nm=(length, 4000, 0), wavelengths=list(wavelengths),
                         ports=[port2d("o1", -length / 2, width=4000), port2d("o2", length / 2, width=4000, direction="-")],
                         options={"bpm": {"polarization": pol}, "eme": {"polarization": pol, "num_modes": 10}})


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_straight_slab_is_lossless_with_the_fde_phase(pol):
    """|S21| = 1 to 1e-9 and the phase equals k0 n_eff L of the port mode to 1e-9 rad (port modes are exact
    eigenvectors of the BPM operator; tolerance set by the PEC port window vs the PML-lined BPM window)."""
    p = straight2d(pol=pol)
    r = ls.solve(p, "bpm", device="cpu")
    s = r.s()
    np.testing.assert_allclose(np.abs(s[:, 1, 0]), 1, atol=1e-9)
    np.testing.assert_allclose(s[:, 0, 1], s[:, 1, 0], atol=1e-12)  # both directions
    assert np.abs(s[:, 0, 0]).max() == 0 and r.provenance["unidirectional"]
    assert any("unidirectional" in w for w in r.warnings)
    e = ls.solve(p, "eme").s()  # EME phase = exp(i k0 n_eff L) of the same FDE port mode
    np.testing.assert_allclose(np.angle(s[:, 1, 0] / e[:, 1, 0]), 0, atol=1e-9)


def test_straight_strip_3d_full_vector_keeps_te_and_tm():
    """Full-vectorial 3D: TE0 and TM0 of a 500x220 nm SOI strip transmit |S| = 1 within 2e-3 and do not convert
    (|S| < 2e-3); the TE0 phase is within 1e-2 rad of k0 n_eff L of the FDE port mode. Both limits come from the
    port modes being solved on a PEC window while BPM lines the window with a PML."""
    L, w, h, d = 2000.0, 2400.0, 2000.0, 40.0
    p = DeviceProblem(structures=[Structure(geometry=Box(size_nm=(L + 1000, 500, 220)), material=SI)], background=SIO2,
                      domain_size_nm=(L, w, h), wavelengths=[1550],
                      ports=[Port(name="o1", center_nm=(-L / 2, 0, 0), size_nm=(0, w, h), direction="+", num_modes=2),
                             Port(name="o2", center_nm=(L / 2, 0, 0), size_nm=(0, w, h), direction="-", num_modes=2)],
                      options={"bpm": {"grid_spacing_nm": d, "pml_cells": 6, "precision": "double"}})
    r = ls.solve(p, "bpm", device=CPU_3D)
    s = r.s()[0]
    assert r.ports == ["o1@0", "o1@1", "o2@0", "o2@1"] and r.provenance["vector"] == "full"
    np.testing.assert_allclose(np.abs(s[2:, :2]), np.eye(2), atol=2e-3)
    neff = port_modes(p, p.ports[0], 1550, grid_spacing_nm=d)[0].neff.real
    assert abs(np.angle(s[2, 0] * np.exp(-2j * np.pi / 1550 * neff * L))) < 1e-2  # measured 4.6e-3


# --- tapers, MMIs, branches, bends --------------------------------------------------------------


def taper(length, w1=500.0, w2=3000.0, dl=20.0, directions="forward"):
    x = length / 2 + 500
    st = [Structure(geometry=Box(center_nm=(-length / 2 - 1000, 0, 0), size_nm=(2000, w1, 1)), material=CORE),
          Structure(geometry=Box(center_nm=(length / 2 + 1000, 0, 0), size_nm=(2000, w2, 1)), material=CORE),
          Structure(geometry=Prism(profile=Polygon(vertices_nm=[(-length / 2, -w1 / 2), (length / 2, -w2 / 2),
                                                                 (length / 2, w2 / 2), (-length / 2, w1 / 2)]),
                                   z_min_nm=-1, z_max_nm=1), material=CORE)]
    return DeviceProblem(dimension=2, structures=st, background=CLAD, domain_size_nm=(2 * x, 6000, 0),
                         wavelengths=[1550],
                         ports=[port2d("o1", -x, width=6000), port2d("o2", x, width=6000, direction="-")],
                         options={"eme": {"num_modes": 30, "grid_spacing_nm": dl},
                                  "bpm": {"grid_spacing_nm": dl, "directions": directions}})


def test_linear_taper_matches_eme():
    """500 nm -> 3 um taper (diverging direction): |S21|^2 within 0.02 of EME on the same grid (measured
    differences <= 0.010 at 10 nm and 20 nm grids for 3-30 um tapers). The phase agrees within 0.01 rad."""
    for length in (3000, 10000):
        b = ls.solve(taper(length), "bpm", device="cpu").s()[0]
        e = ls.solve(taper(length), "eme").s()[0]
        assert abs(abs(b[1, 0]) ** 2 - abs(e[1, 0]) ** 2) < 0.02, (length, abs(b[1, 0]) ** 2, abs(e[1, 0]) ** 2)
        assert abs(np.angle(b[1, 0] / e[1, 0])) < 0.01


@pytest.mark.slow
def test_long_taper_is_reciprocal_and_adiabatic():
    """30 um taper on a 10 nm grid, both directions: |S21|^2 and |S12|^2 within 0.01 of each other and of EME (0.994;
    measured 0.996 / 0.995). The converging direction needs the finer grid (20 nm: 0.972), and short converging
    tapers (wide -> narrow) are where one-way BPM is weakest: at 3 um it gives 0.68 (te) vs EME's 0.77."""
    s = ls.solve(taper(30000, dl=10.0, directions="both"), "bpm", device="cpu").s()[0]
    e = ls.solve(taper(30000, dl=10.0), "eme").s()[0]
    t21, t12, te = abs(s[1, 0]) ** 2, abs(s[0, 1]) ** 2, abs(e[1, 0]) ** 2
    assert abs(t21 - t12) < 0.01 and abs(t21 - te) < 0.01 and abs(t12 - te) < 0.01


def mmi(length, W=3000.0, wa=800.0, lam=1550.0):
    x = length / 2 + 1000
    st = [Structure(geometry=Box(center_nm=(-x, 0, 0), size_nm=(2 * x - length, wa, 1)), material=CORE),
          Structure(geometry=Box(size_nm=(length, W, 1)), material=CORE)]
    st += [Structure(geometry=Box(center_nm=(x, y, 0), size_nm=(2 * x - length, wa, 1)), material=CORE)
           for y in (-W / 4, W / 4)]
    return DeviceProblem(dimension=2, structures=st, background=CLAD, domain_size_nm=(2 * x, W + 3000, 0),
                         wavelengths=[lam], ports=[port2d("o1", -x, width=W / 2), port2d("o2", x, -W / 4, W / 2, "-"),
                                                   port2d("o3", x, W / 4, W / 2, "-")],
                         options={"eme": {"num_modes": 40, "grid_spacing_nm": 10},
                                  "bpm": {"grid_spacing_nm": 10, "directions": "forward"}})


def test_mmi_1x2_self_imaging_length():
    """Soldano & Pennings (1995) centre-fed 1x2 MMI, L = 3 L_pi / 8 (see test_eme): BPM finds the best length at
    0.95 of the paraxial estimate, as EME does; the split is 50/50 by symmetry to 1e-9; total output within 0.06 of
    EME at every length (EME also keeps the MMI end-wall reflections that BPM drops)."""
    nr, nc, lam, W = 2.85, 1.44, 1550.0, 3000.0
    we = W + lam / np.pi * (nc / nr) ** 2 / np.sqrt(nr**2 - nc**2)
    l_th = nr * we**2 / (2 * lam)
    fs = np.array([0.9, 0.95, 1.0, 1.05])
    out_b, out_e = [], []
    for f in fs:
        b = np.abs(ls.solve(mmi(f * l_th), "bpm", device="cpu").s()[0]) ** 2
        e = np.abs(ls.solve(mmi(f * l_th), "eme").s()[0]) ** 2
        assert b[1, 0] == pytest.approx(b[2, 0], rel=1e-9)
        out_b.append(b[1, 0] + b[2, 0])
        out_e.append(e[1, 0] + e[2, 0])
    out_b, out_e = np.array(out_b), np.array(out_e)
    assert fs[np.argmax(out_b)] == fs[np.argmax(out_e)] == 0.95
    assert out_b.max() > 0.9
    np.testing.assert_allclose(out_b, out_e, atol=0.06)


def test_slab_bend_loss_matches_the_fde_bend_mode():
    """2D te bend, R = 20 um, n 1.60 / 1.45, 800 nm core: the power decay rate between 20 and 30 um of arc equals
    the FDE bent-mode loss (same grid and PML) within 2% (measured 0.2%)."""
    from lightsmith.backends._fde import solve_modes
    from lightsmith.problems import Rectangle, Structure2D

    co, cl, R = ls.ConstantIndex(n_real=1.60), ls.ConstantIndex(n_real=1.45), 20000.0
    ref = solve_modes([Structure2D(profile=Rectangle(size_nm=(800, 1e9)), material=co)], cl, 1550,
                      center_nm=(0, 0), size_nm=(8000, 0), grid_spacing_nm=20, boundary="pml", pml_cells=15,
                      bend_radius_nm=R, polarization="te")[0].loss_db_per_cm

    def bend(L):
        return DeviceProblem(dimension=2, structures=[Structure(geometry=Box(size_nm=(L + 2000, 800, 1)), material=co)],
                             background=cl, domain_size_nm=(L, 8000, 0), wavelengths=[1550],
                             ports=[port2d("o1", -L / 2, width=8000), port2d("o2", L / 2, width=8000, direction="-")],
                             options={"bpm": {"grid_spacing_nm": 20, "bend_radius_nm": R, "directions": "forward"}})

    t = {L: abs(ls.solve(bend(L), "bpm", device="cpu").s()[0][1, 0]) ** 2 for L in (20000, 30000)}
    loss = -10 * np.log10(t[30000] / t[20000]) / 10000 * 1e7
    assert loss == pytest.approx(ref, rel=0.02)


def _strip3d(along, bend_radius_nm=None, length=3000.0):
    """220 x 500 nm Si strip in SiO2 running along `along`, ports on the two end planes."""
    big, w = length + 2000, (2400, 1600)
    if along == "x":
        box, dom = Box(size_nm=(big, 500, 220)), (length, *w)
        ends = [(-length / 2, 0, 0), (length / 2, 0, 0)]
        size = (0, *w)
    else:
        box, dom = Box(size_nm=(500, big, 220)), (w[0], length, w[1])
        ends = [(0, -length / 2, 0), (0, length / 2, 0)]
        size = (w[0], 0, w[1])
    opts = {"grid_spacing_nm": 50, "directions": "forward", "vector": "semi", "polarization": "te"}
    if bend_radius_nm:
        opts["bend_radius_nm"] = bend_radius_nm
    return DeviceProblem(structures=[Structure(geometry=box, material=SI)], background=SIO2, domain_size_nm=dom,
                         wavelengths=[1550], options={"bpm": opts},
                         ports=[Port(name="o1", center_nm=ends[0], size_nm=size, direction="+"),
                                Port(name="o2", center_nm=ends[1], size_nm=size, direction="-")])


@pytest.mark.parametrize("radius", [None, 3000.0])
def test_3d_results_do_not_depend_on_the_propagation_axis(radius):
    """A strip along y is the strip along x rotated in the chip plane: 'te' must launch the same in-plane-E
    mode and a bend must curve in the chip plane, so S21 agrees exactly. (Before the fix, ports normal to y
    mapped local x to global z: 'te' launched the vertical-E mode and bends curved out of plane.)"""
    a = ls.solve(_strip3d("x", radius), "bpm", device="cpu").s()[0][1, 0]
    b = ls.solve(_strip3d("y", radius), "bpm", device="cpu").s()[0][1, 0]
    assert abs(a - b) < 1e-9


def ybranch(Lb=15000.0, Ls=3000.0, w=500.0, sep=3000.0):
    """Stem 500 -> 1000 nm over Ls, then two 500 nm arms diverging linearly to `sep` apart over Lb."""
    x0, x1, c1 = -Ls - 1000, Lb + 1000, sep / 2

    def poly(v):
        return Structure(geometry=Prism(profile=Polygon(vertices_nm=v), z_min_nm=-1, z_max_nm=1), material=CORE)

    st = [Structure(geometry=Box(center_nm=(x0 - 500, 0, 0), size_nm=(3000, w, 1)), material=CORE),
          poly([(-Ls, -w / 2), (0, -w), (0, w), (-Ls, w / 2)])]
    for s in (1, -1):
        st.append(poly([(0, 0.0), (Lb, s * (c1 - w / 2)), (Lb, s * (c1 + w / 2)), (0, s * w)]))
        st.append(Structure(geometry=Box(center_nm=(Lb + 1500, s * c1, 0), size_nm=(3000, w, 1)), material=CORE))
    return DeviceProblem(dimension=2, structures=st, background=CLAD, domain_size_nm=(x1 - x0, sep + 4000, 0),
                         domain_center_nm=((x0 + x1) / 2, 0, 0), wavelengths=[1550],
                         ports=[port2d("o1", x0, width=2500), port2d("o2", x1, c1, 2500, "-"),
                                port2d("o3", x1, -c1, 2500, "-")],
                         options={"bpm": {"grid_spacing_nm": 20}, "eme": {"grid_spacing_nm": 20, "num_modes": 40}})


def test_y_branch_splits_evenly():
    """Symmetric Y-branch (arms diverge at 4.8 deg): 50/50 to 1e-9 in both directions, S21 and S12 within 0.02
    (reciprocity of a one-way method), total within 0.1 of EME (0.936; FDFD 0.912). Measured BPM total 0.856-0.864,
    converged in grid (10/20 nm), slicing and step: the high-contrast (2.85/1.44) junction is lossier in one-way BPM."""
    p = ybranch()
    r = ls.solve(p, "bpm", device="cpu")
    s = np.abs(r.s()[0]) ** 2
    assert s[1, 0] == pytest.approx(s[2, 0], rel=1e-9) and s[0, 1] == pytest.approx(s[0, 2], rel=1e-9)
    assert abs(s[1, 0] - s[0, 1]) < 0.02
    assert s[1, 2] == 0 and s[0, 0] == 0
    e = np.abs(ls.solve(p, "eme").s()[0]) ** 2
    assert abs((s[1, 0] + s[2, 0]) - (e[1, 0] + e[2, 0])) < 0.1


# --- interface ------------------------------------------------------------------------------------


def test_supports_estimate_and_auto_order():
    p = straight2d()
    assert NativeBPM.supports(p) == (True, "")
    side = p.model_copy(update={"ports": [p.ports[0], Port(name="o2", center_nm=(0, 2000, 0), size_nm=(1000, 0, 1),
                                                                direction="-")]})
    ok, why = NativeBPM.supports(side)
    assert not ok and "same axis" in why
    bad = p.model_copy(update={"options": {"bpm": {"pade_order": 9}}})
    assert not NativeBPM.supports(bad)[0]
    est = ls.estimate(p, "bpm", device="cpu")
    assert est["backend"] == "lightsmith-bpm" and est["unidirectional"] and est["fits"]
    assert est["block_size"] == 1 and est["cross_section_grid"][1] == 1 and est["memory_bytes"] > 0
    est3 = ls.estimate(DeviceProblem(structures=[Structure(geometry=Box(size_nm=(3000, 500, 220)), material=SI)],
                                     background=SIO2, domain_size_nm=(2000, 2000, 1500), wavelengths=[1550],
                                     ports=[Port(name="a", center_nm=(-1000, 0, 0), size_nm=(0, 2000, 1500), direction="+"),
                                            Port(name="b", center_nm=(1000, 0, 0), size_nm=(0, 2000, 1500), direction="-")],
                                     options={"bpm": {"grid_spacing_nm": 25}}), "bpm", device="cpu")
    assert est3["vector"] == "full" and est3["components"] == ["Hx", "Hy"] and est3["block_size"] == 2 * 60
    from lightsmith.api import AUTO_ORDER

    assert AUTO_ORDER["device"].index("bpm") == AUTO_ORDER["device"].index("eme") + 1


def test_torch_cpu_matches_numpy():
    """The array-API path through torch (CPU, double) reproduces NumPy to 1e-10 on a taper."""
    pytest.importorskip("torch")
    p = taper(3000).model_copy(update={"options": {"bpm": {"grid_spacing_nm": 40, "directions": "forward"}}})
    a = ls.solve(p, "bpm", device="cpu").s()
    b = ls.solve(p, "bpm", device="torch-cpu").s()
    np.testing.assert_allclose(b, a, atol=1e-10)


@pytest.mark.gpu
def test_gpu_matches_cpu_3d_and_small_2d_falls_back_to_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    r2 = ls.solve(straight2d(wavelengths=[1550]), "bpm", device="cuda")
    assert r2.provenance["device"] == "cpu" and "gpu_min_unknowns" in r2.provenance["device_note"]
    L, w = 2000.0, 2400.0
    p = DeviceProblem(structures=[Structure(geometry=Box(size_nm=(L + 1000, 500, 220)), material=SI)], background=SIO2,
                      domain_size_nm=(L, w, 2000), wavelengths=[1550],
                      ports=[Port(name="o1", center_nm=(-L / 2, 0, 0), size_nm=(0, w, 2000), direction="+"),
                             Port(name="o2", center_nm=(L / 2, 0, 0), size_nm=(0, w, 2000), direction="-")],
                      options={"bpm": {"grid_spacing_nm": 40, "pml_cells": 6}})
    g = ls.solve(p, "bpm", device="cuda")
    c = ls.solve(p, "bpm", device="cpu")
    assert g.provenance["precision"] == "single" and g.provenance["device"].startswith("cuda")
    np.testing.assert_allclose(g.s(), c.s(), atol=1e-4)  # complex64 vs complex128


@pytest.mark.slow
def test_strip_bend_loss_3d_matches_the_fde_bend_mode():
    """3D full-vector, SiN 800x400 nm strip in SiO2, R = 8 um: the decay rate between 10 and 20 um of arc equals the
    FDE bent-mode loss (1244 dB/cm, same 50 nm grid, PML cells and reflection) within 2% (measured 0.4%; with the
    default pml_reflection 1e-8 instead of FDE's 1e-6 it is 6% lower: leaky-mode loss depends on the truncation)."""
    from lightsmith.backends._fde import solve_modes
    from lightsmith.problems import Rectangle, Structure2D

    sin, R = ls.ConstantIndex(n_real=2.0), 8000.0
    ref = solve_modes([Structure2D(profile=Rectangle(size_nm=(800, 400)), material=sin)], SIO2, 1550, center_nm=(0, 0),
                      size_nm=(5000, 2000), grid_spacing_nm=50, boundary="pml", pml_cells=10, bend_radius_nm=R,
                      group_index="none")[0].loss_db_per_cm

    def bend(L):
        return DeviceProblem(structures=[Structure(geometry=Box(size_nm=(L + 2000, 800, 400)), material=sin)],
                             background=SIO2, domain_size_nm=(L, 5000, 2000), wavelengths=[1550],
                             ports=[Port(name="o1", center_nm=(-L / 2, 0, 0), size_nm=(0, 5000, 2000), direction="+"),
                                    Port(name="o2", center_nm=(L / 2, 0, 0), size_nm=(0, 5000, 2000), direction="-")],
                             options={"bpm": {"grid_spacing_nm": 50, "pml_cells": 10, "pml_reflection": 1e-6,
                                              "bend_radius_nm": R, "directions": "forward"}})

    t = {L: abs(ls.solve(bend(L), "bpm").s()[0][1, 0]) ** 2 for L in (10000, 20000)}
    loss = -10 * np.log10(t[20000] / t[10000]) / 10000 * 1e7
    assert loss == pytest.approx(ref, rel=0.02)
