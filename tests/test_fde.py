"""Finite-difference eigenmode solver vs analytic and published references."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import jv, jvp, kv, kvp

import lightsmith as ls
from lightsmith.backends._fde import port_modes, solve_modes
from lightsmith.problems import Box, Circle, DeviceProblem, ModeProblem, Port, Rectangle, Structure, Structure2D

SI, SIO2 = ls.ConstantIndex(n_real=3.476), ls.ConstantIndex(n_real=1.444)


def strip(dx=20, wl=(1550.0,), num_modes=1, symmetry=("pec", "pmc"), window=(3000, 2000), **kw):
    opts = {"grid_spacing_nm": dx, "symmetry": list(symmetry), **kw.pop("options", {})}
    return ModeProblem(structures=[Structure2D(profile=Rectangle(size_nm=(500, 220)), material=SI)], background=SIO2,
                       window_size_nm=window, wavelengths=list(wl), num_modes=num_modes,
                       options={"fde": opts}, **kw)


# --- symmetric slab -------------------------------------------------------------

N1, N2, D, LAM = 2.0, 1.45, 600.0, 1550.0


def slab_neff(tm: bool, lam=LAM) -> float:
    """Fundamental even mode of the symmetric slab: tan(kappa d/2) = r gamma/kappa, r = (n1/n2)^2 for TM."""
    k0 = 2 * np.pi / lam
    r = (N1 / N2) ** 2 if tm else 1.0

    def f(n):
        kap, gam = k0 * np.sqrt(N1**2 - n**2), k0 * np.sqrt(n**2 - N2**2)
        return kap * np.sin(kap * D / 2) - r * gam * np.cos(kap * D / 2)

    hi = N1 - 1e-12
    lo = max(N2 + 1e-12, np.sqrt(N1**2 - (np.pi / (k0 * D)) ** 2) + 1e-12)  # kappa d/2 < pi/2
    return brentq(f, lo, hi, xtol=1e-14)


def slab(dx, pol):
    # Slab normal along x, invariant along y (window_size y = 0). TE (E parallel to the slab) is Ey,
    # i.e. 'tm' in the te_fraction (= Ex share) convention, and vice versa.
    core = Structure2D(profile=Rectangle(size_nm=(D, 1e9)), material=ls.ConstantIndex(n_real=N1))
    return ModeProblem(structures=[core],
                       background=ls.ConstantIndex(n_real=N2), window_size_nm=(4000, 0), wavelengths=[LAM],
                       num_modes=1, options={"fde": {"grid_spacing_nm": dx, "polarization": pol}})


@pytest.mark.parametrize("tm", [False, True])
def test_slab_matches_dispersion_relation(tm):
    pol = "te" if tm else "tm"
    errs = [ls.solve(slab(dx, pol), "fde").modes[0][0].neff_real - slab_neff(tm) for dx in (10.0, 2.5)]
    assert abs(errs[1]) < 1e-4
    assert abs(errs[1]) < abs(errs[0]) / 10  # second order: 4x refinement -> ~16x


def test_slab_group_index_matches_analytic_derivative():
    r = ls.solve(slab(2.5, "tm"), "fde")
    h = 0.5
    ng = slab_neff(False) - LAM * (slab_neff(False, LAM + h) - slab_neff(False, LAM - h)) / (2 * h)
    assert r.modes[0][0].ng == pytest.approx(ng, abs=2e-4)
    assert r.modes[0][0].te_fraction < 1e-12


# --- step-index fibre -------------------------------------------------------------


def test_step_index_fibre_he11():
    n1, n2, a, lam = 1.50, 1.45, 1500.0, 1550.0
    k0 = 2 * np.pi / lam

    def eq(n):  # exact vector HE11 eigenvalue equation (Snyder & Love, Optical Waveguide Theory, Table 12-3)
        u, w = a * k0 * np.sqrt(n1**2 - n**2), a * k0 * np.sqrt(n**2 - n2**2)
        J, K = jvp(1, u) / (u * jv(1, u)), kvp(1, w) / (w * kv(1, w))
        return (J + K) * (n1**2 * J + n2**2 * K) - n**2 * (1 / u**2 + 1 / w**2) ** 2

    ns = np.linspace(n2 + 1e-6, n1 - 1e-6, 2000)
    v = np.array([eq(x) for x in ns])
    i = max(j for j in range(len(ns) - 1) if np.sign(v[j]) != np.sign(v[j + 1]) and abs(v[j]) < 1e3)
    exact = brentq(eq, ns[i], ns[i + 1], xtol=1e-14)
    p = ModeProblem(structures=[Structure2D(profile=Circle(radius_nm=a), material=ls.ConstantIndex(n_real=n1))],
                    background=ls.ConstantIndex(n_real=n2), window_size_nm=(9000, 9000), wavelengths=[lam],
                    num_modes=1, options={"fde": {"grid_spacing_nm": 50, "symmetry": ["pec", "pmc"]}})
    m = ls.solve(p, "fde").modes[0][0]
    assert m.neff_real == pytest.approx(exact, abs=2e-6)
    assert m.te_fraction > 0.999  # x-polarised HE11


# --- silicon strip ------------------------------------------------------------------

# 500 x 220 nm Si strip, SiO2 clad, 1550 nm, TE0: neff = 2.44523 from the SiPkit 220 nm SOI effective-index
# table (https://sipkit.readthedocs.io/en/latest/1-%20Effective%20Index.html, 0.5 um width at 1.55 um).
SI_STRIP_TE0 = 2.44523


def test_si_strip_te0_and_grid_convergence():
    n = [ls.solve(strip(dx), "fde").modes[0][0] for dx in (20, 10)]
    errs = [abs(m.neff_real - SI_STRIP_TE0) for m in n]
    assert 2.40 < n[1].neff_real < 2.45
    assert errs[1] < 1e-3
    assert errs[1] < errs[0] / 4
    assert n[1].te_fraction > 0.95
    assert 3.9 < n[1].ng < 4.2  # no material dispersion here
    assert 0.15 < n[1].effective_area_um2 < 0.25
    assert n[1].loss_db_per_cm == pytest.approx(0, abs=1e-6)


@pytest.mark.slow
def test_si_strip_fine_grid():
    m = ls.solve(strip(5), "fde").modes[0][0]
    assert m.neff_real == pytest.approx(SI_STRIP_TE0, abs=3e-4)


def test_symmetry_matches_full_window_and_group_index_methods_agree():
    full = ls.solve(strip(25, symmetry=("none", "none"), num_modes=2, window=(2500, 1600)), "fde")
    quarter = ls.solve(strip(25, window=(2500, 1600)), "fde")
    assert quarter.modes[0][0].neff_real == pytest.approx(full.modes[0][0].neff_real, abs=1e-9)
    assert full.modes[0][1].te_fraction < 0.1  # TM0 next
    fd = ls.solve(strip(25, window=(2500, 1600), options={"group_index": "finite_difference"}), "fde")
    assert fd.modes[0][0].ng == pytest.approx(quarter.modes[0][0].ng, abs=2e-3)


def test_tracking_fields_and_outputs():
    r = ls.solve(strip(25, wl=(1500, 1550, 1600), num_modes=2, symmetry=("none", "none"), window=(2500, 1600),
                        include_fields=True,
                        options={"track_modes": True}), "fde")
    te = [row[0] for row in r.modes]
    assert all(m.te_fraction > 0.9 for m in te)
    assert te[0].neff_real > te[1].neff_real > te[2].neff_real
    f = te[1].field
    ex = np.asarray(f.components["Ex"][0]) + 1j * np.asarray(f.components["Ex"][1])
    assert ex.shape == (len(f.x_nm), len(f.y_nm))
    i, j = np.unravel_index(np.argmax(np.abs(ex)), ex.shape)
    assert abs(f.x_nm[i]) < 260 and abs(f.y_nm[j]) < 120
    assert r.provenance["max_residual"] < 1e-8
    est = ls.estimate(strip(25), "fde")
    assert est["fits"] and est["unknowns"] > 0


# --- bends --------------------------------------------------------------------------


def test_bend_loss_decreases_with_radius():
    def bend(r):
        core = Structure2D(profile=Rectangle(size_nm=(400, 1e9)), material=ls.ConstantIndex(n_real=2.0))
        p = ModeProblem(structures=[core],
                        background=ls.ConstantIndex(n_real=1.45), window_center_nm=(1000, 0), window_size_nm=(4000, 0),
                        wavelengths=[1550], num_modes=1, bend_radius_nm=r,
                        options={"fde": {"grid_spacing_nm": 5, "polarization": "tm", "pml_cells": 60,
                                         "bend_reference_x_nm": 0}})
        return ls.solve(p, "fde").modes[0][0]

    radii = [2000, 3000, 4000, 5000]
    loss = np.array([bend(r).loss_db_per_cm for r in radii])
    assert np.all(np.diff(loss) < 0) and loss[-1] > 0
    # Radiation loss ~ exp(-c R): log-loss close to linear in R.
    fit = np.polyfit(radii, np.log(loss), 1, full=True)
    assert fit[0][0] < 0 and fit[1][0] < 0.3
    straight = bend(None)
    assert straight.loss_db_per_cm < 1e-2 * loss[-1]


@pytest.mark.slow
def test_si_strip_bend_loss_trend():
    def loss(r):
        p = ModeProblem(structures=[Structure2D(profile=Rectangle(size_nm=(400, 220)), material=SI)], background=SIO2,
                        window_size_nm=(2400, 1600), wavelengths=[1550], num_modes=1, bend_radius_nm=r,
                        options={"fde": {"grid_spacing_nm": 20, "polarization": "te", "group_index": "none"}})
        return ls.solve(p, "fde").modes[0][0].loss_db_per_cm

    assert loss(1300) > loss(2500) > 0


# --- reuse from device solvers ---------------------------------------------------------


def test_port_modes_match_mode_problem():
    dev = DeviceProblem(structures=[Structure(geometry=Box(size_nm=(4000, 500, 220)), material=SI)], background=SIO2,
                        domain_size_nm=(4000, 2500, 1600), wavelengths=[1550],
                        ports=[Port(name="o1", center_nm=(-1500, 0, 0), size_nm=(0, 2500, 1600), direction="+")])
    pm = port_modes(dev, dev.ports[0], 1550, grid_spacing_nm=25)
    ref = ls.solve(strip(25, symmetry=("none", "none"), window=(2500, 1600)), "fde").modes[0][0]
    assert pm[0].neff.real == pytest.approx(ref.neff_real, abs=1e-6)
    u, v, comps = pm[0].global_fields()
    assert set(comps) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    # Port normal x: the TE0 field is mostly along global y.
    peak = {k: np.abs(comps[k]).max() for k in ("Ex", "Ey", "Ez")}
    assert peak["Ey"] > 3 * peak["Ez"] and peak["Ey"] > 1.5 * peak["Ex"]  # Ex: longitudinal at the sidewalls
    m = solve_modes([Structure2D(profile=Rectangle(size_nm=(500, 220)), material=SI)], SIO2, 1550,
                    center_nm=(0, 0), size_nm=(2500, 1600), grid_spacing_nm=25)
    assert m[0].neff.real == pytest.approx(ref.neff_real, abs=1e-9)


def test_non_uniform_axes():
    """Explicit uniform widths reproduce the uniform grid to the last bit; a grid graded from 20 to 5 nm across
    a silicon strip (as the FDTD's graded port windows) agrees with a uniform 5 nm grid to 1e-3 in neff."""
    from lightsmith.backends._fde.grid import Grid
    from lightsmith.backends._fde.modes import solve_eigenmodes

    def eps(g):
        def at(dx, dy):
            X, Y = np.meshgrid(g.nodes(0, dx), g.nodes(1, dy), indexing="ij")
            return np.where((np.abs(X - 1000) < 250) & (np.abs(Y - 500) < 110), 3.48**2, 1.444**2)
        return np.stack([at(True, False), at(False, True), at(False, False)])

    def axis(length, lo, hi, fine, coarse):
        w, x = [], 0.0
        while x < length - 1e-9:
            w.append(fine if lo - 1e-9 <= x < hi - 1e-9 else coarse)
            x += w[-1]
        return tuple(w)

    g = Grid(0, 0, 20, 20, 100, 50)
    gw = Grid(0, 0, 20, 20, 100, 50, widths=((20.0,) * 100, (20.0,) * 50))
    a, b = solve_eigenmodes(g, eps(g), 1550, 2), solve_eigenmodes(gw, eps(gw), 1550, 2)
    assert [m.neff for m in a] == [m.neff for m in b]
    np.testing.assert_array_equal(g.weights(0, False), gw.weights(0, False))
    g, gw = Grid(0, 0, 20, 10, 100, 100), Grid(0, 0, 20, 10, 100, 100, widths=((20.0,) * 100, None))
    assert [m.neff for m in solve_eigenmodes(g, eps(g), 1550, 1)] == [m.neff for m in solve_eigenmodes(gw, eps(gw), 1550, 1)]
    fine = Grid(0, 0, 5, 5, 400, 200)
    wx, wy = axis(2000, 700, 1300, 5, 20), axis(1000, 360, 640, 5, 20)
    gg = Grid(0, 0, 20, 20, len(wx), len(wy), widths=(wx, wy))
    assert gg.nodes(0, False)[-1] + wx[-1] == pytest.approx(2000)
    n_fine = solve_eigenmodes(fine, eps(fine), 1550, 1)[0].neff.real
    n_graded = solve_eigenmodes(gg, eps(gg), 1550, 1)[0].neff.real
    assert abs(n_graded - n_fine) < 1e-3  # measured 9e-4 (the cell edges fall differently on the strip)
