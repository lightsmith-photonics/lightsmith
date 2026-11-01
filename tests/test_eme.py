"""Eigenmode expansion: straight guides, tapers and MMIs against physical limits and self-imaging theory."""

import numpy as np
import pytest

import omniscatter as osc
from omniscatter.backends._fde import port_modes
from omniscatter.backends.eme import NativeEME
from omniscatter.problems import Box, DeviceProblem, Polygon, Port, Prism, Structure

SI, SIO2 = osc.ConstantIndex(n_real=3.476), osc.ConstantIndex(n_real=1.444)
# 2D devices: core = slab effective index of 220 nm SOI (TE), cladding SiO2.
CORE, CLAD = osc.ConstantIndex(n_real=2.85), osc.ConstantIndex(n_real=1.44)


def port2d(name, x, y=0.0, width=2000.0, direction="+"):
    return Port(name=name, center_nm=(x, y, 0), size_nm=(0, width, 1), direction=direction)


def straight3d(length=2000.0, width=500.0, num_modes=2):
    w = 2000.0
    return DeviceProblem(
        structures=[Structure(geometry=Box(size_nm=(length + 1000, width, 220)), material=SI)], background=SIO2,
        domain_size_nm=(length, w, 1500), wavelengths=[1550],
        ports=[Port(name="o1", center_nm=(-length / 2, 0, 0), size_nm=(0, w, 1500), direction="+", num_modes=num_modes),
               Port(name="o2", center_nm=(length / 2, 0, 0), size_nm=(0, w, 1500), direction="-", num_modes=num_modes)],
        options={"eme": {"num_modes": 6, "grid_spacing_nm": 25}})


def test_uniform_waveguide_3d_transmits_fully_with_the_fde_phase():
    r = osc.solve(straight3d(), "eme")
    s = r.s()[0]
    assert r.ports == ["o1@0", "o1@1", "o2@0", "o2@1"]
    np.testing.assert_allclose(np.abs(s[2:, :2]), np.eye(2), atol=1e-6)  # TE0 -> TE0, TM0 -> TM0
    assert np.abs(s[:2, :2]).max() < 1e-8  # no reflection
    dev = straight3d()
    neff = port_modes(dev, dev.ports[0], 1550, grid_spacing_nm=25)[0].neff.real
    phase = 2 * np.pi / 1550 * neff * 2000
    assert np.angle(s[2, 0] * np.exp(-1j * phase)) == pytest.approx(0, abs=1e-6)


def test_uniform_waveguide_2d():
    L = 3000.0
    p = DeviceProblem(dimension=2, structures=[Structure(geometry=Box(size_nm=(L + 2000, 500, 1)), material=CORE)],
                      background=CLAD, domain_size_nm=(L, 4000, 0), wavelengths=[1500, 1550],
                      ports=[port2d("o1", -L / 2), port2d("o2", L / 2, direction="-")],
                      options={"eme": {"num_modes": 10}})
    s = osc.solve(p, "eme").s()
    np.testing.assert_allclose(np.abs(s[:, 1, 0]), 1, atol=1e-5)
    assert np.abs(s[:, 0, 0]).max() < 1e-6


def test_width_step_3d_is_reciprocal_and_passive():
    p = DeviceProblem(
        structures=[Structure(geometry=Box(center_nm=(-1000, 0, 0), size_nm=(2000, 500, 220)), material=SI),
                    Structure(geometry=Box(center_nm=(1000, 0, 0), size_nm=(2000, 700, 220)), material=SI)],
        background=SIO2, domain_size_nm=(2000, 2000, 1500), wavelengths=[1550],
        ports=[Port(name="o1", center_nm=(-1000, 0, 0), size_nm=(0, 2000, 1500), direction="+"),
               Port(name="o2", center_nm=(1000, 0, 0), size_nm=(0, 2000, 1500), direction="-")],
        options={"eme": {"num_modes": 8, "grid_spacing_nm": 25}})
    r = osc.solve(p, "eme")
    s = r.s()[0]
    assert r.provenance["cells"] == 2
    assert abs(s[1, 0] - s[0, 1]) < 1e-6  # reciprocity
    assert 0.97 < abs(s[1, 0]) ** 2 < 1.0  # converges to ~0.989 with 48+ modes
    assert np.all(np.sum(np.abs(s) ** 2, axis=0) < 1.0 + 1e-3)


def taper(length, w1=500.0, w2=3000.0):
    x = length / 2 + 500
    st = [Structure(geometry=Box(center_nm=(-length / 2 - 1000, 0, 0), size_nm=(2000, w1, 1)), material=CORE),
          Structure(geometry=Box(center_nm=(length / 2 + 1000, 0, 0), size_nm=(2000, w2, 1)), material=CORE),
          Structure(geometry=Prism(profile=Polygon(vertices_nm=[(-length / 2, -w1 / 2), (length / 2, -w2 / 2),
                                                                 (length / 2, w2 / 2), (-length / 2, w1 / 2)]),
                                   z_min_nm=-1, z_max_nm=1), material=CORE)]
    return DeviceProblem(dimension=2, structures=st, background=CLAD, domain_size_nm=(2 * x, 6000, 0),
                         wavelengths=[1550],
                         ports=[port2d("o1", -x, width=6000), port2d("o2", x, width=6000, direction="-")],
                         options={"eme": {"num_modes": 30, "grid_spacing_nm": 10}})


def test_linear_taper_approaches_adiabatic_limit():
    t = []
    for length in (1000, 3000, 10000, 30000):
        s = osc.solve(taper(length), "eme").s()[0]
        t.append(abs(s[1, 0]) ** 2)
        assert abs(s[1, 0] - s[0, 1]) < 1e-3
    assert np.all(np.diff(t) > 0), t
    assert t[0] < 0.6 and t[-1] > 0.98


def test_mmi_1x2_self_imaging_length():
    """Soldano & Pennings, J. Lightwave Technol. 13, 615 (1995): centre-fed 1xN images at L = 3 L_pi / (4N),
    L_pi = 4 n_r W_e^2 / (3 lambda), W_e = W + (lambda/pi)(n_c/n_r)^(2 sigma) / sqrt(n_r^2 - n_c^2), sigma = 1
    for E in the plane (normal to the MMI sidewalls). The formula is paraxial: the true optimum is a few % shorter."""
    nr, nc, lam, W, wa = 2.85, 1.44, 1550.0, 3000.0, 800.0
    we = W + lam / np.pi * (nc / nr) ** 2 / np.sqrt(nr**2 - nc**2)
    l_th = nr * we**2 / (2 * lam)

    def mmi(length):
        x = length / 2 + 1000
        st = [Structure(geometry=Box(center_nm=(-x, 0, 0), size_nm=(2 * x - length, wa, 1)), material=CORE),
              Structure(geometry=Box(size_nm=(length, W, 1)), material=CORE)]
        st += [Structure(geometry=Box(center_nm=(x, y, 0), size_nm=(2 * x - length, wa, 1)), material=CORE)
               for y in (-W / 4, W / 4)]
        return DeviceProblem(dimension=2, structures=st, background=CLAD, domain_size_nm=(2 * x, W + 3000, 0),
                             wavelengths=[lam], ports=[port2d("o1", -x, width=W / 2),
                                                       port2d("o2", x, -W / 4, W / 2, "-"),
                                                       port2d("o3", x, W / 4, W / 2, "-")],
                             options={"eme": {"num_modes": 40, "grid_spacing_nm": 10}})

    fs = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
    out = []
    for f in fs:
        r = osc.solve(mmi(f * l_th), "eme")
        s = np.abs(r.s()[0]) ** 2
        assert s[1, 0] == pytest.approx(s[2, 0], rel=1e-6)  # 50/50
        out.append(s[1, 0] + s[2, 0])
    out = np.array(out)
    best = fs[np.argmax(out)]
    assert 0.9 <= best <= 1.0  # within 10% of the analytic self-imaging length
    assert out.max() > 0.9
    assert out[fs == 1.0][0] > 0.8  # ~50/50 at the analytic length itself
    assert out[0] < out.max() and out[-1] < out.max()


def test_supports_explains_unsuitable_devices():
    base = straight3d()
    side = base.model_copy(update={"ports": [base.ports[0],
                                             Port(name="o2", center_nm=(0, 1000, 0), size_nm=(1000, 0, 1000),
                                                  direction="-")]})
    ok, why = NativeEME.supports(side)
    assert not ok and "same axis" in why
    wrong_dir = base.model_copy(update={"ports": [base.ports[0], base.ports[1].model_copy(update={"direction": "+"})]})
    ok, why = NativeEME.supports(wrong_dir)
    assert not ok and "into the device" in why
    three = base.model_copy(update={"ports": [*base.ports, base.ports[1].model_copy(
        update={"name": "o3", "center_nm": (0, 0, 0)})]})
    ok, why = NativeEME.supports(three)
    assert not ok and "two opposite planes" in why
    assert NativeEME.supports(base) == (True, "")
    est = osc.estimate(base, "eme")
    assert est["axis"] == "x" and est["fits"]
