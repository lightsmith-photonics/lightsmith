"""Dipole emission (EmissionProblem): analytic Mie limits, and FDTD against Mie.

Measured (RTX 5070, double precision, dx = 2.5 nm): dipole 10 nm from an n = 2, r = 60 nm sphere,
450-800 nm: Purcell factor within 0.07 % (radial) and 0.17 % (tangential) of Mie; a lone dipole's
radiation pattern within 0.05 % of 3/(8 pi) sin^2(theta), integrating to 1.000. Single precision: 1-4 %.
Near a Drude sphere (10 nm gap, dx = 2 nm) the errors are 10 % (radial) to 2.4x (tangential, off resonance):
the metal interface cells absorb too much (see the FDTD page).
"""

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends.mie_native import dipole_emission
from lightsmith.device import _torch_cuda
from lightsmith.problems import Dipole, EmissionProblem, RadiationPatternRequest, parse_problem

needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")


def sphere(r, material):
    return [ls.Particle(shape=ls.Sphere(radius_nm=r), material=material)]


def test_problem_parses_and_auto_picks_mie():
    p = parse_problem({"kind": "emission", "particles": [{"shape": {"kind": "sphere", "radius_nm": 40},
                                                          "material": {"kind": "constant_index", "n_real": 2}}],
                       "dipole": {"position_nm": [0, 0, 50], "orientation": [1, 0, 0]}, "wavelengths": [600]})
    assert isinstance(p, EmissionProblem)
    res = ls.solve(p)
    assert res.backend == "lightsmith-mie" and res.kind == "emission"
    assert res.efficiency[0] == pytest.approx(1.0, abs=1e-12)  # lossless sphere: everything radiates


def test_mie_emission_limits():
    # lossless sphere: total = radiative, any orientation
    t, r = dipole_emission([60.0], [np.full(3, 2.25 + 0j)], [450.0, 600.0, 800.0], 1.0, 75.0, 0.3)
    np.testing.assert_allclose(t, r, rtol=1e-10)
    # far from the sphere: no change
    t, r = dipole_emission([60.0], [np.full(1, 2.25 + 0j)], [600.0], 1.0, 1e5, 1.0)
    assert t[0] == pytest.approx(1, abs=1e-4) and r[0] == pytest.approx(1, abs=1e-4)
    # quasi-static quenching by a small lossy sphere, 1 + c R^3 Im(beta) / (k^3 d^6), approached as (R/d)^2
    eps, R, wl = 4 + 2j, 5.0, 1e5
    k, beta = 2 * np.pi / wl, (eps - 1) / (eps + 2)
    for c2, c in ((1.0, 6.0), (0.0, 1.5)):
        err = []
        for d in (50.0, 100.0):
            tot, _ = dipole_emission([R], [np.array([eps])], [wl], 1.0, d, c2)
            err.append((tot[0] - 1) / (c * R**3 * beta.imag / (k**3 * d**6)) - 1)
        assert abs(err[1]) < 0.01 and err[0] / err[1] == pytest.approx(4, rel=0.1)


def test_mie_emission_refuses_what_it_cannot_do():
    mie = ls.get_backend("lightsmith-mie")
    inside = EmissionProblem(particles=sphere(40, ls.ConstantIndex(n_real=2)), wavelengths=[600],
                             dipole=Dipole(position_nm=(0, 0, 20)))
    assert not mie.supports(inside)[0]
    pattern = EmissionProblem(particles=sphere(40, ls.ConstantIndex(n_real=2)), wavelengths=[600],
                              dipole=Dipole(position_nm=(0, 0, 50)),
                              radiation_pattern=RadiationPatternRequest(theta_deg=[0], wavelengths_nm=[600]))
    assert not mie.supports(pattern)[0] and ls.get_backend("fdtd").supports(pattern)[0]
    # scattering backends do not take emission problems
    with pytest.raises(ValueError):
        ls.solve(pattern, "dda")


@pytest.mark.gpu
@needs_cuda
def test_fdtd_emission_near_dielectric_sphere_matches_mie():
    """A dipole tilted 45 degrees 10 nm from the sphere: radial and tangential rates, cos^2-weighted."""
    prob = EmissionProblem(particles=sphere(60, ls.ConstantIndex(n_real=2.0)), wavelengths=[450.0, 600.0, 800.0],
                           dipole=Dipole(position_nm=(0, 0, 70), orientation=(1, 0, 1)),
                           options={"fdtd": {"spacing_nm": 2.5}})
    f = ls.solve(prob, "fdtd", "gpu")
    m = ls.solve(prob, "mie")
    assert f.provenance["precision"] == "double" and f.provenance["reference_balance"] < 1e-4
    np.testing.assert_allclose(f.purcell, m.purcell, rtol=5e-3)  # measured 0.07-0.17 %
    np.testing.assert_allclose(f.radiative, m.radiative, rtol=5e-3)
    np.testing.assert_allclose(f.efficiency, 1.0, atol=2e-3)


@pytest.mark.gpu
@needs_cuda
def test_fdtd_lone_dipole_radiation_pattern():
    th = np.linspace(0, 180, 19)
    prob = EmissionProblem(medium_index=1.33, dipole=Dipole(position_nm=(0.3, 0.7, 1.1)), wavelengths=[500.0, 600.0],
                           radiation_pattern=RadiationPatternRequest(theta_deg=list(th), phi_deg=[0, 45, 90],
                                                                     wavelengths_nm=[500.0, 600.0]),
                           options={"fdtd": {"spacing_nm": 2.5}})
    f = ls.solve(prob, "fdtd", "gpu")
    np.testing.assert_allclose(f.purcell, 1.0, atol=1e-6)
    np.testing.assert_allclose(f.radiative, 1.0, atol=1e-4)
    t = np.radians(th)
    ref = 3 / (8 * np.pi) * np.sin(t) ** 2
    for d in f.radiation_pattern:
        a = np.asarray(d.dp_domega)
        assert np.abs(a - ref[:, None]).max() < 2e-3 * ref.max()  # measured 5e-4
        assert 2 * np.pi * np.trapezoid(a[:, 0] * np.sin(t), t) == pytest.approx(1.0, abs=2e-3)
