"""Plane-wave-expansion band solver (lightsmith-pwe).

References used below:
- Joannopoulos, Johnson, Winn & Meade, *Photonic Crystals: Molding the Flow of
  Light*, 2nd ed. (2008), ch. 5: square lattice of alumina rods (eps = 8.9,
  r = 0.2a) has a TM gap between bands 1 and 2 with gap/midgap = 31.4 %;
  triangular lattice of air holes (eps = 13, r = 0.48a) has a complete gap of
  18.6 %. (Values as quoted in the book text; our converged numbers are given
  next to each test.)
- MPB tutorial (mpb.readthedocs.io, Python/Scheme tutorials): square lattice of
  eps = 12 rods, r = 0.2a, resolution 32: TM gap 0.282623 - 0.419335 (38.95 %).
"""

import math

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends.pwe import band_gaps, bands_from_permittivity, density_of_states, k_path, supercell
from lightsmith.device import _torch_cuda
from lightsmith.problems import Ball, BandProblem, Circle, KPoint, Structure, Structure2D
from lightsmith.results import BandResult

A = 400.0
AIR = ls.ConstantIndex(n_real=1.0)
SQ = [(A, 0, 0), (0, A, 0)]
TRI = [(A, 0, 0), (A / 2, A * math.sqrt(3) / 2, 0)]
SQ_PATH = [KPoint(label="Γ", reduced=(0, 0, 0)), KPoint(label="X", reduced=(0.5, 0, 0)),
           KPoint(label="M", reduced=(0.5, 0.5, 0)), KPoint(label="Γ", reduced=(0, 0, 0))]
TRI_PATH = [KPoint(label="Γ", reduced=(0, 0, 0)), KPoint(label="M", reduced=(0.5, 0, 0)),
            KPoint(label="K", reduced=(1 / 3, 2 / 3, 0)), KPoint(label="Γ", reduced=(0, 0, 0))]


def rods(eps=8.9, r=0.2, pol="tm", pps=6, nb=4, **opts):
    return BandProblem(lattice_vectors_nm=SQ, background=AIR, k_path=SQ_PATH, points_per_segment=pps,
                       structures_2d=[Structure2D(profile=Circle(radius_nm=r * A),
                                                  material=ls.ConstantPermittivity(eps_real=eps))],
                       num_bands=nb, polarization=pol, options={"pwe": opts})


def tm_gap(result):
    g = band_gaps(result)["tm"][0]
    assert g["between_bands"] == (1, 2)
    return g


def free_photon(lattice_vectors, k_cart, n, nb, nmax=6):
    """Sorted |k+G| a / (2 pi n) for a uniform medium (a = |a1|), 2D or 3D."""
    lv = np.asarray(lattice_vectors, float)
    d = len(lv)
    a = np.linalg.norm(lv[0])
    b = 2 * np.pi * np.linalg.inv(lv[:, :d] / a).T
    idx = np.stack(np.meshgrid(*[np.arange(-nmax, nmax + 1)] * d, indexing="ij"), -1).reshape(-1, d)
    q = np.linalg.norm(k_cart[:, None, :] + (idx @ b)[None], axis=-1)
    return np.sort(q, axis=1)[:, :nb] / (2 * np.pi * n)


# --- analytic limits -----------------------------------------------------------


@pytest.mark.parametrize("solver", ["dense", "lobpcg"])
def test_empty_lattice_2d_free_photon_bands(solver):
    n = 1.5
    p = BandProblem(lattice_vectors_nm=TRI, background=ls.ConstantIndex(n_real=n), k_path=TRI_PATH,
                    points_per_segment=5, num_bands=6,
                    options={"pwe": {"solver": solver, "resolution": 16, "num_plane_waves": 100}})
    r = ls.solve(p, "pwe", "cpu")
    _, cart, _, _ = k_path(p)
    ref = free_photon(TRI, cart, n, 6)
    for pol in ("te", "tm"):  # identical in a uniform medium
        np.testing.assert_allclose(r.frequencies[pol], ref, atol=1e-6)


@pytest.mark.parametrize("solver", ["dense", "lobpcg"])
def test_empty_lattice_3d_fcc_free_photon_bands(solver):
    n = 1.5
    fcc = [(0, A / 2, A / 2), (A / 2, 0, A / 2), (A / 2, A / 2, 0)]
    path = [KPoint(label="Γ", reduced=(0, 0, 0)), KPoint(label="X", reduced=(0.5, 0, 0.5)),
            KPoint(label="L", reduced=(0.5, 0.5, 0.5))]
    p = BandProblem(dimension=3, lattice_vectors_nm=fcc, background=ls.ConstantIndex(n_real=n), k_path=path,
                    points_per_segment=3, num_bands=6,
                    options={"pwe": {"solver": solver, "resolution": 8, "num_plane_waves": 120}})
    r = ls.solve(p, "pwe", "cpu")
    _, cart, _, _ = k_path(p)
    ref = np.repeat(free_photon(fcc, cart, n, 3), 2, axis=1)  # two transverse polarizations per plane wave
    np.testing.assert_allclose(r.frequencies["all"], ref, atol=1e-6)
    assert r.frequencies["all"][0][:2] == pytest.approx([0.0, 0.0], abs=1e-6)


# --- literature references -------------------------------------------------------


def test_square_rods_tm_gap_joannopoulos_and_convergence():
    # Converged: 0.3224 - 0.4425 a/lambda, gap/midgap 31.40 % (book: 31.4 %). Tolerance 0.5 % on edges.
    # HCS TM converges fast: |gap edge(N) - gap edge(800)| shrinks with plane-wave count N.
    # The path keeps the corners X and M, where the gap edges sit (band 1 max at M, band 2 min at X).
    edges = {}
    for npw in (40, 100, 200, 600):
        g = tm_gap(ls.solve(rods(pps=2, num_plane_waves=npw), "pwe", "cpu"))
        edges[npw] = np.array([g["bottom"], g["top"]])
    err = {n: np.abs(e - edges[600]).max() for n, e in edges.items() if n != 600}
    assert err[200] < err[40] and err[200] < 2e-4
    np.testing.assert_allclose(edges[600], [0.3224, 0.4425], rtol=5e-3)
    ratio = (edges[600][1] - edges[600][0]) / edges[600].mean()
    assert ratio == pytest.approx(0.314, abs=0.003)


def test_square_rods_eps12_matches_mpb_tutorial():
    # MPB (resolution 32) reports 0.282623 - 0.419335. We get 0.28066 - 0.41717 (dense HCS, converged to
    # 1e-5; our LOBPCG/subpixel path converges to the same values to 2e-4 at 128/a). Tolerance: 1 %.
    g = tm_gap(ls.solve(rods(eps=12, pps=2), "pwe", "cpu"))
    assert g["bottom"] == pytest.approx(0.282623, rel=1e-2)
    assert g["top"] == pytest.approx(0.419335, rel=1e-2)


def test_triangular_air_holes_complete_gap():
    # eps = 13, r = 0.48a. Converged complete gap 0.4297 - 0.5199 (18.98 %; set by the TM gap, which sits
    # inside the TE gap 0.36 - 0.53). Book: 18.6 %. Tolerance: 1 % on edges, 0.01 on the ratio.
    p = BandProblem(lattice_vectors_nm=TRI, background=ls.ConstantPermittivity(eps_real=13), k_path=TRI_PATH,
                    structures_2d=[Structure2D(profile=Circle(radius_nm=0.48 * A), material=AIR)],
                    points_per_segment=4, num_bands=3, polarization="both")
    r = ls.solve(p, "pwe", "cpu")
    assert r.provenance["solver"] == {"te": "lobpcg", "tm": "dense"}
    gaps = band_gaps(r)
    assert gaps["te"][0]["between_bands"] == (1, 2)
    (c,) = gaps["complete"]
    assert c["bottom"] == pytest.approx(0.4297, rel=1e-2)
    assert c["top"] == pytest.approx(0.5199, rel=1e-2)
    assert c["gap_midgap_ratio"] == pytest.approx(0.186, abs=0.01)


# --- the two discretisations agree --------------------------------------------------


def test_lobpcg_matches_dense_for_tm():
    path = [KPoint(label="X", reduced=(0.5, 0, 0)), KPoint(label="M", reduced=(0.5, 0.5, 0))]
    dense = ls.solve(rods(pps=1, num_plane_waves=300).model_copy(update={"k_path": path}), "pwe", "cpu")
    it = ls.solve(rods(pps=1, resolution=32).model_copy(update={"k_path": path}), "pwe", "cpu")
    np.testing.assert_allclose(it.frequencies["tm"], dense.frequencies["tm"], rtol=3e-3)
    assert it.provenance["eps_treatment"]["tm"] == "subpixel_smoothed_tensor"


def test_te_subpixel_smoothing_converges():
    # TE at X/M for eps = 12 rods; reference from resolution 64 (converged to ~1e-4 against 128).
    path = [KPoint(label="X", reduced=(0.5, 0, 0)), KPoint(label="M", reduced=(0.5, 0.5, 0))]
    f = {res: np.asarray(ls.solve(rods(eps=12, pol="te", pps=1, resolution=res).model_copy(update={"k_path": path}),
                                   "pwe", "cpu").frequencies["te"]) for res in (16, 32, 64)}
    e16, e32 = (np.abs(f[r] - f[64]).max() for r in (16, 32))
    assert e32 < e16 and e32 < 3e-3


# --- helpers -----------------------------------------------------------------------


def test_k_path_and_result_fields():
    r = ls.solve(rods(pps=4, num_plane_waves=60), "pwe", "cpu")
    assert len(r.k_reduced) == 13 and r.k_labels == {0: "Γ", 4: "X", 8: "M", 12: "Γ"}
    assert r.k_reduced[4] == (0.5, 0.0, 0.0)
    assert r.k_distance[4] == pytest.approx(0.5) and r.k_distance[-1] == pytest.approx(1 + math.sqrt(0.5))
    assert np.asarray(r.frequencies["tm"]).shape == (13, 4)
    again = BandResult.model_validate_json(r.model_dump_json())
    assert again.k_labels == r.k_labels


def test_band_gaps_helper_complete_gap():
    f_te = [[0.1, 0.5], [0.2, 0.6]]  # band 1 up to 0.2, band 2 from 0.5: gap 0.2-0.5
    f_tm = [[0.1, 0.4], [0.3, 0.7]]  # gap 0.3-0.4
    r = BandResult(method="pwe", backend="x", k_reduced=[(0, 0, 0), (0.5, 0, 0)], k_distance=[0, 0.5],
                   k_labels={0: "G", 1: "X"}, frequencies={"te": f_te, "tm": f_tm})
    g = band_gaps(r)
    assert (g["te"][0]["bottom"], g["te"][0]["top"]) == (0.2, 0.5)
    (c,) = g["complete"]
    assert (c["bottom"], c["top"]) == (0.3, 0.4)
    assert c["gap_midgap_ratio"] == pytest.approx(0.1 / 0.35)


def test_estimate_reports_matrix_size_without_solving():
    est = ls.estimate(rods(pol="both"), method="pwe", device="cpu")
    assert est["backend"] == "lightsmith-pwe"
    assert est["solver"] == {"te": "lobpcg", "tm": "dense"}
    assert est["matrix_size"]["tm"] >= 400 and est["matrix_size"]["te"] == 32 * 32
    assert est["memory_bytes"] > 0 and est["k_points"] == 19


def test_choose_method_picks_pwe_for_bands():
    b, _ = ls.choose_method(rods())
    assert b.name == "lightsmith-pwe"


def test_material_warnings_and_errors():
    lossy = ls.ConstantIndex(n_real=3.0, n_imag=0.01)
    p = rods(num_plane_waves=60).model_copy(update={"structures_2d": [
        Structure2D(name="rod", profile=Circle(radius_nm=80), material=lossy)]})
    assert any("lossy" in w for w in ls.solve(p, "pwe", "cpu").warnings)
    drude = ls.Drude(eps_inf=12.0, wp_ev=1.0, gamma_ev=0.0)  # eps > 0 at 800 nm, but dispersive
    p = p.model_copy(update={"structures_2d": [Structure2D(name="rod", profile=Circle(radius_nm=80), material=drude)]})
    w = ls.solve(p, "pwe", "cpu").warnings
    assert any("dispersive" in m for m in w) and any("reference_wavelength_nm not set" in m for m in w)
    metal = ls.Drude(eps_inf=1.0, wp_ev=9.0, gamma_ev=0.1)
    p = p.model_copy(update={"structures_2d": [Structure2D(profile=Circle(radius_nm=80), material=metal)]})
    with pytest.raises(ValueError, match="positive permittivity"):
        ls.solve(p, "pwe", "cpu")


def test_supports_rejects_inconsistent_problems():
    bad = rods().model_copy(update={"k_path": [KPoint(label="A", reduced=(0, 0, 0)),
                                               KPoint(label="B", reduced=(0, 0, 0.5))]})
    ok, why = ls.get_backend("pwe").supports(bad)
    assert not ok and "out-of-plane" in why


def test_overlapping_inclusions_use_raster():
    # Two overlapping circles: analytic sum would double count, so eps_hat comes from the raster.
    mat = ls.ConstantPermittivity(eps_real=8.9)
    p = rods(num_plane_waves=100).model_copy(update={"structures_2d": [
        Structure2D(profile=Circle(radius_nm=80, center_nm=(-30, 0)), material=mat),
        Structure2D(profile=Circle(radius_nm=80, center_nm=(30, 0)), material=mat)]})
    r = ls.solve(p, "pwe", "cpu")
    assert "raster" in r.provenance["eps_treatment"]["tm"]


def test_supercell_line_defect_mode_in_gap():
    # Remove the row of rods at y = 0 in a 1x5 supercell: a guided TM band crosses the bulk gap
    # (0.3224 - 0.4425), from ~0.315 at k_x = 0 to ~0.447 at k_x = pi/a, cf. Mekis et al., PRL 77, 3787 (1996).
    bulk = rods(pps=4, num_plane_waves=600)
    path = [KPoint(label="Γ", reduced=(0, 0, 0)), KPoint(label="X", reduced=(0.5, 0, 0))]
    perfect = supercell(bulk, (1, 5), k_path=path).model_copy(update={"num_bands": 7})
    wg = supercell(bulk, (1, 5), remove=[(0, 0)], k_path=path).model_copy(update={"num_bands": 7})
    assert len(perfect.structures_2d) == 5 and len(wg.structures_2d) == 4
    fp = np.asarray(ls.solve(perfect, "pwe", "cpu").frequencies["tm"])
    fw = np.asarray(ls.solve(wg, "pwe", "cpu").frequencies["tm"])
    # Perfect supercell: 5 folded first bands, then nothing until the upper gap edge.
    assert fp[:, 4].max() < 0.3225 + 1e-3 and fp[:, 5].min() > 0.4425 - 1e-3
    # Guided band (5th): 0.315, 0.332, 0.379, 0.446, 0.447 at k_x = 0 ... 0.5 (2 pi / a).
    in_gap = (fw > 0.33) & (fw < 0.44)
    assert in_gap[1:3].sum(axis=1).tolist() == [1, 1] and in_gap[1:3, 4].all()  # one guided band in the gap
    assert np.all(np.diff(fw[:, 4]) > 0)  # rising with k_x


def test_density_of_states():
    d = density_of_states(rods(nb=3), mesh=6, bins=60)
    f = np.asarray(d["frequencies"])
    dos = np.asarray(d["dos"]["tm"])
    assert np.sum(dos) * (f[1] - f[0]) == pytest.approx(3.0)
    gap = (f > 0.335) & (f < 0.43)
    assert dos[gap].sum() == 0  # TM band gap


def test_bands_from_permittivity_is_differentiable():
    torch = pytest.importorskip("torch")
    n = 32
    x = (np.arange(n) + 0.0) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rod = (np.minimum(xx, 1 - xx) ** 2 + np.minimum(yy, 1 - yy) ** 2 <= 0.2**2).astype(float)

    def f(scale):
        eps = 1 + scale * 7.9 * torch.as_tensor(rod)
        return bands_from_permittivity(eps, SQ, [(0.5, 0, 0)], num_bands=2, polarization="tm",
                                       num_plane_waves=60)[0, 1]

    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    f(s).backward()
    h = 1e-4
    fd = (f(torch.tensor(1 + h, dtype=torch.float64)) - f(torch.tensor(1 - h, dtype=torch.float64))) / (2 * h)
    assert float(s.grad) == pytest.approx(float(fd), rel=1e-5)
    assert float(s.grad) < 0  # more dielectric -> lower frequency


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
def test_gpu_matches_cpu():
    p = rods(pol="both", pps=4)
    cpu = ls.solve(p, "pwe", "cpu")
    gpu = ls.solve(p, "pwe", "gpu")
    assert gpu.provenance["framework"] == "torch"
    assert gpu.provenance["solver"] == {"te": "lobpcg", "tm": "dense"}
    # TM dense runs in single precision on the GPU; TE LOBPCG in double.
    np.testing.assert_allclose(gpu.frequencies["tm"], cpu.frequencies["tm"], atol=2e-4)
    np.testing.assert_allclose(gpu.frequencies["te"], cpu.frequencies["te"], atol=1e-6)


@pytest.mark.slow
def test_diamond_lattice_complete_gap_3d():
    # Diamond lattice of eps = 11.56 spheres (r = 0.25 a_cubic, overlapping): complete gap between bands 2
    # and 3. Converged (LOBPCG, 32/a): 0.2815 - 0.3144 a/lambda with a = |a1| = a_cubic / sqrt(2), i.e.
    # 0.398 - 0.445 in a_cubic units, 11.0 %. Dense HCS (600 plane waves) gives 0.2767 - 0.3081.
    ac = 500.0
    fcc = [(0, ac / 2, ac / 2), (ac / 2, 0, ac / 2), (ac / 2, ac / 2, 0)]
    pts = {"X": (0.5, 0, 0.5), "U": (0.625, 0.25, 0.625), "L": (0.5, 0.5, 0.5), "Γ": (0, 0, 0),
           "W": (0.5, 0.25, 0.75), "K": (0.375, 0.375, 0.75)}
    si = ls.ConstantPermittivity(eps_real=11.56)
    p = BandProblem(dimension=3, lattice_vectors_nm=fcc, background=AIR, num_bands=5, points_per_segment=3,
                    k_path=[KPoint(label=lab, reduced=pts[lab]) for lab in ["X", "U", "L", "Γ", "X", "W", "K"]],
                    structures=[Structure(geometry=Ball(center_nm=(s * ac / 8,) * 3, radius_nm=0.25 * ac),
                                          material=si) for s in (1, -1)],
                    options={"pwe": {"resolution": 16}})
    (g,) = band_gaps(ls.solve(p, "pwe", "auto"))["all"]
    assert g["between_bands"] == (2, 3)
    assert g["bottom"] == pytest.approx(0.2815, rel=5e-3) and g["top"] == pytest.approx(0.3144, rel=5e-3)
