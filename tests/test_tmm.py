import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends.tmm import optimize_thicknesses, tmm_tensors
from lightsmith.device import _torch_cuda
from lightsmith.problems import Layer, LayerStackProblem

CI = ls.ConstantIndex


def stack(layers=(), sup=1.0, sub=1.5, wl=(500.0,), angles=(0.0,), pols=("s", "p"), **kw):
    mat = lambda n: n if not isinstance(n, (int, float, complex)) else CI(n_real=complex(n).real,  # noqa: E731
                                                                           n_imag=complex(n).imag)
    return LayerStackProblem(superstrate=mat(sup), substrate=mat(sub),
                             layers=[Layer(thickness_nm=d, material=mat(n), **kw) for n, d in layers],
                             wavelengths=list(wl), angles_deg=list(angles), polarizations=list(pols))


def amp(res, key, pol):
    a = np.asarray(getattr(res, key)[pol])
    return a[..., 0] + 1j * a[..., 1]


def fresnel(n1, n2, theta):
    """Textbook Fresnel amplitudes in Byrnes' convention (r_p = H ratio, t_p = E ratio)."""
    ci = np.cos(theta)
    ct = np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2 + 0j)
    ct = np.where(ct.imag < 0, -ct, ct)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    ts = 2 * n1 * ci / (n1 * ci + n2 * ct)
    tp = 2 * n1 * ci / (n2 * ci + n1 * ct)
    return rs, rp, ts, tp, ct


def test_fresnel_single_interface():
    th = np.array([0.0, 20.0, 45.0, 70.0, 89.0])
    res = ls.solve(stack(angles=th), "tmm", "cpu")
    rs, rp, ts, tp, ct = fresnel(1.0, 1.5, np.radians(th))
    np.testing.assert_allclose(amp(res, "r_amplitude", "s")[:, 0], rs, atol=1e-13)
    np.testing.assert_allclose(amp(res, "r_amplitude", "p")[:, 0], rp, atol=1e-13)
    np.testing.assert_allclose(amp(res, "t_amplitude", "s")[:, 0], ts, atol=1e-13)
    np.testing.assert_allclose(amp(res, "t_amplitude", "p")[:, 0], tp, atol=1e-13)
    np.testing.assert_allclose(np.asarray(res.r["s"])[:, 0], np.abs(rs) ** 2, atol=1e-14)
    np.testing.assert_allclose(np.asarray(res.t["s"])[:, 0], 1.5 * ct.real / np.cos(np.radians(th)) * np.abs(ts) ** 2,
                               atol=1e-13)


def test_brewster_angle():
    th_b = np.degrees(np.arctan(1.5))
    res = ls.solve(stack(angles=[th_b]), "tmm", "cpu")
    assert res.r["p"][0][0] < 1e-28
    assert res.r["s"][0][0] > 0.14


def test_total_internal_reflection():
    th = np.array([45.0, 60.0, 80.0])  # critical angle glass -> air is 41.8 deg
    res = ls.solve(stack(sup=1.5, sub=1.0, angles=th), "tmm", "cpu")
    for p in "sp":
        np.testing.assert_allclose(res.r[p], 1.0, atol=1e-14)
        np.testing.assert_allclose(res.t[p], 0.0, atol=1e-14)
    rs, rp, *_ = fresnel(1.5, 1.0, np.radians(th))  # evanescent branch: Im(cos theta_t) > 0
    np.testing.assert_allclose(amp(res, "r_amplitude", "s")[:, 0], rs, atol=1e-13)
    np.testing.assert_allclose(amp(res, "r_amplitude", "p")[:, 0], rp, atol=1e-13)


def test_frustrated_tir_and_energy_conservation():
    # Glass / thin air gap / glass beyond the critical angle: light tunnels through the gap.
    gaps = [50.0, 200.0, 800.0]
    t = [ls.solve(stack([(1.0, g)], sup=1.5, sub=1.5, angles=[50.0]), "tmm", "cpu").t["s"][0][0] for g in gaps]
    assert 1 > t[0] > t[1] > t[2] > 0 and t[2] < 1e-3
    rng = np.random.default_rng(1)
    lossless = stack([(n, d) for n, d in zip(rng.uniform(1.3, 2.5, 12), rng.uniform(20, 300, 12))],
                     wl=np.linspace(400, 900, 26), angles=[0, 30, 60, 85])
    res = ls.solve(lossless, "tmm", "cpu")
    for p in "sp":
        np.testing.assert_allclose(np.asarray(res.r[p]) + np.asarray(res.t[p]), 1.0, atol=1e-12)
        np.testing.assert_allclose(res.a[p], 0.0, atol=1e-12)
    lossy = ls.solve(stack([(complex(2.0, 0.3), 40.0), (1.4, 90.0)], wl=[450, 600], angles=[0, 40]), "tmm", "cpu")
    for p in "sp":
        a = np.asarray(lossy.a[p])
        assert np.all((a > 0.01) & (a < 1))
        np.testing.assert_allclose(np.asarray(lossy.r[p]) + np.asarray(lossy.t[p]) + a, 1.0, atol=1e-14)


def test_quarter_wave_ar_coating():
    n_s, lam0 = 1.52, 550.0
    n_c = np.sqrt(n_s)
    wl = np.linspace(450, 650, 41)
    res = ls.solve(stack([(n_c, lam0 / (4 * n_c))], sub=n_s, wl=wl, pols=["s"]), "tmm", "cpu")
    r = np.asarray(res.r["s"][0])
    assert r[np.argmin(np.abs(wl - lam0))] < 1e-28
    # Single-film Airy formula as an independent closed form.
    r01, r12 = (1 - n_c) / (1 + n_c), (n_c - n_s) / (n_c + n_s)
    e = np.exp(2j * 2 * np.pi / wl * n_c * lam0 / (4 * n_c))
    np.testing.assert_allclose(r, np.abs((r01 + r12 * e) / (1 + r01 * r12 * e)) ** 2, atol=1e-14)


def test_bragg_mirror_stop_band():
    n_h, n_l, n_s, lam0, pairs = 2.3, 1.45, 1.52, 600.0, 10
    layers = [(n_h, lam0 / (4 * n_h)), (n_l, lam0 / (4 * n_l))] * pairs
    half = (2 / np.pi) * np.arcsin((n_h - n_l) / (n_h + n_l))  # relative half-width in frequency
    x = np.linspace(-3, 3, 241)  # frequency offset in units of the half-width
    res = ls.solve(stack(layers, sub=n_s, wl=lam0 / (1 + x * half), pols=["s"]), "tmm", "cpu")
    r = np.asarray(res.r["s"][0])
    y = (n_h / n_l) ** (2 * pairs) * n_s
    assert r[120] == pytest.approx(((1 - y) / (1 + y)) ** 2, abs=1e-12)  # centre: admittance (n_h/n_l)^2N n_s
    assert np.all(r[np.abs(x) <= 0.8] > 0.98)  # inside the analytic stop band
    assert r[np.abs(x) >= 1.5].mean() < 0.25  # outside: only side lobes


def test_incoherent_thick_slab_matches_closed_form():
    n, k, d, lam = 1.5, 2e-6, 1e6, 600.0  # 1 mm weakly absorbing slab in air
    res = ls.solve(stack([(complex(n, k), d)], sub=1.0, wl=[lam], coherent=False), "tmm", "cpu")
    eps = complex(n, k) ** 2
    r1 = abs((1 - np.sqrt(eps)) / (1 + np.sqrt(eps))) ** 2
    p = np.exp(-4 * np.pi * np.sqrt(eps).imag * d / lam)
    for pol in "sp":
        assert res.r[pol][0][0] == pytest.approx(r1 + (1 - r1) ** 2 * r1 * p**2 / (1 - r1**2 * p**2), rel=1e-10)
        assert res.t[pol][0][0] == pytest.approx((1 - r1) ** 2 * p / (1 - r1**2 * p**2), rel=1e-10)
    assert res.r_amplitude is None


def test_incoherent_equals_phase_averaged_coherent():
    # AR film on a thick slab: the incoherent result is the coherent one averaged over the slab phase.
    lam, angle = 550.0, 30.0
    slab = 20000.0
    film = (1.38, 99.6)
    p_inc = stack([film, (1.52, slab)], sub=1.33, wl=[lam], angles=[angle])
    p_inc = p_inc.model_copy(update={"layers": [p_inc.layers[0], p_inc.layers[1].model_copy(update={"coherent": False})]})
    inc = ls.solve(p_inc, "tmm", "cpu")
    # Sweep the slab thickness over one fringe period (lam / (2 n cos theta_t)) and average.
    n_s = 1.52
    cos_t = np.sqrt(1 - (np.sin(np.radians(angle)) / n_s) ** 2)
    period = lam / (2 * n_s * cos_t)
    ds = slab + np.linspace(0, period, 400, endpoint=False)
    th = np.array([[film[1], d] for d in ds])
    base = stack([film, (1.52, slab)], sub=1.33, wl=[lam], angles=[angle])
    vals = {p: [] for p in "sp"}
    for t in th:
        out = tmm_tensors(base, "cpu", thicknesses=list(t))
        for p in "sp":
            vals[p].append(float(out["R"][p][0, 0]))
    for p in "sp":
        assert inc.r[p][0][0] == pytest.approx(np.mean(vals[p]), rel=1e-6)


def test_tmm_matches_rcwa_for_unpatterned_stack():
    p = stack([(complex(2.0, 0.2), 80.0), (1.38, 120.0), (complex(0.2, 3.0), 15.0)], sub=1.5,
              wl=np.linspace(400, 800, 9), angles=[0, 35, 70])
    a, b = ls.solve(p, "tmm", "cpu"), ls.solve(p, "rcwa", "cpu")
    for pol in "sp":
        for q in ("r", "t", "a"):
            np.testing.assert_allclose(getattr(a, q)[pol], getattr(b, q)[pol], atol=1e-12)
        np.testing.assert_allclose(a.r_amplitude[pol], b.r_amplitude[pol], atol=1e-12)
        np.testing.assert_allclose(a.t_amplitude[pol], b.t_amplitude[pol], atol=1e-12)


def test_rcwa_refuses_incoherent_layers():
    p = stack([(1.5, 1e6)], coherent=False)
    ok, why = ls.get_backend("rcwa", kind="layer_stack").supports(p)
    assert not ok and "incoherent" in why


def test_gradients_match_finite_differences():
    torch = pytest.importorskip("torch")
    p = stack([(2.0, 80.0), (1.38, 120.0)], sub=1.5, wl=[450.0, 600.0], angles=[0.0, 40.0])

    def loss(d, e):
        out = tmm_tensors(p, "torch-cpu", thicknesses=d, eps={0: e})
        return out["R"]["s"].sum() + 0.5 * out["T"]["p"].sum() + out["A"]["s"].sum()

    d = torch.tensor([80.0, 120.0], dtype=torch.float64, requires_grad=True)
    e = torch.tensor(complex(4.0, 0.3), dtype=torch.complex128, requires_grad=True)
    loss(d, e).backward()
    h = 1e-5
    for i in range(2):
        dp, dm = d.detach().clone(), d.detach().clone()
        dp[i] += h
        dm[i] -= h
        fd = (loss(dp, e.detach()) - loss(dm, e.detach())) / (2 * h)
        assert float(d.grad[i]) == pytest.approx(float(fd), rel=1e-6, abs=1e-10)
    # d loss / d Re(eps) and / d Im(eps): torch's complex gradient is dL/dRe + i dL/dIm.
    for unit, part in ((1.0, "real"), (1j, "imag")):
        fd = (loss(d.detach(), e.detach() + h * unit) - loss(d.detach(), e.detach() - h * unit)) / (2 * h)
        assert float(getattr(e.grad, part)) == pytest.approx(float(fd), rel=1e-6, abs=1e-10)


def test_optimize_ar_coating_finds_quarter_wave():
    pytest.importorskip("torch")
    p = stack([(1.38, 60.0)], sub=1.52, wl=[550.0], pols=["s"])
    best, hist = optimize_thicknesses(p, "min_reflectance", steps=150, lr=4.0)
    assert best.layers[0].thickness_nm == pytest.approx(550.0 / (4 * 1.38), abs=0.5)
    assert hist["final_loss"] < hist["loss"][0]


@pytest.mark.gpu
@pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
def test_tmm_gpu_matches_cpu():
    rng = np.random.default_rng(0)
    p = stack([(n, d) for n, d in zip(rng.uniform(1.3, 2.5, 20), rng.uniform(20, 200, 20))],
              wl=np.linspace(400, 900, 101), angles=np.linspace(0, 80, 9))
    cpu, gpu = ls.solve(p, "tmm", "cpu"), ls.solve(p, "tmm", "gpu")
    assert gpu.provenance["framework"] == "torch"
    for pol in "sp":
        np.testing.assert_allclose(gpu.r[pol], cpu.r[pol], atol=1e-12)
