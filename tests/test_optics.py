"""Angular-spectrum propagation (lightsmith.optics) against analytic references.

The angular-spectrum method is the exact solution of the scalar Helmholtz equation for a field
given on a plane, i.e. Rayleigh-Sommerfeld diffraction of the first kind, so closed forms of that
integral are references for it, not approximations.
"""

import numpy as np
import pytest

from lightsmith.optics import angular_spectrum, focal_spot

LAM = 500.0
K = 2 * np.pi / LAM
DX = LAM / 4
N = 1024


def _grid(n: int = N, dx: float = DX):
    c = (np.arange(n) - n // 2) * dx  # the optical axis is pixel (n // 2, n // 2)
    return np.meshgrid(c, c, indexing="xy")


def _disk(radius: float, n: int = N, dx: float = DX, supersample: int = 8) -> np.ndarray:
    """Area-weighted circular aperture (each pixel's covered fraction)."""
    x, y = _grid(n, dx)
    sub = ((np.arange(supersample) + 0.5) / supersample - 0.5) * dx
    w = np.zeros((n, n))
    for sy in sub:
        for sx in sub:
            w += (x + sx) ** 2 + (y + sy) ** 2 <= radius**2
    return w / supersample**2


def test_plane_wave_components_are_exact():
    """A propagating and an evanescent Fourier component pick up exactly exp(i kz z)."""
    x, y = _grid()
    span = N * DX
    for m, p in ((37, -12), (300, 0)):  # the second has kx = 1.17 k: evanescent
        kx, ky = 2 * np.pi * m / span, 2 * np.pi * p / span
        kz = np.sqrt(complex(K**2 - kx**2 - ky**2))
        f = np.exp(1j * (kx * (x - x[0, 0]) + ky * (y - y[0, 0])))
        z = 3000.0 if kz.imag == 0 else 100.0
        u = angular_spectrum(f, DX, LAM, z, band_limit=False)
        assert np.max(np.abs(u - f * np.exp(1j * kz * z))) < 1e-12
    assert abs(np.exp(1j * kz * 100.0)) < 0.5  # the evanescent component really decayed


def test_power_is_conserved_for_propagating_fields():
    rng = np.random.default_rng(1)
    spec = np.zeros((N, N), complex)
    kx = 2 * np.pi * np.fft.fftfreq(N, DX)
    prop = (kx[None, :] ** 2 + kx[:, None] ** 2) < (0.9 * K) ** 2
    spec[prop] = rng.normal(size=prop.sum()) + 1j * rng.normal(size=prop.sum())
    f = np.fft.ifft2(spec)
    u = angular_spectrum(f, DX, LAM, 12_345.0, band_limit=False)
    assert abs(np.sum(np.abs(u) ** 2) / np.sum(np.abs(f) ** 2) - 1) < 1e-12


@pytest.mark.parametrize("z_over_zf", [1.0, 2.0])
def test_circular_aperture_on_axis_matches_rayleigh_sommerfeld(z_over_zf):
    """On-axis intensity behind a uniformly lit disk: |e^{ikz} - z/R e^{ikR}|^2, R = sqrt(z^2 + a^2)."""
    a = 10 * LAM
    z = z_over_zf * a**2 / LAM  # an on-axis maximum (I = 4) and a far point (I = 2)
    u = angular_spectrum(_disk(a).astype(complex), DX, LAM, z)
    r = np.sqrt(z**2 + a**2)
    ref = abs(np.exp(1j * K * z) - z / r * np.exp(1j * K * r)) ** 2
    num = abs(u[N // 2, N // 2]) ** 2
    assert abs(num / ref - 1) < 3e-3


@pytest.mark.parametrize("z_over_zr", [0.5, 1.0, 2.0])
def test_gaussian_beam_matches_paraxial_theory(z_over_zr):
    """w0 = 10 lambda, so the paraxial approximation itself is good to (lambda / pi w0)^2 = 1e-3."""
    w0 = 10 * LAM
    zr = np.pi * w0**2 / LAM
    z = z_over_zr * zr
    x, y = _grid()
    u = angular_spectrum(np.exp(-(x**2 + y**2) / w0**2).astype(complex), DX, LAM, z)
    wz = w0 * np.sqrt(1 + z_over_zr**2)
    spot = focal_spot(np.abs(u) ** 2, DX)
    assert abs(spot["peak"] / (w0 / wz) ** 2 - 1) < 2e-3
    assert abs(spot["fwhm_x_nm"] / (wz * np.sqrt(2 * np.log(2))) - 1) < 2e-3
    assert abs(spot["fwhm_y_nm"] / spot["fwhm_x_nm"] - 1) < 1e-9


def test_ideal_lens_focus():
    """A disk with the hyperbolic lens phase: exact on-axis intensity at the focus, and an Airy spot.

    Rayleigh-Sommerfeld at z = f, where the lens phase cancels the propagation phase exactly:
    U(f) = f e^{ikf} [(1/f - 1/R_a) - i k ln(R_a / f)], R_a = sqrt(a^2 + f^2).
    """
    a, f = 10 * LAM, 40 * LAM
    x, y = _grid()
    lens = _disk(a) * np.exp(-1j * K * (np.sqrt(x**2 + y**2 + f**2) - f))
    u = angular_spectrum(lens, DX, LAM, f)
    ra = np.sqrt(a**2 + f**2)
    ref = f**2 * ((1 / f - 1 / ra) ** 2 + K**2 * np.log(ra / f) ** 2)
    num = abs(u[N // 2, N // 2]) ** 2
    assert abs(num / ref - 1) < 3e-3
    spot = focal_spot(np.abs(u) ** 2, DX)
    na = a / ra
    # focal_spot measures from the array centre, half a pixel from the axis pixel of an even grid
    assert abs(spot["x_nm"] - DX / 2) < 1e-9 and abs(spot["y_nm"] - DX / 2) < 1e-9
    assert abs(spot["fwhm_x_nm"] / (0.5145 * LAM / na) - 1) < 0.01  # Airy FWHM, low NA (0.24)


def test_padding_is_zero_padding_then_crop():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(64, 80)) + 1j * rng.normal(size=(64, 80))
    big = np.pad(f, 16)
    ref = angular_spectrum(big, 50.0, LAM, 4000.0)[16:-16, 16:-16]
    assert np.max(np.abs(angular_spectrum(f, 50.0, LAM, 4000.0, pad=16) - ref)) < 1e-12


def test_z_stack_broadcasts_like_a_loop():
    rng = np.random.default_rng(3)
    f = rng.normal(size=(48, 48)) + 1j * rng.normal(size=(48, 48))
    zs = np.array([500.0, 2000.0, 7000.0])
    stack = angular_spectrum(f, 60.0, LAM, zs[:, None, None], band_limit=False)
    for i, z in enumerate(zs):
        assert np.max(np.abs(stack[i] - angular_spectrum(f, 60.0, LAM, z, band_limit=False))) < 1e-12


def test_torch_matches_numpy_and_is_differentiable():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    f = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    ref = angular_spectrum(f, 60.0, LAM, 3000.0)
    out = angular_spectrum(torch.tensor(f), 60.0, LAM, 3000.0)
    assert np.max(np.abs(out.numpy() - ref)) < 1e-12

    def power_at_centre(z):
        u = angular_spectrum(torch.tensor(f), 60.0, LAM, z, band_limit=False)
        return (u[32, 32].abs() ** 2)

    z = torch.tensor(3000.0, dtype=torch.float64, requires_grad=True)
    power_at_centre(z).backward()
    h = 1e-3
    fd = (power_at_centre(torch.tensor(3000.0 + h, dtype=torch.float64))
          - power_at_centre(torch.tensor(3000.0 - h, dtype=torch.float64))) / (2 * h)
    assert abs(z.grad.item() / fd.item() - 1) < 1e-6

    phase = torch.zeros((64, 64), dtype=torch.float64, requires_grad=True)
    u = angular_spectrum(torch.tensor(f) * torch.exp(1j * phase), 60.0, LAM, 3000.0)
    (u[32, 32].abs() ** 2).backward()
    g = phase.grad.numpy()
    e = np.zeros((64, 64))
    e[10, 20] = 1e-6

    def p(ph):
        return abs(angular_spectrum(f * np.exp(1j * ph), 60.0, LAM, 3000.0)[32, 32]) ** 2

    assert abs(g[10, 20] - (p(e) - p(-e)) / 2e-6) < 1e-6 * max(1.0, abs(g[10, 20]))


@pytest.mark.gpu
def test_cuda_matches_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU with PyTorch")
    rng = np.random.default_rng(5)
    f = rng.normal(size=(256, 256)) + 1j * rng.normal(size=(256, 256))
    ref = angular_spectrum(f, 60.0, LAM, 3000.0)
    out = angular_spectrum(torch.tensor(f, dtype=torch.complex128, device="cuda"), 60.0, LAM, 3000.0)
    assert np.max(np.abs(out.cpu().numpy() - ref)) < 1e-10


@pytest.mark.slow
def test_metalens_example_focuses_at_the_design_length():
    """examples/metalens_rcwa.py: RCWA phase library + angular spectrum, end to end (CPU, 20-60 s)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "metalens_rcwa.py"
    spec = importlib.util.spec_from_file_location("metalens_rcwa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.main(device="cpu")
    assert abs(out["z_focus_nm"] / out["design_focal_nm"] - 1) <= 0.025  # the z scan's step
    assert max(map(abs, out["spot_offset_nm"])) <= mod.PERIOD
    # A discretised lens with |t| varying by 15 % may not beat the diffraction limit by much, nor miss it.
    for w in (out["fwhm_x_nm"], out["fwhm_y_nm"]):
        assert 0.85 < w / out["diffraction_limit_nm"] < 1.15
