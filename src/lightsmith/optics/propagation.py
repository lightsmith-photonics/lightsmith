"""Angular-spectrum propagation of complex scalar fields in a homogeneous medium.

U(x, y, z) = IFFT2[ FFT2[U(x, y, 0)] exp(i kz z) ],  kz = sqrt(k^2 - kx^2 - ky^2),
with Im(kz) >= 0 so evanescent components decay (exp(-i omega t) convention).
Optionally band-limited (Matsushima & Shimobaba, Opt. Express 17, 19662 (2009))
to suppress the aliasing of long-distance transfer functions. Written against the
array API, so NumPy, CuPy and PyTorch inputs work (and torch inputs stay
differentiable).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def angular_spectrum(field, dx_nm: float, wavelength_nm: float, distance_nm: float | Any, *,
                     dy_nm: float | None = None, n_medium: float = 1.0, pad: int = 0,
                     band_limit: bool = True):
    """Propagate `field` (..., ny, nx) sampled at pitch (dx_nm, dy_nm) by `distance_nm`.

    pad: zero-padding added on every side (in samples) to reduce wrap-around; the output
    is cropped back to the input size. `distance_nm` may be a scalar or an array/tensor
    broadcastable against the leading axes (e.g. shape (nz, 1, 1) for a z-stack).
    """
    import array_api_compat

    xp = array_api_compat.array_namespace(field)
    dev = array_api_compat.device(field)
    dy_nm = dx_nm if dy_nm is None else dy_nm
    if pad:
        ny0, nx0 = field.shape[-2:]
        big = xp.zeros(field.shape[:-2] + (ny0 + 2 * pad, nx0 + 2 * pad), dtype=field.dtype, device=dev)
        big[..., pad:pad + ny0, pad:pad + nx0] = field
        field = big
    ny, nx = field.shape[-2:]
    k = 2 * np.pi * n_medium / wavelength_nm
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx_nm)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy_nm)
    kx2 = kx[None, :] ** 2 + ky[:, None] ** 2
    kz = np.sqrt((k**2 - kx2).astype(complex))
    if band_limit and np.ndim(distance_nm) == 0:
        # Largest spatial frequency whose transfer-function phase is sampled without aliasing.
        z = abs(float(distance_nm))
        fx_lim = 1 / (wavelength_nm / n_medium * np.sqrt((2 * z / (nx * dx_nm)) ** 2 + 1))
        fy_lim = 1 / (wavelength_nm / n_medium * np.sqrt((2 * z / (ny * dy_nm)) ** 2 + 1))
        mask = (np.abs(kx[None, :]) / (2 * np.pi) <= fx_lim) & (np.abs(ky[:, None]) / (2 * np.pi) <= fy_lim)
    else:
        mask = np.ones(kz.shape, dtype=bool)
    if hasattr(distance_nm, "shape") and not isinstance(distance_nm, np.ndarray):  # tensor: keep it differentiable
        kz_d = xp.asarray(kz * mask, dtype=field.dtype, device=dev)
        transfer = xp.exp(1j * kz_d * distance_nm) * xp.asarray(mask.astype(float), dtype=field.dtype, device=dev)
    else:
        z = np.asarray(distance_nm, dtype=float)
        transfer = xp.asarray(np.exp(1j * kz * z[..., None, None] if z.ndim else 1j * kz * z) * mask,
                              dtype=field.dtype, device=dev)
    out = xp.fft.ifft2(xp.fft.fft2(field) * transfer)
    if pad:
        out = out[..., pad:pad + ny - 2 * pad, pad:pad + nx - 2 * pad]
    return out


def focal_spot(intensity: np.ndarray, dx_nm: float) -> dict[str, float]:
    """Peak position (nm, relative to the array centre) and FWHM along x and y of a 2D intensity map."""
    a = np.asarray(intensity, dtype=float)
    iy, ix = np.unravel_index(int(np.argmax(a)), a.shape)
    peak = a[iy, ix]

    def fwhm(line, i):
        half = peak / 2
        lo = i
        while lo > 0 and line[lo] > half:
            lo -= 1
        hi = i
        while hi < len(line) - 1 and line[hi] > half:
            hi += 1
        # linear interpolation of both half-maximum crossings
        left = lo + (half - line[lo]) / (line[lo + 1] - line[lo]) if line[lo + 1] != line[lo] else lo
        right = hi - 1 + (line[hi - 1] - half) / (line[hi - 1] - line[hi]) if line[hi - 1] != line[hi] else hi
        return float((right - left) * dx_nm)

    return {"x_nm": float((ix - (a.shape[1] - 1) / 2) * dx_nm), "y_nm": float((iy - (a.shape[0] - 1) / 2) * dx_nm),
            "peak": float(peak), "fwhm_x_nm": fwhm(a[iy, :], ix), "fwhm_y_nm": fwhm(a[:, ix], iy)}
