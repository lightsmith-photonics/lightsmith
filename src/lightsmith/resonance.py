"""Resonances (complex frequencies, Q factors, amplitudes) of a ringing time signal.

``harmonic_inversion`` fits a sum of decaying exponentials,

    x_c(t) = sum_k a_ck exp(-i w_k t),   w_k = omega_k - i gamma_k / 2,   Q_k = omega_k / gamma_k,

shared by every channel c (field components at several probe points), by a multi-channel matrix pencil
(Hua & Sarkar, IEEE TASSP 38, 814 (1990)): the job Harminv does by filter diagonalisation
(Mandelshtam & Taylor, J. Chem. Phys. 107, 6756 (1997)). An oversampled signal is first averaged and
subsampled down to a Nyquist frequency of 1.25 omega_max, so the pencil sees a few hundred samples
whatever the time step. A real signal holds each mode at +omega and -omega; both stay below the new
Nyquist frequency, so neither aliases.

Time dependence exp(-i w t), as everywhere in lightsmith: decaying modes have Im w < 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mode:
    omega: complex  # complex angular frequency, in the units of 1 / dt
    amplitude: float  # rms over channels of |a_ck| at the first sample
    residual: float  # relative residual of the whole fit

    @property
    def q(self) -> float:
        return float(self.omega.real / (-2 * self.omega.imag)) if self.omega.imag < 0 else float("inf")


def harmonic_inversion(x: np.ndarray, dt: float, omega_min: float, omega_max: float, *, max_modes: int = 30,
                       svd_tol: float = 1e-5, max_samples: int = 800, min_q: float = 0.5) -> list[Mode]:
    """Modes of x (channels x samples, or one channel), sampled every dt, with omega_min <= Re w <= omega_max.

    Modes are sorted by amplitude. svd_tol drops singular values below svd_tol * the largest (the noise
    floor); max_samples caps the subsampled record (a decaying signal's tail is mostly noise)."""
    x = np.atleast_2d(np.asarray(x))
    n_ch, n = x.shape
    nyq = np.pi / dt
    if not 0 <= omega_min < omega_max < nyq:
        raise ValueError(f"need 0 <= omega_min < omega_max < pi/dt = {nyq:.4g}")
    D = max(1, min(int(nyq / (1.25 * omega_max)), n // 16))
    m = min(n // D, max_samples)
    if m < 8:
        raise ValueError("harmonic_inversion needs at least 8 samples after subsampling; record longer")
    # boxcar average over D samples, then keep every D-th: suppresses what lies above the new Nyquist frequency
    y = x[:, :m * D].reshape(n_ch, m, D).mean(axis=2)
    dtd = D * dt
    L = m // 2
    H = np.concatenate([np.lib.stride_tricks.sliding_window_view(ch, L + 1) for ch in y])  # (n_ch (m - L), L + 1)
    _, s, vh = np.linalg.svd(H, full_matrices=False)
    k = int(min(4 * max_modes, np.sum(s > svd_tol * s[0])))  # +-omega pairs, out-of-band and noise poles
    if k == 0:
        return []
    V = vh[:k].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    z = z[np.abs(z) > 1e-12]
    w = 1j * np.log(z) / dtd  # principal branch: |Re w| <= pi / dtd
    Z = z[None, :] ** np.arange(m)[:, None]
    B, *_ = np.linalg.lstsq(Z, y.T, rcond=None)  # (k, n_ch)
    residual = float(np.linalg.norm(Z @ B - y.T) / max(np.linalg.norm(y), 1e-300))
    # the boxcar scales a mode exp(-i w t) by mean_j exp(-i w j dt) at the start of each block
    gain = np.exp(-1j * np.outer(w, np.arange(D)) * dt).mean(axis=1)
    amp = np.sqrt(np.mean(np.abs(B / gain[:, None]) ** 2, axis=1))
    modes = [Mode(complex(wk), float(ak), residual) for wk, ak in zip(w, amp)
             if omega_min <= wk.real <= omega_max and wk.imag < 0 and wk.real / (-2 * wk.imag) >= min_q]
    modes.sort(key=lambda md: -md.amplitude)
    return modes[:max_modes]
