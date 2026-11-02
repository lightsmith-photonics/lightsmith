"""Harmonic inversion (lightsmith.resonance) on synthetic ring-down signals with known modes."""

import numpy as np
import pytest

from lightsmith.resonance import harmonic_inversion

MODES = [(0.0120, 35.6, 1.0), (0.0100, 2000.0, 0.3), (0.0145, 8.0, 0.5)]  # omega, Q, real amplitude


def signal(dt, n, channels=4, noise=1e-6, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    out = []
    for c in range(channels):
        s = np.zeros(n)
        for w, q, a in MODES:
            s += a * np.real(np.exp(-1j * (w - 1j * w / (2 * q)) * t + 1j * c))
        out.append(s + noise * rng.standard_normal(n))
    return np.array(out)


@pytest.mark.parametrize("dt,n", [(0.5, 200_000), (40.0, 6000)])  # heavily oversampled; near Nyquist
def test_recovers_frequencies_q_and_amplitudes(dt, n):
    modes = harmonic_inversion(signal(dt, n), dt, 0.008, 0.016)
    assert len(modes) == 3  # no spurious poles
    for (w, q, a), m in zip(sorted(MODES, key=lambda md: -md[2]), modes):
        assert m.omega.real == pytest.approx(w, rel=1e-6)
        assert m.q == pytest.approx(q, rel=1e-3)
        assert m.amplitude == pytest.approx(a / 2, rel=1e-3)  # a cos = (a/2) e^{-iwt} + c.c.
    assert modes[0].residual < 1e-4


def test_band_limits_the_modes():
    modes = harmonic_inversion(signal(40.0, 6000), 40.0, 0.011, 0.013)
    assert [round(m.omega.real, 6) for m in modes] == [0.012]


def test_rejects_a_band_above_nyquist():
    with pytest.raises(ValueError, match="pi/dt"):
        harmonic_inversion(signal(40.0, 100), 40.0, 0.01, 0.1)
