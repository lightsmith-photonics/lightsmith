"""All-pass ring resonator: FSR, extinction and critical coupling.

The ring is built twice: from the ``ring_single`` compact model, and as a
netlist of a directional coupler closed by a waveguide loop. Both must agree
with the analytic all-pass transmission

    T = (a^2 - 2 a t cos(phi) + t^2) / (1 - 2 a t cos(phi) + (a t)^2),

with FSR = lambda^2 / (ng L) and zero transmission on resonance when t = a
(critical coupling). The gradient of the transmission with respect to neff is
then computed with PyTorch autograd (if installed).

    uv run python examples/circuit_ring.py [--plot]
"""

import math
import sys

import numpy as np

import omniscatter as osc
from omniscatter.circuits import circuit_sparameters
from omniscatter.problems import CircuitProblem

NEFF, NG, WL0 = 2.4, 4.2, 1550.0
RADIUS = 10_000.0
LOSS = 3.0  # dB/cm
L = 2 * math.pi * RADIUS
a = 10 ** (-LOSS * L * 1e-7 / 20)  # roundtrip amplitude
coupling = 1 - a**2  # critical coupling: t = sqrt(1 - kappa^2) = a
guide = {"neff": NEFF, "ng": NG, "wl0_nm": WL0, "loss_db_per_cm": LOSS}
wavelengths = {"start_nm": 1540, "stop_nm": 1560, "num": 4001}

model = CircuitProblem(
    instances={"ring": {"component": "ring_single", "settings": {"radius_nm": RADIUS, "coupling": coupling, **guide}}},
    ports={"in": "ring,o1", "out": "ring,o2"}, wavelengths=wavelengths)
netlist = CircuitProblem(
    instances={"dc": {"component": "coupler", "settings": {"coupling": coupling}},
               "loop": {"component": "waveguide", "settings": {"length_nm": L, **guide}}},
    connections=[("dc,o3", "loop,o1"), ("loop,o2", "dc,o2")],
    ports={"in": "dc,o1", "out": "dc,o4"}, wavelengths=wavelengths)

r1, r2 = osc.solve(model, device="cpu"), osc.solve(netlist, device="cpu")
wl = np.asarray(r1.wavelengths_nm)
T1, T2 = np.abs(r1.s()[:, 1, 0]) ** 2, np.abs(r2.s()[:, 1, 0]) ** 2
phi = 2 * np.pi * (NEFF + (NEFF - NG) * (wl - WL0) / WL0) * L / wl
t = math.sqrt(1 - coupling)
T_ref = (a**2 - 2 * a * t * np.cos(phi) + t**2) / (1 - 2 * a * t * np.cos(phi) + (a * t) ** 2)

mins = wl[1:-1][(T1[1:-1] < T1[:-2]) & (T1[1:-1] < T1[2:])]
fsr = np.diff(mins)
print(f"ring L = {L / 1000:.2f} um, roundtrip loss {-20 * math.log10(a):.4f} dB, critical coupling {coupling:.5f}")
print(f"max |T_model - T_analytic| = {np.max(np.abs(T1 - T_ref)):.2e}; netlist vs model = {np.max(np.abs(T1 - T2)):.2e}")
print(f"FSR measured {fsr.mean():.4f} nm vs lambda^2/(ng L) = {np.mean(((mins[1:] + mins[:-1]) / 2) ** 2 / (NG * L)):.4f} nm")
print(f"extinction on the sampled grid {10 * np.log10(T1.max() / T1.min()):.1f} dB (exactly zero on resonance)")

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    neff = torch.tensor(NEFF, dtype=torch.float64, requires_grad=True)
    sol = circuit_sparameters(model, device="torch-cpu", params={"ring": {"neff": neff}},
                              wavelengths_nm=[float(mins[0]) + 0.02])
    T = sol["out", "in"].abs()[0] ** 2
    T.backward()
    print(f"dT/dneff on the resonance flank: {neff.grad.item():.3f} (autograd)")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    plt.plot(wl, 10 * np.log10(np.maximum(T1, 1e-12)), label="ring_single")
    plt.plot(wl, 10 * np.log10(np.maximum(T_ref, 1e-12)), "--", label="analytic")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("through (dB)")
    plt.legend()
    plt.show()
