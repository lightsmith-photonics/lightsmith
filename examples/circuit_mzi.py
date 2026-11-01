"""Mach-Zehnder interferometer spectrum from a circuit netlist.

Two 50:50 directional couplers joined by arms that differ by dL. The bar-port
transmission is sin^2(pi neff(lambda) dL / lambda) and the free spectral range
is lambda^2 / (ng dL).

    uv run python examples/circuit_mzi.py [--plot]
"""

import sys

import numpy as np

import omniscatter as osc
from omniscatter.problems import CircuitProblem

NEFF, NG, DL = 2.4, 4.2, 20_000.0  # nm
guide = {"neff": NEFF, "ng": NG, "wl0_nm": 1550.0, "loss_db_per_cm": 2.0}

mzi = CircuitProblem(
    instances={
        "split": {"component": "coupler", "settings": {"coupling": 0.5}},
        "combine": {"component": "coupler", "settings": {"coupling": 0.5}},
        "long": {"component": "waveguide", "settings": {"length_nm": 100_000.0 + DL, **guide}},
        "short": {"component": "waveguide", "settings": {"length_nm": 100_000.0, **guide}},
        "dump": {"component": "terminator"},
    },
    connections=[("split,o3", "long,o1"), ("long,o2", "combine,o2"),
                 ("split,o4", "short,o1"), ("short,o2", "combine,o1"),
                 ("split,o2", "dump,o1")],
    ports={"in": "split,o1", "bar": "combine,o4", "cross": "combine,o3"},
    wavelengths={"start_nm": 1500, "stop_nm": 1600, "num": 2001},
)

res = osc.solve(mzi, device="cpu")
wl = np.asarray(res.wavelengths_nm)
s = res.s()
bar = np.abs(s[:, res.ports.index("bar"), res.ports.index("in")]) ** 2
cross = np.abs(s[:, res.ports.index("cross"), res.ports.index("in")]) ** 2

mins = wl[1:-1][(bar[1:-1] < bar[:-2]) & (bar[1:-1] < bar[2:])]
fsr = np.diff(mins)
centre = (mins[1:] + mins[:-1]) / 2
print(f"backend {res.backend}, {res.provenance['instances']} instances, runtime {res.provenance['runtime_s']:.3f} s")
print(f"bar-port minima (nm): {np.round(mins, 3).tolist()}")
print(f"FSR measured {fsr.mean():.4f} nm vs lambda^2/(ng dL) = {np.mean(centre**2 / (NG * DL)):.4f} nm")
print(f"extinction (bar) {10 * np.log10(bar.max() / bar.min()):.1f} dB; max bar+cross {np.max(bar + cross):.4f}")
print(f"warnings: {res.warnings}")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    plt.plot(wl, 10 * np.log10(bar), label="bar")
    plt.plot(wl, 10 * np.log10(cross), label="cross")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("transmission (dB)")
    plt.legend()
    plt.show()
