"""Design flow from components to a circuit: an unbalanced MZI in the generic 220 nm SOI technology.

1. Build the MZI layout from parametric cells (two 1x2 MMIs, four bends, straights) and write it to GDS.
2. Simulate each distinct component once (lightsmith.solve picks EME for the MMIs, FDFD for the bends),
   fit compact models and cache them by parameter hash.
3. Solve the circuit netlist of the parts ("pdk:<cell>" instances) on a dense wavelength grid.
4. Simulate the whole MZI as one device (2D FDFD) at a few wavelengths and compare with the circuit.

    uv run python examples/pdk_mzi.py [--no-direct] [--plot]

Everything is 2D (plan view, effective-index materials). Outputs go to examples/output/. Model files are
cached in the lightsmith cache directory (LIGHTSMITH_PDK_CACHE overrides), so a second run skips step 2.
To compare like with like, step 2 and step 4 use the same solver and grid (FDFD, dl = 25 nm) when the
direct simulation is enabled, with 16x subpixel sampling (at 4x, where a waveguide's edges fall on the grid
shifts its neff by ~7e-3, which over tens of um is a visible phase error); with --no-direct the parts use
the default solvers (EME/FDFD, mode solves for straights).
"""

import sys
import time
from pathlib import Path

import numpy as np

import lightsmith as ls
from lightsmith.pdk import generic_soi, get_component
from lightsmith.pdk.cells import mzi_arm_lengths
from lightsmith.pdk.flow import FlowSettings, component_model, netlist_problem, simulate

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
direct = "--no-direct" not in sys.argv

tech = generic_soi()
params = {"delta_length_nm": 4000.0, "radius_nm": 3000.0}
mzi = get_component("mzi", **params)

# 1. Layout -> GDS (hierarchical: one cell per distinct component, placed by references).
gds = mzi.write_gds(OUT / "pdk_mzi.gds", tech)
print(f"GDS: {gds} (top cell {mzi.cell_name}, {len(mzi.instances)} instances)")
ref, delay = mzi_arm_lengths(params)
ng = float(tech.group_index([1550.0])[0])
print(f"arm lengths {ref:.0f} / {delay:.0f} nm; expected FSR ~ {1550**2 / (ng * (delay - ref)):.1f} nm (ng = {ng:.3f})")

# 2. Per-component simulation and compact models.
if direct:
    flow = FlowSettings(wl_min_nm=1500, wl_max_nm=1600, num_points=11, method="fdfd", waveguide_model="device",
                        sim_options={"fdfd": {"dl_nm": 25.0, "subsamples": 16}})
else:
    flow = FlowSettings(wl_min_nm=1500, wl_max_nm=1600, num_points=11)
seen = set()
for name, inst in mzi.netlist["instances"].items():
    cell = inst["component"].removeprefix("pdk:")
    key = (cell, tuple(sorted(inst["settings"].items(), key=str)))
    if key in seen:
        continue
    seen.add(key)
    t0 = time.perf_counter()
    m = component_model(cell, flow, **inst["settings"])
    s = m.summary()
    print(f"  {cell:14s} {s['backend']:24s} fit={s['fit']:15s} poles={s['poles']} "
          f"fit_err={s['fit_error']:.1e} ({time.perf_counter() - t0:.1f} s)")

# 3. Circuit of the parts.
wl = np.linspace(1500, 1600, 1001)
circuit = netlist_problem(mzi, wl.tolist(), flow)
res = ls.solve(circuit, device="cpu")
t_circ = np.abs(res.s()[:, res.ports.index("o2"), res.ports.index("o1")]) ** 2
mins = wl[1:-1][(t_circ[1:-1] < t_circ[:-2]) & (t_circ[1:-1] < t_circ[2:])]
print(f"circuit: {len(circuit.instances)} instances; transmission minima at {np.round(mins, 1).tolist()} nm; "
      f"max T = {t_circ.max():.3f}")

# 4. Direct simulation of the whole layout.
if direct:
    wl_d = np.linspace(1500, 1600, 9)
    t0 = time.perf_counter()
    whole = simulate(mzi, tech, wl_d, method="fdfd", direct=True, options=flow.sim_options)
    t_dir = np.abs(whole.s()[:, 1, 0]) ** 2
    t_c = np.interp(wl_d, wl, t_circ)
    print(f"direct FDFD of the whole MZI ({whole.provenance.get('grid')}, {time.perf_counter() - t0:.0f} s):")
    for w, a, b in zip(wl_d, t_dir, t_c):
        print(f"  {w:7.1f} nm  direct {a:.3f}  circuit {b:.3f}  diff {a - b:+.3f}")
    print(f"max |T_direct - T_circuit| = {np.abs(t_dir - t_c).max():.3f}")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    plt.plot(wl, t_circ, label="circuit of simulated parts")
    if direct:
        plt.plot(wl_d, t_dir, "o", label="direct simulation")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("|S21|^2")
    plt.legend()
    plt.savefig(OUT / "pdk_mzi.png", dpi=150)
    plt.show()
