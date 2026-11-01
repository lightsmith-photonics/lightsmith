"""1x4 splitter tree from two levels of 1x2 MMIs.

Each MMI has 0.2 dB excess loss and a 60 nm 1 dB bandwidth, so the tree
delivers 1/4 of the input to each output minus 2 x 0.2 dB at 1550 nm, with the
roll-off away from the centre wavelength. The netlist is generated
programmatically, as an agent or a script would, and exported to Touchstone.

    uv run python examples/circuit_splitter_tree.py
"""

import tempfile
from pathlib import Path

import numpy as np

import lightsmith as ls
from lightsmith.circuits import write_touchstone
from lightsmith.problems import CircuitProblem

LEVELS = 2
mmi = {"component": "mmi1x2", "settings": {"loss_db": 0.2, "bandwidth_1db_nm": 60.0}}
guide = {"component": "waveguide", "settings": {"length_nm": 25_000.0, "loss_db_per_cm": 2.0}}

instances, connections = {}, []
for level in range(LEVELS):
    for k in range(2**level):
        instances[f"mmi_{level}_{k}"] = mmi
        if level > 0:  # feed from the parent through a waveguide
            parent, out = f"mmi_{level - 1}_{k // 2}", "o2" if k % 2 == 0 else "o3"
            instances[f"wg_{level}_{k}"] = guide
            connections += [(f"{parent},{out}", f"wg_{level}_{k},o1"), (f"wg_{level}_{k},o2", f"mmi_{level}_{k},o1")]
leaves = [f"mmi_{LEVELS - 1}_{k},{p}" for k in range(2 ** (LEVELS - 1)) for p in ("o2", "o3")]
ports = {"in": "mmi_0_0,o1", **{f"out{i + 1}": ref for i, ref in enumerate(leaves)}}

tree = CircuitProblem(instances=instances, connections=connections, ports=ports,
                      wavelengths={"start_nm": 1500, "stop_nm": 1600, "num": 101})
res = ls.solve(tree, device="cpu")
wl = np.asarray(res.wavelengths_nm)
s = res.s()
p_out = np.abs(s[:, 1:, 0]) ** 2  # [wavelength, output]
i0 = int(np.argmin(np.abs(wl - 1550)))
print(f"{len(instances)} instances, {len(connections)} connections, {len(ports) - 1} outputs")
print("power per output at 1550 nm:", np.round(p_out[i0], 5).tolist(),
      f"(ideal {0.25 * 10 ** (-0.04) * 10 ** (-2.0 * 2.5e-3 / 10):.5f})")
print(f"total at 1550 nm {p_out[i0].sum():.4f}; at 1500 nm {p_out[0].sum():.4f} (MMI roll-off)")
print(f"output uniformity {10 * np.log10(p_out[i0].max() / p_out[i0].min()):.2e} dB")

out = Path(tempfile.gettempdir()) / "splitter_tree.s5p"
write_touchstone(out, wl, s, res.ports)
print(f"wrote {out}; reuse it in another circuit as component 'sparams:{out}'")
