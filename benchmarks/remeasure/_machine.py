"""Machine load recorded with the timings of the remeasure scripts, so readers can judge contention.

Earlier runs of these scripts (2026-11-01 to 11-05) shared the GPU and CPU with the full-paper benchmark
and were marked provisional. Since 2026-11-12 they run with no other compute jobs on the machine; the
CPU and GPU load sampled before and after each script is stored with its results.
"""

from __future__ import annotations

import datetime as _dt
import subprocess

NOTE = ("timed with no other compute jobs on the machine (an interactive desktop with idle applications); "
        "machine_state holds the CPU and GPU load sampled before and after")


def state() -> dict:
    out: dict = {"time": _dt.datetime.now().isoformat(timespec="seconds")}
    try:
        import psutil

        out["cpu_pct"] = psutil.cpu_percent(interval=0.5)
    except Exception as e:  # noqa: BLE001
        out["cpu_error"] = str(e)
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, m = (float(x) for x in s.strip().split(","))
        out.update(gpu_util_pct=u, gpu_mem_used_mib=m)
    except Exception as e:  # noqa: BLE001
        out["gpu_error"] = str(e)
    return out
