# Re-timing of the full-paper benchmark, 2026-11-12

`benchmarks/retime.py` re-ran 1448 runs of `benchmarks/results/20261105-bem-reference` with their
recorded options, at version 0.3.0 (commit 1aa6404), on the benchmark machine (AMD Ryzen 5 5600X,
NVIDIA GeForce RTX 5070) with no other compute jobs, and without `tracemalloc`.

    uv run --no-sync python benchmarks/retime.py --out benchmarks/results/20261112-retime
    uv run --no-sync python benchmarks/retime.py --summarize benchmarks/results/20261112-retime

- The machine is a desktop with idle interactive applications. Before each timed solve the script
  waited until CPU load was at most 25 % and GPU utilisation at most 10 %. The loads it measured
  (5478 checks) had medians of 11 % (CPU) and 0 % (GPU). Each check is stored in `idle_state`.
- Runs under 0.5 s were repeated 5 times and runs under 20 s 3 times, alternating CPU and GPU.
  `wall_s` is the median.
- Every run reproduced its recorded C_ext exactly (`c_ext_vs_recorded` = 0).
- `summary.json` holds the numbers quoted in the paper. `figures/gpu_speedup.pdf` is
  `benchmarks/plots.py --retime`, left panel only re-timed.

One record was replaced. The BEM GPU run of `sphere-diel-x5-m1.2-water` took 247 s against 85 s
recorded. PyTorch's cache from earlier runs held 11.6 of the card's 12 GB, so part of the 6 GB matrix
went to system memory (Windows sysmem fallback). `retime.py` now empties the CUDA cache before each GPU
run. That run was re-timed with `--resume` in a fresh process (86.9 s; `environment-1.json`). The
original record is kept in `superseded.jsonl`.
