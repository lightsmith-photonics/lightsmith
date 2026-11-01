This directory is the first smoke run, made with the definition now called `smoke-extended`
(`benchmarks/problems.py::smoke_extended`): every DDA variant (center, weighted, extrapolate where
listed) on every problem, DDA max_iter 10000, CPU budget 1200 dipoles / 1.5e9 FDTD cell-steps.
The `suite` column in runs.csv says "smoke" because the suite was renamed afterwards; `suite.json`
here is the exact definition that ran. It took 44 min (2628 s) on a desktop CPU plus a shared RTX 5070,
mostly because the experimental `weighted` DDA variant often stalls at the iteration cap or breaks down
in single precision on the GPU (runs with blank errors have NaN cross sections; see warnings).
