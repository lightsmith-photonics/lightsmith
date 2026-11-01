"""Solver subprocess for one MCP job.

    python -m lightsmith.mcp_server._worker <job_dir>     # run one job now
    python -m lightsmith.mcp_server._worker --serve       # warm spare: import, then read <job_dir> from stdin

Reads ``<job_dir>/request.json``, writes ``<job_dir>/result.json`` and reports
progress as JSON lines on its stdout. Anything a solver prints goes to stderr,
so it can never corrupt the event stream (or an MCP stdio transport).
Running each job in its own process is what makes cancellation immediate; the
spare mode hides the import time (seconds, with PyTorch) from the next job.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def _events():
    events = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)  # solver prints -> stderr
    sys.stdout = sys.stderr
    return events


def run(job_dir: str, events, t0: float) -> int:
    def emit(**ev) -> None:
        ev.setdefault("elapsed_s", round(time.perf_counter() - t0, 3))
        events.write(json.dumps(ev, default=str) + "\n")

    d = Path(job_dir)
    req = json.loads((d / "request.json").read_text(encoding="utf-8"))
    try:
        emit(type="progress", stage="importing", progress=0.01)
        from lightsmith import api
        from lightsmith.device import resolve_device
        from lightsmith.mcp_server.summaries import json_safe
        from lightsmith.problems import parse_problem

        problem = parse_problem(req["problem"])
        emit(type="progress", stage="started", progress=0.05)
        if req["operation"] == "compare":
            orig = api.solve
            n = max(1, len(req.get("methods") or [1]))
            done = [0]

            def traced(p, m="auto", dev="auto"):  # this process only: report per-method progress
                emit(type="progress", stage="method_started", method=m, progress=0.05 + 0.9 * done[0] / n)
                r = orig(p, m, dev)
                done[0] += 1
                emit(type="progress", stage="method_done", method=m, progress=0.05 + 0.9 * done[0] / n)
                return r

            api.solve = traced
            c = api.compare(problem, req.get("methods"), req.get("device", "auto"), req.get("reference"))
            result = {**c, "results": {k: v.model_dump(mode="json") for k, v in c["results"].items()}}
        else:
            b = api._resolve(problem, req.get("method", "auto"))
            if b.method == "inverse":  # per-iteration progress for inverse design
                from lightsmith.inverse.topopt import optimize

                dev = resolve_device(req.get("device", "auto") if b.gpu else "cpu")
                max_iter = problem.optimizer.max_iter

                def cb(state):
                    emit(type="progress", stage="iteration", iteration=state.iteration, max_iter=max_iter,
                         objective=state.objective, binarization=state.binarization, beta=state.beta,
                         progress=min(0.95, 0.05 + 0.9 * (state.iteration + 1) / max_iter))

                res = optimize(problem, device=dev, callback=cb)
            else:
                emit(type="progress", stage="solving", method=b.method, backend=b.name, progress=0.1)
                res = api.solve(problem, b.name, req.get("device", "auto"))
            result = res.model_dump(mode="json")
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(json_safe(result), allow_nan=False), encoding="utf-8")
        tmp.replace(d / "result.json")
        emit(type="done", progress=1.0)
        return 0
    except Exception as e:
        traceback.print_exc()
        emit(type="error", error=type(e).__name__, message=str(e))
        return 1


def main(argv: list[str]) -> int:
    events = _events()
    if argv and argv[0] == "--serve":
        import lightsmith.api  # noqa: F401  (the slow part: backends, numpy/scipy, torch)
        import lightsmith.mcp_server.summaries  # noqa: F401

        events.write(json.dumps({"type": "ready"}) + "\n")
        line = sys.stdin.readline().strip()
        if not line:  # the server went away
            return 0
        return run(line, events, time.perf_counter())
    return run(argv[0], events, time.perf_counter())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
