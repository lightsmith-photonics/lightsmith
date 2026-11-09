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
        from lightsmith.mcp_server.summaries import json_safe
        from lightsmith.problems import parse_problem

        problem = parse_problem(req["problem"])
        emit(type="progress", stage="started", progress=0.05)
        if req["operation"] == "compare":
            def on_method(ev: dict) -> None:  # lightsmith.progress events, per method
                f = ev.get("fraction")
                emit(type="progress", stage="method_done" if ev["stage"] == "solve_done" else "method_started",
                     method=ev["compare_method"], **({} if f is None else {"progress": 0.05 + 0.9 * f}))

            c = api.compare(problem, req.get("methods"), req.get("device", "auto"), req.get("reference"),
                            progress=on_method)
            result = {**c, "results": {k: v.model_dump(mode="json") for k, v in c["results"].items()}}
        else:
            b = api._resolve(problem, req.get("method", "auto"))
            emit(type="progress", stage="solving", method=b.method, backend=b.name, progress=0.1)

            def on_solve(ev: dict) -> None:  # lightsmith.progress events: inverse design per iteration, others as they go
                f = ev.get("fraction")
                where = {} if f is None else {"progress": min(0.95, 0.1 + 0.85 * f)}
                if ev["stage"] == "iteration":
                    emit(type="progress", stage="iteration", **{k: ev[k] for k in ("iteration", "max_iter", "objective",
                                                                                   "binarization", "beta")}, **where)
                else:
                    emit(type="progress", stage="solving", method=b.method, backend=b.name, **where)

            result = api.solve(problem, b.name, req.get("device", "auto"), progress=on_solve).model_dump(mode="json")
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
