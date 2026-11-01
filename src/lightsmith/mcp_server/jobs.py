"""Solve jobs: queued, run, observed, cancelled.

Three runners:

- ``thread``: quick problems (estimated well under a second or two) run in a
  thread of the server process; cancelling one discards its result.
- ``process``: everything else runs in its own solver subprocess
  (``_worker.py``), so cancelling kills it at once. At most
  ``ServerConfig.max_jobs`` run at the same time; the rest wait in the queue.
- ``remote``: the job runs on Lightsmith Cloud and is polled.

Results are kept as ``<data_dir>/<job_id>/result.json`` next to ``job.json``,
so finished jobs survive a server restart.
"""

from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ServerConfig
from .remote import CloudClient, RemoteError
from .summaries import json_safe

FINISHED = ("succeeded", "failed", "cancelled")
QUICK_S = 1.5  # estimated run time below which a job runs in a thread instead of a subprocess
QUICK_METHODS = ("mie", "tmm", "circuit", "quasistatic")  # cheap enough to trust a rough estimate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    operation: str  # solve | compare
    problem: dict[str, Any]
    problem_kind: str
    method: str = "auto"
    methods: list[str] | None = None
    reference: str | None = None
    device: str = "auto"
    label: str | None = None
    runner: str = "process"
    status: str = "queued"
    progress: float = 0.0
    stage: str | None = None
    message: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)  # last progress event's extra fields
    estimate: dict[str, Any] | None = None
    budget_check: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    remote_id: str | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    _t_start: float | None = field(default=None, repr=False)
    _t_stage: float | None = field(default=None, repr=False)  # when the current stage began
    _result: dict[str, Any] | None = field(default=None, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    def stage_elapsed_s(self) -> float | None:
        return None if self._t_stage is None else time.monotonic() - self._t_stage

    def elapsed_s(self) -> float | None:
        return None if self._t_start is None else round(time.monotonic() - self._t_start, 2)

    def view(self) -> dict[str, Any]:
        uris = {"job": f"lightsmith://jobs/{self.id}"}
        if self.status == "succeeded":
            uris.update(result=f"lightsmith://jobs/{self.id}/result", csv=f"lightsmith://jobs/{self.id}/result.csv",
                        plot=f"lightsmith://jobs/{self.id}/plot.png")
        v = {"job_id": self.id, "operation": self.operation, "problem_kind": self.problem_kind,
             "status": self.status, "progress": round(self.progress, 3), "stage": self.stage,
             "message": self.message, "method": self.method, "methods": self.methods, "device": self.device,
             "label": self.label, "runner": self.runner, "created_at": self.created_at,
             "started_at": self.started_at, "finished_at": self.finished_at,
             "elapsed_s": self.elapsed_s() if not self.finished else self.detail.get("runtime_s"),
             "error": self.error, "resources": uris}
        if self.remote_id:
            v["remote_job_id"] = self.remote_id
        if self.detail and not self.finished:
            v["progress_detail"] = {k: val for k, val in self.detail.items() if k not in ("type", "progress", "stage")}
        if self.budget_check:
            v["estimated_runtime_s"] = self.budget_check.get("estimated_runtime_s")
        return v

    def meta(self) -> dict[str, Any]:
        """What job.json keeps."""
        keys = ("id", "operation", "problem", "problem_kind", "method", "methods", "reference", "device", "label",
                "runner", "status", "progress", "stage", "message", "estimate", "budget_check", "error",
                "remote_id", "created_at", "started_at", "finished_at")
        return {k: getattr(self, k) for k in keys} | {"runtime_s": self.detail.get("runtime_s")}


class JobManager:
    def __init__(self, cfg: ServerConfig, cloud: CloudClient | None = None):
        self.cfg = cfg
        self.cloud = cloud
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max(1, cfg.max_jobs))
        self._spare: subprocess.Popen | None = None
        if cfg.data_dir:
            cfg.data_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # --- bookkeeping -------------------------------------------------------------------

    def job_dir(self, job: Job) -> Path | None:
        return self.cfg.data_dir / job.id if self.cfg.data_dir else None

    def _save(self, job: Job) -> None:
        d = self.job_dir(job)
        if d is None:
            return
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(json_safe(job.meta()), indent=1), encoding="utf-8")
        tmp.replace(d / "job.json")

    def _load(self) -> None:
        for f in sorted(self.cfg.data_dir.glob("*/job.json")):
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            runtime = m.pop("runtime_s", None)
            job = Job(**{k: v for k, v in m.items() if k in Job.__dataclass_fields__})
            job.detail["runtime_s"] = runtime
            if not job.finished:  # the server stopped while it ran
                job.status, job.error = "failed", {"error": "ServerShutdown",
                                                   "message": "the MCP server stopped while this job was running"}
                job.finished_at = job.finished_at or _now()
            job._done.set()
            self.jobs[job.id] = job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job_id {job_id!r}; list_jobs shows the jobs this server knows about")
        return job

    def list(self, status: str | None = None, limit: int = 50) -> list[Job]:
        jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j for j in jobs if status in (None, j.status)][:limit]

    def result(self, job: Job) -> dict[str, Any]:
        if job.status != "succeeded":
            raise ValueError(f"job {job.id} has no result: status is {job.status!r}"
                             + (f" ({job.error['error']}: {job.error['message']})" if job.error else ""))
        if job._result is None:
            d = self.job_dir(job)
            if d is not None and (d / "result.json").is_file():
                job._result = json.loads((d / "result.json").read_text(encoding="utf-8"))
            elif job.runner == "remote" and self.cloud:
                job._result = self.cloud.result(job.remote_id)
            else:
                raise ValueError(f"the result of job {job.id} is no longer available")
        return job._result

    # --- lifecycle ---------------------------------------------------------------------

    def submit(self, operation: str, problem: dict[str, Any], problem_kind: str, *, method: str = "auto",
               methods: list[str] | None = None, reference: str | None = None, device: str = "auto",
               label: str | None = None, estimate: dict | None = None, budget_check: dict | None = None,
               runner: str | None = None) -> Job:
        if runner is None:
            bc = budget_check or {}
            secs = bc.get("estimated_runtime_s")
            quick = (secs is not None and secs < QUICK_S and problem_kind != "optimization"
                     and ((estimate or {}).get("method") in QUICK_METHODS or bc.get("runtime_basis") != "rough"))
            runner = "remote" if self.cloud else ("thread" if quick and operation == "solve" else "process")
        job = Job(id=secrets.token_hex(16), operation=operation, problem=problem, problem_kind=problem_kind,
                  method=method, methods=methods, reference=reference, device=device, label=label, runner=runner,
                  estimate=estimate, budget_check=budget_check)
        with self._lock:
            self.jobs[job.id] = job
        if runner == "remote":
            r = self.cloud.submit(operation, problem, method=method, methods=methods, device=device,
                                  reference=reference, label=label)
            job.remote_id = r.get("job_id") or r.get("id")
            job.detail["credits_estimate"] = r.get("credits_estimate")
        self._save(job)
        target = {"thread": self._run_thread, "process": self._run_process, "remote": self._run_remote}[runner]
        threading.Thread(target=self._guard, args=(job, target), name=f"lightsmith-job-{job.id[:8]}",
                         daemon=True).start()
        return job

    def cancel(self, job_id: str, wait: bool = True) -> Job:
        job = self.get(job_id)
        if job.finished:
            return job
        job._cancel.set()
        if job.runner == "remote" and job.remote_id and self.cloud:
            try:
                self.cloud.cancel(job.remote_id)
            except RemoteError:
                pass
        if job._proc is not None and job._proc.poll() is None:
            job._proc.kill()
        if job.status == "queued" or job.runner == "thread":
            self._finish(job, "cancelled", error={"error": "Cancelled", "message": "the job was cancelled"})
        elif wait:
            job._done.wait(10)
        return job

    def delete(self, job_id: str) -> None:
        job = self.cancel(job_id)
        with self._lock:
            self.jobs.pop(job.id, None)
        d = self.job_dir(job)
        if d is not None and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)

    def _start(self, job: Job) -> None:
        job.status, job.started_at, job._t_start = "running", _now(), time.monotonic()
        job.stage = job.stage or "started"
        self._save(job)

    def _finish(self, job: Job, status: str, error: dict | None = None, result: dict | None = None) -> None:
        with self._lock:
            if job.finished:
                return
            job.status, job.error, job.finished_at = status, error, _now()
            if job._t_start is not None:
                job.detail["runtime_s"] = round(time.monotonic() - job._t_start, 3)
            if status == "succeeded":
                job.progress, job.stage = 1.0, "done"
                if result is not None:
                    job._result = json_safe(result)
                    d = self.job_dir(job)
                    if d is not None:
                        d.mkdir(parents=True, exist_ok=True)
                        (d / "result.json").write_text(json.dumps(job._result), encoding="utf-8")
        self._save(job)
        job._done.set()

    def _guard(self, job: Job, target) -> None:
        try:
            target(job)
        except Exception as e:  # a runner bug must still finish the job
            self._finish(job, "failed", error={"error": type(e).__name__, "message": str(e)})

    def _run_thread(self, job: Job) -> None:
        from .. import api

        self._start(job)
        job.stage, job._t_stage = "solving", time.monotonic()
        try:
            if job.operation == "compare":
                c = api.compare(job.problem, job.methods, job.device, job.reference)
                result = {**c, "results": {k: v.model_dump(mode="json") for k, v in c["results"].items()}}
            else:
                result = api.solve(job.problem, job.method, job.device).model_dump(mode="json")
        except Exception as e:
            if not job._cancel.is_set():
                self._finish(job, "failed", error={"error": type(e).__name__, "message": str(e)})
            return
        if not job._cancel.is_set():
            self._finish(job, "succeeded", result=result)

    def _run_process(self, job: Job) -> None:
        while not self._slots.acquire(timeout=0.5):
            if job._cancel.is_set():
                return
        try:
            if job._cancel.is_set():
                return
            self._process(job)
        finally:
            self._slots.release()

    @staticmethod
    def _spawn() -> subprocess.Popen:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        return subprocess.Popen([sys.executable, "-m", "lightsmith.mcp_server._worker", "--serve"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=flags)

    def warm(self) -> None:
        """Start a spare solver process (imports done, waiting for a job) unless one is ready."""
        if self.cloud is not None:
            return
        with self._lock:
            if self._spare is None or self._spare.poll() is not None:
                self._spare = self._spawn()

    def _take_worker(self) -> subprocess.Popen:
        with self._lock:
            proc, self._spare = self._spare, None
        if proc is None or proc.poll() is not None:
            proc = self._spawn()
        self.warm()  # for the next job
        return proc

    def close(self) -> None:
        """Stop the spare worker and every running job."""
        with self._lock:
            spare, self._spare = self._spare, None
        if spare is not None and spare.poll() is None:
            spare.kill()
        for job in list(self.jobs.values()):
            if not job.finished:
                self.cancel(job.id, wait=False)

    def _process(self, job: Job) -> None:
        d = self.job_dir(job)
        tmp = None
        if d is None:
            import tempfile

            tmp = tempfile.TemporaryDirectory(prefix="lightsmith-job-")
            d = Path(tmp.name)
        d.mkdir(parents=True, exist_ok=True)
        req = {"operation": job.operation, "problem": job.problem, "method": job.method, "methods": job.methods,
               "reference": job.reference, "device": job.device}
        (d / "request.json").write_text(json.dumps(req), encoding="utf-8")
        self._start(job)
        job._proc = proc = self._take_worker()
        proc.stdin.write((str(d) + "\n").encode())
        proc.stdin.flush()
        proc.stdin.close()
        stderr: list[bytes] = []
        threads = [threading.Thread(target=self._read_events, args=(job,), daemon=True),
                   threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True)]
        for t in threads:
            t.start()
        timeout = self.cfg.hard_timeout_s
        while proc.poll() is None:
            if time.monotonic() - job._t_start > timeout:
                proc.kill()
                self._finish(job, "failed", error={
                    "error": "Timeout", "message": f"killed after {timeout:.0f} s (LIGHTSMITH_MCP_JOB_TIMEOUT_S); "
                    "shrink the problem or raise the limit"})
                break
            time.sleep(0.1)
        for t in threads:
            t.join(5)
        (d / "worker.log").write_bytes(b"".join(stderr))
        if job._cancel.is_set():
            self._finish(job, "cancelled", error={"error": "Cancelled", "message": "the job was cancelled"})
        elif (d / "result.json").is_file():
            result = json.loads((d / "result.json").read_text(encoding="utf-8"))
            self._finish(job, "succeeded", result=result)
        elif not job.finished:
            err = job.detail.get("_error") or {"error": "WorkerCrashed", "message": self._tail(d / "worker.log")}
            self._finish(job, "failed", error=err)
        if tmp is not None:
            if job._result is None and job.status == "succeeded":
                job._result = json.loads((d / "result.json").read_text(encoding="utf-8"))
            tmp.cleanup()

    @staticmethod
    def _tail(path: Path, n: int = 1500) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "the solver process died without reporting an error"
        return ("the solver process died: " + text[-n:].strip()) if text.strip() else \
            "the solver process died without reporting an error"

    def _read_events(self, job: Job) -> None:
        for line in job._proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "ready":
                continue
            if ev.get("type") == "error":
                job.detail["_error"] = {"error": ev.get("error"), "message": ev.get("message")}
                continue
            if "progress" in ev:
                job.progress = float(ev["progress"])
            if ev.get("stage", job.stage) != job.stage:
                job.stage, job._t_stage = ev["stage"], time.monotonic()
            job.detail = {**{k: v for k, v in job.detail.items() if k.startswith("_") or k == "runtime_s"},
                          **{k: v for k, v in ev.items() if k not in ("type",)}}
            job.message = _describe(ev)

    def _run_remote(self, job: Job) -> None:
        delay = 0.5
        while True:
            if job._cancel.is_set():  # cancel() already asked the service
                self._finish(job, "cancelled", error={"error": "Cancelled", "message": "the job was cancelled"})
                return
            try:
                r = self.cloud.get(job.remote_id)
            except RemoteError as e:
                if e.status == 404 or e.status == 401:
                    self._finish(job, "failed", error={"error": e.error, "message": str(e)})
                    return
                time.sleep(min(delay, 5))
                continue
            if r.get("status") == "running" and job.status == "queued":
                self._start(job)
            job.progress = float(r.get("progress") or job.progress)
            job.stage, job.message = r.get("stage") or job.stage, r.get("message") or job.message
            if r.get("status") in FINISHED:
                if job._t_start is None:
                    job._t_start = time.monotonic()
                job.detail["usage"] = r.get("usage")
                if r["status"] == "succeeded":
                    self._finish(job, "succeeded", result=self.cloud.result(job.remote_id))
                else:
                    self._finish(job, r["status"], error=r.get("error") or {"error": r["status"], "message": ""})
                return
            time.sleep(min(delay, 3))
            delay *= 1.5


def _describe(ev: dict[str, Any]) -> str | None:
    s = ev.get("stage")
    if s == "iteration":
        return (f"iteration {ev.get('iteration')}/{ev.get('max_iter')}: objective {ev.get('objective'):.4g}, "
                f"binarization {ev.get('binarization'):.2f}")
    if s in ("method_started", "method_done"):
        return f"{ev.get('method')}: {'running' if s == 'method_started' else 'done'}"
    if s == "solving":
        return f"solving with {ev.get('backend')}"
    return s
