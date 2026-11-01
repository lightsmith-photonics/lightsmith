"""The Lightsmith MCP server: tools, resources, prompts and completions.

No ``from __future__ import annotations`` here: tool signatures carry run-time
JSON Schemas (``WithJsonSchema``) built from the problem registry.
"""

import base64
import json
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import (CallToolResult, Completion, ImageContent, PromptReference, ResourceTemplateReference,
                       TextContent, ToolAnnotations)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, WithJsonSchema

from .. import __version__, api
from ..backends import all_backends
from ..device import doctor
from ..materials import MaterialRangeError
from ..problems import parse_problem, problem_json_schema
from . import budget, kinds, matlib
from .config import ServerConfig
from .jobs import FINISHED, Job, JobManager
from .remote import CloudClient, MaterialsClient, RemoteError
from .summaries import plot_png, result_kind, summarize, to_csv

INSTRUCTIONS = """\
Lightsmith solves electromagnetics / photonics problems (Mie, T-matrix, DDA, FDTD, TMM, RCWA, FDE, EME, FDFD,
PWE, circuit models, inverse design). Everything is JSON. Units: nm, vacuum wavelengths, exp(-i omega t).

Workflow:
1. list_problem_kinds, then get_schema(kind) for the exact JSON Schema and a working example.
2. Materials: search_materials / get_material; fit_drude_lorentz turns measured data into an FDTD-ready model.
3. validate_problem, then estimate (memory, run time, budget verdict). Solves over the server's budget are
   refused with ways to shrink them.
4. solve (or compare to cross-check methods). Quick problems return a summary at once; longer ones become jobs:
   poll get_job(job_id, wait_s=...) or cancel_job. start_solve always returns a job id immediately.
5. Summaries hold the key numbers; full arrays are resources lightsmith://jobs/{job_id}/result (JSON) and
   .../result.csv, a plot at .../plot.png (or get_result / plot=true). Always read the result's warnings.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
READ_ONLY_NET = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                                open_world_hint=True)
RUNS = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
CANCELS = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False)

DEVICE_SCHEMA = {"type": "string", "default": "auto", "pattern": r"^(auto|cpu|gpu|torch-cpu|cuda(:\d+)?)$",
                 "description": "auto (GPU when usable, else CPU) | cpu | gpu | cuda:N | torch-cpu"}


class JobOutput(BaseModel):
    """What solve / compare / get_job return."""

    model_config = ConfigDict(extra="allow")
    status: str = Field(description="queued | running | succeeded | failed | cancelled")
    job_id: str
    summary: dict[str, Any] | None = Field(None, description="Key numbers with units, warnings, provenance "
                                                              "(when succeeded).")
    job: dict[str, Any] = Field(description="Job state, timing and resource URIs of the full result.")
    next_step: str | None = None


def _problem_param_schema() -> dict[str, Any]:
    ks = kinds.kinds()
    return {
        "type": "object", "additionalProperties": True,
        "properties": {"kind": {"type": "string", "enum": ks, "description": "Problem kind; missing = scattering."}},
        "description": ("A Lightsmith problem as JSON. `kind` is one of " + ", ".join(ks) + ". The full schema of "
                        "each kind is get_schema(kind) or the resource lightsmith://schema/{kind}; a working example "
                        "is lightsmith://examples/{kind}. Units nm, vacuum wavelengths."),
    }


def _method_param_schema() -> dict[str, Any]:
    names = sorted({b.method for b in all_backends()} | {b.name for b in all_backends()})
    return {"type": "string", "default": "auto",
            "description": "auto (best available for the kind, see list_backends) or a method / backend name: "
                           + ", ".join(names)}


def _format_validation(e: ValidationError, obj: Any) -> str:
    kind = obj.get("kind", "scattering") if isinstance(obj, dict) else "?"
    lines = []
    for err in e.errors()[:15]:
        loc = list(err["loc"])
        if loc and loc[0] == "problem":
            loc = loc[1:]
        if loc and loc[0] == kind:
            loc = loc[1:]
        where = ".".join(str(p) for p in loc) or "(top level)"
        got = json.dumps(err.get("input"), default=str)
        lines.append(f"- {where}: {err['msg']}" + (f" (got {got[:120]})" if err["type"] != "missing" else ""))
    more = f"\n(and {len(e.errors()) - 15} more)" if len(e.errors()) > 15 else ""
    hint = (f"Fix these fields; the exact schema is get_schema(kind='{kind}') and a valid example is "
            f"lightsmith://examples/{kind}." if kind in kinds.problem_models() else
            f"Set `kind` to one of: {', '.join(kinds.kinds())}.")
    return f"invalid {kind} problem:\n" + "\n".join(lines) + more + "\n" + hint


@contextmanager
def _friendly():
    """Turn library and service errors into tool errors that say what to change."""
    try:
        yield
    except ToolError:
        raise
    except ValidationError as e:
        lines = [f"- {'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}" for err in e.errors()[:15]]
        raise ToolError("invalid input:\n" + "\n".join(lines)) from None
    except MaterialRangeError as e:
        raise ToolError(f"material data out of range: {e}") from None
    except KeyError as e:
        raise ToolError(str(e.args[0]) if e.args else str(e)) from None
    except RemoteError as e:
        msg = str(e)
        if e.details:
            msg += "\ndetails: " + json.dumps(e.details, default=str)[:2000]
        if e.status == 401:
            msg += "\nCheck LIGHTSMITH_API_KEY for LIGHTSMITH_CLOUD_URL."
        raise ToolError(msg) from None
    except (ValueError, RuntimeError, ConnectionError, LookupError, NotImplementedError) as e:
        raise ToolError(f"{type(e).__name__}: {e}") from None


def build_server(config: ServerConfig | None = None) -> MCPServer:
    cfg = config or ServerConfig.from_env()
    cloud = CloudClient(cfg.cloud_url, cfg.api_key) if cfg.remote else None
    mats_api = MaterialsClient(cfg.materials_url) if cfg.materials_url else None
    jobs = JobManager(cfg, cloud)

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield {}
        finally:  # stop the spare worker and running solver processes with the server
            await anyio.to_thread.run_sync(jobs.close)

    mcp = MCPServer("lightsmith", title="Lightsmith", version=__version__, instructions=INSTRUCTIONS,
                    description="Open photonics simulation suite: one JSON problem spec, many solvers.",
                    website_url="https://github.com/lightsmith-photonics/lightsmith", log_level="WARNING",
                    lifespan=lifespan)
    mcp.jobs = jobs  # for tests and embedding

    Problem = Annotated[dict[str, Any], WithJsonSchema(_problem_param_schema())]
    Method = Annotated[str, WithJsonSchema(_method_param_schema())]
    Device = Annotated[str, WithJsonSchema(DEVICE_SCHEMA)]
    WaitS = Annotated[float | None, Field(None, ge=0, le=600, description=(
        f"Seconds to wait for the result before returning the job id (default {cfg.inline_wait_s:g}); "
        "progress notifications are sent meanwhile."))]

    # --- helpers ---------------------------------------------------------------------

    def parse(problem: Any):
        if isinstance(problem, str):
            try:
                problem = json.loads(problem)
            except ValueError as e:
                raise ToolError(f"problem is not valid JSON: {e}") from None
        if not isinstance(problem, dict):
            raise ToolError("problem must be a JSON object; see get_schema(kind)")
        try:
            model = parse_problem(problem)
        except ValidationError as e:
            raise ToolError(_format_validation(e, problem)) from None
        return model, getattr(model, "kind", "scattering"), model.model_dump(mode="json")

    def estimate_one(model, spec: dict, method: str, device: str) -> tuple[dict, dict]:
        if cloud is not None:
            est = cloud.estimate(spec, method, device)
            mem_cfg = ServerConfig(max_memory_bytes=float("inf"), max_runtime_s=cfg.max_runtime_s)
            return est, budget.check(est, model, mem_cfg)
        est = api.estimate(model, method, device)
        return est, budget.check(est, model, cfg)

    def compare_methods(model, kind: str, methods: list[str] | None) -> list[str]:
        if methods:
            return methods
        names = [b.name for b in all_backends()
                 if kind in b.problem_kinds and b.available()[0] and b.supports(model)[0]]
        if not names:
            raise ToolError(f"no available backend supports this {kind} problem; see validate_problem")
        return names

    async def admit(problem, method: str, device: str, operation: str = "solve",
                    methods: list[str] | None = None) -> tuple[Any, str, dict, dict, dict, list[str] | None]:
        model, kind, spec = parse(problem)
        with _friendly():
            if operation == "compare":
                methods = await anyio.to_thread.run_sync(compare_methods, model, kind, methods)
                per = {}
                for m in methods:
                    per[m] = await anyio.to_thread.run_sync(estimate_one, model, spec, m, device)
                checks = [c for _, c in per.values()]
                est = {m: e for m, (e, _) in per.items()}
                chk = {"ok": all(c["ok"] for c in checks),
                       "estimated_runtime_s": round(sum(c["estimated_runtime_s"] for c in checks), 3),
                       "per_method": {m: c for m, (_, c) in per.items()}}
                if not chk["ok"]:
                    chk["reasons"] = [f"{m}: {r}" for m, (_, c) in per.items() for r in c.get("reasons", [])]
                    chk["how_to_shrink"] = sorted({h for c in checks for h in c.get("how_to_shrink", [])}) + [
                        "or drop the expensive methods from `methods`"]
            else:
                est, chk = await anyio.to_thread.run_sync(estimate_one, model, spec, method, device)
        if not chk["ok"]:
            raise ToolError(budget.OverBudget({"estimate": est, "budget_check": chk}).args[0] + "\n" +
                            json.dumps({"estimate": est, "budget_check": chk}, default=str, indent=1)[:6000])
        return model, kind, spec, est, chk, methods

    def progress_of(job: Job) -> float:
        """Reported progress; while a single solve runs, extrapolated from its estimated run time."""
        if job.finished:
            return 1.0 if job.status == "succeeded" else job.progress
        p = job.progress
        est = (job.budget_check or {}).get("estimated_runtime_s")
        if job.stage == "solving" and est and job.stage_elapsed_s() is not None:
            p = max(p, min(0.9, 0.1 + 0.8 * job.stage_elapsed_s() / max(est, 1e-3)))
        return min(p, 0.99)

    async def wait(job: Job, wait_s: float, ctx: Context | None) -> None:
        deadline = time.monotonic() + wait_s
        last = None
        step = 0.05
        try:
            while not job.finished and time.monotonic() < deadline:
                await anyio.sleep(min(step, max(0.0, deadline - time.monotonic())))
                step = min(step * 1.5, 1.0)
                key = (round(progress_of(job), 2), job.message, job.status)
                if ctx is not None and key != last:
                    last = key
                    msg = job.message or job.stage or job.status
                    await ctx.report_progress(progress_of(job), 1.0, f"{job.status}: {msg}")
        except anyio.get_cancelled_exc_class():
            # The client cancelled the request: stop the work too.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(lambda: jobs.cancel(job.id, wait=False))
            raise

    def job_output(job: Job, plot: bool = False) -> CallToolResult:
        view = job.view()
        view["progress"] = round(progress_of(job), 3)
        out: dict[str, Any] = {"status": job.status, "job_id": job.id, "job": view}
        content: list = []
        if job.status == "succeeded":
            with _friendly():
                result = jobs.result(job)
            out["summary"] = summarize(result)
            out["next_step"] = ("Full arrays: read " + view["resources"]["result"] + " (JSON) or "
                                + view["resources"]["csv"] + ", or call get_result. Check summary.warnings.")
            if plot:
                png = plot_png(result, title=f"{job.problem_kind} · {out['summary'].get('backend') or job.operation}")
                if png:
                    content.append(ImageContent(type="image", data=base64.b64encode(png).decode(),
                                                mime_type="image/png"))
                else:
                    out["plot_note"] = "no plot: matplotlib is not installed or this result kind has no plot"
        elif job.status in ("queued", "running"):
            out["next_step"] = (f"Still {job.status}. Call get_job(job_id='{job.id}', wait_s=30) to wait for it, "
                                "or cancel_job to stop it.")
        else:
            out["next_step"] = "The job did not succeed; see job.error, fix the problem and solve again."
        text = json.dumps(out, default=str)
        return CallToolResult(content=[TextContent(type="text", text=text), *content], structured_content=out,
                              is_error=job.status == "failed")

    def submit(operation: str, model, kind: str, spec: dict, est, chk, method="auto", methods=None,
               reference=None, device="auto", label=None) -> Job:
        with _friendly():
            return jobs.submit(operation, spec, kind, method=method, methods=methods, reference=reference,
                               device=device, label=label, estimate=est, budget_check=chk)

    # --- discovery ---------------------------------------------------------------------

    @mcp.tool(title="List problem kinds", annotations=READ_ONLY)
    def list_problem_kinds() -> dict[str, Any]:
        """Every problem kind this server can solve (scattering, layer_stack, periodic, mode, device, bands,
        circuit, optimization, plus any added by plugins), with the methods available for each, the
        auto-selection order and where to find its schema and example. Start here."""
        return {"kinds": [kinds.kind_summary(k) for k in kinds.kinds()], "conventions": kinds.CONVENTIONS}

    @mcp.tool(title="Problem JSON Schema", annotations=READ_ONLY)
    def get_schema(kind: Annotated[str | None, Field(description="Problem kind; omit for the schema of all kinds "
                                                     "(large).")] = None,
                   include_example: bool = True) -> dict[str, Any]:
        """The exact JSON Schema of one problem kind, a small working example, the backends that solve it and
        the conventions. Build problems from this; validate_problem checks one without solving."""
        if kind is None:
            return {"kinds": kinds.kinds(), "schema": problem_json_schema(), "conventions": kinds.CONVENTIONS}
        with _friendly():
            kinds.check_kind(kind)
        out = {"kind": kind, "schema": kinds.kind_schema(kind), "conventions": kinds.CONVENTIONS,
               "backends": [{k: b[k] for k in ("name", "method", "available", "gpu", "auto_rank")}
                            for b in kinds.backends_for(kind)]}
        if include_example and kind in kinds.EXAMPLES:
            out["example"] = kinds.EXAMPLES[kind]
        return out

    @mcp.tool(title="List solver backends", annotations=READ_ONLY)
    def list_backends(kind: Annotated[str | None, Field(description="Only backends for this problem kind.")] = None
                      ) -> dict[str, Any]:
        """Solver backends: method, problem kinds, licence, GPU support, availability (and why not), citation,
        auto-selection rank and options key. This is the backend capability matrix."""
        m = kinds.capability_matrix()
        if kind is not None:
            with _friendly():
                kinds.check_kind(kind)
            m["backends"] = [b for b in m["backends"] if kind in b["problem_kinds"]]
        if cloud is not None:
            m["note"] = "remote mode: solves run on " + cfg.cloud_url + "; availability shown is this machine's"
        return m

    @mcp.tool(title="Check environment", annotations=READ_ONLY)
    def check_environment() -> dict[str, Any]:
        """CPU/GPU frameworks and devices visible to the solvers, plus this server's mode (local or remote),
        budget, job slots and results directory. The API key is never shown, only whether one is set."""
        out = {"lightsmith_version": __version__, "devices": doctor(), "server": cfg.describe()}
        if cloud is not None:
            try:
                out["remote_health"] = cloud.health()
            except RemoteError as e:
                out["remote_health"] = {"error": str(e)}
        return out

    @mcp.tool(title="Validate a problem", annotations=READ_ONLY)
    def validate_problem(problem: Problem) -> dict[str, Any]:
        """Check a problem without solving: schema errors (with the field paths to fix), the normalised problem
        with defaults filled in, and which backends can solve it (and why the others cannot)."""
        try:
            model, kind, spec = parse(problem)
        except ToolError as e:
            return {"valid": False, "errors": str(e)}
        rows = []
        for b in all_backends():
            if kind not in b.problem_kinds:
                continue
            ok, why = b.available()
            sup, swhy = b.supports(model) if ok else (False, "")
            rows.append({"name": b.name, "method": b.method, "available": ok, "supported": sup,
                         "reason": (why or swhy) or None})
        try:
            auto = api.choose_method(model)[0].name
        except ValueError as e:
            auto = None
            rows.append({"auto": str(e)})
        return {"valid": True, "kind": kind, "auto_backend": auto, "backends": rows, "problem": spec}

    # --- estimate and solve ----------------------------------------------------------

    @mcp.tool(title="Estimate cost", annotations=READ_ONLY)
    async def estimate(problem: Problem, method: Method = "auto", device: Device = "auto") -> dict[str, Any]:
        """Pre-flight: which backend would run, on what device, problem size, memory, estimated run time and the
        budget verdict (ok, or why it would be refused and how to shrink it). Never solves. Call it before
        solving anything grid-based (DDA, FDTD, FDFD, RCWA, PWE, FDE, EME, inverse design)."""
        model, kind, spec = parse(problem)
        jobs.warm()  # a solve usually follows: have a solver process ready
        with _friendly():
            est, chk = await anyio.to_thread.run_sync(estimate_one, model, spec, method, device)
        return {"kind": kind, "estimate": est, "budget_check": chk,
                "runs_on": "remote: " + cfg.cloud_url if cloud else "this machine"}

    @mcp.tool(title="Solve", annotations=RUNS)
    async def solve(problem: Problem, method: Method = "auto", device: Device = "auto", wait_s: WaitS = None,
                    plot: Annotated[bool, Field(description="Attach a PNG plot of the result.")] = False,
                    label: str | None = None, ctx: Context = None) -> Annotated[CallToolResult, JobOutput]:
        """Solve a problem. It is estimated first and refused if over the server's memory/time budget (with
        the estimate and how to shrink it). Waits up to wait_s for the result, then returns a compact summary
        (key numbers, units, warnings, provenance) plus resource URIs of the full arrays; a longer solve keeps
        running as a job (poll get_job, stop with cancel_job)."""
        model, kind, spec, est, chk, _ = await admit(problem, method, device)
        job = submit("solve", model, kind, spec, est, chk, method=method, device=device, label=label)
        await wait(job, cfg.inline_wait_s if wait_s is None else wait_s, ctx)
        return job_output(job, plot)

    @mcp.tool(title="Start a solve job", annotations=RUNS)
    async def start_solve(problem: Problem, method: Method = "auto", device: Device = "auto",
                          label: str | None = None) -> dict[str, Any]:
        """Start a solve in the background and return its job id at once (after the same budget check as
        solve). Follow it with get_job(job_id, wait_s=...), stop it with cancel_job."""
        model, kind, spec, est, chk, _ = await admit(problem, method, device)
        job = submit("solve", model, kind, spec, est, chk, method=method, device=device, label=label)
        return job.view()

    @mcp.tool(title="Cross-check methods", annotations=RUNS)
    async def compare(problem: Problem,
                      methods: Annotated[list[str] | None, Field(description="Methods or backend names; default: "
                                                                 "every available backend for this problem.")] = None,
                      reference: Annotated[str | None, Field(description="Reference method; default Mie when it "
                                                             "applies, else the first.")] = None,
                      device: Device = "auto", wait_s: WaitS = None, plot: bool = False,
                      ctx: Context = None) -> Annotated[CallToolResult, JobOutput]:
        """Solve one problem with several methods and report their relative differences against a reference.
        The best way to check a result you are unsure of. Budget-checked per method; runs as a job."""
        model, kind, spec, est, chk, methods = await admit(problem, "auto", device, "compare", methods)
        job = submit("compare", model, kind, spec, est, chk, methods=methods, reference=reference, device=device)
        await wait(job, cfg.inline_wait_s if wait_s is None else wait_s, ctx)
        return job_output(job, plot)

    # --- jobs ------------------------------------------------------------------------

    @mcp.tool(title="Get a job", annotations=READ_ONLY)
    async def get_job(job_id: str,
                      wait_s: Annotated[float, Field(ge=0, le=600, description="Wait up to this long for the job to "
                                                     "finish, with progress notifications.")] = 0,
                      plot: bool = False, ctx: Context = None) -> Annotated[CallToolResult, JobOutput]:
        """Status, progress and (when finished) the result summary of a job."""
        with _friendly():
            job = jobs.get(job_id)
        if wait_s:
            await wait(job, wait_s, ctx)
        return job_output(job, plot)

    @mcp.tool(title="List jobs", annotations=READ_ONLY)
    def list_jobs(status: Literal["queued", "running", "succeeded", "failed", "cancelled"] | None = None,
                  limit: Annotated[int, Field(ge=1, le=500)] = 50) -> dict[str, Any]:
        """Jobs this server knows about, newest first (finished ones survive restarts)."""
        return {"jobs": [j.view() for j in jobs.list(status, limit)]}

    @mcp.tool(title="Cancel a job", annotations=CANCELS)
    async def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel a queued or running job (a running solver process is killed). Finished jobs are unchanged."""
        with _friendly():
            job = await anyio.to_thread.run_sync(jobs.cancel, job_id)
        return job.view()

    @mcp.tool(title="Get full result", annotations=READ_ONLY)
    def get_result(job_id: str,
                   format: Annotated[Literal["json", "csv", "summary"], Field(
                       description="json = the complete result; csv = main spectra as a table.")] = "json",
                   max_chars: Annotated[int, Field(ge=1000, le=5_000_000)] = 200_000) -> CallToolResult:
        """The full result of a finished job, for clients that cannot read resources. Prefer the resources
        lightsmith://jobs/{job_id}/result and .../result.csv; large results are truncated to max_chars."""
        with _friendly():
            result = jobs.result(jobs.get(job_id))
        if format == "summary":
            text = json.dumps(summarize(result), default=str)
        elif format == "csv":
            text = to_csv(result) or "no CSV table for result kind " + result_kind(result)
        else:
            text = json.dumps(result)
        note = None
        if len(text) > max_chars:
            note = (f"[truncated: {len(text)} characters, showing {max_chars}; read lightsmith://jobs/{job_id}/"
                    f"result{'.csv' if format == 'csv' else ''} for all of it]")
            text = text[:max_chars]
        return CallToolResult(content=[TextContent(type="text", text=text)] +
                              ([TextContent(type="text", text=note)] if note else []))

    # --- materials -------------------------------------------------------------------

    MaterialArg = Annotated[str | dict[str, Any], Field(description=matlib.MATERIAL_HELP)]

    @mcp.tool(title="Search materials", annotations=READ_ONLY_NET)
    def search_materials(query: Annotated[str, Field(description="e.g. 'gold', 'Si', 'TiO2', 'LiNbO3'")],
                         sources: Annotated[list[Literal["bundled", "refractiveindex", "materials_api"]] | None, Field(
                             description="Default: bundled, plus the Materials API when configured. "
                             "'refractiveindex' searches the refractiveindex.info catalogue (downloaded once).")
                         ] = None, limit: Annotated[int, Field(ge=1, le=200)] = 25) -> dict[str, Any]:
        """Find optical-constant data. Each hit carries a `material` / `material_json` value to pass to
        get_material or to put straight into a problem."""
        sources = sources or (["bundled"] + (["materials_api"] if mats_api else []))
        out: dict[str, Any] = {"query": query, "results": [], "errors": {}}
        for s in sources:
            try:
                if s == "bundled":
                    out["results"] += matlib.search_bundled(query)[:limit]
                elif s == "refractiveindex":
                    out["results"] += matlib.search_refractiveindex(query, limit)
                elif s == "materials_api":
                    if mats_api is None:
                        raise ValueError("no Materials API configured (LIGHTSMITH_MATERIALS_URL)")
                    for m in mats_api.search(query, limit):
                        for d in m.get("datasets") or []:
                            out["results"].append({"source": "materials_api", "name": m.get("name"),
                                                   "dataset": d.get("name"), "range_nm": d.get("range_nm"),
                                                   "has_k": d.get("has_k"), "reference": d.get("reference"),
                                                   "material": {"materials_api": d.get("slug") or d.get("id")}})
            except (ValueError, ConnectionError, RemoteError) as e:
                out["errors"][s] = str(e)
        if not out["errors"]:
            del out["errors"]
        if not out["results"] and "refractiveindex" not in sources:
            out["hint"] = "nothing bundled matches; try sources=['refractiveindex']"
        return out

    @mcp.tool(title="Get material data", annotations=READ_ONLY_NET)
    def get_material(material: MaterialArg,
                     wavelengths_nm: Annotated[list[float] | None, Field(description="Explicit wavelengths.")] = None,
                     start_nm: float | None = None, stop_nm: float | None = None,
                     num: Annotated[int, Field(ge=1, le=500)] = 11) -> dict[str, Any]:
        """n, k and permittivity of a material at some wavelengths, its valid range and citation, and the
        material JSON to put into a problem (exp(-i omega t): k > 0 is loss)."""
        with _friendly():
            return matlib.sample(material, wavelengths_nm, start_nm, stop_nm, num, mats_api)

    @mcp.tool(title="Fit Drude-Lorentz model", annotations=READ_ONLY_NET)
    def fit_drude_lorentz(material: MaterialArg, start_nm: float, stop_nm: float,
                          n_poles: Annotated[int | None, Field(ge=0, le=8, description="Lorentz poles; default: "
                                                               "fewest reaching 2% error.")] = None,
                          drude: Annotated[bool | None, Field(description="Add a Drude term; default: when "
                                                              "Re(eps) < 0 in the band.")] = None) -> dict[str, Any]:
        """Fit a passive Drude-Lorentz model to measured data over [start_nm, stop_nm]: FDTD needs one. Returns
        the model JSON (drop it into the problem) and the fit errors."""
        with _friendly():
            return matlib.fit(material, start_nm, stop_nm, n_poles, drude, mats_api)

    # --- resources ---------------------------------------------------------------------

    @mcp.resource("lightsmith://conventions", name="conventions", title="Units and conventions",
                  mime_type="text/markdown")
    def conventions_resource() -> str:
        agents = Path(__file__).resolve().parents[3] / "AGENTS.md"
        text = "# Lightsmith conventions\n\n" + kinds.CONVENTIONS
        if agents.is_file():
            text += "\n\n---\n\n" + agents.read_text(encoding="utf-8")
        return text

    @mcp.resource("lightsmith://schema", name="schema", title="JSON Schema of every problem kind",
                  mime_type="application/json")
    def schema_all() -> str:
        return json.dumps(problem_json_schema())

    @mcp.resource("lightsmith://schema/{kind}", name="schema-kind", title="JSON Schema of one problem kind",
                  mime_type="application/json")
    def schema_kind(kind: str) -> str:
        kinds.check_kind(kind)
        return json.dumps(kinds.kind_schema(kind))

    @mcp.resource("lightsmith://examples", name="examples", title="Example problems",
                  mime_type="application/json")
    def examples_index() -> str:
        idx = [{"name": k, "kind": k, "uri": f"lightsmith://examples/{k}", "source": "built-in"}
               for k in kinds.EXAMPLES]
        idx += [{"name": n, "kind": kinds.example(n).get("kind", "scattering"), "uri": f"lightsmith://examples/{n}",
                 "source": f"examples/{p.name}"} for n, p in kinds.repo_examples().items() if n not in kinds.EXAMPLES]
        return json.dumps(idx)

    @mcp.resource("lightsmith://examples/{name}", name="example", title="An example problem",
                  mime_type="application/json")
    def example_resource(name: str) -> str:
        return json.dumps(kinds.example(name), indent=1)

    @mcp.resource("lightsmith://backends", name="backends", title="Backend capability matrix",
                  mime_type="application/json")
    def backends_resource() -> str:
        return json.dumps(kinds.capability_matrix(), default=str)

    @mcp.resource("lightsmith://materials", name="materials", title="Bundled material datasets",
                  mime_type="application/json")
    def materials_resource() -> str:
        return json.dumps(matlib.search_bundled(""))

    @mcp.resource("lightsmith://jobs", name="jobs", title="Jobs", mime_type="application/json")
    def jobs_resource() -> str:
        return json.dumps([j.view() for j in jobs.list()], default=str)

    @mcp.resource("lightsmith://jobs/{job_id}", name="job", title="Job state", mime_type="application/json")
    def job_resource(job_id: str) -> str:
        return json.dumps(jobs.get(job_id).view(), default=str)

    @mcp.resource("lightsmith://jobs/{job_id}/problem", name="job-problem", title="The problem a job solved",
                  mime_type="application/json")
    def job_problem(job_id: str) -> str:
        return json.dumps(jobs.get(job_id).problem)

    @mcp.resource("lightsmith://jobs/{job_id}/result", name="job-result", title="Full result (JSON)",
                  mime_type="application/json")
    def job_result(job_id: str) -> str:
        return json.dumps(jobs.result(jobs.get(job_id)))

    @mcp.resource("lightsmith://jobs/{job_id}/result.csv", name="job-result-csv", title="Main spectra (CSV)",
                  mime_type="text/csv")
    def job_result_csv(job_id: str) -> str:
        result = jobs.result(jobs.get(job_id))
        csv_text = to_csv(result)
        if csv_text is None:
            raise ValueError(f"no CSV table for result kind {result_kind(result)!r}; read the JSON result")
        return csv_text

    @mcp.resource("lightsmith://jobs/{job_id}/plot.png", name="job-plot", title="Plot of the result",
                  mime_type="image/png")
    def job_plot(job_id: str) -> bytes:
        job = jobs.get(job_id)
        png = plot_png(jobs.result(job), title=f"{job.problem_kind} {job.id[:8]}")
        if png is None:
            raise ValueError("no plot: matplotlib is not installed or this result kind has no plot")
        return png

    # --- prompts -----------------------------------------------------------------------

    @mcp.prompt(title="Set up a simulation")
    def setup_simulation(kind: str, goal: str) -> str:
        """Guided set-up of any problem kind, from goal to validated, budget-checked solve."""
        return (f"Set up a Lightsmith `{kind}` simulation for this goal: {goal}\n\n"
                f"1. Call get_schema(kind='{kind}') and start from its example.\n"
                "2. Choose materials with search_materials / get_material; check the data covers the wavelengths. "
                "For FDTD use fit_drude_lorentz.\n"
                "3. Units are nm and vacuum wavelengths; time convention exp(-i omega t).\n"
                "4. validate_problem, then estimate. If the budget check refuses it, apply its how_to_shrink hints.\n"
                "5. solve (plot=true for a figure). Read summary.warnings and provenance before reporting numbers; "
                "state the method, backend and any convergence caveats.\n"
                "6. If accuracy matters, cross-check with compare or a convergence study (refine the grid once).")

    @mcp.prompt(title="Scattering by a nanoparticle")
    def scattering_study(particle: str, material: str, wavelength_range_nm: str = "400-900",
                         medium: str = "water (n = 1.33)") -> str:
        """Optical cross sections of a nanoparticle (Mie for spheres, T-matrix/DDA otherwise)."""
        return (f"Compute extinction, scattering and absorption spectra of {particle} made of {material} in "
                f"{medium} over {wavelength_range_nm} nm with Lightsmith.\n"
                "- get_schema(kind='scattering') for the format; search_materials for the material (prefer bundled "
                "measured data; check range_nm covers the band).\n"
                "- Spheres and layered spheres: method auto picks Mie (exact). Other shapes: T-matrix or DDA; call "
                "estimate first.\n"
                "- solve with plot=true; report the peak wavelength, C_ext/Q_ext at the peak and the albedo.\n"
                "- Non-spherical particles: compare DDA against T-matrix (or a sphere of equal volume against Mie).")

    @mcp.prompt(title="Cross-check methods")
    def cross_check_methods(problem_description: str) -> str:
        """Verify a result by solving the same problem with independent methods."""
        return (f"Cross-check this simulation with independent methods: {problem_description}\n"
                "1. Build the problem (get_schema) and validate_problem: it lists every backend that supports it.\n"
                "2. estimate each candidate method; drop any the budget refuses or that would take too long.\n"
                "3. compare(problem, methods=[...]) with the most exact method as reference (Mie for spheres, "
                "TMM for planar stacks, RCWA for gratings, FDE/EME for waveguides).\n"
                "4. Report relative differences, whether they are within each method's expected discretisation "
                "error, and which result to trust. If they disagree, refine the grid of the approximate method "
                "once and compare again.")

    @mcp.prompt(title="Design a waveguide component")
    def design_waveguide_component(component: str, platform: str = "SOI, 220 nm Si on SiO2",
                                   wavelength_nm: str = "1550") -> str:
        """From waveguide modes to device S-parameters, a circuit, and optionally inverse design."""
        return (f"Design a {component} on {platform} at {wavelength_nm} nm with Lightsmith.\n"
                "1. Waveguide: solve a `mode` problem (FDE) for the strip/rib cross-section; pick a single-mode "
                "width; note neff, ng and loss.\n"
                "2. Device: build a `device` problem (EME for tapers/MMIs/couplers, FDFD for compact 2D devices, "
                "FDTD for full 3D); estimate first, then solve for S-parameters over the band.\n"
                "3. Circuit: if the component is part of a larger circuit, build a `circuit` problem from compact "
                "models or the device's S-parameters.\n"
                "4. Optional: an `optimization` problem (inverse design) for a compact footprint; start_solve it "
                "and follow get_job.\n"
                "5. Report insertion loss, reflection, bandwidth and footprint, with the methods and warnings.")

    @mcp.prompt(title="Prepare a material for FDTD")
    def fit_material_for_fdtd(material: str, band_nm: str = "400-900") -> str:
        """Turn measured optical constants into a passive Drude-Lorentz model for FDTD."""
        return (f"Prepare {material} for an FDTD simulation over {band_nm} nm.\n"
                "1. search_materials to find measured data covering the band (get_material to inspect n, k).\n"
                "2. fit_drude_lorentz over the band; accept it if rms_rel_error and rms_imag_error are below ~2%, "
                "otherwise allow more poles or split the band.\n"
                "3. Use the returned material_json in the problem and mention the fit error in the report.")

    # --- completions -------------------------------------------------------------------

    @mcp.completion()
    async def complete(ref, argument, context):
        prefix = (argument.value or "").lower()
        values: list[str] = []
        if argument.name == "kind":
            values = kinds.kinds()
        elif argument.name == "name" and isinstance(ref, ResourceTemplateReference):
            values = [*kinds.EXAMPLES, *kinds.repo_examples()]
        elif argument.name == "job_id":
            values = [j.id for j in jobs.list(limit=100)]
        elif argument.name == "material" and isinstance(ref, PromptReference):
            values = sorted({r["name"] for r in matlib.search_bundled("")})
        hits = [v for v in values if v.lower().startswith(prefix)]
        return Completion(values=hits[:100], total=len(hits), has_more=len(hits) > 100)

    return mcp


__all__ = ["FINISHED", "JobOutput", "build_server"]
