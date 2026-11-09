"""MCP server: tools, resources, prompts, completions, jobs, cancellation, budget, remote mode and transports.

Most tests talk to the server through the SDK's in-process client; the transport tests launch
``lightsmith-mcp`` over stdio and streamable HTTP like a real MCP client does.
"""

import base64
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402
from mcp_types import PromptReference, ResourceTemplateReference  # noqa: E402

from lightsmith.backends.base import _REGISTRY, Backend, register  # noqa: E402
from lightsmith.mcp_server import ServerConfig, build_server  # noqa: E402
from lightsmith.mcp_server.config import GIB, Secret  # noqa: E402
from lightsmith.mcp_server.jobs import JobManager  # noqa: E402
from lightsmith.mcp_server.kinds import EXAMPLES, kinds  # noqa: E402
from lightsmith.problems import PROBLEM_KINDS, parse_problem  # noqa: E402

SPHERE = EXAMPLES["scattering"]


def run(fn, cfg=None, **client_kw):
    """Run `fn(client)` against a fresh server in-process."""

    async def main():
        async with Client(build_server(cfg), **client_kw) as c:
            return await fn(c)

    return anyio.run(main)


@pytest.fixture
def cfg(tmp_path):
    return ServerConfig(data_dir=tmp_path / "jobs")


def data(r):
    assert not r.is_error, r.content[0].text
    return r.structured_content


# --- discovery -----------------------------------------------------------------------


def test_tool_surface_schemas_and_annotations(cfg):
    async def go(c):
        return {t.name: t for t in (await c.list_tools()).tools}

    tools = run(go, cfg)
    assert {"list_problem_kinds", "get_schema", "list_backends", "check_environment", "validate_problem", "estimate",
            "solve", "start_solve", "compare", "get_job", "list_jobs", "cancel_job", "get_result",
            "search_materials", "get_material", "fit_drude_lorentz", "get_discretization"} <= set(tools)
    prob = tools["solve"].input_schema["properties"]["problem"]
    assert prob["properties"]["kind"]["enum"] == kinds() and set(PROBLEM_KINDS) <= set(kinds())
    assert tools["solve"].input_schema["properties"]["device"]["pattern"]
    assert set(tools["solve"].output_schema["properties"]) >= {"status", "job_id", "summary", "job"}
    for name in ("get_schema", "list_backends", "estimate", "validate_problem", "get_job", "get_discretization"):
        assert tools[name].annotations.read_only_hint is True, name
    assert tools["solve"].annotations.read_only_hint is False
    assert tools["cancel_job"].annotations.destructive_hint is True
    assert all(t.description and len(t.description) > 40 for t in tools.values())


def test_get_discretization_shows_what_a_backend_discretises(cfg):
    async def go(c):
        return data(await c.call_tool("get_discretization", {"problem": EXAMPLES["layer_stack"], "method": "rcwa"}))

    out = run(go, cfg)
    assert out["backend"] == "lightsmith-rcwa" and out["layout_kind"] == "harmonics"
    assert out["summary"]["harmonics"] == 1 and out["error"] is None


def test_every_kind_has_schema_and_valid_example(cfg):
    async def go(c):
        listed = data(await c.call_tool("list_problem_kinds", {}))
        out = {k["kind"]: data(await c.call_tool("get_schema", {"kind": k["kind"]})) for k in listed["kinds"]}
        return listed, out

    listed, schemas = run(go, cfg)
    assert [k["kind"] for k in listed["kinds"]] == kinds()
    for kind, s in schemas.items():
        assert f'"{kind}"' in json.dumps(s["schema"]["properties"]["kind"])
        assert getattr(parse_problem(s["example"]), "kind", "scattering") == kind
        assert s["backends"], kind


def test_new_backends_appear_at_run_time(cfg):
    @register
    class Dummy(Backend):
        name, method, license, problem_kinds = "test-dummy-mcp", "dummy", "MIT", ("layer_stack",)

        @classmethod
        def supports(cls, sim):
            return False, "test backend"

        @classmethod
        def solve(cls, sim, device):
            raise NotImplementedError

    try:
        async def go(c):
            b = data(await c.call_tool("list_backends", {"kind": "layer_stack"}))
            v = data(await c.call_tool("validate_problem", {"problem": EXAMPLES["layer_stack"]}))
            return b, v

        b, v = run(go, cfg)
        assert "test-dummy-mcp" in [r["name"] for r in b["backends"]]
        row = next(r for r in v["backends"] if r.get("name") == "test-dummy-mcp")
        assert row == {"name": "test-dummy-mcp", "method": "dummy", "available": True, "supported": False,
                       "reason": "test backend"}
        assert v["auto_backend"] == "lightsmith-tmm"
    finally:
        _REGISTRY.pop("test-dummy-mcp", None)


def test_validation_errors_name_the_fields_to_fix(cfg):
    bad = {"kind": "scattering", "particles": [{"shape": {"kind": "sphere"}}]}

    async def go(c):
        return (data(await c.call_tool("validate_problem", {"problem": bad})),
                await c.call_tool("solve", {"problem": bad}),
                await c.call_tool("solve", {"problem": {"kind": "plasma"}}))

    v, s, k = run(go, cfg)
    assert v["valid"] is False and "particles.0.shape.sphere.radius_nm" in v["errors"]
    assert s.is_error and "wavelengths: Field required" in s.content[0].text
    assert "get_schema(kind='scattering')" in s.content[0].text
    assert k.is_error and "layer_stack" in k.content[0].text  # lists the valid kinds


def test_environment_never_reveals_the_api_key(tmp_path):
    cfg = ServerConfig(data_dir=tmp_path, cloud_url="http://127.0.0.1:9", api_key=Secret("lsk_abc_supersecret"))
    out = run(lambda c: c.call_tool("check_environment", {}), cfg)
    text = out.content[0].text
    assert "supersecret" not in text and "supersecret" not in repr(cfg)
    srv = out.structured_content["server"]
    assert srv["mode"] == "remote" and srv["api_key_set"] is True
    assert "error" in out.structured_content["remote_health"]  # nothing listens on port 9


# --- solving -----------------------------------------------------------------------


def test_solve_inline_summary_plot_and_resources(cfg):
    async def go(c):
        r = await c.call_tool("solve", {"problem": SPHERE, "plot": True})
        jid = r.structured_content["job_id"]
        reads = {}
        for suffix in ("", "/result", "/result.csv", "/plot.png", "/problem"):
            reads[suffix] = (await c.read_resource(f"lightsmith://jobs/{jid}{suffix}")).contents[0]
        csv_tool = await c.call_tool("get_result", {"job_id": jid, "format": "csv"})
        return r, reads, csv_tool

    r, reads, csv_tool = run(go, cfg)
    out = data(r)
    assert out["status"] == "succeeded" and out["job"]["runner"] == "thread"
    s = out["summary"]
    assert s["method"] == "mie" and s["warnings"] == [] and s["units"]["cross_sections"] == "nm^2"
    assert 500 < s["peak_extinction"]["wavelength_nm"] < 580 and s["peak_extinction"]["q_ext"] > 1
    full = json.loads(reads["/result"].text)
    assert len(full["c_ext_nm2"]) == 81
    assert reads["/result.csv"].text.startswith("wavelength_nm,c_ext_nm2")
    assert reads["/result.csv"].mime_type == "text/csv"
    assert json.loads(reads["/problem"].text)["kind"] == "scattering"
    assert csv_tool.content[0].text == reads["/result.csv"].text
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    img = [x for x in r.content if x.type == "image"]
    assert img and base64.b64decode(img[0].data)[:4] == b"\x89PNG"
    assert base64.b64decode(reads["/plot.png"].blob)[:4] == b"\x89PNG"


def test_worker_reports_library_progress(tmp_path):
    """The solver process turns lightsmith.progress events into job events, for compare and for solve."""
    from lightsmith.mcp_server import _worker

    def events(request):
        d = tmp_path / request["operation"]
        d.mkdir()
        (d / "request.json").write_text(json.dumps(request), encoding="utf-8")
        out = io.StringIO()
        assert _worker.run(str(d), out, time.perf_counter()) == 0, out.getvalue()
        assert json.loads((d / "result.json").read_text(encoding="utf-8"))
        return [json.loads(line) for line in out.getvalue().splitlines()]

    ev = events({"operation": "compare", "problem": EXAMPLES["layer_stack"], "methods": ["tmm", "rcwa"], "device": "cpu"})
    seen = [(e["stage"], e.get("method")) for e in ev if e.get("method")]
    assert ("method_started", "tmm") in seen and ("method_done", "tmm") in seen and ("method_done", "rcwa") in seen
    done = [e["progress"] for e in ev if e.get("stage") == "method_done"]
    assert done == sorted(done) and done[-1] == pytest.approx(0.95)
    assert ev[-1]["type"] == "done"

    sphere = {"kind": "scattering", "medium_index": 1.0, "wavelengths": [500.0, 550.0, 600.0],
              "particles": [{"shape": {"kind": "sphere", "radius_nm": 40}, "material": {"kind": "constant_index", "n_real": 1.5}}]}
    ev = events({"operation": "solve", "problem": sphere, "method": "bem", "device": "cpu"})
    solving = [e for e in ev if e.get("stage") == "solving"]
    assert len(solving) > 2 and solving[0]["backend"] == "lightsmith-bem"  # one per wavelength
    assert [e["progress"] for e in solving] == sorted(e["progress"] for e in solving)

    design = json.loads(json.dumps(EXAMPLES["optimization"]))
    design["optimizer"]["max_iter"] = 2
    ev = events({"operation": "optimize", "problem": design, "method": "auto", "device": "cpu"})
    iterations = [e for e in ev if e.get("stage") == "iteration"]
    assert [e["iteration"] for e in iterations] == [0, 1] and iterations[0]["max_iter"] == 2
    assert isinstance(iterations[0]["objective"], float)


def test_compare_reports_relative_differences(cfg):
    out = data(run(lambda c: c.call_tool("compare", {"problem": EXAMPLES["layer_stack"],
                                                      "methods": ["tmm", "rcwa"], "wait_s": 300}), cfg))
    assert out["status"] == "succeeded"
    s = out["summary"]
    assert s["result_kind"] == "compare" and s["reference"] == "tmm"
    assert max(d["max"] for d in s["relative_difference"]["rcwa"].values()) < 1e-6


def test_budget_refuses_with_estimate_and_hints(tmp_path):
    cfg = ServerConfig(data_dir=tmp_path, max_memory_bytes=0.001 * GIB)
    big = {**SPHERE, "options": {"dda": {"dipoles_per_diameter": 64}}}

    async def go(c):
        return (data(await c.call_tool("estimate", {"problem": big, "method": "dda"})),
                await c.call_tool("solve", {"problem": big, "method": "dda"}),
                data(await c.call_tool("list_jobs", {})))

    est, refused, jobs = run(go, cfg)
    assert est["budget_check"]["ok"] is False and est["estimate"]["dipoles"] > 0
    assert any("dipoles_per_diameter" in h for h in est["budget_check"]["how_to_shrink"])
    text = refused.content[0].text
    assert refused.is_error and "refused before solving" in text and "exceeds the budget" in text
    assert '"memory_bytes"' in text  # the estimate is included
    assert jobs["jobs"] == []  # nothing was started


def test_process_job_progress_result_and_persistence(cfg):
    progress = []

    async def on_progress(p, total, msg):
        progress.append((p, msg))

    async def go(c):
        started = data(await c.call_tool("start_solve", {"problem": EXAMPLES["mode"], "label": "strip"}))
        done = await c.call_tool("get_job", {"job_id": started["job_id"], "wait_s": 300},
                                 progress_callback=on_progress)
        return started, data(done)

    started, done = run(go, cfg)
    assert started["status"] in ("queued", "running") and started["runner"] == "process"
    assert done["status"] == "succeeded", done["job"]["error"]
    modes = done["summary"]["at_wavelengths"][0]["modes"]
    assert 2.3 < modes[0]["neff_real"] < 2.6 and modes[0]["te_fraction"] > 0.9
    assert progress and progress[-1][0] == 1.0
    # A restarted server still knows the finished job and its result.
    jm = JobManager(cfg)
    job = jm.get(started["job_id"])
    assert job.status == "succeeded" and job.label == "strip"
    assert jm.result(job)["kind"] == "modes"


def test_cancel_running_job_kills_the_solver(cfg):
    async def go(c):
        job = data(await c.call_tool("start_solve", {"problem": EXAMPLES["optimization"]}))
        await anyio.sleep(1.0)
        t0 = time.monotonic()
        cancelled = data(await c.call_tool("cancel_job", {"job_id": job["job_id"]}))
        took = time.monotonic() - t0
        again = data(await c.call_tool("cancel_job", {"job_id": job["job_id"]}))  # idempotent
        listed = data(await c.call_tool("list_jobs", {"status": "cancelled"}))
        res = await c.call_tool("get_result", {"job_id": job["job_id"]})
        return cancelled, took, again, listed, res

    cancelled, took, again, listed, res = run(go, cfg)
    assert cancelled["status"] == "cancelled" and cancelled["error"]["error"] == "Cancelled"
    assert took < 10 and again["status"] == "cancelled"
    assert [j["job_id"] for j in listed["jobs"]] == [cancelled["job_id"]]
    assert res.is_error and "cancelled" in res.content[0].text


def test_solve_returns_job_when_wait_runs_out(cfg):
    async def go(c):
        r = data(await c.call_tool("solve", {"problem": EXAMPLES["optimization"], "wait_s": 0.2}))
        await c.call_tool("cancel_job", {"job_id": r["job_id"]})
        return r

    r = run(go, cfg)
    assert r["status"] in ("queued", "running") and "get_job" in r["next_step"]


def test_unknown_job_is_a_clear_error(cfg):
    async def go(c):
        return await c.call_tool("get_job", {"job_id": "nope"})

    r = run(go, cfg)
    assert r.is_error and "unknown job_id" in r.content[0].text and "list_jobs" in r.content[0].text


# --- materials -----------------------------------------------------------------------


def test_materials_search_sample_and_fit(cfg):
    async def go(c):
        return (data(await c.call_tool("search_materials", {"query": "gold"})),
                data(await c.call_tool("get_material", {"material": "Au", "start_nm": 500, "stop_nm": 700, "num": 3})),
                data(await c.call_tool("get_material", {"material": {"kind": "constant_index", "n_real": 1.5}})),
                data(await c.call_tool("fit_drude_lorentz", {"material": "Au", "start_nm": 500, "stop_nm": 800})),
                await c.call_tool("get_material", {"material": "Au", "start_nm": 50, "stop_nm": 60}))

    found, au, glass, fit, out_of_range = run(go, cfg)
    assert any(r["name"] == "Au" and r["material_json"]["kind"] == "library" for r in found["results"])
    assert [s["wavelength_nm"] for s in au["samples"]] == [500, 600, 700]
    assert all(s["k"] > 0 and s["eps_imag"] > 0 for s in au["samples"])  # exp(-i omega t): loss is positive
    assert "Johnson" in au["source"] and au["material_json"]["kind"] == "library"
    assert glass["samples"][0]["n"] == pytest.approx(1.5)
    assert fit["material_json"]["kind"] == "drude_lorentz" and fit["fit"]["rms_rel_error"] < 0.05
    assert out_of_range.is_error and "has data for" in out_of_range.content[0].text


# --- resources, prompts, completions --------------------------------------------------


def test_static_resources(cfg):
    async def go(c):
        listed = [str(r.uri) for r in (await c.list_resources()).resources]
        read = {u: (await c.read_resource(u)).contents[0].text for u in listed}
        kind = (await c.read_resource("lightsmith://schema/mode")).contents[0].text
        ex = (await c.read_resource("lightsmith://examples/periodic")).contents[0].text
        return listed, read, kind, ex

    listed, read, kind, ex = run(go, cfg)
    assert {"lightsmith://conventions", "lightsmith://schema", "lightsmith://examples", "lightsmith://backends",
            "lightsmith://materials", "lightsmith://jobs"} <= set(listed)
    assert "exp(-i omega t)" in read["lightsmith://conventions"]
    matrix = json.loads(read["lightsmith://backends"])
    assert {"mie", "fdtd", "fde", "circuit"} <= {b["method"] for b in matrix["backends"]}
    idx = json.loads(read["lightsmith://examples"])
    assert {e["kind"] for e in idx} >= set(PROBLEM_KINDS)
    assert json.loads(kind)["title"] == "ModeProblem"
    assert parse_problem(json.loads(ex)).kind == "periodic"


def test_prompts_and_completions(cfg):
    async def go(c):
        names = [p.name for p in (await c.list_prompts()).prompts]
        p = await c.get_prompt("setup_simulation", {"kind": "mode", "goal": "single-mode width of a Si strip"})
        comp = await c.complete(PromptReference(type="ref/prompt", name="setup_simulation"),
                                {"name": "kind", "value": "la"})
        comp2 = await c.complete(ResourceTemplateReference(type="ref/resource", uri="lightsmith://schema/{kind}"),
                                 {"name": "kind", "value": ""})
        return names, p, comp, comp2

    names, p, comp, comp2 = run(go, cfg)
    assert {"setup_simulation", "scattering_study", "cross_check_methods", "design_waveguide_component",
            "fit_material_for_fdtd"} <= set(names)
    text = p.messages[0].content.text
    assert "get_schema(kind='mode')" in text and "single-mode width" in text
    assert comp.completion.values == ["layer_stack"]
    assert comp2.completion.values == kinds()


# --- remote mode ---------------------------------------------------------------------


class _FakeCloud(BaseHTTPRequestHandler):
    """Just enough of Lightsmith Cloud's /v1 API; records the Authorization headers it saw."""

    seen_auth: list[str] = []
    jobs: dict[str, dict] = {}

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        self.seen_auth.append(self.headers.get("Authorization", ""))
        if self.headers.get("Authorization") != "Bearer lsk_test_secret":
            self._send(401, {"error": "Unauthorized", "message": "bad key"})
            return False
        return True

    def do_POST(self):
        if not self._auth():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/estimate":
            return self._send(200, {"backend": "miepython", "method": "mie", "framework": "numpy", "device": "cpu",
                                    "memory_bytes": 0, "estimated_runtime_s": 7.0, "credits_estimate": 7})
        if self.path == "/v1/jobs":
            from lightsmith import api

            jid = "f" * 32
            self.jobs[jid] = {"job_id": jid, "status": "succeeded", "progress": 1.0, "stage": "done",
                              "result": api.solve(body["simulation"]).model_dump(mode="json")}
            return self._send(202, {"job_id": jid, "status": "queued", "credits_estimate": 7})
        self._send(404, {"error": "NotFound", "message": self.path})

    def do_GET(self):
        if not self._auth():
            return
        path = self.path.split("?")[0]
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[1] == "jobs" and parts[2] in self.jobs:
            job = self.jobs[parts[2]]
            if path.endswith("/result"):
                return self._send(200, job["result"])
            return self._send(200, {k: v for k, v in job.items() if k != "result"})
        if path == "/v1/health":
            return self._send(200, {"status": "ok", "service": "Lightsmith Cloud"})
        self._send(404, {"error": "NotFound", "message": path})


def test_remote_mode_delegates_to_cloud(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeCloud)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        cfg = ServerConfig(data_dir=tmp_path, cloud_url=url, api_key=Secret("lsk_test_secret"))

        async def go(c):
            est = data(await c.call_tool("estimate", {"problem": SPHERE}))
            sol = data(await c.call_tool("solve", {"problem": SPHERE, "wait_s": 30}))
            env = await c.call_tool("check_environment", {})
            return est, sol, env

        est, sol, env = run(go, cfg)
        assert est["runs_on"].startswith("remote") and est["estimate"]["credits_estimate"] == 7
        assert sol["status"] == "succeeded" and sol["job"]["runner"] == "remote"
        assert sol["job"]["remote_job_id"] == "f" * 32 and sol["summary"]["method"] == "mie"
        assert set(_FakeCloud.seen_auth) == {"Bearer lsk_test_secret"}
        assert "lsk_test_secret" not in env.content[0].text and "lsk_test_secret" not in json.dumps(sol)

        bad = ServerConfig(data_dir=tmp_path / "b", cloud_url=url, api_key=Secret("lsk_wrong"))
        r = run(lambda c: c.call_tool("solve", {"problem": SPHERE}), bad)
        assert r.is_error and "LIGHTSMITH_API_KEY" in r.content[0].text and "lsk_wrong" not in r.content[0].text
    finally:
        srv.shutdown()


# --- transports: a real client over stdio and streamable HTTP -------------------------


def _env(tmp_path):
    env = {**os.environ, "LIGHTSMITH_MCP_DATA_DIR": str(tmp_path)}
    for k in ("LIGHTSMITH_CLOUD_URL", "LIGHTSMITH_API_KEY"):
        env.pop(k, None)
    return env


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_stdio_end_to_end(tmp_path, mode):
    params = StdioServerParameters(command=sys.executable, args=["-m", "lightsmith.mcp_server"], env=_env(tmp_path))

    async def go(c):
        tools = [t.name for t in (await c.list_tools()).tools]
        r = await c.call_tool("solve", {"problem": SPHERE})
        return tools, r

    tools, r = anyio.run(_with_client, params, go, mode)
    assert "solve" in tools
    assert data(r)["summary"]["peak_extinction"]["q_ext"] > 1


async def _with_client(server, fn, mode="auto"):
    async with Client(server, mode=mode, read_timeout_seconds=300) as c:
        return await fn(c)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_streamable_http_end_to_end(tmp_path):
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "lightsmith.mcp_server", "--http", "--port", str(port)],
                            env=_env(tmp_path), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            assert proc.poll() is None, proc.stderr.read().decode()
            time.sleep(0.5)

        async def go(c):
            kinds_ = data(await c.call_tool("list_problem_kinds", {}))
            r = await c.call_tool("solve", {"problem": SPHERE})
            return kinds_, r

        kinds_, r = anyio.run(_with_client, f"http://127.0.0.1:{port}/mcp", go)
        assert [k["kind"] for k in kinds_["kinds"]] == kinds()
        assert data(r)["status"] == "succeeded"
    finally:
        proc.kill()
        proc.wait(10)


def test_http_refuses_public_bind_without_token(tmp_path):
    env = {k: v for k, v in _env(tmp_path).items() if k != "LIGHTSMITH_MCP_HTTP_TOKEN"}
    r = subprocess.run([sys.executable, "-m", "lightsmith.mcp_server", "--http", "--host", "0.0.0.0"],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 2 and "LIGHTSMITH_MCP_HTTP_TOKEN" in r.stderr
