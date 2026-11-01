"""Minimal clients for Lightsmith Cloud (``/v1/jobs``) and the Lightsmith Materials API.

Standard library only. The API key is sent in the ``Authorization`` header and
appears nowhere else: not in URLs, logs, errors or anything returned to a client.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .config import Secret


class RemoteError(Exception):
    def __init__(self, status: int | None, error: str, message: str, details: Any = None):
        self.status, self.error, self.details = status, error, details
        super().__init__(f"{error}: {message}" + (f" (HTTP {status})" if status else ""))


class _Http:
    def __init__(self, base_url: str, api_key: Secret | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.timeout = timeout

    def request(self, method: str, path: str, body: Any = None, query: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        headers = {"Accept": "application/json", "User-Agent": "lightsmith-mcp"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self._key:
            headers["Authorization"] = "Bearer " + self._key.get()
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read() or b"{}")
            except ValueError:
                err = {}
            raise RemoteError(e.code, str(err.get("error") or e.reason),
                              str(err.get("message") or e.reason), err.get("details") or err.get("detail")) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise RemoteError(None, "Unreachable", f"could not reach {self.base_url} ({reason})") from None
        return json.loads(raw) if raw else None


class CloudClient(_Http):
    """Lightsmith Cloud, or a local ``lightsmith-server`` (see lightsmith-cloud/API.md)."""

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/v1/health")

    def estimate(self, problem: dict, method: str, device: str) -> dict[str, Any]:
        return self.request("POST", "/v1/estimate", {"simulation": problem, "method": method, "device": device})

    def submit(self, operation: str, problem: dict, *, method: str = "auto", methods: list[str] | None = None,
               device: str = "auto", reference: str | None = None, label: str | None = None) -> dict[str, Any]:
        body = {"simulation": problem, "kind": operation, "method": method, "device": device, "label": label}
        if operation == "compare":
            body.update(methods=methods, reference=reference)
        return self.request("POST", "/v1/jobs", body)

    def get(self, job_id: str, include_result: bool = False) -> dict[str, Any]:
        return self.request("GET", f"/v1/jobs/{urllib.parse.quote(job_id)}",
                            query={"include_result": int(include_result), "include_spec": 0})

    def result(self, job_id: str) -> dict[str, Any]:
        return self.request("GET", f"/v1/jobs/{urllib.parse.quote(job_id)}/result")

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/v1/jobs/{urllib.parse.quote(job_id)}")


class MaterialsClient(_Http):
    """Lightsmith Materials API (read endpoints only; see lightsmith-materials/API.md)."""

    def search(self, q: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.request("GET", "/v1/materials", query={"q": q, "limit": limit}).get("items", [])

    def material_json(self, dataset_id: str, model: str = "tabulated", min_nm: float | None = None,
                      max_nm: float | None = None) -> dict[str, Any]:
        path = "/v1/datasets/" + urllib.parse.quote(dataset_id, safe="/") + "/lightsmith"
        return self.request("GET", path, query={"model": model, "min": min_nm, "max": max_nm})
