"""Server configuration: environment variables, overridable from the command line.

| Variable | Default | Meaning |
|---|---|---|
| ``LIGHTSMITH_MCP_MAX_MEMORY_GB`` | 8 | refuse solves whose estimated memory exceeds this |
| ``LIGHTSMITH_MCP_MAX_RUNTIME_S`` | 1800 | refuse solves whose estimated run time exceeds this |
| ``LIGHTSMITH_MCP_JOB_TIMEOUT_S`` | 2 x max runtime | kill local jobs running longer than this |
| ``LIGHTSMITH_MCP_INLINE_WAIT_S`` | 20 | how long ``solve``/``compare`` wait before handing back a job id |
| ``LIGHTSMITH_MCP_MAX_JOBS`` | 1 | local solves running at once (the rest queue) |
| ``LIGHTSMITH_MCP_DATA_DIR`` | per-user cache | where job results are kept |
| ``LIGHTSMITH_CLOUD_URL`` | unset | delegate solves to Lightsmith Cloud / a Lightsmith server |
| ``LIGHTSMITH_API_KEY`` | unset | API key for ``LIGHTSMITH_CLOUD_URL`` (never logged or echoed) |
| ``LIGHTSMITH_MATERIALS_URL`` | unset | Lightsmith Materials API, searched by ``search_materials`` |
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

GIB = 1024**3


class Secret:
    """A string that never shows up in repr/str/logs."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def get(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._value)


def _float(name: str, default: float | None) -> float | None:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _default_data_dir() -> Path:
    from ..materials import default_cache_dir

    return default_cache_dir() / "mcp-jobs"


def is_local_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


@dataclass
class ServerConfig:
    max_memory_bytes: float = 8 * GIB
    max_runtime_s: float = 1800.0
    job_timeout_s: float | None = None  # None -> 2 * max_runtime_s
    inline_wait_s: float = 20.0
    max_jobs: int = 1
    data_dir: Path | None = None  # None -> keep results in memory only
    cloud_url: str | None = None
    api_key: Secret | None = field(default=None, repr=False)
    materials_url: str | None = None

    @classmethod
    def from_env(cls, **overrides) -> "ServerConfig":
        key = os.environ.get("LIGHTSMITH_API_KEY") or os.environ.get("LIGHTSMITH_CLOUD_API_KEY")
        data_dir = os.environ.get("LIGHTSMITH_MCP_DATA_DIR")
        cfg = cls(
            max_memory_bytes=_float("LIGHTSMITH_MCP_MAX_MEMORY_GB", 8.0) * GIB,
            max_runtime_s=_float("LIGHTSMITH_MCP_MAX_RUNTIME_S", 1800.0),
            job_timeout_s=_float("LIGHTSMITH_MCP_JOB_TIMEOUT_S", None),
            inline_wait_s=_float("LIGHTSMITH_MCP_INLINE_WAIT_S", 20.0),
            max_jobs=int(_float("LIGHTSMITH_MCP_MAX_JOBS", 1)),
            data_dir=Path(data_dir) if data_dir else _default_data_dir(),
            cloud_url=(os.environ.get("LIGHTSMITH_CLOUD_URL") or "").rstrip("/") or None,
            api_key=Secret(key) if key else None,
            materials_url=(os.environ.get("LIGHTSMITH_MATERIALS_URL") or "").rstrip("/") or None,
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    @property
    def remote(self) -> bool:
        """Solves go to LIGHTSMITH_CLOUD_URL: needs an API key, except for a local Lightsmith server."""
        return bool(self.cloud_url) and (bool(self.api_key) or is_local_url(self.cloud_url))

    @property
    def hard_timeout_s(self) -> float:
        return self.job_timeout_s if self.job_timeout_s is not None else 2 * self.max_runtime_s

    def describe(self) -> dict:
        """Safe to show to a client: no key, only whether one is set."""
        return {
            "mode": "remote" if self.remote else "local",
            "cloud_url": self.cloud_url, "api_key_set": bool(self.api_key),
            "remote_disabled_reason": ("LIGHTSMITH_CLOUD_URL is set but LIGHTSMITH_API_KEY is not"
                                       if self.cloud_url and not self.remote else None),
            "materials_url": self.materials_url,
            "budget": {"max_memory_gb": round(self.max_memory_bytes / GIB, 3), "max_runtime_s": self.max_runtime_s,
                       "job_timeout_s": self.hard_timeout_s},
            "inline_wait_s": self.inline_wait_s, "max_concurrent_jobs": self.max_jobs,
            "results_dir": str(self.data_dir) if self.data_dir else None,
        }
