"""Optional progress reporting and cooperative cancellation.

``solve(..., progress=callback)`` installs `callback` for the duration of the solve; backends call
:func:`report` from their loops. The callback receives a dict ``{"stage": str, "fraction": float | None,
**info}`` and may raise :class:`Cancelled` to stop the solve at the next report. `fraction` is None where the
end is not known in advance (an FDTD run stops when its fields have decayed).

Every backend reports ``solve_started`` and ``solve_done``. DDA (per wavelength and incidence), BEM and FDFD (per
wavelength), FDTD (per chunk of steps) and inverse design (per iteration) also report as they go. Third-party
backends report nothing unless they call :func:`report`.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Iterator

ProgressCallback = Callable[[dict[str, Any]], None]

_callback: contextvars.ContextVar[ProgressCallback | None] = contextvars.ContextVar(
    "lightsmith_progress", default=None)


class Cancelled(Exception):
    """Raise from a progress callback to abandon the running solve."""


def report(stage: str, fraction: float | None = None, **info: Any) -> None:
    """Tell the caller how far the current solve is (no-op without a callback)."""
    cb = _callback.get()
    if cb is not None:
        cb({"stage": stage, "fraction": fraction, **info})


@contextmanager
def reporting(callback: ProgressCallback | None) -> Iterator[None]:
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def scaled(callback: ProgressCallback | None, start: float, stop: float, **extra: Any) -> ProgressCallback | None:
    """Map a sub-task's 0..1 onto [start, stop] of the caller's progress."""
    if callback is None:
        return None

    def cb(event: dict[str, Any]) -> None:
        f = event.get("fraction")
        callback({**event, **extra, "fraction": None if f is None else start + (stop - start) * f})

    return cb


__all__ = ["Cancelled", "ProgressCallback", "report", "reporting", "scaled"]
