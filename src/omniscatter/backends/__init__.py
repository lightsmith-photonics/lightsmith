"""Built-in backends. Every public module in this package is imported so its
``@register`` decorators run; a module that fails to import (e.g. a broken
optional dependency) is skipped with a warning instead of breaking the package.
Backends must import optional third-party libraries lazily (inside methods).
"""

import importlib
import pkgutil
import warnings

from .base import Backend, all_backends, get_backend, register, unsupported_outputs

for _m in pkgutil.iter_modules(__path__):
    if _m.name.startswith("_") or _m.name == "base":
        continue
    try:
        importlib.import_module(f"{__name__}.{_m.name}")
    except Exception as e:  # pragma: no cover
        warnings.warn(f"omniscatter backend module {_m.name!r} failed to import: {e}", stacklevel=2)

__all__ = ["Backend", "all_backends", "get_backend", "register", "unsupported_outputs"]
