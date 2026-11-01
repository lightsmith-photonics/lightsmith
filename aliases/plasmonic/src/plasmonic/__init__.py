"""Alias for :mod:`lightsmith`. Please ``import lightsmith`` instead.

``import plasmonic`` exposes the same top-level API; submodules live under
``lightsmith`` (e.g. ``lightsmith.problems``).
"""

import lightsmith as _lightsmith
from lightsmith import *  # noqa: F401,F403
from lightsmith import __all__, __version__  # noqa: F401


def __getattr__(name):  # forward lazy attributes (e.g. fit_drude_lorentz)
    return getattr(_lightsmith, name)
