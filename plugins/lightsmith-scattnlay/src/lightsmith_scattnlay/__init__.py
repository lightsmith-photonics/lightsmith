"""scattnlay backend for lightsmith (GPL-3.0-or-later).

Registers the backend ``"scattnlay"`` (method ``"mie"``) for spheres and
layered (core-shell, multilayer) spheres through the ``lightsmith.backends``
entry-point group.
"""

from .backend import ScattnlayBackend, ScattnlayOptions

__version__ = "0.1.0"

__all__ = ["ScattnlayBackend", "ScattnlayOptions"]
