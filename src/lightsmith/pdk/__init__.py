"""Component-to-circuit design flow (an open process design kit).

- :mod:`.technology`: layer stacks as data; :func:`generic_soi` is a generic 220 nm SOI stack built from
  open material data only.
- :mod:`.cells`: parametric components (layout polygons + ports + simulation recipe).
- :mod:`.component`: :class:`Component`, GDS output, device problems at the port reference planes, and
  :class:`Assembly` for hierarchical layouts.
- :mod:`.flow`: simulate a component (dispatching through :func:`lightsmith.solve`), fit a compact model
  of its S-parameters, cache it by parameter hash, and use it in circuit netlists as ``pdk:<cell>``.
"""

from .cells import CELLS, cell, get_component, list_components
from .component import Assembly, Component, CPort, Transform
from .technology import (LayerDef, Technology, generic_soi, get_technology, list_technologies, load_technology,
                         register_technology)

__all__ = [
    "CELLS", "Assembly", "CPort", "Component", "LayerDef", "Technology", "Transform", "cell", "generic_soi",
    "get_component", "get_technology", "list_components", "list_technologies", "load_technology",
    "register_technology",
]
