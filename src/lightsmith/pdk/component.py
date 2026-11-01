"""Components: layout polygons + ports, and the device problems simulated from them.

A :class:`Component` holds polygons per technology layer name (in nm), named
optical ports, and the parameters that built it. Ports follow the gdsfactory
convention: ``orientation_deg`` points *out of* the component, and names are
``o1, o2, ...``. A port's centre is the S-parameter reference plane: device
simulations put their port planes exactly there, so cascading components in a
circuit adds no spurious phase.

:class:`Assembly` places components by connecting ports and yields a
hierarchical component (GDS with references, a flattened device for direct
simulation, and the circuit netlist of its parts).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..problems import DeviceProblem, Polygon, Port, Prism, Structure, Wavelengths, wavelength_array
from .technology import Technology, get_technology


@dataclass(frozen=True)
class CPort:
    """An optical port of a component: reference plane centre, outward orientation, waveguide width."""

    name: str
    center: tuple[float, float]
    orientation_deg: float
    width_nm: float
    layer: str = "WG"

    @property
    def normal_axis(self) -> int:
        return 0 if round(self.orientation_deg / 90) % 2 == 0 else 1

    @property
    def outward(self) -> np.ndarray:
        a = math.radians(self.orientation_deg)
        return np.array([round(math.cos(a), 12), round(math.sin(a), 12)])

    def transformed(self, t: "Transform", name: str | None = None) -> "CPort":
        o = self.orientation_deg
        if t.mirror:
            o = -o
        return CPort(name or self.name, tuple(t.apply(np.array([self.center]))[0].tolist()),
                     (o + t.rotation_deg) % 360, self.width_nm, self.layer)


@dataclass(frozen=True)
class Transform:
    """Mirror about the x axis (optional), then rotate about the origin, then translate (GDS order)."""

    origin: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0
    mirror: bool = False

    def apply(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=float).copy()
        if self.mirror:
            p[:, 1] = -p[:, 1]
        a = math.radians(self.rotation_deg)
        c, s = round(math.cos(a), 15), round(math.sin(a), 15)
        p = p @ np.array([[c, s], [-s, c]])
        return p + np.asarray(self.origin, dtype=float)


@dataclass
class Instance:
    name: str
    component: "Component"
    transform: Transform


@dataclass
class Component:
    """Layout + ports + parameters of one cell.

    ``kind`` is the registry name of the parametric cell (``mmi1x2``, ...) and
    ``settings`` its validated parameters; together they identify the circuit
    model ``pdk:<kind>``. Composite components (from :class:`Assembly`) also
    carry ``instances`` and a ``netlist`` of their parts.
    """

    kind: str
    settings: dict[str, Any]
    polygons: dict[str, list[np.ndarray]] = field(default_factory=dict)
    ports: list[CPort] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)
    netlist: dict[str, Any] | None = None
    model: str = "simulate"  # "simulate" (device solve) | "waveguide" (mode solve) | "netlist" (composite)
    sim: dict[str, Any] = field(default_factory=dict)  # view, default solver options, problem builder

    # --- identity -----------------------------------------------------------------

    @property
    def cell_name(self) -> str:
        h = hashlib.sha1(json.dumps(self.settings, sort_keys=True, default=str).encode()).hexdigest()[:8]
        return f"{self.kind}_{h}"

    def port(self, name: str) -> CPort:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"{self.kind} has no port {name!r}; ports: {', '.join(p.name for p in self.ports)}")

    @property
    def port_names(self) -> list[str]:
        return [p.name for p in self.ports]

    def add_polygon(self, layer: str, pts) -> None:
        self.polygons.setdefault(layer, []).append(np.asarray(pts, dtype=float))

    # --- geometry -----------------------------------------------------------------

    def flat_polygons(self) -> dict[str, list[np.ndarray]]:
        """All polygons including those of placed instances, in this component's coordinates."""
        out = {k: list(v) for k, v in self.polygons.items()}
        for inst in self.instances:
            for layer, polys in inst.component.flat_polygons().items():
                out.setdefault(layer, []).extend(inst.transform.apply(p) for p in polys)
        return out

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        pts = [p for ps in self.flat_polygons().values() for p in ps]
        pts += [np.array([p.center]) for p in self.ports]
        allp = np.concatenate(pts, axis=0)
        return allp.min(axis=0), allp.max(axis=0)

    # --- GDS ------------------------------------------------------------------------

    def gds_cells(self, tech: Technology | str | None = None) -> dict[str, dict[str, Any]]:
        """Cells for :func:`lightsmith.layout.write_gds`: this cell plus every distinct child cell."""
        tech = get_technology(tech)
        cells: dict[str, dict[str, Any]] = {}

        def visit(c: Component):
            if c.cell_name in cells:
                return
            for inst in c.instances:
                visit(inst.component)
            polys: dict[tuple[int, int], list[np.ndarray]] = {}
            for layer, ps in c.polygons.items():
                polys.setdefault(tech.gds_layer(layer), []).extend(ps)
            refs = [{"cell": i.component.cell_name, "origin_nm": i.transform.origin,
                     "angle_deg": i.transform.rotation_deg % 360, "x_reflection": i.transform.mirror}
                    for i in c.instances]
            cells[c.cell_name] = {"polygons": polys, "refs": refs}

        visit(self)
        return cells

    def write_gds(self, path: str | Path, tech: Technology | str | None = None) -> Path:
        """Write this component (hierarchically) to a GDSII file; the top cell is ``cell_name``."""
        from ..layout import write_gds

        return write_gds(path, self.gds_cells(tech))

    # --- simulation ---------------------------------------------------------------------

    def device_problem(self, tech: Technology | str | None = None, wavelengths: Wavelengths | None = None, *,
                       dimension: int = 2, options: dict[str, dict[str, Any]] | None = None,
                       port_margin_nm: float = 1500.0, padding_nm: float = 1500.0) -> DeviceProblem:
        """A :class:`DeviceProblem` of this component with ports just outside its reference planes.

        dimension=2: plan view (effective-index materials, structures in xy); dimension=3: polygons
        extruded with the technology's z ranges. Each port gets a straight input stub reaching
        through the absorbing boundary, and a mode window of width + 2 * port_margin_nm (narrowed so
        that windows of neighbouring ports on the same plane do not overlap). Default solver options
        of the component are merged under `options`. Solver ports sit PORT_OFFSET_NM outside the
        component's ports; :func:`lightsmith.pdk.flow.simulate` moves S back onto the component's ports.
        """
        tech = get_technology(tech)
        wavelengths = wavelengths if wavelengths is not None else [1550.0]
        builder: Callable | None = self.sim.get("builder")
        if builder is not None:
            return builder(self, tech, wavelengths, dimension=dimension, options=options)
        return plan_view_problem(self, tech, wavelengths, dimension=dimension, options=options,
                                 port_margin_nm=port_margin_nm, padding_nm=padding_nm)



def _merge(a: dict, b: dict | None) -> dict:
    out = {k: dict(v) for k, v in (a or {}).items()}
    for k, v in (b or {}).items():
        out[k] = {**out.get(k, {}), **v}
    return out


def _port_windows(ports: list[CPort], margin: float) -> list[float]:
    """Transverse window per port, capped at the pitch to other ports on the same plane."""
    out = []
    for p in ports:
        w = p.width_nm + 2 * margin
        a, t = p.normal_axis, 1 - p.normal_axis
        for q in ports:
            if q is p or q.normal_axis != a or abs(q.center[a] - p.center[a]) > 1e-3:
                continue
            gap = abs(q.center[t] - p.center[t])
            w = min(w, gap)  # each window extends at most half the pitch towards its neighbour
        out.append(w)
    return out


STUB_NM = 6000.0  # input/output waveguide stubs beyond the port planes (into the absorbing boundary)
PORT_BUFFER_NM = 500.0  # domain extent beyond the outermost port planes
# Solver port planes sit this far out on the straight stub: FDFD measures modal amplitudes one and two cells
# inside its plane, which must still be straight waveguide (not the start of a bend or taper). The flow moves
# the S-parameters back to the component's ports (lightsmith.pdk.flow).
PORT_OFFSET_NM = 100.0


def plan_view_problem(comp: Component, tech: Technology, wavelengths: Wavelengths, *, dimension: int = 2,
                      options: dict | None = None, port_margin_nm: float = 1500.0,
                      padding_nm: float = 1500.0) -> DeviceProblem:
    wl = wavelength_array(wavelengths)
    polys = comp.flat_polygons()
    stubs: dict[str, list[np.ndarray]] = {}
    for p in comp.ports:
        o, n = p.outward, np.array([-p.outward[1], p.outward[0]])
        c = np.asarray(p.center)
        a, b = c - o * 1.0, c + o * STUB_NM
        hw = p.width_nm / 2
        stubs.setdefault(p.layer, []).append(np.array([a + n * hw, b + n * hw, b - n * hw, a - n * hw]))
    windows = _port_windows(comp.ports, port_margin_nm)
    pts = np.concatenate([q for ps in polys.values() for q in ps]
                         + [np.array([p.center]) for p in comp.ports], axis=0)
    lo, hi = pts.min(axis=0) - padding_nm, pts.max(axis=0) + padding_nm
    # Along each port normal the domain ends a little beyond the port plane.
    for ax in (0, 1):
        plus = [p.center[ax] for p in comp.ports if p.normal_axis == ax and p.outward[ax] > 0]
        minus = [p.center[ax] for p in comp.ports if p.normal_axis == ax and p.outward[ax] < 0]
        if plus:
            hi[ax] = max(plus) + PORT_BUFFER_NM
        if minus:
            lo[ax] = min(minus) - PORT_BUFFER_NM
    structures: list[Structure] = []
    core = tech.core
    for layer, ld in tech.layers.items():
        if ld.kind == "drawing":
            continue
        ps = polys.get(layer, []) + stubs.get(layer, [])
        if not ps:
            continue
        if dimension == 2:
            mat = tech.eim_material(layer, wl)
            z0, z1 = -1.0, 1.0
        else:
            mat = ld.material if ld.kind == "material" else tech.cladding
            z0, z1 = ld.z_min_nm, ld.z_max_nm
        for i, q in enumerate(ps):
            structures.append(Structure(name=f"{layer}#{i}", material=mat, geometry=Prism(
                profile=Polygon(vertices_nm=[(float(x), float(y)) for x, y in q]), z_min_nm=z0, z_max_nm=z1,
                sidewall_angle_deg=0.0 if dimension == 2 else ld.sidewall_angle_deg)))
    if dimension == 3:
        zc = (core.z_min_nm + core.z_max_nm) / 2
        h = core.z_max_nm - core.z_min_nm + 2 * port_margin_nm
        box = tech.box
        if box != tech.cladding:
            structures.insert(0, _slab(lo, hi, -tech.box_thickness_nm, 0.0, box, "BOX"))
        z_lo, z_hi = zc - h / 2, zc + h / 2
    else:
        zc, h, z_lo, z_hi = 0.0, 1.0, 0.0, 0.0
    ports = []
    for p, w in zip(comp.ports, windows):
        ax = p.normal_axis
        size = [0.0, 0.0, h]
        size[1 - ax] = w
        direction = "-" if p.outward[ax] > 0 else "+"
        c = np.asarray(p.center) + p.outward * PORT_OFFSET_NM
        ports.append(Port(name=p.name, center_nm=(float(c[0]), float(c[1]), zc), size_nm=tuple(size),
                          direction=direction))
    defaults = {"eme": {"num_modes": 30} if dimension == 2 else {"num_modes": 10}, "fdfd": {}}
    opts = _merge(_merge(defaults, comp.sim.get("options")), options)
    center = ((lo + hi) / 2).tolist()
    size = (hi - lo).tolist()
    return DeviceProblem(dimension=dimension, structures=structures, background=tech.cladding,
                         domain_center_nm=(center[0], center[1], (z_lo + z_hi) / 2),
                         domain_size_nm=(size[0], size[1], z_hi - z_lo), ports=ports,
                         wavelengths=wl.tolist(), options=opts)


def _slab(lo, hi, z0, z1, material, name) -> Structure:
    from ..problems import Box

    return Structure(name=name, material=material, geometry=Box(
        center_nm=((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (z0 + z1) / 2),
        size_nm=(hi[0] - lo[0] + 2 * STUB_NM, hi[1] - lo[1] + 2 * STUB_NM, z1 - z0)))


# --- assembling components ---------------------------------------------------------------


class Assembly:
    """Place components by connecting their ports; produce a hierarchical :class:`Component`.

    ``asm.add("mmi_in", mmi)`` places a first instance; ``asm.connect("arm", bend, "o1", "mmi_in,o2")``
    moves `bend` so that its port o1 meets mmi_in's o2 (facing it) and records the connection;
    ``asm.expose("in", "mmi_in,o1")`` makes an external port. ``asm.build(kind, settings)`` returns the
    composite component whose circuit model is the netlist of its parts.
    """

    def __init__(self):
        self.instances: dict[str, Instance] = {}
        self.connections: list[tuple[str, str]] = []
        self.exposed: dict[str, str] = {}

    def add(self, name: str, component: Component, origin=(0.0, 0.0), rotation_deg: float = 0.0,
            mirror: bool = False) -> Instance:
        if name in self.instances or "," in name:
            raise ValueError(f"instance name {name!r} is taken or contains ','")
        inst = Instance(name, component, Transform(tuple(map(float, origin)), float(rotation_deg) % 360, mirror))
        self.instances[name] = inst
        return inst

    def port(self, ref: str) -> CPort:
        inst, _, pname = ref.partition(",")
        i = self.instances[inst]
        return i.component.port(pname).transformed(i.transform)

    def connect(self, name: str, component: Component, port: str, to: str, mirror: bool = False) -> Instance:
        """Add `component` with its `port` attached to the placed port `to` ('instance,port')."""
        target = self.port(to)
        p = component.port(port)
        o = -p.orientation_deg if mirror else p.orientation_deg
        rot = (target.orientation_deg + 180 - o) % 360
        t0 = Transform((0.0, 0.0), rot, mirror)
        moved = t0.apply(np.array([p.center]))[0]
        origin = np.asarray(target.center) - moved
        inst = self.add(name, component, tuple(origin.tolist()), rot, mirror)
        if abs(p.width_nm - target.width_nm) > 1e-6:
            raise ValueError(f"width mismatch connecting {name},{port} ({p.width_nm:g} nm) to {to} "
                             f"({target.width_nm:g} nm)")
        self.connections.append((to, f"{name},{port}"))
        return inst

    def link(self, a: str, b: str, tol_nm: float = 1.0) -> None:
        """Record a connection between two already placed ports (they must coincide and face each other)."""
        pa, pb = self.port(a), self.port(b)
        if np.hypot(*(np.asarray(pa.center) - pb.center)) > tol_nm or \
                abs(((pa.orientation_deg - pb.orientation_deg) % 360) - 180) > 1e-6:
            raise ValueError(f"ports {a} at {pa.center} ({pa.orientation_deg:g} deg) and {b} at {pb.center} "
                             f"({pb.orientation_deg:g} deg) do not meet")
        self.connections.append((a, b))

    def expose(self, name: str, ref: str) -> None:
        self.exposed[name] = ref

    def build(self, kind: str, settings: dict[str, Any]) -> Component:
        ports = [self.port(ref).transformed(Transform(), name) for name, ref in self.exposed.items()]
        netlist = {
            "instances": {n: {"component": f"pdk:{i.component.kind}", "settings": dict(i.component.settings)}
                          for n, i in self.instances.items()},
            "connections": [list(c) for c in self.connections],
            "ports": dict(self.exposed),
        }
        return Component(kind=kind, settings=settings, ports=ports, instances=list(self.instances.values()),
                         netlist=netlist, model="netlist")


def rename_ports(comp: Component, mapping: dict[str, str]) -> Component:
    return replace(comp, ports=[replace(p, name=mapping.get(p.name, p.name)) for p in comp.ports])
