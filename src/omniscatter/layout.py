"""Layout integration: GDS files and gdsfactory components/netlists.

- :func:`resolve_layout` turns the ``GdsLayout`` of a ``DeviceProblem`` into
  extruded :class:`~omniscatter.problems.Prism` structures. It uses a small
  built-in GDSII reader (no dependencies; flattens SREF/AREF with rotation,
  magnification and reflection), so every solver can call it.
- :func:`from_gdsfactory_component` builds a ``DeviceProblem`` from a
  gdsfactory component: polygons become prisms with z extents and materials,
  gdsfactory ports become ``Port`` planes.
- :func:`from_gdsfactory_netlist` builds a ``CircuitProblem`` from a
  gdsfactory netlist, mapping gdsfactory cells to omniscatter compact models.

gdsfactory (MIT) is optional and imported lazily; install it with
``pip install gdsfactory`` (or ``uv run --with gdsfactory``). Note that recent
gdsfactory releases depend on kfactory and KLayout's Python module, which is
GPL-3.0-or-later; omniscatter itself never imports KLayout.

Units: gdsfactory works in um, omniscatter in nm; conversion happens here.
"""

from __future__ import annotations

import math
import re
import struct
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .materials import Material
from .problems import (
    CircuitProblem,
    DeviceProblem,
    GdsLayout,
    Instance,
    Polygon,
    Port,
    Prism,
    Structure,
    Wavelengths,
)

UM = 1000.0  # nm per um

LayerKey = tuple[int, int]


class LayerSpec(BaseModel):
    """How one GDS layer/datatype is extruded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    material: Material
    z_min_nm: float
    z_max_nm: float
    sidewall_angle_deg: float = Field(0.0, description="0 = vertical; positive = narrower at the top.")
    name: str | None = None


_MATERIAL = TypeAdapter(Material)


def parse_layer(key: Any) -> LayerKey:
    """(layer, datatype) from a tuple, 'L/D' string, int (datatype 0) or gdsfactory layer enum/LayerInfo."""
    if hasattr(key, "layer") and hasattr(key, "datatype"):
        return int(key.layer), int(key.datatype)
    if isinstance(key, str):
        m = re.fullmatch(r"\s*(\d+)\s*[/:,]\s*(\d+)\s*", key)
        if m:
            return int(m.group(1)), int(m.group(2))
        if key.strip().isdigit():
            return int(key), 0
        try:  # a layer name in the active gdsfactory PDK, e.g. "WG"
            import gdsfactory as gf

            return parse_layer(getattr(gf.get_active_pdk().layers, key))
        except (ImportError, AttributeError, ValueError):
            raise ValueError(f"cannot parse layer {key!r}; use 'layer/datatype', e.g. '1/0'") from None
    if isinstance(key, int):
        return key, 0
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return int(key[0]), int(key[1])
    raise ValueError(f"cannot parse layer {key!r}; use (layer, datatype) or 'layer/datatype'")


def normalize_layer_map(layer_map: Mapping[Any, Any]) -> dict[LayerKey, LayerSpec]:
    """Accept {'1/0': {'material': ..., 'z_min_nm': .., 'z_max_nm': ..}} or LayerSpec values, any key form."""
    out: dict[LayerKey, LayerSpec] = {}
    for k, v in layer_map.items():
        spec = v if isinstance(v, LayerSpec) else LayerSpec.model_validate(v)
        if spec.z_max_nm <= spec.z_min_nm:
            raise ValueError(f"layer {k!r}: z_max_nm must exceed z_min_nm")
        out[parse_layer(k)] = spec
    return out


# --- built-in GDSII reader / writer ----------------------------------------------

_HEADER, _BGNLIB, _LIBNAME, _UNITS, _ENDLIB, _BGNSTR, _STRNAME, _ENDSTR = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
_BOUNDARY, _PATH, _SREF, _AREF, _TEXT, _LAYER, _DATATYPE, _WIDTH, _XY, _ENDEL = (0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                                                               0x0E, 0x0F, 0x10, 0x11)
_SNAME, _COLROW, _STRANS, _MAG, _ANGLE, _PATHTYPE, _BOX, _BOXTYPE, _BGNEXTN, _ENDEXTN = (0x12, 0x13, 0x1A, 0x1B, 0x1C,
                                                                                         0x21, 0x2D, 0x2E, 0x30, 0x31)


def _real8_decode(b: bytes) -> float:
    sign = -1.0 if b[0] & 0x80 else 1.0
    exp = (b[0] & 0x7F) - 64
    mant = int.from_bytes(b[1:8], "big") / 2.0**56
    return sign * mant * 16.0**exp


def _real8_encode(x: float) -> bytes:
    if x == 0:
        return b"\0" * 8
    sign = 0x80 if x < 0 else 0
    x, e = abs(x), 64
    while x >= 1:
        x /= 16
        e += 1
    while x < 1 / 16:
        x *= 16
        e -= 1
    m = round(x * 2**56)
    if m >= 2**56:
        m //= 16
        e += 1
    return bytes([sign | e]) + m.to_bytes(7, "big")


def _records(data: bytes) -> Iterable[tuple[int, Any]]:
    i, n = 0, len(data)
    while i + 4 <= n:
        size, rtype, dtype = struct.unpack(">HBB", data[i:i + 4])
        if size < 4:
            break
        body = data[i + 4:i + size]
        i += size
        if dtype == 2:
            val: Any = struct.unpack(f">{len(body) // 2}h", body)
        elif dtype == 3:
            val = struct.unpack(f">{len(body) // 4}i", body)
        elif dtype == 5:
            val = [_real8_decode(body[k:k + 8]) for k in range(0, len(body), 8)]
        elif dtype == 6:
            val = body.rstrip(b"\0").decode("ascii", errors="replace")
        elif dtype == 1:
            val = struct.unpack(">H", body[:2])[0] if body else 0
        else:
            val = body
        yield rtype, val
        if rtype == _ENDLIB:
            break


class _Cell:
    def __init__(self, name):
        self.name = name
        self.polygons: list[tuple[LayerKey, np.ndarray]] = []  # in database units
        self.refs: list[dict[str, Any]] = []


def _path_polygons(xy: np.ndarray, width: float, pathtype: int, bgn: float, end: float) -> list[np.ndarray]:
    """Rectangles covering each segment of a GDS PATH (joints are approximated)."""
    hw = abs(width) / 2
    out = []
    for k in range(len(xy) - 1):
        p0, p1 = xy[k].astype(float), xy[k + 1].astype(float)
        d = p1 - p0
        length = np.hypot(*d)
        if length == 0:
            continue
        u = d / length
        nrm = np.array([-u[1], u[0]])
        e0 = (hw if pathtype in (1, 2) else bgn if pathtype == 4 else 0.0) if k == 0 else 0.0
        e1 = (hw if pathtype in (1, 2) else end if pathtype == 4 else 0.0) if k == len(xy) - 2 else 0.0
        a, b = p0 - u * e0, p1 + u * e1
        out.append(np.array([a + nrm * hw, b + nrm * hw, b - nrm * hw, a - nrm * hw]))
    return out


def _parse_gds(data: bytes) -> tuple[float, dict[str, _Cell]]:
    dbu_m = 1e-9
    cells: dict[str, _Cell] = {}
    cur: _Cell | None = None
    el: dict[str, Any] | None = None
    for rtype, val in _records(data):
        if rtype == _UNITS:
            dbu_m = float(val[1])
        elif rtype == _BGNSTR:
            cur = None
        elif rtype == _STRNAME:
            cur = cells.setdefault(val, _Cell(val))
        elif rtype in (_BOUNDARY, _PATH, _SREF, _AREF, _TEXT, _BOX):
            el = {"type": rtype, "datatype": 0, "width": 0, "pathtype": 0, "bgn": 0, "end": 0,
                  "strans": 0, "mag": 1.0, "angle": 0.0}
        elif el is not None and rtype == _LAYER:
            el["layer"] = val[0]
        elif el is not None and rtype in (_DATATYPE, _BOXTYPE):
            el["datatype"] = val[0]
        elif el is not None and rtype == _WIDTH:
            el["width"] = val[0]
        elif el is not None and rtype == _PATHTYPE:
            el["pathtype"] = val[0]
        elif el is not None and rtype == _BGNEXTN:
            el["bgn"] = val[0]
        elif el is not None and rtype == _ENDEXTN:
            el["end"] = val[0]
        elif el is not None and rtype == _XY:
            el["xy"] = np.asarray(val, dtype=np.int64).reshape(-1, 2)
        elif el is not None and rtype == _SNAME:
            el["sname"] = val
        elif el is not None and rtype == _COLROW:
            el["colrow"] = (val[0], val[1])
        elif el is not None and rtype == _STRANS:
            el["strans"] = val
        elif el is not None and rtype == _MAG:
            el["mag"] = val[0]
        elif el is not None and rtype == _ANGLE:
            el["angle"] = val[0]
        elif rtype == _ENDEL and el is not None:
            if cur is None:
                raise ValueError("GDS element outside a cell")
            t = el["type"]
            key = (int(el.get("layer", 0)), int(el["datatype"]))
            if t in (_BOUNDARY, _BOX):
                xy = el["xy"]
                if len(xy) > 1 and np.array_equal(xy[0], xy[-1]):
                    xy = xy[:-1]
                if len(xy) >= 3:
                    cur.polygons.append((key, xy.astype(float)))
            elif t == _PATH:
                for poly in _path_polygons(el["xy"], el["width"], el["pathtype"], el["bgn"], el["end"]):
                    cur.polygons.append((key, poly))
            elif t in (_SREF, _AREF):
                cur.refs.append(el)
            el = None
    return dbu_m, cells


def _transform(pts: np.ndarray, ref: dict[str, Any], origin: np.ndarray) -> np.ndarray:
    p = pts.copy()
    if ref["strans"] & 0x8000:
        p[:, 1] = -p[:, 1]
    p = p * ref["mag"]
    a = math.radians(ref["angle"])
    c, s = math.cos(a), math.sin(a)
    p = p @ np.array([[c, s], [-s, c]])
    return p + origin


def _flatten(cells: dict[str, _Cell], name: str, cache: dict, stack: tuple = ()) -> list[tuple[LayerKey, np.ndarray]]:
    if name in cache:
        return cache[name]
    if name in stack:
        raise ValueError(f"GDS cell reference cycle: {' -> '.join(stack + (name,))}")
    if name not in cells:
        raise ValueError(f"GDS references a missing cell {name!r}")
    cell = cells[name]
    out = list(cell.polygons)
    for ref in cell.refs:
        sub = _flatten(cells, ref["sname"], cache, stack + (name,))
        xy = ref["xy"].astype(float)
        if ref["type"] == _SREF:
            origins = [xy[0]]
        else:
            cols, rows = ref.get("colrow", (1, 1))
            dc, dr = (xy[1] - xy[0]) / cols, (xy[2] - xy[0]) / rows
            origins = [xy[0] + i * dc + j * dr for i in range(cols) for j in range(rows)]
        for o in origins:
            out.extend((k, _transform(p, ref, o)) for k, p in sub)
    cache[name] = out
    return out


def read_gds(path: str | Path, cell: str | None = None) -> dict[LayerKey, list[np.ndarray]]:
    """Flattened polygons per (layer, datatype) in nm from a GDSII file.

    ``cell`` defaults to the single top cell (one that no other cell references)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GDS file not found: {path}")
    dbu_m, cells = _parse_gds(path.read_bytes())
    if not cells:
        raise ValueError(f"{path.name}: no cells found (is it a GDSII stream file?)")
    if cell is None:
        referenced = {r["sname"] for n, c in cells.items() if not n.startswith("$$$CONTEXT_INFO") for r in c.refs}
        # KLayout writes a "$$$CONTEXT_INFO$$$" cell of metadata; it is not geometry.
        tops = [n for n in cells if n not in referenced and not n.startswith("$$$CONTEXT_INFO")]
        if len(tops) != 1:
            raise ValueError(f"{path.name}: {len(tops)} top cells ({', '.join(tops)}); choose one with `cell`")
        cell = tops[0]
    if cell not in cells:
        raise ValueError(f"{path.name}: no cell {cell!r}; cells: {', '.join(cells)}")
    scale = dbu_m * 1e9
    out: dict[LayerKey, list[np.ndarray]] = {}
    for key, poly in _flatten(cells, cell, {}):
        if _area(poly) != 0:  # skip degenerate (metadata) polygons
            out.setdefault(key, []).append(poly * scale)
    return out


def write_gds(path: str | Path, cells: Mapping[str, Mapping[str, Any]], dbu_nm: float = 1.0,
              libname: str = "omniscatter") -> Path:
    """Write a minimal GDSII file.

    ``cells`` maps cell name -> {"polygons": {(layer, datatype) or 'L/D': [array (N, 2) in nm, ...]},
    "refs": [{"cell": name, "origin_nm": (x, y), "angle_deg": 0, "magnification": 1,
    "x_reflection": False, "columns": 1, "rows": 1, "spacing_nm": ((dx, 0), (0, dy))}]}."""
    out = bytearray()

    def rec(rtype: int, dtype: int, payload: bytes = b""):
        if len(payload) % 2:
            payload += b"\0"
        out.extend(struct.pack(">HBB", 4 + len(payload), rtype, dtype) + payload)

    def ints(fmt, vals):
        return struct.pack(f">{len(vals)}{fmt}", *vals)

    def db(v):
        return round(float(v) / dbu_nm)

    stamp = [2024, 1, 1, 0, 0, 0] * 2
    rec(_HEADER, 2, ints("h", [600]))
    rec(_BGNLIB, 2, ints("h", stamp))
    rec(_LIBNAME, 6, libname.encode("ascii"))
    rec(_UNITS, 5, _real8_encode(1e-3) + _real8_encode(dbu_nm * 1e-9))
    for name, c in cells.items():
        rec(_BGNSTR, 2, ints("h", stamp))
        rec(_STRNAME, 6, name.encode("ascii"))
        for key, polys in (c.get("polygons") or {}).items():
            layer, dt = parse_layer(key)
            for poly in polys:
                pts = [(db(x), db(y)) for x, y in np.asarray(poly, dtype=float)]
                pts.append(pts[0])
                rec(_BOUNDARY, 0)
                rec(_LAYER, 2, ints("h", [layer]))
                rec(_DATATYPE, 2, ints("h", [dt]))
                rec(_XY, 3, ints("i", [v for p in pts for v in p]))
                rec(_ENDEL, 0)
        for r in c.get("refs") or []:
            cols, rows = int(r.get("columns", 1)), int(r.get("rows", 1))
            aref = cols > 1 or rows > 1
            rec(_AREF if aref else _SREF, 0)
            rec(_SNAME, 6, r["cell"].encode("ascii"))
            flags = 0x8000 if r.get("x_reflection") else 0
            mag, ang = float(r.get("magnification", 1.0)), float(r.get("angle_deg", 0.0))
            if flags or mag != 1 or ang != 0:
                rec(_STRANS, 1, struct.pack(">H", flags))
                if mag != 1:
                    rec(_MAG, 5, _real8_encode(mag))
                if ang != 0:
                    rec(_ANGLE, 5, _real8_encode(ang))
            ox, oy = r.get("origin_nm", (0.0, 0.0))
            if aref:
                (cx, cy), (rx, ry) = r.get("spacing_nm", ((0.0, 0.0), (0.0, 0.0)))
                rec(_COLROW, 2, ints("h", [cols, rows]))
                xy = [db(ox), db(oy), db(ox + cols * cx), db(oy + cols * cy), db(ox + rows * rx), db(oy + rows * ry)]
            else:
                xy = [db(ox), db(oy)]
            rec(_XY, 3, ints("i", xy))
            rec(_ENDEL, 0)
        rec(_ENDSTR, 0)
    rec(_ENDLIB, 0)
    path = Path(path)
    path.write_bytes(bytes(out))
    return path


# --- GdsLayout -> structures ----------------------------------------------------


def polygons_to_structures(polygons: Mapping[LayerKey, list[np.ndarray]], layer_map: Mapping[Any, Any],
                           warn_unmapped: bool = True) -> list[Structure]:
    """Extrude polygons (nm) into Prism structures, in layer_map order (later layers override earlier ones)."""
    specs = normalize_layer_map(layer_map)
    unmapped = sorted(set(polygons) - set(specs))
    if unmapped and warn_unmapped:
        warnings.warn("layout layers without a layer_map entry are ignored: "
                      + ", ".join(f"{a}/{b}" for a, b in unmapped), stacklevel=2)
    out = []
    for key, spec in specs.items():
        for i, poly in enumerate(polygons.get(key, [])):
            poly = np.asarray(poly, dtype=float)
            if len(poly) < 3 or abs(_area(poly)) == 0:
                continue
            label = spec.name or f"{key[0]}/{key[1]}"
            out.append(Structure(
                name=f"{label}#{i}",
                geometry=Prism(profile=Polygon(vertices_nm=[(float(x), float(y)) for x, y in poly]),
                               z_min_nm=spec.z_min_nm, z_max_nm=spec.z_max_nm,
                               sidewall_angle_deg=spec.sidewall_angle_deg),
                material=spec.material))
    return out


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def layout_structures(layout: GdsLayout, base_dir: str | Path | None = None) -> list[Structure]:
    """Structures for a ``GdsLayout`` (path resolved against ``base_dir`` when relative)."""
    path = Path(layout.path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return polygons_to_structures(read_gds(path, layout.cell), layout.layer_map)


def resolve_layout(problem: DeviceProblem, base_dir: str | Path | None = None) -> list[Structure]:
    """All structures of a device: explicit ``structures`` first, then the extruded GDS layout.

    Later structures override earlier ones where they overlap, so the layout
    (waveguide cores) wins over explicit background slabs (substrate, cladding).
    Solvers call this instead of reading ``problem.structures`` directly."""
    out = list(problem.structures)
    if problem.layout is not None:
        out += layout_structures(problem.layout, base_dir)
    return out


def resolved_device(problem: DeviceProblem, base_dir: str | Path | None = None) -> DeviceProblem:
    """A copy of `problem` with the layout expanded into ``structures`` and ``layout`` cleared."""
    return DeviceProblem.model_validate({**problem.model_dump(), "structures":
                                         [s.model_dump() for s in resolve_layout(problem, base_dir)],
                                         "layout": None})


# --- gdsfactory -> DeviceProblem ------------------------------------------------


def _require_gdsfactory():
    try:
        import gdsfactory as gf
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("this function needs gdsfactory (MIT): `pip install gdsfactory` "
                          "or `uv run --with gdsfactory ...`") from e
    return gf


def component_polygons(component, merge: bool = False) -> dict[LayerKey, list[np.ndarray]]:
    """Flattened polygons of a gdsfactory component per (layer, datatype), in nm."""
    if merge:  # merging modifies the cell; gdsfactory cells are locked, so merge a copy
        component = component.copy()
    raw = component.get_polygons_points(merge=merge, by="tuple")
    return {parse_layer(k): [np.asarray(p, dtype=float) * UM for p in v] for k, v in raw.items()}


def _port_layer(port) -> LayerKey | None:
    # kfactory ports carry a LayerInfo (layer, datatype); `port.layer` is a layout index, not a GDS layer.
    info = getattr(port, "layer_info", None)
    return parse_layer(info) if info is not None and hasattr(info, "datatype") else None


def gdsfactory_port(port, layer_map: Mapping[LayerKey, LayerSpec], margin_nm: float, num_modes: int,
                    z_range: tuple[float, float]) -> Port:
    """A gdsfactory port as an omniscatter Port plane (normal along x or y)."""
    ang = float(port.orientation) % 360
    k = round(ang / 90) % 4
    if abs(ang - 90 * round(ang / 90)) > 1e-6:
        raise ValueError(f"port {port.name!r}: only Manhattan orientations are supported (got {ang} deg)")
    spec = layer_map.get(_port_layer(port)) if _port_layer(port) is not None else None
    z0, z1 = (spec.z_min_nm, spec.z_max_nm) if spec is not None else z_range
    x, y = (float(v) * UM for v in port.center)
    w = float(port.width) * UM + 2 * margin_nm
    h = (z1 - z0) + 2 * margin_nm
    # gdsfactory orientation points out of the device; light enters against it.
    size = (0.0, w, h) if k in (0, 2) else (w, 0.0, h)
    direction = {0: "-", 1: "-", 2: "+", 3: "+"}[k]
    return Port(name=str(port.name), center_nm=(x, y, (z0 + z1) / 2), size_nm=size, direction=direction,
                num_modes=num_modes)


def from_gdsfactory_component(component, layer_map: Mapping[Any, Any], *, background, wavelengths: Wavelengths,
                              padding_nm: float = 1000.0, port_margin_nm: float = 1000.0, num_modes: int = 1,
                              port_types: Iterable[str] | None = ("optical",), dimension: int = 3,
                              extra_structures: Iterable[Structure] = (), merge: bool = False,
                              options: dict[str, dict[str, Any]] | None = None) -> DeviceProblem:
    """A ``DeviceProblem`` from a gdsfactory component.

    ``layer_map`` maps layers (``(1, 0)``, ``"1/0"``, a gdsfactory ``LAYER.WG`` or a PDK
    layer name) to ``{"material": ..., "z_min_nm": ..., "z_max_nm": ...}``; its order is the
    override order. Each polygon becomes a ``Prism``; ports of the given ``port_types``
    become ``Port`` planes sized width + 2 * port_margin_nm by thickness + 2 * port_margin_nm,
    with ``direction`` pointing into the device. The domain is the bounding box of the mapped
    geometry plus ``padding_nm``. ``extra_structures`` (substrate, cladding) come first.
    """
    _require_gdsfactory()
    specs = normalize_layer_map(layer_map)
    if not specs:
        raise ValueError("layer_map is empty")
    polys = component_polygons(component, merge=merge)
    extra = list(extra_structures)
    structures = extra + polygons_to_structures(polys, specs)
    if len(structures) == len(extra):
        raise ValueError("no polygons on any layer of layer_map; component layers are "
                         + ", ".join(f"{a}/{b}" for a, b in sorted(polys)))
    used = [p for k in specs for p in polys.get(k, [])]
    pts = np.concatenate(used, axis=0)
    zlo = min(s.z_min_nm for k, s in specs.items() if k in polys)
    zhi = max(s.z_max_nm for k, s in specs.items() if k in polys)
    lo = np.array([pts[:, 0].min(), pts[:, 1].min(), zlo]) - padding_nm
    hi = np.array([pts[:, 0].max(), pts[:, 1].max(), zhi]) + padding_nm
    types = None if port_types is None else set(port_types)
    ports = [gdsfactory_port(p, specs, port_margin_nm, num_modes, (zlo, zhi)) for p in component.ports
             if types is None or getattr(p, "port_type", "optical") in types]
    if not ports:
        raise ValueError(f"component has no ports of type {sorted(types or [])}")
    if background is not None and not isinstance(background, BaseModel):
        background = _MATERIAL.validate_python(background)
    return DeviceProblem(dimension=dimension, structures=structures, background=background,
                         domain_center_nm=tuple(((lo + hi) / 2).tolist()), domain_size_nm=tuple((hi - lo).tolist()),
                         ports=ports, wavelengths=wavelengths, options=options or {})


# --- gdsfactory netlist -> CircuitProblem ----------------------------------------

Translator = Callable[[dict[str, Any], dict[str, Any]], "Instance | dict[str, Any]"]


def _num(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return float(v)
    return default


def _straight(settings, info):
    length = _num(info, "length", default=None)
    if length is None:
        length = _num(settings, "length", default=10.0)
    return {"component": "waveguide", "settings": {"length_nm": length * UM}}


def _bend(settings, info):
    radius = _num(info, "radius", default=None) or _num(settings, "radius", default=10.0)
    out = {"radius_nm": radius * UM, "angle_deg": _num(settings, "angle", default=90.0)}
    length = _num(info, "length")
    if length is not None:
        out["length_nm"] = length * UM
    return {"component": "bend", "settings": out}


def _ring(kind):
    def f(settings, info):
        radius = _num(info, "radius", default=None) or _num(settings, "radius", default=10.0)
        straight = (_num(settings, "length_x", default=0.0) or 0.0) + (_num(settings, "length_y", default=0.0) or 0.0)
        return {"component": kind, "settings": {"radius_nm": radius * UM, "straight_length_nm": straight * UM}}
    return f


def _same(name):
    return lambda settings, info: {"component": name, "settings": {}}


DEFAULT_GDSFACTORY_MODELS: dict[str, Translator] = {
    "straight": _straight, "taper": _straight, "taper_strip_to_ridge": _straight,
    "bend_euler": _bend, "bend_circular": _bend,
    "mmi1x2": _same("mmi1x2"), "mmi2x2": _same("mmi2x2"), "coupler": _same("coupler"),
    "ring_single": _ring("ring_single"), "ring_double": _ring("ring_double"),
    "grating_coupler_elliptical": _same("grating_coupler"),
    "grating_coupler_elliptical_trenches": _same("grating_coupler"),
    "grating_coupler_rectangular": _same("grating_coupler"),
}
"""gdsfactory cell name -> translator(settings, info) giving an omniscatter instance (lengths um -> nm).
Geometry-dependent optics (coupling vs gap, neff vs width) are not derived: set them through ``defaults``."""


def from_gdsfactory_netlist(netlist, wavelengths: Wavelengths, *,
                            models: Mapping[str, str | Translator | Instance | dict] | None = None,
                            defaults: Mapping[str, Mapping[str, Any]] | None = None,
                            options: dict[str, dict[str, Any]] | None = None) -> CircuitProblem:
    """A ``CircuitProblem`` from a gdsfactory component (its ``get_netlist()``) or netlist dict.

    ``models`` maps gdsfactory cell names to an omniscatter model name, a fixed
    instance (dict/Instance) or a translator ``f(settings, info) -> instance``; it
    extends :data:`DEFAULT_GDSFACTORY_MODELS` (straight/taper -> waveguide using the
    netlist's ``info.length``, bends, MMIs, coupler, rings, grating couplers).
    ``defaults`` adds settings per omniscatter model, e.g. ``{"waveguide": {"neff": 2.4,
    "ng": 4.2}, "bend": {...}}`` (instance-specific settings win). Nets that touch ports the
    target model does not have (e.g. electrical pads) are skipped with a warning.
    """
    from .circuits.models import get_model

    if hasattr(netlist, "get_netlist"):
        netlist = netlist.get_netlist()
    table: dict[str, Any] = {**DEFAULT_GDSFACTORY_MODELS, **(models or {})}
    defaults = defaults or {}
    instances: dict[str, dict[str, Any]] = {}
    unmapped: dict[str, list[str]] = {}
    for name, inst in (netlist.get("instances") or {}).items():
        comp = inst.get("component") if isinstance(inst, dict) else str(inst)
        settings = dict(inst.get("settings") or {}) if isinstance(inst, dict) else {}
        info = dict(inst.get("info") or {}) if isinstance(inst, dict) else {}
        tr = table.get(comp)
        if tr is None:
            unmapped.setdefault(comp, []).append(name)
            continue
        if isinstance(tr, str):
            res: Any = {"component": tr, "settings": {}}
        elif callable(tr):
            res = tr(settings, info)
        else:
            res = tr
        res = res.model_dump() if isinstance(res, Instance) else dict(res)
        model_name = res["component"]
        res["settings"] = {**defaults.get(model_name, {}), **(res.get("settings") or {})}
        instances[name] = res
    if unmapped:
        raise ValueError("no circuit model for gdsfactory cell(s): " + "; ".join(
            f"{c} (instances {', '.join(n)})" for c, n in unmapped.items())
            + ". Pass models={'<cell>': '<omniscatter model>' or a translator}.")

    def ports_of(inst_name: str) -> tuple[str, ...] | None:
        try:
            return get_model(instances[inst_name]["component"]).ports
        except KeyError:
            return None

    def known(ref: str) -> bool:
        inst, _, port = ref.partition(",")
        ps = ports_of(inst) if inst in instances else ()
        return ps is None or port in ps

    raw = netlist.get("nets")
    if raw is None:
        raw = [{"p1": a, "p2": b} for a, b in (netlist.get("connections") or {}).items()]
    connections, skipped = [], []
    for net in raw:
        a, b = (net["p1"], net["p2"]) if isinstance(net, dict) else tuple(net)
        if known(a) and known(b):
            connections.append((a, b))
        else:
            skipped.append(f"{a}<->{b}")
    ports = {}
    for name, ref in (netlist.get("ports") or {}).items():
        if known(ref):
            ports[name] = ref
        else:
            skipped.append(f"port {name}={ref}")
    if skipped:
        warnings.warn("skipped nets/ports on non-optical or unknown model ports: " + ", ".join(skipped), stacklevel=2)
    return CircuitProblem(instances=instances, connections=connections, ports=ports, wavelengths=wavelengths,
                          options=options or {})
