"""Parametric photonic components (layout + ports + simulation recipe).

Every cell is registered with a pydantic parameter model, so
``get_component("mmi1x2", length_mmi_nm=6000)`` validates its arguments and
``pdk:mmi1x2`` with the same settings names it in a circuit netlist. Lengths
are in nm. Cells start with port ``o1`` at the origin facing west (180 deg) and
extend towards +x; port names follow gdsfactory and the compact models in
:mod:`lightsmith.circuits.models`:

- two-ports: o1 (in) -> o2 (out);
- 1x2 (mmi1x2, y_branch): o1 west, o2 east-top, o3 east-bottom;
- 2x2 (mmi2x2, coupler): o1 west-bottom, o2 west-top, o3 east-top, o4 east-bottom (bar o1-o4, o2-o3);
- crossing: o1 west, o2 north, o3 east, o4 south;
- coupler_ring: o1 bus west, o2 ring left (facing up), o3 ring right (facing up), o4 bus east;
- grating_coupler: o1 waveguide, o2 fiber.

Every port carries the fundamental mode (TE0 for the default technology) of
the core-layer waveguide of the port's width. Default dimensions are designed
for the generic 220 nm SOI technology near 1550 nm in the 2D effective-index
model (see ``tests/test_pdk.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .component import Assembly, Component, CPort

# --- geometry helpers ----------------------------------------------------------------


def extrude(path: np.ndarray, width) -> np.ndarray:
    """Polygon of a centreline `path` (N, 2) with a constant or per-point width."""
    path = np.asarray(path, dtype=float)
    w = np.broadcast_to(np.asarray(width, dtype=float), (len(path),))
    d = np.gradient(path, axis=0)
    d /= np.hypot(d[:, 0], d[:, 1])[:, None]
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    left = path + n * (w[:, None] / 2)
    right = path - n * (w[:, None] / 2)
    return np.concatenate([left, right[::-1]], axis=0)


def rect(x0, x1, y0, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _segments(length: float, curvature: float, sagitta_nm: float = 2.0, minimum: int = 8) -> int:
    """Polyline segments so that chords deviate from the curve by at most `sagitta_nm` (~ L^2 kappa / 8)."""
    step = min(math.sqrt(8 * sagitta_nm / max(abs(curvature), 1e-12)), 500.0)
    return max(minimum, int(math.ceil(length / step)))


def arc_path(radius: float, angle_deg: float) -> np.ndarray:
    """Left-turning circular arc from (0, 0) heading +x."""
    t = np.linspace(0, math.radians(angle_deg), _segments(radius * math.radians(angle_deg), 1 / radius) + 1)
    return np.stack([radius * np.sin(t), radius * (1 - np.cos(t))], axis=1)


def euler_path(radius: float, angle_deg: float, p: float = 0.5) -> tuple[np.ndarray, float]:
    """Left-turning partial Euler bend from (0, 0) heading +x; returns (path, length).

    A fraction `p` of the angle is turned in two clothoid sections (curvature rising
    linearly from 0), the rest on a circular arc of the minimum radius. The shape is
    scaled so the bend has the same footprint as a circular bend of `radius`
    (for 90 degrees: the end point is at (radius, radius))."""
    theta = math.radians(angle_deg)
    th_e = p * theta / 2  # turned by each clothoid
    # Curvature k(s): clothoid 0 -> 1/R0 over L_e (th_e = L_e / (2 R0)), arc 1/R0 over L_c, clothoid back.
    r0 = 1.0
    le = 2 * r0 * th_e
    lc = r0 * (theta - 2 * th_e)
    total = 2 * le + lc
    n = 4000
    s = np.linspace(0, total, n + 1)
    k = np.where(s < le, s / (le * r0) if le > 0 else 1 / r0,
                 np.where(s <= le + lc, 1 / r0, (total - s) / (le * r0) if le > 0 else 1 / r0))
    ang = np.concatenate([[0.0], np.cumsum((k[1:] + k[:-1]) / 2 * np.diff(s))])
    xy = np.concatenate([[[0.0, 0.0]], np.cumsum(np.stack([(np.cos(ang[1:]) + np.cos(ang[:-1])) / 2,
                                                           (np.sin(ang[1:]) + np.sin(ang[:-1])) / 2], 1)
                                               * np.diff(s)[:, None], axis=0)])
    # Footprint: for 90 deg match the end x; in general match the circular bend's chord length.
    chord = math.hypot(*xy[-1])
    scale = radius * 2 * math.sin(theta / 2) / chord
    xy *= scale
    idx = np.linspace(0, n, _segments(total * scale, 1 / (r0 * scale)) + 1).round().astype(int)
    return xy[idx], total * scale


def sbend_path(dx: float, dy: float) -> np.ndarray:
    """Raised-cosine S-bend from (0, 0) to (dx, dy), horizontal at both ends.

    Finer than other paths (0.5 nm sagitta): S-bend devices run EME with one cell per segment, and the lateral
    step between cells costs mode-mismatch loss (coupler at 2 nm sagitta: 3.6 % loss, at 0.5 nm: 0.3 %)."""
    n = _segments(math.hypot(dx, dy), (math.pi / dx) ** 2 * abs(dy) / 2, sagitta_nm=0.5)
    x = np.linspace(0, dx, n + 1)
    return np.stack([x, dy / 2 * (1 - np.cos(np.pi * x / dx))], axis=1)


def path_length(path: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(path, axis=0).T)))


# --- registry ----------------------------------------------------------------------------


class CellParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class CellSpec:
    name: str
    params: type[CellParams]
    fn: Callable[[Any], Component]
    ports: tuple[str, ...]
    doc: str

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "ports": list(self.ports), "doc": self.doc,
                "params_schema": self.params.model_json_schema()}


CELLS: dict[str, CellSpec] = {}


def cell(name: str, params: type[CellParams], ports: tuple[str, ...]):
    """Register a parametric cell ``fn(params) -> Component`` under `name`."""

    def deco(fn):
        CELLS[name] = CellSpec(name, params, fn, ports, " ".join((fn.__doc__ or "").split()))
        return fn

    return deco


def cell_params(name: str, settings: dict[str, Any] | None = None) -> CellParams:
    if name not in CELLS:
        raise KeyError(f"unknown component {name!r}; available: {', '.join(sorted(CELLS))}")
    return CELLS[name].params.model_validate(settings or {})


def get_component(name: str, **settings) -> Component:
    """Build a registered component, e.g. ``get_component("mmi1x2", length_mmi_nm=6000)``."""
    p = cell_params(name, settings)
    comp = CELLS[name].fn(p)
    comp.kind, comp.settings = name, p.model_dump()
    return comp


def list_components() -> list[dict[str, Any]]:
    return [c.info() for c in CELLS.values()]


def _new(name: str, p: CellParams, **kw) -> Component:
    return Component(kind=name, settings=p.model_dump(), **kw)


W0 = 500.0


# --- waveguides --------------------------------------------------------------------------


class StraightParams(CellParams):
    length_nm: float = Field(10_000.0, ge=0)
    width_nm: float = Field(W0, gt=0)


@cell("straight", StraightParams, ("o1", "o2"))
def straight(p: StraightParams) -> Component:
    """Straight waveguide. Its model comes from a mode solve (neff vs wavelength), not a device solve."""
    c = _new("straight", p, model="waveguide")
    c.add_polygon("WG", rect(0, p.length_nm, -p.width_nm / 2, p.width_nm / 2))
    c.ports = [CPort("o1", (0.0, 0.0), 180.0, p.width_nm), CPort("o2", (p.length_nm, 0.0), 0.0, p.width_nm)]
    return c


class BendParams(CellParams):
    radius_nm: float = Field(5000.0, gt=0)
    angle_deg: float = Field(90.0, gt=0, le=180)
    width_nm: float = Field(W0, gt=0)


def _bend_component(name, p, path) -> Component:
    c = _new(name, p)
    c.add_polygon("WG", extrude(path, p.width_nm))
    c.ports = [CPort("o1", (0.0, 0.0), 180.0, p.width_nm),
               CPort("o2", tuple(path[-1].tolist()), p.angle_deg % 360, p.width_nm)]
    c.sim = {"options": {"fdfd": {}}}
    return c


@cell("bend_circular", BendParams, ("o1", "o2"))
def bend_circular(p: BendParams) -> Component:
    """Circular bend turning left by angle_deg."""
    return _bend_component("bend_circular", p, arc_path(p.radius_nm, p.angle_deg))


class EulerParams(BendParams):
    p: float = Field(0.5, ge=0, le=1, description="Fraction of the turn in clothoid (Euler) sections.")


@cell("bend_euler", EulerParams, ("o1", "o2"))
def bend_euler(p: EulerParams) -> Component:
    """Partial Euler bend (clothoid - arc - clothoid) with the footprint of a circular bend of radius_nm."""
    path, _ = euler_path(p.radius_nm, p.angle_deg, p.p)
    return _bend_component("bend_euler", p, path)


class TaperParams(CellParams):
    length_nm: float = Field(10_000.0, gt=0)
    width1_nm: float = Field(W0, gt=0)
    width2_nm: float = Field(1000.0, gt=0)


@cell("taper", TaperParams, ("o1", "o2"))
def taper(p: TaperParams) -> Component:
    """Linear width taper from width1_nm (o1) to width2_nm (o2)."""
    c = _new("taper", p)
    c.add_polygon("WG", np.array([[0, -p.width1_nm / 2], [p.length_nm, -p.width2_nm / 2],
                                  [p.length_nm, p.width2_nm / 2], [0, p.width1_nm / 2]]))
    c.ports = [CPort("o1", (0.0, 0.0), 180.0, p.width1_nm), CPort("o2", (p.length_nm, 0.0), 0.0, p.width2_nm)]
    return c


# --- splitters and couplers -----------------------------------------------------------------


class MMIParams(CellParams):
    width_nm: float = Field(W0, gt=0, description="Access waveguide width.")
    length_taper_nm: float = Field(2000.0, gt=0)


class MMI1x2Params(MMIParams):
    width_taper_nm: float = Field(1000.0, gt=0, description="Taper width at the MMI body.")
    width_mmi_nm: float = Field(2500.0, gt=0)
    gap_mmi_nm: float = Field(250.0, ge=0, description="Gap between the output tapers (outputs at +-W/4).")
    length_mmi_nm: float = Field(5400.0, gt=0, description="Default: 50:50 with >98% transmission at 1550 nm "
                                 "(2D EIM, generic_soi; Soldano's paraxial estimate is 5980 nm).")


class MMI2x2Params(MMIParams):
    width_taper_nm: float = Field(800.0, gt=0, description="Taper width at the MMI body.")
    width_mmi_nm: float = Field(3000.0, gt=0)
    gap_mmi_nm: float = Field(200.0, ge=0, description="Gap between the tapers (ports at +-W/6: paired "
                              "interference).")
    length_mmi_nm: float = Field(10_500.0, gt=0, description="Default: 3 dB, 90 degrees at 1550 nm (2D EIM, "
                                 "generic_soi; paired-interference estimate L_pi / 2 = 11400 nm).")


def _mmi(name: str, p, n_in: int) -> Component:
    c = _new(name, p)
    lt, lm, wt = p.length_taper_nm, p.length_mmi_nm, p.width_taper_nm
    d = (wt + p.gap_mmi_nm) / 2
    xe = 2 * lt + lm
    c.add_polygon("WG", rect(lt, lt + lm, -p.width_mmi_nm / 2, p.width_mmi_nm / 2))

    def tap(x0, x1, y, w0, w1):
        c.add_polygon("WG", np.array([[x0, y - w0 / 2], [x1, y - w1 / 2], [x1, y + w1 / 2], [x0, y + w0 / 2]]))

    ins = [0.0] if n_in == 1 else [-d, d]
    for y in ins:
        tap(0.0, lt, y, p.width_nm, wt)
    for y in (-d, d):
        tap(lt + lm, xe, y, wt, p.width_nm)
    w = p.width_nm
    c.sim = {"options": {"eme": {"taper_cells": 6}}}  # short tapers: 6 steps change T by ~1e-3
    if n_in == 1:
        c.ports = [CPort("o1", (0.0, 0.0), 180.0, w), CPort("o2", (xe, d), 0.0, w), CPort("o3", (xe, -d), 0.0, w)]
    else:
        c.ports = [CPort("o1", (0.0, -d), 180.0, w), CPort("o2", (0.0, d), 180.0, w),
                   CPort("o3", (xe, d), 0.0, w), CPort("o4", (xe, -d), 0.0, w)]
    return c


@cell("mmi1x2", MMI1x2Params, ("o1", "o2", "o3"))
def mmi1x2(p: MMI1x2Params) -> Component:
    """1x2 multimode-interference splitter with tapered access waveguides (centre-fed, symmetric)."""
    return _mmi("mmi1x2", p, 1)


@cell("mmi2x2", MMI2x2Params, ("o1", "o2", "o3", "o4"))
def mmi2x2(p: MMI2x2Params) -> Component:
    """2x2 multimode-interference 3 dB coupler (general interference)."""
    return _mmi("mmi2x2", p, 2)


class CouplerParams(CellParams):
    gap_nm: float = Field(200.0, ge=0, description="Edge-to-edge gap in the coupling section.")
    length_nm: float = Field(5000.0, ge=0, description="Length of the straight coupling section.")
    width_nm: float = Field(W0, gt=0)
    dx_nm: float = Field(8000.0, gt=0, description="Length of each S-bend.")
    dy_nm: float = Field(3000.0, gt=0, description="Centre-to-centre port pitch on each side.")


@cell("coupler", CouplerParams, ("o1", "o2", "o3", "o4"))
def coupler(p: CouplerParams) -> Component:
    """Directional coupler: two waveguides brought to `gap_nm` by raised-cosine S-bends over `length_nm`."""
    c = _new("coupler", p)
    c.sim = {"options": {"eme": {"taper_cells": 1}}}  # EME cuts at every vertex: one cell per polyline segment
    yc = (p.gap_nm + p.width_nm) / 2
    off = p.dy_nm / 2 - yc
    if off < 0:
        raise ValueError("dy_nm must be at least gap_nm + width_nm")
    sb = sbend_path(p.dx_nm, off)
    xe = 2 * p.dx_nm + p.length_nm
    for sgn in (-1, 1):
        left = np.stack([sb[:, 0], sgn * (p.dy_nm / 2 - sb[:, 1])], axis=1)
        mid = np.array([[p.dx_nm, sgn * yc], [p.dx_nm + p.length_nm, sgn * yc]])
        right = np.stack([xe - sb[::-1, 0], sgn * (p.dy_nm / 2 - sb[::-1, 1])], axis=1)
        path = np.concatenate([left, mid[1:] if p.length_nm > 0 else mid[:0], right[1:]], axis=0)
        c.add_polygon("WG", extrude(path, p.width_nm))
    w, h = p.width_nm, p.dy_nm / 2
    c.ports = [CPort("o1", (0.0, -h), 180.0, w), CPort("o2", (0.0, h), 180.0, w),
               CPort("o3", (xe, h), 0.0, w), CPort("o4", (xe, -h), 0.0, w)]
    return c


class YBranchParams(CellParams):
    width_nm: float = Field(W0, gt=0)
    length_taper_nm: float = Field(2000.0, gt=0, description="Stem taper from width_nm to 2 width_nm.")
    length_branch_nm: float = Field(8000.0, gt=0, description="Length of the diverging S-bend arms.")
    pitch_nm: float = Field(2000.0, gt=0, description="Centre-to-centre separation of the output ports.")


@cell("y_branch", YBranchParams, ("o1", "o2", "o3"))
def y_branch(p: YBranchParams) -> Component:
    """Y-junction: a stem taper to twice the width splitting into two raised-cosine S-bend arms."""
    c = _new("y_branch", p)
    c.sim = {"options": {"eme": {"taper_cells": 1}}}  # EME cuts at every vertex: one cell per polyline segment
    w, lt = p.width_nm, p.length_taper_nm
    c.add_polygon("WG", np.array([[0, -w / 2], [lt, -w], [lt, w], [0, w / 2]]))
    off = p.pitch_nm / 2 - w / 2
    sb = sbend_path(p.length_branch_nm, off)
    for sgn in (-1, 1):
        path = np.stack([lt + sb[:, 0], sgn * (w / 2 + sb[:, 1])], axis=1)
        c.add_polygon("WG", extrude(path, w))
    xe = lt + p.length_branch_nm
    h = p.pitch_nm / 2
    c.ports = [CPort("o1", (0.0, 0.0), 180.0, w), CPort("o2", (xe, h), 0.0, w), CPort("o3", (xe, -h), 0.0, w)]
    return c


class CrossingParams(CellParams):
    width_nm: float = Field(W0, gt=0)
    width_center_nm: float = Field(1200.0, gt=0, description="Width of each arm at the centre.")
    length_taper_nm: float = Field(500.0, gt=0, description="Taper length of each arm.")
    length_center_nm: float | None = Field(4500.0, gt=0, description="Length of the flat centre section of each "
                                           "arm, a multimode section that re-images the input across the "
                                           "intersection; None = width_center_nm (plain tapered crossing). "
                                           "Defaults: ~91% through, < -30 dB crosstalk at 1550 nm (2D EIM).")


@cell("crossing", CrossingParams, ("o1", "o2", "o3", "o4"))
def crossing(p: CrossingParams) -> Component:
    """Waveguide crossing: two perpendicular arms, each tapered up to width_center_nm and flat over
    length_center_nm around the intersection."""
    c = _new("crossing", p)
    w, wc, lt = p.width_nm, p.width_center_nm, p.length_taper_nm
    h = max(p.length_center_nm or wc, wc) / 2
    a = lt + h
    arm = np.array([[-a, -w / 2], [-h, -wc / 2], [h, -wc / 2], [a, -w / 2],
                    [a, w / 2], [h, wc / 2], [-h, wc / 2], [-a, w / 2]])
    c.add_polygon("WG", arm + [a, 0])
    c.add_polygon("WG", arm[:, ::-1] + [a, 0])
    c.ports = [CPort("o1", (0.0, 0.0), 180.0, w), CPort("o2", (a, a), 90.0, w),
               CPort("o3", (2 * a, 0.0), 0.0, w), CPort("o4", (a, -a), 270.0, w)]
    return c


# --- rings -------------------------------------------------------------------------------


class CouplerRingParams(CellParams):
    radius_nm: float = Field(5000.0, gt=0)
    gap_nm: float = Field(200.0, ge=0)
    width_nm: float = Field(W0, gt=0)
    length_x_nm: float = Field(0.0, ge=0, description="Straight coupling length (racetrack); 0 = ring.")
    bus_extension_nm: float = Field(1000.0, ge=0, description="Bus length beyond the ring on each side.")


@cell("coupler_ring", CouplerRingParams, ("o1", "o2", "o3", "o4"))
def coupler_ring(p: CouplerRingParams) -> Component:
    """Bus waveguide and the lower half of a ring/racetrack; o2 and o3 face up into the rest of the ring."""
    c = _new("coupler_ring", p)
    r, lx, w = p.radius_nm, p.length_x_nm, p.width_nm
    yc = p.gap_nm + w  # ring waveguide centre line height above the bus centre
    x1 = r + lx / 2 + p.bus_extension_nm
    c.add_polygon("WG", rect(0, 2 * x1, -w / 2, w / 2))
    q = arc_path(r, 90.0)  # from (0,0) heading +x to (r, r)
    right = np.stack([x1 + lx / 2 + q[:, 0], yc + q[:, 1]], axis=1)
    left = np.stack([x1 - lx / 2 - q[:, 0], yc + q[:, 1]], axis=1)
    if lx > 0:
        c.add_polygon("WG", rect(x1 - lx / 2, x1 + lx / 2, yc - w / 2, yc + w / 2))
    c.add_polygon("WG", extrude(right, w))
    c.add_polygon("WG", extrude(left[::-1], w))
    top = yc + r
    c.ports = [CPort("o1", (0.0, 0.0), 180.0, w), CPort("o2", (x1 - lx / 2 - r, top), 90.0, w),
               CPort("o3", (x1 + lx / 2 + r, top), 90.0, w), CPort("o4", (2 * x1, 0.0), 0.0, w)]
    return c


class RingParams(CellParams):
    radius_nm: float = Field(5000.0, gt=0)
    gap_nm: float = Field(200.0, ge=0)
    width_nm: float = Field(W0, gt=0)
    length_x_nm: float = Field(0.0, ge=0, description="Straight sections (racetrack) at bottom and top.")


def _ring(kind: str, p: RingParams) -> Component:
    asm = Assembly()
    cr = get_component("coupler_ring", radius_nm=p.radius_nm, gap_nm=p.gap_nm, width_nm=p.width_nm,
                       length_x_nm=p.length_x_nm)
    asm.add("coupler", cr)
    b = get_component("bend_circular", radius_nm=p.radius_nm, angle_deg=90.0, width_nm=p.width_nm)
    asm.connect("bend_right", b, "o1", "coupler,o3")
    last = "bend_right,o2"
    if p.length_x_nm > 0:
        asm.connect("top", get_component("straight", length_nm=p.length_x_nm, width_nm=p.width_nm), "o1", last)
        last = "top,o2"
    asm.connect("bend_left", b, "o1", last)
    asm.link("bend_left,o2", "coupler,o2")
    asm.expose("o1", "coupler,o1")
    asm.expose("o2", "coupler,o4")
    return asm.build(kind, p.model_dump())


@cell("ring_single", RingParams, ("o1", "o2"))
def ring_single(p: RingParams) -> Component:
    """All-pass ring (or racetrack with length_x_nm > 0): coupler_ring closed by two 90 degree bends.
    Its circuit model is the netlist of the simulated coupler and bends."""
    return _ring("ring_single", p)


class RacetrackParams(RingParams):
    length_x_nm: float = Field(5000.0, gt=0, description="Straight sections at bottom and top.")


@cell("racetrack_single", RacetrackParams, ("o1", "o2"))
def racetrack_single(p: RacetrackParams) -> Component:
    """All-pass racetrack resonator (ring_single with straight coupling sections)."""
    return _ring("racetrack_single", p)


# --- grating coupler (simulated in a 2D side view) -----------------------------------------


class GratingCouplerParams(CellParams):
    period_nm: float = Field(580.0, gt=0)
    fill_factor: float = Field(0.5, gt=0, lt=1, description="Unetched fraction of each period.")
    num_periods: int = Field(20, ge=1)
    width_nm: float = Field(W0, gt=0, description="Access waveguide width.")
    width_grating_nm: float = Field(12_000.0, gt=0)
    length_taper_nm: float = Field(15_000.0, gt=0)
    fiber_mfd_nm: float = Field(10_400.0, gt=0, description="Mode-field diameter of the (single-mode) fiber.")
    fiber_index_contrast: float = Field(0.0064, gt=0, description="Fiber core minus cladding index.")
    fiber_offset_nm: float = Field(5000.0, description="Fiber axis position from the grating start.")
    fiber_gap_nm: float = Field(1000.0, gt=0, description="Height of the fiber facet above the top of the core.")


@cell("grating_coupler", GratingCouplerParams, ("o1", "o2"))
def grating_coupler(p: GratingCouplerParams) -> Component:
    """Shallow-etched uniform 1D grating coupler with a linear taper, for a vertical fiber.

    Simulated in a 2D side view (x along the grating, y vertical, E normal to the plane): the waveguide
    port o1 is the slab mode of the device layer and the fiber port o2 is the fundamental mode of a
    vertical slab 'fiber' of mode-field diameter fiber_mfd_nm. The lateral taper is assumed adiabatic and
    is not simulated."""
    c = _new("grating_coupler", p)
    lt, wg = p.length_taper_nm, p.width_grating_nm
    lg = p.num_periods * p.period_nm
    c.add_polygon("WG", np.array([[0, -p.width_nm / 2], [lt, -wg / 2], [lt, wg / 2], [0, p.width_nm / 2]]))
    c.add_polygon("WG", rect(lt, lt + lg + 1000.0, -wg / 2, wg / 2))
    for k in range(p.num_periods):
        x0 = lt + k * p.period_nm + p.fill_factor * p.period_nm
        c.add_polygon("SHALLOW_ETCH", rect(x0, lt + (k + 1) * p.period_nm, -wg / 2, wg / 2))
    c.ports = [CPort("o1", (0.0, 0.0), 180.0, p.width_nm),
               CPort("o2", (lt + p.fiber_offset_nm, 0.0), 90.0, p.fiber_mfd_nm)]
    c.sim = {"builder": _grating_side_view, "view": "side", "port_neff": _grating_port_neff}
    return c


def _grating_port_neff(comp: Component, tech, name: str, wl: np.ndarray) -> np.ndarray:
    """Effective indices of the side-view port modes (to move FDFD's grid-snapped planes)."""
    from ..backends._fdfd.modes import slab_neff_analytic

    p = GratingCouplerParams.model_validate(comp.settings)
    if name == "o1":
        return tech.slab_neff(tech.core_thickness_nm(), wl)
    n_clad = np.real(tech.cladding.n(wl))
    core_w = _fiber_core_width(p.fiber_mfd_nm, float(np.real(tech.cladding.n(np.array([1550.0])))[0]),
                               p.fiber_index_contrast)
    return np.array([slab_neff_analytic(n + p.fiber_index_contrast, n, core_w, w, "tm") for n, w in zip(n_clad, wl)])


def _grating_side_view(comp: Component, tech, wavelengths, *, dimension=2, options=None):
    """2D side-view DeviceProblem of a grating coupler: device x = along the grating, device y = height z."""
    from ..materials import Tabulated
    from ..problems import Box, DeviceProblem, Port, Structure, wavelength_array
    from .component import _merge

    if dimension != 2:
        raise ValueError("grating_coupler is simulated in a 2D side view only (dimension=2)")
    p = GratingCouplerParams.model_validate(comp.settings)
    wl = wavelength_array(wavelengths)
    core = tech.core
    etch = next((ld for ld in tech.layers.values() if ld.kind == "etch"), None)
    if etch is None:
        raise ValueError(f"technology {tech.name!r} has no etch layer for grating teeth")
    z0, z1 = core.z_min_nm, core.z_max_nm
    lg = p.num_periods * p.period_nm
    x_end = lg + 1000.0  # grating region starts at x = 0 in the side view
    xf = p.fiber_offset_nm
    big = 40_000.0
    grid = np.linspace(wl.min() - 4, wl.max() + 4, max(int(np.ptp(wl) / 2) + 5, 3))
    fiber = Tabulated(wavelength_nm=grid.tolist(), n_real=(np.real(tech.cladding.n(grid)) + p.fiber_index_contrast)
                      .tolist(), n_imag=[0.0] * len(grid), source="slab fiber core (cladding + contrast)")
    core_w = _fiber_core_width(p.fiber_mfd_nm, float(np.real(tech.cladding.n(np.array([1550.0])))[0]),
                               p.fiber_index_contrast)
    y_fiber = z1 + p.fiber_gap_nm
    st = []
    if tech.substrate is not None:
        st.append(Structure(name="substrate", material=tech.substrate, geometry=Box(
            center_nm=(0, -tech.box_thickness_nm - big / 2, 0), size_nm=(4 * big, big, 1))))
    st.append(Structure(name="box", material=tech.box, geometry=Box(
        center_nm=(0, -tech.box_thickness_nm / 2, 0), size_nm=(4 * big, tech.box_thickness_nm, 1))))
    st.append(Structure(name="core", material=core.material, geometry=Box(
        center_nm=((x_end - big) / 2, (z0 + z1) / 2, 0), size_nm=(x_end + big, z1 - z0, 1))))
    for k in range(p.num_periods):
        a = k * p.period_nm + p.fill_factor * p.period_nm
        b = (k + 1) * p.period_nm
        lo, hi = max(etch.z_min_nm, z0), min(etch.z_max_nm, z1)
        st.append(Structure(name=f"tooth#{k}", material=tech.cladding, geometry=Box(
            center_nm=((a + b) / 2, (lo + hi) / 2, 0), size_nm=(b - a, hi - lo, 1))))
    st.append(Structure(name="fiber", material=fiber, geometry=Box(
        center_nm=(xf, y_fiber + big / 2, 0), size_nm=(core_w, big, 1))))
    y_port = y_fiber + 500.0
    half = 1.1 * p.fiber_mfd_nm  # fiber port window half width (field ~1e-2 at its edges)
    x_lo = min(-1000.0, xf - half - 300.0)
    x_hi = max(x_end + 1000.0, xf + half + 300.0)
    y_lo = -tech.box_thickness_nm - 400.0 if tech.substrate is not None else -tech.box_thickness_nm + 500.0
    y_hi = y_port + 600.0
    ports = [Port(name="o1", center_nm=(-500.0, (z0 + z1) / 2, 0), size_nm=(0, 3000.0, 1), direction="+"),
             Port(name="o2", center_nm=(xf, y_port, 0), size_nm=(2 * half, 0, 1), direction="-")]
    opts = _merge({"fdfd": {"polarization": "tm" if tech.polarization == "te" else "te", "dl_nm": 20.0}},
                  options)
    return DeviceProblem(dimension=2, structures=st, background=tech.cladding,
                         domain_center_nm=((x_lo + x_hi) / 2, (y_lo + y_hi) / 2, 0),
                         domain_size_nm=(x_hi - x_lo, y_hi - y_lo, 0), ports=ports, wavelengths=wl.tolist(),
                         options=opts)


def _fiber_core_width(mfd_nm: float, n_clad: float, dn: float, wavelength_nm: float = 1550.0) -> float:
    """Slab core width whose fundamental mode has a 1/e^2 intensity full width close to mfd_nm.

    Marcuse's Gaussian-fit estimate for a step-index fiber, w/a = 0.65 + 1.619 V^-1.5 + 2.879 V^-6,
    applied with the fiber's V number; solved for the core radius a by bisection."""
    na = math.sqrt((n_clad + dn) ** 2 - n_clad**2)
    k0 = 2 * math.pi / wavelength_nm
    lo, hi = 500.0, 20_000.0
    for _ in range(80):
        a = (lo + hi) / 2
        v = k0 * a * na
        w = a * (0.65 + 1.619 * v**-1.5 + 2.879 * v**-6)
        lo, hi = (a, hi) if 2 * w < mfd_nm else (lo, a)
    return 2 * (lo + hi) / 2


# --- the worked MZI ---------------------------------------------------------------------------


class MZIParams(CellParams):
    delta_length_nm: float = Field(10_000.0, ge=0, description="Total length of the two vertical straights in "
                                   "the delay loop.")
    radius_nm: float = Field(3000.0, gt=0, description="Radius of the four 90 degree bends of the delay loop.")
    length_x_nm: float = Field(0.0, ge=0, description="Straight at the bottom of the delay loop (the reference "
                               "arm gets the same length).")
    width_nm: float = Field(W0, gt=0)
    splitter: str = Field("mmi1x2", description="1x2 splitter cell used as splitter and (rotated) combiner.")
    splitter_settings: dict[str, Any] = Field(default_factory=dict)


@cell("mzi", MZIParams, ("o1", "o2"))
def mzi(p: MZIParams) -> Component:
    """Unbalanced Mach-Zehnder interferometer: 1x2 splitter, straight reference arm, delay loop (four 90 degree
    bends, two vertical straights of delta_length_nm / 2), 1x2 combiner. The arm length difference is
    delta_length_nm + (2 pi - 4) radius_nm (see :func:`mzi_arm_lengths`)."""
    asm = Assembly()
    sp = get_component(p.splitter, **{"width_nm": p.width_nm, **p.splitter_settings})
    asm.add("split", sp)
    w = p.width_nm
    bend = get_component("bend_circular", radius_nm=p.radius_nm, angle_deg=90.0, width_nm=w)
    vs = p.delta_length_nm / 2

    def straight_(name, length, last):
        if length <= 0:
            return last
        asm.connect(name, get_component("straight", length_nm=length, width_nm=w), "o1", last)
        return f"{name},o2"

    # Delay loop from the bottom output: right turn down, left turn east, left turn up, right turn east.
    asm.connect("b1", bend, "o2", "split,o3")
    last = straight_("v1", vs, "b1,o1")
    asm.connect("b2", bend, "o1", last)
    last = straight_("hx", p.length_x_nm, "b2,o2")
    asm.connect("b3", bend, "o1", last)
    last = straight_("v2", vs, "b3,o2")
    asm.connect("b4", bend, "o2", last)
    asm.connect("combine", sp, "o2", "b4,o1")  # rotated by 180 degrees: its o3 is now on top
    a, b = asm.port("split,o2"), asm.port("combine,o3")
    if abs(b.center[1] - a.center[1]) > 1e-6:
        raise ValueError("splitter outputs are not level; use a symmetric splitter")
    asm.connect("ref", get_component("straight", length_nm=float(b.center[0] - a.center[0]), width_nm=w), "o1",
                "split,o2")
    asm.link("ref,o2", "combine,o3")
    asm.expose("o1", "split,o1")
    asm.expose("o2", "combine,o1")
    return asm.build("mzi", p.model_dump())


def mzi_arm_lengths(p: MZIParams | dict) -> tuple[float, float]:
    """(reference arm, delay arm) centre-line lengths between the splitter and combiner ports."""
    p = MZIParams.model_validate(p) if isinstance(p, dict) else p
    r = p.radius_nm
    ref = 4 * r + p.length_x_nm
    delay = 2 * math.pi * r + p.length_x_nm + p.delta_length_nm
    return ref, delay
