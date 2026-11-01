"""Component -> simulation -> fitted compact model -> circuit netlists.

``simulate(component)`` builds the component's :class:`DeviceProblem` and solves
it with :func:`lightsmith.solve` (``method="auto"`` walks ``AUTO_ORDER``, so EME,
FDFD or any later device backend is chosen by what the device needs), then
moves FDFD's grid-snapped port planes back onto the component's reference
planes. Straight waveguides use a mode solve instead, and composite components
(rings, the MZI) evaluate the circuit of their parts.

``component_model(...)`` fits the sampled S-parameters (:mod:`.fitting`) and
caches the result on disk under a hash of everything that determines it
(cell, parameters, technology, wavelengths, solver choice and options).

In a circuit netlist, ``{"component": "pdk:mmi1x2", "settings": {"length_mmi_nm": 4300}}``
instantiates that model; the settings are the cell's parameters plus
:class:`FlowSettings` (technology, simulated band, solver, fit). Missing models
are simulated on first use.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from ..circuits.models import CompactModel, Ctx, ModelSettings
from ..problems import CircuitProblem, Instance, Wavelengths, wavelength_array
from ..results import SParameterResult
from .cells import CELLS, cell_params, get_component
from .component import Component
from .fitting import SFit, fit_sparameters
from .technology import Technology, get_technology

SCHEMA = "lightsmith-pdk-model-1"


# --- simulation -------------------------------------------------------------------------


def _port_neff(comp: Component, tech: Technology, name: str, wl: np.ndarray, dimension: int) -> np.ndarray:
    custom = comp.sim.get("port_neff")
    if custom is not None:
        return np.asarray(custom(comp, tech, name, wl), dtype=float)
    p = comp.port(name)
    return tech.waveguide_neff(wl, p.width_nm, 2, layer=p.layer) if dimension == 2 else \
        tech.waveguide_neff(wl, p.width_nm, 3, layer=p.layer)


def _fdfd_port_neff(problem, wl: np.ndarray) -> dict[str, np.ndarray]:
    """Port-mode effective indices on FDFD's own grid (its numerical dispersion included)."""
    from ..backends.fdfd import FDFD

    eng = FDFD.engine(problem)
    out = {pg.name: np.zeros(len(wl)) for pg in eng.ports}
    for i, lam in enumerate(wl):
        for pg, ms in zip(eng.ports, eng.modes(eng.base_eps(float(lam)), float(lam))):
            out[pg.name][i] = ms[0].neff.real
    return out


def _move_reference_planes(res: SParameterResult, problem, comp: Component, tech: Technology,
                           dimension: int) -> SParameterResult:
    """Shift S from the solver's port planes to the component's port centres.

    The solver planes are the problem's port centres, or the grid lines a solver reports in
    ``provenance["port_planes_nm"]`` (FDFD). Components with their own problem builder (side views)
    have no layout reference planes: only the solver's snapping is undone. FDFD results are moved with
    FDFD's own port-mode indices, so its numerical dispersion does not leak into the phase."""
    planes = res.provenance.get("port_planes_nm")
    planes = planes if isinstance(planes, dict) else {}
    layout_planes = comp.sim.get("builder") is None
    wl = np.asarray(res.wavelengths_nm)
    solver_neff = _fdfd_port_neff(problem, wl) if res.backend == "lightsmith-fdfd" else {}
    shift = np.ones((len(wl), len(res.ports)), complex)
    moved = {}
    for k, label in enumerate(res.ports):
        name = label.split("@")[0]
        port = next(p for p in problem.ports if p.name == name)
        ax = port.normal_axis
        d = 1.0 if port.direction == "+" else -1.0
        solver = planes.get(name, port.center_nm[ax])
        target = comp.port(name).center[ax] if layout_planes else port.center_nm[ax]
        inward = (solver - target) * d  # > 0: the solver plane lies inside the reference plane
        if abs(inward) < 1e-9:
            continue
        neff = solver_neff[name] if name in solver_neff else _port_neff(comp, tech, name, wl, dimension)
        shift[:, k] = np.exp(1j * 2 * np.pi * neff * inward / wl)
        moved[name] = float(-inward)
    if not moved:
        return res
    s = res.s() * shift[:, :, None] * shift[:, None, :]
    prov = {**res.provenance, "reference_plane_shift_nm": moved}
    return res.model_copy(update={"s_real": s.real.tolist(), "s_imag": s.imag.tolist(), "provenance": prov})


def simulate(component: Component | str, tech: Technology | str | dict | None = None,
             wavelengths: Wavelengths | None = None, *, method: str = "auto", device: str = "auto",
             dimension: int = 2, options: dict[str, dict[str, Any]] | None = None, direct: bool = False,
             **settings) -> SParameterResult:
    """S-parameters of a component at its port reference planes.

    `component` is a :class:`Component` or a cell name (with `settings`). By default straights use a mode
    solve and composites the circuit of their parts; ``direct=True`` solves the whole layout as one device.
    ``method``/``device``/``options`` go to :func:`lightsmith.solve` (``options`` per method, e.g.
    ``{"fdfd": {"dl_nm": 25}}``)."""
    from ..api import solve

    comp = get_component(component, **settings) if isinstance(component, str) else component
    tech = get_technology(tech)
    wl = wavelength_array(wavelengths if wavelengths is not None else [1550.0])
    t0 = time.perf_counter()
    if comp.model == "waveguide" and not direct:
        res = _waveguide_result(comp, tech, wl, dimension)
    elif comp.model == "netlist" and not direct:
        res = _netlist_result(comp, wl, FlowSettings(technology=tech.name if _is_registered(tech) else
                                                     tech.model_dump(mode="json"), method=method, device=device,
                                                     dimension=dimension, sim_options=options or {},
                                                     wl_min_nm=float(wl.min()), wl_max_nm=float(wl.max())))
    else:
        problem = comp.device_problem(tech, wl.tolist(), dimension=dimension, options=options)
        res = solve(problem, method=method, device=device)
        res = _move_reference_planes(res, problem, comp, tech, dimension)
    order = [p.name for p in comp.ports]
    if res.ports != order and sorted(res.ports) == sorted(order):
        idx = [res.ports.index(p) for p in order]
        s = res.s()[:, idx][:, :, idx]
        res = res.model_copy(update={"ports": order, "s_real": s.real.tolist(), "s_imag": s.imag.tolist()})
    prov = {**res.provenance, "component": comp.kind, "settings": comp.settings, "technology": tech.name,
            "dimension": dimension, "flow_runtime_s": time.perf_counter() - t0}
    return res.model_copy(update={"provenance": prov})


def _is_registered(tech: Technology) -> bool:
    try:
        return get_technology(tech.name) == tech
    except KeyError:
        return False


def _waveguide_result(comp: Component, tech: Technology, wl: np.ndarray, dimension: int) -> SParameterResult:
    p_in, p_out = comp.ports
    length = float(np.hypot(*(np.asarray(p_out.center) - p_in.center)))
    neff = tech.waveguide_neff(wl, p_in.width_nm, dimension, layer=p_in.layer)
    t = np.exp(1j * 2 * np.pi * neff * length / wl)
    s = np.zeros((len(wl), 2, 2), complex)
    s[:, 1, 0] = s[:, 0, 1] = t
    return SParameterResult(method="fde" if dimension == 3 else "slab", backend="lightsmith-pdk-waveguide",
                            wavelengths_nm=wl.tolist(), ports=[p_in.name, p_out.name], s_real=s.real.tolist(),
                            s_imag=s.imag.tolist(), provenance={"neff": neff.tolist(), "length_nm": length,
                                                                "mode_solver": "fde" if dimension == 3 else
                                                                "analytic slab (2D effective index)"})


def _netlist_result(comp: Component, wl: np.ndarray, flow: "FlowSettings") -> SParameterResult:
    from ..circuits.engine import circuit_sparameters

    prob = netlist_problem(comp, wl.tolist(), flow)
    sol = circuit_sparameters(prob, device="cpu", wavelengths_nm=wl)
    s = sol.numpy()
    return SParameterResult(method="circuit", backend="lightsmith-circuit", wavelengths_nm=wl.tolist(),
                            ports=list(sol.ports), s_real=s.real.tolist(), s_imag=s.imag.tolist(),
                            provenance={"instances": len(prob.instances)})


# --- flow settings and the model cache -------------------------------------------------------


class FlowSettings(ModelSettings):
    """How a ``pdk:<cell>`` circuit model is obtained (the rest of its settings are the cell parameters)."""

    technology: str | dict[str, Any] = Field("generic_soi", description="Registered technology name or its JSON.")
    wl_min_nm: float = Field(1500.0, gt=0, description="Simulated band (the circuit must stay inside it).")
    wl_max_nm: float = Field(1600.0, gt=0)
    num_points: int = Field(11, ge=1, description="Simulated wavelengths across the band.")
    method: str = Field("auto", description="Device solver (auto = first applicable in AUTO_ORDER['device']).")
    device: str = Field("auto", description="Where to simulate (not part of the cache key).")
    dimension: Literal[2, 3] = Field(2, description="2 = plan view with effective indices; 3 = full 3D.")
    waveguide_model: Literal["mode", "device"] = Field(
        "mode", description="Straights: S from a mode solve (exact phase, any length), or a device solve with "
        "`method` (same discretisation as a direct simulation of a larger layout).")
    fit: Literal["auto", "pole_residue", "table"] = "auto"
    fit_tol: float = Field(5e-3, gt=0, description="Max |S_fit - S| on held-out samples for a pole-residue fit.")
    sim_options: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-method solver options.")
    cache: bool = Field(True, description="Read/write the on-disk model cache.")

    def wavelengths(self) -> list[float]:
        if self.num_points == 1 or self.wl_max_nm == self.wl_min_nm:
            return [float(self.wl_min_nm)]
        return np.linspace(self.wl_min_nm, self.wl_max_nm, self.num_points).tolist()


FLOW_FIELDS = tuple(FlowSettings.model_fields)


def cache_dir() -> Path:
    """Model cache directory: ``LIGHTSMITH_PDK_CACHE``, else ``<lightsmith cache>/pdk``."""
    if os.environ.get("LIGHTSMITH_PDK_CACHE"):
        return Path(os.environ["LIGHTSMITH_PDK_CACHE"])
    from ..materials import default_cache_dir

    return default_cache_dir() / "pdk"


def model_key(kind: str, settings: dict[str, Any], tech: Technology, flow: FlowSettings) -> str:
    from .. import __version__

    payload = {"schema": SCHEMA, "version": __version__, "cell": kind, "settings": settings,
               "technology": json.loads(tech.fingerprint()), "wavelengths": [round(w, 9) for w in flow.wavelengths()],
               "method": flow.method, "dimension": flow.dimension, "fit": flow.fit, "fit_tol": flow.fit_tol,
               "waveguide_model": flow.waveguide_model if kind == "straight" else None,
               "options": flow.sim_options}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class FittedModel:
    """A cached compact model: sampled S-parameters, the fit, and where they came from."""

    def __init__(self, kind: str, settings: dict, technology: str, key: str, ports: list[str],
                 result: SParameterResult, fit: SFit):
        self.kind, self.settings, self.technology, self.key = kind, settings, technology, key
        self.ports, self.result, self.fit = ports, result, fit

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return np.asarray(self.result.wavelengths_nm)

    def evaluate(self, wavelengths_nm) -> np.ndarray:
        wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
        lo, hi = self.wavelengths_nm.min(), self.wavelengths_nm.max()
        if len(self.wavelengths_nm) > 1 and (wl.min() < lo - 1e-6 or wl.max() > hi + 1e-6):
            raise ValueError(f"pdk:{self.kind} was simulated for {lo:g}-{hi:g} nm but is evaluated at "
                             f"{wl.min():g}-{wl.max():g} nm; widen wl_min_nm/wl_max_nm in its settings")
        if len(self.wavelengths_nm) == 1:
            return np.repeat(self.result.s(), len(wl), axis=0)
        neff = self.neff_poly
        if neff is not None:  # straight from a mode solve: exact phase for any length
            t = np.exp(2j * np.pi * np.polyval(neff, wl) * self.result.provenance["length_nm"] / wl)
            s = np.zeros((len(wl), 2, 2), complex)
            s[:, 1, 0] = s[:, 0, 1] = t
            return s
        return self.fit.evaluate(wl)

    @property
    def neff_poly(self) -> np.ndarray | None:
        """Polynomial neff(lambda) (degree <= 3) for mode-solved straights, else None."""
        prov = self.result.provenance
        if "neff" not in prov or "length_nm" not in prov or len(self.wavelengths_nm) < 2:
            return None
        deg = min(3, len(self.wavelengths_nm) - 1)
        return np.polyfit(self.wavelengths_nm, np.asarray(prov["neff"]), deg)

    def to_json(self) -> str:
        return json.dumps({"schema": SCHEMA, "cell": self.kind, "settings": self.settings,
                           "technology": self.technology, "key": self.key, "ports": self.ports,
                           "result": self.result.model_dump(mode="json"), "fit": self.fit.to_dict()})

    @classmethod
    def from_json(cls, text: str) -> "FittedModel":
        d = json.loads(text)
        if d.get("schema") != SCHEMA:
            raise ValueError("model file has another schema")
        return cls(d["cell"], d["settings"], d["technology"], d["key"], d["ports"],
                   SParameterResult.model_validate(d["result"]), SFit.from_dict(d["fit"]))

    def summary(self) -> dict[str, Any]:
        return {"cell": self.kind, "technology": self.technology, "fit": self.fit.kind,
                "poles": 0 if self.fit.kind == "neff_polynomial" else int(len(self.fit.poles)), "fit_error": self.fit.fit_error,
                "validation_error": self.fit.validation_error, "backend": self.result.backend,
                "wavelengths_nm": [float(self.wavelengths_nm.min()), float(self.wavelengths_nm.max()),
                                   len(self.wavelengths_nm)], "warnings": self.result.warnings}


_MEMORY: dict[str, FittedModel] = {}


def component_model(cell: str | Component, flow: FlowSettings | dict | None = None, *,
                    refresh: bool = False, **settings) -> FittedModel:
    """The fitted compact model of a cell (simulating it if no cached model matches).

    ``component_model("mmi1x2", length_mmi_nm=4300)`` or ``component_model(comp, flow={"num_points": 21})``."""
    flow = flow if isinstance(flow, FlowSettings) else FlowSettings.model_validate(flow or {})
    if isinstance(cell, Component):
        comp = cell
    else:
        comp = get_component(cell, **settings)
    kind, params = comp.kind, comp.settings
    tech = get_technology(flow.technology)
    key = model_key(kind, params, tech, flow)
    if not refresh:
        if key in _MEMORY:
            return _MEMORY[key]
        path = cache_dir() / f"{kind}-{key[:20]}.json"
        if flow.cache and path.is_file():
            try:
                m = FittedModel.from_json(path.read_text(encoding="utf-8"))
                if m.key == key:
                    _MEMORY[key] = m
                    return m
            except (ValueError, KeyError):
                pass
    res = simulate(comp, tech, flow.wavelengths(), method=flow.method, device=flow.device,
                   dimension=flow.dimension, options=flow.sim_options,
                   direct=comp.model == "waveguide" and flow.waveguide_model == "device")
    s = res.s()
    fit = fit_sparameters(res.wavelengths_nm, s, flow.fit, flow.fit_tol) if len(res.wavelengths_nm) > 1 else \
        SFit("table", 1 / res.wavelengths_nm[0], 1.0, np.zeros(s.shape[1:]), x_table=np.zeros(1), s_table=s)
    m = FittedModel(kind, params, tech.name, key, list(res.ports), res, fit)
    if m.neff_poly is not None:
        fit.kind, fit.fit_error = "neff_polynomial", float(np.abs(m.evaluate(res.wavelengths_nm) - s).max())
        fit.validation_error = None
    _MEMORY[key] = m
    if flow.cache:
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{kind}-{key[:20]}.json.tmp{os.getpid()}"
        tmp.write_text(m.to_json(), encoding="utf-8")
        tmp.replace(d / f"{kind}-{key[:20]}.json")
    return m


def clear_memory_cache() -> None:
    _MEMORY.clear()


# --- circuits ----------------------------------------------------------------------------


def split_settings(values: dict[str, Any]) -> tuple[dict[str, Any], FlowSettings]:
    flow = {k: v for k, v in values.items() if k in FLOW_FIELDS}
    params = {k: v for k, v in values.items() if k not in FLOW_FIELDS}
    return params, FlowSettings.model_validate(flow)


def netlist_problem(comp: Component, wavelengths: Wavelengths, flow: FlowSettings | dict | None = None,
                    port_names: dict[str, str] | None = None) -> CircuitProblem:
    """The circuit of a composite component's parts, every ``pdk:`` part using `flow`."""
    if comp.netlist is None:
        raise ValueError(f"{comp.kind} is not a composite component (no netlist)")
    flow = flow if isinstance(flow, FlowSettings) else FlowSettings.model_validate(flow or {})
    fl = flow.model_dump(exclude_defaults=True)
    instances = {}
    for name, inst in comp.netlist["instances"].items():
        settings = dict(inst["settings"])
        if inst["component"].startswith("pdk:"):
            settings.update(fl)
        instances[name] = Instance(component=inst["component"], settings=settings)
    ports = {(port_names or {}).get(k, k): v for k, v in comp.netlist["ports"].items()}
    return CircuitProblem(instances=instances, connections=[tuple(c) for c in comp.netlist["connections"]],
                          ports=ports, wavelengths=list(map(float, wavelength_array(wavelengths))))


def instance(cell: str, flow: FlowSettings | dict | None = None, **settings) -> Instance:
    """A netlist instance ``pdk:<cell>`` with validated cell parameters and flow settings."""
    params = cell_params(cell, settings).model_dump(exclude_defaults=True)
    fl = (flow if isinstance(flow, FlowSettings) else FlowSettings.model_validate(flow or {}))
    return Instance(component=f"pdk:{cell}", settings={**params, **fl.model_dump(exclude_defaults=True)})


_MODEL_CACHE: dict[str, CompactModel] = {}


def pdk_model(cell: str) -> CompactModel:
    """The circuit :class:`CompactModel` behind ``pdk:<cell>`` (used by ``circuits.models.get_model``)."""
    if cell in _MODEL_CACHE:
        return _MODEL_CACHE[cell]
    if cell not in CELLS:
        raise KeyError(f"unknown pdk component 'pdk:{cell}'; available: "
                       + ", ".join(f"pdk:{c}" for c in sorted(CELLS)))
    spec = CELLS[cell]
    settings_cls = type(f"PDK_{cell}", (spec.params, FlowSettings), {"__doc__": spec.doc})

    def fn(ctx: Ctx, values: dict[str, Any]):
        params, flow = split_settings(values)
        comp = get_component(cell, **params)
        wl = ctx.wl_np
        if comp.model == "netlist":
            res = _netlist_result(comp, wl, flow)
            s = res.s()
        else:
            s = component_model(comp, flow).evaluate(wl)
        return [p.name for p in comp.ports], ctx.device.asarray(s, dtype=ctx.cdt)

    m = CompactModel(f"pdk:{cell}", settings_cls, fn, "Simulated + fitted PDK component. " + spec.doc,
                     spec.ports)
    _MODEL_CACHE[cell] = m
    return m


def fsr_estimate(tech: Technology, length_nm: float, wavelength_nm: float = 1550.0, width_nm: float | None = None,
                 dimension: int = 2) -> float:
    """lambda^2 / (ng L) with the technology's group index."""
    ng = float(tech.group_index([wavelength_nm], width_nm, dimension)[0])
    return wavelength_nm**2 / (ng * length_nm)


__all__ = ["FittedModel", "FlowSettings", "cache_dir", "component_model", "fsr_estimate", "instance",
           "netlist_problem", "pdk_model", "simulate"]

