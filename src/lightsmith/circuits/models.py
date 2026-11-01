"""Compact S-parameter models for photonic circuits.

Every model is registered by name with a pydantic settings class, so a
``CircuitProblem`` instance such as ``{"component": "waveguide", "settings":
{"length_nm": 1e5}}`` is validated before anything runs. :func:`list_models`
returns each model's ports, documentation and settings JSON schema.

Conventions (shared with the rest of lightsmith):

- lengths in nm, vacuum wavelengths in nm, losses in dB (or dB/cm);
- time dependence exp(-i omega t), so forward propagation over L is exp(+i 2 pi neff L / lambda);
- a symmetric lossless coupler has bar amplitude t = sqrt(1 - kappa^2) and cross
  amplitude i*kappa;
- port names follow gdsfactory: ``o1, o2, ...`` counter-clockwise from the
  bottom-left (west) port. Two-port: o1 -> o2. 2x2: o1, o2 west (bottom, top),
  o3, o4 east (top, bottom); bar paths are o1<->o4 and o2<->o3. 1x2: o1 west,
  o2 east-top, o3 east-bottom.
- dispersion from the group index: neff(lambda) = neff + (neff - ng) (lambda - wl0) / wl0.

Models are written once against the Python array API (``ctx.xp``), so they
run on NumPy, or on PyTorch (CPU/CUDA) where gradients flow through any
setting passed as a tensor (see :func:`lightsmith.circuits.circuit_sparameters`).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..device import Device

LN10 = math.log(10.0)


# --- evaluation context ------------------------------------------------------


@dataclass
class Ctx:
    """Wavelength grid and array namespace a model is evaluated on."""

    wl_np: np.ndarray  # (W,) nm, NumPy copy for table interpolation
    device: Device
    single: bool = False

    def __post_init__(self):
        self.xp = self.device.namespace()
        self.rdt = self.xp.float32 if self.single else self.xp.float64
        self.cdt = self.xp.complex64 if self.single else self.xp.complex128
        self.wl = self.device.asarray(self.wl_np, dtype=self.rdt)

    @property
    def n(self) -> int:
        return int(self.wl_np.size)

    def p(self, v):
        """A setting as a (possibly differentiable) real array; tensors pass through."""
        if hasattr(v, "shape"):
            return v
        return self.device.asarray(float(v), dtype=self.rdt)

    def zeros(self):
        return self.xp.zeros((self.n,), dtype=self.cdt, device=self.wl.device)

    def c(self, v):
        """Broadcast a real/complex scalar or (W,) array to a complex (W,) array."""
        return self.zeros() + v

    def expi(self, phi):
        """exp(i phi) for real phi."""
        return self.xp.exp(1j * self.c(phi))

    def table(self, x_nm, y):
        """Interpolate tabulated data (not differentiable) onto the wavelength grid."""
        x_nm, y = np.asarray(x_nm, dtype=float), np.asarray(y, dtype=float)
        lo, hi = x_nm.min(), x_nm.max()
        if self.wl_np.min() < lo - 1e-9 or self.wl_np.max() > hi + 1e-9:
            raise ValueError(f"wavelengths {self.wl_np.min():g}-{self.wl_np.max():g} nm fall outside the "
                             f"table range {lo:g}-{hi:g} nm; no extrapolation is done")
        order = np.argsort(x_nm)
        return self.device.asarray(np.interp(self.wl_np, x_nm[order], y[order]), dtype=self.rdt)

    def smatrix(self, ports: list[str], entries: dict[tuple[str, str], Any], reciprocal: bool = True):
        """Assemble S[w, out, in] from {(out, in): (W,) array}; reciprocal fills (in, out) too."""
        idx = {p: i for i, p in enumerate(ports)}
        cells: dict[tuple[int, int], Any] = {}
        for (o, i), v in entries.items():
            cells[(idx[o], idx[i])] = self.c(v)
            if reciprocal and (idx[i], idx[o]) not in cells:
                cells[(idx[i], idx[o])] = self.c(v)
        z = self.zeros()
        rows = [self.xp.stack([cells.get((r, c), z) for c in range(len(ports))], axis=-1) for r in range(len(ports))]
        return self.xp.stack(rows, axis=-2)  # (W, P, P)


def amplitude_db(ctx: Ctx, loss_db):
    """Field amplitude 10^(-loss_db/20) as an array (loss may be a tensor or (W,) array)."""
    return ctx.xp.exp(-ctx.p(loss_db) * (LN10 / 20.0))


def effective_index(ctx: Ctx, neff, ng, wl0_nm):
    """neff(lambda) to first order from the group index: neff + (neff - ng)(lambda - wl0)/wl0."""
    neff, ng, wl0 = ctx.p(neff), ctx.p(ng), ctx.p(wl0_nm)
    return neff + (neff - ng) * (ctx.wl - wl0) / wl0


def propagation(ctx: Ctx, length_nm, neff, ng, wl0_nm, loss_db_per_cm, neff_lambda=None):
    """Complex transmission exp(i 2 pi neff L / lambda) * 10^(-loss L / 20) over (W,)."""
    L = ctx.p(length_nm)
    n = effective_index(ctx, neff, ng, wl0_nm) if neff_lambda is None else neff_lambda
    phase = 2 * math.pi * n * L / ctx.wl
    return ctx.expi(phase) * amplitude_db(ctx, ctx.p(loss_db_per_cm) * L * 1e-7)


def rolloff_db(ctx: Ctx, wl0_nm, bandwidth_1db_nm):
    """Parabolic excess loss in dB (Gaussian in power): 1 dB at wl0 +/- bandwidth/2."""
    if bandwidth_1db_nm is None:
        return ctx.p(0.0)
    x = (ctx.wl - ctx.p(wl0_nm)) / (ctx.p(bandwidth_1db_nm) / 2)
    return x * x


# --- registry -----------------------------------------------------------------


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class CompactModel:
    name: str
    settings: type[ModelSettings]
    fn: Callable[[Ctx, dict[str, Any]], tuple[list[str], Any]]
    doc: str
    ports: tuple[str, ...] | None  # None when the ports depend on settings
    aliases: tuple[str, ...] = ()

    def evaluate(self, ctx: Ctx, values: dict[str, Any]):
        return self.fn(ctx, values)

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "aliases": list(self.aliases), "ports": list(self.ports) if self.ports else None,
                "doc": self.doc, "settings_schema": self.settings.model_json_schema()}


MODELS: dict[str, CompactModel] = {}
_ALIASES: dict[str, str] = {}


def compact_model(name: str, settings: type[ModelSettings], ports: tuple[str, ...] | None,
                  aliases: tuple[str, ...] = ()):
    """Register a compact model ``fn(ctx, values) -> (ports, S[w, out, in])`` under `name`."""

    def deco(fn):
        doc = " ".join((fn.__doc__ or "").split())
        MODELS[name] = CompactModel(name, settings, fn, doc, ports, aliases)
        for a in aliases:
            _ALIASES[a] = name
        return fn

    return deco


def model_names() -> list[str]:
    return sorted(MODELS) + sorted(_ALIASES) + ["sparams:<path>", "pdk:<cell>"]


def get_model(component: str) -> CompactModel:
    """Look up a model by name or alias; ``sparams:<path>`` loads S-parameters from a file and ``pdk:<cell>``
    is a simulated, fitted component of :mod:`lightsmith.pdk`."""
    if component.startswith("sparams:"):
        return _sparams_model(component[len("sparams:"):].strip())
    if component.startswith("pdk:"):  # simulated + fitted components of lightsmith.pdk
        from ..pdk.flow import pdk_model

        return pdk_model(component[len("pdk:"):].strip())
    name = _ALIASES.get(component, component)
    if name not in MODELS:
        raise KeyError(f"unknown component model {component!r}; available: {', '.join(model_names())}")
    return MODELS[name]


def list_models() -> list[dict[str, Any]]:
    """Every registered model: name, aliases, ports, documentation and settings JSON schema."""
    out = [m.info() for m in MODELS.values()]
    out.append({"name": "sparams:<path>", "aliases": [], "ports": None,
                "doc": " ".join((_sparams_fn.__doc__ or "").split()),
                "settings_schema": SParamsSettings.model_json_schema()})
    from ..pdk.cells import CELLS

    out.append({"name": "pdk:<cell>", "aliases": [], "ports": None,
                "doc": "Simulated, fitted and cached component of lightsmith.pdk; settings are the cell parameters "
                       "plus FlowSettings (technology, band, solver, fit). Cells: " + ", ".join(sorted(CELLS)),
                "settings_schema": {}})
    return out


# --- waveguides -----------------------------------------------------------------


class _Guide(ModelSettings):
    neff: float = Field(2.4, description="Effective index at wl0_nm.")
    ng: float = Field(4.2, description="Group index at wl0_nm (sets the first-order dispersion of neff).")
    wl0_nm: float = Field(1550.0, gt=0, description="Reference wavelength for neff and ng.")
    loss_db_per_cm: float = Field(0.0, ge=0, description="Propagation loss in dB/cm.")


class WaveguideSettings(_Guide):
    length_nm: float = Field(10_000.0, ge=0, description="Physical length.")
    wavelengths_table_nm: list[float] | None = Field(
        None, description="Optional tabulated dispersion: wavelengths for neff_table (e.g. from an FDE solve).")
    neff_table: list[float] | None = Field(None, description="neff at wavelengths_table_nm (overrides neff/ng).")
    loss_table_db_per_cm: list[float] | None = Field(None, description="Loss at wavelengths_table_nm (adds to "
                                                     "loss_db_per_cm).")

    @model_validator(mode="after")
    def _tables(self):
        n = None if self.wavelengths_table_nm is None else len(self.wavelengths_table_nm)
        for name in ("neff_table", "loss_table_db_per_cm"):
            v = getattr(self, name)
            if v is not None and (n is None or len(v) != n):
                raise ValueError(f"{name} needs wavelengths_table_nm of the same length")
        return self


@compact_model("waveguide", WaveguideSettings, ("o1", "o2"), aliases=("straight",))
def _waveguide(ctx: Ctx, s):
    """Straight waveguide o1 -> o2: exp(i 2 pi neff(lambda) L / lambda) with loss in dB/cm.
    Dispersion from ng, or from tabulated neff (neff_table) e.g. built by waveguide_from_modes()."""
    n_lam = None
    if s["neff_table"] is not None:
        n_lam = ctx.table(s["wavelengths_table_nm"], s["neff_table"])
    t = propagation(ctx, s["length_nm"], s["neff"], s["ng"], s["wl0_nm"], s["loss_db_per_cm"], n_lam)
    if s["loss_table_db_per_cm"] is not None:
        t = t * amplitude_db(ctx, ctx.table(s["wavelengths_table_nm"], s["loss_table_db_per_cm"])
                             * ctx.p(s["length_nm"]) * 1e-7)
    return ["o1", "o2"], ctx.smatrix(["o1", "o2"], {("o2", "o1"): t})


class BendSettings(_Guide):
    radius_nm: float = Field(5_000.0, gt=0, description="Bend radius.")
    angle_deg: float = Field(90.0, gt=0, le=360, description="Bend angle.")
    length_nm: float | None = Field(None, gt=0, description="Path length override (e.g. an Euler bend's length); "
                                    "default radius * angle.")
    bend_loss_db: float = Field(0.0, ge=0, description="Excess (radiation + mode-mismatch) loss per bend in dB.")


@compact_model("bend", BendSettings, ("o1", "o2"), aliases=("bend_circular", "bend_euler"))
def _bend(ctx: Ctx, s):
    """Waveguide bend o1 -> o2: a waveguide of length radius*angle (or length_nm) plus an excess loss per bend."""
    L = s["length_nm"] if s["length_nm"] is not None else ctx.p(s["radius_nm"]) * math.radians(1.0) * s["angle_deg"]
    t = propagation(ctx, L, s["neff"], s["ng"], s["wl0_nm"], s["loss_db_per_cm"]) * amplitude_db(ctx, s["bend_loss_db"])
    return ["o1", "o2"], ctx.smatrix(["o1", "o2"], {("o2", "o1"): t})


class PhaseShifterSettings(_Guide):
    length_nm: float = Field(100_000.0, ge=0, description="Length of the phase-shifter section.")
    mechanism: Literal["thermo-optic", "electro-optic"] = Field(
        "thermo-optic", description="thermo-optic: phase = phase_per_volt * V^2 (heater power ~ V^2); "
        "electro-optic: phase = phase_per_volt * V.")
    phase_per_volt: float = Field(0.0, description="rad/V^2 (thermo-optic) or rad/V (electro-optic). "
                                  "E.g. pi / Vpi for electro-optic, pi / Vpi^2 for a heater.")
    voltage: float = Field(0.0, description="Applied voltage.")
    phase_offset_rad: float = Field(0.0, description="Extra static phase.")
    excess_loss_db: float = Field(0.0, ge=0, description="Insertion loss of the section (e.g. doping, metal).")


@compact_model("phase_shifter", PhaseShifterSettings, ("o1", "o2"),
               aliases=("thermo_optic_phase_shifter", "electro_optic_phase_shifter"))
def _phase_shifter(ctx: Ctx, s):
    """Waveguide with a voltage-controlled extra phase: o1 -> o2. phase = phase_per_volt * V (electro-optic)
    or phase_per_volt * V^2 (thermo-optic), plus phase_offset_rad."""
    v = ctx.p(s["voltage"])
    dphi = ctx.p(s["phase_per_volt"]) * (v * v if s["mechanism"] == "thermo-optic" else v) + ctx.p(s["phase_offset_rad"])
    t = (propagation(ctx, s["length_nm"], s["neff"], s["ng"], s["wl0_nm"], s["loss_db_per_cm"])
         * ctx.expi(dphi) * amplitude_db(ctx, s["excess_loss_db"]))
    return ["o1", "o2"], ctx.smatrix(["o1", "o2"], {("o2", "o1"): t})


# --- splitters and couplers ---------------------------------------------------

_P4 = ["o1", "o2", "o3", "o4"]
_P3 = ["o1", "o2", "o3"]


def _two_by_two(ctx: Ctx, bar, cross):
    return ctx.smatrix(_P4, {("o4", "o1"): bar, ("o3", "o2"): bar, ("o3", "o1"): cross, ("o4", "o2"): cross})


class CouplerSettings(ModelSettings):
    coupling: float = Field(0.5, ge=0, le=1, description="Power cross-coupling kappa^2 at wl0_nm.")
    wl0_nm: float = Field(1550.0, gt=0)
    coupling_dispersion_per_nm: float = Field(
        0.0, description="Relative change of the coupling angle theta = asin(kappa) per nm: "
        "theta(lambda) = theta0 (1 + d (lambda - wl0)). Evanescent couplers couple more at longer "
        "wavelengths (d > 0, typically ~1e-3/nm).")
    loss_db: float = Field(0.0, ge=0, description="Excess insertion loss.")
    length_nm: float = Field(0.0, ge=0, description="Length of the common propagation phase (0 = none).")
    neff: float = Field(2.4)
    ng: float = Field(4.2)


@compact_model("coupler", CouplerSettings, tuple(_P4), aliases=("directional_coupler",))
def _coupler(ctx: Ctx, s):
    """Directional coupler (2x2): bar o1<->o4, o2<->o3 with amplitude cos(theta), cross o1<->o3, o2<->o4 with
    i sin(theta), sin^2(theta0) = coupling at wl0 and theta linear in wavelength. Unitary when lossless."""
    theta0 = ctx.xp.asin(ctx.xp.sqrt(ctx.p(s["coupling"])))
    theta = theta0 * (1 + ctx.p(s["coupling_dispersion_per_nm"]) * (ctx.wl - ctx.p(s["wl0_nm"])))
    common = amplitude_db(ctx, s["loss_db"]) * propagation(ctx, s["length_nm"], s["neff"], s["ng"], s["wl0_nm"], 0.0)
    return _P4, _two_by_two(ctx, common * ctx.c(ctx.xp.cos(theta)), common * 1j * ctx.c(ctx.xp.sin(theta)))


class _Splitter(ModelSettings):
    loss_db: float = Field(0.0, ge=0, description="Excess insertion loss at wl0_nm.")
    imbalance_db: float = Field(0.0, description="Power ratio of the two outputs in dB (o2/o3 for 1x2, bar/cross "
                                "for 2x2); 0 = perfect 50:50.")
    wl0_nm: float = Field(1550.0, gt=0)
    bandwidth_1db_nm: float | None = Field(None, gt=0, description="Full width where the excess loss grows by "
                                           "1 dB (parabolic in dB); None = flat.")


class MMI1x2Settings(_Splitter):
    loss_db: float = Field(0.2, ge=0, description="Excess insertion loss at wl0_nm.")


class YBranchSettings(_Splitter):
    pass


def _split_amplitudes(ctx: Ctx, s):
    r = ctx.xp.exp(ctx.p(s["imbalance_db"]) * (LN10 / 10))  # P_a / P_b
    total = amplitude_db(ctx, ctx.p(s["loss_db"]) + rolloff_db(ctx, s["wl0_nm"], s["bandwidth_1db_nm"]))
    return total * ctx.xp.sqrt(r / (1 + r)), total * ctx.xp.sqrt(1 / (1 + r))


def _one_by_two(ctx: Ctx, s):
    a, b = _split_amplitudes(ctx, s)
    return _P3, ctx.smatrix(_P3, {("o2", "o1"): a, ("o3", "o1"): b})


@compact_model("mmi1x2", MMI1x2Settings, tuple(_P3))
def _mmi1x2(ctx: Ctx, s):
    """1x2 multimode-interference splitter: o1 -> o2, o3 in phase, each 1/sqrt(2) of the transmitted amplitude.
    Not unitary: light entering the outputs in antiphase radiates away, as in a real 1x2."""
    return _one_by_two(ctx, s)


@compact_model("y_branch", YBranchSettings, tuple(_P3), aliases=("ybranch", "splitter_1x2"))
def _y_branch(ctx: Ctx, s):
    """Y-branch splitter: o1 -> o2, o3 in phase (lossless at the default settings for light from o1)."""
    return _one_by_two(ctx, s)


class MMI2x2Settings(_Splitter):
    loss_db: float = Field(0.2, ge=0, description="Excess insertion loss at wl0_nm.")


@compact_model("mmi2x2", MMI2x2Settings, tuple(_P4))
def _mmi2x2(ctx: Ctx, s):
    """2x2 multimode-interference 3 dB coupler: bar o1<->o4, o2<->o3 and cross (+90 degrees) o1<->o3, o2<->o4.
    imbalance_db sets the bar/cross power ratio; unitary apart from loss_db."""
    a, b = _split_amplitudes(ctx, s)
    return _P4, _two_by_two(ctx, ctx.c(a), 1j * ctx.c(b))


# --- rings ------------------------------------------------------------------------


class RingSettings(_Guide):
    radius_nm: float = Field(10_000.0, gt=0)
    straight_length_nm: float = Field(0.0, ge=0, description="Length of each straight (coupling) section; "
                                      "> 0 makes a racetrack, roundtrip = 2 pi R + 2 * straight_length_nm.")
    coupling: float = Field(0.1, ge=0, le=1, description="Power coupling kappa^2 between bus and ring.")
    extra_phase_rad: float = Field(0.0, description="Extra roundtrip phase (e.g. a heater on the ring).")


class RacetrackSettings(RingSettings):
    straight_length_nm: float = Field(5_000.0, ge=0, description="Length of each straight (coupling) section.")


class RingDoubleSettings(RingSettings):
    coupling_drop: float | None = Field(None, ge=0, le=1, description="Power coupling to the drop bus "
                                        "(default: equal to coupling).")


class RacetrackDoubleSettings(RingDoubleSettings):
    straight_length_nm: float = Field(5_000.0, ge=0, description="Length of each straight (coupling) section.")


def _ring_phase(ctx: Ctx, s):
    """(roundtrip amplitude a, exp(i phi)) for a ring/racetrack."""
    L = 2 * math.pi * ctx.p(s["radius_nm"]) + 2 * ctx.p(s["straight_length_nm"])
    rt = propagation(ctx, L, s["neff"], s["ng"], s["wl0_nm"], s["loss_db_per_cm"]) * ctx.expi(s["extra_phase_rad"])
    return rt


def _ring_single(ctx: Ctx, s):
    rt = _ring_phase(ctx, s)  # a exp(i phi)
    t = ctx.c(ctx.xp.sqrt(1 - ctx.p(s["coupling"])))
    through = (t - rt) / (1 - t * rt)
    return ["o1", "o2"], ctx.smatrix(["o1", "o2"], {("o2", "o1"): through})


def _ring_double(ctx: Ctx, s):
    k1 = ctx.p(s["coupling"])
    k2 = k1 if s["coupling_drop"] is None else ctx.p(s["coupling_drop"])
    t1, t2 = ctx.c(ctx.xp.sqrt(1 - k1)), ctx.c(ctx.xp.sqrt(1 - k2))
    rt = _ring_phase(ctx, s)
    half = _ring_phase(ctx, {**s, "radius_nm": ctx.p(s["radius_nm"]) / 2,
                             "straight_length_nm": ctx.p(s["straight_length_nm"]) / 2,
                             "extra_phase_rad": ctx.p(s["extra_phase_rad"]) / 2})
    den = 1 - t1 * t2 * rt
    thru_bottom = (t1 - t2 * rt) / den
    thru_top = (t2 - t1 * rt) / den
    drop = -ctx.xp.sqrt(k1 * k2) * half / den
    return _P4, ctx.smatrix(_P4, {("o2", "o1"): thru_bottom, ("o3", "o4"): thru_top,
                                  ("o4", "o1"): drop, ("o3", "o2"): drop})


@compact_model("ring_single", RingSettings, ("o1", "o2"), aliases=("ring_allpass", "ring"))
def _ring_single_model(ctx: Ctx, s):
    """All-pass ring on one bus, o1 -> o2: (t - a e^{i phi}) / (1 - t a e^{i phi}) with t = sqrt(1 - coupling),
    a e^{i phi} the roundtrip transmission (L = 2 pi R + 2 straight). Critical coupling at t = a."""
    return _ring_single(ctx, s)


@compact_model("racetrack_single", RacetrackSettings, ("o1", "o2"))
def _racetrack_single(ctx: Ctx, s):
    """All-pass racetrack resonator (ring_single with straight coupling sections), o1 -> o2."""
    return _ring_single(ctx, s)


@compact_model("ring_double", RingDoubleSettings, tuple(_P4), aliases=("ring_adddrop",))
def _ring_double_model(ctx: Ctx, s):
    """Add-drop ring between two buses (gdsfactory ring_double ports): input o1, through o2, drop o4, add o3.
    through = (t1 - t2 a e^{i phi}) / D, drop = -k1 k2 sqrt(a) e^{i phi/2} / D, D = 1 - t1 t2 a e^{i phi}."""
    return _ring_double(ctx, s)


@compact_model("racetrack_double", RacetrackDoubleSettings, tuple(_P4))
def _racetrack_double(ctx: Ctx, s):
    """Add-drop racetrack resonator (ports as ring_double)."""
    return _ring_double(ctx, s)


# --- grating coupler and terminator -------------------------------------------


class GratingCouplerSettings(ModelSettings):
    peak_loss_db: float = Field(3.0, ge=0, description="Insertion loss at the peak wavelength.")
    wl0_nm: float = Field(1550.0, gt=0, description="Peak wavelength.")
    bandwidth_1db_nm: float = Field(35.0, gt=0, description="1 dB bandwidth (full width); Gaussian passband.")
    reflection_db: float | None = Field(None, le=0, description="Back-reflection into the waveguide (dB, <= 0); "
                                        "None = no reflection.")
    reflection_fiber_db: float | None = Field(None, le=0, description="Back-reflection into the fiber (dB).")


@compact_model("grating_coupler", GratingCouplerSettings, ("o1", "o2"), aliases=("gc",))
def _grating_coupler(ctx: Ctx, s):
    """Fiber grating coupler: waveguide port o1 <-> fiber port o2 with a Gaussian passband, i.e. the loss in
    dB is peak_loss_db + ((lambda - wl0) / (bandwidth_1db / 2))^2."""
    t = amplitude_db(ctx, ctx.p(s["peak_loss_db"]) + rolloff_db(ctx, s["wl0_nm"], s["bandwidth_1db_nm"]))
    e = {("o2", "o1"): t}
    if s["reflection_db"] is not None:
        e[("o1", "o1")] = amplitude_db(ctx, -ctx.p(s["reflection_db"]))
    if s["reflection_fiber_db"] is not None:
        e[("o2", "o2")] = amplitude_db(ctx, -ctx.p(s["reflection_fiber_db"]))
    return ["o1", "o2"], ctx.smatrix(["o1", "o2"], e)


class TerminatorSettings(ModelSettings):
    reflection_db: float | None = Field(None, le=0, description="Residual reflection in dB (<= 0); None = matched.")


@compact_model("terminator", TerminatorSettings, ("o1",), aliases=("termination", "absorber"))
def _terminator(ctx: Ctx, s):
    """One-port absorber (matched load) on o1, optionally with a residual reflection."""
    r = 0.0 if s["reflection_db"] is None else amplitude_db(ctx, -ctx.p(s["reflection_db"]))
    return ["o1"], ctx.smatrix(["o1"], {("o1", "o1"): r})


# --- measured / simulated S-parameters -----------------------------------------


class SParamsSettings(ModelSettings):
    port_names: list[str] | None = Field(None, description="Rename the file's ports (in file order).")
    time_convention: Literal["+iwt", "-iwt"] = Field(
        "+iwt", description="Touchstone files: exp(+j w t) (the RF standard, conjugated on load) or exp(-i w t). "
        "Ignored for lightsmith JSON, which is always exp(-i w t).")
    interpolation: Literal["linear", "polar"] = Field(
        "polar", description="linear: real/imag; polar: magnitude and unwrapped phase (better for long devices).")
    out_of_range: Literal["error", "hold"] = Field("error", description="Wavelengths outside the file's range: "
                                                   "raise, or hold the nearest end value.")


@lru_cache(maxsize=64)
def _load_sparams_cached(path: str, mtime: float, time_convention: str):
    from .touchstone import read_sparameters

    return read_sparameters(path, time_convention)  # type: ignore[arg-type]


def load_sparams_file(path: str | os.PathLike, time_convention: str = "+iwt"):
    """(wavelengths_nm, ports, S) from a Touchstone or SParameterResult JSON file (cached by mtime)."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"S-parameter file not found: {p}")
    return _load_sparams_cached(str(p), p.stat().st_mtime, time_convention)


def _interp_complex(x, s, xq, mode: str) -> np.ndarray:
    """Interpolate S[x, i, j] onto xq along axis 0."""
    flat = s.reshape(len(x), -1)
    if len(x) == 1:
        return np.repeat(flat, len(xq), axis=0).reshape((len(xq),) + s.shape[1:])
    out = np.empty((len(xq), flat.shape[1]), dtype=complex)
    for k in range(flat.shape[1]):
        if mode == "linear":
            out[:, k] = np.interp(xq, x, flat[:, k].real) + 1j * np.interp(xq, x, flat[:, k].imag)
        else:
            mag = np.interp(xq, x, np.abs(flat[:, k]))
            ph = np.interp(xq, x, np.unwrap(np.angle(flat[:, k])))
            out[:, k] = mag * np.exp(1j * ph)
    return out.reshape((len(xq),) + s.shape[1:])


def _sparams_fn(ctx: Ctx, s, path: str):
    """S-parameters from a file: ``sparams:<path>`` with a Touchstone .sNp (ports o1..oN, or names from
    '! Port[n] = name' comments) or an lightsmith SParameterResult JSON (e.g. an FDTD/EME/FDFD result).
    Interpolated onto the circuit wavelengths."""
    wl, ports, S = load_sparams_file(path, s["time_convention"])
    if s["port_names"] is not None:
        if len(s["port_names"]) != len(ports):
            raise ValueError(f"port_names has {len(s['port_names'])} names but {path} has {len(ports)} ports")
        ports = list(s["port_names"])
    q = ctx.wl_np
    lo, hi = float(wl.min()), float(wl.max())
    if (q.min() < lo - 1e-6 or q.max() > hi + 1e-6) and s["out_of_range"] == "error":
        raise ValueError(f"{path}: data cover {lo:g}-{hi:g} nm but the circuit asks for {q.min():g}-{q.max():g} nm "
                         "(set settings.out_of_range='hold' to clamp)")
    Sq = _interp_complex(wl, S, np.clip(q, lo, hi), s["interpolation"])
    return list(ports), ctx.device.asarray(Sq, dtype=ctx.cdt)


def _sparams_model(path: str) -> CompactModel:
    if not path:
        raise KeyError("'sparams:' needs a file path, e.g. 'sparams:mmi.s3p' or 'sparams:fdtd_result.json'")
    return CompactModel(f"sparams:{path}", SParamsSettings, lambda ctx, s: _sparams_fn(ctx, s, path),
                        "S-parameters loaded from " + path, None)


# --- FDE mode results -> waveguide models ---------------------------------------


def _mode_loss_db_per_cm(m, wl_nm: float) -> float:
    if m.loss_db_per_cm is not None:
        return float(m.loss_db_per_cm)
    # Power attenuation 4 pi Im(neff) / lambda, in dB/cm.
    return float(10 / LN10 * 4 * math.pi * max(m.neff_imag, 0.0) / (wl_nm * 1e-7))


def waveguide_from_modes(result, length_nm: float, mode: int = 0, wl0_nm: float | None = None,
                         extra_loss_db_per_cm: float = 0.0):
    """Build a ``waveguide`` :class:`~lightsmith.problems.Instance` from an FDE ``ModeResult``.

    With one wavelength the model uses neff, ng (from the mode, or neff if absent)
    and the mode loss at that wavelength. With several wavelengths neff and loss are
    tabulated (and interpolated, no extrapolation), so the circuit sees the full
    dispersion that the mode solver computed. ``mode`` indexes modes sorted by
    decreasing neff, as in ``ModeResult.modes[wavelength][mode]``.
    """
    from ..problems import Instance

    wls = list(result.wavelengths_nm)
    if not wls:
        raise ValueError("ModeResult has no wavelengths")
    try:
        modes = [result.modes[i][mode] for i in range(len(wls))]
    except IndexError:
        raise ValueError(f"ModeResult has fewer than {mode + 1} modes at some wavelength") from None
    i0 = int(np.argmin(np.abs(np.asarray(wls) - (wl0_nm if wl0_nm is not None else np.mean(wls)))))
    m0 = modes[i0]
    ng0 = m0.ng if m0.ng is not None else m0.neff_real
    if len(wls) >= 2 and m0.ng is None:
        order = np.argsort(wls)
        lam, n = np.asarray(wls)[order], np.asarray([modes[i].neff_real for i in order])
        ng0 = float(m0.neff_real - wls[i0] * np.interp(wls[i0], lam, np.gradient(n, lam)))
    settings: dict[str, Any] = {"length_nm": float(length_nm), "neff": float(m0.neff_real), "ng": float(ng0),
                                "wl0_nm": float(wls[i0]), "loss_db_per_cm": float(extra_loss_db_per_cm)}
    if len(wls) == 1:
        settings["loss_db_per_cm"] += _mode_loss_db_per_cm(m0, wls[0])
    else:
        settings["wavelengths_table_nm"] = [float(w) for w in wls]
        settings["neff_table"] = [float(m.neff_real) for m in modes]
        settings["loss_table_db_per_cm"] = [_mode_loss_db_per_cm(m, w) for m, w in zip(modes, wls)]
    return Instance(component="waveguide", settings=settings)
