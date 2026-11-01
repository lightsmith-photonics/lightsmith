"""Quantities read out of solver results (plain JSON dicts, as returned over MCP).

One extractor spec, e.g. ``{"kind": "mode", "field": "ng", "wavelength_nm": 1550}``,
is applied to the reference result when tasks are generated, to the oracle's own
MCP results, and to the agent's results for self-consistency checks. Extractors
work on dicts (``Result.model_dump()``) so they do not depend on how a server
wraps its answer, and raise :class:`ExtractError` with a readable reason.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

HC_EV_NM = 1239.841984


class ExtractError(ValueError):
    pass


def _wl_index(wavelengths: list[float], wl: float, tol: float = 0.05) -> int | None:
    w = np.asarray(wavelengths, float)
    i = int(np.argmin(np.abs(w - wl)))
    return i if abs(w[i] - wl) <= tol else None


def _at_wavelength(wavelengths: list[float], values, wl: float) -> float:
    """Value at `wl`: exact grid point, else linear interpolation between neighbours."""
    w, v = np.asarray(wavelengths, float), np.asarray(values, float)
    i = _wl_index(wavelengths, wl)
    if i is not None:
        return float(v[i])
    if w.size < 2 or not (w.min() <= wl <= w.max()):
        raise ExtractError(f"wavelength {wl:g} nm not in the result (have {w.min():g}-{w.max():g} nm)")
    order = np.argsort(w)
    return float(np.interp(wl, w[order], v[order]))


def _peak(wavelengths, values, mode: str = "max") -> float:
    """Wavelength of the extremum, refined with a parabola through the three nearest samples."""
    w, v = np.asarray(wavelengths, float), np.asarray(values, float)
    order = np.argsort(w)
    w, v = w[order], v[order]
    i = int(np.argmax(v) if mode == "max" else np.argmin(v))
    if 0 < i < len(w) - 1:
        x, y = w[i - 1:i + 2], v[i - 1:i + 2]
        a, b, _ = np.polyfit(x, y, 2)
        if a != 0:
            xv = -b / (2 * a)
            if x[0] <= xv <= x[2]:
                return float(xv)
    return float(w[i])


def _pol_values(table: dict[str, list[list[float]]], pol: str, ia: int) -> np.ndarray:
    if pol == "avg":
        if not {"s", "p"} <= set(table):
            raise ExtractError("unpolarised average needs both s and p results")
        return (np.asarray(table["s"][ia]) + np.asarray(table["p"][ia])) / 2
    if pol not in table:
        raise ExtractError(f"polarization {pol!r} not in the result (have {sorted(table)})")
    return np.asarray(table[pol][ia], float)


def _angle_index(angles: list[float], angle: float | None) -> int:
    """Index of `angle` in the result; None picks the only angle of a single-angle result."""
    if angle is None:
        if len(angles) != 1:
            raise ExtractError(f"result has several angles {angles}; the extractor must name one")
        return 0
    i = int(np.argmin(np.abs(np.asarray(angles, float) - angle)))
    if abs(angles[i] - angle) > 1e-3:
        raise ExtractError(f"angle {angle:g} deg not in the result (have {angles})")
    return i


def _modes_at(res: dict, wl: float) -> list[dict]:
    i = _wl_index(res["wavelengths_nm"], wl, tol=0.5)
    if i is None:
        raise ExtractError(f"no modes at {wl:g} nm (have {res['wavelengths_nm']})")
    return res["modes"][i]


def _select_mode(modes: list[dict], select: str | None, index: int) -> dict:
    ms = sorted(modes, key=lambda m: -m["neff_real"])
    if select == "te":
        ms = [m for m in ms if (m.get("te_fraction") or 0) > 0.5]
    elif select == "tm":
        ms = [m for m in ms if (m.get("te_fraction") or 0) <= 0.5]
    if index >= len(ms):
        raise ExtractError(f"only {len(ms)} {select or ''} mode(s) in the result")
    return ms[index]


def _port_index(ports: list[str], name: str) -> int:
    for cand in (name, f"{name}@0"):
        if cand in ports:
            return ports.index(cand)
    raise ExtractError(f"port {name!r} not in {ports}")


def _s_matrix(res: dict) -> np.ndarray:
    return np.asarray(res["s_real"]) + 1j * np.asarray(res["s_imag"])


def _sparams(res: dict) -> dict:
    if res.get("kind") == "optimization":
        if not res.get("sparameters"):
            raise ExtractError("optimization result has no final S-parameters")
        return res["sparameters"]
    return res


def _band_gaps(res: dict) -> dict:
    from lightsmith.backends.pwe import band_gaps
    from lightsmith.results import BandResult

    return band_gaps(BandResult.model_validate(res))


def _minima(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    order = np.argsort(w)
    w, v = w[order], v[order]
    idx = [i for i in range(1, len(v) - 1) if v[i] < v[i - 1] and v[i] <= v[i + 1]]
    out = []
    for i in idx:  # parabolic refinement of each dip
        a, b, _ = np.polyfit(w[i - 1:i + 2], v[i - 1:i + 2], 2)
        out.append(-b / (2 * a) if a > 0 else w[i])
    return np.asarray(out)


def extract(spec: dict[str, Any], res: dict[str, Any]) -> float:
    """Apply one extractor spec to a result dict and return a float."""
    k = spec["kind"]
    try:
        if k == "cross_section":
            q = spec["quantity"]
            vals = res[q.replace("q_", "c_") + "_nm2"] if q.startswith("q_") else res[q]
            v = _at_wavelength(res["wavelengths_nm"], vals, spec["wavelength_nm"])
            return v / (math.pi * res["equivalent_radius_nm"] ** 2) if q.startswith("q_") else v
        if k == "peak_wavelength":
            return _peak(res["wavelengths_nm"], res[spec.get("quantity", "c_ext_nm2")], spec.get("mode", "max"))
        if k == "spectrum":
            ia = _angle_index(res["angles_deg"], spec.get("angle_deg"))
            vals = _pol_values(res[spec["quantity"]], spec.get("pol", "avg"), ia)
            return _at_wavelength(res["wavelengths_nm"], vals, spec["wavelength_nm"])
        if k == "spectrum_extremum_angle":
            # quantity vs angle at one wavelength -> angle of the extremum
            iw = _wl_index(res["wavelengths_nm"], spec["wavelength_nm"], tol=0.5)
            if iw is None:
                raise ExtractError(f"wavelength {spec['wavelength_nm']} nm not in the result")
            table = res[spec["quantity"]]
            vals = [_pol_values(table, spec.get("pol", "p"), ia)[iw] for ia in range(len(res["angles_deg"]))]
            if len(vals) < 3:
                raise ExtractError("need an angle sweep (at least 3 angles)")
            return _peak(res["angles_deg"], vals, spec.get("mode", "min"))
        if k == "diffraction_order":
            want = tuple(spec["order"])
            for o in res.get("orders") or []:
                if tuple(o["order"]) == want:
                    ia = _angle_index(res["angles_deg"], spec.get("angle_deg"))
                    vals = _pol_values(o[spec.get("side", "t")], spec.get("pol", "avg"), ia)
                    return _at_wavelength(res["wavelengths_nm"], vals, spec["wavelength_nm"])
            return 0.0  # order absent = evanescent
        if k == "propagating_orders":
            iw = _wl_index(res["wavelengths_nm"], spec["wavelength_nm"], tol=0.5)
            if iw is None:
                raise ExtractError(f"wavelength {spec['wavelength_nm']} nm not in the result")
            ia = _angle_index(res["angles_deg"], spec.get("angle_deg"))
            n = 0
            for o in res.get("orders") or []:
                vals = _pol_values(o[spec.get("side", "t")], spec.get("pol", "avg"), ia)
                n += vals[iw] > spec.get("threshold", 1e-6)
            return float(n)
        if k == "mode":
            m = _select_mode(_modes_at(res, spec["wavelength_nm"]), spec.get("select"), spec.get("index", 0))
            v = m.get(spec["field"])
            if v is None:
                raise ExtractError(f"mode field {spec['field']!r} not computed")
            return float(v)
        if k == "guided_mode_count":
            return float(sum(m["neff_real"] > spec["cladding_index"] + 1e-4
                             for m in _modes_at(res, spec["wavelength_nm"])))
        if k == "sparam":
            r = _sparams(res)
            iw = _wl_index(r["wavelengths_nm"], spec["wavelength_nm"], tol=0.5)
            if iw is None:
                raise ExtractError(f"wavelength {spec['wavelength_nm']} nm not in the S-parameters")
            s = _s_matrix(r)[iw, _port_index(r["ports"], spec["out"]), _port_index(r["ports"], spec["in"])]
            p = abs(s) ** 2
            return {"power": p, "db": 10 * math.log10(max(p, 1e-30)), "phase": math.atan2(s.imag, s.real)}[
                spec.get("as", "power")]
        if k == "sparam_fsr":
            r = _sparams(res)
            s = _s_matrix(r)[:, _port_index(r["ports"], spec["out"]), _port_index(r["ports"], spec["in"])]
            w = np.asarray(r["wavelengths_nm"], float)
            mins = _minima(w, np.abs(s) ** 2)
            near = spec.get("near_nm")
            if len(mins) < 2:
                raise ExtractError("fewer than two transmission minima in the result")
            gaps = np.diff(np.sort(mins))
            if near is not None:
                mids = (np.sort(mins)[1:] + np.sort(mins)[:-1]) / 2
                return float(gaps[int(np.argmin(np.abs(mids - near)))])
            return float(np.mean(gaps))
        if k == "band_gap":
            gaps = _band_gaps(res).get(spec.get("pol", "tm"), [])
            if spec.get("between_bands") is not None:
                gaps = [g for g in gaps if tuple(g.get("between_bands", ())) == tuple(spec["between_bands"])]
            if not gaps:
                return 0.0 if spec["field"] in ("gap_midgap_ratio", "width") else math.nan
            g = max(gaps, key=lambda g: g["gap_midgap_ratio"]) if spec.get("largest", True) else gaps[0]
            if spec["field"].startswith("wavelength_"):  # vacuum wavelength edge for lattice constant a
                a = spec["lattice_constant_nm"]
                return a / g["top"] if spec["field"] == "wavelength_min_nm" else a / g["bottom"]
            return float(g[spec["field"]])
        if k == "optimization":
            return float(res[spec["field"]])
    except ExtractError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ExtractError(f"{k}: cannot read the result ({type(e).__name__}: {e})") from e
    raise ExtractError(f"unknown extractor kind {k!r}")


def result_kind(res: Any) -> str | None:
    """The result kind of a solver output dict ('cross_sections', 'spectrum', 'modes', ...), else None."""
    if isinstance(res, dict) and isinstance(res.get("kind"), str) and "backend" in res and "method" in res:
        return res["kind"]
    return None


RESULT_KIND = {"scattering": "cross_sections", "layer_stack": "spectrum", "periodic": "spectrum",
               "mode": "modes", "device": "sparameters", "bands": "bands", "circuit": "sparameters",
               "optimization": "optimization"}
