"""Compact summaries, CSV tables and PNG plots of result JSON.

Everything works on the JSON form of a result (``model_dump()``), so local and
remote (Lightsmith Cloud) results are handled the same way. Summaries hold key
numbers with units, warnings and provenance; the full arrays stay in resources.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any

import numpy as np

_PROVENANCE_KEYS = ("device", "framework", "gpu_name", "precision", "runtime_s", "iterations", "residual",
                    "converged", "solver", "miepython_version", "dipoles", "grid", "cells", "harmonics")


def json_safe(obj: Any) -> Any:
    """Replace NaN/Inf with None (JSON has neither), recursively; tuples become lists."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    return obj


def _arr(x) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in x] if isinstance(x, list) and x and not
                      isinstance(x[0], list) else x, dtype=float)


def _num(x: float, digits: int = 6) -> float | None:
    x = float(x)
    return None if not math.isfinite(x) else float(f"{x:.{digits}g}")


def _wl_info(wl: list[float]) -> dict[str, Any]:
    if not wl:
        return {"n_wavelengths": 0}
    return {"n_wavelengths": len(wl), "wavelength_range_nm": [_num(min(wl)), _num(max(wl))]}


def _provenance(p: dict[str, Any]) -> dict[str, Any]:
    return {k: p[k] for k in _PROVENANCE_KEYS
            if k in p and not isinstance(p[k], dict) and not (isinstance(p[k], list) and len(p[k]) > 4)}


def result_kind(result: dict[str, Any]) -> str:
    if "relative_difference" in result and "results" in result:
        return "compare"
    return str(result.get("kind", "unknown"))


def _cross_sections(r: dict[str, Any]) -> dict[str, Any]:
    wl = r["wavelengths_nm"]
    ext, sca, ab = (_arr(r[k]) for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"))
    area = math.pi * float(r["equivalent_radius_nm"]) ** 2
    i = int(np.nanargmax(ext))
    out = {**_wl_info(wl), "units": {"cross_sections": "nm^2", "wavelength": "nm"},
           "equivalent_radius_nm": r["equivalent_radius_nm"],
           "peak_extinction": {"wavelength_nm": wl[i], "c_ext_nm2": _num(ext[i]), "c_sca_nm2": _num(sca[i]),
                               "c_abs_nm2": _num(ab[i]), "q_ext": _num(ext[i] / area)},
           "c_ext_nm2_range": [_num(np.nanmin(ext)), _num(np.nanmax(ext))],
           "albedo_at_peak": _num(sca[i] / ext[i]) if ext[i] else None}
    if r.get("near_field"):
        out["near_field"] = [{"wavelength_nm": f["wavelength_nm"], "plane": f["plane"],
                              "max_enhancement": _num(np.nanmax(_arr(f["enhancement"])))} for f in r["near_field"]]
    if r.get("far_field"):
        out["far_field"] = [{"wavelength_nm": f["wavelength_nm"], "n_theta": len(f["theta_deg"]),
                             "n_phi": len(f["phi_deg"])} for f in r["far_field"]]
    return out


def _spectrum(r: dict[str, Any]) -> dict[str, Any]:
    wl, angles = r["wavelengths_nm"], r["angles_deg"]
    out: dict[str, Any] = {**_wl_info(wl), "angles_deg": angles, "polarizations": r["polarizations"],
                           "units": {"R/T/A": "power fraction", "wavelength": "nm"}, "per_polarization": {}}
    worst = 0.0
    for pol in r["polarizations"]:
        R, T, A = (np.asarray(r[q][pol], dtype=float) for q in ("r", "t", "a"))
        worst = max(worst, float(np.nanmax(np.abs(R + T + A - 1))))
        rows = []
        for j, ang in enumerate(angles[:4]):
            iR, iT = int(np.nanargmax(R[j])), int(np.nanargmin(T[j]))
            rows.append({"angle_deg": ang, "R_mean": _num(np.nanmean(R[j]), 4), "T_mean": _num(np.nanmean(T[j]), 4),
                         "A_mean": _num(np.nanmean(A[j]), 4),
                         "R_max": {"value": _num(R[j][iR], 4), "wavelength_nm": wl[iR]},
                         "T_min": {"value": _num(T[j][iT], 4), "wavelength_nm": wl[iT]}})
        out["per_polarization"][pol] = rows
    if len(angles) > 4:
        out["note"] = f"first 4 of {len(angles)} angles shown; read the result resource for all"
    out["max_energy_balance_error"] = _num(worst, 3)
    if r.get("orders"):
        out["n_diffraction_orders"] = len(r["orders"])
    return out


def _modes(r: dict[str, Any]) -> dict[str, Any]:
    wl = r["wavelengths_nm"]
    idx = sorted({0, len(wl) // 2, len(wl) - 1}) if wl else []
    keys = ("neff_real", "neff_imag", "ng", "te_fraction", "loss_db_per_cm", "effective_area_um2")
    return {**_wl_info(wl), "units": {"loss": "dB/cm", "effective_area": "um^2"},
            "modes_per_wavelength": len(r["modes"][0]) if r["modes"] else 0,
            "at_wavelengths": [{"wavelength_nm": wl[i],
                                "modes": [{k: _num(m[k]) for k in keys if m.get(k) is not None}
                                          for m in r["modes"][i]]} for i in idx]}


def _smatrix(r: dict[str, Any]) -> np.ndarray:
    return np.asarray(r["s_real"], dtype=float) + 1j * np.asarray(r["s_imag"], dtype=float)


def _sparameters(r: dict[str, Any]) -> dict[str, Any]:
    wl, ports = r["wavelengths_nm"], r["ports"]
    S = _smatrix(r)
    P = np.abs(S) ** 2
    ic = len(wl) // 2
    paths = []
    for i_in, p_in in enumerate(ports):
        for i_out, p_out in enumerate(ports):
            p = P[:, i_out, i_in]
            if np.nanmax(p) < 1e-3:
                continue
            paths.append({"path": f"{p_in} -> {p_out}", "power_center": _num(p[ic], 4),
                          "dB_center": _num(10 * np.log10(max(p[ic], 1e-30)), 4),
                          "power_min": _num(np.nanmin(p), 4), "power_max": _num(np.nanmax(p), 4)})
    paths.sort(key=lambda d: -(d["power_max"] or 0))
    out = {**_wl_info(wl), "ports": ports, "center_wavelength_nm": wl[ic] if wl else None,
           "units": {"power": "|S|^2 fraction", "dB": "10 log10 |S|^2"},
           "significant_paths": paths[:24],
           "max_reciprocity_error": _num(np.nanmax(np.abs(S - np.swapaxes(S, 1, 2))), 3),
           "max_column_power": _num(np.nanmax(P.sum(axis=1)), 4)}
    if len(paths) > 24:
        out["note"] = f"{len(paths)} paths above 1e-3; strongest 24 shown"
    return out


def _bands(r: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_k_points": len(r["k_reduced"]), "k_labels": r["k_labels"],
                           "units": {"frequency": "a/lambda (normalised)"}, "per_polarization": {}}
    for pol, f in r["frequencies"].items():
        F = np.asarray(f, dtype=float)  # [k][band]
        lo, hi = np.nanmin(F, axis=0), np.nanmax(F, axis=0)
        gaps = [{"between_bands": [b + 1, b + 2], "from": _num(hi[b], 5), "to": _num(lo[b + 1], 5),
                 "gap_midgap_ratio": _num(2 * (lo[b + 1] - hi[b]) / (lo[b + 1] + hi[b]), 4)}
                for b in range(F.shape[1] - 1) if lo[b + 1] > hi[b]]
        out["per_polarization"][pol] = {"bands": F.shape[1],
                                        "band_ranges": [[_num(a, 5), _num(b, 5)] for a, b in zip(lo, hi)],
                                        "complete_gaps": gaps}
    return out


def _optimization(r: dict[str, Any]) -> dict[str, Any]:
    h = r.get("history") or []
    out = {"iterations": len(h), "objective_initial": _num(r["objective_initial"]),
           "objective_final": _num(r["objective_final"]), "objective_binary": _num(r["objective_binary"]),
           "final_binarization": _num(h[-1]["binarization"], 4) if h else None,
           "exported_structures": len(r.get("structures") or []),
           "designs": [{k: d[k] for k in ("name", "shape", "binarization") if k in d} for d in r.get("designs", [])]}
    if r.get("sparameters"):
        out["validated_sparameters"] = _sparameters(r["sparameters"])
    return out


def _compare(c: dict[str, Any]) -> dict[str, Any]:
    return {"reference": c["reference"], "methods": list(c["results"]),
            "relative_difference": c["relative_difference"],
            **({"not_compared": c["not_compared"]} if c.get("not_compared") else {}),
            "per_method": {m: {"backend": r.get("backend"), "runtime_s": (r.get("provenance") or {}).get("runtime_s"),
                               "warnings": r.get("warnings", [])} for m, r in c["results"].items()},
            "reference_summary": summarize(c["results"][c["reference"]])}


_SUMMARIZERS = {"cross_sections": _cross_sections, "spectrum": _spectrum, "modes": _modes,
                "sparameters": _sparameters, "bands": _bands, "optimization": _optimization, "compare": _compare}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Key numbers of any result, with method, backend, warnings and provenance."""
    kind = result_kind(result)
    out: dict[str, Any] = {"result_kind": kind}
    if kind != "compare":
        out.update({"method": result.get("method"), "backend": result.get("backend"),
                    "warnings": result.get("warnings", []),
                    "provenance": _provenance(result.get("provenance") or {})})
    fn = _SUMMARIZERS.get(kind)
    try:
        out.update(fn(result) if fn else {"fields": sorted(result)})
    except Exception as e:  # never let a summary hide a finished result
        out["summary_error"] = f"{type(e).__name__}: {e}"
    return out


# --- CSV -------------------------------------------------------------------------


def to_csv(result: dict[str, Any]) -> str | None:
    """A flat table of the main spectra (None for result kinds without one)."""
    kind = result_kind(result)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if kind == "compare":
        return to_csv(result["results"][result["reference"]])
    if kind == "cross_sections":
        r = result
        area = math.pi * float(r["equivalent_radius_nm"]) ** 2
        w.writerow(["wavelength_nm", "c_ext_nm2", "c_sca_nm2", "c_abs_nm2", "q_ext", "q_sca", "q_abs"])
        for i, wl in enumerate(r["wavelengths_nm"]):
            c = [r[k][i] for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2")]
            w.writerow([wl, *c, *[(x / area if x is not None else None) for x in c]])
    elif kind == "spectrum":
        r = result
        cols = [(q, pol, j) for pol in r["polarizations"] for j in range(len(r["angles_deg"])) for q in ("r", "t", "a")]
        w.writerow(["wavelength_nm"] + [f"{q.upper()}_{pol}_{r['angles_deg'][j]:g}deg" for q, pol, j in cols])
        for i, wl in enumerate(r["wavelengths_nm"]):
            w.writerow([wl] + [r[q][pol][j][i] for q, pol, j in cols])
    elif kind == "modes":
        keys = ("neff_real", "neff_imag", "ng", "te_fraction", "loss_db_per_cm", "effective_area_um2")
        w.writerow(["wavelength_nm", "mode", *keys])
        for wl, ms in zip(result["wavelengths_nm"], result["modes"]):
            for m_i, m in enumerate(ms):
                w.writerow([wl, m_i, *[m.get(k) for k in keys]])
    elif kind in ("sparameters", "optimization"):
        r = result if kind == "sparameters" else result.get("sparameters")
        if kind == "optimization":
            hist = result.get("history") or []
            w.writerow(["iteration", "objective", "beta", "binarization", "penalty", "evaluations"])
            for h in hist:
                w.writerow([h.get("iteration"), h.get("objective"), h.get("beta"), h.get("binarization"),
                            h.get("penalty"), h.get("evaluations")])
            return buf.getvalue()
        S = _smatrix(r)
        ports = r["ports"]
        pairs = [(o, i) for i in range(len(ports)) for o in range(len(ports))]
        w.writerow(["wavelength_nm"] + [f"S[{ports[o]}<-{ports[i]}]_{q}" for o, i in pairs for q in ("re", "im")])
        for k, wl in enumerate(r["wavelengths_nm"]):
            w.writerow([wl] + [v for o, i in pairs for v in (S[k, o, i].real, S[k, o, i].imag)])
    elif kind == "bands":
        r = result
        w.writerow(["k_index", "k_distance", "label", "polarization", "band", "frequency_a_over_lambda"])
        labels = {int(k): v for k, v in r["k_labels"].items()}
        for pol, F in r["frequencies"].items():
            for ki, row in enumerate(F):
                for b, f in enumerate(row):
                    w.writerow([ki, r["k_distance"][ki], labels.get(ki, ""), pol, b + 1, f])
    else:
        return None
    return buf.getvalue()


# --- plots -----------------------------------------------------------------------


def plot_png(result: dict[str, Any], title: str | None = None) -> bytes | None:
    """A PNG plot of the main quantity, or None without matplotlib or for unplottable kinds."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    kind = result_kind(result)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=110)
    try:
        if kind == "compare":
            for m, r in result["results"].items():
                if result_kind(r) == "cross_sections":
                    ax.plot(r["wavelengths_nm"], _arr(r["c_ext_nm2"]), label=f"{m} C_ext")
                elif result_kind(r) == "spectrum":
                    pol = r["polarizations"][0]
                    ax.plot(r["wavelengths_nm"], r["t"][pol][0], label=f"{m} T_{pol}")
            ax.set_xlabel("wavelength (nm)")
            ax.legend()
        elif kind == "cross_sections":
            for k, lab in (("c_ext_nm2", "extinction"), ("c_sca_nm2", "scattering"), ("c_abs_nm2", "absorption")):
                ax.plot(result["wavelengths_nm"], _arr(result[k]), label=lab)
            ax.set_xlabel("wavelength (nm)")
            ax.set_ylabel("cross section (nm²)")
            ax.legend()
        elif kind == "spectrum":
            for pol in result["polarizations"]:
                for q in ("r", "t", "a"):
                    ax.plot(result["wavelengths_nm"], result[q][pol][0],
                            label=f"{q.upper()} {pol} {result['angles_deg'][0]:g}°")
            ax.set_xlabel("wavelength (nm)")
            ax.set_ylabel("power fraction")
            ax.legend(fontsize=7)
        elif kind == "modes":
            wl = result["wavelengths_nm"]
            for m in range(len(result["modes"][0])):
                ax.plot(wl, [ms[m]["neff_real"] if m < len(ms) else np.nan for ms in result["modes"]], "o-",
                        label=f"mode {m}")
            ax.set_xlabel("wavelength (nm)")
            ax.set_ylabel("n_eff")
            ax.legend()
        elif kind in ("sparameters", "optimization"):
            if kind == "optimization":
                h = result.get("history") or []
                ax.plot([x["iteration"] for x in h], [x["objective"] for x in h], "o-")
                ax.set_xlabel("iteration")
                ax.set_ylabel("objective")
            else:
                S, ports, wl = _smatrix(result), result["ports"], result["wavelengths_nm"]
                P = np.abs(S) ** 2
                for i in range(len(ports)):
                    for o in range(len(ports)):
                        if np.nanmax(P[:, o, i]) > 1e-3 and len(ax.lines) < 12:
                            ax.plot(wl, 10 * np.log10(np.maximum(P[:, o, i], 1e-30)), ".-" if len(wl) < 30 else "-",
                                    label=f"{ports[i]}→{ports[o]}")
                ax.set_xlabel("wavelength (nm)")
                ax.set_ylabel("|S|² (dB)")
                ax.legend(fontsize=7)
        elif kind == "bands":
            for pol, F in result["frequencies"].items():
                F = np.asarray(F, dtype=float)
                for b in range(F.shape[1]):
                    ax.plot(result["k_distance"], F[:, b], "-", color="C0" if pol != "tm" else "C3",
                            label=pol if b == 0 else None)
            labels = {int(k): v for k, v in result["k_labels"].items()}
            ax.set_xticks([result["k_distance"][k] for k in labels], list(labels.values()))
            ax.set_ylabel("frequency (a/λ)")
            ax.legend()
        else:
            return None
        ax.grid(alpha=0.3)
        if title:
            ax.set_title(title, fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()
    finally:
        plt.close(fig)
