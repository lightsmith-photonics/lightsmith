"""Material search, sampling and Drude-Lorentz fits for the MCP tools.

Sources: the bundled library (offline), any refractiveindex.info entry
(downloaded once, then cached), and optionally a Lightsmith Materials API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import TypeAdapter

from .. import materials as M
from .remote import MaterialsClient

_MATERIAL = TypeAdapter(M.Material)
MATERIAL_HELP = (
    "A material is one of: a bundled name ('Au', 'gold', 'Au/Johnson-Christy'); any material JSON "
    "({'kind': 'constant_index', 'n_real': 1.5}, {'kind': 'library', 'name': 'Si'}, drude, drude_lorentz, "
    "tabulated, constant_permittivity); a refractiveindex.info entry {'refractiveindex': {'shelf': 'main', "
    "'book': 'Au', 'page': 'Olmon-sc'}}; or a Lightsmith Materials API dataset {'materials_api': 'au/johnson-christy-1972'}"
    " (needs LIGHTSMITH_MATERIALS_URL).")


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def search_bundled(query: str) -> list[dict[str, Any]]:
    q = _norm(query)
    rows = []
    for r in M.list_materials():
        hay = [r["name"], r["description"], r["dataset"], *r["aliases"], *r["dataset_aliases"]]
        if not q or any(q in _norm(h) for h in hay):
            rows.append({"source": "bundled", "name": r["name"], "dataset": r["dataset"], "default": r["default"],
                         "description": r["description"], "range_nm": r["range_nm"], "has_k": r["has_k"],
                         "reference": r["reference"], "material_json": r["material_json"]})
    return rows


def _rii_catalog(ref: str = "main", timeout: float = 30.0):
    cat = M.default_cache_dir() / "refractiveindex.info" / ref / "catalog-nk.yml"
    if not cat.is_file():
        cat.parent.mkdir(parents=True, exist_ok=True)
        try:
            cat.write_bytes(M._download(M._raw_url(ref, "catalog-nk.yml"), timeout))
        except Exception as e:
            raise ConnectionError(f"could not download the refractiveindex.info catalogue ({e}); "
                                  "offline? search the bundled library instead (source='bundled')") from None
    shelf = book = book_name = page = None
    for line in cat.read_text(encoding="utf-8").splitlines():
        s = line.strip().removeprefix("- ")
        key, _, val = s.partition(":")
        val = M._yaml_scalar(val)
        if key == "SHELF":
            shelf = val
        elif key == "BOOK":
            book, book_name, page = val, None, None
        elif key == "PAGE":
            page = val
        elif key == "name":
            if page is None:
                book_name = val
            else:
                yield {"shelf": shelf, "book": book, "book_name": book_name, "page": page, "page_name": val}


def search_refractiveindex(query: str, limit: int = 25) -> list[dict[str, Any]]:
    q = _norm(query)
    out = []
    for e in _rii_catalog():
        if q and not (q == _norm(e["book"]) or q in _norm(e["book_name"] or "") or q in _norm(e["page"])):
            continue
        out.append({"source": "refractiveindex.info", "name": e["book_name"] or e["book"],
                    "dataset": e["page_name"],
                    "material": {"refractiveindex": {"shelf": e["shelf"], "book": e["book"], "page": e["page"]}}})
        if len(out) >= limit:
            break
    return out


def resolve(material: Any, materials_api: MaterialsClient | None = None):
    """(model, spec_json, source) for any accepted material form; see MATERIAL_HELP."""
    if isinstance(material, str):
        name, _, dataset = material.partition("/")
        m = M.LibraryMaterial(name=name, dataset=dataset or None)
        return m, m.model_dump(), m.source
    if not isinstance(material, dict):
        raise ValueError("unrecognised material. " + MATERIAL_HELP)
    if "refractiveindex" in material:
        r = material["refractiveindex"]
        t = M.fetch_refractiveindex(r["shelf"], r["book"], r["page"])
        return t, t.model_dump(), t.source
    if "materials_api" in material:
        if materials_api is None:
            raise ValueError("no Lightsmith Materials API configured; set LIGHTSMITH_MATERIALS_URL")
        spec = materials_api.material_json(str(material["materials_api"]))
        m = _MATERIAL.validate_python(spec)
        return m, m.model_dump(), getattr(m, "source", None)
    m = _MATERIAL.validate_python(material)
    return m, m.model_dump(), getattr(m, "source", None)


def material_range(m) -> tuple[float, float] | None:
    r = getattr(m, "range_nm", None)
    return tuple(r) if r else None


def sample(material: Any, wavelengths_nm: list[float] | None = None, start_nm: float | None = None,
           stop_nm: float | None = None, num: int = 11, materials_api: MaterialsClient | None = None) -> dict[str, Any]:
    m, spec, source = resolve(material, materials_api)
    rng = material_range(m)
    if wavelengths_nm:
        wl = np.asarray(wavelengths_nm, dtype=float)
    else:
        lo = start_nm if start_nm is not None else (rng[0] if rng else 400.0)
        hi = stop_nm if stop_nm is not None else (rng[1] if rng else 1600.0)
        wl = np.linspace(lo, hi, max(1, min(int(num), 500)))
    eps = np.asarray(m.eps(wl), dtype=complex)
    n = np.sqrt(eps)
    n = np.where(n.imag < 0, -n, n)
    big = len(str(spec)) > 4000
    return {
        "material_json": spec, "material_json_note": ("tabulated data: paste material_json into the problem as is"
                                                      if big else None),
        "source": source, "range_nm": list(rng) if rng else None,
        "units": {"wavelength": "nm (vacuum)", "convention": "exp(-i omega t): k > 0 and Im(eps) > 0 mean loss"},
        "samples": [{"wavelength_nm": float(w), "n": float(a.real), "k": float(a.imag),
                     "eps_real": float(e.real), "eps_imag": float(e.imag)} for w, a, e in zip(wl, n, eps)],
    }


def fit(material: Any, start_nm: float, stop_nm: float, n_poles: int | None = None, drude: bool | None = None,
        materials_api: MaterialsClient | None = None) -> dict[str, Any]:
    from ..materials_fit import fit_drude_lorentz

    m, _, source = resolve(material, materials_api)
    if not stop_nm > start_nm > 0:
        raise ValueError("need 0 < start_nm < stop_nm")
    wl = np.linspace(start_nm, stop_nm, 64)
    model, info = fit_drude_lorentz(m, wl, n_poles, drude=drude, return_info=True)
    d = info.to_dict()
    d.pop("model", None)
    return {"material_json": model.model_dump(), "fit": d, "source_of_data": source,
            "note": "Passive Drude-Lorentz model, usable by FDTD (and every other method). Valid inside band_nm only."}
