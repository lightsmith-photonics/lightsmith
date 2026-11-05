"""Optical material models.

Convention (used everywhere in lightsmith): time dependence exp(-i*omega*t),
so absorbing materials have Im(eps) > 0 and Im(n) > 0. Backends that use the
opposite convention convert at their boundary.

Measured data come from the refractiveindex.info database (CC0 1.0,
https://github.com/polyanskiy/refractiveindex.info-database). A small, cited
subset ships with the package (``{"kind": "library", "name": "Au",
"dataset": "Johnson-Christy"}``, see :func:`list_materials`); any other entry
can be downloaded with :func:`fetch_refractiveindex`.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

HC_EV_NM = 1239.841984  # h*c in eV*nm


class MaterialRangeError(ValueError):
    """A wavelength lies outside the range where a material's data are valid."""


class _Material(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def eps(self, wavelength_nm: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def n(self, wavelength_nm: np.ndarray) -> np.ndarray:
        return np.sqrt(self.eps(wavelength_nm).astype(complex))


class ConstantIndex(_Material):
    """Wavelength-independent complex refractive index n + ik."""

    kind: Literal["constant_index"] = "constant_index"
    n_real: float = Field(description="Real part of the refractive index.")
    n_imag: float = Field(0.0, ge=0, description="Extinction coefficient k (>= 0 for loss).")

    def eps(self, wavelength_nm):
        wl = np.asarray(wavelength_nm, dtype=float)
        return np.full(wl.shape, complex(self.n_real, self.n_imag) ** 2)


class ConstantPermittivity(_Material):
    """Wavelength-independent complex relative permittivity."""

    kind: Literal["constant_permittivity"] = "constant_permittivity"
    eps_real: float
    eps_imag: float = Field(0.0, ge=0)

    def eps(self, wavelength_nm):
        wl = np.asarray(wavelength_nm, dtype=float)
        return np.full(wl.shape, complex(self.eps_real, self.eps_imag))


class Drude(_Material):
    """eps(w) = eps_inf - wp^2 / (w^2 + i*gamma*w), energies in eV."""

    kind: Literal["drude"] = "drude"
    eps_inf: float = 1.0
    wp_ev: float = Field(gt=0, description="Plasma frequency in eV.")
    gamma_ev: float = Field(ge=0, description="Damping rate in eV.")

    def eps(self, wavelength_nm):
        w = HC_EV_NM / np.asarray(wavelength_nm, dtype=float)
        return self.eps_inf - self.wp_ev**2 / (w**2 + 1j * self.gamma_ev * w)


class LorentzPole(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    delta_eps: float = Field(description="Oscillator strength (permittivity step).")
    omega0_ev: float = Field(gt=0, description="Resonance energy in eV.")
    gamma_ev: float = Field(ge=0, description="Damping in eV.")


class DrudeLorentz(_Material):
    """eps(w) = eps_inf - wp^2/(w^2 + i g w) + sum_j d_j w0_j^2 / (w0_j^2 - w^2 - i G_j w).

    The form FDTD needs for auxiliary-differential-equation dispersion.
    Fit one to measured data with :func:`lightsmith.materials_fit.fit_drude_lorentz`."""

    kind: Literal["drude_lorentz"] = "drude_lorentz"
    eps_inf: float = 1.0
    wp_ev: float = Field(0.0, ge=0)
    gamma_ev: float = Field(0.0, ge=0)
    poles: list[LorentzPole] = Field(default_factory=list)

    def eps(self, wavelength_nm):
        w = HC_EV_NM / np.asarray(wavelength_nm, dtype=float)
        e = self.eps_inf + 0j * w
        if self.wp_ev > 0:
            e = e - self.wp_ev**2 / (w**2 + 1j * self.gamma_ev * w)
        for p in self.poles:
            e = e + p.delta_eps * p.omega0_ev**2 / (p.omega0_ev**2 - w**2 - 1j * p.gamma_ev * w)
        return e


class Tabulated(_Material):
    """Tabulated n, k data, linearly interpolated (no extrapolation)."""

    kind: Literal["tabulated"] = "tabulated"
    wavelength_nm: list[float]
    n_real: list[float]
    n_imag: list[float]
    source: str | None = Field(None, description="Citation / provenance of the data.")

    @model_validator(mode="before")
    @classmethod
    def _sort_by_wavelength(cls, data: Any) -> Any:
        """Accept data in any order (energy-ordered files list wavelengths descending); store it ascending.

        Linear interpolation needs ascending, distinct wavelengths, and silently returns nonsense otherwise.
        """
        if not isinstance(data, dict) or not all(k in data for k in ("wavelength_nm", "n_real", "n_imag")):
            return data
        wl, n, k = (np.asarray(data[key], dtype=float) for key in ("wavelength_nm", "n_real", "n_imag"))
        if not (wl.ndim == n.ndim == k.ndim == 1 and len(wl) == len(n) == len(k)):
            raise ValueError("wavelength_nm, n_real and n_imag must be lists of the same length")
        if len(wl) < 2:
            raise ValueError("tabulated data needs at least two wavelengths")
        if not np.all(np.isfinite(wl)) or np.any(wl <= 0):
            raise ValueError("wavelength_nm must be positive and finite")
        order = np.argsort(wl, kind="stable")
        wl = wl[order]
        if np.any(np.diff(wl) == 0):
            raise ValueError(f"duplicate wavelengths in tabulated data: {sorted({float(w) for w in wl[1:][np.diff(wl) == 0]})}")
        return {**data, "wavelength_nm": wl.tolist(), "n_real": n[order].tolist(), "n_imag": k[order].tolist()}

    @classmethod
    def from_csv(cls, path: str | Path, unit: Literal["nm", "um"] = "nm", source: str | None = None):
        """Load a 3-column CSV (wavelength, n, k), e.g. a refractiveindex.info export."""
        data = np.loadtxt(path, delimiter=",", comments="#", skiprows=0, ndmin=2)
        wl = data[:, 0] * (1000.0 if unit == "um" else 1.0)
        return cls(wavelength_nm=wl.tolist(), n_real=data[:, 1].tolist(),
                   n_imag=data[:, 2].tolist(), source=source or str(path))

    @property
    def range_nm(self) -> tuple[float, float]:
        return min(self.wavelength_nm), max(self.wavelength_nm)

    def eps(self, wavelength_nm):
        wl = np.asarray(wavelength_nm, dtype=float)
        lo, hi = self.range_nm
        if np.any((wl < lo) | (wl > hi)):
            raise MaterialRangeError(
                f"wavelength outside tabulated range [{lo:g}, {hi:g}] nm "
                f"(requested {wl.min():g}-{wl.max():g} nm); no extrapolation is done")
        n = np.interp(wl, self.wavelength_nm, self.n_real)
        k = np.interp(wl, self.wavelength_nm, self.n_imag)
        return (n + 1j * k) ** 2


# --- refractiveindex.info YAML ------------------------------------------------

RII_REPOSITORY = "https://github.com/polyanskiy/refractiveindex.info-database"
RII_RAW_URL = "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/{ref}/database/{path}"
RII_PAGE_URL = "https://refractiveindex.info/?shelf={shelf}&book={book}&page={page}"
RII_CITATION = "M. N. Polyanskiy, Sci. Data 11, 94 (2024), doi:10.1038/s41597-023-02898-2"
_FORMULA_SAMPLES = 400


def _yaml_scalar(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        q = s[0]
        s = s[1:-1]
        s = s.replace("''", "'") if q == "'" else s.replace('\\"', '"')
    return s


def _parse_rii_yaml(text: str) -> dict:
    """Parse the YAML subset used by refractiveindex.info entries (stdlib only).

    Supports nested mappings, block literals (``|``), lists of mappings and
    plain/quoted scalars, which covers every entry in the database."""
    toks: list[list] = []
    for raw in text.replace("\t", "    ").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            toks.append([len(raw) - len(raw.lstrip(" ")), s])

    def node(i: int, indent: int):
        if toks[i][1].startswith("- ") or toks[i][1] == "-":
            out = []
            while i < len(toks) and toks[i][0] == indent and toks[i][1].startswith("-"):
                rest = toks[i][1][1:].strip()
                if not rest:
                    item, i = node(i + 1, toks[i + 1][0])
                else:
                    col = indent + len(toks[i][1]) - len(rest)  # column of the text after "- "
                    toks[i] = [col, rest]
                    item, i = node(i, col) if ":" in rest else (_yaml_scalar(rest), i + 1)
                out.append(item)
            return out, i
        out: dict[str, Any] = {}
        while i < len(toks) and toks[i][0] == indent:
            key, sep, rest = toks[i][1].partition(":")
            if not sep:
                raise ValueError(f"unparseable YAML line: {toks[i][1]!r}")
            rest, i = rest.strip(), i + 1
            if rest[:1] in ("|", ">"):
                lines = []
                while i < len(toks) and toks[i][0] > indent:
                    lines.append(toks[i][1])
                    i += 1
                out[key.strip()] = ("\n" if rest[0] == "|" else " ").join(lines)
            elif rest:
                out[key.strip()] = _yaml_scalar(rest)
            elif i < len(toks) and (toks[i][0] > indent or (toks[i][0] == indent and toks[i][1].startswith("-"))):
                out[key.strip()], i = node(i, toks[i][0])
            else:
                out[key.strip()] = None
        return out, i

    if not toks:
        raise ValueError("empty YAML document")
    doc, _ = node(0, toks[0][0])
    return doc


def _floats(s) -> np.ndarray:
    return np.array([float(x) for x in str(s).replace(",", " ").split()], dtype=float)


def _rii_formula(num: int, c: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Refractive index n(lam) for refractiveindex.info formula types 1-9 (lam in um)."""
    c = np.concatenate([c, np.zeros(max(0, 17 - len(c)))])
    l2 = lam**2
    pairs = range(1, len(c) - 1, 2)
    if num == 1:  # Sellmeier
        n2 = 1 + c[0] + sum(c[i] * l2 / (l2 - c[i + 1] ** 2) for i in pairs if c[i] != 0)
    elif num == 2:  # Sellmeier-2
        n2 = 1 + c[0] + sum(c[i] * l2 / (l2 - c[i + 1]) for i in pairs if c[i] != 0)
    elif num == 3:  # polynomial
        n2 = c[0] + sum(c[i] * lam ** c[i + 1] for i in pairs if c[i] != 0)
    elif num == 4:  # RefractiveIndex.INFO
        n2 = c[0] + 0 * lam
        if c[1] != 0:
            n2 = n2 + c[1] * lam ** c[2] / (l2 - c[3] ** c[4])
        if c[5] != 0:
            n2 = n2 + c[5] * lam ** c[6] / (l2 - c[7] ** c[8])
        n2 = n2 + sum(c[i] * lam ** c[i + 1] for i in range(9, len(c) - 1, 2) if c[i] != 0)
    elif num == 5:  # Cauchy
        return c[0] + sum(c[i] * lam ** c[i + 1] for i in pairs if c[i] != 0)
    elif num == 6:  # gases
        return 1 + c[0] + sum(c[i] / (c[i + 1] - lam**-2) for i in pairs if c[i] != 0)
    elif num == 7:  # Herzberger
        d = 1 / (l2 - 0.028)
        return c[0] + c[1] * d + c[2] * d**2 + c[3] * l2 + c[4] * l2**2 + c[5] * l2**3
    elif num == 8:  # Retro
        x = c[0] + c[1] * l2 / (l2 - c[2]) + c[3] * l2
        n2 = (1 + 2 * x) / (1 - x)
    elif num == 9:  # Exotic
        n2 = c[0] + c[1] / (l2 - c[2]) + c[3] * (lam - c[4]) / ((lam - c[4]) ** 2 + c[5])
    else:
        raise ValueError(f"unsupported refractiveindex.info formula type {num}")
    return np.sqrt(n2 + 0 * lam)


def _rii_nk(doc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(wavelength_nm, n, k) arrays from a parsed refractiveindex.info entry.

    Formulas are sampled on a dense log-spaced grid. When n and k come from
    different blocks, the valid range is that of n and k is 0 outside its data."""
    n_tab = k_tab = None
    n_fun = None
    for entry in doc.get("DATA") or []:
        typ = str(entry.get("type", "")).split()
        if typ[:1] == ["tabulated"]:
            what = typ[1] if len(typ) > 1 else "nk"
            rows = [_floats(r) for r in str(entry["data"]).splitlines() if r.strip()]
            arr = np.array(rows, dtype=float)
            arr = arr[np.argsort(arr[:, 0], kind="stable")]
            arr = arr[np.concatenate([[True], np.diff(arr[:, 0]) > 0])]
            if what == "nk":
                n_tab, k_tab = arr[:, [0, 1]], arr[:, [0, 2]]
            elif what == "n":
                n_tab = arr[:, :2]
            elif what == "k":
                k_tab = arr[:, :2]
            else:
                raise ValueError(f"unsupported data type {entry['type']!r}")
        elif typ[:1] == ["formula"]:
            num, coeffs = int(typ[1]), _floats(entry["coefficients"])
            lo, hi = _floats(entry["wavelength_range"])[:2]
            n_fun = (num, coeffs, lo, hi)
        else:
            raise ValueError(f"unsupported data type {entry.get('type')!r}")
    if n_fun is not None:
        num, coeffs, lo, hi = n_fun
        lam = np.geomspace(lo, hi, _FORMULA_SAMPLES)
        if k_tab is not None:
            inside = k_tab[(k_tab[:, 0] > lo) & (k_tab[:, 0] < hi), 0]
            lam = np.unique(np.concatenate([lam, inside]))
        n = np.real(_rii_formula(num, coeffs, lam))
    elif n_tab is not None:
        lam, n = n_tab[:, 0], n_tab[:, 1]
    else:
        raise ValueError("entry has no refractive-index (n) data")
    if k_tab is None:
        k = np.zeros_like(lam)
    else:
        k = np.interp(lam, k_tab[:, 0], k_tab[:, 1], left=0.0, right=0.0)
    if not np.all(np.isfinite(n)):
        raise ValueError("formula gives non-finite n inside its stated wavelength range")
    return lam * 1000.0, n, k


def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "; ", s or "")
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return re.sub(r"\s+", " ", s).strip(" ;")


def _rii_reference(doc: dict) -> str:
    """Plain-text REFERENCES of an entry (HTML stripped)."""
    return _strip_html(str(doc.get("REFERENCES") or ""))


def _rii_dois(doc: dict) -> list[str]:
    refs = str(doc.get("REFERENCES") or "")
    return [urllib.parse.unquote(d) for d in re.findall(r"doi\.org/([^\"'<\s]+)", refs)]


def read_refractiveindex_yaml(path_or_text: str | Path, source: str | None = None) -> Tabulated:
    """Load one refractiveindex.info YAML entry (a file path or its text) as :class:`Tabulated`."""
    text = str(path_or_text)
    if isinstance(path_or_text, Path) or ("\n" not in text and Path(text).is_file()):
        text = Path(path_or_text).read_text(encoding="utf-8")
    doc = _parse_rii_yaml(text)
    wl, n, k = _rii_nk(doc)
    return Tabulated(wavelength_nm=wl.tolist(), n_real=n.tolist(), n_imag=k.tolist(),
                     source=source or _rii_reference(doc) or None)


# --- online access to the full database --------------------------------------

# Database names include Greek letters, umlauts and inner spaces ("Walling-α",
# "C6H12_n - polymethylpentene"); path separators, ":" and ".." are refused.
_SAFE_NAME = re.compile(r"^\w(?:[\w .,+()\-]*[\w.,+()\-])?$")


def default_cache_dir() -> Path:
    """User cache directory for downloaded data (``LIGHTSMITH_CACHE_DIR`` overrides)."""
    if os.environ.get("LIGHTSMITH_CACHE_DIR"):
        return Path(os.environ["LIGHTSMITH_CACHE_DIR"])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "lightsmith" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "lightsmith"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "lightsmith"


def _raw_url(ref: str, path: str) -> str:
    # Data paths may contain spaces and non-ASCII characters
    # ("organic/C50H80N2O4 - ProSQ-C16/nk/..."), so percent-encode them.
    return RII_RAW_URL.format(ref=urllib.parse.quote(ref, safe=""), path=urllib.parse.quote(path, safe="/"))


def _download(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "lightsmith (+https://github.com/lightsmith)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https host)
        return r.read()


def _catalog_path(shelf: str, book: str, page: str, cache: Path, ref: str, timeout: float,
                  refresh: bool = False) -> str | None:
    """Look up the data path of shelf/book/page in catalog-nk.yml (cached; re-downloaded
    once when ``refresh`` is set or the entry is missing from a cached copy)."""
    cat = cache / "catalog-nk.yml"
    fresh = refresh or not cat.is_file()
    if fresh:
        cat.parent.mkdir(parents=True, exist_ok=True)
        cat.write_bytes(_download(_raw_url(ref, "catalog-nk.yml"), timeout))
    found = _catalog_lookup(cat, shelf, book, page)
    if found is None and not fresh:
        return _catalog_path(shelf, book, page, cache, ref, timeout, refresh=True)
    return found


def _catalog_lookup(cat: Path, shelf: str, book: str, page: str) -> str | None:
    cur_shelf = cur_book = cur_page = None
    for line in cat.read_text(encoding="utf-8").splitlines():
        s = line.strip().removeprefix("- ")
        key, _, val = s.partition(":")
        val = _yaml_scalar(val)
        if key == "SHELF":
            cur_shelf, cur_book, cur_page = val, None, None
        elif key == "BOOK":
            cur_book, cur_page = val, None
        elif key == "PAGE":
            cur_page = val
        elif key == "data" and (cur_shelf, cur_book, cur_page) == (shelf, book, page):
            return val
    return None


def fetch_refractiveindex(shelf: str, book: str, page: str, *, cache_dir: str | Path | None = None,
                          refresh: bool = False, ref: str = "main", timeout: float = 30.0) -> Tabulated:
    """Download one refractiveindex.info entry (e.g. ``"main", "Au", "Olmon-sc"``) as :class:`Tabulated`.

    Names are those in the refractiveindex.info URL (``?shelf=main&book=Au&page=Olmon-sc``).
    The YAML is cached under :func:`default_cache_dir` (or ``cache_dir``); later calls
    work offline. ``ref`` is a branch, tag or commit of the database repository.
    Tabulated n/k and formula types 1-9 are supported. The database is CC0; cite the
    dataset's own reference (kept in ``.source``) and Polyanskiy, Sci. Data 11, 94 (2024)."""
    for label, v in (("shelf", shelf), ("book", book), ("page", page), ("ref", ref)):
        if not _SAFE_NAME.match(v) or ".." in v:
            raise ValueError(f"invalid {label} name {v!r}")
    cache = (Path(cache_dir) if cache_dir else default_cache_dir()) / "refractiveindex.info" / ref
    target = cache / shelf / book / f"{page}.yml"
    path = f"data/{shelf}/{book}/nk/{page}.yml"
    if refresh or not target.is_file():
        try:
            try:
                text = _download(_raw_url(ref, path), timeout)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                found = _catalog_path(shelf, book, page, cache, ref, timeout, refresh=refresh)
                if found is None:
                    raise LookupError(
                        f"refractiveindex.info has no entry shelf={shelf!r} book={book!r} page={page!r}; "
                        f"browse {RII_REPOSITORY}/tree/main/database/data") from None
                path = f"data/{found}"
                text = _download(_raw_url(ref, path), timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ConnectionError(
                f"could not download {shelf}/{book}/{page} from refractiveindex.info ({e}). "
                "Offline? Use a bundled dataset instead: see lightsmith.materials.list_materials()") from e
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text)
    doc = _parse_rii_yaml(target.read_text(encoding="utf-8"))
    wl, n, k = _rii_nk(doc)
    q = urllib.parse.quote
    url = RII_PAGE_URL.format(shelf=q(shelf, safe=""), book=q(book, safe=""), page=q(page, safe=""))
    source = f"{_rii_reference(doc)} Data: refractiveindex.info database (CC0 1.0), {url}; cite also {RII_CITATION}."
    return Tabulated(wavelength_nm=wl.tolist(), n_real=n.tolist(), n_imag=k.tolist(), source=source)


# --- bundled material library -------------------------------------------------


def _data_dir():
    return resources.files("lightsmith").joinpath("data", "materials")


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads(_data_dir().joinpath("manifest.json").read_text(encoding="utf-8"))


def _lookup(name: str, dataset: str | None) -> tuple[str, str]:
    """Canonical (material, dataset) keys for possibly aliased names; clear errors otherwise."""
    mats = _manifest()["materials"]
    canon = {k.lower(): k for k in mats}
    for k, m in mats.items():
        canon.update({a.lower(): k for a in m.get("aliases", [])})
    mat = canon.get(str(name).strip().lower())
    if mat is None:
        raise ValueError(f"unknown library material {name!r}; available: {', '.join(mats)} "
                         "(see list_materials(), or fetch_refractiveindex() for any refractiveindex.info entry)")
    m = mats[mat]
    if dataset is None:
        return mat, m["default"]
    sets = {k.lower(): k for k in m["datasets"]}
    for k, d in m["datasets"].items():
        sets.update({a.lower(): k for a in d.get("aliases", [])})
    key = sets.get(str(dataset).strip().lower())
    if key is None:
        raise ValueError(f"unknown dataset {dataset!r} for {mat}; available: {', '.join(m['datasets'])}")
    return mat, key


@lru_cache(maxsize=None)
def _library_tabulated(material: str, dataset: str) -> Tabulated:
    info = _manifest()["materials"][material]["datasets"][dataset]
    text = _data_dir().joinpath(*info["file"].split("/")).read_text(encoding="utf-8")
    wl, n, k = _rii_nk(_parse_rii_yaml(text))
    return Tabulated(wavelength_nm=wl.tolist(), n_real=n.tolist(), n_imag=k.tolist(), source=_source_text(info))


def _source_text(info: dict) -> str:
    man = _manifest()
    return (f"{info['reference']} Data: refractiveindex.info database ({man['license']}), {info['url']} "
            f"(snapshot {man['commit'][:10]}); cite also {RII_CITATION}.")


def list_materials(name: str | None = None) -> list[dict]:
    """Bundled datasets (no network needed): one dict per dataset with its name, range and citation.

    Each row has ``material_json``, ready to drop into a :class:`Simulation`."""
    out = []
    for mat, m in _manifest()["materials"].items():
        if name is not None and _lookup(name, None)[0] != mat:
            continue
        for key, d in m["datasets"].items():
            out.append({
                "name": mat, "description": m["description"], "aliases": m.get("aliases", []),
                "dataset": key, "default": key == m["default"], "dataset_aliases": d.get("aliases", []),
                "range_nm": d["range_nm"], "data_type": d["type"], "has_k": d["has_k"],
                "comments": d["comments"], "reference": d["reference"], "doi": d["doi"],
                "url": d["url"], "source_url": d["source_url"], "license": _manifest()["license"],
                "material_json": {"kind": "library", "name": mat, "dataset": key},
            })
    return out


def load_material(name: str, dataset: str | None = None) -> Tabulated:
    """Bundled dataset as a :class:`Tabulated` (e.g. ``load_material("Au", "Johnson-Christy")``)."""
    return _library_tabulated(*_lookup(name, dataset))


class LibraryMaterial(_Material):
    """Measured n, k bundled with lightsmith, e.g. ``{"kind": "library", "name": "Au",
    "dataset": "Johnson-Christy"}``. Linear interpolation in n and k, no extrapolation.

    Names and datasets accept aliases ("gold", "JC") and are stored in canonical form;
    an omitted dataset is filled with the material's default so the spec stays
    reproducible. See :func:`list_materials`."""

    kind: Literal["library"] = "library"
    name: str = Field(description="Material, e.g. 'Au', 'Ag', 'Al', 'Cu', 'Si', 'TiO2', 'SiO2', 'H2O'.")
    dataset: str | None = Field(None, description="Dataset, e.g. 'Johnson-Christy'; omitted = material default.")

    @model_validator(mode="before")
    @classmethod
    def _canonical(cls, data):
        if isinstance(data, dict) and "name" in data:
            mat, key = _lookup(data["name"], data.get("dataset"))
            data = {**data, "name": mat, "dataset": key}
        return data

    def tabulated(self) -> Tabulated:
        return _library_tabulated(self.name, self.dataset)

    @property
    def range_nm(self) -> tuple[float, float]:
        return self.tabulated().range_nm

    @property
    def source(self) -> str:
        return self.tabulated().source

    def eps(self, wavelength_nm):
        wl = np.asarray(wavelength_nm, dtype=float)
        lo, hi = self.range_nm
        if wl.size and (wl.min() < lo or wl.max() > hi):
            raise MaterialRangeError(self._range_message(float(wl.min()), float(wl.max())))
        return self.tabulated().eps(wl)

    def _range_message(self, wmin: float, wmax: float) -> str:
        lo, hi = self.range_nm
        msg = (f"{self.name} ({self.dataset}) has data for {lo:g}-{hi:g} nm only; "
               f"requested {wmin:g}-{wmax:g} nm.")
        rows = [r for r in list_materials(self.name) if r["dataset"] != self.dataset]
        cover = [r for r in rows if r["range_nm"][0] <= wmin and r["range_nm"][1] >= wmax]
        if cover:
            alt = ", ".join(f"'{r['dataset']}' ({r['range_nm'][0]:g}-{r['range_nm'][1]:g} nm)" for r in cover)
            msg += f" Datasets covering it: {alt}."
        else:
            msg += (" No bundled dataset covers it; restrict the wavelengths, or use "
                    "fetch_refractiveindex(shelf, book, page) for other refractiveindex.info data.")
        return msg


Material = Annotated[
    Union[ConstantIndex, ConstantPermittivity, Drude, DrudeLorentz, Tabulated, LibraryMaterial],
    Field(discriminator="kind"),
]
