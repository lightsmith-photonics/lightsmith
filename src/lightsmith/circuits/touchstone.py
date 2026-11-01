"""Touchstone (.sNp) reading and writing, plus SParameterResult JSON loading.

Touchstone files use the engineering time convention exp(+j omega t), while
lightsmith uses exp(-i omega t); :func:`read_touchstone` and
:func:`write_touchstone` conjugate at the boundary by default
(``time_convention="+iwt"``). Pass ``"-iwt"`` for files known to be written in
the physics convention.

Supported: Touchstone 1.x (``# <unit> S <RI|MA|DB> R <z0>``, any port count,
two-port ``S11 S21 S12 S22`` order) and the common Touchstone 2.0 keywords
(``[Version]``, ``[Number of Ports]``, ``[Two-Port Data Order]``,
``[Network Data]``, full matrix format). Port names are read from
``! Port[n] = name`` comments when present (as written by
:func:`write_touchstone`, scikit-rf and others); otherwise ports are ``o1..oN``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np

C0_M_S = 299_792_458.0
_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}
TimeConvention = Literal["+iwt", "-iwt"]


def _ports_from_name(path: Path) -> int | None:
    m = re.search(r"\.s(\d+)p$", path.name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def read_touchstone(path: str | Path, time_convention: TimeConvention = "+iwt"
                    ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read a Touchstone file. Returns (wavelengths_nm ascending, port names, S[wl, out, in]).

    S is returned in lightsmith's exp(-i omega t) convention (conjugated when the
    file uses ``time_convention="+iwt"``, the Touchstone standard)."""
    path = Path(path)
    n_ports = _ports_from_name(path)
    unit, fmt = "ghz", "ma"
    order_21_12 = True  # v1 two-port order: S11 S21 S12 S22
    names: dict[int, str] = {}
    numbers: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line, _, comment = raw.partition("!")
        m = re.match(r"\s*port\s*\[\s*(\d+)\s*\]\s*=\s*(.+?)\s*$", comment, flags=re.IGNORECASE)
        if m:
            names[int(m.group(1))] = m.group(2)
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            toks = line[1:].lower().split()
            i = 0
            while i < len(toks):
                t = toks[i]
                if t in _UNITS:
                    unit = t
                elif t in ("ri", "ma", "db"):
                    fmt = t
                elif t == "r" and i + 1 < len(toks):
                    i += 1  # reference impedance: informational (photonic S is mode-normalised)
                elif t != "s":
                    raise ValueError(f"{path.name}: only S-parameter Touchstone files are supported (option {t!r})")
                i += 1
            continue
        if line.startswith("["):
            key, _, val = line[1:].partition("]")
            key, val = key.strip().lower(), val.strip()
            if key == "number of ports":
                n_ports = int(val)
            elif key == "two-port data order":
                order_21_12 = val.replace(" ", "") == "21_12"
            elif key == "matrix format" and val.lower() != "full":
                raise ValueError(f"{path.name}: only [Matrix Format] Full is supported, got {val!r}")
            continue
        numbers.extend(float(t) for t in line.split())
    if n_ports is None:
        raise ValueError(f"{path.name}: cannot tell the port count (use a .sNp extension or [Number of Ports])")
    per = 1 + 2 * n_ports**2
    data = np.asarray(numbers, dtype=float)
    if n_ports == 2:
        # Drop trailing two-port noise data: it starts where the frequency stops increasing.
        rows = []
        for s in range(0, len(data) - per + 1, per):
            if rows and data[s] <= rows[-1][0]:
                break
            rows.append(data[s:s + per])
        data = np.asarray(rows)
    else:
        if len(data) % per:
            raise ValueError(f"{path.name}: {len(data)} numbers is not a multiple of {per} "
                             f"(1 frequency + {n_ports}x{n_ports} complex values)")
        data = data.reshape(-1, per)
    if data.size == 0:
        raise ValueError(f"{path.name}: no network data")
    f = data[:, 0] * _UNITS[unit]
    a, b = data[:, 1::2], data[:, 2::2]
    if fmt == "ri":
        v = a + 1j * b
    elif fmt == "ma":
        v = a * np.exp(1j * np.radians(b))
    else:
        v = 10 ** (a / 20) * np.exp(1j * np.radians(b))
    s = v.reshape(-1, n_ports, n_ports)  # row-major [out][in]
    if n_ports == 2 and order_21_12:
        s = s.transpose(0, 2, 1)  # S11 S21 S12 S22 is column-major
    if time_convention == "+iwt":
        s = np.conj(s)
    if np.any(f <= 0):
        raise ValueError(f"{path.name}: frequencies must be positive to convert to wavelength")
    wl = C0_M_S / f * 1e9
    idx = np.argsort(wl)
    ports = [names.get(i + 1, f"o{i + 1}") for i in range(n_ports)]
    return wl[idx], ports, s[idx]


def write_touchstone(path: str | Path, wavelengths_nm, s, ports: list[str] | None = None,
                     time_convention: TimeConvention = "+iwt") -> Path:
    """Write S[wl, out, in] (exp(-i omega t) convention) as a Touchstone 1.1 RI file in Hz.

    The extension should be ``.sNp`` with N the port count. Port names go into
    ``! Port[n] = name`` comments so :func:`read_touchstone` restores them."""
    path = Path(path)
    s = np.asarray(s, dtype=complex)
    wl = np.asarray(wavelengths_nm, dtype=float)
    if s.ndim != 3 or s.shape[1] != s.shape[2] or s.shape[0] != wl.size:
        raise ValueError(f"S must have shape (n_wavelengths, N, N); got {s.shape} for {wl.size} wavelengths")
    n = s.shape[1]
    expected = _ports_from_name(path)
    if expected is not None and expected != n:
        raise ValueError(f"file name {path.name!r} implies {expected} ports but S has {n}")
    if time_convention == "+iwt":
        s = np.conj(s)
    f = C0_M_S / (wl * 1e-9)
    idx = np.argsort(f)
    out = ["! Written by lightsmith (S-parameters of a photonic circuit/device)",
           f"! Time convention of the data: exp({'+j' if time_convention == '+iwt' else '-i'} omega t)"]
    for i, p in enumerate(ports or [f"o{i + 1}" for i in range(n)]):
        out.append(f"! Port[{i + 1}] = {p}")
    out.append("# Hz S RI R 50")
    for k in idx:
        m = s[k].T if n == 2 else s[k]  # two-port order S11 S21 S12 S22
        vals = [x for z in m.reshape(-1) for x in (z.real, z.imag)]
        if n <= 2:
            out.append(" ".join(f"{v:.12e}" for v in [f[k], *vals]))
        else:
            for r in range(n):
                row = vals[2 * n * r:2 * n * (r + 1)]
                chunks = [row[c:c + 8] for c in range(0, len(row), 8)]  # at most 4 complex values per line
                for ci, ch in enumerate(chunks):
                    lead = f"{f[k]:.12e} " if (r == 0 and ci == 0) else " " * 19
                    out.append(lead + " ".join(f"{v:.12e}" for v in ch))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def read_sparameter_json(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read an lightsmith ``SParameterResult`` JSON file (already exp(-i omega t))."""
    from ..results import SParameterResult

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "result" in doc and isinstance(doc["result"], dict):
        doc = doc["result"]
    r = SParameterResult.model_validate(doc)
    wl = np.asarray(r.wavelengths_nm, dtype=float)
    idx = np.argsort(wl)
    return wl[idx], list(r.ports), r.s()[idx]


def read_sparameters(path: str | Path, time_convention: TimeConvention = "+iwt"
                     ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read S-parameters from a Touchstone (.sNp/.ts) or SParameterResult JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"S-parameter file not found: {path}")
    if path.suffix.lower() == ".json":
        return read_sparameter_json(path)
    return read_touchstone(path, time_convention)
