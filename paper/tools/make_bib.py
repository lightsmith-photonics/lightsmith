"""Build paper/references.bib from registry metadata only (no hand-typed bibliographic fields).

Every entry is fetched from doi.org (CSL-JSON content negotiation, which covers Crossref and
DataCite) and/or the arXiv API, then checked against the first author's surname and year we
expect. Entries that fail to resolve or do not match are written *commented out* with the reason,
so nothing unverified reaches the bibliography.

    uv run --no-sync python paper/tools/make_bib.py          # needs network access

Writes paper/references.bib and paper/tools/bib_verification.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent

# key, doi, arxiv, expected first-author surname (substring, accent-insensitive), expected year (+-1)
ENTRIES = [
    # DDA
    ("Purcell1973", "10.1086/152538", None, "Purcell", 1973),
    ("Draine1994", "10.1364/JOSAA.11.001491", None, "Draine", 1994),
    ("Goodman1991", "10.1364/OL.16.001198", None, "Goodman", 1991),
    ("DraineGoodman1993", "10.1086/172396", None, "Draine", 1993),
    ("Yurkin2007review", "10.1016/j.jqsrt.2007.01.034", None, "Yurkin", 2007),
    ("Yurkin2011adda", "10.1016/j.jqsrt.2011.01.031", None, "Yurkin", 2011),
    ("Yurkin2006extrap", "10.1364/JOSAA.23.002578", None, "Yurkin", 2006),
    ("Yurkin2010gold", "10.1117/1.3335329", None, "Yurkin", 2010),
    ("DraineFlatau2013guide", None, "1305.6497", "Draine", 2013),
    ("Argentin2026ddabench", None, "2603.02871", "Argentin", 2026),
    ("Xu2025cpdda", "10.3390/nano15070500", None, "Xu", 2025),
    ("Ponomareva2025torchgdm", "10.21468/SciPostPhysCodeb.60", "2505.09545", "Ponomareva", 2025),
    ("Wiecha2018pygdm", None, "1802.04071", "Wiecha", 2018),
    # Mie / T-matrix
    ("Mie1908", "10.1002/andp.19083300302", None, "Mie", 1908),
    ("BohrenHuffman1998", "10.1002/9783527618156", None, "Bohren", 1998),
    ("Wiscombe1980", "10.1364/AO.19.001505", None, "Wiscombe", 1980),
    ("Prahl2023miepython", "10.5281/zenodo.7949263", None, "Prahl", None),
    ("Waterman1965", "10.1109/PROC.1965.4058", None, "Waterman", 1965),
    ("Beutel2024treams", "10.1016/j.cpc.2023.109076", None, "Beutel", 2024),
    ("Egel2021smuthi", "10.1016/j.jqsrt.2021.107846", "2105.04259", "Egel", 2021),
    ("Somerville2016smarties", "10.1016/j.jqsrt.2016.01.005", "1511.00798", "Somerville", 2016),
    ("Asadova2025tmatrixformat", "10.1016/j.jqsrt.2024.109310", "2408.10727", "Asadova", 2025),
    ("PoinsinetdeSivry2023pymiesim", "10.1364/OPTCON.473102", None, "Poinsinet", 2023),
    ("Yang2003", "10.1364/AO.42.001710", None, "Yang", 2003),
    ("PenaPal2009", "10.1016/j.cpc.2009.07.010", None, "Pe", 2009),
    ("Mishchenko1991", "10.1364/JOSAA.8.000871", None, "Mishchenko", 1991),
    ("MeierWokaun1983", "10.1364/OL.8.000581", None, "Meier", 1983),
    ("Kelly2003", "10.1021/jp026731y", None, "Kelly", 2003),
    # layered media, periodic, bands, inverse design
    ("KatsidisSiapkas2002", "10.1364/AO.41.003978", None, "Katsidis", 2002),
    ("Moharam1995", "10.1364/JOSAA.12.001068", None, "Moharam", 1995),
    ("Li1996", "10.1364/JOSAA.13.001870", None, "Li", 1996),
    ("Schuster2007", "10.1364/JOSAA.24.002880", None, "Schuster", 2007),
    ("Ho1990", "10.1103/PhysRevLett.65.3152", None, "Ho", 1990),
    ("JohnsonJoannopoulos2001mpb", "10.1364/OE.8.000173", None, "Johnson", 2001),
    ("BendsoeSigmund2004", "10.1007/978-3-662-05086-6", None, "Bends", 2004),
    ("Molesky2018", "10.1038/s41566-018-0246-9", None, "Molesky", 2018),
    ("Hughes2018adjoint", "10.1021/acsphotonics.8b01522", None, "Hughes", 2018),
    ("Mekis1996", "10.1103/PhysRevLett.77.3787", None, "Mekis", 1996),
    ("SoldanoPennings1995", "10.1109/50.372474", None, "Soldano", 1995),
    # FDTD
    ("Yee1966", "10.1109/TAP.1966.1138693", None, "Yee", 1966),
    ("Roden2000cpml", "10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A", None, "Roden", 2000),
    ("Oskooi2010meep", "10.1016/j.cpc.2009.11.008", None, "Oskooi", 2010),
    ("Farjadpour2006", "10.1364/OL.31.002972", None, "Farjadpour", 2006),
    ("Mahlau2026fdtdx", "10.21105/joss.08912", None, "Mahlau", 2026),
    # materials
    ("JohnsonChristy1972", "10.1103/PhysRevB.6.4370", None, "Johnson", 1972),
    ("Rakic1998", "10.1364/AO.37.005271", None, "Raki", 1998),
    ("McPeak2015", "10.1021/ph5004237", None, "McPeak", 2015),
    ("Polyanskiy2024", "10.1038/s41597-023-02898-2", None, "Polyanskiy", 2024),
    # cross-method benchmarks
    ("WriedtComberg1998", "10.1016/S0022-4073(98)00016-8", None, "Wriedt", 1998),
    ("Yurkin2007ddafdtd", "10.1364/OE.15.017902", None, "Yurkin", 2007),
    ("Myroshnychenko2008", "10.1039/B711486A", None, "Myroshnychenko", 2008),
    ("Hoffmann2009", None, "0907.3570", "Hoffmann", 2009),
    ("Parsons2010", "10.1080/09500341003628702", None, "Parsons", 2010),
    ("Loke2014", "10.1007/s00340-013-5594-z", None, "Loke", 2014),
    ("CalaLesina2015", "10.1364/OE.23.010481", None, "Lesina", 2015),
    ("Gallinet2015", "10.1002/lpor.201500122", None, "Gallinet", 2015),
    ("Lalanne2019", "10.1364/JOSAA.36.000686", "1811.11751", "Lalanne", 2019),
    ("Dieperink2024", "10.1515/nanoph-2024-0238", "2410.08093", "Dieperink", 2024),
    # software
    ("Harris2020numpy", "10.1038/s41586-020-2649-2", None, "Harris", 2020),
    ("Paszke2019pytorch", None, "1912.01703", "Paszke", 2019),
    # agents
    ("Lupoiu2025metachat", "10.1126/sciadv.adx8006", "2503.20479", "Lupoiu", 2025),
    ("Kim2025llmphotonics", "10.1515/nanoph-2024-0674", None, "Kim", 2025),
    ("Huang2025mcp", None, "2508.10277", None, 2025),
    ("Sharma2025phido", None, "2508.14123", "Sharma", 2025),
    ("Wu2025picbench", None, "2502.03159", None, 2025),
    ("Picopilot2026", None, "2608.01791", None, 2026),
    ("Kharel2026", None, "2606.00915", "Kharel", 2026),
    ("Halo2026", None, "2608.28877", None, 2026),
    ("Biswas2026", None, "2606.22493", "Biswas", 2026),
    ("Mudur2025feabench", None, "2504.06260", None, 2025),
    ("Fermilink2026", None, "2604.03460", None, 2026),
]

# Web resources: cited as @misc only if the URL resolves (HTTP 200) when this script runs.
URLS = [
    ("MCP2025spec", "Model Context Protocol specification", "https://modelcontextprotocol.io/specification"),
    ("ArrayAPI", "Python array API standard", "https://data-apis.org/array-api/latest/"),
    ("Pydantic", "Pydantic", "https://docs.pydantic.dev/latest/"),
    ("SiPkitNeff", "SiPkit documentation: effective index of 220 nm SOI strip waveguides",
     "https://sipkit.readthedocs.io/en/latest/1-%20Effective%20Index.html"),
    ("MPBTutorial", "MPB Python tutorial (square lattice of rods, TM gap)",
     "https://mpb.readthedocs.io/en/latest/Python_Tutorial/"),
    ("refractiveindexinfo", "refractiveindex.info database",
     "https://github.com/polyanskiy/refractiveindex.info-database"),
]

UA = {"User-Agent": "omniscatter-paper-bib/0.1 (mailto:noreply@example.org)"}


def _get(url, accept=None, timeout=30):
    req = urllib.request.Request(url, headers={**UA, **({"Accept": accept} if accept else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace"), r.status


def _fold(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def _tex(s):
    s = re.sub(r"<[^>]+>", "", s or "")  # strip JATS/MathML tags
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def doi_meta(doi):
    text, _ = _get("https://doi.org/" + urllib.parse.quote(doi, safe="/:;()<>.-"),
                   accept="application/vnd.citationstyles.csl+json")
    return json.loads(text)


def arxiv_meta(aid):
    text, _ = _get(f"https://export.arxiv.org/api/query?id_list={aid}")
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(text).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        raise ValueError("arXiv id not found")
    return {
        "title": " ".join(e.find("a:title", ns).text.split()),
        "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
        "year": int(e.find("a:published", ns).text[:4]),
        "primary": (e.find("x:primary_category", ns).attrib.get("term") if e.find("x:primary_category", ns)
                    is not None else ""),
        "doi": (e.find("x:doi", ns).text if e.find("x:doi", ns) is not None else None),
        "journal_ref": (e.find("x:journal_ref", ns).text if e.find("x:journal_ref", ns) is not None else None),
    }


def _year(m):
    for k in ("published-print", "published", "issued", "published-online", "created"):
        dp = (m.get(k) or {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            return int(dp[0][0])
    return None


def _names(authors):
    out = []
    for a in authors:
        if "family" in a:
            out.append(f"{a['family']}, {a.get('given', '')}".strip(", "))
        elif "literal" in a:
            out.append("{" + a["literal"] + "}")
        elif "name" in a:
            out.append("{" + a["name"] + "}")
    return " and ".join(_tex(n) for n in out)


def bib_from_doi(key, m, eprint=None):
    t = m.get("type", "")
    kind = {"article-journal": "article", "journal-article": "article", "book": "book",
            "chapter": "incollection", "paper-conference": "inproceedings", "dataset": "misc",
            "software": "misc"}.get(t, "misc" if "zenodo" in m.get("DOI", "").lower() else "article")
    title = m.get("title")
    title = title[0] if isinstance(title, list) else title
    cont = m.get("container-title")
    cont = cont[0] if isinstance(cont, list) and cont else (cont if isinstance(cont, str) else "")
    f = {"author": _names(m.get("author", [])), "title": "{" + _tex(title) + "}", "year": _year(m)}
    if kind == "article":
        f["journal"] = _tex(cont)
        f["volume"] = m.get("volume")
        f["number"] = m.get("issue")
        f["pages"] = (m.get("page") or m.get("article-number") or "").replace("-", "--")
    elif kind == "book":
        f["publisher"] = _tex(m.get("publisher"))
    elif kind in ("incollection", "inproceedings"):
        f["booktitle"] = _tex(cont)
        f["pages"] = (m.get("page") or "").replace("-", "--")
        f["publisher"] = _tex(m.get("publisher"))
    else:
        f["howpublished"] = _tex(m.get("publisher") or cont)
        if m.get("version"):
            f["note"] = "Version " + str(m["version"])
    f["doi"] = m.get("DOI", "")
    if eprint:
        f["eprint"], f["archivePrefix"] = eprint, "arXiv"
    return kind, f


def bib_from_arxiv(key, a, aid):
    f = {"author": " and ".join(_tex(n) for n in a["authors"]), "title": "{" + _tex(a["title"]) + "}",
         "year": a["year"], "eprint": aid, "archivePrefix": "arXiv", "primaryClass": a["primary"],
         "journal": f"arXiv:{aid}"}
    if a.get("doi"):
        f["doi"] = a["doi"]
    if a.get("journal_ref"):
        f["note"] = _tex(a["journal_ref"])
    return "article", f


def render(key, kind, fields, comment=False):
    body = ",\n".join(f"  {k} = {{{v}}}" if not str(v).startswith("{") else f"  {k} = {v}"
                      for k, v in fields.items() if v not in (None, ""))
    s = f"@{kind}{{{key},\n{body}\n}}\n"
    return "".join("% " + line + "\n" for line in s.splitlines()) if comment else s


def check(expected_name, expected_year, authors_text, year):
    probs = []
    if expected_name and _fold(expected_name) not in _fold(authors_text.split(" and ")[0] if authors_text else ""):
        probs.append(f"first author mismatch (expected {expected_name!r}, got {authors_text.split(' and ')[0]!r})")
    if expected_year and year and abs(int(year) - expected_year) > 1:
        probs.append(f"year mismatch (expected {expected_year}, got {year})")
    return probs


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out, log = [], []
    hdr = ("% references.bib -- generated by paper/tools/make_bib.py from doi.org (CSL-JSON) and the arXiv API.\n"
           "% Do not hand-edit bibliographic fields. Entries that failed verification are commented out\n"
           "% with the reason. Verification log: paper/tools/bib_verification.json\n\n")
    for key, doi, aid, name, year in ENTRIES:
        rec = {"key": key, "doi": doi, "arxiv": aid}
        try:
            arx = arxiv_meta(aid) if aid else None
            if doi:
                m = doi_meta(doi)
                kind, f = bib_from_doi(key, m, eprint=aid if arx else None)
            else:
                kind, f = bib_from_arxiv(key, arx, aid)
            probs = check(name, year, f["author"], f["year"])
            if arx and doi:
                probs += check(name, None, " and ".join(arx["authors"]), None)
            rec.update(status="verified" if not probs else "mismatch", problems=probs,
                       title=f["title"].strip("{}"), year=f["year"], first_author=f["author"].split(" and ")[0])
            out.append(("% UNVERIFIED: " + "; ".join(probs) + "\n" if probs else "") + render(key, kind, f, bool(probs)))
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed", problems=[f"{type(e).__name__}: {e}"])
            out.append(f"% UNVERIFIED ({type(e).__name__}: {e}): {key} doi={doi} arXiv={aid}\n")
        log.append(rec)
        print(f"{rec['status']:9s} {key}: {rec.get('first_author', '')} {rec.get('year', '')} "
              f"{rec.get('title', '')[:70]} {rec.get('problems') or ''}", flush=True)
        time.sleep(0.4)
    for key, title, url in URLS:
        rec = {"key": key, "url": url}
        try:
            _, status = _get(url)
            ok = status == 200
        except Exception as e:  # noqa: BLE001
            ok, status = False, f"{type(e).__name__}: {e}"
        rec.update(status="verified" if ok else "failed", http=str(status))
        f = {"title": "{" + title + "}", "howpublished": "\\url{" + url + "}", "note": "Accessed 2026-11-01"}
        out.append(render(key, "misc", f, not ok))
        log.append(rec)
        print(f"{rec['status']:9s} {key}: {url}")
    (PAPER / "references.bib").write_text(hdr + "\n".join(out), encoding="utf-8")
    (HERE / "bib_verification.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    bad = [r["key"] for r in log if r["status"] != "verified"]
    print(f"{len(log) - len(bad)} verified, {len(bad)} commented out: {bad}")


if __name__ == "__main__":
    sys.exit(main())
