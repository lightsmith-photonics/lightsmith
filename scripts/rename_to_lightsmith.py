"""One-shot rename of omniscatter -> lightsmith across the library, plugins and docs.

Run from the repo root once no agent is editing the library:
    uv run python scripts/rename_to_lightsmith.py          # dry run, lists changes
    uv run python scripts/rename_to_lightsmith.py --apply  # rewrite and move

apps/ and services/ are skipped: their sessions own them and switch imports themselves.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "apps", "services", "results"}
TEXT_SUFFIXES = {".py", ".toml", ".md", ".txt", ".json", ".yml", ".yaml", ".cfg", ".tex", ".bib", ".lock"}

# Order matters: specific forms first.
REPLACEMENTS = [
    (r"omniscatter-scattnlay", "lightsmith-scattnlay"),
    (r"omniscatter_scattnlay", "lightsmith_scattnlay"),
    (r"omniscatter-adda", "lightsmith-adda"),
    (r"omniscatter_adda", "lightsmith_adda"),
    (r"omniscatter-mcp", "lightsmith-mcp"),
    (r"omniscatter-server", "lightsmith-server"),
    (r"omniscatter-(dda|mie|fdtd|fdfd|fde|eme|tmm|rcwa|pwe|circuit|quasistatic)", r"lightsmith-\1"),
    (r"OMNISCATTER_", "LIGHTSMITH_"),
    (r"github\.com/omniscatter/omniscatter", "github.com/lightsmith-photonics/lightsmith"),
    (r"\bomniscatter\b", "lightsmith"),
    (r"\bOmniscatter\b", "Lightsmith"),
    (r"\bosc\b", "ls"),  # conventional import alias in docs and tests: `import omniscatter as osc`
]


def files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]  # prune before descending
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix in TEXT_SUFFIXES and p.suffix != ".lock" and name != "rename_to_lightsmith.py":
                yield p  # lockfiles are regenerated, not rewritten


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    changed = 0
    for p in files():
        text = p.read_text(encoding="utf-8")
        new = text
        for pat, rep in REPLACEMENTS:
            new = re.sub(pat, rep, new)
        if new != text:
            changed += 1
            print(f"edit {p.relative_to(ROOT)}")
            if a.apply:
                p.write_text(new, encoding="utf-8", newline="\n")
    moves = [
        ("src/omniscatter", "src/lightsmith"),
        ("plugins/omniscatter-adda/src/omniscatter_adda", "plugins/omniscatter-adda/src/lightsmith_adda"),
        ("plugins/omniscatter-adda", "plugins/lightsmith-adda"),
        ("plugins/omniscatter-scattnlay/src/omniscatter_scattnlay",
         "plugins/omniscatter-scattnlay/src/lightsmith_scattnlay"),
        ("plugins/omniscatter-scattnlay", "plugins/lightsmith-scattnlay"),
    ]
    for src, dst in moves:
        if (ROOT / src).exists():
            print(f"move {src} -> {dst}")
            if a.apply:
                subprocess.run(["git", "mv", src, dst], cwd=ROOT, check=True)
    print(f"{changed} files to edit" + ("" if a.apply else " (dry run; pass --apply)"))


if __name__ == "__main__":
    main()
