"""Export densities to polygons (marching squares), omniscatter Structures and GDSII.

No third-party dependency: contours are traced with a small marching-squares
implementation and GDS files are written directly (GDSII stream format, 1 nm
database unit). With gdsfactory installed, :func:`to_gdsfactory` builds a
``gf.Component`` instead, with holes subtracted properly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..problems import Polygon, Prism, Rectangle, Structure
from ..raster import inside_profile


@dataclass
class Contour:
    vertices: np.ndarray  # (n, 2) in nm, open (first vertex not repeated)
    depth: int  # nesting depth: even = solid (density above threshold), odd = hole

    @property
    def solid(self) -> bool:
        return self.depth % 2 == 0


def _simplify(v: np.ndarray, tol: float) -> np.ndarray:
    """Drop vertices that are (nearly) collinear with their neighbours."""
    keep = []
    n = len(v)
    for i in range(n):
        a, b, c = v[i - 1], v[i], v[(i + 1) % n]
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross > tol * max(np.hypot(*(c - a)), 1e-30):
            keep.append(i)
    return v[keep] if len(keep) >= 3 else v


def marching_squares(field: np.ndarray, x: np.ndarray, y: np.ndarray, level: float = 0.5,
                     simplify_tol: float | None = None) -> list[Contour]:
    """Closed contours of `field` (indexed [x][y] on node coordinates x, y) at `level`.

    The field is padded with values below `level`, so regions touching the edge
    are closed along the boundary of the cells (half a cell outside the outer nodes).
    """
    f = np.asarray(field, float)
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    F = np.full((f.shape[0] + 2, f.shape[1] + 2), level - 1.0)
    F[1:-1, 1:-1] = f
    X = np.concatenate([[x[0] - dx], x, [x[-1] + dx]])
    Y = np.concatenate([[y[0] - dy], y, [y[-1] + dy]])
    above = F > level

    def point(key):
        kind, i, j = key
        if kind == "h":  # edge (i,j)-(i+1,j)
            t = (level - F[i, j]) / (F[i + 1, j] - F[i, j])
            return X[i] + t * (X[i + 1] - X[i]), Y[j]
        t = (level - F[i, j]) / (F[i, j + 1] - F[i, j])
        return X[i], Y[j] + t * (Y[j + 1] - Y[j])

    adj: dict[tuple, list[tuple]] = {}

    def link(a, b):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    nx, ny = F.shape
    for i in range(nx - 1):
        for j in range(ny - 1):
            c = (above[i, j] * 1) | (above[i + 1, j] * 2) | (above[i + 1, j + 1] * 4) | (above[i, j + 1] * 8)
            if c in (0, 15):
                continue
            bottom, right, top, left = ("h", i, j), ("v", i + 1, j), ("h", i, j + 1), ("v", i, j)
            segs = {1: [(left, bottom)], 2: [(bottom, right)], 3: [(left, right)], 4: [(right, top)],
                    6: [(bottom, top)], 7: [(left, top)], 8: [(top, left)], 9: [(top, bottom)],
                    11: [(top, right)], 12: [(right, left)], 13: [(right, bottom)], 14: [(bottom, left)]}
            if c in (5, 10):
                centre = (F[i, j] + F[i + 1, j] + F[i + 1, j + 1] + F[i, j + 1]) / 4 > level
                if (c == 5) == centre:  # diagonal 00-11 connected
                    s = [(left, top), (bottom, right)]
                else:
                    s = [(left, bottom), (right, top)]
            else:
                s = segs[c]
            for a, b in s:
                link(a, b)

    loops = []
    seen: set = set()
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = [k for k in adj[cur] if k != prev]
            nxt = nxt[0] if nxt else adj[cur][0]
            if nxt == start:
                break
            if nxt in seen:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.array([point(k) for k in loop]))

    tol = (0.01 * min(dx, dy)) if simplify_tol is None else simplify_tol
    loops = [_simplify(v, tol) for v in loops]
    contours = []
    for a, va in enumerate(loops):
        depth = 0
        for b, vb in enumerate(loops):
            if a != b and bool(inside_profile(Polygon(vertices_nm=[tuple(p) for p in vb]), va[0, 0], va[0, 1])):
                depth += 1
        contours.append(Contour(vertices=va, depth=depth))
    contours.sort(key=lambda c: c.depth)
    return contours


def to_structures(contours: list[Contour], material_high, material_low, z_min_nm: float, z_max_nm: float,
                  region_center_nm=None, region_size_nm=None, name: str = "design") -> list[Structure]:
    """Structures reproducing a thresholded design (painter's order: region box, solids, holes, ...)."""
    out: list[Structure] = []
    if region_center_nm is not None and region_size_nm is not None:
        out.append(Structure(name=f"{name}:region", material=material_low, geometry=Prism(
            profile=Rectangle(center_nm=tuple(region_center_nm[:2]), size_nm=tuple(region_size_nm[:2])),
            z_min_nm=z_min_nm, z_max_nm=z_max_nm)))
    for k, c in enumerate(contours):
        out.append(Structure(name=f"{name}:{'solid' if c.solid else 'hole'}{k}",
                             material=material_high if c.solid else material_low,
                             geometry=Prism(profile=Polygon(vertices_nm=[(float(a), float(b)) for a, b in c.vertices]),
                                            z_min_nm=z_min_nm, z_max_nm=z_max_nm)))
    return out


# --- GDSII --------------------------------------------------------------------


def write_gds(path: str, contours: list[Contour], layer: tuple[int, int] = (1, 0), cell: str = "DESIGN",
              hole_layer: tuple[int, int] | None = None) -> dict:
    """Write solids to `layer` and holes to `hole_layer` (default datatype + 1). Coordinates in nm.

    Uses the dependency-free writer of :mod:`omniscatter.layout` (1 nm database unit).
    Plain GDSII polygons cannot have holes; subtract `hole_layer` from `layer` downstream
    (or use :func:`to_gdsfactory`, which does it for you). Read back with
    :func:`omniscatter.layout.read_gds`.
    """
    from ..layout import write_gds as _write

    hole_layer = hole_layer or (layer[0], layer[1] + 1)
    polys: dict[tuple[int, int], list[np.ndarray]] = {}
    for c in contours:
        polys.setdefault(tuple(layer) if c.solid else tuple(hole_layer), []).append(np.asarray(c.vertices, float))
    _write(path, {cell: {"polygons": polys}})
    return {"path": str(path), "polygons": len(contours), "holes": sum(not c.solid for c in contours),
            "layer": list(layer), "hole_layer": list(hole_layer)}


def to_gdsfactory(contours: list[Contour], layer: tuple[int, int] = (1, 0), name: str = "design"):
    """A gdsfactory Component with holes subtracted (needs `gdsfactory`). Coordinates converted to um."""
    try:
        import gdsfactory as gf
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("to_gdsfactory needs gdsfactory (`uv pip install gdsfactory`); "
                          "write_gds works without it") from e
    solids, holes = gf.Component(), gf.Component()  # pragma: no cover
    for c in contours:  # pragma: no cover
        (solids if c.solid else holes).add_polygon([(a / 1e3, b / 1e3) for a, b in c.vertices], layer=layer)
    if not any(not c.solid for c in contours):  # pragma: no cover
        return solids
    return gf.boolean(solids, holes, operation="not", layer=layer)  # pragma: no cover
