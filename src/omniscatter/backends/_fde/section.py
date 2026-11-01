"""Cross-sections: which material fills each point of a plane, and subpixel permittivity on a Yee grid.

A :class:`CrossSection` wraps an "owner" function (point -> material index,
painter's order, -1 = background) built from :mod:`omniscatter.raster`'s
inside tests, either from 2D profiles (mode problems) or from 3D structures
cut by a plane (device ports and EME cells).

Permittivity is averaged over each Yee cell from ``subsamples**2`` point
samples. The default ``anisotropic`` scheme averages 1/eps along a
component's own direction and eps across it (harmonic along the field,
arithmetic transverse), which is exact for interfaces aligned with the grid
and second-order accurate for slabs (cf. Farjadpour et al., Opt. Lett. 31,
2972 (2006)). ``volume`` is the plain fill-fraction average used by
:func:`omniscatter.raster.permittivity_2d`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ...raster import inside_geometry, inside_profile
from .grid import Grid

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def plane_axes(axis: int, invariant_axis: int | None = None) -> tuple[int, int, float]:
    """Global axes (u, v) of the local cross-section (x, y) for a plane normal to `axis`.

    (u, v, axis) is right-handed. For 2D devices the local y must be the invariant
    global axis; when that breaks handedness the local y axis is flipped (sign -1).
    """
    u, v = {2: (0, 1), 0: (1, 2), 1: (2, 0)}[axis]
    if invariant_axis is not None and u == invariant_axis:
        return v, u, -1.0
    return u, v, 1.0


@dataclass
class CrossSection:
    """Materials on a plane. ``owner(x, y)`` gives -1 for the background or the index into ``materials``."""

    owner: Callable[[np.ndarray, np.ndarray], np.ndarray]
    materials: list
    background: object
    label: str = ""
    _mat_json: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._mat_json = [m.model_dump_json() for m in [self.background, *self.materials]]

    @classmethod
    def from_profiles(cls, structures: Sequence, background) -> "CrossSection":
        """From :class:`~omniscatter.problems.Structure2D` (mode problems), local (x, y) = global (x, y)."""
        profiles = [s.profile for s in structures]

        def owner(x, y):
            out = np.full(np.broadcast(x, y).shape, -1, dtype=np.int16)
            for i, p in enumerate(profiles):
                out[inside_profile(p, x, y)] = i
            return out

        return cls(owner, [s.material for s in structures], background, "profiles")

    @classmethod
    def from_structures(cls, structures: Sequence, background, axis: int, position_nm: float,
                        dimension: int = 3, plane_z_nm: float = 0.0) -> "CrossSection":
        """Cut 3D :class:`~omniscatter.problems.Structure` s by the plane ``global[axis] = position_nm``.

        For ``dimension == 2`` the device lies in the xy plane and is invariant along z: prisms and
        boxes are treated as infinitely tall, balls are cut at ``plane_z_nm``.
        """
        inv = 2 if dimension == 2 else None
        iu, iv, sv = plane_axes(axis, inv)
        geoms = [s.geometry for s in structures]

        def z_for(g):
            kind = getattr(g, "kind", "")
            if kind == "box":
                return g.center_nm[2]
            if kind == "prism":
                return g.z_min_nm
            return plane_z_nm

        def owner(x, y):
            x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
            out = np.full(x.shape, -1, dtype=np.int16)
            for i, g in enumerate(geoms):
                pts = [None, None, None]
                pts[axis] = np.full(x.shape, position_nm)
                pts[iu] = x
                pts[iv] = sv * y
                if inv is not None:
                    pts[2] = np.full(x.shape, z_for(g))
                out[inside_geometry(g, *pts)] = i
            return out

        return cls(owner, [s.material for s in structures], background, f"plane {'xyz'[axis]}={position_nm:g}")

    def material_eps(self, wavelength_nm: float) -> np.ndarray:
        w = np.array([float(wavelength_nm)])
        return np.array([complex(m.eps(w)[0]) for m in [self.background, *self.materials]])

    def material_key(self) -> str:
        return "|".join(self._mat_json)


@dataclass
class _Component:
    """Owner data for one eps component: whole cells, plus row fill fractions in boundary cells."""

    base: np.ndarray  # (nx, ny) owner + 1 at the cell centre
    cells: np.ndarray  # flat indices of cells cut by an interface
    frac: np.ndarray  # (n_cells, rows, n_materials): fill fractions of each row along the harmonic axis
    harmonic: bool  # rows run along the component's own axis


@dataclass
class OwnerMaps:
    """Rasterised cross-section for the eps_xx, eps_yy and eps_zz positions of a Yee grid.

    Each Yee cell is probed at its corners, edges and ``coarse**2`` interior points; cells
    where the material changes are resampled with ``fine x fine`` points (``fine_1d`` along x
    for slabs), so fill fractions resolve interfaces to ~cell/(2 fine) at small cost.
    """

    grid: Grid
    n_materials: int
    comps: tuple[_Component, _Component, _Component]

    @classmethod
    def build(cls, section: CrossSection, grid: Grid, fine: int = 32, fine_1d: int = 512,
              coarse: int = 4, chunk: int = 2_000_000) -> "OwnerMaps":
        slab = grid.invariant_y
        nm = len(section.materials) + 1
        comps = []
        for c, (dual_x, dual_y) in enumerate(((True, False), (False, True), (False, False))):
            xs = grid.nodes(0, dual_x)
            ys = grid.nodes(1, dual_y and not slab)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            base = (section.owner(X, Y) + 1).astype(np.int16)
            if fine <= 1:
                comps.append(_Component(base, np.zeros(0, int), np.zeros((0, 1, nm)), c < 2))
                continue
            # Probe cell borders and interior to find cells cut by an interface.
            t = np.linspace(-0.5, 0.5, coarse + 1)
            ox, oy = t * grid.dx, (t * grid.dy if not slab else np.zeros(1))
            cut = np.zeros(base.shape, bool)
            for a in ox:
                for b in oy:
                    cut |= (section.owner(X + a, Y + b) + 1) != base
            cells = np.flatnonzero(cut)
            sx = fine_1d if slab else fine
            sy = 1 if slab else fine
            fx = ((np.arange(sx) + 0.5) / sx - 0.5) * grid.dx
            fy = ((np.arange(sy) + 0.5) / sy - 0.5) * grid.dy if not slab else np.zeros(1)
            owners = np.empty((len(cells), sx, sy), np.int16)
            step = max(1, chunk // (sx * sy))
            for s in range(0, len(cells), step):
                idx = cells[s:s + step]
                cx, cy = X.ravel()[idx], Y.ravel()[idx]
                px = cx[:, None, None] + fx[None, :, None]
                py = cy[:, None, None] + fy[None, None, :]
                owners[s:s + step] = section.owner(*np.broadcast_arrays(px, py)) + 1
            # Rows along the harmonic axis: x for eps_xx, y for eps_yy (z: any; it is arithmetic).
            along_y = c == 1
            o = owners.transpose(0, 2, 1) if along_y else owners
            frac = np.stack([(o == m).mean(axis=2) for m in range(nm)], axis=-1)
            comps.append(_Component(base, cells, frac, c < 2))
        return cls(grid, nm, tuple(comps))

    def eps(self, eps_values: np.ndarray, scheme: str = "anisotropic") -> np.ndarray:
        """(3, nx, ny) complex eps_xx, eps_yy, eps_zz at their Yee positions."""
        out = []
        for comp in self.comps:
            e = eps_values[comp.base].astype(complex)
            if scheme != "none" and len(comp.cells):
                rows = comp.frac @ eps_values  # (n_cells, rows) arithmetic mean across each row
                if scheme == "anisotropic" and comp.harmonic:
                    v = 1.0 / (1.0 / rows).mean(axis=1)
                else:
                    v = rows.mean(axis=1)
                e.ravel()[comp.cells] = v
            out.append(e)
        return np.stack(out)

    def digest(self) -> str:
        h = hashlib.sha1()
        for c in self.comps:
            h.update(c.base.tobytes())
            h.update(c.cells.tobytes())
            h.update(np.round(c.frac, 12).tobytes())
        return h.hexdigest()

    def same_as(self, other: "OwnerMaps") -> bool:
        return all(np.array_equal(a.base, b.base) and np.array_equal(a.cells, b.cells) and np.allclose(a.frac, b.frac)
                   for a, b in zip(self.comps, other.comps))
