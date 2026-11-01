"""Finite-difference eigenmode (FDE) engine, reusable by other solvers.

Typical reuse from a device solver (EME, FDTD, FDFD)::

    from lightsmith.backends._fde import port_modes
    modes = port_modes(problem, problem.ports[0], wavelength_nm=1550, grid_spacing_nm=20)
    m = modes[0]
    m.neff, m.E, m.H                      # local (x, y) Yee fields, H in units of E (Z0 H)
    u, v, comps = m.global_fields()       # cell-centred fields keyed by global component names

or directly on 2D profiles with :func:`solve_modes`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from .grid import Grid, make_grid, sub_grid
from .modes import Mode, bend_transform, overlap_matrix, solve_eigenmodes
from .section import AXIS_INDEX, CrossSection, OwnerMaps, plane_axes

__all__ = ["CrossSection", "Grid", "Mode", "ModeSolver", "OwnerMaps", "default_spacing", "make_grid",
           "overlap_matrix", "plane_axes", "port_modes", "solve_eigenmodes", "solve_modes", "sub_grid", "track"]


def default_spacing(wavelength_nm: float, n_max: float, points_per_wavelength: float = 20,
                    min_feature_nm: float | None = None, cells_per_feature: float = 8) -> float:
    """Grid step: `points_per_wavelength` in the densest material, `cells_per_feature` across the smallest feature."""
    d = wavelength_nm / (n_max * points_per_wavelength)
    if min_feature_nm:
        d = min(d, min_feature_nm / cells_per_feature)
    return d


@dataclass
class ModeSolver:
    """Modes of one cross-section on one grid, at any wavelength. Geometry is rasterised once."""

    section: CrossSection
    grid: Grid
    subpixel: Literal["anisotropic", "volume", "none"] = "anisotropic"
    subsamples: int = 32
    bend_radius_nm: float | None = None
    bend_center_nm: float = 0.0
    pml_reflection: float = 1e-6
    owners: OwnerMaps | None = field(default=None, repr=False)  # precomputed rasterisation (optional)

    def __post_init__(self):
        if self.owners is not None:
            return
        fine = 1 if self.subpixel == "none" else self.subsamples
        self.owners = OwnerMaps.build(self.section, self.grid, fine, 16 * fine)

    def raw_eps(self, wavelength_nm: float) -> np.ndarray:
        return self.owners.eps(self.section.material_eps(wavelength_nm), self.subpixel)

    def _transform(self, eps):
        if self.bend_radius_nm is None:
            return eps, None
        return bend_transform(self.grid, eps, self.bend_radius_nm, self.bend_center_nm)

    def eps_group(self, wavelength_nm: float, rel_step: float = 1e-4) -> np.ndarray:
        """eps_g = d(omega eps)/d omega = eps - lambda d eps/d lambda, bend-transformed."""
        lo = self.raw_eps(wavelength_nm * (1 - rel_step))
        hi = self.raw_eps(wavelength_nm * (1 + rel_step))
        e = self.raw_eps(wavelength_nm)
        eg = e - (hi - lo) / (2 * rel_step)
        return self._transform(eg)[0]

    def solve(self, wavelength_nm: float, num_modes: int, target_neff: float | None = None,
              polarization: Literal["te", "tm"] | None = None,
              group_index: Literal["auto", "perturbation", "finite_difference", "none"] = "auto",
              extra_modes: int = 2) -> list[Mode]:
        """`num_modes` modes by decreasing Re(neff). polarization 'te' = dominant Ex (te_fraction > 0.5)."""
        modes = self._solve(wavelength_nm, num_modes, target_neff, polarization, extra_modes)
        has_pml = any(any(p) for p in self.grid.pml)
        how = ("finite_difference" if has_pml else "perturbation") if group_index == "auto" else group_index
        if how == "perturbation":
            eg = self.eps_group(wavelength_nm)
            for m in modes:
                m.ng = m.group_index_perturbation(eg)
                m.meta["ng_method"] = "perturbation"
        elif how == "finite_difference":
            step = 1e-3
            side = [self._solve(wavelength_nm * (1 + s * step), num_modes, target_neff, polarization, extra_modes + 2)
                    for s in (-1, 1)]
            for m in modes:
                ns = []
                for ms in side:
                    ov = np.abs(overlap_matrix([m], ms))[0]
                    ns.append(ms[int(np.argmax(ov))].neff.real if len(ms) else np.nan)
                dn = (ns[1] - ns[0]) / (2 * step * wavelength_nm)
                m.ng = float(m.neff.real - wavelength_nm * dn)
                m.meta["ng_method"] = "finite_difference"
        return modes

    def _solve(self, wavelength_nm, num_modes, target_neff, polarization, extra_modes):
        raw = self.raw_eps(wavelength_nm)
        eps, mu = self._transform(raw)
        if self.bend_radius_nm is not None and target_neff is None:
            return self._solve_bend(wavelength_nm, num_modes, polarization, extra_modes, raw, eps, mu)
        if target_neff is None:
            target_neff = float(np.sqrt(np.max(raw.real)))
        slab = self.grid.invariant_y
        block = {"te": "x", "tm": "y"}.get(polarization) if (slab and polarization) else None
        want = num_modes + extra_modes
        if polarization and not slab:
            want = 2 * num_modes + extra_modes + 2
        n_pml = float(np.sqrt(np.abs(raw[2][-1, -1])))
        modes = solve_eigenmodes(self.grid, eps, wavelength_nm, want, target_neff, mu, block,
                                 self.pml_reflection, n_pml)
        for m in modes:
            m.meta["pml_fraction"] = m.pml_fraction()
        if polarization and not slab:
            modes = [m for m in modes if (m.te_fraction > 0.5) == (polarization == "te")]
        # Drop PML-dominated (spurious) modes when there are spares.
        good = [m for m in modes if m.meta["pml_fraction"] < 0.5]
        if len(good) >= num_modes:
            modes = good
        return modes[:num_modes]

    def _solve_bend(self, wavelength_nm, num_modes, polarization, extra_modes, raw, eps, mu):
        """Bend modes, each found next to (and identified by overlap with) a mode of the straight guide.

        A bend's outer cladding has a transformed index above the guided neff, so a search around the
        highest index lands on whispering-gallery and PML modes; seeding with the straight modes avoids them.
        """
        radius, self.bend_radius_nm = self.bend_radius_nm, None
        try:
            straight = self._solve(wavelength_nm, num_modes, None, polarization, extra_modes)
        finally:
            self.bend_radius_nm = radius
        block = {"te": "x", "tm": "y"}.get(polarization) if (self.grid.invariant_y and polarization) else None
        n_pml = float(np.sqrt(np.abs(raw[2][-1, -1])))
        out = []
        for s in straight:
            cands = solve_eigenmodes(self.grid, eps, wavelength_nm, 10, s.neff.real, mu, block,
                                     self.pml_reflection, n_pml)
            ov = [_similarity(c, s) for c in cands]
            best = cands[int(np.argmax(ov))]
            best.meta["pml_fraction"] = best.pml_fraction()
            best.meta["straight_overlap"] = float(max(ov))
            out.append(best)
        out.sort(key=lambda m: -m.neff.real)
        return out


def _similarity(a: Mode, b: Mode) -> float:
    """|<E_a, E_b>| / (|E_a| |E_b|) over the transverse E (conjugated, insensitive to normalisation)."""
    wa, wb, _, _ = a.weights()
    dot = np.sum(wa * a.E[0] * np.conj(b.E[0])) + np.sum(wb * a.E[1] * np.conj(b.E[1]))
    na = np.sum(wa * np.abs(a.E[0]) ** 2) + np.sum(wb * np.abs(a.E[1]) ** 2)
    nb = np.sum(wa * np.abs(b.E[0]) ** 2) + np.sum(wb * np.abs(b.E[1]) ** 2)
    return float(abs(dot) / np.sqrt(na * nb)) if na > 0 and nb > 0 else 0.0


def track(previous: list[Mode], current: list[Mode]) -> list[int]:
    """Order of `current` that best continues `previous` (max total |overlap|); unmatched modes follow."""
    from scipy.optimize import linear_sum_assignment

    if not previous or not current:
        return list(range(len(current)))
    ov = np.abs(overlap_matrix(previous, current))
    rows, cols = linear_sum_assignment(-ov)
    order = [int(c) for _, c in sorted(zip(rows, cols))]
    return order + [i for i in range(len(current)) if i not in order]


def solve_modes(structures: Sequence, background, wavelength_nm: float, *, center_nm: Sequence[float],
                size_nm: Sequence[float], num_modes: int = 1, target_neff: float | None = None,
                axis: int | str | None = None, position_nm: float | None = None, dimension: int = 3,
                grid_spacing_nm: float | Sequence[float] | None = None, points_per_wavelength: float = 20,
                boundary: Literal["pec", "pml"] = "pec", pml_cells: int = 12,
                symmetry: tuple[str | None, str | None] = (None, None), bend_radius_nm: float | None = None,
                polarization: Literal["te", "tm"] | None = None, subpixel: str = "anisotropic",
                subsamples: int = 32, group_index: str = "auto") -> list[Mode]:
    """Eigenmodes of a cross-section.

    - 2D profiles (``Structure2D``): leave `axis` None; `center_nm`/`size_nm` are (x, y) windows.
    - 3D ``Structure`` s: give the plane normal `axis` and `position_nm`; `center_nm`/`size_nm` are
      3-vectors (as in a ``Port``) whose entries along the other two axes define the window.
      ``dimension=2`` treats the structures as an xy-plane device invariant along z.
    """
    if axis is None:
        section = CrossSection.from_profiles(structures, background)
        c2, s2 = tuple(center_nm[:2]), tuple(size_nm[:2])
        axes = (0, 1, 1.0)
    else:
        a = AXIS_INDEX[axis] if isinstance(axis, str) else int(axis)
        if position_nm is None:
            position_nm = center_nm[a]
        section = CrossSection.from_structures(structures, background, a, position_nm, dimension,
                                               plane_z_nm=center_nm[2] if len(center_nm) > 2 else 0.0)
        iu, iv, sv = plane_axes(a, 2 if dimension == 2 else None)
        c2 = (center_nm[iu], sv * center_nm[iv])
        s2 = (size_nm[iu], 0.0 if dimension == 2 else size_nm[iv])
        axes = (iu, iv, sv)
    if grid_spacing_nm is None:
        eps = section.material_eps(wavelength_nm)
        grid_spacing_nm = default_spacing(wavelength_nm, float(np.sqrt(eps.real.max())), points_per_wavelength)
    sp = (grid_spacing_nm, grid_spacing_nm) if np.isscalar(grid_spacing_nm) else tuple(grid_spacing_nm)
    grid = make_grid(c2, s2, sp, symmetry, pml_cells if boundary == "pml" else 0)
    solver = ModeSolver(section, grid, subpixel, subsamples, bend_radius_nm, c2[0])
    modes = solver.solve(wavelength_nm, num_modes, target_neff, polarization, group_index)
    for m in modes:
        m.meta["axes"] = axes
    return modes


def port_modes(problem, port, wavelength_nm: float, **kwargs) -> list[Mode]:
    """Modes of a :class:`~lightsmith.problems.Port` of a ``DeviceProblem`` (cut at the port plane).

    Extra keyword arguments go to :func:`solve_modes` (grid_spacing_nm, boundary, polarization, ...).
    ``port.num_modes`` modes are returned, by decreasing Re(neff); the forward direction of every mode
    is +axis, so a port with direction '-' launches the same fields with H reversed.
    """
    if getattr(problem, "layout", None) is not None:
        from ...layout import resolved_device  # GDS layouts become Prism structures

        problem = resolved_device(problem)
    axis = port.size_nm.index(0)
    kwargs.setdefault("num_modes", port.num_modes)
    return solve_modes(problem.structures, problem.background, wavelength_nm, center_nm=port.center_nm,
                       size_nm=port.size_nm, axis=axis, position_nm=port.center_nm[axis],
                       dimension=problem.dimension, **kwargs)


def _global_fields(self: Mode):
    """Cell-centred fields keyed by global component names, with the global coordinates of both local axes."""
    iu, iv, sv = self.meta.get("axes", (0, 1, 1.0))
    x, y, c = self.collocated()
    names = "xyz"
    a = 3 - iu - iv
    out = {}
    for f in ("E", "H"):
        out[f + names[iu]] = c[f + "x"]
        out[f + names[iv]] = sv * c[f + "y"]
        out[f + names[a]] = c[f + "z"]
    return x, sv * y, out


Mode.global_fields = _global_fields  # type: ignore[attr-defined]
