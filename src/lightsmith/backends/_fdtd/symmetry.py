"""Mirror symmetries of scattering runs: PEC/PMC planes through the scene's centre (Meep's mirror symmetries).

A plane x_a = c (c the centre of the particles' bounding box) is a mirror of the run when the particles map onto
themselves under x_a -> 2c - x_a (same shapes and materials at mirrored positions; all shapes are axis-aligned and
mirror-symmetric about their own centres) and the plane wave travels in the plane (k_a = 0) with its polarisation
either in the plane or along a. A z-normal substrate leaves x and y mirrors intact. The incident field is then even
(polarisation in the plane: tangential E even, a PMC plane, parity +1) or odd (polarisation along a: tangential E
odd, a PEC plane, parity -1), and so is the scattered field. The grid keeps the half x_a >= c: the plane is node
plane 1 and index 0 is a ghost plane (engine.e_region), the low side has no PML, and the half grid steps exactly
as the symmetric field of the whole one, conformal tensors included. One or two mirrors make the run 2x or 4x
smaller (the transverse PML and gap layers shrink with it).

The analysis accounts for the missing part: flux boxes are open on the mirror planes (no flux crosses them;
the faces they cut keep half weight on the plane) and their fluxes are scaled by 2 per mirror, far-zone fields
are the sum over the mirror images of the part of the box in the grid, and near fields beyond a plane are the
mirror image of those inside times the parity.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from .engine import parity
from .setup import Layout, particle_bbox, particle_slot_fn

AXES = "xyz"
NONE = (0, 0, 0)


def _key(p) -> str:
    return p.model_copy(update={"position_nm": (0.0, 0.0, 0.0)}).model_dump_json()


def _slot_fn(sim):
    """pts -> material id, with the run's overlap rule (the first particle listed wins)."""
    keys: dict[str, int] = {}
    slot_of = [[keys.setdefault(m.model_dump_json(), len(keys) + 1) for m in p.region_materials()]
               for p in sim.particles]
    return particle_slot_fn(sim, slot_of)


def symmetric_about(sim, a: int, c: float, tol: float = 1e-6) -> bool:
    """Whether the particles map onto themselves under x_a -> 2c - x_a."""
    parts = sim.particles
    pos = np.array([p.position_nm for p in parts], dtype=float)
    keys = [_key(p) for p in parts]
    used = np.zeros(len(parts), dtype=bool)
    for i in range(len(parts)):
        img = pos[i].copy()
        img[a] = 2 * c - img[a]
        cand = [j for j in range(len(parts)) if not used[j] and keys[j] == keys[i]
                and np.all(np.abs(pos[j] - img) <= tol)]
        if not cand:
            return False
        used[cand[0]] = True
    # overlapping particles: the first listed wins, which need not be mirror-symmetric; check by sampling
    half = np.array([p.shape.bounding_half_extent() for p in parts])
    fn, rng = None, np.random.default_rng(0)
    for i, j in itertools.combinations(range(len(parts)), 2):
        lo, hi = np.maximum(pos[i] - half[i], pos[j] - half[j]), np.minimum(pos[i] + half[i], pos[j] + half[j])
        if np.any(hi <= lo):
            continue
        fn = fn or _slot_fn(sim)
        pts = lo + (hi - lo) * rng.random((4096, 3))
        img = pts.copy()
        img[:, a] = 2 * c - img[:, a]
        if np.any(fn(pts) != fn(img)):
            return False
    return True


def find(sim, opts) -> tuple[tuple[int, int, int], dict]:
    """Mirror parity per axis (0: none, +1: symmetric / PMC, -1: antisymmetric / PEC) and a provenance record."""
    info = {"mirrors": {}, "reduction": 1}
    if opts.symmetry == "off":
        return NONE, {**info, "skipped": "symmetry: off"}
    if sim.illumination.kind != "plane_wave":
        return NONE, {**info, "skipped": "Gaussian beams are not used with mirror symmetry"}
    if not sim.particles:
        return NONE, info
    k, e = sim.illumination.unit_vectors()
    lo, hi = particle_bbox(sim)
    ctr = (lo + hi) / 2
    mirror, why = [0, 0, 0], []
    for a in range(3):
        name = AXES[a]
        if a == 2 and sim.substrate is not None:
            continue
        if abs(k[a]) > 1e-9:
            why.append(f"{name}: the incidence direction is not in the plane")
        elif abs(e[a]) > 1e-9 and abs(abs(e[a]) - 1) > 1e-9:
            why.append(f"{name}: the polarisation is neither in the plane nor normal to it")
        elif not symmetric_about(sim, a, float(ctr[a])):
            why.append(f"{name}: the particles are not mirror-symmetric")
        else:
            mirror[a] = 1 if abs(e[a]) <= 1e-9 else -1
    mirror = tuple(mirror)
    info["mirrors"] = {AXES[a]: "symmetric (PMC)" if m > 0 else "antisymmetric (PEC)"
                       for a, m in enumerate(mirror) if m}
    info["reduction"] = factor(mirror)
    if info["mirrors"]:
        info["planes_nm"] = {AXES[a]: float(ctr[a]) for a, m in enumerate(mirror) if m}
    if why:
        info["skipped"] = "; ".join(why)
    return mirror, info


def factor(mirror) -> int:
    """How many copies of the grid make the whole scene."""
    return 2 ** sum(1 for m in mirror if m)


def halve(lay: Layout, mirror) -> Layout:
    """The part x_a >= centre of `lay` along every mirror axis, with the plane at node 1 (index 0 a ghost)."""
    if not any(mirror):
        return lay
    off = np.array([lay.center[a] - 1 if m else 0 for a, m in enumerate(mirror)], dtype=int)
    shape = tuple(int(n - o) for n, o in zip(lay.shape, off))
    grid = None if lay.grid is None else lay.grid.crop(off)
    origin = lay.origin + off * lay.dx if grid is None else grid.origin
    return replace(lay, shape=shape, origin=origin, center=lay.center - off, grid=grid,
                   tfsf=tuple(v - off for v in lay.tfsf), tot=tuple(v - off for v in lay.tot),
                   sca=tuple(v - off for v in lay.sca),
                   roi=(np.maximum(lay.roi[0] - off, 1), lay.roi[1] - off), mirror=tuple(int(m) for m in mirror))


def pml_cells(lay: Layout) -> tuple:
    """CPML cells (low, high) per axis: none on the side of a mirror plane."""
    return tuple((0 if m else lay.pml, lay.pml) for m in lay.mirror)


def open_faces(lay: Layout) -> tuple:
    """Axes whose low box faces lie beyond a mirror plane (flux_points / surface_samples open_lo)."""
    return tuple(a for a, m in enumerate(lay.mirror) if m)


def cut_box(box, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """A box (lo, hi) of the layout cut at the mirror planes (lo = 1 there), for flux_points / surface_samples
    with open_lo = open_faces(lay)."""
    lo = np.array(box[0])
    for a, m in enumerate(lay.mirror):
        if m:
            lo[a] = 1
    return lo, box[1]


def unfold(fn, nhat, mirror) -> np.ndarray:
    """Far-zone E (n, 3) of the whole scene in directions nhat (n, 3), from fn(directions) -> that of the part of
    the box in the grid: E(n) = sum_g sigma_g g E_part(g n) over the mirror images g, sigma_g the product of their
    parities (the image of the box part radiates the mirrored field)."""
    nhat = np.asarray(nhat, dtype=float)
    out = None
    for flips in itertools.product(*[(False, True) if m else (False,) for m in mirror]):
        g = np.where(flips, -1.0, 1.0)
        sigma = float(np.prod([m for m, f in zip(mirror, flips) if f]))
        term = sigma * g * fn(nhat * g)
        out = term if out is None else out + term
    return out


class UnfoldedFarField:
    """A far-field object (layered_far.LayeredFarField) whose amplitudes are those of the whole scene."""

    def __init__(self, ff, mirror):
        self.ff, self.mirror = ff, tuple(mirror)

    def amplitudes(self, omega, e, h, nhat):
        return unfold(lambda n: self.ff.amplitudes(omega, e, h, n), nhat, self.mirror)

    def side_index(self, omega, up):
        return self.ff.side_index(omega, up)


def plane_positions(lay: Layout) -> np.ndarray:
    """World coordinate of each axis' mirror plane (node 1)."""
    return lay.world(np.ones(3))


def fold_points(pts, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """Points (..., 3) beyond the mirror planes mapped to their images, and the parity (..., 3) of each E
    component there: E(p) = sign * E(image)."""
    pts = np.array(pts, dtype=float)
    sign = np.ones(pts.shape[:-1] + (3,))
    x0 = plane_positions(lay)
    for a, m in enumerate(lay.mirror):
        if not m:
            continue
        neg = pts[..., a] < x0[a]
        pts[..., a] = np.where(neg, 2 * x0[a] - pts[..., a], pts[..., a])
        for c in range(3):
            sign[..., c] = np.where(neg, sign[..., c] * parity(c, a, m), sign[..., c])
    return pts, sign


def fold_interval(lo: float, hi: float, a: int, lay: Layout) -> tuple[float, float]:
    """The world interval [lo, hi] along axis a folded onto the grid's side of a mirror plane."""
    if not lay.mirror[a]:
        return lo, hi
    x0 = plane_positions(lay)[a]
    d = sorted((abs(lo - x0), abs(hi - x0)))
    return (x0, x0 + d[1]) if lo < x0 < hi else (x0 + d[0], x0 + d[1])


def fill_ghosts(e: np.ndarray, rng, lay: Layout) -> None:
    """E (..., 3, n0, n1, n2) sampled at node ranges rng: set the ghost plane (index 0) of every mirror axis
    whose range starts there from its image (index 2 along a for nodes, 1 for half-integer positions)."""
    for a, m in enumerate(lay.mirror):
        if not m or rng[a][0] != 0:
            continue
        for c in range(3):
            img = 1 if c == a else 2
            dst, src = [slice(None)] * 3, [slice(None)] * 3
            dst[a], src[a] = 0, img
            ec = e[..., c, :, :, :]
            ec[(Ellipsis, *dst)] = parity(c, a, m) * ec[(Ellipsis, *src)]


def plane_weights(flat, shape, mirror) -> np.ndarray:
    """Volume weight of E nodes (flat indices) in the whole scene: 1/2 per symmetric plane a node lies on."""
    comp, rest = np.divmod(np.asarray(flat, dtype=np.int64), int(np.prod(shape)))
    idx3 = np.stack(np.unravel_index(rest, shape), axis=1)
    w = np.ones(len(comp))
    for a, m in enumerate(mirror):
        if m > 0:
            w[(comp != a) & (idx3[:, a] == 1)] *= 0.5
    return w


def node_index(pt, comp: int, lay: Layout) -> np.ndarray:
    """Nearest stepped E_comp node to a world point, folded onto the grid (for probes)."""
    p, _ = fold_points(np.asarray(pt, dtype=float)[None], lay)
    off = np.zeros(3)
    off[comp] = 0.5
    i = np.rint(np.array([lay.index_of(a, p[0][a]) for a in range(3)]) - off).astype(int)
    lo = np.array([1 + (lay.mirror[a] < 0 and comp != a) if lay.mirror[a] else 1 for a in range(3)])
    return np.clip(i, lo, np.asarray(lay.shape) - 2)

