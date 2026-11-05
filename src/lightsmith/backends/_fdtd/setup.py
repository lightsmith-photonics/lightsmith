"""Host-side (NumPy) set-up shared by FDTD problem kinds: grid layout, material
voxelisation with subpixel smoothing, and flux-box monitors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .engine import DispersiveGroup, OffDiagonal, TensorMesh, flat_index, mirror_fold, pole_coefficients
from .media import Medium

FS = 299.792458  # normalised time units (nm/c) per femtosecond
DIVERGED_GROWTH = 1e8  # monitor energy / its peak during the pulse beyond which a run counts as diverged


def e_offset(c: int) -> np.ndarray:
    o = np.zeros(3)
    o[c] = 0.5
    return o


def _amplification(m: Medium, dt: float, kappa: np.ndarray) -> float:
    """Largest |eigenvalue| of one step of the Yee + D-form ADE update in medium m, over spatial modes
    whose discrete curl-curl eigenvalue is in `kappa`. State: D^n, D^{n-1}, and P_k^n, P_k^{n-1} per pole;
    D^{n+1} - 2 D^n + D^{n-1} = -dt^2 kappa E^n, E^n = (D^n - sum P^n) / eps_inf."""
    k = len(m.poles)
    n = 2 + 2 * k
    e = np.zeros(n)
    e[0] = 1 / m.eps_inf
    e[2::2] = -1 / m.eps_inf
    A = np.zeros((kappa.size, n, n))
    A[:, 0] = -(dt**2) * kappa[:, None] * e
    A[:, 0, 0] += 2
    A[:, 0, 1] -= 1
    A[:, 1, 0] = 1
    for j, p in enumerate(m.poles):
        c1, c2, c3 = pole_coefficients(p, dt)
        i = 2 + 2 * j
        A[:, i] = c3 * e
        A[:, i, i] += c1
        A[:, i, i + 1] += c2
        A[:, i + 1, i] = 1
    return float(np.abs(np.linalg.eigvals(A)).max())


def dispersive_dt_limit(media: list[Medium], kappa_max: float) -> float:
    """Largest stable dt of the explicit ADE update over all dispersive media (inf if none).

    Drude and Lorentz poles lower the permittivity at high grid frequencies below eps_inf, so the
    Courant step computed from eps_inf can be unstable on coarse grids (von Neumann analysis of a
    homogeneous medium, which bounds the subpixel-smoothed interface cells too)."""
    kappa = np.linspace(0.0, kappa_max, 65)
    limit = math.inf
    for m in media:
        if not m.dispersive:
            continue
        hi = 2 / math.sqrt(kappa_max / m.eps_inf)  # Courant limit of eps_inf alone
        if _amplification(m, hi, kappa) <= 1 + 1e-6:
            continue
        lo = 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _amplification(m, mid, kappa) <= 1 + 1e-6 else (lo, mid)
        limit = min(limit, lo)
    return limit


def limit_dt(dt: float, courant: float, media: list[Medium], kappa_max: float) -> float:
    """dt unchanged when every dispersive medium is stable at it, else `courant` times their limit."""
    limit = dispersive_dt_limit(media, kappa_max)
    return dt if dt <= limit else courant * limit


def time_step(dx: float, courant: float, media: list[Medium], grid=None) -> float:
    """Courant-stable dt; media with eps_inf < 1 shrink it, and so do dispersive media whose ADE
    update would be unstable at that step (coarse grids). On a graded grid the smallest cell per axis sets it,
    dt = courant / sqrt(sum_a 1 / h_min,a^2) (each axis' difference operator is bounded by 2 / h_min,a)."""
    eps_min = min([1.0] + [m.eps_inf for m in media])
    if grid is not None:
        h = [grid.min_spacing(a) for a in range(3)]
        kappa_max = sum(4 / v**2 for v in h)
        if max(h) == min(h):  # cubic smallest cells: the uniform formula, to the last bit
            dt = courant * h[0] * math.sqrt(eps_min) / math.sqrt(3)
        else:
            dt = courant * math.sqrt(eps_min) / math.sqrt(sum(1 / v**2 for v in h))
        return limit_dt(dt, courant, media, kappa_max)
    return limit_dt(courant * dx * math.sqrt(eps_min) / math.sqrt(3), courant, media, 12 / dx**2)


# --- voxelisation ---------------------------------------------------------------


_TEST = np.stack(np.meshgrid(*[np.array([-0.5, 0.0, 0.5])] * 3, indexing="ij"), axis=-1).reshape(-1, 3)


MESH_MODES = ("staircase", "volume_average", "subpixel", "conformal")


def mesh_mode(opts) -> str:
    """The interface treatment: opts.mesh_refinement, else conformal (subpixel=True) or staircase."""
    if opts.mesh_refinement is not None:
        return opts.mesh_refinement
    return "conformal" if opts.subpixel else "staircase"


def voxelise(slot_fn, media: list[Medium], shape, origin, dx, roi_lo, roi_hi, *, mode: str = "subpixel",
             samples: int = 8, chunk: int = 2048, periodic=(False, False, False), base=None, background=(),
             mirror=(0, 0, 0), grid=None):
    """Material on the Yee E grid.

    slot_fn(points[..., 3] in nm) -> int medium index (0 = background) per point.
    Only nodes in [roi_lo, roi_hi) (per axis) are examined; elsewhere is background.
    mode (MESH_MODES): "staircase" takes the medium at each E position; the others supersample interface
    cells. "volume_average" uses <eps>; "subpixel" the diagonal of the Kottke tensor
    1/eps = P <1/eps> + (1 - P) / <eps>, P = n n^T; "conformal" adds its off-diagonal terms
    n_c n_d (<1/eps> - 1/<eps>) for non-dispersive cells (symmetrised; see Engine._build_offdiag). In
    "conformal" mode, interfaces with a dispersive medium take the full tensor at grid vertices instead
    (vertex_tensor; the nodes around them leave the other groups and the off-diagonal terms).
    Returns (inv_eps[3, *shape] with 1/eps for non-dispersive cells and 1 for D-form cells,
    list of DispersiveGroup (and at most one TensorMesh), stats dict, OffDiagonal or None).

    base: (inv_eps, groups) of a layered background filled beforehand (e.g. layered_background), and
    background the media indices it holds: nodes whose pixel holds only those media keep the base values,
    so a planar interface is meshed the same inside the region of interest as everywhere else.

    grid: a graded grid (grid.Grid); each node's pixel is then its own cell (primary along c, dual along the
    other axes), not a cube of side dx."""
    if mode not in MESH_MODES:
        raise ValueError(f"mesh refinement {mode!r} is not one of {MESH_MODES}")
    subpixel = mode != "staircase"
    shape = tuple(shape)
    dx = np.broadcast_to(np.asarray(dx, dtype=float), (3,))  # cell size per axis
    S = len(media)
    eps0 = media[0].eps_inf
    inv_eps = np.full((3,) + shape, 1.0 / eps0) if base is None else base[0].copy()
    bg = np.zeros(S, dtype=bool)
    bg[list(background)] = True
    claimed = []
    disp = np.array([m.dispersive for m in media])
    eps_inf = np.array([m.eps_inf for m in media])
    pure_idx = {s: [] for s in range(S) if disp[s]}
    mixed_idx, mixed_f, mixed_w = [], [], []
    od_idx3, od_comp, od_coef = [], [], []
    n_mixed = 0
    axes_idx = [np.arange(roi_lo[a], roi_hi[a]) for a in range(3)]
    sub = ((np.arange(samples) + 0.5) / samples - 0.5)
    sub3 = np.stack(np.meshgrid(sub, sub, sub, indexing="ij"), axis=-1).reshape(-1, 3)
    for c in range(3):
        idx3 = np.stack(np.meshgrid(*axes_idx, indexing="ij"), axis=-1).reshape(-1, 3)
        if grid is None:
            pos = origin + (idx3 + e_offset(c)) * dx
            ctr, size = pos, dx
        else:
            pos = grid.coords(idx3 + e_offset(c))
            ctr, size = grid.e_cells(c, idx3)
        center = slot_fn(pos)
        if subpixel:
            mixed = np.zeros(len(pos), dtype=bool)
            for o in _TEST:
                if np.any(o):
                    mixed |= slot_fn(ctr + o * size) != center
        else:
            mixed = np.zeros(len(pos), dtype=bool)
        # pure cells
        flat = flat_index(c, idx3[:, 0], idx3[:, 1], idx3[:, 2], shape)
        for s in range(1, S):
            sel = (~mixed) & (center == s)
            if bg[s] or not sel.any():
                continue
            claimed.append(flat[sel])
            if disp[s]:
                pure_idx[s].append(flat[sel])
                inv_eps.reshape(-1)[flat[sel]] = 1.0
            else:
                inv_eps.reshape(-1)[flat[sel]] = 1.0 / eps_inf[s]
        # interface cells: supersample
        mi = np.nonzero(mixed)[0]
        for a in range(0, len(mi), chunk):
            sel = mi[a:a + chunk]
            sz = size if grid is None else size[sel]
            pts = ctr[sel][:, None, :] + sub3[None, :, :] * (sz if grid is None else sz[:, None, :])
            sl = slot_fn(pts)  # (n, samples^3)
            frac = np.stack([(sl == s).mean(axis=1) for s in range(S)])  # (S, n)
            if bg.any():
                part = frac[~bg].sum(axis=0) > 0  # holds a particle medium
                sel, sl, frac = sel[part], sl[part], frac[:, part]
                if grid is not None:
                    sz = sz[part]
                if not len(sel):
                    continue
            claimed.append(flat[sel])
            n_mixed += len(sel)
            dom = np.argmax(frac, axis=0)
            chi = (sl == dom[:, None]).astype(float)
            mom = (chi - chi.mean(axis=1, keepdims=True)) @ sub3 / sz  # (n, 3) first moment ~ normal
            nrm = np.linalg.norm(mom, axis=1)
            nhat = np.where(nrm[:, None] > 1e-12, mom / np.where(nrm > 0, nrm, 1)[:, None], 0.0)
            w = np.where(nrm > 1e-12, nhat[:, c] ** 2, 1 / 3)
            if mode == "volume_average":
                w = np.zeros_like(w)
            has_disp = (frac[disp] > 0).any(axis=0) if disp.any() else np.zeros(len(sel), bool)
            f_idx = flat[sel]
            # non-dispersive interface cells: 1/eps_cc = w <1/eps> + (1-w)/<eps>
            nd = ~has_disp
            if nd.any():
                harm = (frac[:, nd] / eps_inf[:, None]).sum(axis=0)
                arit = (frac[:, nd] * eps_inf[:, None]).sum(axis=0)
                inv_eps.reshape(-1)[f_idx[nd]] = w[nd] * harm + (1 - w[nd]) / arit
                if mode == "conformal":
                    coef = nhat[nd] * nhat[nd][:, c:c + 1] * (harm - 1 / arit)[:, None]
                    coef[:, c] = 0
                    keep = np.any(np.abs(coef) > 1e-9, axis=1)
                    od_idx3.append(idx3[sel][nd][keep])
                    od_comp.append(np.full(int(keep.sum()), c))
                    od_coef.append(coef[keep])
            if has_disp.any():
                mixed_idx.append(f_idx[has_disp])
                mixed_f.append(frac[:, has_disp])
                mixed_w.append(w[has_disp])
                inv_eps.reshape(-1)[f_idx[has_disp]] = 1.0
    groups = []
    if base is not None:
        taken = np.concatenate(claimed) if claimed else np.zeros(0, dtype=np.int64)
        for g in base[1]:
            keep = ~np.isin(g.idx, taken)
            if keep.any():
                groups.append(DispersiveGroup(g.idx[keep], g.media, g.frac[:, keep], None if g.w is None else g.w[keep]))
    for s, lst in pure_idx.items():
        if lst:
            groups.append(DispersiveGroup(np.concatenate(lst), [media[s]], np.ones((1, sum(map(len, lst)))), None))
    if mixed_idx:
        f = np.concatenate(mixed_f, axis=1)
        used = [s for s in range(S) if np.any(f[s] > 0)]
        groups.append(DispersiveGroup(np.concatenate(mixed_idx), [media[s] for s in used], f[used],
                                      np.concatenate(mixed_w)))
    offdiag = None
    if od_comp:
        offdiag = OffDiagonal(np.concatenate(od_idx3), np.concatenate(od_comp), np.concatenate(od_coef))
    tm = None
    if mode == "conformal" and disp.any():
        tm = vertex_tensor(slot_fn, media, shape, origin, dx, roi_lo, roi_hi, samples=samples, chunk=chunk,
                           periodic=periodic, background=background, mirror=mirror, grid=grid)
    if tm is not None:
        taken = tm.idx
        inv_eps.reshape(-1)[taken] = 1.0
        kept = []
        for g in groups:
            keep = ~np.isin(g.idx, taken)
            if keep.any():
                kept.append(DispersiveGroup(g.idx[keep], g.media, g.frac[:, keep], None if g.w is None else g.w[keep]))
        groups = kept + [tm]
        if offdiag is not None:
            i = offdiag.idx3
            keep = ~np.isin(flat_index(offdiag.comp, i[:, 0], i[:, 1], i[:, 2], shape), taken)
            offdiag = OffDiagonal(i[keep], offdiag.comp[keep], offdiag.coef[keep]) if keep.any() else None
    stats = {"interface_cells": int(n_mixed), "dispersive_cells": int(sum(len(g.idx) for g in groups)),
             "mesh_refinement": mode, "offdiagonal_cells": 0 if offdiag is None else len(offdiag.comp),
             "tensor_vertices": 0 if tm is None else len(tm.normal)}
    return inv_eps, groups, stats, offdiag


def _canonical(x: np.ndarray, shape, periodic):
    """Real indices (n, 3) and period shifts for possibly wrapped node/vertex indices x (n, 3)."""
    x = np.array(x, dtype=np.int64)
    shift = np.zeros_like(x)
    for a in range(3):
        if periodic[a]:
            P = shape[a] - 2
            c = 1 + np.mod(x[:, a] - 1, P)
            shift[:, a] = (x[:, a] - c) // P
            x[:, a] = c
    return x, shift


def _pixel_average(slot_fn, pos, dx, S: int, samples: int, chunk: int):
    """Medium fractions (S, n) and unit normals (n, 3) (0 where undefined) of cubes of side dx centred on pos
    (dx (3,), or per point (n, 3) for the cells of a graded grid)."""
    sub = ((np.arange(samples) + 0.5) / samples - 0.5)
    sub3 = np.stack(np.meshgrid(sub, sub, sub, indexing="ij"), axis=-1).reshape(-1, 3)
    frac, nhat = np.zeros((S, len(pos))), np.zeros((len(pos), 3))
    dx = np.asarray(dx, dtype=float)
    for a in range(0, len(pos), chunk):
        sz = dx if dx.ndim == 1 else dx[a:a + chunk]
        sl = slot_fn(pos[a:a + chunk][:, None, :] + sub3[None, :, :] * (sz if dx.ndim == 1 else sz[:, None, :]))
        f = np.stack([(sl == s).mean(axis=1) for s in range(S)])
        dom = np.argmax(f, axis=0)
        chi = (sl == dom[:, None]).astype(float)
        mom = (chi - chi.mean(axis=1, keepdims=True)) @ sub3 / sz
        nrm = np.linalg.norm(mom, axis=1)
        frac[:, a:a + chunk] = f
        nhat[a:a + chunk] = np.where(nrm[:, None] > 1e-12, mom / np.where(nrm > 0, nrm, 1)[:, None], 0.0)
    return frac, nhat


def vertex_tensor(slot_fn, media: list[Medium], shape, origin, dx, roi_lo, roi_hi, *, samples: int = 8,
                  chunk: int = 2048, periodic=(False, False, False), background=(),
                  mirror=(0, 0, 0), grid=None) -> TensorMesh | None:
    """The TensorMesh of interfaces with a dispersive medium (None if there are none).

    Tensor vertices: every vertex whose cube (side dx, centred on it) holds an interface with a dispersive
    medium, plus every mixed vertex connected to one through shared nodes, so that each node next to a
    tensor vertex has, at its other vertex, either another tensor vertex or a single medium. Vertices whose
    cube holds only `background` media (a layered background, meshed apart) do not count as mixed."""
    shape = tuple(shape)
    S = len(media)
    disp = np.array([m.dispersive for m in media])
    dx3 = np.broadcast_to(np.asarray(dx, dtype=float), (3,))
    rng = [np.arange(1, n - 1) if periodic[a] else np.arange(max(int(roi_lo[a]), 1), min(int(roi_hi[a]), n - 2) + 1)
           for a, n in enumerate(shape)]
    V = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, 3)
    if grid is None:
        pos = origin + V * dx3
        ctr, size = pos, dx3
    else:  # a vertex's cell is its dual cell
        pos = grid.coords(V)
        ctr, size = grid.vertex_cells(V)
    center = slot_fn(pos)
    mixed = np.zeros(len(V), dtype=bool)
    for o in _TEST:
        if np.any(o):
            mixed |= slot_fn(ctr + o * size) != center
    Vm = V[mixed]
    if not len(Vm):
        return None
    frac, nhat = _pixel_average(slot_fn, ctr[mixed], size if grid is None else size[mixed], S, samples, chunk)
    if len(background):
        part = np.delete(frac, list(background), axis=0).sum(axis=0) > 0
        Vm, frac, nhat = Vm[part], frac[:, part], nhat[part]
        if not len(Vm):
            return None
    seed = (frac[disp] > 0).any(axis=0) & (frac.max(axis=0) < 1)
    if not seed.any():
        return None

    def key(x):
        return (x[:, 0] * shape[1] + x[:, 1]) * shape[2] + x[:, 2]

    kv = key(Vm)
    order = np.argsort(kv)
    ks = kv[order]

    def lookup(x):
        """Position in Vm of the (real) vertices x, -1 where not mixed."""
        k = key(x)
        j = np.clip(np.searchsorted(ks, k), 0, len(ks) - 1)
        return np.where(ks[j] == k, order[j], -1)

    eye = np.eye(3, dtype=np.int64)
    inT = seed.copy()
    front = np.nonzero(seed)[0]
    while len(front):
        nb = np.concatenate([_canonical(Vm[front] + s * eye[a], shape, periodic)[0] for a in range(3) for s in (-1, 1)])
        j = lookup(nb)
        j = np.unique(j[j >= 0])
        front = j[~inT[j]]
        inT[front] = True
    T = Vm[inT]
    fracT, nT = frac[:, inT], nhat[inT]
    nV = len(T)
    # the two d-nodes of each vertex: E_d at index v - e_d (minus side) and v (plus side)
    raw = np.stack([np.stack([T - eye[d], T], axis=1) for d in range(3)], axis=1)  # (nV, 3, 2, 3)
    comp = np.broadcast_to(np.arange(3)[None, :, None], (nV, 3, 2)).reshape(-1)
    real, shift = _canonical(raw.reshape(-1, 3), shape, periodic)
    real, sign = mirror_fold(comp, real, mirror)  # ghost-side nodes of a vertex on a mirror plane: their images
    live = sign != 0
    nkey = comp * int(np.prod(shape)) + key(real)
    _, first, inv_live = np.unique(nkey[live], return_index=True, return_inverse=True)
    inv = np.zeros(len(nkey), dtype=np.int64)
    inv[live] = inv_live
    first = np.nonzero(live)[0][first]
    node_comp, node_idx3 = comp[first], real[first]
    nN = len(first)
    # slots: the vertex at the node index (k = 0) and at index + e_c (k = 1)
    tkey = key(T)
    torder = np.argsort(tkey)
    tks = tkey[torder]
    slot_vert = np.full((nN, 2), -1, dtype=np.int64)
    slot_medium = np.zeros((nN, 2), dtype=np.int64)
    for k in range(2):
        vx, _ = _canonical(node_idx3 + k * eye[node_comp], shape, periodic)
        kk = key(vx)
        j = np.clip(np.searchsorted(tks, kk), 0, len(tks) - 1)
        hit = tks[j] == kk
        slot_vert[hit, k] = torder[j[hit]]
        if (~hit).any():
            slot_medium[~hit, k] = slot_fn(origin + vx[~hit] * dx3 if grid is None else grid.coords(vx[~hit]))
    return TensorMesh(media=list(media), frac=fracT, normal=nT, node_comp=node_comp, node_idx3=node_idx3,
                      vert_nodes=inv.reshape(nV, 3, 2), shift=shift.reshape(nV, 3, 2, 3), slot_vert=slot_vert,
                      slot_medium=slot_medium, shape=shape,
                      sign=sign.reshape(nV, 3, 2).astype(float) if any(mirror) else None, vert_idx3=T)


# --- flux boxes -------------------------------------------------------------------


def _face_weights(grid, a: int, lo: int, hi: int, half: bool) -> np.ndarray:
    """Quadrature weights (nm) along axis a of a box face from lo to hi: the primary widths at the half-integer
    positions lo..hi-1 (half=True), else trapezoid weights at the nodes lo..hi."""
    return grid.prim(a)[lo:hi] if half else grid.node_weights(a, lo, hi)


def flux_points(lo, hi, shape, dx: float, open_lo=(), grid=None):
    """Closed box with faces on E-node planes lo[d], hi[d] (no face at lo[d] for d in open_lo: a mirror plane).

    Returns (e_idx, h_idx1, h_idx2, coeff): outward flux = 0.5 Re sum coeff * E * conj(H),
    with E_a1 H_a2 and E_a2 H_a1 sampled at coincident in-plane points and H averaged
    across the face (trapezoid weights along node directions; on a graded grid the cells' own widths, the
    faces lying where the grid is uniform along their normal)."""
    E_i, H1, H2, W = [], [], [], []
    for d in range(3):
        a1, a2 = (d + 1) % 3, (d + 2) % 3
        half = np.arange(lo[a1], hi[a1])
        nod2 = np.arange(lo[a2], hi[a2] + 1)
        w2 = np.ones(len(nod2))
        w2[[0, -1]] = 0.5
        nod1 = np.arange(lo[a1], hi[a1] + 1)
        w1 = np.ones(len(nod1))
        w1[[0, -1]] = 0.5
        half2 = np.arange(lo[a2], hi[a2])
        for face, sign in ((lo[d], -1.0), (hi[d], 1.0))[d in open_lo:]:
            for e_comp, h_comp, i1, i2, ww, tsign in ((a1, a2, half, nod2, w2, 1.0), (a2, a1, nod1, half2, w1, -1.0)):
                A, B = np.meshgrid(i1, i2, indexing="ij")
                wgt = np.broadcast_to(ww[None, :] if tsign > 0 else ww[:, None], A.shape)
                idx = [None] * 3
                idx[a1], idx[a2] = A.ravel(), B.ravel()
                idx[d] = np.full(A.size, face)
                E_i.append(flat_index(e_comp, *idx, shape))
                idx[d] = np.full(A.size, face - 1)
                H1.append(flat_index(h_comp, *idx, shape))
                idx[d] = np.full(A.size, face)
                H2.append(flat_index(h_comp, *idx, shape))
                if grid is None:
                    W.append(sign * tsign * dx**2 * wgt.ravel())
                else:
                    w_1 = _face_weights(grid, a1, lo[a1], hi[a1], tsign > 0)
                    w_2 = _face_weights(grid, a2, lo[a2], hi[a2], tsign < 0)
                    W.append(sign * tsign * np.outer(w_1, w_2).ravel())
    return np.concatenate(E_i), np.concatenate(H1), np.concatenate(H2), np.concatenate(W)


@dataclass
class Surface:
    """Equivalent-current samples on a closed box (the flux_points layout): J = n x H, M = -n x E.

    Sample s carries E component e_idx and the face-averaged H component (h1, h2) at the same point
    pos (cell units, (n, 3)), so J = H_s * jdir and M = E_s * mdir, with area weight `area` (nm^2).
    ``blocks``: per (face, component pair) the sample range and its grid, (start, d, face, e_c, h_c, i1, i2),
    samples ordered as meshgrid(i1, i2, indexing="ij") with i1 along (d + 1) % 3 and i2 along (d + 2) % 3."""

    e_idx: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    pos: np.ndarray
    jdir: np.ndarray
    mdir: np.ndarray
    area: np.ndarray
    blocks: list = field(default_factory=list)


def surface_samples(lo, hi, shape, dx: float, open_lo=(), grid=None) -> Surface:
    """The Surface of the box [lo, hi] (see flux_points; grid: a graded grid's cell widths as area weights)."""
    E_i, H1, H2, pos, jd, md, area = [], [], [], [], [], [], []
    blocks, start = [], 0
    eye = np.eye(3)
    for d in range(3):
        a1, a2 = (d + 1) % 3, (d + 2) % 3
        w_n1 = np.ones(hi[a1] - lo[a1] + 1)
        w_n1[[0, -1]] = 0.5
        w_n2 = np.ones(hi[a2] - lo[a2] + 1)
        w_n2[[0, -1]] = 0.5
        for face, sign in ((lo[d], -1.0), (hi[d], 1.0))[d in open_lo:]:
            nrm = sign * eye[d]
            # E_a1 (half-integer along a1, nodes along a2) with H_a2; E_a2 with H_a1
            for e_c, h_c, i1, i2, w1, w2 in ((a1, a2, np.arange(lo[a1], hi[a1]), np.arange(lo[a2], hi[a2] + 1),
                                              np.ones(hi[a1] - lo[a1]), w_n2),
                                             (a2, a1, np.arange(lo[a1], hi[a1] + 1), np.arange(lo[a2], hi[a2]),
                                              w_n1, np.ones(hi[a2] - lo[a2]))):
                A, B = np.meshgrid(i1, i2, indexing="ij")
                W = np.outer(w1, w2).ravel()
                idx = [None] * 3
                idx[a1], idx[a2] = A.ravel(), B.ravel()
                idx[d] = np.full(A.size, face)
                E_i.append(flat_index(e_c, *idx, shape))
                H2.append(flat_index(h_c, *idx, shape))
                idx[d] = np.full(A.size, face - 1)
                H1.append(flat_index(h_c, *idx, shape))
                p = np.zeros((A.size, 3))
                p[:, a1], p[:, a2], p[:, d] = A.ravel(), B.ravel(), face
                p[:, e_c] += 0.5
                pos.append(p)
                jd.append(np.broadcast_to(np.cross(nrm, eye[h_c]), (A.size, 3)))
                md.append(np.broadcast_to(-np.cross(nrm, eye[e_c]), (A.size, 3)))
                if grid is None:
                    area.append(dx**2 * W)
                else:
                    area.append(np.outer(_face_weights(grid, a1, lo[a1], hi[a1], e_c == a1),
                                         _face_weights(grid, a2, lo[a2], hi[a2], e_c == a2)).ravel())
                blocks.append((start, d, int(face), e_c, h_c, i1, i2))
                start += A.size
    return Surface(*(np.concatenate(v) for v in (E_i, H1, H2, pos, jd, md, area)), blocks=blocks)


def far_field(surf: Surface, e: np.ndarray, h: np.ndarray, r: np.ndarray, nhat: np.ndarray, k: float,
              eta: float, chunk: int = 256) -> np.ndarray:
    """Far-zone E (times r e^{-ikr}) radiated by the surface currents, in directions nhat (m, 3).

    E_far = (-ik / 4pi) [eta n x (n x N) + n x L], N = sum J e^{-ik n.r'} dA, L = sum M e^{-ik n.r'} dA
    (exp(-i w t)); e, h are the DFT samples at one frequency and r (n, 3) the sample positions in nm."""
    J = (h * surf.area)[:, None] * surf.jdir
    M = (e * surf.area)[:, None] * surf.mdir
    out = np.empty((len(nhat), 3), dtype=complex)
    for a in range(0, len(nhat), chunk):
        nh = nhat[a:a + chunk]
        ph = np.exp(-1j * k * (nh @ r.T))  # (m, n)
        N, L = ph @ J, ph @ M
        nxN = np.cross(nh, N)
        out[a:a + chunk] = (-1j * k / (4 * np.pi)) * (eta * np.cross(nh, nxN) + np.cross(nh, L))
    return out


def node_volumes(flat, shape, grid, mirror=(0, 0, 0)) -> np.ndarray:
    """Volume (nm^3) of the cells of E nodes (flat indices) on a graded grid: primary width along the component,
    dual widths along the other axes (the grid's energy weights)."""
    comp, rest = np.divmod(np.asarray(flat, dtype=np.int64), int(np.prod(shape)))
    idx3 = np.stack(np.unravel_index(rest, shape), axis=1)
    v = np.ones(len(comp))
    for a in range(3):
        prim, dual = grid.prim(a), grid.dual(a, mirror[a])
        j = idx3[:, a]
        v *= np.where(comp == a, prim[np.clip(j, 0, len(prim) - 1)], dual[j])
    return v


def flux(e: np.ndarray, h: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """0.5 Re sum coeff E H*, per frequency (rows)."""
    return 0.5 * np.real((e * np.conj(h)) @ coeff)


# --- scattering layout --------------------------------------------------------------


@dataclass
class Layout:
    dx: float
    shape: tuple[int, int, int]
    origin: np.ndarray  # world position of node (0, 0, 0)
    center: np.ndarray  # index of the reference node r0
    tfsf: tuple[np.ndarray, np.ndarray]
    tot: tuple[np.ndarray, np.ndarray]
    sca: tuple[np.ndarray, np.ndarray]
    roi: tuple[np.ndarray, np.ndarray]
    pml: int
    mirror: tuple = (0, 0, 0)  # mirror plane per axis at node 1 (see engine.e_region; symmetry.py)
    grid: object = None  # a graded grid (grid.Grid; dx is then its base spacing), None for a uniform one

    @property
    def cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def r0(self) -> np.ndarray:
        return self.world(self.center)

    def world(self, u) -> np.ndarray:
        """World positions (..., 3) of fractional index triples u (..., 3)."""
        if self.grid is None:
            return self.origin + np.asarray(u) * self.dx
        return self.grid.coords(u)

    def coord(self, a: int, u) -> np.ndarray:
        """World coordinate along axis a of fractional indices u."""
        if self.grid is None:
            return self.origin[a] + np.asarray(u) * self.dx
        return self.grid.coord(a, u)

    def index_of(self, a: int, x) -> np.ndarray:
        """Fractional node index of world coordinate x along axis a."""
        if self.grid is None:
            return (np.asarray(x) - self.origin[a]) / self.dx
        return self.grid.index_of(a, x)

    @property
    def min_spacing(self) -> float:
        return self.dx if self.grid is None else self.grid.min_spacing()

    def widths(self):
        """Engine ``widths`` (None on a uniform grid)."""
        return None if self.grid is None else self.grid.widths()


def particle_bbox(sim) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min([np.asarray(p.position_nm) - p.shape.bounding_half_extent() for p in sim.particles], axis=0)
    hi = np.max([np.asarray(p.position_nm) + p.shape.bounding_half_extent() for p in sim.particles], axis=0)
    return lo, hi


def scattering_layout(sim, opts, dx: float, bbox=None) -> Layout:
    """Grid around `bbox` (default: the particles' bounding box): TFSF box, flux boxes, gaps and CPML."""
    lo, hi = particle_bbox(sim) if bbox is None else bbox
    ctr = (lo + hi) / 2
    half_cells = np.ceil((hi - lo) / 2 / dx - 1e-9).astype(int) + 1
    # default gaps: 4 cells, but no less than 3% of the longest wavelength in the medium, so that refining the
    # grid does not move the TFSF box and the flux monitors closer to the particle and the CPML
    floor = 0.03 * float(np.max(sim.wavelength_array())) / sim.medium_index
    n_pad = max(4, math.ceil((floor if opts.padding_nm is None else opts.padding_nm) / dx - 1e-9))
    n_gap = max(4, math.ceil((floor if opts.pml_gap_nm is None else opts.pml_gap_nm) / dx - 1e-9))
    half_tfsf = half_cells + n_pad
    n_pml = opts.pml_cells or (16 if getattr(sim, "substrate", None) is not None else 10)
    half_tot = half_tfsf + 2 + n_gap + n_pml + 1
    shape = tuple(int(v) for v in 2 * half_tot + 1)
    c = half_tot.astype(int)
    origin = ctr - c * dx
    roi_lo = np.maximum(c - half_cells - 2, 1)
    roi_hi = np.minimum(c + half_cells + 3, np.asarray(shape) - 1)
    return Layout(dx=dx, shape=shape, origin=origin, center=c, tfsf=(c - half_tfsf, c + half_tfsf),
                  tot=(c - half_tfsf + 2, c + half_tfsf - 2), sca=(c - half_tfsf - 2, c + half_tfsf + 2),
                  roi=(roi_lo, roi_hi), pml=n_pml)


def graded_layout(sim, opts, base: float, regions, bbox=None, mirror=(0, 0, 0), ratio: float = 1.25) -> Layout:
    """scattering_layout on a graded grid (grid.py): `base` spacing, refined in `regions` and graded at `ratio`.

    Every box face (TFSF, flux boxes) lies at least three cells beyond the graded zone, where the grid is
    uniform along the face normal, and so do the gaps and the CPML; the gaps and padding keep their physical
    lengths. Returns the uniform layout when no region refines the grid. Along mirror axes the grid is symmetric
    about the centre (symmetry.halve then keeps one half)."""
    from .grid import Grid, axis_nodes

    lo, hi = particle_bbox(sim) if bbox is None else bbox
    ctr = (lo + hi) / 2
    floor = 0.03 * float(np.max(sim.wavelength_array())) / sim.medium_index
    pad = floor if opts.padding_nm is None else opts.padding_nm
    n_gap = max(4, math.ceil((floor if opts.pml_gap_nm is None else opts.pml_gap_nm) / base - 1e-9))
    n_pml = opts.pml_cells or (16 if getattr(sim, "substrate", None) is not None else 10)
    nodes, idx = [], []
    for a in range(3):
        sides = axis_nodes(float(ctr[a]), base, regions, a, ratio, float(ctr[a] - lo[a]), float(hi[a] - ctr[a]),
                           bool(mirror[a]))
        counts, widths = [], []
        for w, ext in zip(sides, (ctr[a] - lo[a], hi[a] - ctr[a])):
            g = len(w)
            dist = np.concatenate([[0.0], np.cumsum(w)])

            def reach(d, dist=dist, g=g):
                """First node index at distance >= d from the centre (base cells beyond the graded zone)."""
                if d <= dist[-1] + 1e-9:
                    return int(np.searchsorted(dist, d - 1e-9))
                return g + math.ceil((d - dist[-1]) / base - 1e-9)

            def at(i, dist=dist, g=g):
                return dist[i] if i <= g else dist[-1] + (i - g) * base

            i_p = reach(ext) + 1  # the particles' half extent plus one cell (scattering_layout's half_cells)
            i_t = max(reach(at(i_p) + pad), i_p + 4, g + 3)
            n_side = i_t + 2 + n_gap + n_pml + 1
            counts.append((i_p, i_t, n_side))
            widths.append(np.concatenate([w, np.full(n_side - g, base)]))
        (p_lo, t_lo, n_lo), (p_hi, t_hi, n_hi) = counts
        x = np.concatenate([ctr[a] - np.cumsum(widths[0])[::-1], [ctr[a]], ctr[a] + np.cumsum(widths[1])])
        nodes.append(x)
        idx.append((n_lo, p_lo, t_lo, p_hi, t_hi))
    grid = Grid(tuple(nodes))
    c = np.array([v[0] for v in idx])
    p_lo, t_lo = np.array([v[1] for v in idx]), np.array([v[2] for v in idx])
    p_hi, t_hi = np.array([v[3] for v in idx]), np.array([v[4] for v in idx])
    shape = grid.shape
    tfsf = (c - t_lo, c + t_hi)
    roi_lo = np.maximum(c - p_lo - 2, 1)
    roi_hi = np.minimum(c + p_hi + 3, np.asarray(shape) - 1)
    lay = Layout(dx=float(base), shape=shape, origin=grid.origin, center=c, tfsf=tfsf, tot=(tfsf[0] + 2, tfsf[1] - 2),
                 sca=(tfsf[0] - 2, tfsf[1] + 2), roi=(roi_lo, roi_hi), pml=n_pml, grid=grid)
    if grid.uniform:
        return scattering_layout(sim, opts, base, bbox)
    return lay


def particle_slot_fn(sim, slot_of: list[list[int]]):
    """pts -> medium index; the first particle listed wins overlaps (as in DDA)."""
    def fn(pts):
        out = np.zeros(pts.shape[:-1], dtype=int)
        for p, slots in zip(sim.particles, slot_of):
            region = p.region(pts - np.asarray(p.position_nm))
            for r, s in enumerate(slots):
                out[(region == r) & (out == 0)] = s
        return out
    return fn


# --- time loop ---------------------------------------------------------------------


def run_until_decayed(eng, opts, *, max_steps: int, min_steps: int, window: int, warnings: list[str]):
    """Run chunks until the monitor energy has fallen below opts.decay_by of its peak (over a sliding
    window of `window` chunks, after `min_steps`), or until max_steps / opts.run_time_fs.

    The energy is that of the monitors with energy=1. Monitors with energy=2 give a reference: the peak is
    taken as at least 1e-6 of theirs, so a signal that never rises above noise does not run to the cap
    (scattering: the scattered field decays, the total field is the reference; a weak, long-ringing
    resonance would otherwise be cut off by the incident pulse dominating the peak).
    Raises FloatingPointError when the run diverges (non-finite or runaway monitor energy).
    Returns (steps, decay_level, used_cuda_graphs)."""
    from collections import deque

    from ...device import Device

    run, graphs = eng.make_runner(opts.cuda_graphs)
    win = deque(maxlen=max(3, int(window)))
    peak, peak_ref, grown_ref, level, steps = 0.0, 0.0, 0.0, float("nan"), 0
    # the growth reference: the peak while the pulse passes, but within the first half of the run, so the check
    # still works when a source whose background rings on (coarse grids) sets min_steps beyond max_steps
    ref_steps = min(min_steps, max_steps // 2)
    try:
        while steps < max_steps:
            run()
            steps += eng.stride
            e, e_ref = (float(v) for v in Device.to_numpy(eng.energy))  # one sync per chunk
            # float32 overflows to inf, but float64 (CPU) stays finite to ~1e300: also stop on growth far
            # above the energy seen while the pulse passed (grown_ref), which a passive structure cannot reach.
            finite = np.isfinite(e) and np.isfinite(e_ref)
            if not finite or (steps > ref_steps and e > DIVERGED_GROWTH * grown_ref > 0):
                why = "non-finite fields" if not finite else \
                    f"monitor energy grew to {e / grown_ref:.0e} times its peak while the pulse passed"
                raise FloatingPointError(
                    f"FDTD diverged after {steps} steps ({why}); the results would be meaningless. "
                    "Use a finer mesh (smaller spacing_nm or larger cells_per_wavelength), lower courant, "
                    "or give lossy/dispersive materials as Drude/DrudeLorentz")
            win.append(e)
            peak, peak_ref = max(peak, e), max(peak_ref, e_ref)
            if steps <= ref_steps:
                grown_ref = peak
            floor = max(peak, 1e-6 * peak_ref)
            level = max(win) / floor if floor > 0 else float("nan")
            if not opts.run_time_fs and steps >= min_steps and len(win) == win.maxlen and level < opts.decay_by:
                break
        else:
            if not opts.run_time_fs:
                warnings.append(f"fields had decayed only to {level:.1e} of peak at the time cap "
                                f"({opts.max_time_fs} fs); raise max_time_fs or decay_by")
    finally:
        eng.release()
    return steps, level, graphs
