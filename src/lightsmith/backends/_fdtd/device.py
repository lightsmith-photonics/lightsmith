"""Device problems: S-parameters of waveguide devices from mode sources and modal monitors (``ports.py``).

The grid covers the domain plus a few cells (ports may lie on its faces) with cubic cells, and CPML outside it;
structures (waveguide stubs) run on through the CPML. A ``dimension=2`` device (structures in the xy plane,
invariant along z) is one periodic cell along z, i.e. a 2D run with both polarisations decoupled;
``polarization`` picks the one the ports launch.

Every port mode is excited in turn (one run each). Its column of S comes from the modal amplitudes at all ports:
S[out, in] = c_out / a_in, both at the ports' reference planes (their centres), with unit-power modes.

Port modes are exact for the grid only where the material is diagonal, so device problems default to ``subpixel``
smoothing: a straight guide then transmits with |S21| = 1 and the grid's own phase (reflection at the CPML's
level, 3e-5). Under an explicit ``conformal`` the off-diagonal terms are dropped in a slab of a few cells around
each port's source and monitor planes; elsewhere they change the guide's discrete beta slightly (2.5e-4 in neff
for a 500 x 220 nm silicon wire at dx = 25 nm, from its corners), and the slab's edges reflect ~2e-4. A port
crossing vertex-tensor cells (a dispersive interface under ``conformal``) makes the whole run use ``subpixel``.

Graded grids (``grid: "auto"``, ``mesh_override``): per axis a tensor-product grid (``grid.py``). ``auto`` takes
the base spacing from the background medium (``cells_per_wavelength`` there, or ``background_cells_per_wavelength``
when given: the scattering default of 60 guards against waves crossing the grading, while a guided wave runs
along its own refined core) and refines each structure's
bounding box to ``cells_per_wavelength`` in its own material; override boxes refine where asked. A refinement
that reaches a face of the domain continues through the buffer and the CPML (structures run on through them, and
the CPML stays uniform). Each port sits in a zone of uniform cells along its normal (its source, monitor and slab),
because the discrete mode relation B = 2 sin(beta h / 2) / h holds in a uniform section; across the window the
cells may be graded, and the port modes come from the FDE problem on the same non-uniform cross-section.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ...device import Device
from ...problems import Ball, Box, DeviceProblem, Polygon, Rectangle, wavelength_array
from ...raster import inside_geometry
from ...results import SParameterResult
from .engine import DFTMonitor, Engine, OffDiagonal, PMLSpec, TensorMesh
from .media import Medium, needs_fit, fit_module, to_medium
from .options import FDTDOptions
from .ports import ModeSource, ModeTable, PortFrame, node_eta
from .grid import Grid as GradedGrid, Region, spacing_fn
from .setup import FS, run_until_decayed, voxelise

SLAB = 3  # cells beyond the source and monitor planes kept free of off-diagonal terms
BUFFER = SLAB + 5  # cells added to the domain on every side (ports on its faces need room for source and slab)


def _resolved(problem: DeviceProblem) -> DeviceProblem:
    if problem.layout is not None:
        from ...layout import resolved_device

        return resolved_device(problem)
    return problem


def _materials(problem) -> list:
    return [problem.background] + [s.material for s in problem.structures]


def supports(problem) -> tuple[bool, str]:
    if not isinstance(problem, DeviceProblem):
        return False, "not a device problem"
    for p in problem.ports:
        a = p.normal_axis
        if problem.dimension == 2 and a == 2:
            return False, f"port {p.name!r}: in a 2D device ports must be normal to x or y"
        if problem.dimension == 3 and sum(s == 0 for s in p.size_nm) != 1:
            return False, f"port {p.name!r}: a 3D port window needs a nonzero size along both transverse axes"
    try:
        materials = _materials(_resolved(problem))
    except Exception as e:  # noqa: BLE001
        return False, f"layout: {e}"
    if any(needs_fit(m) for m in materials) and fit_module() is None:
        return False, "tabulated/library materials need lightsmith.materials_fit"
    return True, ""


# --- grid ----------------------------------------------------------------------------------------


@dataclass
class _Grid:
    dx: float  # the cell size (uniform), or the base spacing (graded)
    shape: tuple[int, int, int]
    origin: np.ndarray
    interior: tuple[tuple[int, int], ...]  # first and last node index of the domain per axis
    n_pml: int
    two_d: bool
    graded: GradedGrid | None = None  # node coordinates of a graded grid

    @property
    def cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def periodic(self) -> tuple[bool, bool, bool]:
        return (False, False, self.two_d)

    def coord(self, a: int, i) -> float:
        """World coordinate of node index i along a."""
        if self.graded is None:
            return float(self.origin[a] + i * self.dx)
        return float(self.graded.nodes[a][i])

    def index(self, a: int, x) -> float:
        """Fractional node index of world coordinate x along a."""
        if self.graded is None:
            return (float(x) - self.origin[a]) / self.dx
        return float(self.graded.index_of(a, x))

    def prim(self, a: int, lo: int, hi: int) -> np.ndarray:
        """Primary widths of cells lo .. hi - 1 along a."""
        if self.graded is None:
            return np.full(hi - lo, self.dx)
        return self.graded.prim(a)[lo:hi]

    def engine_spacing(self):
        """The Engine's dx: the cell size, or per axis the (uniform) spacing of each axis of a graded grid."""
        if self.graded is None:
            return self.dx
        return tuple(float(self.graded.prim(a)[0]) for a in range(3))

    def engine_widths(self):
        """The Engine's per-axis widths: None on uniform (and periodic) axes."""
        if self.graded is None:
            return None
        return tuple(None if (self.periodic[a] or self.graded.uniform_axis(a)) else self.graded.prim(a)
                     for a in range(3))

    def min_spacing(self, a: int) -> float:
        return self.dx if self.graded is None else self.graded.min_spacing(a)


def _n_max(problem, wl) -> float:
    n = 1.0
    for m in _materials(problem):
        try:
            n = max(n, float(np.abs(np.sqrt(np.asarray(m.eps(wl), dtype=complex))).max()))
        except Exception:  # noqa: BLE001 (range errors surface later, with context)
            pass
    return n


def graded(opts: FDTDOptions) -> bool:
    return opts.grid == "auto" or bool(opts.mesh_override)


def _n_of(material, wl) -> float:
    try:
        return max(1.0, float(np.abs(np.sqrt(np.asarray(material.eps(wl), dtype=complex))).max()))
    except Exception:  # noqa: BLE001 (range errors surface later, with context)
        return 1.0


def _regions(problem, opts: FDTDOptions, wl, base: float) -> list[Region]:
    """Refinement boxes: each structure's bounding box (grid 'auto') and the override boxes. A box that reaches
    a face of the domain runs on to infinity there (through the buffer and the CPML)."""
    c = np.asarray(problem.domain_center_nm, dtype=float)
    size = np.asarray(problem.domain_size_nm, dtype=float)
    lo_d, hi_d = c - size / 2, c + size / 2
    boxes = []
    if opts.grid == "auto":
        for st in problem.structures:
            h = float(wl.min()) / (_n_of(st.material, wl) * opts.cells_per_wavelength)
            if h < base:
                lo, hi = _bbox(st.geometry)
                boxes.append((lo, hi, np.full(3, h)))
    for o in opts.mesh_override:
        oc, sz = np.asarray(o.center_nm, dtype=float), np.asarray(o.size_nm, dtype=float)
        boxes.append((oc - sz / 2, oc + sz / 2, np.broadcast_to(np.asarray(o.spacing_nm, dtype=float), (3,))))
    out = []
    for lo, hi, sp in boxes:
        lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
        tol = 1e-6 * base
        lo[lo <= lo_d + tol] = -np.inf
        hi[hi >= hi_d - tol] = np.inf
        out.append(Region.make(lo, hi, sp))
    return out


def _min_h(base: float, regions, a: int, ratio: float, lo: float, hi: float) -> float:
    """Exact minimum of the spacing function over [lo, hi] along a (it is piecewise linear)."""
    v = base
    for r in regions:
        if r.spacing[a] < base:
            dist = max(r.lo[a] - hi, lo - r.hi[a], 0.0)
            v = min(v, r.spacing[a] + (ratio - 1) * dist)
    return float(v)


def _march(h, x0: float, direction: int, until: float | None = None, count: int | None = None) -> list[float]:
    """Cell widths from x0 outward: until past `until`, or `count` cells. Each cell takes the minimum of h over
    itself (fixed-point passes), so it never overruns a finer region."""
    widths, x = [], float(x0)
    t = np.linspace(0.0, 1.0, 17)
    while (direction * (x - until) < 0) if count is None else (len(widths) < count):
        w = float(h(np.array([x]))[0])
        for _ in range(8):
            w_new = float(np.min(h(x + direction * t * w)))
            if w_new >= w * (1 - 1e-12):
                break
            w = w_new
        widths.append(w)
        x += direction * w
        if len(widths) > 1_000_000:
            raise ValueError("graded grid: too many cells (check mesh_override spacings)")
    return widths


def _port_zones(problem, regions, a: int, base: float, ratio: float) -> list[Region]:
    """A zone of uniform cells along axis a around each port normal to a: its source, monitor and slab."""
    out = []
    for p in problem.ports:
        if p.normal_axis != a:
            continue
        x = p.center_nm[a]
        hz = _min_h(base, regions, a, ratio, x, x)
        for _ in range(3):  # the zone's width depends on its spacing
            half = (SLAB + 6) * hz
            hz = _min_h(base, regions, a, ratio, x - half, x + half)
        if hz < base:
            sp = np.full(3, base)
            sp[a] = hz
            lo, hi = np.full(3, -np.inf), np.full(3, np.inf)
            lo[a], hi[a] = x - half, x + half
            out.append(Region.make(lo, hi, sp))
    return out


def _graded_grid(problem, opts: FDTDOptions, wl) -> _Grid:
    n_pml = opts.pml_cells or 12
    two_d = problem.dimension == 2
    if opts.spacing_nm:
        base = float(opts.spacing_nm)
    elif opts.grid == "auto":  # guided waves run along their refined cores, so the background need not be finer
        cpw = (opts.background_cells_per_wavelength if "background_cells_per_wavelength" in opts.model_fields_set
               else opts.cells_per_wavelength)
        base = float(wl.min()) / (_n_of(problem.background, wl) * cpw)
    else:
        base = float(wl.min()) / (_n_max(problem, wl) * opts.cells_per_wavelength)
    regions = _regions(problem, opts, wl, base)
    ratio = opts.grading_ratio
    nodes, interior = [], []
    for a in range(3):
        c, size = problem.domain_center_nm[a], problem.domain_size_nm[a]
        if a == 2 and two_d:
            nodes.append(np.array([c - base, c, c + base]))
            interior.append((1, 1))
            continue
        regs = regions + _port_zones(problem, regions, a, base, ratio)
        h, _ = spacing_fn(base, regs, a, ratio)
        up = _march(h, c, 1, until=c + size / 2)
        dn = _march(h, c, -1, until=c - size / 2)
        up += _march(h, c + sum(up), 1, count=BUFFER)  # ports may sit on the domain's faces
        dn += _march(h, c - sum(dn), -1, count=BUFFER)
        w_lo, w_hi = dn[-1], up[-1]
        widths = [w_lo] * (n_pml + 1) + dn[::-1] + up + [w_hi] * (n_pml + 2)  # CPML and PEC backstop: uniform
        x0 = c - sum(dn) - (n_pml + 1) * w_lo
        nodes.append(x0 + np.concatenate([[0.0], np.cumsum(widths)]))
        i0 = n_pml + 1
        interior.append((i0, i0 + len(dn) + len(up)))
    g = GradedGrid(tuple(nodes))
    return _Grid(float(base), g.shape, g.origin, tuple(interior), n_pml, two_d, graded=g)


def make_grid(problem, opts: FDTDOptions) -> _Grid:
    wl = wavelength_array(problem.wavelengths)
    if graded(opts):
        return _graded_grid(problem, opts, wl)
    dx = opts.spacing_nm or float(wl.min()) / (_n_max(problem, wl) * opts.cells_per_wavelength)
    n_pml = opts.pml_cells or 12
    two_d = problem.dimension == 2
    shape, origin, interior = [], [], []
    for a in range(3):
        c, s = problem.domain_center_nm[a], problem.domain_size_nm[a]
        if a == 2 and two_d:
            shape.append(3)
            origin.append(c - dx)
            interior.append((1, 1))
            continue
        n_dom = max(1, math.ceil(s / dx - 1e-9)) + 2 * BUFFER  # ports may sit on the domain's faces
        i0 = n_pml + 1
        shape.append(i0 + n_dom + n_pml + 2)
        origin.append(c - 0.5 * n_dom * dx - i0 * dx)
        interior.append((i0, i0 + n_dom))
    return _Grid(float(dx), tuple(shape), np.array(origin), tuple(interior), n_pml, two_d)


def port_frames(problem, grid: _Grid) -> list[PortFrame]:
    out = []
    for p in problem.ports:
        a = p.normal_axis
        if grid.two_d and a == 1:
            t1, t2, sgn = 0, 2, -1  # (x, z, -y) is right-handed, with the invariant axis second (as FDE wants)
        else:
            t1, t2, sgn = (a + 1) % 3, (a + 2) % 3, 1
        d = 1 if p.direction == "+" else -1
        k_p = int(round(grid.index(a, p.center_nm[a])))
        k_m = k_p + 2 * d
        lo_a, hi_a = grid.interior[a]
        if min(k_p, k_m) - SLAB - 1 < lo_a or max(k_p, k_m) + SLAB + 1 > hi_a:
            raise ValueError(f"port {p.name!r}: its plane must lie at least {SLAB + 3} cells inside the domain "
                             f"along {'xyz'[a]} (the source, monitor and their slab sit there)")
        k0, k1 = min(k_p, k_m) - SLAB - 1, max(k_p, k_m) + SLAB + 1
        wn = grid.prim(a, k0, k1)
        if np.ptp(wn) > 1e-9 * wn.mean():
            raise ValueError(f"port {p.name!r}: the cells along {'xyz'[a]} around the port are not uniform "
                             f"({wn.min():.3g}-{wn.max():.3g} nm); move it away from the mesh override's edge")
        ds = float(wn.mean())
        lo, n = [], []
        for t in (t1, t2):
            if grid.two_d and t == 2:
                lo.append(1)
                n.append(1)
                continue
            i_lo = int(round(grid.index(t, p.center_nm[t] - p.size_nm[t] / 2)))
            i_hi = int(round(grid.index(t, p.center_nm[t] + p.size_nm[t] / 2)))
            i_lo, i_hi = max(i_lo, grid.interior[t][0]), min(i_hi, grid.interior[t][1])
            if i_hi - i_lo < 4:
                raise ValueError(f"port {p.name!r}: the window spans {i_hi - i_lo} cells inside the domain along "
                                 f"{'xyz'[t]}; make it wider or the grid finer")
            lo.append(i_lo)
            n.append(i_hi - i_lo)
        widths = (None, None)
        if grid.graded is not None:
            widths = tuple(None if nn == 1 else tuple(grid.prim(t, l_, l_ + nn).tolist())
                           for t, l_, nn in ((t1, lo[0], n[0]), (t2, lo[1], n[1])))
        out.append(PortFrame(name=p.name, a=a, t1=t1, t2=t2, sgn=sgn, d=d, k_p=k_p, k_m=k_m, lo=tuple(lo),
                             n=tuple(n), s_ref=sgn * p.center_nm[a], s_m=sgn * grid.coord(a, k_m),
                             num_modes=p.num_modes, shape=grid.shape, widths=widths, ds=ds))
    return out


def _slab_mask(idx3: np.ndarray, frames: list[PortFrame]) -> np.ndarray:
    """Nodes (n, 3) in the slab around any port's source and monitor planes (window plus two cells)."""
    m = np.zeros(len(idx3), dtype=bool)
    for f in frames:
        k0, k1 = min(f.k_p, f.k_m) - SLAB, max(f.k_p, f.k_m) + SLAB
        sel = (idx3[:, f.a] >= k0) & (idx3[:, f.a] <= k1)
        for t, lo, n in ((f.t1, f.lo[0], f.n[0]), (f.t2, f.lo[1], f.n[1])):
            if n > 1:
                sel &= (idx3[:, t] >= lo - 2) & (idx3[:, t] <= lo + n + 2)
        m |= sel
    return m


# --- materials -----------------------------------------------------------------------------------


def _media(problem, wl, warnings) -> tuple[list[Medium], list[int]]:
    media = [to_medium(problem.background, wl, warnings)]
    keys = {problem.background.model_dump_json(): 0}
    slots = []
    for s in problem.structures:
        k = s.material.model_dump_json()
        if k not in keys:
            keys[k] = len(media)
            media.append(to_medium(s.material, wl, warnings))
        slots.append(keys[k])
    return media, slots


def _bbox(geometry) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(geometry, Box):
        c, s = np.asarray(geometry.center_nm, float), np.asarray(geometry.size_nm, float)
        return c - s / 2, c + s / 2
    if isinstance(geometry, Ball):
        c = np.asarray(geometry.center_nm, float)
        return c - geometry.radius_nm, c + geometry.radius_nm
    pr = geometry.profile
    if isinstance(pr, Rectangle):
        c, s = np.asarray(pr.center_nm, float), np.asarray(pr.size_nm, float)
        lo, hi = c - s / 2, c + s / 2
    elif isinstance(pr, Polygon):
        v = np.asarray(pr.vertices_nm, float)
        lo, hi = v.min(axis=0), v.max(axis=0)
    else:  # circles and ellipses: their largest radius
        r = getattr(pr, "radius_nm", None) or max(pr.semi_axes_nm)
        lo, hi = np.asarray(pr.center_nm, float) - r, np.asarray(pr.center_nm, float) + r
    return np.array([lo[0], lo[1], geometry.z_min_nm]), np.array([hi[0], hi[1], geometry.z_max_nm])


def _slot_fn(problem, slots: list[int], plane_z: float | None):
    """pts -> medium index; later structures override earlier ones. 2D devices are evaluated on plane_z.
    Each structure only tests the points inside its bounding box."""
    boxes = [_bbox(s.geometry) for s in problem.structures]

    def fn(pts):
        pts = np.asarray(pts, dtype=float)
        if plane_z is not None:
            pts = pts.copy()
            pts[..., 2] = plane_z
        flat = pts.reshape(-1, 3)
        out = np.zeros(len(flat), dtype=int)
        for s, slot, (lo, hi) in zip(problem.structures, slots, boxes):
            sel = np.nonzero(np.all((flat >= lo - 1e-9) & (flat <= hi + 1e-9), axis=1))[0]
            if len(sel):
                p = flat[sel]
                out[sel[inside_geometry(s.geometry, p[:, 0], p[:, 1], p[:, 2])]] = slot
        return out.reshape(pts.shape[:-1])
    return fn


def _band(wl):
    """Requested band (omega_1, omega_2), the pulse centre and width, and the source band (omega_a, omega_b)."""
    om = 2 * np.pi / np.asarray(wl, dtype=float)
    w1, w2 = float(om.min()), float(om.max())
    wc = 0.5 * (w1 + w2)
    sigma = max((w2 - w1) / 3, 0.05 * wc)
    wa = max(w1 - 4.2 * sigma, 0.5 * w1)  # the pulse is down to e^-17.6 at the edges (and tapered to 0 there)
    wb = w2 + 4.2 * sigma
    return w1, w2, wc, sigma, wa, wb


def plan(problem, opts: FDTDOptions, single: bool) -> dict:
    problem = _resolved(problem)
    wl = wavelength_array(problem.wavelengths)
    grid = make_grid(problem, opts)
    frames = port_frames(problem, grid)
    warnings: list[str] = []
    media, _ = _media(problem, wl, warnings)
    eps_min = min([1.0] + [m.eps_inf for m in media])
    dims = 2 if grid.two_d else 3
    if grid.graded is None:
        dt = opts.courant * grid.dx * math.sqrt(eps_min) / math.sqrt(dims)
    else:  # the smallest cell per axis sets it (the invariant axis of a 2D run has no differences)
        h = [grid.min_spacing(a) for a in range(dims)]
        dt = opts.courant * math.sqrt(eps_min) / math.sqrt(sum(1 / v**2 for v in h))
    w1, w2, wc, sigma, wa, wb = _band(wl)
    stride = opts.dft_stride or max(1, int(2 * np.pi / ((wb + w2) * dt)))
    max_steps = int(math.ceil((opts.run_time_fs or opts.max_time_fs) * FS / dt / stride)) * stride
    runs = sum(f.num_modes for f in frames)
    n_mon = sum(4 * f.n[0] * f.n[1] for f in frames)
    mem = (4 if single else 8) * (12 * grid.cells + 2 * n_mon * len(wl))
    return {"grid": grid, "frames": frames, "dt": dt, "stride": stride, "max_steps": max_steps, "runs": runs,
            "band": (w1, w2, wc, sigma, wa, wb), "memory_bytes": int(mem), "media": media, "warnings": warnings}


def uses_single(opts: FDTDOptions, device: Device) -> bool:
    if opts.precision != "auto":
        return opts.precision == "single"
    return device.is_gpu


# --- solve ---------------------------------------------------------------------------------------


def _drop_offdiag(od: OffDiagonal | None, frames) -> tuple[OffDiagonal | None, int]:
    if od is None:
        return None, 0
    m = _slab_mask(od.idx3, frames)
    if not m.any():
        return od, 0
    keep = ~m
    return (OffDiagonal(od.idx3[keep], od.comp[keep], od.coef[keep]) if keep.any() else None), int(m.sum())


def _eps_fn(frame: PortFrame, inv_eps, groups, dt):
    comps = [(frame.t1, frame.k_p), (frame.t2, frame.k_p), (frame.a, frame.half(frame.k_p))]
    flats = [frame.nodes(c, k) for c, k in comps]

    def fn(omega):
        eps = np.stack([1.0 / node_eta(inv_eps, groups, f, omega, dt) for f in flats], axis=1)  # (n_w, 3, n)
        return eps.reshape(len(np.atleast_1d(omega)), 3, *frame.n)
    return fn


def solve(problem, opts: FDTDOptions, device: Device) -> SParameterResult:
    t_start = time.perf_counter()
    ok, why = supports(problem)
    if not ok:
        raise ValueError(why)
    problem = _resolved(problem)
    single = uses_single(opts, device)
    p = plan(problem, opts, single)
    grid, frames, dt, stride = p["grid"], p["frames"], p["dt"], p["stride"]
    media, warnings = p["media"], list(p["warnings"])
    if grid.cells > opts.max_cells:
        raise ValueError(f"grid {grid.shape} has {grid.cells} cells > max_cells={opts.max_cells}; "
                         "increase spacing_nm or raise max_cells")
    wl = wavelength_array(problem.wavelengths)
    omega = 2 * np.pi / wl
    _, slots = _media(problem, wl, [])
    w1, w2, wc, sigma, wa, wb = p["band"]
    dx = grid.engine_spacing()

    t0 = time.perf_counter()
    slot_fn = _slot_fn(problem, slots, problem.domain_center_nm[2] if grid.two_d else None)
    roi_lo = np.array([1, 1, 1 if grid.two_d else 1])
    roi_hi = np.array(grid.shape) - 1
    mode = opts.mesh_refinement or ("subpixel" if opts.subpixel else "staircase")

    def vox(m):
        return voxelise(slot_fn, media, grid.shape, grid.origin, dx, roi_lo, roi_hi, mode=m,
                        samples=opts.subpixel_samples, periodic=grid.periodic, grid=grid.graded)

    inv_eps, groups, vstats, offdiag = vox(mode)
    tm = next((g for g in groups if isinstance(g, TensorMesh)), None)
    if tm is not None and _slab_mask(tm.node_idx3, frames).any():
        warnings.append("a port crosses a dispersive interface: the run uses subpixel (diagonal) smoothing, "
                        "as port modes need a diagonal permittivity")
        mode = "subpixel"
        inv_eps, groups, vstats, offdiag = vox(mode)
    offdiag, n_dropped = _drop_offdiag(offdiag, frames)
    t_vox = time.perf_counter() - t0

    t0 = time.perf_counter()
    pol = None
    if grid.two_d:
        pol = {"te": "x", "tm": "y"}[opts.polarization or "te"]
    tables = []
    for f in frames:
        tables.append(ModeTable(f, f.ds, dt, wa, wb, _eps_fn(f, inv_eps, groups, dt), polarization=pol))
    t_modes = time.perf_counter() - t0

    n_bg = float(np.sqrt(max(media[0].eps_inf, 1.0)))
    pml = PMLSpec(cells=tuple((0, 0) if per else (grid.n_pml, grid.n_pml) for per in grid.periodic),
                  kappa_max=opts.pml_kappa, alpha_max=0.05 * 2 * np.pi * n_bg / float(wl.max()),
                  eps_ref=media[0].eps_inf)
    n_max = _n_max(problem, wl)
    transit = 1.5 * n_max * float(np.sum([s for s in problem.domain_size_nm]))
    labels = [lab for f in frames for lab in f.labels()]
    n_out = len(labels)
    S = np.zeros((len(wl), n_out, n_out), dtype=complex)
    runs, back = [], 0.0
    col = 0
    for fi, (f, tab) in enumerate(zip(frames, tables)):
        for m in range(f.num_modes):
            src = ModeSource(tab, m, inv_eps=inv_eps, dt=dt, omega_c=wc, sigma=sigma,
                             n_steps=p["max_steps"] + 4 * stride)
            mons = [DFTMonitor(omega, g.e_nodes(g.k_m), g.h_nodes(g.k_m), g.h_nodes(g.k_m), energy=True)
                    for g in frames]
            eng = Engine(device, shape=grid.shape, dx=dx, dt=dt, single=single, inv_eps=inv_eps, groups=groups,
                         pml=pml, sources=[src], monitors=mons, dft_stride=stride, periodic=grid.periodic,
                         offdiag=offdiag, widths=grid.engine_widths())
            t_loop = time.perf_counter()
            steps, level, graphs = run_until_decayed(
                eng, opts, max_steps=p["max_steps"], min_steps=int((src.t_off + transit) / dt),
                window=math.ceil(2 * np.pi * n_max / (w1 * stride * dt)), warnings=warnings)
            t_loop = time.perf_counter() - t_loop
            amp_in, amp_out = [], []
            for g, gt, mon in zip(frames, tables, mons):
                e, h = mon.results()
                for mm in range(g.num_modes):
                    a_, c_ = gt.amplitudes(omega, e, h, mm)
                    amp_in.append(a_)
                    amp_out.append(c_)
            row0 = sum(g.num_modes for g in frames[:fi])
            a_src = amp_in[row0 + m]
            S[:, :, col] = np.stack(amp_out, axis=1) / a_src[:, None]
            others = [i for i in range(n_out) if i != row0 + m]
            if others:
                back = max(back, float(np.max(np.abs(np.stack([amp_in[i] for i in others], axis=1))
                                              / np.abs(a_src)[:, None])))
            runs.append({"source": labels[col], "steps": steps, "decay_level": level, "time_loop_s": t_loop,
                         "cuda_graphs": graphs, "source_rank": list(src.rank)})
            del eng
            col += 1
    if back > 1e-2:
        warnings.append(f"waves enter the device through non-source ports at up to {back:.1e} of the source "
                        "amplitude (reflections from the CPML or the domain edge); thicken the PML (pml_cells)")
    lossless = all(not m.dispersive for m in media)
    if lossless:
        excess = float((np.abs(S) ** 2).sum(axis=1).max() - 1)
        if excess > 1e-2:
            warnings.append(f"output power exceeds input by {excess:.2%} in a lossless device; refine the grid")
    edge = []  # field fraction on the window's edge, over the requested band (the source band's ends spread more)
    for tab in tables:
        sel = (tab.nodes >= w1 - sigma) & (tab.nodes <= w2 + sigma)
        edge.append(float(tab.confinement[sel].max() if sel.any() else tab.confinement.max()))
    for f, tab, e in zip(frames, tables, edge):
        if tab.interp_error > 1e-5:
            warnings.append(f"port {f.name!r}: mode table interpolation error {tab.interp_error:.1e} "
                            "(the mode changes fast over the source band, e.g. near cutoff)")
        if e > 1e-2:
            warnings.append(f"port {f.name!r}: {e:.1%} of the mode's field is on the window "
                            "edge; widen the port")
    provenance = {
        **device.describe(), "grid": list(grid.shape), "cells": grid.cells, "spacing_nm": grid.dx,
        **({"graded_grid": {**grid.graded.describe(), "base_spacing_nm": grid.dx}} if grid.graded is not None else {}),
        "dt_fs": dt / FS, "dft_stride": stride, "precision": "single" if single else "double",
        "dimension": problem.dimension, **vstats, "mesh_refinement": mode, "offdiagonal_dropped_at_ports": n_dropped,
        "source_band_nm": [2 * np.pi / wb, 2 * np.pi / wa],
        "ports": {f.name: {"plane_nm": grid.coord(f.a, f.k_p), "window_cells": list(f.n), "normal_spacing_nm": f.ds,
                           "neff": (np.real(tab.at(omega)[2]) / omega[:, None]).tolist(),  # discrete beta / k0
                           "table_nodes": len(tab.nodes), "interp_error": tab.interp_error,
                           "edge_fraction": e}
                  for f, tab, e in zip(frames, tables, edge)},
        "runs": runs, "backflow": back, "time_voxelise_s": t_vox, "time_modes_s": t_modes,
        "runtime_s": time.perf_counter() - t_start,
    }
    return SParameterResult(method="fdtd", backend="lightsmith-fdtd", wavelengths_nm=wl.tolist(), ports=labels,
                            s_real=S.real.tolist(), s_imag=S.imag.tolist(), provenance=provenance,
                            warnings=warnings)
