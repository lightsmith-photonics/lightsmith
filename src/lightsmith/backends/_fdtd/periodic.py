"""Periodic structures and planar stacks: R, T, A and diffraction orders, at any angle of incidence.

The unit cell is the computational domain: x and y are periodic (ghost planes), z ends in CPML.
Cells may be non-cubic so that each period holds a whole number of cells. A one-sided TFSF plane
in the superstrate launches the pulse; everything above it is the reflected field alone. Running
DFTs of the tangential fields on a plane above the source (reflection) and just below the stack
(transmission), split into diffraction orders by a 2D FFT over the cell, give per-order
efficiencies; their sum is the total flux.

Normal incidence: real fields, plain periodic ghosts, and one broadband pulse gives every wavelength.

Oblique incidence uses the constant-transverse-wavenumber (fixed k_par) method: complex fields
with Bloch-periodic ghosts, F(x + L) = F(x) exp(i k_par L), and a TFSF plane fed by an auxiliary
one-cell Bloch column (``tfsf.BlochPlaneSource``) that solves the same discrete equations, so the
source is exact on the grid. It keeps the usual Courant limit and works unchanged with the
dispersive ADE media, subpixel smoothing and CPML, for conical incidence (phi != 0) and 2D
lattices alike. The split-field (field-transformation) alternative gives a fixed-angle spectrum
from one broadband run, but its time step shrinks with angle, every dispersive ADE path and the
PML need split-field versions, and it degrades towards grazing angles. Sine-cosine is the same
fixed-k_par method run as two real simulations instead of one complex one.

Fixed k_par vs fixed angle: one run at fixed k_par covers a band, but each frequency in it sees a
different angle (sin theta = k_par / (n_sup k0)). Results here are on the requested grid:
R[angle][j] is the response at vacuum wavelength wavelengths[j] and angle of incidence ``angle``
in the superstrate, from its own run with k_par = n_sup (2 pi / wavelengths[j]) sin(angle). That
run's pulse is centred on wavelengths[j] and kept clear of the frequency at which the zeroth order
turns grazing in the superstrate or substrate (omega = c k_par / n), where fields ring for a long
time. Cost: one run per (oblique angle, wavelength, polarisation), each sampling one frequency,
instead of one broadband run per polarisation (``estimate`` reports the count).

Geometry (the conventions of lightsmith.problems): layers stacked along +z from the superstrate
side, the first layer's top face at z = 0; the unit cell spans [-P/2, P/2) along each lattice axis;
the incident in-plane wavevector is n_sup k0 sin(theta) (cos phi, sin phi); s polarisation has E
along (-sin phi, cos phi, 0), p has its in-plane part along (cos phi, sin phi, 0). Diffraction order
(m, n) has in-plane wavevector k_par + m b1 + n b2 (as in RCWA).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from ...device import Device
from ...problems import LayerStackProblem, PeriodicProblem, wavelength_array
from ...raster import inside_profile
from ...results import DiffractionOrder, SpectrumResult
from .engine import DFTMonitor, Engine, PMLSpec, flat_index
from .media import Medium, to_medium
from .options import FDTDOptions
from .setup import FS, limit_dt, mesh_mode, run_until_decayed, voxelise
from .tfsf import BlochPlaneSource, TFSFPlaneWave

BIG = 10**9
MAX_THETA_DEG = 85.0


# --- support checks -------------------------------------------------------------


def _const_real_eps(material, wl) -> float | None:
    try:
        e = np.asarray(material.eps(wl), dtype=complex)
    except Exception:  # noqa: BLE001
        return None
    if np.all(np.abs(e.imag) < 1e-12) and np.ptp(e.real) < 1e-9 * max(1.0, abs(e.real[0])) and e.real[0] >= 1:
        return float(e.real[0])
    return None


def supports(problem) -> tuple[bool, str]:
    wl = wavelength_array(problem.wavelengths)
    if _const_real_eps(problem.superstrate, wl) is None:
        return False, ("FDTD needs a lossless, non-dispersive superstrate (constant real eps >= 1) "
                       "for its plane-wave source")
    grazing = (f"FDTD needs angles of incidence within {MAX_THETA_DEG:g} deg of the normal (grazing incidence "
               "needs ever narrower-band pulses)")
    if isinstance(problem, LayerStackProblem):
        if any(abs(a) > MAX_THETA_DEG for a in problem.angles_deg):
            return False, grazing + "; use 'tmm'"
        if any(not lay.coherent for lay in problem.layers):
            return False, "FDTD treats every layer coherently; use TMM for incoherent layers"
        return True, ""
    if abs(problem.incidence.theta_deg) > MAX_THETA_DEG:
        return False, grazing + "; use 'rcwa'"
    a1, a2 = np.asarray(problem.lattice_a1_nm, float), problem.lattice_a2_nm
    if abs(a1[1]) > 1e-9 * abs(a1[0]) or a1[0] <= 0:
        return False, "FDTD needs lattice_a1_nm along +x, e.g. (period, 0)"
    if a2 is not None and (abs(a2[0]) > 1e-9 * abs(a2[1]) or a2[1] <= 0):
        return False, "FDTD needs a rectangular lattice: lattice_a2_nm along +y, e.g. (0, period)"
    return True, ""


# --- set-up ------------------------------------------------------------------------


@dataclass
class _Cell:
    periods: tuple[float, float]  # nm; for a 1D grating / layer stack the cell width is nominal
    n_cells: tuple[int, int]  # real cells along x, y
    spacing: tuple[float, float, float]
    shape: tuple[int, int, int]  # including ghosts (x, y) and PEC planes (z)
    origin: np.ndarray
    k_r: int  # reflection plane (E nodes)
    k_src: int  # TFSF plane
    k_top: int  # node index of z = 0
    k_t: int  # transmission plane
    thickness: float
    pml: int

    @property
    def cells(self) -> int:
        return int(np.prod(self.shape))


def _materials(problem) -> list:
    out = [problem.superstrate, problem.substrate]
    for lay in problem.layers:
        if isinstance(problem, LayerStackProblem):
            out.append(lay.material)
        else:
            out.append(lay.background)
            out.extend(s.material for s in lay.shapes)
    return out


def _angles(problem) -> list[float]:
    return list(problem.angles_deg) if isinstance(problem, LayerStackProblem) else [problem.incidence.theta_deg]


def _polarizations(problem) -> list[str]:
    return list(problem.polarizations if isinstance(problem, LayerStackProblem) else problem.incidence.polarizations)


def _kpar(problem, wl, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Incident in-plane wavevector (kx, ky) per wavelength, rad/nm (zero at normal incidence)."""
    k = math.sqrt(_const_real_eps(problem.superstrate, wl)) * 2 * np.pi / np.asarray(wl, float)
    k = k * math.sin(math.radians(theta_deg))
    phi = 0.0 if isinstance(problem, LayerStackProblem) else math.radians(problem.incidence.phi_deg)
    return k * math.cos(phi), k * math.sin(phi)


def _band(wl):
    om = 2 * np.pi / np.asarray(wl, float)
    omega_c = 0.5 * (om.min() + om.max())
    return omega_c, max(0.5 * (om.max() - om.min()), 0.15 * omega_c), om.max()


def make_cell(problem, opts: FDTDOptions) -> _Cell:
    wl = wavelength_array(problem.wavelengths)
    n_max = 1.0
    for m in _materials(problem):
        try:
            n_max = max(n_max, float(np.abs(np.sqrt(np.asarray(m.eps(wl), dtype=complex))).max()))
        except Exception:  # noqa: BLE001 (range errors surface later, with context)
            pass
    d0 = opts.spacing_nm or wl.min() / (n_max * opts.cells_per_wavelength)
    thick = [lay.thickness_nm for lay in problem.layers if lay.thickness_nm > 0]
    dz = min(d0, min(thick) / 2) if (thick and not opts.spacing_nm) else d0
    # a one-cell (invariant) axis enters only through its Bloch phase difference at oblique incidence,
    # exp(i k d) - 1, which a narrower nominal cell makes closer to i k d (no effect at normal incidence)
    nominal = d0 / 4 if any(abs(a) > 1e-12 for a in _angles(problem)) else d0
    periods, ncell = [nominal, nominal], [1, 1]
    if isinstance(problem, PeriodicProblem):
        periods[0] = float(problem.lattice_a1_nm[0])
        ncell[0] = max(1, math.ceil(periods[0] / d0 - 1e-9))
        if problem.lattice_a2_nm is not None:
            periods[1] = float(problem.lattice_a2_nm[1])
            ncell[1] = max(1, math.ceil(periods[1] / d0 - 1e-9))
    dx, dy = periods[0] / ncell[0], periods[1] / ncell[1]
    H = float(sum(lay.thickness_nm for lay in problem.layers))
    n_pad = max(4, math.ceil((opts.padding_nm or 0) / dz))
    if opts.pml_gap_nm is not None:
        gap = opts.pml_gap_nm
    else:
        gap = max(_default_gap(problem, wl, periods, ncell, _kpar(problem, wl, a)) for a in _angles(problem))
    n_gap = max(4, math.ceil(gap / dz))
    n_pml = opts.pml_cells or 10
    k_r = n_pml + n_gap
    k_src = k_r + 2
    k_top = k_src + n_pad
    k_bot = k_top + math.ceil(H / dz - 1e-9)
    k_t = k_bot + 3
    nz = k_t + n_gap + n_pml + 1
    shape = (ncell[0] + 2, ncell[1] + 2, nz)
    origin = np.array([-periods[0] / 2 - dx, -periods[1] / 2 - dy, -k_top * dz])
    return _Cell((periods[0], periods[1]), (ncell[0], ncell[1]), (dx, dy, dz), shape, origin, k_r, k_src, k_top,
                 k_t, H, n_pml)


def _default_gap(problem, wl, periods, ncell, kpar=(0.0, 0.0)) -> float:
    """Monitor-to-PML distance: 3 decay lengths of the slowest evanescent diffraction order in the
    band (CPML does not absorb evanescent waves), clipped to [0.25, 1.5] x the longest wavelength."""
    eps = [np.real(np.asarray(m.eps(wl), dtype=complex)) for m in (problem.superstrate, problem.substrate)]
    lam_med = wl.max() / math.sqrt(max(1.0, float(np.min(eps[0]))))
    k0 = 2 * np.pi / wl
    kappa_min = np.inf
    for m in range(-ncell[0] // 2, ncell[0] // 2 + 1):
        for n in range(-ncell[1] // 2, ncell[1] // 2 + 1):
            if not (m or n):
                continue
            g2 = (kpar[0] + 2 * np.pi * m / periods[0]) ** 2 + (kpar[1] + 2 * np.pi * n / periods[1]) ** 2
            for e in eps:
                q2 = g2 - np.maximum(e, 0) * k0**2
                if np.any(q2 > 0):
                    kappa_min = min(kappa_min, float(np.sqrt(q2[q2 > 0].min())))
    if not np.isfinite(kappa_min):
        return 0.25 * lam_med
    return float(np.clip(3 / kappa_min, 0.25 * lam_med, 1.5 * wl.max()))


def _slot_fn(problem, slot_of_layer, slot_sup, slot_sub, cell: _Cell):
    """pts -> medium index. Patterned layers: later shapes override earlier ones (periodic images included)."""
    px, py = cell.periods
    tops = np.concatenate([[0.0], np.cumsum([lay.thickness_nm for lay in problem.layers])])
    one_d = isinstance(problem, LayerStackProblem) or problem.lattice_a2_nm is None
    shifts = [(sx, sy) for sx in (-px, 0.0, px) for sy in ((0.0,) if one_d else (-py, 0.0, py))]

    def fn(pts):
        x = (pts[..., 0] + px / 2) % px - px / 2
        # 1D gratings are invariant along y: profiles are evaluated on the line y = 0 (as RCWA does)
        y = np.zeros_like(pts[..., 1]) if one_d else (pts[..., 1] + py / 2) % py - py / 2
        z = pts[..., 2]
        out = np.full(z.shape, slot_sup, dtype=int)
        out[z >= tops[-1]] = slot_sub
        for i, layer in enumerate(problem.layers):
            sel = (z >= tops[i]) & (z < tops[i + 1])
            if not sel.any():
                continue
            bg, shapes = slot_of_layer[i]
            vals = np.full(int(sel.sum()), bg, dtype=int)
            xs, ys = x[sel], y[sel]
            for prof, s in shapes:
                hit = np.zeros(xs.shape, dtype=bool)
                for sx, sy in shifts:
                    hit |= inside_profile(prof, xs - sx, ys - sy)
                vals[hit] = s
            out[sel] = vals
        return out
    return fn


def _media(problem, wl, warnings):
    eps_sup = _const_real_eps(problem.superstrate, wl)
    media = [Medium(eps_sup, (), "superstrate")]
    keys = {problem.superstrate.model_dump_json(): 0}

    def slot(m):
        k = m.model_dump_json()
        if k not in keys:
            keys[k] = len(media)
            media.append(to_medium(m, wl, warnings))
        return keys[k]

    layers = []
    for lay in problem.layers:
        if isinstance(problem, LayerStackProblem):
            layers.append((slot(lay.material), []))
        else:
            layers.append((slot(lay.background), [(s.profile, slot(s.material)) for s in lay.shapes]))
    return media, layers, slot(problem.superstrate), slot(problem.substrate)


def _runs(problem, theta_deg: float) -> dict[str, str]:
    """Polarisation -> the run that serves it. A uniform stack at normal incidence is polarisation
    independent: one run serves both."""
    pols = _polarizations(problem)
    if isinstance(problem, LayerStackProblem) and pols and abs(theta_deg) <= 1e-12:
        return {pol: pols[0] for pol in pols}
    return {pol: pol for pol in pols}


def plan(problem, opts: FDTDOptions, single: bool) -> dict:
    wl = wavelength_array(problem.wavelengths)
    cell = make_cell(problem, opts)
    dt = _time_step(cell, opts)
    omega_c, omega_half, omega_max = _band(wl)
    stride = opts.dft_stride or max(1, int(2 * np.pi / (omega_c + 4 * omega_half + omega_max) / dt))
    max_t = (opts.run_time_fs or opts.max_time_fs) * FS
    max_steps = int(math.ceil(max_t / dt / stride)) * stride
    n_pts = 2 * 2 * cell.n_cells[0] * cell.n_cells[1]
    angles = list(dict.fromkeys(_angles(problem)))
    oblique = [a for a in angles if abs(a) > 1e-12]
    normal_runs = len(set(_runs(problem, 0.0).values())) if len(oblique) < len(angles) else 0
    oblique_runs = sum(len(set(_runs(problem, a).values())) for a in oblique) * len(wl)
    # oblique runs carry complex fields (2x the field memory)
    mem = (4 if single else 8) * (12 * cell.cells * (2 if oblique else 1) + 4 * n_pts * len(wl))
    # the longest oblique run: the largest angle at the shortest wavelength has the narrowest pulse
    steps_oblique = 0
    if oblique:
        th = max(abs(a) for a in oblique)
        kx, ky = _kpar(problem, wl, th)
        om = 2 * np.pi / wl
        j = int(np.argmin([_oblique_half_width(problem, o, math.hypot(a, b)) / o for o, a, b in zip(om, kx, ky)]))
        half = _oblique_half_width(problem, om[j], math.hypot(kx[j], ky[j]))
        transit = cell.shape[2] * cell.spacing[2] * math.sqrt(_const_real_eps(problem.superstrate, wl))
        steps_oblique = int((14 / half + transit / math.cos(math.radians(th))) / dt)
    return {"cell": cell, "dt": dt, "stride": stride, "max_steps": max_steps, "memory_bytes": int(mem),
            "omega_c": omega_c, "omega_half": omega_half, "runs": normal_runs + oblique_runs,
            "oblique_runs": oblique_runs, "oblique_run_min_steps": steps_oblique}


def _time_step(cell: _Cell, opts, kpar=(0.0, 0.0)) -> float:
    """Courant limit. A one-cell periodic axis carries a single Bloch wavenumber k, whose grid
    eigenvalue is 2 sin(k d / 2) / d instead of up to 2 / d."""
    inv2 = 1 / cell.spacing[2] ** 2
    for a in (0, 1):
        if cell.n_cells[a] > 1:
            inv2 += 1 / cell.spacing[a] ** 2
        elif kpar[a]:
            inv2 += (math.sin(kpar[a] * cell.spacing[a] / 2) / cell.spacing[a]) ** 2
    return opts.courant / math.sqrt(inv2)


def _dt(c, kpar=(0.0, 0.0)) -> float:
    """Time step of one run: the Courant limit scaled by sqrt(min eps_inf), lowered further where a
    dispersive medium's ADE update would be unstable (setup.limit_dt)."""
    dt0 = _time_step(c.cell, c.opts, kpar)
    return limit_dt(dt0 * c.dt_scale, c.opts.courant, c.media, 4 * (c.opts.courant / dt0) ** 2)


def _oblique_half_width(problem, omega: float, kpar: float) -> float:
    """Pulse half-bandwidth for a fixed-k_par run aimed at omega: a quarter of the distance to the
    nearest frequency where the zeroth order is grazing in the superstrate or substrate (the pulse
    spectrum there is e^-16 of its peak), within [0.01, 0.15] omega."""
    wl = np.array([2 * np.pi / omega])
    eps = [_const_real_eps(problem.superstrate, wl),
           float(np.real(np.asarray(problem.substrate.eps(wl), dtype=complex))[0])]
    gaps = [abs(omega - kpar / math.sqrt(e)) for e in eps if e > 0]
    return float(np.clip(min(gaps) / 4, 0.01 * omega, 0.15 * omega))


def _plane(cell: _Cell, k: int):
    """Flat indices for S_z = Ex Hy - Ey Hx on node plane k over the real cells (term 1, then term 2)."""
    i = np.arange(1, cell.n_cells[0] + 1)
    j = np.arange(1, cell.n_cells[1] + 1)
    gi, gj = np.meshgrid(i, j, indexing="ij")
    gi, gj = gi.ravel(), gj.ravel()
    e = np.concatenate([flat_index(0, gi, gj, k, cell.shape), flat_index(1, gi, gj, k, cell.shape)])
    h1 = np.concatenate([flat_index(1, gi, gj, k - 1, cell.shape), flat_index(0, gi, gj, k - 1, cell.shape)])
    h2 = np.concatenate([flat_index(1, gi, gj, k, cell.shape), flat_index(0, gi, gj, k, cell.shape)])
    return e, h1, h2


def _orders(e: np.ndarray, h: np.ndarray, cell: _Cell, kpar=(0.0, 0.0)):
    """Per-order z flux (unnormalised): (n_f, nx, ny) of 0.5 Re(E1 H1* - E2 H2*) summed over the cell. FFT bin
    (m, n) holds order k_par + m b1 + n b2: Bloch fields are demodulated by exp(-i k_par . r) first."""
    nx, ny = cell.n_cells
    n = nx * ny
    nf = e.shape[0]
    E1, E2 = e[:, :n].reshape(nf, nx, ny), e[:, n:].reshape(nf, nx, ny)
    H1, H2 = h[:, :n].reshape(nf, nx, ny), h[:, n:].reshape(nf, nx, ny)
    if any(kpar):
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        d = np.exp(-1j * (kpar[0] * i * cell.spacing[0] + kpar[1] * j * cell.spacing[1]))
        E1, E2, H1, H2 = E1 * d, E2 * d, H1 * d, H2 * d
    f = lambda a: np.fft.fft2(a, axes=(1, 2))  # noqa: E731
    flux = 0.5 * np.real(f(E1) * np.conj(f(H1)) - f(E2) * np.conj(f(H2))) / n * cell.spacing[0] * cell.spacing[1]
    mx = np.fft.fftfreq(nx, 1 / nx).astype(int)
    my = np.fft.fftfreq(ny, 1 / ny).astype(int)
    return flux, mx, my


# --- solve -----------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Everything shared by the runs of one solve."""

    problem: object
    opts: FDTDOptions
    device: Device
    single: bool
    cell: _Cell
    inv_eps: np.ndarray
    groups: list
    media: list
    pml: PMLSpec
    eps_sup: float
    dt_scale: float  # sqrt(min eps_inf) Courant factor
    area: float
    warnings: list
    offdiag: object = None  # engine.OffDiagonal

    def transit(self) -> float:
        return self.cell.shape[2] * self.cell.spacing[2] * max(math.sqrt(abs(m.eps_inf)) for m in self.media)


def _run_normal(c: _Ctx, pol: str, p: dict, omega: np.ndarray):
    """One broadband real run at normal incidence: per-order (r, t) of shape (n_wl, nx, ny)."""
    cell, opts = c.cell, c.opts
    phi = 0.0 if isinstance(c.problem, LayerStackProblem) else math.radians(c.problem.incidence.phi_deg)
    dt = _dt(c)
    stride = p["stride"]
    max_steps = int(math.ceil((opts.run_time_fs or opts.max_time_fs) * FS / dt / stride)) * stride
    lam_max = 2 * np.pi / omega.min()
    r0 = cell.origin + np.array([1, 1, cell.k_top]) * np.asarray(cell.spacing)
    ehat = np.array([-np.sin(phi), np.cos(phi), 0.0]) if pol == "s" else np.array([np.cos(phi), np.sin(phi), 0.0])
    src = TFSFPlaneWave(shape=cell.shape, origin=cell.origin, dx=cell.spacing, dt=dt, r0=r0,
                        lo=np.array([-BIG, -BIG, cell.k_src]), hi=np.array([BIG, BIG, BIG]),
                        khat=np.array([0.0, 0.0, 1.0]), ehat=ehat, eps_b=c.eps_sup, omega_c=p["omega_c"],
                        omega_half=p["omega_half"], max_steps=max_steps + 4 * stride, periodic=(True, True, False))
    mon_r = DFTMonitor(omega, *_plane(cell, cell.k_r), energy=True)
    mon_t = DFTMonitor(omega, *_plane(cell, cell.k_t), energy=True)
    mon_i = DFTMonitor(omega, *src.probe(), source=src)
    eng = Engine(c.device, shape=cell.shape, dx=cell.spacing, dt=dt, single=c.single, inv_eps=c.inv_eps,
                 groups=c.groups, pml=c.pml, sources=[src], monitors=[mon_r, mon_t, mon_i], dft_stride=stride,
                 periodic=(True, True, False), offdiag=c.offdiag)
    t_loop = time.perf_counter()
    steps, level, graphs = run_until_decayed(
        eng, opts, max_steps=max_steps, min_steps=int((src.t_passed + c.transit()) / dt),
        window=math.ceil(lam_max / (stride * dt)), warnings=c.warnings)
    t_loop = time.perf_counter() - t_loop
    e_i, h_i = mon_i.results()
    inc = 0.5 * np.real(e_i[:, 0] * np.conj(h_i[:, 0])) * c.area
    fr, mx, my = _orders(*mon_r.results(), cell)
    ft, _, _ = _orders(*mon_t.results(), cell)
    prov = {"polarization": pol, "angle_deg": 0.0, "steps": steps, "decay_level": level, "time_loop_s": t_loop,
            "cuda_graphs": graphs}
    return -fr / inc[:, None, None], ft / inc[:, None, None], (mx, my), prov


def _run_oblique(c: _Ctx, pol: str, kpar: tuple[float, float], wavelength: float):
    """One fixed-k_par complex run aimed at one wavelength: per-order (r, t) of shape (1, nx, ny)."""
    cell, opts = c.cell, c.opts
    omega = 2 * np.pi / wavelength
    k = math.hypot(*kpar)
    dt = _dt(c, kpar)
    half = _oblique_half_width(c.problem, omega, k)
    stride = opts.dft_stride or max(1, int(2 * np.pi / (2 * omega + 4 * half) / dt))
    max_steps = int(math.ceil((opts.run_time_fs or opts.max_time_fs) * FS / dt / stride)) * stride
    # the column is all superstrate: its CPML is graded for that medium at both ends
    src = BlochPlaneSource(shape=cell.shape, spacing=cell.spacing, dt=dt, k_src=cell.k_src,
                           k_launch=cell.k_src - 1, kpar=kpar, pol=pol, eps_b=c.eps_sup,
                           pml=replace(c.pml, eps_ref_high=None), omega_c=omega, omega_half=half,
                           max_steps=max_steps + 4 * stride)
    om = np.array([omega])
    mon_r = DFTMonitor(om, *_plane(cell, cell.k_r), energy=True)
    mon_t = DFTMonitor(om, *_plane(cell, cell.k_t), energy=True)
    mon_i = DFTMonitor(om, *src.probe(cell.k_top), source=src)
    eng = Engine(c.device, shape=cell.shape, dx=cell.spacing, dt=dt, single=c.single, inv_eps=c.inv_eps,
                 groups=c.groups, pml=c.pml, sources=[src], monitors=[mon_r, mon_t, mon_i], dft_stride=stride,
                 periodic=(True, True, False), bloch=(kpar[0], kpar[1], 0.0), offdiag=c.offdiag)
    cos_t = math.sqrt(max(1 - k**2 / (c.eps_sup * omega**2), 0.01))  # slower transit along z
    t_loop = time.perf_counter()
    steps, level, graphs = run_until_decayed(
        eng, opts, max_steps=max_steps, min_steps=int((src.t_passed + c.transit() / cos_t) / dt),
        window=math.ceil(wavelength / (stride * dt)), warnings=c.warnings)
    t_loop = time.perf_counter() - t_loop
    e_i, h_i = mon_i.results()
    inc = 0.5 * np.real(e_i[:, 0] * np.conj(h_i[:, 0]) - e_i[:, 1] * np.conj(h_i[:, 1])) * c.area
    fr, _, _ = _orders(*mon_r.results(), cell, kpar)
    ft, _, _ = _orders(*mon_t.results(), cell, kpar)
    prov = {"polarization": pol, "wavelength_nm": float(wavelength), "steps": steps, "decay_level": level,
            "time_loop_s": t_loop, "cuda_graphs": graphs}
    return -fr / inc[:, None, None], ft / inc[:, None, None], prov


def solve(problem, opts: FDTDOptions, device: Device) -> SpectrumResult:
    t_start = time.perf_counter()
    ok, why = supports(problem)
    if not ok:
        raise ValueError(why)
    warnings: list[str] = []
    wl = wavelength_array(problem.wavelengths)
    single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu)
    media, layer_slots, s_sup, s_sub = _media(problem, wl, warnings)
    p = plan(problem, opts, single)
    cell = p["cell"]
    if cell.cells > opts.max_cells:
        raise ValueError(f"grid {cell.shape} has {cell.cells} cells > max_cells={opts.max_cells}")
    lam_max = wl.max()

    t0 = time.perf_counter()
    roi_lo = np.array([1, 1, max(1, cell.k_top - 3)])
    roi_hi = np.array([cell.shape[0] - 1, cell.shape[1] - 1, cell.shape[2] - 1])
    inv_eps, groups, vstats, offdiag = voxelise(
        _slot_fn(problem, layer_slots, s_sup, s_sub, cell), media, cell.shape, cell.origin, cell.spacing, roi_lo,
        roi_hi, mode=mesh_mode(opts), samples=opts.subpixel_samples, periodic=(True, True, False))
    t_vox = time.perf_counter() - t0

    eps_sup = media[0].eps_inf
    eps_sub_ref = float(np.real(media[s_sub].eps(np.array([p["omega_c"]]))[0]))
    pml = PMLSpec(cells=((0, 0), (0, 0), (cell.pml, cell.pml)), kappa_max=opts.pml_kappa,
                  alpha_max=0.05 * 2 * np.pi * math.sqrt(eps_sup) / lam_max, eps_ref=eps_sup,
                  eps_ref_high=max(1.0, eps_sub_ref))
    area = cell.n_cells[0] * cell.spacing[0] * cell.n_cells[1] * cell.spacing[1]
    ctx = _Ctx(problem, opts, device, single, cell, inv_eps, groups, media, pml, eps_sup,
               math.sqrt(min([1.0] + [m.eps_inf for m in media])), area, warnings, offdiag)
    pols, angles = _polarizations(problem), _angles(problem)
    omega = 2 * np.pi / wl
    nx, ny = cell.n_cells
    mx = np.fft.fftfreq(nx, 1 / nx).astype(int)
    my = np.fft.fftfreq(ny, 1 / ny).astype(int)

    # per (polarisation, angle index): per-order r, t of shape (n_wl, nx, ny), and k_par per wavelength
    r_ord, t_ord, kpars = {}, {}, {}
    done: dict[tuple, tuple] = {}  # (run polarisation, angle) -> (r, t)
    prov_runs = []
    for ia, theta in enumerate(angles):
        kpars[ia] = _kpar(problem, wl, theta)
        for pol, run_pol in _runs(problem, theta).items():
            key = (run_pol, float(theta))
            if key not in done:
                if abs(theta) <= 1e-12:
                    r, t, _, pr = _run_normal(ctx, run_pol, p, omega)
                    prov_runs.append(pr)
                else:
                    r = np.zeros((len(wl), nx, ny))
                    t = np.zeros((len(wl), nx, ny))
                    for j, lam in enumerate(wl):
                        r[j:j + 1], t[j:j + 1], pr = _run_oblique(
                            ctx, run_pol, (float(kpars[ia][0][j]), float(kpars[ia][1][j])), float(lam))
                        prov_runs.append({**pr, "angle_deg": float(theta)})
                done[key] = (r, t)
            r_ord[pol, ia], t_ord[pol, ia] = done[key]

    # orders propagating in the superstrate or substrate at some wavelength and angle
    k0 = omega
    eps_sub_band = np.real(np.asarray(problem.substrate.eps(wl), dtype=complex))
    n2 = np.maximum(eps_sup, eps_sub_band)
    gx, gy = 2 * np.pi / cell.periods[0], 2 * np.pi / cell.periods[1]
    one_d = isinstance(problem, LayerStackProblem) or problem.lattice_a2_nm is None

    def kpar2(m, n, ia):
        return (kpars[ia][0] + m * gx) ** 2 + (kpars[ia][1] + n * gy) ** 2

    orders = []
    for ii, m in enumerate(mx):
        for jj, n in enumerate(my):
            if isinstance(problem, LayerStackProblem) and (m or n):
                continue
            if one_d and n:
                continue
            if any(np.any(kpar2(m, n, ia) < n2 * k0**2) for ia in range(len(angles))):
                orders.append(((int(m), int(n)), ii, jj))
    # evanescent orders carry no power; what flux the DFT finds in them is PML/near-field noise
    for (pol, ia) in r_ord:
        r_ord[pol, ia] = r_ord[pol, ia].copy()
        t_ord[pol, ia] = t_ord[pol, ia].copy()
        for (m, n), ii, jj in orders:
            r_ord[pol, ia][kpar2(m, n, ia) >= eps_sup * k0**2, ii, jj] = 0.0
            t_ord[pol, ia][kpar2(m, n, ia) >= eps_sub_band * k0**2, ii, jj] = 0.0

    def total(d, pol, ia):
        return sum(d[pol, ia][:, ii, jj] for _, ii, jj in orders) if orders else np.zeros(len(wl))

    r_tot = {pol: [total(r_ord, pol, ia).tolist() for ia in range(len(angles))] for pol in pols}
    t_tot = {pol: [total(t_ord, pol, ia).tolist() for ia in range(len(angles))] for pol in pols}
    a_tot = {pol: [(1 - np.asarray(r_tot[pol][ia]) - np.asarray(t_tot[pol][ia])).tolist()
                   for ia in range(len(angles))] for pol in pols}
    diff_orders = None
    if isinstance(problem, PeriodicProblem):
        diff_orders = [DiffractionOrder(order=mn,
                                        r={pol: [r_ord[pol, ia][:, ii, jj].tolist() for ia in range(len(angles))]
                                           for pol in pols},
                                        t={pol: [t_ord[pol, ia][:, ii, jj].tolist() for ia in range(len(angles))]
                                           for pol in pols})
                       for mn, ii, jj in orders]
    lossless = all(np.all(np.abs(np.imag(np.asarray(m.eps(wl), dtype=complex))) < 1e-12) for m in _materials(problem))
    if lossless and pols:
        a_max = max(float(np.max(np.abs(a_tot[pol]))) for pol in pols)
        if a_max > 1e-2:
            warnings.append(f"energy balance off by {a_max:.1e} for a lossless structure; refine the grid "
                            "or lower decay_by")
    oblique = any(abs(a) > 1e-12 for a in angles)
    notes = "R and T are fluxes on planes above the source and below the stack"
    if oblique:
        notes += ("; oblique angles: one fixed-k_par (Bloch-periodic, complex-field) run per (angle, wavelength, "
                  "polarisation), k_par = n_sup k0 sin(angle) matched to that wavelength, so every value is at "
                  "the stated angle and vacuum wavelength")
    prov = {**device.describe(), "grid": list(cell.shape), "real_cells": list(cell.n_cells) + [cell.shape[2]],
            "cells": cell.cells, "spacing_nm": list(cell.spacing), "dt_fs": _dt(ctx) / FS,
            "dft_stride": p["stride"], "precision": "single" if single else "double",
            "subpixel": mesh_mode(opts) != "staircase", "mesh_refinement": mesh_mode(opts),
            "pml_cells": cell.pml, "runs": prov_runs, "steps": sum(r["steps"] for r in prov_runs),
            "decay_level": max(r["decay_level"] for r in prov_runs) if prov_runs else None,
            "materials": [m.describe() for m in media], **vstats, "voxelise_s": t_vox,
            "runtime_s": time.perf_counter() - t_start, "notes": notes}
    return SpectrumResult(method="fdtd", backend="lightsmith-fdtd", wavelengths_nm=wl.tolist(), angles_deg=angles,
                          polarizations=pols, r=r_tot, t=t_tot, a=a_tot, orders=diff_orders, provenance=prov,
                          warnings=list(dict.fromkeys(warnings)))
