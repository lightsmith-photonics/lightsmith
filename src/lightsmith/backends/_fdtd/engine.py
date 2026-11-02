"""Device-side 3D Yee FDTD engine, written once against the Python array API.

The engine knows nothing about particles or plane waves. It takes

* a material description on the Yee grid: per-component update coefficients
  (non-dispersive cells, subpixel-smoothed) plus groups of dispersive cells
  integrated with auxiliary differential equations in D-form;
* boundaries per axis: CPML with a PEC backstop, or periodic. A periodic axis with a nonzero Bloch
  wavevector k makes the fields complex (one Bloch mode, exp(i k x) times a periodic function: the
  constant-transverse-wavenumber method); k = 0 keeps them real;
* sources (objects with ``correct_H``/``advance_H``/``correct_E``/``advance_E`` hooks,
  e.g. the TFSF plane wave in ``tfsf.py``);
* DFT monitors (flat-index point sets on the stacked E and H arrays).

Normalised units: nm, c = eps0 = mu0 = 1 (time in nm/c). Fields are stacked
as E[3, N0, N1, N2] and H[3, N0, N1, N2]. Yee positions in cell units:
E_c at node + e_c/2, H_c at node + (1 - e_c)/2. The outermost node planes are PEC.

On PyTorch CUDA a chunk of steps (plus one DFT sample) is captured once as a
CUDA graph and replayed, which removes the kernel-launch overhead that
otherwise dominates. All state is therefore updated in place and time enters
only through a device step counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...device import Device


def shift(R: tuple, d: int, s: int) -> tuple:
    R = list(R)
    R[d] = slice(R[d].start + s, R[d].stop + s)
    return tuple(R)


def e_region(c: int, shape, periodic=(False, False, False)) -> tuple:
    """Index region of E_c that is time-stepped (tangential E on the outer planes stays 0: PEC).

    Along a periodic axis the arrays carry one ghost plane at each end (index 0 and n-1);
    the real cells are 1..n-2 and the ghosts are refreshed every half step."""
    return tuple(slice(1, n - 1) if (periodic[a] or a != c) else slice(0, n - 1) for a, n in enumerate(shape))


def h_region(c: int, shape, periodic=(False, False, False)) -> tuple:
    return tuple(slice(1, n - 1) if periodic[a] else (slice(0, n) if a == c else slice(0, n - 1))
                 for a, n in enumerate(shape))


def flat_index(c, i, j, k, shape) -> np.ndarray:
    n0, n1, n2 = shape
    return ((np.asarray(c) * n0 + np.asarray(i)) * n1 + np.asarray(j)) * n2 + np.asarray(k)


def pole_coefficients(pole, dt: float) -> tuple[float, float, float]:
    """P^{n+1} = c1 P^n + c2 P^{n-1} + c3 E^n for P'' + g P' + w0^2 P = sigma E (central differences)."""
    den = 1 + pole.gamma * dt / 2
    return ((2 - pole.omega0**2 * dt**2) / den, -(1 - pole.gamma * dt / 2) / den, pole.sigma * dt**2 / den)


# --- host-side descriptions ---------------------------------------------------


@dataclass
class OffDiagonal:
    """Off-diagonal inverse-permittivity terms of tensor subpixel smoothing (Kottke et al., PRE 77, 036611 (2008)).

    Row r is E_comp[r] at node idx3[r], with (1/eps)_cd = coef[r, d]: E_c couples to curl_d H at the four E_d
    positions around it. Engine._build_offdiag symmetrises the coupling between each pair of cells."""

    idx3: np.ndarray  # (n, 3) node indices
    comp: np.ndarray  # (n,) component c
    coef: np.ndarray  # (n, 3) (1/eps)_cd; column c is unused


@dataclass
class DispersiveGroup:
    """E cells integrated in D-form. ``media[s]`` with fraction ``frac[s]``; ``w`` = n_c^2 weight of the
    harmonic (series) path, None for pure cells (one medium, f = 1)."""

    idx: np.ndarray  # flat indices into E (3*N0*N1*N2)
    media: list
    frac: np.ndarray  # (S, n)
    w: np.ndarray | None


@dataclass
class PMLSpec:
    cells: tuple[tuple[int, int], ...]  # (low, high) CPML cells per axis; 0 = none
    m: float = 3.0
    kappa_max: float = 1.0
    alpha_max: float = 0.0  # CFS alpha (normalised units)
    eps_ref: float = 1.0  # sigma_opt is scaled for this permittivity
    eps_ref_high: float | None = None  # a different one for the high side of every axis (e.g. a substrate)


@dataclass
class DFTMonitor:
    """Running DFT of E at ``e_idx`` and of 0.5*(H[h_idx1] + H[h_idx2]) on the engine's (or a source's) arrays."""

    omega: np.ndarray
    e_idx: np.ndarray
    h_idx1: np.ndarray | None = None
    h_idx2: np.ndarray | None = None
    energy: bool = False  # contributes to the decay-shutoff energy proxy
    source: object | None = None  # sample this object's E_flat/H_flat instead of the grid
    acc_e: object = field(default=None, repr=False)
    acc_h: object = field(default=None, repr=False)

    def results(self) -> tuple[np.ndarray, np.ndarray | None]:
        e = Device.to_numpy(self.acc_e).astype(complex)
        h = None if self.acc_h is None else Device.to_numpy(self.acc_h).astype(complex)
        return e, h


@dataclass
class TimeMonitor:
    """E at ``e_idx`` recorded at every DFT sample (every ``dft_stride`` steps), up to ``n_max`` samples."""

    e_idx: np.ndarray
    n_max: int
    buf: object = field(default=None, repr=False)
    pos: object = field(default=None, repr=False)

    def results(self, n: int) -> np.ndarray:
        """(channels, samples) for the first n samples."""
        return Device.to_numpy(self.buf)[:n].T


# --- the engine ---------------------------------------------------------------


class Engine:
    def __init__(self, device: Device, *, shape, dx, dt: float, single: bool, inv_eps: np.ndarray,
                 groups: list[DispersiveGroup], pml: PMLSpec, sources: list, monitors: list[DFTMonitor],
                 dft_stride: int, periodic=(False, False, False), bloch=(0.0, 0.0, 0.0),
                 offdiag: OffDiagonal | None = None, time_monitors: tuple = ()):
        """dx: cell size, or (dx, dy, dz) for non-cubic cells. periodic: per axis; those axes carry ghost
        planes (see e_region) and no PML. bloch: Bloch wavevector per periodic axis (rad/nm); any nonzero
        entry makes E, H and all auxiliary fields complex. offdiag: tensor-smoothing terms of
        non-dispersive interface cells."""
        self.device, self.xp = device, device.namespace()
        xp = self.xp
        self.shape, self.dt, self.single = tuple(int(n) for n in shape), float(dt), single
        self.spacing = tuple(float(v) for v in np.broadcast_to(np.asarray(dx, dtype=float), (3,)))
        self.uniform = len(set(self.spacing)) == 1
        self.dx = self.spacing[0]
        self.periodic = tuple(bool(p) for p in periodic)
        self.bloch = tuple(float(k) for k in bloch)
        if any(k and not per for k, per in zip(self.bloch, self.periodic)):
            raise ValueError("a Bloch wavevector needs a periodic axis")
        self.complex = any(self.bloch)
        self.rdt = xp.float32 if single else xp.float64
        self.cdt = xp.complex64 if single else xp.complex128
        self.stride = int(dft_stride)
        dev = device.native_device()
        self._dev = dev
        self.E = self.fzeros((3,) + self.shape)
        self.H = self.fzeros((3,) + self.shape)
        self.E_flat = xp.reshape(self.E, (-1,))
        self.H_flat = xp.reshape(self.H, (-1,))
        self.n = device.asarray(np.zeros(1, dtype=np.int64))  # completed steps (device counter)
        self.n_host = 0

        # uniform cells: coefficients carry dt/dx; otherwise dt, and differences are scaled by 1/d_axis
        cdx = self.dt / self.dx if self.uniform else self.dt
        self.inv_d = tuple(1.0 / d for d in self.spacing)
        self.RE = [e_region(c, self.shape, self.periodic) for c in range(3)]
        self.RH = [h_region(c, self.shape, self.periodic) for c in range(3)]
        self.cbE = [self.asarray(cdx * inv_eps[c][self.RE[c]]) for c in range(3)]
        self.cH = cdx
        self._build_pml(pml, inv_eps)
        self._build_offdiag(offdiag, groups)
        self.groups = [_Group(self, g) for g in groups]
        self.sources = sources
        for s in sources:
            s.bind(self)
        self.monitors = monitors
        for m in monitors:
            self._bind_monitor(m)
        self.time_monitors = list(time_monitors)
        for m in self.time_monitors:
            m._e = self.index(m.e_idx)
            m.buf = self.fzeros((m.n_max, len(m.e_idx)))
            m.pos = self.device.asarray(np.zeros(1, dtype=np.int64))
        self.energy = self.zeros((1,))

    # array helpers
    def zeros(self, shape, complex_=False):
        return self.xp.zeros(tuple(shape), dtype=self.cdt if complex_ else self.rdt, device=self._dev)

    def fzeros(self, shape):
        """Zeros of the field dtype (complex under a Bloch phase, else real)."""
        return self.zeros(shape, complex_=self.complex)

    def asarray(self, a, dtype=None):
        return self.device.asarray(np.asarray(a), dtype=dtype or self.rdt)

    def index(self, a):
        return self.device.asarray(np.asarray(a, dtype=np.int64), dtype=self.xp.int64)

    # --- CPML -------------------------------------------------------------------
    def _build_pml(self, spec: PMLSpec, inv_eps):
        """Precompute CPML terms. Each term: (target, comp, src, axis, slab region, b, c', k' or None, psi)."""
        self.pml_E, self.pml_H = [], []
        dt = self.dt
        for d in range(3):
            lo_n, hi_n = spec.cells[d]
            if not (lo_n or hi_n):
                continue
            if self.periodic[d]:
                raise ValueError("a periodic axis cannot have a PML")
            N = self.shape[d]
            dx = self.spacing[d]
            sig_lo = 0.8 * (spec.m + 1) / (np.sqrt(spec.eps_ref) * dx)
            sig_hi = 0.8 * (spec.m + 1) / (np.sqrt(spec.eps_ref_high or spec.eps_ref) * dx)

            def profile(u):
                r_lo = np.clip((lo_n - u) / lo_n, 0, 1) if lo_n else np.zeros_like(u)
                r_hi = np.clip((u - (N - 1 - hi_n)) / hi_n, 0, 1) if hi_n else np.zeros_like(u)
                rho = np.maximum(r_lo, r_hi)
                sig = np.where(r_lo >= r_hi, sig_lo, sig_hi) * rho**spec.m
                kap = 1 + (spec.kappa_max - 1) * rho**spec.m
                alp = spec.alpha_max * (1 - rho) * (rho > 0)
                b = np.exp(-(sig / kap + alp) * dt)
                den = sig * kap + kap**2 * alp
                c = np.where(sig > 0, sig * (b - 1) / np.where(den > 0, den, 1), 0.0)
                return b, c, kap

            for field_, comps in (("E", self.RE), ("H", self.RH)):
                for c in range(3):
                    if c == d:
                        continue
                    d1, d2 = (c + 1) % 3, (c + 2) % 3
                    # E_c: +dH_{d2}/dd1 - dH_{d1}/dd2 ; H_c: -dE_{d2}/dd1 + dE_{d1}/dd2
                    if d == d1:
                        src, sgn = d2, 1.0
                    else:
                        src, sgn = d1, -1.0
                    if field_ == "H":
                        sgn = -sgn
                    R = comps[c]
                    start, stop = R[d].start, R[d].stop
                    half = 0.5 if field_ == "H" else 0.0
                    ranges = []
                    if lo_n:
                        ranges.append((start, min(stop, lo_n + 1)))
                    if hi_n:
                        ranges.append((max(start, N - 2 - hi_n), stop))
                    for a, b_ in ranges:
                        if b_ <= a:
                            continue
                        Rs = list(R)
                        Rs[d] = slice(a, b_)
                        Rs = tuple(Rs)
                        u = np.arange(a, b_) + half
                        bb, cc, kk = profile(u)
                        bshape = [1, 1, 1]
                        bshape[d] = b_ - a
                        cd = dt / dx if self.uniform else dt
                        if field_ == "E":
                            scale = cd * inv_eps[c][Rs]  # slab-shaped
                        else:
                            scale = np.full([s.stop - s.start for s in Rs], cd)
                        if not self.uniform:
                            scale = scale / dx
                        cprime = sgn * scale * cc.reshape(bshape)
                        kprime = sgn * scale * (1 / kk - 1).reshape(bshape)
                        has_k = bool(np.any(kk > 1))
                        term = (c, src, d, Rs, self.asarray(bb.reshape(bshape)), self.asarray(cprime),
                                self.asarray(kprime) if has_k else None, self.fzeros(cprime.shape))
                        (self.pml_E if field_ == "E" else self.pml_H).append(term)

    # --- tensor smoothing ---------------------------------------------------------
    def _build_offdiag(self, od: OffDiagonal | None, groups: list[DispersiveGroup]):
        """Gather tables (e, h, w): row r adds sum_k w[r, k] * H_flat[h[r, k]] to E_flat[e[r]].

        E_c at p couples to D_d (= the time integral of curl_d H) at the four E_d nodes q around it. The
        coupling K(p, q) is symmetrised, K = (o_cd(p) + o_dc(q)) / 8 for both directions (Hermitian under a
        Bloch phase), so the discrete 1/eps operator stays symmetric; the one-sided Meep-style average
        o_cd(p) / 4 is not, and grows slowly at late times. Cells integrated in D-form (dispersive) take no
        off-diagonal terms. Periodic axes wrap q onto the real cells with the Bloch phase of the period it
        crossed."""
        self.od = None
        if od is None or not len(od.comp):
            return
        shape = self.shape
        nflat = 3 * int(np.prod(shape))
        idx3 = np.asarray(od.idx3, dtype=np.int64)
        tgt, srcs, vals = [], [], []
        t_flat = flat_index(od.comp, idx3[:, 0], idx3[:, 1], idx3[:, 2], shape)
        for d in range(3):
            for corner in range(4):
                q = idx3.copy()
                q += np.eye(3, dtype=np.int64)[od.comp] * (corner & 1)  # + e_c
                q[:, d] -= corner >> 1  # - e_d
                phase = np.ones(len(q), dtype=complex)
                ok = (od.comp != d) & (od.coef[:, d] != 0)
                for a in range(3):
                    n = shape[a]
                    if self.periodic[a]:
                        kl = self.bloch[a] * (n - 2) * self.spacing[a]
                        lo, hi = q[:, a] == 0, q[:, a] == n - 1
                        q[lo, a], q[hi, a] = n - 2, 1
                        phase[lo] *= np.exp(-1j * kl)
                        phase[hi] *= np.exp(1j * kl)
                    else:
                        rd = e_region(d, shape)[a]
                        ok &= (q[:, a] >= rd.start) & (q[:, a] < rd.stop)
                tgt.append(t_flat[ok])
                srcs.append(flat_index(d, q[ok, 0], q[ok, 1], q[ok, 2], shape))
                vals.append(od.coef[ok, d] / 8 * phase[ok])
        t, s_, v = np.concatenate(tgt), np.concatenate(srcs), np.concatenate(vals)
        t, s_, v = np.concatenate([t, s_]), np.concatenate([s_, t]), np.concatenate([v, np.conj(v)])
        if groups:
            disp = np.concatenate([g.idx for g in groups])
            keep = ~(np.isin(t, disp) | np.isin(s_, disp))
            t, s_, v = t[keep], s_[keep], v[keep]
        if not len(t):
            return
        key, inv = np.unique(t * nflat + s_, return_inverse=True)
        v = np.bincount(inv, v.real, len(key)) + 1j * np.bincount(inv, v.imag, len(key))
        t, s_ = key // nflat, key % nflat
        targets, start, count = np.unique(t, return_index=True, return_counts=True)
        slot = np.arange(len(t)) - np.repeat(start, count)
        row = np.repeat(np.arange(len(targets)), count)
        width = int(count.max())
        h = np.ones((len(targets), 4 * width), dtype=np.int64)  # padding: index 1, weight 0
        w = np.zeros((len(targets), 4 * width), dtype=complex)
        d, rest = np.divmod(s_, int(np.prod(shape)))
        q = np.stack(np.unravel_index(rest, shape), axis=1)
        d1, d2 = (d + 1) % 3, (d + 2) % 3
        inv_d = np.asarray(self.inv_d)
        eye = np.eye(3, dtype=np.int64)
        # curl_d H at q = (H_d2[q] - H_d2[q - e_d1]) / dx_d1 - (H_d1[q] - H_d1[q - e_d2]) / dx_d2
        for k, (comp, back, sgn, inv_ax) in enumerate(((d2, None, 1.0, d1), (d2, d1, -1.0, d1),
                                                       (d1, None, -1.0, d2), (d1, d2, 1.0, d2))):
            p = q if back is None else q - eye[back]
            h[row, 4 * slot + k] = flat_index(comp, p[:, 0], p[:, 1], p[:, 2], shape)
            w[row, 4 * slot + k] = self.dt * sgn * inv_d[inv_ax] * v
        if not self.complex:
            w = w.real
        self.od = (self.index(targets), self.index(h),
                   self.device.asarray(w, dtype=self.cdt if self.complex else self.rdt))

    # --- monitors ----------------------------------------------------------------
    def _bind_monitor(self, m: DFTMonitor):
        m._omega = self.device.asarray(np.asarray(m.omega, dtype=float), dtype=self.xp.float64)
        m._e = self.index(m.e_idx)
        m.acc_e = self.zeros((len(m.omega), len(m.e_idx)), complex_=True)
        if m.h_idx1 is not None:
            m._h1, m._h2 = self.index(m.h_idx1), self.index(m.h_idx2)
            m.acc_h = self.zeros((len(m.omega), len(m.h_idx1)), complex_=True)

    def _phase(self, m: DFTMonitor, t):
        wt = m._omega * t
        return self.xp.astype(self.xp.cos(wt) + 1j * self.xp.sin(wt), self.cdt)

    def _sample(self):
        xp = self.xp
        t_e = xp.astype(self.n, xp.float64) * self.dt
        t_h = t_e - 0.5 * self.dt
        energy = None
        for m in self.monitors:
            ef, hf = (self.E_flat, self.H_flat) if m.source is None else (m.source.E_flat, m.source.H_flat)
            se = ef[m._e]
            m.acc_e += self._phase(m, t_e)[:, None] * se[None, :]
            if m.acc_h is not None:
                sh = 0.5 * (hf[m._h1] + hf[m._h2])
                m.acc_h += self._phase(m, t_h)[:, None] * sh[None, :]
                if m.energy:
                    if self.complex:
                        v = xp.sum(xp.real(se * xp.conj(se))) + xp.sum(xp.real(sh * xp.conj(sh)))
                    else:
                        v = xp.sum(se * se) + xp.sum(sh * sh)
                    energy = v if energy is None else energy + v
        if energy is not None:
            self.energy[...] = energy
        for m in self.time_monitors:
            m.buf[m.pos] = self.E_flat[m._e][None, :]
            m.pos += 1

    # --- time stepping -----------------------------------------------------------
    def _ghosts(self, F, low: bool):
        """Refresh periodic ghost planes: E's high ghost (read by the H update), H's low ghost (E update).
        Under a Bloch wavevector k the ghost is its periodic image times exp(+-i k L), L the period."""
        for a, per in enumerate(self.periodic):
            if not per:
                continue
            n = self.shape[a]
            dst, src = (0, n - 2) if low else (n - 1, 1)
            ix_d = [slice(None)] * 4
            ix_s = [slice(None)] * 4
            ix_d[a + 1], ix_s[a + 1] = slice(dst, dst + 1), slice(src, src + 1)
            k = self.bloch[a]
            if k:
                kl = k * (n - 2) * self.spacing[a] * (-1 if low else 1)
                F[tuple(ix_d)] = F[tuple(ix_s)] * complex(np.cos(kl), np.sin(kl))
            else:
                F[tuple(ix_d)] = F[tuple(ix_s)]

    def _curl(self, F, c, R, sgn: int, coef):
        """coef * curl(F)_c over region R; sgn=+1 forward differences (H update), -1 backward (E update)."""
        d1, d2 = (c + 1) % 3, (c + 2) % 3
        if sgn > 0:
            a = F[d2][shift(R, d1, 1)] - F[d2][R]
            b = F[d1][shift(R, d2, 1)] - F[d1][R]
        else:
            a = F[d2][R] - F[d2][shift(R, d1, -1)]
            b = F[d1][R] - F[d1][shift(R, d2, -1)]
        if not self.uniform:
            a *= self.inv_d[d1]
            b *= self.inv_d[d2]
        a -= b
        a *= coef
        return a

    def _update_H(self):
        E, H = self.E, self.H
        if any(self.periodic):
            self._ghosts(E, low=False)
        for c in range(3):
            H[c][self.RH[c]] -= self._curl(E, c, self.RH[c], 1, self.cH)
        for c, src, d, Rs, b, cp, kp, psi in self.pml_H:
            der = E[src][shift(Rs, d, 1)] - E[src][Rs]
            psi *= b
            if kp is not None:
                H[c][Rs] += kp * der
            der *= cp
            psi += der
            H[c][Rs] += psi

    def _update_E(self):
        E, H = self.E, self.H
        if any(self.periodic):
            self._ghosts(H, low=True)
        for c in range(3):
            E[c][self.RE[c]] += self._curl(H, c, self.RE[c], -1, self.cbE[c])
        if self.od is not None:
            e_idx, h_idx, w = self.od
            self.E_flat[e_idx] += self.xp.sum(w * self.H_flat[h_idx], axis=1)
        for c, src, d, Rs, b, cp, kp, psi in self.pml_E:
            der = H[src][Rs] - H[src][shift(Rs, d, -1)]
            psi *= b
            if kp is not None:
                E[c][Rs] += kp * der
            der *= cp
            psi += der
            E[c][Rs] += psi

    def step(self):
        self._update_H()
        for s in self.sources:
            s.correct_H(self)
        for s in self.sources:
            s.advance_H(self)
        self._update_E()
        for s in self.sources:
            s.correct_E(self)
        for g in self.groups:
            g.step(self)
        for s in self.sources:
            s.advance_E(self)
        self.n += 1

    def chunk(self):
        for _ in range(self.stride):
            self.step()
        self._sample()

    def make_runner(self, use_graphs: bool):
        """Return (run_one_chunk, used_graphs)."""
        if not (use_graphs and self.device.framework == "torch" and self.device.is_gpu):
            return self.chunk, False
        import torch

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self._graph_state = state = {"graph": None, "warm": 0}

        def run():
            if state["graph"] is None:
                if state["warm"] < 1:  # the warm-up chunk is real steps, run on a side stream
                    with torch.cuda.stream(stream):
                        self.chunk()
                    torch.cuda.current_stream().wait_stream(stream)
                    state["warm"] += 1
                    return
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.chunk()  # recorded, not executed
                state["graph"] = g
            state["graph"].replay()

        return run, True

    def release(self):
        """Drop a captured CUDA graph (and its private memory pool) once the run is over."""
        state = getattr(self, "_graph_state", None)
        if state and state["graph"] is not None:
            import torch

            torch.cuda.synchronize()
            state["graph"].reset()
            state["graph"] = None


class _Group:
    """Dispersive cells, D-form ADE (Meep-style): D += dt curl H, P^{n+1} from E^n, E = (D - P)/eps_inf.

    Mixed (interface) cells use anisotropic subpixel smoothing with two parallel paths:
    1/eps_cc = w <1/eps> + (1 - w) / <eps>, w = n_c^2. The harmonic path inverts each
    medium separately (series combination); the arithmetic path inverts the
    fraction-weighted mixture. Both reduce to the standard update for pure cells."""

    def __init__(self, eng: Engine, g: DispersiveGroup):
        dt = eng.dt
        n = len(g.idx)
        self.idx = eng.index(g.idx)
        self.D = eng.fzeros((n,))
        self.E_prev = eng.fzeros((n,))
        self.pure = g.w is None
        self.coef = [[pole_coefficients(p, dt) for p in m.poles] for m in g.media]
        if self.pure:
            (m,) = g.media
            self.inv_eps_inf = 1.0 / m.eps_inf
            self.P = [(eng.fzeros((n,)), eng.fzeros((n,))) for _ in m.poles]
            return
        f = g.frac
        self.disp = [s for s, m in enumerate(g.media) if m.dispersive]
        h0 = sum(f[s] / m.eps_inf for s, m in enumerate(g.media) if not m.dispersive) + 0 * f[0]
        self.h0 = eng.asarray(h0)
        self.f = {s: eng.asarray(f[s]) for s in self.disp}
        self.inv_eps_inf = {s: 1.0 / g.media[s].eps_inf for s in self.disp}
        self.inv_eps_a = eng.asarray(1.0 / sum(f[s] * m.eps_inf for s, m in enumerate(g.media)))
        self.w = eng.asarray(g.w)
        self.has_h = bool(np.any(g.w > 0))
        self.has_a = bool(np.any(g.w < 1))
        self.Eh = {s: eng.fzeros((n,)) for s in self.disp}
        self.Ea = eng.fzeros((n,))
        self.Ph = {s: [(eng.fzeros((n,)), eng.fzeros((n,))) for _ in g.media[s].poles] for s in self.disp}
        self.Pa = {s: [(eng.fzeros((n,)), eng.fzeros((n,))) for _ in g.media[s].poles] for s in self.disp}
        self.media_idx = list(range(len(g.media)))

    @staticmethod
    def _advance(pairs, coefs, e):
        tot = None
        for (P, Pp), (c1, c2, c3) in zip(pairs, coefs):
            new = c1 * P + c2 * Pp + c3 * e
            Pp[...] = P
            P[...] = new
            tot = P if tot is None else tot + P
        return tot

    def step(self, eng: Engine):
        e_now = eng.E_flat[self.idx]
        self.D += e_now - self.E_prev
        if self.pure:
            tot = self._advance(self.P, self.coef[0], self.E_prev)
            e_new = (self.D - tot) * self.inv_eps_inf
        else:
            parts = []
            if self.has_h:
                harm = self.D * self.h0
                for s in self.disp:
                    tot = self._advance(self.Ph[s], self.coef[s], self.Eh[s])
                    self.Eh[s][...] = (self.D - tot) * self.inv_eps_inf[s]
                    harm = harm + self.f[s] * self.Eh[s]
                parts.append(self.w * harm)
            if self.has_a:
                acc = self.D
                for s in self.disp:
                    tot = self._advance(self.Pa[s], self.coef[s], self.Ea)
                    acc = acc - self.f[s] * tot
                self.Ea[...] = acc * self.inv_eps_a
                parts.append((1 - self.w) * self.Ea)
            e_new = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        eng.E_flat[self.idx] = e_new
        self.E_prev[...] = e_new
