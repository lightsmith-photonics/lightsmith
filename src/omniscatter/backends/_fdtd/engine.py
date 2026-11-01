"""Device-side 3D Yee FDTD engine, written once against the Python array API.

The engine knows nothing about particles or plane waves. It takes

* a material description on the Yee grid: per-component update coefficients
  (non-dispersive cells, subpixel-smoothed) plus groups of dispersive cells
  integrated with auxiliary differential equations in D-form;
* boundaries per axis (CPML with a PEC backstop; Bloch-periodic is a planned addition);
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


def e_region(c: int, shape) -> tuple:
    """Index region of E_c that is time-stepped (tangential E on the outer planes stays 0: PEC)."""
    return tuple(slice(0, n - 1) if a == c else slice(1, n - 1) for a, n in enumerate(shape))


def h_region(c: int, shape) -> tuple:
    return tuple(slice(0, n) if a == c else slice(0, n - 1) for a, n in enumerate(shape))


def flat_index(c, i, j, k, shape) -> np.ndarray:
    n0, n1, n2 = shape
    return ((np.asarray(c) * n0 + np.asarray(i)) * n1 + np.asarray(j)) * n2 + np.asarray(k)


def pole_coefficients(pole, dt: float) -> tuple[float, float, float]:
    """P^{n+1} = c1 P^n + c2 P^{n-1} + c3 E^n for P'' + g P' + w0^2 P = sigma E (central differences)."""
    den = 1 + pole.gamma * dt / 2
    return ((2 - pole.omega0**2 * dt**2) / den, -(1 - pole.gamma * dt / 2) / den, pole.sigma * dt**2 / den)


# --- host-side descriptions ---------------------------------------------------


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


# --- the engine ---------------------------------------------------------------


class Engine:
    def __init__(self, device: Device, *, shape, dx: float, dt: float, single: bool, inv_eps: np.ndarray,
                 groups: list[DispersiveGroup], pml: PMLSpec, sources: list, monitors: list[DFTMonitor],
                 dft_stride: int):
        self.device, self.xp = device, device.namespace()
        xp = self.xp
        self.shape, self.dx, self.dt, self.single = tuple(int(n) for n in shape), float(dx), float(dt), single
        self.rdt = xp.float32 if single else xp.float64
        self.cdt = xp.complex64 if single else xp.complex128
        self.stride = int(dft_stride)
        dev = device.native_device()
        self._dev = dev
        self.E = self.zeros((3,) + self.shape)
        self.H = self.zeros((3,) + self.shape)
        self.E_flat = xp.reshape(self.E, (-1,))
        self.H_flat = xp.reshape(self.H, (-1,))
        self.n = device.asarray(np.zeros(1, dtype=np.int64))  # completed steps (device counter)
        self.n_host = 0

        cdx = self.dt / self.dx
        self.RE = [e_region(c, self.shape) for c in range(3)]
        self.RH = [h_region(c, self.shape) for c in range(3)]
        self.cbE = [self.asarray(cdx * inv_eps[c][self.RE[c]]) for c in range(3)]
        self.cH = cdx
        self._build_pml(pml, inv_eps)
        self.groups = [_Group(self, g) for g in groups]
        self.sources = sources
        for s in sources:
            s.bind(self)
        self.monitors = monitors
        for m in monitors:
            self._bind_monitor(m)
        self.energy = self.zeros((1,))

    # array helpers
    def zeros(self, shape, complex_=False):
        return self.xp.zeros(tuple(shape), dtype=self.cdt if complex_ else self.rdt, device=self._dev)

    def asarray(self, a, dtype=None):
        return self.device.asarray(np.asarray(a), dtype=dtype or self.rdt)

    def index(self, a):
        return self.device.asarray(np.asarray(a, dtype=np.int64), dtype=self.xp.int64)

    # --- CPML -------------------------------------------------------------------
    def _build_pml(self, spec: PMLSpec, inv_eps):
        """Precompute CPML terms. Each term: (target, comp, src, axis, slab region, b, c', k' or None, psi)."""
        self.pml_E, self.pml_H = [], []
        dt, dx = self.dt, self.dx
        for d in range(3):
            lo_n, hi_n = spec.cells[d]
            if not (lo_n or hi_n):
                continue
            N = self.shape[d]
            sig_max_lo = 0.8 * (spec.m + 1) / (np.sqrt(spec.eps_ref) * dx)

            def profile(u):
                rho = np.zeros_like(u)
                if lo_n:
                    rho = np.maximum(rho, (lo_n - u) / lo_n)
                if hi_n:
                    rho = np.maximum(rho, (u - (N - 1 - hi_n)) / hi_n)
                rho = np.clip(rho, 0, 1)
                sig = sig_max_lo * rho**spec.m
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
                        if field_ == "E":
                            scale = (dt / dx) * inv_eps[c][Rs]  # slab-shaped
                        else:
                            scale = np.full([s.stop - s.start for s in Rs], dt / dx)
                        cprime = sgn * scale * cc.reshape(bshape)
                        kprime = sgn * scale * (1 / kk - 1).reshape(bshape)
                        has_k = bool(np.any(kk > 1))
                        term = (c, src, d, Rs, self.asarray(bb.reshape(bshape)), self.asarray(cprime),
                                self.asarray(kprime) if has_k else None, self.zeros(cprime.shape))
                        (self.pml_E if field_ == "E" else self.pml_H).append(term)

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
                    v = xp.sum(se * se) + xp.sum(sh * sh)
                    energy = v if energy is None else energy + v
        if energy is not None:
            self.energy[...] = energy

    # --- time stepping -----------------------------------------------------------
    def _update_H(self):
        E, H = self.E, self.H
        for c in range(3):
            d1, d2 = (c + 1) % 3, (c + 2) % 3
            R = self.RH[c]
            curl = E[d2][shift(R, d1, 1)] - E[d2][R]
            curl -= E[d1][shift(R, d2, 1)]
            curl += E[d1][R]
            curl *= self.cH
            H[c][R] -= curl
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
        for c in range(3):
            d1, d2 = (c + 1) % 3, (c + 2) % 3
            R = self.RE[c]
            curl = H[d2][R] - H[d2][shift(R, d1, -1)]
            curl -= H[d1][R]
            curl += H[d1][shift(R, d2, -1)]
            curl *= self.cbE[c]
            E[c][R] += curl
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
        xp, dt = eng.xp, eng.dt
        n = len(g.idx)
        self.idx = eng.index(g.idx)
        self.D = eng.zeros((n,))
        self.E_prev = eng.zeros((n,))
        self.pure = g.w is None
        self.coef = [[pole_coefficients(p, dt) for p in m.poles] for m in g.media]
        if self.pure:
            (m,) = g.media
            self.inv_eps_inf = 1.0 / m.eps_inf
            self.P = [(eng.zeros((n,)), eng.zeros((n,))) for _ in m.poles]
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
        self.Eh = {s: eng.zeros((n,)) for s in self.disp}
        self.Ea = eng.zeros((n,))
        self.Ph = {s: [(eng.zeros((n,)), eng.zeros((n,))) for _ in g.media[s].poles] for s in self.disp}
        self.Pa = {s: [(eng.zeros((n,)), eng.zeros((n,))) for _ in g.media[s].poles] for s in self.disp}
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
