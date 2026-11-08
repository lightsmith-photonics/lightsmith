"""Native discrete dipole approximation (DDA), CPU or GPU.

Dipoles sit on a cubic lattice. The interaction matrix is block-Toeplitz, so
matrix-vector products use FFTs on a zero-padded grid (Goodman, Draine &
Flatau 1991) and the system is solved with BiCGStab. Written against the
Python array API, it runs unchanged on NumPy, PyTorch (CPU/CUDA) or CuPy.

Physics follows Draine & Flatau, JOSA A 11, 1491 (1994): Gaussian units,
exp(-i omega t), lattice-dispersion-relation (LDR) polarizability
(Draine & Goodman, ApJ 405, 685 (1993)), and

    C_ext = 4 pi k sum Im(E_inc^* . P)
    C_abs = 4 pi k sum [ -Im(1/alpha) |P|^2 - (2/3) k^3 |P|^2 ]
    dC_sca/dOmega = k^4 |sum (I - nn) P_j exp(-i k n.r_j)|^2

Because DDA converges slowly for metals, `extrapolate` runs several lattice
resolutions and extrapolates to zero dipole size (Yurkin, Maltsev & Hoekstra,
JOSA A 23, 2578 (2006)).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..progress import report
from ..results import FarFieldData, NearFieldData, Result
from ..spec import Simulation
from .base import Backend, register, unsupported_outputs

LDR_B1, LDR_B2, LDR_B3 = -1.8915316, 0.1648469, -1.7700004


class DDAOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dipoles_per_diameter: int | None = Field(
        None, ge=2, description="Lattice sites across the smallest particle's largest dimension. "
        "Default: max(16, enough for |m| k d <= 0.5).")
    polarizability: Literal["ldr", "cm_rr"] = Field("ldr", description="LDR, or Clausius-Mossotti + radiative reaction.")
    discretization: Literal["center", "weighted"] = Field(
        "center", description="center: classic staircase with volume rescaling. weighted: boundary sites mix "
        "full-site polarizabilities by volume fraction; as accurate on dielectrics, but BiCGStab often fails to "
        "converge on metals.")
    precision: Literal["auto", "single", "double"] = Field("auto", description="auto = double on CPU, single on GPU.")
    tol: float | None = Field(None, gt=0, description="Relative residual target. Default 1e-8 double, 1e-5 single.")
    max_iter: int = Field(10000, ge=1)
    max_dipoles: int = Field(5_000_000, ge=1, description="Refuse problems larger than this.")
    orientations: int = Field(32, ge=1, description="Incidence directions (x2 polarizations) for orientation_average.")
    extrapolate: list[int] | None = Field(
        None, min_length=2, description="dipoles_per_diameter values to extrapolate to zero dipole size, "
        "e.g. [12, 16, 24]. Quadratic in 1/npd for 3+ values, linear for 2. Improves smooth (dielectric) "
        "ladders; staircase noise makes it unreliable for metals.")


# --- geometry ---------------------------------------------------------------


@dataclass
class _Lattice:
    n: tuple[int, int, int]
    frac: np.ndarray  # (n_slots, nx, ny, nz) volume fraction of each material slot per site
    d: float
    pts: np.ndarray  # (nx, ny, nz, 3) site positions in nm
    npd: int
    materials: list  # Material per slot

    @property
    def occupied(self) -> np.ndarray:
        return self.frac.sum(axis=0) > 0

    @property
    def n_dip(self) -> int:
        return int(self.occupied.sum())


def _default_npd(sim: Simulation) -> int:
    wl = sim.wavelength_array()
    k_max = 2 * np.pi * sim.medium_index / wl.min()
    m_max = max(np.abs(m.n(wl) / sim.medium_index).max() for p in sim.particles for m in p.region_materials())
    diam = min(2 * p.shape.bounding_half_extent().max() for p in sim.particles)
    return max(16, math.ceil(diam * m_max * k_max / 0.5))


def _slots(sim: Simulation, pts: np.ndarray, materials: list | None = None) -> np.ndarray:
    """Material slot per point (-1 = outside); the first particle listed wins overlaps."""
    slot = np.full(pts.shape[:-1], -1, dtype=int)
    s0 = 0
    for p in sim.particles:
        region = p.region(pts - np.asarray(p.position_nm))
        mats = p.region_materials()
        for r in range(len(mats)):
            slot[(region == r) & (slot < 0)] = s0 + r
        if materials is not None:
            materials.extend(mats)
        s0 += len(mats)
    return slot


def _lattice(sim: Simulation, npd: int, discretization: str = "weighted", subsamples: int = 4) -> _Lattice:
    """Voxelise all particles (and their shells) on one lattice.

    "center": a site is filled if its centre is inside; the lattice is then rescaled
    so the dipole volume equals the true volume. "weighted": each site carries the
    volume fraction of every material (supersampled; cf. Piller, Opt. Lett. 24, 1001
    (1999)). On metals its strongly mixed boundary sites leave BiCGStab unconverged at
    most resolutions, so it does not remove the staircase noise there in practice.
    """
    diam = min(2 * p.shape.bounding_half_extent().max() for p in sim.particles)
    d0 = diam / npd
    lo = np.min([np.asarray(p.position_nm) - p.shape.bounding_half_extent() for p in sim.particles], axis=0)
    hi = np.max([np.asarray(p.position_nm) + p.shape.bounding_half_extent() for p in sim.particles], axis=0)
    n = np.maximum(np.round((hi - lo) / d0).astype(int), 1)
    if discretization == "weighted":
        n = n + 1  # room for partially filled boundary sites
    centre = (lo + hi) / 2
    axes = [centre[i] + (np.arange(n[i]) - (n[i] - 1) / 2) * d0 for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    materials: list = []
    slot = _slots(sim, pts, materials)
    frac = np.zeros((len(materials),) + tuple(n))
    if discretization == "center":
        for s in range(len(materials)):
            frac[s] = slot == s
        n_dip = int((slot >= 0).sum())
        if n_dip == 0:
            raise ValueError("no dipoles inside the particles; increase dipoles_per_diameter")
        d = (sum(p.shape.volume() for p in sim.particles) / n_dip) ** (1 / 3)
        pts = centre + (pts - centre) * (d / d0)
    else:
        sub = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * d0
        for dx in sub:
            for dy in sub:
                for dz in sub:
                    s_off = _slots(sim, pts + np.array([dx, dy, dz]))
                    for s in range(len(materials)):
                        frac[s] += s_off == s
        frac /= subsamples**3
        d = d0
        if not frac.any():
            raise ValueError("no dipoles inside the particles; increase dipoles_per_diameter")
    return _Lattice(tuple(int(v) for v in n), frac, d, pts, npd, materials)


def _green(r: np.ndarray, k: float) -> np.ndarray:
    """Dipole field tensor G for separations r[..., 3] (zero at r=0), shape (3, 3, ...).

    G = e^{ikr}/r [k^2 (I - rr) + (1 - ikr)/r^2 (3 rr - I)]
    """
    rr = np.sqrt((r**2).sum(axis=-1))
    zero = rr == 0
    rr = np.where(zero, 1.0, rr)
    rh = np.moveaxis(r, -1, 0) / rr
    pref = np.exp(1j * k * rr) / rr
    b = (1 - 1j * k * rr) / rr**2
    eye = np.eye(3)
    g = np.empty((3, 3) + rr.shape, dtype=complex)
    for i in range(3):
        for j in range(i, 3):
            rij = rh[i] * rh[j]
            g[i, j] = np.where(zero, 0.0, pref * (k**2 * (eye[i, j] - rij) + b * (3 * rij - eye[i, j])))
            g[j, i] = g[i, j]
    return g


def _interaction_fft(n, d, k, device, dtype):
    """FFT of G on the doubled grid, shape (3, 3, 2nx, 2ny, 2nz)."""
    offs = []
    for m in n:
        o = np.where(np.arange(2 * m) < m, np.arange(2 * m), np.arange(2 * m) - 2 * m)
        o[m] = 0  # offset -n never occurs; G(0) = 0 there
        offs.append(o)
    r = np.stack(np.meshgrid(*offs, indexing="ij"), axis=-1).astype(float) * d
    return device.asarray(np.fft.fftn(_green(r, k), axes=(2, 3, 4)), dtype=dtype)


def _polarizability(eps_rel, d, k, khat, ehat, kind):
    a_cm = 3 * d**3 / (4 * np.pi) * (eps_rel - 1) / (eps_rel + 2)
    if kind == "cm_rr":
        return a_cm / (1 - 2j / 3 * k**3 * a_cm)
    s = float(np.sum((khat * ehat) ** 2))
    return a_cm / (1 + (a_cm / d**3) * ((LDR_B1 + eps_rel * LDR_B2 + eps_rel * LDR_B3 * s) * (k * d) ** 2
                                        - 2j / 3 * (k * d) ** 3))


def _bicgstab(matvec, b, x, xp, tol, max_iter):
    vdot = lambda u, v: xp.sum(xp.conj(u) * v)  # noqa: E731
    norm = lambda u: float(xp.sqrt(xp.real(vdot(u, u))))  # noqa: E731
    bnorm = norm(b)
    r = b - matvec(x)
    r0 = r
    rho = alpha = omega = 1.0
    v = p = xp.zeros_like(b)
    res = norm(r) / bnorm
    for it in range(1, max_iter + 1):
        if res < tol:
            return x, it - 1, res
        rho_new = vdot(r0, r)
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = matvec(p)
        alpha = rho_new / vdot(r0, v)
        s = r - alpha * v
        if norm(s) / bnorm < tol:
            return x + alpha * p, it, norm(s) / bnorm
        t = matvec(s)
        omega = vdot(t, s) / vdot(t, t)
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        res = norm(r) / bnorm
    return x, max_iter, res


def _orientation_set(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fibonacci-sphere incidence directions, each with two orthogonal polarizations."""
    out = []
    golden = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rho, phi = np.sqrt(1 - z * z), golden * i
        k = np.array([rho * np.cos(phi), rho * np.sin(phi), z])
        e1 = np.cross(k, [0.0, 0.0, 1.0] if abs(z) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        out += [(k, e1), (k, np.cross(k, e1))]
    return out


class _Solver:
    """One lattice on one device; solves for dipole moments per (wavelength, incidence)."""

    def __init__(self, sim: Simulation, lat: _Lattice, opts: DDAOptions, device: Device):
        self.sim, self.lat, self.opts, self.device = sim, lat, opts, device
        self.xp = device.namespace()
        self.single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu)
        self.cdt = self.xp.complex64 if self.single else self.xp.complex128
        self.tol = opts.tol or (1e-5 if self.single else 1e-8)
        self.mask_np = lat.occupied
        self.mask = device.asarray(self.mask_np.astype(float), dtype=self.cdt)
        self._ghat_cache: tuple[float, object] | None = None
        self.iterations: list[int] = []
        self.residuals: list[float] = []
        self.warnings: list[str] = []

    def k(self, lam: float) -> float:
        return 2 * np.pi * self.sim.medium_index / lam

    def _ghat(self, lam):
        if self._ghat_cache is None or self._ghat_cache[0] != lam:
            self._ghat_cache = (lam, _interaction_fft(self.lat.n, self.lat.d, self.k(lam), self.device, self.cdt))
        return self._ghat_cache[1]

    def solve(self, lam: float, khat: np.ndarray, ehat: np.ndarray):
        """Return (P on the grid (3, nx, ny, nz), 1/alpha grid, E_inc grid)."""
        xp, n, k, lat = self.xp, self.lat.n, self.k(lam), self.lat
        ghat = self._ghat(lam)
        # Partially filled sites mix full-site polarizabilities by volume fraction.
        alpha = np.zeros(n, dtype=complex)
        for s, mat in enumerate(lat.materials):
            eps_rel = complex(mat.eps(np.array([lam]))[0]) / self.sim.medium_index**2
            alpha += lat.frac[s] * _polarizability(eps_rel, lat.d, k, khat, ehat, self.opts.polarizability)
        inv_alpha_np = np.zeros(n, dtype=complex)
        inv_alpha_np[self.mask_np] = 1 / alpha[self.mask_np]
        inv_alpha = self.device.asarray(inv_alpha_np, dtype=self.cdt)
        phase = np.exp(1j * k * (lat.pts @ khat)) * self.mask_np
        e_inc = self.device.asarray(ehat[:, None, None, None] * phase[None], dtype=self.cdt)
        cdt = self.cdt

        def matvec(x):
            pad = xp.zeros((3, 2 * n[0], 2 * n[1], 2 * n[2]), dtype=cdt, device=x.device)
            pad[:, : n[0], : n[1], : n[2]] = x
            ph = xp.fft.fftn(pad, axes=(1, 2, 3))
            field = xp.fft.ifftn(xp.sum(ghat * ph[None, ...], axis=1), axes=(1, 2, 3))[:, : n[0], : n[1], : n[2]]
            return self.mask * (inv_alpha * x - field)  # (1/alpha) P - E_dip(P) on occupied sites

        x0 = e_inc / xp.where(inv_alpha == 0, xp.ones_like(inv_alpha), inv_alpha)
        pol, it, res = _bicgstab(matvec, e_inc, x0, xp, self.tol, self.opts.max_iter)
        self.iterations.append(it)
        self.residuals.append(res)
        if not np.isfinite(res):
            self.warnings.append(f"BiCGStab broke down at {lam:.1f} nm; try precision='double'")
        elif res > self.tol:
            self.warnings.append(f"BiCGStab did not converge at {lam:.1f} nm (residual {res:.2e}, {it} it)")
        return pol, inv_alpha, e_inc

    def cross_sections(self, lam, khat, ehat) -> tuple[float, float]:
        xp, k = self.xp, self.k(lam)
        pol, inv_alpha, e_inc = self.solve(lam, khat, ehat)
        p2 = xp.sum(xp.real(pol * xp.conj(pol)), axis=0)
        ext = 4 * np.pi * k * float(xp.sum(xp.imag(xp.sum(xp.conj(e_inc) * pol, axis=0))))
        absr = 4 * np.pi * k * float(xp.sum(p2 * (-xp.imag(inv_alpha) - 2 / 3 * k**3) * xp.real(self.mask)))
        return ext, absr

    def dipoles(self, lam, khat, ehat) -> tuple[np.ndarray, np.ndarray]:
        """Occupied-site positions (N, 3) and moments (N, 3) as NumPy arrays."""
        pol, _, _ = self.solve(lam, khat, ehat)
        pol = Device.to_numpy(pol)
        return self.lat.pts[self.mask_np], np.moveaxis(pol, 0, -1)[self.mask_np].astype(complex)


def _far_field(solver: _Solver, req, khat, ehat) -> list[FarFieldData]:
    out = []
    hhat = np.cross(khat, ehat)
    th = np.radians(np.asarray(req.theta_deg))[:, None]
    ph = np.radians(np.asarray(req.phi_deg))[None, :]
    nhat = (np.cos(th)[..., None] * khat + np.sin(th)[..., None]
            * (np.cos(ph)[..., None] * ehat + np.sin(ph)[..., None] * hhat))  # (T, P, 3)
    for lam in req.wavelengths_nm:
        k = solver.k(lam)
        r, p = solver.dipoles(lam, khat, ehat)
        f = np.einsum("tpn,nc->tpc", np.exp(-1j * k * nhat @ r.T), p)  # sum_j P_j e^{-ik n.r_j}
        f_perp = f - nhat * np.sum(nhat * f, axis=-1, keepdims=True)
        out.append(FarFieldData(wavelength_nm=lam, theta_deg=list(req.theta_deg), phi_deg=list(req.phi_deg),
                                dcsca_domega_nm2=(k**4 * np.sum(np.abs(f_perp) ** 2, axis=-1)).tolist()))
    return out


def _near_field(solver: _Solver, sim: Simulation, req, khat, ehat, chunk: int = 2048) -> list[NearFieldData]:
    axis_of = {"x": 0, "y": 1, "z": 2}
    a1, a2 = (axis_of[c] for c in req.plane)
    normal = 3 - a1 - a2
    u = np.linspace(-req.half_width_nm, req.half_width_nm, req.points)
    g1, g2 = np.meshgrid(u, u, indexing="ij")
    pts = np.zeros(g1.shape + (3,))
    pts[..., a1], pts[..., a2], pts[..., normal] = g1, g2, req.offset_nm
    flat = pts.reshape(-1, 3)
    inside = np.zeros(len(flat), dtype=bool)
    for p in sim.particles:
        inside |= p.shape.contains(flat - np.asarray(p.position_nm))
    out = []
    for lam in req.wavelengths_nm:
        k = solver.k(lam)
        r, p = solver.dipoles(lam, khat, ehat)
        e = ehat[None, :] * np.exp(1j * k * flat @ khat)[:, None]
        for s in range(0, len(flat), chunk):
            g = _green(flat[s:s + chunk, None, :] - r[None, :, :], k)  # (3, 3, M, N)
            e[s:s + chunk] += np.einsum("ijmn,nj->mi", g, p)
        enh = np.sum(np.abs(e) ** 2, axis=-1)
        enh[inside] = np.nan
        e[inside] = np.nan
        shape = g1.shape
        out.append(NearFieldData(
            wavelength_nm=lam, plane=req.plane, offset_nm=req.offset_nm, axis1_nm=u.tolist(), axis2_nm=u.tolist(),
            enhancement=enh.reshape(shape).tolist(),
            e_real=np.moveaxis(e.real.reshape(shape + (3,)), -1, 0).tolist(),
            e_imag=np.moveaxis(e.imag.reshape(shape + (3,)), -1, 0).tolist()))
    return out


def _extrapolate(npds: list[int], values: np.ndarray) -> np.ndarray:
    """Polynomial extrapolation in h = 1/npd to h = 0; values has shape (len(npds), ...)."""
    h = 1.0 / np.asarray(npds, dtype=float)
    deg = 2 if len(npds) >= 3 else 1
    vander = np.vander(h, deg + 1)
    coef, *_ = np.linalg.lstsq(vander, values.reshape(len(npds), -1), rcond=None)
    return coef[-1].reshape(values.shape[1:])


@register
class NativeDDA(Backend):
    name = "lightsmith-dda"
    method = "dda"
    options_model = DDAOptions
    license = "MIT"
    citation = "Draine & Flatau, JOSA A 11, 1491 (1994); Goodman, Draine & Flatau, Opt. Lett. 16, 1198 (1991)"
    gpu = True

    @classmethod
    def supports(cls, sim):
        why = unsupported_outputs(sim, near_field=True, far_field=True, orientation_average=True)
        if not why and sim.orientation_average and (sim.outputs.near_field or sim.outputs.far_field):
            why = "field outputs are not available with orientation_average"
        return (not why), why

    @classmethod
    def _options(cls, sim) -> DDAOptions:
        return DDAOptions.model_validate(sim.options_for(cls.name, cls.method))

    @classmethod
    def estimate(cls, sim, device):
        opts = cls._options(sim)
        npds = opts.extrapolate or [opts.dipoles_per_diameter or _default_npd(sim)]
        lat = _lattice(sim, max(npds), opts.discretization)
        cells = int(np.prod(lat.n))
        single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu)
        csize = 8 if single else 16
        # 9 tensor FFTs + ~6 padded work arrays on the doubled grid, ~10 Krylov vectors on the grid.
        mem = csize * (15 * 3 * 8 * cells + 10 * 3 * cells)
        n_inc = 2 * opts.orientations if sim.orientation_average else 1
        return {"dipoles": lat.n_dip, "grid": list(lat.n), "dipole_spacing_nm": lat.d, "dipoles_per_diameter": npds,
                "precision": "single" if single else "double", "memory_bytes": int(mem),
                "solves": len(npds) * n_inc * len(sim.wavelength_array()),
                "fits": lat.n_dip <= opts.max_dipoles}

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        t0 = time.perf_counter()
        opts = cls._options(sim)
        wl = sim.wavelength_array()
        khat, ehat = sim.illumination.unit_vectors()
        incidences = _orientation_set(opts.orientations) if sim.orientation_average else [(khat, ehat)]
        npds = opts.extrapolate or [opts.dipoles_per_diameter or _default_npd(sim)]

        total, done = len(npds) * len(wl) * len(incidences), 0
        runs, solvers = [], []
        for npd in npds:
            lat = _lattice(sim, npd, opts.discretization)
            if lat.n_dip > opts.max_dipoles:
                raise ValueError(f"{lat.n_dip} dipoles exceeds max_dipoles={opts.max_dipoles}; "
                                 "lower dipoles_per_diameter or raise max_dipoles")
            solver = _Solver(sim, lat, opts, device)
            rows = []
            for lam in wl:
                row = []
                for kh, eh in incidences:
                    row.append(solver.cross_sections(lam, kh, eh))
                    done += 1
                    report("dda_solve", 0.95 * done / total, wavelength_nm=float(lam), dipoles=lat.n_dip,
                           dipoles_per_diameter=npd)
                rows.append(row)
            cs = np.array(rows)
            runs.append(cs.mean(axis=1))  # (n_wl, 2): ext, abs averaged over incidences
            solvers.append(solver)

        runs = np.asarray(runs)
        c_ext, c_abs = (_extrapolate(npds, runs) if len(npds) > 1 else runs[0]).T
        finest = solvers[int(np.argmax(npds))]
        warnings = [w for s in solvers for w in s.warnings]
        if len(npds) > 1:
            warnings.append(f"cross sections extrapolated to zero dipole size from dipoles_per_diameter={npds}")

        near = far = None
        if sim.outputs.far_field:
            far = _far_field(finest, sim.outputs.far_field, khat, ehat)
        if sim.outputs.near_field:
            near = _near_field(finest, sim, sim.outputs.near_field, khat, ehat)

        provenance = {
            **device.describe(), "dipoles": [s.lat.n_dip for s in solvers], "grid": [list(s.lat.n) for s in solvers],
            "dipole_spacing_nm": [s.lat.d for s in solvers], "dipoles_per_diameter": npds,
            "precision": "single" if finest.single else "double", "polarizability": opts.polarizability,
            "tol": finest.tol, "incidences": len(incidences),
            "iterations": [s.iterations for s in solvers], "residuals": [s.residuals for s in solvers],
            "runtime_s": time.perf_counter() - t0,
        }
        if len(npds) > 1:
            provenance["unextrapolated"] = {str(n): {"c_ext_nm2": r[:, 0].tolist(), "c_abs_nm2": r[:, 1].tolist()}
                                            for n, r in zip(npds, runs)}
        res = Result.from_arrays(
            method=cls.method, backend=cls.name, wavelengths_nm=wl,
            c_ext=c_ext, c_sca=c_ext - c_abs, c_abs=c_abs, r_eq=sim.equivalent_radius_nm(),
            provenance=provenance, warnings=warnings,
        )
        res.near_field, res.far_field = near, far
        return res
