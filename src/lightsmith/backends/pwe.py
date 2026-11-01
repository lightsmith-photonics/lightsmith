"""Native plane-wave-expansion (PWE) photonic band solver, CPU or GPU.

Solves Maxwell's eigenproblem for a lossless, non-dispersive periodic
permittivity: 2D crystals in TE (H_z; E in plane) and TM (E_z) polarisation and
3D crystals in the transverse plane-wave basis (H = sum_G h_G,a e_a(G), with
e_a perpendicular to k+G, so div H = 0 exactly and there are no spurious modes).

Fourier treatment of eps (documented choice; both are implemented):

- ``solver="dense"``: the inverse-matrix rule of Ho, Chan & Soukoulis, PRL 65,
  3152 (1990): eta = [eps_hat(G - G')]^-1, one matrix inverse per structure, then
  one batched ``eigvalsh`` per chunk of k-points. For TM this is exactly the
  generalised problem |k+G|^2 e = (omega/c)^2 [eps] e and converges to ~1e-5 with
  ~300 plane waves. eps_hat is analytic for non-overlapping circles, ellipses,
  rectangles, polygons, balls, boxes and vertical prisms (Bessel/sinc/polygon-edge
  formulas) and comes from a supersampled raster otherwise (``eps_fourier``).
- ``solver="lobpcg"``: matrix-free FFT operators on a real-space grid with the
  anisotropic subpixel-smoothed inverse-permittivity tensor
  <1/eps> nn + <eps>^-1 (1 - nn) of Farjadpour et al., Opt. Lett. 31, 2972 (2006),
  as in MPB (Johnson & Joannopoulos, Opt. Express 8, 173 (2001)), a block LOBPCG
  eigensolver with an approximate-inverse preconditioner, warm-started along the
  k-path. Converges ~quadratically in resolution, also for TE.

``solver="auto"`` uses dense for TM (and empty cells) and lobpcg for TE and 3D,
because HCS treats every E component as tangential and converges slowly for TE
at curved interfaces (~1 % error at 1200 plane waves vs ~0.3 % for lobpcg at 32/a).

Units: frequencies are a/lambda_0 = omega a / (2 pi c) with a = |lattice_vectors_nm[0]|;
``k_distance`` is in units of 2 pi / a. Materials are evaluated once at
``reference_wavelength_nm`` (real part only).
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device, resolve_device
from ..problems import (Ball, BandProblem, Box, Circle, Ellipse, KPoint, Polygon, Prism, Rectangle, Structure,
                        Structure2D, options_for)
from ..results import BandResult
from ._pwe.engine import DEFAULT_PW_DENSITY, compute_bands, estimate_memory, make_plan
from ._pwe.geometry import Cell, Inclusion, Lattice
from ._pwe.solvers import dense_bands, eigen_to_freq, plane_wave_set, toeplitz_eps
from .base import Backend, register


class PWEOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solver: Literal["auto", "dense", "lobpcg"] = Field(
        "auto", description="dense: Ho-Chan-Soukoulis matrices + batched eigh. lobpcg: matrix-free FFT operators "
        "with subpixel-smoothed eps. auto: dense for TM (matrix <= 2500), lobpcg for TE and 3D; setting only "
        "num_plane_waves forces dense, setting only resolution forces lobpcg.")
    num_plane_waves: int | None = Field(
        None, ge=1, description=f"Dense basis size (completed to a full |G| shell). Default "
        f"{DEFAULT_PW_DENSITY[2]} (2D) / {DEFAULT_PW_DENSITY[3]} (3D) per unit-cell area/volume in a^d.")
    resolution: int | None = Field(
        None, ge=4, description="lobpcg grid points per lattice constant a (default 32 in 2D, 16 in 3D).")
    eps_fourier: Literal["auto", "analytic", "raster"] = Field(
        "auto", description="Dense eps_hat: analytic shape transforms (needs non-overlapping inclusions) or a "
        "supersampled raster. auto = analytic when valid.")
    subsamples: int | None = Field(None, ge=1, description="Raster supersampling per axis (default 4 in 2D, 2 in 3D).")
    reference_wavelength_nm: float | None = Field(
        None, gt=0, description="Vacuum wavelength at which materials are evaluated. Needed for dispersive "
        "materials (default 2a, i.e. a/lambda = 0.5, with a warning).")
    precision: Literal["auto", "single", "double"] = Field(
        "auto", description="auto = double, except single for the dense solver on a GPU (~1e-5 in a/lambda).")
    tol: float | None = Field(None, gt=0, description="LOBPCG relative residual (default 1e-6 double, 1e-4 single).")
    max_iter: int = Field(400, ge=1, description="LOBPCG iterations per k-point.")
    extra_bands: int | None = Field(None, ge=0, description="LOBPCG block padding (default max(2, num_bands/4)).")


# --- problem -> engine ----------------------------------------------------------------


def _lattice(problem: BandProblem) -> Lattice:
    lv = np.asarray(problem.lattice_vectors_nm, float)
    a_nm = float(np.linalg.norm(lv[0]))
    d = problem.dimension
    return Lattice(lv[:d, :d] / a_nm, a_nm)


def _eps_at(mat, wl: float, what: str, warnings: list[str]) -> float:
    e = complex(mat.eps(np.array([wl]))[0])
    try:
        nearby = mat.eps(np.array([0.97 * wl, 1.03 * wl]))
        if np.max(np.abs(nearby - e)) > 1e-9 * abs(e):
            warnings.append(f"{what} is dispersive; PWE uses eps={e.real:.4g} from {wl:g} nm for all frequencies")
    except ValueError:  # data do not extend +-3 %: cannot tell, assume tabulated (dispersive)
        warnings.append(f"{what} is tabulated; PWE uses eps={e.real:.4g} from {wl:g} nm for all frequencies")
    if e.imag > 1e-12 * abs(e):
        warnings.append(f"{what} is lossy (Im eps={e.imag:.3g}); PWE ignores the loss")
    if e.real <= 0:
        raise ValueError(f"{what} has Re(eps)={e.real:.4g} <= 0 at {wl:g} nm; plane-wave expansion needs "
                         "positive permittivity (metallic crystals need a dispersive solver)")
    return float(e.real)


def _is_dispersive(mat) -> bool:
    return getattr(mat, "kind", "") not in ("constant_index", "constant_permittivity")


def _cell(problem: BandProblem, opts: PWEOptions, warnings: list[str]) -> tuple[Cell, float]:
    lat = _lattice(problem)
    structs = problem.structures_2d if problem.dimension == 2 else problem.structures
    mats = [problem.background] + [s.material for s in structs]
    wl = opts.reference_wavelength_nm
    if wl is None:
        wl = 2 * lat.a_nm
        if any(_is_dispersive(m) for m in mats):
            warnings.append(f"reference_wavelength_nm not set; dispersive materials evaluated at {wl:g} nm "
                            "(a/lambda = 0.5)")
    eps_bg = _eps_at(problem.background, wl, "background", warnings)
    incs = []
    for i, s in enumerate(structs):
        shape = s.profile if problem.dimension == 2 else s.geometry
        incs.append(Inclusion(shape, _eps_at(s.material, wl, s.name or f"structure {i}", warnings)))
    return Cell(lat, eps_bg, incs), wl


def k_path(problem: BandProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """Interpolated path: reduced k (n, 3), Cartesian k (n, d) in 1/a, distance in 2 pi/a, labels."""
    lat = _lattice(problem)
    d = lat.dim
    pts = np.asarray([kp.reduced for kp in problem.k_path], float)
    n = problem.points_per_segment
    red = [pts[0]]
    labels = {0: problem.k_path[0].label}
    for s in range(len(pts) - 1):
        t = (np.arange(1, n + 1) / n)[:, None]
        red.extend(pts[s] + t * (pts[s + 1] - pts[s]))
        labels[(s + 1) * n] = problem.k_path[s + 1].label
    red = np.asarray(red)
    cart = red[:, :d] @ lat.reciprocal
    step = np.linalg.norm(np.diff(cart, axis=0), axis=1) / (2 * np.pi)
    return red, cart, np.concatenate([[0.0], np.cumsum(step)]), labels


def _pols(problem: BandProblem) -> dict[str, str]:
    """Result key -> engine polarization."""
    if problem.dimension == 3:
        return {"all": "3d"}
    return {"te": "te", "tm": "tm"} if problem.polarization == "both" else {problem.polarization: problem.polarization}


@register
class NativePWE(Backend):
    name = "lightsmith-pwe"
    method = "pwe"
    license = "MIT"
    citation = ("Ho, Chan & Soukoulis, PRL 65, 3152 (1990); Johnson & Joannopoulos, Opt. Express 8, 173 (2001); "
                "Farjadpour et al., Opt. Lett. 31, 2972 (2006)")
    gpu = True
    differentiable = False  # see bands_from_permittivity for a differentiable entry point
    problem_kinds = ("bands",)

    @classmethod
    def supports(cls, problem):
        if not isinstance(problem, BandProblem):
            return False, "PWE solves 'bands' problems only"
        d = problem.dimension
        lv = np.asarray(problem.lattice_vectors_nm, float)
        if len(lv) < d:
            return False, f"a {d}D crystal needs {d} lattice vectors"
        if d == 2:
            if problem.structures:
                return False, "2D crystals take structures_2d, not 3D structures"
            if np.any(lv[:2, 2] != 0):
                return False, "2D lattice vectors must lie in the xy plane (z component 0)"
            if any(kp.reduced[2] != 0 for kp in problem.k_path):
                return False, "out-of-plane k (third reduced coordinate) is not supported for 2D crystals"
        else:
            if problem.structures_2d:
                return False, "3D crystals take structures (3D geometry), not structures_2d"
        if abs(np.linalg.det(lv[:d, :d])) < 1e-9 * np.linalg.norm(lv[0]) ** d:
            return False, "lattice vectors are linearly dependent"
        try:
            PWEOptions.model_validate(options_for(problem, cls.name, cls.method))
        except Exception as e:  # pydantic ValidationError
            return False, f"invalid pwe options: {e}"
        return True, ""

    @classmethod
    def _options(cls, problem) -> PWEOptions:
        return PWEOptions.model_validate(options_for(problem, cls.name, cls.method))

    @classmethod
    def estimate(cls, problem, device):
        opts = cls._options(problem)
        warnings: list[str] = []
        cell, _ = _cell(problem, opts, warnings)
        pols = _pols(problem)
        n_k = len(k_path(problem)[0])
        plans = {key: make_plan(cell, p, opts, device) for key, p in pols.items()}
        return {"solver": {k: p.solver for k, p in plans.items()},
                "num_plane_waves": {k: p.num_plane_waves for k, p in plans.items()},
                "matrix_size": {k: p.matrix_size for k, p in plans.items()},
                "grid": {k: list(p.grid) for k, p in plans.items() if p.grid},
                "precision": next(iter(plans.values())).precision, "k_points": n_k, "polarizations": list(pols),
                "memory_bytes": max(estimate_memory(p, n_k, problem.num_bands) for p in plans.values()),
                "eigensolves": n_k * len(pols)}

    @classmethod
    def solve(cls, problem: BandProblem, device: Device) -> BandResult:
        t0 = time.perf_counter()
        opts = cls._options(problem)
        warnings: list[str] = []
        cell, wl = _cell(problem, opts, warnings)
        red, cart, dist, labels = k_path(problem)
        pols = _pols(problem)
        out = compute_bands(cell, cart, list(pols.values()), problem.num_bands, opts, device)
        warnings += out.warnings
        provenance = {**device.describe(), **out.info, "reference_wavelength_nm": wl,
                      "eps": cell.eps_values, "a_nm": cell.lattice.a_nm, "k_points": len(red),
                      "units": {"frequency": "a/lambda", "k_distance": "2 pi / a"},
                      "runtime_s": time.perf_counter() - t0}
        return BandResult(method=cls.method, backend=cls.name, provenance=provenance, warnings=warnings,
                          k_reduced=[tuple(map(float, k)) for k in red], k_distance=dist.tolist(), k_labels=labels,
                          frequencies={key: out.freqs[p].tolist() for key, p in pols.items()})


# --- post-processing ------------------------------------------------------------------


def _gaps_from_ranges(lo: np.ndarray, hi: np.ndarray) -> list[tuple[float, float, int]]:
    """Frequency intervals free of states below the top band's minimum: (bottom, top, lower_band)."""
    gaps = []
    ceiling = float(lo[-1])  # above this, uncomputed bands may exist
    order = np.argsort(lo)
    reach = -np.inf
    for j, b in enumerate(order):
        if j > 0 and lo[b] > reach and reach < ceiling:
            gaps.append((float(reach), float(min(lo[b], ceiling)), j - 1))
        reach = max(reach, hi[b])
    return [g for g in gaps if g[1] > g[0]]


def _gap_dict(bottom: float, top: float, lower_band: int | None = None) -> dict[str, Any]:
    mid = (bottom + top) / 2
    d = {"bottom": bottom, "top": top, "width": top - bottom, "midgap": mid, "gap_midgap_ratio": (top - bottom) / mid}
    if lower_band is not None:
        d["between_bands"] = (lower_band + 1, lower_band + 2)  # 1-based band numbers
    return d


def band_gaps(result: BandResult, min_ratio: float = 1e-3) -> dict[str, list[dict[str, Any]]]:
    """Gaps along the computed k-path for each polarization, plus 'complete' gaps.

    A gap here is a frequency range (a/lambda) where no computed band has a state
    at any k on the path; for 2D 'both' results, 'complete' lists the ranges that
    are gaps for TE and TM simultaneously. Gaps are reported only below the
    minimum of the highest computed band. They are true band gaps when the path
    traces the edge of the irreducible Brillouin zone (the usual high-symmetry
    path), which holds for the common lattices but is not guaranteed in general.
    Gaps with gap/midgap below `min_ratio` (default 0.1 %) are dropped: they are
    usually bands touching at a degeneracy, split only by round-off.
    Each entry: bottom, top, width, midgap, gap_midgap_ratio (and between_bands,
    1-based, for single-polarization gaps).
    """
    out: dict[str, list[dict[str, Any]]] = {}
    intervals: dict[str, list[tuple[float, float]]] = {}
    for pol, f in result.frequencies.items():
        f = np.asarray(f)
        gaps = _gaps_from_ranges(f.min(axis=0), f.max(axis=0))
        intervals[pol] = [(b, t) for b, t, _ in gaps]
        out[pol] = [_gap_dict(b, t, j) for b, t, j in gaps if (t - b) / ((t + b) / 2) > min_ratio]
    complete = None
    for pol, ivs in intervals.items():
        if complete is None:
            complete = ivs
            continue
        complete = [(max(a0, b0), min(a1, b1)) for a0, a1 in complete for b0, b1 in ivs if min(a1, b1) > max(a0, b0)]
    out["complete"] = [_gap_dict(b, t) for b, t in (complete or []) if (t - b) / ((t + b) / 2) > min_ratio]
    return out


def density_of_states(problem: BandProblem | dict | str, mesh: Sequence[int] | int = 16, bins: int = 200,
                      broadening: float | None = None, device: str = "auto") -> dict[str, Any]:
    """Photonic density of states by k-mesh sampling (Monkhorst-Pack, full Brillouin zone).

    Returns frequencies (bin centres, a/lambda) and, per polarization, the DOS in
    states per unit a/lambda per unit cell (integrates to num_bands). With
    `broadening` (a/lambda), each state is a Gaussian instead of a histogram count.
    Only frequencies below the minimum of the highest band are complete
    (`valid_below`).
    """
    from ..problems import parse_problem

    problem = parse_problem(problem)
    ok, why = NativePWE.supports(problem)
    if not ok:
        raise ValueError(why)
    d = problem.dimension
    mesh = (mesh,) * d if isinstance(mesh, int) else tuple(mesh)
    red = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) / n - 0.5 for n in mesh], indexing="ij"),
                   axis=-1).reshape(-1, d)
    opts = NativePWE._options(problem)
    warnings: list[str] = []
    cell, _ = _cell(problem, opts, warnings)
    dev = resolve_device(device)
    pols = _pols(problem)
    out = compute_bands(cell, red @ cell.lattice.reciprocal, list(pols.values()), problem.num_bands, opts, dev)
    fmax = max(float(f.max()) for f in out.freqs.values())
    edges = np.linspace(0, 1.02 * fmax, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    df = edges[1] - edges[0]
    dos, valid = {}, {}
    for key, p in pols.items():
        f = out.freqs[p]
        if broadening:
            g = np.exp(-0.5 * ((centres[:, None] - f.ravel()[None]) / broadening) ** 2)
            dos[key] = (g.sum(axis=1) / (broadening * math.sqrt(2 * math.pi)) / len(red)).tolist()
        else:
            dos[key] = (np.histogram(f.ravel(), edges)[0] / (df * len(red))).tolist()
        valid[key] = float(f[:, -1].min())
    return {"frequencies": centres.tolist(), "dos": dos, "valid_below": valid, "mesh": list(mesh),
            "warnings": warnings + out.warnings, "provenance": {**dev.describe(), **out.info}}


# --- supercells for defects -----------------------------------------------------------


def _translate(shape, t: np.ndarray):
    if isinstance(shape, (Circle, Ellipse, Rectangle)):
        return shape.model_copy(update={"center_nm": tuple(np.asarray(shape.center_nm) + t[:2])})
    if isinstance(shape, Polygon):
        return shape.model_copy(update={"vertices_nm": [tuple(np.asarray(v) + t[:2]) for v in shape.vertices_nm]})
    if isinstance(shape, (Ball, Box)):
        return shape.model_copy(update={"center_nm": tuple(np.asarray(shape.center_nm) + t[:3])})
    if isinstance(shape, Prism):
        return shape.model_copy(update={"profile": _translate(shape.profile, t), "z_min_nm": shape.z_min_nm + t[2],
                                        "z_max_nm": shape.z_max_nm + t[2]})
    raise TypeError(f"cannot translate {type(shape).__name__}")


def supercell(problem: BandProblem, repeats: Sequence[int], remove: Sequence[Sequence[int]] = (),
              add: Sequence[Structure2D | Structure] = (), k_path: list[KPoint] | None = None) -> BandProblem:
    """Tile the unit cell `repeats` times for line (waveguide) or point (cavity) defects.

    Cells are labelled by integer offsets c_i in range(-(n_i // 2), n_i - n_i // 2), so
    (0, 0) is the central cell. `remove` lists the cells whose inclusions are dropped
    (e.g. one row for a W1 waveguide, one cell for an H0/H1-type cavity); `add` appends
    extra structures (nm, supercell coordinates). The k-path keeps its reduced
    coordinates unless `k_path` is given. Frequencies stay normalised by the first
    supercell vector, so keep repeats[0] = 1 to retain the primitive a.
    """
    d = problem.dimension
    reps = tuple(int(r) for r in repeats)
    if len(reps) != d or min(reps) < 1:
        raise ValueError(f"repeats needs {d} positive integers")
    lv = np.asarray(problem.lattice_vectors_nm, float)
    drop = {tuple(int(v) for v in c) for c in remove}
    structs = problem.structures_2d if d == 2 else problem.structures
    tiled = []
    for c in itertools.product(*[range(-(n // 2), n - n // 2) for n in reps]):
        if c in drop:
            continue
        t = np.asarray(c, float) @ lv[:d]
        for s in structs:
            if d == 2:
                tiled.append(s.model_copy(update={"profile": _translate(s.profile, t)}))
            else:
                tiled.append(s.model_copy(update={"geometry": _translate(s.geometry, t)}))
    tiled.extend(add)
    new_lv = [tuple(float(v) for v in lv[i] * (reps[i] if i < d else 1)) for i in range(len(lv))]
    upd = {"lattice_vectors_nm": new_lv, ("structures_2d" if d == 2 else "structures"): tiled}
    if k_path is not None:
        upd["k_path"] = k_path
    return problem.model_copy(update=upd)


# --- differentiable entry point -----------------------------------------------------


def bands_from_permittivity(eps, lattice_vectors_nm: Sequence[Sequence[float]], k_reduced: Sequence[Sequence[float]],
                            num_bands: int = 8, polarization: Literal["te", "tm", "3d"] = "tm",
                            num_plane_waves: int = 300, device: Device | None = None):
    """Band frequencies (a/lambda) from a real-space permittivity grid, differentiable in `eps`.

    `eps` is a 2D/3D grid of real permittivity samples on the unit cell (pixel j at
    fractional coordinate j/N), as a NumPy array or a torch tensor; with a torch tensor
    that requires grad, gradients flow through the FFT, the Ho-Chan-Soukoulis inverse
    and ``eigh`` (well defined for non-degenerate bands). Returns an array in the
    framework of `eps`, shape (len(k_reduced), num_bands).
    """
    is_torch = hasattr(eps, "detach")
    if device is None:
        device = Device("torch", str(eps.device)) if is_torch else Device("numpy", "cpu")
    xp = device.namespace()
    lv = np.asarray(lattice_vectors_nm, float)
    d = eps.ndim
    lat = Lattice(lv[:d, :d] / np.linalg.norm(lv[0]), float(np.linalg.norm(lv[0])))
    idx = plane_wave_set(lat.reciprocal, num_plane_waves)
    half = tuple(2 * int(np.abs(idx[:, i]).max()) for i in range(d))
    grid = tuple(eps.shape)
    if any(2 * h + 1 > n for h, n in zip(half, grid)):
        raise ValueError(f"eps grid {grid} too coarse for {num_plane_waves} plane waves; need >= "
                         f"{tuple(2 * h + 1 for h in half)}")
    e = eps if is_torch else device.asarray(eps)
    coef = xp.fft.fftn(xp.astype(e, xp.complex128)) / math.prod(grid)
    axes = [np.arange(-h, h + 1) % n for h, n in zip(half, grid)]
    flat = np.ravel_multi_index(np.meshgrid(*axes, indexing="ij"), grid).reshape(-1)
    dev = getattr(coef, "device", None)
    ind = xp.asarray(flat, device=dev) if is_torch else xp.asarray(flat)
    table = xp.take(xp.reshape(coef, (-1,)), ind)
    emat = toeplitz_eps(table, half, idx, xp)
    eta = xp.linalg.inv(emat)
    eta = (eta + xp.conj(xp.permute_dims(eta, (1, 0)))) / 2
    k_cart = np.asarray(k_reduced, float)[:, :d] @ lat.reciprocal
    lam = dense_bands(eta, idx, lat.reciprocal, k_cart, polarization, num_bands, xp, device, xp.complex128)
    return eigen_to_freq(lam, xp)
