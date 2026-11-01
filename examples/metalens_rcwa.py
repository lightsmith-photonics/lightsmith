"""Metalens from an RCWA phase library, focused with angular-spectrum propagation.

1. RCWA: transmission of a periodic array of TiO2-like pillars (n = 2.4, 600 nm tall,
   350 nm square lattice) on glass, illuminated through the glass at 633 nm, for a
   sweep of pillar radii -> complex transmission t(r) (the "phase library").
2. Layout: a 72 x 72 cell lens (25.2 um wide) that imposes the hyperbolic phase
   phi(x, y) = -k (sqrt(x^2 + y^2 + f^2) - f) for f = 20 um, choosing at each cell
   the pillar whose t is closest to exp(i phi) (local periodic approximation).
3. Propagation: the transmitted field is propagated through air with the angular
   spectrum method; the script scans z, reports the focal spot and compares its
   FWHM with the diffraction limit lambda / (2 NA).

Run:  uv run python examples/metalens_rcwa.py   (about a minute on a CPU)
"""

import time

import numpy as np

import omniscatter as osc
from omniscatter.optics import angular_spectrum, focal_spot
from omniscatter.problems import Circle, Incidence, PatternedLayer, PeriodicProblem, Structure2D

LAMBDA, PERIOD, HEIGHT = 633.0, 350.0, 600.0
N_CELLS, FOCAL = 72, 20_000.0
GLASS, PILLAR, AIR = osc.ConstantIndex(n_real=1.46), osc.ConstantIndex(n_real=2.4), osc.ConstantIndex(n_real=1.0)


def pillar_array(radius_nm: float) -> PeriodicProblem:
    return PeriodicProblem(
        lattice_a1_nm=(PERIOD, 0), lattice_a2_nm=(0, PERIOD), superstrate=GLASS, substrate=AIR,
        layers=[PatternedLayer(thickness_nm=HEIGHT, background=AIR,
                               shapes=[Structure2D(profile=Circle(radius_nm=radius_nm), material=PILLAR)])],
        wavelengths=[LAMBDA], incidence=Incidence(polarizations=["s"]), options={"rcwa": {"harmonics": 4}})


def phase_library(radii: np.ndarray, device: str = "auto") -> np.ndarray:
    t = []
    for r in radii:
        res = osc.solve(pillar_array(float(r)), "rcwa", device)
        re, im = res.t_amplitude["s"][0][0]
        t.append(complex(re, im))
    return np.asarray(t)


def main(device: str = "auto") -> dict:
    t0 = time.perf_counter()
    radii = np.linspace(40, 150, 45)
    lib = phase_library(radii, device)
    phase = np.unwrap(np.angle(lib))
    print(f"phase library: {len(radii)} radii, phase span {np.degrees(phase.max() - phase.min()):.0f} deg, "
          f"|t| in [{np.abs(lib).min():.2f}, {np.abs(lib).max():.2f}]  ({time.perf_counter() - t0:.1f} s)")

    # Lens layout: best library entry for the target phase at each cell centre.
    c = (np.arange(N_CELLS) - (N_CELLS - 1) / 2) * PERIOD
    x, y = np.meshgrid(c, c, indexing="xy")
    k = 2 * np.pi / LAMBDA
    target = -k * (np.sqrt(x**2 + y**2 + FOCAL**2) - FOCAL)
    choice = np.argmax(np.real(lib[None, None, :] * np.exp(-1j * target[..., None])), axis=-1)
    cells = lib[choice]

    # Sample each cell 3 x 3, add an opaque margin, propagate through air.
    up = 3
    dx = PERIOD / up
    field = np.kron(cells, np.ones((up, up)))
    margin = field.shape[0] // 2
    field = np.pad(field, margin)
    zs = np.linspace(0.6, 1.4, 33) * FOCAL
    peaks = [float(np.max(np.abs(angular_spectrum(field, dx, LAMBDA, z)) ** 2)) for z in zs]
    z_focus = float(zs[int(np.argmax(peaks))])
    spot = focal_spot(np.abs(angular_spectrum(field, dx, LAMBDA, z_focus)) ** 2, dx)
    na = np.sin(np.arctan(N_CELLS * PERIOD / 2 / FOCAL))
    out = {"z_focus_nm": z_focus, "design_focal_nm": FOCAL, "fwhm_x_nm": spot["fwhm_x_nm"],
           "fwhm_y_nm": spot["fwhm_y_nm"], "diffraction_limit_nm": float(LAMBDA / (2 * na)),
           "spot_offset_nm": (spot["x_nm"], spot["y_nm"]), "runtime_s": time.perf_counter() - t0}
    print(f"focus at z = {z_focus / 1000:.2f} um (design {FOCAL / 1000:.1f} um); FWHM {spot['fwhm_x_nm']:.0f} x "
          f"{spot['fwhm_y_nm']:.0f} nm vs lambda/2NA = {LAMBDA / (2 * na):.0f} nm (NA {na:.2f}); "
          f"total {out['runtime_s']:.1f} s")
    return out


if __name__ == "__main__":
    main()
