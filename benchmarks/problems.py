"""Generate the lightsmith cross-method benchmark problem suites as JSON specs.

Each problem is a plain dict:

    {
      "id": "sphere-au-jc-r20-water",
      "family": "metal_sphere",
      "params": {...},                     # physical parameters, for tables and regime maps
      "map_axes": ["radius_nm", "medium"],  # which params a regime map should use as x / y
      "simulation": {...},                 # an lightsmith Simulation, schema 0.1, without solver options
      "sweeps": {                          # resolution ladders per method (semantic quantity + values)
        "dda": {"quantity": "dipoles_per_diameter", "values": [...]},
        "fdtd": {"quantity": "mesh_nm", "values": [...]},
        "tmatrix": {"quantity": "multipole_order", "values": [...]}
      },
      "variants": {"dda": {"center": null, "weighted": null, "extrapolate": [16, 24]}},
      "notes": "..."
    }

Sweep values are *semantic*; benchmarks/run.py maps them onto whatever option the
installed backend exposes (e.g. `mesh_nm` -> `resolution_nm`, or pixels/um). A
variant value of null means "every value in the sweep"; a list restricts it.

Usage:
    uv run --no-sync python benchmarks/problems.py --suite smoke     # writes benchmarks/suites/smoke.json
    uv run --no-sync python benchmarks/problems.py --suite full --split   # also one file per problem
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE_DIR = HERE / "suites"

WATER = 1.33
AU = {"kind": "library", "name": "Au", "dataset": "Johnson-Christy"}
AG = {"kind": "library", "name": "Ag", "dataset": "Johnson-Christy"}
SIO2 = {"kind": "library", "name": "SiO2", "dataset": "Malitson"}
# The Drude sphere behind the known DDA staircase finding (r=40 nm, water, 700 nm).
DRUDE = {"kind": "drude", "eps_inf": 9.5, "wp_ev": 9.0, "gamma_ev": 0.07}

DEFAULT_TMATRIX = {"quantity": "multipole_order", "values": [4, 8, 12, 16]}


def _wl(start, stop, num):
    return {"start_nm": float(start), "stop_nm": float(stop), "num": int(num)}


def _lambda_min(wavelengths) -> float:
    return wavelengths["start_nm"] if isinstance(wavelengths, dict) else min(wavelengths)


def _sim(particles, wavelengths, medium=1.0, direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0)):
    return {
        "schema_version": "0.1",
        "particles": particles,
        "medium_index": float(medium),
        "illumination": {"kind": "plane_wave", "direction": list(direction), "polarization": list(polarization)},
        "wavelengths": wavelengths,
    }


def _sphere(r, material, pos=(0.0, 0.0, 0.0)):
    return {"shape": {"kind": "sphere", "radius_nm": float(r)}, "material": material, "position_nm": list(pos)}


def _medium_name(n):
    return {1.0: "vacuum", WATER: "water"}.get(n, f"n{n:g}")


def _fdtd_sweep(feature_nm, cells=(5, 10, 20, 40), wavelength_nm=None, per_wavelength=(6, 10, 15, 20)):
    """Mesh ladder: `cells` Yee cells across the smallest geometric feature and, when `wavelength_nm`
    (the shortest wavelength inside the densest medium) is given, at least `per_wavelength` cells per
    wavelength. Below about 6 cells per wavelength no FDTD code resolves the field."""
    mesh = [feature_nm / k for k in cells]
    if wavelength_nm is not None:
        mesh = [min(h, wavelength_nm / m) for h, m in zip(mesh, per_wavelength)]
    return {"quantity": "mesh_nm", "values": [round(h, 3) for h in mesh]}


def _coarsen_fdtd(problem, cells=(5, 10)):
    """Smoke suite: shorter FDTD ladder, re-derived from the full one's feature size."""
    feature = problem["sweeps"]["fdtd"]["values"][0] * 5
    problem["sweeps"]["fdtd"] = _fdtd_sweep(feature, cells)
    return problem


# --- families ---------------------------------------------------------------


def dielectric_sphere(x, m, k=0.0, medium=1.0, wavelengths=None, npd=None):
    """Sphere with constant index n = m * medium (+ i k), radius chosen so the size
    parameter at 600 nm is x = 2 pi n_med r / lambda."""
    lam0 = 600.0
    r = x * lam0 / (2 * math.pi * medium)
    wavelengths = wavelengths or _wl(500, 700, 5)
    # |m| k d <= 0.5 rule of thumb gives npd ~ 4 x |m|; ladder brackets it.
    base = max(8, math.ceil(4 * x * m))
    npd = npd or sorted({max(6, base // 2), base, int(base * 1.5), 2 * base})
    return {
        "id": f"sphere-diel-x{x:g}-m{m:g}" + (f"-k{k:g}" if k else "") + f"-{_medium_name(medium)}",
        "family": "dielectric_sphere",
        "params": {"size_parameter": x, "relative_index": m, "k": k, "radius_nm": round(r, 3),
                   "medium": _medium_name(medium), "medium_index": medium},
        "map_axes": ["size_parameter", "relative_index"],
        "simulation": _sim([_sphere(r, {"kind": "constant_index", "n_real": m * medium, "n_imag": k})],
                           wavelengths, medium),
        "sweeps": {"dda": {"quantity": "dipoles_per_diameter", "values": npd},
                   "fdtd": _fdtd_sweep(2 * r, wavelength_nm=_lambda_min(wavelengths) / (m * medium)),
                   "tmatrix": DEFAULT_TMATRIX},
        "variants": {"dda": {"center": None, "weighted": None}},
    }


def metal_sphere(metal, r, medium, wavelengths, npd, extrapolate=None, material=None, tag=None):
    mat = material or {"Au": AU, "Ag": AG}[metal]
    variants = {"center": None, "weighted": None}
    if extrapolate:
        variants["extrapolate"] = extrapolate
    name = tag or f"{metal.lower()}-{'jc' if mat.get('kind') == 'library' else mat['kind']}"
    return {
        "id": f"sphere-{name}-r{r:g}-{_medium_name(medium)}",
        "family": "metal_sphere",
        "params": {"metal": metal, "material": mat, "radius_nm": r, "medium": _medium_name(medium),
                   "medium_index": medium},
        "map_axes": ["radius_nm", "medium"],
        "simulation": _sim([_sphere(r, mat)], wavelengths, medium),
        "sweeps": {"dda": {"quantity": "dipoles_per_diameter", "values": npd},
                   "fdtd": _fdtd_sweep(2 * r), "tmatrix": DEFAULT_TMATRIX},
        "variants": {"dda": variants},
    }


def core_shell(core_r, shell_t, core_mat, shell_mat, label, medium, wavelengths, npd):
    return {
        "id": f"coreshell-{label}-c{core_r:g}-s{shell_t:g}-{_medium_name(medium)}",
        "family": "core_shell",
        "params": {"core_radius_nm": core_r, "shell_nm": shell_t, "label": label, "medium": _medium_name(medium),
                   "medium_index": medium},
        "map_axes": ["core_radius_nm", "shell_nm"],
        "simulation": _sim([{"shape": {"kind": "layered_sphere", "core_radius_nm": float(core_r),
                                       "shells": [{"thickness_nm": float(shell_t), "material": shell_mat}]},
                             "material": core_mat}], wavelengths, medium),
        "sweeps": {"dda": {"quantity": "dipoles_per_diameter", "values": npd},
                   "fdtd": _fdtd_sweep(shell_t), "tmatrix": DEFAULT_TMATRIX},
        "variants": {"dda": {"center": None, "weighted": None}},
        "notes": "Mie reference only if a Mie backend supports layered spheres; otherwise highest-resolution reference.",
    }


def spheroid(aspect, r_eq, material, medium, wavelengths, npd, metal="Au"):
    """Prolate spheroid (aspect = c/a >= 1) with volume-equivalent radius r_eq, axis along z,
    illuminated along x with E along the long axis."""
    a = r_eq / aspect ** (1 / 3)
    c = a * aspect
    return {
        "id": f"spheroid-{metal.lower()}-ar{aspect:g}-req{r_eq:g}-{_medium_name(medium)}",
        "family": "spheroid",
        "params": {"aspect_ratio": aspect, "r_eq_nm": r_eq, "semi_axis_xy_nm": round(a, 4),
                   "semi_axis_z_nm": round(c, 4), "metal": metal, "medium": _medium_name(medium),
                   "medium_index": medium},
        "map_axes": ["aspect_ratio", "r_eq_nm"],
        "simulation": _sim([{"shape": {"kind": "spheroid", "semi_axis_xy_nm": a, "semi_axis_z_nm": c},
                             "material": material}], wavelengths, medium,
                           direction=(1.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0)),
        "sweeps": {"dda": {"quantity": "dipoles_per_diameter", "values": npd},
                   "fdtd": _fdtd_sweep(2 * a), "tmatrix": DEFAULT_TMATRIX},
        "variants": {"dda": {"center": None, "weighted": None}},
    }


def rod(aspect, radius, material, medium, wavelengths, npd, metal="Au"):
    """Flat-ended cylinder, length = 2 * radius * aspect, axis along z, E along the axis."""
    length = 2 * radius * aspect
    return {
        "id": f"rod-{metal.lower()}-ar{aspect:g}-r{radius:g}-{_medium_name(medium)}",
        "family": "rod",
        "params": {"aspect_ratio": aspect, "radius_nm": radius, "length_nm": length, "metal": metal,
                   "medium": _medium_name(medium), "medium_index": medium},
        "map_axes": ["aspect_ratio", "radius_nm"],
        "simulation": _sim([{"shape": {"kind": "cylinder", "radius_nm": float(radius), "length_nm": float(length)},
                             "material": material}], wavelengths, medium,
                           direction=(1.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0)),
        "sweeps": {"dda": {"quantity": "dipoles_per_diameter", "values": npd},
                   "fdtd": _fdtd_sweep(2 * radius), "tmatrix": DEFAULT_TMATRIX},
        "variants": {"dda": {"center": None, "weighted": None}},
    }


def dimer(gap, r, material, medium, wavelengths, npd, metal="Au"):
    """Two identical spheres on the x axis, E along the dimer axis (x), k along z."""
    off = r + gap / 2
    return {
        "id": f"dimer-{metal.lower()}-r{r:g}-gap{gap:g}-{_medium_name(medium)}",
        "family": "dimer",
        "params": {"gap_nm": gap, "radius_nm": r, "metal": metal, "medium": _medium_name(medium),
                   "medium_index": medium, "gap_over_diameter": gap / (2 * r)},
        "map_axes": ["gap_nm", "radius_nm"],
        "simulation": _sim([_sphere(r, material, (-off, 0.0, 0.0)), _sphere(r, material, (off, 0.0, 0.0))],
                           wavelengths, medium),
        "sweeps": {"dda": {"quantity": "dipoles_per_diameter", "values": npd},
                   "fdtd": _fdtd_sweep(min(gap, 2 * r)), "tmatrix": DEFAULT_TMATRIX},
        "variants": {"dda": {"center": None, "weighted": None}},
        "notes": "DDA dipoles_per_diameter refers to one sphere's diameter; gaps need d < gap.",
    }


# --- suites -----------------------------------------------------------------


def smoke() -> dict:
    """About five minutes on one desktop CPU + one consumer GPU. One problem per family.

    Only the dielectric sphere and the Drude staircase study run the experimental DDA variants;
    everything else uses the default (center) discretisation. `smoke-extended` keeps every variant.
    """
    suite = smoke_extended()
    p = suite["problems"]
    p[1]["sweeps"]["dda"]["values"] = list(range(8, 26, 2)) + [28, 32]
    p[1]["variants"]["dda"] = {"center": None, "weighted": [8, 12, 16, 24, 32], "extrapolate": [16, 24, 32]}
    p[4]["sweeps"]["dda"]["values"] = [12, 16, 24]
    for q in p[2:]:
        q["variants"]["dda"] = {"center": None}
    for i, mesh in {0: [21.54, 10.77], 1: [8.0, 4.0, 2.0], 2: [8.0], 3: [12.0], 4: [4.0], 5: [6.35],
                    6: [4.0, 3.2], 7: [4.0]}.items():
        p[i]["sweeps"]["fdtd"] = {"quantity": "mesh_nm", "values": mesh}
    suite.update(
        suite="smoke",
        description="Smoke suite: one problem per family, 1-3 wavelengths, short ladders, DDA max_iter 3000.",
        budgets={"cpu": {"max_dipoles": 600, "max_cell_steps": 1.0e9},
                 "gpu": {"max_dipoles": 60000, "max_cell_steps": 1.2e11}},
        options={"dda": {"max_iter": 3000}},
    )
    return suite


def smoke_extended() -> dict:
    """The first smoke definition: every DDA variant on every problem (about 45 min on a shared GPU)."""
    wl3 = [450.0, 550.0, 700.0]
    au3 = [500.0, 530.0, 650.0]
    problems = [
        dielectric_sphere(1.5, 1.5, k=0.01, medium=WATER, wavelengths=wl3, npd=[8, 12, 16, 20]),
        {
            **metal_sphere("Drude", 40, WATER, [700.0], list(range(8, 26, 2)) + [28, 32],
                           extrapolate=[16, 24, 32], material=DRUDE, tag="drude"),
            "notes": "Known DDA staircase case: Mie Q_abs = 0.107 at 700 nm.",
        },
        metal_sphere("Au", 20, WATER, au3, [8, 12, 16, 24]),
        metal_sphere("Ag", 30, 1.0, [360.0, 400.0, 500.0], [8, 12, 16, 24]),
        core_shell(30, 8, SIO2, AU, "sio2-au", WATER, [600.0, 700.0, 800.0], [8, 12, 16, 24, 32]),
        spheroid(2.0, 20, AU, WATER, [520.0, 600.0, 700.0], [8, 12, 16, 24]),
        rod(3.0, 8, AU, WATER, [520.0, 650.0, 800.0], [8, 12, 16, 20]),
        dimer(4.0, 15, AU, WATER, [520.0, 560.0, 650.0], [8, 12, 16, 24]),
    ]
    for p in problems:
        _coarsen_fdtd(p)
    # Hand-tuned FDTD ladders (nm) so each run stays under ~30 s on one consumer GPU.
    for i, mesh in {0: [21.54, 10.77], 1: [8.0, 4.0, 2.0], 4: [4.0, 2.0], 6: [4.0, 3.2, 2.0], 7: [4.0, 2.0]}.items():
        problems[i]["sweeps"]["fdtd"] = {"quantity": "mesh_nm", "values": mesh}
    return {
        "suite": "smoke-extended",
        "description": "Smoke suite with every DDA variant on every problem; 1-3 wavelengths.",
        # Resolution budgets (per device); skipped runs are recorded.
        "budgets": {"cpu": {"max_dipoles": 1200, "max_cell_steps": 1.5e9},
                    "gpu": {"max_dipoles": 60000, "max_cell_steps": 1.2e11}},
        "problems": problems,
    }


def full() -> dict:
    """The paper suite. Expect hours on a GPU; see benchmarks/README.md."""
    wl = _wl(400, 800, 41)
    problems = []
    for medium in (1.0, WATER):
        for x in (0.5, 1.0, 2.0, 5.0, 10.0):
            for m in (1.2, 1.5, 2.0, 3.5):
                problems.append(dielectric_sphere(x, m, k=0.0, medium=medium))
                problems.append(dielectric_sphere(x, m, k=0.05, medium=medium))
    npd_metal = [8, 12, 16, 20, 24, 32, 40, 48, 64]
    for metal in ("Au", "Ag"):
        for medium in (1.0, WATER):
            for r in (10, 20, 40, 60, 80, 100):
                problems.append(metal_sphere(metal, r, medium, wl, npd_metal, extrapolate=[16, 24, 32, 48, 64]))
    p = metal_sphere("Drude", 40, WATER, [700.0], list(range(8, 65, 2)), extrapolate=list(range(12, 65, 4)),
                     material=DRUDE, tag="drude")
    p["notes"] = "Dense staircase study."
    problems.append(p)
    for core_r, t in ((20, 5), (30, 8), (40, 10), (50, 15)):
        problems.append(core_shell(core_r, t, SIO2, AU, "sio2-au", WATER, wl, [8, 12, 16, 24, 32, 48, 64]))
        problems.append(core_shell(core_r, t, AU, SIO2, "au-sio2", WATER, wl, [8, 12, 16, 24, 32, 48, 64]))
    for metal, mat in (("Au", AU), ("Ag", AG)):
        for ar in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
            problems.append(spheroid(ar, 20, mat, WATER, _wl(400, 1000, 61), [8, 12, 16, 24, 32, 48, 64], metal))
        for ar in (2.0, 3.0, 4.0, 5.0):
            problems.append(rod(ar, 10, mat, WATER, _wl(400, 1100, 71), [8, 12, 16, 20, 24, 32, 40], metal))
        for gap in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
            problems.append(dimer(gap, 20, mat, WATER, _wl(400, 900, 51), [8, 12, 16, 24, 32, 48, 64, 80], metal))
    return {
        "suite": "full",
        "description": "Full paper suite.",
        "budgets": {"cpu": {"max_dipoles": 20000, "max_cell_steps": 5e10},
                    "gpu": {"max_dipoles": 2_000_000, "max_cell_steps": 5e12}},
        "problems": problems,
    }


def full_paper() -> dict:
    """The full suite's 145 problems at a cost one shared consumer GPU finishes in about a day:
    wavelength grids thinned to 9 points, experimental DDA variants only on the dielectric spheres,
    the Drude staircase study and four representative metal problems, DDA max_iter 3000, CPU runs
    only on small lattices and coarse meshes (for matched CPU/GPU comparisons). Run with --max-run-s
    (see benchmarks/README.md)."""
    suite = full()
    keep_variants = {"sphere-au-jc-r40-water", "sphere-ag-jc-r40-water", "coreshell-sio2-au-c30-s8-water",
                     "dimer-au-r20-gap5-water"}
    for p in suite["problems"]:
        wl = p["simulation"]["wavelengths"]
        if isinstance(wl, dict) and wl["num"] > 9:
            wl["num"] = 9
        if p["family"] in ("dielectric_sphere",) or p["id"] == "sphere-drude-r40-water" or p["id"] in keep_variants:
            continue
        p["variants"]["dda"] = {"center": None}
    suite.update(
        suite="full-paper",
        description="Full suite, 9 wavelengths per problem, experimental DDA variants on a subset, DDA max_iter 3000.",
        budgets={"cpu": {"max_dipoles": 1000, "max_cell_steps": 3e8},
                 "gpu": {"max_dipoles": 2_000_000, "max_cell_steps": 5e11}},
        options={"dda": {"max_iter": 3000}},
    )
    return suite


SUITES = {"smoke": smoke, "smoke-extended": smoke_extended, "full": full, "full-paper": full_paper}


def build_suite(name: str) -> dict:
    if name in SUITES:
        suite = SUITES[name]()
    else:
        suite = json.loads(Path(name).read_text(encoding="utf-8"))
    ids = [p["id"] for p in suite["problems"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate problem ids: {sorted(dupes)}")
    return suite


def validate(suite: dict) -> None:
    """Every simulation must validate against the installed lightsmith schema."""
    import lightsmith as ls

    for p in suite["problems"]:
        ls.Simulation.model_validate(p["simulation"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="smoke", choices=sorted(SUITES))
    ap.add_argument("--out", type=Path, default=SUITE_DIR)
    ap.add_argument("--split", action="store_true", help="also write one Simulation JSON per problem")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)
    suite = build_suite(args.suite)
    if not args.no_validate:
        validate(suite)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.suite}.json"
    path.write_text(json.dumps(suite, indent=1), encoding="utf-8")
    if args.split:
        d = args.out / args.suite
        d.mkdir(exist_ok=True)
        for p in suite["problems"]:
            (d / f"{p['id']}.json").write_text(json.dumps(p["simulation"], indent=1), encoding="utf-8")
    print(json.dumps({"suite": args.suite, "problems": len(suite["problems"]), "path": str(path)}))


if __name__ == "__main__":
    main()
