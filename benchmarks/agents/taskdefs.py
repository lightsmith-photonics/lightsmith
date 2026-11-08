"""Task definitions of the Lightsmith agent benchmark.

Each task is a plain dict; ``generate.py`` solves ``reference_spec`` in process,
applies each target's extractor and writes ``tasks/<id>.json`` with the
reference values filled in. Fields:

- ``id``, ``kind`` (problem kind), ``category``: standard | trap | pushback | warning
- ``traps``: what the task tests beyond plain setup (unit conversion, conventions, ...)
- ``prompt``: what a user would write. Hidden from the agent: everything else.
- ``answer_fields``: keys the agent must report via ``submit_answer`` (key, unit, description)
- ``reference_spec`` / ``reference_method`` / ``reference_device``: the correct problem (None for pure
  pushback tasks) and how the reference and the oracle solve it
- ``targets``: key -> {"extract": extractor spec (see extract.py), "tol": {"abs": x} or {"rel": x}}
- ``checks``: extra graders (see grading.py): spec, estimate_before_solve, pushback, mentions, ...
- ``naive_spec``: what a careless agent would solve (tests prove the graders reject it)
- ``oracle``: how the scripted oracle behaves beyond playing the reference spec

Units everywhere: nm, vacuum wavelengths, exp(-i omega t) (Im n > 0 is loss).
"""

from __future__ import annotations

import copy
import math

WATER, OIL = 1.33, 1.515
AU = {"kind": "library", "name": "Au", "dataset": "Johnson-Christy"}
AG = {"kind": "library", "name": "Ag", "dataset": "Johnson-Christy"}
SI_LIB = {"kind": "library", "name": "Si", "dataset": "Green-2008"}
AIR = {"kind": "constant_index", "n_real": 1.0}


def n(re, im=0.0):
    return {"kind": "constant_index", "n_real": re, "n_imag": im}


def sphere(r, mat, pos=(0.0, 0.0, 0.0)):
    return {"shape": {"kind": "sphere", "radius_nm": r}, "material": mat, "position_nm": list(pos)}


def sca(particles, wavelengths, medium=1.0, direction=(0, 0, 1), polarization=(1, 0, 0)):
    return {"kind": "scattering", "particles": particles, "medium_index": medium,
            "illumination": {"kind": "plane_wave", "direction": list(direction), "polarization": list(polarization)},
            "wavelengths": wavelengths}


def wl_range(a, b, num):
    return {"start_nm": a, "stop_nm": b, "num": num}


def stack(sup, layers, sub, wavelengths, angles=(0.0,), pols=("s", "p")):
    return {"kind": "layer_stack", "superstrate": sup, "substrate": sub, "wavelengths": wavelengths,
            "layers": [{"thickness_nm": t, "material": m, **({"coherent": c} if c is not None else {})}
                       for t, m, c in ((L + (None,))[:3] for L in layers)],
            "angles_deg": list(angles), "polarizations": list(pols)}


def rect2d(w, h, mat, cx=0.0, cy=0.0):
    return {"profile": {"kind": "rectangle", "center_nm": [cx, cy], "size_nm": [w, h]}, "material": mat}


def box(center, size, mat):
    return {"geometry": {"kind": "box", "center_nm": list(center), "size_nm": list(size)}, "material": mat}


def port(name, center, size, direction, num_modes=1):
    return {"name": name, "center_nm": list(center), "size_nm": list(size), "direction": direction,
            "num_modes": num_modes}


def val(key, extract, **tol):
    return key, {"extract": extract, "tol": tol}


def field(key, unit, description):
    return {"key": key, "unit": unit, "description": description}


def approx(path, value, **tol):
    return {"type": "spec", "path": path, "approx": value, **({"abs": 1e-6 * max(1, abs(value))} | tol)}


def material(path, wavelength_nm, n_real, n_imag, tol=0.02):
    return {"type": "spec_material", "path": path, "wavelength_nm": wavelength_nm, "n_real": n_real,
            "n_imag": n_imag, "abs": tol}


def has_wavelength(value, tol=0.5):
    return {"type": "spec_wavelength", "wavelength_nm": value, "abs": tol}


ESTIMATE = {"type": "estimate_before_solve"}


def mentions(*words, required=True):
    return {"type": "mentions", "any": list(words), "required": required}


PUSHBACK = {"type": "pushback"}

TASKS: list[dict] = []


def task(**t):
    t.setdefault("category", "standard")
    t.setdefault("traps", [])
    t.setdefault("reference_method", "auto")
    t.setdefault("reference_device", "auto")
    t.setdefault("checks", [])
    t.setdefault("oracle", {})
    t["targets"] = dict(t.get("targets", []))
    for tgt in t["targets"].values():  # name the angle so an agent's extra angles do not make it ambiguous
        if tgt["extract"]["kind"] in ("spectrum", "diffraction_order", "propagating_orders"):
            tgt["extract"].setdefault("angle_deg", 0.0)
    TASKS.append(t)
    return t


# --- scattering ---------------------------------------------------------------------

task(id="sca-au40-water-peak", kind="scattering",
     prompt="I'm working with 40 nm gold nanospheres (that's the diameter) dispersed in water. At what wavelength "
            "does the extinction peak, somewhere between 450 and 650 nm? Johnson & Christy optical data for the gold "
            "is fine.",
     answer_fields=[field("peak_wavelength_nm", "nm", "vacuum wavelength of maximum extinction")],
     reference_spec=sca([sphere(20.0, AU)], wl_range(450.0, 650.0, 201), WATER),
     targets=[val("peak_wavelength_nm", {"kind": "peak_wavelength", "quantity": "c_ext_nm2"}, abs=2.0)],
     checks=[approx("particles[0].shape.radius_nm", 20.0, abs=0.5), approx("medium_index", WATER, abs=0.01)],
     naive_spec=sca([sphere(40.0, AU)], wl_range(450.0, 650.0, 201), 1.0))

task(id="sca-au80-vendor-size-abs", kind="scattering", category="trap", traps=["diameter_vs_radius"],
     prompt="We bought 80 nm gold nanoparticles (vendor size) suspended in water and will heat them with a 532 nm "
            "laser. What is the absorption cross-section of one particle at 532 nm, in nm^2?",
     answer_fields=[field("c_abs_nm2", "nm^2", "absorption cross-section at 532 nm")],
     reference_spec=sca([sphere(40.0, AU)], [532.0], WATER),
     targets=[val("c_abs_nm2", {"kind": "cross_section", "quantity": "c_abs_nm2", "wavelength_nm": 532.0}, rel=0.03)],
     checks=[approx("particles[0].shape.radius_nm", 40.0, abs=0.5), has_wavelength(532.0)],
     naive_spec=sca([sphere(80.0, AU)], [532.0], WATER))

task(id="sca-au-medium-wavelength", kind="scattering", category="trap", traps=["medium_vs_vacuum_wavelength"],
     prompt="A 25 nm radius gold sphere sits in water. Our source is specified by its wavelength inside the water, "
            "which is 450 nm. What's the extinction cross-section at that source? Use J&C gold.",
     answer_fields=[field("c_ext_nm2", "nm^2", "extinction cross-section at the source wavelength")],
     reference_spec=sca([sphere(25.0, AU)], [450.0 * WATER], WATER),
     targets=[val("c_ext_nm2", {"kind": "cross_section", "quantity": "c_ext_nm2", "wavelength_nm": 450.0 * WATER},
                  rel=0.03)],
     checks=[has_wavelength(450.0 * WATER, tol=1.0), approx("medium_index", WATER, abs=0.01)],
     naive_spec=sca([sphere(25.0, AU)], [450.0], WATER))

task(id="sca-si-sphere-ev", kind="scattering", category="trap", traps=["photon_energy_units", "diameter_vs_radius"],
     prompt="Silicon nanosphere, 150 nm in diameter, in air, illuminated with 2.0 eV photons. What's its scattering "
            "efficiency Q_sca? Please use the bundled crystalline silicon data.",
     answer_fields=[field("q_sca", "dimensionless", "scattering efficiency C_sca / (pi r^2)")],
     reference_spec=sca([sphere(75.0, SI_LIB)], [1239.841984 / 2.0], 1.0),
     targets=[val("q_sca", {"kind": "cross_section", "quantity": "q_sca", "wavelength_nm": 1239.841984 / 2.0},
                  rel=0.03)],
     checks=[approx("particles[0].shape.radius_nm", 75.0, abs=0.5), has_wavelength(619.92, tol=0.5)],
     naive_spec=sca([sphere(150.0, SI_LIB)], [620.0], 1.0))

task(id="sca-au-oil-peak", kind="scattering",
     prompt="Where is the localized surface plasmon (extinction maximum) of a 60 nm diameter Au sphere embedded in "
            "immersion oil, n = 1.515? Search between 500 and 700 nm.",
     answer_fields=[field("peak_wavelength_nm", "nm", "vacuum wavelength of maximum extinction")],
     reference_spec=sca([sphere(30.0, AU)], wl_range(500.0, 700.0, 201), OIL),
     targets=[val("peak_wavelength_nm", {"kind": "peak_wavelength", "quantity": "c_ext_nm2"}, abs=2.0)],
     checks=[approx("medium_index", OIL, abs=0.01), approx("particles[0].shape.radius_nm", 30.0, abs=0.5)],
     naive_spec=sca([sphere(30.0, AU)], wl_range(500.0, 700.0, 201), 1.0))

task(id="sca-coreshell-nir", kind="scattering",
     prompt="Nanoshell question: silica core of 60 nm radius (n = 1.45) coated with a 10 nm gold shell, in water. "
            "What's the extinction cross-section at 800 nm?",
     answer_fields=[field("c_ext_nm2", "nm^2", "extinction cross-section at 800 nm")],
     reference_spec=sca([{"shape": {"kind": "layered_sphere", "core_radius_nm": 60.0,
                                    "shells": [{"thickness_nm": 10.0, "material": AU}]},
                          "material": n(1.45)}], [800.0], WATER),
     targets=[val("c_ext_nm2", {"kind": "cross_section", "quantity": "c_ext_nm2", "wavelength_nm": 800.0}, rel=0.03)],
     checks=[approx("particles[0].shape.shells[0].thickness_nm", 10.0, abs=0.5),
             approx("particles[0].shape.core_radius_nm", 60.0, abs=0.5)],
     naive_spec=sca([{"shape": {"kind": "layered_sphere", "core_radius_nm": 50.0,
                                "shells": [{"thickness_nm": 10.0, "material": AU}]},
                      "material": n(1.45)}], [800.0], WATER))

# The longitudinal plasmon of this rod is sharp (FWHM ~25 nm, peak ~640 nm). Q_ext at a fixed wavelength on its flank
# is ill-conditioned: a few nm of peak shift moves it by 20-40 %, and neither BEM (3 -> 1 nm mesh) nor DDA (24 -> 80
# dipoles per diameter) converged there to 8 %. The peak wavelength is robust: BEM 638.9 / 639.5 nm at 2 / 1.5 nm
# meshes, DDA 645.5 / 643.9 / 643.2 nm at 80 / 120 / 160 dipoles per diameter (1 nm sampling, parabolic peak; the
# quasistatic Gans model gives 628 nm). The naive set-up (polarisation across the rod: only the transverse mode,
# ~520 nm) lands far outside the tolerance.
task(id="sca-au-nanorod-peak", kind="scattering", traps=["estimate_first"],
     reference_method="bem",
     prompt="I'm modelling a small gold nanorod as a prolate spheroid with semi-axes 10 nm x 10 nm x 25 nm, in water, "
            "with light polarised along the long axis (propagating perpendicular to it). At what wavelength does its "
            "longitudinal plasmon peak in extinction, somewhere between 550 and 750 nm? Johnson & Christy data for the "
            "gold is fine.",
     answer_fields=[field("peak_wavelength_nm", "nm", "vacuum wavelength of maximum extinction")],
     reference_spec=sca([{"shape": {"kind": "spheroid", "semi_axis_xy_nm": 10.0, "semi_axis_z_nm": 25.0},
                          "material": AU}], wl_range(626.0, 656.0, 16), WATER, direction=(1, 0, 0),
                        polarization=(0, 0, 1)),
     targets=[val("peak_wavelength_nm", {"kind": "peak_wavelength", "quantity": "c_ext_nm2"}, abs=8.0)],
     checks=[ESTIMATE, approx("particles[0].shape.semi_axis_z_nm", 25.0, abs=0.5),
             approx("medium_index", WATER, abs=0.01)],
     naive_spec=sca([{"shape": {"kind": "spheroid", "semi_axis_xy_nm": 10.0, "semi_axis_z_nm": 25.0},
                      "material": AU}], wl_range(550.0, 750.0, 41), WATER))

task(id="sca-quasistatic-too-big", kind="scattering", category="warning", traps=["check_warnings"],
     reference_method="quasistatic",
     prompt="Quick one: use the quasistatic (dipole) approximation, since it's the fastest, to get Q_ext for a gold "
            "sphere of 150 nm radius in air at 600 nm.",
     answer_fields=[field("q_ext", "dimensionless", "extinction efficiency at 600 nm")],
     reference_spec=sca([sphere(150.0, AU)], [600.0], 1.0),
     checks=[mentions("unreliable", "not valid", "invalid", "too large", "size parameter", "inaccurate",
                      "breaks down", "mie", "not appropriate", "not accurate")],
     oracle={"concerns": ["The quasistatic approximation is unreliable here: the size parameter is far above 1. "
                          "Use Mie theory instead."]})

task(id="sca-ejwt-absorbing", kind="scattering", category="trap", traps=["exp_plus_iwt_convention"],
     prompt="My collaborator measured n = 2.0 - 0.3j for a particle material at 633 nm (they use the electrical "
            "engineering convention, fields varying as e^{jωt}). For a 100 nm radius sphere of it in air, what is the "
            "absorption cross-section at 633 nm?",
     answer_fields=[field("c_abs_nm2", "nm^2", "absorption cross-section at 633 nm")],
     reference_spec=sca([sphere(100.0, n(2.0, 0.3))], [633.0], 1.0),
     targets=[val("c_abs_nm2", {"kind": "cross_section", "quantity": "c_abs_nm2", "wavelength_nm": 633.0}, rel=0.02)],
     checks=[material("particles[0].material", 633.0, 2.0, 0.3, tol=0.01)],
     naive_spec=sca([sphere(100.0, n(2.0, 0.0))], [633.0], 1.0))

task(id="sca-ejwt-gain-pushback", kind="scattering", category="pushback", traps=["exp_plus_iwt_convention", "gain"],
     prompt="We have dye-doped polymer beads, radius 200 nm, in water. Our ellipsometry software (e^{+jωt} "
            "convention) reports n = 1.6 + 0.02j at 590 nm. What's the absorption cross-section of one bead at 590 nm?",
     answer_fields=[field("c_abs_nm2", "nm^2", "absorption cross-section at 590 nm")],
     reference_spec=None,
     checks=[PUSHBACK, mentions("gain", "amplif", "negative absorption", "active")],
     naive_spec=sca([sphere(200.0, n(1.6, 0.02))], [590.0], WATER),
     oracle={"decline": True,
             "concerns": ["With e^{+jωt}, n = 1.6 + 0.02j has gain (amplification), not loss; Lightsmith uses "
                          "exp(-iωt) and passive materials only. Please confirm the sign of the measured k."]})

task(id="sca-au-midir-dataset", kind="scattering", category="trap", traps=["material_range"],
     prompt="What's the extinction cross-section of a 30 nm radius gold sphere in air at 2.5 µm wavelength? Use "
            "measured/fitted gold optical constants rather than a toy Drude model.",
     answer_fields=[field("c_ext_nm2", "nm^2", "extinction cross-section at 2.5 um")],
     reference_spec=sca([sphere(30.0, {"kind": "library", "name": "Au", "dataset": "Rakic-LD"})], [2500.0], 1.0),
     targets=[val("c_ext_nm2", {"kind": "cross_section", "quantity": "c_ext_nm2", "wavelength_nm": 2500.0}, rel=0.15)],
     checks=[has_wavelength(2500.0)],
     naive_spec=sca([sphere(30.0, AU)], [1900.0], 1.0))

task(id="sca-ps-bead-wavenumber", kind="scattering", category="trap", traps=["wavenumber_units", "diameter_vs_radius"],
     prompt="Polystyrene bead, 1 µm diameter, n = 1.59 (assume non-absorbing), in water. What is its scattering "
            "efficiency at 15,800 cm^-1?",
     answer_fields=[field("q_sca", "dimensionless", "scattering efficiency")],
     reference_spec=sca([sphere(500.0, n(1.59))], [1e7 / 15800.0], WATER),
     targets=[val("q_sca", {"kind": "cross_section", "quantity": "q_sca", "wavelength_nm": 1e7 / 15800.0}, rel=0.03)],
     checks=[has_wavelength(1e7 / 15800.0, tol=0.5), approx("particles[0].shape.radius_nm", 500.0, abs=2)],
     naive_spec=sca([sphere(1000.0, n(1.59))], [633.0], WATER))

# --- layer stacks -------------------------------------------------------------------

D_MGF2 = 550.0 / (4 * 1.38)
task(id="ls-quarter-wave-ar", kind="layer_stack",
     prompt="Single-layer MgF2 antireflection coating (n = 1.38) on crown glass (n = 1.52), designed as a quarter-wave "
            "layer for 550 nm. What's the reflectance at 550 nm at normal incidence from air?",
     answer_fields=[field("reflectance", "fraction (0-1)", "reflectance at 550 nm, normal incidence")],
     reference_spec=stack(AIR, [(D_MGF2, n(1.38))], n(1.52), [550.0]),
     targets=[val("reflectance", {"kind": "spectrum", "quantity": "r", "pol": "avg", "wavelength_nm": 550.0},
                  abs=0.002)],
     checks=[approx("layers[0].thickness_nm", D_MGF2, abs=1.0)],
     naive_spec=stack(AIR, [(550.0 / 4, n(1.38))], n(1.52), [550.0]))

task(id="ls-brewster-angle", kind="layer_stack",
     prompt="For light going from air onto an uncoated glass surface (n = 1.5) at 633 nm, find the angle where "
            "p-polarized reflection vanishes, from a simulation of the reflectance versus angle.",
     answer_fields=[field("brewster_angle_deg", "degrees", "angle of incidence of minimum p reflectance")],
     reference_spec=stack(AIR, [], n(1.5), [633.0], angles=[50 + 0.1 * i for i in range(121)], pols=("p",)),
     targets=[val("brewster_angle_deg", {"kind": "spectrum_extremum_angle", "quantity": "r", "pol": "p",
                                         "wavelength_nm": 633.0, "mode": "min"}, abs=0.3)],
     naive_spec=stack(AIR, [], n(1.5), [633.0], angles=[45.0, 50.0, 55.0], pols=("p",)))

QW = 1550.0 / 4
task(id="ls-bragg-mirror", kind="layer_stack",
     prompt="Dielectric mirror for 1550 nm: 8 pairs of quarter-wave TiO2 (n = 2.35) / SiO2 (n = 1.45), TiO2 first on "
            "the air side, deposited on a silicon wafer (n = 3.48). Reflectance at 1550 nm, normal incidence?",
     answer_fields=[field("reflectance", "fraction (0-1)", "reflectance at 1550 nm")],
     reference_spec=stack(AIR, [(QW / n_, n(n_)) for _ in range(8) for n_ in (2.35, 1.45)], n(3.48), [1550.0]),
     targets=[val("reflectance", {"kind": "spectrum", "quantity": "r", "pol": "avg", "wavelength_nm": 1550.0},
                  abs=0.003)],
     checks=[approx("layers[0].thickness_nm", QW / 2.35, abs=1.0)],
     naive_spec=stack(AIR, [(QW, n(n_)) for _ in range(8) for n_ in (2.35, 1.45)], n(3.48), [1550.0]))

task(id="ls-gold-film-glass-side", kind="layer_stack", category="trap", traps=["incidence_side"],
     prompt="A 50 nm gold film is evaporated on a glass slide (n = 1.5). In our setup the 633 nm beam comes in through "
            "the glass and hits the gold from the glass side, at normal incidence; the other side of the gold is air. "
            "What reflectance do we measure on the glass side? (Ignore the back surface of the slide, J&C gold.)",
     answer_fields=[field("reflectance", "fraction (0-1)", "reflectance at 633 nm on the glass side")],
     reference_spec=stack(n(1.5), [(50.0, AU)], AIR, [633.0]),
     targets=[val("reflectance", {"kind": "spectrum", "quantity": "r", "pol": "avg", "wavelength_nm": 633.0},
                  abs=0.005)],
     checks=[material("superstrate", 633.0, 1.5, 0.0), material("substrate", 633.0, 1.0, 0.0)],
     naive_spec=stack(AIR, [(50.0, AU)], n(1.5), [633.0]))

task(id="ls-silver-film-ev", kind="layer_stack", category="trap", traps=["photon_energy_units"],
     prompt="How much of the incident light is absorbed by a 30 nm silver film on glass (n = 1.52) at a photon energy "
            "of 2.0 eV? Light comes from air at normal incidence. Johnson & Christy silver.",
     answer_fields=[field("absorptance", "fraction (0-1)", "absorbed fraction")],
     reference_spec=stack(AIR, [(30.0, AG)], n(1.52), [1239.841984 / 2.0]),
     targets=[val("absorptance", {"kind": "spectrum", "quantity": "a", "pol": "avg",
                                  "wavelength_nm": 1239.841984 / 2.0}, abs=0.005)],
     checks=[has_wavelength(619.92, tol=0.5)],
     naive_spec=stack(AIR, [(30.0, AG)], n(1.52), [400.0]))

task(id="ls-tir-glass-air", kind="layer_stack",
     prompt="Light inside a glass block (n = 1.5) hits the flat glass/air interface at 45 degrees, s-polarized, "
            "600 nm. "
            "What fraction is transmitted into the air?",
     answer_fields=[field("transmittance", "fraction (0-1)", "transmitted power fraction")],
     reference_spec=stack(n(1.5), [], AIR, [600.0], angles=[45.0], pols=("s",)),
     targets=[val("transmittance", {"kind": "spectrum", "quantity": "t", "pol": "s", "angle_deg": 45.0,
                                    "wavelength_nm": 600.0}, abs=0.002)],
     checks=[material("superstrate", 600.0, 1.5, 0.0)],
     naive_spec=stack(AIR, [], n(1.5), [600.0], angles=[45.0], pols=("s",)))

task(id="ls-si-film-wavenumber", kind="layer_stack", category="trap", traps=["wavenumber_units", "um_units"],
     prompt="Free-standing silicon membrane, 1.0 µm thick, treat it as lossless with n = 3.45. What's the normal-"
            "incidence transmittance at 5000 cm^-1?",
     answer_fields=[field("transmittance", "fraction (0-1)", "transmittance at 5000 cm^-1")],
     reference_spec=stack(AIR, [(1000.0, n(3.45))], AIR, [2000.0]),
     targets=[val("transmittance", {"kind": "spectrum", "quantity": "t", "pol": "avg", "wavelength_nm": 2000.0},
                  abs=0.005)],
     checks=[has_wavelength(2000.0), approx("layers[0].thickness_nm", 1000.0, abs=1.0)],
     naive_spec=stack(AIR, [(1000.0, n(3.45))], AIR, [5000.0]))

task(id="ls-thick-slide-incoherent", kind="layer_stack", category="trap", traps=["incoherent_layer"],
     prompt="What's the transmittance of a plain 1 mm thick glass microscope slide (n = 1.5, no absorption) in air at "
            "550 nm, normal incidence? We measure with a broadband lamp and a spectrometer of ~1 nm resolution.",
     answer_fields=[field("transmittance", "fraction (0-1)", "transmittance at 550 nm")],
     reference_spec=stack(AIR, [(1e6, n(1.5), False)], AIR, [550.0]),
     targets=[val("transmittance", {"kind": "spectrum", "quantity": "t", "pol": "avg", "wavelength_nm": 550.0},
                  abs=0.003)],
     checks=[{"type": "spec", "path": "layers[0].coherent", "equals": False}],
     naive_spec=stack(AIR, [(1e6 + 50, n(1.5))], AIR, [550.0]))

# --- periodic -------------------------------------------------------------------------


def grating1d(period, height, fill, ridge, wavelengths, sub=n(1.5), theta=0.0, pols=("s", "p"), sup=AIR):
    return {"kind": "periodic", "lattice_a1_nm": [period, 0.0], "superstrate": sup, "substrate": sub,
            "layers": [{"thickness_nm": height, "background": AIR,
                        "shapes": [rect2d(fill * period, 1000.0, ridge)]}],
            "wavelengths": wavelengths, "incidence": {"theta_deg": theta, "phi_deg": 0.0, "polarizations": list(pols)}}


task(id="per-grating-first-order", kind="periodic", traps=["estimate_first"],
     prompt="Binary transmission grating: period 1 µm, 400 nm tall ridges of n = 1.5 with 50% fill factor, on a glass "
            "substrate (n = 1.5), air above. At 633 nm, normal incidence from the air side, with the E-field parallel "
            "to the grating lines (TE), what fraction of the power goes into the +1 transmitted order?",
     answer_fields=[field("t_plus1", "fraction (0-1)", "+1 order transmitted diffraction efficiency")],
     reference_spec=grating1d(1000.0, 400.0, 0.5, n(1.5), [633.0], pols=("s",)),
     reference_options={"rcwa": {"harmonics": 40}},
     targets=[val("t_plus1", {"kind": "diffraction_order", "order": [1, 0], "side": "t", "pol": "s",
                              "wavelength_nm": 633.0}, abs=0.01)],
     checks=[ESTIMATE, approx("lattice_a1_nm[0]", 1000.0, abs=1.0), approx("layers[0].thickness_nm", 400.0, abs=1.0)],
     naive_spec=grating1d(1000.0, 400.0, 0.5, n(1.5), [633.0], pols=("p",)))

task(id="per-subwavelength-orders", kind="periodic", traps=["estimate_first"],
     prompt="A 300 nm period binary grating (200 nm tall, 50% duty cycle, ridges n = 2.0) sits on glass (n = 1.5) "
            "under air. Illuminated at normal incidence at 633 nm, how many transmitted diffraction orders propagate "
            "in the glass (count the zeroth order too)? Please verify with a simulation.",
     answer_fields=[field("num_transmitted_orders", "count", "number of propagating transmitted orders")],
     reference_spec=grating1d(300.0, 200.0, 0.5, n(2.0), [633.0]),
     targets=[val("num_transmitted_orders", {"kind": "propagating_orders", "side": "t", "wavelength_nm": 633.0},
                  abs=0.4)],
     checks=[ESTIMATE],
     naive_spec=grating1d(300.0, 200.0, 0.5, n(2.0), [300.0]))

task(id="per-tio2-metasurface", kind="periodic", traps=["estimate_first"],
     prompt="Metasurface: square lattice, 400 nm pitch, of TiO2 nanopillars (n = 2.4, lossless), 100 nm radius and "
            "300 nm tall, on a glass substrate (n = 1.46), air above. Total reflectance at 700 nm for unpolarized "
            "light at normal incidence?",
     answer_fields=[field("reflectance", "fraction (0-1)", "total reflectance (all orders), unpolarized")],
     reference_spec={"kind": "periodic", "lattice_a1_nm": [400.0, 0.0], "lattice_a2_nm": [0.0, 400.0],
                     "superstrate": AIR, "substrate": n(1.46),
                     "layers": [{"thickness_nm": 300.0, "background": AIR,
                                 "shapes": [{"profile": {"kind": "circle", "radius_nm": 100.0}, "material": n(2.4)}]}],
                     "wavelengths": [700.0]},
     # Laurent's rule with 3 orders per direction is within 0.001 of the converged default (normal-vector
     # factorisation, 7 orders: R = 0.0399) here and takes < 1 s; the defaults take minutes.
     reference_options={"rcwa": {"harmonics": 3, "factorization": "laurent"}},
     targets=[val("reflectance", {"kind": "spectrum", "quantity": "r", "pol": "avg", "wavelength_nm": 700.0},
                  abs=0.02)],
     checks=[ESTIMATE, approx("lattice_a1_nm[0]", 400.0, abs=1.0)],
     naive_spec={"kind": "periodic", "lattice_a1_nm": [400.0, 0.0], "lattice_a2_nm": [0.0, 400.0],
                 "superstrate": AIR, "substrate": n(1.46),
                 "layers": [{"thickness_nm": 300.0, "background": AIR,
                             "shapes": [{"profile": {"kind": "circle", "radius_nm": 200.0}, "material": n(2.4)}]}],
                 "wavelengths": [700.0]})

task(id="per-grating-oblique-r0", kind="periodic", traps=["estimate_first"],
     prompt="Same kind of grating as a textbook example: period 1 µm, 400 nm tall n = 1.5 ridges, 50% fill, glass "
            "substrate n = 1.5, air on top. Light arrives at 20 degrees from the normal in the plane perpendicular to "
            "the grating lines, p-polarized (TM), 633 nm. What's the specular (zeroth-order) reflectance?",
     answer_fields=[field("r_zeroth", "fraction (0-1)", "zeroth-order reflectance")],
     reference_spec=grating1d(1000.0, 400.0, 0.5, n(1.5), [633.0], theta=20.0, pols=("p",)),
     reference_options={"rcwa": {"harmonics": 40}},
     targets=[val("r_zeroth", {"kind": "diffraction_order", "order": [0, 0], "side": "r", "pol": "p",
                               "angle_deg": 20.0, "wavelength_nm": 633.0}, abs=0.005)],
     checks=[ESTIMATE, approx("incidence.theta_deg", 20.0, abs=0.1)],
     naive_spec=grating1d(1000.0, 400.0, 0.5, n(1.5), [633.0], theta=0.0, pols=("p",)))

# --- waveguide modes ---------------------------------------------------------------


def strip(w, h, core, clad, wavelengths, num_modes=2, window=(2500.0, 1600.0), **opts):
    return {"kind": "mode", "structures": [rect2d(w, h, core)], "background": clad, "window_size_nm": list(window),
            "wavelengths": wavelengths, "num_modes": num_modes, **({"options": {"fde": opts}} if opts else {})}


SI, SIO2 = n(3.476), n(1.444)
task(id="mode-si-strip-te0", kind="mode",
     prompt="Standard SOI strip waveguide: 500 nm wide, 220 nm thick silicon (n = 3.476), fully clad in SiO2 "
            "(n = 1.444). What's the effective index of the fundamental TE mode at 1550 nm?",
     answer_fields=[field("neff", "dimensionless", "effective index of TE0 at 1550 nm")],
     reference_spec=strip(500.0, 220.0, SI, SIO2, [1550.0], grid_spacing_nm=20.0),
     targets=[val("neff", {"kind": "mode", "field": "neff_real", "select": "te", "wavelength_nm": 1550.0}, abs=0.01)],
     checks=[approx("structures[0].profile.size_nm[0]", 500.0, abs=1.0)],
     naive_spec=strip(220.0, 500.0, SI, SIO2, [1550.0]))

task(id="mode-si-strip-ng", kind="mode", category="trap", traps=["group_vs_phase_index"],
     prompt="For a 500 x 220 nm silicon strip (n = 3.476) in SiO2 (n = 1.444), what's the group index of the TE0 "
            "mode at 1550 nm? I need it to estimate a ring's FSR.",
     answer_fields=[field("ng", "dimensionless", "group index of TE0 at 1550 nm")],
     reference_spec=strip(500.0, 220.0, SI, SIO2, [1550.0], grid_spacing_nm=20.0),
     targets=[val("ng", {"kind": "mode", "field": "ng", "select": "te", "wavelength_nm": 1550.0}, abs=0.08)],
     naive_spec=strip(500.0, 220.0, SI, SIO2, [1550.0]),
     oracle={"naive_extract": {"ng": {"kind": "mode", "field": "neff_real", "select": "te", "wavelength_nm": 1550.0}}})

task(id="mode-si-strip-tm0", kind="mode",
     prompt="What's the effective index of the fundamental TM-like mode (E mostly vertical) of a 500 nm x 220 nm "
            "silicon strip waveguide (n = 3.476) in oxide (n = 1.444) at 1550 nm?",
     answer_fields=[field("neff_tm0", "dimensionless", "effective index of TM0")],
     reference_spec=strip(500.0, 220.0, SI, SIO2, [1550.0], num_modes=3, grid_spacing_nm=20.0),
     targets=[val("neff_tm0", {"kind": "mode", "field": "neff_real", "select": "tm", "wavelength_nm": 1550.0},
                  abs=0.015)],
     naive_spec=strip(500.0, 220.0, SI, SIO2, [1550.0], num_modes=1))

task(id="mode-sin-thz", kind="mode", category="trap", traps=["frequency_units"],
     prompt="Silicon nitride waveguide (n = 2.0), 1.0 µm wide and 400 nm thick, buried in SiO2 (n = 1.444). What is "
            "the TE0 effective index at an optical frequency of 193.4 THz?",
     answer_fields=[field("neff", "dimensionless", "TE0 effective index")],
     reference_spec=strip(1000.0, 400.0, n(2.0), SIO2, [299792.458 / 193.4], window=(3500.0, 2500.0),
                          grid_spacing_nm=25.0),
     targets=[val("neff", {"kind": "mode", "field": "neff_real", "select": "te",
                           "wavelength_nm": 299792.458 / 193.4}, abs=0.01)],
     checks=[has_wavelength(1550.1, tol=1.0), approx("structures[0].profile.size_nm[0]", 1000.0, abs=1.0)],
     naive_spec=strip(1000.0, 400.0, n(2.0), SIO2, [1934.0], window=(4000.0, 3000.0)))

task(id="mode-guided-count", kind="mode", category="warning", traps=["check_warnings"],
     prompt="How many guided modes (counting both polarizations) does a 400 nm x 220 nm silicon wire (n = 3.476) "
            "in SiO2 (n = 1.444) support at 1550 nm?",
     answer_fields=[field("num_guided_modes", "count", "number of guided modes")],
     reference_spec=strip(400.0, 220.0, SI, SIO2, [1550.0], num_modes=6, grid_spacing_nm=20.0),
     targets=[val("num_guided_modes", {"kind": "guided_mode_count", "cladding_index": 1.444,
                                       "wavelength_nm": 1550.0}, abs=0.4)],
     naive_spec=strip(400.0, 220.0, SI, SIO2, [1550.0], num_modes=6, window=(400.0, 220.0)))

task(id="mode-neff-above-core-pushback", kind="mode", category="pushback", traps=["unphysical_request"],
     prompt="Find the width of a 220 nm thick silicon strip (n = 3.476) in SiO2 for which the fundamental TE mode has "
            "an effective index of 3.8 at 1550 nm.",
     answer_fields=[field("width_nm", "nm", "strip width giving neff = 3.8")],
     reference_spec=None,
     checks=[PUSHBACK, mentions("core index", "exceed", "larger than", "greater than", "cannot", "impossible",
                                "above the", "bounded", "not possible", "3.476")],
     oracle={"decline": True, "concerns": ["A guided mode's effective index cannot exceed the core index (3.476); "
                                           "neff = 3.8 is impossible for any width."]})

task(id="mode-fiber-fundamental", kind="mode", traps=["um_units"],
     prompt="Weakly guiding step-index fibre: core n = 1.46, cladding n = 1.45, core radius 2.5 µm. Effective index "
            "of the fundamental (HE11) mode at λ = 1.55 µm?",
     answer_fields=[field("neff", "dimensionless", "HE11 effective index")],
     reference_spec={"kind": "mode", "structures": [{"profile": {"kind": "circle", "radius_nm": 2500.0},
                                                     "material": n(1.46)}],
                     "background": n(1.45), "window_size_nm": [16000.0, 16000.0], "wavelengths": [1550.0],
                     "num_modes": 1, "options": {"fde": {"grid_spacing_nm": 80.0, "symmetry": ["pec", "pmc"]}}},
     targets=[val("neff", {"kind": "mode", "field": "neff_real", "wavelength_nm": 1550.0}, abs=6e-4)],
     checks=[approx("structures[0].profile.radius_nm", 2500.0, abs=5.0)],
     naive_spec={"kind": "mode", "structures": [{"profile": {"kind": "circle", "radius_nm": 1250.0},
                                                 "material": n(1.46)}],
                 "background": n(1.45), "window_size_nm": [16000.0, 16000.0], "wavelengths": [1550.0], "num_modes": 1})

# --- devices (S-parameters) --------------------------------------------------------

CORE2D, CLAD2D = n(2.85), n(1.44)
MMI_W, MMI_ACCESS = 3000.0, 800.0
_we = MMI_W + 1550 / math.pi * (1.44 / 2.85) ** 2 / math.sqrt(2.85**2 - 1.44**2)
MMI_L = round(0.95 * 2.85 * _we**2 / (2 * 1550) / 10) * 10


def mmi(length, access=MMI_ACCESS):
    x = length / 2 + 1000
    st = [box((-x, 0, 0), (2 * x - length, access, 1), CORE2D), box((0, 0, 0), (length, MMI_W, 1), CORE2D)]
    st += [box((x, y, 0), (2 * x - length, access, 1), CORE2D) for y in (-MMI_W / 4, MMI_W / 4)]
    return {"kind": "device", "dimension": 2, "structures": st, "background": CLAD2D,
            "domain_size_nm": [2 * x, MMI_W + 3000, 0], "wavelengths": [1550.0],
            "ports": [port("o1", (-x, 0, 0), (0, MMI_W / 2, 1), "+"),
                      port("o2", (x, -MMI_W / 4, 0), (0, MMI_W / 2, 1), "-"),
                      port("o3", (x, MMI_W / 4, 0), (0, MMI_W / 2, 1), "-")]}


task(id="dev-mmi-1x2-2d", kind="device", traps=["estimate_first"],
     prompt=f"2D effective-index model of a 1x2 MMI splitter at 1550 nm (TE, E in plane): core index 2.85, cladding "
            f"1.44. The multimode section is {MMI_W / 1000:g} µm wide and {MMI_L / 1000:g} µm long, centred on the "
            f"origin; a {MMI_ACCESS:g} nm wide input waveguide enters on the axis, and two {MMI_ACCESS:g} nm output "
            f"waveguides leave at y = ±{MMI_W / 4:g} nm. What fraction of the input power reaches each output in the "
            f"fundamental mode?",
     answer_fields=[field("t_upper", "fraction (0-1)", "power to the y = +750 nm output, fundamental mode"),
                    field("t_lower", "fraction (0-1)", "power to the y = -750 nm output, fundamental mode")],
     reference_spec=mmi(MMI_L),
     reference_options={"eme": {"num_modes": 40, "grid_spacing_nm": 10.0}},
     targets=[val("t_upper", {"kind": "sparam", "out": "o3", "in": "o1", "wavelength_nm": 1550.0}, abs=0.04),
              val("t_lower", {"kind": "sparam", "out": "o2", "in": "o1", "wavelength_nm": 1550.0}, abs=0.04)],
     checks=[ESTIMATE],
     naive_spec=mmi(MMI_L / 2))


def width_step(w1, w2, length=4000.0):
    return {"kind": "device", "dimension": 2, "background": CLAD2D,
            "structures": [box((-length / 4, 0, 0), (length / 2, w1, 1), CORE2D),
                           box((length / 4, 0, 0), (length / 2, w2, 1), CORE2D)],
            "domain_size_nm": [length, w2 + 3000, 0], "wavelengths": [1550.0],
            "ports": [port("o1", (-length / 2, 0, 0), (0, w2 + 2000, 1), "+"),
                      port("o2", (length / 2, 0, 0), (0, w2 + 2000, 1), "-")]}


task(id="dev-width-step-2d", kind="device", traps=["estimate_first"],
     prompt="In a 2D slab model (core n = 2.85, cladding n = 1.44, TE), a 500 nm wide waveguide is butt-joined to a "
            "2 µm wide one with no taper. At 1550 nm, what fraction of the fundamental-mode power arrives in the "
            "fundamental mode of the wide guide?",
     answer_fields=[field("t_fundamental", "fraction (0-1)", "fundamental-to-fundamental power transmission")],
     reference_spec=width_step(500.0, 2000.0),
     reference_options={"eme": {"num_modes": 40, "grid_spacing_nm": 10.0}},
     targets=[val("t_fundamental", {"kind": "sparam", "out": "o2", "in": "o1", "wavelength_nm": 1550.0}, abs=0.02)],
     checks=[ESTIMATE],
     naive_spec=width_step(500.0, 500.0))

# --- photonic bands ---------------------------------------------------------------

SQ_PATH = [{"label": "Γ", "reduced": [0, 0, 0]}, {"label": "X", "reduced": [0.5, 0, 0]},
           {"label": "M", "reduced": [0.5, 0.5, 0]}, {"label": "Γ", "reduced": [0, 0, 0]}]
TRI_PATH = [{"label": "Γ", "reduced": [0, 0, 0]}, {"label": "M", "reduced": [0.5, 0, 0]},
            {"label": "K", "reduced": [1 / 3, 2 / 3, 0]}, {"label": "Γ", "reduced": [0, 0, 0]}]


def rods(a=400.0, eps=8.9, r=0.2, pol="tm", nb=4, bg=AIR):
    return {"kind": "bands", "dimension": 2, "lattice_vectors_nm": [[a, 0, 0], [0, a, 0]], "background": bg,
            "structures_2d": [{"profile": {"kind": "circle", "radius_nm": r * a},
                               "material": {"kind": "constant_permittivity", "eps_real": eps}}],
            "k_path": SQ_PATH, "points_per_segment": 8, "num_bands": nb, "polarization": pol}


task(id="bands-rods-tm-gap", kind="bands", traps=["estimate_first"],
     prompt="Classic 2D photonic crystal: square lattice of alumina rods (ε = 8.9) in air, rod radius 0.2a. How wide "
            "is the TM band gap (E along the rods) between the first and second bands, as gap-to-midgap ratio?",
     answer_fields=[field("gap_midgap_ratio", "fraction (e.g. 0.3 = 30%)", "TM gap width / midgap frequency")],
     reference_spec=rods(),
     targets=[val("gap_midgap_ratio", {"kind": "band_gap", "pol": "tm", "field": "gap_midgap_ratio",
                                       "between_bands": [1, 2]}, abs=0.015)],
     checks=[ESTIMATE, approx("structures_2d[0].profile.radius_nm", 0.2 * 400.0, rel=0.01)],
     naive_spec=rods(pol="te"))

task(id="bands-gap-wavelengths", kind="bands", category="trap", traps=["normalized_frequency", "estimate_first"],
     prompt="We want to fabricate a square lattice of alumina rods (ε = 8.9, radius = 0.2 × lattice constant) in air "
            "with a 600 nm lattice constant. Which range of free-space wavelengths falls in the TM band gap? Give the "
            "short and long wavelength edges.",
     answer_fields=[field("lambda_min_nm", "nm", "short-wavelength edge of the TM gap"),
                    field("lambda_max_nm", "nm", "long-wavelength edge of the TM gap")],
     reference_spec=rods(a=600.0),
     targets=[val("lambda_min_nm", {"kind": "band_gap", "pol": "tm", "field": "wavelength_min_nm",
                                    "lattice_constant_nm": 600.0}, rel=0.015),
              val("lambda_max_nm", {"kind": "band_gap", "pol": "tm", "field": "wavelength_max_nm",
                                    "lattice_constant_nm": 600.0}, rel=0.015)],
     checks=[ESTIMATE, approx("lattice_vectors_nm[0][0]", 600.0, abs=1.0)],
     naive_spec=rods(a=400.0),
     oracle={"naive_extract": {  # reports the normalised gap edges a/lambda instead of wavelengths
         "lambda_min_nm": {"kind": "band_gap", "pol": "tm", "field": "top"},
         "lambda_max_nm": {"kind": "band_gap", "pol": "tm", "field": "bottom"}}})

task(id="bands-triangular-holes-complete", kind="bands", traps=["estimate_first"],
     prompt="Triangular lattice of air holes (radius 0.45a) in a dielectric with ε = 13, 2D. Is there a complete band "
            "gap (both TE and TM)? Give the gap-to-midgap ratio of the largest complete gap.",
     answer_fields=[field("complete_gap_midgap_ratio", "fraction", "largest complete gap / midgap (0 if none)")],
     reference_spec={"kind": "bands", "dimension": 2, "lattice_vectors_nm": [[400.0, 0, 0],
                                                                             [200.0, 400.0 * math.sqrt(3) / 2, 0]],
                     "background": {"kind": "constant_permittivity", "eps_real": 13.0},
                     "structures_2d": [{"profile": {"kind": "circle", "radius_nm": 0.45 * 400.0}, "material": AIR}],
                     "k_path": TRI_PATH, "points_per_segment": 8, "num_bands": 8, "polarization": "both"},
     targets=[val("complete_gap_midgap_ratio", {"kind": "band_gap", "pol": "complete", "field": "gap_midgap_ratio"},
                  abs=0.025)],
     checks=[ESTIMATE])

task(id="bands-no-contrast", kind="bands", category="trap", traps=["trivial_physics"],
     prompt="Square lattice (a = 500 nm) of circular holes, radius 150 nm, in a slab of index 1.0 (air), 2D model, "
            "holes also filled with air. What's the TM band gap (gap-to-midgap ratio)?",
     answer_fields=[field("gap_midgap_ratio", "fraction", "TM gap / midgap (0 if there is no gap)")],
     reference_spec=rods(a=500.0, eps=1.0, r=0.3),
     targets=[val("gap_midgap_ratio", {"kind": "band_gap", "pol": "tm", "field": "gap_midgap_ratio"}, abs=0.005)],
     oracle={"concerns": ["There is no index contrast, so there is no band gap (empty lattice)."]})

# --- circuits ---------------------------------------------------------------------

GUIDE = {"neff": 2.4, "ng": 4.2, "wl0_nm": 1550.0}


def mzi(dl, wl=wl_range(1540.0, 1560.0, 4001)):
    return {"kind": "circuit", "wavelengths": wl,
            "instances": {"split": {"component": "y_branch"}, "comb": {"component": "y_branch"},
                          "short": {"component": "waveguide", "settings": {"length_nm": 50000.0, **GUIDE}},
                          "long": {"component": "waveguide", "settings": {"length_nm": 50000.0 + dl, **GUIDE}}},
            "connections": [["split,o2", "short,o1"], ["split,o3", "long,o1"],
                            ["short,o2", "comb,o2"], ["long,o2", "comb,o3"]],
            "ports": {"in": "split,o1", "out": "comb,o1"}}


task(id="circ-mzi-fsr", kind="circuit",
     prompt="Unbalanced Mach-Zehnder made of two ideal Y-branches and two waveguides (neff = 2.4, group index 4.2 at "
            "1550 nm, lossless) whose lengths differ by 100 µm. What's the free spectral range near 1550 nm?",
     answer_fields=[field("fsr_nm", "nm", "free spectral range near 1550 nm")],
     reference_spec=mzi(1e5),
     targets=[val("fsr_nm", {"kind": "sparam_fsr", "out": "out", "in": "in", "near_nm": 1550.0}, rel=0.01)],
     naive_spec=mzi(1e5 / 2))


def ring(radius, wl=wl_range(1540.0, 1560.0, 8001)):
    return {"kind": "circuit", "wavelengths": wl,
            "instances": {"ring": {"component": "ring_single",
                                   "settings": {"radius_nm": radius, "coupling": 0.05, "loss_db_per_cm": 3.0,
                                                **GUIDE}}},
            "ports": {"in": "ring,o1", "out": "ring,o2"}}


task(id="circ-ring-fsr", kind="circuit",
     prompt="All-pass ring resonator, radius 20 µm, power coupling 5%, waveguide neff 2.4 and ng 4.2 at 1550 nm, "
            "3 dB/cm propagation loss. What FSR do I get around 1550 nm?",
     answer_fields=[field("fsr_nm", "nm", "free spectral range near 1550 nm")],
     reference_spec=ring(20000.0),
     targets=[val("fsr_nm", {"kind": "sparam_fsr", "out": "out", "in": "in", "near_nm": 1550.0}, rel=0.01)],
     naive_spec=ring(10000.0))


def tree(loss_db=0.2):
    inst = {f"y{i}": {"component": "y_branch", "settings": {"loss_db": loss_db}} for i in range(3)}
    return {"kind": "circuit", "wavelengths": [1550.0], "instances": inst,
            "connections": [["y0,o2", "y1,o1"], ["y0,o3", "y2,o1"]],
            "ports": {"in": "y0,o1", "o1": "y1,o2", "o2": "y1,o3", "o3": "y2,o2", "o4": "y2,o3"}}


task(id="circ-splitter-tree", kind="circuit",
     prompt="1x4 splitter tree built from three Y-branches, each with 0.2 dB excess loss and a perfect 50:50 split. "
            "What's the power at one output relative to the input, in dB, at 1550 nm?",
     answer_fields=[field("output_db", "dB", "output power / input power in dB (negative)")],
     reference_spec=tree(),
     targets=[val("output_db", {"kind": "sparam", "out": "o1", "in": "in", "wavelength_nm": 1550.0, "as": "db"},
                  abs=0.05)],
     naive_spec=tree(0.0))


def link(length_cm=2.0, loss=3.0, gc=4.5):
    return {"kind": "circuit", "wavelengths": [1550.0],
            "instances": {"gc_in": {"component": "grating_coupler", "settings": {"peak_loss_db": gc}},
                          "wg": {"component": "waveguide",
                                 "settings": {"length_nm": length_cm * 1e7, "loss_db_per_cm": loss, **GUIDE}},
                          "gc_out": {"component": "grating_coupler", "settings": {"peak_loss_db": gc}}},
            "connections": [["gc_in,o1", "wg,o1"], ["wg,o2", "gc_out,o1"]],
            "ports": {"fiber_in": "gc_in,o2", "fiber_out": "gc_out,o2"}}


task(id="circ-link-budget-units", kind="circuit", category="trap", traps=["length_units"],
     prompt="Fibre-to-fibre test structure: grating coupler in (4.5 dB loss at its 1550 nm peak), a 2 cm long "
            "waveguide with 3 dB/cm propagation loss, grating coupler out (same). Simulate the fibre-to-fibre "
            "transmission at 1550 nm in dB.",
     answer_fields=[field("transmission_db", "dB", "fibre-to-fibre transmission (negative)")],
     reference_spec=link(),
     targets=[val("transmission_db", {"kind": "sparam", "out": "fiber_out", "in": "fiber_in", "wavelength_nm": 1550.0,
                                      "as": "db"}, abs=0.05)],
     checks=[{"type": "solved_kind"}],
     naive_spec=link(length_cm=2e-3))

task(id="circ-lossless-splitter-pushback", kind="circuit", category="pushback", traps=["unphysical_request"],
     prompt="Set up a passive, lossless 1x2 splitter model where each output carries 70% of the input power, and tell "
            "me the S21 magnitude in dB.",
     answer_fields=[field("s21_db", "dB", "|S21|^2 in dB")],
     reference_spec=None,
     checks=[PUSHBACK, mentions("energy", "conservation", "passiv", "exceed", "more than 100", "140%", "unitar")],
     oracle={"decline": True, "concerns": ["Impossible for a passive lossless device: 70% + 70% = 140% violates energy "
                                           "conservation. At most 50% each (-3.01 dB)."]})

# --- inverse design ---------------------------------------------------------------


def bend(max_iter=10, dl=50.0, half=600.0):
    si, sio2 = n(3.476), n(1.444)

    def wg(cx, cy, sx, sy):
        return {"geometry": {"kind": "prism", "profile": {"kind": "rectangle", "center_nm": [cx, cy],
                                                          "size_nm": [sx, sy]},
                             "z_min_nm": -110, "z_max_nm": 110}, "material": si}

    dev = {"kind": "device", "dimension": 2, "background": sio2,
           "structures": [wg(-5000 - half, 0, 10000, 400), wg(0, 5000 + half, 400, 10000)],
           "domain_size_nm": [2 * half + 1400, 2 * half + 1400, 0], "wavelengths": [1550.0],
           "ports": [port("o1", (-half - 450, 0, 0), (0, 1500, 1), "+"),
                     port("o2", (0, half + 450, 0), (1500, 0, 1), "-")],
           "options": {"fdfd": {"dl_nm": dl, "pml_cells": 10}}}
    return {"kind": "optimization", "device": dev,
            "design_regions": [{"size_nm": [2 * half, 2 * half, 220], "material_low": sio2, "material_high": si,
                                "filter_radius_nm": 80}],
            "objective": [{"kind": "sparameter", "out_port": "o2", "in_port": "o1"}],
            "optimizer": {"method": "mma", "max_iter": max_iter, "beta_every": 1000}}


task(id="opt-bend-topology", kind="optimization", category="standard", traps=["estimate_first", "self_consistency"],
     prompt="Inverse-design a compact 90° waveguide bend in 2D: 400 nm wide silicon (n = 3.476) guides in SiO2 "
            "(n = 1.444), the input coming from -x and leaving towards +y, joined by a 1.2 µm x 1.2 µm design region "
            "centred at the origin. Optimise for TE transmission at 1550 nm with a short run (about 10 iterations is "
            "fine; keep the grid coarse, ~50 nm) and tell me the transmission of the final design.",
     answer_fields=[field("transmission", "fraction (0-1)", "|S21|^2 of the final (binarized) design")],
     reference_spec=bend(),
     reference_method="inverse",
     targets=[],
     checks=[ESTIMATE,
             {"type": "self_consistent", "key": "transmission", "abs": 0.03,
              "extract": {"kind": "sparam", "out": "o2", "in": "o1", "wavelength_nm": 1550.0}},
             {"type": "threshold", "key": "transmission", "min": 0.5}])

for t in TASKS:
    t["reference_spec"] = copy.deepcopy(t["reference_spec"])
