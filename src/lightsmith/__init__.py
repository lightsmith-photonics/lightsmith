"""lightsmith: one problem spec, many electromagnetic scattering solvers.

Quick start::

    import lightsmith as ls

    sim = ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=40),
                                material=ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07))],
        medium_index=1.33,
        wavelengths=ls.WavelengthRange(start_nm=400, stop_nm=800, num=81),
    )
    ls.estimate(sim, method="dda")       # size/memory before running
    res = ls.solve(sim)                  # method="auto" -> Mie for a sphere
    cmp = ls.compare(sim, ["mie", "dda"])
"""

from .api import choose_method, compare, estimate, solve
from .backends import all_backends, get_backend
from .device import doctor, resolve_device
from .materials import (ConstantIndex, ConstantPermittivity, Drude, DrudeLorentz, LibraryMaterial, LorentzPole,
                        Tabulated, fetch_refractiveindex, list_materials, load_material)
from .results import FarFieldData, NearFieldData, Result, relative_difference
from .spec import (Cylinder, FarFieldRequest, LayeredSphere, NearFieldRequest, Outputs, Particle, PlaneWave, Shell,
                   Simulation, Sphere, Spheroid, Substrate, WavelengthRange, json_schema)

__version__ = "0.2.0"


def __getattr__(name):  # lazy: materials_fit pulls in scipy.optimize
    if name in ("fit_drude_lorentz", "fit_quality"):
        from . import materials_fit

        return getattr(materials_fit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ConstantIndex", "ConstantPermittivity", "Cylinder", "Drude", "DrudeLorentz", "FarFieldData", "FarFieldRequest",
    "LayeredSphere", "LibraryMaterial", "LorentzPole", "NearFieldData", "NearFieldRequest", "Outputs", "Particle",
    "PlaneWave", "Result", "Shell", "Simulation", "Sphere", "Spheroid", "Substrate", "Tabulated", "WavelengthRange",
    "all_backends", "choose_method", "compare", "doctor", "estimate", "fetch_refractiveindex", "fit_drude_lorentz",
    "fit_quality", "get_backend",
    "json_schema", "list_materials", "load_material", "relative_difference", "resolve_device", "solve",
]
