"""Engine of the native RCWA backend (see ``omniscatter.backends.rcwa``)."""

from .core import Core, PatternedMatrices
from .fourier import Lattice, form_factors, harmonic_set, normal_field_coefficients, pixel_coefficients, toeplitz_index
from .linalg import eig, framework, is_tensor, sqrt_exp_functions

__all__ = ["Core", "Lattice", "PatternedMatrices", "eig", "form_factors", "framework", "harmonic_set", "is_tensor",
           "normal_field_coefficients", "pixel_coefficients", "sqrt_exp_functions", "toeplitz_index"]
