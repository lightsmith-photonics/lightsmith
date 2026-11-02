"""Options for the native FDTD backend (``Simulation.options["fdtd"]``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FDTDOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spacing_nm: float | None = Field(
        None, gt=0, description="Yee cell size. Overrides cells_per_wavelength and cells_per_radius.")
    cells_per_wavelength: float = Field(
        20.0, ge=6, description="Cells per shortest wavelength inside the optically densest material (|n|).")
    cells_per_radius: float | None = Field(
        None, ge=2, description="Minimum cells across the smallest particle half-extent. Default 10, "
        "or 20 when a material has Re(eps) < 0 in the band (plasmonic; ~10% peak-height error at 20).")
    pml_cells: int = Field(10, ge=4, description="CPML thickness in cells.")
    pml_kappa: float = Field(1.0, ge=1, description="CPML kappa_max (>1 damps evanescent fields faster).")
    padding_nm: float | None = Field(
        None, ge=0, description="Gap between the particles' bounding box (periodic: the stack) and the TFSF "
        "box (plane). Default 4 cells, and for scattering at least 3% of the longest wavelength in the medium.")
    pml_gap_nm: float | None = Field(
        None, ge=0, description="Gap between the flux monitors and the CPML. Scattering: 4 cells, at least 3% of the longest "
        "wavelength in the medium; "
        "periodic: 3 decay lengths of the slowest evanescent diffraction order, 0.25-1.5 wavelengths.")
    courant: float = Field(0.95, gt=0, le=1, description="Time step as a fraction of the 3D Courant limit.")
    run_time_fs: float | None = Field(
        None, gt=0, description="Fixed simulated time in fs. Disables the decay shutoff.")
    decay_by: float = Field(
        1e-5, gt=0, lt=1, description="Stop when the field energy on the flux monitors has fallen below "
        "this fraction of its peak (after the pulse has passed).")
    max_time_fs: float = Field(
        2000.0, gt=0, description="Hard cap on simulated time in fs (high-Q resonances need ps).")
    precision: Literal["auto", "single", "double"] = Field("auto", description="auto = single on GPU, double on CPU.")
    subpixel: bool = Field(True, description="Anisotropic subpixel smoothing at interfaces (dispersive too). "
                           "False means staircase. Superseded by mesh_refinement when that is set.")
    mesh_refinement: Literal["staircase", "volume_average", "subpixel", "conformal"] | None = Field(
        None, description="Interface treatment. staircase: the medium at each field position. volume_average: "
        "<eps> over the cell. subpixel: diagonal anisotropic averaging (Farjadpour 2006). conformal: the full "
        "tensor with off-diagonal terms (Kottke 2008); at interfaces with a dispersive medium (metals) it is "
        "applied at grid vertices as a sum of passive congruences. Default: conformal, or "
        "staircase when subpixel is False.")
    subpixel_samples: int = Field(8, ge=2, le=24, description="Supersamples per axis for interface cells.")
    dft_stride: int | None = Field(
        None, ge=1, description="Time steps between DFT samples. Default: from the source bandwidth (aliasing-safe).")
    cuda_graphs: bool = Field(True, description="Replay time steps as CUDA graphs on PyTorch CUDA (much faster).")
    max_cells: int = Field(300_000_000, ge=1, description="Refuse grids larger than this.")
