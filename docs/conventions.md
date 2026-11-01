# Conventions

Every backend follows these conventions, so their results can be compared directly. Changing any of them
requires a schema version bump.

## Units

- **Lengths in nanometres**, including every `*_nm` field.
- **Wavelengths are vacuum wavelengths**, in nm, even when the host medium has an index above 1.
- **Cross sections in nm²**; differential cross sections in nm²/sr.
- Mode results are the exception: effective areas are in µm² and losses in dB/cm, as their field names say.
- Band frequencies are normalised, as `a/λ`, where `a = |lattice_vectors_nm[0]|`.

## Time convention

Fields vary as **`exp(-iωt)`**. Consequences:

- Absorbing materials have `Im(ε) > 0` and `Im(n) > 0`.
- A wave travelling along +z is `exp(+ikz)`.
- Backends built on libraries that use `exp(+iωt)` conjugate at their boundary. One example is
  miepython, which uses `n − ik`. Another is Touchstone files, which are conjugated when loaded.

## Normalisation of cross sections

Efficiencies use the **volume-equivalent radius** of all the particles together:

```
Q = C / (π r_eq²),   r_eq = (3 V_total / 4π)^(1/3)
```

`Result.equivalent_radius_nm` stores `r_eq`. Because of this, `Q` values are comparable across shapes and
methods. For a single sphere, `r_eq` is the sphere's radius, which gives the familiar Mie efficiencies.

## Geometry

**Scattering problems**
- Particles are centred at `position_nm`.
- Spheroids and cylinders have their symmetry axis along z.
- The default illumination is a plane wave travelling along +z, polarised along x.

**Layer stacks and periodic structures**
- Layers are stacked along z and listed from the superstrate side to the substrate side.
- Light comes from the semi-infinite superstrate and travels towards the substrate (+z).
- `R` is measured in the superstrate and `T` in the substrate.

**Modes**
- The waveguide cross-section lies in the xy plane, and propagation is along +z.
- Bends are about the y axis.

**Devices**
- Structures live in 3D, or in the xy plane when `dimension == 2`.
- Ports are planes. A port's normal is the axis along which its `size_nm` is zero.
- `direction` (`"+"` or `"-"`) is the direction of propagation *into* the device.

**Circuits**
- Port names follow gdsfactory: `o1`, `o2`, … counter-clockwise from the bottom left.
- Connections are written as `"instance,port"`.

## Options

Each backend reads its options from `options[<method>]` (for example `options["fdtd"]`), overlaid with
`options[<backend name>]` (for example `options["lightsmith-fdtd"]`). The backend's own pydantic model
validates them and rejects unknown keys. The [solver pages](solvers/index.md) list every option.

## Provenance and warnings

Every result carries:

- `provenance`: backend and dependency versions, device, precision, grid sizes, iteration counts,
  residuals and run time;
- `warnings`: problems that would affect the numbers, such as unconverged iterations, fitted material
  models, energy-balance violations or approximations outside their range of validity.

A solver that cannot handle a problem refuses it before running, and says why and what to change.
