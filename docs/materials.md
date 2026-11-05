# Materials

You can use a material anywhere a problem asks for one: pass one of the models below. Permittivities
follow the `exp(-iωt)` convention, so loss means `Im(ε) > 0` and `Im(n) > 0`.

| Model | Fields | Saved as `kind` |
|---|---|---|
| `ConstantIndex` | `n_real`, `n_imag` (≥ 0) | `constant_index` |
| `ConstantPermittivity` | `eps_real`, `eps_imag` (≥ 0) | `constant_permittivity` |
| `Drude` | `eps_inf`, `wp_ev`, `gamma_ev`: ε(ω) = ε∞ − ωp² / (ω² + iγω), with energies in eV | `drude` |
| `DrudeLorentz` | `eps_inf`, `wp_ev`, `gamma_ev`, `poles: [{delta_eps, omega0_ev, gamma_ev}]` | `drude_lorentz` |
| `Tabulated` | `wavelength_nm`, `n_real`, `n_imag`, `source`: linear interpolation, no extrapolation. Data in any order is sorted by wavelength; duplicate wavelengths are refused. | `tabulated` |
| `LibraryMaterial` | `name`, `dataset`: a bundled measured dataset (below) | `library` |

## Bundled datasets

These datasets ship inside the package, so they work offline. They are verbatim copies of
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database) entries
(CC0). Names and datasets accept aliases, for example `"gold"` or `"JC"`. When `dataset` is omitted, the
material's default is filled in, so a stored spec stays reproducible.

| Material | Dataset | Range (nm) | Reference (DOI) |
|---|---|---|---|
| Au | Johnson-Christy (default) | 187.9–1937 | 10.1103/PhysRevB.6.4370 |
| Au | McPeak | 300–1700 | 10.1021/ph5004237 |
| Au | Rakic-LD | 247.97–6199.2 | 10.1364/AO.37.005271 |
| Ag | Johnson-Christy (default) | 187.9–1937 | 10.1103/PhysRevB.6.4370 |
| Ag | McPeak | 300–1700 | 10.1021/ph5004237 |
| Ag | Rakic-LD | 247.97–12398 | 10.1364/AO.37.005271 |
| Al | McPeak (default) | 150–1700 | 10.1021/ph5004237 |
| Al | Rakic-1995 | 0.124–200000 | 10.1364/AO.34.004755 |
| Al | Rakic-LD | 61.992–247970 | 10.1364/AO.37.005271 |
| Cu | Johnson-Christy (default) | 187.9–1937 | 10.1103/PhysRevB.6.4370 |
| Cu | McPeak | 300–1700 | 10.1021/ph5004237 |
| Cu | Rakic-LD | 206.64–12398 | 10.1364/AO.37.005271 |
| Si | Green-2008 (default) | 250–1450 | 10.1016/j.solmat.2008.06.009 |
| Si | Schinke | 250–1450 | 10.1063/1.4923379 |
| Si | Aspnes-Studna | 206.6–826.6 | 10.1103/PhysRevB.27.985 |
| TiO2 | Siefke (default) | 120.181–125123 | 10.1002/adom.201600250 |
| TiO2 | Sarkar | 300–1690 | 10.1021/acsami.8b20535 |
| TiO2 | Devore-o | 430–1530 | 10.1364/JOSA.41.000416 |
| SiO2 | Malitson (default) | 210–6700 | 10.1364/JOSA.55.001205 |
| SiO2 | Lemarchand | 250–2500 | 10.1364/OE.20.015734 |
| H2O | Hale-Querry (default) | 200–200000 | 10.1364/AO.12.000555 |
| H2O | Daimon-24C | 182–1129 | 10.1364/AO.46.003811 |

```python
import lightsmith as ls

ls.list_materials("Au")                      # datasets, ranges and references
gold = ls.LibraryMaterial(name="Au", dataset="McPeak")
tab = ls.load_material("Au")                 # the default dataset as a Tabulated material
```

If you use a dataset, cite its reference and the database: M. N. Polyanskiy, *Sci. Data* 11, 94 (2024).

## Other refractiveindex.info entries

`fetch_refractiveindex(shelf, book, page)` downloads any database entry. The names are the ones in the
refractiveindex.info URL, for example `?shelf=main&book=Au&page=Olmon-sc`. The entry is returned as a
`Tabulated` material and cached on disk, so later calls work offline. Tabulated n, k data and formula
types 1–9 are supported.

```python
olmon = ls.fetch_refractiveindex("main", "Au", "Olmon-sc")
```

## Drude–Lorentz fits for time-domain solvers

FDTD needs a material written as a sum of poles. `fit_drude_lorentz(material, wavelengths_nm)` fits a
passive `DrudeLorentz` model over a band.

- The fit weights the real and imaginary parts so that metal losses, which set plasmon linewidths, are
  fitted accurately.
- By default it picks the smallest number of poles that meets the tolerance.
- `fit_quality(model, material, wavelengths_nm)` reports the errors.

The [FDTD](solvers/fdtd.md) backend runs this fit automatically for tabulated and library materials, and
warns with the RMS error.

```python
model = ls.fit_drude_lorentz("Au", ls.WavelengthRange(start_nm=450, stop_nm=900, num=91).values())
ls.fit_quality(model, "Au", [500, 600, 700])
```
