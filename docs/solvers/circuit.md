# Circuit solver

Backend `lightsmith-circuit` (method `circuit`, modules `lightsmith.backends.circuit` and
`lightsmith.circuits`). Solves the S-matrix of a [circuit netlist](../problems/circuit.md) of compact
models. It is differentiable and runs on a GPU.

## Method

- **`dense`**: S_ext = S_ee + S_ei·C·(I − S_ii·C)⁻¹·S_ie, in one batched solve over all wavelengths.
- **`reduce`**: pairwise merging with scikit-rf's `innerconnect` formula.
- **`auto`**: uses `dense` up to 300 connections and `reduce` beyond.

The solver uses the array API, so it runs on NumPy, PyTorch (CPU or CUDA) or CuPy. With PyTorch, any
setting passed as a tensor gets gradients:

```python
from lightsmith.circuits import circuit_sparameters

# problem: a CircuitProblem with an instance named "ring"; coupling: a torch tensor with requires_grad=True
solution = circuit_sparameters(problem, params={"ring": {"coupling": coupling}}, device="torch-cpu")
```

## Options

`options["circuit"]` or `options["lightsmith-circuit"]`.

| Option | Default | Meaning |
|---|---|---|
| `algorithm` | `"auto"` | `"dense"`, `"reduce"` or `"auto"` |
| `precision` | `"double"` | `"double"` or `"single"` |
| `unconnected` | `"error"` | `"ignore"` treats loose ports as matched ends, with a warning |
| `base_dir` | `None` | Directory for relative `sparams:` paths |
| `passivity_tol` | `1e-6` | Warn if the largest singular value of S exceeds 1 + tol |

## Compact models

`lightsmith.circuits.list_models()` returns each model's documentation and the JSON schema of its
settings. Aliases are in parentheses.

| Model | Ports | Main settings |
|---|---|---|
| `waveguide` (`straight`) | o1, o2 | `length_nm`, `neff`, `ng`, `wl0_nm`, `loss_db_per_cm`; optional tables of neff and loss against wavelength |
| `bend` (`bend_circular`, `bend_euler`) | o1, o2 | waveguide settings, plus `radius_nm`, `angle_deg`, `bend_loss_db` |
| `phase_shifter` (`thermo_optic_phase_shifter`, `electro_optic_phase_shifter`) | o1, o2 | waveguide settings, plus `mechanism`, `phase_per_volt`, `voltage`, `phase_offset_rad` |
| `coupler` (`directional_coupler`) | o1–o4 | `coupling` (power), `coupling_dispersion_per_nm`, `loss_db` |
| `mmi1x2`, `y_branch` (`ybranch`, `splitter_1x2`) | o1–o3 | `loss_db`, `imbalance_db`, `bandwidth_1db_nm` |
| `mmi2x2` | o1–o4 | `loss_db`, `imbalance_db`, `bandwidth_1db_nm` |
| `ring_single` (`ring_allpass`, `ring`), `racetrack_single` | o1, o2 | waveguide settings, plus `radius_nm`, `straight_length_nm`, `coupling` |
| `ring_double` (`ring_adddrop`), `racetrack_double` | o1 in, o2 through, o3 add, o4 drop | as above, plus `coupling_drop` |
| `grating_coupler` (`gc`) | o1, o2 | `peak_loss_db`, `wl0_nm`, `bandwidth_1db_nm`, optional reflections |
| `terminator` (`termination`, `absorber`) | o1 | optional `reflection_db` |
| `sparams:<path>` | from the file | a Touchstone `.sNp` file or a Lightsmith `SParameterResult` JSON. Settings: `port_names`, `time_convention`, `interpolation`, `out_of_range`. |
| `pdk:<cell>` | those of the cell | a simulated, fitted and cached component of the [PDK flow](../pdk.md). Settings: the cell's parameters plus the flow settings (technology, band, solver, fit). |

**Touchstone files.** `lightsmith.circuits.touchstone` reads and writes Touchstone 1.x and the common
Touchstone 2.0 keywords (full matrix format only).

**Modes to waveguides.** `waveguide_from_modes(mode_result, length_nm)` turns an [FDE](fde.md) mode
result into a `waveguide` instance.

## Validation

All tests are in `tests/test_circuit.py`.

| Test | Reference | Tolerance |
|---|---|---|
| `test_mzi_matches_analytic`: ΔL = 20 µm | closed-form MZI | amplitudes < 1e-12; FSR = λ²/(ng ΔL) to 1 % |
| `test_ring_netlist_and_model_match_allpass_formula` | analytic all-pass ring | netlist < 1e-10; model < 1e-12 |
| `test_ring_critical_coupling_extinction` | critical coupling | \|T\|² < 1e-20 on resonance |
| `test_unitarity_and_reciprocity`: lossless 6-port network with a feedback loop | S†S = I; S = Sᵀ | 1e-12 |
| `test_dense_and_reduce_agree_on_lossy_circuit` | the other algorithm | 1e-12 |
| `test_ring_double_drop_and_grating_passband` | analytic | 1e-12 |
| `test_gradient_matches_finite_difference_through_ring_resonance` | central differences | 1e-5 |
| `test_touchstone_round_trip` (1–4 ports) | round trip | 1e-11 |
| `test_gpu_matches_cpu` (gpu) | CPU | 1e-10 |

## Limits

- Raises `NetlistError` for:
  - malformed or unknown `"instance,port"` references;
  - self-connections and ports used twice;
  - unconnected ports (unless `unconnected="ignore"`);
  - unknown or invalid settings;
  - wavelengths outside the range of an S-parameter file.
- Warns about non-finite or non-passive S-matrices.
- Tabulated settings (neff and loss tables) are not differentiable.
