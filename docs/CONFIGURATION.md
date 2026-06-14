# Configuration

Supported run files use schema version 2. Legacy `run.yaml` files are intentionally unsupported.

## Case Shape

Minimum top-level sections:

```yaml
case:
  name: smoke_swarm
  schema_version: 2

files:
  chamber: chamber.yaml
  recipe: recipe_smoke.yaml
  chemistry:
    manifest: ../chemistry/chemistry_manifest.yaml
  output_dir: ../outputs/smoke_swarm
```

Common optional sections:

- `physics`: backend and model switches.
- `numerics`: tolerances, step limits, positivity floors.
- `outputs`: output formats and plots.
- `swarm`: EEDF/swarm backend configuration.

## EEDF and Rate Tables

Select the EEDF backend through `physics.eedf_backend`. The compact analytic
and internal backends are useful for smoke tests and development; externally
generated tables are recommended when electron kinetics accuracy matters.

Current `rate_table` configuration:

```yaml
physics:
  eedf_backend: rate_table

swarm:
  closure: local_field
  table:
    file: tables/example_rates.h5
    lookup: field
    electron_energy_mode: table_relaxation
    energy_relaxation_time_s: 1.0e-6
```

`swarm.table.file` is required for `rate_table`. Relative paths are resolved
under the resolved chemistry directory. `swarm.table.lookup` is optional and
may be `mean_energy` or `field`; when omitted, field-gridded tables use field
lookup when a reduced field is available. The shared `swarm.closure` setting
supports `auto`, `mean_energy`, and `local_field`.

For field lookup, `electron_energy_mode: table_relaxation` is optional. It
relaxes the electron-energy state toward the table mean energy using
`energy_relaxation_time_s`. Use it as a prescribed table closure, not as a
detailed power-balance diagnostic.

The currently supported HDF5 layout is documented in [External Swarm and Rate
Tables](SWARM_RATE_TABLES.md). Interchange metadata guidance on that page is
for table producers, not additional current configuration keys.

## External Circuit Tables

Use `external_circuit_table` for one-way coupling from measured or
SPICE-generated CSV waveform data. The backend reads the table during the run;
it does not invoke ngspice or perform bidirectional co-simulation.

```yaml
physics:
  electrical_backend: external_circuit_table

files:
  external_inputs:
    circuit_result_csv: ../external/circuit_waveform.csv

# In the recipe step:
power_ports:
  wafer_bias:
    file_key: circuit_result_csv
    interpolation: linear
    hold: edge
    power_scale: 1.0
    voltage_scale: 1.0
    current_scale: 1.0
```

The CSV file must include `time_s` and either a power column or both voltage
and current columns:

| Purpose | Supported columns |
|---|---|
| absorbed power | `absorbed_power_W`, `power_W`, `plasma_power_W`, `P_abs_W` |
| voltage | `voltage_V`, `gap_voltage_V`, `plasma_voltage_V` |
| current | `current_A`, `plasma_current_A` |
| reduced field, optional | `reduced_field_Td`, `EoverN_Td`, `E_over_N_Td` |

Column names can be overridden with `power_column`, `voltage_column`,
`current_column`, and `reduced_field_column`. `file` or `csv_file` may be used
instead of `file_key`; relative file paths are resolved from the case file
directory. `interpolation` defaults to linear, while `previous`, `zoh`, and
`zero_order_hold` select zero-order hold. Outside the table range, the default
policy holds the edge value; use `hold: error` to reject out-of-range times.
If no reduced-field column is present, `gap_m` plus voltage can derive E/N.
Configured `total_density_m3` is used first; otherwise the current zone total
density from the plasma state is used when available.

Run summaries include `electrical_coupling.circuit_interface.table_sources`
with the resolved file path, time range, selected columns, interpolation mode,
hold policy, and power source.

## Wall-Loss Models

Ion wall loss is configured on chamber surfaces through `surface.models`.
Omitting `ion_loss` keeps the default Bohm-like global loss behavior.

Preferred fixed-frequency configuration:

```yaml
models:
  ion_loss: prescribed_loss_frequency
  loss_rate_s: 3230.0
```

Implemented ion-loss families:

| `ion_loss` value | Meaning |
|---|---|
| omitted, `bohm`, `bohm_edge_loss`, `bohm_global_loss` | Bohm-like global ion loss using active wall area, zone volume, ion sound speed, and an edge-to-center `h_factor` |
| `prescribed_loss_frequency`, `loss_frequency`, `global_loss_frequency` | Fixed global ion-loss frequency from `loss_rate_s` or `ion_loss_frequency_s` |
| `ambipolar_diffusion` | Backward-compatible effective-frequency mode using `loss_rate_s`, `ambipolar_loss_rate_s`, or `diffusion_coefficient_m2_s / diffusion_length_m^2` |
| `disabled`, `off`, `none` | No ion wall loss on that surface |

Do not mix Bohm-like and effective-frequency ion-loss families within the same
zone. The validator rejects mixed families to avoid double counting.
Surface ion-flux observables and ion-enhanced rates use explicit IED flux when
available; otherwise they use the same zone wall-loss flux used by the global
RHS.

Bohm-like surfaces may also provide:

```yaml
models:
  ion_loss: bohm_global_loss
  h_factor: auto
  ion_neutral_cross_section_m2: 1.0e-18
  characteristic_length_m: 0.05
```

For calibrated electronegative or strongly nonlocal cases, use a calibrated
`h_factor` or `prescribed_loss_frequency` for now. Detailed RF sheath,
spatial diffusion, PIC/fluid coupling, and feature-scale wall models are
outside the core scope; see the [Extension Guide](EXTENSION_GUIDE.md).

## Numerical Diagnostics and Events

Projection and solver diagnostics are written to `summary.yaml` automatically.
Large projection counts or large projection magnitudes indicate that a run is
leaning on numerical clipping and should be reviewed before interpreting the
physics.

Initial states include small charged-species seeds for numerical robustness.
For quantitative ignition or early-transient studies, set species-specific
`initial_densities_m3` in the chamber zone.

The steady-state solver event is optional and disabled unless explicitly
enabled:

```yaml
numerics:
  events:
    steady_state:
      enabled: true
      relative_rhs_norm_s_inv: 1.0e-3
      min_step_time_s: 0.0
```

`relative_rhs_norm_s_inv` is the threshold for
`max(abs(dy_i/dt) / scale_i)`. `min_step_time_s` prevents immediate event
termination at the start of each recipe step.

## Includes

Cases may use `include` to share defaults:

```yaml
include: base_case.yaml
```

Included files are deep-merged, and the including file wins on conflicts.

## Generated Config Snapshots

Runs normally write:

- `effective_case.yaml`
- `resolved_paths.yaml`

These are review artifacts. Regenerate them instead of committing generated outputs.
