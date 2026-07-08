# Configuration

Run files use schema version 2 only.

## Minimal Case

```yaml
case:
  name: smoke_maxwell
  schema_version: 2

files:
  chamber: chamber.yaml
  recipe: recipe_smoke.yaml
  chemistry:
    manifest: ../chemistry/chemistry_manifest.yaml
  output_dir: ../outputs/smoke_maxwell
```

Relative paths are resolved from the case file location. `include` or
`includes` may be used to deep-merge shared defaults; the including file wins.

## Top-Level Sections

- `case`: `name`, optional `description`, `tags`, `schema_version`.
- `files`: `chamber`, `recipe`, `chemistry.manifest`, `output_dir`, optional
  `external_inputs`.
- `runtime`: `export_effective_config`, `export_resolved_paths`.
- `physics`: backend and model switches.
- `numerics`: tolerances, step limits, positivity, optional events.
- `outputs`: formats, plots, optional detailed reaction budgets.
- `swarm`: EEDF/swarm backend settings.

Unknown keys are rejected for the strict sections handled by the loader and
validators.

## Backends

Common selections:

```yaml
physics:
  eedf_backend: maxwell        # maxwell or swarm
  electrical_backend: icp      # direct_power, icp, ccp, rf_envelope,
                               # dc_series_circuit, external_circuit_table
  integrator: scipy_bdf
```

Use `python -m plasma_global.cli list-backends` for the current registry.
Use the `swarm` backend when a prepared table is available or when explicitly
running the experimental `boltzmann_2term` closure.

## Numerics

```yaml
numerics:
  rtol: 1.0e-6
  atol: 1.0e-14
  first_step: 1.0e-10
  max_step: 1.0e-6
  events:
    steady_state:
      enabled: false
```

`atol` is configured as a scalar. The BDF backend receives a state-sized
tolerance vector generated from that scalar and small per-state floors.

Recipe steps must be contiguous in time. Use an explicit zero-power or idle step
when a hold period is intended.

## Outputs

```yaml
runtime:
  export_effective_config: true
  export_resolved_paths: true

outputs:
  formats:
    solution_h5: true
    observables_csv: true
    summary_yaml: true
  budgets:
    enabled: false
```

Normal output files are `summary.yaml`, `observables.csv`, `solution.h5`,
`effective_case.yaml`, and `resolved_paths.yaml`, depending on the switches
above.

## EEDF Swarm HDF5 Tables

```yaml
physics:
  eedf_backend: swarm

swarm:
  model_name: table
  closure: local_field
  table:
    file: tables/example_rates.h5
    lookup: field
    bounds_policy: clip
    electron_energy_mode: table_relaxation
    energy_relaxation_time_s: 1.0e-6
```

`swarm.table.file` is required when `swarm.model_name: table`. `lookup` may be
`mean_energy` or `field`. `bounds_policy` may be `clip` or `error`; default is
`clip`. The HDF5 layout is described in [Swarm Rate Tables](SWARM_RATE_TABLES.md).

## External Circuit Tables

`external_circuit_table` reads a prepared CSV during the run; it does not run a
circuit executable.

```yaml
physics:
  electrical_backend: external_circuit_table

files:
  external_inputs:
    circuit_result_csv: ../external/circuit_waveform.csv

# In a recipe step:
power_ports:
  powered_port:
    file_key: circuit_result_csv
    interpolation: linear
    hold: edge
```

The CSV must include `time_s` and either `absorbed_power_W` or both `voltage_V`
and `current_A`. `reduced_field_Td` is optional.

## Chamber Flow

`edges` are directed constant-conductance links. Gas and electron energy are
transported from `from_zone` to `to_zone`; the model does not solve pressure
differences or bidirectional flow through an aperture.

`pumps` use fixed `speed_m3_s` removal. Pressure targets are not accepted as
chamber inputs; prescribe the pump speed and inlet flows that define the desired
reduced global-model operating point.

## Wall Loss

Ion wall loss is configured on chamber surfaces:

```yaml
models:
  ion_loss: prescribed_loss_frequency
  frequency_s: 3230.0
```

Supported `ion_loss` values:

- omitted or `bohm`
- `prescribed_loss_frequency`
- `ambipolar_diffusion`
- `off`

Do not mix Bohm-like and effective-frequency ion-loss families in one zone.
