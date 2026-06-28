# External Swarm and Rate Tables

The `rate_table` EEDF backend reads prepared electron-impact rate coefficients
and transport data. Use external swarm or Boltzmann tools offline, then import
their results through the HDF5 table format below.

```text
cross sections and gas mixture
  -> external swarm solver
  -> HDF5 rate/transport table
  -> rate_table backend
  -> global model
```

The global model does not call external swarm executables during RHS evaluation.

## Backend Roles

| Backend | Intended use |
|---|---|
| `maxwell` | Simple estimates and smoke cases |
| `boltzmann_2term` | Internal approximate kinetic backend |
| `swarm` | Replaceable internal wrapper |
| `rate_table` | Prepared external rate and transport data |

## Current HDF5 Layout

The backend reads `swarm.table.file`. Relative paths are resolved under the
resolved chemistry directory.

```text
/
  attrs:
    grid_column              # optional; defaults to mean_energy_eV

  mean_energy_eV
  effective_field_Td
  mobility_m2_V_s
  diffusion_m2_s

  rate_coefficients/
    <cross_section_id>
```

The interpolation coordinate is either mean energy or reduced field. Field
tables use `effective_field_Td` in HDF5. The backend clips lookup coordinates to
the available table range; production studies should check that operating
points remain inside the intended range.

During case assembly, enabled gas-phase `electron_impact_xsec` reactions must
have matching `rate_coefficients/<cross_section_id>` datasets. Missing required
rates fail fast.

## YAML Configuration

```yaml
physics:
  eedf_backend: rate_table

swarm:
  closure: local_field
  table:
    file: tables/example_rates.h5
    lookup: field
```

`swarm.table.file` is required for `rate_table`. `swarm.table.lookup` may be
`mean_energy` or `field`. When omitted, the backend chooses field lookup for
field-gridded tables when a reduced field is available. `swarm.closure`
supports `auto`, `mean_energy`, and `local_field`.

No current configuration option selects an extrapolation policy.

## Minimum Data

A useful table should include:

- interpolation grid: mean energy or reduced field
- electron-impact rate coefficients used by the mechanism
- mobility
- diffusion
- enough nearby metadata to reproduce how the table was generated

The backend keeps only compact table provenance in workflow return values.
HDF5 attributes are not copied into normal summaries.

## Non-Goals

The `rate_table` backend should not:

- call external executables from the RHS
- silently replace missing rates with Maxwellian estimates
- depend on a specific external solver package
- become a full Boltzmann or Monte Carlo solver
