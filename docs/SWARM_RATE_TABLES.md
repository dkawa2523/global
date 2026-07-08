# Swarm Rate Tables

The `swarm` EEDF backend with `swarm.model_name: table` reads prepared electron-impact rates and
transport data from HDF5. External swarm/Boltzmann tools should run offline; the
global model only consumes the generated table.

## YAML

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
```

`swarm.table.file` is required for `swarm.model_name: table`. Relative paths are resolved under the chemistry
directory. `lookup` may be `mean_energy` or `field`; when omitted, field tables
use field lookup when reduced field is available. `bounds_policy` defaults to
`clip`; set it to `error` to fail on out-of-range lookup.

## HDF5 Layout

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

All one-dimensional datasets must have the same length. `grid_column` may be
`mean_energy_eV` or `effective_field_Td`. Enabled gas-phase
`electron_impact_xsec` reactions must have matching
`rate_coefficients/<cross_section_id>` datasets.

## Bounds Policy

- `clip`: clamp lookup coordinates to the table range. This is the current
  default behavior.
- `error`: raise `ValueError` when lookup is outside the table range.

The error path reports the zone, lookup mode, value, axis bounds, and table
path.

## Electron Energy Relaxation

For field lookup, `electron_energy_mode: table_relaxation` relaxes the
electron-energy state toward the table mean energy:

```yaml
swarm:
  table:
    electron_energy_mode: table_relaxation
    energy_relaxation_time_s: 1.0e-6
```

Treat this as a prescribed table closure, not a detailed electron power-balance
model. In zones using field-table relaxation, reaction losses, wall-loss
electron-energy losses, flow energy transport, and absorbed-power electron
heating are not also applied to the electron-energy RHS. Species, surface, and
gas-temperature equations still use their configured source terms.
