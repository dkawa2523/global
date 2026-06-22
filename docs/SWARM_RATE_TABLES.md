# External Swarm and Rate Tables

This package can use electron-impact rate coefficients and transport data from external swarm or Boltzmann solvers. This is the recommended production workflow when accurate electron kinetics are important.

## Recommended Workflow

```text
cross sections and gas mixture
  -> BOLSIG+ / LoKI-B / Magboltz / other swarm solver
  -> normalized rate and transport table
  -> rate_table backend
  -> 0D global plasma model
```

The global model should not depend directly on a specific external solver executable. External tools should be used offline first, and their results should be imported through a documented table format.

## Why Rate Tables?

Electron-impact reaction rates are sensitive to the electron energy distribution function. Specialized swarm solvers are better suited for generating electron transport and reaction data than a compact global-model package.

The role of this package is to combine those data with:

- gas chemistry
- wall losses
- surface reactions
- reactor residence time
- absorbed power
- electrical coupling
- recipe time evolution
- observables and validation output

## Backend Roles

| Backend | Intended use |
|---|---|
| `maxwell` | Simple estimates, smoke tests, educational cases |
| `boltzmann_2term` | Internal approximate kinetic backend; useful for development and comparison, not a full replacement for mature swarm solvers |
| `rate_table` | Recommended production path for externally generated rates and transport data |
| external adapter | Optional online coupling layer, if needed later |

## Current `rate_table` Contract

The current backend reads an HDF5 table from `swarm.table.file`. Relative table paths are resolved under the resolved chemistry directory.

Supported HDF5 contents are:

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

The supported table grid is either `mean_energy_eV` or a field grid identified by `EoverN_Td` / `effective_field_Td` in the converter input and stored as `effective_field_Td` in HDF5. The backend clips lookup coordinates to the available table range, so production cases should validate that operating points remain inside the intended range.

At runtime, all EEDF backends expose transport through a typed
`EEDFTransport` object with `mean_energy_eV`, `mobility_m2_V_s`,
`diffusion_m2_s`, `effective_field_Td`, and `lookup_mode`. Backend names and
table metadata are not part of that runtime transport contract.

During case assembly, the backend checks that enabled gas-phase
`electron_impact_xsec` reactions have matching `rate_coefficients/<cross_section_id>`
datasets. Missing required rates fail fast instead of silently falling back to
zero or Maxwellian estimates.

## YAML Configuration

Use the existing `rate_table` backend configuration supported by this repository:

```yaml
physics:
  eedf_backend: rate_table

swarm:
  closure: local_field
  table:
    file: tables/example_rates.h5
    lookup: field
```

`swarm.table.file` is required for `rate_table`. `swarm.table.lookup` may be set to `mean_energy` or `field`; when omitted, the backend chooses field lookup for field-gridded tables when a reduced field is available. The shared `swarm.closure` setting supports `auto`, `mean_energy`, and `local_field`.

No current configuration option selects an extrapolation policy. Do not rely on out-of-range behavior for production studies.

## Minimum Production Data

A production rate table should include:

| Data | Purpose |
|---|---|
| mean electron energy or E/N grid | interpolation coordinate |
| electron-impact rate coefficients | gas reaction source terms |
| mobility | charged particle transport closure when needed |
| diffusion | diffusion or wall-loss closure when needed |
| metadata | reproducibility and validation |

Power-loss channels are useful for diagnostics and interchange-table work, but they are not part of the current `rate_table` backend contract.

## Interchange Schema Guidance

This section describes a richer interchange schema for externally generated rate and transport tables. It is not the current runtime HDF5 layout or a current configuration contract.

```text
/
  attrs:
    schema_version
    generator
    generator_version
    source_solver
    source_solver_version
    cross_section_set
    gas_mixture
    gas_temperature_K
    pressure_Pa
    created_at
    notes

  grid/
    mean_energy_eV
    reduced_field_Td

  rate_coefficients/
    <reaction_or_cross_section_id>

  transport/
    mobility_m2_V_s
    diffusion_m2_s

  power_loss/
    <channel_id>
```

If both `mean_energy_eV` and `reduced_field_Td` are present in an interchange table, the conversion or backend configuration should explicitly choose the interpolation coordinate.

## Required Metadata

Every externally generated table should identify:

- source solver
- solver version
- cross section source
- gas mixture
- grid range
- interpolation coordinate
- units
- generation script or command
- creation date

This information should be preserved near the generated table and copied to simulation output when supported.

## Validation Checklist

Before using a table in production:

- Check that all required electron-impact reactions are covered.
- Check units.
- Check interpolation range.
- Check monotonicity and finite values.
- Compare selected rates with the source solver output.
- Run at least one smoke global-model case.
- Save the effective case and resolved paths.

## Non-Goals

The rate table backend should not:

- call external executables directly during RHS evaluation
- silently replace missing rates with Maxwellian estimates
- depend on a specific external solver package
- hide the source of rate coefficients
- become a full electron Boltzmann or Monte Carlo solver
