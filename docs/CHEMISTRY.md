# Chemistry

Chemistry is loaded from an explicit manifest.

## Manifest Inputs

A chemistry manifest points to:

- species CSV
- gas reactions CSV
- surface reactions CSV
- rate model YAML files
- cross-section manifest
- optional extension YAML files for extra state variables and compact process terms

Optional extension files are declared under `extensions`:

```yaml
extensions:
  state_variables: state_variables.yaml
  processes: processes.yaml
```

Existing manifests that omit `extensions` remain valid.

## Validation

The mechanism validator checks:

- species and alias collisions
- missing or invalid cross sections
- unknown reaction species or rate models
- charge conservation
- element conservation
- surface site conservation
- duplicate reaction signatures
- extra state scopes, owners, bounds, and process targets

## Extra States and Processes

Use reactions for stoichiometric chemistry. Use processes only for compact RHS
terms that are not naturally represented as a reaction, such as empirical
sources, relaxation terms, or wall-charge accumulation.

`state_variables.yaml` supports `zone` and `surface` state variables:

```yaml
state_variables:
  wall_charge:
    scope: surface
    unit: C_m2
    initial: 0.0
    lower_bound: null
    scale: 1.0e-3
    output: true
    surfaces: [wafer]
```

`processes.yaml` supports three small process kinds:

```yaml
processes:
  constant_source:
    kind: source
    target: some_zone_state
    value: 1.0
    zones: [plasma]

  relaxation:
    kind: relaxation
    target: some_zone_state
    tau_s: 1.0e-4
    equilibrium: 0.0

  wall_charge_from_flux:
    kind: ion_flux_source
    target: wall_charge
    coefficient: 1.602176634e-19
    surfaces: [wafer]
```

Units are SI metadata in this version; no automatic unit conversion is
performed. Arbitrary Python expressions and runtime external database access are
not part of the mechanism format. Convert external databases into the standard
CSV/YAML mechanism files before running cases.

## Reaction Data Provenance

Reaction and cross-section data should be easy to inspect and trace. Where the
input format has metadata fields, prefer recording optional provenance such as:

- `source`
- `reference`
- `version`
- `units`
- `valid_temperature_range`
- `valid_pressure_range`
- `uncertainty`
- `notes`
- `cross_section_id` for electron-impact reactions

These fields are recommendations for reviewability and reproducibility; they
are not mandatory unless a specific parser or validator already requires them.
Avoid hiding rate origins in generated tables or scripts.

Gas and surface reaction CSV files may include any of the optional provenance
columns above. Existing files that omit them remain valid. Example:

```csv
reaction_id,phase,equation,rate_model_key,enabled,source,reference,units,cross_section_id
G_AR_ION,gas,"e + Ar -> Ar_plus + e + e",RM_E_AR_ION,true,LXCat,"example set",m3/s,xs_ar_ion
```

Rate model YAML may also carry a small `provenance` mapping:

```yaml
rate_models:
  RM_E_AR_ION:
    backend: electron_impact_xsec
    cross_section_id: xs_ar_ion
    provenance:
      source: LXCat
      reference: example set
      version: 2024
```

When available, `summary.yaml` records compact provenance counts under
`chemistry_provenance`. Missing optional provenance does not fail validation.

## Tabulated Rate Models

External databases are outside the runtime scope. Normalize database, literature,
or fitting results into this package's standard CSV/YAML inputs before running a
case.

Gas reactions can use a one-dimensional table when a rate coefficient is easier
to maintain as data than as a built-in formula:

```yaml
rate_models:
  RM_META_QUENCH:
    backend: tabulated_1d
    file: rates/meta_quench.csv
    x: gas_temperature_K
    x_column: Tg_K
    value_column: k_m3_s
    bounds_policy: clip
    interpolation: linear
```

Supported `x` values are `gas_temperature_K`, `mean_energy_eV`,
`reduced_field_Td`, and `pressure_Pa`. Tables must have a header, numeric
columns, a strictly increasing axis, and non-negative values. `bounds_policy`
is `clip` or `error`; interpolation is linear.

Surface reactions can use an ion-energy yield table:

```yaml
rate_models:
  RM_ION_ETCH:
    backend: ion_yield_table
    file: yields/ar_ion_sio2.csv
    energy_column: ion_energy_eV
    yield_column: yield
    bounds_policy: clip
    coverage_factor:
      kind: species_power
      species: wafer:F*
      exponent: 1.0
```

`ion_yield_table` reuses the existing ion flux and coverage-factor machinery.
Use it for externally prepared yield curves; keep `ion_assisted` for compact
threshold-style models. Arbitrary expression DSLs, reaction template engines,
and external database importers are not part of this core format.

## EEDF Swarm HDF5 Tables

The table swarm model is fail-fast. If `swarm.table.file` is missing,
does not exist, or lacks rate coefficients required by enabled
`electron_impact_xsec` gas reactions, case assembly fails. Use the `maxwell`
backend for cheap smoke/debug runs.

For production electron-impact kinetics, prefer externally generated
rate/transport tables from BOLSIG+, LoKI-B, Magboltz, or similar tools. The
current table format is described in [External Swarm and Rate
Tables](SWARM_RATE_TABLES.md).
