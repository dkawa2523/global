# Chemistry

Chemistry is loaded from an explicit manifest.

## Manifest Inputs

A chemistry manifest points to:

- species CSV
- gas reactions CSV
- surface reactions CSV
- rate model YAML files
- aliases YAML
- cross-section manifest

## Validation

The mechanism validator checks:

- species and alias collisions
- missing or invalid cross sections
- unknown reaction species or rate models
- charge conservation
- element conservation
- surface site conservation
- duplicate reaction signatures

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

## Rate Tables

The `rate_table` EEDF backend is fail-fast. If `swarm.table.file` is missing,
does not exist, or lacks rate coefficients required by enabled
`electron_impact_xsec` gas reactions, case assembly fails. Use the `maxwell`
backend for cheap smoke/debug runs.

For production electron-impact kinetics, prefer externally generated
rate/transport tables from BOLSIG+, LoKI-B, Magboltz, or similar tools. The
current table format is described in [External Swarm and Rate
Tables](SWARM_RATE_TABLES.md).
