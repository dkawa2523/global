# Chemistry

Chemistry is loaded from an explicit manifest or a compatible chemistry directory.

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

## Rate Tables

The `rate_table` EEDF backend is fail-fast. If `swarm.table.file` is missing or does not exist, case assembly fails. Use the `maxwell` backend for cheap smoke/debug runs.

