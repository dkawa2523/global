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

