# Input / output specification

## Primary inputs

### Case YAML
Controls paths, physics selection, numerics, outputs, and runtime behavior.

### Chamber YAML
Defines zones, edges, surfaces, inlets, pumps, and power ports.

### Recipe YAML
Defines time-ordered recipe steps.

### Chemistry manifest YAML
Defines the chemistry file bundle.

## Primary outputs

### `solution.h5`
Raw state history and diagnostics.

### `observables.csv`
Engineering- and physics-oriented derived quantities versus time.
Power-port diagnostics are flattened when an electrical backend exposes numeric
`PowerResult.metadata["port_details"]` entries. The column pattern is:

```text
port_<port_id>_<metric>
```

Examples include `port_source_rf_absorbed_power_W`,
`port_dc_drive_gap_voltage_V`, `port_dc_drive_current_A`, and
`port_lf_bias_self_bias_V`. Text metadata such as backend names and file paths
is intentionally not expanded into the main time-series table.

### `summary.yaml`
Compact end-of-run summary.
It includes final and per-step statistics for the core plasma observables and
for a small whitelist of power-port diagnostics: absorbed/delivered power,
frequency, voltage, current, self-bias, plasma potential, coupling efficiency,
reduced field, and plasma resistance. More specialized backend diagnostics
remain in `observables.csv` to keep the summary compact.

### `reaction_budget.yaml`
Optional gas-phase reaction and ion wall-loss budget at the final saved state.
Enable it per case:

```yaml
outputs:
  diagnostics:
    reaction_budget:
      enabled: true
      species: [Ar_star, Ar_plus, Ar2_plus]
      max_reactions_per_species: 6
```

The file reports, by zone and species, total production, total loss, net rate,
and the largest production/loss contributors. It is intended for mechanism
review, not for dense time-series output.

### `electron_energy_budget.yaml`
Optional electron energy-density budget at the final saved state. Enable it per
case:

```yaml
outputs:
  diagnostics:
    electron_energy_budget:
      enabled: true
      max_reactions: 8
```

The file reports absorbed power density, electron-impact energy losses from
`constant_event_loss` models, positive-ion wall energy losses, pump/edge
electron-energy transport, and the dominant loss reactions. The sign convention
is explicit in the file: positive terms add electron energy density and
negative terms remove it.

### `surface_reaction_budget.yaml`
Optional surface reaction, gas-flux, coverage, and film-growth budget at the
final saved state. Enable it per case:

```yaml
outputs:
  diagnostics:
    surface_reaction_budget:
      enabled: true
      max_reactions_per_species: 6
```

The file reports, by surface, surface species coverage production/loss, gas
fluxes to/from the surface, film growth rate, and per-reaction surface rates.
Cases without surface reactions still produce a small surface inventory summary.

### `state_manifest.yaml`
Optional manifest of the actual ODE state vector:

```yaml
outputs:
  diagnostics:
    state_manifest:
      enabled: true
```

The file maps each state-vector index to its label, unit, zone/surface, and
species metadata where applicable. It is generated from the loaded chemistry,
chamber, and physics switches, so users do not have to maintain a duplicate
hand-written state list.

The manifest also includes:
- `species_catalog`, which reports whether each species is solved as a state,
  algebraic, prescribed externally, or metadata-only.
- `electron_density`, which reports the electron density closure. In the
  default quasi-neutral mode, `e` is not an ODE state. In
  `prescribed_profile` mode, `e` is still not an ODE state, but its density is
  read from the configured profile.
- `state_groups`, which records state-vector slice boundaries for downstream
  tools.

### `run_provenance.yaml`
Optional run-level provenance summary:

```yaml
outputs:
  diagnostics:
    provenance:
      enabled: true
      filename: run_provenance.yaml
```

The file records the resolved input paths, selected physics backends, EEDF/rate
table provenance, chemistry cross-section metadata, selected rate-model
metadata, chamber power-port settings, backend classes, and validation
messages. It is intended for review and reproducibility; `solution.h5` remains
the authoritative numerical output.

### `effective_case.yaml`
Resolved case configuration used for the run.

### `resolved_paths.yaml`
Absolute path map used for the run.

## Why both raw and derived outputs exist
The HDF5 state is the authoritative numerical result.
The CSV and YAML outputs are reviewer-facing summaries.
This split keeps debugging and process interpretation separate.
