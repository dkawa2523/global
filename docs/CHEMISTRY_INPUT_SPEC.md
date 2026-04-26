# Chemistry input specification

## Philosophy
Chemistry is intentionally separated from the main case configuration so that process engineers and chemistry developers can edit reaction sets without touching solver settings.

## File set

### `chemistry_manifest.yaml`
Top-level index of chemistry files.

### `species.csv`
One row per species.
Main fields:
- `canonical_id`
- `phase`
- `charge`
- `mass_amu`
- `elements`
- `aliases`
- `state_tags`
- `zones`
- `surfaces`

### `gas_reactions.csv`
One row per gas-phase reaction.

### `surface_reactions.csv`
One row per surface reaction.

### `reaction_models.yaml`
Detailed rate-law and energy-loss models.

This legacy single-file form is still supported for existing chemistry
bundles. New chemistry bundles should prefer the split `model_files` form below
so that cross-section-driven electron-impact rates are not mixed with
empirical gas-rate expressions.

### Split model files
`chemistry_manifest.yaml` can declare separate model files:

```yaml
model_files:
  electron_impact: electron_impact_models.yaml
  gas_rate: gas_rate_models.yaml
  energy_loss: energy_loss_models.yaml
```

Supported backends by file category:

| Category | Intended content | Allowed backends |
|---|---|---|
| `electron_impact` | rates evaluated from tabulated electron-collision cross sections | `electron_impact_xsec` |
| `gas_rate` | formula rates without cross-section data | `constant`, `first_order_loss`, `arrhenius`, `te_power_law` |
| `energy_loss` | per-event electron energy loss models | `constant_event_loss` |

Use `cross_sections_manifest.yaml` only for the tabulated cross-section curves.
Use `electron_impact_models.yaml` to connect a reaction to a cross-section ID.
Use `gas_rate_models.yaml` when the source mechanism gives a rate expression
directly, for example a gas-temperature Arrhenius law or electron-temperature
power law.

Use `first_order_loss` for explicit species-specific volumetric losses such as
a neutral/metastable diffusion-loss surrogate or a calibrated quench-to-wall
term. This is still an ordinary gas reaction, not a machine-learning surrogate
and not a resolved sheath or surface-coverage model. The reaction must have one
non-electron gas reactant with stoichiometry 1:

```csv
reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes
ARSTAR_DIFFUSION_LOSS,gas,Ar_star -> Ar,RM_ARSTAR_DIFFUSION_LOSS,,plasma,,true,metastable first-order wall/diffusion loss
```

```yaml
rate_models:
  RM_ARSTAR_DIFFUSION_LOSS:
    backend: first_order_loss
    rate_s_inv: 2.0e5
```

The helper script below can generate those CSV/YAML entries from diffusion
length or calibrated first-order settings:

```bash
py scripts/generate_species_losses.py losses.yaml --output-dir generated_loss_chemistry
```

Example input:

```yaml
loss_models:
  - species: Ar_star
    return_species: Ar
    zone: plasma
    model: diffusion_length
    diffusion_coefficient_m2_s: 0.05
    diffusion_length_m: 0.01
```

For `te_power_law`, the supported form is:

```yaml
rate_models:
  RM_EXAMPLE:
    backend: te_power_law
    A: 8.5e-13
    Tref_K: 300.0
    alpha: -0.67
    electron_temperature_factor: 0.6666666666666666
```

The local electron temperature is inferred from mean electron energy as
`Te[K] = electron_temperature_factor * mean_energy_eV * 11604.5`. The default
factor is `2/3`, matching a Maxwellian relation between mean energy and
temperature. Do not use `te_power_law` for reactions that have cross-section
data; those should stay in `electron_impact`.

### `cross_sections_manifest.yaml`
Cross-section datasets used by swarm/EEDF backends.

### External rate-table workflow
For BOLSIG+ or other external swarm solvers, keep the raw/exported tables
outside the solver and convert them into the standard HDF5 `rate_table` format:

```text
rate_table_dir/
  rates.csv
  transport.csv
  metadata.yaml
```

`rates.csv` uses one grid column, usually `mean_energy_eV` or `EoverN_Td`,
followed by rate-coefficient columns named by cross-section ID. `transport.csv`
uses the same grid column and may include `mean_energy_eV`,
`effective_field_Td`, `mobility_m2_V_s`, and `diffusion_m2_s`.

```bash
py scripts/build_rate_table_h5.py rate_table_dir --output rate_table.h5
```

The generated file can be consumed by the `rate_table` EEDF backend.

For E/N-gridded tables, use the local-field lookup explicitly:

```yaml
physics:
  eedf_backend: rate_table

swarm:
  model_name: table
  closure: local_field
  table:
    file: tables/my_eovern_rates.h5
    lookup: local_field
    electron_energy_mode: table_relaxation
    energy_relaxation_time_s: 1.0e-6
```

`table_relaxation` keeps the evolved electron energy diagnostic consistent
with the table mean energy while the reaction rates are controlled by the
electrical backend's E/N. For mean-energy-gridded tables, omit
`electron_energy_mode` and use `closure: mean_energy`.

## Why CSV + YAML
CSV is good for large reaction tables.
YAML is better for nested model details.
The code deliberately uses both.

## Validation
Mechanism validation currently checks, among other things:
- species presence
- equation parseability
- charge conservation
- element conservation
- duplicate reactions
- surface-site related checks
- cross-section sanity checks

Validation is not just convenience; it is intended to keep research chemistry datasets usable over time.
