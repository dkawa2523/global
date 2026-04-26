# Configuration guide

## Recommended case workflow
Use three layers:

1. `base_case.yaml` — shared defaults
2. `case.yaml` — main case
3. `case_smoke.yaml` or process-specific override files

The loader supports:
```yaml
include: base_case.yaml
```

## Main sections of `case.yaml`

### `case`
Case metadata.

### `files`
References all external input files.

Important fields:
- `chamber`
- `recipe`
- `chemistry.manifest`
- `output_dir`
- `external_inputs.*`

### `physics`
Selects physical models and major switches.

Electrical model choices currently include:
- `direct_power` for prescribed absorbed power,
- `dc_series_circuit` for a reduced voltage source, ballast resistor, and conductive plasma load,
- `external_circuit_table` for one-way measured/SPICE waveform CSV input,
- `rf_envelope` for cycle-averaged HF/LF source and bias settings,
- `icp` and `ccp` for reduced source and bias proxies.

For circuit-specific parameters and future external-circuit coupling inputs,
see `CIRCUIT_COUPLING.md`.
For HF/LF `rf_envelope` coefficient calibration, start from
`examples/configs/case_rf_envelope_calibration.yaml` and
`RF_ENVELOPE_CALIBRATION.md`.

Electron density choices are configured separately from the EEDF closure:
- `electron_density_closure: quasi_neutral` is the normal global-model mode. Electron density is not an ODE state; it is inferred from ion charge balance.
- `electron_density_closure: prescribed_profile` reads electron density from a CSV profile. Use this for benchmark-driven chemistry or one-way coupling studies, not for self-consistent plasma production.
- `external_profile` and `profile` are accepted aliases for `prescribed_profile`.

Example:

```yaml
files:
  external_inputs:
    electron_profile_csv: ../external/electron_density_profile.csv

physics:
  electron_density_closure: prescribed_profile

swarm:
  prescribed_electron_profile:
    file_key: electron_profile_csv
    density_column: electron_density_m3
    interpolation: linear
    hold: edge
```

The profile CSV must contain `time_s` and either a shared density column such
as `electron_density_m3` or zone-specific columns:

```yaml
swarm:
  prescribed_electron_profile:
    file_key: electron_profile_csv
    zone_columns:
      plasma: ne_plasma_m3
      downstream: ne_downstream_m3
```

Supported shared column names include `electron_density_m3`, `ne_m3`,
`electrons_m3`, `electron_density_cm3`, `ne_cm3`, and `Electrons_cm-3`.

### Chamber zone initial densities
By default, gas species are initialized from the zone pressure and the first
recipe-step gas mix. Benchmark cases can override selected species explicitly:

```yaml
zones:
  - zone_id: plasma
    pressure_Pa: 103548.675
    gas_temperature_K: 300.0
    initial_densities_m3:
      Ar: 2.5e25
      Ar_plus: 1.0e6
```

Only listed gas-state species are overridden. Electron density remains
quasi-neutral and is inferred from charged species, so do not list `e` here.

Useful reduced gas-temperature knobs:
- `gas_heating_fraction` — fraction of absorbed power deposited into the gas-temperature equation.
- `wall_relaxation_s_inv` — first-order gas-temperature relaxation rate toward the area-weighted wall temperature.

Surface model controls:
- `surface.models.ion_loss` is consumed by the solver. Prefer `bohm_edge_loss` for the current default positive-ion wall-loss closure. Legacy `Bohm_like`, `sheath_flux`, `on`, or an omitted key are accepted as aliases. `off`, `none`, or `disabled` remove that surface from the reduced ion-loss closure.
- `bohm_edge_loss` uses the configured surface area and Bohm speed. It defaults to `h_factor: 1.0` for backward-compatible edge-density behavior.
- `bohm_global_loss` uses the same Bohm-speed closure with an automatic edge-to-center correction unless `h_factor` is set explicitly. The automatic correction is a first-order transport-limited estimate, not a calibrated geometry model.
- Optional Bohm keys: `h_factor`, `edge_to_center_factor`, `characteristic_length_m`, `ion_neutral_cross_section_m2`, `min_h_factor`, `max_h_factor`.
- `ambipolar_diffusion` is a separate first-order volumetric ion-loss mode. Configure either `ambipolar_loss_rate_s` directly, or `diffusion_coefficient_m2_s` with optional `diffusion_length_m`.
- Do not mix Bohm-family and ambipolar-diffusion ion-loss modes in the same zone; validation rejects this to avoid double counting the same wall loss.
- Other current `surface.models` keys are retained as metadata unless they are also represented by explicit surface reactions.

Examples:

```yaml
models:
  ion_loss: bohm_edge_loss
```

```yaml
models:
  ion_loss: bohm_global_loss
  characteristic_length_m: 0.015
  ion_neutral_cross_section_m2: 1.0e-18
```

```yaml
models:
  ion_loss: ambipolar_diffusion
  ambipolar_loss_rate_s: 2.0e5
```

```yaml
models:
  ion_loss: ambipolar_diffusion
  diffusion_coefficient_m2_s: 1.0
  diffusion_length_m: 0.01
```

### `swarm`
Swarm/EEDF model-specific configuration.

Useful closure choices:
- `closure: auto` currently resolves to the mean-energy closure for the internal global ODE model.
- `closure: mean_energy` ties electron-impact rates to the evolved electron energy density through `We/ne`.
- `closure: local_field` ties rates to the electrical backend reduced-field proxy. Validation warns on this mode because production use needs calibrated `zone_reduced_field_Td`.

`swarm.prescribed_electron_profile` is only used when
`physics.electron_density_closure` is `prescribed_profile`. It does not change
which gas species are ODE states; it only replaces the electron density passed
to the electrical/EEDF coupling and diagnostics.

### `numerics`
Integrator tolerances, step control, Jacobian behavior, positivity safeguards.

### `outputs`
Output formats and plots.

### `runtime`
Run-control toggles such as effective-config export.

## Why the chemistry manifest exists
Without a manifest, the chemistry file set is implicit and easy to break during long projects. With `chemistry_manifest.yaml`, a reviewer can see the entire chemistry bundle in one file.

## Generated reproducibility files
Every run can export:
- `effective_case.yaml` — resolved effective case settings
- `resolved_paths.yaml` — absolute resolved path map

These files are intentionally written to the output directory so that runs remain reproducible even when the original include chain changes later.
