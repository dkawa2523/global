# Core split refactor

This note documents the refactor that decomposes the former monolithic
`GlobalPlasmaSystem` into four explicit collaborators.

## New internal split

### `GasPhaseCore`
Owns:
- gas-species state initialization
- gas-phase reaction compilation
- gas-phase chemistry source terms
- inlet / pump / inter-zone transport
- electron-density and positive-ion-density reconstruction
- reduced gas-temperature source terms
- gas-side Jacobian contributions

File:
- `plasma_global/physics/gas_phase_core.py`

### `SurfaceCore`
Owns:
- surface-reaction compilation
- coverage projection and free-site reconstruction
- wall inventory and film state initialization
- surface reaction-rate laws
- surface RHS contributions
- surface Jacobian contributions
- surface net gas flux bookkeeping

File:
- `plasma_global/physics/surface_core.py`

### `ElectricalCouplingAdapter`
Owns:
- packaging `PowerRequest`
- packaging `EEDFRequest`
- calling the selected electrical backend
- calling the selected EEDF backend
- building a coupled per-zone evaluation object shared by the other cores

File:
- `plasma_global/electrical/coupling_adapter.py`

### `ObservablesAdapter`
Owns:
- time-sampled engineering and plasma-physics observables
- wafer/surface KPI extraction
- warning flag generation

File:
- `plasma_global/observables/adapter.py`

## What `GlobalPlasmaSystem` still owns

`GlobalPlasmaSystem` remains the orchestration object seen by the time
integrator. It still owns:
- stable public solver-facing interface
- top-level state metadata
- shared immutable indexing information
- recipe-step lookup
- top-level state projection
- coordination of RHS / Jacobian / observables calls

That means the solver API does not change, but the physics logic is no longer
concentrated in a single file.

## Why this split matters

### Architect view
- lower cognitive load per module
- clearer responsibility boundaries
- easier future replacement of one collaborator without editing all physics

### Simulation engineer view
- easier to inspect which block changes source terms
- easier to profile gas vs surface vs coupling time
- easier to add validation for one block at a time

### Plasma physicist view
- easier to replace one closure family independently
- easier to audit which approximations affect gas chemistry vs sheath coupling
- easier to connect higher-fidelity modules later

## Compatibility

The refactor preserves the external solver interface used by:
- `workflows/runner.py`
- `SciPyBDFIntegrator`
- existing YAML cases

A backward-compatible `_eval_power_and_eedf(...)` wrapper is retained so that
older analysis code can still access the coupled evaluation tuple.
