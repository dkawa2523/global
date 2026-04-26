# Architecture guide

## Design goal
The code is organized so that a reviewer can answer three questions quickly:

1. Where does a given input come from?
2. Which module owns a physical approximation?
3. What must be changed to replace a backend?

## Package map

### `plasma_global/config`
Owns case schema loading, YAML include/override handling, path resolution, and export of the effective configuration.

### `plasma_global/chemistry`
Owns species definitions, reaction definitions, reaction parsing, cross-section loading, and mechanism validation.

### `plasma_global/reactor`
Owns chamber topology and recipe definitions.

### `plasma_global/eedf`
Owns EEDF/swarm closures. The important architectural rule is that the plasma chemistry solver only sees `EEDFRequest -> EEDFResult`.

### `plasma_global/electrical`
Owns absorbed-power and reduced electrical models. The plasma chemistry solver only sees `PowerRequest -> PowerResult`.

### `plasma_global/numerics`
Owns state layout, system assembly, Jacobian assembly, and time integration interfaces.

### `plasma_global/observables`
Owns postprocessed engineering and physics outputs.

### `plasma_global/workflows`
Owns assembly of a runnable case.

## Key architectural choices

### 1. Case assembly separated from execution
`workflows/context.py` performs:
- load config
- resolve paths
- validate config
- load chamber/recipe/mechanism
- build EEDF/electrical/integrator
- construct the system

`workflows/runner.py` performs:
- execute recipe segments
- write outputs
- export reproducibility files

This separation makes it easier to inspect, test, and replace pieces.

### 2. Replaceable backend interfaces
The following abstractions are the primary extension points:
- `EEDFBackend`
- `SwarmModel`
- `ElectricalBackend`
- `TimeIntegrator`

### 3. Explicit file manifests
A case points to:
- chamber YAML
- recipe YAML
- chemistry manifest YAML
- optional external inputs

A chemistry manifest points to:
- species CSV
- gas reaction CSV
- surface reaction CSV
- reaction model YAML
- cross-section manifest YAML

This is more reviewable than assuming implicit filenames everywhere.

### 4. Split internal physics collaborators
The transient system is now internally decomposed into:
- `GasPhaseCore`
- `SurfaceCore`
- `ElectricalCouplingAdapter`
- `ObservablesAdapter`

`GlobalPlasmaSystem` remains the object seen by the ODE solver, but it now mainly coordinates these collaborators instead of directly owning all gas, surface, electrical, and observables logic.

## Current large coupling point
`GlobalPlasmaSystem` is still the central multiphysics coupling object. This is acceptable, but it should remain disciplined:
- it may assemble physics,
- it should not become the only place where configuration or file I/O logic lives.

That is why loader, manifest resolution, registry, and documentation were moved outside it.
