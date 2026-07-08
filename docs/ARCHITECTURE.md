# Architecture

The package is a small YAML-to-ODE workflow for 0D / multi-zone global plasma
models.

## Design Principles

- Keep the runtime path explicit: YAML/CSV/HDF5 inputs are normalized before the
  ODE system is built.
- Keep the solver-facing core small: `GlobalPlasmaSystem` owns the state vector,
  RHS composition, projection, labels, and events, then delegates domain logic.
- Prefer simple registries over implicit plugin discovery. Adding a backend
  should be a visible change in the registry owned by that backend domain.
- Treat EEDF, electrical power, wall loss, surface chemistry, and postprocessing
  as separate concerns. Data should cross boundaries through typed request/result
  objects or compact dataclasses.
- Keep external solvers offline. The runtime can read prepared rate, transport,
  or circuit tables, but the RHS should not launch external programs.

## Run Flow

```text
run_from_yaml(case.yaml)
  -> load_case_from_yaml
       -> load schema-v2 case config
       -> resolve paths relative to the case file
       -> validate config, reactor/recipe structure, and chemistry
       -> load chamber, recipe, species, reactions, rate models, cross sections
  -> build_case
       -> build EEDF, electrical, and integrator backends from registries
       -> prepare backends with loaded inputs
       -> build the ODE state layout
       -> create GlobalPlasmaSystem
  -> solve_built_case
       -> initialize and project state
       -> solve each recipe step with the configured integrator
       -> project segment outputs and pass final state to the next step
  -> write_run_outputs
       -> compute observables and summaries
       -> write configured files and plots
```

During each RHS call:

```text
GlobalPlasmaSystem.rhs(t, y)
  -> current recipe step
  -> PlasmaCouplingEvaluator.evaluate
       -> project state into zone electrical metadata
       -> evaluate electrical backend
       -> evaluate EEDF backend using power/field and plasma state
       -> optionally iterate transport mobility coupling
  -> GasPhaseCore.apply_rhs
       -> gas reactions, ion wall losses, flows, power, energy relaxation
  -> SurfaceCore.apply_rhs, if enabled
  -> ProcessCore.apply_rhs, if enabled
  -> positivity guard
```

## Layers

- `config`: schema-v2 loading, includes, path resolution, validation, exports.
- `chemistry`: species, reactions, cross sections, rate models, provenance.
- `reactor`: chamber, zone, surface, inlet, pump, edge, recipe data, and structural validation.
- `eedf`: Maxwell closure plus a swarm wrapper for Boltzmann-like and HDF5 table models.
- `electrical`: reduced absorbed-power, RF, DC, and one-way table backends.
- `coupling`: per-RHS-call projection and algebraic power/EEDF coupling.
- `physics`: gas-phase RHS, wall loss, and optional surface RHS terms.
- `numerics`: state layout, `GlobalPlasmaSystem`, and SciPy BDF integration.
- `observables`: compact postprocessed records and summaries.
- `workflows`: load, build, solve, and output orchestration.

## Entry Points

Python API:

- `load_case_from_yaml(path)`
- `build_case(loaded_case)`
- `run_from_yaml(path)`

CLI:

- `validate`
- `run`
- `list-backends`
- `export-config`

See [CLI and API](CLI_API.md).

## Core Boundary

`GlobalPlasmaSystem` owns the solver-facing state layout, initial state, RHS,
projection, labels, and solver events. Gas, surface, process, EEDF, and
electrical details stay in their modules. `GasPhaseCore` is the gas RHS
orchestrator; gas state projection/initialization, wall-loss terms, gas
reactions, flow terms, and table-driven electron-energy relaxation live in
small helper modules. Coupled-state construction and the power/EEDF handshake
live in `coupling/state_view.py` and `coupling/evaluation.py`.

Chemistry validation is similarly split by responsibility: `validators.py`
coordinates the checks, `reaction_validation.py` validates reaction balance and
rate references, and `extension_validation.py` validates extra state/process
extensions. Chemistry loading keeps manifest parsing and rate-model file
loading outside the top-level `io.py` entry point.

The core consumes prepared YAML, CSV, and HDF5 inputs. It does not call external
solver executables from the RHS or normal case-building path. Observables,
optional budget fields, summaries, plots, reports, and generated files are
postprocessing.

Backends are selected by simple domain registries in `eedf/registry.py`,
`electrical/registry.py`, and `numerics/registry.py`. Avoid plugin discovery,
hidden metadata buses, or large protocol hierarchies in the core workflow.
