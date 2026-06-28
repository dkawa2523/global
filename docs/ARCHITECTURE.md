# Architecture

The package is organized around a small 0D / multi-zone global-model core.

## Main Layers

- `config`: case loading, path resolution, schema-v2 normalization, and validation.
- `chemistry`: species, reactions, cross sections, rate models, and mechanism validation.
- `reactor`: chamber, zone, surface, inlet, pump, edge, and recipe data.
- `eedf`: analytic, internal approximate, swarm-wrapper, and rate-table EEDF backends.
- `electrical`: direct, table, RF-envelope, CCP, ICP, and DC reduced power backends.
- `numerics`: state layout, ODE system, solver result containers, and SciPy BDF integration.
- `physics`: gas-phase and optional surface RHS contributions.
- `observables`: postprocessed time-series and summary fields.
- `workflows`: load/build/run orchestration and output writing.

## Stable Entry Points

The supported Python entry points are:

- `load_case_from_yaml(path)`
- `build_case(loaded_case)`
- `run_from_yaml(path)`

The supported CLI commands are documented in [CLI and API](CLI_API.md).

## Core Boundary

`GlobalPlasmaSystem` is the solver-facing coordinator. It owns the state layout,
initial state, RHS composition, projection, labels, and solver events. Gas,
surface, EEDF, and electrical details are delegated to their modules.

Observables and file outputs are postprocessing. They read the solution and
system state after the solver path; they should not add requirements to the RHS
or state vector.

The core consumes prepared data. External swarm solvers, SPICE tools, plotting,
benchmark reports, and generated output files belong outside the RHS and normal
case-building path.

## Backends

Backends are selected by simple registries in `workflows/registries.py`.

- EEDF backends return rate coefficients and transport data.
- Electrical backends return zone absorbed power; reduced field, port values,
  and compact surface IED data are optional backend outputs.
- Integrator backends solve the ODE system.

Do not add plugin discovery, hidden metadata buses, or executable calls to the
core workflow. Add small backend code only when the existing registry and result
objects are enough for the job.
