# Architecture

The package is organized around a small workflow core.

## Main Layers

- `config`: case loading, includes, schema-v2 normalization, path resolution, and validation.
- `chemistry`: species, reactions, cross sections, rate models, and mechanism validation.
- `reactor`: chamber, zone, surface, inlet, pump, edge, and recipe models.
- `eedf`: Maxwell, table, swarm, and internal two-term-like EEDF closures.
- `electrical`: direct power, DC series, external table, RF envelope, CCP, and ICP reduced backends.
- `numerics`: state layout, ODE system, Jacobian, and SciPy BDF integration.
- `observables`: compact postprocessed time-series and summary outputs.
- `workflows`: load/build/run orchestration.

## Stable Entry Points

The supported Python entry points are:

- `load_case_from_yaml(path)`
- `build_case(loaded_case)`
- `run_from_yaml(path)`

The supported CLI commands are documented in [CLI and API](CLI_API.md).

## Extension Points

Backends are selected by registry name:

- EEDF backend
- electrical backend
- integrator backend

Each backend should keep its request/result interface small and avoid reading files directly unless the backend owns that file format.

