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

## Core Contracts

The core plasma system consumes stable internal objects: mechanism data,
reactor/recipe configuration, EEDF results, electrical coupling results,
numerical options, and observables. It should not know whether rate
coefficients came from `maxwell`, `boltzmann_2term`, `rate_table`, BOLSIG+,
LoKI-B, Magboltz, or another offline source. It should also not know whether
electrical waveforms came from measurements, a simple reduced backend, or a
SPICE-generated table.

External solver integration belongs in offline tools, adapters, or backend
file readers. Avoid direct calls to external executables from the ODE RHS,
Jacobian, or core workflow. See the [Extension Guide](EXTENSION_GUIDE.md) for
the project boundary policy.

## Extension Points

Backends are selected by registry name:

- EEDF backend
- electrical backend
- integrator backend

Each backend should keep its request/result interface small and avoid reading files directly unless the backend owns that file format.
