# Extension Guide

This project is intended to remain a compact, inspectable 0D / multi-zone plasma global model foundation. New features should extend the model through clear data contracts and small backends, not by adding solver-specific logic directly into the core.

## Design Goals

The extension policy is based on the following goals:

1. Keep the public API small.
2. Keep the plasma core independent of external tools.
3. Prefer data contracts over runtime coupling.
4. Make physical assumptions explicit.
5. Make validation part of every new feature, and expose only diagnostics that belong in normal user output.
6. Avoid adding features that belong to fluid, PIC, feature-scale, or full circuit simulators.

## Core Boundary

The core plasma system should not know about specific external programs such as BOLSIG+, LoKI-B, Magboltz, or ngspice.

The core should only consume stable internal objects such as:

- mechanism data
- reactor and recipe configuration
- EEDF or rate coefficient results
- electrical coupling results
- numerical options
- observables definitions

External tools must be connected through one of the following layers:

| Layer | Use for |
|---|---|
| backend | Internal selectable models such as EEDF, electrical, wall-loss, or integrator models |
| adapter | External data readers or optional tool interfaces |
| tool | Offline converters, benchmark runners, sweep utilities |
| docs/examples | User-facing workflows and reference cases |

## Preferred Extension Pattern

A new model should follow this pattern:

```text
configuration
  -> typed config or validated dictionary
  -> backend or adapter
  -> stable result object
  -> GlobalPlasmaSystem
  -> observables
```

Avoid this pattern:

```text
GlobalPlasmaSystem
  -> direct call to external executable
  -> solver-specific parsing
  -> hidden mutable state
```

## External Swarm Solvers

BOLSIG+, LoKI-B, and Magboltz are specialized electron swarm / Boltzmann tools. This package should not try to reimplement all of their functionality in the core.

The recommended production path is:

```text
external swarm solver
  -> offline rate/transport table
  -> rate_table backend
  -> global model
```

An online adapter may be added later, but it must preserve the same request/result contract and remain optional.

## Circuit Coupling

Circuit coupling should start with one-way waveform or table coupling:

```text
measured or SPICE-generated waveform
  -> external_circuit_table backend
  -> plasma model
```

Direct bidirectional ngspice co-simulation should be treated as an advanced optional adapter. It must not become a required dependency of the core package.

The current supported circuit-table path is one-way: measured or SPICE-generated
CSV data are converted to stable waveform tables and consumed by the
`external_circuit_table` backend. Do not add direct ngspice execution to the ODE
RHS or Jacobian path.

## Wall-Loss and Surface Models

Wall-loss and surface models should be small and explicit.

Recommended wall-loss model families:

- prescribed loss frequency
- Bohm-type global loss

Electronegative cases should use calibrated Bohm factors or prescribed loss
frequency until a carefully scoped optional closure is added outside the core
RHS.

Recommended surface model scope:

- sticking coefficients
- ion-enhanced yields
- site balance when needed
- film thickness or etch/deposition rate observables

Do not add feature-scale profile evolution, detailed surface Monte Carlo, or large material databases to the core.

## Configuration Rules

New configuration sections should be:

- documented in `docs/CONFIGURATION.md`
- validated before runtime
- represented by typed dataclasses when the structure is stable
- migration-friendly when that does not keep obsolete contracts alive
- explicit about defaults

Avoid unstructured `Any`-style configuration for production-facing features.

## Documentation Rules

Every new feature should update:

- relevant user documentation
- configuration examples
- CLI/API documentation if needed
- physics/numerics documentation if assumptions change
- tests or benchmark notes

Remove or update obsolete statements. Do not leave contradictory descriptions in older pages.

## Testing Rules

A new extension should include at least one of:

- unit tests for parsing or validation
- contract tests for backend input/output
- finite-difference or conservation checks for numerical terms
- smoke examples
- benchmark comparisons

For numerical features, prefer small reproducible cases over large expensive tests.

## Non-Goals

The following are intentionally outside the core scope:

- full n-term Boltzmann solver implementation
- 2D or 3D fluid plasma simulation
- PIC simulation
- feature-scale etch/deposition modeling
- detailed RF sheath time resolution
- mandatory ngspice dependency
- GUI-first workflow
- large bundled reaction or material databases
