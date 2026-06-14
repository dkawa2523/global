# Plasma Global Model

YAML-driven low-pressure plasma global model for CLI and Python API workflows.

The project focuses on a compact, inspectable core:

- load and validate a case from YAML
- assemble chemistry, reactor, EEDF, electrical, and ODE components
- run a transient global model
- write `summary.yaml`, `observables.csv`, resolved config snapshots, and optional `solution.h5`

It is a reduced-order 0D / multi-zone modeling base, not a spatial fluid/PIC solver and not a full RF sheath, feature-scale, or circuit simulator.

## Install

Python 3.11 or newer is expected.

```bash
python -m pip install -e ".[io,dev]"
```

Use the minimal install when HDF5 output and test tools are not needed:

```bash
python -m pip install -e .
```

## Quickstart

Validate a case:

```bash
python -m plasma_global.cli validate examples/configs/case_smoke.yaml
```

Run it:

```bash
python -m plasma_global.cli run examples/configs/case_smoke.yaml
```

List available backends:

```bash
python -m plasma_global.cli list-backends
python -m plasma_global.cli list-backends --json
```

Export the effective configuration without running the solver:

```bash
python -m plasma_global.cli export-config examples/configs/case_smoke.yaml tmp_case_export
```

## Recommended Production Workflow

1. Prepare chemistry and cross-section data.
2. Generate electron-impact rate and transport tables with an external swarm solver when high-fidelity electron kinetics are required.
3. Run the global model with the `rate_table` backend.
4. Inspect particle, power, wall-loss, surface, and electrical observables.
5. Save `effective_case.yaml`, `resolved_paths.yaml`, `summary.yaml`, and `observables.csv` for reproducibility.

For table guidance, see [External Swarm and Rate Tables](docs/SWARM_RATE_TABLES.md). For extension boundaries, see the [Extension Guide](docs/EXTENSION_GUIDE.md).

## Public API

```python
from plasma_global import load_case_from_yaml, build_case, run_from_yaml

loaded = load_case_from_yaml("examples/configs/case_smoke.yaml")
built = build_case(loaded)
result = run_from_yaml("examples/configs/case_smoke.yaml")
```

## Inputs

Supported case files use schema version 2 with these top-level sections:

- `case`
- `files`
- `physics`
- `numerics`
- `outputs`
- `swarm`

Legacy `run.yaml` style files are no longer supported.

## Outputs

Normal runs write only the core artifacts requested by `outputs.formats`:

- `summary.yaml`
- `observables.csv`
- `solution.h5` when enabled and `h5py` is installed
- `effective_case.yaml`
- `resolved_paths.yaml`

Generated outputs are ignored by git and can be regenerated from the case files.

## Model Limits

This code is useful for mechanism checks, reduced-order discharge studies, sensitivity sweeps, and CLI/API integration. Interpret results with the configured model choices in mind:

- EEDF and swarm behavior are reduced-order or table driven.
- The internal `boltzmann_2term` backend is approximate and is not a replacement for mature swarm solvers such as BOLSIG+, LoKI-B, or Magboltz.
- Electrical backends are lumped or prescribed models.
- Sheath and ion-energy diagnostics are proxy-level.
- Spatial transport is represented by global zones and conductance links.
- Quantitative process claims require calibration or external validation for the target regime.

## Tests

Run the lean gate:

```bash
python -m pytest
```

The default test set covers validation, core numerics, representative electrical backends, smoke execution, and strict compact benchmark checks.
