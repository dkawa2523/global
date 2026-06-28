# Plasma Global Model

Compact YAML-driven 0D / multi-zone low-pressure plasma global model.

The core is intentionally small: load a case, build chemistry/reactor/EEDF/electrical/numerics components, solve the ODE system, and write configured outputs. It is not a spatial fluid/PIC solver, a full RF sheath model, a feature-scale surface simulator, or a runtime wrapper around external swarm or circuit solvers.

## Install

Python 3.11 or newer is expected.

```bash
python -m pip install -e ".[io,dev]"
```

For a minimal install without HDF5 output and test tools:

```bash
python -m pip install -e .
```

## CLI

```bash
python -m plasma_global.cli validate examples/configs/case_smoke.yaml
python -m plasma_global.cli run examples/configs/case_smoke.yaml
python -m plasma_global.cli list-backends
python -m plasma_global.cli export-config examples/configs/case_smoke.yaml tmp_case_export
```

The CLI consumes prepared YAML, chemistry, rate-table, and waveform-table inputs. It does not invoke external solvers during RHS evaluation.

## Python API

```python
from plasma_global import load_case_from_yaml, build_case, run_from_yaml

loaded = load_case_from_yaml("examples/configs/case_smoke.yaml")
built = build_case(loaded)
result = run_from_yaml("examples/configs/case_smoke.yaml")
```

The stable public API is `load_case_from_yaml`, `build_case`, and `run_from_yaml`.

## Outputs

Runs write the artifacts requested by the case configuration:

- `summary.yaml`
- `observables.csv`
- `solution.h5` when HDF5 output is enabled and `h5py` is installed
- `effective_case.yaml`
- `resolved_paths.yaml`

Generated outputs are ignored by git and can be regenerated from case files.

## Scope

The model is useful for compact discharge studies, mechanism checks, reduced-order multi-zone cases, and CLI/API integration. Interpret results through the configured closures:

- EEDF behavior is analytic, internal approximate, or table driven.
- Electrical coupling is lumped, prescribed, or table driven.
- Wall and surface terms are global closures, not detailed sheath or feature-scale models.
- Quantitative use requires calibration or validation for the target regime.

See `docs/` for configuration, architecture, and rate-table notes.

## Tests

```bash
python -m pytest
```
