# Plasma Global Model

YAML-driven 0D / multi-zone low-pressure plasma global model.

The core loads a case, builds chemistry/reactor/EEDF/electrical/numerics
components, solves the ODE system, and writes configured outputs.

## Reading Map

Start here when reviewing or extending the code:

1. `plasma_global/workflows/runner.py` is the shortest end-to-end path:
   load, build, solve, write outputs.
2. `plasma_global/workflows/case_loader.py` validates and loads YAML, reactor,
   recipe, chemistry, rate, and cross-section inputs.
3. `plasma_global/workflows/case_builder.py` turns loaded inputs into backend
   instances, state layout, and `GlobalPlasmaSystem`.
4. `plasma_global/numerics/system.py` is the solver-facing coordinator. It owns
   the state vector contract and delegates gas, surface, process, EEDF, and
   electrical details to smaller modules.
5. `plasma_global/coupling/evaluation.py` evaluates the algebraic
   coupling needed by each RHS call: electrical power, EEDF rates, transport,
   electron density, mean energy, pressure, and ion metadata.
6. `plasma_global/physics/*_core.py` modules apply RHS terms. They should remain
   small orchestrators over helper modules, not become new monoliths.

The main design rule is: keep external data loading, validation, backend
selection, RHS assembly, and postprocessing in separate layers. Runtime RHS code
consumes prepared Python objects and does not call external executables.

## Non-Goals

This package is not:

- a spatial fluid or PIC solver
- a detailed RF sheath or electromagnetic solver
- a feature-scale surface simulator
- a runtime wrapper around external swarm or circuit executables

Use the results as reduced-order model output. Quantitative cases need
case-specific calibration or validation.

## Install

Python 3.11 or newer is expected.

```bash
python -m pip install -e ".[io,dev]"
```

Minimal install:

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

## Python API

```python
from plasma_global import load_case_from_yaml, build_case, run_from_yaml

loaded = load_case_from_yaml("examples/configs/case_smoke.yaml")
built = build_case(loaded)
result = run_from_yaml("examples/configs/case_smoke.yaml")
```

Stable public API:

- `load_case_from_yaml`
- `build_case`
- `run_from_yaml`

## Outputs

Configured runs may write:

- `summary.yaml`
- `observables.csv`
- `solution.h5` when HDF5 output is enabled and `h5py` is installed
- `effective_case.yaml`
- `resolved_paths.yaml`

Generated outputs are ignored by git and should be regenerated from case files.

## Tests

```bash
python -m pytest
```
