# Plasma Global Model — architecture-refined revision

This revision focuses on four practical questions:

1. Can a user manage a case through YAML without chasing paths across the code?
2. Can a third party understand where each physics model lives?
3. Can a developer replace one subsystem without rewriting the whole solver?
4. Are the code, approximations, and I/O spec documented enough for long-term maintenance?

The answer is now **substantially more yes than before**, but still with the normal caveat of a research code: some physics closures remain reduced-order and should be treated as replaceable modules rather than final truth models.

## What changed in this revision

### Configuration and case management
- added **schema_version 2** case format: `examples/configs/case.yaml`
- preserved backward compatibility with legacy `run.yaml`
- added recursive YAML `include:` support for base/override workflows
- added explicit `chemistry_manifest.yaml` so chemistry file paths are reviewable in one place
- added automatic export of
  - `effective_case.yaml`
  - `resolved_paths.yaml`
  into the output directory
- added `scripts/validate_case.py` and `scripts/export_effective_config.py`

### Architecture and maintainability

#### Internal physics split
- split the former monolithic `GlobalPlasmaSystem` internals into
  - `GasPhaseCore`
  - `SurfaceCore`
  - `ElectricalCouplingAdapter`
  - `ObservablesAdapter`
- kept the public solver-facing API stable while moving most domain logic out of `numerics/system.py`
- retained a backward-compatible `_eval_power_and_eedf(...)` wrapper for downstream tooling

- added `plasma_global/workflows/context.py` to separate
  - loading
  - validation
  - component construction
  - solver assembly
- improved registry with descriptions so available backends can be listed and reviewed
- added public `plasma_global/api.py`
- kept EEDF, swarm, electrical, reactor, numerics, chemistry as separate packages

### Documentation
A new `docs/` directory now explains the code from architect, simulation-engineer, and plasma-physics viewpoints.
A compact review memo is also available in `ARCHITECT_SIM_PHYS_REVIEW.md`.

Start with:
- `docs/ARCHITECTURE.md`
- `docs/CONFIG_GUIDE.md`
- `docs/CHEMISTRY_INPUT_SPEC.md`
- `docs/PHYSICS_MODELS_AND_APPROXIMATIONS.md`
- `docs/MODEL_MATURITY.md`
- `docs/NUMERICS_AND_SOLVERS.md`
- `docs/DEVELOPER_GUIDE.md`

## Install for development

```bash
python -m pip install -e ".[io,plot,dev]"
```

The core package keeps HDF5 and plotting optional. Install `.[io]` for
`solution.h5` output and `.[plot]` for requested plots.

## Recommended entry points

### Validate a case
```bash
plasma-global validate examples/configs/case_smoke.yaml
```

### Run a smoke case
```bash
plasma-global run examples/configs/case_smoke.yaml
```

### Run the main example
```bash
plasma-global run examples/configs/case.yaml
```

### Export the fully resolved configuration without running
```bash
plasma-global export-config examples/configs/case.yaml ./tmp_case_dump
```

### List model backends and maturity labels
```bash
plasma-global list-backends
```

### Run the compact regression set
```bash
python -m pytest
```

### Run external-code benchmark tools
ZDPlaskin, CRANE, and PyGMol comparison handling is kept outside the core solver package:

```bash
python tools/external_benchmarks/run_external_benchmarks.py
```

Broader pressure/electrical/chemistry stress checks live in the same external
tool area:

```bash
python tools/external_benchmarks/robustness_sweep.py
```

The benchmark data inventory is in:

```bash
tools/external_benchmarks/benchmark_data_manifest.yaml
```

The external model candidate registry is in:

```bash
tools/external_benchmarks/external_model_registry_report.yaml
```

## Current judgment on the code base

### Is it YAML-driven and usable?
Yes, with caveats.
- The simulation is now case-driven by YAML.
- File paths, physics selections, numerics, outputs, and external anchors are managed from case YAML.
- Chemistry file paths can be managed either as a directory or through `chemistry_manifest.yaml`.
- The new include/override mechanism reduces duplication.

### Is the code maintainable and understandable for third parties?
Mostly yes.
- Subsystems are separated into chemistry, reactor, swarm/EEDF, electrical, numerics, observables, and workflows.
- The new docs describe what each package owns.
- `context.py` reduces orchestration complexity inside the runner.

Remaining weakness:
- `GlobalPlasmaSystem` is now much thinner than before, but it still remains the top-level transient coordinator. Very high-fidelity future additions should continue to go into the dedicated collaborators rather than back into one central file.

### Are physics models separable and reusable?
Largely yes.
- EEDF backends are replaceable.
- Swarm models are replaceable inside the swarm wrapper.
- Electrical backends are replaceable.
- The workflow builder can assemble a system without hiding the chosen components.

### Has input/output complexity been controlled?
Improved, but not magically eliminated.
- Complex multiphysics cases are inherently complex.
- The new case/base/override workflow is intended to control that complexity.
- The main mitigation is explicit manifests plus generated `effective_case.yaml`.

## Outputs
Depending on case settings, the run writes:
- `solution.h5`
- `observables.csv`
- `summary.yaml`
- `effective_case.yaml`
- `resolved_paths.yaml`
- requested plots

## Important scientific caveat
This code is a modular research-grade platform, not a claim of final semiconductor-process truth. In particular:
- the sheath / IED path remains a proxy module,
- some transport and coupling closures are reduced-order,
- any production use should anchor important regimes with experiment and/or higher-fidelity models.
