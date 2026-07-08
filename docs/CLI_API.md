# CLI and API

## CLI

Validate:

```bash
python -m plasma_global.cli validate examples/configs/case_smoke.yaml
```

Run:

```bash
python -m plasma_global.cli run examples/configs/case_smoke.yaml
```

Export resolved configuration:

```bash
python -m plasma_global.cli export-config examples/configs/case_smoke.yaml tmp_case_export
```

List backends:

```bash
python -m plasma_global.cli list-backends
python -m plasma_global.cli list-backends --json
```

Backend listings include a short description plus `maturity`, `intended_use`,
and `caveat` fields so reduced, data-driven, experimental, and development
models are easy to distinguish before running a case.

The CLI validates and runs configured cases. It consumes already prepared
chemistry, rate tables, and waveform tables; it does not invoke external swarm
or circuit solvers during RHS evaluation.

## Python API

```python
from plasma_global import load_case_from_yaml, build_case, run_from_yaml

loaded = load_case_from_yaml("examples/configs/case_smoke.yaml")
built = build_case(loaded)
result = run_from_yaml("examples/configs/case_smoke.yaml")
```

`run_from_yaml` is the convenience entry point for scripts. Prefer `summary`,
`observables`, `solution`, and `output_dir` from its result; use
`load_case_from_yaml` and `build_case` directly when a workflow needs prepared
inputs or backend objects.

Run results and `summary.yaml` include `backend_metadata` for the selected
EEDF, electrical, and integrator backends.
