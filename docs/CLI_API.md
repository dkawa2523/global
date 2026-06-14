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

Check Jacobian:

```bash
python -m plasma_global.cli check-jacobian examples/configs/case_smoke.yaml --top 10
```

## Python API

```python
from plasma_global import load_case_from_yaml, build_case, run_from_yaml

loaded = load_case_from_yaml("examples/configs/case_smoke.yaml")
built = build_case(loaded)
result = run_from_yaml("examples/configs/case_smoke.yaml")
```

`run_from_yaml` returns loaded inputs, selected backends, the assembled system, the solution, the summary, and the output directory.

