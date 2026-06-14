# Review Results

Generated comparison files under `examples/outputs/` are intentionally not
tracked. They are reproducible run products, and `.gitignore` keeps them out of
the repository. This page records the review-relevant results so reviewers can
understand the branch without checking in large YAML, CSV, HDF5, or PNG output.

## Verification Commands

Run on branch `update_2` after the product-core refactor:

```powershell
py -m pytest
py -m plasma_global.cli validate examples/configs/case_smoke.yaml
py -m plasma_global.cli run examples/configs/case_smoke.yaml
py tools\external_benchmarks\crane_two_reaction_argon.py
py -m plasma_global.cli run examples/configs/case_zdplaskin_example2.yaml
py tools\external_benchmarks\zdplaskin_example2.py
py tools\external_benchmarks\zdplaskin_eovern.py
py -m plasma_global.cli list-backends --json
py -m mkdocs build --strict
```

## Gate Results

| Check | Result |
| --- | --- |
| Lean pytest gate | `49 passed` |
| Smoke case validation | OK; only metadata-only surface-model warnings |
| Smoke run | solver success, 200 samples |
| MkDocs strict build | OK |
| Generated output tracking | run outputs stay ignored under `examples/outputs/` |

The smoke run writes only the product-core output set:

- `effective_case.yaml`
- `observables.csv`
- `resolved_paths.yaml`
- `solution.h5`
- `summary.yaml`

## Physics Review

The code is useful as a reduced low-pressure plasma global-model basis. It is
strongest for fast case setup, backend comparison, chemistry validation, and
screening-level trends across electrical closures and EEDF closures. It should
not be read as a calibrated reactor digital twin without case-specific
validation of wall losses, surface reactions, EEDF data, and electrical
coupling.

The refactor keeps the physical equations and state history intact except for
removing diagnostic outputs and removing the missing-rate-table fallback. The
latter is an intentional safety change: `rate_table` now fails fast when the
table file is absent, and quick analytic checks should use the explicit
`maxwell` backend.

## Numerics Review

The retained numerical path is a stiff ODE workflow with SciPy BDF, stable state
layout labels, projected positive state handling, smoke Jacobian coverage, and
focused external parity tests. This is a practical basis for CLI/API use.

`check-jacobian` remains a diagnostic command rather than an acceptance gate.
For `case_smoke.yaml`, it exits successfully but reports `max_relative_error: 1`
on selected diagonal finite-difference comparisons. That means the current
analytic Jacobian is useful for solver support and smoke inspection, but should
not be claimed as fully finite-difference exact.

## Architecture Review

The public product surface is now deliberately small:

- CLI: `validate`, `run`, `export-config`, `list-backends`, `check-jacobian`
- API: `load_case_from_yaml`, `build_case`, `run_from_yaml`
- Config: schema version 2 case files only
- Output: summary, observables, effective case, resolved paths, optional HDF5

The old generated reports, dashboard outputs, broad benchmark dashboards,
legacy config normalization, registry maturity contracts, provenance outputs,
state manifests, budget reports, and warning columns were removed to keep the
core package readable and maintainable.

## Comparison Results

| Reference | Main result | Interpretation |
| --- | --- | --- |
| CRANE TwoReactionArgon | final electron density relative error `2.366e-6` | Scalar ODE chemistry and unit conversion reproduce the reference tightly. |
| ZDPlaskin example2 | final electron density ratio `1.0155`; reduced-field ratio `1.0051` | The example2-inspired DC-series case remains close on final density and field. |
| ZDPlaskin E/N formula check | local reported E/N `2.5743 Td`; saved E/N `2.561342 Td`; ratio `1.0051` | The local reported field is a circuit-derived field on the same physical footing. |
| Smoke CF4/O2/Ar case | final electron density `3.2007e14 m^-3`; mean electron energy `2.4083 eV` | End-to-end mixed chemistry, RF coupling, surfaces, and observables execute successfully. |

CRANE details:

- Local final electron density: `2.1735225936808305e19 m^-3`
- Reference final electron density: `2.1735277364398e19 m^-3`
- Local/reference ratio: `0.9999976339`
- Final Ar density relative error: `2.044e-12`

ZDPlaskin details:

- Final electron density ratio: `1.0155415965`
- Peak electron density ratio: `0.9013794634`
- Final Ar* ratio: `1.0213884975`
- Final Ar+ ratio: `1.0285347990`
- Final Ar2+ ratio: `1.0154557128`
- Final reduced-field ratio: `1.0050693208`
- Final absorbed power: `0.3204667388 W`

Smoke run details:

- Final electron density: `3.200743856319844e14 m^-3`
- Final mean electron energy: `2.4083241689741204 eV`
- Final self bias: `-62.722215231823135 V`
- Final plasma potential: `24.566711521794797 V`
- Final source RF absorbed power: `55.487908598829115 W`
- Final wafer-bias absorbed power: `0.007578006524567824 W`

## Reproducibility Notes

The comparison YAML files can be regenerated locally with the commands above.
They should remain ignored unless a future release explicitly wants fixed
benchmark artifacts as versioned reference data.
