# Review Results

This page records historical review and verification results. It is not the
normative product specification. For current extension policy and user-facing
guidance, see [Architecture](ARCHITECTURE.md), [Physics and
Numerics](PHYSICS_NUMERICS.md), [Configuration](CONFIGURATION.md),
[Chemistry](CHEMISTRY.md), [External Swarm and Rate
Tables](SWARM_RATE_TABLES.md), and the [Extension Guide](EXTENSION_GUIDE.md).

Large generated run files under `examples/outputs/` are intentionally not
tracked. Lightweight external-benchmark review artifacts are tracked only under
`examples/outputs/external_benchmarks/diagnostic_suite/` and
`examples/outputs/external_benchmarks/remediation/` so reviewers have one
current benchmark surface without large HDF5 outputs or older dashboards.

## Verification Commands

Run on branch `update_2` after the product-core refactor:

```powershell
py -m pytest
py tools\external_benchmarks\diagnostic_suite.py
plasma-global validate examples/configs/case_smoke.yaml
plasma-global run examples/configs/case_smoke.yaml
py tools\external_benchmarks\crane_two_reaction_argon.py
plasma-global run examples/configs/case_zdplaskin_example2.yaml
py tools\external_benchmarks\zdplaskin_example2.py
py tools\external_benchmarks\zdplaskin_eovern.py
plasma-global list-backends --json
py -m mkdocs build --strict
```

## Gate Results

| Check | Result |
| --- | --- |
| Lean pytest gate | OK at the time of this historical review |
| Smoke case validation | OK |
| Smoke run | solver success, 200 samples |
| MkDocs strict build | OK |
| Generated output tracking | large run outputs stay ignored; current lightweight benchmark artifacts may be tracked |

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

The refactor kept the physical equations and state history intact while
removing broad non-core report surfaces and the missing-rate-table fallback.
The latter is an intentional safety change: `rate_table` fails fast when the
table file is absent or required electron-impact rate coefficients are missing.
Quick analytic checks should use the explicit `maxwell` backend.

## Numerics Review

The retained numerical path is a stiff ODE workflow with SciPy BDF, stable state
layout labels, projected positive state handling, and focused external parity
tests. The analytic Jacobian and CLI checker were removed from the normal
product surface because the former did not justify its maintenance cost.

## Architecture Review

The public product surface is now deliberately small:

- CLI: `validate`, `run`, `export-config`, `list-backends`
- API: `load_case_from_yaml`, `build_case`, `run_from_yaml`
- Config: schema version 2 case files only
- Output: summary, observables, effective case, resolved paths, optional HDF5

The old generated reports, dashboard outputs, broad benchmark dashboards,
legacy config normalization, registry maturity contracts, state manifests,
budget reports, and warning columns were removed to keep the core package
readable and maintainable.

## Comparison Results

Problem-scoped external benchmark diagnostics are generated with
`tools/external_benchmarks/diagnostic_suite.py`. The suite separates accuracy,
runtime, and diagnostic findings, then maps threshold misses to likely core-code
areas. Lightweight artifacts are written under
`examples/outputs/external_benchmarks/diagnostic_suite/`.

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
Lightweight external-benchmark review artifacts (`md`, `csv`, `yaml`, `png`)
are limited to the current diagnostic/remediation output directories. Large run
products such as `solution.h5` remain regenerated artifacts.

Current diagnostic-suite findings:

- `ZDPlaskin-1 / peak_e_relative_error`: final values agree, but the peak
  electron density differs beyond the 5% transient threshold. Next checks should
  isolate rate-table relaxation, circuit loading, and diffusion-loss closure.
- `PyGMol-1`: the current PyGMol comparison is a useful sanity check only.
  Precision parity is explicitly scoped out of the pass/fail gate until a true
  single-cylinder, same-chemistry, same-wall-loss local case exists.
- `PyGMol-Precision-1`: a separate tools-only same-footing harness compares the
  compact PyGMol Ar equations against a local implementation with identical
  geometry, chemistry, power, wall return, initial state, and sample times. The
  current run passes with max final relative error `1.09e-4` and max waveform
  NRMSE `2.32e-5`.

PyGMol-specific figures are generated under
`examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/`. They
include powered-step sanity ratios, production-vs-PyGMol final values,
same-footing species/energy/power overlays, precision error bars, and a scope
matrix explaining which comparison axes are aligned.

Remediation actions can be regenerated with:

```powershell
py tools\external_benchmarks\benchmark_remediation.py
```

Current remediation result:

- ZDPlaskin peak-density sensitivity was run over electron-energy relaxation,
  wall-loss frequency, and circuit electron mobility. The best simple variant
  was `wall_0p5x`, improving peak relative error from `9.86%` to `6.85%`, but
  it still did not meet the 5% gate. The recommended next step is not scalar
  tuning; it is a transient decomposition benchmark with E/N, voltage/current,
  mean energy, and per-reaction Ar2+ source/loss terms around the first 20 us.
- PyGMol precision validation is not counted as a current failure. It remains a
  scoped tools benchmark rather than a production-core claim. The same-footing
  harness checks compact-equation parity, while the production LXCat/two-zone
  comparison remains sanity only.
- The model-implementation candidate report separates generally useful
  low-pressure plasma improvements from benchmark-fit changes. The P1 items
  now have initial implementations: reaction source/loss observables,
  rate-table boundary/provenance diagnostics, and standardized electrical
  waveform observables. Re-running the diagnostic suite kept the existing
  accuracy interpretation unchanged: CRANE and final ZDPlaskin quantities pass,
  ZDPlaskin peak timing remains a transient-model finding, and PyGMol passes as
  a sanity/scoping benchmark rather than a precision claim.
- Cleanup findings are written to
  `examples/outputs/external_benchmarks/remediation/code_simplification_findings.md`.
  The current cleanup removes stale benchmark dashboards, a legacy external
  benchmark runner, one-off plot scripts, and the empty core benchmark runner,
  while keeping the current diagnostic/remediation suite as the single review
  surface.
