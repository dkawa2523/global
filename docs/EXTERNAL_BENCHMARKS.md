# External Benchmarks

External benchmarks live under `tools/external_benchmarks/` and do not belong
to the core solver package. This keeps reference-code quirks, imported outputs,
and optional dependencies out of production plasma-model logic.

## Current runnable checks

| Benchmark | Tool | Role | Verdict meaning |
| --- | --- | --- | --- |
| ZDPlaskin example2 | `zdplaskin_example2.py`, `zdplaskin_eovern.py` | strict same-footing Ar DC circuit/composition check | final E/N, circuit quantities, and species are compared against committed ZDPlaskin output |
| CRANE TwoReactionArgon | `crane_two_reaction_argon.py` | strict scalar ODE chemistry check | reaction assembly, units, and stiff ODE integration match the public tutorial output |
| PyGMol Ar | `pygmol_argon.py` + `pygmol_argon_model.yaml` | executable global-model sanity check | powered-step electron density is in a broad order/trend range; rates are intentionally not aligned with the local LXCat case |
| PyGMol Ar same-rate | `pygmol_argon.py --rate-mode local-table` | external same electron-collision-rate check | PyGMol electron-impact coefficients are interpolated from local LXCat/two-term rate tables; geometry, wall, flow, and equation-form differences remain |
| PyGMol Ar same-footing decomposition | `pygmol_same_footing.py` | external diagnostic only | cumulatively fixes same local electron-impact rates, same absorbed-power waveform, then same global ion wall-loss coefficient to isolate the dominant difference |
| PyGMol sensitivity | `pygmol_sensitivity.py` | external diagnostic only | perturbs PyGMol-only rate, loss, geometry, surface, and power assumptions without changing core physics |

Run the implemented external checks:

```powershell
py tools\external_benchmarks\run_external_benchmarks.py --skip-zdplaskin-run
```

Run the integrated external benchmark dashboard and export overview graphs:

```powershell
py tools\external_benchmarks\benchmark_dashboard.py
```

Run the robustness/applicability dashboard for the added stress cases:

```powershell
py tools\external_benchmarks\robustness_dashboard.py
```

The dashboard currently exports two reader-oriented figures:

- `external_benchmark_overview.png`: ZDPlaskin exact-match deviation, CRANE
  final-state relative error, PyGMol production-step electron-density mismatch,
  and LoKI max reference differences in one page.
- `external_benchmark_pygmol_decomposition.png`: PyGMol same-footing raw ratios
  plus mismatch-from-local percentages so the effect of same rates, same power,
  and same wall loss can be read stage by stage.
- `external_benchmark_coverage_matrix.png`: benchmark-by-benchmark coverage of
  circuit/power, E/N, species, wall loss, profiles, and scalar ODE validation,
  with direct versus diagnostic coverage distinguished.
- `external_benchmark_report.md`: a paper-oriented narrative summary that
  explains the comparison target, footing, quantitative result, and physical
  interpretation of each benchmark. The report now includes a one-level
  cross-reference to the robustness/applicability suite so strict validation
  and broader operating-envelope evidence can be read side by side without
  conflating them.
- `external_benchmark_report_ja.md`: a Japanese paper-style companion report
  with the same benchmark structure, plus a concise model-equation summary and
  a future-work section focused on remaining low-pressure-plasma modeling gaps.

The robustness/applicability dashboard exports a separate set of files under
`examples/outputs/robustness_sweep/`:

- `robustness_stress_overview.png`: added pressure/electrical/chemistry stress
  cases in one page, focused on boundedness and practical usability rather than
  exact external parity.
- `robustness_rate_table_coverage.png`: observed E/N ranges against the active
  widened diagnostic table for the ZDPlaskin-derived stress family.
- `robustness_applicability_report.md`: short narrative summary explaining what
  these extra cases do and do not justify scientifically.

Refresh the candidate registry and ZDPlaskin public-example inventory:

```powershell
py tools\external_benchmarks\external_model_registry.py
py tools\external_benchmarks\zdplaskin_examples_inventory.py
```

## Candidate handling

- ZDPlaskin `example3` is useful, but it prescribes electron density and E/N.
  Treat it as a future driven-chemistry regression, not as a self-consistent
  global-discharge benchmark.
- ZDPlaskin `example1` is not currently a strict reference because the public
  repository does not include a clean committed time-history output for that
  folder.
- LoKI-B and ThunderBoltz should first be connected as rate/transport-table
  generators. They are valuable, but they do not directly validate the whole
  reactor global model without a separate discharge closure.
- LoKI O2 DC glow is the next public low-pressure benchmark candidate. The
  benchmark paper and LoKI tool description are public, but we have not yet
  located a machine-readable reference output bundle from official sources, so
  it remains a digitized-reference benchmark rather than a strict raw-output
  parity case. The current code-independent candidate contract is stored in
  `examples/external/loki_o2_dc_glow_candidate.yaml`. The external-only
  digitization scaffold and readiness report are created by
  `tools/external_benchmarks/loki_o2_dc_glow_digitization.py`, and the
  digitized benchmark report is written by
  `tools/external_benchmarks/loki_o2_dc_glow_benchmark.py`. The current
  external benchmark now runs through the full diagnostic stage, including the
  populated figures 1, 2, 3, 4, 7b, and 10 profile datasets.
- PLASIMO has public global-model pages and downloadable published input
  archives, but the official evaluation version does not save output to disk and
  runtime access still goes through login/request gates.
- Quantemol Global Model is publicly visible online and its chemistry export API
  is documented, but we have not yet confirmed a public machine-readable export
  path for global-model result files.
- plasma-R has a public site and literature references for low-pressure wall
  coupled reactor cases, but trying the software requires contacting the
  developers and no public benchmark output bundle has been located.
- ngspice should remain a future external circuit waveform source. The current
  code already has one-way `external_circuit_table` ingestion; fully coupled
  plasma-to-circuit feedback should wait for a real benchmark need.

PyGMol-specific Arrhenius rates are external-benchmark data only. They are
stored in `tools/external_benchmarks/pygmol_argon_model.yaml` and should not be
imported by `plasma_global` core modules.

For a closer same-rate comparison, run:

```powershell
py tools\external_benchmarks\run_external_benchmarks.py --only pygmol --pygmol-rate-mode local-table
```

This writes:

- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_local_rate_table_summary.csv`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_local_rate_table_model.yaml`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_local_rate_table_metadata.yaml`

`local-table` mode subclasses PyGMol's equation object inside the external
tool and overrides electron-impact rate coefficients from local LXCat/two-term
tables. It is intentionally outside the core package.

To decompose the same-rate PyGMol difference on a progressively tighter
footing, run:

```powershell
py tools\external_benchmarks\pygmol_same_footing.py --rerun-local
```

This writes:

- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_same_footing_decomposition.csv`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_same_footing_decomposition.yaml`

The stages are cumulative:

- same local electron-impact rate tables only,
- same rates plus the local absorbed-power waveform,
- same rates, same absorbed power, and the local global ion wall-loss coefficient.

Only the external PyGMol adapter is modified. These outputs diagnose why two
global models differ; they are not calibration data for core chemistry or wall
physics.

The PyGMol sensitivity tool writes:

- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_sensitivity_summary.csv`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_sensitivity_steps.csv`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_sensitivity.yaml`

## Output files

- `tools/external_benchmarks/last_report.yaml`
- `tools/external_benchmarks/external_model_registry_report.yaml`
- `examples/external/zdplaskin_examples_inventory.yaml`
- per-case comparison files under `examples/outputs/`
