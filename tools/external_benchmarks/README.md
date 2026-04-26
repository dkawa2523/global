# External Benchmarks

This directory holds comparison tooling for external reference codes. These
tools are intentionally outside the `plasma_global` package so ZDPlaskin,
CRANE, and future third-party benchmark handling do not become core solver
logic.

Current comparisons:

- `zdplaskin_example2.py` compares the local ZDPlaskin example2-inspired
  DC-series output with the committed ZDPlaskin example2 reference summary.
- `zdplaskin_eovern.py` compares reduced field on the physical
  `E/N = (V_gap / gap) / N_gas` footing used by the ZDPlaskin circuit output.
- `build_zdplaskin_rate_table.py` generates the wide diagnostic E/N-rate table
  used by pressure/voltage stress cases from the imported ZDPlaskin
  cross-section bundle. It is not the strict parity table.
- `build_zdplaskin_output_rate_table.py` generates the active ZDPlaskin
  example2 parity table by reverse-engineering E/N-rate coefficients from the
  committed ZDPlaskin output file.
- `zdplaskin_examples_inventory.py` inventories the public ZDPlaskin examples
  and separates implemented strict references from future benchmark candidates.
- `benchmark_data_manifest.yaml` lists the external and generated benchmark
  data sources, their roles, and their E/N coverage.
- `external_model_registry.py` writes the candidate registry for executable
  models, rate-table generators, and future circuit tools.
- `crane_two_reaction_argon.py` runs the local CRANE TwoReactionArgon case and
  compares it with the committed CRANE tutorial output converted to SI units.
- `pygmol_argon.py` runs the local pure-Ar LXCat case against an executable
  PyGMol compact Ar global model as a broad sanity check. It also supports
  `--rate-mode local-table`, where electron-impact rates are interpolated from
  local LXCat/two-term tables inside the external PyGMol adapter.
- `pygmol_same_footing.py` keeps the core solver untouched and cumulatively
  aligns PyGMol to the local run by using the same electron-impact rate tables,
  then the same absorbed-power waveform, then the same global ion wall-loss
  coefficient.
- `pygmol_argon_model.yaml` stores the PyGMol-only compact Arrhenius rates. It
  is external benchmark data, not core chemistry input.
- `pygmol_sensitivity.py` perturbs only the PyGMol external benchmark model and
  input mapping to diagnose whether rate, energy-loss, surface, geometry, or
  power assumptions dominate the PyGMol difference.
- `loki_o2_dc_glow_digitization.py` creates the external-only scaffold for the
  public LoKI O2 DC glow benchmark candidate, writes empty CSV templates and a
  schema file, and reports whether the candidate is still scaffold-only or
  ready for a real parity run.
- `loki_o2_dc_glow_benchmark.py` evaluates the digitized LoKI O2 DC glow
  first-wave reference set without touching `plasma_global` core code. It
  checks readiness, summarizes direct `Tg/Tnw/E/N` differences, and packages
  the figure 11 acceptance envelope as external reference data. When figures
  8, 9a, and 9b are digitized, the same tool expands to the extended-species
  pressure-sweep benchmark. When figures 1, 2, 3, 4, 7b, and 10 are also
  digitized, it adds the full radial-profile diagnostic benchmark.
- `benchmark_dashboard.py` runs the currently implemented external benchmarks,
  writes an integrated YAML/CSV summary, and exports overview graphs so the
  current benchmark status can be read in one place. The overview figure uses
  percent deviation from exact agreement for ZDPlaskin, percent relative error
  for CRANE, production-step electron-density mismatch for PyGMol, and
  reference-difference bars for the LoKI digitized benchmark. The companion
  PyGMol figure separates raw ratios from mismatch-to-local percentages so the
  rate/power/wall-loss gap can be read without mentally converting ratios. It
  also writes a paper-oriented Markdown report and a benchmark-coverage matrix
  so readers can see which physical axes are directly benchmarked, only
  diagnostically covered, or not constrained by each external reference. The
  report also includes a one-level pointer to the separate robustness suite so
  strict validation and applicability evidence are visible together.
- `run_external_benchmarks.py` runs the comparison workflows and writes
  a compact summary.
- `same_footing_assessment.py` writes a plain pass/fail-style verdict first,
  then separates details such as power parity, E/N formula checks, and scalar
  ODE chemistry parity.
- `robustness_sweep.py` runs the strict external references and additional
  pressure, electrical-backend, and chemistry stress cases. Stress cases check
  solver health and closure range; they are not accuracy validation without an
  external reference.
- `robustness_dashboard.py` turns the robustness sweep into reader-oriented
  outputs: a stress-overview figure, a reduced-field table-coverage figure, and
  a short applicability report. It keeps strict validation and stress evidence
  separate.

Run all external comparisons:

```powershell
py tools\external_benchmarks\run_external_benchmarks.py
```

Build the same-footing assessment:

```powershell
py tools\external_benchmarks\same_footing_assessment.py
```

Run the broader robustness sweep:

```powershell
py tools\external_benchmarks\robustness_sweep.py
```

Render the robustness/applicability dashboard:

```powershell
py tools\external_benchmarks\robustness_dashboard.py
```

Run only one:

```powershell
py tools\external_benchmarks\run_external_benchmarks.py --only crane
py tools\external_benchmarks\run_external_benchmarks.py --only pygmol
py tools\external_benchmarks\run_external_benchmarks.py --only zdplaskin
```

Run only the PyGMol comparison:

```powershell
py tools\external_benchmarks\pygmol_argon.py --rerun-local
```

Run PyGMol with the same local electron-collision rate tables:

```powershell
py tools\external_benchmarks\pygmol_argon.py --rate-mode local-table --rerun-local
```

or through the aggregate runner:

```powershell
py tools\external_benchmarks\run_external_benchmarks.py --only pygmol --pygmol-rate-mode local-table
```

Decompose the PyGMol difference with progressively tighter same-footing inputs:

```powershell
py tools\external_benchmarks\pygmol_same_footing.py --rerun-local
```

Run the PyGMol-only sensitivity diagnostic:

```powershell
py tools\external_benchmarks\pygmol_sensitivity.py
```

Refresh the external-model registry:

```powershell
py tools\external_benchmarks\external_model_registry.py
```

Create and validate the LoKI O2 DC glow digitization scaffold:

```powershell
py tools\external_benchmarks\loki_o2_dc_glow_digitization.py
```

Run the LoKI O2 DC glow digitized external benchmark:

```powershell
py tools\external_benchmarks\loki_o2_dc_glow_benchmark.py
```

Run the full external benchmark dashboard:

```powershell
py tools\external_benchmarks\benchmark_dashboard.py
```

The dashboard writes:

- `examples/outputs/external_benchmark_dashboard/external_benchmark_dashboard.yaml`
- `examples/outputs/external_benchmark_dashboard/external_benchmark_dashboard_metrics.csv`
- `examples/outputs/external_benchmark_dashboard/external_benchmark_overview.png`
- `examples/outputs/external_benchmark_dashboard/external_benchmark_pygmol_decomposition.png`
- `examples/outputs/external_benchmark_dashboard/external_benchmark_coverage_matrix.png`
- `examples/outputs/external_benchmark_dashboard/external_benchmark_report.md`
- `examples/outputs/external_benchmark_dashboard/external_benchmark_report_ja.md`

The robustness dashboard writes:

- `examples/outputs/robustness_sweep/robustness_dashboard.yaml`
- `examples/outputs/robustness_sweep/robustness_stress_overview.png`
- `examples/outputs/robustness_sweep/robustness_rate_table_coverage.png`
- `examples/outputs/robustness_sweep/robustness_applicability_report.md`

Run only the ZDPlaskin E/N comparison:

```powershell
py tools\external_benchmarks\zdplaskin_eovern.py
```

Regenerate the ZDPlaskin example2 external E/N-rate table:

```powershell
py tools\external_benchmarks\build_zdplaskin_output_rate_table.py
```

Regenerate the diagnostic table from the imported cross sections instead:

```powershell
py tools\external_benchmarks\build_zdplaskin_rate_table.py
```

Refresh the ZDPlaskin public-example inventory:

```powershell
py tools\external_benchmarks\zdplaskin_examples_inventory.py
```

These tools consume normal case YAML files and output files under `examples/`.
They may import `plasma_global.workflows.runner` to execute a local case, but
the core solver does not import these tools.
