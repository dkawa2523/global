# External Benchmarks

This directory keeps external-reference comparisons out of the core
`plasma_global` package. These tools are validation and review aids, not part of
the normal CLI/API product surface.

The main entry point is now `diagnostic_suite.py`. It reads `manifest.yaml`,
runs problem-scoped comparisons, writes metrics/figures, and maps any misses to
suspected core-code areas.

```powershell
py tools\external_benchmarks\diagnostic_suite.py
py tools\external_benchmarks\diagnostic_suite.py --only CRANE-1
py tools\external_benchmarks\diagnostic_suite.py --skip-case-runs --repetitions 1
py tools\external_benchmarks\benchmark_remediation.py
py tools\external_benchmarks\plot_pygmol_benchmarks.py
```

Benchmark problems:

- `CRANE-1`: CRANE TwoReactionArgon constant-rate ODE and SI-conversion parity.
- `ZDPlaskin-1`: ZDPlaskin example2 final/transient species and circuit checks.
- `ZDPlaskin-2`: output-derived ZDPlaskin E/N rate-table schema and interpolation.
- `PyGMol-1`: PyGMol pure-Ar sanity comparison plus precision-claim scoping.
- `PyGMol-Precision-1`: tools-only same-footing compact-Ar precision harness.
- `SWARM-1`: external swarm-table ingestion contract for `rate_table`.
- `Runtime-1`: local wall time and solver-stat context.

Supporting tools:

- `build_zdplaskin_output_rate_table.py` regenerates the parity table derived
  from the committed ZDPlaskin output.
- `build_zdplaskin_rate_table.py` regenerates the wider diagnostic E/N-rate
  table from imported ZDPlaskin cross sections.
- `benchmark_remediation.py` turns diagnostic findings into concrete action
  plans. It currently runs ZDPlaskin peak-density sensitivity checks, writes
  the PyGMol same-footing gap analysis, and separates core low-pressure-model
  implementation candidates from rejected benchmark-fit tuning. PyGMol precision
  parity for the production LXCat/two-zone case is not claimed; the separate
  `pygmol_precision.py` harness checks PyGMol's compact equations on identical
  footing.
- `plot_pygmol_benchmarks.py` creates PyGMol-specific figures that separate the
  production-case sanity comparison from the same-footing precision harness.
- The diagnostic suite also creates cross-benchmark review figures:
  `benchmark_threshold_margin.png` for quantitative error margin against
  acceptance thresholds, `benchmark_claim_support_matrix.png` for which
  external code supports which physics/implementation claim, and
  `benchmark_status_by_software.png` for pass/attention counts by external
  software.
- `CRANE-1_density_timeseries.png` and
  `ZDPlaskin-1_density_timeseries.png` add electron-density and Ar+ density
  time-series views with the committed external final/peak markers where full
  external species waveforms are not available.

Lightweight review artifacts are kept under
`examples/outputs/external_benchmarks/diagnostic_suite/` and
`examples/outputs/external_benchmarks/remediation/`. Older dashboard and
one-off plot artifacts were removed so reviewers have one current benchmark
surface. Large run products such as HDF5 solutions remain regenerated
artifacts.
