# External Table Builders

This directory is outside the core `plasma_global` package. It keeps small
helpers for regenerating example ZDPlaskin-derived rate tables.

```powershell
py tools\external_benchmarks\build_zdplaskin_output_rate_table.py
py tools\external_benchmarks\plot_pygmol_benchmarks.py `
  --sanity-summary path\to\comparison_pygmol_summary.csv `
  --precision-timeseries path\to\pygmol_precision_timeseries.csv `
  --precision-metrics path\to\pygmol_precision_metrics.csv
```

- `build_zdplaskin_output_rate_table.py` regenerates the active E/N table and
  circuit waveform derived from the saved ZDPlaskin example2 output.
- `plot_pygmol_benchmarks.py` is a plotting helper for already-generated
  PyGMol sanity and same-footing precision CSV files. Inputs are explicit
  command-line arguments because local benchmark output directories are
  disposable and ignored by git. It keeps the sanity check to one scale plot
  and uses precision overlays plus error bars for same-footing comparisons.

Generated tables, CSV sidecars, and run outputs should be regenerated locally
rather than committed as benchmark reports.
