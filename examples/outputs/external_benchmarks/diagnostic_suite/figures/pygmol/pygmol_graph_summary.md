# PyGMol Benchmark Graphs

These figures separate the production-case PyGMol sanity comparison from the same-footing precision harness.

## Interpretation

- `PyGMol-1` is a sanity comparison: it checks scale only and does not claim precision because the production model and PyGMol compact model are not same-footing.
- `PyGMol-Precision-1` is a tools-only compact-equation parity check: geometry, chemistry, power, wall return, initial state, and sample times are aligned.
- Current precision max final relative error: `0.000109`.
- Current precision max waveform NRMSE: `2.32e-05`.

## Figures

- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_sanity_density_ratio_by_powered_step.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_sanity_powered_final_values.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_precision_species_timeseries_overlay.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_precision_energy_power_overlay.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_precision_error_metrics.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_benchmark_scope_matrix.png`
