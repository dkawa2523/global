# External Benchmark Diagnostic Suite

This suite separates benchmark claims by external software and problem setup, then maps threshold misses to suspected core-code areas.

## Benchmark Status

| Benchmark | Software | Claim | Status | Findings |
| --- | --- | --- | --- | --- |
| CRANE-1 | CRANE | strong | pass | 0 |
| ZDPlaskin-1 | ZDPlaskin | scoped | attention | 1 |
| ZDPlaskin-2 | ZDPlaskin | strong | pass | 0 |
| PyGMol-1 | PyGMol | sanity | pass | 0 |
| PyGMol-Precision-1 | PyGMol | scoped | pass | 0 |
| SWARM-1 | BOLSIG+/LoKI-compatible table | strong | pass | 0 |
| Runtime-1 | this code | context | pass | 0 |

## Generated Figures

- `examples/outputs/external_benchmarks/diagnostic_suite/figures/benchmark_threshold_margin.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/benchmark_claim_support_matrix.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/benchmark_status_by_software.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/CRANE-1_metrics.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/ZDPlaskin-1_metrics.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/ZDPlaskin-2_metrics.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/PyGMol-1_metrics.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/PyGMol-Precision-1_metrics.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/SWARM-1_metrics.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/CRANE-1_density_timeseries.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/ZDPlaskin-1_density_timeseries.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/ZDPlaskin-2_rate_table_vs_EoverN.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/Runtime-1_solver_stats.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_sanity_density_ratio_by_powered_step.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_sanity_powered_final_values.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_precision_species_timeseries_overlay.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_precision_energy_power_overlay.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_precision_error_metrics.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/pygmol/pygmol_benchmark_scope_matrix.png`
- `examples/outputs/external_benchmarks/diagnostic_suite/figures/failure_to_module_heatmap.png`

## Generated Tables

- `examples/outputs/external_benchmarks/diagnostic_suite/benchmark_metrics.csv`
- `examples/outputs/external_benchmarks/diagnostic_suite/benchmark_findings.md`
- `examples/outputs/external_benchmarks/diagnostic_suite/benchmark_report.md`
- `examples/outputs/external_benchmarks/diagnostic_suite/diagnostic_report.yaml`
