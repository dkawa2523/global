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

## Tracked Review Artifacts

- `examples/outputs/external_benchmarks/diagnostic_suite/benchmark_metrics.csv`
- `examples/outputs/external_benchmarks/diagnostic_suite/benchmark_findings.md`

Detailed reports, figures, runtime profiles, and per-benchmark YAML files are regenerated artifacts and are intentionally not tracked.
