# External Benchmark Rerun Evaluation

Generated: 2026-06-22T12:40:32+09:00

## Scope

- CRANE TwoReactionArgon: local case rerun and compared with committed CRANE tutorial reference converted to SI.
- ZDPlaskin example2: local DC-series case rerun and compared with committed ZDPlaskin output / output-derived E/N table. The live ZDPlaskin executable is not invoked.
- PyGMol argon: executable PyGMol adapter rerun with the compact external surrogate model. This is a sanity comparison, not a matched LXCat/two-term physics equivalence test.

## Compact External Command Timing

| Command scope | Wall time s | Report |
| --- | ---: | --- |
| CRANE compact external report | 0.718190 | `tools/external_benchmarks/last_report_crane_rerun.yaml` |
| ZDPlaskin compact external report | 1.017677 | `tools/external_benchmarks/last_report_zdplaskin_rerun.yaml` |
| PyGMol compact external report | 3.794109 | `tools/external_benchmarks/last_report_pygmol_rerun.yaml` |
| All compact external reports | 4.249704 | `tools/external_benchmarks/last_report_rerun.yaml` |
| PyGMol full step report | 3.589512 | `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_report_rerun.yaml` |

## Local Solver Runtime Profile

| Problem | Median s | Warm median s | nfev | njev | nlu | Final ne m^-3 | Final mean eV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| crane_two_reaction_argon_ode_parity | 0.111632 | 0.110849 | 1634 | 1 | 86 | 2.173523e+19 | 1.380248e-13 |
| zdplaskin_example2_argon_dc_series | 0.472626 | 0.466899 | 3497 | 16 | 218 | 3.219636e+17 | 3.91089 |
| pygmol_argon_local_case | 2.79566 | 2.80461 | 5006 | 67 | 398 | 4.180334e+14 | 0.883917 |

## CRANE TwoReactionArgon Accuracy

| Quantity | Local | Reference | Ratio local/ref | Relative error |
| --- | ---: | ---: | ---: | ---: |
| final_Ar_density_m3 | 2.499998e+25 | 2.499998e+25 | 1 | 2.043032e-12 |
| final_Ar_plus_density_m3 | 2.173523e+19 | 2.173528e+19 | 0.999998 | 2.366088e-06 |
| final_electron_density_m3 | 2.173523e+19 | 2.173528e+19 | 0.999998 | 2.366088e-06 |

Assessment: excellent parity for the intended scalar ODE/unit-conversion benchmark; final electron-density relative error is 2.366088e-06.

## ZDPlaskin Example2 Accuracy

| Quantity | Ratio local/ZDPlaskin | Local value | Reference value |
| --- | ---: | ---: | ---: |
| final electron density | 1.01554 | 3.219636e+17 | 3.170363e+17 |
| peak electron density | 0.901379 | 3.219636e+17 | 3.571898e+17 |
| final Ar* density | 1.02139 | 2.057352e+17 | 2.014270e+17 |
| final Ar+ density | 1.02853 | 2.138665e+15 | 2.079332e+15 |
| final Ar2+ density | 1.01546 | 3.198249e+17 | 3.149570e+17 |
| final reduced field | 1.00507 | 2.57433 | 2.56134 |

Reduced-field footing: local reported / saved ZDPlaskin E/N ratio is 1.00507; same-voltage footing gives 2.58032 Td.

Assessment: close for this output-derived-table DC-series surrogate. It remains a committed-output comparison, not a live ZDPlaskin co-simulation.

## PyGMol Argon Step Accuracy

| Step | Powered | final ne ratio PyGMol/local | mean ne ratio | mean energy ratio | absorbed power ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| ignition | yes | 0.292151 | 0.0782156 | 1.65734 | 1.51301 |
| production | yes | 0.601862 | 0.298538 | 1.46034 | 1.36112 |
| afterglow | no | 276.848 | 3.58268 | 0.399441 |  |

Assessment: powered-step electron-density ratios remain within the configured sanity band, but PyGMol is intentionally not rate/geometry/wall-loss aligned with the local LXCat/two-term case in this compact report.

## Pass Summary

- zdplaskin_example2_argon_dc_series: passed=True
- crane_two_reaction_argon_ode_parity: passed=True
- pygmol_argon_global_model_sanity: passed=True

## Artifacts

- `tools/external_benchmarks/last_report_rerun.yaml`
- `tools/external_benchmarks/last_report_crane_rerun.yaml`
- `tools/external_benchmarks/last_report_zdplaskin_rerun.yaml`
- `tools/external_benchmarks/last_report_pygmol_rerun.yaml`
- `examples/outputs/external_benchmarks/current_runtime_profile.yaml`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_report_rerun.yaml`
- `examples/outputs/external_benchmarks/rerun_evaluation_report.md`
