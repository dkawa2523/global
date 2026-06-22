# Benchmark Graph Summary

Generated figures:

- `examples/outputs/external_benchmarks/figures/accuracy_crane_this_code_vs_external.png`
- `examples/outputs/external_benchmarks/figures/accuracy_zdplaskin_this_code_vs_external.png`
- `examples/outputs/external_benchmarks/figures/accuracy_pygmol_external_over_this_code.png`
- `examples/outputs/external_benchmarks/figures/speed_this_code_runtime.png`
- `examples/outputs/external_benchmarks/figures/speed_external_vs_this_code_context.png`

Interpretation notes:

- CRANE accuracy plot: the baseline/zero line is the external CRANE reference; bars are this code deviation from that reference.
- ZDPlaskin accuracy plot: the baseline/zero line is the external ZDPlaskin committed output; bars are this code deviation from that output.
- PyGMol accuracy plot: bars are external PyGMol result divided by this code result; 1.0 means equal.
- This-code runtime plot contains only this code timings.
- Runtime provenance plot contains the one measured external runtime, PyGMol model.run(), and separates it from this code and full report timing.
- CRANE and ZDPlaskin live executable runtimes were not measured.
