# Benchmark Findings

This report lists benchmark threshold failures, warnings, and action-required items that should guide future core-code fixes.

## ZDPlaskin-1 / peak_e_relative_error

- Status: `fail`
- Severity: `medium`
- Value: `0.09862053657390668`; threshold: `<=0.05`
- Diagnostic category: `zdplaskin_species_transient`
- Suspected modules: plasma_global/eedf/table.py; plasma_global/electrical/dc_series.py; plasma_global/physics/gas_phase_core.py
- Recommended next check: Compare peak timing, rate-table relaxation, circuit loading, and diffusion-loss closure separately.
