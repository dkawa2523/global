# External Table Builders

This directory is outside the core `plasma_global` package. It keeps small
helpers for regenerating example ZDPlaskin-derived rate tables.

```powershell
py tools\external_benchmarks\build_zdplaskin_output_rate_table.py
py tools\external_benchmarks\build_zdplaskin_rate_table.py
```

- `build_zdplaskin_output_rate_table.py` regenerates the active E/N table and
  circuit waveform derived from the saved ZDPlaskin example2 output.
- `build_zdplaskin_rate_table.py` regenerates the wider E/N table
  from imported cross sections with the local internal two-term-like solver.

Generated tables, CSV sidecars, and run outputs should be regenerated locally
rather than committed as benchmark reports.
