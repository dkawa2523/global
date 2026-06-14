# External Benchmarks

This directory keeps external-reference comparisons out of the core
`plasma_global` package. These tools are useful for validation and example
maintenance, but they are not part of the normal CLI/API product surface.

Kept tools:

- `crane_two_reaction_argon.py` runs the local CRANE two-reaction argon case and
  compares it with the committed CRANE tutorial reference.
- `zdplaskin_example2.py` compares the local ZDPlaskin example2-inspired
  DC-series output with the committed reference summary.
- `zdplaskin_eovern.py` checks the reduced-field footing used by the ZDPlaskin
  circuit output.
- `build_zdplaskin_output_rate_table.py` regenerates the parity table derived
  from the committed ZDPlaskin output.
- `build_zdplaskin_rate_table.py` regenerates the wider diagnostic E/N-rate
  table from imported ZDPlaskin cross sections.
- `pygmol_argon.py` remains as an optional executable PyGMol sanity comparison.
- `run_external_benchmarks.py` runs the compact CRANE/ZDPlaskin/PyGMol set and
  writes an ignored YAML report.

Typical commands:

```powershell
py tools\external_benchmarks\run_external_benchmarks.py --only crane
py tools\external_benchmarks\run_external_benchmarks.py --only zdplaskin
py tools\external_benchmarks\run_external_benchmarks.py --only pygmol
py tools\external_benchmarks\build_zdplaskin_output_rate_table.py
```

Generated reports and run outputs are intentionally ignored by git.
