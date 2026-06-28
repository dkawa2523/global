# ZDPlaskin Example2 Ar Bundle

This bundle imports the Ar cross sections and compact Ar/Ar*/Ar2+ chemistry
from `https://github.com/Hemadityamalla/ZDPlaskin`, folder `example2`.

The local solver now has a reduced DC series-circuit backend and an external
E/N-rate table for this case. The active parity table is derived from the
committed ZDPlaskin example2 output, so it carries the rate coefficients and
superelastic/de-excitation behavior present in that saved run. The numerical
integrator and software architecture remain local to this code.

Input files are split by model type:

- `cross_sections_manifest.yaml`: tabulated electron-collision data.
- `electron_impact_models.yaml`: electron-impact rates evaluated from those
  cross sections.
- `gas_rate_models.yaml`: empirical constant, gas-temperature, or
  electron-temperature rate expressions without cross-section data.
- `energy_loss_models.yaml`: per-event electron energy losses.
- `tables/zdplaskin_example2_eovern_rates.h5`: E/N-gridded electron-impact
  rate table consumed by the `rate_table` EEDF backend. This is the active
  parity table.
- `tables/zdplaskin_example2_eovern/`: CSV and metadata sidecar files used to
  inspect the generated table.
- `tables/zdplaskin_example2_wide_eovern_rates.h5`: wider E/N table
  generated from the imported
  ZDPlaskin cross sections by the local two-term-like solver.
- `tables/zdplaskin_example2_wide_eovern/`: CSV and metadata sidecars for the
  wide table.

The active table is regenerated with:

```powershell
py tools\external_benchmarks\build_zdplaskin_output_rate_table.py
```

`tools/external_benchmarks/build_zdplaskin_rate_table.py` remains available as
a local two-term/cross-section table generator. Its default output is the wide
table, not the strict ZDPlaskin parity table.
