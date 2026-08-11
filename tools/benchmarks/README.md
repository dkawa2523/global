# External Table Builders

This directory is outside the runtime package. It contains the one offline
builder needed to regenerate the canonical ZDPlaskin-derived v3 rate table.

```powershell
py -m tools.benchmarks.build_zdplaskin_rate_table
```

- Input: `benchmarks/raw/zdplaskin_example2_eovern/`.
- Output: `examples/v3/chemistry/zdplaskin_example2/tables/zdplaskin_example2_eovern_rates.h5`.
- Validation and conversion are performed by the strict v3 rate-table importer.

The raw CSVs and canonical HDF5 are committed. Disposable run outputs and plots
are not benchmark reference data.
