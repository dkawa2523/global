# Benchmark assets

`references/` contains published-code outputs used for cross-code regression.
`raw/` contains immutable source-derived tabular inputs. Runtime-ready artifacts
remain with their v3 chemistry bundle so a case has one canonical table path.

The ZDPlaskin-derived HDF5 is rebuilt offline with:

```powershell
py -m tools.benchmarks.build_zdplaskin_rate_table
```

The checksum and cross-code tests verify that the strict importer reproduces
the committed HDF5 byte-for-byte. These references are code-to-code regression
fixtures, not independent experimental validation.

`v2_baseline.yaml` preserves the legacy suite result and the three recorded
case comparisons. Its `capture_completeness` mapping is part of the artifact
contract: conservation balances, external-table read counts, and isolated RHS
timing were not recorded before the v2 runtime was removed. They are therefore
marked `not_recorded` instead of being reconstructed or estimated.
