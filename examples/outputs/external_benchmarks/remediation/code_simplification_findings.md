# Code Simplification Findings

This file records simplification and deletion findings from the external-benchmark cleanup. It separates completed cleanup from future work so benchmark-specific concerns do not leak into the core model.

| Area | Status | Issue | Action | Next step |
| --- | --- | --- | --- | --- |
| core_diagnostics | fixed | GasPhaseCore mixed RHS terms and diagnostic budget reconstruction. | Added shared GasReactionTerm and IonWallLossTerm flows for RHS and budgets. | Keep diagnostic budgets derived from the same physical terms used by the RHS. |
| benchmark_tools | fixed | Legacy external benchmark runner and one-off plot scripts duplicated diagnostic_suite outputs. | Deleted the old runner/dashboard/time-series plotting entry points. | Use diagnostic_suite.py and benchmark_remediation.py as the only external-benchmark review surfaces. |
| tracked_artifacts | fixed | Older dashboard/agreement/time-series artifacts competed with current problem-scoped figures. | Removed stale tracked artifacts outside diagnostic_suite/ and remediation/. | Track only lightweight current artifacts and regenerate large run outputs. |
| observables_width | fixed | Detailed budget columns are useful for validation but can make normal observables wide. | Added outputs.diagnostics.budgets and kept detailed reaction/source/loss columns opt-in. | Keep normal cases compact and enable budgets only for validation or benchmark cases. |
| electrical_contract | fixed | PowerResult.port_observables was a generic nested dict. | Introduced ElectricalPortSnapshot with a stable observables serializer. | Keep backend-specific electrical details typed before flattening to CSV columns. |
| pygmol_precision | fixed | PyGMol production-case precision and compact-equation parity needed separate claims. | Added PyGMol-Precision-1 as a tools-only same-footing harness and kept PyGMol-specific work out of core physics. | Keep PyGMol-1 as sanity/scoping and PyGMol-Precision-1 as compact-equation parity only. |
