# Model Implementation Candidates

This report separates low-pressure-plasma model improvements from benchmark fitting. The benchmark evidence is used only to identify missing observability, data contracts, or validation structure that would be useful beyond one external code.

## Implement

### reaction_source_loss_diagnostics (P1)

- Status: implemented_initial
- Scope: core_optional_diagnostic
- External basis: ZDPlaskin transient outputs and CRANE reaction-ODE parity both make reaction-level balances explicit.
- Current evidence: ZDPlaskin peak-density mismatch is not fixed by scalar sensitivity alone; the next discriminant is which Ar2+ and electron-energy terms create the early transient.
- Low-pressure model value: Improves mechanism debugging, chemistry validation, conservation audits, and transient interpretation for low-pressure plasma cases.
- Concrete method: Add an opt-in per-zone reaction ledger for species source/loss and electron-energy loss terms, sampled into observables or benchmark diagnostics without changing default RHS behavior.
- First validation gate: ZDPlaskin-1 early-time peak window can identify dominant source/loss terms; CRANE-1 balances stay conserved.
- Owner area: plasma_global/physics/gas_phase_core.py; plasma_global/observables/defaults.py
- Why this is not benchmark tuning: Does not change reaction rates or fit a parameter; it exposes the physical budget already implied by the model.

### rate_table_boundary_provenance_diagnostics (P1)

- Status: implemented_initial
- Scope: core_runtime_safety
- External basis: BOLSIG+/LoKI-style swarm tables are defined by grids, units, transport coefficients, rates, and source provenance.
- Current evidence: SWARM-1 proves ingestion and finite interpolation, but boundary clipping, table coverage, and source metadata are not yet promoted to runtime diagnostics.
- Low-pressure model value: Prevents silent use of out-of-domain electron kinetics, which is central to low-pressure global-model validity.
- Concrete method: Report lookup coordinate, clipping flags/counts, table min/max coverage, source tool/version, and units through diagnostics/provenance; keep interpolation math unchanged.
- First validation gate: SWARM-1 detects out-of-range queries and records table provenance while preserving existing in-range parity.
- Owner area: plasma_global/eedf/table.py; scripts/build_rate_table_h5.py; plasma_global/workflows/runner.py
- Why this is not benchmark tuning: Does not alter coefficients to match BOLSIG+ or LoKI; it makes invalid table use visible.

### electrical_waveform_observables (P1)

- Status: implemented_initial
- Scope: core_observability
- External basis: ZDPlaskin circuit comparisons center on voltage, current, E/N, conductance/loading, and absorbed power waveforms.
- Current evidence: Current comparisons can use final E/N and a limited waveform metric, but the model should expose the full circuit state for any pulsed/DC low-pressure case.
- Low-pressure model value: Makes power coupling, field history, and plasma loading auditable without tying the model to one external solver.
- Concrete method: Standardize optional PowerResult waveform observables: source voltage, gap voltage, current, plasma conductance, E/N, absorbed power, and port identifier.
- First validation gate: ZDPlaskin-1 circuit waveform metrics can separate electrical mismatch from chemistry mismatch.
- Owner area: plasma_global/electrical/base.py; plasma_global/electrical/dc_series.py; plasma_global/observables/defaults.py
- Why this is not benchmark tuning: Adds physical observability of the electrical closure; it does not calibrate circuit parameters.

### transient_window_budget_sampler (P2)

- Status: implemented_initial
- Scope: analysis_workflow
- External basis: ZDPlaskin and CRANE diagnostics are most useful when evaluated around physically important transient windows, not only at final state.
- Current evidence: The ZDPlaskin electron-density peak time is far from the local peak time even when final values are closer.
- Low-pressure model value: Supports low-pressure pulsed and afterglow validation where peak timing and energy relaxation are the physics of interest.
- Concrete method: Add a reusable sampler that records selected budgets around configured time windows: mean energy, E/N, V/I/P, per-reaction sources, and wall-loss terms.
- First validation gate: ZDPlaskin-1 peak-window report explains whether the early mismatch is electrical, table-relaxation, chemistry, or wall-loss dominated.
- Owner area: tools/external_benchmarks/diagnostic_suite.py; plasma_global/workflows/runner.py
- Why this is not benchmark tuning: Narrows cause of model discrepancy; it does not force the model toward a reference trace.

### swarm_interchange_metadata_contract (P2)

- Status: pending
- Scope: core_data_contract
- External basis: BOLSIG+ and LoKI outputs carry assumptions about gas composition, E/N or mean-energy grids, cross-section sources, units, and transport definitions.
- Current evidence: The current HDF5 table path is compact and useful, but the model cannot yet assert whether two swarm tables are semantically comparable.
- Low-pressure model value: Improves reproducibility and prevents mixing incompatible electron-kinetics data in production low-pressure studies.
- Concrete method: Extend the table manifest with required grid variable, gas composition, transport definitions, source tool/version, cross-section provenance, and optional uncertainty fields.
- First validation gate: SWARM-1 rejects missing required metadata for precision claims while accepting legacy compact tables as compatibility inputs.
- Owner area: scripts/build_rate_table_h5.py; plasma_global/eedf/table.py; docs
- Why this is not benchmark tuning: Strengthens data semantics; it does not modify rate values.

### same_footing_external_fixture_generation (P2)

- Status: implemented
- Scope: benchmark_infrastructure_not_core_physics
- External basis: PyGMol precision comparison requires identical geometry, chemistry, power deposition, and wall-loss definitions.
- Current evidence: PyGMol-Precision-1 now passes as a tools-only same-footing compact-Ar harness; PyGMol-1 remains the production-case sanity comparison.
- Low-pressure model value: Prevents overstating accuracy and creates a clean fixture for testing low-pressure global-model assumptions.
- Concrete method: Implemented tools/external_benchmarks/pygmol_precision.py to generate an identical compact-Ar ODE fixture from the PyGMol model file.
- First validation gate: PyGMol-Precision-1 passes max final relative error and max waveform NRMSE thresholds while keeping production-core claims scoped out.
- Owner area: tools/external_benchmarks/pygmol_precision.py; tools/external_benchmarks/pygmol_argon.py
- Why this is not benchmark tuning: Keeps comparison fair without adding PyGMol-specific behavior to the production solver.

## Do Not Implement As Core Model Changes

### benchmark_tuned_wall_loss_frequency

- Reason: The sensitivity run improved the ZDPlaskin peak error but did not satisfy the peak gate or fix peak timing.
- Risk: Would hide model-form or transient-coupling errors behind a case-specific constant.
- Acceptable alternative: Expose wall-loss budgets and validate the selected closure before changing any default.

### benchmark_tuned_energy_relaxation_time

- Reason: Changing one relaxation time cannot distinguish rate-table coverage, circuit loading, and chemistry source terms.
- Risk: Would make one ZDPlaskin trace look closer while reducing physical meaning for other low-pressure regimes.
- Acceptable alternative: Add transient-window source/loss and energy-budget diagnostics, then justify any new closure physically.

### pygmol_specific_core_mode

- Reason: PyGMol is useful for a same-footing global-model check, not as a production-mode target.
- Risk: Would mix benchmark scaffolding with the solver API and obscure the model assumptions.
- Acceptable alternative: Generate same-footing benchmark fixtures under tools/external_benchmarks only.

### live_external_solver_calls_inside_rhs

- Reason: External tools are references and data generators, not runtime dependencies for the compact model ODE.
- Risk: Would reduce reproducibility, speed, and portability of the low-pressure model.
- Acceptable alternative: Import external outputs into explicit, versioned tables or reference artifacts.

## Recommended Work Order

1. Add reaction source/loss diagnostics and electrical waveform observables so ZDPlaskin transient misses can be decomposed without fitting constants.
2. Add rate-table boundary/provenance diagnostics so BOLSIG+/LoKI-derived data cannot be used outside its physical domain silently.
3. Add transient-window budget sampling to separate peak-timing physics from final-state agreement.
4. Keep the PyGMol same-footing fixture under benchmark tools, with PyGMol-specific adapters out of production solver code.
5. Re-run CRANE-1, ZDPlaskin-1/2, PyGMol-1, SWARM-1, and Runtime-1 after each P1 implementation.
