# Benchmark Remediation Actions

This file turns diagnostic-suite findings into concrete next actions and small experiments.

## ZDPlaskin-1 Peak Electron Density

- Baseline peak relative error: `0.09862`.
- Best tested variant: `wall_0p5x` on `wall_loss_frequency_s` with peak relative error `0.06848`.
- Concrete measure: do not tune a scalar parameter blindly; the tested simple parameters did not bring peak density under the 5% gate.
- Implementation method: add a dedicated transient decomposition benchmark that logs rate-table mean energy, E/N, circuit voltage/current, and individual Ar2+ source/loss terms around the external peak time.
- Likely code areas: `plasma_global/eedf/table.py`, `plasma_global/electrical/dc_series.py`, and `plasma_global/physics/gas_phase_core.py`.

Recommended ZDPlaskin work order:

1. Add per-reaction source/loss observables for the ZDPlaskin case, limited to benchmark diagnostics.
2. Add a circuit waveform comparison around the first 20 microseconds, where the peak mismatch is created.
3. Rebuild the output-derived rate table from the full ZDPlaskin output if more transient columns become available.
4. Only after the decomposition identifies a single cause, adjust `rate_table` relaxation, `dc_series_circuit`, or wall-loss closure.

## PyGMol-1 Precision Gap Analysis

The current PyGMol comparison remains useful as a sanity check, but not as a precision benchmark. Required concrete measures:

### geometry

- Current state: 2 zones, 3 surfaces, edges=True
- Needed for precision: single equivalent cylinder with one volume and one surface-loss area
- Method: Add a generated same-footing local case or adapter output that collapses the local reactor to PyGMol geometry.
- Code area: plasma_global/reactor/models.py; tools/external_benchmarks/pygmol_argon.py

### chemistry

- Current state: local LXCat mechanism vs PyGMol model pygmol_minimal_argon_arrhenius_surrogate
- Needed for precision: same species, same ionization/excitation/elastic rates, same energy losses
- Method: Create a dedicated local chemistry bundle generated from pygmol_argon_model.yaml for PyGMol-precision tests.
- Code area: plasma_global/chemistry/io.py; tools/external_benchmarks/pygmol_argon.py

### power_definition

- Current state: local stepped absorbed power [500.0, 1500.0, 0.0] W; PyGMol consumes its own power array
- Needed for precision: identical power deposition history and electron-energy definition
- Method: Export the exact local step power array to both solvers and compare mean-energy-equivalent consistently.
- Code area: plasma_global/electrical/direct_power.py; tools/external_benchmarks/pygmol_argon.py

### wall_loss

- Current state: local Bohm wall loss over multiple surfaces; PyGMol total cylinder-wall loss
- Needed for precision: same wall-loss coefficient or same wall-flux equation
- Method: Add a PyGMol same-wall-loss mode that uses local k_wall(t), then use it as the precision benchmark gate.
- Code area: plasma_global/physics/surface_core.py; plasma_global/physics/gas_phase_core.py

Recommended PyGMol work order:

1. Generate a dedicated single-zone local case from `pygmol_argon_model.yaml` rather than comparing against the two-zone LXCat case.
2. Use the same power waveform and same compact chemistry in both solvers.
3. Add a same-wall-loss mode before comparing ion flux or afterglow decay.
4. Use `PyGMol-Precision-1` for compact same-footing precision; keep `PyGMol-1` scoped as production-case sanity only.
