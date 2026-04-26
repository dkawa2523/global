# Semiconductor-manufacturing / plasma-physics review and reflected improvements

## 1. Semiconductor manufacturing engineer viewpoint: issues and reflected improvements

### A. Missing process KPIs for recipe tuning
**Issue**
- The previous output focused on averaged plasma quantities and aggregate wafer IED only.
- It was hard to answer process-engineering questions such as:
  - Which ion species reaches the wafer?
  - What is the radical-to-ion flux ratio on the wafer?
  - Is the chemistry in an etch-like or polymerizing direction?
  - What are the residence times by zone?
  - Is pressure control drifting from the nominal chamber setting?

**Reflected improvements**
- Added zone-by-zone process observables:
  - `pressure_<zone>_Pa`
  - `total_density_<zone>_m3`
  - `residence_time_<zone>_s`
  - `electronegativity_<zone>`
  - `debye_length_<zone>_m`
- Added wafer / surface KPIs:
  - `ion_flux_<surface>_m2_s`
  - `mean_ion_energy_<surface>_eV`
  - `radical_incident_flux_<surface>_m2_s`
  - `halogen_atom_flux_<surface>_m2_s`
  - `carbon_atom_flux_<surface>_m2_s`
  - `radical_to_ion_flux_ratio_<surface>`
  - `halogen_to_C_radical_flux_ratio_<surface>`
  - `etch_deposition_balance_<surface>`
- Added net gas-surface exchange per species:
  - `net_flux_<surface>_<species>_m2_s`

### B. Species-resolved wafer ion data were insufficient
**Issue**
- Previous IED proxy was effectively single-ion / dominant-ion oriented.
- That is not enough for semiconductor etch where Ar+, CFx+, Clx+, HBr+ related ions can have different process meaning.

**Reflected improvements**
- Added species-resolved ion flux / energy fields under the electrical sheath payload.
- Exposed them into observables, for example:
  - `ion_flux_wafer_Ar_plus_m2_s`
  - `mean_ion_energy_wafer_Ar_plus_eV`
  - `charge_fraction_wafer_Ar_plus`
- The design remains generic and works for arbitrary positive ion species defined by the user chemistry.

### C. Recipe review was hard
**Issue**
- Reviewers and process engineers need step-wise summaries, not only a single end value.

**Reflected improvements**
- Added `step_summary` in `summary.yaml`.
- Each recipe step now reports mean/final/min/max values for major KPIs.
- Added aggregated warning counts by step.

### D. Surface occupancy drift was not controlled
**Issue**
- Surface coverages could drift numerically and become less useful for recipe optimization.

**Reflected improvements**
- Added state projection for surface coverages.
- For each surface, the free-site fraction is reconstructed from occupied-site fractions when a free-site species exists.
- The runner projects states after each segment and before writing output.

## 2. Plasma physicist viewpoint: issues and reflected improvements

### A. Surface-site conservation was under-constrained
**Issue**
- Even though many examples encoded `site:1` in the surface species elemental map, validation did not explicitly protect site consistency.

**Reflected improvements**
- Validator now checks:
  - surface species must carry positive `site` stoichiometry,
  - surface reactions must have a `surface_filter`,
  - explicit site conservation,
  - duplicate reaction signatures.

### B. State-based sheath conditions were not used consistently
**Issue**
- Sheath / IED estimation used chamber-level pressure/temperature in parts of the code instead of instantaneous state-based zone values.

**Reflected improvements**
- `_eval_power_and_eedf()` now reconstructs instantaneous zone pressure from state:
  - `P = k_B * T_g * sum(n_k)`
- Electrical backends now receive:
  - zone pressure,
  - zone gas temperature,
  - positive ion species composition,
  - total density.
- CCP sheath proxy now uses these dynamic values.

### C. Multi-ion electronegative plasmas were reduced too aggressively
**Issue**
- Dominant-ion-only treatment obscures physically important differences in transit time, collisionality weighting, and ion flux splitting.

**Reflected improvements**
- Added `build_species_resolved_ied()` in `electrical/sheath.py`.
- CCP now produces a multi-ion sheath payload with per-species IED proxies while preserving a compact aggregate surface payload.

### D. Model validity diagnostics were weak
**Issue**
- A reviewer could not quickly tell when the run approached regimes where the reduced closures should be treated cautiously.

**Reflected improvements**
- Added automatic regime indicators:
  - `warning_high_electronegativity_<zone>`
  - `warning_ion_ion_afterglow_<zone>`
  - `warning_pressure_deviation_<zone>`
  - `warning_debye_ratio_<zone>`
  - `warning_site_overfill_<surface>`
  - `warning_site_depletion_<surface>`
- These are also accumulated into `summary.yaml`.

## 3. Main code changes

### `plasma_global/numerics/system.py`
- Added state projection (`project_state`, `project_trajectory`)
- Added surface-site metrics and residence-time metrics
- Added dynamic pressure reconstruction and positive-ion species payload generation
- Added surface net gas-flux diagnostics
- Added process and physics observables / warnings

### `plasma_global/electrical/sheath.py`
- Added multi-ion IED helper `build_species_resolved_ied`

### `plasma_global/electrical/ccp.py`
- Switched CCP sheath payload generation from dominant-ion proxy to species-resolved proxy
- Uses instantaneous pressure / gas temperature from state metadata

### `plasma_global/electrical/icp.py`
- Updated to consume the extended zone metadata interface safely

### `plasma_global/chemistry/validators.py`
- Added surface-site checks
- Added duplicate reaction detection
- Added stronger cross-section consistency checks

### `plasma_global/observables/defaults.py`
- Added step-wise summaries and warning aggregation

### `plasma_global/io/hdf5_writer.py`
- `observables.csv` writer now unions all keys across records so dynamic species-resolved columns can be saved safely

### `plasma_global/workflows/runner.py`
- Added state projection after each solver segment and before final export

## 4. Remaining known limitations
- The sheath module is still a proxy, not a full time-dependent sheath transit solver.
- EEDF / swarm remains reduced-order relative to a full BOLSIG+-level implementation with all operator terms.
- Spatial nonuniformity remains reduced to multi-zone transport rather than a resolved fluid/PIC treatment.
- Calibration against OES / QMS / VI-probe is still not automated.

These are still important next steps, but the current revision is materially closer to what both a semiconductor process engineer and a plasma physicist can review and use.
