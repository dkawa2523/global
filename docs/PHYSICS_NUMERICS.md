# Physics and Numerics

This is a reduced-order low-pressure plasma global model. It represents each
zone with volume-averaged state variables and optional inter-zone conductance
links; it does not resolve spatial fluid fields, PIC kinetics, detailed RF
sheath dynamics, or feature-scale surface profiles.

## Physics Scope

The ODE state can include:

- gas species densities
- electron energy density
- optional gas temperature
- surface coverages
- wall inventory
- film thickness

Electron density is normally algebraic from quasi-neutrality. A prescribed electron-density profile is available for one-way benchmark or coupling studies.

## Model Limits

- EEDF and swarm behavior are reduced-order or table driven.
- `boltzmann_2term` is an internal approximate kinetic backend for development, comparison, and compact studies. It is not a full replacement for mature swarm solvers such as BOLSIG+, LoKI-B, or Magboltz.
- Use externally generated rate/transport tables through `rate_table` when electron kinetics accuracy matters. See [External Swarm and Rate Tables](SWARM_RATE_TABLES.md).
- Electrical backends are lumped or prescribed models.
- RF-envelope models are cycle-averaged.
- Sheath and ion energy handling is proxy-level.
- Spatial variation is represented only by global zones and transport edges.

## Power, Wall, and Surface Assumptions

Absorbed power enters the electron-energy balance as zone-level power density.
Electrical backends may also provide reduced-field or ion-energy proxy
diagnostics, but these are not detailed electromagnetic or sheath solutions.

`external_circuit_table` is a one-way waveform coupling backend. It can read
measured or SPICE-generated CSV tables containing absorbed power, optional
voltage/current diagnostics, and optional reduced field. The backend
interpolates those tabulated values and passes stable `PowerResult` data to the
global model; it does not run ngspice, solve a circuit DAE, or iterate plasma
and circuit states bidirectionally.

If a reduced-field column is not supplied, `external_circuit_table` can derive a
proxy E/N from voltage, gap length, and either a configured or current zone
total density. Supplying `reduced_field_Td` directly is preferred for production
tables.

Wall loss and surface terms are global closures. Sticking, ion-enhanced yields,
site balances, wall inventory, and film thickness are intended as compact
reactor-model terms. Case-specific calibration or validation is required before
making quantitative process claims.

Implemented ion wall-loss families are intentionally small:

- Bohm-like global loss uses active wall area, zone volume, ion sound speed,
  and an edge-to-center `h_factor`. It is a global sheath-edge closure, not a
  detailed RF sheath or spatial diffusion model.
- Prescribed loss frequency uses a fixed global ion-loss frequency such as
  `loss_rate_s`. This is the preferred name when a calibrated or externally
  estimated loss rate is provided.
- `ambipolar_diffusion` remains a backward-compatible effective-frequency mode
  and can derive the frequency from `D / L^2`.

Electronegative global wall-loss closure is not implemented as a separate
family in the core. For now, electronegative cases should use calibrated Bohm
factors or prescribed loss frequency. More detailed sheath, 2D diffusion,
PIC/fluid coupling, and feature-scale models are outside the core boundary; see
the [Extension Guide](EXTENSION_GUIDE.md).

Observable output includes zone-level ion wall-loss diagnostics derived from
the same terms used by the RHS:

- `ion_wall_loss_frequency_<zone>_s`
- `ion_wall_loss_source_<zone>_m3_s`
- `ion_wall_flux_<zone>_m2_s`

Surface ion-flux observables and ion-enhanced surface rates use explicit IED
flux when provided. Without explicit IED data, they use the zone wall-loss flux
from the same global closure; the older Bohm proxy remains only as a fallback
when no active wall-loss flux is available.

For field-gridded `rate_table` cases, `swarm.table.electron_energy_mode:
table_relaxation` relaxes the electron-energy state toward the table mean
energy. Treat this as a prescribed table closure, not as a detailed term-by-term
electron power balance.

## Numerics

The default integrator is SciPy `solve_ivp(method="BDF")` with an analytic sparse Jacobian. State projection is applied between recipe segments to keep densities, electron energy, gas temperature, and surface coverage values admissible.

Initial states include small charged-species seeds for numerical robustness.
For ignition studies or quantitative early transients, set explicit
`initial_densities_m3` in the chamber file.

The Jacobian covers the main gas/surface reaction and transport terms. Backend coupling derivatives are intentionally limited; use finite-difference checks before relying on the Jacobian for new backend work.

`check-jacobian` is a diagnostic command, not a release acceptance criterion.
It compares the analytic sparse Jacobian with centered finite differences and
reports both the largest entries and state-layout group summaries, making it
easier to distinguish gas-density, electron-energy, surface, inventory, and
film-thickness mismatch sources. Solver counters, projected state histories,
observables, and generated config snapshots are the main lightweight
diagnostics in the normal run path.

## Numerical Diagnostics

Run summaries include lightweight scalar diagnostics:

- density and electron-energy projection clip counts
- maximum absolute projection change
- non-finite state entry count
- final RHS infinity norm
- final relative RHS norm in `s^-1`
- solver event counts

These diagnostics are intended to flag questionable runs, especially cases that
require repeated positivity projection. They are not a substitute for case
validation or benchmark comparison.

An optional steady-state event can stop a run when the relative RHS norm falls
below a configured threshold. It is disabled by default and should be enabled
only for cases where early steady-state termination is expected.

The package currently reports absorbed-power observables, but it does not emit a
term-by-term power balance residual. That diagnostic should only be added when
the required source, loss, transport, and storage terms are available explicitly.
