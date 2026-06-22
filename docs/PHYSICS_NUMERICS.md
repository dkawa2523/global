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
Electrical backends may also provide compact reduced-field or ion-energy proxy
outputs, but these are not detailed electromagnetic or sheath solutions.

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

Implemented ion wall-loss modes are intentionally small:

- `bohm` uses active wall area, zone volume, ion sound speed, and `h_factor`.
  It is a global sheath-edge closure, not a detailed RF sheath or spatial
  diffusion model.
- `prescribed_loss_frequency` uses a fixed global ion-loss `frequency_s`.
- `ambipolar_diffusion` derives the frequency from `D / L^2`.
- `off` disables ion wall loss on that surface.

Electronegative global wall-loss closure is not implemented as a separate
family in the core. For now, electronegative cases should use calibrated Bohm
factors or prescribed loss frequency. More detailed sheath, 2D diffusion,
PIC/fluid coupling, and feature-scale models are outside the core boundary; see
the [Extension Guide](EXTENSION_GUIDE.md).

Surface ion-flux observables and ion-enhanced surface rates use compact surface
IED data when provided: ion flux and mean ion energy only. Species-resolved IEDF
details are outside the normal output contract. Without explicit compact IED
data, surface terms use the zone wall-loss flux from the same global closure.

For field-gridded `rate_table` cases, `swarm.table.electron_energy_mode:
table_relaxation` relaxes the electron-energy state toward the table mean
energy. Treat this as a prescribed table closure, not as a detailed term-by-term
electron power balance.

## Numerics

The default integrator is SciPy `solve_ivp(method="BDF")` without an analytic
Jacobian. State projection is applied between recipe segments to keep
densities, electron energy, gas temperature, and surface coverage values
admissible.

Initial states include small charged-species seeds for numerical robustness.
For ignition studies or quantitative early transients, set explicit
`initial_densities_m3` in the chamber file.

Run summaries include success state, time range, solver counters, event counts,
chemistry provenance, and final compact observables. They intentionally omit
projection counters, final RHS norms, port internals, and ion-loss internals
from the normal output contract.

An optional steady-state event can stop a run when the relative RHS norm falls
below a configured threshold. It is disabled by default and should be enabled
only for cases where early steady-state termination is expected.

The package currently reports absorbed-power observables, but it does not emit a
term-by-term power balance residual. That diagnostic should only be added when
the required source, loss, transport, and storage terms are available explicitly.
