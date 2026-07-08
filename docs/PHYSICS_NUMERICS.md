# Physics and Numerics

This is a reduced-order low-pressure plasma global model. Each zone is
volume-averaged, with optional inter-zone conductance links.

## State

The ODE state can include:

- gas species densities
- electron energy density
- optional gas temperature
- surface coverages
- wall inventory
- film thickness

Electron density is normally algebraic from quasi-neutrality. A prescribed
electron-density profile is available for one-way table/profile coupling.

## Implemented Approximations

- EEDF behavior is analytic by default, or HDF5 table driven when the `swarm`
  backend is selected.
- `swarm.model_name: boltzmann_2term` is an internal approximate kinetic backend, not a full
  replacement for mature swarm solvers.
- `swarm.model_name: table` reads prepared rate/transport tables and fails fast when required
  rates are missing.
- Electrical backends are lumped, prescribed, reduced RF/DC models, or one-way
  table inputs.
- Inter-zone `edges` are directed, constant-conductance transport links from
  `from_zone` to `to_zone`; they are not pressure-difference flow solvers.
- Absorbed power is split between electron-energy and optional gas-temperature
  equations. When gas temperature is evolved, `physics.gas_heating_fraction`
  goes to gas heating and the remaining fraction goes to electron energy.
- RF-envelope models are cycle-averaged.
- Wall loss and surface reactions are global closures.
- Surface IED data is compact: ion flux and mean ion energy only.

## Wall and Surface Scope

Implemented ion wall-loss modes:

- `bohm`: active wall area, zone volume, ion sound speed, and `h_factor`
- `prescribed_loss_frequency`: fixed `frequency_s`
- `ambipolar_diffusion`: `diffusion_coefficient_m2_s / diffusion_length_m^2`
- `off`: disables ion wall loss on that surface

Do not mix Bohm-like and effective-frequency ion-loss families in one zone. The
validator rejects mixed families to avoid double counting.

Electronegative sheath models, spatial diffusion solvers, detailed IEDF/IEDF
models, feature-scale surfaces, PIC, and 2D/3D fluid models are outside the
core.

## Numerics

The default integrator is SciPy `solve_ivp(method="BDF")` without an analytic
Jacobian. State projection is applied between recipe segments to keep densities,
electron energy, gas temperature, and surface coverage values admissible.

Density and coverage floors are used for projection, tolerances, and guarded
division. They are not used as artificial reactants in gas or surface reaction
mass-action rates.

`numerics.atol` remains a scalar in case files. For BDF solves, the system
expands it to a state-vector tolerance with small floors for density, electron
energy, gas temperature, and surface coverage states.

Recipe steps with `pulsed_square` power waveforms and positive `repetition_Hz`
cap BDF `max_step` at about one twentieth of the pulse period, unless the
configured `numerics.max_step` is already smaller.

Optional steady-state events can terminate a recipe step when the relative RHS
norm is below a configured threshold. They are disabled by default.

`summary.yaml` and `observables.csv` are compact by default. Detailed
reaction/source/loss budget columns are opt-in through
`outputs.budgets.enabled: true`.
