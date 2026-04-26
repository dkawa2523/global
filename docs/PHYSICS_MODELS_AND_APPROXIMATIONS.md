# Physics models and approximations

## Gas-phase core
The code solves a transient global model over one or more zones.
State variables include gas species densities, electron energy, optional gas temperature, surface coverages, wall inventory, and film thickness.

## Electron kinetics
The EEDF layer is replaceable.
Current options include:
- Maxwell closure
- table-based swarm backend
- internal two-term Boltzmann swarm backend
- swarm wrapper for future models

Important approximation:
The global model interacts with the EEDF layer through rates and transport summaries, not through a full kinetic distribution in the main ODE state.
For the default global ODE path, electron-impact rates are tied to the evolved
electron energy density through the mean-energy lookup. Explicit local-field
lookup remains available for reduced E/N studies, but it should not be mixed
with an independently interpreted electron-energy equation.

Electron density is normally an algebraic quantity from quasi-neutrality, not
an independent ODE state. This is the recommended production setting for global
model studies because ion production and ion loss determine the charged-particle
inventory self-consistently at the model level.

`physics.electron_density_closure: prescribed_profile` is available for
benchmark-driven or one-way coupled studies. In this mode, a CSV profile
provides `ne(t)` for each zone, and that density is passed to the EEDF and
electrical coupling. The gas ion densities remain ordinary ODE states. This is
useful when comparing chemistry/rate handling on a fixed electron-density
trajectory, but it is not a self-consistent discharge solution.

## Surface chemistry
Surface reactions are handled separately from gas-phase reactions.
The code supports coverage-dependent laws and keeps surface species distinct from gas species.

## Electrical models
Electrical behavior is reduced-order and modular.
Current backends include:
- direct absorbed power
- DC/pulsed-DC series circuit
- external circuit/result table, used as one-way loose coupling
- RF envelope for HF/LF cycle-averaged source and bias inputs
- reduced ICP coupling
- reduced CCP / bias model

The `dc_series_circuit` backend solves a reduced voltage-source plus ballast
resistor model with the plasma represented as a conductive load. It is useful
for DC and pulsed-DC studies where voltage/current feedback matters, but it is
not a sheath-capacitive RF or SPICE circuit solver.

The `external_circuit_table` backend reads measured or externally computed
waveform/result CSV files and deposits the interpolated power into the global
plasma model. This is useful for loose comparison with ngspice or laboratory
waveforms, but it does not feed plasma impedance back to the external circuit.

The `rf_envelope` backend represents HF/LF sources by RF-period-averaged power
or RMS voltage commands. It is appropriate for recipe-level studies where the
RF cycle is not resolved. It does not solve matching networks, sheath motion,
harmonics, or electron kinetics within the RF period.

## Sheath / IED
The current IED module is a proxy, not a full time-dependent ion-transit solver.
This is acceptable for many sweeps and architecture work, but should be treated as replaceable.
Positive-ion wall loss defaults to a Bohm-family global flux closure. It
provides the missing leading-order charged-particle sink, not a spatial sheath
solution. Prefer `surface.models.ion_loss: bohm_edge_loss` for the current
default behavior. Legacy `Bohm_like` and `sheath_flux` settings still map to
this same reduced closure; they do not imply a resolved sheath-flux model.

Bohm-family modes use
`Gamma_i = h_factor * 0.61 * n_i * sqrt(eps_e / m_i)`. The default
`bohm_edge_loss` keeps `h_factor = 1.0` unless configured. `bohm_global_loss`
uses an automatic edge-to-center estimate based on nominal pressure,
temperature, characteristic length, and an ion-neutral cross section unless
`h_factor` is set explicitly. The automatic estimate is intentionally simple:
it is useful for sensitivity studies and for avoiding a hidden `h = 1`
assumption, but it is not a calibrated Lieberman/Godyak geometry model.
Cases with known geometry should set `characteristic_length_m` or `h_factor`
explicitly.

`ambipolar_diffusion` is available as a separate first-order volumetric ion
loss, `dn_i/dt = -k_loss n_i`, when a case is intended to use a diffusion-loss
global closure. The solver does not add Bohm and ambipolar losses in the same
zone; configuration validation rejects mixed ion-loss families to avoid double
counting the same wall sink.

Ion-loss diagnostics are written to the observables table for each zone:
`ion_loss_family`, `ion_loss_area_m2`, `ion_loss_h_factor`,
`ion_loss_characteristic_length_m`, and `ambipolar_loss_rate_s`.

## Main current limitations
- reduced-order sheath model
- reduced-order power coupling
- global rather than spatially resolved transport
- model accuracy depends strongly on chemistry quality and calibration
