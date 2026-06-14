# Physics and Numerics

This is a reduced-order low-pressure plasma global model.

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
- Electrical backends are lumped or prescribed models.
- RF-envelope models are cycle-averaged.
- Sheath and ion energy handling is proxy-level.
- Spatial variation is represented only by global zones and transport edges.

## Numerics

The default integrator is SciPy `solve_ivp(method="BDF")` with an analytic sparse Jacobian. State projection is applied between recipe segments to keep densities, electron energy, gas temperature, and surface coverage values admissible.

The Jacobian covers the main gas/surface reaction and transport terms. Backend coupling derivatives are intentionally limited; use finite-difference checks before relying on the Jacobian for new backend work.

