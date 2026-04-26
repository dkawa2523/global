# Numerics and solvers

## Time integration
The default integrator is SciPy BDF through `solve_ivp`.

## Jacobian strategy
The code uses an analytic sparse Jacobian path. This is important because reaction networks become stiff and large quickly.
The Jacobian covers the chemistry, transport, and Bohm-like ion wall-loss
terms used by the global ODE. Reduced electrical coupling derivatives are still
treated as frozen during a Jacobian evaluation; this is acceptable for the
current proxy backends, but high-fidelity power-coupled models should expose
their own derivative contribution.

## State projection
After integration segments, the state is projected back into an admissible region. This is used to control:
- negative densities
- invalid surface coverage sums
- free-site reconstruction

## Positivity safeguards
The configuration includes density and energy floors. These are practical numerical safeguards, not physics claims.

## Recommended future extensions
- optional AD-assisted Jacobian for backend contributions
- alternative stiff integrators
- periodic-orbit acceleration for pulse recipes
- checkpoint/restart for long process windows
