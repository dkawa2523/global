# Model maturity

This project separates "callable" from "validated". A backend can be useful for
workflow development without being ready for process interpretation.

## Labels

- `stable`: suitable as a default engineering mechanism when the surrounding
  case has been validated.
- `stable-debug`: useful for smoke tests, debugging, and controlled reduced
  studies, but not intended as a high-fidelity physics claim.
- `experimental`: implemented and runnable, but the approximation and regime
  limits must be reviewed before using results quantitatively.
- `placeholder`: interface shell only. Do not use output from this model for
  process conclusions.

## Current backend status

| Category | Backend | Status | Notes |
| --- | --- | --- | --- |
| EEDF | `maxwell` | `stable-debug` | Fast analytic closure for smoke tests and mechanism debugging. |
| EEDF | `swarm` | `experimental` | Wrapper; maturity depends on the selected swarm model. |
| EEDF | `boltzmann_2term` | `experimental` | Reduced two-term-like closure, not a BOLSIG+ replacement. |
| EEDF | `rate_table` | `experimental` | Accuracy depends on the table source and metadata. |
| Electrical | `direct_power` | `stable-debug` | Direct power deposition with no circuit or EM physics. |
| Electrical | `dc_series_circuit` | `experimental` | Reduced DC/pulsed-DC voltage-source and ballast-resistor circuit. |
| Electrical | `external_circuit_table` | `experimental` | One-way measured/SPICE waveform CSV input; no plasma-to-circuit feedback. |
| Electrical | `rf_envelope` | `experimental` | Cycle-averaged HF/LF power and bias envelope model. |
| Electrical | `ccp` | `experimental` | Lumped CCP and sheath proxy. |
| Electrical | `icp` | `experimental` | Reduced ICP coupling proxy. |
| Electron density | `quasi_neutral` | `stable-debug` | Default algebraic electron density from charged gas species. |
| Electron density | `prescribed_profile` | `experimental` | One-way `ne(t)` CSV driver for benchmarks and loose coupling; not self-consistent plasma production. |
| Integrator | `scipy_bdf` | `stable` | Standard stiff ODE backend. |

The command below prints the same information from the live registry:

```bash
plasma-global list-backends
```

## Promotion checklist

A model should not move to a stronger maturity label until it has:

- a documented request/result interface,
- a short statement of assumptions and known failure regimes,
- at least one smoke or unit test that exercises the backend,
- a conservation or consistency check where applicable,
- a comparison against either an analytic limit, an external solver, or a
  reference dataset for the intended regime.
