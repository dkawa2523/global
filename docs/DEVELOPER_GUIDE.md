# Developer guide

## Where to add new code

### New swarm model
Implement `SwarmModel`, register it in `SWARM_MODEL_REGISTRY`, and document the new model in the case schema.

### New electrical model
Implement `ElectricalBackend`, register it in `ELECTRICAL_REGISTRY`, and document expected `PowerRequest` metadata.

### New integrator
Implement `TimeIntegrator`, register it in `INTEGRATOR_REGISTRY`.

## Recommended review path for new contributors
1. Read `docs/ARCHITECTURE.md`
2. Validate and run `examples/configs/case_smoke.yaml`
3. Read `plasma_global/workflows/context.py`
4. Read the backend interface you want to extend
5. Read the relevant core collaborator (`GasPhaseCore`, `SurfaceCore`, `ElectricalCouplingAdapter`, or `ObservablesAdapter`)
6. Only then read `GlobalPlasmaSystem`

## Maintenance rules
- Keep file I/O outside `GlobalPlasmaSystem`
- Keep case loading outside runner execution logic
- Keep physics choices explicit in YAML
- Keep replaceable backends behind stable request/result interfaces
- Add a maturity label and assumptions when registering a new backend
- Keep optional dependencies out of module imports unless the requested feature needs them
- Document new approximations in markdown, not just code comments

## Why this matters
Research codes become unmaintainable when the workflow, physics, and I/O all couple inside one file. This revision explicitly pushes against that failure mode.

## Internal core split maintenance rule
- Add gas-species / transport source terms in `GasPhaseCore`
- Add coverage / film / wall kinetics in `SurfaceCore`
- Add power / EEDF coupling orchestration in `ElectricalCouplingAdapter`
- Add KPIs and postprocessed warnings in `ObservablesAdapter`
- Keep `GlobalPlasmaSystem` as an orchestrator, not as the place where new detailed closures accumulate

## Jacobian checks
Use the finite-difference checker when changing RHS or Jacobian code:

```bash
plasma-global check-jacobian examples/configs/case_smoke.yaml --advance-s 1e-7 --top 8
```

The command is diagnostic. Reduced electrical backend derivatives are not yet
included in the analytic Jacobian, so inspect the largest rows before turning a
threshold into a CI gate.

## Pre-merge checklist

- `py -m plasma_global.cli validate examples/configs/case_smoke.yaml`
- `py -m plasma_global.cli run examples/configs/case_smoke.yaml`
- `py -m pytest`
- New YAML keys are documented in `CONFIG_GUIDE.md`
- New physics approximations are documented in `PHYSICS_MODELS_AND_APPROXIMATIONS.md`
- New backends are listed by `py -m plasma_global.cli list-backends`
