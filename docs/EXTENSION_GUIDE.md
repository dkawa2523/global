# Extension Guide

Keep extensions small and outside the numerical core unless the RHS truly needs
them. The stable public API remains:

- `load_case_from_yaml`
- `build_case`
- `run_from_yaml`

## Boundary Rules

- Keep `GlobalPlasmaSystem` focused on state, RHS composition, projection,
  labels, and solver events.
- Put EEDF behavior in `eedf` backends.
- Put absorbed-power and reduced electrical models in `electrical` backends.
- Put compact non-stoichiometric RHS terms in process extensions, not in
  reaction equations.
- Treat observables, summaries, plots, benchmark reports, and provenance as
  postprocessing.
- Use prepared files for external solver results; do not call external
  executables from the ODE RHS.

## Adding a Backend

Use the domain registries in `eedf/registry.py`, `electrical/registry.py`,
and `numerics/registry.py`.

1. Implement the existing backend interface.
2. Add a registry entry with `description`, `maturity`, `intended_use`, and any
   important `caveat`.
3. Validate any new configuration before runtime.
4. Add a focused test or smoke case for the externally visible behavior.

Use `maturity` to keep reduced-order assumptions visible in API and CLI output.
Good values are short labels such as `development`, `reduced`,
`data_driven`, `experimental`, or `stable`.

Avoid broad plugin frameworks, entry-point discovery, hidden metadata buses, and
large protocol hierarchies.

## Extending Chemistry State

Prefer the smallest extension that matches the model:

- Add species, reactions, and rate models for stoichiometric chemistry.
- Add `state_variables` for extra zone or surface states that belong in the ODE
  vector.
- Add `processes` for simple source, relaxation, or ion-flux source terms.
- Add a new Python backend only when the process cannot be represented by the
  three built-in process kinds.

Do not put arbitrary expressions or large DSLs into YAML. Keep external
databases outside the runtime path and normalize them into the standard
mechanism files first.

## External Data

External swarm or circuit tools should normally run offline:

```text
external tool output
  -> normalized table or YAML/CSV input
  -> backend
  -> global model
```

The current runtime paths are `swarm.model_name: table` for EEDF data and
`external_circuit_table` for one-way electrical waveform data. Online coupling
can be explored later as an optional adapter, not as a core dependency.

## Surface and Wall Scope

Surface and wall models are optional RHS contributions. Suitable additions are
small global closures such as sticking, ion-enhanced yields, prescribed wall-loss
frequencies, or compact film/inventory state when the case needs them.

Detailed sheath resolution, feature-scale profile evolution, surface Monte
Carlo, large material databases, 2D/3D fluid simulation, and PIC simulation are
outside the core scope.

## Documentation and Tests

Update the smallest relevant docs page when behavior changes. Prefer compact
unit or smoke tests over large benchmark gates for core changes.
