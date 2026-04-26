# Argon Executable Global-Model Comparison

## Recommended External Model

Use PyGMol for the first executable cross-check. It is a Python 0D plasma
global-model package, so it can be installed and run locally without COMSOL.

```powershell
py -m pip install ".[compare]"
py tools\external_benchmarks\pygmol_argon.py --rerun-local
```

The runner compares `examples/configs/case_argon_lxcat.yaml` against a compact
PyGMol Ar case and writes comparison artifacts into:

```text
examples/outputs/argon_lxcat_icp_baseline/
```

Key files:

- `comparison_pygmol_summary.csv` - per-step scalar comparison
- `comparison_pygmol_solution.csv` - PyGMol time history
- `comparison_pygmol_metadata.yaml` - geometry mapping, package version, and
  comparison limitations
- `tools/external_benchmarks/pygmol_argon_model.yaml` - PyGMol-only compact
  Arrhenius model definition
- `comparison_pygmol_sensitivity_summary.csv` - PyGMol-only sensitivity summary
- `comparison_pygmol_sensitivity_steps.csv` - per-step sensitivity details
- `comparison_pygmol_same_footing_decomposition.csv` - cumulative same-rate,
  same-power, same-wall-loss decomposition

## What Is Matched

- gas: pure Ar
- feed: Ar sccm from the local recipe
- pressure: volume-weighted chamber pressure
- neutral temperature: volume-weighted chamber gas temperature
- power timing: ignition, production, and afterglow steps from the local recipe
- geometry: one equivalent PyGMol cylinder preserving total modeled volume and
  surface area from the local two-zone chamber

## What Is Not Matched

- PyGMol uses one equivalent zone; the local case uses source and process zones.
- PyGMol uses Arrhenius-like electron-impact fits; the local case uses the
  Zenodo/LXCat-style cross-section bundle through the two-term swarm backend.
- The PyGMol Ar mechanism is intentionally compact and does not evolve Ar
  metastables or resonant states.
- The local wafer ion flux and PyGMol total cylinder-wall ion flux are related
  but not identical observables.

## Rate Policy

The PyGMol electron-impact rates are intentionally isolated in
`tools/external_benchmarks/pygmol_argon_model.yaml`.

That file is marked `external_benchmark_only` and
`rate_alignment_with_local_case: intentionally_not_aligned`. Do not copy those
rates into the production chemistry bundle to improve the PyGMol comparison.
Doing so would make the local production case less physical rather than more
validated.

## Sensitivity Diagnostic

Use the sensitivity diagnostic to understand the PyGMol difference without
touching the local production case:

```powershell
py tools\external_benchmarks\pygmol_sensitivity.py
```

The diagnostic perturbs only the external PyGMol benchmark inputs:

- ionization Arrhenius amplitude,
- lumped excitation Arrhenius amplitude,
- electron energy loss,
- Ar+ surface sticking,
- equivalent-cylinder wall area,
- PyGMol input power.

These sweeps are not calibration of the core solver. They are a guardrail for
answering "which PyGMol assumption dominates this sanity-check difference?"

For a more controlled PyGMol comparison, use the same-footing decomposition:

```powershell
py tools\external_benchmarks\pygmol_same_footing.py --rerun-local
```

It runs three cumulative external-adapter stages:

- same local LXCat/two-term electron-impact rate tables,
- same rates and same local absorbed-power waveform,
- same rates, same absorbed power, and the local global ion wall-loss
  coefficient.

This is still an external diagnostic. It should explain which assumptions
matter most before considering core model changes.

## How To Read The Result

Use the PyGMol comparison as a sanity check rather than a calibration target.
The useful first-pass questions are:

- Are electron densities in the same broad range during powered operation?
- Does electron energy decrease when power is removed?
- Does ion flux track electron density qualitatively?
- Are pressure and feed handling stable?

Large afterglow differences are expected because wall loss, transport geometry,
and electron cooling closures differ between the two models. If powered-step
electron density differs by many orders of magnitude, check the local Ar cross
section set, surface ion-loss closure, and absorbed-power partitioning before
expanding the chemistry.

`scripts\compare_argon_with_pygmol.py` remains as a compatibility wrapper, but
new benchmark automation should call `tools\external_benchmarks\pygmol_argon.py`.
