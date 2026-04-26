# CRANE Comparison

The executable CRANE comparison currently targets the public `tutorials/TwoReactionArgon` benchmark from `https://github.com/lcpp-org/crane`.

This is a scalar chemistry ODE comparison, not a full plasma-discharge comparison. The CRANE tutorial prescribes:

- species: `e`, `Ar`, `Ar+`
- `E/N = 30 Td`
- `n_Ar(0) = 2.5e19 cm^-3`
- `n_e(0) = n_Ar+(0) = 1 cm^-3`
- `e + Ar -> e + e + Ar+` with `k = 2.1736169000623e-12 cm3/s`
- `e + Ar+ + Ar -> Ar + Ar` with `k = 1.0e-25 cm6/s`
- final time `7.5e-7 s`

The local case converts all densities and rates to SI:

- `n_Ar(0) = 2.5e25 m^-3`
- `n_Ar+(0) = 1.0e6 m^-3`
- ionization `k = 2.1736169000623e-18 m3/s`
- three-body recombination `k = 1.0e-37 m6/s`

Run:

```powershell
py -m plasma_global.cli run examples\configs\case_crane_two_reaction_argon.yaml
py tools\external_benchmarks\crane_two_reaction_argon.py
```

The compatibility wrapper `scripts\evaluate_crane_two_reaction_argon.py`
remains, but new benchmark processing lives under
`tools\external_benchmarks` so it stays outside the core solver package.

For a combined axis-by-axis assessment with ZDPlaskin, run:

```powershell
py tools\external_benchmarks\same_footing_assessment.py
```

Outputs:

- `examples/outputs/crane_two_reaction_argon/summary.yaml`
- `examples/outputs/crane_two_reaction_argon/observables.csv`
- `examples/outputs/crane_two_reaction_argon/reaction_budget.yaml`
- `examples/outputs/crane_two_reaction_argon/state_manifest.yaml`
- `examples/outputs/crane_two_reaction_argon/comparison_crane_two_reaction_argon.yaml`

Interpretation:

- Agreement primarily validates reaction parsing, stoichiometry, cm-to-SI conversion, explicit initial species densities, quasi-neutral electron handling, and stiff ODE integration.
- It does not validate electron Boltzmann kinetics, wall loss, surface kinetics, gas heating, external circuits, or RF/DC closure.
- CRANE's public tutorial does not solve electron energy. The local electron-energy state remains present because it is part of this solver architecture, but the constant-rate chemistry does not depend on it.
