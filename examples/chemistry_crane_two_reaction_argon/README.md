# CRANE TwoReactionArgon Chemistry

This bundle mirrors the public CRANE `tutorials/TwoReactionArgon` example in SI units for the local global-model runner.

Source:
- Repository: `https://github.com/lcpp-org/crane`
- Input: `tutorials/TwoReactionArgon/TwoReactionArgon.i`
- Committed output: `tutorials/TwoReactionArgon/TwoReactionArgon_out.csv`

Unit conversions:
- CRANE densities are in `cm^-3`; this bundle uses `m^-3`.
- CRANE ionization rate at `E/N = 30 Td` is `2.1736169000623e-12 cm3/s`, stored here as `2.1736169000623e-18 m3/s`.
- CRANE three-body recombination rate is `1.0e-25 cm6/s`, stored here as `1.0e-37 m6/s`.

This is a chemistry and ODE benchmark. It intentionally does not exercise wall losses, surface kinetics, circuit closure, or electron-energy closure.
