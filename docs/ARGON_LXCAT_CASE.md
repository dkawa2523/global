# Argon LXCat Production Baseline

## Purpose

`examples/configs/case_argon_lxcat.yaml` is the recommended first production
case for pure argon. It uses `examples/configs/chamber_argon_icp.yaml`, replaces
the synthetic Ar demo cross sections with a public BOLSIG/LXCat-style data
bundle, and keeps the plasma chemistry small enough for global-model sweeps.
Gas temperature is fixed in this baseline so the first comparison isolates
cross-section, EEDF, ionization, and transport behavior.

## Data Source

The chemistry bundle in `examples/chemistry_argon_lxcat` is generated from:

- Anthony Schmalzried, "Bolsig+ format cross section of e-N2, O2, NO, Ar, O, N",
  Zenodo, DOI `10.5281/zenodo.8192503`, license `CC-BY-4.0`.
- The source file is distributed in BOLSIG+ cross-section format and contains
  Ar momentum-transfer, excitation, and ionization blocks.

The local bundle includes:

- one Ar elastic momentum-transfer curve,
- one Ar ground-state ionization curve,
- 30 Ar excitation curves represented as electron-energy-loss channels.

Excited argon densities are not evolved in this first baseline. This is a
deliberate reduced mechanism: the excitation channels improve electron energy
balance without adding metastable, resonant, and stepwise-ionization chemistry
before the base case has been checked.

## Run

```powershell
py -m plasma_global.cli validate examples\configs\case_argon_lxcat.yaml
py -m plasma_global.cli run examples\configs\case_argon_lxcat.yaml
```

Outputs are written to `examples/outputs/argon_lxcat_icp_baseline`.

## Executable Comparison

Use this case for first-order comparison against executable public global-model
tools. The repository provides a PyGMol comparison runner:

```powershell
py -m pip install ".[compare]"
py tools\external_benchmarks\pygmol_argon.py --rerun-local
```

The comparison writes:

- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_summary.csv`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_solution.csv`
- `examples/outputs/argon_lxcat_icp_baseline/comparison_pygmol_metadata.yaml`

PyGMol is a separate Python global-model package. The runner uses an equivalent
single cylinder with the same total modeled volume and surface area as this
two-zone case, then applies the same pressure, feed, and power timing. The
chemistry is intentionally compact: Ar, Ar+, ground-state ionization, a lumped
excitation energy-loss channel, and elastic electron energy loss. It is not the
same LXCat/Boltzmann cross-section integration used in this code.

Recommended comparison quantities:

- volume-averaged electron density,
- mean electron energy,
- absorbed source power,
- pressure stability,
- Ar+ ion flux to the wafer or wall,
- warning counts, especially Debye-length-to-characteristic-length warnings.

Do not expect exact agreement. This code is a two-zone global model with an
EEDF table derived from cross sections, while PyGMol is a single-zone global
model using Arrhenius-style electron collision fits. Treat agreement within
trends and order of magnitude as the first gate, then add metastable argon
chemistry if the baseline is stable.

COMSOL examples can still be useful as literature anchors, but they are not
used as the executable comparison path here.
