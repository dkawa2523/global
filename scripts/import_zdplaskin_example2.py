from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = 'https://github.com/Hemadityamalla/ZDPlaskin'
SOURCE_RAW_BOLSIG = (
    'https://raw.githubusercontent.com/Hemadityamalla/ZDPlaskin/master/example2/bolsigdb.dat'
)
OUT_DIR = ROOT / 'examples' / 'chemistry_zdplaskin_example2'

TORR_TO_PA = 101325.0 / 760.0
CM3_TO_M3_RATE = 1.0e-6
CM6_TO_M6_RATE = 1.0e-12
K_TO_EV = 8.617333262145e-5


def _download_text(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode('utf-8')


def _is_number_pair(line: str) -> bool:
    parts = line.replace('\t', ' ').split()
    if len(parts) < 2:
        return False
    try:
        float(parts[0].replace('D', 'E').replace('d', 'e'))
        float(parts[1].replace('D', 'E').replace('d', 'e'))
    except ValueError:
        return False
    return True


def _parse_pair(line: str) -> tuple[float, float]:
    parts = line.replace('\t', ' ').split()
    return (
        float(parts[0].replace('D', 'E').replace('d', 'e')),
        float(parts[1].replace('D', 'E').replace('d', 'e')),
    )


def parse_bolsig_blocks(text: str) -> dict[str, dict]:
    blocks: dict[str, dict] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        kind = lines[i].strip()
        if kind not in {'EFFECTIVE', 'EXCITATION', 'IONIZATION'}:
            i += 1
            continue
        process = lines[i + 1].strip()
        threshold = 0.0
        comment = ''
        pairs: list[tuple[float, float]] = []
        i += 2
        while i < len(lines):
            line = lines[i].strip()
            if line in {'EFFECTIVE', 'EXCITATION', 'IONIZATION'}:
                break
            if '/ threshold' in line.lower() or 'threshold energy' in line.lower():
                parts = line.split('/', 1)[0].split()
                if parts:
                    threshold = float(parts[0].replace('D', 'E').replace('d', 'e'))
            elif line.upper().startswith('COMMENT:'):
                comment = line.split(':', 1)[1].strip()
            elif _is_number_pair(line):
                pairs.append(_parse_pair(line))
            i += 1
        if len(pairs) >= 2:
            blocks[process] = {
                'kind': kind,
                'process': process,
                'threshold_eV': threshold,
                'comment': comment,
                'pairs': pairs,
            }
    return blocks


def write_xy(path: Path, pairs: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['energy_eV', 'sigma_m2'])
        writer.writerows(pairs)


def write_species(path: Path) -> None:
    rows = [
        ['e', 'e', 'gas', -1, 0.00054858, '', 'electron', 'electron', 'plasma', ''],
        ['Ar', 'Ar', 'gas', 0, 39.948, 'Ar:1', 'argon', 'stable|parent', 'plasma', ''],
        ['Ar_star', 'Ar*', 'gas', 0, 39.948, 'Ar:1', 'Ar*|Ar_excited', 'excited|metastable|radical', 'plasma', ''],
        ['Ar_plus', 'Ar+', 'gas', 1, 39.948, 'Ar:1', 'Ar+|Ar(+)', 'ion', 'plasma', ''],
        ['Ar2_plus', 'Ar2+', 'gas', 1, 79.896, 'Ar:2', 'Ar2+|Ar2(+)', 'ion|molecular_ion', 'plasma', ''],
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['canonical_id', 'display_name', 'phase', 'charge', 'mass_amu', 'elements', 'aliases', 'state_tags', 'zones', 'surfaces'])
        writer.writerows(rows)


def write_gas_reactions(path: Path) -> None:
    rows = [
        ['ZDP_AR_ION', 'gas', 'e + Ar -> e + e + Ar_plus', 'RM_E_AR_ION_ZDP', 'EM_E_AR_ION_ZDP', 'plasma', '', 'true', 'ZDPlaskin example2 ground-state ionization'],
        ['ZDP_AR_EXC', 'gas', 'e + Ar -> Ar_star + e', 'RM_E_AR_EXC_ZDP', 'EM_E_AR_EXC_ZDP', 'plasma', '', 'true', 'ZDPlaskin example2 lumped excitation Ar <-> Ar*; forward excitation only'],
        ['ZDP_ARSTAR_DEEXC_APPROX', 'gas', 'Ar_star + e -> Ar + e', 'RM_E_ARSTAR_DEEXC_APPROX', '', 'plasma', '', 'true', 'Approximate electron de-excitation placeholder; ZDPlaskin evaluates this through BOLSIG superelastic handling'],
        ['ZDP_ARSTAR_ION', 'gas', 'e + Ar_star -> Ar_plus + e + e', 'RM_E_ARSTAR_ION_ZDP', 'EM_E_ARSTAR_ION_ZDP', 'plasma', '', 'true', 'ZDPlaskin example2 shifted Ar* ionization'],
        ['ZDP_AR2PLUS_DISS_RECOMB', 'gas', 'Ar2_plus + e -> Ar_star + Ar', 'RM_AR2PLUS_DISS_RECOMB_ZDP', '', 'plasma', '', 'true', '8.5e-7 cm3/s electron dissociative recombination; Te power omitted'],
        ['ZDP_AR2PLUS_CONVERSION', 'gas', 'Ar2_plus + Ar -> Ar_plus + Ar + Ar', 'RM_AR2PLUS_CONVERSION_ZDP', '', 'plasma', '', 'true', '6.06e-6/Tgas exp(-15130/Tgas) cm3/s'],
        ['ZDP_ARSTAR_POOLING', 'gas', 'Ar_star + Ar_star -> Ar2_plus + e', 'RM_ARSTAR_POOLING_ZDP', '', 'plasma', '', 'true', '6.0e-10 cm3/s'],
        ['ZDP_ARPLUS_THREE_BODY_RECOMB', 'gas', 'Ar_plus + e + e -> Ar + e', 'RM_ARPLUS_THREE_BODY_RECOMB_ZDP', '', 'plasma', '', 'true', '8.75e-27 cm6/s three-body recombination; Te power omitted'],
        ['ZDP_ARSTAR_THREE_BODY_QUENCH', 'gas', 'Ar_star + Ar + Ar -> Ar + Ar + Ar', 'RM_ARSTAR_THREE_BODY_QUENCH_ZDP', '', 'plasma', '', 'true', '1.4e-32 cm6/s'],
        ['ZDP_ARPLUS_CLUSTERING', 'gas', 'Ar_plus + Ar + Ar -> Ar2_plus + Ar', 'RM_ARPLUS_CLUSTERING_ZDP', '', 'plasma', '', 'true', '2.25e-31 (Tgas/300)^-0.4 cm6/s'],
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['reaction_id', 'phase', 'equation', 'rate_model_key', 'energy_model_key', 'zone_filter', 'surface_filter', 'enabled', 'notes'])
        writer.writerows(rows)


def write_electron_impact_models(path: Path) -> None:
    models = {
        'rate_models': {
            'RM_E_AR_ION_ZDP': {'backend': 'electron_impact_xsec', 'cross_section_id': 'xs_zdp_ar_ionization', 'branching_yield': 1.0},
            'RM_E_AR_EXC_ZDP': {'backend': 'electron_impact_xsec', 'cross_section_id': 'xs_zdp_ar_excitation', 'branching_yield': 1.0},
            'RM_E_ARSTAR_ION_ZDP': {'backend': 'electron_impact_xsec', 'cross_section_id': 'xs_zdp_ar_star_ionization', 'branching_yield': 1.0},
        }
    }
    path.write_text(yaml.safe_dump(models, sort_keys=False), encoding='utf-8')


def write_gas_rate_models(path: Path) -> None:
    models = {
        'rate_models': {
            'RM_E_ARSTAR_DEEXC_APPROX': {'backend': 'constant', 'value': 1.0e-13},
            'RM_AR2PLUS_DISS_RECOMB_ZDP': {
                'backend': 'te_power_law',
                'A': 8.5e-7 * CM3_TO_M3_RATE,
                'Tref_K': 300.0,
                'alpha': -0.67,
                'electron_temperature_factor': 2.0 / 3.0,
            },
            'RM_AR2PLUS_CONVERSION_ZDP': {
                'backend': 'arrhenius',
                'A': (6.06e-6 * CM3_TO_M3_RATE) / 300.0,
                'beta': -1.0,
                'Ea_eV': 15130.0 * K_TO_EV,
            },
            'RM_ARSTAR_POOLING_ZDP': {'backend': 'constant', 'value': 6.0e-10 * CM3_TO_M3_RATE},
            'RM_ARPLUS_THREE_BODY_RECOMB_ZDP': {
                'backend': 'te_power_law',
                'A': 8.75e-27 * CM6_TO_M6_RATE,
                'Tref_K': 11600.0,
                'alpha': -4.5,
                'electron_temperature_factor': 2.0 / 3.0,
            },
            'RM_ARSTAR_THREE_BODY_QUENCH_ZDP': {'backend': 'constant', 'value': 1.4e-32 * CM6_TO_M6_RATE},
            'RM_ARPLUS_CLUSTERING_ZDP': {'backend': 'arrhenius', 'A': 2.25e-31 * CM6_TO_M6_RATE, 'beta': -0.4, 'Ea_eV': 0.0},
        }
    }
    path.write_text(yaml.safe_dump(models, sort_keys=False), encoding='utf-8')


def write_energy_models(path: Path) -> None:
    models = {
        'rate_models': {
            'EM_E_AR_ION_ZDP': {'backend': 'constant_event_loss', 'energy_loss_eV': 15.8},
            'EM_E_AR_EXC_ZDP': {'backend': 'constant_event_loss', 'energy_loss_eV': 11.5},
            'EM_E_ARSTAR_ION_ZDP': {'backend': 'constant_event_loss', 'energy_loss_eV': 4.3},
        }
    }
    path.write_text(yaml.safe_dump(models, sort_keys=False), encoding='utf-8')


def write_manifest_files(out_dir: Path) -> None:
    (out_dir / 'surface_reactions.csv').write_text(
        'reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes\n',
        encoding='utf-8',
    )
    (out_dir / 'aliases.yaml').write_text(
        yaml.safe_dump({'aliases': {'Ar+': 'Ar_plus', 'Ar2+': 'Ar2_plus', 'Ar*': 'Ar_star'}}, sort_keys=False),
        encoding='utf-8',
    )
    (out_dir / 'chemistry_manifest.yaml').write_text(
        yaml.safe_dump(
            {
                'species_file': 'species.csv',
                'gas_reactions_file': 'gas_reactions.csv',
                'surface_reactions_file': 'surface_reactions.csv',
                'aliases_file': 'aliases.yaml',
                'cross_sections_manifest': 'cross_sections_manifest.yaml',
                'model_files': {
                    'electron_impact': 'electron_impact_models.yaml',
                    'gas_rate': 'gas_rate_models.yaml',
                    'energy_loss': 'energy_loss_models.yaml',
                },
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )


def write_cross_sections_manifest(out_dir: Path, blocks: dict[str, dict]) -> None:
    entries = [
        {
            'cross_section_id': 'xs_zdp_ar_effective',
            'kind': 'momentum_transfer',
            'target_species': 'Ar',
            'threshold_eV': 0.0,
            'energy_loss_eV': 0.0,
            'source': 'file',
            'file': 'cross_sections/xs_zdp_ar_effective.csv',
            'format': 'csv_xy',
            'unit_energy': 'eV',
            'unit_sigma': 'm2',
            'metadata': {'source_repo': SOURCE_REPO, 'source_file': 'example2/bolsigdb.dat', 'original_process': 'Ar', 'original_kind': 'EFFECTIVE', 'mass_ratio': 1.36e-5},
        },
        {
            'cross_section_id': 'xs_zdp_ar_excitation',
            'kind': 'excitation',
            'target_species': 'Ar',
            'threshold_eV': 11.5,
            'energy_loss_eV': 11.5,
            'source': 'file',
            'file': 'cross_sections/xs_zdp_ar_excitation.csv',
            'format': 'csv_xy',
            'unit_energy': 'eV',
            'unit_sigma': 'm2',
            'metadata': {'source_repo': SOURCE_REPO, 'source_file': 'example2/bolsigdb.dat', 'original_process': 'Ar <-> Ar*', 'original_kind': 'EXCITATION'},
        },
        {
            'cross_section_id': 'xs_zdp_ar_ionization',
            'kind': 'ionization',
            'target_species': 'Ar',
            'threshold_eV': 15.8,
            'energy_loss_eV': 15.8,
            'source': 'file',
            'file': 'cross_sections/xs_zdp_ar_ionization.csv',
            'format': 'csv_xy',
            'unit_energy': 'eV',
            'unit_sigma': 'm2',
            'metadata': {'source_repo': SOURCE_REPO, 'source_file': 'example2/bolsigdb.dat', 'original_process': 'Ar -> Ar^+', 'original_kind': 'IONIZATION'},
        },
        {
            'cross_section_id': 'xs_zdp_ar_star_ionization',
            'kind': 'ionization',
            'target_species': 'Ar_star',
            'threshold_eV': 4.3,
            'energy_loss_eV': 4.3,
            'source': 'file',
            'file': 'cross_sections/xs_zdp_ar_star_ionization.csv',
            'format': 'csv_xy',
            'unit_energy': 'eV',
            'unit_sigma': 'm2',
            'metadata': {
                'source_repo': SOURCE_REPO,
                'source_file': 'example2/bolsigdb.dat',
                'original_process': 'Ar* -> Ar^+',
                'original_kind': 'IONIZATION',
                'source_warning': 'ZDPlaskin file states this is shifted from ground-state ionization and is not a real cross section.',
            },
        },
    ]
    (out_dir / 'cross_sections_manifest.yaml').write_text(yaml.safe_dump({'cross_sections': entries}, sort_keys=False), encoding='utf-8')


def write_readme(out_dir: Path) -> None:
    text = """# ZDPlaskin Example2 Ar Bundle

This bundle imports the Ar cross sections and compact Ar/Ar*/Ar2+ chemistry
from `https://github.com/Hemadityamalla/ZDPlaskin`, folder `example2`.

The local solver does not reproduce ZDPlaskin's voltage-resistor circuit,
DVODE integration, BOLSIG+ superelastic handling, or closed-volume assumptions.
It provides the same species family, the same tabulated Ar BOLSIG cross-section
curves, and the same nominal operating geometry/pressure for a local global
model evaluation.

Input files are split by model type:

- `cross_sections_manifest.yaml`: tabulated electron-collision data.
- `electron_impact_models.yaml`: electron-impact rates evaluated from those
  cross sections.
- `gas_rate_models.yaml`: empirical constant, gas-temperature, or
  electron-temperature rate expressions without cross-section data.
- `energy_loss_models.yaml`: per-event electron energy losses.
"""
    (out_dir / 'README.md').write_text(text, encoding='utf-8')


def generate(out_dir: Path, source_file: Path | None) -> None:
    if source_file is not None:
        text = source_file.read_text(encoding='utf-8')
    else:
        text = _download_text(SOURCE_RAW_BOLSIG)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'cross_sections').mkdir(parents=True, exist_ok=True)
    (out_dir / 'source_bolsigdb.dat').write_text(text, encoding='utf-8')
    blocks = parse_bolsig_blocks(text)

    required = {
        'Ar': 'xs_zdp_ar_effective.csv',
        'Ar <-> Ar*': 'xs_zdp_ar_excitation.csv',
        'Ar -> Ar^+': 'xs_zdp_ar_ionization.csv',
        'Ar* -> Ar^+': 'xs_zdp_ar_star_ionization.csv',
    }
    missing = [name for name in required if name not in blocks]
    if missing:
        raise RuntimeError(f'Missing expected BOLSIG blocks: {missing}')
    for process, filename in required.items():
        write_xy(out_dir / 'cross_sections' / filename, blocks[process]['pairs'])
    write_species(out_dir / 'species.csv')
    write_gas_reactions(out_dir / 'gas_reactions.csv')
    write_electron_impact_models(out_dir / 'electron_impact_models.yaml')
    write_gas_rate_models(out_dir / 'gas_rate_models.yaml')
    write_energy_models(out_dir / 'energy_loss_models.yaml')
    legacy_models = out_dir / 'reaction_models.yaml'
    if legacy_models.exists():
        legacy_models.unlink()
    write_cross_sections_manifest(out_dir, blocks)
    write_manifest_files(out_dir)
    write_readme(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description='Import ZDPlaskin example2 Ar chemistry into this repository.')
    parser.add_argument('--source-file', type=Path, default=None, help='Local ZDPlaskin example2 bolsigdb.dat')
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    generate(args.output_dir.resolve(), args.source_file.resolve() if args.source_file else None)
    print(args.output_dir.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
