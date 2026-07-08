from __future__ import annotations

import argparse
import csv
import re
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_URL = 'https://zenodo.org/records/8192503/files/Cross%20section.txt?download=1'
SOURCE_DOI = '10.5281/zenodo.8192503'
KEYWORDS = {'TOTAL', 'EFFECTIVE', 'ELASTIC', 'EXCITATION', 'IONIZATION', 'ATTACHMENT'}
NUM_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$')


@dataclass
class Block:
    kind: str
    title: str
    threshold_eV: float
    species: str
    process: str
    comment: str
    updated: str
    pairs: list[tuple[float, float]]


def _download(path: Path) -> None:
    with urllib.request.urlopen(DEFAULT_URL, timeout=60) as response:
        path.write_bytes(response.read())


def _decode(path: Path) -> list[str]:
    data = path.read_bytes()
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return data.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace').splitlines()


def _float_token(text: str) -> float | None:
    token = text.strip().replace('D', 'E').replace('d', 'e')
    if not NUM_RE.match(token):
        return None
    return float(token)


def _numeric_pair(line: str) -> tuple[float, float] | None:
    clean = line.strip()
    if not clean or clean.startswith('#'):
        return None
    toks = clean.replace(',', ' ').replace(';', ' ').split()
    if len(toks) < 2:
        return None
    a = _float_token(toks[0])
    b = _float_token(toks[1])
    if a is None or b is None:
        return None
    return a, b


def _threshold(lines: list[str]) -> float:
    for line in lines[:5]:
        val = _float_token(line)
        if val is not None:
            return val
    joined = ' '.join(lines)
    match = re.search(r'Eth\s*=\s*([0-9.+\-EeDd]+)', joined)
    return float(match.group(1).replace('D', 'E').replace('d', 'e')) if match else 0.0


def _field(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line.split(':', 1)[1].strip()
    return ''


def _finalize(raw: list[str]) -> Block | None:
    if not raw:
        return None
    kind = raw[0].strip()
    if kind not in KEYWORDS:
        return None
    pairs = [pair for line in raw for pair in [_numeric_pair(line)] if pair is not None]
    if len(pairs) < 2:
        return None
    title = raw[1].strip() if len(raw) > 1 else ''
    return Block(
        kind=kind,
        title=title,
        threshold_eV=_threshold(raw[1:8]),
        species=_field(raw, 'SPECIES:'),
        process=_field(raw, 'PROCESS:'),
        comment=_field(raw, 'COMMENT:'),
        updated=_field(raw, 'UPDATED:'),
        pairs=pairs,
    )


def parse_blocks(path: Path) -> list[Block]:
    blocks: list[Block] = []
    current: list[str] = []
    for line in _decode(path):
        stripped = line.strip()
        if stripped in KEYWORDS:
            block = _finalize(current)
            if block is not None:
                blocks.append(block)
            current = [stripped]
        elif current:
            current.append(line)
    block = _finalize(current)
    if block is not None:
        blocks.append(block)
    return blocks


def select_first_argon_set(blocks: list[Block]) -> tuple[Block, Block, list[Block]]:
    argon = [block for block in blocks if block.species == 'e / Ar']
    selected: list[Block] = []
    seen_argon = False
    for block in blocks:
        if block.species == 'e / Ar':
            seen_argon = True
            selected.append(block)
            if block.kind == 'IONIZATION':
                break
        elif seen_argon:
            continue
    if not selected:
        selected = argon
    momentum = next((b for b in selected if b.kind == 'ELASTIC' and 'Momentum' in b.process), None)
    ionization = next((b for b in selected if b.kind == 'IONIZATION'), None)
    excitations = [b for b in selected if b.kind == 'EXCITATION']
    if momentum is None or ionization is None or not excitations:
        raise RuntimeError('Could not find a complete first Ar set with momentum, excitation, and ionization blocks')
    return momentum, ionization, excitations


def write_curve(path: Path, block: Block) -> None:
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['energy_eV', 'sigma_m2'])
        for e, sigma in block.pairs:
            writer.writerow([f'{e:.8e}', f'{max(sigma, 0.0):.8e}'])


def yaml_quote(text: str) -> str:
    text = text.encode('ascii', errors='replace').decode('ascii')
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def write_bundle(output: Path, momentum: Block, ionization: Block, excitations: list[Block]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    xs_dir = output / 'cross_sections'
    xs_dir.mkdir(parents=True, exist_ok=True)

    write_curve(xs_dir / 'xs_ar_momentum_zenodo.csv', momentum)
    write_curve(xs_dir / 'xs_ar_ionization_zenodo.csv', ionization)
    for i, block in enumerate(excitations, start=1):
        write_curve(xs_dir / f'xs_ar_excitation_{i:02d}_zenodo.csv', block)

    entries: list[str] = []

    def add_entry(cs_id: str, kind: str, file_name: str, block: Block, energy_loss: float | None = None) -> None:
        loss = block.threshold_eV if energy_loss is None else energy_loss
        entries.append(textwrap.dedent(f"""\
          - cross_section_id: {cs_id}
            kind: {kind}
            target_species: Ar
            threshold_eV: {block.threshold_eV:.8g}
            energy_loss_eV: {loss:.8g}
            source: file
            file: cross_sections/{file_name}
            format: csv_xy
            unit_energy: eV
            unit_sigma: m2
            metadata:
              source_dataset: {yaml_quote('Bolsig+ format cross section of e-N2, O2, NO, Ar, O, N')}
              source_doi: {yaml_quote(SOURCE_DOI)}
              source_url: {yaml_quote(DEFAULT_URL)}
              license: {yaml_quote('CC-BY-4.0')}
              original_kind: {yaml_quote(block.kind)}
              original_process: {yaml_quote(block.process)}
              original_comment: {yaml_quote(block.comment)}
              updated: {yaml_quote(block.updated)}
        """))

    add_entry('xs_ar_momentum_zenodo', 'momentum_transfer', 'xs_ar_momentum_zenodo.csv', momentum, 0.0)
    add_entry('xs_ar_ionization_zenodo', 'ionization', 'xs_ar_ionization_zenodo.csv', ionization)
    for i, block in enumerate(excitations, start=1):
        add_entry(f'xs_ar_excitation_{i:02d}_zenodo', 'excitation', f'xs_ar_excitation_{i:02d}_zenodo.csv', block)
    (output / 'cross_sections_manifest.yaml').write_text('cross_sections:\n' + ''.join(entries), encoding='utf-8')

    species_text = textwrap.dedent("""\
        canonical_id,display_name,phase,charge,mass_amu,elements,aliases,state_tags,zones,surfaces
        e,e,gas,-1,0.00054858,,electron,electron,source|process,
        Ar,Ar,gas,0,39.948,Ar:1,argon,stable|parent,source|process,
        Ar_plus,Ar+,gas,1,39.948,Ar:1,Ar+|Ar(+),ion,source|process,
        wafer:*,wafer:*,surface,0,0.0,site:1,wafer_site|wafer_*,site,,wafer
        wall:*,wall:*,surface,0,0.0,site:1,wall_site|wall_*,site,,grounded_wall
        source:*,source:*,surface,0,0.0,site:1,source_site|source_*,site,,source_wall
    """)
    (output / 'species.csv').write_text(species_text, encoding='utf-8')

    rows = [
        ['reaction_id', 'phase', 'equation', 'rate_model_key', 'energy_model_key', 'zone_filter', 'surface_filter', 'enabled', 'notes'],
        ['G_AR_ION', 'gas', 'e + Ar -> Ar_plus + e + e', 'RM_E_AR_ION', 'EM_E_AR_ION', 'source|process', '', 'true', 'Ground-state argon ionization from the Zenodo/LXCat-derived BOLSIG+ data file'],
    ]
    for i, _block in enumerate(excitations, start=1):
        rows.append([
            f'G_AR_EXC_{i:02d}',
            'gas',
            'e + Ar -> Ar + e',
            f'RM_E_AR_EXC_{i:02d}',
            f'EM_E_AR_EXC_{i:02d}',
            'source|process',
            '',
            'true',
            'Argon excitation represented as an electron-energy-loss channel; no excited-state density is evolved',
        ])
    with (output / 'gas_reactions.csv').open('w', encoding='utf-8', newline='') as fh:
        csv.writer(fh).writerows(rows)

    (output / 'surface_reactions.csv').write_text(
        'reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes\n',
        encoding='utf-8',
    )

    electron_model_lines = ['rate_models:']
    energy_model_lines = ['rate_models:']
    electron_model_lines.extend([
        '  RM_E_AR_ION:',
        '    backend: electron_impact_xsec',
        '    cross_section_id: xs_ar_ionization_zenodo',
        '    branching_yield: 1.0',
    ])
    energy_model_lines.extend([
        '  EM_E_AR_ION:',
        '    backend: constant_event_loss',
        f'    energy_loss_eV: {ionization.threshold_eV:.8g}',
    ])
    for i, block in enumerate(excitations, start=1):
        electron_model_lines.extend([
            f'  RM_E_AR_EXC_{i:02d}:',
            '    backend: electron_impact_xsec',
            f'    cross_section_id: xs_ar_excitation_{i:02d}_zenodo',
            '    branching_yield: 1.0',
        ])
        energy_model_lines.extend([
            f'  EM_E_AR_EXC_{i:02d}:',
            '    backend: constant_event_loss',
            f'    energy_loss_eV: {block.threshold_eV:.8g}',
        ])
    (output / 'electron_impact_models.yaml').write_text('\n'.join(electron_model_lines) + '\n', encoding='utf-8')
    (output / 'energy_loss_models.yaml').write_text('\n'.join(energy_model_lines) + '\n', encoding='utf-8')

    (output / 'chemistry_manifest.yaml').write_text(textwrap.dedent("""\
        species_file: species.csv
        gas_reactions_file: gas_reactions.csv
        surface_reactions_file: surface_reactions.csv
        cross_sections_manifest: cross_sections_manifest.yaml
        model_files:
          electron_impact: electron_impact_models.yaml
          energy_loss: energy_loss_models.yaml
    """), encoding='utf-8')
    (output / 'README.md').write_text(textwrap.dedent(f"""\
        # Argon chemistry from public BOLSIG+/LXCat-style data

        This bundle is generated from the open Zenodo dataset:
        "Bolsig+ format cross section of e-N2, O2, NO, Ar, O, N",
        DOI: {SOURCE_DOI}, license: CC-BY-4.0.

        The production example uses the first complete Ar section found in the
        source file:

        - one elastic momentum-transfer curve,
        - one ground-state ionization curve,
        - {len(excitations)} excitation curves represented as electron-energy-loss channels.

        Excited argon densities are not evolved in this reduced bundle. The
        excitation channels are included to improve the electron energy balance
        relative to a one-ionization-reaction mechanism while keeping the case
        small enough for global-model sweeps.
    """), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('tmp_cross_section_zenodo.txt'))
    parser.add_argument('--output', type=Path, default=Path('examples/chemistry_argon_lxcat'))
    args = parser.parse_args()

    if not args.source.exists():
        _download(args.source)
    blocks = parse_blocks(args.source)
    momentum, ionization, excitations = select_first_argon_set(blocks)
    write_bundle(args.output, momentum, ionization, excitations)
    print(f'Wrote {args.output} with {len(excitations)} excitation channels')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
