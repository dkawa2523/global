from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = Path(tempfile.gettempdir()) / 'zdplaskin_repo_check'
DEFAULT_WRITE = ROOT / 'examples' / 'external' / 'zdplaskin_examples_inventory.yaml'
SOURCE_URL = 'https://github.com/Hemadityamalla/ZDPlaskin'


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def ensure_repo(repo: Path = DEFAULT_REPO) -> Path:
    if (repo / '.git').exists():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    _run(['git', 'clone', '--depth', '1', SOURCE_URL, str(repo)])
    return repo


def _git_commit(repo: Path) -> str | None:
    try:
        return _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _display_repo(repo: Path) -> str:
    try:
        if repo.resolve() == DEFAULT_REPO.resolve():
            return '<temp>/zdplaskin_repo_check'
    except OSError:
        pass
    return str(repo)


def _read_table(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    lines = [line.strip() for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()]
    if not lines:
        return [], []
    header = lines[0].split()
    rows: list[dict[str, float]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < len(header):
            continue
        row: dict[str, float] = {}
        for key, value in zip(header, parts):
            try:
                row[key] = float(value)
            except ValueError:
                pass
        if row:
            rows.append(row)
    return header, rows


def _range(rows: list[dict[str, float]], key: str) -> dict[str, float | None]:
    values = [row[key] for row in rows if key in row]
    if not values:
        return {'min': None, 'max': None}
    return {'min': min(values), 'max': max(values)}


def _final_values(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    if not rows:
        return {}
    final = rows[-1]
    return {key: final[key] for key in keys if key in final}


def _example3_summary(repo: Path) -> dict[str, Any]:
    input_header, input_rows = _read_table(repo / 'example3' / 'data_in.dat')
    output_header, output_rows = _read_table(repo / 'example3' / 'out.dat')
    species = [key for key in output_header if key not in {'TIME', 'E/N'}]
    important_species = ['E', 'N2', 'N2(A)', "N2(A')", 'N2(C)', 'N2^+', 'N3^+', 'N4^+']
    return {
        'path': 'example3',
        'description': 'N2 DBD0D driven-chemistry example with prescribed E/N and prescribed electron density.',
        'available_files': {
            'input_profile': 'example3/data_in.dat',
            'output_profile': 'example3/out.dat',
            'chemistry': 'example3/kinetics.inp',
            'driver': 'example3/main.F90',
        },
        'input_profile': {
            'columns': input_header,
            'row_count': len(input_rows),
            'time_s': _range(input_rows, 'Time_s'),
            'reduced_field_Td': _range(input_rows, 'Field_Td'),
            'electron_density_cm3': _range(input_rows, 'Electrons_cm-3'),
        },
        'output_profile': {
            'columns': output_header,
            'row_count': len(output_rows),
            'time_s': _range(output_rows, 'TIME'),
            'reduced_field_Td': _range(output_rows, 'E/N'),
            'species_count': len(species),
            'final_values_cm3': _final_values(output_rows, important_species),
        },
        'benchmark_status': 'reference_ingested_only',
        'why_not_strict_parity_yet': [
            'The ZDPlaskin driver fixes electron density from data_in.dat at every step.',
            'This code currently treats electron density as a solved plasma state in normal global-model cases.',
            'The N2 mechanism has many excited/cluster-ion states and would need explicit species metadata plus a prescribed-electron-density chemistry-driver mode.',
        ],
        'recommended_next_use': 'Use as a future driven-chemistry regression after adding a clean prescribed-ne, prescribed-E/N driver outside production plasma closures.',
    }


def build_report(repo: Path = DEFAULT_REPO) -> dict[str, Any]:
    repo = ensure_repo(repo)
    example2_output = repo / 'example2' / 'Ar2Reac_1000V_100kOhm_100t_0.4d_0.4r.dat'
    example1_reference_outputs = [
        candidate
        for candidate in (repo / 'example1').glob('*.dat')
        if not candidate.name.lower().startswith('bolsig')
    ]
    return {
        'tool': 'zdplaskin_examples_inventory',
        'source': {
            'repository': SOURCE_URL,
            'local_repo': _display_repo(repo),
            'commit': _git_commit(repo),
        },
        'examples': [
            {
                'path': 'example1',
                'description': 'Minimal Ar ionization plus three-body recombination at fixed E/N.',
                'available_files': {
                    'chemistry': 'example1/kinetics.inp',
                    'driver': 'example1/main.F90',
                },
                'committed_reference_output': bool(example1_reference_outputs),
                'benchmark_status': 'live_run_required',
                'why_not_used_now': [
                    'The repository does not include a clean committed time-history output for example1.',
                    'On this Windows environment the Linux ZDPlaskin/BOLSIG shared library is not directly rerun as the external reference.',
                    'CRANE already covers the same scalar two-reaction Ar ODE axis with committed reference output.',
                ],
            },
            {
                'path': 'example2',
                'description': 'Ar DC-series discharge with circuit output and committed time history.',
                'available_files': {
                    'output': 'example2/Ar2Reac_1000V_100kOhm_100t_0.4d_0.4r.dat',
                    'chemistry': 'example2/kinetics.inp',
                    'driver': 'example2/main.f90',
                    'conditions': 'example2/conditions.ini',
                },
                'committed_reference_output': example2_output.exists(),
                'benchmark_status': 'implemented_strict_external_reference',
                'local_tools': [
                    'tools/external_benchmarks/zdplaskin_example2.py',
                    'tools/external_benchmarks/zdplaskin_eovern.py',
                    'tools/external_benchmarks/same_footing_assessment.py',
                ],
            },
            _example3_summary(repo),
        ],
        'recommendation_order': [
            'Keep example2 as the strict ZDPlaskin same-footing benchmark.',
            'Use example3 next only as a driven-chemistry benchmark after adding prescribed electron-density and prescribed E/N driver support.',
            'Use example1 only if a reproducible external run artifact is generated and committed, because CRANE already validates the same simple ODE axis.',
        ],
        'passed': example2_output.exists(),
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Inventory ZDPlaskin public examples as external benchmark candidates.')
    parser.add_argument('--repo', type=Path, default=Path(os.environ.get('ZDPLASKIN_REPO', DEFAULT_REPO)))
    parser.add_argument('--write', type=Path, default=DEFAULT_WRITE)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.repo)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f'ERROR: could not inspect ZDPlaskin repository: {exc}', file=sys.stderr)
        return 2
    _write_yaml(args.write.resolve(), report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
