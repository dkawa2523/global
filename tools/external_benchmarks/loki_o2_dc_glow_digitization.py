from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_candidate.yaml'
DEFAULT_REPORT = ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitization_report.yaml'


SCHEMA_DEFINITIONS: dict[str, dict[str, Any]] = {
    'pressure_sweep': {
        'columns': [
            'pressure_Torr',
            'series_id',
            'quantity_id',
            'symbol',
            'value',
            'unit',
            'source_figure',
            'source_panel',
            'digitization_method',
            'note',
        ],
        'required_nonempty_columns': [
            'pressure_Torr',
            'series_id',
            'quantity_id',
            'value',
            'unit',
            'source_figure',
        ],
        'description': 'Pressure-sweep digitization points for paired 1D and 0D curves.',
    },
    'profile_curve': {
        'columns': [
            'pressure_Torr',
            'radius_mm',
            'series_id',
            'quantity_id',
            'symbol',
            'value',
            'unit',
            'source_figure',
            'source_panel',
            'digitization_method',
            'note',
        ],
        'required_nonempty_columns': [
            'pressure_Torr',
            'radius_mm',
            'series_id',
            'quantity_id',
            'value',
            'unit',
            'source_figure',
        ],
        'description': 'Radial-profile digitization points at one pressure or a small pressure set.',
    },
    'acceptance_envelope': {
        'columns': [
            'pressure_Torr',
            'metric_id',
            'value_percent',
            'source_figure',
            'source_panel',
            'digitization_method',
            'note',
        ],
        'required_nonempty_columns': [
            'pressure_Torr',
            'metric_id',
            'value_percent',
            'source_figure',
        ],
        'description': 'Relative-difference envelope from the paper-level acceptance summary.',
    },
}


DATASET_BLUEPRINTS: dict[str, dict[str, Any]] = {
    'fig05a_avg_temperatures_vs_pressure.csv': {
        'subdir': 'pressure_sweep',
        'schema_name': 'pressure_sweep',
        'priority': 'primary_required',
        'source_figure': '5a',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_quantities': [
            {'quantity_id': 'average_gas_temperature', 'symbol': 'Tg,av', 'unit': 'K'},
            {'quantity_id': 'near_wall_temperature', 'symbol': 'Tnw', 'unit': 'K'},
        ],
    },
    'fig05b_avg_reduced_field_vs_pressure.csv': {
        'subdir': 'pressure_sweep',
        'schema_name': 'pressure_sweep',
        'priority': 'primary_required',
        'source_figure': '5b',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_quantities': [
            {'quantity_id': 'average_reduced_electric_field', 'symbol': 'E_over_N', 'unit': 'Td'},
        ],
    },
    'fig06_molar_fractions_vs_pressure.csv': {
        'subdir': 'pressure_sweep',
        'schema_name': 'pressure_sweep',
        'priority': 'primary_required',
        'source_figure': '6',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_quantities': [
            {'quantity_id': 'average_molar_fraction_atomic_oxygen', 'symbol': '[O]/ng', 'unit': '1'},
            {'quantity_id': 'average_molar_fraction_molecular_oxygen', 'symbol': '[O2]/ng', 'unit': '1'},
        ],
    },
    'fig07a_main_charged_species_vs_pressure.csv': {
        'subdir': 'pressure_sweep',
        'schema_name': 'pressure_sweep',
        'priority': 'primary_required',
        'source_figure': '7a',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_quantities': [
            {'quantity_id': 'average_electron_density', 'symbol': 'ne', 'unit': 'm^-3'},
            {'quantity_id': 'average_negative_ion_density', 'symbol': 'O_minus', 'unit': 'm^-3'},
            {'quantity_id': 'average_positive_ion_density', 'symbol': 'O2_plus', 'unit': 'm^-3'},
        ],
    },
    'fig08_atomic_excited_states_vs_pressure.csv': {
        'subdir': 'pressure_sweep',
        'schema_name': 'pressure_sweep',
        'priority': 'secondary_pressure_sweep',
        'source_figure': '8',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_quantities': [
            {'quantity_id': 'average_atomic_excited_state_density', 'symbol': 'O(3P)', 'unit': 'm^-3'},
            {'quantity_id': 'average_atomic_excited_state_density', 'symbol': 'O(1D)', 'unit': 'm^-3'},
            {'quantity_id': 'average_atomic_excited_state_density', 'symbol': 'O(1S)', 'unit': 'm^-3'},
        ],
    },
    'fig09a_molecular_states_vs_pressure.csv': {
        'subdir': 'pressure_sweep',
        'schema_name': 'pressure_sweep',
        'priority': 'secondary_pressure_sweep',
        'source_figure': '9a',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_quantities': [
            {'quantity_id': 'average_molecular_state_density', 'symbol': 'O2(X)', 'unit': 'm^-3'},
            {'quantity_id': 'average_molecular_state_density', 'symbol': 'O2(a)', 'unit': 'm^-3'},
            {'quantity_id': 'average_molecular_state_density', 'symbol': 'O2(b)', 'unit': 'm^-3'},
        ],
    },
    'fig09b_ozone_and_herzberg_vs_pressure.csv': {
        'subdir': 'pressure_sweep',
        'schema_name': 'pressure_sweep',
        'priority': 'secondary_pressure_sweep',
        'source_figure': '9b',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_quantities': [
            {'quantity_id': 'average_molecular_state_density', 'symbol': 'O2(Hz)', 'unit': 'm^-3'},
            {'quantity_id': 'average_molecular_state_density', 'symbol': 'O3', 'unit': 'm^-3'},
        ],
    },
    'fig11_relative_difference_envelope_vs_pressure.csv': {
        'subdir': 'acceptance_envelope_reference',
        'schema_name': 'acceptance_envelope',
        'priority': 'primary_required',
        'source_figure': '11',
        'pressure_grid_override_Torr': [0.5, 1.0, 2.0, 4.0, 7.5, 10.0],
        'expected_metric_ids': [
            'avg_gas_temperature_relative_difference',
            'near_wall_temperature_relative_difference',
            'electron_temperature_relative_difference',
            'avg_reduced_electric_field_relative_difference',
            'avg_charged_species_relative_difference',
            'avg_neutral_species_relative_difference',
        ],
    },
    'fig01_gas_temperature_profiles.csv': {
        'subdir': 'profile_assumption_diagnostics',
        'schema_name': 'profile_curve',
        'priority': 'diagnostic_profile',
        'source_figure': '1',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_profile_groups': ['1Torr', '10Torr'],
        'expected_quantities': [
            {'quantity_id': 'gas_temperature_profile', 'symbol': 'Tg(r)', 'unit': 'K'},
        ],
    },
    'fig02_reduced_field_profiles.csv': {
        'subdir': 'profile_assumption_diagnostics',
        'schema_name': 'profile_curve',
        'priority': 'diagnostic_profile',
        'source_figure': '2',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_profile_groups': ['1Torr', '10Torr'],
        'expected_quantities': [
            {'quantity_id': 'reduced_electric_field_profile', 'symbol': 'E_over_N(r)', 'unit': 'Td'},
        ],
    },
    'fig03_electron_density_profiles.csv': {
        'subdir': 'profile_assumption_diagnostics',
        'schema_name': 'profile_curve',
        'priority': 'diagnostic_profile',
        'source_figure': '3',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_profile_groups': ['1Torr', '10Torr'],
        'expected_quantities': [
            {'quantity_id': 'electron_density_profile', 'symbol': 'ne(r)', 'unit': 'm^-3'},
        ],
    },
    'fig04_atomic_oxygen_profiles.csv': {
        'subdir': 'profile_assumption_diagnostics',
        'schema_name': 'profile_curve',
        'priority': 'diagnostic_profile',
        'source_figure': '4',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_profile_groups': ['1Torr', '10Torr'],
        'expected_quantities': [
            {'quantity_id': 'atomic_oxygen_density_profile', 'symbol': 'O(3P)(r)', 'unit': 'm^-3'},
        ],
    },
    'fig07b_charged_species_profiles_10Torr.csv': {
        'subdir': 'profile_assumption_diagnostics',
        'schema_name': 'profile_curve',
        'priority': 'diagnostic_profile',
        'source_figure': '7b',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_profile_groups': ['10Torr'],
        'expected_quantities': [
            {'quantity_id': 'electron_density_profile', 'symbol': 'e(r)', 'unit': 'm^-3'},
            {'quantity_id': 'negative_ion_density_profile', 'symbol': 'O_minus(r)', 'unit': 'm^-3'},
            {'quantity_id': 'positive_ion_density_profile', 'symbol': 'O2_plus(r)', 'unit': 'm^-3'},
        ],
    },
    'fig10_ozone_reactant_profiles_10Torr.csv': {
        'subdir': 'profile_assumption_diagnostics',
        'schema_name': 'profile_curve',
        'priority': 'diagnostic_profile',
        'source_figure': '10',
        'expected_series': ['1D_fluid', '0D_LoKI'],
        'expected_profile_groups': ['10Torr'],
        'expected_quantities': [
            {'quantity_id': 'ozone_reactant_profile', 'symbol': 'digitized_reactant', 'unit': 'm^-3'},
        ],
        'note': 'Figure 10 contains several reactant curves. Use quantity_id and symbol to distinguish them when digitizing.',
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _resolve_relative(path_str: str, workspace_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (workspace_root / path).resolve()


def _dataset_specs(spec: dict[str, Any], workspace_root: Path) -> list[dict[str, Any]]:
    benchmark = spec['benchmark']
    plan = benchmark['digitization_plan']
    pressure_grid = list(benchmark['discharge_definition']['pressure_grid_Torr']['values'])
    expected_names = set(plan['first_wave_outputs']) | set(plan['second_wave_outputs'])
    unknown = sorted(expected_names - set(DATASET_BLUEPRINTS))
    if unknown:
        raise ValueError(f'Unsupported LoKI digitization files in spec: {unknown!r}')

    root_dir = _resolve_relative(str(plan['destination_dir']), workspace_root)
    specs: list[dict[str, Any]] = []
    for filename in list(plan['first_wave_outputs']) + list(plan['second_wave_outputs']):
        blueprint = dict(DATASET_BLUEPRINTS[filename])
        schema = SCHEMA_DEFINITIONS[blueprint['schema_name']]
        file_path = root_dir / blueprint['subdir'] / filename
        entry = {
            'filename': filename,
            'path': file_path,
            'relative_path': str(file_path.relative_to(workspace_root.resolve())).replace('\\', '/'),
            'schema_name': blueprint['schema_name'],
            'schema_columns': list(schema['columns']),
            'priority': blueprint['priority'],
            'source_figure': blueprint['source_figure'],
            'expected_series': list(blueprint.get('expected_series', [])),
            'expected_quantities': list(blueprint.get('expected_quantities', [])),
            'expected_metric_ids': list(blueprint.get('expected_metric_ids', [])),
            'expected_profile_groups': list(blueprint.get('expected_profile_groups', [])),
            'pressure_grid_override_Torr': list(blueprint.get('pressure_grid_override_Torr', [])),
            'note': blueprint.get('note'),
        }
        pressure_values = entry['pressure_grid_override_Torr'] or pressure_grid
        if entry['schema_name'] == 'pressure_sweep':
            entry['expected_row_count'] = (
                len(pressure_values)
                * max(len(entry['expected_series']), 1)
                * max(len(entry['expected_quantities']), 1)
            )
        elif entry['schema_name'] == 'acceptance_envelope':
            entry['expected_row_count'] = len(pressure_values) * max(len(entry['expected_metric_ids']), 1)
        else:
            entry['expected_row_count'] = None
        specs.append(entry)
    return specs


def _scaffold_readme() -> str:
    return """# LoKI O2 DC Glow Digitization Scaffold

This directory is an external-only scaffold for digitizing the public
LoKI O2 DC glow benchmark paper.

Rules:
- Keep CSV headers unchanged.
- Put raw digitized points into the provided CSV files.
- Do not mix units inside one file.
- Keep 1D and 0D series separated with `series_id`.
- Run `py tools\\external_benchmarks\\loki_o2_dc_glow_digitization.py`
  after editing files to validate headers and readiness.

This scaffold does not depend on `plasma_global` core modules.
"""


def _schema_report(spec: dict[str, Any], dataset_specs: list[dict[str, Any]], workspace_root: Path) -> dict[str, Any]:
    return {
        'tool': 'loki_o2_dc_glow_digitization_schema',
        'scope': 'external_candidate_only',
        'source_spec': str(Path(spec['__spec_path__']).resolve()),
        'digitization_root': str(_resolve_relative(spec['benchmark']['digitization_plan']['destination_dir'], workspace_root)),
        'schema_definitions': SCHEMA_DEFINITIONS,
        'datasets': [
            {
                'file': item['relative_path'],
                'schema_name': item['schema_name'],
                'priority': item['priority'],
                'source_figure': item['source_figure'],
                'expected_row_count': item['expected_row_count'],
                'expected_series': item['expected_series'],
                'expected_quantities': item['expected_quantities'],
                'expected_metric_ids': item['expected_metric_ids'],
                'expected_profile_groups': item['expected_profile_groups'],
                'pressure_grid_override_Torr': item['pressure_grid_override_Torr'],
                'note': item['note'],
            }
            for item in dataset_specs
        ],
    }


def ensure_scaffold(spec_path: Path = DEFAULT_SPEC, *, workspace_root: Path = ROOT) -> dict[str, Any]:
    spec = _load_yaml(spec_path)
    spec['__spec_path__'] = str(spec_path)
    dataset_specs = _dataset_specs(spec, workspace_root)
    root_dir = _resolve_relative(spec['benchmark']['digitization_plan']['destination_dir'], workspace_root)
    root_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ('pressure_sweep', 'acceptance_envelope_reference', 'profile_assumption_diagnostics'):
        (root_dir / subdir).mkdir(parents=True, exist_ok=True)

    readme_path = root_dir / 'README.md'
    readme_path.write_text(_scaffold_readme(), encoding='utf-8')

    schema_path = root_dir / 'schema.yaml'
    _write_yaml(schema_path, _schema_report(spec, dataset_specs, workspace_root))

    for item in dataset_specs:
        path = item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(item['schema_columns'])

    return {
        'spec': spec,
        'dataset_specs': dataset_specs,
        'digitization_root': root_dir,
        'schema_file': schema_path,
        'readme_file': readme_path,
    }


def _read_csv_header_and_rows(path: Path) -> tuple[list[str], int]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], 0
    header = [str(item).strip() for item in rows[0]]
    data_rows = sum(1 for row in rows[1:] if any(str(cell).strip() for cell in row))
    return header, data_rows


def _validate_dataset(item: dict[str, Any]) -> dict[str, Any]:
    path = item['path']
    result = {
        'file': item['relative_path'],
        'priority': item['priority'],
        'schema_name': item['schema_name'],
        'source_figure': item['source_figure'],
        'exists': path.exists(),
        'header_matches_schema': False,
        'data_row_count': 0,
        'expected_row_count': item['expected_row_count'],
        'complete_for_expected_grid': False,
        'status': 'missing',
        'expected_series': item['expected_series'],
        'expected_quantities': item['expected_quantities'],
        'expected_metric_ids': item['expected_metric_ids'],
        'expected_profile_groups': item['expected_profile_groups'],
    }
    if not path.exists():
        return result
    header, data_rows = _read_csv_header_and_rows(path)
    result['data_row_count'] = data_rows
    result['header_matches_schema'] = header == item['schema_columns']
    if not result['header_matches_schema']:
        result['status'] = 'wrong_header'
        return result
    if item['expected_row_count'] is not None:
        result['complete_for_expected_grid'] = data_rows >= int(item['expected_row_count'])
    else:
        result['complete_for_expected_grid'] = data_rows > 0
    if data_rows == 0:
        result['status'] = 'schema_only'
    elif result['complete_for_expected_grid']:
        result['status'] = 'digitized_complete'
    else:
        result['status'] = 'digitized_partial'
    return result


def build_report(
    spec_path: Path = DEFAULT_SPEC,
    *,
    workspace_root: Path = ROOT,
    write_files: bool = True,
    report_path: Path | None = DEFAULT_REPORT,
) -> dict[str, Any]:
    scaffold = ensure_scaffold(spec_path, workspace_root=workspace_root) if write_files else {
        'spec': _load_yaml(spec_path) | {'__spec_path__': str(spec_path)},
        'dataset_specs': _dataset_specs(_load_yaml(spec_path) | {'__spec_path__': str(spec_path)}, workspace_root),
        'digitization_root': _resolve_relative(str(_load_yaml(spec_path)['benchmark']['digitization_plan']['destination_dir']), workspace_root),
        'schema_file': _resolve_relative(str(_load_yaml(spec_path)['benchmark']['digitization_plan']['destination_dir']), workspace_root) / 'schema.yaml',
        'readme_file': _resolve_relative(str(_load_yaml(spec_path)['benchmark']['digitization_plan']['destination_dir']), workspace_root) / 'README.md',
    }
    spec = scaffold['spec']
    dataset_specs = scaffold['dataset_specs']
    validations = [_validate_dataset(item) for item in dataset_specs]

    expected = len(validations)
    present = sum(1 for item in validations if item['exists'])
    good_headers = sum(1 for item in validations if item['header_matches_schema'])
    with_data = sum(1 for item in validations if item['data_row_count'] > 0)
    primary = [item for item in validations if item['priority'] == 'primary_required']
    secondary = [item for item in validations if item['priority'] == 'secondary_pressure_sweep']
    diagnostics = [item for item in validations if item['priority'] == 'diagnostic_profile']

    primary_complete = sum(1 for item in primary if item['complete_for_expected_grid'])
    secondary_complete = sum(1 for item in secondary if item['complete_for_expected_grid'])
    diagnostics_started = sum(1 for item in diagnostics if item['data_row_count'] > 0)

    runnable_primary = bool(primary) and all(item['complete_for_expected_grid'] for item in primary)
    runnable_extended = runnable_primary and bool(secondary) and all(item['complete_for_expected_grid'] for item in secondary)
    runnable_full = runnable_extended and bool(diagnostics) and all(item['data_row_count'] > 0 for item in diagnostics)
    schema_valid = present == expected and good_headers == expected

    if not schema_valid:
        readiness_stage = 'scaffold_invalid'
    elif with_data == 0:
        readiness_stage = 'scaffold_only'
    elif runnable_full:
        readiness_stage = 'runnable_full_diagnostic_benchmark'
    elif runnable_extended:
        readiness_stage = 'runnable_extended_species_benchmark'
    elif runnable_primary:
        readiness_stage = 'primary_pressure_benchmark_runnable'
    else:
        readiness_stage = 'digitization_in_progress'

    next_actions: list[str] = []
    if not schema_valid:
        next_actions.append('Restore missing files or fix CSV headers so every scaffold file matches schema.yaml.')
    elif not runnable_primary:
        next_actions.append('Populate the primary pressure-sweep datasets from figures 5a, 5b, 6, 7a, and 11.')
    if runnable_primary and not runnable_extended:
        next_actions.append('Populate the secondary pressure-sweep datasets from figures 8, 9a, and 9b.')
    if runnable_extended and not runnable_full:
        next_actions.append('Populate the radial-profile diagnostic datasets from figures 1, 2, 3, 4, 7b, and 10.')

    report = {
        'tool': 'loki_o2_dc_glow_digitization',
        'scope': 'external_candidate_only',
        'spec_file': str(Path(spec['__spec_path__']).resolve()),
        'digitization_root': str(scaffold['digitization_root']),
        'schema_file': str(scaffold['schema_file']),
        'readme_file': str(scaffold['readme_file']),
        'passed': schema_valid,
        'summary': {
            'files_expected': expected,
            'files_present': present,
            'files_with_correct_header': good_headers,
            'files_with_data_rows': with_data,
            'primary_required_files_complete': f'{primary_complete}/{len(primary)}',
            'secondary_pressure_sweep_files_complete': f'{secondary_complete}/{len(secondary)}',
            'diagnostic_profile_files_started': f'{diagnostics_started}/{len(diagnostics)}',
            'runnable_primary_pressure_benchmark': runnable_primary,
            'runnable_extended_species_benchmark': runnable_extended,
            'runnable_full_diagnostic_benchmark': runnable_full,
            'readiness_stage': readiness_stage,
        },
        'evaluation': {
            'execution_mode': 'readiness_check',
            'physics_parity_executed': False,
            'physics_parity_status': (
                'blocked_waiting_for_digitized_data'
                if not runnable_primary
                else 'digitized_reference_ready_for_local_case_comparison'
            ),
            'assessment': (
                'The digitization scaffold and CSV schemas are valid.'
                if schema_valid
                else 'The digitization scaffold is incomplete or has header mismatches.'
            ),
            'next_actions': next_actions,
        },
        'datasets': validations,
    }
    if report_path is not None:
        _write_yaml(report_path.resolve(), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create and validate the LoKI O2 DC glow digitization scaffold and report benchmark readiness.')
    parser.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    parser.add_argument('--write', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = build_report(args.spec.resolve(), write_files=True, report_path=args.write.resolve())
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
