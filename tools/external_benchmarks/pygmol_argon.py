from __future__ import annotations

import argparse
import csv
import math
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.workflows.context import load_case_from_yaml


DEFAULT_CASE = ROOT / 'examples' / 'configs' / 'case_argon_lxcat.yaml'
DEFAULT_BENCHMARK_MODEL = ROOT / 'tools' / 'external_benchmarks' / 'pygmol_argon_model.yaml'
RATE_MODE_SURROGATE = 'surrogate'


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def load_benchmark_model(path: Path = DEFAULT_BENCHMARK_MODEL) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f'Invalid PyGMol benchmark model YAML: {path}')
    model = data.get('model', {})
    if model.get('scope') != 'external_benchmark_only':
        raise ValueError(f'PyGMol benchmark model must be marked external_benchmark_only: {path}')
    if model.get('rate_alignment_with_local_case') != 'intentionally_not_aligned':
        raise ValueError(f'PyGMol benchmark model must declare intentionally_not_aligned rates: {path}')
    return data


def _step_power_w(step: Any) -> float:
    return sum(_as_float(command.get('value_W', 0.0)) for command in step.power_ports.values())


def _first_feed_sccm(recipe: Any, species_id: str) -> float:
    for step in recipe.steps:
        for flows in step.gas_inlets.values():
            if species_id in flows:
                return _as_float(flows[species_id])
    return 0.0


def _weighted_zone_value(zones: list[Any], attr: str) -> float:
    total_v = sum(float(z.volume_m3) for z in zones)
    if total_v <= 0.0:
        return 0.0
    return sum(float(z.volume_m3) * float(getattr(z, attr)) for z in zones) / total_v


def equivalent_cylinder(total_volume_m3: float, total_area_m2: float, preferred_radius_m: float) -> tuple[float, float]:
    def area_error(radius: float) -> float:
        return 2.0 * math.pi * radius * radius + 2.0 * total_volume_m3 / radius - total_area_m2

    roots: list[float] = []
    r_min = max(0.005, preferred_radius_m * 0.25)
    r_max = max(1.0, preferred_radius_m * 4.0)
    previous_r = r_min
    previous_f = area_error(previous_r)
    for i in range(1, 801):
        r = r_min + (r_max - r_min) * i / 800
        f = area_error(r)
        if f == 0.0:
            roots.append(r)
        elif f * previous_f < 0.0:
            lo, hi = previous_r, r
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if area_error(lo) * area_error(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
        previous_r, previous_f = r, f

    radius = min(roots, key=lambda candidate: abs(candidate - preferred_radius_m)) if roots else preferred_radius_m
    length = total_volume_m3 / (math.pi * radius * radius)
    return radius, length


def case_geometry(loaded: Any) -> dict[str, float]:
    total_volume = sum(float(zone.volume_m3) for zone in loaded.chamber.zones)
    total_area = sum(float(surface.area_m2) for surface in loaded.chamber.surfaces)
    wafer_areas = [float(surface.area_m2) for surface in loaded.chamber.surfaces if surface.kind == 'wafer']
    preferred_radius = math.sqrt((wafer_areas[0] if wafer_areas else total_area / 6.0) / math.pi)
    radius, length = equivalent_cylinder(total_volume, total_area, preferred_radius)
    return {
        'volume_m3': total_volume,
        'area_m2': total_area,
        'radius_m': radius,
        'length_m': length,
    }


def _case_power_series(recipe: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    times: list[float] = []
    powers: list[float] = []
    for step in recipe.steps:
        power = _step_power_w(step)
        times.extend([float(step.t_start_s), float(step.t_end_s)])
        powers.extend([power, power])
    return tuple(times), tuple(powers)


def pygmol_argon_chemistry(model_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    model_spec = model_spec or load_benchmark_model()
    species = model_spec['species']
    reactions = model_spec['reactions']
    return {
        'species_ids': [item['id'] for item in species],
        'species_charges': [item['charge'] for item in species],
        'species_masses': [item['mass_amu'] for item in species],
        'species_lj_sigma_coefficients': [item['lj_sigma'] for item in species],
        'species_surface_sticking_coefficients': [item['surface_sticking'] for item in species],
        'species_surface_return_matrix': model_spec['surface_return_matrix'],
        'reactions_ids': [item['id'] for item in reactions],
        'reactions_strings': [item['equation'] for item in reactions],
        'reactions_arrh_a': [item['arrhenius']['a'] for item in reactions],
        'reactions_arrh_b': [item['arrhenius']['b'] for item in reactions],
        'reactions_arrh_c': [item['arrhenius']['c_eV'] for item in reactions],
        'reactions_el_energy_losses': [item['electron_energy_loss_eV'] for item in reactions],
        'reactions_elastic_flags': [item['elastic'] for item in reactions],
        'reactions_electron_stoich_lhs': [item['electron_stoich']['lhs'] for item in reactions],
        'reactions_electron_stoich_rhs': [item['electron_stoich']['rhs'] for item in reactions],
        'reactions_arbitrary_stoich_lhs': [0 for _ in reactions],
        'reactions_arbitrary_stoich_rhs': [0 for _ in reactions],
        'reactions_species_stoichiomatrix_lhs': [item['species_stoich_lhs'] for item in reactions],
        'reactions_species_stoichiomatrix_rhs': [item['species_stoich_rhs'] for item in reactions],
    }


def _pygmol_parameters(loaded: Any, geometry: dict[str, float]) -> dict[str, Any]:
    t_power, power = _case_power_series(loaded.recipe)
    return {
        'radius': geometry['radius_m'],
        'length': geometry['length_m'],
        'pressure': _weighted_zone_value(loaded.chamber.zones, 'pressure_Pa'),
        'power': power,
        't_power': t_power,
        'feeds': {'Ar': _first_feed_sccm(loaded.recipe, 'Ar')},
        'temp_e': 3.0,
        'temp_n': _weighted_zone_value(loaded.chamber.zones, 'gas_temperature_K'),
        't_end': float(loaded.recipe.steps[-1].t_end_s),
    }


def run_pygmol(loaded: Any, geometry: dict[str, float], benchmark_model: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    try:
        from pygmol.model import Model
    except ImportError as exc:
        raise RuntimeError('PyGMol is required. Install it with: py -m pip install ".[compare]"') from exc

    try:
        pygmol_version = version('pygmol')
    except PackageNotFoundError:
        pygmol_version = 'unknown'

    model = Model(pygmol_argon_chemistry(benchmark_model), _pygmol_parameters(loaded, geometry))
    model.run(initial_densities={'Ar': 1.0, 'Ar+': 1.0e-9})
    return model, {'pygmol_version': pygmol_version}


def _local_step_rows(rows: list[dict[str, str]], step_id: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get('step_id') == step_id]


def _pygmol_step_rows(solution: Any, t_start: float, t_end: float) -> list[dict[str, float]]:
    eps = max(1.0e-15, abs(t_end - t_start) * 1.0e-9)
    mask = (solution['t'] >= t_start - eps) & (solution['t'] <= t_end + eps)
    return solution.loc[mask].to_dict('records')


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(candidate: float, reference: float) -> float | None:
    return None if reference == 0.0 else candidate / reference


def _add_ratio_fields(row: dict[str, Any], pygmol_key: str, local_key: str, ratio_key: str) -> None:
    ratio = _ratio(_as_float(row.get(pygmol_key)), _as_float(row.get(local_key)))
    row[ratio_key] = '' if ratio is None else ratio


def build_comparison_rows(loaded: Any, local_rows: list[dict[str, str]], pygmol_model: Any) -> list[dict[str, Any]]:
    solution = pygmol_model.get_solution()
    surface_rates = pygmol_model.get_surface_loss_rates()
    solution = solution.merge(surface_rates[['t', 'Ar+']], on='t', how='left', suffixes=('', '_surface_loss_m3_s'))
    if 'Ar+_surface_loss_m3_s' not in solution.columns:
        solution['Ar+_surface_loss_m3_s'] = solution['Ar+']
    area_over_volume = pygmol_model.equations.area / pygmol_model.equations.volume
    solution['Ar+_wall_flux_m2_s'] = solution['Ar+_surface_loss_m3_s'].abs() / area_over_volume

    rows: list[dict[str, Any]] = []
    for step in loaded.recipe.steps:
        step_id = step.step_id
        local_step = _local_step_rows(local_rows, step_id)
        pygmol_step = _pygmol_step_rows(solution, float(step.t_start_s), float(step.t_end_s))
        if not local_step or not pygmol_step:
            continue

        local_final = local_step[-1]
        pygmol_final = pygmol_step[-1]
        row: dict[str, Any] = {
            'step_id': step_id,
            't_start_s': float(step.t_start_s),
            't_end_s': float(step.t_end_s),
            'local_final_ne_m3': _as_float(local_final.get('electron_density_m3')),
            'pygmol_final_ne_m3': _as_float(pygmol_final.get('e')),
            'local_mean_ne_m3': _mean([_as_float(item.get('electron_density_m3')) for item in local_step]),
            'pygmol_mean_ne_m3': _mean([_as_float(item.get('e')) for item in pygmol_step]),
            'local_final_mean_energy_eV': _as_float(local_final.get('mean_electron_energy_eV')),
            'pygmol_final_Te_eV': _as_float(pygmol_final.get('T_e')),
            'pygmol_final_mean_energy_equiv_eV': 1.5 * _as_float(pygmol_final.get('T_e')),
            'local_mean_absorbed_power_W': _mean([_as_float(item.get('total_absorbed_power_W')) for item in local_step]),
            'pygmol_mean_absorbed_power_W': _mean([_as_float(item.get('P')) for item in pygmol_step]),
            'local_final_wafer_ion_flux_m2_s': _as_float(local_final.get('ion_flux_wafer_m2_s')),
            'pygmol_final_cylinder_wall_Arplus_flux_m2_s': _as_float(pygmol_final.get('Ar+_wall_flux_m2_s')),
        }
        _add_ratio_fields(row, 'pygmol_final_ne_m3', 'local_final_ne_m3', 'ratio_final_ne_pygmol_over_local')
        _add_ratio_fields(row, 'pygmol_mean_ne_m3', 'local_mean_ne_m3', 'ratio_mean_ne_pygmol_over_local')
        _add_ratio_fields(row, 'pygmol_final_mean_energy_equiv_eV', 'local_final_mean_energy_eV', 'ratio_final_mean_energy_equiv_pygmol_over_local')
        _add_ratio_fields(row, 'pygmol_mean_absorbed_power_W', 'local_mean_absorbed_power_W', 'ratio_mean_absorbed_power_pygmol_over_local')
        rows.append(row)
    return rows


def ensure_local_outputs(case_path: Path, rerun_local: bool) -> tuple[Any, Path]:
    loaded = load_case_from_yaml(case_path)
    output_dir = Path(loaded.resolved_paths.output_dir)
    observables_path = output_dir / 'observables.csv'
    if rerun_local or not observables_path.exists():
        from plasma_global.workflows.runner import run_from_yaml

        run_from_yaml(case_path)
    return loaded, observables_path


def build_report(
    case_path: Path = DEFAULT_CASE,
    *,
    rerun_local: bool = False,
    write_solution_csv: bool = True,
) -> dict[str, Any]:
    loaded, observables_path = ensure_local_outputs(case_path.resolve(), rerun_local)
    local_rows = _read_csv_rows(observables_path)
    output_dir = observables_path.parent
    geometry = case_geometry(loaded)
    benchmark_model = load_benchmark_model()
    pygmol_model, package_metadata = run_pygmol(loaded, geometry, benchmark_model)
    comparison_rows = build_comparison_rows(loaded, local_rows, pygmol_model)

    summary_path = output_dir / 'comparison_pygmol_summary.csv'
    fieldnames = [
        'step_id',
        't_start_s',
        't_end_s',
        'local_final_ne_m3',
        'pygmol_final_ne_m3',
        'ratio_final_ne_pygmol_over_local',
        'local_mean_ne_m3',
        'pygmol_mean_ne_m3',
        'ratio_mean_ne_pygmol_over_local',
        'local_final_mean_energy_eV',
        'pygmol_final_Te_eV',
        'pygmol_final_mean_energy_equiv_eV',
        'ratio_final_mean_energy_equiv_pygmol_over_local',
        'local_mean_absorbed_power_W',
        'pygmol_mean_absorbed_power_W',
        'ratio_mean_absorbed_power_pygmol_over_local',
        'local_final_wafer_ion_flux_m2_s',
        'pygmol_final_cylinder_wall_Arplus_flux_m2_s',
    ]
    _write_csv(summary_path, comparison_rows, fieldnames)
    solution_path = output_dir / 'comparison_pygmol_solution.csv'
    if write_solution_csv:
        pygmol_model.get_solution().to_csv(solution_path, index=False)
    metadata_path = output_dir / 'comparison_pygmol_metadata.yaml'
    metadata = {
        'comparison': 'argon_lxcat_vs_pygmol_minimal_argon',
        'local_case': str(Path(loaded.resolved_paths.source_config)),
        'pygmol': {
            **package_metadata,
            'project_url': 'https://github.com/hanicinecm/pygmol',
            'pypi_url': 'https://pypi.org/project/pygmol/',
            'benchmark_model_file': str(DEFAULT_BENCHMARK_MODEL),
            'benchmark_model_id': benchmark_model['model']['id'],
            'rate_alignment_with_local_case': benchmark_model['model']['rate_alignment_with_local_case'],
            'rate_source': benchmark_model['model']['rate_source'],
            'guardrails': benchmark_model['model']['guardrails'],
        },
        'equivalent_cylinder': geometry,
        'limitations': [
            'Local model is two-zone; PyGMol comparison is a single equivalent cylinder.',
            'Local electron energy is EEDF mean energy; PyGMol reports electron temperature.',
            'Local wafer ion flux and PyGMol total cylinder-wall Ar+ flux are not identical observables.',
            'The PyGMol chemistry file is a compact external surrogate, not the local LXCat mechanism.',
        ],
    }
    _write_yaml(metadata_path, metadata)

    powered_rows = [row for row in comparison_rows if _as_float(row.get('local_mean_absorbed_power_W')) > 0.0]
    ratios = [_as_float(row.get('ratio_final_ne_pygmol_over_local'), default=float('nan')) for row in powered_rows]
    finite_ratios = [ratio for ratio in ratios if math.isfinite(ratio) and ratio > 0.0]
    passed = bool(finite_ratios) and all(1.0e-3 <= ratio <= 1.0e3 for ratio in finite_ratios)
    return {
        'tool': 'pygmol_argon',
        'case': str(case_path.resolve()),
        'rate_mode': RATE_MODE_SURROGATE,
        'output_dir': str(output_dir),
        'comparison_file': str(summary_path),
        'solution_file': str(solution_path) if write_solution_csv else None,
        'metadata_file': str(metadata_path),
        'passed': passed,
        'summary': {
            'powered_step_count': len(powered_rows),
            'powered_ne_ratio_min_pygmol_over_local': min(finite_ratios) if finite_ratios else None,
            'powered_ne_ratio_max_pygmol_over_local': max(finite_ratios) if finite_ratios else None,
            'rate_alignment_with_local_case': benchmark_model['model']['rate_alignment_with_local_case'],
        },
        'rows': comparison_rows,
        'assessment': (
            'PyGMol is treated as an executable global-model sanity benchmark. Passing means powered-step '
            'electron densities are within three orders of magnitude.'
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='External-tool PyGMol sanity comparison for the local pure-Ar LXCat case.')
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--rerun-local', action='store_true')
    parser.add_argument('--no-solution-csv', action='store_true')
    parser.add_argument('--write', type=Path, default=None, help='Optional YAML report path; defaults to <output_dir>/comparison_pygmol_report.yaml')
    args = parser.parse_args(argv)

    try:
        report = build_report(
            args.case,
            rerun_local=args.rerun_local,
            write_solution_csv=not args.no_solution_csv,
        )
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    out = args.write or (
        Path(report['output_dir'])
        / 'comparison_pygmol_report.yaml'
    )
    _write_yaml(out, report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
