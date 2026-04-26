from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external_benchmarks.pygmol_argon import (
    DEFAULT_CASE,
    build_comparison_rows,
    case_geometry,
    ensure_local_outputs,
    equivalent_cylinder,
    load_benchmark_model,
    pygmol_argon_chemistry,
)


DEFAULT_OUTPUT_NAME = 'comparison_pygmol_sensitivity'


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _weighted_zone_value(zones: list[Any], attr: str) -> float:
    total_v = sum(float(z.volume_m3) for z in zones)
    if total_v <= 0.0:
        return 0.0
    return sum(float(z.volume_m3) * float(getattr(z, attr)) for z in zones) / total_v


def _first_feed_sccm(recipe: Any, species_id: str) -> float:
    for step in recipe.steps:
        for flows in step.gas_inlets.values():
            if species_id in flows:
                return _as_float(flows[species_id])
    return 0.0


def _step_power_w(step: Any) -> float:
    return sum(_as_float(command.get('value_W', 0.0)) for command in step.power_ports.values())


def _case_power_series(recipe: Any, scale: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    times: list[float] = []
    powers: list[float] = []
    for step in recipe.steps:
        power = scale * _step_power_w(step)
        times.extend([float(step.t_start_s), float(step.t_end_s)])
        powers.extend([power, power])
    return tuple(times), tuple(powers)


def pygmol_parameters(loaded: Any, geometry: dict[str, float], *, power_scale: float = 1.0) -> dict[str, Any]:
    t_power, power = _case_power_series(loaded.recipe, power_scale)
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


def variant_specs() -> list[dict[str, Any]]:
    return [
        {'id': 'baseline', 'category': 'baseline'},
        {'id': 'ionization_rate_x0p5', 'category': 'rate', 'reaction_id': 'Ar_ionization', 'arrhenius_a_scale': 0.5},
        {'id': 'ionization_rate_x2', 'category': 'rate', 'reaction_id': 'Ar_ionization', 'arrhenius_a_scale': 2.0},
        {'id': 'excitation_rate_x0p5', 'category': 'rate', 'reaction_id': 'Ar_excitation_lumped', 'arrhenius_a_scale': 0.5},
        {'id': 'excitation_rate_x2', 'category': 'rate', 'reaction_id': 'Ar_excitation_lumped', 'arrhenius_a_scale': 2.0},
        {'id': 'energy_loss_x0p5', 'category': 'energy_loss', 'electron_energy_loss_scale': 0.5},
        {'id': 'energy_loss_x1p5', 'category': 'energy_loss', 'electron_energy_loss_scale': 1.5},
        {'id': 'arplus_sticking_x0p5', 'category': 'surface', 'species_id': 'Ar+', 'surface_sticking_scale': 0.5},
        {'id': 'wall_area_x1p5', 'category': 'geometry', 'wall_area_scale': 1.5},
        {'id': 'wall_area_x2', 'category': 'geometry', 'wall_area_scale': 2.0},
        {'id': 'power_x0p75', 'category': 'power', 'power_scale': 0.75},
        {'id': 'power_x1p25', 'category': 'power', 'power_scale': 1.25},
    ]


def apply_variant_to_model(base_model: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    model = copy.deepcopy(base_model)
    if 'reaction_id' in spec:
        for reaction in model['reactions']:
            if reaction['id'] == spec['reaction_id']:
                reaction['arrhenius']['a'] *= spec.get('arrhenius_a_scale', 1.0)
                break
    if 'electron_energy_loss_scale' in spec:
        scale = spec['electron_energy_loss_scale']
        for reaction in model['reactions']:
            reaction['electron_energy_loss_eV'] *= scale
    if 'surface_sticking_scale' in spec:
        scale = spec['surface_sticking_scale']
        for species in model['species']:
            if species['id'] == spec.get('species_id'):
                species['surface_sticking'] = max(0.0, min(1.0, species['surface_sticking'] * scale))
                break
    model['model']['variant_id'] = spec['id']
    model['model']['variant_category'] = spec['category']
    return model


def apply_variant_to_geometry(base_geometry: dict[str, float], spec: dict[str, Any]) -> dict[str, float]:
    scale = spec.get('wall_area_scale')
    if scale is None:
        return dict(base_geometry)
    radius, length = equivalent_cylinder(base_geometry['volume_m3'], base_geometry['area_m2'] * scale, base_geometry['radius_m'])
    return {
        'volume_m3': base_geometry['volume_m3'],
        'area_m2': base_geometry['area_m2'] * scale,
        'radius_m': radius,
        'length_m': length,
    }


def run_pygmol_variant(loaded: Any, local_rows: list[dict[str, str]], base_model: dict[str, Any], base_geometry: dict[str, float], spec: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from pygmol.model import Model
    except ImportError as exc:
        raise RuntimeError('PyGMol is required. Install it with: py -m pip install ".[compare]"') from exc

    model_spec = apply_variant_to_model(base_model, spec)
    geometry = apply_variant_to_geometry(base_geometry, spec)
    parameters = pygmol_parameters(loaded, geometry, power_scale=spec.get('power_scale', 1.0))
    model = Model(pygmol_argon_chemistry(model_spec), parameters)
    model.run(initial_densities={'Ar': 1.0, 'Ar+': 1.0e-9})
    rows = build_comparison_rows(loaded, local_rows, model)
    for row in rows:
        row['variant_id'] = spec['id']
        row['variant_category'] = spec['category']
        row['power_scale'] = spec.get('power_scale', 1.0)
        row['wall_area_scale'] = spec.get('wall_area_scale', 1.0)
        row['surface_sticking_scale'] = spec.get('surface_sticking_scale', 1.0)
        row['arrhenius_a_scale'] = spec.get('arrhenius_a_scale', 1.0)
        row['electron_energy_loss_scale'] = spec.get('electron_energy_loss_scale', 1.0)
    return rows


def _mean(values: list[float]) -> float | None:
    finite = [item for item in values if math.isfinite(item)]
    return sum(finite) / len(finite) if finite else None


def summarize_variant(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    powered = [row for row in rows if _as_float(row.get('local_mean_absorbed_power_W')) > 0.0]
    density_ratios = [_as_float(row.get('ratio_final_ne_pygmol_over_local'), default=float('nan')) for row in powered]
    energy_ratios = [_as_float(row.get('ratio_final_mean_energy_equiv_pygmol_over_local'), default=float('nan')) for row in powered]
    production = next((row for row in rows if row.get('step_id') == 'production'), None)
    return {
        'variant_id': spec['id'],
        'category': spec['category'],
        'powered_density_ratio_min': min(density_ratios) if density_ratios else None,
        'powered_density_ratio_max': max(density_ratios) if density_ratios else None,
        'powered_density_ratio_mean': _mean(density_ratios),
        'powered_energy_ratio_mean': _mean(energy_ratios),
        'production_density_ratio': _as_float(production.get('ratio_final_ne_pygmol_over_local'), default=float('nan')) if production else None,
        'production_energy_ratio': _as_float(production.get('ratio_final_mean_energy_equiv_pygmol_over_local'), default=float('nan')) if production else None,
    }


def build_report(case_path: Path = DEFAULT_CASE, *, rerun_local: bool = False) -> dict[str, Any]:
    loaded, observables_path = ensure_local_outputs(case_path.resolve(), rerun_local)
    local_rows = _read_csv_rows(observables_path)
    base_model = load_benchmark_model()
    base_geometry = case_geometry(loaded)

    all_step_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in variant_specs():
        rows = run_pygmol_variant(loaded, local_rows, base_model, base_geometry, spec)
        all_step_rows.extend(rows)
        summary_rows.append(summarize_variant(rows, spec))

    output_dir = observables_path.parent
    summary_path = output_dir / f'{DEFAULT_OUTPUT_NAME}_summary.csv'
    step_path = output_dir / f'{DEFAULT_OUTPUT_NAME}_steps.csv'
    report_path = output_dir / f'{DEFAULT_OUTPUT_NAME}.yaml'

    summary_fields = [
        'variant_id',
        'category',
        'powered_density_ratio_min',
        'powered_density_ratio_max',
        'powered_density_ratio_mean',
        'powered_energy_ratio_mean',
        'production_density_ratio',
        'production_energy_ratio',
    ]
    step_fields = [
        'variant_id',
        'variant_category',
        'step_id',
        'local_final_ne_m3',
        'pygmol_final_ne_m3',
        'ratio_final_ne_pygmol_over_local',
        'local_final_mean_energy_eV',
        'pygmol_final_mean_energy_equiv_eV',
        'ratio_final_mean_energy_equiv_pygmol_over_local',
        'local_mean_absorbed_power_W',
        'pygmol_mean_absorbed_power_W',
        'local_final_wafer_ion_flux_m2_s',
        'pygmol_final_cylinder_wall_Arplus_flux_m2_s',
        'power_scale',
        'wall_area_scale',
        'surface_sticking_scale',
        'arrhenius_a_scale',
        'electron_energy_loss_scale',
    ]
    _write_csv(summary_path, summary_rows, summary_fields)
    _write_csv(step_path, all_step_rows, step_fields)
    report = {
        'tool': 'pygmol_sensitivity',
        'scope': 'external_benchmark_only',
        'case': str(case_path.resolve()),
        'output_dir': str(output_dir),
        'summary_csv': str(summary_path),
        'steps_csv': str(step_path),
        'benchmark_model': {
            'file': str(ROOT / 'tools' / 'external_benchmarks' / 'pygmol_argon_model.yaml'),
            'rate_alignment_with_local_case': base_model['model']['rate_alignment_with_local_case'],
            'guardrails': base_model['model']['guardrails'],
        },
        'interpretation': [
            'This sweep changes only the external PyGMol benchmark model and PyGMol input mapping.',
            'It must not be used to tune plasma_global production chemistry or core physics.',
            'Powered-step density ratios indicate sensitivity of the PyGMol sanity check, not local solver correctness.',
        ],
        'variant_count': len(summary_rows),
        'summary': summary_rows,
    }
    _write_yaml(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run PyGMol-only sensitivity sweeps without modifying the core solver case.')
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--rerun-local', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = build_report(args.case, rerun_local=args.rerun_local)
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
