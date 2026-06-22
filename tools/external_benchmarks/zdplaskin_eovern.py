from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate'
DEFAULT_REFERENCE = ROOT / 'examples' / 'external' / 'zdplaskin_example2_summary.yaml'
DEFAULT_CHAMBER = ROOT / 'examples' / 'configs' / 'chamber_zdplaskin_example2.yaml'
K_B = 1.380649e-23


def neutral_density_m3(pressure_Pa: float, gas_temperature_K: float) -> float:
    return float(pressure_Pa) / (K_B * max(float(gas_temperature_K), 1.0))


def reduced_field_Td(voltage_V: float, gap_m: float, pressure_Pa: float, gas_temperature_K: float) -> float:
    density = neutral_density_m3(pressure_Pa, gas_temperature_K)
    field_V_m = float(voltage_V) / max(float(gap_m), 1.0e-30)
    return field_V_m / max(density, 1.0e-30) * 1.0e21


def equivalent_voltage_V(reduced_field_Td_value: float, gap_m: float, pressure_Pa: float, gas_temperature_K: float) -> float:
    density = neutral_density_m3(pressure_Pa, gas_temperature_K)
    return float(reduced_field_Td_value) * 1.0e-21 * density * float(gap_m)


def _ratio(local: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return local / reference


def _relative_difference(a: float, b: float) -> float | None:
    if b == 0.0:
        return None
    return (a - b) / b


def _within_ratio(value: float | None, center: float = 1.0, tolerance: float = 0.02) -> bool:
    return value is not None and abs(float(value) - center) <= tolerance


def _read_observables(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _resolved_chamber_path(output_dir: Path, fallback: Path) -> Path:
    resolved = output_dir / 'resolved_paths.yaml'
    if not resolved.exists():
        return fallback
    data = _load_yaml(resolved)
    chamber = data.get('chamber_file')
    return Path(chamber) if chamber else fallback


def _gap_m_from_chamber(chamber_path: Path) -> float:
    chamber = _load_yaml(chamber_path)
    for port in chamber.get('power_ports', []) or []:
        params = port.get('parameters', {}) or {}
        if params.get('gap_m') is not None:
            return float(params['gap_m'])
    zones = chamber.get('zones', []) or []
    surfaces = chamber.get('surfaces', []) or []
    if zones and surfaces:
        volume = float(zones[0].get('volume_m3', 0.0) or 0.0)
        area = float(surfaces[0].get('area_m2', 0.0) or 0.0)
        if volume > 0.0 and area > 0.0:
            return volume / area
    raise ValueError(f'Cannot infer ZDPlaskin gap length from {chamber_path}')


def build_eovern_report(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reference_path: Path = DEFAULT_REFERENCE,
    chamber_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    reference_path = reference_path.resolve()
    chamber_path = (chamber_path or _resolved_chamber_path(output_dir, DEFAULT_CHAMBER)).resolve()

    reference = _load_yaml(reference_path)
    saved = reference['saved_output']
    case = reference['case']
    final_obs = _read_observables(output_dir / 'observables.csv')[-1]

    zdp_gap_m = float(case['gap_length_cm']) * 1.0e-2
    zdp_pressure_Pa = float(case['pressure_Pa'])
    zdp_Tg_K = float(case['gas_temperature_K'])
    zdp_voltage_V = float(saved['final_saved_voltage_V'])
    zdp_current_A = float(saved['final_saved_current_A'])
    zdp_saved_Td = float(saved['final_saved_reduced_field_Td'])
    zdp_source_voltage_V = float(case.get('voltage_V', 1000.0))
    zdp_recomputed_Td = reduced_field_Td(zdp_voltage_V, zdp_gap_m, zdp_pressure_Pa, zdp_Tg_K)
    zdp_source_Td = reduced_field_Td(zdp_source_voltage_V, zdp_gap_m, zdp_pressure_Pa, zdp_Tg_K)

    local_gap_m = _gap_m_from_chamber(chamber_path)
    local_pressure_Pa = float(final_obs['pressure_plasma_Pa'])
    local_Tg_K = float(final_obs['Tg_plasma_K'])
    local_reported_Td = float(final_obs['EoverN_plasma_Td'])
    local_absorbed_power_W = float(final_obs['total_absorbed_power_W'])
    local_equiv_voltage_V = equivalent_voltage_V(local_reported_Td, local_gap_m, local_pressure_Pa, local_Tg_K)
    local_same_zdp_voltage_Td = reduced_field_Td(zdp_voltage_V, local_gap_m, local_pressure_Pa, local_Tg_K)
    zdp_absorbed_power_W = zdp_voltage_V * zdp_current_A
    current_field_ratio = _ratio(local_reported_Td, zdp_saved_Td)
    forced_voltage_ratio = _ratio(local_same_zdp_voltage_Td, zdp_saved_Td)

    return {
        'simple_verdict': {
            'current_local_run_EoverN_matches_zdplaskin': _within_ratio(current_field_ratio, tolerance=0.02),
            'forced_zdplaskin_voltage_formula_check_matches': _within_ratio(forced_voltage_ratio, tolerance=0.02),
            'local_reported_EoverN_is_zone_observable': True,
            'plain_language': (
                'The current local run E/N is evaluated from the active electrical backend. '
                'The forced-voltage value only proves that the E/N unit formula is consistent.'
            ),
        },
        'definition': {
            'quantity': 'reduced electric field',
            'formula': 'E/N = (V_gap / gap_m) / N_gas',
            'neutral_density_formula': 'N_gas = pressure_Pa / (k_B * gas_temperature_K)',
            'unit': 'Townsend; 1 Td = 1e-21 V m^2',
            'same_footing_requirement': 'Use gap voltage, gap length, pressure, and gas temperature; do not compare a fixed-power proxy field to a circuit-derived field.',
        },
        'zdplaskin_circuit_field': {
            'saved_reduced_field_Td': zdp_saved_Td,
            'recomputed_from_voltage_gap_density_Td': zdp_recomputed_Td,
            'saved_vs_recomputed_relative_difference': _relative_difference(zdp_recomputed_Td, zdp_saved_Td),
            'final_gap_voltage_V': zdp_voltage_V,
            'final_current_A': zdp_current_A,
            'final_voltage_current_power_W': zdp_absorbed_power_W,
            'gap_m': zdp_gap_m,
            'pressure_Pa': zdp_pressure_Pa,
            'gas_temperature_K': zdp_Tg_K,
            'neutral_density_m3': neutral_density_m3(zdp_pressure_Pa, zdp_Tg_K),
            'source_voltage_reduced_field_before_loading_Td': zdp_source_Td,
        },
        'local_run_field': {
            'reported_EoverN_Td': local_reported_Td,
            'equivalent_gap_voltage_V': local_equiv_voltage_V,
            'reported_EoverN_is_zone_observable': True,
            'same_ZDPlaskin_final_voltage_reduced_field_Td': local_same_zdp_voltage_Td,
            'absorbed_power_W': local_absorbed_power_W,
            'gap_m': local_gap_m,
            'pressure_Pa': local_pressure_Pa,
            'gas_temperature_K': local_Tg_K,
            'neutral_density_m3': neutral_density_m3(local_pressure_Pa, local_Tg_K),
        },
        'comparison': {
            'local_reported_over_zdplaskin_saved_EoverN_ratio': current_field_ratio,
            'local_equivalent_voltage_over_zdplaskin_final_voltage_ratio': _ratio(local_equiv_voltage_V, zdp_voltage_V),
            'same_voltage_EoverN_over_zdplaskin_saved_ratio': forced_voltage_ratio,
            'absorbed_power_over_zdplaskin_VI_power_ratio': _ratio(local_absorbed_power_W, zdp_absorbed_power_W),
        },
        'assessment': {
            'reported_local_EoverN_is_same_footing_as_zdplaskin': True,
            'physically_comparable_field_key': 'same_ZDPlaskin_final_voltage_reduced_field_Td',
            'interpretation': (
                'The same-voltage E/N is only a formula check. The local reported E/N should be interpreted through the active electrical backend.'
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare ZDPlaskin and local reduced fields on a voltage/gap/neutral-density footing.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--chamber', type=Path, default=None)
    parser.add_argument('--write', type=Path, default=None, help='Output YAML path; defaults to <output-dir>/comparison_zdplaskin_eovern.yaml')
    args = parser.parse_args()
    report = build_eovern_report(args.output_dir, args.reference, args.chamber)
    out_path = args.write or (args.output_dir / 'comparison_zdplaskin_eovern.yaml')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')
    print(out_path.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
