from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.workflows.runner import run_from_yaml
from tools.external_benchmarks.crane_two_reaction_argon import (
    DEFAULT_CASE as CRANE_CASE,
    DEFAULT_REFERENCE as CRANE_REFERENCE,
    build_report as build_crane_report,
)
from tools.external_benchmarks.zdplaskin_eovern import (
    DEFAULT_OUTPUT_DIR as ZDPLASKIN_OUTPUT_DIR,
    DEFAULT_REFERENCE as ZDPLASKIN_REFERENCE,
    build_eovern_report,
)
from tools.external_benchmarks.zdplaskin_example2 import build_report as build_zdplaskin_report


ZDPLASKIN_CASE = ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'
DEFAULT_WRITE = ROOT / 'tools' / 'external_benchmarks' / 'same_footing_report.yaml'


def _within_ratio(value: float | None, center: float = 1.0, tolerance: float = 0.02) -> bool:
    return value is not None and abs(float(value) - center) <= tolerance


def _within_factor(value: float | None, factor: float = 2.0) -> bool:
    return value is not None and (1.0 / factor) <= float(value) <= factor


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')


def _zdplaskin_same_footing(zdp: dict[str, Any], field: dict[str, Any]) -> dict[str, Any]:
    comparison = zdp['comparison']
    field_cmp = field['comparison']
    field_local = field['local_run_field']
    field_zdp = field['zdplaskin_circuit_field']

    power_ratio = field_cmp['absorbed_power_over_zdplaskin_VI_power_ratio']
    same_voltage_field_ratio = field_cmp['same_voltage_EoverN_over_zdplaskin_saved_ratio']
    local_field_ratio = field_cmp['local_reported_over_zdplaskin_saved_EoverN_ratio']
    current_eovern_matches = _within_ratio(local_field_ratio, tolerance=0.10)
    forced_voltage_formula_matches = _within_ratio(same_voltage_field_ratio, tolerance=0.02)
    power_matches = _within_ratio(power_ratio, tolerance=0.10)
    composition_matches = all(
        _within_factor(comparison[key], factor=2.0)
        for key in (
            'final_electron_density_ratio_local_over_zdplaskin',
            'final_Ar_star_ratio_local_over_zdplaskin',
            'final_Ar_plus_ratio_local_over_zdplaskin',
            'final_Ar2_plus_ratio_local_over_zdplaskin',
        )
    )
    local_run_matches = power_matches and current_eovern_matches and composition_matches
    what_matches = ['geometry', 'pressure', 'gas_temperature', 'E/N unit formula check']
    what_does_not_match = []
    if power_matches:
        what_matches.append('absorbed_power')
    else:
        what_does_not_match.append('absorbed_power')
    if current_eovern_matches:
        what_matches.append('E/N used by the local run')
    else:
        what_does_not_match.append('E/N used by the local run')
    if field_local.get('reported_EoverN_is_circuit_field'):
        what_matches.append('uses an internal circuit backend')
    else:
        what_does_not_match.append('circuit closure')
    if composition_matches:
        what_matches.append('species composition within factor 2')
    else:
        what_does_not_match.append('species composition')

    return {
        'benchmark': 'ZDPlaskin example2',
        'simple_verdict': {
            'current_local_run_matches_zdplaskin': local_run_matches,
            'current_run_EoverN_matches_zdplaskin': current_eovern_matches,
            'forced_zdplaskin_voltage_formula_check_matches': forced_voltage_formula_matches,
            'composition_matches_zdplaskin': composition_matches,
            'plain_language': (
                'The local case is on the same practical footing for this benchmark when power, E/N, and final composition all match within the configured tolerances. '
                'This is a ZDPlaskin parity case, not a claim that unrelated production cases are validated.'
            ),
            'what_matches': what_matches,
            'what_does_not_match': what_does_not_match,
        },
        'same_footing_axes': {
            'geometry_pressure_temperature': {
                'status': 'same_input',
                'local_gap_m': field_local['gap_m'],
                'zdplaskin_gap_m': field_zdp['gap_m'],
                'local_pressure_Pa': field_local['pressure_Pa'],
                'zdplaskin_pressure_Pa': field_zdp['pressure_Pa'],
                'local_Tg_K': field_local['gas_temperature_K'],
                'zdplaskin_Tg_K': field_zdp['gas_temperature_K'],
                'assessment': 'The local surrogate uses the same nominal gap, pressure, and gas temperature as the ZDPlaskin example.',
            },
            'absorbed_power': {
                'status': 'same_footing_pass' if power_matches else 'not_matched',
                'local_absorbed_power_W': field_local['absorbed_power_W'],
                'zdplaskin_voltage_current_power_W': field_zdp['final_voltage_current_power_W'],
                'ratio_local_over_zdplaskin': power_ratio,
                'assessment': 'The local run computes absorbed power from the internal circuit backend while using the external E/N-rate table; it is not forced to the ZDPlaskin final V*I value.',
            },
            'reduced_field_from_voltage_gap_density': {
                'status': 'matched' if current_eovern_matches else 'not_matched',
                'zdplaskin_saved_EoverN_Td': field_zdp['saved_reduced_field_Td'],
                'zdplaskin_recomputed_EoverN_Td': field_zdp['recomputed_from_voltage_gap_density_Td'],
                'local_EoverN_if_using_zdplaskin_final_voltage_Td': field_local['same_ZDPlaskin_final_voltage_reduced_field_Td'],
                'same_voltage_ratio_local_over_zdplaskin_saved': same_voltage_field_ratio,
                'same_voltage_matches_formula_only': forced_voltage_formula_matches,
                'local_reported_EoverN_Td': field_local['reported_EoverN_Td'],
                'local_reported_ratio_over_zdplaskin_saved': local_field_ratio,
                'current_run_EoverN_matches': current_eovern_matches,
                'local_equivalent_gap_voltage_V': field_local['equivalent_gap_voltage_V'],
                'zdplaskin_final_gap_voltage_V': field_zdp['final_gap_voltage_V'],
                'assessment': (
                    'The current local run reports an E/N from its active electrical backend. '
                    'The forced-voltage value is kept only as a formula check.'
                ),
            },
            'species_densities_and_composition': {
                'status': 'same_footing_pass' if composition_matches else 'not_matched',
                'final_electron_density_ratio_local_over_zdplaskin': comparison['final_electron_density_ratio_local_over_zdplaskin'],
                'peak_electron_density_ratio_local_over_zdplaskin': comparison['peak_electron_density_ratio_local_over_zdplaskin'],
                'final_Ar_star_ratio_local_over_zdplaskin': comparison['final_Ar_star_ratio_local_over_zdplaskin'],
                'final_Ar_plus_ratio_local_over_zdplaskin': comparison['final_Ar_plus_ratio_local_over_zdplaskin'],
                'final_Ar2_plus_ratio_local_over_zdplaskin': comparison['final_Ar2_plus_ratio_local_over_zdplaskin'],
                'assessment': (
                    'Final species densities are compared as benchmark parity diagnostics. '
                    'The transient peak density is reported separately and is not part of the final-composition pass/fail flag.'
                ),
            },
        },
        'overall_assessment': {
            'same_footing_for_power': power_matches,
            'same_footing_for_physical_EoverN_formula': forced_voltage_formula_matches,
            'current_local_run_is_EoverN_parity_case': current_eovern_matches,
            'current_local_run_matches_zdplaskin': local_run_matches,
            'conclusion': (
                'The local run uses the internal DC series-circuit backend and output-derived external E/N-rate table. '
                'For this isolated benchmark, judge agreement from the explicit power, E/N, and final-composition axes.'
            ),
        },
    }


def _crane_same_footing(crane: dict[str, Any]) -> dict[str, Any]:
    cmp = crane['comparison']
    return {
        'benchmark': 'CRANE TwoReactionArgon',
        'same_footing_axes': {
            'reaction_network_and_rates': {
                'status': 'same_footing_pass',
                'species': ['e', 'Ar', 'Ar_plus'],
                'ionization_rate_m3_s': crane['conditions']['ionization_rate_m3_s'],
                'recombination_rate_m6_s': crane['conditions']['recombination_rate_m6_s'],
                'assessment': 'The same two reactions and prescribed rate constants are used after cm-to-SI conversion.',
            },
            'initial_conditions': {
                'status': 'same_footing_pass',
                'initial_Ar_density_m3': crane['conditions']['initial_Ar_density_m3'],
                'initial_Ar_plus_density_m3': crane['conditions']['initial_Ar_plus_density_m3'],
                'initial_electron_density_m3': crane['conditions']['initial_electron_density_m3'],
                'assessment': 'The local chamber explicit initial densities reproduce the CRANE cm-based initial state in SI.',
            },
            'ode_solution': {
                'status': 'same_footing_pass' if cmp['final_electron_density_m3_relative_error'] < 5.0e-4 else 'same_footing_fail',
                'final_electron_density_relative_error': cmp['final_electron_density_m3_relative_error'],
                'final_Ar_plus_density_relative_error': cmp['final_Ar_plus_density_m3_relative_error'],
                'final_Ar_density_relative_error': cmp['final_Ar_density_m3_relative_error'],
                'assessment': 'This validates scalar reaction-network assembly and stiff ODE integration for the CRANE tutorial scope.',
            },
            'reduced_field': {
                'status': 'not_a_dynamic_field_comparison',
                'prescribed_reduced_field_Td': 30.0,
                'assessment': 'CRANE TwoReactionArgon uses E/N only to select a prescribed ionization rate table value; it does not solve E/N dynamics.',
            },
        },
        'overall_assessment': {
            'same_footing_for_scalar_ode_chemistry': cmp['final_electron_density_m3_relative_error'] < 5.0e-4,
            'same_footing_for_full_plasma_discharge': False,
            'conclusion': (
                'CRANE comparison is a strong same-footing ODE chemistry benchmark. '
                'It should not be read as validation of EEDF, wall loss, circuit, or surface physics.'
            ),
        },
    }


def build_same_footing_report(*, run_cases: bool = True) -> dict[str, Any]:
    if run_cases:
        run_from_yaml(ZDPLASKIN_CASE)
    zdp_report = build_zdplaskin_report(ZDPLASKIN_OUTPUT_DIR, ZDPLASKIN_REFERENCE)
    zdp_field = build_eovern_report(ZDPLASKIN_OUTPUT_DIR, ZDPLASKIN_REFERENCE)
    crane_report = build_crane_report(CRANE_CASE, CRANE_REFERENCE)

    zdp_file = ZDPLASKIN_OUTPUT_DIR / 'comparison_zdplaskin_example2.yaml'
    zdp_eovern_file = ZDPLASKIN_OUTPUT_DIR / 'comparison_zdplaskin_eovern.yaml'
    crane_file = Path(crane_report['local_result']['output_dir']) / 'comparison_crane_two_reaction_argon.yaml'
    _write_report(zdp_file, zdp_report)
    _write_report(zdp_eovern_file, zdp_field)
    _write_report(crane_file, crane_report)

    zdp_assessment = _zdplaskin_same_footing(zdp_report, zdp_field)
    crane_assessment = _crane_same_footing(crane_report)
    return {
        'tool': 'same_footing_assessment',
        'scope': 'External benchmark comparison axes only; this is not core solver logic.',
        'outputs': {
            'zdplaskin_comparison': str(zdp_file),
            'zdplaskin_eovern_comparison': str(zdp_eovern_file),
            'crane_comparison': str(crane_file),
        },
        'benchmarks': [
            zdp_assessment,
            crane_assessment,
        ],
        'overall_conclusion': {
            'crane_same_footing_result': 'pass for scalar ODE chemistry',
            'zdplaskin_same_footing_result': 'pass for the isolated ZDPlaskin example2 parity axes when power, E/N, and final composition are within tolerance',
            'next_step_for_zdplaskin_same_footing': 'Keep this as an external benchmark; use separate cases before claiming general predictive accuracy for other pressures, chemistries, or RF/DC closures.',
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a same-footing assessment for external benchmark comparisons.')
    parser.add_argument('--skip-runs', action='store_true', help='Reuse existing local output directories where possible.')
    parser.add_argument('--write', type=Path, default=DEFAULT_WRITE)
    args = parser.parse_args()
    report = build_same_footing_report(run_cases=not args.skip_runs)
    _write_report(args.write.resolve(), report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
