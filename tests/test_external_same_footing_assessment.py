from __future__ import annotations

from tools.external_benchmarks.same_footing_assessment import build_same_footing_report


def test_same_footing_report_separates_zdplaskin_power_field_and_crane_ode() -> None:
    report = build_same_footing_report(run_cases=False)
    by_name = {item['benchmark']: item for item in report['benchmarks']}

    zdp = by_name['ZDPlaskin example2']
    assert zdp['simple_verdict']['current_local_run_matches_zdplaskin'] is True
    assert zdp['simple_verdict']['current_run_EoverN_matches_zdplaskin'] is True
    assert zdp['simple_verdict']['forced_zdplaskin_voltage_formula_check_matches'] is True
    assert zdp['simple_verdict']['composition_matches_zdplaskin'] is True
    assert 'uses an internal circuit backend' in zdp['simple_verdict']['what_matches']
    assert zdp['overall_assessment']['same_footing_for_power'] is True
    assert zdp['overall_assessment']['same_footing_for_physical_EoverN_formula'] is True
    assert zdp['overall_assessment']['current_local_run_is_EoverN_parity_case'] is True
    assert zdp['overall_assessment']['current_local_run_matches_zdplaskin'] is True
    assert zdp['same_footing_axes']['reduced_field_from_voltage_gap_density']['status'] == 'matched'
    assert zdp['same_footing_axes']['species_densities_and_composition']['status'] == 'same_footing_pass'

    crane = by_name['CRANE TwoReactionArgon']
    assert crane['overall_assessment']['same_footing_for_scalar_ode_chemistry'] is True
    assert crane['overall_assessment']['same_footing_for_full_plasma_discharge'] is False
    assert crane['same_footing_axes']['reduced_field']['status'] == 'not_a_dynamic_field_comparison'
