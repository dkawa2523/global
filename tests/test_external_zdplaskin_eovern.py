from __future__ import annotations

from pathlib import Path

import pytest

from tools.external_benchmarks.zdplaskin_eovern import build_eovern_report, reduced_field_Td


ROOT = Path(__file__).resolve().parents[1]


def test_reduced_field_from_zdplaskin_voltage_is_close_to_saved_field() -> None:
    report = build_eovern_report(
        ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate',
        ROOT / 'examples' / 'external' / 'zdplaskin_example2_summary.yaml',
        ROOT / 'examples' / 'configs' / 'chamber_zdplaskin_example2.yaml',
    )

    zdp = report['zdplaskin_circuit_field']
    comparison = report['comparison']
    local = report['local_run_field']

    assert zdp['recomputed_from_voltage_gap_density_Td'] == pytest.approx(2.56, rel=2.0e-2)
    assert abs(zdp['saved_vs_recomputed_relative_difference']) < 2.0e-2
    assert comparison['local_reported_over_zdplaskin_saved_EoverN_ratio'] == pytest.approx(1.0, rel=2.0e-2)
    assert comparison['local_equivalent_voltage_over_zdplaskin_final_voltage_ratio'] == pytest.approx(1.0, rel=2.0e-2)
    assert report['simple_verdict']['current_local_run_EoverN_matches_zdplaskin'] is True
    assert report['simple_verdict']['forced_zdplaskin_voltage_formula_check_matches'] is True
    assert report['simple_verdict']['local_reported_EoverN_is_circuit_field'] is True
    assert local['same_ZDPlaskin_final_voltage_reduced_field_Td'] == pytest.approx(
        zdp['recomputed_from_voltage_gap_density_Td'],
        rel=2.0e-3,
    )
    assert report['assessment']['reported_local_EoverN_is_same_footing_as_zdplaskin'] is True


def test_reduced_field_formula_uses_voltage_gap_and_neutral_density() -> None:
    value = reduced_field_Td(voltage_V=33.2225, gap_m=0.004, pressure_Pa=13332.0, gas_temperature_K=300.0)
    assert value == pytest.approx(2.58, rel=1.0e-2)
