from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.external_benchmarks.robustness_sweep import assess_summary

ROOT = Path(__file__).resolve().parents[1]


def test_robustness_summary_assessment_accepts_positive_finite_run() -> None:
    report = assess_summary(
        {
            'success': True,
            'final_electron_density_m3': 1.0e16,
            'final_mean_electron_energy_eV': 3.0,
            'final_port_source_rf_absorbed_power_W': 10.0,
        }
    )

    assert report['passed'] is True
    assert report['checks']['positive_final_electron_density']['passed'] is True


def test_robustness_summary_assessment_rejects_nonphysical_density() -> None:
    report = assess_summary(
        {
            'success': True,
            'final_electron_density_m3': 0.0,
            'final_mean_electron_energy_eV': 3.0,
        }
    )

    assert report['passed'] is False
    assert report['checks']['positive_final_electron_density']['passed'] is False


def test_wide_zdplaskin_table_covers_stress_fields() -> None:
    h5py = pytest.importorskip('h5py')
    table = ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'tables' / 'zdplaskin_example2_wide_eovern_rates.h5'

    with h5py.File(table, 'r') as h5:
        field = h5['effective_field_Td'][:]

    assert field[0] <= 0.5
    assert field[-1] >= 250.0


def test_benchmark_data_manifest_separates_parity_and_stress_tables() -> None:
    manifest_path = ROOT / 'tools' / 'external_benchmarks' / 'benchmark_data_manifest.yaml'
    manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
    by_id = {item['id']: item for item in manifest['benchmark_data']}

    assert by_id['zdplaskin_example2_output_derived_eovern_table']['role'] == 'strict_parity'
    assert by_id['zdplaskin_example2_wide_diagnostic_eovern_table']['role'] == 'pressure_voltage_stress'
