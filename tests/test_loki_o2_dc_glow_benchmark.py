from __future__ import annotations

from pathlib import Path

from tools.external_benchmarks.loki_o2_dc_glow_benchmark import build_benchmark_report


ROOT = Path(__file__).resolve().parents[1]


def test_loki_benchmark_runs_on_committed_digitized_full_reference(tmp_path: Path) -> None:
    report = build_benchmark_report(
        ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_candidate.yaml',
        report_path=tmp_path / 'loki_benchmark_report.yaml',
        summary_csv_path=tmp_path / 'loki_benchmark_summary.csv',
    )

    assert report['passed'] is True
    assert report['summary']['runnable_primary_pressure_benchmark'] is True
    assert report['summary']['runnable_extended_species_benchmark'] is True
    assert report['summary']['runnable_full_diagnostic_benchmark'] is True
    assert report['summary']['readiness_stage'] == 'runnable_full_diagnostic_benchmark'
    assert report['summary']['direct_main_parameter_threshold_pass'] is True
    assert report['summary']['extended_neutral_subset_reference_ceiling_pass'] is True
    assert report['summary']['profile_shape_expectation_pass'] is True
    assert 'O3' in report['extended_species_relative_differences']
    assert report['profile_diagnostics']['shape_expectation_pass'] is True
    assert Path(report['summary_csv']).exists()
