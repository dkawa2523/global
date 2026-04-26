from __future__ import annotations

from pathlib import Path

from tools.external_benchmarks.robustness_dashboard import build_robustness_dashboard


def test_robustness_dashboard_writes_summary_graphs_and_report(tmp_path: Path) -> None:
    report = build_robustness_dashboard(
        run_cases=False,
        report_path=tmp_path / 'robustness_dashboard.yaml',
        overview_png_path=tmp_path / 'robustness_stress_overview.png',
        coverage_png_path=tmp_path / 'robustness_rate_table_coverage.png',
        markdown_report_path=tmp_path / 'robustness_applicability_report.md',
    )

    assert report['headline']['all_cases_passed'] is True
    assert report['headline']['external_reference_count'] >= 2
    assert report['headline']['stress_case_count'] >= 8
    assert Path(report['outputs']['overview_png']).exists()
    assert Path(report['outputs']['rate_table_coverage_png']).exists()
    assert Path(report['outputs']['markdown_report_md']).exists()
