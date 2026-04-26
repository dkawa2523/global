from __future__ import annotations

from pathlib import Path

from tools.external_benchmarks.benchmark_dashboard import build_dashboard


ROOT = Path(__file__).resolve().parents[1]


def test_external_benchmark_dashboard_writes_summary_and_graphs(tmp_path: Path) -> None:
    report = build_dashboard(
        rerun_zdplaskin=False,
        rerun_pygmol_local=False,
        report_path=tmp_path / 'external_benchmark_dashboard.yaml',
        metrics_csv_path=tmp_path / 'external_benchmark_dashboard_metrics.csv',
        overview_png_path=tmp_path / 'external_benchmark_overview.png',
        pygmol_png_path=tmp_path / 'external_benchmark_pygmol_decomposition.png',
        coverage_png_path=tmp_path / 'external_benchmark_coverage_matrix.png',
        markdown_report_path=tmp_path / 'external_benchmark_report.md',
        markdown_report_ja_path=tmp_path / 'external_benchmark_report_ja.md',
    )

    assert report['benchmarks']['zdplaskin_example2']['passed'] is True
    assert report['benchmarks']['crane_two_reaction_argon']['passed'] is True
    assert report['benchmarks']['pygmol_same_footing']['passed'] is True
    assert report['benchmarks']['loki_o2_dc_glow']['passed'] is True
    assert report['applicability_evidence_summary'][0]['item'] == 'Strict external references retained'
    assert Path(report['outputs']['report_yaml']).exists()
    assert Path(report['outputs']['metrics_csv']).exists()
    assert Path(report['outputs']['overview_png']).exists()
    assert Path(report['outputs']['pygmol_decomposition_png']).exists()
    assert Path(report['outputs']['coverage_png']).exists()
    assert Path(report['outputs']['paper_report_md']).exists()
    assert Path(report['outputs']['paper_report_ja_md']).exists()
    assert Path(report['outputs']['robustness_dashboard_yaml']).exists()
    assert Path(report['outputs']['robustness_overview_png']).exists()
    assert Path(report['outputs']['robustness_report_md']).exists()

    report_ja = Path(report['outputs']['paper_report_ja_md']).read_text(encoding='utf-8')
    assert '## 2. コード構成と処理の流れ' in report_ja
    assert '### 2.3 設計カテゴリごとの利点' in report_ja
    assert '### 2.4 将来拡張時の入口と利点' in report_ja
    assert '## 3. 支配方程式と closure' in report_ja
    assert '## 4. ベンチマーク体系と理論差の整理' in report_ja
    assert '## 6. ベンチマークごとの詳細解釈' in report_ja
