from __future__ import annotations

from pathlib import Path

import yaml

from tools.external_benchmarks.loki_o2_dc_glow_digitization import build_report


ROOT = Path(__file__).resolve().parents[1]


def _write_temp_spec(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_candidate.yaml').read_text(encoding='utf-8'))
    spec_dir = tmp_path / 'examples' / 'external'
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / 'loki_o2_dc_glow_candidate.yaml'
    spec_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
    return spec_path


def test_loki_digitization_scaffold_creates_expected_files() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        spec_path = _write_temp_spec(tmp_path)
        report = build_report(spec_path, workspace_root=tmp_path, report_path=tmp_path / 'report.yaml')

        digitization_root = tmp_path / 'examples' / 'external' / 'loki_o2_dc_glow_digitized'
        assert report['passed'] is True
        assert digitization_root.exists()
        assert (digitization_root / 'schema.yaml').exists()
        assert (digitization_root / 'README.md').exists()
        assert (digitization_root / 'pressure_sweep' / 'fig05a_avg_temperatures_vs_pressure.csv').exists()
        assert (digitization_root / 'acceptance_envelope_reference' / 'fig11_relative_difference_envelope_vs_pressure.csv').exists()
        assert (digitization_root / 'profile_assumption_diagnostics' / 'fig01_gas_temperature_profiles.csv').exists()


def test_loki_digitization_report_marks_empty_templates_as_scaffold_only() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        spec_path = _write_temp_spec(tmp_path)
        report = build_report(spec_path, workspace_root=tmp_path, report_path=tmp_path / 'report.yaml')

        assert report['summary']['files_expected'] == 14
        assert report['summary']['files_present'] == 14
        assert report['summary']['files_with_correct_header'] == 14
        assert report['summary']['files_with_data_rows'] == 0
        assert report['summary']['runnable_primary_pressure_benchmark'] is False
        assert report['summary']['readiness_stage'] == 'scaffold_only'
        assert report['evaluation']['physics_parity_status'] == 'blocked_waiting_for_digitized_data'
        assert any('Populate the primary pressure-sweep datasets' in item for item in report['evaluation']['next_actions'])


def test_loki_digitization_schema_uses_fig11_observed_pressure_subset() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        spec_path = _write_temp_spec(tmp_path)
        report = build_report(spec_path, workspace_root=tmp_path, report_path=tmp_path / 'report.yaml')

        fig11 = next(
            item
            for item in report['datasets']
            if item['file'].endswith('fig11_relative_difference_envelope_vs_pressure.csv')
        )
        assert fig11['expected_row_count'] == 36
