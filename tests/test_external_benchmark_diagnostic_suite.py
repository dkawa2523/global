from __future__ import annotations

from pathlib import Path

from tools.external_benchmarks.diagnostic_suite import (
    DEFAULT_MANIFEST,
    benchmark_by_id,
    evaluate_swarm_1,
    load_benchmark_manifest,
    metric_row,
    write_findings,
)
from tools.external_benchmarks.benchmark_remediation import (
    public_zdp_rows,
    remediation_catalog,
)


def test_diagnostic_manifest_has_unique_problem_and_metric_ids() -> None:
    manifest = load_benchmark_manifest(DEFAULT_MANIFEST)
    benchmark_ids = [item['id'] for item in manifest['benchmarks']]

    assert len(benchmark_ids) == len(set(benchmark_ids))
    assert {'CRANE-1', 'ZDPlaskin-1', 'ZDPlaskin-2', 'PyGMol-1', 'PyGMol-Precision-1', 'SWARM-1', 'Runtime-1'} <= set(benchmark_ids)
    assert 'pygmol_same_footing' in manifest['diagnostic_categories']
    assert 'pygmol_precision_harness' in manifest['diagnostic_categories']


def test_swarm_ingestion_diagnostic_metrics_pass(tmp_path: Path) -> None:
    manifest = load_benchmark_manifest(DEFAULT_MANIFEST)
    _report, rows = evaluate_swarm_1(manifest, tmp_path)

    assert {row['metric_id'] for row in rows} == {
        'table_reader_prepares',
        'lookup_mode_is_field',
        'transport_values_finite',
        'rate_values_finite',
        'lookup_diagnostics_present',
        'clipping_diagnostics_present',
    }
    assert all(row['status'] == 'pass' for row in rows)
    assert all(row['diagnostic_category'] == 'swarm_table_ingestion' for row in rows)


def test_failed_metric_row_includes_suspected_modules_and_recommendation() -> None:
    manifest = load_benchmark_manifest(DEFAULT_MANIFEST)
    benchmark = benchmark_by_id(manifest, 'ZDPlaskin-1')
    metric = next(item for item in benchmark['metrics'] if item['id'] == 'peak_e_relative_error')

    row = metric_row(manifest, benchmark, metric, 0.25)

    assert row['status'] == 'fail'
    assert row['severity'] == 'medium'
    assert row['diagnostic_category'] == 'zdplaskin_species_transient'
    assert 'plasma_global/eedf/table.py' in row['suspected_modules']
    assert 'peak timing' in row['recommendation']


def test_findings_markdown_lists_actionable_failure(tmp_path: Path) -> None:
    manifest = load_benchmark_manifest(DEFAULT_MANIFEST)
    benchmark = benchmark_by_id(manifest, 'ZDPlaskin-1')
    metric = next(item for item in benchmark['metrics'] if item['id'] == 'peak_e_relative_error')
    row = metric_row(manifest, benchmark, metric, 0.25, note='early transient peak is not a core acceptance target')
    out = tmp_path / 'benchmark_findings.md'

    write_findings(out, [row])
    text = out.read_text(encoding='utf-8')

    assert 'ZDPlaskin-1 / peak_e_relative_error' in text
    assert 'fail' in text
    assert 'early transient peak is not a core acceptance target' in text


def test_pygmol_precision_gap_is_scoped_not_failed() -> None:
    manifest = load_benchmark_manifest(DEFAULT_MANIFEST)
    benchmark = benchmark_by_id(manifest, 'PyGMol-1')
    metric = next(item for item in benchmark['metrics'] if item['id'] == 'precision_claim_scoped_out')

    row = metric_row(manifest, benchmark, metric, 1.0, note='precision parity is intentionally not claimed')

    assert row['status'] == 'pass'
    assert row['severity'] == 'none'
    assert row['group'] == 'claim_scope'


def test_remediation_variant_specs_cover_expected_zdplaskin_axes() -> None:
    axes = {item['axis'] for item in remediation_catalog()['variant_specs']}

    assert {'baseline', 'energy_relaxation_time_s', 'wall_loss_frequency_s', 'electron_mobility_m2_V_s'} <= axes


def test_remediation_public_rows_hide_transient_scratch_paths() -> None:
    rows = public_zdp_rows([
        {
            'variant_id': 'baseline',
            'case_path': 'scratch/case.yaml',
            'output_dir': 'scratch/run',
            'peak_electron_density_relative_error': 0.1,
        }
    ])

    assert rows == [{'variant_id': 'baseline', 'peak_electron_density_relative_error': 0.1}]


def test_pygmol_gap_analysis_names_precision_blockers() -> None:
    gap_ids = {row['gap_id'] for row in remediation_catalog()['pygmol_gaps']}

    assert {'geometry', 'chemistry', 'power_definition', 'wall_loss'} <= gap_ids


def test_model_implementation_candidates_prioritize_core_model_value() -> None:
    candidates = remediation_catalog()['model_candidates']
    candidate_ids = {row['candidate_id'] for row in candidates}

    assert {
        'reaction_source_loss_diagnostics',
        'rate_table_boundary_provenance_diagnostics',
        'electrical_waveform_observables',
    } <= candidate_ids
    assert all(row['model_value'] for row in candidates)
    assert all('not benchmark tuning' not in row['not_benchmark_tuning'].lower() for row in candidates)


def test_benchmark_fit_rejections_exclude_case_specific_tuning_from_core() -> None:
    rejected = remediation_catalog()['benchmark_fit_rejections']
    rejected_ids = {row['rejected_item'] for row in rejected}

    assert 'benchmark_tuned_wall_loss_frequency' in rejected_ids
    assert 'pygmol_specific_core_mode' in rejected_ids
    assert all(row['acceptable_alternative'] for row in rejected)


def test_code_simplification_findings_record_cleanup_scope() -> None:
    rows = remediation_catalog()['code_simplification']
    fixed = {row['area'] for row in rows if row['status'] == 'fixed'}

    assert {'core_diagnostics', 'benchmark_tools', 'tracked_artifacts'} <= fixed
    assert all(row['next_step'] for row in rows)
