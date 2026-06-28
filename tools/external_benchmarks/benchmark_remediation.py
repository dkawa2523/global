from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.workflows.runner import run_from_yaml
from tools.external_benchmarks.zdplaskin_example2 import (
    DEFAULT_REFERENCE as ZDP_REFERENCE,
    build_report as build_zdp_report,
)


DEFAULT_OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks' / 'remediation'
BASE_CASE = ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'
BASE_CHAMBER = ROOT / 'examples' / 'configs' / 'chamber_zdplaskin_example2.yaml'
BASE_RECIPE = ROOT / 'examples' / 'configs' / 'recipe_zdplaskin_example2.yaml'
BASE_CHEMISTRY = ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'chemistry_manifest.yaml'
BASE_INCLUDE = ROOT / 'examples' / 'configs' / 'base_case.yaml'


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ratio_error(ratio: float | None) -> float | None:
    return None if ratio is None else abs(float(ratio) - 1.0)


def _variant_case(
    *,
    variant_id: str,
    output_dir: Path,
    energy_relaxation_time_s: float = 1.0e-6,
    wall_loss_frequency_s: float = 3230.0,
    electron_mobility_m2_V_s: float = 0.45,
) -> Path:
    case_dir = output_dir / 'cases'
    chamber = _load_yaml(BASE_CHAMBER)
    chamber['surfaces'][0]['models']['frequency_s'] = float(wall_loss_frequency_s)
    chamber['power_ports'][0]['parameters']['electron_mobility_m2_V_s'] = float(electron_mobility_m2_V_s)
    chamber_path = case_dir / f'{variant_id}_chamber.yaml'
    _write_yaml(chamber_path, chamber)

    case = {
        'include': str(BASE_INCLUDE),
        'case': {
            'name': f'zdplaskin_example2_{variant_id}',
            'description': f'Remediation sensitivity variant {variant_id} for ZDPlaskin peak-electron-density diagnostics.',
            'tags': ['argon', 'zdplaskin', 'benchmark', 'remediation'],
        },
        'files': {
            'chamber': str(chamber_path),
            'recipe': str(BASE_RECIPE),
            'chemistry': {'manifest': str(BASE_CHEMISTRY)},
            'output_dir': str(output_dir / 'runs' / variant_id),
        },
        'physics': {
            'eedf_backend': 'rate_table',
            'electrical_backend': 'dc_series_circuit',
            'enable_gas_temperature': False,
            'enable_surface_coverages': False,
            'enable_wall_inventory': False,
        },
        'swarm': {
            'model_name': 'table',
            'closure': 'local_field',
            'table': {
                'file': 'tables/zdplaskin_example2_eovern_rates.h5',
                'lookup': 'local_field',
                'electron_energy_mode': 'table_relaxation',
                'energy_relaxation_time_s': float(energy_relaxation_time_s),
            },
        },
        'numerics': {
            'rtol': 3.0e-6,
            'atol': 1.0e-14,
            'first_step': 1.0e-12,
            'max_step': 1.0e-6,
        },
        'outputs': {'plots': {'enabled': False}},
    }
    case_path = case_dir / f'{variant_id}_case.yaml'
    _write_yaml(case_path, case)
    return case_path


def _variant_specs() -> list[dict[str, Any]]:
    baseline = {
        'variant_id': 'baseline',
        'axis': 'baseline',
        'energy_relaxation_time_s': 1.0e-6,
        'wall_loss_frequency_s': 3230.0,
        'electron_mobility_m2_V_s': 0.45,
    }
    specs = [baseline]
    for tau in [1.0e-8, 1.0e-7, 3.0e-7, 3.0e-6, 1.0e-5]:
        specs.append({
            **baseline,
            'variant_id': f'tau_{tau:.0e}'.replace('+', ''),
            'axis': 'energy_relaxation_time_s',
            'energy_relaxation_time_s': tau,
        })
    for multiplier in [0.5, 0.75, 1.25, 1.5]:
        specs.append({
            **baseline,
            'variant_id': f'wall_{multiplier:.2g}x'.replace('.', 'p'),
            'axis': 'wall_loss_frequency_s',
            'wall_loss_frequency_s': 3230.0 * multiplier,
        })
    for mobility in [0.30, 0.60]:
        specs.append({
            **baseline,
            'variant_id': f'mobility_{mobility:.2g}'.replace('.', 'p'),
            'axis': 'electron_mobility_m2_V_s',
            'electron_mobility_m2_V_s': mobility,
        })
    return specs


def run_zdplaskin_peak_sensitivity(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for spec in _variant_specs():
        case_path = _variant_case(output_dir=output_dir, **{k: v for k, v in spec.items() if k != 'axis'})
        run_from_yaml(case_path)
        variant_output = output_dir / 'runs' / spec['variant_id']
        report = build_zdp_report(variant_output, ZDP_REFERENCE)
        comparison = report['comparison']
        peak_error = _ratio_error(comparison['peak_electron_density_ratio_local_over_zdplaskin'])
        final_error = _ratio_error(comparison['final_electron_density_ratio_local_over_zdplaskin'])
        peak_time_ratio = report['local_result']['peak_electron_density_time_s'] / report['zdplaskin_reference']['peak_saved_electron_density_time_s']
        rows.append({
            **spec,
            'case_path': str(case_path),
            'output_dir': str(variant_output),
            'local_peak_electron_density_m3': report['local_result']['peak_electron_density_m3'],
            'reference_peak_electron_density_m3': report['zdplaskin_reference']['peak_saved_electron_density_m3'],
            'peak_electron_density_ratio_local_over_zdplaskin': comparison['peak_electron_density_ratio_local_over_zdplaskin'],
            'peak_electron_density_relative_error': peak_error,
            'local_peak_time_s': report['local_result']['peak_electron_density_time_s'],
            'reference_peak_time_s': report['zdplaskin_reference']['peak_saved_electron_density_time_s'],
            'peak_time_ratio_local_over_zdplaskin': peak_time_ratio,
            'final_electron_density_relative_error': final_error,
            'final_EoverN_relative_error': _ratio_error(comparison['final_reduced_field_ratio_local_over_zdplaskin']),
        })
    return rows


def _pygmol_gap_rows() -> list[dict[str, Any]]:
    chamber = _load_yaml(ROOT / 'examples' / 'configs' / 'chamber_argon_icp.yaml')
    recipe = _load_yaml(ROOT / 'examples' / 'configs' / 'recipe_argon_lxcat.yaml')
    pygmol_model = _load_yaml(ROOT / 'tools' / 'external_benchmarks' / 'pygmol_argon_model.yaml')
    zone_count = len(chamber.get('zones', []))
    has_edges = bool(chamber.get('edges'))
    surface_count = len(chamber.get('surfaces', []))
    source_power_steps = [
        float(step['power_ports']['source_rf'].get('value_W', 0.0))
        for step in recipe.get('steps', [])
        if 'source_rf' in step.get('power_ports', {})
    ]
    rows = [
        {
            'gap_id': 'geometry',
            'current_state': f'{zone_count} zones, {surface_count} surfaces, edges={has_edges}',
            'needed_for_precision': 'single equivalent cylinder with one volume and one surface-loss area',
            'action': 'Add a generated same-footing local case or adapter output that collapses the local reactor to PyGMol geometry.',
            'owner_area': 'plasma_global/reactor/models.py; tools/external_benchmarks/pygmol_argon.py',
        },
        {
            'gap_id': 'chemistry',
            'current_state': f"local LXCat mechanism vs PyGMol model {pygmol_model['model']['id']}",
            'needed_for_precision': 'same species, same ionization/excitation/elastic rates, same energy losses',
            'action': 'Create a dedicated local chemistry bundle generated from pygmol_argon_model.yaml for PyGMol-precision tests.',
            'owner_area': 'plasma_global/chemistry/io.py; tools/external_benchmarks/pygmol_argon.py',
        },
        {
            'gap_id': 'power_definition',
            'current_state': f'local stepped absorbed power {source_power_steps} W; PyGMol consumes its own power array',
            'needed_for_precision': 'identical power deposition history and electron-energy definition',
            'action': 'Export the exact local step power array to both solvers and compare mean-energy-equivalent consistently.',
            'owner_area': 'plasma_global/electrical/direct_power.py; tools/external_benchmarks/pygmol_argon.py',
        },
        {
            'gap_id': 'wall_loss',
            'current_state': 'local Bohm wall loss over multiple surfaces; PyGMol total cylinder-wall loss',
            'needed_for_precision': 'same wall-loss coefficient or same wall-flux equation',
            'action': 'Add a PyGMol same-wall-loss mode that uses local k_wall(t), then use it as the precision benchmark gate.',
            'owner_area': 'plasma_global/physics/surface_core.py; plasma_global/physics/gas_phase_core.py',
        },
    ]
    return rows


def _model_implementation_candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            'candidate_id': 'reaction_source_loss_diagnostics',
            'priority': 'P1',
            'implementation_status': 'implemented_initial',
            'scope': 'core_optional_diagnostic',
            'external_basis': 'ZDPlaskin transient outputs and CRANE reaction-ODE parity both make reaction-level balances explicit.',
            'current_evidence': 'ZDPlaskin peak-density mismatch is not fixed by scalar sensitivity alone; the next discriminant is which Ar2+ and electron-energy terms create the early transient.',
            'model_value': 'Improves mechanism debugging, chemistry validation, conservation audits, and transient interpretation for low-pressure plasma cases.',
            'implementation_method': 'Add an opt-in per-zone reaction ledger for species source/loss and electron-energy loss terms, sampled into observables or benchmark diagnostics without changing default RHS behavior.',
            'first_validation_gate': 'ZDPlaskin-1 early-time peak window can identify dominant source/loss terms; CRANE-1 balances stay conserved.',
            'owner_area': 'plasma_global/physics/gas_phase_core.py; plasma_global/observables/defaults.py',
            'not_benchmark_tuning': 'Does not change reaction rates or fit a parameter; it exposes the physical budget already implied by the model.',
        },
        {
            'candidate_id': 'rate_table_boundary_provenance_diagnostics',
            'priority': 'P1',
            'implementation_status': 'implemented_initial',
            'scope': 'core_runtime_safety',
            'external_basis': 'BOLSIG+/LoKI-style swarm tables are defined by grids, units, transport coefficients, rates, and source provenance.',
            'current_evidence': 'SWARM-1 proves ingestion and finite interpolation, but boundary clipping, table coverage, and source metadata are not yet promoted to runtime diagnostics.',
            'model_value': 'Prevents silent use of out-of-domain electron kinetics, which is central to low-pressure global-model validity.',
            'implementation_method': 'Report lookup coordinate, clipping flags/counts, table min/max coverage, source tool/version, and units through diagnostics/provenance; keep interpolation math unchanged.',
            'first_validation_gate': 'SWARM-1 detects out-of-range queries and records table provenance while preserving existing in-range parity.',
            'owner_area': 'plasma_global/eedf/table.py; scripts/build_rate_table_h5.py; plasma_global/workflows/runner.py',
            'not_benchmark_tuning': 'Does not alter coefficients to match BOLSIG+ or LoKI; it makes invalid table use visible.',
        },
        {
            'candidate_id': 'electrical_waveform_observables',
            'priority': 'P1',
            'implementation_status': 'implemented_initial',
            'scope': 'core_observability',
            'external_basis': 'ZDPlaskin circuit comparisons center on voltage, current, E/N, conductance/loading, and absorbed power waveforms.',
            'current_evidence': 'Current comparisons can use final E/N and a limited waveform metric, but the model should expose the full circuit state for any pulsed/DC low-pressure case.',
            'model_value': 'Makes power coupling, field history, and plasma loading auditable without tying the model to one external solver.',
            'implementation_method': 'Standardize optional PowerResult waveform observables: source voltage, gap voltage, current, plasma conductance, E/N, absorbed power, and port identifier.',
            'first_validation_gate': 'ZDPlaskin-1 circuit waveform metrics can separate electrical mismatch from chemistry mismatch.',
            'owner_area': 'plasma_global/electrical/base.py; plasma_global/electrical/dc_series.py; plasma_global/observables/defaults.py',
            'not_benchmark_tuning': 'Adds physical observability of the electrical closure; it does not calibrate circuit parameters.',
        },
        {
            'candidate_id': 'transient_window_budget_sampler',
            'priority': 'P2',
            'implementation_status': 'implemented_initial',
            'scope': 'analysis_workflow',
            'external_basis': 'ZDPlaskin and CRANE diagnostics are most useful when evaluated around physically important transient windows, not only at final state.',
            'current_evidence': 'The ZDPlaskin electron-density peak time is far from the local peak time even when final values are closer.',
            'model_value': 'Supports low-pressure pulsed and afterglow validation where peak timing and energy relaxation are the physics of interest.',
            'implementation_method': 'Add a reusable sampler that records selected budgets around configured time windows: mean energy, E/N, V/I/P, per-reaction sources, and wall-loss terms.',
            'first_validation_gate': 'ZDPlaskin-1 peak-window report explains whether the early mismatch is electrical, table-relaxation, chemistry, or wall-loss dominated.',
            'owner_area': 'tools/external_benchmarks/diagnostic_suite.py; plasma_global/workflows/runner.py',
            'not_benchmark_tuning': 'Narrows cause of model discrepancy; it does not force the model toward a reference trace.',
        },
        {
            'candidate_id': 'swarm_interchange_metadata_contract',
            'priority': 'P2',
            'implementation_status': 'pending',
            'scope': 'core_data_contract',
            'external_basis': 'BOLSIG+ and LoKI outputs carry assumptions about gas composition, E/N or mean-energy grids, cross-section sources, units, and transport definitions.',
            'current_evidence': 'The current HDF5 table path is compact and useful, but the model cannot yet assert whether two swarm tables are semantically comparable.',
            'model_value': 'Improves reproducibility and prevents mixing incompatible electron-kinetics data in production low-pressure studies.',
            'implementation_method': 'Extend the table manifest with required grid variable, gas composition, transport definitions, source tool/version, cross-section provenance, and optional uncertainty fields.',
            'first_validation_gate': 'SWARM-1 rejects missing required metadata for precision claims while accepting legacy compact tables as compatibility inputs.',
            'owner_area': 'scripts/build_rate_table_h5.py; plasma_global/eedf/table.py; docs',
            'not_benchmark_tuning': 'Strengthens data semantics; it does not modify rate values.',
        },
        {
            'candidate_id': 'same_footing_external_fixture_generation',
            'priority': 'P2',
            'implementation_status': 'implemented',
            'scope': 'benchmark_infrastructure_not_core_physics',
            'external_basis': 'PyGMol precision comparison requires identical geometry, chemistry, power deposition, and wall-loss definitions.',
            'current_evidence': 'PyGMol-Precision-1 now passes as a tools-only same-footing compact-Ar harness; PyGMol-1 remains the production-case sanity comparison.',
            'model_value': 'Prevents overstating accuracy and creates a clean fixture for testing low-pressure global-model assumptions.',
            'implementation_method': 'Implemented tools/external_benchmarks/pygmol_precision.py to generate an identical compact-Ar ODE fixture from the PyGMol model file.',
            'first_validation_gate': 'PyGMol-Precision-1 passes max final relative error and max waveform NRMSE thresholds while keeping production-core claims scoped out.',
            'owner_area': 'tools/external_benchmarks/pygmol_precision.py; tools/external_benchmarks/pygmol_argon.py',
            'not_benchmark_tuning': 'Keeps comparison fair without adding PyGMol-specific behavior to the production solver.',
        },
    ]


def _benchmark_fit_rejection_rows() -> list[dict[str, Any]]:
    return [
        {
            'rejected_item': 'benchmark_tuned_wall_loss_frequency',
            'reason': 'The sensitivity run improved the ZDPlaskin peak error but did not satisfy the peak gate or fix peak timing.',
            'risk': 'Would hide model-form or transient-coupling errors behind a case-specific constant.',
            'acceptable_alternative': 'Expose wall-loss budgets and validate the selected closure before changing any default.',
        },
        {
            'rejected_item': 'benchmark_tuned_energy_relaxation_time',
            'reason': 'Changing one relaxation time cannot distinguish rate-table coverage, circuit loading, and chemistry source terms.',
            'risk': 'Would make one ZDPlaskin trace look closer while reducing physical meaning for other low-pressure regimes.',
            'acceptable_alternative': 'Add transient-window source/loss and energy-budget diagnostics, then justify any new closure physically.',
        },
        {
            'rejected_item': 'pygmol_specific_core_mode',
            'reason': 'PyGMol is useful for a same-footing global-model check, not as a production-mode target.',
            'risk': 'Would mix benchmark scaffolding with the solver API and obscure the model assumptions.',
            'acceptable_alternative': 'Generate same-footing benchmark fixtures under tools/external_benchmarks only.',
        },
        {
            'rejected_item': 'live_external_solver_calls_inside_rhs',
            'reason': 'External tools are references and data generators, not runtime dependencies for the compact model ODE.',
            'risk': 'Would reduce reproducibility, speed, and portability of the low-pressure model.',
            'acceptable_alternative': 'Import external outputs into explicit, versioned tables or reference artifacts.',
        },
    ]


def _code_simplification_rows() -> list[dict[str, Any]]:
    return [
        {
            'area': 'core_diagnostics',
            'issue': 'GasPhaseCore mixed RHS terms and diagnostic budget reconstruction.',
            'action': 'Added shared GasReactionTerm and IonWallLossTerm flows for RHS and budgets.',
            'status': 'fixed',
            'next_step': 'Keep diagnostic budgets derived from the same physical terms used by the RHS.',
        },
        {
            'area': 'benchmark_tools',
            'issue': 'Legacy external benchmark runner and one-off plot scripts duplicated diagnostic_suite outputs.',
            'action': 'Deleted the old runner/dashboard/time-series plotting entry points.',
            'status': 'fixed',
            'next_step': 'Use diagnostic_suite.py and benchmark_remediation.py as the only external-benchmark review surfaces.',
        },
        {
            'area': 'tracked_artifacts',
            'issue': 'Older dashboard/agreement/time-series artifacts competed with current problem-scoped figures.',
            'action': 'Removed stale tracked artifacts outside diagnostic_suite/ and remediation/.',
            'status': 'fixed',
            'next_step': 'Track only lightweight current artifacts and regenerate large run outputs.',
        },
        {
            'area': 'observables_width',
            'issue': 'Detailed budget columns are useful for validation but can make normal observables wide.',
            'action': 'Added outputs.diagnostics.budgets and kept detailed reaction/source/loss columns opt-in.',
            'status': 'fixed',
            'next_step': 'Keep normal cases compact and enable budgets only for validation or benchmark cases.',
        },
        {
            'area': 'electrical_contract',
            'issue': 'PowerResult.port_observables was a generic nested dict.',
            'action': 'Introduced ElectricalPortSnapshot with a stable observables serializer.',
            'status': 'fixed',
            'next_step': 'Keep backend-specific electrical details typed before flattening to CSV columns.',
        },
        {
            'area': 'pygmol_precision',
            'issue': 'PyGMol production-case precision and compact-equation parity needed separate claims.',
            'action': 'Added PyGMol-Precision-1 as a tools-only same-footing harness and kept PyGMol-specific work out of core physics.',
            'status': 'fixed',
            'next_step': 'Keep PyGMol-1 as sanity/scoping and PyGMol-Precision-1 as compact-equation parity only.',
        },
    ]


def remediation_catalog() -> dict[str, list[dict[str, Any]]]:
    return {
        'variant_specs': _variant_specs(),
        'pygmol_gaps': _pygmol_gap_rows(),
        'model_candidates': _model_implementation_candidate_rows(),
        'benchmark_fit_rejections': _benchmark_fit_rejection_rows(),
        'code_simplification': _code_simplification_rows(),
    }


def _write_code_simplification_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        '# Code Simplification Findings',
        '',
        'This file records simplification and deletion findings from the external-benchmark cleanup. It separates completed cleanup from future work so benchmark-specific concerns do not leak into the core model.',
        '',
        '| Area | Status | Issue | Action | Next step |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        lines.append(
            f"| {row['area']} | {row['status']} | {row['issue']} | {row['action']} | {row['next_step']} |"
        )
    lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def _write_model_candidates_markdown(
    path: Path,
    candidates: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    lines = [
        '# Model Implementation Candidates',
        '',
        'This report separates low-pressure-plasma model improvements from benchmark fitting. The benchmark evidence is used only to identify missing observability, data contracts, or validation structure that would be useful beyond one external code.',
        '',
        '## Implement',
        '',
    ]
    for row in candidates:
        lines.extend([
            f"### {row['candidate_id']} ({row['priority']})",
            '',
            f"- Status: {row['implementation_status']}",
            f"- Scope: {row['scope']}",
            f"- External basis: {row['external_basis']}",
            f"- Current evidence: {row['current_evidence']}",
            f"- Low-pressure model value: {row['model_value']}",
            f"- Concrete method: {row['implementation_method']}",
            f"- First validation gate: {row['first_validation_gate']}",
            f"- Owner area: {row['owner_area']}",
            f"- Why this is not benchmark tuning: {row['not_benchmark_tuning']}",
            '',
        ])
    lines.extend([
        '## Do Not Implement As Core Model Changes',
        '',
    ])
    for row in rejected:
        lines.extend([
            f"### {row['rejected_item']}",
            '',
            f"- Reason: {row['reason']}",
            f"- Risk: {row['risk']}",
            f"- Acceptable alternative: {row['acceptable_alternative']}",
            '',
        ])
    lines.extend([
        '## Recommended Work Order',
        '',
        '1. Add reaction source/loss diagnostics and electrical waveform observables so ZDPlaskin transient misses can be decomposed without fitting constants.',
        '2. Add rate-table boundary/provenance diagnostics so BOLSIG+/LoKI-derived data cannot be used outside its physical domain silently.',
        '3. Add transient-window budget sampling to separate peak-timing physics from final-state agreement.',
        '4. Keep the PyGMol same-footing fixture under benchmark tools, with PyGMol-specific adapters out of production solver code.',
        '5. Re-run CRANE-1, ZDPlaskin-1/2, PyGMol-1, SWARM-1, and Runtime-1 after each P1 implementation.',
        '',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def _plot_zdplaskin_sensitivity(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row['variant_id'] for row in rows]
    values = [float(row['peak_electron_density_relative_error']) for row in rows]
    colors = ['#59a14f' if value <= 0.05 else '#e15759' for value in values]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = range(len(rows))
    ax.bar(x, values, color=colors)
    ax.axhline(0.05, color='#333333', linestyle='--', linewidth=1.0, label='5% target')
    ax.set_xticks(list(x), labels, rotation=35, ha='right')
    ax.set_ylabel('Peak electron density relative error')
    ax.set_title('ZDPlaskin peak-density remediation sensitivity')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _write_remediation_markdown(path: Path, zdp_rows: list[dict[str, Any]], pygmol_rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in zdp_rows if row['variant_id'] == 'baseline')
    best = min(zdp_rows, key=lambda row: float(row['peak_electron_density_relative_error']))
    lines = [
        '# Benchmark Remediation Actions',
        '',
        'This file turns diagnostic-suite findings into concrete next actions and small experiments.',
        '',
        '## ZDPlaskin-1 Peak Electron Density',
        '',
        f"- Baseline peak relative error: `{float(baseline['peak_electron_density_relative_error']):.4g}`.",
        f"- Best tested variant: `{best['variant_id']}` on `{best['axis']}` with peak relative error `{float(best['peak_electron_density_relative_error']):.4g}`.",
    ]
    if float(best['peak_electron_density_relative_error']) <= 0.05:
        lines.extend([
            '- Concrete measure: promote this axis to a focused validation branch, then check final species, E/N, and power before changing defaults.',
            '- Implementation method: add an explicit benchmark-calibrated option only if the variant is physically justified by the ZDPlaskin equations.',
        ])
    else:
        lines.extend([
            '- Concrete measure: do not tune a scalar parameter blindly; the tested simple parameters did not bring peak density under the 5% gate.',
            '- Implementation method: add a dedicated transient decomposition benchmark that logs rate-table mean energy, E/N, circuit voltage/current, and individual Ar2+ source/loss terms around the external peak time.',
            '- Likely code areas: `plasma_global/eedf/table.py`, `plasma_global/electrical/dc_series.py`, and `plasma_global/physics/gas_phase_core.py`.',
        ])
    lines.extend([
        '',
        'Recommended ZDPlaskin work order:',
        '',
        '1. Add per-reaction source/loss observables for the ZDPlaskin case, limited to benchmark diagnostics.',
        '2. Add a circuit waveform comparison around the first 20 microseconds, where the peak mismatch is created.',
        '3. Rebuild the output-derived rate table from the full ZDPlaskin output if more transient columns become available.',
        '4. Only after the decomposition identifies a single cause, adjust `rate_table` relaxation, `dc_series_circuit`, or wall-loss closure.',
        '',
        '## PyGMol-1 Precision Gap Analysis',
        '',
        'The current PyGMol comparison remains useful as a sanity check, but not as a precision benchmark. Required concrete measures:',
        '',
    ])
    for row in pygmol_rows:
        lines.extend([
            f"### {row['gap_id']}",
            '',
            f"- Current state: {row['current_state']}",
            f"- Needed for precision: {row['needed_for_precision']}",
            f"- Method: {row['action']}",
            f"- Code area: {row['owner_area']}",
            '',
        ])
    lines.extend([
        'Recommended PyGMol work order:',
        '',
        '1. Generate a dedicated single-zone local case from `pygmol_argon_model.yaml` rather than comparing against the two-zone LXCat case.',
        '2. Use the same power waveform and same compact chemistry in both solvers.',
        '3. Add a same-wall-loss mode before comparing ion flux or afterglow decay.',
        '4. Use `PyGMol-Precision-1` for compact same-footing precision; keep `PyGMol-1` scoped as production-case sanity only.',
        '',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def _public_zdp_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hidden = {'case_path', 'output_dir'}
    return [{key: value for key, value in row.items() if key not in hidden} for row in rows]


def public_zdp_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _public_zdp_rows(rows)


def run_remediation(output_dir: Path = DEFAULT_OUTPUT_DIR, *, keep_scratch: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / '_scratch_zdplaskin_peak_sensitivity'
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    zdp_rows = run_zdplaskin_peak_sensitivity(scratch_dir)
    pygmol_rows = _pygmol_gap_rows()
    public_zdp_rows = _public_zdp_rows(zdp_rows)
    candidate_rows = _model_implementation_candidate_rows()
    rejection_rows = _benchmark_fit_rejection_rows()
    simplification_rows = _code_simplification_rows()
    zdp_csv = output_dir / 'zdplaskin_peak_sensitivity.csv'
    _write_csv(zdp_csv, public_zdp_rows, list(public_zdp_rows[0].keys()))
    pygmol_csv = output_dir / 'pygmol_same_footing_gap_analysis.csv'
    _write_csv(pygmol_csv, pygmol_rows, list(pygmol_rows[0].keys()))
    candidates_csv = output_dir / 'model_implementation_candidates.csv'
    _write_csv(candidates_csv, candidate_rows, list(candidate_rows[0].keys()))
    rejected_csv = output_dir / 'benchmark_fit_rejections.csv'
    _write_csv(rejected_csv, rejection_rows, list(rejection_rows[0].keys()))
    simplification_csv = output_dir / 'code_simplification_findings.csv'
    _write_csv(simplification_csv, simplification_rows, list(simplification_rows[0].keys()))
    fig_path = output_dir / 'figures' / 'zdplaskin_peak_sensitivity.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_zdplaskin_sensitivity(fig_path, zdp_rows)
    actions_md = output_dir / 'benchmark_remediation_actions.md'
    _write_remediation_markdown(actions_md, zdp_rows, pygmol_rows)
    candidates_md = output_dir / 'model_implementation_candidates.md'
    _write_model_candidates_markdown(candidates_md, candidate_rows, rejection_rows)
    simplification_md = output_dir / 'code_simplification_findings.md'
    _write_code_simplification_markdown(simplification_md, simplification_rows)
    report = {
        'tool': 'benchmark_remediation',
        'output_dir': str(output_dir),
        'artifacts': {
            'actions_md': str(actions_md),
            'model_candidates_md': str(candidates_md),
            'zdplaskin_sensitivity_csv': str(zdp_csv),
            'pygmol_gap_csv': str(pygmol_csv),
            'model_candidates_csv': str(candidates_csv),
            'benchmark_fit_rejections_csv': str(rejected_csv),
            'code_simplification_findings_md': str(simplification_md),
            'code_simplification_findings_csv': str(simplification_csv),
            'zdplaskin_sensitivity_png': str(fig_path),
        },
        'zdplaskin_best_variant': _public_zdp_rows([min(zdp_rows, key=lambda row: float(row['peak_electron_density_relative_error']))])[0],
        'model_candidate_count': len(candidate_rows),
        'benchmark_fit_rejection_count': len(rejection_rows),
        'code_simplification_finding_count': len(simplification_rows),
        'pygmol_gap_count': len(pygmol_rows),
    }
    _write_yaml(output_dir / 'benchmark_remediation_report.yaml', report)
    if not keep_scratch and scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Turn benchmark findings into concrete remediation actions and sensitivity checks.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--keep-scratch', action='store_true', help='Keep generated variant cases and run outputs for manual inspection.')
    args = parser.parse_args(argv)
    report = run_remediation(args.output_dir.resolve(), keep_scratch=args.keep_scratch)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
