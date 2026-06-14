from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.chemistry.provenance import chemistry_provenance_summary
from plasma_global.config.export import write_effective_config, write_resolved_paths
from plasma_global.io.hdf5_writer import write_observables_csv, write_solution_h5, write_summary_yaml
from plasma_global.numerics.solver_base import SolverResult
from plasma_global.observables.defaults import observables_dataframe, summarize_solution
from plasma_global.plotters.defaults import make_requested_plots
from plasma_global.workflows.context import build_case, load_case_from_yaml


def _concatenate(results: list[SolverResult]) -> SolverResult:
    if len(results) == 1:
        return results[0]
    t_parts = []
    y_parts = []
    for i, res in enumerate(results):
        if i == 0:
            t_parts.append(res.t)
            y_parts.append(res.y)
        else:
            t_parts.append(res.t[1:])
            y_parts.append(res.y[:, 1:])
    t = np.concatenate(t_parts)
    y = np.concatenate(y_parts, axis=1)
    success = all(r.success for r in results)
    status = 0 if success else next(r.status for r in results if not r.success)
    message = '; '.join(r.message for r in results)
    diagnostics = {'segments': len(results)}
    for key in ['nfev', 'njev', 'nlu']:
        diagnostics[key] = sum((r.diagnostics.get(key) or 0) for r in results)
    event_counts: dict[str, int] = {}
    for res in results:
        for name, count in (res.diagnostics.get('event_counts') or {}).items():
            event_counts[str(name)] = event_counts.get(str(name), 0) + int(count)
    diagnostics['event_counts'] = event_counts
    diagnostics['solver_event_count'] = int(sum(event_counts.values()))
    diagnostics['steady_state_event_count'] = int(event_counts.get('steady_state', 0))
    return SolverResult(t=t, y=y, success=success, status=status, message=message, diagnostics=diagnostics)


def _electrical_coupling_summary(system: Any) -> dict[str, Any] | None:
    power = getattr(getattr(system, 'electrical_adapter', None), 'last_power', None)
    if power is None:
        power = getattr(system, '_last_power', None)
    circuit_interface = ((getattr(power, 'metadata', None) or {}).get('circuit_interface') if power is not None else None)
    if isinstance(circuit_interface, dict):
        return {'circuit_interface': circuit_interface}
    return None


def run_from_yaml(run_yaml_path: str | Path) -> dict[str, Any]:
    loaded = load_case_from_yaml(run_yaml_path)
    built = build_case(loaded)

    system = built.system
    if hasattr(system, 'reset_numerical_diagnostics'):
        system.reset_numerical_diagnostics()
    recipe = loaded.recipe
    y0 = system.initial_state()

    results: list[SolverResult] = []
    t_start = recipe.steps[0].t_start_s
    t_end = recipe.steps[-1].t_end_s
    total_duration = max(t_end - t_start, 1.0e-12)
    for step in recipe.steps:
        dt = max(step.t_end_s - step.t_start_s, 1.0e-12)
        n_eval = max(5, int(200 * dt / total_duration))
        t_eval = np.linspace(step.t_start_s, step.t_end_s, n_eval)
        seg = built.integrator.solve(system=system, y0=y0, t_span=(step.t_start_s, step.t_end_s), t_eval=t_eval)
        if not seg.success:
            step_id = getattr(step, 'step_id', f'step_{len(results)}')
            raise RuntimeError(f"Solver failed in recipe step {step_id!r}: {seg.message}")
        seg.y = system.project_trajectory(seg.y, count_diagnostics=True)
        results.append(seg)
        y0 = system.project_state(seg.y[:, -1].copy(), count_diagnostics=False)
        if int(seg.diagnostics.get('steady_state_event_count', 0) or 0) > 0:
            break

    solution = _concatenate(results)
    solution.y = system.project_trajectory(solution.y, count_diagnostics=False)
    if solution.t.size:
        system.update_final_rhs_diagnostics(float(solution.t[-1]), solution.y[:, -1])
    system.diagnostics['solver_event_count'] = int(solution.diagnostics.get('solver_event_count', 0) or 0)
    system.diagnostics['steady_state_event_count'] = int(solution.diagnostics.get('steady_state_event_count', 0) or 0)
    solution.diagnostics.update(system.diagnostics)
    solution.diagnostics['chemistry_provenance'] = chemistry_provenance_summary(loaded.mechanism)
    solution.diagnostics['observables'] = system.compute_observables(solution.t, solution.y)
    electrical_coupling = _electrical_coupling_summary(system)
    if electrical_coupling:
        solution.diagnostics['electrical_coupling'] = electrical_coupling
    summary = summarize_solution(solution)
    observables = observables_dataframe(solution)

    output_dir = Path(loaded.resolved_paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if loaded.run_config.runtime.export_effective_config:
        write_effective_config(output_dir / 'effective_case.yaml', loaded.run_config)
    if loaded.run_config.runtime.export_resolved_paths:
        write_resolved_paths(output_dir / 'resolved_paths.yaml', loaded.resolved_paths)

    if getattr(loaded.run_config.outputs.formats, 'solution_h5', False):
        write_solution_h5(output_dir / 'solution.h5', solution, state_labels=system.state_labels())
    if getattr(loaded.run_config.outputs.formats, 'observables_csv', False):
        write_observables_csv(output_dir / 'observables.csv', observables)
    if getattr(loaded.run_config.outputs.formats, 'summary_yaml', False):
        write_summary_yaml(output_dir / 'summary.yaml', summary)
    make_requested_plots(loaded.run_config, observables, output_dir)

    return {
        'run_config': loaded.run_config,
        'resolved_paths': loaded.resolved_paths,
        'chamber': loaded.chamber,
        'recipe': loaded.recipe,
        'mechanism': loaded.mechanism,
        'validation_messages': loaded.validation_messages,
        'summary': summary,
        'output_dir': str(output_dir),
        'solution': solution,
        'system': system,
        'eedf_backend': built.eedf_backend,
        'electrical_backend': built.electrical_backend,
        'integrator': built.integrator,
    }
