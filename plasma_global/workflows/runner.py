from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.chemistry.provenance import chemistry_provenance_summary
from plasma_global.config.export import write_effective_config, write_resolved_paths
from plasma_global.io.hdf5_writer import write_observables_csv, write_solution_h5, write_summary_yaml
from plasma_global.numerics.solver_base import SolverResult
from plasma_global.observables.defaults import summarize_solution
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


def run_from_yaml(run_yaml_path: str | Path) -> dict[str, Any]:
    loaded = load_case_from_yaml(run_yaml_path)
    built = build_case(loaded)

    system = built.system
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
        seg.y = system.project_trajectory(seg.y)
        results.append(seg)
        y0 = system.project_state(seg.y[:, -1].copy())
        if int(seg.diagnostics.get('steady_state_event_count', 0) or 0) > 0:
            break

    solution = _concatenate(results)
    solution.y = system.project_trajectory(solution.y)
    chemistry_provenance = chemistry_provenance_summary(loaded.mechanism)
    eedf_provenance_fn = getattr(built.eedf_backend, 'provenance', None)
    eedf_provenance = eedf_provenance_fn() if callable(eedf_provenance_fn) else {}
    observables = system.compute_observables(solution.t, solution.y)
    summary = summarize_solution(solution, observables, chemistry_provenance)
    if eedf_provenance:
        summary['eedf_provenance'] = eedf_provenance

    output_dir = Path(loaded.resolved_paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if loaded.run_config.runtime.export_effective_config:
        write_effective_config(output_dir / 'effective_case.yaml', loaded.run_config)
    if loaded.run_config.runtime.export_resolved_paths:
        write_resolved_paths(output_dir / 'resolved_paths.yaml', loaded.resolved_paths)

    if loaded.run_config.outputs.formats.solution_h5:
        write_solution_h5(output_dir / 'solution.h5', solution, state_labels=system.state_labels())
    if loaded.run_config.outputs.formats.observables_csv:
        write_observables_csv(output_dir / 'observables.csv', observables)
    if loaded.run_config.outputs.formats.summary_yaml:
        write_summary_yaml(output_dir / 'summary.yaml', summary)
    make_requested_plots(loaded.run_config, observables, output_dir)

    return {
        'run_config': loaded.run_config,
        'resolved_paths': loaded.resolved_paths,
        'chamber': loaded.chamber,
        'recipe': loaded.recipe,
        'mechanism': loaded.mechanism,
        'validation_messages': loaded.validation_messages,
        'chemistry_provenance': chemistry_provenance,
        'eedf_provenance': eedf_provenance,
        'observables': observables,
        'summary': summary,
        'output_dir': str(output_dir),
        'solution': solution,
        'system': system,
        'eedf_backend': built.eedf_backend,
        'electrical_backend': built.electrical_backend,
        'integrator': built.integrator,
    }
