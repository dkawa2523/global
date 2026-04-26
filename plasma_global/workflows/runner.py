from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.config.export import write_effective_config, write_resolved_paths
from plasma_global.diagnostics.provenance import build_run_provenance
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
    return SolverResult(t=t, y=y, success=success, status=status, message=message, diagnostics=diagnostics)


def _diagnostic_config(run_config: Any, name: str) -> Any:
    diagnostics = getattr(run_config.outputs, 'diagnostics', None)
    return getattr(diagnostics, name, None) if diagnostics is not None else None


def _reaction_budget_config(run_config: Any) -> Any:
    return _diagnostic_config(run_config, 'reaction_budget')


def _maybe_write_reaction_budget(output_dir: Path, run_config: Any, system: Any, solution: SolverResult) -> None:
    cfg = _reaction_budget_config(run_config)
    if not bool(getattr(cfg, 'enabled', False)):
        return
    species = getattr(cfg, 'species', None)
    max_items = int(getattr(cfg, 'max_reactions_per_species', 5))
    filename = str(getattr(cfg, 'filename', 'reaction_budget.yaml'))
    budget = system.reaction_budget(
        float(solution.t[-1]),
        solution.y[:, -1],
        species_filter=list(species) if species else None,
        max_reactions_per_species=max_items,
    )
    write_summary_yaml(output_dir / filename, budget)


def _maybe_write_electron_energy_budget(output_dir: Path, run_config: Any, system: Any, solution: SolverResult) -> None:
    cfg = _diagnostic_config(run_config, 'electron_energy_budget')
    if not bool(getattr(cfg, 'enabled', False)):
        return
    filename = str(getattr(cfg, 'filename', 'electron_energy_budget.yaml'))
    max_items = int(getattr(cfg, 'max_reactions', 8))
    budget = system.electron_energy_budget(float(solution.t[-1]), solution.y[:, -1], max_reactions=max_items)
    write_summary_yaml(output_dir / filename, budget)


def _maybe_write_surface_reaction_budget(output_dir: Path, run_config: Any, system: Any, solution: SolverResult) -> None:
    cfg = _diagnostic_config(run_config, 'surface_reaction_budget')
    if not bool(getattr(cfg, 'enabled', False)):
        return
    filename = str(getattr(cfg, 'filename', 'surface_reaction_budget.yaml'))
    max_items = int(getattr(cfg, 'max_reactions_per_species', 5))
    budget = system.surface_reaction_budget(
        float(solution.t[-1]),
        solution.y[:, -1],
        max_reactions_per_species=max_items,
    )
    write_summary_yaml(output_dir / filename, budget)


def _maybe_write_state_manifest(output_dir: Path, run_config: Any, system: Any) -> None:
    cfg = _diagnostic_config(run_config, 'state_manifest')
    if not bool(getattr(cfg, 'enabled', False)):
        return
    filename = str(getattr(cfg, 'filename', 'state_manifest.yaml'))
    write_summary_yaml(output_dir / filename, system.state_manifest())


def _maybe_write_provenance(output_dir: Path, run_config: Any, loaded: Any, built: Any) -> None:
    cfg = _diagnostic_config(run_config, 'provenance')
    if not bool(getattr(cfg, 'enabled', False)):
        return
    filename = str(getattr(cfg, 'filename', 'run_provenance.yaml'))
    write_summary_yaml(output_dir / filename, build_run_provenance(loaded, built))


def run_from_yaml(run_yaml_path: str | Path) -> dict[str, Any]:
    loaded = load_case_from_yaml(run_yaml_path)
    built = build_case(loaded)

    system = built.system
    recipe = loaded.recipe
    y0 = system.project_state(system.initial_state())

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

    solution = _concatenate(results)
    solution.y = system.project_trajectory(solution.y)
    solution.diagnostics['observables'] = system.compute_observables(solution.t, solution.y)
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
    _maybe_write_reaction_budget(output_dir, loaded.run_config, system, solution)
    _maybe_write_electron_energy_budget(output_dir, loaded.run_config, system, solution)
    _maybe_write_surface_reaction_budget(output_dir, loaded.run_config, system, solution)
    _maybe_write_state_manifest(output_dir, loaded.run_config, system)
    _maybe_write_provenance(output_dir, loaded.run_config, loaded, built)
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
