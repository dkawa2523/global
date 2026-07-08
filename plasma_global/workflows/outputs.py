from __future__ import annotations

from pathlib import Path
from typing import Any

from plasma_global.chemistry.provenance import chemistry_provenance_summary
from plasma_global.config.export import write_effective_config, write_resolved_paths
from plasma_global.config.models import RunConfig
from plasma_global.eedf.registry import EEDF_REGISTRY
from plasma_global.electrical.registry import ELECTRICAL_REGISTRY
from plasma_global.io.hdf5_writer import write_observables_csv, write_solution_h5, write_summary_yaml
from plasma_global.numerics.registry import INTEGRATOR_REGISTRY
from plasma_global.numerics.system import GlobalPlasmaSystem
from plasma_global.numerics.solver_base import SolverResult
from plasma_global.observables.adapter import compute_observable_record, compute_observables
from plasma_global.observables.defaults import summarize_solution
from plasma_global.plotters.defaults import make_requested_plots
from plasma_global.workflows.types import BuiltCase, LoadedCase


def write_run_outputs(loaded: LoadedCase, built: BuiltCase, solution: SolverResult) -> dict[str, Any]:
    payload = _collect_run_outputs(loaded, built, solution)
    output_dir = Path(loaded.resolved_paths.output_dir)
    if _needs_output_dir(loaded.run_config):
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_output_files(loaded, built.system, solution, output_dir, payload['observables'], payload['summary'])
    return {**payload, 'output_dir': str(output_dir)}


def _collect_run_outputs(loaded: LoadedCase, built: BuiltCase, solution: SolverResult) -> dict[str, Any]:
    system = built.system
    chemistry_provenance = chemistry_provenance_summary(loaded.mechanism)
    eedf_provenance_fn = getattr(built.eedf_backend, 'provenance', None)
    eedf_provenance = eedf_provenance_fn() if callable(eedf_provenance_fn) else {}
    backend_metadata = _selected_backend_metadata(loaded)
    observables = _collect_observables(system, solution, loaded.run_config)
    summary_observables = _summary_observables(system, solution, loaded.run_config, observables)
    summary = summarize_solution(solution, summary_observables, chemistry_provenance)
    summary['backend_metadata'] = backend_metadata
    if eedf_provenance:
        summary['eedf_provenance'] = eedf_provenance

    return {
        'chemistry_provenance': chemistry_provenance,
        'eedf_provenance': eedf_provenance,
        'backend_metadata': backend_metadata,
        'observables': observables,
        'summary': summary,
    }


def _selected_backend_metadata(loaded: LoadedCase) -> dict[str, dict[str, str]]:
    selections = {
        'eedf': (loaded.run_config.physics.eedf_backend, EEDF_REGISTRY),
        'electrical': (loaded.run_config.physics.electrical_backend, ELECTRICAL_REGISTRY),
        'integrator': (loaded.run_config.physics.integrator, INTEGRATOR_REGISTRY),
    }
    metadata: dict[str, dict[str, str]] = {}
    for category, (name, catalog) in selections.items():
        detail = dict(catalog.details().get(name, {}))
        detail['name'] = str(name)
        metadata[category] = detail
    return metadata


def _collect_observables(system: GlobalPlasmaSystem, solution: SolverResult, run_config: RunConfig) -> list[dict[str, Any]]:
    if _needs_full_observables(run_config):
        return compute_observables(system, solution.t, solution.y)
    return []


def _summary_observables(
    system: GlobalPlasmaSystem,
    solution: SolverResult,
    run_config: RunConfig,
    observables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if observables:
        return observables
    if run_config.outputs.formats.summary_yaml and solution.t.size:
        return [_final_observable_record(system, solution)]
    return []


def _final_observable_record(system: GlobalPlasmaSystem, solution: SolverResult) -> dict[str, Any]:
    return compute_observable_record(system, float(solution.t[-1]), solution.y[:, -1])


def _needs_full_observables(run_config: RunConfig) -> bool:
    return bool(
        run_config.outputs.formats.observables_csv
        or run_config.outputs.plots.enabled
    )


def _needs_output_dir(run_config: RunConfig) -> bool:
    return bool(
        run_config.runtime.export_effective_config
        or run_config.runtime.export_resolved_paths
        or run_config.outputs.formats.solution_h5
        or run_config.outputs.formats.observables_csv
        or run_config.outputs.formats.summary_yaml
        or run_config.outputs.plots.enabled
    )


def _write_output_files(
    loaded: LoadedCase,
    system: GlobalPlasmaSystem,
    solution: SolverResult,
    output_dir: Path,
    observables: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
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
