from __future__ import annotations

from pathlib import Path
from typing import Any

from plasma_global.workflows.context import build_case, load_case_from_yaml
from plasma_global.workflows.outputs import write_run_outputs
from plasma_global.workflows.solve import solve_built_case


def run_from_yaml(run_yaml_path: str | Path) -> dict[str, Any]:
    loaded = load_case_from_yaml(run_yaml_path)
    built = build_case(loaded)

    solution = solve_built_case(built)
    outputs = write_run_outputs(loaded, built, solution)
    system = built.system

    return {
        'run_config': loaded.run_config,
        'resolved_paths': loaded.resolved_paths,
        'chamber': loaded.chamber,
        'recipe': loaded.recipe,
        'mechanism': loaded.mechanism,
        'validation_messages': loaded.validation_messages,
        'chemistry_provenance': outputs['chemistry_provenance'],
        'eedf_provenance': outputs['eedf_provenance'],
        'observables': outputs['observables'],
        'summary': outputs['summary'],
        'output_dir': outputs['output_dir'],
        'solution': solution,
        'system': system,
        'eedf_backend': built.eedf_backend,
        'electrical_backend': built.electrical_backend,
        'integrator': built.integrator,
    }
