"""Public end-to-end workflow entry point.

This module is intentionally small so the top-level processing path is easy to
read: load inputs, build runtime objects, solve, and write outputs. Most design
details live one layer down in `case_loader`, `case_builder`, `solve`, and
`outputs`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plasma_global.workflows.case_builder import build_case
from plasma_global.workflows.case_loader import load_case_from_yaml
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
        'backend_metadata': outputs['backend_metadata'],
        'observables': outputs['observables'],
        'summary': outputs['summary'],
        'output_dir': outputs['output_dir'],
        'solution': solution,
        'system': system,
        'eedf_backend': built.eedf_backend,
        'electrical_backend': built.electrical_backend,
        'integrator': built.integrator,
    }
