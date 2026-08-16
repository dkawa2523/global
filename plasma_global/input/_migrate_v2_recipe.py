"""Translate schema-v2 recipe steps and output sampling to schema v3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from plasma_global.input._migrate_v2_common import float_or_none, record_extra_keys
from plasma_global.input._migrate_v2_power import migrate_step_power_commands


def _migrate_step_surfaces(
    step: Any, step_index: int, unused: set[str]
) -> dict[str, Any]:
    surfaces: dict[str, Any] = {}
    for surface_id, raw_values in step.surface_overrides.items():
        values = dict(raw_values or {})
        prefix = f"recipe.steps[{step_index}].surface_overrides.{surface_id}"
        record_extra_keys(values, {"temperature_K"}, prefix, unused)
        surfaces[str(surface_id)] = {
            "temperature_K": float_or_none(values.get("temperature_K"))
        }
    return surfaces


def migrate_recipe_steps(
    *,
    recipe: Any,
    resolved: Any,
    port_model_by_id: Mapping[str, Mapping[str, Any]],
    used_external: set[str],
    unused: set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    previous_end: float | None = None
    recipe_dir = Path(resolved.recipe_file).parent
    for index, step in enumerate(recipe.steps):
        start, end = float(step.t_start_s), float(step.t_end_s)
        if previous_end is not None and abs(start - previous_end) > max(
            1.0e-15, 1.0e-12 * max(abs(start), abs(previous_end), 1.0)
        ):
            warnings.append(
                f"recipe.steps[{index}] began at {start:g}s instead of the previous "
                f"end {previous_end:g}s; v3 makes steps contiguous"
            )
        previous_end = end
        power_commands = migrate_step_power_commands(
            step=step,
            step_index=index,
            port_model_by_id=port_model_by_id,
            recipe_dir=recipe_dir,
            external_inputs=resolved.external_inputs,
            used_external=used_external,
            unused=unused,
            warnings=warnings,
        )
        surfaces = _migrate_step_surfaces(step, index, unused)
        unused.update(
            f"recipe.steps[{index}].imported_inputs.{key}"
            for key in step.imported_inputs
        )
        steps.append(
            {
                "step_id": str(step.step_id),
                "duration_s": end - start,
                "commands": {
                    "gas_inlets": {
                        str(inlet_id): {
                            "flow_sccm": {
                                str(species): float(flow)
                                for species, flow in values.items()
                            }
                        }
                        for inlet_id, values in step.gas_inlets.items()
                    },
                    "power_ports": power_commands,
                    "surfaces": surfaces,
                },
            }
        )
    return steps


def migrate_output(
    run: Any,
    recipe_steps: Sequence[Mapping[str, Any]],
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], float]:
    formats = run.outputs.formats
    if not formats.solution_h5:
        warnings.append("v3 always writes the HDF5 result bundle")
    if not formats.summary_yaml:
        warnings.append("v3 always writes summary.yaml")
    if formats.observables_csv:
        unused.add("outputs.formats.observables_csv")
    if run.outputs.plots.enabled:
        unused.add("outputs.plots")
    if run.outputs.budgets.enabled:
        unused.add("outputs.budgets")
    unused.add("files.output_dir")
    total_duration = sum(float(item["duration_s"]) for item in recipe_steps)
    return {}, max(total_duration / 199.0, 1.0e-15)
