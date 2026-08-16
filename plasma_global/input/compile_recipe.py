"""Compile recipe steps into smooth, command-bound solver segments."""

from __future__ import annotations

from itertools import pairwise

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.domain import RecipeSegment
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.profile import PrescribedElectronProfile
from plasma_global.input._compile_recipe_power import (
    MAX_COMPILED_SEGMENTS,
    compiled_step_boundaries,
    prescribed_electron_density_at,
    resolved_step_power_commands,
)
from plasma_global.input._compile_recipe_transport import (
    compile_segment_transport,
    step_temperatures,
)
from plasma_global.input.schema import CaseSpec
from plasma_global.models.external_table import ExternalTableStore


def compile_recipe(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
    prescribed_electron_profile: PrescribedElectronProfile | None = None,
) -> tuple[RecipeSegment, ...]:
    """Split every discontinuity and bind constant transport/power commands."""

    ports = {port.port_id: port for port in case.reactor.power_ports}
    segments: list[RecipeSegment] = []
    for step, (start, end) in zip(
        case.recipe.steps,
        case.recipe.step_bounds(),
        strict=True,
    ):
        ordered, external_bindings = compiled_step_boundaries(
            step,
            ports,
            external_tables,
            prescribed_electron_profile,
            start,
            end,
        )
        if len(ordered) - 1 + len(segments) > MAX_COMPILED_SEGMENTS:
            raise CaseValidationError("case creates too many compiled recipe segments")
        transport = compile_segment_transport(case, chemistry_data, chemistry, step)
        surface_temperatures, wall_temperatures = step_temperatures(case, step)
        split_count = len(ordered) - 1
        for index, (left, right) in enumerate(pairwise(ordered), start=1):
            midpoint = left + 0.5 * (right - left)
            commands = resolved_step_power_commands(
                step,
                ports,
                external_bindings,
                external_tables,
                midpoint,
                start,
            )
            segment_id = (
                step.step_id if split_count == 1 else f"{step.step_id}[{index:04d}]"
            )
            electron_density = prescribed_electron_density_at(
                case,
                prescribed_electron_profile,
                midpoint,
            )
            segments.append(
                RecipeSegment(
                    segment_id=segment_id,
                    start_s=left,
                    end_s=right,
                    surface_temperature_K_by_surface=surface_temperatures,
                    wall_temperature_K_by_zone=wall_temperatures,
                    transport=transport,
                    port_commands=commands,
                    prescribed_electron_density_m3_by_zone=electron_density,
                )
            )
    return tuple(segments)


__all__ = ["compile_recipe"]
