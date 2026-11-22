"""Compile recipe steps into smooth, command-bound solver segments."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.domain import RecipeSegment
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.profile import PrescribedElectronProfile
from plasma_global.input._compile_recipe_power import (
    MAX_COMPILED_SEGMENTS,
    bind_step_power_tables,
    parse_step_power_commands,
    power_step_boundaries,
    resolved_step_power_commands,
)
from plasma_global.input._compile_recipe_transport import (
    compile_segment_transport,
    step_temperatures,
)
from plasma_global.input._schema_power import PowerPortCommand
from plasma_global.input.schema import CaseSpec, PowerDefinition, RunStep
from plasma_global.models.external_table import ExternalTableBinding, ExternalTableStore


def compile_recipe(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
    prescribed_electron_profile: PrescribedElectronProfile | None = None,
) -> tuple[RecipeSegment, ...]:
    """Split every discontinuity and bind constant transport/power commands."""

    ports = case.reactor.power_ports
    sampled_profile = (
        prescribed_electron_profile
        if prescribed_electron_profile is not None
        and prescribed_electron_profile.interpolation == "previous"
        else None
    )
    segments: list[RecipeSegment] = []
    for step_index, (step, (start, end)) in enumerate(
        zip(case.run.normalized_steps(), case.run.step_bounds(), strict=True), 1
    ):
        commands = parse_step_power_commands(step, ports)
        step_id = f"step_{step_index}"
        external_bindings = bind_step_power_tables(
            commands,
            ports,
            external_tables,
            run_start_s=case.run.start_time_s,
            step_start_s=start,
        )
        boundaries = power_step_boundaries(
            commands, external_bindings, start, end, run_start_s=case.run.start_time_s
        )
        if sampled_profile is not None:
            boundaries.update(
                float(time_s)
                for time_s in sampled_profile.time_s
                if start < time_s < end
            )
        ordered = [start, *sorted(boundaries), end]
        if len(ordered) - 1 + len(segments) > MAX_COMPILED_SEGMENTS:
            raise CaseValidationError("case creates too many compiled recipe segments")
        segments.extend(
            _compile_step(
                case,
                chemistry_data,
                chemistry,
                step,
                step_id,
                commands,
                ordered,
                ports,
                external_bindings,
                sampled_profile,
            )
        )
    return tuple(segments)


def _compile_step(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    step: RunStep,
    step_id: str,
    commands: dict[str, PowerPortCommand],
    boundaries: list[float],
    ports: dict[str, PowerDefinition],
    bindings: dict[str, ExternalTableBinding],
    profile: PrescribedElectronProfile | None,
) -> Iterable[RecipeSegment]:
    transport = compile_segment_transport(case, chemistry_data, chemistry, step)
    surface_temperatures, wall_temperatures = step_temperatures(case, step)
    for index, (left, right) in enumerate(pairwise(boundaries), start=1):
        midpoint = left + 0.5 * (right - left)
        yield RecipeSegment(
            segment_id=(step_id if len(boundaries) == 2 else f"{step_id}[{index:04d}]"),
            start_s=left,
            end_s=right,
            surface_temperature_K_by_surface=surface_temperatures,
            wall_temperature_K_by_zone=wall_temperatures,
            transport=transport,
            port_commands=resolved_step_power_commands(
                commands,
                ports,
                bindings,
                midpoint,
                run_start_s=case.run.start_time_s,
                step_start_s=boundaries[0],
                step_id=step_id,
            ),
            prescribed_electron_density_m3_by_zone=(
                {}
                if profile is None
                else {
                    zone_id: profile.density(midpoint, zone_id)
                    for zone_id in case.reactor.zones
                }
            ),
        )


__all__ = ["compile_recipe"]
