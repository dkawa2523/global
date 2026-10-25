"""Compile output requests against the actual model and keep them in the result."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_global._output_channels import EventChannel, event_channels
from plasma_global._output_values import PORT_UNITS, port_values
from plasma_global._series_selection import expand_series
from plasma_global.chemistry.data import ChemistryData, ReactionData
from plasma_global.core._solver_sampling import _global_sample_times
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import SolverSettings
from plasma_global.core.result import series_unit
from plasma_global.errors import CaseValidationError
from plasma_global.input._resolved_case import ResolvedCase
from plasma_global.input._schema_output import NetworkConfig, PlotConfig, ProbeConfig
from plasma_global.postprocess import available_observables


@dataclass(frozen=True, slots=True)
class OutputPlan:
    stored: tuple[str, ...]
    evaluated: tuple[str, ...]
    aliases: Mapping[str, str]
    port_names: tuple[str, ...]
    metadata: Mapping[str, Any]


def composition_names(plot: PlotConfig, model: CompiledGlobalModel) -> tuple[str, ...]:
    if plot.zone not in model.layout.zone_ids:
        raise CaseValidationError(f"unknown composition zone {plot.zone!r}")
    species = model.species_ids if plot.species == "all" else tuple(plot.species)
    if not species or len(set(species)) != len(species):
        raise CaseValidationError("composition species must be nonempty and unique")
    if unknown := set(species) - set(model.species_ids):
        raise CaseValidationError(f"unknown composition species: {sorted(unknown)}")
    prefix = "n" if plot.quantity == "number_density_m3" else "mole_fraction"
    return tuple(f"{prefix}[{plot.zone},{item}]" for item in species)


def _probe_source_names(
    probe: ProbeConfig, model: CompiledGlobalModel
) -> tuple[str, list[str]]:
    if probe.zone is not None:
        return probe.zone, probe.quantities or [
            "electron_density_m3",
            "electron_temperature_eV",
            "number_density_m3",
        ]
    if probe.surface is not None:
        return probe.surface, probe.quantities or ["incident_flux_m2_s"]
    return str(probe.power_port), probe.quantities or [
        "electron_power_W",
        "gas_power_W",
    ]


def _surface_species(model: CompiledGlobalModel, surface: str) -> tuple[str, ...]:
    names = [
        name for region, name in model.layout.surface_coverage_keys if region == surface
    ]
    if model.surface_model is not None:
        free = model.surface_model.layout.free_species_by_surface.get(surface)
        if free is not None:
            names.append(free)
    return tuple(names)


def _probe_species(
    probe: ProbeConfig, quantity: str, model: CompiledGlobalModel
) -> tuple[str, ...] | None:
    if quantity in {
        "number_density_m3",
        "mole_fraction",
        "partial_pressure_Pa",
        "net_production_m3_s",
    }:
        names = model.species_ids
    elif quantity == "incident_flux_m2_s":
        names = tuple(
            name
            for name, charge in zip(model.species_ids, model.charges, strict=True)
            if charge > 0
        )
    elif quantity == "coverage":
        names = _surface_species(model, str(probe.surface))
    else:
        return None
    return names if probe.species == "all" else tuple(probe.species)


_SURFACE_QUANTITIES = {
    "coverage",
    "incident_flux_m2_s",
    "wall_sheath_energy_eV",
    "surface_temperature_K",
}


def _probe_targets(probe: ProbeConfig, model: CompiledGlobalModel) -> set[str]:
    if probe.zone is not None:
        return set(model.layout.zone_ids)
    if probe.surface is not None:
        targets = {
            str(wall.surface_id)
            for wall in model.wall_boundaries
            if wall.surface_id is not None
        }
        if model.surface_model is not None:
            targets.update(
                surface.surface_id for surface in model.surface_model.surfaces
            )
        return targets
    ports = () if model.power_coordinator is None else model.power_coordinator.ports
    return {str(port.port_id) for port in ports}


def _validate_probe_target(
    probe: ProbeConfig, model: CompiledGlobalModel, target: str, quantities: list[str]
) -> None:
    if probe.zone is not None and set(quantities) & _SURFACE_QUANTITIES:
        raise CaseValidationError("zone probes cannot request surface quantities")
    if probe.surface is not None and set(quantities) - _SURFACE_QUANTITIES:
        raise CaseValidationError("surface probes require modeled surface quantities")
    if probe.power_port is not None and probe.species != "all":
        raise CaseValidationError("power-port probes do not select species")
    if target not in _probe_targets(probe, model):
        raise CaseValidationError(f"unknown probe target {target!r}")


def _probe_aliases(
    identity: str,
    probe: ProbeConfig,
    model: CompiledGlobalModel,
    available: set[str],
) -> dict[str, str]:
    target, quantities = _probe_source_names(probe, model)
    _validate_probe_target(probe, model, target, quantities)
    aliases: dict[str, str] = {}
    uses_species = False
    for quantity in quantities:
        species = _probe_species(probe, quantity, model)
        prefix = "n" if quantity == "number_density_m3" else quantity
        if probe.power_port is not None:
            prefix = f"port_{prefix}"
        if species is None:
            aliases[f"probe[{identity},{quantity}]"] = f"{prefix}[{target}]"
        else:
            uses_species = True
            aliases.update(
                {
                    f"probe[{identity},{quantity},{item}]": f"{prefix}[{target},{item}]"
                    for item in species
                }
            )
    if probe.species != "all" and not uses_species:
        raise CaseValidationError(
            f"probe {identity}: species is unused by its quantities"
        )
    if not aliases:
        raise CaseValidationError(f"probe {identity}: no modeled quantities match")
    if unknown := set(aliases.values()) - available:
        raise CaseValidationError(
            f"probe {identity}: unavailable quantities {sorted(unknown)}"
        )
    return aliases


def _port_names(
    case: ResolvedCase, model: CompiledGlobalModel, initial: np.ndarray
) -> tuple[str, ...]:
    requested = any(
        probe.power_port is not None for probe in case.output.probes.values()
    )
    requested = requested or any("port_" in name for name in case.output.observables)
    requested = requested or any("port_" in name for name in case.output.summary_series)
    requested = requested or any(
        any("port_" in name for name in plot.series)
        for plot in case.output.plots.values()
    )
    if not requested:
        return ()
    segment = model.segments[0]
    return tuple(port_values(model.evaluate(segment.start_s, initial, segment)))


def _reaction_metadata(
    channel: EventChannel, definitions: Mapping[tuple[str, str], ReactionData]
) -> dict[str, Any]:
    reaction = definitions[channel.domain, channel.reaction]
    return {
        "id": f"{channel.domain}:{channel.region}:{channel.reaction}",
        "domain": channel.domain,
        "region": channel.region,
        "zone": channel.zone,
        "reaction_id": channel.reaction,
        "reactants": dict(reaction.reactants),
        "products": dict(reaction.products),
        "rate_series": channel.rate_name,
        "frequency_series": channel.frequency_name,
        "count_series": channel.count_name,
        "measure": channel.measure,
        "measure_unit": "m^3" if channel.domain == "gas" else "m^2",
    }


def _matches_network(channel: EventChannel, config: NetworkConfig) -> bool:
    return (
        channel.domain in config.domains
        and (not config.zones or channel.zone in config.zones)
        and (
            not config.surfaces
            or (channel.domain != "gas" and channel.region in config.surfaces)
        )
    )


def _matches_species(
    channel: EventChannel,
    config: NetworkConfig,
    definitions: Mapping[tuple[str, str], ReactionData],
) -> bool:
    reaction = definitions[channel.domain, channel.reaction]
    return not config.species or bool(
        set(config.species) & (set(reaction.reactants) | set(reaction.products))
    )


def _reaction_definitions(data: ChemistryData) -> dict[tuple[str, str], ReactionData]:
    return {
        (domain, reaction.id): reaction
        for domain, reactions in (
            ("gas", data.gas_reactions),
            ("boundary", data.boundary_reactions),
            ("surface", data.surface_reactions),
        )
        for reaction in reactions
    }


def _network_channels(
    config: NetworkConfig,
    channels: tuple[EventChannel, ...],
    case: ResolvedCase,
    definitions: Mapping[tuple[str, str], ReactionData],
) -> tuple[EventChannel, ...]:
    zone_ids = {zone.zone_id for zone in case.reactor.zones}
    surface_ids = {surface.surface_id for surface in case.reactor.surfaces}
    if set(config.zones) - zone_ids or set(config.surfaces) - surface_ids:
        raise CaseValidationError("network references unknown zones or surfaces")
    chosen = tuple(
        channel
        for channel in channels
        if _matches_network(channel, config)
        and _matches_species(channel, config, definitions)
    )
    if not chosen:
        raise CaseValidationError(
            "network selection contains no modeled reaction events"
        )
    return chosen


def _series_units(
    names: Iterable[str], aliases: Mapping[str, str], declared: Mapping[str, str]
) -> dict[str, str]:
    names = tuple(names)
    if unknown := set(declared) - set(names) - set(aliases.values()):
        raise CaseValidationError(
            f"output.units references unsaved series: {sorted(unknown)}"
        )
    result: dict[str, str] = {}
    for name in names:
        source = aliases.get(name, name)
        unit = series_unit(source)
        if source.startswith("port_"):
            unit = PORT_UNITS[source.partition("[")[0].removeprefix("port_")]
        given = declared.get(name, declared.get(source, unit))
        if unit != "unspecified" and given != unit:
            raise CaseValidationError(
                f"series {name!r} requires canonical unit {unit!r}; "
                "no unit conversion is performed"
            )
        result[name] = given
    return result


def _validate_named_selections(case: ResolvedCase, available: tuple[str, ...]) -> None:
    for label, names in (
        ("observables", case.output.observables),
        ("summary series", case.output.summary_series),
    ):
        if unknown := set(names) - set(available):
            raise CaseValidationError(f"unknown output {label} {sorted(unknown)}")


def _validate_network_species(case: ResolvedCase, data: ChemistryData) -> None:
    species = {item.id for item in data.species}
    for config in case.output.networks.values():
        if unknown := set(config.species) - species:
            raise CaseValidationError(
                f"network references unknown species {sorted(unknown)}"
            )


def _validate_snapshot_times(case: ResolvedCase, model: CompiledGlobalModel) -> None:
    requested = [
        config.time_s
        for config in case.output.networks.values()
        if config.time_s is not None
    ]
    if not requested:
        return
    controls = SolverSettings(
        sample_interval_s=case.solver.sample_interval_s,
        save_at_s=None
        if case.solver.save_at_s is None
        else tuple(case.solver.save_at_s),
    )
    grid = _global_sample_times(model, controls)
    if grid is None:
        grid = np.asarray(
            [model.segments[0].start_s, *(segment.end_s for segment in model.segments)]
        )
    for time in requested:
        if not np.any(np.abs(grid - time) <= 8 * abs(np.spacing(time))):
            raise CaseValidationError(
                "network time_s must match a saved time; "
                "set solver.save_at_s or sample_interval_s"
            )


def _event_selection(
    case: ResolvedCase,
    channels: tuple[EventChannel, ...],
    definitions: Mapping[tuple[str, str], ReactionData],
) -> tuple[str, ...]:
    selected = list(channels) if case.output.reaction_rates else []
    for config in case.output.networks.values():
        chosen = _network_channels(config, channels, case, definitions)
        if config.mode == "events":
            selected.extend(chosen)
    return tuple(
        name
        for channel in selected
        for name in (channel.rate_name, channel.frequency_name, channel.count_name)
    )


def compile_output(
    case: ResolvedCase,
    model: CompiledGlobalModel,
    data: ChemistryData,
    initial: np.ndarray,
) -> OutputPlan:
    definitions = _reaction_definitions(data)
    _validate_network_species(case, data)
    _validate_snapshot_times(case, model)
    states = model.layout.labels
    port_names = _port_names(case, model, initial)
    available = tuple(
        dict.fromkeys((*states, *available_observables(model), *port_names))
    )
    aliases: dict[str, str] = {}
    for name, probe in case.output.probes.items():
        aliases.update(_probe_aliases(name, probe, model, set(available)))
    _validate_named_selections(case, (*available, *aliases))
    channels = event_channels(model)
    selected = [
        *case.output.observables,
        *_artifact_series(case, model, (*available, *aliases)),
        *_event_selection(case, channels, definitions),
        *case.output.summary_series,
    ]
    stored = expand_series(selected, (*available, *aliases))
    stored = tuple(
        name for name in stored if name not in states and name not in aliases
    )
    evaluated = tuple(
        dict.fromkeys(
            (*stored, *(name for name in aliases.values() if name not in states))
        )
    )
    metadata = _metadata(case, model, data, channels, stored, aliases, definitions)
    return OutputPlan(stored, evaluated, aliases, port_names, metadata)


def _validate_plot_range(plot: PlotConfig, case: ResolvedCase) -> None:
    if plot.time_range_s is not None:
        start, end = plot.time_range_s
        if (
            start < case.recipe.start_time_s
            or end > case.recipe.start_time_s + case.recipe.duration_s
        ):
            raise CaseValidationError("plot time_range_s is outside the run")


def _artifact_series(
    case: ResolvedCase, model: CompiledGlobalModel, available: tuple[str, ...]
) -> tuple[str, ...]:
    result: list[str] = []
    for plot in case.output.plots.values():
        _validate_plot_range(plot, case)
        names = (
            composition_names(plot, model)
            if plot.kind == "composition"
            else expand_series(plot.series, available)
        )
        result.extend(names)
    return tuple(result)


def _metadata(
    case: ResolvedCase,
    model: CompiledGlobalModel,
    data: ChemistryData,
    channels: tuple[EventChannel, ...],
    stored: tuple[str, ...],
    aliases: Mapping[str, str],
    definitions: Mapping[tuple[str, str], ReactionData],
) -> dict[str, Any]:
    names = (*model.layout.labels, *stored, *aliases)
    return {
        "output_requests": case.output.model_dump(mode="json"),
        "probes": {
            name: {
                "source_series": source,
                "spatial_interpretation": (
                    "model region average or modeled port; no point interpolation"
                ),
            }
            for name, source in aliases.items()
        },
        "series_units": _series_units(names, aliases, case.output.units),
        "model_geometry": {
            "zones": {
                zone.zone_id: {"volume_m3": zone.volume_m3} for zone in model.zones
            },
            "surfaces": {
                surface.surface_id: {
                    "zone": surface.zone_id,
                    "area_m2": surface.area_m2,
                }
                for surface in case.reactor.surfaces
            },
            "species": {
                item.id: {"phase": item.phase, "charge": item.charge}
                for item in data.species
            },
        },
        "reaction_network": [
            _reaction_metadata(channel, definitions) for channel in channels
        ],
        "composition_basis": (
            "all heavy gas species, including ions; electrons excluded; "
            "selected species are not renormalized"
        ),
        "event_integration": {
            "method": (
                "trapezoid on saved states within each forcing segment, "
                "with one-sided boundary rates"
            ),
            "interpretation": (
                "estimated continuum event count, not sampled stochastic events"
            ),
            "sampling_dependent": True,
        },
        "run_step_boundaries_s": list(
            np.cumsum(
                [
                    case.recipe.start_time_s,
                    *(step.duration_s for step in case.recipe.steps),
                ]
            )
        ),
        "observable_time_convention": (
            "left limit at forcing boundaries; cumulative event counts are continuous"
        ),
    }
